"""Fetch a cover/poster image for a log's material, chosen by the log's tags.

A tadoku log carries a ``tags`` list (e.g. ``["manga"]``, ``["fiction", "game"]``)
and a ``description`` that is the material's title -- often Japanese, and often
carrying volume/episode notes like ``"呪術廻戦 Vol. 1"``.
This module maps a log to a poster image, by tag:

  * vn          -> VNDB (``api.vndb.org/kana``), no key needed
  * game        -> VNDB, then Steam's storefront as a fallback (both keyless);
                   Steam covers general games (Persona, Ys, …) that aren't VNs
  * anime       -> MyAnimeList API v2 (needs ``MAL_CLIENT_ID``)
  * manga       -> MyAnimeList API v2 (needs ``MAL_CLIENT_ID``)
  * ln          -> AniList GraphQL (light novels, keyless), then Google Books
  * book        -> AniList novels (keyless), then Google Books (needs
                   ``GOOGLE_BOOKS_API_KEY`` for quota). AniList carries Japanese
                   light-novel covers Google Books often lacks, so it's tried first
  * audiobook   -> same lookup as ``book`` (a book read aloud): AniList, then
                   Google Books
  * tv/movie/show (live-action) -> TMDB (needs ``TMDB_API_KEY``); anime is routed
                   to MyAnimeList above, so only non-anime screen media lands here

Every lookup is strictly best-effort: a miss, a missing API key, or any
network/parse failure yields ``None`` so the log feed simply falls back to the
poster-less card. Manga/book lookups require a unique compatible title or alias
among up to five candidates. Explicit progress suffixes are removed; possible
title identity such as bare numbers, ranges and parentheticals is preserved.
Google Books format checks rely on subject metadata and can miss sparse or
misclassified entries; ``ln`` additionally requires explicit novel evidence.

Nothing here authenticates as a user or writes anywhere -- it's read-only
lookups against public search endpoints, keyed only by the material title.
"""

import asyncio
import logging
import os
import re
import unicodedata
from typing import NamedTuple, Optional
from urllib.parse import quote

import aiohttp

_log = logging.getLogger(__name__)

# Cap every request so a slow upstream can't stall the log-feed poll. Posters are
# a nice-to-have, so we fail fast rather than hold the card.
_TIMEOUT = aiohttp.ClientTimeout(total=8)

# Cap provider queries (MAL rejects overly long ``q`` values), while retaining
# the complete title for candidate validation.
_MAX_QUERY = 64
_MAX_CANDIDATES = 5
_COMIC_FORMATS = {"manga", "one_shot", "doujinshi", "manhwa", "manhua", "oel"}


class _Candidate(NamedTuple):
    id: object
    titles: list
    format: object
    image_url: Optional[str]


class _AmbiguousMatch(ValueError):
    """Do not resolve competing works by guessing again at another provider."""


def _object(value) -> dict:
    return value if isinstance(value, dict) else {}


def _items(value) -> list:
    return value if isinstance(value, list) else []


def _cover_url(picture, *sizes: str) -> Optional[str]:
    for size in sizes:
        url = _object(picture).get(size)
        if isinstance(url, str) and url.startswith(("https://", "http://")):
            return url
    return None


def _title_key(title: str) -> str:
    # Keep punctuation and digits: C/C++ and numbered sequels are different works.
    return clean_title(unicodedata.normalize("NFKC", title)).casefold()


def _select_cover(provider: str, title: str, candidates: list[_Candidate],
                  formats: set[str], log_id=None) -> Optional[str]:
    """Select a unique exact title/alias match, including coverless rivals."""
    wanted = _title_key(title)
    matches: dict[str, _Candidate] = {}
    # ponytail: five exact-match candidates; expand or add fuzzy scoring only
    # after labeled misses demonstrate a worthwhile improvement in coverage.
    for candidate in candidates[:_MAX_CANDIDATES]:
        titles = [t for t in candidate.titles if isinstance(t, str) and t.strip()]
        if type(candidate.id) not in (str, int) or not candidate.id:
            reason = "missing_id"
        elif not isinstance(candidate.format, str) or candidate.format not in formats:
            reason = "incompatible_format"
        elif not wanted or not any(_title_key(t) == wanted for t in titles):
            reason = "title_mismatch"
        else:
            reason = "compatible"
            key = str(candidate.id)
            if key not in matches or not matches[key].image_url:
                matches[key] = candidate
        _log.info("Poster log_id=%r provider=%s candidate_id=%r title=%r format=%r outcome=%s",
                  log_id, provider, candidate.id, titles[0] if titles else None,
                  candidate.format, reason)
    if len(matches) > 1:
        _log.info("Poster log_id=%r provider=%s outcome=ambiguous candidate_ids=%r",
                  log_id, provider, list(matches))
        raise _AmbiguousMatch(provider)
    candidate = next(iter(matches.values()), None)
    outcome = "no_match" if candidate is None else "selected" if candidate.image_url else "no_cover"
    _log.info("Poster log_id=%r provider=%s outcome=%s candidate_id=%r",
              log_id, provider, outcome,
              candidate.id if candidate else None)
    return candidate.image_url if candidate else None


async def _search_json(session, method: str, url: str, provider: str, log_id, **kwargs) -> Optional[dict]:
    """Read search metadata; diagnostics omit URLs/exception text containing keys."""
    try:
        async with getattr(session, method)(url, timeout=_TIMEOUT, **kwargs) as resp:
            if resp.status != 200:
                reason = "rate_limited" if resp.status == 429 else "provider_error"
                _log.info("Poster log_id=%r provider=%s outcome=%s status=%s",
                          log_id, provider, reason, resp.status)
                return None
            data = await resp.json()
        if isinstance(data, dict):
            if data.get("errors") or data.get("error"):
                _log.info("Poster log_id=%r provider=%s outcome=provider_error", log_id, provider)
                return None
            return data
        _log.info("Poster log_id=%r provider=%s outcome=malformed_response", log_id, provider)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        _log.info("Poster log_id=%r provider=%s outcome=provider_error error_type=%s",
                  log_id, provider, type(exc).__name__)
    return None


# Which service a set of tags routes to. A log can carry several tags
# (``["fiction", "game"]``); the first match here wins, most-specific first.
def _category(tags: Optional[list]) -> Optional[str]:
    """Map a log's tags to a poster source key, or ``None`` if none apply."""
    if not tags:
        return None
    have = {str(t).strip().casefold() for t in tags}
    if {"manga", "ln"} <= have:
        return None
    # ``vn`` is VNDB-only; ``game`` tries VNDB then Steam. Check ``vn`` first so a
    # log tagged both keeps the (better) VNDB art without a needless Steam hop.
    if "vn" in have:
        return "vn"
    if "game" in have:
        return "game"
    if "anime" in have:
        return "anime"
    if "manga" in have:
        return "manga"
    # Keep the more specific tag for the stricter novel fallback requirements.
    if "ln" in have:
        return "ln"
    # An audiobook is a book read aloud -- route it through the same cover lookup
    # as ``book`` (AniList, then Google Books). ``ln`` is checked first, so an
    # audiobook of a light novel still gets the more precise novel search.
    if have & {"book", "audiobook"}:
        return "book"
    # Live-action screen media: TV shows and films. ``anime`` is caught above, so
    # a log tagged both (e.g. an anime that aired on TV) stays on MyAnimeList.
    if have & {"tv", "movie", "show"}:
        return "screen"
    return None


def clean_title(description: str) -> str:
    """Remove explicit trailing progress notes; preserve possible title identity.

    Bare numbers/ranges and meaningful parentheticals can identify a work, so
    retain them. If cleanup would empty the title, keep the original instead.
    """
    t = (description or "").strip()
    if not t:
        return ""
    t = re.sub(r"(?i)\s*[（(【\[]\s*(?:finished|completed|done)\s*[）)】\]]?\s*$", "", t)
    t = re.sub(
        r"(?i)\s*(?:[-–—:：]\s*)?(?<![a-z])"
        r"(?:vol\.?|volume|ep\.?|episode|chapter|ch\.?|#)\s*\d+"
        r"(?:\s*[〜~\-–ー]\s*\d+)?\s*$", "", t,
    )
    t = re.sub(r"\s*(?:[-–—:：]\s*)?第?\s*\d+\s*(?:話|巻|章|集|冊)\s*$", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or (description or "").strip()


async def fetch_poster(
    session: aiohttp.ClientSession,
    tags: Optional[list],
    description: str,
    cache: Optional[dict] = None,
    *,
    log_id=None,
) -> Optional[bytes]:
    """Return poster image bytes for a log, or ``None`` if none can be found.

    Routes on the log's ``tags`` (see ``_category``), cleans ``description`` into
    a search query, looks up an image URL from the matching service, and
    downloads it. Any failure at any step -- unknown category, empty title,
    missing key or network error -- collapses to ``None`` so the caller can fall
    back to the poster-less card. The renderer handles image decoding failures.

    ``cache`` (if given) memoises results by (category, title) so a burst of the
    same material costs a single lookup + download.
    """
    category = _category(tags)
    if category is None:
        _log.info("Poster log_id=%r outcome=unsupported_or_conflicting_tags tags=%r", log_id, tags)
        return None
    title = clean_title(description)
    if not title:
        _log.info("Poster log_id=%r outcome=empty_title", log_id)
        return None
    key = (category, title.lower())
    if cache is not None and key in cache:
        _log.info("Poster log_id=%r category=%s outcome=cache_hit has_cover=%s",
                  log_id, category, cache[key] is not None)
        return cache[key]

    result: Optional[bytes] = None
    try:
        # Cap only provider queries; never shorten the identity used to compare results.
        url = await _image_url(session, category, title, log_id=log_id)
        if url:
            result = await _download(session, url)
            if result is None:
                _log.info("Poster log_id=%r category=%s outcome=download_failed", log_id, category)
    except _AmbiguousMatch:
        result = None
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        _log.warning("Poster log_id=%r category=%s outcome=lookup_failed error_type=%s",
                     log_id, category, type(exc).__name__)
        result = None

    if cache is not None:
        cache[key] = result
    return result


async def _image_url(
    session: aiohttp.ClientSession, category: str, title: str, *, log_id=None
) -> Optional[str]:
    """Dispatch to the right service and return a poster image URL, or ``None``."""
    if category == "vn":
        return await _vndb_image_url(session, title[:_MAX_QUERY])
    if category == "game":
        # VNs have the nicest art on VNDB; fall back to Steam for everything else.
        return await _vndb_image_url(session, title[:_MAX_QUERY]) or await _steam_image_url(session, title[:_MAX_QUERY])
    if category in ("anime", "manga"):
        return await _mal_image_url(session, category, title, log_id=log_id)
    if category in ("ln", "book"):
        url = await _anilist_image_url(session, title, log_id=log_id)
        if url:
            return url
        _log.info("Poster log_id=%r category=%s outcome=fallback provider=google_books", log_id, category)
        return await _google_books_image_url(session, title, novel_only=category == "ln", log_id=log_id)
    if category == "screen":
        return await _tmdb_image_url(session, title[:_MAX_QUERY])
    return None


async def _mal_image_url(
    session: aiohttp.ClientSession, media: str, title: str, *, log_id=None
) -> Optional[str]:
    """Look up an anime/manga cover on MyAnimeList v2 (``main_picture``).

    Needs ``MAL_CLIENT_ID`` in the environment; without it we skip (return
    ``None``) rather than error.
    """
    client_id = os.environ.get("MAL_CLIENT_ID")
    if not client_id:
        _log.info("Poster log_id=%r provider=mal outcome=missing_key", log_id)
        return None
    url = f"https://api.myanimelist.net/v2/{media}"
    params = {"q": title[:_MAX_QUERY], "limit": 1, "fields": "main_picture"}
    if media == "manga":
        params.update(limit=_MAX_CANDIDATES, fields="main_picture,media_type,alternative_titles")
    headers = {"X-MAL-CLIENT-ID": client_id}
    data = await _search_json(session, "get", url, "mal", log_id, params=params, headers=headers)
    if data is None:
        return None
    nodes = _items(data.get("data"))
    if not nodes:
        _log.info("Poster log_id=%r provider=mal outcome=no_match", log_id)
        return None
    if media == "anime":
        return _cover_url(_object(_object(nodes[0]).get("node")).get("main_picture"), "large", "medium")
    candidates = []
    for result in nodes[:_MAX_CANDIDATES]:
        node = _object(_object(result).get("node"))
        aliases = _object(node.get("alternative_titles"))
        candidates.append(_Candidate(
            node.get("id"),
            [node.get("title"), aliases.get("en"), aliases.get("ja")] + _items(aliases.get("synonyms")),
            node.get("media_type"), _cover_url(node.get("main_picture"), "large", "medium"),
        ))
    return _select_cover("mal", title, candidates, _COMIC_FORMATS, log_id)


async def _tmdb_image_url(
    session: aiohttp.ClientSession, title: str
) -> Optional[str]:
    """Look up a live-action TV/movie poster on TMDB (``poster_path``).

    Uses TMDB's *multi* search, which matches a title across both films and TV in
    one call and matches Japanese titles well (TMDB indexes each work's original
    and alternative-language titles). Needs ``TMDB_API_KEY`` (a v3 API key);
    without it we skip (return ``None``) rather than error.
    """
    key = os.environ.get("TMDB_API_KEY")
    if not key:
        return None
    params = {
        "api_key": key,
        "query": title,
        "include_adult": "false",
        # Prefer Japanese-language metadata/artwork where a work has it.
        "language": "ja-JP",
    }
    async with session.get(
        "https://api.themoviedb.org/3/search/multi", params=params, timeout=_TIMEOUT
    ) as resp:
        if resp.status != 200:
            return None
        data = await resp.json()
    for result in data.get("results") or []:
        # Skip ``person`` hits; take the first film/show that actually has a poster.
        if result.get("media_type") in ("movie", "tv") and result.get("poster_path"):
            return f"https://image.tmdb.org/t/p/w500{result['poster_path']}"
    return None


async def _vndb_image_url(
    session: aiohttp.ClientSession, title: str
) -> Optional[str]:
    """Look up a visual-novel cover on VNDB's Kana API (no key needed)."""
    body = {"filters": ["search", "=", title], "fields": "image.url", "results": 1}
    async with session.post(
        "https://api.vndb.org/kana/vn", json=body, timeout=_TIMEOUT
    ) as resp:
        if resp.status != 200:
            return None
        data = await resp.json()
    results = data.get("results") or []
    if not results:
        return None
    image = results[0].get("image") or {}
    return image.get("url")


async def _steam_image_url(
    session: aiohttp.ClientSession, title: str
) -> Optional[str]:
    """Look up a game cover on Steam's storefront (no key needed).

    Searches the keyless app-search endpoint (which matches English *and*
    Japanese titles), then prefers the 2:3 portrait "library" capsule, falling
    back to the landscape header for the rare title without portrait art. Steam
    is PC-only, so console-exclusive titles simply won't match (-> ``None``).
    """
    search_url = "https://steamcommunity.com/actions/SearchApps/" + quote(title, safe="")
    async with session.get(search_url, timeout=_TIMEOUT) as resp:
        if resp.status != 200:
            return None
        results = await resp.json()
    if not results:
        return None
    appid = results[0].get("appid")
    if not appid:
        return None
    base = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{appid}"
    for filename in ("library_600x900.jpg", "header.jpg"):
        candidate = f"{base}/{filename}"
        # A cheap HEAD confirms the image exists before we commit to it.
        async with session.head(candidate, timeout=_TIMEOUT) as resp:
            if resp.status == 200:
                return candidate
    return None


async def _anilist_image_url(
    session: aiohttp.ClientSession, title: str, *, log_id=None
) -> Optional[str]:
    """Search novels only, then validate format and full title (keyless)."""
    query = (
        f"query ($search: String) {{ Page(perPage: {_MAX_CANDIDATES}) {{ "
        "media(search: $search, type: MANGA, format: NOVEL) { "
        "id format title { romaji english native } synonyms "
        "coverImage { extraLarge large medium } } } }"
    )
    data = await _search_json(
        session, "post", "https://graphql.anilist.co", "anilist", log_id,
        json={"query": query, "variables": {"search": title[:_MAX_QUERY]}},
    )
    if data is None:
        return None
    page = _object(_object(data.get("data")).get("Page"))
    candidates = []
    for result in _items(page.get("media"))[:_MAX_CANDIDATES]:
        media = _object(result)
        candidates.append(_Candidate(
            media.get("id"), list(_object(media.get("title")).values()) + _items(media.get("synonyms")),
            media.get("format"), _cover_url(media.get("coverImage"), "extraLarge", "large", "medium"),
        ))
    return _select_cover("anilist", title, candidates, {"NOVEL"}, log_id)


# Google Books exposes subjects, not a prose/comics format enum. Require known
# subjects and veto comic signals; missing or unrecognized metadata cannot match.
_BOOK_SUBJECTS = {
    "fiction", "juvenile fiction", "young adult fiction", "juvenile nonfiction",
    "young adult nonfiction", "biography & autobiography", "history", "science",
    "computers", "business & economics", "education", "travel", "cooking",
    "self-help", "philosophy", "religion", "psychology", "social science",
    "reference", "language arts & disciplines", "foreign language study",
    "literary collections", "literary criticism", "poetry", "mathematics",
    "medical", "nature", "technology & engineering",
}
_NOVEL_LABEL = re.compile(r"(?:a )?(?:light[ -])?novels?|小説|ライトノベル|長編小説|短編小説", re.I)
_COMIC_LABEL = re.compile(r"\b(?:comics?|manga|manhwa|manhua|graphic[ -]+novels?)\b|漫画|マンガ|コミック", re.I)


def _google_candidate(item) -> _Candidate:
    item = _object(item)
    info = _object(item.get("volumeInfo"))
    title = info.get("title")
    subtitle = info.get("subtitle") or ""
    main_category = info.get("mainCategory") or ""
    categories = info.get("categories")
    invalid = _Candidate(item.get("id"), [title], None, None)
    if not all(isinstance(value, str) for value in (title, subtitle, main_category)):
        return invalid
    if info.get("printType") not in (None, "BOOK"):
        return invalid
    if not isinstance(categories, list) or not categories or not all(isinstance(c, str) for c in categories):
        return invalid
    categories = [unicodedata.normalize("NFKC", c) for c in categories]
    subtitle_label = unicodedata.normalize("NFKC", subtitle).strip()
    evidence = unicodedata.normalize("NFKC", " ".join([title, subtitle, main_category, *categories]))
    if _COMIC_LABEL.search(evidence):
        return invalid
    labels = [part.strip() for c in categories for part in c.split("/")] + [subtitle_label]
    # Only explicit format labels qualify as novel evidence, not titles such as
    # "The History of the Novel". Keep substantive subtitles in the match key.
    novel = any(_NOVEL_LABEL.fullmatch(label) for label in labels)
    novel = novel or any(_NOVEL_LABEL.fullmatch(label.strip()) for label in re.findall(r"[（(【\[]([^）)】\]]+)[）)】\]]", evidence))
    subjects = {c.split("/", 1)[0].strip().casefold() for c in categories}
    # ponytail: subject-based evidence can be wrong when the catalogue is
    # misclassified; use an explicit media ID if real reports expose that limit.
    media_format = "NOVEL" if novel else "BOOK" if subjects & _BOOK_SUBJECTS else None
    titles = [title]
    if subtitle:
        titles = [f"{title}: {subtitle}", f"{title} {subtitle}"]
        if _NOVEL_LABEL.fullmatch(subtitle_label):
            titles.append(title)
    url = _cover_url(info.get("imageLinks"), "thumbnail", "smallThumbnail")
    return _Candidate(item.get("id"), titles, media_format, url.replace("http://", "https://", 1) if url else None)


async def _google_books_image_url(
    session: aiohttp.ClientSession, title: str, *, novel_only: bool = False, log_id=None
) -> Optional[str]:
    """Look up a book cover on Google Books (``imageLinks.thumbnail``).

    Needs ``GOOGLE_BOOKS_API_KEY`` for a usable quota; the keyless endpoint is
    shared and rate-limited, so without a key we skip.
    """
    key = os.environ.get("GOOGLE_BOOKS_API_KEY")
    if not key:
        _log.info("Poster log_id=%r provider=google_books outcome=missing_key", log_id)
        return None
    query_title = title[:_MAX_QUERY].replace('"', ' ')
    params = {
        "q": f'intitle:"{query_title}"', "maxResults": _MAX_CANDIDATES,
        "printType": "books", "country": "US", "key": key,
    }
    data = await _search_json(
        session, "get", "https://www.googleapis.com/books/v1/volumes", "google_books", log_id, params=params,
    )
    if data is None:
        return None
    candidates = [_google_candidate(item) for item in _items(data.get("items"))[:_MAX_CANDIDATES]]
    return _select_cover("google_books", title, candidates, {"NOVEL"} if novel_only else {"NOVEL", "BOOK"}, log_id)


async def _download(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
    """Download an image URL to bytes, or ``None`` on a non-200 / transport error."""
    async with session.get(url, timeout=_TIMEOUT) as resp:
        if resp.status != 200:
            return None
        return await resp.read()
