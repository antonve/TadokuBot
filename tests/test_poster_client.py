"""Tests for the poster/cover lookup (lib.poster_client).

The network is faked: a tiny stand-in for aiohttp's ``session.get``/``session.post``
async-context-manager protocol lets us assert the routing (tags -> service), the
title cleaning, the response parsing per service, and that everything degrades to
``None`` on a miss / missing key / error -- without touching the real APIs.
"""

from unittest.mock import AsyncMock

import pytest

import lib.poster_client as poster_client


# ---------------------------------------------------------------------------
# fake aiohttp session
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status=200, json_data=None, body=b""):
        self.status = status
        self._json = json_data if json_data is not None else {}
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._json

    async def read(self):
        return self._body


class _FakeSession:
    """Serves queued responses; records the (method, url, params/json) of each call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def _next(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self._responses.pop(0)

    def get(self, url, **kwargs):
        return self._next("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._next("POST", url, **kwargs)

    def head(self, url, **kwargs):
        return self._next("HEAD", url, **kwargs)


@pytest.fixture(autouse=True)
def _clear_keys(monkeypatch):
    # Default to no API keys so tests are explicit about enabling a source.
    monkeypatch.delenv("MAL_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_BOOKS_API_KEY", raising=False)
    monkeypatch.delenv("TMDB_API_KEY", raising=False)


def _anilist_candidate(primary_title, *, media_id=1, media_format="NOVEL", **extra):
    return {
        "id": media_id,
        "format": media_format,
        "title": {"native": primary_title},
        "coverImage": {"large": f"https://al/{media_id}.jpg"},
        **extra,
    }


def _anilist_response(*candidates):
    return _FakeResponse(json_data={"data": {"Page": {"media": list(candidates)}}})


def _manga_candidate(title, *, media_id=1, media_type="manga", **extra):
    return {"node": {
        "id": media_id,
        "title": title,
        "media_type": media_type,
        "main_picture": {"large": f"https://mal/{media_id}.jpg"},
        **extra,
    }}


def _book_candidate(title, *, book_id="book-1", categories=None, **extra):
    return {
        "id": book_id,
        "volumeInfo": {
            "title": title,
            "categories": ["Fiction"] if categories is None else categories,
            "imageLinks": {"thumbnail": f"http://books.google.com/{book_id}.jpg"},
            **extra,
        },
    }


# ---------------------------------------------------------------------------
# clean_title
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("呪術廻戦 Vol. 1", "呪術廻戦"),
    ("ワンピース vol. 002", "ワンピース"),
    ("ナルト vol. 14 (finished)", "ナルト"),
    ("MAO Vol. 02", "MAO"),
    ("転スラ７３〜８３", "転スラ７３〜８３"),
    ("薫る花は凛と咲く ep 1-3", "薫る花は凛と咲く"),
    ("極道さんはパパで愛妻家 ", "極道さんはパパで愛妻家"),
    ("黄泉のツガイ　1〜２", "黄泉のツガイ 1〜２"),
    ("Brothers Conflict 7", "Brothers Conflict 7"),
    ("Summer Pockets", "Summer Pockets"),
    ("Persona 5", "Persona 5"),
    ("Persona 3 Reload", "Persona 3 Reload"),
    ("86", "86"),
    ("Naruto episode 12", "Naruto"),
    ("薬屋のひとりごと 第12話", "薬屋のひとりごと"),
    ("薬屋のひとりごと ３巻", "薬屋のひとりごと"),
    ("Dune (completed)", "Dune"),
    ("Dune [finished]", "Dune"),
    ("Dune (Illustrated Edition)", "Dune (Illustrated Edition)"),
    ("Re:Zero (Light Novel) Vol. 2", "Re:Zero (Light Novel)"),
    ("Title (Subtitle", "Title (Subtitle"),
])
def test_clean_title_strips_volume_and_episode_noise(raw, expected):
    assert poster_client.clean_title(raw) == expected


def test_clean_title_keeps_original_when_cleaning_would_empty_it():
    # A bare volume marker shouldn't reduce to "" -- fall back to the trimmed input.
    assert poster_client.clean_title("Vol. 3") == "Vol. 3"
    assert poster_client.clean_title("   ") == ""


# ---------------------------------------------------------------------------
# category routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tags, expected", [
    (None, None),
    ([], None),
    (["book"], "book"),
    (["audiobook"], "book"),  # an audiobook is a book read aloud -> book lookup
    (["ln"], "ln"),
    (["ln", "book"], "ln"),  # ln wins -> AniList NOVEL search
    (["ln", "audiobook"], "ln"),  # ln still wins -> more precise novel search
    (["manga", "ln"], None),  # conflicting formats must abstain
    (["ln", "manga"], None),
    (["manga", "book"], "manga"),  # generic book tag does not conflict
    (["fiction", "game"], "game"),
    (["vn"], "vn"),
    (["vn", "game"], "vn"),  # vn wins so it stays VNDB-only, no Steam hop
    (["anime", "tv"], "anime"),  # anime beats tv -> MyAnimeList, not TMDB
    (["anime", "movie"], "anime"),
    (["comic", "manga"], "manga"),
    (["tv"], "screen"),
    (["movie"], "screen"),
    (["show"], "screen"),
    (["drama", "tv"], "screen"),  # live-action tv -> TMDB
    (["video"], None),  # generic "video" isn't routed to TMDB
    (["podcast"], None),
])
def test_category_routing(tags, expected):
    assert poster_client._category(tags) == expected


# ---------------------------------------------------------------------------
# fetch_poster dispatch + parsing
# ---------------------------------------------------------------------------

async def test_fetch_poster_returns_none_for_untagged_without_touching_network():
    session = _FakeSession([])  # no responses queued -> a network call would IndexError
    assert await poster_client.fetch_poster(session, [], "anything") is None
    assert session.calls == []


async def test_fetch_poster_returns_none_for_empty_title():
    session = _FakeSession([])
    assert await poster_client.fetch_poster(session, ["game"], "   ") is None
    assert session.calls == []


async def test_fetch_poster_abstains_for_conflicting_formats_without_network():
    session = _FakeSession([])
    assert await poster_client.fetch_poster(session, ["manga", "ln"], "Overlord") is None
    assert session.calls == []


async def test_vndb_game_lookup_parses_image_url_and_downloads():
    session = _FakeSession([
        _FakeResponse(json_data={"results": [{"image": {"url": "https://t.vndb.org/cv/x.jpg"}}]}),
        _FakeResponse(body=b"IMGBYTES"),
    ])
    out = await poster_client.fetch_poster(session, ["fiction", "game"], "Summer Pockets")
    assert out == b"IMGBYTES"
    # First call is the VNDB search POST with a cleaned search filter.
    method, url, kwargs = session.calls[0]
    assert method == "POST" and "api.vndb.org/kana/vn" in url
    assert kwargs["json"]["filters"] == ["search", "=", "Summer Pockets"]


async def test_vndb_returns_none_when_no_results():
    # ``vn`` is VNDB-only, so an empty result is a clean miss (no Steam fallback).
    session = _FakeSession([_FakeResponse(json_data={"results": []})])
    assert await poster_client.fetch_poster(session, ["vn"], "Nonexistent VN") is None
    assert len(session.calls) == 1  # search only; nothing else attempted


async def test_vndb_returns_none_when_image_is_null():
    session = _FakeSession([_FakeResponse(json_data={"results": [{"image": None}]})])
    assert await poster_client.fetch_poster(session, ["vn"], "Coverless VN") is None


async def test_game_falls_back_to_steam_when_vndb_misses():
    session = _FakeSession([
        _FakeResponse(json_data={"results": []}),                    # VNDB: miss
        _FakeResponse(json_data=[{"appid": "2161700", "name": "Persona 3 Reload"}]),  # Steam search
        _FakeResponse(status=200),                                   # HEAD library_600x900 exists
        _FakeResponse(body=b"STEAMCOVER"),                           # download
    ])
    out = await poster_client.fetch_poster(session, ["game"], "Persona 3 Reload")
    assert out == b"STEAMCOVER"
    # Order: VNDB POST, Steam search GET, HEAD portrait, GET download.
    assert [c[0] for c in session.calls] == ["POST", "GET", "HEAD", "GET"]
    assert "steamcommunity.com/actions/SearchApps" in session.calls[1][1]
    assert session.calls[2][1].endswith("/2161700/library_600x900.jpg")


async def test_steam_falls_back_to_header_when_no_portrait():
    session = _FakeSession([
        _FakeResponse(json_data={"results": []}),                    # VNDB miss
        _FakeResponse(json_data=[{"appid": "42", "name": "Old Game"}]),  # Steam search
        _FakeResponse(status=404),                                   # no portrait capsule
        _FakeResponse(status=200),                                   # header.jpg exists
        _FakeResponse(body=b"HEADER"),                               # download
    ])
    out = await poster_client.fetch_poster(session, ["game"], "Old Game")
    assert out == b"HEADER"
    assert session.calls[3][1].endswith("/42/header.jpg")


async def test_steam_returns_none_when_no_app_matches():
    session = _FakeSession([
        _FakeResponse(json_data={"results": []}),  # VNDB miss
        _FakeResponse(json_data=[]),               # Steam: no match either
    ])
    assert await poster_client.fetch_poster(session, ["game"], "Console Exclusive") is None


async def test_vn_tag_never_falls_back_to_steam():
    # A ``vn`` miss must not spill over to Steam (would IndexError on empty queue).
    session = _FakeSession([_FakeResponse(json_data={"results": []})])
    assert await poster_client.fetch_poster(session, ["vn"], "Obscure VN") is None
    assert len(session.calls) == 1


async def test_mal_anime_requires_client_id(monkeypatch):
    session = _FakeSession([])  # no key -> must not call out
    assert await poster_client.fetch_poster(session, ["anime"], "Frieren") is None
    assert session.calls == []


async def test_mal_manga_lookup_parses_main_picture(monkeypatch):
    monkeypatch.setenv("MAL_CLIENT_ID", "test-client-id")
    session = _FakeSession([
        _FakeResponse(json_data={"data": [
            _manga_candidate("呪術廻戦", main_picture={
                "medium": "https://m/med.jpg", "large": "https://m/lrg.jpg",
            })
        ]}),
        _FakeResponse(body=b"COVER"),
    ])
    out = await poster_client.fetch_poster(session, ["comic", "manga"], "呪術廻戦 Vol. 1")
    assert out == b"COVER"
    method, url, kwargs = session.calls[0]
    assert method == "GET" and url.endswith("/v2/manga")
    assert kwargs["headers"]["X-MAL-CLIENT-ID"] == "test-client-id"
    assert kwargs["params"]["q"] == "呪術廻戦"  # cleaned of "Vol. 1"
    assert kwargs["params"]["limit"] == 5
    assert {"main_picture", "media_type", "alternative_titles"} <= set(
        kwargs["params"]["fields"].split(",")
    )
    assert session.calls[1][1] == "https://m/lrg.jpg"


async def test_mal_prefers_large_but_falls_back_to_medium(monkeypatch):
    monkeypatch.setenv("MAL_CLIENT_ID", "cid")
    session = _FakeSession([
        _FakeResponse(json_data={"data": [{"node": {"main_picture": {"medium": "https://m/med.jpg"}}}]}),
        _FakeResponse(body=b"OK"),
    ])
    assert await poster_client.fetch_poster(session, ["anime"], "X") == b"OK"
    # The download targets the medium URL (no large available).
    assert session.calls[1][1] == "https://m/med.jpg"


async def test_book_anilist_hit_parses_cover_and_downloads():
    # A ``book`` tag tries AniList first; a hit means Google Books is never touched.
    session = _FakeSession([
        _anilist_response(_anilist_candidate("薬屋のひとりごと", coverImage={
            "medium": "https://al/m.jpg", "large": "https://al/l.jpg",
            "extraLarge": "https://al/xl.jpg",
        })),
        _FakeResponse(body=b"ANILISTCOVER"),
    ])
    out = await poster_client.fetch_poster(session, ["book"], "薬屋のひとりごと Vol. 1")
    assert out == b"ANILISTCOVER"
    method, url, kwargs = session.calls[0]
    assert method == "POST" and url == "https://graphql.anilist.co"
    assert kwargs["json"]["variables"]["search"] == "薬屋のひとりごと"  # cleaned of "Vol. 1"
    assert "format: NOVEL" in kwargs["json"]["query"]
    assert "Page" in kwargs["json"]["query"]
    assert "perPage:5" in kwargs["json"]["query"].replace(" ", "")
    assert session.calls[1][1] == "https://al/xl.jpg"  # prefers extraLarge


async def test_audiobook_uses_the_same_book_lookup():
    # An ``audiobook`` tag flows exactly like ``book``: AniList novels,
    # then download; Google Books is only a fallback (untouched on a hit).
    session = _FakeSession([
        _anilist_response(_anilist_candidate("薬屋のひとりごと")),
        _FakeResponse(body=b"AUDIOBOOKCOVER"),
    ])
    out = await poster_client.fetch_poster(session, ["audiobook"], "薬屋のひとりごと Vol. 1")
    assert out == b"AUDIOBOOKCOVER"
    method, url, kwargs = session.calls[0]
    assert method == "POST" and url == "https://graphql.anilist.co"
    assert kwargs["json"]["variables"]["search"] == "薬屋のひとりごと"  # cleaned of "Vol. 1"
    assert "format: NOVEL" in kwargs["json"]["query"]


async def test_ln_anilist_search_narrows_to_novel_format():
    session = _FakeSession([
        _anilist_response(_anilist_candidate("オーバーロード")),
        _FakeResponse(body=b"LNCOVER"),
    ])
    out = await poster_client.fetch_poster(session, ["ln"], "オーバーロード")
    assert out == b"LNCOVER"
    assert "format: NOVEL" in session.calls[0][2]["json"]["query"]


async def test_book_falls_back_to_google_when_anilist_misses(monkeypatch):
    monkeypatch.setenv("GOOGLE_BOOKS_API_KEY", "gkey")
    session = _FakeSession([
        _anilist_response(),  # AniList miss
        _FakeResponse(json_data={"items": [
            _book_candidate("雪国", book_id="c", imageLinks={
                "thumbnail": "http://books.google.com/c.jpg",
            })
        ]}),
        _FakeResponse(body=b"BOOKCOVER"),
    ])
    out = await poster_client.fetch_poster(session, ["book"], "雪国")
    assert out == b"BOOKCOVER"
    assert session.calls[0][1] == "https://graphql.anilist.co"  # AniList tried first
    assert session.calls[1][1].startswith("https://www.googleapis.com/books")
    assert session.calls[1][2]["params"]["maxResults"] == 5
    assert session.calls[1][2]["params"]["printType"] == "books"
    assert session.calls[2][1] == "https://books.google.com/c.jpg"  # http -> https


async def test_book_returns_none_when_anilist_misses_and_no_google_key(monkeypatch):
    # AniList is keyless so it's always tried; Google Books is skipped without a key.
    session = _FakeSession([_anilist_response()])
    assert await poster_client.fetch_poster(session, ["book"], "雪国") is None
    assert len(session.calls) == 1
    assert session.calls[0][1] == "https://graphql.anilist.co"


@pytest.mark.parametrize("media_type", ["manga", "one_shot", "doujinshi", "manhwa", "manhua", "oel"])
async def test_manga_accepts_comic_formats(monkeypatch, media_type):
    monkeypatch.setenv("MAL_CLIENT_ID", "cid")
    session = _FakeSession([
        _FakeResponse(json_data={"data": [_manga_candidate("Overlord", media_type=media_type)]}),
        _FakeResponse(body=b"MANGA"),
    ])
    assert await poster_client.fetch_poster(session, ["manga", "book"], "Overlord") == b"MANGA"


@pytest.mark.parametrize("media_type", ["novel", "light_novel", "unknown", None])
async def test_manga_rejects_novel_and_unknown_formats(monkeypatch, media_type):
    monkeypatch.setenv("MAL_CLIENT_ID", "cid")
    session = _FakeSession([
        _FakeResponse(json_data={"data": [_manga_candidate("Overlord", media_type=media_type)]}),
    ])
    assert await poster_client.fetch_poster(session, ["manga"], "Overlord") is None
    assert len(session.calls) == 1


async def test_manga_skips_wrong_format_and_wrong_title_to_choose_later_match(monkeypatch):
    monkeypatch.setenv("MAL_CLIENT_ID", "cid")
    session = _FakeSession([
        _FakeResponse(json_data={"data": [
            _manga_candidate("Overlord", media_id=1, media_type="novel"),
            _manga_candidate("Overlord: The Undead King Oh!", media_id=2),
            _manga_candidate("Overlord", media_id=3),
        ]}),
        _FakeResponse(body=b"CORRECT"),
    ])
    assert await poster_client.fetch_poster(session, ["manga"], "Overlord") == b"CORRECT"
    assert session.calls[-1][1] == "https://mal/3.jpg"


@pytest.mark.parametrize("aliases", [
    {"en": "Fullmetal Alchemist"},
    {"ja": "Fullmetal Alchemist"},
    {"synonyms": ["FMA", "Fullmetal Alchemist"]},
])
async def test_manga_matches_provider_aliases(monkeypatch, aliases):
    monkeypatch.setenv("MAL_CLIENT_ID", "cid")
    session = _FakeSession([
        _FakeResponse(json_data={"data": [
            _manga_candidate("鋼の錬金術師", alternative_titles=aliases),
        ]}),
        _FakeResponse(body=b"ALIAS"),
    ])
    assert await poster_client.fetch_poster(session, ["manga"], "Fullmetal Alchemist") == b"ALIAS"


async def test_manga_abstains_when_two_distinct_works_share_a_title(monkeypatch):
    monkeypatch.setenv("MAL_CLIENT_ID", "cid")
    session = _FakeSession([
        _FakeResponse(json_data={"data": [
            _manga_candidate("Monster", media_id=1),
            _manga_candidate("Monster", media_id=2),
        ]}),
    ])
    assert await poster_client.fetch_poster(session, ["manga"], "Monster") is None
    assert len(session.calls) == 1


async def test_manga_skips_malformed_results_before_valid_match(monkeypatch):
    monkeypatch.setenv("MAL_CLIENT_ID", "cid")
    session = _FakeSession([
        _FakeResponse(json_data={"data": [
            None, {"node": None}, _manga_candidate(None, main_picture=None),
            _manga_candidate("Monster", media_id=2, alternative_titles=None),
        ]}),
        _FakeResponse(body=b"VALID"),
    ])
    assert await poster_client.fetch_poster(session, ["manga"], "Monster") == b"VALID"
    assert session.calls[-1][1] == "https://mal/2.jpg"


@pytest.mark.parametrize("tag", ["book", "ln", "audiobook"])
@pytest.mark.parametrize("media_format", ["MANGA", "ONE_SHOT", None])
async def test_book_routes_reject_non_novel_anilist_results(tag, media_format):
    session = _FakeSession([
        _anilist_response(_anilist_candidate("Overlord", media_format=media_format)),
    ])
    assert await poster_client.fetch_poster(session, [tag], "Overlord") is None
    assert len(session.calls) == 1


async def test_anilist_checks_later_candidates_and_preserves_sequel_number():
    session = _FakeSession([
        _anilist_response(
            _anilist_candidate("Persona", media_id=1),
            _anilist_candidate("Persona 4", media_id=2),
            _anilist_candidate("Persona 5", media_id=3, media_format="MANGA"),
            _anilist_candidate("Persona 5", media_id=4),
        ),
        _FakeResponse(body=b"CORRECT"),
    ])
    assert await poster_client.fetch_poster(session, ["book"], "Persona 5") == b"CORRECT"
    assert session.calls[0][2]["json"]["variables"]["search"] == "Persona 5"
    assert session.calls[-1][1] == "https://al/4.jpg"


@pytest.mark.parametrize("extra", [
    {"title": {"romaji": "The Apothecary Diaries"}},
    {"title": {"english": "The Apothecary Diaries"}},
    {"title": {"native": "The Apothecary Diaries"}},
    {"synonyms": [None, "Kusuriya", "The Apothecary Diaries"]},
])
async def test_anilist_matches_title_aliases(extra):
    session = _FakeSession([
        _anilist_response(_anilist_candidate("薬屋のひとりごと", **extra)),
        _FakeResponse(body=b"ALIAS"),
    ])
    assert await poster_client.fetch_poster(session, ["ln"], "The Apothecary Diaries") == b"ALIAS"


async def test_anilist_normalizes_unicode_case_spacing_and_full_width_punctuation():
    session = _FakeSession([
        _anilist_response(_anilist_candidate("persona: 5")),
        _FakeResponse(body=b"NORMALIZED"),
    ])
    assert await poster_client.fetch_poster(session, ["book"], "ＰＥＲＳＯＮＡ：　５") == b"NORMALIZED"


async def test_anilist_preserves_meaningful_parenthetical_title():
    session = _FakeSession([
        _anilist_response(_anilist_candidate("Dune")),
    ])
    assert await poster_client.fetch_poster(session, ["book"], "Dune (Illustrated Edition)") is None
    assert len(session.calls) == 1


@pytest.mark.parametrize("second_id, expected", [(1, b"SAME-WORK"), (2, None)])
async def test_anilist_distinguishes_duplicate_records_from_ambiguous_works(second_id, expected):
    session = _FakeSession([
        _anilist_response(
            _anilist_candidate("Overlord", media_id=1),
            _anilist_candidate("Overlord", media_id=second_id),
        ),
        *([_FakeResponse(body=b"SAME-WORK")] if expected else []),
    ])
    assert await poster_client.fetch_poster(session, ["ln"], "Overlord") == expected
    assert len(session.calls) == (2 if expected else 1)


async def test_anilist_skips_malformed_candidates_before_valid_match():
    session = _FakeSession([
        _anilist_response(
            None,
            _anilist_candidate(None, media_id=1),
            _anilist_candidate("Other Work", media_id=2, coverImage=None),
            _anilist_candidate("Overlord", media_id=3),
        ),
        _FakeResponse(body=b"VALID"),
    ])
    assert await poster_client.fetch_poster(session, ["ln"], "Overlord") == b"VALID"
    assert session.calls[-1][1] == "https://al/3.jpg"


async def test_anilist_considers_at_most_five_candidates():
    session = _FakeSession([
        _anilist_response(
            *[_anilist_candidate(f"Other work {i}", media_id=i) for i in range(5)],
            _anilist_candidate("Overlord", media_id=6),
        ),
    ])
    assert await poster_client.fetch_poster(session, ["book"], "Overlord") is None
    assert len(session.calls) == 1


async def test_anilist_coverless_same_title_work_still_makes_match_ambiguous():
    session = _FakeSession([
        _anilist_response(
            _anilist_candidate("Overlord", media_id=1, coverImage=None),
            _anilist_candidate("Overlord", media_id=2),
        ),
    ])
    assert await poster_client.fetch_poster(session, ["ln"], "Overlord") is None
    assert len(session.calls) == 1


async def test_anilist_ambiguity_does_not_fall_through_to_google(monkeypatch):
    monkeypatch.setenv("GOOGLE_BOOKS_API_KEY", "gkey")
    session = _FakeSession([
        _anilist_response(
            _anilist_candidate("Overlord", media_id=1),
            _anilist_candidate("Overlord", media_id=2),
        ),
    ])
    assert await poster_client.fetch_poster(session, ["ln"], "Overlord") is None
    assert len(session.calls) == 1


async def test_anilist_keeps_title_symbols_distinct():
    session = _FakeSession([_anilist_response(_anilist_candidate("C"))])
    assert await poster_client.fetch_poster(session, ["book"], "C++") is None
    assert len(session.calls) == 1


async def test_anilist_validates_complete_title_when_search_query_is_truncated():
    title = "A very long light novel title with more than sixty-four characters in its name 2"
    session = _FakeSession([
        _anilist_response(_anilist_candidate(title[:64])),
    ])
    assert await poster_client.fetch_poster(session, ["ln"], title) is None
    assert len(session.calls[0][2]["json"]["variables"]["search"]) <= 64
    assert len(session.calls) == 1


@pytest.mark.parametrize("tag", ["book", "ln", "audiobook"])
async def test_google_fallback_rejects_comic_then_selects_correct_book(monkeypatch, tag):
    monkeypatch.setenv("GOOGLE_BOOKS_API_KEY", "gkey")
    session = _FakeSession([
        _anilist_response(_anilist_candidate("Overlord", media_format="MANGA")),
        _FakeResponse(json_data={"items": [
            _book_candidate("Overlord", book_id="comic", categories=["Comics & Graphic Novels / Manga"]),
            _book_candidate("Overlord: Side Story", book_id="other"),
            _book_candidate("Overlord", book_id="novel", subtitle="Light Novel"),
        ]}),
        _FakeResponse(body=b"BOOK"),
    ])
    assert await poster_client.fetch_poster(session, [tag], "Overlord") == b"BOOK"
    assert session.calls[-1][1] == "https://books.google.com/novel.jpg"


@pytest.mark.parametrize("categories", [
    [], ["Unknown"], ["Comics & Graphic Novels"], ["Manga"],
    ["Fiction", "Comics & Graphic Novels / Manga"],
])
async def test_google_fallback_requires_positive_non_comic_evidence(monkeypatch, categories):
    monkeypatch.setenv("GOOGLE_BOOKS_API_KEY", "gkey")
    session = _FakeSession([
        _anilist_response(),
        _FakeResponse(json_data={"items": [
            _book_candidate("Overlord", categories=categories, printType="BOOK"),
        ]}),
    ])
    assert await poster_client.fetch_poster(session, ["ln"], "Overlord") is None
    assert len(session.calls) == 2


@pytest.mark.parametrize("metadata", [
    {"categories": ["Science"], "printType": "MAGAZINE"},
    {"categories": ["Fiction"], "mainCategory": "Comics & Graphic Novels"},
])
async def test_google_fallback_rejects_conflicting_format_metadata(monkeypatch, metadata):
    monkeypatch.setenv("GOOGLE_BOOKS_API_KEY", "gkey")
    session = _FakeSession([
        _anilist_response(),
        _FakeResponse(json_data={"items": [_book_candidate("Dune", **metadata)]}),
    ])
    assert await poster_client.fetch_poster(session, ["book"], "Dune") is None
    assert len(session.calls) == 2


@pytest.mark.parametrize("tag, subtitle", [
    ("ln", "Ｌｉｇｈｔ Ｎｏｖｅｌ"),
    ("book", ""),
])
async def test_google_fallback_normalizes_full_width_format_and_subject_labels(monkeypatch, tag, subtitle):
    monkeypatch.setenv("GOOGLE_BOOKS_API_KEY", "gkey")
    session = _FakeSession([
        _anilist_response(),
        _FakeResponse(json_data={"items": [
            _book_candidate("Overlord", subtitle=subtitle, categories=["Ｆｉｃｔｉｏｎ"]),
        ]}),
        _FakeResponse(body=b"NORMALIZED"),
    ])
    assert await poster_client.fetch_poster(session, [tag], "Overlord") == b"NORMALIZED"


async def test_google_fallback_rejects_wrong_title_even_with_valid_book_metadata(monkeypatch):
    monkeypatch.setenv("GOOGLE_BOOKS_API_KEY", "gkey")
    session = _FakeSession([
        _anilist_response(),
        _FakeResponse(json_data={"items": [_book_candidate("Dune Messiah")]}),
    ])
    assert await poster_client.fetch_poster(session, ["book"], "Dune") is None
    assert len(session.calls) == 2


async def test_ln_google_fallback_requires_novel_evidence_beyond_generic_fiction(monkeypatch):
    monkeypatch.setenv("GOOGLE_BOOKS_API_KEY", "gkey")
    session = _FakeSession([
        _anilist_response(),
        _FakeResponse(json_data={"items": [_book_candidate("Overlord")]}),
    ])
    assert await poster_client.fetch_poster(session, ["ln"], "Overlord") is None
    assert len(session.calls) == 2


async def test_google_fallback_preserves_substantive_subtitle_identity(monkeypatch):
    monkeypatch.setenv("GOOGLE_BOOKS_API_KEY", "gkey")
    session = _FakeSession([
        _anilist_response(),
        _FakeResponse(json_data={"items": [
            _book_candidate("Dune", subtitle="The Battle of Corrin"),
        ]}),
    ])
    assert await poster_client.fetch_poster(session, ["book"], "Dune") is None
    assert len(session.calls) == 2


async def test_google_fallback_accepts_matching_complete_title_with_subtitle(monkeypatch):
    monkeypatch.setenv("GOOGLE_BOOKS_API_KEY", "gkey")
    session = _FakeSession([
        _anilist_response(),
        _FakeResponse(json_data={"items": [
            _book_candidate("Dune", subtitle="The Battle of Corrin"),
        ]}),
        _FakeResponse(body=b"COMPLETE-TITLE"),
    ])
    assert await poster_client.fetch_poster(session, ["book"], "Dune: The Battle of Corrin") == b"COMPLETE-TITLE"


async def test_google_fallback_vetoes_comic_subtitle_despite_fiction_category(monkeypatch):
    monkeypatch.setenv("GOOGLE_BOOKS_API_KEY", "gkey")
    session = _FakeSession([
        _anilist_response(),
        _FakeResponse(json_data={"items": [
            _book_candidate("Dune", subtitle="The Graphic Novel"),
        ]}),
    ])
    assert await poster_client.fetch_poster(session, ["book"], "Dune: The Graphic Novel") is None
    assert len(session.calls) == 2


async def test_google_fallback_abstains_for_same_title_with_competing_ids(monkeypatch):
    monkeypatch.setenv("GOOGLE_BOOKS_API_KEY", "gkey")
    session = _FakeSession([
        _anilist_response(),
        _FakeResponse(json_data={"items": [
            _book_candidate("Dune", book_id="one"),
            _book_candidate("Dune", book_id="two"),
        ]}),
    ])
    assert await poster_client.fetch_poster(session, ["book"], "Dune") is None
    assert len(session.calls) == 2


async def test_google_fallback_skips_malformed_results_before_valid_match(monkeypatch):
    monkeypatch.setenv("GOOGLE_BOOKS_API_KEY", "gkey")
    session = _FakeSession([
        _anilist_response(),
        _FakeResponse(json_data={"items": [
            None, {"id": "broken", "volumeInfo": None},
            _book_candidate("Dune", book_id="valid"),
        ]}),
        _FakeResponse(body=b"VALID"),
    ])
    assert await poster_client.fetch_poster(session, ["book"], "Dune") == b"VALID"
    assert session.calls[-1][1] == "https://books.google.com/valid.jpg"


async def test_rejected_book_match_is_cached():
    session = _FakeSession([
        _anilist_response(_anilist_candidate("Overlord", media_format="MANGA")),
    ])
    cache = {}
    assert await poster_client.fetch_poster(session, ["book"], "Overlord", cache) is None
    assert await poster_client.fetch_poster(session, ["book"], "Overlord", cache) is None
    assert len(session.calls) == 1


async def test_candidate_diagnostics_identify_selected_and_rejected_results(caplog):
    caplog.set_level("INFO", logger="lib.poster_client")
    session = _FakeSession([
        _anilist_response(
            _anilist_candidate("Overlord", media_id=1, media_format="MANGA"),
            _anilist_candidate("Different Work", media_id=2),
            _anilist_candidate("Overlord", media_id=3),
        ),
        _FakeResponse(body=b"SELECTED"),
    ])
    assert await poster_client.fetch_poster(session, ["ln"], "Overlord", log_id=42) == b"SELECTED"
    messages = [record.getMessage() for record in caplog.records]
    assert "Poster log_id=42 provider=anilist candidate_id=1 title='Overlord' format='MANGA' outcome=incompatible_format" in messages
    assert "Poster log_id=42 provider=anilist candidate_id=2 title='Different Work' format='NOVEL' outcome=title_mismatch" in messages
    assert "Poster log_id=42 provider=anilist candidate_id=3 title='Overlord' format='NOVEL' outcome=compatible" in messages
    assert "Poster log_id=42 provider=anilist outcome=selected candidate_id=3" in messages


@pytest.mark.parametrize("phase", ["search", "download"])
async def test_transport_errors_omit_urls_and_api_keys_from_diagnostics(monkeypatch, caplog, phase):
    monkeypatch.setenv("GOOGLE_BOOKS_API_KEY", "FAKE_SECRET_KEY")
    caplog.set_level("INFO", logger="lib.poster_client")
    search = _FakeResponse(json_data={"items": [_book_candidate("Dune")]})
    download = _FakeResponse(body=b"UNREACHED")
    failure = poster_client.aiohttp.ClientConnectionError(
        "GET https://www.googleapis.com/books/v1/volumes?key=FAKE_SECRET_KEY failed"
    )
    if phase == "search":
        search.json = AsyncMock(side_effect=failure)
    else:
        download.read = AsyncMock(side_effect=failure)
    session = _FakeSession([_anilist_response(), search, download])

    assert await poster_client.fetch_poster(session, ["book"], "Dune", log_id=43) is None
    assert "log_id=43" in caplog.text
    assert "ClientConnectionError" in caplog.text
    assert "FAKE_SECRET_KEY" not in caplog.text
    assert "https://www.googleapis.com" not in caplog.text
    assert len(session.calls) == (2 if phase == "search" else 3)


async def test_anilist_timeout_still_uses_validated_google_fallback(monkeypatch):
    monkeypatch.setenv("GOOGLE_BOOKS_API_KEY", "gkey")
    timed_out = _FakeResponse()
    timed_out.json = AsyncMock(side_effect=TimeoutError("provider timed out"))
    session = _FakeSession([
        timed_out,
        _FakeResponse(json_data={"items": [
            _book_candidate("Dune Messiah", book_id="wrong"),
            _book_candidate("Dune", book_id="correct"),
        ]}),
        _FakeResponse(body=b"CORRECT"),
    ])
    assert await poster_client.fetch_poster(session, ["book"], "Dune") == b"CORRECT"
    assert session.calls[-1][1] == "https://books.google.com/correct.jpg"


async def test_tmdb_requires_key(monkeypatch):
    session = _FakeSession([])  # no key -> must not call out
    assert await poster_client.fetch_poster(session, ["tv"], "silent tokyo") is None
    assert session.calls == []


async def test_tmdb_lookup_parses_poster_and_cleans_title(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "tkey")
    session = _FakeSession([
        _FakeResponse(json_data={"results": [
            {"media_type": "person", "name": "Some Actor"},          # skipped
            {"media_type": "tv", "poster_path": None},                # skipped (no poster)
            {"media_type": "movie", "poster_path": "/abc123.jpg"},    # taken
        ]}),
        _FakeResponse(body=b"TMDBPOSTER"),
    ])
    out = await poster_client.fetch_poster(session, ["movie"], "silent tokyo 1")
    assert out == b"TMDBPOSTER"
    method, url, kwargs = session.calls[0]
    assert method == "GET" and "api.themoviedb.org/3/search/multi" in url
    assert kwargs["params"]["api_key"] == "tkey"
    assert kwargs["params"]["query"] == "silent tokyo 1"  # preserve ambiguous bare numbers
    # The download targets the built image URL at w500.
    assert session.calls[1][1] == "https://image.tmdb.org/t/p/w500/abc123.jpg"


async def test_tmdb_passes_japanese_title_through(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "tkey")
    session = _FakeSession([
        _FakeResponse(json_data={"results": [{"media_type": "tv", "poster_path": "/jp.jpg"}]}),
        _FakeResponse(body=b"JP"),
    ])
    out = await poster_client.fetch_poster(session, ["tv"], "半沢直樹")
    assert out == b"JP"
    assert session.calls[0][2]["params"]["query"] == "半沢直樹"


async def test_tmdb_returns_none_when_no_screen_result(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "tkey")
    session = _FakeSession([
        _FakeResponse(json_data={"results": [{"media_type": "person", "name": "X"}]}),
    ])
    assert await poster_client.fetch_poster(session, ["show"], "Nobody") is None
    assert len(session.calls) == 1  # search only; no download attempted


async def test_non_200_search_yields_none(monkeypatch):
    # ``vn`` -> VNDB only, so a 500 is a clean miss with no Steam fallback attempt.
    session = _FakeSession([_FakeResponse(status=500, json_data={})])
    assert await poster_client.fetch_poster(session, ["vn"], "Whatever") is None


async def test_non_200_download_yields_none():
    session = _FakeSession([
        _FakeResponse(json_data={"results": [{"image": {"url": "https://x/y.jpg"}}]}),
        _FakeResponse(status=404),
    ])
    assert await poster_client.fetch_poster(session, ["game"], "Summer Pockets") is None


async def test_fetch_poster_memoises_via_cache():
    session = _FakeSession([
        _FakeResponse(json_data={"results": [{"image": {"url": "https://x/y.jpg"}}]}),
        _FakeResponse(body=b"IMG"),
    ])
    cache: dict = {}
    a = await poster_client.fetch_poster(session, ["game"], "Summer Pockets", cache)
    # Second call for the same material must hit the cache, issuing no new requests.
    b = await poster_client.fetch_poster(session, ["game"], "Summer Pockets", cache)
    assert a == b == b"IMG"
    assert len(session.calls) == 2  # one search + one download, not four
