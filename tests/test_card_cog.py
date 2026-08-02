"""Tests for the /card cog (the log-less stats card posted to a channel).

The tadoku client, the contest resolver and the (Pillow) renderer are mocked, so
the cog callback is driven directly with a fake interaction -- no live Discord, no
real image work. The card image itself is covered in test_profile_card.py.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import pytest
from discord import app_commands

import cogs.card as card
import cogs.leaderboard as leaderboard
import cogs.log_feed as log_feed
import lib.config_store as config_store
import lib.profile_card as profile_card
import lib.tadoku_client as tadoku_client
from tests.conftest import make_interaction

CONTEST = {"id": "c1", "title": "2026 Round 4",
           "contest_start": "2026-07-01", "contest_end": "2026-07-31"}


@pytest.fixture(autouse=True)
def patched(monkeypatch):
    monkeypatch.setattr(tadoku_client, "get_latest_official_contest", AsyncMock(return_value=CONTEST))
    monkeypatch.setattr(tadoku_client, "get_contest", AsyncMock(return_value=CONTEST))
    # One-entry leaderboard by default; individual tests override the entry.
    monkeypatch.setattr(
        tadoku_client, "get_contest_leaderboard",
        AsyncMock(return_value={"entries": [_entry(3, "ruby", 177.2)], "total_size": 1}),
    )
    # No immersion history by default -> zeroed stat tiles.
    monkeypatch.setattr(
        tadoku_client, "list_user_logs", AsyncMock(return_value={"logs": [], "total_size": 0})
    )
    monkeypatch.setattr(profile_card, "render_card", AsyncMock(return_value=b"PNGDATA"))


def _entry(rank, name, score, is_tie=False, user_id="u1"):
    return {"rank": rank, "user_id": user_id, "user_display_name": name,
            "score": score, "is_tie": is_tie}


def _bot(avatar_for=None):
    """A fake bot; ``avatar_for`` is a {discord_id: bytes} map for get_user avatars."""
    bot = SimpleNamespace(session=AsyncMock())
    avatar_for = avatar_for or {}

    def _get_user(uid):
        if uid in avatar_for:
            return SimpleNamespace(
                display_avatar=SimpleNamespace(read=AsyncMock(return_value=avatar_for[uid]))
            )
        return None

    bot.get_user = _get_user
    bot.fetch_user = AsyncMock(side_effect=discord.HTTPException(
        SimpleNamespace(status=404, reason="nf"), "no user"))
    return bot


def _kwargs():
    return profile_card.render_card.await_args.kwargs


# ---------------------------------------------------------------------------
# /card username
# ---------------------------------------------------------------------------

async def test_card_posts_stats_card_for_a_named_user():
    tadoku_client.get_contest_leaderboard.return_value = {
        "entries": [_entry(3, "ruby", 177.2, user_id="uuid-ruby")], "total_size": 1,
    }
    tadoku_client.list_user_logs.return_value = {
        "logs": [{"unit_name": "Character", "amount": 6_600_000, "deleted": False,
                  "created_at": "2026-07-05T20:00:00Z"}],
        "total_size": 1,
    }
    cog = card.Card(_bot())
    interaction = make_interaction(guild_id=999)

    await cog.card.callback(cog, interaction, username="ruby")

    # Deferred publicly, then posted an image file (no ephemeral).
    interaction.response.defer.assert_awaited_once()
    sent = interaction.followup.send.await_args
    assert isinstance(sent.kwargs["file"], discord.File)
    # Renderer got the leaderboard's spelling, the standing subtitle, and stats --
    # and, being a stats card, NO log line, title or poster.
    kw = _kwargs()
    assert kw["display_name"] == "ruby"
    assert kw["subtitle"] == "#3 · 177.2 pts in 2026 Round 4"
    assert kw["characters"] == 6_600_000
    assert kw.get("this_log", "") == "" and kw.get("title", "") == ""
    assert kw.get("poster_bytes") is None
    # Stats summed the entry's user id, not the typed name.
    assert tadoku_client.list_user_logs.await_args.args[1] == "uuid-ruby"


async def test_card_uses_leaderboard_spelling_and_marks_ties():
    tadoku_client.get_contest_leaderboard.return_value = {
        "entries": [_entry(1, "Ruby ", 100.0, is_tie=True)], "total_size": 1,
    }
    cog = card.Card(_bot())
    interaction = make_interaction(guild_id=999)

    await cog.card.callback(cog, interaction, username="ruby")

    assert _kwargs()["display_name"] == "Ruby "
    assert _kwargs()["subtitle"] == "#1 (tie) · 100 pts in 2026 Round 4"


async def test_card_draws_claimer_avatar_when_the_person_is_claimed():
    config_store.set_claim(999, 222, "ruby")  # ruby is discord user 222
    cog = card.Card(_bot(avatar_for={222: b"AVATARBYTES"}))
    interaction = make_interaction(guild_id=999, user_id=999_999)  # a different caller

    await cog.card.callback(cog, interaction, username="ruby")

    assert _kwargs()["avatar_bytes"] == b"AVATARBYTES"


async def test_card_uses_placeholder_avatar_for_an_unclaimed_person():
    cog = card.Card(_bot())  # nobody claimed "ruby"
    interaction = make_interaction(guild_id=999)

    await cog.card.callback(cog, interaction, username="ruby")

    assert _kwargs()["avatar_bytes"] is None


async def test_card_reports_when_person_not_on_leaderboard():
    tadoku_client.get_contest_leaderboard.return_value = {"entries": [], "total_size": 0}
    cog = card.Card(_bot())
    interaction = make_interaction(guild_id=999)

    await cog.card.callback(cog, interaction, username="ghost")

    profile_card.render_card.assert_not_awaited()
    args, kwargs = interaction.followup.send.await_args
    assert "file" not in kwargs
    assert "ghost" in args[0] and "2026 Round 4" in args[0]


async def test_card_sends_friendly_message_on_api_error():
    tadoku_client.get_contest_leaderboard.side_effect = tadoku_client.TadokuAPIError("boom")
    cog = card.Card(_bot())
    interaction = make_interaction(guild_id=999)

    await cog.card.callback(cog, interaction, username="ruby")

    args, kwargs = interaction.followup.send.await_args
    assert "file" not in kwargs
    assert "tadoku.app" in args[0]


# ---------------------------------------------------------------------------
# /card (no argument -> the caller's own card)
# ---------------------------------------------------------------------------

async def test_card_no_arg_uses_the_callers_claimed_username():
    config_store.set_claim(999, 111, "ruby")  # caller (user 111) claimed "ruby"
    cog = card.Card(_bot(avatar_for={111: b"MYAVATAR"}))
    interaction = make_interaction(guild_id=999, user_id=111)

    await cog.card.callback(cog, interaction, username=None)

    interaction.response.defer.assert_awaited_once()
    assert isinstance(interaction.followup.send.await_args.kwargs["file"], discord.File)
    assert _kwargs()["display_name"] == "ruby"
    assert _kwargs()["avatar_bytes"] == b"MYAVATAR"


async def test_card_no_arg_without_a_claim_is_ephemeral_and_hits_no_network():
    cog = card.Card(_bot())
    interaction = make_interaction(guild_id=999, user_id=111)  # unclaimed caller

    await cog.card.callback(cog, interaction, username=None)

    interaction.response.defer.assert_not_awaited()
    args, kwargs = interaction.response.send_message.await_args
    assert kwargs.get("ephemeral") is True
    assert "/claim" in args[0]
    tadoku_client.get_contest_leaderboard.assert_not_awaited()
    profile_card.render_card.assert_not_awaited()


# ---------------------------------------------------------------------------
# /weeklycard + /monthlycard (period-scoped cards)
# ---------------------------------------------------------------------------

async def test_weeklycard_shows_period_rank_and_window_stats(monkeypatch):
    tadoku_client.get_contest_leaderboard.return_value = {
        "entries": [_entry(9, "ruby", 500.0, user_id="uuid-ruby")], "total_size": 1,
    }
    # This week: ruby has 88 pts, someone else 100 -> ruby is #2.
    monkeypatch.setattr(leaderboard, "_tally_scores_since", AsyncMock(return_value={
        "uuid-ruby": ["ruby", 88.0], "other": ["max", 100.0],
    }))
    monkeypatch.setattr(log_feed, "compute_window_totals", AsyncMock(return_value={
        "characters": 12000, "pages": 3, "comic_pages": 0, "minutes": 45,
    }))
    cog = card.Card(_bot())
    interaction = make_interaction(guild_id=999)

    await cog.weeklycard.callback(cog, interaction, username="ruby")

    kw = _kwargs()
    assert kw["subtitle"] == "#2 · 88 pts · last 7 days"
    assert kw["characters"] == 12000 and kw["listening_hours"] == 45 / 60
    assert isinstance(interaction.followup.send.await_args.kwargs["file"], discord.File)


async def test_weeklycard_reports_when_nothing_logged_this_week(monkeypatch):
    tadoku_client.get_contest_leaderboard.return_value = {
        "entries": [_entry(9, "ruby", 500.0, user_id="uuid-ruby")], "total_size": 1,
    }
    # ruby isn't in this week's tally at all.
    monkeypatch.setattr(leaderboard, "_tally_scores_since",
                        AsyncMock(return_value={"other": ["max", 100.0]}))
    cog = card.Card(_bot())
    interaction = make_interaction(guild_id=999)

    await cog.weeklycard.callback(cog, interaction, username="ruby")

    profile_card.render_card.assert_not_awaited()
    args, kwargs = interaction.followup.send.await_args
    assert "file" not in kwargs
    assert "logged nothing" in args[0] and "last 7 days" in args[0]


async def test_monthlycard_scopes_to_the_selected_month_of_the_current_year(monkeypatch):
    tadoku_client.get_contest_leaderboard.return_value = {
        "entries": [_entry(1, "ruby", 500.0, user_id="uuid-ruby")], "total_size": 1,
    }
    tally = AsyncMock(return_value={"uuid-ruby": ["ruby", 140.0]})
    monkeypatch.setattr(leaderboard, "_tally_scores_since", tally)
    monkeypatch.setattr(log_feed, "compute_window_totals", AsyncMock(return_value={
        "characters": 5000, "pages": 0, "comic_pages": 0, "minutes": 0,
    }))
    cog = card.Card(_bot())
    interaction = make_interaction(guild_id=999)
    year = datetime.now(timezone.utc).year

    await cog.monthlycard.callback(
        cog, interaction, month=app_commands.Choice(name="July", value=7), username="ruby"
    )

    # Tally was scoped to July of the current year (cutoff positional, until kwarg).
    cutoff = tally.await_args.args[2]
    until = tally.await_args.kwargs["until"]
    assert cutoff == datetime(year, 7, 1, tzinfo=timezone.utc)
    assert until == datetime(year, 8, 1, tzinfo=timezone.utc)
    assert _kwargs()["subtitle"] == f"#1 · 140 pts · July {year}"


async def test_monthlycard_defaults_to_the_current_month(monkeypatch):
    tadoku_client.get_contest_leaderboard.return_value = {
        "entries": [_entry(1, "ruby", 500.0, user_id="uuid-ruby")], "total_size": 1,
    }
    tally = AsyncMock(return_value={"uuid-ruby": ["ruby", 10.0]})
    monkeypatch.setattr(leaderboard, "_tally_scores_since", tally)
    monkeypatch.setattr(log_feed, "compute_window_totals", AsyncMock(return_value={
        "characters": 0, "pages": 0, "comic_pages": 0, "minutes": 0,
    }))
    cog = card.Card(_bot())
    interaction = make_interaction(guild_id=999)
    now = datetime.now(timezone.utc)

    await cog.monthlycard.callback(cog, interaction, month=None, username="ruby")

    assert tally.await_args.args[2] == datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    assert now.strftime("%B %Y") in _kwargs()["subtitle"]
