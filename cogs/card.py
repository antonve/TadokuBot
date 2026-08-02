"""Card cog: the ``/card``, ``/weeklycard`` and ``/monthlycard`` commands.

Each posts a rendered **stats card** (see ``lib.profile_card``) to the current
channel for everyone to see: the person's avatar, their place and score, and
their totals (characters, pages, comic pages, listening hours). It's the log-feed
profile card without the log callout or the material poster.

The three differ only in the window they cover:

  * ``/card``         -- the whole current contest (cumulative place + score).
  * ``/weeklycard``   -- points and logs in the last 7 days.
  * ``/monthlycard``  -- points and logs in a calendar month of the current year.

Each takes an optional ``username``: omit it for your own card (you must have
linked a tadoku.app username via ``/claim``), or pass someone's Tadoku display
name to show theirs. The avatar is drawn when the person is claimed by a Discord
member in this server; otherwise the card shows a placeholder disc.

The heavy lifting is shared: ``leaderboard._find_leaderboard_entry`` resolves the
person (and their user id), ``log_feed.compute_window_totals`` sums a window, and
``leaderboard._tally_scores_since`` ranks a period. ``_post_card`` holds the
common flow; each command supplies a ``compute`` callback for its window.
"""

import io
import logging
from datetime import datetime, timedelta, timezone

import discord
from discord import app_commands
from discord.ext import commands

import cogs.leaderboard as leaderboard
import cogs.log_feed as log_feed
import lib.config_store as config_store
import lib.profile_card as profile_card
import lib.tadoku_client as tadoku

_log = logging.getLogger(__name__)

_UNREACHABLE = "❌ Couldn't reach tadoku.app right now. Try again in a moment."
WEEKLY_WINDOW_DAYS = 7


class Card(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # -- commands -----------------------------------------------------------

    @app_commands.command(
        name="card",
        description="Post a stats card for yourself (or someone else) in this contest.",
    )
    @app_commands.describe(
        username="Whose card to show (their Tadoku display name). Omit for your own.",
    )
    @app_commands.guild_only()
    async def card(self, interaction: discord.Interaction, username: str | None = None):
        """Contest-wide stats card: cumulative place, score and totals."""
        async def compute(contest, entry):
            totals = await log_feed.compute_contest_totals(
                self.bot.session, entry["user_id"], contest
            )
            return log_feed.format_standing(entry, contest), totals

        await self._post_card(interaction, username, compute)

    @app_commands.command(
        name="weeklycard",
        description="Post a stats card for the last 7 days of this contest.",
    )
    @app_commands.describe(
        username="Whose card to show (their Tadoku display name). Omit for your own.",
    )
    @app_commands.guild_only()
    async def weeklycard(self, interaction: discord.Interaction, username: str | None = None):
        """Stats card scoped to the last 7 days (a rolling window ending now)."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=WEEKLY_WINDOW_DAYS)

        async def compute(contest, entry):
            return await self._period_result(
                contest, entry, cutoff=cutoff, until=None,
                label=f"last {WEEKLY_WINDOW_DAYS} days",
                phrase=f"the last {WEEKLY_WINDOW_DAYS} days",
            )

        await self._post_card(interaction, username, compute)

    @app_commands.command(
        name="monthlycard",
        description="Post a stats card for a calendar month of this contest (current year).",
    )
    @app_commands.describe(
        month="Which month to show (defaults to the current month).",
        username="Whose card to show (their Tadoku display name). Omit for your own.",
    )
    @app_commands.choices(month=leaderboard.MONTH_CHOICES)
    @app_commands.guild_only()
    async def monthlycard(
        self,
        interaction: discord.Interaction,
        month: app_commands.Choice[int] | None = None,
        username: str | None = None,
    ):
        """Stats card scoped to a calendar month of the current year."""
        now = datetime.now(timezone.utc)
        target_month = month.value if month else now.month
        # Window is [start of the chosen month, start of the next month), current
        # year; for the current month `until` is in the future so it excludes
        # nothing -- matching "so far this month".
        cutoff = datetime(now.year, target_month, 1, tzinfo=timezone.utc)
        if target_month == 12:
            until = datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            until = datetime(now.year, target_month + 1, 1, tzinfo=timezone.utc)
        label = cutoff.strftime("%B %Y")  # e.g. "July 2026"

        async def compute(contest, entry):
            return await self._period_result(
                contest, entry, cutoff=cutoff, until=until, label=label, phrase=label
            )

        await self._post_card(interaction, username, compute)

    # -- shared core --------------------------------------------------------

    async def _post_card(self, interaction, username, compute):
        """Resolve the person, run ``compute`` for the window, and post the card.

        ``compute(contest, entry)`` returns either ``(subtitle, totals)`` to render
        or a ``str`` message to send instead (e.g. "logged nothing this week").
        """
        claims = config_store.get_guild_claims(interaction.guild_id)

        if username is None:
            # No name given: show the caller's own card -- they must have claimed one.
            target = claims.get(str(interaction.user.id))
            if not target:
                await interaction.response.send_message(
                    "You haven't linked a tadoku.app account yet — use `/claim <username>` "
                    "first, or pass a username to show someone else's card.",
                    ephemeral=True,
                )
                return
        else:
            target = username.strip()

        # Several network calls; defer publicly (the card is for the whole channel).
        await interaction.response.defer()

        try:
            contest = await leaderboard._resolve_contest(self.bot, interaction.guild_id)
            entry = await leaderboard._find_leaderboard_entry(self.bot, contest["id"], target)
            result = None
            if entry is not None:
                result = await compute(contest, entry)
        except tadoku.TadokuAPIError:
            await interaction.followup.send(_UNREACHABLE)
            return

        if entry is None:
            await interaction.followup.send(
                f"**{target}** isn't on the leaderboard for **{contest['title']}**."
            )
            return
        if isinstance(result, str):
            # A window-specific "nothing to show" message (e.g. no logs this week).
            await interaction.followup.send(result)
            return

        subtitle, totals = result
        display_name = entry["user_display_name"]
        claimer = log_feed._claimer_id(claims, display_name)
        avatar_bytes = await self._avatar_bytes(claimer) if claimer is not None else None

        try:
            png = await profile_card.render_card(
                display_name=display_name,
                subtitle=subtitle,
                avatar_bytes=avatar_bytes,
                characters=totals["characters"],
                pages=totals["pages"],
                comic_pages=totals["comic_pages"],
                listening_hours=totals["minutes"] / 60,
                # No this_log/title/poster -> the log-less stats card.
            )
        except Exception:  # noqa: BLE001 -- a render failure shouldn't 500 the command
            _log.exception("/card render failed for %r", display_name)
            await interaction.followup.send("❌ Couldn't render the card right now.")
            return

        await interaction.followup.send(file=discord.File(io.BytesIO(png), filename="card.png"))

    async def _period_result(self, contest, entry, *, cutoff, until, label, phrase):
        """Build ``(subtitle, totals)`` for a period card, or a "nothing" message.

        Ranks the person within the window ``[cutoff, until)`` from the tally the
        weekly/monthly leaderboards use, and sums their logs over the same window
        for the stat tiles. Returns a message string when they logged nothing in
        the window (no rank, and empty tiles would be misleading). ``label`` is the
        subtitle suffix ("last 7 days" / "July 2026"); ``phrase`` is the prose form
        for the "nothing" message ("the last 7 days" / "July 2026").
        """
        uid = entry["user_id"]
        totals_by_user = await leaderboard._tally_scores_since(
            self.bot, contest["id"], cutoff, until=until
        )
        if uid not in totals_by_user:
            return f"**{entry['user_display_name']}** logged nothing in {phrase}."

        score = totals_by_user[uid][1]
        # Standard competition rank within the window (ties share a rank).
        rank = 1 + sum(1 for _, s in totals_by_user.values() if s > score)
        is_tie = sum(1 for _, s in totals_by_user.values() if s == score) > 1
        tie = " (tie)" if is_tie else ""
        subtitle = f"#{rank}{tie} · {log_feed._format_points(score)} pts · {label}"

        totals = await log_feed.compute_window_totals(self.bot.session, uid, cutoff, until)
        return subtitle, totals

    async def _avatar_bytes(self, uid: int) -> bytes | None:
        """Return Discord user ``uid``'s avatar bytes, or ``None`` on any failure."""
        user = self.bot.get_user(uid)
        if user is None:
            try:
                user = await self.bot.fetch_user(uid)
            except discord.HTTPException:
                return None
        try:
            return await user.display_avatar.read()
        except discord.HTTPException:
            return None


async def setup(bot: commands.Bot) -> None:
    """discord.py extension entry point; called by ``load_extension``."""
    await bot.add_cog(Card(bot))
