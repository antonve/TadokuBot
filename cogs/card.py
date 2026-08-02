"""Card cog: the ``/card`` command.

``/card`` posts a rendered **stats card** (see ``lib.profile_card``) to the
current channel for everyone to see: the person's avatar, their place and score
in the server's current contest, and their contest totals (characters, pages,
comic pages, listening hours). It's the log-feed profile card without the log
callout or the material poster.

``/card`` with no argument shows the caller's own card (they must have linked a
tadoku.app username via ``/claim``); ``/card username`` shows someone else's by
their tadoku display name. The avatar is drawn when the person is claimed by a
Discord member in this server; otherwise the card shows a placeholder disc.

The heavy lifting is shared with the log feed: ``log_feed.compute_contest_totals``
sums the window, ``log_feed.format_standing`` builds the subtitle, and
``leaderboard._find_leaderboard_entry`` resolves the person (and their user id).
"""

import io
import logging

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


class Card(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(
        name="card",
        description="Post a stats card for yourself (or someone else) in this contest.",
    )
    @app_commands.describe(
        username="Whose card to show (their Tadoku display name). Omit for your own.",
    )
    @app_commands.guild_only()
    async def card(self, interaction: discord.Interaction, username: str | None = None):
        """Render and post a person's contest stats card to the channel."""
        claims = config_store.get_guild_claims(interaction.guild_id)

        if username is None:
            # No name given: show the caller's own card -- they must have claimed one.
            target = claims.get(str(interaction.user.id))
            if not target:
                await interaction.response.send_message(
                    "You haven't linked a tadoku.app account yet — use `/claim <username>` "
                    "first, or `/card <username>` to show someone else's card.",
                    ephemeral=True,
                )
                return
        else:
            target = username.strip()

        # Scanning the leaderboard + summing logs is several calls; defer publicly
        # (the card is meant for the whole channel, like /leaderboard).
        await interaction.response.defer()

        try:
            contest = await leaderboard._resolve_contest(self.bot, interaction.guild_id)
            entry = await leaderboard._find_leaderboard_entry(self.bot, contest["id"], target)
            totals = None
            if entry is not None:
                totals = await log_feed.compute_contest_totals(
                    self.bot.session, entry["user_id"], contest
                )
        except tadoku.TadokuAPIError:
            await interaction.followup.send(_UNREACHABLE)
            return

        if entry is None:
            # Not on this contest's leaderboard -> nothing to card (and no user id
            # to sum stats from), so say so rather than invent a zeroed card.
            await interaction.followup.send(
                f"**{target}** isn't on the leaderboard for **{contest['title']}**."
            )
            return

        # Use the leaderboard's own spelling of the name (case/spacing may differ
        # from what was typed), and draw the claimer's avatar when there is one.
        display_name = entry["user_display_name"]
        claimer = log_feed._claimer_id(claims, display_name)
        avatar_bytes = await self._avatar_bytes(claimer) if claimer is not None else None

        try:
            png = await profile_card.render_card(
                display_name=display_name,
                subtitle=log_feed.format_standing(entry, contest),
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
