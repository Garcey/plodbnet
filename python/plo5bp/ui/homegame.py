"""Private PLO5 / PLO6 / PLO67 double-board bomb-pot home games (PokerNow-style).

Installed only from ``public.install`` (public build). Every signed-in user
has home games; a CLUB is the private circle — a club's tables, members and
numbers are its members' only (``_table_access``; clubs replaced the old
admin-granted ``homegame_access`` flag on 2026-09-25, which now only means "a
member of the MAIN club"). Signed out, the ``/games`` pages are a sign-in page
and everything else under ``/games`` is the hidden 404.

The game (2026-09-26): a table deals PLO5 (five hole cards, up to 8 seats) or
PLO6 (six hole cards, up to 7 seats — 7 x 6 + two boards is the whole deck; no
burn cards online), chosen when the table is created and fixed for its life
(like the blinds). Everything else is the same game. Only PLO5 decisions are
graded (there is no PLO6 / PLO67 network yet), and the club's numbers are kept
apart per game (``homegames.variant``).

PLO67 (2026-09-27, the owner's friends' game): four hole cards, and the three
burn cards are dealt FACE UP — one before the flops, one before the turns, one
before the rivers. Every red burn deals everyone still in the hand (all-in
players too, folded ones not) one more hole card: 4-5 cards on the flop, 4-6 on
the turn, 4-7 on the river. Up to 5 seats (the slot map reserves seven cards a
seat + 10 board + 3 burns = 48). The engine plays it (``plo67_double_bomb``);
the view shows the burns turned up so far, an all-in runout reveals burn by
burn with each hand trimmed to what it held on the street being shown, and a
hand record keeps each seat's cards in deal order with its count per street.

Stakes: small/big blind set the chip unit and the dollar ledger.
Gameplay is ClubGG-style bomb pots — every in-hand seat posts the ante,
no blinds are posted, the hand starts on the flop.

Money rules (review 2026-09-20 G10) — the ledger is exactly zero-sum:

- All game math is in engine CHIPS (``BB_CHIPS`` per big blind). Cents exist
  only at the boundary: money in (buy-in / rebuy / auto-stack top-up) and
  money out (cash-out), always whole cents, and display.
- The money on a table is ``M = sum(buyin_cents) - sum(leftover_cents)``
  over everyone who ever played it. Chips are finer than cents (pots split
  across two boards and between tied hands), so stacks pick up sub-cent
  remainders. The cents value of the seated stacks is therefore the
  LARGEST-REMAINDER APPORTIONMENT of ``M`` in proportion to chips
  (``apportion_cents``; ties go to the lower seat). With whole-cent stacks
  that is simply "the value of my chips"; otherwise each odd cent goes to
  the largest fractional part. Ledger rows and cash-outs both use it — a
  player who leaves is paid exactly the stack value their ledger row showed
  — so ``sum(net_cents) == 0`` at every moment, and the last player to cash
  out receives exactly what is left.
- Play, buy-ins, rebuys and auto-stack never create or destroy a chip: the
  table's chips are worth exactly ``M``. A cash-out pays whole cents for a
  stack that may hold a fraction of one; that fraction (< 1 cent per
  cash-out) stays with the remaining stacks through the apportionment.
- Raise sizes are snapped to whole-cent chip multiples when the stake
  allows (``BB_CHIPS % bb_cents == 0``); auto-stack moves whole cents and
  carries a stack's sub-cent remainder instead of erasing it.

Game flow (2026-09-21 — the "premium tables" pass):

- The SERVER deals the next hand (``deal_delay_secs`` after a hand — and its
  runout — is over; 0 = manual). It used to be a timer in the host's browser,
  so a host who switched tabs stalled the table for everyone.
- Time bank: once the base shot clock runs out the actor burns their own
  ``time_bank_left`` before being auto-acted; only the seconds actually used
  are deducted, and a little comes back every hand played.
- Hand history is persisted per hand (``homegame_hands``) with EVERY dealt
  hand's cards; the API filters per viewer with the same rule as the live
  table — your own cards, plus hands tabled at a real showdown or shown
  voluntarily (``/show``). Nothing about a hand is served while its runout is
  still revealing.
- ``events`` (joins, rebuys, timeouts, wins …) and ``reactions`` are small
  in-memory feeds the client turns into dealer lines, toasts and emotes.

Chips in (2026-09-22):

- **Host approval** (``approve_buyins``): a buy-in / top-up by anyone but the
  host or a TRUSTED player becomes a pending request the host approves or
  declines; a pending sit request holds its seat. Trust is per player per
  table (``homegame_players.trusted``) — trusted players never wait.
- **Auto top-up** (``topup_mode``): when a stack has dropped below its
  threshold it is topped back UP to the target at the next deal. Never trims
  — no ratholing. **Set stack** (``auto_stack_mode``, the older feature): the
  stack is reset to the target before EVERY deal, up or down — ratholing is
  the point. Each is off / host-set / player's-choice; set-stack wins when a
  seat has both. While approval is on, neither runs for an untrusted player
  (an automatic buy-in is still a buy-in nobody approved).
- A top-up asked for while holding cards is QUEUED (``queue: true``) and lands
  when the hand is over; without the flag it is refused as before.
- ``GET …/stream`` pushes the viewer's state over SSE whenever it changes; the
  450 ms poll stays as the client's fallback.

The code (HGB-006, 2026-09-28): this module is the tables and the hand — state,
dealing, the clock, the verifiable shuffle, money, the view, grading — and the
entry points (``install``, ``shutdown``, ``set_model_provider``). The process's
state lives in ONE ``HomeGames`` object (``CTX``; ``use_context`` swaps it; the old
names ``HUB``, ``_WATCHDOG_THREAD`` … read and write it). The parts that barely
touch the hand live beside it — ``homegame_schema``, ``homegame_people``,
``homegame_clubs``, ``homegame_stats``, ``homegame_pages``, ``homegame_routes``
(see ``SPLIT_MODULES`` near the end): each reaches every home-games name through
this module when it runs, and everything they define is re-exported here, so
``homegame.X`` — importing it, calling it, patching it — works for all of it.
"""

from __future__ import annotations

import functools
import inspect
import json
import queue
import logging
import math
import os
import secrets
import threading
import sys
import time
import types
import unicodedata
import zlib
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

from plo5bp.actions import ALL_IN, CHECK_CALL, FOLD, GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.config import VARIANT_PLO5, VARIANT_PLO6, VARIANT_PLO67, GameConfig
from plo5bp.env import BombPotEnv, StepInfo
from plo5bp.ui.common import STREET_NAMES, position_name
from plo5bp.ui.hand_describe import describe_made_hand
from plo5bp.ui.runout import (
    AWARD_SECS, board_equities, build_awards, display_pots, money_flows, plo67_equities,
    uncalled_bet,
)
from plo5bp.ui import fairdeal
from plo5bp.ui import homegame_export
from plo5bp.ui import public as pub
from plo5bp.ui.ratelimit import KeyedCounter, RateLimiter

logger = logging.getLogger("plo5bp.ui.homegame")

BB_CHIPS = 10_000
TABLE_SEATS = 8  # the most seats any game has (PLO5); see GAMES for each game's own

#: The games a table can deal: short code -> engine variant, hole cards per
#: player ("hole" = the most a hand can hold = the verified shuffle's slots per
#: seat; "dealt" = what a hand starts with), face-up burn cards, seat limit (one
#: deck: seats x hole + 10 board + burns <= 52), whether the network grades its
#: decisions, and how the table names it.
GAMES: dict[str, dict[str, Any]] = {
    "plo5": {"variant": VARIANT_PLO5, "hole": 5, "dealt": 5, "burns": 0, "max_seats": 8,
             "graded": True, "label": "PLO5", "name": "PLO5 double-board bomb pot"},
    "plo6": {"variant": VARIANT_PLO6, "hole": 6, "dealt": 6, "burns": 0, "max_seats": 7,
             "graded": False, "label": "PLO6", "name": "PLO6 double-board bomb pot"},
    # four cards dealt; every red face-up burn deals everyone still in the hand one more
    "plo67": {"variant": VARIANT_PLO67, "hole": 7, "dealt": 4, "burns": 3, "max_seats": 5,
              "graded": False, "label": "PLO67", "name": "PLO67 double-board bomb pot"},
}
DEFAULT_GAME = "plo5"

# Hard limits (review 2026-09-20 G4/G13). Every money field is capped far
# below what sqlite (i64) / the engine (u64) can hold, so a value that made
# it into memory can ALWAYS be persisted.
MAX_CENTS = 1_000_000_000  # $10M — per amount, and per player's total buy-in
MAX_NAME_LEN = 60


def _env_max_tables() -> int:
    return int(os.environ.get("PLO5BP_HOMEGAME_MAX_TABLES", "5"))


MAX_OPEN_TABLES_PER_USER = _env_max_tables()  # (read again by `install` — `_read_env_settings`)
CHAT_MAX_LEN = 240
CHAT_RATE_MAX = 6  # messages ...
CHAT_RATE_WINDOW_S = 10.0  # ... per user per table per window
# New tables ship WITH a shot clock (review G6: clock-less tables wedge on an
# AFK actor). 0 = unlimited stays selectable by the host.
DEFAULT_DECISION_SECS = 30
# Time bank: per-seat reserve burned after the base clock; +TIME_BANK_REFILL_S
# back per hand dealt in, never above the table's ``time_bank_secs``.
MAX_TIME_BANK_SECS = 300
TIME_BANK_REFILL_S = 2.0
# Server-driven dealing: seconds after a hand (and its runout) is over.
# 0 = manual. A request that does not name it gets MANUAL — scripted callers
# (tests) stay deterministic; the create dialog always sends a value.
MAX_DEAL_DELAY_SECS = 30.0
# The server only deals to a table somebody is actually AT: a player counts as
# present while their browser has polled within this window (a hidden tab still
# polls every ~2 s; a closed one does not). Without it a table left running
# overnight kept posting antes and moving money between absent players.
PRESENCE_WINDOW_S = 20.0
# Timing out this many decisions in a row sits the player out (they come back
# with "I'm back"); an absent player then costs the table no more clock.
TIMEOUTS_BEFORE_SIT_OUT = 2
# A pending sit request holds its seat; it lapses when the requester's browser
# has been gone this long.
REQUEST_TTL_ABSENT_S = 90.0
MAX_REQUESTS = 24
# SSE: how often the stream looks for a change, and the longest it stays quiet
# (keep-alive + presence; Cloudflare drops idle streams after ~100 s).
STREAM_TICK_S = 0.12
STREAM_HEARTBEAT_S = 2.5
MIN_SEATS = 2
EVENT_FEED_MAX = 60
REACTION_TTL_S = 6.0
REACTIONS = (
    "gg", "nh", "ty", "gl", "lol", "wow", "cry", "angry", "fire", "clap",
    "think", "ship",
)
HANDS_PAGE_MAX = 50


def _env_grading_on() -> bool:
    return os.environ.get("PLO5BP_HOMEGAME_GRADING", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


# Every action of every hand is graded against the network in the background
# (never on the request path, never during the hand): the same node distribution
# and the same scorer as the Trainer tab, from the ACTOR's own point of view.
GRADING_ON = _env_grading_on()  # (read again by `install` — `_read_env_settings`)
# The verifiable shuffle (``fairdeal``): the deck is sealed before the players'
# devices contribute, their numbers re-permute it, and every card a player is
# shown comes with its proof. Needs the engine's deal-from-an-explicit-deck entry
# point; an older engine build (or PLO5BP_HOMEGAME_FAIR=0) deals the old way and
# the client says "unverified" — never an error at the table.
FAIR_ON = (
    os.environ.get("PLO5BP_HOMEGAME_FAIR", "1").strip().lower() not in ("0", "false", "no", "off")
    and hasattr(BombPotEnv, "reset_with_deck")
)
try:  # the binding itself (an old _engine build lacks it)
    from plo5bp import env as _env_mod
    FAIR_ON = FAIR_ON and hasattr(_env_mod._RustGameState, "reset_with_deck")
except Exception:  # noqa: BLE001
    FAIR_ON = False
try:  # PLO67 needs an engine that plays it (face-up burns, extra hole cards)
    from plo5bp import _engine as _engine_mod
    PLO67_ON = hasattr(_engine_mod, "plo67_runout_equities")
except Exception:  # noqa: BLE001
    PLO67_ON = False
#: All-in runouts are paced by what each street SHOWS (owner, 2026-09-29: the
#: showdown began before the river was down, and a PLO67 runout — burn, extra
#: hole cards, board cards — went by too fast to follow). A street is revealed,
#: then PRESENTED — PLO67: its burn shown big, flipped and captioned, then (red)
#: one more card to each hand still in, one hand after another, then the boards'
#: cards; the other games deal a street's cards within the host's pause — and held
#: for the host's runout pause before the next one. The showdown waits until the
#: river is down. The table animates with exactly these durations (the view sends
#: them as ``runout.timing``), so the two never drift apart (``_runout_plan``).
RUNOUT_BURN_S = 2.0          # PLO67: a burn shown big, flipped, captioned, dropped into its slot
RUNOUT_EXTRA_S = 0.6         # PLO67: one extra card's flight to a hand
RUNOUT_EXTRA_GAP_S = 0.35    # PLO67: from one hand's extra card to the next one's
RUNOUT_BOARD_S = 0.6         # a street's board cards landing
RUNOUT_AWARD_BEAT_S = 0.3    # PLO67: after the river is down, before the showdown
#: Monte-Carlo runouts behind each street's PLO67 all-in equities (Rust, ~10 ms).
PLO67_EQ_SAMPLES = 3000
FAIR_REVEAL_S = 3.0        # a locked device has this long to reveal its number
FAIR_RECOMMIT_S = 1.5      # after a voided attempt: window to commit to the new seal
FAIR_COMMIT_GRACE_S = 0.6  # at the deal: wait this long for a known device's commit
FAIR_PRESENT_S = 8.0       # "its browser is here" for the purposes of that wait
FAIR_MAX_ATTEMPTS = 4      # then the hand is dealt with whoever confirmed
FAIR_STRIKES = 2           # missed confirmations in a row before a time-out …
FAIR_PENALTY_HANDS = 20    # … of this many hands from contributing
#: The home-games client, served ONLY through the gated `/games/static/{name}`
#: route. Every name here must also be "deny" in server.py's
#: `_PUBLIC_STATIC_POLICY` (the public /static mount must never serve them).
GAMES_ASSETS: dict[str, str] = {
    "games.js": "text/javascript; charset=utf-8",
    "games.table.js": "text/javascript; charset=utf-8",
    "games.ui.js": "text/javascript; charset=utf-8",
    # (the UI's feature modules, split out of games.ui.js — FE-005)
    "games.lobby.js": "text/javascript; charset=utf-8",
    "games.history.js": "text/javascript; charset=utf-8",
    "games.review.js": "text/javascript; charset=utf-8",  # (Hand review: handreview_store)
    "games.seat.js": "text/javascript; charset=utf-8",
    "games.manage.js": "text/javascript; charset=utf-8",
    "games.play.js": "text/javascript; charset=utf-8",
    "games.sound.js": "text/javascript; charset=utf-8",
    "games.fair.js": "text/javascript; charset=utf-8",
    # the table's look (felt, cards, seats, bets, the dock) — Study / Trainer load it too
    "games.felt.css": "text/css; charset=utf-8",
    "games.css": "text/css; charset=utf-8",
}
# Hub eviction: nobody polling for this long => drop the in-memory table
# (it reloads from the DB on the next request).
HUB_CLOSED_EVICT_S = 60.0
HUB_IDLE_EVICT_S = 30 * 60.0
HUB_ABANDONED_EVICT_S = 6 * 3600.0  # even mid-hand: the hand is void


def _make_env(cfg: GameConfig) -> BombPotEnv:
    # (env.py and the engine ship together — scripts/deploy_prod.sh rebuilds the
    # engine — so no fallback for an older constructor: HGB-012.)
    return BombPotEnv(cfg, ev_runout_samples=0, obs_mode="minimal")


def _obs_dict(env: BombPotEnv) -> dict[str, Any]:
    return env.observation_dict()  # (the env's public accessors, never its engine handle — HGB-023)


def _div_half_up(num: int, den: int) -> int:
    """round(num/den), half away from zero, in exact integer arithmetic
    (``round()`` is banker's and ``/`` goes through a float)."""
    sign = -1 if num < 0 else 1
    return sign * ((abs(num) * 2 + den) // (2 * den))


def cents_to_chips(cents: int, bb_cents: int) -> int:
    if bb_cents <= 0:
        raise HTTPException(status_code=400, detail="big blind must be positive")
    return _div_half_up(int(cents) * BB_CHIPS, int(bb_cents))


def chips_to_cents(chips: int, bb_cents: int) -> int:
    """DISPLAY rounding of one chip amount. Never sum these for accounting —
    the ledger uses ``apportion_cents`` (see the module docstring)."""
    if bb_cents <= 0:
        return 0
    return _div_half_up(int(chips) * int(bb_cents), BB_CHIPS)


def chips_per_cent(bb_cents: int) -> int:
    """Whole chips in one cent, or 0 when the stake does not divide evenly
    (e.g. a 3c big blind) and whole-cent chip multiples do not exist."""
    bb = int(bb_cents)
    return BB_CHIPS // bb if bb > 0 and BB_CHIPS % bb == 0 else 0


def zero_sum_cents(deltas_chips: list[int], bb_cents: int) -> list[int]:
    """One hand's results in whole cents that still sum to ZERO (OPS-015): each
    seat's exact value floored, and the cents that leaves over go to the largest
    fractions (ties to the lower seat) — every amount is within a cent of exact,
    and a hand's stored results add up the way its chips do. (Rounding each seat
    on its own let a split pot's results sum to +-1 cent.)"""
    bb = int(bb_cents)
    if bb <= 0 or not deltas_chips:
        return [0] * len(deltas_chips)
    scaled = [int(d) * bb for d in deltas_chips]
    base = [v // BB_CHIPS for v in scaled]
    rems = [v % BB_CHIPS for v in scaled]
    extra = sum(rems) // BB_CHIPS  # exact: the chips sum to a whole number of cents
    for i in sorted(range(len(base)), key=lambda k: (-rems[k], k))[: max(0, extra)]:
        base[i] += 1
    return base


def apportion_cents(total_cents: int, chips: list[int]) -> list[int]:
    """Split ``total_cents`` across stacks in proportion to ``chips`` by the
    largest-remainder method; ``sum(result) == max(0, total_cents)``.

    Deterministic: each odd cent goes to the largest fractional remainder,
    ties to the lower index; seats without chips only receive money when
    nobody has any (the residue then goes to the first seat)."""
    n = len(chips)
    total = max(0, int(total_cents))
    if n == 0:
        return []
    weights = [max(0, int(c)) for c in chips]
    pool = sum(weights)
    if pool <= 0:
        return [total] + [0] * (n - 1)
    out = [(total * w) // pool for w in weights]
    rems = [(total * w) % pool for w in weights]
    left = total - sum(out)
    for i in sorted(range(n), key=lambda k: (-rems[k], k)):
        if left <= 0:
            break
        if weights[i] > 0:
            out[i] += 1
            left -= 1
    return out


def _norm_game(v: Any) -> str:
    """A stored / loaded game code; anything unknown is the original game."""
    s = str(v or "").strip().lower()
    return s if s in GAMES else DEFAULT_GAME


def _parse_game(v: Any) -> str:
    """The game a request names: ``plo5`` / ``plo6`` (the engine's variant names
    are accepted too). Unknown = 400, never a silent PLO5."""
    s = str(v or "").strip().lower()
    for code, g in GAMES.items():
        if s in (code, g["variant"]):
            return code
    raise HTTPException(status_code=400, detail="unknown game — choose PLO5, PLO6 or PLO67")


def _game_filter(v: Any) -> str | None:
    """A stats query's game (None = every game)."""
    if v is None or str(v).strip().lower() in ("", "all"):
        return None
    return _parse_game(v)


def _game_info(v: Any) -> dict[str, Any]:
    """What a client needs to know about a game (served on every table view)."""
    code = _norm_game(v)
    g = GAMES[code]
    return {"code": code, "label": g["label"], "name": g["name"], "hole": int(g["hole"]),
            "dealt": int(g["dealt"]), "burns": int(g["burns"]),
            "max_seats": int(g["max_seats"]), "graded": bool(g["graded"])}


def _sorted_hole(cards: list[int] | None) -> list[int] | None:
    """Highest-to-lowest by card index (rank-major), matching trainer display."""
    if cards is None:
        return None
    return sorted((int(c) for c in cards), reverse=True)


def _fmt_cents(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    v = abs(int(cents))
    return f"{sign}${v / 100:.2f}"


AUTO_STACK_MODES = ("off", "host", "player")


def _norm_auto_mode(v: Any) -> str:
    s = str(v or "off").strip().lower()
    return s if s in AUTO_STACK_MODES else "off"


# SEC-007: text people type (table / club names, chat) and names from Google
# lose Unicode control and format characters — a right-to-left override, a
# zero-width space, a bidi isolate can make a line pretend to be someone else's —
# except the ones emoji are built from (zero-width joiner, variation selectors,
# the tag characters of subdivision flags). Whitespace runs become one space.
_EMOJI_FORMAT = {"\u200d", "\ufe0e", "\ufe0f"}


def _clean_text(v: Any) -> str:
    s = "".join(
        ch for ch in str(v or "")
        if unicodedata.category(ch) not in ("Cc", "Cf", "Co", "Cs")
        or ch in _EMOJI_FORMAT or 0xE0020 <= ord(ch) <= 0xE007F
        or ch.isspace()  # (whitespace controls: collapsed to a space below)
    )
    return " ".join(s.split())


def _gate_key(gate: int | str) -> int:
    if gate in (GATE_FOLD, GATE_CHECK_CALL, GATE_RAISE):
        return int(gate)
    s = str(gate).strip().lower().replace("-", "_")
    if s in ("fold", "0"):
        return GATE_FOLD
    if s in ("check", "call", "check_call", "checkcall", "1"):
        return GATE_CHECK_CALL
    if s in ("raise", "bet", "allin", "all_in", "2"):
        return GATE_RAISE
    raise HTTPException(status_code=400, detail=f"unknown gate {gate!r}")


@dataclass
class Seat:
    user_id: int
    name: str  # display name (never the email: HGB-011 dropped the unused copy)
    stack_chips: int
    sitting_out: bool
    buyin_cents: int
    leftover_cents: int
    auto_stack_cents: int = 0
    # In-memory only (a reloaded table comes back paused with full banks).
    time_bank_left: float = 0.0
    sit_out_next: bool = False
    timeouts: int = 0  # consecutive decisions the clock made for them
    trusted: bool = False  # buys in without the host's approval
    topup_target_cents: int = 0  # auto top-up: back up to this ...
    topup_below_cents: int = 0  # ... once the stack is below this (0 = target)
    queued_topup_cents: int = 0  # asked for mid-hand; lands when the hand ends
    queued_remove_cents: int = 0  # chips to take OFF the table when the hand ends
    leave_after_hand: bool = False  # finish this hand, then leave the seat
    # The network at this seat (2026-10-05, the site's owner only — homegame_bot): ""
    # off, "assist" (it shows its move, the player still acts) or "auto" (it acts);
    # bot_mix = its full strategy (sampled) instead of its favourite move. In memory
    # only: a reloaded table comes back with the network off.
    bot_mode: str = ""
    bot_mix: bool = False


@dataclass
class HandState:
    """Everything that belongs to ONE hand: the engine, who was dealt in, the
    clock of the decision on, the runout's script, the rabbit, the shuffle it
    was dealt from (HGB-004). A deal, a resize and a close start from a fresh
    ``HandState()``, so a field can never be left over from the hand before —
    ``pots`` used to survive the deal, and a resize kept ``equity_by_len``,
    ``rabbit_burns`` and ``terminal_*``. ``LiveTable`` exposes every field under
    its old name (``t.env`` is ``t.hand.env``)."""

    env: BombPotEnv | None = None
    info: StepInfo | None = None
    # Actions applied this hand. With hand_no it names a decision: clients
    # echo both on /act and get 409 when they are stale (G8).
    action_seq: int = 0
    # The decision the running shot clock belongs to: (hand_no, action_seq).
    # The clock restarts only when this changes (review 2026-09-20 G5).
    turn_started_mono: float | None = None
    turn_key: tuple[int, int] | None = None
    # The decision whose base clock has run out and is burning the bank.
    bank_key: tuple[int, int] | None = None
    bank_started_mono: float | None = None
    hand_start_stacks: list[int] = field(default_factory=list)
    in_hand_mask: list[bool] = field(default_factory=list)
    # user id DEALT INTO each seat this hand (None = seat not dealt in). Own
    # cards are shown against this, never against who sits there now (G2).
    dealt_user_ids: list[int | None] = field(default_factory=list)
    # Terminal with >= 2 live hands = a real showdown; a fold-out is not (G1).
    showdown_reveal: bool = False
    last_deltas: list[int] = field(default_factory=list)
    # Seats that chose to table their cards after the hand (phase showdown).
    shown_seats: set[int] = field(default_factory=set)
    # The all-in runout: streets and awards revealed by the clock.
    runout_active: bool = False
    runout_start_len: int = 3
    runout_started_mono: float | None = None
    # When each street is revealed / fully shown and when the showdown starts,
    # seconds after runout_started_mono (``_make_runout_plan``; fixed at the start,
    # so the host changing the pause mid-runout never makes a street jump).
    runout_plan: dict[str, Any] = field(default_factory=dict)
    leftover_stacks: list[int] = field(default_factory=list)
    terminal_pot: int = 0
    terminal_commit: list[int] = field(default_factory=list)
    pot_awards: list[dict[str, Any]] = field(default_factory=list)
    pots: list[dict[str, Any]] = field(default_factory=list)  # named layers, deepest first
    # The bet nobody matched (seat -> chips): it went back to its owner when the
    # betting closed — no pot, no award (``runout.uncalled_bet``; owner, 2026-10-02).
    uncalled: dict[int, int] = field(default_factory=dict)
    # (len_a, len_b) -> {seat: {"a": share, "b": share}}, computed ONCE per
    # all-in hand from the alive seats' holes (review 2026-09-20 G3).
    equity_by_len: dict[tuple[int, int], dict[int, dict[str, float]]] = field(default_factory=dict)
    # The rabbit (a fold-out's undealt streets) and, PLO67, all three burns of
    # the finished hand (the rabbit and an all-in runout reveal them street by street).
    rabbit_available: bool = False
    rabbit_shown: bool = False
    rabbit_played_len: int = 3
    rabbit_full_a: list[int] = field(default_factory=list)
    rabbit_full_b: list[int] = field(default_factory=list)
    rabbit_burns: list[int] = field(default_factory=list)
    # What the background grader needs to replay the hand: the deal seed (or the
    # dealt deck) and the exact engine inputs. In MEMORY only — the seed would
    # reveal every card, so it is never persisted and never served.
    hand_seed: int = 0
    hand_actions: list = field(default_factory=list)
    hand_deck: list = field(default_factory=list)
    # The verifiable shuffle this hand was dealt from, and its public facts.
    fair_hand: Any = None
    fair_hand_meta: dict = field(default_factory=dict)
    # The network's part in this hand (homegame_bot): action index -> "auto" (it acted)
    # or "assist" (the player acted with its move on the screen) — never graded; its
    # move per decision (hand_no, action_seq); when autopilot first saw each decision.
    bot_marks: dict = field(default_factory=dict)
    bot_cache: dict = field(default_factory=dict)
    bot_asked: dict = field(default_factory=dict)


@dataclass
class LiveTable:
    game_id: str
    host_user_id: int
    name: str
    num_seats: int
    sb_cents: int
    bb_cents: int
    ante_cents: int
    default_buyin_cents: int
    status: str
    button: int
    hand_no: int
    seats: list[Seat | None]
    running: bool = False
    club_id: str | None = None  # the club it belongs to: only members see / sit / watch it
    auto_stack_mode: str = "off"  # off | host | player
    auto_stack_all_cents: int = 0
    decision_secs: int = 0  # 0 = unlimited
    street_pause_secs: float = 1.5
    # Players leaving / being removed mid-hand: folded-or-checked by the
    # away logic, cashed out once the hand (and its runout) is over.
    pending_kicks: set[int] = field(default_factory=set)
    phase: str = "waiting"  # waiting | in_hand | showdown
    # The hand on the table (or the one just finished): HGB-004.
    hand: HandState = field(default_factory=lambda: HandState())
    # PERF-004: the table-wide half of the view (``_shared``), built once per state
    # of the table for every viewer; who among the viewers only has the "Player
    # <id>" stand-in name; the trust of unseated viewers, per ``rev``.
    view_cache: Any = None
    name_defaults: dict = field(default_factory=dict)
    trust_cache: Any = None
    # Bumped on every state change; `epoch` is unique per in-memory load.
    # Clients drop responses older than the one they already applied (G8).
    rev: int = 0
    epoch: str = field(default_factory=lambda: secrets.token_hex(4))
    # A persist failed after memory had already advanced (engine paths);
    # the watchdog retries (review 2026-09-20 G4).
    persist_dirty: bool = False
    persist_retry_mono: float = 0.0
    # Writes the table still owes the database, committed WITH the stacks by the
    # next ``_persist_safe`` (OPS-006 / OPS-013): ("hand", bundle) = a finished
    # hand's record + results + flows + grading job; ("fair", hand_no, data) = a
    # hand's shuffle transcript.
    unsaved: list = field(default_factory=list)
    # Watchdog health (OPS-002): failing ticks in a row, since when, the last
    # time it was logged, and (backing off) when the table is next ticked.
    wd_failures: int = 0
    wd_failing_since: float | None = None
    wd_logged_mono: float = -1e9
    wd_next_mono: float = 0.0
    last_access_mono: float = field(default_factory=time.monotonic)
    chat_times: dict[int, deque] = field(default_factory=dict)
    chat_cache: list | None = None  # the newest chat lines (None = read them again)
    # --- 2026-09-21 -------------------------------------------------------
    deal_delay_secs: float = 0.0  # 0 = manual dealing
    time_bank_secs: int = 0  # per-seat reserve; 0 = off
    min_buyin_cents: int = 0  # 0 = no limit
    max_buyin_cents: int = 0
    listed: bool = True  # shown in every granted user's lobby
    allow_rabbit: bool = True
    # When the server will deal the next hand (monotonic), or None.
    next_deal_mono: float | None = None
    # FEAT-005: why the server's last automatic deal failed, and the table's ``rev``
    # right after it — it is not retried until something at the table changes.
    deal_error: str | None = None
    deal_error_rev: int = -1
    events: deque = field(default_factory=lambda: deque(maxlen=EVENT_FEED_MAX))
    event_seq: int = 0
    reactions: deque = field(default_factory=lambda: deque(maxlen=40))
    reaction_seq: int = 0
    # The finished hand's one-line result, held back while its runout reveals.
    pending_result: dict | None = None
    # user id -> monotonic time of their last poll (presence, see above)
    seen: dict[int, float] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)  # display names of `seen`
    approve_buyins: bool = False
    topup_mode: str = "off"  # off | host | player
    topup_all_target_cents: int = 0
    topup_all_below_cents: int = 0
    # Pending buy-in / top-up requests (approval mode). In memory: a restart
    # simply asks the player to request again.
    requests: list = field(default_factory=list)
    request_seq: int = 0
    # Accuracy marks on EVERYONE's actions in the replayer (a mark on a mucked
    # hand says a little about it). Off = players see marks on their own only.
    show_grades: bool = True
    # Players may take chips OFF the table between hands ("ratholing"). Off by
    # default: the usual table-stakes rule is that winnings stay in play.
    allow_rathole: bool = False
    # Verifiable shuffle: ``fair_next`` = the sealed deck of the UPCOMING hand and
    # where its confirmation stands (the hand on the table's is ``hand.fair_hand``).
    fair_next: Any = None
    fair_capable: set = field(default_factory=set)       # user ids whose browser takes part
    fair_strikes: dict = field(default_factory=dict)     # user id -> missed reveals in a row
    fair_penalty_until: dict = field(default_factory=dict)  # user id -> hand_no
    fair_void_counts: dict = field(default_factory=dict)  # user id -> voided shuffles this session
    # Requests to join this table's CLUB (``homegame_club_requests``) that its
    # owner / an admin sitting here may decide. Re-read at most every
    # JOIN_REFRESH_S so a decision made elsewhere clears them too.
    join_reqs: list = field(default_factory=list)
    join_checked_mono: float = 0.0
    club_info: dict[str, Any] | None = None
    club_checked_mono: float = 0.0
    # The game dealt here (GAMES code) — chosen at creation, fixed for the table's life.
    variant: str = DEFAULT_GAME
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def ante_chips(self) -> int:
        return cents_to_chips(self.ante_cents, self.bb_cents)

    @property
    def game(self) -> dict[str, Any]:
        return GAMES[_norm_game(self.variant)]

    @property
    def hole_count(self) -> int:
        """The most hole cards a hand holds here (= the shuffle's slots per seat)."""
        return int(self.game["hole"])

    @property
    def burns(self) -> int:
        """Burn cards dealt face up (PLO67 3, else 0)."""
        return int(self.game["burns"])

    def occupied(self) -> list[int]:
        return [i for i, s in enumerate(self.seats) if s is not None]

    def seat_of(self, uid: int) -> int | None:
        for i, s in enumerate(self.seats):
            if s is not None and s.user_id == uid:
                return i
        return None

    def player(self, uid: int) -> Seat | None:
        i = self.seat_of(uid)
        return self.seats[i] if i is not None else None


def _hand_attr(name: str) -> property:
    return property(lambda t: getattr(t.hand, name), lambda t, v: setattr(t.hand, name, v),
                    doc=f"``hand.{name}`` (HGB-004)")


#: Every per-hand field under its old name on the table (``t.env`` = ``t.hand.env``).
HAND_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(HandState))
for _f in HAND_FIELDS:
    setattr(LiveTable, _f, _hand_attr(_f))


@dataclass
class FairPending:
    """The shuffle of the upcoming hand: a sealed deck and its confirmation."""

    sealed: Any                 # fairdeal.SealedDeck
    hand_no: int
    attempt: int = 1
    stage: str = "commit"       # commit -> reveal (-> dealt: becomes LiveTable.fair_hand)
    commits: dict = field(default_factory=dict)       # seat -> commitment
    commit_users: dict = field(default_factory=dict)  # seat -> user id
    reveals: dict = field(default_factory=dict)       # seat -> number
    pending: bool = False       # a deal is waiting on this shuffle
    deadline_mono: float | None = None
    barred: set = field(default_factory=set)          # user ids that voided an attempt of this hand
    voids: list = field(default_factory=list)         # earlier attempts of this hand


# --- locks (HGB-018) -------------------------------------------------------------
#
# Every lock the home games take, OUTERMOST first. Code holding one may take a later
# one, never an earlier one:
#
#   1. ``Hub._loading[game_id]`` — one loader per table (``Hub.get``); held while the
#      table is read from the database, so it may take 2 and 4, never 3.
#   2. ``Hub._lock`` — the dict of loaded tables. Held only to read or change that
#      dict (and a few lock-free reads of table fields): NOTHING is taken under it.
#   3. ``LiveTable.lock`` — ONE table at a time. Code that visits several tables (the
#      clock, a new profile picture, a club's join requests, shutdown) takes their
#      locks one after another, never nested. A table's lock may be held while the
#      database is used (4), never the other way round: no table lock is taken
#      inside ``pub.DB.transaction()``.
#   4. ``pub.DB``'s lock — innermost (every statement, every transaction).
#
#   Leaf locks, held for a few lines and never while taking another: ``CTX.avatar_lock``,
#   ``CTX.grade_lock``.
#
# A function named ``*_locked`` expects its table's lock to be held by the caller
# (so do ``_view``, ``_mutation``, ``_persist_safe`` and ``_cash_out_seat``). In the
# test session (``LOCK_CHECKS``) every one of them checks it (``_assert_lock_held``):
# a bare call — which in production would race the clock thread — fails the test.

#: On in the test session (pytest sets PYTEST_CURRENT_TEST while it runs a test,
#: which is when the fixtures import this module); PLO5BP_HOMEGAME_LOCK_CHECKS=1/0
#: forces it. Off in production: the checks cost a call per locked function.
LOCK_CHECKS = os.environ.get(
    "PLO5BP_HOMEGAME_LOCK_CHECKS", "1" if "PYTEST_CURRENT_TEST" in os.environ else "0"
).strip() == "1"


def _assert_lock_held(t: "LiveTable") -> None:
    """(LOCK_CHECKS only) the calling thread must hold ``t.lock``."""
    if LOCK_CHECKS:
        owned = getattr(t.lock, "_is_owned", None)
        if owned is not None and not owned():
            raise AssertionError(f"homegame: table {t.game_id}'s lock is not held by this thread")


class Hub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tables: dict[str, LiveTable] = {}
        # game id -> the hand number an abandoned table was evicted in the middle
        # of, so the reload can say why that hand vanished (OPS-012)
        self._abandoned: dict[str, int] = {}
        self._loading: dict[str, threading.Lock] = {}  # game id -> its loader's lock

    def pop_abandoned(self, game_id: str) -> int | None:
        with self._lock:
            return self._abandoned.pop(game_id, None)

    def get(self, game_id: str) -> LiveTable:
        with self._lock:
            t = self._tables.get(game_id)
            if t is not None:
                t.last_access_mono = time.monotonic()
                return t
            loading = self._loading.setdefault(game_id, threading.Lock())
        # Loaded OUTSIDE the hub lock (PERF-006: a load — several queries, maybe
        # a club created — used to stall every other table's requests, the
        # watchdog and the stats pages). One loader per table: a second request
        # waits for the first and gets the same copy, never a copy of its own.
        with loading:
            with self._lock:
                t = self._tables.get(game_id)
            if t is None:
                try:
                    t = _load_table(game_id)
                except BaseException:
                    with self._lock:
                        if self._loading.get(game_id) is loading:
                            del self._loading[game_id]
                    raise
                with self._lock:  # stored and the loader retired in one step
                    t = self._tables.setdefault(game_id, t)
                    if self._loading.get(game_id) is loading:
                        del self._loading[game_id]
            t.last_access_mono = time.monotonic()
            return t

    def peek(self, game_id: str) -> LiveTable | None:
        """The table if it is loaded — never loads it."""
        with self._lock:
            return self._tables.get(game_id)

    def put(self, table: LiveTable) -> None:
        with self._lock:
            self._tables[table.game_id] = table

    def drop(self, game_id: str) -> None:
        with self._lock:
            self._tables.pop(game_id, None)

    def evict_idle(self, now: float | None = None) -> list[str]:
        """Forget tables nobody is looking at (review 2026-09-20 G13 — the
        hub used to grow forever). Every request AND every live-stream push
        resolves its table through ``get`` (OPS-001), so "idle" means no
        browser has the table open. Everything that matters is in the DB; the
        next request reloads it (paused, between hands)."""
        now = time.monotonic() if now is None else now
        gone: list[str] = []
        with self._lock:
            for gid, t in list(self._tables.items()):
                if t.persist_dirty:
                    continue  # unsaved state: keep it until a write succeeds
                idle = now - t.last_access_mono
                if t.status != "open":
                    evict = idle > HUB_CLOSED_EVICT_S
                elif t.phase == "in_hand":
                    evict = idle > HUB_ABANDONED_EVICT_S
                else:
                    evict = idle > HUB_IDLE_EVICT_S
                if evict:
                    if t.phase == "in_hand":
                        logger.warning(
                            "evicting abandoned table %s mid-hand (hand void)", gid
                        )
                        if len(self._abandoned) >= 1000:
                            self._abandoned.pop(next(iter(self._abandoned)))
                        self._abandoned[gid] = int(t.hand_no)
                    del self._tables[gid]
                    gone.append(gid)
        return gone


#: Live streams one user may hold open at once (each is a connection and a loop
#: that builds their view): more tables than anyone watches, fewer than a flood (SEC-009).
MAX_STREAMS_PER_USER = 6
_EVICT_EVERY_S = 30.0
WATCHDOG_SLOW_TICK_S = 1.0     # a pass slower than this is logged (at most once a minute)
WATCHDOG_STALE_S = 10.0        # no pass for this long = the clock is stuck (health says so)


class HomeGames:
    """Everything the home games hold in this process's memory, in ONE object
    (HGB-006; the app factory's context — BE-007): the loaded tables (``hub``),
    the three workers — clock, grader, live streams — with their stop flags,
    threads and health, the grading queue, the model the grader scores with,
    the open-stream counts and the caches (pictures, names, club roles, the
    ledger check).

    The module keeps ONE current context, ``CTX``; ``use_context(HomeGames())``
    swaps all of it at once, so a test or an app factory can start from a clean
    slate without re-importing the module. The module's older names (``CTX.hub``,
    ``CTX.watchdog_thread``, ``CTX.grade_q`` …) still read — and write — the current
    context's (``_LEGACY_STATE``). What is not here is not per-context: the
    settings (module constants, read from the environment at import), the
    request budgets (``API_RATE`` …) and the per-process OS lock on the database
    (``_PROCESS_LOCKS``)."""

    def __init__(self) -> None:
        self.hub = Hub()
        # Three workers, one stop flag each (OPS-008: one flag used to stop the
        # clock, the grader AND every live stream together).
        self.watchdog_stop = threading.Event()  # the clock thread
        self.grader_stop = threading.Event()    # the grading thread
        self.streams_stop = threading.Event()   # every live stream (SSE)
        self.watchdog_started = False
        self.watchdog_thread: threading.Thread | None = None
        # Health (OPS-011): when the clock last finished a pass, and how long it took.
        self.watchdog_last_tick = 0.0
        self.watchdog_last_tick_s = 0.0
        self.watchdog_slow_logged = -1e9
        # The PLO5 model the grader scores with (HGB-016: server.py provides it).
        self.model_provider: Any = None
        # The grading queue (OPS-016): bounded; its jobs' (game_id, hand_no).
        self.grade_q: "queue.Queue[dict | None]" = queue.Queue()
        self.grade_thread: threading.Thread | None = None
        self.grade_lock = threading.Lock()
        self.grade_queued: set = set()
        # The network playing a seat (homegame_bot): its moves are worked out here,
        # off every table's lock.
        self.bot_q: "queue.Queue[tuple | None]" = queue.Queue()
        self.bot_thread: threading.Thread | None = None
        self.bot_stop = threading.Event()
        self.owner_cache: dict[int, tuple[float, bool]] = {}  # user id -> (until, the site's owner?)
        self.streams = KeyedCounter(MAX_STREAMS_PER_USER)  # open live streams per user (SEC-009)
        # Caches: picture URLs; chosen names and club nicknames (one lock); club roles
        # (dropped by any write to the clubs' tables); the last ledger check.
        self.avatar_urls: dict[int, str | None] = {}
        self.avatar_lock = threading.Lock()
        self.names_lock = threading.Lock()
        self.chosen_names: dict[int, str | None] = {}   # user id -> their name at the tables (None: none)
        self.club_nicks: dict[str, dict[int, str]] = {}  # club id -> {user id: nickname}
        self.role_cache: dict[tuple[str, int], str | None] = {}
        self.role_lock = threading.Lock()
        self.role_gen = [0]
        self.ledger_check: dict[str, Any] = {"at": -1e9, "out": None}


#: The current context (see HomeGames).
CTX = HomeGames()

#: The module's older names for the context's state (read AND written through the
#: module — see the end of the module): CTX.hub is CTX.hub, and so on.
_LEGACY_STATE = {
    "HUB": "hub", "_WATCHDOG_STOP": "watchdog_stop", "_GRADER_STOP": "grader_stop",
    "_STREAMS_STOP": "streams_stop", "_WATCHDOG_STARTED": "watchdog_started",
    "_WATCHDOG_THREAD": "watchdog_thread", "_WATCHDOG_LAST_TICK": "watchdog_last_tick",
    "_WATCHDOG_LAST_TICK_S": "watchdog_last_tick_s", "_WATCHDOG_SLOW_LOGGED": "watchdog_slow_logged",
    "_MODEL_PROVIDER": "model_provider", "_GRADE_Q": "grade_q", "_GRADE_THREAD": "grade_thread",
    "_GRADE_LOCK": "grade_lock", "_GRADE_QUEUED": "grade_queued", "STREAMS": "streams",
    "_AVATAR_URLS": "avatar_urls", "_AVATAR_LOCK": "avatar_lock", "_NAMES_LOCK": "names_lock",
    "_CHOSEN_NAMES": "chosen_names", "_CLUB_NICKS": "club_nicks", "_ROLE_CACHE": "role_cache",
    "_ROLE_LOCK": "role_lock", "_ROLE_GEN": "role_gen", "_LEDGER_CHECK": "ledger_check",
}


def use_context(ctx: HomeGames) -> HomeGames:
    """Make ``ctx`` the current context; returns the one it replaces (whose
    workers the caller stops — ``shutdown()`` — before or after, as it needs)."""
    global CTX
    old, CTX = CTX, ctx
    return old


# A table whose watchdog steps keep failing (OPS-002): logged at most once per
# WATCHDOG_LOG_EVERY_S, paused and announced after WATCHDOG_PAUSE_AFTER_S of
# failing ticks, and from then on ticked only every WATCHDOG_BACKOFF_S until a
# tick goes through cleanly.
WATCHDOG_LOG_EVERY_S = 60.0
WATCHDOG_PAUSE_AFTER_S = 10.0
WATCHDOG_BACKOFF_S = 2.0


def _watchdog_table(t: LiveTable, now: float) -> None:
    """One watchdog pass over one table. Every step runs on its own: one that
    raises (a bug, a database error, a shuffle that cannot be finished) is
    logged and counted but never stops the table's other steps — nor any other
    table (OPS-002: one try around the whole loop let a single failing table
    stop the clock of every table after it, four times a second, forever)."""
    if now < t.wd_next_mono or _watchdog_idle(t):
        return
    # PERF-007: a table whose lock is busy (a slow view, an all-in's equities)
    # is skipped this tick instead of holding up every table after it.
    if not t.lock.acquire(blocking=False):
        return
    errors: list[tuple[str, BaseException]] = []
    try:
        for step in (
            _bot_tick_locked, _timeout_tick_locked, _settle_locked, _flush_result_locked,
            _apply_queued_topups_locked, _expire_requests_locked, _fair_tick_locked,
            _auto_deal_tick_locked, _retry_persist_locked,
        ):
            try:
                step(t)
            except Exception as e:  # noqa: BLE001 — never kill the clock thread
                errors.append((getattr(step, "__name__", "step"), e))
        _watchdog_account_locked(t, now, errors)
    finally:
        t.lock.release()


def _watchdog_idle(t: LiveTable) -> bool:
    """Nothing a watchdog tick could do at this table (conservative: a paused
    table between hands with no request, leaver, queued chips, owed write or
    result waiting). PERF-007: most loaded tables are like this most of the time."""
    return (
        t.phase != "in_hand" and not t.running and not t.persist_dirty and not t.unsaved
        and not t.pending_kicks and not t.requests and t.pending_result is None
        and not (t.fair_next is not None and t.fair_next.pending)
        and not any(p is not None and (p.leave_after_hand or p.queued_topup_cents or p.queued_remove_cents)
                    for p in t.seats)
    )


def _watchdog_account_locked(t: LiveTable, now: float, errors: list) -> None:
    """Failure accounting for one tick of one table (see _watchdog_table)."""
    if not errors:
        if t.wd_failing_since is not None:
            logger.warning("homegame watchdog: table %s is ticking cleanly again", t.game_id)
        t.wd_failing_since = None
        t.wd_failures = 0
        return
    t.wd_failures += 1
    if t.wd_failing_since is None:
        t.wd_failing_since = now
    if now - t.wd_logged_mono >= WATCHDOG_LOG_EVERY_S:
        t.wd_logged_mono = now
        name, err = errors[0]
        logger.error(
            "homegame watchdog: table %s, %s failed (%d failing ticks in a row; %s)",
            t.game_id, name, t.wd_failures,
            ", ".join(n for n, _ in errors),
            exc_info=(type(err), err, err.__traceback__),
        )
    if now - t.wd_failing_since < WATCHDOG_PAUSE_AFTER_S:
        return
    t.wd_next_mono = now + WATCHDOG_BACKOFF_S
    if t.running and t.status == "open":
        t.running = False
        t.next_deal_mono = None
        t.rev += 1
        _persist_safe(t)
        _emit(t, "run", "Something went wrong at this table, so the game is paused"
                        + (" after this hand" if t.phase == "in_hand" else "")
                        + ". The host can press Start to carry on.")


def _watchdog_loop(ctx: HomeGames | None = None) -> None:
    ctx = ctx or CTX  # (a swapped context never steals a running clock)
    next_evict = time.monotonic() + _EVICT_EVERY_S
    ctx.watchdog_last_tick = time.monotonic()
    while not ctx.watchdog_stop.wait(0.25):
        with ctx.hub._lock:
            tables = list(ctx.hub._tables.values())
        now = time.monotonic()
        for t in tables:
            try:
                _watchdog_table(t, now)
            except Exception:  # noqa: BLE001 — the accounting itself failed
                logger.exception("homegame watchdog (table %s)", t.game_id)
        if time.monotonic() >= next_evict:
            next_evict = time.monotonic() + _EVICT_EVERY_S
            try:
                for gid in ctx.hub.evict_idle():
                    logger.info("homegame table %s unloaded (idle)", gid)
            except Exception:  # noqa: BLE001
                logger.exception("homegame hub eviction")
        done = time.monotonic()
        ctx.watchdog_last_tick, ctx.watchdog_last_tick_s = done, done - now
        if done - now > WATCHDOG_SLOW_TICK_S and done - ctx.watchdog_slow_logged >= 60.0:
            ctx.watchdog_slow_logged = done
            logger.warning("homegame clock: one pass over %d tables took %.1f s", len(tables), done - now)


def _start_watchdog() -> None:
    if CTX.watchdog_started:
        return
    CTX.watchdog_started = True
    CTX.watchdog_stop.clear()
    CTX.streams_stop.clear()
    CTX.watchdog_thread = threading.Thread(
        target=_watchdog_loop, args=(CTX,), name="homegame-clock", daemon=True
    )
    CTX.watchdog_thread.start()


def _ensure_workers() -> None:
    """Restart a worker that died (OPS-011 B) — cheap enough for every request.
    A clock thread can only die from a bug outside its per-table guard; without
    this, every shot clock, auto-deal and eviction would stop silently."""
    th = CTX.watchdog_thread
    if CTX.watchdog_started and not CTX.watchdog_stop.is_set() and (th is None or not th.is_alive()):
        logger.error("homegame clock thread was dead: restarting it")
        CTX.watchdog_started = False
        _start_watchdog()
    g = CTX.grade_thread
    if g is not None and not g.is_alive() and not CTX.grader_stop.is_set() and GRADING_ON:
        logger.error("homegame grading thread was dead: restarting it")
        _start_grader()


def health() -> dict[str, Any]:
    """The workers' health (OPS-011): served to site admins at /games/api/health."""
    now = time.monotonic()
    th = CTX.watchdog_thread
    last = CTX.watchdog_last_tick
    with CTX.hub._lock:
        tables = list(CTX.hub._tables.values())
    return {
        "clock": {
            "alive": bool(th is not None and th.is_alive()),
            "last_tick_age_s": round(now - last, 2) if last else None,
            "last_tick_s": round(CTX.watchdog_last_tick_s, 3),
            "stale": bool(last and now - last > WATCHDOG_STALE_S),
        },
        "grader": {
            "on": bool(GRADING_ON),
            "alive": bool(CTX.grade_thread is not None and CTX.grade_thread.is_alive()),
            "queued": len(CTX.grade_queued),
            "saved_jobs": int(pub.DB.one("SELECT COUNT(*) c FROM homegame_grade_jobs")["c"]),
        },
        "tables": {
            "loaded": len(tables),
            "in_hand": sum(1 for t in tables if t.phase == "in_hand"),
            "unsaved": sum(1 for t in tables if t.persist_dirty),
            "failing": sorted(t.game_id for t in tables if t.wd_failing_since is not None),
        },
        # OPS-017: does every player's money add up to their ledger rows?
        "ledger": _ledger_health(),
    }


def deploy_status(tables: list[LiveTable] | None = None) -> dict[str, Any]:
    """What a restart right now would interrupt, from this process's memory — the
    production deploy's "is anyone playing?" check (ops/deploytool.py ``status``)
    reads it through the loopback-only ``GET /health?deploy=1``. Exact, where the
    database can only guess (``hand_no`` is saved at the deal, and a voided hand is
    never recorded).

    ``hands_in_progress``: tables whose hand is being played or whose all-in runout
    is still revealing (``_hand_busy``). ``games_running``: tables the server is
    dealing to with at least two seated players at the table right now — the next
    hand is seconds away, and a restart pauses the game until its host presses
    Start. Table ids and hand numbers only (the deploy reads names from the
    database). Racy reads of plain attributes, never a table lock: a deploy must not
    wait on a busy table, and a snapshot a moment old is what it needs. (``tables``:
    for tests; default = every table this process holds.)"""
    now = time.monotonic()
    if tables is None:
        with CTX.hub._lock:
            tables = list(CTX.hub._tables.values())
    busy: list[dict[str, Any]] = []
    running: list[dict[str, Any]] = []
    for t in tables:
        if t.status != "open":
            continue
        present = sum(
            1 for p in list(t.seats)
            if p is not None and now - t.seen.get(p.user_id, -1e9) <= PRESENCE_WINDOW_S
        )
        item = {"id": t.game_id, "hand_no": int(t.hand_no), "present": present}
        if _hand_busy(t):
            busy.append(item)
        elif t.running and present >= 2:
            running.append(item)
    return {
        "hands_in_progress": busy,
        "games_running": running,
        "loaded_tables": len(tables),
        "grader_queue": len(CTX.grade_queued),
    }


#: The ledger check behind the health answers is re-run at most this often (the
#: site's /health is polled by the uptime monitor; money changes by the hand).
LEDGER_CHECK_EVERY_S = 300.0


def _ledger_health(*, fresh: bool = True) -> dict[str, Any]:
    now = time.monotonic()
    if not fresh and CTX.ledger_check["out"] is not None and now - CTX.ledger_check["at"] < LEDGER_CHECK_EVERY_S:
        return CTX.ledger_check["out"]
    try:
        problems = reconcile_ledger()
        out = {"ok": not problems, "problems": len(problems), "first": problems[:10]}
    except Exception as e:  # noqa: BLE001 — health must answer
        out = {"ok": False, "error": str(e)}
    CTX.ledger_check.update(at=now, out=out)
    return out


def _site_health() -> dict[str, Any]:
    """The home games' line in the site's GET /health (``middleware.HEALTH_CHECKS``):
    ok = the clock thread is alive and ticking, the grader (when grading is on) is
    alive, no loaded table is failing or holding unsaved money, and the ledger adds
    up (OPS-011 / OPS-017). Not ok makes the site "degraded", never down."""
    now = time.monotonic()
    th, last = CTX.watchdog_thread, CTX.watchdog_last_tick
    clock_ok = bool(th is not None and th.is_alive() and last and now - last <= WATCHDOG_STALE_S)
    grader_ok = (not GRADING_ON) or bool(CTX.grade_thread is not None and CTX.grade_thread.is_alive())
    with CTX.hub._lock:
        tables = list(CTX.hub._tables.values())
    failing = sorted(t.game_id for t in tables if t.wd_failing_since is not None)
    unsaved = sum(1 for t in tables if t.persist_dirty)
    ledger = _ledger_health(fresh=False)
    return {
        "ok": clock_ok and grader_ok and not failing and not unsaved and bool(ledger.get("ok")),
        "clock": clock_ok, "grader": grader_ok, "tables": len(tables), "failing": failing,
        "unsaved": unsaved, "ledger_ok": bool(ledger.get("ok")), "ledger_problems": ledger.get("problems"),
    }


def _stop_watchdog(timeout: float = 2.0) -> None:
    CTX.watchdog_stop.set()
    th = CTX.watchdog_thread
    if th is not None and th.is_alive() and th is not threading.current_thread():
        th.join(timeout)
    CTX.watchdog_started = False


def shutdown(timeout: float = 2.0) -> None:
    """Stop every worker of the current context — the live streams, the grader,
    the clock (a test's app closes with it, ``server.Site.close``; production goes
    through ``_on_app_shutdown``)."""
    CTX.streams_stop.set()
    CTX.grader_stop.set()
    CTX.bot_stop.set()
    for th in (CTX.grade_thread, CTX.bot_thread):
        if th is not None and th.is_alive() and th is not threading.current_thread():
            th.join(timeout)
    CTX.grade_thread = None
    CTX.bot_thread = None
    _stop_watchdog(timeout)


#: How long a stopping server gives the grader to finish what it holds (the
#: rest waits in homegame_grade_jobs for the next start).
GRADER_DRAIN_S = 3.0


def _on_app_shutdown() -> None:
    """The app's shutdown hook (OPS-008): save every table that still owes the
    database a write, give the grader a moment, then stop the workers.

    (Streams: the server stops them before this runs. uvicorn waits for open
    responses first, so run it with ``--timeout-graceful-shutdown`` — without
    it the live streams hold a stop until the service manager kills it.)"""
    with CTX.hub._lock:
        tables = list(CTX.hub._tables.values())
    for t in tables:
        try:
            with t.lock:
                if t.persist_dirty or t.unsaved:
                    _persist_safe(t)
                    if t.persist_dirty:
                        logger.error("homegame table %s: unsaved at shutdown", t.game_id)
        except Exception:  # noqa: BLE001
            logger.exception("homegame shutdown save (table %s)", t.game_id)
    if GRADING_ON:
        wait_for_grading(GRADER_DRAIN_S)
    shutdown()
    logger.info("homegame workers stopped")


# OPS-010: the live tables are held in THIS process's memory, so exactly one
# process may run the home games on a database — a second uvicorn worker (or a
# staging copy pointed at the live file) would run its own copy of every table
# and overwrite the other's stacks. An exclusive lock on a file next to the
# database, held for the life of the process (the OS drops it when the process
# ends, however it ends), makes a second one refuse to start, loudly.
_PROCESS_LOCKS: dict[str, Any] = {}


def _take_process_lock() -> None:
    path = str(getattr(pub.DB, "path", "") or "")
    if not path or path in _PROCESS_LOCKS:
        return
    f = open(path + ".homegames.lock", "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        f.close()
        raise RuntimeError(
            f"another server process already runs the home games on {path} — the live "
            "tables live in one process's memory, so a second one would overwrite its "
            "stacks. Run ONE process (no --workers) per database."
        ) from e
    _PROCESS_LOCKS[path] = f


# The PLO5 model the grader scores with (HGB-016): server.py provides it
# (``set_model_provider``) — the grader used to import the whole UI server.


def set_model_provider(fn: Any) -> None:
    CTX.model_provider = fn


class NoGradingModel(Exception):
    """No real PLO5 model is being served (OPS-021: a random placeholder never
    grades): a hand's job waits, saved, until one is."""


def _grading_model() -> Any:
    """The model to grade with, or None (none provided, or only a placeholder)."""
    return CTX.model_provider() if CTX.model_provider is not None else None


def _load_table(game_id: str) -> LiveTable:
    row = pub.DB.one("SELECT * FROM homegames WHERE id=?", (game_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="Not Found")
    n = int(row["num_seats"])
    seats: list[Seat | None] = [None] * n
    for p in pub.DB.q(
        "SELECT * FROM homegame_players WHERE game_id=? AND seat IS NOT NULL",
        (game_id,),
    ):
        seat = p["seat"]
        if seat is None or seat < 0 or seat >= n:
            continue
        user = pub._user_by_id(int(p["user_id"]))
        if user is None:
            continue
        seats[int(seat)] = Seat(
            user_id=int(p["user_id"]),
            name=_display_name(user, row["club_id"]),
            stack_chips=int(p["stack_chips"]),
            sitting_out=bool(p["sitting_out"]),
            buyin_cents=int(p["buyin_cents"]),
            leftover_cents=int(p["leftover_cents"]),
            auto_stack_cents=int(p["auto_stack_cents"] or 0),
            time_bank_left=float(row["time_bank_secs"] or 0),
            trusted=bool(p["trusted"]),
            topup_target_cents=int(p["topup_target_cents"] or 0),
            topup_below_cents=int(p["topup_below_cents"] or 0),
        )
    # (Every column is there: ``_ensure_schema`` adds what an older database
    # lacks before anything is loaded — HGB-012.)
    kw = {c.attr: c.from_db(row[c.col]) for c in META}
    # (review 2026-09-20 G14) A table loaded from the DB has no hand in
    # memory (phase "waiting"), but `running=1` used to survive the restart:
    # the host saw only "Pause", nobody saw a deal button, and the game
    # looked dead. A reload always comes back PAUSED — the host presses
    # Start, which deals. (An interrupted hand is void: stacks were last
    # persisted at the previous hand's end.)
    if kw["running"]:
        pub.DB.q("UPDATE homegames SET running=0 WHERE id=?", (game_id,))
    kw.update(running=False, num_seats=n)
    t = LiveTable(
        game_id=row["id"],
        sb_cents=int(row["sb_cents"]),
        bb_cents=int(row["bb_cents"]),
        seats=seats,
        club_id=row["club_id"] or _main_club(),
        phase="waiting",
        variant=_norm_game(row["variant"]),
        **kw,
    )
    # A hand dealt (hand_no is saved at the deal) but never recorded was cut short.
    # It is void — stacks are saved only when a hand ends, so everyone has what they
    # had before it — but say so: to the players it just vanished. (OPS-012: why —
    # the hub gave up on a table nobody had looked at for hours, or a restart.)
    if t.status == "open" and t.hand_no > 0:
        last = pub.DB.one("SELECT MAX(hand_no) AS n FROM homegame_hands WHERE game_id=?", (game_id,))
        if last is not None and int(last["n"] or 0) < t.hand_no:
            why = (
                "nobody was at the table for hours" if CTX.hub.pop_abandoned(game_id) == t.hand_no
                else "the server restarted"
            )
            _emit(t, "run", f"Hand #{t.hand_no} was cut short ({why}) and doesn't count — "
                            "everyone has the chips they had before it. The host restarts the game.")
    return t


@dataclass(frozen=True)
class _MetaCol:
    """One ``homegames`` column that mirrors a ``LiveTable`` attribute the table
    can change during its life (settings + a little running state).

    ``META`` is the ONE list of them (HGB-005): it drives the load
    (``_load_table``), the save (``_persist_meta``), the create
    (``_create_table``'s INSERT), the all-or-nothing rollback of a failed change
    (``_SNAPSHOT_FIELDS``) and the additive migration of older databases
    (``ddl``; ``None`` = a column of the first schema). A new table setting is a
    column in ``_SCHEMA`` + one line here — and a decision about the host's
    remembered settings (``_host_prefs_of``; ``test_every_setting_*`` pins it)."""

    attr: str
    col: str
    to_db: Any    # attribute value -> column value
    from_db: Any  # column value -> attribute value
    ddl: str | None = None


def _db_bool(v: Any) -> int:
    return 1 if v else 0


def _db_int(v: Any) -> int:
    return int(v or 0)


def _db_ms(v: Any) -> int:
    return int(round(float(v or 0.0) * 1000))


META: tuple[_MetaCol, ...] = (
    _MetaCol("host_user_id", "host_user_id", int, int),
    _MetaCol("status", "status", str, str),
    # A reload always comes back PAUSED (see _load_table): ``from_db`` is unused.
    _MetaCol("running", "running", _db_bool, bool, "INTEGER NOT NULL DEFAULT 0"),
    _MetaCol("auto_stack_mode", "auto_stack_mode", lambda v: _norm_auto_mode(v),
             lambda v: _norm_auto_mode(v), "TEXT NOT NULL DEFAULT 'off'"),
    _MetaCol("auto_stack_all_cents", "auto_stack_all_cents", _db_int, int,
             "INTEGER NOT NULL DEFAULT 0"),
    _MetaCol("decision_secs", "decision_secs", _db_int, int, "INTEGER NOT NULL DEFAULT 0"),
    _MetaCol("street_pause_secs", "street_pause_ms", lambda v: _db_ms(v or 1.5),
             lambda ms: max(MIN_STREET_PAUSE_S, int(ms) / 1000.0), "INTEGER NOT NULL DEFAULT 1500"),
    _MetaCol("button", "button", int, int),
    _MetaCol("hand_no", "hand_no", int, int),
    _MetaCol("name", "name", str, str),
    _MetaCol("num_seats", "num_seats", int, int),
    _MetaCol("ante_cents", "ante_cents", int, int),
    _MetaCol("default_buyin_cents", "default_buyin_cents", int, int),
    # 2026-09-21. Tables that predate server-side dealing were auto-dealt by the
    # host's browser, so they migrate to a 5 s delay (not manual).
    _MetaCol("deal_delay_secs", "deal_delay_ms", _db_ms, lambda ms: max(0.0, int(ms) / 1000.0),
             "INTEGER NOT NULL DEFAULT 5000"),
    _MetaCol("time_bank_secs", "time_bank_secs", _db_int, int, "INTEGER NOT NULL DEFAULT 0"),
    _MetaCol("min_buyin_cents", "min_buyin_cents", _db_int, int, "INTEGER NOT NULL DEFAULT 0"),
    _MetaCol("max_buyin_cents", "max_buyin_cents", _db_int, int, "INTEGER NOT NULL DEFAULT 0"),
    _MetaCol("listed", "listed", _db_bool, bool, "INTEGER NOT NULL DEFAULT 1"),
    _MetaCol("allow_rabbit", "allow_rabbit", _db_bool, bool, "INTEGER NOT NULL DEFAULT 1"),
    _MetaCol("approve_buyins", "approve_buyins", _db_bool, bool, "INTEGER NOT NULL DEFAULT 0"),
    _MetaCol("topup_mode", "topup_mode", lambda v: _norm_auto_mode(v),
             lambda v: _norm_auto_mode(v), "TEXT NOT NULL DEFAULT 'off'"),
    _MetaCol("topup_all_target_cents", "topup_target_cents", _db_int, int,
             "INTEGER NOT NULL DEFAULT 0"),
    _MetaCol("topup_all_below_cents", "topup_below_cents", _db_int, int,
             "INTEGER NOT NULL DEFAULT 0"),
    _MetaCol("show_grades", "show_grades", _db_bool, bool, "INTEGER NOT NULL DEFAULT 1"),
    _MetaCol("allow_rathole", "allow_rathole", _db_bool, bool, "INTEGER NOT NULL DEFAULT 0"),
)

_PERSIST_META_SQL = (
    "UPDATE homegames SET " + ", ".join(f"{c.col}=?" for c in META)
    + ", closed_at=CASE WHEN ?='closed' THEN COALESCE(closed_at, ?) ELSE closed_at END"
    " WHERE id=?"
)


def _persist_meta(t: LiveTable) -> None:
    pub.DB.q(
        _PERSIST_META_SQL,
        tuple(c.to_db(getattr(t, c.attr)) for c in META) + (t.status, pub._now(), t.game_id),
    )


def _persist_player(t: LiveTable, seat_i: int | None, p: Seat, seated: bool) -> None:
    pub.DB.q(
        "INSERT INTO homegame_players(game_id,user_id,seat,stack_chips,sitting_out,"
        "buyin_cents,leftover_cents,auto_stack_cents,trusted,topup_target_cents,"
        "topup_below_cents) VALUES(?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(game_id,user_id) DO UPDATE SET seat=excluded.seat,"
        " stack_chips=excluded.stack_chips, sitting_out=excluded.sitting_out,"
        " buyin_cents=excluded.buyin_cents, leftover_cents=excluded.leftover_cents,"
        " auto_stack_cents=excluded.auto_stack_cents, trusted=excluded.trusted,"
        " topup_target_cents=excluded.topup_target_cents,"
        " topup_below_cents=excluded.topup_below_cents",
        (
            t.game_id,
            p.user_id,
            seat_i if seated else None,
            p.stack_chips,
            1 if p.sitting_out else 0,
            p.buyin_cents,
            p.leftover_cents,
            int(p.auto_stack_cents or 0),
            1 if p.trusted else 0,
            int(p.topup_target_cents or 0),
            int(p.topup_below_cents or 0),
        ),
    )


def _persist_seats(t: LiveTable) -> None:
    with pub.DB.transaction():  # one commit, not one per row
        _persist_meta(t)
        for i, p in enumerate(t.seats):
            if p is not None:
                _persist_player(t, i, p, True)


@contextmanager
def _savepoint(name: str) -> Iterator[None]:
    """A nested all-or-nothing block INSIDE ``pub.DB.transaction()``: on an
    exception only its own statements are undone, the transaction goes on."""
    pub.DB.q(f"SAVEPOINT {name}")
    try:
        yield
    except BaseException:
        pub.DB.q(f"ROLLBACK TO {name}")
        pub.DB.q(f"RELEASE {name}")
        raise
    pub.DB.q(f"RELEASE {name}")


def _persist_safe(t: LiveTable) -> None:
    """Persist meta + seats on an ENGINE path — together with every write the
    table still owes the database (``t.unsaved``) — in ONE transaction; never
    raises.

    (review 2026-09-20 G4) Once the engine has stepped, memory cannot be
    rolled back — the env has no undo — so a failed write must not surface
    as a half-applied action. Memory stays the truth, ``persist_dirty`` is
    set, and the watchdog retries — the owed writes with it. (OPS-006: a
    hand's record and results used to be committed apart from the stacks it
    moved, so a crash in between left stats and money disagreeing; OPS-013: a
    shuffle transcript that failed to save was only logged.) Each owed write
    runs in its own SAVEPOINT: if IT fails, it alone is dropped (logged) — it
    can never block the stacks. (Inputs are capped, so every in-memory value
    IS persistable; only transient DB errors land here.)"""
    _assert_lock_held(t)
    owed = list(t.unsaved)
    written: list = []
    try:
        with pub.DB.transaction():
            _persist_seats(t)
            for w in owed:
                try:
                    with _savepoint("owed_write"):
                        _write_owed(t, w)
                    written.append(w)
                except Exception:  # noqa: BLE001
                    logger.exception("homegame: an owed %s write was dropped (table %s)", w[0], t.game_id)
        t.persist_dirty = False
        del t.unsaved[: len(owed)]  # (callers hold t.lock: nothing else touched the list)
    except Exception:  # noqa: BLE001
        logger.exception("homegame persist failed for %s; will retry", t.game_id)
        t.persist_dirty = True
        t.persist_retry_mono = time.monotonic() + 2.0
        return
    for w in written:
        if w[0] == "hand" and w[1].get("job") is not None:
            _queue_grading(w[1]["job"])


def _write_owed(t: LiveTable, w: tuple) -> None:
    if w[0] == "fair":
        _, hand_no, data = w
        pub.DB.q(
            "INSERT OR REPLACE INTO homegame_fair(game_id,hand_no,data) VALUES(?,?,?)",
            (t.game_id, int(hand_no), data),
        )
        return
    b = w[1]
    rec = b["record"]
    pub.DB.q(
        "INSERT OR REPLACE INTO homegame_hands(game_id,hand_no,ended_at,pot_cents,summary)"
        " VALUES(?,?,?,?,?)",
        (t.game_id, int(rec["hand_no"]), rec["ended_at"], int(rec["pot_cents"]),
         json.dumps(rec, separators=(",", ":"))),
    )
    for r in b["results"]:
        pub.DB.q(
            "INSERT OR REPLACE INTO homegame_hand_results(game_id,hand_no,user_id,delta_cents,"
            "delta_chips,showdown) VALUES(?,?,?,?,?,?)",
            (t.game_id, int(rec["hand_no"]), int(r["user_id"]), int(r["delta_cents"]),
             int(r["delta_chips"]), 1 if r["shown"] else 0),
        )
    for payer, payee, chips in b["flows"]:
        pub.DB.q(
            "INSERT OR REPLACE INTO homegame_flows(game_id,hand_no,payer,payee,chips)"
            " VALUES(?,?,?,?,?)",
            (t.game_id, int(rec["hand_no"]), int(payer), int(payee), int(chips)),
        )
    job = b.get("job")
    if job is not None and job.get("deck"):
        # (a deck, never a seed: the seed is never persisted. The deck is the
        # same secret-at-rest as the shuffle transcript in homegame_fair.)
        pub.DB.q(
            "INSERT OR REPLACE INTO homegame_grade_jobs(game_id,hand_no,job,created_at,attempts)"
            " VALUES(?,?,?,?,0)",
            (t.game_id, int(rec["hand_no"]), json.dumps(job, separators=(",", ":")), pub._now()),
        )


def _retry_persist_locked(t: LiveTable) -> None:
    if t.persist_dirty and time.monotonic() >= t.persist_retry_mono:
        _persist_safe(t)


# What ``_mutation`` rolls back: every saved setting (derived from META, so a new
# one can never be left out — OPS-003: ``allow_rathole`` was, and a failed save
# left it switched on in memory), plus the in-memory state a change may touch
# (a resize drops the finished hand's per-seat arrays — rolled back by
# reference: they are always replaced, never edited in place).
_SNAPSHOT_FIELDS = tuple(c.attr for c in META) + ("phase", "next_deal_mono")


@contextmanager
def _mutation(t: LiveTable) -> Iterator[None]:
    """All-or-nothing seat/settings change (caller holds ``t.lock``).

    (review 2026-09-20 G4) Handlers used to mutate memory first and persist
    second: a failed write (``rebuy 1e30`` -> sqlite OverflowError) left RAM
    holding a value that could never be saved, and every later
    deal/kick/leave/close 500'd until a restart. Inside this block the DB
    writes form ONE transaction, and on any exception both the transaction
    and the in-memory table (seats + settings) roll back together.

    Never step the engine inside it — an env cannot be restored; engine
    paths use ``_persist_safe`` instead."""
    _assert_lock_held(t)
    snap = {f: getattr(t, f) for f in _SNAPSHOT_FIELDS}
    hand = replace(t.hand)  # (a copy: a change may set its fields, or replace it — resize)
    seats = [replace(p) if p is not None else None for p in t.seats]
    kicks = set(t.pending_kicks)
    try:
        with pub.DB.transaction():
            yield
    except BaseException:
        for f, v in snap.items():
            setattr(t, f, v)
        t.hand = hand
        t.seats = seats
        t.pending_kicks = kicks
        raise
    t.rev += 1


# --- the money ledger (OPS-017) -------------------------------------------------------------
# ``homegame_ledger`` = every movement of money between a player and a table, with
# its time and the hand it came after. The player's running totals
# (``homegame_players.buyin_cents`` / ``leftover_cents``) must equal the sums of
# these rows — ``reconcile_ledger`` checks it (at start and in /games/api/health).
#: Money IN — each adds to the player's ``buyin_cents``:
LEDGER_IN = ("buyin", "rebuy", "topup", "auto_topup", "stack_in")
#: Money OUT — each adds to ``leftover_cents``:
LEDGER_OUT = ("cashout", "take_off", "stack_out")
#: What a receipt calls each kind. Rows from before OPS-017 (``hand_no`` NULL) used
#: three kinds only: "rebuy" for every chip in after the first buy-in and "cashout"
#: for every chip out — the same direction as the specific kinds, so the sums hold.
LEDGER_LABELS = {
    "buyin": "Bought in",
    "rebuy": "Sat down again",
    "topup": "Added chips",
    "auto_topup": "Auto top-up",
    "stack_in": "Set stack: topped up",
    "cashout": "Cashed out",
    "take_off": "Took chips off the table",
    "stack_out": "Set stack: extra chips back",
}
_LEGACY_LEDGER_LABELS = {"buyin": "Bought in", "rebuy": "Chips in", "cashout": "Chips out"}


def _ledger_add(t: LiveTable, uid: int, kind: str, amount_cents: int) -> None:
    """One money movement (inside the caller's transaction, with the player's new
    totals). ``hand_no`` = the last hand dealt when it happened (0: before the
    first) — a mid-hand buy-in by a seat out of the hand names the hand in play."""
    if kind not in LEDGER_IN and kind not in LEDGER_OUT:
        raise ValueError(f"unknown ledger kind {kind!r}")
    pub.DB.q(
        "INSERT INTO homegame_ledger(game_id,user_id,kind,amount_cents,created_at,hand_no)"
        " VALUES(?,?,?,?,?,?)",
        (t.game_id, uid, kind, int(amount_cents), pub._now(), int(t.hand_no)),
    )


def reconcile_ledger(game_id: str | None = None) -> list[dict[str, Any]]:
    """Every place the money does not add up (OPS-017), [] when it all does:

    - a player whose ledger rows do not sum to their running totals (money in =
      ``buyin_cents``, money out = ``leftover_cents``);
    - a CLOSED table that still holds chips, or whose money in and out differ
      (it must be exactly zero-sum once everyone has cashed out);
    - a row of a kind that is neither money in nor money out.
    Read-only; ``game_id`` = one table (None = every table)."""
    ins = ",".join("?" * len(LEDGER_IN))
    outs = ",".join("?" * len(LEDGER_OUT))
    where, args = ("WHERE p.game_id=?", [game_id]) if game_id else ("", [])
    problems: list[dict[str, Any]] = []
    for r in pub.DB.q(
        "SELECT p.game_id, p.user_id, p.buyin_cents, p.leftover_cents, "
        "COALESCE((SELECT SUM(l.amount_cents) FROM homegame_ledger l WHERE l.game_id=p.game_id "
        f"AND l.user_id=p.user_id AND l.kind IN ({ins})),0) AS lin, "
        "COALESCE((SELECT SUM(l.amount_cents) FROM homegame_ledger l WHERE l.game_id=p.game_id "
        f"AND l.user_id=p.user_id AND l.kind IN ({outs})),0) AS lout "
        f"FROM homegame_players p {where}",
        tuple(list(LEDGER_IN) + list(LEDGER_OUT) + args),
    ):
        if (int(r["lin"]), int(r["lout"])) != (int(r["buyin_cents"]), int(r["leftover_cents"])):
            problems.append({
                "table": r["game_id"], "user_id": int(r["user_id"]), "problem": "totals",
                "buyin_cents": int(r["buyin_cents"]), "ledger_in_cents": int(r["lin"]),
                "leftover_cents": int(r["leftover_cents"]), "ledger_out_cents": int(r["lout"]),
            })
    gwhere, gargs = ("AND g.id=?", [game_id]) if game_id else ("", [])
    for r in pub.DB.q(
        "SELECT g.id, COALESCE(SUM(p.buyin_cents),0) b, COALESCE(SUM(p.leftover_cents),0) l, "
        "COALESCE(SUM(p.stack_chips),0) s FROM homegames g JOIN homegame_players p ON p.game_id=g.id "
        f"WHERE g.status='closed' {gwhere} GROUP BY g.id HAVING b<>l OR s<>0", tuple(gargs),
    ):
        problems.append({"table": r["id"], "problem": "closed table not zero-sum",
                         "in_cents": int(r["b"]), "out_cents": int(r["l"]), "chips_left": int(r["s"])})
    kinds = LEDGER_IN + LEDGER_OUT
    kwhere, kargs = ("AND game_id=?", [game_id]) if game_id else ("", [])
    for r in pub.DB.q(
        f"SELECT game_id, kind, COUNT(*) n FROM homegame_ledger WHERE kind NOT IN ({','.join('?' * len(kinds))}) "
        f"{kwhere} GROUP BY game_id, kind", tuple(list(kinds) + kargs),
    ):
        problems.append({"table": r["game_id"], "problem": f"unknown kind {r['kind']!r}", "rows": int(r["n"])})
    return problems


def _check_ledger_at_start() -> None:
    """Startup reconciliation (OPS-017): loud in the log, never fatal."""
    try:
        problems = reconcile_ledger()
    except Exception:  # noqa: BLE001
        logger.exception("homegame ledger reconciliation failed to run")
        return
    for p in problems[:20]:
        logger.error("homegame ledger does not add up: %s", p)
    if problems:
        logger.error("homegame ledger: %d problem(s) — see /games/api/health", len(problems))


# --- settling up (FEAT-001) ----------------------------------------------------------------
#: Players with money to settle up to which the fewest-payments search is exact
#: (2^n subsets); a bigger session is settled greedily (still at most n - 1 payments).
SETTLE_EXACT_MAX = 12


@functools.lru_cache(maxsize=512)
def _settle_plan(nets: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int, int], ...]:
    """(payer, payee, cents) payments that settle ``nets`` ((key, cents) pairs summing
    to 0) in the FEWEST payments: the players are split into as many groups that
    settle among themselves as possible (each group of k needs k - 1 payments, so
    that is the minimum), then each group pays largest debt to largest credit.
    Deterministic: the same nets always give the same payments."""
    items = sorted(((k, c) for k, c in nets if c), key=lambda x: (-abs(x[1]), x[0]))
    if sum(c for _, c in items) != 0:
        raise ValueError("nets must sum to zero")
    groups = _zero_sum_groups(items) if len(items) <= SETTLE_EXACT_MAX else [items]
    out: list[tuple[int, int, int]] = []
    for g in groups:
        debt = [[-c, k] for k, c in g if c < 0]
        cred = [[c, k] for k, c in g if c > 0]
        i = j = 0
        while i < len(debt) and j < len(cred):
            amt = min(debt[i][0], cred[j][0])
            out.append((debt[i][1], cred[j][1], amt))
            debt[i][0] -= amt
            cred[j][0] -= amt
            if not debt[i][0]:
                i += 1
            if not cred[j][0]:
                j += 1
    out.sort(key=lambda x: (-x[2], x[0], x[1]))
    return tuple(out)


def _zero_sum_groups(items: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Split ``items`` (summing to 0) into the most groups that each sum to 0:
    best[m] = max over i in m of best[m - i], +1 when m itself sums to 0 — the
    best ORDER of the players, cut wherever its running total is back at 0."""
    n = len(items)
    vals = [c for _, c in items]
    size = 1 << n
    sums = [0] * size
    best = [0] * size
    last = [0] * size
    for m in range(1, size):
        low = m & -m
        sums[m] = sums[m ^ low] + vals[low.bit_length() - 1]
        b, ch, mm = -1, 0, m
        while mm:
            bit = mm & -mm
            mm ^= bit
            v = best[m ^ bit]
            if v > b:
                b, ch = v, bit.bit_length() - 1
        best[m] = b + (1 if sums[m] == 0 else 0)
        last[m] = ch
    order: list[int] = []
    m = size - 1
    while m:
        order.append(last[m])
        m ^= 1 << last[m]
    order.reverse()
    groups: list[list[tuple[int, int]]] = []
    cur: list[tuple[int, int]] = []
    run = 0
    for i in order:
        cur.append(items[i])
        run += vals[i]
        if run == 0:
            groups.append(cur)
            cur = []
    if cur:  # (cannot happen: the whole set sums to 0)
        groups.append(cur)
    return groups


def settle_up(rows: list[dict[str, Any]], viewer_id: int | None = None) -> list[dict[str, Any]]:
    """Who pays whom to settle ``rows`` (ledger rows: user_id, name, net_cents —
    they sum to exactly 0), in the fewest payments. ``you`` = "pay" / "get" on the
    viewer's own lines. Rows that do not sum to zero (money that cannot exist —
    ``reconcile_ledger`` is where that is reported) settle nothing: a view never
    fails over it."""
    names = {int(r["user_id"]): r["name"] for r in rows}
    nets = tuple(sorted((int(r["user_id"]), int(r["net_cents"])) for r in rows))
    if sum(c for _, c in nets) != 0:
        return []
    plan = _settle_plan(nets)
    return [
        {"from": a, "from_name": names.get(a, "?"), "to": b, "to_name": names.get(b, "?"), "cents": c,
         "you": "pay" if viewer_id == a else "get" if viewer_id == b else None}
        for a, b, c in plan
    ]


def _emit(t: LiveTable, kind: str, text: str, **extra: Any) -> None:
    """Append one line to the table's in-memory event feed (dealer messages,
    toasts). Cosmetic: never raises, never persisted, capped."""
    try:
        t.event_seq += 1
        ev = {"id": t.event_seq, "kind": kind, "text": str(text)[:200], "ts": time.time()}
        ev.update(extra)
        t.events.append(ev)
    except Exception:  # noqa: BLE001
        logger.exception("homegame event feed")


def _user_name(t: LiveTable, uid: Any) -> str:
    """A user's display name at this table: seated, seen here, or looked up."""
    uid = int(uid)
    p = t.player(uid)
    if p is not None:
        return p.name
    if uid in t.names:
        return t.names[uid]
    u = pub._user_by_id(uid)
    return _display_name(u, t.club_id) if u is not None else f"Player {uid}"


def _seat_name(t: LiveTable, i: int | None) -> str:
    p = t.seats[i] if i is not None and 0 <= i < len(t.seats) else None
    return p.name if p is not None else (f"Seat {i + 1}" if i is not None else "?")


def _eligible_mask(t: LiveTable, stacks: list[int] | None = None) -> list[bool]:
    """Seats that would be dealt in. ``stacks`` overrides the seats' own
    chips (the view passes hand-start stacks while a runout is revealing)."""
    ante = t.ante_chips
    out = []
    for i, s in enumerate(t.seats):
        chips = s.stack_chips if s is not None else 0
        if stacks is not None and i < len(stacks):
            chips = int(stacks[i])
        out.append(s is not None and (not s.sitting_out) and chips > ante)
    return out


def _validate_auto_stack_cents(t: LiveTable, cents: int) -> int:
    n = int(cents)
    if n < 0:
        raise HTTPException(status_code=400, detail="auto-stack must be >= 0")
    if n == 0:
        return 0
    if n <= int(t.ante_cents) or cents_to_chips(n, t.bb_cents) <= t.ante_chips:
        raise HTTPException(
            status_code=400,
            detail="auto-stack must be more than the ante",
        )
    hi = int(t.max_buyin_cents or 0)
    if hi and n > hi:
        raise HTTPException(
            status_code=400, detail=f"the table maximum is {_fmt_cents(hi)}"
        )
    return n


def _validate_topup(t: LiveTable, target: int, below: int) -> tuple[int, int]:
    """(target, below) for auto top-up. target 0 = off. below 0 = "whenever the
    stack is under the target"; otherwise 0 < below <= target."""
    target = _validate_auto_stack_cents(t, target)
    below = int(below)
    if target == 0:
        return 0, 0
    if below < 0 or below > target:
        raise HTTPException(
            status_code=400,
            detail="the top-up threshold must be between 0 and the target",
        )
    return target, below


def _auto_cap(t: LiveTable, cents: int) -> int:
    """An automatic-chips target as it applies NOW: never above the table's
    current maximum buy-in (OPS-004: a target set under a higher maximum kept
    refilling stacks past a maximum the host had since lowered)."""
    hi = int(t.max_buyin_cents or 0)
    return min(int(cents), hi) if hi and cents else int(cents)


def _cap_auto_chips_locked(t: LiveTable) -> list[str]:
    """The maximum buy-in went down: every stored automatic-chips target above
    it comes down to it (and a top-up threshold with its target), so what the
    host and the players see is what will happen. Caller is inside
    ``_mutation`` and persists the seats. Returns the players whose own targets
    moved (said to the table)."""
    def capped(target: int, below: int) -> tuple[int, int]:
        target = _auto_cap(t, target)
        return target, min(int(below), target)

    t.auto_stack_all_cents = _auto_cap(t, int(t.auto_stack_all_cents or 0))
    t.topup_all_target_cents, t.topup_all_below_cents = capped(
        int(t.topup_all_target_cents or 0), int(t.topup_all_below_cents or 0))
    moved: list[str] = []
    for p in t.seats:
        if p is None:
            continue
        before = (p.auto_stack_cents, p.topup_target_cents, p.topup_below_cents)
        p.auto_stack_cents = _auto_cap(t, int(p.auto_stack_cents or 0))
        p.topup_target_cents, p.topup_below_cents = capped(
            int(p.topup_target_cents or 0), int(p.topup_below_cents or 0))
        if (p.auto_stack_cents, p.topup_target_cents, p.topup_below_cents) != before:
            moved.append(p.name)
    return moved


def _auto_chips_allowed(t: LiveTable, p: Seat) -> bool:
    """An automatic buy-in is still a buy-in: while the host approves buy-ins
    it only runs for the host and the players the host trusts."""
    return (not t.approve_buyins) or p.trusted or p.user_id == t.host_user_id


def _auto_target_chips(t: LiveTable, p: Seat) -> int:
    """The chips seat ``p`` will hold once the deal's auto set-stack / auto
    top-up has run (its current chips when neither applies)."""
    chips = int(p.stack_chips) + cents_to_chips(int(p.queued_topup_cents or 0), t.bb_cents)
    if not _auto_chips_allowed(t, p):
        return chips
    if _norm_auto_mode(t.auto_stack_mode) != "off" and int(p.auto_stack_cents or 0) > 0:
        return cents_to_chips(_auto_cap(t, int(p.auto_stack_cents)), t.bb_cents)
    target = _auto_cap(t, int(p.topup_target_cents or 0))
    if _norm_auto_mode(t.topup_mode) != "off" and target > 0:
        below = min(int(p.topup_below_cents or 0), target) or target
        if chips_to_cents(chips, t.bb_cents) < below:
            return max(chips, cents_to_chips(target, t.bb_cents))
    return chips


def _apply_auto_stacks_locked(t: LiveTable) -> None:
    """Reset opted-in stacks to their target, then the engine posts ante.

    Surplus chips cash out to leftover; a shortfall is a rebuy. Net is
    unchanged. Sitting-out seats are left alone until they sit back in.

    (review 2026-09-20 G10) Money only moves in whole cents, and the chips
    that move are exactly those cents' worth: a stack's sub-cent remainder
    (pots split across boards / ties) is CARRIED, so the stack lands within
    a cent of the target. It used to be set to the target outright, which
    created or destroyed the remainder while the ledger moved a rounded
    amount.
    """
    set_on = _norm_auto_mode(t.auto_stack_mode) in ("host", "player")
    top_on = _norm_auto_mode(t.topup_mode) in ("host", "player")
    if not (set_on or top_on):
        return
    with _mutation(t):
        for p in t.seats:
            if p is None or p.sitting_out or not _auto_chips_allowed(t, p):
                continue
            target_cents = _auto_cap(t, int(p.auto_stack_cents or 0)) if set_on else 0
            if target_cents <= 0:
                # AUTO TOP-UP (2026-09-22): only ever UP, and only once the
                # stack has dropped below the player's threshold — winnings
                # stay on the table, so this is not a rathole.
                tgt = _auto_cap(t, int(p.topup_target_cents or 0)) if top_on else 0
                if tgt <= 0:
                    continue
                have = chips_to_cents(int(p.stack_chips), t.bb_cents)
                if have >= (min(int(p.topup_below_cents or 0), tgt) or tgt):
                    continue
                add_cents = tgt - have
                if add_cents <= 0 or p.buyin_cents + add_cents > MAX_CENTS:
                    continue
                p.buyin_cents += add_cents
                p.stack_chips += cents_to_chips(add_cents, t.bb_cents)
                _ledger_add(t, p.user_id, "auto_topup", add_cents)
                _emit(t, "rebuy", f"{p.name} auto topped up {_fmt_cents(add_cents)}")
                continue
            target_chips = cents_to_chips(target_cents, t.bb_cents)
            if target_chips <= t.ante_chips:
                continue
            delta_chips = target_chips - int(p.stack_chips)
            delta_cents = chips_to_cents(abs(delta_chips), t.bb_cents)
            if delta_cents <= 0:
                continue  # within a cent of the target already
            moved = cents_to_chips(delta_cents, t.bb_cents)
            if delta_chips > 0:
                if p.buyin_cents + delta_cents > MAX_CENTS:
                    continue  # total buy-in cap — leave the stack alone
                p.buyin_cents += delta_cents
                p.stack_chips += moved
                _ledger_add(t, p.user_id, "stack_in", delta_cents)
            else:
                moved = min(moved, int(p.stack_chips))
                p.leftover_cents += delta_cents
                p.stack_chips -= moved
                _ledger_add(t, p.user_id, "stack_out", delta_cents)
        _persist_seats(t)


# Said to the table when the host switches a mode: "players choose" is only
# useful if the players hear about it (they set theirs from their own seat).
_AUTO_MODE_LINES = {
    "topup": {
        "player": "players now set their own auto top-up (tap your seat)",
        "host": "the host now sets auto top-up",
        "off": "auto top-up is off",
    },
    "set": {
        "player": "players now choose their own stack for every hand (tap your seat)",
        "host": "the host now sets the stack for every hand",
        "off": "set stack is off",
    },
}


def _auto_stack_host_locked(t: LiveTable, uid: int, body: dict) -> None:
    _require_open(t)
    _require_host(t, uid, "change auto-stack settings")
    body = body or {}
    old_mode = t.auto_stack_mode
    with _mutation(t):
        if "mode" in body and body["mode"] is not None:
            mode = str(body["mode"]).strip().lower()
            if mode not in AUTO_STACK_MODES:
                raise HTTPException(
                    status_code=400, detail="mode must be off, host, or player"
                )
            t.auto_stack_mode = mode
        if "all_cents" in body and body["all_cents"] is not None:
            cents = _validate_auto_stack_cents(t, _parse_cents(body, "all_cents", 0))
            t.auto_stack_all_cents = cents
            if t.auto_stack_mode == "host":
                for p in t.seats:
                    if p is not None:
                        p.auto_stack_cents = cents
        players = body.get("players")
        if players:
            if t.auto_stack_mode != "host":
                raise HTTPException(
                    status_code=400,
                    detail="per-player auto-stack is only available when the host sets stacks",
                )
            if not isinstance(players, list) or len(players) > TABLE_SEATS:
                raise HTTPException(status_code=400, detail="players must be a list")
            for item in players:
                if not isinstance(item, dict):
                    raise HTTPException(status_code=400, detail="invalid player entry")
                puid = pub.body_int(item, "user_id")
                pcents = _validate_auto_stack_cents(t, _parse_cents(item, "cents", 0))
                p = t.player(puid)
                if p is None:
                    raise HTTPException(status_code=400, detail="player not seated")
                p.auto_stack_cents = pcents
        _persist_seats(t)
    if t.auto_stack_mode != old_mode:
        _emit(t, "settings", f"Host: {_AUTO_MODE_LINES['set'][_norm_auto_mode(t.auto_stack_mode)]}")
    _remember_host_prefs(t)


def _auto_topup_host_locked(t: LiveTable, uid: int, body: dict) -> None:
    """Host side of auto top-up: the mode (off / host / player), the values for
    everyone in host mode, and per-player overrides."""
    _require_open(t)
    _require_host(t, uid, "change auto top-up settings")
    body = body or {}
    old_mode = t.topup_mode
    with _mutation(t):
        if body.get("mode") is not None:
            mode = str(body["mode"]).strip().lower()
            if mode not in AUTO_STACK_MODES:
                raise HTTPException(
                    status_code=400, detail="mode must be off, host, or player"
                )
            t.topup_mode = mode
        if body.get("all_target_cents") is not None:
            target, below = _validate_topup(
                t, _parse_cents(body, "all_target_cents", 0),
                _parse_cents(body, "all_below_cents", 0),
            )
            t.topup_all_target_cents, t.topup_all_below_cents = target, below
            if t.topup_mode == "host":
                for p in t.seats:
                    if p is not None:
                        p.topup_target_cents, p.topup_below_cents = target, below
        players = body.get("players")
        if players:
            if t.topup_mode != "host":
                raise HTTPException(
                    status_code=400,
                    detail="per-player top-up is only available when the host sets it",
                )
            if not isinstance(players, list) or len(players) > TABLE_SEATS:
                raise HTTPException(status_code=400, detail="players must be a list")
            for item in players:
                if not isinstance(item, dict):
                    raise HTTPException(status_code=400, detail="invalid player entry")
                p = t.player(pub.body_int(item, "user_id"))
                if p is None:
                    raise HTTPException(status_code=400, detail="player not seated")
                p.topup_target_cents, p.topup_below_cents = _validate_topup(
                    t, _parse_cents(item, "target_cents", 0),
                    _parse_cents(item, "below_cents", 0),
                )
        _persist_seats(t)
    if t.topup_mode != old_mode:
        _emit(t, "settings", f"Host: {_AUTO_MODE_LINES['topup'][_norm_auto_mode(t.topup_mode)]}")
    _remember_host_prefs(t)


def _auto_chips_self_locked(t: LiveTable, uid: int, body: dict) -> None:
    """A player picks their own automatic chips — when the host left the
    choice to the players: ``kind`` = off | topup | set."""
    _require_open(t)
    p = t.player(uid)
    if p is None:
        raise HTTPException(status_code=400, detail="not seated")
    kind = str((body or {}).get("kind") or "off").strip().lower()
    if kind not in ("off", "topup", "set"):
        raise HTTPException(status_code=400, detail="kind must be off, topup or set")
    if kind == "topup" and t.topup_mode != "player":
        raise HTTPException(
            status_code=400, detail="players cannot set auto top-up at this table"
        )
    if kind == "set" and t.auto_stack_mode != "player":
        raise HTTPException(
            status_code=400, detail="players cannot set their stack at this table"
        )
    target = _parse_cents(body or {}, "target_cents", 0)
    below = _parse_cents(body or {}, "below_cents", 0)
    with _mutation(t):
        # Only the knobs the PLAYER owns are touched: a host-set value stays.
        if t.topup_mode == "player":
            p.topup_target_cents, p.topup_below_cents = (
                _validate_topup(t, target, below) if kind == "topup" else (0, 0)
            )
        if t.auto_stack_mode == "player":
            p.auto_stack_cents = (
                _validate_auto_stack_cents(t, target) if kind == "set" else 0
            )
        _persist_player(t, t.seat_of(uid), p, True)


def _next_button(t: LiveTable, mask: list[bool]) -> int:
    n = t.num_seats
    start = t.button
    if t.hand_no == 0:
        # First hand: first eligible seat at or after 0.
        for i in range(n):
            if mask[i]:
                return i
        return 0
    for off in range(1, n + 1):
        i = (start + off) % n
        if mask[i]:
            return i
    return start


def _split_board(cards: list[int]) -> dict[str, list[int | None]]:
    c = [int(x) for x in cards]
    flop = (c[:3] + [None, None, None])[:3]
    turn = c[3] if len(c) > 3 else None
    river = c[4] if len(c) > 4 else None
    return {"flop": flop, "turn": turn, "river": river}


def _history_entries(raw: dict[str, Any], bb_cents: int) -> list[dict[str, Any]]:
    street_commit = [0] * TABLE_SEATS
    cur_street: int | None = None
    out: list[dict[str, Any]] = []
    for rec in raw.get("history") or []:
        seat, action, chips, street = int(rec[0]), int(rec[1]), int(rec[2]), int(rec[3])
        if street != cur_street:
            # (review 2026-09-20 G9) "Raise to" is a PER-STREET total: the
            # accumulator used to run across streets, so a $5 turn bet after
            # $10 of flop action read "Raise to $15".
            street_commit = [0] * TABLE_SEATS
            cur_street = street
        street_commit[seat] += chips
        if action == FOLD:
            label = "Fold"
        elif action == CHECK_CALL:
            label = "Check" if chips == 0 else f"Call {_fmt_cents(chips_to_cents(chips, bb_cents))}"
        else:
            label = f"Raise to {_fmt_cents(chips_to_cents(street_commit[seat], bb_cents))}"
            if action == ALL_IN:
                label = f"All-in {_fmt_cents(chips_to_cents(chips, bb_cents))}"
        out.append({
            "seat": seat,
            "street": STREET_NAMES.get(street, str(street)),
            "action": action,
            "chips": chips,
            "cents": chips_to_cents(chips, bb_cents),
            "to_cents": chips_to_cents(street_commit[seat], bb_cents),
            "label": label,
        })
    return out


def _ledger_state(
    t: LiveTable, stacks: list[int] | None = None
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """(ledger rows, {seat: stack cents}) under the zero-sum money rule.

    ``stacks`` overrides the seated players' chips (hand-start stacks while a
    runout is still revealing, so the rows cannot spoil it — review G11).
    The seated stacks' cents are the largest-remainder apportionment of the
    money on the table (module docstring, review G10): ``sum(net_cents)`` over
    the returned rows is exactly 0.

    (review 2026-09-20 G12) No emails: every flagged user can open every
    table, and the row used to hand each of them everyone's address."""
    rows = pub.DB.q(
        "SELECT p.user_id, p.buyin_cents, p.leftover_cents, u.email, u.name "
        "FROM homegame_players p JOIN users u ON u.id=p.user_id "
        "WHERE p.game_id=? ORDER BY p.user_id",
        (t.game_id,),
    )
    seated = {s.user_id: (i, s) for i, s in enumerate(t.seats) if s is not None}
    entries: list[dict[str, Any]] = []
    seen: set[int] = set()
    for r in rows:
        uid = int(r["user_id"])
        seen.add(uid)
        live = seated.get(uid)
        entries.append({
            "user_id": uid,
            "name": live[1].name if live else _display_name(r, t.club_id),
            "seat": live[0] if live else None,
            "buyin": live[1].buyin_cents if live else int(r["buyin_cents"]),
            "leftover": live[1].leftover_cents if live else int(r["leftover_cents"]),
        })
    for uid, (i, s) in seated.items():  # seated but not persisted yet
        if uid not in seen:
            entries.append({
                "user_id": uid, "name": s.name, "seat": i,
                "buyin": s.buyin_cents, "leftover": s.leftover_cents,
            })
    money = sum(e["buyin"] - e["leftover"] for e in entries)
    live_entries = [e for e in entries if e["seat"] is not None]
    live_entries.sort(key=lambda e: e["seat"])
    chips = []
    for e in live_entries:
        i = e["seat"]
        if stacks is not None and i < len(stacks):
            chips.append(int(stacks[i]))
        else:
            chips.append(int(t.seats[i].stack_chips))
    cents = apportion_cents(money, chips)
    seat_cents = {e["seat"]: c for e, c in zip(live_entries, cents)}
    out = []
    for e in entries:
        stack_cents = seat_cents.get(e["seat"], 0) if e["seat"] is not None else 0
        out.append({
            "user_id": e["user_id"],
            "name": e["name"],
            "seated": e["seat"] is not None,
            "seat": e["seat"],
            "buyin_cents": e["buyin"],
            "stack_cents": stack_cents,
            "leftover_cents": e["leftover"],
            "net_cents": e["leftover"] + stack_cents - e["buyin"],
        })
    out.sort(key=lambda x: (not x["seated"], x["seat"] is None, x["seat"] or 0))
    return out, seat_cents


def _turn_remaining_secs(t: LiveTable) -> float | None:
    if t.phase != "in_hand" or int(t.decision_secs or 0) <= 0:
        return None
    if t.turn_started_mono is None:
        return None
    left = float(t.decision_secs) - (time.monotonic() - t.turn_started_mono)
    return max(0.0, round(left, 2))


def _passive_choice(t: LiveTable) -> tuple[int, int]:
    """Check if free, otherwise fold. Used for the shot clock and away."""
    if t.env is None or t.info is None:
        raise HTTPException(status_code=400, detail="no hand in progress")
    gm = t.info.gate_mask
    raw = _obs_dict(t.env)
    actor = raw.get("actor")
    if actor is None:
        raise HTTPException(status_code=400, detail="no actor")
    actor = int(actor)
    to_call = min(
        max(0, int(raw["bet_to_call"]) - int(raw["street_commit"][actor])),
        int(raw["stacks"][actor]),
    )
    if to_call <= 0 and bool(gm[GATE_CHECK_CALL]):
        return GATE_CHECK_CALL, 0
    if bool(gm[GATE_FOLD]):
        return GATE_FOLD, 0
    if bool(gm[GATE_CHECK_CALL]):
        return GATE_CHECK_CALL, 0
    raise HTTPException(status_code=400, detail="no legal auto-action")


def _actor_is_away(t: LiveTable, actor: int) -> bool:
    """The seat to act has nobody who will act: sitting out (Away, or being
    removed), vacated — or leaving after this hand with the browser already
    gone (a leaver plays the hand out as normal while they are here; once they
    have closed the tab the clock must not wait on them, review G6)."""
    p = t.seats[actor] if 0 <= actor < len(t.seats) else None
    if p is None or bool(p.sitting_out):
        return True
    if p.leave_after_hand:
        return time.monotonic() - t.seen.get(p.user_id, -1e9) > PRESENCE_WINDOW_S
    return False


def _start_clock_locked(t: LiveTable, *, restart: bool = False) -> None:
    """Run the shot clock for the CURRENT decision.

    (review 2026-09-20 G5) A decision is ``(hand_no, action_seq)``. The clock
    restarts only when that changes — it used to restart on every call, and
    any seated player toggling Away (or a kick) called it, handing the actor
    a fresh clock as often as a friend cared to click."""
    key = (t.hand_no, t.action_seq)
    if int(t.decision_secs or 0) <= 0:
        t.turn_started_mono = None
        t.turn_key = key
        return
    if not restart and t.turn_key == key and t.turn_started_mono is not None:
        return
    t.turn_key = key
    t.turn_started_mono = time.monotonic()


def _resume_turn_locked(t: LiveTable) -> None:
    """Start the shot clock, or immediately auto-act an away actor.

    Passive play is at most one action per seat per street, so the bound
    below always drains a hand full of away players (review 2026-09-20 G6:
    the old ``num_seats + 4`` cap ran out with six of them, leaving an away
    actor that nothing would ever move)."""
    for _ in range(3 * (t.num_seats or TABLE_SEATS) + 8):
        if t.phase != "in_hand" or t.env is None or t.info is None:
            t.turn_started_mono = None
            return
        actor = t.env.current_actor()
        if actor is None:
            t.turn_started_mono = None
            return
        if _actor_is_away(t, int(actor)):
            gate, chips = _passive_choice(t)
            _apply_action_locked(t, gate, chips, _resume=False)
            continue
        _start_clock_locked(t)
        return
    # Not drained (cannot happen with passive play): the watchdog's away
    # check picks it up on its next tick.
    t.turn_started_mono = None


def _bank_state(t: LiveTable) -> tuple[bool, float]:
    """(burning the time bank right now?, seconds of bank left for the actor)."""
    if t.phase != "in_hand" or t.env is None:
        return False, 0.0
    actor = t.env.current_actor()
    p = t.seats[int(actor)] if actor is not None and int(actor) < len(t.seats) else None
    if p is None:
        return False, 0.0
    left = max(0.0, float(p.time_bank_left or 0.0))
    key = (t.hand_no, t.action_seq)
    if t.bank_key == key and t.bank_started_mono is not None:
        used = max(0.0, time.monotonic() - t.bank_started_mono)
        return True, max(0.0, left - used)
    return False, left


def _settle_bank_locked(t: LiveTable, actor: int | None) -> None:
    """The decision is over: charge the actor the bank seconds they used."""
    if t.bank_started_mono is not None and t.bank_key == (t.hand_no, t.action_seq):
        p = t.seats[actor] if actor is not None and 0 <= actor < len(t.seats) else None
        if p is not None:
            used = max(0.0, time.monotonic() - t.bank_started_mono)
            p.time_bank_left = max(0.0, float(p.time_bank_left or 0.0) - used)
    t.bank_key = None
    t.bank_started_mono = None


def _timeout_tick_locked(t: LiveTable) -> None:
    if t.phase != "in_hand" or t.env is None or t.info is None:
        return
    actor = t.env.current_actor()
    if actor is None:
        return
    actor = int(actor)
    timed_out = False
    # (review 2026-09-20 G6) An away actor is auto-acted even on a clock-less
    # table — the watchdog used to return early when decision_secs == 0.
    if not _actor_is_away(t, actor):
        if int(t.decision_secs or 0) <= 0 or t.turn_started_mono is None:
            return
        now = time.monotonic()
        base_end = t.turn_started_mono + float(t.decision_secs)
        if now < base_end:
            return
        # Base clock is out: burn the actor's time bank before acting for them.
        p = t.seats[actor]
        bank = max(0.0, float(p.time_bank_left or 0.0)) if p is not None else 0.0
        key = (t.hand_no, t.action_seq)
        if bank > 0.0:
            if t.bank_key != key or t.bank_started_mono is None:
                t.bank_key = key
                t.bank_started_mono = base_end
                t.rev += 1
            if now - t.bank_started_mono < bank:
                return
        timed_out = True
    name = _seat_name(t, actor)
    who = t.seats[actor]
    gate, chips = _passive_choice(t)
    _apply_action_locked(t, gate, chips)
    if timed_out:
        _emit(
            t, "timeout",
            f"{name} timed out — {'folded' if gate == GATE_FOLD else 'checked'}",
            seat=actor,
        )
        if who is not None and t.seats[actor] is who:
            who.timeouts += 1
            if who.timeouts >= TIMEOUTS_BEFORE_SIT_OUT and not who.sitting_out:
                who.timeouts = 0
                who.sitting_out = True
                who.sit_out_next = False
                t.rev += 1
                _persist_safe(t)
                _emit(t, "away", f"{name} is sitting out (timed out twice)", seat=actor)
                if t.phase == "in_hand":
                    _resume_turn_locked(t)


# --- verifiable shuffle: seal -> commit -> lock -> reveal -> cut -> deal ----------------


def _fair_hand_id(t: LiveTable, hand_no: int, attempt: int) -> str:
    """Names ONE sealed deck. ``epoch`` (per in-memory load of the table) keeps an
    id from ever naming two different seals across a restart or an eviction. A
    game other than PLO5 is named in it too, so every commitment (and the seal)
    also says which slot map the deck is dealt by."""
    gid = f"{t.game_id}:{int(hand_no)}:{int(attempt)}:{t.epoch}"
    return gid if _norm_game(t.variant) == DEFAULT_GAME else f"{gid}:{_norm_game(t.variant)}"


def _fair_seal(t: LiveTable, hand_no: int, attempt: int) -> Any:
    return fairdeal.SealedDeck.create(
        _fair_hand_id(t, hand_no, attempt), t.num_seats, hole=t.hole_count, burns=t.burns)


def _seal_fits(t: LiveTable, sealed: Any) -> bool:
    """Can this sealed deck be dealt at the table as it is now? Its slot map
    depends on the seat count, the hole cards per seat and the burns (HGB-024:
    the check was written out twice)."""
    return (sealed.num_seats == t.num_seats and sealed.hole == t.hole_count
            and sealed.burns == t.burns)


def _fair_prepare_locked(t: LiveTable) -> None:
    """Seal the deck of the upcoming hand as soon as there is an upcoming hand."""
    if not FAIR_ON or t.status != "open" or t.phase == "in_hand":
        return
    nxt = t.fair_next
    if nxt is not None and nxt.hand_no == t.hand_no + 1:
        if not _seal_fits(t, nxt.sealed):
            _fair_void_locked(t, "the table was resized")
        return
    t.fair_next = FairPending(sealed=_fair_seal(t, t.hand_no + 1, 1), hand_no=t.hand_no + 1)
    t.rev += 1


def _fair_penalized(t: LiveTable, uid: int) -> bool:
    return int(t.fair_penalty_until.get(int(uid), 0)) > int(t.hand_no)


def _fair_commit_locked(t: LiveTable, uid: int, hand_id: str, commit: str) -> None:
    _require_open(t)
    nxt = t.fair_next
    seat = t.seat_of(uid)
    if not FAIR_ON or nxt is None or seat is None:
        raise HTTPException(status_code=409, detail="no shuffle to take part in")
    if nxt.sealed.hand_id != str(hand_id) or nxt.stage != "commit":
        raise HTTPException(status_code=409, detail="that shuffle is closed")
    if not fairdeal.is_hex64(commit):
        raise HTTPException(status_code=400, detail="a commitment is 64 hex characters")
    if int(uid) in nxt.barred or _fair_penalized(t, uid):
        raise HTTPException(status_code=409, detail="this device sits this shuffle out")
    # Until the list is LOCKED a seat may replace its commitment: a reopened tab
    # (or a second device) no longer has the old number, and a commitment nobody
    # can open would void the shuffle and blame the player. Nothing has been
    # revealed at this point, so the newest commitment is as good as the first.
    nxt.commits[seat] = str(commit)
    nxt.commit_users[seat] = int(uid)
    t.fair_capable.add(int(uid))
    if nxt.pending:
        _fair_tick_locked(t)  # the deal may have been waiting for exactly this


def _fair_reveal_locked(t: LiveTable, uid: int, hand_id: str, nonce: str) -> None:
    nxt = t.fair_next
    if not FAIR_ON or nxt is None or nxt.sealed.hand_id != str(hand_id) or nxt.stage != "reveal":
        raise HTTPException(status_code=409, detail="that shuffle is closed")
    seat = next((st for st, u in nxt.commit_users.items()
                 if u == int(uid) and st in dict(nxt.sealed.locked)), None)
    if seat is None:
        raise HTTPException(status_code=409, detail="this device is not part of the lock list")
    if not nxt.sealed.accepts(seat, str(nonce)):
        raise HTTPException(status_code=400, detail="that number does not open your commitment")
    nxt.reveals[seat] = str(nonce)
    t.fair_strikes.pop(int(uid), None)
    if len(nxt.reveals) == len(nxt.sealed.locked):
        _fair_complete_locked(t)


def _fair_expected(t: LiveTable, mask: list[bool]) -> set[int]:
    """Dealt-in seats whose browser has taken part before and is here now."""
    nxt = t.fair_next
    now = time.monotonic()
    out = set()
    for i, p in enumerate(t.seats):
        if p is None or i >= len(mask) or not mask[i]:
            continue
        uid = int(p.user_id)
        if uid not in t.fair_capable or uid in nxt.barred or _fair_penalized(t, uid):
            continue
        if now - t.seen.get(uid, -1e9) > FAIR_PRESENT_S:
            continue
        out.add(i)
    return out


def _fair_lock_locked(t: LiveTable, mask: list[bool]) -> bool:
    """Freeze the commitments of the seats being dealt in. True = waiting for
    their numbers; False = nobody contributed, deal now."""
    nxt = t.fair_next
    now = time.monotonic()
    # Only a device that is HERE is asked for its number: one that committed and
    # then went to sleep (a locked phone) would hold the table up for nothing.
    locked = {
        st: c for st, c in nxt.commits.items()
        if st < len(mask) and mask[st] and t.seats[st] is not None
        and int(t.seats[st].user_id) == nxt.commit_users.get(st)
        and nxt.commit_users.get(st) not in nxt.barred
        and now - t.seen.get(int(t.seats[st].user_id), -1e9) <= FAIR_PRESENT_S
    }
    nxt.sealed.set_lock(locked)
    nxt.reveals = {}
    if not locked:
        nxt.pending = False
        nxt.deadline_mono = None
        return False
    nxt.stage = "reveal"
    nxt.pending = True
    nxt.deadline_mono = time.monotonic() + FAIR_REVEAL_S
    t.rev += 1
    return True


def _fair_begin_locked(t: LiveTable, mask: list[bool]) -> bool:
    """The deal wants to happen. True = it now waits on the shuffle."""
    nxt = t.fair_next
    missing = _fair_expected(t, mask) - set(nxt.commits)
    if missing and nxt.attempt < FAIR_MAX_ATTEMPTS:
        nxt.pending = True
        nxt.deadline_mono = time.monotonic() + (FAIR_COMMIT_GRACE_S if nxt.attempt == 1 else FAIR_RECOMMIT_S)
        t.rev += 1
        return True
    return _fair_lock_locked(t, mask)


def _fair_void_locked(t: LiveTable, reason: str, seats: list[int] | None = None) -> None:
    """Throw the sealed deck away (its cut may already be known to the server)
    and seal a new one. ALWAYS announced: a voided shuffle is a re-roll."""
    nxt = t.fair_next
    if nxt is None:
        return
    # OPS-014: the device that failed to confirm is the USER who committed from
    # that seat — not whoever sits there now — and the tally is per user.
    names = [_user_name(t, nxt.commit_users[i]) if i in nxt.commit_users else _seat_name(t, i)
             for i in (seats or [])]
    for i in seats or []:
        uid = nxt.commit_users.get(i)
        if uid is None:
            continue
        t.fair_void_counts[int(uid)] = int(t.fair_void_counts.get(int(uid), 0)) + 1
        nxt.barred.add(int(uid))
        n = int(t.fair_strikes.get(int(uid), 0)) + 1
        t.fair_strikes[int(uid)] = n
        if n >= FAIR_STRIKES:
            t.fair_penalty_until[int(uid)] = int(t.hand_no) + 1 + FAIR_PENALTY_HANDS
            t.fair_strikes.pop(int(uid), None)
    voids = list(nxt.voids) + [{
        "attempt": nxt.attempt, "seal": nxt.sealed.seal, "reason": reason, "names": names,
    }]
    was_pending = bool(nxt.pending)
    t.fair_next = FairPending(
        sealed=_fair_seal(t, nxt.hand_no, nxt.attempt + 1),
        hand_no=nxt.hand_no, attempt=nxt.attempt + 1, barred=set(nxt.barred), voids=voids,
    )
    _emit(t, "fair", "Shuffle redone — " + (
        f"{', '.join(names)} didn't confirm in time" if names else reason),
        seats=[int(i) for i in seats or []])  # (the client explains it to the device it names)
    t.rev += 1
    if was_pending and seats:  # the deal is still wanted: a short window to commit to the new seal
        t.fair_next.pending = True
        t.fair_next.deadline_mono = time.monotonic() + FAIR_RECOMMIT_S


def _fair_complete_locked(t: LiveTable) -> None:
    """Every locked device revealed: cut, and deal."""
    nxt = t.fair_next
    if nxt is None:
        return
    nxt.pending = False
    nxt.deadline_mono = None
    try:
        _deal_now_locked(t)
    except HTTPException as e:  # nobody left to deal to, game paused, …
        if t.fair_next is nxt:
            _fair_void_locked(t, f"the hand could not be dealt ({e.detail})")
    except Exception:  # noqa: BLE001 — a deck that cannot be finished must not stick
        logger.exception("homegame shuffle could not be dealt (table %s)", t.game_id)
        if t.fair_next is nxt:
            _fair_void_locked(t, "the hand could not be dealt")


def _fair_tick_locked(t: LiveTable) -> None:
    nxt = t.fair_next
    if not FAIR_ON or nxt is None or not nxt.pending:
        return
    if t.status != "open" or not t.running or t.phase == "in_hand":
        nxt.pending = False
        nxt.deadline_mono = None
        if nxt.stage == "reveal":
            _fair_void_locked(t, "the deal was called off")
        return
    now = time.monotonic()
    if nxt.stage == "commit":
        mask = _eligible_mask(t)
        waiting = _fair_expected(t, mask) - set(nxt.commits)
        if waiting and nxt.deadline_mono is not None and now < nxt.deadline_mono:
            return
        if not _fair_lock_locked(t, mask):
            _fair_complete_locked(t)
        return
    if len(nxt.reveals) == len(nxt.sealed.locked):
        _fair_complete_locked(t)
    elif nxt.deadline_mono is not None and now >= nxt.deadline_mono:
        _fair_void_locked(t, "a device did not confirm", [
            st for st, _ in nxt.sealed.locked if st not in nxt.reveals])


def _fair_openings(sealed: Any, cards: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in cards:
        try:
            out[str(int(c))] = sealed.opening(int(c))
        except KeyError:
            continue
    return out


def _fair_view(t: LiveTable, viewer_id: int, visible: list[int]) -> dict[str, Any]:
    """What the viewer's device needs: the upcoming shuffle (to take part) and
    the proof of every card the viewer can see right now (never of any other)."""
    out: dict[str, Any] = {"supported": bool(FAIR_ON), "spec": fairdeal.SPEC}
    if not FAIR_ON:
        return out
    seat = t.seat_of(viewer_id)
    nxt = t.fair_next
    if nxt is not None and t.status == "open":
        mine = seat is not None and nxt.commit_users.get(seat) == int(viewer_id)
        out["next"] = {
            "hand_no": nxt.hand_no, "attempt": nxt.attempt, "hand_id": nxt.sealed.hand_id,
            "seal": nxt.sealed.seal, "stage": nxt.stage, "pending": bool(nxt.pending),
            "locked": [[st, c] for st, c in nxt.sealed.locked] if nxt.stage == "reveal" else [],
            "lock": nxt.sealed.lock if nxt.stage == "reveal" else None,
            "you": {
                "seat": seat, "committed": bool(mine),
                "revealed": bool(mine and seat in nxt.reveals),
                "barred": bool(int(viewer_id) in nxt.barred or _fair_penalized(t, viewer_id)),
                # hands this device still sits out of the shuffle (it missed confirming
                # in time: the client says so and why — HGT-012)
                "benched_hands": max(0, int(t.fair_penalty_until.get(int(viewer_id), 0)) - int(t.hand_no)),
            },
        }
    fh = t.fair_hand
    if fh is not None and t.phase in ("in_hand", "showdown"):
        meta = t.fair_hand_meta or {}
        out["hand"] = {
            "hand_no": int(meta.get("hand_no") or t.hand_no), "hand_id": fh.hand_id,
            "seal": fh.seal, "lock": fh.lock, "num_seats": fh.num_seats, "hole": fh.hole,
            "burns": fh.burns,
            "contributors": [st for st, _ in fh.locked],
            "names": list(meta.get("names") or []),
            "voids": list(meta.get("voids") or []),
            "open": _fair_openings(fh, visible),
        }
    # (wire format: display name -> count; two players with one name stay apart)
    counts: dict[str, int] = {}
    for uid, n in t.fair_void_counts.items():
        base = _user_name(t, uid)
        name, k = base, 2
        while name in counts:
            name, k = f"{base} ({k})", k + 1
        counts[name] = int(n)
    out["void_counts"] = counts
    return out


def _fair_transcript(t: LiveTable, viewer_id: int, hand_no: int) -> dict[str, Any]:
    """The public transcript of a dealt hand + the openings of the cards THIS
    viewer may see (live hand: what is on their screen; older: their history
    view — the table's reveal rule either way)."""
    if not FAIR_ON or hand_no < 1 or hand_no > t.hand_no:
        raise HTTPException(status_code=404, detail="Not Found")
    live = hand_no == t.hand_no and (t.phase == "in_hand" or _runout_blocking(t))
    if hand_no == t.hand_no and t.fair_hand is not None:
        sealed, meta = t.fair_hand, dict(t.fair_hand_meta or {})
    else:
        row = pub.DB.one("SELECT data FROM homegame_fair WHERE game_id=? AND hand_no=?",
                         (t.game_id, int(hand_no)))
        if row is None:
            raise HTTPException(status_code=404, detail="Not Found")
        data = json.loads(row["data"])
        sealed, meta = fairdeal.SealedDeck.from_store(data["sealed"]), dict(data.get("meta") or {})
    visible: list[int] = []
    if live:
        visible = _visible_cards(t, viewer_id)  # (HGB-003: it used to build a whole view)
    else:
        # (SEC-008: any member of the table's club may browse its hands — and so
        # verify their shuffle; the openings still follow the reveal rule)
        rec_row = pub.DB.one("SELECT summary FROM homegame_hands WHERE game_id=? AND hand_no=?",
                             (t.game_id, int(hand_no)))
        if rec_row is not None:
            rec = _hand_for_viewer(json.loads(rec_row["summary"]), viewer_id, bool(t.show_grades))
            for srow in rec.get("seats") or []:
                visible += [c for c in (srow.get("hole") or []) if isinstance(c, int) and c >= 0]
            visible += [int(c) for c in (rec.get("board_a") or []) + (rec.get("board_b") or [])]
            visible += [int(c) for c in rec.get("burns") or []]
    out = sealed.public()
    out.update({"hand_no": int(hand_no), "names": list(meta.get("names") or []),
                "voids": list(meta.get("voids") or []), "open": _fair_openings(sealed, visible)})
    return out


def _deal_ready(t: LiveTable, *, present_only: bool = False) -> bool:
    """Could a hand be dealt right now (2+ seats that would be dealt in)?
    Counts a busted seat whose auto-stack tops it up at the deal.
    ``present_only``: count only players whose browser is at the table."""
    if t.status != "open" or not t.running or _hand_busy(t):
        return False
    ante = t.ante_chips
    now = time.monotonic()
    n = 0
    for p in t.seats:
        if p is None or p.sitting_out or p.sit_out_next or p.leave_after_hand or p.user_id in t.pending_kicks:
            continue
        if present_only and now - t.seen.get(p.user_id, -1e9) > PRESENCE_WINDOW_S:
            continue
        if _auto_target_chips(t, p) > ante:
            n += 1
    return n >= 2


def _auto_deal_tick_locked(t: LiveTable) -> None:
    """Server-driven dealing (watchdog). The next hand used to be dealt by a
    timer in the HOST'S browser — a host who switched tabs or locked their
    phone stalled the table for everyone."""
    delay = float(t.deal_delay_secs or 0.0)
    if FAIR_ON and t.fair_next is not None and t.fair_next.pending:
        t.next_deal_mono = None  # this deal is already under way: its shuffle is being confirmed
        return
    if t.deal_error is not None and t.deal_error_rev == t.rev:
        t.next_deal_mono = None  # it failed and nothing has changed since: don't loop (FEAT-005)
        return
    if delay <= 0.0 or not _deal_ready(t, present_only=True):
        t.next_deal_mono = None
        return
    now = time.monotonic()
    if t.next_deal_mono is None:
        t.next_deal_mono = now + delay
        return
    if now < t.next_deal_mono:
        return
    t.next_deal_mono = None
    try:
        _deal_locked(t)
    except HTTPException as e:
        # Not dealable after all (an automatic top-up that could not happen, say):
        # the table is told why, and it is tried again once something changes.
        t.deal_error = str(e.detail)
        t.rev += 1
        t.deal_error_rev = t.rev


def _names_phrase(names: list[str]) -> str:
    """"Sam", "Sam and Jo", "Sam, Jo and Al", "Sam, Jo and 3 others"."""
    if len(names) <= 1:
        return "".join(names)
    if len(names) <= 3:
        return ", ".join(names[:-1]) + " and " + names[-1]
    return f"{names[0]}, {names[1]} and {len(names) - 2} others"


def _deal_blocked_reason(t: LiveTable) -> str | None:
    """Why the next hand is not coming (FEAT-005: the countdown just disappeared,
    and nobody knew why) — or None when it is (a countdown or a shuffle under way,
    a deal button) or the table is not dealing anyway (closed, paused, a hand on)."""
    if t.status != "open" or not t.running or _hand_busy(t) or t.next_deal_mono is not None:
        return None
    if FAIR_ON and t.fair_next is not None and t.fair_next.pending:
        return None
    if t.deal_error is not None:
        return f"The next hand couldn't be dealt: {t.deal_error}."
    now = time.monotonic()
    ante = t.ante_chips
    ready: list[str] = []
    absent: list[str] = []
    broke: list[str] = []
    away: list[str] = []
    for p in t.seats:
        if p is None:
            continue
        if p.sitting_out or p.sit_out_next or p.leave_after_hand or p.user_id in t.pending_kicks:
            away.append(p.name)
        elif _auto_target_chips(t, p) <= ante:
            broke.append(p.name)
        elif now - t.seen.get(p.user_id, -1e9) > PRESENCE_WINDOW_S:
            absent.append(p.name)
        else:
            ready.append(p.name)
    if len(ready) + len(absent) < 2:
        if broke:
            return f"{_names_phrase(broke)} {'needs' if len(broke) == 1 else 'need'} chips to play the next hand."
        if away:
            return f"Waiting for another player — {_names_phrase(away)} {'is' if len(away) == 1 else 'are'} sitting out."
        return "Waiting for a second player."
    if float(t.deal_delay_secs or 0.0) > 0.0 and len(ready) < 2:
        return f"Waiting for {_names_phrase(absent)} to come back to the table."
    return None


#: The shortest runout pause (the host's setting is 0.3–5 s; there is no
#: "instant" runout — HGB-014 removed the unreachable branches for one).
MIN_STREET_PAUSE_S = 0.3


def _street_pause(t: LiveTable) -> float:
    """The host's runout pause: how long each street of an all-in runout stays up
    once its cards are down (0.3–5 s). Never below MIN_STREET_PAUSE_S."""
    pause = float(t.street_pause_secs if t.street_pause_secs is not None else 1.5)
    if not (math.isfinite(pause) and pause >= MIN_STREET_PAUSE_S):
        pause = MIN_STREET_PAUSE_S
    return pause


def _street_present_s(t: LiveTable, n: int, hands: int) -> float:
    """Seconds street ``n`` (board length) takes to come down once revealed.
    PLO67: its burn, then — red — one more card to each of the ``hands`` hands
    still in, one after another, then the boards' cards. The other games: 0 (their
    street's cards land within the host's pause, as they always did)."""
    if not t.burns:
        return 0.0
    burns = list(t.rabbit_burns)
    j = n - 3  # this street's burn: 0 flop, 1 turn, 2 river
    red = 0 <= j < len(burns) and _is_red_card(int(burns[j]))
    extra = RUNOUT_EXTRA_S + (hands - 1) * RUNOUT_EXTRA_GAP_S if (red and hands > 0) else 0.0
    return RUNOUT_BURN_S + extra + RUNOUT_BOARD_S


def _is_red_card(c: int) -> bool:
    """A red card (hearts or diamonds — the engine's suits 1 and 2): a red burn
    deals every hand still in one more card (PLO67)."""
    return int(c) % 4 in (1, 2)


def _make_runout_plan(t: LiveTable, hands: int) -> dict[str, Any]:
    """The all-in runout's timeline, in seconds after it starts: when each street
    is REVEALED (``reveal``), when its cards are all DOWN (``settled`` — what the
    equities on the table describe) and when the showdown starts (``award_at``).
    Street n+1 is revealed one host pause after street n is down; the showdown
    comes once the river is down (the other games: once its card has landed —
    it used to start the moment the river was revealed)."""
    base = _street_pause(t)
    start = max(3, min(5, int(t.runout_start_len or 3)))
    reveal = {start: 0.0}
    settled = {start: 0.0}
    for n in range(start + 1, 6):
        reveal[n] = settled[n - 1] + base
        settled[n] = reveal[n] + _street_present_s(t, n, hands)
    if start >= 5:
        award_at = 0.0  # all in on the river: every card is already on the table
    else:
        award_at = settled[5] + (RUNOUT_AWARD_BEAT_S if t.burns else RUNOUT_BOARD_S)
    return {"base": base, "hands": int(hands), "reveal": reveal, "settled": settled,
            "award_at": award_at}


def _runout_plan(t: LiveTable) -> dict[str, Any]:
    if not t.runout_plan:  # (a runout always sets one at its start; this is a safety net)
        first = next(iter(t.equity_by_len.values()), {}) if t.equity_by_len else {}
        t.runout_plan = _make_runout_plan(t, max(2, len(first)))
    return t.runout_plan


def _runout_elapsed(t: LiveTable) -> float:
    started = t.runout_started_mono if t.runout_started_mono is not None else time.monotonic()
    return max(0.0, time.monotonic() - started)


def _runout_shown_len(t: LiveTable) -> int:
    """The board length an all-in runout has REVEALED so far (5 when none runs)."""
    if not t.runout_active:
        return 5
    el = _runout_elapsed(t)
    return max(n for n, at in _runout_plan(t)["reveal"].items() if at <= el)


def _runout_settled_len(t: LiveTable) -> int:
    """The board length whose cards are all DOWN on the table — a revealed PLO67
    street is still being presented (burn, extra cards, board cards) until then.
    The equities shown describe this street, so they never run ahead of what the
    players can see (owner, 2026-09-29: "the all-in equities seem off")."""
    if not t.runout_active:
        return 5
    el = _runout_elapsed(t)
    return max(n for n, at in _runout_plan(t)["settled"].items() if at <= el)


def _runout_award_index(t: LiveTable) -> int:
    n = len(t.pot_awards or [])
    if not t.runout_active:
        return n
    into = _runout_elapsed(t) - _runout_plan(t)["award_at"]
    if into < 0:
        return -1
    return min(n, int(into / AWARD_SECS))


def _runout_timing(t: LiveTable) -> dict[str, int] | None:
    """What the table animates a PLO67 all-in runout with (ms) — the same numbers
    the plan paces it by. None for the other games and outside a runout."""
    if not (t.runout_active and t.burns):
        return None
    return {"burn_ms": int(RUNOUT_BURN_S * 1000), "extra_ms": int(RUNOUT_EXTRA_S * 1000),
            "extra_gap_ms": int(RUNOUT_EXTRA_GAP_S * 1000), "board_ms": int(RUNOUT_BOARD_S * 1000)}


def _runout_blocking(t: LiveTable) -> bool:
    if not t.runout_active:
        return False
    n = len(t.pot_awards or [])
    return _runout_shown_len(t) < 5 or _runout_award_index(t) < n


def _hand_busy(t: LiveTable) -> bool:
    """A hand is being played OR its all-in runout is still revealing —
    either way the hand is not over for the people watching it."""
    return t.phase == "in_hand" or _runout_blocking(t)


def _settle_locked(t: LiveTable) -> None:
    """Deferred end-of-hand work: cash out players who left / were removed
    mid-hand. Waits for the runout to finish revealing — a cashed-out row in
    the ledger would announce the result early (review 2026-09-20 G11).
    Called from the watchdog, every view, and before a deal; a failed
    cash-out stays pending and is retried."""
    _apply_leaves_locked(t)
    if not t.pending_kicks or _hand_busy(t):
        return
    for uid in list(t.pending_kicks):
        i = t.seat_of(uid)
        try:
            if i is not None:
                _cash_out_seat(t, i)
            t.pending_kicks.discard(uid)
        except Exception:  # noqa: BLE001 — rolled back; retry next tick
            logger.exception("deferred cash-out failed (table %s)", t.game_id)


@dataclass
class _Shared:
    """The table-wide half of every viewer's view (HGB-003 / PERF-004): the engine's
    state, the boards, the runout's progress, the ledger, the seats as everybody
    sees them, the pots and the action history. Built ONCE per state of the table
    (``_shared_key``) and reused for every viewer until the table changes — each
    view only adds what is the viewer's own: which cards they may see, their
    seat, their turn."""

    key: tuple
    raw: dict[str, Any]
    actor: int | None
    all_holes: list[list[int]]
    folded: list[bool]
    in_hand: list[bool]
    dealt: list[int | None]
    reveal: bool
    trim_len: int | None
    ba_src: list[int]
    bb_src: list[int]
    burns_shown: list[int]
    burns_played: int
    street: str | None
    runout_live: bool
    award_idx: int
    start_stacks: list[int]
    display_stacks: list[int]
    ledger: list[dict[str, Any]]
    seat_cents: dict[int, int]
    seats: list[dict[str, Any]]      # without the viewer's own fields (see _seat_rows)
    history: list[dict[str, Any]]
    live_pots: list[dict[str, Any]]
    eligible: int


def _shared_key(t: LiveTable) -> tuple:
    """Everything the shared half depends on. ``rev`` moves with every change to the
    table; the runout reveals itself by the clock, so its progress is in the key
    too (no rev bump when a street or an award step comes)."""
    return (
        t.epoch, t.rev, t.phase, t.hand_no, t.action_seq, t.num_seats,
        _runout_shown_len(t) if t.runout_active else 0,
        _runout_settled_len(t) if t.runout_active else 0,  # (the equities follow it)
        _runout_award_index(t) if t.runout_active else -2,
    )


def _shared(t: LiveTable) -> _Shared:
    """The shared half of the view, from the cache when the table has not changed."""
    key = _shared_key(t)
    hit = t.view_cache
    if hit is not None and hit.key == key:
        return hit
    sh = _build_shared(t, key)
    t.view_cache = sh
    return sh


def _build_shared(t: LiveTable, key: tuple) -> _Shared:
    n = t.num_seats
    raw: dict[str, Any] = {}
    actor = None
    all_holes: list[list[int]] = []
    folded = [True] * n
    in_hand = list(t.in_hand_mask or []) + [False] * n
    in_hand = in_hand[:n]
    dealt = list(t.dealt_user_ids or []) + [None] * n
    dealt = dealt[:n]
    if t.env is not None and t.phase in ("in_hand", "showdown"):
        raw = _obs_dict(t.env)
        actor_raw = raw.get("actor")
        actor = int(actor_raw) if actor_raw is not None else None
        all_holes = [[int(c) for c in h] for h in t.env.all_hole_cards()]
        folded = [bool(x) for x in raw.get("folded", [])]
        folded += [True] * (n - len(folded))
    # (review 2026-09-20 G1) Cards are tabled only at a REAL showdown: two or
    # more live hands at terminal. "phase == showdown" alone also covers a
    # fold-out, where the winner's hand used to be shown to the whole table
    # (and to unseated viewers) — every successful bluff was exposed.
    reveal = t.phase == "showdown" and bool(t.showdown_reveal)
    # PLO67: while an all-in runout is revealing, every hand shows what it held
    # on the street being shown (red burns still to come deal the rest)
    trim_len = max(3, _runout_shown_len(t)) if (t.runout_active and t.burns) else None
    ba_src, bb_src, burns_shown, burns_played, street = _boards(t, raw)

    # --- all-in runout: what has been revealed SO FAR -----------------------
    # (review 2026-09-20 G11) While the streets/awards are still animating,
    # nothing in the payload may run ahead of them: no final deltas, no
    # final ledger, no award list, no unrevealed board card.
    runout_live = bool(t.runout_active and _runout_blocking(t))
    award_idx = _runout_award_index(t) if t.runout_active else -1
    start_stacks = (list(t.hand_start_stacks or []) + [0] * n)[:n]
    display_stacks = [0] * n
    if runout_live:
        display_stacks = (list(t.leftover_stacks or []) + [0] * n)[:n]
        for step in (t.pot_awards or [])[: max(0, award_idx)]:
            for k, v in (step.get("shares") or {}).items():
                si = int(k)
                if 0 <= si < n:
                    display_stacks[si] += int(v)
    # Ledger + settled stack cents. During the reveal the ledger stays on the
    # hand-START stacks (exactly as it does while a hand is being played).
    ledger, seat_cents = _ledger_state(t, start_stacks if runout_live else None)
    sh = _Shared(
        key=key, raw=raw, actor=actor, all_holes=all_holes, folded=folded, in_hand=in_hand,
        dealt=dealt, reveal=reveal, trim_len=trim_len, ba_src=ba_src, bb_src=bb_src,
        burns_shown=burns_shown, burns_played=burns_played, street=street,
        runout_live=runout_live, award_idx=award_idx, start_stacks=start_stacks,
        display_stacks=display_stacks, ledger=ledger, seat_cents=seat_cents, seats=[],
        history=_history_entries(raw, t.bb_cents) if raw else [],
        live_pots=(_live_pots(raw, in_hand, n) if raw and t.phase == "in_hand" and not t.runout_active else []),
        eligible=sum(_eligible_mask(t, start_stacks if runout_live else None)),
    )
    sh.seats = _shared_seat_rows(t, sh)
    return sh


def _boards(t: LiveTable, raw: dict[str, Any]) -> tuple[list[int], list[int], list[int], int, str | None]:
    """(board A, board B, the PLO67 burns shown, how many of them were played, the
    street) as the table shows them now: the rabbit's streets only once it is
    hunted, an all-in runout's street by street."""
    ba_src = list(raw.get("board_a") or t.rabbit_full_a or [])
    bb_src = list(raw.get("board_b") or t.rabbit_full_b or [])
    if t.phase == "showdown" and t.rabbit_full_a:
        ba_src = list(t.rabbit_full_a)
        bb_src = list(t.rabbit_full_b)
        if t.runout_active:
            k = max(3, _runout_shown_len(t))
            ba_src, bb_src = ba_src[:k], bb_src[:k]
        elif not t.rabbit_shown:
            k = max(3, int(t.rabbit_played_len or 3))
            ba_src, bb_src = ba_src[:k], bb_src[:k]
    ba_src = [int(c) for c in ba_src]
    bb_src = [int(c) for c in bb_src]
    # PLO67's face-up burns: one per street on the board (the rabbit and an
    # all-in runout show them street by street, like the board cards)
    burns_shown: list[int] = []
    if t.burns and t.phase in ("in_hand", "showdown"):
        src = [int(c) for c in raw.get("burns") or []] if t.phase == "in_hand" else list(t.rabbit_burns)
        if t.phase == "showdown" and not src and raw:
            src = [int(c) for c in raw.get("burns") or []]
        burns_shown = src[: max(0, min(t.burns, len(ba_src) - 2))]
    # the burns that came while the hand was played; the rest (a fold-out's
    # rabbit) dealt nobody a card
    burns_played = len(burns_shown)
    if t.phase == "showdown" and t.rabbit_available and t.rabbit_shown:
        burns_played = min(burns_played, max(0, int(t.rabbit_played_len or 3) - 2))
    street = STREET_NAMES.get(int(raw["street"]), "flop") if raw else None
    if t.runout_active:
        street = {3: "flop", 4: "turn", 5: "river"}.get(len(ba_src), street)
    return ba_src, bb_src, burns_shown, burns_played, street


def _visible_holes(t: LiveTable, sh: _Shared, viewer_id: int) -> tuple[list[list[int] | None], list[list[int] | None]]:
    """(each seat's hole cards as THIS viewer may see them — sorted, or all -1 face
    down, or None — and, PLO67, the same hands in the order their cards came).

    (review 2026-09-20 G2) "Own" = the user who was DEALT this hand, not whoever
    sits in the seat now: a seated-but-not-dealt player used to see dead hole cards
    mid-hand, and a newcomer taking the seat after the hand saw the previous
    occupant's mucked cards. The engine deals EVERY seat, dealt-in or not; a
    masked-out seat's hole cards are live-deck information and never shown."""
    n = t.num_seats
    holes: list[list[int] | None] = [None] * n
    seqs: list[list[int] | None] = [None] * n
    for i in range(min(n, len(sh.all_holes))):
        if not sh.in_hand[i]:
            continue
        own = sh.dealt[i] is not None and sh.dealt[i] == viewer_id
        occupant = t.seats[i].user_id if t.seats[i] is not None else None
        if occupant is not None and occupant != sh.dealt[i]:
            # Someone else has taken the seat since: the old hand is not theirs
            # to show (or to sit behind), face-down included.
            continue
        # Tabled voluntarily after the hand ("show cards") — the player's own
        # choice, so it also covers a fold-out winner and a folded hand.
        shown = t.phase == "showdown" and i in t.shown_seats
        cards_i = list(sh.all_holes[i])
        if sh.trim_len is not None:
            cards_i = _hand_on(t, i, cards_i, sh.trim_len)
        if own or shown or (sh.reveal and not sh.folded[i]):
            holes[i] = _sorted_hole(cards_i)
            if t.burns:
                seqs[i] = list(cards_i)
        else:
            holes[i] = [-1] * len(cards_i)  # facedown
    return holes, seqs


def _visible_cards(t: LiveTable, viewer_id: int) -> list[int]:
    """Every card THIS viewer has on their screen right now — the hole cards they may
    see, both boards, the burns up — the cards whose shuffle proofs they get
    (``_fair_view``, ``_fair_transcript``: HGB-003, it used to build a whole view)."""
    sh = _shared(t)
    holes, _ = _visible_holes(t, sh, viewer_id)
    return ([int(c) for h in holes if h for c in h if int(c) >= 0]
            + list(sh.ba_src) + list(sh.bb_src) + list(sh.burns_shown))


def _shared_seat_rows(t: LiveTable, sh: _Shared) -> list[dict[str, Any]]:
    """Every seat as everybody sees it (cards, the viewer's own queued chips, the
    host's request badge and presence come per viewer in ``_seat_rows``)."""
    raw, n = sh.raw, t.num_seats
    settled = t.phase != "in_hand" and not sh.runout_live
    in_hand_set = {j for j, m in enumerate(sh.in_hand) if m} if any(sh.in_hand) else None
    # the equities of the street whose cards are all DOWN — a revealed PLO67 street
    # is still presenting its burn / extra cards / board cards (``_runout_settled_len``)
    n_eq = _runout_settled_len(t) if t.runout_active else 0
    eq_map = (t.equity_by_len.get((n_eq, n_eq)) or {}) if t.runout_active else {}
    rows: list[dict[str, Any]] = []
    for i in range(n):
        p = t.seats[i]
        if p is not None and not sh.in_hand[i]:
            stack_chips = p.stack_chips  # not in this hand (sat / reloaded mid-hand)
        elif sh.runout_live:
            stack_chips = sh.display_stacks[i]
        elif t.phase == "showdown" and p is not None:
            stack_chips = p.stack_chips
        elif raw and t.phase == "in_hand":
            stack_chips = int(raw["stacks"][i])
        else:
            stack_chips = p.stack_chips if p else 0
        street_commit = int(raw["street_commit"][i]) if raw else 0
        if settled and p is not None and i in sh.seat_cents:
            stack_cents = sh.seat_cents[i]  # agrees with the ledger to the cent
        else:
            stack_cents = chips_to_cents(stack_chips, t.bb_cents)
        # (review 2026-09-20 G3) equities: computed once per hand in `_capture_rabbit`
        # from the ALIVE holes — the same numbers for every viewer.
        eq = eq_map.get(i)
        rows.append({
            "seat": i,
            "empty": p is None,
            "user_id": p.user_id if p else None,
            "name": p.name if p else None,
            "avatar": _avatar_url(p.user_id) if p else None,
            "sitting_out": bool(p.sitting_out) if p else False,
            "stack_chips": stack_chips,
            "stack_cents": stack_cents,
            "committed_this_street_chips": street_commit,
            "committed_this_street_cents": chips_to_cents(street_commit, t.bb_cents),
            "folded": bool(raw["folded"][i]) if raw else False,
            "all_in": bool(raw["all_in"][i]) if raw else False,
            "in_hand": bool(sh.in_hand[i]),
            "is_actor": sh.actor is not None and i == sh.actor,
            "is_hero": False,
            "is_host": p is not None and p.user_id == t.host_user_id,
            "position": position_name(i, t.button, n, in_hand_set),
            "hole": None,
            "hole_seq": None,
            "hand_desc": None,
            "auto_stack_cents": int(p.auto_stack_cents or 0) if p else 0,
            "pending_remove": bool(p and (p.user_id in t.pending_kicks or p.leave_after_hand)),
            "leaving": bool(p and p.leave_after_hand),
            "equity_a": eq["a"] if eq else None,
            "equity_b": eq["b"] if eq else None,
            "trusted": bool(p.trusted) if p else False,
            "topup_target_cents": int(p.topup_target_cents or 0) if p else 0,
            "topup_below_cents": int(p.topup_below_cents or 0) if p else 0,
            "queued_topup_cents": 0,
            "queued_remove_cents": 0,
            "request": None,
            "present": False,
            "reserved_by": next(
                (r["name"] for r in t.requests if r["kind"] == "sit" and r["seat"] == i), None
            ) if p is None else None,
            "sit_out_next": bool(p.sit_out_next) if p else False,
            "bank_left_secs": round(float(p.time_bank_left or 0.0), 1) if p else 0.0,
            "shown": bool(t.phase == "showdown" and i in t.shown_seats),
            # everyone sees when the network plays a seat or suggests its moves (homegame_bot)
            "bot": (p.bot_mode or None) if p else None,
        })
    return rows


def _seat_rows(t: LiveTable, sh: _Shared, viewer_id: int, holes: list, seqs: list,
               now: float) -> list[dict[str, Any]]:
    """The seats as THIS viewer sees them: the shared rows plus the cards they may
    see (and those hands' made-hand labels), which seat is theirs, their own
    queued chips, the host's request badges, who is here."""
    viewer_seat = t.seat_of(viewer_id)
    host = viewer_id == t.host_user_id
    out = []
    for i, base in enumerate(sh.seats):
        row = dict(base)
        p = t.seats[i]
        hole = holes[i] if i < len(holes) else None
        row["hole"] = hole
        row["hole_seq"] = seqs[i] if i < len(seqs) else None
        if hole and hole[0] >= 0:
            ba, bb, held = sh.ba_src, sh.bb_src, hole
            if t.runout_active:
                # a PLO67 street still coming down (its burn, extra cards, board cards):
                # label what is ON the table, never the hand the river is about to make
                n_down = _runout_settled_len(t)
                seq = seqs[i] if i < len(seqs) else None
                if n_down < len(ba):
                    ba, bb = ba[:n_down], bb[:n_down]
                    if t.burns and seq:
                        held = list(seq)[: _hole_count_on(t, i, max(3, n_down) - 2)]
            row["hand_desc"] = [describe_made_hand(held, ba), describe_made_hand(held, bb)]
        row["is_hero"] = viewer_seat is not None and i == viewer_seat
        if p is not None and p.user_id == viewer_id:  # your own queued chips only
            row["queued_topup_cents"] = int(p.queued_topup_cents or 0)
            row["queued_remove_cents"] = int(p.queued_remove_cents or 0)
        if host and p is not None:  # the host sees who is waiting on them, right on the seat
            row["request"] = next(({"id": r["id"], "kind": r["kind"], "amount_cents": r["amount_cents"]}
                                   for r in t.requests if r["user_id"] == p.user_id), None)
        row["present"] = bool(p and now - t.seen.get(p.user_id, -1e9) <= PRESENCE_WINDOW_S)
        out.append(row)
    return out


def _action_view(t: LiveTable, sh: _Shared, viewer_seat: int | None) -> dict[str, Any]:
    """The decision on the table, for its actor: what is legal, the raise window,
    what a call costs (zeros for everybody else)."""
    legal = {"fold": False, "check_call": False, "raise": False}
    raise_bounds = {"min_chips": 0, "max_chips": 0, "min_cents": 0, "max_cents": 0}
    to_call_chips = 0
    info, raw, actor = t.info, sh.raw, sh.actor
    me = t.seats[viewer_seat] if viewer_seat is not None and 0 <= viewer_seat < len(t.seats) else None
    # (the network plays this seat: nothing for its owner to press — homegame_bot)
    autopilot = me is not None and me.bot_mode == "auto"
    if (t.phase == "in_hand" and actor is not None and viewer_seat is not None and actor == viewer_seat
            and info is not None and not autopilot):
        gm = info.gate_mask
        legal = {"fold": bool(gm[GATE_FOLD]), "check_call": bool(gm[GATE_CHECK_CALL]),
                 "raise": bool(gm[GATE_RAISE])}
        min_c, max_c = int(info.min_raise_chips), int(info.max_raise_chips)
        if legal["raise"] and min_c == 0 and max_c > 0:
            min_c = max_c
        raise_bounds = {"min_chips": min_c, "max_chips": max_c,
                        "min_cents": chips_to_cents(min_c, t.bb_cents),
                        "max_cents": chips_to_cents(max_c, t.bb_cents)}
        to_call_chips = min(max(0, int(raw["bet_to_call"]) - int(raw["street_commit"][actor])),
                            int(raw["stacks"][actor]))
    return {
        "legal": legal,
        "raise_bounds": raise_bounds,
        "to_call_chips": to_call_chips,
        "to_call_cents": chips_to_cents(to_call_chips, t.bb_cents),
        "street_commit_chips": int(raw["street_commit"][actor]) if raw and actor is not None else 0,
    }


def _runout_view(t: LiveTable, sh: _Shared) -> dict[str, Any]:
    """The pot, the hand's results so far and the all-in runout's clock: nothing
    runs ahead of what the runout has revealed (G11)."""
    raw, n = sh.raw, t.num_seats
    award_idx, n_awards = sh.award_idx, len(t.pot_awards or [])
    if sh.runout_live:
        # Public so far: what each dealt-in seat put in, plus awards shown.
        deltas_cents = [chips_to_cents(sh.display_stacks[i] - sh.start_stacks[i], t.bb_cents)
                        if sh.in_hand[i] else 0 for i in range(n)]
        awarded = sum(int(st.get("chips") or 0) for st in (t.pot_awards or [])[: max(0, award_idx)])
        pot_chips = max(0, int(t.terminal_pot) - awarded)
    else:
        deltas_cents = [chips_to_cents(d, t.bb_cents) for d in (t.last_deltas or [])]
        pot_chips = 0 if t.phase == "showdown" else (int(raw["pot"]) if raw else 0)
    award_step = t.pot_awards[award_idx] if t.runout_active and 0 <= award_idx < n_awards else None
    if not t.runout_active:
        shown_awards: list[dict[str, Any]] = []
    elif sh.runout_live:
        # Award steps only start once all five cards are out, so a released
        # step can never name an unrevealed board card.
        shown_awards = list(t.pot_awards[: max(0, award_idx + 1)])
    else:
        shown_awards = list(t.pot_awards or [])
    pause = _street_pause(t)  # the host's setting (Manage shows it)
    shown_len = _runout_shown_len(t) if t.runout_active else 0
    next_in = 0.0
    if t.runout_active and shown_len < 5:
        next_in = max(0.0, _runout_plan(t)["reveal"][shown_len + 1] - _runout_elapsed(t))
    return {
        "pot_chips": pot_chips,
        "pot_cents": chips_to_cents(pot_chips, t.bb_cents),
        "settled_pot_chips": int(raw["pot"]) - sum(int(x) for x in raw["street_commit"]) if raw else 0,
        "hand_deltas_cents": deltas_cents,
        "street_pause_secs": pause,
        "runout": {
            "active": bool(t.runout_active),
            "start_len": max(3, int(t.runout_start_len or 3)),
            "shown_len": shown_len,
            "settled_len": _runout_settled_len(t) if t.runout_active else 0,
            "pause_secs": pause,
            "next_in_secs": round(next_in, 2),
            "timing": _runout_timing(t),
            "award_index": award_idx,
            "award_count": n_awards,
            "award_step": award_step,
            "blocking": _runout_blocking(t),
        },
        "pot_awards": shown_awards,
        "pots": list(t.pots or []) if t.runout_active else [],
        "live_pots": sh.live_pots,
        # the bet nobody matched, back with its owner (seat -> cents): the table slides
        # those chips home instead of into the pot
        "returned": ({str(s): chips_to_cents(c, t.bb_cents) for s, c in t.uncalled.items()}
                     if t.phase == "showdown" else {}),
    }


def _settings_view(t: LiveTable) -> dict[str, Any]:
    """The table's settings as the view carries them."""
    return {
        "stakes": {
            "sb_cents": t.sb_cents, "bb_cents": t.bb_cents, "ante_cents": t.ante_cents,
            "default_buyin_cents": t.default_buyin_cents, "bb_chips": BB_CHIPS,
        },
        "running": bool(t.running),
        "auto_stack": {"mode": _norm_auto_mode(t.auto_stack_mode), "all_cents": int(t.auto_stack_all_cents or 0)},
        "decision_secs": int(t.decision_secs or 0),
        "host_user_id": t.host_user_id,
        "settings": {
            "deal_delay_secs": float(t.deal_delay_secs or 0.0),
            "time_bank_secs": int(t.time_bank_secs or 0),
            "min_buyin_cents": int(t.min_buyin_cents or 0),
            "max_buyin_cents": int(t.max_buyin_cents or 0),
            "listed": bool(t.listed),
            "allow_rabbit": bool(t.allow_rabbit),
            "approve_buyins": bool(t.approve_buyins),
            "show_grades": bool(t.show_grades),
            "allow_rathole": bool(t.allow_rathole),
        },
        "auto_topup": {
            "mode": _norm_auto_mode(t.topup_mode),
            "all_target_cents": int(t.topup_all_target_cents or 0),
            "all_below_cents": int(t.topup_all_below_cents or 0),
        },
    }


def _view_ticks_locked(t: LiveTable) -> None:
    """A view never lags the clock: the watchdog does this work every quarter
    second, and a view does it first too (a clock ran out, a runout finished and
    someone who left is due their cash-out, a result line is due) — otherwise the
    viewer could see a state up to a tick old."""
    _timeout_tick_locked(t)
    _settle_locked(t)
    _flush_result_locked(t)


def _view(t: LiveTable, viewer_id: int) -> dict[str, Any]:
    """Everything one viewer sees of the table (HGB-003: assembled from
    ``_shared`` — built once per state of the table for every viewer (PERF-004) —
    and the viewer's own parts: ``_visible_holes``, ``_seat_rows``,
    ``_action_view``, ``_runout_view``, ``_settings_view``, the shuffle's proofs of
    the cards they see). Also marks the viewer present."""
    _assert_lock_held(t)
    viewer_id = int(viewer_id)
    now_mono = time.monotonic()
    t.seen[viewer_id] = now_mono  # presence (server-side dealing)
    if viewer_id not in t.names:
        u = pub._user_by_id(viewer_id)
        t.names[viewer_id] = _display_name(u, t.club_id) if u is not None else f"Player {viewer_id}"
    _view_ticks_locked(t)
    sh = _shared(t)
    holes, seqs = _visible_holes(t, sh, viewer_id)
    viewer_seat = t.seat_of(viewer_id)
    blocking = _runout_blocking(t)
    can_show = False
    if t.phase == "showdown" and t.env is not None and sh.raw:
        for i in range(t.num_seats):
            if sh.dealt[i] is None or sh.dealt[i] != viewer_id or not sh.in_hand[i]:
                continue
            occupant = t.seats[i].user_id if t.seats[i] is not None else None
            if occupant is not None and occupant != sh.dealt[i]:
                continue
            can_show = not (i in t.shown_seats or (sh.reveal and not bool(sh.raw["folded"][i])))
    bank_active, bank_left = _bank_state(t)
    now_wall = time.time()
    last_hand_no = t.hand_no if (t.phase != "in_hand" and not blocking) else t.hand_no - 1
    _fair_prepare_locked(t)
    visible = ([int(c) for h in holes if h for c in h if int(c) >= 0]
               + list(sh.ba_src) + list(sh.bb_src) + list(sh.burns_shown))
    out: dict[str, Any] = {
        "fair": _fair_view(t, viewer_id, visible),
        "id": t.game_id,
        "name": t.name,
        # the game dealt here: PLO5 / PLO6 / PLO67 (hole cards per player, seat limit, graded?)
        "variant": _norm_game(t.variant),
        "game": _game_info(t.variant),
        "hole_count": t.hole_count,
        "status": t.status,
        "phase": t.phase,
        "hand_no": t.hand_no,
        "action_seq": t.action_seq,
        "rev": t.rev,
        "epoch": t.epoch,
        "num_seats": t.num_seats,
        "button_seat": t.button,
        "hero_seat": viewer_seat if viewer_seat is not None else 0,
        "actor": sh.actor,
        "my_user_id": viewer_id,
        "my_seat": viewer_seat,
        "is_host": viewer_id == t.host_user_id,
        "is_member": any(row["user_id"] == viewer_id for row in sh.ledger),
        # SEC-008: the table's hands (history, replayer, proofs) are open to every
        # member of its club — and the view is only ever built for one
        "can_browse_hands": True,
        "seats": _seat_rows(t, sh, viewer_id, holes, seqs, now_mono),
        "board": {"a": _split_board(sh.ba_src), "b": _split_board(sh.bb_src)},
        # PLO67: the burn cards turned face up so far (flop, turn, river); [] otherwise
        "burns": list(sh.burns_shown),
        "burns_played": sh.burns_played,
        "street": sh.street,
        "history": sh.history,
        "ledger": sh.ledger,
        # who pays whom, in the fewest payments (FEAT-001; the ledger sums to 0)
        "settle_up": settle_up(sh.ledger, viewer_id),
        "can_deal": (
            t.status == "open" and t.running and t.phase != "in_hand" and not blocking
            and viewer_seat is not None and sh.eligible >= 2
        ),
        "eligible_count": sh.eligible,
        "chat": _chat_messages(t),
        "can_rabbit": bool(t.phase == "showdown" and t.rabbit_available and not t.rabbit_shown),
        "rabbit_shown": bool(t.rabbit_shown),
        "turn_remaining_secs": _turn_remaining_secs(t),
        # buy-in approval: the host sees the queue, a player their own request
        "needs_approval": bool(t.approve_buyins) and not _is_trusted_cached(t, viewer_id),
        "requests": (
            [{**{k: r[k] for k in ("id", "user_id", "name", "kind", "seat", "amount_cents")},
              "avatar": _avatar_url(r["user_id"])}
             for r in t.requests]
            if viewer_id == t.host_user_id else []
        ),
        "my_avatar": _avatar_url(viewer_id),
        # FEAT-008: the name this table shows for the viewer, and whether it is only the
        # "Player <id>" stand-in (then the client asks them to choose one)
        "my_name": t.names.get(viewer_id) or _user_name(t, viewer_id),
        "my_name_default": _viewer_name_default_cached(t, viewer_id),
        "client_build": client_build(),  # (an older page offers a refresh — OPS-039)
        "my_request": next(
            ({k: r[k] for k in ("id", "kind", "seat", "amount_cents")}
             for r in t.requests if r["user_id"] == viewer_id), None,
        ),
        # people asking to join the home games from this table's link (admins only)
        "join_requests": _table_join_requests(t, viewer_id),
        "club": _table_club(t, viewer_id),
        "spectators": sorted(
            t.names.get(uid, "?") for uid, last in list(t.seen.items())
            if now_mono - last <= PRESENCE_WINDOW_S and t.seat_of(uid) is None
        ),
        "next_deal_in_secs": (
            round(max(0.0, t.next_deal_mono - now_mono), 2) if t.next_deal_mono is not None else None
        ),
        # FEAT-005: why the next hand isn't coming (None = it is, or nothing is due)
        "deal_blocked_reason": _deal_blocked_reason(t),
        "time_bank": {
            "secs": int(t.time_bank_secs or 0),
            "active": bool(bank_active),
            "remaining_secs": round(bank_left, 1),
        },
        "can_show": bool(can_show),
        "last_hand_no": last_hand_no if last_hand_no >= 1 else None,
        # the viewer's own network settings — the site's owner only (homegame_bot)
        "bot": _bot_view(t, viewer_id),
        "events": list(t.events),
        "reactions": [
            {"id": r["id"], "seat": r["seat"], "emote": r["emote"]}
            for r in t.reactions
            if now_wall - float(r["ts"]) <= REACTION_TTL_S
        ],
    }
    out.update(_action_view(t, sh, viewer_seat))
    out.update(_runout_view(t, sh))
    out.update(_settings_view(t))
    return out


def _viewer_name_default_cached(t: LiveTable, uid: int) -> bool:
    """``my_name_default`` without a query per view (PERF-004): kept per table, and
    forgotten there when the name changes (``_rename_at_tables``)."""
    hit = t.name_defaults.get(int(uid))
    if hit is None:
        hit = t.name_defaults[int(uid)] = _viewer_name_default(uid)
    return hit


def _viewer_name_default(uid: int) -> bool:
    u = pub._user_by_id(int(uid))
    return bool(u is not None and _name_is_default(u))


def _is_trusted_cached(t: LiveTable, uid: int) -> bool:
    """`_is_trusted` without a DB hit on every poll (PERF-004: the name promised a
    cache that did not exist): the host and seated players answer from memory, an
    unseated viewer's lookup is kept until the table changes (trust is only ever
    set inside a change, which moves ``rev``). Only while approval is on."""
    if not t.approve_buyins or int(uid) == int(t.host_user_id):
        return True
    p = t.player(int(uid))
    if p is not None:
        return bool(p.trusted)
    if t.trust_cache is None or t.trust_cache[0] != t.rev:
        t.trust_cache = (t.rev, {})
    known = t.trust_cache[1]
    if int(uid) not in known:
        known[int(uid)] = _is_trusted(t, uid)
    return known[int(uid)]


def _require_host(t: LiveTable, uid: int, what: str) -> None:
    """Host-only actions: 403 (a permission, like the club's rules — HGB-007: the
    same check was pasted 13 times and answered 400)."""
    if int(uid) != int(t.host_user_id):
        raise HTTPException(status_code=403, detail=f"only the host can {what}")


def _require_open(t: LiveTable) -> None:
    if t.status != "open":
        raise HTTPException(status_code=400, detail="table is closed")


def _is_member(t: LiveTable, uid: int) -> bool:
    """Seated now, or sat at this table before (has a ledger row)."""
    if t.seat_of(uid) is not None:
        return True
    row = pub.DB.one(
        "SELECT 1 FROM homegame_players WHERE game_id=? AND user_id=?",
        (t.game_id, int(uid)),
    )
    return row is not None


def _require_member(t: LiveTable, uid: int) -> None:
    """(review 2026-09-20 G12) Every flagged user can OPEN every table, but
    only people who play (or played) at it may write to it."""
    if not _is_member(t, uid):
        raise HTTPException(status_code=403, detail="take a seat first")


def _cash_out_seat(t: LiveTable, i: int, *, closing: bool = False) -> None:
    """Seat ``i`` leaves with its share of the money on the table.

    A host who leaves hands the table to the next player — but NOT when the
    table is closing (``closing``): closing cashed the host out first, so the
    role walked round the table and the finished session ended up "hosted by"
    whoever was cashed out second to last (with an "X is now the host" toast).

    The amount is the seat's entry in the largest-remainder apportionment of
    the table's money over ALL seated stacks (module docstring, review
    2026-09-20 G10) — whole cents, zero-sum; it used to be an independent
    ``round()`` of the stack. All-or-nothing via ``_mutation``."""
    _assert_lock_held(t)
    p = t.seats[i]
    if p is None:
        return
    with _mutation(t):
        leftover = _ledger_state(t)[1].get(i, 0)
        if leftover:
            p.leftover_cents += leftover
            _ledger_add(t, p.user_id, "cashout", leftover)
        p.stack_chips = 0
        _persist_player(t, None, p, False)
        t.seats[i] = None
        new_host = None
        if t.host_user_id == p.user_id and not closing:
            occ = t.occupied()
            t.host_user_id = t.seats[occ[0]].user_id if occ else t.host_user_id
            new_host = t.seats[occ[0]].name if occ else None
            _persist_meta(t)
    _emit(t, "leave", f"{p.name} left with {_fmt_cents(leftover)}", seat=i)
    if new_host:
        _emit(t, "host", f"{new_host} is now the host")


def _set_running_locked(t: LiveTable, uid: int, running: bool) -> None:
    _require_open(t)
    _require_host(t, uid, "start or pause")
    was = bool(t.running)
    with _mutation(t):
        t.running = bool(running)
        _persist_meta(t)
    t.next_deal_mono = None
    if running:  # the host's Start gives a table the watchdog paused a fresh chance
        t.wd_failing_since = None
        t.wd_next_mono = 0.0
    if not running:
        _fair_tick_locked(t)  # a deal waiting on its shuffle is called off (announced)
    if was != bool(running):
        _emit(t, "run", "Game started" if running else (
            "Game pauses after this hand" if t.phase == "in_hand" else "Game paused"))
    if t.running and not _hand_busy(t) and sum(_eligible_mask(t)) >= 2:
        _deal_locked(t)


def _deal_preconditions(t: LiveTable) -> None:
    _require_open(t)
    if not t.running:
        raise HTTPException(status_code=400, detail="game is paused")
    if t.phase == "in_hand":
        raise HTTPException(status_code=409, detail="hand already in progress")  # (someone dealt first)
    if _runout_blocking(t):
        # (review 2026-09-20 G7) `can_deal` was only a hint in the payload:
        # POST /deal itself never checked, so any seated player could cut
        # everyone's river and pot awards short.
        raise HTTPException(status_code=400, detail="wait for the runout to finish")


def _prepare_deal_locked(t: LiveTable) -> list[bool]:
    """Everything a deal checks and does between hands — before the shuffle
    wait AND again at the deal itself (things may have changed while the
    devices confirmed): the preconditions, deferred cash-outs, "sit out next
    hand", queued top-ups, automatic chips. Returns the seats dealt in (HGB-009:
    it was written out twice, with drifting error texts)."""
    _deal_preconditions(t)
    _settle_locked(t)
    _apply_sit_out_next_locked(t)
    _apply_queued_topups_locked(t, force=True)
    _apply_auto_stacks_locked(t)
    mask = _eligible_mask(t)
    if sum(mask) < 2:
        raise HTTPException(
            status_code=400,
            detail="need at least 2 players with more than the ante",
        )
    return mask


def _player_deal_locked(t: LiveTable, uid: int) -> None:
    """A player's "Deal" (POST /deal). Any seated player may deal the next hand
    on a table dealt by hand; while the server is counting down to the next hand,
    dealing NOW cuts short everyone's look at the showdown — that is the host's
    call (HGT-024: the client offered "Deal now" to the host only, but the server
    took it from anyone seated)."""
    if t.seat_of(uid) is None:
        raise HTTPException(status_code=400, detail="sit to deal")
    if t.next_deal_mono is not None and int(uid) != int(t.host_user_id):
        raise HTTPException(status_code=403, detail="only the host can skip the countdown to the next hand")
    _deal_locked(t)


def _deal_locked(t: LiveTable) -> None:
    _deal_preconditions(t)
    if FAIR_ON and t.fair_next is not None and t.fair_next.pending:
        return  # this deal is already waiting on its shuffle (an impatient second click)
    mask = _prepare_deal_locked(t)
    if FAIR_ON:
        # Verifiable shuffle: devices that take part get a moment to confirm.
        # A table nobody's browser takes part in (scripts, tests) deals at once.
        _fair_prepare_locked(t)
        if _fair_begin_locked(t, mask):
            return
    _deal_now_locked(t)


def _deal_now_locked(t: LiveTable) -> None:
    """Put the hand on the table (the shuffle, if any, is settled)."""
    mask = _prepare_deal_locked(t)
    button = _next_button(t, mask)
    stacks = tuple(s.stack_chips if s is not None else 0 for s in t.seats)
    cfg = GameConfig(
        num_seats=t.num_seats,
        starting_stack=0,
        starting_stacks=stacks,
        ante=t.ante_chips,
        bb=BB_CHIPS,
        variant=t.game["variant"],
        # (owner, 2026-10-02) a bet is capped at the pot and the bettor's own stack —
        # never at what the shorter stacks can call; what nobody matches comes back
        reach_cap=False,
    )
    env = _make_env(cfg)
    nxt = t.fair_next if FAIR_ON else None
    if nxt is not None and (nxt.hand_no != t.hand_no + 1 or not _seal_fits(t, nxt.sealed)):
        _fair_void_locked(t, "the table changed before the deal")
        raise HTTPException(status_code=409, detail="the shuffle is being redone")
    if nxt is not None:
        if not nxt.sealed.lock:
            nxt.sealed.set_lock({})  # nobody contributed: the seal still pins every card
        deck = nxt.sealed.finish(dict(nxt.reveals))
        _, info = env.reset_with_deck(deck, button, in_hand_mask=mask)
        fair_meta = {
            "hand_no": int(t.hand_no) + 1,
            "names": [[st, _seat_name(t, st)] for st, _ in nxt.sealed.locked],
            "voids": list(nxt.voids),
        }
        hand = HandState(hand_deck=list(deck), fair_hand=nxt.sealed, fair_hand_meta=fair_meta)
        t.fair_next = None
        # The transcript outlives the process (history, later audits): owed, and
        # saved with the table by the ``_persist_safe`` below — retried like the
        # stacks if it fails (OPS-013).
        t.unsaved.append(("fair", int(t.hand_no) + 1, json.dumps(
            {"sealed": nxt.sealed.to_store(), "meta": fair_meta}, separators=(",", ":"))))
    else:
        seed = secrets.randbits(63)
        _, info = env.reset(seed, button, in_hand_mask=mask)
        hand = HandState(hand_seed=int(seed))
        t.fair_next = None
    t.deal_error = None  # (FEAT-005: whatever stopped the last automatic deal is over)
    # The engine accepted the hand — commit it to the table: ONE fresh HandState
    # (HGB-004: every per-hand field starts over; nothing is reset by hand).
    hand.env, hand.info = env, info
    hand.hand_start_stacks = list(stacks)
    hand.in_hand_mask = mask
    # (review 2026-09-20 G2) WHO was dealt each seat — own-card visibility is
    # checked against this, never against the seat's current occupant.
    hand.dealt_user_ids = [
        s.user_id if (s is not None and mask[i]) else None
        for i, s in enumerate(t.seats)
    ]
    t.button = button
    t.hand = hand
    # (``phase`` BEFORE ``hand_no``: ``_unpublished_guard`` reads the two without
    # the lock, and this order makes a torn read hide a hand, never show one.)
    t.phase = "showdown" if env.is_terminal() else "in_hand"
    t.hand_no += 1
    t.rev += 1
    t.next_deal_mono = None
    cap = float(max(0, int(t.time_bank_secs or 0)))
    for i, p in enumerate(t.seats):
        if p is not None and mask[i]:
            # A little bank comes back every hand played, never above the cap.
            p.time_bank_left = min(cap, float(p.time_bank_left or 0.0) + TIME_BANK_REFILL_S)
    if t.phase == "showdown":
        _finish_hand_locked(t)
    else:
        _persist_safe(t)
        _resume_turn_locked(t)


def _apply_sit_out_next_locked(t: LiveTable) -> None:
    """"Sit out next hand" takes effect between hands."""
    flagged = [p for p in t.seats if p is not None and p.sit_out_next]
    if not flagged:
        return
    with _mutation(t):
        for i, p in enumerate(t.seats):
            if p is not None and p.sit_out_next:
                p.sit_out_next = False
                p.sitting_out = True
                _persist_player(t, i, p, True)


def _finish_hand_locked(t: LiveTable) -> None:
    assert t.env is not None
    raw = _obs_dict(t.env)
    # Engine stacks at terminal are leftover chips AFTER commits; the pot
    # is not credited back. ``payouts()`` is the chip delta vs hand start
    # (won − total_commit), zero-sum, which is what we persist.
    payouts = t.env.payouts()
    if sum(payouts) != 0:  # (OPS-009: an engine or bookkeeping bug — say so)
        logger.error("homegame table %s hand #%s: payouts do not sum to zero: %s",
                     t.game_id, t.hand_no, payouts)
    t.last_deltas = list(payouts)
    folded = [bool(x) for x in raw["folded"]]
    alive = [
        i
        for i in range(t.num_seats)
        if t.in_hand_mask and i < len(t.in_hand_mask) and t.in_hand_mask[i]
        and i < len(folded) and not folded[i]
    ]
    # (review 2026-09-20 G1) Two or more live hands = showdown, cards are
    # tabled. One = everyone else folded: the winner stays face-down.
    t.showdown_reveal = len(alive) >= 2
    for i, p in enumerate(t.seats):
        if p is not None:
            dealt = bool(t.in_hand_mask and i < len(t.in_hand_mask) and t.in_hand_mask[i])
            if dealt:
                start = t.hand_start_stacks[i] if i < len(t.hand_start_stacks) else p.stack_chips
                after = int(start) + (payouts[i] if i < len(payouts) else 0)
                if after < 0:  # (OPS-009: impossible — the clamp stays, but it is reported)
                    logger.error("homegame table %s hand #%s seat %d: negative stack %d (start %d)",
                                 t.game_id, t.hand_no, i, after, int(start))
                p.stack_chips = max(0, after)
            if p.sit_out_next:  # "sit out next hand" takes effect now
                p.sit_out_next = False
                p.sitting_out = True
    t.phase = "showdown"
    t.info = None
    t.turn_started_mono = None
    t.turn_key = None
    t.bank_key = None
    t.bank_started_mono = None
    t.next_deal_mono = None
    t.rev += 1
    _record_hand_locked(t, raw, payouts, folded)  # owed: saved WITH the stacks
    _persist_safe(t)
    _settle_locked(t)  # no-op while an all-in runout is still revealing
    _flush_result_locked(t)
    _fair_prepare_locked(t)


def _record_equities(t: LiveTable) -> dict[str, Any]:
    """The record's ``equities`` / ``runout_from`` for an all-in runout: each street
    that still had cards to come, each live seat's share of board 1 and board 2
    (the numbers the table showed). Nothing for a hand without one."""
    if not (t.runout_active and t.equity_by_len):
        return {}
    eq = {
        str(n): {str(seat): [round(float(v["a"]), 4), round(float(v["b"]), 4)] for seat, v in sorted(shares.items())}
        for (n, _), shares in sorted(t.equity_by_len.items())
        if n < 5 and shares
    }
    if not eq:
        return {}
    return {"equities": eq, "runout_from": int(t.runout_start_len)}


def _record_hand_locked(
    t: LiveTable, raw: dict[str, Any], payouts: list[int], folded: list[bool]
) -> None:
    """Build the finished hand's record (history + per-player results + who
    paid whom + its grading job) as an OWED write — committed together with the
    stacks by the ``_persist_safe`` that follows (OPS-006) — and queue its
    one-line result for the event feed. Cosmetic: a failure here must never
    stop the hand from finishing.

    The record holds EVERY dealt hand's cards; ``_hand_for_viewer`` filters
    them per viewer with the live table's rule (own cards + tabled hands)."""
    try:
        mask = list(t.in_hand_mask or [])
        mask += [False] * (t.num_seats - len(mask))
        dealt = list(t.dealt_user_ids or [])
        dealt += [None] * (t.num_seats - len(dealt))
        all_holes = t.env.all_hole_cards() if t.env is not None else []
        full_a = [int(c) for c in (raw.get("board_a") or [])]
        full_b = [int(c) for c in (raw.get("board_b") or [])]
        n_board = len(full_a) if t.showdown_reveal else max(3, int(t.rabbit_played_len or 3))
        commit = [int(x) for x in (raw.get("total_commit") or [])]
        # every dealt seat's result in cents, summing to zero like the chips (OPS-015)
        dealt_seats = [i for i in range(t.num_seats) if mask[i] and dealt[i] is not None]
        hand_cents = dict(zip(dealt_seats, zero_sum_cents(
            [int(payouts[i]) if i < len(payouts) else 0 for i in dealt_seats], t.bb_cents)))
        # PLO67: the burns turned up while the hand was played (one per street)
        played_burns = [int(c) for c in (raw.get("burns") or [])][: max(0, n_board - 2)] if t.burns else []
        seats = []
        winners: list[tuple[str, int]] = []
        for i in range(t.num_seats):
            if not mask[i] or dealt[i] is None:
                continue
            p = t.seats[i]
            name = p.name if (p is not None and p.user_id == dealt[i]) else None
            if name is None:
                u = pub._user_by_id(int(dealt[i]))
                name = _display_name(u, t.club_id) if u is not None else f"Player {dealt[i]}"
            delta = int(payouts[i]) if i < len(payouts) else 0
            start = int(t.hand_start_stacks[i]) if i < len(t.hand_start_stacks) else 0
            is_folded = bool(folded[i]) if i < len(folded) else True
            seats.append({
                "seat": i,
                "user_id": int(dealt[i]),
                "name": name,
                "start_cents": chips_to_cents(start, t.bb_cents),
                "start_chips": start,
                "delta_cents": hand_cents.get(i, 0),
                "folded": is_folded,
                "shown": bool(t.showdown_reveal and not is_folded),
                "hole": _sorted_hole([int(c) for c in all_holes[i]]) if i < len(all_holes) else [],
            })
            if t.burns and i < len(all_holes):
                # PLO67: the cards in the order they came (the first four, then one
                # per red burn) and how many the seat held on the flop / turn / river
                seats[-1]["hole_seq"] = [int(c) for c in all_holes[i]]
                seats[-1]["counts"] = [_hole_count_on(t, i, st) for st in (1, 2, 3)]
            if delta > 0:
                winners.append((name, hand_cents.get(i, 0)))
        # the bet nobody matched went back to its owner: it was never in the pot (v3)
        uncalled_rec = None
        for u_seat, u_chips in t.uncalled.items():  # (one seat at most)
            uncalled_rec = {"seat": int(u_seat), "cents": chips_to_cents(u_chips, t.bb_cents)}
        returned = sum(int(c) for c in t.uncalled.values())
        pot_cents = chips_to_cents(sum(commit) - returned, t.bb_cents)
        actions = _history_entries(raw, t.bb_cents)
        for k, a in enumerate(actions):  # who decided: the player, the clock, or the network
            if k < len(t.hand_actions):
                mark = t.bot_marks.get(k)  # "auto" / "assist" (homegame_bot)
                a["auto"] = not t.hand_actions[k][3] and not mark
                if mark:
                    a["bot"] = mark
        # who paid whom (runout.money_flows), by user id
        flow_holes: list[list[int] | None] = [None] * t.num_seats
        for i in range(t.num_seats):
            if mask[i] and i < len(all_holes) and not (folded[i] if i < len(folded) else True):
                flow_holes[i] = [int(c) for c in all_holes[i]]
        fold_full = [bool(folded[i]) if i < len(folded) else True for i in range(t.num_seats)]
        flows = money_flows(
            (commit + [0] * t.num_seats)[: t.num_seats], fold_full, flow_holes,
            full_a, full_b, t.button,
        )
        record = {
            "v": HAND_RECORD_VERSION,
            "hand_no": int(t.hand_no),
            "ended_at": pub._now(),
            "variant": _norm_game(t.variant),
            "hole_count": int(t.hole_count),
            "button": int(t.button),
            "num_seats": int(t.num_seats),
            "bb_cents": int(t.bb_cents),
            "ante_cents": int(t.ante_cents),
            "pot_cents": pot_cents,
            **({"uncalled": uncalled_rec} if uncalled_rec else {}),
            "showdown": bool(t.showdown_reveal),
            "board_a": full_a[:n_board],
            "board_b": full_b[:n_board],
            "burns": played_burns,
            # an all-in runout's equities, street by street, as the table showed them
            # (the replayer and the text history show them too; owner, 2026-09-29)
            **_record_equities(t),
            "actions": actions,
            "ante_chips": int(t.ante_chips),
            "bb_chips": BB_CHIPS,
            "flows": [
                {"from": int(a), "to": int(b), "cents": chips_to_cents(int(v), t.bb_cents)}
                for (a, b), v in sorted(flows.items())
            ],
            # filled in by the background grader; a game without a network is
            # never graded (an empty list, not "still being worked out")
            "grades": None if t.game["graded"] else [],
            "fair": (
                {"hand_id": t.fair_hand.hand_id,
                 "contributors": [st for st, _ in t.fair_hand.locked],
                 "voids": len((t.fair_hand_meta or {}).get("voids") or [])}
                if t.fair_hand is not None else None
            ),
            "awards": [
                {
                    "board": a.get("board"),
                    "winners": [int(w) for w in (a.get("winners") or [])],
                    "cents": chips_to_cents(int(a.get("chips") or 0), t.bb_cents),
                    "uncontested": bool(a.get("uncontested")),
                    "labels": {
                        str(k): (v or {}).get("label")
                        for k, v in (a.get("combos") or {}).items()
                    },
                }
                for a in (t.pot_awards or [])
            ],
            "seats": seats,
        }
        uid_of = {s["seat"]: s["user_id"] for s in seats}
        t.unsaved.append(("hand", {
            "record": record,
            "results": [
                {"user_id": s["user_id"], "delta_cents": s["delta_cents"],
                 "delta_chips": int(payouts[s["seat"]]) if s["seat"] < len(payouts) else 0,
                 "shown": s["shown"]}
                for s in seats
            ],
            "flows": [(uid_of[a], uid_of[b], int(v)) for (a, b), v in flows.items()
                      if a in uid_of and b in uid_of and v > 0],
            "job": _grading_job(t, mask, uid_of),
        }))
        t.pending_result = {
            "hand_no": int(t.hand_no),
            # Net winners; a hand where everyone got their money back (one
            # board each) has none.
            "text": " · ".join(f"{n} wins {_fmt_cents(c)}" for n, c in winners)
            or f"pot of {_fmt_cents(pot_cents)} split evenly",
        }
    except Exception:  # noqa: BLE001
        logger.exception("homegame hand record failed (table %s)", t.game_id)


# --- background grading ------------------------------------------------------------



def _grading_job(t: LiveTable, mask: list[bool], uid_of: dict[int, int]) -> dict[str, Any] | None:
    """What the grader needs to replay a finished hand (plain values — the worker
    never touches the live table), or None when it is not graded. Only games the
    network knows are graded: there is no PLO6 network (yet)."""
    if not GRADING_ON or not t.game["graded"] or not t.hand_actions or not (t.hand_seed or t.hand_deck):
        return None
    job = {
        "game_id": t.game_id, "hand_no": int(t.hand_no),
        "variant": _norm_game(t.variant),
        "deck": list(t.hand_deck or []),
        "button": int(t.button), "num_seats": int(t.num_seats),
        "stacks": [int(x) for x in (t.hand_start_stacks or [])],
        "ante": int(t.ante_chips), "mask": [bool(x) for x in mask],
        "actions": [list(a) for a in t.hand_actions], "uid_of": {str(k): v for k, v in uid_of.items()},
    }
    if not job["deck"]:
        job["seed"] = int(t.hand_seed)  # (memory only: a seed is never persisted)
    return job


# The grading queue (OPS-016): bounded in memory; a job with a deck is ALSO in
# homegame_grade_jobs from the moment its hand is saved until its grades are, so
# neither a restart nor a full queue loses it — the grader sweeps the table for
# jobs it does not hold (at start, and every GRADE_SWEEP_S while idle).
GRADE_QUEUE_MAX = 200
GRADE_MAX_ATTEMPTS = 3
GRADE_SWEEP_S = 30.0


def _queue_grading(job: dict[str, Any]) -> None:
    key = (str(job["game_id"]), int(job["hand_no"]))
    with CTX.grade_lock:
        if key in CTX.grade_queued:
            return
        if len(CTX.grade_queued) >= GRADE_QUEUE_MAX:
            if not job.get("deck"):  # memory only: it cannot wait for a sweep
                logger.warning("homegame grading queue full: hand %s #%s not graded", *key)
                _store_grades(job, [], note="not graded (the grading queue was full)")
            return
        CTX.grade_queued.add(key)
    CTX.grade_q.put(job)
    _start_grader()


def _sweep_grade_jobs() -> int:
    """Queue saved jobs the grader does not hold yet (after a restart, or once a
    full queue has drained). Returns how many were queued."""
    n = 0
    if _grading_model() is None:
        return 0  # nothing to grade with yet: the jobs wait
    for r in pub.DB.q(
        "SELECT job FROM homegame_grade_jobs ORDER BY created_at, game_id, hand_no LIMIT ?",
        (GRADE_QUEUE_MAX,),
    ):
        try:
            job = json.loads(r["job"])
        except ValueError:
            continue
        with CTX.grade_lock:
            if (str(job["game_id"]), int(job["hand_no"])) in CTX.grade_queued:
                continue
        _queue_grading(job)
        n += 1
    return n


def _start_grader() -> None:
    if CTX.grade_thread is not None and CTX.grade_thread.is_alive():
        return
    CTX.grader_stop.clear()
    CTX.grade_thread = threading.Thread(target=_grader_loop, args=(CTX,), name="homegame-grader", daemon=True)
    CTX.grade_thread.start()


def _grader_loop(ctx: HomeGames | None = None) -> None:
    ctx = ctx or CTX  # (a swapped context never steals a running grader)
    next_sweep = 0.0  # (at once: jobs saved before a restart)
    while not ctx.grader_stop.is_set():
        if time.monotonic() >= next_sweep and ctx.grade_q.empty():
            next_sweep = time.monotonic() + GRADE_SWEEP_S
            try:
                _sweep_grade_jobs()
            except Exception:  # noqa: BLE001
                logger.exception("homegame grading sweep")
        try:
            job = ctx.grade_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if job is None:
            ctx.grade_q.task_done()
            return
        try:
            _grade_job(job)
        finally:
            with ctx.grade_lock:
                ctx.grade_queued.discard((str(job["game_id"]), int(job["hand_no"])))
            ctx.grade_q.task_done()


def _grade_job(job: dict[str, Any]) -> None:
    """Grade one hand. A replay that fails is tried again later (up to
    GRADE_MAX_ATTEMPTS: a model being reloaded, say); then the hand is settled
    with no marks and a note — never left "still being worked out" (OPS-016)."""
    try:
        grades = grade_hand(job)
    except NoGradingModel:
        return  # (saved jobs wait for a real model; see NoGradingModel)
    except Exception:  # noqa: BLE001 — cosmetic; never take the table down
        logger.exception("homegame grading failed (%s #%s)", job.get("game_id"), job.get("hand_no"))
        attempts = GRADE_MAX_ATTEMPTS
        if job.get("deck"):
            with pub.DB.transaction():
                pub.DB.q(
                    "UPDATE homegame_grade_jobs SET attempts=attempts+1 WHERE game_id=? AND hand_no=?",
                    (job["game_id"], int(job["hand_no"])),
                )
                row = pub.DB.one(
                    "SELECT attempts FROM homegame_grade_jobs WHERE game_id=? AND hand_no=?",
                    (job["game_id"], int(job["hand_no"])),
                )
            attempts = int(row["attempts"]) if row is not None else GRADE_MAX_ATTEMPTS
        if attempts >= GRADE_MAX_ATTEMPTS:
            _store_grades(job, [], note="could not be graded")
        return
    _store_grades(job, grades)


def grade_hand(job: dict[str, Any], model: Any = None) -> list[dict[str, Any]]:
    """Replay the hand in a full-observation env and score every PLAYER decision
    with the Trainer's scorer against the served PLO5 model's node distribution
    (each node from the actor's own seat — exactly what Study would show them)."""
    from plo5bp.sizing import PLO_ANCHOR_SPEC
    from plo5bp.ui import trainer as tr

    if not GAMES[_norm_game(job.get("variant"))]["graded"]:
        return []  # (never queued; a PLO5 network must never grade another game)
    if model is None:
        model = _grading_model()
    if model is None:
        raise NoGradingModel()
    device = next(model.parameters()).device
    cfg = GameConfig(
        num_seats=job["num_seats"], starting_stack=0,
        starting_stacks=tuple(job["stacks"]), ante=job["ante"], bb=BB_CHIPS,
        variant=VARIANT_PLO5, cover_short_bets=True,  # (a bet into a sub-1bb stack: GameConfig)
    )
    env = BombPotEnv(cfg, ev_runout_samples=0)  # FULL observation (not "minimal")
    if job.get("deck"):
        obs, info = env.reset_with_deck(job["deck"], job["button"], in_hand_mask=job["mask"])
    else:
        obs, info = env.reset(job["seed"], job["button"], in_hand_mask=job["mask"])
    spec = getattr(model, "anchor_spec", PLO_ANCHOR_SPEC)
    out: list[dict[str, Any]] = []
    for k, (seat, gate, chips, by_player) in enumerate(job["actions"]):
        if env.is_terminal() or info is None or info.actor is None:
            break
        if int(info.actor) != int(seat):
            raise RuntimeError(f"replay diverged at action {k}: actor {info.actor} != {seat}")
        # The table caps a bet at the pot and the bettor's stack only (``reach_cap``
        # off); the network knows the rule it was trained on, where a bet also stops at
        # what the deepest opponent can still put in. A bigger bet IS that bet — the
        # rest came back uncalled — so it is replayed and graded as that.
        chips = int(chips)
        if int(gate) == GATE_RAISE and int(info.max_raise_chips) > 0:
            chips = max(int(info.min_raise_chips), min(chips, int(info.max_raise_chips)))
        if by_player:
            dist = tr.attach_unclamped_brackets(
                tr.compute_node_distribution(model, device, obs, info), info, spec
            )
            if dist["head_version"] >= 2:
                sc = tr.score_move_v2(dist, int(gate), int(chips))
            else:
                sc = tr.score_move(
                    dist["gate_probs"], dist["alpha"], dist["beta"],
                    dist["min_chips"], dist["max_chips"], int(gate), int(chips),
                )
            out.append({
                "i": k, "seat": int(seat), "score": round(float(sc["score"]), 1),
                "cat": str(sc["category"]),
            })
        obs, _, _done, info = env.step_hybrid(int(gate), int(chips))
    return out


@contextmanager
def _edit_hand_record(game_id: str, hand_no: int) -> Iterator[dict[str, Any] | None]:
    """Read-change-write ONE stored hand record atomically (OPS-007: ``/show``
    and the grader each read the JSON, changed one field and wrote it all back,
    so a Show landing just as the grades did could erase the other's change).
    Yields the record (None if there is none) inside a transaction — the
    database lock is held throughout — and writes it back when the block ends."""
    with pub.DB.transaction():
        row = pub.DB.one(
            "SELECT summary FROM homegame_hands WHERE game_id=? AND hand_no=?",
            (game_id, int(hand_no)),
        )
        rec = json.loads(row["summary"]) if row is not None else None
        yield rec
        if rec is not None:
            pub.DB.q(
                "UPDATE homegame_hands SET summary=? WHERE game_id=? AND hand_no=?",
                (json.dumps(rec, separators=(",", ":")), game_id, int(hand_no)),
            )


def _store_grades(job: dict[str, Any], grades: list[dict[str, Any]], *, note: str | None = None) -> None:
    """The hand's marks (and each player's accuracy) land and its saved job is
    done — in one transaction."""
    per_seat: dict[int, list[float]] = {}
    for g in grades:
        per_seat.setdefault(int(g["seat"]), []).append(float(g["score"]))
    uid_of = {int(k): v for k, v in (job.get("uid_of") or {}).items()}
    with _edit_hand_record(job["game_id"], job["hand_no"]) as rec:
        if rec is not None:
            rec["grades"] = grades
            if note:
                rec["grades_note"] = note
            for seat, scores in per_seat.items():
                uid = uid_of.get(seat)
                if uid is None:
                    continue
                pub.DB.q(
                    "UPDATE homegame_hand_results SET acc_sum=?, acc_n=? "
                    "WHERE game_id=? AND hand_no=? AND user_id=?",
                    (float(sum(scores)), len(scores), job["game_id"], job["hand_no"], int(uid)),
                )
        pub.DB.q("DELETE FROM homegame_grade_jobs WHERE game_id=? AND hand_no=?",
                 (job["game_id"], int(job["hand_no"])))


def wait_for_grading(timeout: float = 20.0) -> bool:
    """Block until the grader has drained its queue (tests; the shutdown hook)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if CTX.grade_q.unfinished_tasks == 0:
            return True
        time.sleep(0.05)
    return False


def _flush_result_locked(t: LiveTable) -> None:
    """Release the finished hand's result line — only once its runout has
    finished revealing (review 2026-09-20 G11: nothing runs ahead of it)."""
    pending = t.pending_result
    if not pending or _runout_blocking(t):
        return
    t.pending_result = None
    _emit(t, "win", f"Hand #{pending['hand_no']}: {pending['text']}")
    t.rev += 1


def _capture_rabbit(t: LiveTable, played_len: int) -> None:
    """Fold-out rabbit OR multiway all-in runout (delayed streets + awards)."""
    raw = _obs_dict(t.env)
    folded = [bool(x) for x in raw.get("folded", [])]
    alive: list[int] = []
    for i, f in enumerate(folded):
        if f:
            continue
        if t.in_hand_mask and i < len(t.in_hand_mask) and not t.in_hand_mask[i]:
            continue
        alive.append(i)
    full_a = [int(c) for c in (raw.get("board_a") or [])]
    full_b = [int(c) for c in (raw.get("board_b") or [])]
    t.rabbit_full_a = full_a
    t.rabbit_full_b = full_b
    t.rabbit_burns = _all_burns(t)
    t.leftover_stacks = [int(x) for x in (raw.get("stacks") or [])]
    t.terminal_pot = int(raw.get("pot") or 0)
    t.terminal_commit = [int(x) for x in (raw.get("total_commit") or [])]
    t.pot_awards = []
    t.pots = []
    t.equity_by_len = {}
    t.runout_active = False
    # The bet nobody matched goes back now, the way any card room does it: the
    # bettor's stack has it from here on, the pot never had it, and no award pays it
    # (it used to be a one-player "side pot" the bettor won at the showdown).
    t.uncalled = {}
    unc = uncalled_bet(t.terminal_commit)
    if unc is not None:
        u_seat, u_chips = unc
        t.uncalled = {int(u_seat): int(u_chips)}
        if u_seat < len(t.leftover_stacks):
            t.leftover_stacks[u_seat] += int(u_chips)
        t.terminal_pot = max(0, t.terminal_pot - int(u_chips))
        _emit(t, "uncalled", f"Uncalled {_fmt_cents(chips_to_cents(u_chips, t.bb_cents))} "
              f"returned to {_seat_name(t, u_seat)}")
    if len(alive) <= 1:
        if len(alive) == 1 and len(full_a) > played_len:
            # Host can switch rabbit hunting off: the undealt streets then
            # simply stay hidden (``rabbit_shown`` never becomes True).
            t.rabbit_available = bool(t.allow_rabbit)
            t.rabbit_shown = False
            t.rabbit_played_len = int(played_len)
        else:
            t.rabbit_available = False
            t.rabbit_shown = True
            t.rabbit_played_len = len(full_a)
        return
    t.rabbit_available = False
    t.rabbit_shown = True
    t.runout_active = True
    t.runout_start_len = max(3, min(int(played_len), len(full_a) or 5))
    t.runout_started_mono = time.monotonic()
    # (PLO67: the plan needs this hand's burns — captured above — and how many
    # hands a red one deals to)
    t.runout_plan = _make_runout_plan(t, len(alive))
    all_holes = t.env.all_hole_cards() if t.env is not None else []
    holes: list[list[int] | None] = [None] * t.num_seats
    for i in alive:
        if i < len(all_holes):
            holes[i] = [int(c) for c in all_holes[i]]  # (build_awards: the hands at the river)
    folded_full = [True] * t.num_seats
    for i in alive:
        folded_full[i] = False
    commit = list(t.terminal_commit)
    if len(commit) < t.num_seats:
        commit.extend([0] * (t.num_seats - len(commit)))
    for u_seat, u_chips in t.uncalled.items():  # (returned above: not in any pot)
        commit[u_seat] -= u_chips
    t.pot_awards = build_awards(
        holes,
        folded_full,
        commit[: t.num_seats],
        full_a,
        full_b,
        t.button,
    )
    # The award steps carry the index of the pot they pay from (``build_awards``).
    t.pots = _named_pots(display_pots(commit[: t.num_seats], folded_full))
    _compute_runout_equities(t, alive, full_a, full_b)


def _named_pots(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``runout.display_pots`` named the way the table talks about them
    (ClubGG-style, 2026-09-22): deepest first; the LAST is the main pot, the ones
    before it side pots numbered from the main pot up. Layers the same players
    can win are ONE pot (a fold is dead money, not a side pot)."""
    n = len(groups)
    return [
        {"index": k, "label": "Main pot" if k == n - 1 else f"Side pot {n - 1 - k}",
         "chips": int(g["chips"]), "eligible": list(g["eligible"]), "level": int(g["level"])}
        for k, g in enumerate(groups)
    ]


def _live_pots(raw: dict[str, Any], in_hand: list[bool], n: int) -> list[dict[str, Any]]:
    """The pots WHILE a hand is played (2026-09-25; they used to appear only at
    the showdown): what is already in the middle — the antes and the finished
    streets — split exactly the way the showdown will pay it, so a side pot
    shows the moment someone all-in for less has been called past, and a
    fold's chips stay dead money in the pot they went into. This street's bets
    are still in front of the players and join the pots when the betting round
    closes, as on any poker site. Same names and order as ``t.pots``, so the
    runout takes over without a pot changing place."""
    total = [int(x) for x in (raw.get("total_commit") or [])]
    street = [int(x) for x in (raw.get("street_commit") or [])]
    folded = [bool(x) for x in (raw.get("folded") or [])]
    middle = [
        max(0, (total[i] if i < len(total) else 0) - (street[i] if i < len(street) else 0))
        for i in range(n)
    ]
    gone = [
        (folded[i] if i < len(folded) else True) or not (in_hand[i] if i < len(in_hand) else False)
        for i in range(n)
    ]
    return _named_pots(display_pots(middle, gone))


def _hand_on(t: LiveTable, seat: int, cards: list[int], board_len: int) -> list[int]:
    """The hole cards ``seat`` held when the boards showed ``board_len`` cards: all
    of them, except in PLO67 — there only the ones dealt by then (a red burn
    still to come deals the rest). ``cards`` in deal order."""
    if not t.burns or board_len >= 5:
        return list(cards)
    return list(cards)[: _hole_count_on(t, seat, max(3, board_len) - 2)]


def _street_equities(t: LiveTable, holes: dict[int, list[int]], ba: list[int], bb: list[int],
                     seed: int) -> dict[int, dict[str, float]]:
    """{seat: {"a": share, "b": share}} with the boards at ``ba`` / ``bb``: PLO5 and
    PLO6 enumerate every runout of each board (``runout.board_equities``);
    PLO67 samples whole runouts the way the game deals them — burns still to
    come, the cards red ones deal, the boards (``runout.plo67_equities``, Rust),
    with the burns up so far dead."""
    if t.burns:
        return plo67_equities(holes, ba, bb, list(t.rabbit_burns)[: max(0, len(ba) - 2)],
                              samples=PLO67_EQ_SAMPLES, seed=seed)
    return board_equities(holes, ba, bb, seed=seed)


def _compute_runout_equities(
    t: LiveTable, alive: list[int], full_a: list[int], full_b: list[int]
) -> None:
    """Per-board equities for every street the runout will show, each from the
    ALIVE hands as they were on that street (HGB-024: one loop for every game).

    (review 2026-09-20 G3) ONCE per all-in hand, here, from the ALIVE seats'
    holes. It used to run inside every poll under the table lock, from the
    holes visible to that viewer — so a folded viewer's own cards entered as
    a contender (wrong numbers) and the per-viewer cache key made
    alternating polls recompute 0.5-2.9 s of pure Python each time. With the
    per-board marginal enumeration in runout.py the whole thing is ~0.1 s
    worst case (a flop all-in), a few ms from the turn; PLO67's sampler ~10 ms
    a street. Cosmetic: a failure here must never stop the hand from finishing."""
    t.equity_by_len = {}
    if len(alive) < 2 or t.env is None:
        return
    try:
        all_holes = t.env.all_hole_cards()
        for n in range(max(3, int(t.runout_start_len or 3)), 6):
            ba, bb = full_a[:n], full_b[:n]
            holes = {i: _hand_on(t, i, [int(c) for c in all_holes[i]], n) for i in alive if i < len(all_holes)}
            seed = zlib.crc32(f"{t.game_id}:{t.hand_no}:{n}".encode())
            t.equity_by_len[(len(ba), len(bb))] = _street_equities(t, holes, ba, bb, seed)
    except Exception:  # noqa: BLE001
        logger.exception("runout equities failed (table %s)", t.game_id)
        t.equity_by_len = {}


def _all_burns(t: LiveTable) -> list[int]:
    """PLO67: all three burns of the current hand (a reveal accessor — the
    view trims them to the streets shown). [] for every other game."""
    if not t.burns or t.env is None:
        return []
    return t.env.all_burns()


def _hole_count_on(t: LiveTable, seat: int, street: int) -> int:
    """PLO67: the hole cards ``seat`` held on ``street`` (1 flop, 2 turn,
    3 river) — the engine's own count (a seat's extras are a prefix of the red
    burns it was in the hand for)."""
    return t.env.hole_count_on(seat, street)


def _rabbit_locked(t: LiveTable, uid: int) -> None:
    _require_member(t, uid)
    if t.phase != "showdown" or not t.rabbit_available or t.rabbit_shown:
        raise HTTPException(status_code=400, detail="no rabbit hunt available")
    t.rabbit_shown = True
    t.rev += 1


def _quantize_raise(t: LiveTable, chips: int, committed: int, lo: int, hi: int) -> int:
    """Snap a raise so the street total lands on a whole cent when the stake
    allows and the legal window has room (review 2026-09-20 G10). The window
    ends stay exact — an all-in is the whole stack, sub-cent dust included."""
    cpc = chips_per_cent(t.bb_cents)
    if cpc <= 1 or not (lo < chips < hi):
        return chips
    to = committed + chips
    snapped = ((to + cpc // 2) // cpc) * cpc
    if snapped - committed < lo:
        snapped += cpc
    if snapped - committed > hi:
        snapped -= cpc
    by = snapped - committed
    return by if lo <= by <= hi else chips


def _apply_action_locked(
    t: LiveTable, gate: int, raise_chips: int, *, _resume: bool = True,
    by_player: bool = False,
) -> None:
    """Apply one action for the CURRENT actor (turn ownership is the
    caller's business: `_act_locked` for players, the away/clock paths for
    auto-actions)."""
    if t.phase != "in_hand" or t.env is None or t.info is None:
        raise HTTPException(status_code=400, detail="no hand in progress")
    gm = t.info.gate_mask
    if not bool(gm[gate]):
        raise HTTPException(status_code=400, detail="illegal action")
    chips = int(raise_chips) if gate == GATE_RAISE else 0
    raw = _obs_dict(t.env)
    if gate == GATE_RAISE:
        min_c = int(t.info.min_raise_chips)
        max_c = int(t.info.max_raise_chips)
        if min_c == 0 and max_c > 0:
            chips = max_c
        else:
            chips = max(min_c, min(max_c, chips))
            actor = raw.get("actor")
            if actor is not None:
                committed = int(raw["street_commit"][int(actor)])
                chips = _quantize_raise(t, chips, committed, min_c, max_c)
    pre_len = len(list(raw.get("board_a") or []))
    actor_raw = raw.get("actor")
    try:
        _, _, done, info = t.env.step_hybrid(gate, chips)
    except Exception as e:  # noqa: BLE001 — engine rejects illegal amounts
        raise HTTPException(status_code=400, detail=str(e)) from e
    # (seat, gate, chips, the player's own decision?) — what the grader replays.
    # Clock / away / host auto-actions are recorded but never graded.
    t.hand_actions.append(
        (int(actor_raw) if actor_raw is not None else -1, int(gate), int(chips), bool(by_player))
    )
    # Charge the time bank for THIS decision before the decision id moves on.
    _settle_bank_locked(t, int(actor_raw) if actor_raw is not None else None)
    t.info = info
    t.action_seq += 1
    t.rev += 1
    if done:
        # (the runout is set up BEFORE the hand is finished and its result written:
        # ``_unpublished_guard`` relies on that order, see there)
        _capture_rabbit(t, pre_len)
        _finish_hand_locked(t)
    else:
        t.phase = "in_hand"
        if _resume:
            _resume_turn_locked(t)


def _act_locked(t: LiveTable, uid: int, gate: int, raise_chips: int) -> None:
    _require_open(t)
    if t.phase != "in_hand" or t.env is None or t.info is None:
        raise HTTPException(status_code=400, detail="no hand in progress")
    actor = t.env.current_actor()
    seat = t.seat_of(uid)
    if actor is None or seat is None or actor != seat:
        raise HTTPException(status_code=409, detail="not your turn")  # (a race, not a bad request — HGB-022)
    me = t.seats[seat]
    if me is not None and me.bot_mode == "auto":
        # (the network plays this seat: its owner switches it off to act — homegame_bot)
        raise HTTPException(status_code=409, detail="the network is playing your seat — switch it off to act yourself")
    # A move made with the network's suggestion on the screen is not the player's own:
    # marked for the table's history and never graded (homegame_bot).
    assisted = me is not None and me.bot_mode == "assist"
    idx = len(t.hand_actions)
    if assisted:
        t.bot_marks[idx] = "assist"
    try:
        _apply_action_locked(t, gate, raise_chips, by_player=not assisted)
    except BaseException:
        t.bot_marks.pop(idx, None)
        raise
    if me is not None:
        me.timeouts = 0  # acted for themselves: the timeout streak is over


def _dealt_in(t: LiveTable, i: int | None) -> bool:
    """Seat ``i`` holds cards in the hand on the table (played or still
    revealing)."""
    return bool(
        i is not None
        and _hand_busy(t)
        and t.in_hand_mask
        and i < len(t.in_hand_mask)
        and t.in_hand_mask[i]
    )


def _sync_idle_stack(t: LiveTable, i: int | None) -> None:
    """A seat that is NOT in the current hand changed its chips mid-hand (sat
    down, topped up): the hand-start snapshot must follow, because the hand's
    end writes ``start + payout`` and the runout-time ledger reads it."""
    if i is None or not _hand_busy(t):
        return
    p = t.seats[i]
    while len(t.hand_start_stacks) <= i:
        t.hand_start_stacks.append(0)
    t.hand_start_stacks[i] = int(p.stack_chips) if p is not None else 0


def _sit_locked(
    t: LiveTable, user: Any, seat: int, buyin_cents: int, *, trust: bool = False
) -> None:
    _require_open(t)
    # (2026-09-21) Sitting down no longer waits for the hand to end: with the
    # server dealing every few seconds the between-hands window was too short
    # to buy in. An empty seat is never part of the hand in progress, so the
    # newcomer just waits for the next deal.
    if seat < 0 or seat >= t.num_seats:
        raise HTTPException(status_code=400, detail="invalid seat")
    if t.seats[seat] is not None:
        raise HTTPException(status_code=409, detail="seat taken")
    existing = t.seat_of(int(user["id"]))
    if existing is not None:
        raise HTTPException(status_code=400, detail="already seated")
    _check_amount(t, buyin_cents)
    _check_buyin_limits(t, buyin_cents, 0)
    chips = cents_to_chips(buyin_cents, t.bb_cents)
    prev = pub.DB.one(
        "SELECT * FROM homegame_players WHERE game_id=? AND user_id=?",
        (t.game_id, int(user["id"])),
    )
    leftover = int(prev["leftover_cents"]) if prev else 0
    prev_buyin = int(prev["buyin_cents"]) if prev else 0
    if prev_buyin + buyin_cents > MAX_CENTS:
        raise HTTPException(status_code=400, detail="buy-in limit reached")
    holder = next(
        (r for r in t.requests if r["kind"] == "sit" and r["seat"] == seat
         and r["user_id"] != int(user["id"])), None,
    )
    if holder is not None:
        raise HTTPException(
            status_code=400, detail=f"that seat is reserved for {holder['name']}"
        )
    if t.auto_stack_mode == "host":
        auto = int(t.auto_stack_all_cents or 0)
    else:
        auto = int(prev["auto_stack_cents"] or 0) if prev is not None else 0
    if t.topup_mode == "host":
        top = (int(t.topup_all_target_cents or 0), int(t.topup_all_below_cents or 0))
    elif prev is not None:
        top = (int(prev["topup_target_cents"] or 0), int(prev["topup_below_cents"] or 0))
    else:
        top = (0, 0)
    p = Seat(
        user_id=int(user["id"]),
        name=_display_name(user, t.club_id),
        stack_chips=chips,
        sitting_out=False,
        buyin_cents=prev_buyin + buyin_cents,
        leftover_cents=leftover,
        auto_stack_cents=auto,
        time_bank_left=float(max(0, int(t.time_bank_secs or 0))),
        trusted=bool(trust) or (bool(prev["trusted"]) if prev is not None else False),
        topup_target_cents=top[0],
        topup_below_cents=top[1],
    )
    if _dealt_in(t, seat):
        # (cannot happen: a dealt seat stays occupied until its hand is over)
        raise HTTPException(status_code=400, detail="wait for the hand to finish")
    with _mutation(t):
        t.seats[seat] = p
        _ledger_add(t, p.user_id, "buyin" if not prev else "rebuy", buyin_cents)
        _persist_player(t, seat, p, True)
    _sync_idle_stack(t, seat)
    _emit(t, "join", f"{p.name} sat down with {_fmt_cents(buyin_cents)}", seat=seat)


def _more_than_ante(t: LiveTable, cents: int) -> bool:
    """Would a stack worth ``cents`` be dealt in (it must hold more than the ante)?"""
    return int(cents) > int(t.ante_cents) and cents_to_chips(int(cents), t.bb_cents) > t.ante_chips


def _check_amount(t: LiveTable, cents: int, stack_cents: int = 0, *, what: str = "buy-in") -> None:
    """The floor and ceiling of any money brought to the table: at least 1 bb,
    no absurd amount, and — so the seat can actually be dealt in (OPS-005: a $1
    buy-in at a $3-ante table seated someone who was never dealt a hand) —
    enough that the stack holds more than the ante afterwards."""
    cents = int(cents)
    if cents < t.bb_cents:
        raise HTTPException(status_code=400, detail=f"{what} must be at least 1 bb")
    if cents > MAX_CENTS:
        raise HTTPException(status_code=400, detail=f"that {what} is too large")
    if not _more_than_ante(t, int(stack_cents) + cents):
        raise HTTPException(
            status_code=400,
            detail=(f"{what} must be more than the ante ({_fmt_cents(int(t.ante_cents))})" if not stack_cents
                    else f"add enough to have more than the ante ({_fmt_cents(int(t.ante_cents))})"),
        )


def _check_buyin_limits(
    t: LiveTable, amount_cents: int, stack_cents: int, *, fresh: bool = True
) -> None:
    """Host-set buy-in window (0 = no limit). A top-up may not take the stack
    past the maximum; the minimum applies to a fresh seat only."""
    lo, hi = int(t.min_buyin_cents or 0), int(t.max_buyin_cents or 0)
    if fresh and lo > 0 and amount_cents < lo:
        raise HTTPException(
            status_code=400, detail=f"minimum buy-in is {_fmt_cents(lo)}"
        )
    if hi > 0 and stack_cents + amount_cents > hi:
        if not fresh:
            raise HTTPException(
                status_code=400,
                detail=f"you can top up to {_fmt_cents(hi)} at most",
            )
        raise HTTPException(
            status_code=400, detail=f"maximum buy-in is {_fmt_cents(hi)}"
        )


# --- host approval of buy-ins ------------------------------------------------


def _is_trusted(t: LiveTable, uid: int) -> bool:
    if int(uid) == int(t.host_user_id):
        return True
    p = t.player(int(uid))
    if p is not None:
        return bool(p.trusted)
    row = pub.DB.one(
        "SELECT trusted FROM homegame_players WHERE game_id=? AND user_id=?",
        (t.game_id, int(uid)),
    )
    return bool(row is not None and int(row["trusted"] or 0))


def _needs_approval(t: LiveTable, uid: int) -> bool:
    return bool(t.approve_buyins) and not _is_trusted(t, uid)


def _request_locked(t: LiveTable, user: Any, kind: str, seat: int | None, cents: int) -> None:
    """Queue a buy-in (``sit``) or top-up (``rebuy``) for the host. Validated
    like the real thing so an approval rarely bounces; one request per player
    (a new one replaces it); a sit request holds its seat."""
    _require_open(t)
    uid = int(user["id"])
    if kind == "sit":
        _check_amount(t, cents)
        if seat is None or seat < 0 or seat >= t.num_seats:
            raise HTTPException(status_code=400, detail="invalid seat")
        if t.seats[seat] is not None:
            raise HTTPException(status_code=409, detail="seat taken")
        if t.seat_of(uid) is not None:
            raise HTTPException(status_code=400, detail="already seated")
        if any(r["kind"] == "sit" and r["seat"] == seat and r["user_id"] != uid for r in t.requests):
            raise HTTPException(status_code=400, detail="that seat is reserved")
        _check_buyin_limits(t, cents, 0)
    else:
        p = t.player(uid)
        if p is None:
            raise HTTPException(status_code=400, detail="not seated")
        seat = t.seat_of(uid)
        have = chips_to_cents(p.stack_chips, t.bb_cents) + int(p.queued_topup_cents or 0)
        _check_amount(t, cents, have, what="top-up")
        _check_buyin_limits(t, cents, have, fresh=False)
    # The same request again (a double tap, an impatient second tap) is not news:
    # the host used to get one "asks to" toast per tap, stacked over the table.
    if any(r["user_id"] == uid and r["kind"] == kind and r["seat"] == seat and int(r["amount_cents"]) == int(cents)
           for r in t.requests):
        return
    t.requests = [r for r in t.requests if r["user_id"] != uid]
    if len(t.requests) >= MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="too many pending requests")
    t.request_seq += 1
    name = _display_name(user, t.club_id)
    t.requests.append({
        "id": t.request_seq, "user_id": uid, "name": name, "kind": kind,
        "seat": seat, "amount_cents": int(cents), "ts": time.time(),
    })
    t.rev += 1
    _emit(
        t, "request",
        f"{name} asks to {'buy in for' if kind == 'sit' else 'add'} {_fmt_cents(cents)}",
    )


def _cancel_request_locked(t: LiveTable, uid: int) -> None:
    n = len(t.requests)
    t.requests = [r for r in t.requests if r["user_id"] != int(uid)]
    if len(t.requests) != n:
        t.rev += 1


def _expire_requests_locked(t: LiveTable) -> None:
    """A request whose player has left the page lapses (and frees its seat);
    so does one the table no longer needs (approval off / player now trusted)."""
    if not t.requests:
        return
    now = time.monotonic()
    keep = []
    for r in t.requests:
        gone = now - t.seen.get(r["user_id"], now) > REQUEST_TTL_ABSENT_S
        if gone or t.status != "open":
            continue
        keep.append(r)
    if len(keep) != len(t.requests):
        t.requests = keep
        t.rev += 1


def _resolve_request_locked(
    t: LiveTable, uid: int, req_id: int, approve: bool, trust: bool,
    amount_cents: int | None = None,
) -> None:
    """Approve / decline a buy-in request. ``amount_cents`` lets the host approve
    a DIFFERENT amount than the one asked for (asked $150, seated with $80): the
    player is told, and the usual buy-in limits apply to the new amount.

    (HGB-001) The chips move FIRST; only then is the request taken off the list
    and the approval announced. A buy-in that no longer fits (the host changed
    the limits since) is an error for the host and the request stays — it used
    to vanish after the table had already seen "approved"."""
    _require_open(t)
    _require_host(t, uid, "approve buy-ins")
    req = next((r for r in t.requests if r["id"] == int(req_id)), None)
    if req is None:
        raise HTTPException(status_code=404, detail="that request is gone")
    asked = int(req["amount_cents"])
    cents = asked if (not approve or amount_cents is None) else int(amount_cents)
    if not approve:
        t.requests = [r for r in t.requests if r["id"] != req["id"]]
        t.rev += 1
        _emit(t, "request", f"Host declined {req['name']}'s request")
        return
    user = pub._user_by_id(int(req["user_id"]))
    if user is None:
        t.requests = [r for r in t.requests if r["id"] != req["id"]]
        t.rev += 1
        raise HTTPException(status_code=404, detail="that player no longer exists")
    # The request's seat reservation must not block its own sit: it is set aside
    # while the chips move, and put back if they cannot.
    t.requests = [r for r in t.requests if r["id"] != req["id"]]
    try:
        if req["kind"] == "sit":
            _sit_locked(t, user, int(req["seat"]), cents, trust=trust)
        else:
            _topup_locked(t, int(req["user_id"]), cents, queue_ok=True)
            if trust:
                _set_trust_locked(t, uid, int(req["user_id"]), True)
    except BaseException:
        t.requests.append(req)
        t.requests.sort(key=lambda r: r["id"])
        raise
    t.rev += 1
    if cents != asked:
        _emit(t, "request", f"Host approved {req['name']} for {_fmt_cents(cents)} (asked {_fmt_cents(asked)})")


def _let_through_locked(t: LiveTable, host_uid: int, requests: list[dict]) -> None:
    """Approve requests nobody needs to approve any more (approval switched off,
    the player now trusted). One that no longer fits — the limits changed since
    it was asked — is dropped and the table is told why (HGB-001: it used to
    vanish without a word)."""
    for r in requests:
        try:
            _resolve_request_locked(t, host_uid, r["id"], True, False)
        except HTTPException as e:
            if any(x["id"] == r["id"] for x in t.requests):
                t.requests = [x for x in t.requests if x["id"] != r["id"]]
                t.rev += 1
            _emit(t, "request", f"{r['name']}'s request could not go through: {e.detail}")


def _set_trust_locked(t: LiveTable, uid: int, target_uid: int, on: bool) -> None:
    """Trusted players buy in, top up and auto top-up without asking."""
    _require_open(t)
    _require_host(t, uid, "trust a player")
    target_uid = int(target_uid)
    p = t.player(target_uid)
    with _mutation(t):
        if p is not None:
            p.trusted = bool(on)
            _persist_player(t, t.seat_of(target_uid), p, True)
        else:
            row = pub.DB.one(
                "SELECT 1 FROM homegame_players WHERE game_id=? AND user_id=?",
                (t.game_id, target_uid),
            )
            if row is None:
                raise HTTPException(status_code=400, detail="that player has not played here")
            pub.DB.q(
                "UPDATE homegame_players SET trusted=? WHERE game_id=? AND user_id=?",
                (1 if on else 0, t.game_id, target_uid),
            )
    if on:  # whatever they were waiting for goes through now
        _let_through_locked(t, uid, [r for r in t.requests if r["user_id"] == target_uid])


def _topup_locked(t: LiveTable, uid: int, amount_cents: int, *, queue_ok: bool) -> None:
    """Add chips now — or, while the player holds cards, queue them for the end
    of the hand (``queue_ok``; chips behind never change mid-hand)."""
    p = t.player(uid)
    if p is None:
        raise HTTPException(status_code=400, detail="not seated")
    if not (queue_ok and _dealt_in(t, t.seat_of(uid))):
        _rebuy_locked(t, uid, amount_cents)
        return
    _require_open(t)
    have = chips_to_cents(p.stack_chips, t.bb_cents) + int(p.queued_topup_cents or 0)
    _check_amount(t, amount_cents, have, what="top-up")
    _check_buyin_limits(t, amount_cents, have, fresh=False)
    if p.buyin_cents + p.queued_topup_cents + amount_cents > MAX_CENTS:
        raise HTTPException(status_code=400, detail="buy-in limit reached")
    p.queued_topup_cents += int(amount_cents)
    t.rev += 1
    _emit(t, "rebuy", f"{p.name} adds {_fmt_cents(amount_cents)} after this hand")


def _remove_chips_locked(t: LiveTable, uid: int, amount_cents: int, *, queue_ok: bool) -> None:
    """Take chips OFF the table (the host's "ratholing" switch). Whole cents;
    the chips that leave are exactly those cents' worth, credited to the
    player's leftover — the same move set-stack makes when it banks a surplus.
    While the player holds cards it is queued for the end of the hand: chips
    behind never change mid-hand. A player keeps at least an ante + 1 bb, so a
    withdrawal never turns into a silent sit-out; leaving is a separate act."""
    _require_open(t)
    if not t.allow_rathole:
        raise HTTPException(status_code=400, detail="the host does not allow taking chips off the table")
    p = t.player(uid)
    if p is None:
        raise HTTPException(status_code=400, detail="not seated")
    i = t.seat_of(uid)
    amount_cents = int(amount_cents)
    if amount_cents < t.bb_cents:
        raise HTTPException(status_code=400, detail="take off at least 1 bb")
    floor_cents = t.ante_cents + t.bb_cents
    if _dealt_in(t, i):
        if not queue_ok:
            raise HTTPException(status_code=400, detail="wait for the hand to finish")
        # (the hand may change the stack: the amount is re-checked when it lands)
        p.queued_remove_cents = amount_cents
        p.queued_topup_cents = 0
        t.rev += 1
        _emit(t, "cashout", f"{p.name} takes {_fmt_cents(amount_cents)} off the table after this hand", seat=i)
        return
    have = chips_to_cents(int(p.stack_chips), t.bb_cents)
    if have - amount_cents < floor_cents:
        raise HTTPException(
            status_code=400,
            detail=f"you can take off at most {_fmt_cents(max(0, have - floor_cents))} and stay seated — leave the table to cash out",
        )
    with _mutation(t):
        p = t.player(uid)
        assert p is not None
        moved = min(cents_to_chips(amount_cents, t.bb_cents), int(p.stack_chips))
        p.stack_chips -= moved
        p.leftover_cents += amount_cents
        _ledger_add(t, uid, "take_off", amount_cents)
        _persist_player(t, i, p, True)
    _sync_idle_stack(t, i)
    _emit(t, "cashout", f"{p.name} took {_fmt_cents(amount_cents)} off the table", seat=i)


def _apply_queued_topups_locked(t: LiveTable, *, force: bool = False) -> None:
    """Land the top-ups (and withdrawals) that were asked for mid-hand, once the
    hand (and its runout) is over."""
    if not any(p is not None and (p.queued_topup_cents or p.queued_remove_cents) for p in t.seats):
        return
    if _hand_busy(t) and not force:
        return
    for p in t.seats:
        if p is None or not p.queued_remove_cents:
            continue
        cents, p.queued_remove_cents = int(p.queued_remove_cents), 0
        if not t.allow_rathole:
            continue
        have = chips_to_cents(int(p.stack_chips), t.bb_cents)
        cents = min(cents, have - (t.ante_cents + t.bb_cents))
        if cents < t.bb_cents:
            _emit(t, "cashout", f"{p.name}'s withdrawal was skipped — not enough left on the table")
            continue
        try:
            with _mutation(t):
                moved = min(cents_to_chips(cents, t.bb_cents), int(p.stack_chips))
                p.stack_chips -= moved
                p.leftover_cents += cents
                _ledger_add(t, p.user_id, "take_off", cents)
                _persist_player(t, t.seat_of(p.user_id), p, True)
            _emit(t, "cashout", f"{p.name} took {_fmt_cents(cents)} off the table")
        except Exception:  # noqa: BLE001 — rolled back; nothing left the table
            logger.exception("queued withdrawal failed (table %s)", t.game_id)
    for p in t.seats:
        if p is None or not p.queued_topup_cents:
            continue
        cents, p.queued_topup_cents = int(p.queued_topup_cents), 0
        hi = int(t.max_buyin_cents or 0)
        if hi:
            cents = min(cents, max(0, hi - chips_to_cents(p.stack_chips, t.bb_cents)))
        if cents < 1:
            continue
        try:
            with _mutation(t):
                p.stack_chips += cents_to_chips(cents, t.bb_cents)
                p.buyin_cents += cents
                _ledger_add(t, p.user_id, "topup", cents)
                _persist_player(t, t.seat_of(p.user_id), p, True)
            _emit(t, "rebuy", f"{p.name} added {_fmt_cents(cents)}")
        except Exception:  # noqa: BLE001 — rolled back; the chips were never added
            logger.exception("queued top-up failed (table %s)", t.game_id)


def _defer_removal_locked(t: LiveTable, i: int) -> None:
    """Seat ``i`` is out of the game but a hand (or its runout) is still
    live: mark it away + pending. The away logic folds it when it faces a
    bet (checks when that is free — the engine has no fold-for-free), and
    ``_settle_locked`` cashes it out once the hand is over."""
    p = t.seats[i]
    assert p is not None
    with _mutation(t):
        p.sitting_out = True
        t.pending_kicks.add(p.user_id)
        _persist_player(t, i, p, True)
    if t.phase == "in_hand" and t.env is not None and t.env.current_actor() == i:
        _resume_turn_locked(t)


def _leave_locked(t: LiveTable, uid: int, *, now: bool = False) -> None:
    """Leave the seat. Holding cards: the hand is PLAYED OUT first (the player
    keeps acting as normal) and the seat is cashed out when it ends — never a
    forced fold. ``now`` is the old behaviour (out of the hand at once, away =
    fold when facing a bet): what the host's Remove and a kicked player get."""
    i = t.seat_of(uid)
    if i is None:
        raise HTTPException(status_code=400, detail="not seated")
    dealt_in = bool(t.in_hand_mask and i < len(t.in_hand_mask) and t.in_hand_mask[i])
    if (t.phase == "in_hand" and dealt_in) or _runout_blocking(t):
        if now:
            # (review 2026-09-20 G6) with an AFK opponent and no clock the only
            # way out of the table used to be the host
            _defer_removal_locked(t, i)
            return
        p = t.seats[i]
        assert p is not None
        if not p.leave_after_hand:
            p.leave_after_hand = True
            p.sit_out_next = False
            t.rev += 1
            _emit(t, "leave", f"{p.name} leaves after this hand", seat=i)
        return
    _cash_out_seat(t, i)


def _move_locked(t: LiveTable, uid: int, seat: int) -> None:
    """Change seats (FEAT-013, 2026-09-28): a seated player takes an empty seat, with
    everything the seat carries (stack, ledger totals, automatic chips, time bank).
    Only while they hold no cards — a player in the hand moves once it is over (the
    client sends the move then) — and not while the next hand's shuffle is being
    confirmed (its lock list names seats). A shuffle commitment made from the old seat
    names that seat, so it is dropped: the device commits again for the new one."""
    _require_open(t)
    i = t.seat_of(uid)
    if i is None:
        raise HTTPException(status_code=400, detail="not seated")
    if seat < 0 or seat >= t.num_seats:
        raise HTTPException(status_code=400, detail="invalid seat")
    if seat == i:
        return
    if t.seats[seat] is not None:
        raise HTTPException(status_code=409, detail="seat taken")
    holder = next((r for r in t.requests if r["kind"] == "sit" and r["seat"] == seat), None)
    if holder is not None:
        raise HTTPException(status_code=409, detail=f"that seat is reserved for {holder['name']}")
    p = t.seats[i]
    assert p is not None
    if p.leave_after_hand or p.user_id in t.pending_kicks:
        raise HTTPException(status_code=400, detail="you are leaving this table")
    if _dealt_in(t, i):
        raise HTTPException(status_code=409, detail="you can change seats when this hand is over")
    nxt = t.fair_next
    if nxt is not None and nxt.pending:
        raise HTTPException(status_code=409, detail="the next hand is being dealt — try again in a moment")
    with _mutation(t):
        t.seats[seat] = p
        t.seats[i] = None
        _persist_player(t, seat, p, True)
    # (a hand in progress that this player is not in: its snapshot follows both seats)
    _sync_idle_stack(t, i)
    _sync_idle_stack(t, seat)
    if nxt is not None and nxt.stage == "commit" and nxt.commit_users.get(i) == int(uid):
        nxt.commits.pop(i, None)
        nxt.commit_users.pop(i, None)
    _emit(t, "move", f"{p.name} moved to seat {seat + 1}", seat=seat)


def _cancel_leave_locked(t: LiveTable, uid: int) -> None:
    p = t.player(uid)
    if p is None:
        raise HTTPException(status_code=400, detail="not seated")
    if p.leave_after_hand:
        p.leave_after_hand = False
        t.rev += 1
        _emit(t, "back", f"{p.name} is staying", seat=t.seat_of(uid))


def _apply_leaves_locked(t: LiveTable) -> None:
    """Players who asked to leave after the hand: cash them out now that it is
    over (the runout included — a cashed-out ledger row would spoil it)."""
    if _hand_busy(t):
        return
    for i, p in enumerate(list(t.seats)):
        if p is not None and p.leave_after_hand:
            try:
                _cash_out_seat(t, i)
            except Exception:  # noqa: BLE001 — rolled back; retry next tick
                logger.exception("leave-after-hand cash-out failed (table %s)", t.game_id)


def _rebuy_locked(t: LiveTable, uid: int, amount_cents: int) -> None:
    _require_open(t)
    p = t.player(uid)
    if p is None:
        raise HTTPException(status_code=400, detail="not seated")
    if _dealt_in(t, t.seat_of(uid)):
        # Chips behind cannot change while you hold cards (table stakes). A
        # busted / sitting-out player is not in the hand and may reload now.
        raise HTTPException(status_code=400, detail="wait for the hand to finish")
    have = chips_to_cents(p.stack_chips, t.bb_cents)
    _check_amount(t, amount_cents, have, what="rebuy")
    if p.buyin_cents + amount_cents > MAX_CENTS:
        raise HTTPException(status_code=400, detail="buy-in limit reached")
    _check_buyin_limits(t, amount_cents, have, fresh=False)
    with _mutation(t):
        p = t.player(uid)
        assert p is not None
        p.stack_chips += cents_to_chips(amount_cents, t.bb_cents)
        p.buyin_cents += amount_cents
        _ledger_add(t, uid, "topup", amount_cents)
        _persist_player(t, t.seat_of(uid), p, True)
    _sync_idle_stack(t, t.seat_of(uid))
    _emit(t, "rebuy", f"{p.name} added {_fmt_cents(amount_cents)}", seat=t.seat_of(uid))


def _sit_out_locked(
    t: LiveTable, uid: int, on: bool, *, target_uid: int | None = None,
    next_hand: bool = False,
) -> None:
    _require_open(t)
    who = int(target_uid) if target_uid is not None else int(uid)
    if target_uid is not None:
        _require_host(t, uid, "sit another player out")
    if t.player(who) is None:
        raise HTTPException(status_code=400, detail="not seated")
    if (not on) and who in t.pending_kicks:
        raise HTTPException(
            status_code=400, detail="player is being removed from the table"
        )
    i = t.seat_of(who)
    if on and next_hand:
        # "Sit out next hand": finish the hand you are in, then go away. Only
        # meaningful while dealt into a live hand — otherwise it is immediate.
        live = (
            t.phase == "in_hand"
            and i is not None
            and bool(t.in_hand_mask and i < len(t.in_hand_mask) and t.in_hand_mask[i])
        )
        if live:
            p = t.player(who)
            assert p is not None
            if not p.sit_out_next:
                p.sit_out_next = True
                t.rev += 1
            return
    with _mutation(t):
        p = t.player(who)
        assert p is not None
        was = bool(p.sitting_out)
        p.sitting_out = bool(on)
        p.sit_out_next = False
        _persist_player(t, i, p, True)
    if was != bool(on):
        _emit(t, "away" if on else "back",
              f"{p.name} is sitting out" if on else f"{p.name} is back", seat=i)
    # (review 2026-09-20 G5) Only when the player who just went away IS the
    # actor does anything about the turn change (they get auto-acted and the
    # next actor's clock starts). Anyone else toggling Away used to restart
    # the actor's shot clock.
    if (
        p.sitting_out
        and t.phase == "in_hand"
        and t.env is not None
        and t.env.current_actor() == i
    ):
        _resume_turn_locked(t)


def _kick_locked(t: LiveTable, uid: int, target_uid: int) -> None:
    _require_open(t)
    _require_host(t, uid, "remove a player")
    target_uid = int(target_uid)
    if target_uid == uid:
        raise HTTPException(
            status_code=400, detail="host cannot remove themselves — leave or close"
        )
    i = t.seat_of(target_uid)
    if i is None:
        raise HTTPException(status_code=400, detail="player not seated")
    if _hand_busy(t):
        _defer_removal_locked(t, i)
        return
    _cash_out_seat(t, i)


def _valid_decision_secs(secs: int) -> int:
    n = int(secs)
    if n < 0:
        raise HTTPException(status_code=400, detail="decision time must be >= 0")
    if n != 0 and (n < 5 or n > 180):
        raise HTTPException(
            status_code=400,
            detail="decision time must be 0 (off) or 5–180 seconds",
        )
    return n


MAX_STREET_PAUSE_S = 5.0


def _valid_street_pause(secs: float) -> float:
    n = float(secs)
    # `not (a <= n <= b)` also rejects NaN, which sails through `n < a or
    # n > b` and then 500'd every GET of the table (review 2026-09-20 G4).
    if not math.isfinite(n) or not (MIN_STREET_PAUSE_S <= n <= MAX_STREET_PAUSE_S):
        raise HTTPException(
            status_code=400,
            detail=f"runout pause must be between {MIN_STREET_PAUSE_S:g} and {MAX_STREET_PAUSE_S:g} seconds",
        )
    return n


def _set_street_pause_locked(t: LiveTable, uid: int, secs: float) -> None:
    _require_open(t)
    _require_host(t, uid, "set runout speed")
    n = _valid_street_pause(secs)
    with _mutation(t):
        t.street_pause_secs = n
        _persist_meta(t)


def _close_table_locked(t: LiveTable) -> None:
    """Close the table: everyone is cashed out (the host stays the host of the
    finished session), the game stops, nothing is waiting any more. The ONE
    closing path (HGB-008) — the host's Close and the club owner's Exclude."""
    if _hand_busy(t):
        raise HTTPException(status_code=400, detail="wait for the hand to finish")
    with _mutation(t):
        for i in list(t.occupied()):
            _cash_out_seat(t, i, closing=True)
        t.pending_kicks.clear()
        t.status = "closed"
        t.running = False
        t.phase = "waiting"
        _persist_meta(t)
    t.hand = HandState()  # (HGB-004: nothing of the last hand is left on a closed table)
    t.next_deal_mono = None
    t.fair_next = None  # never dealt: a closed table deals nothing
    if t.requests:
        t.requests = []
    t.rev += 1


def _close_locked(t: LiveTable, uid: int) -> None:
    _require_host(t, uid, "close the table")
    _close_table_locked(t)


def _host_fold_locked(t: LiveTable, uid: int) -> None:
    _require_host(t, uid, "fold a player")
    if t.phase != "in_hand" or t.env is None or t.info is None:
        raise HTTPException(status_code=400, detail="no hand in progress")
    if t.env.current_actor() is None:
        raise HTTPException(status_code=400, detail="no actor")
    # The engine has no fold-for-free: when the stalled player faces no bet
    # the host's button checks them instead, so it ALWAYS moves the hand on
    # (review 2026-09-20 G6 — it used to 400 "illegal action" there).
    if bool(t.info.gate_mask[GATE_FOLD]):
        _apply_action_locked(t, GATE_FOLD, 0)
    else:
        gate, chips = _passive_choice(t)
        _apply_action_locked(t, gate, chips)


def _show_locked(t: LiveTable, uid: int) -> None:
    """Table your own cards after the hand ("show"): a fold-out winner's
    bluff, or a folded hand. Your choice only — nobody can show for you —
    and only for the hand that just ended."""
    if t.phase != "showdown" or t.env is None:
        raise HTTPException(status_code=400, detail="nothing to show right now")
    dealt = list(t.dealt_user_ids or [])
    seat = next((i for i, d in enumerate(dealt) if d is not None and d == uid), None)
    if seat is None or not (t.in_hand_mask and seat < len(t.in_hand_mask) and t.in_hand_mask[seat]):
        raise HTTPException(status_code=400, detail="you were not dealt into this hand")
    occupant = t.seats[seat].user_id if t.seats[seat] is not None else None
    if occupant is not None and occupant != uid:
        raise HTTPException(status_code=400, detail="you were not dealt into this hand")
    if seat in t.shown_seats:
        return
    t.shown_seats.add(seat)
    t.rev += 1
    _emit(t, "show", f"{_seat_name(t, seat)} shows", seat=seat)
    # Keep the stored hand in step, so history shows what the table saw
    # (atomically: the grader may be writing the same record — OPS-007).
    try:
        with _edit_hand_record(t.game_id, int(t.hand_no)) as rec:
            for s in (rec or {}).get("seats") or []:
                if int(s.get("seat", -1)) == seat:
                    s["shown"] = True
            pub.DB.q(  # (SEC-004: a shown hand's marks sort like a tabled one's)
                "UPDATE homegame_hand_results SET shown=1 WHERE game_id=? AND hand_no=? AND user_id=?",
                (t.game_id, int(t.hand_no), int(uid)),
            )
    except Exception:  # noqa: BLE001 — cosmetic
        logger.exception("homegame show: history update failed (%s)", t.game_id)


def _react_locked(t: LiveTable, uid: int, emote: Any) -> None:
    """A quick emote over your seat. Whitelisted keys only (the client maps
    them to glyphs); shares the chat rate limit."""
    seat = t.seat_of(uid)
    if seat is None:
        raise HTTPException(status_code=400, detail="take a seat first")
    key = str(emote or "").strip().lower()
    if key not in REACTIONS:
        raise HTTPException(status_code=400, detail="unknown reaction")
    now = time.monotonic()
    times = t.chat_times.setdefault(uid, deque())
    while times and now - times[0] > CHAT_RATE_WINDOW_S:
        times.popleft()
    if len(times) >= CHAT_RATE_MAX:
        raise HTTPException(status_code=429, detail="slow down")
    times.append(now)
    t.reaction_seq += 1
    t.reactions.append(
        {"id": t.reaction_seq, "seat": seat, "emote": key, "ts": time.time()}
    )
    t.rev += 1


def _transfer_host_locked(t: LiveTable, uid: int, target_uid: int) -> None:
    _require_open(t)
    _require_host(t, uid, "hand over the table")
    target_uid = int(target_uid)
    p = t.player(target_uid)
    if p is None:
        raise HTTPException(status_code=400, detail="the new host must be seated")
    if target_uid == uid:
        return
    with _mutation(t):
        t.host_user_id = target_uid
        _persist_meta(t)
    _emit(t, "host", f"{p.name} is now the host")


def _settings_locked(t: LiveTable, uid: int, body: dict) -> None:
    """Host edits the table. Everything here is safe between AND during hands:
    the hand in progress keeps the config it was dealt with (the engine owns
    it), so stakes/pace changes simply apply from the next deal. Seat count
    is the exception — it reshapes the table, so it waits for the hand."""
    _require_open(t)
    _require_host(t, uid, "change table settings")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="invalid settings")
    changed: list[str] = []
    with _mutation(t):
        if body.get("name") is not None:
            name = _clean_text(body["name"])
            if not name:
                raise HTTPException(status_code=400, detail="table name cannot be empty")
            if len(name) > MAX_NAME_LEN:
                raise HTTPException(
                    status_code=400,
                    detail=f"table name is limited to {MAX_NAME_LEN} characters",
                )
            if name != t.name:
                t.name = name
                changed.append(f"table renamed to “{name}”")
        ante_changed = False
        if body.get("ante_cents") is not None:
            ante = _parse_cents(body, "ante_cents")
            if ante <= 0 or cents_to_chips(ante, t.bb_cents) <= 0:
                raise HTTPException(status_code=400, detail="ante must be positive")
            if ante != t.ante_cents:
                t.ante_cents = ante
                ante_changed = True
                changed.append(f"ante is now {_fmt_cents(ante)} (next hand)")
        lo = _parse_cents(body, "min_buyin_cents", t.min_buyin_cents)
        hi = _parse_cents(body, "max_buyin_cents", t.max_buyin_cents)
        dflt = _parse_cents(body, "default_buyin_cents", t.default_buyin_cents)
        if (lo, hi, dflt) != (t.min_buyin_cents, t.max_buyin_cents, t.default_buyin_cents) or ante_changed:
            # (checked when the window or the ante moves — a table saved under an
            # older, looser rule can still be renamed without fixing it first)
            _validate_buyin_window(t.bb_cents, lo, hi, dflt, t.ante_cents)
        if (lo, hi, dflt) != (t.min_buyin_cents, t.max_buyin_cents, t.default_buyin_cents):
            t.min_buyin_cents, t.max_buyin_cents, t.default_buyin_cents = lo, hi, dflt
            changed.append("buy-in limits changed")
            capped = _cap_auto_chips_locked(t)
            if capped:
                changed.append(f"automatic chips now stop at the new maximum ({_fmt_cents(hi)})"
                               + f" — {', '.join(capped)}")
        if body.get("decision_secs") is not None:
            secs = _valid_decision_secs(pub.body_int(body, "decision_secs"))
            if secs != t.decision_secs:
                t.decision_secs = secs
                changed.append(
                    "shot clock off" if secs == 0 else f"shot clock {secs}s"
                )
        if body.get("time_bank_secs") is not None:
            bank = _valid_time_bank(pub.body_int(body, "time_bank_secs"))
            if bank != t.time_bank_secs:
                t.time_bank_secs = bank
                for p in t.seats:
                    if p is not None:
                        p.time_bank_left = float(bank)
                changed.append("time bank off" if bank == 0 else f"time bank {bank}s")
        if body.get("deal_delay_secs") is not None:
            delay = _valid_deal_delay(_parse_secs(body, "deal_delay_secs", 0.0))
            if delay != t.deal_delay_secs:
                t.deal_delay_secs = delay
                t.next_deal_mono = None
                changed.append(
                    "manual dealing" if delay == 0 else f"next hand after {delay:g}s"
                )
        if body.get("street_pause_secs") is not None:
            t.street_pause_secs = _valid_street_pause(_parse_secs(body, "street_pause_secs", 1.5))
        if body.get("listed") is not None:
            t.listed = _parse_bool(body, "listed", t.listed)
        if body.get("allow_rabbit") is not None:
            t.allow_rabbit = _parse_bool(body, "allow_rabbit", t.allow_rabbit)
        if body.get("show_grades") is not None:
            t.show_grades = _parse_bool(body, "show_grades", t.show_grades)
        if body.get("allow_rathole") is not None:
            t.allow_rathole = _parse_bool(body, "allow_rathole", t.allow_rathole)
        if body.get("approve_buyins") is not None:
            on = _parse_bool(body, "approve_buyins", t.approve_buyins)
            if on != t.approve_buyins:
                t.approve_buyins = on
                changed.append(
                    "buy-ins now need the host's approval" if on
                    else "buy-ins no longer need approval"
                )
        if body.get("num_seats") is not None:
            n = pub.body_int(body, "num_seats")
            if n != t.num_seats:
                _resize_locked(t, n)
                changed.append(f"table is now {n}-max")
        _persist_seats(t)  # the settings, and any automatic-chips target a lower maximum capped
    if body.get("decision_secs") is not None:
        if t.phase == "in_hand":
            _start_clock_locked(t, restart=True)  # the host changed the rules
        else:
            t.turn_started_mono = None
    if not t.approve_buyins and t.requests:
        # approval was switched off: everything that was waiting goes through
        _let_through_locked(t, uid, list(t.requests))
    for line in changed:
        _emit(t, "settings", f"Host: {line}")
    _remember_host_prefs(t)


def _validate_buyin_window(bb_cents: int, lo: int, hi: int, dflt: int, ante_cents: int = 0) -> None:
    """The host's buy-in window. Every amount in it must also buy MORE than the
    ante (OPS-005: a table whose default buy-in was at or below the ante could
    never deal, and its players never knew why)."""
    def dealable(c: int) -> bool:
        return c > ante_cents and cents_to_chips(c, bb_cents) > cents_to_chips(ante_cents, bb_cents)

    if dflt < bb_cents:
        raise HTTPException(status_code=400, detail="default buy-in must be at least 1 bb")
    if not dealable(dflt):
        raise HTTPException(status_code=400,
                            detail=f"default buy-in must be more than the ante ({_fmt_cents(ante_cents)})")
    if lo and lo < bb_cents:
        raise HTTPException(status_code=400, detail="minimum buy-in must be at least 1 bb")
    if lo and not dealable(lo):
        raise HTTPException(status_code=400,
                            detail=f"minimum buy-in must be more than the ante ({_fmt_cents(ante_cents)})")
    if lo and hi and hi < lo:
        raise HTTPException(status_code=400, detail="maximum buy-in is below the minimum")
    if lo and dflt < lo:
        raise HTTPException(status_code=400, detail="default buy-in is below the minimum")
    if hi and dflt > hi:
        raise HTTPException(status_code=400, detail="default buy-in is above the maximum")


def _valid_time_bank(secs: int) -> int:
    n = int(secs)
    if not (0 <= n <= MAX_TIME_BANK_SECS):
        raise HTTPException(
            status_code=400,
            detail=f"time bank must be 0–{MAX_TIME_BANK_SECS} seconds",
        )
    return n


def _valid_deal_delay(secs: float) -> float:
    if not (secs == 0 or 1.0 <= secs <= MAX_DEAL_DELAY_SECS):
        raise HTTPException(
            status_code=400,
            detail=f"next-hand delay must be 0 (manual) or 1–{MAX_DEAL_DELAY_SECS:g} seconds",
        )
    return float(secs)


def _valid_num_seats(n: int, variant: str = DEFAULT_GAME) -> int:
    """Seats for a table of this game. PLO6 stops at 7: six cards each for seven
    players plus the two boards is the whole deck (there are no burn cards)."""
    n = int(n)
    g = GAMES[_norm_game(variant)]
    top = int(g["max_seats"])
    if not (MIN_SEATS <= n <= top):
        why = (
            "" if top == TABLE_SEATS
            else f" ({g['label']}: up to {g['hole']} cards each plus {g['burns']} face-up burns use up the deck)"
            if g["burns"] else f" ({g['label']}: {g['hole']} cards each use up the deck)"
        )
        raise HTTPException(status_code=400, detail=f"seats must be {MIN_SEATS}–{top}" + why)
    return n


def _resize_locked(t: LiveTable, n: int) -> None:
    """Change the seat count (caller is inside ``_mutation``). Between hands
    only; shrinking needs the removed seats empty. The button stays on a seat
    that still exists."""
    n = _valid_num_seats(n, t.variant)
    if _hand_busy(t):
        raise HTTPException(status_code=400, detail="wait for the hand to finish")
    if n < t.num_seats and any(s is not None for s in t.seats[n:]):
        raise HTTPException(
            status_code=400,
            detail="move or remove the players in the higher seats first",
        )
    seats = list(t.seats[:n])
    seats += [None] * (n - len(seats))
    t.seats = seats
    t.num_seats = n
    t.button = int(t.button) % n
    # Per-hand state is sized to the table that played it: the finished hand is no
    # longer drawn after a reshape (HGB-004: a fresh HandState — every field of it).
    t.hand = HandState()
    t.phase = "waiting"


def _exclude_locked(t: LiveTable, user: Any, on: bool) -> None:
    """The CLUB'S OWNER: take a session out of the club's record (test tables) or
    put it back — not the host (a host must not be able to erase a losing night).
    Nothing is deleted — hands, ledger and flows stay in the database and the
    table still opens by its link; it just counts nowhere. An open table is
    closed first (everyone is cashed out), so it also leaves the lobby."""
    if user is None or _club_role(t.club_id, int(user["id"])) != "owner":
        raise HTTPException(status_code=403, detail="only the club's owner can do that")
    if on and t.status == "open":
        _close_table_locked(t)
    pub.DB.q("UPDATE homegames SET excluded=? WHERE id=?", (1 if on else 0, t.game_id))
    t.rev += 1


def _parse_cents(body: dict, key: str, default: int | None = None) -> int:
    """A money field in whole cents: finite, >= 0, <= MAX_CENTS — else 400.

    (review 2026-09-20 G4) ``int(round(float(v)))`` accepted ``1e30`` (sqlite
    OverflowError AFTER memory had been mutated — the table was bricked until
    a restart), and raised on ``1e999`` / NaN (500)."""
    if not isinstance(body, dict) or key not in body or body[key] is None:
        if default is None:
            raise HTTPException(status_code=400, detail=f"missing {key}")
        return int(default)
    v = body[key]
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise HTTPException(status_code=400, detail=f"invalid {key}")
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError) as e:
        raise HTTPException(status_code=400, detail=f"invalid {key}") from e
    if not math.isfinite(f):
        raise HTTPException(status_code=400, detail=f"invalid {key}")
    if f < 0:
        raise HTTPException(status_code=400, detail=f"{key} must be >= 0")
    if f > MAX_CENTS:
        raise HTTPException(status_code=400, detail=f"{key} is too large")
    if f != int(f):
        # HGB-021: money moves in whole cents; a fraction is a client bug, and
        # round() would have rounded it banker's-style (12.5 -> 12, 13.5 -> 14)
        raise HTTPException(status_code=400, detail=f"{key} must be whole cents")
    return int(f)


def _parse_secs(body: dict, key: str, default: float) -> float:
    """A finite float field (seconds), or 400."""
    v = body.get(key, default) if isinstance(body, dict) else default
    if v is None:
        v = default
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise HTTPException(status_code=400, detail=f"invalid {key}")
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError) as e:
        raise HTTPException(status_code=400, detail=f"invalid {key}") from e
    if not math.isfinite(f):
        raise HTTPException(status_code=400, detail=f"invalid {key}")
    return f


def _parse_bool(body: dict, key: str, default: bool) -> bool:
    v = body.get(key, default) if isinstance(body, dict) else default
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    raise HTTPException(status_code=400, detail=f"{key} must be true or false")


# --- the host's settings from last time (owner, 2026-09-26) ---------------------
# Every setting a host gives a table is saved per HOST — amounts as multiples of
# the big blind, so a new big blind scales them. The create dialog starts from
# them (GET /games/api/host_prefs); a create with ``remembered: true`` also takes
# the ones only Manage has (grades, rabbit, runout pause, automatic chips).
def _host_prefs_of(t: LiveTable) -> dict[str, Any]:
    bb = max(1, int(t.bb_cents))

    def per_bb(cents: Any) -> float:
        return round(int(cents or 0) / bb, 4)

    return {
        "variant": _norm_game(t.variant),
        "bb_cents": bb, "ante_bb": per_bb(t.ante_cents), "num_seats": int(t.num_seats),
        "buyin_bb": per_bb(t.default_buyin_cents), "min_buyin_bb": per_bb(t.min_buyin_cents),
        "max_buyin_bb": per_bb(t.max_buyin_cents),
        "decision_secs": int(t.decision_secs or 0), "time_bank_secs": int(t.time_bank_secs or 0),
        "deal_delay_secs": float(t.deal_delay_secs or 0.0),
        "street_pause_secs": float(t.street_pause_secs),
        "listed": bool(t.listed), "approve_buyins": bool(t.approve_buyins),
        "allow_rathole": bool(t.allow_rathole), "allow_rabbit": bool(t.allow_rabbit),
        "show_grades": bool(t.show_grades),
        "topup_mode": _norm_auto_mode(t.topup_mode),
        "topup_target_bb": per_bb(t.topup_all_target_cents),
        "topup_below_bb": per_bb(t.topup_all_below_cents),
        "auto_stack_mode": _norm_auto_mode(t.auto_stack_mode),
        "auto_stack_bb": per_bb(t.auto_stack_all_cents),
    }


def _remember_host_prefs(t: LiveTable) -> None:
    """Cosmetic: a failure here never touches the table."""
    try:
        pub.DB.q(
            "INSERT INTO homegame_host_prefs(user_id,prefs,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET prefs=excluded.prefs, updated_at=excluded.updated_at",
            (int(t.host_user_id), json.dumps(_host_prefs_of(t), separators=(",", ":")), pub._now()),
        )
    except Exception:  # noqa: BLE001
        logger.exception("homegame host prefs (%s)", t.game_id)


def _host_prefs(uid: int) -> dict[str, Any] | None:
    row = pub.DB.one("SELECT prefs FROM homegame_host_prefs WHERE user_id=?", (int(uid),))
    try:
        out = json.loads(row["prefs"]) if row is not None else None
    except ValueError:
        return None
    return out if isinstance(out, dict) else None


def _apply_remembered_locked(t: LiveTable, uid: int, prefs: dict[str, Any]) -> None:
    """The settings the create dialog doesn't carry, from the host's last table,
    each through the host's own endpoint logic; one that no longer fits (say a
    top-up target above this table's maximum) is skipped."""
    def cents(key: str) -> int:
        try:
            return int(round(float(prefs.get(key) or 0) * int(t.bb_cents)))
        except (TypeError, ValueError):
            return 0

    steps: list[tuple[Any, dict[str, Any]]] = [
        (_settings_locked, {k: prefs[k] for k in ("street_pause_secs", "allow_rabbit", "show_grades")
                            if prefs.get(k) is not None}),
        (_auto_topup_host_locked, {"mode": prefs.get("topup_mode") or "off", **(
            {"all_target_cents": cents("topup_target_bb"), "all_below_cents": cents("topup_below_bb")}
            if cents("topup_target_bb") > 0 else {})}),
        (_auto_stack_host_locked, {"mode": prefs.get("auto_stack_mode") or "off", **(
            {"all_cents": cents("auto_stack_bb")} if cents("auto_stack_bb") > 0 else {})}),
    ]
    for fn, body in steps:
        try:
            fn(t, uid, body)
        except HTTPException:
            pass


def _create_table(user: Any, body: dict) -> LiveTable:
    name = _clean_text(body.get("name")) or "Table 1"
    if len(name) > MAX_NAME_LEN:
        raise HTTPException(
            status_code=400, detail=f"table name is limited to {MAX_NAME_LEN} characters"
        )
    variant = _parse_game(body.get("variant")) if body.get("variant") is not None else DEFAULT_GAME
    if GAMES[variant]["burns"] and not PLO67_ON:
        raise HTTPException(status_code=400, detail="this server's engine does not deal PLO67 yet")
    n = (
        _valid_num_seats(pub.body_int(body, "num_seats"), variant)
        if body.get("num_seats") is not None
        else int(GAMES[variant]["max_seats"])
    )
    bb = _parse_cents(body, "bb_cents", 100)
    sb = _parse_cents(body, "sb_cents", max(1, bb // 2))  # (nobody posts it: the chip unit is the bb)
    ante = _parse_cents(body, "ante_cents", 300)
    buyin = _parse_cents(body, "default_buyin_cents", 4000)
    secs = _valid_decision_secs(
        pub.body_int(body, "decision_secs", DEFAULT_DECISION_SECS)
    )
    if bb <= 0:
        raise HTTPException(status_code=400, detail="big blind must be positive")
    if ante <= 0 or cents_to_chips(ante, bb) <= 0:
        raise HTTPException(status_code=400, detail="ante must be positive")
    if buyin < bb:
        raise HTTPException(status_code=400, detail="default buy-in must be at least 1 bb")
    lo = _parse_cents(body, "min_buyin_cents", 0)
    hi = _parse_cents(body, "max_buyin_cents", 0)
    _validate_buyin_window(bb, lo, hi, buyin, ante)
    bank = _valid_time_bank(pub.body_int(body, "time_bank_secs", 0))
    # Absent = MANUAL dealing (see MAX_DEAL_DELAY_SECS); the dialog sends 5.
    delay = _valid_deal_delay(_parse_secs(body, "deal_delay_secs", 0.0))
    listed = _parse_bool(body, "listed", True)
    approve = _parse_bool(body, "approve_buyins", False)
    rathole = _parse_bool(body, "allow_rathole", False)
    club_id = _club_for_new_table(int(user["id"]), body.get("club_id"))
    # (review 2026-09-20 G13) 50 tables in 0.32 s, none ever evicted.
    open_n = pub.DB.one(
        "SELECT COUNT(*) c FROM homegames WHERE host_user_id=? AND status='open'",
        (int(user["id"]),),
    )["c"]
    if int(open_n) >= MAX_OPEN_TABLES_PER_USER:
        raise HTTPException(
            status_code=429,
            detail=(
                f"you already host {MAX_OPEN_TABLES_PER_USER} open tables —"
                " close one first"
            ),
        )
    gid = secrets.token_urlsafe(6)
    t = LiveTable(
        game_id=gid,
        host_user_id=int(user["id"]),
        name=name,
        num_seats=n,
        sb_cents=sb,
        bb_cents=bb,
        ante_cents=ante,
        default_buyin_cents=buyin,
        status="open",
        button=0,
        hand_no=0,
        seats=[None] * n,
        running=False,
        club_id=club_id,
        decision_secs=secs,
        deal_delay_secs=delay,
        time_bank_secs=bank,
        min_buyin_cents=lo,
        max_buyin_cents=hi,
        listed=listed,
        approve_buyins=approve,
        allow_rathole=rathole,
        variant=variant,
    )
    # The table row and the host's seat land together or not at all. (The table
    # is nobody else's yet; its lock is taken anyway — BEFORE the transaction, in
    # the lock order — because `_sit_locked` expects it.)
    with t.lock, pub.DB.transaction():
        cols = ["id", "sb_cents", "bb_cents", "club_id", "variant", "created_at"] + [c.col for c in META]
        pub.DB.q(
            f"INSERT INTO homegames({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
            (gid, sb, bb, club_id, variant, pub._now(), *(c.to_db(getattr(t, c.attr)) for c in META)),
        )
        # Host sits seat 0 with the default buy-in so they can deal once a
        # second player sits.
        _sit_locked(t, user, 0, buyin)
    CTX.hub.put(t)
    return t


CHAT_SHOWN = 80  # the newest lines a view carries


def _chat_messages(t: LiveTable) -> list[dict[str, Any]]:
    """The table's newest chat lines. Read from the database once and kept on
    the table until a new line is posted (PERF-003: every view — every poll and
    every live push, for every viewer — used to query the chat, and without an
    index that scanned every line of every table)."""
    if t.chat_cache is None:
        rows = pub.DB.q(
            "SELECT c.id, c.body, c.created_at, c.user_id, u.name, u.email "
            "FROM homegame_chat c JOIN users u ON u.id=c.user_id "
            "WHERE c.game_id=? ORDER BY c.id DESC LIMIT ?",
            (t.game_id, CHAT_SHOWN),
        )
        t.chat_cache = [
            {"id": int(r["id"]), "name": _display_name(r, t.club_id), "user_id": int(r["user_id"]),
             "text": r["body"], "created_at": r["created_at"]}
            for r in reversed(list(rows))
        ]
    return [
        {
            "id": m["id"],
            "name": m["name"],
            "user_id": m["user_id"],  # (the client matches on this, not the name — HGT-027)
            "avatar": _avatar_url(m["user_id"]),
            "text": m["text"],
            "created_at": m["created_at"],
        }
        for m in t.chat_cache
    ]


def _chat_add(t: LiveTable, user: Any, text: Any) -> None:
    uid = int(user["id"])
    _require_member(t, uid)  # review 2026-09-20 G12
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="invalid message")
    body = _clean_text(text[: CHAT_MAX_LEN * 8])
    if not body:
        raise HTTPException(status_code=400, detail="empty message")
    if len(body) > CHAT_MAX_LEN:
        body = body[:CHAT_MAX_LEN]
    # (review 2026-09-20 G13) Per-user, per-table sliding-window rate limit.
    now = time.monotonic()
    times = t.chat_times.setdefault(uid, deque())
    while times and now - times[0] > CHAT_RATE_WINDOW_S:
        times.popleft()
    if len(times) >= CHAT_RATE_MAX:
        raise HTTPException(status_code=429, detail="slow down — too many messages")
    pub.DB.q(
        "INSERT INTO homegame_chat(game_id,user_id,body,created_at) VALUES(?,?,?,?)",
        (t.game_id, uid, body, pub._now()),
    )
    t.chat_cache = None  # (the next view reads the newest lines again)
    times.append(now)
    t.rev += 1


def _read_env_settings() -> None:
    """The environment-driven settings, read again for the app being installed
    (BE-007: the app factory can build several apps in one process — tests; the
    server installs once, right after import, so these are the values it started with)."""
    global MAX_OPEN_TABLES_PER_USER, GRADING_ON
    MAX_OPEN_TABLES_PER_USER = _env_max_tables()
    GRADING_ON = _env_grading_on()


def install(app: FastAPI, *, static_dir: Path, model_provider: Any = None) -> None:
    """Mount the home games on the public app: the pages and the gated client
    files here, the API from the module's ``router``. ``model_provider()`` returns
    the PLO5 model grading scores with (or server.py calls ``set_model_provider``).

    The app's state is the CURRENT context: ``server.create_app`` makes a fresh
    ``HomeGames()`` current before installing (``use_context``), so every app
    starts with its own tables, workers and caches; the settings are read again
    and the request budgets (keyed by the app database's user ids) start empty."""
    _read_env_settings()
    for budget in (API_RATE, JOIN_RATE, AVATAR_RATE):
        if budget is not None:
            budget.reset()
    _take_process_lock()
    _ensure_schema()
    # The clubs' role cache forgets itself on any write to the club tables (PERF-004).
    if _on_clubs_write not in pub.DB.write_listeners:
        pub.DB.write_listeners.append(_on_clubs_write)
    _check_ledger_at_start()  # (OPS-017: logged, never fatal)
    try:  # the site's /health reports the home games' workers and money (OPS-011 / OPS-017)
        from plo5bp.ui import middleware as _mw

        _mw.register_health_check("home_games", _site_health)
    except Exception:  # noqa: BLE001 — an older site without the hook still runs the games
        logger.warning("homegame: the site's health checks are not available")
    if model_provider is not None:
        set_model_provider(model_provider)
    # OPS-008: save what tables still owe the database and stop the workers
    # when the server stops (it had no shutdown hook at all).
    app.router.on_shutdown.append(_on_app_shutdown)
    _build_cache["dir"] = static_dir  # (client_build() in the views)
    pub._GAMES_INVITE_HOOK = _invite_response
    pub._GAMES_ACCESS_HOOK = _admin_games_access
    pub._GAMES_MEMBER_HOOK = _main_club_member
    pub._GAMES_MEMBERS_HOOK = _main_club_members  # (/admin's user list: one lookup — PERF-010)

    def _html() -> HTMLResponse:
        return HTMLResponse(_page_html(static_dir), headers=dict(PAGE_HEADERS))

    @app.get("/games")
    def games_index():
        return _html()

    @app.get("/games/t/{game_id}")
    def games_table_page(game_id: str):
        row = pub.DB.one("SELECT id FROM homegames WHERE id=?", (game_id,))
        if row is None:
            raise HTTPException(status_code=404, detail="Not Found")
        return _html()

    @app.get("/games/join/{code}")
    def games_join_page(code: str):
        """A club invite link (the page shows the invitation — or that it expired)."""
        return _html()

    @app.get("/games/static/{name}")
    def games_asset(name: str, v: str | None = None):
        """Gated copies of the lobby JS/CSS.

        Served under /games/ (not /static/) so a previously cached 404 on
        /static/games.js cannot keep the create button dead. The address the
        page links (`?v=` = the file's content hash, see `_page_html`) is kept
        by the browser for a year; any other address is never cached.
        """
        media = GAMES_ASSETS.get(name)
        if media is None:
            raise HTTPException(status_code=404, detail="Not Found")
        path = static_dir / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Not Found")
        body = path.read_text(encoding="utf-8")
        current = bool(v) and v == _asset_hash(static_dir, name)
        return Response(
            body,
            media_type=media,
            headers={
                "Cache-Control": "private, max-age=31536000, immutable" if current else "no-store, must-revalidate",
                "X-Content-Type-Options": "nosniff",
            },
        )

    app.include_router(router)  # (the API: every /games/api route — HGB-002)
    _start_watchdog()
    if GRADING_ON:
        _start_grader()  # (its first sweep picks up jobs saved before a restart)
    logger.info("homegame routes installed")


# --- account export / deletion (main site ACC-009 / SEC-017) ------------------------
# Registered in `public.ACCOUNT_HOOKS`: the site's "Download my data" includes a
# player's home games, and "Delete my account" anonymizes them WITHOUT breaking
# anyone else's history — ledgers, hand results and money flows keep the user id
# (their sums stay exact: every ledger still adds up to zero), the id's row in
# `users` becomes an anonymous tombstone, and the name stored inside hand records
# is replaced. What was only theirs (picture, chat lines, host preferences, club
# memberships and join requests) is removed.


def _account_export(uid: int) -> dict[str, Any]:
    uid = int(uid)
    q = pub.DB.q
    results = q(
        "SELECT r.game_id, r.hand_no, r.delta_cents, r.showdown, r.acc_sum, r.acc_n, h.summary "
        "FROM homegame_hand_results r LEFT JOIN homegame_hands h "
        "ON h.game_id=r.game_id AND h.hand_no=r.hand_no WHERE r.user_id=? "
        "ORDER BY r.game_id, r.hand_no", (uid,),
    )
    hands = []
    for r in results:
        own_hole = None
        try:
            rec = json.loads(r["summary"]) if r["summary"] else {}
            mine = next((s for s in rec.get("seats", []) if int(s.get("user_id", -1)) == uid), None)
            own_hole = mine.get("hole") if mine else None
        except (ValueError, TypeError):
            pass
        hands.append({
            "table": r["game_id"], "hand_no": r["hand_no"], "net_cents": r["delta_cents"],
            "showdown": bool(r["showdown"]), "your_cards": own_hole,
            "accuracy": round(r["acc_sum"] / r["acc_n"], 1) if r["acc_n"] else None,
        })
    prefs = pub.DB.one("SELECT prefs FROM homegame_host_prefs WHERE user_id=?", (uid,))
    return {
        "name_at_the_tables": _chosen_name(uid),
        "clubs": [dict(r) for r in q(
            "SELECT c.id, c.name, m.role, m.joined_at, m.nickname FROM homegame_club_members m "
            "JOIN homegame_clubs c ON c.id=m.club_id WHERE m.user_id=?", (uid,))],
        "tables_hosted": [dict(r) for r in q(
            "SELECT id, name, variant, status, created_at, closed_at FROM homegames "
            "WHERE host_user_id=? ORDER BY created_at", (uid,))],
        "ledger": [dict(r) for r in q(
            "SELECT game_id AS table_id, kind, amount_cents, created_at FROM homegame_ledger "
            "WHERE user_id=? ORDER BY id", (uid,))],
        "hands": hands,
        "chat": [dict(r) for r in q(
            "SELECT game_id AS table_id, body, created_at FROM homegame_chat "
            "WHERE user_id=? ORDER BY id", (uid,))],
        "host_preferences": json.loads(prefs["prefs"]) if prefs else None,
        "has_picture": pub.DB.one(
            "SELECT 1 FROM homegame_avatars WHERE user_id=?", (uid,)) is not None,
    }


def _account_blockers(uid: int) -> list[str]:
    """Deleting is refused while the player is part of a live game, or owns a
    club other people are still in (it would have no owner)."""
    uid = int(uid)
    out: list[str] = []
    seat = pub.DB.one(
        "SELECT g.name FROM homegame_players p JOIN homegames g ON g.id=p.game_id "
        "WHERE g.status='open' AND p.user_id=? AND p.seat IS NOT NULL LIMIT 1", (uid,),
    )
    if seat is not None:
        out.append(f"Leave the table “{seat['name']}” first.")
    hosting = pub.DB.one(
        "SELECT name FROM homegames WHERE status='open' AND host_user_id=? LIMIT 1", (uid,),
    )
    if hosting is not None:
        out.append(f"Hand over or close the table “{hosting['name']}” first.")
    owned = pub.DB.one(
        "SELECT c.name FROM homegame_clubs c WHERE c.owner_user_id=? AND c.archived_at IS NULL AND EXISTS ("
        "SELECT 1 FROM homegame_club_members m WHERE m.club_id=c.id AND m.user_id<>?) LIMIT 1",
        (uid, uid),
    )  # (an archived club no longer needs an owner: FEAT-007)
    if owned is not None:
        out.append(f"Hand the club “{owned['name']}” over to another member first.")
    return out


def _account_anonymize(uid: int) -> None:
    uid = int(uid)
    user = pub._user_by_id(uid)
    old_name = _display_name(user) if user is not None else None
    deleted = "Deleted player"
    game_ids = [r["game_id"] for r in pub.DB.q(
        "SELECT DISTINCT game_id FROM homegame_hand_results WHERE user_id=?", (uid,))]
    with pub.DB.transaction():
        for gid in game_ids:
            for h in pub.DB.q(
                "SELECT hand_no, summary FROM homegame_hands WHERE game_id=?", (gid,),
            ):
                try:
                    rec = json.loads(h["summary"])
                except (ValueError, TypeError):
                    continue
                changed = False
                for s in rec.get("seats", []):
                    if int(s.get("user_id", -1)) == uid and s.get("name") != deleted:
                        s["name"] = deleted
                        changed = True
                wins = rec.get("winners")
                if old_name and isinstance(wins, list):
                    for w in wins:
                        if isinstance(w, list) and w and w[0] == old_name:
                            w[0] = deleted
                            changed = True
                if changed:
                    pub.DB.q(
                        "UPDATE homegame_hands SET summary=? WHERE game_id=? AND hand_no=?",
                        (json.dumps(rec, separators=(",", ":")), gid, h["hand_no"]),
                    )
        pub.DB.q("DELETE FROM homegame_avatars WHERE user_id=?", (uid,))
        pub.DB.q("DELETE FROM homegame_names WHERE user_id=?", (uid,))
        pub.DB.q("DELETE FROM homegame_chat WHERE user_id=?", (uid,))
        pub.DB.q("DELETE FROM homegame_host_prefs WHERE user_id=?", (uid,))
        pub.DB.q("DELETE FROM homegame_club_members WHERE user_id=?", (uid,))
        pub.DB.q("DELETE FROM homegame_club_requests WHERE user_id=?", (uid,))
    with CTX.avatar_lock:
        CTX.avatar_urls.pop(uid, None)
    with CTX.names_lock:  # (their table name, and every club's nicknames they were in)
        CTX.chosen_names.pop(uid, None)
        CTX.club_nicks.clear()


_hooks = getattr(pub, "ACCOUNT_HOOKS", None)
if isinstance(_hooks, dict):
    _hooks["home_games"] = {
        "export": _account_export,
        "blockers": _account_blockers,
        "anonymize": _account_anonymize,
    }


# --- the parts split out of this module (HGB-006) ---------------------------------------------
# homegame_schema (the tables and their migrations), homegame_people (names, nicknames,
# pictures), homegame_clubs (clubs, who may see what), homegame_stats (histories, numbers,
# receipts, the lobby's rows), homegame_pages (the client's build and the pages) and
# homegame_routes (the /games/api router). Their code reaches every home-games name through
# this module (``hg.X``, looked up when it runs): patching ``homegame.X`` patches it for them
# too, and ``use_context`` swaps their state. Every name they define is re-exported here, so
# ``homegame.X`` keeps working for all of it. They are imported afresh with this module: a
# test that purges and re-imports homegame gets parts bound to the new module.
SPLIT_MODULES = (
    "homegame_schema",
    "homegame_people",
    "homegame_clubs",
    "homegame_stats",
    "homegame_pages",
    "homegame_bot",
    "homegame_routes",
)


def _import_split_modules() -> None:
    import importlib

    for part in SPLIT_MODULES:
        name = f"{__package__}.{part}"
        sys.modules.pop(name, None)  # (a part bound to an earlier import of this module)
        mod = importlib.import_module(name)
        for export in mod.__all__:
            globals()[export] = getattr(mod, export)


_import_split_modules()

# (the clubs' role cache forgets itself on any write to the club tables — PERF-004 —
# through a listener on the app's database, registered by `install`: the database is
# opened when an app installs the public layer, not when a module is imported — BE-007)


def _lock_checked(fn: Any) -> Any:
    """``fn(t, ...)`` that first checks ``t.lock`` is held (LOCK_CHECKS only)."""
    @functools.wraps(fn)
    def checked(t: LiveTable, *args: Any, **kw: Any) -> Any:
        _assert_lock_held(t)
        return fn(t, *args, **kw)

    checked.__hg_lock_checked__ = True  # type: ignore[attr-defined]
    return checked


def _locked_functions(ns: dict[str, Any]) -> list[str]:
    """The names the ``*_locked`` rule covers in a module namespace."""
    ours = {__name__} | {f"{__package__}.{part}" for part in SPLIT_MODULES}
    return sorted(
        name for name, fn in ns.items()
        if name.endswith("_locked") and inspect.isfunction(fn) and fn.__module__ in ours
    )


if LOCK_CHECKS:  # (HGB-018: in the test session every `*_locked` function checks its lock)
    for _name in _locked_functions(globals()):
        globals()[_name] = _lock_checked(globals()[_name])


class _HomegameModule(types.ModuleType):
    """``homegame.HUB``, ``homegame._WATCHDOG_THREAD`` … read and write the CURRENT
    context's state (HGB-006: other modules and the tests keep their names)."""

    def __getattr__(self, name: str) -> Any:
        attr = _LEGACY_STATE.get(name)
        if attr is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        return getattr(self.__dict__["CTX"], attr)

    def __setattr__(self, name: str, value: Any) -> None:
        attr = _LEGACY_STATE.get(name)
        if attr is not None:
            setattr(self.__dict__["CTX"], attr, value)
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _HomegameModule
