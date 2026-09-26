"""Private PLO5 double-board bomb-pot home games (PokerNow-style).

Installed only from ``public.install`` (public build). Access is the
admin-granted ``homegame_access`` flag, independent of subscription;
the HTTP middleware 404s the whole ``/games`` tree for everyone else.

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
"""

from __future__ import annotations

import asyncio
import html as html_mod
import json
import queue
import logging
import math
import os
import re
import secrets
import threading
import time
import zlib
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from plo5bp.actions import FOLD, GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.config import VARIANT_PLO5, GameConfig
from plo5bp.env import BombPotEnv, StepInfo
from plo5bp.ui.common import STREET_NAMES, position_name
from plo5bp.ui.hand_describe import describe_made_hand
from plo5bp.ui.runout import AWARD_SECS, board_equities, build_awards, display_pots, money_flows
from plo5bp.ui import fairdeal
from plo5bp.ui import public as pub

logger = logging.getLogger("plo5bp.ui.homegame")

BB_CHIPS = 10_000
TABLE_SEATS = 8

# Hard limits (review 2026-09-20 G4/G13). Every money field is capped far
# below what sqlite (i64) / the engine (u64) can hold, so a value that made
# it into memory can ALWAYS be persisted.
MAX_CENTS = 1_000_000_000  # $10M — per amount, and per player's total buy-in
MAX_NAME_LEN = 60
MAX_OPEN_TABLES_PER_USER = int(os.environ.get("PLO5BP_HOMEGAME_MAX_TABLES", "5"))
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
# Every action of every hand is graded against the network in the background
# (never on the request path, never during the hand): the same node distribution
# and the same scorer as the Trainer tab, from the ACTOR's own point of view.
GRADING_ON = os.environ.get("PLO5BP_HOMEGAME_GRADING", "1").strip().lower() not in (
    "0", "false", "no", "off",
)
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
    "games.play.js": "text/javascript; charset=utf-8",
    "games.sound.js": "text/javascript; charset=utf-8",
    "games.fair.js": "text/javascript; charset=utf-8",
    "games.css": "text/css; charset=utf-8",
}
# Hub eviction: nobody polling for this long => drop the in-memory table
# (it reloads from the DB on the next request).
HUB_CLOSED_EVICT_S = 60.0
HUB_IDLE_EVICT_S = 30 * 60.0
HUB_ABANDONED_EVICT_S = 6 * 3600.0  # even mid-hand: the hand is void
_SCHEMA = """
CREATE TABLE IF NOT EXISTS homegames (
  id TEXT PRIMARY KEY,
  host_user_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  num_seats INTEGER NOT NULL,
  sb_cents INTEGER NOT NULL,
  bb_cents INTEGER NOT NULL,
  ante_cents INTEGER NOT NULL,
  default_buyin_cents INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'open',
  running INTEGER NOT NULL DEFAULT 0,
  auto_stack_mode TEXT NOT NULL DEFAULT 'off',
  auto_stack_all_cents INTEGER NOT NULL DEFAULT 0,
  decision_secs INTEGER NOT NULL DEFAULT 0,
  street_pause_ms INTEGER NOT NULL DEFAULT 1500,
  button INTEGER NOT NULL DEFAULT 0,
  hand_no INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  closed_at TEXT,
  deal_delay_ms INTEGER NOT NULL DEFAULT 5000,
  time_bank_secs INTEGER NOT NULL DEFAULT 0,
  min_buyin_cents INTEGER NOT NULL DEFAULT 0,
  max_buyin_cents INTEGER NOT NULL DEFAULT 0,
  listed INTEGER NOT NULL DEFAULT 1,
  allow_rabbit INTEGER NOT NULL DEFAULT 1,
  approve_buyins INTEGER NOT NULL DEFAULT 0,
  topup_mode TEXT NOT NULL DEFAULT 'off',
  topup_target_cents INTEGER NOT NULL DEFAULT 0,
  topup_below_cents INTEGER NOT NULL DEFAULT 0,
  show_grades INTEGER NOT NULL DEFAULT 1,
  excluded INTEGER NOT NULL DEFAULT 0,
  allow_rathole INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS homegame_hands (
  game_id TEXT NOT NULL,
  hand_no INTEGER NOT NULL,
  ended_at TEXT NOT NULL,
  pot_cents INTEGER NOT NULL DEFAULT 0,
  summary TEXT NOT NULL,
  PRIMARY KEY (game_id, hand_no)
);
CREATE TABLE IF NOT EXISTS homegame_hand_results (
  game_id TEXT NOT NULL,
  hand_no INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  delta_cents INTEGER NOT NULL,
  showdown INTEGER NOT NULL DEFAULT 0,
  acc_sum REAL NOT NULL DEFAULT 0,
  acc_n INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (game_id, hand_no, user_id)
);
CREATE TABLE IF NOT EXISTS homegame_flows (
  game_id TEXT NOT NULL,
  hand_no INTEGER NOT NULL,
  payer INTEGER NOT NULL,
  payee INTEGER NOT NULL,
  chips INTEGER NOT NULL,
  PRIMARY KEY (game_id, hand_no, payer, payee)
);
CREATE TABLE IF NOT EXISTS homegame_fair (
  game_id TEXT NOT NULL,
  hand_no INTEGER NOT NULL,
  data TEXT NOT NULL,
  PRIMARY KEY (game_id, hand_no)
);
CREATE INDEX IF NOT EXISTS homegame_results_user ON homegame_hand_results(user_id);
CREATE INDEX IF NOT EXISTS homegame_flows_payer ON homegame_flows(payer);
CREATE INDEX IF NOT EXISTS homegame_flows_payee ON homegame_flows(payee);
CREATE TABLE IF NOT EXISTS homegame_players (
  game_id TEXT NOT NULL,
  user_id INTEGER NOT NULL,
  seat INTEGER,
  stack_chips INTEGER NOT NULL DEFAULT 0,
  sitting_out INTEGER NOT NULL DEFAULT 0,
  buyin_cents INTEGER NOT NULL DEFAULT 0,
  leftover_cents INTEGER NOT NULL DEFAULT 0,
  auto_stack_cents INTEGER NOT NULL DEFAULT 0,
  trusted INTEGER NOT NULL DEFAULT 0,
  topup_target_cents INTEGER NOT NULL DEFAULT 0,
  topup_below_cents INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (game_id, user_id)
);
CREATE TABLE IF NOT EXISTS homegame_ledger (
  id INTEGER PRIMARY KEY,
  game_id TEXT NOT NULL,
  user_id INTEGER NOT NULL,
  kind TEXT NOT NULL,
  amount_cents INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS homegame_chat (
  id INTEGER PRIMARY KEY,
  game_id TEXT NOT NULL,
  user_id INTEGER NOT NULL,
  body TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS homegame_clubs (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  owner_user_id INTEGER NOT NULL,
  invite_code TEXT NOT NULL UNIQUE,
  approve_joins INTEGER NOT NULL DEFAULT 0,
  is_main INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS homegame_club_members (
  club_id TEXT NOT NULL,
  user_id INTEGER NOT NULL,
  role TEXT NOT NULL DEFAULT 'member',
  joined_at TEXT NOT NULL,
  PRIMARY KEY (club_id, user_id)
);
CREATE INDEX IF NOT EXISTS homegame_club_members_user ON homegame_club_members(user_id);
CREATE TABLE IF NOT EXISTS homegame_club_requests (
  club_id TEXT NOT NULL,
  user_id INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  created_at TEXT NOT NULL,
  decided_at TEXT,
  PRIMARY KEY (club_id, user_id)
);
"""


def _ensure_schema() -> None:
    with pub.DB._lock:
        pub.DB._conn.executescript(_SCHEMA)
        cols = {
            r[1] for r in pub.DB._conn.execute("PRAGMA table_info(homegames)").fetchall()
        }
        if "running" not in cols:
            pub.DB._conn.execute(
                "ALTER TABLE homegames ADD COLUMN running INTEGER NOT NULL DEFAULT 0"
            )
        if "auto_stack_mode" not in cols:
            pub.DB._conn.execute(
                "ALTER TABLE homegames ADD COLUMN auto_stack_mode "
                "TEXT NOT NULL DEFAULT 'off'"
            )
        if "auto_stack_all_cents" not in cols:
            pub.DB._conn.execute(
                "ALTER TABLE homegames ADD COLUMN auto_stack_all_cents "
                "INTEGER NOT NULL DEFAULT 0"
            )
        if "decision_secs" not in cols:
            pub.DB._conn.execute(
                "ALTER TABLE homegames ADD COLUMN decision_secs "
                "INTEGER NOT NULL DEFAULT 0"
            )
        if "street_pause_ms" not in cols:
            pub.DB._conn.execute(
                "ALTER TABLE homegames ADD COLUMN street_pause_ms "
                "INTEGER NOT NULL DEFAULT 1500"
            )
        # 2026-09-21. Tables that predate server-side dealing were auto-dealt
        # by the host's browser, so they migrate to a 5 s delay (not manual).
        for col, ddl in (
            ("deal_delay_ms", "INTEGER NOT NULL DEFAULT 5000"),
            ("time_bank_secs", "INTEGER NOT NULL DEFAULT 0"),
            ("min_buyin_cents", "INTEGER NOT NULL DEFAULT 0"),
            ("max_buyin_cents", "INTEGER NOT NULL DEFAULT 0"),
            ("listed", "INTEGER NOT NULL DEFAULT 1"),
            ("allow_rabbit", "INTEGER NOT NULL DEFAULT 1"),
            ("approve_buyins", "INTEGER NOT NULL DEFAULT 0"),
            ("topup_mode", "TEXT NOT NULL DEFAULT 'off'"),
            ("topup_target_cents", "INTEGER NOT NULL DEFAULT 0"),
            ("topup_below_cents", "INTEGER NOT NULL DEFAULT 0"),
            ("show_grades", "INTEGER NOT NULL DEFAULT 1"),
            ("excluded", "INTEGER NOT NULL DEFAULT 0"),
            ("allow_rathole", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if col not in cols:
                pub.DB._conn.execute(f"ALTER TABLE homegames ADD COLUMN {col} {ddl}")
        pcols = {
            r[1]
            for r in pub.DB._conn.execute(
                "PRAGMA table_info(homegame_players)"
            ).fetchall()
        }
        if "auto_stack_cents" not in pcols:
            pub.DB._conn.execute(
                "ALTER TABLE homegame_players ADD COLUMN auto_stack_cents "
                "INTEGER NOT NULL DEFAULT 0"
            )
        rcols = {
            r[1]
            for r in pub.DB._conn.execute(
                "PRAGMA table_info(homegame_hand_results)"
            ).fetchall()
        }
        if "acc_sum" not in rcols:
            pub.DB._conn.execute(
                "ALTER TABLE homegame_hand_results ADD COLUMN acc_sum REAL NOT NULL DEFAULT 0"
            )
            pub.DB._conn.execute(
                "ALTER TABLE homegame_hand_results ADD COLUMN acc_n INTEGER NOT NULL DEFAULT 0"
            )
        for col in ("trusted", "topup_target_cents", "topup_below_cents"):
            if col not in pcols:
                pub.DB._conn.execute(
                    f"ALTER TABLE homegame_players ADD COLUMN {col} "
                    "INTEGER NOT NULL DEFAULT 0"
                )
        if "club_id" not in cols:  # 2026-09-25: every table belongs to a club
            pub.DB._conn.execute("ALTER TABLE homegames ADD COLUMN club_id TEXT")
        pub.DB._conn.execute("CREATE INDEX IF NOT EXISTS homegames_club ON homegames(club_id)")
        pub.DB._conn.commit()
    _migrate_clubs()


def _make_env(cfg: GameConfig) -> BombPotEnv:
    """Prod's env.py may predate ``obs_mode``; fall back to the 2-arg ctor."""
    try:
        return BombPotEnv(cfg, ev_runout_samples=0, obs_mode="minimal")
    except TypeError:
        return BombPotEnv(cfg, ev_runout_samples=0)


def _obs_dict(env: BombPotEnv) -> dict[str, Any]:
    fn = env._rs.observation_dict
    try:
        return dict(fn(skip_outcome_mc=True))
    except TypeError:
        return dict(fn())


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


def _display_name(user: Any) -> str:
    name = (user["name"] or "").strip()
    if name:
        return name
    email = (user["email"] or "").strip()
    return email.split("@")[0] if email else f"player-{user['id']}"


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
    name: str
    email: str
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
    turn_started_mono: float | None = None
    # The decision the running shot clock belongs to: (hand_no, action_seq).
    # The clock restarts only when this changes (review 2026-09-20 G5).
    turn_key: tuple[int, int] | None = None
    # Players leaving / being removed mid-hand: folded-or-checked by the
    # away logic, cashed out once the hand (and its runout) is over.
    pending_kicks: set[int] = field(default_factory=set)
    runout_active: bool = False
    runout_start_len: int = 3
    runout_started_mono: float | None = None
    leftover_stacks: list[int] = field(default_factory=list)
    terminal_pot: int = 0
    terminal_commit: list[int] = field(default_factory=list)
    pot_awards: list[dict[str, Any]] = field(default_factory=list)
    pots: list[dict[str, Any]] = field(default_factory=list)  # named layers, deepest first
    # (len_a, len_b) -> {seat: {"a": share, "b": share}}, computed ONCE per
    # all-in hand from the alive seats' holes (review 2026-09-20 G3).
    equity_by_len: dict[tuple[int, int], dict[int, dict[str, float]]] = field(
        default_factory=dict
    )
    env: BombPotEnv | None = None
    info: StepInfo | None = None
    phase: str = "waiting"  # waiting | in_hand | showdown
    hand_start_stacks: list[int] = field(default_factory=list)
    in_hand_mask: list[bool] = field(default_factory=list)
    # user id DEALT INTO each seat this hand (None = seat not dealt in). Own
    # cards are shown against this, never against who sits there now (G2).
    dealt_user_ids: list[int | None] = field(default_factory=list)
    # Terminal with >= 2 live hands = a real showdown; a fold-out is not (G1).
    showdown_reveal: bool = False
    # Actions applied this hand. With hand_no it names a decision: clients
    # echo both on /act and get 409 when they are stale (G8).
    action_seq: int = 0
    # Bumped on every state change; `epoch` is unique per in-memory load.
    # Clients drop responses older than the one they already applied (G8).
    rev: int = 0
    epoch: str = field(default_factory=lambda: secrets.token_hex(4))
    # A persist failed after memory had already advanced (engine paths);
    # the watchdog retries (review 2026-09-20 G4).
    persist_dirty: bool = False
    persist_retry_mono: float = 0.0
    last_access_mono: float = field(default_factory=time.monotonic)
    chat_times: dict[int, deque] = field(default_factory=dict)
    last_deltas: list[int] = field(default_factory=list)
    last_holes: list[list[int] | None] = field(default_factory=list)
    rabbit_available: bool = False
    rabbit_shown: bool = False
    rabbit_played_len: int = 3
    rabbit_full_a: list[int] = field(default_factory=list)
    rabbit_full_b: list[int] = field(default_factory=list)
    # --- 2026-09-21 -------------------------------------------------------
    deal_delay_secs: float = 0.0  # 0 = manual dealing
    time_bank_secs: int = 0  # per-seat reserve; 0 = off
    min_buyin_cents: int = 0  # 0 = no limit
    max_buyin_cents: int = 0
    listed: bool = True  # shown in every granted user's lobby
    allow_rabbit: bool = True
    # When the server will deal the next hand (monotonic), or None.
    next_deal_mono: float | None = None
    # The decision whose base clock has run out and is burning the bank.
    bank_key: tuple[int, int] | None = None
    bank_started_mono: float | None = None
    # Seats that chose to table their cards after the hand (phase showdown).
    shown_seats: set[int] = field(default_factory=set)
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
    # What the background grader needs to replay the hand being played: the
    # deal seed and the exact engine inputs. In MEMORY only — the seed would
    # reveal every card, so it is never persisted and never served.
    hand_seed: int = 0
    hand_actions: list = field(default_factory=list)
    # Verifiable shuffle. ``hand_deck`` = the 52 cards as dealt (memory only, like
    # the seed it replaces); ``fair_next`` = the sealed deck of the UPCOMING hand
    # and where its confirmation stands; ``fair_hand`` = the hand on the table.
    hand_deck: list = field(default_factory=list)
    fair_next: Any = None
    fair_hand: Any = None
    fair_hand_meta: dict = field(default_factory=dict)
    fair_capable: set = field(default_factory=set)       # user ids whose browser takes part
    fair_strikes: dict = field(default_factory=dict)     # user id -> missed reveals in a row
    fair_penalty_until: dict = field(default_factory=dict)  # user id -> hand_no
    fair_void_counts: dict = field(default_factory=dict)  # name -> voided shuffles this session
    # People without home-games access who asked to join from this table's link
    # (``homegame_join_requests``; admins see them). Re-read at most every
    # JOIN_REFRESH_S so an /admin grant clears them too.
    join_reqs: list = field(default_factory=list)
    join_checked_mono: float = 0.0
    club_info: dict[str, Any] | None = None
    club_checked_mono: float = 0.0
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def ante_chips(self) -> int:
        return cents_to_chips(self.ante_cents, self.bb_cents)

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


class Hub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tables: dict[str, LiveTable] = {}

    def get(self, game_id: str) -> LiveTable:
        with self._lock:
            t = self._tables.get(game_id)
            if t is None:
                t = _load_table(game_id)
                self._tables[game_id] = t
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
        hub used to grow forever). Clients poll every ~0.5 s, so "idle"
        means no browser has the table open. Everything that matters is in
        the DB; the next request reloads it (paused, between hands)."""
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
                    del self._tables[gid]
                    gone.append(gid)
        return gone


HUB = Hub()

_WATCHDOG_STOP = threading.Event()
_WATCHDOG_STARTED = False
_WATCHDOG_THREAD: threading.Thread | None = None
_EVICT_EVERY_S = 30.0


def _watchdog_loop() -> None:
    next_evict = time.monotonic() + _EVICT_EVERY_S
    while not _WATCHDOG_STOP.wait(0.25):
        try:
            with HUB._lock:
                tables = list(HUB._tables.values())
            for t in tables:
                with t.lock:
                    _timeout_tick_locked(t)
                    _settle_locked(t)
                    _flush_result_locked(t)
                    _apply_queued_topups_locked(t)
                    _expire_requests_locked(t)
                    _fair_tick_locked(t)
                    _auto_deal_tick_locked(t)
                    _retry_persist_locked(t)
            if time.monotonic() >= next_evict:
                next_evict = time.monotonic() + _EVICT_EVERY_S
                HUB.evict_idle()
        except Exception:  # noqa: BLE001 — never kill the clock thread
            logger.exception("homegame shot-clock watchdog")


def _start_watchdog() -> None:
    global _WATCHDOG_STARTED, _WATCHDOG_THREAD
    if _WATCHDOG_STARTED:
        return
    _WATCHDOG_STARTED = True
    _WATCHDOG_STOP.clear()
    _WATCHDOG_THREAD = threading.Thread(
        target=_watchdog_loop, name="homegame-clock", daemon=True
    )
    _WATCHDOG_THREAD.start()


def shutdown(timeout: float = 2.0) -> None:
    """Stop the clock thread (tests purge + reimport this module)."""
    global _WATCHDOG_STARTED
    _WATCHDOG_STOP.set()
    th = _WATCHDOG_THREAD
    if th is not None and th.is_alive() and th is not threading.current_thread():
        th.join(timeout)
    _WATCHDOG_STARTED = False


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
            name=_display_name(user),
            email=user["email"],
            stack_chips=int(p["stack_chips"]),
            sitting_out=bool(p["sitting_out"]),
            buyin_cents=int(p["buyin_cents"]),
            leftover_cents=int(p["leftover_cents"]),
            auto_stack_cents=int(
                p["auto_stack_cents"] if "auto_stack_cents" in p.keys() else 0
            ),
            time_bank_left=float(_row_int(row, "time_bank_secs", 0)),
            trusted=bool(_row_int(p, "trusted", 0)),
            topup_target_cents=_row_int(p, "topup_target_cents", 0),
            topup_below_cents=_row_int(p, "topup_below_cents", 0),
        )
    # (review 2026-09-20 G14) A table loaded from the DB has no hand in
    # memory (phase "waiting"), but `running=1` used to survive the restart:
    # the host saw only "Pause", nobody saw a deal button, and the game
    # looked dead. A reload always comes back PAUSED — the host presses
    # Start, which deals. (An interrupted hand is void: stacks were last
    # persisted at the previous hand's end.)
    was_running = bool(int(row["running"] if "running" in row.keys() else 0))
    if was_running:
        pub.DB.q("UPDATE homegames SET running=0 WHERE id=?", (game_id,))
    t = LiveTable(
        game_id=row["id"],
        host_user_id=int(row["host_user_id"]),
        name=row["name"],
        num_seats=n,
        sb_cents=int(row["sb_cents"]),
        bb_cents=int(row["bb_cents"]),
        ante_cents=int(row["ante_cents"]),
        default_buyin_cents=int(row["default_buyin_cents"]),
        status=row["status"],
        button=int(row["button"]),
        hand_no=int(row["hand_no"]),
        seats=seats,
        running=False,
        club_id=(row["club_id"] if "club_id" in row.keys() else None) or _main_club(),
        auto_stack_mode=_norm_auto_mode(
            row["auto_stack_mode"] if "auto_stack_mode" in row.keys() else "off"
        ),
        auto_stack_all_cents=int(
            row["auto_stack_all_cents"]
            if "auto_stack_all_cents" in row.keys()
            else 0
        ),
        decision_secs=int(
            row["decision_secs"] if "decision_secs" in row.keys() else 0
        ),
        street_pause_secs=max(
            0.3,
            (
                int(row["street_pause_ms"] if "street_pause_ms" in row.keys() else 1500)
                / 1000.0
            ),
        ),
        phase="waiting",
        deal_delay_secs=max(0.0, _row_int(row, "deal_delay_ms", 5000) / 1000.0),
        time_bank_secs=_row_int(row, "time_bank_secs", 0),
        min_buyin_cents=_row_int(row, "min_buyin_cents", 0),
        max_buyin_cents=_row_int(row, "max_buyin_cents", 0),
        listed=bool(_row_int(row, "listed", 1)),
        allow_rabbit=bool(_row_int(row, "allow_rabbit", 1)),
        approve_buyins=bool(_row_int(row, "approve_buyins", 0)),
        topup_mode=_norm_auto_mode(row["topup_mode"] if "topup_mode" in row.keys() else "off"),
        topup_all_target_cents=_row_int(row, "topup_target_cents", 0),
        topup_all_below_cents=_row_int(row, "topup_below_cents", 0),
        show_grades=bool(_row_int(row, "show_grades", 1)),
        allow_rathole=bool(_row_int(row, "allow_rathole", 0)),
    )
    # A hand dealt (hand_no is saved at the deal) but never recorded was cut short
    # by a restart. It is void — stacks are saved only when a hand ends, so everyone
    # has what they had before it — but say so: to the players it just vanished.
    if t.status == "open" and t.hand_no > 0:
        last = pub.DB.one("SELECT MAX(hand_no) AS n FROM homegame_hands WHERE game_id=?", (game_id,))
        if last is not None and int(last["n"] or 0) < t.hand_no:
            _emit(t, "run", f"Hand #{t.hand_no} was cut short (the server restarted) and doesn't count — "
                            "everyone has the chips they had before it. The host restarts the game.")
    return t


def _row_int(row: Any, key: str, default: int) -> int:
    """A column that older databases may not have yet."""
    try:
        v = row[key] if key in row.keys() else default
    except (IndexError, KeyError):
        v = default
    return int(default if v is None else v)


def _persist_meta(t: LiveTable) -> None:
    pub.DB.q(
        "UPDATE homegames SET host_user_id=?, status=?, running=?, "
        "auto_stack_mode=?, auto_stack_all_cents=?, decision_secs=?, "
        "street_pause_ms=?, "
        "button=?, hand_no=?, "
        "name=?, num_seats=?, ante_cents=?, default_buyin_cents=?, "
        "deal_delay_ms=?, time_bank_secs=?, min_buyin_cents=?, "
        "max_buyin_cents=?, listed=?, allow_rabbit=?, "
        "approve_buyins=?, topup_mode=?, topup_target_cents=?, topup_below_cents=?, "
        "show_grades=?, allow_rathole=?, "
        "closed_at=CASE WHEN ?= 'closed' THEN COALESCE(closed_at, ?) ELSE closed_at END "
        "WHERE id=?",
        (
            t.host_user_id,
            t.status,
            1 if t.running else 0,
            t.auto_stack_mode,
            int(t.auto_stack_all_cents or 0),
            int(t.decision_secs or 0),
            int(round(float(t.street_pause_secs or 1.5) * 1000)),
            t.button,
            t.hand_no,
            t.name,
            int(t.num_seats),
            int(t.ante_cents),
            int(t.default_buyin_cents),
            int(round(float(t.deal_delay_secs or 0.0) * 1000)),
            int(t.time_bank_secs or 0),
            int(t.min_buyin_cents or 0),
            int(t.max_buyin_cents or 0),
            1 if t.listed else 0,
            1 if t.allow_rabbit else 0,
            1 if t.approve_buyins else 0,
            _norm_auto_mode(t.topup_mode),
            int(t.topup_all_target_cents or 0),
            int(t.topup_all_below_cents or 0),
            1 if t.show_grades else 0,
            1 if t.allow_rathole else 0,
            t.status,
            pub._now(),
            t.game_id,
        ),
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


def _persist_safe(t: LiveTable) -> None:
    """Persist meta + seats on an ENGINE path; never raises.

    (review 2026-09-20 G4) Once the engine has stepped, memory cannot be
    rolled back — the env has no undo — so a failed write must not surface
    as a half-applied action. Memory stays the truth, ``persist_dirty`` is
    set, and the watchdog retries. (Inputs are capped, so every in-memory
    value IS persistable; only transient DB errors land here.)"""
    try:
        _persist_seats(t)
        t.persist_dirty = False
    except Exception:  # noqa: BLE001
        logger.exception("homegame persist failed for %s; will retry", t.game_id)
        t.persist_dirty = True
        t.persist_retry_mono = time.monotonic() + 2.0


def _retry_persist_locked(t: LiveTable) -> None:
    if t.persist_dirty and time.monotonic() >= t.persist_retry_mono:
        _persist_safe(t)


_SNAPSHOT_FIELDS = (
    "host_user_id", "status", "running", "auto_stack_mode",
    "auto_stack_all_cents", "decision_secs", "street_pause_secs", "button",
    "hand_no", "turn_started_mono", "turn_key", "phase",
    "name", "num_seats", "ante_cents", "default_buyin_cents", "deal_delay_secs",
    "time_bank_secs", "min_buyin_cents", "max_buyin_cents", "listed",
    "allow_rabbit", "approve_buyins", "topup_mode", "topup_all_target_cents",
    "topup_all_below_cents", "show_grades",
)


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
    snap = {f: getattr(t, f) for f in _SNAPSHOT_FIELDS}
    seats = [replace(p) if p is not None else None for p in t.seats]
    kicks = set(t.pending_kicks)
    try:
        with pub.DB.transaction():
            yield
    except BaseException:
        for f, v in snap.items():
            setattr(t, f, v)
        t.seats = seats
        t.pending_kicks = kicks
        raise
    t.rev += 1


def _ledger_add(t: LiveTable, uid: int, kind: str, amount_cents: int) -> None:
    pub.DB.q(
        "INSERT INTO homegame_ledger(game_id,user_id,kind,amount_cents,created_at)"
        " VALUES(?,?,?,?,?)",
        (t.game_id, uid, kind, int(amount_cents), pub._now()),
    )


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
        return cents_to_chips(int(p.auto_stack_cents), t.bb_cents)
    if _norm_auto_mode(t.topup_mode) != "off" and int(p.topup_target_cents or 0) > 0:
        below = int(p.topup_below_cents or 0) or int(p.topup_target_cents)
        if chips_to_cents(chips, t.bb_cents) < below:
            return max(chips, cents_to_chips(int(p.topup_target_cents), t.bb_cents))
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
            target_cents = int(p.auto_stack_cents or 0) if set_on else 0
            if target_cents <= 0:
                # AUTO TOP-UP (2026-09-22): only ever UP, and only once the
                # stack has dropped below the player's threshold — winnings
                # stay on the table, so this is not a rathole.
                tgt = int(p.topup_target_cents or 0) if top_on else 0
                if tgt <= 0:
                    continue
                have = chips_to_cents(int(p.stack_chips), t.bb_cents)
                if have >= (int(p.topup_below_cents or 0) or tgt):
                    continue
                add_cents = tgt - have
                if add_cents <= 0 or p.buyin_cents + add_cents > MAX_CENTS:
                    continue
                p.buyin_cents += add_cents
                p.stack_chips += cents_to_chips(add_cents, t.bb_cents)
                _ledger_add(t, p.user_id, "rebuy", add_cents)
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
                _ledger_add(t, p.user_id, "rebuy", delta_cents)
            else:
                moved = min(moved, int(p.stack_chips))
                p.leftover_cents += delta_cents
                p.stack_chips -= moved
                _ledger_add(t, p.user_id, "cashout", delta_cents)
        _persist_seats(t)


def _auto_stack_host_locked(t: LiveTable, uid: int, body: dict) -> None:
    _require_open(t)
    if uid != t.host_user_id:
        raise HTTPException(
            status_code=400, detail="only the host can change auto-stack settings"
        )
    body = body or {}
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


def _auto_topup_host_locked(t: LiveTable, uid: int, body: dict) -> None:
    """Host side of auto top-up: the mode (off / host / player), the values for
    everyone in host mode, and per-player overrides."""
    _require_open(t)
    if uid != t.host_user_id:
        raise HTTPException(
            status_code=400, detail="only the host can change auto top-up settings"
        )
    body = body or {}
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


def _auto_stack_self_locked(t: LiveTable, uid: int, cents: int) -> None:
    _require_open(t)
    if t.auto_stack_mode != "player":
        raise HTTPException(
            status_code=400, detail="players cannot set auto-stack in this mode"
        )
    if t.player(uid) is None:
        raise HTTPException(status_code=400, detail="not seated")
    target = _validate_auto_stack_cents(t, cents)
    with _mutation(t):
        p = t.player(uid)
        assert p is not None
        p.auto_stack_cents = target
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
    street_commit = [0] * 16
    cur_street: int | None = None
    out: list[dict[str, Any]] = []
    for rec in raw.get("history") or []:
        seat, action, chips, street = int(rec[0]), int(rec[1]), int(rec[2]), int(rec[3])
        if street != cur_street:
            # (review 2026-09-20 G9) "Raise to" is a PER-STREET total: the
            # accumulator used to run across streets, so a $5 turn bet after
            # $10 of flop action read "Raise to $15".
            street_commit = [0] * 16
            cur_street = street
        street_commit[seat] += chips
        if action == FOLD:
            label = "Fold"
        elif action == 1:  # CHECK_CALL
            label = "Check" if chips == 0 else f"Call {_fmt_cents(chips_to_cents(chips, bb_cents))}"
        else:
            label = f"Raise to {_fmt_cents(chips_to_cents(street_commit[seat], bb_cents))}"
            if action == 7:
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
            "name": live[1].name if live else _display_name(r),
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


def _ledger_rows(t: LiveTable) -> list[dict[str, Any]]:
    return _ledger_state(t)[0]


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
    id from ever naming two different seals across a restart or an eviction."""
    return f"{t.game_id}:{int(hand_no)}:{int(attempt)}:{t.epoch}"


def _fair_prepare_locked(t: LiveTable) -> None:
    """Seal the deck of the upcoming hand as soon as there is an upcoming hand."""
    if not FAIR_ON or t.status != "open" or t.phase == "in_hand":
        return
    nxt = t.fair_next
    if nxt is not None and nxt.hand_no == t.hand_no + 1:
        if nxt.sealed.num_seats != t.num_seats:  # the slot map depends on the seat count
            _fair_void_locked(t, "the table was resized")
        return
    t.fair_next = FairPending(
        sealed=fairdeal.SealedDeck.create(_fair_hand_id(t, t.hand_no + 1, 1), t.num_seats),
        hand_no=t.hand_no + 1,
    )
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
    names = [_seat_name(t, i) for i in (seats or [])]
    for i in seats or []:
        uid = nxt.commit_users.get(i)
        if uid is None:
            continue
        nxt.barred.add(int(uid))
        n = int(t.fair_strikes.get(int(uid), 0)) + 1
        t.fair_strikes[int(uid)] = n
        if n >= FAIR_STRIKES:
            t.fair_penalty_until[int(uid)] = int(t.hand_no) + 1 + FAIR_PENALTY_HANDS
            t.fair_strikes.pop(int(uid), None)
    for nm in names:
        t.fair_void_counts[nm] = int(t.fair_void_counts.get(nm, 0)) + 1
    voids = list(nxt.voids) + [{
        "attempt": nxt.attempt, "seal": nxt.sealed.seal, "reason": reason, "names": names,
    }]
    was_pending = bool(nxt.pending)
    t.fair_next = FairPending(
        sealed=fairdeal.SealedDeck.create(_fair_hand_id(t, nxt.hand_no, nxt.attempt + 1), t.num_seats),
        hand_no=nxt.hand_no, attempt=nxt.attempt + 1, barred=set(nxt.barred), voids=voids,
    )
    _emit(t, "fair", "Shuffle redone — " + (
        f"{', '.join(names)} didn't confirm in time" if names else reason))
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
            },
        }
    fh = t.fair_hand
    if fh is not None and t.phase in ("in_hand", "showdown"):
        meta = t.fair_hand_meta or {}
        out["hand"] = {
            "hand_no": int(meta.get("hand_no") or t.hand_no), "hand_id": fh.hand_id,
            "seal": fh.seal, "lock": fh.lock, "num_seats": fh.num_seats,
            "contributors": [st for st, _ in fh.locked],
            "names": list(meta.get("names") or []),
            "voids": list(meta.get("voids") or []),
            "open": _fair_openings(fh, visible),
        }
    out["void_counts"] = dict(t.fair_void_counts)
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
        v = _view(t, viewer_id)
        for srow in v["seats"]:
            visible += [c for c in (srow.get("hole") or []) if isinstance(c, int) and c >= 0]
        for b in ("a", "b"):
            bd = v["board"][b]
            visible += [c for c in list(bd["flop"]) + [bd["turn"], bd["river"]] if isinstance(c, int) and c >= 0]
    else:
        _require_member(t, viewer_id)
        rec_row = pub.DB.one("SELECT summary FROM homegame_hands WHERE game_id=? AND hand_no=?",
                             (t.game_id, int(hand_no)))
        if rec_row is not None:
            rec = _hand_for_viewer(json.loads(rec_row["summary"]), viewer_id, bool(t.show_grades))
            for srow in rec.get("seats") or []:
                visible += [c for c in (srow.get("hole") or []) if isinstance(c, int) and c >= 0]
            visible += [int(c) for c in (rec.get("board_a") or []) + (rec.get("board_b") or [])]
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
    except HTTPException:
        pass  # not dealable after all (e.g. auto-stack could not top up)


def _runout_shown_len(t: LiveTable) -> int:
    if not t.runout_active:
        return 5
    start = max(3, int(t.runout_start_len or 3))
    pause = float(t.street_pause_secs if t.street_pause_secs is not None else 1.5)
    if not (pause > 0) or not math.isfinite(pause):
        return 5
    started = t.runout_started_mono if t.runout_started_mono is not None else time.monotonic()
    extra = int((time.monotonic() - started) / pause)
    return min(5, start + extra)


def _runout_award_index(t: LiveTable) -> int:
    n = len(t.pot_awards or [])
    if not t.runout_active:
        return n
    if _runout_shown_len(t) < 5:
        return -1
    pause = float(t.street_pause_secs if t.street_pause_secs is not None else 1.5)
    if not math.isfinite(pause):
        pause = 0.0
    start = max(3, int(t.runout_start_len or 3))
    river_at = (5 - start) * max(pause, 0.0)
    elapsed = time.monotonic() - (t.runout_started_mono or time.monotonic())
    into = elapsed - river_at
    if into < 0:
        return -1
    return min(n, int(into / AWARD_SECS))


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


def _view(t: LiveTable, viewer_id: int) -> dict[str, Any]:
    t.seen[int(viewer_id)] = time.monotonic()  # presence (server-side dealing)
    if int(viewer_id) not in t.names:
        u = pub._user_by_id(int(viewer_id))
        t.names[int(viewer_id)] = _display_name(u) if u is not None else f"player-{viewer_id}"
    _timeout_tick_locked(t)
    _settle_locked(t)
    _flush_result_locked(t)
    raw: dict[str, Any] = {}
    actor = None
    info = t.info
    holes: list[list[int] | None] = [None] * t.num_seats
    in_hand = list(t.in_hand_mask or [])
    in_hand += [False] * (t.num_seats - len(in_hand))
    dealt = list(t.dealt_user_ids or [])
    dealt += [None] * (t.num_seats - len(dealt))
    # (review 2026-09-20 G1) Cards are tabled only at a REAL showdown: two or
    # more live hands at terminal. "phase == showdown" alone also covers a
    # fold-out, where the winner's hand used to be shown to the whole table
    # (and to unseated viewers) — every successful bluff was exposed.
    reveal = t.phase == "showdown" and bool(t.showdown_reveal)
    if t.env is not None and t.phase in ("in_hand", "showdown"):
        raw = _obs_dict(t.env)
        actor_raw = raw.get("actor")
        actor = int(actor_raw) if actor_raw is not None else None
        all_holes = t.env.all_hole_cards()
        folded = [bool(x) for x in raw.get("folded", [])]
        folded += [True] * (t.num_seats - len(folded))
        for i in range(t.num_seats):
            if i >= len(all_holes) or not in_hand[i]:
                # The engine deals EVERY seat, dealt-in or not; a masked-out
                # seat's five cards are live-deck information.
                continue
            # (review 2026-09-20 G2) "Own" = the user who was DEALT this hand,
            # not whoever sits in the seat now: a seated-but-not-dealt player
            # used to see five dead cards mid-hand, and a newcomer taking the
            # seat after the hand saw the previous occupant's mucked cards.
            own = dealt[i] is not None and dealt[i] == viewer_id
            occupant = t.seats[i].user_id if t.seats[i] is not None else None
            if occupant is not None and occupant != dealt[i]:
                # Someone else has taken the seat since: the old hand is
                # not theirs to show (or to sit behind), face-down included.
                continue
            # Tabled voluntarily after the hand ("show cards") — the player's
            # own choice, so it also covers a fold-out winner and a folded hand.
            shown = t.phase == "showdown" and i in t.shown_seats
            if own or shown or (reveal and not folded[i]):
                holes[i] = _sorted_hole([int(c) for c in all_holes[i]])
            else:
                holes[i] = [-1] * len(all_holes[i])  # facedown
    elif reveal and t.last_holes:
        holes = [_sorted_hole(h) if h and h[0] >= 0 else h for h in t.last_holes]

    ba_src = list(raw.get("board_a") or t.rabbit_full_a or [])
    bb_src = list(raw.get("board_b") or t.rabbit_full_b or [])
    if t.phase == "showdown" and t.rabbit_full_a:
        ba_src = list(t.rabbit_full_a)
        bb_src = list(t.rabbit_full_b)
        if t.runout_active:
            n = max(3, _runout_shown_len(t))
            ba_src, bb_src = ba_src[:n], bb_src[:n]
        elif not t.rabbit_shown:
            n = max(3, int(t.rabbit_played_len or 3))
            ba_src, bb_src = ba_src[:n], bb_src[:n]
    ba_src = [int(c) for c in ba_src]
    bb_src = [int(c) for c in bb_src]
    ba = _split_board(ba_src)
    bb = _split_board(bb_src)
    street = STREET_NAMES.get(int(raw["street"]), "flop") if raw else None
    if t.runout_active:
        street = {3: "flop", 4: "turn", 5: "river"}.get(len(ba_src), street)

    viewer_seat = t.seat_of(viewer_id)
    hero_seat = viewer_seat if viewer_seat is not None else 0

    # --- all-in runout: what has been revealed SO FAR -----------------------
    # (review 2026-09-20 G11) While the streets/awards are still animating,
    # nothing in the payload may run ahead of them: no final deltas, no
    # final ledger, no award list, no unrevealed board card.
    runout_live = bool(t.runout_active and _runout_blocking(t))
    award_idx = _runout_award_index(t) if t.runout_active else -1
    n_awards = len(t.pot_awards or [])
    start_stacks = list(t.hand_start_stacks or [])
    start_stacks += [0] * (t.num_seats - len(start_stacks))
    display_stacks = [0] * t.num_seats
    if runout_live:
        base = list(t.leftover_stacks or [])
        base += [0] * (t.num_seats - len(base))
        display_stacks = base[: t.num_seats]
        for step in (t.pot_awards or [])[: max(0, award_idx)]:
            for k, v in (step.get("shares") or {}).items():
                si = int(k)
                if 0 <= si < t.num_seats:
                    display_stacks[si] += int(v)

    # Ledger + settled stack cents. During the reveal the ledger stays on the
    # hand-START stacks (exactly as it does while a hand is being played).
    ledger, seat_cents = _ledger_state(t, start_stacks if runout_live else None)
    settled = t.phase != "in_hand" and not runout_live

    seats_out: list[dict[str, Any]] = []
    now_seen = time.monotonic()
    for i in range(t.num_seats):
        p = t.seats[i]
        if p is not None and not in_hand[i]:
            stack_chips = p.stack_chips  # not in this hand (sat / reloaded mid-hand)
        elif runout_live:
            stack_chips = display_stacks[i]
        elif t.phase == "showdown" and p is not None:
            stack_chips = p.stack_chips
        elif raw and t.phase == "in_hand":
            stack_chips = int(raw["stacks"][i])
        else:
            stack_chips = p.stack_chips if p else 0
        street_commit = int(raw["street_commit"][i]) if raw else 0
        hole = holes[i]
        made = None
        if hole and hole[0] >= 0:
            made = [
                describe_made_hand(hole, ba_src),
                describe_made_hand(hole, bb_src),
            ]
        if settled and p is not None and i in seat_cents:
            stack_cents = seat_cents[i]  # agrees with the ledger to the cent
        else:
            stack_cents = chips_to_cents(stack_chips, t.bb_cents)
        seats_out.append({
            "seat": i,
            "empty": p is None,
            "user_id": p.user_id if p else None,
            "name": p.name if p else None,
            "sitting_out": bool(p.sitting_out) if p else False,
            "stack_chips": stack_chips,
            "stack_cents": stack_cents,
            "committed_this_street_chips": street_commit,
            "committed_this_street_cents": chips_to_cents(street_commit, t.bb_cents),
            "folded": bool(raw["folded"][i]) if raw else False,
            "all_in": bool(raw["all_in"][i]) if raw else False,
            "in_hand": bool(in_hand[i]),
            "is_actor": actor is not None and i == actor,
            "is_hero": viewer_seat is not None and i == viewer_seat,
            "is_host": p is not None and p.user_id == t.host_user_id,
            "position": position_name(
                i, t.button, t.num_seats,
                {j for j, m in enumerate(in_hand) if m} if any(in_hand) else None,
            ),
            "hole": hole,
            "hand_desc": made,
            "auto_stack_cents": int(p.auto_stack_cents or 0) if p else 0,
            "pending_remove": bool(p and (p.user_id in t.pending_kicks or p.leave_after_hand)),
            "leaving": bool(p and p.leave_after_hand),
            "equity_a": None,
            "equity_b": None,
            "trusted": bool(p.trusted) if p else False,
            "topup_target_cents": int(p.topup_target_cents or 0) if p else 0,
            "topup_below_cents": int(p.topup_below_cents or 0) if p else 0,
            # your own queued chips only — nobody else needs to see them
            "queued_topup_cents": (
                int(p.queued_topup_cents or 0) if p and p.user_id == viewer_id else 0
            ),
            "queued_remove_cents": (
                int(p.queued_remove_cents or 0) if p and p.user_id == viewer_id else 0
            ),
            # the host sees who is waiting on them, right on the seat
            "request": (
                next(({"id": r["id"], "kind": r["kind"], "amount_cents": r["amount_cents"]}
                      for r in t.requests if p is not None and r["user_id"] == p.user_id), None)
                if viewer_id == t.host_user_id else None
            ),
            "present": bool(p and now_seen - t.seen.get(p.user_id, -1e9) <= PRESENCE_WINDOW_S),
            "reserved_by": next(
                (r["name"] for r in t.requests if r["kind"] == "sit" and r["seat"] == i), None
            ) if p is None else None,
            "sit_out_next": bool(p.sit_out_next) if p else False,
            "bank_left_secs": round(float(p.time_bank_left or 0.0), 1) if p else 0.0,
            "shown": bool(t.phase == "showdown" and i in t.shown_seats),
        })

    if t.runout_active:
        # (review 2026-09-20 G3) Computed once per hand in `_capture_rabbit`
        # from the ALIVE holes — the same numbers for every viewer.
        eq_map = t.equity_by_len.get((len(ba_src), len(bb_src))) or {}
        for i, eq in eq_map.items():
            if 0 <= i < len(seats_out):
                seats_out[i]["equity_a"] = eq["a"]
                seats_out[i]["equity_b"] = eq["b"]

    legal = {"fold": False, "check_call": False, "raise": False}
    raise_bounds = {"min_chips": 0, "max_chips": 0, "min_cents": 0, "max_cents": 0}
    to_call_chips = 0
    my_turn = (
        t.phase == "in_hand"
        and actor is not None
        and viewer_seat is not None
        and actor == viewer_seat
        and info is not None
    )
    if my_turn and info is not None:
        gm = info.gate_mask
        legal = {
            "fold": bool(gm[GATE_FOLD]),
            "check_call": bool(gm[GATE_CHECK_CALL]),
            "raise": bool(gm[GATE_RAISE]),
        }
        min_c = int(info.min_raise_chips)
        max_c = int(info.max_raise_chips)
        if legal["raise"] and min_c == 0 and max_c > 0:
            min_c = max_c
        raise_bounds = {
            "min_chips": min_c,
            "max_chips": max_c,
            "min_cents": chips_to_cents(min_c, t.bb_cents),
            "max_cents": chips_to_cents(max_c, t.bb_cents),
        }
        street_commit = int(raw["street_commit"][actor])
        stack = int(raw["stacks"][actor])
        bet_to_call = int(raw["bet_to_call"])
        to_call_chips = min(max(0, bet_to_call - street_commit), stack)

    if runout_live:
        # Public so far: what each dealt-in seat put in, plus awards shown.
        deltas_cents = [
            chips_to_cents(display_stacks[i] - start_stacks[i], t.bb_cents)
            if in_hand[i] else 0
            for i in range(t.num_seats)
        ]
    else:
        deltas_cents = [chips_to_cents(d, t.bb_cents) for d in (t.last_deltas or [])]

    if runout_live:
        awarded = sum(
            int(st.get("chips") or 0)
            for st in (t.pot_awards or [])[: max(0, award_idx)]
        )
        pot_chips = max(0, int(t.terminal_pot) - awarded)
    elif t.phase == "showdown":
        pot_chips = 0
    else:
        pot_chips = int(raw["pot"]) if raw else 0

    award_step = None
    if t.runout_active and 0 <= award_idx < n_awards:
        award_step = t.pot_awards[award_idx]
    if not t.runout_active:
        shown_awards: list[dict[str, Any]] = []
    elif runout_live:
        # Award steps only start once all five cards are out, so a released
        # step can never name an unrevealed board card.
        shown_awards = list(t.pot_awards[: max(0, award_idx + 1)])
    else:
        shown_awards = list(t.pot_awards or [])
    pause = float(t.street_pause_secs if t.street_pause_secs is not None else 1.5)
    shown_len = _runout_shown_len(t) if t.runout_active else 0
    start_len = max(3, int(t.runout_start_len or 3))
    elapsed = (
        time.monotonic() - t.runout_started_mono
        if t.runout_active and t.runout_started_mono is not None
        else 0.0
    )
    if t.runout_active and shown_len < 5 and pause > 0:
        next_in = pause - (elapsed % pause)
    else:
        next_in = 0.0

    eligible = sum(_eligible_mask(t, start_stacks if runout_live else None))
    blocking = _runout_blocking(t)
    can_show = False
    if t.phase == "showdown" and t.env is not None and raw:
        for i in range(t.num_seats):
            if dealt[i] is None or dealt[i] != viewer_id or not in_hand[i]:
                continue
            occupant = t.seats[i].user_id if t.seats[i] is not None else None
            if occupant is not None and occupant != dealt[i]:
                continue
            tabled = i in t.shown_seats or (reveal and not bool(raw["folded"][i]))
            can_show = not tabled
    bank_active, bank_left = _bank_state(t)
    now_mono = time.monotonic()
    now_wall = time.time()
    last_hand_no = t.hand_no if (t.phase != "in_hand" and not blocking) else t.hand_no - 1
    _fair_prepare_locked(t)
    visible_cards = [int(c) for h in holes if h for c in h if int(c) >= 0] + ba_src + bb_src
    return {
        "fair": _fair_view(t, viewer_id, visible_cards),
        "id": t.game_id,
        "name": t.name,
        "status": t.status,
        "phase": t.phase,
        "hand_no": t.hand_no,
        "action_seq": t.action_seq,
        "rev": t.rev,
        "epoch": t.epoch,
        "num_seats": t.num_seats,
        "button_seat": t.button,
        "hero_seat": hero_seat,
        "actor": actor,
        "my_user_id": viewer_id,
        "my_seat": viewer_seat,
        "is_host": viewer_id == t.host_user_id,
        "is_member": any(row["user_id"] == viewer_id for row in ledger),
        "stakes": {
            "sb_cents": t.sb_cents,
            "bb_cents": t.bb_cents,
            "ante_cents": t.ante_cents,
            "default_buyin_cents": t.default_buyin_cents,
            "bb_chips": BB_CHIPS,
        },
        "seats": seats_out,
        "board": {"a": ba, "b": bb},
        "pot_chips": pot_chips,
        "pot_cents": chips_to_cents(pot_chips, t.bb_cents),
        "settled_pot_chips": (
            int(raw["pot"]) - sum(int(x) for x in raw["street_commit"]) if raw else 0
        ),
        "street": street,
        "history": _history_entries(raw, t.bb_cents) if raw else [],
        "legal": legal,
        "raise_bounds": raise_bounds,
        "to_call_chips": to_call_chips,
        "to_call_cents": chips_to_cents(to_call_chips, t.bb_cents),
        "street_commit_chips": (
            int(raw["street_commit"][actor]) if raw and actor is not None else 0
        ),
        "hand_deltas_cents": deltas_cents,
        "ledger": ledger,
        "running": bool(t.running),
        "can_deal": (
            t.status == "open"
            and t.running
            and t.phase != "in_hand"
            and not blocking
            and viewer_seat is not None
            and eligible >= 2
        ),
        "eligible_count": eligible,
        "chat": _chat_messages(t.game_id),
        "can_rabbit": bool(
            t.phase == "showdown" and t.rabbit_available and not t.rabbit_shown
        ),
        "rabbit_shown": bool(t.rabbit_shown),
        "auto_stack": {
            "mode": _norm_auto_mode(t.auto_stack_mode),
            "all_cents": int(t.auto_stack_all_cents or 0),
        },
        "decision_secs": int(t.decision_secs or 0),
        "turn_remaining_secs": _turn_remaining_secs(t),
        "street_pause_secs": pause,
        "runout": {
            "active": bool(t.runout_active),
            "start_len": start_len,
            "shown_len": shown_len if t.runout_active else 0,
            "pause_secs": pause,
            "next_in_secs": round(next_in, 2),
            "award_index": award_idx,
            "award_count": n_awards,
            "award_step": award_step,
            "blocking": blocking,
        },
        "pot_awards": shown_awards,
        "pots": list(t.pots or []) if t.runout_active else [],
        "live_pots": (
            _live_pots(raw, in_hand, t.num_seats)
            if raw and t.phase == "in_hand" and not t.runout_active else []
        ),
        # --- 2026-09-21 ----------------------------------------------------
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
        # buy-in approval: the host sees the queue, a player their own request
        "needs_approval": bool(t.approve_buyins) and not _is_trusted_cached(t, viewer_id),
        "requests": (
            [{k: r[k] for k in ("id", "user_id", "name", "kind", "seat", "amount_cents")}
             for r in t.requests]
            if viewer_id == t.host_user_id else []
        ),
        "my_request": next(
            ({k: r[k] for k in ("id", "kind", "seat", "amount_cents")}
             for r in t.requests if r["user_id"] == viewer_id), None,
        ),
        # people asking to join the home games from this table's link (admins only)
        "join_requests": _table_join_requests(t, viewer_id),
        "club": _table_club(t, viewer_id),
        "spectators": sorted(
            t.names.get(uid, "?") for uid, last in t.seen.items()
            if now_mono - last <= PRESENCE_WINDOW_S and t.seat_of(uid) is None
        ),
        "next_deal_in_secs": (
            round(max(0.0, t.next_deal_mono - now_mono), 2)
            if t.next_deal_mono is not None
            else None
        ),
        "time_bank": {
            "secs": int(t.time_bank_secs or 0),
            "active": bool(bank_active),
            "remaining_secs": round(bank_left, 1),
        },
        "can_show": bool(can_show),
        "last_hand_no": last_hand_no if last_hand_no >= 1 else None,
        "events": list(t.events),
        "reactions": [
            {"id": r["id"], "seat": r["seat"], "emote": r["emote"]}
            for r in t.reactions
            if now_wall - float(r["ts"]) <= REACTION_TTL_S
        ],
    }


def _is_trusted_cached(t: LiveTable, uid: int) -> bool:
    """`_is_trusted` without a DB hit on every poll: only an UNSEATED viewer
    needs the lookup, and only while approval is on."""
    if not t.approve_buyins:
        return True
    return _is_trusted(t, uid)


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
    if uid != t.host_user_id:
        raise HTTPException(status_code=400, detail="only the host can start or pause")
    was = bool(t.running)
    with _mutation(t):
        t.running = bool(running)
        _persist_meta(t)
    t.next_deal_mono = None
    if not running:
        _fair_tick_locked(t)  # a deal waiting on its shuffle is called off (announced)
    if was != bool(running):
        _emit(t, "run", "Game started" if running else (
            "Game pauses after this hand" if t.phase == "in_hand" else "Game paused"))
    if t.running and not _hand_busy(t) and sum(_eligible_mask(t)) >= 2:
        _deal_locked(t)


def _deal_locked(t: LiveTable) -> None:
    _require_open(t)
    if not t.running:
        raise HTTPException(status_code=400, detail="game is paused")
    if t.phase == "in_hand":
        raise HTTPException(status_code=400, detail="hand already in progress")
    if _runout_blocking(t):
        # (review 2026-09-20 G7) `can_deal` was only a hint in the payload:
        # POST /deal itself never checked, so any seated player could cut
        # everyone's river and pot awards short.
        raise HTTPException(status_code=400, detail="wait for the runout to finish")
    if FAIR_ON and t.fair_next is not None and t.fair_next.pending:
        return  # this deal is already waiting on its shuffle (an impatient second click)
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
    if FAIR_ON:
        # Verifiable shuffle: devices that take part get a moment to confirm.
        # A table nobody's browser takes part in (scripts, tests) deals at once.
        _fair_prepare_locked(t)
        if _fair_begin_locked(t, mask):
            return
    _deal_now_locked(t)


def _deal_now_locked(t: LiveTable) -> None:
    """Put the hand on the table (the shuffle, if any, is settled)."""
    _require_open(t)
    if not t.running:
        raise HTTPException(status_code=400, detail="game is paused")
    if t.phase == "in_hand" or _runout_blocking(t):
        raise HTTPException(status_code=400, detail="hand already in progress")
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
    button = _next_button(t, mask)
    stacks = tuple(s.stack_chips if s is not None else 0 for s in t.seats)
    cfg = GameConfig(
        num_seats=t.num_seats,
        starting_stack=0,
        starting_stacks=stacks,
        ante=t.ante_chips,
        bb=BB_CHIPS,
        variant=VARIANT_PLO5,
    )
    env = _make_env(cfg)
    nxt = t.fair_next if FAIR_ON else None
    if nxt is not None and (nxt.hand_no != t.hand_no + 1 or nxt.sealed.num_seats != t.num_seats):
        _fair_void_locked(t, "the table changed before the deal")
        raise HTTPException(status_code=409, detail="the shuffle is being redone")
    if nxt is not None:
        if not nxt.sealed.lock:
            nxt.sealed.set_lock({})  # nobody contributed: the seal still pins every card
        deck = nxt.sealed.finish(dict(nxt.reveals))
        _, info = env.reset_with_deck(deck, button, in_hand_mask=mask)
        t.hand_seed = 0
        t.hand_deck = list(deck)
        t.fair_hand = nxt.sealed
        t.fair_hand_meta = {
            "hand_no": int(t.hand_no) + 1,
            "names": [[st, _seat_name(t, st)] for st, _ in nxt.sealed.locked],
            "voids": list(nxt.voids),
        }
        t.fair_next = None
        try:  # the transcript outlives the process (history, later audits)
            pub.DB.q(
                "INSERT OR REPLACE INTO homegame_fair(game_id,hand_no,data) VALUES(?,?,?)",
                (t.game_id, int(t.hand_no) + 1, json.dumps(
                    {"sealed": nxt.sealed.to_store(), "meta": t.fair_hand_meta},
                    separators=(",", ":"))),
            )
        except Exception:  # noqa: BLE001 — never stop a hand over its paperwork
            logger.exception("homegame fair transcript not stored (table %s)", t.game_id)
    else:
        seed = secrets.randbits(63)
        _, info = env.reset(seed, button, in_hand_mask=mask)
        t.hand_seed = int(seed)
        t.hand_deck = []
        t.fair_hand = None
        t.fair_hand_meta = {}
        t.fair_next = None
    t.hand_actions = []
    # The engine accepted the hand — commit it to the table.
    t.button = button
    t.env = env
    t.info = info
    t.phase = "showdown" if env.is_terminal() else "in_hand"
    t.hand_no += 1
    t.action_seq = 0
    t.rev += 1
    t.turn_key = None
    t.turn_started_mono = None
    t.hand_start_stacks = list(stacks)
    t.in_hand_mask = mask
    # (review 2026-09-20 G2) WHO was dealt each seat — own-card visibility is
    # checked against this, never against the seat's current occupant.
    t.dealt_user_ids = [
        s.user_id if (s is not None and mask[i]) else None
        for i, s in enumerate(t.seats)
    ]
    t.showdown_reveal = False
    t.last_deltas = []
    t.last_holes = [None] * t.num_seats
    t.rabbit_available = False
    t.rabbit_shown = False
    t.rabbit_played_len = 3
    t.rabbit_full_a = []
    t.rabbit_full_b = []
    t.runout_active = False
    t.runout_start_len = 3
    t.runout_started_mono = None
    t.leftover_stacks = []
    t.terminal_pot = 0
    t.terminal_commit = []
    t.pot_awards = []
    t.equity_by_len = {}
    t.shown_seats = set()
    t.next_deal_mono = None
    t.bank_key = None
    t.bank_started_mono = None
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
    payouts = [int(x) for x in t.env._rs.payouts()]
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
    holes: list[list[int] | None] = [None] * t.num_seats
    if t.showdown_reveal:
        all_holes = t.env.all_hole_cards()
        for i in alive:
            holes[i] = _sorted_hole([int(c) for c in all_holes[i]])
    t.last_holes = holes
    for i, p in enumerate(t.seats):
        if p is not None:
            dealt = bool(t.in_hand_mask and i < len(t.in_hand_mask) and t.in_hand_mask[i])
            if dealt:
                start = t.hand_start_stacks[i] if i < len(t.hand_start_stacks) else p.stack_chips
                p.stack_chips = max(0, int(start) + (payouts[i] if i < len(payouts) else 0))
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
    _persist_safe(t)
    _record_hand_locked(t, raw, payouts, folded)
    _settle_locked(t)  # no-op while an all-in runout is still revealing
    _flush_result_locked(t)
    _fair_prepare_locked(t)


def _record_hand_locked(
    t: LiveTable, raw: dict[str, Any], payouts: list[int], folded: list[bool]
) -> None:
    """Persist the finished hand (history + per-player results) and queue its
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
        seats = []
        winners: list[tuple[str, int]] = []
        for i in range(t.num_seats):
            if not mask[i] or dealt[i] is None:
                continue
            p = t.seats[i]
            name = p.name if (p is not None and p.user_id == dealt[i]) else None
            if name is None:
                u = pub._user_by_id(int(dealt[i]))
                name = _display_name(u) if u is not None else f"player-{dealt[i]}"
            delta = int(payouts[i]) if i < len(payouts) else 0
            start = int(t.hand_start_stacks[i]) if i < len(t.hand_start_stacks) else 0
            is_folded = bool(folded[i]) if i < len(folded) else True
            seats.append({
                "seat": i,
                "user_id": int(dealt[i]),
                "name": name,
                "start_cents": chips_to_cents(start, t.bb_cents),
                "start_chips": start,
                "delta_cents": chips_to_cents(delta, t.bb_cents),
                "folded": is_folded,
                "shown": bool(t.showdown_reveal and not is_folded),
                "hole": _sorted_hole([int(c) for c in all_holes[i]]) if i < len(all_holes) else [],
            })
            if delta > 0:
                winners.append((name, chips_to_cents(delta, t.bb_cents)))
        pot_cents = chips_to_cents(sum(commit), t.bb_cents)
        actions = _history_entries(raw, t.bb_cents)
        for k, a in enumerate(actions):  # who decided: the player, or the clock
            if k < len(t.hand_actions):
                a["auto"] = not t.hand_actions[k][3]
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
            "hand_no": int(t.hand_no),
            "ended_at": pub._now(),
            "button": int(t.button),
            "num_seats": int(t.num_seats),
            "bb_cents": int(t.bb_cents),
            "ante_cents": int(t.ante_cents),
            "pot_cents": pot_cents,
            "showdown": bool(t.showdown_reveal),
            "board_a": full_a[:n_board],
            "board_b": full_b[:n_board],
            "actions": actions,
            "ante_chips": int(t.ante_chips),
            "bb_chips": BB_CHIPS,
            "flows": [
                {"from": int(a), "to": int(b), "cents": chips_to_cents(int(v), t.bb_cents)}
                for (a, b), v in sorted(flows.items())
            ],
            "grades": None,  # filled in by the background grader
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
        with pub.DB.transaction():
            pub.DB.q(
                "INSERT OR REPLACE INTO homegame_hands(game_id,hand_no,ended_at,"
                "pot_cents,summary) VALUES(?,?,?,?,?)",
                (t.game_id, int(t.hand_no), record["ended_at"], pot_cents,
                 json.dumps(record, separators=(",", ":"))),
            )
            for s in seats:
                pub.DB.q(
                    "INSERT OR REPLACE INTO homegame_hand_results(game_id,hand_no,"
                    "user_id,delta_cents,showdown) VALUES(?,?,?,?,?)",
                    (t.game_id, int(t.hand_no), s["user_id"], s["delta_cents"],
                     1 if s["shown"] else 0),
                )
            uid_of = {s["seat"]: s["user_id"] for s in seats}
            for (a, b), v in flows.items():
                if a in uid_of and b in uid_of and v > 0:
                    pub.DB.q(
                        "INSERT OR REPLACE INTO homegame_flows(game_id,hand_no,payer,"
                        "payee,chips) VALUES(?,?,?,?,?)",
                        (t.game_id, int(t.hand_no), uid_of[a], uid_of[b], int(v)),
                    )
        _enqueue_grading(t, mask, uid_of)
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

_GRADE_Q: "queue.Queue[dict | None]" = queue.Queue()
_GRADE_THREAD: threading.Thread | None = None


def _enqueue_grading(t: LiveTable, mask: list[bool], uid_of: dict[int, int]) -> None:
    """Hand over a finished hand to the grader (cheap: a snapshot of plain
    values — the worker never touches the live table)."""
    if not GRADING_ON or not t.hand_actions or not (t.hand_seed or t.hand_deck):
        return
    _GRADE_Q.put({
        "game_id": t.game_id, "hand_no": int(t.hand_no), "seed": int(t.hand_seed),
        "deck": list(t.hand_deck or []),
        "button": int(t.button), "num_seats": int(t.num_seats),
        "stacks": [int(x) for x in (t.hand_start_stacks or [])],
        "ante": int(t.ante_chips), "mask": [bool(x) for x in mask],
        "actions": list(t.hand_actions), "uid_of": dict(uid_of),
    })
    _start_grader()


def _start_grader() -> None:
    global _GRADE_THREAD
    if _GRADE_THREAD is not None and _GRADE_THREAD.is_alive():
        return
    _GRADE_THREAD = threading.Thread(target=_grader_loop, name="homegame-grader", daemon=True)
    _GRADE_THREAD.start()


def _grader_loop() -> None:
    while not _WATCHDOG_STOP.is_set():
        try:
            job = _GRADE_Q.get(timeout=0.5)
        except queue.Empty:
            continue
        if job is None:
            return
        try:
            _store_grades(job, grade_hand(job))
        except Exception:  # noqa: BLE001 — cosmetic; never take the table down
            logger.exception("homegame grading failed (%s #%s)", job.get("game_id"), job.get("hand_no"))
        finally:
            _GRADE_Q.task_done()


def grade_hand(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Replay the hand in a full-observation env and score every PLAYER decision
    with the Trainer's scorer against the served PLO5 model's node distribution
    (each node from the actor's own seat — exactly what Study would show them)."""
    from plo5bp.sizing import PLO_ANCHOR_SPEC
    from plo5bp.ui import server as srv
    from plo5bp.ui import trainer as tr

    model = srv.MODEL
    device = next(model.parameters()).device
    cfg = GameConfig(
        num_seats=job["num_seats"], starting_stack=0,
        starting_stacks=tuple(job["stacks"]), ante=job["ante"], bb=BB_CHIPS,
        variant=VARIANT_PLO5,
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


def _store_grades(job: dict[str, Any], grades: list[dict[str, Any]]) -> None:
    row = pub.DB.one(
        "SELECT summary FROM homegame_hands WHERE game_id=? AND hand_no=?",
        (job["game_id"], job["hand_no"]),
    )
    if row is None:
        return
    rec = json.loads(row["summary"])
    rec["grades"] = grades
    per_seat: dict[int, list[float]] = {}
    for g in grades:
        per_seat.setdefault(int(g["seat"]), []).append(float(g["score"]))
    with pub.DB.transaction():
        pub.DB.q(
            "UPDATE homegame_hands SET summary=? WHERE game_id=? AND hand_no=?",
            (json.dumps(rec, separators=(",", ":")), job["game_id"], job["hand_no"]),
        )
        for seat, scores in per_seat.items():
            uid = job["uid_of"].get(seat)
            if uid is None:
                continue
            pub.DB.q(
                "UPDATE homegame_hand_results SET acc_sum=?, acc_n=? "
                "WHERE game_id=? AND hand_no=? AND user_id=?",
                (float(sum(scores)), len(scores), job["game_id"], job["hand_no"], int(uid)),
            )


def wait_for_grading(timeout: float = 20.0) -> bool:
    """Tests: block until the grader has drained its queue."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if _GRADE_Q.unfinished_tasks == 0:
            return True
        time.sleep(0.05)
    return False


def _flush_result_locked(t: LiveTable) -> None:
    """Release the finished hand's result line — only once its runout has
    finished revealing (review 2026-09-20 G11: nothing runs ahead of it)."""
    pending = getattr(t, "pending_result", None)
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
    t.leftover_stacks = [int(x) for x in (raw.get("stacks") or [])]
    t.terminal_pot = int(raw.get("pot") or 0)
    t.terminal_commit = [int(x) for x in (raw.get("total_commit") or [])]
    t.pot_awards = []
    t.pots = []
    t.equity_by_len = {}
    t.runout_active = False
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
    all_holes = t.env.all_hole_cards() if t.env is not None else []
    holes: list[list[int] | None] = [None] * t.num_seats
    for i in alive:
        if i < len(all_holes):
            holes[i] = [int(c) for c in all_holes[i]]
    folded_full = [True] * t.num_seats
    for i in alive:
        folded_full[i] = False
    commit = list(t.terminal_commit)
    if len(commit) < t.num_seats:
        commit.extend([0] * (t.num_seats - len(commit)))
    t.pot_awards = build_awards(
        holes,
        folded_full,
        commit[: t.num_seats],
        full_a if len(full_a) >= 3 else full_a,
        full_b if len(full_b) >= 3 else full_b,
        t.button,
    )
    # The award steps carry the index of the pot they pay from (``build_awards``).
    t.pots = _named_pots(display_pots(commit[: t.num_seats], folded_full))
    _compute_runout_equities(t, {i: holes[i] for i in alive if holes[i]}, full_a, full_b)


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


def _compute_runout_equities(
    t: LiveTable, alive_holes: dict[int, list[int]], full_a: list[int], full_b: list[int]
) -> None:
    """Per-board equities for every street the runout will show.

    (review 2026-09-20 G3) ONCE per all-in hand, here, from the ALIVE seats'
    holes. It used to run inside every poll under the table lock, from the
    holes visible to that viewer — so a folded viewer's own cards entered as
    a contender (wrong numbers) and the per-viewer cache key made
    alternating polls recompute 0.5-2.9 s of pure Python each time. With the
    per-board marginal enumeration in runout.py the whole thing is ~0.1 s
    worst case (a flop all-in), a few ms from the turn. Cosmetic: a failure
    here must never stop the hand from finishing."""
    t.equity_by_len = {}
    if len(alive_holes) < 2:
        return
    try:
        for n in range(max(3, int(t.runout_start_len or 3)), 6):
            ba, bb = full_a[:n], full_b[:n]
            seed = zlib.crc32(f"{t.game_id}:{t.hand_no}:{n}".encode())
            t.equity_by_len[(len(ba), len(bb))] = board_equities(
                alive_holes, ba, bb, seed=seed
            )
    except Exception:  # noqa: BLE001
        logger.exception("runout equities failed (table %s)", t.game_id)
        t.equity_by_len = {}


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
        raise HTTPException(status_code=400, detail="not your turn")
    me = t.seats[seat]
    _apply_action_locked(t, gate, raise_chips, by_player=True)
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
        raise HTTPException(status_code=400, detail="seat taken")
    existing = t.seat_of(int(user["id"]))
    if existing is not None:
        raise HTTPException(status_code=400, detail="already seated")
    if buyin_cents < t.bb_cents:
        raise HTTPException(status_code=400, detail="buy-in must be at least 1 bb")
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
    prev_auto = 0
    if prev is not None:
        try:
            prev_auto = int(prev["auto_stack_cents"] or 0)
        except (KeyError, IndexError, TypeError):
            prev_auto = 0
    if t.auto_stack_mode == "host":
        auto = int(t.auto_stack_all_cents or 0)
    else:
        auto = prev_auto
    if t.topup_mode == "host":
        top = (int(t.topup_all_target_cents or 0), int(t.topup_all_below_cents or 0))
    else:
        top = (
            _row_int(prev, "topup_target_cents", 0) if prev is not None else 0,
            _row_int(prev, "topup_below_cents", 0) if prev is not None else 0,
        )
    p = Seat(
        user_id=int(user["id"]),
        name=_display_name(user),
        email=user["email"],
        stack_chips=chips,
        sitting_out=False,
        buyin_cents=prev_buyin + buyin_cents,
        leftover_cents=leftover,
        auto_stack_cents=auto,
        time_bank_left=float(max(0, int(t.time_bank_secs or 0))),
        trusted=bool(trust) or (bool(_row_int(prev, "trusted", 0)) if prev is not None else False),
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
    if cents < t.bb_cents:
        raise HTTPException(status_code=400, detail="buy-in must be at least 1 bb")
    if kind == "sit":
        if seat is None or seat < 0 or seat >= t.num_seats:
            raise HTTPException(status_code=400, detail="invalid seat")
        if t.seats[seat] is not None:
            raise HTTPException(status_code=400, detail="seat taken")
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
    name = _display_name(user)
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
    player is told, and the usual buy-in limits apply to the new amount."""
    _require_open(t)
    if uid != t.host_user_id:
        raise HTTPException(status_code=400, detail="only the host can approve buy-ins")
    req = next((r for r in t.requests if r["id"] == int(req_id)), None)
    if req is None:
        raise HTTPException(status_code=404, detail="that request is gone")
    cents = int(req["amount_cents"])
    if approve and amount_cents is not None and int(amount_cents) != cents:
        cents = int(amount_cents)
        if cents < t.bb_cents or cents > MAX_CENTS:
            raise HTTPException(status_code=400, detail="buy-in must be at least 1 bb")
    t.requests = [r for r in t.requests if r["id"] != req["id"]]
    t.rev += 1
    if not approve:
        _emit(t, "request", f"Host declined {req['name']}'s request")
        return
    user = pub._user_by_id(int(req["user_id"]))
    if user is None:
        raise HTTPException(status_code=404, detail="that player no longer exists")
    if cents != int(req["amount_cents"]):
        _emit(t, "request", f"Host approved {req['name']} for {_fmt_cents(cents)} (asked {_fmt_cents(int(req['amount_cents']))})")
    if req["kind"] == "sit":
        _sit_locked(t, user, int(req["seat"]), cents, trust=trust)
    else:
        if trust:
            _set_trust_locked(t, uid, int(req["user_id"]), True)
        _topup_locked(t, int(req["user_id"]), cents, queue_ok=True)


def _set_trust_locked(t: LiveTable, uid: int, target_uid: int, on: bool) -> None:
    """Trusted players buy in, top up and auto top-up without asking."""
    _require_open(t)
    if uid != t.host_user_id:
        raise HTTPException(status_code=400, detail="only the host can trust a player")
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
    if on:
        # whatever they were waiting for goes through now
        for r in [r for r in t.requests if r["user_id"] == target_uid]:
            try:
                _resolve_request_locked(t, uid, r["id"], True, False)
            except HTTPException:
                pass


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
    if amount_cents < t.bb_cents:
        raise HTTPException(status_code=400, detail="rebuy must be at least 1 bb")
    have = chips_to_cents(p.stack_chips, t.bb_cents) + int(p.queued_topup_cents or 0)
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
        _ledger_add(t, uid, "cashout", amount_cents)
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
                _ledger_add(t, p.user_id, "cashout", cents)
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
                _ledger_add(t, p.user_id, "rebuy", cents)
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
    if amount_cents < t.bb_cents:
        raise HTTPException(status_code=400, detail="rebuy must be at least 1 bb")
    if p.buyin_cents + amount_cents > MAX_CENTS:
        raise HTTPException(status_code=400, detail="buy-in limit reached")
    _check_buyin_limits(
        t, amount_cents, chips_to_cents(p.stack_chips, t.bb_cents), fresh=False
    )
    with _mutation(t):
        p = t.player(uid)
        assert p is not None
        p.stack_chips += cents_to_chips(amount_cents, t.bb_cents)
        p.buyin_cents += amount_cents
        _ledger_add(t, uid, "rebuy", amount_cents)
        _persist_player(t, t.seat_of(uid), p, True)
    _sync_idle_stack(t, t.seat_of(uid))
    _emit(t, "rebuy", f"{p.name} added {_fmt_cents(amount_cents)}", seat=t.seat_of(uid))


def _sit_out_locked(
    t: LiveTable, uid: int, on: bool, *, target_uid: int | None = None,
    next_hand: bool = False,
) -> None:
    _require_open(t)
    who = int(target_uid) if target_uid is not None else int(uid)
    if target_uid is not None and uid != t.host_user_id:
        raise HTTPException(
            status_code=400, detail="only the host can sit another player out"
        )
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
    if uid != t.host_user_id:
        raise HTTPException(
            status_code=400, detail="only the host can remove a player"
        )
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


def _set_decision_secs_locked(t: LiveTable, uid: int, secs: int) -> None:
    _require_open(t)
    if uid != t.host_user_id:
        raise HTTPException(
            status_code=400, detail="only the host can set decision time"
        )
    n = _valid_decision_secs(secs)
    with _mutation(t):
        t.decision_secs = n
        _persist_meta(t)
    if t.phase == "in_hand":
        _start_clock_locked(t, restart=True)  # the host changed the rules
    else:
        t.turn_started_mono = None


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


def _set_street_pause_locked(t: LiveTable, uid: int, secs: float) -> None:
    _require_open(t)
    if uid != t.host_user_id:
        raise HTTPException(
            status_code=400, detail="only the host can set runout speed"
        )
    n = float(secs)
    # `not (a <= n <= b)` also rejects NaN, which sails through `n < a or
    # n > b` and then 500'd every GET of the table (review 2026-09-20 G4).
    if not math.isfinite(n) or not (0.3 <= n <= 5.0):
        raise HTTPException(
            status_code=400, detail="runout pause must be between 0.3 and 5 seconds"
        )
    with _mutation(t):
        t.street_pause_secs = n
        _persist_meta(t)


def _close_locked(t: LiveTable, uid: int) -> None:
    if uid != t.host_user_id:
        raise HTTPException(status_code=400, detail="only the host can close the table")
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
    t.env = None
    t.info = None
    t.runout_active = False


def _host_fold_locked(t: LiveTable, uid: int) -> None:
    if uid != t.host_user_id:
        raise HTTPException(status_code=400, detail="only the host can fold a player")
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
    # Keep the stored hand in step, so history shows what the table saw.
    try:
        row = pub.DB.one(
            "SELECT summary FROM homegame_hands WHERE game_id=? AND hand_no=?",
            (t.game_id, int(t.hand_no)),
        )
        if row is not None:
            rec = json.loads(row["summary"])
            for s in rec.get("seats") or []:
                if int(s.get("seat", -1)) == seat:
                    s["shown"] = True
            pub.DB.q(
                "UPDATE homegame_hands SET summary=? WHERE game_id=? AND hand_no=?",
                (json.dumps(rec, separators=(",", ":")), t.game_id, int(t.hand_no)),
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
    if uid != t.host_user_id:
        raise HTTPException(status_code=400, detail="only the host can hand over the table")
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
    if uid != t.host_user_id:
        raise HTTPException(status_code=400, detail="only the host can change table settings")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="invalid settings")
    changed: list[str] = []
    with _mutation(t):
        if body.get("name") is not None:
            name = " ".join(str(body["name"]).split())
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
        if body.get("ante_cents") is not None:
            ante = _parse_cents(body, "ante_cents")
            if ante <= 0 or cents_to_chips(ante, t.bb_cents) <= 0:
                raise HTTPException(status_code=400, detail="ante must be positive")
            if ante != t.ante_cents:
                t.ante_cents = ante
                changed.append(f"ante is now {_fmt_cents(ante)} (next hand)")
        lo = _parse_cents(body, "min_buyin_cents", t.min_buyin_cents)
        hi = _parse_cents(body, "max_buyin_cents", t.max_buyin_cents)
        dflt = _parse_cents(body, "default_buyin_cents", t.default_buyin_cents)
        _validate_buyin_window(t.bb_cents, lo, hi, dflt)
        if (lo, hi, dflt) != (t.min_buyin_cents, t.max_buyin_cents, t.default_buyin_cents):
            t.min_buyin_cents, t.max_buyin_cents, t.default_buyin_cents = lo, hi, dflt
            changed.append("buy-in limits changed")
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
            pause = _parse_secs(body, "street_pause_secs", 1.5)
            if not (0.3 <= pause <= 5.0):
                raise HTTPException(
                    status_code=400,
                    detail="runout pause must be between 0.3 and 5 seconds",
                )
            t.street_pause_secs = pause
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
        _persist_meta(t)
    if body.get("decision_secs") is not None:
        if t.phase == "in_hand":
            _start_clock_locked(t, restart=True)  # the host changed the rules
        else:
            t.turn_started_mono = None
    if not t.approve_buyins and t.requests:
        # approval was switched off: everything that was waiting goes through
        for r in list(t.requests):
            try:
                _resolve_request_locked(t, uid, r["id"], True, False)
            except HTTPException:
                pass
    for line in changed:
        _emit(t, "settings", f"Host: {line}")


def _validate_buyin_window(bb_cents: int, lo: int, hi: int, dflt: int) -> None:
    if dflt < bb_cents:
        raise HTTPException(status_code=400, detail="default buy-in must be at least 1 bb")
    if lo and lo < bb_cents:
        raise HTTPException(status_code=400, detail="minimum buy-in must be at least 1 bb")
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


def _valid_num_seats(n: int) -> int:
    n = int(n)
    if not (MIN_SEATS <= n <= TABLE_SEATS):
        raise HTTPException(
            status_code=400, detail=f"seats must be {MIN_SEATS}–{TABLE_SEATS}"
        )
    return n


def _resize_locked(t: LiveTable, n: int) -> None:
    """Change the seat count (caller is inside ``_mutation``). Between hands
    only; shrinking needs the removed seats empty. The button stays on a seat
    that still exists."""
    n = _valid_num_seats(n)
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
    # Per-hand arrays are sized to the table that played them: drop them, the
    # finished hand is no longer drawn after a reshape.
    t.env = None
    t.info = None
    t.phase = "waiting"
    t.in_hand_mask = []
    t.dealt_user_ids = []
    t.hand_start_stacks = []
    t.last_deltas = []
    t.last_holes = []
    t.shown_seats = set()
    t.runout_active = False
    t.pot_awards = []
    t.rabbit_available = False
    t.rabbit_full_a = []
    t.rabbit_full_b = []


def _hand_for_viewer(
    rec: dict[str, Any], viewer_id: int, show_all_grades: bool = True
) -> dict[str, Any]:
    """A stored hand as THIS viewer may see it: own cards, plus hands tabled
    at showdown or shown voluntarily. Everything else is face-down — same
    rule as the live table (review 2026-09-20 G1/G2)."""
    out = dict(rec)
    seats = []
    for s in rec.get("seats") or []:
        s2 = dict(s)
        mine = int(s.get("user_id") or -1) == int(viewer_id)
        if not (mine or s.get("shown")):
            s2["hole"] = None
        s2["is_me"] = mine
        s2.pop("user_id", None)
        seats.append(s2)
    out["seats"] = seats
    my_seats = {int(s["seat"]) for s in seats if s.get("is_me")}
    grades = rec.get("grades")
    if grades is not None and not show_all_grades:
        grades = [g for g in grades if int(g.get("seat", -1)) in my_seats]
    out["grades"] = grades
    out["grades_public"] = bool(show_all_grades)
    return out


def _hands_visible_upto(t: LiveTable) -> int:
    """Newest hand number whose result may be served: never the hand being
    played, never one whose runout is still revealing (G11)."""
    if t.phase == "in_hand" or _runout_blocking(t):
        return int(t.hand_no) - 1
    return int(t.hand_no)


def _hands_list(t: LiveTable, viewer_id: int, before: int | None, limit: int) -> dict[str, Any]:
    upto = _hands_visible_upto(t)
    if before is not None:
        upto = min(upto, int(before) - 1)
    limit = max(1, min(HANDS_PAGE_MAX, int(limit)))
    rows = pub.DB.q(
        "SELECT hand_no, ended_at, pot_cents, summary FROM homegame_hands "
        "WHERE game_id=? AND hand_no<=? ORDER BY hand_no DESC LIMIT ?",
        (t.game_id, upto, limit),
    )
    hands = []
    for r in rows:
        try:
            rec = _hand_for_viewer(json.loads(r["summary"]), viewer_id, t.show_grades)
        except Exception:  # noqa: BLE001
            continue
        me = next((s for s in rec["seats"] if s.get("is_me")), None)
        hands.append({
            "hand_no": int(r["hand_no"]),
            "ended_at": r["ended_at"],
            "pot_cents": int(r["pot_cents"]),
            "showdown": bool(rec.get("showdown")),
            "board_a": rec.get("board_a") or [],
            "board_b": rec.get("board_b") or [],
            "winners": [
                {"name": s["name"], "delta_cents": s["delta_cents"]}
                for s in rec["seats"] if int(s.get("delta_cents") or 0) > 0
            ],
            "my_delta_cents": int(me["delta_cents"]) if me else None,
            "my_hole": me.get("hole") if me else None,
            "my_accuracy": _seat_accuracy(rec.get("grades"), me["seat"]) if me else None,
        })
    stats = pub.DB.q(
        "SELECT r.user_id, COUNT(*) hands, SUM(CASE WHEN r.delta_cents>0 THEN 1 ELSE 0 END) wins, "
        "SUM(r.showdown) showdowns, MAX(r.delta_cents) biggest, SUM(r.acc_sum) acc_sum, "
        "SUM(r.acc_n) acc_n, u.name, u.email "
        "FROM homegame_hand_results r JOIN users u ON u.id=r.user_id "
        "WHERE r.game_id=? AND r.hand_no<=? GROUP BY r.user_id ORDER BY hands DESC",
        (t.game_id, _hands_visible_upto(t)),
    )
    return {
        "hands": hands,
        "more": bool(hands) and hands[-1]["hand_no"] > 1 and len(hands) == limit,
        "stats": [
            {
                "name": _display_name(r),
                "is_me": int(r["user_id"]) == int(viewer_id),
                "hands": int(r["hands"] or 0),
                "wins": int(r["wins"] or 0),
                "showdowns": int(r["showdowns"] or 0),
                "biggest_win_cents": max(0, int(r["biggest"] or 0)),
                "accuracy": (
                    round(float(r["acc_sum"] or 0) / int(r["acc_n"]), 1)
                    if int(r["acc_n"] or 0) else None
                ),
                "graded": int(r["acc_n"] or 0),
            }
            for r in stats
        ],
        "h2h": _table_h2h(t),
    }


def _seat_accuracy(grades: list | None, seat: int) -> float | None:
    mine = [float(g["score"]) for g in (grades or []) if int(g.get("seat", -1)) == int(seat)]
    return round(sum(mine) / len(mine), 1) if mine else None


def _table_h2h(t: LiveTable) -> list[dict[str, Any]]:
    """Net money between every pair at this table: ``to`` is up ``cents`` on
    ``from``. Hands still revealing are excluded (G11)."""
    rows = pub.DB.q(
        "SELECT payer, payee, SUM(chips) chips FROM homegame_flows "
        "WHERE game_id=? AND hand_no<=? GROUP BY payer, payee",
        (t.game_id, _hands_visible_upto(t)),
    )
    gross = {(int(r["payer"]), int(r["payee"])): int(r["chips"] or 0) for r in rows}
    names: dict[int, str] = {}
    out = []
    for (a, b), v in gross.items():
        net = v - gross.get((b, a), 0)
        if net <= 0:
            continue
        for uid in (a, b):
            if uid not in names:
                u = pub._user_by_id(uid)
                names[uid] = _display_name(u) if u is not None else f"player-{uid}"
        out.append({"from": names[a], "to": names[b],
                    "cents": chips_to_cents(net, t.bb_cents)})
    out.sort(key=lambda x: -x["cents"])
    return out


# --- the club: everyone's numbers, side by side ------------------------------------------
#
# Home games are a private circle (admin-granted), so stats are open inside it:
# every member sees every player's accuracy, profit and loss, the money between
# every pair, and can browse anyone's hands (cards still follow the reveal
# rule). Sessions an admin took out of the record (``homegames.excluded`` —
# test tables) count nowhere; that is a soft flag and can be undone.


def _revealing_now() -> list[tuple[str, int]]:
    """(game_id, hand_no) of hands whose runout is still being shown: their
    result must not surface through an aggregate a few seconds early (G11)."""
    with HUB._lock:
        tables = list(HUB._tables.values())
    out = []
    for tb in tables:
        with tb.lock:
            if _runout_blocking(tb):
                out.append((tb.game_id, int(tb.hand_no)))
    return out


def _club_clause(clubs: list[str] | None, col: str = "g.club_id") -> tuple[str, list[Any]]:
    """`` AND g.club_id IN (...)`` for a stats scope (None = no restriction)."""
    if clubs is None:
        return "", []
    if not clubs:
        return " AND 0", []
    return f" AND {col} IN ({','.join('?' * len(clubs))})", list(clubs)


def _community(viewer_id: int, club_id: str, can_manage: bool) -> dict[str, Any]:
    """ONE club's numbers (rankings never mix clubs). ``can_manage`` = the club's
    owner: sees excluded sessions and may exclude / restore them."""
    skip = _revealing_now()
    guard = "".join(" AND NOT (r.game_id=? AND r.hand_no=?)" for _ in skip)
    gargs = [x for pair in skip for x in pair]
    rows = pub.DB.q(
        "SELECT r.user_id, u.name, u.email, COUNT(*) hands, SUM(r.delta_cents) net, "
        "SUM(r.acc_sum) a, SUM(r.acc_n) n, SUM(CASE WHEN r.delta_cents>0 THEN 1 ELSE 0 END) wins, "
        "COUNT(DISTINCT r.game_id) sessions, MAX(r.delta_cents) best "
        "FROM homegame_hand_results r JOIN homegames g ON g.id=r.game_id "
        f"JOIN users u ON u.id=r.user_id WHERE g.excluded=0 AND g.club_id=?{guard} GROUP BY r.user_id",
        tuple([club_id] + gargs),
    )
    players = [
        {
            "user_id": int(r["user_id"]), "name": _display_name(r),
            "is_me": int(r["user_id"]) == int(viewer_id),
            "hands": int(r["hands"] or 0), "wins": int(r["wins"] or 0),
            "sessions": int(r["sessions"] or 0), "net_cents": int(r["net"] or 0),
            "best_cents": max(0, int(r["best"] or 0)),
            "accuracy": round(float(r["a"] or 0) / int(r["n"]), 1) if int(r["n"] or 0) else None,
            "graded": int(r["n"] or 0),
        }
        for r in rows
    ]
    players.sort(key=lambda x: -x["net_cents"])
    fguard = guard.replace("r.game_id", "f.game_id").replace("r.hand_no", "f.hand_no")
    gross: dict[tuple[int, int], float] = {}
    for r in pub.DB.q(
        "SELECT f.payer, f.payee, SUM(f.chips * g.bb_cents) v FROM homegame_flows f "
        f"JOIN homegames g ON g.id=f.game_id WHERE g.excluded=0 AND g.club_id=?{fguard} "
        "GROUP BY f.payer, f.payee", tuple([club_id] + gargs),
    ):
        gross[(int(r["payer"]), int(r["payee"]))] = float(r["v"] or 0) / BB_CHIPS
    pairs = []
    for (a, b), v in gross.items():
        net = v - gross.get((b, a), 0.0)
        if net > 0.5:  # b is up `cents` on a
            pairs.append({"from": a, "to": b, "cents": int(round(net))})
    sess = pub.DB.q(
        "SELECT g.id, g.name, g.status, g.excluded, g.created_at, g.closed_at, g.hand_no, "
        "g.sb_cents, g.bb_cents, g.ante_cents, "
        "(SELECT COUNT(*) FROM homegame_players p WHERE p.game_id=g.id AND p.buyin_cents>0) players "
        "FROM homegames g WHERE g.club_id=? " + ("" if can_manage else "AND g.excluded=0 ") +
        "ORDER BY g.created_at DESC LIMIT 300", (club_id,),
    )
    return {
        "club": club_id, "players": players, "pairs": pairs,
        "can_manage": bool(can_manage), "is_admin": bool(can_manage),
        "sessions": [
            {"id": r["id"], "name": r["name"], "open": r["status"] == "open",
             "excluded": bool(int(r["excluded"] or 0)), "created_at": r["created_at"],
             "closed_at": r["closed_at"], "hands": int(r["hand_no"] or 0),
             "players": int(r["players"] or 0), "sb_cents": int(r["sb_cents"]),
             "bb_cents": int(r["bb_cents"]), "ante_cents": int(r["ante_cents"])}
            for r in sess
        ],
    }


def _exclude_locked(t: LiveTable, user: Any, on: bool) -> None:
    """The CLUB'S OWNER: take a session out of the club's record (test tables) or
    put it back — not the host (a host must not be able to erase a losing night).
    Nothing is deleted — hands, ledger and flows stay in the database and the
    table still opens by its link; it just counts nowhere. An open table is
    closed first (everyone is cashed out), so it also leaves the lobby."""
    if user is None or _club_role(t.club_id, int(user["id"])) != "owner":
        raise HTTPException(status_code=403, detail="only the club's owner can do that")
    if on and t.status == "open":
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
        t.env = None
        t.info = None
        t.runout_active = False
    pub.DB.q("UPDATE homegames SET excluded=? WHERE id=?", (1 if on else 0, t.game_id))
    t.rev += 1


# --- a player's lifetime database ---------------------------------------------------

_MY_SORTS = {
    "time": "h.ended_at",
    "pot": "h.pot_cents",
    "net": "r.delta_cents",
    "accuracy": "(CASE WHEN r.acc_n>0 THEN r.acc_sum/r.acc_n ELSE NULL END)",
}


def _my_hands(viewer_id: int, sort: str, direction: str, game_id: str | None,
              offset: int, limit: int, player_id: int | None = None,
              clubs: list[str] | None = None) -> dict[str, Any]:
    """``player_id``'s hands (default: the viewer's own). The club is private and
    everyone may browse everyone's history — but the CARDS in it follow the live
    table's rule for the VIEWER: their own, plus hands that were tabled or shown.
    Browsing Riley's history never turns over a hand Riley mucked."""
    player = int(player_id) if player_id is not None else int(viewer_id)
    col = _MY_SORTS.get(sort, _MY_SORTS["time"])
    desc = str(direction).lower() != "asc"
    limit = max(1, min(HANDS_PAGE_MAX, int(limit)))
    offset = max(0, int(offset))
    scope, sargs = _club_clause(clubs)
    where = "r.user_id=? AND g.excluded=0" + scope
    args: list[Any] = [player] + sargs
    if game_id:
        where += " AND r.game_id=?"
        args.append(str(game_id))
    # A hand is never served while its table is still playing / revealing it.
    live = {}
    with HUB._lock:
        tables = list(HUB._tables.values())
    for tb in tables:
        with tb.lock:
            live[tb.game_id] = _hands_visible_upto(tb)
    total = pub.DB.one(
        "SELECT COUNT(*) c FROM homegame_hand_results r "
        f"JOIN homegames g ON g.id=r.game_id WHERE {where}", tuple(args)
    )["c"]
    rows = pub.DB.q(
        "SELECT r.game_id, r.hand_no, r.delta_cents, r.acc_sum, r.acc_n, h.ended_at, "
        "h.pot_cents, h.summary, g.name AS table_name FROM homegame_hand_results r "
        "JOIN homegame_hands h ON h.game_id=r.game_id AND h.hand_no=r.hand_no "
        "JOIN homegames g ON g.id=r.game_id "
        f"WHERE {where} ORDER BY ({col} IS NULL), {col} {'DESC' if desc else 'ASC'}, "
        "h.ended_at DESC LIMIT ? OFFSET ?",
        tuple(args + [limit, offset]),
    )
    hands = []
    for r in rows:
        if r["game_id"] in live and int(r["hand_no"]) > live[r["game_id"]]:
            continue
        try:
            full = json.loads(r["summary"])
            seat_no = next(
                (int(x["seat"]) for x in full.get("seats") or []
                 if int(x.get("user_id") or -1) == player), None,
            )
            rec = _hand_for_viewer(full, viewer_id)
        except Exception:  # noqa: BLE001
            continue
        me = next((x for x in rec["seats"] if int(x["seat"]) == seat_no), None)
        hands.append({
            "game_id": r["game_id"], "table_name": r["table_name"],
            "hand_no": int(r["hand_no"]), "ended_at": r["ended_at"],
            "pot_cents": int(r["pot_cents"]), "net_cents": int(r["delta_cents"]),
            "accuracy": round(float(r["acc_sum"]) / int(r["acc_n"]), 1) if int(r["acc_n"] or 0) else None,
            "showdown": bool(rec.get("showdown")),
            "my_hole": me.get("hole") if me else None,
            "board_a": rec.get("board_a") or [], "board_b": rec.get("board_b") or [],
        })
    return {"hands": hands, "total": int(total), "offset": offset, "limit": limit}


def _my_stats(viewer_id: int, clubs: list[str] | None = None) -> dict[str, Any]:
    """A player's numbers, within ``clubs`` (None = everything they played)."""
    uid = int(viewer_id)
    who = pub._user_by_id(uid)
    scope, sargs = _club_clause(clubs)
    tot = pub.DB.one(
        "SELECT COUNT(*) hands, COALESCE(SUM(r.delta_cents),0) net, COALESCE(SUM(r.acc_sum),0) a, "
        "COALESCE(SUM(r.acc_n),0) n, SUM(CASE WHEN r.delta_cents>0 THEN 1 ELSE 0 END) wins "
        "FROM homegame_hand_results r JOIN homegames g ON g.id=r.game_id "
        f"WHERE r.user_id=? AND g.excluded=0{scope}", tuple([uid] + sargs),
    )
    sessions = pub.DB.q(
        "SELECT g.id, g.name, g.status, g.sb_cents, g.bb_cents, g.ante_cents, COUNT(*) hands, "
        "SUM(r.delta_cents) net, SUM(r.acc_sum) a, SUM(r.acc_n) n, MAX(h.ended_at) last "
        "FROM homegame_hand_results r JOIN homegames g ON g.id=r.game_id "
        "JOIN homegame_hands h ON h.game_id=r.game_id AND h.hand_no=r.hand_no "
        f"WHERE r.user_id=? AND g.excluded=0{scope} GROUP BY g.id ORDER BY last DESC LIMIT 200",
        tuple([uid] + sargs),
    )
    # head to head, in CENTS (chips are relative to each table's big blind)
    vs: dict[int, float] = {}
    for r in pub.DB.q(
        "SELECT f.payer, f.payee, SUM(f.chips * g.bb_cents) v FROM homegame_flows f "
        f"JOIN homegames g ON g.id=f.game_id WHERE (f.payer=? OR f.payee=?) AND g.excluded=0{scope} "
        "GROUP BY f.payer, f.payee", tuple([uid, uid] + sargs),
    ):
        other = int(r["payee"]) if int(r["payer"]) == uid else int(r["payer"])
        sign = -1 if int(r["payer"]) == uid else 1
        vs[other] = vs.get(other, 0.0) + sign * float(r["v"] or 0) / BB_CHIPS
    versus = []
    for other, cents in vs.items():
        u = pub._user_by_id(other)
        versus.append({"user_id": other,
                       "name": _display_name(u) if u is not None else f"player-{other}",
                       "net_cents": int(round(cents))})
    versus.sort(key=lambda x: -x["net_cents"])
    n = int(tot["n"] or 0)
    return {
        "user_id": uid, "name": _display_name(who) if who is not None else f"player-{uid}",
        "hands": int(tot["hands"] or 0), "wins": int(tot["wins"] or 0),
        "net_cents": int(tot["net"] or 0),
        "accuracy": round(float(tot["a"]) / n, 1) if n else None, "graded": n,
        "sessions": [
            {"id": r["id"], "name": r["name"], "open": r["status"] == "open",
             "sb_cents": int(r["sb_cents"]), "bb_cents": int(r["bb_cents"]),
             "ante_cents": int(r["ante_cents"]), "hands": int(r["hands"]),
             "net_cents": int(r["net"] or 0),
             "accuracy": round(float(r["a"]) / int(r["n"]), 1) if int(r["n"] or 0) else None,
             "last_played": r["last"]}
            for r in sessions
        ],
        "versus": versus,
    }


def _hand_detail(t: LiveTable, viewer_id: int, hand_no: int) -> dict[str, Any]:
    if int(hand_no) > _hands_visible_upto(t):
        raise HTTPException(status_code=404, detail="Not Found")
    row = pub.DB.one(
        "SELECT summary FROM homegame_hands WHERE game_id=? AND hand_no=?",
        (t.game_id, int(hand_no)),
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Not Found")
    out = _hand_for_viewer(json.loads(row["summary"]), viewer_id, t.show_grades)
    out["game_id"] = t.game_id
    out["table_name"] = t.name
    return out


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
    return int(round(f))


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


def _create_table(user: Any, body: dict) -> LiveTable:
    name = " ".join(str(body.get("name") or "").split()) or "Table 1"
    if len(name) > MAX_NAME_LEN:
        raise HTTPException(
            status_code=400, detail=f"table name is limited to {MAX_NAME_LEN} characters"
        )
    n = (
        _valid_num_seats(pub.body_int(body, "num_seats"))
        if body.get("num_seats") is not None
        else TABLE_SEATS
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
    _validate_buyin_window(bb, lo, hi, buyin)
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
    )
    # The table row and the host's seat land together or not at all.
    with pub.DB.transaction():
        pub.DB.q(
            "INSERT INTO homegames(id,host_user_id,name,num_seats,sb_cents,bb_cents,"
            "ante_cents,default_buyin_cents,status,running,decision_secs,button,"
            "hand_no,created_at,deal_delay_ms,time_bank_secs,min_buyin_cents,"
            "max_buyin_cents,listed,approve_buyins,allow_rathole,club_id) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (gid, int(user["id"]), name, n, sb, bb, ante, buyin, "open", 0, secs,
             0, 0, pub._now(), int(round(delay * 1000)), bank, lo, hi,
             1 if listed else 0, 1 if approve else 0, 1 if rathole else 0, club_id),
        )
        # Host sits seat 0 with the default buy-in so they can deal once a
        # second player sits.
        _sit_locked(t, user, 0, buyin)
    HUB.put(t)
    return t


def _lobby_row(row: Any, viewer_id: int | None = None) -> dict[str, Any]:
    players = pub.DB.q(
        "SELECT p.user_id, p.seat, u.name, u.email FROM homegame_players p "
        "JOIN users u ON u.id=p.user_id WHERE p.game_id=? AND p.seat IS NOT NULL "
        "ORDER BY p.seat",
        (row["id"],),
    )
    host = pub._user_by_id(int(row["host_user_id"]))
    member = False
    if viewer_id is not None:
        member = pub.DB.one(
            "SELECT 1 FROM homegame_players WHERE game_id=? AND user_id=?",
            (row["id"], int(viewer_id)),
        ) is not None
    return {
        "id": row["id"],
        "name": row["name"],
        "num_seats": row["num_seats"],
        "seated": len(players),
        "sb_cents": row["sb_cents"],
        "bb_cents": row["bb_cents"],
        "ante_cents": row["ante_cents"],
        "default_buyin_cents": row["default_buyin_cents"],
        "hand_no": row["hand_no"],
        "running": bool(int(row["running"] if "running" in row.keys() else 0)),
        "host_name": _display_name(host) if host else "?",
        "created_at": row["created_at"],
        # No emails, no user ids of other people (review 2026-09-20 G12).
        "players": [
            {"seat": int(p["seat"]), "name": _display_name(p),
             "is_me": viewer_id is not None and int(p["user_id"]) == int(viewer_id)}
            for p in players
        ],
        "is_host": viewer_id is not None and int(row["host_user_id"]) == int(viewer_id),
        "is_seated": any(
            viewer_id is not None and int(p["user_id"]) == int(viewer_id) for p in players
        ),
        "is_member": bool(member),
        "listed": bool(_row_int(row, "listed", 1)),
    }


def _lobby_visible(row: Any, viewer_id: int | None) -> bool:
    """Unlisted tables are link-only: they show up in the lobby of the host
    and of anyone who has played at them, and nobody else's."""
    if bool(_row_int(row, "listed", 1)):
        return True
    if viewer_id is None:
        return False
    if int(row["host_user_id"]) == int(viewer_id):
        return True
    return pub.DB.one(
        "SELECT 1 FROM homegame_players WHERE game_id=? AND user_id=?",
        (row["id"], int(viewer_id)),
    ) is not None


def _my_sessions(viewer_id: int, club_id: str | None = None, limit: int = 12) -> list[dict[str, Any]]:
    """The viewer's finished sessions (closed tables they played at), in one club."""
    scope, sargs = _club_clause([club_id] if club_id else None)
    rows = pub.DB.q(
        "SELECT g.id, g.name, g.sb_cents, g.bb_cents, g.ante_cents, g.hand_no, "
        "g.closed_at, p.buyin_cents, p.leftover_cents FROM homegame_players p "
        "JOIN homegames g ON g.id=p.game_id "
        f"WHERE p.user_id=? AND g.status='closed' AND p.buyin_cents>0 AND g.excluded=0{scope} "
        "ORDER BY g.closed_at DESC LIMIT ?",
        tuple([int(viewer_id)] + sargs + [int(limit)]),
    )
    out = []
    for r in rows:
        hands = pub.DB.one(
            "SELECT COUNT(*) c FROM homegame_hand_results WHERE game_id=? AND user_id=?",
            (r["id"], int(viewer_id)),
        )["c"]
        out.append({
            "id": r["id"],
            "name": r["name"],
            "sb_cents": int(r["sb_cents"]),
            "bb_cents": int(r["bb_cents"]),
            "ante_cents": int(r["ante_cents"]),
            "closed_at": r["closed_at"],
            "hands": int(hands or 0),
            "buyin_cents": int(r["buyin_cents"]),
            "net_cents": int(r["leftover_cents"]) - int(r["buyin_cents"]),
        })
    return out


def _chat_messages(game_id: str) -> list[dict[str, Any]]:
    rows = pub.DB.q(
        "SELECT c.id, c.body, c.created_at, u.name, u.email "
        "FROM homegame_chat c JOIN users u ON u.id=c.user_id "
        "WHERE c.game_id=? ORDER BY c.id DESC LIMIT 80",
        (game_id,),
    )
    out = []
    for r in reversed(list(rows)):
        out.append({
            "id": int(r["id"]),
            "name": _display_name(r),
            "text": r["body"],
            "created_at": r["created_at"],
        })
    return out


def _chat_add(t: LiveTable, user: Any, text: Any) -> None:
    uid = int(user["id"])
    _require_member(t, uid)  # review 2026-09-20 G12
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="invalid message")
    body = " ".join(text[: CHAT_MAX_LEN * 8].split())
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
    times.append(now)
    t.rev += 1


def _page_html(static_dir: Path) -> str:
    return (static_dir / "games.html").read_text(encoding="utf-8")


# Security headers for the home-games page (the lobby and every table). Scripts
# run ONLY from this site: no inline <script>, no inline event-handler attribute
# (onclick=...), no javascript: URL — a script injected through a name or a chat
# line cannot run. Styles/fonts come from this site and Google Fonts; images from
# this site or data: URIs (so injected CSS cannot send anything elsewhere either);
# no other site may frame the page. A new outside resource (a CDN, an image host)
# needs its origin added here; test_homegame_page_headers.py scans the client for
# inline handlers the policy would silently block.
PAGE_CSP = "; ".join((
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com",
    "img-src 'self' data:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'self'",
    "frame-ancestors 'none'",
))
PAGE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-store, must-revalidate",
    "Content-Security-Policy": PAGE_CSP,
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
    # allow-popups: "Open in Study" fills the new tab it opens before sending it on
    "Cross-Origin-Opener-Policy": "same-origin-allow-popups",
}


# --- clubs (2026-09-25) -----------------------------------------------------------
#
# Home games are open to every signed-in user; a CLUB is the private circle: its
# tables, its members and its numbers. A table belongs to exactly one club. Only
# members see it in the lobby, sit, watch or open its history, and every ranking,
# accuracy score, profit table and head-to-head is computed from ONE club's tables:
# nothing ranks the whole user base. Anyone can start a club and host in it.
# People join with the club's invite link (the owner can make it "ask first"), or
# ask from a table link and wait for the owner or an admin to let them in.
# The site's original private circle is the MAIN club: the migration gives it every
# table and player that predate clubs, and the /admin home-games switch adds and
# removes people there.

CLUB_ROLES = ("owner", "admin", "member")
MAX_CLUBS_OWNED = 5
MAX_CLUB_NAME = 40
JOIN_RETRY_S = 60.0     # a declined request can be sent again after this
JOIN_REFRESH_S = 5.0    # how stale a club manager's view of the requests may be
_CODE_RE = re.compile(r"^[A-Za-z0-9_-]{6,40}$")


def _club(club_id: Any) -> Any:
    if not club_id or not _CODE_RE.match(str(club_id)):
        return None
    return pub.DB.one("SELECT * FROM homegame_clubs WHERE id=?", (str(club_id),))


def _club_role(club_id: Any, uid: Any) -> str | None:
    if not club_id or uid is None:
        return None
    r = pub.DB.one(
        "SELECT role FROM homegame_club_members WHERE club_id=? AND user_id=?", (str(club_id), int(uid))
    )
    return str(r["role"]) if r is not None else None


def _require_club(club_id: Any, uid: int, *roles: str) -> tuple[Any, str]:
    """The club and the viewer's role in it. Not a member = 404 (a club's
    existence is nobody else's business); a member without one of ``roles`` = 403."""
    club = _club(club_id)
    role = _club_role(club["id"], uid) if club is not None else None
    if club is None or role is None:
        raise HTTPException(status_code=404, detail="Not Found")
    if roles and role not in roles:
        who = "the club's owner" if roles == ("owner",) else "the club's owner or an admin"
        raise HTTPException(status_code=403, detail=f"only {who} can do that")
    return club, role


def _club_member_ids(club_id: str) -> set[int]:
    return {int(r["user_id"]) for r in pub.DB.q(
        "SELECT user_id FROM homegame_club_members WHERE club_id=?", (club_id,))}


def _add_member(club_id: str, uid: int, role: str = "member") -> None:
    now = pub._now()
    pub.DB.q(
        "INSERT OR IGNORE INTO homegame_club_members(club_id,user_id,role,joined_at) VALUES(?,?,?,?)",
        (club_id, int(uid), role, now),
    )
    pub.DB.q(
        "UPDATE homegame_club_requests SET status='approved', decided_at=? "
        "WHERE club_id=? AND user_id=? AND status='pending'", (now, club_id, int(uid)),
    )


def _create_club(user: Any, name: Any, *, main: bool = False) -> str:
    name = " ".join(str(name or "").split())
    if not name:
        raise HTTPException(status_code=400, detail="give the club a name")
    if len(name) > MAX_CLUB_NAME:
        raise HTTPException(status_code=400, detail=f"a club name is limited to {MAX_CLUB_NAME} characters")
    uid = int(user["id"])
    owned = pub.DB.one("SELECT COUNT(*) c FROM homegame_clubs WHERE owner_user_id=?", (uid,))["c"]
    if not main and int(owned) >= MAX_CLUBS_OWNED:
        raise HTTPException(status_code=429, detail=f"you already run {MAX_CLUBS_OWNED} clubs")
    cid = secrets.token_urlsafe(6)
    now = pub._now()
    with pub.DB.transaction():
        pub.DB.q(
            "INSERT INTO homegame_clubs(id,name,owner_user_id,invite_code,approve_joins,is_main,created_at) "
            "VALUES(?,?,?,?,0,?,?)",
            (cid, name, uid, secrets.token_urlsafe(9), 1 if main else 0, now),
        )
        pub.DB.q(
            "INSERT OR IGNORE INTO homegame_club_members(club_id,user_id,role,joined_at) VALUES(?,?,'owner',?)",
            (cid, uid, now),
        )
    return cid


def _main_club(create_for: int | None = None) -> str | None:
    """The site's original circle (see above). Made on first need: by the
    migration, or by the first /admin home-games grant (``create_for`` = the
    granting admin, who then owns it)."""
    r = pub.DB.one("SELECT id FROM homegame_clubs WHERE is_main=1 ORDER BY created_at LIMIT 1")
    if r is not None:
        return str(r["id"])
    owner = pub._user_by_id(int(create_for)) if create_for is not None else None
    if owner is None:
        return None
    first = (_display_name(owner) or "Home").split(" ")[0]
    return _create_club(owner, f"{first}'s club"[:MAX_CLUB_NAME], main=True)


def _migrate_clubs() -> None:
    """Tables that predate clubs (``club_id`` NULL) join the MAIN club, and so
    does everyone who ever hosted or sat at one, plus everyone who had the old
    admin-granted home-games flag: the circle and its numbers carry on exactly as
    they were. Idempotent (nothing left to move = nothing happens)."""
    legacy = pub.DB.q("SELECT id, host_user_id FROM homegames WHERE club_id IS NULL ORDER BY created_at")
    if not legacy:
        return
    granted = [int(r["id"]) for r in pub.DB.q("SELECT id FROM users WHERE homegame_access=1 ORDER BY created_at")]
    admins = [int(r["id"]) for r in pub.DB.q("SELECT id, email FROM users ORDER BY created_at") if pub._is_admin(r)]
    owner = admins[0] if admins else int(legacy[0]["host_user_id"])
    cid = _main_club(create_for=owner)
    if cid is None:
        return
    members = set(granted) | {int(r["host_user_id"]) for r in legacy}
    for r in pub.DB.q("SELECT DISTINCT p.user_id FROM homegame_players p JOIN homegames g ON g.id=p.game_id "
                      "WHERE g.club_id IS NULL"):
        members.add(int(r["user_id"]))
    with pub.DB.transaction():
        for uid in sorted(members):
            if pub._user_by_id(uid) is not None:
                pub.DB.q(
                    "INSERT OR IGNORE INTO homegame_club_members(club_id,user_id,role,joined_at) "
                    "VALUES(?,?,'member',?)", (cid, uid, pub._now()),
                )
        pub.DB.q("UPDATE homegames SET club_id=? WHERE club_id IS NULL", (cid,))
    logger.info("clubs: %d table(s) and %d player(s) moved into the main club %s", len(legacy), len(members), cid)


def _club_for_new_table(uid: int, requested: Any) -> str:
    """Any member may host in a club. Without a choice (scripts, older clients):
    the main club when the host is in it, else their oldest club."""
    if requested:
        club, _ = _require_club(requested, uid)
        return str(club["id"])
    r = pub.DB.one(
        "SELECT m.club_id FROM homegame_club_members m JOIN homegame_clubs c ON c.id=m.club_id "
        "WHERE m.user_id=? ORDER BY c.is_main DESC, m.joined_at LIMIT 1", (int(uid),),
    )
    if r is None:
        raise HTTPException(status_code=400, detail="start a club (or join one) before you host a table")
    return str(r["club_id"])


def _mask_email(email: Any) -> str:
    """Enough to tell two Alexes apart, not a mailing list: ``a•••@gmail.com``."""
    s = str(email or "")
    if "@" not in s:
        return ""
    user, dom = s.split("@", 1)
    return f"{user[:1]}•••@{dom}"


def _join_retry_in(decided_at: str | None) -> float:
    try:
        dt = datetime.fromisoformat(str(decided_at))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, JOIN_RETRY_S - (datetime.now(timezone.utc) - dt).total_seconds())


def _request_state(club_id: str, uid: int) -> tuple[str | None, float]:
    r = pub.DB.one(
        "SELECT status, decided_at FROM homegame_club_requests WHERE club_id=? AND user_id=?", (club_id, int(uid))
    )
    if r is None:
        return None, 0.0
    status = str(r["status"])
    return status, (_join_retry_in(r["decided_at"]) if status == "declined" else 0.0)


def _request_club(club_id: str, uid: int) -> None:
    """Ask to join (from a table link, or an invite link of an ask-first club).
    Once pending, or declined less than a minute ago, asking again is a no-op."""
    if _club_role(club_id, uid) is not None:
        return
    status, retry = _request_state(club_id, uid)
    if status == "pending" or (status == "declined" and retry > 0):
        return
    pub.DB.q(
        "INSERT OR REPLACE INTO homegame_club_requests(club_id,user_id,status,created_at,decided_at) "
        "VALUES(?,?,'pending',?,NULL)", (club_id, int(uid), pub._now()),
    )
    user = pub._user_by_id(uid)
    for t in _open_club_tables_in_memory(club_id):
        with t.lock:
            t.join_checked_mono = 0.0
            _emit(t, "joinreq", f"{_display_name(user) if user else 'Someone'} asks to join the club")
            t.rev += 1


def _open_club_tables_in_memory(club_id: str) -> list[LiveTable]:
    with HUB._lock:
        return [t for t in HUB._tables.values() if t.club_id == club_id and t.status == "open"]


def _club_requests(club_id: str) -> list[dict[str, Any]]:
    """Pending requests, oldest first (people who got in meanwhile drop out)."""
    club = _club(club_id)
    rows = pub.DB.q(
        "SELECT r.user_id, r.created_at, u.name, u.email FROM homegame_club_requests r "
        "JOIN users u ON u.id=r.user_id WHERE r.club_id=? AND r.status='pending' ORDER BY r.created_at",
        (club_id,),
    )
    members = _club_member_ids(club_id)
    return [
        {"club_id": club_id, "club_name": str(club["name"]) if club else "", "user_id": int(r["user_id"]),
         "name": _display_name(r), "email": _mask_email(r["email"]), "since": str(r["created_at"])}
        for r in rows if int(r["user_id"]) not in members
    ]


def _decide_club_request(club_id: str, by_uid: int, target: int, allow: bool) -> dict[str, Any]:
    _require_club(club_id, by_uid, "owner", "admin")
    status, _ = _request_state(club_id, target)
    if status != "pending" or _club_role(club_id, target) is not None:
        raise HTTPException(status_code=409, detail="that request is gone")
    user = pub._user_by_id(target)
    if allow:
        _add_member(club_id, target)
    else:
        pub.DB.q("UPDATE homegame_club_requests SET status='declined', decided_at=? WHERE club_id=? AND user_id=?",
                 (pub._now(), club_id, int(target)))
    for t in _open_club_tables_in_memory(club_id):
        with t.lock:
            t.join_checked_mono = 0.0
            if allow:
                _emit(t, "join", f"{_display_name(user) if user else 'A new player'} joined the club")
            t.rev += 1
    return {"ok": True, "allowed": bool(allow)}


def _table_join_requests(t: LiveTable, viewer_id: int | None) -> list[dict[str, Any]]:
    """What the club's owner / an admin at this table is asked to decide."""
    if viewer_id is None or _club_role(t.club_id, viewer_id) not in ("owner", "admin"):
        return []
    now = time.monotonic()
    if now - t.join_checked_mono > JOIN_REFRESH_S:
        t.join_checked_mono = now
        try:
            t.join_reqs = _club_requests(t.club_id) if t.club_id else []
        except Exception:  # noqa: BLE001 — cosmetic: never take the view down
            logger.exception("club requests (table %s)", t.game_id)
    return list(t.join_reqs)


def _table_club(t: LiveTable, viewer_id: int | None) -> dict[str, Any] | None:
    """The table's club as its view shows it (cached per table: names rarely change)."""
    now = time.monotonic()
    if t.club_info is None or now - t.club_checked_mono > 30.0:
        t.club_checked_mono = now
        club = _club(t.club_id)
        t.club_info = {"id": str(club["id"]), "name": str(club["name"])} if club is not None else None
    if t.club_info is None:
        return None
    return dict(t.club_info, role=_club_role(t.club_id, viewer_id))


def _table_access(t: LiveTable, uid: int) -> None:
    """Only the table's club may see it, sit, watch or act. Anyone else gets 403
    naming the club (and where their request stands), so a table link can offer
    "ask to join the club" instead of a dead end."""
    if t.club_id and _club_role(t.club_id, uid) is not None:
        return
    club = _club(t.club_id)
    if club is None:  # (no club at all cannot happen after the migration: its host only)
        if int(uid) == int(t.host_user_id):
            return
        raise HTTPException(status_code=404, detail="Not Found")
    status, retry = _request_state(str(club["id"]), uid)
    raise HTTPException(status_code=403, detail={
        "error": "club", "club": {"id": str(club["id"]), "name": str(club["name"])},
        "request": status, "retry_in": round(retry, 1),
        "message": f"This table belongs to the club “{club['name']}”.",
    })


def _busy_in_club(club_id: str, uid: int, who: str) -> str | None:
    """Why this person can't leave the club right now (``who`` = "you" or their
    name), or None: a seat at one of its open tables, or hosting one (a table
    whose host is outside its club would have nobody left to run it)."""
    r = pub.DB.one(
        "SELECT g.name FROM homegame_players p JOIN homegames g ON g.id=p.game_id "
        "WHERE g.club_id=? AND g.status='open' AND p.user_id=? AND p.seat IS NOT NULL LIMIT 1",
        (club_id, int(uid)),
    )
    if r is not None:
        return f"{who} {'have' if who == 'you' else 'has'} a seat at “{r['name']}” — leave the table first"
    r = pub.DB.one(
        "SELECT name FROM homegames WHERE club_id=? AND status='open' AND host_user_id=? LIMIT 1",
        (club_id, int(uid)),
    )
    if r is not None:
        return f"{who} {'host' if who == 'you' else 'hosts'} “{r['name']}” — hand the table over or close it first"
    return None


def _club_summary(r: Any, uid: int) -> dict[str, Any]:
    cid, role = str(r["id"]), str(r["role"])
    manage = role in ("owner", "admin")
    return {
        "id": cid, "name": str(r["name"]), "role": role, "is_main": bool(int(r["is_main"] or 0)),
        "members": int(pub.DB.one("SELECT COUNT(*) c FROM homegame_club_members WHERE club_id=?", (cid,))["c"]),
        "open_tables": int(pub.DB.one(
            "SELECT COUNT(*) c FROM homegames WHERE club_id=? AND status='open'", (cid,))["c"]),
        "requests": len(_club_requests(cid)) if manage else 0,
    }


def _my_clubs(uid: int) -> list[dict[str, Any]]:
    rows = pub.DB.q(
        "SELECT c.*, m.role FROM homegame_club_members m JOIN homegame_clubs c ON c.id=m.club_id "
        "WHERE m.user_id=? ORDER BY c.is_main DESC, c.created_at", (int(uid),),
    )
    return [_club_summary(r, uid) for r in rows]


_ROLE_RANK = {"owner": 0, "admin": 1, "member": 2}


def _club_view(club_id: Any, uid: int) -> dict[str, Any]:
    club, role = _require_club(club_id, uid)
    cid = str(club["id"])
    manage = role in ("owner", "admin")
    owner = pub._user_by_id(int(club["owner_user_id"]))
    rows = pub.DB.q(
        "SELECT m.user_id, m.role, m.joined_at, u.name, u.email FROM homegame_club_members m "
        "JOIN users u ON u.id=m.user_id WHERE m.club_id=?", (cid,),
    )
    members = sorted(
        ({"user_id": int(r["user_id"]), "name": _display_name(r), "role": str(r["role"]),
          "is_me": int(r["user_id"]) == int(uid), "joined_at": str(r["joined_at"])} for r in rows),
        key=lambda m: (_ROLE_RANK.get(m["role"], 9), m["name"].lower()),
    )
    return {
        "id": cid, "name": str(club["name"]), "role": role, "is_main": bool(int(club["is_main"] or 0)),
        "approve_joins": bool(int(club["approve_joins"] or 0)),
        "owner": {"user_id": int(club["owner_user_id"]), "name": _display_name(owner) if owner else "?"},
        "members": members,
        "invite_code": str(club["invite_code"]) if manage else None,
        "requests": _club_requests(cid) if manage else [],
        "created_at": str(club["created_at"]),
    }


def _club_settings(club_id: Any, uid: int, body: dict) -> dict[str, Any]:
    club, _ = _require_club(club_id, uid, "owner")
    cid = str(club["id"])
    name = str(club["name"])
    if body.get("name") is not None:
        name = " ".join(str(body.get("name") or "").split())
        if not name:
            raise HTTPException(status_code=400, detail="give the club a name")
        if len(name) > MAX_CLUB_NAME:
            raise HTTPException(status_code=400, detail=f"a club name is limited to {MAX_CLUB_NAME} characters")
    approve = _parse_bool(body, "approve_joins", bool(int(club["approve_joins"] or 0)))
    pub.DB.q("UPDATE homegame_clubs SET name=?, approve_joins=? WHERE id=?", (name, 1 if approve else 0, cid))
    return _club_view(cid, uid)


def _reset_invite(club_id: Any, uid: int) -> dict[str, Any]:
    club, _ = _require_club(club_id, uid, "owner", "admin")
    pub.DB.q("UPDATE homegame_clubs SET invite_code=? WHERE id=?", (secrets.token_urlsafe(9), str(club["id"])))
    return _club_view(str(club["id"]), uid)


def _set_member(club_id: Any, by_uid: int, target: int, *, role: Any = None, remove: bool = False) -> dict[str, Any]:
    club, my_role = _require_club(club_id, by_uid, "owner", "admin")
    cid = str(club["id"])
    their = _club_role(cid, target)
    if their is None:
        raise HTTPException(status_code=404, detail="they are not in the club")
    who = pub._user_by_id(target)
    name = _display_name(who) if who is not None else "They"
    if remove:
        if int(target) == int(by_uid):
            raise HTTPException(status_code=400, detail="to go, use Leave club")
        if their == "owner":
            raise HTTPException(status_code=403, detail="the owner can't be removed")
        if my_role == "admin" and their != "member":
            raise HTTPException(status_code=403, detail="only the owner can remove an admin")
        busy = _busy_in_club(cid, target, name)
        if busy:
            raise HTTPException(status_code=409, detail=busy)
        pub.DB.q("DELETE FROM homegame_club_members WHERE club_id=? AND user_id=?", (cid, int(target)))
        return _club_view(cid, by_uid)
    if my_role != "owner":
        raise HTTPException(status_code=403, detail="only the club's owner can change roles")
    role = str(role or "")
    if role not in CLUB_ROLES:
        raise HTTPException(status_code=400, detail="role must be owner, admin or member")
    if role == "owner":
        # handing the club over: the old owner stays on as an admin
        if int(target) == int(by_uid):
            return _club_view(cid, by_uid)
        with pub.DB.transaction():
            pub.DB.q("UPDATE homegame_clubs SET owner_user_id=? WHERE id=?", (int(target), cid))
            pub.DB.q("UPDATE homegame_club_members SET role='owner' WHERE club_id=? AND user_id=?", (cid, int(target)))
            pub.DB.q("UPDATE homegame_club_members SET role='admin' WHERE club_id=? AND user_id=?", (cid, int(by_uid)))
        return _club_view(cid, by_uid)
    if their == "owner":
        raise HTTPException(status_code=400, detail="the owner's role only changes by handing the club over")
    pub.DB.q("UPDATE homegame_club_members SET role=? WHERE club_id=? AND user_id=?", (role, cid, int(target)))
    return _club_view(cid, by_uid)


def _leave_club(club_id: Any, uid: int) -> None:
    club, role = _require_club(club_id, uid)
    cid = str(club["id"])
    if role == "owner":
        raise HTTPException(status_code=400, detail="hand the club to another member first (Club settings → Members)")
    busy = _busy_in_club(cid, uid, "you")
    if busy:
        raise HTTPException(status_code=409, detail=busy[0].upper() + busy[1:])
    pub.DB.q("DELETE FROM homegame_club_members WHERE club_id=? AND user_id=?", (cid, int(uid)))


def _club_by_code(code: Any) -> Any:
    if not code or not _CODE_RE.match(str(code)):
        return None
    return pub.DB.one("SELECT * FROM homegame_clubs WHERE invite_code=?", (str(code),))


def _invite_info(code: Any, uid: int | None) -> dict[str, Any]:
    club = _club_by_code(code)
    if club is None:
        raise HTTPException(status_code=404, detail="This invite link is no longer valid — ask for a new one.")
    cid = str(club["id"])
    owner = pub._user_by_id(int(club["owner_user_id"]))
    status, retry = _request_state(cid, uid) if uid is not None else (None, 0.0)
    return {
        "club": {"id": cid, "name": str(club["name"]), "owner_name": _display_name(owner) if owner else "?",
                 "members": len(_club_member_ids(cid))},
        "member": uid is not None and _club_role(cid, uid) is not None,
        "approve": bool(int(club["approve_joins"] or 0)),
        "request": status, "retry_in": round(retry, 1),
    }


def _join_by_invite(code: Any, uid: int) -> dict[str, Any]:
    club = _club_by_code(code)
    if club is None:
        raise HTTPException(status_code=404, detail="This invite link is no longer valid — ask for a new one.")
    cid = str(club["id"])
    if _club_role(cid, uid) is None:
        if bool(int(club["approve_joins"] or 0)):
            _request_club(cid, uid)
        else:
            _add_member(cid, uid)
    return _invite_info(code, uid)


# --- pages for someone who is not signed in ----------------------------------------
#
# Home games need an account: the lobby, a table link or a club invite link opened
# signed out answers with a small page (no scripts) naming what it is and a
# "Sign in with Google" button that comes straight back to the same link.

_SIGNIN_PATH = re.compile(r"^/games(?:/t/([A-Za-z0-9_-]{1,40})|/join/([A-Za-z0-9_-]{6,40}))?$")
INVITE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-store, must-revalidate",
    "Content-Security-Policy": "; ".join((
        "default-src 'none'",
        "style-src 'unsafe-inline' https://fonts.googleapis.com",
        "font-src https://fonts.gstatic.com",
        "img-src 'self' data:",
        "form-action 'self'",
        "base-uri 'none'",
        "frame-ancestors 'none'",
    )),
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
}
_INVITE_CSS = """
:root{color-scheme:dark}*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px 16px;
background:radial-gradient(1200px 600px at 50% -10%,#123127 0,#070b11 60%) #070b11;color:#e8edf3;
font:15px/1.5 Geist,system-ui,-apple-system,"Segoe UI",sans-serif}
.card{width:100%;max-width:420px;background:#0f1620;border:1px solid #223041;border-radius:18px;
padding:28px 24px;box-shadow:0 20px 60px rgba(0,0,0,.45);text-align:center}
.brand{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#8794a6;margin-bottom:14px}
h1{font-size:22px;line-height:1.25;margin:0 0 10px}p{margin:0 0 18px;color:#b7c2d0}
.btn{display:inline-block;width:100%;border:0;border-radius:12px;padding:13px 16px;font:inherit;
font-weight:700;cursor:pointer;text-decoration:none;background:linear-gradient(#f3d27a,#d9a93c);color:#1a1405}
"""


def _signin_html(head: str, text: str, next_path: str) -> str:
    """``text`` is HTML (its names already escaped); ``head`` is plain text."""
    def esc(s: str) -> str:  # (text nodes: & < > only — "You're" stays readable)
        return html_mod.escape(str(s), quote=False)
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{esc(head)} · Home games</title>"
            '<link rel="icon" type="image/svg+xml" href="/static/brand/wrap-app-icon-dark.svg">'
            '<link rel="apple-touch-icon" sizes="180x180" href="/static/brand/apple-touch-icon.png">'
            '<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400..800&display=swap" rel="stylesheet">'
            f"<style>{_INVITE_CSS}</style></head><body><main class=\"card\">"
            f'<div class="brand">WrapGTO · Home games</div><h1>{esc(head)}</h1><p>{text}</p>'
            f'<a class="btn" href="/auth/login?next={html_mod.escape(next_path, quote=True)}">Sign in with Google</a></main></body></html>')


def _invite_response(request: Request, path: str, user: Any) -> Response | None:
    """``pub._GAMES_INVITE_HOOK``: a /games page opened SIGNED OUT (or None for
    the hidden 404 — the API, the assets and unknown links)."""
    if user is not None or request.method.upper() not in ("GET", "HEAD"):
        return None
    if "text/html" not in request.headers.get("accept", "text/html"):
        return None
    m = _SIGNIN_PATH.match(path)
    if m is None:
        return None
    esc = html_mod.escape
    game_id, code = m.group(1), m.group(2)
    if game_id:
        table = pub.DB.one(
            "SELECT g.name, g.status, u.name AS host_name, c.name AS club_name FROM homegames g "
            "JOIN users u ON u.id=g.host_user_id LEFT JOIN homegame_clubs c ON c.id=g.club_id WHERE g.id=?",
            (game_id,),
        )
        if table is None or table["status"] != "open":
            return None
        club = f" in <b>{esc(str(table['club_name']))}</b>" if table["club_name"] else ""
        html = _signin_html(
            f"You're invited to {table['name']}",
            f"<b>{esc(str(table['host_name'] or 'A friend'))}</b> is hosting a PLO5 double-board bomb-pot "
            f"game{club}. Sign in with Google to join — you'll come straight back to the table.",
            path,
        )
    elif code:
        club = _club_by_code(code)
        if club is None:
            return None
        owner = pub._user_by_id(int(club["owner_user_id"]))
        html = _signin_html(
            f"Join {club['name']}",
            f"<b>{esc(_display_name(owner) if owner else 'A friend')}</b> invited you to their home-games club: "
            "PLO5 double-board bomb pots with friends, with the stats kept inside the club. "
            "Sign in with Google to join.",
            path,
        )
    else:
        html = _signin_html(
            "Home games",
            "PLO5 double-board bomb pots with your friends, in private clubs. Sign in with Google to start a "
            "club or join one.",
            "/games",
        )
    return HTMLResponse(html, headers=dict(INVITE_HEADERS))


def _main_club_member(uid: int) -> bool:
    """/admin's home-games column: is this person in the main club?"""
    cid = _main_club()
    return cid is not None and _club_role(cid, uid) is not None


def _admin_games_access(admin_uid: int, uid: int, grant: bool) -> bool:
    """/admin's home-games switch: add to / remove from the MAIN club (made on the
    first grant, owned by the granting admin). Returns membership afterwards."""
    cid = _main_club(create_for=admin_uid if grant else None)
    if cid is None:
        return False
    if grant:
        _add_member(cid, uid)
    elif _club_role(cid, uid) not in (None, "owner"):
        who = pub._user_by_id(uid)
        busy = _busy_in_club(cid, uid, _display_name(who) if who is not None else "They")
        if busy:
            raise HTTPException(status_code=409, detail=busy)
        pub.DB.q("DELETE FROM homegame_club_members WHERE club_id=? AND user_id=?", (cid, int(uid)))
    return _club_role(cid, uid) is not None


_ADMIN_IDS: dict[int, bool] = {}


def _viewer_is_admin(uid: int | None) -> bool:
    if uid is None:
        return False
    hit = _ADMIN_IDS.get(int(uid))
    if hit is None:
        hit = _ADMIN_IDS[int(uid)] = bool(pub._is_admin(pub._user_by_id(int(uid))))
    return hit


def _require_sync(t: LiveTable, body: dict, *, with_seq: bool) -> None:
    """Reject a request composed against a state the table has left.

    (review 2026-09-20 G8) `/act` carried no hand or decision identity, so a
    click that arrived late landed on the NEXT decision — "Call $1" could
    call a different bet, and an armed pre-fold folded the next hand. The
    client echoes the ``hand_no`` (and for `/act` the ``action_seq``) of the
    state it was looking at; a mismatch is 409 and the client just re-renders.
    Optional so scripted callers keep working."""
    if not isinstance(body, dict):
        return
    if body.get("hand_no") is not None:
        if pub.body_int(body, "hand_no") != t.hand_no:
            raise HTTPException(status_code=409, detail="stale: the hand has moved on")
    if with_seq and body.get("action_seq") is not None:
        if pub.body_int(body, "action_seq") != t.action_seq:
            raise HTTPException(status_code=409, detail="stale: the action has moved on")


def _stream_sig(t: LiveTable) -> tuple:
    """Everything that can change what a viewer sees WITHOUT bumping ``rev``
    (the runout reveals itself by the clock). Cheap, lock-free reads."""
    return (
        t.epoch, t.rev, t.phase, t.hand_no, t.action_seq, len(t.requests),
        _runout_shown_len(t) if t.runout_active else 0,
        _runout_award_index(t) if t.runout_active else 0,
        t.next_deal_mono is not None, t.bank_key,
    )


def install(app: FastAPI, *, static_dir: Path) -> None:
    _ensure_schema()
    pub._GAMES_INVITE_HOOK = _invite_response
    pub._GAMES_ACCESS_HOOK = _admin_games_access
    pub._GAMES_MEMBER_HOOK = _main_club_member

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
    def games_asset(name: str):
        """Gated, no-store copies of the lobby JS/CSS.

        Served under /games/ (not /static/) so a previously cached 404 on
        /static/games.js cannot keep the create button dead.
        """
        allowed = {name: media for name, media in GAMES_ASSETS.items()}
        media = allowed.get(name)
        if media is None:
            raise HTTPException(status_code=404, detail="Not Found")
        path = static_dir / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Not Found")
        return Response(
            path.read_text(encoding="utf-8"),
            media_type=media,
            headers={"Cache-Control": "no-store, must-revalidate", "X-Content-Type-Options": "nosniff"},
        )

    def _uid() -> int:
        uid = pub._CURRENT_USER_ID.get()
        if uid is None:
            raise HTTPException(status_code=404, detail="Not Found")
        return int(uid)

    def _table_for(game_id: str) -> LiveTable:
        """Every table endpoint comes through here: members of the table's club only."""
        t = HUB.get(game_id)
        _table_access(t, _uid())
        return t

    def _scope_clubs(uid: int, club: str | None, *, own: bool) -> list[str] | None:
        """Whose tables a stats query may read: one club the viewer is in; else, for
        the viewer's OWN numbers, everything they played (None); for somebody
        else's, only the clubs the viewer shares with them."""
        if club:
            row, _ = _require_club(club, uid)
            return [str(row["id"])]
        if own:
            return None
        return [c["id"] for c in _my_clubs(uid)]

    @app.get("/games/api/tables")
    def api_list(club: str | None = None):
        """The lobby: the viewer's clubs, one club's tables (plus the viewer's own
        seats elsewhere), their finished sessions there, and — for the club's
        owner and admins — who is asking to join."""
        viewer = _uid()
        clubs = _my_clubs(viewer)
        by_id = {c["id"]: c for c in clubs}
        if club and club not in by_id:
            raise HTTPException(status_code=404, detail="Not Found")
        rows = pub.DB.q(
            "SELECT * FROM homegames WHERE status='open' AND club_id IN (%s) ORDER BY created_at DESC"
            % ",".join("?" * len(by_id)), tuple(by_id),
        ) if by_id else []
        tables = []
        for r in rows:
            if not _lobby_visible(r, viewer):
                continue
            row = _lobby_row(r, viewer)
            if club and r["club_id"] != club and not (row["is_host"] or row["is_seated"]):
                continue
            row["club_id"] = str(r["club_id"])
            row["club_name"] = by_id[str(r["club_id"])]["name"]
            tables.append(row)
        manage = bool(club) and by_id[club]["role"] in ("owner", "admin")
        return {
            "clubs": clubs,
            "club": club,
            "tables": tables,
            "sessions": _my_sessions(viewer, club),
            "join_requests": _club_requests(club) if manage else [],
        }

    # --- clubs --------------------------------------------------------------------
    @app.get("/games/api/clubs")
    def api_clubs():
        return {"clubs": _my_clubs(_uid())}

    @app.post("/games/api/clubs")
    def api_club_create(body: dict = Body({})):
        uid = _uid()
        user = pub._user_by_id(uid)
        if user is None:
            raise HTTPException(status_code=404, detail="Not Found")
        return _club_view(_create_club(user, (body or {}).get("name")), uid)

    @app.get("/games/api/clubs/{club_id}")
    def api_club(club_id: str):
        return _club_view(club_id, _uid())

    @app.post("/games/api/clubs/{club_id}/settings")
    def api_club_settings(club_id: str, body: dict = Body({})):
        return _club_settings(club_id, _uid(), body or {})

    @app.post("/games/api/clubs/{club_id}/invite")
    def api_club_invite_reset(club_id: str):
        """A new invite link (the old one stops working)."""
        return _reset_invite(club_id, _uid())

    @app.post("/games/api/clubs/{club_id}/members")
    def api_club_member(club_id: str, body: dict = Body({})):
        body = body or {}
        return _set_member(club_id, _uid(), pub.body_int(body, "user_id"), role=body.get("role"),
                           remove=_parse_bool(body, "remove", False))

    @app.post("/games/api/clubs/{club_id}/leave")
    def api_club_leave(club_id: str):
        _leave_club(club_id, _uid())
        return {"ok": True}

    @app.post("/games/api/clubs/{club_id}/request")
    def api_club_request(club_id: str):
        """Ask to join (from a table link: the club's id only travels in the answer
        a table link gives a non-member). The owner or an admin decides."""
        uid = _uid()
        club = _club(club_id)
        if club is None:
            raise HTTPException(status_code=404, detail="Not Found")
        cid = str(club["id"])
        _request_club(cid, uid)
        status, retry = _request_state(cid, uid)
        return {"ok": True, "member": _club_role(cid, uid) is not None, "request": status,
                "retry_in": round(retry, 1)}

    @app.post("/games/api/clubs/{club_id}/requests/decide")
    def api_club_decide(club_id: str, body: dict = Body({})):
        return _decide_club_request(club_id, _uid(), pub.body_int(body, "user_id"), bool((body or {}).get("allow")))

    @app.get("/games/api/invites/{code}")
    def api_invite(code: str):
        return _invite_info(code, _uid())

    @app.post("/games/api/invites/{code}/join")
    def api_invite_join(code: str):
        return _join_by_invite(code, _uid())

    @app.post("/games/api/tables")
    def api_create(body: dict = Body({})):
        uid = pub._CURRENT_USER_ID.get()
        user = pub._user_by_id(int(uid)) if uid is not None else None
        if user is None:
            raise HTTPException(status_code=404, detail="Not Found")
        t = _create_table(user, body or {})
        with t.lock:
            return _view(t, int(user["id"]))

    @app.get("/games/api/tables/{game_id}")
    def api_get(game_id: str):
        t = _table_for(game_id)
        with t.lock:
            return _view(t, _uid())

    @app.post("/games/api/tables/{game_id}/sit")
    def api_sit(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        user = pub._user_by_id(uid)
        if user is None:
            raise HTTPException(status_code=404, detail="Not Found")
        seat = pub.body_int(body, "seat", -1)
        buyin = _parse_cents(body, "buyin_cents", t.default_buyin_cents)
        with t.lock:
            if _needs_approval(t, uid):
                _request_locked(t, user, "sit", seat, buyin)
            else:
                _sit_locked(t, user, seat, buyin)
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/leave")
    def api_leave(game_id: str, body: dict = Body({})):
        """Leave the seat. Holding cards: after this hand (the default) — or
        ``now`` = out of the hand at once (folded when facing a bet)."""
        t = _table_for(game_id)
        uid = _uid()
        with t.lock:
            _leave_locked(t, uid, now=_parse_bool(body or {}, "now", False))
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/sit_out")
    def api_sit_out(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        on = _parse_bool(body, "on", True)
        next_hand = _parse_bool(body, "next_hand", False)
        with t.lock:
            _sit_out_locked(t, uid, on, next_hand=next_hand)
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/settings")
    def api_settings(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        with t.lock:
            _settings_locked(t, uid, body or {})
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/transfer_host")
    def api_transfer_host(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        target = pub.body_int(body, "user_id")
        with t.lock:
            _transfer_host_locked(t, uid, target)
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/show")
    def api_show(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        with t.lock:
            _require_sync(t, body, with_seq=False)
            _show_locked(t, uid)
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/react")
    def api_react(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        with t.lock:
            _react_locked(t, uid, (body or {}).get("emote"))
            return _view(t, uid)

    @app.get("/games/api/my/hands")
    def api_my_hands(sort: str = "time", dir: str = "desc", game: str | None = None,
                     offset: int = 0, limit: int = 40, club: str | None = None):
        """The signed-in player's hand database (one club's, or all of it)."""
        uid = _uid()
        return _my_hands(uid, sort, dir, game, offset, limit, clubs=_scope_clubs(uid, club, own=True))

    @app.get("/games/api/my/stats")
    def api_my_stats(club: str | None = None):
        uid = _uid()
        return _my_stats(uid, clubs=_scope_clubs(uid, club, own=True))

    @app.get("/games/api/community")
    def api_community(club: str | None = None):
        """One club's numbers: player cards, the pairwise money, all sessions. The
        rankings never mix clubs. (No club named: the main club, else the first.)"""
        uid = _uid()
        if not club:
            mine = _my_clubs(uid)
            if not mine:
                return {"club": None, "players": [], "pairs": [], "sessions": [], "can_manage": False, "is_admin": False}
            club = mine[0]["id"]
        row, role = _require_club(club, uid)
        return _community(uid, str(row["id"]), role == "owner")

    @app.get("/games/api/players/{player_id}/stats")
    def api_player_stats(player_id: int, club: str | None = None):
        uid = _uid()
        if pub._user_by_id(int(player_id)) is None:
            raise HTTPException(status_code=404, detail="Not Found")
        return _my_stats(int(player_id), clubs=_scope_clubs(uid, club, own=int(player_id) == uid))

    @app.get("/games/api/players/{player_id}/hands")
    def api_player_hands(player_id: int, sort: str = "time", dir: str = "desc",
                         game: str | None = None, offset: int = 0, limit: int = 40, club: str | None = None):
        uid = _uid()
        return _my_hands(uid, sort, dir, game, offset, limit, player_id=int(player_id),
                         clubs=_scope_clubs(uid, club, own=int(player_id) == uid))

    @app.post("/games/api/tables/{game_id}/remove_chips")
    def api_remove_chips(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        with t.lock:
            _remove_chips_locked(t, uid, _parse_cents(body, "amount_cents"), queue_ok=_parse_bool(body, "queue", False))
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/stay")
    def api_stay(game_id: str):
        t = _table_for(game_id)
        uid = _uid()
        with t.lock:
            _cancel_leave_locked(t, uid)
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/fair/commit")
    def api_fair_commit(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        with t.lock:
            _fair_commit_locked(t, uid, str(body.get("hand_id") or ""), str(body.get("commit") or ""))
            return {"ok": True}

    @app.post("/games/api/tables/{game_id}/fair/reveal")
    def api_fair_reveal(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        with t.lock:
            _fair_reveal_locked(t, uid, str(body.get("hand_id") or ""), str(body.get("nonce") or ""))
            return {"ok": True}

    @app.get("/games/api/tables/{game_id}/fair/{hand_no}")
    def api_fair_transcript(game_id: str, hand_no: int):
        t = _table_for(game_id)
        uid = _uid()
        with t.lock:
            return _fair_transcript(t, uid, int(hand_no))

    @app.post("/games/api/tables/{game_id}/exclude")
    def api_exclude(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        user = pub._user_by_id(uid)
        on = _parse_bool(body, "on", True)
        with t.lock:
            _exclude_locked(t, user, on)
        return {"ok": True, "id": t.game_id, "excluded": on}

    @app.get("/games/api/tables/{game_id}/hands")
    def api_hands(game_id: str, before: int | None = None, limit: int = 30):
        t = _table_for(game_id)
        uid = _uid()
        with t.lock:
            _require_member(t, uid)
            return _hands_list(t, uid, before, limit)

    @app.get("/games/api/tables/{game_id}/hands/{hand_no}")
    def api_hand(game_id: str, hand_no: int):
        t = _table_for(game_id)
        uid = _uid()
        with t.lock:
            _require_member(t, uid)
            return _hand_detail(t, uid, hand_no)

    @app.post("/games/api/tables/{game_id}/sit_out_player")
    def api_sit_out_player(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        target = pub.body_int(body, "user_id")
        on = _parse_bool(body, "on", True)
        with t.lock:
            _sit_out_locked(t, uid, on, target_uid=target)
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/kick")
    def api_kick(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        target = pub.body_int(body, "user_id")
        with t.lock:
            _kick_locked(t, uid, target)
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/decision_time")
    def api_decision_time(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        secs = pub.body_int(body, "secs", 0)
        with t.lock:
            _set_decision_secs_locked(t, uid, secs)
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/street_pause")
    def api_street_pause(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        secs = _parse_secs(body, "secs", 1.5)
        with t.lock:
            _set_street_pause_locked(t, uid, secs)
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/rebuy")
    def api_rebuy(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        amount = _parse_cents(body, "amount_cents")
        queue = _parse_bool(body, "queue", False)
        with t.lock:
            if _needs_approval(t, uid):
                user = pub._user_by_id(uid)
                if user is None:
                    raise HTTPException(status_code=404, detail="Not Found")
                _request_locked(t, user, "rebuy", None, amount)
            else:
                _topup_locked(t, uid, amount, queue_ok=queue)
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/request")
    def api_request(game_id: str, body: dict = Body({})):
        """Host: approve / decline a pending buy-in. Requester: cancel their own."""
        t = _table_for(game_id)
        uid = _uid()
        action = str((body or {}).get("action") or "").strip().lower()
        with t.lock:
            if action == "cancel":
                _cancel_request_locked(t, uid)
            elif action in ("approve", "deny"):
                _resolve_request_locked(
                    t, uid, pub.body_int(body, "id"), action == "approve",
                    _parse_bool(body, "trust", False),
                    amount_cents=_parse_cents(body, "amount_cents") if body.get("amount_cents") is not None else None,
                )
            else:
                raise HTTPException(status_code=400, detail="action must be approve, deny or cancel")
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/trust")
    def api_trust(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        target = pub.body_int(body, "user_id")
        on = _parse_bool(body, "on", True)
        with t.lock:
            _set_trust_locked(t, uid, target, on)
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/auto_topup")
    def api_auto_topup(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        with t.lock:
            _auto_topup_host_locked(t, uid, body or {})
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/auto_chips_self")
    def api_auto_chips_self(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        with t.lock:
            _auto_chips_self_locked(t, uid, body or {})
            return _view(t, uid)

    @app.get("/games/api/tables/{game_id}/stream")
    async def api_stream(game_id: str, max_events: int = 0):
        """Live push (SSE): the viewer's state whenever it changes, plus a
        heartbeat. Replaces the 450 ms poll (which stays as the fallback)."""
        t = _table_for(game_id)
        uid = _uid()

        def snapshot() -> tuple[tuple, str | None]:
            # (removed from the club mid-stream: the push ends, and the client's
            # next poll gets the "ask to join" answer)
            try:
                _table_access(t, uid)
            except HTTPException:
                return (), None
            with t.lock:
                view = _view(t, uid)
                return _stream_sig(t), json.dumps(view, separators=(",", ":"))

        async def gen():
            sent = 0
            last_sig: tuple | None = None
            last_push = 0.0
            yield "retry: 1500\n\n"
            while not _WATCHDOG_STOP.is_set():
                now = time.monotonic()
                if _stream_sig(t) != last_sig or now - last_push >= STREAM_HEARTBEAT_S:
                    last_sig, payload = await run_in_threadpool(snapshot)
                    if payload is None:
                        return
                    last_push = now
                    yield f"data: {payload}\n\n"
                    sent += 1
                    if max_events and sent >= max_events:
                        return
                await asyncio.sleep(STREAM_TICK_S)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.post("/games/api/tables/{game_id}/run")
    def api_run(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        running = _parse_bool(body, "running", True)
        with t.lock:
            _set_running_locked(t, uid, running)
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/deal")
    def api_deal(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        with t.lock:
            if t.seat_of(uid) is None:
                raise HTTPException(status_code=400, detail="sit to deal")
            # `hand_no` = the finished hand the client saw; 409 if another
            # player (or the host's auto-deal) already dealt the next one.
            _require_sync(t, body, with_seq=False)
            _deal_locked(t)
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/act")
    def api_act(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        gate = _gate_key(body.get("gate"))
        chips = pub.body_int(body, "chips", 0)
        raise_to = (
            pub.body_int(body, "raise_to_chips")
            if gate == GATE_RAISE and body.get("raise_to_chips") is not None
            else None
        )
        # (review 2026-09-20 G8) ONE lock acquisition: the raise-TO -> raise-BY
        # conversion used to read the actor's street commit under one `with
        # t.lock`, release, and act under a second — another action could
        # land in between and the conversion applied to the wrong node.
        with t.lock:
            _require_sync(t, body, with_seq=True)
            if raise_to is not None:
                if t.env is None or t.env.current_actor() is None:
                    raise HTTPException(status_code=400, detail="no hand in progress")
                raw = _obs_dict(t.env)
                actor = int(raw["actor"])
                # Optional raise-TO total in chips -> the engine's raise-BY.
                chips = raise_to - int(raw["street_commit"][actor])
            _act_locked(t, uid, gate, chips)
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/host_fold")
    def api_host_fold(game_id: str):
        t = _table_for(game_id)
        uid = _uid()
        with t.lock:
            _host_fold_locked(t, uid)
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/rabbit")
    def api_rabbit(game_id: str):
        t = _table_for(game_id)
        uid = _uid()
        with t.lock:
            _rabbit_locked(t, uid)
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/auto_stack")
    def api_auto_stack(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        with t.lock:
            _auto_stack_host_locked(t, uid, body or {})
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/auto_stack_self")
    def api_auto_stack_self(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        cents = _parse_cents(body or {}, "cents", 0)
        with t.lock:
            _auto_stack_self_locked(t, uid, cents)
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/chat")
    def api_chat(game_id: str, body: dict = Body({})):
        t = _table_for(game_id)
        uid = _uid()
        user = pub._user_by_id(uid)
        if user is None:
            raise HTTPException(status_code=404, detail="Not Found")
        with t.lock:
            _chat_add(t, user, body.get("text", ""))
            return _view(t, uid)

    @app.post("/games/api/tables/{game_id}/close")
    def api_close(game_id: str):
        t = _table_for(game_id)
        uid = _uid()
        with t.lock:
            _close_locked(t, uid)
            return _view(t, uid)

    _start_watchdog()
    logger.info("homegame routes installed")
