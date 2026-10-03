"""The home games' database: the tables and their migrations.

Split out of ``homegame`` (HGB-006) and imported by it: every name defined here is
re-exported as ``homegame.<name>``. The code reaches every other home-games name
through ``hg`` (the ``homegame`` module), looked up when it runs, so patching
``homegame.X`` in a test reaches this module too and ``homegame.use_context`` swaps
its state. Patch ``homegame.X``, never this module's copy.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from typing import Any

from plo5bp.ui import public as pub

#: The home games' main module: every home-games name is looked up there when used.
hg = sys.modules.get("plo5bp.ui.homegame") or importlib.import_module("plo5bp.ui.homegame")
logger = logging.getLogger("plo5bp.ui.homegame")

__all__ = (
    "HAND_RECORD_VERSION", "HOMEGAME_MIGRATIONS", "_SCHEMA", "_ensure_schema",
    "_settle_lost_grades", "_v1_columns_added_before_versioning",
)


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
  allow_rathole INTEGER NOT NULL DEFAULT 0,
  variant TEXT NOT NULL DEFAULT 'plo5'
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
CREATE TABLE IF NOT EXISTS homegame_host_prefs (
  user_id INTEGER PRIMARY KEY,
  prefs TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS homegame_avatars (
  user_id INTEGER PRIMARY KEY,
  mime TEXT NOT NULL,
  data BLOB NOT NULL,
  version TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""


def _v1_columns_added_before_versioning(conn: Any) -> None:
    """Schema v1's second half: every column the pre-versioning code added one
    ALTER at a time, each checked on its own (OPS-018: ``acc_n`` used to be added
    only together with ``acc_sum`` — a crash between the two was never repaired).
    A database that already has them (every production one) is left as it is."""
    def cols(table: str) -> set[str]:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}

    def add(table: str, have: set[str], col: str, ddl: str) -> None:
        if col not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")

    g = cols("homegames")
    for c in hg.META:  # every setting added after the first schema carries its DDL
        if c.ddl is not None:
            add("homegames", g, c.col, c.ddl)
    add("homegames", g, "excluded", "INTEGER NOT NULL DEFAULT 0")
    add("homegames", g, "variant", "TEXT NOT NULL DEFAULT 'plo5'")  # 2026-09-26: PLO6
    add("homegames", g, "club_id", "TEXT")  # 2026-09-25: every table belongs to a club
    conn.execute("CREATE INDEX IF NOT EXISTS homegames_club ON homegames(club_id)")
    pl = cols("homegame_players")
    for col in ("auto_stack_cents", "trusted", "topup_target_cents", "topup_below_cents"):
        add("homegame_players", pl, col, "INTEGER NOT NULL DEFAULT 0")
    rs = cols("homegame_hand_results")
    add("homegame_hand_results", rs, "acc_sum", "REAL NOT NULL DEFAULT 0")
    add("homegame_hand_results", rs, "acc_n", "INTEGER NOT NULL DEFAULT 0")


#: The format of a stored hand record (``homegame_hands.summary``, its "v";
#: records from before the field are the same format, version 1). Bump it when
#: the format changes, and read older records by their "v" (OPS-018).
#: 2 (2026-09-29): an all-in runout's equities per street (``equities``: board
#: length -> seat -> [board 1, board 2] share) and ``runout_from`` (the board
#: length the players were all in on). Absent = no all-in runout (or an older record).
#: 3 (2026-10-02): ``uncalled`` {"seat", "cents"} — the bet nobody matched, returned
#: to its owner — and ``pot_cents`` without it (it was never in a pot). Absent = none.
HAND_RECORD_VERSION = 3

#: The home games' schema steps (OPS-018), tracked per component in
#: ``schema_migrations`` (``public.Db.migrate``). APPEND ONLY, never renumber;
#: every step so far only adds (rolling the code back stays safe). Data
#: migrations that must run on every start (``_migrate_clubs``) are not steps.
HOMEGAME_MIGRATIONS: list[Any] = [
    pub.Migration(
        1, "home games base schema",
        statements=tuple(x.strip() for x in _SCHEMA.split(";") if x.strip()),
        fn=_v1_columns_added_before_versioning,
    ),
    # PERF-003: the chat of one table, newest first, without scanning every line
    pub.Migration(2, "homegame_chat index", statements=(
        "CREATE INDEX IF NOT EXISTS homegame_chat_game ON homegame_chat(game_id, id)",
    )),
    # OPS-016: a finished hand waiting for its grades survives a restart
    pub.Migration(3, "grade jobs", statements=(
        "CREATE TABLE IF NOT EXISTS homegame_grade_jobs ("
        " game_id TEXT NOT NULL, hand_no INTEGER NOT NULL, job TEXT NOT NULL,"
        " created_at TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,"
        " PRIMARY KEY (game_id, hand_no))",
    )),
    # OPS-015: each hand's EXACT result in chips (delta_cents is rounded per hand;
    # NULL on hands recorded before this step)
    pub.Migration(4, "hand results in chips", fn=pub._add_column(
        "homegame_hand_results", "delta_chips", "INTEGER")),
    # SEC-004: the player tabled the hand with /show after it (showdown = at showdown)
    pub.Migration(5, "hand results: shown", fn=pub._add_column(
        "homegame_hand_results", "shown", "INTEGER NOT NULL DEFAULT 0")),
    # OPS-016: hands the old in-memory grading queue lost at a restart
    pub.Migration(6, "settle lost grading", fn=lambda conn: hg._settle_lost_grades(conn)),
    # TEST-008: the last hot lookups that scanned a growing table
    pub.Migration(7, "host and ledger indexes", statements=(
        "CREATE INDEX IF NOT EXISTS homegames_host ON homegames(host_user_id, status)",
        "CREATE INDEX IF NOT EXISTS homegame_ledger_game ON homegame_ledger(game_id, user_id)",
    )),
    # OPS-017: the hand a money movement came after (NULL on rows from before
    # this step — they also use the three old kinds, see LEDGER_IN / LEDGER_OUT)
    pub.Migration(8, "ledger hand numbers", fn=pub._add_column("homegame_ledger", "hand_no", "INTEGER")),
    # FEAT-008 / FEAT-006: the name a player chose for the tables, and a per-club
    # nickname (NULL = none: the club shows their table name)
    pub.Migration(9, "names at the table", statements=(
        "CREATE TABLE IF NOT EXISTS homegame_names (user_id INTEGER PRIMARY KEY, name TEXT NOT NULL,"
        " updated_at TEXT NOT NULL)",
    ), fn=pub._add_column("homegame_club_members", "nickname", "TEXT")),
    # FEAT-007: an archived club (NULL = active) — hidden, its history kept, restorable
    pub.Migration(10, "archived clubs", fn=pub._add_column("homegame_clubs", "archived_at", "TEXT")),
]


def _settle_lost_grades(conn: Any) -> None:
    """One-time data step: a hand still waiting for its grades when this step
    runs was queued by the pre-OPS-016 code in memory only — that queue died
    with the process, so the hand would say "still being worked out" forever.
    It is settled with no marks and a note."""
    rows = conn.execute(
        "SELECT game_id, hand_no, summary FROM homegame_hands WHERE summary LIKE ? OR summary LIKE ?",
        ('%"grades":null%', '%"grades": null%'),
    ).fetchall()
    for r in rows:
        try:
            rec = json.loads(r["summary"])
        except ValueError:
            continue
        if rec.get("grades") is None:
            rec["grades"] = []
            rec["grades_note"] = "not graded (the server restarted before it could)"
            conn.execute(
                "UPDATE homegame_hands SET summary=? WHERE game_id=? AND hand_no=?",
                (json.dumps(rec, separators=(",", ":")), r["game_id"], r["hand_no"]),
            )


def _ensure_schema() -> None:
    pub.DB.migrate("homegame", hg.HOMEGAME_MIGRATIONS)
    hg._migrate_clubs()
