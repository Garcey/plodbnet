#!/usr/bin/env python3
"""Server-side helper for ``scripts/deploy_prod.sh`` and the ``ops/`` jobs.

The deploy script BUNDLES this file with every call, so it never has to exist in
the live tree, and the same code is unit-tested on any machine
(``tests/python/ops/test_ops_deploytool.py``). Standard library only — except
``preflight``, which imports the app it is checking.

Subcommands (all print plain lines for a person; ``--json`` where noted):

  envfile PATH [--unit-show F] the environment the service really gets, as JSON:
                               the unit's Environment= then its EnvironmentFile=s
                               (``systemctl show`` output in F); PATH alone
                               without it
  envset --file F --set K=V --out F2   F with every K= line replaced by one K=V
  paths --env-json F --app D   the database and served-checkpoint paths the app
                               would use with that environment
  status --db PATH [--guard]   home games: hands in progress and games running —
                               from the running app's memory (``/health?deploy=1``)
                               when it can say, else from the database; hands cut
                               short earlier (void, not blocking); grading work.
                               ``--guard`` exits 3 when a restart now would cut a
                               hand short or stop a game (the deploy's check)
  preflight --root DIR ...     import the app from DIR the way systemd will run it
                               (production env as JSON on stdin, a COPY of the live
                               database, the live checkpoints) and check the served
                               model (loaded, critic, obs revision) and the engine
                               (built from this release's Rust sources?)
  health --url URL ...         poll /health until the app reports itself healthy
  unitcheck ...                what the systemd unit runs, where its settings come
                               from, and what installing ops/systemd/wrapgto.service
                               would break (``--new-unit``: exit 1 when it would)
  backup-check --dir DIR       the newest database backup: age, size, integrity
  dbcheck FILE                 ``PRAGMA quick_check`` of one database file (or .gz)
  dbcopy SRC DST               a consistent single-file copy of a database
  deps --lock FILE             compare ``pip freeze`` (stdin) with a lock file
  deployinfo --out F ...       write BUILD_INFO.json (what code is live; server.py's
                               /health reports it as build.commit)
  lockfile                     turn ``pip freeze`` (stdin) into a lock file
  engine-hash --crate DIR      the Rust sources' hash (= a fresh build's SOURCE_HASH)
  jsonget KEY                  one key of the JSON object on stdin

Exit codes: 0 ok, 1 check failed, 2 usage / environment problem, 3 guard tripped.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import sqlite3
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# environment files
# ---------------------------------------------------------------------------

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def parse_envfile(text: str) -> dict[str, str]:
    """Parse the ``KEY=VALUE`` subset of systemd's EnvironmentFile syntax.

    Blank lines and lines starting with ``#`` or ``;`` are ignored; a leading
    ``export`` is tolerated; whitespace around the key and the value is removed;
    one pair of matching outer quotes is removed (inside double quotes ``\\"``
    and ``\\\\`` are unescaped). Later assignments win, as in systemd.
    """
    out: dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        # systemd joins a line ending in a backslash with the next one
        while line.endswith("\\") and i < len(lines):
            line = line[:-1] + lines[i]
            i += 1
        s = line.strip().lstrip("﻿")
        if not s or s[0] in "#;":
            continue
        if s.startswith("export "):
            s = s[len("export "):].lstrip()
        if "=" not in s:
            continue
        key, value = s.split("=", 1)
        key = key.strip()
        if not _NAME.fullmatch(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            inner = value[1:-1]
            if value[0] == '"':
                inner = re.sub(r'\\(["\\])', r"\1", inner)
            value = inner
        out[key] = value
    return out


_ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def envfile_set(text: str, key: str, value: str) -> str:
    """``text`` with every assignment of ``key`` removed and ``key=value`` appended.

    Every other line stays byte for byte (comments, CRLF, continuation lines); a
    missing final newline is added before the new line — appending with ``echo >>``
    onto a last line without one would glue two settings together. Only plain
    values (no spaces, quotes, ``#`` or backslashes) are written."""
    if not _NAME.fullmatch(key):
        raise ValueError(f"not a variable name: {key!r}")
    if not value or re.search(r"[\s\"'\\#;$`]", value):
        raise ValueError(f"only a plain value can be written: {value!r}")
    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    i = 0
    while i < len(lines):
        j = i  # an assignment continued with a trailing backslash spans lines i..j
        while lines[j].rstrip("\r\n").endswith("\\") and j + 1 < len(lines):
            j += 1
        m = _ASSIGN.match(lines[i])
        if not (m and m.group(1) == key):
            kept.extend(lines[i:j + 1])
        i = j + 1
    out = "".join(kept)
    if out and not out.endswith("\n"):
        out += "\n"
    return out + f"{key}={value}\n"


def parse_systemctl_show(text: str) -> dict[str, list[str]]:
    """``systemctl show -p A -p B UNIT`` output -> {property: [values]} (a property
    printed on several lines, like EnvironmentFiles, keeps every line)."""
    out: dict[str, list[str]] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out.setdefault(k.strip(), []).append(v.strip())
    return out


def _first(show: dict[str, list[str]], key: str) -> str:
    vals = [v for v in show.get(key, []) if v]
    return vals[0] if vals else ""


def unit_environment(show: dict[str, list[str]]) -> dict[str, str]:
    """The unit's own ``Environment=`` assignments (``systemctl show`` quotes an
    assignment that contains spaces)."""
    env: dict[str, str] = {}
    for value in show.get("Environment", []):
        try:
            words = shlex.split(value)
        except ValueError:
            words = value.split()
        for w in words:
            k, sep, v = w.partition("=")
            if sep and _NAME.fullmatch(k):
                env[k] = v
    return env


def unit_env_files(show: dict[str, list[str]]) -> list[tuple[str, bool]]:
    """``[(path, optional)]`` of the unit's ``EnvironmentFile=`` lines, in order
    (``systemctl show`` prints ``/etc/wrapgto/env (ignore_errors=no)``)."""
    files = []
    for value in show.get("EnvironmentFiles", []):
        v = value.strip()
        if not v:
            continue
        m = re.match(r"^(.*?)\s+\(ignore_errors=(yes|no)\)$", v)
        path, optional = (m.group(1), m.group(2) == "yes") if m else (v, False)
        if path.startswith("-"):
            path, optional = path[1:], True
        files.append((path, optional))
    return files


def effective_env(show: dict[str, list[str]], fallback: Path | None
                  ) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """The environment systemd gives the service: the unit's ``Environment=``,
    then each ``EnvironmentFile=`` in order (a file overrides Environment=, a
    later file an earlier one). Returns (env, where each name came from, files
    that are listed but missing). With no unit information (the unit is not
    installed yet, or a test) the env file alone."""
    env: dict[str, str] = {}
    src: dict[str, str] = {}
    missing: list[str] = []
    for k, v in unit_environment(show).items():
        env[k], src[k] = v, "unit"
    files = unit_env_files(show)
    if not files and not env and fallback is not None:
        files = [(str(fallback), True)]
    for path, optional in files:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            if not optional:
                missing.append(path)
            continue
        for k, v in parse_envfile(text).items():
            env[k], src[k] = v, path
    return env, src, missing


def _read_env_json(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    if path == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8")
    raw = raw.strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise SystemExit("env JSON must be an object")
    return {str(k): str(v) for k, v in data.items()}


def resolve_paths(env: dict[str, str], app: Path) -> dict[str, str]:
    """The files the app would use with ``env`` when started in ``app``.

    Mirrors ``public.DB_PATH`` (``PLO5BP_DB`` or ``<app>/data/public.db``) and
    ``server._format_ckpt_path`` for PLO5 (``PLO5BP_CHECKPOINT`` or
    ``checkpoints/stub.pt``, relative paths resolved against the working
    directory, which systemd sets to the app folder)."""
    def _abs(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else app / q

    db = env.get("PLO5BP_DB", "").strip() or "data/public.db"
    ckpt = env.get("PLO5BP_CHECKPOINT", "").strip() or "checkpoints/stub.pt"
    nlh = env.get("PLO5BP_CHECKPOINT_NLH", "").strip() or "checkpoints/nlh_stub.pt"
    stats = env.get("PLO5BP_TRAINER_STATS", "").strip() or "checkpoints/trainer_stats.json"
    # Forward slashes: the shell script takes `dirname` of these.
    return {
        "db": _abs(db).as_posix(),
        "checkpoint": _abs(ckpt).as_posix(),
        "checkpoint_nlh": _abs(nlh).as_posix(),
        "trainer_stats": _abs(stats).as_posix(),
    }


def obs_rev_of(env: dict[str, str]) -> int | None:
    """The observation revision a process started with ``env`` encodes (the
    ``encoding._read_obs_semantics_rev`` rule: unset/blank = 2); None when the
    value is not one the app accepts (it would refuse to start)."""
    value = env.get("PLO5BP_OBS_REV", "").strip()
    if not value:
        return 2
    return int(value) if value in ("1", "2") else None


# ---------------------------------------------------------------------------
# home-games status (the "not mid-hand" guard)
# ---------------------------------------------------------------------------

#: Used only when the running app cannot report its tables from memory (an app
#: from before ``/health?deploy=1``): a hand finished this recently means a game
#: is being played — the server deals the next one within seconds.
ACTIVE_WINDOW = timedelta(minutes=15)
#: Finished hands still waiting for their grades this recently are reported.
GRADING_WINDOW = timedelta(minutes=30)


def _parse_time(s: Any) -> datetime | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _connect_ro(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def fetch_app_status(url: str, timeout: float = 5.0) -> tuple[str, dict[str, Any] | None]:
    """Ask the running app what a restart would interrupt (``GET /health?deploy=1``,
    answered from its memory, loopback callers only).

    Returns ``("app", home_games)`` when it answered — ``home_games`` is
    ``{"hands_in_progress": [...], "games_running": [...]}``, empty lists for a
    build without home games; ``("old", None)`` when it answers /health without
    the deploy block (an app from before it existed); ``("down", None)`` when
    nothing answers."""
    try:
        _code, body = fetch_health(url, timeout)
    except (urllib.error.URLError, OSError, ValueError):
        return "down", None
    if isinstance(body, dict) and isinstance(body.get("deploy"), dict):
        hg = body["deploy"].get("home_games")
        if hg is None:
            return "app", {"hands_in_progress": [], "games_running": []}
        if isinstance(hg, dict):
            return "app", hg
    return "old", None


def home_games_status(db: Path, now: datetime | None = None, live: dict[str, Any] | None = None,
                      app_running: bool = True) -> dict[str, Any]:
    """Read-only snapshot of what a restart would interrupt.

    ``live`` = the running app's own answer (``fetch_app_status``): exact —
    ``in_progress`` are its tables whose hand is being played (or whose all-in
    runout is still revealing), ``active`` its running games with players at
    the table. Without it, when the app is running: the database only — a hand
    recorded within ``ACTIVE_WINDOW`` means a game is on (and an unrecorded
    hand at such a table is in progress). ``app_running=False``: nothing can be
    in progress (it lives in the process's memory).

    ``void``: tables whose ``hand_no`` (saved at the deal) is above their last
    recorded hand but that nobody is playing — hands cut short earlier (a
    restart, or a table left idle for hours); they never block. ``grading``:
    finished hands still waiting for their grades — informational, since grading
    jobs are saved in ``homegame_grade_jobs`` and finish after a restart."""
    now = now or datetime.now(timezone.utc)
    source = "app" if live is not None else ("database" if app_running else "stopped")
    out: dict[str, Any] = {
        "db": str(db), "exists": db.exists(), "open_tables": 0, "source": source,
        "in_progress": [], "active": [], "void": [],
        "grading": {"waiting": 0, "older": 0, "saved_jobs": None},
        "last_hand_at": None,
    }
    if not db.exists():
        return out
    con = _connect_ro(db)
    try:
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "homegames" not in names:
            return out
        has_hands = "homegame_hands" in names
        rows = con.execute(
            "SELECT id, name, hand_no FROM homegames WHERE status = 'open'"
        ).fetchall()
        out["open_tables"] = len(rows)
        tables: dict[str, dict[str, Any]] = {}
        for r in rows:
            last_no, last_at = 0, None
            if has_hands:
                h = con.execute(
                    "SELECT hand_no, ended_at FROM homegame_hands WHERE game_id = ? "
                    "ORDER BY hand_no DESC LIMIT 1", (r["id"],)
                ).fetchone()
                if h is not None:
                    last_no, last_at = int(h["hand_no"]), _parse_time(h["ended_at"])
            tables[str(r["id"])] = {
                "id": str(r["id"]), "name": r["name"], "hand_no": int(r["hand_no"] or 0),
                "last_recorded": last_no, "last_hand_at": last_at.isoformat() if last_at else None,
                "_recent": last_at is not None and now - last_at <= ACTIVE_WINDOW,
            }
        busy: set[str] = set()
        if live is not None:
            for x in live.get("hands_in_progress") or []:
                if isinstance(x, dict) and x.get("id") is not None:
                    gid = str(x["id"])
                    t = tables.get(gid, {"id": gid, "name": gid, "hand_no": 0})
                    busy.add(gid)
                    out["in_progress"].append({"id": gid, "name": t["name"],
                                               "hand_no": int(x.get("hand_no") or t["hand_no"])})
            for x in live.get("games_running") or []:
                if isinstance(x, dict) and x.get("id") is not None and str(x["id"]) not in busy:
                    gid = str(x["id"])
                    t = tables.get(gid, {"id": gid, "name": gid, "last_hand_at": None})
                    out["active"].append({"id": gid, "name": t["name"], "present": x.get("present"),
                                          "last_hand_at": t.get("last_hand_at")})
        for t in tables.values():
            unrecorded = t["hand_no"] > t["last_recorded"]
            if live is None and app_running and t["_recent"]:
                if unrecorded:
                    out["in_progress"].append({k: t[k] for k in ("id", "name", "hand_no")})
                else:
                    out["active"].append({k: t[k] for k in ("id", "name", "last_hand_at")})
            elif unrecorded and t["id"] not in busy:
                out["void"].append({k: t[k] for k in ("id", "name", "hand_no", "last_hand_at")})
        if has_hands:
            out["last_hand_at"] = con.execute("SELECT MAX(ended_at) FROM homegame_hands").fetchone()[0]
            try:
                pend = con.execute(
                    "SELECT ended_at FROM homegame_hands "
                    "WHERE json_type(summary, '$.grades') = 'null'"
                ).fetchall()
            except sqlite3.OperationalError:  # an SQLite without JSON functions
                pend = []
            for (ended,) in pend:
                t_end = _parse_time(ended)
                key = "waiting" if t_end is not None and now - t_end <= GRADING_WINDOW else "older"
                out["grading"][key] += 1
        if "homegame_grade_jobs" in names:
            out["grading"]["saved_jobs"] = int(
                con.execute("SELECT COUNT(*) FROM homegame_grade_jobs").fetchone()[0])
    finally:
        con.close()
    return out


def format_status(st: dict[str, Any]) -> list[str]:
    if not st["exists"]:
        return [f"   database: not found ({st['db']})"]
    how = {
        "app": "asked the running app (its memory — exact)",
        "database": "from the database: the running app is too old to report its tables",
        "stopped": "the app is not running, so no hand can be in progress",
    }[st["source"]]
    lines = [f"   open home-game tables: {st['open_tables']} ({how})"]
    for t in st["in_progress"]:
        lines.append(f"   HAND IN PROGRESS: table {t['name']!r} — hand #{t['hand_no']} is being played")
    for t in st["active"]:
        if st["source"] == "app":
            who = f"{t['present']} players at the table" if t.get("present") is not None else "players at the table"
            lines.append(f"   game running: table {t['name']!r} — the server is dealing ({who})")
        else:
            lines.append(f"   game running: table {t['name']!r} — last hand ended {t['last_hand_at']}")
    for t in st["void"]:
        lines.append(f"   (void, not blocking) table {t['name']!r}: hand #{t['hand_no']} never finished — "
                     "cut short by an earlier restart or an idle table; its chips went back")
    if st["void"] and st["source"] == "database":
        lines.append("   (an app this old cannot say whether a game started at one of those tables in the "
                     "last few minutes — its first hand would be cut short too; the lobby shows who is playing)")
    g = st["grading"]
    if g["waiting"]:
        if g["saved_jobs"] is not None:
            lines.append(f"   hands still being graded: {g['waiting']} (saved — they finish after a restart)")
        else:
            lines.append(f"   hands still being graded: {g['waiting']} (this app keeps its grading queue in "
                         "memory: a restart now settles them without grades)")
    if g["older"]:
        lines.append(f"   (older hands left ungraded by an earlier restart: {g['older']})")
    if st["last_hand_at"]:
        lines.append(f"   last home-game hand ended: {st['last_hand_at']}")
    return lines


def guard_problems(st: dict[str, Any]) -> list[str]:
    """What a restart now would interrupt — the deploy's reasons to wait. Void
    hands and grading never block (grading jobs survive a restart)."""
    probs = []
    if st["in_progress"]:
        probs.append(f"{len(st['in_progress'])} hand(s) in progress")
    if st["active"]:
        if st["source"] == "app":
            probs.append(f"{len(st['active'])} game(s) running — the server deals the next hand within seconds")
        else:
            probs.append(f"{len(st['active'])} table(s) with a game running (a hand ended in the last "
                         f"{int(ACTIVE_WINDOW.total_seconds() // 60)} min)")
    return probs


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


def _find_key(obj: Any, key: str) -> tuple[bool, Any]:
    """Depth-first search for ``key`` in nested dicts/lists (first hit wins)."""
    if isinstance(obj, dict):
        if key in obj:
            return True, obj[key]
        for v in obj.values():
            hit, val = _find_key(v, key)
            if hit:
                return hit, val
    elif isinstance(obj, list):
        for v in obj:
            hit, val = _find_key(v, key)
            if hit:
                return hit, val
    return False, None


def evaluate_health(payload: Any, expect_commit: str = "", allow_no_critic: bool = False,
                    status: int = 200, expect_obs_rev: int | None = None) -> list[str]:
    """Problems with a ``/health`` answer (empty list = healthy).

    Tolerant of the payload's shape: each of ``model_loaded`` / ``critic_loaded``
    / ``obs_rev_mismatch`` / ``commit`` / ``process_obs_rev`` is checked wherever
    it appears, and ``ok: false`` counts through the app's own ``problems`` list
    when it sends one (``server.health_report``: a missing critic makes the app
    "degraded", which ``allow_no_critic`` accepts). An older app that answers
    only ``{"ok": true}`` is healthy — unless ``expect_commit`` is given: code
    deployed with a BUILD_INFO.json must answer with that commit, so a missing
    one means an older process is still answering (or PLO5BP_BUILD_COMMIT
    overrides it)."""
    if not isinstance(payload, dict):
        return [f"/health did not return a JSON object (HTTP {status})"]
    probs = []
    if payload.get("ok") is False:
        listed = payload.get("problems")
        if isinstance(listed, list) and listed:
            for p in listed:
                text = str(p)
                if "critic" in text.lower() and allow_no_critic:
                    continue
                if "model not loaded" in text or "obs rev" in text or "critic" in text.lower():
                    continue  # reported below from the named fields, in plain words
                probs.append(f"the app reports: {text}")
        else:
            probs.append("/health says ok=false")
    if status >= 500 and not probs and payload.get("ok") is not False:
        probs.append(f"/health answered HTTP {status}")
    hit, v = _find_key(payload, "model_loaded")
    if hit and v is not True:
        probs.append("the PLO5 model did NOT load — the site would serve a random, untrained network")
    hit, v = _find_key(payload, "critic_loaded")
    if hit and v is not True and not allow_no_critic:
        probs.append("the critic did not load (the review's true EV is off)")
    hit, v = _find_key(payload, "obs_rev_mismatch")
    if hit and v:
        probs.append("OBS-REV MISMATCH: the served model and the encoder disagree (PLO5BP_OBS_REV)")
    if expect_commit:
        hit, v = _find_key(payload, "commit")
        want = expect_commit.strip().lower()
        if not (hit and isinstance(v, str) and v.strip()):
            probs.append(f"/health reports no commit — expected {expect_commit[:12]} "
                         "(an older process still answering, or no BUILD_INFO.json)")
        else:
            got = v.strip().lower()
            if not (got.startswith(want) or want.startswith(got)):
                probs.append(f"/health reports commit {v} — expected {expect_commit} (an old process?)")
    if expect_obs_rev is not None:
        hit, v = _find_key(payload, "process_obs_rev")
        if hit and v is not None and str(v) != str(expect_obs_rev):
            probs.append(f"the app encodes obs rev {v} — expected {expect_obs_rev} (PLO5BP_OBS_REV)")
    return probs


def summarize_health(payload: Any) -> str:
    """One line for a person: the app's verdict, model, critic, obs rev, commit."""
    if not isinstance(payload, dict):
        return repr(payload)[:200]
    parts = [str(payload.get("status") or ("ok" if payload.get("ok") else "NOT ok"))]
    for key, label in (("model_loaded", "model"), ("critic_loaded", "critic")):
        hit, v = _find_key(payload, key)
        if hit:
            parts.append(f"{label} {'loaded' if v else 'NOT LOADED'}")
    hit, v = _find_key(payload, "obs_rev")
    if hit:
        parts.append(f"obs rev {v}")
    hit, v = _find_key(payload, "commit")
    if hit and v:
        parts.append(f"commit {str(v)[:7]}")
    if isinstance(payload.get("uptime_s"), int):
        parts.append(f"up {payload['uptime_s']} s")
    return " · ".join(parts)


def fetch_health(url: str, timeout: float = 5.0) -> tuple[int, Any]:
    """(HTTP status, parsed JSON). A 503 still carries the app's own verdict
    (``server.health_report`` answers 503 when the model is broken)."""
    req = urllib.request.Request(url, headers={"User-Agent": "wrapgto-deploytool"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (loopback URL)
            return resp.status, json.loads(resp.read(1 << 20).decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read(1 << 20)
        try:
            return e.code, json.loads(body.decode("utf-8"))
        except ValueError:
            return e.code, None


# ---------------------------------------------------------------------------
# the systemd unit (check / install-unit)
# ---------------------------------------------------------------------------

#: ProtectHome=yes hides these from the service: a venv (or the interpreter it
#: was made from) living there cannot start under the hardened unit.
PROTECTED_HOME = ("/root", "/home", "/run/user")


def parse_unit_file(text: str) -> dict[str, list[str]]:
    """``Key=Value`` lines of a unit file (all sections; repeats accumulate,
    continuation lines joined). Enough to read what ops/systemd/wrapgto.service
    sets — not a full systemd parser."""
    out: dict[str, list[str]] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        while line.endswith("\\") and i < len(lines):
            line = line[:-1] + " " + lines[i].strip()
            i += 1
        if not line or line[0] in "#;[" or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out.setdefault(k.strip(), []).append(v.strip())
    return out


def _execstart_argv(show: dict[str, list[str]], cat_text: str) -> str:
    m = re.search(r"argv\[\]=(.*?) ;", _first(show, "ExecStart"))
    if m:
        return m.group(1).strip()
    for line in cat_text.splitlines():
        s = line.strip()
        if s.startswith("ExecStart=") and s != "ExecStart=":
            return s.split("=", 1)[1].strip()
    return ""


def _under(path: str, roots: Iterable[str]) -> bool:
    p = path.rstrip("/") + "/"
    return any(p.startswith(r.rstrip("/") + "/") for r in roots)


def venv_paths(venv_python: Path) -> tuple[str, str]:
    """(where the venv's python really is, the ``home`` its pyvenv.cfg names)."""
    real = os.path.realpath(str(venv_python))
    home = ""
    cfg = Path(os.path.realpath(str(venv_python.parent.parent))) / "pyvenv.cfg"
    try:
        for line in cfg.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() == "home":
                home = v.strip()
    except OSError:
        pass
    return real, home


def unit_findings(show: dict[str, list[str]], cat_text: str, app: str, env: dict[str, str],
                  sources: dict[str, str], envfile: str, missing_files: list[str],
                  new_unit_text: str | None = None, venv_python: str | None = None,
                  venv_home: str | None = None,
                  exists: Any = os.path.exists) -> tuple[list[str], list[str], list[str]]:
    """(info lines, warnings, blocking problems) about the service definition.

    Without ``new_unit_text``: what the live unit runs and anything that makes
    the deploy's view of it incomplete (settings the tools cannot see, drop-ins,
    ``--workers``). With it: what installing that unit would break — each one a
    blocking problem with the fix."""
    info: list[str] = []
    warn: list[str] = []
    block: list[str] = []
    frag = _first(show, "FragmentPath")
    dropins = [p for v in show.get("DropInPaths", []) for p in v.split() if p]
    argv = _execstart_argv(show, cat_text)
    info.append(f"   definition: {frag or '(not installed)'}"
                + (f" + drop-ins: {' '.join(dropins)}" if dropins else ""))
    if _first(show, "User") or _first(show, "WorkingDirectory"):
        info.append(f"   runs as {_first(show, 'User') or 'root'} in {_first(show, 'WorkingDirectory') or '/'}")
    if argv:
        info.append(f"   starts: {argv}")
    files = [p for p, _opt in unit_env_files(show)]
    from_unit = sorted(k for k, s in sources.items() if s == "unit")
    other_files = [f for f in files if f != envfile]
    info.append(f"   settings: {sum(1 for s in sources.values() if s == envfile)} from {envfile}"
                + (f"; in the unit itself (Environment=): {' '.join(from_unit)}" if from_unit else "")
                + (f"; other files: {' '.join(other_files)}" if other_files else ""))
    if venv_python:
        info.append(f"   python: {venv_python}" + (f" (made from {venv_home})" if venv_home else ""))
    if "--workers" in argv:
        warn.append("the unit starts uvicorn with --workers: the home games need exactly ONE process")
    if argv and "--timeout-graceful-shutdown" not in argv:
        warn.append("the unit lacks --timeout-graceful-shutdown 3 — every stop waits for open live "
                    "streams until systemd kills the app")
    if env.get("PLO5BP_BUILD_COMMIT", "").strip():
        warn.append("PLO5BP_BUILD_COMMIT is set: /health reports it instead of the deployed commit, so a "
                    "deploy cannot confirm the new code answers — remove that line")
    for f in missing_files:
        warn.append(f"the unit reads {f}, which does not exist — the app cannot start")
    unit_only = [k for k in from_unit if k not in ("PYTHONUNBUFFERED", "HOME", "XDG_CACHE_HOME")]
    if unit_only:
        warn.append(f"settings kept in the unit itself (Environment=): {' '.join(unit_only)} — the nightly "
                    f"backup and the ops tools read only {envfile}; move them there")
    if other_files:
        warn.append(f"settings also come from {' '.join(other_files)} — the ops tools read only {envfile}")
    if dropins:
        warn.append(f"drop-ins change the unit: {' '.join(dropins)}")
    for f in [frag] + dropins:
        if f and _under(os.path.realpath(f), [os.path.realpath(app)]):
            warn.append(f"{f} lives inside {app}, which every deploy moves away — install the unit with "
                        "bash scripts/deploy_prod.sh install-unit")
    if new_unit_text is None:
        return info, warn, block

    # --- what installing the new unit would break -------------------------------------
    new = parse_unit_file(new_unit_text)
    new_env = unit_environment({"Environment": new.get("Environment", [])})
    new_files = [f.lstrip("-") for f in new.get("EnvironmentFile", [])]
    if dropins:
        block.append(f"drop-ins would still apply on top of the new unit (and can override it): "
                     f"{' '.join(dropins)} — keep what you need from them in {envfile}, then move them "
                     f"away: mkdir -p /root/wrapgto-dropins.old && mv {' '.join(dropins)} "
                     "/root/wrapgto-dropins.old/")
    lost = sorted(k for k, s in sources.items()
                  if s not in new_files and k not in new_env)
    if lost:
        block.append(f"these settings would be LOST (they are not in {' '.join(new_files) or 'its env file'}): "
                     f"{' '.join(lost)} — see their values with: systemctl show -p Environment wrapgto; "
                     f"add each NAME=value line to {envfile}, then run this again")
    for f in new_files:
        if not exists(f):
            block.append(f"the new unit reads {f}, which does not exist")
    wd = (new.get("WorkingDirectory") or [""])[-1]
    if wd and wd.rstrip("/") != app.rstrip("/"):
        block.append(f"the new unit runs the app in {wd}, but the app is in {app}")
    user = (new.get("User") or [""])[-1]
    if user and user != _first(show, "User") and _first(show, "User"):
        block.append(f"the new unit runs as {user}, the current one as {_first(show, 'User')} — "
                     "the data files belong to the current user")
    if (new.get("ProtectHome") or ["no"])[-1] in ("yes", "true", "tmpfs"):
        for label, p in (("the venv's python", venv_python), ("the python it was made from", venv_home)):
            if p and _under(p, PROTECTED_HOME):
                block.append(f"{label} is {p}, under {'/'.join(PROTECTED_HOME)} — the new unit's "
                             "ProtectHome=yes hides it and the app would not start. Rebuild the venv from "
                             "/usr/bin/python3 (ops/SERVER_SETUP.md, 'Python environment')")
    rw = [p for v in new.get("ReadWritePaths", []) for p in v.split() if p]
    if (new.get("ProtectSystem") or [""])[-1] == "strict" and rw:
        paths = resolve_paths(env, Path(app))
        for key, label in (("db", "PLO5BP_DB (the database)"), ("trainer_stats", "PLO5BP_TRAINER_STATS"),
                           ("checkpoint", "PLO5BP_CHECKPOINT"), ("checkpoint_nlh", "PLO5BP_CHECKPOINT_NLH")):
            folder = os.path.dirname(paths[key])
            if not _under(folder, rw):
                block.append(f"{label} is in {folder}, which the new unit makes read-only (it may write "
                             f"only {' '.join(rw)}) — move it under {app}/data or {app}/checkpoints")
    exec_new = (new.get("ExecStart") or [""])[-1]
    py = exec_new.split()[0] if exec_new else ""
    if py and not exists(py):
        block.append(f"the new unit starts {py}, which does not exist")
    return info, warn, block


# ---------------------------------------------------------------------------
# backups
# ---------------------------------------------------------------------------

_BACKUP_SKIP = re.compile(r"^(app-before-|replaced-|\.)")
_SQLITE_MAGIC = b"SQLite format 3\x00"


def _backup_candidates(folder: Path) -> list[Path]:
    """Regular files in ``folder`` and one level of subfolders, newest first,
    ignoring the deploy's own ``app-before-*`` code copies, ``replaced-*``
    folders (restore-db) and dotfiles."""
    if not folder.is_dir():
        return []
    files = []
    for p in folder.iterdir():
        if _BACKUP_SKIP.match(p.name):
            continue
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(q for q in p.iterdir() if q.is_file() and not _BACKUP_SKIP.match(q.name))
    return sorted(files, key=lambda q: q.stat().st_mtime, reverse=True)


def looks_like_sqlite(path: Path) -> bool:
    """True for an SQLite file or a gzip of one — judged by content, not name,
    so a backup job with any naming scheme is recognised."""
    try:
        if path.name.endswith(".gz"):
            with gzip.open(path, "rb") as fh:
                return fh.read(16) == _SQLITE_MAGIC
        with open(path, "rb") as fh:
            return fh.read(16) == _SQLITE_MAGIC
    except (OSError, EOFError):
        return False


def newest_backup(folder: Path) -> Path | None:
    """The newest database backup (SQLite or gzipped SQLite) in ``folder``."""
    for p in _backup_candidates(folder):
        if looks_like_sqlite(p):
            return p
    return None


def sqlite_quick_check(path: Path) -> str:
    """``PRAGMA quick_check`` of a database file (``.gz`` is decompressed to a
    temporary copy first). Returns "ok" or the first problem."""
    tmp = None
    try:
        if path.name.endswith(".gz"):
            fd, tmp = tempfile.mkstemp(suffix=".db")
            with os.fdopen(fd, "wb") as out, gzip.open(path, "rb") as src:
                shutil.copyfileobj(src, out)
            target = Path(tmp)
        else:
            target = path
        with open(target, "rb") as fh:
            if fh.read(16) != b"SQLite format 3\x00":
                return "not an SQLite database"
        con = sqlite3.connect(f"file:{target.as_posix()}?mode=ro&immutable=1", uri=True)
        try:
            row = con.execute("PRAGMA quick_check").fetchone()
        finally:
            con.close()
        return str(row[0]) if row else "no result"
    except (OSError, sqlite3.DatabaseError, EOFError) as exc:
        return f"unreadable: {exc}"
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def backup_report(folder: Path, max_age_hours: float, newer_than: Path | None = None,
                  now: float | None = None) -> tuple[list[str], list[str]]:
    """(info lines, problems) about the newest backup in ``folder``.

    With ``newer_than`` (a marker touched just before the backup job ran): the
    job must have written SOMETHING newer than the marker, and a new database
    file must pass ``PRAGMA quick_check``. A new file in an unrecognised format
    (an older job's archive) is reported but not treated as a failure."""
    now = time.time() if now is None else now
    info: list[str] = []
    probs: list[str] = []
    marker = newer_than.stat().st_mtime if newer_than is not None and newer_than.exists() else None
    cands = _backup_candidates(folder)
    if marker is not None and not any(p.stat().st_mtime >= marker for p in cands):
        return info, ["the backup job did not write a new file"]
    b = newest_backup(folder)
    if b is None:
        if marker is not None and cands:
            return [f"   new backup file: {cands[0].name} (not a plain SQLite file — not verified)"], []
        return info, [f"no database backup found in {folder}"]
    st = b.stat()
    age_h = (now - st.st_mtime) / 3600.0
    info.append(f"   newest database backup: {b.name} · {st.st_size / 1e6:.1f} MB · {age_h:.1f} h old")
    if marker is not None and st.st_mtime < marker:
        info.append(f"   new backup file: {cands[0].name} (not a plain SQLite file — not verified)")
        return info, probs
    if st.st_size == 0:
        probs.append(f"the newest backup {b.name} is EMPTY")
    if age_h > max_age_hours:
        probs.append(f"the newest backup is {age_h:.1f} h old (limit {max_age_hours:g} h)")
    if not probs:
        qc = sqlite_quick_check(b)
        if qc != "ok":
            probs.append(f"the newest backup failed its integrity check: {qc}")
        else:
            info.append("   integrity: ok")
    return info, probs


# ---------------------------------------------------------------------------
# dependencies
# ---------------------------------------------------------------------------


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pins(text: str) -> dict[str, str]:
    """``name==version`` lines of a lock / ``pip freeze`` output -> {name: version}.
    Options (``--extra-index-url``), comments, editable and URL installs are skipped."""
    pins = {}
    for line in text.splitlines():
        s = line.split("#", 1)[0].strip()
        if not s or s.startswith("-") or " @ " in s:
            continue
        s = s.split(";", 1)[0].strip()  # environment markers
        s = s.split(" \\", 1)[0].strip()  # "--hash" continuation
        m = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?==([^\s]+)", s)
        if m:
            pins[_norm(m.group(1))] = m.group(3)
    return pins


def deps_diff(lock_text: str, freeze_text: str) -> list[str]:
    """What the installed packages (``pip freeze``) lack compared with the lock."""
    lock, have = parse_pins(lock_text), parse_pins(freeze_text)
    out = []
    for name, ver in sorted(lock.items()):
        if name in _UNPINNED:
            continue
        got = have.get(name)
        if got is None:
            out.append(f"{name}=={ver} (not installed)")
        elif got != ver:
            out.append(f"{name}=={ver} (installed: {got})")
    return out


TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
#: Never pinned (as pip-tools does): the installer's own tools, and the app itself.
_UNPINNED = {"pip", "setuptools", "wheel", "distribute", "plo5bp"}


def make_lockfile(freeze_text: str, header: str = "") -> str:
    """A lock file from ``pip freeze`` output: sorted ``name==version`` lines,
    the app itself (editable / local installs) and pip's own tools left out, and
    PyTorch's CPU index added when a ``+cpu`` wheel is pinned (pip needs it to
    find that build)."""
    pins = []
    for line in freeze_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("-") or " @ " in s:
            continue
        name = re.split(r"[=<>!~ \[]", s, maxsplit=1)[0]
        if _norm(name) in _UNPINNED:
            continue
        pins.append(s)
    pins.sort(key=lambda s: _norm(re.split(r"[=<>!~ \[]", s, maxsplit=1)[0]))
    out = []
    if header:
        out += [f"# {h}" if h else "#" for h in header.splitlines()]
    if any("+cpu" in p for p in pins):
        out.append(f"--extra-index-url {TORCH_CPU_INDEX}")
    out += pins
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# the engine and BUILD_INFO.json
# ---------------------------------------------------------------------------


def engine_source_hash(crate: Path) -> str:
    """FNV-1a 64 over ``crate/src/**`` (path-sorted as "src/<path>"), then the
    crate's ``Cargo.toml`` and the workspace ``../Cargo.lock``: for each, the
    path, a NUL, the bytes with CRLF normalised to LF, a NUL. The THIRD copy of
    this rule — ``rust_engine/build.rs`` bakes it into the engine as
    ``SOURCE_HASH`` and ``tests/conftest.py::engine_source_hash`` checks builds;
    test_ops_deploytool.py pins all three together."""
    h = 0xCBF29CE484222325
    prime = 0x100000001B3
    src = sorted(("src/" + p.relative_to(crate / "src").as_posix(), p)
                 for p in (crate / "src").rglob("*") if p.is_file())
    for rel, p in src + [("Cargo.toml", crate / "Cargo.toml"), ("../Cargo.lock", crate.parent / "Cargo.lock")]:
        if not p.is_file():
            continue
        for chunk in (rel.encode(), b"\0", p.read_bytes().replace(b"\r\n", b"\n"), b"\0"):
            for b in chunk:
                h = ((h ^ b) * prime) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def deploy_info(commit: str, dirty: bool, diff_sha256: str, branch: str, deployed_by: str,
                engine_so: Path | None, engine_source: str, lock_sha256: str,
                when: str | None = None, preflight: dict[str, Any] | None = None) -> dict[str, Any]:
    when = when or datetime.now(timezone.utc).isoformat(timespec="seconds")
    info: dict[str, Any] = {
        "commit": commit,
        "short": commit[:7],
        "branch": branch,
        "dirty": bool(dirty),
        "diff_sha256": diff_sha256 or None,
        "deployed_at": when,
        "built_at": when,  # the name server._build_info() reads
        "deployed_by": deployed_by,
        "engine": {"source": engine_source},
        "requirements_sha256": lock_sha256 or None,
    }
    if engine_so is not None and engine_so.exists():
        info["engine"].update(file=engine_so.name, sha256=sha256_file(engine_so))
    if preflight:
        # what the engine was built from vs the Rust sources this release ships
        info["engine"].update(
            source_hash=preflight.get("engine_source_hash"),
            rust_sources_hash=preflight.get("rust_sources_hash"),
            matches_sources=preflight.get("engine_matches_sources"),
        )
    return info


# ---------------------------------------------------------------------------
# preflight (imports the app — run as the service user)
# ---------------------------------------------------------------------------


def _copy_db(src: Path, dst: Path) -> str:
    """Consistent copy of a live database via SQLite's backup API (safe while
    the app writes to it; reads through its -wal). Returns a note for the log."""
    s = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True, timeout=30)
    try:
        d = sqlite3.connect(str(dst))
        try:
            s.backup(d)
        finally:
            d.close()
    finally:
        s.close()
    return f"a copy of the live database ({dst.stat().st_size / 1e6:.1f} MB)"


def _engine_check(root: Path, args: argparse.Namespace) -> tuple[dict[str, Any], list[str], list[str]]:
    """(summary fields, notes, problems): was the engine built from the Rust
    sources this release ships? (``SOURCE_HASH`` baked in by rust_engine/build.rs
    vs the same hash of ``root/rust_engine``.) The engine decides the rules and
    pays out every pot, so a deploy refuses one it cannot match to its sources."""
    import importlib

    eng = sys.modules.get("plo5bp._engine")
    if eng is None:
        try:
            eng = importlib.import_module("plo5bp._engine")
        except Exception:  # noqa: BLE001 — reported as "unknown" below
            eng = None
    built = getattr(eng, "SOURCE_HASH", None) if eng is not None else None
    built = str(built) if built else None
    crate = root / "rust_engine"
    shipped = engine_source_hash(crate) if (crate / "src").is_dir() else None
    match = (built == shipped) if (built and shipped) else None
    fields = {"engine_source_hash": built, "rust_sources_hash": shipped, "engine_matches_sources": match}
    notes = [f"   engine built from Rust sources {built or '(unknown — built before SOURCE_HASH)'} · "
             f"this release's Rust sources {shipped or '(not shipped)'}"
             + (" · match" if match else " · DIFFERENT" if match is False else "")]
    probs: list[str] = []
    mode = args.engine_check
    allowed = args.allow_engine_mismatch
    if match is False and mode != "info":
        if allowed:
            notes.append("!! ALLOW_ENGINE_MISMATCH=1: using an engine built from OTHER Rust sources than "
                         "this release's")
        else:
            probs.append("the engine was built from OTHER Rust sources than this release's — the engine "
                         "deals, applies the betting rules and pays out every pot. Build it from these "
                         "sources (drop SKIP_ENGINE=1, or ship CI's wheel of THIS commit), or "
                         "ALLOW_ENGINE_MISMATCH=1 if you are sure the difference does not matter")
    elif match is None and mode == "strict":
        why = ("this release ships no rust_engine/" if not shipped
               else "the engine predates SOURCE_HASH" if eng is not None else "the engine did not import")
        if allowed:
            notes.append(f"!! ALLOW_ENGINE_MISMATCH=1: nothing checked the engine against the Rust sources ({why})")
        else:
            probs.append(f"nothing can check that the engine matches this release's Rust sources ({why}). "
                         "Build it from these sources (drop SKIP_ENGINE=1), or ALLOW_ENGINE_MISMATCH=1")
    elif match is None and mode == "lenient" and built is None:
        notes.append("   (this version's engine predates SOURCE_HASH: it cannot be matched to its Rust "
                     "sources — it is the engine that ran with this code)")
    return fields, notes, probs


def run_preflight(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    work = Path(args.work).resolve()
    env = _read_env_json(args.env_json)
    for item in args.env_set or []:
        k, sep, v = item.partition("=")
        if not sep or not _NAME.fullmatch(k):
            print(f"!! --env-set wants NAME=value, not {item!r}")
            return 2
        env[k] = v
    # The production environment first, then the throwaway overrides: the
    # preflight must never write to the live database or start the grader.
    os.environ.update(env)
    db = work / "preflight.db"
    note = "an empty throwaway database"
    if args.db_copy_from:
        src = Path(args.db_copy_from)
        if src.exists():
            try:
                note = _copy_db(src, db)
            except (sqlite3.Error, OSError) as exc:
                print(f"   (could not copy the live database: {exc} — using an empty one)")
                if db.exists():
                    db.unlink()
    os.environ.update({
        "PLO5BP_PUBLIC": "1",
        "PLO5BP_DB": str(db),
        "PLO5BP_TRAINER_STATS": str(work / "trainer_stats.json"),
        "PLO5BP_HOMEGAME_GRADING": "0",
    })
    if args.checkpoint:
        os.environ["PLO5BP_CHECKPOINT"] = args.checkpoint
    sys.path.insert(0, str(root / "python"))
    import importlib

    plo5bp = importlib.import_module("plo5bp")
    where = Path(plo5bp.__file__).resolve()
    if root / "python" not in where.parents:
        print(f"!! imported the wrong code: {where} (expected under {root}/python)")
        return 1
    env_mod = importlib.import_module("plo5bp.env")
    engine_fields, engine_notes, engine_probs = _engine_check(root, args)
    homegame = importlib.import_module("plo5bp.ui.homegame")
    server = importlib.import_module("plo5bp.ui.server")  # the whole app, public layer included

    variant = getattr(server, "VARIANT_PLO5", "plo5_double_bomb")
    fmt = server.FORMATS.get(variant, {})
    loaded = bool(getattr(server, "MODEL_LOADED", False))
    critic = getattr(server, "MODEL_CRITIC", None) is not None
    mismatch = bool(fmt.get("obs_rev_mismatch", False))
    ckpt = server._format_ckpt_path(variant) if hasattr(server, "_format_ckpt_path") else None
    has_deck = hasattr(env_mod._RustGameState, "reset_with_deck")
    own_rev = getattr(server, "_process_obs_rev", None)  # the app's own answer, when it has one
    process_rev = own_rev() if callable(own_rev) else obs_rev_of(dict(os.environ))
    summary = {
        "python": sys.version.split()[0],
        "code": str(root),
        "database": note,
        "checkpoint": os.path.abspath(ckpt) if ckpt is not None else None,
        "model_loaded": loaded,
        "critic_loaded": critic,
        "obs_rev": fmt.get("obs_rev"),
        "process_obs_rev": process_rev,
        "obs_rev_mismatch": mismatch,
        "engine_reset_with_deck": has_deck,
        "verifiable_shuffle": bool(getattr(homegame, "FAIR_ON", False)),
        **engine_fields,
    }
    print(f"   imports ok · python {summary['python']} · tested on {note}")
    print(f"   model: {summary['checkpoint']} · loaded {loaded} · critic {critic} · "
          f"trained on obs rev {summary['obs_rev']}, served at rev {process_rev}"
          f"{' MISMATCH' if mismatch else ''}")
    print(f"   engine has reset_with_deck: {has_deck} · verifiable shuffle on: {summary['verifiable_shuffle']}")
    for line in engine_notes:
        print(line)
    print("PREFLIGHT " + json.dumps(summary, sort_keys=True))
    probs = []
    if not loaded:
        probs.append("the PLO5 model did not load — the site would serve a random, untrained network")
    if not critic and not args.allow_no_critic:
        probs.append("the critic did not load (ALLOW_NO_CRITIC=1 accepts that)")
    if mismatch:
        probs.append(f"OBS-REV MISMATCH: the model was trained on obs rev {summary['obs_rev']}, the server "
                     f"serves rev {process_rev} (PLO5BP_OBS_REV)")
    probs += engine_probs
    if not has_deck:
        print("   note: this engine cannot deal a verified shuffle (tables will say 'Unverified shuffle')")
    for p in probs:
        print(f"!! {p}")
    return 1 if probs else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # the same bytes on every locale and platform
        try:  # (newline: a Windows python would end lines with \r\n, which `read` keeps)
            stream.reconfigure(encoding="utf-8", errors="replace", newline="\n")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="deploytool", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("envfile")
    p.add_argument("path")
    p.add_argument("--unit-show", default="", help="`systemctl show -p Environment -p EnvironmentFiles` output")

    p = sub.add_parser("envset")
    p.add_argument("--file", required=True)
    p.add_argument("--set", required=True, dest="assign")
    p.add_argument("--out", required=True)

    p = sub.add_parser("paths")
    p.add_argument("--env-json", default="")
    p.add_argument("--app", required=True)
    p.add_argument("--key", default="",
                   help="db, checkpoint, checkpoint_nlh, trainer_stats or obs_rev — several comma-separated "
                        "print one per line, in that order (no key: all paths as JSON)")

    p = sub.add_parser("status")
    p.add_argument("--db", required=True)
    p.add_argument("--guard", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--health-url", default="", help="the running app's /health?deploy=1")
    p.add_argument("--app-running", default="1", choices=("0", "1"))
    p.add_argument("--quiet", action="store_true", help="print only when the guard trips")

    p = sub.add_parser("preflight")
    p.add_argument("--root", required=True, help="the code to test (a release folder)")
    p.add_argument("--work", required=True, help="a folder the service user can write")
    p.add_argument("--env-json", default="-", help="production env as JSON ('-' = stdin)")
    p.add_argument("--env-set", action="append", default=[], help="NAME=value over the production env")
    p.add_argument("--db-copy-from", default="", help="the live database (copied, never opened for writing)")
    p.add_argument("--checkpoint", default="", help="serve this PLO5 checkpoint instead")
    p.add_argument("--allow-no-critic", action="store_true")
    p.add_argument("--engine-check", default="strict", choices=("strict", "lenient", "info"),
                   help="strict: refuse an engine not provably built from this release's Rust sources; "
                        "lenient: refuse only a proven mismatch; info: report only")
    p.add_argument("--allow-engine-mismatch", action="store_true")

    p = sub.add_parser("health")
    p.add_argument("--url", default="http://127.0.0.1:8770/health")
    p.add_argument("--timeout", type=float, default=150.0, help="seconds to keep polling")
    p.add_argument("--interval", type=float, default=3.0)
    p.add_argument("--progress", type=float, default=0.0, help="print a waiting line every N seconds")
    p.add_argument("--expect-commit", default="")
    p.add_argument("--expect-obs-rev", type=int, default=None)
    p.add_argument("--allow-no-critic", action="store_true")
    p.add_argument("--once", action="store_true", help="one request, print it, no polling")
    p.add_argument("--field", default="", help="print this /health field and exit (e.g. process_obs_rev)")

    p = sub.add_parser("unitcheck")
    p.add_argument("--show", required=True, help="`systemctl show UNIT` output")
    p.add_argument("--cat", default="", help="`systemctl cat UNIT` output")
    p.add_argument("--app", required=True)
    p.add_argument("--envfile", required=True)
    p.add_argument("--venv-python", default="")
    p.add_argument("--new-unit", default="", help="the unit file that would be installed")

    p = sub.add_parser("backup-check")
    p.add_argument("--dir", required=True)
    p.add_argument("--max-age-hours", type=float, default=26.0)
    p.add_argument("--newer-than", default="")

    p = sub.add_parser("dbcheck")
    p.add_argument("path")

    p = sub.add_parser("dbcopy")
    p.add_argument("src")
    p.add_argument("dst")

    p = sub.add_parser("deps")
    p.add_argument("--lock", required=True)

    p = sub.add_parser("lockfile")
    p.add_argument("--header", default="")

    p = sub.add_parser("deployinfo")
    p.add_argument("--out", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--dirty", default="0")
    p.add_argument("--diff-sha256", default="")
    p.add_argument("--branch", default="")
    p.add_argument("--deployed-by", default="")
    p.add_argument("--engine-so", default="")
    p.add_argument("--engine-source", default="built")
    p.add_argument("--lock-sha256", default="")
    p.add_argument("--preflight-json", default="", help="the pre-flight's PREFLIGHT line (engine hashes)")

    p = sub.add_parser("engine-hash")
    p.add_argument("--crate", required=True)

    p = sub.add_parser("jsonget")
    p.add_argument("key")

    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "envfile":
        if args.unit_show:
            show = parse_systemctl_show(Path(args.unit_show).read_text(encoding="utf-8", errors="replace"))
            env, _src, _missing = effective_env(show, Path(args.path))
        else:
            env = parse_envfile(Path(args.path).read_text(encoding="utf-8"))
        print(json.dumps(env, sort_keys=True))
        return 0
    if args.cmd == "envset":
        key, sep, value = args.assign.partition("=")
        old = Path(args.file).read_text(encoding="utf-8")
        try:
            new = envfile_set(old, key, value) if sep else None
        except ValueError as exc:
            print(f"!! {exc}")
            return 2
        if new is None:
            print("!! --set wants NAME=value")
            return 2
        want = dict(parse_envfile(old), **{key: value})
        if parse_envfile(new) != want:  # never write a file that means anything else
            print("!! the rewritten env file would change other settings — nothing written")
            return 1
        Path(args.out).write_text(new, encoding="utf-8", newline="")
        return 0
    if args.cmd == "paths":
        env = _read_env_json(args.env_json)
        paths = resolve_paths(env, Path(args.app))
        if not args.key:
            print(json.dumps(paths, sort_keys=True))
            return 0
        rev = obs_rev_of(env)
        paths["obs_rev"] = "" if rev is None else str(rev)
        keys = args.key.split(",")
        unknown = [k for k in keys if k not in paths]
        if unknown:
            print(f"!! unknown key(s): {', '.join(unknown)}")
            return 2
        print("\n".join(paths[k] for k in keys))
        return 0
    if args.cmd == "status":
        live = None
        running = args.app_running == "1"
        if args.health_url and running:
            kind, live = fetch_app_status(args.health_url)
            if kind == "down":
                live = None
        st = home_games_status(Path(args.db), live=live, app_running=running)
        probs = guard_problems(st) if args.guard else []
        if args.json:
            print(json.dumps(st, sort_keys=True))
        elif not args.quiet or probs:
            print("\n".join(format_status(st)))
        if probs:
            print("!! a restart now would interrupt the home games: " + "; ".join(probs))
            return 3
        return 0
    if args.cmd == "preflight":
        rc = run_preflight(args)
        # Leave at once: the app started background threads (clocks, graders); a
        # non-daemon one must never keep the deploy waiting at interpreter exit.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    if args.cmd == "health":
        if args.field:
            try:
                _code, body = fetch_health(args.url)
            except (urllib.error.URLError, OSError, ValueError):
                return 1
            hit, v = _find_key(body, args.field)
            if not hit or v is None:
                return 1
            print(v)
            return 0
        start = time.monotonic()
        deadline = start + max(0.0, args.timeout)
        next_note = start + args.progress if args.progress > 0 else float("inf")
        last: Any = None
        last_probs: list[str] = ["no answer from /health"]
        while True:
            try:
                code, last = fetch_health(args.url)
                last_probs = evaluate_health(last, args.expect_commit, args.allow_no_critic, code,
                                             args.expect_obs_rev)
            except (urllib.error.URLError, OSError, ValueError) as exc:
                last_probs = [f"no healthy answer from {args.url}: {exc}"]
            now = time.monotonic()
            if args.once or not last_probs or now >= deadline:
                break
            if now >= next_note:
                print(f"   … not healthy yet after {int(now - start)} s ({last_probs[0]}) — waiting "
                      f"up to {int(args.timeout)} s", flush=True)
                next_note = now + args.progress
            time.sleep(args.interval)
        if last is not None:
            print("   /health: " + summarize_health(last))
        for p in last_probs:
            print(f"!! {p}")
        return 0 if not last_probs else 1
    if args.cmd == "unitcheck":
        show = parse_systemctl_show(Path(args.show).read_text(encoding="utf-8", errors="replace"))
        cat = Path(args.cat).read_text(encoding="utf-8", errors="replace") if args.cat else ""
        env, src, missing = effective_env(show, Path(args.envfile))
        vpy = vhome = None
        if args.venv_python:
            vpy, vhome = venv_paths(Path(args.venv_python))
        new = Path(args.new_unit).read_text(encoding="utf-8") if args.new_unit else None
        info, warn, block = unit_findings(show, cat, args.app, env, src, args.envfile, missing,
                                          new_unit_text=new, venv_python=vpy, venv_home=vhome)
        for line in info:
            print(line)
        for w in warn:
            print(f"   !! {w}")
        for b in block:
            print(f"!! {b}")
        return 1 if block else 0
    if args.cmd == "backup-check":
        info, probs = backup_report(Path(args.dir), args.max_age_hours,
                                    Path(args.newer_than) if args.newer_than else None)
        for line in info:
            print(line)
        for p in probs:
            print(f"!! {p}")
        return 0 if not probs else 1
    if args.cmd == "dbcheck":
        qc = sqlite_quick_check(Path(args.path))
        print(f"   {Path(args.path).name}: {qc}")
        return 0 if qc == "ok" else 1
    if args.cmd == "dbcopy":
        try:
            note = _copy_db(Path(args.src), Path(args.dst))
        except (sqlite3.Error, OSError) as exc:
            print(f"!! could not copy {args.src}: {exc}")
            return 1
        qc = sqlite_quick_check(Path(args.dst))
        print(f"   {note.replace('the live database', Path(args.src).name)} · integrity {qc}")
        return 0 if qc == "ok" else 1
    if args.cmd == "deps":
        missing = deps_diff(Path(args.lock).read_text(encoding="utf-8"), sys.stdin.read())
        for m in missing:
            print(f"   needs {m}")
        return 0 if not missing else 1
    if args.cmd == "lockfile":
        sys.stdout.write(make_lockfile(sys.stdin.read(), args.header))
        return 0
    if args.cmd == "deployinfo":
        pf = None
        if args.preflight_json:
            try:
                pf = json.loads(args.preflight_json)
            except ValueError:
                pf = None
        info = deploy_info(args.commit, args.dirty == "1", args.diff_sha256, args.branch,
                           args.deployed_by, Path(args.engine_so) if args.engine_so else None,
                           args.engine_source, args.lock_sha256,
                           preflight=pf if isinstance(pf, dict) else None)
        Path(args.out).write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    if args.cmd == "engine-hash":
        crate = Path(args.crate)
        if not (crate / "src").is_dir():
            print(f"!! {crate} has no src/")
            return 2
        print(engine_source_hash(crate))
        return 0
    if args.cmd == "jsonget":
        try:
            data = json.loads(sys.stdin.read() or "{}")
        except ValueError:
            return 1
        v = data.get(args.key) if isinstance(data, dict) else None
        if v is None:
            return 1
        print(v)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
