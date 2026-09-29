"""ops/deploytool.py — the helper the production deploy runs on the server.

Pure functions (env files, the "not mid-hand" guard, /health evaluation, backup
checks, dependency locks) are tested directly; the end-to-end behaviour of the
deploy that uses them is in test_deploy_remote.py.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "deploytool", Path(__file__).resolve().parents[3] / "ops" / "deploytool.py")
dt = importlib.util.module_from_spec(_SPEC)
sys.modules["deploytool"] = dt
_SPEC.loader.exec_module(dt)

NOW = datetime(2026, 9, 28, 20, 0, tzinfo=timezone.utc)


def _iso(delta: timedelta) -> str:
    return (NOW - delta).isoformat(timespec="seconds")


# --- environment files --------------------------------------------------------------

def test_envfile_follows_systemd_rules():
    text = (
        "﻿# comment\n; also a comment\n\n"
        "PLAIN=1\n"
        "  SPACED =  value with spaces  \n"
        'QUOTED="a \\"b\\" c"\n'
        "SINGLE='x y'\n"
        "export EXPORTED=yes\n"
        "CONT=abc\\\ndef\n"
        "bad line\n"
        "1BAD=no\n"
        "PLAIN=2\n"
    )
    env = dt.parse_envfile(text)
    assert env == {"PLAIN": "2", "SPACED": "value with spaces", "QUOTED": 'a "b" c',
                   "SINGLE": "x y", "EXPORTED": "yes", "CONT": "abcdef"}


def test_paths_resolve_like_the_app(tmp_path):
    app = tmp_path / "app"
    p = dt.resolve_paths({}, app)
    assert p["db"] == (app / "data" / "public.db").as_posix()
    assert p["checkpoint"] == (app / "checkpoints" / "stub.pt").as_posix()
    p = dt.resolve_paths({"PLO5BP_DB": "/srv/x.db", "PLO5BP_CHECKPOINT": "checkpoints/other.pt"}, app)
    assert p["db"].endswith("/srv/x.db")
    assert p["checkpoint"] == (app / "checkpoints" / "other.pt").as_posix()


# --- the "not mid-hand" guard ---------------------------------------------------------

def _db(path: Path, games, hands):
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE homegames (id TEXT PRIMARY KEY, name TEXT, status TEXT, hand_no INTEGER);"
        "CREATE TABLE homegame_hands (game_id TEXT, hand_no INTEGER, ended_at TEXT, summary TEXT);")
    con.executemany("INSERT INTO homegames VALUES (?,?,?,?)", games)
    con.executemany("INSERT INTO homegame_hands VALUES (?,?,?,?)", hands)
    con.commit()
    con.close()
    return path


def _games_db(tmp_path, name="p.db"):
    return _db(tmp_path / name, [
        ("g1", "Dealt", "open", 5),       # hand 5 dealt, only 4 recorded
        ("g2", "Busy", "open", 7),        # last hand 2 min ago
        ("g3", "Idle", "open", 3),        # last hand a day ago -> nothing to protect
        ("g4", "Closed", "closed", 9),    # closed tables never count
        ("g5", "New", "open", 0),         # never dealt
        ("g6", "Voided", "open", 8),      # hand 8 dealt a day ago, cut short by a restart
    ], [
        ("g1", 4, _iso(timedelta(minutes=1)), '{"grades": []}'),
        ("g2", 7, _iso(timedelta(minutes=2)), '{"grades": null}'),   # being graded
        ("g3", 3, _iso(timedelta(days=1)), '{"grades": null}'),      # lost in an old restart
        ("g4", 9, _iso(timedelta(minutes=1)), '{"grades": [1]}'),
        ("g6", 7, _iso(timedelta(days=1)), '{"grades": []}'),
    ])


def test_status_from_the_database_when_the_app_cannot_say(tmp_path):
    """An app from before /health?deploy=1: recent hands mean a game is on; an
    unrecorded hand at a quiet table is VOID (cut short earlier) and never blocks."""
    st = dt.home_games_status(_games_db(tmp_path), now=NOW)
    assert st["source"] == "database" and st["open_tables"] == 5
    assert [t["name"] for t in st["in_progress"]] == ["Dealt"]
    assert [t["name"] for t in st["active"]] == ["Busy"]
    assert [t["name"] for t in st["void"]] == ["Voided"]
    assert st["grading"] == {"waiting": 1, "older": 1, "saved_jobs": None}
    assert len(dt.guard_problems(st)) == 2          # grading never blocks
    text = "\n".join(dt.format_status(st))
    assert "HAND IN PROGRESS" in text and "'Dealt'" in text and "game running" in text
    assert "void, not blocking" in text and "'Voided'" in text and "too old to report" in text
    assert "settles them without grades" in text     # an old app's in-memory queue


def test_status_from_the_apps_memory_is_exact(tmp_path):
    """(finding 3) The running app knows which hands are live: a recent recorded hand
    is not a game, and the unrecorded hands it does not hold are void."""
    live = {"hands_in_progress": [{"id": "g2", "hand_no": 8}],
            "games_running": [{"id": "g3", "hand_no": 3, "present": 4}]}
    st = dt.home_games_status(_games_db(tmp_path), now=NOW, live=live)
    assert st["source"] == "app"
    assert [(t["name"], t["hand_no"]) for t in st["in_progress"]] == [("Busy", 8)]
    assert [(t["name"], t["present"]) for t in st["active"]] == [("Idle", 4)]
    assert sorted(t["name"] for t in st["void"]) == ["Dealt", "Voided"]
    assert len(dt.guard_problems(st)) == 2
    text = "\n".join(dt.format_status(st))
    assert "4 players at the table" in text and "running app" in text
    quiet = dt.home_games_status(_games_db(tmp_path, "q.db"), now=NOW,
                                 live={"hands_in_progress": [], "games_running": []})
    assert dt.guard_problems(quiet) == [] and len(quiet["void"]) == 2


def test_status_when_the_app_is_stopped(tmp_path):
    st = dt.home_games_status(_games_db(tmp_path), now=NOW, app_running=False)
    assert st["source"] == "stopped" and dt.guard_problems(st) == []
    assert sorted(t["name"] for t in st["void"]) == ["Dealt", "Voided"]


def test_status_grading_is_informational_when_jobs_are_saved(tmp_path):
    db = _games_db(tmp_path)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE homegame_grade_jobs (game_id TEXT, hand_no INTEGER, job TEXT)")
    con.execute("INSERT INTO homegame_grade_jobs VALUES ('g2', 7, '{}')")
    con.commit()
    con.close()
    st = dt.home_games_status(db, now=NOW, live={"hands_in_progress": [], "games_running": []})
    assert st["grading"]["saved_jobs"] == 1 and dt.guard_problems(st) == []
    assert "they finish after a restart" in "\n".join(dt.format_status(st))


def test_fetch_app_status_kinds(monkeypatch):
    answers = {"new": (200, {"ok": True, "deploy": {"home_games": {"hands_in_progress": [], "games_running": []}}}),
               "local": (200, {"ok": True, "deploy": {"home_games": None}}),
               "old": (200, {"ok": True})}
    for key, answer in answers.items():
        monkeypatch.setattr(dt, "fetch_health", lambda url, timeout=5.0, a=answer: a)
        kind, live = dt.fetch_app_status("http://127.0.0.1:1/health?deploy=1")
        assert kind == ("old" if key == "old" else "app")
        assert (live is None) == (key == "old")

    def down(url, timeout=5.0):
        raise OSError("refused")

    monkeypatch.setattr(dt, "fetch_health", down)
    assert dt.fetch_app_status("http://x") == ("down", None)


def test_status_is_quiet_when_nobody_plays(tmp_path):
    db = _db(tmp_path / "p.db", [("g3", "Idle", "open", 3)],
             [("g3", 3, _iso(timedelta(hours=3)), '{"grades": [0.9]}')])
    st = dt.home_games_status(db, now=NOW)
    assert dt.guard_problems(st) == []


def test_status_tolerates_missing_db_and_tables(tmp_path):
    assert dt.home_games_status(tmp_path / "missing.db")["exists"] is False
    empty = tmp_path / "e.db"
    sqlite3.connect(empty).close()
    assert dt.guard_problems(dt.home_games_status(empty)) == []


def test_status_never_writes_to_the_database(tmp_path):
    db = _db(tmp_path / "p.db", [("g1", "T", "open", 1)], [])
    before = db.read_bytes()
    dt.home_games_status(db, now=NOW)
    assert db.read_bytes() == before
    assert not Path(str(db) + "-wal").exists()


def test_status_cli_guard_exit_code(tmp_path, capsys):
    # (the CLI reads the real clock)
    recent = (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat(timespec="seconds")
    db = _db(tmp_path / "p.db", [("g1", "T", "open", 2)], [("g1", 1, recent, "{}")])
    assert dt.main(["status", "--db", str(db), "--guard"]) == 3     # database fallback: a game is on
    assert "a restart now would interrupt" in capsys.readouterr().out
    assert dt.main(["status", "--db", str(db)]) == 0
    assert dt.main(["status", "--db", str(db), "--guard", "--app-running", "0"]) == 0
    capsys.readouterr()
    # --quiet prints nothing unless it trips
    old = _db(tmp_path / "q.db", [("g1", "T", "open", 1)], [("g1", 1, _iso(timedelta(days=1)), "{}")])
    assert dt.main(["status", "--db", str(old), "--guard", "--quiet"]) == 0
    assert capsys.readouterr().out == ""


# --- /health --------------------------------------------------------------------

@pytest.mark.parametrize("payload,expect", [
    ({"ok": True}, []),                                           # the app before the new fields
    ({"ok": True, "model_loaded": True, "critic_loaded": True, "obs_rev_mismatch": False}, []),
    ({"ok": False}, ["ok=false"]),
    ({"ok": True, "model_loaded": False}, ["random, untrained"]),
    ({"ok": True, "models": {"plo5": {"model_loaded": True, "obs_rev_mismatch": True}}}, ["OBS-REV"]),
    ({"ok": True, "critic_loaded": False}, ["critic"]),
    ([1, 2], ["JSON object"]),
])
def test_health_evaluation(payload, expect):
    probs = dt.evaluate_health(payload)
    assert len(probs) == len(expect)
    for p, e in zip(probs, expect):
        assert e in p


def test_health_reads_the_apps_own_verdict():
    """server.health_report: 503 + model_loaded false when broken; ok false +
    a problems list when degraded (a missing critic, an unhealthy worker)."""
    broken = {"ok": False, "status": "broken", "model_loaded": False, "critic_loaded": True,
              "obs_rev_mismatch": False, "problems": ["PLO5 model not loaded — a random placeholder is serving"],
              "build": {"commit": "abc"}}
    probs = dt.evaluate_health(broken, status=503)
    assert probs == ["the PLO5 model did NOT load — the site would serve a random, untrained network"]
    degraded = {"ok": False, "status": "degraded", "model_loaded": True, "critic_loaded": False,
                "obs_rev_mismatch": False, "problems": ["PLO5 critic not loaded — the review's true EV is off"]}
    assert len(dt.evaluate_health(degraded)) == 1
    assert dt.evaluate_health(degraded, allow_no_critic=True) == []
    worker = {"ok": False, "model_loaded": True, "critic_loaded": True, "problems": ["homegame-clock unhealthy"]}
    assert dt.evaluate_health(worker) == ["the app reports: homegame-clock unhealthy"]
    assert "HTTP 503" in dt.evaluate_health({"ok": True}, status=503)[0]
    line = dt.summarize_health({**broken, "obs_rev": 1, "uptime_s": 12})
    assert line == "broken · model NOT LOADED · critic loaded · obs rev 1 · commit abc · up 12 s"


def test_health_commit_must_match_when_expected():
    assert dt.evaluate_health({"ok": True, "commit": "abc1234"}, expect_commit="abc1234ffff") == []
    assert dt.evaluate_health({"ok": True, "build": {"commit": "abc1234ffff"}}, expect_commit="abc1234") == []
    assert "old process" in dt.evaluate_health({"ok": True, "commit": "0000000"}, expect_commit="abc1234")[0]
    # (finding 12) code deployed with a BUILD_INFO.json must SAY its commit: no commit =
    # an older process still answering (or PLO5BP_BUILD_COMMIT hiding it)
    assert "reports no commit" in dt.evaluate_health({"ok": True, "commit": None}, expect_commit="abc1234")[0]
    assert "reports no commit" in dt.evaluate_health({"ok": True}, expect_commit="abc1234")[0]
    assert dt.evaluate_health({"ok": True}) == []                    # nothing expected: an old app is fine
    assert dt.evaluate_health({"ok": True, "critic_loaded": False}, allow_no_critic=True) == []


def test_health_obs_rev_must_match_when_expected():
    assert dt.evaluate_health({"ok": True, "process_obs_rev": 2}, expect_obs_rev=2) == []
    assert "encodes obs rev 1" in dt.evaluate_health({"ok": True, "process_obs_rev": 1}, expect_obs_rev=2)[0]
    assert dt.evaluate_health({"ok": True}, expect_obs_rev=2) == []   # an app that does not report it


# --- backups --------------------------------------------------------------------------

def _sqlite(path: Path) -> Path:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (x)")
    con.execute("INSERT INTO t VALUES (1)")
    con.commit()
    con.close()
    return path


def test_backup_check_by_content_not_name(tmp_path):
    keep = tmp_path / "backups"
    keep.mkdir()
    (keep / "app-before-20260901.tgz").write_bytes(b"code copy")      # ignored
    b = _sqlite(keep / "whatever-name.bak")
    info, probs = dt.backup_report(keep, max_age_hours=26)
    assert probs == [] and "whatever-name.bak" in info[0] and "integrity: ok" in info[-1]
    # gzipped backups are recognised and checked too
    gz = keep / "sub" / "public.db.gz"
    gz.parent.mkdir()
    time.sleep(0.02)
    gz.write_bytes(gzip.compress(b.read_bytes()))
    assert dt.newest_backup(keep) == gz
    assert dt.sqlite_quick_check(gz) == "ok"


def test_backup_check_problems(tmp_path):
    keep = tmp_path / "backups"
    keep.mkdir()
    assert "no database backup" in dt.backup_report(keep, 26)[1][0]
    old = _sqlite(keep / "old.db")
    t = time.time() - 30 * 3600
    import os
    os.utime(old, (t, t))
    assert "h old" in dt.backup_report(keep, 26)[1][0]
    marker = tmp_path / "marker"
    marker.write_text("")
    # the job ran but wrote nothing new
    assert "did not write" in dt.backup_report(keep, 1, newer_than=marker)[1][0]
    # a new file in an unknown format is reported, not failed
    time.sleep(0.02)
    (keep / "public.tar.xz").write_bytes(b"\xfd7zXZ")
    info, probs = dt.backup_report(keep, 1, newer_than=marker)
    assert probs == [] and "not verified" in "\n".join(info)
    # a new SQLite file that is corrupt fails
    time.sleep(0.02)
    (keep / "public-new.db").write_bytes(b"SQLite format 3\x00" + b"\x00" * 50)
    info, probs = dt.backup_report(keep, 1, newer_than=marker)
    assert probs and "integrity" in probs[0]


def test_dbcheck_cli(tmp_path, capsys):
    good = _sqlite(tmp_path / "g.db")
    assert dt.main(["dbcheck", str(good)]) == 0
    bad = tmp_path / "b.db"
    bad.write_bytes(b"nope")
    assert dt.main(["dbcheck", str(bad)]) == 1


# --- dependency locks -----------------------------------------------------------------------

FREEZE = """\
Authlib==1.3.2
fastapi==0.115.0
numpy==2.1.1
pip==24.0
-e git+https://github.com/Garcey/plodbnet@abc#egg=plo5bp
plo5bp @ file:///opt/wrapgto/app
setuptools==70.0
torch==2.4.1+cpu
typing_extensions==4.12.2
"""


def test_lockfile_from_freeze():
    lock = dt.make_lockfile(FREEZE, header="line one\nline two")
    lines = lock.splitlines()
    assert lines[:2] == ["# line one", "# line two"]
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in lines
    pins = [ln for ln in lines if "==" in ln]
    assert pins == ["Authlib==1.3.2", "fastapi==0.115.0", "numpy==2.1.1", "torch==2.4.1+cpu",
                    "typing_extensions==4.12.2"]
    assert not any("plo5bp" in ln or ln.startswith(("pip==", "setuptools==")) for ln in lines)


def test_deps_diff():
    lock = dt.make_lockfile(FREEZE)
    assert dt.deps_diff(lock, FREEZE) == []
    assert dt.deps_diff(lock, FREEZE.replace("numpy==2.1.1", "numpy==2.0.0")) == ["numpy==2.1.1 (installed: 2.0.0)"]
    assert dt.deps_diff(lock + "stripe==11.1.0\n", FREEZE) == ["stripe==11.1.0 (not installed)"]
    # names compare case- and separator-insensitively, pip's own tools never count
    assert dt.deps_diff("Typing-Extensions==4.12.2\npip==99\n", FREEZE) == []


def test_parse_pins_ignores_options_markers_and_hashes():
    pins = dt.parse_pins(
        "--extra-index-url https://x\n# c\nfoo==1.0 ; python_version >= '3.8'\n"
        "bar[extra]==2.0 \\\n    --hash=sha256:abc\nbaz>=3\n")
    assert pins == {"foo": "1.0", "bar": "2.0"}


# --- BUILD_INFO.json ------------------------------------------------------------------------------

def test_deployinfo_records_commit_and_engine(tmp_path):
    so = tmp_path / "_engine.so"
    so.write_bytes(b"engine")
    out = tmp_path / "BUILD_INFO.json"
    assert dt.main(["deployinfo", "--out", str(out), "--commit", "0123456789abcdef", "--dirty", "1",
                    "--diff-sha256", "d" * 64, "--branch", "main", "--deployed-by", "laptop",
                    "--engine-so", str(so), "--engine-source", "built"]) == 0
    info = json.loads(out.read_text())
    assert info["short"] == "0123456" and info["dirty"] is True and info["branch"] == "main"
    assert info["engine"]["sha256"] == dt.sha256_file(so) and info["engine"]["source"] == "built"
    datetime.fromisoformat(info["deployed_at"])


# --- preflight (imports "the app" the way the server will) -------------------------------------------

def test_preflight_env_production_values_but_never_live_state(tmp_path, monkeypatch):
    """The production env reaches the app; the database, trainer stats and grader
    are always overridden. (The import of a whole fake app — model / critic / obs
    rev checks — is exercised end to end in test_deploy_remote.py.)"""
    import importlib
    import io
    import os

    work = tmp_path / "work"
    work.mkdir()
    env = {"PLO5BP_DB": "/live/public.db", "PLO5BP_HOMEGAME_GRADING": "1", "PLO5BP_OBS_REV": "1"}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(env)))
    for k in (*env, "PLO5BP_PUBLIC", "PLO5BP_TRAINER_STATS", "PLO5BP_CHECKPOINT"):
        monkeypatch.delenv(k, raising=False)
    seen = {}

    def stop_at_import(name):
        seen.update({k: os.environ.get(k) for k in (*env, "PLO5BP_CHECKPOINT")})
        raise ImportError("stop before importing anything")

    monkeypatch.setattr(importlib, "import_module", stop_at_import)
    monkeypatch.setattr(sys, "path", list(sys.path))
    saved = dict(os.environ)  # the preflight writes os.environ; never leak it into other tests
    try:
        with pytest.raises(ImportError):
            dt.main(["preflight", "--root", str(tmp_path / "release"), "--work", str(work),
                     "--checkpoint", "/tmp/candidate.pt"])
    finally:
        os.environ.clear()
        os.environ.update(saved)
    assert seen["PLO5BP_OBS_REV"] == "1"                                # production value kept
    assert seen["PLO5BP_DB"] == str(work.resolve() / "preflight.db")   # never the live DB
    assert seen["PLO5BP_HOMEGAME_GRADING"] == "0"                       # never grades
    assert seen["PLO5BP_CHECKPOINT"] == "/tmp/candidate.pt"


def test_preflight_copies_the_live_database(tmp_path):
    live = _sqlite(tmp_path / "live.db")
    dst = tmp_path / "copy.db"
    note = dt._copy_db(live, dst)
    assert "copy of the live database" in note
    assert dt.sqlite_quick_check(dst) == "ok"
    shutil.rmtree(tmp_path / "nothing", ignore_errors=True)


def test_preflight_env_set_overrides_the_production_value(tmp_path, monkeypatch):
    """promote OBS_REV=N: the candidate is checked on the revision it will be served at."""
    import importlib
    import io
    import os

    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"PLO5BP_OBS_REV": "1"})))
    seen = {}

    def stop_at_import(name):
        seen["rev"] = os.environ.get("PLO5BP_OBS_REV")
        raise ImportError("stop")

    monkeypatch.setattr(importlib, "import_module", stop_at_import)
    monkeypatch.setattr(sys, "path", list(sys.path))
    saved = dict(os.environ)
    try:
        with pytest.raises(ImportError):
            dt.main(["preflight", "--root", str(tmp_path), "--work", str(tmp_path),
                     "--env-set", "PLO5BP_OBS_REV=2"])
    finally:
        os.environ.clear()
        os.environ.update(saved)
    assert seen["rev"] == "2"


# --- the service's settings (findings 4, 6) --------------------------------------------------------

def test_envfile_set_replaces_only_that_setting():
    text = "# settings\r\nPLO5BP_PUBLIC=1\r\nPLO5BP_OBS_REV=1\r\nexport PLO5BP_OBS_REV = 1\r\nLONG=a\\\nb\r\nZ=9"
    out = dt.envfile_set(text, "PLO5BP_OBS_REV", "2")
    # every other line byte for byte (CRLF, the continuation); the last line gets its newline
    assert out == "# settings\r\nPLO5BP_PUBLIC=1\r\nLONG=a\\\nb\r\nZ=9\nPLO5BP_OBS_REV=2\n"
    assert dt.parse_envfile(out) == {"PLO5BP_PUBLIC": "1", "LONG": "ab", "Z": "9", "PLO5BP_OBS_REV": "2"}
    assert dt.envfile_set("", "A", "1") == "A=1\n"
    assert dt.envfile_set("# PLO5BP_OBS_REV=1\n", "PLO5BP_OBS_REV", "2") == "# PLO5BP_OBS_REV=1\nPLO5BP_OBS_REV=2\n"
    for bad in ("two words", "", "a#b", "$(x)", 'q"'):
        with pytest.raises(ValueError):
            dt.envfile_set("A=1\n", "A", bad)


def test_envset_cli_writes_a_new_file_and_leaves_the_old(tmp_path):
    src = tmp_path / "env"
    src.write_text("PLO5BP_OBS_REV=1\nPLO5BP_PUBLIC=1")
    out = tmp_path / "env.new"
    assert dt.main(["envset", "--file", str(src), "--set", "PLO5BP_OBS_REV=2", "--out", str(out)]) == 0
    assert dt.parse_envfile(out.read_text()) == {"PLO5BP_OBS_REV": "2", "PLO5BP_PUBLIC": "1"}
    assert src.read_text() == "PLO5BP_OBS_REV=1\nPLO5BP_PUBLIC=1"
    assert dt.main(["envset", "--file", str(src), "--set", "PLO5BP_OBS_REV=two words", "--out", str(out)]) == 2


def test_the_services_effective_environment(tmp_path):
    """(finding 6) What systemd gives the process: Environment= first, then each
    EnvironmentFile= in order — the pre-flight must see exactly that."""
    a = tmp_path / "a.env"
    a.write_text("SHARED=from-a\nONLY_A=1\n")
    b = tmp_path / "b.env"
    b.write_text("SHARED=from-b\n")
    show = dt.parse_systemctl_show(
        'Environment=PYTHONUNBUFFERED=1 "SPACED=x y" SHARED=from-unit UNIT_ONLY=1\n'
        f"EnvironmentFiles={a.as_posix()} (ignore_errors=no)\n"
        f"EnvironmentFiles={b.as_posix()} (ignore_errors=yes)\n"
        f"EnvironmentFiles={(tmp_path / 'missing').as_posix()} (ignore_errors=no)\n"
        f"EnvironmentFiles={(tmp_path / 'optional').as_posix()} (ignore_errors=yes)\n")
    env, src, missing = dt.effective_env(show, tmp_path / "fallback")
    assert env == {"PYTHONUNBUFFERED": "1", "SPACED": "x y", "SHARED": "from-b", "UNIT_ONLY": "1", "ONLY_A": "1"}
    assert src["SHARED"] == b.as_posix() and src["UNIT_ONLY"] == "unit" and src["ONLY_A"] == a.as_posix()
    assert missing == [(tmp_path / "missing").as_posix()]
    # no unit information (not installed yet): the env file alone
    fb = tmp_path / "fallback"
    fb.write_text("X=1\n")
    assert dt.effective_env({}, fb)[0] == {"X": "1"}


def test_paths_several_keys_and_the_obs_rev(tmp_path, capsys):
    env = tmp_path / "env.json"
    env.write_text(json.dumps({"PLO5BP_OBS_REV": " 1 "}))
    assert dt.main(["paths", "--env-json", str(env), "--app", "/app", "--key", "db,obs_rev"]) == 0
    db, rev = capsys.readouterr().out.splitlines()
    assert db.endswith("/app/data/public.db") and rev == "1"
    assert dt.obs_rev_of({}) == 2 and dt.obs_rev_of({"PLO5BP_OBS_REV": ""}) == 2
    assert dt.obs_rev_of({"PLO5BP_OBS_REV": "3"}) is None
    assert dt.main(["paths", "--env-json", str(env), "--app", "/app", "--key", "nope"]) == 2


NEW_UNIT = """\
[Service]
User=wrapgto
WorkingDirectory=/opt/wrapgto/app
EnvironmentFile=/etc/wrapgto/env
Environment=PYTHONUNBUFFERED=1 HOME=/var/lib/wrapgto
ExecStart=/opt/wrapgto/app/.venv/bin/python -m uvicorn plo5bp.ui.server:app
ProtectSystem=strict
ReadWritePaths=/opt/wrapgto/app/data /opt/wrapgto/app/checkpoints
ProtectHome=yes
"""


def _findings(env=None, sources=None, show=None, new=NEW_UNIT, venv_python="/usr/bin/python3.11",
              venv_home="/usr/bin", missing=()):
    show = show if show is not None else {"FragmentPath": ["/etc/systemd/system/wrapgto.service"],
                                          "User": ["wrapgto"], "WorkingDirectory": ["/opt/wrapgto/app"],
                                          "ExecStart": ["{ path=/x ; argv[]=/x -m uvicorn app --workers 2 ; }"]}
    env = env if env is not None else {"PLO5BP_PUBLIC": "1"}
    sources = sources if sources is not None else {k: "/etc/wrapgto/env" for k in env}
    return dt.unit_findings(show, "", "/opt/wrapgto/app", env, sources, "/etc/wrapgto/env", list(missing),
                            new_unit_text=new, venv_python=venv_python, venv_home=venv_home,
                            exists=lambda p: True)


def test_unit_check_passes_the_normal_case_and_reports_what_runs():
    info, warn, block = _findings()
    assert block == []
    assert any("--workers" in w for w in warn) and any("timeout-graceful-shutdown" in w for w in warn)
    assert any("runs as wrapgto" in i for i in info)


def test_unit_check_blocks_what_would_break_the_site():
    # settings kept only in the old unit would be dropped
    _i, _w, block = _findings(env={"PLO5BP_PUBLIC": "1", "PLO5BP_OBS_REV": "1"},
                              sources={"PLO5BP_PUBLIC": "/etc/wrapgto/env", "PLO5BP_OBS_REV": "unit"})
    assert any("would be LOST" in b and "PLO5BP_OBS_REV" in b for b in block)
    # the new unit's own Environment= keeps its names
    _i, _w, block = _findings(env={"HOME": "/x"}, sources={"HOME": "unit"})
    assert block == []
    # ProtectHome=yes hides a venv living under /root or /home
    _i, _w, block = _findings(venv_python="/root/.pyenv/versions/3.11/bin/python3.11")
    assert any("ProtectHome" in b for b in block)
    _i, _w, block = _findings(venv_home="/home/deploy/python/bin")
    assert any("ProtectHome" in b for b in block)
    # a setting pointing where the hardened unit cannot write
    _i, _w, block = _findings(env={"PLO5BP_DB": "/srv/wrapgto/public.db"},
                              sources={"PLO5BP_DB": "/etc/wrapgto/env"})
    assert any("PLO5BP_DB" in b and "read-only" in b for b in block)
    _i, _w, block = _findings(env={"PLO5BP_TRAINER_STATS": "/var/tmp/stats.json"},
                              sources={"PLO5BP_TRAINER_STATS": "/etc/wrapgto/env"})
    assert any("PLO5BP_TRAINER_STATS" in b for b in block)
    # drop-ins keep overriding whatever is installed
    show = {"FragmentPath": ["/etc/systemd/system/wrapgto.service"], "User": ["wrapgto"],
            "DropInPaths": ["/etc/systemd/system/wrapgto.service.d/override.conf"]}
    _i, _w, block = _findings(show=show)
    assert any("drop-ins" in b and "override.conf" in b for b in block)


def test_unit_check_warns_without_a_new_unit():
    show = {"DropInPaths": ["/etc/systemd/system/wrapgto.service.d/x.conf"]}
    info, warn, block = dt.unit_findings(show, "", "/opt/wrapgto/app",
                                         {"PLO5BP_OBS_REV": "1", "PLO5BP_BUILD_COMMIT": "x"},
                                         {"PLO5BP_OBS_REV": "unit", "PLO5BP_BUILD_COMMIT": "/etc/wrapgto/env"},
                                         "/etc/wrapgto/env", ["/etc/wrapgto/other"])
    assert block == []
    text = "\n".join(warn)
    assert "PLO5BP_OBS_REV" in text and "Environment=" in text
    assert "PLO5BP_BUILD_COMMIT" in text and "drop-ins" in text and "/etc/wrapgto/other" in text


def test_venv_paths_follow_symlinks(tmp_path):
    venv = tmp_path / "v"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\ninclude-system-site-packages = false\n")
    (venv / "bin" / "python").write_text("")
    real, home = dt.venv_paths(venv / "bin" / "python")
    assert home == "/usr/bin" and Path(real).name == "python"


# --- the engine vs its sources (finding 5) --------------------------------------------------------

def test_engine_source_hash_is_the_build_rs_rule(tmp_path):
    """Three copies of one rule: rust_engine/build.rs (SOURCE_HASH), tests/conftest.py
    and this. They must agree, or a deploy refuses every engine."""
    spec = importlib.util.spec_from_file_location(
        "root_conftest_for_hash", Path(__file__).resolve().parents[2] / "conftest.py")
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)
    crate = tmp_path / "rust_engine"
    (crate / "src" / "a").mkdir(parents=True)
    (crate / "src" / "lib.rs").write_bytes(b"fn x() {}\r\n")
    (crate / "src" / "a" / "b.rs").write_bytes(b"// b\n")
    (crate / "Cargo.toml").write_bytes(b"[package]\n")
    (tmp_path / "Cargo.lock").write_bytes(b"# lock\n")
    assert dt.engine_source_hash(crate) == conf.engine_source_hash(crate)
    real = Path(__file__).resolve().parents[3] / "rust_engine"
    assert dt.engine_source_hash(real) == conf.engine_source_hash(real)
    try:
        import plo5bp._engine as eng
    except ImportError:
        return
    built = getattr(eng, "SOURCE_HASH", None)
    if built and conf.engine_staleness() is None:
        assert dt.engine_source_hash(real) == built


def test_deployinfo_records_both_engine_hashes(tmp_path):
    so = tmp_path / "_engine.so"
    so.write_bytes(b"engine")
    out = tmp_path / "BUILD_INFO.json"
    pf = json.dumps({"engine_source_hash": "aaaa", "rust_sources_hash": "bbbb", "engine_matches_sources": False})
    assert dt.main(["deployinfo", "--out", str(out), "--commit", "0123456789", "--engine-so", str(so),
                    "--engine-source", "reused", "--preflight-json", pf]) == 0
    eng = json.loads(out.read_text())["engine"]
    assert eng == {"source": "reused", "file": "_engine.so", "sha256": dt.sha256_file(so),
                   "source_hash": "aaaa", "rust_sources_hash": "bbbb", "matches_sources": False}


def test_jsonget_and_engine_hash_cli(tmp_path, capsys, monkeypatch):
    import io
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"a": 1, "b": null}'))
    assert dt.main(["jsonget", "a"]) == 0 and capsys.readouterr().out.strip() == "1"
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"a": 1, "b": null}'))
    assert dt.main(["jsonget", "b"]) == 1
    assert dt.main(["engine-hash", "--crate", str(tmp_path)]) == 2   # no src/


def test_dbcopy_cli(tmp_path):
    live = _sqlite(tmp_path / "live.db")
    assert dt.main(["dbcopy", str(live), str(tmp_path / "copy.db")]) == 0
    assert dt.sqlite_quick_check(tmp_path / "copy.db") == "ok"
    assert dt.main(["dbcopy", str(tmp_path / "nope.db"), str(tmp_path / "c2.db")]) == 1
