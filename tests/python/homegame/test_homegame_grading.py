"""Home games — the end of a hand and its grades survive failures (improvements
backlog, 2026-09-28): stacks and the hand's record are one commit (OPS-006), a
shuffle transcript that failed to save is retried (OPS-013), grading jobs are
saved until graded and a replay that keeps failing is settled (OPS-016 /
TEST-007), /show and the grader never erase each other's change (OPS-007),
another player's marks follow the table's "show grades" switch (SEC-004), the
lifetime history pages cleanly (PERF-011), and the workers' health and shutdown
(OPS-008 / OPS-011).

Booted once for the module in PUBLIC mode with grading ON against a temp DB.
"""

from __future__ import annotations

import json
import sys
import threading
import time

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "themilesgarcia@icloud.com"
NAMES = ["gia", "hux", "ima", "jed"]


@pytest.fixture(scope="module")
def server(boot_public_server):
    srv = boot_public_server(PLO5BP_HOMEGAME_GRADING="1", PLO5BP_ADMIN_EMAILS=ADMIN_EMAIL)
    # No checkpoint in the tests: the site serves a random placeholder, which
    # never grades in production (OPS-021). These tests grade with it anyway.
    sys.modules["plo5bp.ui.homegame"].set_model_provider(lambda: srv.MODEL)
    return srv


@pytest.fixture(scope="module")
def hg(server):
    return sys.modules["plo5bp.ui.homegame"]


@pytest.fixture(scope="module")
def cast(server):
    def login(email):
        cl = TestClient(server.app, raise_server_exceptions=False)
        assert cl.get("/auth/dev", params={"email": email}).status_code == 200
        return cl

    adm = login(ADMIN_EMAIL)
    players = [login(f"{n}@example.com") for n in NAMES]
    ids = {u["email"]: u["id"] for u in adm.get("/admin/api/users").json()["users"]}
    for n in NAMES:
        adm.post("/admin/api/games_access", json={"user_id": ids[f"{n}@example.com"], "action": "grant"})
    return {"p": players, "adm": adm, "srv": server,
            "uid": {n: ids[f"{n}@example.com"] for n in NAMES},
            "by_uid": {ids[f"{n}@example.com"]: players[i] for i, n in enumerate(NAMES)}}


def _post(cl, gid, what, body=None):
    return cl.post(f"/games/api/tables/{gid}/{what}", json=body or {})


def _state(cl, gid):
    return cl.get(f"/games/api/tables/{gid}").json()


def _table(cast, n, **kw):
    body = {"name": "grading", "sb_cents": 50, "bb_cents": 100, "ante_cents": 1000,
            "default_buyin_cents": 20000, **kw}
    gid = cast["p"][0].post("/games/api/tables", json=body).json()["id"]
    for i in range(1, n):
        assert _post(cast["p"][i], gid, "sit", {"seat": i, "buyin_cents": 20000}).status_code == 200
    return gid


def _actor(cast, gid):
    s = _state(cast["p"][0], gid)
    if s["phase"] != "in_hand" or s["actor"] is None:
        return None, s
    cl = cast["by_uid"][s["seats"][s["actor"]]["user_id"]]
    return cl, _state(cl, gid)


def _bet_and_fold_out(cast, gid):
    cl, s = _actor(cast, gid)
    assert _post(cl, gid, "act", {"gate": "raise", "raise_to_chips":
                                  s["raise_bounds"]["min_chips"] + s["street_commit_chips"]}).status_code == 200
    for _ in range(12):
        cl, s = _actor(cast, gid)
        if cl is None:
            break
        _post(cl, gid, "act", {"gate": "fold" if s["legal"]["fold"] else "check_call"})


def _check_down(cast, gid):
    for _ in range(40):
        cl, s = _actor(cast, gid)
        if cl is None:
            return
        _post(cl, gid, "act", {"gate": "check_call"})


def _skip_runout(hg, gid):
    t = hg.HUB.get(gid)
    with t.lock:
        if t.runout_active and t.runout_started_mono is not None:
            t.runout_started_mono -= 600.0


def _stored(hg, gid, hand_no):
    row = hg.pub.DB.one("SELECT summary FROM homegame_hands WHERE game_id=? AND hand_no=?", (gid, hand_no))
    return json.loads(row["summary"]) if row is not None else None


def _db_stacks(hg, gid):
    return {int(r["seat"]): int(r["stack_chips"]) for r in hg.pub.DB.q(
        "SELECT seat, stack_chips FROM homegame_players WHERE game_id=? AND seat IS NOT NULL", (gid,))}


# --- OPS-006 / OPS-013: one commit for the end of a hand; owed writes are retried ------------


def test_a_hand_and_its_stacks_are_saved_together_or_retried_together(cast, hg, monkeypatch):
    p = cast["p"]
    gid = _table(cast, 2)
    _post(p[0], gid, "run", {"running": True})
    t = hg.HUB.get(gid)
    real = hg._persist_player

    def disk_full(*a, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(hg, "_persist_player", disk_full)
    _bet_and_fold_out(cast, gid)  # the hand ends while nothing can be saved
    with t.lock:
        assert t.phase == "showdown" and t.persist_dirty
        assert [w[0] for w in t.unsaved] == ["hand"]
        mem = {i: q.stack_chips for i, q in enumerate(t.seats) if q is not None}
    # the database never holds the result without the stacks it moved
    assert _stored(hg, gid, 1) is None
    assert _db_stacks(hg, gid) != mem
    monkeypatch.setattr(hg, "_persist_player", real)
    with t.lock:
        t.persist_retry_mono = 0.0
        hg._retry_persist_locked(t)
        assert not t.persist_dirty and t.unsaved == []
    assert _stored(hg, gid, 1)["hand_no"] == 1 and _stored(hg, gid, 1)["v"] == hg.HAND_RECORD_VERSION
    assert _db_stacks(hg, gid) == mem
    res = hg.pub.DB.q("SELECT delta_cents, delta_chips FROM homegame_hand_results WHERE game_id=?", (gid,))
    assert sum(int(r["delta_chips"]) for r in res) == 0 and len(res) == 2  # exact chips (OPS-015)


def test_a_record_that_cannot_be_written_never_blocks_the_stacks(cast, hg, monkeypatch):
    p = cast["p"]
    gid = _table(cast, 2)
    _post(p[0], gid, "run", {"running": True})
    t = hg.HUB.get(gid)
    real = hg._write_owed

    def broken(tt, w):
        if w[0] == "hand":
            raise ValueError("this record cannot be stored")
        real(tt, w)

    monkeypatch.setattr(hg, "_write_owed", broken)
    _bet_and_fold_out(cast, gid)
    with t.lock:
        assert not t.persist_dirty and t.unsaved == []
        mem = {i: q.stack_chips for i, q in enumerate(t.seats) if q is not None}
    assert _db_stacks(hg, gid) == mem, "the stacks were saved"
    assert _stored(hg, gid, 1) is None, "the record alone was dropped (and logged)"


def test_a_shuffle_transcript_that_failed_to_save_is_retried(cast, hg, monkeypatch):
    """OPS-013: it used to be only logged — proofs 404'd while the hand record
    still named the shuffle."""
    if not hg.FAIR_ON:
        pytest.skip("engine without reset_with_deck")
    p = cast["p"]
    gid = _table(cast, 2)
    real = hg._persist_player
    monkeypatch.setattr(hg, "_persist_player", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("io")))
    _post(p[0], gid, "run", {"running": True})
    t = hg.HUB.get(gid)
    with t.lock:
        assert t.phase == "in_hand" and [w[0] for w in t.unsaved] == ["fair"]
    fair_row = lambda: hg.pub.DB.one("SELECT 1 FROM homegame_fair WHERE game_id=? AND hand_no=1", (gid,))  # noqa: E731
    assert fair_row() is None
    monkeypatch.setattr(hg, "_persist_player", real)
    with t.lock:
        t.persist_retry_mono = 0.0
        hg._retry_persist_locked(t)
        assert t.unsaved == []
    assert fair_row() is not None, "saved with the next successful write"
    r = p[0].get(f"/games/api/tables/{gid}/fair/1")
    assert r.status_code == 200


# --- OPS-016 / TEST-007: grading jobs are saved until graded ----------------------------------


def test_a_hands_grading_job_waits_for_a_model_and_survives_a_restart(cast, hg, monkeypatch):
    p = cast["p"]
    srv = cast["srv"]
    hg.set_model_provider(lambda: None)  # no real model served (OPS-021)
    try:
        gid = _table(cast, 2)
        _post(p[0], gid, "run", {"running": True})
        _bet_and_fold_out(cast, gid)
        assert hg.wait_for_grading(30.0)
        assert _stored(hg, gid, 1)["grades"] is None, "still being worked out: no model yet"
        row = hg.pub.DB.one("SELECT job FROM homegame_grade_jobs WHERE game_id=? AND hand_no=1", (gid,))
        job = json.loads(row["job"])
        assert job["deck"] and "seed" not in job, "a saved job carries the deck, never a seed"
        # "a restart": the in-memory queue is gone; the saved job is not
        with hg._GRADE_LOCK:
            hg._GRADE_QUEUED.clear()
    finally:
        hg.set_model_provider(lambda: srv.MODEL)
    assert hg._sweep_grade_jobs() >= 1
    assert hg.wait_for_grading(30.0)
    rec = _stored(hg, gid, 1)
    assert rec["grades"] and "grades_note" not in rec
    assert hg.pub.DB.one("SELECT 1 FROM homegame_grade_jobs WHERE game_id=? AND hand_no=1", (gid,)) is None


def test_a_replay_that_keeps_failing_is_settled_with_a_note(cast, hg, monkeypatch):
    p = cast["p"]
    calls = []

    def diverges(job, model=None):
        calls.append(job["hand_no"])
        raise RuntimeError("replay diverged at action 0")

    monkeypatch.setattr(hg, "grade_hand", diverges)
    gid = _table(cast, 2)
    _post(p[0], gid, "run", {"running": True})
    _bet_and_fold_out(cast, gid)
    for _ in range(hg.GRADE_MAX_ATTEMPTS + 2):
        assert hg.wait_for_grading(30.0)
        if hg.pub.DB.one("SELECT 1 FROM homegame_grade_jobs WHERE game_id=?", (gid,)) is None:
            break
        hg._sweep_grade_jobs()
    assert len(calls) == hg.GRADE_MAX_ATTEMPTS, calls
    rec = _stored(hg, gid, 1)
    assert rec["grades"] == [] and rec["grades_note"] == "could not be graded"


def test_hands_the_old_queue_lost_are_settled_once(hg):
    """Schema step 6: a hand still "being worked out" when the new code first
    starts was in a queue that died with the old process."""
    hg.pub.DB.q("INSERT INTO homegame_hands(game_id,hand_no,ended_at,pot_cents,summary) VALUES(?,?,?,?,?)",
                ("lostgrades", 1, "2026-09-27T00:00:00+00:00", 0,
                 json.dumps({"hand_no": 1, "seats": [], "grades": None})))
    with hg.pub.DB.transaction():
        hg._settle_lost_grades(hg.pub.DB._conn)
    rec = _stored(hg, "lostgrades", 1)
    assert rec["grades"] == [] and "restarted" in rec["grades_note"]


# --- OPS-007: /show and the grader edit the stored hand atomically ----------------------------


def test_show_and_the_grader_never_erase_each_others_change(cast, hg):
    p = cast["p"]
    gid = _table(cast, 2)
    _post(p[0], gid, "run", {"running": True})
    _bet_and_fold_out(cast, gid)
    assert hg.wait_for_grading(30.0)
    holding, done = threading.Event(), threading.Event()

    def a_slow_show():
        with hg._edit_hand_record(gid, 1) as rec:
            rec["seats"][0]["shown"] = True
            holding.set()
            time.sleep(0.3)  # the grader lands in the middle of it

    th = threading.Thread(target=a_slow_show)
    th.start()
    assert holding.wait(5)
    g = threading.Thread(target=lambda: (hg._store_grades(
        {"game_id": gid, "hand_no": 1, "uid_of": {}}, [{"i": 0, "seat": 0, "score": 50.0, "cat": "correct"}]),
        done.set()))
    g.start()
    th.join(5)
    g.join(5)
    assert done.is_set()
    rec = _stored(hg, gid, 1)
    assert rec["seats"][0]["shown"] is True and rec["grades"][0]["score"] == 50.0


# --- SEC-004: someone else's marks follow the table's "show grades" switch --------------------


@pytest.mark.parametrize("show_grades", [True, False])
def test_another_players_marks_follow_the_tables_switch(cast, hg, show_grades):
    p = cast["p"]
    gid = _table(cast, 2)
    _post(p[0], gid, "settings", {"show_grades": show_grades})
    _post(p[0], gid, "run", {"running": True})
    _check_down(cast, gid)  # a showdown: both hands tabled
    _skip_runout(hg, gid)
    _state(p[0], gid)
    assert hg.wait_for_grading(30.0)
    hux = cast["uid"]["hux"]
    mine = p[1].get("/games/api/my/hands", params={"game": gid}).json()["hands"]
    theirs = p[0].get(f"/games/api/players/{hux}/hands", params={"game": gid}).json()["hands"]
    assert mine[0]["accuracy"] is not None, "your own marks: always"
    assert (theirs[0]["accuracy"] is not None) is show_grades
    # sorting by accuracy never ranks by the hidden marks either
    srt = p[0].get(f"/games/api/players/{hux}/hands", params={"game": gid, "sort": "accuracy"}).json()
    assert [h["hand_no"] for h in srt["hands"]] == [1]


def test_a_shown_hand_is_marked_in_the_results(cast, hg):
    p = cast["p"]
    gid = _table(cast, 2)
    _post(p[0], gid, "run", {"running": True})
    _bet_and_fold_out(cast, gid)
    s = _state(p[0], gid)
    who = next(i for i, x in enumerate(s["seats"]) if x["in_hand"] and not x["folded"])
    cl = p[who]
    assert _post(cl, gid, "show").status_code == 200
    uid = cast["uid"][NAMES[who]]
    row = hg.pub.DB.one("SELECT shown, showdown FROM homegame_hand_results WHERE game_id=? AND user_id=?",
                        (gid, uid))
    assert (row["shown"], row["showdown"]) == (1, 0), "shown by choice — not counted as a showdown"


# --- PERF-011: lifetime history pages neither skip nor repeat -----------------------------------


def test_history_pages_cover_every_hand_exactly_once(cast, hg):
    p = cast["p"]
    gid = _table(cast, 2)
    _post(p[0], gid, "run", {"running": True})
    for _ in range(5):  # quick fold-outs: several hands end within the same second
        _bet_and_fold_out(cast, gid)
        _skip_runout(hg, gid)
        s = _state(p[0], gid)
        r = _post(p[0], gid, "deal", {"hand_no": s["hand_no"]})
        assert r.status_code in (200, 400), r.text
    _post(p[0], gid, "run", {"running": False})
    for sort in ("time", "pot", "net"):
        seen = []
        for k in range(40):
            page = p[0].get("/games/api/my/hands", params={"game": gid, "sort": sort, "limit": 1, "offset": k}).json()
            if not page["hands"]:
                break
            seen.append(page["hands"][0]["hand_no"])
        total = page["total"]
        assert len(seen) == total and len(set(seen)) == total, (sort, seen, total)


# --- OPS-011: health; OPS-008: the shutdown hook ------------------------------------------------


def test_the_health_signal_is_for_admins_and_a_dead_clock_is_restarted(cast, hg):
    assert cast["p"][0].get("/games/api/health").status_code == 404
    h = cast["adm"].get("/games/api/health").json()
    assert h["clock"]["alive"] is True and h["clock"]["stale"] is False
    assert h["grader"]["on"] is True and "saved_jobs" in h["grader"]
    # a clock thread that died is started again by the next request
    hg._stop_watchdog()
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    hg._WATCHDOG_THREAD = dead
    hg._WATCHDOG_STARTED = True
    hg._WATCHDOG_STOP.clear()
    assert cast["adm"].get("/games/api/health").json()["clock"]["alive"] is True
    assert hg._WATCHDOG_THREAD is not dead and hg._WATCHDOG_THREAD.is_alive()


def test_the_shutdown_hook_saves_what_tables_owe_and_stops_the_workers(cast, hg, monkeypatch):
    """(Last in the module: it stops the workers, then starts them again.)"""
    assert hg._on_app_shutdown in cast["srv"].app.router.on_shutdown
    p = cast["p"]
    gid = _table(cast, 2)
    t = hg.HUB.get(gid)
    with t.lock:
        t.seats[1].stack_chips += 0  # (nothing changes, but the table owes a save)
        t.persist_dirty = True
    hg._on_app_shutdown()
    assert not t.persist_dirty
    assert hg._STREAMS_STOP.is_set() and hg._GRADER_STOP.is_set() and hg._WATCHDOG_STOP.is_set()
    assert not hg._WATCHDOG_THREAD.is_alive()
    hg._start_watchdog()
    hg._start_grader()
    assert not hg._STREAMS_STOP.is_set() and hg._WATCHDOG_THREAD.is_alive()
    assert _state(p[0], gid)["id"] == gid
