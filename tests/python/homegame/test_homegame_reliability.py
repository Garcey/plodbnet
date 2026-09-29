"""Home games — reliability (improvements backlog, 2026-09-28): the live stream
keeps its table alive and follows a reload (OPS-001), one failing table never
stops the other tables' clocks (OPS-002), failed saves roll every setting back
(OPS-003), automatic chips respect the current maximum buy-in (OPS-004).

Booted once for the module in PUBLIC mode against a temp DB
(`boot_public_server`, tests/python/conftest.py).
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "themilesgarcia@icloud.com"
NAMES = ["uma", "vic", "wes", "xan", "yul", "zed"]


@pytest.fixture(scope="module")
def server(boot_public_server):
    return boot_public_server()


@pytest.fixture(scope="module")
def hg(server):
    return sys.modules["plo5bp.ui.homegame"]


@pytest.fixture(scope="module")
def cast(server):
    def login(email):
        c = TestClient(server.app, raise_server_exceptions=False)
        assert c.get("/auth/dev", params={"email": email}).status_code == 200
        return c

    adm = login(ADMIN_EMAIL)
    players = [login(f"{n}@example.com") for n in NAMES]
    ids = {u["email"]: u["id"] for u in adm.get("/admin/api/users").json()["users"]}
    for n in NAMES:
        r = adm.post("/admin/api/games_access", json={"user_id": ids[f"{n}@example.com"], "action": "grant"})
        assert r.status_code == 200
    by_uid = {ids[f"{n}@example.com"]: players[i] for i, n in enumerate(NAMES)}
    uid = {n: ids[f"{n}@example.com"] for n in NAMES}
    return {"p": players, "adm": adm, "by_uid": by_uid, "uid": uid, "app": server.app}


@pytest.fixture()
def no_watchdog(hg):
    """Stop the background clock thread so a test can drive ``_watchdog_table``
    by hand (restarted afterwards: the live streams run while it runs)."""
    hg._stop_watchdog()
    yield
    hg._start_watchdog()


def _create(client, **kw):
    body = {"name": "reliability", "sb_cents": 50, "bb_cents": 100,
            "ante_cents": 300, "default_buyin_cents": 4000}
    body.update(kw)
    r = client.post("/games/api/tables", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _state(client, gid):
    r = client.get(f"/games/api/tables/{gid}")
    assert r.status_code == 200, r.text
    return r.json()


def _post(client, gid, what, body=None):
    return client.post(f"/games/api/tables/{gid}/{what}", json=body or {})


def _ok(r):
    assert r.status_code == 200, r.text
    return r.json()


def _table(cast, n, host=0, **kw):
    """A table hosted by player ``host`` with players host..host+n-1 seated in order."""
    p = cast["p"]
    gid = _create(p[host], **kw)["id"]
    for i in range(1, n):
        _ok(_post(p[host + i], gid, "sit", {"seat": i, "buyin_cents": 4000}))
    return gid


def _close(cast, gid, host=0):
    """Close a table (MAX_OPEN_TABLES_PER_USER), folding out a hand in progress."""
    for _ in range(40):
        s = _state(cast["p"][host], gid)
        if s["phase"] != "in_hand" or s["actor"] is None:
            break
        cl = cast["by_uid"][s["seats"][s["actor"]]["user_id"]]
        me = _state(cl, gid)
        _ok(_post(cl, gid, "act", {"gate": "fold" if me["legal"]["fold"] else "check_call"}))
    t = sys.modules["plo5bp.ui.homegame"].HUB.get(gid)
    with t.lock:  # skip the showdown's award animation
        if t.runout_active and t.runout_started_mono is not None:
            t.runout_started_mono -= 600.0
    _ok(_post(cast["p"][host], gid, "close"))


def _stream(client, gid, n, *, pings=False):
    """The data pushes of a stream of ``n`` messages (pushes and heartbeat pings);
    with ``pings``: (pushes, number of pings)."""
    with client.stream("GET", f"/games/api/tables/{gid}/stream", params={"max_events": n}) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    chunks = body.split("\n\n")
    data = [json.loads(x[len("data: "):]) for x in chunks if x.startswith("data: ")]
    return (data, sum(1 for x in chunks if x.startswith(": ping"))) if pings else data


# --- OPS-001 / TEST-001: a table watched only through the live stream ------------------------------


def test_a_streamed_table_is_never_evicted_as_idle(cast, hg, monkeypatch):
    """With a healthy stream the client stops polling. Every push must count as
    activity, or a paused table (a break) is dropped after HUB_IDLE_EVICT_S while
    the streams keep serving the dead copy."""
    p = cast["p"]
    gid = _create(p[0])["id"]
    t = hg.HUB.get(gid)
    monkeypatch.setattr(hg, "STREAM_HEARTBEAT_S", 0.2)
    real_view = hg._view
    calls = []

    def view(tt, uid):
        calls.append(1)
        if len(calls) == 1:  # after the first push: half an hour of nothing but the stream
            tt.last_access_mono -= hg.HUB_IDLE_EVICT_S + 60
        return real_view(tt, uid)

    monkeypatch.setattr(hg, "_view", view)
    data, pings = _stream(p[0], gid, 2, pings=True)
    assert [e["id"] for e in data] == [gid] and pings == 1  # the second message = a heartbeat (PERF-008)
    monkeypatch.setattr(hg, "_view", real_view)
    assert gid not in hg.HUB.evict_idle()
    assert hg.HUB.peek(gid) is t
    _close(cast, gid)


def test_the_stream_follows_a_reloaded_table(cast, hg, monkeypatch):
    """A table the process loaded again (a second in-memory copy) is followed by
    the open streams: their next push carries the live copy's epoch, never the
    old copy's state."""
    p = cast["p"]
    gid = _create(p[0])["id"]
    old = hg.HUB.get(gid)
    monkeypatch.setattr(hg, "STREAM_HEARTBEAT_S", 0.2)
    real_view = hg._view
    seen = []

    def view(t, uid):
        out = real_view(t, uid)
        seen.append(t)
        if len(seen) == 1:
            hg.HUB.drop(gid)  # forgotten between two pushes
        return out

    monkeypatch.setattr(hg, "_view", view)
    ev = _stream(p[0], gid, 2)
    live = hg.HUB.peek(gid)
    assert live is not None and live is not old
    assert ev[0]["epoch"] == old.epoch and ev[1]["epoch"] == live.epoch != old.epoch
    monkeypatch.setattr(hg, "_view", real_view)
    _close(cast, gid)


def test_a_stream_ends_when_its_table_is_gone(cast, hg, monkeypatch):
    p = cast["p"]
    gid = _create(p[0])["id"]
    monkeypatch.setattr(hg, "STREAM_HEARTBEAT_S", 0.2)
    real_load = hg._load_table

    def gone(game_id):
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Not Found")

    real_view = hg._view

    def view(t, uid):
        out = real_view(t, uid)
        hg.HUB.drop(gid)
        monkeypatch.setattr(hg, "_load_table", gone)
        return out

    monkeypatch.setattr(hg, "_view", view)
    assert len(_stream(p[0], gid, 5)) == 1  # the first push, then the stream ends
    monkeypatch.setattr(hg, "_view", real_view)
    monkeypatch.setattr(hg, "_load_table", real_load)
    _close(cast, gid)


# --- OPS-002 / TEST-002: one failing table never stops the others -----------------------------------


def test_one_failing_table_leaves_the_other_clocks_running(cast, hg, monkeypatch):
    p = cast["p"]
    bad = _create(p[4])["id"]  # loaded first, so the watchdog ticks it first
    hg.HUB.get(bad).running = True  # (not idle: an idle table is not ticked at all — PERF-007)
    good = _table(cast, 2, decision_secs=5)
    _ok(_post(p[0], good, "run", {"running": True}))
    real = hg._timeout_tick_locked

    def tick(t):
        if t.game_id == bad:
            raise RuntimeError("this table's clock is broken")
        real(t)

    monkeypatch.setattr(hg, "_timeout_tick_locked", tick)
    tg = hg.HUB.get(good)
    with tg.lock:
        seq = tg.action_seq
        tg.turn_started_mono -= 60  # the actor's clock ran out
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline and tg.action_seq == seq:
        time.sleep(0.05)
    assert tg.action_seq > seq, "the good table's actor was never timed out"
    assert hg.HUB.get(bad).wd_failures > 0
    monkeypatch.setattr(hg, "_timeout_tick_locked", real)
    _close(cast, bad, host=4)
    _close(cast, good)


def test_a_failing_step_is_isolated_logged_rarely_and_pauses_the_table(cast, hg, monkeypatch, no_watchdog, caplog):
    p = cast["p"]
    gid = _table(cast, 2)
    _ok(_post(p[0], gid, "run", {"running": True}))
    t = hg.HUB.get(gid)
    ran = []

    def broken(t):
        raise ValueError("the sealed deck cannot be finished")

    monkeypatch.setattr(hg, "_fair_tick_locked", broken)
    monkeypatch.setattr(hg, "_retry_persist_locked", lambda t: ran.append(1))
    caplog.set_level(logging.ERROR, logger=hg.logger.name)
    now = time.monotonic()
    hg._watchdog_table(t, now)
    assert ran == [1], "the steps after the failing one still ran"
    assert t.wd_failures == 1 and t.running
    hg._watchdog_table(t, now + 1)
    assert t.wd_failures == 2 and len(ran) == 2
    logged = [r for r in caplog.records if "watchdog" in r.getMessage()]
    assert len(logged) == 1, "logged once, not four times a second"
    # still failing after WATCHDOG_PAUSE_AFTER_S: paused, announced, saved
    hg._watchdog_table(t, now + hg.WATCHDOG_PAUSE_AFTER_S + 1)
    assert t.running is False
    assert "went wrong" in t.events[-1]["text"] and "Start" in t.events[-1]["text"]
    assert hg.pub.DB.one("SELECT running FROM homegames WHERE id=?", (gid,))["running"] == 0
    # backing off: the next tick waits WATCHDOG_BACKOFF_S
    n = len(ran)
    hg._watchdog_table(t, now + hg.WATCHDOG_PAUSE_AFTER_S + 1.5)
    assert len(ran) == n
    # fixed: the next clean tick resets the streak
    monkeypatch.setattr(hg, "_fair_tick_locked", lambda t: None)
    hg._watchdog_table(t, now + hg.WATCHDOG_PAUSE_AFTER_S + 1 + hg.WATCHDOG_BACKOFF_S)
    assert len(ran) == n + 1 and t.wd_failures == 0 and t.wd_failing_since is None
    _close(cast, gid)


def test_a_deck_that_cannot_be_finished_is_voided_not_stuck(cast, hg, monkeypatch, no_watchdog):
    """``_fair_complete_locked`` caught only HTTPException: any other error from
    the deal (a ValueError from SealedDeck.finish) escaped every tick."""
    if not hg.FAIR_ON:
        pytest.skip("engine without reset_with_deck")
    p = cast["p"]
    gid = _table(cast, 2)
    t = hg.HUB.get(gid)
    real = hg._deal_now_locked
    calls = []

    def broken(t):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("every locked seat must reveal before the cut")
        real(t)

    monkeypatch.setattr(hg, "_deal_now_locked", broken)
    with t.lock:
        t.running = True
        hg._fair_prepare_locked(t)
        first = t.fair_next
        hg._fair_complete_locked(t)
        assert t.fair_next is not first and t.fair_next.attempt == 2  # a new seal
        assert any("Shuffle redone" in e["text"] for e in t.events)
    monkeypatch.setattr(hg, "_deal_now_locked", real)
    _close(cast, gid)


# --- OPS-003 / HGB-005 / TEST-004: one list of settings; a failed save changes nothing -------------

# A new value for every key /settings reads (the source scan below keeps this complete).
SETTINGS_CHANGES = {
    "name": "renamed table", "ante_cents": 200, "min_buyin_cents": 1000, "max_buyin_cents": 9000,
    "default_buyin_cents": 5000, "decision_secs": 60, "time_bank_secs": 30, "deal_delay_secs": 7,
    "street_pause_secs": 2.5, "listed": False, "allow_rabbit": False, "show_grades": False,
    "allow_rathole": True, "approve_buyins": True,
}


def _meta(hg, t):
    return {c.attr: getattr(t, c.attr) for c in hg.META}


def test_the_settings_list_names_every_key_the_route_reads(hg):
    import inspect
    import re

    src = inspect.getsource(hg._settings_locked)
    keys = set(re.findall(r'body\.get\("(\w+)"\)', src)) | set(re.findall(r'_parse_\w+\(body, "(\w+)"', src))
    assert keys - {"num_seats"} == set(SETTINGS_CHANGES), keys ^ set(SETTINGS_CHANGES)


def test_a_failed_settings_save_changes_nothing(cast, hg):
    """Each setting sent together with a seat count that cannot be applied (the
    high seats are taken): refused, and NOTHING moved — in memory or in the DB.
    ``allow_rathole`` used to stay switched on (OPS-003)."""
    p = cast["p"]
    gid = _table(cast, 3)
    t = hg.HUB.get(gid)
    before = _meta(hg, t)
    banks = [s.time_bank_left for s in t.seats if s is not None]
    row = dict(hg.pub.DB.one("SELECT * FROM homegames WHERE id=?", (gid,)))
    for key, value in list(SETTINGS_CHANGES.items()) + [("*", None)]:
        body = dict(SETTINGS_CHANGES) if key == "*" else {key: value}
        r = _post(p[0], gid, "settings", {**body, "num_seats": 2})
        assert r.status_code == 400 and "higher seats" in r.text, (key, r.text)
        assert _meta(hg, t) == before, key
        assert [s.time_bank_left for s in t.seats if s is not None] == banks, key
        assert dict(hg.pub.DB.one("SELECT * FROM homegames WHERE id=?", (gid,))) == row, key
    # the same changes without the impossible resize all go through
    s = _ok(_post(p[0], gid, "settings", dict(SETTINGS_CHANGES)))
    assert s["name"] == "renamed table"
    after = _meta(hg, t)
    assert after["allow_rathole"] is True and after["street_pause_secs"] == 2.5
    assert after["deal_delay_secs"] == 7.0 and after["approve_buyins"] is True
    _close(cast, gid)


def test_every_setting_survives_a_reload(cast, hg):
    """META drives the save AND the load: every setting comes back as it was saved
    (the table comes back paused)."""
    p = cast["p"]
    gid = _table(cast, 2)
    _ok(_post(p[0], gid, "settings", dict(SETTINGS_CHANGES)))
    _ok(_post(p[0], gid, "auto_topup", {"mode": "host", "all_target_cents": 6000, "all_below_cents": 2000}))
    _ok(_post(p[0], gid, "auto_stack", {"mode": "player"}))
    t = hg.HUB.get(gid)
    saved = _meta(hg, t)
    hg.HUB.drop(gid)
    again = hg.HUB.get(gid)
    assert again is not t
    assert _meta(hg, again) == {**saved, "running": False}
    assert again.seats[1].topup_target_cents == 6000 and again.seats[1].topup_below_cents == 2000
    _close(cast, gid)


def test_every_setting_is_remembered_for_the_host_or_deliberately_not(cast, hg):
    """A new table setting must be a decision about the host's remembered
    settings (``_host_prefs_of``), not an accident."""
    remembered = {  # META attribute -> its key in the host's prefs
        "num_seats": "num_seats", "ante_cents": "ante_bb", "default_buyin_cents": "buyin_bb",
        "min_buyin_cents": "min_buyin_bb", "max_buyin_cents": "max_buyin_bb",
        "decision_secs": "decision_secs", "time_bank_secs": "time_bank_secs",
        "deal_delay_secs": "deal_delay_secs", "street_pause_secs": "street_pause_secs",
        "listed": "listed", "approve_buyins": "approve_buyins", "allow_rathole": "allow_rathole",
        "allow_rabbit": "allow_rabbit", "show_grades": "show_grades", "topup_mode": "topup_mode",
        "topup_all_target_cents": "topup_target_bb", "topup_all_below_cents": "topup_below_bb",
        "auto_stack_mode": "auto_stack_mode", "auto_stack_all_cents": "auto_stack_bb",
    }
    per_table = {"host_user_id", "status", "running", "button", "hand_no", "name"}
    attrs = {c.attr for c in hg.META}
    assert attrs == set(remembered) | per_table, attrs ^ (set(remembered) | per_table)
    gid = _create(cast["p"][0])["id"]
    prefs = hg._host_prefs_of(hg.HUB.get(gid))
    assert set(remembered.values()) <= set(prefs)
    _close(cast, gid)


# --- OPS-004: automatic chips never pass a lowered maximum buy-in ----------------------------------


def test_automatic_chips_come_down_with_a_lowered_maximum(cast, hg):
    p = cast["p"]
    gid = _table(cast, 2, max_buyin_cents=10000)
    _ok(_post(p[0], gid, "auto_topup", {"mode": "host", "all_target_cents": 8000, "all_below_cents": 6000}))
    t = hg.HUB.get(gid)
    assert t.seats[1].topup_target_cents == 8000
    s = _ok(_post(p[0], gid, "settings", {"max_buyin_cents": 5000, "default_buyin_cents": 4000}))
    assert (t.topup_all_target_cents, t.topup_all_below_cents) == (5000, 5000)
    assert all((q.topup_target_cents, q.topup_below_cents) == (5000, 5000) for q in t.seats if q)
    assert any("automatic chips now stop at the new maximum ($50.00)" in e["text"] for e in s["events"])
    # the next deal tops a short stack up to the NEW maximum, not the old target
    with t.lock:
        t.seats[1].stack_chips = hg.cents_to_chips(1000, t.bb_cents)
        assert hg._auto_target_chips(t, t.seats[1]) == hg.cents_to_chips(5000, t.bb_cents)
        hg._apply_auto_stacks_locked(t)
        assert hg.chips_to_cents(t.seats[1].stack_chips, t.bb_cents) == 5000
    _close(cast, gid)


def test_a_target_set_before_the_maximum_is_still_capped_when_it_is_applied(cast, hg):
    """Belt and braces: a stored target above the maximum (set before the cap
    existed) is applied AT the maximum."""
    p = cast["p"]
    gid = _table(cast, 2, max_buyin_cents=6000)
    _ok(_post(p[0], gid, "auto_stack", {"mode": "host", "all_cents": 6000}))
    t = hg.HUB.get(gid)
    with t.lock:
        t.seats[1].auto_stack_cents = 9000  # stored under an older, higher maximum
        assert hg._auto_target_chips(t, t.seats[1]) == hg.cents_to_chips(6000, t.bb_cents)
        hg._apply_auto_stacks_locked(t)
        assert hg.chips_to_cents(t.seats[1].stack_chips, t.bb_cents) == 6000
    _close(cast, gid)


# --- OPS-005: every buy-in buys more than the ante ----------------------------------------------------


def test_a_buyin_must_buy_more_than_the_ante(cast, hg):
    p = cast["p"]
    # a table whose default buy-in can never be dealt in is refused
    r = p[1].post("/games/api/tables", json={"name": "x", "bb_cents": 100, "ante_cents": 300,
                                             "default_buyin_cents": 300})
    assert r.status_code == 400 and "more than the ante" in r.text
    r = p[1].post("/games/api/tables", json={"name": "x", "bb_cents": 100, "ante_cents": 300,
                                             "default_buyin_cents": 4000, "min_buyin_cents": 200})
    assert r.status_code == 400 and "minimum buy-in must be more than the ante" in r.text
    gid = _create(p[0])["id"]  # ante $3
    for cents in (100, 300):
        r = _post(p[1], gid, "sit", {"seat": 1, "buyin_cents": cents})
        assert r.status_code == 400 and "more than the ante ($3.00)" in r.text, r.text
    s = _ok(_post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 301}))
    assert s["seats"][1]["stack_cents"] == 301
    # raising the ante past the default buy-in is refused, and changes nothing
    r = _post(p[0], gid, "settings", {"ante_cents": 4000})
    assert r.status_code == 400 and "default buy-in must be more than the ante" in r.text
    assert hg.HUB.get(gid).ante_cents == 300
    # a busted player's rebuy must get them back over the ante
    t = hg.HUB.get(gid)
    with t.lock:
        t.seats[1].stack_chips = 0
    r = _post(p[1], gid, "rebuy", {"amount_cents": 200})
    assert r.status_code == 400 and "more than the ante" in r.text
    _ok(_post(p[1], gid, "rebuy", {"amount_cents": 400}))
    _close(cast, gid)


def test_a_table_saved_under_the_older_rule_can_still_be_renamed(cast, hg):
    p = cast["p"]
    gid = _create(p[0])["id"]
    t = hg.HUB.get(gid)
    with t.lock:
        t.default_buyin_cents = 200  # below the $3 ante: allowed before OPS-005
    assert _ok(_post(p[0], gid, "settings", {"name": "still works"}))["name"] == "still works"
    with t.lock:
        t.default_buyin_cents = 4000
    _close(cast, gid)


# --- HGB-001: an approval that fails keeps the request -------------------------------------------------


def test_an_approval_that_cannot_be_seated_keeps_the_request(cast, hg):
    p = cast["p"]
    gid = _create(p[0], approve_buyins=True, max_buyin_cents=10000)["id"]
    _ok(_post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 8000}))  # a request, not a seat
    t = hg.HUB.get(gid)
    assert [r["amount_cents"] for r in t.requests] == [8000]
    req = t.requests[0]["id"]
    _ok(_post(p[0], gid, "settings", {"max_buyin_cents": 5000}))  # the host changes the limits
    n_events = len(t.events)
    r = _post(p[0], gid, "request", {"action": "approve", "id": req})
    assert r.status_code == 400 and "maximum buy-in is $50.00" in r.text
    assert [x["id"] for x in t.requests] == [req], "the request is still there"
    assert t.seats[1] is None
    assert not any("approved" in e["text"] for e in list(t.events)[n_events:])
    # a different amount, too large for anything: says so (it used to say "at least 1 bb")
    r = _post(p[0], gid, "request", {"action": "approve", "id": req, "amount_cents": 10**12})
    assert r.status_code == 400 and "too large" in r.text
    # approving an amount that fits works, and only then is it announced
    s = _ok(_post(p[0], gid, "request", {"action": "approve", "id": req, "amount_cents": 5000}))
    assert s["seats"][1]["stack_cents"] == 5000 and t.requests == []
    assert any("approved" in e["text"] and "$50.00" in e["text"] for e in s["events"])
    _close(cast, gid)


def test_a_request_that_no_longer_fits_is_dropped_with_a_reason_when_approval_goes_off(cast, hg):
    p = cast["p"]
    gid = _create(p[0], approve_buyins=True, max_buyin_cents=10000)["id"]
    _ok(_post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 8000}))
    t = hg.HUB.get(gid)
    _ok(_post(p[0], gid, "settings", {"max_buyin_cents": 5000}))
    s = _ok(_post(p[0], gid, "settings", {"approve_buyins": False}))
    assert t.requests == [] and t.seats[1] is None
    assert any("request could not go through: maximum buy-in is $50.00" in e["text"] for e in s["events"])
    _close(cast, gid)


# --- OPS-012: why a hand vanished --------------------------------------------------------------------


def test_a_hand_voided_by_abandonment_is_not_blamed_on_a_restart(cast, hg):
    p = cast["p"]
    gid = _table(cast, 2)
    _ok(_post(p[0], gid, "run", {"running": True}))
    t = hg.HUB.get(gid)
    assert t.phase == "in_hand"
    assert gid in hg.HUB.evict_idle(time.monotonic() + hg.HUB_ABANDONED_EVICT_S + 60)
    s = _state(p[0], gid)
    line = next(e["text"] for e in s["events"] if "cut short" in e["text"])
    assert "nobody was at the table for hours" in line and "restarted" not in line
    # a restart (no eviction on record): a hand dealt, then the process died
    hg.HUB.drop(gid)
    hg.pub.DB.q("UPDATE homegames SET hand_no=hand_no+1 WHERE id=?", (gid,))
    s = _state(p[0], gid)
    assert any("the server restarted" in e["text"] for e in s["events"])
    _close(cast, gid)


# --- TEST-008: hot queries never scan a growing table ---------------------------------------------

# (name, SQL, args): every query that runs per view, per poll, per stats page
HOT_QUERIES = [
    ("chat", "SELECT c.id FROM homegame_chat c JOIN users u ON u.id=c.user_id "
             "WHERE c.game_id=? ORDER BY c.id DESC LIMIT 80", ("g",)),
    ("ledger players", "SELECT p.user_id FROM homegame_players p JOIN users u ON u.id=p.user_id "
                       "WHERE p.game_id=? ORDER BY p.user_id", ("g",)),
    ("seated", "SELECT * FROM homegame_players WHERE game_id=? AND seat IS NOT NULL", ("g",)),
    ("open tables of a host", "SELECT COUNT(*) c FROM homegames WHERE host_user_id=? AND status='open'", (1,)),
    ("club tables", "SELECT * FROM homegames WHERE status='open' AND club_id IN (?)", ("c",)),
    ("table hands", "SELECT hand_no FROM homegame_hands WHERE game_id=? AND hand_no<=? "
                    "ORDER BY hand_no DESC LIMIT 30", ("g", 5)),
    ("my results", "SELECT COUNT(*) FROM homegame_hand_results r JOIN homegames g ON g.id=r.game_id "
                   "WHERE r.user_id=? AND g.excluded=0", (1,)),
    ("my head-to-head", "SELECT f.payer FROM homegame_flows f JOIN homegames g ON g.id=f.game_id "
                        "WHERE (f.payer=? OR f.payee=?) AND g.excluded=0", (1, 1)),
    ("table head-to-head", "SELECT payer, payee, SUM(chips) FROM homegame_flows WHERE game_id=? AND hand_no<=? "
                           "GROUP BY payer, payee", ("g", 3)),
    ("last hand", "SELECT MAX(hand_no) AS n FROM homegame_hands WHERE game_id=?", ("g",)),
    ("my clubs", "SELECT c.* FROM homegame_club_members m JOIN homegame_clubs c ON c.id=m.club_id "
                 "WHERE m.user_id=?", (1,)),
    ("shares a club", "SELECT 1 FROM homegame_club_members x JOIN homegame_club_members y "
                      "ON x.club_id=y.club_id WHERE x.user_id=? AND y.user_id=? LIMIT 1", (1, 2)),
    ("a table's ledger", "SELECT * FROM homegame_ledger WHERE game_id=? AND user_id=?", ("g", 1)),
    ("a shuffle proof", "SELECT data FROM homegame_fair WHERE game_id=? AND hand_no=?", ("g", 1)),
]
GROWING = ("homegame_chat", "homegame_hands", "homegame_hand_results", "homegame_flows",
           "homegame_ledger", "homegames", "homegame_players", "homegame_fair", "homegame_club_members")


@pytest.mark.parametrize("name,sql,args", HOT_QUERIES, ids=[q[0] for q in HOT_QUERIES])
def test_hot_queries_use_an_index(hg, name, sql, args):
    """PERF-003 found every view scanning the whole chat table; nothing caught
    it. A full SCAN of a table that grows with play fails here."""
    with hg.pub.DB.transaction():
        plan = [r[3] for r in hg.pub.DB._conn.execute("EXPLAIN QUERY PLAN " + sql, args).fetchall()]
    aliases = {}
    for m in __import__("re").finditer(r"\b(?:FROM|JOIN)\s+(\w+)(?:\s+(?!ON\b|WHERE\b|JOIN\b)(\w+))?", sql):
        aliases[m.group(2) or m.group(1)] = m.group(1)
    for line in plan:
        if line.startswith("SCAN "):
            table = aliases.get(line.split()[1], line.split()[1])
            assert table not in GROWING, (name, plan)


def test_the_chat_is_read_once_per_new_line_not_per_view(cast, hg, monkeypatch):
    p = cast["p"]
    gid = _create(p[0])["id"]
    _ok(_post(p[0], gid, "chat", {"text": "hello"}))
    reads = []
    real = hg.pub.DB.q

    def counting(sql, args=()):
        if "FROM homegame_chat" in sql:
            reads.append(sql)
        return real(sql, args)

    monkeypatch.setattr(hg.pub.DB, "q", counting)
    for _ in range(5):
        assert _state(p[0], gid)["chat"][-1]["text"] == "hello"
    assert len(reads) <= 1
    _ok(_post(p[0], gid, "chat", {"text": "again"}))
    assert _state(p[0], gid)["chat"][-1]["text"] == "again"
    monkeypatch.setattr(hg.pub.DB, "q", real)
    _close(cast, gid)


# --- small correctness items ---------------------------------------------------------------------


def test_allin_means_the_biggest_raise(cast, hg):
    """HGB-019: {"gate": "allin"} used to make the MINIMUM raise."""
    p = cast["p"]
    gid = _table(cast, 2)
    s = _ok(_post(p[0], gid, "run", {"running": True}))
    actor = s["actor"]
    cl = p[actor]
    me = _state(cl, gid)
    top = me["raise_bounds"]["max_chips"]
    after = _ok(_post(cl, gid, "act", {"gate": "allin"}))
    assert after["seats"][actor]["committed_this_street_chips"] == top + me["street_commit_chips"]
    _close(cast, gid)


def test_races_with_another_player_are_409_not_400(cast, hg):
    """HGB-022: the client redraws on 409 ("someone beat you to it") instead of
    showing an error."""
    p = cast["p"]
    gid = _table(cast, 2)
    s = _ok(_post(p[0], gid, "run", {"running": True}))
    waiting = p[1 - s["actor"]]
    assert _post(waiting, gid, "act", {"gate": "check_call"}).status_code == 409  # not your turn
    assert _post(p[2], gid, "sit", {"seat": 1, "buyin_cents": 4000}).status_code == 409  # seat taken
    _close(cast, gid)


def test_money_fields_must_be_whole_cents_and_flags_real_booleans(cast, hg):
    p = cast["p"]
    gid = _table(cast, 2)
    r = _post(p[1], gid, "rebuy", {"amount_cents": 1250.5})
    assert r.status_code == 400 and "whole cents" in r.text  # HGB-021 (round() was banker's)
    assert _post(p[1], gid, "rebuy", {"amount_cents": 1250.0}).status_code == 200  # an integral float is fine
    club = p[0].post("/games/api/clubs", json={"name": "strict"}).json()
    r = p[0].post(f"/games/api/clubs/{club['id']}/requests/decide", json={"user_id": 1, "allow": "false"})
    assert r.status_code == 400  # HGB-020: "false" used to count as yes
    _close(cast, gid)


def test_impossible_results_are_reported(cast, hg, monkeypatch, caplog):
    """OPS-009: the clamp to zero stays, but a negative stack or payouts that do
    not sum to zero are logged as the bugs they are."""
    p = cast["p"]
    gid = _table(cast, 2)
    _ok(_post(p[0], gid, "run", {"running": True}))
    t = hg.HUB.get(gid)
    caplog.set_level(logging.ERROR, logger=hg.logger.name)
    with t.lock:
        real = t.env._rs.payouts

        class Rs:
            def __getattr__(self, name):
                return getattr(real.__self__, name)

            def payouts(self):
                return [-10**9, 5]

        t.env._rs = Rs()
        hg._finish_hand_locked(t)
        t.env._rs = real.__self__
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "do not sum to zero" in msgs and "negative stack" in msgs
    assert all(q.stack_chips >= 0 for q in t.seats if q is not None)


def test_a_second_process_cannot_run_the_home_games_on_the_same_database(hg):
    """OPS-010: the live tables are in one process's memory."""
    path = str(hg.pub.DB.path)
    held = hg._PROCESS_LOCKS.pop(path)
    try:
        with pytest.raises(RuntimeError, match="another server process"):
            hg._take_process_lock()
    finally:
        hg._PROCESS_LOCKS[path] = held


def test_the_clock_skips_a_busy_table_and_an_idle_one(cast, hg, monkeypatch, no_watchdog):
    """PERF-007: a table whose lock is held (a slow view) no longer holds up the
    tables after it, and a paused table with nothing waiting is not ticked."""
    p = cast["p"]
    gid = _table(cast, 2)
    t = hg.HUB.get(gid)
    ticks = []
    monkeypatch.setattr(hg, "_retry_persist_locked", lambda tt: ticks.append(tt.game_id))
    hg._watchdog_table(t, time.monotonic())
    assert ticks == [], "idle: paused, between hands, nothing waiting"
    _ok(_post(p[0], gid, "run", {"running": True}))
    held = threading.Event()
    release = threading.Event()

    def hold():
        with t.lock:
            held.set()
            release.wait(5)

    th = threading.Thread(target=hold)
    th.start()
    held.wait(5)
    t0 = time.monotonic()
    hg._watchdog_table(t, time.monotonic())
    assert time.monotonic() - t0 < 0.5 and ticks == [], "a busy table is skipped, not waited for"
    release.set()
    th.join(5)
    hg._watchdog_table(t, time.monotonic())
    assert ticks == [gid]
    _close(cast, gid)


def test_names_and_chat_lose_invisible_and_direction_characters(cast, hg):
    """SEC-007: a right-to-left override or a zero-width space could make a line
    pretend to be someone else's; emoji (joiners, flags) survive."""
    p = cast["p"]
    gid = _create(p[0], name="Fri‮day​ game")["id"]
    assert _state(p[0], gid)["name"] == "Friday game"
    s = _ok(_post(p[0], gid, "chat", {"text": "gg ⁦wp⁩ \U0001F468‍\U0001F469‍\U0001F467"}))
    assert s["chat"][-1]["text"] == "gg wp \U0001F468‍\U0001F469‍\U0001F467"
    _close(cast, gid)


# --- PERF-008: a heartbeat is a ping, not the whole view ----------------------------------------


def test_the_heartbeat_pings_and_a_new_face_is_pushed(cast, hg, monkeypatch):
    p, uid = cast["p"], cast["uid"]
    gid = _create(p[0])["id"]
    t = hg.HUB.get(gid)
    monkeypatch.setattr(hg, "STREAM_HEARTBEAT_S", 0.15)
    real_view = hg._view
    views = []

    def counting(tt, u):
        views.append(u)
        return real_view(tt, u)

    monkeypatch.setattr(hg, "_view", counting)
    with t.lock:  # (nobody else is here)
        t.seen = {k: v for k, v in t.seen.items() if k == uid[NAMES[0]]}
    data, pings = _stream(p[0], gid, 4, pings=True)
    assert len(data) == 1 and pings == 3, "nothing changed: one view, then pings"
    assert views == [uid[NAMES[0]]], "a ping builds no view"
    with t.lock:
        t.seen[uid[NAMES[0]]] -= 5.0
    before = t.seen[uid[NAMES[0]]]
    _stream(p[0], gid, 2)
    assert t.seen[uid[NAMES[0]]] > before, "a ping keeps the viewer present"
    # someone arriving at the table changes what everyone sees (their seat's "present")
    sig = hg._stream_sig(t)
    with t.lock:
        t.seen[uid[NAMES[1]]] = time.monotonic()
    assert hg._stream_sig(t) != sig
    monkeypatch.setattr(hg, "_view", real_view)
    _close(cast, gid)
