"""Home games — the 2026-09-21 "premium tables" backend.

Table settings + buy-in limits, server-driven dealing, time bank, sit-out-next,
show cards, reactions, the event feed, hand history (and its privacy rules),
unlisted tables + finished sessions, host transfer, rabbit toggle.

The app is booted once for the module in PUBLIC mode against a temp DB
(`boot_public_server`, tests/python/conftest.py).
"""

from __future__ import annotations

import sys
import time

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "themilesgarcia@icloud.com"
NAMES = ["ann", "ben", "cat", "dan"]


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
    spec = login("watcher@example.com")
    ids = {u["email"]: u["id"] for u in adm.get("/admin/api/users").json()["users"]}
    for email in [f"{n}@example.com" for n in NAMES] + ["watcher@example.com"]:
        r = adm.post("/admin/api/games_access", json={"user_id": ids[email], "action": "grant"})
        assert r.status_code == 200
    by_uid = {ids[f"{n}@example.com"]: players[i] for i, n in enumerate(NAMES)}
    return {"p": players, "spec": spec, "ids": ids, "by_uid": by_uid}


# --- helpers -------------------------------------------------------------------


def _create(client, **kw):
    body = {"name": "premium", "sb_cents": 50, "bb_cents": 100,
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


def _table(cast, n, **kw):
    p = cast["p"]
    gid = _create(p[0], **kw)["id"]
    for i in range(1, n):
        assert _post(p[i], gid, "sit", {"seat": i, "buyin_cents": 4000}).status_code == 200
    return gid


def _start(cast, gid):
    r = _post(cast["p"][0], gid, "run", {"running": True})
    assert r.status_code == 200, r.text
    assert r.json()["phase"] == "in_hand"
    return r.json()


def _actor(cast, gid):
    s = _state(cast["p"][0], gid)
    if s["phase"] != "in_hand" or s["actor"] is None:
        return None, s
    cl = cast["by_uid"][s["seats"][s["actor"]]["user_id"]]
    return cl, _state(cl, gid)


def _act(cl, gid, **body):
    r = _post(cl, gid, "act", body)
    assert r.status_code == 200, r.text
    return r.json()


def _fold_out(cast, gid):
    """First actor bets the minimum, everyone else folds. Returns (winner seat,
    final state as seen by the host)."""
    cl, s = _actor(cast, gid)
    winner = s["actor"]
    _act(cl, gid, gate="raise",
         raise_to_chips=s["raise_bounds"]["min_chips"] + s["street_commit_chips"])
    for _ in range(12):
        cl, s = _actor(cast, gid)
        if cl is None:
            break
        _act(cl, gid, gate="fold" if s["legal"]["fold"] else "check_call")
    return winner, _state(cast["p"][0], gid)


def _jam_out(cast, gid):
    for _ in range(40):
        cl, s = _actor(cast, gid)
        if cl is None:
            return s
        if s["legal"]["raise"]:
            _act(cl, gid, gate="raise",
                 raise_to_chips=s["raise_bounds"]["max_chips"] + s["street_commit_chips"])
        else:
            _act(cl, gid, gate="check_call")
    raise AssertionError("hand did not end")


def _finish_runout(hg, gid):
    t = hg.HUB.get(gid)
    with t.lock:
        if t.runout_active and t.runout_started_mono is not None:
            t.runout_started_mono -= 600.0


# --- create options + settings ---------------------------------------------------


def test_create_options_land_in_the_view(cast):
    s = _create(cast["p"][0], num_seats=6, time_bank_secs=30, deal_delay_secs=5,
                min_buyin_cents=2000, max_buyin_cents=10000, listed=False)
    assert s["num_seats"] == 6 and len(s["seats"]) == 6
    assert s["settings"] == {
        "deal_delay_secs": 5.0, "time_bank_secs": 30, "min_buyin_cents": 2000,
        "max_buyin_cents": 10000, "listed": False, "allow_rabbit": True,
        "approve_buyins": False, "show_grades": True, "allow_rathole": False,
    }
    assert s["seats"][0]["bank_left_secs"] == 30.0  # the host is seated with a full bank


def test_create_defaults_stay_manual_and_unlimited(cast):
    s = _create(cast["p"][0])
    assert s["num_seats"] == 8
    assert s["settings"]["deal_delay_secs"] == 0.0  # scripted callers stay deterministic
    assert s["settings"]["time_bank_secs"] == 0
    assert s["settings"]["min_buyin_cents"] == 0 and s["settings"]["max_buyin_cents"] == 0


@pytest.mark.parametrize("bad", [
    {"num_seats": 1}, {"num_seats": 9}, {"time_bank_secs": -1}, {"time_bank_secs": 9999},
    {"deal_delay_secs": 0.5}, {"deal_delay_secs": 500},
    {"min_buyin_cents": 5000, "max_buyin_cents": 1000},
    {"min_buyin_cents": 5000, "default_buyin_cents": 4000},
    {"max_buyin_cents": 3000, "default_buyin_cents": 4000},
])
def test_create_rejects_bad_options(cast, bad):
    body = {"name": "x", "sb_cents": 50, "bb_cents": 100, "ante_cents": 300,
            "default_buyin_cents": 4000, **bad}
    assert cast["p"][1].post("/games/api/tables", json=body).status_code == 400


def test_buyin_limits(cast):
    p = cast["p"]
    gid = _create(p[0], min_buyin_cents=2000, max_buyin_cents=6000)["id"]
    assert _post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 1000}).status_code == 400
    assert _post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 7000}).status_code == 400
    assert _post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 6000}).status_code == 200
    # a top-up may not take the stack past the maximum ...
    r = _post(p[1], gid, "rebuy", {"amount_cents": 100})
    assert r.status_code == 400 and "top up" in r.json()["detail"]
    # ... but the host (seated with 4000) has room for exactly 2000 more
    assert _post(p[0], gid, "rebuy", {"amount_cents": 2100}).status_code == 400
    assert _post(p[0], gid, "rebuy", {"amount_cents": 2000}).status_code == 200


def test_settings_are_host_only_and_validated(cast):
    p = cast["p"]
    gid = _table(cast, 2)
    assert _post(p[1], gid, "settings", {"name": "mine now"}).status_code == 400
    for bad in ({"name": "   "}, {"ante_cents": 0}, {"decision_secs": 3},
                {"time_bank_secs": 100000}, {"deal_delay_secs": 0.2},
                {"street_pause_secs": 99}, {"num_seats": 1},
                {"min_buyin_cents": 9000, "max_buyin_cents": 100}):
        assert _post(p[0], gid, "settings", bad).status_code == 400, bad
    before = _state(p[0], gid)
    s = _post(p[0], gid, "settings", {
        "name": "  Friday   game ", "ante_cents": 500, "decision_secs": 20,
        "time_bank_secs": 45, "deal_delay_secs": 3, "listed": False,
        "allow_rabbit": False, "min_buyin_cents": 1000, "max_buyin_cents": 20000,
    }).json()
    assert s["name"] == "Friday game" and s["stakes"]["ante_cents"] == 500
    assert s["decision_secs"] == 20
    assert s["settings"]["time_bank_secs"] == 45 and s["settings"]["deal_delay_secs"] == 3.0
    assert s["settings"]["listed"] is False and s["settings"]["allow_rabbit"] is False
    assert all(x["bank_left_secs"] == 45.0 for x in s["seats"] if not x["empty"])
    assert s["rev"] > before["rev"]
    texts = [e["text"] for e in s["events"]]
    assert any("ante is now $5.00" in x for x in texts)
    assert any("renamed" in x for x in texts)


def test_settings_survive_a_reload(cast, hg):
    p = cast["p"]
    gid = _table(cast, 2)
    _post(p[0], gid, "settings", {"name": "Reloaded", "ante_cents": 400,
                                  "deal_delay_secs": 7, "time_bank_secs": 20,
                                  "listed": False, "allow_rabbit": False,
                                  "min_buyin_cents": 500, "max_buyin_cents": 90000})
    hg.HUB.drop(gid)
    s = _state(p[0], gid)
    assert s["name"] == "Reloaded" and s["stakes"]["ante_cents"] == 400
    assert s["settings"] == {
        "deal_delay_secs": 7.0, "time_bank_secs": 20, "min_buyin_cents": 500,
        "max_buyin_cents": 90000, "listed": False, "allow_rabbit": False,
        "approve_buyins": False, "show_grades": True, "allow_rathole": False,
    }
    assert all(x["bank_left_secs"] == 20.0 for x in s["seats"] if not x["empty"])


def test_resize_between_hands_only_and_never_over_a_player(cast):
    p = cast["p"]
    gid = _create(p[0])["id"]
    assert _post(p[1], gid, "sit", {"seat": 5, "buyin_cents": 4000}).status_code == 200
    r = _post(p[0], gid, "settings", {"num_seats": 4})
    assert r.status_code == 400 and "higher seats" in r.json()["detail"]
    s = _post(p[0], gid, "settings", {"num_seats": 6}).json()
    assert s["num_seats"] == 6 and len(s["seats"]) == 6 and s["seats"][5]["name"]
    _start(cast, gid)
    assert _post(p[0], gid, "settings", {"num_seats": 7}).status_code == 400  # mid-hand
    # everything else IS editable mid-hand (the hand keeps the config it was dealt with)
    assert _post(p[0], gid, "settings", {"ante_cents": 600}).status_code == 200


def test_ante_change_applies_to_the_next_hand(cast):
    p = cast["p"]
    gid = _table(cast, 2)
    s = _start(cast, gid)
    assert s["pot_cents"] == 600
    _post(p[0], gid, "settings", {"ante_cents": 500})
    assert _state(p[0], gid)["pot_cents"] == 600  # the live hand is untouched
    _fold_out(cast, gid)
    s = _post(p[0], gid, "deal", {}).json()
    assert s["phase"] == "in_hand" and s["pot_cents"] == 1000


# --- server-driven dealing -------------------------------------------------------


def _wait_for(pred, timeout=6.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.1)
    return None


def test_server_deals_the_next_hand(cast):
    p = cast["p"]
    gid = _table(cast, 2, deal_delay_secs=1)
    _start(cast, gid)
    _, end = _fold_out(cast, gid)
    assert end["phase"] == "showdown" and end["hand_no"] == 1
    got = _wait_for(lambda: (lambda s: s if s["hand_no"] == 2 else None)(_state(p[1], gid)))
    assert got is not None, "nobody pressed Deal: the server must deal hand 2"
    assert got["phase"] == "in_hand"


def test_next_deal_countdown_is_published(cast):
    p = cast["p"]
    gid = _table(cast, 2, deal_delay_secs=8)
    _start(cast, gid)
    _fold_out(cast, gid)
    got = _wait_for(lambda: (lambda s: s if s["next_deal_in_secs"] is not None else None)(
        _state(p[0], gid)), timeout=3.0)
    assert got is not None and 0 < got["next_deal_in_secs"] <= 8.0
    # pausing cancels the countdown
    s = _post(p[0], gid, "run", {"running": False}).json()
    assert s["next_deal_in_secs"] is None
    time.sleep(0.6)
    assert _state(p[0], gid)["next_deal_in_secs"] is None


def test_manual_tables_are_never_dealt_by_the_server(cast):
    p = cast["p"]
    gid = _table(cast, 2)  # no deal_delay_secs -> manual
    _start(cast, gid)
    _fold_out(cast, gid)
    time.sleep(1.2)
    s = _state(p[0], gid)
    assert s["phase"] == "showdown" and s["hand_no"] == 1 and s["next_deal_in_secs"] is None


# --- time bank -------------------------------------------------------------------


def test_time_bank_burns_after_the_base_clock_then_times_out(cast, hg):
    gid = _table(cast, 2, decision_secs=10, time_bank_secs=20)
    _start(cast, gid)
    cl, s = _actor(cast, gid)
    seat = s["actor"]
    assert s["time_bank"] == {"secs": 20, "active": False, "remaining_secs": 22.0} or \
        s["time_bank"]["active"] is False
    t = hg.HUB.get(gid)
    with t.lock:
        bank0 = t.seats[seat].time_bank_left
        t.turn_started_mono -= 11.0  # the base clock ran out a second ago
        hg._timeout_tick_locked(t)
        assert t.phase == "in_hand" and t.env.current_actor() == seat, "bank keeps them alive"
    s = _state(cl, gid)
    assert s["time_bank"]["active"] is True and s["turn_remaining_secs"] == 0.0
    assert 0 < s["time_bank"]["remaining_secs"] < bank0
    # acting now charges only the seconds used (about one)
    _act(cl, gid, gate="check_call")
    with t.lock:
        left = t.seats[seat].time_bank_left
    assert bank0 - 2.5 < left < bank0 - 0.5

    # the next actor burns the WHOLE bank -> auto-acted, bank empty, a timeout line
    cl2, s2 = _actor(cast, gid)
    seat2 = s2["actor"]
    with t.lock:
        t.turn_started_mono -= 11.0
        hg._timeout_tick_locked(t)
        t.bank_started_mono -= 60.0
        hg._timeout_tick_locked(t)
        assert t.seats[seat2].time_bank_left == 0.0
        assert t.env is None or t.env.current_actor() != seat2 or t.phase != "in_hand" \
            or t.action_seq > s2["action_seq"]
    assert any(e["kind"] == "timeout" for e in _state(cl, gid)["events"])


def test_time_bank_refills_a_little_each_hand_up_to_the_cap(cast, hg):
    gid = _table(cast, 2, time_bank_secs=10)
    t = hg.HUB.get(gid)
    with t.lock:
        t.seats[0].time_bank_left = 3.0
    _start(cast, gid)
    with t.lock:
        assert t.seats[0].time_bank_left == pytest.approx(3.0 + hg.TIME_BANK_REFILL_S)
        assert t.seats[1].time_bank_left == 10.0  # capped


# --- sit out next hand -----------------------------------------------------------


def test_sit_out_next_hand_finishes_the_hand_first(cast, hg):
    p = cast["p"]
    gid = _table(cast, 3)
    _start(cast, gid)
    s = _post(p[2], gid, "sit_out", {"on": True, "next_hand": True}).json()
    me = s["seats"][2]
    assert me["sit_out_next"] is True and me["sitting_out"] is False and not me["folded"]
    # toggling it back off before the hand ends cancels it
    s = _post(p[2], gid, "sit_out", {"on": False}).json()
    assert s["seats"][2]["sit_out_next"] is False
    _post(p[2], gid, "sit_out", {"on": True, "next_hand": True})
    for _ in range(40):
        cl, s = _actor(cast, gid)
        if cl is None:
            break
        _act(cl, gid, gate="check_call")
    s = _state(p[0], gid)
    assert s["phase"] == "showdown"
    assert s["seats"][2]["sitting_out"] is True and s["seats"][2]["sit_out_next"] is False
    _finish_runout(hg, gid)  # a checked-down showdown still steps through its awards
    nxt = _post(p[0], gid, "deal", {}).json()
    assert nxt["seats"][2]["in_hand"] is False


def test_sit_out_next_between_hands_is_immediate(cast):
    p = cast["p"]
    gid = _table(cast, 2)
    s = _post(p[1], gid, "sit_out", {"on": True, "next_hand": True}).json()
    assert s["seats"][1]["sitting_out"] is True


# --- show cards ------------------------------------------------------------------


def test_winner_can_show_after_a_fold_out(cast):
    p = cast["p"]
    gid = _table(cast, 2)
    _start(cast, gid)
    winner, end = _fold_out(cast, gid)
    loser = 1 - winner
    assert _state(p[loser], gid)["seats"][winner]["hole"] == [-1] * 5
    assert _state(p[winner], gid)["can_show"] is True
    assert _state(cast["spec"], gid)["can_show"] is False
    assert _post(cast["spec"], gid, "show").status_code == 400
    r = _post(p[winner], gid, "show", {"hand_no": end["hand_no"]})
    assert r.status_code == 200 and r.json()["can_show"] is False
    seen = _state(p[loser], gid)["seats"][winner]
    assert seen["hole"][0] >= 0 and seen["shown"] is True and seen["hand_desc"]
    # the folder may show too; a stale hand number is refused
    assert _post(p[loser], gid, "show", {"hand_no": 99}).status_code == 409
    assert _post(p[loser], gid, "show").status_code == 200
    # a new hand clears it
    s = _post(p[0], gid, "deal", {}).json()
    assert all(not x["shown"] for x in s["seats"])
    assert _post(p[winner], gid, "show").status_code == 400  # mid-hand


# --- reactions + events ----------------------------------------------------------


def test_reactions_are_whitelisted_and_seated_only(cast):
    p = cast["p"]
    gid = _table(cast, 2)
    assert _post(p[1], gid, "react", {"emote": "<script>"}).status_code == 400
    assert _post(cast["spec"], gid, "react", {"emote": "gg"}).status_code == 400
    s = _post(p[1], gid, "react", {"emote": "GG"}).json()
    assert s["reactions"] and s["reactions"][-1] == {
        "id": s["reactions"][-1]["id"], "seat": 1, "emote": "gg"}
    assert _state(p[0], gid)["reactions"][-1]["emote"] == "gg"


def test_event_feed_and_no_result_spoiler_during_a_runout(cast, hg):
    p = cast["p"]
    gid = _table(cast, 2)
    kinds = [e["kind"] for e in _state(p[0], gid)["events"]]
    assert kinds.count("join") == 2
    _start(cast, gid)
    _jam_out(cast, gid)
    s = _state(p[0], gid)
    assert s["runout"]["blocking"] is True
    assert not any(e["kind"] == "win" for e in s["events"]), "result line ran ahead of the runout"
    assert s["last_hand_no"] is None
    _finish_runout(hg, gid)
    s = _state(p[0], gid)
    assert any(e["kind"] == "win" and "Hand #1" in e["text"] for e in s["events"])
    assert s["last_hand_no"] == 1
    ids = [e["id"] for e in s["events"]]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


# --- hand history ----------------------------------------------------------------


def test_hand_history_hides_what_the_table_never_saw(cast):
    p = cast["p"]
    gid = _table(cast, 3)
    _start(cast, gid)
    winner, _ = _fold_out(cast, gid)
    assert _post(cast["spec"], gid, "show").status_code == 400
    assert cast["spec"].get(f"/games/api/tables/{gid}/hands").status_code == 403
    assert cast["spec"].get(f"/games/api/tables/{gid}/hands/1").status_code == 403
    for i, cl in enumerate(p[:3]):
        lst = cl.get(f"/games/api/tables/{gid}/hands").json()
        assert [h["hand_no"] for h in lst["hands"]] == [1]
        h = lst["hands"][0]
        assert h["showdown"] is False and len(h["board_a"]) == 3
        assert h["my_hole"] and len(h["my_hole"]) == 5
        assert h["winners"] and h["winners"][0]["delta_cents"] > 0
        det = cl.get(f"/games/api/tables/{gid}/hands/1").json()
        assert "user_id" not in str(det["seats"])
        for s in det["seats"]:
            if s["is_me"]:
                assert s["hole"] and s["seat"] == i
            else:
                assert s["hole"] is None, "a hand nobody tabled leaked through history"
        assert det["actions"] and det["actions"][0]["label"].startswith("Raise to")
        assert sum(s["delta_cents"] for s in det["seats"]) == 0
    stats = {r["name"]: r for r in p[0].get(f"/games/api/tables/{gid}/hands").json()["stats"]}
    assert all(r["hands"] == 1 for r in stats.values())
    assert sum(r["wins"] for r in stats.values()) == 1
    # a voluntary show reaches history too
    assert _post(p[winner], gid, "show").status_code == 200
    other = p[(winner + 1) % 3]
    det = other.get(f"/games/api/tables/{gid}/hands/1").json()
    shown = next(s for s in det["seats"] if s["seat"] == winner)
    assert shown["hole"] and shown["shown"] is True


def test_showdown_hands_are_tabled_in_history_but_not_before_the_runout_ends(cast, hg):
    p = cast["p"]
    gid = _table(cast, 2)
    _start(cast, gid)
    _jam_out(cast, gid)
    assert _state(p[0], gid)["runout"]["blocking"] is True
    assert p[0].get(f"/games/api/tables/{gid}/hands").json()["hands"] == []
    assert p[0].get(f"/games/api/tables/{gid}/hands/1").status_code == 404
    assert p[0].get(f"/games/api/tables/{gid}/hands").json()["stats"] == []
    _finish_runout(hg, gid)
    det = p[0].get(f"/games/api/tables/{gid}/hands/1").json()
    assert det["showdown"] is True and len(det["board_a"]) == 5 and len(det["board_b"]) == 5
    assert all(s["hole"] and s["shown"] for s in det["seats"])
    assert det["awards"] and {a["board"] for a in det["awards"]} <= {"a", "b"}
    assert p[0].get(f"/games/api/tables/{gid}/hands/2").status_code == 404


def test_history_pages_backwards(cast):
    p = cast["p"]
    gid = _table(cast, 2)
    _start(cast, gid)
    for _ in range(3):
        _fold_out(cast, gid)
        _post(p[0], gid, "deal", {})
    lst = p[0].get(f"/games/api/tables/{gid}/hands", params={"limit": 2}).json()
    assert [h["hand_no"] for h in lst["hands"]] == [3, 2] and lst["more"] is True
    lst = p[0].get(f"/games/api/tables/{gid}/hands", params={"limit": 2, "before": 2}).json()
    assert [h["hand_no"] for h in lst["hands"]] == [1] and lst["more"] is False


# --- lobby -----------------------------------------------------------------------


def test_unlisted_tables_are_link_only(cast):
    p = cast["p"]
    gid = _create(p[0], name="secret", listed=False)["id"]

    def ids(cl):
        return {t["id"] for t in cl.get("/games/api/tables").json()["tables"]}

    assert gid in ids(p[0]) and gid not in ids(p[1])
    assert _state(p[1], gid)["name"] == "secret"  # the link still works
    assert _post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 4000}).status_code == 200
    assert gid in ids(p[1])  # ... and once you have played there it is in your lobby
    row = next(t for t in p[1].get("/games/api/tables").json()["tables"] if t["id"] == gid)
    assert row["is_seated"] is True and row["is_host"] is False and row["listed"] is False
    assert [x["name"] for x in row["players"]] == ["ann", "ben"]
    assert "email" not in str(row) and "user_id" not in str(row)


def test_finished_sessions_show_the_net_result(cast):
    p = cast["p"]
    gid = _table(cast, 2)
    _start(cast, gid)
    winner, _ = _fold_out(cast, gid)
    assert _post(p[0], gid, "close").status_code == 200
    nets = {}
    for i in (0, 1):
        sess = p[i].get("/games/api/tables").json()["sessions"]
        row = next(s for s in sess if s["id"] == gid)
        assert row["hands"] == 1 and row["buyin_cents"] == 4000
        nets[i] = row["net_cents"]
    assert nets[winner] > 0 and nets[0] + nets[1] == 0


# --- host transfer + rabbit toggle -------------------------------------------------


def test_transfer_host(cast):
    p = cast["p"]
    gid = _table(cast, 2)
    assert _post(p[1], gid, "transfer_host", {"user_id": cast["ids"]["ann@example.com"]}).status_code == 400
    assert _post(p[0], gid, "transfer_host", {"user_id": cast["ids"]["cat@example.com"]}).status_code == 400
    s = _post(p[0], gid, "transfer_host", {"user_id": cast["ids"]["ben@example.com"]}).json()
    assert s["is_host"] is False and s["seats"][1]["is_host"] is True
    assert _state(p[1], gid)["is_host"] is True
    assert _post(p[0], gid, "settings", {"name": "nope"}).status_code == 400


def test_rabbit_can_be_switched_off(cast):
    p = cast["p"]
    gid = _table(cast, 2)
    _post(p[0], gid, "settings", {"allow_rabbit": False})
    _start(cast, gid)
    _, end = _fold_out(cast, gid)
    assert end["can_rabbit"] is False
    assert end["board"]["a"]["turn"] is None and end["board"]["b"]["river"] is None
    assert _post(p[0], gid, "rabbit").status_code == 400
    assert _state(p[1], gid)["board"]["a"]["turn"] is None


def test_new_assets_are_gated(cast, server):
    unsigned = TestClient(server.app)
    for name in ("games.js", "games.table.js", "games.ui.js", "games.play.js",
                 "games.sound.js", "games.fair.js", "games.css"):
        assert unsigned.get(f"/games/static/{name}").status_code == 404
        assert unsigned.get(f"/static/{name}").status_code == 404
        assert cast["p"][0].get(f"/static/{name}").status_code == 404


# --- sitting down / reloading while a hand is running ------------------------------


def test_sit_and_reload_while_a_hand_is_running(cast):
    """With the server dealing every few seconds the between-hands window is too
    short to buy in: a seat that is NOT in the hand may sit / top up right away."""
    p = cast["p"]
    gid = _table(cast, 2)
    _start(cast, gid)
    r = _post(p[2], gid, "sit", {"seat": 2, "buyin_cents": 5000})
    assert r.status_code == 200, r.text
    me = r.json()["seats"][2]
    assert me["in_hand"] is False and me["stack_cents"] == 5000 and me["hole"] is None
    assert _post(p[2], gid, "rebuy", {"amount_cents": 1000}).status_code == 200
    assert _state(p[0], gid)["seats"][2]["stack_cents"] == 6000
    # table stakes: the players holding cards cannot reload until the hand is over
    r = _post(p[0], gid, "rebuy", {"amount_cents": 1000})
    assert r.status_code == 400 and "hand" in r.json()["detail"]
    _fold_out(cast, gid)
    s = _state(p[0], gid)
    assert s["seats"][2]["stack_cents"] == 6000, "the hand's end reset an idle seat's stack"
    assert sum(row["net_cents"] for row in s["ledger"]) == 0
    nxt = _post(p[0], gid, "deal", {}).json()
    assert nxt["phase"] == "in_hand" and nxt["seats"][2]["in_hand"] is True


def test_reload_during_a_runout_keeps_the_ledger_zero_sum(cast, hg):
    p = cast["p"]
    gid = _table(cast, 3)
    assert _post(p[2], gid, "sit_out", {"on": True}).status_code == 200  # seat 2 sits this one out
    _start(cast, gid)
    _jam_out(cast, gid)
    s = _state(p[0], gid)
    assert s["runout"]["blocking"] is True
    assert _post(p[2], gid, "rebuy", {"amount_cents": 2500}).status_code == 200
    s = _state(p[0], gid)
    assert s["seats"][2]["stack_cents"] == 6500
    assert sum(row["net_cents"] for row in s["ledger"]) == 0
    _finish_runout(hg, gid)
    s = _state(p[0], gid)
    assert s["seats"][2]["stack_cents"] == 6500
    assert sum(row["net_cents"] for row in s["ledger"]) == 0


# --- nobody home: the server must not keep a table churning --------------------------


def test_server_does_not_deal_to_an_empty_room(cast, hg):
    p = cast["p"]
    gid = _table(cast, 2, deal_delay_secs=1)
    _start(cast, gid)
    _fold_out(cast, gid)
    t = hg.HUB.get(gid)
    with t.lock:  # both browsers went away a minute ago
        for uid in list(t.seen):
            t.seen[uid] = time.monotonic() - 60.0
        t.next_deal_mono = None
    time.sleep(1.8)
    with t.lock:
        assert t.hand_no == 1 and t.next_deal_mono is None, "dealt to absent players"
    # one player looking at the table is not a game either ...
    assert _state(p[0], gid)["hand_no"] == 1
    time.sleep(1.5)
    with t.lock:
        assert t.hand_no == 1
    # ... two are
    _state(p[1], gid)
    got = _wait_for(lambda: (lambda s: s if s["hand_no"] == 2 else None)(_state(p[1], gid)))
    assert got is not None


def test_two_timeouts_in_a_row_sit_the_player_out(cast, hg):
    gid = _table(cast, 3, decision_secs=10)
    _start(cast, gid)
    t = hg.HUB.get(gid)
    cl, s = _actor(cast, gid)
    sleeper = s["actor"]

    def expire():
        with t.lock:
            t.turn_started_mono -= 11.0
            hg._timeout_tick_locked(t)

    expire()  # 1st timeout: checked (nothing to call on the flop)
    with t.lock:
        assert t.seats[sleeper].timeouts == 1 and not t.seats[sleeper].sitting_out
    for _ in range(12):  # everyone else checks until the sleeper is up again
        cl, s = _actor(cast, gid)
        assert cl is not None
        if s["actor"] == sleeper:
            break
        _act(cl, gid, gate="check_call")
    expire()  # 2nd in a row
    s = _state(cast["p"][0], gid)
    assert s["seats"][sleeper]["sitting_out"] is True
    assert any("timed out twice" in e["text"] for e in s["events"])
    with t.lock:
        assert t.seats[sleeper].timeouts == 0


def test_acting_yourself_clears_the_timeout_streak(cast, hg):
    gid = _table(cast, 2, decision_secs=10)
    _start(cast, gid)
    t = hg.HUB.get(gid)
    cl, s = _actor(cast, gid)
    seat = s["actor"]
    with t.lock:
        t.turn_started_mono -= 11.0
        hg._timeout_tick_locked(t)
        assert t.seats[seat].timeouts == 1
    for _ in range(6):
        cl, s = _actor(cast, gid)
        if cl is None or s["actor"] == seat:
            break
        _act(cl, gid, gate="check_call")
    if cl is not None:
        _act(cl, gid, gate="check_call")
        with t.lock:
            assert t.seats[seat].timeouts == 0
