"""Home games — chips in (2026-09-22): host approval of buy-ins + trusted players,
queued top-ups, auto top-up vs set-stack (rathole), spectators / presence, and
the SSE stream.

Booted once for the module in PUBLIC mode against a temp DB
(`boot_public_server`, tests/python/conftest.py).
"""

from __future__ import annotations

import json
import sys

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "themilesgarcia@icloud.com"
NAMES = ["hal", "ivy", "jon", "kim"]


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
    spec = login("railbird@example.com")
    ids = {u["email"]: u["id"] for u in adm.get("/admin/api/users").json()["users"]}
    for email in [f"{n}@example.com" for n in NAMES] + ["railbird@example.com"]:
        r = adm.post("/admin/api/games_access", json={"user_id": ids[email], "action": "grant"})
        assert r.status_code == 200
    by_uid = {ids[f"{n}@example.com"]: players[i] for i, n in enumerate(NAMES)}
    uid = {n: ids[f"{n}@example.com"] for n in NAMES}
    return {"p": players, "spec": spec, "ids": ids, "by_uid": by_uid, "uid": uid, "app": server.app}


def _create(client, **kw):
    body = {"name": "chips", "sb_cents": 50, "bb_cents": 100,
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


def _actor(cast, gid):
    s = _state(cast["p"][0], gid)
    if s["phase"] != "in_hand" or s["actor"] is None:
        return None, s
    cl = cast["by_uid"][s["seats"][s["actor"]]["user_id"]]
    return cl, _state(cl, gid)


def _fold_out(cast, gid):
    """First actor bets the minimum, everyone else folds -> (winner seat, state)."""
    cl, s = _actor(cast, gid)
    winner = s["actor"]
    _ok(_post(cl, gid, "act", {"gate": "raise", "raise_to_chips":
                               s["raise_bounds"]["min_chips"] + s["street_commit_chips"]}))
    for _ in range(12):
        cl, s = _actor(cast, gid)
        if cl is None:
            break
        _ok(_post(cl, gid, "act", {"gate": "fold" if s["legal"]["fold"] else "check_call"}))
    return winner, _state(cast["p"][0], gid)


def _nets(s):
    return sum(r["net_cents"] for r in s["ledger"])


# --- host approval + trust -------------------------------------------------------


def test_untrusted_buy_in_waits_for_the_host_and_holds_the_seat(cast):
    p = cast["p"]
    gid = _create(p[0], approve_buyins=True)["id"]
    s = _ok(_post(p[1], gid, "sit", {"seat": 2, "buyin_cents": 5000}))
    assert s["my_seat"] is None, "sat down without the host's approval"
    assert s["needs_approval"] is True
    assert s["my_request"] == {"id": s["my_request"]["id"], "kind": "sit", "seat": 2, "amount_cents": 5000}
    assert s["seats"][2]["empty"] and s["seats"][2]["reserved_by"] == "ivy"
    assert s["requests"] == [], "only the host sees the queue"
    # the seat is held: nobody else can take or request it
    r = _post(p[2], gid, "sit", {"seat": 2, "buyin_cents": 4000})
    assert r.status_code == 400 and "reserved" in r.json()["detail"]
    host = _state(p[0], gid)
    assert host["needs_approval"] is False  # the host never asks
    assert [(q["name"], q["kind"], q["seat"], q["amount_cents"]) for q in host["requests"]] == [
        ("ivy", "sit", 2, 5000)]
    assert any(e["kind"] == "request" for e in host["events"])
    # only the host resolves it
    rid = host["requests"][0]["id"]
    assert _post(p[2], gid, "request", {"action": "approve", "id": rid}).status_code == 400
    s = _ok(_post(p[0], gid, "request", {"action": "approve", "id": rid}))
    assert s["requests"] == [] and s["seats"][2]["name"] == "ivy"
    assert s["seats"][2]["stack_cents"] == 5000 and s["seats"][2]["trusted"] is False
    assert _nets(s) == 0
    me = _state(p[1], gid)
    assert me["my_seat"] == 2 and me["my_request"] is None


def test_decline_and_cancel(cast):
    p = cast["p"]
    gid = _create(p[0], approve_buyins=True)["id"]
    _ok(_post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 4000}))
    rid = _state(p[0], gid)["requests"][0]["id"]
    s = _ok(_post(p[0], gid, "request", {"action": "deny", "id": rid}))
    assert s["requests"] == [] and s["seats"][1]["empty"] and s["seats"][1]["reserved_by"] is None
    assert _state(p[1], gid)["my_request"] is None
    assert _post(p[0], gid, "request", {"action": "approve", "id": rid}).status_code == 404
    # a player can take their own request back; a new one replaces the old
    _ok(_post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 4000}))
    s = _ok(_post(p[1], gid, "sit", {"seat": 3, "buyin_cents": 6000}))
    assert s["my_request"]["seat"] == 3 and len(_state(p[0], gid)["requests"]) == 1
    s = _ok(_post(p[1], gid, "request", {"action": "cancel"}))
    assert s["my_request"] is None and _state(p[0], gid)["requests"] == []
    assert _post(p[1], gid, "request", {"action": "nonsense"}).status_code == 400


def test_trusted_players_never_wait(cast):
    p = cast["p"]
    gid = _create(p[0], approve_buyins=True)["id"]
    _ok(_post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 4000}))
    rid = _state(p[0], gid)["requests"][0]["id"]
    s = _ok(_post(p[0], gid, "request", {"action": "approve", "id": rid, "trust": True}))
    assert s["seats"][1]["trusted"] is True
    # ivy tops up straight away — no request
    s = _ok(_post(p[1], gid, "rebuy", {"amount_cents": 1000}))
    assert s["my_request"] is None and s["seats"][1]["stack_cents"] == 5000
    assert s["needs_approval"] is False
    # trust survives leaving and coming back (it is per player per table)
    _ok(_post(p[1], gid, "leave"))
    s = _ok(_post(p[1], gid, "sit", {"seat": 4, "buyin_cents": 4000}))
    assert s["my_seat"] == 4 and s["seats"][4]["trusted"] is True
    # ... until the host takes it back
    assert _post(p[1], gid, "trust", {"user_id": cast["uid"]["ivy"], "on": False}).status_code == 400
    s = _ok(_post(p[0], gid, "trust", {"user_id": cast["uid"]["ivy"], "on": False}))
    assert s["seats"][4]["trusted"] is False
    s = _ok(_post(p[1], gid, "rebuy", {"amount_cents": 1000}))
    assert s["my_request"] and s["my_request"]["kind"] == "rebuy"
    assert s["seats"][4]["stack_cents"] == 4000
    # trusting someone who is waiting lets their request through
    s = _ok(_post(p[0], gid, "trust", {"user_id": cast["uid"]["ivy"], "on": True}))
    assert s["requests"] == [] and s["seats"][4]["stack_cents"] == 5000
    assert _nets(s) == 0
    # nobody can be trusted before they have ever played here
    assert _post(p[0], gid, "trust", {"user_id": cast["uid"]["kim"], "on": True}).status_code == 400


def test_switching_approval_off_lets_the_queue_through(cast):
    p = cast["p"]
    gid = _create(p[0], approve_buyins=True)["id"]
    _ok(_post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 4000}))
    _ok(_post(p[2], gid, "sit", {"seat": 2, "buyin_cents": 4500}))
    s = _ok(_post(p[0], gid, "settings", {"approve_buyins": False}))
    assert s["settings"]["approve_buyins"] is False and s["requests"] == []
    assert s["seats"][1]["stack_cents"] == 4000 and s["seats"][2]["stack_cents"] == 4500
    # and with approval off a sit is immediate again
    assert _ok(_post(p[3], gid, "sit", {"seat": 3, "buyin_cents": 4000}))["my_seat"] == 3


def test_requests_respect_the_buy_in_window(cast):
    p = cast["p"]
    gid = _create(p[0], approve_buyins=True, min_buyin_cents=2000, max_buyin_cents=6000)["id"]
    assert _post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 1000}).status_code == 400
    assert _post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 9000}).status_code == 400
    assert _post(p[1], gid, "sit", {"seat": 0, "buyin_cents": 4000}).status_code == 400  # host's seat
    assert _state(p[0], gid)["requests"] == []


def test_a_request_lapses_when_its_player_is_gone(cast, hg):
    import time

    p = cast["p"]
    gid = _create(p[0], approve_buyins=True)["id"]
    _ok(_post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 4000}))
    t = hg.HUB.get(gid)
    with t.lock:
        t.seen[cast["uid"]["ivy"]] = time.monotonic() - hg.REQUEST_TTL_ABSENT_S - 5
        hg._expire_requests_locked(t)
        assert t.requests == []
    assert _state(p[0], gid)["seats"][1]["reserved_by"] is None


# --- queued top-ups ---------------------------------------------------------------


def test_top_up_while_holding_cards_is_queued_until_the_hand_ends(cast):
    p = cast["p"]
    gid = _create(p[0])["id"]
    _ok(_post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 4000}))
    _ok(_post(p[0], gid, "run", {"running": True}))
    # legacy callers (no `queue`) are still refused mid-hand
    assert _post(p[1], gid, "rebuy", {"amount_cents": 2000}).status_code == 400
    s = _ok(_post(p[1], gid, "rebuy", {"amount_cents": 2000, "queue": True}))
    assert s["seats"][1]["queued_topup_cents"] == 2000
    assert s["seats"][1]["stack_cents"] == 3700, "chips behind changed while holding cards"
    assert _state(p[0], gid)["seats"][1]["queued_topup_cents"] == 0  # private to the player
    _ok(_post(p[1], gid, "rebuy", {"amount_cents": 500, "queue": True}))
    winner, s = _fold_out(cast, gid)
    nxt = _ok(_post(p[0], gid, "deal", {}))
    ivy = nxt["seats"][1]
    assert ivy["queued_topup_cents"] == 0
    row = next(r for r in nxt["ledger"] if r["name"] == "ivy")
    assert row["buyin_cents"] == 6500 and _nets(nxt) == 0


def test_queued_top_up_respects_the_table_maximum(cast):
    p = cast["p"]
    gid = _create(p[0], max_buyin_cents=5000)["id"]
    _ok(_post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 4000}))
    _ok(_post(p[0], gid, "run", {"running": True}))
    assert _post(p[1], gid, "rebuy", {"amount_cents": 2000, "queue": True}).status_code == 400
    _ok(_post(p[1], gid, "rebuy", {"amount_cents": 1000, "queue": True}))
    assert _post(p[1], gid, "rebuy", {"amount_cents": 1000, "queue": True}).status_code == 400


# --- auto top-up vs set stack -----------------------------------------------------


def test_auto_top_up_only_tops_up_and_only_below_the_threshold(cast, hg):
    p = cast["p"]
    gid = _create(p[0])["id"]
    _ok(_post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 4000}))
    # players may not choose until the host allows it
    assert _post(p[1], gid, "auto_chips_self", {"kind": "topup", "target_cents": 5000}).status_code == 400
    assert _post(p[1], gid, "auto_topup", {"mode": "player"}).status_code == 400  # host only
    s = _ok(_post(p[0], gid, "auto_topup", {"mode": "player"}))
    assert s["auto_topup"]["mode"] == "player"
    for bad in ({"kind": "topup", "target_cents": 200}, {"kind": "topup", "target_cents": 5000, "below_cents": 6000},
                {"kind": "weird"}, {"kind": "set", "target_cents": 5000}):
        assert _post(p[1], gid, "auto_chips_self", bad).status_code == 400, bad
    s = _ok(_post(p[1], gid, "auto_chips_self", {"kind": "topup", "target_cents": 5000, "below_cents": 3800}))
    assert (s["seats"][1]["topup_target_cents"], s["seats"][1]["topup_below_cents"]) == (5000, 3800)
    # 4000 is not below 3800: the first deal leaves the stack alone
    s = _ok(_post(p[0], gid, "run", {"running": True}))
    assert s["seats"][1]["stack_cents"] == 3700
    winner, s = _fold_out(cast, gid)
    t = hg.HUB.get(gid)
    with t.lock:  # make ivy the one who is short, whoever won the hand
        t.seats[1].stack_chips = hg.cents_to_chips(3000, 100)
        t.seats[0].stack_chips = hg.cents_to_chips(9000, 100)
    nxt = _ok(_post(p[0], gid, "deal", {}))
    assert nxt["seats"][1]["stack_cents"] == 5000 - 300, "topped up to the target, then anted"
    assert nxt["seats"][0]["stack_cents"] == 9000 - 300, "auto top-up must never trim a big stack"
    assert any("auto topped up $20.00" in e["text"] for e in nxt["events"])
    # switching it off
    _fold_out(cast, gid)
    s = _ok(_post(p[1], gid, "auto_chips_self", {"kind": "off"}))
    assert s["seats"][1]["topup_target_cents"] == 0


def test_set_stack_resets_both_ways_and_beats_top_up(cast, hg):
    p = cast["p"]
    gid = _create(p[0])["id"]
    _ok(_post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 4000}))
    _ok(_post(p[0], gid, "auto_topup", {"mode": "player"}))
    _ok(_post(p[0], gid, "auto_stack", {"mode": "player"}))
    _ok(_post(p[1], gid, "auto_chips_self", {"kind": "topup", "target_cents": 6000}))
    s = _ok(_post(p[1], gid, "auto_chips_self", {"kind": "set", "target_cents": 10000}))
    me = s["seats"][1]
    assert me["auto_stack_cents"] == 10000 and me["topup_target_cents"] == 0, "one or the other"
    t = hg.HUB.get(gid)
    s = _ok(_post(p[0], gid, "run", {"running": True}))
    assert s["seats"][1]["stack_cents"] == 10000 - 300  # reset UP to $100
    _fold_out(cast, gid)
    with t.lock:
        t.seats[1].stack_chips = hg.cents_to_chips(25000, 100)  # a big win
    nxt = _ok(_post(p[0], gid, "deal", {}))
    assert nxt["seats"][1]["stack_cents"] == 10000 - 300, "set-stack ratholes back DOWN to $100"
    row = next(r for r in nxt["ledger"] if r["name"] == "ivy")
    assert row["leftover_cents"] == 15000  # the winnings left the table


def test_host_sets_auto_top_up_for_everyone_and_newcomers_inherit(cast):
    p = cast["p"]
    gid = _create(p[0])["id"]
    _ok(_post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 4000}))
    s = _ok(_post(p[0], gid, "auto_topup", {"mode": "host", "all_target_cents": 8000, "all_below_cents": 4000}))
    assert s["auto_topup"] == {"mode": "host", "all_target_cents": 8000, "all_below_cents": 4000}
    assert all((x["topup_target_cents"], x["topup_below_cents"]) == (8000, 4000)
               for x in s["seats"] if not x["empty"])
    s = _ok(_post(p[2], gid, "sit", {"seat": 2, "buyin_cents": 4000}))
    assert s["seats"][2]["topup_target_cents"] == 8000
    # in host mode the players cannot override it, the host can — per player
    assert _post(p[2], gid, "auto_chips_self", {"kind": "topup", "target_cents": 5000}).status_code == 400
    s = _ok(_post(p[0], gid, "auto_topup", {"players": [
        {"user_id": cast["uid"]["jon"], "target_cents": 5000, "below_cents": 0}]}))
    assert s["seats"][2]["topup_target_cents"] == 5000 and s["seats"][1]["topup_target_cents"] == 8000
    # a target above the table maximum is refused
    _ok(_post(p[0], gid, "settings", {"max_buyin_cents": 9000}))
    assert _post(p[0], gid, "auto_topup", {"all_target_cents": 12000}).status_code == 400
    assert _post(p[0], gid, "auto_stack", {"mode": "host", "all_cents": 12000}).status_code == 400


def test_auto_chips_need_trust_while_the_host_approves_buy_ins(cast, hg):
    p = cast["p"]
    gid = _create(p[0])["id"]
    _ok(_post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 4000}))
    _ok(_post(p[2], gid, "sit", {"seat": 2, "buyin_cents": 4000}))
    _ok(_post(p[0], gid, "auto_topup", {"mode": "host", "all_target_cents": 8000}))
    _ok(_post(p[0], gid, "settings", {"approve_buyins": True}))
    _ok(_post(p[0], gid, "trust", {"user_id": cast["uid"]["ivy"], "on": True}))
    s = _ok(_post(p[0], gid, "run", {"running": True}))
    assert s["seats"][0]["stack_cents"] == 8000 - 300  # the host
    assert s["seats"][1]["stack_cents"] == 8000 - 300  # trusted
    assert s["seats"][2]["stack_cents"] == 4000 - 300, "an automatic buy-in nobody approved"
    assert _nets(s) == 0


# --- spectators / presence -------------------------------------------------------------


def test_spectators_are_listed_by_name_and_seated_players_are_not(cast):
    p = cast["p"]
    gid = _create(p[0])["id"]
    _ok(_post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 4000}))
    assert _state(p[0], gid)["spectators"] == []
    _state(cast["spec"], gid)
    _state(p[3], gid)
    s = _state(p[0], gid)
    assert s["spectators"] == ["kim", "railbird"]
    assert "email" not in json.dumps(s["spectators"])
    assert s["seats"][0]["present"] is True and s["seats"][1]["present"] is True


def test_presence_goes_stale(cast, hg):
    import time

    p = cast["p"]
    gid = _create(p[0])["id"]
    _ok(_post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 4000}))
    _state(cast["spec"], gid)
    t = hg.HUB.get(gid)
    with t.lock:
        for uid in list(t.seen):
            if uid != cast["uid"]["hal"]:
                t.seen[uid] = time.monotonic() - hg.PRESENCE_WINDOW_S - 5
    s = _state(p[0], gid)
    assert s["spectators"] == [] and s["seats"][1]["present"] is False


# --- live push ---------------------------------------------------------------------------


def test_stream_pushes_the_viewers_own_state(cast):
    p = cast["p"]
    gid = _create(p[0])["id"]
    _ok(_post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 4000}))
    _ok(_post(p[0], gid, "run", {"running": True}))
    for who, seat in ((p[0], 0), (p[1], 1)):
        with who.stream("GET", f"/games/api/tables/{gid}/stream", params={"max_events": 2}) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            assert "no-cache" in r.headers["cache-control"]
            body = "".join(r.iter_text())
        events = [json.loads(x[len("data: "):]) for x in body.split("\n\n") if x.startswith("data: ")]
        assert len(events) == 2 and body.startswith("retry: ")
        s = events[0]
        assert s["id"] == gid and s["my_seat"] == seat and s["phase"] == "in_hand"
        assert s["seats"][seat]["hole"][0] >= 0            # my cards ...
        assert s["seats"][1 - seat]["hole"] == [-1] * 5    # ... never the other player's


def test_stream_is_gated_like_the_rest_of_the_tree(cast):
    gid = _create(cast["p"][0])["id"]
    unsigned = TestClient(cast["app"])
    assert unsigned.get(f"/games/api/tables/{gid}/stream", params={"max_events": 1}).status_code == 404
    assert cast["p"][0].get("/games/api/tables/nope/stream", params={"max_events": 1}).status_code == 404
