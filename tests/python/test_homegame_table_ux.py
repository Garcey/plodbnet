"""Home games, table UX round 2 (2026-09-22): the host approves a request for a
DIFFERENT amount; chips can be taken OFF the table when the host allows it;
leaving is "after this hand", never a forced fold; award steps are tagged with
the named pot (deepest side pot first, main pot last) they pay from."""
from __future__ import annotations

import sys

import pytest
from starlette.testclient import TestClient

ADMIN_EMAIL = "admin@ux.example"
NAMES = ["pat", "quinn", "rae", "sol"]


@pytest.fixture(scope="module")
def server(boot_public_server):
    return boot_public_server(PLO5BP_ADMIN_EMAILS=ADMIN_EMAIL)


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
    players = [login(f"{n}@ux.example") for n in NAMES]
    ids = {u["email"]: u["id"] for u in adm.get("/admin/api/users").json()["users"]}
    for n in NAMES:
        assert adm.post("/admin/api/games_access", json={"user_id": ids[f"{n}@ux.example"], "action": "grant"}).status_code == 200
    by_uid = {ids[f"{n}@ux.example"]: players[i] for i, n in enumerate(NAMES)}
    return {"p": players, "by_uid": by_uid}


def _post(cl, gid, what, body=None):
    return cl.post(f"/games/api/tables/{gid}/{what}", json=body or {})


def _ok(r):
    assert r.status_code == 200, r.text
    return r.json()


def _state(cl, gid):
    return _ok(cl.get(f"/games/api/tables/{gid}"))


def _table(cast, n, **kw):
    body = {"name": "ux", "bb_cents": 100, "ante_cents": 300, "default_buyin_cents": 4000, **kw}
    gid = _ok(cast["p"][0].post("/games/api/tables", json=body))["id"]
    for i in range(1, n):
        _ok(_post(cast["p"][i], gid, "sit", {"seat": i, "buyin_cents": 4000}))
    return gid


def _actor(cast, gid):
    s = _state(cast["p"][0], gid)
    if s["phase"] != "in_hand" or s["actor"] is None:
        return None, s
    cl = cast["by_uid"][s["seats"][s["actor"]]["user_id"]]
    return cl, _state(cl, gid)


def _check_down(cast, gid, folders=()):
    for _ in range(40):
        cl, s = _actor(cast, gid)
        if cl is None:
            return s
        gate = "fold" if (s["actor"] in folders and s["legal"]["fold"]) else "check_call"
        _ok(_post(cl, gid, "act", {"gate": gate, "hand_no": s["hand_no"], "action_seq": s["action_seq"]}))
    raise AssertionError("hand did not end")


def _finish_runout(hg, gid):
    t = hg.HUB.get(gid)
    with t.lock:
        if t.runout_active and t.runout_started_mono is not None:
            t.runout_started_mono -= 600.0


def _nets(s):
    return sum(r["net_cents"] for r in s["ledger"])


# --- the create dialog's new shape ---------------------------------------------------------


def test_a_table_is_created_from_a_big_blind_and_an_ante_alone(cast):
    p = cast["p"]
    s = _ok(p[0].post("/games/api/tables", json={"name": "bb only", "bb_cents": 250, "ante_cents": 750, "default_buyin_cents": 10000}))
    assert s["stakes"]["bb_cents"] == 250 and s["stakes"]["ante_cents"] == 750
    assert s["stakes"]["sb_cents"] == 125, "nobody posts it — it defaults to half a bb"
    assert s["settings"]["allow_rathole"] is False


# --- approving a different amount ------------------------------------------------------------


def test_the_host_can_approve_a_smaller_buy_in_than_asked(cast):
    p = cast["p"]
    gid = _ok(p[0].post("/games/api/tables", json={"name": "approve", "bb_cents": 100, "ante_cents": 300, "default_buyin_cents": 4000, "approve_buyins": True}))["id"]
    _ok(_post(p[1], gid, "sit", {"seat": 3, "buyin_cents": 15000}))
    host = _state(p[0], gid)
    rid = host["requests"][0]["id"]
    assert host["seats"][3]["reserved_by"] == "quinn"
    # below a bb is refused and the request survives
    r = _post(p[0], gid, "request", {"action": "approve", "id": rid, "amount_cents": 50})
    assert r.status_code == 400 and len(_state(p[0], gid)["requests"]) == 1
    s = _ok(_post(p[0], gid, "request", {"action": "approve", "id": rid, "amount_cents": 8000, "trust": True}))
    assert s["seats"][3]["name"] == "quinn" and s["seats"][3]["stack_cents"] == 8000
    assert s["seats"][3]["trusted"] is True and s["requests"] == []
    assert any("$80.00" in e["text"] and "$150.00" in e["text"] for e in s["events"]), "the table is told what changed"
    assert _nets(s) == 0
    # a top-up request the same way (unchanged amount when none is given)
    _ok(_post(p[0], gid, "trust", {"user_id": s["seats"][3]["user_id"], "on": False}))
    _ok(_post(p[1], gid, "rebuy", {"amount_cents": 5000}))
    host = _state(p[0], gid)
    rid = host["requests"][0]["id"]
    assert host["seats"][3]["request"] == {"id": rid, "kind": "rebuy", "amount_cents": 5000}, "the host sees it on the seat"
    assert _state(p[1], gid)["seats"][3]["request"] is None, "nobody else does"
    s = _ok(_post(p[0], gid, "request", {"action": "approve", "id": rid, "amount_cents": 2000}))
    assert s["seats"][3]["stack_cents"] == 10000 and _nets(s) == 0


# --- taking chips off the table -----------------------------------------------------------------


def test_removing_chips_needs_the_hosts_switch_and_keeps_the_ledger_square(cast, hg):
    p = cast["p"]
    gid = _table(cast, 2)
    r = _post(p[1], gid, "remove_chips", {"amount_cents": 1000})
    assert r.status_code == 400 and "does not allow" in r.json()["detail"]
    s = _ok(_post(p[0], gid, "settings", {"allow_rathole": True}))
    assert s["settings"]["allow_rathole"] is True
    s = _ok(_post(p[1], gid, "remove_chips", {"amount_cents": 1000}))
    assert s["seats"][1]["stack_cents"] == 3000 and _nets(s) == 0
    row = next(x for x in s["ledger"] if x["name"] == "quinn")
    assert row["leftover_cents"] == 1000 and row["buyin_cents"] == 4000 and row["net_cents"] == 0
    # never below an ante + a bb: that is leaving, not a withdrawal
    r = _post(p[1], gid, "remove_chips", {"amount_cents": 2700})
    assert r.status_code == 400 and "leave the table" in r.json()["detail"]
    assert _ok(_post(p[1], gid, "remove_chips", {"amount_cents": 2600}))["seats"][1]["stack_cents"] == 400
    # mid-hand it is queued and lands after the hand, re-checked against the stack THEN
    _ok(_post(p[1], gid, "rebuy", {"amount_cents": 3600}))
    _ok(_post(p[0], gid, "run", {"running": True}))
    s = _state(p[1], gid)
    assert s["phase"] == "in_hand"
    r = _post(p[1], gid, "remove_chips", {"amount_cents": 1500})
    assert r.status_code == 400, "not while holding cards, unless queued"
    s = _ok(_post(p[1], gid, "remove_chips", {"amount_cents": 1500, "queue": True}))
    assert s["seats"][1]["queued_remove_cents"] == 1500
    _check_down(cast, gid)
    _finish_runout(hg, gid)
    _ok(_post(p[0], gid, "run", {"running": False}))
    t = hg.HUB.get(gid)
    with t.lock:  # (what the watchdog does every quarter second)
        hg._apply_queued_topups_locked(t)
    done = _state(p[1], gid)
    quinn = done["seats"][1]
    assert quinn["queued_remove_cents"] == 0
    row = next(x for x in done["ledger"] if x["name"] == "quinn")
    assert row["leftover_cents"] in (1000 + 2600 + 1500, 1000 + 2600), "landed (or skipped if the hand left too little)"
    assert _nets(done) == 0


# --- leaving after the hand ---------------------------------------------------------------------


def test_leaving_mid_hand_plays_the_hand_out_then_cashes_out(cast, hg):
    p = cast["p"]
    gid = _table(cast, 3)
    _ok(_post(p[0], gid, "run", {"running": True}))
    s = _state(p[0], gid)
    leaver = (s["actor"] + 1) % 3
    r = _ok(_post(p[leaver], gid, "leave"))
    seat = r["seats"][leaver]
    assert seat["leaving"] is True and seat["pending_remove"] is True
    assert seat["sitting_out"] is False, "they are still IN the hand — nobody folded them"
    assert seat["in_hand"] is True
    assert any(e["kind"] == "leave" and "after this hand" in e["text"] for e in r["events"])
    # they can change their mind while the hand is on
    assert _ok(_post(p[leaver], gid, "stay"))["seats"][leaver]["leaving"] is False
    _ok(_post(p[leaver], gid, "leave"))
    # …and they act for themselves as normal until the hand ends
    acted = False
    for _ in range(40):
        cl, st = _actor(cast, gid)
        if cl is None:
            break
        if st["actor"] == leaver:
            acted = True
        _ok(_post(cl, gid, "act", {"gate": "check_call", "hand_no": st["hand_no"], "action_seq": st["action_seq"]}))
    assert acted
    _finish_runout(hg, gid)
    done = _state(p[0], gid)
    assert done["seats"][leaver]["empty"] is True
    row = next(x for x in done["ledger"] if x["name"] == NAMES[leaver])
    assert row["seated"] is False and _nets(done) == 0
    # …and the next hand is dealt without them (manual dealing here)
    nxt = _ok(_post(p[0], gid, "deal", {}))
    assert nxt["phase"] == "in_hand" and nxt["seats"][leaver]["empty"] is True


def test_a_leaver_whose_browser_is_gone_does_not_stall_the_table(cast, hg):
    p = cast["p"]
    gid = _table(cast, 2, decision_secs=0)
    _ok(_post(p[0], gid, "run", {"running": True}))
    s = _state(p[0], gid)
    actor = s["actor"]
    _ok(_post(p[actor], gid, "leave"))
    t = hg.HUB.get(gid)
    with t.lock:
        t.seen[t.seats[actor].user_id] -= 600.0  # tab closed a while ago
        seq = t.action_seq
        hg._timeout_tick_locked(t)
        assert t.action_seq > seq, "the watchdog acts for a leaver who is no longer here"


# --- the award script names its pots ---------------------------------------------------------------


def test_award_steps_pay_from_named_pots_deepest_side_pot_first(hg):
    from plo5bp.ui import runout
    # three all-in stacks: 100 / 300 / 600 chips -> main pot, side pot 1 (two players), side pot 2 (the deepest two: one player => uncontested)
    commit = [100, 300, 600, 0]
    folded = [False, False, False, True]
    holes = [[0, 5, 10, 15, 20], [1, 6, 11, 16, 21], [2, 7, 12, 17, 22], None]
    board_a, board_b = [3, 8, 13, 18, 23], [4, 9, 14, 19, 24]
    awards = runout.build_awards(holes, folded, commit, board_a, board_b, 0)
    layers = [ly for ly in runout.pot_layers(commit, folded) if ly["chips"] > 0 and ly["eligible"]]
    hg._assign_award_pots(awards, layers)
    labels = ["Main pot" if k == len(layers) - 1 else f"Side pot {len(layers) - 1 - k}" for k in range(len(layers))]
    assert labels == ["Side pot 2", "Side pot 1", "Main pot"]
    assert [ly["chips"] for ly in layers] == [300, 400, 300]
    seq = [(labels[a["pot"]], a["board"]) for a in awards]
    assert seq == [("Side pot 2", "a"), ("Side pot 2", "b"), ("Side pot 1", "a"), ("Side pot 1", "b"), ("Main pot", "a"), ("Main pot", "b")]
    for k, ly in enumerate(layers):
        assert sum(a["chips"] for a in awards if a["pot"] == k) == ly["chips"], "every pot is paid out exactly"


def test_the_live_payload_carries_the_pots_while_the_runout_plays(cast, hg):
    p = cast["p"]
    gid = _table(cast, 3)
    _ok(_post(p[0], gid, "run", {"running": True}))
    s = _check_down(cast, gid)
    assert s["phase"] == "showdown" and s["runout"]["active"]
    assert s["pots"] and s["pots"][-1]["label"] == "Main pot" and len(s["pots"]) == 1
    assert sum(x["chips"] for x in s["pots"]) == 3 * 30000
    _finish_runout(hg, gid)
    done = _state(p[0], gid)
    for a in done["pot_awards"]:
        assert a["pot"] == 0
