"""Home games — tracking (2026-09-23): who-paid-whom money flows, background AI
grading of every action, replayable hand records, the lifetime hand database.

`money_flows` is checked two ways: hand-built scenarios with known answers
(fold-out, scoop + chop, quartering, side pots, dead money) and a randomized
sweep against the ENGINE's own payouts — every player's flows must add up to
exactly what the engine paid them.
"""

from __future__ import annotations

import json
import random
import sys

import pytest
from starlette.testclient import TestClient

from plo5bp.ui.runout import money_flows

ADMIN_EMAIL = "admin@tables.example"  # this module's own admin (PLO5BP_ADMIN_EMAILS below)
NAMES = ["me", "jeff", "bob", "bill"]

# card ints: rank * 4 + suit (c d h s), rank 0 = deuce ... 12 = ace
A, K, Q, J, T = 12, 11, 10, 9, 8


def c(rank, suit):
    return rank * 4 + suit


def _net(flows, n):
    out = [0] * n
    for (a, b), v in flows.items():
        out[a] -= v
        out[b] += v
    return out


# --- the arithmetic ---------------------------------------------------------------


def test_fold_out_everyone_pays_the_winner_their_ante():
    # ante 10 each; seat 0 bets 20 (uncalled), everyone folds
    flows = money_flows([30, 10, 10, 10], [False, True, True, True],
                        [None] * 4, [], [], button=0)
    assert flows == {(1, 0): 10, (2, 0): 10, (3, 0): 10}, "the uncalled 20 is nobody's loss"


# Two dry boards (no flush, no straight possible for the hands below):
#   A: Kc Kd 8h 3s 2c      B: Qc Qd 7h 4s 2d
BOARD_A = [c(K, 0), c(K, 1), c(6, 2), c(1, 3), c(0, 0)]
BOARD_B = [c(Q, 0), c(Q, 1), c(5, 2), c(2, 3), c(0, 1)]
ACES_1 = [c(A, 0), c(A, 1), c(7, 3), c(4, 3), c(3, 2)]      # Ac Ad 9s 6s 5h  -> aces up on both
ACES_2 = [c(A, 2), c(A, 3), c(7, 1), c(4, 1), c(3, 3)]      # Ah As 9d 6d 5s  -> the same hand
AIR = [c(J, 0), c(T, 1), c(7, 2), c(4, 0), c(3, 1)]         # Jc Td 9h 6c 5d  -> one pair (the board's)


def test_three_way_all_in_one_scooped_two_chop():
    flows = money_flows([100, 100, 100], [False] * 3, [ACES_1, ACES_2, AIR],
                        BOARD_A, BOARD_B, button=0)
    assert flows == {(2, 0): 50, (2, 1): 50}, "the choppers owe each other nothing"
    assert _net(flows, 3) == [50, 50, -100]


def test_quartering_scooped_quartered_and_three_quarters():
    # X wins board A alone (aces up) and TIES board B with Y (both trip queens, ace kicker).
    x = [c(A, 0), c(A, 1), c(Q, 2), c(J, 3), c(7, 3)]       # Ac Ad Qh Js 9s
    y = [c(Q, 3), c(A, 2), c(T, 3), c(4, 3), c(3, 2)]       # Qs Ah Ts 6s 5h
    flows = money_flows([100, 100, 100], [False] * 3, [x, y, AIR], BOARD_A, BOARD_B, button=0)
    # pot 300: X takes 150 + 75 (+125), Y 75 (-25), Z nothing (-100)
    assert _net(flows, 3) == [125, -25, -100]
    assert flows == {(1, 0): 50, (2, 0): 75, (2, 1): 25}


def test_side_pots_and_dead_money_follow_their_own_layer():
    # seat 3 folded 40 of dead money; seat 0 is all-in for 100 with the best hand;
    # seats 1 and 2 play a 300-each side pot that seat 1 wins.
    second = [c(A, 2), c(J, 3), c(T, 3), c(5, 3), c(2, 2)]  # Ah Js Ts 7s 4h: beats AIR, loses to aces up
    flows = money_flows([100, 400, 400, 40], [False, False, False, True],
                        [ACES_1, second, AIR, None], BOARD_A, BOARD_B, 0)
    assert _net(flows, 4) == [240, 200, -400, -40]
    assert flows[(3, 0)] == 40, "folded chips go to whoever won the layer they sit in"
    assert flows[(2, 1)] == 300 and flows[(2, 0)] == 100 and flows[(1, 0)] == 100


def test_flows_always_add_up_to_the_engines_payouts():
    from plo5bp.config import VARIANT_PLO5, GameConfig
    from plo5bp.env import BombPotEnv

    rng = random.Random(20260923)
    checked = 0
    for trial in range(60):
        n = rng.choice([2, 3, 4, 5, 6])
        stacks = tuple(rng.choice([40_000, 90_000, 150_000, 400_000, 700_000]) for _ in range(n))
        cfg = GameConfig(num_seats=n, starting_stack=0, starting_stacks=stacks,
                         ante=30_000, bb=10_000, variant=VARIANT_PLO5)
        env = BombPotEnv(cfg, ev_runout_samples=0, obs_mode="minimal")
        _, info = env.reset(rng.getrandbits(62), rng.randrange(n))
        for _ in range(200):
            if env.is_terminal():
                break
            gm = info.gate_mask
            r = rng.random()
            if gm[2] and r < 0.45:
                lo, hi = int(info.min_raise_chips), int(info.max_raise_chips)
                chips = hi if rng.random() < 0.5 or lo >= hi else rng.randint(lo, hi)
                _, _, _, info = env.step_hybrid(2, chips)
            elif gm[0] and r > 0.8:
                _, _, _, info = env.step_hybrid(0, 0)
            else:
                _, _, _, info = env.step_hybrid(1, 0)
        assert env.is_terminal()
        raw = dict(env._rs.observation_dict())
        folded = [bool(x) for x in raw["folded"]]
        holes = [None if folded[i] else [int(x) for x in h]
                 for i, h in enumerate(env.all_hole_cards())]
        flows = money_flows([int(x) for x in raw["total_commit"]], folded, holes,
                            [int(x) for x in raw["board_a"]], [int(x) for x in raw["board_b"]],
                            int(raw["button"]) if "button" in raw else 0)
        payouts = [int(x) for x in env._rs.payouts()]
        net = _net(flows, n)
        assert sum(net) == 0
        # odd chips of a split pot may land on a different seat than the engine's
        # (left-of-button rule): never more than a couple of chips (1 chip = $0.0001)
        assert all(abs(a - b) <= 4 for a, b in zip(net, payouts)), (trial, net, payouts)
        checked += 1
    assert checked == 60


# --- the app: grading, replay records, lifetime database -------------------------------


@pytest.fixture(scope="module")
def server(boot_public_server):
    return boot_public_server(PLO5BP_HOMEGAME_GRADING="1", PLO5BP_ADMIN_EMAILS=ADMIN_EMAIL)


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
    return {"p": players, "by_uid": {ids[f"{n}@example.com"]: players[i] for i, n in enumerate(NAMES)}}


def _post(cl, gid, what, body=None):
    return cl.post(f"/games/api/tables/{gid}/{what}", json=body or {})


def _state(cl, gid):
    return cl.get(f"/games/api/tables/{gid}").json()


def _table(cast, n, **kw):
    body = {"name": "tracking", "sb_cents": 50, "bb_cents": 100, "ante_cents": 1000,
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
    winner = s["actor"]
    assert _post(cl, gid, "act", {"gate": "raise", "raise_to_chips":
                                  s["raise_bounds"]["min_chips"] + s["street_commit_chips"]}).status_code == 200
    for _ in range(12):
        cl, s = _actor(cast, gid)
        if cl is None:
            break
        _post(cl, gid, "act", {"gate": "fold" if s["legal"]["fold"] else "check_call"})
    return winner


def test_the_owners_example_everyone_pays_ten(cast, hg):
    """Ante $10 four-handed, one bet, three folds: +$10 from each of the three."""
    p = cast["p"]
    gid = _table(cast, 4)
    assert _post(p[0], gid, "run", {"running": True}).status_code == 200
    winner = _bet_and_fold_out(cast, gid)
    det = p[winner].get(f"/games/api/tables/{gid}/hands/1").json()
    assert sorted((f["from"], f["to"], f["cents"]) for f in det["flows"]) == sorted(
        (i, winner, 1000) for i in range(4) if i != winner)
    assert "seed" not in str(det) and "replay" not in det, "the deal seed must never be served"
    # every action can be replayed: who, what, how much, engine chips, whose decision
    a0 = det["actions"][0]
    assert a0["action"] not in (0, 1) and a0["chips"] > 0 and a0["auto"] is False
    assert all(s["start_chips"] == 2_000_000 for s in det["seats"]) and det["ante_chips"] == 100_000
    # table head-to-head + the winner's lifetime view
    h2h = p[0].get(f"/games/api/tables/{gid}/hands").json()["h2h"]
    wname = NAMES[winner]
    assert sorted((x["from"], x["to"], x["cents"]) for x in h2h) == sorted(
        (NAMES[i], wname, 1000) for i in range(4) if i != winner)
    mine = p[winner].get("/games/api/my/stats").json()
    vs = {v["name"]: v["net_cents"] for v in mine["versus"]}
    assert all(vs[NAMES[i]] == 1000 for i in range(4) if i != winner)
    loser = next(i for i in range(4) if i != winner)
    theirs = p[loser].get("/games/api/my/stats").json()
    assert {v["name"]: v["net_cents"] for v in theirs["versus"]}[wname] == -1000


def test_every_player_decision_is_graded_in_the_background(cast, hg):
    p = cast["p"]
    gid = _table(cast, 3)
    _post(p[0], gid, "run", {"running": True})
    _bet_and_fold_out(cast, gid)
    assert hg.wait_for_grading(30.0), "the grader never finished"
    raw = _stored(hg, gid, 1)
    assert {g["i"] for g in raw["grades"]} == set(range(len(raw["actions"]))), "one grade per player action"
    for g in raw["grades"]:
        assert 0.0 <= g["score"] <= 100.0
        assert g["cat"] in ("best", "correct", "inaccuracy", "wrong", "blunder")
        assert g["seat"] == raw["actions"][g["i"]]["seat"]
    # nobody reached showdown: everyone sees the marks on their own decisions only
    for i in range(3):
        det = p[i].get(f"/games/api/tables/{gid}/hands/1").json()
        assert det["grades_public"] is True
        assert det["grades"] and {g["seat"] for g in det["grades"]} == {i}
    lst = p[0].get(f"/games/api/tables/{gid}/hands").json()
    assert lst["hands"][0]["my_accuracy"] is not None
    assert all(r["accuracy"] is not None and r["graded"] >= 1 for r in lst["stats"])
    me = p[0].get("/games/api/my/stats").json()
    assert me["accuracy"] is not None and me["graded"] >= 1


def test_grades_can_be_limited_to_your_own_actions(cast, hg):
    p = cast["p"]
    gid = _table(cast, 2)
    assert _post(p[0], gid, "settings", {"show_grades": False}).json()["settings"]["show_grades"] is False
    _post(p[0], gid, "run", {"running": True})
    _bet_and_fold_out(cast, gid)
    assert hg.wait_for_grading(30.0)
    for i in (0, 1):
        det = p[i].get(f"/games/api/tables/{gid}/hands/1").json()
        assert det["grades_public"] is False
        assert det["grades"] and all(g["seat"] == i for g in det["grades"])


def _stored(hg, gid, hand_no):
    row = hg.pub.DB.one("SELECT summary FROM homegame_hands WHERE game_id=? AND hand_no=?", (gid, hand_no))
    return json.loads(row["summary"])


def _showdown_with_a_fold(cast, gid, folder):
    """The first other player to act bets the minimum, `folder` folds to it, the
    rest call and check it down: a showdown with one mucked hand."""
    bet = False
    for _ in range(60):
        cl, s = _actor(cast, gid)
        if cl is None:
            return
        if s["actor"] == folder:
            _post(cl, gid, "act", {"gate": "fold" if bet and s["legal"]["fold"] else "check_call"})
        elif not bet:
            assert _post(cl, gid, "act", {"gate": "raise", "raise_to_chips":
                                          s["raise_bounds"]["min_chips"] + s["street_commit_chips"]}).status_code == 200
            bet = True
        else:
            _post(cl, gid, "act", {"gate": "check_call"})


def test_grades_follow_the_cards_a_tabled_hand_yes_a_mucked_hand_no(cast, hg):
    """(owner, 2026-09-26) You see the network's marks on your own decisions and
    on hands that reached showdown — never on a hand that was mucked, in the
    replayer or in anyone's history — and the hidden marks still count."""
    p = cast["p"]
    gid = _table(cast, 3)
    _post(p[0], gid, "run", {"running": True})
    _showdown_with_a_fold(cast, gid, folder=2)
    t = hg.HUB.get(gid)
    with t.lock:  # skip the runout's reveal pauses
        if t.runout_active and t.runout_started_mono is not None:
            t.runout_started_mono -= 600.0
    assert hg.wait_for_grading(30.0)
    raw = _stored(hg, gid, 1)
    assert raw["showdown"] and {s["seat"] for s in raw["seats"] if s["shown"]} == {0, 1}
    assert {g["seat"] for g in raw["grades"]} == {0, 1, 2}

    def marks(i):
        return {g["seat"] for g in p[i].get(f"/games/api/tables/{gid}/hands/1").json()["grades"]}

    assert marks(0) == {0, 1} and marks(1) == {0, 1}, "bob mucked: his marks stay hidden"
    assert marks(2) == {0, 1, 2}, "your own marks, plus the hands that were tabled"
    ids = {x["name"]: x["user_id"] for x in p[0].get("/games/api/community").json()["players"]}
    theirs = p[0].get(f"/games/api/players/{ids['bob']}/hands", params={"game": gid}).json()["hands"][0]
    assert theirs["my_hole"] is None and theirs["accuracy"] is None, "no cards, no marks"
    for key in ("accuracy", "time"):  # a sort can't dig the hidden marks up either
        assert p[0].get(f"/games/api/players/{ids['bob']}/hands", params={"sort": key}).json()["hands"][0]["accuracy"] is None
    own = p[2].get(f"/games/api/players/{ids['bob']}/hands", params={"game": gid}).json()["hands"][0]
    assert own["my_hole"] and own["accuracy"] is not None
    tabled = p[2].get(f"/games/api/players/{ids['me']}/hands", params={"game": gid}).json()["hands"][0]
    assert tabled["my_hole"] and tabled["accuracy"] is not None, "a tabled hand keeps both"
    st = p[0].get(f"/games/api/players/{ids['bob']}/stats").json()
    assert st["graded"] >= 1 and st["accuracy"] is not None, "the hidden marks still count"


def test_clock_actions_are_recorded_but_not_graded(cast, hg):
    p = cast["p"]
    gid = _table(cast, 2, decision_secs=10)
    _post(p[0], gid, "run", {"running": True})
    t = hg.HUB.get(gid)
    for _ in range(12):
        with t.lock:
            if t.phase != "in_hand":
                break
            t.turn_started_mono -= 11.0
            hg._timeout_tick_locked(t)
            if t.bank_started_mono is not None:
                t.bank_started_mono -= 600.0
                hg._timeout_tick_locked(t)
    with t.lock:  # a checked-down showdown still steps through its awards
        if t.runout_active and t.runout_started_mono is not None:
            t.runout_started_mono -= 600.0
    assert hg.wait_for_grading(30.0)
    det = p[0].get(f"/games/api/tables/{gid}/hands/1").json()
    assert det["actions"] and all(a["auto"] is True for a in det["actions"])
    assert det["grades"] in ([], None), "the clock's decisions are not the player's"


def test_lifetime_database_sorts_filters_and_stays_private(cast, hg):
    p = cast["p"]
    g1 = _table(cast, 2)
    _post(p[0], g1, "run", {"running": True})
    for _ in range(3):
        _bet_and_fold_out(cast, g1)
        _post(p[0], g1, "deal", {})
    g2 = _table(cast, 2)
    _post(p[0], g2, "run", {"running": True})
    _bet_and_fold_out(cast, g2)
    assert hg.wait_for_grading(40.0)

    def hands(cl, **q):
        r = cl.get("/games/api/my/hands", params=q)
        assert r.status_code == 200, r.text
        return r.json()

    allh = hands(p[0], limit=50)
    assert allh["total"] >= 4 and len(allh["hands"]) >= 4
    assert {h["game_id"] for h in allh["hands"]} >= {g1, g2}
    only = hands(p[0], game=g2)
    assert only["total"] == 1 and only["hands"][0]["table_name"] == "tracking"
    for key, field in (("pot", "pot_cents"), ("net", "net_cents"), ("accuracy", "accuracy"), ("time", "ended_at")):
        for direction in ("asc", "desc"):
            vals = [h[field] for h in hands(p[0], sort=key, dir=direction, limit=50)["hands"] if h[field] is not None]
            assert vals == sorted(vals, reverse=(direction == "desc")), (key, direction)
    h = allh["hands"][0]
    assert h["my_hole"] and len(h["my_hole"]) == 5 and len(h["board_a"]) >= 3
    # a player who never played sees nothing; nobody's list carries another user's hands
    assert hands(p[3])["total"] == 0 or all(x["game_id"] not in (g1, g2) for x in hands(p[3])["hands"])
    stats = p[0].get("/games/api/my/stats").json()
    sess = {s["id"]: s for s in stats["sessions"]}
    assert sess[g1]["hands"] >= 3 and sess[g2]["hands"] == 1
    assert stats["hands"] >= 4 and isinstance(stats["net_cents"], int)
    anon = TestClient(p[0].app, raise_server_exceptions=False)
    assert anon.get("/games/api/my/hands").status_code == 404
    assert anon.get("/games/api/my/stats").status_code == 404


# --- the club: everyone's stats, anyone's history, excluded sessions ---------------------


def test_the_club_sees_everyone_but_never_a_mucked_hand(cast, hg):
    p = cast["p"]
    gid = _table(cast, 3)
    _post(p[0], gid, "run", {"running": True})
    winner = _bet_and_fold_out(cast, gid)
    assert hg.wait_for_grading(30.0)
    club = p[2].get("/games/api/community").json()
    by_name = {x["name"]: x for x in club["players"]}
    assert {"me", "jeff", "bob"} <= set(by_name)
    assert club["is_admin"] is False
    assert sum(x["is_me"] for x in club["players"]) == 1
    assert all("email" not in x for x in club["players"])
    ids = {x["name"]: x["user_id"] for x in club["players"]}
    # the money between every pair is there for everybody
    assert any(pr["to"] == ids[NAMES[winner]] for pr in club["pairs"])
    assert any(s["id"] == gid and s["excluded"] is False for s in club["sessions"])
    # bob browses the winner's history: numbers yes, the uncontested winner's cards no
    other = p[2] if winner != 2 else p[1]
    lst = other.get(f"/games/api/players/{ids[NAMES[winner]]}/hands", params={"game": gid}).json()
    assert lst["total"] == 1
    row = lst["hands"][0]
    assert row["net_cents"] > 0 and row["my_hole"] is None, "a fold-out winner stays face-down"
    mine = p[winner].get(f"/games/api/players/{ids[NAMES[winner]]}/hands", params={"game": gid}).json()
    assert mine["hands"][0]["my_hole"] and len(mine["hands"][0]["my_hole"]) == 5
    st = other.get(f"/games/api/players/{ids[NAMES[winner]]}/stats").json()
    assert st["name"] == NAMES[winner] and st["hands"] >= 1
    assert other.get("/games/api/players/999999/stats").status_code == 404


def test_admin_can_take_a_test_session_out_of_the_record_and_put_it_back(cast, hg, server):
    p = cast["p"]
    adm = TestClient(server.app, raise_server_exceptions=False)
    assert adm.get("/auth/dev", params={"email": ADMIN_EMAIL}).status_code == 200
    gid = _table(cast, 2)
    _post(p[0], gid, "run", {"running": True})
    _bet_and_fold_out(cast, gid)

    def hands_in(cl):
        return cl.get("/games/api/my/hands", params={"game": gid}).json()["total"]

    before = p[0].get("/games/api/my/stats").json()["hands"]
    assert hands_in(p[0]) == 1
    assert _post(p[0], gid, "exclude", {"on": True}).status_code == 403, "the host is not the site admin"
    r = _post(adm, gid, "exclude", {"on": True})
    assert r.status_code == 200 and r.json()["excluded"] is True
    assert _state(p[0], gid)["status"] == "closed", "an excluded table leaves the lobby"
    assert hands_in(p[0]) == 0
    assert p[0].get("/games/api/my/stats").json()["hands"] == before - 1
    assert all(s["id"] != gid for s in p[0].get("/games/api/community").json()["sessions"])
    assert any(s["id"] == gid and s["excluded"] for s in adm.get("/games/api/community").json()["sessions"])
    assert all(s["id"] != gid for s in p[0].get("/games/api/tables").json()["sessions"])
    # nothing was deleted: the hand is still there, and the session can come back
    assert p[0].get(f"/games/api/tables/{gid}/hands/1").status_code == 200
    assert _post(adm, gid, "exclude", {"on": False}).status_code == 200
    assert hands_in(p[0]) == 1
    assert p[0].get("/games/api/my/stats").json()["hands"] == before
