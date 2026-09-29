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
    layers = runout.display_pots(commit, folded)
    labels = ["Main pot" if k == len(layers) - 1 else f"Side pot {len(layers) - 1 - k}" for k in range(len(layers))]
    assert labels == ["Side pot 2", "Side pot 1", "Main pot"]
    assert [ly["chips"] for ly in layers] == [300, 400, 300]
    seq = [(labels[a["pot"]], a["board"]) for a in awards]
    assert seq == [("Side pot 2", "a"), ("Side pot 2", "b"), ("Side pot 1", "a"), ("Side pot 1", "b"), ("Main pot", "a"), ("Main pot", "b")]
    for k, ly in enumerate(layers):
        assert sum(a["chips"] for a in awards if a["pot"] == k) == ly["chips"], "every pot is paid out exactly"


def test_a_fold_is_dead_money_in_the_pot_not_a_side_pot(hg):
    """Two players fold after the antes, four see a bet through: that is ONE pot.
    The layer math used to name the folders' ante level "Main pot" and the rest
    "Side pot 1", and the caption said "Priya wins $9.00" for a $27 board."""
    from plo5bp.ui import runout
    commit = [1200, 300, 1200, 1200, 1200, 300]
    folded = [False, True, False, False, False, True]
    holes = [[0, 5, 10, 15, 20], None, [1, 6, 11, 16, 21], [2, 7, 12, 17, 22], [26, 30, 34, 38, 42], None]
    board_a, board_b = [3, 8, 13, 18, 23], [4, 9, 14, 19, 24]
    pots = runout.display_pots(commit, folded)
    assert len(pots) == 1 and pots[0]["chips"] == sum(commit) and pots[0]["eligible"] == [0, 2, 3, 4]
    awards = runout.build_awards(holes, folded, commit, board_a, board_b, 0)
    assert [(a["board"], a["pot"]) for a in awards] == [("a", 0), ("b", 0)]
    assert [a["chips"] for a in awards] == [sum(commit) // 2, sum(commit) - sum(commit) // 2]
    # an all-in for less still makes a real side pot; the folds stay dead money in the main pot
    commit = [600, 300, 1200, 1200]
    folded = [False, True, False, False]
    pots = runout.display_pots(commit, folded)
    assert [(p["chips"], p["eligible"]) for p in pots] == [(1200, [2, 3]), (300 * 4 + 300 * 3, [0, 2, 3])]


def test_merged_pots_pay_every_seat_exactly_what_the_layers_do(hg):
    """Merging is presentation only: per seat, the steps add up to the same chips
    as paying each layer on its own (odd chips included), across random hands."""
    import random

    from plo5bp.ui import runout
    rng = random.Random(20260925)
    for _ in range(300):
        n = rng.randint(2, 6)
        deck = list(range(52))
        rng.shuffle(deck)
        holes = [deck[5 * i:5 * i + 5] for i in range(n)]
        board_a, board_b = deck[5 * n:5 * n + 5], deck[5 * n + 5:5 * n + 10]
        commit = [rng.choice([300, 301, 450, 900, 1333, 2000]) for _ in range(n)]
        folded = [rng.random() < 0.35 for _ in range(n)]
        if all(folded):
            folded[0] = False
        want = [0] * n
        for ly in runout.pot_layers(commit, folded):  # the old, one-layer-at-a-time payout
            elig = ly["eligible"]
            if ly["chips"] <= 0 or not elig or sum(1 for f in folded if not f) <= 1:
                continue
            for board, half in ((board_a, ly["chips"] // 2), (board_b, ly["chips"] - ly["chips"] // 2)):
                winners = list(elig) if len(elig) == 1 else (runout._winners_on_board(holes, elig, board)[0] or list(elig))
                for s, v in runout._distribute(half, winners, 0, n).items():
                    want[s] += v
        if sum(1 for f in folded if not f) <= 1:
            continue
        got = [0] * n
        for a in runout.build_awards(holes, folded, commit, board_a, board_b, 0):
            for s, v in a["shares"].items():
                got[int(s)] += int(v)
        assert got == want, (commit, folded)


def test_a_bet_a_fold_and_a_showdown_show_one_pot(cast, hg):
    p = cast["p"]
    gid = _table(cast, 4)
    _ok(_post(p[0], gid, "run", {"running": True}))
    cl, s = _actor(cast, gid)
    _ok(_post(cl, gid, "act", {"gate": "raise", "chips": s["raise_bounds"]["min_chips"],
                               "hand_no": s["hand_no"], "action_seq": s["action_seq"]}))
    cl, s = _actor(cast, gid)
    _ok(_post(cl, gid, "act", {"gate": "fold", "hand_no": s["hand_no"], "action_seq": s["action_seq"]}))
    s = _check_down(cast, gid)
    assert s["phase"] == "showdown" and s["runout"]["active"]
    assert len(s["pots"]) == 1 and s["pots"][0]["label"] == "Main pot"
    assert s["pots"][0]["chips"] == 4 * 30000 + 3 * 10000, "the folder's ante is in the one pot"
    _finish_runout(hg, gid)
    done = _state(p[0], gid)
    assert {a["pot"] for a in done["pot_awards"]} == {0} and len(done["pot_awards"]) == 2
    assert sum(done["hand_deltas_cents"]) == 0


def test_closing_a_table_keeps_its_host(cast, hg):
    """Closing cashes every seat out — it used to hand the host role round the
    table as it went, so the finished session was "hosted by" another player
    (and an "X is now the host" toast popped up at the close)."""
    p = cast["p"]
    gid = _table(cast, 3)
    host_before = _state(p[0], gid)["host_user_id"]
    _ok(_post(p[0], gid, "close"))
    t = hg.HUB.get(gid)
    assert t.status == "closed" and t.host_user_id == host_before
    row = hg.pub.DB.one("SELECT host_user_id FROM homegames WHERE id=?", (gid,))
    assert int(row["host_user_id"]) == host_before
    assert not any("is now the host" in e["text"] for e in t.events)
    assert sum(r["net_cents"] for r in _state(p[0], gid)["ledger"]) == 0


def test_a_restart_mid_hand_voids_it_and_says_so(cast, hg):
    """A deploy or crash mid-hand: the hand never happened (stacks are saved only
    when a hand ends), the table comes back paused — and the players are TOLD,
    instead of watching the hand vanish."""
    p = cast["p"]
    gid = _table(cast, 3)
    before = {x["seat"]: x["stack_cents"] for x in _state(p[0], gid)["seats"] if not x["empty"]}
    _ok(_post(p[0], gid, "run", {"running": True}))
    cl, s = _actor(cast, gid)
    _ok(_post(cl, gid, "act", {"gate": "raise", "chips": s["raise_bounds"]["min_chips"],
                               "hand_no": s["hand_no"], "action_seq": s["action_seq"]}))
    hand = s["hand_no"]
    with hg.HUB._lock:
        hg.HUB._tables.pop(gid)  # the process is gone; the next request reloads from the DB
    s = _state(p[0], gid)
    assert s["phase"] == "waiting" and s["running"] is False
    assert {x["seat"]: x["stack_cents"] for x in s["seats"] if not x["empty"]} == before
    assert any(f"Hand #{hand} was cut short" in e["text"] for e in s["events"])
    # a clean reload (no hand in the air) says nothing
    _ok(_post(p[0], gid, "run", {"running": True}))
    s = _check_down(cast, gid)
    _finish_runout(hg, gid)
    _state(p[0], gid)
    with hg.HUB._lock:
        t = hg.HUB._tables.pop(gid)
    with t.lock:
        t.running = False
        hg._persist_meta(t)
    s = _state(p[0], gid)
    assert not any("cut short" in e["text"] for e in s["events"])


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


# --- the pots while the hand is played (2026-09-25) ------------------------------------------

B = 10000  # chips per big blind


def test_live_pots_split_what_is_already_in_the_middle(hg):
    """Side pots used to appear only at the showdown. ``live_pots`` splits the
    chips already in the middle (antes + finished streets) the way the showdown
    will pay them; this street's bets stay out until the round closes."""
    raw = {"total_commit": [8 * B, 13 * B, 13 * B], "street_commit": [0, 5 * B, 5 * B], "folded": [False] * 3}
    pots = hg._live_pots(raw, [True] * 3, 3)
    assert [(p["label"], p["chips"], p["eligible"]) for p in pots] == [("Main pot", 24 * B, [0, 1, 2])]
    # the short stack is all-in for 8, the other two put in 18: a side pot for them
    raw = {"total_commit": [18 * B, 8 * B, 18 * B], "street_commit": [0, 0, 0], "folded": [False] * 3}
    pots = hg._live_pots(raw, [True] * 3, 3)
    assert [(p["label"], p["chips"], p["eligible"]) for p in pots] == [
        ("Side pot 1", 20 * B, [0, 2]), ("Main pot", 24 * B, [0, 1, 2])]
    # a fold is dead money in the pots it reached — never a pot of its own
    raw = {"total_commit": [18 * B, 8 * B, 18 * B, 12 * B], "street_commit": [0] * 4,
           "folded": [False, False, False, True]}
    pots = hg._live_pots(raw, [True] * 4, 4)
    assert [(p["label"], p["chips"], p["eligible"]) for p in pots] == [
        ("Side pot 1", 24 * B, [0, 2]), ("Main pot", 32 * B, [0, 1, 2])]
    # a seat not dealt in can never be in a pot
    raw = {"total_commit": [8 * B, 8 * B, 0], "street_commit": [0] * 3, "folded": [False] * 3}
    assert hg._live_pots(raw, [True, True, False], 3)[0]["eligible"] == [0, 1]


def test_a_side_pot_shows_as_soon_as_the_betting_goes_past_an_all_in(cast, hg):
    p = cast["p"]
    body = {"name": "ux", "bb_cents": 100, "ante_cents": 300, "default_buyin_cents": 4000}
    gid = _ok(p[0].post("/games/api/tables", json=body))["id"]
    _ok(_post(p[1], gid, "sit", {"seat": 1, "buyin_cents": 800}))  # 8 bb: all-in after the ante + 5
    _ok(_post(p[2], gid, "sit", {"seat": 2, "buyin_cents": 4000}))
    _ok(_post(p[0], gid, "run", {"running": True}))
    s = _state(p[0], gid)
    assert [(x["label"], x["chips"]) for x in s["live_pots"]] == [("Main pot", 9 * B)]  # the antes
    # flop: the short stack shoves, the others call
    for _ in range(12):
        cl, s = _actor(cast, gid)
        if s["street"] != "flop":
            break
        body = {"hand_no": s["hand_no"], "action_seq": s["action_seq"], "gate": "check_call"}
        if s["actor"] == 1 and s["legal"]["raise"]:
            body.update(gate="raise", chips=s["raise_bounds"]["max_chips"])
        _ok(_post(cl, gid, "act", body))
    assert s["street"] == "turn" and s["seats"][1]["all_in"]
    assert [(x["label"], x["chips"], x["eligible"]) for x in s["live_pots"]] == [("Main pot", 24 * B, [0, 1, 2])]
    # turn: a bet and a call past the all-in
    cl, s = _actor(cast, gid)
    _ok(_post(cl, gid, "act", {"gate": "raise", "chips": s["raise_bounds"]["min_chips"],
                               "hand_no": s["hand_no"], "action_seq": s["action_seq"]}))
    bet = s["raise_bounds"]["min_chips"]
    cl, s = _actor(cast, gid)
    assert [x["chips"] for x in s["live_pots"]] == [24 * B], "a bet in front of a player is in no pot yet"
    _ok(_post(cl, gid, "act", {"gate": "check_call", "hand_no": s["hand_no"], "action_seq": s["action_seq"]}))
    s = _state(p[0], gid)
    assert s["street"] == "river"
    live = [(x["label"], x["chips"], x["eligible"]) for x in s["live_pots"]]
    assert live == [("Side pot 1", 2 * bet, [0, 2]), ("Main pot", 24 * B, [0, 1, 2])]
    assert sum(x["chips"] for x in s["live_pots"]) == s["settled_pot_chips"]
    # the showdown pays exactly these pots, in the same places
    s = _check_down(cast, gid)
    assert s["phase"] == "showdown" and not s["live_pots"]
    assert [(x["label"], x["chips"], x["eligible"]) for x in s["pots"]] == live
    _finish_runout(hg, gid)
    assert sum(_state(p[0], gid)["hand_deltas_cents"]) == 0
