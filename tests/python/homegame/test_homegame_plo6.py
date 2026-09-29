"""Home games — PLO6 double-board bomb pots next to PLO5 (2026-09-26).

A table deals PLO5 (five hole cards, up to 8 seats) or PLO6 (six, up to 7 —
7 x 6 + two boards is the whole deck; no burn cards online), chosen when it is
created. Pinned here: the seat limit, the six-card deal and its privacy, the
showdown / award script with six-card hands, the verified shuffle's six-card
slot map, that PLO6 decisions are NEVER graded (there is no PLO6 network), and
that the club's numbers are kept apart per game.
"""
from __future__ import annotations

import sys

import pytest
from starlette.testclient import TestClient

from plo5bp.ui import fairdeal as fd

ADMIN_EMAIL = "admin@plo6.example"
NAMES = ["ann", "ben", "cat", "dov", "eve", "fay", "gus", "hal", "ida"]  # (hal + ida: the club test only)


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
    players = [login(f"{n}@plo6.example") for n in NAMES]
    ids = {u["email"]: u["id"] for u in adm.get("/admin/api/users").json()["users"]}
    for n in NAMES:  # (the main club: everyone's tables and numbers in one place)
        r = adm.post("/admin/api/games_access", json={"user_id": ids[f"{n}@plo6.example"], "action": "grant"})
        assert r.status_code == 200
    uid = [ids[f"{n}@plo6.example"] for n in NAMES]
    return {"p": players, "uid": uid, "by_uid": dict(zip(uid, players))}


# --- helpers -------------------------------------------------------------------------


def _post(cl, gid, what, body=None):
    return cl.post(f"/games/api/tables/{gid}/{what}", json=body or {})


def _state(cl, gid):
    r = cl.get(f"/games/api/tables/{gid}")
    assert r.status_code == 200, r.text
    return r.json()


def _create(cl, **kw):
    body = {"name": "six", "sb_cents": 50, "bb_cents": 100, "ante_cents": 300,
            "default_buyin_cents": 20000, **kw}
    return cl.post("/games/api/tables", json=body)


def _table(cast, n, **kw):
    r = _create(cast["p"][0], **kw)
    assert r.status_code == 200, r.text
    gid = r.json()["id"]
    for i in range(1, n):
        assert _post(cast["p"][i], gid, "sit", {"seat": i, "buyin_cents": 20000}).status_code == 200
    return gid


def _actor(cast, gid):
    s = _state(cast["p"][0], gid)
    if s["phase"] != "in_hand" or s["actor"] is None:
        return None, s
    cl = cast["by_uid"][s["seats"][s["actor"]]["user_id"]]
    return cl, _state(cl, gid)


def _act(cl, gid, s, **body):
    r = _post(cl, gid, "act", {"hand_no": s["hand_no"], "action_seq": s["action_seq"], **body})
    assert r.status_code == 200, r.text
    return r.json()


def _check_down(cast, gid):
    """Everybody checks / calls to the river: a showdown with every hand live."""
    for _ in range(60):
        cl, s = _actor(cast, gid)
        if cl is None:
            return s
        _act(cl, gid, s, gate="check_call")
    raise AssertionError("hand did not end")


def _jam_out(cast, gid):
    for _ in range(60):
        cl, s = _actor(cast, gid)
        if cl is None:
            return s
        if s["legal"]["raise"]:
            _act(cl, gid, s, gate="raise", raise_to_chips=s["raise_bounds"]["max_chips"] + s["street_commit_chips"])
        else:
            _act(cl, gid, s, gate="check_call")
    raise AssertionError("hand did not end")


def _finish_runout(hg, gid):
    t = hg.HUB.get(gid)
    with t.lock:
        if t.runout_active and t.runout_started_mono is not None:
            t.runout_started_mono -= 600.0


def _next_hand(cast, gid):
    r = _post(cast["p"][0], gid, "deal")
    assert r.status_code == 200, r.text
    return r.json()


# --- the table -----------------------------------------------------------------------------


def test_a_plo6_table_seats_at_most_seven_and_says_what_it_deals(cast, hg):
    host = cast["p"][0]
    six = _create(host, variant="plo6").json()
    assert six["variant"] == "plo6" and six["hole_count"] == 6 and six["num_seats"] == 7, "7-max by default"
    assert six["game"] == {"code": "plo6", "label": "PLO6", "name": "PLO6 double-board bomb pot", "hole": 6,
                           "dealt": 6, "burns": 0, "max_seats": 7, "graded": False}
    assert six["burns"] == [], "no face-up burns outside PLO67"
    five = _create(host).json()
    assert five["variant"] == "plo5" and five["hole_count"] == 5 and five["num_seats"] == 8, "PLO5 is unchanged"
    assert five["game"]["graded"] is True and five["game"]["max_seats"] == 8
    # 8 x 6 + 10 = 58 cards: one deck can't deal it
    bad = _create(host, variant="plo6", num_seats=8)
    assert bad.status_code == 400 and "2–7" in bad.json()["detail"]
    assert _create(host, variant="plo7").status_code == 400, "an unknown game is refused, never a silent PLO5"
    assert _create(host, variant="plo6_double_bomb", num_seats=6).json()["variant"] == "plo6"
    # the seat count can change later, within the game's own limit
    gid = six["id"]
    assert _post(host, gid, "settings", {"num_seats": 8}).status_code == 400
    r = _post(host, gid, "settings", {"num_seats": 5})
    assert r.status_code == 200 and r.json()["num_seats"] == 5 and r.json()["variant"] == "plo6"
    # the lobby says which game each table deals
    rows = {t["id"]: t for t in host.get("/games/api/tables").json()["tables"]}
    assert rows[gid]["variant"] == "plo6" and rows[five["id"]]["variant"] == "plo5"
    # and the host's next table starts from this one's game
    assert host.get("/games/api/host_prefs").json()["prefs"]["variant"] == "plo6"
    for t in (gid, five["id"]):  # (tidy: tables count against the host's limit)
        assert _post(host, t, "close").status_code == 200


def test_seven_players_are_dealt_six_cards_each_from_one_deck(cast, hg):
    p = cast["p"]
    gid = _table(cast, 7, variant="plo6")
    s = _post(p[0], gid, "run", {"running": True}).json()
    assert s["phase"] == "in_hand"
    t = hg.HUB.get(gid)
    with t.lock:
        holes = t.env.all_hole_cards()
    assert len(holes) == 7 and all(len(h) == 6 for h in holes)
    dealt = [c for h in holes for c in h]
    assert len(set(dealt)) == 42, "42 different hole cards"
    # every player sees exactly their own six, everyone else's six face down
    for i in range(7):
        me = _state(p[i], gid)
        assert me["my_seat"] == i and me["hole_count"] == 6
        for j, seat in enumerate(me["seats"]):
            if j == i:
                assert sorted(seat["hole"]) == sorted(holes[i]) and len(seat["hole"]) == 6
            else:
                assert seat["hole"] == [-1] * 6
    _check_down(cast, gid)
    _finish_runout(hg, gid)
    with t.lock:
        board = list(t.rabbit_full_a) + list(t.rabbit_full_b)
    assert len(board) == 10, "checked down to the river: both boards are complete"
    assert len(set(board) | set(dealt)) == 52, "no burn cards: seven players use the WHOLE deck"
    s = _state(p[0], gid)
    assert s["phase"] == "showdown"
    assert sum(1 for x in s["seats"] if x["hole"] and x["hole"][0] >= 0) == 7, "all seven hands are tabled"
    assert all(len(x["hole"]) == 6 for x in s["seats"])
    assert _post(p[0], gid, "run", {"running": False}).status_code == 200


def test_a_plo6_showdown_tables_six_cards_and_pays_exactly_what_the_engine_pays(cast, hg):
    p = cast["p"]
    gid = _table(cast, 3, variant="plo6", num_seats=6)
    assert _post(p[0], gid, "run", {"running": True}).json()["phase"] == "in_hand"
    _jam_out(cast, gid)
    t = hg.HUB.get(gid)
    with t.lock:
        awards = list(t.pot_awards)
        eq = dict(t.equity_by_len)
        holes = {i: sorted(h) for i, h in enumerate(t.env.all_hole_cards())}
        terminal_pot = int(t.terminal_pot)
        commit = list(t.terminal_commit)
        payouts = list(t.last_deltas)
        start_chips = list(t.hand_start_stacks)
        final_chips = [x.stack_chips if x is not None else 0 for x in t.seats]
    for shares in eq.values():  # (all in before the river: the equities on the badges)
        for v in shares.values():
            assert 0.0 <= v["a"] <= 1.0 and 0.0 <= v["b"] <= 1.0
    assert sum(sum(int(x) for x in a["shares"].values()) for a in awards) == terminal_pot, "every chip is awarded"
    # the award ANIMATION (runout.py, six-card hands) pays each seat exactly what the ENGINE paid
    for i in range(3):
        won = sum(int(a["shares"].get(str(i), 0)) for a in awards)
        assert won - commit[i] == payouts[i], (i, won, commit[i], payouts[i])
        assert final_chips[i] == start_chips[i] + payouts[i]
    for a in awards:
        for seat, combo in (a.get("combos") or {}).items():
            assert len(combo["hole"]) == 2 and set(combo["hole"]) <= set(holes[int(seat)]), \
                "exactly two of the player's SIX cards play"
            assert len(combo["board"]) == 3
    _finish_runout(hg, gid)
    s = _state(p[0], gid)
    assert s["phase"] == "showdown"
    tabled = [x for x in s["seats"] if x["hole"] and x["hole"][0] >= 0 and x["in_hand"] and not x["folded"]]
    assert len(tabled) >= 2, "a real showdown tables the live hands"
    for x in tabled:
        assert len(x["hole"]) == 6 and sorted(x["hole"]) == holes[x["seat"]]
        assert x["hand_desc"] and all(isinstance(d, str) and d for d in x["hand_desc"])
    assert sum(r["net_cents"] for r in s["ledger"]) == 0, "the ledger still sums to zero"
    # the hand record names its game and keeps all six cards
    rec = p[0].get(f"/games/api/tables/{gid}/hands/{s['hand_no']}").json()
    assert rec["variant"] == "plo6" and rec["hole_count"] == 6
    shown = [x for x in rec["seats"] if x["hole"]]
    assert shown and all(len(x["hole"]) == 6 for x in shown)
    assert rec["grades"] == [], "PLO6 is not graded — an empty list, never 'still being worked out'"
    assert _post(p[0], gid, "run", {"running": False}).status_code == 200


def test_plo6_decisions_are_never_graded_and_plo5_ones_still_are(cast, hg):
    p = cast["p"]
    queued: list[dict] = []
    real_put = hg._GRADE_Q.put

    def spy(job, *a, **kw):
        queued.append(job)
        return real_put(job, *a, **kw)

    hg._GRADE_Q.put = spy
    try:
        six = _table(cast, 2, variant="plo6", num_seats=2)
        _post(p[0], six, "run", {"running": True})
        _check_down(cast, six)
        _finish_runout(hg, six)
        assert not [j for j in queued if j["game_id"] == six], "a PLO6 hand never reaches the grader"
        five = _table(cast, 2, num_seats=2)
        _post(p[0], five, "run", {"running": True})
        _check_down(cast, five)
        _finish_runout(hg, five)
        assert [j["variant"] for j in queued if j["game_id"] == five] == ["plo5"]
    finally:
        hg._GRADE_Q.put = real_put
    assert hg.wait_for_grading(60.0)
    graded = p[0].get(f"/games/api/tables/{five}/hands/1").json()
    assert graded["grades"], "PLO5 is graded as before"
    plain = p[0].get(f"/games/api/tables/{six}/hands/1").json()
    assert plain["grades"] == [] and plain["variant"] == "plo6"
    row = hg.pub.DB.one("SELECT SUM(acc_n) n FROM homegame_hand_results WHERE game_id=?", (six,))
    assert int(row["n"] or 0) == 0, "no accuracy is recorded for PLO6"
    # a job for another game handed to the grader directly is refused, too
    assert hg.grade_hand({"variant": "plo6"}) == []
    for gid in (six, five):
        _post(p[0], gid, "run", {"running": False})


# --- the verified shuffle ------------------------------------------------------------------


def test_the_verified_shuffle_deals_plo6_by_the_six_card_slot_map(cast, hg):
    if not hg.FAIR_ON:
        pytest.skip("this engine build has no explicit-deck deal")
    p = cast["p"]
    gid = _table(cast, 3, variant="plo6", num_seats=5)
    nonces = {}
    for i in range(3):
        s = _state(p[i], gid)
        nx = s["fair"]["next"]
        assert nx["hand_id"].endswith(":plo6"), "the seal names the game (and so its slot map)"
        n = fd.sha(f"nonce {i}")
        nonces[i] = n
        c = fd.nonce_commitment(nx["hand_id"], nx["seal"], s["my_seat"], n)
        assert _post(p[i], gid, "fair/commit", {"hand_id": nx["hand_id"], "commit": c}).status_code == 200
    _post(p[0], gid, "run", {"running": True})
    nx = _state(p[0], gid)["fair"]["next"]
    for i in range(3):
        assert _post(p[i], gid, "fair/reveal", {"hand_id": nx["hand_id"], "nonce": nonces[i]}).status_code == 200
    for i in range(3):
        s = _state(p[i], gid)
        assert s["phase"] == "in_hand"
        h = s["fair"]["hand"]
        assert h["hole"] == 6 and h["contributors"] == [0, 1, 2]
        tr = p[i].get(f"/games/api/tables/{gid}/fair/{h['hand_no']}").json()
        assert tr["hole"] == 6
        perm = fd.verify_transcript(tr, my_seat=i, my_nonce=nonces[i])
        n, seen = s["num_seats"], 0
        for j, seat in enumerate(s["seats"]):
            for c in seat.get("hole") or []:
                if c >= 0:
                    fd.verify_opening(tr, perm, c, h["open"][str(c)],
                                      expect_slots=[fd.hole_slot(j, k, 6) for k in range(6)])
                    seen += 1
        for b in ("a", "b"):
            bd = s["board"][b]
            for m, c in enumerate(list(bd["flop"]) + [bd["turn"], bd["river"]]):
                if c is not None:
                    fd.verify_opening(tr, perm, c, h["open"][str(c)], expect_slots=[fd.board_slot(n, b, m, 6)])
                    seen += 1
        assert seen == 6 + 6, "my six cards and the two flops"
    assert _post(p[0], gid, "run", {"running": False}).status_code == 200


def test_a_sealed_deck_refuses_a_slot_map_that_does_not_fit_one_deck():
    with pytest.raises(ValueError):
        fd.SealedDeck.create("x:1:1", 8, hole=6)  # 48 + 10 = 58 cards
    sd = fd.SealedDeck.create("x:1:1:plo6", 7, hole=6)
    assert fd.board_slot(7, "b", 4, 6) == 51, "seven PLO6 players end exactly on the deck's last card"
    sd.set_lock({})
    sd.finish({})
    again = fd.SealedDeck.from_store(sd.to_store())
    assert again.hole == 6 and again.public() == sd.public()
    old = sd.to_store()
    old.pop("hole")  # stored before PLO6
    assert fd.SealedDeck.from_store(dict(old, hand_id="x:1:1")).hole == 5


# --- the club's numbers, per game ------------------------------------------------------------


def test_the_club_keeps_its_numbers_apart_per_game(cast, hg):
    p, uid = cast["p"], cast["uid"]
    club = p[0].get("/games/api/clubs").json()["clubs"][0]["id"]
    q = lambda **kw: "&".join(f"{k}={v}" for k, v in {"club": club, **kw}.items())  # noqa: E731
    before = {g["code"]: g["hands"] for g in p[0].get(f"/games/api/community?{q()}").json()["games"]}
    # two more PLO6 hands than PLO5 hands, between two new players (seats 5 and 6)
    five = _create(p[7], num_seats=2).json()["id"]
    assert _post(p[8], five, "sit", {"seat": 1, "buyin_cents": 20000}).status_code == 200
    six = _create(p[7], variant="plo6", num_seats=2).json()["id"]
    assert _post(p[8], six, "sit", {"seat": 1, "buyin_cents": 20000}).status_code == 200
    pair = {"p": [p[7], p[8]], "by_uid": {uid[7]: p[7], uid[8]: p[8]}}
    want6 = max(1, before.get("plo5", 0) + 1 - before.get("plo6", 0) + 1)  # PLO6 ends up the most-played
    for gid, hands in ((five, 1), (six, want6)):
        assert _post(p[7], gid, "run", {"running": True}).status_code == 200
        for k in range(hands):
            if k:
                _finish_runout(hg, gid)
                _next_hand(pair, gid)
            _check_down(pair, gid)
        _finish_runout(hg, gid)
        assert _post(p[7], gid, "run", {"running": False}).status_code == 200
    n5 = len(p[7].get(f"/games/api/tables/{five}/hands").json()["hands"])
    n6 = len(p[7].get(f"/games/api/tables/{six}/hands").json()["hands"])
    assert (n5, n6) == (1, want6)

    def card(data, who):
        return next((x for x in data["players"] if x["user_id"] == who), None)

    c5 = p[0].get(f"/games/api/community?{q(variant='plo5')}").json()
    c6 = p[0].get(f"/games/api/community?{q(variant='plo6')}").json()
    assert c5["variant"] == "plo5" and c6["variant"] == "plo6"
    games = {g["code"]: g for g in c6["games"]}
    assert games["plo6"]["hands"] == before.get("plo6", 0) + n6 and games["plo5"]["hands"] == before.get("plo5", 0) + n5
    assert games["plo5"]["graded"] is True and games["plo6"]["graded"] is False
    # the two players' cards: PLO5 numbers from the PLO5 table only, PLO6 from the PLO6 one
    assert card(c5, uid[7])["hands"] == n5 and card(c6, uid[7])["hands"] == n6
    assert card(c6, uid[7])["accuracy"] is None and card(c6, uid[7])["graded"] == 0
    assert card(c5, uid[7])["sessions"] == 1 and card(c6, uid[7])["sessions"] == 1
    net5 = sum(r["delta_cents"] for r in hg.pub.DB.q(
        "SELECT delta_cents FROM homegame_hand_results WHERE game_id=? AND user_id=?", (five, uid[7])))
    net6 = sum(r["delta_cents"] for r in hg.pub.DB.q(
        "SELECT delta_cents FROM homegame_hand_results WHERE game_id=? AND user_id=?", (six, uid[7])))
    assert card(c5, uid[7])["net_cents"] == net5 and card(c6, uid[7])["net_cents"] == net6
    # head to head follows the game too: only that game's money between the two
    def owed(data):
        return sum(x["cents"] if x["to"] == uid[7] else -x["cents"] for x in data["pairs"]
                   if {x["from"], x["to"]} == {uid[7], uid[8]})
    assert owed(c5) == net5 and owed(c6) == net6
    # no game named: the one the club has played most (here PLO6)
    assert p[0].get(f"/games/api/community?{q()}").json()["variant"] == "plo6"
    assert p[0].get(f"/games/api/community?{q(variant='omaha')}").status_code == 400
    # every session is listed, each with its game
    sess = {x["id"]: x for x in c5["sessions"]}
    assert sess[five]["variant"] == "plo5" and sess[six]["variant"] == "plo6"
    # a player's own numbers: per game, or everything together
    me6 = p[7].get(f"/games/api/my/stats?{q(variant='plo6')}").json()
    me_all = p[7].get(f"/games/api/my/stats?{q()}").json()
    assert me6["variant"] == "plo6" and me6["hands"] == n6 and me6["net_cents"] == net6
    assert me_all["variant"] is None and me_all["hands"] == n5 + n6 and me_all["net_cents"] == net5 + net6
    assert {g["code"]: g["hands"] for g in me_all["games"]} == {"plo5": n5, "plo6": n6, "plo67": 0}
    assert {s["id"]: s["variant"] for s in me_all["sessions"]} == {five: "plo5", six: "plo6"}
    assert sum(v["net_cents"] for v in me6["versus"]) == net6
    # … and somebody else's, the same way
    other = p[0].get(f"/games/api/players/{uid[8]}/stats?{q(variant='plo5')}").json()
    assert other["hands"] == n5 and other["variant"] == "plo5"
    hands6 = p[0].get(f"/games/api/players/{uid[8]}/hands?{q(variant='plo6')}").json()
    assert hands6["total"] == n6 and {h["variant"] for h in hands6["hands"]} == {"plo6"}
    mine5 = p[7].get(f"/games/api/my/hands?{q(variant='plo5')}").json()
    assert mine5["total"] == n5 and {h["game_id"] for h in mine5["hands"]} == {five}
    for gid in (five, six):
        assert _post(p[7], gid, "close").status_code == 200


# --- the browser's half: games.fair.js reads the six-card slot map ------------------------------

FAIR_JS = __import__("pathlib").Path(__file__).resolve().parents[3] / "python" / "plo5bp" / "ui" / "static" / "games.fair.js"
FAIR_HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const input = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const ctx = { console, JSON, Math, Number, String, Object, Array, Set, Map, Promise, Error, Uint8Array, Uint32Array,
  parseInt, encodeURIComponent, crypto: require("crypto").webcrypto, setTimeout };
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx, { filename: "games.fair.js" });
const A = ctx.HG.fair.__api;
const tr = input.transcript, perm = A.verifyTranscript(tr, null);
const out = { slots: A.visibleCards(input.state), five: A.visibleCards(Object.assign({}, input.state, { hole_count: undefined })) };
out.ok = A.visibleCards(input.state).every(([card, slots]) => { A.verifyOpening(tr, perm, card, input.open[String(card)], slots); return true; });
try { A.visibleCards(Object.assign({}, input.state, { hole_count: 5 })).forEach(([card, slots]) => A.verifyOpening(tr, perm, card, input.open[String(card)], slots)); out.five_map_passes = true; }
catch (e) { out.five_map_passes = false; }
process.stdout.write(JSON.stringify(out));
"""


def test_the_browser_checks_plo6_cards_against_the_six_card_slot_map(tmp_path):
    import json
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    n = 7
    sd = fd.SealedDeck.create("g:3:1:e:plo6", n, hole=6)
    nonce = fd.sha("device")
    sd.set_lock({4: fd.nonce_commitment(sd.hand_id, sd.seal, 4, nonce)})
    deck = sd.finish({4: nonce})
    mine = [deck[fd.hole_slot(4, k, 6)] for k in range(6)]
    flop_a = [deck[fd.board_slot(n, "a", m, 6)] for m in range(3)]
    flop_b = [deck[fd.board_slot(n, "b", m, 6)] for m in range(3)]
    seats = [{"hole": mine if i == 4 else [-1] * 6} for i in range(n)]
    state = {"num_seats": n, "hole_count": 6, "seats": seats,
             "board": {"a": {"flop": flop_a, "turn": None, "river": None},
                       "b": {"flop": flop_b, "turn": None, "river": None}}}
    payload = {"transcript": sd.public(), "state": state,
               "open": {str(c): sd.opening(c) for c in mine + flop_a + flop_b}}
    (tmp_path / "in.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "h.js").write_text(FAIR_HARNESS, encoding="utf-8")
    run = subprocess.run(["node", str(tmp_path / "h.js"), str(FAIR_JS), str(tmp_path / "in.json")],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    assert out["ok"] is True, "every card on a PLO6 screen checks out in the browser"
    assert [slots for _, slots in out["slots"][:6]] == [list(range(24, 30))] * 6, "seat 4's six cards: slots 24-29"
    assert [slots for _, slots in out["slots"][6:9]] == [[42], [43], [44]], "board A starts at 6 x 7 = 42"
    assert [slots for _, slots in out["slots"][9:]] == [[47], [48], [49]], "board B five slots later"
    assert [slots for _, slots in out["five"][:1]] == [list(range(20, 25))], "no hole count = the PLO5 map"
    assert out["five_map_passes"] is False, "a PLO6 deal never checks out under the five-card map"


def test_the_award_script_and_who_paid_whom_match_the_engine_on_random_plo6_hands():
    """runout.py (the award animation, the money flows, the equities) is pure Python
    and hole-count generic — pinned here against the ENGINE's PLO6 payouts: random
    tables of 2-7, random betting, every terminal state."""
    import random

    from plo5bp.config import VARIANT_PLO6, GameConfig
    from plo5bp.env import BombPotEnv
    from plo5bp.ui.runout import board_equities, build_awards, money_flows

    rng = random.Random(20260926)
    showdowns = 0
    for trial in range(50):
        n = rng.choice([2, 3, 4, 5, 6, 7])
        stacks = tuple(rng.choice([40_000, 90_000, 150_000, 400_000, 700_000]) for _ in range(n))
        cfg = GameConfig(num_seats=n, starting_stack=0, starting_stacks=stacks,
                         ante=30_000, bb=10_000, variant=VARIANT_PLO6)
        env = BombPotEnv(cfg, ev_runout_samples=0, obs_mode="minimal")
        _, info = env.reset(rng.getrandbits(62), rng.randrange(n))
        for _ in range(250):
            if env.is_terminal():
                break
            gm, r = info.gate_mask, rng.random()
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
        all_holes = env.all_hole_cards()
        assert all(len(h) == 6 for h in all_holes)
        holes = [None if folded[i] else [int(x) for x in h] for i, h in enumerate(all_holes)]
        commit = [int(x) for x in raw["total_commit"]]
        ba, bb = [int(x) for x in raw["board_a"]], [int(x) for x in raw["board_b"]]
        button = int(raw["button"])
        payouts = [int(x) for x in env._rs.payouts()]
        awards = build_awards(holes, folded, commit, ba, bb, button)
        won = [sum(int(a["shares"].get(str(i), 0)) for a in awards) for i in range(n)]
        assert sum(won) == sum(commit)
        assert all(abs((w - c) - p) <= 4 for w, c, p in zip(won, commit, payouts)), (trial, won, commit, payouts)
        flows = money_flows(commit, folded, holes, ba, bb, button)
        net = [0] * n
        for (a, b), v in flows.items():
            net[a] -= v
            net[b] += v
        assert sum(net) == 0 and all(abs(x - p) <= 4 for x, p in zip(net, payouts)), (trial, net, payouts)
        alive = {i: h for i, h in enumerate(holes) if h}
        if len(alive) >= 2:
            showdowns += 1
            eq = board_equities(alive, ba[:3], bb[:3], seed=trial)  # (from the flop: sampled runouts)
            for b in ("a", "b"):
                assert abs(sum(v[b] for v in eq.values()) - 1.0) < 1e-3
    assert showdowns >= 10
