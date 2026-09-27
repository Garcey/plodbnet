"""Home games — PLO67 (2026-09-27), the owner's friends' format.

A double-board PLO bomb pot where every player is dealt FOUR cards and the three
burn cards are dealt FACE UP — one before the flops, one before the turns, one
before the rivers. Every red burn deals everyone still in the hand (all-in players
too, folded ones not) one more hole card: 4-5 cards on the flop, 4-6 on the turn,
4-7 on the river. Up to 5 seats (the verified shuffle reserves seven cards a seat,
+ 10 board + 3 burns = 48). Pinned here: the seat limit, the burns on the table
and the extra cards they deal (from the sealed deck's public slot map), the
all-in runout revealed burn by burn with each hand as it was on that street, the
hand record's per-street cards, the browser verifier's PLO67 checks, and that
PLO67 is never graded and keeps its own club numbers.
"""
from __future__ import annotations

import random
import sys
import time

import pytest
from starlette.testclient import TestClient

from plo5bp.ui import fairdeal as fd

ADMIN_EMAIL = "admin@plo67.example"
NAMES = ["ann", "ben", "cat", "dov", "eve", "fay"]
RED = lambda c: c % 4 in (1, 2)  # noqa: E731  (diamonds, hearts)


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
    players = [login(f"{n}@plo67.example") for n in NAMES]
    ids = {u["email"]: u["id"] for u in adm.get("/admin/api/users").json()["users"]}
    for n in NAMES:
        r = adm.post("/admin/api/games_access", json={"user_id": ids[f"{n}@plo67.example"], "action": "grant"})
        assert r.status_code == 200
    uid = [ids[f"{n}@plo67.example"] for n in NAMES]
    return {"p": players, "uid": uid, "by_uid": dict(zip(uid, players))}


# --- helpers -------------------------------------------------------------------------


def _post(cl, gid, what, body=None):
    return cl.post(f"/games/api/tables/{gid}/{what}", json=body or {})


def _state(cl, gid):
    r = cl.get(f"/games/api/tables/{gid}")
    assert r.status_code == 200, r.text
    return r.json()


def _create(cl, **kw):
    body = {"name": "sixty-seven", "sb_cents": 50, "bb_cents": 100, "ante_cents": 300,
            "default_buyin_cents": 20000, "variant": "plo67", **kw}
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


def _runout_at(hg, gid, board_len):
    """Wind an all-in runout's clock to the moment it shows ``board_len`` cards."""
    t = hg.HUB.get(gid)
    with t.lock:
        pause = hg._street_pause(t)
        start = max(3, int(t.runout_start_len or 3))
        t.runout_started_mono = time.monotonic() - pause * (board_len - start) - 0.05


def _finish_runout(hg, gid):
    t = hg.HUB.get(gid)
    with t.lock:
        if t.runout_active and t.runout_started_mono is not None:
            t.runout_started_mono -= 600.0


def _rig(hg, gid, colours, seed=0):
    """Seal the next hand's deck so its three burns come out ``colours``
    (True = red) once the (empty) cut has permuted it. Returns the dealt order
    F, so every card of the hand is known: seat s's k-th card F[7s+k], board A
    F[7n..], board B F[7n+5..], burn j F[7n+10+j] — the public slot map."""
    t = hg.HUB.get(gid)
    rng = random.Random(seed)
    with t.lock:
        hg._fair_prepare_locked(t)
        nxt = t.fair_next
        assert nxt is not None and nxt.sealed.hole == 7 and nxt.sealed.burns == 3
        hid, n = nxt.sealed.hand_id, t.num_seats
        for _ in range(500):
            deck = list(range(52))
            rng.shuffle(deck)
            sd = fd.SealedDeck.create(hid, n, deck=deck, hole=7, burns=3)
            probe = fd.SealedDeck.create(hid, n, key=sd.key, deck=deck, hole=7, burns=3)
            probe.set_lock({})
            dealt = probe.finish({})
            if tuple(RED(dealt[fd.burn_slot(n, j, 7)]) for j in range(3)) == tuple(colours):
                nxt.sealed = sd
                return dealt
    raise AssertionError("no deck with those burns")


def _hole(dealt, seat, m):
    return sorted(dealt[7 * seat: 7 * seat + m])


# --- the table -----------------------------------------------------------------------


def test_a_plo67_table_seats_at_most_five_and_says_what_it_deals(cast, hg):
    host = cast["p"][0]
    r = _create(host)
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["variant"] == "plo67" and s["num_seats"] == 5, "5-max by default"
    assert s["hole_count"] == 7, "seven slots a seat: the most a PLO67 hand can hold"
    assert s["game"] == {"code": "plo67", "label": "PLO67", "name": "PLO67 double-board bomb pot",
                         "hole": 7, "dealt": 4, "burns": 3, "max_seats": 5, "graded": False}
    assert s["burns"] == [] and s["burns_played"] == 0, "nothing turned up before the first hand"
    # 6 x 7 + 10 + 3 = 55 cards: one deck can't deal it
    bad = _create(host, num_seats=6)
    assert bad.status_code == 400 and "2–5" in bad.json()["detail"] and "burns" in bad.json()["detail"]
    assert _post(host, s["id"], "settings", {"num_seats": 6}).status_code == 400
    assert _create(host, variant="plo67_double_bomb", num_seats=3).json()["variant"] == "plo67"
    assert _post(host, s["id"], "close").status_code == 200
    rows = host.get("/games/api/tables").json()["tables"]
    for row in rows:
        if row.get("variant") == "plo67" and row.get("status") == "open":
            _post(host, row["id"], "close")


def test_red_burns_deal_everyone_still_in_one_more_card_and_folded_players_none(cast, hg):
    if not hg.FAIR_ON:
        pytest.skip("this engine build has no explicit-deck deal")
    p = cast["p"]
    gid = _table(cast, 3, num_seats=3)
    dealt = _rig(hg, gid, (True, True, False))  # red flop burn, red turn burn, black river burn
    n = 3
    s = _post(p[0], gid, "run", {"running": True}).json()
    assert s["phase"] == "in_hand" and s["street"] == "flop"
    assert s["burns"] == [dealt[fd.burn_slot(n, 0, 7)]], "the flop's burn is face up from the deal"
    for i in range(3):
        me = _state(p[i], gid)
        for j, seat in enumerate(me["seats"]):
            if j == i:
                assert sorted(seat["hole"]) == _hole(dealt, i, 5), "four cards + one for the red burn"
            else:
                assert seat["hole"] == [-1] * 5, "everyone else's five, face down"
    # the first player to act bets, the next folds, the last calls
    cl, s = _actor(cast, gid)
    first = s["actor"]
    _act(cl, gid, s, gate="raise", raise_to_chips=s["raise_bounds"]["min_chips"])
    cl, s = _actor(cast, gid)
    folder = s["actor"]
    _act(cl, gid, s, gate="fold")
    cl, s = _actor(cast, gid)
    _act(cl, gid, s, gate="check_call")
    s = _state(p[0], gid)
    assert s["street"] == "turn" and len(s["burns"]) == 2
    counts = {x["seat"]: len(x["hole"]) for x in s["seats"] if x["hole"]}
    assert counts[folder] == 5, "folded before the red turn burn: no sixth card"
    assert [counts[x] for x in range(3) if x != folder] == [6, 6]
    mine = _state(cast["by_uid"][s["seats"][first]["user_id"]], gid)["seats"][first]["hole"]
    assert sorted(mine) == _hole(dealt, first, 6), "the sixth card is the seat's sixth slot"
    s = _check_down(cast, gid)
    _finish_runout(hg, gid)
    s = _state(p[0], gid)
    assert s["phase"] == "showdown" and s["burns"] == [dealt[fd.burn_slot(n, j, 7)] for j in range(3)]
    assert s["burns_played"] == 3
    tabled = {x["seat"]: x["hole"] for x in s["seats"] if x["hole"] and x["hole"][0] >= 0}
    assert set(tabled) == {x for x in range(3) if x != folder}
    for seat, hole in tabled.items():
        assert sorted(hole) == _hole(dealt, seat, 6), "black river burn: still six"
    # the record keeps every hand in deal order and its count on each street
    rec = p[0].get(f"/games/api/tables/{gid}/hands/{s['hand_no']}").json()
    assert rec["variant"] == "plo67" and rec["hole_count"] == 7
    assert rec["burns"] == s["burns"]
    by = {x["seat"]: x for x in rec["seats"]}
    assert by[folder]["counts"] == [5, 5, 5]
    for seat in tabled:
        assert by[seat]["counts"] == [5, 6, 6]
        assert by[seat]["hole_seq"] == dealt[7 * seat: 7 * seat + 6], "deal order: slot by slot"
    # a folded hand stays mucked in the history: its cards hidden, its count public
    viewer = next(i for i in range(3) if i not in (folder,))
    rec_v = cast["p"][viewer].get(f"/games/api/tables/{gid}/hands/{s['hand_no']}").json()
    fold_row = next(x for x in rec_v["seats"] if x["seat"] == folder)
    assert fold_row["hole"] is None and fold_row["hole_seq"] is None and fold_row["counts"] == [5, 5, 5]
    assert rec["grades"] == [], "PLO67 is never graded"
    assert _post(p[0], gid, "run", {"running": False}).status_code == 200


def test_an_all_in_runout_reveals_burn_by_burn_with_each_hand_as_it_was(cast, hg):
    if not hg.FAIR_ON:
        pytest.skip("this engine build has no explicit-deck deal")
    p = cast["p"]
    gid = _table(cast, 3, num_seats=3)
    dealt = _rig(hg, gid, (False, True, True), seed=1)  # black flop burn, then two red ones
    n = 3
    s = _post(p[0], gid, "run", {"running": True}).json()
    assert all(len(x["hole"]) == 4 for x in s["seats"] if x["hole"]), "black flop burn: four cards"
    _jam_out(cast, gid)
    t = hg.HUB.get(gid)
    with t.lock:
        assert t.runout_active and t.runout_start_len == 3
        assert abs(hg._street_pause(t) - (1.5 + hg.BURN_SHOW_S)) < 1e-9, "a runout street waits for its burn"
        eq = dict(t.equity_by_len)
        payouts = list(t.last_deltas)
        commit = list(t.terminal_commit)
        awards = list(t.pot_awards)
        all_holes = [list(h) for h in t.env.all_hole_cards()]
    assert set(eq) == {(3, 3), (4, 4), (5, 5)}, "equities for every street the runout shows"
    for shares in eq.values():
        for b in ("a", "b"):
            assert abs(sum(v[b] for v in shares.values()) - 1.0) < 1e-3
    live = [i for i in range(3) if len(all_holes[i]) == 6]
    assert len(live) >= 2, "everyone all in: two red burns dealt every hand two more"
    for i in live:
        assert all_holes[i] == dealt[7 * i: 7 * i + 6]
    # flop shown: one burn, four cards a hand (the two red burns are still to come)
    _runout_at(hg, gid, 3)
    s = _state(p[0], gid)
    assert s["runout"]["active"] and s["runout"]["shown_len"] == 3
    assert s["burns"] == [dealt[fd.burn_slot(n, 0, 7)]]
    for x in s["seats"]:
        if x["seat"] in live:
            assert sorted(x["hole"]) == _hole(dealt, x["seat"], 4), "tabled as it was on the flop"
            assert x["equity_a"] is not None
    # turn shown: the red turn burn dealt everyone a fifth card
    _runout_at(hg, gid, 4)
    s = _state(p[0], gid)
    assert s["runout"]["shown_len"] == 4 and len(s["burns"]) == 2
    for x in s["seats"]:
        if x["seat"] in live:
            assert sorted(x["hole"]) == _hole(dealt, x["seat"], 5)
    # river: the sixth
    _runout_at(hg, gid, 5)
    s = _state(p[0], gid)
    assert len(s["burns"]) == 3
    for x in s["seats"]:
        if x["seat"] in live:
            assert sorted(x["hole"]) == _hole(dealt, x["seat"], 6)
    # the award script pays exactly what the engine paid (six-card hands)
    for i in range(3):
        won = sum(int(a["shares"].get(str(i), 0)) for a in awards)
        assert won - commit[i] == payouts[i]
    for a in awards:
        for seat, combo in (a.get("combos") or {}).items():
            assert len(combo["hole"]) == 2 and set(combo["hole"]) <= set(all_holes[int(seat)])
    _finish_runout(hg, gid)
    s = _state(p[0], gid)
    assert s["phase"] == "showdown" and not s["runout"]["blocking"]
    assert sum(r["net_cents"] for r in s["ledger"]) == 0
    assert _post(p[0], gid, "run", {"running": False}).status_code == 200


def test_the_rabbit_shows_the_burns_that_would_have_come_without_dealing_them(cast, hg):
    if not hg.FAIR_ON:
        pytest.skip("this engine build has no explicit-deck deal")
    p = cast["p"]
    gid = _table(cast, 2, num_seats=2)
    dealt = _rig(hg, gid, (False, True, True), seed=2)
    _post(p[0], gid, "run", {"running": True})
    cl, s = _actor(cast, gid)
    _act(cl, gid, s, gate="raise", raise_to_chips=s["raise_bounds"]["min_chips"])
    cl, s = _actor(cast, gid)
    _act(cl, gid, s, gate="fold")
    s = _state(p[0], gid)
    assert s["phase"] == "showdown" and s["burns"] == [dealt[fd.burn_slot(2, 0, 7)]] and s["burns_played"] == 1
    assert _post(p[0], gid, "rabbit").status_code == 200
    s = _state(p[0], gid)
    assert s["burns"] == [dealt[fd.burn_slot(2, j, 7)] for j in range(3)], "the rabbit turns up every burn"
    assert s["burns_played"] == 1, "only the flop's burn was part of the hand"
    assert all(len(x["hole"]) == 4 for x in s["seats"] if x["hole"]), "red rabbit burns deal nobody a card"
    assert _post(p[0], gid, "run", {"running": False}).status_code == 200


# --- the verified shuffle ------------------------------------------------------------------


def test_every_card_on_a_plo67_screen_opens_to_its_slot_burns_included(cast, hg):
    if not hg.FAIR_ON:
        pytest.skip("this engine build has no explicit-deck deal")
    p = cast["p"]
    gid = _table(cast, 3, num_seats=4)
    nonces = {}
    for i in range(3):
        s = _state(p[i], gid)
        nx = s["fair"]["next"]
        assert nx["hand_id"].endswith(":plo67"), "the seal names the game (and so its slot map)"
        nonces[i] = fd.sha(f"nonce {i}")
        c = fd.nonce_commitment(nx["hand_id"], nx["seal"], s["my_seat"], nonces[i])
        assert _post(p[i], gid, "fair/commit", {"hand_id": nx["hand_id"], "commit": c}).status_code == 200
    _post(p[0], gid, "run", {"running": True})
    nx = _state(p[0], gid)["fair"]["next"]
    for i in range(3):
        assert _post(p[i], gid, "fair/reveal", {"hand_id": nx["hand_id"], "nonce": nonces[i]}).status_code == 200
    _check_down(cast, gid)  # (all the way: every burn is turned up)
    for i in range(3):
        s = _state(p[i], gid)
        h = s["fair"]["hand"]
        assert h["hole"] == 7 and h["burns"] == 3 and h["contributors"] == [0, 1, 2]
        tr = p[i].get(f"/games/api/tables/{gid}/fair/{h['hand_no']}").json()
        assert tr["hole"] == 7 and tr["burns"] == 3
        perm = fd.verify_transcript(tr, my_seat=i, my_nonce=nonces[i])
        n = s["num_seats"]
        for j, seat in enumerate(s["seats"]):
            hole = [c for c in seat.get("hole") or [] if c >= 0]
            for c in hole:  # a hand of m cards: its seat's FIRST m slots
                fd.verify_opening(tr, perm, c, tr["open"][str(c)],
                                  expect_slots=[fd.hole_slot(j, k, 7) for k in range(len(hole))])
        for j, c in enumerate(s["burns"]):
            fd.verify_opening(tr, perm, c, tr["open"][str(c)], expect_slots=[fd.burn_slot(n, j, 7)])
        assert len(s["burns"]) == 3
    assert _post(p[0], gid, "run", {"running": False}).status_code == 200


def test_a_sealed_plo67_deck_fits_five_seats_and_round_trips():
    with pytest.raises(ValueError):
        fd.SealedDeck.create("x:1:1:plo67", 6, hole=7, burns=3)  # 42 + 10 + 3 = 55
    sd = fd.SealedDeck.create("x:1:1:plo67", 5, hole=7, burns=3)
    assert fd.burn_slot(5, 2, 7) == 47, "five players: the last burn is the deck's 48th card"
    sd.set_lock({})
    sd.finish({})
    again = fd.SealedDeck.from_store(sd.to_store())
    assert (again.hole, again.burns) == (7, 3) and again.public() == sd.public()
    assert sd.public()["burns"] == 3
    assert "burns" not in fd.SealedDeck.create("x:1:1", 5).public(), "PLO5 transcripts are unchanged"


# --- the browser's half: games.fair.js reads the PLO67 slot map ------------------------------

FAIR_JS = __import__("pathlib").Path(__file__).resolve().parents[2] / "python" / "plo5bp" / "ui" / "static" / "games.fair.js"
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
const check = (state) => { try { A.visibleCards(state).forEach(([card, slots]) => A.verifyOpening(tr, perm, card, input.open[String(card)], slots)); return true; } catch (e) { return false; } };
const out = { slots: A.visibleCards(input.state), ok: check(input.state), late: check(input.late),
  count_ok: A.holeCountProblem(input.state), count_bad: A.holeCountProblem(input.short),
  count_rabbit: A.holeCountProblem(input.rabbit) };
process.stdout.write(JSON.stringify(out));
"""


def test_the_browser_checks_plo67_cards_burns_and_counts(tmp_path):
    import json
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    n = 4
    sd = fd.SealedDeck.create("g:3:1:e:plo67", n, hole=7, burns=3)
    nonce = fd.sha("device")
    sd.set_lock({2: fd.nonce_commitment(sd.hand_id, sd.seal, 2, nonce)})
    deck = sd.finish({2: nonce})
    burns = [deck[fd.burn_slot(n, j, 7)] for j in range(2)]  # on the turn: two burns up
    red = sum(1 for c in burns if RED(c))
    mine = [deck[fd.hole_slot(2, k, 7)] for k in range(4 + red)]
    flop_a = [deck[fd.board_slot(n, "a", m, 7)] for m in range(4)]
    flop_b = [deck[fd.board_slot(n, "b", m, 7)] for m in range(4)]
    game = {"code": "plo67", "hole": 7, "dealt": 4, "burns": 3}

    def state(hole, burn_list, played=None):
        seats = [{"seat": i, "in_hand": True, "folded": False,
                  "hole": hole if i == 2 else [-1] * (4 + red)} for i in range(n)]
        return {"num_seats": n, "hole_count": 7, "game": game, "seats": seats, "burns": burn_list,
                "burns_played": len(burn_list) if played is None else played,
                "board": {"a": {"flop": flop_a[:3], "turn": flop_a[3], "river": None},
                          "b": {"flop": flop_b[:3], "turn": flop_b[3], "river": None}}}

    late = [deck[fd.hole_slot(2, k, 7)] for k in range(3)] + [deck[fd.hole_slot(2, 6, 7)]]  # a 7th-slot card shown early
    rabbit_burns = [deck[fd.burn_slot(n, j, 7)] for j in range(3)]
    payload = {
        "transcript": sd.public(),
        "state": state(mine, burns),
        "late": state(late + mine[4:], burns),
        "short": state(mine[:4], burns) if red else state(mine + [deck[fd.hole_slot(2, 4, 7)]], burns),
        "rabbit": state(mine, rabbit_burns, played=2),
        "open": {str(c): sd.opening(c) for c in set(mine + late + flop_a + flop_b + rabbit_burns
                                                     + [deck[fd.hole_slot(2, 4, 7)]])},
    }
    (tmp_path / "in.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "h.js").write_text(FAIR_HARNESS, encoding="utf-8")
    run = subprocess.run(["node", str(tmp_path / "h.js"), str(FAIR_JS), str(tmp_path / "in.json")],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    assert out["ok"] is True, "every card on a PLO67 screen checks out in the browser"
    m = 4 + red
    assert [slots for _, slots in out["slots"][:m]] == [list(range(14, 14 + m))] * m, "seat 2: its first m slots"
    assert [slots for _, slots in out["slots"][-2:]] == [[38], [39]], "the burns: 7 x 4 + 10 onwards"
    assert out["late"] is False, "a card from a later slot, shown early, is refused"
    assert out["count_ok"] is None
    assert out["count_bad"], "a hand that missed (or gained) a red burn's card is caught"
    assert out["count_rabbit"] is None, "a fold-out's rabbit burns dealt nobody a card"


# --- grading and the club --------------------------------------------------------------------


def test_plo67_is_never_graded_and_keeps_its_own_club_numbers(cast, hg):
    p, uid = cast["p"], cast["uid"]
    queued: list[dict] = []
    real_put = hg._GRADE_Q.put

    def spy(job, *a, **kw):
        queued.append(job)
        return real_put(job, *a, **kw)

    hg._GRADE_Q.put = spy
    try:
        gid = _create(p[4], num_seats=2).json()["id"]
        assert _post(p[5], gid, "sit", {"seat": 1, "buyin_cents": 20000}).status_code == 200
        pair = {"p": [p[4], p[5]], "by_uid": {uid[4]: p[4], uid[5]: p[5]}}
        assert _post(p[4], gid, "run", {"running": True}).status_code == 200
        _check_down(pair, gid)
        _finish_runout(hg, gid)
        assert _post(p[4], gid, "run", {"running": False}).status_code == 200
    finally:
        hg._GRADE_Q.put = real_put
    assert not [j for j in queued if j["game_id"] == gid], "a PLO67 hand never reaches the grader"
    assert hg.grade_hand({"variant": "plo67"}) == []
    club = p[4].get("/games/api/clubs").json()["clubs"][0]["id"]
    c67 = p[4].get(f"/games/api/community?club={club}&variant=plo67").json()
    assert c67["variant"] == "plo67"
    games = {g["code"]: g for g in c67["games"]}
    assert games["plo67"]["hands"] >= 1 and games["plo67"]["graded"] is False
    me = next(x for x in c67["players"] if x["user_id"] == uid[4])
    assert me["hands"] == 1 and me["accuracy"] is None
    mine = p[4].get(f"/games/api/my/stats?club={club}&variant=plo67").json()
    assert mine["variant"] == "plo67" and mine["hands"] == 1
    hands = p[4].get(f"/games/api/my/hands?club={club}&variant=plo67").json()
    assert hands["total"] == 1 and {h["variant"] for h in hands["hands"]} == {"plo67"}
    assert _post(p[4], gid, "close").status_code == 200


def test_runout_equities_wrapper_matches_the_engine_shape():
    from plo5bp.ui.runout import plo67_equities

    eq = plo67_equities({0: [48, 49, 0, 4], 3: [44, 45, 1, 5, 9]}, [50, 51, 20], [36, 37, 21], [3], samples=500, seed=4)
    assert set(eq) == {0, 3}
    for b in ("a", "b"):
        assert abs(eq[0][b] + eq[3][b] - 1.0) < 1e-3
    assert eq == plo67_equities({0: [48, 49, 0, 4], 3: [44, 45, 1, 5, 9]}, [50, 51, 20], [36, 37, 21], [3],
                                samples=500, seed=4), "deterministic from the seed"
