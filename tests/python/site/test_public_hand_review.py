"""Hand review (2026-10-03): a player's ClubGG hand histories on the site's one paid page.

The hands here are SYNTHETIC — written by ``gg_hand`` below in ClubGG's export format
(GG network text; each street's betting printed once per board), their pots paid the
way the site pays them (``runout.build_awards``). No real player's hand history is in
the repository.

Covered: the parser (every line kind of the export, the per-board copies), the money
(ClubGG's numbers, uncalled bets, dead money, side pots), the engine replay that checks
it and becomes the grading job (including the corner where ClubGG's rules differ), the
all-in EV against a brute-force enumeration of BOTH boards together, the zip reader's
limits, and the store / API: the paywall (even while the site is free), duplicates, the
numbers, the hand list, a hand's record, deletion, and account export / deletion.
"""

from __future__ import annotations

import collections
import io
import json
import sys
import zipfile
from itertools import permutations

import pytest
from starlette.testclient import TestClient

from plo5bp.ui import handreview as hr
from plo5bp.ui.runout import board_equities, build_awards, uncalled_bet

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def cards(s: str) -> list[int]:
    return [RANKS.index(t[0]) * 4 + SUITS.index(t[1]) for t in s.split()]


def money(d: float) -> str:
    """ClubGG's way: $400, $59.4, $1,191.39."""
    s = f"{d:,.2f}"
    return "$" + (s.rstrip("0").rstrip(".") if "." in s else s)


def gg_hand(*, hid: str, button: int, seats: list[tuple[int, str, float]], deal: dict[str, str],
            board_a: str, board_b: str, streets: dict[str, list[tuple]], dead: dict[str, float] | None = None,
            when: str = "2026/10/02 00:03:26", ante: float = 60.0) -> str:
    """One PLO5 BP/DB hand in ClubGG's text. ``streets`` = {street: [(name, verb,
    amount, all_in)]} (``raises`` amount = the raise-TO total; ``rto`` = "raises to
    $X" without the increment, as ClubGG prints a short all-in). ``deal`` = the hero's
    cards and every hand that reaches the showdown."""
    dead = dead or {}
    order = sorted(seats)
    idx = {nm: i for i, (_n, nm, _s) in enumerate(order)}
    n = len(order)
    btn = max((i for i, (no, _nm, _s) in enumerate(order) if no <= button), default=n - 1)
    ba, bb = cards(board_a), cards(board_b)
    put = [round((ante + dead.get(nm, 0)) * 100) for _no, nm, _s in order]
    folded = [False] * n
    lines = [f"Poker Hand #ring_{hid}: PLO Pot Limit ($10/$20) - {when}",
             f"Table 'PLO5: BP/DB' 6-max Seat #{button} is the button"]
    lines += [f"Seat {no}: {nm} ({money(st)} in chips)" for no, nm, st in seats]
    lines += [f"{nm}: posts the ante {money(ante)}" for _no, nm, _st in seats]
    lines += [f"{nm}: posts missed blind {money(v)}" for nm, v in dead.items()]
    lines += ["*** HOLE CARDS ***"]
    lines += [f"Dealt to {nm} [{deal[nm]}]" if nm == "Hero" else f"Dealt to {nm} " for _no, nm, _st in seats]
    names = {"flop": "FLOP", "turn": "TURN", "river": "RIVER"}
    acted = [st for st in ("flop", "turn", "river") if streets.get(st)]
    last = acted[-1] if acted else "flop"
    uncalled = None
    blocks: dict[str, list[str]] = {}
    for st in ("flop", "turn", "river"):
        acts = streets.get(st, [])
        sc = [0] * n
        out = []
        for nm, verb, amt, ai in acts:
            i = idx[nm]
            tail = " and is all-in" if ai else ""
            c = round(amt * 100) if amt is not None else 0
            if verb in ("checks", "folds"):
                out.append(f"{nm}: {verb}")
                folded[i] = folded[i] or verb == "folds"
            elif verb in ("bets", "calls"):
                out.append(f"{nm}: {verb} {money(amt)}{tail}")
                sc[i] += c
            elif verb == "raises":
                out.append(f"{nm}: raises {money((c - max(sc)) / 100)} to {money(amt)}{tail}")
                sc[i] = c
            elif verb == "rto":
                out.append(f"{nm}: raises to {money(amt)}{tail}")
                sc[i] = c
        for i in range(n):
            put[i] += sc[i]
        blocks[st] = out
    # the bet nobody matched goes back to its owner when the betting closes
    unc = uncalled_bet(put)
    if unc is not None:
        uncalled = (order[unc[0]][1], int(unc[1]))
        put[unc[0]] -= int(unc[1])
    alive = [i for i in range(n) if not folded[i]]
    showdown = len(alive) >= 2
    streets_dealt = ["flop", "turn", "river"] if showdown else acted
    for st in streets_dealt:
        for copy, b in ((0, ba), (1, bb)):
            ln = {"flop": 3, "turn": 4, "river": 5}[st]
            head = f"[{' '.join(hr.card_str(x) for x in b[:3])}]" if st == "flop" else \
                f"[{' '.join(hr.card_str(x) for x in b[:ln - 1])}] [{hr.card_str(b[ln - 1])}]"
            lines.append(f"*** {names[st]} *** {head}")
            lines += blocks.get(st, [])
            if st == last:
                if uncalled:
                    lines.append(f"Uncalled bet ({money(uncalled[1] / 100)}) returned to {uncalled[0]}")
                if showdown:
                    lines += [f"{order[i][1]}: shows [{deal[order[i][1]]}] (Two-High)" for i in alive]
    holes = [cards(deal[order[i][1]]) if (i in alive and showdown) else None for i in range(n)]
    awards = build_awards(holes, folded, put, ba if showdown else ba[:3], bb if showdown else bb[:3], btn)
    per_board: dict[str, list[tuple[str, int]]] = {"a": [], "b": []}
    for a in awards:
        if a.get("uncontested") and not showdown:
            w = order[a["winners"][0]][1]
            half = int(a["chips"]) // 2
            per_board["a"].append((w, half))
            per_board["b"].append((w, int(a["chips"]) - half))
            continue
        for s, v in a["shares"].items():
            per_board[a["board"]].append((order[int(s)][1], int(v)))
    for key in ("a", "b"):
        lines.append("*** SHOWDOWN ***")
        lines += [f"{nm} collected {money(c / 100)} from pot" for nm, c in per_board[key] if c]
    lines += ["*** SUMMARY ***", f"Total pot {money(sum(put) / 100)}"]
    shown_len = 5 if showdown else {"flop": 3, "turn": 4, "river": 5}[last]
    lines += [f"Board [{' '.join(hr.card_str(x) for x in ba[:shown_len])}]",
              f"Board [{' '.join(hr.card_str(x) for x in bb[:shown_len])}]"]
    lines += [f"Seat {no}: {nm} folded on the Flop" if folded[idx[nm]] else f"Seat {no}: {nm} showed and won"
              for no, nm, _st in seats]
    return "\n".join(lines) + "\n"


# --- the hands ----------------------------------------------------------------------------------

FOLD_OUT = gg_hand(
    hid="9000000001", button=5,
    seats=[(5, "Hero", 400), (6, "aa11bb22", 448.3), (3, "cc33dd44", 447.65)],
    deal={"Hero": "Kh Jd Td 8d 2s"}, board_a="6h 6c Qc", board_b="9c 5c 4d",
    streets={"flop": [("aa11bb22", "checks", None, False), ("cc33dd44", "bets", 59.4, False),
                      ("Hero", "folds", None, False), ("aa11bb22", "folds", None, False)]},
)

# Three-way all-in on the flop with a side pot: X is all in short, Hero all in for
# less than a full raise, Y (the deepest) calls — then both boards are run out.
SIDE_POT = gg_hand(
    hid="9000000002", button=1,
    seats=[(1, "ee55ff66", 600), (2, "Hero", 400), (4, "aa77bb88", 250)],
    deal={"Hero": "Ah Kh Qd Jd 4c", "ee55ff66": "9s 9c 8s 7c 2h", "aa77bb88": "Tc Ts 6d 5d 3s"},
    board_a="Th 9h 2d Kc 3h", board_b="8h 7h 4d Qs Js",
    streets={"flop": [("Hero", "checks", None, False), ("aa77bb88", "checks", None, False),
                      ("ee55ff66", "bets", 180, False), ("Hero", "raises", 340, True),
                      ("aa77bb88", "calls", 190, True), ("ee55ff66", "calls", 160, False)]},
)

# A missed blind posted as dead money; the poster is all in on the turn.
DEAD_MONEY = gg_hand(
    hid="9000000003", button=3,
    seats=[(1, "Hero", 500), (3, "1234abcd", 280), (5, "beefcafe", 700)],
    deal={"Hero": "Ac Ad Ks Qs 9h", "1234abcd": "Jh Jc 8d 7d 6s", "beefcafe": "5h 4h 3c 3d 2c"},
    board_a="As 7h 2c Jd 8c", board_b="Kd Qh 5s 4c Tc", dead={"1234abcd": 10},
    streets={"flop": [("beefcafe", "checks", None, False), ("Hero", "bets", 100, False),
                      ("1234abcd", "calls", 100, False), ("beefcafe", "calls", 100, False)],
             "turn": [("beefcafe", "checks", None, False), ("Hero", "bets", 300, False),
                      ("1234abcd", "calls", 110, True), ("beefcafe", "folds", None, False)]},
)

# ClubGG's one rule the engine doesn't share: the hero checked, a short stack bet all in
# for less than a full bet ($12.72 < the $20 minimum), and ClubGG let the hero raise.
SHORT_ALLIN_RAISE = gg_hand(
    hid="9000000004", button=5,
    seats=[(1, "Hero", 555.24), (2, "72f0aaa1", 759.9), (3, "5fd2bbb2", 72.72), (5, "73e7ccc3", 991.47)],
    deal={"Hero": "Ad Qs Js 4s 4h", "72f0aaa1": "Jc 8d 8h 6d 6c", "5fd2bbb2": "Tc 8c 7c 5s 3d"},
    board_a="8s 3s 2d 6h 9d", board_b="Qh 2s 3c 2c 5c",
    streets={"flop": [("Hero", "checks", None, False), ("72f0aaa1", "checks", None, False),
                      ("5fd2bbb2", "rto", 12.72, True), ("73e7ccc3", "folds", None, False),
                      ("Hero", "raises", 278.16, False), ("72f0aaa1", "raises", 699.9, True),
                      ("Hero", "calls", 217.08, True)]},
)


def zip_of(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in files.items():
            z.writestr(name, text)
    return buf.getvalue()


# --- parsing and the money --------------------------------------------------------------------


def test_the_export_reads_back_and_the_betting_is_one_round_printed_per_board():
    p = hr.parse_hand(FOLD_OUT)
    assert p.hand_id == "ring_9000000001" and p.key == "clubgg:ring_9000000001"
    assert p.bb_cents == 2000 and p.hero == "Hero" and p.hero_hole == cards("Kh Jd Td 8d 2s")
    assert p.board_a == cards("6h 6c Qc") and p.board_b == cards("9c 5c 4d")
    assert [a.verb for a in p.streets["flop"]] == ["checks", "bets", "folds", "folds"]
    assert p.uncalled == ("cc33dd44", 5940)
    L = hr.ledger(p)
    assert L.net(L.idx["cc33dd44"]) == 12000 and L.net(L.hero) == -6000
    assert sum(L.net(i) for i in range(L.n)) == 0


def test_two_boards_that_disagree_are_refused():
    bad = FOLD_OUT.replace("*** FLOP *** [9c 5c 4d]\naa11bb22: checks\ncc33dd44: bets $59.4",
                           "*** FLOP *** [9c 5c 4d]\naa11bb22: checks\ncc33dd44: bets $60")
    with pytest.raises(hr.HandError) as e:
        hr.parse_hand(bad)
    assert e.value.reason == "boards_disagree"


def test_other_games_and_broken_hands_are_skipped_with_a_reason():
    with pytest.raises(hr.HandError) as e:
        hr.parse_hand(FOLD_OUT.replace("PLO Pot Limit", "Hold'em No Limit"))
    assert e.value.reason == "other_game"
    with pytest.raises(hr.HandError) as e:
        hr.parse_hand(FOLD_OUT.replace("Dealt to Hero [Kh Jd Td 8d 2s]", "Dealt to Hero [Kh Jd Td 8d]"))
    assert e.value.reason == "no_hero_cards"


@pytest.mark.parametrize("text", [FOLD_OUT, SIDE_POT, DEAD_MONEY, SHORT_ALLIN_RAISE])
def test_every_hand_builds_and_its_money_adds_up(text):
    b = hr.build_hand(hr.parse_hand(text))
    rec = b.record
    assert sum(s["delta_cents"] for s in rec["seats"]) == 0
    assert rec["net_cents"] == b.net_cents == next(s for s in rec["seats"] if s["is_me"])["delta_cents"]
    assert len(rec["actions"]) == sum(len(v) for v in hr.parse_hand(text).streets.values())
    me = next(s for s in rec["seats"] if s["is_me"])
    assert me["name"] == "You" and me["hole"] == sorted(hr.parse_hand(text).hero_hole, reverse=True)
    for s in rec["seats"]:
        if not s["is_me"] and not s["shown"]:
            assert s["hole"] == []  # (a hand nobody showed is never in the record)


def test_the_engine_replay_agrees_with_clubgg_and_grades_only_the_heros_decisions():
    for text in (FOLD_OUT, SIDE_POT):
        b = hr.build_hand(hr.parse_hand(text))
        assert b.notes == []
        assert b.record["study_upto"] == len(b.record["actions"])
        hero = next(s["seat"] for s in b.record["seats"] if s["is_me"])
        assert [a[3] for a in b.job["actions"]] == [a["seat"] == hero for a in b.record["actions"]]


def test_dead_money_is_in_the_pot_and_cut_from_the_posters_engine_stack():
    b = hr.build_hand(hr.parse_hand(DEAD_MONEY))
    rec = b.record
    poster = next(s for s in rec["seats"] if s.get("dead_cents"))
    assert poster["dead_cents"] == 1000 and rec["dead_cents"] == 1000
    assert rec["pot_cents"] == hr.parse_hand(DEAD_MONEY).total_pot  # (ClubGG's "Total pot", dead money in)
    assert "dead money" in b.notes  # (the engine's pot is that much short)
    assert b.job["stacks"][poster["seat"]] == (28000 - 1000) * 5  # (cents x 5 chips at $20 bb)


def test_the_raise_clubgg_allows_after_a_check_stops_the_grading_replay_there():
    b = hr.build_hand(hr.parse_hand(SHORT_ALLIN_RAISE))
    assert any("short all-in" in n for n in b.notes)
    rec = b.record
    assert rec["study_upto"] == 4 < len(rec["actions"])  # (the hero's raise is action 5)
    assert len(b.job["actions"]) == 4 and b.gradable == 1  # (the hero's check before it)
    assert sum(s["delta_cents"] for s in rec["seats"]) == 0  # (the money is ClubGG's either way)


# --- all-in EV, side pots included --------------------------------------------------------------


def _brute_force_ev(text: str) -> float:
    """The hero's expected collection minus what they put in, from EVERY pair of
    runouts of the two boards together, paid by the site's own pot code."""
    p = hr.parse_hand(text)
    L = hr.ledger(p)
    start = hr.STREET_LEN[L.last_street]
    holes = [L.holes.get(i) if not L.folded[i] else None for i in range(L.n)]
    known = {c for i in L.alive for c in L.holes[i]} | set(p.board_a[:start]) | set(p.board_b[:start])
    stub = [c for c in range(52) if c not in known]
    miss = 5 - start
    total, count = 0.0, 0
    for extra in permutations(stub, 2 * miss) if miss == 1 else _pairs_of_pairs(stub):
        ba = p.board_a[:start] + list(extra[:miss])
        bb = p.board_b[:start] + list(extra[miss:])
        aw = build_awards(holes, L.folded, L.put, ba, bb, L.button)
        total += sum(int(a["shares"].get(str(L.hero), 0)) for a in aw)
        count += 1
    return total / count - L.put[L.hero]


def _pairs_of_pairs(stub):
    """Flop all-in: a fixed, even subset of the joint runouts (every 2-card turn+river
    for board A crossed with a stride through board B's) — enough to pin the EV."""
    import random

    rng = random.Random(1234)
    for _ in range(6000):
        s = rng.sample(stub, 4)
        yield tuple(s)


def test_side_pot_ev_matches_the_brute_force_of_both_boards():
    p = hr.parse_hand(SIDE_POT)
    L = hr.ledger(p)
    ev, detail = hr.allin_ev(L, p)
    assert detail is not None and detail["from"] == 3
    assert [x["players"] for x in detail["pots"]] == [2, 3]  # (the side pot, then the main pot)
    brute = _brute_force_ev(SIDE_POT)
    pot = sum(L.put)
    assert abs(ev - brute) < 0.02 * pot, (ev, brute)  # (sampled: within 2 % of the pot)
    assert detail["luck_cents"] == L.net(L.hero) - ev


def test_a_side_pots_equity_keeps_the_out_of_race_hand_dead():
    # board_equities(dead=...): the side pot between two players must not deal
    # the all-in third player's cards onto the board.
    p = hr.parse_hand(SIDE_POT)
    L = hr.ledger(p)
    two = {i: L.holes[i] for i in L.alive if i != L.idx["aa77bb88"]}
    with_dead = board_equities(two, p.board_a[:3], p.board_b[:3], dead=L.holes[L.idx["aa77bb88"]], digits=None)
    without = board_equities(two, p.board_a[:3], p.board_b[:3], digits=None)
    assert with_dead != without
    assert board_equities(two, p.board_a[:3], p.board_b[:3]) == {
        s: {k: round(v, 4) for k, v in d.items()} for s, d in without.items()}  # (defaults unchanged)


def test_a_turn_all_in_ev_is_exact():
    text = gg_hand(
        hid="9000000005", button=2,
        seats=[(2, "Hero", 400), (4, "77aa88bb", 400)],
        deal={"Hero": "As Ks Qh Jh 2c", "77aa88bb": "9d 9c 8h 8c 3d"},
        board_a="Ts 9s 2h 4c 7d", board_b="Kd 8d 3s 5h 6s",
        streets={"flop": [("77aa88bb", "checks", None, False), ("Hero", "checks", None, False)],
                 "turn": [("77aa88bb", "bets", 120, False), ("Hero", "raises", 340, True),
                          ("77aa88bb", "calls", 220, True)]},
    )
    p = hr.parse_hand(text)
    ev, detail = hr.allin_ev(hr.ledger(p), p)
    assert detail["from"] == 4
    assert abs(ev - _brute_force_ev(text)) <= 1  # (exact: a cent of rounding)


def test_no_all_in_means_the_ev_line_is_the_result():
    p = hr.parse_hand(FOLD_OUT)
    L = hr.ledger(p)
    assert hr.allin_ev(L, p) == (L.net(L.hero), None)


# --- the upload --------------------------------------------------------------------------------


def test_a_zip_needs_no_unpacking_and_duplicates_count_once():
    data = zip_of({"GG20261002 - PLO5 BPDB - 10 - 20 - 6max.txt": FOLD_OUT + "\n\n" + SIDE_POT,
                   "folder/second.txt": SIDE_POT + "\n\n" + DEAD_MONEY, "notes.pdf": "ignored"})
    res = hr.build_upload(data, "hands.zip")
    assert res.files == 2 and len(res.hands) == 3
    assert res.skipped == {"duplicate_in_upload": 1}


def test_bad_uploads_say_why():
    with pytest.raises(hr.UploadError):
        hr.read_upload(b"PK\x03\x04 not really a zip")
    with pytest.raises(hr.UploadError) as e:
        hr.read_upload(b"x" * (hr.MAX_UPLOAD_BYTES + 1))
    assert e.value.args[0] == "too_big"
    # a zip whose text unpacks past the limit is refused while it is read
    big = zip_of({"a.txt": "Poker Hand #x\n" + "y" * 1024})
    old = hr.MAX_FILE_BYTES
    try:
        hr.MAX_FILE_BYTES = 512
        with pytest.raises(hr.UploadError) as e:
            hr.read_upload(big)
        assert e.value.args[0] == "too_big"
    finally:
        hr.MAX_FILE_BYTES = old


# --- the store and the API ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def server(boot_public_server):
    # The way production runs today: the site free for everyone — Hand review still paid.
    return boot_public_server(PLO5BP_FREE_FOR_ALL="1")


@pytest.fixture(scope="module")
def pub(server):
    return sys.modules["plo5bp.ui.public"]


@pytest.fixture(scope="module")
def store(server):
    return sys.modules["plo5bp.ui.handreview_store"]


def _login(server, email):
    c = TestClient(server.app, raise_server_exceptions=False)
    assert c.get("/auth/dev", params={"email": email}).status_code == 200
    return c


def _comp(pub, email):
    pub.DB.q("UPDATE users SET sub_status='active', sub_source='comp' WHERE email=?", (email,))


def _upload(c, store, files, name="hands.zip"):
    r = c.post(f"/games/api/review/upload?name={name}", content=zip_of(files),
               headers={"Content-Type": "application/zip"})
    assert r.status_code == 200, r.text
    assert store.wait_idle(60)
    return c.get(f"/games/api/review/uploads/{r.json()['upload_id']}").json()


def test_hand_review_is_paid_even_while_the_site_is_free(server, pub):
    assert pub.FREE_FOR_ALL is True
    c = _login(server, "free@example.com")
    me = c.get("/me").json()
    assert me["sub"]["active"] is True and me["paid"] is False  # (Study is free; storage isn't)
    assert me["review"]["href"] == "/games/review"
    r = c.get("/games/api/review/summary")
    assert r.status_code == 402 and r.json()["detail"]["error"] == "subscription_required"
    assert c.post("/games/api/review/upload", content=b"PK").status_code == 402
    # checkout stays open for it (no Stripe key in the tests: "not configured", not "free")
    assert c.post("/billing/checkout", json={"next": "/games/review"}).status_code == 503
    # the page itself opens for every signed-in player (it shows what the subscription buys)
    page = c.get("/games/review")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]


def test_signed_out_a_link_signs_you_in_and_the_api_stays_hidden(server):
    c = TestClient(server.app, raise_server_exceptions=False)
    page = c.get("/games/review", headers={"Accept": "text/html"})
    assert page.status_code == 200 and "Hand review" in page.text and "Sign in" in page.text
    assert "/games/review" in page.text  # (sign-in comes straight back here)
    assert c.get("/games/review", headers={"Accept": "application/json"}).status_code == 404
    assert c.get("/games/api/review/summary").status_code in (401, 404)
    assert c.get("/games/static/games.review.js").status_code == 404


def test_upload_numbers_list_and_record(server, pub, store):
    email = "grinder@example.com"
    c = _login(server, email)
    _comp(pub, email)
    up = _upload(c, store, {"one.txt": FOLD_OUT + "\n\n" + SIDE_POT, "two.txt": DEAD_MONEY + "\n\n" + SIDE_POT})
    assert up["status"] == "done" and up["added"] == 3 and up["duplicates"] == 1
    s = c.get("/games/api/review/summary").json()
    expect = [hr.build_hand(hr.parse_hand(t)) for t in (FOLD_OUT, SIDE_POT, DEAD_MONEY)]
    assert s["hands"] == 3
    assert s["net_cents"] == sum(b.net_cents for b in expect)
    assert s["ev_net_cents"] == sum(b.ev_net_cents for b in expect)
    sr = c.get("/games/api/review/series").json()
    assert sr["points"][-1][1:3] == [s["net_cents"], s["ev_net_cents"]] and sr["hands"] == 3
    lst = c.get("/games/api/review/hands?sort=net&dir=desc").json()
    assert lst["total"] == 3 and [h["net_cents"] for h in lst["hands"]] == sorted((b.net_cents for b in expect), reverse=True)
    key = lst["hands"][0]["key"]
    rec = c.get(f"/games/api/review/hands/{key}").json()
    assert rec["hand_key"] == key and rec["kind"] == "review" and rec["grading"] is True
    # the same zip again: every hand is a duplicate
    again = _upload(c, store, {"one.txt": FOLD_OUT + "\n\n" + SIDE_POT})
    assert again["added"] == 0 and again["duplicates"] == 2


def test_graded_hands_sort_worst_first(server, pub, store):
    email = "student2@example.com"
    c = _login(server, email)
    _comp(pub, email)
    _upload(c, store, {"h.txt": FOLD_OUT + "\n\n" + SIDE_POT})
    uid = int(pub.DB.one("SELECT id FROM users WHERE email=?", (email,))["id"])
    keys = [r["hand_key"] for r in pub.DB.q("SELECT hand_key FROM review_hands WHERE user_id=?", (uid,))]
    store.store_grades(uid, keys[0], [{"i": 0, "seat": 0, "score": 92.0, "cat": "correct"}])
    store.store_grades(uid, keys[1], [{"i": 0, "seat": 1, "score": 4.0, "cat": "blunder"}])
    rows = c.get("/games/api/review/hands?sort=worst").json()["hands"]
    assert [r["key"] for r in rows] == [keys[1], keys[0]] and rows[0]["mistakes"] == 1
    assert c.get("/games/api/review/hands?filter=mistakes").json()["total"] == 1
    s = c.get("/games/api/review/summary").json()
    assert s["graded"] == 2 and s["accuracy"] == 48.0 and s["grading_pending"] == 0


def test_grading_a_stored_hand_with_the_network(server, pub, store):
    from plo5bp.ui.server import current_site

    model = current_site().formats["plo5_double_bomb"]["model"]
    email = "graded@example.com"
    c = _login(server, email)
    _comp(pub, email)
    _upload(c, store, {"h.txt": SIDE_POT})
    uid = int(pub.DB.one("SELECT id FROM users WHERE email=?", (email,))["id"])
    row = pub.DB.one("SELECT hand_key, job FROM review_hands WHERE user_id=?", (uid,))
    grades = store.grade_one(uid, row["hand_key"], row["job"], model)
    rec = c.get(f"/games/api/review/hands/{row['hand_key']}").json()
    hero = next(s["seat"] for s in rec["seats"] if s["is_me"])
    assert grades and all(rec["actions"][g["i"]]["seat"] == hero for g in rec["grades"])
    assert rec["grading"] is False


def test_your_hands_are_yours_alone_and_deleting_them_works(server, pub, store):
    a, b = _login(server, "alice@example.com"), _login(server, "bob@example.com")
    _comp(pub, "alice@example.com")
    _comp(pub, "bob@example.com")
    _upload(a, store, {"a.txt": FOLD_OUT})
    key = a.get("/games/api/review/hands").json()["hands"][0]["key"]
    assert b.get(f"/games/api/review/hands/{key}").status_code == 404
    assert b.get("/games/api/review/summary").json()["hands"] == 0
    assert a.post("/games/api/review/delete", json={"confirm": True}).json() == {"deleted": 1}
    assert a.get("/games/api/review/summary").json()["hands"] == 0


def test_account_export_and_deletion_include_the_hand_histories(server, pub, store):
    email = "leaving@example.com"
    c = _login(server, email)
    _comp(pub, email)
    _upload(c, store, {"x.txt": SIDE_POT})
    uid = int(pub.DB.one("SELECT id FROM users WHERE email=?", (email,))["id"])
    exported = pub.export_user(uid)
    assert len(exported["hand_review"]["hands"]) == 1
    store._account_anonymize(uid)
    assert pub.DB.one("SELECT COUNT(*) n FROM review_hands WHERE user_id=?", (uid,))["n"] == 0


# --- the network's choice in the replayer, and the mistakes drill (2026-10-03) ------------------


def _model():
    from plo5bp.ui.server import current_site

    return current_site().formats["plo5_double_bomb"]["model"]


def _uid(pub, email):
    return int(pub.DB.one("SELECT id FROM users WHERE email=?", (email,))["id"])


def _hero_moves(rec):
    hero = next(s["seat"] for s in rec["seats"] if s["is_me"])
    return hero, [i for i, a in enumerate(rec["actions"]) if a["seat"] == hero and i < rec["study_upto"]]


def test_a_spot_rebuilt_from_the_record_is_the_node_the_grader_scored(server, monkeypatch):
    """The replayer's "network's choice" and the drill rebuild a decision from the hand
    record (``spot_env``: Study's way, the other hands placeholders); the grader replayed
    the dealt deck. The network must see the SAME observation in both, bit for bit."""
    import numpy as np

    from plo5bp.ui import homegame as hg
    from plo5bp.ui import trainer as tr

    model = _model()
    seen: list = []
    real = tr.compute_node_distribution

    def spy(m, device, obs, info):
        seen.append(np.array(obs, copy=True))
        return real(m, device, obs, info)

    monkeypatch.setattr(tr, "compute_node_distribution", spy)
    checked = 0
    for text in (FOLD_OUT, SIDE_POT, DEAD_MONEY, SHORT_ALLIN_RAISE):
        b = hr.build_hand(hr.parse_hand(text))
        seen.clear()
        grades = hg.grade_hand(b.job, model)
        assert len(grades) == len(seen) >= 1
        for g, obs in zip(grades, seen):
            log: list = []
            _env, o2, info, actor = hr.spot_env(b.record, g["i"], log=log)
            assert actor == g["seat"] and int(info.actor) == actor
            assert np.array_equal(o2, obs), (text[:40], g)
            assert [x["seat"] for x in log] == [a["seat"] for a in b.record["actions"][:g["i"]]]
            checked += 1
    assert checked >= 6


def test_the_network_choice_at_your_decisions(server, pub, store, monkeypatch):
    import torch

    from plo5bp.ui import homegame as hg
    from plo5bp.ui import trainer as tr

    model = _model()
    monkeypatch.setattr(hg, "_grading_model", lambda: model)
    email = "chooser@example.com"
    c = _login(server, email)
    _comp(pub, email)
    _upload(c, store, {"h.txt": "\n\n".join((FOLD_OUT, SIDE_POT, SHORT_ALLIN_RAISE))})
    keys = {r["hand_id"]: r["key"] for r in c.get("/games/api/review/hands").json()["hands"]}

    side = keys["ring_9000000002"]
    rec = c.get(f"/games/api/review/hands/{side}").json()
    hero, mine = _hero_moves(rec)
    for i in mine:
        r = c.get(f"/games/api/review/hands/{side}/choice", params={"i": i})
        assert r.status_code == 200, r.text
        ch = r.json()
        assert ch["i"] == i and ch["seat"] == hero
        assert abs(sum(ch["probs"].values()) - 1.0) < 1e-3
        assert all(ch["legal"][g] or ch["probs"][g] < 1e-6 for g in ("fold", "call", "raise"))
        # = the network's own forward at the rebuilt node
        _env, obs, info, _a = hr.spot_env(rec, i)
        d = tr.compute_node_distribution(model, torch.device("cpu"), obs, info)
        assert ch["pick"]["gate"] == ("fold", "call", "raise")[d["rec_gate"]]
        assert ch["probs"]["raise"] == round(d["gate_probs"][2], 4)
        if ch["pick"]["gate"] == "raise":
            assert ch["pick"]["to_cents"] > 0
    # a showdown shows every hand: the network can be asked about the others' decisions too
    other = next(i for i, a in enumerate(rec["actions"]) if a["seat"] != hero)
    assert c.get(f"/games/api/review/hands/{side}/choice", params={"i": other}).status_code == 200

    fold_out = keys["ring_9000000001"]
    r = c.get(f"/games/api/review/hands/{fold_out}/choice", params={"i": 1})  # (a villain's bet, never shown)
    assert r.status_code == 400 and "weren't shown" in r.json()["detail"]
    assert c.get(f"/games/api/review/hands/{fold_out}/choice", params={"i": 99}).status_code == 400
    # past where the network's rules follow ClubGG (the raise after a short all-in)
    short = keys["ring_9000000004"]
    r = c.get(f"/games/api/review/hands/{short}/choice", params={"i": 4})
    assert r.status_code == 400 and "short all-in" in r.json()["detail"]
    # nobody else's hands, no subscription, no network
    other_user = _login(server, "nosub@example.com")
    assert other_user.get(f"/games/api/review/hands/{side}/choice", params={"i": mine[0]}).status_code == 402
    monkeypatch.setattr(hg, "_grading_model", lambda: None)
    assert c.get(f"/games/api/review/hands/{side}/choice", params={"i": mine[0]}).status_code == 503


def test_drill_rounds_put_the_worst_first_most_often_and_fixed_spots_less_often():
    import random

    from plo5bp.ui import handreview_store as st

    spots = [{"key": "k:1", "i": 0, "score": 0.0, "weight": 1.0},
             {"key": "k:2", "i": 0, "score": 12.0, "weight": 1.0},
             {"key": "k:3", "i": 0, "score": 28.0, "weight": 1.0}]
    rng = random.Random(7)
    first: dict = {}
    orders = set()
    for _ in range(6000):
        o = st.drill_order(spots, True, rng)
        assert sorted(o) == [("k:1", 0), ("k:2", 0), ("k:3", 0)]  # (every unfixed spot, every round)
        first[o[0][0]] = first.get(o[0][0], 0) + 1
        orders.add(tuple(k for k, _ in o))
    assert first["k:1"] > first["k:2"] > first["k:3"] > 0  # (the worst most likely first, ...
    assert first["k:1"] < 5400  # ... but not always)
    assert {("k:1", "k:2", "k:3"), ("k:2", "k:1", "k:3"), ("k:1", "k:3", "k:2")} <= orders
    # equal priority: every spot equally likely first
    first = {}
    for _ in range(6000):
        o = st.drill_order(spots, False, rng)
        first[o[0][0]] = first.get(o[0][0], 0) + 1
    assert all(1800 < v < 2200 for v in first.values())
    # a fixed spot (weight 1/8) sits most rounds out; a struggling one is in every round
    fixed = [dict(spots[0], weight=0.125), dict(spots[1], weight=4.0)]
    shown = sum(("k:1", 0) in st.drill_order(fixed, True, rng) for _ in range(4000))
    assert 350 < shown < 650
    # never the same spot twice in a row across rounds
    for _ in range(200):
        assert st.drill_order(spots, True, rng, last=("k:1", 0))[0] != ("k:1", 0)


def _graded_mistakes(pub, store, c, email, scores):
    """Upload three hands and mark one hero decision of each a mistake (synthetic marks)."""
    _upload(c, store, {"h.txt": "\n\n".join((FOLD_OUT, SIDE_POT, DEAD_MONEY))})
    uid = _uid(pub, email)
    marked = {}
    rows = pub.DB.q("SELECT hand_key FROM review_hands WHERE user_id=? ORDER BY hand_key", (uid,))
    for row, score in zip(rows, scores):
        key = row["hand_key"]
        rec = c.get(f"/games/api/review/hands/{key}").json()
        hero, mine = _hero_moves(rec)
        i = mine[-1]
        store.store_grades(uid, key, [{"i": i, "seat": hero, "score": score,
                                       "cat": "blunder" if score < 10 else "wrong"}])
        marked[key] = (i, rec)
    return uid, marked


def _play(c, ts, move):
    """Act in the drill spot: the network's own choice ("best") or its least likely legal gate."""
    h = ts.hand
    d = ts._node_dist(h.last_obs, h.last_info)
    if move == "best":
        gate = d["rec_gate"]
    else:
        legal = [g for g in range(3) if h.last_info.gate_mask[g]]
        gate = min(legal, key=lambda g: d["gate_probs"][g])
    body = {"gate": ("fold", "check_call", "raise")[gate]}
    if gate == 2:
        body["chips"] = int(d["rec_chips"]) if move == "best" else int(h.last_info.min_raise_chips)
    return c.post("/trainer/act", json=body)


def test_the_mistakes_drill_deals_your_spots_and_learns_from_each_attempt(server, pub, store):
    email = "driller@example.com"
    c = _login(server, email)
    _comp(pub, email)
    uid, marked = _graded_mistakes(pub, store, c, email, [4.0, 18.0, 27.0])
    s = c.get("/games/api/review/summary").json()
    assert s["drill"] == {"mistakes": 3, "fixed": 0, "struggling": 0, "attempts": 0}

    r = c.post("/trainer/drill/next", json={"prioritize": True})
    assert r.status_code == 200, r.text
    ts = pub._REGISTRY.peek(uid).trainer
    st = r.json()["state"]
    d = st["trainer"]["drill"]
    key, i = d["key"], d["i"]
    assert (key in marked) and marked[key][0] == i and d["done"] is False and "orig" not in d
    assert d["pos"] == 1 and d["size"] == 3 and st["trainer"]["hand_active"] is True
    rec = marked[key][1]
    me = next(x for x in rec["seats"] if x["is_me"])
    assert st["actor"] == st["hero_seat"] == rec["actions"][i]["seat"]
    assert sorted(st["card_spec"]["hero_hole"], reverse=True) == me["hole"]
    assert st["card_spec"]["flop_a"] == rec["board_a"][:3] and st["card_spec"]["flop_b"] == rec["board_b"][:3]

    # the network's own play: fixed, and the spot comes up less often
    r = _play(c, ts, "best")
    assert r.status_code == 200, r.text
    st = r.json()["state"]
    d = st["trainer"]["drill"]
    assert st["trainer"]["hand_active"] is False and d["done"] is True and st["actor"] is None
    assert d["result"]["outcome"] == "fixed" and d["result"]["weight"] == 0.5
    assert d["orig"]["cat"] in ("wrong", "blunder") and d["orig"]["gate"] in ("fold", "check_call", "raise")
    assert abs(sum(d["probs"].values()) - 1.0) < 1e-3
    assert st["trainer"]["feedback"]["category"] == "best"
    assert _play(c, ts, "best").status_code == 400  # (the spot is over)
    # (a drill spot has no dealt hand to review or what-if: it never ends)
    assert c.get("/trainer/review").status_code == 400
    assert c.post("/trainer/whatif", json={"decision": 0}).status_code == 409
    row = pub.DB.one("SELECT weight, fixed, missed FROM review_mistakes WHERE user_id=? AND hand_key=? AND idx=?",
                     (uid, key, i))
    assert (row["weight"], row["fixed"], row["missed"]) == (0.5, 1, 0)

    # Try again = practice: graded, never counted
    r = c.post("/trainer/repeat")
    d = r.json()["state"]["trainer"]["drill"]
    assert (d["key"], d["i"], d["practice"], d["done"]) == (key, i, True, False)
    st = _play(c, ts, "worst").json()["state"]
    assert st["trainer"]["drill"]["done"] is True and st["trainer"]["drill"]["result"] is None
    row2 = pub.DB.one("SELECT weight, fixed, missed FROM review_mistakes WHERE user_id=? AND hand_key=? AND idx=?",
                      (uid, key, i))
    assert dict(row2) == dict(row)

    # the round's next spot: another one; the least likely move learns the other way
    st = c.post("/trainer/drill/next", json={"prioritize": True}).json()["state"]
    d2 = st["trainer"]["drill"]
    assert (d2["key"], d2["i"]) != (key, i) and d2["pos"] == 2
    st = _play(c, ts, "worst").json()["state"]
    cat = st["trainer"]["feedback"]["category"]
    res = st["trainer"]["drill"]["result"]
    want = {"best": "fixed", "correct": "fixed", "wrong": "missed", "blunder": "missed"}.get(cat, "close")
    assert res["outcome"] == want
    assert res["weight"] == {"fixed": 0.5, "missed": 1.5, "close": 1.0}[want]

    # the drill never touches the Trainer's own numbers
    assert ts.session_stats.hands == 0 and ts.lifetime_stats.moves == 0 and ts.recent_hands == []
    s = c.get("/games/api/review/summary").json()["drill"]
    # (attempts = the fixed + missed ones: a "close" try leaves a spot's learning state alone)
    assert s["mistakes"] == 3 and s["attempts"] == 1 + (want != "close") and s["fixed"] >= 1

    # equal priority: a new round, every spot in it
    seen = set()
    for _ in range(3):
        st = c.post("/trainer/drill/next", json={"prioritize": False}).json()["state"]
        d = st["trainer"]["drill"]
        assert d["prioritize"] is False and d["size"] == 3
        seen.add((d["key"], d["i"]))
    assert len(seen) == 3
    # a reload shows the spot dealt (not played yet) instead of skipping it
    again = c.post("/trainer/drill/next", json={"prioritize": False, "resume": True}).json()["state"]
    assert (again["trainer"]["drill"]["key"], again["trainer"]["drill"]["i"]) == (d["key"], d["i"])
    # a dealt hand leaves the drill
    st = c.post("/trainer/new_hand").json()["state"]
    assert st["trainer"]["drill"] is None


def test_a_missed_spot_comes_back_later_in_the_same_round(server, pub, store):
    import random

    email = "misser@example.com"
    c = _login(server, email)
    _comp(pub, email)
    uid, _marked = _graded_mistakes(pub, store, c, email, [3.0, 9.0, 20.0])
    first = store.drill_next(uid, True, rng=random.Random(3))
    res = store.drill_result(uid, first["key"], first["i"], "blunder")
    assert res["outcome"] == "missed" and res["weight"] == 1.5 and res["again"] is True
    rest = [store.drill_next(uid, True, rng=random.Random(3)) for _ in range(3)]
    assert [x["pos"] for x in rest] == [2, 3, 4] and rest[-1]["size"] == 4
    assert (first["key"], first["i"]) in {(x["key"], x["i"]) for x in rest[1:]}  # (not straight after)
    assert (rest[0]["key"], rest[0]["i"]) != (first["key"], first["i"])
    # missed again: x1.5 up to 4; fixed: halves down to 1/8; close: no change
    for want in (2.25, 3.375, 4.0, 4.0):
        assert store.drill_result(uid, first["key"], first["i"], "wrong")["weight"] == want
    for want in (2.0, 1.0, 0.5, 0.25, 0.125, 0.125):
        assert store.drill_result(uid, first["key"], first["i"], "best")["weight"] == want
    assert store.drill_result(uid, first["key"], first["i"], "inaccuracy")["weight"] == 0.125


def test_mistakes_follow_the_grades_and_leave_with_the_hands(server, pub, store):
    email = "mover@example.com"
    c = _login(server, email)
    _comp(pub, email)
    uid, marked = _graded_mistakes(pub, store, c, email, [5.0, 15.0, 25.0])
    key = sorted(marked)[0]
    i, rec = marked[key]
    store.drill_result(uid, key, i, "best")
    # graded again (a new model): a decision that is still a mistake keeps what it
    # learned, one that no longer is leaves the drill
    seat = rec["actions"][i]["seat"]
    store.store_grades(uid, key, [{"i": i, "seat": seat, "score": 2.0, "cat": "blunder"}])
    row = pub.DB.one("SELECT score, weight FROM review_mistakes WHERE user_id=? AND hand_key=?", (uid, key))
    assert (row["score"], row["weight"]) == (2.0, 0.5)
    store.store_grades(uid, key, [{"i": i, "seat": seat, "score": 71.0, "cat": "correct"}])
    assert pub.DB.one("SELECT COUNT(*) n FROM review_mistakes WHERE user_id=? AND hand_key=?", (uid, key))["n"] == 0
    # the migration's backfill finds the mistakes of hands graded before the drill existed
    pub.DB.q("DELETE FROM review_mistakes WHERE user_id=?", (uid,))
    with pub.DB.transaction():
        store._backfill_mistakes(pub.DB._conn)
    assert pub.DB.one("SELECT COUNT(*) n FROM review_mistakes WHERE user_id=?", (uid,))["n"] == 2
    # deleting the hands deletes the drill; the Trainer then has nothing to deal
    assert c.post("/games/api/review/delete", json={"confirm": True}).status_code == 200
    r = c.post("/trainer/drill/next", json={"prioritize": True})
    assert r.status_code == 409 and r.json()["detail"]["error"] == "no_mistakes"
    # and without a subscription there is no drill
    assert _login(server, "nodrill@example.com").post("/trainer/drill/next", json={}).status_code == 402


def test_the_next_spot_comes_from_another_hand():
    """(owner, 2026-10-03: "the next hand button brings you to the next mistake you made in
    a different hand") — a hand with several mistakes never deals two of them in a row
    while another hand can go between, in either mode, across rounds too."""
    import random

    from plo5bp.ui import handreview_store as st

    # (three mistakes in one hand, one in each of three others: always possible to keep
    # them apart, even right after a round that ended on that hand)
    spots = [{"key": "h:1", "i": 1, "score": 0.0, "weight": 1.0},
             {"key": "h:1", "i": 4, "score": 2.0, "weight": 1.0},
             {"key": "h:1", "i": 6, "score": 5.0, "weight": 1.0},
             {"key": "h:2", "i": 0, "score": 25.0, "weight": 1.0},
             {"key": "h:3", "i": 2, "score": 20.0, "weight": 1.0},
             {"key": "h:4", "i": 3, "score": 29.0, "weight": 1.0}]
    rng = random.Random(11)
    for prioritize in (True, False):
        last = None
        for _ in range(500):
            o = st.drill_order(spots, prioritize, rng, last=last)
            keys = ([last[0]] if last else []) + [k for k, _i in o]
            assert all(a != b for a, b in zip(keys, keys[1:])), keys
            last = o[-1]
    # one hand only: nothing to put between them, every spot still dealt
    only = [s for s in spots if s["key"] == "h:1"]
    assert sorted(st.drill_order(only, True, rng)) == [("h:1", 1), ("h:1", 4), ("h:1", 6)]


def test_a_drill_spot_replays_the_hand_to_the_decision_and_ends_there(server, pub, store):
    """(owner, 2026-10-03) "replay all the actions up until your decision node … you make
    your decision, get graded, and the hand should end right there. Since opponent hands
    are unknown, the network cannot play them." A TURN mistake: the flop's actions and the
    turn's before it are replayed — the real ones, one frame each — the river never comes,
    nobody else acts, the other hands stay face down."""
    email = "replayer@example.com"
    c = _login(server, email)
    _comp(pub, email)
    _upload(c, store, {"h.txt": DEAD_MONEY})
    uid = _uid(pub, email)
    key = pub.DB.one("SELECT hand_key FROM review_hands WHERE user_id=?", (uid,))["hand_key"]
    rec = c.get(f"/games/api/review/hands/{key}").json()
    hero, mine = _hero_moves(rec)
    i = mine[-1]
    assert rec["actions"][i]["street"] == "turn" and i == 5  # (the hero's turn bet)
    store.store_grades(uid, key, [{"i": i, "seat": hero, "score": 3.0, "cat": "blunder"}])
    ts_tr = sys.modules["plo5bp.ui.trainer"]

    r = c.post("/trainer/drill/next", json={"prioritize": True})
    assert r.status_code == 200, r.text
    body = r.json()
    frames, st = body["frames"], body["state"]
    # the replay: the flop before anyone acts, then one frame per action before the decision
    assert len(frames) == 1 + i and all(f["trainer"]["drill_replay"] for f in frames)
    assert "anim_action" not in frames[0]["trainer"]
    for k, f in enumerate(frames[1:]):
        a, want = f["trainer"]["anim_action"], rec["actions"][k]
        assert a["seat"] == want["seat"] and a["is_hero"] == (want["seat"] == hero)
        assert a["gate"] == {0: "fold", 1: "check_call"}.get(want["action"], "raise")
        assert len(f["history"]) == k + 1
    assert [len(x) for x in (frames[0]["card_spec"]["turn"],)] == [2] and frames[0]["card_spec"]["turn"] == [None, None]
    assert frames[4]["card_spec"]["turn"] == [rec["board_a"][3], rec["board_b"][3]]  # (with the flop's last call)
    # the decision: every earlier action in the history, the turn down, no river, the hero to act
    assert len(st["history"]) == i and "drill_replay" not in st["trainer"]
    assert [h["seat"] for h in st["history"]] == [a["seat"] for a in rec["actions"][:i]]
    assert st["card_spec"]["turn"] == [rec["board_a"][3], rec["board_b"][3]]
    assert st["card_spec"]["river"] == [None, None]
    assert st["actor"] == st["hero_seat"] == hero and st["trainer"]["hand_active"] is True
    hidden = [s for k, s in enumerate(st["seats"]) if k != hero]
    assert hidden and all(not s.get("hole") for s in hidden)

    # the decision, graded — and the hand ends right there: nobody acts, no river
    ts = pub._REGISTRY.peek(uid).trainer
    r = _play(c, ts, "best")
    assert r.status_code == 200, r.text
    after = r.json()
    st2 = after["state"]
    assert len(after["frames"]) == 1  # (no opponent moves)
    assert st2["trainer"]["drill"]["done"] is True and st2["trainer"]["hand_active"] is False
    assert st2["trainer"]["feedback"]["category"] in ts_tr.CATEGORIES
    assert len(st2["history"]) == i and st2["card_spec"]["river"] == [None, None]
    assert st2["actor"] is None and all(not s.get("hole") for k, s in enumerate(st2["seats"]) if k != hero)
    assert ts.hand.env.observation_dict()["street"] == 2  # (the engine never left the turn)
    # Try again goes straight to the decision (no replay)
    again = c.post("/trainer/repeat").json()
    assert len(again["frames"]) == 1 and again["state"]["trainer"]["drill"]["practice"] is True
    assert len(again["state"]["history"]) == i


# --- My tables: the Trainer deals like the tables in your hands (2026-10-03) ----------------------


def _fold_out(k: int, hero: float, others: list[float]) -> str:
    """A synthetic hand: the first to act bets $60 and everyone else folds (stacks in $)."""
    n = 1 + len(others)
    names = ["Hero"] + [f"{k:04x}{j:04x}" for j in range(n - 1)]
    stacks = [hero, *others]
    at = k % n  # (where Hero sits)
    seats = [(no, names[(no - 1 - at) % n], stacks[(no - 1 - at) % n]) for no in range(1, n + 1)]
    button = (k // n) % n + 1
    first = button % n  # (the seat left of the button, as an index into seats 1..n)
    acts = [(seats[first][1], "bets", 60, False)] + [
        (seats[(first + j) % n][1], "folds", None, False) for j in range(1, n)]
    return gg_hand(hid=str(9200000000 + k), button=button, seats=seats, deal={"Hero": "Kh Jd Td 8d 2s"},
                   board_a="6h 6c Qc", board_b="9c 5c 4d", streets={"flop": acts},
                   when=f"2026/10/01 {k // 60:02d}:{k % 60:02d}:00")


def _session_of(c, settings):
    st = c.get("/trainer/state").json()["state"]
    r = c.post("/trainer/settings", json={**st["trainer"]["settings"], **settings})
    assert r.status_code == 200, r.text
    return r.json()["state"]


def test_my_tables_deal_like_the_tables_in_your_hands(server, pub, store, monkeypatch):
    """(owner) "automatically updates the seat and stack size distributions based on the
    individual user's hand histories … typical stack sizes for opponents and for
    themselves": the store's profile of YOUR latest hands — your stack ($400 with auto
    top-up, more when you won) apart from your opponents' — is what My tables deals from;
    it follows every upload and deletion; without a subscription it's typical tables."""
    import random

    email = "regular@example.com"
    c = _login(server, email)
    _comp(pub, email)
    rng = random.Random(3)
    mine_cfg = []
    texts = []
    for k in range(60):
        n = rng.choice([4, 5, 6, 6])
        hero = 400 if rng.random() < 0.6 else rng.choice([420, 515.5, 760, 1320.25])
        others = [rng.choice([400, 150, 233.4, 512, 880, 2400]) for _ in range(n - 1)]
        mine_cfg.append((n, hero, others))
        texts.append(_fold_out(k, hero, others))
    assert _upload(c, store, {"s.txt": "\n\n".join(texts)})["added"] == 60
    uid = _uid(pub, email)
    prof = store.my_tables(uid)
    assert prof["paid"] is True and prof["hands"] == 60 and prof["min_hands"] == store.PROFILE_MIN_HANDS
    assert prof["antes"] == [[3.0, 1.0]]
    want_seats = collections.Counter(n for n, _h, _o in mine_cfg)
    assert {n: p for n, p in prof["seats"]} == {n: round(c_ / 60, 4) for n, c_ in want_seats.items()}
    assert dict(map(tuple, prof["hero"]["atoms"]))[20.0] == round(sum(h == 400 for _n, h, _o in mine_cfg) / 60, 4)
    v = c.get("/trainer/my_tables").json()
    assert v["source"] == "mine" and v["hands"] == 60 and v["hero"]["median"] == prof["hero"]["median"]

    drawn = []
    real_draw = hr.draw_table

    def spy(p, r):
        drawn.append(p["hands"])
        return real_draw(p, r)

    monkeypatch.setattr(hr, "draw_table", spy)
    _session_of(c, {"tables": "mine"})
    ts = pub._REGISTRY.peek(uid).trainer
    for _ in range(25):
        assert c.post("/trainer/new_hand").status_code == 200
        cfg, h = ts.hand.config, ts.hand.hero_seat
        hero_bb = cfg.starting_stacks[h] / 10000
        assert 20.0 <= hero_bb <= 1320.25 / 20 and cfg.ante == 3 * 10000 and cfg.num_seats in (4, 5, 6)
        assert all(7.5 <= s / 10000 <= 120 for i, s in enumerate(cfg.starting_stacks) if i != h)
    assert drawn and set(drawn) == {60}

    # new hands change it; deleting them takes it away
    more = [_fold_out(100 + k, 400, [400, 400]) for k in range(10)]
    _upload(c, store, {"more.txt": "\n\n".join(more)})
    assert store.my_tables(uid)["hands"] == 70
    assert c.post("/games/api/review/delete", json={"confirm": True}).status_code == 200
    v = c.get("/trainer/my_tables").json()
    assert v["source"] == "typical" and v["paid"] is True and v["hands"] == 0

    # someone without a subscription: typical tables, and the dialog says why
    free = _login(server, "trialist@example.com")
    v = free.get("/trainer/my_tables").json()
    assert v == {"source": "typical", "review": True, "paid": False, "hands": 0, "min_hands": store.PROFILE_MIN_HANDS}
    _session_of(free, {"tables": "mine"})
    assert free.post("/trainer/new_hand").status_code == 200
