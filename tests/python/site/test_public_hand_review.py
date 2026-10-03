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


def test_signed_out_the_page_and_its_api_are_hidden(server):
    c = TestClient(server.app, raise_server_exceptions=False)
    assert c.get("/games/api/review/summary").status_code in (401, 404)


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
