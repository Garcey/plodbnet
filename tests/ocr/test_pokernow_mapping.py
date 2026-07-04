"""Tests for the PokerNow DOM-snapshot → FrameState mapper.

Payloads mirror the ``pokernow.v1`` schema the Tampermonkey userscript emits,
with values taken from a real PLO5 double-board bomb-pot hand captured live
from pokernow.com (2026-06-28): heads-up Miles (seat 1) vs JJ (seat 6).
"""

from __future__ import annotations

from plo5bp.ocr.pokernow import map_payload
from plo5bp.ocr.types import Card


def _river_payload() -> dict:
    """The paused-river snapshot: both boards full, no live bets."""
    return {
        "schema": "pokernow.v1",
        "variant": "plo5",
        "bombPot": True,
        "potDollars": 12.0,
        "button": {"seat": 6},
        "boards": [
            {"run": "1", "cards": ["2c", "5h", "Ac", "6d", "3h"]},
            {"run": "2", "cards": ["7h", "Ks", "6s", "Kh", "Qs"]},
        ],
        "heroCards": ["Ts", "As", "4h", "3d", "2d"],
        "seats": [
            {
                "seat": 1, "name": "Miles", "isHero": True, "isActor": True,
                "angleCW": 147, "stackDollars": 74.0, "folded": False,
                "betText": None, "betDollars": None,
                "cards": ["Ts", "As", "4h", "3d", "2d"],
            },
            {
                "seat": 6, "name": "JJ", "isHero": False, "isActor": False,
                "angleCW": 328, "stackDollars": 74.0, "folded": False,
                "betText": None, "betDollars": None,
                "cards": [None, None, None, None, None],
            },
        ],
    }


def test_basic_fields_and_units():
    r = map_payload(_river_payload())
    assert r.variant == "plo5"
    assert r.bomb_pot is True
    assert r.num_seats == 2
    fs = r.frame
    # cents = dollars * 100
    assert fs.pot_total_chips == 1200
    assert fs.board_a == tuple(Card.parse(c) for c in ("2c", "5h", "Ac", "6d", "3h"))
    assert fs.board_b == tuple(Card.parse(c) for c in ("7h", "Ks", "6s", "Kh", "Qs"))
    assert fs.hero_hole == tuple(Card.parse(c) for c in ("Ts", "As", "4h", "3d", "2d"))


def test_seat_ordering_hero_zero_then_clockwise():
    r = map_payload(_river_payload())
    # Hero (physical seat 1) becomes engine seat 0; JJ (seat 6) → engine 1.
    assert r.physical_to_engine == {1: 0, 6: 1}
    assert r.seat_names == {0: "Miles", 1: "JJ"}
    assert r.frame.seats[0].is_actor is True
    assert r.frame.seats[0].stack_chips == 7400
    # Villain face-down: no hero cards leak through the seat obs path.
    assert r.frame.seats[1].folded is False


def test_button_translates_physical_to_engine():
    r = map_payload(_river_payload())
    # Button at physical seat 6 → engine seat 1.
    assert r.frame.button_seat == 1


def test_numeric_bet_and_check_committed():
    p = _river_payload()
    # Hero raises to 24.00 this street; villain has 6.00 in front.
    p["seats"][0]["betDollars"] = 24.0
    p["seats"][1]["betDollars"] = 6.0
    r = map_payload(p)
    assert r.frame.seats[0].committed_chips == 2400
    assert r.frame.seats[1].committed_chips == 600

    # A "check" verb means zero contributed this street.
    p2 = _river_payload()
    p2["seats"][0]["betText"] = "check"
    r2 = map_payload(p2)
    assert r2.frame.seats[0].committed_chips == 0
    # No bet element at all → unknown (None), so the reconstructor infers.
    assert r2.frame.seats[1].committed_chips is None


def test_fold_keeps_seat_with_empty_cards():
    p = _river_payload()
    # Villain folds: PokerNow strips the card elements and sets the fold class.
    p["seats"][1]["folded"] = True
    p["seats"][1]["cards"] = []
    r = map_payload(p)
    # Folded seat stays in the hand's engine seat set.
    assert r.num_seats == 2
    assert r.frame.seats[1].folded is True
    assert all(c is None for c in r.frame.hero_hole) is False  # hero unaffected


def test_all_in_stack_null_marks_all_in():
    p = _river_payload()
    # All-in shove: PokerNow renders the stack empty (null), bet shows total.
    p["seats"][0]["stackDollars"] = None
    p["seats"][0]["betDollars"] = 86.0
    r = map_payload(p)
    # All-in → $0 behind (NOT None): the reconstructor needs the stack drop to
    # corroborate the shove, and a stack→0 raise routes through the short-shove
    # path. None would make the corroboration guard discard the all-in bet.
    assert r.frame.seats[0].stack_chips == 0
    assert r.frame.seats[0].all_in is True
    assert r.frame.seats[0].committed_chips == 8600


def test_sitting_out_seat_excluded_from_hand():
    p = _river_payload()
    # A third player seated but not dealt in (no cards, not folded).
    p["seats"].append({
        "seat": 3, "name": "Idle", "isHero": False, "isActor": False,
        "angleCW": 240, "stackDollars": 50.0, "folded": False,
        "betText": None, "betDollars": None, "cards": [],
    })
    r = map_payload(p)
    assert r.num_seats == 2
    assert 3 not in r.physical_to_engine


def test_partial_board_flop_only():
    p = _river_payload()
    p["boards"] = [
        {"run": "1", "cards": ["4c", "Jc", "Ad"]},
        {"run": "2", "cards": ["9h", "2s", "Ts"]},
    ]
    r = map_payload(p)
    fs = r.frame
    assert fs.board_a[:3] == tuple(Card.parse(c) for c in ("4c", "Jc", "Ad"))
    assert fs.board_a[3] is None and fs.board_a[4] is None
