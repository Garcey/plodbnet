"""Tests for the PokerNow DOM-snapshot → FrameState mapper.

Payloads mirror the ``pokernow.v1`` schema the Tampermonkey userscript emits,
with values taken from a real PLO5 double-board bomb-pot hand captured live
from pokernow.com (2026-06-28): heads-up Miles (seat 1) vs JJ (seat 6).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from plo5bp.ocr.pokernow import PokerNowPayloadError, map_payload
from plo5bp.ocr.types import Card

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pokernow"


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
    # All-in shove: PokerNow renders "All In" instead of the stack number, so
    # the userscript sends stackDollars=null + allIn=true; bet shows the total.
    # UPDATED 2026-09-20 (review): the explicit `allIn` flag is now REQUIRED —
    # this payload used to omit it and rely on the mapper inferring all-in from
    # the null stack alone (see test_null_stack_without_all_in_flag_*).
    p["seats"][0]["stackDollars"] = None
    p["seats"][0]["allIn"] = True
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


# --- review 2026-09-20: explicit all-in, schema validation, fixture -------


def test_null_stack_without_all_in_flag_is_unknown_not_all_in():
    """A missing `.normal-value` (null stack) is NOT all-in unless the DOM
    says so. It used to map to stack 0 — a full-stack "drop" that corroborates
    any bet oval and routes the seat through the all-in paths."""
    p = _river_payload()
    p["seats"][1]["stackDollars"] = None  # mid-render / non-numeric state
    r = map_payload(p)
    assert r.frame.seats[1].all_in is False
    assert r.frame.seats[1].stack_chips is None  # unknown, NOT 0

    # An explicit false is honoured the same way.
    p["seats"][1]["allIn"] = False
    r = map_payload(p)
    assert r.frame.seats[1].all_in is False
    assert r.frame.seats[1].stack_chips is None


def test_folded_seat_is_never_all_in():
    p = _river_payload()
    p["seats"][1].update(folded=True, cards=[], stackDollars=None, allIn=True)
    r = map_payload(p)
    assert r.frame.seats[1].folded is True
    assert r.frame.seats[1].all_in is False
    assert r.frame.seats[1].stack_chips is None


def _mutated(path, value=None, *, delete=False):
    """Deep-copy the river payload and set/delete one nested key."""
    p = copy.deepcopy(_river_payload())
    node = p
    for key in path[:-1]:
        node = node[key]
    if delete:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return p


_BAD_PAYLOADS = [
    # (label, payload, fragment the error message must name)
    ("not an object", ["seats"], "payload"),
    ("wrong schema", _mutated(["schema"], "pokernow.v0"), "schema"),
    ("seats missing", _mutated(["seats"], delete=True), "seats"),
    ("seats not a list", _mutated(["seats"], {"1": {}}), "seats"),
    ("seat not an object", _mutated(["seats", 0], "Miles"), "seats[0]"),
    ("seat number null", _mutated(["seats", 0, "seat"], None), "seats[0].seat"),
    ("seat number string", _mutated(["seats", 1, "seat"], "six"), "seats[1].seat"),
    ("seat number bool", _mutated(["seats", 1, "seat"], True), "seats[1].seat"),
    ("seat number fractional", _mutated(["seats", 1, "seat"], 6.5), "seats[1].seat"),
    ("duplicate seat", _mutated(["seats", 1, "seat"], 1), "duplicates"),
    ("stack string", _mutated(["seats", 0, "stackDollars"], "74"), "stackDollars"),
    ("stack negative", _mutated(["seats", 0, "stackDollars"], -1.0), "stackDollars"),
    ("stack inf", _mutated(["seats", 0, "stackDollars"], float("inf")), "stackDollars"),
    ("stack nan", _mutated(["seats", 0, "stackDollars"], float("nan")), "stackDollars"),
    ("bet string", _mutated(["seats", 0, "betDollars"], "check"), "betDollars"),
    ("betText number", _mutated(["seats", 0, "betText"], 5), "betText"),
    ("cards string", _mutated(["seats", 0, "cards"], "TsAs4h3d2d"), "cards"),
    ("card not a string", _mutated(["seats", 0, "cards"], [1, 2, 3, 4, 5]), "cards[0]"),
    ("angle string", _mutated(["seats", 0, "angleCW"], "north"), "angleCW"),
    ("stack key missing", _mutated(["seats", 0, "stackDollars"], delete=True), "stackDollars"),
    ("bet key missing", _mutated(["seats", 1, "betDollars"], delete=True), "betDollars"),
    ("cards key missing", _mutated(["seats", 1, "cards"], delete=True), "cards"),
    ("seat key missing", _mutated(["seats", 1, "seat"], delete=True), "'seat'"),
    ("boards not a list", _mutated(["boards"], {"run": "1"}), "boards"),
    ("board not an object", _mutated(["boards", 0], ["2c"]), "boards[0]"),
    ("board cards string", _mutated(["boards", 0, "cards"], "2c5hAc"), "boards[0].cards"),
    ("heroCards string", _mutated(["heroCards"], "TsAs"), "heroCards"),
    ("pot string", _mutated(["potDollars"], "12"), "potDollars"),
    ("pot inf", _mutated(["potDollars"], float("inf")), "potDollars"),
    ("button string", _mutated(["button"], "six"), "button"),
    ("button.seat string", _mutated(["button"], {"seat": "6"}), "button.seat"),
    ("two heroes", _mutated(["seats", 1, "isHero"], True), "isHero"),
]


@pytest.mark.parametrize(
    "payload, fragment",
    [(p, frag) for _label, p, frag in _BAD_PAYLOADS],
    ids=[label for label, _p, _frag in _BAD_PAYLOADS],
)
def test_malformed_payload_raises_a_clear_value_error(payload, fragment):
    """Every malformed shape is a `PokerNowPayloadError` naming the bad path —
    never a TypeError / OverflowError / KeyError from deep inside the mapper
    (which the server surfaces as a 500 on every heartbeat)."""
    with pytest.raises(PokerNowPayloadError) as exc:
        map_payload(payload)
    assert isinstance(exc.value, ValueError)  # server maps ValueError -> 400
    assert fragment in str(exc.value), str(exc.value)


def test_optional_keys_may_be_absent_or_null():
    """Leniency that must survive validation: optional keys, explicit nulls,
    a bare-int button, and junk *card strings* (mid-render DOM → face-down)."""
    p = _river_payload()
    for key in ("schema", "variant", "bombPot", "potDollars", "button", "boards", "heroCards"):
        del p[key]
    for seat in p["seats"]:
        for key in ("name", "isActor", "angleCW", "betText", "folded"):
            seat.pop(key, None)
    r = map_payload(p)
    assert r.num_seats == 2
    assert r.frame.button_seat is None
    assert r.frame.pot_total_chips is None
    assert r.frame.board_a == (None,) * 5
    # heroCards absent → falls back to the hero seat's own cards.
    assert r.frame.hero_hole == tuple(Card.parse(c) for c in ("Ts", "As", "4h", "3d", "2d"))

    p2 = _river_payload()
    p2["button"] = 6                       # bare int form
    p2["seats"][0]["angleCW"] = None       # explicit null
    p2["seats"][1]["cards"] = None         # null == no cards → not in hand
    p2["boards"][0]["cards"] = ["2c", "??", "Ac", None, "10x"]  # junk strings
    r2 = map_payload(p2)
    assert r2.num_seats == 1               # only hero holds cards now
    assert r2.frame.board_a == (Card.parse("2c"), None, Card.parse("Ac"), None, None)


def test_fixture_file_is_valid_v1_and_maps():
    """`fixtures/pokernow/river_paused_heads_up.json` is the on-disk example of
    the wire format. It had rotted to the pre-v1 key spelling (`pot`, `stack`,
    `bet`, bare-int `button`) — which the unvalidated mapper happily turned into
    two all-in players with unknown stacks. Keep it loadable."""
    payload = json.loads(
        (FIXTURE_DIR / "river_paused_heads_up.json").read_text(encoding="utf-8")
    )
    r = map_payload(payload)
    assert r.num_seats == 2
    assert r.physical_to_engine == {1: 0, 6: 1}
    assert r.frame.button_seat == 1
    assert r.frame.pot_total_chips == 1200
    assert [s.stack_chips for s in r.frame.seats] == [7400, 7400]
    assert [s.all_in for s in r.frame.seats] == [False, False]
    assert r.frame.seats[0].is_actor is True
    assert r.frame.hero_hole == tuple(Card.parse(c) for c in ("Ts", "As", "4h", "3d", "2d"))
    assert r.frame.board_b == tuple(Card.parse(c) for c in ("7h", "Ks", "6s", "Kh", "Qs"))


def test_pre_v1_key_spelling_is_rejected():
    """The old fixture's shape must fail loudly, not map to garbage."""
    stale = {
        "schema": "pokernow.v1", "variant": "plo5", "bombPot": True,
        "pot": 12, "button": 6,
        "boards": [{"run": "1", "cards": ["2c", "5h", "Ac"]}],
        "seats": [
            {"seat": 1, "name": "Miles", "stack": 74, "bet": None,
             "cards": ["Ts", "As", "4h", "3d", "2d"], "isHero": True},
            {"seat": 6, "name": "JJ", "stack": 74, "bet": None,
             "cards": [None] * 5},
        ],
    }
    with pytest.raises(PokerNowPayloadError, match="stackDollars"):
        map_payload(stale)
