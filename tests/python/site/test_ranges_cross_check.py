"""(TOOL-029) The app's range parser and the native solver's parser agree.

``plo5bp.cfr_app.ranges`` (grid, validation, canonical string) and the solver's
``Range::parse`` are two implementations of one grammar. Random range texts from
the shared grammar must mean the same combos with the same weights to both, and
the canonical string the app sends must mean to the solver exactly what the
user's text meant to the app.
"""

from __future__ import annotations

import random

import pytest

from plo5bp.cfr_app.ranges import RangeError, parse_range
from plo5bp.gto.cfr_api import rust_cfr_available
from plo5bp.gto.preflop_class import cards_to_combo

pytestmark = pytest.mark.skipif(not rust_cfr_available(), reason="native CFR not built")

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def _native(spec: str, board) -> list[float]:
    from plo5bp.gto.cfr_api import native_parse_range

    return native_parse_range(spec, board)


def _token(rng: random.Random) -> str:
    r = lambda: RANKS[rng.randrange(13)]  # noqa: E731
    kind = rng.randrange(10)
    if kind == 0:
        a = r()
        return a + a
    if kind == 1:
        hi, lo = sorted(rng.sample(range(13), 2), reverse=True)
        return RANKS[hi] + RANKS[lo] + rng.choice(["s", "o", ""])
    if kind == 2:
        return RANKS[rng.randrange(13)] * 2 + "+"
    if kind == 3:
        hi, lo = sorted(rng.sample(range(13), 2), reverse=True)
        return RANKS[hi] + RANKS[lo] + rng.choice(["s", "o", ""]) + "+"
    if kind == 4:
        a, b = rng.sample(range(13), 2)
        return f"{RANKS[a]}{RANKS[a]}-{RANKS[b]}{RANKS[b]}"
    if kind == 5:  # same top card, kicker run
        hi = rng.randrange(2, 13)
        k1, k2 = rng.sample(range(hi), 2)
        suf = rng.choice(["s", "o", ""])
        return f"{RANKS[hi]}{RANKS[k1]}{suf}-{RANKS[hi]}{RANKS[k2]}{suf}"
    if kind == 6:  # constant gap ladder
        gap = rng.randrange(1, 4)
        h1, h2 = rng.sample(range(gap, 13), 2)
        suf = rng.choice(["s", "o", ""])
        return f"{RANKS[h1]}{RANKS[h1 - gap]}{suf}-{RANKS[h2]}{RANKS[h2 - gap]}{suf}"
    if kind == 7:  # explicit combo
        c0, c1 = rng.sample(range(52), 2)
        return RANKS[c0 // 4] + SUITS[c0 % 4] + RANKS[c1 // 4] + SUITS[c1 % 4]
    if kind == 8:
        return f"{rng.randrange(1326):04d}"
    return f"#{rng.randrange(1326)}"


def _spec(rng: random.Random) -> str:
    parts = []
    for _ in range(rng.randrange(1, 7)):
        tok = _token(rng)
        if rng.random() < 0.4:
            tok += ":" + rng.choice(["0.5", "0.25", "1", "0.75", "0.1"])
        parts.append(tok)
    seps = [", ", " ", ",", ";", "\n"]
    return "".join(p + rng.choice(seps) for p in parts[:-1]) + parts[-1]


def _app_weights(pr) -> list[float]:
    out = [0.0] * 1326
    for cid, w in pr.weights.items():
        out[cid] = w
    return out


@pytest.mark.parametrize("seed", range(4))
def test_random_ranges_mean_the_same_to_both_parsers(seed):
    rng = random.Random(seed)
    checked = 0
    for _ in range(150):
        spec = _spec(rng)
        board = sorted(rng.sample(range(52), rng.choice([0, 3, 4, 5])))
        try:
            pr = parse_range(spec, board)
        except RangeError:
            # Then the native parser must refuse it too (e.g. nothing live left).
            with pytest.raises(ValueError):
                _native(spec, board)
            continue
        native = _native(spec, board)
        assert native == pytest.approx(_app_weights(pr), abs=0), spec
        # The string the app actually sends means the same thing again.
        assert _native(pr.canonical(), board) == pytest.approx(native, abs=0), spec
        checked += 1
    assert checked > 100


def test_the_full_range_words_agree():
    for word in ("", "random", "100%", "any", "*"):
        pr = parse_range(word, [0, 5, 10])
        assert pr.full and pr.canonical() == ""
        native = _native(word, [0, 5, 10])
        assert sum(native) == pr.summary()["combos"] == 1176  # C(49, 2)


def test_two_digit_numbers_are_hands_for_both():
    for text in ("22", "99", "72"):
        pr = parse_range(text, [])
        assert _native(text, []) == _app_weights(pr)
    assert sum(_native("AsKs", [])) == 1 and _native("AsKs", [])[cards_to_combo(51, 47)] == 1
