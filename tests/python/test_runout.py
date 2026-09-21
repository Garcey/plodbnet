"""Side-pot layers, winning combos, and all-in equity."""

from __future__ import annotations

from plo5bp.ui.hand_describe import best_combo
from plo5bp.ui.runout import board_equities, build_awards, pot_layers


def _c(rank, suit):
    return rank * 4 + suit


def test_pot_layers_deepest_first():
    layers = pot_layers([500, 500, 1500], [False, False, False])
    assert [ly["chips"] for ly in layers] == [1000, 1500]
    assert layers[0]["eligible"] == [2]
    assert layers[1]["eligible"] == [0, 1, 2]


def test_awards_match_engine_side_pot_split_boards():
    # Mirrors rust double_board::side_pot_plus_split_boards.
    holes = [
        [_c(12, 0), _c(12, 1), _c(12, 2), _c(12, 3), _c(0, 0)],
        [_c(6, 1), _c(6, 2), _c(0, 2), _c(1, 1), _c(2, 3)],
        [_c(11, 0), _c(11, 1), _c(11, 2), _c(11, 3), _c(3, 0)],
    ]
    folded = [False, False, False]
    commit = [500, 500, 1500]
    board_a = [_c(10, 0), _c(10, 1), _c(9, 2), _c(8, 2), _c(7, 3)]
    board_b = [_c(6, 0), _c(5, 2), _c(3, 1), _c(0, 3), _c(1, 0)]
    awards = build_awards(holes, folded, commit, board_a, board_b, 0)
    won = [0, 0, 0]
    for step in awards:
        for k, v in step["shares"].items():
            won[int(k)] += int(v)
    assert won == [750, 750, 1000]
    assert sum(won) == sum(commit)
    # Deepest side pot first, then main; A then B within a layer.
    assert awards[0]["eligible"] == [2]
    assert awards[0]["uncontested"] is True


def test_best_combo_broadway_uses_those_board_cards():
    # Board AKQJ2. JT plays broadway using A,K,Q; QT uses A,K,J.
    board = [_c(12, 0), _c(11, 1), _c(10, 2), _c(9, 3), _c(0, 0)]  # As Kd Qh Js 2c
    jt = best_combo(
        [_c(9, 1), _c(8, 2), _c(1, 2), _c(2, 3), _c(3, 0)],  # Jd Th
        board,
    )
    qt = best_combo(
        [_c(10, 0), _c(8, 1), _c(1, 0), _c(2, 1), _c(3, 2)],  # Qc Td
        board,
    )
    assert jt and qt
    assert jt["label"] == "a straight 10-A"
    assert qt["label"] == "a straight 10-A"
    assert set(jt["hole"]) != set(qt["hole"])
    assert set(jt["board"]) != set(qt["board"])


def test_equity_river_is_certain():
    # Same two broadway hands on a complete board — chop 50/50 on that board.
    board = [_c(12, 0), _c(11, 1), _c(10, 2), _c(9, 3), _c(0, 0)]
    holes = {
        0: [_c(9, 1), _c(8, 2), _c(1, 2), _c(2, 3), _c(3, 0)],
        1: [_c(10, 0), _c(8, 1), _c(1, 0), _c(2, 1), _c(3, 2)],
    }
    eq = board_equities(holes, board, board, samples=8, seed=1)
    assert eq[0]["a"] == 0.5
    assert eq[1]["a"] == 0.5


# --- review 2026-09-20 G3: per-board marginal enumeration ---------------------


def _joint_reference(holes, board_a, board_b):
    """The ORIGINAL method: enumerate every joint completion of both boards
    (board B drawn from what board A left) with plain `best_combo`."""
    from itertools import combinations

    dead = {c for h in holes.values() for c in h} | set(board_a) | set(board_b)
    deck = [c for c in range(52) if c not in dead]
    na, nb = 5 - len(board_a), 5 - len(board_b)
    acc = {s: [0.0, 0.0] for s in holes}
    n = 0

    memo = {}  # (seat, board) -> score; only the evaluation is memoized,
    #            the JOINT weighting below is what is being cross-checked.

    def credit(board, which):
        scores = {}
        for s, h in holes.items():
            key = (s, tuple(board))
            if key not in memo:
                combo = best_combo(h, board)
                memo[key] = (combo["category"], tuple(combo["tiebreak"]))
            scores[s] = memo[key]
        top = max(scores.values())
        winners = [s for s, sc in scores.items() if sc == top]
        for s in winners:
            acc[s][which] += 1.0 / len(winners)

    for xa in combinations(deck, na):
        rest = [c for c in deck if c not in xa]
        for xb in combinations(rest, nb):
            credit(list(board_a) + list(xa), 0)
            credit(list(board_b) + list(xb), 1)
            n += 1
    return {s: {"a": round(v[0] / n, 4), "b": round(v[1] / n, 4)} for s, v in acc.items()}


def _deal(rng, n_players, len_a, len_b):
    deck = list(range(52))
    rng.shuffle(deck)
    holes = {i: deck[i * 5:(i + 1) * 5] for i in range(n_players)}
    rest = deck[n_players * 5:]
    return holes, rest[:len_a], rest[len_a:len_a + len_b]


def test_equity_marginal_enumeration_matches_joint_enumeration():
    """One enumeration PER BOARD is exact: a board's completion is uniform over
    the stub whatever the other board takes. Checked against the brute-force
    joint enumeration (small stubs keep that affordable)."""
    import random

    rng = random.Random(20260920)
    cases = [(8, 3, 3), (8, 4, 4), (7, 3, 3), (7, 3, 4), (6, 4, 4), (8, 5, 3), (2, 4, 4), (3, 4, 4)]
    for n_players, la, lb in cases:
        holes, ba, bb = _deal(rng, n_players, la, lb)
        got = board_equities(holes, ba, bb, seed=5)
        want = _joint_reference(holes, ba, bb)
        for s in holes:
            for k in ("a", "b"):
                assert abs(got[s][k] - want[s][k]) <= 1e-4, (n_players, la, lb, s, k)
        assert abs(sum(v["a"] for v in got.values()) - 1) < 1e-3
        assert abs(sum(v["b"] for v in got.values()) - 1) < 1e-3


def test_equity_is_cheap_and_deterministic():
    import random
    import time

    rng = random.Random(7)
    for n_players in (2, 3, 4, 6):
        holes, ba, bb = _deal(rng, n_players, 3, 3)  # flop all-in: the worst case
        t0 = time.perf_counter()
        first = board_equities(holes, ba, bb, seed=1)
        took = time.perf_counter() - t0
        # It was 0.5-2.9 s (and 250 noisy samples); now exact in well under that.
        assert took < 1.5, (n_players, took)
        assert board_equities(holes, ba, bb, seed=999) == first  # exact => seed-free


def test_equity_ignores_sentinels_and_only_counts_given_holes():
    board = [_c(12, 0), _c(11, 1), _c(10, 2), _c(9, 3), _c(0, 0)]
    holes = {
        0: [_c(9, 1), _c(8, 2), _c(1, 2), _c(2, 3), _c(3, 0)],
        1: [_c(10, 0), _c(8, 1), _c(1, 0), _c(2, 1), _c(3, 2)],
        4: [-1, -1, -1, -1, -1],  # a face-down hand is not a contender
    }
    eq = board_equities(holes, board, board + [-1])
    assert eq[0] == {"a": 0.5, "b": 0.5} and eq[1] == {"a": 0.5, "b": 0.5}
    assert eq[4] == {"a": 0.0, "b": 0.0}
    assert board_equities({}, board, board) == {}


def test_equity_sampling_fallback_for_long_runouts():
    """> 2 missing cards per board never happens in a bomb pot (hands start on
    the flop); the sampled path still has to behave."""
    import random

    holes, _, _ = _deal(random.Random(1), 2, 0, 0)
    eq = board_equities(holes, [], [], samples=40, seed=3)
    assert eq == board_equities(holes, [], [], samples=40, seed=3)  # seeded
    assert abs(eq[0]["a"] + eq[1]["a"] - 1) < 1e-3
