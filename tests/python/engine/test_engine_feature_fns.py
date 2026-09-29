"""Standalone engine feature functions: bad input is a ValueError, never a panic
(ENG-029), and `plo_board_strength_batch` scores what it says (TEST-024; the
full randomized check against an explicit evaluator lives in the Rust test
`board_strength_tests`)."""

from __future__ import annotations

import numpy as np
import pytest

from plo5bp._engine import compute_double_board_payout, plo_board_strength_batch


def c(rank: int, suit: int) -> int:
    return rank * 4 + suit


def test_double_board_payout_validates_instead_of_panicking() -> None:
    n = 3
    holes = list(range(5 * n))
    board_a = [40, 41, 42, 43, 44]
    board_b = [45, 46, 47, 48, 49]
    won = compute_double_board_payout(holes, [False] * n, [100] * n, board_a, board_b, 0)
    assert sum(won) == 300
    # 33 seats used to hit an assert! inside the payout (PanicException escapes
    # `except Exception`); now any count outside 2..=8 is a ValueError.
    for seats in (1, 9, 33):
        with pytest.raises(ValueError, match="num_seats"):
            compute_double_board_payout([0] * 5 * seats, [False] * seats, [1] * seats, board_a, board_b, 0)
    dup = holes.copy()
    dup[3] = board_a[0]
    with pytest.raises(ValueError, match="twice"):
        compute_double_board_payout(dup, [False] * n, [100] * n, board_a, board_b, 0)


def test_board_strength_scores_the_best_two_plus_three() -> None:
    royal = [c(12, 3), c(11, 3), c(0, 0), c(1, 1), c(2, 2)]  # As Ks 2c 3d 4h
    board = [c(10, 3), c(9, 3), c(8, 3), 255, 255]  # Qs Js Ts
    holes = np.array([royal, royal, [255, 1, 2, 3, 5]], dtype=np.uint8)
    boards = np.array([board, board, board], dtype=np.uint8)
    lens = np.array([3, 2, 3], dtype=np.uint8)
    out = plo_board_strength_batch(holes, boards, lens)
    assert out.dtype == np.uint16
    assert list(out) == [7462, 0, 0]  # royal flush; short board; empty hole card
    with pytest.raises(ValueError, match="card index"):
        plo_board_strength_batch(np.full((1, 5), 60, np.uint8), boards[:1], lens[:1])
    with pytest.raises(ValueError, match="holes must be"):
        plo_board_strength_batch(np.zeros((2, 7), np.uint8), boards[:2], lens[:2])
