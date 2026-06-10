"""Pure scoring-rubric tests for trainer.score_move (no model, no env)."""

from __future__ import annotations

import math

import pytest

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.ui.trainer import SCORING, score_move


def test_pure_fold_is_best():
    r = score_move([0.97, 0.02, 0.01], 1.5, 1.5, 0, 0, GATE_FOLD, 0)
    assert r["category"] == "best"
    assert r["score"] == pytest.approx(100.0)


def test_mixed_check_bet_user_bets_well_sized_is_correct():
    # Network mixes check 55 / bet 45; user bets at the Beta mode.
    alpha, beta = 3.0, 5.0
    mode = (alpha - 1) / (alpha + beta - 2)
    lo, hi = 10_000, 100_000
    chips = round(lo + mode * (hi - lo))
    r = score_move([0.0, 0.55, 0.45], alpha, beta, lo, hi, GATE_RAISE, chips)
    assert r["gate_ratio"] == pytest.approx(0.45 / 0.55, rel=1e-6)
    assert r["size_q"] == pytest.approx(1.0, abs=1e-3)
    assert r["score"] == pytest.approx(100 * 0.45 / 0.55, rel=1e-2)
    assert r["category"] == "correct"  # second-best line, well sized


def test_call_into_pure_fold_is_blunder():
    r = score_move([0.99, 0.005, 0.005], 1.5, 1.5, 0, 0, GATE_CHECK_CALL, 0)
    assert r["category"] == "blunder"
    assert r["score"] < SCORING["wrong_min"]


def test_right_gate_wrong_size_drags_score():
    # Betting is the argmax gate, but the user picks the far tail.
    alpha, beta = 2.0, 8.0
    lo, hi = 10_000, 100_000
    r_tail = score_move([0.05, 0.15, 0.80], alpha, beta, lo, hi, GATE_RAISE, hi)
    assert r_tail["size_q"] < 0.05
    # Floor keeps it near 100 * size_floor.
    assert r_tail["score"] == pytest.approx(100 * SCORING["size_floor"], rel=0.15)
    assert r_tail["category"] in ("inaccuracy", "wrong")
    # Same gate at the mode scores ~100 / best.
    chips_mode = round(lo + (alpha - 1) / (alpha + beta - 2) * (hi - lo))
    r_mode = score_move([0.05, 0.15, 0.80], alpha, beta, lo, hi, GATE_RAISE, chips_mode)
    assert r_mode["category"] == "best"


def test_degenerate_ranges_have_no_size_penalty():
    # Short shove (min == 0): chips are moot.
    r = score_move([0.1, 0.1, 0.8], 2.0, 8.0, 0, 50_000, GATE_RAISE, 50_000)
    assert r["size_q"] == 1.0
    # Single-point range (min == max).
    r2 = score_move([0.1, 0.1, 0.8], 2.0, 8.0, 50_000, 50_000, GATE_RAISE, 50_000)
    assert r2["size_q"] == 1.0


def test_uniform_beta_has_no_size_penalty():
    r = score_move([0.0, 0.2, 0.8], 1.0, 1.0, 10_000, 100_000, GATE_RAISE, 99_000)
    assert r["size_q"] == 1.0
    assert r["category"] == "best"


def test_category_bands():
    def cat(ratio):
        # user = check_call with P scaled against a 0.5 argmax fold.
        p_best = 0.5
        p_user = ratio * p_best
        rest = 1.0 - p_best - p_user
        return score_move([p_best, p_user, max(rest, 0.0)], 1.5, 1.5, 0, 0,
                          GATE_CHECK_CALL, 0)

    assert cat(0.90)["category"] == "correct"   # high score but not argmax
    assert cat(0.62)["category"] == "correct"
    assert cat(0.45)["category"] == "inaccuracy"
    assert cat(0.20)["category"] == "wrong"
    assert cat(0.05)["category"] == "blunder"


def test_tied_argmax_is_not_best():
    # Ratio 1.0 against a tied gate still isn't "best" unless it IS argmax.
    r = score_move([0.45, 0.45, 0.10], 1.5, 1.5, 0, 0, GATE_CHECK_CALL, 0)
    assert r["score"] == pytest.approx(100.0)
    assert r["category"] == "correct"


def test_low_probability_overrides_to_blunder():
    r = score_move([0.015, 0.95, 0.035], 1.5, 1.5, 0, 0, GATE_FOLD, 0)
    assert r["category"] == "blunder"


def test_score_clamped():
    r = score_move([0.5, 0.5, 0.0], 1.5, 1.5, 0, 0, GATE_CHECK_CALL, 0)
    assert 0.0 <= r["score"] <= 100.0
    assert math.isfinite(r["score"])
