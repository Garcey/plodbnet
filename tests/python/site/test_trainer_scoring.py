"""Pure scoring-rubric tests for trainer.score_move (no model, no env).

Since 2026-10-03 a move is graded on a LOG scale of how much less often the network plays
it than its favourite (`trainer._grade`): best = at least 3/4 as often, correct = 1/4,
inaccuracy = 1/10, wrong = 1/50, blunder = rarer or under 2% outright."""

from __future__ import annotations

import math

import pytest

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.ui.trainer import SCORE_SPAN, SCORING, score_move


def test_pure_fold_is_best():
    r = score_move([0.97, 0.02, 0.01], 1.5, 1.5, 0, 0, GATE_FOLD, 0)
    assert r["category"] == "best"
    assert r["score"] == pytest.approx(100.0)


def test_the_owners_mixed_flop_fold_is_correct():
    # (owner, 2026-10-03) the network mixed fold 23 / call 40 / raise 37 on the flop; the
    # fold was an "inaccuracy" under the old linear ratio (57.5 < 60). It is part of the mix.
    r = score_move([0.23, 0.40, 0.37], 1.5, 1.5, 0, 0, GATE_FOLD, 0)
    assert r["category"] == "correct"
    assert r["score"] == pytest.approx(100 * (1 - math.log(0.40 / 0.23) / SCORE_SPAN))
    assert score_move([0.23, 0.40, 0.37], 1.5, 1.5, 0, 0, GATE_RAISE, 0)["category"] == "best"


def test_mixed_check_bet_user_bets_well_sized_is_a_near_tie():
    # Network mixes check 55 / bet 45; user bets at the Beta mode.
    alpha, beta = 3.0, 5.0
    mode = (alpha - 1) / (alpha + beta - 2)
    lo, hi = 10_000, 100_000
    chips = round(lo + mode * (hi - lo))
    r = score_move([0.0, 0.55, 0.45], alpha, beta, lo, hi, GATE_RAISE, chips)
    assert r["gate_ratio"] == pytest.approx(0.45 / 0.55, rel=1e-6)
    assert r["size_q"] == pytest.approx(1.0, abs=1e-3)
    assert r["score"] == pytest.approx(100 * (1 - math.log(0.55 / 0.45) / SCORE_SPAN), rel=1e-2)
    assert r["category"] == "best"  # second-best line at 45 vs 55, well sized: a near-tie


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
    # The size's part is capped: the right kind of move is an inaccuracy at worst.
    assert r_tail["score"] == pytest.approx(100 * (1 - SCORING["size_cap"] / SCORE_SPAN))
    assert r_tail["category"] == "inaccuracy"
    # Same gate at the mode scores ~100 / best.
    chips_mode = round(lo + (alpha - 1) / (alpha + beta - 2) * (hi - lo))
    r_mode = score_move([0.05, 0.15, 0.80], alpha, beta, lo, hi, GATE_RAISE, chips_mode)
    assert r_mode["category"] == "best"


def test_recommended_mean_size_scores_full():
    # The deterministic recommendation bets the Beta MEAN; betting exactly
    # that must earn full size credit. Regression: the old mode reference
    # scored the recommended (mean) size below 100% on skewed Betas.
    alpha, beta = 2.0, 8.0
    mean = alpha / (alpha + beta)
    lo, hi = 10_000, 100_000
    chips = round(lo + mean * (hi - lo))
    r = score_move([0.0, 0.3, 0.7], alpha, beta, lo, hi, GATE_RAISE, chips)
    assert r["size_q"] == pytest.approx(1.0, abs=1e-3)
    assert r["category"] == "best"


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

    assert cat(0.90)["category"] == "best"      # a near-tie with the favourite
    assert cat(0.76)["category"] == "best"
    assert cat(0.74)["category"] == "correct"
    assert cat(0.30)["category"] == "correct"   # a regular part of the mix
    assert cat(0.24)["category"] == "inaccuracy"
    assert cat(0.11)["category"] == "inaccuracy"
    assert cat(0.09)["category"] == "wrong"
    assert cat(0.05)["category"] == "wrong"     # (2.5%: rare, not under 2%)
    assert cat(0.03)["category"] == "blunder"   # (1.5%: under 2% outright)
    # the score bands are the ratios on the log scale
    assert SCORING["best_min"] == pytest.approx(100 * (1 - math.log(4 / 3) / SCORE_SPAN))
    assert SCORING["wrong_min"] == pytest.approx(10.0)


def test_a_tie_with_the_favourite_is_best():
    r = score_move([0.45, 0.45, 0.10], 1.5, 1.5, 0, 0, GATE_CHECK_CALL, 0)
    assert r["score"] == pytest.approx(100.0)
    assert r["category"] == "best"


def test_low_probability_overrides_to_blunder():
    r = score_move([0.015, 0.95, 0.035], 1.5, 1.5, 0, 0, GATE_FOLD, 0)
    assert r["category"] == "blunder"


def test_score_clamped():
    r = score_move([0.5, 0.5, 0.0], 1.5, 1.5, 0, 0, GATE_CHECK_CALL, 0)
    assert 0.0 <= r["score"] <= 100.0
    assert math.isfinite(r["score"])
