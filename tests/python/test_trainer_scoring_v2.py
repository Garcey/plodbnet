"""v2 trainer scoring (anchor head) + end-to-end payload schema.

score_move_v2 unit cases drive a hand-built node distribution; the
session test runs a real TrainerSession over an ActorCriticV2 and
asserts the v2 fields flow through DecisionRecord → review payload.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.network import ActorCriticV2
from plo5bp.ui.trainer import SCORING, score_move_v2


def _dist(
    gate_probs=None,
    anchor_probs=None,
    anchor_chips=None,
    anchor_legal=None,
    refine_ok=None,
    refine_params=None,
    anchor_lo=None,
    anchor_hi=None,
):
    if gate_probs is None:
        gate_probs = [0.1, 0.2, 0.7]
    if anchor_chips is None:
        anchor_chips = [100 * k + 100 for k in range(11)]
    if anchor_probs is None:
        anchor_probs = [1.0 / 11.0] * 11
    rec_anchor = int(np.argmax(anchor_probs))
    return {
        "head_version": 2,
        "gate_probs": gate_probs,
        "anchor_probs": anchor_probs,
        "anchor_chips": anchor_chips,
        "anchor_legal": anchor_legal if anchor_legal is not None else [True] * 11,
        "anchor_lo": anchor_lo if anchor_lo is not None else list(anchor_chips),
        "anchor_hi": anchor_hi if anchor_hi is not None else list(anchor_chips),
        "refine_ok": refine_ok if refine_ok is not None else [False] * 11,
        "refine_params": refine_params if refine_params is not None
        else [[1.0, 1.0]] * 9,
        "rec_anchor": rec_anchor,
        "min_chips": anchor_chips[0],
        "max_chips": anchor_chips[10],
        "rec_gate": int(np.argmax(gate_probs)),
        "rec_chips": anchor_chips[rec_anchor],
        "value_bb": 0.0,
    }


def test_exact_rec_pick_scores_100_best():
    probs = [0.02] * 11
    probs[5] = 0.8
    d = _dist(anchor_probs=probs)
    sc = score_move_v2(d, GATE_RAISE, d["anchor_chips"][5])
    assert sc["score"] == pytest.approx(100.0)
    assert sc["category"] == "best"
    assert sc["user_anchor"] == 5


def test_recommended_mean_size_scores_full_v2():
    # Skewed Beta (mean != mode): betting the MEAN — what the deterministic
    # rec bets — must earn full size credit. Regression for the mean/mode
    # scoring mismatch (a matched bet previously scored well under 100%).
    refine_ok = [False] * 11
    refine_ok[5] = True
    alpha, beta = 2.0, 6.0
    refine_params = [[1.0, 1.0]] * 9
    refine_params[4] = [alpha, beta]
    chips = [100 * k + 100 for k in range(11)]
    lo = [chips[k] - 50 for k in range(11)]
    hi = [chips[k] + 50 for k in range(11)]
    probs = [0.0] * 11
    probs[5] = 1.0
    d = _dist(
        anchor_probs=probs, anchor_chips=chips, refine_ok=refine_ok,
        refine_params=refine_params, anchor_lo=lo, anchor_hi=hi,
    )
    mean = alpha / (alpha + beta)
    user_chips = round(lo[5] + mean * (hi[5] - lo[5]))
    sc = score_move_v2(d, GATE_RAISE, user_chips)
    assert sc["user_anchor"] == 5
    assert sc["size_q"] == pytest.approx(1.0, abs=1e-3)
    assert sc["score"] == pytest.approx(100.0, abs=0.5)


def test_nearest_anchor_snapping_tie_goes_lower():
    d = _dist()
    # chips exactly between anchors 3 (400) and 4 (500) → snaps to 3.
    sc = score_move_v2(d, GATE_RAISE, 450)
    assert sc["user_anchor"] == 3
    # just above the midpoint → snaps to 4.
    sc = score_move_v2(d, GATE_RAISE, 451)
    assert sc["user_anchor"] == 4


def test_illegal_anchors_never_snapped_to():
    legal = [True] * 11
    legal[4] = False
    d = _dist(anchor_legal=legal)
    # chips dead-on the illegal anchor 4 must snap to a LEGAL neighbour.
    sc = score_move_v2(d, GATE_RAISE, d["anchor_chips"][4])
    assert sc["user_anchor"] in (3, 5)


def test_anchor_prob_ratio_drives_size_q():
    probs = [0.0] * 11
    probs[5] = 2.0 / 3.0
    probs[7] = 1.0 / 3.0
    d = _dist(anchor_probs=probs)
    sc = score_move_v2(d, GATE_RAISE, d["anchor_chips"][7])
    assert sc["size_q"] == pytest.approx(0.5)
    expected = 100.0 * (SCORING["size_floor"] + (1 - SCORING["size_floor"]) * 0.5)
    assert sc["score"] == pytest.approx(expected)


def test_non_raise_gates_ignore_anchors():
    d = _dist(gate_probs=[0.1, 0.7, 0.2])
    sc = score_move_v2(d, GATE_CHECK_CALL, 0)
    assert sc["score"] == pytest.approx(100.0)
    assert sc["user_anchor"] is None
    assert sc["size_q"] == 1.0


def test_blunder_override_on_tiny_gate_prob():
    d = _dist(gate_probs=[0.01, 0.29, 0.7])
    sc = score_move_v2(d, GATE_FOLD, 0)
    assert sc["category"] == "blunder"


def test_refinement_pdf_ratio_within_bracket():
    refine_ok = [False] * 11
    refine_ok[5] = True
    alpha = beta = 3.0  # mode at bracket center (the anchor itself)
    refine_params = [[1.0, 1.0]] * 9
    refine_params[4] = [alpha, beta]
    lo = [c - 50 for c in range(100, 1201, 100)]
    hi = [c + 50 for c in range(100, 1201, 100)]
    chips = [100 * k + 100 for k in range(11)]
    d = _dist(
        anchor_chips=chips,
        refine_ok=refine_ok,
        refine_params=refine_params,
        anchor_lo=[chips[k] - 50 for k in range(11)],
        anchor_hi=[chips[k] + 50 for k in range(11)],
    )
    probs = [0.0] * 11
    probs[5] = 1.0
    d["anchor_probs"] = probs
    d["rec_anchor"] = 5

    on_anchor = score_move_v2(d, GATE_RAISE, chips[5])
    near_edge = score_move_v2(d, GATE_RAISE, chips[5] - 45)
    assert on_anchor["size_q"] == pytest.approx(1.0, abs=1e-6)
    assert near_edge["size_q"] < on_anchor["size_q"]
    # The edge density of Beta(3,3) vs its mode: u=0.05 → ratio well < 1.
    u = 0.05
    expected = ((u * (1 - u)) / 0.25) ** 2
    assert near_edge["size_q"] == pytest.approx(expected, rel=1e-3)


def test_candidates_equal_v2(trainer_factory):
    from plo5bp.ui.trainer import DecisionRecord

    ts = trainer_factory(model_cls=ActorCriticV2)

    def rec(user_anchor, rec_anchor, user_chips, rec_chips):
        return DecisionRecord(
            decision_idx=0, street=1, action_log_idx=0,
            gate_probs=[0.1, 0.2, 0.7], alpha=1.0, beta=1.0,
            min_chips=100, max_chips=1100,
            rec_gate=GATE_RAISE, rec_chips=rec_chips, value_bb=0.0,
            pot_chips=1000, to_call_chips=0,
            user_gate=GATE_RAISE, user_chips=user_chips,
            gate_ratio=1.0, size_q=1.0, score=100.0, category="best",
            head_version=2, rec_anchor=rec_anchor, user_anchor=user_anchor,
        )

    assert ts._candidates_equal(rec(5, 5, 600, 600))
    assert not ts._candidates_equal(rec(4, 5, 500, 600))  # different anchor
    assert not ts._candidates_equal(rec(5, 5, 600, 700))  # > tol apart


def test_v2_session_payload_schema(trainer_factory):
    ts = trainer_factory(model_cls=ActorCriticV2, mc_rollouts=2)
    ts.new_hand()
    raised = False
    steps = 0
    while ts.hand is not None and not ts.hand.terminal and steps < 80:
        s = ts.project_state()
        legal = s["legal"]
        if legal["raise"] and not raised:
            lo = s["raise_bounds"]["min_chips"]
            hi = s["raise_bounds"]["max_chips"]
            ts.act("raise", min(hi, max(lo, (lo + hi) // 2)))
            raised = True
        elif legal["check_call"]:
            ts.act("check_call", None)
        elif legal["fold"]:
            ts.act("fold", None)
        else:
            ts.act("raise", s["raise_bounds"]["min_chips"])
        steps += 1
    assert ts.hand is not None and ts.hand.terminal

    assert ts.hand.decisions, "no hero decisions recorded"
    for d in ts.hand.decisions:
        assert d.head_version == 2
        assert d.anchor_probs is not None and len(d.anchor_probs) == 11
        assert d.anchor_chips is not None and len(d.anchor_chips) == 11
        assert d.rec_anchor is not None and 0 <= d.rec_anchor <= 10
        if d.user_gate == GATE_RAISE:
            assert d.user_anchor is not None
            assert d.anchor_legal[d.user_anchor]
        assert math.isfinite(d.score)

    review = ts.review_block(0)
    cur = review["current"]
    assert cur["head_version"] == 2
    assert isinstance(cur["anchors"], list) and cur["anchors"]
    for row in cur["anchors"]:
        assert set(row) == {"k", "label", "prob", "chips", "chips_bb"}
    assert cur["rec_anchor"] is not None
    assert cur["beta_alpha"] >= 1.0 and cur["beta_beta"] >= 1.0
