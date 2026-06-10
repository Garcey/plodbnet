"""Monte-Carlo EV-loss estimation."""

from __future__ import annotations

import pytest

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD
from plo5bp.ui.trainer import compute_node_distribution


def _fresh_decision_node(ts, max_tries=10):
    """Deal until hero faces a live decision; return its distribution."""
    for _ in range(max_tries):
        ts.new_hand()
        if not ts.hand.terminal:
            dist = compute_node_distribution(
                ts.model, ts.device, ts.hand.last_obs, ts.hand.last_info
            )
            return dist
    pytest.skip("never reached a live hero node")


def test_matching_action_costs_nothing(trainer_factory):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=3, mc_rollouts=4)
    dist = _fresh_decision_node(ts)
    gate = dist["rec_gate"]
    slug = {0: "fold", 1: "check_call", 2: "raise"}[gate]
    chips = dist["rec_chips"] if gate == 2 else None
    ts.act(slug, chips)
    d = ts.hand.decisions[0]
    assert d.ev_loss_bb == 0.0
    assert d.ev_user_bb is None  # MC skipped entirely


def test_deviating_action_gets_estimate(trainer_factory):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=3, mc_rollouts=4)
    dist = _fresh_decision_node(ts)
    # Pick a legal gate that differs from the recommendation.
    info = ts.hand.last_info
    other = None
    for g in (GATE_FOLD, GATE_CHECK_CALL):
        if g != dist["rec_gate"] and bool(info.gate_mask[g]):
            other = g
            break
    if other is None:
        pytest.skip("no alternate legal gate at this node")
    ts.act({0: "fold", 1: "check_call"}[other], None)
    d = ts.hand.decisions[0]
    assert d.ev_user_bb is not None and d.ev_best_bb is not None
    assert d.ev_loss_bb is not None and d.ev_loss_bb >= 0.0


def test_rollout_crn_determinism(trainer_factory):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=3, mc_rollouts=4)
    _fresh_decision_node(ts)
    h = ts.hand
    prefix = list(h.action_log)
    a = ts._rollout_ev(h, prefix, GATE_CHECK_CALL, 0, 4, 777)
    b = ts._rollout_ev(h, prefix, GATE_CHECK_CALL, 0, 4, 777)
    assert a == b
    c = ts._rollout_ev(h, prefix, GATE_CHECK_CALL, 0, 4, 778)
    # Different seed *may* coincide but should usually differ; only assert
    # it is finite to avoid flakiness.
    assert c == c  # not NaN


def test_zero_rollouts_disables_ev(trainer_factory):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=3, mc_rollouts=0)
    dist = _fresh_decision_node(ts)
    info = ts.hand.last_info
    other = next(
        (g for g in (GATE_FOLD, GATE_CHECK_CALL)
         if g != dist["rec_gate"] and bool(info.gate_mask[g])),
        None,
    )
    if other is None:
        pytest.skip("no alternate legal gate at this node")
    ts.act({0: "fold", 1: "check_call"}[other], None)
    d = ts.hand.decisions[0]
    assert d.ev_loss_bb is None and d.ev_user_bb is None
