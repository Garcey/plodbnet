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


def test_fold_ev_is_forward_facing_zero(trainer_factory):
    """A fold is worth 0 EV measured forward from the decision point —
    already-committed chips are sunk. The EV components are rebased so a
    deviating fold reads as 0.00, not -(committed). The tiny random net
    never bets, so drive the engine directly to create a bet facing hero,
    then run the real MC estimate."""
    from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
    from plo5bp.env import BombPotEnv
    from plo5bp.ui.trainer import DecisionRecord, HandRecord

    ts = trainer_factory(seats_mode="fixed", seats_fixed=3, mc_rollouts=8)
    ts.new_hand()
    base = ts.hand

    # Step the first actor into a raise so the next seat faces a bet.
    env = BombPotEnv(base.config)
    obs, info = env.reset(base.seed, base.button)
    street = int(info.raw_obs["street"])
    actor0 = int(info.actor)
    raise_chips = int(info.min_raise_chips) or int(info.max_raise_chips)
    obs, _, done, info = env.step_hybrid(GATE_RAISE, raise_chips)
    assert not done and info.actor is not None
    hero = int(info.actor)
    assert bool(info.gate_mask[GATE_FOLD])  # hero now faces a bet
    committed = int(info.raw_obs["total_commit"][hero])
    assert committed > 0

    # Install a hand whose hero is the seat facing the bet.
    ts.hand = HandRecord(
        hand_no=base.hand_no, seed=base.seed, button=base.button,
        hero_seat=hero, config=base.config, env=env,
        last_obs=obs, last_info=info,
        action_log=[{"seat": actor0, "gate": GATE_RAISE,
                     "chips": raise_chips, "street": street}],
        all_holes=base.all_holes, all_holes_dealt=base.all_holes_dealt,
    )
    d = DecisionRecord(
        decision_idx=0, street=street, action_log_idx=1,
        gate_probs=[0.4, 0.3, 0.3], alpha=1.0, beta=1.0,
        min_chips=int(info.min_raise_chips), max_chips=int(info.max_raise_chips),
        rec_gate=GATE_CHECK_CALL, rec_chips=0, value_bb=0.0,
        pot_chips=int(info.raw_obs["pot"]), to_call_chips=committed,
        hero_committed_chips=committed,
        user_gate=GATE_FOLD, user_chips=0,
        gate_ratio=0.4, size_q=1.0, score=40.0, category="wrong",
    )
    ts._estimate_ev_loss(d)
    assert d.ev_user_bb == 0.0  # forward fold EV is exactly 0
    assert d.ev_best_bb is not None
    assert d.ev_loss_bb is not None and d.ev_loss_bb >= 0.0
