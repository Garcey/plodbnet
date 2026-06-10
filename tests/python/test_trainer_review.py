"""Review replay fidelity and what-if reconstruction parity."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from plo5bp.ui.trainer import WhatifRequest


def _node_signature(state: dict) -> dict:
    return {
        "pot": state["pot_chips"],
        "to_call": state["to_call_chips"],
        "legal": state["legal"],
        "bounds": state["raise_bounds"],
        "street": state["street"],
        "history_len": len(state["history"]),
        "stacks": [x["stack_chips"] for x in state["seats"]],
        "folded": [x["folded"] for x in state["seats"]],
    }


def _play_capturing(ts, max_steps=80):
    """Play to terminal, capturing the live projection at each decision."""
    sigs = []
    steps = 0
    while not ts.hand.terminal and steps < max_steps:
        s = ts.project_state()
        sigs.append(_node_signature(s))
        legal = s["legal"]
        if legal["check_call"]:
            ts.act("check_call", None)
        elif legal["fold"]:
            ts.act("fold", None)
        else:
            ts.act("raise", s["raise_bounds"]["min_chips"])
        steps += 1
    assert ts.hand.terminal
    return sigs


def test_review_reconstructs_each_decision(trainer_factory):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=4, mc_rollouts=0)
    ts.new_hand()
    live_sigs = _play_capturing(ts)
    assert len(live_sigs) == len(ts.hand.decisions)
    for i, d in enumerate(ts.hand.decisions):
        env, obs, info = ts._replay_to_decision(d)
        review = ts.review_block(i)
        state = ts.project_state(env=env, info=info, reveal=True, review=review)
        assert _node_signature(state) == live_sigs[i]
        assert state["actor"] == ts.hand.hero_seat
        assert all(x["hole"] is not None for x in state["seats"])
        assert state["trainer"]["review"]["decision"] == i


def test_whatif_noop_matches_original_distribution(trainer_factory, play_to_terminal):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=3, mc_rollouts=0)
    ts.new_hand()
    play_to_terminal(ts)
    if not ts.hand.decisions:
        pytest.skip("hand ended before hero acted")
    for i, d in enumerate(ts.hand.decisions):
        state = ts.whatif(WhatifRequest(decision=i))
        rec = state["trainer"]["review"]["whatif"]["recommendation"]
        assert rec["gate_distribution"] == [round(p, 4) for p in d.gate_probs]
        rescored = state["trainer"]["review"]["whatif"]["rescored"]
        assert rescored["score"] == round(d.score, 1)
        assert rescored["category"] == d.category


def test_whatif_swap_changes_spec_and_returns_recommendation(
    trainer_factory, play_to_terminal
):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=3, mc_rollouts=0)
    ts.new_hand()
    play_to_terminal(ts)
    if not ts.hand.decisions:
        pytest.skip("hand ended before hero acted")
    d = ts.hand.decisions[0]
    spec = ts._original_card_spec(d)
    used = {c for v in spec.values() for c in v if c is not None}
    used |= {c for h in ts.hand.all_holes for c in h}
    replacement = next(c for c in range(52) if c not in used)
    new_flop_a = [replacement, None, None]
    state = ts.whatif(WhatifRequest(decision=0, flop_a=new_flop_a))
    wf = state["trainer"]["review"]["whatif"]
    assert wf["card_spec"]["flop_a"][0] == replacement
    assert state["card_spec"]["flop_a"][0] == replacement
    dist = wf["recommendation"]["gate_distribution"]
    assert len(dist) == 3 and abs(sum(dist) - 1.0) < 1e-2


def test_whatif_rejects_duplicates_and_unrevealed(trainer_factory, play_to_terminal):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=3, mc_rollouts=0)
    ts.new_hand()
    play_to_terminal(ts)
    if not ts.hand.decisions:
        pytest.skip("hand ended before hero acted")
    d = ts.hand.decisions[0]
    spec = ts._original_card_spec(d)

    # Duplicate: set flop_a[0] to hero's first hole card.
    dup = spec["hero_hole"][0]
    with pytest.raises(HTTPException) as e:
        ts.whatif(WhatifRequest(decision=0, flop_a=[dup, None, None]))
    assert e.value.status_code == 400

    # Unrevealed: overriding the river at a flop decision.
    if d.street < 3:
        free = next(
            c for c in range(52)
            if c not in {x for v in spec.values() for x in v if x is not None}
        )
        with pytest.raises(HTTPException) as e2:
            ts.whatif(WhatifRequest(decision=0, river=[free, None]))
        assert e2.value.status_code == 400


def test_review_requires_terminal(trainer_factory):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=4, mc_rollouts=0)
    ts.new_hand()
    if ts.hand.terminal:
        pytest.skip("hand ended instantly")
    with pytest.raises(HTTPException):
        ts._decision_for(0)


def test_repeat_redeal_is_identical(trainer_factory, play_to_terminal):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=4, mc_rollouts=0)
    ts.new_hand()
    holes = [list(h) for h in ts.hand.all_holes]
    seed, button = ts.hand.seed, ts.hand.button
    play_to_terminal(ts)
    ts.new_hand(repeat=True)
    assert ts.hand.is_repeat
    assert ts.hand.seed == seed and ts.hand.button == button
    assert ts.hand.all_holes == holes
