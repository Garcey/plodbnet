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


# --- All-decisions node stepper (hero + villain) --------------------------


def _review(ts, node):
    """The review block projected at a given action_log node index."""
    return ts.review_at_node(node)["trainer"]["review"]


def test_node_index_matches_action_log(trainer_factory, play_to_terminal):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=4, mc_rollouts=0)
    ts.new_hand()
    play_to_terminal(ts)
    rv = _review(ts, 0)
    log = ts.hand.action_log
    assert rv["num_nodes"] == len(log)
    assert len(rv["nodes"]) == len(log)
    for i, (row, a) in enumerate(zip(rv["nodes"], log)):
        assert row["node_idx"] == i
        assert row["seat"] == a["seat"]
        assert row["is_hero"] == (a["seat"] == ts.hand.hero_seat)


def test_pills_map_to_hero_nodes(trainer_factory, play_to_terminal):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=4, mc_rollouts=0)
    ts.new_hand()
    play_to_terminal(ts)
    if not ts.hand.decisions:
        pytest.skip("hero never acted")
    rv = _review(ts, 0)
    nodes = rv["nodes"]
    assert rv["decisions"]  # one pill per hero decision
    for pill in rv["decisions"]:
        ni = pill["node_idx"]
        assert nodes[ni]["is_hero"] is True
        assert nodes[ni]["decision_idx"] == pill["decision_idx"]


def test_villain_node_view(trainer_factory, play_to_terminal):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=4, mc_rollouts=0)
    ts.new_hand()
    play_to_terminal(ts)
    villain = next(
        (n for n in _review(ts, 0)["nodes"] if not n["is_hero"]), None
    )
    if villain is None:
        pytest.skip("no villain decision in this hand")
    nc = _review(ts, villain["node_idx"])["node_current"]
    assert nc["is_hero"] is False
    assert nc["node_idx"] == villain["node_idx"]
    assert len(nc["gate_probs"]) == 3
    assert abs(sum(nc["gate_probs"]) - 1.0) < 1e-2
    assert isinstance(nc["value_bb"], float)
    assert nc["actual_label"]
    assert "score" not in nc  # no graded user action at a villain node


def test_hero_node_overlays_stored_record(trainer_factory, play_to_terminal):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=4, mc_rollouts=4)
    ts.new_hand()
    play_to_terminal(ts)
    if not ts.hand.decisions:
        pytest.skip("hero never acted")
    d = ts.hand.decisions[0]
    nc = _review(ts, d.action_log_idx)["node_current"]
    assert nc["is_hero"] is True
    assert nc["decision_idx"] == d.decision_idx
    assert nc["score"] == round(d.score, 1)
    assert nc["category"] == d.category
    assert nc["ev_loss_bb"] == d.ev_loss_bb  # MC value a fresh forward can't reproduce


def test_review_at_node_clamps(trainer_factory, play_to_terminal):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=4, mc_rollouts=0)
    ts.new_hand()
    play_to_terminal(ts)
    n = len(ts.hand.action_log)
    assert _review(ts, -5)["node"] == 0
    assert _review(ts, 10**9)["node"] == n - 1


def test_early_node_hides_turn_river(trainer_factory, play_to_terminal):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=4, mc_rollouts=0)
    ts.new_hand()
    play_to_terminal(ts)
    spec = ts.review_at_node(0)["card_spec"]
    assert spec["turn"] == [None, None]
    assert spec["river"] == [None, None]


def test_review_block_backcompat(trainer_factory, play_to_terminal):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=4, mc_rollouts=0)
    ts.new_hand()
    play_to_terminal(ts)
    if not ts.hand.decisions:
        pytest.skip("hero never acted")
    rb = ts.review_block(0)
    assert rb["current"] is not None
    assert rb["current"]["decision_idx"] == 0
    assert rb["node"] == ts.hand.decisions[0].action_log_idx
    assert rb["node_current"] is None


def test_node_view_true_ev_matches_critic(trainer_factory, play_to_terminal):
    """The review's true-EV must equal an independent critic call built
    from the canonical rotation helper — pins the opp-multihot convention
    (rotation / dealt order / dtype) to the training path."""
    import numpy as np
    import torch

    from plo5bp.network import ActorCriticV2, CentralCritic
    from plo5bp.rollout import _critic_values, _rotate_opp_holes

    ts = trainer_factory(
        model_cls=ActorCriticV2, seats_mode="fixed", seats_fixed=4, mc_rollouts=0
    )
    torch.manual_seed(1)
    critic = CentralCritic(hidden_dim=32, num_blocks=1).eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    ts.critic = critic
    ts.new_hand()
    play_to_terminal(ts)
    node_idx = len(ts.hand.action_log) // 2
    env, obs, info = ts._replay_to_node(node_idx)
    nc = ts._node_view(node_idx, obs, info)
    assert nc["value_true_bb"] is not None
    actor = int(info.actor)
    holes = np.asarray(ts.hand.all_holes_dealt, dtype=np.uint8)
    opp = _rotate_opp_holes(holes, actor)[None]
    expected = round(float(_critic_values(
        critic, ts.device, obs[None].astype(np.float32), opp,
    )[0]), 4)
    assert nc["value_true_bb"] == expected


def test_review_endpoint_node_param(tmp_path, monkeypatch):
    import torch
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from plo5bp.network import ActorCritic
    from plo5bp.ui.trainer import TrainerSettings, create_trainer_router

    monkeypatch.setenv("PLO5BP_TRAINER_STATS", str(tmp_path / "stats.json"))
    torch.manual_seed(0)
    model = ActorCritic(hidden_dim=32, num_layers=1).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    router = create_trainer_router(model, torch.device("cpu"))
    ts = router.trainer_session
    ts.settings = TrainerSettings(**{
        **ts.settings.model_dump(),
        "seats_mode": "fixed", "seats_fixed": 4, "mc_rollouts": 0,
    })
    ts.new_hand()
    steps = 0
    while ts.hand is not None and not ts.hand.terminal and steps < 80:
        s = ts.project_state()
        legal = s["legal"]
        if legal["check_call"]:
            ts.act("check_call", None)
        elif legal["fold"]:
            ts.act("fold", None)
        else:
            ts.act("raise", s["raise_bounds"]["min_chips"])
        steps += 1
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    r = client.get("/trainer/review", params={"node": 0})
    assert r.status_code == 200
    rv = r.json()["state"]["trainer"]["review"]
    assert rv["node_current"]["node_idx"] == 0
    assert rv["num_nodes"] == len(ts.hand.action_log)
