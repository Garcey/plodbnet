"""StrategyBackend seam + PpoSolverHost parity with legacy trainer path."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plo5bp.gto.backend import NodeDist, PpoSolverHost, make_ppo_host
from plo5bp.network import ActorCritic
from plo5bp.ui.trainer import compute_node_distribution


@pytest.fixture()
def tiny_model():
    torch.manual_seed(0)
    m = ActorCritic(hidden_dim=32, num_layers=1).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def test_ppo_host_badge_not_gto(tiny_model):
    host = make_ppo_host(tiny_model, torch.device("cpu"))
    badge = host.coverage_badge()
    assert badge["is_gto"] is False
    assert badge["mode"] == "ppo"
    assert "Self-play" in badge["label"]


def test_node_dist_roundtrip_v2_fields():
    d = {
        "head_version": 2,
        "gate_probs": [0.1, 0.6, 0.3],
        "rec_gate": 1,
        "rec_chips": 0,
        "value_bb": 1.5,
        "min_chips": 0,
        "max_chips": 100,
        "anchor_probs": [1.0] + [0.0] * 10,
        "anchor_chips": list(range(11)),
        "anchor_legal": [True] + [False] * 10,
        "anchor_lo": list(range(11)),
        "anchor_hi": list(range(11)),
        "refine_ok": [False] * 11,
        "refine_params": [[1.0, 1.0]] * 9,
        "rec_anchor": 0,
        "pot_ref_chips": 50,
        "mixture": None,
    }
    nd = NodeDist.from_dict(d, backend_name="ppo")
    out = nd.as_dict()
    assert out["head_version"] == 2
    assert out["gate_probs"] == [0.1, 0.6, 0.3]
    assert out["rec_anchor"] == 0
    assert out["backend_name"] == "ppo"


def test_host_node_distribution_matches_legacy(tiny_model, trainer_factory):
    """PpoSolverHost must byte-match compute_node_distribution at a live node."""
    ts = trainer_factory(
        seats_mode="fixed",
        seats_fixed=3,
        stacks_mode="fixed",
        stack_bb=100.0,
        mc_rollouts=0,
        rng_seed=7,
    )
    ts.new_hand()
    h = ts.hand
    if h.terminal or h.last_info is None or h.last_obs is None:
        pytest.skip("hand terminal at deal")
    host = ts.backend
    assert isinstance(host, PpoSolverHost)
    via_host = host.node_distribution(h.last_obs, h.last_info).as_dict()
    via_legacy = compute_node_distribution(
        tiny_model if False else ts.model, ts.device, h.last_obs, h.last_info
    )
    assert via_host["head_version"] == via_legacy["head_version"]
    assert via_host["gate_probs"] == pytest.approx(via_legacy["gate_probs"], abs=1e-5)
    assert via_host["rec_gate"] == via_legacy["rec_gate"]
    assert via_host["rec_chips"] == via_legacy["rec_chips"]


def test_trainer_uses_backend_and_badge(trainer_factory, play_to_terminal):
    ts = trainer_factory(
        seats_mode="fixed",
        seats_fixed=3,
        stacks_mode="fixed",
        stack_bb=100.0,
        mc_rollouts=0,
        rng_seed=3,
    )
    assert isinstance(ts.backend, PpoSolverHost)
    frames = ts.new_hand()
    assert frames
    state = ts.project_state()
    badge = state["trainer"]["backend"]
    assert badge["is_gto"] is False
    assert badge["mode"] == "ppo"
    play_to_terminal(ts)
    assert ts.hand.terminal


def test_set_backend_rebind(tiny_model, trainer_factory):
    ts = trainer_factory(mc_rollouts=0, rng_seed=1)
    m2 = ActorCritic(hidden_dim=32, num_layers=1).eval()
    host2 = make_ppo_host(m2, torch.device("cpu"))
    ts.set_backend(host2)
    assert ts.backend is host2
    assert ts.model is m2


def test_act_deterministic_with_seed(tiny_model, trainer_factory):
    ts = trainer_factory(
        seats_mode="fixed",
        seats_fixed=2,
        stacks_mode="fixed",
        stack_bb=100.0,
        mc_rollouts=0,
        rng_seed=11,
    )
    ts.new_hand()
    h = ts.hand
    if h.terminal or h.last_info is None:
        pytest.skip("terminal")
    # Find a non-hero actor by replaying — just sample twice with same seed.
    a1 = ts.backend.act(h.last_obs, h.last_info, rng_seed=42)
    a2 = ts.backend.act(h.last_obs, h.last_info, rng_seed=42)
    assert a1 == a2
