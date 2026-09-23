"""The Rust trajectory flush (engine `flush_trajectories`, 2026-09-23) is
bit-identical to the numpy flush it replaces.

`collect_rollout_batched` flushes every finished hand (retroactive-bonus
qualification + optional bonus, GAE / VRPO backward scans, gathers into the
output slabs) in one Rust pass when observations are stored compact; the numpy
block stays as the fallback (`PLO5BP_NUMPY_FLUSH=1`, dense storage, older
engines). Pinned: every Batch field bitwise equal and the integer bonus
counters equal, for plain GAE, GAE with a retroactive bonus, and VRPO -- the
bonus TOTAL is a log-line diagnostic summed in another order (close, not
bitwise).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plo5bp import rollout as R
from plo5bp.compact_obs import as_dense
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM_MINIMAL
from plo5bp.network import ActorCriticV5, CentralCritic
from plo5bp.selfplay import OpponentPool

pytestmark = pytest.mark.skipif(
    R._rust_flush_trajectories is None, reason="engine without flush_trajectories"
)

_FIELDS = (
    "obs", "gate_masks", "gate_actions", "raise_chips", "sizing", "anchor_actions",
    "refine_u", "opp_holes", "log_probs", "values", "returns", "advantages",
    "old_gate_logp", "old_anchor_logp", "is_terminal",
)


def _collect(monkeypatch, numpy_flush: bool, est: str, retro: float):
    monkeypatch.setenv("PLO5_RUST_ENCODER", "1")
    monkeypatch.setenv("PLO5BP_NUMPY_FLUSH", "1" if numpy_flush else "0")
    R._clear_rollout_buffers()
    torch.manual_seed(0)
    learner = ActorCriticV5(
        hidden_dim=16, obs_dim=OBS_DIM_MINIMAL, num_layers=3, torso_layernorm=True
    )
    torch.manual_seed(1)
    critic = CentralCritic(
        obs_dim=OBS_DIM_MINIMAL, hidden_dim=16, num_blocks=1,
        q_actions=3 if est == "vrpo" else 0,
    )
    cfg = TrainingConfig(
        num_envs=48, rollout_length=1500, obs_mode="minimal",
        advantage_estimator=est, retroactive_bonus_c=retro,
    )
    configs = [
        GameConfig(num_seats=s, starting_stack=st * 10_000, ante=30_000, bb=10_000)
        for s, st in ((2, 40), (6, 20), (4, 120))
    ]
    torch.manual_seed(7)
    return R.collect_rollout_multiconfig(
        learner, OpponentPool(capacity=4, seed=0), configs, cfg,
        np.random.default_rng(5), critic=critic,
    )


@pytest.mark.parametrize("est,retro", [("gae", 0.0), ("gae", 0.37), ("vrpo", 0.21)])
def test_rust_flush_matches_numpy_flush(monkeypatch, est, retro) -> None:
    a = _collect(monkeypatch, True, est, retro)
    b = _collect(monkeypatch, False, est, retro)
    for f in _FIELDS:
        x, y = getattr(a, f), getattr(b, f)
        if f == "obs":
            x, y = as_dense(x), as_dense(y)
        x, y = x.numpy(), y.numpy()
        assert x.shape == y.shape, f
        if x.dtype == np.float32:
            x, y = x.view(np.uint32), y.view(np.uint32)
        assert np.array_equal(x, y), f"{f} differs ({est}, retro {retro})"
    assert a.aggr_bonus_steps == b.aggr_bonus_steps
    assert a.aggr_bonus_steps_by_street == b.aggr_bonus_steps_by_street
    assert a.aggr_steps_total == b.aggr_steps_total
    assert b.aggr_bonus_total_bb == pytest.approx(a.aggr_bonus_total_bb, rel=1e-6, abs=1e-6)
