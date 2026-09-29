"""The collector's Rust kernels (engine `flush_trajectories`, 2026-09-23;
`record_learner_steps`, `aggression_record_batch`) are bit-identical to the
numpy code they replaced.

`collect_rollout_batched` flushes every finished hand (winning-aggression
qualification, GAE / VRPO backward scans, gathers into the
output slabs) in one Rust pass; the numpy originals live in
plo5bp/rollout_reference.py (same signatures) and `PLO5BP_NUMPY_FLUSH=1` makes
the collector call them instead. Pinned: every Batch field bitwise equal and
the integer winning-aggression counters equal, for GAE and VRPO, with compact
AND dense observation storage (dense rows go through the same Rust flush as
raw bytes, 2026-09-28). (The retroactive bonus the kernels still take is
retired -- ML-030 -- and always 0 from the collector; the engine's own tests
cover its arithmetic.)
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


def _collect(monkeypatch, numpy_flush: bool, est: str, compact: bool = True):
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
        advantage_estimator=est, compact_obs=compact,
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


@pytest.mark.parametrize("compact", [True, False], ids=["compact", "dense"])
@pytest.mark.parametrize("est", ["gae", "vrpo"])
def test_rust_flush_matches_numpy_flush(monkeypatch, est, compact) -> None:
    a = _collect(monkeypatch, True, est, compact)
    b = _collect(monkeypatch, False, est, compact)
    assert isinstance(b.obs, torch.Tensor) != compact  # dense storage = a plain tensor
    for f in _FIELDS:
        x, y = getattr(a, f), getattr(b, f)
        if f == "obs":
            x, y = as_dense(x), as_dense(y)
        x, y = x.numpy(), y.numpy()
        assert x.shape == y.shape, f
        if x.dtype == np.float32:
            x, y = x.view(np.uint32), y.view(np.uint32)
        assert np.array_equal(x, y), f"{f} differs ({est}, compact {compact})"
    assert a.aggr_bonus_steps == b.aggr_bonus_steps
    assert a.aggr_bonus_steps_by_street == b.aggr_bonus_steps_by_street
    assert a.aggr_steps_total == b.aggr_steps_total
    assert a.aggr_bonus_total_bb == b.aggr_bonus_total_bb == 0.0  # the bonus is retired
