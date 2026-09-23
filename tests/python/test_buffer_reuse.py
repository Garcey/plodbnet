"""Rollout buffers are REUSED across sub-rollouts and updates (2026-09-23): the
batched collector's per-(env, seat) trajectory arrays and per-step observation
pool, and multiconfig's shared staging buffer (CUDA learners only — forced on
here to exercise it on CPU). Fresh multi-hundred-MB allocations 30x per update
stalled a fragmented RunPod host; reuse must be invisible: a collection that
starts on buffers left dirty by an earlier, DIFFERENT collection gives exactly
the batch it gives on fresh buffers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plo5bp import rollout as rollout_mod
from plo5bp.compact_obs import as_dense
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM_MINIMAL
from plo5bp.network import ActorCriticV2, CentralCritic
from plo5bp.rollout import _BATCH_TENSOR_FIELDS, collect_rollout_multiconfig
from plo5bp.selfplay import OpponentPool

BB = 10_000


def _collect(seed: int, use_vrpo: bool):
    torch.manual_seed(1)
    model = ActorCriticV2(hidden_dim=32, obs_dim=OBS_DIM_MINIMAL)
    critic = CentralCritic(
        obs_dim=OBS_DIM_MINIMAL, hidden_dim=32, num_blocks=1,
        q_actions=3 if use_vrpo else 0,
    )
    tc = TrainingConfig(
        num_envs=12, rollout_length=360, hidden_dim=32, obs_mode="minimal",
        advantage_estimator="vrpo" if use_vrpo else "gae",
    )
    cfgs = [  # a repeated seat count -> reuse inside one call as well
        GameConfig(num_seats=2, starting_stack=250 * BB, ante=3 * BB, bb=BB),
        GameConfig(num_seats=5, starting_stack=60 * BB, ante=3 * BB, bb=BB),
        GameConfig(num_seats=2, starting_stack=40 * BB, ante=3 * BB, bb=BB),
    ]
    torch.manual_seed(seed)
    return collect_rollout_multiconfig(
        model, OpponentPool(capacity=1), cfgs, tc, np.random.default_rng(seed),
        critic=critic,
    )


def _snapshot(batch) -> dict[str, torch.Tensor]:
    """Owned copies — with staging reuse the batch tensors are views of the
    staging buffer, which the next collection overwrites."""
    out = {f: as_dense(getattr(batch, f)).clone() for f in _BATCH_TENSOR_FIELDS}
    out["is_terminal"] = batch.is_terminal.clone()
    return out


def _assert_same(a: dict, b) -> None:
    for f in _BATCH_TENSOR_FIELDS:
        tb = as_dense(getattr(b, f))
        assert a[f].dtype == tb.dtype and a[f].shape == tb.shape, f
        assert torch.equal(a[f], tb), f
    assert torch.equal(a["is_terminal"], b.is_terminal)


@pytest.mark.parametrize("use_vrpo", [False, True])
def test_dirty_reused_buffers_give_exactly_the_fresh_batch(monkeypatch, use_vrpo) -> None:
    monkeypatch.setattr(rollout_mod, "_reuse_staging", lambda device: True)
    rollout_mod._clear_rollout_buffers()
    fresh = _snapshot(_collect(7, use_vrpo))  # fresh buffers
    traj_bufs = dict(rollout_mod._TRAJ_BUFFERS)
    assert traj_bufs and rollout_mod._OBS_POOL_BUFFERS and rollout_mod._STAGING_BUFFERS
    _collect(99, use_vrpo)  # a different collection dirties every buffer
    again = _collect(7, use_vrpo)  # the first collection again, on dirty buffers
    _assert_same(fresh, again)
    for key, buf in traj_bufs.items():  # the SAME buffer objects were reused
        assert rollout_mod._TRAJ_BUFFERS[key] is buf


def test_cpu_learner_never_reuses_staging() -> None:
    """On a CPU learner the returned batch IS a view of the staging buffer, so
    reusing it would overwrite a batch the caller may still hold."""
    assert rollout_mod._reuse_staging(torch.device("cpu")) is False
    assert rollout_mod._reuse_staging(torch.device("cuda")) is True
    rollout_mod._clear_rollout_buffers()
    first = _collect(3, False)
    kept = _snapshot(first)
    _collect(4, False)  # must not touch `first`
    _assert_same(kept, first)
    assert not rollout_mod._STAGING_BUFFERS
