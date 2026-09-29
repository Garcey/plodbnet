"""The batched collector's per-(env, seat) trajectory arrays start at
`rollout._TRAJ_CAP_INIT` slots and double on demand up to MAX_STEPS_PER_SEAT
(2026-09-23; they used to be a fixed 192 slots — ~735 MB zero-filled per vMin1
sub-rollout). Growth must be invisible: a run forced to grow over and over
(initial capacity 1) gives exactly the batch of a run that never grows."""

from __future__ import annotations

import numpy as np
import torch

from plo5bp import rollout as rollout_mod
from plo5bp.compact_obs import as_dense
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM_MINIMAL
from plo5bp.network import ActorCriticV2, CentralCritic
from plo5bp.rollout import _BATCH_TENSOR_FIELDS, collect_rollout_multiconfig
from plo5bp.selfplay import OpponentPool

BB = 10_000


def _collect(monkeypatch, init_cap: int):
    monkeypatch.setattr(rollout_mod, "_TRAJ_CAP_INIT", init_cap)
    # The arrays are reused across collections (_TRAJ_BUFFERS): start each run
    # fresh so this one really begins at `init_cap`.
    rollout_mod._clear_rollout_buffers()
    torch.manual_seed(1)
    model = ActorCriticV2(hidden_dim=32, obs_dim=OBS_DIM_MINIMAL)
    critic = CentralCritic(obs_dim=OBS_DIM_MINIMAL, hidden_dim=32, num_blocks=1)
    tc = TrainingConfig(num_envs=8, rollout_length=320, hidden_dim=32, obs_mode="minimal")
    cfgs = [  # deep heads-up hands run long per seat; 6-max spreads across seats
        GameConfig(num_seats=2, starting_stack=250 * BB, ante=3 * BB, bb=BB),
        GameConfig(num_seats=6, starting_stack=80 * BB, ante=3 * BB, bb=BB),
    ]
    torch.manual_seed(2)
    return collect_rollout_multiconfig(
        model, OpponentPool(capacity=1), cfgs, tc, np.random.default_rng(3), critic=critic
    )


def test_growing_trajectory_capacity_is_exact(monkeypatch) -> None:
    grown = _collect(monkeypatch, 1)  # must grow repeatedly (1 -> 2 -> 4 -> ...)
    fixed = _collect(monkeypatch, 192)  # never grows
    for f in _BATCH_TENSOR_FIELDS:
        assert torch.equal(as_dense(getattr(grown, f)), as_dense(getattr(fixed, f))), f
    assert torch.equal(grown.is_terminal, fixed.is_terminal)
    assert grown.aggr_steps_total_by_street == fixed.aggr_steps_total_by_street
