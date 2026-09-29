"""Half-precision rollout storage (TrainingConfig.obs_real_f16, 2026-09-26).

The compact rows' real columns go to the rollout's output slabs as IEEE
float16 (the per-step pool and the act-time uploads stay float32). Pinned:

- the same collection stored float32 and float16 (every RNG re-seeded) gives
  identical flag bits, float16 reals that are EXACTLY numpy's
  round-to-nearest-even cast of the float32 ones (the Rust flush's f32->f16
  conversion == numpy's), identical values for every other batch field, and
  the same RNG streams (storage draws nothing);
- unpacking a float16 batch yields float32 rows equal to the cast values;
- a PPO update runs on it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plo5bp.compact_obs import FULL_LAYOUT_F16, MINIMAL_LAYOUT_F16, PackedObs, as_dense
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM, OBS_DIM_MINIMAL
from plo5bp.network import ActorCriticV2, CentralCritic
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import _BATCH_TENSOR_FIELDS, collect_rollout_batched, collect_rollout_multiconfig
from plo5bp.selfplay import OpponentPool

BB = 10_000
_DIMS = {"minimal": OBS_DIM_MINIMAL, "full": OBS_DIM}
_F16 = {"minimal": MINIMAL_LAYOUT_F16, "full": FULL_LAYOUT_F16}


def _collect(obs_mode: str, multi: bool, f16: bool):
    dim = _DIMS[obs_mode]
    torch.manual_seed(11)
    model = ActorCriticV2(hidden_dim=32, obs_dim=dim)
    critic = CentralCritic(obs_dim=dim, hidden_dim=32, num_blocks=1)
    tc = TrainingConfig(num_envs=8, rollout_length=160, hidden_dim=32, obs_mode=obs_mode,
                        obs_real_f16=f16)
    rng = np.random.default_rng(5)
    torch.manual_seed(12)
    if multi:
        cfgs = [
            GameConfig(num_seats=3, starting_stack=40 * BB, ante=3 * BB, bb=BB),
            GameConfig(num_seats=5, starting_stack=120 * BB, ante=3 * BB, bb=BB),
        ]
        batch = collect_rollout_multiconfig(model, OpponentPool(capacity=1), cfgs, tc, rng, critic=critic)
    else:
        cfg = GameConfig(num_seats=4, starting_stack=60 * BB, ante=3 * BB, bb=BB)
        batch = collect_rollout_batched(model, OpponentPool(capacity=1), cfg, tc, rng, critic=critic)
    return batch, rng.bit_generator.state, torch.get_rng_state(), (model, critic, tc)


@pytest.mark.parametrize("obs_mode", ["minimal", "full"])
@pytest.mark.parametrize("multi", [False, True])
def test_f16_storage_is_the_rounded_f32_storage(obs_mode: str, multi: bool) -> None:
    b32, n32, t32, _ = _collect(obs_mode, multi, f16=False)
    b16, n16, t16, _ = _collect(obs_mode, multi, f16=True)
    assert isinstance(b16.obs, PackedObs) and b16.obs.layout is _F16[obs_mode]
    assert b16.obs.real.dtype == torch.float16 and b32.obs.real.dtype == torch.float32
    assert torch.equal(b16.obs.bits, b32.obs.bits)
    want = b32.obs.real.numpy().astype(np.float16)
    assert np.array_equal(b16.obs.real.numpy().view(np.uint16), want.view(np.uint16))
    for f in _BATCH_TENSOR_FIELDS[1:]:
        assert torch.equal(getattr(b16, f), getattr(b32, f)), f
    assert n32 == n16 and torch.equal(t32, t16)
    dense16 = as_dense(b16.obs)
    dense32 = as_dense(b32.obs)
    cols = _F16[obs_mode].real_cols
    assert torch.equal(dense16[:, cols], torch.from_numpy(want.astype(np.float32)))
    assert torch.equal(dense16[:, _F16[obs_mode].flag_cols], dense32[:, _F16[obs_mode].flag_cols])
    assert b16.obs.nbytes == len(b16.obs) * _F16[obs_mode].row_bytes


def test_ppo_update_runs_on_f16_storage() -> None:
    b16, _n, _t, (model, critic, tc) = _collect("full", True, f16=True)
    trainer = PPOTrainer(model, tc, critic=critic)
    stats = trainer.update(b16, np.random.default_rng(3))
    assert np.isfinite(stats.policy_loss) and np.isfinite(stats.value_loss)
