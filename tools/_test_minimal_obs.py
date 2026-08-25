import numpy as np
from plo5bp.encoding import OBS_DIM, OBS_DIM_MINIMAL, project_obs_minimal
from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv
from plo5bp.env_batched import BatchedBombPotEnv
import torch
from plo5bp.network import ActorCriticV5, CentralCritic
from plo5bp.sizing import PLO_ANCHOR_SPEC

cfg = GameConfig()
env_full = BombPotEnv(cfg, obs_mode="full")
env_min = BombPotEnv(cfg, obs_mode="minimal")
obs_f, _ = env_full.reset(seed=0, button=0)
obs_m, _ = env_min.reset(seed=0, button=0)
assert obs_m.shape == (OBS_DIM_MINIMAL,)
np.testing.assert_array_equal(project_obs_minimal(obs_f), obs_m)
print("serial ok")

be_f = BatchedBombPotEnv(8, cfg, obs_mode="full", opp_outcome_mc=1)
be_m = BatchedBombPotEnv(8, cfg, obs_mode="minimal", opp_outcome_mc=1)
seeds = np.arange(8, dtype=np.uint64)
buttons = np.zeros(8, dtype=np.uint8)
st_f = be_f.reset_batch(seeds, buttons)
st_m = be_m.reset_batch(seeds, buttons)
assert st_m.obs.shape == (8, OBS_DIM_MINIMAL), st_m.obs.shape
np.testing.assert_array_equal(project_obs_minimal(st_f.obs), st_m.obs)
print("batched ok", st_m.obs.shape)

m = ActorCriticV5(hidden_dim=512, num_layers=3, obs_dim=OBS_DIM_MINIMAL, anchor_spec=PLO_ANCHOR_SPEC, torso_layernorm=True, mixture_k=3)
c = CentralCritic(obs_dim=OBS_DIM_MINIMAL, hidden_dim=256, num_blocks=2, q_actions=3, torso_layernorm=True, value_bins=51)
x = torch.as_tensor(st_m.obs[:2])
gm = torch.as_tensor(st_m.gate_mask[:2])
sizing = torch.stack([
    torch.as_tensor(st_m.min_raise[:2].astype(np.float32)),
    torch.as_tensor(st_m.max_raise[:2].astype(np.float32)),
    torch.full((2,), 100.0),
    torch.zeros(2),
], -1)
out = m.act(x, gm, sizing)
print("actor", out.gate.shape)
v = c(x, torch.zeros(2, 260))
print("critic", float(v[0]))
print("ALL SMOKE PASSED")
