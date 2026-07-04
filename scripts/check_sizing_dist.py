"""Inspect vFour_5's actual (mu, s) and the resulting anchor distribution at
normal raise nodes, to show the U-shape is the large-s (wide logistic) regime
of an early/high-entropy model, not a bug.
"""
import random
import numpy as np
import torch

from plo5bp.network import build_actor_from_state_dict
from plo5bp.sizing import sizing_from_info, anchor_grid_torch, ANCHOR_COUNT
from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv
from plo5bp.actions import GATE_FOLD, GATE_CHECK_CALL, GATE_RAISE

ckpt = torch.load("checkpoints/stub.pt", map_location="cpu", weights_only=False)
sd = ckpt["model"]
cfg_b = ckpt.get("config", {}) or {}
model = build_actor_from_state_dict(
    sd, int(cfg_b.get("hidden_dim", 2048)), int(cfg_b.get("num_layers", 4))
).eval()
c = (ANCHOR_COUNT - 1) / 2.0

cfg = GameConfig()
env = BombPotEnv(cfg)
s_vals, endp_mass = [], []
samples = []
for seed in range(500):
    obs, info = env.reset(seed, seed % cfg.num_seats)
    steps = 0
    while not info.terminal and steps < 200:
        steps += 1
        # normal raise node (full range available, not a dust cover)
        if info.gate_mask[GATE_RAISE] and info.max_raise_chips >= cfg.bb:
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            gm_t = torch.from_numpy(info.gate_mask).unsqueeze(0)
            sizing_t = torch.from_numpy(sizing_from_info(info)[None, :].astype(np.int64))
            with torch.no_grad():
                _, sp, _, _ = model(obs_t, gm_t)
                mu = (c + (c + 2.0) * torch.tanh(sp[..., 0])).item()
                s = (model._size_floor + model._size_span * torch.sigmoid(sp[..., 1])).item()
                probs = model._anchor_dist(sp, anchor_grid_torch(sizing_t)).probs.squeeze(0).numpy()
            s_vals.append(s)
            endp_mass.append(float(probs[0] + probs[-1]))
            if len(samples) < 4:
                samples.append((round(mu, 2), round(s, 2), np.round(probs, 3)))
        rng = random.Random(seed * 1000 + steps)
        legal = [g for g in (GATE_FOLD, GATE_CHECK_CALL, GATE_RAISE) if info.gate_mask[g]]
        lo, hi = info.min_raise_chips, info.max_raise_chips
        if GATE_RAISE in legal and lo > 0 and rng.random() < 0.5:
            g, chips = GATE_RAISE, (rng.randint(lo, hi) if hi > lo else hi)
        elif GATE_CHECK_CALL in legal:
            g, chips = GATE_CHECK_CALL, 0
        elif legal:
            g, chips = legal[0], 0
        else:
            break
        obs, rew, done, info = env.step_hybrid(g, chips)

s_vals = np.array(s_vals)
endp_mass = np.array(endp_mass)
print(f"scale s over {len(s_vals)} raise nodes: mean={s_vals.mean():.2f} "
      f"median={np.median(s_vals):.2f} min={s_vals.min():.2f} max={s_vals.max():.2f} "
      f"(floor={model._size_floor} cap={model._size_floor + model._size_span})")
print(f"endpoint mass (min+pot) per node: mean={endp_mass.mean():.2f} "
      f"median={np.median(endp_mass):.2f}  (uniform would be 2/11={2/11:.2f})")
print("\nsample nodes (mu, s, 11-anchor probs  min..pot):")
for mu, s, p in samples:
    print(f"  mu={mu:5}  s={s:4}  endpoints={p[0]+p[-1]:.2f}  interior={p[1:-1].sum():.2f}")
    print(f"            probs={p.tolist()}")
