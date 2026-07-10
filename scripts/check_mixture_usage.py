"""v5 SUCCESS METRIC: is the K-component mixture head actually producing
size MENUS, or has it collapsed to unimodal (== v4 with extra params)?

The v5 thesis (V5_DESIGN.md §2) is that the policy will split mass across
2-3 sizes where solvers do (e.g. a 33% block AND a pot bet). Collapse to
one component is graceful but means v5 bought nothing. This census makes
that visible — run it on a v5 checkpoint periodically during training,
the same way check_sizing_dist.py tracked v4's (mu, s).

Reports, over normal raise nodes across many random hands:
  - Hw  : mixture-weight entropy (0 = one component owns all mass;
          log K = perfectly split). Mean + distribution.
  - eff-K: perplexity exp(Hw), the "effective number of components in
          use", and the fraction of nodes with >=2 components above a
          0.15 weight floor.
  - mu-spread: stdev of the component locations (in anchor-index units)
          WEIGHTED by w — are the live components at DIFFERENT sizes, or
          coincident? A menu needs both split weights AND separated mu.
  - multi-modal %: fraction of nodes whose anchor MARGINAL has >=2 local
          maxima separated by a genuine valley (the thing v4 cannot do).

Usage: .venv/Scripts/python scripts/check_mixture_usage.py [checkpoint.pt]
       (default checkpoints/stub.pt; must be a v5 / mix_head checkpoint.)
"""
import random
import sys

import numpy as np
import torch

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv
from plo5bp.network import build_actor_from_state_dict
from plo5bp.sizing import ANCHOR_COUNT, anchor_grid_torch, sizing_from_info

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/stub.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt["model"] if "model" in ckpt else ckpt
if "mix_head.weight" not in sd:
    sys.exit(
        f"{ckpt_path} is not a v5 mixture checkpoint (no mix_head.weight; "
        f"head_version={ckpt.get('head_version')}). Use check_sizing_dist.py "
        "for v4."
    )
cfg_b = ckpt.get("config", {}) or {}
model = build_actor_from_state_dict(
    sd, int(cfg_b.get("hidden_dim", 2048)), int(cfg_b.get("num_layers", 4))
).eval()
K = model._mixture_k
print(f"loaded {ckpt_path}: v5 head, K={K}, weight_floor={model._mix_floor}")


def _local_maxima(p: np.ndarray, floor: float = 0.06) -> int:
    """Count separated peaks in the anchor marginal: a local max above
    `floor` with a strictly lower neighbor on each side (or the array
    edge), requiring a real valley between successive peaks."""
    peaks = 0
    last_peak = -2
    for i in range(len(p)):
        if p[i] < floor:
            continue
        left = p[i - 1] if i > 0 else -1.0
        right = p[i + 1] if i < len(p) - 1 else -1.0
        if p[i] >= left and p[i] >= right:
            # require a valley since the previous peak
            if last_peak >= 0 and p[last_peak + 1 : i].min(initial=1.0) > 0.5 * min(
                p[last_peak], p[i]
            ):
                continue
            peaks += 1
            last_peak = i
    return peaks


cfg = GameConfig()
env = BombPotEnv(cfg)
idx = torch.arange(ANCHOR_COUNT, dtype=torch.float32)
Hw_all, effK_all, muspread_all, multimodal, split2 = [], [], [], [], []
samples = []

for seed in range(500):
    obs, info = env.reset(seed, seed % cfg.num_seats)
    steps = 0
    while not info.terminal and steps < 200:
        steps += 1
        if info.gate_mask[GATE_RAISE] and info.max_raise_chips >= cfg.bb:
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            gm_t = torch.from_numpy(info.gate_mask).unsqueeze(0)
            sizing_t = torch.from_numpy(
                sizing_from_info(info)[None, :].astype(np.int64)
            )
            with torch.no_grad():
                _, mp, _, _ = model(obs_t, gm_t)
                mu, s, w = (t.squeeze(0) for t in model.mixture_params(mp))
                probs = (
                    model._anchor_dist(mp, anchor_grid_torch(sizing_t))
                    .probs.squeeze(0)
                    .numpy()
                )
            w = w.numpy()
            mu = mu.numpy()
            Hw = float(-(w * np.log(np.clip(w, 1e-9, None))).sum())
            effK = float(np.exp(Hw))
            wmean = float((w * mu).sum())
            muspread = float(np.sqrt((w * (mu - wmean) ** 2).sum()))
            Hw_all.append(Hw)
            effK_all.append(effK)
            muspread_all.append(muspread)
            split2.append(int((w > 0.15).sum() >= 2))
            multimodal.append(int(_local_maxima(probs) >= 2))
            if len(samples) < 5 and (w > 0.15).sum() >= 2:
                samples.append(
                    (np.round(mu, 2), np.round(w, 3), np.round(probs, 3))
                )
        rng = random.Random(seed * 1000 + steps)
        legal = [
            g for g in (GATE_FOLD, GATE_CHECK_CALL, GATE_RAISE) if info.gate_mask[g]
        ]
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

n = len(Hw_all)
Hw_all = np.array(Hw_all)
effK_all = np.array(effK_all)
muspread_all = np.array(muspread_all)
print(f"\n{n} raise nodes sampled. Max Hw = log K = {np.log(K):.2f}\n")
print(f"Hw (weight entropy):   mean={Hw_all.mean():.3f}  median={np.median(Hw_all):.3f}")
print(f"eff-K (exp Hw):        mean={effK_all.mean():.2f}   (1.0 = fully collapsed)")
print(f"mu-spread (idx units): mean={muspread_all.mean():.2f}  median={np.median(muspread_all):.2f}")
print(f"nodes with >=2 components w>0.15:  {100 * np.mean(split2):.1f}%")
print(f"nodes multi-modal (>=2 marginal peaks): {100 * np.mean(multimodal):.1f}%")
print(
    "\nInterpretation: eff-K~1 + mu-spread~0 => collapsed to v4 (fine, but v5 "
    "bought nothing there).\neff-K>1.3 with mu-spread>1.5 AND multi-modal% up "
    "=> the menu is alive (v5 thesis working)."
)
if samples:
    print("\nsample MENU nodes (>=2 live components):")
    for mu, w, p in samples:
        print(f"  mu(idx)={mu.tolist()}  w={w.tolist()}")
        print(f"     marginal(min..pot)={p.tolist()}")
else:
    print("\n(no multi-component nodes found — the head is unimodal here.)")
