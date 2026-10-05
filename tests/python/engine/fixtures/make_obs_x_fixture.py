"""How obs_x_trainer.npz was made (2026-10-05; for test_obs_x_serving.py): seeded hands played
with random legal actions through the TRAINING code's engine -- the side worktree vSix7 trains
from, whose batched env computes the obs-X groups (here run+line, at the serial env's 1024
opp-outcome samples) -- recording every decision's full 1246-wide observation and the action
taken. It needs that worktree's package (this repo's batched env has no obs-X groups):
  PYTHONPATH=<training worktree>/python PLO5BP_OBS_REV=1 python make_obs_x_fixture.py OUT.npz
"""
import os
import sys

os.environ.setdefault("PLO5_RUST_ENCODER", "1")
import numpy as np

from plo5bp.config import GameConfig
from plo5bp.env_batched import BatchedBombPotEnv

out_path = sys.argv[1]
CONFIGS = [
    (6, (150000, 400000, 900000, 2500000, 300000, 700000)),
    (3, (200000, 600000, 1200000)),
    (2, (300000, 800000)),
]
HANDS = 10
rows = {"cfg": [], "hand": [], "seed": [], "button": [], "gate": [], "chips": [], "obs": []}
for ci, (n, stacks) in enumerate(CONFIGS):
    cfg = GameConfig(num_seats=n, starting_stacks=stacks)
    for h in range(HANDS):
        seed, button = 70000 + 97 * h + ci, h % n
        env = BatchedBombPotEnv(1, cfg, ev_runout_samples=0, opp_outcome_mc=1024, obs_mode="full",
                                obs_x_groups=5, obs_x_run_samples=48)
        env.reset_batch(np.array([seed], dtype=np.uint64), np.array([button], dtype=np.uint8))
        rng = np.random.default_rng(seed)
        for _ in range(60):
            if bool(env._dones[0]) or int(env._actors[0]) < 0:
                break
            gm = np.asarray(env._gate_mask[0], dtype=bool)
            legal = np.nonzero(gm)[0]
            gate = int(rng.choice(legal, p=None))
            lo, hi = int(env._min_raise[0]), int(env._max_raise[0])
            chips = int(rng.integers(lo, hi + 1)) if gate == 2 and hi >= lo and hi > 0 else 0
            rows["cfg"].append(ci); rows["hand"].append(h); rows["seed"].append(seed)
            rows["button"].append(button); rows["gate"].append(gate); rows["chips"].append(chips)
            rows["obs"].append(np.asarray(env._obs[0], dtype=np.float32).copy())
            env.step_hybrid_batch(np.array([gate], dtype=np.uint8), np.array([chips], dtype=np.uint64))
arrays = {k: np.asarray(v) for k, v in rows.items()}
arrays["obs"] = np.stack(rows["obs"]).astype(np.float32)
arrays["stacks"] = np.array([list(s) + [0] * (6 - len(s)) for _, s in CONFIGS], dtype=np.int64)
arrays["seats"] = np.array([n for n, _ in CONFIGS], dtype=np.int64)
np.savez_compressed(out_path, **arrays)
tail = arrays["obs"][:, 1171:]
print("decisions", len(rows["gate"]), "obs", arrays["obs"].shape, "tail nonzero share", float((tail != 0).mean()),
      "RUN nonzero", float((tail[:, :7] != 0).mean()), "LINE nonzero", float((tail[:, 17:63] != 0).mean()))
