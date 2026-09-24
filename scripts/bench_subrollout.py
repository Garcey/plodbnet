#!/usr/bin/env python
"""How does one rollout step's cost scale with the number of envs? (2026-09-24)

Runs the production collector (`collect_rollout_multiconfig`, ONE config) with
a trained checkpoint's actor + critic + training config at several env counts
and a fixed number of learner rows per env, and prints the per-step cost of
every timed region. A region whose ms/step barely grows with the env count is
fixed per-step overhead (launches, thread wake-ups, interpreter); one that
grows linearly is per-env work. That split decides whether more envs (fewer,
fatter steps) or leaner per-env code is the bigger lever.

    PLO5_RUST_ENCODER=1 .venv/bin/python scripts/bench_subrollout.py \\
        checkpoints/vMin2_40.pt --num-envs 7333,14666,29333,58666 --rows-per-env 40

Weights only shape the action mix (hand lengths); the pool holds copies of the
checkpoint's own actor.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
os.environ.setdefault("PLO5BP_STEP_TIMERS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from plo5bp import rollout as R  # noqa: E402
from plo5bp.config import GameConfig, TrainingConfig  # noqa: E402
from plo5bp.network import (  # noqa: E402
    build_actor_from_state_dict,
    build_critic_from_state_dict,
)
from plo5bp.selfplay import OpponentPool  # noqa: E402


_LAST: list = []
_orig_report = R._StepTimers.report


def _capture_report(self, label: str = "rollout") -> None:
    _LAST.append(self)
    _orig_report(self, label)


R._StepTimers.report = _capture_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--num-envs", default="7333,14666,29333")
    ap.add_argument("--rows-per-env", type=int, default=40)
    ap.add_argument("--seats", type=int, default=6)
    ap.add_argument("--stack-bb", type=int, default=20)
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfgd = dict(ck.get("config") or {})
    fields = {f.name for f in dataclasses.fields(TrainingConfig)}
    base_cfg = TrainingConfig(**{k: v for k, v in cfgd.items() if k in fields})
    actor = build_actor_from_state_dict(
        ck["model"], int(cfgd.get("hidden_dim", 128)), int(cfgd.get("num_layers", 2))
    ).to(device).eval()
    critic = build_critic_from_state_dict(ck["critic"]).to(device).eval()
    for p in list(actor.parameters()) + list(critic.parameters()):
        p.requires_grad_(False)
    pool = OpponentPool(capacity=args.pool, seed=args.seed)
    for i in range(args.pool):
        pool.snapshot(actor, tag=i)
    gcfg = GameConfig(
        num_seats=args.seats, starting_stack=args.stack_bb * 10_000, ante=30_000, bb=10_000
    )
    print(f"ckpt={args.ckpt} obs_mode={base_cfg.obs_mode} seats={args.seats} "
          f"stack={args.stack_bb}bb rows/env={args.rows_per_env} "
          f"RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS', 'default')}")

    def collect(n_envs: int):
        tcfg = dataclasses.replace(
            base_cfg, num_envs=n_envs, rollout_length=n_envs * args.rows_per_env
        )
        torch.manual_seed(args.seed)
        _LAST.clear()
        t0 = time.perf_counter()
        batch = R.collect_rollout_multiconfig(
            actor, pool, [gcfg], tcfg, np.random.default_rng(args.seed), critic=critic
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        rows = int(batch.obs.shape[0])
        del batch
        return wall, rows

    env_counts = [int(x) for x in args.num_envs.split(",") if x.strip()]
    # Warm-up at the largest size: CUDA kernels, the engine cache and the
    # (reused) staging buffer are all in place before anything is timed.
    print("[bench] warm-up ...", flush=True)
    collect(max(env_counts))
    results = []
    for n_envs in env_counts:
        wall, rows = collect(n_envs)
        tm = _LAST[-1] if _LAST else None
        steps = tm.counts.get("step1a/refresh", 0) if tm else 0
        per = {k: 1000.0 * v / max(steps, 1) for k, v in (tm.totals.items() if tm else [])}
        acc = sum(tm.totals.values()) if tm else 0.0
        results.append((n_envs, wall, rows, steps, per, acc))
        print(f"[bench] envs={n_envs} wall={wall:.1f}s rows={rows} steps={steps} "
              f"ms/step={1000 * wall / max(steps, 1):.2f} rows/s={rows / wall:,.0f}",
              flush=True)

    names = sorted({k for r in results for k in r[4]},
                   key=lambda k: -max(r[4].get(k, 0.0) for r in results))
    print("\nms per step by region")
    print(f"  {'region':30s}" + "".join(f"{r[0]:>10d}" for r in results))
    for k in names:
        print(f"  {k:30s}" + "".join(f"{r[4].get(k, 0.0):10.2f}" for r in results))
    print(f"  {'(untimed)':30s}" + "".join(
        f"{1000 * (r[1] - r[5]) / max(r[3], 1):10.2f}" for r in results))
    print(f"  {'TOTAL wall/step':30s}" + "".join(
        f"{1000 * r[1] / max(r[3], 1):10.2f}" for r in results))
    print(f"  {'rows/s':30s}" + "".join(f"{r[2] / r[1]:10.0f}" for r in results))


if __name__ == "__main__":
    main()
