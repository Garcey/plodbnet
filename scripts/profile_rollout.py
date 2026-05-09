"""Wall-clock comparison of serial vs batched rollout.

Usage::

    .venv/Scripts/python scripts/profile_rollout.py

Runs N_STEPS of rollout data through both paths with a small random-init
`ActorCritic`, no opponent pool, at 6-seat 20bb (matching the plan's
target configuration). Reports steps/sec, total wall clock, and the
speedup ratio. Targets from the plan:

  - Phase A alone      ≥ 3×
  - Phase A+B          ≥ 8×
  - Phase A+B+C        ≥ 12×
  - Phase A+B+C+D      ≥ 15×

The serial path is `rollout.collect_rollout`, the batched path is
`rollout.collect_rollout_batched`. Both configure the same `TrainingConfig`
(num_envs, rollout_length); only the driver differs.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.network import ActorCritic
from plo5bp.rollout import collect_rollout, collect_rollout_batched
from plo5bp.selfplay import OpponentPool


def _run_once(fn, learner, pool, gcfg, tcfg, rng):
    t0 = time.perf_counter()
    batch = fn(learner, pool, gcfg, tcfg, rng)
    t1 = time.perf_counter()
    return t1 - t0, batch.obs.shape[0]


def run(
    num_seats: int,
    starting_stack: int,
    num_envs: int,
    rollout_length: int,
    hidden_dim: int,
    opponents: int,
    warmup: int = 1,
) -> None:
    gcfg = GameConfig(num_seats=num_seats, starting_stack=starting_stack)
    tcfg = TrainingConfig(
        num_envs=num_envs,
        rollout_length=rollout_length,
        hidden_dim=hidden_dim,
        pool_mix_prob=0.5 if opponents > 0 else 0.0,
        pool_opp_seats=min(2, num_seats - 1) if opponents > 0 else 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} seats={num_seats} stack={starting_stack}bb/100 "
          f"num_envs={num_envs} rollout_length={rollout_length}")

    learner = ActorCritic(hidden_dim=hidden_dim).to(device).eval()
    pool = OpponentPool(capacity=max(1, opponents))
    for _ in range(opponents):
        snap = ActorCritic(hidden_dim=hidden_dim).to(device).eval()
        pool.snapshot(snap)

    for warm in range(warmup):
        torch.manual_seed(100 + warm)
        _run_once(
            collect_rollout, learner, pool, gcfg, tcfg,
            np.random.default_rng(100 + warm),
        )
        torch.manual_seed(200 + warm)
        _run_once(
            collect_rollout_batched, learner, pool, gcfg, tcfg,
            np.random.default_rng(200 + warm),
        )

    torch.manual_seed(0)
    serial_t, serial_rows = _run_once(
        collect_rollout, learner, pool, gcfg, tcfg, np.random.default_rng(0)
    )
    torch.manual_seed(0)
    batched_t, batched_rows = _run_once(
        collect_rollout_batched, learner, pool, gcfg, tcfg, np.random.default_rng(0)
    )

    serial_rate = serial_rows / serial_t
    batched_rate = batched_rows / batched_t
    speedup = batched_rate / max(serial_rate, 1e-9)

    print(
        f"  serial  : {serial_t:6.2f}s  {serial_rows:>6} rows  {serial_rate:8.1f} rows/s"
    )
    print(
        f"  batched : {batched_t:6.2f}s  {batched_rows:>6} rows  {batched_rate:8.1f} rows/s"
    )
    print(f"  speedup : {speedup:5.2f}x")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seats", type=int, default=6)
    p.add_argument("--stack", type=int, default=2000)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--rollout-length", type=int, default=2048)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--opponents", type=int, default=0,
                   help="Pool snapshots to seed (enables pool-mix path).")
    args = p.parse_args()

    run(
        num_seats=args.seats,
        starting_stack=args.stack,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        hidden_dim=args.hidden_dim,
        opponents=args.opponents,
    )


if __name__ == "__main__":
    main()
