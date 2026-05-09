"""Bankroll-requirement Monte-Carlo simulation.

Loads a trained network and plays it against itself across N hands
sampled from the clubgg seat-and-stack distribution. Records per-hand
hero (seat 0) payout in big blinds and reports standard deviation,
quantiles, and a small bankroll-requirement table for hypothesized
win rates.

Symmetric self-play has expected mean ~0 (zero-sum, same policy
every seat). The interesting output is sigma in bb/hand. Raw payouts
are saved to disk so the user can post-process per-band/per-position.

Example:
    .venv/Scripts/python scripts/bankroll_sim.py \
        --num-hands 1000000 --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# scripts/ isn't a package, so make sibling files importable when this
# script is invoked directly.
sys.path.insert(0, str(Path(__file__).parent))
from train import _sample_game_config  # noqa: E402

from plo5bp.env import BombPotEnv  # noqa: E402
from plo5bp.eval import model_policy  # noqa: E402
from plo5bp.network import ActorCritic  # noqa: E402


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("[device] cuda unavailable -- falling back to cpu", flush=True)
        return torch.device("cpu")
    return torch.device(name)


def load_checkpoint(path: Path, device: torch.device) -> ActorCritic:
    if not path.exists():
        raise SystemExit(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        cfg_block = ckpt.get("config", {}) or {}
        hidden_dim = int(cfg_block.get("hidden_dim", 128))
        num_layers = int(cfg_block.get("num_layers", 2))
    else:
        state_dict = ckpt
        hidden_dim = 128
        num_layers = 2
    model = ActorCritic(hidden_dim=hidden_dim, num_layers=num_layers)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(
        f"[load] {path} hidden_dim={hidden_dim} num_layers={num_layers} "
        f"device={device}",
        flush=True,
    )
    return model


def _parse_seats(spec: str) -> tuple[int, ...]:
    out = tuple(int(x.strip()) for x in spec.split(",") if x.strip())
    if not out or any(n < 2 for n in out):
        raise SystemExit(f"--seats-range must list ints >= 2, got {spec!r}")
    return out


def _parse_stack_range(spec: str) -> tuple[float, float]:
    if ":" not in spec:
        raise SystemExit(f"--stack-range must be 'min:max' in bb, got {spec!r}")
    lo, hi = (float(x) for x in spec.split(":", 1))
    if lo <= 0 or hi < lo:
        raise SystemExit(f"invalid --stack-range {spec!r}")
    return lo, hi


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/stub.pt"))
    p.add_argument("--num-hands", type=int, default=1_000_000)
    p.add_argument("--seats-range", type=str, default="2,3,4,5,6")
    p.add_argument("--stack-range", type=str, default="1:300")
    p.add_argument("--bb", type=int, default=10000)
    p.add_argument("--ante", type=int, default=30000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true",
                   help="Use argmax gate + Beta mean (lower sigma; default samples)")
    p.add_argument("--out", type=Path,
                   default=Path("runs/bankroll_sim_payouts.npy"))
    p.add_argument("--print-every", type=int, default=10000)
    args = p.parse_args()

    seats_choices = _parse_seats(args.seats_range)
    stack_lo_bb, stack_hi_bb = _parse_stack_range(args.stack_range)
    device = _resolve_device(args.device)

    model = load_checkpoint(args.checkpoint, device)
    pol = model_policy(model, deterministic=args.deterministic)
    rng = np.random.default_rng(args.seed)

    n = int(args.num_hands)
    payouts_bb = np.empty(n, dtype=np.float64)

    print(
        f"[sim] num_hands={n} seats={seats_choices} stack_bb=[{stack_lo_bb},"
        f"{stack_hi_bb}] bb={args.bb} ante={args.ante} "
        f"deterministic={args.deterministic} seed={args.seed}",
        flush=True,
    )
    t0 = time.time()
    for i in range(n):
        cfg, _ = _sample_game_config(
            seats_choices, stack_lo_bb, stack_hi_bb,
            args.bb, args.ante, rng,
            stack_dist="clubgg", seats_dist="clubgg",
        )
        button = int(rng.integers(0, cfg.num_seats))
        hand_seed = int(rng.integers(0, 2**63 - 1))
        env = BombPotEnv(cfg)
        obs, info = env.reset(hand_seed, button)
        while not info.terminal:
            actor = info.actor
            gate, chips = pol(obs, actor, info)
            obs, rs, done, info = env.step_hybrid(gate, chips)
            if done:
                payouts_bb[i] = float(rs[0]) / float(cfg.bb)
                break

        if (i + 1) % args.print_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta_min = (n - i - 1) / rate / 60.0 if rate > 0 else float("inf")
            running = payouts_bb[: i + 1]
            print(
                f"[{elapsed:7.1f}s] {i + 1:>9} hands | "
                f"mean={running.mean():+.4f} bb std={running.std():.3f} bb | "
                f"{rate:6.1f} hands/s eta={eta_min:.1f}m",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, payouts_bb)

    mean = float(payouts_bb.mean())
    sigma = float(payouts_bb.std())
    stderr = sigma / float(np.sqrt(n))
    print()
    print(f"[done] wrote {args.out} ({n} payouts, float64)")
    print(f"[stats] across {n} hands:")
    print(f"  mean         : {mean:+.5f} bb/hand  (stderr {stderr:.5f})")
    print(f"  std dev      : {sigma:.4f} bb/hand")
    print(f"  variance     : {sigma * sigma:.3f} bb^2/hand")
    for q in (0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99):
        print(f"  P{int(q * 100):02}          : {np.quantile(payouts_bb, q):+8.2f} bb")
    print(f"  min          : {payouts_bb.min():+8.2f} bb")
    print(f"  max          : {payouts_bb.max():+8.2f} bb")

    print()
    print("[bankroll] B = sigma^2 / (2*mu) * ln(1/r) -- Brownian-motion approx")
    print("           sigma from this self-play sim; mu is hypothesized win rate")
    print("           (real-world mu != self-play mean of ~0)")
    print(f"           sigma^2 = {sigma * sigma:.2f} bb^2/hand")
    header = f"  {'win rate':>14} | " + " | ".join(
        f"RoR {int(r * 100):>2}%" for r in (0.01, 0.05, 0.10)
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for win_rate_per_100 in (0.5, 1.0, 2.0, 5.0):
        mu = win_rate_per_100 / 100.0  # bb/hand
        cells = []
        for r in (0.01, 0.05, 0.10):
            B_bb = (sigma * sigma) / (2.0 * mu) * float(np.log(1.0 / r))
            cells.append(f"{B_bb:8.0f} bb")
        print(f"  {win_rate_per_100:5.1f} bb/100  | " + " | ".join(cells))


if __name__ == "__main__":
    main()
