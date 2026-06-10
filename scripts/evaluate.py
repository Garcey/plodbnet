"""Evaluate a trained checkpoint against baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.eval import (
    always_call_policy,
    always_pot_bet_policy,
    model_policy,
    random_legal_policy,
    run_match,
)
from plo5bp.network import model_class_for_state_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-hands", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cpu", choices=("cpu", "cuda")
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but torch.cuda.is_available() is False.")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_cfg_raw = ckpt.get("config", {})
    hidden = int(train_cfg_raw.get("hidden_dim", TrainingConfig.hidden_dim))
    num_layers = int(train_cfg_raw.get("num_layers", TrainingConfig.num_layers))
    model_cls = model_class_for_state_dict(ckpt["model"])
    model = model_cls(hidden_dim=hidden, num_layers=num_layers)
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    model.eval()

    game_cfg = GameConfig()
    rng = np.random.default_rng(args.seed)
    hero = model_policy(model, deterministic=False)

    baselines = {
        "random_legal": random_legal_policy(rng),
        "always_call": always_call_policy(),
        "always_pot_bet": always_pot_bet_policy(),
    }
    for name, policy in baselines.items():
        stats = run_match(hero, policy, game_cfg, args.num_hands, seed=args.seed)
        print(
            f"vs {name:16s}  mean={stats.hero_reward_mean:+.4f}  "
            f"std={stats.hero_reward_std:.4f}  n={stats.num_hands}"
        )


if __name__ == "__main__":
    main()
