"""Evaluate a trained checkpoint against baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from plo5bp.config import (
    VARIANT_NLH,
    VARIANT_PLO4,
    VARIANT_PLO5,
    VARIANT_PLO6,
    GameConfig,
    TrainingConfig,
)
from plo5bp.eval import (
    always_call_policy,
    always_pot_bet_policy,
    model_policy,
    random_legal_policy,
    run_match,
)
from plo5bp.network import build_actor_from_state_dict


def _build_game_config(args: argparse.Namespace) -> GameConfig:
    """Fixed GameConfig from the CLI flags, mirroring scripts/train.py's
    variant/stake conventions: bomb-pot variants have no blinds (sb=0, ante
    default 3bb, 200000-chip stacks); NLH posts SB/BB (defaults bb/2) with a
    0.5bb ante and 100bb (1000000-chip) stacks."""
    bb = args.bb
    if args.variant == VARIANT_NLH:
        return GameConfig(
            num_seats=args.num_seats,
            starting_stack=(
                args.starting_stack if args.starting_stack is not None else 1_000_000
            ),
            ante=args.ante if args.ante is not None else bb // 2,
            bb=bb,
            sb=args.sb if args.sb is not None else bb // 2,
            variant=VARIANT_NLH,
        )
    return GameConfig(
        num_seats=args.num_seats,
        starting_stack=(
            args.starting_stack if args.starting_stack is not None else 200_000
        ),
        ante=args.ante if args.ante is not None else bb * 3,
        bb=bb,
        sb=0,
        variant=args.variant,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-hands", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cpu", choices=("cpu", "cuda")
    )
    # Game config — mirrors scripts/train.py's flag names so non-PLO5
    # checkpoints can be evaluated. Must match the checkpoint's trained
    # variant (enforced against ckpt["variant"]).
    parser.add_argument(
        "--variant",
        choices=[VARIANT_PLO5, VARIANT_PLO4, VARIANT_PLO6, VARIANT_NLH],
        default=VARIANT_PLO5,
        help="Game variant of the checkpoint. plo4/plo6_double_bomb = the "
        "PLO5 bomb pot with 4/6 hole cards; nlh_single = no-limit hold'em "
        "(obs width 995 vs PLO 991). Default plo5_double_bomb.",
    )
    parser.add_argument("--num-seats", type=int, default=6,
                        help="Seats at the table.")
    parser.add_argument("--starting-stack", type=int, default=None,
                        help="Per-seat starting stack in chips. Default 200000 "
                        "(bomb pots) / 1000000 (NLH, 100bb).")
    parser.add_argument("--ante", type=int, default=None,
                        help="Per-player ante in chips. Default 3bb (bomb pots) "
                        "/ 0.5bb (NLH).")
    parser.add_argument("--bb", type=int, default=10000,
                        help="Chips per bb (default 10000).")
    parser.add_argument("--sb", type=int, default=None,
                        help="Small blind in chips (NLH only). Default bb/2; "
                        "ignored for bomb pots.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but torch.cuda.is_available() is False.")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_variant = ckpt.get("variant") if isinstance(ckpt, dict) else None
    if ckpt_variant is not None and ckpt_variant != args.variant:
        raise SystemExit(
            f"checkpoint variant {ckpt_variant!r} != --variant {args.variant!r}. "
            f"Pass --variant {ckpt_variant} so the obs width and game rules "
            "match (an NLH net is 995-wide, PLO 991)."
        )
    train_cfg_raw = ckpt.get("config", {})
    hidden = int(train_cfg_raw.get("hidden_dim", TrainingConfig.hidden_dim))
    num_layers = int(train_cfg_raw.get("num_layers", TrainingConfig.num_layers))
    model = build_actor_from_state_dict(ckpt["model"], hidden, num_layers)
    model.to(args.device)
    model.eval()

    game_cfg = _build_game_config(args)
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
