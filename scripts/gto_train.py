#!/usr/bin/env python
"""Train a GTO PolicyNet (Phase 1).

Default data path is the **rule-based bootstrap** (no PPO). The retired
nlh1–nlh4 PPO lineage is not used. Optional ``--teacher`` remains for
any future external ActorCritic you trust; native rust_cfr JSONL is
the GTO teacher via ``scripts/train_policy_from_cfr.py``.

Examples::

  # Bootstrap curriculum (recommended day-1)
  .venv/Scripts/python scripts/gto_train.py \\
      --bootstrap --n-decisions 8192 --epochs 10 \\
      --out checkpoints/gto_policy.pt

  # Pre-collected rows
  .venv/Scripts/python scripts/gto_train.py --data data/gto_nlh/bootstrap.npz \\
      --epochs 10 --out checkpoints/gto_policy.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

import torch  # noqa: E402

from plo5bp.gto.bootstrap import collect_bootstrap  # noqa: E402
from plo5bp.gto.dataset import (  # noqa: E402
    collect_teacher_distill,
    load_rows_npz,
    save_rows_npz,
)
from plo5bp.gto.train import TrainConfig, train_policy_net  # noqa: E402
from plo5bp.network import build_actor_from_state_dict  # noqa: E402


def _load_teacher(path: Path, device: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit(f"bad checkpoint {path}")
    state = ckpt.get("model") or ckpt.get("actor") or ckpt
    if "torso.0.weight" not in state and "torso.0.0.weight" not in state:
        raise SystemExit(f"no actor weights in {path}")
    w = state.get("torso.0.weight")
    if w is None:
        w = state["torso.0.0.weight"]
        n_blocks = sum(
            1
            for k in state
            if k.startswith("torso.") and k.endswith(".linear.weight")
        )
        layers = max(3, n_blocks + 1)
    else:
        n_lin = sum(
            1
            for k in state
            if k.startswith("torso.")
            and k.endswith(".weight")
            and k.count(".") == 2
        )
        layers = max(1, n_lin)
    hidden = int(w.shape[0])
    model = build_actor_from_state_dict(state, hidden, layers)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def main() -> int:
    p = argparse.ArgumentParser(description="Train NLH GTO PolicyNet")
    p.add_argument(
        "--bootstrap",
        action="store_true",
        help="Rule-based pure-node curriculum (default if no --data/--teacher)",
    )
    p.add_argument(
        "--teacher",
        type=Path,
        default=None,
        help="Optional external ActorCritic to distill (not nlh PPO lineage)",
    )
    p.add_argument("--data", type=Path, default=None, help="Pre-collected .npz")
    p.add_argument("--save-data", type=Path, default=None)
    p.add_argument("--n-decisions", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("checkpoints/gto_policy.pt"))
    args = p.parse_args()

    if args.data is not None:
        print(f"[gto-train] loading rows from {args.data}")
        rows = load_rows_npz(args.data)
        source = "npz_bundle"
    elif args.teacher is not None:
        print(
            f"[gto-train] WARNING: distilling from {args.teacher} "
            f"(not the retired nlh PPO stem)"
        )
        teacher = _load_teacher(args.teacher, args.device)
        rows = collect_teacher_distill(
            teacher,
            n_decisions=args.n_decisions,
            seed=args.seed,
            device=args.device,
        )
        source = f"teacher_distill:{args.teacher.name}"
    else:
        # Default: bootstrap
        print(
            f"[gto-train] bootstrap curriculum n={args.n_decisions} "
            f"(no PPO teacher)"
        )
        rows = collect_bootstrap(
            n_decisions=args.n_decisions,
            seed=args.seed,
            pure_only=False,
        )
        source = "rule_bootstrap"

    if args.save_data is not None:
        save_rows_npz(args.save_data, rows)
        print(f"[gto-train] wrote {args.save_data} ({len(rows)} rows)")

    if not rows:
        raise SystemExit("no training rows collected")

    print(f"[gto-train] {len(rows)} rows; training…")
    cfg = TrainConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
    )
    result = train_policy_net(
        rows,
        args.out,
        cfg=cfg,
        meta={
            "source": source,
            "n_train": len(rows),
            "teacher": str(args.teacher) if args.teacher else None,
            "is_gto_validated": False,
        },
    )
    print(json.dumps(result.as_dict(), indent=2))
    from plo5bp.gto.policy_net import source_is_gto_teacher  # noqa: E402

    if not source_is_gto_teacher(source):
        print(
            f"[gto-train] source={source!r} — badge will show Curriculum / "
            "unvalidated (not GTO AI)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
