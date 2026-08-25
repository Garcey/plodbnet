#!/usr/bin/env python3
"""CFR → LabelRecord → PolicyNet supervised training entrypoint.

Examples
--------
# Export push/fold solve + train a tiny net
.venv/Scripts/python scripts/train_policy_from_cfr.py \\
    --strategies data/cfr/pushfold_4handed_10bb_300k.json \\
    --labels-out data/gto_nlh/pushfold_4h.jsonl \\
    --ckpt checkpoints/gto_pushfold_smoke.pt \\
    --hidden-dim 256 --num-layers 2 --epochs 3

# Export a batch directory of strategy JSONs then train
.venv/Scripts/python scripts/train_policy_from_cfr.py \\
    --strategies data/cfr/verify/batch/strategies \\
    --labels-out data/gto_nlh/batch_labels.jsonl \\
    --ckpt checkpoints/gto_batch.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root on path when run as script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train PolicyNet from native CFR solves")
    p.add_argument(
        "--strategies",
        type=Path,
        required=True,
        help="SolveReport JSON file or directory of strategy JSONs",
    )
    p.add_argument(
        "--labels-out",
        type=Path,
        default=Path("data/gto_nlh/cfr_labels.jsonl"),
        help="Output LabelRecord JSONL path",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=Path("checkpoints/gto_policy_cfr.pt"),
        help="Output PolicyNet checkpoint",
    )
    p.add_argument("--source", type=str, default="rust_cfr", help="Label source tag")
    p.add_argument(
        "--max-infosets",
        type=int,
        default=None,
        help="Cap infosets per file (default: all)",
    )
    p.add_argument(
        "--require-hole",
        action="store_true",
        help="Skip labels without decodable hole/class",
    )
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--export-only",
        action="store_true",
        help="Only write labels JSONL; skip training",
    )
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="Use existing --labels-out; skip strategy export",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on supervised rows after synthesis",
    )
    from plo5bp.gto.teacher import (  # noqa: E402
        TEACHER_HOLDOUT_FRAC,
        TEACHER_MAX_EXPL_BB,
        TEACHER_MIN_VISIT_MASS,
        TEACHER_SPLIT_SEED,
    )

    p.add_argument(
        "--max-expl-bb",
        type=float,
        default=TEACHER_MAX_EXPL_BB,
        help="Skip high-expl roots at export (default 1.0; --no-expl-floor disables)",
    )
    p.add_argument("--no-expl-floor", action="store_true")
    p.add_argument(
        "--min-visit-mass",
        type=float,
        default=TEACHER_MIN_VISIT_MASS,
        help="Drop infosets with visit_mass present and below this (default 1.0)",
    )
    p.add_argument(
        "--holdout-frac",
        type=float,
        default=TEACHER_HOLDOUT_FRAC,
        help="Holdout root fraction written next to --labels-out (default 0.15)",
    )
    p.add_argument("--holdout-out", type=Path, default=None)
    p.add_argument("--split-seed", type=int, default=TEACHER_SPLIT_SEED)
    p.add_argument("--split-manifest", type=Path, default=None)
    args = p.parse_args(argv)

    from plo5bp.gto.cfr_export import export_teacher_dir
    from plo5bp.gto.dataset import load_label_shard_rows
    from plo5bp.gto.train import TrainConfig, train_policy_net

    if not args.skip_export:
        res = export_teacher_dir(
            args.strategies,
            args.labels_out,
            source=args.source,
            max_infosets_per_file=args.max_infosets,
            require_hole=args.require_hole,
            max_expl_bb=None if args.no_expl_floor else args.max_expl_bb,
            min_visit_mass=args.min_visit_mass,
            holdout_frac=args.holdout_frac,
            holdout_jsonl=args.holdout_out,
            split_seed=args.split_seed,
            split_manifest=args.split_manifest,
        )
        n_lab = res.n_train
        print(
            f"[cfr-train] exported train={n_lab} holdout={res.n_holdout} "
            f"skipped_expl={len(res.skipped_expl)} -> {args.labels_out}"
        )
        if res.holdout_path:
            print(f"[cfr-train] holdout -> {res.holdout_path}")
        if n_lab == 0:
            print("[cfr-train] ERROR: zero labels exported", file=sys.stderr)
            return 2
    else:
        if not args.labels_out.is_file():
            print(f"[cfr-train] missing labels {args.labels_out}", file=sys.stderr)
            return 2
        print(f"[cfr-train] using existing labels {args.labels_out}")

    if args.export_only:
        return 0

    rows = load_label_shard_rows(args.labels_out, synthesize_obs=True)
    if args.max_rows is not None and len(rows) > args.max_rows:
        rows = rows[: args.max_rows]
    print(f"[cfr-train] supervised rows={len(rows)} (obs synthesized)")
    if not rows:
        print(
            "[cfr-train] ERROR: no rows after obs synthesis "
            "(need hero_hole or class_id on labels)",
            file=sys.stderr,
        )
        return 3

    # Quality summary
    n_raise = sum(1 for r in rows if float(r.gate_probs[2]) > 0.05)
    n_fold = sum(1 for r in rows if float(r.gate_probs[0]) > 0.05)
    print(f"[cfr-train] rows_with_raise_mass~={n_raise} fold_mass~={n_fold}")

    cfg = TrainConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        log_every=max(1, len(rows) // (args.batch_size * 4) or 1),
    )
    meta = {
        "source": args.source,
        "labels_path": str(args.labels_out),
        "strategies": str(args.strategies),
        "n_labels_file": None,
        "pipeline": "cfr_export→obs_from_label→train_policy_net",
        "is_gto_validated": False,  # need holdout probe for badge
    }
    try:
        meta["n_labels_file"] = sum(
            1 for _ in open(args.labels_out, encoding="utf-8") if _.strip()
        )
    except OSError:
        pass

    result = train_policy_net(rows, args.ckpt, cfg=cfg, meta=meta)
    summary = {
        **result.as_dict(),
        "labels": str(args.labels_out),
        "ckpt": str(args.ckpt),
    }
    print("[cfr-train] done", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
