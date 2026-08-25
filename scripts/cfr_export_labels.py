#!/usr/bin/env python
"""Export CFR strategy JSON → LabelRecord JSONL (source=rust_cfr_*)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.cfr_export import export_teacher_dir  # noqa: E402
from plo5bp.gto.teacher import (  # noqa: E402
    TEACHER_HOLDOUT_FRAC,
    TEACHER_MAX_EXPL_BB,
    TEACHER_MIN_VISIT_MASS,
    TEACHER_SPLIT_SEED,
)


def main() -> int:
    p = argparse.ArgumentParser(description="CFR strategy → labels")
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", type=str, default="rust_cfr_river")
    p.add_argument(
        "--max-infosets",
        type=int,
        default=None,
        help="Cap infosets per file (default: export all)",
    )
    p.add_argument(
        "--max-expl-bb",
        type=float,
        default=TEACHER_MAX_EXPL_BB,
        help="Skip roots with missing/high expl (default 1.0)",
    )
    p.add_argument(
        "--no-expl-floor",
        action="store_true",
        help="Do not skip high-expl roots at export",
    )
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
        help="Fraction of roots written to the holdout JSONL (default 0.15)",
    )
    p.add_argument(
        "--holdout-out",
        type=Path,
        default=None,
        help="Holdout JSONL path (default: <out-stem>_holdout.jsonl)",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=TEACHER_SPLIT_SEED,
        help="Deterministic holdout hash seed (default 0)",
    )
    p.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Root-id split manifest (default: <out-stem>_split.json)",
    )
    args = p.parse_args()

    res = export_teacher_dir(
        args.in_path,
        args.out,
        source=args.source,
        max_infosets_per_file=args.max_infosets,
        max_expl_bb=None if args.no_expl_floor else args.max_expl_bb,
        min_visit_mass=args.min_visit_mass,
        holdout_frac=args.holdout_frac,
        holdout_jsonl=args.holdout_out,
        split_seed=args.split_seed,
        split_manifest=args.split_manifest,
    )
    print(
        f"[cfr_export] train={res.n_train} holdout={res.n_holdout} "
        f"skipped_expl={len(res.skipped_expl)} -> {args.out}"
        + (f" holdout={res.holdout_path}" if res.holdout_path else "")
    )
    if res.split_path:
        print(f"[cfr_export] split manifest -> {res.split_path}")
    if res.n_train == 0 and res.n_holdout == 0:
        return 1
    if res.n_train == 0:
        print("[cfr_export] warning: train split is empty (all roots holdout or dropped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
