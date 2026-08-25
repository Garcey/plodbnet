#!/usr/bin/env python
"""Batch native CFR solves with resume + manifest.

Examples::

  .venv/Scripts/python scripts/cfr_batch.py --n-roots 4 --dry-run --out-dir data/cfr/smoke
  .venv/Scripts/python scripts/cfr_batch.py --n-roots 4 --iters 100 --out-dir data/cfr/smoke
  .venv/Scripts/python scripts/cfr_batch.py --n-roots 4 --resume --out-dir data/cfr/smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.cfr_batch import expand_river_grid, run_batch  # noqa: E402
from plo5bp.gto.teacher import TEACHER_MAX_EXPL_BB  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Batch NLH CFR solves")
    p.add_argument("--n-roots", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pot-bb", type=float, default=10.0)
    p.add_argument("--stack-bb", type=float, default=50.0)
    p.add_argument("--size-preset", type=str, default="coarse")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--streets",
        type=str,
        default="3",
        help="Comma streets 1=flop 2=turn 3=river (default river)",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--max-expl-bb",
        type=float,
        default=TEACHER_MAX_EXPL_BB,
        help="Reject roots with missing/high exploitability (default 1.0 HU river)",
    )
    p.add_argument(
        "--no-expl-floor",
        action="store_true",
        help="Disable the per-root exploitability cap (keep every ok solve)",
    )
    args = p.parse_args()

    resume = not args.no_resume
    streets = [int(x) for x in args.streets.split(",") if x.strip()]
    jobs = expand_river_grid(
        n_roots=args.n_roots,
        seed=args.seed,
        pot_bb=args.pot_bb,
        stack_bb=args.stack_bb,
        size_preset=args.size_preset,
        iters=args.iters,
        streets=streets,
    )
    max_expl = None if args.no_expl_floor else args.max_expl_bb
    man = run_batch(
        jobs,
        args.out_dir,
        workers=args.workers,
        resume=resume,
        dry_run=args.dry_run,
        max_expl_bb=max_expl,
    )
    print(
        json.dumps(
            {
                "jobs": len(man.jobs),
                "completed": len(man.completed),
                "skipped": len(man.skipped),
                "failed": len(man.failed),
                "rejected": man.rejected,
                "out_dir": man.out_dir,
                "max_expl_bb": max_expl,
            },
            indent=2,
        )
    )
    return 1 if man.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
