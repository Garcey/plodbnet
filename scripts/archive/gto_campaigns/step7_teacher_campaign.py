#!/usr/bin/env python
"""Step 7: 24-root HU river teacher campaign (SPR mix, not overnight_grid).

Kill-safe: per-root progress_file, shared STOP, resume markers.
Floors ON. Cap stays 1.0. Rejected roots are retried at 40k (iters, not cap).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.cfr_api import apply_teacher_iso_policy  # noqa: E402
from plo5bp.gto.cfr_batch import (  # noqa: E402
    MARKER_REJECTED,
    clear_job,
    expand_river_spr_grid,
    marker_status,
    rejected_path,
    resolve_job_ids,
    run_jobs_incremental,
    teacher_split,
)
from plo5bp.gto.jsonio import atomic_write_text  # noqa: E402
from plo5bp.gto.teacher import (  # noqa: E402
    TEACHER_HOLDOUT_FRAC,
    TEACHER_MAX_EXPL_BB,
    TEACHER_MIN_VISIT_MASS,
    TEACHER_SPLIT_SEED,
    root_stratum,
    split_root_ids,
)


def campaign_split(jobs) -> tuple[list[str], list[str]]:
    """The stratified split the export will compute (cfr_batch.teacher_split)."""
    return teacher_split(jobs)


def run_jobs(jobs, out_dir: Path, *, stop_file: Path, threads: int, pass_name: str) -> None:
    """One campaign pass: full iteration budget per root, manifest rewritten
    after every root (cfr_batch.run_jobs_incremental)."""
    for j in jobs:
        j.config.poll_every = 10_000
    run_jobs_incremental(
        jobs, out_dir, stop_file=stop_file, threads=threads, tag=f"step7 {pass_name}"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/cfr/teacher_s7"))
    p.add_argument("--n-boards", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--iters", type=int, default=20_000)
    p.add_argument("--retry-iters", type=int, default=40_000)
    p.add_argument("--pot-bb", type=float, default=10.0)
    p.add_argument("--size-preset", type=str, default="micro")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--no-retry", action="store_true")
    args = p.parse_args()

    out_dir = args.out_dir
    stop_file = out_dir / "STOP"
    jobs = expand_river_spr_grid(
        n_boards=args.n_boards,
        seed=args.seed,
        pot_bb=args.pot_bb,
        size_preset=args.size_preset,
        iters=args.iters,
    )
    # Campaign dirs written under the pre-fingerprint ids keep resuming.
    ids = resolve_job_ids(out_dir, jobs)
    train_ids, hold_ids = campaign_split(jobs)
    print(
        f"[step7] CAMPAIGN n={len(jobs)} spr={{1,2,3,5}}×{args.n_boards} boards "
        f"iters={args.iters} iso=OFF max_expl_bb={TEACHER_MAX_EXPL_BB}",
        flush=True,
    )
    print(f"[step7] split train={len(train_ids)} holdout={len(hold_ids)}", flush=True)
    print(f"[step7] holdout_ids={hold_ids}", flush=True)
    (out_dir).mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        out_dir / "plan.json",
        json.dumps(
            {
                "n_jobs": len(jobs),
                "jobs": ids,
                "train_root_ids": train_ids,
                "holdout_root_ids": hold_ids,
                "max_expl_bb": TEACHER_MAX_EXPL_BB,
                "min_visit_mass": TEACHER_MIN_VISIT_MASS,
                "holdout_frac": TEACHER_HOLDOUT_FRAC,
                "spr_points": [1.0, 2.0, 3.0, 5.0],
                "iters": args.iters,
                "retry_iters": args.retry_iters,
                "size_preset": args.size_preset,
                "stop_file": str(stop_file),
            },
            indent=2,
        )
        + "\n",
    )

    run_jobs(
        jobs,
        out_dir,
        stop_file=stop_file,
        threads=args.threads,
        pass_name=f"pass1_{args.iters}",
    )

    if args.no_retry or stop_file.exists():
        return 0

    retry = []
    for j in jobs:
        rp = rejected_path(out_dir, j.job_id)
        if rp.exists() and marker_status(out_dir, j.job_id) == MARKER_REJECTED:
            retry.append(j)
    if not retry:
        print("[step7] no rejected roots to retry", flush=True)
        return 0

    print(f"[step7] RETRY {len(retry)} rejected roots at {args.retry_iters} iters", flush=True)
    for j in retry:
        clear_job(out_dir, j.job_id)
        j.config.max_iterations = int(args.retry_iters)
    run_jobs(
        retry,
        out_dir,
        stop_file=stop_file,
        threads=args.threads,
        pass_name=f"pass2_{args.retry_iters}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
