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

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.cfr_api import apply_teacher_iso_policy  # noqa: E402
from plo5bp.gto.cfr_batch import (  # noqa: E402
    BatchManifest,
    _marker_path,
    _rejected_path,
    _run_one,
    _strategy_path,
    expand_river_spr_grid,
)
from plo5bp.gto.teacher import (  # noqa: E402
    TEACHER_HOLDOUT_FRAC,
    TEACHER_MAX_EXPL_BB,
    TEACHER_MIN_VISIT_MASS,
    TEACHER_SPLIT_SEED,
    split_root_ids,
)


def _clear_job(out_dir: Path, job_id: str) -> None:
    for fn in (_marker_path, _rejected_path, _strategy_path):
        p = fn(out_dir, job_id)
        if p.exists():
            p.unlink()
    prog = out_dir / "progress" / f"{job_id}.progress.json"
    if prog.exists():
        prog.unlink()


def _write_manifest(out_dir: Path, man: BatchManifest) -> None:
    man.finished_at = time.time()
    man.write(out_dir / "manifest.json")


def run_jobs(
    jobs,
    out_dir: Path,
    *,
    stop_file: Path,
    threads: int,
    pass_name: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "strategies").mkdir(exist_ok=True)
    (out_dir / "markers").mkdir(exist_ok=True)
    (out_dir / "rejected").mkdir(exist_ok=True)
    (out_dir / "progress").mkdir(exist_ok=True)

    man_path = out_dir / "manifest.json"
    if man_path.exists():
        man = BatchManifest.load(man_path)
    else:
        man = BatchManifest(out_dir=str(out_dir), jobs=[j.job_id for j in jobs])
        man.started_at = time.time()

    known = set(man.jobs)
    for j in jobs:
        if j.job_id not in known:
            man.jobs.append(j.job_id)
            known.add(j.job_id)

    for j in jobs:
        if stop_file.exists():
            print(f"[step7] STOP {stop_file} — remaining jobs skipped", flush=True)
            _write_manifest(out_dir, man)
            return
        apply_teacher_iso_policy(j.config)
        j.config.target_exploitability_bb = 0.0
        j.config.thread_num = int(threads)
        j.config.poll_every = 10_000
        j.config.stop_file = str(stop_file)
        j.config.progress_file = str(out_dir / "progress" / f"{j.job_id}.progress.json")

        marker = _marker_path(out_dir, j.job_id)
        if marker.exists():
            text = marker.read_text(encoding="utf-8").strip()
            print(f"[step7] SKIP {j.job_id} marker={text} ({pass_name})", flush=True)
            if j.job_id not in man.skipped:
                man.skipped.append(j.job_id)
            _write_manifest(out_dir, man)
            continue

        print(
            f"[step7] ROOT START {j.job_id} {pass_name} "
            f"spr={j.root.effective_stack_bb / j.root.pot_bb:g} "
            f"stack={j.root.effective_stack_bb:g} board={j.root.board} "
            f"iters={j.config.max_iterations}",
            flush=True,
        )
        t0 = time.time()
        r = _run_one(
            {
                "root": j.root.as_dict(),
                "config": j.config.as_dict(),
                "out_dir": str(out_dir),
                "job_id": j.job_id,
                "max_expl_bb": TEACHER_MAX_EXPL_BB,
            }
        )
        dt = time.time() - t0
        expl = r.get("exploitability_bb")
        status = "REJECTED" if r.get("rejected") else ("OK" if r.get("ok") else "FAILED")
        print(
            f"[step7] ROOT {status} {j.job_id} expl_bb={expl} "
            f"iters={r.get('iterations')} wall_s={dt:.1f} err={r.get('error')}",
            flush=True,
        )
        if r.get("rejected"):
            if not any(x.get("job_id") == j.job_id for x in man.rejected):
                man.rejected.append(
                    {
                        "job_id": r["job_id"],
                        "error": str(r.get("error", "rejected")),
                        "exploitability_bb": expl,
                        "iters": r.get("iterations"),
                    }
                )
        elif r.get("ok"):
            if j.job_id not in man.completed:
                man.completed.append(j.job_id)
            man.rejected = [x for x in man.rejected if x.get("job_id") != j.job_id]
        else:
            man.failed.append(
                {"job_id": r["job_id"], "error": r.get("error", r.get("status", "?"))}
            )
        _write_manifest(out_dir, man)


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
    ids = [j.job_id for j in jobs]
    train_ids, hold_ids = split_root_ids(
        ids, seed=TEACHER_SPLIT_SEED, holdout_frac=TEACHER_HOLDOUT_FRAC
    )
    print(
        f"[step7] CAMPAIGN n={len(jobs)} spr={{1,2,3,5}}×{args.n_boards} boards "
        f"iters={args.iters} iso=OFF max_expl_bb={TEACHER_MAX_EXPL_BB}",
        flush=True,
    )
    print(f"[step7] split train={len(train_ids)} holdout={len(hold_ids)}", flush=True)
    print(f"[step7] holdout_ids={hold_ids}", flush=True)
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "plan.json").write_text(
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
        encoding="utf-8",
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
        rp = _rejected_path(out_dir, j.job_id)
        mk = _marker_path(out_dir, j.job_id)
        if rp.exists() and mk.exists() and mk.read_text(encoding="utf-8").strip() == "rejected":
            retry.append(j)
    if not retry:
        print("[step7] no rejected roots to retry", flush=True)
        return 0

    print(f"[step7] RETRY {len(retry)} rejected roots at {args.retry_iters} iters", flush=True)
    for j in retry:
        _clear_job(out_dir, j.job_id)
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
