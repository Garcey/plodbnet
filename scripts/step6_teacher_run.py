#!/usr/bin/env python
"""Step 6: small HU-river teacher batch -> export -> train -> probe.

Floors stay ON. Grid seed 3 so the 15% SHA-256 split has 1 holdout root.
Micro sizes so DCFR can actually land expl_bb <= 1.0 (coarse 200-iter was ~18 bb).
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
from plo5bp.gto.cfr_batch import BatchManifest, _run_one, expand_river_grid  # noqa: E402
from plo5bp.gto.teacher import (  # noqa: E402
    TEACHER_HOLDOUT_FRAC,
    TEACHER_MAX_EXPL_BB,
    TEACHER_MIN_VISIT_MASS,
    TEACHER_SPLIT_SEED,
    split_root_ids,
)


def run_batch_printed(
    *,
    out_dir: Path,
    n_roots: int,
    seed: int,
    iters: int,
    stack_bb: float,
    pot_bb: float,
    size_preset: str,
    threads: int,
) -> BatchManifest:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "strategies").mkdir(exist_ok=True)
    (out_dir / "markers").mkdir(exist_ok=True)
    (out_dir / "rejected").mkdir(exist_ok=True)

    jobs = expand_river_grid(
        n_roots=n_roots,
        seed=seed,
        pot_bb=pot_bb,
        stack_bb=stack_bb,
        size_preset=size_preset,
        iters=iters,
        streets=[3],
    )
    ids = [j.job_id for j in jobs]
    train_ids, hold_ids = split_root_ids(
        ids, seed=TEACHER_SPLIT_SEED, holdout_frac=TEACHER_HOLDOUT_FRAC
    )
    print(
        f"[step6] BATCH START n={len(jobs)} seed={seed} iters_cap={iters} "
        f"stack_bb={stack_bb} pot_bb={pot_bb} sizes={size_preset} "
        f"threads={threads} iso=OFF max_expl_bb={TEACHER_MAX_EXPL_BB}",
        flush=True,
    )
    print(f"[step6] split train={train_ids} holdout={hold_ids}", flush=True)

    for j in jobs:
        apply_teacher_iso_policy(j.config)
        # Run the full iter cap so exploitability_bb is the final infoset-BR
        # report (hero-enum), not the 24-deal poll used for early-stop.
        j.config.target_exploitability_bb = 0.0
        j.config.thread_num = int(threads)
        j.config.poll_every = 10_000

    man = BatchManifest(out_dir=str(out_dir), jobs=ids, started_at=time.time())
    (out_dir / "plan.json").write_text(
        json.dumps(
            {
                "n_jobs": len(jobs),
                "jobs": ids,
                "train_root_ids": train_ids,
                "holdout_root_ids": hold_ids,
                "max_expl_bb": TEACHER_MAX_EXPL_BB,
                "target_exploitability_bb": TEACHER_MAX_EXPL_BB,
                "min_visit_mass": TEACHER_MIN_VISIT_MASS,
                "holdout_frac": TEACHER_HOLDOUT_FRAC,
                "size_preset": size_preset,
                "stack_bb": stack_bb,
                "iters_cap": iters,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for j in jobs:
        print(f"[step6] ROOT START {j.job_id} board={j.root.board}", flush=True)
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
            f"[step6] ROOT {status} {j.job_id} expl_bb={expl} "
            f"iters={r.get('iterations')} wall_s={dt:.1f} "
            f"err={r.get('error')}",
            flush=True,
        )
        if r.get("rejected"):
            man.rejected.append(
                {"job_id": r["job_id"], "error": str(r.get("error", "rejected"))}
            )
        elif r.get("ok"):
            man.completed.append(r["job_id"])
        else:
            man.failed.append(
                {"job_id": r["job_id"], "error": r.get("error", r.get("status", "?"))}
            )

    man.finished_at = time.time()
    man.write(out_dir / "manifest.json")
    print(
        f"[step6] BATCH DONE completed={len(man.completed)} "
        f"rejected={len(man.rejected)} failed={len(man.failed)}",
        flush=True,
    )
    return man


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/cfr/teacher_ibr"))
    p.add_argument("--n-roots", type=int, default=4)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--stack-bb", type=float, default=20.0)
    p.add_argument("--pot-bb", type=float, default=10.0)
    p.add_argument("--size-preset", type=str, default="micro")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    man = run_batch_printed(
        out_dir=args.out_dir,
        n_roots=args.n_roots,
        seed=args.seed,
        iters=args.iters,
        stack_bb=args.stack_bb,
        pot_bb=args.pot_bb,
        size_preset=args.size_preset,
        threads=args.threads,
    )
    return 1 if man.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
