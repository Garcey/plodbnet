#!/usr/bin/env python
"""Retry only close-to-cap rejected roots at 40k (holdout first). Cap stays 1.0."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.cfr_batch import expand_river_spr_grid  # noqa: E402
from plo5bp.gto.teacher import TEACHER_HOLDOUT_FRAC, TEACHER_SPLIT_SEED, split_root_ids  # noqa: E402

# Import campaign helpers
sys.path.insert(0, str(_ROOT / "scripts"))
from step7_teacher_campaign import _clear_job, run_jobs  # noqa: E402

CLOSE_MAX = 1.35  # 40k may land these; skip 1.5+ this session


def main() -> int:
    out = Path("data/cfr/teacher_s7")
    stop = out / "STOP"
    if stop.exists():
        stop.unlink()
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    jobs = expand_river_spr_grid(n_boards=6, seed=7, iters=40_000)
    by_id = {j.job_id: j for j in jobs}
    tr, ho = split_root_ids(
        [j.job_id for j in jobs],
        seed=TEACHER_SPLIT_SEED,
        holdout_frac=TEACHER_HOLDOUT_FRAC,
    )
    hold = set(ho)
    retry_ids = []
    truncated = []
    for row in man.get("rejected") or []:
        rid = row["job_id"]
        expl = row.get("exploitability_bb")
        iters = int(row.get("iters") or 0)
        if rid in man.get("completed") or []:
            continue
        if iters and iters < 15_000:
            truncated.append(rid)
            continue
        if rid in hold:
            retry_ids.append(rid)
            continue
        if expl is not None and float(expl) <= CLOSE_MAX:
            retry_ids.append(rid)
    if truncated:
        print(f"[step7] re-solve truncated at 20k first: {truncated}", flush=True)
        tjobs = []
        for rid in truncated:
            j = by_id[rid]
            j.config.max_iterations = 20_000
            _clear_job(out, rid)
            tjobs.append(j)
        run_jobs(tjobs, out, stop_file=stop, threads=4, pass_name="pass1b_20k_truncated")
        man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    # holdout first
    retry_ids = [r for r in retry_ids if r in hold] + [r for r in retry_ids if r not in hold]
    print(f"[step7] targeted retry {retry_ids} at 40k (close<={CLOSE_MAX} + holdout)", flush=True)
    retry_jobs = []
    for rid in retry_ids:
        j = by_id[rid]
        j.config.max_iterations = 40_000
        _clear_job(out, rid)
        retry_jobs.append(j)
    if not retry_jobs:
        print("[step7] nothing to retry", flush=True)
        return 0
    run_jobs(retry_jobs, out, stop_file=stop, threads=4, pass_name="pass2_40k_close")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
