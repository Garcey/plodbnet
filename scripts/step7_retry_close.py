#!/usr/bin/env python
"""Retry only close-to-cap rejected roots at 40k (holdout first). Cap stays 1.0.

(review 2026-09-20 F10) A STOP file is the operator's halt request for the
whole campaign; this script used to delete it unconditionally and start
solving. It now refuses to run while STOP exists unless ``--clear-stop`` says
the operator means it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.cfr_batch import expand_river_spr_grid, resolve_job_ids  # noqa: E402

# Import campaign helpers
sys.path.insert(0, str(_ROOT / "scripts"))
from step7_teacher_campaign import _clear_job, campaign_split, run_jobs  # noqa: E402

CLOSE_MAX = 1.35  # 40k may land these; skip 1.5+ this session


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--clear-stop",
        action="store_true",
        help="Remove an existing STOP file and run (default: refuse while it exists)",
    )
    args = p.parse_args()

    out = Path("data/cfr/teacher_s7")
    stop = out / "STOP"
    if stop.exists():
        if not args.clear_stop:
            print(
                f"[step7] {stop} exists — the campaign was told to halt. "
                f"Remove it (or pass --clear-stop) to retry.",
                file=sys.stderr,
            )
            return 2
        stop.unlink()
        print(f"[step7] removed {stop} (--clear-stop)", flush=True)
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    jobs = expand_river_spr_grid(n_boards=6, seed=7, iters=40_000)
    resolve_job_ids(out, jobs)  # pre-fingerprint campaign dirs keep their ids
    by_id = {j.job_id: j for j in jobs}
    _tr, ho = campaign_split(jobs)
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
