#!/usr/bin/env python
"""Raise iters on one already-rejected Step 6 root. Floor stays 1.0 bb.

(review 2026-09-20 D8) The solve runs to its iteration cap with NO
exploitability target. The old ``target == cap`` made the solver stop at the
first 24-deal POLL estimate that dipped under 1.0 bb — a noisy, upward-biased
number (5.27 polled vs 1.99 final) — and the root was accepted on it. Acceptance
now goes through the batch gate: only a FINAL-estimator number is judged
against the cap; anything else lands in ``unverified/``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.cfr_api import RootSpec, SolveConfig, apply_teacher_iso_policy  # noqa: E402
from plo5bp.gto.cfr_batch import BatchJob, _run_one, job_payload  # noqa: E402
from plo5bp.gto.teacher import TEACHER_MAX_EXPL_BB  # noqa: E402


def main() -> int:
    out = Path("data/cfr/step6_teacher")
    job_id = "s3_s3_i1"
    rej = json.loads((out / "rejected" / f"{job_id}.json").read_text(encoding="utf-8"))
    root = RootSpec(**rej["report"]["root"])
    progress = out / "progress_i1.json"
    cfg = SolveConfig.teacher(
        max_iterations=80000,
        seed=4,
        # 0 = no early stop: the reported number is the final estimator's.
        target_exploitability_bb=0.0,
        thread_num=4,
        poll_every=2000,
        card_abstraction="none",
        progress_file=str(progress),
    )
    apply_teacher_iso_policy(cfg)
    print(
        f"[step6] RAISE ITERS {job_id} cap=80000 floor={TEACHER_MAX_EXPL_BB} "
        f"prev_expl={rej['exploitability_bb']}",
        flush=True,
    )
    t0 = time.time()
    r = _run_one(
        job_payload(
            BatchJob(root=root, config=cfg, job_id=job_id), out, TEACHER_MAX_EXPL_BB
        )
    )
    dt = time.time() - t0
    print(
        f"[step6] RAISE DONE {job_id} status={r.get('status')} "
        f"expl_bb={r.get('exploitability_bb')} iters={r.get('iterations')} "
        f"wall_s={dt:.1f} expl={r.get('expl')} err={r.get('error')}",
        flush=True,
    )
    if r.get("ok"):
        print(f"[step6] ACCEPTED {job_id} -> {out / 'strategies' / (job_id + '.json')}", flush=True)
        return 0
    verdict = "UNVERIFIED" if r.get("unverified") else "STILL REJECTED"
    print(f"[step6] {verdict} {job_id} {r.get('error')}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
