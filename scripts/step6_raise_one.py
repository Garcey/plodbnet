#!/usr/bin/env python
"""Raise iters on one already-rejected Step 6 root. Floor stays 1.0 bb."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.cfr_api import RootSpec, SolveConfig, apply_teacher_iso_policy, solve  # noqa: E402
from plo5bp.gto.teacher import TEACHER_MAX_EXPL_BB, expl_reject_reason  # noqa: E402


def main() -> int:
    out = Path("data/cfr/step6_teacher")
    job_id = "s3_s3_i1"
    rej = json.loads((out / "rejected" / f"{job_id}.json").read_text(encoding="utf-8"))
    root = RootSpec(**rej["report"]["root"])
    progress = out / "progress_i1.json"
    cfg = SolveConfig.teacher(
        max_iterations=80000,
        seed=4,
        target_exploitability_bb=TEACHER_MAX_EXPL_BB,
        thread_num=4,
        poll_every=2000,
        card_abstraction="none",
        progress_file=str(progress),
    )
    apply_teacher_iso_policy(cfg)
    print(
        f"[step6] RAISE ITERS {job_id} cap=80000 target={TEACHER_MAX_EXPL_BB} "
        f"prev_expl={rej['exploitability_bb']}",
        flush=True,
    )
    t0 = time.time()
    rep = solve(root, cfg)
    dt = time.time() - t0
    why = expl_reject_reason(rep.exploitability_bb, max_expl_bb=TEACHER_MAX_EXPL_BB)
    print(
        f"[step6] RAISE DONE {job_id} expl_bb={rep.exploitability_bb} "
        f"iters={rep.iterations_run} wall_s={dt:.1f} reject={why} "
        f"notes={rep.notes[-4:]}",
        flush=True,
    )

    marker = out / "markers" / f"{job_id}.done"
    if why is None:
        path = out / "strategies" / f"{job_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rep.as_dict(), indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        rej_path = out / "rejected" / f"{job_id}.json"
        if rej_path.exists():
            rej_path.unlink()
        marker.write_text("ok\n", encoding="utf-8")
        print(f"[step6] ACCEPTED {job_id} -> {path}", flush=True)
        return 0

    rpath = out / "rejected" / f"{job_id}.json"
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "reason": why,
                "status": "rejected",
                "exploitability_bb": rep.exploitability_bb,
                "max_expl_bb": TEACHER_MAX_EXPL_BB,
                "report": rep.as_dict(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    marker.write_text("rejected\n", encoding="utf-8")
    print(f"[step6] STILL REJECTED {job_id} {why}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
