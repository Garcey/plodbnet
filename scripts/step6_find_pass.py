#!/usr/bin/env python
"""Find HU river boards that can pass expl_bb <= 1.0. Does not change the cap.

(review 2026-09-20 D8) Runs every candidate to its iteration cap with NO
exploitability target: ``target == cap`` stopped at the first noisy 24-deal
POLL dip under 1.0 bb and reported that as a pass. PASS now needs a number
from a FINAL estimator (``expl_provenance(...).verified``) under the cap.
"""

from __future__ import annotations

import time
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.cfr_api import RootSpec, SolveConfig, apply_teacher_iso_policy, solve
from plo5bp.gto.teacher import TEACHER_MAX_EXPL_BB, expl_provenance, expl_reject_reason

CANDIDATES = [
    ("99TTT", [30, 31, 32, 33, 34]),
    ("KKK23", [44, 45, 46, 0, 5]),
    ("quad3s", [8, 9, 10, 11, 16]),
    ("AAA45", [48, 49, 50, 12, 16]),
    ("pair22", [0, 1, 20, 36, 44]),
    ("flushc", [0, 4, 8, 16, 24]),
    ("dry", [2, 17, 26, 35, 50]),
]


def main() -> None:
    for name, board in CANDIDATES:
        root = RootSpec(
            street=3,
            pot_bb=10.0,
            effective_stack_bb=20.0,
            board=board,
            raise_sizes_pm=[500, 1000],
            allin_atom=True,
            root_id=f"pass_{name}",
        )
        cfg = SolveConfig.teacher(
            max_iterations=10000,
            seed=3,
            target_exploitability_bb=0.0,  # no early stop on the poll estimate
            thread_num=4,
            poll_every=500,
            card_abstraction="none",
        )
        apply_teacher_iso_policy(cfg)
        print(f"[step6] TRY {name} {board}", flush=True)
        t0 = time.time()
        rep = solve(root, cfg)
        prov = expl_provenance(rep)
        hit = prov.verified and (
            expl_reject_reason(rep.exploitability_bb, max_expl_bb=TEACHER_MAX_EXPL_BB)
            is None
        )
        print(
            f"[step6] TRY {name} expl_bb={rep.exploitability_bb} "
            f"kind={prov.kind} verified={prov.verified} "
            f"iters={rep.iterations_run} wall_s={time.time()-t0:.1f} "
            f"PASS={hit}",
            flush=True,
        )


if __name__ == "__main__":
    main()
