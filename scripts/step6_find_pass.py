#!/usr/bin/env python
"""Find HU river boards that can pass expl_bb <= 1.0. Does not change the cap."""

from __future__ import annotations

import time
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.cfr_api import RootSpec, SolveConfig, apply_teacher_iso_policy, solve
from plo5bp.gto.teacher import TEACHER_MAX_EXPL_BB

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
            target_exploitability_bb=TEACHER_MAX_EXPL_BB,
            thread_num=4,
            poll_every=500,
            card_abstraction="none",
        )
        apply_teacher_iso_policy(cfg)
        print(f"[step6] TRY {name} {board}", flush=True)
        t0 = time.time()
        rep = solve(root, cfg)
        hit = (
            rep.exploitability_bb is not None
            and rep.exploitability_bb <= TEACHER_MAX_EXPL_BB
        )
        print(
            f"[step6] TRY {name} expl_bb={rep.exploitability_bb} "
            f"iters={rep.iterations_run} wall_s={time.time()-t0:.1f} "
            f"PASS={hit}",
            flush=True,
        )


if __name__ == "__main__":
    main()
