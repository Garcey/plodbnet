#!/usr/bin/env python
from __future__ import annotations

import time
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.cfr_api import RootSpec, SolveConfig, apply_teacher_iso_policy, solve
from plo5bp.gto.teacher import TEACHER_MAX_EXPL_BB


def main() -> None:
    boards = [
        [5, 7, 30, 44, 47],
        [3, 13, 19, 28, 29],
        [1, 5, 22, 45, 51],
        [6, 7, 27, 33, 41],
    ]
    for i, board in enumerate(boards):
        root = RootSpec(
            street=3,
            pot_bb=10.0,
            effective_stack_bb=20.0,
            board=board,
            raise_sizes_pm=[],
            allin_atom=True,
            root_id=f"jamcheck_s3_s3_i{i}",
        )
        cfg = SolveConfig.teacher(
            max_iterations=20000,
            seed=3 + i,
            target_exploitability_bb=TEACHER_MAX_EXPL_BB,
            thread_num=4,
            poll_every=500,
            card_abstraction="none",
        )
        apply_teacher_iso_policy(cfg)
        print(f"[step6] JAMCHECK START {root.root_id} board={board}", flush=True)
        t0 = time.time()
        rep = solve(root, cfg)
        print(
            f"[step6] JAMCHECK {root.root_id} expl_bb={rep.exploitability_bb} "
            f"iters={rep.iterations_run} wall_s={time.time()-t0:.1f} "
            f"infosets={len(rep.strategy.get('infosets') or [])}",
            flush=True,
        )


if __name__ == "__main__":
    main()
