#!/usr/bin/env python
"""After-fix expl on the same Step 6 roots (same seed / board / iters)."""

from __future__ import annotations

import time
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.cfr_api import RootSpec, SolveConfig, apply_teacher_iso_policy, solve


def run(name: str, root: RootSpec, iters: int, seed: int) -> None:
    cfg = SolveConfig.teacher(
        max_iterations=iters,
        seed=seed,
        target_exploitability_bb=0.0,
        thread_num=4,
        poll_every=10_000,
        card_abstraction="none",
    )
    apply_teacher_iso_policy(cfg)
    print(f"[expl] START {name} iters={iters} seed={seed} board={root.board}", flush=True)
    t0 = time.time()
    rep = solve(root, cfg)
    print(
        f"[expl] {name} expl_bb={rep.exploitability_bb} "
        f"iters={rep.iterations_run} wall_s={time.time()-t0:.1f} "
        f"notes={rep.notes[-4:]}",
        flush=True,
    )


def main() -> None:
    # Step 6 s3_s3_i2 — deal-BR was 4.098 at 20k
    r = RootSpec(
        street=3,
        pot_bb=10.0,
        effective_stack_bb=20.0,
        board=[1, 5, 22, 45, 51],
        raise_sizes_pm=[500, 1000],
        allin_atom=True,
        root_id="s3_s3_i2",
    )
    run("s3_s3_i2", r, 20_000, 5)

    # Quads-on-board full range — deal-BR was 4.82 at 10k
    q = RootSpec(
        street=3,
        pot_bb=10.0,
        effective_stack_bb=20.0,
        board=[8, 9, 10, 11, 16],
        raise_sizes_pm=[500, 1000],
        allin_atom=True,
        root_id="quads",
    )
    run("quads_full", q, 10_000, 3)

    # Tiny 4x4 jam/check (the new unit-test tree)
    t = RootSpec(
        street=3,
        pot_bb=10.0,
        effective_stack_bb=10.0,
        board=[0, 5, 10, 15, 20],
        raise_sizes_pm=[],
        allin_atom=True,
        range_oop="2:1,9:1,27:1,44:1",
        range_ip="77:1,104:1,152:1,189:1",
        root_id="tiny4x4",
    )
    run("tiny4x4", t, 3_000, 1)


if __name__ == "__main__":
    main()
