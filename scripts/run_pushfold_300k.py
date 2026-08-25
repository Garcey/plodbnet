"""4-handed push/fold solve at 300k iterations; write labeled strategy JSON."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from plo5bp.gto.cfr_api import RootSpec, SolveConfig, solve, rust_cfr_available


def main() -> int:
    print(f"rust_cfr_available={rust_cfr_available()}", flush=True)
    root = RootSpec(
        street=0,
        pot_bb=1.5,
        effective_stack_bb=10.0,
        board=[],
        num_seats=4,
        bb_chips=10_000,
        sb_chips=5_000,
        ante_chips=0,
        raise_sizes_pm=[],
        allin_atom=True,
        stacks_bb=[10.0] * 4,
        root_id="pf4_pushfold_10bb_5_10_noante_norake_300k",
    )
    cfg = SolveConfig(
        max_iterations=300_000,
        seed=42,
        algorithm="mccfr_es",
        target_exploitability_bb=0.1,
        thread_num=1,
    )
    print("solving 4-handed push/fold 300k iters...", flush=True)
    print(root.as_dict(), flush=True)
    t0 = time.time()
    report = solve(root, cfg)
    dt = time.time() - t0
    out = Path("data/cfr/pushfold_4handed_10bb_300k.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(out)
    n_info = len(report.strategy.get("infosets") or [])
    print(
        f"status={report.status} iters={report.iterations_run} "
        f"expl={report.exploitability_bb} infosets={n_info} wall_s={dt:.1f}",
        flush=True,
    )
    print(f"notes={report.notes}", flush=True)
    if n_info:
        print(f"sample_id={report.strategy['infosets'][0]['infoset_id']}", flush=True)
    print(f"wrote {out}", flush=True)
    return 0 if report.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
