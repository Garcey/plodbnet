#!/usr/bin/env python
"""Rich end-to-end correctness certificate for the native CFR solver.

Writes metrics to data/cfr/verify/ (project-local, skeptic-readable).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.cfr_api import (  # noqa: E402
    RootSpec,
    SolveConfig,
    rust_cfr_available,
    solve,
    solve_kuhn,
)
from plo5bp.gto.cfr_batch import expand_river_grid, run_batch  # noqa: E402
from plo5bp.gto.cfr_export import export_dir  # noqa: E402


OUT = _ROOT / "data" / "cfr" / "verify"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"ts": time.time(), "checks": {}}

    if not rust_cfr_available():
        print("FAIL: rust_cfr not available — maturin develop --release")
        return 1

    # 1) Kuhn NE
    k = solve_kuhn(15000)
    k_ok = abs(k["value_p0"] - k["nash_value"]) < 0.02 and abs(k["exploitability"]) < 0.01
    report["checks"]["kuhn"] = {
        "ok": k_ok,
        "value_p0": k["value_p0"],
        "nash": k["nash_value"],
        "exploitability": k["exploitability"],
    }

    # 2) HU river
    r = RootSpec.river_hu([0, 5, 10, 15, 20], pot_bb=10, effective_stack_bb=20, size_preset="micro")
    river = solve(
        r,
        SolveConfig(
            max_iterations=150,
            seed=2,
            use_isomorphism=True,
            thread_num=2,
            target_exploitability_bb=0.0,
        ),
    )
    river_ok = (
        river.status == "ok"
        and river.exploitability_bb is not None
        and river.exploitability_bb < 30.0
        and len(river.strategy["infosets"]) > 0
        and any("isomorphism=on" in n for n in river.notes)
        and any("rayon" in n for n in river.notes)
    )
    report["checks"]["river_hu"] = {
        "ok": river_ok,
        "status": river.status,
        "expl_bb": river.exploitability_bb,
        "infosets": len(river.strategy["infosets"]),
        "notes": river.notes,
    }

    # 3) Flop OCHS@200
    flop = RootSpec(
        street=1, pot_bb=8, effective_stack_bb=20, board=[0, 5, 10], raise_sizes_pm=[500, 1000]
    )
    flop_rep = solve(
        flop,
        SolveConfig(max_iterations=40, card_abstraction="ochs", target_exploitability_bb=0.0, seed=4),
    )
    flop_ok = flop_rep.status == "ok" and any(
        "ochs" in n or "bucket" in n for n in flop_rep.notes
    )
    report["checks"]["flop_ochs"] = {
        "ok": flop_ok,
        "status": flop_rep.status,
        "expl_bb": flop_rep.exploitability_bb,
        "notes": flop_rep.notes,
    }

    # 4) Multiway postflop unequal
    mw = RootSpec(
        street=3,
        pot_bb=12,
        effective_stack_bb=25,
        board=[0, 5, 10, 15, 20],
        num_seats=3,
        raise_sizes_pm=[500, 1000],
        stacks_bb=[40.0, 20.0, 10.0],
        root_id="mw3_uneq_v",
    )
    mw_rep = solve(mw, SolveConfig(max_iterations=50, algorithm="mccfr_es", seed=5))
    mw_ok = (
        mw_rep.status == "ok"
        and mw_rep.exploitability_bb is not None
        and any("sidepot" in n for n in mw_rep.notes)
        and any("unequal" in n for n in mw_rep.notes)
    )
    report["checks"]["multiway_postflop"] = {
        "ok": mw_ok,
        "status": mw_rep.status,
        "expl_bb": mw_rep.exploitability_bb,
        "notes": mw_rep.notes,
    }

    # 5) Multiway preflop
    mwp = RootSpec.preflop_hu(40.0)
    mwp.num_seats = 3
    mwp.raise_sizes_pm = [500, 1000]
    mwp_rep = solve(mwp, SolveConfig(max_iterations=60, algorithm="mccfr_es", seed=6))
    mwp_ok = mwp_rep.status == "ok" and any("multiway preflop" in n for n in mwp_rep.notes)
    report["checks"]["multiway_preflop"] = {
        "ok": mwp_ok,
        "status": mwp_rep.status,
        "expl_bb": mwp_rep.exploitability_bb,
        "infosets": len(mwp_rep.strategy["infosets"]),
        "notes": mwp_rep.notes,
    }

    # 6) Preflop HU
    pf = solve(RootSpec.preflop_hu(50.0), SolveConfig(max_iterations=100, algorithm="mccfr_es", seed=7))
    pf_ok = pf.status == "ok" and pf.exploitability_bb is not None and len(pf.strategy["infosets"]) > 0
    report["checks"]["preflop_hu"] = {
        "ok": pf_ok,
        "status": pf.status,
        "expl_bb": pf.exploitability_bb,
        "infosets": len(pf.strategy["infosets"]),
    }

    # 7) Batch + export
    jobs = expand_river_grid(n_roots=2, seed=0, iters=30, size_preset="micro", streets=[3])
    batch_dir = OUT / "batch"
    man = run_batch(jobs, batch_dir, dry_run=False, resume=False, max_expl_bb=None)
    labels = OUT / "labels.jsonl"
    n_lab = export_dir(batch_dir / "strategies", labels, source="rust_cfr_river")
    batch_ok = len(man.completed) == 2 and n_lab > 0
    report["checks"]["batch_export"] = {
        "ok": batch_ok,
        "completed": len(man.completed),
        "labels": n_lab,
    }

    # 8) Pipeline
    from plo5bp import _engine  # type: ignore

    pipe = dict(
        _engine.cfr_pipeline(
            stack_bb=50.0,
            preflop_iters=60,
            postflop_iters=30,
            postflop_board=[0, 5, 10, 15, 20],
            pot_bb=12.0,
            postflop_stack_bb=30.0,
            seed=1,
        )
    )
    pipe_ok = pipe["preflop_status"] == "ok" and pipe["postflop_status"] == "ok"
    report["checks"]["pipeline"] = {"ok": pipe_ok, **{k: pipe[k] for k in pipe if k != "notes"}}

    all_ok = all(c.get("ok") for c in report["checks"].values())
    report["all_ok"] = all_ok

    (OUT / "certificate.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"all_ok={all_ok}",
        f"kuhn_expl={k['exploitability']:.6f}",
        f"river_expl={river.exploitability_bb}",
        f"flop_status={flop_rep.status}",
        f"mw_post_expl={mw_rep.exploitability_bb}",
        f"mw_pre_status={mwp_rep.status} expl={mwp_rep.exploitability_bb}",
        f"preflop_expl={pf.exploitability_bb}",
        f"batch_labels={n_lab}",
        f"pipeline_ok={pipe_ok}",
    ]
    (OUT / "VERIFY_OK.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("---")
    print("\n".join(lines))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
