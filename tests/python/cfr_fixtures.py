"""Small CFR artifacts for the CFR-app tests (TEST-037).

The tests used to read files from the git-ignored `data/cfr/` that existed on one
machine only, so everywhere else they skipped. These builders make equivalent
files in about a second with the REAL solver and the REAL chart exporter
(`scripts/export_pushfold_14_charts.py`) — same kinds, same shapes, same names:

- a heads-up river solve report  (kind "solve_report", street 3, hundreds of infosets)
- the 14-chart push/fold pack     (00_CO_open.json … + INDEX.json, 169 classes a chart)
"""
from __future__ import annotations

import contextlib
import io
import runpy
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def build_river_report(out_dir: Path) -> Path:
    from plo5bp.gto.cfr_api import RootSpec, SolveConfig, solve

    root = RootSpec(street=3, pot_bb=10.0, effective_stack_bb=20.0, board=[0, 5, 10, 15, 20],
                    raise_sizes_pm=[500, 1000], root_id="river_fixture")
    rep = solve(root, SolveConfig(max_iterations=40, seed=5, target_exploitability_bb=0.0))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "s3_s0_i0.json"
    rep.write_json(path)
    return path


def build_pushfold_pack(out_dir: Path) -> Path:
    """The 4-handed 10 bb push/fold solve of scripts/archive/gto_campaigns/solve_pushfold_4handed.py with
    20k iterations (enough for every hand class to reach every node), exported into
    <out_dir>/pushfold_14_charts/. Returns that folder."""
    from plo5bp.gto.cfr_api import RootSpec, SolveConfig, solve

    root = RootSpec(street=0, pot_bb=1.5, effective_stack_bb=10.0, board=[], num_seats=4, bb_chips=10_000,
                    sb_chips=5_000, ante_chips=0, raise_sizes_pm=[], allin_atom=True, stacks_bb=[10.0] * 4,
                    root_id="pf4_pushfold_fixture")
    rep = solve(root, SolveConfig(max_iterations=20_000, target_exploitability_bb=0.0, thread_num=1, seed=42,
                                  algorithm="mccfr_es", card_abstraction="none"))
    out_dir.mkdir(parents=True, exist_ok=True)
    solved = out_dir / "pushfold_4handed_10bb_fixture.json"
    rep.write_json(solved)
    charts = out_dir / "pushfold_14_charts"
    exporter = runpy.run_path(str(REPO / "scripts" / "export_pushfold_14_charts.py"), run_name="cfr_fixtures")
    import sys

    argv = sys.argv
    sys.argv = ["export_pushfold_14_charts.py", str(solved), str(charts)]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exporter["main"]()
    finally:
        sys.argv = argv
    return charts
