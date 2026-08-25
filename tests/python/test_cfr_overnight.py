"""Overnight runner wiring — no multi-hour solves."""

from __future__ import annotations

import json
from pathlib import Path

from plo5bp.gto.cfr_overnight import (
    blueprint_root_and_config,
    promote_progress_if_any,
    run_overnight,
)


def test_blueprint_config_is_kill_safe():
    job = {
        "job_id": "bp_test",
        "stack_bb": 100.0,
        "num_seats": 2,
        "raise_sizes_pm": [330, 500, 1000, 1500],
        "allin_atom": True,
        "max_iterations": 99,
        "time_budget_secs": 12,
        "algorithm": "mccfr_es",
        "seed": 1,
    }
    root, cfg = blueprint_root_and_config(
        job, stop_file="data/cfr/overnight/STOP", progress_file="x.progress.json"
    )
    assert root.street == 0
    assert cfg.progress_file == "x.progress.json"
    assert cfg.poll_every == 250
    assert cfg.time_budget_secs == 12
    assert cfg.stop_file.endswith("STOP")
    assert cfg.use_isomorphism is False


def test_promote_progress_on_resume(tmp_path: Path):
    out = tmp_path / "overnight"
    (out / "strategies").mkdir(parents=True)
    (out / "markers").mkdir()
    job_id = "bp_hu100_coarse"
    job = {
        "job_id": job_id,
        "stack_bb": 100.0,
        "num_seats": 2,
        "raise_sizes_pm": [500, 1000],
        "allin_atom": True,
        "seed": 0,
    }
    prog = out / "strategies" / f"{job_id}.progress.json"
    prog.write_text(
        json.dumps(
            {
                "status": "running",
                "iterations_run": 777,
                "strategy": {
                    "root_id": job_id,
                    "schema_version": 2,
                    "infosets": [
                        {
                            "infoset_id": "pf_p1_h0_c12",
                            "actions": ["FOLD", "RAISE_500"],
                            "probs": [0.1, 0.9],
                            "schema_version": 2,
                            "private_kind": "class",
                            "to_call_chips": 10000,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    got = promote_progress_if_any(out, job_id, job)
    assert got is not None
    assert got["promoted_from_progress"] is True
    assert got["iterations"] == 777
    strat = out / "strategies" / f"{job_id}.json"
    assert strat.is_file()
    assert not prog.exists()
    assert (out / "markers" / f"{job_id}.done").is_file()


def test_overnight_dry_run(tmp_path: Path):
    grid = {
        "grid_id": "dry",
        "out_dir": str(tmp_path / "out"),
        "stop_file": str(tmp_path / "STOP"),
        "jobs": [
            {
                "job_id": "bp_a",
                "kind": "preflop_blueprint",
                "stack_bb": 40.0,
                "num_seats": 2,
            }
        ],
    }
    gpath = tmp_path / "grid.json"
    gpath.write_text(json.dumps(grid), encoding="utf-8")
    report = run_overnight(gpath, dry_run=True)
    assert report.results == []
    assert report.seconds == 0.0
