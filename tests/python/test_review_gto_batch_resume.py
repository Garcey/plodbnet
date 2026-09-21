"""Review 2026-09-20 D13 / F10 — overnight resume + batch identity hazards.

D13 a thin progress snapshot is never promoted; a substantial one is kept as
    ``partial`` (never ``ok``, never exported).
F10 new root / job ids carry a game fingerprint while campaign dirs written
    under the old bare ids still resume; a changed solve config is not
    "resumed"; ``_run_pipeline_board`` honours ``resume=False``; manifests are
    written atomically; ``step7_retry_close`` leaves the operator's STOP alone.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

import plo5bp.gto.cfr_batch as cfr_batch
import plo5bp.gto.cfr_overnight as overnight
from plo5bp.gto.cfr_api import RootSpec, SolveReport
from plo5bp.gto.cfr_batch import (
    BatchManifest,
    expand_river_grid,
    expand_river_spr_grid,
    marker_status,
    resolve_job_id,
    run_batch,
)
from plo5bp.gto.cfr_export import export_dir_detailed
from plo5bp.gto.teacher import root_fingerprint

REPO = Path(__file__).resolve().parents[2]
FINAL = "expl_kind=infoset_br samples=128"


def _fake_solve(calls: list, expl: float = 0.3):
    def solve(root, cfg):
        calls.append((root.root_id, cfg.max_iterations))
        return SolveReport(
            status="ok", root=root.as_dict(), config=cfg.as_dict(),
            strategy={"root_id": root.root_id, "infosets": []},
            iterations_run=cfg.max_iterations, exploitability_bb=expl, notes=[FINAL],
        )

    return solve


# --- D13 ------------------------------------------------------------------------

JOB = {
    "job_id": "bp_hu100",
    "kind": "preflop_blueprint",
    "stack_bb": 100.0,
    "num_seats": 2,
    "raise_sizes_pm": [500, 1000],
    "time_budget_secs": 3600,
}


def _write_progress(out: Path, job_id: str, iters: int) -> Path:
    (out / "strategies").mkdir(parents=True, exist_ok=True)
    (out / "markers").mkdir(parents=True, exist_ok=True)
    prog = out / "strategies" / f"{job_id}.progress.json"
    prog.write_text(
        json.dumps(
            {
                "status": "running",
                "iterations_run": iters,
                "exploitability_bb": 7.5,
                "strategy": {
                    "root_id": job_id,
                    "schema_version": 2,
                    "infosets": [
                        {"infoset_id": "pf_p1_h0_c12", "actions": ["FOLD", "RAISE_500"],
                         "probs": [0.1, 0.9], "schema_version": 2,
                         "private_kind": "class", "to_call_chips": 10000}
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return prog


def test_iteration_one_snapshot_is_not_promoted(tmp_path: Path):
    """The Rust solver writes a snapshot at iteration 1 — never a finished job."""
    prog = _write_progress(tmp_path, JOB["job_id"], 1)
    assert overnight.promote_progress_if_any(tmp_path, JOB["job_id"], JOB) is None
    assert prog.exists()
    assert not (tmp_path / "markers" / f"{JOB['job_id']}.done").exists()
    assert not (tmp_path / "strategies" / f"{JOB['job_id']}.json").exists()


@pytest.mark.parametrize(
    "iters,job_over,elapsed,want",
    [
        (1, {}, None, False),
        (99_999, {}, None, False),
        (100_000, {}, None, True),  # default floor, no budget evidence
        (50, {}, 1700.0, False),  # < half of the 3600 s budget
        (50, {}, 1900.0, True),
        (400, {"max_iterations": 1000}, None, False),
        (500, {"max_iterations": 1000}, None, True),
        (10, {"min_promote_iterations": 10}, None, True),
    ],
)
def test_progress_is_substantial(iters, job_over, elapsed, want):
    ok, _why = overnight.progress_is_substantial(
        iters, {**JOB, **job_over}, elapsed_secs=elapsed
    )
    assert ok is want


def test_resume_resolves_a_thin_snapshot_and_keeps_a_substantial_one(
    tmp_path: Path, monkeypatch
):
    out = tmp_path / "overnight"
    thin = {**JOB, "job_id": "bp_thin", "max_iterations": 1000}
    thick = {**JOB, "job_id": "bp_thick", "max_iterations": 1000}
    _write_progress(out, "bp_thin", 1)
    _write_progress(out, "bp_thick", 900)
    grid = tmp_path / "grid.json"
    grid.write_text(
        json.dumps({"out_dir": str(out), "stop_file": str(tmp_path / "STOP"),
                    "jobs": [thin, thick]}),
        encoding="utf-8",
    )
    calls: list = []
    monkeypatch.setattr(overnight, "rust_cfr_available", lambda: True)
    monkeypatch.setattr(overnight, "solve", _fake_solve(calls))
    rep = overnight.run_overnight(grid).as_dict()
    by_id = {r["job_id"]: r for r in rep["results"]}
    assert [c[0] for c in calls] == ["bp_thin"]  # solved again, from scratch
    assert by_id["bp_thin"]["status"] == "ok"
    assert by_id["bp_thick"]["status"] == "partial"
    assert (rep["n_ok"], rep["n_partial"]) == (1, 1)
    partial = out / "strategies" / "bp_thick.partial.json"
    payload = json.loads(partial.read_text(encoding="utf-8"))
    assert payload["status"] == "partial" and payload["promoted_from_progress"] is True
    assert not (out / "strategies" / "bp_thick.json").exists()
    # Nothing salvaged from a kill is ever a teacher root.
    res = export_dir_detailed(out / "strategies", tmp_path / "labels.jsonl")
    assert res.train_root_ids == ["bp_thin"]
    # A later resume keeps skipping it … unless asked to finish the job.
    calls.clear()
    again = overnight.run_overnight(grid).as_dict()
    assert calls == [] and {r["status"] for r in again["results"]} == {"skipped", "partial"}
    overnight.run_overnight(grid, rerun_partial=True)
    assert [c[0] for c in calls] == ["bp_thick"]
    assert marker_status(out, "bp_thick") == "ok"
    res = export_dir_detailed(out / "strategies", tmp_path / "labels.jsonl")
    assert res.train_root_ids == ["bp_thick", "bp_thin"]  # the FINISHED solve


def test_pipeline_board_honours_no_resume(tmp_path: Path, monkeypatch):
    import plo5bp._engine as engine

    calls: list = []

    def fake_pipeline(**kw):
        calls.append(kw["postflop_board"])
        return {"preflop_iterations": 1, "postflop_iterations": 1,
                "postflop_exploitability_bb": 0.5}

    monkeypatch.setattr(engine, "cfr_pipeline", fake_pipeline, raising=False)
    job = {"job_id": "pipe", "stack_bb": 50.0}
    kw = dict(stop_file="", out_dir=tmp_path)
    (tmp_path / "markers").mkdir()
    overnight._run_pipeline_board(job, "b0", [0, 5, 10, 15, 20], **kw)
    assert overnight._run_pipeline_board(job, "b0", [0, 5, 10, 15, 20], **kw)["status"] == "skipped"
    assert len(calls) == 1
    got = overnight._run_pipeline_board(job, "b0", [0, 5, 10, 15, 20], resume=False, **kw)
    assert got["status"] == "ok" and len(calls) == 2  # --no-resume used to be ignored


# --- F10: ids -------------------------------------------------------------------


def test_new_ids_distinguish_board_pot_stack_and_sizes():
    base = dict(street=3, pot_bb=10.0, effective_stack_bb=20.0, board=[0, 5, 10, 15, 20])
    ids = {
        RootSpec(**base).root_id,
        RootSpec(**{**base, "board": [1, 5, 10, 15, 20]}).root_id,
        RootSpec(**{**base, "raise_sizes_pm": [500, 1000]}).root_id,
        RootSpec(**{**base, "num_seats": 3}).root_id,
        RootSpec(**{**base, "range_oop": "AA,KK"}).root_id,
    }
    assert len(ids) == 5 and all(i.startswith("s3_pot10_eff20-") for i in ids)
    # order of board cards / size menu is not part of the game
    assert RootSpec(**base).root_id == RootSpec(**{**base, "board": [20, 15, 10, 5, 0]}).root_id
    assert RootSpec(**base, root_id="mine").root_id == "mine"  # explicit ids kept
    a = RootSpec.river_hu([0, 5, 10, 15, 20], pot_bb=10)
    b = RootSpec.river_hu([1, 5, 10, 15, 20], pot_bb=10)
    assert a.root_id != b.root_id and a.root_id.startswith("river_pot10-")
    # the fingerprint ignores the id itself and float spelling
    assert root_fingerprint({**a.as_dict(), "root_id": "x", "pot_bb": 10}) == a.fingerprint()


def test_grid_ids_change_with_the_game_but_not_with_the_solve_config():
    a = expand_river_grid(n_roots=2, seed=3, stack_bb=20.0, iters=100, size_preset="micro")
    b = expand_river_grid(n_roots=2, seed=3, stack_bb=50.0, iters=100, size_preset="micro")
    c = expand_river_grid(n_roots=2, seed=3, stack_bb=20.0, iters=20_000, size_preset="micro")
    assert [j.legacy_job_id for j in a] == [j.legacy_job_id for j in b] == ["s3_s3_i0", "s3_s3_i1"]
    assert {j.job_id for j in a}.isdisjoint({j.job_id for j in b})  # different stacks
    assert [j.job_id for j in a] == [j.job_id for j in c]  # same ROOT (split-stable)
    assert all(j.root.root_id == j.job_id for j in a)


def test_campaign_dir_written_under_legacy_ids_still_resumes(tmp_path: Path, monkeypatch):
    """data/cfr/teacher_s7-style dirs hold ``s3_spr1p0_s7_b0.json`` etc."""
    jobs = expand_river_spr_grid(n_boards=1, seed=7, iters=50)
    for j in jobs:  # write the dir the way the OLD code did
        legacy_root = {**j.root.as_dict(), "root_id": j.legacy_job_id}
        p = tmp_path / "strategies" / f"{j.legacy_job_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"status": "ok", "root": legacy_root, "config": j.config.as_dict(),
                        "strategy": {"infosets": []}, "iterations_run": 50,
                        "exploitability_bb": 0.4, "notes": [FINAL]}),
            encoding="utf-8",
        )
        m = tmp_path / "markers" / f"{j.legacy_job_id}.done"
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text("ok\n", encoding="utf-8")
    calls: list = []
    monkeypatch.setattr(cfr_batch, "solve", _fake_solve(calls))
    fresh = expand_river_spr_grid(n_boards=1, seed=7, iters=50)
    man = run_batch(fresh, tmp_path, resume=True)
    assert calls == [] and sorted(man.skipped) == sorted(j.legacy_job_id for j in jobs)
    assert all(j.job_id == j.legacy_job_id == j.root.root_id for j in fresh)


def test_legacy_id_of_a_different_game_is_not_mistaken_for_this_job(
    tmp_path: Path, monkeypatch
):
    old = expand_river_grid(n_roots=1, seed=3, stack_bb=20.0, iters=50, size_preset="micro")[0]
    legacy = old.legacy_job_id
    (tmp_path / "strategies").mkdir()
    (tmp_path / "markers").mkdir()
    (tmp_path / "strategies" / f"{legacy}.json").write_text(
        json.dumps({"status": "ok", "root": {**old.root.as_dict(), "root_id": legacy},
                    "config": old.config.as_dict(), "strategy": {"infosets": []}}),
        encoding="utf-8",
    )
    (tmp_path / "markers" / f"{legacy}.done").write_text("ok\n", encoding="utf-8")
    # Same grid cell, DIFFERENT stack: the old code skipped it as "done".
    new = expand_river_grid(n_roots=1, seed=3, stack_bb=50.0, iters=50, size_preset="micro")
    assert resolve_job_id(tmp_path, new[0]) == new[0].job_id != legacy
    calls: list = []
    monkeypatch.setattr(cfr_batch, "solve", _fake_solve(calls))
    man = run_batch(new, tmp_path, resume=True)
    assert man.skipped == [] and man.completed == [new[0].job_id]


def test_more_iterations_requested_is_not_silently_resumed(tmp_path: Path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(cfr_batch, "solve", _fake_solve(calls))
    run_batch(expand_river_grid(n_roots=1, seed=0, iters=100, size_preset="micro"), tmp_path)
    big = expand_river_grid(n_roots=1, seed=0, iters=20_000, size_preset="micro")
    man = run_batch(big, tmp_path, resume=True)
    assert [c[1] for c in calls] == [100, 20_000] and man.skipped == []
    plan = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    assert "max_iterations 100 < requested 20000" in plan["stale_resolve"][big[0].job_id]
    # … while asking for FEWER (step-7: 40k retry done, 20k pass re-run) skips.
    small = expand_river_grid(n_roots=1, seed=0, iters=5_000, size_preset="micro")
    assert run_batch(small, tmp_path, resume=True).skipped == [small[0].job_id]
    assert len(calls) == 2


# --- F10: atomic manifests / STOP -----------------------------------------------


def test_manifest_write_is_atomic(tmp_path: Path, monkeypatch):
    path = tmp_path / "manifest.json"
    BatchManifest(out_dir=str(tmp_path), jobs=["a"], completed=["a"]).write(path)
    before = path.read_text(encoding="utf-8")

    def boom(src, dst):
        raise OSError("killed mid-rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        BatchManifest(out_dir=str(tmp_path), jobs=["a", "b"]).write(path)
    assert path.read_text(encoding="utf-8") == before  # never torn / truncated
    assert BatchManifest.load(path).completed == ["a"]


def test_old_manifests_without_the_unverified_list_still_load(tmp_path: Path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"out_dir": "x", "jobs": ["a"], "rejected": []}), encoding="utf-8")
    assert BatchManifest.load(p).unverified == []


def _load_script(name: str):
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_step7_retry_close_does_not_delete_the_operators_stop(
    tmp_path: Path, monkeypatch
):
    mod = _load_script("step7_retry_close")
    camp = tmp_path / "data" / "cfr" / "teacher_s7"
    camp.mkdir(parents=True)
    (camp / "STOP").write_text("", encoding="utf-8")
    (camp / "manifest.json").write_text(
        json.dumps({"out_dir": str(camp), "rejected": [], "completed": []}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["step7_retry_close.py"])
    assert mod.main() == 2
    assert (camp / "STOP").exists()
    monkeypatch.setattr(sys, "argv", ["step7_retry_close.py", "--clear-stop"])
    assert mod.main() == 0  # nothing to retry, but it ran
    assert not (camp / "STOP").exists()
