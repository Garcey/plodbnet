"""Teacher floors: expl reject, visit_mass drop, deterministic holdout split."""

from __future__ import annotations

import json
from pathlib import Path

from plo5bp.gto.cfr_api import SolveReport
from plo5bp.gto.cfr_batch import expand_river_grid, run_batch
from plo5bp.gto.cfr_export import (
    REJECT_LOW_VISIT,
    REJECT_UNUSED_UNIFORM,
    ExportStats,
    export_dir,
    export_dir_detailed,
    export_teacher_dir,
    reject_reason,
    strategy_to_labels,
)
from plo5bp.gto.labels import read_jsonl
from plo5bp.gto.teacher import (
    SPLIT_HOLDOUT,
    SPLIT_TRAIN,
    TEACHER_HOLDOUT_FRAC,
    TEACHER_MAX_EXPL_BB,
    TEACHER_MIN_VISIT_MASS,
    expl_reject_reason,
    holdout_assignment,
    split_root_ids,
)


def _v2_iset(
    *,
    combo: int = 200,
    visit_mass: float | None = 2.0,
    probs: list[float] | None = None,
) -> dict:
    return {
        "infoset_id": f"p0_h1_c{combo}",
        "actions": ["CHECK_CALL", "RAISE_500"],
        "probs": list(probs or [0.6, 0.4]),
        "schema_version": 2,
        "street": 3,
        "actor": 0,
        "to_call_chips": 0,
        "pot_chips": 100_000,
        "min_raise_chips": 10_000,
        "max_raise_chips": 200_000,
        "stacks_chips": [200_000, 200_000],
        "private_kind": "combo",
        "raw_combo": combo,
        "visit_mass": visit_mass,
    }


def _report(
    root_id: str,
    *,
    expl: float | None = 0.2,
    visit_mass: float | None = 2.0,
    combo: int = 200,
) -> dict:
    rep: dict = {
        "status": "ok",
        "iterations_run": 10,
        "notes": [],
        "root": {
            "street": 3,
            "pot_bb": 10.0,
            "effective_stack_bb": 20.0,
            "board": [30, 31, 32, 33, 34],
            "num_seats": 2,
            "bb_chips": 10_000,
            "root_id": root_id,
        },
        "strategy": {
            "root_id": root_id,
            "infosets": [_v2_iset(combo=combo, visit_mass=visit_mass)],
        },
        "config": {},
    }
    if expl is not None:
        rep["exploitability_bb"] = expl
    return rep


def test_expl_reject_reason_defaults():
    assert TEACHER_MAX_EXPL_BB == 1.0
    assert expl_reject_reason(0.2) is None
    assert expl_reject_reason(1.0) is None  # cap is exclusive (e > 1.0)
    assert expl_reject_reason(1.0001) is not None
    assert expl_reject_reason(5.0) == "expl_5.0000_gt_1"
    assert expl_reject_reason(None) == "expl_missing"
    assert expl_reject_reason(float("nan")) == "expl_missing"
    assert expl_reject_reason(-0.1) == "expl_missing"
    assert expl_reject_reason(2.0, max_expl_bb=3.0) is None


def test_high_expl_root_rejected_at_batch(tmp_path: Path, monkeypatch):
    def fake_solve(root, cfg):
        return SolveReport(
            status="ok",
            root=root.as_dict(),
            config=cfg.as_dict(),
            strategy={"root_id": root.root_id, "infosets": []},
            iterations_run=3,
            exploitability_bb=5.0,
            notes=[],
        )

    monkeypatch.setattr("plo5bp.gto.cfr_batch.solve", fake_solve)
    jobs = expand_river_grid(n_roots=1, seed=0, iters=3, size_preset="micro")
    out = tmp_path / "batch"
    man = run_batch(jobs, out, dry_run=False, resume=False, max_expl_bb=1.0)
    jid = jobs[0].job_id
    assert man.completed == []
    assert man.failed == []
    assert len(man.rejected) == 1
    assert man.rejected[0]["job_id"] == jid
    assert "expl_" in man.rejected[0]["error"]
    assert not (out / "strategies" / f"{jid}.json").exists()
    rejected = json.loads((out / "rejected" / f"{jid}.json").read_text(encoding="utf-8"))
    assert rejected["status"] == "rejected"
    assert rejected["exploitability_bb"] == 5.0
    assert (out / "markers" / f"{jid}.done").read_text(encoding="utf-8").strip() == "rejected"

    man2 = run_batch(jobs, out, dry_run=False, resume=True, max_expl_bb=1.0)
    assert man2.skipped == [jid]
    assert man2.completed == []
    assert man2.rejected == []


def test_high_expl_root_rejected_at_export(tmp_path: Path):
    good = tmp_path / "ok.json"
    bad = tmp_path / "bad.json"
    missing = tmp_path / "miss.json"
    good.write_text(json.dumps(_report("ok_root", expl=0.4)), encoding="utf-8")
    bad.write_text(json.dumps(_report("bad_root", expl=3.2, combo=201)), encoding="utf-8")
    miss_rep = _report("miss_root", expl=None, combo=202)
    missing.write_text(json.dumps(miss_rep), encoding="utf-8")

    out = tmp_path / "labels.jsonl"
    res = export_dir_detailed(tmp_path, out, max_expl_bb=1.0)
    assert res.n_train == 1
    assert len(res.skipped_expl) == 2
    reasons = {s["root_id"]: s["reason"] for s in res.skipped_expl}
    assert "expl_" in reasons["bad_root"]
    assert reasons["miss_root"] == "expl_missing"
    labs = list(read_jsonl(out))
    assert len(labs) == 1
    assert labs[0].root_name == "ok_root"


def test_low_visit_mass_dropped():
    assert TEACHER_MIN_VISIT_MASS == 1.0
    assert (
        reject_reason(
            actions=["CHECK_CALL", "RAISE_500"],
            probs=[0.6, 0.4],
            to_call=0,
            pot_chips=100_000,
            hero_stack=200_000,
            visit_mass=0.25,
            v2_missing_to_call=False,
            hole=[1, 2],
            require_hole=False,
            min_visit_mass=TEACHER_MIN_VISIT_MASS,
        )
        == REJECT_LOW_VISIT
    )
    # Zero mass stays unused_uniform (existing gate), not low_visit.
    assert (
        reject_reason(
            actions=["CHECK_CALL", "RAISE_500"],
            probs=[0.6, 0.4],
            to_call=0,
            pot_chips=100_000,
            hero_stack=200_000,
            visit_mass=0.0,
            v2_missing_to_call=False,
            hole=[1, 2],
            require_hole=False,
            min_visit_mass=TEACHER_MIN_VISIT_MASS,
        )
        == REJECT_UNUSED_UNIFORM
    )

    stats = ExportStats()
    labs = strategy_to_labels(
        _report("low_vm", expl=0.1, visit_mass=0.25),
        stats=stats,
        min_visit_mass=TEACHER_MIN_VISIT_MASS,
    )
    assert labs == []
    assert stats.rejected.get(REJECT_LOW_VISIT) == 1

    # Library default (0) does not drop a positive-but-low visit.
    stats0 = ExportStats()
    labs0 = strategy_to_labels(
        _report("low_vm_lib", expl=0.1, visit_mass=0.25),
        stats=stats0,
    )
    assert len(labs0) == 1
    assert stats0.kept == 1

    # Floor is exclusive: mass == 1.0 is kept.
    stats1 = ExportStats()
    labs1 = strategy_to_labels(
        _report("edge", expl=0.1, visit_mass=1.0),
        stats=stats1,
        min_visit_mass=1.0,
    )
    assert len(labs1) == 1


def test_holdout_split_deterministic_and_disjoint():
    assert 0.10 <= TEACHER_HOLDOUT_FRAC <= 0.20
    ids = [f"s3_s0_i{i}" for i in range(40)]
    train_a, hold_a = split_root_ids(ids, seed=0, holdout_frac=0.15)
    train_b, hold_b = split_root_ids(ids, seed=0, holdout_frac=0.15)
    assert train_a == train_b
    assert hold_a == hold_b
    assert set(train_a).isdisjoint(hold_a)
    assert set(train_a) | set(hold_a) == set(ids)
    assert hold_a  # 40 ids at 15% is vanishingly unlikely to be empty
    assert train_a
    # Different seed is allowed to reshuffle.
    train_c, hold_c = split_root_ids(ids, seed=99, holdout_frac=0.15)
    assert (train_c, hold_c) != (train_a, hold_a)
    assert holdout_assignment("r", holdout_frac=0.0) == SPLIT_TRAIN
    assert holdout_assignment("r", holdout_frac=1.0) == SPLIT_HOLDOUT


def test_export_holdout_jsonl_disjoint(tmp_path: Path):
    for i in range(20):
        p = tmp_path / f"root_{i}.json"
        p.write_text(
            json.dumps(_report(f"root_{i}", expl=0.1, combo=200 + i)),
            encoding="utf-8",
        )
    train_path = tmp_path / "out" / "train.jsonl"
    res1 = export_teacher_dir(
        tmp_path,
        train_path,
        holdout_frac=0.5,
        split_seed=0,
        max_expl_bb=1.0,
        min_visit_mass=1.0,
    )
    res2 = export_teacher_dir(
        tmp_path,
        tmp_path / "out2" / "train.jsonl",
        holdout_frac=0.5,
        split_seed=0,
        max_expl_bb=1.0,
        min_visit_mass=1.0,
    )
    assert res1.train_root_ids == res2.train_root_ids
    assert res1.holdout_root_ids == res2.holdout_root_ids
    assert set(res1.train_root_ids).isdisjoint(res1.holdout_root_ids)
    assert set(res1.train_root_ids) | set(res1.holdout_root_ids) == {
        f"root_{i}" for i in range(20)
    }
    assert res1.holdout_path
    train_names = {lab.root_name for lab in read_jsonl(res1.train_path)}
    hold_names = {lab.root_name for lab in read_jsonl(res1.holdout_path)}
    assert train_names == set(res1.train_root_ids)
    assert hold_names == set(res1.holdout_root_ids)
    split = json.loads(Path(res1.split_path).read_text(encoding="utf-8"))
    assert split["holdout_frac"] == 0.5
    assert split["min_visit_mass"] == 1.0
    assert split["train_path"] == res1.train_path
    assert split["holdout_path"] == res1.holdout_path


def test_expand_river_spr_grid_count_and_ids():
    from plo5bp.gto.cfr_batch import expand_river_spr_grid

    jobs = expand_river_spr_grid(n_boards=2, seed=7, iters=10)
    assert len(jobs) == 8  # 4 SPR × 2 boards
    sprs = {round(j.root.effective_stack_bb / j.root.pot_bb, 3) for j in jobs}
    assert sprs == {1.0, 2.0, 3.0, 5.0}
    assert all(j.root.street == 3 for j in jobs)
    assert all(j.config.use_isomorphism is False for j in jobs)
    ids = [j.job_id for j in jobs]
    assert len(ids) == len(set(ids))


def test_export_dir_backcompat_no_floors(tmp_path: Path):
    """Library export_dir keeps high-expl / low-visit rows unless floors passed."""
    p = tmp_path / "s.json"
    p.write_text(
        json.dumps(_report("legacy", expl=9.0, visit_mass=0.2)),
        encoding="utf-8",
    )
    n = export_dir(p, tmp_path / "all.jsonl")
    assert n == 1
