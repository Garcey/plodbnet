"""Review 2026-09-20 D8 / F6 — exploitability provenance + GTO badge provenance.

- a report is exploitability-VERIFIED only on a final-estimator note without an
  early-stop / promoted marker; ``cfr_batch`` neither accepts nor rejects on
  anything else;
- the step-6 helper scripts never set ``target == cap``;
- the GTO badge needs label provenance DERIVED from the training records (and
  the holdout's), a teacher cap no looser than 1.0 bb, and never honours a bare
  ``is_gto_validated=True`` or a script-asserted ``source``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from plo5bp.gto.cfr_api import SolveReport
from plo5bp.gto.cfr_batch import (
    MARKER_UNVERIFIED,
    expand_river_grid,
    marker_status,
    run_batch,
)
from plo5bp.gto.dataset import RowProvenance, SupervisedRow, load_rows_npz, save_rows_npz
from plo5bp.gto.labels import make_smoke_label, write_jsonl
from plo5bp.gto.obs_from_label import labels_to_supervised_rows
from plo5bp.gto.policy_net import (
    build_policy_net,
    is_validated_gto_checkpoint,
    is_validated_gto_meta,
    label_provenance_problem,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from plo5bp.gto.probe import ProbeGates, probe_checkpoint, stamp_probe_on_checkpoint
from plo5bp.gto.teacher import TEACHER_MAX_EXPL_BB, expl_provenance
from plo5bp.gto.train import TrainConfig, derive_training_provenance, train_policy_net

REPO = Path(__file__).resolve().parents[2]
FINAL = "expl_kind=infoset_br samples=128"


# --- expl_provenance ------------------------------------------------------------


@pytest.mark.parametrize(
    "notes,verified,reason",
    [
        ([FINAL], True, ""),
        (["expl_kind=exact_infoset"], True, ""),
        (["expl_kind=hero_enum samples=64"], True, ""),
        ([FINAL, "early_stop=time_budget"], False, "early_stop=time_budget"),
        ([FINAL, "early_stop=stop_file"], False, "early_stop=stop_file"),
        (["expl_kind=mc_poll"], False, "expl_kind=mc_poll"),
        (["DCFR River HU early stop iter 50"], False, "early_stop=target_poll"),
        (["mc_br_proxy_bb=0.31"], False, "expl_kind_missing"),
        ([], False, "expl_kind_missing"),
        ([FINAL, "promoted_from_progress"], False, "promoted_from_progress"),
    ],
)
def test_expl_provenance_table(notes, verified, reason):
    prov = expl_provenance({"status": "ok", "notes": notes})
    assert (prov.verified, prov.reason) == (verified, reason)


def test_expl_provenance_needs_an_ok_finished_report():
    assert not expl_provenance({"status": "partial", "notes": [FINAL]}).verified
    assert not expl_provenance(
        {"status": "ok", "notes": [FINAL], "promoted_from_progress": True}
    ).verified
    rep = SolveReport("ok", {}, {}, {}, 10, 0.5, [FINAL])
    assert expl_provenance(rep).verified  # dataclass form too


# --- cfr_batch ------------------------------------------------------------------


def _fake_solve(expl: float, notes: list[str]):
    def solve(root, cfg):
        return SolveReport(
            status="ok", root=root.as_dict(), config=cfg.as_dict(),
            strategy={"root_id": root.root_id, "infosets": []},
            iterations_run=3, exploitability_bb=expl, notes=list(notes),
        )

    return solve


@pytest.mark.parametrize("expl", [0.2, 5.0])
def test_batch_neither_accepts_nor_rejects_on_a_time_budget_estimate(
    tmp_path: Path, monkeypatch, expl: float
):
    """Same verdict whether the biased poll number is under or over the cap."""
    monkeypatch.setattr(
        "plo5bp.gto.cfr_batch.solve", _fake_solve(expl, [FINAL, "early_stop=time_budget"])
    )
    jobs = expand_river_grid(n_roots=1, seed=0, iters=3, size_preset="micro")
    man = run_batch(jobs, tmp_path, resume=False, max_expl_bb=1.0)
    jid = jobs[0].job_id
    assert man.completed == [] and man.rejected == [] and man.failed == []
    assert [u["job_id"] for u in man.unverified] == [jid]
    assert "early_stop=time_budget" in man.unverified[0]["error"]
    assert marker_status(tmp_path, jid) == MARKER_UNVERIFIED
    assert (tmp_path / "unverified" / f"{jid}.json").is_file()
    assert not (tmp_path / "strategies" / f"{jid}.json").exists()
    assert not (tmp_path / "rejected" / f"{jid}.json").exists()


def test_batch_stop_file_exit_is_interrupted_and_resolved_on_resume(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        "plo5bp.gto.cfr_batch.solve", _fake_solve(0.3, [FINAL, "early_stop=stop_file"])
    )
    jobs = expand_river_grid(n_roots=1, seed=0, iters=3, size_preset="micro")
    man = run_batch(jobs, tmp_path, resume=True, max_expl_bb=1.0)
    jid = jobs[0].job_id
    assert man.unverified[0]["status"] == "interrupted"
    assert marker_status(tmp_path, jid) is None  # NOT done → resume solves it
    monkeypatch.setattr("plo5bp.gto.cfr_batch.solve", _fake_solve(0.3, [FINAL]))
    man2 = run_batch(jobs, tmp_path, resume=True, max_expl_bb=1.0)
    assert man2.completed == [jid] and man2.skipped == []


def test_batch_without_floors_keeps_every_ok_solve(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("plo5bp.gto.cfr_batch.solve", _fake_solve(9.0, []))
    jobs = expand_river_grid(n_roots=1, seed=0, iters=3, size_preset="micro")
    man = run_batch(jobs, tmp_path, resume=False, max_expl_bb=None)
    assert man.completed == [jobs[0].job_id] and man.unverified == []


# --- step-6 scripts never target the cap ----------------------------------------


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_step6_find_pass_runs_to_the_cap_and_needs_a_verified_number(
    monkeypatch, capsys
):
    mod = _load_script("step6_find_pass")
    seen: list[float] = []

    def fake(root, cfg):
        seen.append(cfg.target_exploitability_bb)
        # under the cap, but a poll-grade number → must NOT read as a pass
        notes = ["expl_kind=mc_poll"] if len(seen) == 1 else [FINAL]
        return SolveReport("ok", root.as_dict(), cfg.as_dict(), {}, 10, 0.6, notes)

    monkeypatch.setattr(mod, "solve", fake)
    monkeypatch.setattr(mod, "CANDIDATES", mod.CANDIDATES[:2])
    mod.main()
    assert seen == [0.0, 0.0]  # never target == cap (first noisy poll dip)
    out = capsys.readouterr().out
    assert "verified=False" in out and out.count("PASS=False") == 1
    assert out.count("PASS=True") == 1


def test_step6_raise_one_gates_acceptance_on_provenance(tmp_path: Path, monkeypatch):
    mod = _load_script("step6_raise_one")
    job = expand_river_grid(n_roots=2, seed=3, iters=5, size_preset="micro")[1]
    job.root.root_id = "s3_s3_i1"
    rej = tmp_path / "data" / "cfr" / "step6_teacher" / "rejected"
    rej.mkdir(parents=True)
    (rej / "s3_s3_i1.json").write_text(
        json.dumps({"exploitability_bb": 3.0, "report": {"root": job.root.as_dict()}}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    targets: list[float] = []

    def fake(root, cfg):
        targets.append(cfg.target_exploitability_bb)
        return SolveReport(
            "ok", root.as_dict(), cfg.as_dict(), {"infosets": []}, 10, 0.5,
            [FINAL, "early_stop=time_budget"],
        )

    monkeypatch.setattr("plo5bp.gto.cfr_batch.solve", fake)
    assert mod.main() == 1  # 0.5 <= cap, but unverified → not accepted
    assert targets == [0.0]
    out = tmp_path / "data" / "cfr" / "step6_teacher"
    assert not (out / "strategies" / "s3_s3_i1.json").exists()
    assert marker_status(out, "s3_s3_i1") == MARKER_UNVERIFIED


# --- checkpoint provenance is derived from the rows ------------------------------


def _teacher_label(root: str, *, jam: bool, seed: int = 0, **notes):
    lab = make_smoke_label(seed=seed, trash_fold=not jam)
    lab.source = "rust_cfr_river"
    lab.root_name = root
    lab.notes = {
        **lab.notes,
        "exploitability_bb": 0.4,
        "expl_kind": "infoset_br",
        "expl_verified": True,
        "teacher_max_expl_bb": 1.0,
        **notes,
    }
    return lab


def _labels(root: str, **notes):
    return [_teacher_label(root, jam=bool(i % 2), seed=i, **notes) for i in range(16)]


def _train(tmp_path: Path, labels, name="ck.pt", **kw) -> Path:
    ckpt = tmp_path / name
    train_policy_net(
        labels_to_supervised_rows(labels),
        ckpt,
        cfg=TrainConfig(hidden_dim=64, num_layers=1, epochs=40, batch_size=16,
                        lr=3e-3, log_every=10**9, seed=0),
        **kw,
    )
    return ckpt


GATES = ProbeGates(min_n=4, min_pure_agree=0.75, max_mean_gate_kl=0.8, min_pure_n=2)


def test_checkpoint_meta_is_derived_from_the_training_records(tmp_path: Path):
    ckpt = _train(
        tmp_path,
        _labels("root_a") + _labels("root_b", exploitability_bb=0.9),
        # a script ASSERTING things must not win over the records
        meta={"train_root_ids": ["lies"], "is_gto_validated": True,
              "label_provenance": {"derived_from_records": True, "sources": {}}},
    )
    _, meta = load_policy_checkpoint(ckpt)
    assert meta["train_root_ids"] == ["root_a", "root_b"]
    assert meta["train_roots_known"] is True
    lp = meta["label_provenance"]
    assert lp["sources"] == {"rust_cfr_river": 32}
    assert lp["n_unverified_expl"] == 0
    assert lp["max_label_expl_bb"] == pytest.approx(0.9)
    assert lp["teacher_max_expl_bb"] == pytest.approx(1.0)
    assert meta["source"] == "rust_cfr_river"  # named after the records
    assert meta["coverage"]["seats"] == [2] and meta["coverage"]["streets"] == [3]
    assert meta["is_gto_validated"] is False  # no probe yet
    assert label_provenance_problem(meta) is None


def test_badge_needs_probe_plus_train_and_holdout_provenance(tmp_path: Path):
    ckpt = _train(tmp_path, _labels("train_root"))
    hold = tmp_path / "hold.jsonl"
    write_jsonl(hold, _labels("holdout_root"))
    res = probe_checkpoint(ckpt, hold, gates=GATES)
    assert res.passed, res.reasons
    assert res.report.holdout_provenance["problem"] is None
    meta = stamp_probe_on_checkpoint(ckpt, res)
    assert meta["is_gto_validated"] is True and meta["gto_badge_note"] is None
    assert is_validated_gto_checkpoint(ckpt)


def test_badge_refused_for_unverified_or_loose_cap_teachers(tmp_path: Path):
    hold = tmp_path / "hold.jsonl"
    write_jsonl(hold, _labels("holdout_root"))
    for name, notes, why in [
        ("unverified.pt", {"expl_verified": False}, "without a verified exploitability"),
        ("oneoff5.pt", {"teacher_max_expl_bb": 5.0, "exploitability_bb": 3.9}, "looser than"),
        ("nocap.pt", {"teacher_max_expl_bb": None}, "cap missing"),
    ]:
        ckpt = _train(tmp_path, _labels("train_root", **notes), name=name)
        res = probe_checkpoint(ckpt, hold, gates=GATES)
        assert res.passed, res.reasons  # the NET matches its teacher …
        meta = stamp_probe_on_checkpoint(ckpt, res)
        assert meta["is_gto_validated"] is False  # … the TEACHER is the problem
        assert why in meta["gto_badge_note"]
        assert not is_validated_gto_checkpoint(ckpt)


def test_badge_refused_when_the_holdout_teacher_is_unverified(tmp_path: Path):
    ckpt = _train(tmp_path, _labels("train_root"))
    hold = tmp_path / "hold.jsonl"
    write_jsonl(hold, _labels("holdout_root", expl_verified=False))
    res = probe_checkpoint(ckpt, hold, gates=GATES)
    meta = stamp_probe_on_checkpoint(ckpt, res)
    assert res.passed and meta["is_gto_validated"] is False
    assert "holdout labels" in meta["gto_badge_note"]


def test_script_asserted_source_and_flag_are_not_honoured(tmp_path: Path):
    """The old F6 hole: meta={'source': 'rust_cfr'} + a probe pass == GTO AI."""
    smoke = [make_smoke_label(seed=i, trash_fold=bool(i % 2)) for i in range(16)]
    for lab in smoke:
        lab.root_name = "train_root"
    ckpt = _train(tmp_path, smoke, meta={"source": "rust_cfr", "is_gto_validated": True})
    _, meta = load_policy_checkpoint(ckpt)
    assert meta["is_gto_validated"] is False
    assert meta["label_provenance"]["sources"] == {"synthetic_smoke": 16}
    assert "synthetic_smoke" in label_provenance_problem(meta)
    # bare flag / bare probe pass on an otherwise empty meta
    m = build_policy_net(hidden_dim=32, num_layers=1)
    bare = tmp_path / "bare.pt"
    save_policy_checkpoint(bare, m, meta={"source": "rust_cfr", "is_gto_validated": True})
    assert not is_validated_gto_checkpoint(bare)
    save_policy_checkpoint(
        bare, m, meta={"source": "rust_cfr", "probe": {"passed": True, "report": {}}}
    )
    assert not is_validated_gto_checkpoint(bare)
    assert not is_validated_gto_meta({"source": "rust_cfr", "is_gto_validated": True})


def test_warm_start_inherits_roots_and_unknown_roots_stay_unknown(tmp_path: Path):
    first = _train(tmp_path, _labels("root_a"), name="a.pt")
    second = _train(tmp_path, _labels("root_b"), name="b.pt", init_ckpt=first)
    _, meta = load_policy_checkpoint(second)
    assert meta["train_root_ids"] == ["root_a", "root_b"] and meta["train_roots_known"]
    assert set(meta["label_provenance"]["sources"]) == {
        "rust_cfr_river", "warm_start:rust_cfr_river"
    }
    legacy = tmp_path / "legacy.pt"  # a pre-2026-09-20 checkpoint: no root ids
    save_policy_checkpoint(
        legacy, build_policy_net(hidden_dim=64, num_layers=1), meta={"source": "rust_cfr"}
    )
    third = _train(tmp_path, _labels("root_c"), name="c.pt", init_ckpt=legacy)
    _, meta3 = load_policy_checkpoint(third)
    assert meta3["train_roots_known"] is False
    assert label_provenance_problem(meta3) is not None


def test_cap_constant_is_the_badge_bar():
    assert TEACHER_MAX_EXPL_BB == 1.0
    prov = derive_training_provenance(labels_to_supervised_rows(_labels("r")))
    assert label_provenance_problem(prov) is None
    assert "looser" in label_provenance_problem(prov, max_teacher_cap_bb=0.5)


# --- F11: value targets of None are masked, not trained toward 0 -----------------


def _value_rows(value: float, mask: bool) -> list[SupervisedRow]:
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(32):
        g = rng.random(3).astype(np.float32)
        rows.append(
            SupervisedRow(
                obs=rng.standard_normal(995).astype(np.float32),
                gate_mask=np.array([True, True, True]),
                sizing=np.array([10_000, 500_000, 100_000, 0], dtype=np.int64),
                gate_probs=g / g.sum(),
                anchor_probs=np.full(12, 1 / 12, dtype=np.float32),
                value_bb=value,
                street=3,
                value_mask=mask,
            )
        )
    return rows


def _value_head(tmp_path: Path, name: str, value: float, mask: bool):
    ckpt = tmp_path / name
    train_policy_net(
        _value_rows(value, mask), ckpt,
        cfg=TrainConfig(hidden_dim=32, num_layers=1, epochs=3, batch_size=16,
                        value_coef=1.0, log_every=10**9, seed=0),
    )
    model, meta = load_policy_checkpoint(ckpt)
    return model.value_head.weight.detach().clone(), meta


def test_masked_value_rows_do_not_train_the_value_head(tmp_path: Path):
    w_a, meta_a = _value_head(tmp_path, "a.pt", 0.0, False)
    w_b, _ = _value_head(tmp_path, "b.pt", 500.0, False)
    assert bool((w_a == w_b).all())  # the (absent) target never entered the loss
    assert meta_a["n_value_targets"] == 0
    w_c, meta_c = _value_head(tmp_path, "c.pt", 500.0, True)
    assert not bool((w_a == w_c).all()) and meta_c["n_value_targets"] == 32


def test_cfr_labels_have_no_value_target_and_say_so():
    lab = make_smoke_label()
    lab.value_bb = None
    (row,) = labels_to_supervised_rows([lab])
    assert row.value_mask is False and row.value_bb == 0.0
    (row2,) = labels_to_supervised_rows([make_smoke_label()])
    assert row2.value_mask is True


def test_row_bundle_roundtrips_mask_and_provenance(tmp_path: Path):
    rows = _value_rows(1.5, False)
    rows[3].prov = RowProvenance(root_id="r9", source="rust_cfr", num_seats=2,
                                 expl_bb=0.3, expl_verified=True, teacher_cap_bb=1.0,
                                 obs_form="canonical")
    save_rows_npz(tmp_path / "rows.npz", rows)
    back = load_rows_npz(tmp_path / "rows.npz")
    assert [r.value_mask for r in back] == [False] * 32
    assert back[3].prov == rows[3].prov and back[0].prov == RowProvenance()
