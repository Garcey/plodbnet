"""Review 2026-09-20 — CFR strategy → LabelRecord export + row building.

D1  ALLIN lands on the grid-LEGAL jam anchor; illegal teacher mass raises.
D2  push/fold ``ALLIN`` facing a jam is a CALL all-in, not a masked-out raise.
D14 multiway preflop button = n-3.
D13 progress / partial / non-ok reports are never exported.
D16 lossy synthetic obs rows are counted, logged and excluded by default.
F10 legacy root-id collisions are disambiguated; exports are atomic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from plo5bp.gto.bootstrap import collect_bootstrap
from plo5bp.gto.cfr_api import RootSpec, SolveConfig, rust_cfr_available, solve
from plo5bp.gto.cfr_export import (
    IllegalTeacherMassError,
    export_dir_detailed,
    export_teacher_dir,
    strategy_to_labels,
)
from plo5bp.gto.dataset import rows_from_label_records
from plo5bp.gto.labels import (
    ActionProb,
    illegal_teacher_mass,
    jam_anchor_index,
    make_smoke_label,
    read_jsonl,
)
from plo5bp.gto.obs_from_label import (
    OBS_KIND_LOSSY,
    ObsSynthesisStats,
    labels_to_supervised_rows,
)
from plo5bp.gto.train import TrainConfig, train_policy_net
from plo5bp.sizing import NLH_ANCHOR_SPEC, anchor_grid_np, anchor_grid_torch

BOARD = [0, 5, 10, 15, 20]


def _river_report(
    isets: list[dict],
    *,
    root_id: str = "river_t",
    board: list[int] | None = None,
    status: str = "ok",
    expl: float | None = 0.4,
    notes: list[str] | None = None,
    iterations: int = 100,
) -> dict:
    return {
        "status": status,
        "iterations_run": iterations,
        "exploitability_bb": expl,
        "notes": ["expl_kind=infoset_br samples=128"] if notes is None else notes,
        "config": {},
        "root": {
            "street": 3,
            "pot_bb": 10.0,
            "effective_stack_bb": 20.0,
            "board": list(board or BOARD),
            "num_seats": 2,
            "bb_chips": 10_000,
            "sb_chips": 5_000,
            "ante_chips": 5_000,
            "raise_sizes_pm": [500, 1000],
            "root_id": root_id,
        },
        "strategy": {"root_id": root_id, "schema_version": 2, "infosets": isets},
    }


def _iset(actions, probs, *, combo=100, board=None, **over) -> dict:
    """v2 dump row at the pot-10bb / stack-20bb river root."""
    d = {
        "infoset_id": f"p0_h0_c{combo}",
        "actions": list(actions),
        "probs": list(probs),
        "schema_version": 2,
        "street": 3,
        "actor": 0,
        "pot_chips": 100_000,
        "to_call_chips": 0,
        "min_raise_chips": 10_000,
        "max_raise_chips": 200_000,
        "stacks_chips": [200_000, 200_000],
        "folded": [False, False],
        "board": list(board or BOARD),
        "path": [],
        "private_kind": "combo",
        "private_id": combo,
        "raw_combo": combo,
        "iso_id": None,
        "visit_mass": 50.0,
    }
    d.update(over)
    return d


def _train_masked_anchor_target(row) -> np.ndarray:
    """The anchor target exactly as ``gto/train.py`` fits it."""
    legal = anchor_grid_torch(torch.from_numpy(row.sizing)[None], NLH_ANCHOR_SPEC).legal
    t = torch.from_numpy(row.anchor_probs)[None] * legal.float()
    return (t / t.sum(-1, keepdim=True).clamp_min(1e-8))[0].numpy()


# --- D1 -----------------------------------------------------------------------


def test_allin_keeps_its_share_of_raise_mass_through_training_mask():
    """The review's fixture: pot 10bb / stack 20bb, RAISE_500 0.1 + ALLIN 0.7.

    The 200% anchor clamps to the stack, so the ALL-IN atom (11) is grid-illegal
    and the old export's jam mass (87.5% of raises) was masked to nothing.
    """
    rep = _river_report([_iset(["CHECK_CALL", "RAISE_500", "ALLIN"], [0.2, 0.1, 0.7])])
    lab = strategy_to_labels(rep)[0]
    grid = anchor_grid_np(10_000, 200_000, 100_000, 0, NLH_ANCHOR_SPEC)
    assert not bool(grid.legal[NLH_ANCHOR_SPEC.count - 1])  # the old target
    jam = [a for a in lab.action_probs if a.gate == "raise" and a.chips == 200_000]
    assert len(jam) == 1 and jam[0].prob == pytest.approx(0.7)
    k_jam = jam[0].anchor_k
    assert bool(grid.legal[k_jam]) and int(grid.chips[k_jam]) == 200_000
    assert k_jam == jam_anchor_index(
        min_raise=10_000, max_raise=200_000, pot=100_000, to_call=0
    )
    row = rows_from_label_records([lab])[0]
    target = _train_masked_anchor_target(row)
    assert target[k_jam] == pytest.approx(0.875, abs=1e-6)
    assert target.sum() == pytest.approx(1.0, abs=1e-6)
    assert illegal_teacher_mass(
        lab.action_probs, min_raise=10_000, max_raise=200_000, pot_chips=100_000, to_call=0
    ) == 0.0


def test_allin_atom_is_used_when_it_is_the_legal_jam_anchor():
    """Deep (SPR 5): no fraction anchor reaches the stack → atom 11 is legal."""
    rep = _river_report(
        [
            _iset(
                ["CHECK_CALL", "ALLIN"],
                [0.5, 0.5],
                max_raise_chips=500_000,
                stacks_chips=[500_000, 500_000],
            )
        ]
    )
    lab = strategy_to_labels(rep)[0]
    (jam,) = [a for a in lab.action_probs if a.gate == "raise"]
    assert jam.anchor_k == NLH_ANCHOR_SPEC.count - 1 and jam.chips == 500_000


def test_twin_jam_actions_merge_onto_one_anchor():
    """Facing a pot bet at SPR 2 the solver lists RAISE_500 (clamped to the
    stack) AND ALLIN — one serve action. Their mass is summed, not split."""
    rep = _river_report(
        [
            _iset(
                ["FOLD", "CHECK_CALL", "RAISE_500", "ALLIN"],
                [0.1, 0.3, 0.25, 0.35],
                pot_chips=200_000,
                to_call_chips=100_000,
                min_raise_chips=200_000,
                max_raise_chips=200_000,
                stacks_chips=[200_000, 100_000],
                path=["CHECK_CALL", "RAISE_1000"],
            )
        ]
    )
    lab = strategy_to_labels(rep)[0]
    raises = [a for a in lab.action_probs if a.gate == "raise"]
    assert len(raises) == 1
    assert raises[0].prob == pytest.approx(0.6) and raises[0].chips == 200_000
    keys = [(a.gate, a.anchor_k) for a in lab.action_probs]
    assert len(keys) == len(set(keys))
    assert lab.gate_probs == pytest.approx([0.1, 0.3, 0.6])


def test_bootstrap_and_smoke_targets_sit_on_legal_anchors_only():
    rows = collect_bootstrap(n_decisions=200, seed=1, seats=(2, 3))
    rows += rows_from_label_records(
        [make_smoke_label(trash_fold=True), make_smoke_label(trash_fold=False)]
    )
    checked = 0
    for r in rows:
        if not bool(r.gate_mask[2]):
            continue
        mn, mx, pot, tc = (int(x) for x in r.sizing)
        legal = anchor_grid_np(mn, mx, pot, tc, NLH_ANCHOR_SPEC).legal
        assert float(r.anchor_probs[~legal].sum()) <= 1e-6
        assert float(r.anchor_probs.sum()) == pytest.approx(1.0, abs=1e-5)
        checked += 1
    assert checked > 50


def _stale_label():
    """A label as the PRE-FIX export wrote it: ALLIN on the illegal atom."""
    rep = _river_report([_iset(["CHECK_CALL", "RAISE_500", "ALLIN"], [0.2, 0.1, 0.7])])
    lab = strategy_to_labels(rep)[0]
    lab.action_probs = [
        ActionProb("check_call", None, None, 0, 0.2),
        ActionProb("raise", 3, 0.5, 50_000, 0.1),
        ActionProb("raise", NLH_ANCHOR_SPEC.count - 1, None, 200_000, 0.7),
    ]
    return lab


def test_stale_label_is_refused_not_renormalized():
    with pytest.raises(IllegalTeacherMassError, match="river_t"):
        rows_from_label_records([_stale_label()])


def test_training_refuses_illegal_anchor_mass_naming_the_root(tmp_path: Path):
    good = rows_from_label_records(
        strategy_to_labels(
            _river_report([_iset(["CHECK_CALL", "RAISE_500", "ALLIN"], [0.2, 0.1, 0.7])])
        )
    )[0]
    bad_target = np.zeros_like(good.anchor_probs)
    bad_target[NLH_ANCHOR_SPEC.count - 1] = 1.0  # the illegal atom
    good.anchor_probs = bad_target
    with pytest.raises(IllegalTeacherMassError, match="river_t"):
        train_policy_net(
            [good],
            tmp_path / "x.pt",
            cfg=TrainConfig(hidden_dim=16, num_layers=1, epochs=1, log_every=10**9),
        )


def test_illegal_gate_mass_is_refused():
    lab = make_smoke_label(trash_fold=False)
    lab.to_call_chips = 0  # nothing to call → FOLD illegal
    lab.gate_probs = [0.3, 0.2, 0.5]
    with pytest.raises(IllegalTeacherMassError, match="illegal gate"):
        rows_from_label_records([lab])


# --- D2 -----------------------------------------------------------------------


@pytest.mark.skipif(not rust_cfr_available(), reason="extension not built")
def test_pushfold_facing_jam_rows_keep_the_cfr_continue_frequency():
    rep = solve(
        RootSpec.preflop_pushfold(num_seats=3, stack_bb=10.0),
        SolveConfig.teacher(
            max_iterations=30_000,
            algorithm="mccfr_es",
            seed=1,
            target_exploitability_bb=0.0,
            poll_every=10**6,
        ),
    )
    by_id = {str(i["infoset_id"]): i for i in rep.strategy["infosets"]}
    labs = strategy_to_labels(rep.as_dict(), min_visit_mass=1.0)
    facing = [l for l in labs if l.to_call_chips > 10_000]
    assert len(facing) > 100
    rows = rows_from_label_records(facing)
    assert len(rows) == len(facing)
    cont_cfr, cont_row = [], []
    for lab, row in zip(facing, rows):
        iset = by_id[lab.solve_id]
        assert int(iset["max_raise_chips"]) == 0  # no raise legal: continuing == calling
        # continue = everything but FOLD (the solver spells it ALLIN today)
        p_cfr = 1.0 - dict(zip(iset["actions"], iset["probs"])).get("FOLD", 0.0)
        assert row.gate_mask.tolist() == [True, True, False]
        assert float(row.gate_probs[1]) == pytest.approx(p_cfr, abs=1e-5)
        assert all(a.gate != "raise" for a in lab.action_probs)
        call = next(a for a in lab.action_probs if a.gate == "check_call")
        assert call.chips == lab.to_call_chips
        cont_cfr.append(p_cfr)
        cont_row.append(float(row.gate_probs[1]))
    # The old export trained EVERY one of these rows as 100% fold.
    assert np.mean(cont_cfr) > 0.2
    assert np.mean(cont_row) == pytest.approx(np.mean(cont_cfr), abs=1e-5)
    aa = [r for l, r in zip(facing, rows) if l.notes.get("class_label") == "AA"]
    assert aa and all(float(r.gate_probs[1]) > 0.9 for r in aa)
    # Open jams stay raises, on a legal anchor whose chips are the stack.
    opens = [l for l in labs if any(a.gate == "raise" for a in l.action_probs)]
    assert opens
    for lab in opens:
        grid = anchor_grid_np(
            lab.min_raise_chips, lab.max_raise_chips, lab.pot_chips,
            lab.to_call_chips, NLH_ANCHOR_SPEC,
        )
        for a in lab.action_probs:
            if a.gate == "raise":
                assert bool(grid.legal[a.anchor_k])
                assert int(grid.chips[a.anchor_k]) == lab.max_raise_chips


# --- D14 ----------------------------------------------------------------------


@pytest.mark.parametrize("n,want", [(3, 0), (4, 1), (6, 3)])
def test_multiway_preflop_button_is_n_minus_3(n: int, want: int):
    rep = {
        "status": "ok",
        "notes": [],
        "config": {},
        "root": {
            "street": 0, "pot_bb": 1.5, "effective_stack_bb": 10.0, "board": [],
            "num_seats": n, "bb_chips": 10_000, "sb_chips": 5_000, "ante_chips": 0,
            "root_id": f"pf{n}", "stacks_bb": [10.0] * n,
        },
        "strategy": {
            "root_id": f"pf{n}",
            "infosets": [
                {"infoset_id": "mwpf_p0_pathopen_c12", "actions": ["FOLD", "ALLIN"],
                 "probs": [0.1, 0.9]}
            ],
        },
    }
    lab = strategy_to_labels(rep)[0]
    assert lab.button == want  # Rust seats: BTN n-3, SB n-2, BB n-1


def test_hu_and_postflop_buttons_unchanged():
    hu = strategy_to_labels(_river_report([_iset(["CHECK_CALL", "RAISE_500"], [0.6, 0.4])]))
    assert hu[0].button == 1
    mw = _river_report([_iset(["CHECK_CALL", "RAISE_500"], [0.6, 0.4],
                              stacks_chips=[200_000] * 3, folded=[False] * 3)])
    mw["root"]["num_seats"] = 3
    assert strategy_to_labels(mw)[0].button == 2


# --- D13 (export side) --------------------------------------------------------


def test_export_skips_progress_partial_and_non_ok_reports(tmp_path: Path):
    ok = _river_report([_iset(["CHECK_CALL", "RAISE_500"], [0.6, 0.4])], root_id="done")
    (tmp_path / "done.json").write_text(json.dumps(ok), encoding="utf-8")
    running = _river_report(
        [_iset(["CHECK_CALL", "RAISE_500"], [0.5, 0.5], combo=101)],
        root_id="live", status="running", expl=None, iterations=1,
    )
    (tmp_path / "live.progress.json").write_text(json.dumps(running), encoding="utf-8")
    (tmp_path / "renamed_snapshot.json").write_text(json.dumps(running), encoding="utf-8")
    partial = _river_report(
        [_iset(["CHECK_CALL", "RAISE_500"], [0.5, 0.5], combo=102)],
        root_id="killed", status="partial",
    )
    (tmp_path / "killed.partial.json").write_text(json.dumps(partial), encoding="utf-8")
    (tmp_path / "killed_copy.json").write_text(json.dumps(partial), encoding="utf-8")
    # what the pre-fix overnight resume wrote: status "ok" + the promoted flag
    old_promoted = _river_report(
        [_iset(["CHECK_CALL", "RAISE_500"], [0.5, 0.5], combo=103)], root_id="old_promo"
    )
    old_promoted["promoted_from_progress"] = True
    (tmp_path / "old_promo.json").write_text(json.dumps(old_promoted), encoding="utf-8")

    res = export_dir_detailed(tmp_path, tmp_path / "out" / "labels.jsonl")
    assert res.train_root_ids == ["done"]
    assert {lab.root_name for lab in read_jsonl(res.train_path)} == {"done"}
    assert {s["status"] for s in res.skipped_status} == {
        "running", "partial", "promoted_from_progress"
    }


# --- D8 / F6 at export --------------------------------------------------------


def test_teacher_export_skips_unverified_exploitability(tmp_path: Path):
    isets = [_iset(["CHECK_CALL", "RAISE_500"], [0.6, 0.4])]
    cases = {
        "final": ["expl_kind=infoset_br samples=128"],
        "exact": ["expl_kind=exact_infoset"],
        "budget": ["expl_kind=infoset_br samples=128", "early_stop=time_budget"],
        "poll": ["expl_kind=mc_poll"],
        "target_stop": ["DCFR River HU early stop iter 50"],
        "proxy": ["mc_br_proxy_bb=0.3"],
    }
    for i, (rid, notes) in enumerate(cases.items()):
        rep = _river_report(isets, root_id=rid, notes=notes, board=[i, 5 + i, 20, 30, 40])
        (tmp_path / f"{rid}.json").write_text(json.dumps(rep), encoding="utf-8")
    res = export_teacher_dir(tmp_path, tmp_path / "o" / "t.jsonl", holdout_frac=0.0)
    assert res.train_root_ids == ["exact", "final"]
    reasons = {s["root_id"]: s["reason"] for s in res.skipped_expl}
    assert reasons["budget"] == "expl_unverified:early_stop=time_budget"
    assert reasons["poll"] == "expl_unverified:expl_kind=mc_poll"
    assert reasons["target_stop"].startswith("expl_unverified:early_stop")
    assert reasons["proxy"] == "expl_unverified:expl_kind_missing"
    # Provenance rides on every record (the checkpoint derives its meta from it).
    lab = next(iter(read_jsonl(res.train_path)))
    assert lab.notes["expl_verified"] is True
    assert lab.notes["teacher_max_expl_bb"] == 1.0
    assert lab.notes["exploitability_bb"] == 0.4
    # Opt-out exports them, flagged unverified on the records.
    res2 = export_teacher_dir(
        tmp_path, tmp_path / "o2" / "t.jsonl", holdout_frac=0.0,
        require_verified_expl=False,
    )
    assert "budget" in res2.train_root_ids
    flags = {l.root_name: l.notes["expl_verified"] for l in read_jsonl(res2.train_path)}
    assert flags["budget"] is False and flags["final"] is True


# --- F10 ----------------------------------------------------------------------


def test_same_legacy_root_id_on_two_boards_is_disambiguated(tmp_path: Path):
    a = _river_report([_iset(["CHECK_CALL", "RAISE_500"], [0.6, 0.4])], root_id="s3_pot10_eff20")
    b = _river_report(
        [_iset(["CHECK_CALL", "RAISE_500"], [0.3, 0.7], board=[1, 6, 11, 16, 21])],
        root_id="s3_pot10_eff20", board=[1, 6, 11, 16, 21],
    )
    (tmp_path / "a.json").write_text(json.dumps(a), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(b), encoding="utf-8")
    res = export_dir_detailed(tmp_path, tmp_path / "o" / "l.jsonl")
    assert len(res.train_root_ids) == 2
    assert all(r.startswith("s3_pot10_eff20-") for r in res.train_root_ids)
    assert list(res.root_id_collisions) == ["s3_pot10_eff20"]
    assert {l.root_name for l in read_jsonl(res.train_path)} == set(res.train_root_ids)


def test_duplicate_solves_of_one_root_are_exported_once(tmp_path: Path):
    isets = [_iset(["CHECK_CALL", "RAISE_500"], [0.6, 0.4])]
    (tmp_path / "short.json").write_text(
        json.dumps(_river_report(isets, iterations=100)), encoding="utf-8"
    )
    (tmp_path / "long.json").write_text(
        json.dumps(_river_report(isets, iterations=5000)), encoding="utf-8"
    )
    res = export_dir_detailed(tmp_path, tmp_path / "o" / "l.jsonl")
    assert res.n_train == 1 and res.train_root_ids == ["river_t"]
    assert [Path(d["path"]).name for d in res.skipped_dupes] == ["short.json"]
    assert next(iter(read_jsonl(res.train_path))).notes["iterations"] == 5000


def test_aborted_export_leaves_no_partial_jsonl(tmp_path: Path):
    good = _river_report([_iset(["CHECK_CALL", "RAISE_500"], [0.6, 0.4])], root_id="a_good")
    bad = _river_report(
        [_iset(["FOLD", "CHECK_CALL", "RAISE_500"], [0.2, 0.5, 0.3], combo=7,
               to_call_chips=50_000, max_raise_chips=0, min_raise_chips=0)],
        root_id="z_bad", board=[2, 7, 12, 17, 22],
    )
    (tmp_path / "a.json").write_text(json.dumps(good), encoding="utf-8")
    (tmp_path / "z.json").write_text(json.dumps(bad), encoding="utf-8")
    out = tmp_path / "o" / "labels.jsonl"
    with pytest.raises(IllegalTeacherMassError, match="z_bad"):
        export_dir_detailed(tmp_path, out)
    assert not out.exists()
    assert list(out.parent.glob("*.tmp")) == []


# --- D16 ----------------------------------------------------------------------


@pytest.mark.skipif(not rust_cfr_available(), reason="extension not built")
def test_turn_root_later_street_rows_are_excluded_loudly(capsys):
    root = RootSpec(
        street=2, pot_bb=10.0, effective_stack_bb=20.0, board=[5, 7, 30, 44],
        raise_sizes_pm=[500, 1000], root_id="turnroot",
    )
    rep = solve(
        root,
        SolveConfig.teacher(max_iterations=60, seed=1, target_exploitability_bb=0.0,
                            thread_num=1, poll_every=10**6),
    )
    labs = strategy_to_labels(rep.as_dict(), min_visit_mass=0.0)
    turn = [l for l in labs if l.street == 2][:150]
    river = [l for l in labs if l.street == 3][:150]
    assert turn and river
    stats = ObsSynthesisStats()
    rows = labels_to_supervised_rows(turn + river, stats=stats)
    assert len(rows) == len(turn)  # root-street rows only
    assert {r.street for r in rows} == {2}
    assert stats.kinds == {"engine": len(turn), OBS_KIND_LOSSY: len(river)}
    assert stats.excluded_lossy == len(river)
    assert stats.lossy_roots == {"turnroot": len(river)}
    assert "WARNING lossy synthetic obs" in capsys.readouterr().out
    # Opt-in keeps them — and says what they are.
    rows_all = labels_to_supervised_rows(turn + river, include_lossy_obs=True)
    assert len(rows_all) == len(turn) + len(river)
    assert {r.prov.obs_form for r in rows_all} == {"canonical", OBS_KIND_LOSSY}


def test_label_the_engine_cannot_rebuild_is_lossy_not_silent(capsys):
    """Hole card on the board: the engine refuses the node."""
    lab = make_smoke_label(trash_fold=False)
    lab.hero_hole = [int(lab.board[0]), 50]
    stats = ObsSynthesisStats()
    assert labels_to_supervised_rows([lab], stats=stats) == []
    assert stats.n_lossy == 1 and stats.excluded_lossy == 1
    assert "WARNING lossy synthetic obs" in capsys.readouterr().out
