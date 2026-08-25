"""CFR export → obs → PolicyNet train pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from plo5bp.encoding_nlh import OBS_DIM_NLH
from plo5bp.gto.cfr_api import RootSpec, SolveConfig, rust_cfr_available, solve
from plo5bp.gto.cfr_export import (
    REJECT_HOLE_DECODE,
    REJECT_ILLEGAL_FOLD,
    REJECT_INCONSISTENT_PUBLIC,
    REJECT_UNUSED_UNIFORM,
    ExportStats,
    export_dir,
    pot_frac_pm_to_anchor,
    reject_reason,
    strategy_to_labels,
)
from plo5bp.gto.dataset import load_label_shard_rows, rows_from_label_records
from plo5bp.gto.obs_from_label import labels_to_supervised_rows, obs_from_label
from plo5bp.gto.preflop_class import (
    NUM_PREFLOP_CLASSES,
    combo_to_cards,
    preflop_class_from_cards,
    preflop_class_from_id,
    preflop_class_label,
    representative_hole,
)
from plo5bp.gto.train import TrainConfig, train_policy_net
from plo5bp.sizing import NLH_ANCHOR_SPEC


def test_preflop_class_roundtrip():
    for cid in range(NUM_PREFLOP_CLASSES):
        hi, lo, suited = preflop_class_from_id(cid)
        hole = representative_hole(cid)
        assert len(hole) == 2
        assert hole[0] != hole[1]
        back = preflop_class_from_cards(hole[0], hole[1])
        assert back == cid, f"class {cid} hole {hole} → {back}"
        assert preflop_class_label(cid)


def test_combo_decode_matches_export_legacy():
    for cid in (0, 1, 50, 100, 500, 1325):
        c0, c1 = combo_to_cards(cid)
        assert 0 <= c0 < c1 < 52


def test_pot_frac_maps_to_nlh_anchor():
    # 500‰ = 50% pot → near 500 in fracs
    k = pot_frac_pm_to_anchor(500)
    assert 0 <= k < NLH_ANCHOR_SPEC.count
    k_ai = pot_frac_pm_to_anchor(None, is_allin=True)
    assert k_ai == NLH_ANCHOR_SPEC.count - 1


def test_strategy_to_labels_pushfold_full(tmp_path: Path):
    # Minimal synthetic SolveReport (mwpf ids)
    rep = {
        "status": "ok",
        "iterations_run": 10,
        "exploitability_bb": 1.0,
        "notes": ["mc_br_proxy_bb=1.0", "tree=push_fold"],
        "root": {
            "street": 0,
            "pot_bb": 1.5,
            "effective_stack_bb": 10.0,
            "board": [],
            "num_seats": 4,
            "bb_chips": 10_000,
            "sb_chips": 5_000,
            "ante_chips": 0,
            "stacks_bb": [10.0] * 4,
            "root_id": "pf4_test",
            "raise_sizes_pm": [],
            "allin_atom": True,
        },
        "strategy": {
            "root_id": "pf4_test",
            "infosets": [
                {
                    "infoset_id": "mwpf_p0_pathopen_c12",  # AA class
                    "actions": ["ALLIN"],
                    "probs": [1.0],
                },
                {
                    "infoset_id": "mwpf_p1_pathF_c0",  # 22 facing fold
                    "actions": ["FOLD", "ALLIN"],
                    "probs": [0.3, 0.7],
                },
                {
                    "infoset_id": "mwpf_p3_pathAI_c91",  # BB vs jam, 32o
                    "actions": ["FOLD", "ALLIN"],
                    "probs": [0.9, 0.1],
                },
            ],
        },
        "config": {},
    }
    labels = strategy_to_labels(rep, source="rust_cfr_test", max_infosets=None)
    assert len(labels) == 3
    # Full export (no 64 cap)
    assert all(len(lab.hero_hole) == 2 for lab in labels)
    # AA open: all-in → raise gate, all-in anchor
    open_lab = labels[0]
    assert open_lab.gate_probs[2] > 0.9  # raise
    assert any(a.gate == "raise" and a.anchor_k == NLH_ANCHOR_SPEC.count - 1 for a in open_lab.action_probs)
    # Facing jam: fold legal
    jam_lab = labels[2]
    assert jam_lab.to_call_chips > 0
    assert jam_lab.gate_probs[0] > 0.5
    assert jam_lab.notes.get("mc_br_proxy") is True
    assert jam_lab.notes.get("class_id") == 91

    # Obs synthesis
    obs = obs_from_label(open_lab)
    assert obs is not None
    assert obs.shape == (OBS_DIM_NLH,)
    assert obs.dtype == np.float32

    rows = rows_from_label_records(labels)
    assert len(rows) == 3
    assert rows[0].obs.shape == (OBS_DIM_NLH,)


def test_export_dir_no_default_cap(tmp_path: Path):
    # 100 fake infosets — must all export when max is None
    infosets = [
        {
            "infoset_id": f"mwpf_p0_pathopen_c{i % 169}",
            "actions": ["FOLD", "ALLIN"],
            "probs": [0.3, 0.7],
        }
        for i in range(100)
    ]
    rep = {
        "status": "ok",
        "notes": ["mc_br_proxy_bb=0"],
        "root": {
            "street": 0,
            "pot_bb": 1.5,
            "effective_stack_bb": 10.0,
            "board": [],
            "num_seats": 2,
            "bb_chips": 10_000,
            "sb_chips": 5_000,
            "ante_chips": 0,
            "root_id": "cap_test",
            "stacks_bb": [10.0, 10.0],
        },
        "strategy": {"root_id": "cap_test", "infosets": infosets},
        "config": {},
    }
    strat = tmp_path / "s.json"
    strat.write_text(json.dumps(rep), encoding="utf-8")
    out = tmp_path / "labels.jsonl"
    n = export_dir(strat, out, max_infosets_per_file=None)
    assert n == 100
    # Old default of 64 would fail this


@pytest.mark.skipif(not rust_cfr_available(), reason="extension not built")
def test_live_pushfold_export_and_train(tmp_path: Path):
    root = RootSpec.preflop_pushfold(num_seats=3, stack_bb=10.0, ante_chips=0)
    rep = solve(root, SolveConfig(max_iterations=40, algorithm="mccfr_es", seed=2))
    assert rep.status == "ok"
    labels = strategy_to_labels(rep.as_dict(), max_infosets=None)
    has_vm = any(
        i.get("visit_mass") is not None for i in (rep.strategy.get("infosets") or [])
    )
    # Without visit_mass, first-visit 0.5/0.5 PF nodes look unused and drop.
    assert len(labels) > (50 if has_vm else 5)
    # Most should decode class → hole
    with_hole = sum(1 for l in labels if len(l.hero_hole) == 2)
    assert with_hole > 40
    # Anchors set on ALLIN
    for lab in labels[:20]:
        for a in lab.action_probs:
            if a.gate == "raise":
                assert a.anchor_k is not None

    out_jsonl = tmp_path / "pf.jsonl"
    export_dir(
        # write report then export
        tmp_path,
        out_jsonl,
        max_infosets_per_file=None,
    )
    # write solve report into tmp
    rep.write_json(tmp_path / "solve.json")
    n = export_dir(tmp_path / "solve.json", out_jsonl, max_infosets_per_file=None)
    assert n == len(labels) or n > 50

    rows = load_label_shard_rows(out_jsonl)
    assert len(rows) > 40
    ckpt = tmp_path / "pol.pt"
    r = train_policy_net(
        rows[: min(256, len(rows))],
        ckpt,
        cfg=TrainConfig(
            hidden_dim=64,
            num_layers=1,
            epochs=2,
            batch_size=32,
            device="cpu",
            seed=0,
            log_every=1000,
        ),
        meta={"source": "rust_cfr", "pipeline": "test"},
    )
    assert r.n_train > 0
    assert ckpt.is_file()
    assert np.isfinite(r.final_loss)


def test_river_combo_label_anchor_map():
    rep = {
        "status": "ok",
        "notes": [],
        "root": {
            "street": 3,
            "pot_bb": 10.0,
            "effective_stack_bb": 20.0,
            "board": [0, 5, 10, 15, 20],
            "num_seats": 2,
            "bb_chips": 10_000,
            "sb_chips": 5_000,
            "ante_chips": 5_000,
            "root_id": "river_t",
        },
        "strategy": {
            "root_id": "river_t",
            "infosets": [
                {
                    "infoset_id": "p0_h0_c100",  # combo 100
                    "actions": ["CHECK_CALL", "RAISE_500", "ALLIN"],
                    "probs": [0.2, 0.5, 0.3],
                }
            ],
        },
        "config": {},
    }
    labs = strategy_to_labels(rep, max_infosets=None)
    assert len(labs) == 1
    lab = labs[0]
    assert len(lab.hero_hole) == 2
    # combo 100 is NOT preflop class 100 (65o)
    assert lab.hero_hole == combo_to_cards(100)
    assert lab.notes.get("private_kind") == "combo"
    raise_aps = [a for a in lab.action_probs if a.gate == "raise"]
    assert all(a.anchor_k is not None for a in raise_aps)
    assert abs(sum(lab.gate_probs) - 1.0) < 1e-6
    rows = rows_from_label_records(labs)
    assert len(rows) == 1
    assert rows[0].anchor_probs.sum() > 0.99


def test_legacy_river_combo_below_169_is_not_class():
    """Old dumps: p0_hHASH_c1 must decode as combo 1, not class 33."""
    from plo5bp.gto.preflop_class import representative_hole

    rep = {
        "status": "ok",
        "notes": [],
        "root": {
            "street": 3,
            "pot_bb": 10.0,
            "effective_stack_bb": 20.0,
            "board": [12, 28, 38, 41, 45],
            "num_seats": 2,
            "bb_chips": 10_000,
            "sb_chips": 5_000,
            "ante_chips": 5_000,
            "root_id": "legacy_c1",
        },
        "strategy": {
            "root_id": "legacy_c1",
            "infosets": [
                {
                    "infoset_id": "p0_h12222850381629986102_c1",
                    "actions": ["CHECK_CALL", "RAISE_500"],
                    "probs": [0.4, 0.6],
                }
            ],
        },
        "config": {},
    }
    labs = strategy_to_labels(rep)
    assert len(labs) == 1
    assert labs[0].notes["private_kind"] == "combo"
    assert labs[0].hero_hole == combo_to_cards(1)
    assert labs[0].hero_hole != representative_hole(1)


def test_dump_v2_fields_are_authoritative():
    """New solves: pot/to_call/path/kind come from the dump, not heuristics."""
    combo = 50
    hole = combo_to_cards(combo)
    rep = {
        "status": "ok",
        "notes": [],
        "root": {
            "street": 3,
            "pot_bb": 10.0,
            "effective_stack_bb": 50.0,
            "board": [30, 31, 32, 33, 34],
            "num_seats": 2,
            "bb_chips": 10_000,
            "root_id": "dump_v2",
        },
        "strategy": {
            "schema_version": 2,
            "root_id": "dump_v2",
            "infosets": [
                {
                    "infoset_id": "p0_h999_c50",
                    "actions": ["FOLD", "CHECK_CALL"],
                    "probs": [0.2, 0.8],
                    "schema_version": 2,
                    "street": 3,
                    "actor": 0,
                    "pot_chips": 123456,
                    "to_call_chips": 25000,
                    "min_raise_chips": 25000,
                    "max_raise_chips": 400000,
                    "stacks_chips": [400000, 400000],
                    "folded": [False, False],
                    "board": [30, 31, 32, 33, 34],
                    "path": ["RAISE_500"],
                    "private_kind": "combo",
                    "private_id": combo,
                    "raw_combo": combo,
                    "iso_id": None,
                }
            ],
        },
        "config": {},
    }
    labs = strategy_to_labels(rep)
    assert len(labs) == 1
    lab = labs[0]
    assert lab.to_call_chips == 25000  # not invented bb=10000
    assert lab.pot_chips == 123456
    assert lab.hero_hole == hole
    assert lab.notes["private_kind"] == "combo"
    assert lab.notes["path"] == "RAISE_500"
    assert lab.gate_probs[0] > 0  # fold legal because dump to_call > 0


def test_dump_encode_golden_path_stub():
    """Dump → obs_from_label shape. Bit-exact vs live engine encode is later."""
    combo = 200
    hole = combo_to_cards(combo)
    lab = strategy_to_labels(
        {
            "status": "ok",
            "root": {
                "street": 3,
                "pot_bb": 10.0,
                "effective_stack_bb": 20.0,
                "board": [30, 31, 32, 33, 34],
                "num_seats": 2,
                "bb_chips": 10_000,
                "root_id": "golden_stub",
            },
            "strategy": {
                "schema_version": 2,
                "infosets": [
                    {
                        "infoset_id": "p0_h0_c200",
                        "actions": ["CHECK_CALL", "RAISE_500"],
                        "probs": [0.7, 0.3],
                        "schema_version": 2,
                        "street": 3,
                        "actor": 0,
                        "pot_chips": 100000,
                        "to_call_chips": 0,
                        "min_raise_chips": 10000,
                        "max_raise_chips": 200000,
                        "stacks_chips": [200000, 200000],
                        "folded": [False, False],
                        "board": [30, 31, 32, 33, 34],
                        "path": [],
                        "private_kind": "combo",
                        "private_id": combo,
                        "raw_combo": combo,
                    }
                ],
            },
        }
    )[0]
    assert lab.hero_hole == hole
    assert lab.to_call_chips == 0
    obs = obs_from_label(lab)
    assert obs is not None
    assert obs.shape == (OBS_DIM_NLH,)
    for c in hole:
        assert obs[int(c)] == 1.0
    rows = labels_to_supervised_rows([lab])
    assert len(rows) == 1
    # Engine fold mask: legal iff to_call > 0
    assert bool(rows[0].gate_mask[0]) is False
    assert lab.gate_probs[0] == 0.0


def test_gate_unused_uniform_05_05_dropped():
    stats = ExportStats()
    labs = strategy_to_labels(
        {
            "status": "ok",
            "root": {
                "street": 3,
                "pot_bb": 10.0,
                "effective_stack_bb": 20.0,
                "board": [30, 31, 32, 33, 34],
                "num_seats": 2,
                "bb_chips": 10_000,
                "root_id": "unused",
            },
            "strategy": {
                "infosets": [
                    {
                        "infoset_id": "p0_h1_c200",
                        "actions": ["FOLD", "CHECK_CALL"],
                        "probs": [0.5, 0.5],
                        "schema_version": 2,
                        "to_call_chips": 10_000,
                        "pot_chips": 100_000,
                        "stacks_chips": [200_000, 200_000],
                        "private_kind": "combo",
                        "raw_combo": 200,
                    }
                ]
            },
        },
        stats=stats,
    )
    assert labs == []
    assert stats.rejected.get(REJECT_UNUSED_UNIFORM) == 1


def test_gate_visit_mass_keeps_first_visit_uniform():
    """visit_mass>0 means touched; 0.5/0.5 after a real visit is kept."""
    stats = ExportStats()
    labs = strategy_to_labels(
        {
            "status": "ok",
            "root": {
                "street": 3,
                "pot_bb": 10.0,
                "effective_stack_bb": 20.0,
                "board": [30, 31, 32, 33, 34],
                "num_seats": 2,
                "bb_chips": 10_000,
                "root_id": "visited",
            },
            "strategy": {
                "infosets": [
                    {
                        "infoset_id": "p0_h1_c200",
                        "actions": ["FOLD", "CHECK_CALL"],
                        "probs": [0.5, 0.5],
                        "schema_version": 2,
                        "to_call_chips": 10_000,
                        "pot_chips": 100_000,
                        "stacks_chips": [200_000, 200_000],
                        "private_kind": "combo",
                        "raw_combo": 200,
                        "visit_mass": 1.0,
                    }
                ]
            },
        },
        stats=stats,
    )
    assert len(labs) == 1
    assert labs[0].to_call_chips == 10_000
    assert stats.kept == 1


def test_gate_illegal_fold_when_to_call_zero():
    stats = ExportStats()
    labs = strategy_to_labels(
        {
            "status": "ok",
            "root": {
                "street": 3,
                "pot_bb": 10.0,
                "effective_stack_bb": 20.0,
                "board": [30, 31, 32, 33, 34],
                "num_seats": 2,
                "bb_chips": 10_000,
                "root_id": "bad_fold",
            },
            "strategy": {
                "infosets": [
                    {
                        "infoset_id": "p0_h1_c200",
                        "actions": ["FOLD", "CHECK_CALL"],
                        "probs": [0.2, 0.8],
                        "schema_version": 2,
                        "to_call_chips": 0,
                        "pot_chips": 100_000,
                        "stacks_chips": [200_000, 200_000],
                        "private_kind": "combo",
                        "raw_combo": 200,
                        "visit_mass": 3.0,
                    }
                ]
            },
        },
        stats=stats,
    )
    assert labs == []
    assert stats.rejected.get(REJECT_ILLEGAL_FOLD) == 1


def test_gate_v2_facing_bet_fold_kept():
    stats = ExportStats()
    labs = strategy_to_labels(
        {
            "status": "ok",
            "root": {
                "street": 3,
                "pot_bb": 10.0,
                "effective_stack_bb": 20.0,
                "board": [30, 31, 32, 33, 34],
                "num_seats": 2,
                "bb_chips": 10_000,
                "root_id": "good_fold",
            },
            "strategy": {
                "infosets": [
                    {
                        "infoset_id": "p0_h1_c200",
                        "actions": ["FOLD", "CHECK_CALL"],
                        "probs": [0.2, 0.8],
                        "schema_version": 2,
                        "to_call_chips": 25_000,
                        "pot_chips": 123_456,
                        "min_raise_chips": 25_000,
                        "max_raise_chips": 200_000,
                        "stacks_chips": [200_000, 200_000],
                        "private_kind": "combo",
                        "raw_combo": 200,
                        "visit_mass": 4.0,
                    }
                ]
            },
        },
        stats=stats,
    )
    assert len(labs) == 1
    assert labs[0].to_call_chips == 25_000
    assert labs[0].gate_probs[0] > 0
    rows = labels_to_supervised_rows(labs)
    assert bool(rows[0].gate_mask[0]) is True  # fold legal iff to_call > 0


def test_gate_inconsistent_public():
    stats = ExportStats()
    # v2 missing to_call
    strategy_to_labels(
        {
            "status": "ok",
            "root": {
                "street": 3,
                "pot_bb": 10.0,
                "effective_stack_bb": 20.0,
                "board": [30, 31, 32, 33, 34],
                "num_seats": 2,
                "bb_chips": 10_000,
                "root_id": "miss_tc",
            },
            "strategy": {
                "schema_version": 2,
                "infosets": [
                    {
                        "infoset_id": "p0_h1_c200",
                        "actions": ["CHECK_CALL"],
                        "probs": [1.0],
                        "schema_version": 2,
                        "pot_chips": 100_000,
                        "private_kind": "combo",
                        "raw_combo": 200,
                        "visit_mass": 1.0,
                    }
                ],
            },
        },
        stats=stats,
    )
    assert stats.rejected.get(REJECT_INCONSISTENT_PUBLIC) == 1
    # to_call > hero stack
    stats2 = ExportStats()
    strategy_to_labels(
        {
            "status": "ok",
            "root": {
                "street": 3,
                "pot_bb": 10.0,
                "effective_stack_bb": 20.0,
                "board": [30, 31, 32, 33, 34],
                "num_seats": 2,
                "bb_chips": 10_000,
                "root_id": "tc_gt_stack",
            },
            "strategy": {
                "infosets": [
                    {
                        "infoset_id": "p0_h1_c200",
                        "actions": ["FOLD", "CHECK_CALL"],
                        "probs": [0.2, 0.8],
                        "schema_version": 2,
                        "to_call_chips": 500_000,
                        "pot_chips": 100_000,
                        "stacks_chips": [10_000, 10_000],
                        "private_kind": "combo",
                        "raw_combo": 200,
                        "visit_mass": 2.0,
                    }
                ]
            },
        },
        stats=stats2,
    )
    assert stats2.rejected.get(REJECT_INCONSISTENT_PUBLIC) == 1


def test_gate_hole_decode_when_required():
    stats = ExportStats()
    labs = strategy_to_labels(
        {
            "status": "ok",
            "root": {
                "street": 1,
                "pot_bb": 8.0,
                "effective_stack_bb": 20.0,
                "board": [0, 5, 10],
                "num_seats": 2,
                "bb_chips": 10_000,
                "root_id": "ochs",
            },
            "strategy": {
                "infosets": [
                    {
                        "infoset_id": "p0_h1_c2000003",
                        "actions": ["CHECK_CALL", "RAISE_500"],
                        "probs": [0.6, 0.4],
                        "schema_version": 2,
                        "to_call_chips": 0,
                        "pot_chips": 80_000,
                        "stacks_chips": [200_000, 200_000],
                        "private_kind": "ochs_bucket",
                        "private_id": 2_000_003,
                        "raw_combo": None,
                        "visit_mass": 2.0,
                    }
                ]
            },
        },
        require_hole=True,
        stats=stats,
    )
    assert labs == []
    assert stats.rejected.get(REJECT_HOLE_DECODE) == 1


def test_reject_reason_helper():
    assert (
        reject_reason(
            actions=["FOLD", "CHECK_CALL"],
            probs=[0.5, 0.5],
            to_call=10_000,
            pot_chips=100_000,
            hero_stack=200_000,
            visit_mass=None,
            v2_missing_to_call=False,
            hole=[1, 2],
            require_hole=False,
        )
        == REJECT_UNUSED_UNIFORM
    )
    assert (
        reject_reason(
            actions=["FOLD", "CHECK_CALL"],
            probs=[0.2, 0.8],
            to_call=0,
            pot_chips=100_000,
            hero_stack=200_000,
            visit_mass=1.0,
            v2_missing_to_call=False,
            hole=[1, 2],
            require_hole=False,
        )
        == REJECT_ILLEGAL_FOLD
    )


@pytest.mark.skipif(not rust_cfr_available(), reason="extension not built")
def test_live_river_emits_dump_schema():
    root = RootSpec.river_hu(
        [0, 5, 10, 15, 20],
        pot_bb=6.0,
        effective_stack_bb=12.0,
        size_preset="micro",
    )
    rep = solve(
        root,
        SolveConfig(max_iterations=40, seed=3, target_exploitability_bb=0.0),
    )
    assert rep.status == "ok"
    isets = rep.strategy["infosets"]
    dumped = [i for i in isets if i.get("schema_version") == 2 or i.get("private_kind")]
    if not dumped:
        pytest.skip("rebuild extension (maturin develop --release) for dump schema")
    assert dumped
    for i in dumped:
        assert i["private_kind"] == "combo"
        tc = int(i["to_call_chips"])
        has_fold = any(str(a).upper() == "FOLD" for a in i["actions"])
        assert has_fold == (tc > 0)
        raw = i.get("raw_combo")
        if raw is None or int(raw) >= 169:
            continue
        vm = i.get("visit_mass")
        if vm is not None and float(vm) <= 0:
            continue
        labs = strategy_to_labels(
            {**rep.as_dict(), "strategy": {"infosets": [i], "root_id": "x"}},
        )
        if not labs:
            continue
        assert labs[0].hero_hole == combo_to_cards(int(raw))
        assert labs[0].notes["private_kind"] == "combo"
        break
