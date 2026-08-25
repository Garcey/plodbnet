"""Teacher iso policy: raw holes in train, raw 52-hot at serve."""

from __future__ import annotations

import numpy as np

from plo5bp.gto.cfr_api import SolveConfig, apply_teacher_iso_policy
from plo5bp.gto.cfr_batch import expand_river_grid, run_batch
from plo5bp.gto.cfr_export import (
    REJECT_ISO_WITHOUT_RAW,
    ExportStats,
    _hole_from_private,
    strategy_to_labels,
)
from plo5bp.gto.iso import TEACHER_USE_ISOMORPHISM, iso_combo_id
from plo5bp.gto.obs_from_label import obs_from_label
from plo5bp.gto.preflop_class import cards_to_combo, combo_to_cards


def _raw_not_iso_pair() -> tuple[list[int], int, int]:
    """Board diamonds-first remaps club hole → diamond hole (iso != raw)."""
    board = [1, 5, 9, 21, 25]  # 2d 3d 4d + two more diamonds
    raw = cards_to_combo(12 * 4 + 0, 11 * 4 + 0)  # Ac Kc
    iso = iso_combo_id(raw, board)
    assert iso != raw
    return board, raw, iso


def test_iso_combo_id_matches_rust_orbit():
    # Rust card_abs iso_tests: (Ac,Kc) on club board == (Ad,Kd) on diamond board
    board = [0, 4, 8]
    board_rot = [1, 5, 9]
    hand = cards_to_combo(12 * 4 + 0, 11 * 4 + 0)  # Ac Kc
    hand_rot = cards_to_combo(12 * 4 + 1, 11 * 4 + 1)  # Ad Kd
    assert iso_combo_id(hand, board) == iso_combo_id(hand_rot, board_rot)
    _raw_not_iso_pair()


def test_teacher_batch_forces_iso_off(tmp_path):
    assert TEACHER_USE_ISOMORPHISM is False
    jobs = expand_river_grid(n_roots=2, seed=0, iters=5)
    assert all(j.config.use_isomorphism is False for j in jobs)
    # Even if a caller turns iso on, run_batch snaps it off
    jobs[0].config.use_isomorphism = True
    man = run_batch(jobs, tmp_path / "b", dry_run=True)
    assert jobs[0].config.use_isomorphism is False
    plan = (tmp_path / "b" / "plan.json").read_text(encoding="utf-8")
    assert '"use_isomorphism": false' in plan
    assert man.jobs


def test_export_uses_raw_combo_not_iso_id():
    board, raw, iso = _raw_not_iso_pair()
    raw_hole = combo_to_cards(raw)
    iso_hole = combo_to_cards(iso)

    stats = ExportStats()
    labs = strategy_to_labels(
        {
            "status": "ok",
            "root": {
                "street": 3,
                "pot_bb": 10.0,
                "effective_stack_bb": 20.0,
                "board": board,
                "num_seats": 2,
                "bb_chips": 10_000,
                "root_id": "iso_poison",
            },
            "strategy": {
                "infosets": [
                    {
                        "infoset_id": f"p0_h0_c{iso}",
                        "actions": ["CHECK_CALL", "RAISE_500"],
                        "probs": [0.6, 0.4],
                        "schema_version": 2,
                        "street": 3,
                        "actor": 0,
                        "pot_chips": 100_000,
                        "to_call_chips": 0,
                        "min_raise_chips": 10_000,
                        "max_raise_chips": 200_000,
                        "stacks_chips": [200_000, 200_000],
                        "board": board,
                        "path": [],
                        "private_kind": "combo",
                        "private_id": iso,
                        "raw_combo": raw,
                        "iso_id": iso,
                        "visit_mass": 2.0,
                    }
                ]
            },
        },
        stats=stats,
    )
    assert len(labs) == 1
    assert labs[0].hero_hole == raw_hole
    assert labs[0].hero_hole != iso_hole
    assert labs[0].notes["iso_id"] == iso
    assert labs[0].notes["raw_combo"] == raw


def test_export_drops_iso_id_without_raw_combo():
    board = [0, 4, 8, 20, 24]
    iso = iso_combo_id(cards_to_combo(50, 51), board)
    stats = ExportStats()
    labs = strategy_to_labels(
        {
            "status": "ok",
            "root": {
                "street": 3,
                "pot_bb": 10.0,
                "effective_stack_bb": 20.0,
                "board": board,
                "num_seats": 2,
                "bb_chips": 10_000,
                "root_id": "iso_bare",
            },
            "strategy": {
                "infosets": [
                    {
                        "infoset_id": f"p0_h0_c{iso}",
                        "actions": ["CHECK_CALL"],
                        "probs": [1.0],
                        "schema_version": 2,
                        "to_call_chips": 0,
                        "pot_chips": 100_000,
                        "stacks_chips": [200_000, 200_000],
                        "board": board,
                        "private_kind": "combo",
                        "private_id": iso,
                        "raw_combo": None,
                        "iso_id": iso,
                        "visit_mass": 1.0,
                    }
                ]
            },
        },
        stats=stats,
    )
    assert labs == []
    assert stats.rejected.get(REJECT_ISO_WITHOUT_RAW) == 1
    # Decoder itself must not emit the iso hole
    assert _hole_from_private(iso, "combo", board=board, iso_id=iso) == []


def test_train_obs_differs_for_iso_canonical_vs_raw():
    """Serve is raw 52-hot: iso-canonical cards are a different obs."""
    from plo5bp.gto.labels import LABEL_SCHEMA_VERSION, ActionProb, LabelRecord

    board, raw, iso = _raw_not_iso_pair()

    def _lab(hole: list[int]) -> LabelRecord:
        return LabelRecord(
            schema_version=LABEL_SCHEMA_VERSION,
            source="rust_cfr",
            root_name="iso_obs",
            num_seats=2,
            street=3,
            spr=5.0,
            pot_chips=100_000,
            to_call_chips=0,
            min_raise_chips=10_000,
            max_raise_chips=200_000,
            hero_seat=0,
            button=1,
            hero_hole=hole,
            board=board,
            stacks_chips=[200_000, 200_000],
            gate_probs=[0.0, 0.7, 0.3],
            action_probs=[ActionProb("check_call", None, None, 0, 0.7)],
            notes={"path": "", "path_tokens": []},
        )

    raw_obs = obs_from_label(_lab(combo_to_cards(raw)))
    iso_obs = obs_from_label(_lab(combo_to_cards(iso)))
    assert raw_obs is not None and iso_obs is not None
    assert not np.array_equal(raw_obs, iso_obs)
    for c in combo_to_cards(raw):
        assert raw_obs[int(c)] == 1.0
        assert iso_obs[int(c)] != 1.0 or c in combo_to_cards(iso)


def test_teacher_solve_config_factory():
    cfg = SolveConfig.teacher(max_iterations=10)
    assert cfg.use_isomorphism is False
    other = SolveConfig(use_isomorphism=True)
    apply_teacher_iso_policy(other)
    assert other.use_isomorphism is False
