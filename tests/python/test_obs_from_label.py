"""Dump → live-matching NLH obs (Step 3)."""

from __future__ import annotations

import numpy as np
import pytest

from plo5bp.encoding_nlh import OBS_DIM_NLH, encode_observation_nlh
from plo5bp.gto.cfr_api import rust_cfr_available
from plo5bp.gto.labels import LABEL_SCHEMA_VERSION, ActionProb, LabelRecord
from plo5bp.gto.obs_from_label import (
    encode_live_engine,
    live_config_from_label,
    obs_from_label,
    reconstruct_live_engine,
)
from plo5bp.gto.preflop_class import combo_to_cards


def _engine_has_cfr_node() -> bool:
    try:
        from plo5bp._engine import GameState

        return hasattr(GameState, "reset_nlh_cfr_node")
    except Exception:
        return False


def _river_label(*, path: list[str], hero_seat: int, hole: list[int] | None = None) -> LabelRecord:
    hole = hole or combo_to_cards(200)  # does not collide with board 30..34
    return LabelRecord(
        schema_version=LABEL_SCHEMA_VERSION,
        source="rust_cfr",
        root_name="golden_river",
        num_seats=2,
        street=3,
        spr=5.0,
        pot_chips=100_000,
        to_call_chips=0 if not path else 50_000,
        min_raise_chips=10_000,
        max_raise_chips=500_000,
        hero_seat=hero_seat,
        button=1,
        hero_hole=hole,
        board=[30, 31, 32, 33, 34],
        stacks_chips=[500_000, 500_000],
        gate_probs=[0.0, 0.7, 0.3] if not path else [0.2, 0.8, 0.0],
        action_probs=[
            ActionProb("check_call", None, None, 0, 0.7),
            ActionProb("raise", 6, 1.0, 100_000, 0.3),
        ],
        notes={
            "path": ",".join(path),
            "path_tokens": list(path),
            "bb_chips": 10_000,
            "root_pot_chips": 100_000,
            "root_stacks_chips": [500_000, 500_000],
            "dump_schema": 2,
            "private_kind": "combo",
        },
    )


@pytest.mark.skipif(not _engine_has_cfr_node(), reason="rebuild: reset_nlh_cfr_node")
def test_reconstruct_root_empty_path():
    lab = _river_label(path=[], hero_seat=0)
    gs = reconstruct_live_engine(lab)
    assert gs is not None
    assert gs.current_actor() == 0
    raw = dict(gs.observation_dict())
    assert raw["history"] == [] or list(raw["history"]) == []
    assert int(raw["pot"]) == 100_000
    assert int(raw["bet_to_call"]) == 0
    assert int(raw["street"]) == 3


@pytest.mark.skipif(not _engine_has_cfr_node(), reason="rebuild: reset_nlh_cfr_node")
def test_reconstruct_facing_raise_has_history():
    lab = _river_label(path=["RAISE_500"], hero_seat=1)
    gs = reconstruct_live_engine(lab)
    assert gs is not None
    assert gs.current_actor() == 1
    raw = dict(gs.observation_dict())
    assert len(raw["history"]) == 1
    seat, action, chips, street = raw["history"][0]
    assert int(seat) == 0
    assert int(chips) > 0
    assert int(street) == 3
    hero_sc = int(raw["street_commit"][1])
    assert int(raw["bet_to_call"]) - hero_sc > 0  # facing a bet
    assert int(raw["pot"]) > 100_000


@pytest.mark.skipif(not _engine_has_cfr_node(), reason="rebuild: reset_nlh_cfr_node")
def test_golden_obs_bit_exact_vs_live_encode():
    lab = _river_label(path=["RAISE_500"], hero_seat=1)
    got = obs_from_label(lab)
    assert got is not None
    assert got.shape == (OBS_DIM_NLH,)

    gs = reconstruct_live_engine(lab)
    assert gs is not None
    raw = dict(gs.observation_dict())
    actor = int(raw["actor"])
    raw["hero_category_a"] = int(gs.hero_category(actor, 0))
    raw["hero_category_b"] = int(gs.hero_category(actor, 1))
    want = encode_observation_nlh(raw, live_config_from_label(lab))
    np.testing.assert_array_equal(got, want)

    # Fold legal iff to_call > 0 (engine: bet_to_call - hero street commit)
    to_call = int(raw["bet_to_call"]) - int(raw["street_commit"][actor])
    assert to_call > 0
    from plo5bp.gto.obs_from_label import labels_to_supervised_rows

    lab.to_call_chips = to_call
    rows = labels_to_supervised_rows([lab])
    assert bool(rows[0].gate_mask[0]) is True


@pytest.mark.skipif(not _engine_has_cfr_node(), reason="rebuild: reset_nlh_cfr_node")
def test_golden_empty_path_bit_exact():
    lab = _river_label(path=[], hero_seat=0)
    got = obs_from_label(lab)
    want = encode_live_engine(reconstruct_live_engine(lab), lab)
    np.testing.assert_array_equal(got, want)
    assert got[int(lab.hero_hole[0])] == 1.0
    assert got[int(lab.hero_hole[1])] == 1.0


def test_synthetic_fallback_fills_history_and_last_aggressor():
    """Preflop / no root notes: do not leave history empty or last_aggressor=None."""
    lab = _river_label(path=["RAISE_500"], hero_seat=1)
    lab.notes = {"path": "RAISE_500", "path_tokens": ["RAISE_500"]}
    # Force fallback: no root pot
    from plo5bp.gto.obs_from_label import _raw_obs_from_label

    raw = _raw_obs_from_label(lab)
    assert raw["history"], "path must populate history in the fallback"
    assert raw["last_aggressor"] != None  # noqa: E711 — must be int, not None
    assert isinstance(raw["last_aggressor"], int)
    # encode must not throw
    from plo5bp.gto.obs_from_label import game_config_from_label

    vec = encode_observation_nlh(raw, game_config_from_label(lab))
    assert vec.shape == (OBS_DIM_NLH,)


@pytest.mark.skipif(not rust_cfr_available() or not _engine_has_cfr_node(), reason="engine")
def test_live_solve_export_roundtrip_shape():
    from plo5bp.gto.cfr_api import RootSpec, SolveConfig, solve
    from plo5bp.gto.cfr_export import ExportStats, strategy_to_labels

    root = RootSpec.river_hu(
        [30, 31, 32, 33, 34],
        pot_bb=10.0,
        effective_stack_bb=50.0,
        size_preset="micro",
    )
    rep = solve(root, SolveConfig(max_iterations=30, seed=4, target_exploitability_bb=0.0))
    stats = ExportStats()
    labs = strategy_to_labels(rep.as_dict(), stats=stats)
    assert labs
    # Prefer a row we can reconstruct (has hole + board)
    lab = next(x for x in labs if len(x.hero_hole) == 2)
    obs = obs_from_label(lab)
    assert obs is not None
    assert obs.shape == (OBS_DIM_NLH,)
    gs = reconstruct_live_engine(lab)
    if gs is not None:
        np.testing.assert_array_equal(obs, encode_live_engine(gs, lab))
