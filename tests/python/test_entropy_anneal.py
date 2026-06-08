"""Unit tests for the automated F/T/R-gated entropy anneal in scripts/train.py.

Covers the pure decision helper (`_anneal_decision`), the per-block F/T/R
accumulation arithmetic, and checkpoint serialization of the anneal state.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_TRAIN_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train.py"


def _load_train():
    spec = importlib.util.spec_from_file_location("train_under_test", _TRAIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train = _load_train()
decide = train._anneal_decision

STEP = 0.002
FLOOR = 0.0
TOL = 0.5


def test_first_block_records_baseline_no_change():
    ent, base, action = decide((20.0, 28.0, 35.0), None, 0.04, STEP, FLOOR, TOL)
    assert ent == 0.04  # unchanged
    assert base == (20.0, 28.0, 35.0)
    assert action == "record-baseline"


def test_held_all_streets_lowers_and_advances_baseline():
    ent, base, action = decide((21.0, 29.0, 36.0), (20.0, 28.0, 35.0), 0.04, STEP, FLOOR, TOL)
    assert ent == pytest.approx(0.038)
    assert base == (21.0, 29.0, 36.0)  # baseline ratchets up to the new level
    assert action == "lowered"


def test_held_exactly_at_tolerance_boundary_counts_as_held():
    # each street exactly baseline - tol -> still >= baseline - tol -> held
    ent, base, action = decide((19.5, 27.5, 34.5), (20.0, 28.0, 35.0), 0.04, STEP, FLOOR, TOL)
    assert action == "lowered"
    assert ent == pytest.approx(0.038)


def test_one_street_just_below_tolerance_is_a_drop():
    # flop slips past tol -> drop, even though turn/river held
    ent, base, action = decide((19.4, 28.0, 35.0), (20.0, 28.0, 35.0), 0.04, STEP, FLOOR, TOL)
    assert action == "drop:hold"
    assert ent == 0.04  # held, not lowered
    assert base == (20.0, 28.0, 35.0)  # baseline kept (no ratchet-down)


def test_drop_keeps_old_baseline_to_require_recovery():
    # turn craters; baseline must stay so the next block must recover to it
    ent, base, action = decide((20.0, 22.0, 36.0), (20.0, 28.0, 35.0), 0.03, STEP, FLOOR, TOL)
    assert action == "drop:hold"
    assert ent == 0.03
    assert base == (20.0, 28.0, 35.0)


def test_floor_clamp_never_negative():
    ent, base, action = decide((30.0, 40.0, 50.0), (20.0, 28.0, 35.0), 0.001, STEP, FLOOR, TOL)
    assert ent == 0.0  # max(floor, 0.001 - 0.002) == 0.0
    assert action == "lowered"


def test_at_floor_stays_at_floor():
    ent, base, action = decide((30.0, 40.0, 50.0), (20.0, 28.0, 35.0), 0.0, STEP, FLOOR, TOL)
    assert ent == 0.0
    assert action == "held@floor"
    assert base == (30.0, 40.0, 50.0)  # baseline still tracks


def test_custom_nonzero_floor():
    ent, _b, action = decide((30.0, 40.0, 50.0), (20.0, 28.0, 35.0), 0.011, STEP, 0.01, TOL)
    assert ent == pytest.approx(0.01)  # 0.011 - 0.002 clamped up to floor 0.01
    assert action == "lowered"


def test_block_ftr_is_count_weighted_not_percent_averaged():
    """Block F/T/R must aggregate raw counts, not average per-update percents.

    Mirrors the in-loop accumulation: sum bonus_steps and steps across updates,
    then take 100*bonus/steps per street.
    """
    # update A: flop 1/10 aggressive; update B: flop 9/10 aggressive
    bonus = [0, 0, 0]
    steps = [0, 0, 0]
    for b_upd, s_upd in [((1, 0, 0), (10, 0, 0)), ((9, 0, 0), (10, 0, 0))]:
        for s in range(3):
            bonus[s] += b_upd[s]
            steps[s] += s_upd[s]
    flop = 100.0 * bonus[0] / max(1, steps[0])
    assert flop == pytest.approx(50.0)  # (1+9)/(10+10) = 50%, not avg(10%,90%)


def test_checkpoint_anneal_keys_roundtrip(tmp_path):
    """torch.save/load preserves the anneal state dicts (tuples, None, floats)."""
    import torch

    state = {
        "update_counter": 237,
        "anneal_tier_ent": {"clubgg": 0.034, "clubgg_deep": 0.06, "deep": 0.085},
        "anneal_baseline": {
            "clubgg": (20.1, 28.4, 35.2),
            "clubgg_deep": None,
            "deep": (22.0, 30.0, 35.0),
        },
        "anneal_block_acc": {"bonus_steps": [11, 22, 33], "steps": [50, 60, 70], "tier": "deep"},
    }
    p = tmp_path / "ckpt.pt"
    torch.save(state, p)
    loaded = torch.load(p, map_location="cpu", weights_only=False)
    assert loaded["update_counter"] == 237
    assert loaded["anneal_tier_ent"]["clubgg"] == pytest.approx(0.034)
    assert loaded["anneal_baseline"]["clubgg_deep"] is None
    assert tuple(loaded["anneal_baseline"]["deep"]) == (22.0, 30.0, 35.0)
    assert loaded["anneal_block_acc"]["tier"] == "deep"


def test_anneal_convergence_to_floor_over_repeated_holds():
    """Repeated held blocks walk a tier from 0.04 down to 0.0 in 20 steps."""
    ent = 0.04
    base = (20.0, 28.0, 35.0)
    for _ in range(40):  # more than enough
        ent, base, _action = decide((20.0, 28.0, 35.0), base, ent, STEP, FLOOR, TOL)
    assert ent == 0.0
