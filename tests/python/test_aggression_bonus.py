"""Unit tests for the pot-fraction aggression-bonus reward shaping helper."""

from __future__ import annotations

import pytest

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.rollout import _aggression_bonus_bb


C: float = 0.10


def test_c_zero_disables_bonus_for_all_gates() -> None:
    for gate in (GATE_FOLD, GATE_CHECK_CALL, GATE_RAISE):
        bonus = _aggression_bonus_bb(
            gate=gate,
            commit_delta_chips=10000,
            bet_to_call_chips=0,
            street_commit_actor_chips=0,
            pot_chips_pre=10000,
            c=0.0,
        )
        assert bonus == 0.0


def test_fold_and_check_call_yield_zero_bonus() -> None:
    for gate in (GATE_FOLD, GATE_CHECK_CALL):
        bonus = _aggression_bonus_bb(
            gate=gate,
            commit_delta_chips=10000,
            bet_to_call_chips=0,
            street_commit_actor_chips=0,
            pot_chips_pre=10000,
            c=C,
        )
        assert bonus == 0.0, f"gate {gate} should not earn bonus"


def test_raise_pot_sized_bet_returns_full_c() -> None:
    # Pot = 30000, no facing bet, raise of 30000 chips. ratio = 1.0.
    bonus = _aggression_bonus_bb(
        gate=GATE_RAISE,
        commit_delta_chips=30000,
        bet_to_call_chips=0,
        street_commit_actor_chips=0,
        pot_chips_pre=30000,
        c=C,
    )
    assert bonus == pytest.approx(C * 1.0)


def test_raise_half_pot_bet_returns_half_c() -> None:
    bonus = _aggression_bonus_bb(
        gate=GATE_RAISE,
        commit_delta_chips=15000,
        bet_to_call_chips=0,
        street_commit_actor_chips=0,
        pot_chips_pre=30000,
        c=C,
    )
    assert bonus == pytest.approx(C * 0.5)


def test_raise_two_x_overbet_caps_at_pot() -> None:
    # 60000 into 30000 pot → ratio 2.0, capped at 1.0.
    bonus = _aggression_bonus_bb(
        gate=GATE_RAISE,
        commit_delta_chips=60000,
        bet_to_call_chips=0,
        street_commit_actor_chips=0,
        pot_chips_pre=30000,
        c=C,
    )
    assert bonus == pytest.approx(C * 1.0)


def test_raise_subtracts_call_portion_before_ratio() -> None:
    # Facing 10000 to call, putting in 40000 total → 30000 voluntary
    # aggression on a 30000 pre-step pot → ratio 1.0.
    bonus = _aggression_bonus_bb(
        gate=GATE_RAISE,
        commit_delta_chips=40000,
        bet_to_call_chips=10000,
        street_commit_actor_chips=0,
        pot_chips_pre=30000,
        c=C,
    )
    assert bonus == pytest.approx(C * 1.0)


def test_raise_after_partial_street_commit_uses_call_diff() -> None:
    # Street commit so far = 5000; bet_to_call = 12000, so call_chips
    # = 7000. commit_delta = 25000 → aggressive = 18000. Pot pre = 30000.
    bonus = _aggression_bonus_bb(
        gate=GATE_RAISE,
        commit_delta_chips=25000,
        bet_to_call_chips=12000,
        street_commit_actor_chips=5000,
        pot_chips_pre=30000,
        c=C,
    )
    assert bonus == pytest.approx(C * 18000 / 30000)


def test_short_shove_as_call_yields_zero() -> None:
    # Facing 50000; can only commit 30000 (short shove via GATE_RAISE
    # at u=1, engine clamps to stack). commit_delta ≤ call_chips → no
    # voluntary aggression.
    bonus = _aggression_bonus_bb(
        gate=GATE_RAISE,
        commit_delta_chips=30000,
        bet_to_call_chips=50000,
        street_commit_actor_chips=0,
        pot_chips_pre=80000,
        c=C,
    )
    assert bonus == 0.0


def test_short_shove_as_raise_uses_raise_increment() -> None:
    # Facing 10000 to call, shoving 50000 → aggressive = 40000 over a
    # 60000 pot → ratio 40/60.
    bonus = _aggression_bonus_bb(
        gate=GATE_RAISE,
        commit_delta_chips=50000,
        bet_to_call_chips=10000,
        street_commit_actor_chips=0,
        pot_chips_pre=60000,
        c=C,
    )
    assert bonus == pytest.approx(C * 40000 / 60000)


def test_zero_pot_yields_zero_bonus() -> None:
    # Should never happen in practice (ante guarantees nonzero pot), but
    # guard against div-by-zero just in case.
    bonus = _aggression_bonus_bb(
        gate=GATE_RAISE,
        commit_delta_chips=10000,
        bet_to_call_chips=0,
        street_commit_actor_chips=0,
        pot_chips_pre=0,
        c=C,
    )
    assert bonus == 0.0
