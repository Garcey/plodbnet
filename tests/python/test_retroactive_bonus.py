"""Unit tests for the retroactive (pot-share-gated) aggression bonus.

Bonus formula: per qualifying step `t`, `bonus = c * pots_bb[t]`. The
bonus is pot-relative — `c` is in bb-of-bonus per bb-of-pot. Per-step
counts are bucketed by street (0=flop, 1=turn, 2=river) using the
engine street index passed in `streets` (1=flop, 2=turn, 3=river).
Bomb pots have no preflop action, so no bucket-0 / preflop case
exists.
"""

from __future__ import annotations

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.rollout import _apply_retroactive_bonus


C: float = 1.0
FLOP = 1
TURN = 2
RIVER = 3


def _step(gate: int) -> tuple:
    """Minimal step tuple for the helper: only gate (index 2) is read."""
    return (None, None, gate, 0, None, 0.0, 0.0)


def test_c_zero_zero_chips_but_counters_track_qualifying() -> None:
    # c=0 zeroes the chip bonus but counters still track qualifying steps
    # so the trainer can report bonus%(F/T/R) even with the bonus disabled.
    # payout=200, total=200 → 2*payout > total → share > 50% → RAISE qualifies.
    steps = [_step(GATE_RAISE), _step(GATE_CHECK_CALL), _step(GATE_FOLD)]
    costs = [0.0, -0.5, 0.0]
    pots = [10.0, 10.0, 10.0]
    streets = [FLOP, FLOP, FLOP]
    added, bumped = _apply_retroactive_bonus(
        steps, costs, pots, streets,
        payout_chips=200, total_pot_chips=200, c=0.0,
    )
    assert added == 0.0
    assert bumped == [1, 0, 0]
    assert costs == [0.0, -0.5, 0.0]


def test_zero_total_pot_disables_bonus() -> None:
    steps = [_step(GATE_RAISE)]
    costs = [-0.5]
    pots = [10.0]
    streets = [FLOP]
    added, bumped = _apply_retroactive_bonus(
        steps, costs, pots, streets,
        payout_chips=0, total_pot_chips=0, c=C,
    )
    assert added == 0.0
    assert bumped == [0, 0, 0]
    assert costs == [-0.5]


def test_share_below_half_no_bonus() -> None:
    # Quartered: hero gets 50 of a 200 pot → 25% share → no bonus.
    steps = [_step(GATE_RAISE), _step(GATE_CHECK_CALL), _step(GATE_FOLD)]
    costs = [-0.3, -0.2, 0.0]
    pots = [5.0, 8.0, 10.0]
    streets = [FLOP, TURN, RIVER]
    added, bumped = _apply_retroactive_bonus(
        steps, costs, pots, streets,
        payout_chips=50, total_pot_chips=200, c=C,
    )
    assert added == 0.0
    assert bumped == [0, 0, 0]
    assert costs == [-0.3, -0.2, 0.0]


def test_share_above_half_bonus_on_raise_only() -> None:
    # Hero scoops both boards: payout 200 of 200 pot → 100% > 50%.
    # Trajectory: raise (flop), call (flop), check (turn), raise (turn),
    # fold (river). Pots vary across the trajectory so we can verify
    # per-step scaling.
    steps = [
        _step(GATE_RAISE),
        _step(GATE_CHECK_CALL),
        _step(GATE_CHECK_CALL),
        _step(GATE_RAISE),
        _step(GATE_FOLD),
    ]
    costs = [-1.0, -0.5, 0.0, -2.0, 0.0]
    pots = [10.0, 12.0, 12.0, 50.0, 50.0]
    streets = [FLOP, FLOP, TURN, TURN, RIVER]
    added, bumped = _apply_retroactive_bonus(
        steps, costs, pots, streets,
        payout_chips=200, total_pot_chips=200, c=C,
    )
    # Only the two RAISE steps qualify: one on flop, one on turn.
    assert bumped == [1, 1, 0]
    # bonuses: c*10 = 10, c*50 = 50.
    assert added == 60.0
    assert costs == [9.0, -0.5, 0.0, 48.0, 0.0]


def test_share_exactly_half_bonus_on_raise_and_call() -> None:
    # Hero chops both boards HU: payout 100 of 200 pot → 50% exactly.
    # Trajectory: raise (flop), call (flop), check (turn), call (turn),
    # fold (river).
    steps = [
        _step(GATE_RAISE),
        _step(GATE_CHECK_CALL),
        _step(GATE_CHECK_CALL),
        _step(GATE_CHECK_CALL),
        _step(GATE_FOLD),
    ]
    costs = [-1.0, -0.5, 0.0, -0.3, 0.0]
    pots = [4.0, 6.0, 6.0, 20.0, 20.0]
    streets = [FLOP, FLOP, TURN, TURN, RIVER]
    added, bumped = _apply_retroactive_bonus(
        steps, costs, pots, streets,
        payout_chips=100, total_pot_chips=200, c=C,
    )
    # RAISE + 2x CHECK_CALL(call). The CHECK_CALL with cost==0 (check)
    # and the FOLD do NOT get the bonus.
    # Bonuses: 4 + 6 (flop) + 20 (turn) = 30.
    assert bumped == [2, 1, 0]
    assert added == 30.0
    assert costs == [3.0, 5.5, 0.0, 19.7, 0.0]


def test_fold_out_treated_as_above_half() -> None:
    # Survivor gets 100% of the pot via fold-out → bonus on RAISE only.
    steps = [_step(GATE_RAISE), _step(GATE_CHECK_CALL), _step(GATE_RAISE)]
    costs = [-1.0, -0.5, -2.0]
    pots = [3.0, 5.0, 8.0]
    streets = [FLOP, FLOP, TURN]
    added, bumped = _apply_retroactive_bonus(
        steps, costs, pots, streets,
        payout_chips=300, total_pot_chips=300, c=C,
    )
    assert bumped == [1, 1, 0]
    assert added == 11.0  # c*3 + c*8
    assert costs == [2.0, -0.5, 6.0]


def test_share_just_below_half_no_bonus() -> None:
    # Won 99 of 200 (49.5%) — strictly less than half → no bonus, even
    # though it's a near-chop.
    steps = [_step(GATE_RAISE), _step(GATE_CHECK_CALL)]
    costs = [-1.0, -0.5]
    pots = [10.0, 12.0]
    streets = [FLOP, FLOP]
    added, bumped = _apply_retroactive_bonus(
        steps, costs, pots, streets,
        payout_chips=99, total_pot_chips=200, c=C,
    )
    assert added == 0.0
    assert bumped == [0, 0, 0]
    assert costs == [-1.0, -0.5]


def test_check_call_with_zero_cost_is_a_check_no_bonus() -> None:
    # Half-pot share, but the CHECK_CALL has cost==0 (check, not call).
    # Should NOT receive bonus — only chip-committing CHECK_CALLs do.
    steps = [_step(GATE_CHECK_CALL), _step(GATE_RAISE)]
    costs = [0.0, -1.0]
    pots = [6.0, 10.0]
    streets = [FLOP, TURN]
    added, bumped = _apply_retroactive_bonus(
        steps, costs, pots, streets,
        payout_chips=100, total_pot_chips=200, c=C,
    )
    assert bumped == [0, 1, 0]  # only the turn RAISE
    assert added == 10.0  # c*10
    assert costs == [0.0, 9.0]


def test_pot_relative_scaling_river_louder_than_flop() -> None:
    # Same trajectory shape but pots increase across streets — river-
    # weighted bonus is the load-bearing change vs the old flat-c.
    steps = [_step(GATE_RAISE), _step(GATE_RAISE), _step(GATE_RAISE)]
    costs = [-1.0, -2.0, -3.0]
    pots = [6.0, 30.0, 100.0]  # flop, turn, river bb
    streets = [FLOP, TURN, RIVER]
    added, bumped = _apply_retroactive_bonus(
        steps, costs, pots, streets,
        payout_chips=400, total_pot_chips=400, c=0.05,
    )
    assert bumped == [1, 1, 1]
    # 0.05 * (6 + 30 + 100) = 6.8.
    assert added == 6.8
    assert costs == [-0.7, -0.5, 2.0]


def test_zero_pot_step_yields_zero_bonus() -> None:
    # A qualifying step with pot==0 contributes nothing — degenerate
    # but worth pinning down so the helper is robust to edge cases.
    steps = [_step(GATE_RAISE), _step(GATE_RAISE)]
    costs = [-1.0, -2.0]
    pots = [0.0, 10.0]
    streets = [FLOP, TURN]
    added, bumped = _apply_retroactive_bonus(
        steps, costs, pots, streets,
        payout_chips=200, total_pot_chips=200, c=C,
    )
    assert bumped == [1, 1, 0]  # both qualify; only one contributes magnitude
    assert added == 10.0
    assert costs == [-1.0, 8.0]


def test_per_street_bucketing_river_only() -> None:
    # Every qualifying RAISE on the river → bumped[2] only. Confirms
    # the engine-street → bucket mapping (3 → 2) and that turn / flop
    # buckets stay zero.
    steps = [_step(GATE_RAISE), _step(GATE_RAISE), _step(GATE_RAISE)]
    costs = [-1.0, -2.0, -3.0]
    pots = [40.0, 40.0, 40.0]
    streets = [RIVER, RIVER, RIVER]
    added, bumped = _apply_retroactive_bonus(
        steps, costs, pots, streets,
        payout_chips=400, total_pot_chips=400, c=0.1,
    )
    assert bumped == [0, 0, 3]
    assert added == 12.0  # 0.1 * 40 * 3


def test_empty_trajectory_returns_zero() -> None:
    steps: list[tuple] = []
    costs: list[float] = []
    pots: list[float] = []
    streets: list[int] = []
    added, bumped = _apply_retroactive_bonus(
        steps, costs, pots, streets,
        payout_chips=200, total_pot_chips=200, c=C,
    )
    assert added == 0.0
    assert bumped == [0, 0, 0]
    assert costs == []
