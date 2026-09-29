"""The F/T/R telemetry's numerator: a finished hand's WINNING aggression steps.

`rollout._winning_aggression_steps` (serial collector) and the flush kernel
(batched collector) count, per street, a learner seat's raises when it won
more than half the pot, and with exactly half also its calls (a CHECK_CALL
that committed chips: cost < 0). The retroactive aggression BONUS these steps
once earned is retired (2026-09-28, ML-030; every stem since vTwo ran it at
0) -- these are the qualification rules of the retired
tests/python/test_retroactive_bonus.py, without the bonus arithmetic.
"""

from __future__ import annotations

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.rollout import _winning_aggression_steps

FLOP, TURN, RIVER = 1, 2, 3


def _step(gate: int) -> tuple:
    """Minimal trajectory tuple: only the gate (index 2) is read."""
    return (None, None, gate, 0, None, 0.0, 0.0)


def _count(gates, costs, streets, payout, total):
    return _winning_aggression_steps(
        [_step(g) for g in gates], list(costs), list(streets), payout, total
    )


def test_more_than_half_counts_raises_only() -> None:
    gates = [GATE_RAISE, GATE_CHECK_CALL, GATE_CHECK_CALL, GATE_RAISE, GATE_FOLD]
    costs = [-1.0, -0.5, 0.0, -2.0, 0.0]
    streets = [FLOP, FLOP, TURN, TURN, RIVER]
    assert _count(gates, costs, streets, 200, 200) == [1, 1, 0]
    # a fold-out win is a 100% share too
    assert _count([GATE_RAISE, GATE_CHECK_CALL, GATE_RAISE], [-1.0, -0.5, -2.0],
                  [FLOP, FLOP, TURN], 300, 300) == [1, 1, 0]


def test_exactly_half_also_counts_calls_but_not_checks() -> None:
    gates = [GATE_RAISE, GATE_CHECK_CALL, GATE_CHECK_CALL, GATE_CHECK_CALL, GATE_FOLD]
    costs = [-1.0, -0.5, 0.0, -0.3, 0.0]  # the 0-cost CHECK_CALL is a check
    streets = [FLOP, FLOP, TURN, TURN, RIVER]
    assert _count(gates, costs, streets, 100, 200) == [2, 1, 0]


def test_less_than_half_counts_nothing() -> None:
    gates = [GATE_RAISE, GATE_CHECK_CALL]
    assert _count(gates, [-0.3, -0.2], [FLOP, TURN], 50, 200) == [0, 0, 0]
    assert _count(gates, [-1.0, -0.5], [FLOP, FLOP], 99, 200) == [0, 0, 0]


def test_empty_pot_and_empty_trajectory_count_nothing() -> None:
    assert _count([GATE_RAISE], [-0.5], [FLOP], 0, 0) == [0, 0, 0]
    assert _count([], [], [], 200, 200) == [0, 0, 0]


def test_streets_bucket_flop_turn_river() -> None:
    gates = [GATE_RAISE] * 3
    assert _count(gates, [-1.0] * 3, [RIVER] * 3, 400, 400) == [0, 0, 3]
    assert _count(gates, [-1.0] * 3, [FLOP, TURN, RIVER], 400, 400) == [1, 1, 1]


def test_counting_never_touches_the_costs() -> None:
    costs = [-1.0, -0.5]
    _winning_aggression_steps(
        [_step(GATE_RAISE), _step(GATE_CHECK_CALL)], costs, [FLOP, FLOP], 100, 200
    )
    assert costs == [-1.0, -0.5]
