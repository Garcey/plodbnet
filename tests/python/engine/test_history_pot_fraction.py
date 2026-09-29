"""Semantics of the history pot-fraction dim (slot dim 17, v2 encoder).

Each history slot's last dim = chips_of_action / pot_before_action,
reconstructed via suffix sums (history chips are per-action deltas;
antes live in the pot but never in history records).
"""

from __future__ import annotations

import numpy as np

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.config import GameConfig
from plo5bp.encoding import (
    _HISTORY_DEPTH,
    _HISTORY_FRAC_OFF_REL,
    _HISTORY_OFF,
    _HISTORY_SLOT_DIM,
)
from plo5bp.env import BombPotEnv


def _slot_frac(obs: np.ndarray, slot: int) -> float:
    return float(obs[_HISTORY_OFF + slot * _HISTORY_SLOT_DIM + _HISTORY_FRAC_OFF_REL])


def test_hand_computed_fractions():
    # 3 seats, ante 300 -> flop pot 900.
    cfg = GameConfig(num_seats=3, starting_stack=2_000_000, ante=300, bb=100)
    env = BombPotEnv(cfg)
    obs, info = env.reset(11, 2)  # order 0, 1, 2

    obs, _, _, info = env.step_hybrid(GATE_RAISE, 450)      # bet 450 into 900
    obs, _, _, info = env.step_hybrid(GATE_CHECK_CALL)      # call 450 into 1350
    obs, _, _, info = env.step_hybrid(GATE_FOLD)            # fold into 1800

    # Street closed -> turn; next actor's obs carries 3 history slots.
    assert _slot_frac(obs, 0) == np.float32(450 / 900)
    assert _slot_frac(obs, 1) == np.float32(450 / 1350)
    assert _slot_frac(obs, 2) == np.float32(0.0)


def test_fraction_clipped_at_two():
    # HU: opening pot bet then a PL re-raise can exceed 2x the prior pot?
    # PL raise delta <= to_call + (pot + to_call): with to_call=c and
    # pot=p the max frac is (c + p + c)/p = 1 + 2c/p — for c == p this
    # is 3 -> clipped to 2.
    cfg = GameConfig(num_seats=2, starting_stack=10_000_000, ante=300, bb=100)
    env = BombPotEnv(cfg)
    obs, info = env.reset(5, 0)
    obs, _, _, info = env.step_hybrid(GATE_RAISE, 600)      # pot bet 600 into 600
    # PL max delta now: to_call 600 + (pot 1200 + 600) = 2400 = 2x prior pot(1200)
    obs, _, _, info = env.step_hybrid(GATE_RAISE, int(info.max_raise_chips))
    frac = _slot_frac(obs, 1)
    assert frac == np.float32(min(2400 / 1200, 2.0))
    assert frac <= 2.0


def test_truncation_suffix_anchor():
    # Generate > _HISTORY_DEPTH actions via an HU min-raise war and check
    # the visible slots' fractions against independently tracked pots.
    cfg = GameConfig(num_seats=2, starting_stack=100_000_000, ante=300, bb=100)
    env = BombPotEnv(cfg)
    obs, info = env.reset(99, 0)

    pots_before: list[int] = []
    chips_log: list[int] = []
    pot = 600  # two antes
    steps = 0
    while steps < _HISTORY_DEPTH + 6 and not info.terminal:
        mn = int(info.min_raise_chips)
        mx = int(info.max_raise_chips)
        if mn == 0 or mx == 0:
            break
        chips = mn  # min-raise keeps the war going cheaply
        pots_before.append(pot)
        chips_log.append(chips)
        obs, _, done, info = env.step_hybrid(GATE_RAISE, chips)
        pot += chips
        steps += 1
        if done:
            break

    assert steps > _HISTORY_DEPTH, "need more actions than the history window"
    offset = steps - _HISTORY_DEPTH
    for slot in range(_HISTORY_DEPTH):
        g = offset + slot
        expect = np.float32(min(chips_log[g] / max(pots_before[g], 1), 2.0))
        assert _slot_frac(obs, slot) == expect, f"slot {slot} mismatch"
