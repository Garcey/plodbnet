"""The betting cap (owner, 2026-10-02): every network trains on bets also capped at
what the deepest opponent can still put in (``GameConfig.reach_cap``, the default —
Study and the Trainer serve it); the home games switch it off, so a bet stops only at
the pot limit and the bettor's own stack, and what nobody matches comes back."""

from __future__ import annotations

import dataclasses

import pytest

from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv
from plo5bp.env_batched import BatchedBombPotEnv

# The owner's example: 200 / 150 / 100 behind after a 60 ante, 180 in the pot.
EXAMPLE = GameConfig(num_seats=3, starting_stack=0, starting_stacks=(260, 210, 160), ante=60, bb=10)


def test_the_trained_rule_is_the_default_and_the_home_games_rule_a_switch():
    assert EXAMPLE.reach_cap is True
    _, info = BombPotEnv(EXAMPLE).reset(7, 2)
    assert info.actor == 0 and info.max_raise_chips == 150, "capped at the 150 stack's reach"
    env = BombPotEnv(dataclasses.replace(EXAMPLE, reach_cap=False))
    _, info = env.reset(7, 2)
    assert (info.min_raise_chips, info.max_raise_chips) == (10, 180), "the pot"
    env.step_hybrid(2, 180)
    env.step_hybrid(0)  # the 150 stack folds
    _, _, done, _ = env.step_hybrid(1)  # the 100 stack calls all in
    assert done
    pay = env.payouts()
    assert pay[1] == -60 and sum(pay) == 0
    assert -160 <= pay[0] <= 220, "the 80 nobody matched came back"


def test_the_batched_training_engine_refuses_the_home_games_rule():
    home = dataclasses.replace(EXAMPLE, reach_cap=False)
    with pytest.raises(ValueError, match="reach-capped"):
        BatchedBombPotEnv(2, home)
    env = BatchedBombPotEnv(2, EXAMPLE)
    with pytest.raises(ValueError, match="reach-capped"):
        env.reconfigure(home)
