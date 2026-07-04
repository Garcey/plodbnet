"""Regression: an all-in *call* that leaves a sub-bet residual must still
be 'betting moot' so the trainer runs the hand out to showdown instead of
stranding the next actor on a meaningless check.

Reproduces the bug where a seat called off to a ~15-chip residual (the UI
showed "$0.00") that the engine did NOT flag as all_in; the old 0.001bb
dust floor let that residual read as a live bettor, so the river stopped
on hero. See ``plo5bp.ui.trainer._DUST_CHIPS``.
"""

from plo5bp.actions import GATE_CHECK_CALL, GATE_RAISE
from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv

import plo5bp.ui.trainer as T


def _drive_to_residual_node():
    """Heads-up: whittle the short seat down to a tiny live residual via
    calls (odd stack + a pot-sized bet leaves chips that don't divide
    evenly), then return the river node where the long seat faces nothing.
    """
    cfg = GameConfig(num_seats=2, starting_stacks=(60015, 400000),
                     ante=10000, bb=10000)
    SHORT, LONG = 0, 1
    env = BombPotEnv(cfg)
    _, info = env.reset(0, 0)
    done = False
    for _ in range(40):
        if done or info.actor is None:
            break
        a = int(info.actor)
        raw = info.raw_obs
        stacks = [int(x) for x in raw["stacks"]]
        all_in = [bool(x) for x in raw["all_in"]]
        tc = T._to_call_chips(raw, a)
        # Target node: long seat to act facing nothing, short seat live
        # with a tiny residual the engine never flagged all-in. (literal
        # 100-chip probe so the test is independent of the dust constant.)
        if (a == LONG and tc <= T._DUST_CHIPS and not all_in[SHORT]
                and 0 < stacks[SHORT] <= 100):
            return raw, a, stacks[SHORT]
        if tc > 0:
            gate, chips = GATE_CHECK_CALL, 0
        elif a == LONG:
            want = stacks[SHORT] - 15
            mn, mr = info.min_raise_chips, info.max_raise_chips
            if mr > 0 and want >= mn:
                gate, chips = GATE_RAISE, min(want, mr)
            else:
                gate, chips = GATE_CHECK_CALL, 0
        else:
            gate, chips = GATE_CHECK_CALL, 0
        _, _, done, info = env.step_hybrid(gate, chips)
    return None


def test_residual_allin_call_is_betting_moot():
    node = _drive_to_residual_node()
    assert node is not None, "failed to reproduce the residual node"
    raw, actor, residual = node
    # The engine left a tiny LIVE residual and did not flag all-in...
    assert 0 < residual <= 100
    assert not bool(raw["all_in"][0])
    # ...yet the seat can neither bet nor be meaningfully bet into, so the
    # actor's node is moot -> _advance auto-checks -> the hand runs out.
    assert T._betting_moot(raw, actor) is True
