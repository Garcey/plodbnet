"""Dust guard: near-all-in raises snap to a full shove.

Continuous Beta-sampled sizings frequently land a few chips shy of
all-in, leaving an absurd sub-display "live" stack ($0.00 in the UI)
that forces degenerate extra streets (cover-short bets of a few chips)
instead of a clean run-out. Reported from a live trainer hand.
"""

from __future__ import annotations

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv


def _env(stacks):
    cfg = GameConfig(num_seats=3, starting_stack=0, ante=30_000, bb=10_000,
                     starting_stacks=stacks)
    env = BombPotEnv(cfg)
    _, info = env.reset(seed=3, button=2)
    return env, info


def test_dust_raise_snaps_to_allin_and_runs_out():
    # Seat 1 behind after ante: 270_000; stack-capped max raise. Raising
    # 269_990 would leave 10 chips (<= bb/100 = 100) -> snapped to all-in.
    env, info = _env((2_000_000, 300_000, 2_000_000))
    _, _, _, info = env.step_hybrid(GATE_RAISE, 90_000)   # hero pots it
    assert info.max_raise_chips == 270_000
    _, _, _, info = env.step_hybrid(GATE_RAISE, 269_990)
    raw = info.raw_obs
    assert int(raw["stacks"][1]) == 0, "dust raise must snap to full stack"
    assert bool(raw["all_in"][1])
    _, _, _, info = env.step_hybrid(GATE_FOLD)            # seat 2
    _, rewards, done, info = env.step_hybrid(GATE_CHECK_CALL)  # hero calls
    assert done, "all-in call must run straight out to showdown"
    raw = env._rs.observation_dict()
    assert len(raw["board_a"]) == 5 and len(raw["board_b"]) == 5
    assert abs(float(rewards.sum())) < 1e-6


def test_raise_leaving_real_chips_not_snapped():
    # Leaving 200 chips (> bb/100 = 100) is a real (tiny) stack: no snap,
    # and the hand continues to a normal next street.
    env, info = _env((2_000_000, 300_000, 2_000_000))
    _, _, _, info = env.step_hybrid(GATE_RAISE, 90_000)
    _, _, _, info = env.step_hybrid(GATE_RAISE, 269_800)
    raw = info.raw_obs
    assert int(raw["stacks"][1]) == 200
    assert not bool(raw["all_in"][1])
