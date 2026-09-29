"""Basic env reset + random-legal rollout smoke test.

Uses the hybrid gate API (`step_hybrid`) — Raise picks a uniform chip
amount inside the engine-reported `[min_raise, max_raise]` range.
"""

from __future__ import annotations

import numpy as np

from plo5bp.actions import GATE_RAISE
from plo5bp.config import GameConfig
from plo5bp.encoding import OBS_DIM
from plo5bp.env import BombPotEnv


def test_reset_and_three_hands() -> None:
    env = BombPotEnv(GameConfig())
    rng = np.random.default_rng(0)
    for _ in range(3):
        obs, info = env.reset(int(rng.integers(0, 2**63 - 1)), 0)
        assert obs.shape == (OBS_DIM,)
        assert info.actor is not None
        while not info.terminal:
            legal_gates = np.flatnonzero(info.gate_mask)
            assert legal_gates.size > 0
            gate = int(rng.choice(legal_gates))
            chips = 0
            if gate == GATE_RAISE:
                lo = int(info.min_raise_chips)
                hi = int(info.max_raise_chips)
                # Short-shove regime: engine zeros lo while legal[ALL_IN]
                # is set; gate-mask OR makes GATE_RAISE legal and the env
                # redirects to apply(ALL_IN) (chip amount moot).
                assert hi >= lo and hi > 0
                chips = int(rng.integers(lo, hi + 1))
            obs, rewards, done, info = env.step_hybrid(gate, chips)
            if done:
                assert abs(float(rewards.sum())) < 1e-6
                break
