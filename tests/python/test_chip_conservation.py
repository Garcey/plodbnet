"""Chip conservation across 1000 random-legal hands."""

from __future__ import annotations

import numpy as np

from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv


def test_chip_conservation_1000_hands() -> None:
    env = BombPotEnv(GameConfig())
    rng = np.random.default_rng(123)
    for h in range(1000):
        obs, info = env.reset(h, h % env.num_seats)
        steps = 0
        while not info.terminal:
            legal = np.flatnonzero(info.legal_mask)
            action = int(rng.choice(legal))
            obs, rewards, done, info = env.step(action)
            steps += 1
            assert steps < 200
            if done:
                assert abs(int(rewards.sum())) == 0
                break
