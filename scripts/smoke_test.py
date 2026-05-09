"""Run N random-legal hands, assert invariants, print frequency stats."""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np

from plo5bp.actions import ACTION_NAMES, NUM_ACTIONS
from plo5bp.config import GameConfig
from plo5bp.encoding import OBS_DIM
from plo5bp.env import BombPotEnv


def run_smoke(num_hands: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    env = BombPotEnv(GameConfig())

    action_counter: Counter[int] = Counter()
    showdown_count = 0
    foldout_count = 0
    pot_sum = 0.0
    total_actions = 0
    max_actions_per_hand = 0

    start = time.perf_counter()
    for h in range(num_hands):
        hand_seed = int(rng.integers(0, 2**63 - 1))
        button = int(rng.integers(0, env.num_seats))
        obs, info = env.reset(hand_seed, button)
        assert obs.shape == (OBS_DIM,), f"obs shape {obs.shape}"
        steps = 0
        while not info.terminal:
            legal = np.flatnonzero(info.legal_mask)
            assert legal.size > 0, "no legal actions"
            action = int(rng.choice(legal))
            action_counter[action] += 1
            obs, rewards, done, info = env.step(action)
            steps += 1
            assert steps < 200, "hand failed to terminate within 200 actions"
            if done:
                assert abs(float(rewards.sum())) < 1e-6, f"non-zero sum {rewards.sum()}"
                total_actions += steps
                max_actions_per_hand = max(max_actions_per_hand, steps)
                folded = info.raw_obs.get("folded", [])
                num_non_folded = env.num_seats - int(sum(folded))
                if num_non_folded == 1:
                    foldout_count += 1
                else:
                    showdown_count += 1
                pot_sum += float((-rewards[rewards < 0]).sum())
                break
    elapsed = time.perf_counter() - start

    print(f"Ran {num_hands} hands in {elapsed:.2f}s ({num_hands / elapsed:.0f} hands/s)")
    print(f"Avg actions per hand: {total_actions / num_hands:.2f}")
    print(f"Max actions in a hand: {max_actions_per_hand}")
    print(f"Mean observed pot (chips): {pot_sum / num_hands:.1f}")
    print(f"Fold-out: {foldout_count}   Showdown/chop: {showdown_count}")
    print("Action frequencies:")
    for a in range(NUM_ACTIONS):
        name = ACTION_NAMES[a]
        count = action_counter.get(a, 0)
        pct = 100.0 * count / max(sum(action_counter.values()), 1)
        print(f"  {name:12s} {count:8d}  ({pct:5.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-hands", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_smoke(args.num_hands, args.seed)


if __name__ == "__main__":
    main()
