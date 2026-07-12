"""BatchedBombPotEnv.reconfigure + multiconfig env_cache reuse."""

from __future__ import annotations

import numpy as np
import torch

from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.env_batched import BatchedBombPotEnv
from plo5bp.network import ActorCritic
from plo5bp.rollout import collect_rollout_batched, collect_rollout_multiconfig
from plo5bp.selfplay import OpponentPool


def test_reconfigure_changes_stacks_and_requires_reset():
    n = 8
    cfg_a = GameConfig(num_seats=6, starting_stack=200_000, ante=30_000)
    cfg_b = GameConfig(
        num_seats=6,
        starting_stacks=(100_000, 150_000, 200_000, 250_000, 300_000, 350_000),
        ante=20_000,
    )
    env = BatchedBombPotEnv(n, cfg_a, opp_outcome_mc=64)
    assert env.can_reconfigure(cfg_b)
    assert not env.can_reconfigure(GameConfig(num_seats=4))

    seeds = np.arange(n, dtype=np.uint64)
    buttons = np.zeros(n, dtype=np.uint8)
    env.reset_batch(seeds, buttons)
    stacks_a = np.asarray(env._be.observation_arrays()["stacks"]).copy()

    env.reconfigure(cfg_b)
    # After reconfigure, no live hand until reset.
    assert env._dones.all()
    env.reset_batch(seeds + 100, buttons)
    stacks_b = np.asarray(env._be.observation_arrays()["stacks"])
    # Heterogeneous stacks should differ from the uniform-200k hand.
    assert not np.array_equal(stacks_a, stacks_b)
    # Resolved stacks appear as starting remaining (+/- ante on flop open).
    # At least the relative order / max should reflect cfg_b's spread.
    assert stacks_b.max() > stacks_a.max() or stacks_b.min() < stacks_a.min()


def test_reconfigure_rejects_seat_mismatch():
    env = BatchedBombPotEnv(4, GameConfig(num_seats=6))
    try:
        env.reconfigure(GameConfig(num_seats=4))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_env_cache_reuses_same_object_across_sub_rollouts():
    """Multiconfig with two same-seat configs should hit reconfigure, not
    allocate two engines. We probe via a caller-owned env_cache dict."""
    device = torch.device("cpu")
    model = ActorCritic(hidden_dim=32, num_layers=1).to(device).eval()
    pool = OpponentPool(capacity=1)
    pool.snapshot(model)
    train_cfg = TrainingConfig(
        num_envs=4,
        rollout_length=32,
        hidden_dim=32,
        num_layers=1,
        pool_mix_prob=0.0,
    )
    cfg_a = GameConfig(num_seats=3, starting_stack=100_000)
    cfg_b = GameConfig(num_seats=3, starting_stack=300_000)
    cache: dict = {}
    rng = np.random.default_rng(1)
    collect_rollout_batched(
        model, pool, cfg_a, train_cfg, rng, env_cache=cache
    )
    assert len(cache) == 1
    env_obj = next(iter(cache.values()))
    collect_rollout_batched(
        model, pool, cfg_b, train_cfg, np.random.default_rng(2), env_cache=cache
    )
    assert len(cache) == 1
    assert next(iter(cache.values())) is env_obj
    # Different seat count → second cache entry.
    cfg_c = GameConfig(num_seats=5, starting_stack=200_000)
    collect_rollout_batched(
        model, pool, cfg_c, train_cfg, np.random.default_rng(3), env_cache=cache
    )
    assert len(cache) == 2


def test_multiconfig_still_runs_with_env_reuse():
    device = torch.device("cpu")
    model = ActorCritic(hidden_dim=32, num_layers=1).to(device).eval()
    pool = OpponentPool(capacity=1)
    pool.snapshot(model)
    train_cfg = TrainingConfig(
        num_envs=8,
        rollout_length=64,
        hidden_dim=32,
        num_layers=1,
        pool_mix_prob=0.0,
    )
    configs = [
        GameConfig(num_seats=4, starting_stack=150_000),
        GameConfig(num_seats=4, starting_stack=400_000),
        GameConfig(num_seats=6, starting_stack=200_000),
    ]
    batch = collect_rollout_multiconfig(
        model, pool, configs, train_cfg, np.random.default_rng(7)
    )
    assert batch.obs.shape[0] > 0
    assert batch.obs.shape[1] > 0
