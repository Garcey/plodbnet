"""`payouts_ev_subset` (2026-09-23) returns exactly the chosen rows of
`payouts_ev_batch`.

The rollout reads only the newly-terminal rows of the EV payouts; in the
drain phase every env that finished on an earlier step is still terminal and
the whole-batch call re-ran all of their Monte-Carlo runouts on every step.
The subset call computes just the requested rows (one hand per rayon task).
Pinned: identical rows (terminal ones, all-in runouts included, and zeros for
non-terminal envs) under the same per-env seeds, plus argument validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from plo5bp._engine import BatchedEngine
from plo5bp.actions import GATE_CHECK_CALL, GATE_RAISE
from plo5bp.config import GameConfig
from plo5bp.env_batched import BatchedBombPotEnv

pytestmark = pytest.mark.skipif(
    not hasattr(BatchedEngine, "payouts_ev_subset"),
    reason="payouts_ev_subset not in this engine build",
)


def _shove_heavy_actions(env: BatchedBombPotEnv, rng: np.random.Generator):
    """Legal actions biased to max-size raises so many hands go all-in before
    the river (the Monte-Carlo runout path)."""
    n = env.n
    gates = np.full(n, GATE_CHECK_CALL, dtype=np.uint8)
    chips = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        legal = np.nonzero(env._gate_mask[i])[0]
        if legal.size == 0:
            continue
        if env._gate_mask[i, GATE_RAISE] and rng.random() < 0.5:
            gates[i] = GATE_RAISE
            chips[i] = env._max_raise[i]
        else:
            gates[i] = int(rng.choice(legal))
            if gates[i] == GATE_RAISE:
                chips[i] = env._min_raise[i]
    return gates, chips


@pytest.mark.parametrize("seats,samples", [(2, 64), (3, 16), (6, 64)])
def test_subset_rows_match_batch(seats, samples) -> None:
    n = 48
    cfg = GameConfig(num_seats=seats, starting_stack=300000, ante=30000, bb=10000)
    env = BatchedBombPotEnv(n, cfg, opp_outcome_mc=0, obs_mode="minimal")
    rng = np.random.default_rng(seats * 7 + samples)
    env.reset_batch(
        rng.integers(0, 2**63 - 1, size=n, dtype=np.int64).astype(np.uint64),
        rng.integers(0, seats, size=n).astype(np.uint8),
    )
    be = env._be
    runouts = 0
    for _ in range(40):
        gates, chips = _shove_heavy_actions(env, rng)
        env.step_hybrid_batch(gates, chips)
        seeds = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64).astype(np.uint64)
        full = np.asarray(be.payouts_ev_batch(samples, seeds))
        # All envs, a random subset in random order, and the terminal ones.
        term = np.nonzero(env._dones)[0].astype(np.int64)
        picks = [
            np.arange(n, dtype=np.int64),
            rng.permutation(n)[: int(rng.integers(0, n + 1))].astype(np.int64),
            term,
        ]
        for idx in picks:
            sub = np.asarray(be.payouts_ev_subset(samples, seeds[idx], idx))
            assert sub.dtype == full.dtype and sub.shape == (idx.size, seats)
            assert np.array_equal(sub, full[idx])
        # A runout hand's EV differs from its realized payout (almost surely).
        realized = np.asarray(be.payouts_batch())
        runouts += int((full[term] != realized[term]).any(axis=1).sum())
        if env._dones.any():
            env.reset_terminal_batch(
                rng.integers(0, 2**63 - 1, size=n, dtype=np.int64).astype(np.uint64),
                rng.integers(0, seats, size=n).astype(np.uint8),
                env._dones.copy(),
            )
    assert runouts > 0, "the drive never produced an all-in runout"


def test_subset_validation() -> None:
    be = BatchedEngine(4, num_seats=3, opp_outcome_mc=0)
    be.reset_batch(np.arange(4, dtype=np.uint64), np.zeros(4, dtype=np.uint8))
    seeds = np.arange(2, dtype=np.uint64)
    with pytest.raises(ValueError):
        be.payouts_ev_subset(64, seeds, np.asarray([0], dtype=np.int64))
    with pytest.raises(ValueError):
        be.payouts_ev_subset(64, seeds, np.asarray([0, 4], dtype=np.int64))
    with pytest.raises(ValueError):
        be.payouts_ev_subset(64, seeds, np.asarray([-1, 2], dtype=np.int64))
    empty = np.asarray(
        be.payouts_ev_subset(64, np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.int64))
    )
    assert empty.shape == (0, 3)
