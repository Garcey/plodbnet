"""Batched opponent assignment (`rollout._draw_pool_mix`, 2026-09-23).

The batched collector used to assign pool opponents one finished hand at a
time in a Python loop (~3 numpy RNG calls per hand, 4-10 ms per step at the
vMin1 table count). The draws are now batched. That changes the RNG STREAM
(not bit-identical to the old collector) but must keep the DISTRIBUTION the
per-hand loop drew — pinned here.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from plo5bp.rollout import _draw_pool_mix


def test_distribution_matches_the_per_hand_rule() -> None:
    k, n_seats, pool_size, opp_seats, p = 240_000, 6, 5, 2, 0.5
    snap, mask = _draw_pool_mix(np.random.default_rng(0), k, n_seats, pool_size, opp_seats, p)
    assert snap.shape == (k,) and mask.shape == (k, n_seats) and mask.dtype == bool
    mixed = snap >= 0
    assert abs(mixed.mean() - p) < 0.005
    # self-play hands: every seat is a learner seat
    assert mask[~mixed].all()
    # mixed hands: exactly `opp_seats` opponent seats, one snapshot each
    assert ((~mask[mixed]).sum(axis=1) == opp_seats).all()
    assert snap[mixed].min() == 0 and snap[mixed].max() == pool_size - 1
    for i in range(pool_size):  # snapshots uniform
        assert abs((snap[mixed] == i).mean() - 1 / pool_size) < 0.006
    opp = ~mask[mixed]
    for s in range(n_seats):  # each seat an opponent 2/6 of the time
        assert abs(opp[:, s].mean() - opp_seats / n_seats) < 0.006
    # the opponent PAIR is uniform over all C(6, 2) = 15 pairs
    code = (opp * (1 << np.arange(n_seats))).sum(axis=1)
    for a, b in itertools.combinations(range(n_seats), 2):
        assert abs((code == (1 << a) + (1 << b)).mean() - 1 / 15) < 0.004


@pytest.mark.parametrize(
    "pool_size,opp_seats,prob", [(0, 2, 0.5), (4, 0, 0.5), (4, 2, 0.0)]
)
def test_inactive_pool_mix_is_all_self_play_and_draws_nothing(pool_size, opp_seats, prob) -> None:
    rng = np.random.default_rng(3)
    before = rng.bit_generator.state
    snap, mask = _draw_pool_mix(rng, 50, 6, pool_size, opp_seats, prob)
    assert (snap == -1).all() and mask.all()
    assert rng.bit_generator.state == before


def test_always_mixed_and_empty_batch() -> None:
    snap, mask = _draw_pool_mix(np.random.default_rng(1), 1000, 3, 2, 2, 1.0)
    assert (snap >= 0).all() and ((~mask).sum(axis=1) == 2).all()  # one learner seat left
    snap, mask = _draw_pool_mix(np.random.default_rng(1), 0, 6, 3, 2, 0.5)
    assert snap.shape == (0,) and mask.shape == (0, 6)
