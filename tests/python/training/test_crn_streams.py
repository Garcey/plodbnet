"""Common random numbers for paired comparisons (--crn-streams, 2026-09-28, ML-004).

With the one shared stream, a re-deal WAVE draws n_envs seeds, so env j's
k-th hand depends on how many waves came before it -- on the policy -- and two
recipe candidates stop seeing the same hands at their first different hand
length. `rollout._CrnDeals` makes env j's k-th hand (deal seed, button,
opponent assignment) a pure function of (key, j, k).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from plo5bp import rollout as R
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM_MINIMAL
from plo5bp.env_batched import BatchedBombPotEnv
from plo5bp.network import ActorCriticV5
from plo5bp.selfplay import OpponentPool


def test_a_hands_numbers_depend_only_on_key_env_and_hand() -> None:
    a = R._CrnDeals((7, 12, 3), 50)
    b = R._CrnDeals((7, 12, 3), 50)
    # a deals every env three times in one call each; b in scattered groups
    seq_a = [a.deal(np.arange(50), 6) for _ in range(3)]
    got = {j: [] for j in range(50)}
    for ids in ([3, 9, 40], list(range(50)), [0, 1, 2], [3, 9, 40]):
        s, btn = b.deal(np.array(ids), 6)
        for j, x, y in zip(ids, s, btn):
            got[j].append((int(x), int(y)))
    for j in range(50):
        want = [(int(seq_a[k][0][j]), int(seq_a[k][1][j])) for k in range(3)]
        assert got[j] == want[: len(got[j])]
    assert all(s < 2**63 for s in seq_a[0][0]) and seq_a[0][1].max() < 6
    # another key, other hands
    c = R._CrnDeals((7, 13, 3), 50)
    assert not np.array_equal(c.deal(np.arange(50), 6)[0], seq_a[0][0])


def test_opponent_assignment_has_the_draw_pool_mix_distribution() -> None:
    crn = R._CrnDeals((1, 2, 3), 40_000)
    ids = np.arange(40_000)
    snap, mask = crn.pool_mix(ids, 6, 5, 2, 0.5)
    mixed = snap >= 0
    assert abs(mixed.mean() - 0.5) < 0.01
    assert set(np.unique(snap[mixed])) == set(range(5))
    assert (mask[mixed].sum(axis=1) == 4).all() and mask[~mixed].all()
    # each seat is an opponent seat 2/6 of the time
    assert np.allclose((~mask[mixed]).mean(axis=0), 2 / 6, atol=0.02)
    # the same hand's assignment again: identical (it is keyed, not drawn)
    snap2, mask2 = crn.pool_mix(ids, 6, 5, 2, 0.5)
    assert np.array_equal(snap, snap2) and np.array_equal(mask, mask2)
    # no pool / no opponent seats: pure self-play, like _draw_pool_mix
    s0, m0 = crn.pool_mix(ids[:5], 6, 0, 2, 0.5)
    assert (s0 == -1).all() and m0.all()


class _DealSpy:
    """Records every seed/button each env is dealt (a proxy of env._be)."""

    def __init__(self, be) -> None:
        self._be = be
        self.log: dict[int, list] = defaultdict(list)

    def __getattr__(self, name):
        return getattr(self._be, name)

    def reset_batch(self, seeds, buttons):
        for j, (s, b) in enumerate(zip(seeds, buttons)):
            self.log[j].append((int(s), int(b)))
        return self._be.reset_batch(seeds, buttons)

    def reset_terminal_batch(self, seeds, buttons, mask):
        for j in np.nonzero(mask)[0]:
            self.log[int(j)].append((int(seeds[j]), int(buttons[j])))
        return self._be.reset_terminal_batch(seeds, buttons, mask)


def _hands(learner_seed: int, crn_key) -> dict:
    torch.manual_seed(learner_seed)
    learner = ActorCriticV5(hidden_dim=16, obs_dim=OBS_DIM_MINIMAL, num_layers=3,
                            torso_layernorm=True)
    cfg = TrainingConfig(num_envs=24, rollout_length=600, obs_mode="minimal")
    game = GameConfig(num_seats=4, starting_stack=400_000, ante=30_000, bb=10_000)
    env = BatchedBombPotEnv(24, game, obs_mode="minimal", opp_outcome_mc=0)
    env._be = spy = _DealSpy(env._be)
    R._clear_rollout_buffers()
    R.collect_rollout_batched(
        learner, OpponentPool(capacity=2, seed=0), game, cfg,
        np.random.default_rng(learner_seed), env=env, crn_key=crn_key,
    )
    return spy.log


def test_different_policies_play_the_same_hands_under_crn() -> None:
    a, b = _hands(1, (5, 100, 0)), _hands(2, (5, 100, 0))
    for j in range(24):
        n = min(len(a[j]), len(b[j]))
        assert n >= 2 and a[j][:n] == b[j][:n], f"env {j}"
    # the shared stream: the two policies' hands part ways
    x, y = _hands(1, None), _hands(2, None)
    assert any(
        x[j][: min(len(x[j]), len(y[j]))] != y[j][: min(len(x[j]), len(y[j]))]
        for j in range(24)
    )
