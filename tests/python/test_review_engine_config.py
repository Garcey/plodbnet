"""Config / env fixes from the 2026-09-20 code review (B8, C4 Python side,
and the C8 "minor" env items).

Everything here runs against ANY build of the extension — it exercises the
Python layer only. The engine-rule regressions that need the rebuilt
`_engine` live in `test_review_engine_rules.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from plo5bp.config import (
    MAX_SEATS,
    MIN_SEATS,
    VARIANT_NLH,
    VARIANT_PLO4,
    VARIANT_PLO5,
    VARIANT_PLO6,
    GameConfig,
    TrainingConfig,
)
from plo5bp.env import BombPotEnv
from plo5bp.env_batched import BatchedBombPotEnv


# ---- B8 / C4: GameConfig rejects what the engine / encoders cannot hold ----


@pytest.mark.parametrize("n", [-1, 0, 1, 9, 10, 23])
def test_num_seats_outside_2_to_8_is_a_value_error(n: int) -> None:
    # 1 seat used to panic in Rust ("need at least 2 seats"; PanicException is
    # a BaseException), 9+ silently corrupted the observation (seat 8's
    # active flag lands in the all-in block).
    with pytest.raises(ValueError, match="num_seats"):
        GameConfig(num_seats=n)
    with pytest.raises(ValueError, match="num_seats"):
        GameConfig(num_seats=n, variant=VARIANT_NLH, sb=5_000)


def test_seat_bounds_are_the_documented_ones() -> None:
    assert (MIN_SEATS, MAX_SEATS) == (2, 8)
    for variant in (VARIANT_PLO4, VARIANT_PLO5, VARIANT_NLH):
        for n in (MIN_SEATS, MAX_SEATS):
            assert GameConfig(num_seats=n, variant=variant).num_seats == n


def test_plo6_is_capped_at_seven_seats_by_the_deck() -> None:
    # 8 x 6 hole + 10 board = 58 cards: the deal used to index past the deck
    # and panic (cards.rs deal_one).
    with pytest.raises(ValueError, match="max 7 seats"):
        GameConfig(num_seats=8, variant=VARIANT_PLO6)
    cfg = GameConfig(num_seats=7, variant=VARIANT_PLO6)
    assert cfg.num_seats * cfg.hole_count + 10 == 52


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"bb": 0}, "bb must be positive"),
        ({"bb": -10_000}, "bb must be positive"),
        ({"ante": -1}, "ante must be >= 0"),
        ({"sb": -1}, "sb must be >= 0"),
        ({"starting_stack": -1}, "starting_stack must be >= 0"),
        (
            {"num_seats": 3, "starting_stacks": (200_000, -5, 200_000)},
            "starting_stacks must be >= 0",
        ),
        ({"num_seats": 3, "starting_stacks": (1, 2)}, "starting_stacks length"),
        ({"variant": "plo7_triple_bomb"}, "unknown variant"),
    ],
)
def test_bad_chip_values_are_value_errors(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        GameConfig(**kwargs)


def test_degenerate_but_legal_configs_still_build() -> None:
    # Zero ante (UI default-able) and zero stacks (a busted home-game seat
    # that the in-hand mask sits out) are legal; only negatives are not.
    assert GameConfig(ante=0).ante == 0
    assert GameConfig(num_seats=3, starting_stacks=(0, 200_000, 200_000)).resolved_stacks[0] == 0
    nlh = GameConfig.nlh_default(num_seats=8)
    assert (nlh.sb, nlh.bb, nlh.ante, nlh.hole_count) == (5_000, 10_000, 5_000, 2)


def test_largest_tables_deal_without_touching_the_deck_bound() -> None:
    # The edge of every variant's range goes through the real engine: all
    # cards distinct, no panic.
    for variant, n in (
        (VARIANT_PLO4, 8),
        (VARIANT_PLO5, 8),
        (VARIANT_PLO6, 7),
        (VARIANT_NLH, 8),
    ):
        sb = 5_000 if variant == VARIANT_NLH else 0
        env = BombPotEnv(GameConfig(num_seats=n, variant=variant, sb=sb))
        env.reset(123, n - 1)
        holes = env.all_hole_cards()
        flat = [c for h in holes for c in h]
        assert len(holes) == n and len(set(flat)) == len(flat)


def test_training_config_drains_inflight_hands_by_default() -> None:
    # review 2026-09-20 A6 — read by the collectors via getattr(..., True).
    assert TrainingConfig().drain_inflight is True
    assert TrainingConfig(drain_inflight=False).drain_inflight is False


# ---- C8 minors: env bookkeeping ----


class _ObsDictSpy:
    """Forwarding proxy around the Rust GameState that records how
    `observation_dict` was called."""

    def __init__(self, rs) -> None:
        self._rs = rs
        self.calls: list[bool] = []

    def observation_dict(self, skip_outcome_mc: bool = False):
        self.calls.append(bool(skip_outcome_mc))
        return self._rs.observation_dict(skip_outcome_mc=skip_outcome_mc)

    def __getattr__(self, name):
        return getattr(self._rs, name)


def test_step_runs_the_outcome_mc_once_not_three_times() -> None:
    # `_read_total_commit` (pre + post step) only needs `total_commit`; it
    # used to run the full 1024-sample MC on both reads.
    env = BombPotEnv(GameConfig())
    env.reset(7, 0)
    spy = _ObsDictSpy(env._rs)
    env._rs = spy
    _, _, done, info = env.step_hybrid(1, 0)
    assert not done
    assert spy.calls == [True, True, False], spy.calls
    assert int(info.total_commit.sum()) == 6 * 30_000
    assert info.commit_delta.tolist() == [0] * 6


def test_terminal_step_never_runs_the_outcome_mc() -> None:
    env = BombPotEnv(GameConfig(num_seats=2))
    _, info = env.reset(11, 0)
    spy = _ObsDictSpy(env._rs)
    env._rs = spy
    # Seat 1 pots it, seat 0 folds → terminal.
    env.step_hybrid(2, int(info.max_raise_chips))
    del spy.calls[:]
    _, rewards, done, info = env.step_hybrid(0, 0)
    assert done and info.terminal
    assert spy.calls == [True, True, True], spy.calls
    assert float(rewards.sum()) == 0.0
    # Same key set as a live node's raw dict (minus the two actor-only
    # hero_category keys `_pack_obs` adds).
    live = BombPotEnv(GameConfig(num_seats=2))
    _, live_info = live.reset(11, 0)
    assert set(info.raw_obs) == set(live_info.raw_obs) - {
        "hero_category_a",
        "hero_category_b",
    }


def test_skip_flag_does_not_change_observations() -> None:
    # The serial env's observations still carry the full-fidelity MC dims:
    # only the bookkeeping reads skip it.
    env = BombPotEnv(GameConfig())
    obs, info = env.reset(7, 0)
    direct = np.asarray(env._rs.observation_dict()["opp_outcome_fractions"])
    assert direct[4:].any(), "MC arms must be populated on the flop"
    np.testing.assert_array_equal(
        np.asarray(info.raw_obs["opp_outcome_fractions"]), direct
    )


def test_batched_env_pot_cache_exists_before_the_first_refresh() -> None:
    cfg = GameConfig(num_seats=3)
    env = BatchedBombPotEnv(4, cfg, opp_outcome_mc=8)
    # Latent AttributeError: `_pot` was only created inside `_unpack_post`.
    assert env._pot.shape == (4,) and env._pot.dtype == np.uint64
    assert not env._pot.any()
    env.reset_batch(np.arange(4, dtype=np.uint64), np.zeros(4, dtype=np.uint8))
    assert env._pot.tolist() == [3 * 30_000] * 4
    # Partial refresh scatters into it in place.
    env._refresh_subset(np.array([True, False, True, False]))
    assert env._pot.tolist() == [3 * 30_000] * 4
    # Reconfigure invalidates it along with the other caches.
    env.reconfigure(GameConfig(num_seats=3, ante=10_000))
    assert env._pot.shape == (4,) and not env._pot.any()
    env.reset_batch(np.arange(4, dtype=np.uint64), np.zeros(4, dtype=np.uint8))
    assert env._pot.tolist() == [3 * 10_000] * 4
