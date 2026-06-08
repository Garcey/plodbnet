"""Bit-exact parity: `_refresh_subset` vs full `_refresh` after reset.

`collect_rollout_batched` replaced the post-reset full `env._refresh()` with
`env._refresh_subset(reset_mask)` — re-packing/re-encoding only the envs that
`reset_terminal_batch` actually changed. This is only correct if the subset
refresh produces cached arrays byte-identical to a full refresh whenever the
non-masked envs' engine state is unchanged (which holds after
`reset_terminal_batch`, since it mutates only masked envs).

Strategy: drive two identical batched engines in lockstep with the SAME
random-but-LEGAL actions (sampled from the engine's own gate mask), asserting
they stay identical every step. When a step produces a `newly_terminal` mask,
apply `reset_terminal_batch` to both, then a full `_refresh()` to one and
`_refresh_subset(mask)` to the other, and assert EVERY cached array is exactly
equal. Covers mixed masks, all-true masks, and the empty-mask no-op.
"""

from __future__ import annotations

import numpy as np

from plo5bp.actions import GATE_FOLD, GATE_CHECK_CALL, GATE_RAISE
from plo5bp.config import GameConfig
from plo5bp.env_batched import BatchedBombPotEnv

# Every cached array the rollout reads from the env each step.
_CACHED = (
    "_obs",
    "_legal",
    "_gate_mask",
    "_min_raise",
    "_max_raise",
    "_actors",
    "_dones",
    "_total_commit",
    "_bet_to_call",
    "_street_commit",
    "_street",
)


def _assert_envs_equal(a: BatchedBombPotEnv, b: BatchedBombPotEnv, ctx: str) -> None:
    for name in _CACHED:
        av = getattr(a, name)
        bv = getattr(b, name)
        assert av.dtype == bv.dtype, f"{ctx}: dtype mismatch {name}: {av.dtype} vs {bv.dtype}"
        assert av.shape == bv.shape, f"{ctx}: shape mismatch {name}: {av.shape} vs {bv.shape}"
        assert np.array_equal(av, bv), f"{ctx}: value mismatch in {name}"


def _make_pair(n: int, config: GameConfig, base_seed: int):
    full = BatchedBombPotEnv(n, config)
    sub = BatchedBombPotEnv(n, config)
    rng = np.random.default_rng(base_seed)
    seeds = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64).astype(np.uint64)
    buttons = rng.integers(0, config.num_seats, size=n, dtype=np.int64).astype(np.uint8)
    full.reset_batch(seeds, buttons)
    sub.reset_batch(seeds, buttons)
    _assert_envs_equal(full, sub, "after reset_batch")
    return full, sub, rng


def _sample_legal_actions(env: BatchedBombPotEnv, rng: np.random.Generator):
    """Per env, pick a uniformly-random LEGAL gate from the cached gate_mask
    and a legal chip amount for raises. Terminal envs get a no-op (CheckCall,
    ignored by the engine)."""
    n = env.n
    gm = env._gate_mask  # (n, 3) bool
    gates = np.full(n, GATE_CHECK_CALL, dtype=np.uint8)
    chips = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        legal = np.nonzero(gm[i])[0]
        if legal.size == 0:
            continue  # terminal env — no-op
        g = int(rng.choice(legal))
        gates[i] = g
        if g == GATE_RAISE:
            lo = int(env._min_raise[i])
            hi = int(env._max_raise[i])
            if hi <= lo:
                chips[i] = np.uint64(max(lo, 0))
            else:
                chips[i] = np.uint64(int(rng.integers(lo, hi + 1)))
    return gates, chips


def _drive_until_mask(full, sub, rng, *, want, max_steps=200):
    """Step both engines in lockstep with identical legal actions until a
    `newly_terminal` mask matching `want` appears. `want` is 'mixed',
    'all', or 'any'. Returns the mask. Asserts lockstep every step."""
    n = full.n
    for _ in range(max_steps):
        gates, chips = _sample_legal_actions(full, rng)
        sf = full.step_hybrid_batch(gates, chips)
        ss = sub.step_hybrid_batch(gates, chips)
        assert np.array_equal(sf.newly_terminal, ss.newly_terminal), "lockstep diverged"
        _assert_envs_equal(full, sub, "lockstep step")
        m = np.asarray(sf.newly_terminal, dtype=bool)
        cnt = int(m.sum())
        if want == "any" and cnt > 0:
            return m
        if want == "mixed" and 0 < cnt < n:
            return m
        if want == "all" and cnt == n:
            return m
    raise AssertionError(f"did not reach a '{want}' newly_terminal mask in {max_steps} steps")


def _apply_reset_and_refresh(full, sub, mask, rng):
    """Mirror the rollout's terminal block exactly: identical raw Rust reset
    on both, then full `_refresh()` on `full` and `_refresh_subset(mask)` on
    `sub`."""
    n = full.n
    new_seeds = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64).astype(np.uint64)
    new_buttons = rng.integers(0, full.num_seats, size=n, dtype=np.int64).astype(np.uint8)
    mask_b = np.ascontiguousarray(mask, dtype=bool)

    full._be.reset_terminal_batch(new_seeds, new_buttons, mask_b)
    full._reset_seeds = np.where(mask_b, new_seeds, full._reset_seeds)
    full._refresh()

    sub._be.reset_terminal_batch(new_seeds, new_buttons, mask_b)
    sub._reset_seeds = np.where(mask_b, new_seeds, sub._reset_seeds)
    sub._refresh_subset(mask_b)


def test_refresh_subset_mixed_mask_two_seat() -> None:
    """Mixed reset mask (some envs terminal, some not) — the real rollout case."""
    config = GameConfig(num_seats=2, starting_stack=200000, ante=30000, bb=10000)
    full, sub, rng = _make_pair(16, config, base_seed=100)
    mask = _drive_until_mask(full, sub, rng, want="mixed")
    assert mask.any() and not mask.all()
    _apply_reset_and_refresh(full, sub, mask, rng)
    _assert_envs_equal(full, sub, "mixed mask (2-seat) after reset+refresh")


def test_refresh_subset_three_seat_mixed() -> None:
    """Mixed mask at 3 seats — exercises per-seat packed fields at a different
    seat count."""
    config = GameConfig(num_seats=3, starting_stack=400000, ante=30000, bb=10000)
    full, sub, rng = _make_pair(16, config, base_seed=400)
    mask = _drive_until_mask(full, sub, rng, want="mixed")
    assert mask.any() and not mask.all()
    _apply_reset_and_refresh(full, sub, mask, rng)
    _assert_envs_equal(full, sub, "mixed mask (3-seat) after reset+refresh")


def test_refresh_subset_six_seat_mixed() -> None:
    """Mixed mask at 6 seats (the live training max)."""
    config = GameConfig(num_seats=6, starting_stack=400000, ante=30000, bb=10000)
    full, sub, rng = _make_pair(24, config, base_seed=600)
    mask = _drive_until_mask(full, sub, rng, want="mixed")
    assert mask.any() and not mask.all()
    _apply_reset_and_refresh(full, sub, mask, rng)
    _assert_envs_equal(full, sub, "mixed mask (6-seat) after reset+refresh")


def test_refresh_subset_all_terminal() -> None:
    """All-true mask: subset refresh re-packs every row → must equal full."""
    config = GameConfig(num_seats=2, starting_stack=200000, ante=30000, bb=10000)
    # n=1 so a single terminal step yields an all-true mask deterministically.
    full, sub, rng = _make_pair(1, config, base_seed=222)
    mask = _drive_until_mask(full, sub, rng, want="all")
    assert mask.all()
    _apply_reset_and_refresh(full, sub, mask, rng)
    _assert_envs_equal(full, sub, "all-terminal mask after reset+refresh")


def test_refresh_subset_empty_mask_is_noop() -> None:
    """Empty mask: `_refresh_subset` is a no-op; a full refresh of the same
    unchanged state must leave arrays identical to the untouched subset env."""
    config = GameConfig(num_seats=3, starting_stack=300000, ante=30000, bb=10000)
    full, sub, _ = _make_pair(4, config, base_seed=300)
    mask = np.zeros(4, dtype=bool)
    full._refresh()            # full re-pack of unchanged state
    sub._refresh_subset(mask)  # no-op
    _assert_envs_equal(full, sub, "empty mask: full refresh vs subset no-op")
