"""NLH (`nlh_single`) batched engine + encoder — parity tests.

The batched NLH path must be indistinguishable from the serial one: same
deals, same masks/bounds, and BIT-EXACT observations from
`encode_observation_batch_nlh` vs the scalar `encode_observation_nlh` at
every decision point. Coverage deliberately spans the NLH-only surface:

  - the live preflop street (blind flags, hole-class block, empty board,
    zero opp-outcome) — the very first observation after reset;
  - blinds/actor order heads-up (button = SB) and multiway;
  - raises/folds/all-ins driven by a scripted deterministic policy so
    histories carry all four gates, last-aggressor, pot odds, bet-faced,
    and the log1p pot-fraction scaling;
  - a deep heads-up min-raise war that overflows the 40-slot history
    window (the packer must hand the encoder exactly the NEWEST 40);
  - terminal rewards vs serial payouts;
  - reset_terminal_batch (the _refresh_subset path) re-encoding parity.
"""

import numpy as np
import pytest

from plo5bp.actions import (
    GATE_CHECK_CALL,
    GATE_FOLD,
    GATE_RAISE,
)
from plo5bp.config import GameConfig
from plo5bp.encoding_nlh import OBS_DIM_NLH
from plo5bp.env import BombPotEnv
from plo5bp.env_batched import BatchedBombPotEnv


def _cfg(num_seats=6, stack=1_000_000):
    return GameConfig.nlh_default(num_seats=num_seats, starting_stack=stack)


def _scripted(k: int, gate_mask, min_raise: int, max_raise: int):
    """Deterministic per-decision policy, a pure function of the decision
    ordinal + legality — identical inputs on the serial and batched paths
    therefore produce identical actions. Mixes all gates: min-raises,
    occasional shoves, folds vs bets, calls/checks."""
    if gate_mask[GATE_RAISE] and min_raise > 0:
        if k % 5 == 0:
            return GATE_RAISE, int(min_raise)
        if k % 11 == 4:
            return GATE_RAISE, int(max_raise)  # jam
    if gate_mask[GATE_FOLD] and k % 7 == 3:
        return GATE_FOLD, 0
    return GATE_CHECK_CALL, 0


def test_batched_accepts_nlh_and_shapes():
    cfg = _cfg()
    N = 6
    b = BatchedBombPotEnv(N, cfg)
    assert b.obs_dim == OBS_DIM_NLH
    seeds = np.arange(N, dtype=np.uint64)
    buttons = np.zeros(N, dtype=np.uint8)
    step = b.reset_batch(seeds, buttons)
    assert step.obs.shape == (N, OBS_DIM_NLH)
    ah = b._be.all_hole_cards_batch()
    assert ah.shape == (N, 6, 2)
    d = b._be.observation_arrays()
    assert d["hero_hole"].shape == (N, 2)
    assert ((d["hero_hole"] < 52).sum(axis=1) == 2).all()


def test_batched_deal_parity_vs_serial():
    cfg = _cfg()
    N = 8
    b = BatchedBombPotEnv(N, cfg)
    seeds = np.arange(100, 100 + N, dtype=np.uint64)
    buttons = (np.arange(N) % 6).astype(np.uint8)
    b.reset_batch(seeds, buttons)
    ah = b._be.all_hole_cards_batch()
    for i in (0, 3, 7):
        env = BombPotEnv(cfg)
        env.reset(int(seeds[i]), int(buttons[i]))
        assert np.array_equal(
            np.asarray(env.all_hole_cards(), dtype=np.uint8), ah[i]
        )


def test_reset_obs_is_preflop_and_bit_exact():
    """The very first batched observation is a live PREFLOP node — blind
    flags, hole-class, empty board — and must equal the serial encoding."""
    for num_seats in (2, 3, 6):
        cfg = _cfg(num_seats=num_seats)
        N = 6
        b = BatchedBombPotEnv(N, cfg)
        seeds = np.arange(200, 200 + N, dtype=np.uint64)
        buttons = (np.arange(N) % num_seats).astype(np.uint8)
        step = b.reset_batch(seeds, buttons)
        for i in range(N):
            env = BombPotEnv(cfg)
            o, info = env.reset(int(seeds[i]), int(buttons[i]))
            assert np.array_equal(step.obs[i], o), (
                f"{num_seats}-max reset obs mismatch env {i}"
            )
            # Preflop street one-hot set; board multi-hot empty.
            assert step.obs[i, 104] == 1.0
            assert step.obs[i, 52:104].sum() == 0.0
            # Exactly one SB flag and one BB flag across the two dims —
            # heads-up the button posts SB.
            assert np.array_equal(
                step.obs[i, 993:995], o[993:995]
            )
            assert np.array_equal(step.gate_mask[i], info.gate_mask)
            assert int(step.min_raise[i]) == info.min_raise_chips
            assert int(step.max_raise[i]) == info.max_raise_chips


@pytest.mark.parametrize("num_seats", [2, 3, 6])
def test_batched_obs_bit_exact_scripted_full_hands(num_seats):
    """Drive serial and batched with the same scripted policy through
    complete hands (preflop → showdown/fold-out) and require bit-exact
    observations, masks, bounds, and terminal rewards at every step."""
    cfg = _cfg(num_seats=num_seats)
    N = 10
    batched = BatchedBombPotEnv(N, cfg)
    seeds = np.arange(3000, 3000 + N, dtype=np.uint64) * np.uint64(7919)
    buttons = (np.arange(N) % num_seats).astype(np.uint8)
    step = batched.reset_batch(seeds, buttons)

    serial = []
    for i in range(N):
        env = BombPotEnv(cfg)
        o, info = env.reset(int(seeds[i]), int(buttons[i]))
        assert np.array_equal(step.obs[i], o), f"reset obs mismatch env {i}"
        serial.append({"env": env, "info": info, "done": False, "k": 0})

    for t in range(120):
        if all(s["done"] for s in serial):
            break
        gates = np.full(N, GATE_CHECK_CALL, dtype=np.uint8)
        chips = np.zeros(N, dtype=np.uint64)
        for i, s in enumerate(serial):
            if s["done"]:
                continue
            g, c = _scripted(
                s["k"],
                s["info"].gate_mask,
                s["info"].min_raise_chips,
                s["info"].max_raise_chips,
            )
            gates[i], chips[i] = g, c
            s["k"] += 1
        step = batched.step_hybrid_batch(gates, chips.astype(np.uint64))
        for i, s in enumerate(serial):
            if s["done"]:
                continue
            o2, rewards, d, info2 = s["env"].step_hybrid(
                int(gates[i]), int(chips[i])
            )
            s["done"] = d
            s["info"] = info2
            if d:
                assert step.newly_terminal[i], f"step {t} env {i}: done drift"
                assert np.allclose(step.rewards[i], rewards), (
                    f"step {t} env {i}: terminal rewards mismatch"
                )
            else:
                assert np.array_equal(step.obs[i], o2), (
                    f"step {t} env {i} ({num_seats}-max): batched obs != serial"
                )
                assert np.array_equal(step.gate_mask[i], info2.gate_mask)
                assert int(step.min_raise[i]) == info2.min_raise_chips
                assert int(step.max_raise[i]) == info2.max_raise_chips
    assert all(s["done"] for s in serial), "hands did not complete in 120 steps"


def test_history_overflow_past_40_stays_bit_exact():
    """Deep heads-up min-raise war: the hand accumulates > 40 history
    records, so the packer's 40-slot window must hold exactly the NEWEST
    40 (a wider or misaligned window shifts every history feature)."""
    cfg = _cfg(num_seats=2, stack=8_000_000)  # 800bb
    N = 2
    batched = BatchedBombPotEnv(N, cfg)
    seeds = np.asarray([42, 43], dtype=np.uint64)
    buttons = np.zeros(N, dtype=np.uint8)
    step = batched.reset_batch(seeds, buttons)
    serial = []
    for i in range(N):
        env = BombPotEnv(cfg)
        o, info = env.reset(int(seeds[i]), int(buttons[i]))
        assert np.array_equal(step.obs[i], o)
        serial.append({"env": env, "info": info, "done": False})

    longest = 0
    for t in range(300):
        if all(s["done"] for s in serial):
            break
        gates = np.full(N, GATE_CHECK_CALL, dtype=np.uint8)
        chips = np.zeros(N, dtype=np.uint64)
        for i, s in enumerate(serial):
            if s["done"]:
                continue
            info = s["info"]
            if info.gate_mask[GATE_RAISE] and info.min_raise_chips > 0:
                gates[i] = GATE_RAISE
                chips[i] = info.min_raise_chips
        step = batched.step_hybrid_batch(gates, chips)
        for i, s in enumerate(serial):
            if s["done"]:
                continue
            o2, rewards, d, info2 = s["env"].step_hybrid(
                int(gates[i]), int(chips[i])
            )
            s["done"] = d
            s["info"] = info2
            longest = max(longest, len(info2.raw_obs.get("history", [])))
            if not d:
                assert np.array_equal(step.obs[i], o2), (
                    f"step {t} env {i}: obs diverged at history len "
                    f"{len(info2.raw_obs.get('history', []))}"
                )
    assert longest > 40, (
        f"war too short to exercise the window (max history {longest})"
    )


def test_reset_terminal_batch_reencodes_bit_exact():
    """Checkdown a wave to terminal, re-seed HALF the envs via
    reset_terminal_batch, and require the refreshed rows to match fresh
    serial resets (exercises the _refresh_subset NLH branch)."""
    cfg = _cfg(num_seats=3)
    N = 6
    batched = BatchedBombPotEnv(N, cfg)
    seeds = np.arange(500, 500 + N, dtype=np.uint64)
    buttons = (np.arange(N) % 3).astype(np.uint8)
    batched.reset_batch(seeds, buttons)
    for _ in range(80):
        if batched.is_terminal().all():
            break
        batched.step_hybrid_batch(
            np.full(N, GATE_CHECK_CALL, dtype=np.uint8),
            np.zeros(N, dtype=np.uint64),
        )
    assert batched.is_terminal().all()

    new_seeds = np.arange(900, 900 + N, dtype=np.uint64)
    mask = np.zeros(N, dtype=bool)
    mask[::2] = True
    step = batched.reset_terminal_batch(new_seeds, buttons, mask)
    for i in range(0, N, 2):
        env = BombPotEnv(cfg)
        o, _ = env.reset(int(new_seeds[i]), int(buttons[i]))
        assert np.array_equal(step.obs[i], o), f"subset re-encode env {i}"


def test_nlh_opp_outcome_populated_postflop():
    """The bundle's 3-dim exhaustive opp-outcome must be live postflop
    (sums to 1) and zero preflop, matching the serial feature dims."""
    cfg = _cfg(num_seats=2)
    N = 2
    b = BatchedBombPotEnv(N, cfg)
    b.reset_batch(np.asarray([7, 8], dtype=np.uint64), np.zeros(N, dtype=np.uint8))
    # Preflop: zeros.
    assert b._obs[:, 984:987].sum() == 0.0
    # Call/check through preflop to reach the flop.
    for _ in range(30):
        if (b._street >= 1).all() or b.is_terminal().all():
            break
        b.step_hybrid_batch(
            np.full(N, GATE_CHECK_CALL, dtype=np.uint8),
            np.zeros(N, dtype=np.uint64),
        )
    live = ~b.is_terminal()
    assert live.any()
    sums = b._obs[live, 984:987].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5), f"opp outcome sums {sums}"
