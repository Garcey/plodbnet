"""PLO6 double-board bomb pot (`plo6_double_bomb`) — engine + encoding tests.

The variant is PLO5-double-board with 6 hole cards and nothing else changed:
same OBS_DIM 991 (hole cards live in a 52-dim multi-hot), same 11-anchor
pot-limit sizing head, same bomb-pot street structure. These tests pin:
  - 6-card dealing (unique cards, deck accounting), seed determinism;
  - serial obs shape/multi-hot correctness + full-hand playthrough;
  - hero_category parity with the pure-python made-hand describer;
  - batched engine support with bit-exact batched==serial obs parity
    (the contract that lets PLO6 train on the 49k-env batched path);
  - PL sizing parity with PLO5 (the cap must not depend on hole count);
  - study mode rejection (UI phase not built yet).
"""

import numpy as np
import pytest

from plo5bp.actions import GATE_CHECK_CALL, GATE_RAISE
from plo5bp.config import GameConfig, VARIANT_PLO5, VARIANT_PLO6
from plo5bp.encoding import OBS_DIM
from plo5bp.env import BombPotEnv
from plo5bp.env_batched import BatchedBombPotEnv
from plo5bp.ui.hand_describe import best_hand_category


def _cfg(num_seats=6, stack=200000, variant=VARIANT_PLO6):
    return GameConfig(
        num_seats=num_seats, starting_stack=stack, ante=30000, bb=10000,
        variant=variant,
    )


def test_config_accepts_plo6():
    cfg = _cfg()
    assert cfg.variant == "plo6_double_bomb"
    with pytest.raises(ValueError):
        GameConfig(variant="plo7_double_bomb")


def test_serial_deals_six_unique_cards_and_obs_dims():
    env = BombPotEnv(_cfg())
    obs, info = env.reset(123, 0)
    holes = env.all_hole_cards()
    assert len(holes) == 6 and all(len(h) == 6 for h in holes)
    flat = [c for h in holes for c in h]
    assert len(set(flat)) == 36, "all dealt hole cards unique"
    assert obs.shape == (OBS_DIM,)
    assert int(obs[:52].sum()) == 6, "hero multi-hot must carry 6 bits"


def test_serial_seed_determinism():
    e1, e2 = BombPotEnv(_cfg()), BombPotEnv(_cfg())
    e1.reset(777, 3)
    e2.reset(777, 3)
    assert e1.all_hole_cards() == e2.all_hole_cards()


def test_full_hand_checkdown_zero_sum():
    env = BombPotEnv(_cfg())
    env.reset(7, 2)
    done, steps = False, 0
    while not done and steps < 200:
        obs, rewards, done, info = env.step_hybrid(GATE_CHECK_CALL, 0)
        steps += 1
    assert done
    assert abs(sum(rewards)) < 1e-6


def test_raise_path_runs_and_respects_pot_limit():
    env = BombPotEnv(_cfg())
    obs, info = env.reset(11, 0)
    assert info.gate_mask[GATE_RAISE], "raise legal on the flop"
    hi = int(info.raw_obs["max_raise"])
    obs, rewards, done, info = env.step_hybrid(GATE_RAISE, hi)
    assert info.actor is not None or done


def test_pot_limit_cap_matches_plo5():
    e5 = BombPotEnv(_cfg(variant=VARIANT_PLO5))
    e6 = BombPotEnv(_cfg())
    _, i5 = e5.reset(42, 0)
    _, i6 = e6.reset(42, 0)
    assert int(i5.raw_obs["max_raise"]) == int(i6.raw_obs["max_raise"])
    assert int(i5.raw_obs["min_bet"]) == int(i6.raw_obs["min_bet"])


def test_hero_category_matches_describer_all_seats_and_streets():
    cfg = _cfg()
    mism = n = 0
    for seed in range(40):
        env = BombPotEnv(cfg)
        _, info = env.reset(seed, seed % 6)
        done = False
        for _ in range(30):
            raw = env._rs.observation_dict()
            holes = env.all_hole_cards()
            for bi, key in ((0, "board_a"), (1, "board_b")):
                board = [int(c) for c in raw[key]]
                if len(board) < 3:
                    continue
                for seat in range(cfg.num_seats):
                    mine = best_hand_category(holes[seat], board)
                    eng = int(env._rs.hero_category(seat, bi))
                    n += 1
                    if mine != eng:
                        mism += 1
            if done or info.actor is None:
                break
            _, _, done, info = env.step_hybrid(GATE_CHECK_CALL, 0)
    assert n > 1500, f"insufficient samples ({n})"
    assert mism == 0, f"{mism}/{n} category mismatches"


def test_batched_shapes_and_deal_parity():
    cfg = _cfg()
    N = 8
    b = BatchedBombPotEnv(N, cfg)
    seeds = np.arange(N, dtype=np.uint64)
    buttons = np.zeros(N, dtype=np.uint8)
    b.reset_batch(seeds, buttons)
    ah = b._be.all_hole_cards_batch()
    assert ah.shape == (N, 6, 6)
    d = b._be.observation_arrays()
    assert d["hero_hole"].shape == (N, 6)
    assert ((d["hero_hole"] < 52).sum(axis=1) == 6).all()
    env = BombPotEnv(cfg)
    env.reset(3, 0)
    assert np.array_equal(
        np.asarray(env.all_hole_cards(), dtype=np.uint8), ah[3]
    )


def test_batched_obs_bit_exact_vs_serial_over_steps():
    cfg = _cfg()
    N = 12
    batched = BatchedBombPotEnv(N, cfg)
    seeds = np.arange(500, 500 + N, dtype=np.uint64)
    buttons = (np.arange(N) % 6).astype(np.uint8)
    step = batched.reset_batch(seeds, buttons)
    serial = []
    for i in range(N):
        env = BombPotEnv(cfg)
        o, inf = env.reset(int(seeds[i]), int(buttons[i]))
        assert np.array_equal(step.obs[i], o), f"reset obs mismatch env {i}"
        serial.append((env, False))
    for t in range(6):
        step = batched.step_hybrid_batch(
            np.full(N, GATE_CHECK_CALL, dtype=np.int64),
            np.zeros(N, dtype=np.uint64),
        )
        for i, (env, done) in enumerate(serial):
            if done:
                continue
            o2, _, d, _ = env.step_hybrid(GATE_CHECK_CALL, 0)
            serial[i] = (env, d)
            assert d or np.array_equal(step.obs[i], o2), (
                f"step {t} env {i}: batched obs != serial obs"
            )


def test_rotate_opp_holes_width_six():
    from plo5bp.rollout import _rotate_opp_holes

    holes = np.arange(36, dtype=np.uint8).reshape(6, 6)
    out = _rotate_opp_holes(holes, actor=2)
    assert out.shape == (5, 6)
    assert np.array_equal(out[0], holes[3])
    # PLO5 shape unchanged
    holes5 = np.arange(30, dtype=np.uint8).reshape(6, 5)
    assert _rotate_opp_holes(holes5, actor=0).shape == (5, 5)


def test_opp_holes_multihot_absorbs_six_cards():
    import torch
    from plo5bp.network import opp_holes_multihot

    h6 = torch.full((2, 5, 6), 255, dtype=torch.uint8)
    h6[0, 0, :6] = torch.arange(6)
    out = opp_holes_multihot(h6)
    assert out.shape == (2, 260), "critic input dim unchanged by PLO6"
    assert out[0, :52].sum() == 6


def test_study_mode_rejected_for_plo6():
    env = BombPotEnv(_cfg())
    with pytest.raises(Exception):
        env.reset_study(
            button=0, hero_seat=0,
            hero_hole=[0, 4, 8, 12, 16],
            flop_a=[20, 24, 28], flop_b=[32, 36, 40],
        )
