"""NLH env + encoder: layout, blind visibility, scaling, determinism,
and the any-combo feature rules that differ from PLO."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plo5bp.config import GameConfig
from plo5bp.encoding_nlh import (
    _BLIND_FLAGS_OFF,
    _BOARD_OFF,
    _DRAW_OFF,
    _HOLE_CLASS_OFF,
    _HOLE_OFF,
    _OPP_OUTCOME_OFF,
    _SF_OFF,
    _SPR_OFF,
    _STREET_COMMIT_OFF,
    _STREET_OFF,
    OBS_DIM_NLH,
    _draw_flags_nlh,
    _sf_features_nlh,
)
from plo5bp.env import BombPotEnv


def _card(rank: int, suit: int) -> int:
    return rank * 4 + suit


@pytest.fixture()
def env() -> BombPotEnv:
    return BombPotEnv(GameConfig.nlh_default())


def test_preflop_obs_layout(env: BombPotEnv):
    obs, info = env.reset(seed=42, button=0)
    assert obs.shape == (OBS_DIM_NLH,)
    assert env.obs_dim == OBS_DIM_NLH
    # Street one-hot: preflop slot live.
    assert obs[_STREET_OFF + 0] == 1.0
    assert obs[_STREET_OFF + 1 : _STREET_OFF + 4].sum() == 0.0
    # Exactly 2 hole bits, no board bits.
    assert obs[_HOLE_OFF : _HOLE_OFF + 52].sum() == 2.0
    assert obs[_BOARD_OFF : _BOARD_OFF + 52].sum() == 0.0
    # UTG (seat 3, button 0) acts first; blinds visible in the
    # hero-rotated street-commit block: SB is 4 seats CW from UTG,
    # BB is 5 — 0.5bb and 1bb respectively.
    assert info.actor == 3
    sc = obs[_STREET_COMMIT_OFF : _STREET_COMMIT_OFF + 8]
    np.testing.assert_allclose(sc[4], 0.5, rtol=1e-6)
    np.testing.assert_allclose(sc[5], 1.0, rtol=1e-6)
    # UTG is neither blind.
    assert obs[_BLIND_FLAGS_OFF] == 0.0
    assert obs[_BLIND_FLAGS_OFF + 1] == 0.0
    # Opp-outcome zeros preflop.
    assert obs[_OPP_OUTCOME_OFF : _OPP_OUTCOME_OFF + 3].sum() == 0.0
    # SPR is log1p-scaled: UTG effective 99.5bb behind vs 4.5bb pot.
    expect = math.log1p(995_000 / 45_000)
    np.testing.assert_allclose(obs[_SPR_OFF], expect, rtol=1e-5)


def test_blind_flags_fire_for_blind_heroes(env: BombPotEnv):
    env.reset(seed=42, button=0)
    # Fold UTG..button so the SB (seat 1) becomes the actor.
    obs, _, done, info = env.step_hybrid(0)
    obs, _, done, info = env.step_hybrid(0)
    obs, _, done, info = env.step_hybrid(0)
    obs, _, done, info = env.step_hybrid(0)
    assert not done and info.actor == 1
    assert obs[_BLIND_FLAGS_OFF] == 1.0 and obs[_BLIND_FLAGS_OFF + 1] == 0.0
    # SB completes → BB's option.
    obs, _, done, info = env.step_hybrid(1)
    assert not done and info.actor == 2
    assert obs[_BLIND_FLAGS_OFF] == 0.0 and obs[_BLIND_FLAGS_OFF + 1] == 1.0


def test_flop_obs_and_opp_outcome(env: BombPotEnv):
    env.reset(seed=7, button=0)
    # Limp around; BB checks → flop.
    done = False
    obs, info = env._pack_obs()
    for _ in range(6):
        obs, _, done, info = env.step_hybrid(1)
        if done:
            break
    assert not done
    assert obs[_STREET_OFF + 1] == 1.0, "flop one-hot"
    assert obs[_BOARD_OFF : _BOARD_OFF + 52].sum() == 3.0
    frac = obs[_OPP_OUTCOME_OFF : _OPP_OUTCOME_OFF + 3]
    np.testing.assert_allclose(frac.sum(), 1.0, atol=1e-5)


def test_reset_is_deterministic(env: BombPotEnv):
    a, _ = env.reset(seed=123, button=2)
    b, _ = env.reset(seed=123, button=2)
    np.testing.assert_array_equal(a, b)


def test_hole_class_block(env: BombPotEnv):
    obs, _ = env.reset(seed=99, button=1)
    raw = dict(env._rs.observation_dict())
    hole = [int(c) for c in raw["hero_hole"]]
    r0, r1 = hole[0] // 4, hole[1] // 4
    hi, lo = max(r0, r1), min(r0, r1)
    assert obs[_HOLE_CLASS_OFF + 0] == (1.0 if r0 == r1 else 0.0)
    assert obs[_HOLE_CLASS_OFF + 1] == (1.0 if hole[0] % 4 == hole[1] % 4 else 0.0)
    np.testing.assert_allclose(obs[_HOLE_CLASS_OFF + 2], (hi - lo) / 12.0, rtol=1e-6)
    np.testing.assert_allclose(obs[_HOLE_CLASS_OFF + 3], hi / 12.0, rtol=1e-6)
    np.testing.assert_allclose(obs[_HOLE_CLASS_OFF + 4], lo / 12.0, rtol=1e-6)


def test_full_hand_terminates_with_zero_sum_rewards(env: BombPotEnv):
    env.reset(seed=5, button=3)
    total = None
    for _ in range(60):
        _, rewards, done, _ = env.step_hybrid(1)
        if done:
            total = rewards.sum()
            break
    assert total is not None, "check-down hand must terminate"
    assert abs(total) < 1e-6


# ---- any-combo feature rules ----


def test_draw_flags_one_hole_card_flush_draw():
    # Ah + three board hearts: an NLH flush draw (PLO's exactly-2 rule
    # would say no).
    hole = [_card(12, 2), _card(0, 0)]
    board = [_card(11, 2), _card(7, 2), _card(3, 2)]
    f, _s = _draw_flags_nlh(hole, board)
    assert f == 1.0


def test_draw_flags_board_only_four_flush_is_not_hero_draw():
    hole = [_card(12, 0), _card(0, 1)]
    board = [_card(11, 2), _card(7, 2), _card(3, 2), _card(5, 2)]
    f, _s = _draw_flags_nlh(hole, board)
    assert f == 0.0
    # ...but the texture is visible via flush_possible in the SF block.
    vc = np.zeros((13, 4), dtype=np.int32)
    for c in hole + board:
        vc[c // 4, c % 4] = 1
    sf = _sf_features_nlh(hole, board, vc)
    assert sf[22 + 2] == 1.0  # flush_possible for hearts


def test_sf_block_board_straight_is_made():
    # Broadway on board: hero "makes" it with zero hole cards → no
    # higher window → straight_nut_distance 0, and no outs reported for
    # the made window.
    hole = [_card(0, 0), _card(1, 1)]
    board = [_card(12, 0), _card(11, 1), _card(10, 2), _card(9, 3), _card(8, 0)]
    vc = np.zeros((13, 4), dtype=np.int32)
    for c in hole + board:
        vc[c // 4, c % 4] = 1
    sf = _sf_features_nlh(hole, board, vc)
    assert sf[1] == 0.0  # nut straight
    assert sf[2 + 9] == 0.0  # broadway window made → no outs


def test_sf_block_oesd_outs():
    # T9 on 8-7-2 rainbow: open-ender — windows 6..T and 7..J each
    # carry 4 outs (no 6s or Js visible).
    hole = [_card(8, 0), _card(7, 1)]
    board = [_card(6, 2), _card(5, 3), _card(0, 0)]
    vc = np.zeros((13, 4), dtype=np.int32)
    for c in hole + board:
        vc[c // 4, c % 4] = 1
    sf = _sf_features_nlh(hole, board, vc)
    assert sf[2 + 5] == 4.0  # window {4..8} completed by a 6
    assert sf[2 + 6] == 4.0  # window {5..9} completed by a J
    assert sf[1] == 0.0 and sf[0] == 0.0


def test_sf_block_one_card_flush_draw_outs():
    # One hole heart + three board hearts = a 4-flush DRAW (hero-involved).
    # Ah: every completing heart makes the nut flush → nut outs == outs.
    # Qh: only an ace-hit produces the nut → nut outs == 1.
    board = [_card(11, 2), _card(5, 2), _card(1, 2)]  # Kh 7h 3h
    vc = np.zeros((13, 4), dtype=np.int32)
    hole_a = [_card(12, 2), _card(0, 0)]  # Ah 2c
    for c in hole_a + board:
        vc[c // 4, c % 4] = 1
    sf = _sf_features_nlh(hole_a, board, vc)
    assert sf[26 + 2] == 9.0  # 13 hearts - 4 visible
    assert sf[30 + 2] == 9.0  # holding the A: every hit is the nut

    vc2 = np.zeros((13, 4), dtype=np.int32)
    hole_q = [_card(10, 2), _card(0, 0)]  # Qh 2c
    for c in hole_q + board:
        vc2[c // 4, c % 4] = 1
    sf2 = _sf_features_nlh(hole_q, board, vc2)
    assert sf2[26 + 2] == 9.0
    assert sf2[30 + 2] == 1.0  # one blocker rank above (unseen Ah)


def test_sf_block_made_flush_with_one_hole_card():
    # Board 4-flush + one hole heart = a MADE hero-involved flush.
    # 9h: unaccounted higher hearts are T, J, A (K, Q on board) → dist 3.
    board = [_card(11, 2), _card(10, 2), _card(5, 2), _card(1, 2)]  # KhQh7h3h
    vc = np.zeros((13, 4), dtype=np.int32)
    hole = [_card(7, 2), _card(0, 0)]  # 9h 2c
    for c in hole + board:
        vc[c // 4, c % 4] = 1
    sf = _sf_features_nlh(hole, board, vc)
    assert sf[0] == 3.0


def test_plo_env_unchanged_default():
    env = BombPotEnv(GameConfig())
    obs, info = env.reset(seed=1, button=0)
    from plo5bp.encoding import OBS_DIM

    assert obs.shape == (OBS_DIM,)
    assert env.obs_dim == OBS_DIM
