"""Correctness + parity for the v7 obs batch-2 tail (stack + board + dual).

Serial-vs-batched BIT-EXACT parity for these dims is already swept by
test_encoding_batch.py's `_drive_encoder_parity`. This module adds
CONCRETE-VALUE checks (parity alone can't catch a spec-wrong value that
both encoders share) plus reserved-[ENGINE]-column-zero and layout
invariants. Dims: V7_OBS_CANDIDATES.md; plan: V7_OBS_IMPL_PLAN.md.
"""

import numpy as np

from plo5bp import encoding as E
from plo5bp.config import GameConfig
from plo5bp.encoding import (
    OBS_DIM,
    _BRD1_OFF,
    _BRD2_OFF,
    _BRD7_OFF,
    _BRD11_OFF,
    _BRD12_OFF,
    _BRD13_OFF,
    _DUAL1_OFF,
    _DUAL2_OFF,
    _DUAL3_OFF,
    _DUAL4_OFF,
    _DUAL5_OFF,
    _STK1_OFF,
    _STK4_OFF,
    _STK6_OFF,
    _STK10_OFF,
    encode_observation,
    _encode_board_v3,
    _encode_board_v3_batch,
    _encode_dual_v3,
    _encode_dual_v3_batch,
    _encode_stack_v3,
    _encode_stack_v3_batch,
)


# ---- layout invariants -----------------------------------------------------

def test_tail_ends_at_obs_dim():
    assert OBS_DIM == 1171
    assert _STK1_OFF == 1020
    assert _DUAL5_OFF + 9 == OBS_DIM


def test_reserved_engine_columns_zero_on_real_encode():
    """STK-1/BRD-7/BRD-12/DUAL-2/DUAL-4 are Chunk-B [ENGINE] dims — they
    MUST read exactly 0.0 in a real encoded obs until their plumbing lands."""
    from plo5bp.env import BombPotEnv

    cfg = GameConfig(num_seats=6, starting_stack=150 * 10000, ante=3 * 10000, bb=10000)
    env = BombPotEnv(cfg)
    hit = 0
    for seed in range(40):
        obs, info = env.reset(seed=seed, button=seed % cfg.num_seats)
        if info.actor is None:
            continue
        # advance a few random-ish legal steps to reach varied streets
        vec = env._last_obs_vec
        for off, n in (
            (_STK1_OFF, 4), (_BRD7_OFF, 2), (_BRD12_OFF, 4),
            (_DUAL2_OFF, 10), (_DUAL4_OFF, 5),
        ):
            assert np.all(vec[off : off + n] == 0.0), f"reserved @{off} nonzero"
        hit += 1
    assert hit > 0


# ---- STACK -----------------------------------------------------------------

def test_stk_v3_serial_batched_parity():
    cfg = GameConfig(num_seats=3, starting_stack=200000, ante=30000, bb=10000)
    S = 3
    hero = 1
    folded = [False, True, False]
    all_in = [False, False, False]
    eff = [80000.0, 50000.0, 120000.0]
    total_commit = [45000, 30000, 45000]
    street_commit = [15000, 0, 15000]
    pot = 120000.0
    btc = 15000.0
    min_bet = 45000.0
    max_bet = 150000.0
    to_call = max(btc - street_commit[hero], 0.0)
    hero_stack = eff[hero]
    eff_to_call = min(to_call, hero_stack)
    inv_bb = 1.0 / cfg.bb

    out_s = np.zeros(OBS_DIM, dtype=np.float32)
    _encode_stack_v3(
        out_s, config=cfg, hero=hero, num_seats=S,
        folded=folded, all_in=all_in, eff_per_seat=eff,
        total_commit=total_commit, street_commit=street_commit,
        pot=pot, btc=btc, min_bet=min_bet, max_bet=max_bet,
        to_call=to_call, eff_to_call=eff_to_call, hero_stack=hero_stack,
        inv_bb=inv_bb, street_idx=2,
    )
    out_b = np.zeros((1, OBS_DIM), dtype=np.float32)
    _encode_stack_v3_batch(
        out_b, config=cfg, num_seats=S, live_mask=np.array([True]),
        hero_idx=np.array([hero], dtype=np.int64),
        folded=np.array([folded], dtype=bool),
        all_in=np.array([all_in], dtype=bool),
        effective=np.array([eff], dtype=np.float64),
        total_commit=np.array([total_commit], dtype=np.float64),
        street_commit=np.array([street_commit], dtype=np.float64),
        pot=np.array([pot], dtype=np.float64),
        bet_to_call=np.array([btc], dtype=np.float64),
        min_bet=np.array([min_bet], dtype=np.float64),
        max_bet=np.array([max_bet], dtype=np.float64),
        to_call=np.array([to_call], dtype=np.float64),
        inv_bb=inv_bb, street=np.array([2], dtype=np.int64),
    )
    np.testing.assert_array_equal(out_s[1020:1061], out_b[0, 1020:1061])


def test_stk10_ante_and_bloat_values():
    cfg = GameConfig(num_seats=3, starting_stack=200000, ante=30000, bb=10000)
    out = np.zeros(OBS_DIM, dtype=np.float32)
    _encode_stack_v3(
        out, config=cfg, hero=0, num_seats=3,
        folded=[False, False, False], all_in=[False, False, False],
        eff_per_seat=[90000.0, 90000.0, 90000.0],
        total_commit=[30000, 30000, 30000], street_commit=[0, 0, 0],
        pot=120000.0, btc=0.0, min_bet=120000.0, max_bet=180000.0,
        to_call=0.0, eff_to_call=0.0, hero_stack=90000.0,
        inv_bb=1.0 / cfg.bb, street_idx=1,
    )
    assert out[_STK10_OFF + 0] == np.float32(3.0)
    assert out[_STK10_OFF + 1] == np.float32(np.log1p(30000.0 / 90000.0))
    assert np.all(out[_STK1_OFF : _STK1_OFF + 4] == 0.0)


def test_stk_allin_hero_geometric_zero():
    cfg = GameConfig(num_seats=2, starting_stack=200000, ante=30000, bb=10000)
    out = np.zeros(OBS_DIM, dtype=np.float32)
    _encode_stack_v3(
        out, config=cfg, hero=0, num_seats=2,
        folded=[False, False], all_in=[True, False],
        eff_per_seat=[0.0, 100000.0],
        total_commit=[200000, 45000], street_commit=[0, 0],
        pot=245000.0, btc=0.0, min_bet=0.0, max_bet=0.0,
        to_call=0.0, eff_to_call=0.0, hero_stack=0.0,
        inv_bb=1.0 / cfg.bb, street_idx=1,
    )
    assert out[_STK6_OFF + 0] == np.float32(0.0)
    assert out[_STK6_OFF + 1] == np.float32(0.0)
    assert out[_STK4_OFF + 0] == np.float32(1.0)


# ---- BOARD -----------------------------------------------------------------

def _vc(cards):
    v = np.zeros((13, 4), dtype=np.int32)
    for c in cards:
        v[c // 4, c % 4] = 1
    return v


def test_board_concrete_values():
    # card int c: rank = c//4, suit = c%4. Ks=44,Kh=45,9c=36 (rank9=idx7).
    hole = [0, 5, 8, 13, 20]
    ba = [44, 45, 30]                 # Ks Kh 9c — paired, no straight
    bb = [2, 3, 4]
    out = np.zeros(OBS_DIM, dtype=np.float32)
    _encode_board_v3(out, hole_list=hole, board_a_list=ba, board_b_list=bb,
                     visible_count=_vc(hole + ba + bb), street_idx=1)
    # BRD-1 ladder A: ranks [11,11,7] -> (12/13,12/13,8/13,0,0)
    assert abs(out[_BRD1_OFF + 0] - 12.0 / 13.0) < 1e-6
    assert abs(out[_BRD1_OFF + 1] - 12.0 / 13.0) < 1e-6
    assert abs(out[_BRD1_OFF + 2] - 8.0 / 13.0) < 1e-6
    assert out[_BRD1_OFF + 3] == 0.0
    # BRD-13 A: not sf, paired -> ceiling class 7/8; sf flag 0.
    assert out[_BRD13_OFF + 0] == 0.0
    assert abs(out[_BRD13_OFF + 1] - 7.0 / 8.0) < 1e-6
    # BRD-11: flop -> no turn/river dealt.
    assert np.all(out[_BRD11_OFF : _BRD11_OFF + 20] == 0.0)


def test_board_serial_batched_parity_turn():
    def _mk(cards):
        a = np.full(5, 255, dtype=np.uint8)
        for i, c in enumerate(cards):
            a[i] = c
        return a

    holes = [[0, 5, 8, 13, 20], [1, 6, 9, 14, 40]]
    b_a = [[44, 45, 30], [44, 45, 30, 10]]
    b_b = [[2, 3, 4], [2, 3, 4, 50]]
    streets = [1, 2]
    out_b = np.zeros((2, OBS_DIM), dtype=np.float32)
    _encode_board_v3_batch(
        out_b, live_mask=np.array([True, True]),
        hole=np.stack([_mk(h) for h in holes]),
        board_a=np.stack([_mk(b) for b in b_a]),
        board_b=np.stack([_mk(b) for b in b_b]),
        street=np.array(streets, dtype=np.int64),
    )
    for i in range(2):
        out_s = np.zeros(OBS_DIM, dtype=np.float32)
        _encode_board_v3(out_s, hole_list=holes[i], board_a_list=b_a[i],
                         board_b_list=b_b[i],
                         visible_count=_vc(holes[i] + b_a[i] + b_b[i]),
                         street_idx=streets[i])
        assert np.array_equal(
            out_s[_BRD1_OFF:_DUAL1_OFF], out_b[i, _BRD1_OFF:_DUAL1_OFF]
        )


# ---- DUAL ------------------------------------------------------------------

_cfg = GameConfig()


def test_dual1_split_price():
    o1 = np.zeros(OBS_DIM, dtype=np.float32)
    _encode_dual_v3(
        o1, hole_list=[], board_a_list=[], board_b_list=[], visible_count=None,
        per_board_outcome=None, pot=100.0, to_call=50.0, eff_to_call=50.0,
        hero_stack=1e9, config=_cfg,
    )
    assert abs(o1[_DUAL1_OFF + 0] - 0.5) < 1e-6
    assert abs(o1[_DUAL1_OFF + 1] - (50.0 / 75.0)) < 1e-6
    o1b = np.zeros(OBS_DIM, dtype=np.float32)
    _encode_dual_v3(
        o1b, hole_list=[], board_a_list=[], board_b_list=[], visible_count=None,
        per_board_outcome=None, pot=100.0, to_call=0.0, eff_to_call=0.0,
        hero_stack=1e9, config=_cfg,
    )
    assert o1b[_DUAL1_OFF + 0] == 0.0 and o1b[_DUAL1_OFF + 1] == 0.0


def test_dual3_lock_flags():
    o3 = np.zeros(OBS_DIM, dtype=np.float32)
    pbo = [1.0, 0.0, 0.0, 0.3, 0.0, 0.7, 1.0, 0.0]  # A locked, B behind>0
    _encode_dual_v3(
        o3, hole_list=[], board_a_list=[0, 4, 8], board_b_list=[1, 5, 9],
        visible_count=None, per_board_outcome=pbo, pot=0.0, to_call=0.0,
        eff_to_call=0.0, hero_stack=0.0, config=_cfg,
    )
    assert o3[_DUAL3_OFF + 0] == 1.0 and o3[_DUAL3_OFF + 1] == 1.0
    assert o3[_DUAL3_OFF + 2] == 0.0 and o3[_DUAL3_OFF + 3] == 0.0
    assert o3[_DUAL3_OFF + 4] == 0.0 and o3[_DUAL3_OFF + 5] == 1.0
    o3b = np.zeros(OBS_DIM, dtype=np.float32)
    _encode_dual_v3(
        o3b, hole_list=[], board_a_list=[0, 4, 8], board_b_list=[1, 5, 9],
        visible_count=None, per_board_outcome=[0.0] * 8, pot=0.0, to_call=0.0,
        eff_to_call=0.0, hero_stack=0.0, config=_cfg,
    )
    assert not o3b[_DUAL3_OFF : _DUAL3_OFF + 6].any()


def test_dual5_coverage_and_batch_parity():
    o5 = np.zeros(OBS_DIM, dtype=np.float32)
    _encode_dual_v3(
        o5, hole_list=[], board_a_list=[0, 4, 2], board_b_list=[8, 12, 3],
        visible_count=None, per_board_outcome=None, pot=0.0, to_call=0.0,
        eff_to_call=0.0, hero_stack=0.0, config=_cfg,
    )
    assert o5[_DUAL5_OFF + 0] == 1.0        # suit0 >=2 on both boards
    assert o5[_DUAL5_OFF + 4 + 0] == 0.0    # not >=3 on both
    assert 0.0 <= o5[_DUAL5_OFF + 8] <= 1.0

    N = 2
    ob = np.zeros((N, OBS_DIM), dtype=np.float32)
    board_a = np.full((N, 5), 255, np.uint8); board_a[:, :3] = np.array([0, 4, 2], np.uint8)
    board_b = np.full((N, 5), 255, np.uint8); board_b[:, :3] = np.array([8, 12, 3], np.uint8)
    _encode_dual_v3_batch(
        ob, live_mask=np.array([True, True]), hero_idx=np.zeros(N, dtype=np.int64),
        hole=np.full((N, 5), 255, np.uint8), board_a=board_a, board_b=board_b,
        per_board_outcome=np.array([[1.0, 0.0, 0.0, 0.3, 0.0, 0.7, 1.0, 0.0]] * N, np.float64),
        pot=np.array([100.0, 100.0]), to_call=np.array([50.0, 50.0]),
        effective=np.full((N, _cfg.num_seats), 1e9, np.float64), config=_cfg,
    )
    assert abs(ob[0, _DUAL1_OFF + 0] - 0.5) < 1e-6
    assert ob[0, _DUAL3_OFF + 0] == 1.0 and ob[0, _DUAL3_OFF + 5] == 1.0
    assert ob[0, _DUAL5_OFF + 0] == 1.0
    assert abs(ob[0, _DUAL5_OFF + 8] - o5[_DUAL5_OFF + 8]) < 1e-7
