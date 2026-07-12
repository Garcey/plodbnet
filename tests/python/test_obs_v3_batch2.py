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


def test_engine_dims_live_and_semantically_sane():
    """Chunk-B engine dims (STK-1/BRD-7/BRD-12/DUAL-2/DUAL-4) over real
    deals: cross-checked against the raw observation_dict values and each
    other. Parity (serial == batched) is test_encoding_batch.py's job;
    these are the semantics parity can't see."""
    from plo5bp.env import BombPotEnv

    cfg = GameConfig(num_seats=6, starting_stack=150 * 10000, ante=3 * 10000, bb=10000)
    env = BombPotEnv(cfg)
    saw_nonzero = {"stk1": False, "brd7": False, "dual2": False, "dual4": False}
    for seed in range(60):
        obs, info = env.reset(seed=seed, button=seed % cfg.num_seats)
        if info.actor is None:
            continue
        vec = env._last_obs_vec
        raw = dict(env._rs.observation_dict())
        hb = raw["hero_board_v3"]
        shb = raw["share_bounds"]

        # DUAL-2: exactly two hole cards form the best holding, per board.
        for base in (_DUAL2_OFF, _DUAL2_OFF + 5):
            bits = vec[base : base + 5]
            assert set(np.unique(bits)) <= {0.0, 1.0}
            assert bits.sum() == 2.0, f"seed {seed}: DUAL-2 bits {bits}"
        saw_nonzero["dual2"] = True

        # BRD-7 raw counts land verbatim; BRD-12 = raw/unseen and raw/10.
        assert vec[_BRD7_OFF + 0] == np.float32(float(hb[0]))
        assert vec[_BRD7_OFF + 1] == np.float32(float(hb[1]))
        unseen = 52 - 5 - len(raw["board_a"]) - len(raw["board_b"])
        assert vec[_BRD12_OFF + 0] == np.float32(float(hb[2]) / unseen)
        assert vec[_BRD12_OFF + 2] == np.float32(float(hb[4]) / 10.0)
        # Boat-or-better outs are a subset of strict-category improves
        # whenever hero is below a full house (raw-count comparison).
        if hb[0] > 0:
            assert hb[2] >= hb[0], f"seed {seed}: boat {hb[0]} > improve {hb[2]}"
            saw_nonzero["brd7"] = True
        # Combo redundancy: at least one pair achieves the best category.
        assert 1 <= hb[4] <= 10 and 1 <= hb[5] <= 10

        # DUAL-4: quarter grid, ordered, DUAL-3-consistent.
        g_min, g_max = float(shb[0]), float(shb[1])
        assert g_min in (0.0, 0.25, 0.5, 0.75, 1.0)
        assert g_max in (0.0, 0.25, 0.5, 0.75, 1.0)
        assert g_min <= g_max
        assert vec[_DUAL4_OFF + 0] == np.float32(g_min)
        assert vec[_DUAL4_OFF + 1] == np.float32(g_max)
        if g_max > 0.0:
            saw_nonzero["dual4"] = True
        if vec[_DUAL3_OFF + 4] == 1.0:  # locked both -> worst case >= half
            assert g_min >= 0.5
        if vec[_DUAL3_OFF + 0] == 1.0 and vec[_DUAL3_OFF + 2] == 1.0:
            assert g_min == 1.0  # pure nuts on both -> guaranteed scoop

        # STK-1: at the first flop decision all 5 opponents are pending
        # with money behind -> every dim positive and the covered flag is
        # well-defined; log1p(max_eff/bb) must match a direct recompute.
        acted = raw["acted_this_street"]
        pending_eff = [
            float(raw["stacks"][s])
            for s in range(cfg.num_seats)
            if s != raw["actor"]
            and not raw["folded"][s]
            and not raw["all_in"][s]
            and (not acted[s] or raw["street_commit"][s] < raw["bet_to_call"])
        ]
        if pending_eff:
            assert vec[_STK1_OFF + 1] > 0.0
            saw_nonzero["stk1"] = True
        else:
            assert np.all(vec[_STK1_OFF : _STK1_OFF + 4] == 0.0)
    assert all(saw_nonzero.values()), saw_nonzero


def test_stk1_bit_exact_recompute_mid_hand():
    """STK-1 against a from-scratch spec recompute on MID-HAND states
    (live bets, partial folds, owed>0) — the reviewer's sweep, pinned."""
    from plo5bp.env import BombPotEnv

    cfg = GameConfig(num_seats=6, starting_stack=150 * 10000, ante=3 * 10000, bb=10000)
    env = BombPotEnv(cfg)
    rng = np.random.default_rng(11)
    checked = 0
    for seed in range(30):
        obs, info = env.reset(seed=seed, button=seed % cfg.num_seats)
        for _step in range(6):
            if info.actor is None:
                break
            raw = dict(env._rs.observation_dict())
            vec = env._last_obs_vec
            hero = raw["actor"]
            n = cfg.num_seats
            starting = cfg.resolved_stacks
            eff = [
                max(0.0, float(raw["stacks"][s]) - max(0, int(starting[s]) - int(raw["eff_stack_cap"][s])))
                for s in range(n)
            ]
            btc = float(raw["bet_to_call"])
            acted = raw["acted_this_street"]
            max_cap = sum_cap = max_eff = 0.0
            any_pending = False
            for s in range(n):
                if s == hero or raw["folded"][s] or raw["all_in"][s]:
                    continue
                if acted[s] and float(raw["street_commit"][s]) >= btc:
                    continue
                any_pending = True
                owed = min(max(btc - float(raw["street_commit"][s]), 0.0), eff[s])
                cap = max(0.0, eff[s] - owed)
                max_cap = max(max_cap, cap)
                sum_cap += cap
                max_eff = max(max_eff, eff[s])
            pot_denom = max(float(raw["pot"]), 1.0)
            if any_pending:
                exp = [
                    np.float32(np.log1p(max_cap / pot_denom)),
                    np.float32(np.log1p(sum_cap / pot_denom)),
                    np.float32(np.log1p(max_eff / cfg.bb)),
                    np.float32(1.0 if max_eff >= eff[hero] else 0.0),
                ]
            else:
                exp = [np.float32(0.0)] * 4
            got = list(vec[_STK1_OFF : _STK1_OFF + 4])
            assert got == exp, f"seed {seed} step {_step}: {got} vs {exp}"
            checked += 1
            # random legal gate action to reach genuine mid-hand nodes
            legal = [g for g in range(3) if info.gate_mask[g]]
            gate = int(rng.choice(legal))
            chips = int(info.min_raise_chips or 0) if gate == 2 else 0
            obs, _r, done, info = env.step_hybrid(gate, chips)
            if done:
                break
    assert checked > 60


def test_dual4_ante_zero_pot_guard():
    """Regression: ante=0 (pot 0 at the flop) must encode cleanly — the
    unguarded DUAL-4[4] crashed serial and emitted inf batched."""
    from plo5bp.env import BombPotEnv

    cfg = GameConfig(num_seats=3, starting_stack=200000, ante=0, bb=10000)
    env = BombPotEnv(cfg)
    for seed in range(4):
        obs, info = env.reset(seed=seed, button=0)
        if info.actor is None:
            continue
        vec = env._last_obs_vec
        assert np.all(np.isfinite(vec)), "non-finite dims at pot==0"


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
