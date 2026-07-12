"""obs v2 tail (V5_DESIGN.md §3.2): per-board outcome decomposition,
unconditional blockers-to-nuts, effective price, log1p SPR.

Covers what the serial↔batched parity suite can't: the SEMANTICS of the
new dims — hand-built blocker cases, the exact probability identities
tying the 8-dim per-board block to the legacy 12-dim joint block (same
exhaustive k=2 universe, same fused Rust pass), the stack-capped price
vs the legacy uncapped pot-odds dim, and the deep-tier SPR saturation
fix. Plus scalar↔batched fuzz parity for the pure-Python blocker helper
and the layout invariants (pure tail append; 991-slice adapter).
"""

from __future__ import annotations

import numpy as np

from plo5bp.config import GameConfig
from plo5bp.encoding import (
    OBS_DIM,
    OBS_DIM_V2,
    _BLOCKER_A_OFF,
    _BLOCKER_B_OFF,
    _EFF_PRICE_OFF,
    _OPP_OUTCOME_OFF,
    _PER_BOARD_OUTCOME_OFF,
    _POT_ODDS_OFF,
    _SCALARS_OFF,
    _SPR_LOG_OFF,
    _SPR_OFF,
    _STACKS_OFF,
    _blocker_features,
    _blocker_features_batch,
    downgrade_obs_to_v2,
)
from plo5bp.env import BombPotEnv


# ---- layout invariants ------------------------------------------------------

def test_obs_v2_layout_is_pure_tail_append() -> None:
    # OBS_DIM moved to 1171 with the v7 batch-2 tail (2026-07-12); the obs-v2
    # blocks stay at their original offsets, so downgrade_obs_to_v2 is still an
    # exact tail slice and old checkpoints keep serving.
    assert OBS_DIM == 1171
    assert OBS_DIM_V2 == 991
    assert _PER_BOARD_OUTCOME_OFF == OBS_DIM_V2  # first v2-appended dim
    assert _SPR_LOG_OFF + 8 == 1020              # last obs-v2 block ends at 1020
    v = np.arange(OBS_DIM, dtype=np.float32)
    sliced = downgrade_obs_to_v2(v)
    assert sliced.shape == (OBS_DIM_V2,)
    assert np.array_equal(sliced, v[:OBS_DIM_V2])


# ---- blockers: hand-built cases --------------------------------------------
# Card index = rank * 4 + suit (rank 12 = Ace .. 0 = deuce).

def test_blockers_flush_and_straight_case() -> None:
    # Board: Ks 9s 4s 2h 6d (suit 0 = "s") → 3-flush in suit 0, board
    # ranks {11, 7, 2, 0, 4}; nut straight window = {0..4} (missing
    # ranks {1, 3}); unpaired.
    board = [44, 28, 8, 1, 18]
    # Hero: As 3h 3d 5h Jh → holds the top missing flush card (As), one
    # of the top-3 missing flush cards, and 3 straight-completing cards
    # (two rank-1, one rank-3).
    hole = [48, 5, 6, 13, 37]
    out = _blocker_features(hole, board)
    assert out[0] == 1.0
    assert abs(out[1] - 1.0 / 3.0) < 1e-6
    assert out[2] == 0.75
    assert out[3] == 0.0


def test_blockers_paired_board_case() -> None:
    # Board: Ks Kh 4s 2h 6d → paired kings, no 3-flush; nut straight
    # window {0..4}, missing {1, 3}.
    board = [44, 45, 8, 1, 18]
    # Hero: Kd 3h Qs Js Ts → one king (trips/boat blocker), one rank-1.
    hole = [46, 5, 40, 36, 33]
    out = _blocker_features(hole, board)
    assert out[0] == 0.0 and out[1] == 0.0
    assert out[2] == 0.25
    assert out[3] == 0.5


def test_blockers_short_board_zero() -> None:
    assert np.array_equal(_blocker_features([0, 5, 10, 15, 20], []), np.zeros(4))
    assert np.array_equal(_blocker_features([0, 5, 10, 15, 20], [44, 28]), np.zeros(4))


def test_blockers_scalar_batch_parity_fuzz() -> None:
    rng = np.random.default_rng(7)
    n = 300
    hole = np.full((n, 5), 255, dtype=np.uint8)
    board = np.full((n, 5), 255, dtype=np.uint8)
    for i in range(n):
        deck = rng.permutation(52)
        hole[i] = deck[:5]
        board_len = int(rng.choice([0, 3, 4, 5], p=[0.1, 0.3, 0.2, 0.4]))
        board[i, :board_len] = deck[5 : 5 + board_len]
    batch = _blocker_features_batch(hole, hole < 52, board, board < 52)
    for i in range(n):
        h = [int(c) for c in hole[i] if c < 52]
        b = [int(c) for c in board[i] if c < 52]
        scalar = _blocker_features(h, b)
        assert np.array_equal(batch[i], scalar), (i, h, b, batch[i], scalar)


# ---- per-board outcome: exact identities vs the 12-dim joint block ---------

def test_per_board_outcome_identities() -> None:
    env = BombPotEnv(GameConfig())
    rng = np.random.default_rng(11)
    checked = 0
    for seed in range(20):
        obs, info = env.reset(seed, seed % 6)
        while not info.terminal:
            pb = obs[_PER_BOARD_OUTCOME_OFF : _PER_BOARD_OUTCOME_OFF + 8]
            joint = obs[_OPP_OUTCOME_OFF : _OPP_OUTCOME_OFF + 4]  # k=2 row
            scoop_opp, quarter_opp, scoop_hero, quarter_hero = (
                float(joint[0]), float(joint[1]), float(joint[2]), float(joint[3])
            )
            a_a, t_a, b_a, a_b, t_b, b_b, split, dtie = (float(x) for x in pb)
            # Per-board fractions partition the exhaustive combo universe.
            assert abs(a_a + t_a + b_a - 1.0) < 1e-5
            assert abs(a_b + t_b + b_b - 1.0) < 1e-5
            # Exact identities (same universe, same pass):
            #   ahead_a + ahead_b = 2·P(hero scoops) + P(hero quarters) + split
            #   behind_a + behind_b = 2·P(opp scoops) + P(opp quarters) + split
            #   tie_a + tie_b = 2·P(tie both) + P(hero quarters) + P(opp quarters)
            assert abs((a_a + a_b) - (2 * scoop_hero + quarter_hero + split)) < 1e-4
            assert abs((b_a + b_b) - (2 * scoop_opp + quarter_opp + split)) < 1e-4
            assert abs((t_a + t_b) - (2 * dtie + quarter_hero + quarter_opp)) < 1e-4
            checked += 1
            legal = np.flatnonzero(info.legal_mask)
            obs, _, done, info = env.step(int(rng.choice(legal)))
            if done:
                break
    assert checked > 30  # the loop actually exercised decision nodes


def test_per_board_outcome_deterministic() -> None:
    env = BombPotEnv(GameConfig())
    obs1, _ = env.reset(123, 2)
    obs2, _ = env._pack_obs()
    a = obs1[_PER_BOARD_OUTCOME_OFF : _PER_BOARD_OUTCOME_OFF + 8]
    b = obs2[_PER_BOARD_OUTCOME_OFF : _PER_BOARD_OUTCOME_OFF + 8]
    assert np.array_equal(a, b)
    assert a.sum() > 0.0  # populated at the flop


# ---- effective price vs uncapped pot odds -----------------------------------

def test_eff_price_caps_at_hero_stack() -> None:
    # Heads-up the engine caps bets at the OPPONENT's reachable total (no
    # uncallable chips), so to_call can only exceed hero's stack in a
    # MULTIWAY pot: two deep seats (20bb) + one short (6bb). 3bb antes →
    # 9bb flop pot; the short seat has 3bb behind. Button = 2 → seat 0
    # acts first (pot-bets 9bb, legal: seat 1 covers it), seat 1 calls,
    # and the short seat 2 then faces to_call (9bb) > its 3bb stack.
    cfg = GameConfig(
        num_seats=3, starting_stacks=[200_000, 200_000, 60_000]
    )
    env = BombPotEnv(cfg)
    obs, info = env.reset(5, 2)
    assert info.actor == 0
    from plo5bp.actions import GATE_CHECK_CALL, GATE_RAISE

    obs, _, done, info = env.step_hybrid(GATE_RAISE, 90_000)
    assert not done and info.actor == 1
    obs, _, done, info = env.step_hybrid(GATE_CHECK_CALL)
    assert not done and info.actor == 2
    # Pot now 27bb (9 antes + 9 bet + 9 call); hero faces 9bb with 3bb.
    # Legacy uncapped pot odds: 9 / (27 + 9).
    assert abs(obs[_POT_ODDS_OFF] - 9.0 / 36.0) < 1e-6
    # Effective: to_call capped at the 3bb stack → 3 / (27 + 3).
    assert abs(obs[_EFF_PRICE_OFF + 0] - 3.0 / 30.0) < 1e-6
    assert obs[_EFF_PRICE_OFF + 1] == 1.0          # calling = all-in
    # Commitment: 3bb ante in, 3bb behind → 0.5.
    assert abs(obs[_EFF_PRICE_OFF + 2] - 0.5) < 1e-6
    assert abs(obs[_EFF_PRICE_OFF + 3] - np.log1p(3.0)) < 1e-6
    assert abs(obs[_EFF_PRICE_OFF + 4] - np.log1p(27.0)) < 1e-6


def test_eff_price_zero_when_no_bet() -> None:
    env = BombPotEnv(GameConfig())
    obs, _ = env.reset(1, 0)
    assert obs[_EFF_PRICE_OFF + 0] == 0.0
    assert obs[_EFF_PRICE_OFF + 1] == 0.0
    assert obs[_EFF_PRICE_OFF + 3] == 0.0
    assert obs[_EFF_PRICE_OFF + 4] > 0.0  # pot (antes) is never zero


# ---- log1p SPR: the deep-tier saturation fix --------------------------------

def test_spr_log_unclipped_at_deep_stacks() -> None:
    # 200bb 6-max: flop pot 18bb, eff ≈ 197bb → true SPR ≈ 10.9. The
    # legacy dim clips to 4.0 (constant across the whole deep tier); the
    # log1p dim must carry the real value.
    env = BombPotEnv(GameConfig(num_seats=6, starting_stack=2_000_000))
    obs, _ = env.reset(9, 0)
    assert obs[_SPR_OFF] == 4.0
    eff_bb = float(obs[_STACKS_OFF])
    pot_bb = float(obs[_SCALARS_OFF])
    expected = np.log1p(eff_bb / pot_bb)
    assert obs[_SPR_LOG_OFF] > np.log1p(4.0)
    assert abs(obs[_SPR_LOG_OFF] - expected) < 1e-4
