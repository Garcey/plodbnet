"""Observation encoding shape + feature sanity tests."""

from __future__ import annotations

import numpy as np

from plo5bp.config import GameConfig
from plo5bp.encoding import (
    OBS_DIM,
    _BOARD_STRUCT_A_OFF,
    _BOARD_STRUCT_B_OFF,
    _CAT_A_OFF,
    _CAT_B_OFF,
    _DRAW_A_OFF,
    _FLUSH_DRAW_BOTH_OFF,
    _FLUSH_DRAW_OUTS_A_OFF,
    _FLUSH_MADE_BOTH_OFF,
    _FLUSH_MIXED_OFF,
    _FLUSH_NUT_DIST_A_OFF,
    _FLUSH_POSSIBLE_A_OFF,
    _HERO_RANK_HIST_OFF,
    _HISTORY_CHIPS_OFF_REL,
    _HISTORY_GATE_OFF_REL,
    _HISTORY_OFF,
    _HISTORY_SLOT_DIM,
    _HISTORY_STREET_OFF_REL,
    _NUM_CATEGORIES,
    _NUT_FLUSH_DRAW_OUTS_A_OFF,
    _HERO_BTN_DIST_OFF,
    _LAST_AGGRESSOR_OFF,
    _BET_PCT_POT_OFF,
    _OPP_OUTCOME_DIM,
    _OPP_OUTCOME_OFF,
    _PAIR_COUNT_A_OFF,
    _PAIR_COUNT_B_OFF,
    _POT_ODDS_OFF,
    _SEAT_EXISTS_OFF,
    _SHARED_RANKS_OFF,
    _STREET_COMMIT_OFF,
    _TOTAL_COMMIT_OFF,
    _SF_DRAW_OUTS_A_OFF,
    _SPR_OFF,
    _STACKS_OFF,
    _STRAIGHT_DRAW_BOTH_OFF,
    _STRAIGHT_MADE_BOTH_OFF,
    _STRAIGHT_MIXED_OFF,
    _STRAIGHT_NUT_DIST_A_OFF,
    _STRAIGHT_OUTS_A_OFF,
    _STRAIGHT_POSSIBLE_A_OFF,
    encode_observation,
)
from plo5bp.env import BombPotEnv


def test_obs_shape_on_various_states() -> None:
    env = BombPotEnv(GameConfig())
    rng = np.random.default_rng(1)

    obs, info = env.reset(42, 0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    # Hero hole multi-hot has exactly 5 ones.
    assert int(obs[0:52].sum()) == 5
    # Board A and B multi-hot each have 3 ones on the flop.
    assert int(obs[52:104].sum()) == 3
    assert int(obs[104:156].sum()) == 3

    # Mid-street: apply a bet.
    env.step(int(np.flatnonzero(info.legal_mask)[1]))  # CheckCall
    obs2, info2 = env._pack_obs()
    assert obs2.shape == (OBS_DIM,)

    # Drive through the hand to reach the river, then check board sizes.
    while not info2.terminal:
        legal = np.flatnonzero(info2.legal_mask)
        _, _, done, info2 = env.step(int(rng.choice(legal)))
        if done:
            break


def test_pot_odds_zero_preaction() -> None:
    env = BombPotEnv(GameConfig())
    obs, _ = env.reset(1, 0)
    # No bet to face at flop-start → pot_odds == 0.
    assert obs[_POT_ODDS_OFF] == 0.0


def test_pot_odds_positive_after_bet() -> None:
    env = BombPotEnv(GameConfig())
    _, info = env.reset(3, 0)
    # Actor faces no bet; apply a BetPct50. Next actor faces to_call > 0.
    from plo5bp.actions import BET_PCT_50
    assert info.legal_mask[BET_PCT_50]
    obs2, _, _, info2 = env.step(BET_PCT_50)
    # Next actor is facing a bet.
    assert obs2[_POT_ODDS_OFF] > 0.0
    assert obs2[_POT_ODDS_OFF] <= 1.0


def test_bet_pct_pot_zero_preaction() -> None:
    env = BombPotEnv(GameConfig())
    obs, _ = env.reset(1, 0)
    assert obs[_BET_PCT_POT_OFF] == 0.0


def test_bet_pct_pot_half_pot_bet() -> None:
    env = BombPotEnv(GameConfig())
    _, info = env.reset(3, 0)
    from plo5bp.actions import BET_PCT_50
    assert info.legal_mask[BET_PCT_50]
    obs2, _, _, _ = env.step(BET_PCT_50)
    # Half-pot bet → bet/pot ≈ 0.5.
    val = float(obs2[_BET_PCT_POT_OFF])
    assert 0.4 <= val <= 0.6, f"expected ~0.5, got {val}"


def test_bet_pct_pot_clipped_below_four() -> None:
    env = BombPotEnv(GameConfig())
    obs, _ = env.reset(5, 0)
    # Whatever the state, value must be in [0, 4].
    assert 0.0 <= float(obs[_BET_PCT_POT_OFF]) <= 4.0


def test_spr_decreases_as_actor_commits() -> None:
    env = BombPotEnv(GameConfig())
    obs, info = env.reset(7, 0)
    spr_before = obs[_SPR_OFF]  # hero's own SPR (rotation offset 0)
    from plo5bp.actions import BET_PCT_50
    assert info.legal_mask[BET_PCT_50]
    obs2, _, _, info2 = env.step(BET_PCT_50)
    if info2.actor is not None:
        # Actor is now a different seat; hero (seat from previous step) has
        # committed chips and its SPR should have decreased. We read hero's
        # SPR via the active mask rotation — that's now at an offset, not 0.
        # Simpler: assert pot increased and stacks for the prior actor
        # decreased by the sizing (594 for BetPct33 pot bet on 1800).
        import numpy as _np
        # Actor changed, so the 'hero' of obs2 is a different seat. We check
        # that SPR for the prior aggressor has indeed dropped by verifying
        # the minimum SPR across the padded block is below the pre-bet hero SPR.
        spr_after = obs2[_SPR_OFF : _SPR_OFF + 8]
        alive_spr = spr_after[spr_after > 0]
        assert float(alive_spr.min()) < float(spr_before)


def test_hand_category_onehot_exactly_one() -> None:
    env = BombPotEnv(GameConfig())
    obs, _ = env.reset(11, 0)
    cat_a = obs[_CAT_A_OFF : _CAT_A_OFF + _NUM_CATEGORIES]
    cat_b = obs[_CAT_B_OFF : _CAT_B_OFF + _NUM_CATEGORIES]
    # Exactly one slot set on each board (category is always defined at flop).
    assert int(cat_a.sum()) == 1
    assert int(cat_b.sum()) == 1


def test_draw_flags_binary() -> None:
    env = BombPotEnv(GameConfig())
    obs, _ = env.reset(17, 0)
    flags = obs[_DRAW_A_OFF : _DRAW_A_OFF + 4]
    for v in flags:
        assert float(v) in (0.0, 1.0)


def test_effective_stack_encoding() -> None:
    # The stack feature uses a hand-start frozen effective cap: per seat,
    # cap = min(own_starting, max(other_alive_starting_stacks)). The encoded
    # stack is min(current_stack, cap) so it represents prospective leverage
    # against the deepest reachable opponent at hand start. The cap does NOT
    # shrink mid-hand as opponents fold — varied stacks must continue to
    # influence flop/turn/river strategy after early folds.
    bb = 10_000
    cfg = GameConfig(
        num_seats=3,
        starting_stack=0,
        starting_stacks=(500 * bb, 200 * bb, 1000 * bb),
        ante=30_000,
        bb=bb,
    )
    # 3 seats alive; stacks 500bb / 200bb / 1000bb.
    # Frozen cap at hand start:
    #   seat 0 (500): min(500, max(200, 1000)) = 500
    #   seat 1 (200): min(200, max(500, 1000)) = 200
    #   seat 2 (1000): min(1000, max(500, 200)) = 500
    eff_cap = [500 * bb, 200 * bb, 500 * bb]
    obs_dict = {
        "actor": 0,
        "hero_hole": [0, 1, 2, 3, 4],
        "board_a": [10, 11, 12],
        "board_b": [20, 21, 22],
        "street": 1,
        "folded": [False, False, False],
        "all_in": [False, False, False],
        "stacks": [500 * bb, 200 * bb, 1000 * bb],
        "eff_stack_cap": eff_cap,
        "pot": 30_000 * 3,
        "bet_to_call": 0,
        "min_bet": bb,
        "max_bet": 500 * bb,
        "history": [],
        "street_commit": [0, 0, 0],
        "total_commit": [0, 0, 0],
        "button": 0,
        "last_aggressor": -1,
        "hero_category_a": 0,
        "hero_category_b": 0,
    }
    obs = encode_observation(obs_dict, cfg)
    # Hero-rotated with hero=seat 0:
    #   slot 0: min(stacks[0]=500, cap[0]=500) = 500
    #   slot 1: min(stacks[1]=200, cap[1]=200) = 200
    #   slot 2: min(stacks[2]=1000, cap[2]=500) = 500
    assert float(obs[_STACKS_OFF + 0]) == 500.0
    assert float(obs[_STACKS_OFF + 1]) == 200.0
    assert float(obs[_STACKS_OFF + 2]) == 500.0

    # Folding seat 2 must NOT shrink the encoded stacks. Cap is frozen
    # at hand start, so even though seat 2 leaves the hand, seat 0 and 1
    # continue to encode their pre-fold effective stack — preserving the
    # prospective-leverage signal the network was trained on.
    obs_dict["folded"] = [False, False, True]
    obs2 = encode_observation(obs_dict, cfg)
    assert float(obs2[_STACKS_OFF + 0]) == 500.0
    assert float(obs2[_STACKS_OFF + 1]) == 200.0


# --- Pair-with-board features ------------------------------------------------
#
# Card index = rank*4 + suit. Ranks: 0=2, 1=3, ..., 7=9, 8=T, 9=J, 10=Q, 11=K,
# 12=A. Suits 0=c, 1=d, 2=h, 3=s. Tests build obs_dicts directly so they
# don't depend on dealing a specific hand from the engine.


def _pair_obs(
    hero_hole: list[int], board_a: list[int], board_b: list[int] | None = None
) -> dict:
    return {
        "actor": 0,
        "hero_hole": hero_hole,
        "board_a": board_a,
        "board_b": board_b if board_b is not None else [],
        "street": 1,
        "folded": [False, False],
        "all_in": [False, False],
        "stacks": [200_000, 200_000],
        "eff_stack_cap": [200_000, 200_000],
        "pot": 30_000,
        "bet_to_call": 0,
        "min_bet": 10_000,
        "max_bet": 200_000,
        "history": [],
        "street_commit": [0, 0],
        "total_commit": [0, 0],
        "button": 0,
        "last_aggressor": -1,
        "hero_category_a": 0,
        "hero_category_b": 0,
    }


_PAIR_CFG = GameConfig(num_seats=2, starting_stack=200_000, ante=30_000, bb=10_000)


def _counts_a(obs: np.ndarray) -> list[float]:
    return [float(obs[_PAIR_COUNT_A_OFF + i]) for i in range(5)]


def _struct_a(obs: np.ndarray) -> list[float]:
    return [float(obs[_BOARD_STRUCT_A_OFF + i]) for i in range(4)]


def test_pair_count_top_pair() -> None:
    # hero K-Q-J-T-2 on flop K-9-4 → top pair only.
    hero = [47, 42, 37, 32, 2]  # Ks, Qh, Jd, Tc, 2h
    board = [44, 28, 8]  # Kc, 9c, 4c
    obs = encode_observation(_pair_obs(hero, board), _PAIR_CFG)
    assert _counts_a(obs) == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert _struct_a(obs) == [0.0, 0.0, 0.0, 0.0]


def test_pair_count_top_set() -> None:
    # hero K-K-Q-J-T on flop K-9-4 → top set, slot value 2.0.
    hero = [45, 46, 40, 36, 32]  # Kd, Kh, Qc, Jc, Tc
    board = [44, 28, 8]  # Kc, 9c, 4c
    obs = encode_observation(_pair_obs(hero, board), _PAIR_CFG)
    assert _counts_a(obs) == [2.0, 0.0, 0.0, 0.0, 0.0]
    assert _struct_a(obs) == [0.0, 0.0, 0.0, 0.0]


def test_pair_count_three_pair() -> None:
    # hero K-Q-J-9-4 on flop K-9-4 → matches all three board ranks.
    hero = [47, 40, 36, 30, 10]  # Ks, Qc, Jc, 9h, 4h
    board = [44, 28, 8]  # Kc, 9c, 4c
    obs = encode_observation(_pair_obs(hero, board), _PAIR_CFG)
    assert _counts_a(obs) == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert _struct_a(obs) == [0.0, 0.0, 0.0, 0.0]


def test_pair_count_overpair_zeros() -> None:
    # hero A-A-Q-J-T on flop K-9-4 → known blind spot, all zeros.
    hero = [48, 49, 40, 36, 32]  # Ac, Ad, Qc, Jc, Tc
    board = [44, 28, 8]  # Kc, 9c, 4c
    obs = encode_observation(_pair_obs(hero, board), _PAIR_CFG)
    assert _counts_a(obs) == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert _struct_a(obs) == [0.0, 0.0, 0.0, 0.0]


def test_paired_board_flop_repeated_slots() -> None:
    # flop K-K-9 → slot0 and slot1 both K-rank. Hero with one K.
    hero = [47, 40, 36, 32, 0]  # Ks, Qc, Jc, Tc, 2c
    board = [44, 45, 28]  # Kc, Kd, 9c
    obs = encode_observation(_pair_obs(hero, board), _PAIR_CFG)
    assert _counts_a(obs) == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert _struct_a(obs) == [1.0, 0.0, 0.0, 0.0]


def test_tripled_board_flop_repeated_slots() -> None:
    # flop K-K-K → all three slots K-rank. Hero with one K.
    hero = [47, 0, 4, 20, 36]  # Ks, 2c, 3c, 7c, Jc
    board = [44, 45, 46]  # Kc, Kd, Kh
    obs = encode_observation(_pair_obs(hero, board), _PAIR_CFG)
    assert _counts_a(obs) == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert _struct_a(obs) == [1.0, 0.0, 1.0, 0.0]


def test_double_paired_river_struct() -> None:
    # river K-K-9-9-2 → 5 active slots [K, K, 9, 9, 2]. Hero K + 9.
    hero = [47, 30, 4, 12, 20]  # Ks, 9h, 3c, 5c, 7c
    board = [44, 45, 28, 29, 0]  # Kc, Kd, 9c, 9d, 2c
    obs_dict = _pair_obs(hero, board)
    obs_dict["street"] = 3
    obs = encode_observation(obs_dict, _PAIR_CFG)
    assert _counts_a(obs) == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert _struct_a(obs) == [1.0, 1.0, 0.0, 0.0]


def test_full_house_board_river_struct() -> None:
    # river K-K-K-9-9 → slots [K,K,K,9,9]. Hero K + 9.
    hero = [47, 30, 4, 12, 20]  # Ks, 9h, 3c, 5c, 7c
    board = [44, 45, 46, 28, 29]  # Kc, Kd, Kh, 9c, 9d
    obs_dict = _pair_obs(hero, board)
    obs_dict["street"] = 3
    obs = encode_observation(obs_dict, _PAIR_CFG)
    assert _counts_a(obs) == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert _struct_a(obs) == [1.0, 1.0, 1.0, 0.0]


def test_quadded_river_struct() -> None:
    # river K-K-K-K-9 → slots [K,K,K,K,9]. Hero can never have a K
    # (deck exhausted) → slots 0..3 = 0; slot 4 = count of 9s in hand.
    hero = [30, 0, 4, 12, 20]  # 9h, 2c, 3c, 5c, 7c
    board = [44, 45, 46, 47, 28]  # Kc, Kd, Kh, Ks, 9c
    obs_dict = _pair_obs(hero, board)
    obs_dict["street"] = 3
    obs = encode_observation(obs_dict, _PAIR_CFG)
    assert _counts_a(obs) == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert _struct_a(obs) == [1.0, 0.0, 1.0, 1.0]


# --- Hero rank histogram -----------------------------------------------------


def _hist(obs: np.ndarray) -> list[float]:
    return [float(obs[_HERO_RANK_HIST_OFF + r]) for r in range(13)]


def test_hero_rank_hist_pocket_aces() -> None:
    # AA + 3 unrelated overcards on K-9-4 board → slot 12 (A) = 2.0.
    hero = [48, 49, 36, 32, 0]  # Ac, Ad, Jc, Tc, 2c
    board = [44, 28, 8]  # Kc, 9c, 4c (irrelevant for histogram)
    obs = encode_observation(_pair_obs(hero, board), _PAIR_CFG)
    h = _hist(obs)
    assert h[12] == 2.0  # A
    assert h[11] == 0.0  # K
    assert h[9] == 1.0  # J
    assert h[8] == 1.0  # T
    assert h[0] == 1.0  # 2
    assert sum(h) == 5.0


def test_hero_rank_hist_trip_aces() -> None:
    # AAA + 2 unrelated → slot 12 = 3.0.
    hero = [48, 49, 50, 32, 0]  # Ac, Ad, Ah, Tc, 2c
    board = [44, 28, 8]
    obs = encode_observation(_pair_obs(hero, board), _PAIR_CFG)
    h = _hist(obs)
    assert h[12] == 3.0
    assert h[8] == 1.0
    assert h[0] == 1.0
    assert sum(h) == 5.0


def test_hero_rank_hist_two_pocket_pairs() -> None:
    # AAKK + 1 unrelated on a 9-7-3 board → slot 12 = 2.0, slot 11 = 2.0.
    # (Hero has both overpair ranks not on board.)
    hero = [48, 49, 46, 47, 0]  # Ac, Ad, Kh, Ks, 2c
    board = [28, 20, 4]  # 9c, 7c, 3c
    obs = encode_observation(_pair_obs(hero, board), _PAIR_CFG)
    h = _hist(obs)
    assert h[12] == 2.0  # A
    assert h[11] == 2.0  # K
    assert h[0] == 1.0  # 2
    assert sum(h) == 5.0


def test_hero_rank_hist_quads_in_hand() -> None:
    # AAAA + 1 unrelated → slot 12 = 4.0. Rare but legal in PLO5.
    hero = [48, 49, 50, 51, 0]  # Ac, Ad, Ah, As, 2c
    board = [28, 20, 4]
    obs = encode_observation(_pair_obs(hero, board), _PAIR_CFG)
    h = _hist(obs)
    assert h[12] == 4.0
    assert h[0] == 1.0
    assert sum(h) == 5.0


def test_hero_rank_hist_unpaired_overcards() -> None:
    # AKQJT — no pair, five distinct ranks → five slots = 1.0.
    hero = [48, 47, 42, 37, 32]  # Ac, Ks, Qh, Jd, Tc
    board = [28, 20, 4]
    obs = encode_observation(_pair_obs(hero, board), _PAIR_CFG)
    h = _hist(obs)
    assert h[12] == 1.0  # A
    assert h[11] == 1.0  # K
    assert h[10] == 1.0  # Q
    assert h[9] == 1.0  # J
    assert h[8] == 1.0  # T
    assert sum(h) == 5.0


def test_encoding_invariant_to_unreachable_chips_above_eff_cap() -> None:
    # HU bomb-pot, hero $340. Villain $340 vs $440 vs $2000 — the deeper
    # seat's chips above max-other-reachable must not enter the encoded
    # observation. After eff_cap clamp on both _STACKS_OFF and _SPR_OFF,
    # all three vectors must be byte-identical.
    base = dict(num_seats=2, starting_stack=0, ante=60, bb=20)
    cfg_a = GameConfig(**base, starting_stacks=(340, 340))
    cfg_b = GameConfig(**base, starting_stacks=(340, 440))
    cfg_c = GameConfig(**base, starting_stacks=(340, 2000))
    obs_a, _ = BombPotEnv(cfg_a).reset(0, 0)
    obs_b, _ = BombPotEnv(cfg_b).reset(0, 0)
    obs_c, _ = BombPotEnv(cfg_c).reset(0, 0)
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(obs_a, obs_c)


# --- Straight / flush / SF features ------------------------------------------
#
# Slot indexing for straight windows: slot 0 = wheel {A,2,3,4,5}; slots 1..9
# = consecutive 5-rank windows starting at rank 0..8; slot 9 = broadway
# {T,J,Q,K,A}. Suits: 0=c, 1=d, 2=h, 3=s.


def test_straight_outs_q9_on_ajt() -> None:
    # Hero Q♣9♣ + low-club fillers; board_a A♠J♥T♦. Two open-end-style
    # double-gutter draws: (8) makes 8-9-T-J-Q (slot 7) and (K) makes
    # 9-T-J-Q-K (slot 8). Broadway (slot 9) needs both K and T, but hero
    # only contributes one rank (Q) — no draw, just board-feasible.
    hero = [40, 28, 0, 4, 8]  # Q♣, 9♣, 2♣, 3♣, 4♣
    board_a = [51, 38, 33]  # A♠, J♥, T♦
    board_b = [3, 7, 11]  # 2♠, 3♠, 4♠ (irrelevant ranks)
    obs = encode_observation(_pair_obs(hero, board_a, board_b), _PAIR_CFG)
    assert obs[_STRAIGHT_OUTS_A_OFF + 7] == 4.0  # one missing rank: 8
    assert obs[_STRAIGHT_OUTS_A_OFF + 8] == 4.0  # one missing rank: K
    assert obs[_STRAIGHT_OUTS_A_OFF + 9] == 0.0  # only one hero rank in window
    assert obs[_STRAIGHT_POSSIBLE_A_OFF + 9] == 1.0  # board has 3 broadway ranks


def test_broadway_outs_with_cross_board_blocker() -> None:
    # Hero A♠K♥ (+ low fillers); board_a Q♣J♦2♣; board_b T♠ + fillers.
    # Broadway (slot 9) needs T to complete; T♠ is on the OTHER board, so
    # only 3 unseen Ts remain — outs[9] = 4 - 1 = 3.
    hero = [51, 46, 10, 14, 18]  # A♠, K♥, 4♥, 5♥, 6♥
    board_a = [40, 37, 0]  # Q♣, J♦, 2♣
    board_b = [35, 9, 13]  # T♠, 4♦, 5♦
    obs = encode_observation(_pair_obs(hero, board_a, board_b), _PAIR_CFG)
    assert obs[_STRAIGHT_OUTS_A_OFF + 9] == 3.0


def test_already_made_straight_outs_zero() -> None:
    # Hero K♣Q♣ + low hearts; board_a J♥T♦9♠ → made 9-T-J-Q-K (slot 8).
    # straight_outs[8] must be 0 (already made). No higher window is
    # board-feasible (broadway needs 3 board ranks, board has only J,T,9
    # in slot 9), so straight_nut_distance = 0.
    hero = [44, 40, 2, 6, 10]  # K♣, Q♣, 2♥, 3♥, 4♥
    board_a = [38, 33, 31]  # J♥, T♦, 9♠
    board_b = [0, 4, 8]  # 2♣, 3♣, 4♣
    obs = encode_observation(_pair_obs(hero, board_a, board_b), _PAIR_CFG)
    assert obs[_STRAIGHT_OUTS_A_OFF + 8] == 0.0
    assert obs[_STRAIGHT_NUT_DIST_A_OFF] == 0.0


def test_straight_nut_distance_higher_window_possible() -> None:
    # Hero A♠2♣ + K♥Q♥J♥; board_a 3♦4♥5♣ → made wheel (slot 0). Higher
    # windows still board-possible: slot 1 = {2,3,4,5,6} has 3-4-5 → 1;
    # slot 2 = {3,4,5,6,7} has 3-4-5 → 1. Slots 3+ have ≤2 board ranks.
    # straight_nut_distance = 2.
    hero = [51, 0, 46, 42, 38]  # A♠, 2♣, K♥, Q♥, J♥
    board_a = [5, 10, 12]  # 3♦, 4♥, 5♣
    board_b = [19, 23, 27]  # 6♠, 7♠, 8♠ (don't add to A_a windows)
    obs = encode_observation(_pair_obs(hero, board_a, board_b), _PAIR_CFG)
    assert obs[_STRAIGHT_OUTS_A_OFF + 0] == 0.0  # wheel made → no outs
    assert obs[_STRAIGHT_NUT_DIST_A_OFF] == 2.0


def test_multi_suit_flush_draws() -> None:
    # Hero A♠Q♠K♥J♥2♣; board_a T♠5♠9♥3♥7♣ (river). Two simultaneous flush
    # draws: spades (hero top A♠ → 0 blockers → all 9 outs are nut) and
    # hearts (hero top K♥ → A♥ is the only blocker, unseen → 1 nut out).
    hero = [51, 43, 46, 38, 0]  # A♠, Q♠, K♥, J♥, 2♣
    board_a = [35, 15, 30, 6, 20]  # T♠, 5♠, 9♥, 3♥, 7♣
    board_b = [4, 8, 12]  # 3♣, 4♣, 5♣ (no hearts/spades)
    obs_dict = _pair_obs(hero, board_a, board_b)
    obs_dict["street"] = 3  # river, 5 board cards
    obs = encode_observation(obs_dict, _PAIR_CFG)
    # flush_draw_outs: spades (s=3) and hearts (s=2).
    assert obs[_FLUSH_DRAW_OUTS_A_OFF + 3] == 9.0  # 13 - 4 visible spades
    assert obs[_FLUSH_DRAW_OUTS_A_OFF + 2] == 9.0  # 13 - 4 visible hearts
    # nut_flush_draw_outs:
    assert obs[_NUT_FLUSH_DRAW_OUTS_A_OFF + 3] == 9.0  # all spades nut (A♠ in hand)
    assert obs[_NUT_FLUSH_DRAW_OUTS_A_OFF + 2] == 1.0  # one blocker (A♥)


def test_made_flush_nut_distance_zero_when_blockers_on_other_board() -> None:
    # Hero Q♥J♥ + non-heart fillers; board_a T♥7♥3♥ + fillers (3-flush →
    # made flush). board_b contains A♥ and K♥ — both blockers visible →
    # flush_nut_distance = 0.
    hero = [42, 38, 1, 5, 9]  # Q♥, J♥, 2♦, 3♦, 4♦
    board_a = [34, 22, 6, 15, 27]  # T♥, 7♥, 3♥, 5♠, 8♠
    board_b = [50, 46, 16, 17, 18]  # A♥, K♥, 6♣, 6♦, 6♥
    obs_dict = _pair_obs(hero, board_a, board_b)
    obs_dict["street"] = 3
    obs = encode_observation(obs_dict, _PAIR_CFG)
    assert obs[_FLUSH_NUT_DIST_A_OFF] == 0.0
    assert obs[_FLUSH_POSSIBLE_A_OFF + 2] == 1.0  # board has 3 hearts


def test_made_flush_nut_distance_with_unseen_blockers() -> None:
    # Hero Q♥J♥ + non-heart fillers; board_a T♥7♥3♥ + fillers; board_b has
    # zero hearts. K♥ and A♥ unseen → flush_nut_distance = 2.
    hero = [42, 38, 1, 5, 9]  # Q♥, J♥, 2♦, 3♦, 4♦
    board_a = [34, 22, 6, 15, 27]  # T♥, 7♥, 3♥, 5♠, 8♠
    board_b = [0, 4, 8, 13, 17]  # 2♣, 3♣, 4♣, 5♦, 6♦ (no hearts)
    obs_dict = _pair_obs(hero, board_a, board_b)
    obs_dict["street"] = 3
    obs = encode_observation(obs_dict, _PAIR_CFG)
    assert obs[_FLUSH_NUT_DIST_A_OFF] == 2.0
    assert obs[_FLUSH_POSSIBLE_A_OFF + 2] == 1.0


def test_no_flush_no_nut_distance() -> None:
    # Hero monotone clubs but board has no 3-suit and no 2-suit-in-hero
    # match → no flush, no draw, no nut distance.
    hero = [48, 44, 40, 36, 32]  # A♣, K♣, Q♣, J♣, T♣ (5 clubs)
    board_a = [13, 22, 31, 9, 18]  # 5♦, 7♥, 9♠, 4♦, 6♥ — d=2,h=2,s=1,c=0
    board_b = [3, 7, 11]
    obs_dict = _pair_obs(hero, board_a, board_b)
    obs_dict["street"] = 3
    obs = encode_observation(obs_dict, _PAIR_CFG)
    assert obs[_FLUSH_NUT_DIST_A_OFF] == 0.0
    for s in range(4):
        assert obs[_FLUSH_POSSIBLE_A_OFF + s] == 0.0
        assert obs[_FLUSH_DRAW_OUTS_A_OFF + s] == 0.0  # board has 0 clubs
        assert obs[_NUT_FLUSH_DRAW_OUTS_A_OFF + s] == 0.0


def test_flush_possible_per_suit_hearts() -> None:
    # board_a with 3 hearts on the flop → flush_possible[hearts] = 1, others 0.
    hero = [0, 4, 8, 12, 16]  # all clubs (no hearts)
    board_a = [50, 46, 22]  # A♥, K♥, 7♥
    board_b = [3, 7, 11]
    obs = encode_observation(_pair_obs(hero, board_a, board_b), _PAIR_CFG)
    assert obs[_FLUSH_POSSIBLE_A_OFF + 0] == 0.0  # clubs
    assert obs[_FLUSH_POSSIBLE_A_OFF + 1] == 0.0  # diamonds
    assert obs[_FLUSH_POSSIBLE_A_OFF + 2] == 1.0  # hearts
    assert obs[_FLUSH_POSSIBLE_A_OFF + 3] == 0.0  # spades


def test_sf_draw_separates_from_flush_draw() -> None:
    # Hero A♠K♠Q♣2♣3♥; board_a J♠T♠5♦. Spade flush draw (9 outs total),
    # but only Q♠ also completes a straight (broadway A-K-Q-J-T) → exactly
    # 1 SF out on spades. Other suits: no flush draw → SF outs = 0.
    hero = [51, 47, 40, 0, 6]  # A♠, K♠, Q♣, 2♣, 3♥
    board_a = [39, 35, 13]  # J♠, T♠, 5♦
    board_b = [4, 8, 12]  # 3♣, 4♣, 5♣
    obs = encode_observation(_pair_obs(hero, board_a, board_b), _PAIR_CFG)
    assert obs[_FLUSH_DRAW_OUTS_A_OFF + 3] == 9.0  # spades
    assert obs[_SF_DRAW_OUTS_A_OFF + 3] == 1.0  # Q♠ completes both
    assert obs[_SF_DRAW_OUTS_A_OFF + 0] == 0.0
    assert obs[_SF_DRAW_OUTS_A_OFF + 1] == 0.0
    assert obs[_SF_DRAW_OUTS_A_OFF + 2] == 0.0


# -----------------------------------------------------------------------------
# History-slot tests (gate one-hot + street one-hot + chips/bb).
#
# Slot dim = 17:  [seat 0..7] [gate 8..11] [street 12..15] [chips 16].
# Gate enum: Fold=0, Check=1, Call=2, Raise=3 — derived from
# (action, chips). Engine action codes: FOLD=0, CHECK_CALL=1, BetPct*/AllIn=2..7.
# -----------------------------------------------------------------------------


def _history_slot(obs: np.ndarray, slot: int) -> np.ndarray:
    base = _HISTORY_OFF + slot * _HISTORY_SLOT_DIM
    return obs[base : base + _HISTORY_SLOT_DIM]


def test_history_slot_fold_writes_gate_zero_chips_zero() -> None:
    from plo5bp.actions import FOLD
    obs_dict = _pair_obs([0, 1, 2, 3, 4], [44, 28, 8])
    # (seat, action, chips, street): seat 1 folded on flop (street=1).
    obs_dict["history"] = [(1, FOLD, 0, 1)]
    obs = encode_observation(obs_dict, _PAIR_CFG)
    slot = _history_slot(obs, 0)
    # Seat one-hot: hero=0, so rel_seat = (1-0) % 2 = 1.
    assert slot[1] == 1.0
    # Gate one-hot: Fold=0.
    assert slot[_HISTORY_GATE_OFF_REL + 0] == 1.0
    assert slot[_HISTORY_GATE_OFF_REL + 1] == 0.0
    assert slot[_HISTORY_GATE_OFF_REL + 2] == 0.0
    assert slot[_HISTORY_GATE_OFF_REL + 3] == 0.0
    # Street one-hot: flop = slot 1.
    assert slot[_HISTORY_STREET_OFF_REL + 1] == 1.0
    # Chips/bb = 0.
    assert slot[_HISTORY_CHIPS_OFF_REL] == 0.0


def test_history_slot_check_writes_gate_one_chips_zero() -> None:
    from plo5bp.actions import CHECK_CALL
    obs_dict = _pair_obs([0, 1, 2, 3, 4], [44, 28, 8])
    obs_dict["history"] = [(1, CHECK_CALL, 0, 1)]
    obs = encode_observation(obs_dict, _PAIR_CFG)
    slot = _history_slot(obs, 0)
    assert slot[_HISTORY_GATE_OFF_REL + 0] == 0.0
    assert slot[_HISTORY_GATE_OFF_REL + 1] == 1.0  # Check
    assert slot[_HISTORY_GATE_OFF_REL + 2] == 0.0
    assert slot[_HISTORY_GATE_OFF_REL + 3] == 0.0
    assert slot[_HISTORY_CHIPS_OFF_REL] == 0.0


def test_history_slot_call_writes_gate_two_chips_positive() -> None:
    from plo5bp.actions import CHECK_CALL
    obs_dict = _pair_obs([0, 1, 2, 3, 4], [44, 28, 8])
    # CHECK_CALL with chips > 0 → Call gate. chips = 20000 = 2 bb.
    obs_dict["history"] = [(1, CHECK_CALL, 20_000, 1)]
    obs = encode_observation(obs_dict, _PAIR_CFG)
    slot = _history_slot(obs, 0)
    assert slot[_HISTORY_GATE_OFF_REL + 0] == 0.0
    assert slot[_HISTORY_GATE_OFF_REL + 1] == 0.0
    assert slot[_HISTORY_GATE_OFF_REL + 2] == 1.0  # Call
    assert slot[_HISTORY_GATE_OFF_REL + 3] == 0.0
    # cfg.bb = 10_000, so chips/bb = 2.0.
    assert float(slot[_HISTORY_CHIPS_OFF_REL]) == 2.0


def test_history_slot_raise_writes_gate_three_chips_amount() -> None:
    from plo5bp.actions import BET_PCT_50
    obs_dict = _pair_obs([0, 1, 2, 3, 4], [44, 28, 8])
    # BetPct50 raise to 15000 chips = 1.5 bb.
    obs_dict["history"] = [(1, BET_PCT_50, 15_000, 1)]
    obs = encode_observation(obs_dict, _PAIR_CFG)
    slot = _history_slot(obs, 0)
    assert slot[_HISTORY_GATE_OFF_REL + 0] == 0.0
    assert slot[_HISTORY_GATE_OFF_REL + 1] == 0.0
    assert slot[_HISTORY_GATE_OFF_REL + 2] == 0.0
    assert slot[_HISTORY_GATE_OFF_REL + 3] == 1.0  # Raise
    assert float(slot[_HISTORY_CHIPS_OFF_REL]) == 1.5


def test_history_slot_allin_writes_gate_three() -> None:
    from plo5bp.actions import ALL_IN
    obs_dict = _pair_obs([0, 1, 2, 3, 4], [44, 28, 8])
    obs_dict["history"] = [(0, ALL_IN, 200_000, 1)]
    obs = encode_observation(obs_dict, _PAIR_CFG)
    slot = _history_slot(obs, 0)
    assert slot[_HISTORY_GATE_OFF_REL + 3] == 1.0
    # Hero seat = 0, rel_seat = 0.
    assert slot[0] == 1.0
    assert float(slot[_HISTORY_CHIPS_OFF_REL]) == 20.0  # 200k / 10k bb


def test_history_slot_street_one_hot() -> None:
    from plo5bp.actions import CHECK_CALL
    obs_dict = _pair_obs([0, 1, 2, 3, 4], [44, 28, 8])
    # 4 entries spanning preflop, flop, turn, river.
    obs_dict["history"] = [
        (0, CHECK_CALL, 0, 0),
        (1, CHECK_CALL, 0, 1),
        (0, CHECK_CALL, 0, 2),
        (1, CHECK_CALL, 0, 3),
    ]
    obs = encode_observation(obs_dict, _PAIR_CFG)
    for slot_i, expected_street in enumerate((0, 1, 2, 3)):
        slot = _history_slot(obs, slot_i)
        for s in range(4):
            expected = 1.0 if s == expected_street else 0.0
            assert slot[_HISTORY_STREET_OFF_REL + s] == expected, (
                f"slot {slot_i} street bit {s} mismatch"
            )


def test_history_slot_chips_in_bb_units() -> None:
    from plo5bp.actions import BET_PCT_100
    obs_dict = _pair_obs([0, 1, 2, 3, 4], [44, 28, 8])
    raw_chips = 75_000
    obs_dict["history"] = [(1, BET_PCT_100, raw_chips, 1)]
    obs = encode_observation(obs_dict, _PAIR_CFG)
    slot = _history_slot(obs, 0)
    # cfg.bb = 10_000, so 75_000 / 10_000 = 7.5.
    assert float(slot[_HISTORY_CHIPS_OFF_REL]) == raw_chips / _PAIR_CFG.bb


def test_seat_exists_mask_matches_num_seats() -> None:
    # 2-seat: slots 0..1 = 1, slots 2..7 = 0.
    obs2 = encode_observation(_pair_obs([0, 1, 2, 3, 4], [44, 28, 8]), _PAIR_CFG)
    mask2 = obs2[_SEAT_EXISTS_OFF : _SEAT_EXISTS_OFF + 8]
    assert list(mask2) == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # 6-seat: slots 0..5 = 1, slots 6..7 = 0. Build a 6-seat obs dict by
    # overriding the 2-seat-shaped fields in _pair_obs's output.
    six_cfg = GameConfig(num_seats=6, starting_stack=200_000, ante=30_000, bb=10_000)
    obs_dict = _pair_obs([0, 1, 2, 3, 4], [44, 28, 8])
    obs_dict["folded"] = [False] * 6
    obs_dict["all_in"] = [False] * 6
    obs_dict["stacks"] = [200_000] * 6
    obs_dict["eff_stack_cap"] = [200_000] * 6
    obs_dict["street_commit"] = [0] * 6
    obs_dict["total_commit"] = [0] * 6
    obs6 = encode_observation(obs_dict, six_cfg)
    mask6 = obs6[_SEAT_EXISTS_OFF : _SEAT_EXISTS_OFF + 8]
    assert list(mask6) == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_seat_exists_independent_of_stack() -> None:
    # A 0-chip seat should still register as seated; the seat-exists mask
    # is structural, not stack-driven.
    obs_dict = _pair_obs([0, 1, 2, 3, 4], [44, 28, 8])
    obs_dict["stacks"] = [0, 200_000]  # hero has 0 chips left, opp full
    obs = encode_observation(obs_dict, _PAIR_CFG)
    mask = obs[_SEAT_EXISTS_OFF : _SEAT_EXISTS_OFF + 8]
    assert mask[0] == 1.0  # hero seat exists despite 0 stack
    assert mask[1] == 1.0
    assert list(mask[2:]) == [0.0] * 6


def test_total_commit_hero_rotated() -> None:
    obs_dict = _pair_obs([0, 1, 2, 3, 4], [44, 28, 8])
    obs_dict["total_commit"] = [5_000, 30_000]  # hero=0
    obs = encode_observation(obs_dict, _PAIR_CFG)
    block = obs[_TOTAL_COMMIT_OFF : _TOTAL_COMMIT_OFF + 8]
    # cfg.bb=10_000 → slot 0 = 0.5, slot 1 = 3.0, rest 0.
    assert float(block[0]) == 0.5
    assert float(block[1]) == 3.0
    assert list(block[2:]) == [0.0] * 6


def test_street_commit_hero_rotated() -> None:
    obs_dict = _pair_obs([0, 1, 2, 3, 4], [44, 28, 8])
    obs_dict["street_commit"] = [10_000, 0]
    obs = encode_observation(obs_dict, _PAIR_CFG)
    block = obs[_STREET_COMMIT_OFF : _STREET_COMMIT_OFF + 8]
    assert float(block[0]) == 1.0
    assert float(block[1]) == 0.0
    assert list(block[2:]) == [0.0] * 6


def test_last_aggressor_one_hot() -> None:
    obs_dict = _pair_obs([0, 1, 2, 3, 4], [44, 28, 8])
    obs_dict["last_aggressor"] = 1  # opponent raised
    obs = encode_observation(obs_dict, _PAIR_CFG)
    block = obs[_LAST_AGGRESSOR_OFF : _LAST_AGGRESSOR_OFF + 8]
    assert float(block[1]) == 1.0
    assert list(block[:1]) + list(block[2:]) == [0.0] * 7


def test_last_aggressor_none_all_zero() -> None:
    obs_dict = _pair_obs([0, 1, 2, 3, 4], [44, 28, 8])
    obs_dict["last_aggressor"] = -1
    obs = encode_observation(obs_dict, _PAIR_CFG)
    block = obs[_LAST_AGGRESSOR_OFF : _LAST_AGGRESSOR_OFF + 8]
    assert list(block) == [0.0] * 8


def test_hero_button_distance_two_seat() -> None:
    obs_dict = _pair_obs([0, 1, 2, 3, 4], [44, 28, 8])
    obs_dict["button"] = 1
    obs = encode_observation(obs_dict, _PAIR_CFG)
    block = obs[_HERO_BTN_DIST_OFF : _HERO_BTN_DIST_OFF + 8]
    # (button - hero) % num_seats = (1 - 0) % 2 = 1.
    assert float(block[1]) == 1.0
    assert list(block[:1]) + list(block[2:]) == [0.0] * 7


def test_hero_button_distance_six_seat() -> None:
    six_cfg = GameConfig(num_seats=6, starting_stack=200_000, ante=30_000, bb=10_000)
    obs_dict = _pair_obs([0, 1, 2, 3, 4], [44, 28, 8])
    obs_dict["folded"] = [False] * 6
    obs_dict["all_in"] = [False] * 6
    obs_dict["stacks"] = [200_000] * 6
    obs_dict["eff_stack_cap"] = [200_000] * 6
    obs_dict["street_commit"] = [0] * 6
    obs_dict["total_commit"] = [0] * 6
    obs_dict["button"] = 3
    obs = encode_observation(obs_dict, six_cfg)
    block = obs[_HERO_BTN_DIST_OFF : _HERO_BTN_DIST_OFF + 8]
    # (3 - 0) % 6 = 3.
    assert float(block[3]) == 1.0
    others = list(block[:3]) + list(block[4:])
    assert others == [0.0] * 7


# Cross-board interaction tests (shared ranks + hero-involved cross-board
# flush + hero-rank-pair-aggregated cross-board straight).
def test_shared_ranks_flop() -> None:
    # A=[Ks, 9c, 4c] ranks {11, 7, 2}; B=[Kh, 9d, 2s] ranks {11, 7, 0}.
    # Shared = {11, 7}. Hero unrelated.
    obs = encode_observation(
        _pair_obs([49, 14, 16, 32, 39], [47, 28, 8], [46, 29, 3]), _PAIR_CFG
    )
    mask = obs[_SHARED_RANKS_OFF : _SHARED_RANKS_OFF + 13]
    expected = [0.0] * 13
    expected[7] = 1.0
    expected[11] = 1.0
    assert list(mask) == expected


def test_shared_ranks_no_overlap() -> None:
    # A=[2c, 3c, 4c] ranks {0,1,2}; B=[Tc, Jc, Qc] ranks {8,9,10}. No overlap.
    obs = encode_observation(
        _pair_obs([49, 14, 18, 47, 39], [0, 4, 8], [32, 36, 40]), _PAIR_CFG
    )
    mask = obs[_SHARED_RANKS_OFF : _SHARED_RANKS_OFF + 13]
    assert list(mask) == [0.0] * 13


def test_flush_made_both_per_suit() -> None:
    # Hero 2 hearts (Ah, Kh); A 3 hearts; B 3 hearts.
    obs = encode_observation(
        _pair_obs([50, 46, 0, 5, 11], [2, 6, 10], [14, 18, 22]), _PAIR_CFG
    )
    made = obs[_FLUSH_MADE_BOTH_OFF : _FLUSH_MADE_BOTH_OFF + 4]
    draw = obs[_FLUSH_DRAW_BOTH_OFF : _FLUSH_DRAW_BOTH_OFF + 4]
    mixed = obs[_FLUSH_MIXED_OFF : _FLUSH_MIXED_OFF + 4]
    assert list(made) == [0.0, 0.0, 1.0, 0.0]  # suit 2 = hearts
    assert list(draw) == [0.0] * 4
    assert list(mixed) == [0.0] * 4


def test_flush_draw_both_per_suit() -> None:
    # Hero 2 hearts; A 2 hearts; B 2 hearts.
    obs = encode_observation(
        _pair_obs([50, 46, 0, 5, 11], [2, 6, 23], [10, 14, 27]), _PAIR_CFG
    )
    made = obs[_FLUSH_MADE_BOTH_OFF : _FLUSH_MADE_BOTH_OFF + 4]
    draw = obs[_FLUSH_DRAW_BOTH_OFF : _FLUSH_DRAW_BOTH_OFF + 4]
    mixed = obs[_FLUSH_MIXED_OFF : _FLUSH_MIXED_OFF + 4]
    assert list(made) == [0.0] * 4
    assert list(draw) == [0.0, 0.0, 1.0, 0.0]
    assert list(mixed) == [0.0] * 4


def test_flush_mixed_per_suit() -> None:
    # Hero 2 clubs (Ac, Kc); A 3 clubs; B 2 clubs.
    obs = encode_observation(
        _pair_obs([48, 44, 1, 6, 11], [0, 4, 8], [12, 16, 23]), _PAIR_CFG
    )
    made = obs[_FLUSH_MADE_BOTH_OFF : _FLUSH_MADE_BOTH_OFF + 4]
    draw = obs[_FLUSH_DRAW_BOTH_OFF : _FLUSH_DRAW_BOTH_OFF + 4]
    mixed = obs[_FLUSH_MIXED_OFF : _FLUSH_MIXED_OFF + 4]
    assert list(made) == [0.0] * 4
    assert list(draw) == [0.0] * 4
    assert list(mixed) == [1.0, 0.0, 0.0, 0.0]


def test_flush_cross_requires_hero_involvement() -> None:
    # Hero 0 hearts; both boards 3 hearts.
    obs = encode_observation(
        _pair_obs([0, 5, 8, 13, 16], [2, 6, 10], [14, 18, 22]), _PAIR_CFG
    )
    made = obs[_FLUSH_MADE_BOTH_OFF : _FLUSH_MADE_BOTH_OFF + 4]
    draw = obs[_FLUSH_DRAW_BOTH_OFF : _FLUSH_DRAW_BOTH_OFF + 4]
    mixed = obs[_FLUSH_MIXED_OFF : _FLUSH_MIXED_OFF + 4]
    assert list(made) == [0.0] * 4
    assert list(draw) == [0.0] * 4
    assert list(mixed) == [0.0] * 4


def test_straight_freeroll_jt_across_windows() -> None:
    # Hero JT plus low fillers. A=KQ7 → JT draws via {7..11}. B=Q9-5 → JT
    # draws via {6..10}. Same hero pair {T,J}, different windows on each
    # board.
    obs = encode_observation(
        _pair_obs([39, 32, 0, 4, 8], [47, 41, 20], [42, 29, 12]), _PAIR_CFG
    )
    assert float(obs[_STRAIGHT_MADE_BOTH_OFF]) == 0.0
    assert float(obs[_STRAIGHT_DRAW_BOTH_OFF]) == 1.0
    assert float(obs[_STRAIGHT_MIXED_OFF]) == 0.0


def test_straight_made_both_same_pair() -> None:
    # Hero JT plus fillers. A=9QK → JT plays straight in {7..11}. B=9QK
    # different suits → same straight on B.
    obs = encode_observation(
        _pair_obs([39, 32, 0, 4, 11], [28, 41, 44], [29, 42, 47]), _PAIR_CFG
    )
    assert float(obs[_STRAIGHT_MADE_BOTH_OFF]) == 1.0
    assert float(obs[_STRAIGHT_DRAW_BOTH_OFF]) == 0.0
    assert float(obs[_STRAIGHT_MIXED_OFF]) == 0.0


def test_straight_no_cross() -> None:
    # Hero ranks {0, 5, 8} only — only contiguous-window pair is (5, 8).
    # Boards have ranks {2, 11, 12} which lie outside windows {4..8} and
    # {5..9}, so no straight indicator fires.
    obs = encode_observation(
        _pair_obs([0, 1, 20, 21, 34], [9, 47, 50], [10, 44, 49]), _PAIR_CFG
    )
    assert float(obs[_STRAIGHT_MADE_BOTH_OFF]) == 0.0
    assert float(obs[_STRAIGHT_DRAW_BOTH_OFF]) == 0.0
    assert float(obs[_STRAIGHT_MIXED_OFF]) == 0.0


def test_straight_mixed_made_a_draw_b() -> None:
    # Hero JT. A=9QK (made via {7..11}). B=Q9-2 (draw — {7..11} cov=4).
    obs = encode_observation(
        _pair_obs([39, 32, 0, 4, 8], [28, 41, 44], [42, 29, 3]), _PAIR_CFG
    )
    assert float(obs[_STRAIGHT_MADE_BOTH_OFF]) == 0.0
    assert float(obs[_STRAIGHT_DRAW_BOTH_OFF]) == 0.0
    assert float(obs[_STRAIGHT_MIXED_OFF]) == 1.0


# --- Opp-outcome combo-count features ---------------------------------------
#
# 12 dims at _OPP_OUTCOME_OFF: row-major [k=2,3,4][outcome] with outcome
# enum (scoop_opp, quarter_opp, scoop_hero, quarter_hero). Computed by the
# Rust engine; here we exercise the encoder splice and run a few real-engine
# states through it for plausibility.


def _opp_slice(obs: np.ndarray) -> np.ndarray:
    return obs[_OPP_OUTCOME_OFF : _OPP_OUTCOME_OFF + _OPP_OUTCOME_DIM]


def test_opp_outcome_dim_constants() -> None:
    assert _OPP_OUTCOME_DIM == 12
    assert _OPP_OUTCOME_OFF + _OPP_OUTCOME_DIM == _BET_PCT_POT_OFF
    assert _BET_PCT_POT_OFF + 1 == OBS_DIM


def test_opp_outcome_slice_default_zero_when_dict_missing_key() -> None:
    # _pair_obs doesn't set opp_outcome_fractions → encoder leaves zeros.
    obs = encode_observation(
        _pair_obs([0, 1, 2, 3, 4], [10, 11, 12], [20, 21, 22]), _PAIR_CFG
    )
    assert _opp_slice(obs).tolist() == [0.0] * 12


def test_opp_outcome_slice_writes_supplied_values() -> None:
    raw = _pair_obs([0, 1, 2, 3, 4], [10, 11, 12], [20, 21, 22])
    payload = [0.1 * (i + 1) for i in range(12)]
    raw["opp_outcome_fractions"] = payload
    obs = encode_observation(raw, _PAIR_CFG)
    np.testing.assert_allclose(_opp_slice(obs), payload, rtol=1e-6)


def test_opp_outcome_real_engine_flop_in_range() -> None:
    env = BombPotEnv(GameConfig())
    obs, _ = env.reset(123, 0)
    sl = _opp_slice(obs)
    # Fractions live in [0, 1].
    assert (sl >= 0.0).all()
    assert (sl <= 1.0).all()
    # On the flop with a real engine, at least one entry should fire
    # (chops/splits are excluded but full domination of all 4 outcomes
    # being identically zero across all k is implausible for a random hand).
    assert sl.sum() > 0.0


def test_opp_outcome_deterministic_across_repeated_pack() -> None:
    # Same engine state encoded twice must produce identical fractions.
    env = BombPotEnv(GameConfig())
    obs1, _ = env.reset(7, 0)
    obs2, _ = env._pack_obs()
    np.testing.assert_array_equal(_opp_slice(obs1), _opp_slice(obs2))


def test_opp_outcome_preflop_returns_zero_via_partial_dict() -> None:
    # No board → engine returns zeros; encoder slice all-zero.
    raw = _pair_obs([0, 1, 2, 3, 4], [], [])
    raw["street"] = 0  # preflop
    raw["opp_outcome_fractions"] = [0.0] * 12
    obs = encode_observation(raw, _PAIR_CFG)
    assert _opp_slice(obs).tolist() == [0.0] * 12
