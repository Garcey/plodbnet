"""Pin the made-hand describer against the engine's own ranking.

The label shown in the trainer must never disagree with how a showdown
actually ranks the hand, so `best_hand_category` is compared to the
engine's `hero_category(seat, board)` for every seat, both boards, across
flop/turn/river over many dealt hands. Plus spot checks on the exact
ClubGG-style wording.
"""

from itertools import combinations

from plo5bp.actions import GATE_CHECK_CALL
from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv
from plo5bp.ui.hand_describe import (
    _eval5,
    best_hand_category,
    describe_made_hand,
)


def test_category_matches_engine_over_dealt_hands():
    cfg = GameConfig()  # 6-seat default bomb pot
    mism = []
    n = 0
    for seed in range(150):
        env = BombPotEnv(cfg)
        _, info = env.reset(seed, seed % cfg.num_seats)
        done = False
        for _ in range(40):
            raw = info.raw_obs
            holes = env.all_hole_cards()
            for board_idx, key in ((0, "board_a"), (1, "board_b")):
                board = [int(c) for c in raw[key]]
                if len(board) < 3:
                    continue
                for seat in range(cfg.num_seats):
                    mine = best_hand_category(holes[seat], board)
                    eng = int(env._rs.hero_category(seat, board_idx))
                    n += 1
                    if mine != eng:
                        mism.append((seed, seat, board_idx, mine, eng,
                                     list(holes[seat]), board))
            if done or info.actor is None:
                break
            _, _, done, info = env.step_hybrid(GATE_CHECK_CALL, 0)
    assert n > 5000, f"too few comparisons ({n}) — test not exercising streets"
    assert not mism, f"{len(mism)}/{n} category mismatches; first 5: {mism[:5]}"


def _c(rank_idx, suit):  # build a card index from rank 0..12 + suit 0..3
    return rank_idx * 4 + suit


def test_wording_spot_checks():
    # exactly-2-hole + 3-board, so hand the relevant cards as hole+board.
    # broadway straight T-J-Q-K-A: hole has T,A; board has J,Q,K
    assert describe_made_hand(
        [_c(8, 0), _c(12, 1), _c(0, 2), _c(2, 3), _c(4, 0)],   # 10s,Ad,4h,6s,...
        [_c(9, 0), _c(10, 1), _c(11, 2)],                       # Jc,Qd,Kh
    ) == "a straight 10-A"

    # wheel A-2-3-4-5: hole A,2; board 3,4,5
    assert describe_made_hand(
        [_c(12, 0), _c(0, 1), _c(8, 2), _c(9, 3), _c(11, 0)],
        [_c(1, 0), _c(2, 1), _c(3, 2)],
    ) == "a straight A-5"

    # trip Qs: hole Q,Q; board Q,x,y (distinct, non-pairing)
    assert describe_made_hand(
        [_c(10, 0), _c(10, 1), _c(0, 2), _c(5, 3), _c(7, 0)],
        [_c(10, 2), _c(2, 0), _c(4, 1)],
    ) == "three of a kind, Qs"

    # pair of 8s: hole 8,8; board three distinct lowers/non-pairs
    assert describe_made_hand(
        [_c(6, 0), _c(6, 1), _c(0, 2), _c(11, 3), _c(9, 0)],
        [_c(2, 0), _c(4, 1), _c(7, 2)],
    ) == "a pair of 8s"

    # quads 8s: hole 8,8; board 8,8,x
    assert describe_made_hand(
        [_c(6, 0), _c(6, 1), _c(12, 2), _c(0, 3), _c(5, 0)],
        [_c(6, 2), _c(6, 3), _c(2, 0)],
    ) == "four of a kind, 8s"

    # nut flush A high: hole Ah, Kh (two hearts); board has 3 hearts
    assert describe_made_hand(
        [_c(12, 2), _c(11, 2), _c(0, 0), _c(2, 1), _c(4, 3)],
        [_c(3, 2), _c(7, 2), _c(9, 2)],
    ) == "a flush A high"

    # straight flush 9-K in clubs (suit 0): hole 9c,Kc... need 9,10,J,Q,K clubs
    assert describe_made_hand(
        [_c(7, 0), _c(11, 0), _c(0, 1), _c(2, 2), _c(4, 3)],     # 9c,Kc
        [_c(8, 0), _c(9, 0), _c(10, 0)],                         # 10c,Jc,Qc
    ) == "a straight flush, 9-K"


def test_too_few_cards_returns_none():
    assert describe_made_hand([_c(6, 0), _c(6, 1)], [_c(2, 0), _c(4, 1)]) is None
    assert best_hand_category([], []) is None
