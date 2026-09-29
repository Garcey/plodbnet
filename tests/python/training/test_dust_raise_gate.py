"""The cover-short DUST regime must not surface a degenerate raise gate.

When the deepest alive opponent can be covered for only a sub-1bb amount
(everyone else all-in), the engine reports a tiny `max_raise` (the opponent's
dust reach) with NO legal discrete bet/all-in action. Pre-fix,
`gate_mask_from_bounds` offered GATE_RAISE whenever `max_raise > 0`, which the
UI rendered as a degenerate "bet $0.00" option (and let the network 'raise' a
sub-cent amount). The dust screen suppresses that, while preserving genuine
sub-1bb all-in short shoves and >=1bb covering bets.
"""
import numpy as np

from plo5bp.actions import (
    ALL_IN,
    BET_PCT_50,
    CHECK_CALL,
    FOLD,
    GATE_CHECK_CALL,
    GATE_RAISE,
    gate_mask_from_bounds,
)

BB = 10000


def _legal(*idxs: int) -> np.ndarray:
    m = np.zeros(8, dtype=bool)
    for i in idxs:
        m[i] = True
    return m


def test_legacy_default_offers_dust_raise():
    # min_bet defaults to 0 -> screen disabled -> the old (buggy) behaviour,
    # which the existing callers/tests relied on.
    legal = _legal(CHECK_CALL)
    assert bool(gate_mask_from_bounds(legal, 10)[GATE_RAISE])


def test_dust_cover_without_allin_is_suppressed():
    # Actor faces no bet; only a sub-1bb continuous cover exists, no discrete
    # bet and no all-in shove are legal -> the user's "bet $0.00" node.
    legal = _legal(CHECK_CALL)
    gm = gate_mask_from_bounds(legal, 10, BB)
    assert not gm[GATE_RAISE]
    assert gm[GATE_CHECK_CALL]


def test_sub_bb_allin_shove_is_preserved():
    # Everyone short with equal sub-1bb stacks: a real, callable all-in shove.
    legal = _legal(CHECK_CALL, ALL_IN)
    assert gate_mask_from_bounds(legal, 2019, BB)[GATE_RAISE]


def test_cover_of_at_least_one_bb_is_preserved():
    # Covering an opponent who can still reach >=1bb is a real wager.
    legal = _legal(CHECK_CALL)
    assert gate_mask_from_bounds(legal, BB, BB)[GATE_RAISE]
    assert gate_mask_from_bounds(legal, 3 * BB, BB)[GATE_RAISE]


def test_full_raise_is_preserved():
    legal = _legal(CHECK_CALL, BET_PCT_50, ALL_IN)
    assert gate_mask_from_bounds(legal, 50000, BB)[GATE_RAISE]


def test_no_raise_when_max_raise_zero():
    # All-in-for-less as a CALL: max_raise == 0 -> never the raise gate.
    legal = _legal(FOLD, CHECK_CALL, ALL_IN)
    assert not gate_mask_from_bounds(legal, 0, BB)[GATE_RAISE]


def test_batched_matches_scalar():
    legal = np.stack([
        _legal(CHECK_CALL),                     # dust cover, no all-in
        _legal(CHECK_CALL, ALL_IN),             # sub-bb all-in shove
        _legal(CHECK_CALL, BET_PCT_50, ALL_IN),  # full raise
        _legal(FOLD, CHECK_CALL, ALL_IN),       # all-in-as-call, max_raise 0
    ])
    max_raise = np.array([10, 2019, 50000, 0], dtype=np.uint64)
    gm = gate_mask_from_bounds(legal, max_raise, BB)
    assert gm[:, GATE_RAISE].tolist() == [False, True, True, False]
    # row-wise parity with the scalar path
    for i in range(legal.shape[0]):
        scalar = gate_mask_from_bounds(legal[i], int(max_raise[i]), BB)
        assert scalar.tolist() == gm[i].tolist()
