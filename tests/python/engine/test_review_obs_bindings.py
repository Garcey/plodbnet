"""Regression tests for the PyO3-binding findings of the 2026-09-20 code
review (C4, C6, C7 — binding side).

C4: bad Python input must raise `ValueError`, never a Rust panic. A panic
surfaces as `pyo3_runtime.PanicException`, which derives from `BaseException`
and therefore sails straight past `except Exception` (e.g. the
`try/except Exception` around `reset_nlh_cfr_node` in gto/obs_from_label.py).
`pytest.raises(ValueError)` fails on a PanicException, so every case below
pins both "it raises" and "it is an ordinary Exception".

All of these exercise the Rust side and need an `_engine` binary rebuilt from
this tree.
"""

from __future__ import annotations

import numpy as np
import pytest

from plo5bp._engine import (  # type: ignore[attr-defined]
    BatchedEngine,
    GameState,
    cross_board_straight_batch,
    draw_flags_batch,
    pair_features_batch,
    straight_flush_features_batch,
)
from plo5bp.config import VARIANT_NLH, GameConfig
from plo5bp.encoding_nlh import encode_observation_batch_nlh

BB = 10_000


def _be(n: int = 4, **kw) -> BatchedEngine:
    be = BatchedEngine(n, **kw)
    be.reset_batch(np.arange(n, dtype=np.uint64), np.zeros(n, dtype=np.uint8))
    return be


# ---------------------------------------------------------------------------
# C4 — constructors validate the table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        dict(num_seats=0),
        dict(num_seats=1),                               # "need at least 2 seats" panic
        dict(num_seats=9),                               # PLO5: 45 + 10 cards
        dict(num_seats=11),
        dict(num_seats=8, variant="plo6_double_bomb"),   # 48 + 10 cards
        dict(num_seats=9, variant="plo4_double_bomb"),   # fits the deck, not the 8-slot obs
        dict(num_seats=9, variant="nlh_single", sb=5_000),
        dict(bb=0),                                      # every encoder scales by 1/bb
    ],
)
def test_constructors_reject_bad_tables(kw) -> None:
    with pytest.raises(ValueError):
        GameState(**kw)
    with pytest.raises(ValueError):
        BatchedEngine(2, **kw)


@pytest.mark.parametrize(
    "kw",
    [
        dict(num_seats=8),
        dict(num_seats=7, variant="plo6_double_bomb"),   # exactly 52 cards
        dict(num_seats=8, variant="plo4_double_bomb"),
        dict(num_seats=8, variant="nlh_single", sb=5_000),
        dict(num_seats=2),
    ],
)
def test_constructors_accept_the_largest_legal_tables(kw) -> None:
    gs = GameState(**kw)
    gs.reset(1, 0)
    assert gs.num_seats() == kw["num_seats"]
    be = _be(2, **kw)
    assert np.asarray(be.observation_and_features_batch()["stacks"]).shape == (2, kw["num_seats"])


def test_reconfigure_rejects_zero_bb() -> None:
    be = _be(2, num_seats=3)
    stacks = np.array([200_000] * 3, dtype=np.uint64)
    with pytest.raises(ValueError):
        be.reconfigure(stacks, 30_000, 0)
    with pytest.raises(ValueError):
        be.reconfigure(stacks[:2], 30_000, BB)  # wrong seat count
    be.reconfigure(stacks, 30_000, BB)


# ---------------------------------------------------------------------------
# C4 — subset entry points bounds-check their indices
# ---------------------------------------------------------------------------

_SUBSET_METHODS = (
    "observation_and_features_subset_batch",
    "observation_encoded_subset_batch",
    "observation_encoded_minimal_subset_batch",
    "all_hole_cards_subset_batch",
)


@pytest.mark.parametrize("method", _SUBSET_METHODS)
@pytest.mark.parametrize("bad", [[-1], [4], [99], [0, 1, -3], [2**40]])
def test_subset_entry_points_reject_out_of_range_indices(method: str, bad) -> None:
    be = _be(4, opp_outcome_mc=8)
    with pytest.raises(ValueError, match="out of range"):
        getattr(be, method)(np.asarray(bad, dtype=np.int64))
    # The engine is untouched and still serves valid requests.
    ok = getattr(be, method)(np.asarray([3, 0], dtype=np.int64))
    rows = ok if method == "all_hole_cards_subset_batch" else ok["actor"]
    assert np.asarray(rows).shape[0] == 2


# ---------------------------------------------------------------------------
# C4 — reset_nlh_cfr_node validates its node
# ---------------------------------------------------------------------------


def _nlh(num_seats: int = 2) -> GameState:
    return GameState(num_seats=num_seats, starting_stack=1_000_000, ante=0, bb=BB,
                     variant="nlh_single", sb=5_000)


_GOOD_NODE = dict(pot=100_000, stacks=[500_000, 500_000], board=[0, 5, 10], street=1,
                  hero_seat=0, hero_hole=[3, 4], path=[])


@pytest.mark.parametrize(
    "override",
    [
        dict(board=[60, 1, 2]),                      # card index out of range
        dict(board=[1, 1, 2]),                       # duplicate board card
        dict(hero_hole=[3, 3]),
        dict(hero_hole=[0, 4]),                      # hero hole collides with the board
        dict(hero_hole=[3, 52]),
        dict(stacks=[500_000]),                      # "need at least 2 seats" panic
        dict(stacks=[]),
        dict(stacks=[500_000] * 30),                 # deck overrun
        dict(stacks=[500_000] * 4),                  # != the wrapper's 2 seats
        dict(bb=0),
        dict(hero_seat=5),
        dict(street=0),
        dict(board=[0, 5]),                          # too short for the flop
    ],
)
def test_reset_nlh_cfr_node_rejects_bad_nodes(override) -> None:
    gs = _nlh()
    with pytest.raises(ValueError):
        gs.reset_nlh_cfr_node(**{**_GOOD_NODE, **override})


def test_reset_nlh_cfr_node_keeps_seat_count_consistent() -> None:
    # A 2-seat node on a 6-seat wrapper used to leave num_seats() == 6 over a
    # 2-seat state: hero_category(4, 0) and pack_range_nlh then panicked.
    with pytest.raises(ValueError, match="num_seats"):
        _nlh(6).reset_nlh_cfr_node(**_GOOD_NODE)

    gs = _nlh(2)
    gs.reset_nlh_cfr_node(**_GOOD_NODE)
    raw = gs.observation_dict()
    assert gs.num_seats() == len(raw["stacks"]) == 2
    with pytest.raises(ValueError):
        gs.hero_category(4, 0)
    pack = gs.pack_range_nlh(np.array([[20, 21], [30, 35]], dtype=np.uint8))
    assert np.asarray(pack["stacks"]).shape == (2, 2)


def test_serial_hero_category_rejects_a_third_board() -> None:
    gs = GameState()
    gs.reset(1, 0)
    assert 0 <= gs.hero_category(0, 0) <= 8 and 0 <= gs.hero_category(0, 1) <= 8
    for board in (2, 7, 255):
        with pytest.raises(ValueError, match="0 or 1"):  # was read as board B silently
            gs.hero_category(0, board)


# ---------------------------------------------------------------------------
# C4 — feature pyfunctions: F-ordered and empty inputs
# ---------------------------------------------------------------------------


def _deal(n: int, hole_w: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    hole = np.full((n, hole_w), 255, dtype=np.uint8)
    ba = np.full((n, 5), 255, dtype=np.uint8)
    bb = np.full((n, 5), 255, dtype=np.uint8)
    for i in range(n):
        deck = rng.permutation(52).astype(np.uint8)
        blen = int(rng.choice([0, 3, 4, 5]))
        hole[i] = deck[:hole_w]
        ba[i, :blen] = deck[hole_w : hole_w + blen]
        bb[i, :blen] = deck[hole_w + 5 : hole_w + 5 + blen]
    vc = np.zeros((n, 13, 4), dtype=np.int8)
    for src in (hole, ba, bb):
        ei, si = np.nonzero(src < 52)
        vc[ei, src[ei, si] >> 2, src[ei, si] & 3] = 1
    return hole, ba, bb, vc


def _rank_masks(hole, ba, bb):
    def mask(src):
        m = np.zeros((src.shape[0], 13), dtype=bool)
        ei, si = np.nonzero(src < 52)
        m[ei, src[ei, si] >> 2] = True
        return m

    return mask(hole), mask(ba), mask(bb), (ba < 52).any(axis=1) & (bb < 52).any(axis=1)


@pytest.mark.parametrize("hole_w", [4, 5, 6])
def test_feature_pyfunctions_accept_f_ordered_arrays(hole_w: int) -> None:
    """`np.asfortranarray` inputs used to panic: `to_owned()` kept the
    F-layout, whose rows are not contiguous slices."""
    hole, ba, bb, vc = _deal(64, hole_w)
    f = np.asfortranarray
    assert not f(hole).flags["C_CONTIGUOUS"]

    def same(a, b) -> None:
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert np.array_equal(np.asarray(x), np.asarray(y))

    same(draw_flags_batch(f(hole), f(ba), f(bb)), draw_flags_batch(hole, ba, bb))
    same(pair_features_batch(f(hole), f(ba), f(bb)), pair_features_batch(hole, ba, bb))
    same(
        straight_flush_features_batch(f(hole), f(ba), f(bb), f(vc)),
        straight_flush_features_batch(hole, ba, bb, vc),
    )
    hm, am, bm, valid = _rank_masks(hole, ba, bb)
    same(
        cross_board_straight_batch(f(hm), f(am), f(bm), valid),
        cross_board_straight_batch(hm, am, bm, valid),
    )
    # A strided (non-contiguous) view is the same hazard.
    same(
        draw_flags_batch(hole[::2], ba[::2], bb[::2]),
        draw_flags_batch(hole[::2].copy(), ba[::2].copy(), bb[::2].copy()),
    )


@pytest.mark.parametrize("hole_w", [4, 5, 6])
def test_feature_pyfunctions_accept_empty_batches(hole_w: int) -> None:
    """An empty PLO4 / PLO6 batch is (0, 4) / (0, 6); the width used to be
    guessed as 5 whenever N == 0 and the call was refused."""
    hole = np.zeros((0, hole_w), dtype=np.uint8)
    board = np.zeros((0, 5), dtype=np.uint8)
    vc = np.zeros((0, 13, 4), dtype=np.int8)
    assert [np.asarray(x).shape for x in draw_flags_batch(hole, board, board)] == [(0,)] * 4
    assert [np.asarray(x).shape for x in pair_features_batch(hole, board, board)] == [
        (0, 5), (0, 4), (0, 5), (0, 4)
    ]
    assert [
        np.asarray(x).shape for x in straight_flush_features_batch(hole, board, board, vc)
    ] == [(0, 38)] * 2


def test_feature_pyfunctions_still_reject_bad_shapes() -> None:
    board = np.zeros((2, 5), dtype=np.uint8)
    for bad_hole in (np.zeros((2, 3), dtype=np.uint8), np.zeros((2, 7), dtype=np.uint8),
                     np.zeros((0, 0), dtype=np.uint8)):
        with pytest.raises(ValueError):
            draw_flags_batch(bad_hole, board[: bad_hole.shape[0]], board[: bad_hole.shape[0]])
    with pytest.raises(ValueError):
        pair_features_batch(np.zeros((2, 5), dtype=np.uint8), board[:1], board)


# ---------------------------------------------------------------------------
# C4 — observation_arrays() carries the same keys as the other dict builders
# ---------------------------------------------------------------------------


def test_observation_arrays_matches_the_features_bundle_keys() -> None:
    be = _be(3, opp_outcome_mc=8)
    arrays = dict(be.observation_arrays())
    bundle = dict(be.observation_and_features_batch())
    assert set(bundle) - set(arrays) == {"legal_mask", "hero_cat_a", "hero_cat_b"}
    assert set(arrays) <= set(bundle)
    for key in arrays:
        assert np.array_equal(np.asarray(arrays[key]), np.asarray(bundle[key])), key


def test_observation_arrays_feeds_the_nlh_batch_encoder() -> None:
    kw = dict(num_seats=3, starting_stack=1_000_000, ante=5_000, bb=BB)
    be = _be(4, variant="nlh_single", sb=5_000, **kw)
    cfg = GameConfig(variant=VARIANT_NLH, sb=5_000, **kw)
    bundle = dict(be.observation_and_features_batch())
    cat = np.asarray(bundle["hero_cat_a"])
    arrays = dict(be.observation_arrays())  # KeyError on sb_seat before the fix
    assert (np.asarray(arrays["sb_seat"]) >= 0).all() and (np.asarray(arrays["bb_seat"]) >= 0).all()
    assert np.array_equal(
        encode_observation_batch_nlh(arrays, cat, cfg),
        encode_observation_batch_nlh(bundle, cat, cfg),
    )


# ---------------------------------------------------------------------------
# C7 — serial payouts()/payouts_ev() on a live hand
# ---------------------------------------------------------------------------


def test_serial_payouts_are_zero_until_terminal() -> None:
    """On a live hand the engine's `payouts` evaluates the PRE-DEALT full
    boards — a showdown result (undealt turn/river + villain holes) leaking
    through a getter. `payouts_batch` already zeroed non-terminal envs."""
    gs = GameState()
    be = _be(1)
    gs.reset(0, 0)
    steps = 0
    while not gs.is_terminal():
        assert gs.payouts() == [0] * 6
        assert gs.payouts_ev(8, 1) == [0] * 6
        assert not np.asarray(be.payouts_batch()).any()
        gs.apply_action(1)  # check / call down
        be.apply_action_batch(np.array([1], dtype=np.uint8))
        steps += 1
    assert steps >= 6
    payouts = gs.payouts()
    assert sum(payouts) == 0 and any(payouts)
    assert payouts == np.asarray(be.payouts_batch())[0].tolist()
    assert gs.payouts_ev(8, 1) == np.asarray(
        be.payouts_ev_batch(8, np.array([1], dtype=np.uint64))
    )[0].tolist()


# ---------------------------------------------------------------------------
# C6 — the opp-outcome memo is slotted per (env, seat) and actually hits
# ---------------------------------------------------------------------------


def _lockstep(num_seats: int, n: int, mc: int, seed: int):
    rng = np.random.default_rng(seed)
    be = BatchedEngine(n, num_seats=num_seats, opp_outcome_mc=mc)
    serial = [GameState(num_seats=num_seats) for _ in range(n)]
    return rng, be, serial


def _deal_all(rng, be, serial, mask=None):
    n = len(serial)
    ns = serial[0].num_seats()
    seeds = rng.integers(0, 2**62, size=n).astype(np.uint64)
    buttons = rng.integers(0, ns, size=n).astype(np.uint8)
    if mask is None:
        be.reset_batch(seeds, buttons)
        mask = np.ones(n, dtype=bool)
    else:
        be.reset_terminal_batch(seeds, buttons, mask)
    for i in np.nonzero(mask)[0]:
        serial[i].reset(int(seeds[i]), int(buttons[i]))


def _assert_outcome_blocks_fresh(be, serial, mc: int) -> None:
    """Whatever the memo served must equal a fresh serial MC, bit for bit."""
    bundle = be.observation_and_features_batch()
    packed = np.concatenate(
        [np.asarray(bundle[k]) for k in ("opp_outcome_fractions", "per_board_outcome", "share_bounds")],
        axis=1,
    )
    for i, gs in enumerate(serial):
        fresh = np.asarray(gs.outcome_features_mc(mc), dtype=np.float32)
        assert np.array_equal(packed[i], fresh), f"env {i}: memo != fresh MC"
    return bundle


@pytest.mark.parametrize("num_seats", [2, 6])
def test_outcome_memo_hits_and_stays_bit_exact(num_seats: int) -> None:
    mc, n = 32, 16
    rng, be, serial = _lockstep(num_seats, n, mc, seed=num_seats)
    assert be.outcome_cache_stats() == (0, 0)
    _deal_all(rng, be, serial)
    for _ in range(120):
        bundle = _assert_outcome_blocks_fresh(be, serial, mc)
        legal = np.asarray(bundle["legal_mask"])
        actions = np.ones(n, dtype=np.uint8)
        for i in range(n):
            if not serial[i].is_terminal():
                # Raise-heavy so seats get re-opened and act twice on a street.
                options = np.nonzero(legal[i])[0]
                weights = np.where(options >= 2, 3.0, 1.0)
                actions[i] = int(rng.choice(options, p=weights / weights.sum()))
        done = np.asarray(be.apply_action_batch(actions), dtype=bool)
        for i in range(n):
            if not serial[i].is_terminal():
                serial[i].apply_action(int(actions[i]))
        if done.any():
            _deal_all(rng, be, serial, mask=done)

    lookups, hits = be.outcome_cache_stats()
    assert lookups > 1000
    # One slot per ENV (the old layout) measured 0 hits in 51,200 lookups: the
    # actor — hence the key — changes on every action.
    assert hits > 0.05 * lookups, (lookups, hits)


def test_outcome_memo_is_cleared_when_an_env_is_redealt() -> None:
    be = BatchedEngine(4, num_seats=3, opp_outcome_mc=16)
    seeds = np.arange(4, dtype=np.uint64) + 10
    buttons = np.zeros(4, dtype=np.uint8)
    be.reset_batch(seeds, buttons)
    be.observation_and_features_batch()
    be.observation_and_features_batch()
    assert be.outcome_cache_stats() == (8, 4)  # the repeat pack is all hits

    # Re-dealing the SAME hands yields the same keys; only a cleared memo
    # misses on them.
    be.reset_batch(seeds, buttons)
    be.observation_and_features_batch()
    assert be.outcome_cache_stats() == (12, 4)

    mask = np.array([True, False, False, True])
    be.reset_terminal_batch(seeds, buttons, mask)
    be.observation_and_features_batch()
    assert be.outcome_cache_stats() == (16, 6)  # envs 1, 2 kept their entries

    be.reconfigure(np.array([200_000] * 3, dtype=np.uint64), 30_000, BB)
    be.reset_batch(seeds, buttons)
    be.observation_and_features_batch()
    assert be.outcome_cache_stats() == (20, 6)


def test_outcome_memo_subset_pack_shares_the_slots() -> None:
    be = _be(6, num_seats=4, opp_outcome_mc=16)
    full = be.observation_and_features_batch()
    lookups, hits = be.outcome_cache_stats()
    idx = np.array([4, 1], dtype=np.int64)
    sub = be.observation_and_features_subset_batch(idx)
    assert be.outcome_cache_stats() == (lookups + 2, hits + 2)
    for key in ("opp_outcome_fractions", "per_board_outcome", "share_bounds"):
        assert np.array_equal(np.asarray(sub[key]), np.asarray(full[key])[idx])
