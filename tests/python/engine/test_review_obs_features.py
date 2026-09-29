"""Regression tests for the observation-feature findings of the 2026-09-20
code review (section B + the STK-6 latent hazard).

Every finding here was a case where all three encoders (scalar Python, numpy
batch, fused Rust) AGREED with each other, so the parity suite stayed green
while the value itself was wrong. These tests therefore pin SEMANTICS against
an independent oracle (the sizing head's own anchor grid, the env's gate
mask, a from-scratch reference), and add the heterogeneous-stack three-way
parity sweep the uniform-stack parity grids never exercised.

  B1  STK-2 / STK-5[2:4] describe the LEGAL raise window   (dims 1024-1029, 1040-1041)
  B2  straight-draw flag: ace-low shadow below the deuce   (dims 800, 802)
  B3  min/max scalars = legal window; dead-chip invariance (PLO 186/187, NLH 134/135)
  B5  flush blockers skip cards face-up on either board    (dims 999/1000, 1003/1004)
  B6  STK-10 pot_at_flop sums antes over dealt-in seats    (dim 1052)
  B7  NLH flush nut distance on a five-flush board         (NLH dim 906)
  STK-6 exact power-of-3 comparison chain                  (dim 1042)

Everything here asserts the FIXED semantics (observation-semantics revision 2,
the default). The legacy revision (`PLO5BP_OBS_REV=1`, the pre-fix values kept
for old checkpoints) is covered by test_review_obs_rev.py; the autouse fixture
below pins rev 2 so this file passes whichever revision the suite runs under.

Tests that touch the Rust side (`draw_flags_batch`, the fused
`observation_encoded_batch`) need an `_engine` binary rebuilt from this tree.
"""

from __future__ import annotations

import numpy as np
import pytest

from plo5bp import encoding as E
from plo5bp import encoding_nlh as EN
from plo5bp._engine import (  # type: ignore[attr-defined]
    BatchedEngine,
    GameState,
    draw_flags_batch,
)
from plo5bp.actions import (
    ALL_IN,
    CHECK_CALL,
    FOLD,
    GATE_CHECK_CALL,
    GATE_FOLD,
    GATE_RAISE,
    gate_mask_from_bounds,
)
from plo5bp.config import VARIANT_NLH, VARIANT_PLO4, VARIANT_PLO6, GameConfig
from plo5bp.env import BombPotEnv
from plo5bp.sizing import ANCHOR_COUNT, anchor_grid_np, sizing_from_info

BB = 10_000


def pin_obs_rev(monkeypatch: pytest.MonkeyPatch, rev: int) -> None:
    """Run the rest of the test under observation-semantics revision `rev`.

    The Python encoders branch on the module constant (read once at import);
    the Rust engines read the env var when they are CONSTRUCTED — so set both,
    before building any env / engine."""
    monkeypatch.setenv(E.OBS_REV_ENV, str(rev))
    monkeypatch.setattr(E, "OBS_SEMANTICS_REV", rev)
    monkeypatch.setattr(EN, "OBS_SEMANTICS_REV", rev)


@pytest.fixture(autouse=True)
def _rev2(monkeypatch: pytest.MonkeyPatch) -> None:
    pin_obs_rev(monkeypatch, E.OBS_REV_CURRENT)


def _card(rank: int, suit: int) -> int:
    """rank 0 = deuce .. 12 = ace; suit 0..3."""
    return rank * 4 + suit


def _hetero_cfg(rng: np.random.Generator, lo: int = 4, hi: int = 120, **kw) -> GameConfig:
    ns = int(rng.integers(2, 7))
    stacks = tuple(int(x) * BB for x in rng.integers(lo, hi, size=ns))
    return GameConfig(num_seats=ns, starting_stacks=stacks, ante=30_000, bb=BB, **kw)


def _random_gate_step(env: BombPotEnv, info, rng: np.random.Generator):
    """One random legal hybrid action (continuous raise sizes, so short-shove
    and cover-short regimes show up)."""
    legal = np.nonzero(info.gate_mask)[0]
    p = np.array(
        [0.1 if g == GATE_FOLD else (0.5 if g == GATE_CHECK_CALL else 0.4) for g in legal]
    )
    gate = int(rng.choice(legal, p=p / p.sum()))
    chips = 0
    if gate == GATE_RAISE:
        mn, mx = int(info.min_raise_chips), int(info.max_raise_chips)
        chips = mx if mn == 0 else int(rng.integers(mn, mx + 1))
    return env.step_hybrid(gate, chips)


def _nodes(env: BombPotEnv, rng: np.random.Generator, **reset_kw):
    """Yield (obs, info) at every decision node of one random hand."""
    cfg = env.config
    obs, info = env.reset(
        int(rng.integers(0, 2**62)), int(rng.integers(0, cfg.num_seats)), **reset_kw
    )
    while info.actor is not None:  # None: all-in from the antes / hand over
        yield obs, info
        obs, _, _, info = _random_gate_step(env, info, rng)


# ---------------------------------------------------------------------------
# B1 — STK-2 / STK-5[2:4] describe the legal raise window
# ---------------------------------------------------------------------------


def test_b1_stk2_matches_sizing_head_on_heterogeneous_stacks() -> None:
    """On random heterogeneous-stack nodes the raise-ladder block must agree
    with what the SIZING HEAD will actually be offered: the anchor count from
    `anchor_grid_np(*sizing_from_info(info))`, an independent recompute of the
    envelope from the engine's min/max raise, all-zero whenever the env's gate
    mask has Raise illegal, and STK-5[2] never negative."""
    rng = np.random.default_rng(5)
    seen = {"legal": 0, "illegal": 0, "short_shove": 0, "dust": 0, "stack_capped": 0}
    for _ in range(90):
        cfg = _hetero_cfg(rng)
        for obs, info in _nodes(BombPotEnv(cfg), rng):
            raw = info.raw_obs
            hero = info.actor
            stk2 = obs[E._STK2_OFF : E._STK2_OFF + 6]
            stk5 = obs[E._STK5_OFF : E._STK5_OFF + 4]
            mn, mx = int(info.min_raise_chips), int(info.max_raise_chips)
            assert stk5[2] >= 0.0, f"STK-5[2] negative: {stk5[2]}"

            if not info.gate_mask[GATE_RAISE]:
                seen["illegal"] += 1
                seen["dust"] += int(mx > 0)  # engine window, screened by the gate
                assert not stk2.any(), f"STK-2 non-zero with Raise illegal: {stk2}"
                assert stk5[2] == 0.0 and stk5[3] == 0.0
            else:
                seen["legal"] += 1
                seen["short_shove"] += int(mn == 0)
                sizing = sizing_from_info(info)
                n_legal = int(anchor_grid_np(*sizing).legal.sum())
                assert stk2[5] == np.float32(n_legal / ANCHOR_COUNT)
                assert round(float(stk2[5]) * ANCHOR_COUNT) == n_legal

                pot = float(raw["pot"])
                to_call = float(sizing[3])
                starting = cfg.resolved_stacks[hero]
                dead = max(0, int(starting) - int(raw["eff_stack_cap"][hero]))
                eff = max(0.0, float(raw["stacks"][hero]) - dead)
                min_d = float(mn if mn > 0 else mx)
                max_d = float(mx)
                base = pot + to_call
                expect = [
                    min(max((min_d - to_call) / max(base, 1.0), 0.0), 1.0),
                    min(max((max_d - to_call) / max(base, 1.0), 0.0), 1.0),
                    min(max(min_d / max(eff, 1.0), 0.0), 1.0),
                    min(max(max_d / max(eff, 1.0), 0.0), 1.0),
                    1.0 if max_d < to_call + base else 0.0,
                ]
                assert list(stk2[:5]) == [np.float32(x) for x in expect]
                seen["stack_capped"] += int(expect[4] == 1.0)
                # The max raise never exceeds what hero can put in.
                assert max_d <= float(raw["stacks"][hero])
                tot2 = pot + 2.0 * max_d - to_call
                assert stk5[2] == np.float32(np.log1p((eff - max_d) / max(tot2, 1.0)))
                assert stk5[3] == np.float32(np.log1p(tot2 / BB))
    # The sweep must actually reach every regime the fix is about.
    assert all(v > 0 for v in seen.values()), seen


def test_b1_short_hero_in_bloated_pot() -> None:
    """The review's example: hero 7bb behind in an 18bb bomb pot. The old
    block said "max raise = pot, not stack-capped, 11/11 anchors"."""
    stacks = (100_000, 500_000, 500_000, 500_000, 500_000, 500_000)
    cfg = GameConfig(num_seats=6, starting_stacks=stacks, ante=30_000, bb=BB)
    env = BombPotEnv(cfg)
    for button in range(6):
        obs, info = env.reset(123, button)
        if info.actor == 0:
            break
    assert info.actor == 0
    assert (info.min_raise_chips, info.max_raise_chips) == (10_000, 70_000)
    stk2 = obs[E._STK2_OFF : E._STK2_OFF + 6]
    assert stk2[1] == np.float32(70_000 / 180_000)  # max raise as a pot fraction
    assert stk2[3] == 1.0 and stk2[4] == 1.0        # = hero's whole stack, capped
    assert stk2[5] == np.float32(5 / 11)
    assert obs[E._STK5_OFF + 2] == 0.0              # nothing behind after the max raise

    # Facing a pot bet hero cannot raise at all: the whole block is zero.
    for button in range(6):
        obs, info = env.reset(123, button)
        if info.actor == 1:
            break
    obs, _, _, info = env.step_hybrid(GATE_RAISE, int(info.max_raise_chips))
    while info.actor != 0:
        obs, _, _, info = env.step_hybrid(GATE_FOLD)
    assert not info.gate_mask[GATE_RAISE]
    assert not obs[E._STK2_OFF : E._STK2_OFF + 6].any()
    assert obs[E._STK5_OFF + 2] == 0.0 and obs[E._STK5_OFF + 3] == 0.0


def test_legal_raise_window_is_the_gate_predicate() -> None:
    """`_legal_raise_window` must agree with `gate_mask_from_bounds` (the env's
    Raise gate) on every node — including the cover-short DUST screen, which
    the encoder re-derives from `max_raise == stack` instead of legal[ALL_IN] —
    and its batch twin must agree with it."""
    rng = np.random.default_rng(17)
    checked = 0
    for _ in range(60):
        cfg = _hetero_cfg(rng, lo=1, hi=40)
        for _obs, info in _nodes(BombPotEnv(cfg), rng):
            raw = info.raw_obs
            hero = info.actor
            mn, mx = int(raw["min_raise"]), int(raw["max_raise"])
            stack = int(raw["stacks"][hero])
            legal, min_d, max_d, anchor_min = E._legal_raise_window(mn, mx, stack, BB)
            assert legal == bool(info.gate_mask[GATE_RAISE])
            assert anchor_min == float(mn)  # RAW min_raise: 0 flags a short shove
            if mx > 0:  # the identity the dust screen rests on
                assert bool(info.legal_mask[ALL_IN]) == (mx == stack)
            if legal:
                assert (min_d, max_d) == (float(mn if mn > 0 else mx), float(mx))
            else:
                assert (min_d, max_d) == (0.0, 0.0)
            batch = E._legal_raise_window_batch(
                np.array([mn], dtype=np.uint64),
                np.array([mx], dtype=np.uint64),
                np.array([stack], dtype=np.uint64),
                BB,
            )
            assert (
                bool(batch.legal[0]), float(batch.min_d[0]), float(batch.max_d[0]),
                float(batch.anchor_min[0]),
            ) == (legal, min_d, max_d, anchor_min)
            checked += 1
    assert checked > 300


# ---------------------------------------------------------------------------
# B3 — min/max scalars are the legal window; dead-chip invariance at every node
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["plo5", "nlh"])
def test_b3_scalars_are_the_legal_window_as_totals(variant: str) -> None:
    rng = np.random.default_rng(23)
    off = E._SCALARS_OFF if variant == "plo5" else EN._SCALARS_OFF
    legal_nodes = illegal_nodes = 0
    for _ in range(40):
        if variant == "plo5":
            cfg = _hetero_cfg(rng)
        else:
            ns = int(rng.integers(2, 7))
            stacks = tuple(int(x) * BB for x in rng.integers(20, 251, size=ns))
            cfg = GameConfig(
                num_seats=ns, starting_stacks=stacks, ante=5_000, bb=BB, sb=5_000,
                variant=VARIANT_NLH,
            )
        for obs, info in _nodes(BombPotEnv(cfg), rng):
            raw = info.raw_obs
            hero = info.actor
            sc = float(raw["street_commit"][hero])
            mn, mx = int(info.min_raise_chips), int(info.max_raise_chips)
            if info.gate_mask[GATE_RAISE]:
                legal_nodes += 1
                assert obs[off + 2] == np.float32((sc + (mn if mn > 0 else mx)) / BB)
                assert obs[off + 3] == np.float32((sc + mx) / BB)
                # A legal total never exceeds what hero can commit this street.
                assert obs[off + 3] <= np.float32((sc + float(raw["stacks"][hero])) / BB)
            else:
                illegal_nodes += 1
                assert obs[off + 2] == 0.0 and obs[off + 3] == 0.0
    assert legal_nodes > 100 and illegal_nodes > 10


def _assert_invariant_to_dead_chips(mk_cfg, deep_a: int, deep_b: int, base_range, trials, seed):
    """Same seed + same actions, the deepest seat's stack changed ABOVE every
    other seat's reach => the observation is identical at EVERY node."""
    rng = np.random.default_rng(seed)
    nodes = 0
    for _ in range(trials):
        ns = int(rng.integers(2, 6))
        base = [int(x) * BB for x in rng.integers(*base_range, size=ns)]
        deep = int(rng.integers(0, ns))
        stacks_a, stacks_b = list(base), list(base)
        stacks_a[deep], stacks_b[deep] = deep_a * BB, deep_b * BB
        env_a, env_b = BombPotEnv(mk_cfg(stacks_a)), BombPotEnv(mk_cfg(stacks_b))
        hand_seed, button = int(rng.integers(0, 2**62)), int(rng.integers(0, ns))
        obs_a, info_a = env_a.reset(hand_seed, button)
        obs_b, info_b = env_b.reset(hand_seed, button)
        done = info_a.actor is None
        while not done:
            # Premise of the contract: the legal bounds are identical.
            assert info_a.actor == info_b.actor
            assert info_a.min_raise_chips == info_b.min_raise_chips
            assert info_a.max_raise_chips == info_b.max_raise_chips
            assert np.array_equal(info_a.gate_mask, info_b.gate_mask)
            if not np.array_equal(obs_a, obs_b):
                dims = np.nonzero(obs_a != obs_b)[0].tolist()
                raise AssertionError(
                    f"stacks {stacks_a} vs {stacks_b}: obs differs at dims {dims[:12]} "
                    f"(hero {info_a.actor})"
                )
            nodes += 1
            legal = np.nonzero(info_a.gate_mask)[0]
            gate = int(rng.choice(legal))
            chips = 0
            if gate == GATE_RAISE:
                mn, mx = int(info_a.min_raise_chips), int(info_a.max_raise_chips)
                chips = mx if mn == 0 else int(rng.integers(mn, mx + 1))
            obs_a, _, done, info_a = env_a.step_hybrid(gate, chips)
            obs_b, _, done_b, info_b = env_b.step_hybrid(gate, chips)
            assert done == done_b
    return nodes


def test_b3_dead_chip_invariance_every_node_plo() -> None:
    """Extends `test_encoding_invariant_to_unreachable_chips_above_eff_cap`
    (first HU flop node only) to every node of multiway hands. Before the fix
    `max_bet` alone broke this at ~25% of nodes (dims 187, 1025, 1028-1029,
    1040-1041)."""
    nodes = _assert_invariant_to_dead_chips(
        lambda s: GameConfig(num_seats=len(s), starting_stacks=tuple(s), ante=30_000, bb=BB),
        deep_a=40, deep_b=90, base_range=(6, 31), trials=60, seed=11,
    )
    assert nodes > 300


def test_b3_dead_chip_invariance_every_node_nlh() -> None:
    """NLH twin: before the fix dim 135 differed at ~50% of nodes."""
    nodes = _assert_invariant_to_dead_chips(
        lambda s: GameConfig(
            num_seats=len(s), starting_stacks=tuple(s), ante=5_000, bb=BB, sb=5_000,
            variant=VARIANT_NLH,
        ),
        deep_a=260, deep_b=400, base_range=(100, 251), trials=40, seed=3,
    )
    assert nodes > 300


# ---------------------------------------------------------------------------
# B2 — straight-draw flag
# ---------------------------------------------------------------------------

# A-2-3-4 up to J-Q-K-A: the ace plays BOTH ends, nothing wraps around it.
_FOUR_RUNS = [(12, 0, 1, 2)] + [tuple(range(s, s + 4)) for s in range(10)]


def _straight_draw_reference(hole: list[int], board: list[int]) -> float:
    if not board:
        return 0.0
    ranks = {c // 4 for c in hole} | {c // 4 for c in board}
    return float(any(all(r in ranks for r in run) for run in _FOUR_RUNS))


def _pad(cards: list[int], width: int) -> np.ndarray:
    out = np.full(width, 255, dtype=np.uint8)
    out[: len(cards)] = cards
    return out


def _straight_flags_all_paths(hole: list[int], board: list[int]) -> dict[str, float]:
    h = _pad(hole, len(hole))[None, :]
    b = _pad(board, 5)[None, :]
    empty = np.full((1, 5), 255, dtype=np.uint8)
    return {
        "scalar": E._draw_flags(hole, board)[1],
        "numpy_batch": float(E._draw_flags_batch(h, b)[1][0]),
        "rust": float(draw_flags_batch(h, b, empty)[1][0]),  # obs_rev defaults to 2
    }


@pytest.mark.parametrize(
    "name, hole, board, expected",
    [
        # Q-K-A is three cards at the TOP of the ladder — not a 4-run. The old
        # shadow bit above the ace made it one (false positive).
        ("QsKd2c7h7d on Ah3s8c",
         [_card(10, 3), _card(11, 1), _card(0, 0), _card(5, 2), _card(5, 1)],
         [_card(12, 2), _card(1, 3), _card(6, 0)], 0.0),
        # A-2-3-4 is the wheel draw the old code could never see.
        ("As2d9c9hKd on 3h4sJc",
         [_card(12, 3), _card(0, 1), _card(7, 0), _card(7, 2), _card(11, 1)],
         [_card(1, 2), _card(2, 3), _card(9, 0)], 1.0),
        # Ordinary 4-runs are unchanged.
        ("5s6dKcKh2d on 7h8sQc",
         [_card(3, 3), _card(4, 1), _card(11, 0), _card(11, 2), _card(0, 1)],
         [_card(5, 2), _card(6, 3), _card(10, 0)], 1.0),
        # Broadway end still fires with four of T-J-Q-K-A ...
        ("JsQd2c2h7d on KhAs8c",
         [_card(9, 3), _card(10, 1), _card(0, 0), _card(0, 2), _card(5, 1)],
         [_card(11, 2), _card(12, 3), _card(6, 0)], 1.0),
        # ... and K-A-2-3 must NOT wrap around the ace.
        ("KsAd9c9h6d on 2h3sJc",
         [_card(11, 3), _card(12, 1), _card(7, 0), _card(7, 2), _card(4, 1)],
         [_card(0, 2), _card(1, 3), _card(9, 0)], 0.0),
    ],
)
def test_b2_straight_draw_flag_cases(name, hole, board, expected) -> None:
    assert _straight_draw_reference(hole, board) == expected, "bad test case"
    flags = _straight_flags_all_paths(hole, board)
    assert flags == {"scalar": expected, "numpy_batch": expected, "rust": expected}, name


def test_b2_straight_draw_flag_matches_reference_fuzz() -> None:
    """All three implementations vs the from-scratch reference on random
    PLO4/5/6 deals, flop/turn/river and the empty board."""
    rng = np.random.default_rng(2)
    n = 1500
    for hole_w in (4, 5, 6):
        hole = np.full((n, hole_w), 255, dtype=np.uint8)
        board = np.full((n, 5), 255, dtype=np.uint8)
        expected = np.zeros(n, dtype=np.float32)
        for i in range(n):
            deck = rng.permutation(52)
            blen = int(rng.choice([0, 3, 4, 5]))
            hole[i] = deck[:hole_w]
            board[i, :blen] = deck[hole_w : hole_w + blen]
            h, b = [int(c) for c in hole[i]], [int(c) for c in board[i, :blen]]
            expected[i] = _straight_draw_reference(h, b)
            assert E._draw_flags(h, b)[1] == expected[i]
        assert 0.2 < expected.mean() < 0.95  # both outcomes well represented
        assert np.array_equal(E._draw_flags_batch(hole, board)[1], expected)
        _, s_a, _, s_b = draw_flags_batch(hole, board, board)
        assert np.array_equal(np.asarray(s_a), expected)
        assert np.array_equal(np.asarray(s_b), expected)


# ---------------------------------------------------------------------------
# B5 — flush blockers skip cards face-up on the other board
# ---------------------------------------------------------------------------


def test_b5_nut_flush_blocker_sees_the_other_board() -> None:
    # Board A: Ks 7s 2s (suit 3 = spades). The As is face-up on board B, so
    # nobody can hold it and hero's Qs IS the top holdable spade.
    board_a = [_card(11, 3), _card(5, 3), _card(0, 3)]
    board_b = [_card(12, 3), _card(8, 1), _card(3, 2)]
    hole = [_card(10, 3), _card(6, 0), _card(6, 1), _card(2, 2), _card(1, 0)]
    out = E._blocker_features(hole, board_a, board_b)
    assert out[0] == 1.0
    assert out[1] == np.float32(1.0 / 3.0)  # of Qs/Js/Ts hero holds the Qs
    # Without the other board the As still counts as "missing" (old behavior).
    assert E._blocker_features(hole, board_a)[0] == 0.0
    # Board B has no three-flush: its own flush dims stay silent either way.
    assert not E._blocker_features(hole, board_b, board_a)[:2].any()

    # End to end through the scalar encoder: board A block = dims 999..1003.
    obs = {
        "actor": 0, "hero_hole": hole, "board_a": board_a, "board_b": board_b,
        "street": 1, "folded": [False, False], "all_in": [False, False],
        "stacks": [200_000, 200_000], "eff_stack_cap": [230_000, 230_000],
        "pot": 60_000, "bet_to_call": 0, "min_bet": 10_000, "max_bet": 60_000,
        "min_raise": 10_000, "max_raise": 60_000, "history": [],
        "street_commit": [0, 0], "total_commit": [30_000, 30_000],
        "button": 0, "last_aggressor": -1, "hero_category_a": 0, "hero_category_b": 0,
    }
    vec = E.encode_observation(obs, GameConfig(num_seats=2, starting_stack=230_000))
    assert vec[E._BLOCKER_A_OFF] == 1.0
    assert vec[E._BLOCKER_A_OFF + 1] == np.float32(1.0 / 3.0)


def test_b5_blockers_scalar_batch_parity_with_other_board() -> None:
    rng = np.random.default_rng(7)
    n = 600
    hole = np.full((n, 5), 255, dtype=np.uint8)
    board = np.full((n, 5), 255, dtype=np.uint8)
    other = np.full((n, 5), 255, dtype=np.uint8)
    for i in range(n):
        # Suit-biased decks so three-flush boards (and an other-board card of
        # the same suit) are common rather than a 1-in-20 event.
        deck = rng.permutation(52)
        if i % 2:
            deck = np.concatenate([deck[deck % 4 == 0], deck[deck % 4 != 0]])
            deck[:20] = rng.permutation(deck[:20])
        blen = int(rng.choice([0, 3, 4, 5]))
        board[i, :blen] = deck[:blen]
        other[i, :blen] = deck[5 : 5 + blen]
        hole[i] = deck[10:15]
    batch = E._blocker_features_batch(
        hole, hole < 52, board, board < 52, other, other < 52
    )
    changed = 0
    for i in range(n):
        h = [int(c) for c in hole[i]]
        b = [int(c) for c in board[i] if c < 52]
        o = [int(c) for c in other[i] if c < 52]
        scalar = E._blocker_features(h, b, o)
        assert np.array_equal(batch[i], scalar), (i, h, b, o, batch[i], scalar)
        changed += int(not np.array_equal(scalar, E._blocker_features(h, b)))
    assert changed > 10  # the other board really does move the flush dims


# ---------------------------------------------------------------------------
# B6 — STK-10 pot_at_flop sums antes over the dealt-in seats only
# ---------------------------------------------------------------------------


def test_b6_pot_at_flop_ignores_sitting_out_seats() -> None:
    cfg = GameConfig(num_seats=6, starting_stack=200_000, ante=30_000, bb=BB)
    env = BombPotEnv(cfg)
    obs, info = env.reset(5, 0, in_hand_mask=[True, True, False, True, False, False])
    assert info.raw_obs["pot"] == 90_000  # three antes, not six
    assert obs[E._STK10_OFF + 1] == 0.0
    bet = int(info.max_raise_chips)
    obs, _, _, info = env.step_hybrid(GATE_RAISE, bet)
    # bloat = chips added on top of the antes actually posted: bet / 3 antes.
    # (Six antes would give max(90k + bet - 180k, 0) / 180k = 0.)
    assert obs[E._STK10_OFF + 1] == np.float32(np.log1p(bet / 90_000.0))

    # A fully dealt-in table is unchanged (what training always sees).
    obs, info = env.reset(5, 0)
    bet = int(info.min_raise_chips)
    obs, _, _, _ = env.step_hybrid(GATE_RAISE, bet)
    assert obs[E._STK10_OFF + 1] == np.float32(np.log1p(bet / 180_000.0))


def test_b6_serial_batch_helper_parity_with_sitting_out_seat() -> None:
    """The batched engine never deals masked hands, so pin the numpy twin of
    STK-10 directly on a synthetic row with a sitting-out seat."""
    cfg = GameConfig(num_seats=3, starting_stack=200_000, ante=30_000, bb=BB)
    folded = [False, False, True]
    total_commit = [45_000, 45_000, 0]  # seat 2 never posted: sitting out
    kw = dict(pot=90_000.0, to_call=0.0, inv_bb=1.0 / BB)
    out_s = np.zeros(E.OBS_DIM, dtype=np.float32)
    E._encode_stack_v3(
        out_s, config=cfg, hero=0, num_seats=3, folded=folded,
        all_in=[False] * 3, eff_per_seat=[155_000.0, 155_000.0, 200_000.0],
        total_commit=total_commit, street_commit=[15_000, 15_000, 0],
        btc=15_000.0, window=E._RaiseWindow(True, 15_000.0, 105_000.0, 15_000.0),
        eff_to_call=0.0, hero_stack=155_000.0, street_idx=1, **kw,
    )
    out_b = np.zeros((1, E.OBS_DIM), dtype=np.float32)
    E._encode_stack_v3_batch(
        out_b, config=cfg, num_seats=3, live_mask=np.array([True]),
        hero_idx=np.array([0], dtype=np.int64),
        folded=np.array([folded]), all_in=np.zeros((1, 3), dtype=bool),
        effective=np.array([[155_000.0, 155_000.0, 200_000.0]]),
        total_commit=np.array([total_commit], dtype=np.float64),
        street_commit=np.array([[15_000.0, 15_000.0, 0.0]]),
        pot=np.array([90_000.0]), bet_to_call=np.array([15_000.0]),
        window=E._RaiseWindow(
            np.array([True]), np.array([15_000.0]), np.array([105_000.0]),
            np.array([15_000]),
        ),
        to_call=np.array([0.0]),
        inv_bb=1.0 / BB, street=np.array([1], dtype=np.int64),
    )
    assert np.array_equal(out_s[1020:1061], out_b[0, 1020:1061])
    # 90k pot over the TWO antes posted (60k): bloat = 30k / 60k.
    assert out_s[E._STK10_OFF + 1] == np.float32(np.log1p(30_000.0 / 60_000.0))


# ---------------------------------------------------------------------------
# B7 — NLH flush nut distance on a five-flush board
# ---------------------------------------------------------------------------


def _nlh_sf(hole: list[int], board: list[int]) -> tuple[float, float]:
    vc = np.zeros((13, 4), dtype=np.int32)
    for c in hole + board:
        vc[c // 4, c % 4] = 1
    scalar = float(EN._sf_features_nlh(hole, board, vc)[0])
    batch = float(
        EN._sf_features_nlh_batch(
            np.array([hole], dtype=np.uint8), _pad(board, 5)[None, :], vc[None]
        )[0, 0]
    )
    return scalar, batch


def test_b7_five_flush_board_cards_below_the_board_only_tie() -> None:
    # Board As Ks 8s 6s 5s. Everyone plays at least the board, so only an
    # unseen spade ABOVE the 5s beats it: Qs Js Ts 9s 7s = 5.
    board = [_card(12, 3), _card(11, 3), _card(6, 3), _card(4, 3), _card(3, 3)]
    # Hero's 2s does not play; the 3s/4s below the board's 5s would only TIE.
    # The old code started counting from hero's deuce and reported 7.
    assert _nlh_sf([_card(0, 3), _card(9, 1)], board) == (5.0, 5.0)
    # No spade at all: same board-flush answer (unchanged).
    assert _nlh_sf([_card(9, 1), _card(9, 2)], board) == (5.0, 5.0)
    # A spade that PLAYS (9s > 5s) counts from itself: Qs Js Ts = 3 (unchanged).
    assert _nlh_sf([_card(7, 3), _card(9, 1)], board) == (3.0, 3.0)
    # Four-flush board: hero's card always plays, no floor (unchanged).
    four = [_card(12, 3), _card(11, 3), _card(6, 3), _card(4, 3), _card(3, 1)]
    assert _nlh_sf([_card(0, 3), _card(9, 1)], four) == (8.0, 8.0)


def test_b7_scalar_batch_parity_on_flush_boards() -> None:
    rng = np.random.default_rng(9)
    five = 0
    for _ in range(400):
        suit = int(rng.integers(0, 4))
        n_suited = int(rng.choice([3, 4, 5]))
        suited = [int(r) * 4 + suit for r in rng.permutation(13)]
        rest = [int(c) for c in rng.permutation(52) if c % 4 != suit]
        board = suited[:n_suited] + rest[: 5 - n_suited]
        hole = [suited[5], rest[6]] if rng.random() < 0.6 else rest[6:8]
        scalar, batch = _nlh_sf(hole, board)
        assert scalar == batch, (hole, board, scalar, batch)
        five += int(n_suited == 5)
    assert five > 50


# ---------------------------------------------------------------------------
# STK-6 — exact power-of-3 comparison chain
# ---------------------------------------------------------------------------


def test_stk6_bets_to_jam_chain_values() -> None:
    # x6 = 1 + 2·SPR sits exactly on 3^k at SPR 1, 4, 13, 40, 121.
    for k, x6 in enumerate((1.0, 3.0, 9.0, 27.0, 81.0, 243.0)):
        assert E._bets_to_jam(x6) == k
        assert E._bets_to_jam(np.nextafter(x6, np.inf)) == k + 1
        if k:
            assert E._bets_to_jam(np.nextafter(x6, 0.0)) == k
    assert E._bets_to_jam(729.0) == 6 and E._bets_to_jam(1e9) == 6
    # Away from the boundaries the chain IS clip(ceil(log3 x), 0, 6).
    xs = np.random.default_rng(0).uniform(1.0, 900.0, size=4000)
    ref = np.clip(np.ceil(np.log(xs) / np.log(3.0)), 0.0, 6.0)
    assert [E._bets_to_jam(float(x)) for x in xs] == ref.astype(int).tolist()


@pytest.mark.parametrize("spr, expected", [(1, 1), (4, 2), (13, 3), (40, 4), (121, 5)])
def test_stk6_integer_boundaries_all_three_encoders(spr: int, expected: int) -> None:
    """HU, 3bb antes -> 6bb flop pot; `spr` pots behind puts x6 exactly on
    3^expected. Scalar, numpy batch and fused Rust must all report it."""
    start = 30_000 + spr * 60_000
    cfg = GameConfig(num_seats=2, starting_stacks=(start, start), ante=30_000, bb=BB)
    obs, _ = BombPotEnv(cfg).reset(1, 0)
    assert obs[E._STK6_OFF] == float(expected)

    be = BatchedEngine(1, num_seats=2, ante=30_000, bb=BB,
                       starting_stacks=np.array([start, start], dtype=np.uint64))
    be.reset_batch(np.array([1], dtype=np.uint64), np.array([0], dtype=np.uint8))
    bundle = be.observation_and_features_batch()
    numpy_obs = E.encode_observation_batch(
        bundle, np.asarray(bundle["hero_cat_a"]), np.asarray(bundle["hero_cat_b"]), cfg
    )
    rust_obs = np.asarray(be.observation_encoded_batch()["obs"])
    assert numpy_obs[0, E._STK6_OFF] == float(expected)
    assert rust_obs[0, E._STK6_OFF] == float(expected)
    assert np.array_equal(numpy_obs[0], obs) and np.array_equal(rust_obs[0], obs)


# ---------------------------------------------------------------------------
# Three-way parity on heterogeneous stacks (what the uniform grids never hit)
# ---------------------------------------------------------------------------


def _three_way_parity(variant: str, trials: int, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    seen = {"nodes": 0, "short_shove": 0, "raise_illegal": 0, "straight_draw": 0}
    n = 6
    for _ in range(trials):
        ns = int(rng.integers(2, 7))
        stacks = [int(x) * BB for x in rng.integers(4, 120, size=ns)]
        cfg = GameConfig(
            num_seats=ns, starting_stacks=tuple(stacks), ante=30_000, bb=BB, variant=variant
        )
        kw = dict(num_seats=ns, ante=30_000, bb=BB, variant=variant,
                  starting_stacks=np.asarray(stacks, dtype=np.uint64))
        be = BatchedEngine(n, **kw)
        seeds = rng.integers(0, 2**62, size=n).astype(np.uint64)
        buttons = rng.integers(0, ns, size=n).astype(np.uint8)
        be.reset_batch(seeds, buttons)
        serial = []
        for i in range(n):
            gs = GameState(**kw)
            gs.reset(int(seeds[i]), int(buttons[i]))
            serial.append(gs)

        for _step in range(80):
            if be.is_terminal_batch().all():
                break
            bundle = be.observation_and_features_batch()
            cat_a, cat_b = np.asarray(bundle["hero_cat_a"]), np.asarray(bundle["hero_cat_b"])
            numpy_obs = E.encode_observation_batch(bundle, cat_a, cat_b, cfg)
            rust_obs = np.asarray(be.observation_encoded_batch()["obs"])
            mini_numpy = E.encode_observation_batch_minimal(bundle, cfg)
            mini_rust = np.asarray(be.observation_encoded_minimal_batch()["obs"])
            for i in range(n):
                if serial[i].is_terminal():
                    assert not numpy_obs[i].any() and not rust_obs[i].any()
                    continue
                raw = dict(serial[i].observation_dict())
                raw["hero_category_a"] = int(serial[i].hero_category(raw["actor"], 0))
                raw["hero_category_b"] = int(serial[i].hero_category(raw["actor"], 1))
                scalar = E.encode_observation(raw, cfg)
                mini = E.encode_observation_minimal(raw, cfg)
                for name, got, want in (
                    ("numpy batch", numpy_obs[i], scalar),
                    ("fused rust", rust_obs[i], scalar),
                    ("minimal numpy batch", mini_numpy[i], mini),
                    ("minimal fused rust", mini_rust[i], mini),
                    ("minimal projection", E.project_obs_minimal(scalar), mini),
                ):
                    assert np.array_equal(got, want), (
                        f"{variant} {name} != scalar at dims "
                        f"{np.nonzero(got != want)[0][:10].tolist()} (stacks {stacks})"
                    )
                seen["nodes"] += 1
                seen["short_shove"] += int(raw["min_raise"] == 0 < raw["max_raise"])
                seen["raise_illegal"] += int(raw["max_raise"] == 0)
                seen["straight_draw"] += int(scalar[E._DRAW_A_OFF + 1])

            legal = np.asarray(bundle["legal_mask"])
            mn, mx = np.asarray(bundle["min_raise"]), np.asarray(bundle["max_raise"])
            gate_mask = gate_mask_from_bounds(legal, mx, BB)
            gates = np.ones(n, dtype=np.uint8)
            chips = np.zeros(n, dtype=np.uint64)
            for i in range(n):
                if serial[i].is_terminal():
                    continue
                gate = int(rng.choice(np.nonzero(gate_mask[i])[0]))
                if gate == GATE_RAISE and mn[i] == 0:
                    gate = 3  # short shove: the env redirects Raise to AllIn
                elif gate == GATE_RAISE:
                    chips[i] = int(rng.integers(int(mn[i]), int(mx[i]) + 1))
                gates[i] = gate
            be.apply_hybrid_batch(gates, chips)
            for i in range(n):
                if serial[i].is_terminal():
                    continue
                if gates[i] == GATE_RAISE:
                    serial[i].apply_raise_chips(int(chips[i]))
                else:
                    serial[i].apply_action({0: FOLD, 1: CHECK_CALL, 3: ALL_IN}[int(gates[i])])
    return seen


@pytest.mark.parametrize("rev", [E.OBS_REV_LEGACY, E.OBS_REV_CURRENT])
@pytest.mark.parametrize(
    "variant", ["plo5_double_bomb", VARIANT_PLO4, VARIANT_PLO6]
)
def test_three_way_parity_heterogeneous_stacks(monkeypatch, variant: str, rev: int) -> None:
    """scalar == numpy batch == fused Rust (full AND minimal layouts), bit for
    bit, on heterogeneous stacks with continuous raise sizes — the regime where
    B1/B3 lived (0% of uniform-stack nodes, 23-36% of heterogeneous ones). In
    BOTH observation-semantics revisions: rev 1 (the pre-fix values, kept for
    old checkpoints) must be just as exact across the three encoders."""
    pin_obs_rev(monkeypatch, rev)  # before any engine is built
    seen = _three_way_parity(variant, trials=14, seed=41)
    assert seen["nodes"] > 400
    assert seen["short_shove"] > 0 and seen["raise_illegal"] > 0 and seen["straight_draw"] > 0


# ---------------------------------------------------------------------------
# Latent guards
# ---------------------------------------------------------------------------


def _nlh_bundle(n: int = 3):
    be = BatchedEngine(n, num_seats=3, starting_stack=1_000_000, ante=5_000, bb=BB,
                       variant="nlh_single", sb=5_000)
    be.reset_batch(np.arange(n, dtype=np.uint64), np.zeros(n, dtype=np.uint8))
    bundle = dict(be.observation_and_features_batch())
    cfg = GameConfig(num_seats=3, starting_stack=1_000_000, ante=5_000, bb=BB,
                     sb=5_000, variant=VARIANT_NLH)
    return bundle, cfg


def test_nlh_batch_encoder_rejects_a_mis_sized_history_pack() -> None:
    bundle, cfg = _nlh_bundle()
    cat = np.asarray(bundle["hero_cat_a"])
    assert EN.encode_observation_batch_nlh(bundle, cat, cfg).shape == (3, EN.OBS_DIM_NLH)
    for width in (32, 48):  # the PLO width, and one that would overrun the SPR block
        bad = dict(bundle)
        for key, fill in (("history_seat", -1), ("history_action", -1),
                          ("history_chips", 0), ("history_street", -1)):
            src = np.asarray(bundle[key])
            bad[key] = np.full((src.shape[0], width), fill, dtype=src.dtype)
        with pytest.raises(ValueError, match="history pack width"):
            EN.encode_observation_batch_nlh(bad, cat, cfg)


def test_batch_encoder_gates_outcome_blocks_by_live_mask() -> None:
    """Terminal rows are all-zero even when the caller's bundle carries values
    in the opp-outcome / per-board blocks (the encoder used to write those two
    blocks unmasked and rely on the packer zeroing them)."""
    be = BatchedEngine(3, num_seats=2, starting_stack=200_000, ante=30_000, bb=BB)
    be.reset_batch(np.arange(3, dtype=np.uint64), np.zeros(3, dtype=np.uint8))
    bundle = {k: np.array(v) for k, v in dict(be.observation_and_features_batch()).items()}
    bundle["actor"][1] = -1  # present row 1 as terminal
    bundle["opp_outcome_fractions"][:] = 0.25
    bundle["per_board_outcome"][:] = 0.5
    cfg = GameConfig(num_seats=2, starting_stack=200_000, ante=30_000, bb=BB)
    out = E.encode_observation_batch(bundle, bundle["hero_cat_a"], bundle["hero_cat_b"], cfg)
    assert not out[1].any()
    assert (out[0, E._OPP_OUTCOME_OFF : E._OPP_OUTCOME_OFF + 12] == 0.25).all()
    assert (out[2, E._PER_BOARD_OUTCOME_OFF : E._PER_BOARD_OUTCOME_OFF + 8] == 0.5).all()
