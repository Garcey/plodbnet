"""Engine-rule regressions from the 2026-09-20 code review (C1, C2, C3, C4,
C8, B4), driven through the Python bindings / serial env.

NEEDS THE REBUILT EXTENSION. Every test here pins behavior that changed in
`rust_engine/src/{engine,double_board,state}.rs`; against a stale
`_engine.pyd` they fail, by design. The Rust-side twins live in
`engine::review_2026_09_20_tests` and `double_board::tests`.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from plo5bp import _engine as E
from plo5bp.actions import ALL_IN
from plo5bp.config import (
    VARIANT_NLH,
    VARIANT_PLO4,
    VARIANT_PLO5,
    VARIANT_PLO6,
    GameConfig,
)
from plo5bp.env import BombPotEnv

FOLD, CHECK_CALL, RAISE = 0, 1, 2
BB = 10_000


def _nlh(stacks: tuple[int, ...], ante: int = 5_000, sb: int = 5_000, bb: int = BB) -> GameConfig:
    return GameConfig(
        num_seats=len(stacks),
        starting_stacks=stacks,
        ante=ante,
        bb=bb,
        sb=sb,
        variant=VARIANT_NLH,
    )


def _assert_settlement(env: BombPotEnv, raw: dict, payouts) -> None:
    """Zero-sum; nobody wins more than the other seats MATCHED against
    their own commit; a folded seat loses exactly its commit (so nobody
    ever folded while out-committing every alive seat)."""
    tc = [int(x) for x in raw["total_commit"]]
    folded = [bool(x) for x in raw["folded"]]
    p = [int(x) for x in payouts]
    n = len(tc)
    assert sum(p) == 0, p
    alive_max = max(tc[i] for i in range(n) if not folded[i])
    for i in range(n):
        matched = sum(min(tc[j], tc[i]) for j in range(n) if j != i)
        assert p[i] <= matched, f"seat {i} won {p[i]} > matched {matched} (tc {tc})"
        assert p[i] >= -tc[i]
        if folded[i]:
            assert tc[i] <= alive_max, f"seat {i} folded to a phantom bet (tc {tc})"
            assert p[i] == -tc[i]


# ---- C1: NLH phantom bet / lone actor ----


def test_c1_covering_sb_is_not_offered_a_fold_vs_short_all_in_bb() -> None:
    # Review scenario A: HU [100bb, 0.8bb]; the BB posts its last 3000
    # all-in, the SB's 5000 already covers it. The SB used to be offered
    # Fold against the NOMINAL 10000 and folding paid [-10000, +10000].
    env = BombPotEnv(_nlh((1_000_000, 8_000)))
    obs, info = env.reset(11, 0)
    assert info.actor is None and env.is_terminal(), "terminal at deal"
    assert not obs.any()
    raw = info.raw_obs
    assert [int(x) for x in raw["total_commit"]] == [10_000, 8_000]
    assert int(raw["street"]) == 4 and len(raw["board_a"]) == 5
    assert list(raw["history"]) == []
    payouts = env._rs.payouts()
    _assert_settlement(env, raw, payouts)
    assert abs(payouts[0]) in (0, 8_000), payouts  # never the unmatched 2000


def test_c1_uncalled_sb_chips_come_back() -> None:
    # Review scenario B: 3-way [51, 10000, 51], ante 50, blinds 50/100.
    cfg = _nlh((51, 10_000, 51), ante=50, sb=50, bb=100)
    for btn_gate in (CHECK_CALL, FOLD):
        env = BombPotEnv(cfg)
        _, info = env.reset(3, 0)
        assert info.actor == 0, "BTN faces a real bet (the SB's 50)"
        _, rewards, done, info = env.step_hybrid(btn_gate, 0)
        assert done, "the SB has nothing to contest → no fold node, run-out"
        assert not info.raw_obs["folded"][1]
        _assert_settlement(env, info.raw_obs, rewards)
        assert rewards[1] >= -51, rewards  # only 51 of the SB's 100 was matched


def test_c1_lone_stack_at_hand_start_runs_out() -> None:
    # PLO: two seats all-in from the ante, one deep stack with nobody to
    # bet against. It used to get a forced check node.
    cfg = GameConfig(num_seats=3, starting_stacks=(30_000, 30_000, 500_000))
    env = BombPotEnv(cfg)
    _, info = env.reset(5, 0)
    assert info.actor is None and env.is_terminal()
    raw = info.raw_obs
    assert (len(raw["board_a"]), len(raw["board_b"])) == (5, 5)
    assert int(raw["stacks"][2]) == 470_000
    _assert_settlement(env, raw, env._rs.payouts())
    # Two stacks behind → a normal hand.
    env = BombPotEnv(GameConfig(num_seats=3, starting_stacks=(30_000, 500_000, 500_000)))
    assert env.reset(5, 0)[1].actor == 1


def _rand_stack(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.25:
        return rng.randint(1, 3 * BB)  # micro: can be below the ante / a blind
    if r < 0.6:
        return rng.randint(3 * BB, 40 * BB)
    return rng.randint(40 * BB, 300 * BB)


def test_c1_chip_conservation_fuzz_micro_stacks() -> None:
    # Port of the reviewer's fuzzer: the win<=matched / phantom-fold checks
    # fired for nlh_single only, within the first few dozen hands.
    rng = random.Random(7)
    for hand in range(600):
        variant = rng.choice([VARIANT_NLH, VARIANT_NLH, VARIANT_PLO5, VARIANT_PLO4, VARIANT_PLO6])
        n = rng.randint(2, 6)
        stacks = tuple(_rand_stack(rng) for _ in range(n))
        if variant == VARIANT_NLH:
            cfg = _nlh(stacks, ante=rng.choice([0, 5_000]))
        else:
            cfg = GameConfig(
                num_seats=n,
                starting_stacks=stacks,
                ante=rng.choice([10_000, 30_000]),
                bb=BB,
                variant=variant,
            )
        mask = None
        if n >= 3 and rng.random() < 0.3:
            mask = [True] * n
            for i in rng.sample(range(n), rng.randint(1, n - 2)):
                mask[i] = False
        env = BombPotEnv(cfg)
        seed, button = rng.getrandbits(63), rng.randrange(n)
        ctx = f"hand {hand} {variant} stacks {stacks} button {button} mask {mask} seed {seed}"
        _, info = env.reset(seed, button, mask)
        done, steps = info.actor is None, 0
        while not done:
            raw, a = info.raw_obs, info.actor
            sc = [int(x) for x in raw["street_commit"]]
            assert not raw["folded"][a] and not raw["all_in"][a] and raw["stacks"][a] > 0, ctx
            if info.gate_mask[FOLD]:
                real_bet = any(
                    sc[j] > sc[a] for j in range(n) if j != a and not raw["folded"][j]
                )
                assert real_bet, f"{ctx}: seat {a} offered Fold with no real bet (sc {sc})"
            gates = [g for g in (FOLD, CHECK_CALL, RAISE) if info.gate_mask[g]]
            gate = rng.choice(gates + ([RAISE] * 2 if info.gate_mask[RAISE] else []))
            chips = 0
            mn, mx = info.min_raise_chips, info.max_raise_chips
            if gate == RAISE and mn > 0:
                chips = rng.choice([mn, mx, rng.randint(mn, mx)])
            elif gate == RAISE:
                assert info.legal_mask[ALL_IN], ctx
            _, _, done, info = env.step_hybrid(gate, chips)
            steps += 1
            assert steps < 400, ctx
        # Exact integer payouts (the env's float32 rewards round past 2^24).
        raw = info.raw_obs
        _assert_settlement(env, raw, env._rs.payouts())
        for i in range(n):
            assert int(raw["stacks"][i]) + int(raw["total_commit"][i]) == stacks[i], ctx


# ---- C2: heads-up NLH with the button on a sitting-out seat ----


def test_c2_heads_up_dead_button_bb_acts_first_postflop() -> None:
    env = BombPotEnv(_nlh((1_000_000,) * 3))
    _, info = env.reset(7, 0, [False, True, True])
    raw = info.raw_obs
    # Clockwise from button+1: seat 1 = BB, seat 2 inherits the button (SB).
    assert (raw["sb_seat"], raw["bb_seat"]) == (2, 1)
    assert info.actor == 2, "SB first preflop"
    _, _, _, info = env.step_hybrid(CHECK_CALL, 0)
    assert info.actor == 1, "BB option"
    _, _, _, info = env.step_hybrid(CHECK_CALL, 0)
    assert int(info.raw_obs["street"]) == 1
    assert info.actor == 1, "BB first postflop (the SB used to act first on every street)"
    # Button in the hand: unchanged (button = SB, acts first pre, last post).
    _, info = env.reset(7, 1, [False, True, True])
    assert (info.raw_obs["sb_seat"], info.raw_obs["bb_seat"]) == (1, 2)


# ---- C3: study terminal classification ----


def test_c3_study_river_all_in_and_call_is_a_showdown() -> None:
    g = E.GameState(
        num_seats=2,
        starting_stacks=np.array([60_000, 900_000], dtype=np.uint64),
        ante=30_000,
        bb=BB,
    )
    g.reset_study(0, 1, [0, 1, 2, 3, 4], [5, 6, 7], [8, 9, 10])
    g.apply_action(CHECK_CALL)
    g.apply_action(CHECK_CALL)
    g.set_turn(11, 12)
    g.apply_action(CHECK_CALL)
    g.apply_action(CHECK_CALL)
    g.set_river(13, 14)
    g.apply_action(CHECK_CALL)  # seat 1 checks
    g.apply_action(ALL_IN)  # seat 0 shoves its last 3bb
    g.apply_action(CHECK_CALL)  # seat 1 calls with chips behind
    # 0=FoldOut, 1=RunOut ("turn/river cards were not entered"), 2=Showdown.
    assert g.study_terminal() == 2
    assert g.observation_dict(skip_outcome_mc=True)["street"] == 4


def test_c3_nlh_study_river_all_in_and_call_is_a_showdown() -> None:
    g = E.GameState(
        num_seats=2,
        starting_stacks=np.array([60_000, 900_000], dtype=np.uint64),
        ante=5_000,
        bb=BB,
        variant=VARIANT_NLH,
        sb=5_000,
    )
    g.reset_study_nlh(0, 1, [51, 47])
    g.apply_action(CHECK_CALL)
    g.apply_action(CHECK_CALL)
    g.set_flop_nlh(0, 5, 10)
    g.apply_action(CHECK_CALL)
    g.apply_action(CHECK_CALL)
    g.set_turn_nlh(15)
    g.apply_action(CHECK_CALL)
    g.apply_action(CHECK_CALL)
    g.set_river_nlh(20)
    g.apply_action(CHECK_CALL)
    g.apply_raise_chips(g.max_raise_chips())
    g.apply_action(CHECK_CALL)
    assert g.study_terminal() == 2


@pytest.mark.parametrize("stacks", [(30_000, 30_000, 30_000), (30_000, 30_000, 500_000)])
def test_c3_study_hand_nobody_can_act_in_is_classified(stacks) -> None:
    # Everyone all-in from the ante (or a lone stack with nothing to
    # contest): used to sit terminal with study_terminal=None.
    g = E.GameState(
        num_seats=3, starting_stacks=np.array(stacks, dtype=np.uint64), ante=30_000, bb=BB
    )
    g.reset_study(0, 1, [0, 1, 2, 3, 4], [5, 6, 7], [8, 9, 10])
    assert g.is_terminal()
    assert g.study_terminal() == 1
    assert g.awaiting_next_street() is None
    assert g.payouts() == [0, 0, 0]


# ---- C4: too many seats for the study placeholder deck ----


def test_c4_reset_study_with_too_many_seats_is_a_value_error() -> None:
    # 10 seats indexed past the 41-card unseen deck: PanicException (a
    # BaseException — `except Exception` never saw it). The raw binding is
    # used on purpose: the Python GameConfig already refuses 9+ seats.
    for n in (9, 10):
        try:
            g = E.GameState(num_seats=n)
        except ValueError:
            continue  # constructor-level rejection is just as good
        with pytest.raises(ValueError):
            g.reset_study(0, 0, [0, 1, 2, 3, 4], [5, 6, 7], [8, 9, 10])


# ---- C8: study placeholder collisions ----


def _all_distinct(env: BombPotEnv) -> bool:
    raw = env._rs.observation_dict(skip_outcome_mc=True)
    cards = [c for h in env.all_hole_cards() for c in h]
    cards += list(raw["board_a"]) + list(raw["board_b"])
    return len(cards) == len(set(cards))


def test_c8_street_cards_colliding_with_placeholders_are_redrawn() -> None:
    def build() -> BombPotEnv:
        env = BombPotEnv(GameConfig())
        env.reset_study(0, 1, [0, 1, 2, 3, 4], [5, 6, 7], [8, 9, 10])
        for _ in range(6):
            env.step_hybrid(CHECK_CALL, 0)
        assert env.awaiting_next_street() == 2
        return env

    env = build()
    holes = env.all_hole_cards()
    # The user cannot see villain placeholders; two of them as the turn.
    x, y = holes[0][0], holes[3][2]
    env.set_turn(x, y)
    assert _all_distinct(env), "placeholder must move out of the board card's way"
    after = env.all_hole_cards()
    assert after[1] == [0, 1, 2, 3, 4], "hero untouched"
    moved = [(s, k) for s in range(6) for k in range(5) if after[s][k] != holes[s][k]]
    assert moved == [(0, 0), (3, 2)]
    # Pure function of the state → `_rebuild_env` replays redraw identically.
    replay = build()
    replay.set_turn(x, y)
    assert replay.all_hole_cards() == after
    # Hero / board collisions are still the user's error.
    with pytest.raises(ValueError, match="duplicate card"):
        build().set_turn(0, 20)


def test_c8_nlh_flop_colliding_with_placeholders_is_redrawn() -> None:
    env = BombPotEnv(_nlh((1_000_000,) * 6))
    env.reset_study_nlh(0, 3, [51, 47])
    for _ in range(6):
        env.step_hybrid(CHECK_CALL, 0)
    holes = env.all_hole_cards()
    env.set_flop_nlh(holes[0][0], holes[0][1], holes[5][1])
    assert _all_distinct(env)
    assert env.all_hole_cards()[3] == [51, 47]


# ---- B4: opp-outcome MC dims are a function of the card SETS ----


def _study_obs(hero_seat: int, hole: list[int], flop_a: list[int], flop_b: list[int]) -> np.ndarray:
    env = BombPotEnv(GameConfig(num_seats=6, starting_stack=50 * BB))
    # Button one seat behind the hero → hero is first to act, empty history.
    obs, info = env.reset_study((hero_seat - 1) % 6, hero_seat, hole, flop_a, flop_b)
    assert info.actor == hero_seat
    return obs


def test_b4_mc_dims_ignore_card_order_and_absolute_seat() -> None:
    hole, fa, fb = [48, 44, 21, 10, 3], [50, 37, 8], [29, 17, 1]
    base = _study_obs(0, hole, fa, fb)
    assert base[982:990].any(), "k=3/k=4 MC dims must be live on the flop"
    # Dims 982-989 used to move with all three (sampling noise only).
    np.testing.assert_array_equal(_study_obs(0, hole[::-1], fa, fb), base)
    np.testing.assert_array_equal(_study_obs(0, hole, fa[::-1], fb[::-1]), base)
    np.testing.assert_array_equal(_study_obs(3, hole, fa, fb), base)


def test_b4_outcome_features_follow_the_cards_not_the_seat() -> None:
    # Random deals: whichever seat holds a hand, the raw fused MC output for
    # it is identical to a study hand holding the same cards elsewhere.
    env = BombPotEnv(GameConfig())
    _, info = env.reset(2024, 0)
    raw = info.raw_obs
    live = env._rs.outcome_features_mc(64)
    hole = [int(c) for c in raw["hero_hole"]]
    fa, fb = [int(c) for c in raw["board_a"]], [int(c) for c in raw["board_b"]]
    for hero_seat in (0, 4):
        g = E.GameState(num_seats=6)
        g.reset_study((hero_seat - 1) % 6, hero_seat, sorted(hole), fa[::-1], fb)
        assert g.current_actor() == hero_seat
        assert g.outcome_features_mc(64) == live
