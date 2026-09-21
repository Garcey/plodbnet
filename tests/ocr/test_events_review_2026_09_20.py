"""Regression tests for the 2026-09-20 code review, section I (live capture).

Synthetic `FrameState` sequences through the real `EventReconstructor` — no
pixels, no OpenCV, no server. Each block names the review item it pins and
mirrors the reviewer's repro script (`.claude/reviews/repro-2026-09-20/
agent_ocr/h*.py`). The I4 block additionally replays every emitted action
through a REAL engine env, because "the engine rejects it and the action is
lost" is only provable against the engine.

Units: frames are in CENTS, the engine in CHIPS; `SCALE`/`MINBET` below are the
production defaults (bb=10000 chips, $20/bb → 5 chips per cent, 1bb = 2000c).
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from plo5bp.ocr import events as events_mod
from plo5bp.ocr.events import (
    EngineView,
    EventReconstructor,
    OcrWarning,
    SeatAction,
    StreetReveal,
    cents_to_engine_chips,
)
from plo5bp.ocr.types import Card, FrameState, SeatObs

SCALE, MINBET = 5.0, 2000


def _c(rank: int, suit: int) -> Card:
    return Card(rank=rank, suit=suit)


FLOP_A = (_c(10, 0), _c(8, 0), _c(11, 0), None, None)
FLOP_B = (_c(5, 1), _c(7, 2), _c(1, 1), None, None)
TURN_A = FLOP_A[:3] + (_c(2, 3), None)
TURN_B = FLOP_B[:3] + (_c(9, 2), None)
NO_BOARD = (None,) * 5
HERO = (_c(8, 1), _c(7, 1), _c(7, 0), _c(3, 2), _c(1, 2))


def fs(commits, stacks, *, hero=HERO, board_a=FLOP_A, board_b=FLOP_B, button=0,
       banners=None, actors=None, folded=None, pot=70_000) -> FrameState:
    n = len(commits)
    banners = banners or [False] * n
    actors = actors or [False] * n
    folded = folded or [False] * n
    return FrameState(
        board_a=board_a, board_b=board_b, hero_hole=hero, button_seat=button,
        pot_total_chips=pot,
        seats=tuple(
            SeatObs(seat=i, stack_chips=stacks[i], committed_chips=commits[i],
                    bet_banner=banners[i], is_actor=actors[i], folded=folded[i])
            for i in range(n)
        ),
    )


def engine(current_actor, commits, stacks, *, street=1, folded=None, all_in=None,
           bet_to_call=None, button=0, chips_per_cent=SCALE, min_bet_cents=MINBET,
           sitting_out=None, exact_folds=False) -> EngineView:
    n = len(commits)
    return EngineView(
        num_seats=n, current_actor=current_actor, street=street,
        awaiting_next_street=None, button_seat=button,
        committed_this_street=tuple(commits), stacks=tuple(stacks),
        folded=tuple(folded or [False] * n), all_in=tuple(all_in or [False] * n),
        bet_to_call=bet_to_call if bet_to_call is not None else max(commits),
        chips_per_cent=chips_per_cent, min_bet_cents=min_bet_cents,
        sitting_out=tuple(sitting_out or [False] * n), exact_folds=exact_folds,
    )


def acts(events) -> list[SeatAction]:
    return [e for e in events if isinstance(e, SeatAction)]


HU_SITTING_OUT = [False, False, True, True, True, True]


# ===========================================================================
# I2 — the engine can be a street AHEAD of the screen
# ===========================================================================


def _closing_call_then_engine_on_turn(*, closer_banner: bool):
    """Heads-up in the 6-seat layout. Seat 1 bet $180, hero's closing call is
    seen on tick N; the server applies it and `_rebuild_env` advances the engine
    to the TURN with padded cards. Returns (rec, stale_frame)."""
    rec = EventReconstructor(num_seats=6)
    eng_flop = engine(0, [0, 90_000, 0, 0, 0, 0], [500_000, 410_000, 0, 0, 0, 0],
                      bet_to_call=90_000, sitting_out=HU_SITTING_OUT, street=1)
    pre = fs([None, 18_000, None, None, None, None],
             [100_000, 82_000, None, None, None, None],
             actors=[True, False, False, False, False, False])
    rec.step(pre, eng_flop)
    call = fs([18_000, 18_000, None, None, None, None],
              [82_000, 82_000, None, None, None, None],
              banners=[closer_banner, False, False, False, False, False])
    assert acts(rec.step(call, eng_flop)) == [SeatAction(0, "check_call", 0)]
    return rec, call


def test_i2_stale_flop_ovals_do_not_cascade_checks_on_the_padded_turn():
    """H1: chips not swept yet, engine already on the turn with commits 0.
    Pre-fix: CHECK(1) CHECK(0) CHECK(1) CHECK(0) + loop-guard warning — one
    tick took the hand to the river."""
    rec, stale = _closing_call_then_engine_on_turn(closer_banner=False)
    eng_turn = engine(1, [0] * 6, [410_000, 410_000, 0, 0, 0, 0], bet_to_call=0,
                      sitting_out=HU_SITTING_OUT, street=2)
    ev = rec.step(replace(stale), eng_turn)
    assert acts(ev) == []
    assert not [e for e in ev if isinstance(e, OcrWarning)]


def test_i2_stale_oval_with_banner_is_not_a_phantom_turn_raise():
    """H1c: the closer's banner (300-500 ms) is still up, so Fix N accepted the
    stale oval. Pre-fix: RAISE 90000 by hero on the turn."""
    rec, stale = _closing_call_then_engine_on_turn(closer_banner=True)
    eng_turn = engine(0, [0] * 6, [410_000, 410_000, 0, 0, 0, 0], bet_to_call=0,
                      sitting_out=HU_SITTING_OUT, street=2, button=1)
    assert acts(rec.step(replace(stale), eng_turn)) == []


def test_i2_walk_resumes_once_the_board_catches_up():
    rec, stale = _closing_call_then_engine_on_turn(closer_banner=False)
    eng_turn = engine(1, [0] * 6, [410_000, 410_000, 0, 0, 0, 0], bet_to_call=0,
                      sitting_out=HU_SITTING_OUT, street=2)
    assert acts(rec.step(replace(stale), eng_turn)) == []
    # Turn dealt, chips swept.
    swept = fs([None] * 6, [82_000, 82_000, None, None, None, None],
               board_a=TURN_A, board_b=TURN_B)
    ev = rec.step(swept, eng_turn)
    assert [e for e in ev if isinstance(e, StreetReveal)]
    assert acts(ev) == []
    # Seat 1 bets $100 on the turn: the walk is live again.
    bet = fs([None, 10_000, None, None, None, None],
             [82_000, 72_000, None, None, None, None],
             board_a=TURN_A, board_b=TURN_B,
             banners=[False, True, False, False, False, False])
    assert acts(rec.step(bet, eng_turn)) == [SeatAction(1, "raise", 50_000)]


def test_i2_street_gate_is_bounded_so_an_unreadable_card_cannot_stall_the_hand():
    """A turn card that never classifies leaves the visible street at "flop"
    for the rest of the hand. The gate must lapse rather than suppress every
    later action."""
    rec = EventReconstructor(num_seats=2)
    base = fs([None, None], [100_000, 100_000])
    eng_turn = engine(1, [0, 0], [500_000, 500_000], bet_to_call=0, street=2)
    rec.step(base, eng_turn)
    for _ in range(events_mod._MAX_STREET_GATE_TICKS):
        assert acts(rec.step(base, eng_turn)) == []
    bet = fs([None, 18_000], [100_000, 82_000], banners=[False, True])
    assert acts(rec.step(bet, eng_turn)) == [SeatAction(1, "raise", 90_000)]


def test_i2_uncorroborated_downstream_oval_is_not_evidence_of_action():
    """Same stale-oval shape but with the board already caught up (so the
    street gate is out of the picture): an oval nobody's stack paid for and no
    banner announces must not certify a CHECK for the seats before it — the
    walk itself would reject that read on arrival (Fix N)."""
    rec = EventReconstructor(num_seats=6)
    eng = engine(1, [0] * 6, [410_000, 410_000, 0, 0, 0, 0], bet_to_call=0,
                 sitting_out=HU_SITTING_OUT, street=2)
    stale = fs([18_000, 18_000, None, None, None, None],
               [82_000, 82_000, None, None, None, None],
               board_a=TURN_A, board_b=TURN_B)
    rec.step(stale, eng)
    ev = rec.step(replace(stale), eng)
    assert acts(ev) == []
    assert not [e for e in ev if isinstance(e, OcrWarning)]


def test_i2_any_remaining_delta_signal_a_needs_corroboration():
    rec = EventReconstructor(num_seats=3)
    quiet = SeatObs(seat=0, stack_chips=100_000)
    last = {0: quiet, 1: SeatObs(seat=1, stack_chips=100_000), 2: SeatObs(seat=2, stack_chips=100_000)}

    def ard(seat1: SeatObs, **kw) -> bool:
        return rec._any_remaining_delta(
            {0: quiet, 1: seat1, 2: last[2]}, last, [0, 0, 0], SCALE,
            actor_start=1, min_bet_cents=MINBET, **kw)

    bare = SeatObs(seat=1, stack_chips=100_000, committed_chips=18_000)
    assert ard(bare) is False                                   # nothing backs it
    assert ard(replace(bare, bet_banner=True)) is True          # banner backs it
    assert ard(replace(bare, stack_chips=82_000)) is True       # stack paid for it
    assert ard(replace(bare, stack_chips=82_000), skip={1}) is False  # excluded seat
    # Sub-1bb oval noise stays ignored even with a banner.
    assert ard(SeatObs(seat=1, stack_chips=100_000, committed_chips=300,
                       bet_banner=True)) is False


def test_i2_no_pass_ever_acts_twice_for_one_seat():
    """Property: whatever the frames look like, one walk attributes at most one
    action to a seat and never runs into the loop guard. (H1's cascade was the
    same two seats visited twice each.)"""
    rng = random.Random(20260920)
    amounts = [None, 0, 400, 2_000, 9_000, 18_000, 36_000]
    stacks = [None, 0, 64_000, 82_000, 91_000, 100_000]
    for _ in range(3_000):
        n = rng.choice([2, 3, 4, 6])
        commits = [rng.choice([0, 0, 0, 45_000, 90_000]) for _ in range(n)]
        eng = engine(
            rng.randrange(n), commits, [500_000] * n,
            bet_to_call=max(commits), street=rng.choice([1, 2]),
            folded=[rng.random() < 0.15 for _ in range(n)],
            all_in=[rng.random() < 0.1 for _ in range(n)],
            sitting_out=[rng.random() < 0.15 for _ in range(n)],
            exact_folds=rng.random() < 0.3,
        )

        def frame():
            return fs(
                [rng.choice(amounts) for _ in range(n)],
                [rng.choice(stacks) for _ in range(n)],
                board_a=TURN_A, board_b=TURN_B,
                banners=[rng.random() < 0.2 for _ in range(n)],
                actors=[rng.random() < 0.2 for _ in range(n)],
                folded=[rng.random() < 0.2 for _ in range(n)],
            )

        rec = EventReconstructor(num_seats=n)
        rec.step(frame(), eng)
        for _tick in range(3):
            ev = rec.step(frame(), eng)
            seats = [a.seat for a in acts(ev)]
            assert len(seats) == len(set(seats)), (seats, ev)
            assert not [w for w in ev if isinstance(w, OcrWarning) and "loop guard" in w.message]
            for a in acts(ev):
                assert not eng.folded[a.seat] and not eng.all_in[a.seat]
                assert not eng.sitting_out[a.seat]


# ===========================================================================
# I3 — a coasted seat facing a raise is NOT folded because someone else acted
# ===========================================================================


def _i3_setup():
    """4 seats: seat1 bet $100, seat2 called, seat3 raised to $300; hero (0) to
    act. Seats 1 and 2 have chips in and have not answered the raise yet."""
    rec = EventReconstructor(num_seats=4)
    eng = engine(0, [0, 50_000, 50_000, 150_000], [500_000] * 4, bet_to_call=150_000)
    base = fs([None, 10_000, 10_000, 30_000], [100_000, 90_000, 90_000, 70_000],
              actors=[True, False, False, False])
    rec.step(base, eng)
    return rec, eng


def test_i3_upstream_call_does_not_fold_seats_still_to_act():
    """H9a. Pre-fix: CALL(0) FOLD(1) FOLD(2) in one tick — cards visible,
    ovals unchanged, neither seat had had a turn."""
    rec, eng = _i3_setup()
    t = fs([30_000, 10_000, 10_000, 30_000], [70_000, 90_000, 90_000, 70_000],
           banners=[True, False, False, False], actors=[False, True, False, False])
    assert acts(rec.step(t, eng)) == [SeatAction(0, "check_call", 0)]


def test_i3_upstream_fold_does_not_fold_seats_still_to_act():
    """H9b: hero folds instead. Only hero's (confirmed) fold may be emitted."""
    rec, eng = _i3_setup()
    t = fs([None, 10_000, 10_000, 30_000], [100_000, 90_000, 90_000, 70_000],
           folded=[True, False, False, False], hero=(None,) * 5,
           actors=[False, True, False, False])
    assert acts(rec.step(t, eng)) == []                       # first sighting
    assert acts(rec.step(replace(t), eng)) == [SeatAction(0, "fold", 0)]


def test_i3_real_fold_of_a_seat_with_chips_in_arrives_via_obs_folded():
    """The replacement path for the deleted inference: once the seat's cards
    AND chips are gone (`folded=True`, two frames), Fix J folds it."""
    rec = EventReconstructor(num_seats=4)
    eng = engine(1, [150_000, 50_000, 50_000, 150_000], [500_000] * 4, bet_to_call=150_000)
    live = fs([30_000, 10_000, 10_000, 30_000], [70_000, 90_000, 90_000, 70_000])
    rec.step(live, eng)
    gone = fs([30_000, None, 10_000, 30_000], [70_000, 90_000, 90_000, 70_000],
              folded=[False, True, False, False])
    assert acts(rec.step(gone, eng)) == []
    assert acts(rec.step(replace(gone), eng)) == [SeatAction(1, "fold", 0)]


# ===========================================================================
# I4 — an all-in CALL is a call; every emitted action is legal in a real engine
# ===========================================================================


def _engine_env(stacks_dollars, *, button):
    """Real engine, PokerNow-style units: bb = 2 chips = $1 → 0.02 chips/cent.
    `stacks_dollars` are POST-ante (what the table shows at the flop)."""
    pytest.importorskip("plo5bp._engine")
    from plo5bp.config import GameConfig
    from plo5bp.env import BombPotEnv

    ante = 12  # chips ($6)
    cfg = GameConfig(
        num_seats=len(stacks_dollars), ante=ante, bb=2,
        starting_stacks=tuple(int(d * 2) + ante for d in stacks_dollars),
    )
    env = BombPotEnv(cfg)
    env.reset(seed=7, button=button)
    return env


def _view(env, *, exact_folds=True) -> EngineView:
    """Mirror of server._engine_view_from_session for a bare env."""
    raw = dict(env._rs.observation_dict())
    n = env.num_seats
    actor = raw.get("actor")
    return EngineView(
        num_seats=n, current_actor=None if actor is None else int(actor),
        street=int(raw["street"]), awaiting_next_street=raw.get("awaiting_next_street"),
        button_seat=int(raw["button"]),
        committed_this_street=tuple(int(x) for x in raw["street_commit"][:n]),
        stacks=tuple(int(x) for x in raw["stacks"][:n]),
        folded=tuple(bool(x) for x in raw["folded"][:n]),
        all_in=tuple(bool(x) for x in raw["all_in"][:n]),
        bet_to_call=int(raw["bet_to_call"]),
        chips_per_cent=0.02, min_bet_cents=100, exact_folds=exact_folds,
    )


_GATE = {"fold": 0, "check_call": 1, "raise": 2}


def _drive(env, rec, frame) -> list[SeatAction]:
    """One server tick: step the reconstructor, then apply every emitted action
    to the engine exactly as `_rebuild_env` replays the (seat-less) action log.
    Fails on the first action the engine rejects or that targets the wrong seat."""
    emitted = acts(rec.step(frame, _view(env)))
    for a in emitted:
        assert env.current_actor() == a.seat, (
            f"{a} would be applied to seat {env.current_actor()} — the log is seat-less"
        )
        chips = a.chips
        if a.gate == "raise":
            # Same clamp `_rebuild_env` applies: a live site lets a deep player
            # bet more than the effective stack; the engine caps raises there.
            mx = int(env._rs.max_raise_chips())
            if 0 < mx < chips:
                chips = mx
        env.step_hybrid(_GATE[a.gate], chips)  # raises if illegal
    return emitted


def _cents(*dollars):
    return [None if d is None else int(round(d * 100)) for d in dollars]


def _pn(commits, stacks, actors=None):
    return fs(_cents(*commits), _cents(*stacks), actors=actors, hero=HERO)


def test_i4_heads_up_all_in_call_for_less_is_legal_and_closes_the_hand():
    """H8b. Hero bets $10; villain, with $6 behind, calls all-in. Pre-fix:
    RAISE 12 → "chip amount out of legal range" → entry dropped → the engine
    sat on the villain for the rest of the hand. (Heads-up the engine clamps
    hero's over-bet to the $6 effective stack, so on the engine side this is
    an all-in for EXACTLY the bet; the 3-way test below is the true
    call-for-less.)"""
    env = _engine_env([86, 6], button=1)          # hero (0) acts first
    rec = EventReconstructor(num_seats=2)
    rec.step(_pn([None, None], [86, 6], actors=[True, False]), _view(env))
    assert _drive(env, rec, _pn([10, None], [76, 6], actors=[False, True])) == [
        SeatAction(0, "raise", 20)]
    assert _drive(env, rec, _pn([10, 6], [76, 0])) == [SeatAction(1, "check_call", 0)]
    raw = dict(env._rs.observation_dict())
    assert raw["all_in"][1] is True and raw["stacks"][1] == 0
    assert env.current_actor() is None            # nobody left to act


def test_i4_all_in_for_exactly_the_bet_is_a_call():
    """Stack 0 AND commit == bet: the stack-zero branch used to win and emit a
    raise of exactly `to_call`, which the engine also rejects."""
    env = _engine_env([86, 10], button=1)
    rec = EventReconstructor(num_seats=2)
    rec.step(_pn([None, None], [86, 10], actors=[True, False]), _view(env))
    _drive(env, rec, _pn([10, None], [76, 10], actors=[False, True]))
    assert _drive(env, rec, _pn([10, 10], [76, 0])) == [SeatAction(1, "check_call", 0)]
    assert dict(env._rs.observation_dict())["all_in"][1] is True


def test_i4_three_way_short_all_in_call_then_the_deep_seat_still_acts():
    """H8c. After the short seat's all-in call the DEEP seat must still be
    reachable — pre-fix every later action was lost behind the rejected raise."""
    env = _engine_env([86, 6, 90], button=2)      # hero, short, deep
    rec = EventReconstructor(num_seats=3)
    rec.step(_pn([None] * 3, [86, 6, 90], actors=[True, False, False]), _view(env))
    assert _drive(env, rec, _pn([10, None, None], [76, 6, 90], actors=[False, True, False])) == [
        SeatAction(0, "raise", 20)]
    assert _drive(env, rec, _pn([10, 6, None], [76, 0, 90], actors=[False, False, True])) == [
        SeatAction(1, "check_call", 0)]
    assert _drive(env, rec, _pn([10, 6, 10], [76, 0, 80])) == [SeatAction(2, "check_call", 0)]
    raw = dict(env._rs.observation_dict())
    assert raw["all_in"][:3] == [False, True, False]
    assert int(raw["street"]) == 2                # flop betting closed correctly


def test_i4_all_in_call_and_the_closing_call_seen_in_one_frame():
    """Gap-fill across the all-in: both calls land in a single poll."""
    env = _engine_env([86, 6, 90], button=2)
    rec = EventReconstructor(num_seats=3)
    rec.step(_pn([None] * 3, [86, 6, 90], actors=[True, False, False]), _view(env))
    _drive(env, rec, _pn([10, None, None], [76, 6, 90]))
    assert _drive(env, rec, _pn([10, 6, 10], [76, 0, 80])) == [
        SeatAction(1, "check_call", 0), SeatAction(2, "check_call", 0)]
    assert int(dict(env._rs.observation_dict())["street"]) == 2


def test_i4_short_all_in_over_the_bet_is_still_a_raise():
    """Stack 0 with commit > bet but under a min-raise: gate="raise" is right —
    step_hybrid routes it through the engine's ALL_IN path."""
    env = _engine_env([86, 14], button=1)
    rec = EventReconstructor(num_seats=2)
    rec.step(_pn([None, None], [86, 14], actors=[True, False]), _view(env))
    _drive(env, rec, _pn([10, None], [76, 14], actors=[False, True]))
    assert _drive(env, rec, _pn([10, 14], [76, 0], actors=[True, False])) == [
        SeatAction(1, "raise", 28)]
    raw = dict(env._rs.observation_dict())
    assert raw["all_in"][1] is True and env.current_actor() == 0
    assert int(raw["bet_to_call"]) == 28


def test_i4_open_shove_is_a_raise():
    env = _engine_env([86, 5], button=0)          # villain (1) acts first
    rec = EventReconstructor(num_seats=2)
    rec.step(_pn([None, None], [86, 5], actors=[False, True]), _view(env))
    assert _drive(env, rec, _pn([None, 5], [86, 0], actors=[True, False])) == [
        SeatAction(1, "raise", 10)]
    assert dict(env._rs.observation_dict())["all_in"][1] is True


# ===========================================================================
# I6 — single-frame signals
# ===========================================================================


def _facing_a_bet():
    rec = EventReconstructor(num_seats=6)
    eng = engine(2, [0, 90_000, 0, 0, 0, 0],
                 [500_000, 410_000, 500_000, 500_000, 500_000, 500_000], bet_to_call=90_000)
    good = fs([None, 18_000, None, None, None, None],
              [100_000, 82_000, 100_000, 100_000, 100_000, 100_000],
              actors=[False, False, True, False, False, False])
    rec.step(good, eng)
    assert acts(rec.step(good, eng)) == []
    return rec, eng, good


def test_i6_one_dark_frame_folds_nobody_and_never_becomes_the_baseline():
    """H7. Pre-fix: FOLD 2,3,4,5,0 — every live seat — from ONE unreadable
    frame; irreversible downstream (`folded_this_hand` is sticky)."""
    rec, eng, good = _facing_a_bet()
    baseline = rec.last_fs
    dark = fs([None] * 6, [None] * 6, hero=(None,) * 5, button=None,
              folded=[True] * 6, board_a=NO_BOARD, board_b=NO_BOARD)
    assert rec.step(dark, eng) == []
    assert rec.last_fs is baseline                 # dropped, not adopted
    assert acts(rec.step(good, eng)) == []         # and nothing confirms later


def test_i6_a_dark_first_frame_is_not_adopted_as_the_bootstrap_baseline():
    rec = EventReconstructor(num_seats=6)
    eng = engine(2, [0] * 6, [500_000] * 6, bet_to_call=0)
    dark = fs([None] * 6, [None] * 6, hero=(None,) * 5, folded=[True] * 6,
              board_a=NO_BOARD, board_b=NO_BOARD)
    assert rec.step(dark, eng) == []
    assert rec.last_fs is None


def test_i6_single_frame_fold_flicker_is_ignored():
    """One seat's card backs drop under threshold for one frame (banner
    overlap: 0.21-0.23 in hand vs 0.13 occluded, threshold 0.15)."""
    rec, eng, good = _facing_a_bet()
    flicker = replace(good, seats=tuple(
        replace(s, folded=True) if s.seat == 2 else s for s in good.seats))
    assert acts(rec.step(flicker, eng)) == []
    assert acts(rec.step(good, eng)) == []
    assert acts(rec.step(flicker, eng)) == []      # non-consecutive: still nothing


def test_i6_fold_is_emitted_once_two_consecutive_frames_agree():
    rec, eng, good = _facing_a_bet()
    gone = replace(good, seats=tuple(
        replace(s, folded=True) if s.seat == 2 else s for s in good.seats))
    assert acts(rec.step(gone, eng)) == []
    assert acts(rec.step(replace(gone), eng)) == [SeatAction(2, "fold", 0)]


def test_i6_exact_fold_source_is_trusted_on_the_first_frame():
    """PokerNow's DOM `fold` class does not flicker and frames only arrive on
    change — a second identical frame may never come."""
    rec = EventReconstructor(num_seats=2)
    eng = engine(1, [20, 0], [152, 172], bet_to_call=20, chips_per_cent=0.02,
                 min_bet_cents=100, exact_folds=True)
    rec.step(fs([1_000, None], [7_600, 8_600]), eng)
    folded = fs([1_000, None], [7_600, 8_600], folded=[False, True])
    assert acts(rec.step(folded, eng)) == [SeatAction(1, "fold", 0)]


def test_i6_hero_cards_unreadable_for_one_frame_is_not_a_check():
    """H12. The deleted Task-C branch read "all 5 hero cards visible → all 5
    hidden" as hero having acted. ClubGG never re-hides them mid-hand, so it
    only ever fired on a glitch — here with the timer bar STILL on hero."""
    rec = EventReconstructor(num_seats=6)
    eng = engine(0, [0] * 6, [500_000] * 6, bet_to_call=0)
    on_hero = [True, False, False, False, False, False]
    vis = fs([None] * 6, [100_000] * 6, actors=on_hero)
    rec.step(vis, eng)
    assert acts(rec.step(vis, eng)) == []
    hid = fs([None] * 6, [100_000] * 6, hero=(None,) * 5, actors=on_hero)
    assert acts(rec.step(hid, eng)) == []
    assert not hasattr(events_mod, "_hero_hole_just_hid")


# ===========================================================================
# I7 — the stack baseline must not eat evidence, nor count it twice
# ===========================================================================


def test_i7_snap_call_behind_a_raise_is_recovered_on_the_next_tick():
    """H2. One poll shows seat1's bet (banner), seat2's snap-fold and seat3's
    snap-call. The walk stops after the raise it just emitted. Pre-fix the
    baseline adopted seat3's lower stack, so once its banner faded Fix N had
    nothing to corroborate the oval with: the call was lost for good."""
    rec = EventReconstructor(num_seats=4)
    stk = 100_000
    eng0 = engine(1, [0] * 4, [stk * 5] * 4, bet_to_call=0)
    rec.step(fs([None] * 4, [stk] * 4, actors=[False, True, False, False]), eng0)

    t_n = fs([None, 18_000, None, 18_000], [stk, stk - 18_000, stk, stk - 18_000],
             banners=[False, True, False, True], folded=[False, False, True, False])
    assert acts(rec.step(t_n, eng0)) == [SeatAction(1, "raise", 90_000)]
    # Seat 3's drop is unexplained but corroborated (oval + banner): held.
    assert rec.last_fs.seats[3].stack_chips == stk
    assert rec.last_fs.seats[1].stack_chips == stk - 18_000   # explained: adopted

    eng1 = engine(2, [0, 90_000, 0, 0], [stk * 5, stk * 5 - 90_000, stk * 5, stk * 5],
                  bet_to_call=90_000)
    t_n1 = fs([None, 18_000, None, 18_000], [stk, stk - 18_000, stk, stk - 18_000],
              folded=[False, False, True, False])          # banners gone
    assert acts(rec.step(t_n1, eng1)) == [
        SeatAction(2, "fold", 0), SeatAction(3, "check_call", 0)]
    assert rec.last_fs.seats[3].stack_chips == stk - 18_000   # now explained


def test_i7_held_drop_survives_an_oval_that_flickers_to_none():
    """Evidence is remembered: the oval reading None for a tick (it does)
    must not drop the held stack while the walk still cannot reach the seat."""
    rec = EventReconstructor(num_seats=4)
    stk = 100_000
    eng0 = engine(1, [0] * 4, [stk * 5] * 4, bet_to_call=0)
    rec.step(fs([None] * 4, [stk] * 4), eng0)
    t_n = fs([None, 18_000, None, 18_000], [stk, stk - 18_000, stk, stk - 18_000],
             banners=[False, True, False, True])
    assert acts(rec.step(t_n, eng0)) == [SeatAction(1, "raise", 90_000)]

    # Seat 2 is still thinking (engine waits on it); seat 3's oval flickers out.
    eng1 = engine(2, [0, 90_000, 0, 0], [stk * 5, stk * 5 - 90_000, stk * 5, stk * 5],
                  bet_to_call=90_000)
    flick = fs([None, 18_000, None, None], [stk, stk - 18_000, stk, stk - 18_000])
    assert acts(rec.step(flick, eng1)) == []
    assert rec.last_fs.seats[3].stack_chips == stk            # still held

    # Seat 2 calls; the walk reaches seat 3 and its call is still provable.
    done = fs([None, 18_000, 18_000, 18_000],
              [stk, stk - 18_000, stk - 18_000, stk - 18_000],
              banners=[False, False, True, False])
    assert acts(rec.step(done, eng1)) == [
        SeatAction(2, "check_call", 0), SeatAction(3, "check_call", 0)]


def test_i7_drop_that_only_becomes_readable_after_the_banner_faded_is_held():
    """The realistic timing. While the blue banner is up it COVERS the stack
    label, so the drop is not readable on the banner ticks — it shows up on
    the tick the banner fades, and the oval may well read None on that same
    tick. The banner seen earlier is what makes that drop credible."""
    rec = EventReconstructor(num_seats=4)
    stk = 100_000
    eng0 = engine(1, [0] * 4, [stk * 5] * 4, bet_to_call=0)
    rec.step(fs([None] * 4, [stk] * 4), eng0)
    # t0: seat1 bets (explained). Seat3 snap-calls: banner up, stack label
    # covered (None), oval mid-animation (None). Walk stops behind the raise.
    t0 = fs([None, 18_000, None, None], [stk, stk - 18_000, stk, None],
            banners=[False, True, False, True])
    assert acts(rec.step(t0, eng0)) == [SeatAction(1, "raise", 90_000)]
    assert rec._unexplained_evidence_seats == {3}

    # t1: banners gone. Seat3's stack is readable now (dropped); oval still
    # None. Seat2 is thinking, so the walk cannot reach seat3 yet.
    eng1 = engine(2, [0, 90_000, 0, 0], [stk * 5, stk * 5 - 90_000, stk * 5, stk * 5],
                  bet_to_call=90_000)
    t1 = fs([None, 18_000, None, None], [stk, stk - 18_000, stk, stk - 18_000])
    assert acts(rec.step(t1, eng1)) == []
    assert rec.last_fs.seats[3].stack_chips == stk            # held, not absorbed

    # t2: seat2 calls; the walk reaches seat3 — recovered from the held drop
    # even though its oval never read.
    t2 = fs([None, 18_000, 18_000, None], [stk, stk - 18_000, stk - 18_000, stk - 18_000],
            banners=[False, False, True, False])
    assert acts(rec.step(t2, eng1)) == [
        SeatAction(2, "check_call", 0), SeatAction(3, "check_call", 0)]
    assert not rec._unexplained_evidence_seats
    assert rec.last_fs.seats[3].stack_chips == stk - 18_000


def test_i7_uncorroborated_drop_is_absorbed_so_a_misread_cannot_resurface():
    """The other half of the policy: a stack drop with NO banner and NO
    unexplained oval is absorbed exactly as before. Holding it would let a
    one-frame stack misread come back as a phantom bet whenever the walk next
    reached that seat."""
    rec = EventReconstructor(num_seats=3)
    # Hero (0) bet $180; seat 1 is thinking; seat 2 is still to act.
    eng = engine(1, [90_000, 0, 0], [410_000, 500_000, 500_000], bet_to_call=90_000)
    rec.step(fs([18_000, None, None], [82_000, 100_000, 450_500]), eng)   # seat 2 misread HIGH
    rec.step(fs([18_000, None, None], [82_000, 100_000, 45_050]), eng)    # ...reads right again
    assert rec.last_fs.seats[2].stack_chips == 45_050                     # not held at 450500
    assert not rec._unexplained_evidence_seats
    # Seat 1 folds (two frames); the walk reaches seat 2: no phantom action
    # out of the 405450-cent "drop".
    gone = fs([18_000, None, None], [82_000, 100_000, 45_050], folded=[False, True, False])
    assert acts(rec.step(gone, eng)) == []
    assert acts(rec.step(replace(gone), eng)) == [SeatAction(1, "fold", 0)]


def test_i7_carried_forward_stack_is_reduced_by_what_the_walk_explained():
    """H17. Tick A: seat1's call is explained by oval + banner while the banner
    covers its stack label (None). Pre-fix the baseline carried 100000 forward
    untouched, so tick B's readable 82000 looked like a NEW 18000 drop:
    CHECK(0) + RAISE(1) 90000 on a street where nothing happened."""
    rec = EventReconstructor(num_seats=6)
    eng_flop = engine(1, [90_000, 0, 0, 0, 0, 0], [410_000, 500_000, 0, 0, 0, 0],
                      bet_to_call=90_000, sitting_out=HU_SITTING_OUT, street=1, button=1)
    rec.step(fs([18_000, None, None, None, None, None],
                [82_000, 100_000, None, None, None, None]), eng_flop)
    t_a = fs([18_000, 18_000, None, None, None, None],
             [82_000, None, None, None, None, None],
             banners=[False, True, False, False, False, False])
    assert acts(rec.step(t_a, eng_flop)) == [SeatAction(1, "check_call", 0)]
    assert rec.last_fs.seats[1].stack_chips == 82_000          # 100000 - 18000

    eng_turn = engine(0, [0] * 6, [410_000, 410_000, 0, 0, 0, 0], bet_to_call=0,
                      sitting_out=HU_SITTING_OUT, street=2, button=1)
    t_b = fs([None] * 6, [82_000, 82_000, None, None, None, None],
             board_a=TURN_A, board_b=TURN_B)
    assert acts(rec.step(t_b, eng_turn)) == []


def test_i7_street_boundary_forgets_held_drops():
    """A drop still held when the next street is revealed is moot — the
    server's reveal reconcilers fill that street's gaps by count. Carrying it
    over would turn last street's call into this street's bet."""
    rec = EventReconstructor(num_seats=3)
    stk = 100_000
    # Seat1 bet $180; hero (0) is the engine's actor and silent, so the walk
    # cannot reach seat 2, whose snap-call is visible (oval + banner + drop).
    eng = engine(0, [0, 90_000, 0], [stk * 5, stk * 5 - 90_000, stk * 5], bet_to_call=90_000)
    rec.step(fs([None, 18_000, None], [stk, stk - 18_000, stk]), eng)
    rec.step(fs([None, 18_000, 18_000], [stk, stk - 18_000, stk - 18_000],
                banners=[False, False, True]), eng)
    assert rec.last_fs.seats[2].stack_chips == stk             # held behind hero
    assert rec._unexplained_evidence_seats == {2}

    turn = fs([None] * 3, [stk, stk - 18_000, stk - 18_000], board_a=TURN_A, board_b=TURN_B)
    ev = rec.step(turn, eng)
    assert [e for e in ev if isinstance(e, StreetReveal)]
    assert rec.last_fs.seats[2].stack_chips == stk - 18_000    # adopted
    assert not rec._unexplained_evidence_seats
    eng_turn = engine(2, [0, 0, 0], [stk * 5 - 90_000] * 3, bet_to_call=0, street=2)
    assert acts(rec.step(replace(turn), eng_turn)) == []       # no phantom turn bet


def test_i7_held_drop_on_a_seat_that_cannot_act_does_not_keep_the_walk_alive():
    """A held drop persists tick after tick by design, so it must not count as
    "someone downstream acted" when it sits on a folded / all-in seat — that
    would presume a CHECK for the live actor on every single tick."""
    rec = EventReconstructor(num_seats=3)
    stk = 100_000
    eng = engine(0, [0, 0, 0], [stk * 5] * 3, bet_to_call=0,
                 folded=[False, False, True])              # engine: seat 2 is out
    rec.step(fs([None] * 3, [stk] * 3), eng)
    noisy = fs([None, None, 18_000], [stk, stk, stk - 18_000], banners=[False, False, True])
    for _ in range(3):
        assert acts(rec.step(replace(noisy), eng)) == []
    assert rec.last_fs.seats[2].stack_chips == stk         # held, yet inert


def test_i7_rebaseline_and_reset_clear_reconstructor_tracking():
    rec = EventReconstructor(num_seats=2)
    rec._unexplained_evidence_seats = {1}
    rec._street_gate_ticks = 9
    rec.rebaseline(fs([None, None], [100_000, 100_000]))
    assert not rec._unexplained_evidence_seats and rec._street_gate_ticks == 0
    rec._unexplained_evidence_seats = {0}
    rec.reset()
    assert not rec._unexplained_evidence_seats and rec._last_engine_street is None


# ===========================================================================
# Reviewer F10 — timer-bar CHECK needs a readable, unchanged stack
# ===========================================================================

ON_HERO = [True, False, False, False, False, False]
ON_VILLAIN = [False, True, False, False, False, False]


def _timer_setup():
    rec = EventReconstructor(num_seats=6)
    eng = engine(0, [0] * 6, [500_000, 500_000, 0, 0, 0, 0], bet_to_call=0,
                 sitting_out=HU_SITTING_OUT)
    rec.step(fs([None] * 6, [100_000, 100_000, None, None, None, None], actors=ON_HERO), eng)
    return rec, eng


def test_f10_bar_moved_but_stack_unreadable_is_not_a_check():
    """H15. Hero BET; first post-action frame: stack label None, oval
    mid-animation, banner suppressed (hero's cards are face-up), bar already
    on the villain. Pre-fix: CHECK(0) — then the real bet landed as
    CHECK(1) + RAISE(0) against an engine that had moved on."""
    rec, eng = _timer_setup()
    blind = fs([None] * 6, [None, 100_000, None, None, None, None], actors=ON_VILLAIN)
    assert acts(rec.step(blind, eng)) == []
    # Evidence readable next tick; the engine is still on hero → the bet is his.
    readable = fs([18_000, None, None, None, None, None],
                  [82_000, 100_000, None, None, None, None], actors=ON_VILLAIN)
    assert acts(rec.step(readable, eng)) == [SeatAction(0, "raise", 90_000)]


def test_f10_real_check_with_a_late_stack_read_is_still_recovered():
    """The bar transition is a one-shot edge. When the stack is unreadable on
    that tick the lock is HELD, so the same transition is judged again once
    the stack reads — unchanged ⇒ the CHECK is emitted one tick late instead
    of never."""
    rec, eng = _timer_setup()
    blind = fs([None] * 6, [None, 100_000, None, None, None, None], actors=ON_VILLAIN)
    assert acts(rec.step(blind, eng)) == []
    assert rec._observed_active_actor == 0                     # lock held on hero
    readable = fs([None] * 6, [100_000, 100_000, None, None, None, None], actors=ON_VILLAIN)
    assert acts(rec.step(readable, eng)) == [SeatAction(0, "check_call", 0)]
    assert rec._observed_active_actor == 1                     # and released


def test_f10_sub_1bb_stack_jitter_does_not_block_the_check():
    rec, eng = _timer_setup()
    jitter = fs([None] * 6, [99_987, 100_000, None, None, None, None], actors=ON_VILLAIN)
    assert acts(rec.step(jitter, eng)) == [SeatAction(0, "check_call", 0)]


# ===========================================================================
# Rounding — one conversion, and a 1-chip mismatch is the call
# ===========================================================================


def test_rounding_call_recovered_via_stack_drop_at_a_fractional_scale():
    """H13. $40/bb → 2.5 chips/cent. base 3001c → 7502 (half-even), facing
    9002c → 22505, drop 6001c → 15002: 7502 + 15002 = 22504 ≠ 22505. Pre-fix
    the call fell through to Fix Q and was never emitted."""
    s = 2.5
    c1, c2 = 3001, 9002
    base, facing = cents_to_engine_chips(c1, s), cents_to_engine_chips(c2, s)
    assert base + cents_to_engine_chips(c2 - c1, s) == facing - 1   # the mismatch
    rec = EventReconstructor(num_seats=3)
    eng = engine(0, [base, facing, 0], [10**6] * 3, bet_to_call=facing, chips_per_cent=s,
                 min_bet_cents=4000, folded=[False, False, True])
    rec.step(fs([c1, c2, None], [100_000, 100_000, None], folded=[False, False, True]), eng)
    t = fs([None, c2, None], [100_000 - (c2 - c1), 100_000, None], folded=[False, False, True])
    assert acts(rec.step(t, eng)) == [SeatAction(0, "check_call", 0)]


@pytest.mark.parametrize("dpb", [20.0, 2.0, 40.0, 3.0, 200.0, 15.0, 7.0])
def test_rounding_every_split_of_a_call_lands_within_the_tolerance(dpb):
    s = 10_000.0 / (100.0 * dpb)
    rng = random.Random(1)
    for _ in range(5_000):
        c1 = rng.randrange(100, 50_000)
        c2 = c1 + rng.randrange(c1, 3 * c1)
        split = cents_to_engine_chips(c1, s) + cents_to_engine_chips(c2 - c1, s)
        assert abs(split - cents_to_engine_chips(c2, s)) <= events_mod._CALL_ROUNDING_TOLERANCE_CHIPS


def test_rounding_a_real_raise_is_not_swallowed_by_the_tolerance():
    rec = EventReconstructor(num_seats=2)
    eng = engine(0, [0, 90_000], [500_000, 410_000], bet_to_call=90_000)
    rec.step(fs([None, 18_000], [100_000, 82_000]), eng)
    min_raise = fs([36_000, 18_000], [64_000, 82_000], banners=[True, False])
    assert acts(rec.step(min_raise, eng)) == [SeatAction(0, "raise", 180_000)]


@pytest.mark.parametrize("bb, dpb", [(10_000, 20.0), (10_000, 40.0), (10_000, 3.0),
                                     (10_000, 15.0), (10_000, 200.0), (2, 1.0), (4, 0.5)])
def test_rounding_matches_the_server_side_conversion(bb, dpb):
    """`server._ocr_cents_to_engine_chips` multiplies before it divides
    (`cents * bb / (100 * dpb)`); the reconstructor multiplies by the
    precomputed ratio. They must agree on every amount, or a stack seeded by
    the server and a bet derived here disagree by a chip. (The server should
    import `cents_to_engine_chips` — see the cross-ownership note.)"""
    ratio = float(bb) / (100.0 * dpb)
    for cents in list(range(0, 5_000)) + list(range(5_000, 3_000_000, 997)):
        server_side = int(round(int(cents) * float(bb) / (100.0 * dpb)))
        assert cents_to_engine_chips(cents, ratio) == server_side, cents


# ===========================================================================
# Latent — no board on screen ⇒ it is the ante, not a bet (H14)
# ===========================================================================


def test_preflop_ante_deduction_is_not_read_as_a_bet_and_calls():
    """The OCR hand-start can anchor BEFORE ClubGG deducts the bomb-pot antes.
    The uniform $60 drop that follows used to read as BET + CALL + CALL + CALL
    and carry the hand to the turn before the flop was even dealt."""
    rec = EventReconstructor(num_seats=4)
    eng = engine(1, [0] * 4, [500_000] * 4, bet_to_call=0)
    pre = fs([None] * 4, [100_000] * 4, board_a=NO_BOARD, board_b=NO_BOARD, hero=(None,) * 5)
    rec.rebaseline(pre)
    post = fs([None] * 4, [94_000] * 4, board_a=NO_BOARD, board_b=NO_BOARD, hero=(None,) * 5)
    assert acts(rec.step(post, eng)) == []
    assert [s.stack_chips for s in rec.last_fs.seats] == [94_000] * 4   # absorbed
    flop = fs([None] * 4, [94_000] * 4)
    assert acts(rec.step(flop, eng)) == []
    bet = fs([None, 18_000, None, None], [94_000, 76_000, 94_000, 94_000],
             banners=[False, True, False, False])
    assert acts(rec.step(bet, eng)) == [SeatAction(1, "raise", 90_000)]


def test_one_unreadable_board_card_does_not_close_the_no_board_gate():
    """The gate is deliberately the weakest test that works: ANY readable card
    opens it, so a flop card that never classifies cannot block inference."""
    rec = EventReconstructor(num_seats=2)
    eng = engine(1, [0, 0], [500_000, 500_000], bet_to_call=0)
    ragged_a = (None, _c(8, 0), None, None, None)
    rec.step(fs([None, None], [100_000, 100_000], board_a=ragged_a, board_b=NO_BOARD), eng)
    bet = fs([None, 18_000], [100_000, 82_000], board_a=ragged_a, board_b=NO_BOARD,
             banners=[False, True])
    assert acts(rec.step(bet, eng)) == [SeatAction(1, "raise", 90_000)]
