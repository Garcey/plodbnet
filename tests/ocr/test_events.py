"""Unit tests for the event reconstructor.

Synthetic FrameState sequences — no frames are read. Each test walks a
sequence of (FrameState, EngineView) through `EventReconstructor.step`
and asserts the emitted events match the scripted scenario.
"""

from __future__ import annotations

from plo5bp.ocr.events import (
    EngineView,
    EventReconstructor,
    HeroHoleRevealed,
    SeatAction,
    StreetReveal,
)
from plo5bp.ocr.types import Card, FrameState, SeatObs


def _c(rank: int, suit: int) -> Card:
    return Card(rank=rank, suit=suit)


def _flop_a() -> tuple[Card | None, ...]:
    return (_c(10, 0), _c(8, 0), _c(11, 0), None, None)  # Kc-ish set


def _flop_b() -> tuple[Card | None, ...]:
    return (_c(5, 1), _c(7, 2), _c(1, 1), None, None)


def _hero() -> tuple[Card | None, ...]:
    return (_c(8, 1), _c(7, 1), _c(7, 0), _c(3, 2), _c(1, 2))


def _seats(
    commits: list[int | None],
    stacks: list[int | None],
    *,
    banners: list[bool] | None = None,
    actors: list[bool] | None = None,
) -> tuple[SeatObs, ...]:
    n = len(commits)
    banners = banners if banners is not None else [False] * n
    actors = actors if actors is not None else [False] * n
    return tuple(
        SeatObs(
            seat=i,
            stack_chips=stacks[i],
            committed_chips=commits[i],
            bet_banner=banners[i],
            is_actor=actors[i],
        )
        for i in range(n)
    )


def _fs(
    commits: list[int | None],
    stacks: list[int | None],
    *,
    pot: int = 70_000,
    hero: tuple[Card | None, ...] | None = None,
    board_a: tuple[Card | None, ...] | None = None,
    board_b: tuple[Card | None, ...] | None = None,
    button: int | None = 0,
    banners: list[bool] | None = None,
    actors: list[bool] | None = None,
) -> FrameState:
    return FrameState(
        board_a=board_a if board_a is not None else _flop_a(),
        board_b=board_b if board_b is not None else _flop_b(),
        hero_hole=hero if hero is not None else _hero(),
        button_seat=button,
        pot_total_chips=pot,
        seats=_seats(commits, stacks, banners=banners, actors=actors),
    )


def _engine(
    current_actor: int | None,
    commits: list[int],
    stacks: list[int],
    *,
    street: int = 1,
    folded: list[bool] | None = None,
    all_in: list[bool] | None = None,
    bet_to_call: int | None = None,
    awaiting: int | None = None,
    button: int = 0,
    chips_per_cent: float = 1.0,
    min_bet_cents: int = 0,
    sitting_out: list[bool] | None = None,
    exact_folds: bool = False,
) -> EngineView:
    n = len(commits)
    folded = folded if folded is not None else [False] * n
    all_in = all_in if all_in is not None else [False] * n
    sitting_out = sitting_out if sitting_out is not None else [False] * n
    btc = bet_to_call if bet_to_call is not None else max(commits)
    return EngineView(
        num_seats=n,
        current_actor=current_actor,
        street=street,
        awaiting_next_street=awaiting,
        button_seat=button,
        committed_this_street=tuple(commits),
        stacks=tuple(stacks),
        folded=tuple(folded),
        all_in=tuple(all_in),
        bet_to_call=btc,
        chips_per_cent=chips_per_cent,
        min_bet_cents=min_bet_cents,
        sitting_out=tuple(sitting_out),
        exact_folds=exact_folds,
    )


# --- bootstrap / hand start ---------------------------------------------


def test_bootstrap_emits_hero_only():
    # Server-side now owns hand-start detection — reconstructor just
    # adopts the first frame as its baseline and emits HeroHoleRevealed
    # if hero hole is fully visible.
    rec = EventReconstructor(num_seats=6)
    fs = _fs([0] * 6, [30_000] * 6)
    ev = rec.step(fs, _engine(current_actor=1, commits=[0] * 6, stacks=[30_000] * 6))
    assert any(isinstance(e, HeroHoleRevealed) for e in ev)
    hole = next(e for e in ev if isinstance(e, HeroHoleRevealed)).cards
    assert hole == (8 * 4 + 1, 7 * 4 + 1, 7 * 4 + 0, 3 * 4 + 2, 1 * 4 + 2)


def test_bootstrap_without_hero_emits_nothing():
    rec = EventReconstructor(num_seats=6)
    fs = _fs([0] * 6, [30_000] * 6, hero=(None,) * 5)
    ev = rec.step(fs, _engine(current_actor=1, commits=[0] * 6, stacks=[30_000] * 6))
    assert ev == []


# --- sequential play: CHECK, BET, CALL, RAISE ---------------------------


def test_sequential_check_then_bet_then_call():
    rec = EventReconstructor(num_seats=6)
    # Bootstrap with the yellow timer bar on seat 1 (the actor).
    base = _fs(
        [0] * 6, [30_000] * 6,
        actors=[False, True, False, False, False, False],
    )
    rec.step(base, _engine(current_actor=1, commits=[0] * 6, stacks=[30_000] * 6))

    # Poll 2: seat 1 checks. The bar moves to seat 2 — that's the
    # positive corroboration the walk needs (branch 4 no longer fires
    # for the engine's own `current_actor` when their oval reads 0 ==
    # base_commit, since "no chip change" can't distinguish "checked"
    # from "still thinking").
    fs2 = _fs(
        [0] * 6, [30_000] * 6,
        actors=[False, False, True, False, False, False],
    )
    ev2 = rec.step(fs2, _engine(current_actor=1, commits=[0] * 6, stacks=[30_000] * 6))
    seat_actions = [e for e in ev2 if isinstance(e, SeatAction)]
    assert seat_actions == [SeatAction(seat=1, gate="check_call", chips=0)]

    # Poll 3: seat 2 bets 10_000, seat 3 folds (no change with facing bet).
    fs3 = _fs([0, 0, 10_000, 0, 0, 0], [30_000, 30_000, 20_000, 30_000, 30_000, 30_000])
    ev3 = rec.step(
        fs3,
        _engine(current_actor=2, commits=[0] * 6, stacks=[30_000] * 6),
    )
    seat_actions = [e for e in ev3 if isinstance(e, SeatAction)]
    assert SeatAction(seat=2, gate="raise", chips=10_000) in seat_actions

    # Poll 4: seat 3 calls.
    fs4 = _fs(
        [0, 0, 10_000, 10_000, 0, 0],
        [30_000, 30_000, 20_000, 20_000, 30_000, 30_000],
    )
    ev4 = rec.step(
        fs4,
        _engine(
            current_actor=3,
            commits=[0, 0, 10_000, 0, 0, 0],
            stacks=[30_000, 30_000, 20_000, 30_000, 30_000, 30_000],
            bet_to_call=10_000,
        ),
    )
    seat_actions = [e for e in ev4 if isinstance(e, SeatAction)]
    assert SeatAction(seat=3, gate="check_call", chips=0) in seat_actions


# --- gap-fill: missed middle action -------------------------------------


def test_gap_fill_missed_bet_then_call():
    """We miss seat 2's bet poll and only see the frame after seat 3 has
    called. Reconstructor should explain both deltas from current_actor=2."""
    rec = EventReconstructor(num_seats=6)
    base = _fs([0] * 6, [30_000] * 6)
    rec.step(base, _engine(current_actor=2, commits=[0] * 6, stacks=[30_000] * 6))

    # Single poll observes seat 2's bet AND seat 3's call.
    fs = _fs(
        [0, 0, 10_000, 10_000, 0, 0],
        [30_000, 30_000, 20_000, 20_000, 30_000, 30_000],
    )
    ev = rec.step(
        fs,
        _engine(current_actor=2, commits=[0] * 6, stacks=[30_000] * 6),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert SeatAction(seat=2, gate="raise", chips=10_000) in seat_actions
    assert SeatAction(seat=3, gate="check_call", chips=0) in seat_actions


def test_gap_fill_missed_check_before_bet():
    """We miss seat 2's check and only see the frame after seat 3 bet.
    Reconstructor should infer CHECK for seat 2 and BET for seat 3."""
    rec = EventReconstructor(num_seats=6)
    base = _fs([0] * 6, [30_000] * 6)
    rec.step(base, _engine(current_actor=2, commits=[0] * 6, stacks=[30_000] * 6))

    fs = _fs(
        [0, 0, 0, 15_000, 0, 0],
        [30_000, 30_000, 30_000, 15_000, 30_000, 30_000],
    )
    ev = rec.step(
        fs,
        _engine(current_actor=2, commits=[0] * 6, stacks=[30_000] * 6),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert SeatAction(seat=2, gate="check_call", chips=0) in seat_actions
    assert SeatAction(seat=3, gate="raise", chips=15_000) in seat_actions


# --- Fix J: positive fold evidence --------------------------------------


def test_snap_fold_chain_via_obs_folded_signal():
    """Fix J: when seats snap-fold between polls, Tesseract typically
    returns None for the empty chip oval — NOT 0. The reconstructor
    must still emit FOLDs for those seats based on the positive
    `SeatObs.folded` signal (cards_back gone, no banner, no commit).

    Scenario: 4 seats. Seat 0 (hero) bet 10_000 on the flop. Current
    actor is seat 1. Between polls all three remaining opponents
    (seats 1, 2, 3) snap-fold. Walk starts at seat 1 and must emit
    FOLD for each even though every folded seat's committed_chips
    reads None.
    """
    rec = EventReconstructor(num_seats=4)

    # Baseline: everyone in hand, hero has bet 10_000.
    base_seats = tuple(
        SeatObs(
            seat=i,
            stack_chips=20_000 if i == 0 else 30_000,
            committed_chips=10_000 if i == 0 else 0,
            folded=False,
            bet_banner=False,
        )
        for i in range(4)
    )
    base = FrameState(
        board_a=_flop_a(),
        board_b=_flop_b(),
        hero_hole=_hero(),
        button_seat=3,
        pot_total_chips=10_000,
        seats=base_seats,
    )
    rec.step(
        base,
        _engine(
            current_actor=1,
            commits=[10_000, 0, 0, 0],
            stacks=[20_000, 30_000, 30_000, 30_000],
            bet_to_call=10_000,
            button=3,
        ),
    )

    # Snap-fold tick: seats 1/2/3 all folded, cards gone, no banner,
    # chip oval reads None (Tesseract returning None on an empty crop).
    # Stacks unchanged because they never committed chips.
    fold_seats = (
        SeatObs(
            seat=0,
            stack_chips=20_000,
            committed_chips=10_000,
            folded=False,
            bet_banner=False,
        ),
        SeatObs(
            seat=1,
            stack_chips=30_000,
            committed_chips=None,
            folded=True,
            bet_banner=False,
        ),
        SeatObs(
            seat=2,
            stack_chips=30_000,
            committed_chips=None,
            folded=True,
            bet_banner=False,
        ),
        SeatObs(
            seat=3,
            stack_chips=30_000,
            committed_chips=None,
            folded=True,
            bet_banner=False,
        ),
    )
    snap_fold = FrameState(
        board_a=_flop_a(),
        board_b=_flop_b(),
        hero_hole=_hero(),
        button_seat=3,
        pot_total_chips=10_000,
        seats=fold_seats,
    )
    ev = rec.step(
        snap_fold,
        _engine(
            current_actor=1,
            commits=[10_000, 0, 0, 0],
            stacks=[20_000, 30_000, 30_000, 30_000],
            bet_to_call=10_000,
            button=3,
        ),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    # Walk may also coast into hero seat with a phantom check (hero
    # already matched the bet so delta==0, to_call==0). That's a
    # harmless side effect for this test — assert the three FOLDs
    # land in order.
    assert seat_actions[:3] == [
        SeatAction(seat=1, gate="fold", chips=0),
        SeatAction(seat=2, gate="fold", chips=0),
        SeatAction(seat=3, gate="fold", chips=0),
    ]


def test_folded_signal_ignored_mid_raise_pass():
    """Fix J must NOT emit FOLD for a seat whose `obs.folded=True`
    reading arrived in the same pass as a raise we just inferred.
    Downstream seats haven't had a turn yet — the transient folded
    reading is noise (banner flicker, occlusion). The `facing_bet_bumped`
    guard protects this.
    """
    rec = EventReconstructor(num_seats=4)

    base_seats = tuple(
        SeatObs(
            seat=i,
            stack_chips=30_000,
            committed_chips=0,
            folded=False,
            bet_banner=False,
        )
        for i in range(4)
    )
    base = FrameState(
        board_a=_flop_a(),
        board_b=_flop_b(),
        hero_hole=_hero(),
        button_seat=3,
        pot_total_chips=0,
        seats=base_seats,
    )
    rec.step(
        base,
        _engine(
            current_actor=0,
            commits=[0, 0, 0, 0],
            stacks=[30_000, 30_000, 30_000, 30_000],
            bet_to_call=0,
            button=3,
        ),
    )

    # Hero bets 10_000. Simultaneously seat 2 momentarily reads
    # folded=True (banner flicker or transition artifact). Seat 2
    # has NOT actually folded and is still waiting to act.
    mixed = (
        SeatObs(seat=0, stack_chips=20_000, committed_chips=10_000, folded=False, bet_banner=True),
        SeatObs(seat=1, stack_chips=30_000, committed_chips=0, folded=False, bet_banner=False),
        SeatObs(seat=2, stack_chips=30_000, committed_chips=None, folded=True, bet_banner=False),
        SeatObs(seat=3, stack_chips=30_000, committed_chips=0, folded=False, bet_banner=False),
    )
    fs = FrameState(
        board_a=_flop_a(),
        board_b=_flop_b(),
        hero_hole=_hero(),
        button_seat=3,
        pot_total_chips=10_000,
        seats=mixed,
    )
    ev = rec.step(
        fs,
        _engine(
            current_actor=0,
            commits=[0, 0, 0, 0],
            stacks=[30_000, 30_000, 30_000, 30_000],
            bet_to_call=0,
            button=3,
        ),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    # Hero's raise should land; seat 2 must NOT be folded (still waiting).
    assert SeatAction(seat=0, gate="raise", chips=10_000) in seat_actions
    assert all(
        not (e.seat == 2 and e.gate == "fold") for e in seat_actions
    )


# --- street reveal -------------------------------------------------------


def test_turn_reveal_both_boards():
    rec = EventReconstructor(num_seats=6)
    base = _fs([0] * 6, [30_000] * 6)
    rec.step(base, _engine(current_actor=1, commits=[0] * 6, stacks=[30_000] * 6))

    # Next poll: turn card on both boards (slot index 3).
    turn_a = _flop_a()[:3] + (_c(2, 3), None)  # some random turn
    turn_b = _flop_b()[:3] + (_c(9, 2), None)
    fs = _fs([0] * 6, [30_000] * 6, board_a=tuple(turn_a), board_b=tuple(turn_b))
    ev = rec.step(
        fs,
        _engine(
            current_actor=1,
            commits=[0] * 6,
            stacks=[30_000] * 6,
            street=2,
        ),
    )
    reveals = [e for e in ev if isinstance(e, StreetReveal)]
    assert len(reveals) == 1
    assert reveals[0].street == "turn"
    assert reveals[0].board_a_card == 2 * 4 + 3
    assert reveals[0].board_b_card == 9 * 4 + 2


def test_river_reveal_only_after_both_boards_show():
    rec = EventReconstructor(num_seats=6)
    turn_a = _flop_a()[:3] + (_c(2, 3), None)
    turn_b = _flop_b()[:3] + (_c(9, 2), None)
    base = _fs(
        [0] * 6, [30_000] * 6, board_a=tuple(turn_a), board_b=tuple(turn_b)
    )
    rec.step(base, _engine(current_actor=1, commits=[0] * 6, stacks=[30_000] * 6, street=2))

    # Partial poll — board A river but board B still 4 cards. No reveal yet.
    river_a_partial = turn_a[:3] + (_c(2, 3), _c(12, 0))
    fs1 = _fs(
        [0] * 6, [30_000] * 6, board_a=tuple(river_a_partial), board_b=tuple(turn_b)
    )
    ev1 = rec.step(
        fs1,
        _engine(current_actor=1, commits=[0] * 6, stacks=[30_000] * 6, street=2),
    )
    assert not any(isinstance(e, StreetReveal) for e in ev1)

    # Next poll catches board B river. Emit the pair.
    river_b = turn_b[:3] + (_c(9, 2), _c(0, 3))
    fs2 = _fs(
        [0] * 6, [30_000] * 6, board_a=tuple(river_a_partial), board_b=tuple(river_b)
    )
    ev2 = rec.step(
        fs2,
        _engine(current_actor=1, commits=[0] * 6, stacks=[30_000] * 6, street=3),
    )
    reveals = [e for e in ev2 if isinstance(e, StreetReveal)]
    assert len(reveals) == 1
    assert reveals[0].street == "river"
    assert reveals[0].board_a_card == 12 * 4 + 0
    assert reveals[0].board_b_card == 0 * 4 + 3


# --- rebaseline (used by server-side hand-start detection) --------------


def test_rebaseline_adopts_frame_without_emitting():
    # Server calls rebaseline() when it commits a new hand — this
    # flushes the reconstructor's last_fs so diff inference starts
    # fresh, but it must NOT emit any events (the server already
    # wrote the corresponding state via _begin_new_hand).
    rec = EventReconstructor(num_seats=6)
    base = _fs([0] * 6, [30_000] * 6)
    rec.step(base, _engine(current_actor=1, commits=[0] * 6, stacks=[30_000] * 6))

    fresh = _fs([0] * 6, [30_000] * 6, button=1)
    rec.rebaseline(fresh)
    assert rec.last_fs is fresh
    assert rec.hero_hole_emitted is True  # hero hole fully visible

    # Subsequent step should NOT re-emit HeroHoleRevealed.
    ev = rec.step(fresh, _engine(current_actor=1, commits=[0] * 6, stacks=[30_000] * 6))
    assert not any(isinstance(e, HeroHoleRevealed) for e in ev)


def test_rebaseline_without_hero_clears_emitted_flag():
    rec = EventReconstructor(num_seats=6)
    base = _fs([0] * 6, [30_000] * 6)
    rec.step(base, _engine(current_actor=1, commits=[0] * 6, stacks=[30_000] * 6))
    assert rec.hero_hole_emitted is True

    fresh = _fs([0] * 6, [30_000] * 6, hero=(None,) * 5)
    rec.rebaseline(fresh)
    assert rec.hero_hole_emitted is False


# --- all-in detection ---------------------------------------------------


def test_allin_detected_when_stack_reaches_zero():
    rec = EventReconstructor(num_seats=6)
    base = _fs([0] * 6, [30_000] * 6)
    rec.step(base, _engine(current_actor=2, commits=[0] * 6, stacks=[30_000] * 6))

    # Seat 2 shoves: commit = entire stack, new stack = 0.
    fs = _fs(
        [0, 0, 30_000, 0, 0, 0],
        [30_000, 30_000, 0, 30_000, 30_000, 30_000],
    )
    ev = rec.step(
        fs,
        _engine(current_actor=2, commits=[0] * 6, stacks=[30_000] * 6),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert SeatAction(seat=2, gate="raise", chips=30_000) in seat_actions


# --- reset clears state -------------------------------------------------


def test_reset_clears_baseline_and_emits_hero_on_next_step():
    rec = EventReconstructor(num_seats=6)
    base = _fs([0] * 6, [30_000] * 6)
    rec.step(base, _engine(current_actor=1, commits=[0] * 6, stacks=[30_000] * 6))

    rec.reset()
    assert rec.last_fs is None
    assert rec.hero_hole_emitted is False

    ev = rec.step(base, _engine(current_actor=1, commits=[0] * 6, stacks=[30_000] * 6))
    assert any(isinstance(e, HeroHoleRevealed) for e in ev)


# --- engine-chip unit conversion at the reconstructor boundary ----------


def test_opening_bet_converts_cents_to_engine_chip_delta():
    # SB leads out $180 on the flop. OCR reads 18000 cents. At defaults
    # (bb=10000, $20/bb), chips_per_cent=5.0 → 90000 engine-chips.
    # Base_commit is 0, so delta == new_commit == 90000.
    rec = EventReconstructor(num_seats=6)
    base = _fs([0] * 6, [200_000] * 6)
    rec.step(
        base,
        _engine(
            current_actor=1,
            commits=[0] * 6,
            stacks=[200_000] * 6,
            chips_per_cent=5.0,
        ),
    )

    fs = _fs([0, 18_000, 0, 0, 0, 0], [200_000, 182_000, 200_000, 200_000, 200_000, 200_000])
    ev = rec.step(
        fs,
        _engine(
            current_actor=1,
            commits=[0] * 6,
            stacks=[200_000] * 6,
            chips_per_cent=5.0,
        ),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert SeatAction(seat=1, gate="raise", chips=90_000) in seat_actions


def test_fold_suppression_after_raise_in_same_pass():
    # After SB's $180 raise bumps facing_bet, the next seat (downstream
    # in the betting order) has committed_chips=0 and is waiting to
    # act — NOT folded. The reconstructor must break instead of
    # speculating a FOLD.
    rec = EventReconstructor(num_seats=6)
    base = _fs([0] * 6, [200_000] * 6)
    rec.step(
        base,
        _engine(
            current_actor=1,
            commits=[0] * 6,
            stacks=[200_000] * 6,
            chips_per_cent=5.0,
        ),
    )

    # Only seat 1 acted; seats 2..5 are untouched (waiting).
    fs = _fs([0, 18_000, 0, 0, 0, 0], [200_000, 182_000, 200_000, 200_000, 200_000, 200_000])
    ev = rec.step(
        fs,
        _engine(
            current_actor=1,
            commits=[0] * 6,
            stacks=[200_000] * 6,
            chips_per_cent=5.0,
        ),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    # Exactly one event: the SB raise. No spurious folds for seats 2..5.
    assert seat_actions == [SeatAction(seat=1, gate="raise", chips=90_000)]


def test_facing_bet_with_cards_visible_waits_for_fix_j():
    """Fix P: when a seat is current_actor facing a bet with
    committed_chips=0, base_commit=0, and cards still visible
    (obs.folded=False), the reconstructor does NOT infer a FOLD from
    the ladder — the state is ambiguous between "still thinking" and
    "folded but cards_back hasn't disappeared yet". Instead the walk
    breaks and waits. When the real fold lands, cards disappear
    (obs.folded=True) and Fix J emits the FOLD authoritatively.
    """
    rec = EventReconstructor(num_seats=6)
    base = _fs(
        [0, 18_000, 0, 0, 0, 0],
        [200_000, 182_000, 200_000, 200_000, 200_000, 200_000],
    )
    rec.step(
        base,
        _engine(
            current_actor=0,
            commits=[0, 90_000, 0, 0, 0, 0],
            stacks=[200_000, 182_000, 200_000, 200_000, 200_000, 200_000],
            bet_to_call=90_000,
            chips_per_cent=5.0,
        ),
    )

    # Tick A: hero's commit still 0, cards still visible. Ambiguous;
    # Fix P breaks the walk without emitting FOLD.
    fs_a = _fs(
        [0, 18_000, 0, 0, 0, 0],
        [200_000, 182_000, 200_000, 200_000, 200_000, 200_000],
    )
    ev_a = rec.step(
        fs_a,
        _engine(
            current_actor=0,
            commits=[0, 90_000, 0, 0, 0, 0],
            stacks=[200_000, 182_000, 200_000, 200_000, 200_000, 200_000],
            bet_to_call=90_000,
            chips_per_cent=5.0,
        ),
    )
    sa_a = [e for e in ev_a if isinstance(e, SeatAction)]
    assert not any(a.seat == 0 for a in sa_a), (
        f"expected no inferred FOLD on seat 0 (ambiguous), got {sa_a}"
    )

    # Tick B: hero's cards disappear — obs.folded=True. Fix J's
    # positive-fold branch fires authoritatively.
    fold_seats = list(fs_a.seats)
    fold_seats[0] = SeatObs(
        seat=0, stack_chips=200_000, committed_chips=None,
        folded=True, bet_banner=False,
    )
    fs_b = FrameState(
        board_a=fs_a.board_a, board_b=fs_a.board_b,
        hero_hole=fs_a.hero_hole, button_seat=fs_a.button_seat,
        pot_total_chips=fs_a.pot_total_chips, seats=tuple(fold_seats),
    )
    ev_b = rec.step(
        fs_b,
        _engine(
            current_actor=0,
            commits=[0, 90_000, 0, 0, 0, 0],
            stacks=[200_000, 182_000, 200_000, 200_000, 200_000, 200_000],
            bet_to_call=90_000,
            chips_per_cent=5.0,
        ),
    )
    sa_b = [e for e in ev_b if isinstance(e, SeatAction)]
    assert SeatAction(seat=0, gate="fold", chips=0) in sa_b


def test_reraise_emits_delta_not_total():
    # SB opens for $180 (engine-chip total 90_000). Next poll: BB
    # re-raises to $500 (50_000 cents → 250_000 engine-chips total).
    # Engine state reflects SB's 90_000; BB's base_commit is 0.
    # Emitted delta for BB = 250_000 - 0 = 250_000 (raise-by).
    rec = EventReconstructor(num_seats=6)
    base = _fs(
        [0, 18_000, 0, 0, 0, 0],
        [200_000, 182_000, 200_000, 200_000, 200_000, 200_000],
    )
    rec.step(
        base,
        _engine(
            current_actor=2,
            commits=[0, 90_000, 0, 0, 0, 0],
            stacks=[200_000, 182_000, 200_000, 200_000, 200_000, 200_000],
            bet_to_call=90_000,
            chips_per_cent=5.0,
        ),
    )

    fs = _fs(
        [0, 18_000, 50_000, 0, 0, 0],
        [200_000, 182_000, 150_000, 200_000, 200_000, 200_000],
    )
    ev = rec.step(
        fs,
        _engine(
            current_actor=2,
            commits=[0, 90_000, 0, 0, 0, 0],
            stacks=[200_000, 182_000, 200_000, 200_000, 200_000, 200_000],
            bet_to_call=90_000,
            chips_per_cent=5.0,
        ),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert SeatAction(seat=2, gate="raise", chips=250_000) in seat_actions


# --- multi-signal fallback: stack-delta / bet-banner --------------------


def test_stack_delta_fallback_recovers_bet_when_committed_ocr_misses():
    # SB bet $180 but `committed_chips` OCR returned None (ROI miss).
    # Stack read succeeds: went from 163513¢ → 145513¢ between polls.
    # Drop exceeds 1 bb (2000¢ min_bet_cents), so the reconstructor
    # derives new_commit from the stack delta and emits the raise.
    rec = EventReconstructor(num_seats=6)
    base = _fs(
        [0, 0, 0, 0, 0, 0],
        [163_513, 200_000, 200_000, 200_000, 200_000, 200_000],
    )
    rec.step(
        base,
        _engine(
            current_actor=0,
            commits=[0] * 6,
            stacks=[163_513 * 5] + [200_000 * 5] * 5,
            chips_per_cent=5.0,
            min_bet_cents=2_000,
        ),
    )

    fs = _fs(
        [None, 0, 0, 0, 0, 0],
        [145_513, 200_000, 200_000, 200_000, 200_000, 200_000],
    )
    ev = rec.step(
        fs,
        _engine(
            current_actor=0,
            commits=[0] * 6,
            stacks=[163_513 * 5] + [200_000 * 5] * 5,
            chips_per_cent=5.0,
            min_bet_cents=2_000,
        ),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    # 163513 - 145513 = 18000 cents × 5 chips/cent = 90_000 engine-chips.
    assert SeatAction(seat=0, gate="raise", chips=90_000) in seat_actions


def test_banner_gates_small_stack_drop_below_noise_floor():
    # Stack drop is 500¢ — below min_bet_cents (2000¢) — but bet banner
    # is visible. Banner overrides the noise floor: we still derive the
    # bet amount from the stack delta. (Unusual in practice since real
    # bets clear 1 bb, but keeps banner + small-stack scenarios sane.)
    rec = EventReconstructor(num_seats=6)
    base = _fs(
        [0, 0, 0, 0, 0, 0],
        [200_000, 100_500, 200_000, 200_000, 200_000, 200_000],
    )
    rec.step(
        base,
        _engine(
            current_actor=1,
            commits=[0] * 6,
            stacks=[200_000 * 5, 100_500 * 5] + [200_000 * 5] * 4,
            chips_per_cent=5.0,
            min_bet_cents=2_000,
        ),
    )

    fs = _fs(
        [0, None, 0, 0, 0, 0],
        [200_000, 100_000, 200_000, 200_000, 200_000, 200_000],
        banners=[False, True, False, False, False, False],
    )
    ev = rec.step(
        fs,
        _engine(
            current_actor=1,
            commits=[0] * 6,
            stacks=[200_000 * 5, 100_500 * 5] + [200_000 * 5] * 4,
            chips_per_cent=5.0,
            min_bet_cents=2_000,
        ),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    # 500¢ × 5 = 2500 engine-chips. Delta > 0, facing_bet was 0, so raise.
    assert SeatAction(seat=1, gate="raise", chips=2_500) in seat_actions


def test_glitch_guard_rejects_tiny_stack_drop_without_banner():
    # 100¢ stack drop (pure Tesseract jitter like "1,455.13" → "1,455")
    # with no banner. Below min_bet_cents and no confirming signal →
    # reconstructor must NOT emit a phantom bet.
    rec = EventReconstructor(num_seats=6)
    base = _fs(
        [0, 0, 0, 0, 0, 0],
        [200_000, 145_513, 200_000, 200_000, 200_000, 200_000],
    )
    rec.step(
        base,
        _engine(
            current_actor=1,
            commits=[0] * 6,
            stacks=[200_000 * 5, 145_513 * 5] + [200_000 * 5] * 4,
            chips_per_cent=5.0,
            min_bet_cents=2_000,
        ),
    )

    fs = _fs(
        [0, None, 0, 0, 0, 0],
        [200_000, 145_413, 200_000, 200_000, 200_000, 200_000],
    )
    ev = rec.step(
        fs,
        _engine(
            current_actor=1,
            commits=[0] * 6,
            stacks=[200_000 * 5, 145_513 * 5] + [200_000 * 5] * 4,
            chips_per_cent=5.0,
            min_bet_cents=2_000,
        ),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert seat_actions == []


# --- Fix 4: _any_remaining_delta respects stack drop / banner ----------


def test_walk_stalls_at_check_when_only_downstream_signal_is_banner():
    # SB (seat 0) checks; CO (seat 1) has banner up but committed_chips
    # hasn't read yet. Banner alone is no longer a "remaining delta"
    # signal (Signal C dropped from `_any_remaining_delta`), so the walk
    # MUST stall at SB rather than coast-check it. Rationale: hero's
    # face-up cards persistently trigger has_bet_banner under the
    # ClubGG anti-collusion rule, and Signal C was driving phantom
    # CHECKs onto upstream actors. The SB's real check still gets
    # filled in on a later tick (when seat 1's chip oval reads) or via
    # the StreetReveal CHECK reconciler.
    rec = EventReconstructor(num_seats=6)
    base = _fs(
        [0, 0, 0, 0, 0, 0],
        [200_000, 145_513, 200_000, 200_000, 200_000, 200_000],
    )
    rec.step(
        base,
        _engine(
            current_actor=0,
            commits=[0] * 6,
            stacks=[200_000 * 5] * 6,
            chips_per_cent=5.0,
            min_bet_cents=2_000,
        ),
    )

    # Poll: nothing changed on seat 0 (check), seat 1 banner visible.
    fs = _fs(
        [0, None, 0, 0, 0, 0],
        [200_000, 145_513, 200_000, 200_000, 200_000, 200_000],
        banners=[False, True, False, False, False, False],
    )
    ev = rec.step(
        fs,
        _engine(
            current_actor=0,
            commits=[0] * 6,
            stacks=[200_000 * 5] * 6,
            chips_per_cent=5.0,
            min_bet_cents=2_000,
        ),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    # No phantom CHECK emitted on SB — walk waits for actionable
    # evidence (chip change or stack drop) from a downstream seat.
    assert seat_actions == []


def test_walk_continues_past_check_when_downstream_has_stack_drop():
    # SB checks. CO bet $180: stack dropped 18000¢ but committed OCR
    # missed. No banner signal. Walk must continue to CO via stack delta.
    rec = EventReconstructor(num_seats=6)
    base = _fs(
        [0, 0, 0, 0, 0, 0],
        [200_000, 163_513, 200_000, 200_000, 200_000, 200_000],
    )
    rec.step(
        base,
        _engine(
            current_actor=0,
            commits=[0] * 6,
            stacks=[200_000 * 5, 163_513 * 5] + [200_000 * 5] * 4,
            chips_per_cent=5.0,
            min_bet_cents=2_000,
        ),
    )

    fs = _fs(
        [0, None, 0, 0, 0, 0],
        [200_000, 145_513, 200_000, 200_000, 200_000, 200_000],  # s1 dropped 18000
    )
    ev = rec.step(
        fs,
        _engine(
            current_actor=0,
            commits=[0] * 6,
            stacks=[200_000 * 5, 163_513 * 5] + [200_000 * 5] * 4,
            chips_per_cent=5.0,
            min_bet_cents=2_000,
        ),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    # SB check emitted, then walk reaches seat 1 and infers bet from
    # the 18000¢ stack drop × 5 = 90000 engine-chips raise-by.
    assert SeatAction(seat=0, gate="check_call", chips=0) in seat_actions
    assert SeatAction(seat=1, gate="raise", chips=90_000) in seat_actions


# --- Fix 5: fallback fires on zero-glitch committed read ----------------


def test_fallback_fires_when_committed_reads_zero_but_stack_dropped_and_banner_up():
    # Primary OCR reads committed_chips=0 (Tesseract glitch on a small
    # chip oval like "180"), but stack dropped 18000¢ AND banner is
    # visible. Classic zero-glitch scenario — must still emit raise.
    rec = EventReconstructor(num_seats=6)
    base = _fs(
        [0, 0, 0, 0, 0, 0],
        [200_000, 163_513, 200_000, 200_000, 200_000, 200_000],
    )
    rec.step(
        base,
        _engine(
            current_actor=1,
            commits=[0] * 6,
            stacks=[200_000 * 5, 163_513 * 5] + [200_000 * 5] * 4,
            chips_per_cent=5.0,
            min_bet_cents=2_000,
        ),
    )

    # Zero-glitch: commit reads as 0 (same as base_commit), but the
    # bet is actually real — confirmed by stack drop + banner.
    fs = _fs(
        [0, 0, 0, 0, 0, 0],
        [200_000, 145_513, 200_000, 200_000, 200_000, 200_000],
        banners=[False, True, False, False, False, False],
    )
    ev = rec.step(
        fs,
        _engine(
            current_actor=1,
            commits=[0] * 6,
            stacks=[200_000 * 5, 163_513 * 5] + [200_000 * 5] * 4,
            chips_per_cent=5.0,
            min_bet_cents=2_000,
        ),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert SeatAction(seat=1, gate="raise", chips=90_000) in seat_actions


def test_banner_only_without_chip_amount_emits_warning():
    # Banner visible but both committed_chips AND stack read failed.
    # Can't derive a bet amount — emit an OcrWarning instead of stalling
    # silently, and break the loop so we retry next tick.
    from plo5bp.ocr.events import OcrWarning

    rec = EventReconstructor(num_seats=6)
    base = _fs(
        [0, 0, 0, 0, 0, 0],
        [200_000] * 6,
    )
    rec.step(
        base,
        _engine(
            current_actor=1,
            commits=[0] * 6,
            stacks=[200_000 * 5] * 6,
            chips_per_cent=5.0,
            min_bet_cents=2_000,
        ),
    )

    fs = _fs(
        [0, None, 0, 0, 0, 0],
        [200_000, None, 200_000, 200_000, 200_000, 200_000],
        banners=[False, True, False, False, False, False],
    )
    ev = rec.step(
        fs,
        _engine(
            current_actor=1,
            commits=[0] * 6,
            stacks=[200_000 * 5] * 6,
            chips_per_cent=5.0,
            min_bet_cents=2_000,
        ),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    warnings = [e for e in ev if isinstance(e, OcrWarning)]
    assert seat_actions == []
    assert any("bet banner" in w.message for w in warnings)


# --- silent-seat coast-past (bug B) -------------------------------------


def test_silent_seats_coast_past_to_reach_real_bettor():
    """Live ClubGG frames leave `committed_chips=None` for seats with
    no chip oval on the table. Without a coast-past rule, the walk
    breaks at the first silent seat and never reaches the one that
    actually acted. Mirrors the Butt2Butt post-bet bug frame."""
    rec = EventReconstructor(num_seats=6)
    # Pre-bet baseline: seat 1 has the full 163513-cent stack
    # ($1,635.13), no commits anywhere, all seats silent (commit=None).
    pre = _fs(
        [None] * 6,
        [34_000, 163_513, 41_361, None, 24_650, None],
        button=2,
    )
    rec.step(
        pre,
        _engine(
            current_actor=3,
            commits=[0] * 6,
            stacks=[170_000, 817_565, 206_805, 200_000, 123_250, 1_669_830],
            folded=[False, False, False, True, False, True],
            button=2,
            chips_per_cent=5.0,
            min_bet_cents=2_000,
        ),
    )

    # Post-bet: seat 1 stack dropped by 18000 cents (= $180), commit
    # still reads None because OCR missed the chip-oval glyph.
    post = _fs(
        [None] * 6,
        [34_000, 145_513, 41_361, None, 24_650, None],
        button=2,
    )
    ev = rec.step(
        post,
        _engine(
            current_actor=3,
            commits=[0] * 6,
            stacks=[170_000, 817_565, 206_805, 200_000, 123_250, 1_669_830],
            folded=[False, False, False, True, False, True],
            button=2,
            chips_per_cent=5.0,
            min_bet_cents=2_000,
        ),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    # Walk: 3 skip (folded) → 4 silent→CHECK → 5 skip (folded) → 0
    # silent→CHECK → 1 stack-drop 18000 cents * 5 = 90_000 chip raise.
    assert SeatAction(seat=4, gate="check_call", chips=0) in seat_actions
    assert SeatAction(seat=0, gate="check_call", chips=0) in seat_actions
    assert SeatAction(seat=1, gate="raise", chips=90_000) in seat_actions


def test_silent_seats_with_no_downstream_evidence_break():
    """Flip-side: if every seat is silent and nothing dropped, the walk
    must break rather than emit phantom checks for the whole table."""
    rec = EventReconstructor(num_seats=6)
    pre = _fs([None] * 6, [30_000] * 6)
    rec.step(
        pre,
        _engine(
            current_actor=1,
            commits=[0] * 6,
            stacks=[30_000] * 6,
            chips_per_cent=1.0,
            min_bet_cents=0,
        ),
    )
    # Same frame again — nothing happened anywhere.
    ev = rec.step(
        pre,
        _engine(
            current_actor=1,
            commits=[0] * 6,
            stacks=[30_000] * 6,
            chips_per_cent=1.0,
            min_bet_cents=0,
        ),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert seat_actions == []


def test_silent_seat_facing_bet_does_not_phantom_fold():
    """When the silent seat has a facing bet and no downstream
    evidence, the walk should break (wait for next poll) rather than
    speculating a FOLD just because the commit OCR came back None."""
    rec = EventReconstructor(num_seats=6)
    # Seat 1 already bet 10_000 last tick; seat 2 is up but silent.
    pre = _fs(
        [0, 10_000, None, None, None, None],
        [30_000, 20_000, 30_000, 30_000, 30_000, 30_000],
    )
    rec.step(
        pre,
        _engine(
            current_actor=2,
            commits=[0, 10_000, 0, 0, 0, 0],
            stacks=[30_000] * 6,
            bet_to_call=10_000,
            chips_per_cent=1.0,
            min_bet_cents=0,
        ),
    )
    # No change — seat 2 still silent, no downstream action.
    ev = rec.step(
        pre,
        _engine(
            current_actor=2,
            commits=[0, 10_000, 0, 0, 0, 0],
            stacks=[30_000] * 6,
            bet_to_call=10_000,
            chips_per_cent=1.0,
            min_bet_cents=0,
        ),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    # Must NOT emit a FOLD for seat 2 just because OCR returned None.
    assert not any(
        isinstance(e, SeatAction) and e.seat == 2 and e.gate == "fold"
        for e in seat_actions
    )


def test_sub_1bb_commit_jitter_does_not_trigger_loop_guard():
    """Tesseract occasionally misreads the chip oval as a tiny integer
    ("4" = 400 cents → 2000 engine-chips) when the badge is mid-animation
    or partially occluded. Without a >= 1bb threshold in
    `_any_remaining_delta` Signal A, this phantom signal keeps the walk
    alive across 12 cycles. The walk coast-pasts through every seat
    emitting CHECKs; the hand applies all 12, the engine advances
    prematurely through streets to showdown, and then the rust hand-
    evaluator panics on padding cards.

    With the 1bb threshold the walk breaks cleanly after the real
    actions it can resolve. No phantom CHECKs, no loop-guard warning,
    no showdown cascade.
    """
    from plo5bp.ocr.events import OcrWarning

    rec = EventReconstructor(num_seats=6)
    # Four-handed bomb pot flop: hero=0, CO=1 (about to bet, but this
    # frame's OCR for them is glitched), BTN=2, sitting out 3 and 5,
    # BB=4. Button=2; first to act = 4.
    base = _fs(
        [0, 0, 0, 0, 0, 0],
        [34_000, 163_513, 41_361, 0, 24_650, 216_682],
    )
    rec.step(
        base,
        _engine(
            current_actor=4,
            commits=[0] * 6,
            stacks=[170_000, 817_565, 206_805, 170_000, 123_250, 1_083_410],
            folded=[False, False, False, True, False, True],
            bet_to_call=0,
            chips_per_cent=5.0,
            min_bet_cents=2_000,
            button=2,
            street=1,
        ),
    )
    # Glitched post-bet frame: seat 1 chip-oval OCR reads 400 cents
    # (0.4 bb) — the sub-1bb guard in the main walk rejects this, and
    # `_any_remaining_delta` must also reject it. Stack OCR for seat 1
    # is banner-covered and stuck at the anchor value (no drop signal).
    # No banner detected. Other seats unchanged.
    glitch = _fs(
        [0, 400, 0, 0, 0, 0],
        [34_000, 163_513, 41_361, 0, 24_650, 216_682],
    )
    ev = rec.step(
        glitch,
        _engine(
            current_actor=4,
            commits=[0] * 6,
            stacks=[170_000, 817_565, 206_805, 170_000, 123_250, 1_083_410],
            folded=[False, False, False, True, False, True],
            bet_to_call=0,
            chips_per_cent=5.0,
            min_bet_cents=2_000,
            button=2,
            street=1,
        ),
    )
    warnings = [w for w in ev if isinstance(w, OcrWarning)]
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert not any(
        "loop guard" in w.message for w in warnings
    ), f"loop guard fired; events={ev}"
    # Walk must break quickly — not emit 12 phantom CHECKs that would
    # carry the hand to showdown.
    assert len(seat_actions) <= 2, f"excessive CHECK emissions: {seat_actions}"


# --- Fix L: sitting-out seats must be skipped by the walk ------------


def test_sitting_out_seat_between_actors_emits_no_phantom_fold():
    """Fix L: a sitting-out seat between two in-hand seats must NOT
    be processed as a fold. Scenario mirrors the live bug —
    hero (seat 0) calls a facing bet, seat 1 is sitting out (no
    cards_back, no banner, no commit → obs.folded=True), seat 2
    (Castor876) has not yet acted. The walk must skip seat 1 without
    appending a `SeatAction(seat=1, gate="fold")` and without
    emitting a phantom FOLD on seat 2 either.
    """
    rec = EventReconstructor(num_seats=6)

    # Baseline: hero matched, seat 1 sitting out, others not yet acted.
    base_seats = (
        SeatObs(seat=0, stack_chips=20_000, committed_chips=10_000,
                folded=False, bet_banner=False),
        SeatObs(seat=1, stack_chips=0, committed_chips=None,
                folded=True, bet_banner=False),  # sitting out
        SeatObs(seat=2, stack_chips=30_000, committed_chips=0,
                folded=False, bet_banner=False),
        SeatObs(seat=3, stack_chips=30_000, committed_chips=0,
                folded=False, bet_banner=False),
        SeatObs(seat=4, stack_chips=30_000, committed_chips=10_000,
                folded=False, bet_banner=False),
        SeatObs(seat=5, stack_chips=30_000, committed_chips=0,
                folded=False, bet_banner=False),
    )
    base = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=5, pot_total_chips=20_000, seats=base_seats,
    )
    rec.step(
        base,
        _engine(
            current_actor=1,
            commits=[10_000, 0, 0, 0, 10_000, 0],
            stacks=[20_000, 0, 30_000, 30_000, 30_000, 30_000],
            bet_to_call=10_000,
            button=5,
            sitting_out=[False, True, False, False, False, False],
        ),
    )

    # Next tick: same frame (nothing actually happened). The walk
    # should skip seat 1 and NOT emit any fold on it.
    ev = rec.step(
        base,
        _engine(
            current_actor=1,
            commits=[10_000, 0, 0, 0, 10_000, 0],
            stacks=[20_000, 0, 30_000, 30_000, 30_000, 30_000],
            bet_to_call=10_000,
            button=5,
            sitting_out=[False, True, False, False, False, False],
        ),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    # No fold on seat 1 (sitting out) — that was the bug.
    assert all(
        not (a.seat == 1 and a.gate == "fold") for a in seat_actions
    ), f"phantom FOLD on sitting-out seat: {seat_actions}"


def test_walk_past_sitting_out_captures_real_downstream_fold():
    """Fix L: the seat immediately after a sitting-out seat can still
    fold, and that real fold must be attributed to the correct seat.
    Scenario: seats 1 and 3 sitting out; seat 2 really folded
    (cards gone, facing bet unmet). Expect exactly one FOLD event
    on seat 2 — no phantom on seat 1 or seat 3.
    """
    rec = EventReconstructor(num_seats=6)

    base_seats = (
        SeatObs(seat=0, stack_chips=20_000, committed_chips=10_000,
                folded=False, bet_banner=False),
        SeatObs(seat=1, stack_chips=0, committed_chips=None,
                folded=True, bet_banner=False),  # sitting out
        SeatObs(seat=2, stack_chips=30_000, committed_chips=0,
                folded=False, bet_banner=False),
        SeatObs(seat=3, stack_chips=0, committed_chips=None,
                folded=True, bet_banner=False),  # sitting out
        SeatObs(seat=4, stack_chips=30_000, committed_chips=10_000,
                folded=False, bet_banner=False),
        SeatObs(seat=5, stack_chips=30_000, committed_chips=0,
                folded=False, bet_banner=False),
    )
    base = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=5, pot_total_chips=20_000, seats=base_seats,
    )
    rec.step(
        base,
        _engine(
            current_actor=1,
            commits=[10_000, 0, 0, 0, 10_000, 0],
            stacks=[20_000, 0, 30_000, 0, 30_000, 30_000],
            bet_to_call=10_000,
            button=5,
            sitting_out=[False, True, False, True, False, False],
        ),
    )

    # Seat 2 now shows the folded signal (no cards, no banner,
    # no commit). Walk starts at seat 1 but must skip to seat 2
    # and emit exactly one FOLD on seat 2.
    fold_seats = (
        base_seats[0],
        base_seats[1],
        SeatObs(seat=2, stack_chips=30_000, committed_chips=None,
                folded=True, bet_banner=False),
        base_seats[3],
        base_seats[4],
        base_seats[5],
    )
    fold_fs = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=5, pot_total_chips=20_000, seats=fold_seats,
    )
    ev = rec.step(
        fold_fs,
        _engine(
            current_actor=1,
            commits=[10_000, 0, 0, 0, 10_000, 0],
            stacks=[20_000, 0, 30_000, 0, 30_000, 30_000],
            bet_to_call=10_000,
            button=5,
            sitting_out=[False, True, False, True, False, False],
        ),
    )
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    folds = [a for a in seat_actions if a.gate == "fold"]
    assert folds == [SeatAction(seat=2, gate="fold", chips=0)], (
        f"expected one fold on seat 2, got {seat_actions}"
    )


def test_any_remaining_delta_ignores_sitting_out_chatter():
    """Fix L: stale OCR on a sitting-out seat (chip oval noise,
    prior-hand stack read, or a lingering bet banner) must NOT keep
    the walk alive. Directly exercises `_any_remaining_delta`.

    Without the sitting_out skip, a sitting-out seat with
    committed_chips=8_000 jitter would return True and the walk
    would coast past a legitimate CHECK looking for a ghost.
    """
    rec = EventReconstructor(num_seats=4)
    new_seats = {
        0: SeatObs(seat=0, stack_chips=30_000, committed_chips=0,
                   folded=False, bet_banner=False),
        1: SeatObs(seat=1, stack_chips=0, committed_chips=8_000,
                   folded=True, bet_banner=True),  # stale chatter
        2: SeatObs(seat=2, stack_chips=30_000, committed_chips=0,
                   folded=False, bet_banner=False),
        3: SeatObs(seat=3, stack_chips=30_000, committed_chips=0,
                   folded=False, bet_banner=False),
    }
    last_seats = {
        0: new_seats[0],
        1: SeatObs(seat=1, stack_chips=0, committed_chips=8_000,
                   folded=True, bet_banner=True),
        2: new_seats[2],
        3: new_seats[3],
    }
    base_commit = [0, 0, 0, 0]

    # Without sitting_out: seat 1's chatter returns True.
    assert rec._any_remaining_delta(
        new_seats, last_seats, base_commit, scale=1.0,
        actor_start=1, min_bet_cents=1_000,
    ) is True

    # With sitting_out on seat 1: chatter is ignored.
    assert rec._any_remaining_delta(
        new_seats, last_seats, base_commit, scale=1.0,
        actor_start=1, min_bet_cents=1_000,
        sitting_out=[False, True, False, False],
    ) is False


# --- Fix N: corroboration guard on primary_read -------------------------


def _phantom_baseline() -> tuple[FrameState, EngineView]:
    """Shared baseline for Fix N tests.

    4 seats, seat 1 sitting out. Seat 3 bet 10_000; hero (seat 0) called
    10_000. Current actor is seat 2 — the "folded-but-noisy" seat whose
    chip oval is about to misread. Seat 2 starts with no commit and a
    30_000 stack.
    """
    base_seats = (
        SeatObs(seat=0, stack_chips=20_000, committed_chips=10_000,
                folded=False, bet_banner=False),
        SeatObs(seat=1, stack_chips=None, committed_chips=None,
                folded=True, bet_banner=False),
        SeatObs(seat=2, stack_chips=30_000, committed_chips=None,
                folded=False, bet_banner=False),
        SeatObs(seat=3, stack_chips=20_000, committed_chips=10_000,
                folded=False, bet_banner=False),
    )
    base = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=3, pot_total_chips=20_000, seats=base_seats,
    )
    eng = _engine(
        current_actor=2,
        commits=[10_000, 0, 0, 10_000],
        stacks=[20_000, 0, 30_000, 20_000],
        bet_to_call=10_000,
        button=3,
        sitting_out=[False, True, False, False],
    )
    return base, eng


def test_primary_read_rejected_when_uncorroborated():
    """Fix N: a positive `primary_delta` with no banner and no stack
    drop is Tesseract noise on a folded seat's empty chip oval. The
    guard nulls out `primary_read` and the fallback ladder breaks at
    the actor — no phantom CHECK_CALL, no pot inflation.
    """
    rec = EventReconstructor(num_seats=4)
    base, eng = _phantom_baseline()
    rec.step(base, eng)

    # Seat 2 shows a phantom 10_000 commit but stack is unchanged and
    # no banner. Real table state: seat 2 just folded and the chip
    # oval is empty — Tesseract cross-talk reading it as positive.
    phantom_seats = (
        base.seats[0],
        base.seats[1],
        SeatObs(seat=2, stack_chips=30_000, committed_chips=10_000,
                folded=False, bet_banner=False),
        base.seats[3],
    )
    phantom_fs = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=3, pot_total_chips=20_000, seats=phantom_seats,
    )
    ev = rec.step(phantom_fs, eng)
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert not any(a.seat == 2 for a in seat_actions), (
        f"expected no event on seat 2, got {seat_actions}"
    )


def test_primary_read_trusted_with_banner():
    """Fix N: banner=True corroborates the positive primary_delta, so
    the guard lets the read through and the exact-call branch fires.
    """
    rec = EventReconstructor(num_seats=4)
    base, eng = _phantom_baseline()
    rec.step(base, eng)

    call_seats = (
        base.seats[0],
        base.seats[1],
        SeatObs(seat=2, stack_chips=30_000, committed_chips=10_000,
                folded=False, bet_banner=True),
        base.seats[3],
    )
    call_fs = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=3, pot_total_chips=30_000, seats=call_seats,
    )
    ev = rec.step(call_fs, eng)
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert SeatAction(seat=2, gate="check_call", chips=0) in seat_actions


def test_primary_read_trusted_with_stack_drop():
    """Fix N: a matching stack drop corroborates the positive
    primary_delta even without a banner. 30_000 -> 20_000 drop covers
    the 10_000 primary_delta, so the read is trusted.
    """
    rec = EventReconstructor(num_seats=4)
    base, eng = _phantom_baseline()
    rec.step(base, eng)

    call_seats = (
        base.seats[0],
        base.seats[1],
        SeatObs(seat=2, stack_chips=20_000, committed_chips=10_000,
                folded=False, bet_banner=False),
        base.seats[3],
    )
    call_fs = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=3, pot_total_chips=30_000, seats=call_seats,
    )
    ev = rec.step(call_fs, eng)
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert SeatAction(seat=2, gate="check_call", chips=0) in seat_actions


def test_uncorroborated_commit_then_fold_on_next_tick():
    """Fix N + Fix J: tick A rejects the noisy commit (no event). Tick
    B's OCR stabilizes — committed_chips=None, folded=True — and Fix
    J's positive-fold branch emits the FOLD that tick A suppressed.
    """
    rec = EventReconstructor(num_seats=4)
    base, eng = _phantom_baseline()
    rec.step(base, eng)

    # Tick A: phantom commit rejected.
    phantom_seats = (
        base.seats[0],
        base.seats[1],
        SeatObs(seat=2, stack_chips=30_000, committed_chips=10_000,
                folded=False, bet_banner=False),
        base.seats[3],
    )
    phantom_fs = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=3, pot_total_chips=20_000, seats=phantom_seats,
    )
    ev_a = rec.step(phantom_fs, eng)
    sa_a = [e for e in ev_a if isinstance(e, SeatAction)]
    assert not any(a.seat == 2 for a in sa_a)

    # Tick B: OCR stabilizes. Chip oval now reads None, no banner, and
    # multi-signal flips obs.folded=True. Fix J emits the FOLD.
    fold_seats = (
        base.seats[0],
        base.seats[1],
        SeatObs(seat=2, stack_chips=30_000, committed_chips=None,
                folded=True, bet_banner=False),
        base.seats[3],
    )
    fold_fs = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=3, pot_total_chips=20_000, seats=fold_seats,
    )
    ev_b = rec.step(fold_fs, eng)
    sa_b = [e for e in ev_b if isinstance(e, SeatAction)]
    folds = [a for a in sa_b if a.gate == "fold" and a.seat == 2]
    assert folds == [SeatAction(seat=2, gate="fold", chips=0)], (
        f"expected one fold on seat 2, got {sa_b}"
    )


# --- Fix P: zero-on-zero guard on primary_read ---------------------------


def test_zero_on_zero_no_phantom_fold():
    """Fix P: primary_read == 0 with base_commit == 0 is information-
    free. Without the guard, branch 4 of the ladder takes the 0 as
    "no change" and the downstream delta==0 & to_call>0 path emits a
    phantom FOLD on a seat whose real state is "in hand, yet to act".
    """
    rec = EventReconstructor(num_seats=4)
    base, eng = _phantom_baseline()
    rec.step(base, eng)

    # Test tick: seat 2's chip oval OCRs as an explicit 0 (not None).
    # Cards still visible (folded=False), stack unchanged, no banner.
    # Facing a 10_000 bet from seat 3.
    zero_seats = (
        base.seats[0],
        base.seats[1],
        SeatObs(seat=2, stack_chips=30_000, committed_chips=0,
                folded=False, bet_banner=False),
        base.seats[3],
    )
    zero_fs = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=3, pot_total_chips=20_000, seats=zero_seats,
    )
    ev = rec.step(zero_fs, eng)
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert not any(a.seat == 2 for a in seat_actions), (
        f"expected no event on seat 2, got {seat_actions}"
    )


def test_zero_primary_with_positive_base_commit_processes_normally():
    """Fix P sanity check: when base_commit != 0, the guard must not
    fire. A seat that previously committed 10_000 but now OCRs as 0
    (e.g. oval text wiped) hits the ladder normally — branch 1
    computes a negative delta and the walk breaks on the
    "negative delta — street probably closed" branch. No event, no
    crash.
    """
    rec = EventReconstructor(num_seats=4)
    # Engine state: seats 0, 2, 3 have each committed 10_000. Seat 1
    # sits out. current_actor=2 would normally have to_call_here=0
    # (they already match facing_bet), but we're testing the guard's
    # non-firing condition, not downstream walk semantics.
    eng = _engine(
        current_actor=2,
        commits=[10_000, 0, 10_000, 10_000],
        stacks=[20_000, 0, 30_000, 20_000],
        bet_to_call=10_000,
        button=3,
        sitting_out=[False, True, False, False],
    )
    base_seats = (
        SeatObs(seat=0, stack_chips=20_000, committed_chips=10_000,
                folded=False, bet_banner=False),
        SeatObs(seat=1, stack_chips=None, committed_chips=None,
                folded=True, bet_banner=False),
        SeatObs(seat=2, stack_chips=30_000, committed_chips=10_000,
                folded=False, bet_banner=False),
        SeatObs(seat=3, stack_chips=20_000, committed_chips=10_000,
                folded=False, bet_banner=False),
    )
    base = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=3, pot_total_chips=30_000, seats=base_seats,
    )
    rec.step(base, eng)

    # Test tick: seat 2's oval now reads 0. Stack unchanged, no banner.
    wiped_seats = (
        base_seats[0],
        base_seats[1],
        SeatObs(seat=2, stack_chips=30_000, committed_chips=0,
                folded=False, bet_banner=False),
        base_seats[3],
    )
    wiped_fs = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=3, pot_total_chips=30_000, seats=wiped_seats,
    )
    ev = rec.step(wiped_fs, eng)
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert not any(a.seat == 2 for a in seat_actions), (
        f"expected no event on seat 2 (negative-delta break), "
        f"got {seat_actions}"
    )


def test_zero_on_zero_with_banner_emits_warning_not_fold():
    """Fix P + branch 3: after the guard nulls primary_read, an active
    bet banner with no chip amount yet falls through to the existing
    "banner without chips" warning branch. No phantom FOLD.
    """
    from plo5bp.ocr.events import OcrWarning

    rec = EventReconstructor(num_seats=4)
    base, eng = _phantom_baseline()
    rec.step(base, eng)

    banner_seats = (
        base.seats[0],
        base.seats[1],
        SeatObs(seat=2, stack_chips=30_000, committed_chips=0,
                folded=False, bet_banner=True),
        base.seats[3],
    )
    banner_fs = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=3, pot_total_chips=20_000, seats=banner_seats,
    )
    ev = rec.step(banner_fs, eng)
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    warnings = [e for e in ev if isinstance(e, OcrWarning)]
    assert not seat_actions, f"expected no SeatAction, got {seat_actions}"
    assert len(warnings) == 1 and "seat 2" in warnings[0].message


# --- Fix Q: sub-facing-bet raise guard ----------------------------------


def test_fix_q_subfacing_stack_drop_emits_no_events():
    """Fix Q: branch 2 derives new_commit < facing_bet from a small
    stack drift (e.g. 1_000 ¢ flicker when facing a 10_000 bet). The
    walk previously emitted an illegal RAISE that the engine rejected
    AND corrupted base_commit/facing_bet_bumped state, ungating the
    downstream phantom-FOLD branch. Guard breaks the walk instead.
    """
    rec = EventReconstructor(num_seats=4)
    base, eng = _phantom_baseline()
    rec.step(base, eng)

    # Seat 2's stack dropped 1_000 cents between ticks (noise). No
    # banner. base_commit=0, facing_bet=10_000. Branch 2 would compute
    # new_commit=1_000 which is < facing_bet — Fix Q breaks.
    drift_seats = (
        base.seats[0],
        base.seats[1],
        SeatObs(seat=2, stack_chips=29_000, committed_chips=None,
                folded=False, bet_banner=False),
        base.seats[3],
    )
    drift_fs = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=3, pot_total_chips=20_000, seats=drift_seats,
    )
    ev = rec.step(drift_fs, eng)
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert not any(a.seat == 2 for a in seat_actions), (
        f"expected no event on seat 2, got {seat_actions}"
    )


def test_fix_q_legit_raise_above_facing_bet_emits():
    """Fix Q does not fire when new_commit > facing_bet. A real raise
    still emits normally — stack drop of 20_000 cents against
    facing_bet=10_000 is a raise-to-20_000 (delta=20_000).
    """
    rec = EventReconstructor(num_seats=4)
    base, eng = _phantom_baseline()
    rec.step(base, eng)

    raise_seats = (
        base.seats[0],
        base.seats[1],
        SeatObs(seat=2, stack_chips=10_000, committed_chips=None,
                folded=False, bet_banner=False),
        base.seats[3],
    )
    raise_fs = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=3, pot_total_chips=20_000, seats=raise_seats,
    )
    ev = rec.step(raise_fs, eng)
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert SeatAction(seat=2, gate="raise", chips=20_000) in seat_actions


def test_fix_q_exact_call_emits_check_call_not_break():
    """Fix Q uses strict `<`, so new_commit == facing_bet (an exact
    call) is routed to the exact-call branch at line 525 and emits
    CHECK_CALL. Stack drop of 10_000 against facing_bet=10_000.
    """
    rec = EventReconstructor(num_seats=4)
    base, eng = _phantom_baseline()
    rec.step(base, eng)

    call_seats = (
        base.seats[0],
        base.seats[1],
        SeatObs(seat=2, stack_chips=20_000, committed_chips=None,
                folded=False, bet_banner=False),
        base.seats[3],
    )
    call_fs = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=3, pot_total_chips=20_000, seats=call_seats,
    )
    ev = rec.step(call_fs, eng)
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert SeatAction(seat=2, gate="check_call", chips=0) in seat_actions


def test_fix_q_short_all_in_below_facing_bet_still_emits_raise():
    """Fix Q does not pre-empt the short-shove branch (line 513). A
    short shove with new_stack == 0 but delta < facing_bet is detected
    before Fix Q is reached, so the raise emission (encoding the
    shove) is preserved.
    """
    rec = EventReconstructor(num_seats=4)
    # Custom baseline: seat 2 has a short stack (below facing_bet).
    base_seats = (
        SeatObs(seat=0, stack_chips=20_000, committed_chips=10_000,
                folded=False, bet_banner=False),
        SeatObs(seat=1, stack_chips=None, committed_chips=None,
                folded=True, bet_banner=False),
        SeatObs(seat=2, stack_chips=5_000, committed_chips=None,
                folded=False, bet_banner=False),
        SeatObs(seat=3, stack_chips=20_000, committed_chips=10_000,
                folded=False, bet_banner=False),
    )
    base = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=3, pot_total_chips=20_000, seats=base_seats,
    )
    eng = _engine(
        current_actor=2,
        commits=[10_000, 0, 0, 10_000],
        stacks=[20_000, 0, 5_000, 20_000],
        bet_to_call=10_000,
        button=3,
        sitting_out=[False, True, False, False],
    )
    rec.step(base, eng)

    # Tick B: seat 2 shoves all 5_000 in. Stack → 0, new_commit=5_000.
    # facing_bet=10_000, so new_commit < facing_bet — but the
    # short-shove branch fires first on new_stack==0.
    shove_seats = (
        base_seats[0],
        base_seats[1],
        SeatObs(seat=2, stack_chips=0, committed_chips=None,
                folded=False, bet_banner=False),
        base_seats[3],
    )
    shove_fs = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=3, pot_total_chips=25_000, seats=shove_seats,
    )
    ev = rec.step(shove_fs, eng)
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert SeatAction(seat=2, gate="raise", chips=5_000) in seat_actions


def test_fix_q_open_pot_bet_with_no_facing_bet_emits():
    """Fix Q does not fire when facing_bet == 0 (no one has bet yet).
    Seat 2 opens the betting with a 5_000 stack drop. new_commit=5_000
    > facing_bet=0, so the guard is a no-op and the raise emits.
    """
    rec = EventReconstructor(num_seats=4)
    # Custom baseline: nobody has acted this street. facing_bet=0.
    base_seats = (
        SeatObs(seat=0, stack_chips=30_000, committed_chips=None,
                folded=False, bet_banner=False),
        SeatObs(seat=1, stack_chips=None, committed_chips=None,
                folded=True, bet_banner=False),
        SeatObs(seat=2, stack_chips=30_000, committed_chips=None,
                folded=False, bet_banner=False),
        SeatObs(seat=3, stack_chips=30_000, committed_chips=None,
                folded=False, bet_banner=False),
    )
    base = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=3, pot_total_chips=0, seats=base_seats,
    )
    eng = _engine(
        current_actor=2,
        commits=[0, 0, 0, 0],
        stacks=[30_000, 0, 30_000, 30_000],
        bet_to_call=0,
        button=3,
        sitting_out=[False, True, False, False],
    )
    rec.step(base, eng)

    open_seats = (
        base_seats[0],
        base_seats[1],
        SeatObs(seat=2, stack_chips=25_000, committed_chips=None,
                folded=False, bet_banner=False),
        base_seats[3],
    )
    open_fs = FrameState(
        board_a=_flop_a(), board_b=_flop_b(), hero_hole=_hero(),
        button_seat=3, pot_total_chips=5_000, seats=open_seats,
    )
    ev = rec.step(open_fs, eng)
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert SeatAction(seat=2, gate="raise", chips=5_000) in seat_actions


# --- active-actor (timer bar) transition signal -------------------------


def _actors(active: int | None, n: int = 6) -> list[bool]:
    out = [False] * n
    if active is not None:
        out[active] = True
    return out


def test_timer_bar_transition_emits_check_for_prior_actor():
    """Bar moves from seat 0 (hero) to seat 1 with no chip change ⇒
    seat 0 checked. Reproduces the documented hero-CHECK-on-flop bug
    where _any_remaining_delta has nothing to lean on.

    Uses ``committed_chips=None`` for every seat so the existing
    primary-read no-change branch can't fire — the timer-bar branch
    is the only path to a CHECK in this fixture."""
    rec = EventReconstructor(num_seats=6)
    base = _fs([None] * 6, [30_000] * 6, actors=_actors(0))
    eng = _engine(current_actor=0, commits=[0] * 6, stacks=[30_000] * 6,
                  bet_to_call=0)
    rec.step(base, eng)  # bootstrap; locks _observed_active_actor=0

    fs2 = _fs([None] * 6, [30_000] * 6, actors=_actors(1))
    ev = rec.step(fs2, eng)
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert seat_actions == [SeatAction(seat=0, gate="check_call", chips=0)]


def test_timer_bar_no_transition_when_same_seat_stays_active():
    """Time-bank refill: bar stays on the same seat across ticks ⇒
    no CHECK emitted. Same active read on consecutive frames is
    "actor still thinking", not "actor passed turn"."""
    rec = EventReconstructor(num_seats=6)
    base = _fs([None] * 6, [30_000] * 6, actors=_actors(0))
    eng = _engine(current_actor=0, commits=[0] * 6, stacks=[30_000] * 6,
                  bet_to_call=0)
    rec.step(base, eng)

    fs2 = _fs([None] * 6, [30_000] * 6, actors=_actors(0))
    ev = rec.step(fs2, eng)
    assert [e for e in ev if isinstance(e, SeatAction)] == []


def test_timer_bar_ocr_miss_does_not_advance_lock():
    """Frame 2 has zero is_actor reads (OCR missed the bar); frame 3
    sees the bar at seat 1. The miss must not clear the lock — the
    transition 0→1 should still emit CHECK on frame 3, not on frame 2.
    """
    rec = EventReconstructor(num_seats=6)
    base = _fs([None] * 6, [30_000] * 6, actors=_actors(0))
    eng = _engine(current_actor=0, commits=[0] * 6, stacks=[30_000] * 6,
                  bet_to_call=0)
    rec.step(base, eng)

    fs2 = _fs([None] * 6, [30_000] * 6, actors=_actors(None))  # OCR miss
    ev2 = rec.step(fs2, eng)
    assert [e for e in ev2 if isinstance(e, SeatAction)] == []

    fs3 = _fs([None] * 6, [30_000] * 6, actors=_actors(1))
    ev3 = rec.step(fs3, eng)
    assert [e for e in ev3 if isinstance(e, SeatAction)] == [
        SeatAction(seat=0, gate="check_call", chips=0)
    ]


def test_timer_bar_ambiguous_multiple_actors_does_not_update_lock():
    """Animation frame where two seats briefly read is_actor=True ⇒
    no update, no spurious CHECK."""
    rec = EventReconstructor(num_seats=6)
    base = _fs([None] * 6, [30_000] * 6, actors=_actors(0))
    eng = _engine(current_actor=0, commits=[0] * 6, stacks=[30_000] * 6,
                  bet_to_call=0)
    rec.step(base, eng)

    actors = [False] * 6
    actors[0] = True
    actors[1] = True
    fs2 = _fs([None] * 6, [30_000] * 6, actors=actors)
    ev2 = rec.step(fs2, eng)
    assert [e for e in ev2 if isinstance(e, SeatAction)] == []
    # Lock should not have advanced — next clean read at seat 1 must
    # still trigger the CHECK from the original prev=0.
    fs3 = _fs([None] * 6, [30_000] * 6, actors=_actors(1))
    ev3 = rec.step(fs3, eng)
    assert [e for e in ev3 if isinstance(e, SeatAction)] == [
        SeatAction(seat=0, gate="check_call", chips=0)
    ]


def test_timer_bar_does_not_fire_when_facing_bet():
    """If there's a facing bet, the actor's silence is a FOLD signal
    (or pending action), not a CHECK. The new branch is gated on
    `to_call_here == 0`, so a transition with a facing bet must NOT
    use the timer bar to manufacture a CHECK."""
    rec = EventReconstructor(num_seats=6)
    base = _fs([None, 5_000, None, None, None, None],
               [30_000, 25_000, 30_000, 30_000, 30_000, 30_000],
               actors=_actors(2))
    eng = _engine(current_actor=2, commits=[0, 5_000, 0, 0, 0, 0],
                  stacks=[30_000, 25_000, 30_000, 30_000, 30_000, 30_000],
                  bet_to_call=5_000)
    rec.step(base, eng)

    # Bar moves 2→3 with seat 2 still owing 5_000 chips.
    fs2 = _fs([None, 5_000, None, None, None, None],
              [30_000, 25_000, 30_000, 30_000, 30_000, 30_000], actors=_actors(3))
    ev = rec.step(fs2, eng)
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    # The new timer-bar branch must NOT emit a check_call here. The
    # walk's standard delta==0+to_call>0 fold path may emit a FOLD,
    # which is acceptable — the assertion is just that we don't
    # emit a phantom CHECK.
    assert all(
        not (e.seat == 2 and e.gate == "check_call")
        for e in seat_actions
    )


def test_timer_bar_rebaseline_clears_lock():
    """rebaseline() (called at hand-start) must clear
    _observed_active_actor so the first tick of a new hand can't emit
    a stale CHECK."""
    rec = EventReconstructor(num_seats=6)
    base = _fs([None] * 6, [30_000] * 6, actors=_actors(0))
    eng = _engine(current_actor=0, commits=[0] * 6, stacks=[30_000] * 6,
                  bet_to_call=0)
    rec.step(base, eng)
    assert rec._observed_active_actor == 0

    new_hand_fs = _fs([None] * 6, [30_000] * 6, actors=_actors(0))
    rec.rebaseline(new_hand_fs)
    assert rec._observed_active_actor is None

    # First tick after rebaseline with the bar at seat 1 should NOT
    # emit a CHECK (no prior lock to compare against).
    fs2 = _fs([None] * 6, [30_000] * 6, actors=_actors(1))
    ev = rec.step(fs2, eng)
    assert [e for e in ev if isinstance(e, SeatAction)] == []


def test_no_phantom_check_on_first_tick_of_new_street():
    """Regression: when the engine has just advanced to a new street
    and the chip oval reads 0 (Tesseract returns 0 instead of None
    for an empty oval), the walk must NOT phantom-emit a CHECK for
    the new ``current_actor``.

    Reproduces the live bug where heads-up bomb-pot flop both-check
    correctly, engine advances to turn with fastaf as
    ``current_actor``, then the next OCR tick sees ``primary_read=0``
    on fastaf's empty oval and emits a phantom CHECK that
    incorrectly advances the engine past them.

    The guard: branch 4 (``primary_read`` agrees with ``base_commit``)
    only fires for seats whose action is already accounted for in
    the engine state — never for the engine's own ``current_actor``,
    who needs positive corroboration (timer-bar transition,
    hero-hole-hid, downstream activity)."""
    rec = EventReconstructor(num_seats=6)
    # Bootstrap with the bar on seat 1 (fastaf), the new actor.
    base = _fs([0] * 6, [30_000] * 6, actors=_actors(1))
    eng = _engine(current_actor=1, commits=[0] * 6, stacks=[30_000] * 6,
                  bet_to_call=0)
    rec.step(base, eng)

    # Next tick: bar still on seat 1 (still thinking on the new
    # street). Tesseract reads the empty oval as 0 (matching
    # base_commit). No transition, no banner, no stack drop, no
    # downstream activity. Walk MUST break without emitting.
    fs2 = _fs([0] * 6, [30_000] * 6, actors=_actors(1))
    ev = rec.step(fs2, eng)
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert seat_actions == []


def test_no_phantom_check_from_banner_only_downstream_seat():
    """Regression: a persistent false-positive bet_banner downstream
    of the engine's current_actor must NOT keep the walk alive.

    Pre-fix: ``_any_remaining_delta`` returned True on Signal C
    (banner alone, no chip amount), and branch 7 of the walk presumed
    a CHECK on the current_actor whenever any downstream seat
    flagged banner=True. In the live ClubGG bug this fired every
    tick because hero's face-up cards (anti-collusion rule)
    pinned ``bet_banner=True`` indefinitely on seat 0 — driving a
    phantom CHECK on the OOP actor at the start of every postflop
    street.

    Post-fix: Signal C is removed from ``_any_remaining_delta``.
    Banner-only downstream evidence is unactionable (the walk's
    own branch 3 would just break-with-warning at the banner seat
    anyway), so it no longer keeps the walk alive past
    current_actor.
    """
    rec = EventReconstructor(num_seats=6)
    # Bootstrap with the timer bar on seat 1 (fastaf, OOP first to
    # act on the new street). Hero is downstream at seat 0.
    base = _fs(
        [None] * 6, [30_000] * 6,
        actors=_actors(1),
    )
    eng = _engine(
        current_actor=1, commits=[0] * 6, stacks=[30_000] * 6,
        bet_to_call=0,
    )
    rec.step(base, eng)

    # Next tick: fastaf still thinking (bar still on seat 1, no
    # chip activity). Hero (seat 0) has bet_banner=True from the
    # face-up-card false positive but no chip amount and no stack
    # drop — pure noise. Walk visits fastaf (current_actor=1), no
    # local signals, falls through every branch. Branch 7 used to
    # fire because hero's banner=True downstream; with Signal C
    # gone it must not.
    banners = [True, False, False, False, False, False]  # hero banner only
    fs2 = _fs(
        [None] * 6, [30_000] * 6,
        actors=_actors(1),
        banners=banners,
    )
    ev = rec.step(fs2, eng)
    seat_actions = [e for e in ev if isinstance(e, SeatAction)]
    assert seat_actions == [], (
        f"phantom action emitted from banner-only downstream signal: "
        f"{seat_actions}"
    )


def test_closing_call_stops_walk_and_emits_no_phantoms():
    """The street-closing call must emit exactly one action, no phantoms.

    PokerNow shows every player's matched bet simultaneously the instant the
    closing caller acts (before sweeping chips to the pot). The walk used to
    coast past the already-acted seats and emit phantom check_calls — and even
    a phantom raise — inflating the pot. The betting-round-closed guard at the
    top of the walk stops it once everyone still in has matched the bet.
    """
    rec = EventReconstructor(num_seats=3)
    # Baseline: hero(0) bet 100, seat1(1) called 100, seat2(2) to act.
    base = _fs([100, 100, 0], [900, 900, 1000], actors=[False, False, True])
    rec.step(base, _engine(current_actor=2, commits=[100, 100, 0],
                           stacks=[900, 900, 1000], bet_to_call=100))

    # Closing frame: seat2 calls — all three now show a matched 100, bets not
    # yet swept. The engine still has seat2 to act (commits=[100,100,0]).
    closing = _fs([100, 100, 100], [900, 900, 900], actors=[False, False, True])
    events = rec.step(closing, _engine(current_actor=2, commits=[100, 100, 0],
                                       stacks=[900, 900, 1000], bet_to_call=100))
    actions = [e for e in events if isinstance(e, SeatAction)]
    assert len(actions) == 1
    assert actions[0].seat == 2
    assert actions[0].gate == "check_call"


def test_allin_shove_recorded_when_stack_reads_zero():
    """An all-in shove (stack rendered 0, not None) is recorded as a raise.

    PokerNow shows "all in" instead of the stack number; the mapper maps that to
    a 0 stack. With 0 (not None) the corroboration guard sees the full stack
    drop backing the shove, so the bet survives and routes through the
    short-shove path — instead of being discarded and collapsing the hand to a
    phantom check-down.
    """
    rec = EventReconstructor(num_seats=2)
    base = _fs([0, 0], [8800, 1800], actors=[False, True])
    rec.step(base, _engine(current_actor=1, commits=[0, 0],
                           stacks=[8800, 1800], bet_to_call=0, street=2))

    # seat 1 shoves: committed = their whole 1800, stack now 0 (all in).
    allin = _fs([0, 1800], [8800, 0], actors=[False, True])
    events = rec.step(allin, _engine(current_actor=1, commits=[0, 0],
                                     stacks=[8800, 1800], bet_to_call=0, street=2))
    actions = [e for e in events if isinstance(e, SeatAction)]
    assert len(actions) == 1
    assert actions[0].seat == 1
    assert actions[0].gate == "raise"
    assert actions[0].chips == 1800  # chips_per_cent=1.0 in _engine


def test_exact_folds_suppresses_inferred_fold_on_coasted_seat():
    """With exact fold data (PokerNow), the walk must not INFER a fold for a
    seat that is merely facing a bet with no committed change.

    seat2 bet 100; seat1 already has 50 in and is yet to respond; seat0 calls.
    The walk coasts past seat1 (committed unchanged at 50, facing 100) — the
    OCR path infers a FOLD there, but PokerNow says seat1 hasn't folded, so it
    must be left alone (it's a thinking/closing-call seat). This is the
    "last caller registers as a fold and the hand ends" bug.
    """
    def run(exact):
        rec = EventReconstructor(num_seats=3)
        base = _fs([0, 50, 100], [200, 150, 100], actors=[True, False, False])
        rec.step(base, _engine(current_actor=0, commits=[0, 50, 100],
                               stacks=[200, 150, 100], bet_to_call=100,
                               exact_folds=exact))
        fs2 = _fs([100, 50, 100], [100, 150, 100], actors=[False, True, False])
        return rec.step(fs2, _engine(current_actor=0, commits=[0, 50, 100],
                                     stacks=[200, 150, 100], bet_to_call=100,
                                     exact_folds=exact))

    inferred = [e for e in run(False)
                if isinstance(e, SeatAction) and e.gate == "fold" and e.seat == 1]
    assert inferred, "scenario should trigger the inferred fold without exact_folds"

    suppressed = [e for e in run(True)
                  if isinstance(e, SeatAction) and e.gate == "fold" and e.seat == 1]
    assert not suppressed, "exact_folds must not infer a fold for a non-folded seat"
