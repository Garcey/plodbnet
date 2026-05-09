"""Tests for the server-side OCR mirror + hand-start trigger.

These tests drive ``_mirror_observable_state`` directly with synthetic
FrameStates and assert that ``session`` state converges correctly.
They cover the architectural properties that were previously invisible
in the event-level tests: rewind-resync, stability gating, and unit
conversion at hand-start.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
torch = pytest.importorskip("torch")

from plo5bp.config import GameConfig  # noqa: E402
from plo5bp.ocr.types import Card, FrameState, SeatObs  # noqa: E402
from plo5bp.ui import server  # noqa: E402


def _c(rank: int, suit: int) -> Card:
    return Card(rank=rank, suit=suit)


def _seat(seat: int, stack: int | None = 34000, commit: int | None = 0,
          folded: bool = False) -> SeatObs:
    return SeatObs(
        seat=seat,
        stack_chips=stack,
        committed_chips=commit,
        folded=folded,
    )


def _fs(
    *,
    seats: tuple[SeatObs, ...],
    button: int | None = 2,
    hero: tuple[Card | None, ...] | None = None,
    pot: int | None = 0,
    flop_a_visible: bool = False,
) -> FrameState:
    if hero is None:
        hero = (_c(8, 1), _c(7, 1), _c(7, 0), _c(3, 2), _c(1, 2))
    flop_a: tuple[Card | None, ...]
    flop_b: tuple[Card | None, ...]
    if flop_a_visible:
        flop_a = (_c(10, 0), _c(8, 0), _c(11, 0), None, None)
        flop_b = (_c(5, 1), _c(7, 2), _c(1, 1), None, None)
    else:
        flop_a = (None,) * 5
        flop_b = (None,) * 5
    return FrameState(
        board_a=flop_a,
        board_b=flop_b,
        hero_hole=hero,
        button_seat=button,
        pot_total_chips=pot,
        seats=seats,
    )


@pytest.fixture(autouse=True)
def _reset_session():
    """Reset the module-level session between tests so state doesn't leak."""
    server.session.game_config = GameConfig()
    server.session.dollars_per_bb = 20.0
    server.session.num_seats = 6
    server.session.button_seat = 0
    server.session.hero_seat = 0
    server.session.sitting_out_seats = frozenset()
    server.session.observed_stacks = ()
    server.session.observed_pot = None
    server.session.last_hero_hole = None
    server.session._pending_button = None
    server.session._pending_sitting_out = None
    server.session._pending_stable_ticks = 0
    server.session._pending_anchor_fs = None
    server.session._pending_mask_additions = frozenset()
    server.session._pending_mask_additions_ticks = 0
    server.ocr_runner._reconstructor = None
    server._new_session_defaults()
    yield


# --- unit conversion ----------------------------------------------------


def test_ocr_cents_to_engine_chips_at_defaults():
    # bb=10000, $20/bb → 1 cent = 5 engine chips.
    assert server._ocr_cents_to_engine_chips(34000) == 170000
    assert server._ocr_cents_to_engine_chips(145513) == 727565
    assert server._ocr_cents_to_engine_chips(41361) == 206805
    assert server._ocr_cents_to_engine_chips(24650) == 123250


def test_ocr_cents_to_engine_chips_rounding():
    # $0.01 at defaults = 5 chips (rounded).
    assert server._ocr_cents_to_engine_chips(1) == 5
    assert server._ocr_cents_to_engine_chips(0) == 0


# --- SeatAction → action_log entry --------------------------------------


def test_seat_action_raise_passes_chips_through():
    # SeatAction.chips is already a raise-by delta in engine-chips
    # (the reconstructor converts OCR cents at its boundary). The
    # server helper is a straight passthrough — no unit fixup.
    from plo5bp.ocr.events import SeatAction

    ev = SeatAction(seat=3, gate="raise", chips=90_000)
    entry = server._seat_action_to_log_entry(ev)
    assert entry["chips"] == 90_000


def test_seat_action_non_raise_gates_log_zero_chips():
    from plo5bp.ocr.events import SeatAction

    for gate in ("fold", "check_call"):
        ev = SeatAction(seat=2, gate=gate, chips=0)
        entry = server._seat_action_to_log_entry(ev)
        assert entry["chips"] == 0


# --- stability gate -----------------------------------------------------


def test_stability_gate_needs_two_consecutive_ticks():
    # First tick with s3/s5 folded: pending, NOT committed yet.
    seats = (
        _seat(0, stack=34000),
        _seat(1, stack=145513),
        _seat(2, stack=41361),
        _seat(3, folded=True),
        _seat(4, stack=24650),
        _seat(5, folded=True),
    )
    fs = _fs(seats=seats)
    server._mirror_observable_state(fs)
    assert server.session.sitting_out_seats == frozenset()
    assert server.session._pending_sitting_out == frozenset({3, 5})
    assert server.session._pending_stable_ticks == 1

    # Second matching tick commits the snapshot.
    server._mirror_observable_state(fs)
    assert server.session.sitting_out_seats == frozenset({3, 5})


def test_stability_gate_rejects_single_glitch_frame():
    seats_good = (
        _seat(0, stack=34000),
        _seat(1, stack=145513),
        _seat(2, stack=41361),
        _seat(3, folded=True),
        _seat(4, stack=24650),
        _seat(5, folded=True),
    )
    # Two good ticks → commit {3, 5}.
    fs_good = _fs(seats=seats_good)
    server._mirror_observable_state(fs_good)
    server._mirror_observable_state(fs_good)
    assert server.session.sitting_out_seats == frozenset({3, 5})

    # One glitched tick (everyone folded, button None) must not commit.
    seats_glitch = tuple(_seat(i, folded=True) for i in range(6))
    fs_glitch = _fs(seats=seats_glitch, button=None)
    server._mirror_observable_state(fs_glitch)
    # sitting_out_seats still holds the prior valid commit.
    assert server.session.sitting_out_seats == frozenset({3, 5})

    # Subsequent good frame resets the gate back to the correct set.
    server._mirror_observable_state(fs_good)
    server._mirror_observable_state(fs_good)
    assert server.session.sitting_out_seats == frozenset({3, 5})


# --- hand-start: stacks come through unit conversion --------------------


def test_begin_new_hand_converts_ocr_cents_to_engine_chips():
    seats = (
        _seat(0, stack=34000),     # $340
        _seat(1, stack=145513),    # $1455.13
        _seat(2, stack=41361),     # $413.61
        _seat(3, folded=True),
        _seat(4, stack=24650),     # $246.50
        _seat(5, folded=True),
    )
    fs = _fs(seats=seats)
    server._mirror_observable_state(fs)
    server._mirror_observable_state(fs)

    # Committed sitting-out → hand start fired → cfg.starting_stacks
    # reflects OCR reads (cents * 5 + ante).
    cfg = server.session.game_config
    expected_s0 = 170000 + cfg.ante  # 200000
    expected_s1 = 727565 + cfg.ante  # 757565
    expected_s2 = 206805 + cfg.ante  # 236805
    expected_s4 = 123250 + cfg.ante  # 153250
    assert cfg.resolved_stacks[0] == expected_s0
    assert cfg.resolved_stacks[1] == expected_s1
    assert cfg.resolved_stacks[2] == expected_s2
    assert cfg.resolved_stacks[4] == expected_s4


# --- rewind resync ------------------------------------------------------


def test_rewind_resync_re_detects_participants():
    # Step 1: hand A with s3/s5 out.
    seats_a = (
        _seat(0, stack=34000),
        _seat(1, stack=145513),
        _seat(2, stack=41361),
        _seat(3, folded=True),
        _seat(4, stack=24650),
        _seat(5, folded=True),
    )
    fs_a = _fs(seats=seats_a)
    server._mirror_observable_state(fs_a)
    server._mirror_observable_state(fs_a)
    assert server.session.sitting_out_seats == frozenset({3, 5})

    # Step 2: user rewinds → "fresh" frame that looks like nothing
    # special. Participant set should NOT spontaneously change — stays
    # locked at {3, 5}.
    server._mirror_observable_state(fs_a)
    assert server.session.sitting_out_seats == frozenset({3, 5})

    # Step 3: next hand begins with different participants (s4 now out
    # instead of s5) and a rotated button.
    seats_b = (
        _seat(0, stack=40000, commit=None),
        _seat(1, stack=145513, commit=None),
        _seat(2, stack=41361, commit=None),
        _seat(3, folded=True),
        _seat(4, folded=True),
        _seat(5, stack=333966, commit=None),
    )
    fs_b = _fs(seats=seats_b, button=3)
    # Mid-hand `_begin_new_hand` is gated by the locked-mode debouncer
    # (`_STABILITY_TICKS_REQUIRED_LOCKED`). Pump enough ticks for the new
    # snapshot to clear that threshold so button_changed can fire.
    for _ in range(server._STABILITY_TICKS_REQUIRED_LOCKED):
        server._mirror_observable_state(fs_b)
    assert server.session.sitting_out_seats == frozenset({3, 4})
    assert server.session.button_seat == 3


# --- anchor-fs: pre-commit baseline for hand-start ----------------------


def test_anchor_fs_seeds_stacks_from_pre_commit_frame():
    # Tick 1: SB pre-bet with stack $1,635.13.
    seats_pre = (
        _seat(0, stack=34000),
        _seat(1, stack=145513),
        _seat(2, stack=163513),   # SB pre-bet
        _seat(3, folded=True),
        _seat(4, stack=24650),
        _seat(5, folded=True),
    )
    fs_pre = _fs(seats=seats_pre)
    server._mirror_observable_state(fs_pre)
    # Anchor captured on first observation of the snapshot.
    assert server.session._pending_anchor_fs is fs_pre
    assert server.session._pending_stable_ticks == 1

    # Tick 2: same snapshot, but SB just bet $180 → stack now $1,455.13.
    seats_post = (
        _seat(0, stack=34000),
        _seat(1, stack=145513),
        _seat(2, stack=145513, commit=None),  # SB post-bet, OCR missed
        _seat(3, folded=True),
        _seat(4, stack=24650),
        _seat(5, folded=True),
    )
    fs_post = _fs(seats=seats_post)
    server._mirror_observable_state(fs_post)

    # Hand-start committed → seeding used the pre-bet anchor, not fs_post.
    cfg = server.session.game_config
    expected_sb_pre = 163513 * 5 + cfg.ante
    expected_sb_post = 145513 * 5 + cfg.ante
    assert cfg.resolved_stacks[2] == expected_sb_pre
    assert cfg.resolved_stacks[2] != expected_sb_post
    # Anchor consumed.
    assert server.session._pending_anchor_fs is None


# --- anchor stack sanity check (Phase 6c) -------------------------------


def test_begin_new_hand_rejects_glitched_in_hand_stack_read():
    # Seat 1 in-hand but stack_chips read came back 0 (banner covered
    # the label during capture). Without the sanity check, merged[1]
    # would be seeded at 30000 chips = ante exactly, and reset_study
    # would post the ante and leave seat 1 at 0 → phantom all-in.
    seats = (
        _seat(0, stack=34000),
        _seat(1, stack=0),         # GLITCHED
        _seat(2, stack=41361),
        _seat(3, folded=True),
        _seat(4, stack=24650),
        _seat(5, folded=True),
    )
    fs = _fs(seats=seats)
    server._begin_new_hand(fs, button_seat=2, hero_hole_indices=None)
    cfg = server.session.game_config
    # Seat 1's resolved stack should NOT have been seeded from the 0
    # read — it should retain its pre-existing value (default 200000).
    glitched_seed = 0 * 5 + cfg.ante  # 30000 — the value we must NOT pick
    assert cfg.resolved_stacks[1] != glitched_seed
    # Clean seats 0, 2, 4 still seeded correctly.
    assert cfg.resolved_stacks[0] == 34000 * 5 + cfg.ante
    assert cfg.resolved_stacks[2] == 41361 * 5 + cfg.ante
    assert cfg.resolved_stacks[4] == 24650 * 5 + cfg.ante


def test_mirror_delays_hand_start_when_all_ticks_have_glitched_anchor():
    # Both ticks have seat 1 glitched to 0. Debouncer should refuse to
    # commit the hand: hand_in_hand_mask stays empty.
    glitched_seats = (
        _seat(0, stack=34000),
        _seat(1, stack=0),         # GLITCHED every tick
        _seat(2, stack=41361),
        _seat(3, folded=True),
        _seat(4, stack=24650),
        _seat(5, folded=True),
    )
    fs_glitch = _fs(seats=glitched_seats)
    server._mirror_observable_state(fs_glitch)
    server._mirror_observable_state(fs_glitch)
    # Stable for 2 ticks, but commit delayed because anchor is glitched.
    assert server.session._pending_stable_ticks >= 2
    assert not server.session.hand_in_hand_mask


def test_anchor_upgrade_swaps_glitched_anchor_for_clean_tick():
    # Tick 1: seat 1 glitched to 0. Anchor stored, but commit will be
    # blocked if this anchor persists.
    glitched_seats = (
        _seat(0, stack=34000),
        _seat(1, stack=0),         # GLITCHED
        _seat(2, stack=41361),
        _seat(3, folded=True),
        _seat(4, stack=24650),
        _seat(5, folded=True),
    )
    fs_glitch = _fs(seats=glitched_seats)
    server._mirror_observable_state(fs_glitch)
    assert server.session._pending_anchor_fs is fs_glitch

    # Tick 2: same snapshot (button + sitting_out unchanged), but seat 1
    # reads cleanly now. Anchor must be upgraded to this cleaner fs,
    # and the commit must fire using it.
    clean_seats = (
        _seat(0, stack=34000),
        _seat(1, stack=163513),     # clean read
        _seat(2, stack=41361),
        _seat(3, folded=True),
        _seat(4, stack=24650),
        _seat(5, folded=True),
    )
    fs_clean = _fs(seats=clean_seats)
    server._mirror_observable_state(fs_clean)

    # Commit fired — hand_in_hand_mask populated, anchor consumed.
    assert server.session.hand_in_hand_mask == frozenset({0, 1, 2, 4})
    cfg = server.session.game_config
    # Seat 1 seeded from the clean tick, not the glitched one.
    assert cfg.resolved_stacks[1] == 163513 * 5 + cfg.ante


# --- sticky participant mask (Fix 2) ------------------------------------


def test_sticky_mask_survives_banner_flicker():
    # Two good ticks → commit mask = {0,1,2,4}, sitting_out = {3,5}.
    seats_good = (
        _seat(0, stack=34000),
        _seat(1, stack=145513),
        _seat(2, stack=41361),
        _seat(3, folded=True),
        _seat(4, stack=24650),
        _seat(5, folded=True),
    )
    fs_good = _fs(seats=seats_good)
    server._mirror_observable_state(fs_good)
    server._mirror_observable_state(fs_good)
    assert server.session.hand_in_hand_mask == frozenset({0, 1, 2, 4})
    assert server.session.sitting_out_seats == frozenset({3, 5})

    # Glitch tick: seat 1's cards_back + banner BOTH miss → folded=True.
    # Without stickiness, seat 1 would join sitting_out_seats.
    seats_flicker = (
        _seat(0, stack=34000),
        _seat(1, folded=True),   # <-- banner flicker defeats cards_back
        _seat(2, stack=41361),
        _seat(3, folded=True),
        _seat(4, stack=24650),
        _seat(5, folded=True),
    )
    fs_flicker = _fs(seats=seats_flicker)
    server._mirror_observable_state(fs_flicker)
    # Sticky mask wins: seat 1 stays in the hand.
    assert server.session.sitting_out_seats == frozenset({3, 5})
    assert server.session.hand_in_hand_mask == frozenset({0, 1, 2, 4})


def test_mid_hand_fold_event_updates_sitting_out_without_restart():
    # Commit a hand with 4 participants.
    seats = (
        _seat(0, stack=34000),
        _seat(1, stack=145513),
        _seat(2, stack=41361),
        _seat(3, folded=True),
        _seat(4, stack=24650),
        _seat(5, folded=True),
    )
    fs = _fs(seats=seats)
    server._mirror_observable_state(fs)
    server._mirror_observable_state(fs)
    assert server.session.hand_in_hand_mask == frozenset({0, 1, 2, 4})
    pre_cfg = server.session.game_config

    # Simulate the reconstructor emitting a FOLD for seat 1 (exactly
    # what happens inside OcrRunner._tick on a real fold).
    from plo5bp.ocr.events import SeatAction
    ev = SeatAction(seat=1, gate="fold", chips=0)
    server.session.folded_this_hand = frozenset(
        server.session.folded_this_hand | {int(ev.seat)}
    )
    all_seats = frozenset(range(server.session.num_seats))
    server.session.sitting_out_seats = (
        (all_seats - server.session.hand_in_hand_mask)
        | server.session.folded_this_hand
    )

    # Another observable tick with seat 1 still visibly folded doesn't
    # trigger a hand restart (the mask didn't change vs committed).
    seats_after_fold = (
        _seat(0, stack=34000),
        _seat(1, folded=True),
        _seat(2, stack=41361),
        _seat(3, folded=True),
        _seat(4, stack=24650),
        _seat(5, folded=True),
    )
    fs_after = _fs(seats=seats_after_fold)
    server._mirror_observable_state(fs_after)
    server._mirror_observable_state(fs_after)

    # Mask unchanged, fold tracked in folded_this_hand, config untouched.
    assert server.session.hand_in_hand_mask == frozenset({0, 1, 2, 4})
    assert server.session.folded_this_hand == frozenset({1})
    assert server.session.sitting_out_seats == frozenset({1, 3, 5})
    # Hand-start was NOT re-fired (cfg reference is the same object).
    assert server.session.game_config is pre_cfg


def test_new_hand_on_button_rotation_clears_folded_this_hand():
    # Hand 1: commit with s1 folded mid-hand.
    seats_h1 = (
        _seat(0, stack=34000),
        _seat(1, stack=145513),
        _seat(2, stack=41361),
        _seat(3, folded=True),
        _seat(4, stack=24650),
        _seat(5, folded=True),
    )
    fs_h1 = _fs(seats=seats_h1, button=2)
    server._mirror_observable_state(fs_h1)
    server._mirror_observable_state(fs_h1)
    server.session.folded_this_hand = frozenset({1})

    # Hand 2: button rotates → real hand-start.
    seats_h2 = (
        _seat(0, stack=34000),
        _seat(1, stack=145513),
        _seat(2, stack=41361),
        _seat(3, folded=True),
        _seat(4, stack=24650),
        _seat(5, folded=True),
    )
    fs_h2 = _fs(seats=seats_h2, button=3)
    server._mirror_observable_state(fs_h2)
    server._mirror_observable_state(fs_h2)
    assert server.session.button_seat == 3
    assert server.session.folded_this_hand == frozenset()


def test_anchor_fs_rebaselines_reconstructor_to_pre_commit_frame():
    # Ensure the reconstructor exists (normally created in OcrRunner.start).
    from plo5bp.ocr.events import EventReconstructor

    server.ocr_runner._reconstructor = EventReconstructor(num_seats=6)

    seats_pre = (
        _seat(0, stack=34000),
        _seat(1, stack=145513),
        _seat(2, stack=163513),   # SB pre-bet
        _seat(3, folded=True),
        _seat(4, stack=24650),
        _seat(5, folded=True),
    )
    fs_pre = _fs(seats=seats_pre)
    server._mirror_observable_state(fs_pre)

    seats_post = (
        _seat(0, stack=34000),
        _seat(1, stack=145513),
        _seat(2, stack=145513, commit=None),  # SB bet $180
        _seat(3, folded=True),
        _seat(4, stack=24650),
        _seat(5, folded=True),
    )
    fs_post = _fs(seats=seats_post)
    server._mirror_observable_state(fs_post)

    # Reconstructor baseline is the pre-bet anchor, not the post-bet fs.
    rec = server.ocr_runner._reconstructor
    assert rec.last_fs is fs_pre
    assert rec.last_fs is not fs_post
    # The pre-bet seat stack is preserved for the next step's diff.
    assert rec.last_fs.seats[2].stack_chips == 163513


# --- Fix K: fold reconcile on StreetReveal ------------------------------


def test_reconcile_missed_folds_adds_one_fold_per_missing_seat():
    # Hand has 4 in-hand seats {0, 1, 2, 4}. Walk has already
    # emitted a FOLD for seat 1 (folded_this_hand = {1}). By the
    # turn-reveal tick, seats 2 and 4 are also visually folded but
    # the walk never saw them (e.g. snap-fold mid chip-settle
    # animation with cards_back still above threshold). Reconcile
    # must append exactly two FOLD entries and promote {2, 4} into
    # folded_this_hand / sitting_out_seats.
    server.session.hand_in_hand_mask = frozenset({0, 1, 2, 4})
    server.session.folded_this_hand = frozenset({1})
    server.session.action_log = [
        {"gate": int(server.GATE_RAISE), "chips": 90_000},  # hero open
        {"gate": int(server.GATE_FOLD), "chips": 0},        # seat 1 fold
    ]
    # Don't clobber sitting_out_seats — server.py computes it from
    # (all - in_hand) | folded_this_hand. Seats 3 and 5 are out-
    # of-hand; seat 1 folded mid-hand.
    all_seats = frozenset(range(server.session.num_seats))
    server.session.sitting_out_seats = (
        (all_seats - server.session.hand_in_hand_mask)
        | server.session.folded_this_hand
    )

    seats = (
        _seat(0, stack=34000),                   # hero still live
        _seat(1, folded=True),                   # already in fth
        _seat(2, folded=True),                   # MISSED fold
        _seat(3, folded=True),                   # sitting out
        _seat(4, folded=True),                   # MISSED fold
        _seat(5, folded=True),                   # sitting out
    )
    fs = _fs(seats=seats)
    server._reconcile_missed_folds_on_street_reveal(fs)

    assert server.session.folded_this_hand == frozenset({1, 2, 4})
    assert server.session.sitting_out_seats == frozenset({1, 2, 3, 4, 5})
    # Two new FOLD entries appended — one per missing seat. Previous
    # two log entries (raise + explicit fold) preserved.
    assert len(server.session.action_log) == 4
    fold_entries = [
        e for e in server.session.action_log[2:]
        if e["gate"] == int(server.GATE_FOLD) and e["chips"] == 0
    ]
    assert len(fold_entries) == 2


def test_reconcile_missed_folds_noop_when_no_missing():
    # All visually-folded seats are already in folded_this_hand.
    # Reconcile must be a no-op — no action_log append.
    server.session.hand_in_hand_mask = frozenset({0, 1, 2, 4})
    server.session.folded_this_hand = frozenset({1, 4})
    server.session.action_log = [
        {"gate": int(server.GATE_RAISE), "chips": 90_000},
        {"gate": int(server.GATE_FOLD), "chips": 0},
        {"gate": int(server.GATE_FOLD), "chips": 0},
    ]
    log_before = list(server.session.action_log)

    seats = (
        _seat(0, stack=34000),
        _seat(1, folded=True),
        _seat(2, stack=41361),                   # still in hand
        _seat(3, folded=True),
        _seat(4, folded=True),
        _seat(5, folded=True),
    )
    fs = _fs(seats=seats)
    server._reconcile_missed_folds_on_street_reveal(fs)

    assert server.session.action_log == log_before
    assert server.session.folded_this_hand == frozenset({1, 4})


def test_reconcile_missed_folds_skips_when_mask_empty():
    # Before any hand-start has fired, hand_in_hand_mask is empty
    # and a stray StreetReveal must not synthesize phantom folds.
    server.session.hand_in_hand_mask = frozenset()
    server.session.folded_this_hand = frozenset()
    server.session.action_log = []

    seats = tuple(_seat(i, folded=True) for i in range(6))
    fs = _fs(seats=seats)
    server._reconcile_missed_folds_on_street_reveal(fs)

    assert server.session.action_log == []
    assert server.session.folded_this_hand == frozenset()


def test_engine_view_populates_sitting_out_from_session():
    """Fix L: `_engine_view_from_session` must surface
    `session.sitting_out_seats` into the returned `EngineView` so the
    reconstructor walk can skip sitting-out seats instead of treating
    them as folds."""
    # Drive a two-tick commit so hand_in_hand_mask + resolved_stacks
    # are seeded. Seats 1 and 3 are sitting out.
    seats = (
        _seat(0, stack=34000),
        _seat(1, folded=True),
        _seat(2, stack=41361),
        _seat(3, folded=True),
        _seat(4, stack=24650),
        _seat(5, stack=145513),
    )
    fs = _fs(seats=seats)
    server._mirror_observable_state(fs)
    server._mirror_observable_state(fs)
    assert server.session.sitting_out_seats == frozenset({1, 3})

    view = server._engine_view_from_session()
    assert view.sitting_out == (False, True, False, True, False, False)
    # Sanity: the engine's own folded array won't reflect sitting-out
    # between polls until `_auto_fold_sitting_out` runs during replay —
    # that's exactly why `sitting_out` must be surfaced separately.
    assert len(view.folded) == 6


# --- Fix R: _auto_fold_sitting_out must not double-fold mid-hand folds ---


def test_rebuild_env_does_not_double_fold_mid_hand_folded_seat():
    """Regression for phantom BTN fold (Fix R).

    Scenario mirrors the canonical bug hand:
      - 6-seat bomb pot, seats 1 and 3 sitting out.
      - Flop action: seat 5 bets 90000, seat 0 calls, seat 2 folds.
      - folded_this_hand = {2}; sitting_out_seats = {1, 2, 3}.

    Before Fix R, replay auto-folded seat 2 via step_hybrid BEFORE
    applying the FOLD action_log entry, so the FOLD landed on seat 4
    (BTN). After Fix R, only structurally-sitting-out seats (1, 3) are
    auto-folded; the FOLD entry applies to seat 2 as intended.
    """
    server.session.num_seats = 6
    server.session.button_seat = 4
    server.session.hero_seat = 0
    server.session.hand_in_hand_mask = frozenset({0, 2, 4, 5})
    server.session.folded_this_hand = frozenset({2})
    all_seats = frozenset(range(6))
    server.session.sitting_out_seats = (
        (all_seats - server.session.hand_in_hand_mask)
        | server.session.folded_this_hand
    )
    server.session.action_log = [
        {"gate": int(server.GATE_RAISE), "chips": 90_000},
        {"gate": int(server.GATE_CHECK_CALL), "chips": 0},
        {"gate": int(server.GATE_FOLD), "chips": 0},
    ]

    server._rebuild_env()

    env = server.session.env
    assert env is not None
    raw = dict(env._rs.observation_dict())
    folded = [bool(x) for x in raw["folded"][:6]]
    # Seats 1, 3 auto-folded (structurally sitting out).
    # Seat 2 folded via action_log entry (Castor's real fold).
    # Seat 4 (BTN) MUST NOT be folded — this is the regression.
    assert folded == [False, True, True, True, False, False]
    assert int(raw["street"]) == 1  # still on flop
    assert int(raw.get("actor")) == 4  # action on BTN
    assert int(raw.get("bet_to_call")) == 90_000


# --- mid-hand mask expansion (anchor-too-early backstop) ------------------


def test_mid_hand_mask_expansion_adds_late_render_participant():
    """Anchor frame fired before fastaf's cards-back rendered →
    `hand_in_hand_mask` locked at {0} only. Two consecutive ticks of
    fastaf reading folded=False must expand the mask to include them."""
    # Anchor: only hero in-hand (fastaf's cards-back hadn't rendered yet).
    seats_anchor = (
        _seat(0, stack=100_000),
        _seat(1, folded=True),
        _seat(2, folded=True),   # fastaf — wrongly folded at anchor
        _seat(3, folded=True),
        _seat(4, folded=True),
        _seat(5, folded=True),
    )
    fs_anchor = _fs(seats=seats_anchor)
    server._mirror_observable_state(fs_anchor)
    server._mirror_observable_state(fs_anchor)
    assert server.session.hand_in_hand_mask == frozenset({0})

    # Tick 1 of recovery: fastaf's cards-back now reads — folded=False.
    seats_recovered = (
        _seat(0, stack=100_000),
        _seat(1, folded=True),
        _seat(2, stack=34_000),  # fastaf back in
        _seat(3, folded=True),
        _seat(4, folded=True),
        _seat(5, folded=True),
    )
    fs1 = _fs(seats=seats_recovered)
    server._mirror_observable_state(fs1)
    # Not yet — only 1 stable tick.
    assert server.session.hand_in_hand_mask == frozenset({0})

    # Tick 2: same observation persists → expand.
    fs2 = _fs(seats=seats_recovered)
    server._mirror_observable_state(fs2)
    assert server.session.hand_in_hand_mask == frozenset({0, 2})
    assert server.session.sitting_out_seats == frozenset({1, 3, 4, 5})


def test_mid_hand_mask_expansion_does_not_add_late_rebuy():
    """Late rebuy: a player who got stacked last hand and rebuys 1-2s
    into the next hand. They have no cards / banner / commit / timer-bar
    → extract.py reads `folded=True`. The expansion logic must NOT add
    them to the mask just because their stack appeared on screen."""
    # Anchor: hero only in this hand.
    seats_anchor = (
        _seat(0, stack=100_000),
        _seat(1, folded=True),  # rebuyer — no cards, no chips yet
        _seat(2, folded=True),
        _seat(3, folded=True),
        _seat(4, folded=True),
        _seat(5, folded=True),
    )
    fs_anchor = _fs(seats=seats_anchor)
    server._mirror_observable_state(fs_anchor)
    server._mirror_observable_state(fs_anchor)
    assert server.session.hand_in_hand_mask == frozenset({0})

    # Several ticks later: seat 1's stack populates as $400 but they
    # still have no cards (sitting out this hand, will play next).
    # extract.py still reads folded=True for them.
    seats_with_rebuy_stack = (
        _seat(0, stack=100_000),
        _seat(1, stack=40_000, folded=True),  # stack visible, NO cards
        _seat(2, folded=True),
        _seat(3, folded=True),
        _seat(4, folded=True),
        _seat(5, folded=True),
    )
    fs_rebuy = _fs(seats=seats_with_rebuy_stack)
    server._mirror_observable_state(fs_rebuy)
    server._mirror_observable_state(fs_rebuy)
    server._mirror_observable_state(fs_rebuy)

    # Mask never grew: rebuyer stays out of this hand.
    assert server.session.hand_in_hand_mask == frozenset({0})


def test_mid_hand_mask_expansion_excludes_already_folded_seats():
    """When a seat folds mid-hand, the FOLD event puts them in
    `folded_this_hand`. Their cards then disappear (folded=True). If
    a later tick spuriously reads folded=False (e.g. cards_back noise
    over the mucked area), the expansion logic must NOT re-add them.
    The seat should remain registered as folded, not promoted back
    into the active mask."""
    # Anchor: hero + fastaf in-hand.
    seats_anchor = (
        _seat(0, stack=100_000),
        _seat(1, folded=True),
        _seat(2, stack=34_000),
        _seat(3, folded=True),
        _seat(4, folded=True),
        _seat(5, folded=True),
    )
    fs_anchor = _fs(seats=seats_anchor)
    server._mirror_observable_state(fs_anchor)
    server._mirror_observable_state(fs_anchor)
    assert server.session.hand_in_hand_mask == frozenset({0, 2})

    # fastaf folds — server marks them in folded_this_hand (mirrors
    # what OcrRunner._tick does on a real FOLD event).
    server.session.folded_this_hand = frozenset({2})

    # Two stable ticks where seat 2 spuriously reads folded=False
    # (e.g. cards_back classifier hit a stray pixel pattern). Without
    # the folded_this_hand guard, expansion would re-include them.
    seats_spurious = (
        _seat(0, stack=100_000),
        _seat(1, folded=True),
        _seat(2, stack=34_000, folded=False),  # spurious in-hand read
        _seat(3, folded=True),
        _seat(4, folded=True),
        _seat(5, folded=True),
    )
    fs_spur = _fs(seats=seats_spurious)
    server._mirror_observable_state(fs_spur)
    server._mirror_observable_state(fs_spur)

    # Mask unchanged; seat 2 stays in folded_this_hand.
    assert server.session.hand_in_hand_mask == frozenset({0, 2})
    assert server.session.folded_this_hand == frozenset({2})


def test_mid_hand_mask_expansion_resets_on_changed_candidates():
    """A single-tick OCR flicker (seat 2 folded=False on tick A only)
    must NOT add seat 2 — the 2-tick stability gate requires the same
    candidate set across consecutive ticks."""
    seats_anchor = (
        _seat(0, stack=100_000),
        _seat(1, folded=True),
        _seat(2, folded=True),
        _seat(3, folded=True),
        _seat(4, folded=True),
        _seat(5, folded=True),
    )
    server._mirror_observable_state(_fs(seats=seats_anchor))
    server._mirror_observable_state(_fs(seats=seats_anchor))
    assert server.session.hand_in_hand_mask == frozenset({0})

    # Tick A: seat 2 reads folded=False (one-frame glitch).
    seats_glitch = (
        _seat(0, stack=100_000),
        _seat(1, folded=True),
        _seat(2, stack=34_000),
        _seat(3, folded=True),
        _seat(4, folded=True),
        _seat(5, folded=True),
    )
    server._mirror_observable_state(_fs(seats=seats_glitch))
    # Tick B: seat 2 back to folded=True. Different candidate set,
    # debouncer resets.
    server._mirror_observable_state(_fs(seats=seats_anchor))

    assert server.session.hand_in_hand_mask == frozenset({0})
    assert server.session._pending_mask_additions_ticks == 0


# --- false hand-start guard: button must actually move ------------------


def test_anti_collusion_reveal_does_not_trigger_false_hand_start():
    """ClubGG anti-collusion reveals hero's cards mid-hand on hero's
    first flop turn. If `last_hero_hole` was populated from a previous
    hand's cards (lingering at the boundary), the reveal would
    otherwise trip `hero_hole_rotated` against a stale baseline. The
    button-position guard at the trigger site suppresses the false
    `_begin_new_hand` because OCR still reads the button on the same
    seat."""
    hand1 = (_c(8, 1), _c(7, 1), _c(7, 0), _c(3, 2), _c(1, 2))
    hand2 = (_c(11, 0), _c(10, 0), _c(9, 1), _c(2, 0), _c(4, 3))
    h1_idx = tuple(int(c.rank) * 4 + int(c.suit) for c in hand1)
    h2_idx = tuple(int(c.rank) * 4 + int(c.suit) for c in hand2)
    assert set(h1_idx).isdisjoint(h2_idx)

    seats_full = tuple(_seat(i, stack=100_000) for i in range(6))

    # First-commit path seeds button=2 and last_hero_hole=hand1.
    fs_h1 = _fs(seats=seats_full, hero=hand1, button=2)
    server._mirror_observable_state(fs_h1)
    server._mirror_observable_state(fs_h1)
    assert server.session.button_seat == 2
    assert server.session.last_hero_hole == h1_idx
    assert server.session.hand_in_hand_mask == frozenset(range(6))

    # Pump into lock mode (_ticks_since_hand_start >= _LOCK_AFTER_TICKS).
    for _ in range(server._LOCK_AFTER_TICKS):
        server._mirror_observable_state(fs_h1)

    # Mid-hand: seat 1 folds, hero hole goes hidden, flop appears.
    seats_with_fold = (
        _seat(0, stack=100_000),
        _seat(1, folded=True),
        _seat(2, stack=100_000),
        _seat(3, stack=100_000),
        _seat(4, stack=100_000),
        _seat(5, stack=100_000),
    )
    fs_hidden = _fs(
        seats=seats_with_fold,
        hero=(None,) * 5,
        button=2,
        flop_a_visible=True,
    )
    for _ in range(2):
        server._mirror_observable_state(fs_hidden)

    # Anti-collusion reveal: hero hole becomes hand2 cards (disjoint
    # from last_hero_hole=hand1). Button stays on seat 2. Pump past
    # the lock-mode rotation debounce threshold.
    fs_revealed = _fs(
        seats=seats_with_fold,
        hero=hand2,
        button=2,
        flop_a_visible=True,
    )
    for _ in range(server._STABILITY_TICKS_REQUIRED_LOCKED + 2):
        server._mirror_observable_state(fs_revealed)

    # Guard worked: `_begin_new_hand` was not called, so seat 1 stays
    # in the mask (would otherwise be dropped as `folded=True` at the
    # anchor frame), and button_seat is unchanged.
    assert server.session.hand_in_hand_mask == frozenset(range(6))
    assert server.session.button_seat == 2


def test_real_hand_boundary_still_fires_when_button_moves():
    """Counterpoint to the anti-collusion test: a true hand boundary
    moves the button to a different seat, so the guard does not
    suppress the trigger."""
    hand1 = (_c(8, 1), _c(7, 1), _c(7, 0), _c(3, 2), _c(1, 2))
    hand2 = (_c(11, 0), _c(10, 0), _c(9, 1), _c(2, 0), _c(4, 3))

    seats_full = tuple(_seat(i, stack=100_000) for i in range(6))
    fs_h1 = _fs(seats=seats_full, hero=hand1, button=2)
    server._mirror_observable_state(fs_h1)
    server._mirror_observable_state(fs_h1)
    assert server.session.button_seat == 2

    for _ in range(server._LOCK_AFTER_TICKS):
        server._mirror_observable_state(fs_h1)

    # Hand 2 boundary: button rotates 2 → 4 (skipping seat 3, which is
    # legal when seat 3 is sitting out at a smaller table). New hero
    # cards too. Pump past the lock-mode debounce threshold.
    fs_h2 = _fs(seats=seats_full, hero=hand2, button=4)
    for _ in range(server._STABILITY_TICKS_REQUIRED_LOCKED):
        server._mirror_observable_state(fs_h2)

    # Trigger fired: button updated.
    assert server.session.button_seat == 4
