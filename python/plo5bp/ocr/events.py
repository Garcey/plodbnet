"""Reconstruct game events from a stream of OCR FrameStates.

The OCR capture loop polls the target window every ~200ms and feeds
each snapshot to `EventReconstructor.step(fs, engine_view)`. The
reconstructor diffs the new FrameState against the previous processed
one (see "Baseline policy" below), cross-references an engine snapshot
to know whose turn it is, and emits tagged OcrEvent instances describing
what changed.

Events are *emitted*, not applied — the caller (the UI's OCR runner)
is responsible for translating each event into a session mutation
(append to `action_log`, call `_rebuild_env`, etc.). Keeping apply
out of this module lets the reconstructor be unit-tested without an
engine in the loop.

Gap-fill inference
------------------
If the poll rate misses an action, the next FrameState's commit/stack
deltas span multiple expected actors. We walk the action order from
`engine_view.current_actor` forward; for each expected actor, the
change in their committed-this-street chips tells us what they did:

- unchanged, no facing bet  → CHECK (only with positive corroboration:
                              timer-bar transition or downstream action)
- unchanged, facing a bet   → WAIT. A fold is only ever emitted from the
                              positive `SeatObs.folded` signal (Fix J),
                              never inferred from "no chips added"
                              (review 2026-09-20 I3).
- equal to `to_call`        → CALL (±1 engine-chip rounding tolerance)
- stack 0, commit <= bet    → CALL (all-in call for less; the engine
                              rejects it as a raise — review I4)
- exceeds `to_call`         → RAISE to that new total
- stack 0, commit > bet     → RAISE (short shove — engine routes via its
                              short-shove path)

When residual deltas can't be explained, we emit an OcrWarning (or just
stop the walk) rather than raising; the next poll gets another shot.

Baseline policy (review 2026-09-20 I7 — this replaces the old claim that
`last_fs` "only advances on accepted reads", which was never true)
------------------------------------------------------------------------
`last_fs` IS replaced on every processed frame — boards, fold flags,
ovals and actor reads always come from the newest frame, which is what the
StreetReveal dedupe and the two-frame fold confirmation need. Only the
per-seat `stack_chips` baseline is conservative:

- a `None` stack read carries the previous value forward (the bet banner
  covers the label exactly on the action tick), minus whatever the walk
  already explained for that seat this pass via oval+banner, so the same
  chips are not counted twice next tick;
- a stack DROP the walk did not explain is NOT swallowed for a seat that
  has shown unexplained chip evidence this street (bet banner, or an oval
  above the engine's commit): the previous stack is kept so the drop is
  still there when the walk finally reaches that seat. Drops with no such
  evidence are absorbed, so a one-frame stack misread cannot come back as
  a bet;
- at a street boundary (StreetReveal, engine street change) and while no
  board card is on screen (antes posting) the newest stacks are adopted
  wholesale — prior-street evidence is moot and the server's reveal
  reconcilers own those gaps.

Frames in which no seat reads in-hand (dark / dimmed capture) are dropped
entirely and never become the baseline.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace

from plo5bp.ocr.types import Card, FrameState

logger = logging.getLogger("plo5bp.ocr")


def _timer_debug_enabled() -> bool:
    return bool(os.environ.get("PLO5BP_OCR_DEBUG_TIMER"))


# A bet and its call can differ by one engine-chip purely from rounding:
# at a non-integer `chips_per_cent` (e.g. 2.5 at $40/bb),
# round(c1*s) + round((c2-c1)*s) != round(c2*s) for ~1/3 of amounts, so a
# call recovered via the stack-drop path (base + drop) lands one chip off
# the facing bet that was derived from an oval read. No legal action is a
# 1-chip raise, so a +/-1 mismatch IS the call (review 2026-09-20, H13).
_CALL_ROUNDING_TOLERANCE_CHIPS = 1

# Upper bound on consecutive ticks the street gate may suppress the walk
# (see `EventReconstructor.step`). The screen trails the engine by the
# chip-sweep + deal animation (~1-2 s); a board card that never classifies
# (rank NCC below its floor returns None) would otherwise gate the walk for
# the rest of the hand, turning "degraded" into "stuck".
_MAX_STREET_GATE_TICKS = 25


def cents_to_engine_chips(cents: int, chips_per_cent: float) -> int:
    """THE cents → engine-chips conversion for the live-capture stack.

    One float multiply, one `round` — every conversion in this module goes
    through here so a bet and its call can never round through different
    op orders (review 2026-09-20, rounding). `chips_per_cent` is
    ``cfg.bb / (100 * dollars_per_bb)`` exactly as `EngineView` carries it;
    server-side seeding should call this too so both sides stay identical
    by construction. Python's `round` (half-to-even) is deliberate: it is
    what the server has always used, and changing the tie rule on one side
    only would CREATE mismatches.
    """
    return int(round(int(cents) * float(chips_per_cent)))


@dataclass(frozen=True)
class EngineView:
    """Read-only engine snapshot passed to `EventReconstructor.step`.

    The reconstructor does not mutate the engine; it just needs to know
    whose turn it is and what commits/stacks the engine currently
    believes in so it can infer what changed between polls.

    `chips_per_cent` is the scale factor to convert OCR-cent chip reads
    (``obs.committed_chips`` etc.) into engine-chip units so every
    arithmetic step in the reconstructor stays in one unit. At the
    defaults (bb=10000, $20/bb) this is 5.0.
    """

    num_seats: int
    current_actor: int | None
    street: int  # 0=preflop, 1=flop, 2=turn, 3=river, 4=showdown
    awaiting_next_street: int | None  # 2=flop→turn, 3=turn→river
    button_seat: int
    committed_this_street: tuple[int, ...]
    stacks: tuple[int, ...]
    folded: tuple[bool, ...]
    all_in: tuple[bool, ...]
    bet_to_call: int
    chips_per_cent: float
    # Minimum stack-delta (in OCR cents) accepted as a bet when the
    # `committed_chips` OCR misses. Any drop below this floor is treated
    # as Tesseract jitter (`1455.13` → `1455` → back → phantom $0.13
    # bet). Derived from `cfg.bb` on the server side; a full 1bb is
    # always a safe floor since no legal action is smaller.
    min_bet_cents: int = 0
    # Seats that were sitting out when the hand began — tracked on the
    # server in `session.sitting_out_seats` but NOT reflected in the
    # engine's `folded` array between polls (the engine only learns
    # about sitting-out seats via `_auto_fold_sitting_out` during
    # replay). Surfaced here so the walk can skip them and avoid
    # emitting phantom FOLD events for seats with no cards.
    sitting_out: tuple[bool, ...] = ()
    # True for sources whose `SeatObs.folded` is EXACT (PokerNow: the DOM
    # `fold` class). Since the 2026-09-20 review the walk never infers a fold
    # from "faced a bet but added no chips" on ANY source (I3 — it used to,
    # on the OCR path, and folded seats that simply had not acted yet), so
    # this flag now only decides how much a `folded` read is trusted:
    #   True  → a single frame is authoritative (the DOM does not flicker,
    #           and frames only arrive on change);
    #   False → pixel OCR: the fold must be seen on two consecutive frames
    #           before it is emitted (I6).
    exact_folds: bool = False


@dataclass(frozen=True)
class HandStart:
    """A new hand is about to begin. Fires on the first FrameState we
    see with flop cards present, or after a HandEnd when fresh flop
    cards show up."""

    button_seat: int
    starting_stacks: tuple[int | None, ...]
    in_hand_mask: tuple[bool, ...] = ()


@dataclass(frozen=True)
class HeroHoleRevealed:
    """Hero's 5 hole cards are visible. Card indices are `rank*4+suit`
    (the encoding used by `Session.hero_hole` and the rust engine)."""

    cards: tuple[int, ...]


@dataclass(frozen=True)
class StreetReveal:
    """Turn or river card(s) appeared on both boards. Double-board
    bomb-pot reveals board-A and board-B together in a single engine
    call (`set_turn(a, b)` / `set_river(a, b)`)."""

    street: str  # "turn" or "river"
    board_a_card: int  # card index 0..51
    board_b_card: int


@dataclass(frozen=True)
class SeatAction:
    """A seat took a recognizable action. `chips` semantics:
    - fold / check_call → 0 (engine computes the actual amount)
    - raise → engine-chip DELTA the actor adds above their current
      street_commit (raise-by). Matches the signature of
      ``apply_raise_chips`` so the server replays it unchanged.
      Stack-bound short shoves are emitted as gate="raise" with
      chips=delta — the engine's `apply_raise_chips` already routes
      short shoves through its own short-shove path."""

    seat: int
    gate: str  # "fold" / "check_call" / "raise"
    chips: int


@dataclass(frozen=True)
class HandEnd:
    """The current hand ended. Caller should reset session state so the
    next StreetReveal / SeatAction starts fresh."""


@dataclass(frozen=True)
class OcrWarning:
    """Reconstructor couldn't fully explain a state delta. Caller should
    log and keep going — the next poll often resolves the ambiguity."""

    message: str


OcrEvent = (
    HandStart
    | HeroHoleRevealed
    | StreetReveal
    | SeatAction
    | HandEnd
    | OcrWarning
)


def _card_to_index(c: Card) -> int:
    """Convert a `Card(rank, suit)` to the engine's 0..51 index."""
    return c.rank * 4 + c.suit


def _hole_as_indices(hole: tuple[Card | None, ...]) -> tuple[int, ...] | None:
    """Return 5 card indices if all slots populated, else None."""
    if any(c is None for c in hole):
        return None
    return tuple(_card_to_index(c) for c in hole)  # type: ignore[arg-type]


def _count_cards(board: tuple[Card | None, ...]) -> int:
    n = 0
    for c in board:
        if c is None:
            break
        n += 1
    return n


def _resolve_active_actor(fs: FrameState) -> int | None:
    """Return the unique seat with `is_actor=True` in `fs`, else None.

    Zero positive reads (OCR missed, between turns) and multiple
    positive reads (animation frame with two bars briefly visible)
    both resolve to None — callers should treat None as "no update
    this tick" rather than "no active actor". The reconstructor's
    `_observed_active_actor` therefore stays pinned to the last
    positive single-seat reading across flicker.
    """
    actors = [s.seat for s in fs.seats if s.is_actor]
    if len(actors) != 1:
        return None
    return actors[0]


# NOTE (review 2026-09-20 I6): the Task-C `_hero_hole_just_hid` helper and
# its "hero's hole cards hid => hero CHECKed" walk branch were DELETED.
# ClubGG flips hero's cards face-up on hero's first flop turn and keeps
# them face-up through showdown — they never re-hide mid-hand — so the
# visible→hidden diff could only ever fire on a capture glitch (one frame
# where none of hero's five cards classify), where it emitted a phantom
# CHECK. Do not resurrect it; hero's silent CHECK needs a real signal.


def _visible_street(fs: FrameState) -> int:
    """Street implied by the boards ON SCREEN: 1=flop, 2=turn, 3=river.

    Uses the shorter board (both boards deal together). Anything short of
    a full turn reads as the flop — this is a lower bound used to notice
    the ENGINE running ahead of the screen, never to advance anything.
    """
    n = min(_count_cards(fs.board_a), _count_cards(fs.board_b))
    return 1 + int(n >= 4) + int(n >= 5)


def _any_board_card(fs: FrameState) -> bool:
    return any(c is not None for c in fs.board_a) or any(
        c is not None for c in fs.board_b
    )


@dataclass
class _WalkResult:
    """What one `_infer_seat_actions` pass produced, plus the bookkeeping
    `step` needs to store an honest stack baseline afterwards."""

    events: list = field(default_factory=list)  # list[SeatAction]
    warnings: list = field(default_factory=list)  # list[OcrWarning]
    # seat -> engine-chips the walk attributed to that seat this pass.
    explained: dict = field(default_factory=dict)
    # Engine commits updated with this pass's actions; None = walk not run.
    base_commit: list | None = None
    # True when the timer-bar CHECK branch would have fired but the actor's
    # stack was unreadable — `step` keeps the lock so the transition can be
    # re-evaluated next tick instead of being lost (it is a one-shot edge).
    hold_timer_lock: bool = False


class EventReconstructor:
    """Stateful diff-to-events translator.

    One reconstructor per live session. Call `step(fs, engine_view)`
    on each poll; call `reset()` on session reset.
    """

    def __init__(self, num_seats: int) -> None:
        self.num_seats = int(num_seats)
        self.last_fs: FrameState | None = None
        self.hero_hole_emitted: bool = False
        # Last seat positively detected as the active actor via the
        # yellow turn-timer bar (see cards.has_active_timer_bar). Only
        # advances on positive single-seat reads — missed reads (no
        # seat is_actor=True) leave it pinned to the previous value,
        # so OCR flicker and time-bank refills don't cause false
        # transitions. Cleared on reset/rebaseline.
        self._observed_active_actor: int | None = None
        # Seats that have shown chip evidence the walk has not explained
        # yet (bet banner / oval above the engine's commit). A stack drop
        # on such a seat is held in the baseline instead of being absorbed
        # (see `_store_baseline`). Cleared per seat when the walk explains
        # it, and wholesale at a street boundary / reset / rebaseline.
        self._unexplained_evidence_seats: set[int] = set()
        # Engine street seen on the previous processed frame — a change
        # marks a street boundary for the baseline policy.
        self._last_engine_street: int | None = None
        # Consecutive ticks the street gate has suppressed the walk.
        self._street_gate_ticks: int = 0

    def _clear_tracking(self) -> None:
        self._observed_active_actor = None
        self._unexplained_evidence_seats = set()
        self._last_engine_street = None
        self._street_gate_ticks = 0

    def reset(self) -> None:
        self.last_fs = None
        self.hero_hole_emitted = False
        self._clear_tracking()

    def rebaseline(self, fs: FrameState) -> None:
        """Adopt `fs` as the new diff baseline without emitting events.

        Called by the server when a hand-start is detected and the
        reconstructor's previous baseline is no longer meaningful
        (e.g. the action log was just cleared). Leaves
        `hero_hole_emitted` matching whether hero hole is fully
        visible now, so the next tick doesn't double-emit.
        """
        self.last_fs = fs
        self.hero_hole_emitted = _hole_as_indices(fs.hero_hole) is not None
        self._clear_tracking()

    def step(self, fs: FrameState, engine_view: EngineView) -> list[OcrEvent]:
        events: list[OcrEvent] = []

        # Unreadable-capture guard (review 2026-09-20 I6). A dark / dimmed
        # frame (modal dialog, capture hiccup, window minimised) reads
        # EVERY seat as folded: no card backs, no banner, no chips. Nothing
        # can be inferred from it, and letting it in is actively harmful —
        # it used to fold every live seat in one tick (irreversible: the
        # server's `folded_this_hand` is sticky) and then become the diff
        # baseline. Drop the frame entirely: no events, baseline untouched,
        # so the next good frame diffs against the last good one. Between
        # hands (everyone mucked) this is equally a no-op.
        if fs.seats and all(s.folded for s in fs.seats):
            return events

        # First observation: adopt as baseline, emit hero hole if
        # already visible. Server-side owns hand-start detection.
        if self.last_fs is None:
            hole = _hole_as_indices(fs.hero_hole)
            if hole is not None:
                events.append(HeroHoleRevealed(cards=hole))
                self.hero_hole_emitted = True
            self.last_fs = fs
            # Adopt the bootstrap frame's active-actor read too, so the
            # second tick has a `prev_active` to compare against. Without
            # this, detecting a CHECK requires three observations
            # (bootstrap, lock-update, transition) instead of two.
            self._observed_active_actor = _resolve_active_actor(fs)
            self._last_engine_street = int(engine_view.street)
            return events

        last = self.last_fs

        # Hero hole reveal (first time we see all 5 after they were
        # missing). Fires once per hand.
        if not self.hero_hole_emitted:
            hole = _hole_as_indices(fs.hero_hole)
            if hole is not None:
                events.append(HeroHoleRevealed(cards=hole))
                self.hero_hole_emitted = True

        # Street reveals. A "turn" reveal is the 4th card appearing on
        # BOTH boards; "river" is the 5th on both. We only emit when
        # both boards have caught up so the engine's paired set_turn /
        # set_river calls have their inputs ready.
        last_a, last_b = _count_cards(last.board_a), _count_cards(last.board_b)
        new_a, new_b = _count_cards(fs.board_a), _count_cards(fs.board_b)
        if new_a >= 4 and new_b >= 4 and (last_a < 4 or last_b < 4):
            ca, cb = fs.board_a[3], fs.board_b[3]
            assert ca is not None and cb is not None
            events.append(
                StreetReveal(
                    street="turn",
                    board_a_card=_card_to_index(ca),
                    board_b_card=_card_to_index(cb),
                )
            )
        if new_a >= 5 and new_b >= 5 and (last_a < 5 or last_b < 5):
            ca, cb = fs.board_a[4], fs.board_b[4]
            assert ca is not None and cb is not None
            events.append(
                StreetReveal(
                    street="river",
                    board_a_card=_card_to_index(ca),
                    board_b_card=_card_to_index(cb),
                )
            )

        # Active-actor signal: snapshot the prior tick's lock and
        # resolve this frame's positive read. The walk uses (prev,
        # now) to detect "the timer bar moved off this seat" as a
        # positive CHECK signal. Update the lock AFTER the walk so
        # the walk sees the prior value as `prev`.
        prev_active = self._observed_active_actor
        now_active = _resolve_active_actor(fs)

        if _timer_debug_enabled():
            raw_actors = [s.seat for s in fs.seats if s.is_actor]
            is_actor_per_seat = [bool(s.is_actor) for s in fs.seats]
            logger.warning(
                "ocr.timer: prev=%s now=%s raw_actors=%s "
                "is_actor_per_seat=%s engine_actor=%s street=%s "
                "bet_to_call=%s fs.boards=(a=%d, b=%d)",
                prev_active, now_active, raw_actors, is_actor_per_seat,
                engine_view.current_actor, engine_view.street,
                engine_view.bet_to_call,
                _count_cards(fs.board_a), _count_cards(fs.board_b),
            )

        # --- walk gates ------------------------------------------------
        #
        # No board card on screen (review 2026-09-20, latent "no
        # flop-present gate"): a bomb pot has no pre-flop betting, so any
        # chip movement before the flop is the ANTE being deducted — the
        # engine posts antes itself. Walking here read a uniform ante drop
        # as a bet plus calls and took the hand to the turn. Deliberately
        # the weakest possible test (ANY readable card on either board
        # opens the gate) so one unclassifiable flop card can never block
        # action inference.
        board_visible = _any_board_card(fs)

        # Street gate (review 2026-09-20 I2). The server's `_rebuild_env`
        # advances the engine to the next street with PADDED cards the
        # moment betting closes, so for the length of the chip-sweep +
        # deal animation the engine is a street AHEAD of the screen. The
        # frame still shows the previous street's ovals/banners, and the
        # engine now says "commits 0, nobody has acted": explaining that
        # stale evidence produced a phantom CHECK cascade (4 checks in one
        # tick), a phantom RAISE when the closer's banner was still up,
        # and on PokerNow one extra frame took a hand to showdown. Nothing
        # on screen belongs to the engine's street yet — wait for the
        # board to catch up. Bounded, because a turn/river card that never
        # classifies would otherwise suppress the walk for the whole hand;
        # the budget restarts whenever the engine moves to another street.
        engine_street = int(engine_view.street)
        engine_street_changed = (
            self._last_engine_street is not None
            and engine_street != self._last_engine_street
        )
        engine_ahead = engine_street > _visible_street(fs)
        if engine_ahead and not engine_street_changed:
            self._street_gate_ticks += 1
        else:
            self._street_gate_ticks = 1 if engine_ahead else 0
        street_gated = (
            engine_ahead and self._street_gate_ticks <= _MAX_STREET_GATE_TICKS
        )

        # Seat actions: walk the expected-actor queue from the engine's
        # current_actor forward and explain each seat's commit/stack
        # delta.
        if board_visible and not street_gated:
            walk = self._infer_seat_actions(
                last, fs, engine_view, prev_active, now_active
            )
        else:
            walk = _WalkResult()
            if _timer_debug_enabled():
                logger.warning(
                    "ocr.walk: skipped board_visible=%s engine_street=%s "
                    "visible_street=%s gate_ticks=%d",
                    board_visible, engine_view.street, _visible_street(fs),
                    self._street_gate_ticks,
                )
        events.extend(walk.events)
        events.extend(walk.warnings)

        if now_active is not None and not walk.hold_timer_lock:
            self._observed_active_actor = now_active

        # Street boundary ⇒ adopt the newest stacks wholesale (see
        # `_store_baseline`): a reveal this tick, the engine's street
        # moving since the last processed frame, or no board on screen.
        street_boundary = (
            not board_visible
            or engine_street_changed
            or any(isinstance(e, StreetReveal) for e in events)
        )
        self._last_engine_street = engine_street
        self._store_baseline(last, fs, engine_view, walk, street_boundary)
        return events

    # -- helpers ------------------------------------------------------

    def _store_baseline(
        self,
        last: FrameState,
        fs: FrameState,
        engine_view: EngineView,
        walk: _WalkResult,
        street_boundary: bool,
    ) -> None:
        """Store `fs` as the next diff baseline with a conservative
        per-seat `stack_chips` (module docstring: "Baseline policy").

        Everything except `stack_chips` always comes from `fs`:
        `committed_chips`'s None/0 semantics are load-bearing in the ladder
        (a None/0 commit legitimately means "no action") so it is NOT
        carried forward, and board/button/actor/fold fields must be fresh
        for the StreetReveal dedupe and the two-frame fold confirmation.

        Three stack rules:

        1. None read → carry the previous value forward. ClubGG's blue
           bet-banner covers the stack label on the action tick, so a
           bettor's stack OCR-reads None exactly when we most need it;
           baking the None in made next tick's stack-drop incomputable and
           the bet (whose oval reads late) permanently lost (Bug #8).
           review 2026-09-20 I7/H17: if the walk ALREADY explained this
           seat's action this pass (via oval + banner), the carried value
           is reduced by that amount — otherwise next tick's readable
           stack shows the same chips again as a fresh drop and emits a
           phantom raise.
        2. Unexplained drop → keep the previous value for a seat that has
           shown unexplained chip evidence this street (review 2026-09-20
           I7/H2). The walk often cannot reach a seat on the tick its
           stack drops (it stops behind a raise it just emitted; a
           snap-caller acts before the upstream seat is resolved). Storing
           the lower stack consumed the only corroboration Fix N would
           accept once the banner faded, so the call was lost for good.
           "Evidence" = a bet banner, or an oval at least 1bb above the
           engine's commit, seen on ANY tick since the seat's last
           explained action (`_unexplained_evidence_seats`). It has to be
           remembered rather than required on the drop tick: the banner
           hides the stack label, so the drop typically becomes readable
           only AFTER the banner has faded, and the oval flickers to None.
           A drop on a seat with no such evidence is absorbed exactly as
           before — a one-frame stack misread must not resurface later in
           the street as a bet.
        3. Street boundary → adopt everything and forget the evidence.
        """
        scale = float(engine_view.chips_per_cent)
        min_bet_engine = cents_to_engine_chips(
            int(engine_view.min_bet_cents), scale
        )
        base_commit = (
            walk.base_commit
            if walk.base_commit is not None
            else list(engine_view.committed_this_street)
        )
        evidence = self._unexplained_evidence_seats
        if street_boundary:
            evidence.clear()

        prev_seats = {s.seat: s for s in last.seats}
        merged: list = []
        for s in fs.seats:
            prev = prev_seats.get(s.seat)
            prev_stack = prev.stack_chips if prev is not None else None
            explained_chips = int(walk.explained.get(s.seat, 0))

            if explained_chips > 0:
                evidence.discard(s.seat)  # accounted for — start clean
            elif not street_boundary and self._shows_unexplained_chips(
                s, base_commit, scale, min_bet_engine
            ):
                evidence.add(s.seat)

            if s.stack_chips is None:
                if prev_stack is not None:
                    carried = int(prev_stack)
                    if explained_chips > 0 and scale > 0:
                        carried = max(
                            0, carried - int(round(explained_chips / scale))
                        )
                    s = replace(s, stack_chips=carried)
            elif (
                s.seat in evidence
                and prev_stack is not None
                and int(s.stack_chips) < int(prev_stack)
            ):
                s = replace(s, stack_chips=prev_stack)  # hold the drop
            merged.append(s)

        self.last_fs = replace(fs, seats=tuple(merged))

    @staticmethod
    def _shows_unexplained_chips(
        obs, base_commit: list, scale: float, min_bet_engine: int
    ) -> bool:
        """True when the seat visibly has chips the engine does not know
        about: a live bet banner, or an oval >= 1bb above its commit."""
        if bool(getattr(obs, "bet_banner", False)):
            return True
        if obs.committed_chips is None or not (0 <= obs.seat < len(base_commit)):
            return False
        shown = cents_to_engine_chips(int(obs.committed_chips), scale)
        return shown - int(base_commit[obs.seat]) >= max(1, min_bet_engine)

    def _infer_seat_actions(
        self,
        last: FrameState,
        fs: FrameState,
        engine_view: EngineView,
        prev_active: int | None,
        now_active: int | None,
    ) -> _WalkResult:
        result = _WalkResult()
        events: list[SeatAction] = result.events
        warnings: list[OcrWarning] = result.warnings

        if engine_view.current_actor is None:
            return result

        new_seats = {s.seat: s for s in fs.seats}
        # Prior-frame seat obs keyed by seat — used by the stack-delta
        # fallback when `committed_chips` OCR misses but the seat's
        # stack clearly decreased between polls.
        last_seats = {s.seat: s for s in last.seats}

        # Everything below runs in engine-chips. OCR reads
        # `committed_chips` / `stack_chips` in cents; convert once at
        # the boundary so all `delta`/`to_call`/`facing_bet` arithmetic
        # stays in one unit.
        scale = float(engine_view.chips_per_cent)

        def _to_engine(cents: int | None) -> int | None:
            # Single conversion path (review 2026-09-20, rounding): every
            # cents→chips step in this module is `cents_to_engine_chips`.
            if cents is None:
                return None
            return cents_to_engine_chips(int(cents), scale)

        tol = _CALL_ROUNDING_TOLERANCE_CHIPS

        base_commit = list(engine_view.committed_this_street)
        result.base_commit = base_commit
        folded = list(engine_view.folded)
        # Local copy: a seat that goes all-in DURING this pass must drop
        # out of the round-closed test and the walk just like one the
        # engine already knows about.
        all_in = list(engine_view.all_in)
        # Seats this pass already attributed an action to. The walk never
        # visits a seat twice (review 2026-09-20 I2): wrapping back onto a
        # seat that already acted can only re-read the same evidence, and
        # that is exactly how one stale frame became a 4-CHECK cascade.
        acted: set[int] = set()
        # Normalize `sitting_out` to `num_seats` length — callers that
        # predate the field (tests, older code paths) pass an empty
        # tuple and we treat that as "nobody sitting out".
        sitting_out = list(engine_view.sitting_out) + [False] * (
            self.num_seats - len(engine_view.sitting_out)
        )
        facing_bet = int(engine_view.bet_to_call)

        # True once a raise / bet / all-in this pass bumps
        # `facing_bet`. After that, a seat with `delta == 0` hasn't
        # "folded against a facing bet" — they just haven't had a turn
        # since the bump. Stop inferring and wait for the next poll
        # rather than speculating FOLDs for seats still waiting to act.
        facing_bet_bumped = False

        actor = int(engine_view.current_actor)
        steps_left = self.num_seats * 2  # conservative guard
        skipped_in_a_row = 0
        while steps_left > 0:
            steps_left -= 1

            # Betting round closed: every seat still in the hand has matched
            # the current bet, so there are no more actions to infer this
            # street. Stop — coasting further re-emits phantom check/call (and
            # can even synthesize a phantom raise) for seats that already
            # acted. This is the closing-call frame: PokerNow shows every
            # player's matched bet simultaneously the instant the last caller
            # acts, before sweeping the chips to the pot. (Only applies once a
            # bet exists; a quiet check-round closes via the normal CHECK
            # walk + the StreetReveal reconciler.)
            if facing_bet > 0:
                still_in = [
                    i
                    for i in range(self.num_seats)
                    if not folded[i]
                    and not all_in[i]
                    and not sitting_out[i]
                ]
                if still_in and all(base_commit[i] >= facing_bet for i in still_in):
                    break

            # Wrapped onto a seat this pass already explained — stop. Tested
            # BEFORE the skip below: a seat that just folded or shoved is
            # "skippable" too, and with nobody else live the walk would
            # otherwise spin on skips until the loop guard fired.
            if actor in acted:
                break

            if folded[actor] or all_in[actor] or sitting_out[actor]:
                skipped_in_a_row += 1
                if skipped_in_a_row >= self.num_seats:
                    break  # a full lap of seats that cannot act
                actor = (actor + 1) % self.num_seats
                continue
            skipped_in_a_row = 0

            obs = new_seats.get(actor)
            if obs is None:
                break

            prev_obs = last_seats.get(actor)

            # Positive fold signal (Fix J). `obs.folded` is True when
            # extract.py sees none of the multi-signal in-hand evidence
            # (no cards_back, no bet banner, no committed chips). For a
            # just-folded seat, Tesseract typically returns `None` for
            # the empty chip oval — NOT `0` — so the fallback ladder
            # below can't infer the fold via `primary_read ==
            # base_commit`. Without this branch, the walk hits
            # `else: break` and stalls the engine's current_actor on
            # the folded seat indefinitely, which also blocks detection
            # of downstream folds in the same pass. Consulting
            # `obs.folded` directly sidesteps the OCR-oval ambiguity.
            if (
                obs.folded
                and (facing_bet - base_commit[actor]) > 0
                and not facing_bet_bumped
            ):
                # Two-frame confirmation (review 2026-09-20 I6). A fold is
                # irreversible downstream (`folded_this_hand` is sticky and
                # the seat-less action log shifts every later action), yet
                # pixel `folded` is just "no in-hand signal THIS frame" —
                # `has_cards_back` sits at 0.21-0.23 against a 0.15
                # threshold, so one occluded / mid-animation frame reads a
                # live seat as folded. Require the previous frame to agree.
                # Costs one poll (~200 ms) on a real fold. Exact sources
                # (PokerNow's DOM `fold` class) are trusted immediately:
                # they do not flicker, and frames only arrive on change, so
                # a second identical frame may never come.
                if not engine_view.exact_folds and not (
                    prev_obs is not None and prev_obs.folded
                ):
                    if _timer_debug_enabled():
                        logger.warning(
                            "ocr.walk: fold on seat=%d unconfirmed "
                            "(first frame) — waiting",
                            actor,
                        )
                    break
                events.append(SeatAction(seat=actor, gate="fold", chips=0))
                if _timer_debug_enabled():
                    logger.warning(
                        "ocr.walk: emitted seat=%d gate=fold chips=0 "
                        "branch=fix_j_obs_folded",
                        actor,
                    )
                folded[actor] = True
                acted.add(actor)
                actor = (actor + 1) % self.num_seats
                continue

            # Fallback ladder for `new_commit`:
            #   1. `committed_chips` OCR when it reports a CHANGE vs
            #      `base_commit` — authoritative. (A zero / unchanged
            #      read falls through to signals 2-3 so Tesseract
            #      misreads on the in-play chip oval can't silently
            #      erase a live bet.)
            #   2. Stack-delta (prev_stack - new_stack) gated by bet
            #      banner visibility or a >= 1 bb drop.
            #   3. Bet banner alone → emit warning, stall (no chip
            #      amount recoverable this tick).
            #   4. No signals + no facing bet + downstream evidence
            #      → presume unchanged (CHECK) so the walk can reach
            #      the seat that actually acted. Real ClubGG frames
            #      leave `committed_chips=None` for seats with no
            #      chip oval; without this coast-past rule the walk
            #      breaks at the first silent seat in the action
            #      order and never inspects the bettor downstream.
            #   5. Nothing → wait for next poll.
            primary_read = _to_engine(obs.committed_chips)
            prev_stack_cents = (
                prev_obs.stack_chips if prev_obs is not None else None
            )
            new_stack_cents = obs.stack_chips
            banner = bool(getattr(obs, "bet_banner", False))
            stack_drop_cents = None
            if (
                prev_stack_cents is not None
                and new_stack_cents is not None
                and prev_stack_cents > new_stack_cents
            ):
                stack_drop_cents = prev_stack_cents - new_stack_cents

            to_call_here = max(0, facing_bet - base_commit[actor])

            # Reject sub-1bb commit changes from `primary_read`. Tesseract
            # occasionally misreads the chip oval and returns a small
            # integer (e.g. "4" = 400 cents → 2000 engine-chips) when the
            # oval is empty, partially covered, or rendering a badge
            # animation. A real commit change is ALWAYS at least 1bb (the
            # engine's min-bet floor), unless the seat is completing an
            # exact call of a smaller facing bet (handled separately by
            # the `delta == to_call` branch). Without this guard the
            # spurious read becomes a `SeatAction(gate='raise', chips=<1bb)`
            # that the engine rejects on replay — freezing `_rebuild_env`
            # and leaving the session stuck on a stale env forever.
            min_bet_engine = cents_to_engine_chips(
                int(engine_view.min_bet_cents), scale
            )
            if primary_read is not None and min_bet_engine > 0:
                primary_delta = primary_read - base_commit[actor]
                if (
                    primary_delta != 0
                    and primary_delta != to_call_here
                    and abs(primary_delta) < min_bet_engine
                ):
                    primary_read = None

            # Corroboration guard (Fix N). A positive primary_delta must
            # be backed by an active bet banner or a matching stack drop.
            # On a freshly-folded seat the chip oval clears on-screen,
            # but Tesseract occasionally reads noise/cross-talk as a
            # positive integer (e.g. "180" from a neighbor's label). The
            # extract.py multi-signal rule then flips obs.folded=False
            # via the has_commit branch, bypassing Fix J's fold emission.
            # Without this guard the noisy read propagates into the
            # fallback ladder below and emits a phantom CHECK_CALL that
            # inflates the pot and misattributes downstream folds.
            if primary_read is not None:
                primary_delta = primary_read - base_commit[actor]
                if primary_delta > 0:
                    stack_drop_chips = _to_engine(stack_drop_cents) or 0
                    # `+ tol`: round(drop*s) can sit one chip under
                    # round(c2*s) - round(c1*s) at a non-integer scale.
                    if not (banner or stack_drop_chips + tol >= primary_delta):
                        primary_read = None

            # Zero-on-zero guard (Fix P). When the seat faces a bet
            # (to_call_here > 0), primary_read == 0 with
            # base_commit == 0 is information-free: an empty chip
            # oval can legitimately resolve to None OR 0 depending
            # on which preprocessor path runs, and both mean "seat
            # has not contributed chips this street". Without this
            # guard, branch 4 below treats the 0 as an affirmative
            # "no change". (That used to feed a `delta==0 and
            # to_call>0 → FOLD` inference — removed by review
            # 2026-09-20 I3 — and still must not be read as
            # information about a seat that is "in hand, yet to act":
            # cards visible, still thinking.) Nulling the read lets
            # the ladder corroborate via banner / stack drop or
            # break harmlessly until Fix J fires on a real fold.
            # Only fires when facing a bet — a zero-on-zero read
            # with no facing bet is the normal CHECK signal via
            # branch 4, preserved here.
            if (
                primary_read == 0
                and base_commit[actor] == 0
                and to_call_here > 0
            ):
                primary_read = None

            # Seats whose evidence must not count as "someone downstream
            # acted": seats that cannot act (folded / all-in), seats this
            # pass already explained, and the actor itself (review
            # 2026-09-20 I2 — the actor's own stale oval used to certify
            # its own phantom CHECK).
            skip_seats = acted | {actor} | {
                i for i in range(self.num_seats) if folded[i] or all_in[i]
            }

            # Timer-bar transition (see branch below). The CHECK is only
            # trusted with a readable, un-dropped stack on both frames
            # (review 2026-09-20, reviewer F10 / H15): when the actor BET
            # but the first post-action frame had stack None + oval
            # mid-animation (hero's banner is suppressed while cards are
            # face-up), the bare bar movement emitted a CHECK and the real
            # bet then landed on the NEXT seat. A readable drop >= 1bb is
            # already consumed by branch 2 above, so "both reads present"
            # is exactly "readable and unchanged" here.
            timer_moved = (
                to_call_here == 0
                and prev_active == actor
                and now_active is not None
                and now_active != actor
            )
            stacks_readable = (
                prev_stack_cents is not None and new_stack_cents is not None
            )

            new_commit: int | None
            if primary_read is not None and primary_read != base_commit[actor]:
                new_commit = primary_read
            elif stack_drop_cents is not None and (
                banner
                or stack_drop_cents >= int(engine_view.min_bet_cents)
            ):
                new_commit = base_commit[actor] + cents_to_engine_chips(
                    int(stack_drop_cents), scale
                )
            elif banner:
                warnings.append(
                    OcrWarning(
                        f"seat {actor}: bet banner visible but no "
                        "chip amount yet; retrying"
                    )
                )
                break
            elif primary_read is not None and actor != int(engine_view.current_actor):
                # Primary says "no change" and no corroborating
                # evidence of change — trust the read **only for
                # intermediate seats** the walk is coasting past
                # (their action is already accounted for in
                # `base_commit`). For the engine's own
                # `current_actor`, an empty chip oval can legitimately
                # read as either `None` or `0` depending on Tesseract's
                # preprocessing path, and `0 == base_commit` means
                # "seat hasn't contributed chips this street" — i.e.,
                # they haven't acted yet. Without this guard, an
                # actor whose oval reads 0 on the first tick of a
                # new street emits a phantom CHECK that advances the
                # engine past them silently. Positive corroboration
                # (timer-bar transition, downstream activity) lives in
                # the branches below.
                new_commit = primary_read
            elif timer_moved and stacks_readable:
                # Active-actor transition signal: the yellow turn-timer
                # bar was on `actor` last tick and has moved to a
                # different seat this tick, with no facing bet here.
                # That's a positive "this seat checked and the turn
                # passed" reading even when no chip evidence exists
                # downstream (e.g. hero is first-to-act on a postflop
                # street and CHECKs into nobody yet — the existing
                # `_any_remaining_delta` branch below requires
                # downstream activity that doesn't exist). Walk on
                # so the next seat in the loop can be inspected.
                new_commit = base_commit[actor]
                if _timer_debug_enabled():
                    logger.warning(
                        "ocr.walk: timer_bar_branch_fired seat=%d "
                        "prev=%s now=%s",
                        actor, prev_active, now_active,
                    )
            elif to_call_here == 0 and self._any_remaining_delta(
                new_seats,
                last_seats,
                base_commit,
                scale,
                actor_start=(actor + 1) % self.num_seats,
                min_bet_cents=int(engine_view.min_bet_cents),
                sitting_out=sitting_out,
                skip=skip_seats,
            ):
                # No direct evidence for this seat but someone
                # downstream clearly acted — presume CHECK and
                # walk on.
                new_commit = base_commit[actor]
            else:
                if timer_moved and not stacks_readable:
                    # The bar DID move off this seat but its stack is
                    # unreadable this tick, so CHECK-vs-BET is undecided.
                    # The transition is a one-shot edge (the lock would
                    # advance and `prev_active == actor` never be true
                    # again), so keep the lock and re-evaluate next tick:
                    # a readable unchanged stack then yields the CHECK, a
                    # dropped one yields the bet via branch 2.
                    result.hold_timer_lock = True
                if _timer_debug_enabled():
                    logger.warning(
                        "ocr.walk: stalled actor=%d to_call=%d "
                        "primary_read=%s banner=%s stack_drop_cents=%s "
                        "prev_active=%s now_active=%s hold_lock=%s",
                        actor, to_call_here, primary_read, banner,
                        stack_drop_cents, prev_active, now_active,
                        result.hold_timer_lock,
                    )
                break

            delta = new_commit - base_commit[actor]
            to_call = max(0, facing_bet - base_commit[actor])

            if delta == 0:
                if to_call > 0:
                    # Facing a bet with no chips added: WAIT — never infer
                    # a fold here (review 2026-09-20 I3). This is only
                    # reachable for a seat the walk is COASTING past whose
                    # oval still equals its engine commit, i.e. a seat
                    # with chips in that simply has not acted since the
                    # raise. It used to be folded the instant an upstream
                    # seat called/folded in the same pass (cards still
                    # visible, `obs.folded` False) — irreversibly, and
                    # every later action then landed on the wrong seat.
                    # "No new chips" cannot tell thinking from folded;
                    # real folds arrive through the positive `obs.folded`
                    # signal (Fix J above, two-frame confirmed). The old
                    # sub-cases collapse into this one rule:
                    #   * downstream of a raise emitted this pass
                    #     (`facing_bet_bumped`) — waiting, not folded;
                    #   * exact-fold source (PokerNow) not flagged folded
                    #     — a closing call whose chips were swept;
                    #   * `obs.folded` True but Fix J declined (raise this
                    #     pass) — still not safe to fold.
                    # Trade-off: a seat that folds WITH chips in front
                    # keeps `folded=False` (its oval is an in-hand signal
                    # in extract.py) until the street's chips are swept;
                    # the engine waits on it until then, and Fix J / the
                    # server's StreetReveal fold-reconcile pick it up.
                    if _timer_debug_enabled():
                        logger.warning(
                            "ocr.walk: seat=%d faces %d with no new "
                            "chips — waiting (bumped=%s folded=%s)",
                            actor, to_call, facing_bet_bumped, obs.folded,
                        )
                    break
                # No facing bet → CHECK.
                events.append(
                    SeatAction(seat=actor, gate="check_call", chips=0)
                )
                if _timer_debug_enabled():
                    logger.warning(
                        "ocr.walk: emitted seat=%d gate=check_call "
                        "chips=0 branch=delta0_tocall_zero",
                        actor,
                    )
                acted.add(actor)
                actor = (actor + 1) % self.num_seats
                if not self._any_remaining_delta(
                    new_seats,
                    last_seats,
                    base_commit,
                    scale,
                    actor_start=actor,
                    min_bet_cents=int(engine_view.min_bet_cents),
                    sitting_out=sitting_out,
                    skip=acted | {
                        i
                        for i in range(self.num_seats)
                        if folded[i] or all_in[i]
                    },
                ):
                    break
                continue

            if delta > 0:
                new_stack = _to_engine(obs.stack_chips)
                stack_zero = new_stack is not None and new_stack == 0

                # Exact call — tested BEFORE the stack-zero branch (review
                # 2026-09-20 I4) and within a 1-chip rounding tolerance
                # (H13, see `_CALL_ROUNDING_TOLERANCE_CHIPS`). The engine
                # applies a call for exactly `to_call`, so the walk's own
                # commit is pinned to `facing_bet` — leaving it one chip
                # short would defeat the round-closed test above.
                if to_call > 0 and abs(delta - to_call) <= tol:
                    events.append(
                        SeatAction(seat=actor, gate="check_call", chips=0)
                    )
                    if _timer_debug_enabled():
                        logger.warning(
                            "ocr.walk: emitted seat=%d gate=check_call "
                            "chips=0 branch=exact_call delta=%d to_call=%d",
                            actor, int(delta), int(to_call),
                        )
                    result.explained[actor] = (
                        result.explained.get(actor, 0) + int(delta)
                    )
                    base_commit[actor] = facing_bet
                    if stack_zero:
                        all_in[actor] = True
                    acted.add(actor)
                    actor = (actor + 1) % self.num_seats
                    continue

                # All-in CALL for less (review 2026-09-20 I4). Stack hit 0
                # and the seat still has not matched the bet: that is a
                # call, not a raise. It used to be emitted as
                # gate="raise"; the engine rejects it (ALL_IN is illegal
                # when the shove does not exceed `bet_to_call`, and
                # `apply_raise_chips` refuses with min_raise == 0),
                # `_rebuild_env` silently drops the entry, and the engine
                # stays parked on that seat for the rest of the hand.
                # CHECK_CALL makes the engine put the short stack all-in.
                if stack_zero and new_commit < facing_bet:
                    events.append(
                        SeatAction(seat=actor, gate="check_call", chips=0)
                    )
                    if _timer_debug_enabled():
                        logger.warning(
                            "ocr.walk: emitted seat=%d gate=check_call "
                            "chips=0 branch=allin_call_short commit=%d "
                            "facing=%d",
                            actor, int(new_commit), int(facing_bet),
                        )
                    result.explained[actor] = (
                        result.explained.get(actor, 0) + int(delta)
                    )
                    base_commit[actor] = new_commit
                    all_in[actor] = True
                    acted.add(actor)
                    actor = (actor + 1) % self.num_seats
                    continue

                # Sub-facing-bet guard (Fix Q). delta > 0 but
                # new_commit < facing_bet (and the seat is NOT all-in)
                # means the ladder wants to emit a "raise" whose target
                # commit is below the current bet. That's not a legal
                # poker action — apply_raise_chips rejects it and
                # _rebuild_env logs "skipping illegal action". The
                # derivation is noise (sub-1bb stack drift, or a chip-oval
                # misread that survived Fix N corroboration by
                # coincidence). Break and let the next poll re-observe.
                if new_commit < facing_bet:
                    break

                # Otherwise a raise. `chips` is the raise-by delta in
                # engine-chips that `apply_raise_chips` consumes. A stack
                # that hit 0 here is a (possibly short) shove ABOVE the
                # bet — still gate="raise"; the engine's step_hybrid
                # routes sub-min-raise shoves through its ALL_IN path.
                events.append(
                    SeatAction(seat=actor, gate="raise", chips=int(delta))
                )
                if _timer_debug_enabled():
                    logger.warning(
                        "ocr.walk: emitted seat=%d gate=raise chips=%d "
                        "branch=%s (new_commit=%d facing_bet=%d "
                        "primary_read=%s banner=%s stack_drop_cents=%s)",
                        actor, int(delta),
                        "stack_zero" if stack_zero else "raise",
                        new_commit, facing_bet,
                        primary_read, banner, stack_drop_cents,
                    )
                result.explained[actor] = (
                    result.explained.get(actor, 0) + int(delta)
                )
                base_commit[actor] = new_commit
                if new_commit > facing_bet:
                    facing_bet = new_commit
                    facing_bet_bumped = True
                if stack_zero:
                    all_in[actor] = True
                acted.add(actor)
                actor = (actor + 1) % self.num_seats
                continue

            # Negative delta — street probably closed and commits
            # reset. Stop inferring; let the next poll recover via a
            # StreetReveal.
            break
        else:
            # `while ... else`: only when the budget ran out WITHOUT a
            # break. (The old `if steps_left == 0` also fired when a
            # legitimate break happened to land on the last iteration.)
            warnings.append(OcrWarning("action inference hit loop guard"))

        return result

    def _any_remaining_delta(
        self,
        new_seats: dict,
        last_seats: dict,
        base_commit: list[int],
        scale: float,
        actor_start: int,
        min_bet_cents: int,
        sitting_out: list[bool] | None = None,
        skip: set[int] | frozenset[int] | None = None,
    ) -> bool:
        """True if any downstream seat shows evidence of action.

        Two independent signals — the walk continues past a CHECK if
        EITHER fires, so a single-OCR-miss on the in-play chip oval
        can't prematurely break inference:
          A. `committed_chips` exceeds `base_commit` by >= 1bb AND is
             corroborated (bet banner, or a stack drop covering it).
          B. Stack dropped >= min_bet_cents vs last frame.

        Signal A requires a >= 1bb delta for the same reason the main
        walk's `primary_read` guard does: Tesseract regularly misreads
        the chip oval as a small integer (`"4"` = 400 cents → 2000
        engine-chips) when the badge is mid-animation or partially
        occluded. Without the threshold the phantom signal never
        clears, the walk coast-pasts through every seat emitting
        CHECKs, and the hand ends prematurely at showdown.

        Signal A also needs the SAME corroboration the walk's Fix N
        demands of a positive oval read (review 2026-09-20 I2). An oval
        nobody's stack paid for and no banner announces is either
        Tesseract cross-talk or a STALE previous-street oval that has not
        been swept yet; the walk itself would reject it on arrival (Fix
        N), so letting it certify "someone downstream acted" only
        manufactured CHECKs for every seat in between — with two stale
        ovals it ping-ponged into four CHECKs in one tick. A negative
        delta (oval below the engine's commit) is left as it was: it
        cannot occur while the presumed-CHECK branches are reachable in an
        ante-only game (all commits are 0 when `to_call == 0`).

        Banner-only ("Bet" overlay visible but no chip amount) is
        deliberately NOT a signal here. Branch 3 of the main walk
        already breaks-with-warning when it visits a banner-only
        seat, so counting it as "remaining delta" only causes
        intermediate seats to be coast-checked when the walk would
        have stalled at the banner anyway. A persistent false-positive
        banner (e.g., hero's face-up hole cards under anti-collusion)
        would otherwise drive a phantom CHECK on every upstream seat
        on every tick.

        Sitting-out seats are skipped entirely. Stale bet badges or
        prior-hand stack reads on those seats would otherwise falsely
        keep the walk alive past a legitimate CHECK. `skip` extends that
        to seats whose evidence is not "remaining" by construction: the
        actor being judged, seats this pass already explained, and seats
        that cannot act any more (folded / all-in) — the last group
        matters now that an unexplained stack drop can persist in the
        baseline (see `_store_baseline`).
        """
        min_bet_engine = cents_to_engine_chips(int(min_bet_cents), scale)
        for offset in range(self.num_seats):
            seat = (actor_start + offset) % self.num_seats
            if sitting_out is not None and seat < len(sitting_out) and sitting_out[seat]:
                continue
            if skip is not None and seat in skip:
                continue
            obs = new_seats.get(seat)
            if obs is None:
                continue
            last_obs = last_seats.get(seat)
            drop = 0
            if (
                last_obs is not None
                and obs.stack_chips is not None
                and last_obs.stack_chips is not None
            ):
                drop = int(last_obs.stack_chips) - int(obs.stack_chips)
            # Signal A: commit changed by >= 1bb (positive ⇒ corroborated).
            if obs.committed_chips is not None:
                nc = cents_to_engine_chips(int(obs.committed_chips), scale)
                delta = nc - base_commit[seat]
                if delta != 0 and abs(delta) >= max(1, min_bet_engine):
                    if delta < 0:
                        return True
                    drop_chips = (
                        cents_to_engine_chips(drop, scale) if drop > 0 else 0
                    )
                    if (
                        bool(getattr(obs, "bet_banner", False))
                        or drop_chips + _CALL_ROUNDING_TOLERANCE_CHIPS >= delta
                    ):
                        return True
            # Signal B: stack dropped above the noise floor vs last_fs.
            # Require a strictly positive drop so a no-op tick (identical
            # stacks) doesn't falsely keep the walk alive when the noise
            # floor is zero.
            if drop > 0 and drop >= max(1, int(min_bet_cents)):
                return True
        return False
