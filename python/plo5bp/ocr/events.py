"""Reconstruct game events from a stream of OCR FrameStates.

The OCR capture loop polls the target window every ~500ms and feeds
each snapshot to `EventReconstructor.step(fs, engine_view)`. The
reconstructor diffs the new FrameState against its last accepted one,
cross-references an engine snapshot to know whose turn it is, and
emits tagged OcrEvent instances describing what changed.

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

- unchanged, no facing bet  → CHECK
- unchanged, facing a bet   → FOLD (they couldn't have checked)
- equal to `to_call`        → CALL
- exceeds `to_call`         → RAISE to that new total
- stack went to 0           → RAISE to current commit (short shove —
                              engine routes via short-shove path)

When residual deltas can't be explained, we emit an OcrWarning rather
than raising, and leave the last-accepted FrameState alone so the
next poll gets another shot.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace

from plo5bp.ocr.types import Card, FrameState

logger = logging.getLogger("plo5bp.ocr")


def _timer_debug_enabled() -> bool:
    return bool(os.environ.get("PLO5BP_OCR_DEBUG_TIMER"))


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
    # When True (PokerNow, where the DOM exposes an exact `fold` class), the
    # walk NEVER infers a fold from "faced a bet but added no chips" — a
    # closing call sweeps its chips to the pot and would otherwise be misread
    # as a fold. Real folds still come from `obs.folded` (Fix J / reveal fold
    # reconcile). The OCR path leaves this False (its fold signal is noisy, so
    # the inference is load-bearing there).
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


def _hero_hole_just_hid(last: FrameState, fs: FrameState) -> bool:
    """True when hero's hole cards transitioned visible→hidden.

    ClubGG's anti-collusion rule shows hero's hole cards only during
    hero's turn postflop (see extract.py's hero in-hand rule). The
    visible→hidden transition is a positive signal that hero just
    acted — used by the walk ladder when no direct chip evidence
    (commit/banner/stack drop) is available.

    Conservative all-or-nothing semantics: prior frame must have all
    5 face-up AND current frame must have all 5 hidden. Partial /
    mid-animation reads don't trigger.
    """
    last_visible = _hole_as_indices(last.hero_hole) is not None
    fs_hidden = all(c is None for c in fs.hero_hole)
    return last_visible and fs_hidden


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

    def reset(self) -> None:
        self.last_fs = None
        self.hero_hole_emitted = False
        self._observed_active_actor = None

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
        self._observed_active_actor = None

    def step(self, fs: FrameState, engine_view: EngineView) -> list[OcrEvent]:
        events: list[OcrEvent] = []

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

        # Seat actions: walk the expected-actor queue from the engine's
        # current_actor forward and explain each seat's commit/stack
        # delta.
        action_events, warnings = self._infer_seat_actions(
            last, fs, engine_view, prev_active, now_active
        )
        events.extend(action_events)
        events.extend(warnings)

        if now_active is not None:
            self._observed_active_actor = now_active

        # Store the new baseline, but carry forward the PREVIOUS tick's
        # `stack_chips` for any seat whose current read is None. ClubGG's
        # blue bet-banner covers the stack label on the action tick, so a
        # bettor's `stack_chips` OCR-reads None exactly when we most need
        # it. If we baked that None in, next tick's stack-drop would be
        # incomputable (prev_stack is None) and the bet — whose commit
        # oval reads late — would be permanently lost (its corroboration
        # needs a same-tick banner or stack drop, both gone by then).
        # STACK ONLY: `committed_chips`'s None/0 semantics are load-bearing
        # in the ladder (a None/0 commit legitimately means "no action"),
        # so it is NOT carried forward. Board/street/button/actor fields
        # all come from `fs`, preserving the StreetReveal dedupe (which
        # compares board counts between last_fs and fs).
        prev_seats = {s.seat: s for s in last.seats}
        merged_seats = tuple(
            s
            if s.stack_chips is not None or s.seat not in prev_seats
            or prev_seats[s.seat].stack_chips is None
            else replace(s, stack_chips=prev_seats[s.seat].stack_chips)
            for s in fs.seats
        )
        self.last_fs = replace(fs, seats=merged_seats)
        return events

    # -- helpers ------------------------------------------------------

    def _infer_seat_actions(
        self,
        last: FrameState,
        fs: FrameState,
        engine_view: EngineView,
        prev_active: int | None,
        now_active: int | None,
    ) -> tuple[list[SeatAction], list[OcrWarning]]:
        events: list[SeatAction] = []
        warnings: list[OcrWarning] = []

        if engine_view.current_actor is None:
            return events, warnings

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
            if cents is None:
                return None
            return int(round(int(cents) * scale))

        base_commit = list(engine_view.committed_this_street)
        folded = list(engine_view.folded)
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
                    and not engine_view.all_in[i]
                    and not sitting_out[i]
                ]
                if still_in and all(base_commit[i] >= facing_bet for i in still_in):
                    break

            if folded[actor] or engine_view.all_in[actor] or sitting_out[actor]:
                actor = (actor + 1) % self.num_seats
                continue

            obs = new_seats.get(actor)
            if obs is None:
                break

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
                events.append(SeatAction(seat=actor, gate="fold", chips=0))
                if _timer_debug_enabled():
                    logger.warning(
                        "ocr.walk: emitted seat=%d gate=fold chips=0 "
                        "branch=fix_j_obs_folded",
                        actor,
                    )
                folded[actor] = True
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
            prev_obs = last_seats.get(actor)
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
            min_bet_engine = int(round(int(engine_view.min_bet_cents) * scale))
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
                    stack_drop_chips = (
                        int(round(int(stack_drop_cents) * scale))
                        if stack_drop_cents is not None
                        else 0
                    )
                    if not (banner or stack_drop_chips >= primary_delta):
                        primary_read = None

            # Zero-on-zero guard (Fix P). When the seat faces a bet
            # (to_call_here > 0), primary_read == 0 with
            # base_commit == 0 is information-free: an empty chip
            # oval can legitimately resolve to None OR 0 depending
            # on which preprocessor path runs, and both mean "seat
            # has not contributed chips this street". Without this
            # guard, branch 4 below treats the 0 as an affirmative
            # "no change" and the downstream `delta==0 and
            # to_call>0 → FOLD` branch emits a phantom FOLD on a
            # seat whose real state is "in hand, yet to act" (cards
            # still visible, still thinking). Nulling the read lets
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

            new_commit: int | None
            if primary_read is not None and primary_read != base_commit[actor]:
                new_commit = primary_read
            elif stack_drop_cents is not None and (
                banner
                or stack_drop_cents >= int(engine_view.min_bet_cents)
            ):
                new_commit = base_commit[actor] + int(
                    round(stack_drop_cents * scale)
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
                # (timer-bar transition, hero-hole-hid, downstream
                # activity) lives in the branches below.
                new_commit = primary_read
            elif (
                to_call_here == 0
                and prev_active == actor
                and now_active is not None
                and now_active != actor
            ):
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
            elif (
                actor == 0
                and to_call_here == 0
                and _hero_hole_just_hid(last, fs)
            ):
                # Hero-specific positive signal: hole cards
                # transitioned visible→hidden, which means hero
                # just acted. Combined with no facing bet, that's a
                # CHECK — even when no downstream seat has acted yet
                # (`_any_remaining_delta` would still be False).
                new_commit = base_commit[actor]
            elif to_call_here == 0 and self._any_remaining_delta(
                new_seats,
                last_seats,
                base_commit,
                scale,
                actor_start=(actor + 1) % self.num_seats,
                min_bet_cents=int(engine_view.min_bet_cents),
                sitting_out=sitting_out,
            ):
                # No direct evidence for this seat but someone
                # downstream clearly acted — presume CHECK and
                # walk on.
                new_commit = base_commit[actor]
            else:
                if _timer_debug_enabled():
                    logger.warning(
                        "ocr.walk: stalled actor=%d to_call=%d "
                        "primary_read=%s banner=%s stack_drop_cents=%s "
                        "prev_active=%s now_active=%s",
                        actor, to_call_here, primary_read, banner,
                        stack_drop_cents, prev_active, now_active,
                    )
                break

            delta = new_commit - base_commit[actor]
            to_call = max(0, facing_bet - base_commit[actor])

            if delta == 0:
                if to_call > 0:
                    if facing_bet_bumped:
                        # This seat is *downstream* of a raise we just
                        # emitted in this same pass — they're waiting,
                        # not folded. Let the next poll re-observe.
                        break
                    if engine_view.exact_folds and not obs.folded:
                        # Exact-fold source (PokerNow): a seat facing a bet
                        # with no committed change that is NOT flagged folded
                        # has NOT folded — this is a closing call whose chips
                        # were swept to the pot (committed reads 0). Inferring
                        # a fold here is the "last caller registers as a fold,
                        # hand ends" bug. Break; the call is recovered via the
                        # stack-delta next frame or the StreetReveal
                        # call-reconcile, and real folds arrive via obs.folded.
                        break
                    events.append(
                        SeatAction(seat=actor, gate="fold", chips=0)
                    )
                    if _timer_debug_enabled():
                        logger.warning(
                            "ocr.walk: emitted seat=%d gate=fold chips=0 "
                            "branch=delta0_tocall_pos",
                            actor,
                        )
                    folded[actor] = True
                    actor = (actor + 1) % self.num_seats
                    continue
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
                actor = (actor + 1) % self.num_seats
                if not self._any_remaining_delta(
                    new_seats,
                    last_seats,
                    base_commit,
                    scale,
                    actor_start=actor,
                    min_bet_cents=int(engine_view.min_bet_cents),
                    sitting_out=sitting_out,
                ):
                    break
                continue

            if delta > 0:
                # Stack went to 0 → short shove. Emit as gate="raise"
                # with the raise-by delta; the engine's
                # apply_raise_chips already routes short shoves through
                # its own short-shove path regardless of sizing relative
                # to to_call.
                new_stack = _to_engine(obs.stack_chips)
                if new_stack is not None and new_stack == 0:
                    events.append(
                        SeatAction(seat=actor, gate="raise", chips=int(delta))
                    )
                    if _timer_debug_enabled():
                        logger.warning(
                            "ocr.walk: emitted seat=%d gate=raise "
                            "chips=%d branch=stack_zero",
                            actor, int(delta),
                        )
                    base_commit[actor] = new_commit
                    if new_commit > facing_bet:
                        facing_bet = new_commit
                        facing_bet_bumped = True
                    actor = (actor + 1) % self.num_seats
                    continue

                # Exact call.
                if delta == to_call and to_call > 0:
                    events.append(
                        SeatAction(seat=actor, gate="check_call", chips=0)
                    )
                    if _timer_debug_enabled():
                        logger.warning(
                            "ocr.walk: emitted seat=%d gate=check_call "
                            "chips=0 branch=exact_call",
                            actor,
                        )
                    base_commit[actor] = new_commit
                    actor = (actor + 1) % self.num_seats
                    continue

                # Sub-facing-bet guard (Fix Q). delta > 0 but
                # new_commit < facing_bet means the ladder wants to
                # emit a "raise" whose target commit is below the
                # current bet. That's not a legal poker action —
                # apply_raise_chips rejects it and _rebuild_env logs
                # "skipping illegal action". Worse, falling through
                # still advances base_commit[actor] at line 538 while
                # the facing_bet_bumped gate below stays False
                # (new_commit is not > facing_bet), leaving the
                # delta==0+to_call>0 phantom-FOLD branch ungated for
                # every trailing seat the walk visits. The derivation
                # is noise (sub-1bb stack drift, or a chip-oval
                # misread that survived Fix N corroboration by
                # coincidence). Break and let the next poll re-observe.
                if new_commit < facing_bet:
                    break

                # Otherwise a raise. `chips` is the raise-by delta in
                # engine-chips that `apply_raise_chips` consumes.
                events.append(
                    SeatAction(seat=actor, gate="raise", chips=int(delta))
                )
                if _timer_debug_enabled():
                    logger.warning(
                        "ocr.walk: emitted seat=%d gate=raise chips=%d "
                        "branch=raise (new_commit=%d facing_bet=%d "
                        "primary_read=%s banner=%s stack_drop_cents=%s)",
                        actor, int(delta), new_commit, facing_bet,
                        primary_read, banner, stack_drop_cents,
                    )
                base_commit[actor] = new_commit
                if new_commit > facing_bet:
                    facing_bet = new_commit
                    facing_bet_bumped = True
                actor = (actor + 1) % self.num_seats
                continue

            # Negative delta — street probably closed and commits
            # reset. Stop inferring; let the next poll recover via a
            # StreetReveal.
            break

        if steps_left == 0:
            warnings.append(OcrWarning("action inference hit loop guard"))

        return events, warnings

    def _any_remaining_delta(
        self,
        new_seats: dict,
        last_seats: dict,
        base_commit: list[int],
        scale: float,
        actor_start: int,
        min_bet_cents: int,
        sitting_out: list[bool] | None = None,
    ) -> bool:
        """True if any downstream seat shows evidence of action.

        Two independent signals — the walk continues past a CHECK if
        EITHER fires, so a single-OCR-miss on the in-play chip oval
        can't prematurely break inference:
          A. `committed_chips` diverges from `base_commit` by >= 1bb.
          B. Stack dropped >= min_bet_cents vs last frame.

        Signal A requires a >= 1bb delta for the same reason the main
        walk's `primary_read` guard does: Tesseract regularly misreads
        the chip oval as a small integer (`"4"` = 400 cents → 2000
        engine-chips) when the badge is mid-animation or partially
        occluded. Without the threshold the phantom signal never
        clears, the walk coast-pasts through every seat emitting
        CHECKs, and the hand ends prematurely at showdown.

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
        keep the walk alive past a legitimate CHECK.
        """
        min_bet_engine = int(round(int(min_bet_cents) * scale))
        for offset in range(self.num_seats):
            seat = (actor_start + offset) % self.num_seats
            if sitting_out is not None and seat < len(sitting_out) and sitting_out[seat]:
                continue
            obs = new_seats.get(seat)
            if obs is None:
                continue
            # Signal A: commit changed by >= 1bb.
            if obs.committed_chips is not None:
                nc = int(round(int(obs.committed_chips) * scale))
                delta = nc - base_commit[seat]
                if delta != 0 and abs(delta) >= max(1, min_bet_engine):
                    return True
            # Signal B: stack dropped above the noise floor vs last_fs.
            # Require a strictly positive drop so a no-op tick (identical
            # stacks) doesn't falsely keep the walk alive when the noise
            # floor is zero.
            last_obs = last_seats.get(seat)
            if (
                last_obs is not None
                and obs.stack_chips is not None
                and last_obs.stack_chips is not None
            ):
                drop = last_obs.stack_chips - obs.stack_chips
                if drop > 0 and drop >= max(1, int(min_bet_cents)):
                    return True
        return False
