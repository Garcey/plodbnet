"""Live capture -> study session plumbing shared by BOTH live sources.

The ClubGG pixel-OCR runner (`clubgg.py`) and the PokerNow DOM ingest
(`pokernow.py`) turn table frames into mutations of the study `Session`
through the same pieces, all here:

* card-slot writes (`_ocr_apply_card_slot`: debounced for pixels, immediate
  for exact DOM reads, never over a slot the user locked via /cards);
* unit conversion (OCR cents -> engine chips, one definition shared with
  `plo5bp.ocr.events`);
* the debounced hand-start machine (`_mirror_observable_state` ->
  `_begin_new_hand`), the street-reveal fold/check reconcilers and the
  `EngineView` the reconstructor walks (`_engine_view_from_session`);
* which source is driving (`_note_live_source`) and whose reconstructor
  `_begin_new_hand` rebaselines (`_set_active_reconstructor`).

CLAUDE.md "OCR integration architecture" / "PokerNow live source" list the
invariants this code keeps. Local build only: the public build never imports
the `plo5bp.ui.live` package.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
from typing import Any

from fastapi import HTTPException

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.config import VARIANT_PLO5
from plo5bp.ui.live.state import current_live_state, live_state
from plo5bp.ui.server import (
    _GATE_NAME_TO_IDX,
    _build_env,
    _card_spec_attrs,
    _commit_env_build,
    _engine_variant,
    _new_session_defaults,
    _rebuild_env,
    _session_env_spec,
    session,
)

logger = logging.getLogger("plo5bp.ui.live")


# --- Card-slot writes -------------------------------------------------------

# Consecutive identical auto-OCR reads required before a card slot commits and
# locks. ~600ms at 200ms poll / 300ms at 100ms. Raise if misreads still slip
# through; lower if the commit feels sluggish.
_CARD_STABLE_TICKS = 3


def _card_held_by_other_slot(attr: str, idx: int, card: int) -> bool:
    """True when ``card`` already sits in a spec slot other than (attr, idx)."""
    for other_attr, _ in _card_spec_attrs():
        for j, c in enumerate(getattr(session, other_attr)):
            if c is not None and int(c) == int(card) and (other_attr, j) != (attr, idx):
                return True
    return False


def _ocr_apply_card_slot(
    attr: str,
    idx: int,
    ocr_card_idx: int | None,
    debounce: bool = False,
    reject_duplicates: bool | None = None,
) -> None:
    """Write an OCR-detected card into the spec slot if it isn't locked.

    With ``debounce=True`` (the automatic mirror path) the read must repeat
    identically for ``_CARD_STABLE_TICKS`` consecutive ticks before it commits
    and locks — so a transient mid-reveal misread can't latch. ``debounce=False``
    (manual rescan, explicit edits) commits immediately, as before.

    ``reject_duplicates`` refuses to commit a card that another slot already
    holds: a stable misread that duplicated a card used to lock, after which
    ``_pad_all`` 400'd every tick and the session was wedged until a manual
    edit. The slot stays empty + unlocked so a later correct read (or the
    user) can still fill it. Default (None) = on for the debounced automatic
    mirror, off for immediate commits; the PokerNow mirror opts in, manual
    rescan stays off on purpose — its duplicate path is the pinned
    400 + rollback. (review 2026-09-20 H7)
    """
    if reject_duplicates is None:
        reject_duplicates = debounce
    if session._card_slot_locked[attr][idx]:
        return
    if not debounce:
        if ocr_card_idx is None:
            return
        if reject_duplicates and _card_held_by_other_slot(attr, idx, ocr_card_idx):
            return
        getattr(session, attr)[idx] = ocr_card_idx
        session._card_slot_locked[attr][idx] = True
        return

    pending = live_state.card_slot_pending[attr]
    if ocr_card_idx is None:
        # No confident read this tick — break the run.
        pending[idx] = None
        return
    prev = pending[idx]
    count = prev[1] + 1 if (prev is not None and prev[0] == ocr_card_idx) else 1
    pending[idx] = (ocr_card_idx, count)
    if count >= _CARD_STABLE_TICKS:
        pending[idx] = None
        if reject_duplicates and _card_held_by_other_slot(attr, idx, ocr_card_idx):
            logger.info(
                "ocr: %s[%d] read card %d, already held by another slot — "
                "not committing", attr, idx, ocr_card_idx,
            )
            return
        getattr(session, attr)[idx] = ocr_card_idx
        session._card_slot_locked[attr][idx] = True
        _note_suit_read(ocr_card_idx)


#: Suits the pixel classifier tells by colour in ClubGG's 4-colour deck.
_COLOURED_SUITS = (0, 1)  # clubs (green), diamonds (blue)
#: No club or diamond among this many committed cards over this many hands
#: means the 2-colour deck (odds ~1 in 30,000 for a 4-colour one).
_DECK_CHECK_CARDS = 15
_DECK_CHECK_HANDS = 3


def _note_suit_read(card: int) -> None:
    """Track committed pixel reads for the 4-colour deck check (TOOL-042)."""
    st = current_live_state()
    st.deck_cards += 1
    if int(card) % 4 in _COLOURED_SUITS:
        st.deck_colored_seen = True
        st.warnings.pop("deck", None)
    _check_deck_colours()


def _check_deck_colours() -> None:
    """ClubGG's 2-colour deck paints diamonds red and clubs black, so they
    silently read as hearts / spades. Say so once the evidence is clear."""
    st = current_live_state()
    if (
        not st.deck_colored_seen
        and st.deck_hands >= _DECK_CHECK_HANDS
        and st.deck_cards >= _DECK_CHECK_CARDS
    ):
        st.warnings["deck"] = (
            f"no club or diamond read in {st.deck_hands} hands — ClubGG is "
            "probably on the 2-colour deck; switch it to the 4-colour deck "
            "(the OCR tells suits by colour)"
        )


# --- The live lock -------------------------------------------------------------
# ONE lock serializes every mutation of the study session in the local build:
# each OCR tick, each PokerNow payload, and the study + live HTTP routes
# (`routes._LiveLockMiddleware`). A tick snapshots the action log, rebuilds and
# commits; a manual /action (or /undo, /cards, /seats, /config ...) that landed
# on another thread in between used to be ERASED when the tick committed its
# older snapshot (TOOL-004). asyncio.Lock is FIFO, so a click waiting behind a
# tick runs before the next tick starts.
_LIVE_LOCK: tuple[Any, asyncio.Lock] | None = None


def live_lock() -> asyncio.Lock:
    """The live lock for the RUNNING event loop.

    asyncio locks bind to the loop that first waits on them; uvicorn runs one
    loop for the process, but a test client may spin a loop per request, so a
    new loop gets a new (free) lock.
    """
    global _LIVE_LOCK
    loop = asyncio.get_running_loop()
    if _LIVE_LOCK is None or _LIVE_LOCK[0] is not loop:
        _LIVE_LOCK = (loop, asyncio.Lock())
    return _LIVE_LOCK[1]


# --- Active live-source reconstructor ---------------------------------------
# Both live sources (ClubGG WGC OCR and PokerNow DOM ingest) feed the same
# session through an EventReconstructor. `_begin_new_hand` rebaselines
# whichever one is currently driving; each runner registers its reconstructor
# here when it starts so the shared hand-start path stays source-agnostic.
_LIVE_RECONSTRUCTOR: Any = None


def _active_reconstructor() -> Any:
    """The reconstructor of the live source driving the session (None when
    neither runner is up). Each runner registers its own when it starts."""
    return _LIVE_RECONSTRUCTOR


def _set_active_reconstructor(recon: Any) -> None:
    global _LIVE_RECONSTRUCTOR
    _LIVE_RECONSTRUCTOR = recon


#: Which live source last drove the session: "ocr", "pokernow" or None.
_LIVE_SOURCE: str | None = None


def _live_source() -> str | None:
    return _LIVE_SOURCE


def _note_live_source(name: str | None) -> None:
    """Record the driving live source; a SWITCH to another source resets the
    hand-start machine so the previous source's mask / anchor / hero-hole
    baseline can't leak into the new one's first hand. (review 2026-09-20 F10)
    """
    global _LIVE_SOURCE
    if name == _LIVE_SOURCE:
        return
    _LIVE_SOURCE = name
    if name is not None:
        _reset_live_tracking()


# --- Live capture is PLO5-only ----------------------------------------------
# (review 2026-09-20 I10) Both live sources read a 5-card-hole, double-board
# bomb-pot table and mirror it into the session's card spec. Under the NLH
# study format the spec is 2-card / single-board and the game config carries
# blinds: one PokerNow POST used to replace that config with a PLO one while
# `session.variant` stayed NLH, bricking the study session (every rebuild
# 400'd, /reset included) until a format flip. Live entry points therefore
# refuse — before touching any state — unless the engine variant is PLO5.

_LIVE_FORMAT_ERROR = (
    "live capture needs the PLO5 double-board format — switch the format "
    "back to PLO5 to use ClubGG OCR / PokerNow"
)

#: Engine seat ceiling (`GameConfig` validates 2..8; PLO5 double-board can't
#: even deal a 9th seat: 9*5 + 10 board cards > 52). PokerNow tables seat 10.
_MAX_ENGINE_SEATS = 8


def _live_capture_allowed() -> bool:
    return _engine_variant() == VARIANT_PLO5


def _require_live_capture_format() -> None:
    if not _live_capture_allowed():
        raise HTTPException(status_code=409, detail=_LIVE_FORMAT_ERROR)


def _env_signature() -> Any:
    """Everything `_build_env` reads from the session (its `_EnvSpec`), with
    the action-log entries copied so a later in-place edit can't alias."""
    spec = _session_env_spec()
    return dataclasses.replace(spec, action_log=[dict(e) for e in spec.action_log])


def _rebuild_env_if_stale() -> None:
    """`_rebuild_env`, unless the committed env was built from exactly the
    session as it stands (TOOL-011).

    Most live ticks change nothing the replay reads (same cards, log, mask,
    button, stacks), yet each one replayed the whole hand into a fresh env
    and re-packed the observation. The env built last time is reused when it
    is still the session's env and every replay input compares equal.
    """
    st = current_live_state()
    if (
        session.env is not None
        and st.built_env is session.env
        and st.built_spec == _env_signature()
    ):
        return
    _rebuild_env()
    st.built_env = session.env
    st.built_spec = _env_signature()


def _reset_live_tracking() -> None:
    """Forget everything the live hand-start machine has accumulated.

    `_new_session_defaults` deliberately keeps the cross-hand debounce state
    (pending snapshot/anchor, `last_hero_hole`, observed stacks/pot) because
    `_begin_new_hand` calls it mid-stream. A user-level reset is different:
    after `/reset`, `/format`, an OCR (re)start or a live-source switch the
    old pending anchor is a frame from a hand that no longer exists,
    and the still-"stable" tick counter let the very next tick fire
    `_begin_new_hand` with that stale anchor — seeding stacks and the
    reconstructor baseline from minutes-old reads. Clearing the counters
    makes the next hand-start debounce from scratch on fresh frames.
    (review 2026-09-20 F10)
    """
    current_live_state().reset_tracking()
    session.sitting_out_seats = frozenset()
    session.hand_in_hand_mask = frozenset()
    session.folded_this_hand = frozenset()
    # The mask decides who the engine deals in, so the env built with it is
    # stale now. Callers rebuild right away or let the next reader do it.
    session.env = None


# --- Units, log entries, street-reveal reconcilers ---------------------------

def _ocr_cents_to_engine_chips(cents: int) -> int:
    """Convert an OCR-cent amount (ClubGG dollar display × 100) to engine chips.

    OCR reads ClubGG stack text in cents (100 = $1). The engine carries
    chip counts where `cfg.bb` chips = 1 big blind = `dollars_per_bb`
    dollars. At the Study default (bb=10000, $2/bb) that works out to 50
    engine-chips per cent ($20/bb: 5). Doing the mixed-unit arithmetic that used to
    live in ``_reset_for_new_hand`` (raw_cents + cfg.ante_chips) is a
    bug; always run cents through this helper first.
    """
    scale = _chips_per_cent()
    if scale is None:
        return int(cents)
    # Same single multiply + round as `ocr.events.cents_to_engine_chips`, on
    # the same `chips_per_cent` the EngineView carries, so a stack seeded here
    # and a bet converted by the reconstructor can never round differently
    # (the old `cents * bb / (100 * dpb)` op order could be 1 chip off at
    # non-integer scales).
    return int(round(int(cents) * scale))


def _chips_per_cent() -> float | None:
    """Engine chips per OCR cent (None when $/bb is unusable)."""
    dpb = float(session.dollars_per_bb)
    if dpb <= 0:
        return None
    return float(session.game_config.bb) / (100.0 * dpb)


def _seat_action_to_log_entry(ev: Any) -> dict[str, int]:
    """Translate a ``SeatAction`` OCR event into an ``action_log`` entry.

    ``SeatAction.chips`` is already a raise-by delta in engine-chips
    (the reconstructor converts OCR cents at its boundary); no unit
    fixup needed here. ``seat`` records who the walk attributed the
    action to (diagnostic only — see the replay loop in `_build_env`).
    """
    return {
        "gate": int(_GATE_NAME_TO_IDX[ev.gate]),
        "chips": int(ev.chips),
        "seat": int(ev.seat),
    }


def _reconcile_missed_folds_on_street_reveal(fs: Any, exact: bool = False) -> bool:
    """Append FOLD entries for in-hand seats the walk missed (Fix K).

    Triggered when a ``StreetReveal`` fires (and, on the OCR path, once more
    on the following tick — see below). The guard is that
    ``StreetReveal`` already requires both board_a and board_b to
    show 4+ cards (see ocr.events), which rules out the bomb-pot
    intro animation flicker.

    Attribution by count, not by seat: ``step_hybrid`` applies
    actions to the engine's own ``current_actor``, so appending N
    bare FOLD entries retires the next N actors. That's fine because
    the walk already handled any non-fold actions ahead of the
    missed folds — if a call had sat between them, the engine
    would have advanced past those seats before the missed folds
    reached the front of the queue.

    (review 2026-09-20 F11) ``exact=False`` (ClubGG OCR): ``folded`` is a
    noisy pixel read, and this used to trust the single reveal frame — one
    missed card-back on exactly that frame retired a live seat for the rest
    of the hand. A seat is now reconciled only when it reads folded on TWO
    consecutive ticks: the first sighting is parked in
    ``pending_reveal_folds`` and confirmed (or dropped) by the caller's
    follow-up call on the next tick. ``exact=True`` (PokerNow: the DOM
    ``fold`` class is authoritative) reconciles immediately — it must,
    because the forced CHECK_CALL fill that runs right after treats every
    unreconciled seat as a caller.

    Returns True when FOLD entries were appended.
    """
    in_hand = session.hand_in_hand_mask
    if not in_hand:
        live_state.pending_reveal_folds = frozenset()
        return False
    fs_by_seat = {s.seat: s for s in fs.seats}
    visually_folded = frozenset(
        seat
        for seat in in_hand
        if fs_by_seat.get(seat) is not None and fs_by_seat[seat].folded
    )
    missing = visually_folded - session.folded_this_hand
    if not exact:
        confirmed = missing & live_state.pending_reveal_folds
        live_state.pending_reveal_folds = frozenset(missing - confirmed)
        missing = confirmed
    if not missing:
        return False
    for _ in missing:
        session.action_log.append({"gate": int(GATE_FOLD), "chips": 0})
    session.folded_this_hand = frozenset(
        session.folded_this_hand | missing
    )
    all_seats = frozenset(range(session.num_seats))
    session.sitting_out_seats = (
        (all_seats - session.hand_in_hand_mask) | session.folded_this_hand
    )
    logger.warning(
        "ocr: StreetReveal fold reconcile — added %d FOLD entries "
        "for seats %s (walk missed these folds)",
        len(missing),
        sorted(missing),
    )
    return True


def _reconcile_missed_checks_on_street_reveal(
    target_street: int, force: bool = False
) -> None:
    """Append CHECK_CALL entries when the walk missed a pure-check round.

    Loop-fills: appends one CHECK_CALL, rebuilds the engine, rechecks
    ``view.street`` — until the engine has advanced to ``target_street``.
    The engine's own street-advance logic decides when each fill is
    enough, so this is correct whether the walk emitted 0, 1, or N
    CHECKs for the current street before the reconciler ran. The old
    ``len(active) * streets_to_fill`` formula assumed the engine was at
    the start of the current street, but step 5 of the walk ladder
    (events.py ``_infer_seat_actions``) routinely emits a CHECK for the
    first actor on a quiet street — leaving the engine partially
    advanced and the formula over-counting by exactly that emission.

    Safety (``force=False``, the OCR default): only fills when
    ``bet_to_call == 0`` and no seat has committed chips on the current
    street; positive evidence means a real bet was missed and we leave
    the engine stuck rather than silently mis-attribute.

    ``force=True`` (PokerNow): fill CHECK_CALL even when facing a bet.
    A PokerNow street reveal is authoritative proof the prior street's
    betting closed (PokerNow won't deal the next card otherwise), and the
    closing call's signal is unrecoverable — PokerNow sweeps the chips to
    the pot and deals the next card in the same instant the closing caller
    acts, so their committed amount is gone and their stack delta already
    landed in a (likely coalesced-away) earlier frame. Folds are
    reconciled first (``_reconcile_missed_folds_on_street_reveal`` reads
    the DOM fold flags), so every remaining in-hand seat that hasn't
    matched the bet must have CALLED — GATE_CHECK_CALL calls the engine's
    current ``bet_to_call`` for the right amount. A loop-count guard caps
    runaway in case the engine refuses to advance.
    """
    _rebuild_env()
    if session.env is None:
        return
    view = _engine_view_from_session()
    if os.environ.get("PLO5BP_OCR_DEBUG_TIMER"):
        logger.warning(
            "ocr.reconcile.checks: target=%d force=%s view.street=%d "
            "view.bet_to_call=%d committed=%s",
            target_street, force, int(view.street), int(view.bet_to_call),
            tuple(int(c) for c in view.committed_this_street),
        )
    if view.street >= target_street:
        return
    if not force and (
        view.bet_to_call > 0 or any(int(c) > 0 for c in view.committed_this_street)
    ):
        logger.warning(
            "ocr: StreetReveal check reconcile — skipping; "
            "bet_to_call=%d committed=%s (real bets must have been missed)",
            int(view.bet_to_call),
            tuple(int(c) for c in view.committed_this_street),
        )
        return
    if not session.hand_in_hand_mask:
        return
    if not (session.hand_in_hand_mask - session.folded_this_hand):
        return

    cfg = session.game_config
    start_street = int(view.street)
    safety_limit = cfg.num_seats * (target_street - start_street) + cfg.num_seats
    appended = 0
    while view.street < target_street and appended < safety_limit:
        session.action_log.append({"gate": int(GATE_CHECK_CALL), "chips": 0})
        appended += 1
        _rebuild_env()
        view = _engine_view_from_session()

    if view.street < target_street:
        logger.warning(
            "ocr: StreetReveal check reconcile — safety limit hit at "
            "appended=%d (start_street=%d target=%d view.street=%d)",
            appended, start_street, target_street, int(view.street),
        )
    else:
        logger.warning(
            "ocr: StreetReveal check reconcile — appended %d CHECK_CALL "
            "entries (street %d → %d)",
            appended, start_street, int(view.street),
        )


# --- Sync health: refused walk actions, pot drift ---------------------------------

#: A walk action the engine refuses is retried — the reconstructor is restored to
#: its pre-step baseline, so the next frame can explain the same chips with a
#: corrected read — this many ticks in a row before it is given up on and left
#: to the user (TOOL-025).
_REJECTED_ACTION_RETRIES = 3
#: Consecutive ticks the on-screen pot must disagree with the engine's before
#: the "out of sync" warning shows (pixel reads lag and misread; exact DOM
#: reads need only 2 payloads).
_POT_DRIFT_TICKS = 3
_POT_DRIFT_TICKS_EXACT = 2


def live_warnings() -> list[str]:
    """Current sync warnings for the status endpoints (empty = in sync)."""
    return list(current_live_state().warnings.values())


def _bb(chips: int) -> str:
    return f"{int(chips) / float(session.game_config.bb):g} bb"


def _describe_seat_action(ev: Any, to_call: int) -> str:
    if ev.gate == "fold":
        return "fold"
    if ev.gate == "check_call":
        return "call" if to_call > 0 else "check"
    return f"raise by {_bb(ev.chips)}"


def _refusal_reason(prefix: list[dict[str, Any]], entry: dict[str, Any]) -> tuple[str, int]:
    """Why the engine refuses ``entry`` after ``prefix`` (and the amount to
    call at that point), for the warning text."""
    try:
        env = _build_env(_session_env_spec(action_log=prefix)).env
    except HTTPException as e:
        return str(e.detail), 0
    actor = env.current_actor()
    if actor is None:
        return "the betting round is already closed", 0
    raw = env._rs.observation_dict(skip_outcome_mc=True)
    to_call = max(0, int(raw["bet_to_call"]) - int(raw["street_commit"][int(actor)]))
    gate, chips = int(entry["gate"]), int(entry["chips"])
    if gate == int(GATE_FOLD) and to_call == 0:
        return "there is no bet to fold to", to_call
    if gate == int(GATE_RAISE):
        lo, hi = int(raw.get("min_raise") or 0), int(raw.get("max_raise") or 0)
        if hi <= 0:
            return "no raise is possible here", to_call
        if chips < lo:
            return f"the minimum raise is {_bb(lo)}", to_call
    return "the engine refused it", to_call


def _record_seat_actions(
    events: list[Any], recon: Any = None, recon_before: dict | None = None
) -> tuple[int, bool]:
    """Append a walk's `SeatAction`s to the action log — validated FIRST.

    The whole batch is replayed on a candidate log (validate-then-commit, like
    the study /action route) and committed only when the engine accepts every
    action. A refused action used to be dropped silently at the next rebuild
    while the reconstructor had already counted its chips as explained: the
    action was lost and log and table drifted apart (TOOL-025). Now it shows
    as a warning, and the step is undone — nothing appended, the
    reconstructor restored to ``recon_before`` — so the next frame re-derives
    it (a misread amount usually reads right a tick later). After
    `_REJECTED_ACTION_RETRIES` identical refusals the actions before it are
    kept and the warning asks the user to enter that seat's action.

    Returns ``(actions applied, retry)``; on retry the step is void and the
    caller skips the rest of it (street reconcilers included).
    """
    from plo5bp.ocr.events import SeatAction

    actions = [ev for ev in events if isinstance(ev, SeatAction)]
    if not actions:
        return 0, False
    st = current_live_state()
    base = list(session.action_log)
    entries = [_seat_action_to_log_entry(ev) for ev in actions]
    try:
        build = _build_env(_session_env_spec(action_log=base + entries))
        kept = {id(e) for e in build.kept_log}
    except HTTPException:
        build, kept = None, set()
    bad = next((i for i, e in enumerate(entries) if id(e) not in kept), None)
    if bad is None:
        session.action_log.extend(entries)
        _note_folds(actions)
        _commit_env_build(build)
        st.built_env, st.built_spec = session.env, _env_signature()
        st.warnings.pop("rejected_action", None)
        st.rejected_key, st.rejected_streak = None, 0
        return len(entries), False

    ev = actions[bad]
    key = (int(ev.seat), str(ev.gate), int(ev.chips))
    st.rejected_streak = st.rejected_streak + 1 if st.rejected_key == key else 1
    st.rejected_key = key
    retry = (
        st.rejected_streak < _REJECTED_ACTION_RETRIES
        and recon is not None
        and recon_before is not None
    )
    reason, to_call = _refusal_reason(base + entries[:bad], entries[bad])
    what = _describe_seat_action(ev, to_call)
    tail = (
        "retrying on the next frames" if retry
        else f"enter seat {ev.seat}'s action by hand"
    )
    msg = f"live read seat {ev.seat} {what}, which the engine refused ({reason}) — {tail}"
    if st.warnings.get("rejected_action") != msg:
        logger.warning("live: %s", msg)
    st.warnings["rejected_action"] = msg
    if retry:
        recon.restore(recon_before)
        return 0, True
    session.action_log.extend(entries[:bad])
    _note_folds(actions[:bad])
    return bad, False


def _note_folds(actions: list[Any]) -> None:
    """Sticky participant mask: a recorded FOLD retires the seat for the rest
    of the hand (banner / occlusion misses can't un-fold it); the engine
    re-confirms it after every rebuild (`_sync_folds_with_engine`)."""
    folds = {int(ev.seat) for ev in actions if ev.gate == "fold"}
    if not folds:
        return
    session.folded_this_hand = frozenset(session.folded_this_hand | folds)
    if session.hand_in_hand_mask:
        all_seats = frozenset(range(session.num_seats))
        session.sitting_out_seats = (
            (all_seats - session.hand_in_hand_mask) | session.folded_this_hand
        )


def _clear_stale_refusal() -> None:
    """Drop the refused-action warning once the engine has moved past that
    seat (the user entered it, or a later read explained it)."""
    st = current_live_state()
    if st.rejected_key is None or session.env is None:
        return
    if session.env.current_actor() != st.rejected_key[0]:
        st.warnings.pop("rejected_action", None)
        st.rejected_key, st.rejected_streak = None, 0


def _check_pot_sync(fs: Any, *, exact: bool = False) -> None:
    """Compare the pot on screen with the engine's (TOOL-026).

    A phantom raise, a missed call, a wrong ante / $-per-bb setting or a table
    that isn't a bomb pot all leave the engine's pot different from the
    table's. The screen may show the pot with or without this street's bets,
    so either matches (1% / 1 cent tolerance). A disagreement lasting
    `_POT_DRIFT_TICKS` ticks raises a visible warning (and one log line with
    both numbers); it clears as soon as they agree again.
    """
    st = current_live_state()
    observed = fs.pot_total_chips
    env = session.env
    scale = _chips_per_cent()
    if observed is None or env is None or not scale or not session.hand_in_hand_mask:
        st.pot_mismatch_ticks = 0
        return
    raw = env._rs.observation_dict(skip_outcome_mc=True)
    pot = int(raw["pot"])
    settled = pot - sum(int(x) for x in raw["street_commit"])
    candidates = (pot / scale, settled / scale)
    tol = max(1.0, 0.01 * float(observed))
    if any(abs(float(observed) - c) <= tol for c in candidates):
        st.pot_mismatch_ticks = 0
        st.warnings.pop("pot", None)
        return
    st.pot_mismatch_ticks += 1
    need = _POT_DRIFT_TICKS_EXACT if exact else _POT_DRIFT_TICKS
    if st.pot_mismatch_ticks < need:
        return
    shown = f"${int(observed) / 100:.2f}"
    tracked = f"${pot / scale / 100:.2f}"
    msg = (
        f"pot on screen {shown} but {tracked} tracked — the hand may be out of "
        "sync (a missed or phantom action, the ante / $ per bb settings, or "
        "not a bomb pot)"
    )
    if st.warnings.get("pot") is None:
        logger.warning(
            "live: pot drift — screen %s cents, engine pot %d chips (settled %d), "
            "%.4f chips/cent", observed, pot, settled, scale,
        )
    st.warnings["pot"] = msg


# --- Hand-start machine -----------------------------------------------------

def _card_idx(c: Any) -> int | None:
    """FrameState Card → engine card index (rank * 4 + suit)."""
    if c is None:
        return None
    return int(c.rank) * 4 + int(c.suit)


_STABILITY_TICKS_REQUIRED = 2

# Mid-hand lock thresholds. After `_LOCK_AFTER_TICKS` ticks since
# the last `_begin_new_hand`, the snapshot debouncer (and the
# hero-hole rotation gate) require `_STABILITY_TICKS_REQUIRED_LOCKED`
# stable ticks before triggering another hand-start. Real new
# hands persist for many seconds and trivially clear this bar; a
# 1-2-tick chip-settle / banner OCR flicker doesn't.
_LOCK_AFTER_TICKS = 3
_STABILITY_TICKS_REQUIRED_LOCKED = 6
# Consecutive identical button reads required to trust a mid-hand button MOVE
# as a new-hand trigger. Decoupled from the 6-tick snapshot above: a button
# move persists all hand, so 3 reads reject a 1-2 frame glitch while keeping
# new-hand latency low and immune to sitting_out churn.
_BUTTON_STABLE_TICKS_LOCKED = 3

# Minimum plausible stack_chips read (cents) for an in-hand seat's
# anchor OCR. Live ClubGG frames captured during the chip-settle
# animation occasionally OCR the stack label as 0 or a tiny number
# because the blue bet-banner or chip oval covers the digits.
# `_begin_new_hand` refuses to seed from such reads — it keeps the
# existing seeded value instead — and `_mirror_observable_state`
# refuses to commit an anchor where any in-hand seat looks glitched,
# extending the debounce by one tick until a clean frame lands.
_MIN_PLAUSIBLE_STACK_CENTS = 100  # $1 — every in-hand seat has at least this


def _anchor_fs_stacks_plausible(fs: Any) -> bool:
    """Return True iff every non-folded seat has a plausible stack read.

    A glitched anchor is one where a bet-banner or chip-settle
    animation is covering a seat's stack label at capture time, so
    the OCR text read comes back as None or a near-zero value. Using
    such an anchor to seed ``cfg.starting_stacks`` causes the engine
    to wipe the seat's stack to zero after ante posting, which then
    renders in the UI as a phantom pre-bet "all-in".
    """
    for s in fs.seats:
        if s.folded:
            continue
        if s.stack_chips is None or int(s.stack_chips) < _MIN_PLAUSIBLE_STACK_CENTS:
            return False
    return True


#: A hand-start anchor must show the antes in the pot (TOOL-013). Seeding
#: `starting = behind + ante` from a frame taken BEFORE ClubGG collected the
#: antes left every stack one ante high (the engine then misjudged all-in calls
#: and max raises), and the reconstructor's baseline still held the antes, so
#: their collection could read as a bet. The debouncer waits for — and upgrades
#: the anchor to — a frame whose pot holds them, at most this many ticks: an
#: unreadable or misread pot must never block hand-starts.
_ANTE_WAIT_TICKS = 5


def _anchor_shows_antes(fs: Any) -> bool:
    """False only when the frame's pot read proves the antes aren't in yet
    (pot below 90% of in-hand seats x ante); an unreadable pot proves nothing."""
    pot = fs.pot_total_chips
    scale = _chips_per_cent()
    ante = int(session.game_config.ante)
    n = sum(1 for s in fs.seats if not s.folded)
    if pot is None or not scale or ante <= 0 or n == 0:
        return True
    return int(pot) >= 0.9 * n * ante / scale


def _anchor_ready(fs: Any) -> bool:
    """A frame fit to seed a hand: plausible stacks and the antes collected."""
    return _anchor_fs_stacks_plausible(fs) and _anchor_shows_antes(fs)


def _mirror_observable_state(fs: Any) -> None:
    """Sync every directly-observed field from a FrameState into `session`.

    Cards, button, participant mask, and observed stack/pot numbers are
    all overwritten on every tick. The debounced button + participant
    commit is gated by a 2-tick stability check so one glitched frame
    can't trigger a false hand-start. When the debounced state advances
    (new button, new participant mask, or a hero-hole reshuffle), we
    call :func:`_begin_new_hand` to re-seed `cfg.starting_stacks` from
    the current OCR numbers and rebaseline the reconstructor.
    """
    # PLO5-only (review 2026-09-20 I10): a 5-card / two-board frame does not
    # fit any other format's card spec. Entry points refuse earlier; this
    # keeps a stray call a no-op instead of an IndexError.
    if not _live_capture_allowed():
        return
    for i, c in enumerate(fs.hero_hole):
        _ocr_apply_card_slot("hero_hole", i, _card_idx(c), debounce=True)
    for i, c in enumerate(fs.board_a[:3]):
        _ocr_apply_card_slot("flop_a", i, _card_idx(c), debounce=True)
    for i, c in enumerate(fs.board_b[:3]):
        _ocr_apply_card_slot("flop_b", i, _card_idx(c), debounce=True)
    _ocr_apply_card_slot("turn_cards", 0, _card_idx(fs.board_a[3]), debounce=True)
    _ocr_apply_card_slot("turn_cards", 1, _card_idx(fs.board_b[3]), debounce=True)
    _ocr_apply_card_slot("river_cards", 0, _card_idx(fs.board_a[4]), debounce=True)
    _ocr_apply_card_slot("river_cards", 1, _card_idx(fs.board_b[4]), debounce=True)

    live_state.observed_stacks = tuple(s.stack_chips for s in fs.seats)
    live_state.observed_pot = fs.pot_total_chips

    # Tick the mid-hand lock counter. Only counts while a hand is
    # active (mask non-empty); during bootstrap (empty mask) we
    # stay in the fast 2-tick path so initial hand-start isn't
    # delayed.
    if session.hand_in_hand_mask:
        live_state.ticks_since_hand_start += 1
    is_locked = (
        bool(session.hand_in_hand_mask)
        and live_state.ticks_since_hand_start >= _LOCK_AFTER_TICKS
    )
    threshold = (
        _STABILITY_TICKS_REQUIRED_LOCKED if is_locked
        else _STABILITY_TICKS_REQUIRED
    )

    observed_sitting_out = frozenset(
        i for i, s in enumerate(fs.seats) if i != session.hero_seat and s.folded
    )
    observed_button = (
        int(fs.button_seat) if fs.button_seat is not None else session.button_seat
    )

    # Button-ONLY stability, decoupled from the (button, sitting_out) snapshot
    # below. The button never moves mid-hand, so a stable button MOVE is a
    # strong new-hand signal that must not wait on sitting_out churn.
    if observed_button == live_state.last_observed_button:
        live_state.button_stable_ticks += 1
    else:
        live_state.last_observed_button = observed_button
        live_state.button_stable_ticks = 1

    hero_hole_indices: tuple[int, ...] | None = None
    if all(c is not None for c in fs.hero_hole):
        hero_hole_indices = tuple(int(_card_idx(c)) for c in fs.hero_hole)

    # (review 2026-09-20 I1) Adopt hero's first full read of the hand as the
    # rotation baseline. Hands usually start (button move) while hero's cards
    # are still hidden, so `_begin_new_hand` has nothing to record and the
    # baseline is None; without adopting here it stayed on the PREVIOUS
    # hand's cards, and the anti-collusion reveal then looked like a
    # permanent "hole rotated" signal waiting for one bad frame to fire.
    if (
        hero_hole_indices is not None
        and live_state.last_hero_hole is None
        and session.hand_in_hand_mask
    ):
        live_state.last_hero_hole = hero_hole_indices

    # Debounce the button + participant snapshot over 2 consecutive
    # ticks so a glitched frame can't trigger a false hand-start. The
    # anchor frame captures the first tick of a new snapshot and is
    # used downstream for stack seeding / reconstructor rebaseline.
    snapshot = (observed_button, observed_sitting_out)
    if (
        snapshot == (live_state.pending_button, live_state.pending_sitting_out)
    ):
        live_state.pending_stable_ticks += 1
        # Opportunistic anchor upgrade: if the stored anchor has
        # glitched stack reads but this tick's fs is clean, swap in
        # the cleaner anchor. Keeps the debounce counter intact so
        # we don't reset progress just because the original anchor
        # was captured during the chip-settle animation.
        # Same for an anchor taken before the antes were collected.
        if (
            live_state.pending_anchor_fs is not None
            and not _anchor_ready(live_state.pending_anchor_fs)
            and _anchor_ready(fs)
        ):
            live_state.pending_anchor_fs = fs
    else:
        live_state.pending_button = observed_button
        live_state.pending_sitting_out = observed_sitting_out
        live_state.pending_stable_ticks = 1
        live_state.pending_anchor_fs = fs
        live_state.ante_wait_ticks = 0

    committed_ready = live_state.pending_stable_ticks >= threshold
    # Delay the commit one more tick if the anchor looks glitched.
    # The opportunistic upgrade above will replace it as soon as a
    # clean tick lands.
    if (
        committed_ready
        and live_state.pending_anchor_fs is not None
        and not _anchor_fs_stacks_plausible(live_state.pending_anchor_fs)
    ):
        logger.warning(
            "hand-start delayed: anchor_fs has glitched stack reads"
        )
        committed_ready = False
    # ...and while its pot shows the antes still uncollected (bounded wait;
    # the upgrade above swaps in the first frame that shows them).
    antes_pending = (
        not _anchor_shows_antes(live_state.pending_anchor_fs or fs)
        and live_state.ante_wait_ticks < _ANTE_WAIT_TICKS
    )
    # A button MOVE (off its prior seat), confirmed by a short run of identical
    # reads, fires a new hand — gated on the button alone, not the 6-tick
    # snapshot, so sitting_out flicker during the deal no longer delays it.
    # Still require the seeding anchor to have plausible stacks (the same guard
    # `committed_ready` applies for `first_commit`).
    button_threshold = (
        _BUTTON_STABLE_TICKS_LOCKED if is_locked else _STABILITY_TICKS_REQUIRED
    )
    button_changed = (
        observed_button != int(session.button_seat)
        and live_state.button_stable_ticks >= button_threshold
        and _anchor_fs_stacks_plausible(live_state.pending_anchor_fs or fs)
    )

    # Rewind-proof fallback: hero hole re-appears with cards that differ
    # from the snapshot we recorded last hand-start. Pre-lock (bootstrap
    # or first few ticks of a hand) we trust the read on a single tick,
    # since hero_hole OCR uses static per-card ROIs and is normally
    # stable. Once locked we require the same rotated indices to persist
    # for `threshold` consecutive ticks — a one-frame OCR glitch during
    # a banner / chip-settle animation must not wipe the hand.
    hero_hole_rotated_now = (
        hero_hole_indices is not None
        and live_state.last_hero_hole is not None
        and live_state.last_hero_hole != hero_hole_indices
        and set(live_state.last_hero_hole).isdisjoint(hero_hole_indices)
    )
    if not is_locked:
        hero_hole_rotated = hero_hole_rotated_now
        live_state.pending_hero_hole_rotation = None
        live_state.pending_hero_hole_rotation_ticks = 0
    else:
        if hero_hole_rotated_now and (
            hero_hole_indices == live_state.pending_hero_hole_rotation
        ):
            live_state.pending_hero_hole_rotation_ticks += 1
        elif hero_hole_rotated_now:
            live_state.pending_hero_hole_rotation = hero_hole_indices
            live_state.pending_hero_hole_rotation_ticks = 1
        else:
            live_state.pending_hero_hole_rotation = None
            live_state.pending_hero_hole_rotation_ticks = 0
        hero_hole_rotated = (
            live_state.pending_hero_hole_rotation_ticks >= threshold
        )

    # First-time commit: when no hand has been committed yet (empty
    # in-hand mask) and the debounce is ready, seed the hand. This is
    # how we bootstrap when OCR starts mid-hand — there's no button
    # change to trigger on since session.button_seat was at its default.
    # (review 2026-09-20 F10) ...but only when the anchor actually has
    # someone in the hand. Between hands every seat reads folded, the commit
    # produced an EMPTY mask, and an empty mask re-armed `first_commit` —
    # `_begin_new_hand` re-fired on every tick until cards were dealt.
    anchor_for_commit = live_state.pending_anchor_fs or fs
    first_commit = (
        committed_ready
        and not session.hand_in_hand_mask
        and any(not s.folded for s in anchor_for_commit.seats)
    )

    # Hand-start triggers: real hand-boundary events (button rotation,
    # hero-hole rotation) plus the very first commit after OCR start.
    # A mid-hand change in `observed_sitting_out` (a fold, or a banner
    # flicker) is NOT a hand-start — the reconstructor emits FOLD
    # events for real folds, and the `hand_in_hand_mask` / banner
    # signals defend against OCR flicker.
    trigger_fired = button_changed or hero_hole_rotated or first_commit

    # OCR-confirm guard: a real hand boundary always moves the button
    # to a different seat. If OCR still reads the button on the seat
    # we recorded at the last hand-start, the trigger source is
    # spurious (seen with ClubGG anti-collusion's mid-hand hero-hole
    # reveal tripping `hero_hole_rotated` against a stale baseline).
    # `first_commit` is exempt — bootstrap from an empty mask predates
    # any meaningful `session.button_seat`. We only check that the
    # button has moved off its prior seat, not that it landed on the
    # next clockwise seat: with <6 active players the button can skip.
    # (review 2026-09-20 I1) The move must be POSITIVELY read on this frame:
    # an unreadable button (None) confirms nothing. It used to skip the
    # guard, so with `hero_hole_rotated` latched, ONE occluded-button frame
    # restarted a live hand (log wiped, locks cleared, stacks re-seeded).
    # `button_changed` can't be true on such a frame, so only the hero-hole
    # trigger is affected.
    if trigger_fired and not first_commit:
        if (
            fs.button_seat is None
            or int(fs.button_seat) == int(session.button_seat)
        ):
            trigger_fired = False
            # A confirmed hole rotation while the button is positively read
            # on its hand-start seat means the BASELINE was stale (recorded
            # from the previous hand's still-visible cards), not that a hand
            # began. Adopt the cards so the rotation can't stay latched —
            # latched, it bypassed the button debounce: a single misread
            # button frame would have fired it.
            if hero_hole_rotated and fs.button_seat is not None:
                live_state.last_hero_hole = hero_hole_indices
                live_state.pending_hero_hole_rotation = None
                live_state.pending_hero_hole_rotation_ticks = 0

    # Hold a confirmed hand-start while the anchor predates the antes.
    if trigger_fired and antes_pending:
        if live_state.ante_wait_ticks == 0:
            logger.info("hand-start delayed: the pot doesn't show the antes yet")
        live_state.ante_wait_ticks += 1
        trigger_fired = False

    if os.environ.get("PLO5BP_OCR_DEBUG_HANDSTART"):
        logger.info(
            "ocr.handstart: raw_btn=%s obs_btn=%s sess_btn=%s btn_ticks=%d "
            "locked=%s snap_ticks=%d | btn_chg=%s hole_rot=%s first=%s -> %s",
            fs.button_seat, observed_button, session.button_seat,
            live_state.button_stable_ticks, is_locked,
            live_state.pending_stable_ticks,
            button_changed, hero_hole_rotated, first_commit,
            "FIRED" if trigger_fired else "-",
        )

    if trigger_fired:
        anchor_fs = live_state.pending_anchor_fs or fs
        _begin_new_hand(
            anchor_fs,
            button_seat=observed_button,
            hero_hole_indices=hero_hole_indices,
        )
        live_state.pending_anchor_fs = None

    # Mid-hand mask expansion. Backstop for the case where the anchor
    # frame fired before a participant's cards-back rendered — that
    # seat reads `folded=True` at anchor (so they're missing from
    # `hand_in_hand_mask`) but `folded=False` continuously thereafter.
    # We add them after a 2-tick stability window so single-frame OCR
    # flickers can't trigger a false addition. Strictly additive: the
    # mask never shrinks here. Late-rebuy players are safe because
    # they read `folded=True` (no cards/banner/commit/timer-bar);
    # already-folded players are excluded via `folded_this_hand`.
    # (review 2026-09-20 I8) Hero is a candidate like everyone else. Hero
    # was excluded here, so an anchor frame that landed before hero's card
    # backs rendered locked hero out for the whole hand: `_rebuild_env`
    # dropped the mask (six antes) and hero got no recommendation. Hero's
    # `folded` comes from the same multi-signal rule as the other seats.
    if session.hand_in_hand_mask:
        candidate_additions = frozenset(
            i for i, s in enumerate(fs.seats)
            if (not s.folded
                and i not in session.hand_in_hand_mask
                and i not in session.folded_this_hand)
        )
        if candidate_additions and candidate_additions == live_state.pending_mask_additions:
            live_state.pending_mask_additions_ticks += 1
        else:
            live_state.pending_mask_additions = candidate_additions
            live_state.pending_mask_additions_ticks = 1 if candidate_additions else 0
        if live_state.pending_mask_additions_ticks >= _STABILITY_TICKS_REQUIRED:
            session.hand_in_hand_mask = frozenset(
                session.hand_in_hand_mask | candidate_additions
            )
            live_state.pending_mask_additions = frozenset()
            live_state.pending_mask_additions_ticks = 0
            # The mask decides who the engine deals in; the env built with
            # the old mask must not feed this tick's EngineView. (review I5)
            session.env = None

    # Refresh the live sitting-out set every tick from the sticky mask
    # plus any folds the reconstructor has emitted. This makes a
    # banner-flicker or a transient `has_cards_back` miss invisible to
    # downstream consumers: once a seat is in the anchor mask it stays
    # in the hand until the reconstructor says otherwise. Before any
    # commit, leave `sitting_out_seats` as-is (default frozenset()) so
    # the stability gate alone decides when to trust OCR participants.
    if session.hand_in_hand_mask:
        all_seats = frozenset(range(session.num_seats))
        session.sitting_out_seats = (
            (all_seats - session.hand_in_hand_mask) | session.folded_this_hand
        )


def _pre_street_frame(fs: Any) -> Any:
    """``fs`` as it stood before the current street's bets: every visible
    commit moved back behind (stack + commit), nothing committed.

    The walk only accepts a commit a matching stack drop (or bet banner)
    corroborates, measured against the reconstructor's baseline. Rebaselining
    on this frame lets it re-derive, from an EXACT source's per-seat commits,
    bets that were already on the table when the baseline had to be reset.
    """
    return dataclasses.replace(
        fs,
        seats=tuple(
            dataclasses.replace(
                s,
                stack_chips=(
                    None
                    if s.stack_chips is None
                    else int(s.stack_chips) + int(s.committed_chips or 0)
                ),
                committed_chips=None,
            )
            for s in fs.seats
        ),
    )


def _begin_new_hand(
    fs: Any,
    *,
    button_seat: int,
    hero_hole_indices: tuple[int, ...] | None,
    exact_commits: bool = False,
) -> None:
    """Seed session state for a newly-observed hand.

    Pulls ``cfg.starting_stacks`` from the anchor frame's OCR-cent reads
    (converted to engine chips + ante), locks the hand-start participant
    mask from the anchor frame (seats whose `folded=False` at hand-start
    are the ones dealt into this hand), rebaselines the reconstructor so
    diff-based action inference starts from a clean slate, and records
    the hero-hole snapshot so a future rewind can distinguish "same hand
    again" from "new hand".

    ``exact_commits`` (PokerNow): the frame's per-seat ``committed_chips``
    are authoritative, so a hand-start that lands mid-street accounts for
    the chips already in front of each seat (see the seeding loop).
    """
    _new_session_defaults()
    session.button_seat = int(button_seat)
    if _live_source() == "ocr":
        current_live_state().deck_hands += 1
        _check_deck_colours()

    # Lock the participant mask from the anchor frame. Once a seat is
    # in this mask it stays in the hand until the reconstructor emits a
    # FOLD event (tracked in `folded_this_hand`). Banner flickers and
    # transient `has_cards_back` misses can't remove them.
    n = session.num_seats
    in_hand_mask = frozenset(
        i for i in range(n) if i < len(fs.seats) and not fs.seats[i].folded
    )
    session.hand_in_hand_mask = in_hand_mask
    session.folded_this_hand = frozenset()
    all_seats = frozenset(range(n))
    session.sitting_out_seats = all_seats - in_hand_mask

    cfg = session.game_config
    existing = list(cfg.resolved_stacks)
    merged = list(existing)
    for i in range(n):
        cents = fs.seats[i].stack_chips if i < len(fs.seats) else None
        # A glitched anchor OCR read — stack label covered by the blue
        # bet-banner or chip-settle animation — occasionally returns 0
        # or an absurdly small cent value. Seeding merged[i] from such a
        # read can make the engine's ante posting wipe the stack to 0
        # and flip the seat to all-in at hand-start, before any action
        # events have been applied. Skip the read and keep the prior
        # seeded value when it looks implausibly small. `None` already
        # falls through via the untouched branch below.
        if cents is not None and int(cents) < _MIN_PLAUSIBLE_STACK_CENTS:
            continue
        if cents is None:
            continue
        # (review 2026-09-20 I9) starting = behind + this street's visible
        # commit + ante. When the anchor lands mid-street (source attached
        # mid-hand, hand-start signal arriving after a bet) the chips already
        # in front of the seat are NOT in its stack read; seeding
        # `behind + ante` alone meant the walk re-derived that bet as an
        # action and deducted it a second time. Exact sources only: a pixel
        # `committed_chips` read is too noisy to fold into a stack.
        committed_cents = fs.seats[i].committed_chips if exact_commits else None
        committed = (
            _ocr_cents_to_engine_chips(int(committed_cents))
            if committed_cents is not None and int(committed_cents) > 0
            else 0
        )
        merged[i] = (
            _ocr_cents_to_engine_chips(int(cents)) + committed + int(cfg.ante)
        )
    if tuple(merged) != tuple(existing):
        # `replace` keeps variant/sb (review 2026-09-20 I10).
        session.game_config = dataclasses.replace(
            cfg, starting_stacks=tuple(merged)
        )

    # (review 2026-09-20 I1) Unconditional: when hero's cards aren't readable
    # at hand start (the usual case — dealt face-down until hero's turn) the
    # baseline must become None, NOT stay on the previous hand's cards.
    # `_mirror_observable_state` adopts the first full read of this hand.
    live_state.last_hero_hole = hero_hole_indices

    # Rebaseline whichever live source is currently driving the session
    # (ClubGG OCR or PokerNow ingest). Both runners register their
    # reconstructor as the active one when they start. With exact commits
    # the baseline is the frame as it stood BEFORE this street's bets, so
    # the walk re-derives them (matching the stacks seeded above).
    recon = _active_reconstructor()
    if recon is not None:
        recon.rebaseline(_pre_street_frame(fs) if exact_commits else fs)

    # (review 2026-09-20 I5) The env still describes the PREVIOUS hand. Both
    # runners build their EngineView right after this returns; invalidate so
    # it is rebuilt from the new button / mask / stacks (every reader of
    # `session.env` rebuilds on None).
    session.env = None


def _engine_view_from_session(exact_folds: bool = False) -> Any:
    """Build an EngineView snapshot from the current session env.

    ``exact_folds`` is set by the PokerNow path (the DOM exposes an exact
    ``fold`` class) so the walk never *infers* a fold from a swept closing
    call. Imported lazily so server import doesn't pull in ocr deps when the
    OCR extras aren't installed.
    """
    from plo5bp.ocr.events import EngineView

    env = session.env
    if env is None:
        _rebuild_env()
        env = session.env
    assert env is not None
    # Actor / street / commits only: skip the opp-outcome Monte Carlo the
    # default observation computes (TOOL-011).
    raw = env._rs.observation_dict(skip_outcome_mc=True)
    n = session.num_seats
    hero_seat = int(session.hero_seat)
    actor_raw = raw.get("actor")
    actor = int(actor_raw) if actor_raw is not None else None
    cfg = session.game_config
    # One definition of the scale, shared with `_ocr_cents_to_engine_chips`.
    chips_per_cent = _chips_per_cent() or 1.0
    # Noise floor: 1 bb in cents. Any stack-delta smaller than this must
    # be OCR jitter — no legal bet is under 1 bb.
    min_bet_cents = int(round(float(cfg.bb) / chips_per_cent)) if chips_per_cent > 0 else 0
    return EngineView(
        num_seats=n,
        current_actor=actor,
        street=int(raw.get("street", 0)),
        awaiting_next_street=raw.get("awaiting_next_street"),
        button_seat=int(session.button_seat),
        committed_this_street=tuple(int(x) for x in raw["street_commit"][:n]),
        stacks=tuple(int(x) for x in raw["stacks"][:n]),
        folded=tuple(bool(x) for x in raw["folded"][:n]),
        all_in=tuple(bool(x) for x in raw["all_in"][:n]),
        bet_to_call=int(raw.get("bet_to_call", 0)),
        chips_per_cent=chips_per_cent,
        min_bet_cents=min_bet_cents,
        # `sitting_out` tells the walk which seats the ENGINE retires on its
        # own (`_auto_fold_sitting_out`). Hero is never auto-acted (review
        # 2026-09-20 I8), so hero is never reported here even while outside
        # the mask — otherwise the walk would skip a seat the engine is
        # waiting on and attribute the next action to the wrong player. A
        # hero who really folded is covered by the engine's `folded` flag.
        sitting_out=tuple(
            i in session.sitting_out_seats and i != hero_seat for i in range(n)
        ),
        exact_folds=exact_folds,
    )
