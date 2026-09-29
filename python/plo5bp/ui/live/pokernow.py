"""PokerNow live source: the Tampermonkey collector POSTs `pokernow.v1` DOM
snapshots, mapped by `plo5bp.ocr.pokernow.map_payload` into the same
`FrameState` -> `EventReconstructor` -> session pipeline ClubGG uses
(local build only). See tools/pokernow/README.md."""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from typing import Any

from fastapi import HTTPException

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.ui.common import STREET_NAMES
from plo5bp.ui.live import record
from plo5bp.ui.live.state import live_state
from plo5bp.ui.live.tracking import (
    _LIVE_FORMAT_ERROR,
    _MAX_ENGINE_SEATS,
    _begin_new_hand,
    _card_idx,
    _engine_view_from_session,
    _live_capture_allowed,
    _note_live_source,
    _ocr_apply_card_slot,
    _ocr_cents_to_engine_chips,
    _pre_street_frame,
    _reconcile_missed_checks_on_street_reveal,
    _reconcile_missed_folds_on_street_reveal,
    _check_pot_sync,
    _clear_stale_refusal,
    _record_seat_actions,
    live_warnings,
    _rebuild_env_if_stale,
    _set_active_reconstructor,
)
from plo5bp.ui.server import (
    _build_env,
    _clear_hand_state_keep_cards,
    _session_env_spec,
    session,
)

logger = logging.getLogger("plo5bp.ui.live")

#: Oldest userscript that reports everything the server relies on: `allIn`
#: from the DOM (1.2.0), `collector` + `gap` (1.3.0). Bump together with
#: tools/pokernow/pokernow.user.js when the payload contract changes.
REQUIRED_COLLECTOR = (1, 3, 0)


def _version_tuple(v: str | None) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in str(v).split("."))
    except (TypeError, ValueError):
        return (0,)


# --- PokerNow DOM ingest ----------------------------------------------------

def _count_prefix(cards: tuple[Any, ...]) -> int:
    """Length of the leading run of non-None cards (board fill level)."""
    n = 0
    for c in cards:
        if c is None:
            break
        n += 1
    return n


def _pokernow_set_seat_count(n: int) -> bool:
    """Reconfigure the session for an ``n``-seat PokerNow table.

    PokerNow tables vary in size hand-to-hand; unlike ClubGG (fixed 6),
    the engine seat count must track the players actually dealt in. Hero
    stays engine seat 0 (the mapper rotates the table so hero is first).

    Returns False — leaving the session untouched — when the engine can't
    represent the table: PokerNow seats up to 10, the engine 2..8
    (`GameConfig` rejects the rest; a 9-handed PLO5 double board doesn't
    even fit the deck). The caller surfaces that as a status message
    instead of crashing the ingest loop. (review 2026-09-20 B8 contract)
    """
    if not (2 <= int(n) <= _MAX_ENGINE_SEATS):
        return False
    try:
        # `replace` keeps variant/sb (review 2026-09-20 I10).
        new_cfg = dataclasses.replace(
            session.game_config, num_seats=int(n), starting_stacks=None
        )
    except ValueError as e:
        logger.warning("pokernow: %d-seat table rejected by GameConfig: %s", n, e)
        return False
    session.game_config = new_cfg
    session.num_seats = int(n)
    session.hero_seat = 0
    if session.button_seat >= n:
        session.button_seat = 0
    _clear_hand_state_keep_cards()
    session.hand_in_hand_mask = frozenset()
    session.folded_this_hand = frozenset()
    session.sitting_out_seats = frozenset()
    live_state.last_hero_hole = None
    # (review 2026-09-20 I5/F14) The env still has the OLD seat count. The
    # very next thing the runner does is build an EngineView from it:
    # `num_seats` from the session, per-seat arrays from the stale env ⇒
    # IndexError on every frame, i.e. /pokernow/ingest 500'd forever after a
    # player sat down or left. Invalidate; readers rebuild on None.
    session.env = None
    return True


def _commits_are_lingering_antes(fs: Any) -> bool:
    """True when every live seat shows a commit of exactly one ante.

    PokerNow renders the bomb-pot ante as a per-seat bet before the flop and
    normally clears it by the first flop frame. Should a frame still show
    them, they must not be mistaken for flop bets (the engine posts antes
    itself): the hand-start then seeds and rebaselines exactly as it always
    did. A real all-seats-matched bet of precisely one ante is swept to the
    pot the instant it closes, so misreading one costs at worst the old
    behaviour.
    """
    ante = int(session.game_config.ante)
    live = [s for s in fs.seats if not s.folded]
    if ante <= 0 or not live:
        return False
    return all(
        s.committed_chips is not None
        and _ocr_cents_to_engine_chips(int(s.committed_chips)) == ante
        for s in live
    )


def _pokernow_mirror_cards(fs: Any) -> None:
    """Mirror hero hole + both boards from a PokerNow FrameState.

    PokerNow card reads are exact (DOM text), so we commit immediately
    (``debounce=False``) — no multi-tick stability gate is needed. The
    per-slot lock still lets the user override a slot via ``/cards``.
    Villain cards stay hidden (PokerNow only reveals them at showdown),
    so only hero + community cards are mirrored, exactly like the OCR path.
    """
    # reject_duplicates: DOM reads are exact, but a slot the user overrode via
    # /cards can collide with one — never lock a duplicate (review H7).
    def put(attr: str, idx: int, card: Any) -> None:
        _ocr_apply_card_slot(
            attr, idx, _card_idx(card), debounce=False, reject_duplicates=True
        )

    for i, c in enumerate(fs.hero_hole):
        put("hero_hole", i, c)
    for i, c in enumerate(fs.board_a[:3]):
        put("flop_a", i, c)
    for i, c in enumerate(fs.board_b[:3]):
        put("flop_b", i, c)
    put("turn_cards", 0, fs.board_a[3])
    put("turn_cards", 1, fs.board_b[3])
    put("river_cards", 0, fs.board_a[4])
    put("river_cards", 1, fs.board_b[4])


class PokerNowRunner:
    """Receives normalized DOM snapshots from the PokerNow userscript and
    drives the same session pipeline the OCR runner uses.

    Unlike ``OcrRunner`` there is no capture loop — the browser-side
    Tampermonkey collector POSTs a ``pokernow.v1`` payload to
    ``/pokernow/ingest`` on every table change (plus a heartbeat), and each
    payload is processed by ``handle_payload`` (run on a worker thread,
    under the live lock, because ``_rebuild_env`` is synchronous CPU work).

    Because PokerNow data is exact, the reconstructor's OCR-disambiguation
    fallbacks never fire; the per-seat committed amount and explicit actor
    come straight from the DOM. The one PokerNow-specific wrinkle vs ClubGG
    is hand-start: bomb pots post antes (which PokerNow renders as a per-seat
    bet) before the flop, so we wait until the flop is on the felt to fire
    ``_begin_new_hand`` — that captures post-ante stacks and a clean
    zero-commit baseline, matching what ClubGG's debouncer lands on.
    """

    # A POST-based collector looks "connected" while snapshots keep arriving;
    # the userscript heartbeats every ~2s so an idle table stays green.
    _STALE_SECONDS = 6.0
    # Seconds the locked game can be silent before another game may take over.
    # Longer than the heartbeat so a brief gap mid-hand never hands the session
    # to a second open tab; short enough that intentionally switching tables
    # recovers within a few seconds.
    _GAME_SWITCH_STALE = 8.0

    def __init__(self) -> None:
        self.frames_seen: int = 0
        self.events_applied: int = 0
        self.last_error: str | None = None
        self.last_tick_at: float | None = None
        self.last_post_at: float | None = None
        self.bomb_pot: bool = False
        self.variant: str | None = None
        self.table_seats: int = 0
        self._reconstructor: Any = None
        self._last_payload_key: str | None = None
        # Game-lock: the bridge processes frames from exactly ONE PokerNow game
        # at a time. A second open tab (another table) posts a different
        # `gameId`; those frames are ignored so they can't interleave into the
        # live hand. Switches only after the locked game goes quiet.
        self.active_game_id: str | None = None
        self._active_game_at: float | None = None
        self.ignored_frames: int = 0
        # Events the last payload produced (replay / diagnostics).
        self.last_events: list[Any] = []
        # The userscript's version (`collector`; None = before 1.3.0) and a
        # warning while the hand in progress followed a gap in the stream.
        self.collector: str | None = None
        self.gap_warning: str | None = None

    def accept_game(self, game_id: Any) -> bool:
        """Return True if a frame from ``game_id`` should be processed.

        Untagged frames (older userscript) are always accepted. A tagged
        frame is accepted when it matches the locked game, no game is locked
        yet, or the locked game has been silent past ``_GAME_SWITCH_STALE``.
        """
        if game_id is None:
            return True
        now = time.time()
        if (
            self.active_game_id is None
            or game_id == self.active_game_id
            or self._active_game_at is None
            or (now - self._active_game_at) > self._GAME_SWITCH_STALE
        ):
            if game_id != self.active_game_id:
                logger.info("pokernow: locking onto game %s", game_id)
                # New game takes over: drop the prior hand's state cleanly.
                self._reconstructor = None
            self.active_game_id = game_id
            self._active_game_at = now
            return True
        self.ignored_frames += 1
        return False

    @property
    def connected(self) -> bool:
        return (
            self.last_post_at is not None
            and (time.time() - self.last_post_at) < self._STALE_SECONDS
        )

    def status(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "frames_seen": self.frames_seen,
            "events_applied": self.events_applied,
            "last_error": self.last_error,
            "last_tick_at": self.last_tick_at,
            "bomb_pot": self.bomb_pot,
            "variant": self.variant,
            "table_seats": self.table_seats,
            "active_game_id": self.active_game_id,
            "ignored_frames": self.ignored_frames,
            "collector": self.collector,
            "collector_outdated": self.collector_outdated,
            "warnings": [
                *([self._update_message()] if self.collector_outdated else []),
                *([self.gap_warning] if self.gap_warning else []),
                *live_warnings(),
            ],
        }

    @property
    def collector_outdated(self) -> bool:
        """True once a payload came from a userscript older than
        `REQUIRED_COLLECTOR` (TOOL-016)."""
        if self.frames_seen == 0 and self.last_post_at is None:
            return False  # nothing heard yet
        return _version_tuple(self.collector) < REQUIRED_COLLECTOR

    def _update_message(self) -> str:
        have = self.collector or "an older version"
        need = ".".join(map(str, REQUIRED_COLLECTOR))
        return (
            f"update your PokerNow userscript ({have} → {need}): open "
            f"http://127.0.0.1:8765/pokernow/pokernow.user.js"
        )

    def note_post(self) -> None:
        """Record that a snapshot/heartbeat arrived (HTTP POST path)."""
        self.last_post_at = time.time()

    def is_duplicate(self, payload: dict) -> bool:
        """Cheap server-side dedupe so heartbeats don't re-run the pipeline.

        The userscript already dedupes by content, but heartbeats resend the
        last snapshot to keep the connection fresh; skip reprocessing those.
        """
        import json as _json

        try:
            key = _json.dumps(payload, sort_keys=True, default=str)
        except Exception:
            return False
        if key == self._last_payload_key:
            return True
        self._last_payload_key = key
        return False

    def _apply_button_correction(self, fs: Any) -> None:
        """Move the button of the CURRENT hand (cards, mask, stacks, locks and
        the hero-hole baseline all stay) and rebuild under it.

        The acting order hangs off the button, so actions the walk recorded
        under the stale button may now replay onto the wrong seats. Each
        entry carries the seat it was recorded for: when the replay under the
        corrected button contradicts those stamps (or the engine rejects an
        entry) the log is provably mis-ordered and is dropped, and the walk —
        run on this same frame — re-derives the street from PokerNow's exact
        per-seat commits. For that the reconstructor is rebaselined on the
        frame as it stood BEFORE the street's bets (stack + commit behind,
        nothing committed): the walk only accepts a commit that a matching
        stack drop corroborates. A log that still replays consistently is
        kept as is.
        """
        session.button_seat = int(fs.button_seat)
        session.env = None
        if not session.action_log:
            return
        try:
            build = _build_env(_session_env_spec())
        except HTTPException:
            return
        if not (build.seat_mismatches or build.dropped_entries):
            return
        logger.warning(
            "pokernow: button corrected to seat %d mid-hand; re-deriving %d "
            "action(s) recorded under the stale acting order",
            session.button_seat, len(session.action_log),
        )
        session.action_log = []
        session.folded_this_hand = frozenset()
        session._replay_mismatch_warned = frozenset()
        if self._reconstructor is not None:
            self._reconstructor.rebaseline(_pre_street_frame(fs))

    def _skip_frame(self, reason: str) -> None:
        """Ignore a frame the session can't represent, without mutating it.
        The reason shows up in /pokernow/status; logged once per change."""
        if self.last_error != reason:
            logger.warning("pokernow: %s", reason)
        self.last_error = reason
        self.last_tick_at = time.time()

    def handle_payload(self, payload: dict) -> None:
        """Apply one `pokernow.v1` snapshot to the session (also the entry
        point of `live.replay`). Recorded when PLO5BP_LIVE_RECORD is set."""
        self.last_events = []
        rec = record.recorder()
        if rec is not None and not rec.active:
            rec.open("pokernow")
        try:
            self._handle_payload(payload)
        finally:
            if rec is not None and rec.active:
                rec.record(
                    source="pokernow", payload=payload, events=self.last_events,
                    log=session.action_log, warnings=live_warnings(),
                )

    def _handle_payload(self, payload: dict) -> None:
        # PLO5-only (review 2026-09-20 I10) — checked before anything is
        # imported or touched. The HTTP/WS endpoints 409 first; this covers
        # direct callers and a format flip while frames are in flight.
        if not _live_capture_allowed():
            self._skip_frame(_LIVE_FORMAT_ERROR)
            return

        from plo5bp.ocr.events import (
            OcrWarning,
            StreetReveal,
        )
        from plo5bp.ocr.events import EventReconstructor
        from plo5bp.ocr.pokernow import map_payload

        try:
            pf = map_payload(payload)
        except ValueError as e:
            # A malformed snapshot (the mapper raises a ValueError subclass)
            # is the sender's error, not ours: 400, session untouched.
            self._skip_frame(str(e))
            raise HTTPException(status_code=400, detail=str(e)) from e
        fs = pf.frame
        n = pf.num_seats
        self.collector = pf.collector
        if pf.gap:
            # TOOL-040: the userscript had to drop frames (the server was
            # unreachable for a long stretch). Actions may be missing from the
            # hand in progress; the next hand starts clean.
            self.gap_warning = (
                f"{pf.gap} PokerNow frame(s) were lost while the app was "
                "unreachable — this hand may be out of sync (the next hand "
                "starts clean)"
            )
            logger.warning("pokernow: %s", self.gap_warning)
        # Between hands PokerNow can briefly show <2 in-hand seats; skip
        # those frames rather than collapsing the engine seat count.
        if n < 2:
            return

        self.frames_seen += 1
        self.bomb_pot = pf.bomb_pot
        self.variant = pf.variant
        self.table_seats = n

        # PLO4 / PLO6 / NLH / single-board tables map through the same 5-card,
        # two-board shape and would get confident but wrong advice (TOOL-015).
        reason = pf.unsupported_reason()
        if reason is not None:
            self._skip_frame(reason)
            return

        # PokerNow seats up to 10; the engine tops out at 8. Keep the
        # previous config and say so rather than crash-looping the ingest.
        if n > _MAX_ENGINE_SEATS:
            self._skip_frame(
                f"PokerNow hand has {n} players; the engine supports at most "
                f"{_MAX_ENGINE_SEATS} — frame ignored"
            )
            return

        # First PokerNow frame after ClubGG OCR (or ever): reset the shared
        # hand-start machine so nothing from the other source leaks in.
        _note_live_source("pokernow")

        _pn_reconf = n != session.num_seats
        _pn_raw_seats = [
            (
                s.get("seat"),
                s.get("stackDollars"),
                s.get("betDollars") if s.get("betDollars") is not None else s.get("betText"),
                "fold" if s.get("folded") else ("allin" if s.get("allIn") else ""),
                len(s.get("cards") or []),
            )
            for s in (payload.get("seats") or [])
        ]

        # Track the table size; reconfigure + rebuild the reconstructor
        # when it changes (player joined/left between hands).
        if n != session.num_seats:
            if not _pokernow_set_seat_count(n):
                self._skip_frame(
                    f"PokerNow hand has {n} players; the engine can't "
                    "represent that table — frame ignored"
                )
                return
            self._reconstructor = None
        if self._reconstructor is None or self._reconstructor.num_seats != session.num_seats:
            self._reconstructor = EventReconstructor(num_seats=session.num_seats)
        # (review 2026-09-20 I9) Register on EVERY payload, not only when a
        # new reconstructor is created: after a ClubGG session at the same
        # seat count nothing re-registered PokerNow's, so `_begin_new_hand`
        # kept rebaselining the dead OCR reconstructor.
        _set_active_reconstructor(self._reconstructor)

        _pokernow_mirror_cards(fs)
        live_state.observed_stacks = tuple(s.stack_chips for s in fs.seats)
        live_state.observed_pot = fs.pot_total_chips

        flop_present = (
            _count_prefix(fs.board_a) >= 3 and _count_prefix(fs.board_b) >= 3
        )
        hero_indices: tuple[int, ...] | None = None
        if all(c is not None for c in fs.hero_hole):
            hero_indices = tuple(int(_card_idx(c)) for c in fs.hero_hole)

        button_changed = (
            fs.button_seat is not None
            and int(fs.button_seat) != int(session.button_seat)
        )
        # Hero's hole cards are the most reliable new-hand signal: dealt at
        # hand start, visible throughout, exact, and re-dealt every hand.
        # Compare as a SET (order-insensitive — PokerNow may re-sort hero's
        # cards mid-hand) and fire on ANY change. We deliberately do NOT
        # require the new hand to be card-disjoint from the last (that's a
        # ClubGG OCR-noise guard; consecutive PokerNow deals often share a
        # card, which would suppress the trigger). The dealer-button DOM can
        # lag the flop deal, so `button_changed` is only a secondary signal.
        hero_changed = hero_indices is not None and (
            live_state.last_hero_hole is None
            or set(hero_indices) != set(live_state.last_hero_hole)
        )
        first_commit = not session.hand_in_hand_mask
        # (review 2026-09-20 I9) A button change while hero still holds the
        # SAME cards is not a new hand — it is the lagging dealer-button DOM
        # catching up after the hand already started (on the hero-card
        # signal) under the previous button. Treating it as a hand-start
        # wiped the live hand mid-street and re-seeded stacks from a frame
        # with chips already in front of the seats. It is a correction to
        # the current hand instead. (Hero cards unreadable ⇒ the button is
        # the only new-hand signal left, so that case still starts a hand.)
        button_correction = (
            button_changed
            and not first_commit
            and not hero_changed
            and hero_indices is not None
        )
        new_hand = (
            first_commit
            or hero_changed
            or (button_changed and not button_correction)
        )

        # Defer the hand-start to the flop so the anchor frame carries
        # post-ante stacks + cleared street commits (see class docstring).
        # No stack-plausibility gate here (unlike the OCR path): PokerNow
        # reads are exact, and `_begin_new_hand` already skips per-seat
        # None/short stacks — gating the whole hand-start on it would wrongly
        # block a hand where a villain ante'd all-in (stack renders empty).
        if new_hand and flop_present:
            btn = (
                int(fs.button_seat)
                if fs.button_seat is not None
                else int(session.button_seat)
            )
            self.gap_warning = None  # a fresh hand: nothing is missing
            _begin_new_hand(
                fs,
                button_seat=btn,
                hero_hole_indices=hero_indices,
                # Commits on the anchor are real flop bets (the hand-start
                # signal arrived mid-street) — unless they are just the ante
                # bets still on display, which the engine posts itself.
                exact_commits=not _commits_are_lingering_antes(fs),
            )
            # _begin_new_hand wiped the card spec via _new_session_defaults;
            # re-mirror this frame so the flop/hero cards survive the reset.
            _pokernow_mirror_cards(fs)
        elif button_correction:
            self._apply_button_correction(fs)

        _pn_dbg = os.environ.get("PLO5BP_PN_DEBUG")
        _street_before = None
        if _pn_dbg and session.env is not None:
            try:
                _street_before = int(session.env._rs.observation_dict(skip_outcome_mc=True).get("street", -1))
            except Exception:
                _street_before = -2

        # Action inference runs only once flop betting is live (the engine
        # posts antes itself; PokerNow's pre-flop ante bets are not actions).
        events: list = []
        street_reveals: list = []
        if flop_present and session.hand_in_hand_mask:
            # The hand-start / seat-count paths above invalidated the env, so
            # this view is rebuilt from the CURRENT hand. A state that can't
            # rebuild (e.g. a user card override colliding with the DOM) is
            # reported, never raised: an exception here used to 500 every
            # ingest POST from then on. (review 2026-09-20 I5/F14)
            try:
                engine_view = _engine_view_from_session(exact_folds=True)
            except HTTPException as e:
                self._skip_frame(f"rebuild failed: {e.detail}")
                return
            events = self._reconstructor.step(fs, engine_view)
            self.last_events = list(events)
            # Validated against the engine before they enter the log
            # (TOOL-025). No retry: a DOM read is exact, so the same payload
            # would be refused again — the warning asks for a manual entry.
            applied, _ = _record_seat_actions(events)
            self.events_applied += applied
            for ev in events:
                if isinstance(ev, StreetReveal):
                    self.events_applied += 1
                elif isinstance(ev, OcrWarning):
                    logger.warning("pokernow: %s", ev.message)

            street_reveals = [ev for ev in events if isinstance(ev, StreetReveal)]
            if street_reveals:
                # exact=True: the DOM fold flag is authoritative — no 2-tick
                # confirmation (the forced call fill below depends on folds
                # having been reconciled first).
                _reconcile_missed_folds_on_street_reveal(fs, exact=True)
                target_street = max(
                    2 if ev.street == "turn" else 3 for ev in street_reveals
                )
                # force=True: a PokerNow reveal proves the prior street closed,
                # but the closing call's committed/stack signal is gone (chips
                # swept to pot + next card dealt in the same instant). Fill the
                # remaining calls to close the street rather than stalling.
                _reconcile_missed_checks_on_street_reveal(target_street, force=True)

        try:
            _rebuild_env_if_stale()
            self.last_error = None
        except HTTPException as e:
            self.last_error = f"rebuild failed: {e.detail}"
            logger.warning("pokernow rebuild failed: %s", e.detail)
        except Exception as e:
            self.last_error = f"rebuild failed: {type(e).__name__}: {e}"
            logger.exception("pokernow rebuild failed")
        else:
            _clear_stale_refusal()
            _check_pot_sync(fs, exact=True)  # TOOL-026

        if _pn_dbg:
            try:
                # Read street/actor straight from the engine — do NOT call
                # _state_dict() here: it runs _compute_recommendation() (a model
                # forward pass), which on every ingest frame throttles the whole
                # bridge and forces the userscript to coalesce/drop actions.
                _raw = (
                    dict(session.env._rs.observation_dict(skip_outcome_mc=True))
                    if session.env is not None else {}
                )
                _dbg_street = STREET_NAMES.get(int(_raw.get("street", -1)), _raw.get("street"))
                _dbg_actor = _raw.get("actor")
                ev_kinds = ",".join(type(e).__name__ for e in events) or "-"
                _gate_ch = {int(GATE_FOLD): "F", int(GATE_CHECK_CALL): "C", int(GATE_RAISE): "R"}
                alog_str = "".join(
                    _gate_ch.get(int(e["gate"]), "?") + str(int(e["chips"]))
                    for e in session.action_log
                ) or "-"
                seats_str = " ".join(
                    f"{s.seat}:stk={s.stack_chips}bet={s.committed_chips}{'A' if s.is_actor else ''}{'F' if s.folded else ''}"
                    for s in fs.seats
                )
                line = (
                    f"f{self.frames_seen} b1={_count_prefix(fs.board_a)} "
                    f"b2={_count_prefix(fs.board_b)} n={n} reconf={_pn_reconf} "
                    f"mask={sorted(session.hand_in_hand_mask)} "
                    f"raw={_pn_raw_seats} actorDOM="
                    f"{[s.seat for s in fs.seats if s.is_actor]} "
                    f"new_hand={new_hand}(hc={hero_changed},bc={button_changed},fc={first_commit}) "
                    f"flop_present={flop_present} street_before={_street_before} "
                    f"events=[{ev_kinds}] reveals={len(street_reveals)} "
                    f"alog={alog_str} seats=[{seats_str}] -> street={_dbg_street} "
                    f"actor={_dbg_actor} pot={live_state.observed_pot} err={self.last_error}\n"
                )
                with open(_pn_dbg, "a", encoding="utf-8") as fh:
                    fh.write(line)
            except Exception:
                pass

        self.last_tick_at = time.time()


pokernow_runner = PokerNowRunner()
