"""ClubGG live capture: a Windows.Graphics.Capture feed of the table window,
pixel-OCR'd once per poll into a `FrameState` and applied to the study
session through `tracking` (local build only)."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import threading
import time
from typing import Any

from fastapi import HTTPException

from plo5bp.ui.live import record
from plo5bp.ui.live.state import live_state
from plo5bp.ui.live.tracking import (
    _LIVE_FORMAT_ERROR,
    _active_reconstructor,
    _engine_view_from_session,
    _live_capture_allowed,
    _live_source,
    _mirror_observable_state,
    _note_live_source,
    _reconcile_missed_checks_on_street_reveal,
    _reconcile_missed_folds_on_street_reveal,
    _check_pot_sync,
    _clear_stale_refusal,
    _record_seat_actions,
    live_warnings,
    _rebuild_env_if_stale,
    _reset_live_tracking,
    _set_active_reconstructor,
    live_lock,
)
from plo5bp.ui.server import _clear_hand_state_keep_cards, _rebuild_env, session

logger = logging.getLogger("plo5bp.ui.live")


class OcrRunner:
    """WGC-fed OCR session that mutates `session` once per polling tick.

    Frame *acquisition* happens on a free-threaded
    Windows.Graphics.Capture session against the picked window's HWND.
    The WGC callback runs on the binding's worker thread and publishes
    the latest BGR ndarray into ``self.latest_frame`` under
    ``self._frame_lock``. An asyncio task (``_loop``) wakes every
    ``poll_ms`` and runs ``_tick`` on a worker thread (extract +
    reconstruct + ``_rebuild_env`` is CPU-bound and the engine is not
    asyncio-aware).

    Why WGC and not Chrome's getDisplayMedia: ClubGG sets
    ``SetWindowDisplayAffinity`` on its tables. Chrome's per-window
    capture goes through GDI BitBlt / DXGI Desktop Duplication and
    respects WDA, so the captured frame shows whatever is behind the
    table. WGC reads from the DWM compositor surface and bypasses WDA
    in the same way OBS's "Windows 10 (1903 and up)" source does.
    """

    def __init__(self) -> None:
        self.running: bool = False
        self.poll_ms: int = 200
        self.window_match: str | None = None
        self.window_title: str | None = None
        self.candidates: list[str] = []
        # HWND of the captured window, persisted so `_loop` can poll its
        # liveness each tick. None when not running.
        self._hwnd: int | None = None
        # Why the runner last stopped: None for manual/never, "window_closed"
        # when it auto-stopped because the captured window was destroyed. The
        # frontend uses this to reset the picker only on auto-off.
        self.stopped_reason: str | None = None
        self.last_error: str | None = None
        self.last_tick_at: float | None = None
        self.frames_seen: int = 0
        self.events_applied: int = 0
        # Most recent BGR frame published by the WGC callback. Read by
        # `_tick` under `_frame_lock`.
        self.latest_frame: Any = None
        self._frame_lock: threading.Lock = threading.Lock()
        # Wall time the last tick took (ms) — the loop runs on fixed
        # deadlines, so a tick slower than `poll_ms` shows up here.
        self.last_tick_ms: float | None = None
        # Events the last processed frame produced (replay / diagnostics).
        self.last_events: list[Any] = []
        self._tick_view: Any = None
        # Why the last frame was rescaled or refused (`fit_to_calibration`).
        self.frame_note: str | None = None
        self._capture_control: Any = None
        self._reconstructor: Any = None
        self.task: asyncio.Task | None = None
        # Per-crop OCR read cache (stack/commit/pot). Lets `_tick` skip the
        # Tesseract subprocess for any chip ROI whose pixels are unchanged
        # since the last tick — most ticks then do ~0 reads. Cleared on start().
        self._ocr_read_cache: dict = {}
        # Time seams of the tick loop (tests drive a fake clock).
        self._clock: Any = time.monotonic
        self._sleep: Any = asyncio.sleep

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "poll_ms": self.poll_ms,
            "window_match": self.window_match,
            "window_title": self.window_title,
            "candidates": list(self.candidates),
            "last_tick_at": self.last_tick_at,
            "last_error": self.last_error,
            "stopped_reason": self.stopped_reason,
            "frames_seen": self.frames_seen,
            "tick_ms": self.last_tick_ms,
            "frame_note": self.frame_note,
            "warnings": live_warnings(),
            "events_applied": self.events_applied,
        }

    async def start(self, window_match: str, poll_ms: int) -> None:
        from plo5bp.ocr import live as ocr_live
        from plo5bp.ocr.events import EventReconstructor

        if self.running:
            return

        self.window_match = window_match
        self.poll_ms = int(poll_ms)
        self.last_error = None
        self.stopped_reason = None
        self.frames_seen = 0
        self.events_applied = 0
        self.candidates = []
        self.window_title = None
        self._hwnd = None
        self.latest_frame = None
        self._ocr_read_cache = {}

        try:
            wm = await asyncio.to_thread(ocr_live.find_window, window_match)
        except ocr_live.NoWindowError as e:
            try:
                self.candidates = await asyncio.to_thread(ocr_live.list_window_titles)
            except Exception:
                self.candidates = []
            self.last_error = str(e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ocr_live.MultipleWindowsError as e:
            self.candidates = list(e.candidates)
            self.last_error = str(e)
            raise HTTPException(status_code=400, detail=str(e)) from e

        self.window_title = wm.title
        self._hwnd = int(wm.hwnd)

        def _on_frame(bgr: Any) -> None:
            with self._frame_lock:
                self.latest_frame = bgr

        try:
            # Ticks read one frame per poll; don't copy every ~8 MB frame
            # ClubGG repaints at display rate (TOOL-036).
            self._capture_control = await asyncio.to_thread(
                functools.partial(
                    ocr_live.start_wgc_capture,
                    int(wm.hwnd),
                    _on_frame,
                    min_interval_s=self.poll_ms / 2000.0,
                )
            )
        except Exception as e:
            self.last_error = f"WGC start failed: {type(e).__name__}: {e}"
            logger.exception("WGC start failed for hwnd=%s", wm.hwnd)
            raise HTTPException(status_code=500, detail=self.last_error) from e

        # ClubGG always renders 6 physical seat positions; the ROI table is
        # tied to those absolute screen coords. Empty seats are handled by
        # the multi-signal in-hand detector. Force the session to 6 seats
        # so OCR output shape matches the engine state regardless of how
        # many seats were configured for non-OCR study.
        if session.num_seats != 6:
            # `replace` carries variant/sb (and any future field) — the
            # hand-written copy dropped them. (review 2026-09-20 I10)
            session.game_config = dataclasses.replace(
                session.game_config, num_seats=6, starting_stacks=None
            )
            session.num_seats = 6
            if session.button_seat >= 6:
                session.button_seat = 0
            session.hero_seat = 0
            _clear_hand_state_keep_cards()
            _rebuild_env()

        # A capture (re)start is a fresh live session: drop the previous
        # source's mask / debounce state so its stale anchor can't seed the
        # first hand. (review 2026-09-20 F10)
        _note_live_source("ocr")
        _reset_live_tracking()

        self._reconstructor = EventReconstructor(num_seats=session.num_seats)
        _set_active_reconstructor(self._reconstructor)
        rec = record.recorder()
        if rec is not None:
            rec.open("ocr")  # PLO5BP_LIVE_RECORD (TEST-028)
        self.running = True
        self.task = asyncio.create_task(self._loop())

    async def _release_capture(self) -> None:
        """Stop the WGC capture session if one is live (idempotent).

        Shared by the manual `stop()` path and the auto-off
        `_handle_window_closed()` path; the `cc is None` guard makes a second
        call a no-op if a manual Off races a window close.
        """
        cc = self._capture_control
        self._capture_control = None
        if cc is not None:
            try:
                await asyncio.to_thread(cc.stop)
            except Exception as e:
                logger.warning("WGC stop raised: %s", e)

    async def stop(self) -> None:
        was_running = self.running
        self.running = False
        self.stopped_reason = None  # manual stop
        task = self.task
        self.task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        await self._release_capture()
        self._hwnd = None
        if was_running:
            self.window_match = None
            self.window_title = None
        self._retire_reconstructor()
        rec = record.recorder()
        if rec is not None:
            rec.close()

    def _retire_reconstructor(self) -> None:
        """Unregister this runner's reconstructor as the live one.

        (review 2026-09-20 I9) It used to stay registered after OCR stopped,
        so a PokerNow session that followed had `_begin_new_hand` rebaseline
        the dead ClubGG reconstructor instead of its own — PokerNow's stayed
        on the pre-flop ante frame and emitted a phantom ante RAISE/CALL
        every hand.
        """
        if _active_reconstructor() is self._reconstructor:
            _set_active_reconstructor(None)
        self._reconstructor = None
        # Only give up the source if it is ours: a stray /ocr/stop while
        # PokerNow is driving must not make its next frame look like a
        # source switch (which resets the live hand).
        if _live_source() == "ocr":
            _note_live_source(None)

    async def _handle_window_closed(self) -> None:
        """Auto-stop path: the captured window was destroyed.

        Runs *inside* `_loop`, so it must NOT cancel its own task — it tears
        down the capture and lets `_loop` return normally. The frontend reads
        `stopped_reason == "window_closed"` (via `status()`) to reset the
        picker only on this path, not on a manual Off.
        """
        self.running = False
        self.stopped_reason = "window_closed"
        self.task = None
        await self._release_capture()
        self._hwnd = None
        self.window_match = None
        self.window_title = None
        self._retire_reconstructor()

    async def _loop(self) -> None:
        from plo5bp.ocr import live as ocr_live

        # Fixed cadence (TOOL-010): each tick is scheduled on a monotonic
        # deadline instead of sleeping a full `poll_ms` AFTER the previous
        # tick — the real period used to be poll_ms + the tick's own cost
        # (extraction, Tesseract, rebuild), long enough for ClubGG's ~300-500
        # ms bet banner to fall between two ticks. A tick that overruns its
        # slot is followed at once by the next one (no burst to catch up).
        period = self.poll_ms / 1000.0
        clock = self._clock
        next_at = clock() + period
        try:
            while self.running:
                await self._sleep(max(0.0, next_at - clock()))
                next_at = max(next_at + period, clock())
                if not self.running:
                    return
                # Auto-stop if the captured window has been destroyed. Both
                # signals are sub-microsecond, so run them directly on the
                # asyncio side (no to_thread) before any per-tick work.
                # IsWindow is authoritative (stays True on minimize); the
                # WGC is_finished() is a defensive secondary signal.
                cc = self._capture_control
                cc_finished = False
                if cc is not None:
                    try:
                        cc_finished = bool(cc.is_finished())
                    except Exception:
                        cc_finished = False
                if cc_finished or not ocr_live.is_window_alive(self._hwnd):
                    await self._handle_window_closed()
                    return
                try:
                    # The live lock: no study request mutates the session
                    # while this tick snapshots, rebuilds and commits it.
                    async with live_lock():
                        t0 = clock()
                        try:
                            await asyncio.to_thread(self._tick)
                        finally:
                            self.last_tick_ms = round((clock() - t0) * 1000.0, 1)
                except asyncio.CancelledError:
                    raise
                except BaseException as e:
                    # PyO3 panics inherit from BaseException; surface
                    # them via last_error rather than killing the loop.
                    self.last_error = f"{type(e).__name__}: {e}"
                    logger.exception("ocr tick failed")
        except asyncio.CancelledError:
            pass

    def _tick(self) -> None:
        from plo5bp.ocr.extract import extract_frame_state, fit_to_calibration

        with self._frame_lock:
            img = self.latest_frame
        # WGC may not have delivered the first frame yet on the very
        # first tick after start; skip silently and try again next poll.
        if img is None:
            return
        # Live capture only understands the PLO5 double-board table. If the
        # user flips the study format while the runner is up, idle instead of
        # mirroring a 5-card/2-board frame into an NLH-shaped card spec
        # (IndexError every tick). (review 2026-09-20 I10)
        if not _live_capture_allowed():
            self.last_error = _LIVE_FORMAT_ERROR
            return
        # `stop()` clears the reconstructor; a tick already running on its
        # worker thread must not trip over that.
        if self._reconstructor is None:
            return

        # The ROI layout is calibrated on one window shape: refuse another
        # aspect loudly, rescale another size (TOOL-003).
        fitted, note = fit_to_calibration(img)
        self.frame_note = note
        if fitted is None:
            self.last_error = note
            self.last_tick_at = time.time()
            return
        fs = extract_frame_state(
            fitted, num_seats=session.num_seats, cache=self._ocr_read_cache
        )
        self.process_frame(fs, image=fitted)

    def process_frame(self, fs: Any, image: Any = None) -> None:
        """Apply one extracted `FrameState` to the session — the whole tick
        after pixel extraction. Also the entry point of `live.replay`, which
        feeds recorded frames through exactly this code."""
        reconstructor = self._reconstructor
        if reconstructor is None:
            return
        try:
            self._process_frame(fs, reconstructor)
        finally:
            rec = record.recorder()
            if rec is not None and rec.active:
                rec.record(
                    source="ocr", frame=fs, simple=bool(live_state.simple_ocr_mode),
                    view=self._tick_view, events=self.last_events,
                    log=session.action_log, warnings=live_warnings(), image=image,
                )

    def _process_frame(self, fs: Any, reconstructor: Any) -> None:
        from plo5bp.ocr.events import (
            HeroHoleRevealed,
            OcrWarning,
            StreetReveal,
        )

        self._tick_view = None
        self.last_events = []
        self.frames_seen += 1

        # Mirror directly-observable card / button state from the frame
        # before the reconstructor runs — event callbacks can assume
        # the session card spec already reflects the latest reveal.
        _mirror_observable_state(fs)

        if not live_state.simple_ocr_mode:
            # `_begin_new_hand` invalidates the env, so on a hand-start tick
            # this view is rebuilt from the NEW hand (button, mask, stacks)
            # rather than describing the previous one. (review I5)
            try:
                engine_view = _engine_view_from_session()
            except HTTPException as e:
                self.last_error = f"rebuild failed: {e.detail}"
                logger.warning("ocr rebuild failed: %s", e.detail)
                self.last_tick_at = time.time()
                return
            self._tick_view = engine_view
            before = reconstructor.snapshot()
            events = reconstructor.step(fs, engine_view)
            self.last_events = list(events)
            # Actions are validated against the engine before they enter the
            # log; a refused one voids this step for a retry (TOOL-025).
            applied, retry = _record_seat_actions(events, reconstructor, before)
            self.events_applied += applied
            if retry:
                events = []

            for ev in events:
                if isinstance(ev, (HeroHoleRevealed, StreetReveal)):
                    # Already mirrored via _mirror_observable_state.
                    self.events_applied += 1
                elif isinstance(ev, OcrWarning):
                    logger.warning("ocr: %s", ev.message)

            street_reveals = [ev for ev in events if isinstance(ev, StreetReveal)]
            # Fold reconcile needs the seat to read folded on TWO consecutive
            # ticks (review 2026-09-20 F11): the reveal tick parks first
            # sightings, the very next tick confirms them and only then runs
            # the check fill that was blocked behind the missing fold.
            followup_target = None if retry else live_state.pending_reveal_target
            if not retry:
                live_state.pending_reveal_target = None
            if street_reveals:
                target_street = max(
                    2 if ev.street == "turn" else 3 for ev in street_reveals
                )
                _reconcile_missed_folds_on_street_reveal(fs)
                _reconcile_missed_checks_on_street_reveal(target_street)
                if live_state.pending_reveal_folds:
                    live_state.pending_reveal_target = target_street
            elif followup_target is not None:
                if _reconcile_missed_folds_on_street_reveal(fs):
                    _reconcile_missed_checks_on_street_reveal(followup_target)
                live_state.pending_reveal_folds = frozenset()

        try:
            _rebuild_env_if_stale()
            self.last_error = None
        except HTTPException as e:
            self.last_error = f"rebuild failed: {e.detail}"
            logger.warning("ocr rebuild failed: %s", e.detail)
        except Exception as e:
            self.last_error = f"rebuild failed: {type(e).__name__}: {e}"
            logger.exception("ocr rebuild failed")
        else:
            _clear_stale_refusal()
            # Pot drift alarm (TOOL-026): only while actions are tracked — in
            # simple mode the pot disagrees until the user enters each action.
            if not live_state.simple_ocr_mode:
                _check_pot_sync(fs)

        self.last_tick_at = time.time()


ocr_runner = OcrRunner()
