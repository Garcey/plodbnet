"""HTTP surface of live capture (`/ocr/*`, `/pokernow/*`) and its installation
into the study app (local build only: `plo5bp.ui.server` calls `install(app)`
unless `PLO5BP_PUBLIC` is set, and the public build never imports this
package).

Concurrency: every request that reads or writes the study session — the study
routes of `plo5bp.ui.server` and the mutating live routes here — runs under
the ONE live lock the OCR tick and the PokerNow ingest also hold
(`_LiveLockMiddleware`, `tracking.live_lock`). Handlers here are therefore
plain `def`s (FastAPI runs them on a worker thread while the lock is held)
and never take the lock themselves.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from plo5bp.ui.live.clubgg import ocr_runner
from plo5bp.ui.live.pokernow import pokernow_runner
from plo5bp.ui.live.state import live_state
from plo5bp.ui.live.tracking import (
    _card_idx,
    _ocr_apply_card_slot,
    _require_live_capture_format,
    _reset_live_tracking,
    live_lock,
)
from plo5bp.ui.server import (
    _SESSION_RESET_HOOKS,
    _STATE_EXTRAS_HOOKS,
    _lock_filled_card_slots,
    _rebuild_env,
    _state_dict,
    session,
)

logger = logging.getLogger("plo5bp.ui.live")

_REPO_ROOT = Path(__file__).resolve().parents[4]
#: `/ocr/save_frame` destinations. Debug captures go to the gitignored
#: `screenrecords/frames/`; frames meant as TEST FIXTURES must land in the
#: tracked folder — a disk cleanup of screenrecords/ once orphaned every
#: labeled frame (tests/ocr/fixtures/frames/README.md).
_DEBUG_FRAMES_DIR = _REPO_ROOT / "screenrecords" / "frames"
_FIXTURE_FRAMES_DIR = _REPO_ROOT / "tests" / "ocr" / "fixtures" / "frames"


class OcrStartRequest(BaseModel):
    window_match: str = Field(..., min_length=1)
    poll_ms: int = Field(default=200, ge=50, le=5000)


class OcrSimpleRequest(BaseModel):
    enabled: bool


class OcrRescanRequest(BaseModel):
    target: Literal["hole", "board"]


class OcrSaveFrameRequest(BaseModel):
    #: True = save into the TRACKED tests/ocr/fixtures/frames/ (with the
    #: frame's extracted FrameState next to it) instead of screenrecords/.
    to_fixtures: bool = False


_RESCAN_GROUPS: dict[str, tuple[str, ...]] = {
    "hole": ("hero_hole",),
    "board": ("flop_a", "flop_b", "turn_cards", "river_cards"),
}


router = APIRouter()


# --- PokerNow -------------------------------------------------------------------
# HTTP only. The userscript POSTs through `GM_xmlhttpRequest` (a page-context
# fetch or websocket can't reach 127.0.0.1 from an https PokerNow tab). The
# websocket endpoint that also used to exist had no Origin check, so any page
# open in the same browser could inject table snapshots (SEC-024); nothing used
# it. A cross-site POST can't: a JSON body needs a CORS preflight this server
# never grants.


@router.post("/pokernow/ingest")
def pokernow_ingest_http(payload: dict) -> dict[str, Any]:
    """HTTP ingest for the Tampermonkey collector (``GM_xmlhttpRequest``)."""
    if ocr_runner.running:
        raise HTTPException(status_code=409, detail="ClubGG OCR is active")
    # PLO5-only; refuse before any session state is touched (review I10).
    _require_live_capture_format()
    # Game-lock first (before dedup) so the active game's heartbeats keep the
    # lock fresh even when they dedupe away. Foreign frames (a second open
    # PokerNow tab) are ignored so they can't interleave into the live hand.
    if not pokernow_runner.accept_game(payload.get("gameId")):
        return {"ok": True, "ignored": "other_game", "status": pokernow_runner.status()}
    pokernow_runner.note_post()
    if pokernow_runner.is_duplicate(payload):
        return {"ok": True, "deduped": True, "status": pokernow_runner.status()}
    pokernow_runner.handle_payload(payload)
    return {"ok": True, "status": pokernow_runner.status()}


@router.get("/pokernow/status")
def pokernow_status() -> dict[str, Any]:
    return pokernow_runner.status()


_USERSCRIPT = _REPO_ROOT / "tools" / "pokernow" / "pokernow.user.js"


@router.get("/pokernow/pokernow.user.js")
def pokernow_userscript() -> FileResponse:
    """The collector itself (TOOL-016). Opening this URL with Tampermonkey
    installed offers to install the script, and its @updateURL points back
    here, so Tampermonkey keeps it current from then on."""
    return FileResponse(
        _USERSCRIPT,
        media_type="text/javascript; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


# --- ClubGG OCR -------------------------------------------------------------------


@router.get("/ocr/windows")
def ocr_windows() -> dict[str, Any]:
    """List visible top-level window titles for the dropdown.

    The frontend calls this on Refresh and on first focus of the
    window-match input. Backend matching is deliberately kept simple
    (substring + exact-match override in ``find_window``); see
    ``plo5bp.ocr.live`` for the exact semantics.
    """
    try:
        from plo5bp.ocr import live as ocr_live
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"ocr deps missing: {e}") from e
    try:
        titles = ocr_live.list_window_titles()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"windows": list(titles)}


@router.post("/ocr/start")
async def ocr_start(req: OcrStartRequest) -> dict[str, Any]:
    # PLO5-only; refuse before `start` reshapes the session (review I10).
    _require_live_capture_format()
    await ocr_runner.start(req.window_match, req.poll_ms)
    return {"ok": True, "status": ocr_runner.status()}


@router.post("/ocr/stop")
async def ocr_stop() -> dict[str, Any]:
    await ocr_runner.stop()
    return {"ok": True, "status": ocr_runner.status()}


@router.post("/ocr/simple")
def ocr_simple(req: OcrSimpleRequest) -> dict[str, Any]:
    live_state.simple_ocr_mode = bool(req.enabled)
    return {"state": _state_dict()}


@router.post("/ocr/rescan")
def ocr_rescan(req: OcrRescanRequest) -> dict[str, Any]:
    """Re-OCR a single card group from the current frame; preserve hand state.

    Used in simple OCR mode when a card group was misread (capture
    landed mid-reveal animation). Clears the per-slot lock for the
    target group, applies a fresh OCR read, re-latches filled slots,
    and rebuilds the engine. `action_log`, `button_seat`, participant
    mask, observed stacks/pot all survive untouched.
    """
    # PLO5-only (review I10): the rescan groups are PLO5 card-spec slots.
    _require_live_capture_format()
    if not ocr_runner.running:
        raise HTTPException(status_code=400, detail="OCR not running")
    with ocr_runner._frame_lock:
        img = ocr_runner.latest_frame
    if img is None:
        raise HTTPException(
            status_code=400,
            detail="no frame received yet; let one WGC frame land first",
        )

    from plo5bp.ocr.extract import extract_frame_state, fit_to_calibration

    groups = _RESCAN_GROUPS[req.target]
    prev_spec = {g: list(getattr(session, g)) for g in groups}
    prev_locks = {g: list(session._card_slot_locked[g]) for g in groups}

    fitted, note = fit_to_calibration(img)
    if fitted is None:
        raise HTTPException(status_code=400, detail=f"rescan {req.target} failed: {note}")
    try:
        fs = extract_frame_state(fitted, num_seats=session.num_seats)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"rescan {req.target} failed: extract: {e}",
        ) from e

    for g in groups:
        session._card_slot_locked[g] = [False] * len(prev_locks[g])
        live_state.card_slot_pending[g] = [None] * len(prev_locks[g])

    if "hero_hole" in groups:
        for i, c in enumerate(fs.hero_hole):
            _ocr_apply_card_slot("hero_hole", i, _card_idx(c))
    if "flop_a" in groups:
        for i, c in enumerate(fs.board_a[:3]):
            _ocr_apply_card_slot("flop_a", i, _card_idx(c))
        for i, c in enumerate(fs.board_b[:3]):
            _ocr_apply_card_slot("flop_b", i, _card_idx(c))
        _ocr_apply_card_slot("turn_cards", 0, _card_idx(fs.board_a[3]))
        _ocr_apply_card_slot("turn_cards", 1, _card_idx(fs.board_b[3]))
        _ocr_apply_card_slot("river_cards", 0, _card_idx(fs.board_a[4]))
        _ocr_apply_card_slot("river_cards", 1, _card_idx(fs.board_b[4]))

    _lock_filled_card_slots()

    try:
        _rebuild_env()
    except Exception as e:
        for g in groups:
            setattr(session, g, prev_spec[g])
            session._card_slot_locked[g] = prev_locks[g]
        try:
            _rebuild_env()
        except Exception:
            logger.exception("rescan rollback rebuild also failed")
        detail = e.detail if isinstance(e, HTTPException) else str(e)
        raise HTTPException(
            status_code=400, detail=f"rescan {req.target} failed: {detail}"
        )
    return {"state": _state_dict()}


@router.get("/ocr/status")
def ocr_status() -> dict[str, Any]:
    return ocr_runner.status()


@router.post("/ocr/save_frame")
def ocr_save_frame(req: OcrSaveFrameRequest | None = None) -> dict[str, Any]:
    """Write the most recent WGC frame to disk (no body needed).

    Default: a debug capture in the gitignored ``screenrecords/frames/``.
    ``{"to_fixtures": true}``: the frame goes to the TRACKED
    ``tests/ocr/fixtures/frames/`` together with ``<name>.state.json`` —
    the `FrameState` the extractor reads from it right now — ready to be
    labeled (``plo5bp.ocr.tools.label_cards``) and to pin the extractor
    against (``tests/ocr/test_frame_fixtures.py``). Returns 400 until WGC
    has delivered a frame.
    """
    try:
        import cv2
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"ocr deps missing: {e}") from e

    to_fixtures = bool(req is not None and req.to_fixtures)
    with ocr_runner._frame_lock:
        img = ocr_runner.latest_frame
    if img is None:
        raise HTTPException(
            status_code=400,
            detail="no frame received yet; start OCR and let one WGC frame land first",
        )

    if to_fixtures:
        # Fixtures are kept at the calibration geometry — what the extractor
        # really reads (a same-aspect window of another size is rescaled).
        from plo5bp.ocr.extract import fit_to_calibration

        fitted, note = fit_to_calibration(img)
        if fitted is None:
            raise HTTPException(status_code=400, detail=note)
        img = fitted
    out_dir = _FIXTURE_FRAMES_DIR if to_fixtures else _DEBUG_FRAMES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_ms = int(time.time() * 1000)
    stem = f"frame_{ts_ms}" if to_fixtures else f"debug_{ts_ms}"
    path = out_dir / f"{stem}.png"
    if not cv2.imwrite(str(path), img):
        raise HTTPException(status_code=500, detail=f"cv2.imwrite failed for {path}")
    h, w = img.shape[:2]
    out: dict[str, Any] = {
        "path": str(path),
        "tracked": to_fixtures,
        "frame_size": {"width": int(w), "height": int(h)},
    }
    if to_fixtures:
        try:
            import json

            from plo5bp.ocr.extract import extract_frame_state

            fs = extract_frame_state(img, num_seats=session.num_seats)
            state_path = out_dir / f"{stem}.state.json"
            state_path.write_text(
                json.dumps(fs.to_dict(), indent=2) + "\n", encoding="utf-8"
            )
            out["state_path"] = str(state_path)
        except Exception as e:  # noqa: BLE001 — the PNG is still useful
            out["state_error"] = f"{type(e).__name__}: {e}"
    else:
        out["note"] = (
            "screenrecords/ is gitignored and local-only; pass "
            '{"to_fixtures": true} to keep a frame as a test fixture'
        )
    return out


# --- Installation into the study app ------------------------------------------

#: Live routes that mutate the study session (serialized with the tick).
_LOCKED_LIVE_PATHS = frozenset(
    {"/ocr/start", "/ocr/stop", "/ocr/simple", "/ocr/rescan", "/pokernow/ingest"}
)


class _LiveLockMiddleware:
    """Run every study-session request under the live lock (TOOL-004).

    Locked: every route whose endpoint is defined in ``plo5bp.ui.server``
    (the study routes: /state, /action, /undo, /cards, /seats, /config,
    /reset, /format ... — found by module, so a study route added later is
    covered without a list to maintain) and the mutating live routes. Not
    locked: status polls, static files, the trainer, ranges. The OCR tick
    and PokerNow payloads hold the same lock, so a click can no longer land
    between a tick's snapshot of the action log and its commit.
    """

    def __init__(self, app: Any, core_app: FastAPI) -> None:
        self.app = app
        self._core_app = core_app
        self._paths: frozenset[str] | None = None

    def _locked_paths(self) -> frozenset[str]:
        # Computed on the first request: routes registered after `install`
        # (index, static pages) must be seen too.
        if self._paths is None:
            paths = set(_LOCKED_LIVE_PATHS)
            for route in self._core_app.router.routes:
                endpoint = getattr(route, "endpoint", None)
                if getattr(endpoint, "__module__", None) == "plo5bp.ui.server":
                    paths.add(str(route.path))
            self._paths = frozenset(paths)
        return self._paths

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http" and scope["path"] in self._locked_paths():
            async with live_lock():
                await self.app(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _on_session_reset(kind: str) -> None:
    """Study-core reset hook (``server._SESSION_RESET_HOOKS``).

    "hand": the core just restored its per-hand defaults — the live per-hand
    state follows (the cross-hand debounce survives, see ``LiveState``).
    "user": /reset or /format — forget everything the tracker learned.
    """
    if kind == "hand":
        st = session.live
        if st is not None:
            st.reset_hand()
    elif kind == "user":
        _reset_live_tracking()


def _state_extras() -> dict[str, Any]:
    """Live keys of the study state projection (``server._STATE_EXTRAS_HOOKS``)."""
    return {"simple_ocr_mode": bool(live_state.simple_ocr_mode)}


def install(app: FastAPI) -> None:
    """Mount live capture on the study app and hook it into the study core.

    Called by ``plo5bp.ui.server`` for the LOCAL build only; idempotent.
    """
    if getattr(app.state, "live_capture_installed", False):
        return
    app.state.live_capture_installed = True
    app.include_router(router)
    app.add_middleware(_LiveLockMiddleware, core_app=app)
    _SESSION_RESET_HOOKS.append(_on_session_reset)
    _STATE_EXTRAS_HOOKS.append(_state_extras)
