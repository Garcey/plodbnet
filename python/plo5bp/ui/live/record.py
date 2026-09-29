"""Record live sessions for offline replay (TEST-028).

Bugs like hero's silent CHECK only show up at a live table, and a live table
can't be replayed — unless every tick's INPUT is kept. With
``PLO5BP_LIVE_RECORD=<dir>`` set, each capture session writes one JSONL file
into ``<dir>``: a header (the session's game config) and then one line per
tick / payload holding

* the input: the extracted ``FrameState`` (ClubGG) or the raw ``pokernow.v1``
  payload (PokerNow), plus the simple-mode flag;
* what the pipeline did with it: the ``EngineView`` the walk saw, the events
  it emitted, the action log after the tick and any sync warnings;
* for ClubGG, the frame PNG next to the file whenever the tick warned (a
  reconstructor ``OcrWarning`` or a sync warning), for eyeballing.

``python -m plo5bp.ui.live.replay FILE`` feeds a recording back through the
same pipeline and reports the first tick where the result differs;
``tests/ocr/test_live_replay.py`` does that for every recording kept under
``tests/ocr/fixtures/sessions/`` (drop a recording there to pin a bug).

A FrameState is tiny (~2 KB of JSON), so a whole evening of 5 Hz ticks is a
few tens of MB. Recording never raises into the capture loop.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

# The SAME study core the runners bind to (a module-level import, like every
# live module: a lazy import would find whatever server module is in
# sys.modules at call time — after a test re-imported the UI, another one).
from plo5bp.ui.server import session

logger = logging.getLogger("plo5bp.ui.live")

RECORD_ENV = "PLO5BP_LIVE_RECORD"
SCHEMA = "plo5bp.live-record.v1"


def event_dict(ev: Any) -> dict[str, Any]:
    """A reconstructor event as JSON: its class name + its fields."""
    return {"type": type(ev).__name__, **dataclasses.asdict(ev)}


def session_header() -> dict[str, Any]:
    """The session config a replay must start from."""
    return {
        "variant": session.variant,
        "num_seats": int(session.num_seats),
        "button_seat": int(session.button_seat),
        "hero_seat": int(session.hero_seat),
        "dollars_per_bb": float(session.dollars_per_bb),
        "game_config": dataclasses.asdict(session.game_config),
    }


class LiveRecorder:
    """Appends ticks to one JSONL file per capture session (thread-safe)."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.path: Path | None = None
        self._fh: Any = None
        self._n = 0
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._fh is not None

    def open(self, source: str) -> Path | None:
        """Start a new recording file for a capture session."""
        with self._lock:
            self._close_locked()
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                self.path = self.directory / f"live_{source}_{stamp}.jsonl"
                self._fh = open(self.path, "a", encoding="utf-8")
                self._n = 0
                self._write_locked({
                    "schema": SCHEMA, "kind": "header", "source": source,
                    "t": time.time(), "session": session_header(),
                })
            except Exception:  # noqa: BLE001 — never break capture
                logger.exception("live record: cannot open a recording")
                self._close_locked()
                return None
        logger.info("live record: writing %s", self.path)
        return self.path

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def record(
        self,
        *,
        source: str,
        frame: Any = None,
        payload: dict | None = None,
        simple: bool | None = None,
        view: Any = None,
        events: list[Any] = (),
        log: list[dict] = (),
        warnings: list[str] = (),
        image: Any = None,
    ) -> None:
        """Append one tick (no-op unless a recording is open)."""
        with self._lock:
            if self._fh is None:
                return
            try:
                line: dict[str, Any] = {
                    "kind": "tick", "i": self._n, "t": time.time(),
                    "source": source,
                    "frame": frame.to_dict() if frame is not None else None,
                    "payload": payload,
                    "simple": simple,
                    "view": dataclasses.asdict(view) if view is not None else None,
                    "events": [event_dict(e) for e in events],
                    "log": [dict(e) for e in log],
                    "warnings": list(warnings),
                }
                warned = bool(warnings) or any(
                    type(e).__name__ == "OcrWarning" for e in events
                )
                if image is not None and warned and self.path is not None:
                    png = self.path.with_name(f"{self.path.stem}_{self._n:06d}.png")
                    try:
                        import cv2

                        if cv2.imwrite(str(png), image):
                            line["image"] = png.name
                    except Exception:  # noqa: BLE001 — the line matters more
                        pass
                self._write_locked(line)
                self._n += 1
            except Exception:  # noqa: BLE001
                logger.exception("live record: tick not recorded")

    def _write_locked(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self._fh.flush()

    def _close_locked(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass
        self._fh = None


#: The process-wide recorder (None unless PLO5BP_LIVE_RECORD is set; the
#: replayer and tests swap it).
RECORDER: LiveRecorder | None = (
    LiveRecorder(os.environ[RECORD_ENV].strip())
    if os.environ.get(RECORD_ENV, "").strip()
    else None
)


def recorder() -> LiveRecorder | None:
    return RECORDER
