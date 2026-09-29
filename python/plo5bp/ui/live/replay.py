"""Replay a live-session recording through the live pipeline (TEST-028).

    .venv/Scripts/python -m plo5bp.ui.live.replay recordings/live_ocr_20261001-2130.jsonl

Resets the study session to the recording's config, feeds every recorded
input (ClubGG `FrameState`s through `OcrRunner.process_frame`, PokerNow
payloads through `PokerNowRunner.handle_payload` — the code the live runners
use) and compares each tick's events and action log with what was recorded.
Prints one line per tick where something happened and the first divergence;
exit status 1 when the replay differs (a behaviour change since recording).

Recordings come from ``PLO5BP_LIVE_RECORD=<dir>`` (see `record.py`). Keep one
that shows a bug under ``tests/ocr/fixtures/sessions/`` and
``tests/ocr/test_live_replay.py`` replays it on every test run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from plo5bp.ui.live import record
from plo5bp.ui.live.clubgg import OcrRunner
from plo5bp.ui.live.pokernow import PokerNowRunner
from plo5bp.ui.live.state import live_state
from plo5bp.ui.live.tracking import (
    _reset_live_tracking,
    _set_active_reconstructor,
    live_warnings,
)
from plo5bp.ui.server import _new_session_defaults, _rebuild_env, session


def load(path: str | Path) -> tuple[dict, list[dict]]:
    """(header, ticks) of a recording; ValueError if it isn't one."""
    header: dict | None = None
    ticks: list[dict] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("kind") == "header":
            if obj.get("schema") != record.SCHEMA:
                raise ValueError(f"{path}:{n}: unknown schema {obj.get('schema')!r}")
            header = obj
        elif obj.get("kind") == "tick":
            ticks.append(obj)
    if header is None:
        raise ValueError(f"{path}: not a live recording (no header line)")
    return header, ticks


def _restore_session(header: dict) -> None:
    from plo5bp.config import GameConfig

    s = header["session"]
    cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in s["game_config"].items()}
    session.variant = s["variant"]
    session.game_config = GameConfig(**cfg)
    session.dollars_per_bb = float(s["dollars_per_bb"])
    session.num_seats = int(s["num_seats"])
    session.button_seat = int(s["button_seat"])
    session.hero_seat = int(s["hero_seat"])
    _new_session_defaults()
    _reset_live_tracking()
    _rebuild_env()


def _normal(obj: Any) -> Any:
    """JSON round trip, so tuples and lists compare equal."""
    return json.loads(json.dumps(obj))


def _result(events: list[Any]) -> dict:
    return _normal({
        "events": [record.event_dict(e) for e in events],
        "log": [dict(e) for e in session.action_log],
        "warnings": live_warnings(),
    })


def replay(path: str | Path) -> list[dict]:
    """Feed a recording through a fresh live pipeline on the CURRENT study
    session (reset to the recorded config first). Returns one
    ``{"events", "log", "warnings"}`` per recorded tick."""
    from plo5bp.ocr.events import EventReconstructor
    from plo5bp.ocr.types import FrameState

    header, ticks = load(path)
    saved = record.RECORDER
    record.RECORDER = None  # a replay is never recorded
    out: list[dict] = []
    try:
        _restore_session(header)
        if header["source"] == "ocr":
            runner: Any = OcrRunner()
            runner._reconstructor = EventReconstructor(num_seats=session.num_seats)
            _set_active_reconstructor(runner._reconstructor)
            for tick in ticks:
                live_state.simple_ocr_mode = bool(tick.get("simple", True))
                runner.process_frame(FrameState.from_dict(tick["frame"]))
                out.append(_result(runner.last_events))
        else:
            runner = PokerNowRunner()
            for tick in ticks:
                try:
                    runner.handle_payload(tick["payload"])
                except HTTPException:
                    pass  # a malformed payload: the live route answered 400 too
                out.append(_result(runner.last_events))
    finally:
        record.RECORDER = saved
        _set_active_reconstructor(None)
    return out


def compare(path: str | Path) -> tuple[int | None, list[dict], list[dict]]:
    """Replay ``path``; returns (first tick whose events or log differ from
    the recording — None when identical, recorded, replayed)."""
    _, ticks = load(path)
    recorded = [_normal({"events": t["events"], "log": t["log"]}) for t in ticks]
    replayed = replay(path)
    for i, (a, b) in enumerate(zip(recorded, replayed)):
        if a["events"] != b["events"] or a["log"] != b["log"]:
            return i, recorded, replayed
    return None, recorded, replayed


def _describe(res: dict) -> str:
    evs = ", ".join(
        e["type"] + (f"(seat {e['seat']} {e['gate']} {e['chips']})" if e["type"] == "SeatAction" else "")
        for e in res["events"]
    )
    return f"events [{evs}] log {len(res['log'])}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recording", help="a live_*.jsonl file from PLO5BP_LIVE_RECORD")
    args = ap.parse_args(argv)
    first, recorded, replayed = compare(args.recording)
    for i, (a, b) in enumerate(zip(recorded, replayed)):
        if a["events"] or b["events"] or i == first:
            mark = "  " if a == {"events": b["events"], "log": b["log"]} else "!!"
            print(f"{mark} tick {i:5d}  recorded {_describe(a)}")
            if mark == "!!":
                print(f"   {'':10s}  replayed {_describe(b)}")
    if first is None:
        print(f"OK: {len(replayed)} ticks replay identically")
        return 0
    print(f"DIFFERS from tick {first} on (the pipeline changed since this was recorded)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
