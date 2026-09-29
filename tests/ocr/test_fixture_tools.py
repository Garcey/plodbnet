"""`scripts/ocr_collect_fixtures.py` (TEST-027): finds every frame the pixel
tests name and copies surviving copies into the tracked fixtures folder."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ocr_collect_fixtures.py"
_spec = importlib.util.spec_from_file_location("ocr_collect_fixtures", _SCRIPT)
collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect)


def test_every_named_frame_is_wanted():
    names = collect.wanted_frames()
    # slot-0 "10" test, hero de-rotation test, labels.json
    for n in ("debug_1777231338393.png", "debug_1780904885232.png", "frame_0060.png"):
        assert n in names


def test_copies_what_survives_and_reports_the_rest(tmp_path, monkeypatch, capsys):
    src = tmp_path / "old" / "screenrecords" / "frames"
    src.mkdir(parents=True)
    (src / "frame_0060.png").write_bytes(b"png-bytes")
    dest = tmp_path / "fixtures"
    monkeypatch.setattr(collect, "DEST", dest)
    rc = collect.main([str(tmp_path / "old")])
    assert (dest / "frame_0060.png").read_bytes() == b"png-bytes"
    out = capsys.readouterr().out
    assert "copy     frame_0060.png" in out and "MISSING  frame_0300.png" in out
    assert rc == 1  # others are still missing
    # Nothing is copied twice.
    collect.main([str(tmp_path / "old")])
    assert "have     frame_0060.png" in capsys.readouterr().out
