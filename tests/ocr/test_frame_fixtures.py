"""Golden frames: whole ClubGG frames with their reviewed `FrameState` (TEST-027).

Every ``tests/ocr/fixtures/frames/<name>.png`` that has a
``<name>.state.json`` next to it is run through `extract_frame_state` and must
read back exactly that state — cards, stacks, commits, button, pot, who is in
the hand and who holds the timer. ``POST /ocr/save_frame {"to_fixtures":
true}`` writes both files from a live table; review the JSON against the
screenshot (it is the expected answer) and commit them. Skipped without
OpenCV / rank templates, or while no golden frame exists yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from plo5bp.ocr import cards as card_mod  # noqa: E402
from plo5bp.ocr.extract import extract_frame_state  # noqa: E402
from plo5bp.ocr.types import FrameState  # noqa: E402

FRAMES = Path(__file__).parent / "fixtures" / "frames"
GOLDEN = sorted(
    p for p in FRAMES.glob("*.state.json")
    if p.with_name(p.name[: -len(".state.json")] + ".png").exists()
)


def _diff(got: dict, want: dict, path: str = "") -> list[str]:
    if isinstance(want, dict) and isinstance(got, dict):
        out: list[str] = []
        for k in sorted(set(want) | set(got)):
            out += _diff(got.get(k), want.get(k), f"{path}.{k}")
        return out
    if isinstance(want, list) and isinstance(got, list) and len(want) == len(got):
        out = []
        for i, (g, w) in enumerate(zip(got, want)):
            out += _diff(g, w, f"{path}[{i}]")
        return out
    return [] if got == want else [f"{path}: read {got!r}, expected {want!r}"]


@pytest.mark.parametrize("state_path", GOLDEN or [None],
                         ids=[p.name for p in GOLDEN] or ["none"])
def test_golden_frame_reads_as_reviewed(state_path):
    if state_path is None:
        pytest.skip("no golden frames in tests/ocr/fixtures/frames/ yet")
    if not card_mod._load_templates():
        pytest.skip("rank templates not available on this machine")
    png = state_path.with_name(state_path.name[: -len(".state.json")] + ".png")
    img = cv2.imread(str(png))
    assert img is not None, f"could not read {png}"
    want = json.loads(state_path.read_text(encoding="utf-8"))
    got = extract_frame_state(img, num_seats=len(want.get("seats") or []) or 6)
    problems = _diff(got.to_dict(), FrameState.from_dict(want).to_dict())
    assert not problems, f"{png.name}:\n  " + "\n  ".join(problems)
