"""Copy every ClubGG frame the OCR tests name into the tracked fixtures folder.

The pixel tests (card classifier, hero de-rotation, slot-0 "10", labeled
accuracy) name their frames — `debug_<epoch>.png` / `frame_<n>.png` captures
that only ever lived in the gitignored `screenrecords/frames/` and were lost
in a disk cleanup (TEST-027). Point this at wherever copies survive (an old
checkout, another PC, a backup) and they land in `tests/ocr/fixtures/frames/`,
which git tracks, so they can't vanish again:

    .venv/Scripts/python scripts/ocr_collect_fixtures.py D:/old/plodbnet/screenrecords/frames

The names are found by scanning `tests/ocr/*.py` and `fixtures/labels.json`,
so a test that names a new frame is picked up automatically. Prints what was
copied, what was already there and what is still missing; copies nothing
twice. `--dry-run` only reports.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests" / "ocr"
DEST = TESTS / "fixtures" / "frames"
_NAME = re.compile(r"\b((?:debug|frame)_\d+\.png)\b")


def wanted_frames() -> list[str]:
    names: set[str] = set()
    for py in TESTS.glob("*.py"):
        names.update(_NAME.findall(py.read_text(encoding="utf-8")))
    labels = TESTS / "fixtures" / "labels.json"
    if labels.exists():
        for fx in json.loads(labels.read_text(encoding="utf-8")).get("fixtures", []):
            names.add(Path(fx["frame"]).name)
    return sorted(names)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", nargs="+", help="folder(s) to search, recursively")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    found: dict[str, Path] = {}
    for src in map(Path, args.source):
        for p in src.rglob("*.png"):
            found.setdefault(p.name, p)
    DEST.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in wanted_frames():
        dest = DEST / name
        if dest.exists():
            print(f"  have     {name}")
        elif name in found:
            print(f"  copy     {name}  <- {found[name]}")
            if not args.dry_run:
                shutil.copy2(found[name], dest)
        else:
            missing.append(name)
            print(f"  MISSING  {name}")
    print(f"{len(missing)} still missing" if missing else "all named frames present")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
