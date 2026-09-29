"""Apply an axis-decoupled affine remap to every fractional value in
``python/plo5bp/ocr/rois.py``.

Two anchors visible in OLD (WMP-rendered) and NEW (ClubGG-direct)
captures pin down per-axis ``scale + offset`` constants. We then
rewrite every fractional position/size in ``rois.py``:

  new_x = old_x * scale_x + offset_x      (positions)
  new_w = old_w * scale_x                 (widths/strides)
  new_y = old_y * scale_y + offset_y      (positions)
  new_h = old_h * scale_y                 (heights/strides)

Anchors:
  A = top-left corner of seat 0's name_plate
  B = top-right corner of seat 3's name_plate

A_OLD/B_OLD are pulled from the current rois.py and assumed correct
(the existing fixtures pass). A_NEW/B_NEW are CLI args, measured
from a ClubGG-direct ``debug_<ts>.png`` produced by ``/ocr/save_frame``.

Usage::

    .venv/Scripts/python scripts/remap_rois.py \\
        --a-new 0.42 0.95 \\
        --b-new 0.58 0.05         # dry-run, prints diff

    .venv/Scripts/python scripts/remap_rois.py \\
        --a-new 0.42 0.95 \\
        --b-new 0.58 0.05 --apply # actually rewrite rois.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROIS_PATH = REPO_ROOT / "python" / "plo5bp" / "ocr" / "rois.py"

# Anchors in the OLD layout, read straight from rois.py.
A_OLD: tuple[float, float] = (0.435, 0.860)  # _SEATS_6[0].name_plate.{x, y}
B_OLD: tuple[float, float] = (0.565, 0.200)  # _SEATS_6[3].name_plate.{x + w, y}

# Module-level fractional constants. Positions get scale + offset; sizes
# (widths, heights, strides) get scale only. Names that aren't in either
# set are left alone (e.g. NUM_BOARD_CARDS is an integer count).
X_POSITIONS = {"_BOARD_LEFT", "_HERO_CARDS_LEFT"}
X_SIZES = {"_BOARD_CARD_W", "_BOARD_CARD_STEP", "_HERO_CARD_W", "_HERO_CARD_STEP"}
Y_POSITIONS = {"_BOARD_A_TOP", "_BOARD_B_TOP", "_HERO_TOP"}
Y_SIZES = {"_BOARD_CARD_H", "_HERO_CARD_H"}


def fmt(v: float) -> str:
    return f"{v:.4f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--a-new",
        nargs=2,
        type=float,
        required=True,
        metavar=("X", "Y"),
        help="Anchor A in NEW frame (fractional x y).",
    )
    p.add_argument(
        "--b-new",
        nargs=2,
        type=float,
        required=True,
        metavar=("X", "Y"),
        help="Anchor B in NEW frame (fractional x y).",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite rois.py. Default is dry-run (print diff only).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    a_new = (float(args.a_new[0]), float(args.a_new[1]))
    b_new = (float(args.b_new[0]), float(args.b_new[1]))

    dx_old = B_OLD[0] - A_OLD[0]
    dy_old = B_OLD[1] - A_OLD[1]
    if abs(dx_old) < 1e-6 or abs(dy_old) < 1e-6:
        print("ERROR: A_OLD and B_OLD must differ in both axes", file=sys.stderr)
        return 2

    scale_x = (b_new[0] - a_new[0]) / dx_old
    offset_x = a_new[0] - A_OLD[0] * scale_x
    scale_y = (b_new[1] - a_new[1]) / dy_old
    offset_y = a_new[1] - A_OLD[1] * scale_y

    print("# Affine transform")
    print(f"  A_old={A_OLD}  A_new={a_new}")
    print(f"  B_old={B_OLD}  B_new={b_new}")
    print(f"  scale_x={scale_x:+.4f}  offset_x={offset_x:+.4f}")
    print(f"  scale_y={scale_y:+.4f}  offset_y={offset_y:+.4f}")
    print()

    def remap_x_pos(v: float) -> float:
        return v * scale_x + offset_x

    def remap_x_size(v: float) -> float:
        return v * scale_x

    def remap_y_pos(v: float) -> float:
        return v * scale_y + offset_y

    def remap_y_size(v: float) -> float:
        return v * scale_y

    text = ROIS_PATH.read_text(encoding="utf-8")
    changes: list[tuple[str, str]] = []

    def repl_const(m: "re.Match[str]") -> str:
        name = m.group(1)
        val_str = m.group(2)
        val = float(val_str)
        if name in X_POSITIONS:
            new = remap_x_pos(val)
        elif name in X_SIZES:
            new = remap_x_size(val)
        elif name in Y_POSITIONS:
            new = remap_y_pos(val)
        elif name in Y_SIZES:
            new = remap_y_size(val)
        else:
            return m.group(0)
        new_str = fmt(new)
        changes.append((f"{name} = {val_str}", f"{name} = {new_str}"))
        return f"{name} = {new_str}"

    # Module-level fractional constants. Anchored to the start of a line
    # (no leading whitespace) since rois.py declares them at module scope.
    text = re.sub(
        r"^(_[A-Z_]+)\s*=\s*([-+]?\d+\.\d+)\s*$",
        repl_const,
        text,
        flags=re.MULTILINE,
    )

    def repl_roi(m: "re.Match[str]") -> str:
        body = m.group(1)
        kw = dict(re.findall(r"(\w+)\s*=\s*([-+]?\d+\.?\d*)", body))
        keyword_form = {"x", "y", "w", "h"} <= set(kw)
        if keyword_form:
            x, y, w, h = (float(kw[k]) for k in ("x", "y", "w", "h"))
            old_repr = f"ROI(x={kw['x']}, y={kw['y']}, w={kw['w']}, h={kw['h']})"
        else:
            nums = re.findall(r"[-+]?\d+\.?\d*", body)
            if len(nums) != 4:
                return m.group(0)
            x, y, w, h = (float(n) for n in nums)
            old_repr = f"ROI({nums[0]}, {nums[1]}, {nums[2]}, {nums[3]})"
        nx = remap_x_pos(x)
        ny = remap_y_pos(y)
        nw = remap_x_size(w)
        nh = remap_y_size(h)
        if keyword_form:
            new_repr = f"ROI(x={fmt(nx)}, y={fmt(ny)}, w={fmt(nw)}, h={fmt(nh)})"
        else:
            new_repr = f"ROI({fmt(nx)}, {fmt(ny)}, {fmt(nw)}, {fmt(nh)})"
        changes.append((old_repr, new_repr))
        return new_repr

    text = re.sub(r"ROI\(([^)]*)\)", repl_roi, text)

    print(f"# Changes: {len(changes)}")
    for old, new in changes:
        print(f"  {old}")
        print(f"    -> {new}")
    print()

    if args.apply:
        ROIS_PATH.write_text(text, encoding="utf-8")
        print(f"wrote {ROIS_PATH}")
    else:
        print("(dry-run; pass --apply to rewrite rois.py)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
