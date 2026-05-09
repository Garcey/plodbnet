"""Apply a 12-anchor least-squares affine remap to ``rois.py.bak.wmp``,
then override the 12 measured nameplate+stack ROIs with the exact
fractional values measured against the native 1927x1391 ClubGG frame.

Why 12 anchors instead of 2: the 2-anchor version (``remap_rois.py``)
fits scale+offset to the corners of seat 0 / seat 3 plates only. With
all 6 plates and all 6 stacks, we get a more robust per-axis fit and
can detect that hero's plate is rendered larger than non-hero plates
(so position-affine size scaling alone misses the hero/non-hero size
asymmetry).

Per-axis affine (positions): least-squares slope+intercept fit on
the 12 (old_center, new_center) pairs.

Sizes: position-axis scale is applied to all sizes inferred from the
WMP backup. The 12 anchors are then overwritten by the exact measured
values, so size-scale only matters for non-anchor ROIs (committed,
button, cards_back, board, hero_hole, pot).

Run::

    .venv/Scripts/python scripts/remap_rois_lsq.py             # dry-run
    .venv/Scripts/python scripts/remap_rois_lsq.py --apply     # write rois.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROIS_PATH = REPO_ROOT / "python" / "plo5bp" / "ocr" / "rois.py"
BACKUP_PATH = REPO_ROOT / "python" / "plo5bp" / "ocr" / "rois.py.bak.wmp"

# Native ClubGG capture dimensions (post-rescale-removal in live.py).
W, H = 1927, 1391


def _frac(px: int, py: int, pw: int, ph: int) -> tuple[float, float, float, float]:
    return (px / W, py / H, pw / W, ph / H)


# Measured nameplate ROIs per seat (rois.py seat index 0..5).
# User seat numbering: 1=hero(bottom-center), 2=BL, 3=TL, 4=TC, 5=TR, 6=BR
# rois.py numbering:    0=hero,               1=BL, 2=TL, 3=TC, 4=TR, 5=BR
MEASURED_PLATE = [
    _frac(834, 1194, 259, 47),    # seat 0 (hero)
    _frac(190, 1052, 206, 38),    # seat 1 (BL)
    _frac(190,  415, 206, 38),    # seat 2 (TL)
    _frac(861,  273, 206, 38),    # seat 3 (TC)
    _frac(1532, 415, 206, 38),    # seat 4 (TR)
    _frac(1532, 1052, 206, 38),   # seat 5 (BR)
]

MEASURED_STACK = [
    _frac(850, 1248, 227, 51),    # seat 0
    _frac(204, 1095, 183, 38),    # seat 1
    _frac(204,  460, 183, 38),    # seat 2
    _frac(880,  316, 183, 38),    # seat 3
    _frac(1550, 460, 183, 38),    # seat 4
    _frac(1550, 1095, 183, 38),   # seat 5
]

# WMP-backup nameplate centers (computed from rois.py.bak.wmp; verified once).
# (x_center, y_center) for plate then stack, per seat 0..5.
OLD_PLATE_CENTERS = [
    (0.435 + 0.130 / 2, 0.860 + 0.035 / 2),  # 0
    (0.158 + 0.130 / 2, 0.755 + 0.035 / 2),  # 1
    (0.158 + 0.130 / 2, 0.300 + 0.035 / 2),  # 2
    (0.435 + 0.130 / 2, 0.200 + 0.035 / 2),  # 3
    (0.712 + 0.130 / 2, 0.295 + 0.035 / 2),  # 4
    (0.712 + 0.130 / 2, 0.755 + 0.035 / 2),  # 5
]
OLD_STACK_CENTERS = [
    (0.435 + 0.130 / 2, 0.895 + 0.035 / 2),  # 0
    (0.158 + 0.130 / 2, 0.790 + 0.035 / 2),  # 1
    (0.158 + 0.130 / 2, 0.335 + 0.035 / 2),  # 2
    (0.435 + 0.130 / 2, 0.235 + 0.035 / 2),  # 3
    (0.712 + 0.130 / 2, 0.330 + 0.035 / 2),  # 4
    (0.712 + 0.130 / 2, 0.790 + 0.035 / 2),  # 5
]


def _new_center(roi_frac: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = roi_frac
    return (x + w / 2, y + h / 2)


def _lsq_fit(
    pairs: list[tuple[float, float]],
) -> tuple[float, float]:
    """Fit ``new = scale * old + offset`` via OLS on ``(old, new)`` pairs."""
    n = len(pairs)
    mean_old = sum(p[0] for p in pairs) / n
    mean_new = sum(p[1] for p in pairs) / n
    num = sum((p[0] - mean_old) * (p[1] - mean_new) for p in pairs)
    den = sum((p[0] - mean_old) ** 2 for p in pairs)
    if den < 1e-12:
        raise ValueError("singular fit: all old values identical")
    scale = num / den
    offset = mean_new - scale * mean_old
    return scale, offset


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Write rois.py. Default is dry-run (print fit + diff only).",
    )
    args = p.parse_args()

    if not BACKUP_PATH.exists():
        print(f"ERROR: backup not found at {BACKUP_PATH}", file=sys.stderr)
        return 2

    # Build (old, new) pairs for X and Y centers.
    new_plate_centers = [_new_center(r) for r in MEASURED_PLATE]
    new_stack_centers = [_new_center(r) for r in MEASURED_STACK]

    x_pairs: list[tuple[float, float]] = []
    y_pairs: list[tuple[float, float]] = []
    for old, new in zip(OLD_PLATE_CENTERS, new_plate_centers):
        x_pairs.append((old[0], new[0]))
        y_pairs.append((old[1], new[1]))
    for old, new in zip(OLD_STACK_CENTERS, new_stack_centers):
        x_pairs.append((old[0], new[0]))
        y_pairs.append((old[1], new[1]))

    scale_x, offset_x = _lsq_fit(x_pairs)
    scale_y, offset_y = _lsq_fit(y_pairs)

    print("# 12-anchor LSQ affine fit")
    print(f"  scale_x={scale_x:+.4f}  offset_x={offset_x:+.4f}")
    print(f"  scale_y={scale_y:+.4f}  offset_y={offset_y:+.4f}")
    # Residuals as a sanity check.
    res_x = [(o, n, scale_x * o + offset_x - n) for (o, n) in x_pairs]
    res_y = [(o, n, scale_y * o + offset_y - n) for (o, n) in y_pairs]
    max_rx = max(abs(r) for _, _, r in res_x)
    max_ry = max(abs(r) for _, _, r in res_y)
    print(f"  max |residual| x={max_rx:.4f} y={max_ry:.4f}")
    print()

    text = BACKUP_PATH.read_text(encoding="utf-8")

    # ---- Pass 1: rewrite module-level fractional constants. -----------------
    X_POSITIONS = {"_BOARD_LEFT", "_HERO_CARDS_LEFT"}
    X_SIZES = {"_BOARD_CARD_W", "_BOARD_CARD_STEP", "_HERO_CARD_W", "_HERO_CARD_STEP"}
    Y_POSITIONS = {"_BOARD_A_TOP", "_BOARD_B_TOP", "_HERO_TOP"}
    Y_SIZES = {"_BOARD_CARD_H", "_HERO_CARD_H"}

    changes: list[tuple[str, str]] = []

    def repl_const(m: "re.Match[str]") -> str:
        name = m.group(1)
        val_str = m.group(2)
        val = float(val_str)
        if name in X_POSITIONS:
            new = scale_x * val + offset_x
        elif name in X_SIZES:
            new = scale_x * val
        elif name in Y_POSITIONS:
            new = scale_y * val + offset_y
        elif name in Y_SIZES:
            new = scale_y * val
        else:
            return m.group(0)
        new_str = _fmt(new)
        changes.append((f"{name} = {val_str}", f"{name} = {new_str}"))
        return f"{name} = {new_str}"

    text = re.sub(
        r"^(_[A-Z_]+)\s*=\s*([-+]?\d+\.\d+)\s*$",
        repl_const,
        text,
        flags=re.MULTILINE,
    )

    # ---- Pass 2: rewrite ROI(...) calls. ------------------------------------
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
        nx = scale_x * x + offset_x
        ny = scale_y * y + offset_y
        nw = scale_x * w
        nh = scale_y * h
        if keyword_form:
            new_repr = f"ROI(x={_fmt(nx)}, y={_fmt(ny)}, w={_fmt(nw)}, h={_fmt(nh)})"
        else:
            new_repr = f"ROI({_fmt(nx)}, {_fmt(ny)}, {_fmt(nw)}, {_fmt(nh)})"
        changes.append((old_repr, new_repr))
        return new_repr

    text = re.sub(r"ROI\(([^)]*)\)", repl_roi, text)

    # ---- Pass 3: override the 12 measured plates+stacks. --------------------
    # Match each SeatROIs(seat=N, ...) block and substitute name_plate / stack_label.
    def measured_roi_str(roi: tuple[float, float, float, float]) -> str:
        x, y, w, h = roi
        return f"ROI({_fmt(x)}, {_fmt(y)}, {_fmt(w)}, {_fmt(h)})"

    overrides_applied = 0
    for seat in range(6):
        plate_str = measured_roi_str(MEASURED_PLATE[seat])
        stack_str = measured_roi_str(MEASURED_STACK[seat])

        # Replace name_plate=... and stack_label=... within the seat=N block.
        block_re = re.compile(
            r"(SeatROIs\(\s*seat=" + str(seat) + r",.*?)\)",
            re.DOTALL,
        )
        m = block_re.search(text)
        if not m:
            print(f"WARNING: could not find SeatROIs(seat={seat}, ...)", file=sys.stderr)
            continue
        block = m.group(0)
        new_block = re.sub(
            r"name_plate=ROI\([^)]*\)",
            f"name_plate={plate_str}",
            block,
            count=1,
        )
        new_block = re.sub(
            r"stack_label=ROI\([^)]*\)",
            f"stack_label={stack_str}",
            new_block,
            count=1,
        )
        if new_block != block:
            text = text[: m.start()] + new_block + text[m.end():]
            overrides_applied += 2

    print(f"# Overrides applied: {overrides_applied} / 12")
    print(f"# Affine changes:    {len(changes)}")
    print()

    if args.apply:
        ROIS_PATH.write_text(text, encoding="utf-8")
        print(f"wrote {ROIS_PATH}")
    else:
        print("(dry-run; pass --apply to write rois.py)")
        print()
        print("# First 10 affine changes:")
        for old, new in changes[:10]:
            print(f"  {old}")
            print(f"    -> {new}")
        if len(changes) > 10:
            print(f"  ... ({len(changes) - 10} more)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
