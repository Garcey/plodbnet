"""Pre-restart sync check: guardian launch flags vs anneal_control baseline.

Why this exists (V7_DESIGN.md WS4): `runs/anneal_control.json` is applied
only when its CONTENT CHANGES after startup — at launch it is read as the
silent baseline, never applied. So any live-tuned value (lr, tier entropy,
clip rooms, fold-sup coef) that was moved via the control file but NOT baked
back into the guardian's flags silently REVERTS on the next restart. This
happened twice in the vSix1 era (an --entropy-coef 0.15 guardian nearly
reverted a live 0.2; --lr was hand-synced before the canary restart).

Run BEFORE any guardian restart:

    python scripts/check_restart_sync.py scripts/vSix2_guardian.sh \
        runs/anneal_control.json

Exit 0 = every overlapping knob matches; exit 1 = mismatch (table printed).
Only knobs present in BOTH files are compared: the control file is the
declaration of intended live values, the guardian is what a relaunch will
actually use.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# (flag, control-file key). tier_ent is handled separately — it maps to
# --entropy-coef only when every tier carries the same value.
_KNOBS = [
    ("--lr", "lr"),
    ("--clip-room-mid", "clip_room_mid"),
    ("--clip-room-ext", "clip_room_ext"),
    ("--q-fold-sup-coef", "q_fold_sup_coef"),
]


def parse_guardian_flags(text: str) -> dict[str, float]:
    """Last occurrence of each numeric long flag in a shell script.

    Handles `--flag value` and `--flag=value`; strips comments; joins
    backslash line-continuations so flags split across lines still parse.
    """
    joined = re.sub(r"\\\s*\n", " ", text)
    lines = [ln.split("#", 1)[0] for ln in joined.splitlines()]
    body = "\n".join(lines)
    flags: dict[str, float] = {}
    for m in re.finditer(
        r"(--[a-z][a-z0-9-]*)[=\s]+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b", body
    ):
        flags[m.group(1)] = float(m.group(2))
    return flags


def compare(
    guardian_flags: dict[str, float], control: dict
) -> list[tuple[str, float | None, float | None, bool]]:
    """Rows of (knob, guardian value, control value, ok). Knobs missing on
    either side are reported with None and ok=True (nothing to enforce)."""
    rows: list[tuple[str, float | None, float | None, bool]] = []
    for flag, key in _KNOBS:
        g = guardian_flags.get(flag)
        c = control.get(key)
        c = float(c) if isinstance(c, (int, float)) else None
        ok = g is None or c is None or abs(g - c) <= 1e-12 * max(1.0, abs(c))
        rows.append((flag, g, c, ok))

    tier_ent = control.get("tier_ent")
    g_ent = guardian_flags.get("--entropy-coef")
    if isinstance(tier_ent, dict) and tier_ent:
        vals = [float(v) for v in tier_ent.values()]
        if max(vals) - min(vals) <= 1e-12:
            ok = g_ent is None or abs(g_ent - vals[0]) <= 1e-12
            rows.append(("--entropy-coef", g_ent, vals[0], ok))
        else:
            # Per-tier values diverged: a scalar --entropy-coef cannot
            # represent them, so ANY relaunch collapses the tiers back to
            # the scalar. Always flag it.
            rows.append(("--entropy-coef (tiers differ!)", g_ent, None, False))
    elif "entropy_coef" in control:
        c = float(control["entropy_coef"])
        ok = g_ent is None or abs(g_ent - c) <= 1e-12
        rows.append(("--entropy-coef", g_ent, c, ok))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("guardian", type=Path, help="guardian .sh script")
    ap.add_argument("control", type=Path, help="runs/anneal_control.json")
    args = ap.parse_args(argv)

    flags = parse_guardian_flags(args.guardian.read_text(encoding="utf-8"))
    control = json.loads(args.control.read_text(encoding="utf-8"))
    rows = compare(flags, control)

    bad = False
    print(f"{'knob':32s} {'guardian':>14s} {'control':>14s}  verdict")
    for name, g, c, ok in rows:
        gs = "-" if g is None else f"{g:g}"
        cs = "-" if c is None else f"{c:g}"
        print(f"{name:32s} {gs:>14s} {cs:>14s}  {'ok' if ok else 'MISMATCH'}")
        bad |= not ok
    if bad:
        print(
            "\nMISMATCH: a relaunch would silently revert live values. Bake "
            "the control-file values into the guardian flags (or fix the "
            "control file) before restarting."
        )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
