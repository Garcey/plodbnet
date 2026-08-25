#!/usr/bin/env python
"""Overnight native CFR full-hand batch (kill-safe).

Examples::

  .venv/Scripts/python scripts/cfr_overnight.py --grid data/cfr/overnight_grid.json --dry-run
  .venv/Scripts/python scripts/cfr_overnight.py --grid data/cfr/overnight_grid.json

Stop in the morning (exports current strategies)::

  echo. > data/cfr/overnight/STOP
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.cfr_overnight import run_overnight  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Overnight native CFR full-hand batch")
    p.add_argument(
        "--grid",
        type=Path,
        default=Path("data/cfr/overnight_grid.json"),
    )
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.grid.is_file():
        print(f"grid not found: {args.grid}", file=sys.stderr)
        return 1

    report = run_overnight(
        args.grid,
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )
    return 1 if report.as_dict().get("n_fail") else 0


if __name__ == "__main__":
    raise SystemExit(main())
