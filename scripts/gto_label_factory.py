#!/usr/bin/env python
"""Smoke-canary NLH GTO label factory CLI.

Writes trash-fold / nuts-jam schema shards only. Production π* labels
come from the native CFR solver::

  .venv/Scripts/python scripts/cfr_batch.py ...
  .venv/Scripts/python scripts/cfr_export_labels.py ...
  .venv/Scripts/python scripts/train_policy_from_cfr.py ...

Examples::

  .venv/Scripts/python scripts/gto_label_factory.py --n-roots 16 --seed 0
  .venv/Scripts/python scripts/gto_label_factory.py --grid --out data/gto_nlh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.cfr_api import rust_cfr_available  # noqa: E402
from plo5bp.gto.factory import run_factory  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="NLH GTO smoke label factory")
    p.add_argument("--n-roots", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("data/gto_nlh"))
    p.add_argument(
        "--grid",
        action="store_true",
        help="Use deterministic SPR×street grid instead of random roots",
    )
    p.add_argument(
        "--no-smoke",
        action="store_true",
        help="Skip synthetic trash-fold / nuts-jam canaries",
    )
    args = p.parse_args()

    print(f"[gto] ClubGG root locked; rust_cfr available={rust_cfr_available()}")
    report = run_factory(
        n_roots=args.n_roots,
        seed=args.seed,
        out_dir=args.out,
        include_smoke=not args.no_smoke,
        use_grid=args.grid,
    )
    print(json.dumps(report.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
