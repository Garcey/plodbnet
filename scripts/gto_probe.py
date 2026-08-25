#!/usr/bin/env python
"""Holdout probe: PolicyNet checkpoint vs solver LabelRecord JSONL.

Exit codes:
  0 — probe gates passed (and optionally stamped onto the checkpoint)
  1 — probe gates failed or fatal error

Examples::

  .venv/Scripts/python scripts/gto_probe.py \\
      --ckpt checkpoints/gto_policy.pt \\
      --holdout data/gto_nlh/holdout_labels.jsonl \\
      --stamp

  .venv/Scripts/python scripts/gto_probe.py \\
      --ckpt checkpoints/gto_policy.pt \\
      --holdout data/gto_nlh/holdout_labels.jsonl \\
      --min-pure-agree 0.90 --max-gate-kl 0.50 --min-n 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.probe import (  # noqa: E402
    ProbeGates,
    probe_checkpoint,
    stamp_probe_on_checkpoint,
    write_probe_report,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Holdout probe gates for GTO PolicyNet")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument(
        "--holdout",
        type=Path,
        required=True,
        help="Holdout LabelRecord JSONL (export writes <stem>_holdout.jsonl)",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--min-pure-agree", type=float, default=0.90)
    p.add_argument("--max-gate-kl", type=float, default=0.50)
    p.add_argument("--min-n", type=int, default=1)
    p.add_argument("--min-pure-n", type=int, default=0)
    p.add_argument(
        "--stamp",
        action="store_true",
        help="Write probe result + is_gto_validated into the checkpoint",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path",
    )
    args = p.parse_args()

    if not args.ckpt.is_file():
        print(f"[probe] missing ckpt {args.ckpt}", file=sys.stderr)
        return 1
    if not args.holdout.is_file():
        print(f"[probe] missing holdout {args.holdout}", file=sys.stderr)
        return 1

    gates = ProbeGates(
        min_pure_agree=args.min_pure_agree,
        max_mean_gate_kl=args.max_gate_kl,
        min_n=args.min_n,
        min_pure_n=args.min_pure_n,
    )
    result = probe_checkpoint(
        args.ckpt, args.holdout, device=args.device, gates=gates
    )
    print(json.dumps(result.as_dict(), indent=2))

    if args.report is not None:
        write_probe_report(args.report, result)
        print(f"[probe] wrote {args.report}")

    if args.stamp:
        meta = stamp_probe_on_checkpoint(args.ckpt, result, device=args.device)
        print(
            f"[probe] stamped ckpt is_gto_validated="
            f"{meta.get('is_gto_validated')} source={meta.get('source')}"
        )

    if result.passed:
        print("[probe] PASS")
        return 0
    print("[probe] FAIL: " + "; ".join(result.reasons), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
