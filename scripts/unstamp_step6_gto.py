#!/usr/bin/env python
"""Unstamp Step 6 checkpoint: holdout probe is not a Nash teacher cert.

``is_gto_validated`` is the UI \"GTO AI\" badge. Step 6's teacher was
accepted at a one-off 5.0 bb cap (deal-BR ~4 bb). Move the holdout probe
aside so the badge stays off; keep the numbers for the record.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

import torch

from plo5bp.gto.policy_net import (  # noqa: E402
    is_validated_gto_checkpoint,
    load_policy_checkpoint,
    save_policy_checkpoint,
)

CKPT = Path("checkpoints/gto_step6.pt")
NOTE = (
    "Holdout probe measured PolicyNet vs this teacher only. "
    "Teacher expl was deal-BR (~4 bb floor), not a 1.0 bb Nash cert. "
    "Badge is_gto_validated remains false."
)


def main() -> int:
    if not CKPT.is_file():
        print(f"missing {CKPT}", file=sys.stderr)
        return 1
    before = is_validated_gto_checkpoint(CKPT)
    model, meta = load_policy_checkpoint(CKPT, device="cpu")
    meta = dict(meta)
    probe = meta.pop("probe", None)
    if probe is not None:
        meta["holdout_probe"] = probe
    meta["is_gto_validated"] = False
    meta["teacher_nash_certified"] = False
    meta["gto_badge_note"] = NOTE
    for k in ("path", "model", "actor", "policy"):
        meta.pop(k, None)
    save_policy_checkpoint(CKPT, model, meta=meta)
    after = is_validated_gto_checkpoint(CKPT)
    print(
        json.dumps(
            {
                "ckpt": str(CKPT),
                "badge_before": before,
                "badge_after": after,
                "holdout_probe_kept": "holdout_probe" in meta,
                "note": NOTE,
            },
            indent=2,
        )
    )
    return 0 if (before is True and after is False) or after is False else 1


if __name__ == "__main__":
    raise SystemExit(main())
