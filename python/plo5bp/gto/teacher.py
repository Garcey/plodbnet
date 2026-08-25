"""Teacher-batch quality floors (v1 HU river).

Defaults (documented, configurable at the CLI / function args):

- ``TEACHER_MAX_EXPL_BB = 1.0`` — reject a root if exploitability_bb is
  missing, non-finite, negative, or strictly above this cap. 1.0 bb is the
  starting HU-river teacher cap (not a Nash certificate).
- ``TEACHER_MIN_VISIT_MASS = 1.0`` — drop infosets whose dump ``visit_mass``
  is present and ``< 1.0``. ``visit_mass`` is ``sum(strategy_sum)``: one
  DCFR visit accumulates ~1.0. This is *beyond* the existing
  ``unused_uniform`` gate (mass<=0 or legacy exact 1/n). Legacy rows
  without ``visit_mass`` are not dropped by this floor.
- ``TEACHER_HOLDOUT_FRAC = 0.15`` — 15% of roots (in the 10–20% band) go
  to holdout. Assignment is SHA-256 of ``f"{split_seed}\\0{root_id}"``
  (deterministic, disjoint). Probe reads the holdout JSONL.
- ``TEACHER_SPLIT_SEED = 0`` — default hash seed for the split.

Library ``strategy_to_labels`` / ``export_dir`` keep back-compat defaults
(no expl floor, ``min_visit_mass=0``, ``holdout_frac=0``). Teacher CLIs
(``scripts/cfr_batch.py``, ``scripts/cfr_export_labels.py``,
``scripts/train_policy_from_cfr.py``) apply these constants.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable

TEACHER_MAX_EXPL_BB = 1.0
TEACHER_MIN_VISIT_MASS = 1.0
TEACHER_HOLDOUT_FRAC = 0.15
TEACHER_SPLIT_SEED = 0

SPLIT_TRAIN = "train"
SPLIT_HOLDOUT = "holdout"


def expl_reject_reason(
    expl_bb: float | None,
    *,
    max_expl_bb: float = TEACHER_MAX_EXPL_BB,
) -> str | None:
    """None if the root may be a teacher; else a short reason."""
    if expl_bb is None:
        return "expl_missing"
    try:
        e = float(expl_bb)
    except (TypeError, ValueError):
        return "expl_missing"
    if not math.isfinite(e):
        return "expl_missing"
    if e < 0:
        return "expl_missing"
    if e > float(max_expl_bb):
        return f"expl_{e:.4f}_gt_{float(max_expl_bb):g}"
    return None


def holdout_assignment(
    root_id: str,
    *,
    seed: int = TEACHER_SPLIT_SEED,
    holdout_frac: float = TEACHER_HOLDOUT_FRAC,
) -> str:
    """Deterministic train/holdout label for one root_id."""
    frac = float(holdout_frac)
    if frac <= 0.0:
        return SPLIT_TRAIN
    if frac >= 1.0:
        return SPLIT_HOLDOUT
    digest = hashlib.sha256(f"{int(seed)}\0{root_id}".encode("utf-8")).digest()
    u = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return SPLIT_HOLDOUT if u < frac else SPLIT_TRAIN


def split_root_ids(
    root_ids: Iterable[str],
    *,
    seed: int = TEACHER_SPLIT_SEED,
    holdout_frac: float = TEACHER_HOLDOUT_FRAC,
) -> tuple[list[str], list[str]]:
    """Return (train_ids, holdout_ids), each sorted, disjoint, union = input."""
    train: list[str] = []
    hold: list[str] = []
    for rid in root_ids:
        bucket = holdout_assignment(str(rid), seed=seed, holdout_frac=holdout_frac)
        (hold if bucket == SPLIT_HOLDOUT else train).append(str(rid))
    return sorted(train), sorted(hold)


def root_id_from_report(rep: dict[str, Any], *, fallback: str = "") -> str:
    """Prefer ``root.root_id``, then ``strategy.root_id``, then fallback."""
    root = rep.get("root") or {}
    if isinstance(root, dict):
        rid = root.get("root_id")
        if rid:
            return str(rid)
    strat = rep.get("strategy") or {}
    if isinstance(strat, dict):
        rid = strat.get("root_id")
        if rid:
            return str(rid)
    return str(fallback) if fallback else ""


def report_expl_bb(rep: dict[str, Any]) -> float | None:
    v = rep.get("exploitability_bb")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def teacher_export_kwargs(
    *,
    max_expl_bb: float | None = TEACHER_MAX_EXPL_BB,
    min_visit_mass: float = TEACHER_MIN_VISIT_MASS,
    holdout_frac: float = TEACHER_HOLDOUT_FRAC,
    split_seed: int = TEACHER_SPLIT_SEED,
) -> dict[str, Any]:
    """Kwargs ``export_dir`` / ``export_dir_detailed`` accept for teacher runs."""
    return {
        "max_expl_bb": max_expl_bb,
        "min_visit_mass": min_visit_mass,
        "holdout_frac": holdout_frac,
        "split_seed": split_seed,
    }
