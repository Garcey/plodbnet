"""Teacher-batch quality floors (v1 HU river).

Defaults (documented, configurable at the CLI / function args):

- ``TEACHER_MAX_EXPL_BB = 1.0`` — reject a root if exploitability_bb is
  missing, non-finite, negative, or strictly above this cap. 1.0 bb is the
  starting HU-river teacher cap (not a Nash certificate).
- **Exploitability provenance** (review 2026-09-20 D8/F6) — the cap is only
  meaningful on a number produced by a FINAL estimator. A report is
  exploitability-VERIFIED only when its notes carry ``expl_kind=<kind>`` with
  ``kind`` in :data:`VERIFIED_EXPL_KINDS` and NO ``early_stop=`` /
  ``promoted_from_progress`` marker. Everything else — the Monte-Carlo poll
  estimate (``expl_kind=mc_poll``), target-based early stops (their number IS
  the poll), time-budget / stop-file exits, multiway ``mc_br_proxy`` numbers,
  promoted progress snapshots — is UNVERIFIED: batches neither accept nor
  reject on it and teacher export skips it by default. See
  :func:`expl_provenance`.
- ``TEACHER_MIN_VISIT_MASS = 1.0`` — drop infosets whose dump ``visit_mass``
  is present and ``< 1.0``. ``visit_mass`` is ``sum(strategy_sum)``: one
  DCFR visit accumulates ~1.0. This is *beyond* the existing
  ``unused_uniform`` gate (mass<=0 or legacy exact 1/n). Legacy rows
  without ``visit_mass`` are not dropped by this floor.
- ``TEACHER_HOLDOUT_FRAC = 0.15`` — 15% of roots (in the 10–20% band) go
  to holdout. Assignment is SHA-256 of ``f"{split_seed}\\0{root_id}"``
  (deterministic, disjoint, independent of ``PYTHONHASHSEED``). With
  ``strata`` (review 2026-09-20 D4) the split is stratified: every stratum
  (street × seats × SPR bucket, :func:`root_stratum`) with >= 2 roots gets
  >= 1 holdout root AND keeps >= 1 train root. Probe reads the holdout JSONL.
- ``TEACHER_SPLIT_SEED = 0`` — default hash seed for the split.

Library ``strategy_to_labels`` / ``export_dir`` keep back-compat defaults
(no expl floor, ``min_visit_mass=0``, ``holdout_frac=0``). Teacher CLIs
(``scripts/cfr_batch.py``, ``scripts/cfr_export_labels.py``,
``scripts/train_policy_from_cfr.py``) apply these constants.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

TEACHER_MAX_EXPL_BB = 1.0
TEACHER_MIN_VISIT_MASS = 1.0
TEACHER_HOLDOUT_FRAC = 0.15
TEACHER_SPLIT_SEED = 0

SPLIT_TRAIN = "train"
SPLIT_HOLDOUT = "holdout"

# Final (non-poll) exploitability estimators emitted by the Rust solver as an
# ``expl_kind=<kind>`` note. ``infoset_br`` counts only without an early-stop
# marker: the pre-2026-09-20 binary wrote that token on time-budget / stop-file
# exits too, where the number is really the biased 24-deal poll (review D8).
VERIFIED_EXPL_KINDS: frozenset[str] = frozenset(
    {"exact_infoset", "hero_enum", "infoset_br"}
)
EXPL_UNVERIFIED_PREFIX = "expl_unverified"


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


@dataclass(frozen=True)
class ExplProvenance:
    """How a report's ``exploitability_bb`` was produced (review D8/F6)."""

    kind: str | None  # token after ``expl_kind=`` (None when absent)
    verified: bool
    reason: str  # "" when verified, else why not
    early_stop: str | None = None  # token after ``early_stop=``

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "verified": self.verified,
            "reason": self.reason,
            "early_stop": self.early_stop,
        }


def _note_token(notes: Iterable[Any], key: str) -> str | None:
    """Value of the first ``<key>=<value>`` token found in ``notes``."""
    prefix = f"{key}="
    for n in notes:
        for tok in str(n).split():
            if tok.startswith(prefix):
                return tok[len(prefix):].strip().strip(",;") or None
    return None


def expl_provenance(rep: Mapping[str, Any] | Any) -> ExplProvenance:
    """Classify a SolveReport (dict or dataclass) as verified / unverified.

    Verified == ``status == "ok"``, a final estimator kind in the notes, and
    no early-stop / promoted-from-progress marker. Anything else must not be
    judged against the teacher cap.
    """
    if isinstance(rep, Mapping):
        status = rep.get("status")
        notes = list(rep.get("notes") or [])
        promoted = bool(rep.get("promoted_from_progress"))
    else:
        status = getattr(rep, "status", None)
        notes = list(getattr(rep, "notes", None) or [])
        promoted = bool(getattr(rep, "promoted_from_progress", False))
    kind = _note_token(notes, "expl_kind")
    early = _note_token(notes, "early_stop")
    if any("promoted_from_progress" in str(n) for n in notes):
        promoted = True

    if status != "ok":
        return ExplProvenance(kind, False, f"status={status}", early)
    if promoted:
        return ExplProvenance(kind, False, "promoted_from_progress", early)
    if early is not None:
        return ExplProvenance(kind, False, f"early_stop={early}", early)
    # Pre-token binary: a target-based early stop returned the poll estimate
    # with an "... early stop iter N" note and no expl_kind at all.
    if any("early stop" in str(n).lower() for n in notes):
        return ExplProvenance(kind, False, "early_stop=target_poll", "target_poll")
    if kind is None:
        return ExplProvenance(None, False, "expl_kind_missing", early)
    if kind not in VERIFIED_EXPL_KINDS:
        return ExplProvenance(kind, False, f"expl_kind={kind}", early)
    return ExplProvenance(kind, True, "", None)


def holdout_assignment(
    root_id: str,
    *,
    seed: int = TEACHER_SPLIT_SEED,
    holdout_frac: float = TEACHER_HOLDOUT_FRAC,
) -> str:
    """Deterministic train/holdout label for one root_id (unstratified rule)."""
    frac = float(holdout_frac)
    if frac <= 0.0:
        return SPLIT_TRAIN
    if frac >= 1.0:
        return SPLIT_HOLDOUT
    return SPLIT_HOLDOUT if _split_u(root_id, seed) < frac else SPLIT_TRAIN


def _split_u(root_id: str, seed: int) -> float:
    """SHA-256 → [0, 1). Never Python ``hash()`` (PYTHONHASHSEED-dependent)."""
    digest = hashlib.sha256(f"{int(seed)}\0{root_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def spr_bucket(spr: float | None) -> str:
    """Half-octave SPR bucket label (``floor(2*log2(spr))``).

    Fine enough that the step-7 grid points (SPR 1 / 2 / 3 / 5) are four
    different strata, coarse enough that continuous-SPR campaigns still group.
    """
    try:
        s = float(spr)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "spr?"
    if not math.isfinite(s) or s <= 0.0:
        return "spr?"
    return f"spr{int(math.floor(2.0 * math.log2(s) + 1e-9))}"


def root_stratum(root: Mapping[str, Any] | Any) -> str:
    """Stratum key of a root (report ``root`` dict or ``RootSpec``).

    ``street × num_seats × spr_bucket`` with SPR = shortest stack / pot.
    """
    def _get(key: str, default: Any = None) -> Any:
        if isinstance(root, Mapping):
            return root.get(key, default)
        return getattr(root, key, default)

    try:
        pot = float(_get("pot_bb") or 0.0)
    except (TypeError, ValueError):
        pot = 0.0
    stacks = [float(s) for s in (_get("stacks_bb") or []) if s is not None]
    try:
        eff = min(stacks) if stacks else float(_get("effective_stack_bb") or 0.0)
    except (TypeError, ValueError):
        eff = 0.0
    spr = (eff / pot) if pot > 0 else None
    return f"s{_get('street', '?')}_n{_get('num_seats', 2)}_{spr_bucket(spr)}"


def split_root_ids(
    root_ids: Iterable[str],
    *,
    seed: int = TEACHER_SPLIT_SEED,
    holdout_frac: float = TEACHER_HOLDOUT_FRAC,
    strata: Mapping[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return (train_ids, holdout_ids), each sorted, disjoint, union = input.

    Without ``strata`` this is the plain per-root hash rule. With ``strata``
    (``root_id -> stratum key``; review 2026-09-20 D4) the same hash decides
    within each stratum, then every stratum with >= 2 roots is repaired so it
    has >= 1 holdout root (the smallest hash) and >= 1 train root (the largest
    hash). The old unstratified step-7 split put both holdout roots at SPR 3,
    so SPR 1 / 2 / 5 were never probed. Roots missing from ``strata`` share
    one ``"?"`` stratum.
    """
    ids = sorted({str(r) for r in root_ids})
    frac = float(holdout_frac)
    if frac <= 0.0:
        return ids, []
    if frac >= 1.0:
        return [], ids
    if strata is None:
        train = [r for r in ids if _split_u(r, seed) >= frac]
        hold = [r for r in ids if _split_u(r, seed) < frac]
        return train, hold

    by_stratum: dict[str, list[str]] = {}
    for rid in ids:
        by_stratum.setdefault(str(strata.get(rid, "?")), []).append(rid)
    train: list[str] = []
    hold: list[str] = []
    for _key, members in sorted(by_stratum.items()):
        ranked = sorted(members, key=lambda r: (_split_u(r, seed), r))
        h = [r for r in ranked if _split_u(r, seed) < frac]
        if len(ranked) >= 2:
            if not h:
                h = [ranked[0]]
            elif len(h) == len(ranked):
                h = ranked[:-1]
        hset = set(h)
        hold.extend(h)
        train.extend(r for r in ranked if r not in hset)
    return sorted(train), sorted(hold)


# Fields that DEFINE the solved game (review 2026-09-20 F10). Two roots that
# differ in any of these are different solves, whatever their ``root_id``.
_FINGERPRINT_FIELDS = (
    "street",
    "pot_bb",
    "effective_stack_bb",
    "board",
    "num_seats",
    "bb_chips",
    "sb_chips",
    "ante_chips",
    "raise_sizes_pm",
    "allin_atom",
    "range_ip",
    "range_oop",
    "stacks_bb",
)


def root_fingerprint(root: Mapping[str, Any] | Any, *, length: int = 8) -> str:
    """Hex SHA-256 prefix of the game-defining root fields.

    Order-insensitive where the game is (board cards, raise-size menu);
    floats are rounded so ``10`` and ``10.0`` agree. ``root_id`` itself is
    excluded. Used as the discriminator on NEW auto/grid root ids and to spot
    legacy id collisions (same id, different board / pot / stack / sizes).
    """
    def _get(key: str) -> Any:
        if isinstance(root, Mapping):
            return root.get(key)
        return getattr(root, key, None)

    def _num(x: Any) -> Any:
        try:
            return round(float(x), 6)
        except (TypeError, ValueError):
            return None

    canon: dict[str, Any] = {}
    for key in _FINGERPRINT_FIELDS:
        v = _get(key)
        if key == "board":
            canon[key] = sorted(int(c) for c in (v or []))
        elif key == "raise_sizes_pm":
            canon[key] = sorted({int(x) for x in (v or [])})
        elif key == "stacks_bb":
            canon[key] = [_num(x) for x in (v or [])]
        elif key in ("pot_bb", "effective_stack_bb"):
            canon[key] = _num(v)
        elif key in ("range_ip", "range_oop"):
            canon[key] = str(v or "").strip()
        elif key == "allin_atom":
            canon[key] = bool(True if v is None else v)
        else:
            canon[key] = None if v is None else int(v)
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[: max(4, int(length))]


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


def teacher_root_reject_reason(
    rep: Mapping[str, Any],
    *,
    max_expl_bb: float | None = TEACHER_MAX_EXPL_BB,
    require_verified_expl: bool = True,
) -> str | None:
    """Teacher-export gate for one report: provenance first, then the cap.

    ``max_expl_bb=None`` disables both (library / verify paths).
    """
    if max_expl_bb is None:
        return None
    if require_verified_expl:
        prov = expl_provenance(rep)
        if not prov.verified:
            return f"{EXPL_UNVERIFIED_PREFIX}:{prov.reason}"
    return expl_reject_reason(report_expl_bb(dict(rep)), max_expl_bb=max_expl_bb)


def teacher_export_kwargs(
    *,
    max_expl_bb: float | None = TEACHER_MAX_EXPL_BB,
    min_visit_mass: float = TEACHER_MIN_VISIT_MASS,
    holdout_frac: float = TEACHER_HOLDOUT_FRAC,
    split_seed: int = TEACHER_SPLIT_SEED,
    require_verified_expl: bool = True,
) -> dict[str, Any]:
    """Kwargs ``export_dir`` / ``export_dir_detailed`` accept for teacher runs."""
    return {
        "max_expl_bb": max_expl_bb,
        "min_visit_mass": min_visit_mass,
        "holdout_frac": holdout_frac,
        "split_seed": split_seed,
        "require_verified_expl": require_verified_expl,
    }
