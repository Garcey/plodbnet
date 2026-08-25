"""Export native CFR strategy JSON → LabelRecord JSONL for PolicyNet training.

Maps solve-ladder actions (FOLD / CHECK_CALL / RAISE_pm / ALLIN) onto the
shared NLH gate + ``NLH_ANCHOR_SPEC`` menu. Decodes private views from
infoset ids:

- ``…_c{combo}`` with combo 0..1325 → exact hole cards
- ``mwpf_p{seat}_path{path}_c{class}`` with class 0..168 → representative hole
- ``pf_p{}_h{}_c{class}`` HU preflop class

**Export-time gates (per infoset, DROP — never invent ``to_call``):**

- ``unused_uniform`` — untouched node (``visit_mass==0``) or, on legacy
  dumps without visit mass, exact 1/n average strategy (the 0.5/0.5 default).
- ``low_visit`` — ``visit_mass`` present and below ``min_visit_mass``
  (teacher default 1.0 = one DCFR visit). Library default is 0 (off).
- ``illegal_fold`` — FOLD in the menu or fold mass > 0 when ``to_call==0``.
- ``inconsistent_public`` — pot<=0, to_call<0, to_call>hero stack, or a
  v2 row missing ``to_call_chips``.
- ``hole_decode`` — only when ``require_hole=True`` (ochs_bucket without
  raw_combo, board-blocked combo, etc.).
- ``iso_without_raw`` — ``iso_id`` set and ``raw_combo`` missing.

Per-root teacher floors (``export_dir``, off unless passed):

- skip the file when ``exploitability_bb`` is missing / above ``max_expl_bb``
- split remaining roots by SHA-256 into train + holdout JSONL

The whole file is not aborted; rejected rows are counted in ``ExportStats``.
``max_infosets=None`` exports **all infosets that pass the gates**.
"""

from __future__ import annotations

import json
import re
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from plo5bp.gto.labels import (
    LABEL_SCHEMA_VERSION,
    ActionProb,
    LabelRecord,
    map_size_to_anchor,
    normalize_gate_probs,
    write_jsonl,
)
from plo5bp.gto.preflop_class import (
    NUM_PREFLOP_CLASSES,
    combo_to_cards,
    preflop_class_label,
    representative_hole,
)
from plo5bp.gto.roots import CLUBGG_NLH_ROOT
from plo5bp.gto.teacher import (
    TEACHER_HOLDOUT_FRAC,
    TEACHER_MAX_EXPL_BB,
    TEACHER_MIN_VISIT_MASS,
    TEACHER_SPLIT_SEED,
    expl_reject_reason,
    report_expl_bb,
    root_id_from_report,
    split_root_ids,
)
from plo5bp.sizing import NLH_ANCHOR_SPEC

# Matches rust_engine/src/cfr/infoset.rs + card_abs.rs
DUMP_SCHEMA_VERSION = 2
OCHS_BUCKET_BASE = 2_000_000
PRIV_COMBO = "combo"
PRIV_CLASS = "class"
PRIV_OCHS_BUCKET = "ochs_bucket"

# Gate reasons (per-row DROP).
REJECT_UNUSED_UNIFORM = "unused_uniform"
REJECT_ILLEGAL_FOLD = "illegal_fold"
REJECT_INCONSISTENT_PUBLIC = "inconsistent_public"
REJECT_HOLE_DECODE = "hole_decode"
REJECT_ISO_WITHOUT_RAW = "iso_without_raw"
REJECT_LOW_VISIT = "low_visit"

# Unused CFR default is exact 1/n (f64). First-visit mixed nodes have visit_mass>0.
_UNIFORM_ATOL = 1e-9
_FOLD_MASS_EPS = 1e-12


@dataclass
class ExportReject:
    infoset_id: str
    reason: str


@dataclass
class ExportStats:
    """Per-call reject tally. Pass into ``strategy_to_labels`` to inspect gates."""

    kept: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    details: list[ExportReject] = field(default_factory=list)

    def record_keep(self) -> None:
        self.kept += 1

    def record_drop(self, infoset_id: str, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1
        self.details.append(ExportReject(infoset_id, reason))

    @property
    def n_dropped(self) -> int:
        return sum(self.rejected.values())

# Infoset id patterns from mccfr.rs / dcfr export
_RE_MWPF = re.compile(
    r"^mwpf_p(?P<seat>\d+)_path(?P<path>.+)_c(?P<priv>\d+)$"
)
_RE_MW = re.compile(r"^mw_p(?P<seat>\d+)_h(?P<hist>\d+)_c(?P<priv>\d+)$")
_RE_PF = re.compile(r"^pf_p(?P<seat>\d+)_h(?P<hist>\d+)_c(?P<priv>\d+)$")
_RE_GENERIC_C = re.compile(r"_c(?P<priv>\d+)$")


def pot_frac_pm_to_anchor(pm: int | None, *, is_allin: bool = False) -> int:
    """Map solve pot-fraction per-mille (or ALLIN) → NLH_ANCHOR_SPEC index."""
    k_allin = NLH_ANCHOR_SPEC.count - 1
    if is_allin or pm is None:
        return k_allin if NLH_ANCHOR_SPEC.allin_atom else NLH_ANCHOR_SPEC.count - 1
    pm = int(pm)
    # Nearest fraction anchor (exclude all-in atom when matching fracs)
    fracs = list(NLH_ANCHOR_SPEC.fracs_pm)
    best_k = 0
    best_d = abs(fracs[0] - pm)
    for k, f in enumerate(fracs):
        d = abs(int(f) - pm)
        if d < best_d:
            best_d = d
            best_k = k
    # If closer to deep all-in than any frac (e.g. huge overbet), use all-in
    if NLH_ANCHOR_SPEC.allin_atom and pm >= fracs[-1] + (fracs[-1] - fracs[-2]) // 2:
        # large overbets still map to last frac unless explicitly ALLIN
        pass
    return int(best_k)


def _legacy_private_kind(iid: str, priv: int | None) -> str | None:
    """Prefix-based kind for old dumps. Never uses ``priv < 169`` alone.

    River HU ids are ``p{seat}_h{hash}_c{combo}`` with combo 0..1325.
    Preflop class ids are ``pf_`` / ``mwpf_`` / ``chart_``.
    """
    if priv is None:
        return None
    if int(priv) >= OCHS_BUCKET_BASE:
        return PRIV_OCHS_BUCKET
    s = str(iid or "")
    is_preflop_class = (
        s.startswith("pf_") or s.startswith("mwpf_") or s.startswith("chart_")
    )
    is_postflop_combo = (
        s.startswith("mw_")
        or (s.startswith("p") and not s.startswith("pf_"))
        or (
            "_h" in s
            and "_c" in s
            and not s.startswith("pf_")
            and not s.startswith("mwpf_")
        )
    )
    if is_preflop_class and int(priv) < NUM_PREFLOP_CLASSES:
        return PRIV_CLASS
    if is_postflop_combo:
        return PRIV_COMBO
    if int(priv) < NUM_PREFLOP_CLASSES and not is_postflop_combo:
        return PRIV_CLASS
    if 0 <= int(priv) < 1326:
        return PRIV_COMBO
    return PRIV_COMBO


def _parse_infoset_id(iid: str) -> dict[str, Any]:
    """Extract seat, path, private view kind from infoset_id (legacy dumps)."""
    out: dict[str, Any] = {
        "seat": 0,
        "path": "",
        "private": None,
        "private_kind": None,  # "combo" | "class" | "ochs_bucket"
        "history_hash": None,
    }
    s = str(iid or "")
    m = _RE_MWPF.match(s)
    if m:
        out["seat"] = int(m.group("seat"))
        out["path"] = m.group("path")
        priv = int(m.group("priv"))
        out["private"] = priv
        out["private_kind"] = _legacy_private_kind(s, priv)
        return out
    m = _RE_MW.match(s) or _RE_PF.match(s)
    if m:
        out["seat"] = int(m.group("seat"))
        out["history_hash"] = int(m.group("hist"))
        priv = int(m.group("priv"))
        out["private"] = priv
        out["private_kind"] = _legacy_private_kind(s, priv)
        return out
    m = _RE_GENERIC_C.search(s)
    if m:
        priv = int(m.group("priv"))
        out["private"] = priv
        out["private_kind"] = _legacy_private_kind(s, priv)
    # seat from pN if present
    m2 = re.search(r"_p(\d+)_", s)
    if m2:
        out["seat"] = int(m2.group(1))
    return out


def _hole_from_private(
    private: int | None,
    kind: str | None,
    *,
    board: Sequence[int],
    raw_combo: int | None = None,
    iso_id: int | None = None,
) -> list[int]:
    """Decode hole cards for serve-matching raw 52-hot.

    ``raw_combo`` is the dealt combo. ``iso_id`` is the infoset key under
    suit isomorphism — **never** decode it as hole cards.
    """
    if raw_combo is not None:
        try:
            if 0 <= int(raw_combo) < 1326:
                hole = combo_to_cards(int(raw_combo))
                if any(c in board for c in hole):
                    return []
                return hole
        except ValueError:
            pass
    # Iso-canonical private view without a raw combo: refuse (do not poison).
    if iso_id is not None and (private is None or int(private) == int(iso_id)):
        return []
    if private is None:
        return []
    if kind == PRIV_OCHS_BUCKET:
        return []
    if kind == PRIV_CLASS:
        try:
            return representative_hole(int(private), blocked=board)
        except ValueError:
            return []
    # combo (including river ids 0..168 — do NOT treat as 169-class)
    try:
        if 0 <= int(private) < 1326:
            hole = combo_to_cards(int(private))
            if any(c in board for c in hole):
                return []
            return hole
    except ValueError:
        pass
    return []


def is_unused_uniform(
    probs: Sequence[float],
    *,
    visit_mass: float | None = None,
    atol: float = _UNIFORM_ATOL,
) -> bool:
    """True for untouched infosets (empty strategy_sum → exact 1/n default).

    When ``visit_mass`` is present (new dumps), only mass<=0 is unused.
    First-visit nodes can look uniform but have mass>0 — those are kept.
    Legacy dumps without visit_mass fall back to exact-uniform probs.
    """
    if visit_mass is not None:
        vm = float(visit_mass)
        return (not math.isfinite(vm)) or vm <= 0.0
    n = len(probs)
    if n < 2:
        return False
    target = 1.0 / n
    return all(abs(float(p) - target) <= atol for p in probs)


def _hero_stack(stacks: Sequence[int], seat: int, fallback: int) -> int:
    if 0 <= seat < len(stacks):
        return int(stacks[seat])
    return int(fallback)


def reject_reason(
    *,
    actions: Sequence[str],
    probs: Sequence[float],
    to_call: int,
    pot_chips: int,
    hero_stack: int,
    visit_mass: float | None,
    v2_missing_to_call: bool,
    hole: Sequence[int],
    require_hole: bool,
    iso_id: int | None = None,
    raw_combo: int | None = None,
    min_visit_mass: float = 0.0,
) -> str | None:
    """Return a REJECT_* reason or None if the row may be exported."""
    if iso_id is not None and raw_combo is None:
        return REJECT_ISO_WITHOUT_RAW
    if v2_missing_to_call:
        return REJECT_INCONSISTENT_PUBLIC
    if pot_chips <= 0 or to_call < 0:
        return REJECT_INCONSISTENT_PUBLIC
    if to_call > max(0, hero_stack):
        return REJECT_INCONSISTENT_PUBLIC
    if is_unused_uniform(probs, visit_mass=visit_mass):
        return REJECT_UNUSED_UNIFORM
    if (
        visit_mass is not None
        and float(min_visit_mass) > 0.0
        and float(visit_mass) < float(min_visit_mass)
    ):
        return REJECT_LOW_VISIT
    acts_u = [str(a).upper() for a in actions]
    fold_in_menu = "FOLD" in acts_u
    fold_mass = sum(
        float(p) for a, p in zip(actions, probs) if str(a).upper() == "FOLD"
    )
    if to_call <= 0 and (fold_in_menu or fold_mass > _FOLD_MASS_EPS):
        return REJECT_ILLEGAL_FOLD
    if require_hole and len(hole) < 2:
        return REJECT_HOLE_DECODE
    return None


def _path_tokens(path: str) -> list[str]:
    if not path or path == "open":
        return []
    return [t for t in path.split(",") if t]


def _node_chips_from_path(
    path: str,
    *,
    street: int,
    num_seats: int,
    bb: int,
    sb: int,
    ante: int,
    stack_chips: int,
    stacks_chips: list[int],
    seat: int,
    actions: Sequence[str],
) -> dict[str, int]:
    """Estimate pot / to_call / min-max raise for a public path.

    Exact public-state reconstruction is not stored in strategy dumps; this
    uses engine-consistent rules for the two common cases:

    - Preflop push/fold (F / AI path labels)
    - Postflop root (empty history): check/open or facing bet from action menu
    """
    acts_u = [str(a).upper() for a in actions]
    has_fold = "FOLD" in acts_u
    has_allin_only = set(acts_u) <= {"FOLD", "ALLIN", "CHECK_CALL"} and "ALLIN" in acts_u

    n = max(2, num_seats)
    # Multiway preflop pot rebuild (matches mccfr)
    if street == 0 and n >= 2:
        pot0 = n * ante + sb + bb
    else:
        pot0 = 0  # filled by caller from root

    tokens = _path_tokens(path)
    n_ai = sum(1 for t in tokens if t == "AI")
    n_f = sum(1 for t in tokens if t == "F")

    seat_stack = stacks_chips[seat] if seat < len(stacks_chips) else stack_chips
    # Remaining after blinds for preflop multiway
    if street == 0:
        remaining = seat_stack - ante
        sb_seat, bb_seat = n - 2, n - 1
        if seat == sb_seat:
            remaining -= sb
        elif seat == bb_seat:
            remaining -= bb
        remaining = max(0, remaining)
    else:
        remaining = seat_stack

    if street == 0 and has_allin_only:
        # Push/fold tree
        if n_ai > 0:
            # Facing at least one jam: call is all-in remaining
            to_call = remaining
            pot = pot0 + n_ai * stack_chips  # rough: each jam ~ full start
            # Cap pot more carefully: each AI commits remaining at act time ≈ stack
            pot = pot0
            for t in tokens:
                if t == "AI":
                    pot += stack_chips - ante  # order-of-magnitude
            min_r = 0
            max_r = remaining  # reshove / call-all-in via ALLIN atom
        else:
            # Open or facing blinds only (folds don't change bet_to_call)
            to_call = bb if seat != (n - 1) else 0
            # BB facing limps doesn't exist in pure PF; BB only acts after AI
            # For UTG/CO/BTN/SB open: facing bb
            if seat == n - 1 and n_f == n - 1:
                # BB option after all fold — rare in our tree (no BB after F,F,F)
                to_call = 0
            elif not has_fold and "ALLIN" in acts_u and "FOLD" not in acts_u:
                # Open shove (no fold legal) — free open, to_call=0 at checkless open
                # Actually PF open faces BB so to_call=bb and fold is legal...
                # Open shove menu is ALLIN only when to_call==0 in actions.rs
                to_call = 0
            else:
                to_call = bb if has_fold else 0
            pot = pot0
            min_r = remaining
            max_r = remaining
        return {
            "pot_chips": max(pot0, pot),
            "to_call_chips": max(0, int(to_call)),
            "min_raise_chips": max(0, int(min_r if min_r else bb)),
            "max_raise_chips": max(0, int(max_r if max_r else remaining)),
            "hero_stack": max(0, int(remaining)),
        }

    # Postflop / general: do NOT invent to_call from FOLD-in-menu.
    # Missing dump to_call + fold in menu is rejected by illegal_fold.
    min_r = bb
    max_r = remaining if remaining > 0 else stack_chips
    return {
        "pot_chips": pot0,
        "to_call_chips": 0,
        "min_raise_chips": min_r,
        "max_raise_chips": max_r,
        "hero_stack": remaining if remaining > 0 else stack_chips,
    }


def _map_actions_to_probs(
    actions: Sequence[str],
    probs: Sequence[float],
    *,
    pot_chips: int,
    to_call: int,
    min_raise: int,
    max_raise: int,
) -> tuple[list[float], list[ActionProb]]:
    """Aggregate gates + attach NLH anchor indices on raise/all-in."""
    fold_p = xc_p = raise_p = 0.0
    action_probs: list[ActionProb] = []
    for a, p in zip(actions, probs):
        p = float(p)
        if p < 0:
            p = 0.0
        al = str(a).upper()
        if al == "FOLD":
            fold_p += p
            action_probs.append(
                ActionProb(gate="fold", anchor_k=None, pot_frac=None, chips=0, prob=p)
            )
        elif al in ("CHECK_CALL", "CHECK", "CALL"):
            xc_p += p
            action_probs.append(
                ActionProb(
                    gate="check_call",
                    anchor_k=None,
                    pot_frac=None,
                    chips=int(to_call),
                    prob=p,
                )
            )
        elif al == "ALLIN":
            raise_p += p
            k = pot_frac_pm_to_anchor(None, is_allin=True)
            chips = int(max_raise) if max_raise > 0 else int(to_call)
            action_probs.append(
                ActionProb(
                    gate="raise",
                    anchor_k=k,
                    pot_frac=None,
                    chips=chips,
                    prob=p,
                )
            )
        elif al.startswith("RAISE_"):
            raise_p += p
            pm: int | None
            try:
                pm = int(al.split("_", 1)[1])
            except ValueError:
                pm = None
            pot_frac = (pm / 1000.0) if pm is not None else None
            # Chip estimate: pot-fraction after call
            if pm is not None and pot_chips > 0:
                base = pot_chips + to_call
                chips = to_call + (pm * base + 500) // 1000
                chips = max(min_raise, min(max_raise, chips)) if max_raise > 0 else chips
            else:
                chips = max_raise
            if max_raise > 0 and min_raise > 0:
                k = map_size_to_anchor(
                    chips,
                    min_raise=min_raise,
                    max_raise=max_raise,
                    pot=pot_chips,
                    to_call=to_call,
                )
            else:
                k = pot_frac_pm_to_anchor(pm, is_allin=False)
            action_probs.append(
                ActionProb(
                    gate="raise",
                    anchor_k=int(k),
                    pot_frac=pot_frac,
                    chips=int(chips),
                    prob=p,
                )
            )
        else:
            # Unknown label → treat as check/call mass so simplex still works
            xc_p += p
            action_probs.append(
                ActionProb(
                    gate="check_call",
                    anchor_k=None,
                    pot_frac=None,
                    chips=0,
                    prob=p,
                )
            )
    gates = normalize_gate_probs(fold_p, xc_p, raise_p)
    return gates, action_probs


def strategy_to_labels(
    strategy_report: dict[str, Any],
    *,
    source: str = "rust_cfr",
    max_infosets: int | None = None,
    require_hole: bool = False,
    stats: ExportStats | None = None,
    min_visit_mass: float = 0.0,
) -> list[LabelRecord]:
    """Map a SolveReport dict into LabelRecord rows that pass export gates.

    Args:
        max_infosets: cap for smoke/debug applied **before** gates.
        require_hole: DROP ``hole_decode`` when the private view cannot be
            turned into two hole cards.
        stats: optional tally of kept / dropped reasons.
        min_visit_mass: DROP ``low_visit`` when visit_mass is present and
            below this floor. 0 (library default) disables. Teacher CLI
            uses ``TEACHER_MIN_VISIT_MASS`` (1.0).
    """
    root = strategy_report.get("root") or {}
    strategy = strategy_report.get("strategy") or {}
    infosets = list(strategy.get("infosets") or [])
    if max_infosets is not None:
        infosets = infosets[: int(max_infosets)]

    bb = int(root.get("bb_chips") or CLUBGG_NLH_ROOT.bb)
    sb = int(root.get("sb_chips") or CLUBGG_NLH_ROOT.sb)
    ante = int(root.get("ante_chips") or 0)
    pot_bb = float(root.get("pot_bb") or 1.0)
    stack_bb = float(root.get("effective_stack_bb") or 50.0)
    pot_chips_root = int(round(pot_bb * bb))
    stack_chips = int(round(stack_bb * bb))
    board = [int(c) for c in (root.get("board") or [])]
    street = int(root.get("street") or 0)
    num_seats = int(root.get("num_seats") or 2)
    root_name = str(root.get("root_id") or strategy.get("root_id") or "cfr")
    stacks_bb = list(root.get("stacks_bb") or [])
    if len(stacks_bb) == num_seats:
        stacks_chips = [int(round(float(s) * bb)) for s in stacks_bb]
    else:
        stacks_chips = [stack_chips] * num_seats

    # Multiway preflop: pot from blinds (ignore pot_bb desync)
    if street == 0 and num_seats >= 2:
        pot_chips_root = num_seats * ante + sb + bb

    notes_base = {
        "source_status": strategy_report.get("status"),
        "iterations": strategy_report.get("iterations_run"),
        "exploitability_bb": strategy_report.get("exploitability_bb"),
        "solve_notes": list(strategy_report.get("notes") or [])[:12],
    }
    is_mc_proxy = any(
        ("mc_br_proxy" in str(n)) or ("MC BR proxy" in str(n))
        for n in (strategy_report.get("notes") or [])
    )

    labels: list[LabelRecord] = []
    for iset in infosets:
        actions = list(iset.get("actions") or [])
        probs = list(iset.get("probs") or [])
        if len(actions) != len(probs) or not actions:
            continue
        iid = str(iset.get("infoset_id") or "")
        meta = _parse_infoset_id(iid)
        schema_v = int(iset.get("schema_version") or strategy.get("schema_version") or 0)
        is_v2 = schema_v >= DUMP_SCHEMA_VERSION
        has_dump = (
            is_v2
            or iset.get("private_kind")
            or iset.get("to_call_chips") is not None
        )
        if iset.get("actor") is not None:
            seat = int(iset["actor"])
        else:
            seat = int(meta["seat"])
        dump_path = iset.get("path")
        if isinstance(dump_path, list):
            path = ",".join(str(x) for x in dump_path)
        else:
            path = str(meta.get("path") or "")
        kind = iset.get("private_kind") or meta.get("private_kind")
        priv = iset.get("private_id")
        if priv is None:
            priv = meta.get("private")
        raw_combo = iset.get("raw_combo")
        if raw_combo is not None:
            try:
                raw_combo = int(raw_combo)
            except (TypeError, ValueError):
                raw_combo = None
        iso_id = iset.get("iso_id")
        if iso_id is not None:
            try:
                iso_id = int(iso_id)
            except (TypeError, ValueError):
                iso_id = None
        node_board = [int(c) for c in (iset.get("board") or board)]
        hole = _hole_from_private(
            None if priv is None else int(priv),
            kind,
            board=node_board,
            raw_combo=raw_combo,
            iso_id=iso_id,
        )

        row_street = street
        row_stacks = list(stacks_chips)
        row_board = list(node_board)
        v2_missing_to_call = is_v2 and iset.get("to_call_chips") is None
        if has_dump and iset.get("to_call_chips") is not None:
            # Authoritative public state — never invent to_call.
            pot_chips = int(iset.get("pot_chips") or pot_chips_root)
            to_call = int(iset["to_call_chips"])
            min_r = int(iset.get("min_raise_chips") or 0)
            max_r = int(iset.get("max_raise_chips") or 0)
            if iset.get("stacks_chips"):
                row_stacks = [int(x) for x in iset["stacks_chips"]]
            chip_est = {
                "hero_stack": _hero_stack(row_stacks, seat, stack_chips)
            }
            if iset.get("street") is not None:
                row_street = int(iset["street"])
        else:
            chip_est = _node_chips_from_path(
                path,
                street=street,
                num_seats=num_seats,
                bb=bb,
                sb=sb,
                ante=ante,
                stack_chips=stack_chips,
                stacks_chips=stacks_chips,
                seat=seat,
                actions=actions,
            )
            pot_chips = int(chip_est["pot_chips"] or pot_chips_root)
            if pot_chips <= 0:
                pot_chips = pot_chips_root
            to_call = int(chip_est["to_call_chips"])
            min_r = int(chip_est["min_raise_chips"])
            max_r = int(chip_est["max_raise_chips"])
            # Never invent to_call from FOLD-in-menu. If the reconstructed
            # menu has no fold, this is a check option (to_call must be 0).
            acts_u = [str(a).upper() for a in actions]
            if "FOLD" not in acts_u:
                to_call = 0

        visit_mass = iset.get("visit_mass")
        if visit_mass is not None:
            try:
                visit_mass = float(visit_mass)
            except (TypeError, ValueError):
                visit_mass = None
        hero_stack = int(chip_est.get("hero_stack") or _hero_stack(row_stacks, seat, stack_chips))
        why = reject_reason(
            actions=actions,
            probs=probs,
            to_call=to_call,
            pot_chips=pot_chips,
            hero_stack=hero_stack,
            visit_mass=visit_mass,
            v2_missing_to_call=v2_missing_to_call,
            hole=hole,
            require_hole=require_hole,
            iso_id=iso_id,
            raw_combo=raw_combo,
            min_visit_mass=min_visit_mass,
        )
        if why is not None:
            if stats is not None:
                stats.record_drop(iid, why)
            continue

        gates, action_probs = _map_actions_to_probs(
            actions,
            probs,
            pot_chips=pot_chips,
            to_call=to_call,
            min_raise=min_r,
            max_raise=max_r,
        )
        # If raise illegal in menu, force raise mass to 0
        has_raise = any(
            str(a).upper().startswith("RAISE") or str(a).upper() == "ALLIN"
            for a in actions
        )
        if not has_raise:
            gates = normalize_gate_probs(gates[0], gates[1] + gates[2], 0.0)
            max_r = 0

        spr = (max_r / pot_chips) if pot_chips > 0 else 0.0
        # button: multiway last seat; HU seat 1
        button = num_seats - 1 if num_seats > 2 else 1

        class_id = None
        if kind == PRIV_CLASS and priv is not None:
            class_id = int(priv)

        notes = {
            **notes_base,
            "infoset_id": iid,
            "path": path,
            "private_kind": kind,
            "private_id": None if priv is None else int(priv),
            "raw_combo": raw_combo,
            "iso_id": iso_id,
            "teacher_iso": False,
            "dump_schema": int(iset.get("schema_version") or 0),
            "class_id": class_id,
            "class_label": (
                preflop_class_label(class_id) if class_id is not None else None
            ),
            "mc_br_proxy": is_mc_proxy,
            "hero_stack_est": chip_est.get("hero_stack"),
            "visit_mass": visit_mass,
            "bb_chips": bb,
            "root_pot_chips": int(pot_chips_root),
            "root_stacks_chips": list(stacks_chips),
            "path_tokens": (
                [t for t in path.split(",") if t] if path and path != "open" else []
            ),
        }

        labels.append(
            LabelRecord(
                schema_version=LABEL_SCHEMA_VERSION,
                source=source,
                root_name=root_name,
                num_seats=num_seats,
                street=row_street,
                spr=float(spr),
                pot_chips=int(pot_chips),
                to_call_chips=int(to_call),
                min_raise_chips=int(min_r),
                max_raise_chips=int(max_r),
                hero_seat=int(seat) % num_seats,
                button=int(button),
                hero_hole=hole,
                board=list(row_board),
                stacks_chips=list(row_stacks),
                gate_probs=gates,
                action_probs=action_probs,
                value_bb=None,
                solve_id=iid,
                notes=notes,
            )
        )
        if stats is not None:
            stats.record_keep()
    return labels


_EXPORT_SKIP_NAMES = frozenset(
    {
        "manifest.json",
        "plan.json",
        "INDEX.json",
        "certificate.json",
        "split.json",
    }
)
_EXPORT_SKIP_PARENTS = frozenset({"rejected", "markers"})


@dataclass
class ExportDirResult:
    """Train/holdout counts plus skipped roots from ``export_dir_detailed``."""

    n_train: int = 0
    n_holdout: int = 0
    train_root_ids: list[str] = field(default_factory=list)
    holdout_root_ids: list[str] = field(default_factory=list)
    skipped_expl: list[dict[str, Any]] = field(default_factory=list)
    stats: ExportStats = field(default_factory=ExportStats)
    train_path: str = ""
    holdout_path: str | None = None
    split_path: str | None = None

    @property
    def n_kept(self) -> int:
        return int(self.n_train) + int(self.n_holdout)


def _export_json_files(in_path: Path) -> list[Path]:
    if in_path.is_file():
        return [in_path]
    files = sorted(in_path.glob("**/*.json"))
    keep: list[Path] = []
    for f in files:
        if f.name in _EXPORT_SKIP_NAMES:
            continue
        if f.name.endswith("_split.json"):
            continue
        if "chart" in f.name.lower():
            continue
        if any(p.name in _EXPORT_SKIP_PARENTS for p in f.parents):
            continue
        keep.append(f)
    return keep


def _is_solve_report(rep: Any) -> bool:
    if not isinstance(rep, dict):
        return False
    if "strategy" not in rep and "status" not in rep:
        return False
    # Chart exports (hands list) are not SolveReports
    if "hands" in rep and "strategy" not in rep:
        return False
    # Batch rejected wrappers nest the report
    if rep.get("status") == "rejected" and "report" in rep:
        return False
    return True


def export_dir_detailed(
    in_path: Path | str,
    out_jsonl: Path | str,
    *,
    source: str = "rust_cfr",
    max_infosets_per_file: int | None = None,
    require_hole: bool = False,
    max_expl_bb: float | None = None,
    min_visit_mass: float = 0.0,
    holdout_frac: float = 0.0,
    holdout_jsonl: Path | str | None = None,
    split_seed: int = TEACHER_SPLIT_SEED,
    split_manifest: Path | str | None = None,
) -> ExportDirResult:
    """Read strategy JSON file(s) and write train (+ optional holdout) JSONL.

    Library defaults: no expl floor, ``min_visit_mass=0``, no holdout split.
    Teacher CLIs pass :func:`plo5bp.gto.teacher.teacher_export_kwargs`.
    """
    in_path = Path(in_path)
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    files = _export_json_files(in_path)

    loaded: list[tuple[Path, dict[str, Any], str]] = []
    skipped_expl: list[dict[str, Any]] = []
    for f in files:
        try:
            rep = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not _is_solve_report(rep):
            continue
        rid = root_id_from_report(rep, fallback=f.stem)
        if max_expl_bb is not None:
            why = expl_reject_reason(report_expl_bb(rep), max_expl_bb=max_expl_bb)
            if why is not None:
                skipped_expl.append(
                    {
                        "root_id": rid,
                        "reason": why,
                        "path": str(f),
                        "exploitability_bb": report_expl_bb(rep),
                    }
                )
                continue
        loaded.append((f, rep, rid))

    root_ids = [rid for _, _, rid in loaded]
    if float(holdout_frac) > 0.0:
        train_ids, hold_ids = split_root_ids(
            root_ids, seed=split_seed, holdout_frac=holdout_frac
        )
    else:
        train_ids, hold_ids = sorted(set(root_ids)), []
    train_set = set(train_ids)
    hold_set = set(hold_ids)

    hold_path: Path | None = None
    if hold_set:
        hold_path = (
            Path(holdout_jsonl)
            if holdout_jsonl is not None
            else out_jsonl.with_name(f"{out_jsonl.stem}_holdout.jsonl")
        )
        hold_path.parent.mkdir(parents=True, exist_ok=True)

    stats = ExportStats()
    n_train = 0
    n_hold = 0
    with out_jsonl.open("w", encoding="utf-8") as fh_train:
        fh_hold = hold_path.open("w", encoding="utf-8") if hold_path else None
        try:
            for _f, rep, rid in loaded:
                dest = "holdout" if rid in hold_set else "train"
                if dest == "holdout" and fh_hold is None:
                    dest = "train"
                fh = fh_hold if dest == "holdout" else fh_train
                for lab in strategy_to_labels(
                    rep,
                    source=source,
                    max_infosets=max_infosets_per_file,
                    require_hole=require_hole,
                    stats=stats,
                    min_visit_mass=min_visit_mass,
                ):
                    fh.write(json.dumps(lab.as_dict(), separators=(",", ":")) + "\n")
                    if dest == "holdout":
                        n_hold += 1
                    else:
                        n_train += 1
        finally:
            if fh_hold is not None:
                fh_hold.close()

    split_path: Path | None = None
    if float(holdout_frac) > 0.0 or hold_set:
        split_path = (
            Path(split_manifest)
            if split_manifest is not None
            else out_jsonl.with_name(f"{out_jsonl.stem}_split.json")
        )
        split_path.write_text(
            json.dumps(
                {
                    "split_seed": int(split_seed),
                    "holdout_frac": float(holdout_frac),
                    "max_expl_bb": max_expl_bb,
                    "min_visit_mass": float(min_visit_mass),
                    "train_root_ids": list(train_ids),
                    "holdout_root_ids": list(hold_ids),
                    "train_labels": n_train,
                    "holdout_labels": n_hold,
                    "train_path": str(out_jsonl),
                    "holdout_path": None if hold_path is None else str(hold_path),
                    "skipped_expl": skipped_expl,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if stats.n_dropped or skipped_expl:
        print(
            f"[cfr_export] kept={stats.kept} dropped={stats.n_dropped} "
            f"{stats.rejected} skipped_expl={len(skipped_expl)} "
            f"train={n_train} holdout={n_hold}",
            flush=True,
        )
    return ExportDirResult(
        n_train=n_train,
        n_holdout=n_hold,
        train_root_ids=list(train_ids),
        holdout_root_ids=list(hold_ids),
        skipped_expl=skipped_expl,
        stats=stats,
        train_path=str(out_jsonl),
        holdout_path=None if hold_path is None else str(hold_path),
        split_path=None if split_path is None else str(split_path),
    )


def export_dir(
    in_path: Path | str,
    out_jsonl: Path | str,
    *,
    source: str = "rust_cfr",
    max_infosets_per_file: int | None = None,
    require_hole: bool = False,
    max_expl_bb: float | None = None,
    min_visit_mass: float = 0.0,
    holdout_frac: float = 0.0,
    holdout_jsonl: Path | str | None = None,
    split_seed: int = TEACHER_SPLIT_SEED,
    split_manifest: Path | str | None = None,
) -> int:
    """Read strategy JSON file(s) and write LabelRecord JSONL. Returns train count.

    Default exports **all** infosets (``max_infosets_per_file=None``).
    Teacher floors are off unless ``max_expl_bb`` / ``min_visit_mass`` /
    ``holdout_frac`` are passed (see ``export_dir_detailed``).
    """
    return export_dir_detailed(
        in_path,
        out_jsonl,
        source=source,
        max_infosets_per_file=max_infosets_per_file,
        require_hole=require_hole,
        max_expl_bb=max_expl_bb,
        min_visit_mass=min_visit_mass,
        holdout_frac=holdout_frac,
        holdout_jsonl=holdout_jsonl,
        split_seed=split_seed,
        split_manifest=split_manifest,
    ).n_train


def export_teacher_dir(
    in_path: Path | str,
    out_jsonl: Path | str,
    *,
    source: str = "rust_cfr",
    max_infosets_per_file: int | None = None,
    require_hole: bool = False,
    max_expl_bb: float | None = TEACHER_MAX_EXPL_BB,
    min_visit_mass: float = TEACHER_MIN_VISIT_MASS,
    holdout_frac: float = TEACHER_HOLDOUT_FRAC,
    holdout_jsonl: Path | str | None = None,
    split_seed: int = TEACHER_SPLIT_SEED,
    split_manifest: Path | str | None = None,
) -> ExportDirResult:
    """Teacher export: expl cap + visit floor + holdout split (defaults on)."""
    return export_dir_detailed(
        in_path,
        out_jsonl,
        source=source,
        max_infosets_per_file=max_infosets_per_file,
        require_hole=require_hole,
        max_expl_bb=max_expl_bb,
        min_visit_mass=min_visit_mass,
        holdout_frac=holdout_frac,
        holdout_jsonl=holdout_jsonl,
        split_seed=split_seed,
        split_manifest=split_manifest,
    )


def export_report(
    strategy_report: dict[str, Any],
    out_jsonl: Path | str,
    **kwargs: Any,
) -> int:
    """Export one in-memory SolveReport dict to JSONL."""
    labels = strategy_to_labels(strategy_report, **kwargs)
    return write_jsonl(out_jsonl, labels)
