"""GTO label schema + IO for Phase 0 offline factory.

Labels are trajectory nodes: public state + hero hole + π* over the
NLH anchor menu + optional CFV/value. Stored as JSONL for day-1
portability (parquet optional later).

Native rust_cfr strategy dumps are normalized into ``LabelRecord`` so
PolicyNet training never sees solver-specific shapes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from plo5bp.sizing import NLH_ANCHOR_SPEC, anchor_grid_np


LABEL_SCHEMA_VERSION = 1


@dataclass
class ActionProb:
    """One abstract action in the shared NLH menu."""

    gate: str  # "fold" | "check_call" | "raise"
    anchor_k: int | None  # None when gate != raise; else 0..count-1
    pot_frac: float | None  # None for ALL-IN atom / non-raise
    chips: int
    prob: float


@dataclass
class LabelRecord:
    """One supervised (obs, π*) training example at a decision node."""

    schema_version: int
    source: str  # "rust_cfr" | "rust_cfr_*" | "synthetic_smoke"
    root_name: str
    num_seats: int
    street: int
    spr: float
    pot_chips: int
    to_call_chips: int
    min_raise_chips: int
    max_raise_chips: int
    hero_seat: int
    button: int
    hero_hole: list[int]  # 0..51 card ids
    board: list[int]  # 0..5 cards (preflop empty)
    stacks_chips: list[int]
    # Abstract strategy
    gate_probs: list[float]  # len 3: fold, check_call, raise
    action_probs: list[ActionProb]
    # Optional value targets
    value_bb: float | None = None
    cfv_bb: list[float] | None = None  # per-hand bucket later
    # Provenance
    solve_id: str = ""
    seed: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action_probs"] = [asdict(a) for a in self.action_probs]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LabelRecord":
        aps = [
            ActionProb(**a) if not isinstance(a, ActionProb) else a
            for a in d.get("action_probs", [])
        ]
        return cls(
            schema_version=int(d.get("schema_version", LABEL_SCHEMA_VERSION)),
            source=str(d["source"]),
            root_name=str(d["root_name"]),
            num_seats=int(d["num_seats"]),
            street=int(d["street"]),
            spr=float(d["spr"]),
            pot_chips=int(d["pot_chips"]),
            to_call_chips=int(d["to_call_chips"]),
            min_raise_chips=int(d["min_raise_chips"]),
            max_raise_chips=int(d["max_raise_chips"]),
            hero_seat=int(d["hero_seat"]),
            button=int(d["button"]),
            hero_hole=[int(x) for x in d["hero_hole"]],
            board=[int(x) for x in d.get("board", [])],
            stacks_chips=[int(x) for x in d["stacks_chips"]],
            gate_probs=[float(x) for x in d["gate_probs"]],
            action_probs=aps,
            value_bb=(
                None if d.get("value_bb") is None else float(d["value_bb"])
            ),
            cfv_bb=(
                None
                if d.get("cfv_bb") is None
                else [float(x) for x in d["cfv_bb"]]
            ),
            solve_id=str(d.get("solve_id", "")),
            seed=int(d.get("seed", 0)),
            notes=dict(d.get("notes") or {}),
        )


def normalize_gate_probs(fold: float, check_call: float, raise_: float) -> list[float]:
    """Clamp + renorm to a valid 3-simplex (numerical safety for solvers)."""
    xs = [max(0.0, float(fold)), max(0.0, float(check_call)), max(0.0, float(raise_))]
    s = sum(xs)
    if s <= 0.0:
        return [0.0, 1.0, 0.0]
    return [x / s for x in xs]


def map_size_to_anchor(
    chips: int,
    *,
    min_raise: int,
    max_raise: int,
    pot: int,
    to_call: int,
    fracs_pm: Sequence[int] | None = None,
    allin_atom: bool = True,
) -> int:
    """Nearest legal NLH anchor index for a continuous raise size.

    Used when rust_cfr solve-ladder chips / pot-frac remap onto
    the serve menu (``NLH_ANCHOR_SPEC``).
    """
    fracs = list(fracs_pm) if fracs_pm is not None else list(NLH_ANCHOR_SPEC.fracs_pm)
    mr = min(int(min_raise), int(max_raise))
    mx = int(max_raise)
    base = int(pot) + int(to_call)
    tc = int(to_call)
    best_k = 0
    best_d = 1 << 62
    # Fraction anchors
    for k, fpm in enumerate(fracs):
        c = tc + (int(fpm) * base + 500) // 1000
        c = max(mr, min(mx, c))
        d = abs(c - int(chips))
        if d < best_d or (d == best_d and k < best_k):
            best_d = d
            best_k = k
    if allin_atom:
        k_ai = len(fracs)
        d = abs(mx - int(chips))
        if d < best_d or (d == best_d and k_ai < best_k):
            best_k = k_ai
    return int(best_k)


def jam_anchor_index(
    *,
    min_raise: int,
    max_raise: int,
    pot: int,
    to_call: int,
) -> int:
    """Grid-LEGAL anchor index of the jam (chips == ``max_raise``).

    (review 2026-09-20 D1) The network's anchor grid dedupes by "strictly
    greater chips than the previous anchor", so whenever a pot-fraction anchor
    already clamps to ``max_raise`` the explicit ALL-IN atom (last index) is
    ILLEGAL and any teacher mass put there is masked out in training. The
    legal jam anchor is the FIRST anchor whose chips reach ``max_raise`` —
    exactly what :func:`map_size_to_anchor` returns for ``chips=max_raise``
    (ties break to the smaller index). In the short-shove regime
    (``min_raise == 0 < max_raise``) the grid keeps only the top atom legal,
    so that is the jam anchor there.
    """
    if int(min_raise) > 0 and int(max_raise) > 0:
        return map_size_to_anchor(
            int(max_raise),
            min_raise=int(min_raise),
            max_raise=int(max_raise),
            pot=int(pot),
            to_call=int(to_call),
        )
    return NLH_ANCHOR_SPEC.count - 1


# Teacher mass the serve grid would mask out (review 2026-09-20 D1).
ILLEGAL_MASS_TOL = 1e-6


class IllegalTeacherMassError(ValueError):
    """Teacher probability sits on an action the serve grid marks illegal.

    Training multiplies targets by the legality mask and renormalizes, so such
    mass used to vanish silently (D1: ~40% of the jam mass at SPR 2; D2: every
    push/fold call). Export, row building and training now refuse instead —
    re-export the labels with the fixed ``cfr_export``.
    """


def illegal_teacher_mass(
    action_probs: Sequence[ActionProb],
    *,
    min_raise: int,
    max_raise: int,
    pot_chips: int,
    to_call: int,
) -> float:
    """Teacher probability on actions the SERVE grid masks out at this node.

    Counts raise mass when the raise gate is illegal (``max_raise <= 0``), mass
    on grid-illegal / out-of-range anchors, and fold mass with nothing to call.
    Training masks exactly these, so anything > ``ILLEGAL_MASS_TOL`` would be
    erased by renormalization (review 2026-09-20 D1/D2).
    """
    raise_mass = sum(float(a.prob) for a in action_probs if a.gate == "raise")
    bad = 0.0
    if int(to_call) <= 0:
        bad += sum(float(a.prob) for a in action_probs if a.gate == "fold")
    if raise_mass <= 0.0:
        return bad
    if int(max_raise) <= 0:
        return bad + raise_mass
    legal = anchor_grid_np(
        int(min_raise), int(max_raise), int(pot_chips), int(to_call), NLH_ANCHOR_SPEC
    ).legal
    for a in action_probs:
        if a.gate != "raise":
            continue
        k = a.anchor_k
        if k is None or not (0 <= int(k) < len(legal)) or not bool(legal[int(k)]):
            bad += float(a.prob)
    return bad


def merge_action_probs(action_probs: Sequence[ActionProb]) -> list[ActionProb]:
    """Merge entries that land on the same ``(gate, anchor_k)`` by summing probs.

    (review 2026-09-20 D1) Two solver actions can be the same serve-menu
    action — e.g. ``RAISE_500`` clamped to the stack and ``ALLIN`` are one jam,
    and push/fold ``ALLIN``-as-call joins ``CHECK_CALL``. First-seen order is
    kept; the merged entry keeps the first entry's chips / pot_frac.
    """
    merged: dict[tuple[str, int | None], ActionProb] = {}
    for a in action_probs:
        key = (str(a.gate), None if a.anchor_k is None else int(a.anchor_k))
        prev = merged.get(key)
        if prev is None:
            merged[key] = ActionProb(
                gate=a.gate,
                anchor_k=a.anchor_k,
                pot_frac=a.pot_frac,
                chips=int(a.chips),
                prob=float(a.prob),
            )
        else:
            prev.prob = float(prev.prob) + float(a.prob)
    return list(merged.values())


def write_jsonl(path: Path | str, records: Iterable[LabelRecord]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.as_dict(), separators=(",", ":")) + "\n")
            n += 1
    return n


def read_jsonl(path: Path | str) -> Iterator[LabelRecord]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield LabelRecord.from_dict(json.loads(line))


def make_smoke_label(
    *,
    seed: int = 0,
    street: int = 3,
    trash_fold: bool = True,
) -> LabelRecord:
    """Synthetic pure-node label for factory / metrics smoke tests.

    trash_fold=True → almost-sure fold (air facing huge bet).
    trash_fold=False → almost-sure raise all-in (nuts).
    """
    # (review 2026-09-20 D1) The jam sits on the grid-LEGAL jam anchor for this
    # node's sizing (pot 20bb, facing 10bb, 50bb behind → the 160% anchor
    # already clamps to the stack, so the ALL-IN atom (11) is deduped/illegal).
    k_jam = jam_anchor_index(
        min_raise=200_000, max_raise=500_000, pot=200_000, to_call=100_000
    )
    if trash_fold:
        gate = [0.97, 0.02, 0.01]
        actions = [
            ActionProb("fold", None, None, 0, 0.97),
            ActionProb("check_call", None, None, 0, 0.02),
            ActionProb("raise", k_jam, None, 500_000, 0.01),
        ]
        value = -5.0
        hole = [0, 13]  # 2c 2d — weak on wet board narrative
    else:
        gate = [0.0, 0.05, 0.95]
        actions = [
            ActionProb("fold", None, None, 0, 0.0),
            ActionProb("check_call", None, None, 0, 0.05),
            ActionProb("raise", k_jam, None, 500_000, 0.95),
        ]
        value = 12.0
        hole = [51, 50]  # As Ah
    return LabelRecord(
        schema_version=LABEL_SCHEMA_VERSION,
        source="synthetic_smoke",
        root_name="clubgg_5_10_5",
        num_seats=2,
        street=street,
        spr=2.0,
        pot_chips=200_000,
        to_call_chips=100_000,
        min_raise_chips=200_000,
        max_raise_chips=500_000,
        hero_seat=0,
        button=1,
        hero_hole=hole,
        board=[48, 44, 40, 36, 32],  # dummy high board
        stacks_chips=[500_000, 500_000],
        gate_probs=gate,
        action_probs=actions,
        value_bb=value,
        solve_id=f"smoke_{'fold' if trash_fold else 'jam'}_{seed}",
        seed=seed,
        notes={"kind": "trash_fold" if trash_fold else "nuts_jam"},
    )
