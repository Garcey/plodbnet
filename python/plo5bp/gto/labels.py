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

from plo5bp.sizing import NLH_ANCHOR_SPEC


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
    if trash_fold:
        gate = [0.97, 0.02, 0.01]
        actions = [
            ActionProb("fold", None, None, 0, 0.97),
            ActionProb("check_call", None, None, 0, 0.02),
            ActionProb("raise", 11, None, 500_000, 0.01),
        ]
        value = -5.0
        hole = [0, 13]  # 2c 2d — weak on wet board narrative
    else:
        gate = [0.0, 0.05, 0.95]
        actions = [
            ActionProb("fold", None, None, 0, 0.0),
            ActionProb("check_call", None, None, 0, 0.05),
            ActionProb("raise", 11, None, 500_000, 0.95),
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
