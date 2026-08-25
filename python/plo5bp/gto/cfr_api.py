"""Python surface for the native NLH CFR solver.

Rust owns the hot loop (``plo5bp._engine.cfr_solve``). This module is the
script/batch contract: ``RootSpec`` / ``SolveConfig`` / ``solve()``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from plo5bp.gto.iso import TEACHER_USE_ISOMORPHISM
from plo5bp.gto.roots import CLUBGG_NLH_ROOT

# Per-mille pot fractions
DEFAULT_RAISE_SIZES_PM: tuple[int, ...] = (330, 500, 750, 1000, 1500)

SIZE_PRESETS: dict[str, tuple[int, ...]] = {
    "micro": (500, 1000),
    "coarse": (330, 500, 1000, 1500),
    "standard": (330, 500, 750, 1000, 1500),
    "fine": (250, 330, 500, 750, 1000, 1250, 1500),
}

STREET_PREFLOP = 0
STREET_FLOP = 1
STREET_TURN = 2
STREET_RIVER = 3


@dataclass
class RootSpec:
    """One CFR subgame root (HU or multiway 2..6)."""

    street: int  # 0..3
    pot_bb: float
    effective_stack_bb: float
    board: list[int] = field(default_factory=list)  # 0..51
    num_seats: int = 2
    bb_chips: int = CLUBGG_NLH_ROOT.bb
    sb_chips: int = CLUBGG_NLH_ROOT.sb
    ante_chips: int = CLUBGG_NLH_ROOT.ante
    raise_sizes_pm: list[int] = field(
        default_factory=lambda: list(DEFAULT_RAISE_SIZES_PM)
    )
    allin_atom: bool = True
    range_ip: str = ""
    range_oop: str = ""
    stacks_bb: list[float] = field(default_factory=list)
    root_id: str = ""

    def __post_init__(self) -> None:
        if not self.root_id:
            self.root_id = f"s{self.street}_pot{self.pot_bb:g}_eff{self.effective_stack_bb:g}"

    def validate(self) -> None:
        if not 2 <= self.num_seats <= 6:
            raise ValueError("num_seats must be 2..=6")
        if self.street not in (0, 1, 2, 3):
            raise ValueError(f"street must be 0..3, got {self.street}")
        need = {0: 0, 1: 3, 2: 4, 3: 5}[self.street]
        if len(self.board) != need:
            raise ValueError(
                f"board length {len(self.board)} != {need} for street {self.street}"
            )
        if self.pot_bb <= 0 or self.effective_stack_bb <= 0:
            raise ValueError("pot_bb and effective_stack_bb must be > 0")
        if len(set(self.board)) != len(self.board):
            raise ValueError("duplicate board cards")
        for c in self.board:
            if not 0 <= int(c) < 52:
                raise ValueError(f"card {c} out of 0..51")
        if not self.raise_sizes_pm and not self.allin_atom:
            raise ValueError(
                "need at least one raise size or allin_atom=True "
                "(empty raise_sizes_pm + allin_atom is pure push/fold)"
            )
        if self.stacks_bb:
            if len(self.stacks_bb) != self.num_seats:
                raise ValueError(
                    f"stacks_bb length {len(self.stacks_bb)} != num_seats {self.num_seats}"
                )
            for i, s in enumerate(self.stacks_bb):
                if float(s) <= 0:
                    raise ValueError(f"stacks_bb[{i}] must be > 0, got {s}")
        # Footgun note: ClubGG default ante is 0.5bb. AoF / no-ante spots must
        # set ante_chips=0 explicitly (multiway preflop pot = n*ante+sb+bb).
        if self.street == STREET_PREFLOP and self.num_seats > 2:
            # pot_bb is ignored by multiway preflop MCCFR (rebuilt from blinds).
            pass

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def preflop_hu(cls, stack_bb: float = 100.0) -> "RootSpec":
        bb = CLUBGG_NLH_ROOT.bb
        pot_bb = (
            CLUBGG_NLH_ROOT.sb + CLUBGG_NLH_ROOT.bb + 2 * CLUBGG_NLH_ROOT.ante
        ) / float(bb)
        return cls(
            street=STREET_PREFLOP,
            pot_bb=pot_bb,
            effective_stack_bb=float(stack_bb),
            board=[],
            root_id=f"preflop_hu_{stack_bb:g}bb",
        )

    @classmethod
    def preflop_pushfold(
        cls,
        *,
        num_seats: int = 4,
        stack_bb: float = 10.0,
        bb_chips: int | None = None,
        sb_chips: int | None = None,
        ante_chips: int = 0,
    ) -> "RootSpec":
        """Pure push/fold multiway preflop root (empty raise menu + all-in).

        Defaults to **no ante** (AoF / short-stack tournaments). ClubGG ante
        is *not* applied — pass ``ante_chips=CLUBGG_NLH_ROOT.ante`` explicitly
        if you want table ante.
        """
        bb = int(bb_chips if bb_chips is not None else CLUBGG_NLH_ROOT.bb)
        sb = int(sb_chips if sb_chips is not None else CLUBGG_NLH_ROOT.sb)
        n = int(num_seats)
        pot_chips = n * int(ante_chips) + sb + bb
        pot_bb = pot_chips / float(bb)
        return cls(
            street=STREET_PREFLOP,
            pot_bb=pot_bb,
            effective_stack_bb=float(stack_bb),
            board=[],
            num_seats=n,
            bb_chips=bb,
            sb_chips=sb,
            ante_chips=int(ante_chips),
            raise_sizes_pm=[],
            allin_atom=True,
            stacks_bb=[float(stack_bb)] * n,
            root_id=f"pushfold_{n}h_{stack_bb:g}bb_ante{ante_chips}",
        )

    @classmethod
    def river_hu(
        cls,
        board: Sequence[int],
        *,
        pot_bb: float = 10.0,
        effective_stack_bb: float = 50.0,
        size_preset: str = "standard",
    ) -> "RootSpec":
        sizes = list(SIZE_PRESETS.get(size_preset, DEFAULT_RAISE_SIZES_PM))
        return cls(
            street=STREET_RIVER,
            pot_bb=float(pot_bb),
            effective_stack_bb=float(effective_stack_bb),
            board=[int(c) for c in board],
            raise_sizes_pm=sizes,
            root_id=f"river_pot{pot_bb:g}",
        )


@dataclass
class SolveConfig:
    # 0 = unlimited (run until stop file / time budget / pause+stop).
    max_iterations: int = 200
    target_exploitability_bb: float = 0.5
    thread_num: int = 1
    seed: int = 0
    use_isomorphism: bool = True
    algorithm: str = "dcfr"
    card_abstraction: str = "none"
    # Kill-safe continuous solve: 0 = unlimited wall clock.
    time_budget_secs: float = 0.0
    # If this path exists, solver exports average strategy and returns early.
    stop_file: str = ""
    poll_every: int = 500
    # While this path exists, solver spin-pauses (resume by deleting it).
    pause_file: str = ""
    # Partial SolveReport JSON written every poll_every iters for live UI.
    progress_file: str = ""

    def validate(self) -> None:
        # max_iterations == 0 means unlimited — allowed.
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be >= 0 (0 = unlimited)")
        if self.thread_num < 1:
            raise ValueError("thread_num must be >= 1")
        if self.time_budget_secs < 0:
            raise ValueError("time_budget_secs must be >= 0")
        if self.poll_every < 1:
            raise ValueError("poll_every must be >= 1")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def teacher(cls, **kwargs: Any) -> "SolveConfig":
        """SolveConfig for PolicyNet teacher batches (iso OFF — see iso.py)."""
        kwargs.setdefault("use_isomorphism", TEACHER_USE_ISOMORPHISM)
        return cls(**kwargs)


def apply_teacher_iso_policy(cfg: SolveConfig) -> SolveConfig:
    """Force the v1 teacher iso policy on a config (in-place + return)."""
    cfg.use_isomorphism = TEACHER_USE_ISOMORPHISM
    return cfg


@dataclass
class SolveReport:
    status: str
    root: dict[str, Any]
    config: dict[str, Any]
    strategy: dict[str, Any]
    iterations_run: int
    exploitability_bb: float | None
    notes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")

    @property
    def is_mc_br_proxy(self) -> bool:
        """True when exploitability_bb is a Monte-Carlo BR proxy (multiway), not a tight Nash cert."""
        return any("mc_br_proxy" in n for n in self.notes)


def rust_cfr_available() -> bool:
    """True when the extension exposes a real CFR solve."""
    try:
        from plo5bp import _engine  # type: ignore

        return hasattr(_engine, "cfr_solve")
    except Exception:
        return False


def solve(root: RootSpec, config: SolveConfig | None = None) -> SolveReport:
    """Solve a root via Rust CFR when available; else raise."""
    root.validate()
    cfg = config or SolveConfig()
    cfg.validate()

    if not rust_cfr_available():
        return SolveReport(
            status="not_implemented",
            root=root.as_dict(),
            config=cfg.as_dict(),
            strategy={"root_id": root.root_id, "infosets": []},
            iterations_run=0,
            exploitability_bb=None,
            notes=[
                "Rust CFR binding not available — rebuild with maturin develop",
            ],
        )

    from plo5bp import _engine  # type: ignore

    raw = _engine.cfr_solve(
        street=int(root.street),
        pot_bb=float(root.pot_bb),
        effective_stack_bb=float(root.effective_stack_bb),
        board=[int(c) for c in root.board],
        raise_sizes_pm=[int(x) for x in root.raise_sizes_pm],
        max_iterations=int(cfg.max_iterations),
        target_exploitability_bb=float(cfg.target_exploitability_bb),
        thread_num=int(cfg.thread_num),
        seed=int(cfg.seed),
        algorithm=str(cfg.algorithm),
        card_abstraction=str(cfg.card_abstraction),
        num_seats=int(root.num_seats),
        bb_chips=int(root.bb_chips),
        sb_chips=int(root.sb_chips),
        ante_chips=int(root.ante_chips),
        allin_atom=bool(root.allin_atom),
        range_ip=str(root.range_ip),
        range_oop=str(root.range_oop),
        root_id=str(root.root_id),
        use_isomorphism=bool(cfg.use_isomorphism),
        stacks_bb=[float(x) for x in root.stacks_bb],
        time_budget_secs=float(cfg.time_budget_secs),
        stop_file=str(cfg.stop_file or ""),
        poll_every=int(cfg.poll_every),
        pause_file=str(cfg.pause_file or ""),
        progress_file=str(cfg.progress_file or ""),
    )
    root_d = _jsonable(dict(raw["root"]))
    strat_d = _jsonable(dict(raw["strategy"]))
    cfg_d = _jsonable(dict(raw["config"]))
    return SolveReport(
        status=str(raw["status"]),
        root=root_d,
        config=cfg_d,
        strategy=strat_d,
        iterations_run=int(raw["iterations_run"]),
        exploitability_bb=(
            None
            if raw["exploitability_bb"] is None
            else float(raw["exploitability_bb"])
        ),
        notes=[str(n) for n in (raw.get("notes") or [])],
    )


def _jsonable(obj: Any) -> Any:
    """Coerce PyO3 / numpy leftovers into JSON-friendly Python types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, (bytes, bytearray)):
        return list(obj)
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    # Path / numpy scalar etc.
    try:
        return obj.item()  # type: ignore[attr-defined]
    except Exception:
        return str(obj)


def solve_kuhn(iterations: int = 5000) -> dict[str, Any]:
    """Kuhn poker correctness gate via Rust."""
    if not rust_cfr_available():
        raise RuntimeError("cfr_solve_kuhn not available")
    from plo5bp import _engine  # type: ignore

    return dict(_engine.cfr_solve_kuhn(iterations=int(iterations)))


def induce_range(
    prior: Sequence[float],
    action_probs_by_class: Sequence[Sequence[float]],
    action_idx: int,
) -> list[float]:
    """Bayesian range update after observing an abstract action."""
    if rust_cfr_available():
        from plo5bp import _engine  # type: ignore

        return list(
            _engine.cfr_induce_range(
                list(prior),
                [list(row) for row in action_probs_by_class],
                int(action_idx),
            )
        )
    # Pure Python fallback
    out = []
    total = 0.0
    for i, w in enumerate(prior):
        p = float(action_probs_by_class[i][action_idx]) if i < len(action_probs_by_class) else 0.0
        v = float(w) * p
        out.append(v)
        total += v
    if total > 0:
        out = [x / total for x in out]
    return out
