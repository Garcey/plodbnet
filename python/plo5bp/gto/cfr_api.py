"""Python surface for the native NLH CFR solver.

Rust owns the hot loop (``plo5bp._engine.cfr_solve``). This module is the
script/batch contract: ``RootSpec`` / ``SolveConfig`` / ``solve()``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from plo5bp.gto.iso import TEACHER_USE_ISOMORPHISM
from plo5bp.gto.jsonio import atomic_write_json, jsonable
from plo5bp.gto.roots import CLUBGG_NLH_ROOT
from plo5bp.gto.teacher import root_fingerprint

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

# (TOOL-032) RAM budget for a solve's infoset table: CFR_RAM_BUDGET_MB, else 60%
# of this machine's memory (at least 1 GB), else the solver's 8 GB default.
RAM_BUDGET_ENV = "CFR_RAM_BUDGET_MB"


def physical_ram_mb() -> int | None:
    """Total physical memory in MB, or None when it cannot be read."""
    try:
        if os.name == "nt":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullTotalPhys // (1024 * 1024))
            return None
        pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        return int(pages * size // (1024 * 1024)) if pages > 0 and size > 0 else None
    except (AttributeError, OSError, ValueError):
        return None


def default_ram_budget_mb() -> int:
    """The budget new configs get (0 = the solver's own 8 GB default)."""
    raw = os.environ.get(RAM_BUDGET_ENV, "").strip()
    if raw:
        try:
            return max(64, int(float(raw)))
        except ValueError:
            pass
    total = physical_ram_mb()
    return max(1024, int(total * 0.6)) if total else 0


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
            self.root_id = self.with_fingerprint(self.legacy_auto_id())

    def legacy_auto_id(self) -> str:
        """Pre-2026-09-20 auto id. It named street / pot / stack only, so two
        boards (or size menus, seat counts, ranges) shared one id."""
        return f"s{self.street}_pot{self.pot_bb:g}_eff{self.effective_stack_bb:g}"

    def fingerprint(self) -> str:
        """Hash of the fields that DEFINE the solved game — see
        :func:`plo5bp.gto.teacher.root_fingerprint`."""
        return root_fingerprint(self.as_dict())

    def with_fingerprint(self, base: str) -> str:
        """``<base>-<fingerprint>`` — the NEW auto / grid id format.

        (review 2026-09-20 F10) The discriminator keeps different solves from
        colliding on resume markers, split manifests and label ``root_name``.
        It is only ever added to ids generated from now on: explicit ids are
        kept verbatim and batch resume still recognizes a campaign directory
        written under the bare legacy id (``cfr_batch.resolve_job_id``).
        """
        return f"{base}-{self.fingerprint()}"

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

    def _fingerprinted(self) -> "RootSpec":
        """Factory ids (``river_pot10``, ``preflop_hu_100bb`` …) never named
        the board / sizes either — append the discriminator."""
        self.root_id = self.with_fingerprint(self.root_id)
        return self

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
        )._fingerprinted()

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
        )._fingerprinted()

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
        )._fingerprinted()


@dataclass
class SolveConfig:
    # 0 = unlimited (run until stop file / time budget / pause+stop).
    max_iterations: int = 200
    target_exploitability_bb: float = 0.5
    thread_num: int = 1
    seed: int = 0
    use_isomorphism: bool = True
    # "dcfr" (chance-sampled, HU postflop), "dcfr_vector" (full ranges every
    # iteration — HU river / turn only, TOOL-008), "mccfr_es" (preflop /
    # multiway), "linear", "cfr"; the native solver validates the tag.
    algorithm: str = "dcfr"
    card_abstraction: str = "none"
    # Kill-safe continuous solve: 0 = unlimited wall clock.
    time_budget_secs: float = 0.0
    # If this path exists, solver exports average strategy and returns early.
    stop_file: str = ""
    poll_every: int = 500
    # While this path exists, solver spin-pauses (resume by deleting it).
    pause_file: str = ""
    # Live UI progress: counters every poll_every iters to <progress_file>.counters,
    # the full strategy snapshot here at most every progress_secs (TOOL-005).
    progress_file: str = ""
    progress_secs: float = 2.0
    # (TOOL-006) Stream the finished report to this file; the returned
    # SolveReport then carries only a strategy summary (+ report_path).
    report_path: str = ""
    # (TOOL-030) Check exploitability every N seconds while solving (0 = off).
    expl_check_secs: float = 0.0
    # (TOOL-032) Infoset-table budget in MB; default: see default_ram_budget_mb().
    ram_budget_mb: int = field(default_factory=default_ram_budget_mb)

    def validate(self) -> None:
        # max_iterations == 0 means unlimited — allowed HERE because callers
        # (the desktop app's SolveSession) validate the user's config first
        # and wire their own stop_file in afterwards. The native solver is
        # the guard: it rejects `max_iterations=0` with neither a time budget
        # nor a stop file at solve time (review 2026-09-20 F13).
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be >= 0 (0 = unlimited)")
        if self.thread_num < 1:
            raise ValueError("thread_num must be >= 1")
        if self.time_budget_secs < 0:
            raise ValueError("time_budget_secs must be >= 0")
        if self.poll_every < 1:
            raise ValueError("poll_every must be >= 1")
        if self.expl_check_secs < 0 or self.progress_secs < 0:
            raise ValueError("expl_check_secs and progress_secs must be >= 0")
        if self.ram_budget_mb < 0:
            raise ValueError("ram_budget_mb must be >= 0")

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
    # (TOOL-006) Set when the native solver streamed the full report to a file:
    # ``strategy`` is then only a summary {root_id, num_infosets, …}.
    report_path: str | None = None

    @property
    def streamed(self) -> bool:
        return bool(self.report_path) and bool(self.strategy.get("infosets_omitted"))

    def as_dict(self) -> dict[str, Any]:
        """The report as a dict. SHALLOW: the root / config / strategy dicts are
        shared with this object, not copied — ``dataclasses.asdict`` deep-copied
        100k+ infoset dicts per call (TOOL-006)."""
        return {
            "status": self.status,
            "root": self.root,
            "config": self.config,
            "strategy": self.strategy,
            "iterations_run": self.iterations_run,
            "exploitability_bb": self.exploitability_bb,
            "notes": self.notes,
        }

    def write_json(self, path: Path | str) -> None:
        """Atomic (temp file + rename): a kill mid-write must not leave a
        truncated strategy where a finished one is expected (review F10).
        Strict, compact JSON through the shared writer (TOOL-048 / TOOL-006);
        a streamed report is COPIED from its file (never parsed)."""
        if self.streamed:
            import shutil

            dest = Path(path)
            if dest.resolve() == Path(self.report_path).resolve():
                return
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
            try:
                shutil.copyfile(self.report_path, tmp)
                os.replace(tmp, dest)
            finally:
                tmp.unlink(missing_ok=True)
            return
        atomic_write_json(path, self.as_dict())

    def load_full(self) -> dict[str, Any]:
        """The full report dict (parses the streamed file when there is one)."""
        if self.streamed:
            import json

            with open(self.report_path, encoding="utf-8") as f:
                return json.load(f)
        return self.as_dict()

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


def solve(
    root: RootSpec,
    config: SolveConfig | None = None,
    *,
    root_extra: dict[str, str] | None = None,
) -> SolveReport:
    """Solve a root with the native (Rust) CFR solver.

    Raises ``ValueError`` for an invalid root / config (Python checks here, the
    native solver's own checks otherwise). When the extension was built without
    the solver (:func:`rust_cfr_available` is False) nothing is raised: the
    returned report has ``status="not_implemented"``, no infosets and a note
    saying to rebuild — callers check ``status`` (TOOL-050).

    With ``config.report_path`` the native solver streams the full report to
    that file (TOOL-006) and the returned report's ``strategy`` is a summary;
    ``root_extra`` then adds display-only string fields to the file's ``root``
    (the desktop app's ``range_*_text``).
    """
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
        progress_secs=float(cfg.progress_secs),
        report_path=str(cfg.report_path or ""),
        root_extra=[(str(k), str(v)) for k, v in (root_extra or {}).items()],
        expl_check_secs=float(cfg.expl_check_secs),
        ram_budget_mb=int(cfg.ram_budget_mb),
    )
    # (TOOL-006) The binding already returns plain dict / list / str / int /
    # float / bool / None values (the board is list[int]); the old `_jsonable`
    # pass re-copied every infoset for nothing.
    root_d = dict(raw["root"])
    strat_d = dict(raw["strategy"])
    cfg_d = dict(raw["config"])
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
        report_path=str(cfg.report_path) if cfg.report_path else None,
    )


def estimate_memory(root: RootSpec, config: SolveConfig | None = None) -> dict[str, Any]:
    """(TOOL-032) What a solve of ``root`` will need BEFORE it runs: estimated
    infosets / MB, public tree size, the budget, and ``refuse_reason`` when the
    solver would refuse it. Raises ValueError for an invalid root."""
    root.validate()
    cfg = config or SolveConfig()
    from plo5bp import _engine  # type: ignore

    if not hasattr(_engine, "cfr_estimate_memory"):
        raise RuntimeError("this engine build has no cfr_estimate_memory — rebuild the extension")
    kwargs: dict[str, Any] = dict(
        street=int(root.street),
        pot_bb=float(root.pot_bb),
        effective_stack_bb=float(root.effective_stack_bb),
        board=[int(c) for c in root.board],
        raise_sizes_pm=[int(x) for x in root.raise_sizes_pm],
        max_iterations=int(cfg.max_iterations),
        thread_num=int(cfg.thread_num),
        card_abstraction=str(cfg.card_abstraction),
        num_seats=int(root.num_seats),
        bb_chips=int(root.bb_chips),
        sb_chips=int(root.sb_chips),
        ante_chips=int(root.ante_chips),
        allin_atom=bool(root.allin_atom),
        stacks_bb=[float(x) for x in root.stacks_bb],
        ram_budget_mb=int(cfg.ram_budget_mb),
        time_budget_secs=float(cfg.time_budget_secs),
        # (TOOL-008) the full-range solver builds its whole tree up front, and
        # its report holds one row per hand in the ranges.
        algorithm=str(cfg.algorithm),
        range_oop=str(root.range_oop or ""),
        range_ip=str(root.range_ip or ""),
    )
    try:
        return dict(_engine.cfr_estimate_memory(**kwargs))
    except TypeError:
        # An engine built before these keywords: the sampled estimate.
        for key in ("algorithm", "range_oop", "range_ip"):
            kwargs.pop(key)
        return dict(_engine.cfr_estimate_memory(**kwargs))


def native_parse_range(spec: str, board: Sequence[int] = ()) -> list[float]:
    """(TOOL-029) The native parser's 1326 combo weights for ``spec``."""
    from plo5bp import _engine  # type: ignore

    return list(_engine.cfr_parse_range(str(spec), [int(c) for c in board]))


_jsonable = jsonable  # moved to plo5bp.gto.jsonio (TOOL-048)


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
