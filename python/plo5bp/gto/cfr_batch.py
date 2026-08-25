"""Batch orchestration for native CFR solves (ops wedge)."""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from plo5bp.gto.cfr_api import (
    DEFAULT_RAISE_SIZES_PM,
    SIZE_PRESETS,
    RootSpec,
    SolveConfig,
    apply_teacher_iso_policy,
    solve,
)
from plo5bp.gto.iso import TEACHER_USE_ISOMORPHISM
from plo5bp.gto.teacher import TEACHER_MAX_EXPL_BB, expl_reject_reason


@dataclass
class BatchJob:
    """One root in a batch grid."""

    root: RootSpec
    config: SolveConfig
    job_id: str = ""

    def __post_init__(self) -> None:
        if not self.job_id:
            self.job_id = self.root.root_id


@dataclass
class BatchManifest:
    out_dir: str
    jobs: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BatchManifest":
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            out_dir=str(d["out_dir"]),
            jobs=list(d.get("jobs") or []),
            completed=list(d.get("completed") or []),
            failed=list(d.get("failed") or []),
            skipped=list(d.get("skipped") or []),
            rejected=list(d.get("rejected") or []),
            started_at=float(d.get("started_at") or 0.0),
            finished_at=float(d.get("finished_at") or 0.0),
        )


def expand_river_grid(
    *,
    n_roots: int,
    seed: int = 0,
    pot_bb: float = 10.0,
    stack_bb: float = 50.0,
    size_preset: str = "coarse",
    iters: int = 200,
    streets: Sequence[int] | None = None,
) -> list[BatchJob]:
    """Deterministic postflop board grid (river default; flop/turn supported)."""
    import random

    rng = random.Random(seed)
    sizes = list(SIZE_PRESETS.get(size_preset, DEFAULT_RAISE_SIZES_PM))
    street_list = list(streets) if streets else [3]
    jobs: list[BatchJob] = []
    need = {1: 3, 2: 4, 3: 5}
    idx = 0
    for street in street_list:
        n_cards = need.get(int(street), 5)
        for i in range(n_roots):
            deck = list(range(52))
            rng.shuffle(deck)
            board = sorted(deck[:n_cards])
            root = RootSpec(
                street=int(street),
                pot_bb=pot_bb,
                effective_stack_bb=stack_bb,
                board=board,
                raise_sizes_pm=sizes,
                root_id=f"s{street}_s{seed}_i{i}",
            )
            abs_ = "ochs" if street == 1 else "none"
            cfg = SolveConfig.teacher(
                max_iterations=iters,
                seed=seed + idx,
                card_abstraction=abs_,
            )
            jobs.append(BatchJob(root=root, config=cfg, job_id=root.root_id))
            idx += 1
    return jobs


def expand_river_spr_grid(
    *,
    n_boards: int = 6,
    seed: int = 7,
    pot_bb: float = 10.0,
    spr_points: Sequence[float] = (1.0, 2.0, 3.0, 5.0),
    size_preset: str = "micro",
    iters: int = 20_000,
) -> list[BatchJob]:
    """HU river boards × SPR points (not the overnight full-hand grid)."""
    import random

    rng = random.Random(seed)
    sizes = list(SIZE_PRESETS.get(size_preset, DEFAULT_RAISE_SIZES_PM))
    jobs: list[BatchJob] = []
    idx = 0
    for spr in spr_points:
        stack_bb = float(spr) * float(pot_bb)
        spr_tag = str(spr).replace(".", "p")
        for i in range(int(n_boards)):
            deck = list(range(52))
            rng.shuffle(deck)
            board = sorted(deck[:5])
            root = RootSpec(
                street=3,
                pot_bb=float(pot_bb),
                effective_stack_bb=stack_bb,
                board=board,
                raise_sizes_pm=sizes,
                root_id=f"s3_spr{spr_tag}_s{seed}_b{i}",
            )
            cfg = SolveConfig.teacher(
                max_iterations=iters,
                seed=seed + idx + 1,
                card_abstraction="none",
            )
            jobs.append(BatchJob(root=root, config=cfg, job_id=root.root_id))
            idx += 1
    return jobs


def _strategy_path(out_dir: Path, job_id: str) -> Path:
    return out_dir / "strategies" / f"{job_id}.json"


def _marker_path(out_dir: Path, job_id: str) -> Path:
    return out_dir / "markers" / f"{job_id}.done"


def _rejected_path(out_dir: Path, job_id: str) -> Path:
    return out_dir / "rejected" / f"{job_id}.json"


def _write_marker(out_dir: Path, job_id: str, text: str) -> None:
    marker = _marker_path(out_dir, job_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(text, encoding="utf-8")


def _run_one(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker entry (picklable)."""
    root = RootSpec(**payload["root"])
    cfg = SolveConfig(**payload["config"])
    out_dir = Path(payload["out_dir"])
    job_id = payload["job_id"]
    max_expl_bb = payload.get("max_expl_bb", TEACHER_MAX_EXPL_BB)
    try:
        rep = solve(root, cfg)
        if rep.status != "ok":
            return {
                "job_id": job_id,
                "status": rep.status,
                "ok": False,
                "error": rep.status,
                "iterations": rep.iterations_run,
                "exploitability_bb": rep.exploitability_bb,
            }
        why = None
        if max_expl_bb is not None:
            why = expl_reject_reason(
                rep.exploitability_bb, max_expl_bb=float(max_expl_bb)
            )
        if why is not None:
            rpath = _rejected_path(out_dir, job_id)
            rpath.parent.mkdir(parents=True, exist_ok=True)
            tmp_r = rpath.with_suffix(".tmp")
            tmp_r.write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "reason": why,
                        "status": "rejected",
                        "exploitability_bb": rep.exploitability_bb,
                        "max_expl_bb": max_expl_bb,
                        "report": rep.as_dict(),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            tmp_r.replace(rpath)
            _write_marker(out_dir, job_id, "rejected\n")
            return {
                "job_id": job_id,
                "status": "rejected",
                "ok": False,
                "rejected": True,
                "error": why,
                "iterations": rep.iterations_run,
                "exploitability_bb": rep.exploitability_bb,
            }
        path = _strategy_path(out_dir, job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rep.as_dict(), indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        _write_marker(out_dir, job_id, "ok\n")
        return {
            "job_id": job_id,
            "status": rep.status,
            "ok": True,
            "iterations": rep.iterations_run,
            "exploitability_bb": rep.exploitability_bb,
        }
    except Exception as e:
        return {"job_id": job_id, "status": "error", "ok": False, "error": str(e)}


def _record_worker_result(manifest: BatchManifest, r: dict[str, Any]) -> None:
    if r.get("rejected"):
        manifest.rejected.append(
            {"job_id": r["job_id"], "error": str(r.get("error", "rejected"))}
        )
    elif r.get("ok"):
        manifest.completed.append(r["job_id"])
    else:
        manifest.failed.append(
            {"job_id": r["job_id"], "error": r.get("error", r.get("status", "?"))}
        )


def run_batch(
    jobs: Sequence[BatchJob],
    out_dir: Path | str,
    *,
    workers: int = 1,
    resume: bool = True,
    dry_run: bool = False,
    max_expl_bb: float | None = TEACHER_MAX_EXPL_BB,
) -> BatchManifest:
    """Run (or dry-run) a batch of CFR solves with resume markers.

    Roots whose ``exploitability_bb`` is missing or above ``max_expl_bb``
    are written to ``rejected/`` (not ``strategies/``) and listed on
    ``manifest.rejected``. Pass ``max_expl_bb=None`` to disable the floor
    (verify / resume-mechanics tests).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "strategies").mkdir(exist_ok=True)
    (out_dir / "markers").mkdir(exist_ok=True)
    (out_dir / "rejected").mkdir(exist_ok=True)

    manifest = BatchManifest(
        out_dir=str(out_dir),
        jobs=[j.job_id for j in jobs],
        started_at=time.time(),
    )

    pending: list[BatchJob] = []
    for j in jobs:
        apply_teacher_iso_policy(j.config)
        if resume and _marker_path(out_dir, j.job_id).exists():
            manifest.skipped.append(j.job_id)
        else:
            pending.append(j)

    plan_path = out_dir / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "n_jobs": len(jobs),
                "pending": [j.job_id for j in pending],
                "skipped": manifest.skipped,
                "dry_run": dry_run,
                "use_isomorphism": TEACHER_USE_ISOMORPHISM,
                "max_expl_bb": max_expl_bb,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if dry_run:
        manifest.finished_at = time.time()
        manifest.write(out_dir / "manifest.json")
        return manifest

    payloads = [
        {
            "root": j.root.as_dict(),
            "config": j.config.as_dict(),
            "out_dir": str(out_dir),
            "job_id": j.job_id,
            "max_expl_bb": max_expl_bb,
        }
        for j in pending
    ]

    if workers <= 1:
        for p in payloads:
            _record_worker_result(manifest, _run_one(p))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_one, p): p["job_id"] for p in payloads}
            for fut in as_completed(futs):
                _record_worker_result(manifest, fut.result())

    manifest.finished_at = time.time()
    manifest.write(out_dir / "manifest.json")
    return manifest


def iter_strategy_files(dir_path: Path | str) -> Iterator[Path]:
    p = Path(dir_path)
    if p.is_file():
        yield p
        return
    yield from sorted(p.glob("**/*.json"))
