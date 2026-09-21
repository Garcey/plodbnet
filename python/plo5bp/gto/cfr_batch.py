"""Batch orchestration for native CFR solves (ops wedge).

Review 2026-09-20:

- **D8 / F6** A root is judged against the teacher cap only when its
  ``exploitability_bb`` comes from a FINAL estimator
  (:func:`plo5bp.gto.teacher.expl_provenance`). Poll / early-stop /
  time-budget numbers are biased (5.27 polled vs 1.99 final), so such a root
  is neither accepted nor rejected: it is written to ``unverified/`` with
  marker ``unverified`` and teacher export skips it. There is no binding that
  re-evaluates a finished strategy, so nothing can be re-verified here. A
  solve cut short by the STOP file gets NO done-marker — resume re-runs it.
- **F10** New grid ids carry a game fingerprint (board / pot / stacks / sizes),
  a changed solve config is not silently "resumed", and manifests are written
  atomically. Campaign directories written under the old bare ids still
  resume (:func:`resolve_job_id`).

  ID FORMAT. ``<legacy id>-<8 hex>``, e.g. ``s3_spr2p0_s7_b3-1a2b3c4d``. The
  suffix is :func:`plo5bp.gto.teacher.root_fingerprint` — a hash of the fields
  that define the GAME (street, board, pot, stacks, seats, blinds/ante, size
  menu, all-in atom, ranges). The solve config (iterations, algorithm,
  abstraction, seed) is deliberately NOT in the id: the id is also the label
  ``root_name`` and the key of the train/holdout split, and the same root must
  land on the same side however long it was solved. A different iteration
  budget / algorithm is caught at resume time instead
  (:func:`stale_resume_reason`, from the ``markers/<id>.job.json`` sidecar or
  the existing report's ``config``). Ids passed explicitly are kept verbatim.
"""

from __future__ import annotations

import json
import os
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
from plo5bp.gto.teacher import (
    TEACHER_MAX_EXPL_BB,
    expl_provenance,
    expl_reject_reason,
    root_fingerprint,
)

MARKER_OK = "ok"
MARKER_REJECTED = "rejected"
MARKER_UNVERIFIED = "unverified"


@dataclass
class BatchJob:
    """One root in a batch grid."""

    root: RootSpec
    config: SolveConfig
    job_id: str = ""
    # Pre-2026-09-20 id of the same grid cell (no game fingerprint). Only used
    # to keep resuming campaign dirs written under it — see resolve_job_id.
    legacy_job_id: str = ""

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
    # Solved, but the exploitability number is not a final estimate (D8).
    unverified: list[dict[str, str]] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0

    def write(self, path: Path) -> None:
        _atomic_write_text(path, json.dumps(asdict(self), indent=2) + "\n")

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
            unverified=list(d.get("unverified") or []),
            started_at=float(d.get("started_at") or 0.0),
            finished_at=float(d.get("finished_at") or 0.0),
        )


def _atomic_write_text(path: Path, text: str) -> None:
    """Temp file + ``os.replace``: a kill mid-write never leaves a torn or
    empty manifest for the next resume to choke on (review 2026-09-20 F10)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _grid_job(root: RootSpec, cfg: SolveConfig, base_id: str) -> BatchJob:
    """Grid cell with the NEW fingerprinted id (legacy id kept for resume)."""
    root.root_id = root.with_fingerprint(base_id)
    return BatchJob(root=root, config=cfg, job_id=root.root_id, legacy_job_id=base_id)


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
            base_id = f"s{street}_s{seed}_i{i}"
            root = RootSpec(
                street=int(street),
                pot_bb=pot_bb,
                effective_stack_bb=stack_bb,
                board=board,
                raise_sizes_pm=sizes,
                root_id=base_id,
            )
            abs_ = "ochs" if street == 1 else "none"
            cfg = SolveConfig.teacher(
                max_iterations=iters,
                seed=seed + idx,
                card_abstraction=abs_,
            )
            jobs.append(_grid_job(root, cfg, base_id))
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
            base_id = f"s3_spr{spr_tag}_s{seed}_b{i}"
            root = RootSpec(
                street=3,
                pot_bb=float(pot_bb),
                effective_stack_bb=stack_bb,
                board=board,
                raise_sizes_pm=sizes,
                root_id=base_id,
            )
            cfg = SolveConfig.teacher(
                max_iterations=iters,
                seed=seed + idx + 1,
                card_abstraction="none",
            )
            jobs.append(_grid_job(root, cfg, base_id))
            idx += 1
    return jobs


def _strategy_path(out_dir: Path, job_id: str) -> Path:
    return out_dir / "strategies" / f"{job_id}.json"


def _marker_path(out_dir: Path, job_id: str) -> Path:
    return out_dir / "markers" / f"{job_id}.done"


def _job_meta_path(out_dir: Path, job_id: str) -> Path:
    """Sidecar beside the marker: what was solved, under which config."""
    return out_dir / "markers" / f"{job_id}.job.json"


def _rejected_path(out_dir: Path, job_id: str) -> Path:
    return out_dir / "rejected" / f"{job_id}.json"


def _unverified_path(out_dir: Path, job_id: str) -> Path:
    return out_dir / "unverified" / f"{job_id}.json"


def marker_status(out_dir: Path, job_id: str) -> str | None:
    """First word of the done-marker (``ok`` / ``rejected`` / ``unverified``)."""
    p = _marker_path(out_dir, job_id)
    if not p.exists():
        return None
    words = p.read_text(encoding="utf-8").split()
    return words[0] if words else ""


def _write_marker(out_dir: Path, job_id: str, text: str) -> None:
    marker = _marker_path(out_dir, job_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(text, encoding="utf-8")


def _json_object_at(text: str, key: str) -> dict[str, Any] | None:
    """The JSON object following ``"key":`` in ``text`` (brace matching)."""
    at = text.find(f'"{key}"')
    if at < 0:
        return None
    start = text.find("{", at)
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except ValueError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def _report_head(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """``(root, config)`` of a report WITHOUT parsing its strategy body.

    Our writers emit ``status, root, config, strategy, …`` in that order, so
    both objects sit in the first kilobytes; fall back to a full parse.
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            head = fh.read(65536)
    except OSError:
        return None, None
    root, cfg = _json_object_at(head, "root"), _json_object_at(head, "config")
    if root is not None:
        return root, cfg
    try:
        rep = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    rep = rep.get("report", rep) if isinstance(rep, dict) else {}
    return rep.get("root"), rep.get("config")


def _existing_artifact(out_dir: Path, job_id: str) -> Path | None:
    for fn in (_strategy_path, _rejected_path, _unverified_path):
        p = fn(out_dir, job_id)
        if p.exists():
            return p
    return None


def resolve_job_id(out_dir: Path | str, job: BatchJob) -> str:
    """Pick the id this job lives under in ``out_dir`` (and apply it).

    (review 2026-09-20 F10) New ids are ``<legacy>-<game fingerprint>``.
    Campaign directories written before that hold markers / strategies under
    the bare legacy id; when such an artifact exists AND describes the same
    game (board, pot, stacks, size menu …) the job keeps the legacy id so the
    directory still resumes. A legacy artifact for a DIFFERENT game is exactly
    the collision the fingerprint exists to prevent — the job then uses its new
    id and is solved fresh instead of being skipped.
    """
    out_dir = Path(out_dir)
    legacy = job.legacy_job_id
    if not legacy or legacy == job.job_id:
        return job.job_id
    if _marker_path(out_dir, job.job_id).exists() or _existing_artifact(
        out_dir, job.job_id
    ):
        return job.job_id
    art = _existing_artifact(out_dir, legacy)
    if art is None and not _marker_path(out_dir, legacy).exists():
        return job.job_id
    if art is not None:
        root, _cfg = _report_head(art)
        if root is not None and root_fingerprint(root) != job.root.fingerprint():
            return job.job_id  # same legacy id, different game
    job.job_id = legacy
    job.root.root_id = legacy
    return legacy


def resolve_job_ids(out_dir: Path | str, jobs: Sequence[BatchJob]) -> list[str]:
    return [resolve_job_id(out_dir, j) for j in jobs]


def _iters(x: Any) -> float:
    """``max_iterations`` with 0 == unlimited."""
    try:
        v = int(x)
    except (TypeError, ValueError):
        return float("inf")
    return float("inf") if v <= 0 else float(v)


def stale_resume_reason(out_dir: Path, job: BatchJob) -> str | None:
    """Why an existing done-marker does NOT cover this job (None = it does).

    (review 2026-09-20 F10) ``--iters 100`` then ``--iters 20000`` into the
    same directory used to skip every root and leave 100-iteration solves
    behind. A marker now covers a job only if the finished solve used the same
    algorithm / abstraction and at least as many iterations as requested.
    Unknown (bare legacy marker, no report) keeps the old behaviour: covered.
    """
    done: dict[str, Any] | None = None
    meta = _job_meta_path(out_dir, job.job_id)
    if meta.exists():
        try:
            done = json.loads(meta.read_text(encoding="utf-8")).get("config")
        except (OSError, ValueError):
            done = None
    if done is None:
        art = _existing_artifact(out_dir, job.job_id)
        if art is not None:
            _root, done = _report_head(art)
    if not isinstance(done, dict) or "max_iterations" not in done:
        return None
    for key in ("algorithm", "card_abstraction"):
        want, got = getattr(job.config, key), done.get(key)
        if got is not None and str(got) != str(want):
            return f"{key} {got!r} != requested {want!r}"
    if _iters(done.get("max_iterations")) < _iters(job.config.max_iterations):
        return (
            f"max_iterations {done.get('max_iterations')} < requested "
            f"{job.config.max_iterations}"
        )
    return None


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


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
        prov = expl_provenance(rep)
        base = {
            "job_id": job_id,
            "iterations": rep.iterations_run,
            "exploitability_bb": rep.exploitability_bb,
            "expl": prov.as_dict(),
        }

        def _finish(status: str, path: Path, body: dict[str, Any], **extra: Any):
            _write_report(path, body)
            _write_report(
                _job_meta_path(out_dir, job_id),
                {
                    **base,
                    "status": status,
                    "fingerprint": root.fingerprint(),
                    "config": cfg.as_dict(),
                },
            )
            _write_marker(out_dir, job_id, f"{status}\n")
            return {**base, "status": status, **extra}

        # (review 2026-09-20 D8) Never accept OR reject on a poll / early-stop
        # / time-budget estimate. ``max_expl_bb=None`` disables every floor.
        if max_expl_bb is not None and not prov.verified:
            if prov.early_stop == "stop_file":
                # Interrupted, not finished: keep the partial for inspection
                # but write NO done-marker so resume solves it again.
                _write_report(_unverified_path(out_dir, job_id), rep.as_dict())
                return {
                    **base,
                    "status": "interrupted",
                    "ok": False,
                    "unverified": True,
                    "interrupted": True,
                    "error": f"expl_unverified:{prov.reason}",
                }
            return _finish(
                MARKER_UNVERIFIED,
                _unverified_path(out_dir, job_id),
                rep.as_dict(),
                ok=False,
                unverified=True,
                error=f"expl_unverified:{prov.reason}",
            )
        why = None
        if max_expl_bb is not None:
            why = expl_reject_reason(
                rep.exploitability_bb, max_expl_bb=float(max_expl_bb)
            )
        if why is not None:
            return _finish(
                MARKER_REJECTED,
                _rejected_path(out_dir, job_id),
                {
                    "job_id": job_id,
                    "reason": why,
                    "status": "rejected",
                    "exploitability_bb": rep.exploitability_bb,
                    "max_expl_bb": max_expl_bb,
                    "report": rep.as_dict(),
                },
                ok=False,
                rejected=True,
                error=why,
            )
        return _finish(
            MARKER_OK, _strategy_path(out_dir, job_id), rep.as_dict(), ok=True
        )
    except Exception as e:
        return {"job_id": job_id, "status": "error", "ok": False, "error": str(e)}


def _record_worker_result(manifest: BatchManifest, r: dict[str, Any]) -> None:
    if r.get("unverified"):
        manifest.unverified.append(
            {
                "job_id": r["job_id"],
                "error": str(r.get("error", "expl_unverified")),
                "status": str(r.get("status")),
            }
        )
    elif r.get("rejected"):
        manifest.rejected.append(
            {"job_id": r["job_id"], "error": str(r.get("error", "rejected"))}
        )
    elif r.get("ok"):
        manifest.completed.append(r["job_id"])
    else:
        manifest.failed.append(
            {"job_id": r["job_id"], "error": r.get("error", r.get("status", "?"))}
        )


def job_payload(
    job: BatchJob, out_dir: Path | str, max_expl_bb: float | None
) -> dict[str, Any]:
    return {
        "root": job.root.as_dict(),
        "config": job.config.as_dict(),
        "out_dir": str(out_dir),
        "job_id": job.job_id,
        "max_expl_bb": max_expl_bb,
    }


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

    Roots whose VERIFIED ``exploitability_bb`` is missing or above
    ``max_expl_bb`` are written to ``rejected/`` (not ``strategies/``) and
    listed on ``manifest.rejected``; roots whose number is not a final
    estimate go to ``unverified/`` / ``manifest.unverified``. Pass
    ``max_expl_bb=None`` to disable the floors (verify / resume-mechanics
    tests).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("strategies", "markers", "rejected", "unverified"):
        (out_dir / sub).mkdir(exist_ok=True)

    resolve_job_ids(out_dir, jobs)
    manifest = BatchManifest(
        out_dir=str(out_dir),
        jobs=[j.job_id for j in jobs],
        started_at=time.time(),
    )

    pending: list[BatchJob] = []
    stale: dict[str, str] = {}
    for j in jobs:
        apply_teacher_iso_policy(j.config)
        if resume and _marker_path(out_dir, j.job_id).exists():
            why = stale_resume_reason(out_dir, j)
            if why is None:
                manifest.skipped.append(j.job_id)
                continue
            stale[j.job_id] = why
            print(f"[cfr_batch] re-solve {j.job_id}: existing solve {why}", flush=True)
        pending.append(j)

    _atomic_write_text(
        out_dir / "plan.json",
        json.dumps(
            {
                "n_jobs": len(jobs),
                "pending": [j.job_id for j in pending],
                "skipped": manifest.skipped,
                "stale_resolve": stale,
                "dry_run": dry_run,
                "use_isomorphism": TEACHER_USE_ISOMORPHISM,
                "max_expl_bb": max_expl_bb,
            },
            indent=2,
        )
        + "\n",
    )

    if dry_run:
        manifest.finished_at = time.time()
        manifest.write(out_dir / "manifest.json")
        return manifest

    payloads = [job_payload(j, out_dir, max_expl_bb) for j in pending]

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
