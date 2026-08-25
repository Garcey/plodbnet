"""Background solve job manager for the CFR desktop app.

Uses kill-safe ``stop_file`` + optional ``time_budget_secs`` so the UI can
stop a long solve without killing the process hard. One active job at a time
(simple desktop UX); history of completed jobs kept in memory + optional disk.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from plo5bp.gto.cfr_api import (
    SIZE_PRESETS,
    RootSpec,
    SolveConfig,
    SolveReport,
    rust_cfr_available,
    solve,
    solve_kuhn,
)


STREET_NAMES = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River"}


@dataclass
class JobState:
    job_id: str
    status: str  # queued | running | paused | done | error | stopped | cancelled
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    root: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)
    out_path: str | None = None
    stop_file: str | None = None
    pause_file: str | None = None
    progress_file: str | None = None
    progress_message: str = ""
    # Live counters mirrored from progress_file (updated by poll helpers).
    iterations_run: int = 0
    exploitability_bb: float | None = None
    num_infosets: int = 0
    unlimited: bool = False

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Don't embed full strategy in list endpoints — strip if huge
        return d

    def as_dict_light(self) -> dict[str, Any]:
        d = self.as_dict()
        if d.get("report") and isinstance(d["report"], dict):
            rep = dict(d["report"])
            strat = rep.get("strategy") or {}
            n = len(strat.get("infosets") or []) if isinstance(strat, dict) else 0
            # Prefer live counter when report still partial
            if n == 0 and d.get("num_infosets"):
                n = int(d["num_infosets"])
            rep["strategy"] = {
                "root_id": strat.get("root_id") if isinstance(strat, dict) else None,
                "num_infosets": n,
                "infosets_omitted": True,
            }
            if d.get("iterations_run") is not None and rep.get("iterations_run") is None:
                rep["iterations_run"] = d["iterations_run"]
            if d.get("exploitability_bb") is not None and rep.get("exploitability_bb") is None:
                rep["exploitability_bb"] = d["exploitability_bb"]
            d["report"] = rep
        return d


class SolveSession:
    """Thread-safe single-active-job solve session."""

    def __init__(
        self,
        *,
        work_dir: Path | str | None = None,
        solve_fn: Callable[[RootSpec, SolveConfig], SolveReport] | None = None,
    ) -> None:
        repo = Path(__file__).resolve().parents[3]
        self.work_dir = Path(work_dir) if work_dir else repo / "data" / "cfr" / "app_jobs"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._solve_fn = solve_fn or solve
        self._lock = threading.RLock()
        self._jobs: dict[str, JobState] = {}
        self._active_id: str | None = None
        self._thread: threading.Thread | None = None
        self._history_limit = 40

    # --- queries -----------------------------------------------------------

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [j.as_dict_light() for j in jobs]

    def get_job(self, job_id: str, *, full: bool = True) -> dict[str, Any] | None:
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return None
            return j.as_dict() if full else j.as_dict_light()

    def active_job(self) -> dict[str, Any] | None:
        with self._lock:
            if self._active_id and self._active_id in self._jobs:
                return self._jobs[self._active_id].as_dict_light()
            return None

    def rust_available(self) -> bool:
        return rust_cfr_available()

    # --- control -----------------------------------------------------------

    def start(
        self,
        root: RootSpec | dict[str, Any],
        config: SolveConfig | dict[str, Any] | None = None,
        *,
        save: bool = True,
        label: str = "",
    ) -> JobState:
        root_spec = root if isinstance(root, RootSpec) else _root_from_dict(root)
        root_spec.validate()
        cfg = config if isinstance(config, SolveConfig) else _config_from_dict(config or {})
        cfg.validate()

        job_id = uuid.uuid4().hex[:12]
        stop_path = self.work_dir / f"{job_id}.stop"
        pause_path = self.work_dir / f"{job_id}.pause"
        progress_path = self.work_dir / f"{job_id}.progress.json"
        for p in (stop_path, pause_path, progress_path):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        # Wire kill-safe stop / pause / live progress into solver config
        cfg.stop_file = str(stop_path)
        cfg.pause_file = str(pause_path)
        cfg.progress_file = str(progress_path)
        if cfg.poll_every < 1:
            cfg.poll_every = 50  # snappier live UI default
        # Cap poll for huge dumps; still responsive for unlimited mode.
        if cfg.poll_every > 500:
            cfg.poll_every = 200

        out_path = None
        if save:
            out_path = str(self.work_dir / f"{job_id}.json")

        unlimited = int(cfg.max_iterations) == 0
        job = JobState(
            job_id=job_id,
            status="queued",
            created_at=time.time(),
            root=root_spec.as_dict(),
            config=cfg.as_dict(),
            out_path=out_path,
            stop_file=str(stop_path),
            pause_file=str(pause_path),
            progress_file=str(progress_path),
            progress_message="queued",
            notes=[label] if label else [],
            unlimited=unlimited,
        )

        with self._lock:
            if self._active_id is not None:
                active = self._jobs.get(self._active_id)
                if active and active.status in ("queued", "running", "paused"):
                    raise RuntimeError(
                        f"job {self._active_id} still {active.status}; stop it first"
                    )
            self._jobs[job_id] = job
            self._active_id = job_id
            self._prune_history()

        t = threading.Thread(
            target=self._run_job,
            args=(job_id, root_spec, cfg),
            name=f"cfr-solve-{job_id}",
            daemon=True,
        )
        with self._lock:
            self._thread = t
        t.start()
        return job

    def stop(self, job_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            jid = job_id or self._active_id
            if not jid or jid not in self._jobs:
                raise KeyError("no active job")
            job = self._jobs[jid]
            if job.status not in ("queued", "running", "paused"):
                return job.as_dict_light()
            # Clear pause so the solver can observe stop_file while spinning.
            if job.pause_file:
                try:
                    Path(job.pause_file).unlink(missing_ok=True)
                except OSError:
                    pass
            if job.stop_file:
                Path(job.stop_file).write_text("stop\n", encoding="utf-8")
            job.progress_message = "stop requested"
            job.notes.append("stop_requested")
            return job.as_dict_light()

    def pause(self, job_id: str | None = None) -> dict[str, Any]:
        """Pause a running solve (solver spin-waits; strategy state kept)."""
        with self._lock:
            jid = job_id or self._active_id
            if not jid or jid not in self._jobs:
                raise KeyError("no active job")
            job = self._jobs[jid]
            if job.status not in ("running", "queued"):
                return job.as_dict_light()
            if job.pause_file:
                Path(job.pause_file).write_text("pause\n", encoding="utf-8")
            job.status = "paused"
            job.progress_message = "paused"
            job.notes.append("pause_requested")
            return job.as_dict_light()

    def resume(self, job_id: str | None = None) -> dict[str, Any]:
        """Resume a paused solve."""
        with self._lock:
            jid = job_id or self._active_id
            if not jid or jid not in self._jobs:
                raise KeyError("no active job")
            job = self._jobs[jid]
            if job.pause_file:
                try:
                    Path(job.pause_file).unlink(missing_ok=True)
                except OSError:
                    pass
            if job.status == "paused":
                job.status = "running"
                job.progress_message = "resumed"
                job.notes.append("resume_requested")
            return job.as_dict_light()

    def refresh_progress(self, job_id: str | None = None) -> dict[str, Any] | None:
        """Read progress_file into job counters (and partial report if present)."""
        with self._lock:
            jid = job_id or self._active_id
            if not jid or jid not in self._jobs:
                return None
            job = self._jobs[jid]
            pf = job.progress_file
            if not pf:
                return job.as_dict_light()
            path = Path(pf)
            if not path.is_file():
                return job.as_dict_light()
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return job.as_dict_light()
            job.iterations_run = int(data.get("iterations_run") or job.iterations_run or 0)
            if data.get("exploitability_bb") is not None:
                try:
                    job.exploitability_bb = float(data["exploitability_bb"])
                except (TypeError, ValueError):
                    pass
            job.num_infosets = int(data.get("num_infosets") or job.num_infosets or 0)
            # If strategy snapshot present, stash as partial report for live view.
            if isinstance(data.get("strategy"), dict) and data["strategy"].get("infosets"):
                partial = {
                    "status": "running",
                    "root": job.root,
                    "config": job.config,
                    "strategy": data["strategy"],
                    "iterations_run": job.iterations_run,
                    "exploitability_bb": job.exploitability_bb,
                    "notes": list(job.notes)
                    + [f"live_snapshot_iter={job.iterations_run}"],
                }
                # Only overwrite if still running/paused — don't clobber final.
                if job.status in ("queued", "running", "paused"):
                    job.report = partial
            stop_pending = "stop_requested" in job.notes or (
                job.stop_file and Path(job.stop_file).exists()
            )
            if stop_pending and job.status in ("running", "paused", "queued"):
                # Don't flip paused→running while stop is in flight
                job.progress_message = (
                    f"stopping · {job.iterations_run} iters · {job.num_infosets} infosets"
                )
            elif job.status == "running" and job.pause_file and Path(job.pause_file).exists():
                job.status = "paused"
                job.progress_message = (
                    f"{job.status} · {job.iterations_run} iters · {job.num_infosets} infosets"
                )
            elif (
                job.status == "paused"
                and job.pause_file
                and not Path(job.pause_file).exists()
                and not stop_pending
            ):
                # resume happened externally
                job.status = "running"
                job.progress_message = (
                    f"{job.status} · {job.iterations_run} iters · {job.num_infosets} infosets"
                )
            else:
                job.progress_message = (
                    f"{job.status} · {job.iterations_run} iters · {job.num_infosets} infosets"
                )
            return job.as_dict_light()

    def start_kuhn(self, iterations: int = 5000) -> JobState:
        """Quick correctness gate — runs in background like a normal job."""
        job_id = uuid.uuid4().hex[:12]
        job = JobState(
            job_id=job_id,
            status="queued",
            created_at=time.time(),
            root={"game": "kuhn"},
            config={"max_iterations": int(iterations)},
            progress_message="queued kuhn",
            notes=["kuhn"],
        )
        with self._lock:
            if self._active_id is not None:
                active = self._jobs.get(self._active_id)
                if active and active.status in ("queued", "running", "paused"):
                    raise RuntimeError(
                        f"job {self._active_id} still {active.status}; stop it first"
                    )
            self._jobs[job_id] = job
            self._active_id = job_id

        def _run() -> None:
            with self._lock:
                job.status = "running"
                job.started_at = time.time()
                job.progress_message = "solving kuhn"
            try:
                rep = solve_kuhn(iterations=int(iterations))
                with self._lock:
                    job.status = "done"
                    job.finished_at = time.time()
                    job.report = dict(rep) if not isinstance(rep, dict) else rep
                    job.progress_message = "done"
            except Exception as e:
                with self._lock:
                    job.status = "error"
                    job.finished_at = time.time()
                    job.error = str(e)
                    job.progress_message = f"error: {e}"

        threading.Thread(target=_run, name=f"cfr-kuhn-{job_id}", daemon=True).start()
        return job

    def load_report_file(self, path: Path | str) -> dict[str, Any]:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        job_id = f"load_{uuid.uuid4().hex[:8]}"
        job = JobState(
            job_id=job_id,
            status="done",
            created_at=time.time(),
            started_at=time.time(),
            finished_at=time.time(),
            root=data.get("root") or {},
            config=data.get("config") or {},
            report=data,
            out_path=str(p),
            progress_message="loaded from disk",
            notes=[f"loaded:{p.name}"],
        )
        with self._lock:
            self._jobs[job_id] = job
            self._prune_history()
        return job.as_dict()

    # --- internals ---------------------------------------------------------

    def _run_job(self, job_id: str, root: RootSpec, cfg: SolveConfig) -> None:
        with self._lock:
            job = self._jobs[job_id]
            # Don't clobber a pause that landed between start() and thread run.
            if job.status != "paused":
                job.status = "running"
            job.started_at = time.time()
            lim = "∞" if cfg.max_iterations == 0 else str(cfg.max_iterations)
            job.progress_message = (
                f"solving {STREET_NAMES.get(root.street, root.street)} "
                f"iters≤{lim} threads={cfg.thread_num}"
            )
        try:
            report = self._solve_fn(root, cfg)
            rep_d = report.as_dict() if isinstance(report, SolveReport) else dict(report)
            status = str(rep_d.get("status") or "ok")
            # Detect stop
            stopped = False
            notes = [str(n) for n in (rep_d.get("notes") or [])]
            if any("stop_file" in n or "early_stop=stop" in n for n in notes):
                stopped = True
            if job.stop_file and Path(job.stop_file).exists():
                if "stop_requested" in job.notes:
                    stopped = True
            with self._lock:
                job.report = rep_d
                job.finished_at = time.time()
                job.iterations_run = int(rep_d.get("iterations_run") or 0)
                job.exploitability_bb = rep_d.get("exploitability_bb")
                strat = rep_d.get("strategy") or {}
                if isinstance(strat, dict):
                    job.num_infosets = len(strat.get("infosets") or [])
                if status == "not_implemented":
                    job.status = "error"
                    job.error = "Rust CFR not available — maturin develop --release"
                elif stopped:
                    job.status = "stopped"
                else:
                    job.status = "done" if status == "ok" else status
                job.progress_message = job.status
                job.notes.extend(notes)
            if job.out_path and status != "not_implemented":
                Path(job.out_path).parent.mkdir(parents=True, exist_ok=True)
                Path(job.out_path).write_text(
                    json.dumps(rep_d, indent=2) + "\n", encoding="utf-8"
                )
        except Exception as e:
            with self._lock:
                job.status = "error"
                job.finished_at = time.time()
                job.error = f"{type(e).__name__}: {e}"
                job.progress_message = f"error: {e}"
        finally:
            for attr in ("stop_file", "pause_file"):
                p = getattr(job, attr, None)
                if p:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except OSError:
                        pass
            with self._lock:
                if self._active_id == job_id:
                    # keep active_id pointing at last job for UI convenience
                    pass

    def _prune_history(self) -> None:
        if len(self._jobs) <= self._history_limit:
            return
        # drop oldest finished jobs
        finished = [
            j
            for j in self._jobs.values()
            if j.status not in ("queued", "running", "paused")
        ]
        finished.sort(key=lambda j: j.created_at)
        while len(self._jobs) > self._history_limit and finished:
            old = finished.pop(0)
            if old.job_id != self._active_id:
                self._jobs.pop(old.job_id, None)


def _root_from_dict(d: dict[str, Any]) -> RootSpec:
    sizes = d.get("raise_sizes_pm")
    if sizes is None and d.get("size_preset"):
        preset = str(d["size_preset"])
        sizes = list(SIZE_PRESETS.get(preset, SIZE_PRESETS["standard"]))
    return RootSpec(
        street=int(d.get("street", 3)),
        pot_bb=float(d.get("pot_bb", 10.0)),
        effective_stack_bb=float(d.get("effective_stack_bb", d.get("stack_bb", 50.0))),
        board=[int(c) for c in (d.get("board") or [])],
        num_seats=int(d.get("num_seats", 2)),
        bb_chips=int(d.get("bb_chips", 10_000)),
        sb_chips=int(d.get("sb_chips", 5_000)),
        ante_chips=int(d.get("ante_chips", 5_000)),
        raise_sizes_pm=list(sizes) if sizes is not None else list(SIZE_PRESETS["standard"]),
        allin_atom=bool(d.get("allin_atom", True)),
        range_ip=str(d.get("range_ip") or ""),
        range_oop=str(d.get("range_oop") or ""),
        stacks_bb=[float(x) for x in (d.get("stacks_bb") or [])],
        root_id=str(d.get("root_id") or ""),
    )


def _config_from_dict(d: dict[str, Any]) -> SolveConfig:
    # max_iterations: 0 or missing with unlimited flag → unlimited
    raw_iters = d.get("max_iterations", d.get("iters", 200))
    if d.get("unlimited"):
        raw_iters = 0
    return SolveConfig(
        max_iterations=int(raw_iters if raw_iters is not None else 200),
        target_exploitability_bb=float(d.get("target_exploitability_bb", d.get("target_expl", 0.5))),
        thread_num=int(d.get("thread_num", d.get("threads", 1))),
        seed=int(d.get("seed", 0)),
        use_isomorphism=bool(d.get("use_isomorphism", True)),
        algorithm=str(d.get("algorithm", "dcfr")),
        card_abstraction=str(d.get("card_abstraction", "none")),
        time_budget_secs=float(d.get("time_budget_secs", 0.0)),
        stop_file=str(d.get("stop_file") or ""),
        poll_every=int(d.get("poll_every", 50)),
        pause_file=str(d.get("pause_file") or ""),
        progress_file=str(d.get("progress_file") or ""),
    )


def root_presets() -> list[dict[str, Any]]:
    """Handy root templates for the tree-builder UI."""
    return [
        {
            "id": "preflop_hu_100",
            "label": "HU Preflop 100bb",
            "street": 0,
            "pot_bb": 2.5,
            "effective_stack_bb": 100.0,
            "board": [],
            "num_seats": 2,
            "algorithm": "mccfr_es",
            "size_preset": "standard",
            "raise_sizes_pm": list(SIZE_PRESETS["standard"]),
        },
        {
            "id": "preflop_hu_20",
            "label": "HU Preflop 20bb",
            "street": 0,
            "pot_bb": 2.5,
            "effective_stack_bb": 20.0,
            "board": [],
            "num_seats": 2,
            "algorithm": "mccfr_es",
            "size_preset": "coarse",
            "raise_sizes_pm": list(SIZE_PRESETS["coarse"]),
        },
        {
            "id": "river_hu_standard",
            "label": "HU River pot=10bb stack=50bb",
            "street": 3,
            "pot_bb": 10.0,
            "effective_stack_bb": 50.0,
            "board": [12, 28, 38, 41, 45],  # sample board
            "num_seats": 2,
            "algorithm": "dcfr",
            "size_preset": "standard",
            "raise_sizes_pm": list(SIZE_PRESETS["standard"]),
        },
        {
            "id": "river_hu_micro",
            "label": "HU River micro sizes",
            "street": 3,
            "pot_bb": 10.0,
            "effective_stack_bb": 50.0,
            "board": [48, 44, 40, 36, 32],
            "num_seats": 2,
            "algorithm": "dcfr",
            "size_preset": "micro",
            "raise_sizes_pm": list(SIZE_PRESETS["micro"]),
        },
        {
            "id": "pushfold_4h_10bb",
            "label": "4-handed Push/Fold 10bb (no ante)",
            "street": 0,
            "pot_bb": 1.5,
            "effective_stack_bb": 10.0,
            "board": [],
            "num_seats": 4,
            "algorithm": "mccfr_es",
            "size_preset": "micro",
            "raise_sizes_pm": [],
            "allin_atom": True,
            "ante_chips": 0,
            "stacks_bb": [10.0, 10.0, 10.0, 10.0],
        },
        {
            "id": "flop_hu_ochs",
            "label": "HU Flop OCHS@200 pot=6bb stack=40bb",
            "street": 1,
            "pot_bb": 6.0,
            "effective_stack_bb": 40.0,
            "board": [12, 28, 38],
            "num_seats": 2,
            "algorithm": "dcfr",
            "card_abstraction": "ochs",
            "size_preset": "coarse",
            "raise_sizes_pm": list(SIZE_PRESETS["coarse"]),
        },
    ]
