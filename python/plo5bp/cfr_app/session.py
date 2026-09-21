"""Background solve job manager for the CFR desktop app.

Uses kill-safe ``stop_file`` + optional ``time_budget_secs`` so the UI can
stop a long solve without killing the process hard. One active job at a time
(simple desktop UX); history of completed jobs kept in memory + optional disk.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from plo5bp.cfr_app.paths import jobs_dir
from plo5bp.cfr_app.ranges import apply_ranges, attach_range_text
from plo5bp.cfr_app.solve_worker import sanitize_json, write_json_atomic
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

# (review 2026-09-20 E10) How long a child solver may ignore Stop / overrun its
# time budget before the supervisor kills it. Normal iterations finish in
# milliseconds-to-seconds, so a graceful stop (strategy exported) nearly always
# wins the race; the kill is for a single iteration that runs for minutes.
STOP_KILL_GRACE_SECS = 10.0
BUDGET_KILL_GRACE_SECS = 15.0

ACTIVE_STATES = ("queued", "running", "paused")


def _has_infosets(rep: Any) -> bool:
    strat = rep.get("strategy") if isinstance(rep, dict) else None
    return isinstance(strat, dict) and bool(strat.get("infosets"))


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
    # (review 2026-09-20 E10) when Stop was pressed — lets the supervisor kill a
    # child solver that never reaches an iteration boundary.
    stop_requested_at: float | None = None

    def _base_dict(self) -> dict[str, Any]:
        # (review 2026-09-20 E4) Built field by field. The old `asdict(self)`
        # deep-copied the whole report — 100k+ infoset dicts — on EVERY
        # /api/jobs and /progress poll (1.3 s per call at 150k infosets, under
        # the session lock, against an 800 ms timer).
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "root": dict(self.root),
            "config": dict(self.config),
            "error": self.error,
            "notes": list(self.notes),
            "out_path": self.out_path,
            "stop_file": self.stop_file,
            "pause_file": self.pause_file,
            "progress_file": self.progress_file,
            "progress_message": self.progress_message,
            "iterations_run": self.iterations_run,
            "exploitability_bb": self.exploitability_bb,
            "num_infosets": self.num_infosets,
            "unlimited": self.unlimited,
            "stop_requested_at": self.stop_requested_at,
        }

    def as_dict(self) -> dict[str, Any]:
        """Full job dict. ``report`` is shared, not copied — treat as read-only."""
        d = self._base_dict()
        d["report"] = self.report
        return d

    def as_dict_light(self) -> dict[str, Any]:
        """Job dict for list/poll endpoints: report reduced to its scalars."""
        d = self._base_dict()
        d["report"] = light_report(self.report, self)
        return d


def light_report(rep: dict[str, Any] | None, job: "JobState | None" = None) -> dict[str, Any] | None:
    """Report minus the infosets — O(top-level keys), never touches the strategy rows."""
    if not isinstance(rep, dict):
        return None
    out = {k: v for k, v in rep.items() if k != "strategy"}
    strat = rep.get("strategy")
    strat = strat if isinstance(strat, dict) else {}
    n = strat.get("num_infosets")
    if n is None:
        n = len(strat.get("infosets") or [])
    if not n and job is not None and job.num_infosets:
        n = int(job.num_infosets)  # live counter when the report is still partial
    out["strategy"] = {
        "root_id": strat.get("root_id"),
        "num_infosets": int(n or 0),
        "infosets_omitted": True,
    }
    if job is not None:
        if out.get("iterations_run") is None and job.iterations_run is not None:
            out["iterations_run"] = job.iterations_run
        if out.get("exploitability_bb") is None and job.exploitability_bb is not None:
            out["exploitability_bb"] = job.exploitability_bb
    return out


# Counters sit at the HEAD of the progress JSON, before "strategy" (see
# SolveConfig::write_progress in rust_engine/src/cfr/types.rs).
_PROGRESS_HEAD_BYTES = 4096
_RE_HEAD_INT = {
    k: re.compile(rf'"{k}"\s*:\s*(\d+)') for k in ("iterations_run", "num_infosets")
}
_RE_HEAD_EXPL = re.compile(r'"exploitability_bb"\s*:\s*(null|-?[0-9.eE+\-]+)')


def read_progress_counters(path: Path | str) -> dict[str, Any] | None:
    """Live counters from a progress file WITHOUT parsing it.

    (review 2026-09-20 E4) The file carries the full strategy (157 MB on the
    default river preset); json.loads of it took seconds per poll. The counters
    are in the first few hundred bytes, so read a 4 KB head and pick them out.
    Returns None if the file is missing/unreadable or the head has no counters.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(_PROGRESS_HEAD_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return None
    # Never read counters out of the strategy body, should the layout change.
    cut = head.find('"strategy"')
    if cut >= 0:
        head = head[:cut]
    out: dict[str, Any] = {}
    for key, rx in _RE_HEAD_INT.items():
        m = rx.search(head)
        if m:
            out[key] = int(m.group(1))
    m = _RE_HEAD_EXPL.search(head)
    if m and m.group(1) != "null":
        try:
            v = float(m.group(1))
            if math.isfinite(v):
                out["exploitability_bb"] = v
        except ValueError:
            pass
    return out if "iterations_run" in out else None


class SolveSession:
    """Thread-safe single-active-job solve session."""

    def __init__(
        self,
        *,
        work_dir: Path | str | None = None,
        solve_fn: Callable[[RootSpec, SolveConfig], SolveReport] | None = None,
        use_subprocess: bool | None = None,
    ) -> None:
        # (review 2026-09-20 J4) default comes from paths.jobs_dir() (env-overridable)
        # and is created lazily in start(), so constructing a session — which
        # server.py does at import — never touches the real data dir.
        self.work_dir = Path(work_dir) if work_dir else jobs_dir()
        self._solve_fn = solve_fn or solve
        # (review 2026-09-20 E1/E2/E10) The real solver runs in a spawn child so a
        # native crash can't kill the UI and Stop can kill a stuck solve. An
        # injected solve_fn (tests pass closures, which don't pickle) runs
        # in-thread. CFR_APP_INPROCESS=1 forces the old in-thread behaviour.
        if use_subprocess is None:
            use_subprocess = solve_fn is None and os.environ.get(
                "CFR_APP_INPROCESS", ""
            ).strip() not in ("1", "true", "yes")
        self._use_subprocess = bool(use_subprocess)
        self._proc: Any = None  # live multiprocessing.Process, if any
        self._report_cache: dict[str, Any] = {}  # single entry, keyed by file signature
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
            if not full:
                return j.as_dict_light()
            d = j.as_dict()
        # Full report is materialized lazily, outside the lock (may read a big file).
        if not _has_infosets(d.get("report")):
            rep, _sig = self.full_report(job_id)
            if rep is not None:
                d["report"] = rep
        return d

    def full_report(self, job_id: str) -> tuple[dict[str, Any] | None, Any]:
        """Report WITH infosets for the viewer, plus a change signature.

        (review 2026-09-20 E4) Sources, in order: the in-memory report (save=False
        jobs), the saved ``out_path`` (finished / loaded jobs), or the live
        progress snapshot. File parses happen OUTSIDE the session lock and are
        cached by (path, mtime, size): the old code re-parsed the whole snapshot
        under the lock on every poll and stashed it on the job. The signature
        lets callers cache whatever they derive from the report.
        """
        with self._lock:
            j = self._jobs.get(job_id)
            if j is None:
                return None, None
            if _has_infosets(j.report):
                return j.report, ("memory", job_id, j.status, j.iterations_run)
            active = j.status in ACTIVE_STATES
            out_path, progress_file = j.out_path, j.progress_file
            overlay = {
                "status": "running" if active else j.status,
                "root": dict(j.root),
                "config": dict(j.config),
                "notes": list(j.notes),
            }
        # Finished/loaded jobs read their saved report; a live job (or one that
        # was killed before exporting) falls back to the last progress snapshot.
        for kind, src in (("final", out_path), ("live", progress_file)):
            if not src or (kind == "final" and active):
                continue
            try:
                st = os.stat(src)
            except OSError:
                continue
            sig = (kind, str(src), st.st_mtime_ns, st.st_size)
            with self._lock:
                cached = self._report_cache if self._report_cache.get("sig") == sig else None
            if cached is not None:
                return cached["report"], sig
            try:
                data = json.loads(Path(src).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue  # mid-rewrite or truncated — caller retries next tick
            if not isinstance(data, dict):
                continue
            if kind == "live":
                if not _has_infosets(data):
                    # Counters-only dump (or strategy not written yet): keep
                    # serving the last snapshot we parsed for this file, if any.
                    with self._lock:
                        prev = self._report_cache
                    if prev.get("sig") and prev["sig"][:2] == sig[:2]:
                        return prev["report"], prev["sig"]
                    continue
                # Progress files carry counters + strategy only; dress as a report.
                data = {
                    **overlay,
                    "strategy": data.get("strategy") or {},
                    "iterations_run": data.get("iterations_run"),
                    "exploitability_bb": data.get("exploitability_bb"),
                    "notes": overlay["notes"] + [f"live_snapshot_iter={data.get('iterations_run')}"],
                }
            data = sanitize_json(data)
            with self._lock:
                self._report_cache = {"sig": sig, "report": data}  # one entry: bounded memory
            return data, sig
        return None, None

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
        # (review 2026-09-20 E11) Parse the range text strictly (ValueError on a bad
        # token) and hand the solver a canonical string the native parser reads
        # correctly; the user's wording rides along as range_*_text for display.
        root_in = root.as_dict() if isinstance(root, RootSpec) else dict(root)
        root_d, _range_info = apply_ranges(root_in)
        root_spec = _root_from_dict(root_d)  # lossless: covers every RootSpec field
        validate_root_for_app(root_spec)  # (review 2026-09-20 E1) superset of .validate()
        range_text = {k: root_d.get(k, "") for k in ("range_oop_text", "range_ip_text")}
        cfg = config if isinstance(config, SolveConfig) else _config_from_dict(config or {})
        cfg.validate()

        job_id = uuid.uuid4().hex[:12]
        self.work_dir.mkdir(parents=True, exist_ok=True)
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
        # (review 2026-09-20 E4) The solver dumps the FULL strategy every
        # poll_every iterations (iteration-based, inside Rust — not controllable
        # from here), so this used to force a user's large value DOWN to 200,
        # i.e. more 100+ MB dumps. Respect what the user asked for.

        out_path = None
        if save:
            out_path = str(self.work_dir / f"{job_id}.json")

        unlimited = int(cfg.max_iterations) == 0
        job = JobState(
            job_id=job_id,
            status="queued",
            created_at=time.time(),
            root={**root_spec.as_dict(), **range_text},
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
            if job.stop_requested_at is None:
                job.stop_requested_at = time.time()
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
        """Mirror the live counters from ``progress_file`` onto the job (cheap).

        (review 2026-09-20 E4) Reads a 4 KB head, outside the lock — never the
        strategy. The viewer gets the snapshot through :meth:`full_report`.
        """
        with self._lock:
            jid = job_id or self._active_id
            if not jid or jid not in self._jobs:
                return None
            job = self._jobs[jid]
            # (review 2026-09-20 E12) A finished job's counters come from its
            # final report. The UI polls /progress once more after "done", and
            # the stale snapshot (last poll tick, e.g. iter 100 of 120, with the
            # noisier poll-time exploitability) used to overwrite them.
            if job.status not in ACTIVE_STATES or not job.progress_file:
                return job.as_dict_light()
            pf = job.progress_file
        counters = read_progress_counters(pf)  # file IO outside the lock
        with self._lock:
            job = self._jobs.get(jid)
            if job is None:
                return None
            if job.status not in ACTIVE_STATES:
                return job.as_dict_light()  # finished while we were reading
            if counters:
                job.iterations_run = max(job.iterations_run, int(counters["iterations_run"]))
                job.num_infosets = int(counters.get("num_infosets") or job.num_infosets or 0)
                if counters.get("exploitability_bb") is not None:
                    job.exploitability_bb = float(counters["exploitability_bb"])
            stop_pending = job.stop_requested_at is not None or (
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
                    job.report = sanitize_json(dict(rep) if not isinstance(rep, dict) else rep)
                    job.progress_message = "done"
            except BaseException as e:  # noqa: BLE001 — (review 2026-09-20 E2) PanicException
                with self._lock:
                    job.status = "error"
                    job.finished_at = time.time()
                    job.error = f"{type(e).__name__}: {e}"
                    job.progress_message = f"error: {e}"

        threading.Thread(target=_run, name=f"cfr-kuhn-{job_id}", daemon=True).start()
        return job

    def load_report_file(self, path: Path | str) -> dict[str, Any]:
        p = Path(path)
        data = sanitize_json(json.loads(p.read_text(encoding="utf-8")))
        if not isinstance(data, dict):
            raise ValueError("strategy file must be a JSON object")
        job_id = f"load_{uuid.uuid4().hex[:8]}"
        strat = data.get("strategy") if isinstance(data.get("strategy"), dict) else {}
        n_infosets = len(strat.get("infosets") or data.get("infosets") or data.get("hands") or [])
        expl = data.get("exploitability_bb")
        job = JobState(
            job_id=job_id,
            status="done",
            created_at=time.time(),
            started_at=time.time(),
            finished_at=time.time(),
            root=data.get("root") if isinstance(data.get("root"), dict) else {},
            config=data.get("config") if isinstance(data.get("config"), dict) else {},
            # (review 2026-09-20 E4) Keep the scalars only. Every Library "Open"
            # used to pin a full parsed report in memory (×40 history slots) that
            # each /api/jobs poll then deep-copied; the file is the source of
            # truth and full_report() re-reads it (mtime-cached) on demand.
            report=light_report(data),
            out_path=str(p),
            progress_message="loaded from disk",
            notes=[f"loaded:{p.name}"],
            iterations_run=int(data.get("iterations_run") or 0),
            exploitability_bb=float(expl) if isinstance(expl, (int, float)) else None,
            num_infosets=n_infosets,
        )
        if job.report is not None:
            job.report["strategy"]["num_infosets"] = n_infosets
        with self._lock:
            self._jobs[job_id] = job
            self._prune_history()
        d = job.as_dict()
        d["report"] = data  # this call's return value stays the full report
        return d

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
        keep_progress = False
        try:
            if self._use_subprocess:
                rep_d = self._solve_in_child(job, root, cfg)  # already strict JSON
            else:
                report = self._solve_fn(root, cfg)
                rep_d = report.as_dict() if isinstance(report, SolveReport) else dict(report)
                rep_d = sanitize_json(rep_d)  # NaN/inf → None (see solve_worker)
                attach_range_text(rep_d, job.root)
            if rep_d is None:
                # Child was killed (Stop / time budget) before it could export a
                # strategy; _solve_in_child already set the terminal state. The
                # last progress snapshot is the only strategy left — keep it.
                keep_progress = True
            else:
                self._finalize_report(job, rep_d)
        except BaseException as e:  # noqa: BLE001
            # (review 2026-09-20 E2) PyO3's PanicException derives from
            # BaseException, so `except Exception` let a Rust panic kill this
            # thread with the job still "running" — every later solve then got
            # 409 forever and Stop did nothing. Nothing above us in a worker
            # thread can use a KeyboardInterrupt/SystemExit either, so record it.
            with self._lock:
                job.status = "error"
                job.finished_at = time.time()
                job.error = f"{type(e).__name__}: {e}"
                job.progress_message = f"error: {e}"
        finally:
            with self._lock:
                # Belt and braces: whatever happened, never leave the job active.
                if job.status in ("queued", "running", "paused"):
                    job.status = "error"
                    job.error = job.error or "solver exited without a result"
                    job.progress_message = f"error: {job.error}"
                if job.finished_at is None:
                    job.finished_at = time.time()
                self._proc = None
            # (review 2026-09-20 E12) the final report supersedes the live
            # snapshot; a leftover progress file also let refresh_progress()
            # clobber the final counters and showed up in the Library.
            doomed = ["stop_file", "pause_file"]
            if not keep_progress:
                doomed.append("progress_file")
            for attr in doomed:
                p = getattr(job, attr, None)
                if p:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except OSError:
                        pass

    def _finalize_report(self, job: JobState, rep_d: dict[str, Any]) -> None:
        """Move ``job`` to its terminal state from a finished solve report."""
        status = str(rep_d.get("status") or "ok")
        notes = [str(n) for n in (rep_d.get("notes") or [])]
        stopped = any("stop_file" in n or "early_stop=stop" in n for n in notes)
        if job.stop_requested_at is not None and job.stop_file and Path(job.stop_file).exists():
            stopped = True
        # Persist BEFORE flipping the status: a client that sees "done" may open
        # out_path immediately. The child path has already written it.
        if job.out_path and status != "not_implemented" and not Path(job.out_path).is_file():
            Path(job.out_path).parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(job.out_path, rep_d)
        saved = bool(job.out_path) and Path(job.out_path).is_file()
        with self._lock:
            job.finished_at = time.time()
            job.iterations_run = int(rep_d.get("iterations_run") or 0)
            job.exploitability_bb = rep_d.get("exploitability_bb")
            strat = rep_d.get("strategy") or {}
            if isinstance(strat, dict):
                job.num_infosets = len(strat.get("infosets") or [])
            # (review 2026-09-20 E4) Once the report is on disk keep only its
            # scalars in memory; full_report() reloads out_path for the viewer.
            # save=False jobs have no file, so they keep the full dict.
            job.report = light_report(rep_d, job) if saved else rep_d
            if status == "not_implemented":
                job.status = "error"
                job.error = "Rust CFR not available — maturin develop --release"
            elif stopped:
                job.status = "stopped"
            else:
                job.status = "done" if status == "ok" else status
            job.progress_message = job.status
            job.notes.extend(notes)

    def _solve_in_child(
        self, job: JobState, root: RootSpec, cfg: SolveConfig
    ) -> dict[str, Any] | None:
        """Run the solve in a spawn child; return the report dict.

        Returns ``None`` when the child had to be killed (Stop / time budget
        ignored) — the job's terminal state is set here in that case. Raises on
        a worker error or a native crash; ``_run_job`` turns that into ``error``.
        """
        from plo5bp.cfr_app.solve_worker import run_solve

        self.work_dir.mkdir(parents=True, exist_ok=True)
        # save=True: the child writes straight to out_path (no second copy of a
        # 100+ MB report). save=False: a scratch file we delete after loading.
        scratch = not job.out_path
        result_path = Path(job.out_path or self.work_dir / f"{job.job_id}.result.json")
        error_path = self.work_dir / f"{job.job_id}.error.json"
        for p in (result_path, error_path):
            p.unlink(missing_ok=True)

        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=run_solve,
            # job.root = the RootSpec fields + range_*_text (ignored by the solver,
            # copied onto the saved report so the viewer shows the user's wording).
            args=(dict(job.root), cfg.as_dict(), str(result_path), str(error_path)),
            name=f"cfr-solve-{job.job_id}",
            daemon=True,  # dies with the app window
        )
        started = time.time()
        proc.start()
        with self._lock:
            self._proc = proc

        budget = float(cfg.time_budget_secs or 0.0)
        killed: str | None = None
        while proc.is_alive():
            proc.join(0.2)
            now = time.time()
            # (review 2026-09-20 E10) the solver only looks at the stop file and
            # the time budget BETWEEN iterations; one iteration of a deep tree
            # can run for minutes. Give it a grace period, then kill it.
            if job.stop_requested_at is not None and now - job.stop_requested_at > STOP_KILL_GRACE_SECS:
                killed = "stop"
            elif budget > 0 and now - started > budget + BUDGET_KILL_GRACE_SECS:
                killed = "time_budget"
            if killed:
                proc.terminate()
                proc.join(5.0)
                break

        try:
            if result_path.is_file():
                # A result beats everything, even if we also pulled the trigger.
                return json.loads(result_path.read_text(encoding="utf-8"))
            if killed:
                why = (
                    f"Stop: solver did not reach an iteration boundary within "
                    f"{STOP_KILL_GRACE_SECS:g}s, process killed"
                    if killed == "stop"
                    else f"time budget {budget:g}s exceeded by more than "
                    f"{BUDGET_KILL_GRACE_SECS:g}s mid-iteration, process killed"
                )
                with self._lock:
                    job.status = "stopped"
                    job.finished_at = time.time()
                    job.notes.append(f"killed={killed}")
                    job.notes.append(f"{why} — no final strategy (last live snapshot kept)")
                    job.progress_message = f"stopped · {why}"
                return None
            if error_path.is_file():
                err = json.loads(error_path.read_text(encoding="utf-8"))
                raise RuntimeError(str(err.get("worker_error") or "solver worker failed"))
            code = proc.exitcode
            hint = ""
            if code is not None and (code & 0xFFFFFFFF) == 0xC00000FD:
                hint = " — native stack overflow (tree too deep for this root)"
            raise RuntimeError(
                f"solver process crashed (exit code {code}"
                + (f" / 0x{code & 0xFFFFFFFF:08X}" if code is not None else "")
                + f"){hint}"
            )
        finally:
            error_path.unlink(missing_ok=True)
            if scratch:
                result_path.unlink(missing_ok=True)
            Path(f"{result_path}.tmp").unlink(missing_ok=True)

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


def validate_root_for_app(root: RootSpec) -> None:
    """App-layer root checks that ``RootSpec.validate()`` does not make.

    (review 2026-09-20 E1) ``RootSpec.validate()`` only requires pot/stack > 0,
    but the native solver needs more, and its failure modes are fatal:

    - a pot or stack that rounds to 0 chips panics (``InvalidRoot`` unwrap);
    - a preflop stack that does not cover the big blind + ante recurses until
      the process dies with a native stack overflow (exit 0xC00000FD). That was
      reachable from the Stack spinner (``min=0.5``) and killed the whole app.

    RootSpec lives in gto/cfr_api.py (not this package), so the checks live
    here and run in BOTH /api/validate_root and /api/solve. Raises ValueError.
    """
    root.validate()
    bb = int(root.bb_chips)
    if bb < 1:
        raise ValueError(f"bb_chips must be >= 1, got {root.bb_chips}")
    if int(root.sb_chips) < 0 or int(root.ante_chips) < 0:
        raise ValueError("sb_chips and ante_chips must be >= 0")

    def _chips(x_bb: float) -> int:
        return int(round(float(x_bb) * bb))

    if _chips(root.pot_bb) < 1:
        raise ValueError(
            f"pot_bb {root.pot_bb:g} rounds to 0 chips at bb={bb} — increase the pot"
        )
    stacks = [("effective_stack_bb", float(root.effective_stack_bb))]
    stacks += [(f"stacks_bb[{i}]", float(s)) for i, s in enumerate(root.stacks_bb)]
    for name, s in stacks:
        if _chips(s) < 1:
            raise ValueError(f"{name} {s:g} rounds to 0 chips at bb={bb} — increase the stack")
    if int(root.street) == 0:
        need_bb = (bb + int(root.ante_chips)) / float(bb)
        for name, s in stacks:
            if not s > need_bb:
                raise ValueError(
                    f"{name} {s:g}bb must be greater than the big blind + ante "
                    f"({need_bb:g}bb) for a preflop solve"
                )


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
