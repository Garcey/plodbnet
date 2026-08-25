"""Overnight full-hand native CFR batch (kill-safe, time-budgeted).

Runs preflop blueprints and preflop→induce→postflop pipelines from a grid
JSON. Each solve uses a huge max_iterations + wall time_budget + shared
stop_file so morning kill still yields strategy JSON.

Usage::

  .venv/Scripts/python scripts/cfr_overnight.py \\
      --grid data/cfr/overnight_grid.json

Stop early (exports current average strategy)::

  echo. > data/cfr/overnight/STOP
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plo5bp.gto.cfr_api import (
    SIZE_PRESETS,
    RootSpec,
    SolveConfig,
    SolveReport,
    rust_cfr_available,
    solve,
)
from plo5bp.gto.roots import CLUBGG_NLH_ROOT


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _marker(out_dir: Path, job_id: str) -> Path:
    return out_dir / "markers" / f"{job_id}.done"


def _strategy_path(out_dir: Path, job_id: str) -> Path:
    return out_dir / "strategies" / f"{job_id}.json"


def _progress_path(out_dir: Path, job_id: str) -> Path:
    return out_dir / "strategies" / f"{job_id}.progress.json"


def _global_stop(stop_file: Path) -> bool:
    return stop_file.exists()


def blueprint_root_and_config(
    job: dict[str, Any],
    *,
    stop_file: str,
    progress_file: str = "",
) -> tuple[RootSpec, SolveConfig]:
    """Build the RootSpec + SolveConfig for a preflop_blueprint job.

    Exposed so tests can assert kill-safe wiring (progress_file, poll_every)
    without launching a multi-hour solve.
    """
    n_seats = int(job.get("num_seats", 2))
    stack = float(job["stack_bb"])
    sizes = list(job.get("raise_sizes_pm", list(SIZE_PRESETS["coarse"])))
    ante = int(job.get("ante_chips", CLUBGG_NLH_ROOT.ante))
    bb = int(job.get("bb_chips", CLUBGG_NLH_ROOT.bb))
    sb = int(job.get("sb_chips", CLUBGG_NLH_ROOT.sb))
    job_id = str(job.get("job_id") or "blueprint")

    if n_seats > 2 and not sizes:
        root = RootSpec.preflop_pushfold(
            num_seats=n_seats,
            stack_bb=stack,
            bb_chips=bb,
            sb_chips=sb,
            ante_chips=ante,
        )
        root.root_id = job_id
    else:
        pot_bb = (sb + bb + n_seats * ante) / float(bb)
        root = RootSpec(
            street=0,
            pot_bb=pot_bb,
            effective_stack_bb=stack,
            board=[],
            num_seats=n_seats,
            bb_chips=bb,
            sb_chips=sb,
            ante_chips=ante,
            raise_sizes_pm=sizes,
            allin_atom=bool(job.get("allin_atom", True)),
            stacks_bb=[stack] * n_seats if n_seats > 2 else [],
            root_id=job_id,
        )

    cfg = SolveConfig.teacher(
        max_iterations=int(job.get("max_iterations", 2_000_000_000)),
        target_exploitability_bb=0.0,
        thread_num=1,
        seed=int(job.get("seed", 0)),
        algorithm=str(job.get("algorithm", "mccfr_es")),
        time_budget_secs=float(job.get("time_budget_secs", 3600)),
        stop_file=stop_file,
        # 2000 meant a slow tree could miss the wall budget for hours.
        poll_every=int(job.get("poll_every", 250)),
        progress_file=progress_file,
    )
    return root, cfg


def promote_progress_if_any(
    out_dir: Path,
    job_id: str,
    job: dict[str, Any],
) -> dict[str, Any] | None:
    """If a prior kill left a progress snapshot, promote it to a strategy.

    This is the kill-safe contract: morning STOP / process death still yields
    a usable JSON instead of an empty strategies/ dir.
    """
    prog = _progress_path(out_dir, job_id)
    if not prog.is_file():
        return None
    try:
        data = json.loads(prog.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    strat = data.get("strategy") or {}
    if not strat.get("infosets"):
        return None
    root, _cfg = blueprint_root_and_config(job, stop_file="", progress_file="")
    payload = {
        "status": "ok",
        "root": root.as_dict(),
        "config": data.get("config") or {},
        "strategy": strat,
        "iterations_run": int(data.get("iterations_run") or 0),
        "exploitability_bb": data.get("exploitability_bb"),
        "notes": ["promoted_from_progress", f"job_id={job_id}"],
        "job": job,
        "promoted_from_progress": True,
    }
    path = _strategy_path(out_dir, job_id)
    _atomic_write_json(path, payload)
    _marker(out_dir, job_id).parent.mkdir(parents=True, exist_ok=True)
    _marker(out_dir, job_id).write_text(
        f"promoted_from_progress iters={payload['iterations_run']}\n",
        encoding="utf-8",
    )
    try:
        prog.unlink()
    except OSError:
        pass
    return {
        "job_id": job_id,
        "status": "ok",
        "iterations": payload["iterations_run"],
        "exploitability_bb": payload["exploitability_bb"],
        "promoted_from_progress": True,
        "path": str(path),
    }


def _run_blueprint(job: dict[str, Any], *, stop_file: str, out_dir: Path) -> dict[str, Any]:
    job_id = job["job_id"]
    progress_file = str(_progress_path(out_dir, job_id))
    root, cfg = blueprint_root_and_config(
        job, stop_file=stop_file, progress_file=progress_file
    )
    t0 = time.perf_counter()
    print(
        f"[overnight] blueprint {job_id} budget={cfg.time_budget_secs:.0f}s "
        f"poll={cfg.poll_every} progress={progress_file}",
        flush=True,
    )
    rep = solve(root, cfg)
    elapsed = time.perf_counter() - t0
    if rep.status != "ok":
        raise RuntimeError(
            f"blueprint {job_id} status={rep.status} notes={rep.notes}"
        )
    payload = rep.as_dict()
    payload["job"] = job
    payload["wall_secs"] = elapsed
    path = _strategy_path(out_dir, job_id)
    _atomic_write_json(path, payload)
    _marker(out_dir, job_id).parent.mkdir(parents=True, exist_ok=True)
    _marker(out_dir, job_id).write_text(
        f"ok iters={rep.iterations_run} expl={rep.exploitability_bb}\n",
        encoding="utf-8",
    )
    try:
        Path(progress_file).unlink(missing_ok=True)
    except OSError:
        pass
    print(
        f"[overnight]   done {job_id} iters={rep.iterations_run} "
        f"expl={rep.exploitability_bb} wall={elapsed:.1f}s "
        f"infosets={len(rep.strategy.get('infosets', []))}",
        flush=True,
    )
    return {
        "job_id": job_id,
        "status": rep.status,
        "iterations": rep.iterations_run,
        "exploitability_bb": rep.exploitability_bb,
        "wall_secs": elapsed,
        "path": str(path),
    }


def _run_pipeline_board(
    job: dict[str, Any],
    board_id: str,
    board: list[int],
    *,
    stop_file: str,
    out_dir: Path,
) -> dict[str, Any]:
    from plo5bp import _engine  # type: ignore

    job_id = f"{job['job_id']}_{board_id}"
    if _marker(out_dir, job_id).exists():
        print(f"[overnight] skip {job_id} (marker)", flush=True)
        return {"job_id": job_id, "status": "skipped"}

    stack = float(job["stack_bb"])
    sizes = list(job.get("preflop_raise_sizes_pm", [330, 500, 1000, 1500]))
    t0 = time.perf_counter()
    print(
        f"[overnight] pipeline {job_id} board={board} "
        f"pf_budget={job.get('preflop_time_budget_secs')}s "
        f"post_budget={job.get('postflop_time_budget_secs')}s …",
        flush=True,
    )
    raw = dict(
        _engine.cfr_pipeline(
            stack_bb=stack,
            preflop_iters=int(job.get("preflop_max_iterations", 2_000_000_000)),
            postflop_iters=int(job.get("postflop_max_iterations", 2_000_000_000)),
            postflop_board=[int(c) for c in board],
            pot_bb=float(job.get("pot_bb", 7.0)),
            postflop_stack_bb=float(job.get("postflop_stack_bb", stack * 0.95)),
            oop_action=job.get("oop_action"),
            ip_action=job.get("ip_action"),
            seed=int(job.get("seed", 0)) + sum(board),
            preflop_time_budget_secs=float(job.get("preflop_time_budget_secs", 600)),
            postflop_time_budget_secs=float(job.get("postflop_time_budget_secs", 1200)),
            stop_file=stop_file,
            raise_sizes_pm=[int(x) for x in sizes],
        )
    )
    elapsed = time.perf_counter() - t0
    # Make JSON-serializable
    payload = _jsonable(raw)
    payload["job"] = job
    payload["board_id"] = board_id
    payload["board"] = board
    payload["wall_secs"] = elapsed
    payload["source"] = "rust_cfr_pipeline"
    path = _strategy_path(out_dir, job_id)
    _atomic_write_json(path, payload)
    _marker(out_dir, job_id).parent.mkdir(parents=True, exist_ok=True)
    _marker(out_dir, job_id).write_text(
        f"ok pf_iters={payload.get('preflop_iterations')} "
        f"post_iters={payload.get('postflop_iterations')}\n",
        encoding="utf-8",
    )
    print(
        f"[overnight]   done {job_id} "
        f"pf_iters={payload.get('preflop_iterations')} "
        f"post_iters={payload.get('postflop_iterations')} "
        f"post_expl={payload.get('postflop_exploitability_bb')} "
        f"wall={elapsed:.1f}s",
        flush=True,
    )
    return {
        "job_id": job_id,
        "status": "ok",
        "preflop_iterations": payload.get("preflop_iterations"),
        "postflop_iterations": payload.get("postflop_iterations"),
        "postflop_exploitability_bb": payload.get("postflop_exploitability_bb"),
        "wall_secs": elapsed,
        "path": str(path),
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # PyO3 / numpy leftovers
    try:
        return float(obj)
    except Exception:
        return str(obj)


@dataclass
class OvernightReport:
    results: list[dict[str, Any]]
    stopped_early: bool
    out_dir: str
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": self.results,
            "stopped_early": self.stopped_early,
            "out_dir": self.out_dir,
            "seconds": self.seconds,
            "n_ok": sum(
                1
                for r in self.results
                if r.get("status") in ("ok", "skipped")
            ),
            "n_fail": sum(1 for r in self.results if r.get("status") == "error"),
        }


def run_overnight(
    grid_path: Path | str,
    *,
    resume: bool = True,
    dry_run: bool = False,
) -> OvernightReport:
    grid_path = Path(grid_path)
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    out_dir = Path(grid.get("out_dir", "data/cfr/overnight"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "strategies").mkdir(exist_ok=True)
    (out_dir / "markers").mkdir(exist_ok=True)
    stop_file = Path(grid.get("stop_file", str(out_dir / "STOP")))
    # Ensure parent exists; do not create STOP itself
    stop_file.parent.mkdir(parents=True, exist_ok=True)

    boards_map: dict[str, list[int]] = {
        k: [int(c) for c in v] for k, v in grid.get("boards", {}).items()
    }
    jobs: list[dict[str, Any]] = sorted(
        grid.get("jobs", []), key=lambda j: (int(j.get("priority", 9)), j["job_id"])
    )

    if not rust_cfr_available():
        raise RuntimeError("Rust CFR not available — maturin develop --release")

    if dry_run:
        print(json.dumps({"jobs": [j["job_id"] for j in jobs], "out_dir": str(out_dir)}, indent=2))
        return OvernightReport([], False, str(out_dir), 0.0)

    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    stopped = False

    # Write living status
    status_path = out_dir / "status.json"
    _atomic_write_json(
        status_path,
        {
            "phase": "start",
            "grid_id": grid.get("grid_id"),
            "stop_file": str(stop_file),
            "n_jobs": len(jobs),
        },
    )

    for job in jobs:
        if _global_stop(stop_file):
            print(f"[overnight] STOP file seen — graceful halt", flush=True)
            stopped = True
            break

        kind = job.get("kind", "preflop_blueprint")
        job_id = job["job_id"]

        if kind == "preflop_blueprint":
            if resume and _marker(out_dir, job_id).exists():
                print(f"[overnight] skip {job_id} (marker)", flush=True)
                results.append({"job_id": job_id, "status": "skipped"})
                continue
            if resume:
                promoted = promote_progress_if_any(out_dir, job_id, job)
                if promoted is not None:
                    print(
                        f"[overnight] promoted progress → strategy {job_id} "
                        f"iters={promoted.get('iterations')}",
                        flush=True,
                    )
                    results.append(promoted)
                    continue
            _atomic_write_json(
                status_path,
                {
                    "phase": "running",
                    "current_job": job_id,
                    "last_job": job_id,
                    "results": results,
                    "elapsed": time.perf_counter() - t0,
                },
            )
            try:
                results.append(
                    _run_blueprint(job, stop_file=str(stop_file), out_dir=out_dir)
                )
            except Exception as e:
                print(f"[overnight] FAIL {job_id}: {e}", flush=True)
                results.append(
                    {"job_id": job_id, "status": "error", "error": str(e)}
                )
            _atomic_write_json(
                status_path,
                {
                    "phase": "running",
                    "current_job": None,
                    "last_job": job_id,
                    "results": results,
                    "elapsed": time.perf_counter() - t0,
                },
            )
            continue

        if kind == "pipeline":
            board_ids = list(job.get("board_ids", []))
            for bid in board_ids:
                if _global_stop(stop_file):
                    stopped = True
                    break
                board = boards_map.get(bid)
                if board is None:
                    results.append(
                        {
                            "job_id": f"{job_id}_{bid}",
                            "status": "error",
                            "error": f"unknown board {bid}",
                        }
                    )
                    continue
                try:
                    results.append(
                        _run_pipeline_board(
                            job,
                            bid,
                            board,
                            stop_file=str(stop_file),
                            out_dir=out_dir,
                        )
                    )
                except Exception as e:
                    print(f"[overnight] FAIL {job_id}_{bid}: {e}", flush=True)
                    results.append(
                        {
                            "job_id": f"{job_id}_{bid}",
                            "status": "error",
                            "error": str(e),
                        }
                    )
                _atomic_write_json(
                    status_path,
                    {
                        "phase": "running",
                        "last_job": f"{job_id}_{bid}",
                        "results": results,
                        "elapsed": time.perf_counter() - t0,
                    },
                )
            if stopped:
                break
            continue

        results.append(
            {"job_id": job_id, "status": "error", "error": f"unknown kind {kind}"}
        )

    elapsed = time.perf_counter() - t0
    report = OvernightReport(
        results=results,
        stopped_early=stopped,
        out_dir=str(out_dir),
        seconds=elapsed,
    )
    _atomic_write_json(out_dir / "manifest.json", report.as_dict())
    _atomic_write_json(
        status_path,
        {"phase": "done", "stopped_early": stopped, **report.as_dict()},
    )
    print(json.dumps(report.as_dict(), indent=2), flush=True)
    return report
