"""FastAPI backend for the standalone CFR desktop solver app.

Run::

    .venv/Scripts/python scripts/cfr_app.py
    # or: uvicorn plo5bp.cfr_app.server:app --port 8766
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from plo5bp.cfr_app.session import SolveSession, root_presets
from plo5bp.cfr_app.strategy_view import (
    filter_rows,
    list_strategy_library,
    load_report,
    matrix_for_rows,
    summarize_report_light,
)
from plo5bp.cfr_app.tree_model import build_abstract_tree, compare_strategies
from plo5bp.gto.cfr_api import SIZE_PRESETS, rust_cfr_available
from plo5bp.gto.preflop_class import preflop_class_label
from plo5bp.gto.roots import CLUBGG_NLH_ROOT

logger = logging.getLogger("plo5bp.cfr_app")

STATIC_DIR = Path(__file__).parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[3]

app = FastAPI(title="CFR Solver Desktop", version="0.3.0")
session = SolveSession()

# In-memory cache of last fully loaded view (for filter paging without re-parse)
_view_cache: dict[str, Any] = {"key": None, "view": None}


class RootBody(BaseModel):
    street: int = 3
    pot_bb: float = 10.0
    effective_stack_bb: float = 50.0
    stack_bb: float | None = None  # alias
    board: list[int] = Field(default_factory=list)
    num_seats: int = 2
    bb_chips: int = CLUBGG_NLH_ROOT.bb
    sb_chips: int = CLUBGG_NLH_ROOT.sb
    ante_chips: int = CLUBGG_NLH_ROOT.ante
    raise_sizes_pm: list[int] | None = None
    size_preset: str | None = "standard"
    allin_atom: bool = True
    range_ip: str = ""
    range_oop: str = ""
    stacks_bb: list[float] = Field(default_factory=list)
    root_id: str = ""
    algorithm: str | None = None  # carried for UI convenience
    card_abstraction: str | None = None


class ConfigBody(BaseModel):
    # 0 = unlimited (play until pause/stop)
    max_iterations: int = 200
    target_exploitability_bb: float = 0.5
    thread_num: int = 1
    seed: int = 0
    use_isomorphism: bool = True
    algorithm: str = "dcfr"
    card_abstraction: str = "none"
    time_budget_secs: float = 0.0
    poll_every: int = 50
    unlimited: bool = False


class SolveRequest(BaseModel):
    root: RootBody
    config: ConfigBody = Field(default_factory=ConfigBody)
    save: bool = True
    label: str = ""


class KuhnRequest(BaseModel):
    iterations: int = 5000


class LoadPathRequest(BaseModel):
    path: str


class FilterRequest(BaseModel):
    source: str  # job_id or file path
    seat: int | None = None
    path: str | None = None
    hand_query: str = ""
    limit: int = 200
    offset: int = 0
    include_matrix: bool = True


class CompareRequest(BaseModel):
    path_a: str
    path_b: str
    seat: int | None = None
    path_filter: str | None = None


class ExportRequest(BaseModel):
    path: str
    out_name: str = ""


# ---------------------------------------------------------------------------
# Static + health
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "rust_cfr": rust_cfr_available(),
        "app": "cfr_desktop",
        "version": "0.3.0",
    }


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    return {
        "size_presets": {k: list(v) for k, v in SIZE_PRESETS.items()},
        "streets": [
            {"id": 0, "name": "Preflop", "board_cards": 0},
            {"id": 1, "name": "Flop", "board_cards": 3},
            {"id": 2, "name": "Turn", "board_cards": 4},
            {"id": 3, "name": "River", "board_cards": 5},
        ],
        "algorithms": ["dcfr", "mccfr_es"],
        "card_abstractions": ["none", "ochs"],
        "clubgg": {
            "bb": CLUBGG_NLH_ROOT.bb,
            "sb": CLUBGG_NLH_ROOT.sb,
            "ante": CLUBGG_NLH_ROOT.ante,
            "name": CLUBGG_NLH_ROOT.name,
        },
        "presets": root_presets(),
        "preflop_labels": [preflop_class_label(i) for i in range(169)],
        "rust_cfr": rust_cfr_available(),
    }


# ---------------------------------------------------------------------------
# Solve jobs
# ---------------------------------------------------------------------------


@app.post("/api/solve")
def api_solve(body: SolveRequest) -> dict[str, Any]:
    root_d = body.root.model_dump()
    # Prefer explicit stack_bb when client sent it (effective_stack_bb has a default)
    if body.root.stack_bb is not None:
        # model_fields_set is pydantic v2; fall back to truthy override
        fields_set = getattr(body.root, "model_fields_set", None) or set()
        if "stack_bb" in fields_set or (
            "effective_stack_bb" not in fields_set and body.root.stack_bb != 50.0
        ):
            root_d["effective_stack_bb"] = float(body.root.stack_bb)
    cfg_d = body.config.model_dump()
    # Allow algorithm / card_abstraction on root for preset convenience
    if body.root.algorithm:
        cfg_d["algorithm"] = body.root.algorithm
    if body.root.card_abstraction:
        cfg_d["card_abstraction"] = body.root.card_abstraction
    # Preflop defaults to mccfr_es when user left dcfr
    if int(root_d.get("street", 3)) == 0 and cfg_d.get("algorithm") == "dcfr":
        cfg_d["algorithm"] = "mccfr_es"
    try:
        job = session.start(root_d, cfg_d, save=body.save, label=body.label)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return job.as_dict_light()


@app.post("/api/solve/kuhn")
def api_solve_kuhn(body: KuhnRequest) -> dict[str, Any]:
    try:
        job = session.start_kuhn(iterations=body.iterations)
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return job.as_dict_light()


@app.post("/api/solve/stop")
def api_stop(job_id: str | None = None) -> dict[str, Any]:
    try:
        return session.stop(job_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/api/solve/pause")
def api_pause(job_id: str | None = None) -> dict[str, Any]:
    try:
        return session.pause(job_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/api/solve/resume")
def api_resume(job_id: str | None = None) -> dict[str, Any]:
    try:
        return session.resume(job_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@app.get("/api/jobs")
def api_jobs() -> dict[str, Any]:
    return {"jobs": session.list_jobs(), "active": session.active_job()}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str, full: bool = False) -> dict[str, Any]:
    j = session.get_job(job_id, full=full)
    if j is None:
        raise HTTPException(404, f"job {job_id} not found")
    return j


@app.get("/api/jobs/{job_id}/view")
def api_job_view(
    job_id: str,
    seat: int | None = None,
    path: str | None = None,
    hand_query: str = "",
    limit: int = 200,
    offset: int = 0,
) -> dict[str, Any]:
    # Pull latest progress snapshot (live strategy while solving)
    session.refresh_progress(job_id)
    j = session.get_job(job_id, full=True)
    if j is None:
        raise HTTPException(404, f"job {job_id} not found")
    rep = j.get("report")
    if not rep:
        raise HTTPException(
            409,
            f"job {job_id} has no report yet (status={j.get('status')}) — wait for first progress tick",
        )
    try:
        view = load_report(rep)
    except Exception as e:
        raise HTTPException(400, f"view failed: {e}") from e
    _view_cache["key"] = f"job:{job_id}"
    _view_cache["view"] = view
    return _view_payload(
        view,
        seat=seat,
        path=path,
        hand_query=hand_query,
        limit=limit,
        offset=offset,
        extra={
            "job": {
                "job_id": job_id,
                "status": j.get("status"),
                "error": j.get("error"),
                "iterations_run": j.get("iterations_run"),
                "live": j.get("status") in ("running", "paused", "queued"),
            }
        },
    )


# ---------------------------------------------------------------------------
# Library / load files
# ---------------------------------------------------------------------------


@app.get("/api/library")
def api_library(max_files: int = 300, peek: bool = True) -> dict[str, Any]:
    items = list_strategy_library(max_files=max_files)
    if peek:
        for it in items:
            # Skip huge dumps (e.g. 300k-iter push/fold) — listing must stay instant.
            if int(it.get("size") or 0) > 2_000_000:
                it["kind"] = "large"
                continue
            try:
                s = summarize_report_light(it["path"])
                it["kind"] = s.get("kind")
                it["street"] = s.get("street")
                it["board_str"] = s.get("board_str")
                it["iterations_run"] = s.get("iterations_run")
                it["exploitability_bb"] = s.get("exploitability_bb")
                it["num_infosets"] = s.get("num_infosets")
                it["root_id"] = s.get("root_id")
            except Exception:
                it["kind"] = "unknown"
    return {"items": items, "count": len(items)}


@app.get("/api/library/peek")
def api_library_peek(path: str) -> dict[str, Any]:
    p = _safe_path(path)
    try:
        return summarize_report_light(p)
    except Exception as e:
        raise HTTPException(400, f"cannot peek {p}: {e}") from e


@app.post("/api/library/load")
def api_library_load(body: LoadPathRequest) -> dict[str, Any]:
    p = _safe_path(body.path)
    try:
        job = session.load_report_file(p)
    except Exception as e:
        raise HTTPException(400, f"load failed: {e}") from e
    # also warm view cache
    try:
        view = load_report(p)
        _view_cache["key"] = f"file:{p}"
        _view_cache["view"] = view
    except Exception as e:
        logger.warning("view load failed: %s", e)
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "out_path": job.get("out_path"),
        "root": job.get("root"),
        "summary_light": summarize_report_light(p),
    }


@app.get("/api/view")
def api_view(
    path: str,
    seat: int | None = None,
    path_filter: str | None = None,
    hand_query: str = "",
    limit: int = 200,
    offset: int = 0,
) -> dict[str, Any]:
    p = _safe_path(path)
    cache_key = f"file:{p}"
    if _view_cache.get("key") == cache_key and _view_cache.get("view"):
        view = _view_cache["view"]
    else:
        try:
            view = load_report(p)
        except Exception as e:
            raise HTTPException(400, f"view failed: {e}") from e
        _view_cache["key"] = cache_key
        _view_cache["view"] = view
    return _view_payload(
        view,
        seat=seat,
        path=path_filter,
        hand_query=hand_query,
        limit=limit,
        offset=offset,
    )


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a SolveReport / chart JSON into the viewer (GTOW-style open file)."""
    name = (file.filename or "upload.json").strip()
    if not name.lower().endswith(".json"):
        raise HTTPException(400, "only .json strategy files are accepted")
    # sanitize filename
    safe = re.sub(r"[^\w.\-]+", "_", Path(name).name)[:120] or "upload.json"
    if not safe.lower().endswith(".json"):
        safe += ".json"
    up_dir = REPO_ROOT / "data" / "cfr" / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = up_dir / f"{stamp}_{safe}"
    raw = await file.read()
    if len(raw) > 80 * 1024 * 1024:
        raise HTTPException(400, "file too large (max 80 MB)")
    if not raw.strip():
        raise HTTPException(400, "empty file")
    try:
        text = raw.decode("utf-8")
        data = __import__("json").loads(text)
    except Exception as e:
        raise HTTPException(400, f"invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise HTTPException(400, "JSON root must be an object (SolveReport or chart)")
    # Accept SolveReport, chart, or bare strategy blob
    if not (
        "strategy" in data
        or "hands" in data
        or "infosets" in data
        or (isinstance(data.get("strategy"), dict) and "infosets" in (data.get("strategy") or {}))
    ):
        raise HTTPException(
            400,
            "unrecognized strategy format — need strategy.infosets, hands[], or infosets[]",
        )
    dest.write_text(text, encoding="utf-8")
    try:
        view = load_report(dest)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"could not parse strategy: {e}") from e
    _view_cache["key"] = f"file:{dest.resolve()}"
    _view_cache["view"] = view
    try:
        job = session.load_report_file(dest)
        job_id = job.get("job_id")
    except Exception:
        job_id = None
    return {
        "ok": True,
        "path": str(dest.resolve()),
        "rel": f"data/cfr/uploads/{dest.name}",
        "job_id": job_id,
        "summary": view["summary"],
        "num_infosets": view["summary"].get("num_infosets"),
        "kind": view["summary"].get("kind"),
    }


@app.post("/api/validate_root")
def api_validate_root(body: RootBody) -> dict[str, Any]:
    from plo5bp.cfr_app.session import _root_from_dict

    try:
        r = _root_from_dict(body.model_dump())
        r.validate()
        return {"ok": True, "root": r.as_dict()}
    except ValueError as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/tree/preview")
def api_tree_preview(body: RootBody) -> dict[str, Any]:
    """Build the abstract bet-size game tree for the configured root."""
    from plo5bp.cfr_app.session import _root_from_dict

    try:
        r = _root_from_dict(body.model_dump())
        r.validate()
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    tree = build_abstract_tree(r, max_nodes=350, max_depth=7)
    return tree


@app.post("/api/compare")
def api_compare(body: CompareRequest) -> dict[str, Any]:
    pa = _safe_path(body.path_a)
    pb = _safe_path(body.path_b)
    try:
        va = load_report(pa)
        vb = load_report(pb)
    except Exception as e:
        raise HTTPException(400, f"compare load failed: {e}") from e
    rows_a = va["rows"]
    rows_b = vb["rows"]
    if body.seat is not None:
        rows_a = [r for r in rows_a if int(r.get("seat", 0)) == int(body.seat)]
        rows_b = [r for r in rows_b if int(r.get("seat", 0)) == int(body.seat)]
    if body.path_filter:
        rows_a = [r for r in rows_a if str(r.get("path") or "") == body.path_filter]
        rows_b = [r for r in rows_b if str(r.get("path") or "") == body.path_filter]
    return {
        "a": {"path": str(pa), "summary": va["summary"]},
        "b": {"path": str(pb), "summary": vb["summary"]},
        "diff": compare_strategies(rows_a, rows_b),
    }


@app.post("/api/export")
def api_export(body: ExportRequest) -> dict[str, Any]:
    """Copy a strategy into data/cfr/app_export/ for easy access / training handoff."""
    src = _safe_path(body.path)
    out_dir = REPO_ROOT / "data" / "cfr" / "app_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Basename only — block path traversal via out_name
    raw_name = body.out_name.strip() or src.name
    name = Path(raw_name).name
    name = re.sub(r"[^\w.\-]+", "_", name)[:120] or "export.json"
    if not name.lower().endswith(".json"):
        name += ".json"
    dest = (out_dir / name).resolve()
    try:
        dest.relative_to(out_dir.resolve())
    except ValueError as e:
        raise HTTPException(400, "export path escapes app_export") from e
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return {"ok": True, "path": str(dest), "rel": f"data/cfr/app_export/{name}"}


@app.get("/api/jobs/{job_id}/progress")
def api_job_progress(job_id: str) -> dict[str, Any]:
    """Richer progress snapshot for long solves (status + report partials + timing)."""
    session.refresh_progress(job_id)
    j = session.get_job(job_id, full=False)
    if j is None:
        raise HTTPException(404, f"job {job_id} not found")
    started = j.get("started_at")
    finished = j.get("finished_at")
    now = __import__("time").time()
    elapsed = None
    if started:
        end = finished or now
        elapsed = round(end - started, 2)
    rep = j.get("report") or {}
    n_info = j.get("num_infosets") or (rep.get("strategy") or {}).get("num_infosets") or 0
    # Light job strips infosets — use counters / status for live flag.
    has_live = (
        j.get("status") in ("running", "paused", "queued")
        and int(n_info or 0) > 0
    ) or (
        isinstance(rep.get("strategy"), dict)
        and bool((rep.get("strategy") or {}).get("infosets"))
    )
    return {
        "job_id": job_id,
        "status": j.get("status"),
        "progress_message": j.get("progress_message"),
        "elapsed_secs": elapsed,
        "iterations_run": j.get("iterations_run") or rep.get("iterations_run"),
        "exploitability_bb": (
            j.get("exploitability_bb")
            if j.get("exploitability_bb") is not None
            else rep.get("exploitability_bb")
        ),
        "num_infosets": n_info,
        "has_live_strategy": bool(has_live),
        "unlimited": bool(j.get("unlimited")),
        "notes": (j.get("notes") or [])[-8:],
        "error": j.get("error"),
        "out_path": j.get("out_path"),
        "root": j.get("root"),
        "config": j.get("config"),
    }


def _view_payload(
    view: dict[str, Any],
    *,
    seat: int | None = None,
    path: str | None = None,
    hand_query: str = "",
    limit: int = 200,
    offset: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Filter rows + rebuild 13×13 matrix for the selected decision node."""
    page = filter_rows(
        view["rows"],
        seat=seat,
        path=path,
        hand_query=hand_query,
        limit=limit,
        offset=offset,
    )
    # Full filtered set (no page cap) for matrix of the selected node
    filtered = filter_rows(
        view["rows"],
        seat=seat,
        path=path,
        hand_query=hand_query,
        limit=max(len(view["rows"]), 1),
        offset=0,
    )["rows"]
    street = (view.get("summary") or {}).get("street")
    matrix = view.get("matrix")
    if seat is not None or (path is not None and path != "") or hand_query:
        matrix = matrix_for_rows(filtered, street=street)
    elif matrix is None:
        matrix = matrix_for_rows(view["rows"], street=street)
    out: dict[str, Any] = {
        "summary": view["summary"],
        "nodes": view["nodes"],
        "matrix": matrix,
        "solution_tree": view.get("solution_tree"),
        "line_nav": view.get("line_nav"),
        "chart_pack": view.get("chart_pack"),
        "page": page,
    }
    if extra:
        out.update(extra)
    return out


def _safe_path(path: str) -> Path:
    """Resolve path; allow absolute or repo-relative; block path traversal escapes."""
    raw = Path(path)
    if not raw.is_absolute():
        cand = (REPO_ROOT / raw).resolve()
    else:
        cand = raw.resolve()
    if not cand.exists():
        raise HTTPException(404, f"path not found: {path}")
    # Prefer files under repo or data/cfr app_jobs
    try:
        cand.relative_to(REPO_ROOT.resolve())
        return cand
    except ValueError:
        # allow absolute under work_dir
        try:
            cand.relative_to(session.work_dir.resolve())
            return cand
        except ValueError as e:
            raise HTTPException(403, f"path outside allowed roots: {path}") from e
