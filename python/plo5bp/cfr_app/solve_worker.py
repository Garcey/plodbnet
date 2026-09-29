"""Child-process entry point for a CFR solve.

(review 2026-09-20 E1/E2/E10) The native solver can take the whole process down
(stack overflow, exit 0xC00000FD) and cannot be interrupted mid-iteration, so
the desktop app runs it in a ``multiprocessing`` *spawn* child:

- a native crash kills only the child; the parent reports ``error`` and the UI
  stays up;
- Stop / time budget can ``terminate()`` a child that ignores the stop file.

The stop / pause / progress FILE protocol is unchanged — those paths ride inside
the config dict, and the Rust solver reads/writes them exactly as before.

Results travel by file, never through a pipe: a report can exceed 100 MB, and a
large payload on a multiprocessing pipe deadlocks a parent that joins before it
drains. The native solver streams the report to ``result_path`` itself (TOOL-006:
no nested-dict / sanitize / json.dump copies of the strategy in the child) and the
child writes its scalars to ``<result_path>.meta.json`` — all the parent reads for
a saved job. On a Python-level failure it writes a small ``error_path`` instead. A
native crash writes neither, which the parent detects from the exit code.

Must stay importable with no side effects: ``spawn`` re-imports this module in
the child. Arguments are plain dicts/strings so they pickle.
"""

from __future__ import annotations

import traceback
from typing import Any


# One implementation for the whole CFR / GTO stack (TOOL-048). ``sanitize_json``
# replaces NaN / ±inf with None: ``json.dumps`` happily emits bare ``NaN``, which
# Starlette's JSONResponse then refuses — one NaN in a report used to 500 every
# ``/api/jobs`` poll (review 2026-09-20).
from plo5bp.gto.jsonio import atomic_write_json, sanitize_json  # noqa: E402

__all__ = ["meta_path_for", "run_solve", "sanitize_json", "write_json_atomic"]


def write_json_atomic(path: str, payload: Any) -> None:
    """Write strict JSON to ``path`` via a sibling temp file + ``os.replace``."""
    atomic_write_json(path, payload)


def meta_path_for(result_path: str) -> str:
    """The small scalars-only file beside a streamed report (TOOL-006)."""
    return f"{result_path}.meta.json"


def run_solve(
    root_d: dict[str, Any],
    config_d: dict[str, Any],
    result_path: str,
    error_path: str,
) -> None:
    """Solve ``root_d`` with ``config_d``; write the report to ``result_path``."""
    try:
        # Imported here, not at module top: keeps the spawn bootstrap light and
        # means an import failure is reported through error_path like any other.
        from plo5bp.cfr_app.session import _config_from_dict, _root_from_dict
        from plo5bp.gto.cfr_api import SolveReport, solve

        root = _root_from_dict(root_d)
        cfg = _config_from_dict(config_d)
        cfg.report_path = result_path
        # The user's range wording rides into the report's root, for the viewer.
        extra = {k: str(root_d[k]) for k in ("range_oop_text", "range_ip_text") if root_d.get(k)}
        report = solve(root, cfg, root_extra=extra)
        if isinstance(report, SolveReport) and report.streamed:
            write_json_atomic(meta_path_for(result_path), report.as_dict())
            return
        rep_d = report.as_dict() if isinstance(report, SolveReport) else dict(report)
        from plo5bp.cfr_app.ranges import attach_range_text

        attach_range_text(rep_d, root_d)
        write_json_atomic(result_path, rep_d)
    except BaseException as e:  # noqa: BLE001 — PyO3 PanicException is a BaseException
        try:
            write_json_atomic(
                error_path,
                {
                    "worker_error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()[-4000:],
                },
            )
        except OSError:
            pass
        # Non-zero exit so the parent treats this as a failure even if the
        # error file could not be written.
        raise SystemExit(1) from None
