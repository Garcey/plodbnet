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
drains. The child writes ``result_path`` atomically (tmp + ``os.replace``); on a
Python-level failure it writes a small ``error_path`` instead. A native crash
writes neither, which the parent detects from the exit code.

Must stay importable with no side effects: ``spawn`` re-imports this module in
the child. Arguments are plain dicts/strings so they pickle.
"""

from __future__ import annotations

import json
import math
import os
import traceback
from typing import Any


def sanitize_json(obj: Any) -> Any:
    """Replace NaN/±inf with None so the result is strict JSON.

    (review 2026-09-20) ``json.dumps`` happily emits bare ``NaN``, which
    Starlette's JSONResponse then refuses (``allow_nan=False``) — one NaN in a
    report used to 500 every ``/api/jobs`` poll.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    return obj


def write_json_atomic(path: str, payload: Any) -> None:
    """Write strict JSON to ``path`` via a temp file + ``os.replace``."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sanitize_json(payload), f, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


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
        report = solve(root, cfg)
        rep_d = report.as_dict() if isinstance(report, SolveReport) else dict(report)
        from plo5bp.cfr_app.ranges import attach_range_text

        attach_range_text(rep_d, root_d)  # user's range wording, for the viewer
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
