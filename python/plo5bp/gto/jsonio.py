"""JSON file helpers shared by the CFR / GTO pipeline and the desktop app (TOOL-048).

There used to be five atomic-write helpers (different temp names, indent and NaN
behaviour) and two ``_jsonable`` copies. One set now:

- :func:`atomic_write_text` — sibling temp file (``<name>.<pid>.tmp``) +
  ``os.replace``: a kill mid-write never leaves a torn or empty file where a
  finished one is expected, and parallel writers never share a temp name;
- :func:`atomic_write_json` — strict JSON (NaN / ±inf become ``null``, because
  ``json.dumps`` would write bare ``NaN``, which strict readers — Starlette's
  JSONResponse, browsers — refuse). Compact by default: solve reports reach
  100+ MB and ``indent=2`` inflated them; small human-read files (manifests,
  plans, markers) pass ``indent=2``;
- :func:`sanitize_json` / :func:`jsonable` — value clean-ups.

Torch-free and import-light (the desktop app's spawn child imports it).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


def sanitize_json(obj: Any) -> Any:
    """Replace NaN / ±inf with None, recursively, so the result is strict JSON.

    (review 2026-09-20) ``json.dumps`` happily emits bare ``NaN``, which
    Starlette's JSONResponse then refuses (``allow_nan=False``) — one NaN in a
    report used to 500 every ``/api/jobs`` poll. Returns ``obj`` itself when
    there is nothing to replace in a scalar; containers are rebuilt.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    return obj


def jsonable(obj: Any) -> Any:
    """Coerce PyO3 / numpy leftovers (numpy scalars, bytes, Paths …) to JSON types."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, (bytes, bytearray)):
        return list(obj)
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    try:
        return obj.item()  # numpy scalar
    except Exception:
        return str(obj)


def temp_path_for(path: Path | str) -> Path:
    """The sibling temp file :func:`atomic_write_text` writes before the rename."""
    p = Path(path)
    return p.with_name(f"{p.name}.{os.getpid()}.tmp")


def atomic_write_text(path: Path | str, text: str) -> None:
    """Write ``text`` to a sibling temp file, then ``os.replace`` it over ``path``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_path_for(p)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def dumps_strict(obj: Any, *, indent: int | None = None) -> str:
    """``json.dumps`` of the sanitized object; compact separators unless indented."""
    clean = sanitize_json(obj)
    if indent is None:
        return json.dumps(clean, allow_nan=False, separators=(",", ":"))
    return json.dumps(clean, allow_nan=False, indent=indent)


def atomic_write_json(path: Path | str, obj: Any, *, indent: int | None = None) -> None:
    """Strict JSON (+ trailing newline) written atomically; see the module doc.

    Streams to the temp file (no second full-size string in memory).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_path_for(p)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                sanitize_json(obj),
                f,
                allow_nan=False,
                indent=indent,
                separators=None if indent is not None else (",", ":"),
            )
            f.write("\n")
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)
