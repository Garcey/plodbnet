"""Filesystem locations for the CFR desktop app.

(review 2026-09-20 J4) Every directory is resolved at CALL time so tests (and
alternate installs) can redirect all app output with one env var::

    CFR_APP_DATA_DIR=<dir>   # replaces <repo>/data/cfr

Nothing here creates directories — callers mkdir right before they write, so
importing the app has no filesystem side effects.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR_ENV = "CFR_APP_DATA_DIR"


def data_root() -> Path:
    """Root of all CFR app data (default ``<repo>/data/cfr``)."""
    override = os.environ.get(DATA_DIR_ENV, "").strip()
    if override:
        return Path(override)
    return REPO_ROOT / "data" / "cfr"


def jobs_dir() -> Path:
    return data_root() / "app_jobs"


def export_dir() -> Path:
    return data_root() / "app_export"


def uploads_dir() -> Path:
    return data_root() / "uploads"


def library_roots() -> list[Path]:
    """Directories the Library tab scans (most specific first)."""
    root = data_root()
    return [
        uploads_dir(),
        export_dir(),
        jobs_dir(),
        root / "overnight" / "strategies",
        root / "bench",
        root / "verify" / "batch" / "strategies",
        root / "pushfold_14_charts",
        root,
    ]
