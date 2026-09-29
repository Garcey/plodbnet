"""The compiled engine (plo5bp._engine) must be CURRENT — one check, at import.

The engine is built from this repo (`.venv/Scripts/maturin develop --release`
from the repo root; scripts/deploy_prod.sh rebuilds it on the server). The
Python side used to probe each newer engine feature with hasattr / try-import
and quietly fall back to an older, slower or different code path, so a stale
binary trained or encoded differently without a word (2026-09-28, ML-028:
about 15 such gates in the rollout, the batched env, the encoders and the
compact storage). Now each module names what it needs here, once, at import,
and a stale binary is an `EngineOutOfDate` (an ImportError) that says what is
missing and how to rebuild.

The test session goes one step further (tests/conftest.py): it compares the
binary's SOURCE_HASH with the Rust sources on disk, which also catches a stale
build that still has every name.
"""

from __future__ import annotations

from typing import Iterable

from plo5bp import _engine

REBUILD_HINT = (
    "rebuild it from the repo root: .venv/Scripts/maturin develop --release "
    "(on the pod: .venv/bin/maturin develop --release)"
)

_MISSING = object()


class EngineOutOfDate(ImportError):
    """The compiled engine lacks a function, method or constant the Python
    side needs: it was built from older sources."""


def missing(names: Iterable[str]) -> list[str]:
    """The names (module attributes, or dotted `Class.attr`) the engine lacks."""
    out = []
    for name in names:
        obj: object = _engine
        for part in name.split("."):
            obj = getattr(obj, part, _MISSING)
            if obj is _MISSING:
                out.append(name)
                break
    return out


def require(*names: str) -> None:
    """Raise `EngineOutOfDate` unless the engine has every name."""
    gone = missing(names)
    if gone:
        raise EngineOutOfDate(
            f"plo5bp._engine is out of date: it has no {', '.join(gone)} -- "
            + REBUILD_HINT
        )


def functions(*names: str) -> tuple:
    """`require` the module-level names, then return them in order."""
    require(*names)
    return tuple(getattr(_engine, n) for n in names)
