"""plo5bp — PLO5 double-board bomb-pot self-play PPO pipeline.

The convenience names below are resolved lazily (PEP 562, TOOL-020): importing
ANY ``plo5bp.*`` module imports this package first, and the old eager imports
(encoding -> sizing -> torch, ~1.5 s) made torch-free tools — the CFR desktop
app and its solve child, the batch / export CLIs, OCR — pay for torch. Modules
that need the encoder or the env import them directly, as they always did.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY: dict[str, str] = {
    "GameConfig": "plo5bp.config",
    "TrainingConfig": "plo5bp.config",
    "OBS_DIM": "plo5bp.encoding",
    "encode_observation": "plo5bp.encoding",
    "BombPotEnv": "plo5bp.env",
    "ACTION_NAMES": "plo5bp.actions",
    "NUM_ACTIONS": "plo5bp.actions",
}

__all__ = [
    "ACTION_NAMES",
    "BombPotEnv",
    "GameConfig",
    "NUM_ACTIONS",
    "OBS_DIM",
    "TrainingConfig",
    "encode_observation",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
