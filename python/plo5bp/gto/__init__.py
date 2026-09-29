"""NLH GTO stack: StrategyBackend hosts, offline labels, metrics, PolicyNet.

Teacher is the native rust_engine CFR solver (``source=rust_cfr*``).
PolicyNet trains on exported LabelRecords and serves Trainer / Study.

See ``docs/plans/nlh-neural-gto-solver-ground-up-architecture.md``.

Import cost (TOOL-020): the names below are resolved LAZILY (PEP 562 module
``__getattr__``). ``import plo5bp.gto.cfr_api`` — the desktop CFR app, its
solve child and every batch / export script — used to pay for torch (~2 s)
because this package eagerly imported the PolicyNet and backend modules. The
public names still work (``from plo5bp.gto import solve``); the heavy module is
imported the first time one of its names is asked for. Keep ``cfr_api``,
``cfr_batch``, ``teacher``, ``roots``, ``iso`` and ``preflop_class`` free of
top-level torch imports (``tests/python/gto/test_gto_import_cost.py``).
"""

from __future__ import annotations

import importlib
from typing import Any

# public name -> the submodule that defines it
_LAZY: dict[str, str] = {
    "NodeDist": "plo5bp.gto.backend",
    "PpoSolverHost": "plo5bp.gto.backend",
    "StrategyBackend": "plo5bp.gto.backend",
    "make_ppo_host": "plo5bp.gto.backend",
    "collect_bootstrap": "plo5bp.gto.bootstrap",
    "label_node": "plo5bp.gto.bootstrap",
    "rust_cfr_available": "plo5bp.gto.cfr_api",
    "solve": "plo5bp.gto.cfr_api",
    "LabelRecord": "plo5bp.gto.labels",
    "make_smoke_label": "plo5bp.gto.labels",
    "PolicyNetHost": "plo5bp.gto.policy_host",
    "load_policy_host": "plo5bp.gto.policy_host",
    "try_load_gto_host": "plo5bp.gto.policy_host",
    "build_policy_net": "plo5bp.gto.policy_net",
    "is_gto_checkpoint": "plo5bp.gto.policy_net",
    "is_validated_gto_meta": "plo5bp.gto.policy_net",
    "load_policy_checkpoint": "plo5bp.gto.policy_net",
    "source_is_gto_teacher": "plo5bp.gto.policy_net",
    "CLUBGG_NLH_ROOT": "plo5bp.gto.roots",
    "ClubGGRoot": "plo5bp.gto.roots",
    "RootSample": "plo5bp.gto.roots",
    "sample_train_roots": "plo5bp.gto.roots",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # resolve once
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
