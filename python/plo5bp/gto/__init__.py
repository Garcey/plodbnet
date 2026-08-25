"""NLH GTO stack: StrategyBackend hosts, offline labels, metrics, PolicyNet.

Teacher is the native rust_engine CFR solver (``source=rust_cfr*``).
PolicyNet trains on exported LabelRecords and serves Trainer / Study.

See ``.claude/plans/nlh-neural-gto-solver-ground-up-architecture.md``.
"""

from plo5bp.gto.backend import (
    NodeDist,
    PpoSolverHost,
    StrategyBackend,
    make_ppo_host,
)
from plo5bp.gto.bootstrap import collect_bootstrap, label_node
from plo5bp.gto.cfr_api import rust_cfr_available, solve
from plo5bp.gto.labels import LabelRecord, make_smoke_label
from plo5bp.gto.policy_host import PolicyNetHost, load_policy_host, try_load_gto_host
from plo5bp.gto.policy_net import (
    build_policy_net,
    is_gto_checkpoint,
    is_validated_gto_meta,
    load_policy_checkpoint,
    source_is_gto_teacher,
)
from plo5bp.gto.roots import (
    CLUBGG_NLH_ROOT,
    ClubGGRoot,
    RootSample,
    sample_train_roots,
)

__all__ = [
    "NodeDist",
    "PpoSolverHost",
    "PolicyNetHost",
    "StrategyBackend",
    "make_ppo_host",
    "load_policy_host",
    "try_load_gto_host",
    "build_policy_net",
    "is_gto_checkpoint",
    "is_validated_gto_meta",
    "load_policy_checkpoint",
    "source_is_gto_teacher",
    "LabelRecord",
    "make_smoke_label",
    "collect_bootstrap",
    "label_node",
    "rust_cfr_available",
    "solve",
    "CLUBGG_NLH_ROOT",
    "ClubGGRoot",
    "RootSample",
    "sample_train_roots",
]
