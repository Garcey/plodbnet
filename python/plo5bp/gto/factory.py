"""Offline GTO label factory (smoke canaries + shard IO).

Production π* labels come from the native CFR solver:

  scripts/cfr_batch.py / scripts/cfr_solve.py
    → strategy JSON
    → scripts/cfr_export_labels.py  (source=rust_cfr)
    → scripts/gto_train_from_labels.py / scripts/train_policy_from_cfr.py

This factory only writes schema/metrics smoke shards (trash-fold / nuts-jam).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plo5bp.gto.cfr_api import rust_cfr_available
from plo5bp.gto.labels import (
    LabelRecord,
    make_smoke_label,
    read_jsonl,
    write_jsonl,
)
from plo5bp.gto.metrics import (
    evaluate_river_ev_loss,
    score_model_gates,
    smoke_metric_pass,
)
from plo5bp.gto.roots import (
    CLUBGG_NLH_ROOT,
    RootSample,
    iter_spr_grid,
    sample_train_roots,
)


DEFAULT_OUT_DIR = Path("data/gto_nlh")


@dataclass
class FactoryReport:
    n_roots: int
    n_labels: int
    solver_used: bool
    smoke_used: bool
    out_path: str
    metrics: dict[str, Any]
    river: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_roots": self.n_roots,
            "n_labels": self.n_labels,
            "solver_used": self.solver_used,
            "smoke_used": self.smoke_used,
            "out_path": self.out_path,
            "metrics": self.metrics,
            "river": self.river,
            "root": CLUBGG_NLH_ROOT.name,
        }


def collect_labels_for_roots(
    roots: list[RootSample],
    *,
    include_smoke: bool = True,
) -> tuple[list[LabelRecord], bool, bool]:
    """Write smoke canaries only. Real solves go through rust_cfr export."""
    del roots  # sampler still drives n_roots / manifest; labels are canaries
    labels: list[LabelRecord] = []
    smoke_used = False
    if include_smoke:
        labels.append(make_smoke_label(seed=1, trash_fold=True))
        labels.append(make_smoke_label(seed=2, trash_fold=False))
        smoke_used = True
    return labels, False, smoke_used


def run_factory(
    *,
    n_roots: int = 8,
    seed: int = 0,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    include_smoke: bool = True,
    use_grid: bool = False,
) -> FactoryReport:
    """Generate a smoke label shard + self-consistency metrics."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_grid:
        roots = list(iter_spr_grid(seed0=seed))
    else:
        roots = sample_train_roots(n_roots, seed=seed, seats=(2,), streets=(1, 2, 3))

    labels, solver_used, smoke_used = collect_labels_for_roots(
        roots, include_smoke=include_smoke
    )

    out_path = out_dir / f"labels_seed{seed}.jsonl"
    write_jsonl(out_path, labels)

    model_probs = [list(lab.gate_probs) for lab in labels]
    summary = score_model_gates(labels, model_probs)
    if include_smoke and not smoke_metric_pass(summary):
        raise RuntimeError(
            f"smoke metric canary failed: {summary.as_dict()}"
        )

    river = evaluate_river_ev_loss(labels)

    manifest = {
        "root": CLUBGG_NLH_ROOT.name,
        "n_roots": len(roots),
        "n_labels": len(labels),
        "cfr_available": rust_cfr_available(),
        "solver_used": solver_used,
        "smoke_used": smoke_used,
        "labels_path": str(out_path),
        "metrics": summary.as_dict(),
        "river": river.as_dict(),
        "teacher": "rust_cfr",
    }
    (out_dir / f"manifest_seed{seed}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return FactoryReport(
        n_roots=len(roots),
        n_labels=len(labels),
        solver_used=solver_used,
        smoke_used=smoke_used,
        out_path=str(out_path),
        metrics=summary.as_dict(),
        river=river.as_dict(),
    )


def load_shard(path: Path | str) -> list[LabelRecord]:
    return list(read_jsonl(path))
