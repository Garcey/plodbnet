"""Phase 0 strength metrics for GTO labels vs model / vs exact river.

Metrics (architecture plan §6):
  - pure-node agreement (argmax gate match on near-pure labels)
  - gate KL(π* || π_model)
  - EV loss % pot (when value targets available)
  - river EV loss vs exact CFR (Mode 1 scorer hook)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from plo5bp.gto.labels import LabelRecord


@dataclass
class MetricSummary:
    n: int
    pure_node_agree: float | None
    mean_gate_kl: float | None
    mean_ev_loss_pct_pot: float | None
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "pure_node_agree": self.pure_node_agree,
            "mean_gate_kl": self.mean_gate_kl,
            "mean_ev_loss_pct_pot": self.mean_ev_loss_pct_pot,
            "notes": self.notes,
        }


def gate_kl(p_star: Sequence[float], p_model: Sequence[float], eps: float = 1e-8) -> float:
    """KL(p* || p_model) over the 3-gate simplex."""
    if len(p_star) != 3 or len(p_model) != 3:
        raise ValueError("gate probs must be length 3")
    kl = 0.0
    for ps, pm in zip(p_star, p_model):
        ps = max(float(ps), 0.0)
        if ps <= 0.0:
            continue
        pm = max(float(pm), eps)
        kl += ps * math.log(ps / pm)
    return float(kl)


def pure_node(p_star: Sequence[float], thresh: float = 0.90) -> bool:
    return max(float(x) for x in p_star) >= thresh


def argmax_gate(probs: Sequence[float]) -> int:
    best_i, best_v = 0, -1.0
    for i, v in enumerate(probs):
        if float(v) > best_v:
            best_v = float(v)
            best_i = i
    return best_i


def score_model_gates(
    labels: Iterable[LabelRecord],
    model_gate_probs: Sequence[Sequence[float]],
    *,
    pure_thresh: float = 0.90,
) -> MetricSummary:
    """Compare a batch of labels to model gate distributions (aligned lists)."""
    labels_l = list(labels)
    if len(labels_l) != len(model_gate_probs):
        raise ValueError(
            f"labels ({len(labels_l)}) vs model preds ({len(model_gate_probs)}) length mismatch"
        )
    pure_hits = pure_n = 0
    kls: list[float] = []
    for lab, mp in zip(labels_l, model_gate_probs):
        kls.append(gate_kl(lab.gate_probs, mp))
        if pure_node(lab.gate_probs, pure_thresh):
            pure_n += 1
            if argmax_gate(lab.gate_probs) == argmax_gate(mp):
                pure_hits += 1
    return MetricSummary(
        n=len(labels_l),
        pure_node_agree=(pure_hits / pure_n) if pure_n else None,
        mean_gate_kl=(sum(kls) / len(kls)) if kls else None,
        mean_ev_loss_pct_pot=None,
        notes=f"pure_n={pure_n}",
    )


def smoke_metric_pass(summary: MetricSummary) -> bool:
    """Day-0 factory canary: smoke labels must be self-consistent.

    For synthetic smoke we score the label against itself — pure-node
    agree == 1.0 and KL == 0.
    """
    if summary.n <= 0:
        return False
    if summary.mean_gate_kl is None or summary.mean_gate_kl > 1e-9:
        return False
    if summary.pure_node_agree is not None and summary.pure_node_agree < 1.0:
        return False
    return True


# --- River exact EV loss (Mode 1 seed; full CFR lands with Rust later) ------


@dataclass
class RiverEvReport:
    """Placeholder report until Rust HU river CFR is wired."""

    n: int
    mean_ev_loss_pct_pot: float | None
    available: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_ev_loss_pct_pot": self.mean_ev_loss_pct_pot,
            "available": self.available,
            "reason": self.reason,
        }


def river_exact_available() -> bool:
    """True when the native rust_cfr solve binding is importable."""
    try:
        from plo5bp.gto.cfr_api import rust_cfr_available

        return rust_cfr_available()
    except Exception:
        return False


def evaluate_river_ev_loss(
    labels: Iterable[LabelRecord],
) -> RiverEvReport:
    """Score river labels vs exact HU CFR when the engine exposes it."""
    labs = [lab for lab in labels if lab.street == 3]
    if not river_exact_available():
        return RiverEvReport(
            n=len(labs),
            mean_ev_loss_pct_pot=None,
            available=False,
            reason="native rust_cfr binding not available — rebuild extension",
        )
    # Future: call engine per label, compute |v* - v_label| / pot.
    return RiverEvReport(
        n=len(labs),
        mean_ev_loss_pct_pot=None,
        available=True,
        reason="entrypoint present but batch scorer not yet implemented",
    )
