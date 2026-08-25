"""Probe suite: PolicyNet vs solver holdout labels + pass/fail gates.

Used after supervised training to decide whether a checkpoint may claim
``GTO AI``. Bootstrap-only / unvalidated ckpts always fail the GTO badge
even if ``kind=gto_policy_net``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from plo5bp.gto.labels import LabelRecord, read_jsonl
from plo5bp.gto.metrics import argmax_gate, gate_kl, pure_node
from plo5bp.gto.obs_from_label import labels_to_supervised_rows
from plo5bp.gto.policy_net import load_policy_checkpoint, save_policy_checkpoint


# Default gates (tune after larger holdouts; documented in plan Workstream B).
DEFAULT_MIN_PURE_AGREE = 0.90
DEFAULT_MAX_MEAN_GATE_KL = 0.50  # clearly better than uniform (~1.1 nats worst)
DEFAULT_MIN_N = 1  # unit tests; production scripts raise this
DEFAULT_MIN_PURE_N = 0  # if 0 pure nodes, pure_agree gate is skipped


@dataclass
class ProbeReport:
    n: int
    pure_n: int
    pure_agree: float | None
    mean_gate_kl: float | None
    mean_gate_acc: float | None
    labels_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "pure_n": self.pure_n,
            "pure_agree": self.pure_agree,
            "mean_gate_kl": self.mean_gate_kl,
            "mean_gate_acc": self.mean_gate_acc,
            "labels_path": self.labels_path,
        }


@dataclass
class ProbeGates:
    """Pass/fail thresholds for holdout probe."""

    min_pure_agree: float = DEFAULT_MIN_PURE_AGREE
    max_mean_gate_kl: float = DEFAULT_MAX_MEAN_GATE_KL
    min_n: int = DEFAULT_MIN_N
    min_pure_n: int = DEFAULT_MIN_PURE_N

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProbeGateResult:
    passed: bool
    report: ProbeReport
    gates: ProbeGates
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "report": self.report.as_dict(),
            "gates": self.gates.as_dict(),
            "reasons": list(self.reasons),
        }


def probe_policy_vs_labels(
    model,
    labels: Sequence[LabelRecord],
    *,
    device: str = "cpu",
    pure_thresh: float = 0.90,
    labels_path: str | None = None,
) -> ProbeReport:
    device_t = torch.device(device)
    model = model.to(device_t).eval()
    rows = labels_to_supervised_rows(labels)
    if not rows:
        return ProbeReport(0, 0, None, None, None, labels_path=labels_path)

    kls: list[float] = []
    acc = 0
    pure_hits = pure_n = 0
    with torch.no_grad():
        for r in rows:
            obs = torch.from_numpy(r.obs).unsqueeze(0).to(device_t)
            gm = torch.from_numpy(r.gate_mask).unsqueeze(0).to(device_t)
            gl, _, _, _ = model(obs, gm)
            pred = F.softmax(gl, dim=-1).squeeze(0).cpu().numpy()
            pred = pred * r.gate_mask.astype(np.float32)
            pred = pred / max(float(pred.sum()), 1e-8)
            tgt = r.gate_probs
            kls.append(gate_kl(tgt, pred))
            if argmax_gate(tgt) == argmax_gate(pred):
                acc += 1
            if pure_node(tgt, pure_thresh):
                pure_n += 1
                if argmax_gate(tgt) == argmax_gate(pred):
                    pure_hits += 1

    n = len(rows)
    return ProbeReport(
        n=n,
        pure_n=pure_n,
        pure_agree=(pure_hits / pure_n) if pure_n else None,
        mean_gate_kl=float(sum(kls) / n),
        mean_gate_acc=acc / n,
        labels_path=labels_path,
    )


def evaluate_probe_gates(
    report: ProbeReport,
    gates: ProbeGates | None = None,
) -> ProbeGateResult:
    """Apply documented thresholds; return pass/fail + reasons."""
    gates = gates or ProbeGates()
    reasons: list[str] = []

    if report.n < int(gates.min_n):
        reasons.append(f"n={report.n} < min_n={gates.min_n}")

    if report.mean_gate_kl is None:
        reasons.append("mean_gate_kl=None (no rows)")
    elif report.mean_gate_kl > float(gates.max_mean_gate_kl):
        reasons.append(
            f"mean_gate_kl={report.mean_gate_kl:.4f} > "
            f"max={gates.max_mean_gate_kl}"
        )

    # Pure-node gate only when enough pure canaries exist
    if report.pure_n >= int(gates.min_pure_n) and report.pure_n > 0:
        if report.pure_agree is None:
            reasons.append("pure_agree=None with pure_n>0")
        elif report.pure_agree < float(gates.min_pure_agree):
            reasons.append(
                f"pure_agree={report.pure_agree:.4f} < "
                f"min={gates.min_pure_agree}"
            )
    elif int(gates.min_pure_n) > 0 and report.pure_n < int(gates.min_pure_n):
        reasons.append(
            f"pure_n={report.pure_n} < min_pure_n={gates.min_pure_n}"
        )

    return ProbeGateResult(
        passed=len(reasons) == 0,
        report=report,
        gates=gates,
        reasons=reasons,
    )


def probe_checkpoint(
    ckpt: Path | str,
    labels_path: Path | str,
    *,
    device: str = "cpu",
    gates: ProbeGates | None = None,
) -> ProbeGateResult:
    """Score checkpoint vs holdout JSONL; return gated result."""
    model, _ = load_policy_checkpoint(ckpt, device=device)
    path = Path(labels_path)
    labels = list(read_jsonl(path))
    report = probe_policy_vs_labels(
        model, labels, device=device, labels_path=str(path)
    )
    return evaluate_probe_gates(report, gates)


def stamp_probe_on_checkpoint(
    ckpt: Path | str,
    result: ProbeGateResult,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Rewrite checkpoint meta with probe dict + is_gto_validated flag.

    Does not change weights. ``is_gto`` stays True for kind stamp; serving
    uses :func:`plo5bp.gto.policy_net.is_validated_gto_meta`.
    """
    path = Path(ckpt)
    model, meta = load_policy_checkpoint(path, device=device)
    probe_dict = result.as_dict()
    meta = dict(meta)
    meta["probe"] = probe_dict
    meta["is_gto_validated"] = bool(result.passed)
    # Drop non-serializable / load-only keys before resave
    for k in ("path", "model", "actor", "policy"):
        meta.pop(k, None)
    save_policy_checkpoint(path, model, meta=meta)
    return meta


def source_is_gto_teacher(source: str | None) -> bool:
    """True when label source is the native rust_cfr teacher."""
    from plo5bp.gto.policy_net import source_is_gto_teacher as _fn

    return _fn(source)


def checkpoint_claims_gto(meta: dict[str, Any] | None) -> bool:
    """Badge-level GTO claim: rust_cfr source AND probe pass recorded."""
    from plo5bp.gto.policy_net import is_validated_gto_meta

    return is_validated_gto_meta(meta)


def write_probe_report(path: Path | str, result: ProbeGateResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
