"""Probe suite: PolicyNet vs solver holdout labels + pass/fail gates.

Used after supervised training to decide whether a checkpoint may claim
``GTO AI``. Bootstrap-only / unvalidated ckpts always fail the GTO badge
even if ``kind=gto_policy_net``.

What is scored (review 2026-09-20 D4 — the old probe was gates-only, passed a
uniform policy, passed NaN, and skipped its pure-node gate silently):

- **Gates** — mean ``KL(π* || π)`` over the 3 gates, argmax accuracy, and
  agreement on near-pure nodes. The KL has an ABSOLUTE cap and a RELATIVE one
  against the *card-blind per-node mean* policy computed on the same holdout
  (the best policy that ignores hole cards). A net that does not beat it by a
  margin has learned node frequencies, not a strategy.
- **Sizing** — ``p_raise``-weighted anchor KL over the LEGAL anchors of each
  raise row, plus the jam frequency ``P(raise)·P(jam anchor | raise)``: its
  holdout-wide gap ``|mean target − mean model|`` and the per-row mean
  ``|target − model|``. The refine head is untrained and unused at serve (D5),
  so anchors ARE the sizing. A net trained on pre-D1 labels (jam mass erased)
  shows up here at once: ~0 jam frequency against a 0.3 target and an anchor
  KL of several nats.
- **Hygiene** — every comparison is written ``not (x <= max)`` so NaN fails;
  non-finite predictions fail outright; ``pure_n == 0`` fails (the gate is
  never skipped silently); the holdout roots must be DISJOINT from the
  checkpoint's recorded training roots, and a checkpoint with no recorded
  roots cannot be verified, so it fails too.

Calibration (HU river SPR-2 roots, micro menu; repro ``g_trivial_probe`` and
a 3-root train-2 / probe-1 run on 2026-09-20):

==========================  =======  ==========  ===========  =========
policy                      gate KL  pure-agree  anchor KL    jam gaps
==========================  =======  ==========  ===========  =========
uniform over legal          0.26–.36   0.28–.31   0.73–1.01   .14 / .22
card-blind per-node mean    0.18–.26   0.76–.80   ~0.06       —
256×2 net, production size  ~0.075     ~0.99      —           —
256×2 toy (30k labels)      0.13       0.95       0.06        .02 / .13
==========================  =======  ==========  ===========  =========

(jam gaps = holdout-wide / per-row.) Sizing given a raise is only weakly
card-dependent on the micro menu — the card-blind anchor KL is about what a
trained net reaches — so there is deliberately NO relative sizing gate; the
card-blind sizing baseline is reported for context only.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from plo5bp.gto.dataset import SupervisedRow
from plo5bp.gto.labels import LabelRecord, read_jsonl
from plo5bp.gto.obs_from_label import ObsSynthesisStats, labels_to_supervised_rows
from plo5bp.gto.policy_net import (
    label_provenance_problem,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from plo5bp.gto.train import derive_training_provenance
from plo5bp.sizing import NLH_ANCHOR_SPEC, anchor_grid_torch


# Default gates — see the calibration note in the module docstring.
DEFAULT_MIN_PURE_AGREE = 0.90
DEFAULT_MAX_MEAN_GATE_KL = 0.12  # uniform ≈ 0.26–0.36, card-blind ≈ 0.18
DEFAULT_MAX_GATE_KL_VS_CARD_BLIND = 0.60  # model KL <= 0.6 × card-blind KL
DEFAULT_MAX_MEAN_ANCHOR_KL = 0.15  # uniform over legal anchors ≈ 0.7–1.0
DEFAULT_MAX_JAM_FREQ_GAP = 0.05  # holdout-wide |target − model| jam frequency
DEFAULT_MAX_MEAN_JAM_GAP = 0.15  # per-row mean |target − model|
DEFAULT_MIN_N = 1  # unit tests; production scripts raise this
DEFAULT_MIN_PURE_N = 1  # 0 pure nodes = the pure gate is UNVERIFIED → fail
# Below this a baseline carries no card information worth beating.
_BASELINE_FLOOR = 0.02
_PROBE_BATCH = 4096


@dataclass
class ProbeReport:
    n: int
    pure_n: int
    pure_agree: float | None
    mean_gate_kl: float | None
    mean_gate_acc: float | None
    labels_path: str | None = None
    # Sizing (p_raise-weighted over rows where raising is legal).
    n_raise_rows: int = 0
    mean_anchor_kl: float | None = None
    mean_jam_gap: float | None = None
    jam_freq_gap: float | None = None
    jam_freq_target: float | None = None
    jam_freq_model: float | None = None
    # Trivial-policy baselines on THIS holdout.
    uniform_gate_kl: float | None = None
    card_blind_gate_kl: float | None = None
    card_blind_pure_agree: float | None = None
    card_blind_anchor_kl: float | None = None
    n_nonfinite: int = 0
    # Provenance of the holdout itself.
    holdout_root_ids: list[str] = field(default_factory=list)
    holdout_provenance: dict[str, Any] = field(default_factory=dict)
    obs: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProbeGates:
    """Pass/fail thresholds for holdout probe."""

    min_pure_agree: float = DEFAULT_MIN_PURE_AGREE
    max_mean_gate_kl: float = DEFAULT_MAX_MEAN_GATE_KL
    min_n: int = DEFAULT_MIN_N
    min_pure_n: int = DEFAULT_MIN_PURE_N
    max_gate_kl_vs_card_blind: float = DEFAULT_MAX_GATE_KL_VS_CARD_BLIND
    max_mean_anchor_kl: float = DEFAULT_MAX_MEAN_ANCHOR_KL
    max_jam_freq_gap: float = DEFAULT_MAX_JAM_FREQ_GAP
    max_mean_jam_gap: float = DEFAULT_MAX_MEAN_JAM_GAP
    # Explicit opt-outs. Never default: a skipped gate is an UNVERIFIED claim.
    allow_no_pure: bool = False
    allow_no_sizing: bool = False
    # Checkpoint-level: the holdout must be disjoint from the training roots.
    require_disjoint_roots: bool = True

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


def _node_key(lab: LabelRecord) -> tuple:
    """Public decision node (everything except the hole cards)."""
    notes = lab.notes or {}
    path = notes.get("path_tokens")
    if not isinstance(path, list):
        path = [str(notes.get("path") or "")]
    return (
        lab.root_name,
        int(lab.street),
        tuple(int(c) for c in lab.board),
        int(lab.hero_seat),
        int(lab.pot_chips),
        int(lab.to_call_chips),
        int(lab.min_raise_chips),
        int(lab.max_raise_chips),
        tuple(str(t) for t in path),
    )


def _kl_rows(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Row-wise ``KL(t || p)``; NaN in ``p`` propagates (so it can fail)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(np.clip(t, 1e-12, None)) - np.log(np.clip(p, 1e-8, None))
        out = np.where(t > 0, t * ratio, 0.0).sum(axis=-1)
    return np.where(np.isnan(p).any(axis=-1), np.nan, out)


def _mean_or_none(x: np.ndarray, w: np.ndarray | None = None) -> float | None:
    if x.size == 0:
        return None
    if w is None:
        return float(np.mean(x))
    tot = float(w.sum())
    return float((x * w).sum() / tot) if tot > 0 else None


def _holdout_provenance(rows: Sequence[SupervisedRow]) -> dict[str, Any]:
    """Teacher provenance of the HOLDOUT, derived from its records (F6)."""
    block = derive_training_provenance(rows)["label_provenance"]
    block["problem"] = label_provenance_problem({"label_provenance": block})
    return block


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
    obs_stats = ObsSynthesisStats()
    labels = list(labels)
    kept: list[int] = []  # label index of every row (labels without obs drop)
    rows = labels_to_supervised_rows(labels, stats=obs_stats, kept=kept)
    if not rows:
        return ProbeReport(
            0, 0, None, None, None, labels_path=labels_path, obs=obs_stats.as_dict()
        )
    n = len(rows)

    g_tgt = np.stack([r.gate_probs for r in rows]).astype(np.float64)
    a_tgt = np.stack([r.anchor_probs for r in rows]).astype(np.float64)
    gmask = np.stack([r.gate_mask for r in rows]).astype(bool)
    g_pred = np.zeros_like(g_tgt)
    a_pred = np.zeros_like(a_tgt)
    legal = np.zeros(a_tgt.shape, dtype=bool)
    jam = np.zeros(a_tgt.shape, dtype=bool)
    with torch.no_grad():
        for lo in range(0, n, _PROBE_BATCH):
            chunk = rows[lo : lo + _PROBE_BATCH]
            obs = torch.from_numpy(np.stack([r.obs for r in chunk])).to(device_t)
            gm = torch.from_numpy(np.stack([r.gate_mask for r in chunk])).to(device_t)
            sizing = torch.from_numpy(np.stack([r.sizing for r in chunk])).to(device_t)
            gl, al, _refine, _value = model(obs, gm)
            grid = anchor_grid_torch(sizing, NLH_ANCHOR_SPEC)
            hi = lo + len(chunk)
            g_pred[lo:hi] = F.softmax(gl.double(), dim=-1).cpu().numpy()
            # Same masked softmax the trainer fits (no Categorical: it raises
            # on NaN logits, and a NaN net must FAIL the probe, not crash it).
            a_pred[lo:hi] = (
                F.softmax(al.double().masked_fill(~grid.legal, -1e9), dim=-1)
                .cpu()
                .numpy()
            )
            legal[lo:hi] = grid.legal.cpu().numpy()
            jam[lo:hi] = (
                (grid.chips == sizing[:, 1:2]) & grid.legal
            ).cpu().numpy()

    bad_rows = ~(np.isfinite(g_pred).all(axis=1) & np.isfinite(a_pred).all(axis=1))
    n_nonfinite = int(bad_rows.sum())

    # --- gates ---------------------------------------------------------------
    g_pred_m = g_pred * gmask
    g_pred_m = g_pred_m / np.clip(g_pred_m.sum(axis=1, keepdims=True), 1e-8, None)
    gate_kl = _kl_rows(g_tgt, g_pred_m)
    # NaN-safe argmax: a non-finite prediction never "agrees".
    pred_arg = np.where(bad_rows, -1, np.nan_to_num(g_pred_m, nan=-1.0).argmax(axis=1))
    tgt_arg = g_tgt.argmax(axis=1)
    hit = pred_arg == tgt_arg
    pure = g_tgt.max(axis=1) >= float(pure_thresh)
    pure_n = int(pure.sum())

    # --- trivial-policy baselines on this holdout ------------------------------
    uni = gmask / np.clip(gmask.sum(axis=1, keepdims=True), 1, None)
    groups: dict[tuple, list[int]] = {}
    for i, j in enumerate(kept):
        groups.setdefault(_node_key(labels[j]), []).append(i)
    cb_gate = np.zeros_like(g_tgt)
    cb_anchor = np.zeros_like(a_tgt)
    raise_w = g_tgt[:, 2] * gmask[:, 2]
    for idx in groups.values():
        ii = np.asarray(idx)
        cb_gate[ii] = g_tgt[ii].mean(axis=0)
        w = raise_w[ii]
        if float(w.sum()) > 0:
            cb_anchor[ii] = (a_tgt[ii] * w[:, None]).sum(axis=0) / w.sum()
    cb_hit = cb_gate.argmax(axis=1) == tgt_arg

    # --- sizing --------------------------------------------------------------
    sized = raise_w > 0
    n_raise_rows = int(sized.sum())
    anchor_kl = _kl_rows(a_tgt * legal, np.where(legal, a_pred, 1.0))
    cb_anchor_kl = _kl_rows(a_tgt * legal, np.where(legal, cb_anchor, 1.0))
    jam_t = g_tgt[:, 2] * (a_tgt * jam).sum(axis=1)
    jam_p = g_pred_m[:, 2] * (a_pred * jam).sum(axis=1)
    can_raise = gmask[:, 2]
    jam_freq_t = _mean_or_none(jam_t[can_raise])
    jam_freq_p = _mean_or_none(jam_p[can_raise])

    return ProbeReport(
        n=n,
        pure_n=pure_n,
        pure_agree=float(hit[pure].mean()) if pure_n else None,
        mean_gate_kl=_mean_or_none(gate_kl),
        mean_gate_acc=float(hit.mean()),
        labels_path=labels_path,
        n_raise_rows=n_raise_rows,
        mean_anchor_kl=_mean_or_none(anchor_kl[sized], raise_w[sized]),
        mean_jam_gap=_mean_or_none(np.abs(jam_t - jam_p)[can_raise]),
        jam_freq_gap=(
            None
            if jam_freq_t is None or jam_freq_p is None
            else abs(jam_freq_t - jam_freq_p)
        ),
        jam_freq_target=jam_freq_t,
        jam_freq_model=jam_freq_p,
        uniform_gate_kl=_mean_or_none(_kl_rows(g_tgt, uni)),
        card_blind_gate_kl=_mean_or_none(_kl_rows(g_tgt, cb_gate)),
        card_blind_pure_agree=float(cb_hit[pure].mean()) if pure_n else None,
        card_blind_anchor_kl=_mean_or_none(cb_anchor_kl[sized], raise_w[sized]),
        n_nonfinite=n_nonfinite,
        holdout_root_ids=sorted({r.prov.root_id for r in rows if r.prov.root_id}),
        holdout_provenance=_holdout_provenance(rows),
        obs=obs_stats.as_dict(),
    )


def _exceeds(value: float | None, cap: float) -> bool:
    """True unless ``value`` is a finite number ``<= cap`` (NaN / None fail)."""
    return not (value is not None and math.isfinite(value) and value <= float(cap))


def evaluate_probe_gates(
    report: ProbeReport,
    gates: ProbeGates | None = None,
) -> ProbeGateResult:
    """Apply documented thresholds; return pass/fail + reasons."""
    gates = gates or ProbeGates()
    reasons: list[str] = []

    if report.n < int(gates.min_n):
        reasons.append(f"n={report.n} < min_n={gates.min_n}")
    if report.n_nonfinite:
        reasons.append(f"{report.n_nonfinite} row(s) with non-finite predictions")

    # (review 2026-09-20 D4) ``nan > max`` is False — phrase every gate as
    # "fails unless finite and within the cap".
    if report.mean_gate_kl is None:
        reasons.append("mean_gate_kl=None (no rows)")
    elif _exceeds(report.mean_gate_kl, gates.max_mean_gate_kl):
        reasons.append(
            f"mean_gate_kl={report.mean_gate_kl:.4f} > "
            f"max={gates.max_mean_gate_kl}"
        )
    cb = report.card_blind_gate_kl
    if cb is not None and math.isfinite(cb) and cb >= _BASELINE_FLOOR:
        cap = float(gates.max_gate_kl_vs_card_blind) * cb
        if _exceeds(report.mean_gate_kl, cap):
            reasons.append(
                f"mean_gate_kl={report.mean_gate_kl} does not beat the card-blind "
                f"per-node baseline ({cb:.4f}) by the required margin "
                f"(cap {cap:.4f})"
            )

    # Pure-node gate — never skipped silently.
    if report.pure_n < max(1, int(gates.min_pure_n)):
        if not gates.allow_no_pure:
            reasons.append(
                f"pure_n={report.pure_n} < {max(1, int(gates.min_pure_n))}: "
                f"pure-agreement gate UNVERIFIED"
            )
    elif report.pure_agree is None or not (
        math.isfinite(report.pure_agree)
        and report.pure_agree >= float(gates.min_pure_agree)
    ):
        reasons.append(
            f"pure_agree={report.pure_agree} < min={gates.min_pure_agree}"
        )

    # Sizing gates.
    if report.n_raise_rows < 1 or report.mean_anchor_kl is None:
        if not gates.allow_no_sizing:
            reasons.append(
                f"n_raise_rows={report.n_raise_rows}: sizing gates UNVERIFIED"
            )
    else:
        if _exceeds(report.mean_anchor_kl, gates.max_mean_anchor_kl):
            reasons.append(
                f"mean_anchor_kl={report.mean_anchor_kl:.4f} > "
                f"max={gates.max_mean_anchor_kl}"
            )
        if _exceeds(report.jam_freq_gap, gates.max_jam_freq_gap):
            reasons.append(
                f"jam_freq_gap={report.jam_freq_gap} (target "
                f"{report.jam_freq_target} vs model {report.jam_freq_model}) > "
                f"max={gates.max_jam_freq_gap}"
            )
        if _exceeds(report.mean_jam_gap, gates.max_mean_jam_gap):
            reasons.append(
                f"mean_jam_gap={report.mean_jam_gap} > max={gates.max_mean_jam_gap}"
            )

    return ProbeGateResult(
        passed=len(reasons) == 0,
        report=report,
        gates=gates,
        reasons=reasons,
    )


def root_disjointness_problem(
    meta: dict[str, Any] | None, holdout_root_ids: Sequence[str]
) -> str | None:
    """Why the holdout is NOT a holdout for this checkpoint (None = disjoint).

    (review 2026-09-20 D4) Checkpoints recorded no root ids, so a self-fit
    (probing the training labels) passed. Unknown training roots cannot be
    verified and therefore fail.
    """
    meta = meta or {}
    train = meta.get("train_root_ids")
    if not isinstance(train, list) or not meta.get("train_roots_known"):
        return (
            "checkpoint meta records no verifiable train_root_ids — cannot "
            "prove the holdout is disjoint (retrain with the current pipeline)"
        )
    if not holdout_root_ids:
        return "holdout labels carry no root ids"
    overlap = sorted(set(map(str, train)) & set(map(str, holdout_root_ids)))
    if overlap:
        return f"holdout intersects the training roots: {overlap[:8]}"
    return None


def probe_checkpoint(
    ckpt: Path | str,
    labels_path: Path | str,
    *,
    device: str = "cpu",
    gates: ProbeGates | None = None,
) -> ProbeGateResult:
    """Score checkpoint vs holdout JSONL; return gated result."""
    gates = gates or ProbeGates()
    model, meta = load_policy_checkpoint(ckpt, device=device)
    path = Path(labels_path)
    labels = list(read_jsonl(path))
    report = probe_policy_vs_labels(
        model, labels, device=device, labels_path=str(path)
    )
    result = evaluate_probe_gates(report, gates)
    if gates.require_disjoint_roots and report.n > 0:
        why = root_disjointness_problem(meta, report.holdout_root_ids)
        if why is not None:
            result.reasons.append(why)
            result.passed = False
    return result


def stamp_probe_on_checkpoint(
    ckpt: Path | str,
    result: ProbeGateResult,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Rewrite checkpoint meta with the probe dict; re-derive the badge flag.

    Does not change weights. ``is_gto`` stays True for kind stamp; serving
    uses :func:`plo5bp.gto.policy_net.is_validated_gto_meta`, which needs the
    probe pass AND record-derived label provenance for both the training
    labels (checkpoint meta) and the holdout (``probe.report``) — review
    2026-09-20 F6. A failed provenance check is written to
    ``gto_badge_note`` so the refusal is visible.
    """
    path = Path(ckpt)
    model, meta = load_policy_checkpoint(path, device=device)
    probe_dict = result.as_dict()
    meta = dict(meta)
    meta["probe"] = probe_dict
    problems = []
    train_problem = label_provenance_problem(meta)
    if train_problem is not None:
        problems.append(f"training labels: {train_problem}")
    hold_problem = (result.report.holdout_provenance or {}).get(
        "problem", "holdout provenance missing"
    )
    if hold_problem is not None:
        problems.append(f"holdout labels: {hold_problem}")
    if not result.passed:
        problems.append("probe gates failed: " + "; ".join(result.reasons))
    meta["gto_badge_note"] = "; ".join(problems) if problems else None
    # Drop non-serializable / load-only keys before resave
    for k in ("path", "model", "actor", "policy", "is_gto_validated"):
        meta.pop(k, None)
    save_policy_checkpoint(path, model, meta=meta)
    _, stamped = load_policy_checkpoint(path, device=device)
    return stamped


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
