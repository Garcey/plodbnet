"""Supervised PolicyNet training (Phase 1).

Loss = KL(π* || π_θ) on gates + p_raise-weighted KL on anchors
     + optional value MSE (rows WITH a value target only).

Works with teacher-distill rows (today) and LabelRecord rows (when obs
is attached). Checkpoints write ``kind=gto_policy_net`` for T1 serving.

Review 2026-09-20:

- **D1** — teacher anchor mass on a grid-ILLEGAL anchor is an error
  (:class:`IllegalTeacherMassError`, naming the root), never silently
  renormalized away.
- **F11** — rows without a value target (``value_mask=False``; every CFR
  export) are masked out of the value loss instead of being trained toward 0.
- **D4 / F6** — the checkpoint meta records what was ACTUALLY trained on,
  derived from the rows: ``train_root_ids``, ``label_provenance`` (sources,
  teacher exploitability + cap), ``coverage`` (streets / seats),
  ``n_value_targets``. The probe and the GTO badge read these back.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from plo5bp.gto.dataset import PolicyDataset, SupervisedRow
from plo5bp.gto.labels import ILLEGAL_MASS_TOL, IllegalTeacherMassError
from plo5bp.gto.obs_rev import current_obs_rev, obs_rev_mismatch
from plo5bp.gto.policy_net import (
    build_policy_net,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from plo5bp.sizing import NLH_ANCHOR_SPEC, anchor_grid_torch


@dataclass
class TrainConfig:
    hidden_dim: int = 512
    num_layers: int = 2
    lr: float = 3e-4
    batch_size: int = 256
    epochs: int = 5
    value_coef: float = 0.1
    anchor_coef: float = 1.0
    device: str = "cpu"
    seed: int = 0
    log_every: int = 20


@dataclass
class TrainResult:
    steps: int
    final_loss: float
    final_gate_kl: float
    final_anchor_kl: float
    ckpt_path: str
    n_train: int
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "final_loss": self.final_loss,
            "final_gate_kl": self.final_gate_kl,
            "final_anchor_kl": self.final_anchor_kl,
            "ckpt_path": self.ckpt_path,
            "n_train": self.n_train,
            "seconds": self.seconds,
        }


def _soft_kl(
    target: torch.Tensor,
    logits: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean KL(target || softmax(logits)) over rows.

    When ``mask`` is given (bool, same shape as target), both target and
    logits are restricted to the True columns and re-normalized / masked
    so illegal actions never contribute.
    """
    if mask is not None:
        logits = logits.masked_fill(~mask, -1e9)
        t = target * mask.float()
    else:
        t = target
    t = t.clamp_min(0.0)
    t_sum = t.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    t = t / t_sum
    log_p = F.log_softmax(logits, dim=-1)
    # True KL = Σ t (log t - log p); H(t) cancel vs CE for optimization
    # but we report true KL so a perfect fit reads ~0.
    log_t = torch.log(t.clamp_min(1e-8))
    kl_row = (t * (log_t - log_p)).sum(dim=-1)
    return kl_row.mean()


def derive_training_provenance(
    rows: Sequence[SupervisedRow],
    *,
    init_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Checkpoint provenance DERIVED from the rows (review 2026-09-20 D4/F6).

    ``init_meta`` is the warm-start checkpoint's meta: a net that already saw
    a root keeps having seen it, so its roots / sources join the union, and a
    warm start from a checkpoint with UNKNOWN roots makes ours unknown too.
    """
    sources: dict[str, int] = {}
    roots: set[str] = set()
    n_rootless = 0
    n_unverified = 0
    max_expl: float | None = None
    caps: list[float | None] = []
    streets: set[int] = set()
    seats: set[int] = set()
    forms: dict[str, int] = {}
    inexact_streets: set[int] = set()
    n_value = 0
    for r in rows:
        pv = r.prov
        form = pv.obs_form or "unknown"
        forms[form] = forms.get(form, 0) + 1
        if form not in ("canonical", "live"):
            inexact_streets.add(int(r.street))
        sources[pv.source or "unknown"] = sources.get(pv.source or "unknown", 0) + 1
        if pv.root_id:
            roots.add(pv.root_id)
        else:
            n_rootless += 1
        e = pv.expl_bb
        ok = bool(pv.expl_verified) and e is not None and math.isfinite(e) and e >= 0
        if not ok:
            n_unverified += 1
        else:
            max_expl = e if max_expl is None else max(max_expl, e)
        caps.append(pv.teacher_cap_bb)
        streets.add(int(r.street))
        if pv.num_seats:
            seats.add(int(pv.num_seats))
        n_value += int(bool(r.value_mask))

    cap: float | None
    if caps and all(c is not None and math.isfinite(c) for c in caps):
        cap = max(float(c) for c in caps)  # the LOOSEST cap any record passed
    else:
        cap = None

    roots_known = True
    if init_meta is not None:
        prev_roots = init_meta.get("train_root_ids")
        if not init_meta.get("train_roots_known") or not isinstance(prev_roots, list):
            roots_known = False
        else:
            roots |= {str(x) for x in prev_roots}
        prev = init_meta.get("label_provenance")
        if isinstance(prev, dict):
            for src, cnt in (prev.get("sources") or {}).items():
                sources[f"warm_start:{src}"] = int(cnt)
            n_unverified += int(prev.get("n_unverified_expl") or 0)
            pe, pc = prev.get("max_label_expl_bb"), prev.get("teacher_max_expl_bb")
            if pe is not None:
                max_expl = float(pe) if max_expl is None else max(max_expl, float(pe))
            cap = None if (cap is None or pc is None) else max(cap, float(pc))
        else:
            sources["warm_start:unknown"] = 1
            cap = None
    return {
        "train_root_ids": sorted(roots),
        # Every row names its solver root AND any warm-start roots are known.
        "train_roots_known": bool(roots_known and n_rootless == 0 and roots),
        "label_provenance": {
            "derived_from_records": True,
            "sources": dict(sorted(sources.items())),
            "n_rows": len(rows),
            "n_unverified_expl": int(n_unverified),
            "max_label_expl_bb": max_expl,
            "teacher_max_expl_bb": cap,
        },
        "coverage": {
            "streets": sorted(streets),
            "seats": sorted(seats),
            # Streets whose EVERY row had an exact (engine) obs — the only
            # ones the host may claim (review 2026-09-20 F11/D16).
            "streets_exact": sorted(streets - inexact_streets),
        },
        "obs_forms": dict(sorted(forms.items())),
        "n_value_targets": int(n_value),
    }


def train_policy_net(
    rows: list[SupervisedRow],
    out_path: Path | str,
    *,
    cfg: TrainConfig | None = None,
    meta: dict[str, Any] | None = None,
    init_ckpt: Path | str | None = None,
) -> TrainResult:
    cfg = cfg or TrainConfig()
    if not rows:
        raise ValueError("no training rows")
    # (review 2026-09-20 D4) A caller that knows the holdout split passes it as
    # ``meta['planned_holdout_root_ids']``; training on one of those roots
    # would turn the later probe into a self-fit, so refuse up front.
    planned_holdout = {str(x) for x in (meta or {}).get("planned_holdout_root_ids") or []}
    leaked = sorted(planned_holdout & {r.prov.root_id for r in rows if r.prov.root_id})
    if leaked:
        raise ValueError(
            f"training rows include planned HOLDOUT roots {leaked[:8]} — the "
            f"holdout must stay unseen"
        )
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    model = build_policy_net(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    ).to(device)
    init_meta: dict[str, Any] | None = None
    if init_ckpt is not None:
        loaded, init_meta = load_policy_checkpoint(init_ckpt, device=str(device))
        model.load_state_dict(loaded.state_dict())
        print(f"[gto-train] warm-start {init_ckpt}", flush=True)
        stale = obs_rev_mismatch(init_meta)
        if stale is not None:
            print(f"[gto-train] WARNING warm-start {stale}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    ds = PolicyDataset(rows)
    loader = DataLoader(
        ds,
        batch_size=min(cfg.batch_size, len(ds)),
        shuffle=True,
        drop_last=False,
    )

    steps = 0
    last_loss = last_gkl = last_akl = 0.0
    t0 = time.perf_counter()
    model.train()

    for epoch in range(cfg.epochs):
        for batch in loader:
            obs = batch["obs"].to(device)
            gm = batch["gate_mask"].to(device)
            sizing = batch["sizing"].to(device)
            g_tgt = batch["gate_probs"].to(device)
            a_tgt = batch["anchor_probs"].to(device)
            v_tgt = batch["value_bb"].to(device)
            v_mask = batch["value_mask"].to(device).float()

            gate_logits, anchor_logits, _refine, value = model(obs, gm)
            gate_kl = _soft_kl(g_tgt, gate_logits, mask=gm)

            # Anchor KL: renorm teacher mass over LEGAL anchors only
            # (illegal columns get zero target — avoids 1e9 log-prob spikes).
            grid = anchor_grid_torch(sizing, NLH_ANCHOR_SPEC)
            legal = grid.legal
            p_raise = g_tgt[:, 2].clamp(0.0, 1.0)
            raise_legal = gm[:, 2].float()
            w = p_raise * raise_legal
            # (review 2026-09-20 D1) Masking + renormalizing used to ERASE
            # teacher mass parked on a grid-illegal anchor (the ALL-IN atom
            # whenever a fraction anchor already clamps to the stack: ~40% of
            # the jam mass at SPR 2). Refuse instead, naming the root.
            illegal_mass = (a_tgt.clamp_min(0.0) * (~legal).float()).sum(dim=-1)
            bad = (illegal_mass > ILLEGAL_MASS_TOL) & (w > 0)
            if bool(bad.any()):
                j = int(torch.nonzero(bad)[0].item())
                row = ds.rows[int(batch["idx"][j])]
                raise IllegalTeacherMassError(
                    f"root {row.prov.root_id or '?'!r}: "
                    f"{float(illegal_mass[j]):.6g} of anchor target mass on "
                    f"grid-illegal anchors (sizing={row.sizing.tolist()}, "
                    f"target={[round(float(x), 6) for x in row.anchor_probs]}) "
                    f"— re-export the labels (review 2026-09-20 D1)"
                )
            # per-row KL for the p_raise weighting
            a_logits_m = anchor_logits.masked_fill(~legal, -1e9)
            a_t = a_tgt * legal.float()
            a_t = a_t / a_t.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            log_a = F.log_softmax(a_logits_m, dim=-1)
            log_at = torch.log(a_t.clamp_min(1e-8))
            a_kl_rows = (a_t * (log_at - log_a)).sum(dim=-1)
            if float(w.sum()) > 0:
                anchor_kl = (a_kl_rows * w).sum() / w.sum().clamp_min(1e-8)
            else:
                anchor_kl = torch.zeros((), device=device)

            # Value targets are raw bb (±10s–100s); scale so MSE does not
            # drown unit-scale KL. Student value head stays in raw-bb units
            # for UI display (same convention as PPO display head).
            # (review 2026-09-20 F11) Rows without a target are MASKED OUT —
            # ``value_bb=None`` used to be trained toward 0.
            v_scale = 20.0
            v_err = ((value - v_tgt) / v_scale) ** 2
            v_loss = (v_err * v_mask).sum() / v_mask.sum().clamp_min(1.0)
            loss = gate_kl + cfg.anchor_coef * anchor_kl + cfg.value_coef * v_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            steps += 1
            last_loss = float(loss.item())
            last_gkl = float(gate_kl.item())
            last_akl = float(anchor_kl.item())
            if steps % cfg.log_every == 0:
                print(
                    f"[gto-train] ep={epoch} step={steps} "
                    f"loss={last_loss:.4f} gKL={last_gkl:.4f} "
                    f"aKL={last_akl:.4f} v={float(v_loss.item()):.4f}"
                )

    out_path = Path(out_path)
    save_meta = {
        "source": (meta or {}).get("source", "supervised"),
        "n_train": len(rows),
        "epochs": cfg.epochs,
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "final_loss": last_loss,
        "final_gate_kl": last_gkl,
        "final_anchor_kl": last_akl,
    }
    if meta:
        save_meta.update(meta)
    # Derived LAST so a caller's meta can never assert provenance
    # (review 2026-09-20 D4/F6). ``source`` stays the caller's label, but the
    # GTO badge reads ``label_provenance``.
    derived = derive_training_provenance(rows, init_meta=init_meta)
    save_meta.update(derived)
    # The obs semantics these weights were fit to (plo5bp.gto.obs_rev).
    save_meta["obs_rev"] = current_obs_rev()
    if not (meta or {}).get("source"):
        # No label from the caller: name the source after the records.
        srcs = [s for s in derived["label_provenance"]["sources"] if s != "unknown"]
        save_meta["source"] = srcs[0] if len(srcs) == 1 else (
            "mixed:" + "+".join(srcs) if srcs else "supervised"
        )
    if init_ckpt is not None:
        save_meta["warm_start"] = str(init_ckpt)
    save_policy_checkpoint(out_path, model, meta=save_meta)
    elapsed = time.perf_counter() - t0
    print(f"[gto-train] saved {out_path} ({len(rows)} rows, {elapsed:.1f}s)")
    return TrainResult(
        steps=steps,
        final_loss=last_loss,
        final_gate_kl=last_gkl,
        final_anchor_kl=last_akl,
        ckpt_path=str(out_path),
        n_train=len(rows),
        seconds=elapsed,
    )
