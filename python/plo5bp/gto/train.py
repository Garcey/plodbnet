"""Supervised PolicyNet training (Phase 1).

Loss = KL(π* || π_θ) on gates + p_raise-weighted KL on anchors
     + optional value MSE.

Works with teacher-distill rows (today) and LabelRecord rows (when obs
is attached). Checkpoints write ``kind=gto_policy_net`` for T1 serving.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from plo5bp.gto.dataset import PolicyDataset, SupervisedRow
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
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    model = build_policy_net(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    ).to(device)
    if init_ckpt is not None:
        loaded, _ = load_policy_checkpoint(init_ckpt, device=str(device))
        model.load_state_dict(loaded.state_dict())
        print(f"[gto-train] warm-start {init_ckpt}", flush=True)
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

            gate_logits, anchor_logits, _refine, value = model(obs, gm)
            gate_kl = _soft_kl(g_tgt, gate_logits, mask=gm)

            # Anchor KL: renorm teacher mass over LEGAL anchors only
            # (illegal columns get zero target — avoids 1e9 log-prob spikes).
            grid = anchor_grid_torch(sizing, NLH_ANCHOR_SPEC)
            legal = grid.legal
            p_raise = g_tgt[:, 2].clamp(0.0, 1.0)
            raise_legal = gm[:, 2].float()
            w = p_raise * raise_legal
            a_kl_row = _soft_kl(a_tgt, anchor_logits, mask=legal)
            # _soft_kl returns batch mean; recompute per-row for weighting
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
            # silence unused when all-illegal (keeps graph clean)
            _ = a_kl_row

            # Value targets are raw bb (±10s–100s); scale so MSE does not
            # drown unit-scale KL. Student value head stays in raw-bb units
            # for UI display (same convention as PPO display head).
            v_scale = 20.0
            v_loss = F.mse_loss(value / v_scale, v_tgt / v_scale)
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
