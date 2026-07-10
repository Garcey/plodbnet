"""Clipped-surrogate PPO update with value clip and entropy bonus.

v2 additions (anchor sizing head + centralized critic):

- With a `CentralCritic`, GAE values in the buffer came from the critic;
  the critic is trained here with the clipped value loss while the
  actor's own value head ("display head", serves the UI) is trained as
  a plain regression on the same returns with a small coefficient.
- Optional KL-to-EMA-reference regularizer (`kl_anchor_coef > 0`):
  magnetic-mirror-descent-style pull toward a slow EMA copy of the
  actor for last-iterate stability. Fully zero-overhead when the flag
  is off (no EMA model is even built). The reference IS persistable:
  train.py saves `ref_state_dict()` as ckpt["model_ema"] and restores
  it on warm-start, so the magnet's memory survives relaunches; absent
  that key it re-initializes to the loaded weights and ramps in over
  ~1/(1-ema) updates. The same EMA weights double as a smoother
  serving actor (promote model_ema instead of the last iterate).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.profiler import record_function

from plo5bp.actions import GATE_FOLD, GATE_RAISE
from plo5bp.config import TrainingConfig
from plo5bp.network import ActorCritic, CentralCritic, opp_holes_multihot
from plo5bp.rollout import Batch, iter_minibatches
from plo5bp.sizing import PLO_ANCHOR_SPEC, anchor_grid_torch
import numpy as np


@dataclass
class PPOStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    display_loss: float = 0.0
    kl_anchor: float = 0.0
    # Auxiliary Q(s,a) regression loss (v5 critic dueling head); 0 when off.
    q_loss: float = 0.0
    gate_entropy: float = 0.0
    anchor_entropy: float = 0.0
    beta_entropy: float = 0.0
    # Per-head KL decomposition (v2 only): gate_kl + anchor_kl + beta_kl
    # == approx_kl by construction, ALL per-batch means (C4, 2026-07-10:
    # anchor_kl was a per-RAISE-ROW mean, which contaminated the derived
    # beta_kl with weight (1/B - 1/n_raise) — anchor drift read as a
    # strongly NEGATIVE klB and a ~3x-overstated klA at a 33% raise
    # fraction). Diagnostics for which head drives drift. Zero on v1.
    # NOTE: klA on the log line reads ~3x SMALLER than in pre-2026-07-10
    # logs (vFour/vFive/early-vSix eras) — same drift, new normalization.
    gate_kl: float = 0.0
    anchor_kl: float = 0.0
    beta_kl: float = 0.0
    # KL guard: minibatch index (0-based, across epochs) of the trip
    # (soft early-stop OR hard rollback), -1 = never tripped. `kl_stop`
    # is the offending value (excluded from the approx_kl average).
    kl_stopped_at: int = -1
    kl_stop: float = 0.0
    # True ONLY on a hard-threshold (kl_hard) trip — the whole update
    # was rolled back. A soft early-stop leaves this False (it keeps
    # the minibatches already applied).
    rolled_back: bool = False


def _kl_to_reference(
    model: ActorCritic,
    ref: ActorCritic,
    mb: Batch,
    cur: "tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None" = None,
) -> torch.Tensor:
    """Mean KL(current || reference) over the full action distribution.

    The reference forward is f32 (Beta KL uses lgamma/digamma). The
    CURRENT-model outputs are REUSED from evaluate()'s forward via `cur`
    (V5_DESIGN.md magnet memory fix): the KL term backprops through that
    same graph, so the magnet adds only the reference forward (no_grad,
    transient) instead of a second full actor forward-with-grad
    (~+18-20 GiB, which OOM'd the pod). `cur` = evaluate()'s
    (gate_logits, anchor_head_out, refine); it is autocast bf16, upcast
    to f32 below for the KL math. Falls back to a fresh forward when
    `cur` is None (v1 / callers without the reuse).

    Masks are identical on both sides (same states), so masked
    categorical entries contribute zero. v1 models use gate + Beta;
    v2 adds the anchor categorical and per-anchor refinement Betas.
    """
    head_version = getattr(model, "head_version", 1)
    obs = mb.obs.float()
    if head_version >= 2:
        if cur is not None:
            g_cur, a_cur, r_cur = cur
        else:
            g_cur, a_cur, r_cur, _ = model(obs, mb.gate_masks)
        with torch.no_grad():
            g_ref, a_ref, r_ref, _ = ref(obs, mb.gate_masks)
        spec = getattr(model, "anchor_spec", PLO_ANCHOR_SPEC)
        grid = anchor_grid_torch(mb.sizing, spec)
        cat = torch.distributions.Categorical
        kl_gate = torch.distributions.kl_divergence(
            cat(logits=g_cur.float()), cat(logits=g_ref.float())
        )
        # Head-agnostic anchor KL: `_anchor_dist` maps each head's raw
        # second output (v2 flat logits / v4 (mu, s) / v5 mixture
        # params) to the legal-anchor Categorical. The old path
        # masked_fill'ed the raw output as if it were logits — a shape
        # crash on v4's (B, 2) size params and semantically wrong even
        # shape-fixed (V5_DESIGN.md B1). Computed manually over probs
        # (v4/v5 hard-zero illegal anchors) with an explicit clamp_min so
        # the 0-prob terms are 0·(log ε − log ε) = 0, matching torch's
        # own eps-clamp but without depending on it.
        p_cur = model._anchor_dist(a_cur.float(), grid).probs
        with torch.no_grad():
            p_ref = ref._anchor_dist(a_ref.float(), grid).probs
        kl_anchor = (
            p_cur
            * (p_cur.clamp_min(1e-12).log() - p_ref.clamp_min(1e-12).log())
        ).sum(-1)
        beta = torch.distributions.Beta
        kl_refine_all = torch.distributions.kl_divergence(
            beta(r_cur[..., 0].float(), r_cur[..., 1].float()),
            beta(r_ref[..., 0].float(), r_ref[..., 1].float()),
        )  # (B, 9)
        interior_ok = grid.refine_ok[..., 1 : spec.count - 1]
        kl_refine = (
            p_cur[..., 1 : spec.count - 1] * kl_refine_all * interior_ok
        ).sum(-1)
        # p_raise DETACHED: this term is MINIMIZED, so with the gate
        # weight in the graph it pays the gate to shrink p_raise
        # whenever the sizing KL is high — a fold bias. Mirror of the
        # 2026-06-11 entropy-bonus detach (V5_DESIGN.md B2).
        p_raise = F.softmax(g_cur.float(), dim=-1)[..., GATE_RAISE].detach()
        return (kl_gate + p_raise * (kl_anchor + kl_refine)).mean()

    g_cur, rp_cur, _ = model(obs, mb.gate_masks)
    with torch.no_grad():
        g_ref, rp_ref, _ = ref(obs, mb.gate_masks)
    cat = torch.distributions.Categorical
    kl_gate = torch.distributions.kl_divergence(
        cat(logits=g_cur.float()), cat(logits=g_ref.float())
    )
    beta = torch.distributions.Beta
    kl_beta = torch.distributions.kl_divergence(
        beta(rp_cur[..., 0].float(), rp_cur[..., 1].float()),
        beta(rp_ref[..., 0].float(), rp_ref[..., 1].float()),
    )
    p_raise = F.softmax(g_cur.float(), dim=-1)[..., GATE_RAISE]
    return (kl_gate + p_raise * kl_beta).mean()


def _adaptive_grad_clip_(params, clip: float, eps: float = 1e-3) -> None:
    """Stateless per-tensor adaptive gradient clipping (NFNet AGC): clip each
    param's grad norm to ``clip * max(||param||, eps)``. In-place, no running
    state (nothing for the kl_hard rollback to corrupt). A per-tensor complement
    to the per-group split ``clip_grad_norm_``: scale-adapts to each tensor so a
    wide torso matrix and a small head are clipped proportionally."""
    with torch.no_grad():
        for p in params:
            g = p.grad
            if g is None:
                continue
            g_norm = g.detach().norm()
            max_norm = clip * p.detach().norm().clamp_min(eps)
            if float(g_norm) > float(max_norm):
                g.mul_(max_norm / g_norm.clamp_min(1e-12))


class PPOTrainer:
    def __init__(
        self,
        model: ActorCritic,
        config: TrainingConfig,
        critic: CentralCritic | None = None,
    ):
        self.model = model
        self.critic = critic
        self.config = config
        self.head_version = getattr(model, "head_version", 1)
        self._cuda = config.device == "cuda" and torch.cuda.is_available()
        # Actor and critic params kept separate so their gradients can be
        # clipped independently (see the grad-clip in update()): they share one
        # optimizer but the critic's grads are chip-scale and the actor's are
        # unit-scale, so a single global clip lets a critic-loss spike throttle
        # the actor's (gate) gradient.
        self._actor_params = list(model.parameters())
        self._critic_params = (
            list(critic.parameters()) if critic is not None else []
        )
        params = self._actor_params + self._critic_params
        self.optimizer = optim.AdamW(
            params,
            lr=config.lr,
            betas=(0.9, float(getattr(config, "adam_b2", 0.999))),
            fused=self._cuda,
        )
        self._all_params = params
        # Stateless adaptive gradient clipping (NFNet AGC), per-tensor; 0 = off.
        # No running state → nothing for the kl_hard rollback to restore.
        self._agc_clip = float(getattr(config, "agc_clip", 0.0))
        # AGC EXEMPTS the dueling adv_head (NFNet practice excludes final
        # layers): the head starts zero-init, so clip×‖W‖ rate-limits exactly
        # the head that must chase a moving trunk to stay calibrated
        # (2026-07-09 Q-head audit: fold-column error GREW 150→200 while the
        # head norm crawled). The split clip_grad_norm_ below still bounds it.
        self._agc_params = params
        if critic is not None and getattr(critic, "q_actions", 0) > 0:
            _adv_ids = {id(p) for p in critic.adv_head.parameters()}
            self._agc_params = [p for p in params if id(p) not in _adv_ids]
        # v6 probability-dependent gate clip (Over-mixing §6). Widens the clip
        # band for rare gate actions, narrows it near 50/50; keyed on the gate's
        # OLD prob (old_gate_logp) so the sizing menu isn't over-loosened. Off =
        # the flat cfg.clip band. See TrainingConfig.clip_prob_dependent.
        self._clip_prob_dependent = bool(
            getattr(config, "clip_prob_dependent", False)
        )
        self._clip_room_ext = float(getattr(config, "clip_room_ext", 0.10))
        self._clip_room_mid = float(getattr(config, "clip_room_mid", 0.05))
        self._clip_prob_floor = float(getattr(config, "clip_prob_floor", 1e-3))
        # Gradient checkpointing: identical math, recompute-for-memory. Runtime
        # flag on the trainable model + critic only (rollout is no-grad).
        _gc = bool(getattr(config, "grad_checkpoint", False))
        model._grad_checkpoint = _gc
        if critic is not None:
            critic._grad_checkpoint = _gc

        # Weight-decay-to-init companion for torso LayerNorm (V6_RESEARCH.md #4).
        # L2 penalty pulling the TRUNK weight matrices toward their run-start
        # values (regenerative regularization) — bounds weight-norm growth so
        # the effective LR doesn't decay, and counters the generalization hit of
        # norm-solo. Snapshots init once; 0 = off (no snapshot, no term). Trunk
        # only (name contains "torso", 2D weights) so the heads stay free.
        self._l2_init_coef = float(getattr(config, "l2_init_coef", 0.0))
        self._l2_init_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        if self._l2_init_coef > 0.0:
            named = list(model.named_parameters())
            if critic is not None:
                named += list(critic.named_parameters())
            self._l2_init_pairs = [
                (p, p.detach().clone())
                for name, p in named
                if "torso" in name and p.dim() >= 2
            ]

        # KL-to-EMA reference: zero overhead unless the flag is on.
        self.kl_anchor_coef = float(getattr(config, "kl_anchor_coef", 0.0))
        self.kl_anchor_ema = float(getattr(config, "kl_anchor_ema", 0.999))
        # Soft KL guard (early-stop, KEEP applied minibatches); 0 = off.
        # See TrainingConfig.target_kl.
        self.target_kl = float(getattr(config, "target_kl", 0.0))
        # Hard KL guard (full rollback): restore params + optimizer state
        # from the top of update(), discarding the whole update. 0 = off.
        # See TrainingConfig.kl_hard.
        self.kl_hard = float(getattr(config, "kl_hard", 0.0))
        # Sizing-entropy scale (v2): multiplies the anchor+beta (sizing-head)
        # entropy bonus relative to the gate. 1.0 = unchanged. >1 resists the
        # anchor/beta over-sharpening that drives the v2 saturation collapse,
        # without loosening the gate. Live-tunable via anneal_control.
        self.sizing_entropy_scale = float(
            getattr(config, "sizing_entropy_scale", 1.0)
        )
        # Auxiliary Q(s, a) regression on the critic's dueling head (v5
        # stems). Only wired when the critic actually HAS the head; the
        # coef gates training (0 = head stays zero-init).
        self._q_aux_coef = float(getattr(config, "q_aux_coef", 0.0))
        # Dense fold-column supervision (see TrainingConfig.q_fold_sup_coef):
        # fold's forward return is exactly 0, so q[..., GATE_FOLD] gets a
        # perfect-label MSE on every fold-LEGAL row, weighted into q_loss.
        self._q_fold_sup = float(getattr(config, "q_fold_sup_coef", 0.0))
        self._critic_qv = (
            critic.q_values
            if critic is not None and getattr(critic, "q_actions", 0) > 0
            else None
        )
        # Distributional (HL-Gauss) value path: one torso pass → (V, logits, Q).
        # Bound only when the critic has a categorical value head; else None and
        # the scalar clipped-MSE path runs unchanged.
        self._distributional = (
            critic is not None and getattr(critic, "value_bins", 0) > 0
        )
        self._critic_train = critic.train_outputs if self._distributional else None
        self._ref: ActorCritic | None = None
        if self.kl_anchor_coef > 0.0:
            self._ref = copy.deepcopy(model).eval()
            for p in self._ref.parameters():
                p.requires_grad_(False)

        # Compile evaluate() (and the critic) on CUDA when Triton is
        # available (Linux); falls back to eager on Windows. dynamic=True
        # tolerates the last-minibatch shape variance in iter_minibatches.
        self._evaluate = model.evaluate
        self._critic_fwd = critic.forward if critic is not None else None
        if self._cuda:
            try:
                import triton  # noqa: F401
                self._evaluate = torch.compile(model.evaluate, dynamic=True)
                if critic is not None:
                    self._critic_fwd = torch.compile(critic.forward, dynamic=True)
            except ImportError:
                pass

    def _ema_update_ref(self) -> None:
        if self._ref is None:
            return
        with torch.no_grad():
            for p_ref, p in zip(self._ref.parameters(), self.model.parameters()):
                p_ref.lerp_(p.detach(), 1.0 - self.kl_anchor_ema)

    def ref_state_dict(self) -> dict | None:
        """EMA-reference weights for checkpointing (None when the magnet
        is off). Persisting them keeps the magnet's memory across warm
        restarts; they also serve as the smoother `model_ema` actor."""
        return self._ref.state_dict() if self._ref is not None else None

    def load_ref_state_dict(self, state_dict: dict | None) -> None:
        """Restore a persisted EMA reference (no-op when the magnet is
        off or the checkpoint predates model_ema)."""
        if self._ref is not None and state_dict:
            self._ref.load_state_dict(state_dict)

    def _gate_clip_bounds(
        self, old_gate_logp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample PPO ratio band from the gate's OLD probability (v6
        probability-dependent clip, Over-mixing §6). Target absolute
        probability-movement room is a symmetric U in p = π_old(gate):

            R(p) = room_ext − (room_ext − room_mid)·4·p·(1−p),

        and the ratio band is [1 − R/p, 1 + R/p]. p is floored by
        `clip_prob_floor` so the max ratio stays ~1 + room_ext/floor. A rare
        gate (p→0) gets ~room_ext of upward room (fast recovery) and may fall to
        0 (the lower bound clamps at 0); a 50/50 gate gets the tight room_mid
        band. Applied to the joint action ratio, but keyed on the gate prob so
        the parametric sizing menu is not over-loosened (check/fold have no
        sizing, so for them the joint ratio IS the gate ratio)."""
        p = old_gate_logp.exp().clamp(
            self._clip_prob_floor, 1.0 - self._clip_prob_floor
        )
        room = self._clip_room_ext - (
            self._clip_room_ext - self._clip_room_mid
        ) * 4.0 * p * (1.0 - p)
        r_over_p = room / p
        hi = 1.0 + r_over_p
        lo = (1.0 - r_over_p).clamp_min(0.0)
        return lo, hi

    def _q_index(self, mb: Batch, q_all: torch.Tensor) -> torch.Tensor:
        """Column of the TAKEN action in the dueling head's layout. Pooled
        3-column heads (q_pooled) index by gate directly (GATE_RAISE == 2);
        per-anchor heads use 2 + anchor for raises. Keyed on the Q tensor's
        WIDTH so 13-column checkpoints keep training unchanged."""
        if q_all.shape[-1] == 3:
            return mb.gate_actions
        return torch.where(
            mb.gate_actions == GATE_RAISE,
            2 + mb.anchor_actions,
            mb.gate_actions,
        )

    def _q_fold_sup_term(
        self, mb: Batch, q_all: torch.Tensor, q_loss: torch.Tensor
    ) -> torch.Tensor:
        """Dense fold-column supervision (TrainingConfig.q_fold_sup_coef):
        fold's forward return is exactly 0 (per-step-cost rewards, sunk
        chips excluded), so q[..., GATE_FOLD] takes a perfect-label MSE on
        every fold-LEGAL row — not just the ones where fold was taken.
        The bool mask is lifted to f32 so the reduction stays fp32 under
        autocast (562k-row sums are garbage in bf16)."""
        if self._q_fold_sup <= 0.0:
            return q_loss
        fold_ok = mb.gate_masks[..., GATE_FOLD].float()
        fold_mse = (q_all[..., GATE_FOLD].pow(2) * fold_ok).sum() / (
            fold_ok.sum().clamp_min(1.0)
        )
        return q_loss + self._q_fold_sup * fold_mse

    def update(
        self,
        batch: Batch,
        rng: np.random.Generator,
        entropy_coef: float | None = None,
    ) -> PPOStats:
        cfg = self.config
        eff_entropy_coef = (
            float(cfg.entropy_coef) if entropy_coef is None else float(entropy_coef)
        )
        display_coef = float(getattr(cfg, "display_value_coef", 0.125))
        value_coef = float(getattr(cfg, "value_loss_coef", 0.5))
        device = batch.obs.device
        # Accumulate as device tensors; one .item() at the end avoids
        # per-minibatch CUDA syncs that serialize against compute.
        total_policy = torch.zeros((), device=device)
        total_value = torch.zeros((), device=device)
        total_display = torch.zeros((), device=device)
        total_entropy = torch.zeros((), device=device)
        total_kl = torch.zeros((), device=device)
        total_kl_anchor = torch.zeros((), device=device)
        total_q = torch.zeros((), device=device)
        total_gate_h = torch.zeros((), device=device)
        total_anchor_h = torch.zeros((), device=device)
        total_beta_h = torch.zeros((), device=device)
        total_gate_kl = torch.zeros((), device=device)
        total_anchor_kl = torch.zeros((), device=device)
        total_beta_kl = torch.zeros((), device=device)
        count = 0
        kl_stopped_at = -1
        kl_stop_val = 0.0
        rolled_back_flag = False
        # Hard-rollback snapshot: param data + Adam moments, cloned
        # on-device (~3x param memory, copied once per update). Restored
        # verbatim only on a HARD (kl_hard) trip — a soft early-stop
        # keeps its applied minibatches and never touches this.
        snapshot: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]] | None = None
        if self.kl_hard > 0.0:
            snapshot = []
            for p in self._all_params:
                st = self.optimizer.state.get(p, {})
                snapshot.append((
                    p.detach().clone(),
                    st["exp_avg"].clone() if "exp_avg" in st else None,
                    st["exp_avg_sq"].clone() if "exp_avg_sq" in st else None,
                ))
        with record_function("step12/inner_loop"):
            for _ in range(cfg.ppo_epochs):
                if kl_stopped_at >= 0:
                    break
                for mb in iter_minibatches(batch, cfg.batch_size, rng):
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=self._cuda,
                    ):
                        with record_function("step12a/evaluate"):
                            if self.head_version >= 2:
                                (
                                    log_prob, entropy, display_value,
                                    gate_h, anchor_h, beta_h,
                                    gate_lp_new, anchor_lp_new,
                                    cur_gate_logits, cur_anchor_out, cur_refine,
                                ) = self._evaluate(
                                    mb.obs,
                                    mb.gate_masks,
                                    mb.sizing,
                                    mb.gate_actions,
                                    mb.anchor_actions,
                                    mb.refine_u,
                                )
                            else:
                                log_prob, entropy, display_value = self._evaluate(
                                    mb.obs,
                                    mb.gate_masks,
                                    mb.sizing[..., :2],
                                    mb.gate_actions,
                                    mb.raise_chips,
                                )
                                gate_h = anchor_h = beta_h = entropy

                        with record_function("step12b/loss"):
                            ratio = torch.exp(log_prob - mb.log_probs)
                            if self._clip_prob_dependent:
                                clip_lo, clip_hi = self._gate_clip_bounds(
                                    mb.old_gate_logp
                                )
                            else:
                                clip_lo = 1.0 - cfg.clip
                                clip_hi = 1.0 + cfg.clip
                            surr1 = ratio * mb.advantages
                            surr2 = (
                                torch.clamp(ratio, clip_lo, clip_hi)
                                * mb.advantages
                            )
                            policy_loss = -torch.min(surr1, surr2).mean()

                            # Value source: the centralized critic when
                            # present (buffer `values` came from it), else
                            # the actor's own head.
                            q_loss = torch.zeros((), device=device)
                            if self._distributional:
                                # Distributional / HL-Gauss value head: one torso
                                # pass gives V (=symexp(E[bins]), scalar), the
                                # categorical logits, and the scalar dueling Q.
                                # value loss = HL-Gauss cross-entropy — no MSE
                                # clip (the categorical support IS the bound).
                                value, value_logits, q_all = self._critic_train(
                                    mb.obs, opp_holes_multihot(mb.opp_holes)
                                )
                                if self._q_aux_coef > 0.0 and q_all is not None:
                                    q_taken = q_all.gather(
                                        -1, self._q_index(mb, q_all)[..., None]
                                    ).squeeze(-1)
                                    q_loss = (q_taken - mb.returns).pow(2).mean()
                                    q_loss = self._q_fold_sup_term(mb, q_all, q_loss)
                                value_loss = self.critic.hlgauss_value_loss(
                                    value_logits, mb.returns
                                )
                            else:
                                if (
                                    self._q_aux_coef > 0.0
                                    and self._critic_qv is not None
                                ):
                                    # One critic forward yields V (identical to
                                    # forward()) AND the dueling Q row; the
                                    # taken action's Q regresses to the same
                                    # returns. Index: 0 Fold, 1 CheckCall,
                                    # 2+anchor Raise (or 2 = pooled Raise).
                                    value, q_all = self._critic_qv(
                                        mb.obs, opp_holes_multihot(mb.opp_holes)
                                    )
                                    q_taken = q_all.gather(
                                        -1, self._q_index(mb, q_all)[..., None]
                                    ).squeeze(-1)
                                    q_loss = (q_taken - mb.returns).pow(2).mean()
                                    q_loss = self._q_fold_sup_term(mb, q_all, q_loss)
                                elif self._critic_fwd is not None:
                                    value = self._critic_fwd(
                                        mb.obs, opp_holes_multihot(mb.opp_holes)
                                    )
                                else:
                                    value = display_value
                                value_pred_clipped = mb.values + torch.clamp(
                                    value - mb.values,
                                    -cfg.value_clip,
                                    cfg.value_clip,
                                )
                                v1 = (value - mb.returns).pow(2)
                                v2 = (value_pred_clipped - mb.returns).pow(2)
                                value_loss = 0.5 * torch.max(v1, v2).mean()

                            # Display head: plain regression (no clipping —
                            # buffer values belong to the critic), small
                            # coefficient so it stays subordinate.
                            if self._critic_fwd is not None:
                                display_loss = (
                                    (display_value - mb.returns).pow(2).mean()
                                )
                            else:
                                display_loss = torch.zeros_like(value_loss)

                            # Sizing-entropy scale (v2): `entropy` is
                            # gate_h + p_raise.detach()*(anchor_h+beta_h), so
                            # (entropy - gate_h) is exactly the p_raise-weighted
                            # sizing-head entropy. Rescaling it boosts the
                            # anchor/beta entropy bonus while the gate-head
                            # gradient cancels between the two gate_h terms
                            # (gate weight stays 1, sizing weight = scale).
                            if (
                                self.head_version >= 2
                                and self.sizing_entropy_scale != 1.0
                            ):
                                entropy_for_loss = gate_h + (
                                    self.sizing_entropy_scale * (entropy - gate_h)
                                )
                            else:
                                entropy_for_loss = entropy
                            entropy_loss = -entropy_for_loss.mean()
                            # Per-row coefs (mix-configs per-tier entropy,
                            # V5_DESIGN.md B5) override the scalar coef —
                            # each transition is paid its own tier's rate.
                            if mb.ent_coef_rows is not None:
                                entropy_bonus = -(
                                    mb.ent_coef_rows * entropy_for_loss
                                ).mean()
                            else:
                                entropy_bonus = eff_entropy_coef * entropy_loss
                            loss = (
                                policy_loss
                                + value_coef * value_loss
                                + display_coef * display_loss
                                + entropy_bonus
                                + self._q_aux_coef * q_loss
                            )

                    # KL anchor: reference forward in f32 outside autocast
                    # (lgamma/digamma); the current-model outputs are
                    # REUSED from evaluate above (no second forward-with-grad).
                    kl_anchor_term = torch.zeros((), device=device)
                    if self._ref is not None:
                        with record_function("step12b2/kl_anchor"):
                            cur_raw = (
                                (cur_gate_logits, cur_anchor_out, cur_refine)
                                if self.head_version >= 2 else None
                            )
                            kl_anchor_term = _kl_to_reference(
                                self.model, self._ref, mb, cur=cur_raw
                            )
                            loss = loss + self.kl_anchor_coef * kl_anchor_term

                    # Weight-decay-to-init (torso LayerNorm companion): pull the
                    # trunk weights toward their run-start values. No-op unless
                    # l2_init_coef > 0. f32, outside autocast.
                    if self._l2_init_pairs:
                        l2_init = torch.zeros((), device=device)
                        for p, p0 in self._l2_init_pairs:
                            l2_init = l2_init + ((p - p0) ** 2).sum()
                        loss = loss + self._l2_init_coef * l2_init

                    with record_function("step12e/kl"):
                        with torch.no_grad():
                            kl = (mb.log_probs - log_prob).mean()
                            # Per-head KL decomposition (v2 only): gate
                            # and anchor (raise rows) computed directly,
                            # beta derived by the joint identity. Pure
                            # diagnostics — never feeds the loss or guard.
                            if self.head_version >= 2:
                                gate_kl = (mb.old_gate_logp - gate_lp_new).mean()
                                raise_m = (mb.gate_actions == GATE_RAISE)
                                # C4: batch-mean normalization (sum/B, matching
                                # gate_kl and kl) so klG+klA+klB is a true
                                # additive decomposition — sum/n_raise made the
                                # derived klB absorb the anchor term with
                                # negative weight. See PPOStats for the
                                # log-scale note vs pre-2026-07-10 runs.
                                anchor_kl = (
                                    (mb.old_anchor_logp - anchor_lp_new) * raise_m
                                ).sum() / raise_m.numel()
                                beta_kl = kl - gate_kl - anchor_kl

                    # KL guard: checked BEFORE the optimizer step so the
                    # offending minibatch is never applied. Two thresholds:
                    #   |kl| > kl_hard  -> HARD: full rollback (revert the
                    #                      whole update) — catastrophe only.
                    #   |kl| > target_kl -> SOFT: early-stop, KEEP the
                    #                      minibatches already applied.
                    # The .item() forces a per-minibatch sync; negligible
                    # against multi-minute updates, and load-bearing for
                    # aborting in time.
                    if self.kl_hard > 0.0 or self.target_kl > 0.0:
                        kl_now = float(kl.item())
                        abs_kl = abs(kl_now)
                        if self.kl_hard > 0.0 and abs_kl > self.kl_hard:
                            kl_stopped_at = count
                            kl_stop_val = kl_now
                            rolled_back_flag = True
                            if snapshot is not None:
                                with torch.no_grad():
                                    for p, (pd, m1, m2) in zip(
                                        self._all_params, snapshot
                                    ):
                                        p.data.copy_(pd)
                                        st = self.optimizer.state.get(p)
                                        if st is None or not st:
                                            continue
                                        if m1 is None:
                                            # No pre-update state existed
                                            # (first-ever steps): drop the
                                            # polluted moments entirely.
                                            self.optimizer.state[p] = {}
                                            continue
                                        st["exp_avg"].copy_(m1)
                                        if m2 is not None:
                                            st["exp_avg_sq"].copy_(m2)
                                        if "step" in st:
                                            # Rewind Adam's bias-correction
                                            # counter by the steps applied
                                            # this update (tensor or int).
                                            st["step"] -= count
                            break
                        if self.target_kl > 0.0 and abs_kl > self.target_kl:
                            # SOFT early-stop: keep applied minibatches,
                            # do NOT touch the snapshot.
                            kl_stopped_at = count
                            kl_stop_val = kl_now
                            break

                    with record_function("step12c/backward"):
                        self.optimizer.zero_grad()
                        loss.backward()
                    with record_function("step12d/optimizer_step"):
                        # Per-tensor adaptive clip (opt-in) BEFORE the per-group
                        # split clip: tames heavy-tailed spikes tensor-by-tensor.
                        if self._agc_clip > 0.0:
                            _adaptive_grad_clip_(self._agc_params, self._agc_clip)
                        # Clip actor and critic grads SEPARATELY — a single
                        # global clip over both lets a chip-scale critic-loss
                        # spike inflate the shared grad-norm and throttle the
                        # actor's (gate) gradient on that same update.
                        nn.utils.clip_grad_norm_(self._actor_params, 0.5)
                        if self._critic_params:
                            nn.utils.clip_grad_norm_(self._critic_params, 0.5)
                        self.optimizer.step()
                    total_policy += policy_loss.detach().float()
                    total_value += value_loss.detach().float()
                    total_display += display_loss.detach().float()
                    total_entropy += -entropy_loss.detach().float()
                    total_kl += kl.float()
                    total_kl_anchor += kl_anchor_term.detach().float()
                    total_q += q_loss.detach().float()
                    total_gate_h += gate_h.detach().float().mean()
                    total_anchor_h += anchor_h.detach().float().mean()
                    total_beta_h += beta_h.detach().float().mean()
                    if self.head_version >= 2:
                        total_gate_kl += gate_kl.float()
                        total_anchor_kl += anchor_kl.float()
                        total_beta_kl += beta_kl.float()
                    count += 1
        self._ema_update_ref()
        denom = max(count, 1)
        with record_function("step13/stats_sync"):
            return PPOStats(
                policy_loss=float(total_policy.item()) / denom,
                value_loss=float(total_value.item()) / denom,
                entropy=float(total_entropy.item()) / denom,
                approx_kl=float(total_kl.item()) / denom,
                display_loss=float(total_display.item()) / denom,
                kl_anchor=float(total_kl_anchor.item()) / denom,
                q_loss=float(total_q.item()) / denom,
                gate_entropy=float(total_gate_h.item()) / denom,
                anchor_entropy=float(total_anchor_h.item()) / denom,
                beta_entropy=float(total_beta_h.item()) / denom,
                gate_kl=float(total_gate_kl.item()) / denom,
                anchor_kl=float(total_anchor_kl.item()) / denom,
                beta_kl=float(total_beta_kl.item()) / denom,
                kl_stopped_at=kl_stopped_at,
                kl_stop=kl_stop_val,
                rolled_back=rolled_back_flag,
            )
