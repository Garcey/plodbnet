"""Clipped-surrogate PPO update with value clip and entropy bonus.

v2 additions (anchor sizing head + centralized critic):

- With a `CentralCritic`, GAE values in the buffer came from the critic;
  the critic is trained here with the clipped value loss while the
  actor's own value head ("display head", serves the UI) is trained as
  a plain regression on the same returns with a small coefficient.
- Optional KL-to-EMA-reference regularizer (`kl_anchor_coef > 0`):
  magnetic-mirror-descent-style pull toward a slow EMA copy of the
  actor for last-iterate stability. Fully zero-overhead when the flag
  is off (no EMA model is even built). The reference is NOT persisted
  in checkpoints — on (re)start it re-initializes to the current
  weights and ramps in over ~1/(1-ema) updates.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.profiler import record_function

from plo5bp.actions import GATE_RAISE
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
    gate_entropy: float = 0.0
    anchor_entropy: float = 0.0
    beta_entropy: float = 0.0
    # Per-head KL decomposition (v2 only): gate_kl + anchor_kl + beta_kl
    # == approx_kl by construction. Diagnostics for which head drives
    # drift. Zero on v1.
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
) -> torch.Tensor:
    """Mean KL(current || reference) over the full action distribution.

    Computed in f32 outside autocast (Beta KL uses lgamma/digamma).
    Masks are identical on both sides (same states), so masked
    categorical entries contribute zero. v1 models use gate + Beta;
    v2 adds the anchor categorical and per-anchor refinement Betas.
    """
    head_version = getattr(model, "head_version", 1)
    obs = mb.obs.float()
    if head_version >= 2:
        g_cur, a_cur, r_cur, _ = model(obs, mb.gate_masks)
        with torch.no_grad():
            g_ref, a_ref, r_ref, _ = ref(obs, mb.gate_masks)
        spec = getattr(model, "anchor_spec", PLO_ANCHOR_SPEC)
        grid = anchor_grid_torch(mb.sizing, spec)
        a_cur = a_cur.masked_fill(~grid.legal, -1e9)
        a_ref = a_ref.masked_fill(~grid.legal, -1e9)
        cat = torch.distributions.Categorical
        kl_gate = torch.distributions.kl_divergence(
            cat(logits=g_cur.float()), cat(logits=g_ref.float())
        )
        kl_anchor = torch.distributions.kl_divergence(
            cat(logits=a_cur.float()), cat(logits=a_ref.float())
        )
        beta = torch.distributions.Beta
        kl_refine_all = torch.distributions.kl_divergence(
            beta(r_cur[..., 0].float(), r_cur[..., 1].float()),
            beta(r_ref[..., 0].float(), r_ref[..., 1].float()),
        )  # (B, 9)
        anchor_probs = F.softmax(a_cur.float(), dim=-1)
        interior_ok = grid.refine_ok[..., 1 : spec.count - 1]
        kl_refine = (
            anchor_probs[..., 1 : spec.count - 1] * kl_refine_all * interior_ok
        ).sum(-1)
        p_raise = F.softmax(g_cur.float(), dim=-1)[..., GATE_RAISE]
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
        self.optimizer = optim.AdamW(params, lr=config.lr, fused=self._cuda)
        self._all_params = params

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
        device = batch.obs.device
        # Accumulate as device tensors; one .item() at the end avoids
        # per-minibatch CUDA syncs that serialize against compute.
        total_policy = torch.zeros((), device=device)
        total_value = torch.zeros((), device=device)
        total_display = torch.zeros((), device=device)
        total_entropy = torch.zeros((), device=device)
        total_kl = torch.zeros((), device=device)
        total_kl_anchor = torch.zeros((), device=device)
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
                            surr1 = ratio * mb.advantages
                            surr2 = torch.clamp(
                                ratio, 1.0 - cfg.clip, 1.0 + cfg.clip
                            ) * mb.advantages
                            policy_loss = -torch.min(surr1, surr2).mean()

                            # Value source: the centralized critic when
                            # present (buffer `values` came from it), else
                            # the actor's own head.
                            if self._critic_fwd is not None:
                                value = self._critic_fwd(
                                    mb.obs, opp_holes_multihot(mb.opp_holes)
                                )
                            else:
                                value = display_value
                            value_pred_clipped = mb.values + torch.clamp(
                                value - mb.values, -cfg.value_clip, cfg.value_clip
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
                            loss = (
                                policy_loss
                                + 0.5 * value_loss
                                + display_coef * display_loss
                                + eff_entropy_coef * entropy_loss
                            )

                    # KL anchor in f32 OUTSIDE autocast (lgamma/digamma).
                    kl_anchor_term = torch.zeros((), device=device)
                    if self._ref is not None:
                        with record_function("step12b2/kl_anchor"):
                            kl_anchor_term = _kl_to_reference(
                                self.model, self._ref, mb
                            )
                            loss = loss + self.kl_anchor_coef * kl_anchor_term

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
                                denom_r = raise_m.sum().clamp(min=1)
                                anchor_kl = (
                                    (mb.old_anchor_logp - anchor_lp_new) * raise_m
                                ).sum() / denom_r
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
