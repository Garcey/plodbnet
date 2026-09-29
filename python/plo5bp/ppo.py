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
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.profiler import record_function

from plo5bp.actions import GATE_FOLD, GATE_RAISE
from plo5bp.config import TrainingConfig
from plo5bp.network import ActorCritic, CentralCritic, opp_holes_multihot
from plo5bp import rollout as _rollout
from plo5bp.rollout import (  # noqa: F401 (iter_minibatches: re-export)
    Batch,
    HostBatchLoader,
    _minibatch_bounds,
    gather_minibatch,
    iter_minibatch_indices,
    iter_minibatches,
)
from plo5bp.sizing import PLO_ANCHOR_SPEC, anchor_grid_torch
import numpy as np


def _merge_chunk_terms(acc: dict | None, tc: dict, w: float) -> dict:
    """Accumulate one micro-batch chunk's terms (detached): per-row means
    weighted by the chunk's share `w` (q_loss arrives weighted already), the
    canary sums and the (already weighted) loss summed as they are."""
    out = {} if acc is None else acc
    for k, v in tc.items():
        if v is None:
            out.setdefault(k, None)
            continue
        v = v.detach().float()
        if k in ("policy_loss", "value_loss", "display_loss", "entropy_loss",
                 "kl_anchor_term", "kl", "gate_kl", "anchor_kl", "beta_kl",
                 "gate_h", "anchor_h", "beta_h", "entropy_true",
                 "clip_frac", "kl_k3", "ratio_dev"):
            v = w * v
        out[k] = v if out.get(k) is None else out[k] + v
    return out


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
    # Fold-column canary (audit 2026-07-11): mean Q[FOLD] over fold-LEGAL
    # rows this update. Ground truth is EXACTLY 0 (per-step-cost rewards,
    # sunk chips excluded), so sustained drift = a systematic Q-surface
    # offset. 0.0 when the run has no dueling head.
    q_fold_err: float = 0.0
    # Terminal-boundary canary (V7_DESIGN.md WS1.3): mean(return − Q(s,a))
    # over NON-FOLD terminal rows — the exact δ at trajectory boundaries,
    # where estimator bias has no next-state term to cancel against (the
    # mechanism behind the July fold subsidy). Persistent positive = hand-
    # ending actions (showdown calls, steals) collect fake advantage;
    # negative = they are taxed. 0.0 when the head is off or the batch
    # carries no terminal flags (serial collectors).
    q_term_err: float = 0.0
    gate_entropy: float = 0.0
    anchor_entropy: float = 0.0
    beta_entropy: float = 0.0
    # Per-head KL decomposition (v2 only): gate_kl + anchor_kl + beta_kl
    # == approx_kl by construction, ALL per-batch means (C4, 2026-07-10:
    # anchor_kl was a per-RAISE-ROW mean, which contaminated the derived
    # beta_kl with weight (1/B - 1/n_raise) — anchor drift read as a
    # strongly NEGATIVE klB and a ~3x-overstated klA at a 33% raise
    # fraction). Diagnostics for which head drives drift.
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
    # True ONLY when the whole update was rolled back: a hard-threshold
    # (kl_hard) trip, or a NON-FINITE kl with a rollback snapshot available
    # (kl_hard > 0). A soft early-stop leaves this False (it keeps the
    # minibatches already applied) — as does a non-finite trip with kl_hard
    # off, which can only refuse the bad step (`kl_stop` is then nan/inf).
    rolled_back: bool = False
    # ---- per-update health numbers (2026-09-28: ML-003/-010/-012/-040) ----
    # `entropy` above is the POLICY's entropy, gate + p_raise*(anchor + beta):
    # comparable across runs. `entropy_bonus` is what the bonus pays for,
    # gate + sizing_entropy_scale*(sizing part) -- the number `H=` meant before
    # 2026-09-28 (equal to `entropy` at scale 1.0).
    entropy_bonus: float = 0.0
    # Share of rows whose importance ratio left the clip band (flat or
    # probability-dependent), and the always-positive k3 KL estimator
    # mean((r - 1) - log r) -- lower variance than `approx_kl`.
    clip_frac: float = 0.0
    kl_k3: float = 0.0
    # The FIRST minibatch's KL and mean |ratio - 1|, measured before any step
    # of this update: pure rollout-vs-PPO numerical mismatch (bf16 autocast,
    # f16 stored observations). nan when no PPO minibatch ran.
    kl0: float = float("nan")
    ratio_dev0: float = float("nan")
    # Pre-clip gradient norms per optimizer step (the split clip bounds each
    # group at 0.5): mean / max over the steps, and the share of steps the
    # clip scaled down. `grad_norm_display` = the actor's display value head
    # alone (it shares the actor's clip group).
    grad_norm_actor: float = 0.0
    grad_norm_actor_max: float = 0.0
    grad_clip_actor: float = 0.0
    grad_norm_critic: float = 0.0
    grad_norm_critic_max: float = 0.0
    grad_clip_critic: float = 0.0
    grad_norm_display: float = 0.0
    # A PPO step was refused because its gradient norm was non-finite (a hard
    # trip like a non-finite loss: `kl_stopped_at` names the minibatch).
    nonfinite_grad: bool = False
    # Critic-only passes (critic_extra_epochs / actor-frozen warm-up): steps
    # applied, steps REFUSED as non-finite (loss or gradient -- never applied,
    # so the weights stay finite), and their mean value / Q loss.
    critic_steps: int = 0
    critic_skipped: int = 0
    critic_value_loss: float = 0.0
    critic_q_loss: float = 0.0
    # Wall seconds this update spent shuffling minibatch indices (ML-045).
    shuffle_s: float = 0.0


# PPOStats fields that measure wall time, not the update: two identical
# updates differ there. `comparable_stats` leaves them out.
STATS_TIMING_FIELDS = frozenset({"shuffle_s"})


def comparable_stats(stats: "PPOStats") -> dict:
    """The update's numbers without the timing fields -- what two runs of the
    same update must reproduce exactly (NaN compared as equal to NaN)."""
    return {
        k: ("nan" if isinstance(v, float) and v != v else v)
        for k, v in vars(stats).items() if k not in STATS_TIMING_FIELDS
    }


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
    `cur` is None (callers without the reuse).

    Masks are identical on both sides (same states), so masked
    categorical entries contribute zero: the gate categorical, the anchor
    categorical and the per-anchor refinement Betas (v2+ heads; the retired
    v1 Beta raise head is refused by PPOTrainer).
    """
    obs = mb.obs.float()
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


def _adaptive_grad_clip_(params, clip: float, eps: float = 1e-3) -> None:
    """Stateless per-tensor adaptive gradient clipping (NFNet AGC): clip each
    param's grad norm to ``clip * max(||param||, eps)``. In-place, no running
    state (nothing for the kl_hard rollback to corrupt). A per-tensor complement
    to the per-group split ``clip_grad_norm_``: scale-adapts to each tensor so a
    wide torso matrix and a small head are clipped proportionally.

    Sync-free (2026-09-28, ML-041): the old `float(g_norm) > float(max_norm)`
    stalled the GPU once per tensor per step (~4,000 stalls per vSix6 update).
    The comparison now stays on the device and every grad is multiplied by
    `where(g_norm > max_norm, max_norm / g_norm, 1)` -- the same norms and the
    same factor where it clips, and an exact x*1.0 elsewhere (NaN stays NaN,
    as before), so the result is bit-identical."""
    with torch.no_grad():
        for p in params:
            g = p.grad
            if g is None:
                continue
            g_norm = g.detach().norm()
            max_norm = clip * p.detach().norm().clamp_min(eps)
            scale = torch.where(
                g_norm > max_norm,
                max_norm / g_norm.clamp_min(1e-12),
                torch.ones_like(g_norm),
            )
            g.mul_(scale)


def critic_ok_for_extra(rolled_back: bool, critic_params: list) -> bool:
    """Critic-only passes run only when there is a critic and the update was
    not rolled back (a hard KL trip restores the pre-update weights)."""
    return bool(critic_params) and not rolled_back


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
        self.head_version = int(getattr(model, "head_version", 1))
        if self.head_version < 2:
            # (2026-09-28, ML-062) The v1 Beta raise head is retired: nothing
            # trains it (train.py refuses v1 checkpoints) and its PPO branches
            # are gone. The UI can still SERVE a v1 checkpoint.
            raise ValueError(
                "PPOTrainer trains the v2+ anchor heads (ActorCriticV2/V4/V5); "
                f"got head_version {self.head_version} ({type(model).__name__}) "
                "-- the v1 Beta raise head is retired"
            )
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
        # AdamW decoupled weight decay, applied to EVERY tensor (biases,
        # LayerNorm gains, the zero-init heads included): each step shrinks
        # weights by lr*weight_decay. TrainingConfig.weight_decay = 0.01 is
        # AdamW's own default -- the value every stem to date trained with,
        # implicitly until 2026-09-28 (ML-051; now explicit, stamped with the
        # config, and a flag). Exempting tensors or changing the value is a
        # production behavior change for the owner at a stem boundary.
        # critic_lr > 0: the critic gets its own param group at that rate (the
        # parameter ORDER -- actor then critic -- is unchanged, so optimizer
        # sidecars restore either way). 0 = one group at config.lr (exact).
        critic_lr = float(config.critic_lr or 0.0)
        self._critic_lr_ratio = (
            critic_lr / float(config.lr)
            if critic_lr > 0.0 and self._critic_params
            else None
        )
        if self._critic_lr_ratio is not None:
            param_spec = [
                {"params": self._actor_params},
                {"params": self._critic_params, "lr": critic_lr},
            ]
        else:
            param_spec = params
        self.optimizer = optim.AdamW(
            param_spec,
            lr=config.lr,
            betas=(0.9, float(config.adam_b2)),
            weight_decay=float(config.weight_decay),
            fused=self._cuda,
        )
        self._all_params = params
        # The actor's display value head (serves the UI's EV) shares the
        # actor's clip group: its own grad norm is logged beside the group's
        # (PPOStats.grad_norm_display) to show whether it drives the clip.
        _vh = getattr(model, "value_head", None)
        self._display_params = list(_vh.parameters()) if _vh is not None else []
        # Stateless adaptive gradient clipping (NFNet AGC), per-tensor; 0 = off.
        # No running state → nothing for the kl_hard rollback to restore.
        self._agc_clip = float(config.agc_clip)
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
            config.clip_prob_dependent
        )
        self._clip_room_ext = float(config.clip_room_ext)
        self._clip_room_mid = float(config.clip_room_mid)
        self._clip_prob_floor = float(config.clip_prob_floor)
        # Gradient checkpointing: identical math, recompute-for-memory. Runtime
        # flag on the trainable model + critic only (rollout is no-grad).
        _gc = bool(config.grad_checkpoint)
        model._grad_checkpoint = _gc
        if critic is not None:
            critic._grad_checkpoint = _gc

        # Weight-decay-to-init companion for torso LayerNorm (V6_RESEARCH.md #4).
        # L2 penalty pulling the TRUNK weight matrices toward their run-start
        # values (regenerative regularization) — bounds weight-norm growth so
        # the effective LR doesn't decay, and counters the generalization hit of
        # norm-solo. Snapshots init once; 0 = off (no snapshot, no term). Trunk
        # only (name contains "torso", 2D weights) so the heads stay free.
        self._l2_init_coef = float(config.l2_init_coef)
        self._l2_init_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        # Qualified names of the pairs ("actor."/"critic." + param name), so
        # the references can be persisted and re-attached by NAME (review
        # 2026-09-20 A14 — see optimizer_sidecar_state / load_l2_init_refs).
        self._l2_init_names: list[str] = []
        if self._l2_init_coef > 0.0:
            named = [("actor." + n, p) for n, p in model.named_parameters()]
            if critic is not None:
                named += [
                    ("critic." + n, p) for n, p in critic.named_parameters()
                ]
            for name, p in named:
                if "torso" in name and p.dim() >= 2:
                    self._l2_init_pairs.append((p, p.detach().clone()))
                    self._l2_init_names.append(name)

        # KL-to-EMA reference: zero overhead unless the flag is on.
        self.kl_anchor_coef = float(config.kl_anchor_coef)
        self.kl_anchor_ema = float(config.kl_anchor_ema)
        # Soft KL guard (early-stop, KEEP applied minibatches); 0 = off.
        # See TrainingConfig.target_kl.
        self.target_kl = float(config.target_kl)
        # Hard KL guard (full rollback): restore params + optimizer state
        # from the top of update(), discarding the whole update. 0 = off.
        # See TrainingConfig.kl_hard.
        self.kl_hard = float(config.kl_hard)
        # Sizing-entropy scale (v2): multiplies the anchor+beta (sizing-head)
        # entropy bonus relative to the gate. 1.0 = unchanged. >1 resists the
        # anchor/beta over-sharpening that drives the v2 saturation collapse,
        # without loosening the gate. Live-tunable via anneal_control.
        self.sizing_entropy_scale = float(
            config.sizing_entropy_scale
        )
        # Auxiliary Q(s, a) regression on the critic's dueling head (v5
        # stems). Only wired when the critic actually HAS the head; the
        # coef gates training (0 = head stays zero-init).
        self._q_aux_coef = float(config.q_aux_coef)
        self._critic_q_norm = bool(config.critic_q_norm)
        # Dense fold-column supervision (see TrainingConfig.q_fold_sup_coef):
        # fold's forward return is exactly 0, so q[..., GATE_FOLD] gets a
        # perfect-label MSE on every fold-LEGAL row, weighted into q_loss.
        self._q_fold_sup = float(config.q_fold_sup_coef)
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
                    # TrainingConfig.compile_critic_train (ML-013, default
                    # off): the critic's TRAINING forwards too -- every v6
                    # run trains it through these, eager until now.
                    if config.compile_critic_train:
                        if self._critic_train is not None:
                            self._critic_train = torch.compile(
                                critic.train_outputs, dynamic=True
                            )
                        if self._critic_qv is not None:
                            self._critic_qv = torch.compile(
                                critic.q_values, dynamic=True
                            )
            except ImportError:
                pass

    def ppo_batch_size(self, n_rows: int) -> int:
        """The PPO minibatch size for a batch of `n_rows` rows: the config's
        `batch_size`, or -- with `num_minibatches` > 0 (ML-044) --
        ceil(n_rows / num_minibatches) of the rows actually collected, so an
        epoch is exactly that many steps whatever the drain overshoot."""
        k = int(self.config.num_minibatches or 0)
        if k > 0:
            return max(1, -(-int(n_rows) // k))
        return int(self.config.batch_size)

    # ---- the live-tunable surface (train.py's control file) ---------------
    # key in the control file -> attribute read per minibatch/update.
    _LIVE_ATTRS = {
        "target_kl": "target_kl",
        "kl_hard": "kl_hard",
        "sizing_entropy_scale": "sizing_entropy_scale",
        "clip_room_mid": "_clip_room_mid",
        "clip_room_ext": "_clip_room_ext",
        "q_fold_sup_coef": "_q_fold_sup",
    }
    LIVE_KEYS = tuple(_LIVE_ATTRS)

    def live_value(self, key: str) -> float:
        """Current value of a live-tunable knob (one of LIVE_KEYS)."""
        return getattr(self, self._LIVE_ATTRS[key])

    def apply_live_control(self, **vals: float) -> dict[str, tuple[float, float]]:
        """Set live-tunable knobs (LIVE_KEYS) for the next minibatches; returns
        {key: (old, new)} for those that changed. Unknown keys raise. The clip
        rooms only matter with clip_prob_dependent; q_fold_sup_coef only with a
        Q head."""
        changed: dict[str, tuple[float, float]] = {}
        for key, new in vals.items():
            if key not in self._LIVE_ATTRS:
                raise KeyError(f"not live-tunable: {key!r} (valid: {self.LIVE_KEYS})")
            old = self.live_value(key)
            setattr(self, self._LIVE_ATTRS[key], float(new))
            if old != new:
                changed[key] = (old, float(new))
        return changed

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

    # ---- optimizer sidecar (review 2026-09-20 A3 + A14) -------------------
    # Adam moments and the l2-init reference tensors are RUN STATE that was
    # never checkpointed: every guardian relaunch restarted AdamW from zero
    # moments (first step ~lr*sign(g): KL ~ +1.08 measured on a 2048x4 actor
    # at lr 1.5e-4 vs ~0.04 warm — one big step, then KLSTOP), and re-took
    # the "init" snapshot at the relaunch point. train.py persists both in
    # ONE rolling `<stem>.optim.pt` next to the numbered checkpoints (they
    # are ~3x the parameter bytes — too heavy to ride in every checkpoint).

    def set_lr(self, lr: float) -> None:
        """Set the base (actor) learning rate for the next update; a separate
        critic group (critic_lr) follows at its configured ratio, so the
        warmup ramp and live anneal_control lr edits scale both."""
        for i, group in enumerate(self.optimizer.param_groups):
            ratio = self._critic_lr_ratio if (i == 1 and self._critic_lr_ratio) else 1.0
            group["lr"] = lr * ratio

    def optimizer_sidecar_state(self) -> dict:
        """CPU copy of everything the sidecar persists. `param_shapes`
        fingerprints the optimizer's parameter ORDER (actor then critic) so
        a sidecar from a different architecture is refused, not mis-mapped."""
        opt_sd = self.optimizer.state_dict()
        return {
            "optimizer_state": {
                idx: {
                    k: (v.detach().cpu() if torch.is_tensor(v) else v)
                    for k, v in st.items()
                }
                for idx, st in opt_sd["state"].items()
            },
            "param_shapes": [tuple(p.shape) for p in self._all_params],
            "l2_init": {
                name: p0.detach().cpu()
                for name, (_p, p0) in zip(
                    self._l2_init_names, self._l2_init_pairs
                )
            },
        }

    def load_optimizer_moments(
        self, sidecar: dict, allow_actor_only: bool = False
    ) -> "tuple[bool, str]":
        """Restore Adam moments + step counters from a sidecar dict. Returns
        (restored, reason). Only the per-parameter STATE is loaded — this
        run's param_groups (lr, betas, fused, ...) stay as constructed, so a
        changed --lr / --adam-b2 is honored. All-or-nothing: any shape
        mismatch leaves the optimizer cold and says why -- except that
        `allow_actor_only` (a NEW critic: train.py --critic-fresh /
        --critic-init) restores the actor's moments from a sidecar whose
        actor part matches and starts only the critic cold."""
        state = sidecar.get("optimizer_state")
        shapes = sidecar.get("param_shapes")
        if not isinstance(state, dict) or shapes is None:
            return False, "sidecar has no optimizer state"
        want = [tuple(p.shape) for p in self._all_params]
        have = [tuple(s) for s in shapes]
        keep = None
        if have != want:
            # A NEW critic (train.py --critic-fresh / --critic-init) against a
            # sidecar of the same actor: restore the ACTOR's moments (the first
            # len(actor) entries -- the order is actor then critic) and start
            # only the critic cold. Anything else is still refused.
            n_act = len(self._actor_params)
            if not (allow_actor_only and have[:n_act] == want[:n_act]):
                return False, "parameter shapes differ from this run's model/critic"
            keep = n_act
        for idx, st in state.items():
            if keep is not None and int(idx) >= keep:
                continue
            m = st.get("exp_avg")
            if m is not None and tuple(m.shape) != want[int(idx)]:
                return False, f"moment shape mismatch at param {idx}"
        if keep is not None:
            state = {k: v for k, v in state.items() if int(k) < keep}
        self.optimizer.load_state_dict({
            "state": state,
            "param_groups": self.optimizer.state_dict()["param_groups"],
        })
        return True, (f"{len(state)} tensors" if keep is None
                      else f"ACTOR ONLY, {len(state)} tensors; the new critic starts cold")

    def load_l2_init_refs(
        self, sidecar: dict, allow_actor_only: bool = False
    ) -> "tuple[bool, str]":
        """Re-attach the ORIGINAL decay-to-init reference tensors (A14) so the
        L2 pull keeps pointing at the stem's true init across relaunches, not
        at wherever the last relaunch happened to load. No-op (False) when
        l2_init is off. All-or-nothing on name/shape agreement, except that
        `allow_actor_only` (a new critic) restores the references the sidecar
        still has and keeps the new critic's own init for the rest."""
        if not self._l2_init_pairs:
            return False, "l2_init_coef is 0 (no references in use)"
        refs = sidecar.get("l2_init") or {}
        if set(refs) != set(self._l2_init_names):
            if allow_actor_only:
                # A new critic: restore the references the sidecar still has
                # (the actor's), the new critic's anchor at its own init.
                done = 0
                with torch.no_grad():
                    for name, (_p, p0) in zip(self._l2_init_names, self._l2_init_pairs):
                        r = refs.get(name)
                        if r is not None and tuple(r.shape) == tuple(p0.shape):
                            p0.copy_(r.to(p0.device, p0.dtype))
                            done += 1
                return True, f"{done} of {len(self._l2_init_names)} tensors (new critic's own init for the rest)"
            return False, "sidecar l2_init names differ from this run's trunk"
        for name, (_p, p0) in zip(self._l2_init_names, self._l2_init_pairs):
            if tuple(refs[name].shape) != tuple(p0.shape):
                return False, f"l2_init shape mismatch at {name}"
        with torch.no_grad():
            for name, (_p, p0) in zip(
                self._l2_init_names, self._l2_init_pairs
            ):
                p0.copy_(refs[name].to(p0.device, p0.dtype))
        return True, f"{len(refs)} tensors"

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

    def _q_norm(self, mb, q_var: "torch.Tensor | None" = None):
        """TrainingConfig.critic_q_norm (2026-09-26): the Q regression's MSE
        terms divided by the minibatch's return variance (+1, detached), so a
        chip-scale Q loss (~1e3 bb^2) no longer drowns the value head's
        cross-entropy (~3 nats) in the shared critic torso -- which kept a
        FRESH critic from learning its value distribution at all (vSix5 r1crit:
        v 3.6 -> 4.0 while q ran away to 1e4). 1.0 when off (the old loss).

        `q_var` = the WHOLE minibatch's return variance, computed once by a
        micro-batched caller under critic_q_norm_minibatch (ML-011); None =
        the variance of the rows in `mb` (a whole minibatch, or -- the legacy
        micro-batched behavior -- one chunk)."""
        if not self._critic_q_norm:
            return 1.0
        if q_var is not None:
            return 1.0 / (q_var + 1.0)
        return 1.0 / (mb.returns.float().var().detach() + 1.0)

    def _minibatch_q_var(self, batch: Batch, host_loader, sel) -> "torch.Tensor | None":
        """The minibatch-wide return variance for q-norm'd micro-batch chunks
        (critic_q_norm_minibatch), else None (each chunk uses its own)."""
        if not (self._critic_q_norm and self.config.critic_q_norm_minibatch):
            return None
        rets = (
            host_loader.field_rows("returns", sel)
            if host_loader is not None
            else batch.returns[sel]
        )
        return rets.float().var().detach()

    def _q_fold_sup_term(
        self,
        mb: Batch,
        q_all: torch.Tensor,
        q_loss: torch.Tensor,
        fold_denom: torch.Tensor | None = None,
        weight: float | None = None,
        q_var: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """Dense fold-column supervision (TrainingConfig.q_fold_sup_coef):
        fold's forward return is exactly 0 (per-step-cost rewards, sunk
        chips excluded), so q[..., GATE_FOLD] takes a perfect-label MSE on
        every fold-LEGAL row — not just the ones where fold was taken.
        The bool mask is lifted to f32 so the reduction stays fp32 under
        autocast (562k-row sums are garbage in bf16)."""
        if weight is not None:
            # Micro-batch chunk: the MSE part is a per-row mean (weighted by
            # the chunk's share), the fold part is normalized by the
            # MINIBATCH's fold-legal count so the chunks sum to the original.
            q_loss = weight * q_loss
        if self._q_fold_sup <= 0.0:
            return q_loss
        fold_ok = mb.gate_masks[..., GATE_FOLD].float()
        fold_mse = self._q_norm(mb, q_var) * (q_all[..., GATE_FOLD].pow(2) * fold_ok).sum() / (
            fold_ok.sum().clamp_min(1.0) if fold_denom is None else fold_denom
        )
        return q_loss + self._q_fold_sup * fold_mse

    def _minibatch_terms(
        self,
        mb: Batch,
        eff_entropy_coef: float,
        display_coef: float,
        value_coef: float,
        device: torch.device,
        weight: float | None = None,
        fold_denom: torch.Tensor | None = None,
        q_var: "torch.Tensor | None" = None,
    ) -> dict:
        """One minibatch's (or, micro-batched, one chunk's) PPO loss terms --
        the body of `update`'s minibatch loop up to the KL guard, moved here
        verbatim. `weight` None = a whole minibatch (the exact original
        expressions); else the chunk's share of the minibatch rows, with
        `fold_denom` the minibatch-wide fold-legal row count."""
        cfg = self.config
        qf = qf_n = qt = qt_n = None
        gate_kl = anchor_kl = beta_kl = None
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self._cuda,
        ):
            with record_function("step12a/evaluate"):
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
                # Health numbers (no_grad, detached: the loss is untouched):
                # the share of rows outside the clip band, the k3 KL
                # estimator (e^d - 1 - d >= 0, d = log r) and mean |r - 1|.
                with torch.no_grad():
                    _d = (log_prob - mb.log_probs).detach().float()
                    _r = ratio.detach().float()
                    clip_frac = ((_r < clip_lo) | (_r > clip_hi)).float().mean()
                    kl_k3 = (torch.expm1(_d) - _d).mean()
                    ratio_dev = (_r - 1.0).abs().mean()

                # Value source: the centralized critic when
                # present (buffer `values` came from it), else
                # the actor's own head.
                q_loss = torch.zeros((), device=device)
                q_all = None
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
                        q_loss = (q_taken - mb.returns).pow(2).mean() * self._q_norm(mb, q_var)
                        q_loss = self._q_fold_sup_term(mb, q_all, q_loss, fold_denom, weight, q_var)
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
                        q_loss = (q_taken - mb.returns).pow(2).mean() * self._q_norm(mb, q_var)
                        q_loss = self._q_fold_sup_term(mb, q_all, q_loss, fold_denom, weight, q_var)
                    elif self._critic_fwd is not None:
                        value = self._critic_fwd(
                            mb.obs, opp_holes_multihot(mb.opp_holes)
                        )
                    else:
                        value = display_value
                    v1 = (value - mb.returns).pow(2)
                    if cfg.value_clip > 0.0:
                        value_pred_clipped = mb.values + torch.clamp(
                            value - mb.values,
                            -cfg.value_clip,
                            cfg.value_clip,
                        )
                        v2 = (value_pred_clipped - mb.returns).pow(2)
                        value_loss = 0.5 * torch.max(v1, v2).mean()
                    else:
                        # value_clip <= 0 DISABLES clipping — plain
                        # MSE (review 2026-09-20 A20). A literal 0
                        # radius pinned value_pred_clipped to the
                        # rollout values, so max(v1, v2) only ever
                        # passed gradient where the critic was
                        # already WORSE than its old self: "0 = off"
                        # froze the critic instead of unclipping it.
                        value_loss = 0.5 * v1.mean()

                # Fold-column canary (audit 2026-07-11): the
                # per-update mean of Q[FOLD] over fold-LEGAL
                # rows, whose ground truth is exactly 0. f32
                # accumulation (bf16 sums of ~500k rows are
                # garbage); no_grad — diagnostics only.
                if q_all is not None:
                    with torch.no_grad():
                        fold_ok_c = (
                            mb.gate_masks[..., GATE_FOLD].float()
                        )
                        qf = (
                            q_all[..., GATE_FOLD].float() * fold_ok_c
                        ).sum()
                        qf_n = fold_ok_c.sum()
                        # Terminal-boundary canary (V7_DESIGN.md
                        # WS1.3): qT = mean(return − Q(s,a)) over
                        # NON-FOLD terminal rows. At a terminal
                        # row the stored return IS the raw reward
                        # (no future term), so this is the exact
                        # boundary residual δ_terminal — the term
                        # that paid the July fold subsidy.
                        # Positive = terminal actions subsidized,
                        # negative = taxed. Fold rows are the qF
                        # canary's job (truth 0), so they are
                        # excluded here.
                        if mb.is_terminal is not None:
                            term_ok = (
                                mb.is_terminal
                                & (mb.gate_actions != GATE_FOLD)
                            ).float()
                            q_sel = q_all.gather(
                                -1,
                                self._q_index(mb, q_all)[..., None],
                            ).squeeze(-1)
                            qt = (
                                (mb.returns - q_sel).float()
                                * term_ok
                            ).sum()
                            qt_n = term_ok.sum()

                # Display head: plain regression (no clipping —
                # buffer values belong to the critic), small
                # coefficient so it stays subordinate.
                if self._critic_fwd is not None:
                    display_loss = (
                        (display_value - mb.returns).pow(2).mean()
                    )
                else:
                    display_loss = torch.zeros_like(value_loss)

                # Sizing-entropy scale: `entropy` is
                # gate_h + p_raise.detach()*(anchor_h+beta_h), so
                # (entropy - gate_h) is exactly the p_raise-weighted
                # sizing-head entropy. Rescaling it boosts the
                # anchor/beta entropy bonus while the gate-head
                # gradient cancels between the two gate_h terms
                # (gate weight stays 1, sizing weight = scale).
                if self.sizing_entropy_scale != 1.0:
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
                if weight is None:
                    loss = (
                        policy_loss
                        + value_coef * value_loss
                        + display_coef * display_loss
                        + entropy_bonus
                        + self._q_aux_coef * q_loss
                    )
                else:
                    # Micro-batch chunk: per-row means weighted by the chunk's
                    # share of the minibatch rows (q_loss arrives weighted, its
                    # fold term normalized minibatch-wide).
                    loss = (
                        weight * (
                            policy_loss
                            + value_coef * value_loss
                            + display_coef * display_loss
                            + entropy_bonus
                        )
                        + self._q_aux_coef * q_loss
                    )

        # KL anchor: reference forward in f32 outside autocast
        # (lgamma/digamma); the current-model outputs are
        # REUSED from evaluate above (no second forward-with-grad).
        kl_anchor_term = torch.zeros((), device=device)
        if self._ref is not None:
            with record_function("step12b2/kl_anchor"):
                cur_raw = (cur_gate_logits, cur_anchor_out, cur_refine)
                kl_anchor_term = _kl_to_reference(
                    self.model, self._ref, mb, cur=cur_raw
                )
                loss = loss + (
                    self.kl_anchor_coef if weight is None
                    else weight * self.kl_anchor_coef
                ) * kl_anchor_term

        # Weight-decay-to-init (torso LayerNorm companion): pull the
        # trunk weights toward their run-start values. No-op unless
        # l2_init_coef > 0. f32, outside autocast.
        if self._l2_init_pairs:
            l2_init = torch.zeros((), device=device)
            for p, p0 in self._l2_init_pairs:
                l2_init = l2_init + ((p - p0) ** 2).sum()
            loss = loss + (
                self._l2_init_coef if weight is None
                else weight * self._l2_init_coef
            ) * l2_init

        with record_function("step12e/kl"):
            with torch.no_grad():
                kl = (mb.log_probs - log_prob).mean()
                # Per-head KL decomposition: gate and anchor
                # (raise rows) computed directly, beta derived by
                # the joint identity. Pure diagnostics — never
                # feeds the loss or guard.
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
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "display_loss": display_loss,
            "entropy_loss": entropy_loss,
            "q_loss": q_loss,
            "kl_anchor_term": kl_anchor_term,
            "kl": kl,
            "gate_kl": gate_kl,
            "anchor_kl": anchor_kl,
            "beta_kl": beta_kl,
            "gate_h": gate_h.detach().float().mean(),
            "anchor_h": anchor_h.detach().float().mean(),
            "beta_h": beta_h.detach().float().mean(),
            "qf": qf,
            "qf_n": qf_n,
            "qt": qt,
            "qt_n": qt_n,
            # The policy's own entropy (entropy_loss carries the bonus-
            # weighted one) + the health numbers above.
            "entropy_true": entropy.detach().float().mean(),
            "clip_frac": clip_frac,
            "kl_k3": kl_k3,
            "ratio_dev": ratio_dev,
        }

    def _critic_only_terms(
        self,
        mb: Batch,
        value_coef: float,
        weight: float,
        fold_denom: torch.Tensor | None,
        q_var: "torch.Tensor | None" = None,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """(loss, value_loss, q_loss) of the critic ALONE on one minibatch (or
        micro-batch chunk, `weight` = its share of the minibatch rows): the same
        value and Q terms `_minibatch_terms` adds, without the actor. Used by
        the critic-only passes (TrainingConfig.critic_extra_epochs) and the
        actor-frozen warm-up (`actor_frozen`)."""
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self._cuda):
            opp = opp_holes_multihot(mb.opp_holes)
            q_all = None
            if self._distributional:
                _v, value_logits, q_all = self._critic_train(mb.obs, opp)
                value_loss = self.critic.hlgauss_value_loss(value_logits, mb.returns)
            else:
                if self._q_aux_coef > 0.0 and self._critic_qv is not None:
                    value, q_all = self._critic_qv(mb.obs, opp)
                else:
                    value = self._critic_fwd(mb.obs, opp)
                value_loss = 0.5 * (value.float() - mb.returns).pow(2).mean()
            q_loss = torch.zeros((), device=value_loss.device)
            if self._q_aux_coef > 0.0 and q_all is not None:
                q_taken = q_all.gather(-1, self._q_index(mb, q_all)[..., None]).squeeze(-1)
                q_loss = (q_taken.float() - mb.returns).pow(2).mean() * self._q_norm(mb, q_var)
                q_loss = self._q_fold_sup_term(mb, q_all, q_loss, fold_denom, weight, q_var)
        loss = weight * value_coef * value_loss + self._q_aux_coef * q_loss
        return loss, value_loss, q_loss

    def _critic_only_epochs(
        self,
        batch: Batch,
        rng: np.random.Generator,
        host_loader,
        micro: int,
        value_coef: float,
        epochs: int,
        device: torch.device,
    ) -> "tuple[torch.Tensor, torch.Tensor, int, int, list[float]]":
        """`epochs` passes over the rollout training ONLY the critic: every
        actor parameter keeps grad None, so AdamW skips it (no step, no decay,
        moments untouched). Same minibatches, micro-batching, AGC and split
        grad clip as the PPO loop.

        NaN/Inf guard (2026-09-28, ML-001): a step whose loss or critic
        gradient norm is non-finite is REFUSED (its gradient dropped, the
        optimizer not stepped), so one bad minibatch can no longer turn the
        critic's weights into NaN -- which the next numbered checkpoint then
        saved, and every later update rolled back (a livelock resumed from
        poisoned files). Costs one host sync per critic step. A finite run is
        bit-identical. Returns (sum value loss, sum q loss, steps applied,
        steps refused, pre-clip critic grad norms of the applied steps)."""
        tot_v = torch.zeros((), device=device)
        tot_q = torch.zeros((), device=device)
        steps = 0
        skipped = 0
        norms: list[float] = []
        n_rows = int(batch.obs.shape[0])
        k = int(self.config.critic_minibatches or 0)
        bs = max(1, -(-n_rows // k)) if k > 0 else self.ppo_batch_size(n_rows)
        for _ in range(int(epochs)):
            for sel in iter_minibatch_indices(batch, bs, rng):
                n_mb = int(sel.shape[0])
                self.optimizer.zero_grad(set_to_none=True)
                fold_denom = None
                if self._q_fold_sup > 0.0:
                    gm_sel = (
                        host_loader.gate_mask_rows(sel)
                        if host_loader is not None
                        else batch.gate_masks[sel]
                    )
                    fold_denom = gm_sel[..., GATE_FOLD].float().sum().clamp_min(1.0)
                chunk = micro if 0 < micro < n_mb else n_mb
                q_var = (
                    self._minibatch_q_var(batch, host_loader, sel)
                    if chunk < n_mb else None
                )
                step_v = torch.zeros((), device=device)
                step_q = torch.zeros((), device=device)
                step_loss = torch.zeros((), device=device)
                for lo in range(0, n_mb, chunk):
                    sub = sel[lo : lo + chunk]
                    mb = (
                        host_loader.gather(sub)
                        if host_loader is not None
                        else gather_minibatch(batch, sub)
                    )
                    w = float(int(sub.shape[0])) / float(n_mb)
                    loss, vl, ql = self._critic_only_terms(
                        mb, value_coef, w, fold_denom, q_var
                    )
                    loss.backward()
                    step_v += w * vl.detach().float()
                    step_q += ql.detach().float()
                    step_loss += loss.detach().float()
                    del mb
                if self._agc_clip > 0.0:
                    _adaptive_grad_clip_(self._agc_params, self._agc_clip)
                gnorm = nn.utils.clip_grad_norm_(self._critic_params, 0.5)
                loss_now, gnorm_now = torch.stack(
                    (step_loss, gnorm.detach().float())
                ).tolist()
                if not (math.isfinite(loss_now) and math.isfinite(gnorm_now)):
                    self.optimizer.zero_grad(set_to_none=True)  # never applied
                    skipped += 1
                    continue
                self.optimizer.step()
                tot_v += step_v
                tot_q += step_q
                norms.append(gnorm_now)
                steps += 1
        self.optimizer.zero_grad(set_to_none=True)
        return tot_v, tot_q, steps, skipped, norms

    def update(
        self,
        batch: Batch,
        rng: np.random.Generator,
        entropy_coef: float | None = None,
        actor_frozen: bool = False,
    ) -> PPOStats:
        """One PPO update over `batch`. `actor_frozen` (train.py
        --actor-freeze-updates): skip the PPO epochs and train only the critic
        (ppo_epochs + critic_extra_epochs critic-only passes) -- a fresh
        critic learns before its advantages steer the actor."""
        cfg = self.config
        eff_entropy_coef = (
            float(cfg.entropy_coef) if entropy_coef is None else float(entropy_coef)
        )
        display_coef = float(cfg.display_value_coef)
        value_coef = float(cfg.value_loss_coef)
        device = batch.obs.device
        micro = int(cfg.micro_batch_rows or 0)
        # TrainingConfig.batch_on_host: the batch stays in host memory and
        # each minibatch / chunk is shipped to the learner's device as it is
        # needed (HostBatchLoader) -- the same device tensors, so the same
        # update, as a device-resident batch.
        host_loader = None
        learner_device = next(self.model.parameters()).device
        ppo_bs = self.ppo_batch_size(int(batch.obs.shape[0]))
        if device.type == "cpu" and learner_device.type != "cpu":
            n_rows = int(batch.obs.shape[0])
            mb_max = max(
                (b - a for a, b in _minibatch_bounds(n_rows, ppo_bs)),
                default=1,
            )
            host_loader = HostBatchLoader(
                batch, learner_device, min(micro, mb_max) if micro > 0 else mb_max
            )
            device = learner_device
        # Accumulate as device tensors; the one host sync per minibatch is the
        # KL guard's (below) -- the totals are read once at the end.
        total_policy = torch.zeros((), device=device)
        total_value = torch.zeros((), device=device)
        total_display = torch.zeros((), device=device)
        total_entropy = torch.zeros((), device=device)
        total_entropy_bonus = torch.zeros((), device=device)
        total_kl = torch.zeros((), device=device)
        total_kl_k3 = torch.zeros((), device=device)
        total_clip = torch.zeros((), device=device)
        total_kl_anchor = torch.zeros((), device=device)
        total_q = torch.zeros((), device=device)
        total_qf = torch.zeros((), device=device)
        total_qf_n = torch.zeros((), device=device)
        total_qt = torch.zeros((), device=device)
        total_qt_n = torch.zeros((), device=device)
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
        nonfinite_grad = False
        kl0 = float("nan")
        ratio_dev0: torch.Tensor | None = None
        actor_norms: list[float] = []
        critic_norms: list[float] = []
        display_norms: list[float] = []
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
        extra_epochs = int(cfg.critic_extra_epochs or 0)
        _zero = torch.zeros((), device=device)
        shuffle0 = _rollout.SHUFFLE_SECONDS[0]
        with record_function("step12/inner_loop"):
            for _ in range(0 if actor_frozen else cfg.ppo_epochs):
                if kl_stopped_at >= 0:
                    break
                for sel in iter_minibatch_indices(batch, ppo_bs, rng):
                    n_mb = int(sel.shape[0])
                    chunked = micro > 0 and n_mb > micro
                    if not chunked:
                        mb = (
                            host_loader.gather(sel)
                            if host_loader is not None
                            else gather_minibatch(batch, sel)
                        )
                        t = self._minibatch_terms(
                            mb, eff_entropy_coef, display_coef, value_coef, device
                        )
                        with record_function("step12c/backward"):
                            self.optimizer.zero_grad()
                            t["loss"].backward()
                    else:
                        # Micro-batching (TrainingConfig.micro_batch_rows):
                        # gather, forward and backward the minibatch chunk by
                        # chunk, accumulating gradients; every term weighted
                        # to the minibatch mean, so the step is the same.
                        self.optimizer.zero_grad()
                        fold_denom = None
                        if self._q_fold_sup > 0.0:
                            gm_sel = (
                                host_loader.gate_mask_rows(sel)
                                if host_loader is not None
                                else batch.gate_masks[sel]
                            )
                            fold_denom = (
                                gm_sel[..., GATE_FOLD]
                                .float()
                                .sum()
                                .clamp_min(1.0)
                            )
                        q_var = self._minibatch_q_var(batch, host_loader, sel)
                        t = None
                        for lo in range(0, n_mb, micro):
                            sub_mb = (
                                host_loader.gather(sel[lo : lo + micro])
                                if host_loader is not None
                                else gather_minibatch(batch, sel[lo : lo + micro])
                            )
                            w = float(min(micro, n_mb - lo)) / float(n_mb)
                            tc = self._minibatch_terms(
                                sub_mb, eff_entropy_coef, display_coef, value_coef,
                                device, weight=w, fold_denom=fold_denom, q_var=q_var,
                            )
                            with record_function("step12c/backward"):
                                tc["loss"].backward()
                            t = _merge_chunk_terms(t, tc, w)
                            del sub_mb, tc
                    loss = t["loss"]
                    kl = t["kl"]
                    policy_loss = t["policy_loss"]
                    value_loss = t["value_loss"]
                    display_loss = t["display_loss"]
                    entropy_loss = t["entropy_loss"]
                    q_loss = t["q_loss"]
                    kl_anchor_term = t["kl_anchor_term"]
                    gate_kl, anchor_kl, beta_kl = t["gate_kl"], t["anchor_kl"], t["beta_kl"]
                    if t["qf"] is not None:
                        total_qf += t["qf"]
                        total_qf_n += t["qf_n"]
                    if t["qt"] is not None:
                        total_qt += t["qt"]
                        total_qt_n += t["qt_n"]

                    # Clip BEFORE the guard's sync, so the pre-clip gradient
                    # norms ride it (the step applied below is the same: grads
                    # -> AGC -> split clip -> step; the guard only decides
                    # whether it is applied).
                    with record_function("step12d/grad_clip"):
                        # Per-tensor adaptive clip (opt-in) BEFORE the per-group
                        # split clip: tames heavy-tailed spikes tensor-by-tensor.
                        if self._agc_clip > 0.0:
                            _adaptive_grad_clip_(self._agc_params, self._agc_clip)
                        # The display head's own norm (it shares the actor's
                        # clip group), measured before that clip scales it.
                        gn_d = _zero
                        _dg = [p.grad for p in self._display_params if p.grad is not None]
                        if _dg:
                            gn_d = torch.linalg.vector_norm(
                                torch.stack([torch.linalg.vector_norm(g) for g in _dg])
                            )
                        # Clip actor and critic grads SEPARATELY — a single
                        # global clip over both lets a chip-scale critic-loss
                        # spike inflate the shared grad-norm and throttle the
                        # actor's (gate) gradient on that same update.
                        gn_a = nn.utils.clip_grad_norm_(self._actor_params, 0.5)
                        gn_c = (
                            nn.utils.clip_grad_norm_(self._critic_params, 0.5)
                            if self._critic_params else _zero
                        )

                    # KL guard: checked BEFORE the optimizer step so the
                    # offending minibatch is never applied. Two thresholds:
                    #   |kl| > kl_hard  -> HARD: full rollback (revert the
                    #                      whole update) — catastrophe only.
                    #   |kl| > target_kl -> SOFT: early-stop, KEEP the
                    #                      minibatches already applied.
                    # The .tolist() forces a per-minibatch sync; negligible
                    # against multi-minute updates, and load-bearing for
                    # aborting in time.
                    #
                    # A NON-FINITE kl (or loss) is a HARD trip regardless of
                    # either threshold (review 2026-09-20 A8): `abs(nan) > x`
                    # is False, so a NaN/inf minibatch used to sail through
                    # BOTH guards, the step was applied and every actor param
                    # went NaN — train.py's NaN assert only fired afterwards,
                    # on a dead model. Never apply that step; roll the update
                    # back when a snapshot exists (kl_hard > 0), else keep the
                    # finite minibatches already applied, like a soft stop.
                    # The loss rides the same single sync: a NaN confined to
                    # returns/advantages leaves kl finite but poisons the step
                    # just the same (`kl_stop` then reports nan). So do the
                    # gradient norms (2026-09-28, ML-010): a finite loss can
                    # still back-propagate an Inf/NaN gradient, which
                    # clip_grad_norm_ would spread to every tensor.
                    kl_now, loss_now, gna_now, gnc_now, gnd_now = torch.stack((
                        kl.float(), loss.detach().float(), gn_a.detach().float(),
                        gn_c.detach().float(), gn_d.detach().float(),
                    )).tolist()
                    if count == 0 and ratio_dev0 is None:
                        # The first minibatch, before any step of this update:
                        # the rollout-vs-PPO numerical mismatch (ML-012).
                        kl0 = kl_now
                        ratio_dev0 = t["ratio_dev"]
                    step_finite = math.isfinite(kl_now) and math.isfinite(loss_now)
                    grad_finite = math.isfinite(gna_now) and math.isfinite(gnc_now)
                    if (
                        not step_finite
                        or not grad_finite
                        or self.kl_hard > 0.0
                        or self.target_kl > 0.0
                    ):
                        abs_kl = abs(kl_now)
                        if not step_finite or not grad_finite or (
                            self.kl_hard > 0.0 and abs_kl > self.kl_hard
                        ):
                            kl_stopped_at = count
                            nonfinite_grad = step_finite and not grad_finite
                            kl_stop_val = (
                                kl_now
                                if (step_finite and grad_finite)
                                or not math.isfinite(kl_now)
                                else float("nan")  # finite kl, non-finite loss/grad
                            )
                            rolled_back_flag = snapshot is not None
                            self.optimizer.zero_grad()  # never applied
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
                            self.optimizer.zero_grad()  # never applied
                            kl_stopped_at = count
                            kl_stop_val = kl_now
                            break

                    with record_function("step12d/optimizer_step"):
                        self.optimizer.step()
                    actor_norms.append(gna_now)
                    display_norms.append(gnd_now)
                    if self._critic_params:
                        critic_norms.append(gnc_now)
                    total_policy += policy_loss.detach().float()
                    total_value += value_loss.detach().float()
                    total_display += display_loss.detach().float()
                    total_entropy += t["entropy_true"]
                    total_entropy_bonus += -entropy_loss.detach().float()
                    total_kl += kl.float()
                    total_kl_k3 += t["kl_k3"]
                    total_clip += t["clip_frac"]
                    total_kl_anchor += kl_anchor_term.detach().float()
                    total_q += q_loss.detach().float()
                    total_gate_h += t["gate_h"]
                    total_anchor_h += t["anchor_h"]
                    total_beta_h += t["beta_h"]
                    total_gate_kl += gate_kl.float()
                    total_anchor_kl += anchor_kl.float()
                    total_beta_kl += beta_kl.float()
                    count += 1
        crit_steps = crit_skipped = 0
        crit_v = crit_q = _zero
        crit_epochs = (cfg.ppo_epochs + extra_epochs) if actor_frozen else extra_epochs
        if crit_epochs > 0 and critic_ok_for_extra(rolled_back_flag, self._critic_params):
            with record_function("step12e/critic_only"):
                crit_v, crit_q, crit_steps, crit_skipped, crit_norms = (
                    self._critic_only_epochs(
                        batch, rng, host_loader, micro, value_coef, crit_epochs, device
                    )
                )
            critic_norms.extend(crit_norms)
            if actor_frozen:
                # Report the critic's own losses in the value / q columns.
                total_value, total_q, count = crit_v, crit_q, crit_steps
        if not actor_frozen:
            self._ema_update_ref()
        denom = max(count, 1)
        crit_denom = max(crit_steps, 1)

        def _norm_stats(norms: list[float]) -> "tuple[float, float, float]":
            if not norms:
                return 0.0, 0.0, 0.0
            return (
                sum(norms) / len(norms),
                max(norms),
                sum(1 for x in norms if x > 0.5) / len(norms),
            )

        gna, gna_max, gca = _norm_stats(actor_norms)
        gnc, gnc_max, gcc = _norm_stats(critic_norms)
        with record_function("step13/stats_sync"):
            return PPOStats(
                policy_loss=float(total_policy.item()) / denom,
                value_loss=float(total_value.item()) / denom,
                entropy=float(total_entropy.item()) / denom,
                approx_kl=float(total_kl.item()) / denom,
                display_loss=float(total_display.item()) / denom,
                kl_anchor=float(total_kl_anchor.item()) / denom,
                q_loss=float(total_q.item()) / denom,
                q_fold_err=float(
                    (total_qf / total_qf_n.clamp_min(1.0)).item()
                ),
                q_term_err=float(
                    (total_qt / total_qt_n.clamp_min(1.0)).item()
                ),
                gate_entropy=float(total_gate_h.item()) / denom,
                anchor_entropy=float(total_anchor_h.item()) / denom,
                beta_entropy=float(total_beta_h.item()) / denom,
                gate_kl=float(total_gate_kl.item()) / denom,
                anchor_kl=float(total_anchor_kl.item()) / denom,
                beta_kl=float(total_beta_kl.item()) / denom,
                kl_stopped_at=kl_stopped_at,
                kl_stop=kl_stop_val,
                rolled_back=rolled_back_flag,
                entropy_bonus=float(total_entropy_bonus.item()) / denom,
                clip_frac=float(total_clip.item()) / denom,
                kl_k3=float(total_kl_k3.item()) / denom,
                kl0=kl0,
                ratio_dev0=(
                    float(ratio_dev0.item()) if ratio_dev0 is not None else float("nan")
                ),
                grad_norm_actor=gna,
                grad_norm_actor_max=gna_max,
                grad_clip_actor=gca,
                grad_norm_critic=gnc,
                grad_norm_critic_max=gnc_max,
                grad_clip_critic=gcc,
                grad_norm_display=(
                    sum(display_norms) / len(display_norms) if display_norms else 0.0
                ),
                nonfinite_grad=nonfinite_grad,
                critic_steps=crit_steps,
                critic_skipped=crit_skipped,
                critic_value_loss=float(crit_v.item()) / crit_denom,
                critic_q_loss=float(crit_q.item()) / crit_denom,
                shuffle_s=_rollout.SHUFFLE_SECONDS[0] - shuffle0,
            )


def value_health(batch: Batch, max_rows: int = 2_000_000) -> dict:
    """How well the critic's rollout-time values (`batch.values`) predict the
    returns they are trained toward (ML-003): explained variance
    1 - Var(R - V) / Var(R) and bias mean(V - R), overall, per street (from
    the observation's street one-hot) and per stack tier (`batch.tier_rows`,
    mix-configs). A healthy critic reads EV near 1 and bias near 0; the vSix5
    critic sat at EV 0.29 with V ~43% low, found only by an offline dump.

    Measured on an evenly strided sample of at most `max_rows` rows (every
    k-th row: no random draw, so training's streams are untouched; ~2M rows
    pin EV/bias to ~1e-3), which keeps it a fraction of a second even on a
    160M-row host batch. `n` counts the sampled rows."""
    n_all = int(batch.values.shape[0])
    step = max(1, -(-n_all // max(1, int(max_rows))))
    idx = torch.arange(0, n_all, step)
    dev = batch.values.device
    v = batch.values[idx.to(dev)].detach().float().cpu()
    r = batch.returns[idx.to(dev)].detach().float().cpu()

    def _ev(vv: torch.Tensor, rr: torch.Tensor) -> "dict[str, float]":
        k = int(vv.shape[0])
        if k < 2:
            return {"n": k, "ev": float("nan"), "bias": float("nan")}
        var_r = float(rr.var())
        resid = float((rr - vv).var())
        return {
            "n": k,
            "ev": (1.0 - resid / var_r) if var_r > 0.0 else float("nan"),
            "bias": float((vv - rr).mean()),
        }

    out: dict = {"rows": n_all, "stride": step, "all": _ev(v, r)}
    street = _row_streets(batch, idx)
    if street is not None:
        names = ("preflop", "flop", "turn", "river")
        out["street"] = {
            names[s]: _ev(v[street == s], r[street == s])
            for s in range(4)
            if bool((street == s).any())
        }
    tier_rows = getattr(batch, "tier_rows", None)
    if tier_rows:
        per: dict = {}
        for tier, spans in tier_rows.items():
            m = torch.zeros(idx.shape[0], dtype=torch.bool)
            for a, b in spans:
                m |= (idx >= int(a)) & (idx < int(b))
            if bool(m.any()):
                per[tier] = _ev(v[m], r[m])
        out["tier"] = per
    return out


def _row_streets(batch: Batch, rows: torch.Tensor) -> "torch.Tensor | None":
    """Street index (0 preflop .. 3 river) of the observation rows `rows`,
    from the street one-hot (encoding._STREET_OFF..+4: the same columns in the
    full and minimal PLO layouts), or None when the layout does not carry it
    there (NLH). Reads only those 4 columns of those rows."""
    from plo5bp.compact_obs import PackedObs
    from plo5bp.encoding import _NUM_STREET_ONEHOT, _STREET_OFF, OBS_DIM, OBS_DIM_MINIMAL

    obs = batch.obs
    width = int(obs.shape[1])
    if width not in (OBS_DIM, OBS_DIM_MINIMAL):
        return None
    cols = list(range(_STREET_OFF, _STREET_OFF + _NUM_STREET_ONEHOT))
    if isinstance(obs, PackedObs):
        onehot = torch.stack([obs.column(c, rows) for c in cols], dim=1)
    else:
        r = rows.to(obs.device)
        c = torch.tensor(cols, device=obs.device)
        onehot = obs[r[:, None], c[None, :]].detach().float().cpu()
    return onehot.argmax(dim=1)
