"""Actor-critic with hybrid gate + continuous-raise policy head.

Policy factorises as P(action) = P(gate) * P(chips | gate=Raise). The
gate is a 3-way Categorical over {Fold, CheckCall, Raise} masked by
engine legality. The raise amount is a Beta(α, β) sample on [0, 1]
mapped to chips in [min_raise, max_raise] per env via an affine
transform; α, β are produced by a small linear head with softplus so
they stay ≥ 1 (keeps the Beta unimodal and log-prob stable near the
endpoints). Short shoves are encoded as Raise at u=1 — the engine's
`max_raise_chips` already clamps to stack.

For PPO, log-prob is `log P(gate) + 1[gate==Raise] * log p(u)` in
u-space, and entropy is `H(Cat) + P(gate==Raise) * H(Beta)` — the
conditional weighting matches the generative process. The Beta head
is masked to a deterministic 0-log-prob / 0-entropy contribution in
two regimes (symmetrically in `act` and `evaluate` to keep the PPO
ratio at 1.0):

  - `min_raise == max_raise` — pot-corner / max-equals-min: the
    chips action is deterministic regardless of u.
  - `min_raise == 0` (with `max_raise > 0`) — sub-min-raise stack:
    the env redirects GATE_RAISE to `apply(Action::AllIn)` and
    ignores the chip amount, so the Beta sample is moot. Masking
    the Beta head's gradient prevents the policy from training a
    raise-amount distribution that the env never consults.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _checkpoint

from plo5bp.actions import GATE_ACTIONS, GATE_RAISE
from plo5bp.encoding import OBS_DIM
from plo5bp.sizing import (
    ANCHOR_COUNT,
    NLH_ANCHOR_SPEC,
    PLO_ANCHOR_SPEC,
    AnchorSpec,
    anchor_grid_torch,
    refine_chips_torch,
)


def anchor_spec_for_count(count: int) -> AnchorSpec:
    """Resolve the anchor spec a checkpoint was trained with from its
    head width. Each variant has exactly one ladder, so the count is a
    sufficient fingerprint (PLO 11, NLH 12)."""
    for spec in (PLO_ANCHOR_SPEC, NLH_ANCHOR_SPEC):
        if spec.count == count:
            return spec
    raise ValueError(f"no anchor spec with {count} anchors")


class ActOut(NamedTuple):
    """Uniform act() result across head versions.

    `anchor` / `refine_u` are the v2 sizing-head action components the
    rollout buffer stores so `evaluate()` can replay the exact sampled
    action (never reverse-engineered from chips). v1 fills anchor=-1,
    refine_u=the Beta u sample.
    """

    gate: torch.Tensor      # (B,) int64
    chips: torch.Tensor     # (B,) int64 chip delta; 0 for non-Raise
    log_prob: torch.Tensor  # (B,) float — JOINT log-prob
    value: torch.Tensor     # (B,) float (v2: the display value head)
    anchor: torch.Tensor    # (B,) int64
    refine_u: torch.Tensor  # (B,) float — clamped sampled u
    # Per-head log-prob components stored for per-head KL diagnostics.
    # gate is always present; anchor is the raise-row anchor log-prob
    # (v1 fills zeros — no anchor head).
    gate_log_prob: torch.Tensor    # (B,) float
    anchor_log_prob: torch.Tensor  # (B,) float
    # (2 + anchor_count)-way action MARGINAL over the Q-head layout
    # [Fold, CheckCall, Raise@anchor_0..k], used by the Expected-SARSA (VRPO)
    # advantage to form V^π(s) = Σ_a π(a) Q(s, a). Populated only when
    # act(..., return_marginal=True); None otherwise (free when off).
    action_marginal: torch.Tensor | None = None


def _maybe_checkpoint(torso, x, enabled: bool):
    """Gradient-checkpoint the torso forward when `enabled` AND grad is on
    (training). Identical math — trades activation memory for a recompute in
    backward. Skips under no_grad / inference_mode (rollout), where checkpoint
    is invalid and pointless. use_reentrant=False preserves autocast + RNG, so
    the fp32-critical reductions downstream are untouched."""
    if enabled and torch.is_grad_enabled():
        return _checkpoint.checkpoint(torso, x, use_reentrant=False)
    return torso(x)


def _symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log — compresses a huge signed range into a compact one with
    ~linear behaviour near 0 (Dreamer/MuZero). Used to grid the distributional
    critic's value support so a fixed bin set spans tiny-to-~1500bb rewards with
    fine resolution near 0 (the dynamic-range fix)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of _symlog."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


class _ResidualBlock(nn.Module):
    """y = x + ReLU(Linear(x)). Single Linear keeps post-activation output
    on the residual stream — works at depth ≥3 where the plain MLP loses
    gradient flow.

    `use_norm=True` pre-normalizes: y = x + ReLU(Linear(LayerNorm(x))) — the
    v6 plasticity change (V6_RESEARCH.md #4). LayerNorm keeps activations
    well-scaled so units don't die and the effective LR doesn't decay over a
    long self-play run. It is NOT function-preserving (it normalizes x even at
    init), so it is a fresh-stem change and MUST be paired with
    weight-decay-to-init (`l2_init_coef`) — norm-solo can hurt generalization
    (Nauman 2024). Off by default → no `norm.*` params, byte-identical to the
    pre-v6 block."""

    def __init__(self, dim: int, use_norm: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_norm else None
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x if self.norm is None else self.norm(x)
        return x + F.relu(self.linear(h))


class ActorCritic(nn.Module):
    """Shared MLP torso, gate + raise + value heads.

    Torso structure depends on `num_layers`:
    - `num_layers <= 2`: flat `Sequential(Linear, ReLU, [Linear, ReLU]*)`.
      Preserves the original parameter names (`torso.0.weight`, `torso.2.weight`)
      so existing 128×2 checkpoints still load.
    - `num_layers >= 3`: input projection + `num_layers - 1` `_ResidualBlock`s.
      Residuals are needed at depth — without them the 2048×4 net fails to
      train.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        obs_dim: int = OBS_DIM,
        num_layers: int = 2,
        torso_layernorm: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if torso_layernorm and num_layers < 3:
            raise ValueError(
                "torso_layernorm requires num_layers >= 3 (the residual torso)"
            )
        if num_layers >= 3:
            input_block = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
            blocks: list[nn.Module] = [
                _ResidualBlock(hidden_dim, use_norm=torso_layernorm)
                for _ in range(num_layers - 1)
            ]
            self.torso = nn.Sequential(input_block, *blocks)
        else:
            layers: list[nn.Module] = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            self.torso = nn.Sequential(*layers)
        self.gate_head = nn.Linear(hidden_dim, GATE_ACTIONS)
        # Two outputs → (α, β). softplus+1 keeps them ≥ 1 so the Beta is
        # unimodal and log-prob doesn't blow up at u∈{0,1}.
        self.raise_head = nn.Linear(hidden_dim, 2)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, obs: torch.Tensor, gate_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (masked gate logits, (α, β), value)."""
        z = _maybe_checkpoint(self.torso, obs, getattr(self, "_grad_checkpoint", False))
        gate_logits = self.gate_head(z).masked_fill(~gate_mask, -1e9)
        raise_params = F.softplus(self.raise_head(z)) + 1.0
        value = self.value_head(z).squeeze(-1)
        return gate_logits, raise_params, value

    @staticmethod
    def _recover_u(
        chips: torch.Tensor, bounds: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map stored chip amounts back to u ∈ (0, 1) for Beta log-prob.

        Clamps into the open interval to avoid -inf from Beta.log_prob
        at the endpoints (rounding can put `u` exactly at 0 or 1).
        """
        min_raise = bounds[..., 0].to(dtype)
        max_raise = bounds[..., 1].to(dtype)
        width = (max_raise - min_raise).clamp(min=1.0)
        u = ((chips.to(dtype) - min_raise) / width).clamp(1e-6, 1.0 - 1e-6)
        return u, min_raise, width

    head_version = 1

    def act(
        self,
        obs: torch.Tensor,
        gate_mask: torch.Tensor,
        sizing: torch.Tensor,
        deterministic: bool = False,
    ) -> ActOut:
        """Sample or argmax. Returns an ActOut (anchor fixed at -1).

        `sizing` is `(B, 2)` (min_raise, max_raise) or the v2-style
        `(B, 4)` (min, max, pot, to_call) — only the first two columns
        are used. `chips` is the raw chip delta the caller feeds to
        `env.step_hybrid`; zero when `gate != Raise`.
        """
        bounds = sizing[..., :2]
        gate_logits, raise_params, value = self.forward(obs, gate_mask)
        gate_dist = torch.distributions.Categorical(logits=gate_logits)
        if deterministic:
            gate = gate_logits.argmax(dim=-1)
        else:
            gate = gate_dist.sample()
        gate_log_prob = gate_dist.log_prob(gate)

        alpha = raise_params[..., 0]
        beta = raise_params[..., 1]
        beta_dist = torch.distributions.Beta(alpha, beta)
        if deterministic:
            # Beta mean is α/(α+β) — matches the UI's single recommendation.
            u = alpha / (alpha + beta)
        else:
            u = beta_dist.sample()
        u = u.clamp(1e-6, 1.0 - 1e-6)

        min_raise = bounds[..., 0].to(obs.dtype)
        max_raise = bounds[..., 1].to(obs.dtype)
        width = (max_raise - min_raise).clamp(min=1.0)
        raise_chips_f = min_raise + u * width
        raise_chips = (
            raise_chips_f.round()
            .to(torch.long)
            .clamp(min=bounds[..., 0].to(torch.long), max=bounds[..., 1].to(torch.long))
        )

        raise_mask = gate == GATE_RAISE
        beta_log = beta_dist.log_prob(u)
        # Degenerate-range mask: zero the Beta log-prob in two regimes
        # so the PPO ratio matches `evaluate` exactly. (1) max==min:
        # only one valid chip amount. (2) min==0 with max>0: env
        # redirects to apply(AllIn) and ignores the chip amount.
        degenerate = (bounds[..., 0] >= bounds[..., 1]) | (bounds[..., 0] == 0)
        beta_log = torch.where(degenerate, torch.zeros_like(beta_log), beta_log)
        log_prob = gate_log_prob + torch.where(
            raise_mask, beta_log, torch.zeros_like(beta_log)
        )
        chips_out = torch.where(raise_mask, raise_chips, torch.zeros_like(raise_chips))
        return ActOut(
            gate=gate,
            chips=chips_out,
            log_prob=log_prob,
            value=value,
            anchor=torch.full_like(gate, -1),
            refine_u=u,
            gate_log_prob=gate_log_prob,
            anchor_log_prob=torch.zeros_like(log_prob),  # v1: no anchor head
        )

    def evaluate(
        self,
        obs: torch.Tensor,
        gate_mask: torch.Tensor,
        bounds: torch.Tensor,
        gate_actions: torch.Tensor,
        raise_chips: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (log_prob, entropy, value) for the stored actions.

        Entropy is the generative-model entropy: `H(Cat) + P(Raise) *
        H(Beta)`. The ratio `log_prob - old_log_prob` is taken in u-space
        — width is state-dependent and identical between rollout and
        evaluate, so the change-of-variables factor cancels.
        """
        gate_logits, raise_params, value = self.forward(obs, gate_mask)
        gate_dist = torch.distributions.Categorical(logits=gate_logits)
        gate_log_prob = gate_dist.log_prob(gate_actions)
        gate_entropy = gate_dist.entropy()

        alpha = raise_params[..., 0]
        beta = raise_params[..., 1]
        beta_dist = torch.distributions.Beta(alpha, beta)
        u, _, _ = self._recover_u(raise_chips, bounds, obs.dtype)
        beta_log = beta_dist.log_prob(u)

        degenerate = (bounds[..., 0] >= bounds[..., 1]) | (bounds[..., 0] == 0)
        beta_log = torch.where(degenerate, torch.zeros_like(beta_log), beta_log)
        beta_entropy = beta_dist.entropy()
        beta_entropy = torch.where(
            degenerate, torch.zeros_like(beta_entropy), beta_entropy
        )

        raise_mask = gate_actions == GATE_RAISE
        log_prob = gate_log_prob + torch.where(
            raise_mask, beta_log, torch.zeros_like(beta_log)
        )

        gate_probs = F.softmax(gate_logits, dim=-1)
        p_raise = gate_probs[..., GATE_RAISE]
        entropy = gate_entropy + p_raise * beta_entropy
        return log_prob, entropy, value


_U_EPS = 1e-6
_INTERIOR = ANCHOR_COUNT - 2  # 9 refinable anchors (k = 1..9)


class ActorCriticV2(nn.Module):
    """v2 actor: gate + 11-anchor categorical sizing head with per-anchor
    Beta refinement sliders + display value head.

    Sizing factorises as P(chips | Raise) = P(anchor) * p(u | anchor),
    where the anchors are pot fractions 0,10,...,100% (see
    plo5bp.sizing.anchor_grid_torch — the canonical chips/legality/
    bracket math shared with numpy consumers). Anchors 0 and 10 are
    atoms (exact min / pot); interior anchors carry a Beta(α_k, β_k)
    over the bracket [f_k − 0.05, f_k + 0.05], whose lazy center (the
    untrained Beta mean) sits exactly on the anchor.

    `evaluate` replays the STORED (gate, anchor, u) — it never recovers
    them from chips, so act/evaluate log-probs match exactly (the PPO
    ratio is 1.0 at epoch start). Degenerate handling mirrors v1: rows
    where the chosen anchor has no live bracket (atoms, collapsed
    brackets, the short-shove single-atom regime) contribute zero Beta
    log-prob/entropy symmetrically in both paths.

    The value head here is the DISPLAY head (observation-only, serves
    the UI); training advantages come from the separate CentralCritic.
    """

    head_version = 2
    # v5 sets this True: detach the anchor-prob weighting inside
    # beta_h_eff so the maximized entropy BONUS cannot pay the sizing
    # head to shift anchor mass onto the (Beta-masked) atom anchors —
    # Beta differential entropy is <= 0 for alpha,beta >= 1, so with
    # the weight in the graph the bonus rewards moving mass OFF the
    # refinable interior anchors, a verified pressure toward min/pot
    # sizing (2026-07-06 review, V5_DESIGN.md B3). Kept False here and
    # on v4 so those stems stay byte-identical on resume.
    _detach_beta_h_weight = False

    def __init__(
        self,
        hidden_dim: int = 512,
        obs_dim: int = OBS_DIM,
        num_layers: int = 2,
        anchor_spec: AnchorSpec = PLO_ANCHOR_SPEC,
        torso_layernorm: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if torso_layernorm and num_layers < 3:
            raise ValueError(
                "torso_layernorm requires num_layers >= 3 (the residual torso)"
            )
        if num_layers >= 3:
            input_block = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
            blocks: list[nn.Module] = [
                _ResidualBlock(hidden_dim, use_norm=torso_layernorm)
                for _ in range(num_layers - 1)
            ]
            self.torso = nn.Sequential(input_block, *blocks)
        else:
            layers: list[nn.Module] = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            self.torso = nn.Sequential(*layers)
        # The spec is the variant's ladder (PLO 11 anchors / NLH 12 with
        # the all-in atom). Atoms are always first+last, so the refine
        # head is `count - 2` interior slots in every spec.
        self.anchor_spec = anchor_spec
        self._anchor_count = anchor_spec.count
        self._interior = anchor_spec.count - 2
        self.gate_head = nn.Linear(hidden_dim, GATE_ACTIONS)
        self.anchor_head = nn.Linear(hidden_dim, self._anchor_count)
        # (α, β) per interior anchor; softplus+1 keeps each Beta unimodal.
        self.refine_head = nn.Linear(hidden_dim, self._interior * 2)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, obs: torch.Tensor, gate_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (masked gate logits, raw anchor logits, refine params
        (B, 9, 2), display value). Anchor legality masking happens in
        act/evaluate where the sizing context is available.

        The display value head reads a DETACHED torso. In v2 the GAE
        values come from the CentralCritic; this head exists for the UI
        display only, but its regression target is raw-bb returns
        (±100s of bb), so trained through the torso its gradients dwarf
        the unit-scale policy gradient by orders of magnitude and the
        torso becomes a value-fitting network with policy heads as
        passengers (2026-06-11 review). Detaching keeps the head
        readable without letting it steer the torso. v1's ActorCritic
        is untouched — there the value head feeds GAE and is
        load-bearing."""
        z = _maybe_checkpoint(self.torso, obs, getattr(self, "_grad_checkpoint", False))
        gate_logits = self.gate_head(z).masked_fill(~gate_mask, -1e9)
        anchor_logits = self.anchor_head(z)
        refine = F.softplus(self.refine_head(z)).view(
            *z.shape[:-1], self._interior, 2
        ) + 1.0
        value = self.value_head(z.detach()).squeeze(-1)
        return gate_logits, anchor_logits, refine, value

    def _gather_refine(
        self, refine: torch.Tensor, anchor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row (α, β) of the chosen anchor; atoms clamp to a valid
        interior index (their Beta term is masked out downstream)."""
        idx = (anchor - 1).clamp(0, self._interior - 1)
        ab = refine.gather(
            -2, idx[..., None, None].expand(*idx.shape, 1, 2)
        ).squeeze(-2)
        return ab[..., 0], ab[..., 1]

    def _anchor_dist(self, anchor_head_out: torch.Tensor, grid):
        """Categorical over the LEGAL anchors from this head's raw output.

        v2: a flat softmax over the 11 anchor logits with illegal anchors
        masked to -inf. `act`/`evaluate` are head-agnostic — they only touch
        the returned Categorical (`.sample()`, `.log_prob()`, `.probs`,
        `.entropy()`) — so a subclass overrides only this method to change the
        size parameterization (see ActorCriticV4)."""
        logits = anchor_head_out.masked_fill(~grid.legal, -1e9)
        return torch.distributions.Categorical(logits=logits)

    def act(
        self,
        obs: torch.Tensor,
        gate_mask: torch.Tensor,
        sizing: torch.Tensor,
        deterministic: bool = False,
        return_marginal: bool = False,
    ) -> ActOut:
        """Sample or argmax. `sizing` is (B, 4) int64
        [min_raise, max_raise, pot, to_call].

        `return_marginal=True` also returns the (2 + anchor_count)-way action
        marginal π over the Q-head layout [Fold, CheckCall, Raise@anchor_k]
        (for the Expected-SARSA / VRPO advantage). Built from probs already
        computed in THIS forward — no extra network pass, no extra RNG draw,
        so trajectories are bit-identical whether or not it is requested."""
        gate_logits, anchor_logits, refine, value = self.forward(obs, gate_mask)
        grid = anchor_grid_torch(sizing, self.anchor_spec)

        gate_dist = torch.distributions.Categorical(logits=gate_logits)
        if deterministic:
            gate = gate_logits.argmax(dim=-1)
        else:
            gate = gate_dist.sample()
        gate_log_prob = gate_dist.log_prob(gate)

        anchor_dist = self._anchor_dist(anchor_logits, grid)
        if deterministic:
            anchor = anchor_dist.probs.argmax(dim=-1)
        else:
            anchor = anchor_dist.sample()
        anchor_log_prob = anchor_dist.log_prob(anchor)

        alpha, beta = self._gather_refine(refine, anchor)
        beta_dist = torch.distributions.Beta(alpha, beta)
        if deterministic:
            u = alpha / (alpha + beta)
        else:
            u = beta_dist.sample()
        u = u.clamp(_U_EPS, 1.0 - _U_EPS)

        refine_active = grid.refine_ok.gather(-1, anchor[..., None]).squeeze(-1)
        anchor_chips = grid.chips.gather(-1, anchor[..., None]).squeeze(-1)
        refined_chips = refine_chips_torch(anchor, u, sizing, self.anchor_spec)
        raise_chips = torch.where(refine_active, refined_chips, anchor_chips)

        beta_log = beta_dist.log_prob(u)
        sizing_log = anchor_log_prob + torch.where(
            refine_active, beta_log, torch.zeros_like(beta_log)
        )
        raise_mask = gate == GATE_RAISE
        log_prob = gate_log_prob + torch.where(
            raise_mask, sizing_log, torch.zeros_like(sizing_log)
        )
        chips_out = torch.where(
            raise_mask, raise_chips, torch.zeros_like(raise_chips)
        )
        action_marginal = None
        if return_marginal:
            # π over [Fold, CheckCall, Raise@anchor_0..k] — the exact index
            # the CentralCritic Q head uses (ppo.py q_idx: gate for
            # Fold/CheckCall, 2+anchor for Raise). p_raise × anchor-marginal
            # is the joint raise-size mass; sums to 1 (gate probs sum to 1,
            # anchor marginal sums to 1).
            gate_probs_m = gate_dist.probs
            anchor_probs_m = anchor_dist.probs
            action_marginal = torch.cat(
                [
                    gate_probs_m[..., :GATE_RAISE],
                    gate_probs_m[..., GATE_RAISE : GATE_RAISE + 1] * anchor_probs_m,
                ],
                dim=-1,
            )
        return ActOut(
            gate=gate,
            chips=chips_out,
            log_prob=log_prob,
            value=value,
            anchor=anchor,
            refine_u=u,
            gate_log_prob=gate_log_prob,
            anchor_log_prob=anchor_log_prob,
            action_marginal=action_marginal,
        )

    def evaluate(
        self,
        obs: torch.Tensor,
        gate_mask: torch.Tensor,
        sizing: torch.Tensor,
        gate_actions: torch.Tensor,
        anchor_actions: torch.Tensor,
        refine_u: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        """Return (log_prob, entropy, display_value, gate_H, anchor_H,
        beta_H_eff, gate_log_prob, anchor_log_prob) for stored actions.
        The last two are the per-head log-prob components (for per-head
        KL diagnostics in the trainer).

        Entropy follows the generative process:
        H(gate) + P(Raise) · (H(anchor) + Σ_k p_k · H(Beta_k) · refine_ok_k).
        Masked anchors contribute exactly zero. The decomposition terms
        are returned so training can log Hg/Ha/Hb separately (the anchor
        head adds up to log(11) ≈ 2.4 nats vs the v1 entropy scale).

        The P(Raise) weighting is DETACHED from the graph. The joint
        entropy is mathematically correct with the gradient flowing
        through p_raise, but as a maximized bonus that gradient pays the
        GATE head to shift mass onto Raise (the branch holding ~2.4 nats
        of anchor entropy) — the bonus's own optimum is p_fold ≈ 8%, and
        combined with early advantage pressure it drove fold to ~5e-5
        at every node within 5 updates (vTwo1 2026-06-11; four-way code
        review converged on this line). Detaching makes the gate head
        feel pure H(gate) pressure while the sizing heads still receive
        their entropy bonus at the (stop-grad) p_raise weight. v1 had
        the same form but its conditional Beta entropy is ~0, which is
        why this never bit before the anchor head existed.
        """
        gate_logits, anchor_logits, refine, value = self.forward(obs, gate_mask)
        grid = anchor_grid_torch(sizing, self.anchor_spec)

        gate_dist = torch.distributions.Categorical(logits=gate_logits)
        gate_log_prob = gate_dist.log_prob(gate_actions)
        gate_entropy = gate_dist.entropy()

        anchor_dist = self._anchor_dist(anchor_logits, grid)
        anchor_log_prob = anchor_dist.log_prob(anchor_actions)

        alpha, beta = self._gather_refine(refine, anchor_actions)
        beta_dist = torch.distributions.Beta(alpha, beta)
        u = refine_u.clamp(_U_EPS, 1.0 - _U_EPS)
        refine_active = grid.refine_ok.gather(
            -1, anchor_actions[..., None]
        ).squeeze(-1)
        beta_log = beta_dist.log_prob(u)
        sizing_log = anchor_log_prob + torch.where(
            refine_active, beta_log, torch.zeros_like(beta_log)
        )
        raise_mask = gate_actions == GATE_RAISE
        log_prob = gate_log_prob + torch.where(
            raise_mask, sizing_log, torch.zeros_like(sizing_log)
        )

        gate_probs = F.softmax(gate_logits, dim=-1)
        p_raise = gate_probs[..., GATE_RAISE]
        anchor_probs = anchor_dist.probs
        all_beta = torch.distributions.Beta(refine[..., 0], refine[..., 1])
        beta_h = all_beta.entropy()                       # (B, interior)
        interior_ok = grid.refine_ok[..., 1:self._anchor_count - 1]
        w_interior = anchor_probs[..., 1:self._anchor_count - 1]
        if self._detach_beta_h_weight:
            w_interior = w_interior.detach()
        beta_h_eff = (w_interior * beta_h * interior_ok).sum(-1)
        anchor_entropy = anchor_dist.entropy()
        entropy = gate_entropy + p_raise.detach() * (anchor_entropy + beta_h_eff)
        # The raw head outputs (gate_logits, the head-specific 2nd output
        # — v2 anchor logits / v4 (mu,s) / v5 mixture params — and refine)
        # ride along so the kl-anchor magnet can compute KL(current||ref)
        # from THIS forward instead of a second forward-with-grad per
        # minibatch (~+18-20 GiB; ppo._kl_to_reference `cur=`). They are
        # intermediate tensors already in the graph, so returning them is
        # free memory-wise.
        return (
            log_prob, entropy, value, gate_entropy, anchor_entropy,
            beta_h_eff, gate_log_prob, anchor_log_prob,
            gate_logits, anchor_logits, refine,
        )


def _discretized_logistic_probs(
    mu: torch.Tensor,
    s: torch.Tensor,
    legal: torch.Tensor,
    count: int | None = None,
) -> torch.Tensor:
    """P(anchor k) from a Logistic(mu, s) latent discretized over the ordered
    anchor-index axis: anchor k owns the unit interval [k-0.5, k+0.5], and the
    lowest/highest LEGAL anchors absorb the outer tails so exact-min and
    exact-pot stay first-class concentratable actions when legal, and a
    beyond-range mu lands on the nearest legal anchor (not spuriously on min).
    Illegal anchors are masked out and the result renormalized over the legal set.

    `mu`, `s` are (...,) on the index axis (s > 0); `legal` is (..., count) bool.
    The standardized bin edges are clamped for tail stability. The end-bin tail
    absorption is what avoids v1's failure (a plain continuous density gives the
    exact endpoints ~0 mass); the low-dimensional ordered (mu, s) parameterization
    is what avoids the v2/v3 flat-categorical instability.

    `count` defaults to the legal mask's trailing width (the spec's
    anchor count), so the same head serves any ladder length."""
    if count is None:
        count = int(legal.shape[-1])
    idx = torch.arange(count, device=mu.device, dtype=mu.dtype)
    mu_e = mu[..., None]
    s_e = s[..., None].clamp_min(1e-3)
    cdf_hi = torch.sigmoid(((idx + 0.5 - mu_e) / s_e).clamp(-12.0, 12.0))
    cdf_lo = torch.sigmoid(((idx - 0.5 - mu_e) / s_e).clamp(-12.0, 12.0))
    # Tail absorption at the LEGAL edges, not the fixed first/last anchor: the
    # lowest legal anchor absorbs all mass below it, the highest legal anchor all
    # mass above it. Reduces EXACTLY to fixed-edge absorption when anchors 0 and
    # count-1 are themselves legal (the common case). Without this, a narrow s
    # with mu pinned past a capped legal range dumped ~all mass on the min anchor
    # — the only legal anchor whose raw-CDF formula doesn't cancel to ~0 under the
    # [-12,12] clamp — instead of the highest legal anchor nearest mu.
    legal_b = legal.bool()
    legal_l = legal_b.to(torch.long)
    n_before = legal_l.cumsum(-1) - legal_l                   # legal strictly before k
    n_after = legal_l.flip(-1).cumsum(-1).flip(-1) - legal_l  # legal strictly after k
    is_lowest = legal_b & (n_before == 0)
    is_highest = legal_b & (n_after == 0)
    lo_edge = torch.where(is_lowest, torch.zeros_like(cdf_lo), cdf_lo)
    hi_edge = torch.where(is_highest, torch.ones_like(cdf_hi), cdf_hi)
    p = (hi_edge - lo_edge).clamp_min(1e-9) * legal_b.to(cdf_hi.dtype)
    return p / p.sum(-1, keepdim=True).clamp_min(1e-12)


class ActorCriticV4(ActorCriticV2):
    """v4 sizing head: an ordinal *discretized-logistic* over the 11 anchors.

    Identical to ActorCriticV2 in every respect EXCEPT how the anchor is chosen.
    Instead of 11 free, unordered logits (a flat softmax), the size head emits a
    single location `mu` and scale `s`, and each anchor's probability is the
    slice of a Logistic(mu, s) sitting over it (`_discretized_logistic_probs`).
    The per-anchor Beta refine, gate head, value head, torso, `act`/`evaluate`
    and the rollout/PPO interfaces are all inherited unchanged.

    Why: the flat categorical gave heavy-tailed, conflicting gradients across
    stack depths (a 20bb spot wants small bets, a 250bb spot wants large) that
    spiked the size-head KL and collapsed every full-LR run. With a single
    ordered location, that conflict resolves by `mu` becoming a smooth function
    of stack depth (a regression the torso handles) instead of a tug-of-war over
    distant independent logits; a scale FLOOR makes a one-hot spike impossible,
    so the size KL stays small. The end anchors absorb the logistic tails, so
    exact-min and exact-pot stay reliably hittable (v2's win) — the property a
    plain continuous head would lose. Checkpoints sniff on `size_head.weight`."""

    head_version = 3

    def __init__(
        self,
        hidden_dim: int = 512,
        obs_dim: int = OBS_DIM,
        num_layers: int = 2,
        size_scale_floor: float = 0.3,
        size_scale_cap: float = 5.0,
        anchor_spec: AnchorSpec = PLO_ANCHOR_SPEC,
        torso_layernorm: bool = False,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            obs_dim=obs_dim,
            num_layers=num_layers,
            anchor_spec=anchor_spec,
            torso_layernorm=torso_layernorm,
        )
        # Swap the flat anchor head for a 2-output (mu_raw, s_raw) head.
        # The (mu, s) parameterization is count-independent — a longer
        # ladder only lengthens the support the logistic is sliced over.
        del self.anchor_head
        self.size_head = nn.Linear(hidden_dim, 2)
        self._size_floor = float(size_scale_floor)
        self._size_span = float(size_scale_cap) - float(size_scale_floor)

    def forward(self, obs, gate_mask):
        z = _maybe_checkpoint(self.torso, obs, getattr(self, "_grad_checkpoint", False))
        gate_logits = self.gate_head(z).masked_fill(~gate_mask, -1e9)
        size_params = self.size_head(z)  # (..., 2): mu_raw, s_raw
        refine = F.softplus(self.refine_head(z)).view(
            *z.shape[:-1], self._interior, 2
        ) + 1.0
        value = self.value_head(z.detach()).squeeze(-1)
        return gate_logits, size_params, refine, value

    def _anchor_dist(self, size_params: torch.Tensor, grid):
        # mu on the index axis: center (count-1)/2, ranging +/-((count-1)/2 + 2)
        # so it reaches BEYOND [0, count-1] and an end anchor can carry the
        # majority of the mass. s floored (no spike) and soft-capped (not flat).
        c = (self._anchor_count - 1) / 2.0
        mu = c + (c + 2.0) * torch.tanh(size_params[..., 0])
        s = self._size_floor + self._size_span * torch.sigmoid(size_params[..., 1])
        probs = _discretized_logistic_probs(mu, s, grid.legal)
        return torch.distributions.Categorical(probs=probs)


class ActorCriticV5(ActorCriticV4):
    """v5 sizing head: a K-component MIXTURE of discretized logistics.

    One Logistic(mu, s) sliced over the ordered ladder is unimodal in the
    interior (only the legal-edge tail absorption can add end bumps), so v4
    provably cannot put meaningful mass on an interior size AND a distant
    size at the same node — the solver-style menu (e.g. 33% block vs pot).
    v5 emits K (mu_k, s_k) pairs plus K mixture logits; the anchor
    distribution is the weighted sum of the K discretized logistics.
    Because anchors are DISCRETE, that marginal is itself just a
    Categorical over the <= count legal anchors: log-prob and entropy are
    exact closed forms, sampling the marginal is distributionally
    identical to sampling a component first, and no component index is
    ever stored — `act`/`evaluate`, the rollout buffer, PPO, and the UI
    anchors histogram all inherit unchanged through `_anchor_dist`.

    Guardrails from the v2 collapse postmortem (V5_DESIGN.md §2):

    - EPSILON WEIGHT FLOOR (`mix_weight_floor`): keeps every component's
      gradient alive (no permanently dead components — a zero-weight
      component's (mu, s) receive no responsibility-weighted gradient and
      never recover) and bounds importance ratios (P(a) >= eps*P_k(a)).
      Applied INSIDE `_anchor_dist` so act/evaluate see identical weights.
      A fixed constant by design — it is not stored in the state dict, so
      a run-time flag would let serving silently diverge from training.
    - NO H(w) ENTROPY BONUS: the marginal entropy `evaluate` already
      returns pays for multimodal spread exactly when it spreads the
      anchor pmf; a direct bonus on the weight entropy is v2's
      flat-collapse analog (it pins w uniform and blurs the menu into a
      fat unimodal average). `mixture_params` exposes (mu, s, w) so Hw
      can be LOGGED as a diagnostic.
    - INIT SPREAD: zeroed head weights + bias-driven component locations
      spread across the ladder (see `_init_mix_head`), so cold starts
      explore a genuine menu instead of three coincident humps (mode
      collapse to unimodal is graceful — w one-hot IS v4 — but starting
      coalesced makes it the default outcome).
    - per-component s inherits v4's floor/cap — the anti-spike property
      that made v4 trainable — and `_detach_beta_h_weight` is True here
      (the B3 entropy-artifact fix; v4/v2 keep the legacy behavior).

    Checkpoints sniff on `mix_head.weight`, shape (3K, hidden): rows
    [0:K) = mu_raw, [K:2K) = s_raw, [2K:3K) = mixture logits. The tensor
    name is deliberately NOT `size_head` — the UI's class sniffer checks
    key names, and reusing v4's name would rebuild a V4, shape-fail, and
    silently serve a random-init placeholder."""

    head_version = 4
    _detach_beta_h_weight = True

    def __init__(
        self,
        hidden_dim: int = 512,
        obs_dim: int = OBS_DIM,
        num_layers: int = 2,
        size_scale_floor: float = 0.3,
        size_scale_cap: float = 5.0,
        anchor_spec: AnchorSpec = PLO_ANCHOR_SPEC,
        mixture_k: int = 3,
        mix_weight_floor: float = 0.03,
        torso_layernorm: bool = False,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            obs_dim=obs_dim,
            num_layers=num_layers,
            size_scale_floor=size_scale_floor,
            size_scale_cap=size_scale_cap,
            anchor_spec=anchor_spec,
            torso_layernorm=torso_layernorm,
        )
        if mixture_k < 1:
            raise ValueError(f"mixture_k must be >= 1, got {mixture_k}")
        if not 0.0 <= mix_weight_floor < 1.0 / mixture_k:
            raise ValueError(
                "mix_weight_floor must be in [0, 1/mixture_k), got "
                f"{mix_weight_floor} (K={mixture_k})"
            )
        del self.size_head
        self._mixture_k = int(mixture_k)
        self._mix_floor = float(mix_weight_floor)
        self.mix_head = nn.Linear(hidden_dim, 3 * self._mixture_k)
        self._init_mix_head()

    def _init_mix_head(self) -> None:
        """Zero weights + bias-driven spread. mu_raw biases linspace(-1, 1)
        put component locations at ~(-0.3, mid, count+0.3) on the index
        axis for K=3 (tanh(±1) ≈ ±0.76); s_raw bias 0 = mid-scale; logit
        biases 0 = uniform weights (the floor then leaves them uniform).
        Weights must be exactly zero for the spread to hold at init — a
        default-init Linear at hidden 2048 produces O(1) random logits
        that swamp the biases."""
        with torch.no_grad():
            self.mix_head.weight.zero_()
            self.mix_head.bias.zero_()
            k = self._mixture_k
            if k > 1:
                self.mix_head.bias[0:k] = torch.linspace(-1.0, 1.0, k)

    def forward(self, obs, gate_mask):
        z = _maybe_checkpoint(self.torso, obs, getattr(self, "_grad_checkpoint", False))
        gate_logits = self.gate_head(z).masked_fill(~gate_mask, -1e9)
        mix_params = self.mix_head(z)  # (..., 3K)
        refine = F.softplus(self.refine_head(z)).view(
            *z.shape[:-1], self._interior, 2
        ) + 1.0
        value = self.value_head(z.detach()).squeeze(-1)
        return gate_logits, mix_params, refine, value

    def mixture_params(
        self, mix_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(mu, s, w), each (..., K): component locations on the anchor
        index axis, per-component scales, and the FLOORED mixture weights.
        The canonical consumer is `_anchor_dist`; exposed for training
        logs (component diagnostics, Hw) and the UI `mixture` payload."""
        k = self._mixture_k
        c = (self._anchor_count - 1) / 2.0
        mu = c + (c + 2.0) * torch.tanh(mix_params[..., 0:k])
        s = self._size_floor + self._size_span * torch.sigmoid(
            mix_params[..., k:2 * k]
        )
        w = F.softmax(mix_params[..., 2 * k:3 * k], dim=-1)
        if self._mix_floor > 0.0:
            w = self._mix_floor + (1.0 - k * self._mix_floor) * w
        return mu, s, w

    def _anchor_dist(self, mix_params: torch.Tensor, grid):
        mu, s, w = self.mixture_params(mix_params)
        legal = grid.legal[..., None, :].expand(
            *mu.shape, grid.legal.shape[-1]
        )
        probs_k = _discretized_logistic_probs(mu, s, legal)  # (..., K, count)
        probs = (w[..., None] * probs_k).sum(-2)
        # Each component sums to 1 over the legal set and the floored
        # weights sum to 1; the renorm is numerical hygiene only.
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-12)
        return torch.distributions.Categorical(probs=probs)


def obs_adapter(model: nn.Module):
    """Return a numpy function mapping freshly-encoded (..., OBS_DIM)
    observations to the model's expected input width.

    Generations serve side by side: v1-era checkpoints (OBS_DIM_V1 = 959,
    pre-pot-fraction-history) get the exact `downgrade_obs_to_v1`
    projection (its index map only touches dims < 991, so it also drops
    every later tail); models whose first layer already takes the current
    OBS_DIM get identity. Everything from OBS_DIM_V2 (991) onward is a PURE
    TAIL APPEND (obs-v2 tail 991..1020, v7 batch-2 tail 1020..1171, …), so
    a checkpoint trained at any intermediate width W in [991, OBS_DIM) is
    an exact prefix slice `obs[..., :W]` — this covers 991 (v2/v4), 1020
    (v5/v6), and any future intermediate without a new named downgrade.

    Minimal-obs ablation models (OBS_DIM_MINIMAL = 796) are NOT a prefix —
    they gather non-contiguous table-visible dims via `project_obs_minimal`.
    """
    import numpy as np

    from plo5bp.encoding import (
        OBS_DIM,
        OBS_DIM_MINIMAL,
        OBS_DIM_V1,
        OBS_DIM_V2,
        downgrade_obs_to_v1,
        project_obs_minimal,
    )

    first = model.torso[0]
    lin = first[0] if isinstance(first, nn.Sequential) else first
    w = int(lin.in_features)
    if w == OBS_DIM_MINIMAL:
        return project_obs_minimal
    if w == OBS_DIM_V1:
        return downgrade_obs_to_v1
    if OBS_DIM_V2 <= w < OBS_DIM:
        return lambda obs: np.ascontiguousarray(obs[..., :w])
    return lambda obs: obs


def model_class_for_state_dict(state_dict: dict) -> type:
    """Sniff a checkpoint's actor class from its head parameters:
    'mix_head.weight' → ActorCriticV5 (mixture of discretized logistics),
    'size_head.weight' → ActorCriticV4 (ordinal logistic), 'anchor_head.weight'
    → ActorCriticV2, 'raise_head.weight' → v1. Shared by the UI server and the
    eval/exploit/bankroll loaders so every consumer serves all generations."""
    if "mix_head.weight" in state_dict:
        return ActorCriticV5
    if "size_head.weight" in state_dict:
        return ActorCriticV4
    if "anchor_head.weight" in state_dict:
        return ActorCriticV2
    if "raise_head.weight" in state_dict:
        return ActorCritic
    raise ValueError(
        "state_dict has none of 'mix_head.weight' (v5), 'size_head.weight' "
        "(v4), 'anchor_head.weight' (v2), or 'raise_head.weight' (v1) — not "
        "a plo5bp actor checkpoint"
    )


def state_dict_obs_dim(state_dict: dict) -> int:
    """Input width the checkpoint was trained at — 959 (v1-era), 991
    (v2/v4), 1020 (v5/v6 obs-v2 tail), 1171 (v7 batch-2 tail), …. All
    widths >= 991 are pure prefixes of the current OBS_DIM; obs_adapter
    slices accordingly."""
    w = state_dict.get("torso.0.weight")
    if w is None:
        w = state_dict["torso.0.0.weight"]
    return int(w.shape[1])


def state_dict_anchor_count(state_dict: dict) -> int | None:
    """Anchor-ladder length the checkpoint was trained with, sniffed
    from the refine head (`interior * 2` rows → count = rows/2 + 2 —
    atoms are always first+last in every spec). Present on v2/v4
    checkpoints; None for v1 (no anchor ladder)."""
    w = state_dict.get("refine_head.weight")
    if w is None:
        return None
    return int(w.shape[0]) // 2 + 2


def _torso_has_norm(state_dict: dict) -> bool:
    """Detect a LayerNorm'd residual torso (v6 plasticity stems): the blocks
    carry `*.norm.weight` keys only when built with torso_layernorm=True. Lets
    the pool-snapshot rebuild + UI reconstruct the architecture without a saved
    flag."""
    return any(".norm.weight" in k for k in state_dict)


def build_actor_from_state_dict(
    state_dict: dict, hidden_dim: int, num_layers: int
) -> nn.Module:
    """Build the actor a checkpoint was saved from: sniffs the head
    class, the trained obs width (constructing at the current OBS_DIM
    default would shape-fail on 959-era checkpoints), and the anchor
    spec (PLO 11 / NLH 12), then loads the weights. Pair with
    `obs_adapter` at inference time."""
    cls = model_class_for_state_dict(state_dict)
    kwargs = dict(
        hidden_dim=hidden_dim,
        obs_dim=state_dict_obs_dim(state_dict),
        num_layers=num_layers,
        torso_layernorm=_torso_has_norm(state_dict),
    )
    count = state_dict_anchor_count(state_dict)
    if count is not None and cls is not ActorCritic:
        kwargs["anchor_spec"] = anchor_spec_for_count(count)
    if cls is ActorCriticV5:
        # mix_head is (3K, hidden): K (mu, s) pairs + K mixture logits.
        kwargs["mixture_k"] = int(state_dict["mix_head.weight"].shape[0]) // 3
    model = cls(**kwargs)
    model.load_state_dict(state_dict)
    return model


def opp_holes_multihot(holes: torch.Tensor) -> torch.Tensor:
    """Expand compact (B, 5, hole_w) uint8/int hole-card indices (255 =
    empty slot) into the (B, 260) multi-hot the CentralCritic consumes."""
    b = holes.shape[0]
    holes_l = holes.long()
    valid = holes_l < 52
    out = torch.zeros(b, 5, 52, dtype=torch.float32, device=holes.device)
    out.scatter_(2, holes_l.clamp(max=51), valid.float())
    return out.view(b, 5 * 52)


class CentralCritic(nn.Module):
    """Training-only value network (centralized training, decentralized
    execution): conditions on the actor's observation PLUS every
    opponent's hole cards, removing hidden-card luck from the grading
    signal. Never used at serve time — the actor's value head is the
    UI-facing estimator.

    Input = obs (OBS_DIM) ++ hero-rotated opponent-hole multi-hots
    (5 slots × 52; slot j = seat (actor+j+1) % num_seats, zeros beyond
    num_seats; folded opponents included)."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        opp_dim: int = 5 * 52,
        hidden_dim: int = 1536,
        num_blocks: int = 2,
        q_actions: int = 0,
        torso_layernorm: bool = False,
        value_bins: int = 0,
        value_support: float = 1500.0,
        hlgauss_sigma: float = 0.75,
        q_fold_zero: bool = False,
        q_base_raw: bool = False,
    ):
        super().__init__()
        input_block = nn.Sequential(
            nn.Linear(obs_dim + opp_dim, hidden_dim), nn.ReLU()
        )
        blocks = [
            _ResidualBlock(hidden_dim, use_norm=torso_layernorm)
            for _ in range(num_blocks)
        ]
        self.torso = nn.Sequential(input_block, *blocks)
        # Value head: scalar (default) OR a distributional HL-Gauss categorical
        # head over a SYMLOG-transformed support (V6 internals). Symlog packs the
        # huge double-board reward range (tiny 20bb pots to ~1500bb six-way
        # all-ins) into a fixed grid with fine resolution near 0 — the dynamic-
        # range fix. V = symexp(E[bins]) is still a plain scalar, so the dueling
        # Q and every existing caller are unchanged. Support/edges are buffers
        # (persisted) so the pool-rebuild + UI reconstruct them from the ckpt.
        self.value_bins = int(value_bins)
        if self.value_bins > 0:
            self.value_head = nn.Linear(hidden_dim, self.value_bins)
            hi = float(_symlog(torch.tensor(float(value_support))))
            centers = torch.linspace(-hi, hi, self.value_bins)
            step = float(centers[1] - centers[0]) if self.value_bins > 1 else 1.0
            edges = torch.cat([
                centers[:1] - 0.5 * step,
                0.5 * (centers[:-1] + centers[1:]),
                centers[-1:] + 0.5 * step,
            ])
            self.register_buffer("_value_centers", centers)
            self.register_buffer("_value_edges", edges)
            # Raw-space bin centers for the q_base_raw dueling base
            # (v7 WS1.2). Derived from _value_centers, so NOT persisted —
            # state dicts stay interchangeable with pre-flag checkpoints
            # in both directions.
            self.register_buffer(
                "_raw_value_centers", _symexp(centers), persistent=False
            )
            self.hlgauss_sigma = float(hlgauss_sigma) * step
        else:
            self.value_head = nn.Linear(hidden_dim, 1)
        # Optional dueling Q head (v5 stems): Q(s, a) = V(s).detach() +
        # A(s, a), zero-init so Q == V from step 0. Trained as an
        # AUXILIARY regression (`q_aux_coef` in ppo.py); GAE advantages
        # keep coming from V until the Expected-SARSA estimator lands
        # (VRPO, V5_DESIGN.md W2.5). Built into the checkpoint from day
        # one precisely so flipping the estimator later is a code change,
        # not a checkpoint break. Action index = 0 Fold, 1 CheckCall,
        # 2+k Raise@anchor_k → q_actions = 2 + anchor_count (13 for PLO,
        # 14 for NLH). 0 = no head (pre-v5 checkpoints load unchanged).
        self.q_actions = int(q_actions)
        if self.q_actions > 0:
            self.adv_head = nn.Linear(hidden_dim, self.q_actions)
            with torch.no_grad():
                self.adv_head.weight.zero_()
                self.adv_head.bias.zero_()
        # v7 Q-surface semantics flags (V7_DESIGN.md WS1.1/WS1.2). Neither
        # leaves a trace in the state dict — like the mixture ε floor, the
        # SERVING/TRAINING construction must match; train.py stamps them
        # into the checkpoint config and refuses warm-starts across a flip.
        self.q_fold_zero = bool(q_fold_zero)
        self.q_base_raw = bool(q_base_raw)
        if self.q_base_raw and self.value_bins <= 0:
            raise ValueError(
                "q_base_raw needs the distributional value head "
                "(value_bins > 0) — the raw base is a readout of its bins"
            )

    def _value_from_z(self, z: torch.Tensor) -> torch.Tensor:
        """Scalar V from torso features: squeeze for the scalar head, or
        symexp(E[bins]) for the distributional head. Every caller (forward,
        q_values, rollout, UI) sees a plain scalar either way."""
        if self.value_bins > 0:
            probs = F.softmax(self.value_head(z), dim=-1)
            return _symexp((probs * self._value_centers).sum(-1))
        return self.value_head(z).squeeze(-1)

    def _q_from_z(
        self,
        z: torch.Tensor,
        v: torch.Tensor,
        probs: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Dueling compose Q = base.detach() + A(s, a), with the v7 WS1
        semantics flags applied.

        Legacy base = the display V (symexp of the symlog-space bin mean) —
        which straddles value spaces against the RAW-return q targets; that
        Jensen-type gap is what the adv rows absorbed as the July family
        offsets. `q_base_raw` swaps in Σ p_i·symexp(c_i): the raw-space mean
        of the SAME categorical, so base and target share units (residual:
        the HL-Gauss label smear symexps to a small width-scaled skew — the
        qF/qT canaries measure what's left). `q_fold_zero` then pins the
        fold column to its known truth (identity, not estimate); the fold
        params get zero gradient and stay at zero-init."""
        if self.q_actions <= 0:
            return None
        if self.q_base_raw:
            base = (probs * self._raw_value_centers).sum(-1)
        else:
            base = v
        q = base.detach()[..., None] + self.adv_head(z)
        if self.q_fold_zero:
            q = torch.cat([torch.zeros_like(q[..., :1]), q[..., 1:]], dim=-1)
        return q

    def forward(
        self, obs: torch.Tensor, opp_multihot: torch.Tensor
    ) -> torch.Tensor:
        z = _maybe_checkpoint(self.torso, torch.cat([obs, opp_multihot], dim=-1), getattr(self, "_grad_checkpoint", False))
        return self._value_from_z(z)

    def q_values(
        self, obs: torch.Tensor, opp_multihot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(V, Q): V (B,) exactly as forward(); Q (B, q_actions) =
        V.detach() + A(s, a). The detach keeps the aux Q regression from
        double-driving V through the sum (V already has the value loss); the
        trunk still receives the aux gradient through A. Q stays SCALAR under
        the distributional head (uses the symexp-mean V)."""
        z = _maybe_checkpoint(self.torso, torch.cat([obs, opp_multihot], dim=-1), getattr(self, "_grad_checkpoint", False))
        if self.value_bins > 0:
            probs = F.softmax(self.value_head(z), dim=-1)
            v = _symexp((probs * self._value_centers).sum(-1))
        else:
            probs = None
            v = self.value_head(z).squeeze(-1)
        q = self._q_from_z(z, v, probs)
        return v, q

    def train_outputs(self, obs: torch.Tensor, opp_multihot: torch.Tensor):
        """One torso pass → (V_scalar, value_logits_or_None, Q_or_None) for the
        training value loss. value_logits is the raw categorical head output
        (None on the scalar head → caller uses clipped MSE); Q is the dueling
        head (None when q_actions==0)."""
        z = _maybe_checkpoint(self.torso, torch.cat([obs, opp_multihot], dim=-1), getattr(self, "_grad_checkpoint", False))
        if self.value_bins > 0:
            logits = self.value_head(z)
            probs = F.softmax(logits, dim=-1)
            v = _symexp((probs * self._value_centers).sum(-1))
        else:
            logits = None
            probs = None
            v = self.value_head(z).squeeze(-1)
        q = self._q_from_z(z, v, probs)
        return v, logits, q

    def hlgauss_value_loss(
        self, logits: torch.Tensor, returns: torch.Tensor
    ) -> torch.Tensor:
        """HL-Gauss soft-label cross-entropy (distributional head only): target
        is the Gaussian(symlog(return), σ) probability mass in each bin (CDF
        between symlog-space bin edges). fp32 throughout (value reduction)."""
        y = _symlog(returns.float())[:, None]
        inv = 1.0 / (self.hlgauss_sigma * 1.4142135623730951)
        cdf = 0.5 * (1.0 + torch.erf((self._value_edges.float()[None, :] - y) * inv))
        target = cdf[:, 1:] - cdf[:, :-1]
        target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
        return -(target * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()

    @property
    def obs_dim(self) -> int:
        return self.torso[0][0].in_features - 5 * 52


def build_critic_from_state_dict(
    state_dict: dict,
    q_fold_zero: bool = False,
    q_base_raw: bool = False,
) -> CentralCritic:
    """Build the CentralCritic a checkpoint's ``critic`` block was saved
    from: obs width, hidden width, and residual depth are sniffed from the
    state dict (constructing at the PLO OBS_DIM default would shape-fail
    on NLH's 995-wide critics), then the weights load strictly.

    ``q_fold_zero`` / ``q_base_raw`` are NOT sniffable (pure forward-path
    semantics, no parameters — same class as the mixture ε floor): callers
    that consume Q must pass the values the run trained with (stamped in
    ``ckpt["config"]``). V-only consumers (UI review EV) can ignore them —
    the V readout is identical either way."""
    w = state_dict["torso.0.0.weight"]
    hidden_dim = int(w.shape[0])
    obs_dim = int(w.shape[1]) - 5 * 52
    num_blocks = (
        len({k.split(".")[1] for k in state_dict if k.startswith("torso.")}) - 1
    )
    q_actions = 0
    if "adv_head.weight" in state_dict:
        q_actions = int(state_dict["adv_head.weight"].shape[0])
    vb = int(state_dict["value_head.weight"].shape[0])
    value_bins = vb if vb > 1 else 0  # scalar head is shape (1, hidden)
    critic = CentralCritic(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        q_actions=q_actions,
        torso_layernorm=_torso_has_norm(state_dict),
        value_bins=value_bins,
        q_fold_zero=q_fold_zero,
        q_base_raw=q_base_raw,
    )
    critic.load_state_dict(state_dict)
    return critic
