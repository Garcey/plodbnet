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

from plo5bp.actions import GATE_ACTIONS, GATE_RAISE
from plo5bp.encoding import OBS_DIM
from plo5bp.sizing import ANCHOR_COUNT, anchor_grid_torch, refine_chips_torch


class ActOut(NamedTuple):
    """Uniform act() result across head versions.

    `anchor` / `refine_u` are the v2 sizing-head action components the
    rollout buffer stores so `evaluate()` can replay the exact sampled
    action (never reverse-engineered from chips). v1 fills anchor=-1,
    refine_u=the Beta u sample.
    """

    gate: torch.Tensor      # (B,) int64
    chips: torch.Tensor     # (B,) int64 chip delta; 0 for non-Raise
    log_prob: torch.Tensor  # (B,) float
    value: torch.Tensor     # (B,) float (v2: the display value head)
    anchor: torch.Tensor    # (B,) int64
    refine_u: torch.Tensor  # (B,) float — clamped sampled u


class _ResidualBlock(nn.Module):
    """y = x + ReLU(Linear(x)). Single Linear keeps post-activation output
    on the residual stream — works at depth ≥3 where the plain MLP loses
    gradient flow."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.relu(self.linear(x))


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
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if num_layers >= 3:
            input_block = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
            blocks: list[nn.Module] = [_ResidualBlock(hidden_dim) for _ in range(num_layers - 1)]
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
        z = self.torso(obs)
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

    def __init__(
        self,
        hidden_dim: int = 512,
        obs_dim: int = OBS_DIM,
        num_layers: int = 2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if num_layers >= 3:
            input_block = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
            blocks: list[nn.Module] = [
                _ResidualBlock(hidden_dim) for _ in range(num_layers - 1)
            ]
            self.torso = nn.Sequential(input_block, *blocks)
        else:
            layers: list[nn.Module] = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            self.torso = nn.Sequential(*layers)
        self.gate_head = nn.Linear(hidden_dim, GATE_ACTIONS)
        self.anchor_head = nn.Linear(hidden_dim, ANCHOR_COUNT)
        # (α, β) per interior anchor; softplus+1 keeps each Beta unimodal.
        self.refine_head = nn.Linear(hidden_dim, _INTERIOR * 2)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, obs: torch.Tensor, gate_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (masked gate logits, raw anchor logits, refine params
        (B, 9, 2), display value). Anchor legality masking happens in
        act/evaluate where the sizing context is available."""
        z = self.torso(obs)
        gate_logits = self.gate_head(z).masked_fill(~gate_mask, -1e9)
        anchor_logits = self.anchor_head(z)
        refine = F.softplus(self.refine_head(z)).view(
            *z.shape[:-1], _INTERIOR, 2
        ) + 1.0
        value = self.value_head(z).squeeze(-1)
        return gate_logits, anchor_logits, refine, value

    @staticmethod
    def _gather_refine(
        refine: torch.Tensor, anchor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row (α, β) of the chosen anchor; atoms clamp to a valid
        interior index (their Beta term is masked out downstream)."""
        idx = (anchor - 1).clamp(0, _INTERIOR - 1)
        ab = refine.gather(
            -2, idx[..., None, None].expand(*idx.shape, 1, 2)
        ).squeeze(-2)
        return ab[..., 0], ab[..., 1]

    def act(
        self,
        obs: torch.Tensor,
        gate_mask: torch.Tensor,
        sizing: torch.Tensor,
        deterministic: bool = False,
    ) -> ActOut:
        """Sample or argmax. `sizing` is (B, 4) int64
        [min_raise, max_raise, pot, to_call]."""
        gate_logits, anchor_logits, refine, value = self.forward(obs, gate_mask)
        grid = anchor_grid_torch(sizing)

        gate_dist = torch.distributions.Categorical(logits=gate_logits)
        if deterministic:
            gate = gate_logits.argmax(dim=-1)
        else:
            gate = gate_dist.sample()
        gate_log_prob = gate_dist.log_prob(gate)

        anchor_logits_m = anchor_logits.masked_fill(~grid.legal, -1e9)
        anchor_dist = torch.distributions.Categorical(logits=anchor_logits_m)
        if deterministic:
            anchor = anchor_logits_m.argmax(dim=-1)
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
        refined_chips = refine_chips_torch(anchor, u, sizing)
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
        return ActOut(
            gate=gate,
            chips=chips_out,
            log_prob=log_prob,
            value=value,
            anchor=anchor,
            refine_u=u,
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
    ]:
        """Return (log_prob, entropy, display_value, gate_H, anchor_H,
        beta_H_eff) for stored actions.

        Entropy follows the generative process:
        H(gate) + P(Raise) · (H(anchor) + Σ_k p_k · H(Beta_k) · refine_ok_k).
        Masked anchors contribute exactly zero. The decomposition terms
        are returned so training can log Hg/Ha/Hb separately (the anchor
        head adds up to log(11) ≈ 2.4 nats vs the v1 entropy scale).
        """
        gate_logits, anchor_logits, refine, value = self.forward(obs, gate_mask)
        grid = anchor_grid_torch(sizing)

        gate_dist = torch.distributions.Categorical(logits=gate_logits)
        gate_log_prob = gate_dist.log_prob(gate_actions)
        gate_entropy = gate_dist.entropy()

        anchor_logits_m = anchor_logits.masked_fill(~grid.legal, -1e9)
        anchor_dist = torch.distributions.Categorical(logits=anchor_logits_m)
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
        anchor_probs = F.softmax(anchor_logits_m, dim=-1)
        all_beta = torch.distributions.Beta(refine[..., 0], refine[..., 1])
        beta_h = all_beta.entropy()                       # (B, 9)
        interior_ok = grid.refine_ok[..., 1:ANCHOR_COUNT - 1]
        beta_h_eff = (
            anchor_probs[..., 1:ANCHOR_COUNT - 1] * beta_h * interior_ok
        ).sum(-1)
        anchor_entropy = anchor_dist.entropy()
        entropy = gate_entropy + p_raise * (anchor_entropy + beta_h_eff)
        return log_prob, entropy, value, gate_entropy, anchor_entropy, beta_h_eff


def opp_holes_multihot(holes: torch.Tensor) -> torch.Tensor:
    """Expand compact (B, 5, 5) uint8/int hole-card indices (255 =
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
    ):
        super().__init__()
        input_block = nn.Sequential(
            nn.Linear(obs_dim + opp_dim, hidden_dim), nn.ReLU()
        )
        blocks = [_ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        self.torso = nn.Sequential(input_block, *blocks)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, obs: torch.Tensor, opp_multihot: torch.Tensor
    ) -> torch.Tensor:
        z = self.torso(torch.cat([obs, opp_multihot], dim=-1))
        return self.value_head(z).squeeze(-1)
