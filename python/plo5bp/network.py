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

import torch
import torch.nn as nn
import torch.nn.functional as F

from plo5bp.actions import GATE_ACTIONS, GATE_RAISE
from plo5bp.encoding import OBS_DIM


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

    def act(
        self,
        obs: torch.Tensor,
        gate_mask: torch.Tensor,
        bounds: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or argmax. Returns (gate, chips, log_prob, value).

        `bounds` is `(B, 2)` with (min_raise_chips, max_raise_chips) per
        row. `chips` is the raw chip delta the caller feeds to
        `env.step_hybrid`; zero when `gate != Raise`.
        """
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
        return gate, chips_out, log_prob, value

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
