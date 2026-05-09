"""Clipped-surrogate PPO update with value clip and entropy bonus."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from plo5bp.config import TrainingConfig
from plo5bp.network import ActorCritic
from plo5bp.rollout import Batch, iter_minibatches
import numpy as np


@dataclass
class PPOStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float


class PPOTrainer:
    def __init__(self, model: ActorCritic, config: TrainingConfig):
        self.model = model
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=config.lr)

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
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        count = 0
        for _ in range(cfg.ppo_epochs):
            for mb in iter_minibatches(batch, cfg.batch_size, rng):
                log_prob, entropy, value = self.model.evaluate(
                    mb.obs,
                    mb.gate_masks,
                    mb.raise_bounds,
                    mb.gate_actions,
                    mb.raise_chips,
                )
                ratio = torch.exp(log_prob - mb.log_probs)
                surr1 = ratio * mb.advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * mb.advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = mb.values + torch.clamp(
                    value - mb.values, -cfg.value_clip, cfg.value_clip
                )
                v1 = (value - mb.returns).pow(2)
                v2 = (value_pred_clipped - mb.returns).pow(2)
                value_loss = 0.5 * torch.max(v1, v2).mean()

                entropy_loss = -entropy.mean()
                loss = policy_loss + 0.5 * value_loss + eff_entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                with torch.no_grad():
                    kl = (mb.log_probs - log_prob).mean().item()
                total_policy += float(policy_loss.item())
                total_value += float(value_loss.item())
                total_entropy += float(-entropy_loss.item())
                total_kl += float(kl)
                count += 1
        denom = max(count, 1)
        return PPOStats(
            policy_loss=total_policy / denom,
            value_loss=total_value / denom,
            entropy=total_entropy / denom,
            approx_kl=total_kl / denom,
        )
