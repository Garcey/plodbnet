"""Clipped-surrogate PPO update with value clip and entropy bonus."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import record_function

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
        self._cuda = config.device == "cuda" and torch.cuda.is_available()
        self.optimizer = optim.AdamW(
            model.parameters(), lr=config.lr, fused=self._cuda
        )
        # Compile evaluate() on CUDA when Triton is available (Linux);
        # falls back to eager on Windows where Triton typically isn't
        # installed alongside the torch+cuda wheel. dynamic=True tolerates
        # the last-minibatch shape variance in iter_minibatches.
        self._evaluate = model.evaluate
        if self._cuda:
            try:
                import triton  # noqa: F401
                self._evaluate = torch.compile(model.evaluate, dynamic=True)
            except ImportError:
                pass

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
        device = batch.obs.device
        # Accumulate as device tensors; one .item() at the end avoids
        # per-minibatch CUDA syncs that serialize against compute.
        total_policy = torch.zeros((), device=device)
        total_value = torch.zeros((), device=device)
        total_entropy = torch.zeros((), device=device)
        total_kl = torch.zeros((), device=device)
        count = 0
        with record_function("step12/inner_loop"):
            for _ in range(cfg.ppo_epochs):
                for mb in iter_minibatches(batch, cfg.batch_size, rng):
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=self._cuda,
                    ):
                        with record_function("step12a/evaluate"):
                            log_prob, entropy, value = self._evaluate(
                                mb.obs,
                                mb.gate_masks,
                                mb.raise_bounds,
                                mb.gate_actions,
                                mb.raise_chips,
                            )
                        with record_function("step12b/loss"):
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

                    with record_function("step12c/backward"):
                        self.optimizer.zero_grad()
                        loss.backward()
                    with record_function("step12d/optimizer_step"):
                        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.optimizer.step()

                    with record_function("step12e/kl"):
                        with torch.no_grad():
                            kl = (mb.log_probs - log_prob).mean()
                    total_policy += policy_loss.detach().float()
                    total_value += value_loss.detach().float()
                    total_entropy += -entropy_loss.detach().float()
                    total_kl += kl.float()
                    count += 1
        denom = max(count, 1)
        with record_function("step13/stats_sync"):
            return PPOStats(
                policy_loss=float(total_policy.item()) / denom,
                value_loss=float(total_value.item()) / denom,
                entropy=float(total_entropy.item()) / denom,
                approx_kl=float(total_kl.item()) / denom,
            )
