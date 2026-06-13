"""Configuration dataclasses for the game and training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    """Static hand configuration. Chip unit: 1 bb = 10000 chips (cent precision at $20/bb).

    Heterogeneous stacks: pass ``starting_stacks=(s0, s1, ...)`` with
    ``len == num_seats``. Otherwise ``starting_stack`` expands uniformly.
    """

    num_seats: int = 6
    starting_stack: int = 200000
    ante: int = 30000
    bb: int = 10000
    starting_stacks: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.starting_stacks is not None and len(self.starting_stacks) != self.num_seats:
            raise ValueError(
                f"starting_stacks length {len(self.starting_stacks)} != num_seats {self.num_seats}"
            )

    @property
    def resolved_stacks(self) -> tuple[int, ...]:
        if self.starting_stacks is not None:
            return self.starting_stacks
        return (self.starting_stack,) * self.num_seats


@dataclass(frozen=True)
class TrainingConfig:
    """PPO training hyperparameters."""

    lr: float = 3e-4
    clip: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    rollout_length: int = 2048
    num_envs: int = 32
    ppo_epochs: int = 4
    batch_size: int = 256
    entropy_coef: float = 0.1
    value_clip: float = 0.2
    hidden_dim: int = 128
    num_layers: int = 2
    num_updates: int = 1000
    opponent_pool_size: int = 16
    snapshot_every: int = 50
    seed: int = 0

    ev_runout_samples: int = 0
    pool_mix_prob: float = 0.5
    pool_opp_seats: int = 2

    # Pot-fraction aggression bonus (bb units). When > 0, each
    # GATE_RAISE step (covering both normal raises and short shoves
    # encoded at u=1) gets an additional
    # `c * min(1.0, aggressive_chips / pre_step_pot)` added to the
    # forward-EV per-step reward. Default 0.0 disables the bonus.
    aggression_bonus_c: float = 0.0

    # Retroactive aggression bonus (bb units). When > 0, at end-of-hand
    # each learner-seat trajectory receives a flat bonus on qualifying
    # steps based on hero's pot share:
    #   share > 50%  → bonus on GATE_RAISE steps only
    #   share == 50% → bonus on GATE_RAISE + GATE_CHECK_CALL(chips > 0)
    #   share < 50%  → no bonus
    # Folds and pure checks never get bonus. Independent of
    # `aggression_bonus_c`; both can be set but typical use is one or
    # the other.
    retroactive_bonus_c: float = 0.0

    # v2 (anchor head + centralized critic) hyperparameters. Ignored on
    # v1 runs — the critic is only built when train.py constructs one.
    critic_hidden_dim: int = 1536
    critic_num_blocks: int = 2
    # Weight on the actor's own value head ("display head" for the UI)
    # when a CentralCritic owns the GAE values. Plain regression, no
    # clipping; small so it stays subordinate to the policy loss.
    display_value_coef: float = 0.125
    # KL-to-EMA-reference regularizer. 0.0 = off (no EMA model built).
    # The reference re-initializes to current weights on every (re)start
    # — it is NOT persisted in checkpoints.
    kl_anchor_coef: float = 0.0
    kl_anchor_ema: float = 0.999

    # SOFT KL guard (early-stop): when a minibatch's |approx_kl| exceeds
    # this, break the PPO inner loop but KEEP the minibatches already
    # applied this update (the tripping minibatch's step is not applied).
    # Standard PPO early-stopping — bounds per-update drift without
    # discarding progress. 0.0 = off. NOTE: this used to do a full
    # rollback at this threshold, which froze a run (317/318 updates
    # rolled back to no-ops, 2026-06-12) once the policy sharpened enough
    # to cross it every update. Rollback now lives at kl_hard below.
    target_kl: float = 2.0

    # HARD KL guard (full rollback): when a minibatch's |approx_kl|
    # exceeds this, restore params + optimizer moments captured at the
    # top of update() — the ENTIRE update is discarded. Reserved for
    # genuine catastrophe (vTwo2 hit approx_kl ≈ +2417 at update 173 and
    # collapsed entropy to 0). Should be >= target_kl. 0.0 = no hard
    # rollback (soft early-stop still applies).
    kl_hard: float = 10.0

    # Clamp normalized advantages to ±this many σ before the PPO loss.
    # 0.0 = off. PPO's clip bounds the importance RATIO, not the
    # advantage weight, so one fat-tail sample (a 1500bb six-way
    # all-in pot lands at 30σ+ after unit normalization) carries 30x
    # gradient weight — deep-stack blocks generate exactly these and
    # they drove the update-52+ violence. Applied identically in the
    # serial and batched collectors.
    adv_clip: float = 8.0

    device: str = "cpu"
