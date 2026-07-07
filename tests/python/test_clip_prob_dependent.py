"""v6 probability-dependent GATE clip (Over-mixing §6; generalized Clip-Higher).

The clip band is set by a target ABSOLUTE probability-movement room R(p), a
symmetric U in the gate's old prob p:

    R(p) = room_ext − (room_ext − room_mid)·4p(1−p),

and the per-sample ratio band is [1 − R/p, 1 + R/p]. Pins:
  - default OFF (byte-identical to the flat cfg.clip path);
  - the U hits its design targets — ~10 points of room at the extremes, ~5 at
    the middle — so a suppressed 1% gate gets ~50× the flat-clip room and can
    fall to 0, while a 50/50 gate gets the tight ±5-point band;
  - the p-floor caps the max ratio (no blow-up as p→0);
  - a PPO update stays finite with the clip on, AND with the WHOLE v6 kit on
    together (vrpo + distributional HL-Gauss value head + torso LayerNorm +
    l2-init + AGC + grad-checkpoint + this clip) — the combined-stack smoke.
"""

from __future__ import annotations

import numpy as np
import torch

from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.network import ActorCriticV2, ActorCriticV5, CentralCritic
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import collect_rollout_batched
from plo5bp.selfplay import OpponentPool

Q_ACTIONS = 2 + 11


def _trainer(**cfg_kw) -> PPOTrainer:
    return PPOTrainer(ActorCriticV2(hidden_dim=32), TrainingConfig(**cfg_kw))


def test_default_off() -> None:
    assert TrainingConfig().clip_prob_dependent is False
    assert _trainer()._clip_prob_dependent is False


def test_u_shape_hits_design_targets() -> None:
    t = _trainer(
        clip_prob_dependent=True, clip_room_ext=0.10, clip_room_mid=0.05
    )
    p = torch.tensor([0.01, 0.5, 0.99])
    lo, hi = t._gate_clip_bounds(torch.log(p))

    # Middle (p=0.5): room_mid=0.05 → band exactly [0.9, 1.1].
    assert abs(float(hi[1]) - 1.1) < 1e-5
    assert abs(float(lo[1]) - 0.9) < 1e-5

    # Extremes: ~room_ext of absolute movement in the consequential direction.
    up_room_lo = (float(hi[0]) - 1.0) * 0.01   # 1% gate climbing
    down_room_hi = (1.0 - float(lo[2])) * 0.99  # 99% gate falling
    assert abs(up_room_lo - 0.098) < 1e-3
    assert abs(down_room_hi - 0.098) < 1e-3

    # Symmetric U: extremes get more room than the middle.
    assert up_room_lo > (float(hi[1]) - 1.0) * 0.5 + 1e-6

    # A suppressed 1% gate may be driven all the way to 0 (lower bound clamps).
    assert float(lo[0]) == 0.0

    # ...and it gets far more upward room than the flat clip would (0.2*0.01).
    assert (float(hi[0]) - 1.0) * 0.01 > 40 * (0.2 * 0.01)


def test_prob_floor_caps_max_ratio() -> None:
    # As p→0 the ratio would blow up without the floor; with floor 1e-3 the max
    # ratio is ~1 + room_ext/floor = ~101, not infinite.
    t = _trainer(
        clip_prob_dependent=True,
        clip_room_ext=0.10,
        clip_room_mid=0.05,
        clip_prob_floor=1e-3,
    )
    _lo, hi = t._gate_clip_bounds(torch.log(torch.tensor([1e-9])))
    assert 90.0 < float(hi[0]) < 102.0


def test_clip_ppo_update_finite() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    game_cfg = GameConfig(num_seats=4)
    train_cfg = TrainingConfig(
        num_envs=8,
        rollout_length=128,
        hidden_dim=32,
        batch_size=64,
        critic_hidden_dim=64,
        critic_num_blocks=1,
        clip_prob_dependent=True,
    )
    model = ActorCriticV2(hidden_dim=32)
    critic = CentralCritic(hidden_dim=64, num_blocks=1)
    trainer = PPOTrainer(model, train_cfg, critic=critic)
    rng = np.random.default_rng(0)
    pool = OpponentPool(capacity=1)
    model.eval()
    batch = collect_rollout_batched(
        model, pool, game_cfg, train_cfg, rng, critic=critic
    )
    model.train()
    stats = trainer.update(batch, rng)
    assert np.isfinite(stats.policy_loss) and np.isfinite(stats.entropy)


def test_v6_full_stack_update_finite() -> None:
    # The entire v6 kit on together: vrpo advantage + distributional HL-Gauss
    # value head + torso LayerNorm + l2-init + AGC + grad-checkpoint + the
    # probability-dependent gate clip. Pins that they compose end-to-end (the
    # distributional-head × VRPO combination in particular was untested together)
    # and the update is finite.
    torch.manual_seed(0)
    np.random.seed(0)
    game_cfg = GameConfig(num_seats=4)
    train_cfg = TrainingConfig(
        num_envs=8,
        rollout_length=128,
        hidden_dim=32,
        num_layers=3,
        critic_hidden_dim=64,
        critic_num_blocks=1,
        batch_size=64,
        advantage_estimator="vrpo",
        q_aux_coef=0.5,
        torso_layernorm=True,
        l2_init_coef=1e-4,
        agc_clip=0.1,
        grad_checkpoint=True,
        value_bins=51,
        value_support=1500.0,
        clip_prob_dependent=True,
    )
    model = ActorCriticV5(hidden_dim=32, num_layers=3, torso_layernorm=True)
    critic = CentralCritic(
        hidden_dim=64,
        num_blocks=1,
        q_actions=Q_ACTIONS,
        torso_layernorm=True,
        value_bins=51,
    )
    # Perturb the dueling Q head so VRPO genuinely diverges from GAE (exercise
    # the real Expected-SARSA path, not the zero-init Q≡V reduction).
    with torch.no_grad():
        critic.adv_head.weight.normal_(0.0, 0.2)
    trainer = PPOTrainer(model, train_cfg, critic=critic)
    rng = np.random.default_rng(0)
    pool = OpponentPool(capacity=1)
    model.eval()
    batch = collect_rollout_batched(
        model, pool, game_cfg, train_cfg, rng, critic=critic
    )
    model.train()
    stats = trainer.update(batch, rng)
    for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "q_loss"):
        assert np.isfinite(getattr(stats, name)), f"non-finite {name}: {stats}"
