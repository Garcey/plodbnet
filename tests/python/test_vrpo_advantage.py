"""VRPO / Expected-SARSA(λ) advantage flip (V5_DESIGN.md W2.5).

The load-bearing correctness pin is the GOLDEN PARITY: with the critic's
dueling Q head at its zero-init state (Q ≡ V for every action), the
Expected-SARSA(λ) advantage reduces *exactly* to V-based GAE(λ). That is
what makes flipping `advantage_estimator="vrpo"` a no-op on a fresh v5
checkpoint and only a divergence once the Q head is trained — i.e. the
flip is a code change, not a checkpoint break.

Also pinned:
  - the flip actually changes the advantage signal once Q ≠ V;
  - a PPO update stays finite under the vrpo estimator;
  - the serial collector rejects vrpo (batched-only);
  - `act(return_marginal=True)` returns a valid (2+anchor)-way distribution
    over the Q-head action layout, and None when not requested.

2026-07-12 — fixed-Q unit pins on `_vrpo_advantage_scan` added. The golden
parity above CANNOT discriminate the full eq-3.2 estimator
(Â = (Q−V^π) + λ-trace of δ⁺) from the residual-only λ-trace, because the
leading term is identically ~0 at Q ≡ V — which is exactly how the shipped
implementation lost the leading term without any test failing (the root
cause of the v6 lock-fold pathology: a terminal fold's advantage was
−Q[FOLD] instead of −V^π). The pins below use exact small-integer f32
arithmetic (every op exact) so the equalities are bit-exact, and each one
FAILS on the residual-only form.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from plo5bp.actions import GATE_ACTIONS
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM
from plo5bp.network import ActorCriticV2, CentralCritic
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import collect_rollout, collect_rollout_batched
from plo5bp.selfplay import OpponentPool

NUM_SEATS = 4
Q_ACTIONS = 2 + 11  # Fold, CheckCall, Raise@anchor_0..10 (PLO 11-anchor grid)


def _cfg(estimator: str) -> tuple[GameConfig, TrainingConfig]:
    game_cfg = GameConfig(num_seats=NUM_SEATS)
    train_cfg = TrainingConfig(
        num_envs=8,
        rollout_length=256,
        hidden_dim=32,
        critic_hidden_dim=64,
        critic_num_blocks=1,
        q_aux_coef=0.5,
        advantage_estimator=estimator,
    )
    return game_cfg, train_cfg


def _run(estimator: str, seed: int, critic: CentralCritic):
    """One batched rollout with a fresh model seeded by `seed`. The model
    weights AND the post-init torch-RNG state are a pure function of `seed`,
    and `return_marginal` adds no RNG draw, so two runs at the same seed +
    same critic have bit-identical trajectories — only the advantage math
    can differ."""
    game_cfg, train_cfg = _cfg(estimator)
    torch.manual_seed(seed)
    model = ActorCriticV2(hidden_dim=32)
    model.eval()
    rng = np.random.default_rng(seed)
    pool = OpponentPool(capacity=1)
    return collect_rollout_batched(
        model, pool, game_cfg, train_cfg, rng, critic=critic
    )


def test_vrpo_reduces_to_gae_at_zero_init_q() -> None:
    # Zero-init adv_head → Q ≡ V for every action, so Expected-SARSA(λ) is
    # bit-identical to GAE(λ) up to f32 reduction-order noise.
    torch.manual_seed(100)
    critic = CentralCritic(hidden_dim=64, num_blocks=1, q_actions=Q_ACTIONS).eval()
    gae = _run("gae", 7, critic)
    vrpo = _run("vrpo", 7, critic)
    assert gae.advantages.shape == vrpo.advantages.shape, (
        gae.advantages.shape, vrpo.advantages.shape
    )
    assert torch.allclose(gae.advantages, vrpo.advantages, atol=1e-4), (
        f"max |Δadv| = {float((gae.advantages - vrpo.advantages).abs().max())}"
    )
    # `returns` (the value-head target) stay GAE-based under both.
    assert torch.allclose(gae.returns, vrpo.returns, atol=1e-4)


def test_vrpo_differs_from_gae_with_trained_q() -> None:
    # Perturb the dueling advantage head so Q ≠ V (a "trained" head): the
    # flip must now move the advantage signal.
    torch.manual_seed(101)
    critic = CentralCritic(hidden_dim=64, num_blocks=1, q_actions=Q_ACTIONS)
    with torch.no_grad():
        critic.adv_head.weight.normal_(0.0, 0.5)
        critic.adv_head.bias.normal_(0.0, 0.5)
    critic.eval()
    gae = _run("gae", 9, critic)
    vrpo = _run("vrpo", 9, critic)
    assert torch.isfinite(vrpo.advantages).all()
    assert not torch.allclose(gae.advantages, vrpo.advantages, atol=1e-3), (
        "vrpo advantages coincide with GAE even though Q != V"
    )


def test_vrpo_ppo_update_finite() -> None:
    torch.manual_seed(102)
    np.random.seed(102)
    game_cfg, train_cfg = _cfg("vrpo")
    train_cfg = dataclasses.replace(train_cfg, batch_size=64)
    model = ActorCriticV2(hidden_dim=32)
    critic = CentralCritic(hidden_dim=64, num_blocks=1, q_actions=Q_ACTIONS)
    with torch.no_grad():
        critic.adv_head.weight.normal_(0.0, 0.3)
    trainer = PPOTrainer(model, train_cfg, critic=critic)
    rng = np.random.default_rng(102)
    pool = OpponentPool(capacity=1)
    model.eval()
    batch = collect_rollout_batched(
        model, pool, game_cfg, train_cfg, rng, critic=critic
    )
    model.train()
    stats = trainer.update(batch, rng)
    for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "q_loss"):
        assert np.isfinite(getattr(stats, name)), f"non-finite {name}: {stats}"


def test_serial_collector_rejects_vrpo() -> None:
    game_cfg, train_cfg = _cfg("vrpo")
    torch.manual_seed(1)
    model = ActorCriticV2(hidden_dim=32).eval()
    critic = CentralCritic(hidden_dim=64, num_blocks=1, q_actions=Q_ACTIONS).eval()
    rng = np.random.default_rng(1)
    pool = OpponentPool(capacity=1)
    with pytest.raises(NotImplementedError):
        collect_rollout(model, pool, game_cfg, train_cfg, rng, critic=critic)


def test_action_marginal_is_valid_distribution() -> None:
    torch.manual_seed(3)
    model = ActorCriticV2(hidden_dim=32).eval()
    B = 16
    obs = torch.randn(B, OBS_DIM)
    gate_mask = torch.ones(B, GATE_ACTIONS, dtype=torch.bool)
    sizing = (
        torch.tensor([[200, 10000, 5000, 0]], dtype=torch.int64)
        .expand(B, 4)
        .contiguous()
    )
    out = model.act(obs, gate_mask, sizing, return_marginal=True)
    assert out.action_marginal is not None
    assert out.action_marginal.shape == (B, Q_ACTIONS)
    assert (out.action_marginal >= 0).all()
    assert torch.allclose(
        out.action_marginal.sum(-1), torch.ones(B), atol=1e-5
    )
    # Off by default → free when the estimator is GAE.
    assert model.act(obs, gate_mask, sizing).action_marginal is None


# ---------------------------------------------------------------------------
# Fixed-Q pins on the scan itself (discriminate eq 3.2 from residual-only).
# All arrays are small-integer-valued f32 with gamma=1 (and deltas that are
# exactly zero where zero is intended), so float ops are exact and asserts
# can be np.array_equal — no tolerance to hide a missing term behind.
# ---------------------------------------------------------------------------

from plo5bp.rollout import _vrpo_advantage_scan  # noqa: E402

F32 = np.float32
ONE = np.float32(1.0)


def test_scan_bellman_consistent_q_yields_q_minus_vpi() -> None:
    # Rewards chosen so every Expected-SARSA residual is EXACTLY zero
    # (r_t = Q_t − γ·V^π_{t+1}, terminal reward = Q_last). The trace term
    # then vanishes and eq 3.2 says Â ≡ Q − V^π. The residual-only form
    # returns all-zeros here — maximal discrimination.
    q = np.array([[[5, 2, 9], [1, 6, 3]]], dtype=F32)      # (T=1, S=2, L=3)
    vpi = np.array([[[3, 1, 4], [2, 4, 1]]], dtype=F32)    # junk at s0,t2 (inactive)
    last_t = np.array([[1, 2]], dtype=np.int32)            # lengths 2 and 3
    flush = np.array([[True, True]])
    costs = np.array(
        [
            [
                [5 - 1, 0, 0],          # s0 t0: q00 − vpi01; t1 terminal via won
                [1 - 4, 6 - 1, 0],      # s1 t0, t1; t2 terminal via won
            ]
        ],
        dtype=F32,
    )
    won = np.array([[2, 3]], dtype=F32)                    # = q at each seat's last step
    out = _vrpo_advantage_scan(costs, won, q, vpi, last_t, flush, ONE, F32(0.95))
    expected = np.array(
        [
            [
                [5 - 3, 2 - 1, 0],      # Q − V^π at active slots; inactive → 0
                [1 - 2, 6 - 4, 3 - 1],
            ]
        ],
        dtype=F32,
    )
    assert np.array_equal(out, expected), (out, expected)


def test_scan_terminal_fold_advantage_is_minus_vpi_independent_of_qfold() -> None:
    # THE bug signature. A terminal fold has reward exactly 0, so eq 3.2
    # gives Â = (Q_F − V^π) + (0 − Q_F) = −V^π: the Q_F terms CANCEL. The
    # residual-only form gives −Q_F instead — 0 once fold supervision pins
    # the column (no anti-fold pressure at high-V states) and a SUBSIDY
    # when the column drifts negative. Assert the fixed scan returns −V^π
    # for two different Q_F values, bit-identically.
    vpi = np.array([[[6.0]]], dtype=F32)
    last_t = np.array([[0]], dtype=np.int32)
    flush = np.array([[True]])
    costs = np.zeros((1, 1, 1), dtype=F32)
    won = np.zeros((1, 1), dtype=F32)
    outs = []
    for q_fold in (7.0, -5.0):
        q = np.array([[[q_fold]]], dtype=F32)
        out = _vrpo_advantage_scan(
            costs, won, q, vpi, last_t, flush, ONE, F32(0.9)
        )
        assert np.array_equal(out, np.array([[[-6.0]]], dtype=F32)), (
            f"terminal-fold advantage must be −V^π; got {out} at Q_F={q_fold}"
        )
        outs.append(out)
    assert np.array_equal(outs[0], outs[1]), "Â_fold must not depend on Q_F"


def test_scan_zero_init_equals_gae_exactly() -> None:
    # Q ≡ V^π ≡ V (integer-valued) ⇒ leading term is exactly 0 and δ⁺ is
    # the plain GAE residual, so the scan must reproduce hand-rolled
    # GAE(γ=1, λ=1) bit-exactly: rewards [−1, −2, +5] on values [4, 2, 1]
    # → deltas [−3, −3, +4] → advantages [−2, +1, +4].
    v = np.array([[[4, 2, 1]]], dtype=F32)
    last_t = np.array([[2]], dtype=np.int32)
    flush = np.array([[True]])
    costs = np.array([[[-1, -2, 0]]], dtype=F32)
    won = np.array([[5]], dtype=F32)
    out = _vrpo_advantage_scan(costs, won, v, v, last_t, flush, ONE, ONE)
    assert np.array_equal(out, np.array([[[-2, 1, 4]]], dtype=F32)), out


def test_scan_masking_flushmask_and_beyond_length_are_zero() -> None:
    # flush_mask=False seats and beyond-length slots must come back 0 even
    # when q/vpi hold junk there (production arrays are zero-filled, but
    # the function's contract shouldn't depend on it).
    q = np.array([[[9, 8, 7], [1, 2, 3]]], dtype=F32)
    vpi = np.array([[[4, 4, 4], [1, 1, 1]]], dtype=F32)
    last_t = np.array([[0, 2]], dtype=np.int32)
    flush = np.array([[True, False]])
    costs = np.zeros((1, 2, 3), dtype=F32)
    won = np.array([[9, 0]], dtype=F32)                     # reward at s0 last = 9
    out = _vrpo_advantage_scan(costs, won, q, vpi, last_t, flush, ONE, ONE)
    # s0: active t0 only → (9−4) + (9 + 0 − 9) = 5; t1/t2 inactive → 0.
    assert np.array_equal(
        out, np.array([[[5, 0, 0], [0, 0, 0]]], dtype=F32)
    ), out
