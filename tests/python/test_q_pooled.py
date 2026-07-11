"""Pooled dueling head (--q-pooled) + dense fold supervision (2026-07-09
Q-head audit revision). Pins:

- CentralCritic(q_actions=3) shapes and the zero-init Q == V identity.
- build_critic_from_state_dict round-trips the pooled width.
- PPOTrainer._q_index: width-keyed mapping (13-col legacy vs 3-col pooled).
- _q_fold_sup_term: perfect-label MSE on fold-LEGAL rows only, fp32.
- The rollout marginal compression identity: [m0, m1, sum(m2:)] == gate probs.
"""

import numpy as np
import pytest
import torch

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.network import CentralCritic, build_critic_from_state_dict

OBS = 64
HID = 32


def _mini_critic(q_actions: int) -> CentralCritic:
    return CentralCritic(
        obs_dim=OBS, hidden_dim=HID, num_blocks=1,
        q_actions=q_actions, value_bins=11,
    )


def test_pooled_critic_shapes_and_zero_init_q_equals_v():
    critic = _mini_critic(3)
    obs = torch.randn(5, OBS)
    opp = torch.zeros(5, 260)
    v, q = critic.q_values(obs, opp)
    assert q.shape == (5, 3)
    # zero-init adv head -> Q == V in every column
    assert torch.allclose(q, v[:, None].expand_as(q))


def test_builder_roundtrips_pooled_width():
    critic = _mini_critic(3)
    with torch.no_grad():
        critic.adv_head.weight.normal_()
    rebuilt = build_critic_from_state_dict(critic.state_dict())
    assert rebuilt.q_actions == 3
    obs = torch.randn(3, OBS)
    opp = torch.zeros(3, 260)
    _, q_a = critic.q_values(obs, opp)
    _, q_b = rebuilt.q_values(obs, opp)
    assert torch.allclose(q_a, q_b)


class _MB:
    def __init__(self, gates, anchors, masks):
        self.gate_actions = gates
        self.anchor_actions = anchors
        self.gate_masks = masks


def _trainer_stub(q_fold_sup: float):
    """A bare object carrying just what the two helpers read."""
    from plo5bp.ppo import PPOTrainer

    t = object.__new__(PPOTrainer)
    t._q_fold_sup = q_fold_sup
    return t


def test_q_index_width_keyed():
    t = _trainer_stub(0.0)
    gates = torch.tensor([GATE_FOLD, GATE_CHECK_CALL, GATE_RAISE, GATE_RAISE])
    anchors = torch.tensor([0, 0, 4, 10])
    mb = _MB(gates, anchors, None)
    q13 = torch.zeros(4, 13)
    q3 = torch.zeros(4, 3)
    assert t._q_index(mb, q13).tolist() == [0, 1, 6, 12]
    assert t._q_index(mb, q3).tolist() == [0, 1, 2, 2]


def test_fold_supervision_masks_illegal_rows():
    t = _trainer_stub(1.0)
    masks = torch.tensor([
        [True, True, True],    # fold legal
        [False, True, True],   # first-to-act: fold illegal -> excluded
        [True, True, False],
    ])
    q_all = torch.tensor([
        [2.0, 5.0, 5.0],
        [9.0, 5.0, 5.0],       # fold col 9 must NOT contribute
        [-1.0, 5.0, 5.0],
    ])
    mb = _MB(None, None, masks)
    base = torch.tensor(0.0)
    out = t._q_fold_sup_term(mb, q_all, base)
    # mean over legal rows of q_fold^2 = (4 + 1) / 2
    assert out.dtype == torch.float32
    assert torch.isclose(out, torch.tensor(2.5))
    t0 = _trainer_stub(0.0)
    assert t0._q_fold_sup_term(mb, q_all, base) is base


def test_q_fold_err_canary_field():
    # PPOStats carries the fold-column canary (mean Q[FOLD] over
    # fold-legal rows; ground truth 0); defaults 0.0 when no dueling head.
    from plo5bp.ppo import PPOStats

    s = PPOStats(policy_loss=0.0, value_loss=0.0, entropy=0.0, approx_kl=0.0)
    assert s.q_fold_err == 0.0


def test_marginal_compression_identity():
    # act()'s 13-way marginal is [p_f, p_c, p_raise * pi(anchor_k)...];
    # the pooled compression must recover the gate probs exactly.
    rng = np.random.default_rng(7)
    gate = rng.dirichlet(np.ones(3), size=64).astype(np.float32)
    anch = rng.dirichlet(np.ones(11), size=64).astype(np.float32)
    marg13 = np.concatenate([gate[:, :2], gate[:, 2:3] * anch], axis=-1)
    marg3 = np.stack(
        [marg13[:, 0], marg13[:, 1], marg13[:, 2:].sum(-1)], axis=-1
    )
    np.testing.assert_allclose(marg3, gate, rtol=0, atol=1e-6)
    # and vpi under a pooled-constant q row matches the 13-col computation
    q3 = rng.normal(size=(64, 3)).astype(np.float32)
    q13 = np.concatenate([q3[:, :2], np.repeat(q3[:, 2:3], 11, axis=1)], axis=-1)
    np.testing.assert_allclose(
        (marg3 * q3).sum(-1), (marg13 * q13).sum(-1), rtol=1e-5, atol=1e-5
    )
