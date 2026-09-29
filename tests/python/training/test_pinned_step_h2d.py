"""Value parity for _PinnedStepH2D (per-step pinned H2D staging)."""

from __future__ import annotations

import numpy as np
import torch

from plo5bp.actions import GATE_ACTIONS
from plo5bp.rollout import _PinnedStepH2D


def _ref_upload(device, b_obs, b_gm, b_sizing):
    return (
        torch.from_numpy(np.ascontiguousarray(b_obs)).to(device),
        torch.from_numpy(np.ascontiguousarray(b_gm)).to(device),
        torch.from_numpy(np.ascontiguousarray(b_sizing)).to(device),
    )


def test_pinned_step_h2d_cpu_matches_from_numpy():
    """CPU path disables pin; must match from_numpy.to."""
    device = torch.device("cpu")
    obs_dim = 64
    k = 7
    st = _PinnedStepH2D(capacity=16, obs_dim=obs_dim, device=device)
    assert st.enabled is False
    rng = np.random.default_rng(0)
    b_obs = rng.standard_normal((k, obs_dim)).astype(np.float32)
    b_gm = rng.integers(0, 2, size=(k, GATE_ACTIONS)).astype(bool)
    b_sz = rng.integers(0, 1000, size=(k, 4)).astype(np.int64)
    o, m, s = st.upload(b_obs, b_gm, b_sz)
    o2, m2, s2 = _ref_upload(device, b_obs, b_gm, b_sz)
    assert torch.equal(o, o2)
    assert torch.equal(m, m2)
    assert torch.equal(s, s2)


def test_pinned_step_h2d_cuda_matches_from_numpy():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    obs_dim = 128
    cap = 32
    st = _PinnedStepH2D(capacity=cap, obs_dim=obs_dim, device=device)
    assert st.enabled is True
    rng = np.random.default_rng(1)
    for k in (1, 5, 17, cap):
        b_obs = rng.standard_normal((k, obs_dim)).astype(np.float32)
        b_gm = rng.integers(0, 2, size=(k, GATE_ACTIONS)).astype(bool)
        b_sz = rng.integers(0, 1000, size=(k, 4)).astype(np.int64)
        o, m, s = st.upload(b_obs, b_gm, b_sz)
        # Sync as _forward would via D2H
        o_cpu = o.cpu()
        m_cpu = m.cpu()
        s_cpu = s.cpu()
        o2, m2, s2 = _ref_upload(device, b_obs, b_gm, b_sz)
        assert torch.equal(o_cpu, o2.cpu())
        assert torch.equal(m_cpu, m2.cpu())
        assert torch.equal(s_cpu, s2.cpu())


def test_pinned_step_h2d_reuse_after_sync():
    """Two sequential uploads after CPU sync must not corrupt values."""
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    st = _PinnedStepH2D(capacity=8, obs_dim=16, device=device)
    a = np.ones((3, 16), dtype=np.float32)
    b = np.full((4, 16), 2.0, dtype=np.float32)
    gm = np.ones((3, GATE_ACTIONS), dtype=bool)
    gm2 = np.zeros((4, GATE_ACTIONS), dtype=bool)
    sz = np.zeros((3, 4), dtype=np.int64)
    sz2 = np.ones((4, 4), dtype=np.int64)
    o1, _, _ = st.upload(a, gm, sz)
    v1 = o1.cpu().clone()
    o2, _, _ = st.upload(b, gm2, sz2)
    v2 = o2.cpu()
    assert torch.allclose(v1, torch.ones_like(v1))
    assert torch.allclose(v2, torch.full_like(v2, 2.0))
