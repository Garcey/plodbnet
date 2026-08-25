"""Bare-visibility (minimal) observation layout."""
from __future__ import annotations

import numpy as np
import pytest

from plo5bp.config import GameConfig
from plo5bp.encoding import (
    OBS_DIM,
    OBS_DIM_MINIMAL,
    _MINIMAL_INDEX,
    _SPR_OFF,
    _OPP_OUTCOME_OFF,
    project_obs_minimal,
    encode_observation_minimal,
    encode_observation_batch_minimal,
    encode_observation,
    encode_observation_batch,
)
from plo5bp.env import BombPotEnv
from plo5bp.env_batched import BatchedBombPotEnv


def test_minimal_dim_and_index():
    assert OBS_DIM_MINIMAL == 796
    assert _MINIMAL_INDEX.shape == (796,)
    # cards block contiguous
    assert list(_MINIMAL_INDEX[:156]) == list(range(156))
    kept = set(int(x) for x in _MINIMAL_INDEX)
    assert _SPR_OFF not in kept
    assert _OPP_OUTCOME_OFF not in kept
    assert 990 not in kept  # bet_pct_pot


def test_serial_minimal_matches_project():
    cfg = GameConfig()
    full = BombPotEnv(cfg, obs_mode="full")
    mini = BombPotEnv(cfg, obs_mode="minimal")
    obs_f, _ = full.reset(seed=7, button=1)
    obs_m, _ = mini.reset(seed=7, button=1)
    assert obs_f.shape == (OBS_DIM,)
    assert obs_m.shape == (OBS_DIM_MINIMAL,)
    np.testing.assert_array_equal(project_obs_minimal(obs_f), obs_m)


def test_batched_minimal_matches_project():
    cfg = GameConfig()
    full = BatchedBombPotEnv(4, cfg, obs_mode="full", opp_outcome_mc=1)
    mini = BatchedBombPotEnv(4, cfg, obs_mode="minimal", opp_outcome_mc=0)
    seeds = np.arange(4, dtype=np.uint64) + 100
    buttons = np.array([0, 1, 2, 0], dtype=np.uint8)
    st_f = full.reset_batch(seeds, buttons)
    st_m = mini.reset_batch(seeds, buttons)
    assert st_m.obs.shape == (4, OBS_DIM_MINIMAL)
    np.testing.assert_array_equal(project_obs_minimal(st_f.obs), st_m.obs)


def test_minimal_rejects_nlh():
    from plo5bp.config import VARIANT_NLH
    cfg = GameConfig(variant=VARIANT_NLH)
    with pytest.raises(ValueError, match="PLO-only"):
        BombPotEnv(cfg, obs_mode="minimal")



def test_direct_minimal_matches_project_serial():
    """Direct 796-d encode == project(full encode); no full-width intermediate."""
    cfg = GameConfig()
    env = BombPotEnv(cfg, obs_mode="full")
    obs_f, info = env.reset(seed=42, button=2)
    # Re-encode from raw via both paths
    raw = info.raw_obs
    # Need categories etc. on raw - use env pack path
    mini = BombPotEnv(cfg, obs_mode="minimal")
    obs_m, _ = mini.reset(seed=42, button=2)
    np.testing.assert_array_equal(project_obs_minimal(obs_f), obs_m)


def test_direct_minimal_matches_project_batch():
    cfg = GameConfig()
    full = BatchedBombPotEnv(8, cfg, obs_mode="full", opp_outcome_mc=8)
    mini = BatchedBombPotEnv(8, cfg, obs_mode="minimal", opp_outcome_mc=0)
    seeds = np.arange(8, dtype=np.uint64) + 200
    buttons = np.zeros(8, dtype=np.uint8)
    st_f = full.reset_batch(seeds, buttons)
    st_m = mini.reset_batch(seeds, buttons)
    np.testing.assert_array_equal(project_obs_minimal(st_f.obs), st_m.obs)
    assert mini._opp_outcome_mc == 0


def test_opp_outcome_mc_zero_allowed():
    """Rust accepts opp_outcome_mc=0 and skips MC (zeros outcome slots)."""
    cfg = GameConfig()
    env = BatchedBombPotEnv(2, cfg, obs_mode="full", opp_outcome_mc=0)
    assert env._opp_outcome_mc == 0
    seeds = np.array([1, 2], dtype=np.uint64)
    buttons = np.array([0, 1], dtype=np.uint8)
    st = env.reset_batch(seeds, buttons)
    assert st.obs.shape == (2, OBS_DIM)
    # Opp-outcome block must be zeros when mc=0
    from plo5bp.encoding import _OPP_OUTCOME_OFF, _OPP_OUTCOME_DIM
    block = st.obs[:, _OPP_OUTCOME_OFF : _OPP_OUTCOME_OFF + _OPP_OUTCOME_DIM]
    assert np.all(block == 0.0)


def test_batched_minimal_forces_mc_zero():
    cfg = GameConfig()
    # Caller asks for 384; minimal mode must force 0.
    env = BatchedBombPotEnv(2, cfg, obs_mode="minimal", opp_outcome_mc=384)
    assert env._opp_outcome_mc == 0


def test_rust_minimal_encoder_parity(monkeypatch):
    """Lean Rust 796 encoder must match numpy encode_observation_batch_minimal."""
    from plo5bp._engine import BatchedEngine
    if not hasattr(BatchedEngine, "observation_encoded_minimal_batch"):
        pytest.skip("minimal rust encoder not in this binary")
    monkeypatch.setenv("PLO5_RUST_ENCODER", "1")
    # Reset the once-flag so the new env logs/uses the gate freshly.
    if hasattr(BatchedBombPotEnv, "_encoder_log_once"):
        BatchedBombPotEnv._encoder_log_once = False
    cfg = GameConfig()
    rust_env = BatchedBombPotEnv(8, cfg, obs_mode="minimal", opp_outcome_mc=0)
    assert rust_env._use_rust_encoder is True
    assert rust_env._rust_minimal is True
    # Force numpy path for reference
    monkeypatch.setenv("PLO5_RUST_ENCODER", "0")
    if hasattr(BatchedBombPotEnv, "_encoder_log_once"):
        BatchedBombPotEnv._encoder_log_once = False
    np_env = BatchedBombPotEnv(8, cfg, obs_mode="minimal", opp_outcome_mc=0)
    assert np_env._use_rust_encoder is False
    seeds = np.arange(8, dtype=np.uint64) + 300
    buttons = np.array([0, 1, 2, 3, 4, 5, 0, 1], dtype=np.uint8)
    st_r = rust_env.reset_batch(seeds, buttons)
    st_n = np_env.reset_batch(seeds, buttons)
    assert st_r.obs.shape == (8, OBS_DIM_MINIMAL)
    np.testing.assert_array_equal(st_r.obs, st_n.obs)
    # Also match project(full)
    monkeypatch.delenv("PLO5_RUST_ENCODER", raising=False)
    if hasattr(BatchedBombPotEnv, "_encoder_log_once"):
        BatchedBombPotEnv._encoder_log_once = False
    full = BatchedBombPotEnv(8, cfg, obs_mode="full", opp_outcome_mc=0)
    st_f = full.reset_batch(seeds, buttons)
    np.testing.assert_array_equal(project_obs_minimal(st_f.obs), st_r.obs)


def test_rust_minimal_subset_parity(monkeypatch):
    from plo5bp._engine import BatchedEngine
    if not hasattr(BatchedEngine, "observation_encoded_minimal_batch"):
        pytest.skip("minimal rust encoder not in this binary")
    cfg = GameConfig()
    be = BatchedEngine(6, num_seats=6, opp_outcome_mc=0)
    seeds = np.arange(6, dtype=np.uint64) + 50
    buttons = np.zeros(6, dtype=np.uint8)
    be.reset_batch(seeds, buttons)
    full = np.asarray(be.observation_encoded_minimal_batch()["obs"])
    assert full.shape == (6, OBS_DIM_MINIMAL)
    idx = np.array([0, 2, 5], dtype=np.int64)
    sub = np.asarray(be.observation_encoded_minimal_subset_batch(idx)["obs"])
    assert sub.shape == (3, OBS_DIM_MINIMAL)
    np.testing.assert_array_equal(sub, full[idx])

