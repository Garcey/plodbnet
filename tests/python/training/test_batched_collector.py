"""The batched collector's structure (2026-09-28, ML-007 / ML-028 / ML-043).

`collect_rollout_batched` is a `_BatchedCollector` now: setup in `__init__`,
one method per step phase. Its end-to-end bit-exactness is pinned by the
checkpoint digests (tests/python/training/test_exactness.py, scripts/exactness_check.py)
and by the kernel parity (test_rust_flush.py); pinned here:

- the per-step output buffers (`_StepActs`) are reset to exactly what the old
  per-step np.zeros / np.full allocated;
- the frozen opponents' weights are stacked once per UPDATE (`_OpponentCache`)
  and the collection is bitwise the one that restacks per sub-rollout;
- a stale engine is an import-time error naming what is missing
  (`engine_abi`), never a silent fallback.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plo5bp import engine_abi
from plo5bp import rollout as R
from plo5bp.compact_obs import as_dense
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM_MINIMAL
from plo5bp.network import ActorCriticV5, CentralCritic
from plo5bp.selfplay import OpponentPool

_FIELDS = (
    "obs", "gate_masks", "gate_actions", "raise_chips", "sizing", "anchor_actions",
    "refine_u", "opp_holes", "log_probs", "values", "returns", "advantages",
    "old_gate_logp", "old_anchor_logp", "is_terminal",
)


def test_step_acts_reset_restores_the_fresh_contents() -> None:
    acts = R._StepActs(7, vrpo=True)
    fresh = {k: getattr(acts, k).copy() for k in R._StepActs.__slots__}
    assert (fresh["anchor"] == -1).all() and not fresh["gate"].any()
    for k in R._StepActs.__slots__:
        getattr(acts, k)[:] = 3
    acts.reset()
    for k, want in fresh.items():
        got = getattr(acts, k)
        assert got.dtype == want.dtype and np.array_equal(got, want), k
    plain = R._StepActs(3, vrpo=False)
    assert plain.q_taken is None and plain.vpi is None
    plain.reset()
    assert set(plain.per_env(np.zeros((3, 4), np.int64))) == {
        "gate", "chips", "sizing", "anchor", "u", "log_p", "gate_lp",
        "anchor_lp", "value",
    }
    assert {"q_taken", "vpi"} <= set(acts.per_env(np.zeros((7, 4), np.int64)))


def _collect(monkeypatch, cache_cls) -> "tuple[R.Batch, int]":
    """One mixed-config update against a 3-snapshot pool (every hand mixed),
    with `cache_cls` as the per-update opponent cache; returns the batch and
    how many times the snapshots were stacked."""
    monkeypatch.setenv("PLO5_RUST_ENCODER", "1")
    monkeypatch.setattr(R, "_OpponentCache", cache_cls)
    stacks = []
    real_init = R._StackedOpponents.__init__

    def counting_init(self, models):
        stacks.append(len(models))
        real_init(self, models)

    monkeypatch.setattr(R._StackedOpponents, "__init__", counting_init)
    R._clear_rollout_buffers()
    torch.manual_seed(0)
    learner = ActorCriticV5(
        hidden_dim=16, obs_dim=OBS_DIM_MINIMAL, num_layers=3, torso_layernorm=True
    )
    pool = OpponentPool(capacity=4, seed=0)
    for tag in range(3):
        with torch.no_grad():
            for p in learner.parameters():
                p.add_(0.01 * (tag + 1))
        pool.snapshot(learner, tag=tag)
    torch.manual_seed(1)
    critic = CentralCritic(obs_dim=OBS_DIM_MINIMAL, hidden_dim=16, num_blocks=1)
    cfg = TrainingConfig(
        num_envs=36, rollout_length=900, obs_mode="minimal",
        pool_mix_prob=1.0, pool_opp_seats=1,
    )
    configs = [
        GameConfig(num_seats=s, starting_stack=st * 10_000, ante=30_000, bb=10_000)
        for s, st in ((2, 40), (6, 20), (4, 120))
    ]
    torch.manual_seed(7)
    batch = R.collect_rollout_multiconfig(
        learner, pool, configs, cfg, np.random.default_rng(5), critic=critic,
    )
    return batch, len(stacks)


class _ForgetfulCache(R._OpponentCache):
    """Never remembers a stack: the members are still built once per update,
    but stacked again for every sub-rollout -- the behaviour before ML-043."""

    @property
    def stacked_key(self):
        return None

    @stacked_key.setter
    def stacked_key(self, value) -> None:
        pass


def test_opponents_stack_once_per_update_bit_exactly(monkeypatch) -> None:
    once, n_once = _collect(monkeypatch, R._OpponentCache)
    every, n_every = _collect(monkeypatch, _ForgetfulCache)
    assert n_once == 1
    assert n_every == 3
    for f in _FIELDS:
        x, y = getattr(once, f), getattr(every, f)
        if f == "obs":
            x, y = as_dense(x), as_dense(y)
        x, y = x.numpy(), y.numpy()
        if x.dtype == np.float32:
            x, y = x.view(np.uint32), y.view(np.uint32)
        assert np.array_equal(x, y), f


def test_engine_abi_names_what_a_stale_engine_lacks() -> None:
    assert engine_abi.missing(["flush_trajectories", "BatchedEngine.encode"]) == []
    assert engine_abi.missing(
        ["no_such_kernel", "BatchedEngine.no_such_method", "NoSuchClass.x"]
    ) == ["no_such_kernel", "BatchedEngine.no_such_method", "NoSuchClass.x"]
    # A module constant counts too (ENG-009 dropped the old per-feature class flags).
    assert engine_abi.missing(["SOURCE_HASH"]) == []
    with pytest.raises(engine_abi.EngineOutOfDate, match="maturin develop --release") as e:
        engine_abi.require("flush_trajectories", "no_such_kernel")
    assert "no_such_kernel" in str(e.value) and "flush_trajectories" not in str(e.value)
    assert isinstance(e.value, ImportError)
    (fn,) = engine_abi.functions("gather_rows_multi")
    assert fn is R._rust_gather_rows_multi


def test_numpy_flush_switch_selects_the_reference_kernels(monkeypatch) -> None:
    from plo5bp import rollout_reference as ref

    monkeypatch.delenv("PLO5BP_NUMPY_FLUSH", raising=False)
    assert R._kernels() is R._RUST_KERNELS
    monkeypatch.setenv("PLO5BP_NUMPY_FLUSH", "1")
    k = R._kernels()
    assert (k.record, k.aggression, k.flush) == (
        ref.record_learner_steps, ref.aggression_record_batch, ref.flush_trajectories
    )
