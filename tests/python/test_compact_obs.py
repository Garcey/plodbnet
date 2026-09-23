"""Compact rollout-observation storage (plo5bp/compact_obs.py, 2026-09-23).

The batched rollout stores each observation's exact-0/1 columns as bits and
the rest verbatim; PPO unpacks per minibatch. Storage only — these tests pin
that it is invisible to training:

- the layouts mark exactly the documented flag blocks,
- the Rust packer equals the numpy reference and round-trips bit-exactly on
  real engine observations, and refuses any non-0/1 flag value,
- PackedObs row indexing == dense indexing (the iter_minibatches contract),
- a compact rollout and a dense rollout with the same seeds produce identical
  batches (and leave the RNG streams identical),
- a PPO update on either gives identical stats and identical weights.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from plo5bp import compact_obs
from plo5bp import rollout as rollout_mod
from plo5bp.compact_obs import (
    FULL_LAYOUT,
    MINIMAL_LAYOUT,
    PackedObs,
    as_dense,
    layout_for,
    pack_rows_into,
    pack_rows_np,
    unpack,
)
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import (
    _HISTORY_DEPTH,
    _HISTORY_SLOT_DIM,
    _M_HERO_BTN,
    _M_HISTORY,
    _M_SEAT_EXISTS,
    _M_STACKS,
    _MINIMAL_INDEX,
    OBS_DIM,
    OBS_DIM_MINIMAL,
)
from plo5bp.env_batched import BatchedBombPotEnv
from plo5bp.network import ActorCriticV2, CentralCritic
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import (
    _BATCH_TENSOR_FIELDS,
    Batch,
    collect_rollout_batched,
    collect_rollout_multiconfig,
    iter_minibatches,
)
from plo5bp.selfplay import OpponentPool

BB = 10_000
_LAYOUTS = {"minimal": MINIMAL_LAYOUT, "full": FULL_LAYOUT}
_DIMS = {"minimal": OBS_DIM_MINIMAL, "full": OBS_DIM}


# ------------------------------------------------------------------ layouts
def test_layouts_mark_exactly_the_documented_flag_blocks() -> None:
    expected = np.zeros(OBS_DIM_MINIMAL, dtype=bool)
    expected[:_M_STACKS] = True  # hole + board A + board B, street, active, all-in
    for slot in range(_HISTORY_DEPTH):  # seat/gate/street one-hots of each slot
        base = _M_HISTORY + slot * _HISTORY_SLOT_DIM
        expected[base : base + 16] = True
    expected[_M_SEAT_EXISTS : _M_SEAT_EXISTS + 8] = True
    expected[_M_HERO_BTN : _M_HERO_BTN + 8] = True
    got = np.zeros(OBS_DIM_MINIMAL, dtype=bool)
    got[MINIMAL_LAYOUT.flag_cols] = True
    assert np.array_equal(got, expected)
    # the full layout flags the SAME columns (the minimal layout is a gather)
    assert np.array_equal(FULL_LAYOUT.flag_cols, np.sort(_MINIMAL_INDEX[expected]))
    for layout, dim, n_real in ((MINIMAL_LAYOUT, OBS_DIM_MINIMAL, 92), (FULL_LAYOUT, OBS_DIM, 467)):
        assert layout.obs_dim == dim and layout.n_flag == 704 and layout.n_real == n_real
        both = np.concatenate([layout.flag_cols, layout.real_cols])
        assert np.array_equal(np.sort(both), np.arange(dim))  # a partition
    assert MINIMAL_LAYOUT.row_bytes == 456 and FULL_LAYOUT.row_bytes == 1956


def test_layout_for_dispatch() -> None:
    assert layout_for("plo5_double_bomb", "minimal") is MINIMAL_LAYOUT
    assert layout_for("plo6_double_bomb", "full") is FULL_LAYOUT
    assert layout_for("plo4_double_bomb", "minimal") is MINIMAL_LAYOUT
    assert layout_for("nlh_single", "full") is None  # NLH stays dense
    with pytest.raises(ValueError):
        layout_for("plo5_double_bomb", "tiny")


# ------------------------------------------------------------ pack / unpack
def _real_obs(obs_mode: str, n: int = 256, steps: int = 8, seed: int = 0) -> np.ndarray:
    """Engine observations with populated history slots (a few check/call
    steps of a random table set)."""
    rng = np.random.default_rng(seed)
    cfg = GameConfig(num_seats=6, starting_stack=100 * BB, ante=3 * BB, bb=BB)
    env = BatchedBombPotEnv(n, cfg, opp_outcome_mc=32, obs_mode=obs_mode)
    env.reset_batch(
        rng.integers(0, 2**62, size=n, dtype=np.uint64),
        rng.integers(0, 6, size=n).astype(np.uint8),
    )
    for _ in range(steps):
        gm = env._gate_mask
        gates = np.where(gm[:, 1], 1, np.where(gm[:, 0], 0, 2)).astype(np.uint8)
        env._be.apply_hybrid_batch(gates, np.zeros(n, dtype=np.uint64))
        env._refresh()
    return np.ascontiguousarray(env._obs)


@pytest.mark.parametrize("obs_mode", ["minimal", "full"])
def test_rust_packer_matches_numpy_reference_and_round_trips(obs_mode: str) -> None:
    layout = _LAYOUTS[obs_mode]
    obs = _real_obs(obs_mode)
    assert obs.shape[1] == layout.obs_dim
    rows = np.sort(np.random.default_rng(1).choice(obs.shape[0], 150, replace=False))
    bits = np.full((200, layout.n_bytes), 0xAB, dtype=np.uint8)  # poisoned
    real = np.full((200, layout.n_real), np.nan, dtype=np.float32)
    pack_rows_into(obs, rows, layout, bits, real, 30)
    ref_bits, ref_real = pack_rows_np(obs, rows, layout)
    assert np.array_equal(bits[30:180], ref_bits)
    assert np.array_equal(real[30:180].view(np.uint32), ref_real.view(np.uint32))
    assert (bits[:30] == 0xAB).all() and (bits[180:] == 0xAB).all()  # untouched
    dense = unpack(torch.from_numpy(bits[30:180]), torch.from_numpy(real[30:180]), layout)
    assert np.array_equal(dense.numpy().view(np.uint32), obs[rows].view(np.uint32))


@pytest.mark.parametrize("bad", [0.5, -0.0, 2.0, float("nan"), -1.0])
def test_packer_rejects_any_non_flag_value(bad: float) -> None:
    obs = _real_obs("minimal", n=16, steps=2)
    col = int(MINIMAL_LAYOUT.flag_cols[100])
    obs[5, col] = bad
    bits = np.zeros((16, MINIMAL_LAYOUT.n_bytes), dtype=np.uint8)
    real = np.zeros((16, MINIMAL_LAYOUT.n_real), dtype=np.float32)
    with pytest.raises(ValueError, match=f"column {col}"):
        pack_rows_into(obs, np.arange(16), MINIMAL_LAYOUT, bits, real, 0)
    with pytest.raises(ValueError, match=f"column {col}"):
        pack_rows_np(obs, np.arange(16), MINIMAL_LAYOUT)
    # rows that avoid the bad one still pack
    pack_rows_into(obs, np.array([0, 1, 2]), MINIMAL_LAYOUT, bits, real, 0)


def test_packer_validates_indices_and_buffers() -> None:
    obs = _real_obs("minimal", n=8, steps=1)
    L = MINIMAL_LAYOUT
    bits = np.zeros((8, L.n_bytes), dtype=np.uint8)
    real = np.zeros((8, L.n_real), dtype=np.float32)
    with pytest.raises(ValueError, match="row index"):
        pack_rows_into(obs, np.array([8]), L, bits, real, 0)
    with pytest.raises(ValueError, match="overflow"):
        pack_rows_into(obs, np.arange(4), L, bits, real, 6)
    with pytest.raises(ValueError, match="width"):
        pack_rows_into(obs, np.arange(4), L, bits[:, :10], real, 0)
    # column-major memory must be refused (numpy's as_slice accepts it)
    with pytest.raises(ValueError, match="C-contiguous"):
        pack_rows_into(np.asfortranarray(obs), np.arange(4), L, bits, real, 0)
    with pytest.raises(ValueError, match="C-contiguous"):
        pack_rows_into(obs, np.arange(4), L, np.asfortranarray(bits), real, 0)


def test_packed_obs_behaves_like_the_dense_tensor_under_row_indexing() -> None:
    obs = _real_obs("full", n=64, steps=5)
    b, r = pack_rows_np(obs, np.arange(64), FULL_LAYOUT)
    po = PackedObs(torch.from_numpy(b), torch.from_numpy(r), FULL_LAYOUT)
    dense = torch.from_numpy(obs)
    assert po.shape == dense.shape and po.device == dense.device and len(po) == 64
    assert po.dtype == torch.float32 and po.size(1) == OBS_DIM
    sel = torch.tensor([5, 0, 63, 5, 17])
    assert torch.equal(po[sel], dense[sel])
    assert torch.equal(po[3:40], dense[3:40])
    assert torch.equal(po[7], dense[7])
    assert torch.equal(po[dense[:, 0] > 0], dense[dense[:, 0] > 0])  # bool mask
    assert torch.equal(po.to("cpu").dense(), dense)
    assert torch.equal(PackedObs.cat([po, po]).dense(), torch.cat([dense, dense]))
    assert torch.equal(as_dense(po), dense) and as_dense(dense) is dense
    with pytest.raises(TypeError):
        po[:, 0]  # rows only
    with pytest.raises(TypeError):  # fails loudly instead of silently unpacking
        torch.nn.Linear(OBS_DIM, 2)(po)


@pytest.mark.parametrize("path", ["rust", "torch"])
@pytest.mark.parametrize("obs_mode", ["minimal", "full"])
def test_both_unpack_paths_are_exact(monkeypatch, path: str, obs_mode: str) -> None:
    """CPU tensors unpack through Rust; other devices (CUDA) through the torch
    lookup-table path — forced here on CPU so the GPU code path is pinned too
    (with a tiny chunk size, to exercise the chunking)."""
    layout = _LAYOUTS[obs_mode]
    obs = _real_obs(obs_mode, n=96, steps=6)
    b, r = pack_rows_np(obs, np.arange(96), layout)
    if path == "torch":
        monkeypatch.setattr(compact_obs, "_rust_unpack_obs_rows", None)
        monkeypatch.setattr(compact_obs, "_UNPACK_CHUNK_ROWS", 7)
    out = unpack(torch.from_numpy(b), torch.from_numpy(r), layout)
    assert np.array_equal(out.numpy().view(np.uint32), obs.view(np.uint32))
    empty = unpack(torch.from_numpy(b[:0]), torch.from_numpy(r[:0]), layout)
    assert empty.shape == (0, layout.obs_dim)


def test_rust_unpacker_validates_the_column_partition() -> None:
    L = MINIMAL_LAYOUT
    bits = np.zeros((4, L.n_bytes), dtype=np.uint8)
    real = np.zeros((4, L.n_real), dtype=np.float32)
    out = np.empty((4, L.obs_dim), dtype=np.float32)
    unpack_rows = compact_obs._rust_unpack_obs_rows
    dup = L.real_cols.copy()
    dup[0] = L.flag_cols[0]  # a column listed twice
    with pytest.raises(ValueError, match="twice"):
        unpack_rows(bits, real, L.flag_cols, dup, out)
    with pytest.raises(ValueError, match="cover"):  # one output column left unwritten
        unpack_rows(bits, real[:, :-1].copy(), L.flag_cols, L.real_cols[:-1].copy(), out)
    with pytest.raises(ValueError, match="C-contiguous"):
        unpack_rows(bits, real, L.flag_cols, L.real_cols, np.asfortranarray(out))


# ------------------------------------------------------ rollout end to end
def _collect_pair(obs_mode: str, multi: bool, with_critic: bool):
    """The same collection with compact and with dense storage (every RNG
    re-seeded), plus the RNG states each left behind."""
    dim = _DIMS[obs_mode]
    out = []
    for compact in (True, False):
        torch.manual_seed(11)
        model = ActorCriticV2(hidden_dim=32, obs_dim=dim)
        critic = CentralCritic(obs_dim=dim, hidden_dim=32, num_blocks=1) if with_critic else None
        tc = TrainingConfig(
            num_envs=8, rollout_length=160, hidden_dim=32,
            obs_mode=obs_mode, compact_obs=compact,
        )
        rng = np.random.default_rng(5)
        torch.manual_seed(12)
        if multi:
            cfgs = [
                GameConfig(num_seats=3, starting_stack=40 * BB, ante=3 * BB, bb=BB),
                GameConfig(num_seats=5, starting_stack=120 * BB, ante=3 * BB, bb=BB),
            ]
            batch = collect_rollout_multiconfig(model, OpponentPool(capacity=1), cfgs, tc, rng, critic=critic)
        else:
            cfg = GameConfig(num_seats=4, starting_stack=60 * BB, ante=3 * BB, bb=BB)
            batch = collect_rollout_batched(model, OpponentPool(capacity=1), cfg, tc, rng, critic=critic)
        out.append((batch, rng.bit_generator.state, torch.get_rng_state()))
    return out


def _assert_same_batch(a: Batch, b: Batch) -> None:
    for f in _BATCH_TENSOR_FIELDS:
        ta, tb = as_dense(getattr(a, f)), as_dense(getattr(b, f))
        assert ta.dtype == tb.dtype and ta.shape == tb.shape, f
        assert torch.equal(ta, tb), f
    assert torch.equal(a.is_terminal, b.is_terminal)
    assert a.aggr_steps_total == b.aggr_steps_total
    assert a.aggr_steps_total_by_street == b.aggr_steps_total_by_street


@pytest.mark.parametrize("obs_mode", ["minimal", "full"])
@pytest.mark.parametrize("multi", [False, True])
def test_compact_rollout_is_bit_identical_to_dense(obs_mode: str, multi: bool) -> None:
    (cb, c_np, c_torch), (db, d_np, d_torch) = _collect_pair(obs_mode, multi, with_critic=True)
    assert isinstance(cb.obs, PackedObs) and cb.obs.layout is _LAYOUTS[obs_mode]
    assert isinstance(db.obs, torch.Tensor)
    _assert_same_batch(cb, db)
    assert c_np == d_np and torch.equal(c_torch, d_torch)  # storage draws no RNG
    assert cb.obs.nbytes == len(cb.obs) * _LAYOUTS[obs_mode].row_bytes
    assert cb.obs.nbytes * 6 < db.obs.numel() * 4 or obs_mode == "full"  # ~7x on minimal


def test_legacy_staging_concatenates_packed_batches() -> None:
    torch.manual_seed(3)
    model = ActorCriticV2(hidden_dim=32, obs_dim=OBS_DIM_MINIMAL)
    cfgs = [GameConfig(num_seats=n, starting_stack=50 * BB, ante=3 * BB, bb=BB) for n in (2, 4)]
    tc = TrainingConfig(num_envs=6, rollout_length=120, hidden_dim=32, obs_mode="minimal")
    runs = []
    for legacy in (True, False):
        torch.manual_seed(4)
        runs.append(collect_rollout_multiconfig(
            model, OpponentPool(capacity=1), cfgs, tc, np.random.default_rng(9),
            _legacy_staging=legacy,
        ))
    assert all(isinstance(b.obs, PackedObs) for b in runs)
    _assert_same_batch(runs[0], runs[1])


# ----------------------------------------------------------- PPO end to end
@pytest.mark.parametrize("obs_mode", ["minimal", "full"])
def test_ppo_update_is_identical_on_compact_and_dense_storage(obs_mode: str) -> None:
    (cb, _, _), (db, _, _) = _collect_pair(obs_mode, multi=True, with_critic=True)
    dim = _DIMS[obs_mode]
    torch.manual_seed(21)
    model = ActorCriticV2(hidden_dim=32, obs_dim=dim)
    critic = CentralCritic(obs_dim=dim, hidden_dim=32, num_blocks=1)
    tc = TrainingConfig(num_envs=8, rollout_length=160, hidden_dim=32, ppo_epochs=2, batch_size=48)
    results = []
    for batch in (cb, db):
        m, c = copy.deepcopy(model), copy.deepcopy(critic)
        trainer = PPOTrainer(m, tc, critic=c)
        torch.manual_seed(22)
        stats = trainer.update(batch, np.random.default_rng(23))
        results.append((stats, m.state_dict(), c.state_dict()))
    (s1, m1, c1), (s2, m2, c2) = results
    v1, v2 = vars(s1), vars(s2)
    assert v1.keys() == v2.keys()
    for k in v1:  # exact, NaN-tolerant
        assert v1[k] == v2[k] or (v1[k] != v1[k] and v2[k] != v2[k]), (k, v1[k], v2[k])
    for k in m1:
        assert torch.equal(m1[k], m2[k]), k
    for k in c1:
        assert torch.equal(c1[k], c2[k]), k


def test_iter_minibatches_yields_dense_rows_equal_to_dense_storage() -> None:
    (cb, _, _), (db, _, _) = _collect_pair("minimal", multi=False, with_critic=False)
    a = list(iter_minibatches(cb, 37, np.random.default_rng(1)))
    b = list(iter_minibatches(db, 37, np.random.default_rng(1)))
    assert len(a) == len(b) > 1
    for ma, mb in zip(a, b):
        assert isinstance(ma.obs, torch.Tensor) and ma.obs.dtype == torch.float32
        assert torch.equal(ma.obs, mb.obs)


# ------------------------------------------------------------- resolution
def test_resolve_obs_layout(monkeypatch, capsys) -> None:
    tc = TrainingConfig(obs_mode="minimal")
    assert rollout_mod._resolve_obs_layout(tc, "plo5_double_bomb") is MINIMAL_LAYOUT
    off = TrainingConfig(obs_mode="minimal", compact_obs=False)
    assert rollout_mod._resolve_obs_layout(off, "plo5_double_bomb") is None
    assert rollout_mod._resolve_obs_layout(TrainingConfig(), "nlh_single") is None
    # an engine built before pack_obs_rows: dense, said loudly once
    monkeypatch.setattr(rollout_mod, "RUST_PACKER_AVAILABLE", False)
    monkeypatch.setattr(rollout_mod, "_WARNED_DENSE_FALLBACK", False)
    assert rollout_mod._resolve_obs_layout(tc, "plo5_double_bomb") is None
    assert rollout_mod._resolve_obs_layout(tc, "plo5_double_bomb") is None
    assert capsys.readouterr().out.count("storing observations DENSE") == 1
