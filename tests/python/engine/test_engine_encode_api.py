"""`BatchedEngine.encode` — the one Rust-encoder entry point (ENG-012).

Every legacy `observation_encoded_*` method is now a thin alias of the same
implementation; these tests pin that `encode(...)` reproduces each of them
BIT-EXACTLY (obs rows, in-place rows, packed rows and every aux array), and that
the argument validation the old methods had still holds.
"""

from __future__ import annotations

import numpy as np
import pytest

from plo5bp._engine import BatchedEngine
from plo5bp.compact_obs import FULL_LAYOUT, MINIMAL_LAYOUT

N = 24


def _engine(seats: int = 5, mc: int = 64, seed0: int = 11) -> BatchedEngine:
    """An engine mid-hand: dealt, then a few random legal actions applied."""
    be = BatchedEngine(N, num_seats=seats, starting_stack=400_000, opp_outcome_mc=mc)
    be.reset_batch(np.arange(N, dtype=np.uint64) + seed0, (np.arange(N) % seats).astype(np.uint8))
    rng = np.random.default_rng(seed0)
    for _ in range(4):
        d = be.observation_and_features_batch()
        legal = d["legal_mask"]
        gates = np.ones(N, dtype=np.uint8)
        chips = np.zeros(N, dtype=np.uint64)
        for i in range(N):
            if d["actor"][i] < 0:
                continue
            if legal[i, 0] and rng.random() < 0.15:
                gates[i] = 0
            elif d["min_raise"][i] > 0 and rng.random() < 0.4:
                gates[i] = 2
                lo, hi = int(d["min_raise"][i]), int(d["max_raise"][i])
                chips[i] = rng.integers(lo, hi + 1)
        be.apply_hybrid_batch(gates, chips)
    return be


def _same_dict(a: dict, b: dict) -> None:
    assert sorted(a) == sorted(b)
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert x.dtype == y.dtype and x.shape == y.shape, k
        assert np.array_equal(x.view(np.uint8), y.view(np.uint8)), k


@pytest.mark.parametrize("layout", ["full", "minimal"])
def test_encode_returns_the_legacy_batch_and_subset_results(layout: str) -> None:
    be = _engine()
    legacy = be.observation_encoded_batch if layout == "full" else be.observation_encoded_minimal_batch
    _same_dict(be.encode(layout), legacy())
    idx = np.asarray([7, 2, 2, 19, 0], dtype=np.int64)  # any order, repeats allowed
    sub = (
        be.observation_encoded_subset_batch
        if layout == "full"
        else be.observation_encoded_minimal_subset_batch
    )
    _same_dict(be.encode(layout, idx), sub(idx))


@pytest.mark.parametrize("layout", ["full", "minimal"])
def test_encode_in_place_matches_the_legacy_into_methods(layout: str) -> None:
    be = _engine()
    dim = (FULL_LAYOUT if layout == "full" else MINIMAL_LAYOUT).obs_dim
    into = be.observation_encoded_into if layout == "full" else be.observation_encoded_minimal_into
    mask = (np.arange(N) % 3) != 1
    a = np.full((N, dim), 7.0, dtype=np.float32)
    b = np.full((N, dim), 7.0, dtype=np.float32)
    _same_dict(be.encode(layout, out=a, encode_mask=mask), into(b, mask))
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    assert not a[~mask].any()  # masked rows are zero-filled
    # Subset in place: other rows untouched.
    sub_into = (
        be.observation_encoded_subset_into
        if layout == "full"
        else be.observation_encoded_minimal_subset_into
    )
    idx = np.asarray([1, 4, 5, 20], dtype=np.int64)
    a[:] = 3.0
    b[:] = 3.0
    _same_dict(be.encode(layout, idx, out=a), sub_into(idx, b))
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    untouched = np.setdiff1d(np.arange(N), idx)
    assert (a[untouched] == 3.0).all()


@pytest.mark.parametrize("layout", ["full", "minimal"])
def test_encode_packed_outputs_match_the_legacy_packed_only_path(layout: str) -> None:
    be = _engine()
    lay = FULL_LAYOUT if layout == "full" else MINIMAL_LAYOUT
    into = be.observation_encoded_into if layout == "full" else be.observation_encoded_minimal_into
    nb, nr = (lay.flag_cols.size + 7) // 8, lay.real_cols.size
    bits_a, bits_b = np.zeros((N, nb), np.uint8), np.zeros((N, nb), np.uint8)
    real_a, real_b = np.zeros((N, nr), np.float32), np.zeros((N, nr), np.float32)
    _same_dict(
        be.encode(layout, flag_cols=lay.flag_cols, real_cols=lay.real_cols, out_bits=bits_a, out_real=real_a),
        into(None, None, lay.flag_cols, lay.real_cols, bits_b, real_b),
    )
    assert np.array_equal(bits_a, bits_b)
    assert np.array_equal(real_a.view(np.uint32), real_b.view(np.uint32))
    # ... and equal to packing the dense rows.
    dense = be.encode(layout)["obs"]
    assert np.array_equal(np.packbits(dense[:, lay.flag_cols] != 0, axis=1), bits_a)
    assert np.array_equal(dense[:, lay.real_cols].view(np.uint32), real_a.view(np.uint32))


def test_encode_validates_its_arguments() -> None:
    be = _engine(mc=0)
    d = MINIMAL_LAYOUT.obs_dim
    with pytest.raises(ValueError, match="layout"):
        be.encode("dense")
    with pytest.raises(ValueError, match="C-contiguous"):
        be.encode("minimal", out=np.zeros((N - 1, d), np.float32))
    with pytest.raises(ValueError, match="encode_mask"):
        be.encode("minimal", encode_mask=np.ones(N - 1, dtype=bool))
    with pytest.raises(ValueError, match="strictly increasing"):
        be.encode("minimal", np.asarray([3, 1], dtype=np.int64), out=np.zeros((N, d), np.float32))
    with pytest.raises(ValueError, match="out of range"):
        be.encode("minimal", np.asarray([N], dtype=np.int64))
    with pytest.raises(ValueError, match="go together"):
        be.encode("minimal", flag_cols=MINIMAL_LAYOUT.flag_cols)
    # The legacy in-place names keep insisting on somewhere to write.
    with pytest.raises(ValueError, match="out=None needs the packed outputs"):
        be.observation_encoded_minimal_into(None)
    nlh = BatchedEngine(4, num_seats=3, variant="nlh_single", sb=5000, starting_stack=1_000_000)
    nlh.reset_batch(np.arange(4, dtype=np.uint64), np.zeros(4, dtype=np.uint8))
    with pytest.raises(RuntimeError, match="PLO-only"):
        nlh.encode("full")
