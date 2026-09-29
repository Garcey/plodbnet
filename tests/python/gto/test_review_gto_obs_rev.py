"""Observation-semantics revision plumbing in the GTO pipeline (2026-09-20).

The NLH encoder's VALUES changed (``plo5bp.encoding.OBS_SEMANTICS_REV``, env
``PLO5BP_OBS_REV``: 2 = fixed, 1 = legacy). Everything that embeds an obs or was
fit to one carries the revision; no stamp means revision 1.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

import plo5bp.encoding as encoding
from plo5bp.gto.backend import NodeDist
from plo5bp.gto.dataset import (
    StaleObsCacheError,
    SupervisedRow,
    load_rows_npz,
    save_rows_npz,
)
from plo5bp.gto.obs_rev import (
    UNSTAMPED_OBS_REV,
    current_obs_rev,
    obs_rev_mismatch,
    stamped_obs_rev,
)
from plo5bp.gto.policy_host import PolicyNetHost, load_policy_host
from plo5bp.gto.policy_net import (
    build_policy_net,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from plo5bp.gto.train import TrainConfig, train_policy_net


def _rows(n: int = 8) -> list[SupervisedRow]:
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        g = rng.random(3).astype(np.float32)
        out.append(
            SupervisedRow(
                obs=rng.standard_normal(995).astype(np.float32),
                gate_mask=np.array([True, True, True]),
                sizing=np.array([10_000, 500_000, 100_000, 0], dtype=np.int64),
                gate_probs=g / g.sum(),
                anchor_probs=np.full(12, 1 / 12, dtype=np.float32),
                value_bb=0.0,
                street=3,
            )
        )
    return out


@pytest.fixture
def set_rev(monkeypatch):
    def _set(rev: int) -> None:
        monkeypatch.setattr(encoding, "OBS_SEMANTICS_REV", rev, raising=False)

    return _set


def test_rev_helpers(set_rev):
    set_rev(2)
    assert current_obs_rev() == 2
    assert stamped_obs_rev(None) == stamped_obs_rev({}) == UNSTAMPED_OBS_REV == 1
    assert stamped_obs_rev({"obs_rev": 2}) == 2
    assert obs_rev_mismatch({"obs_rev": 2}) is None
    assert "obs_rev=1" in obs_rev_mismatch({})  # unstamped == legacy
    set_rev(1)
    assert obs_rev_mismatch({}) is None and obs_rev_mismatch({"obs_rev": 2}) is not None


def test_cached_row_bundle_is_not_reused_across_revisions(tmp_path: Path, set_rev):
    set_rev(2)
    path = tmp_path / "rows.npz"
    save_rows_npz(path, _rows())
    assert int(np.load(path)["obs_rev"]) == 2
    assert len(load_rows_npz(path)) == 8
    set_rev(1)
    with pytest.raises(StaleObsCacheError, match="rebuild"):
        load_rows_npz(path)
    assert len(load_rows_npz(path, allow_stale_obs=True)) == 8


def test_unstamped_bundle_counts_as_revision_one(tmp_path: Path, set_rev):
    rows = _rows()
    path = tmp_path / "legacy.npz"
    np.savez_compressed(  # exactly what pre-2026-09-20 save_rows_npz wrote
        path,
        obs=np.stack([r.obs for r in rows]),
        gate_mask=np.stack([r.gate_mask for r in rows]),
        sizing=np.stack([r.sizing for r in rows]),
        gate_probs=np.stack([r.gate_probs for r in rows]),
        anchor_probs=np.stack([r.anchor_probs for r in rows]),
        value_bb=np.zeros(len(rows), dtype=np.float32),
        street=np.full(len(rows), 3, dtype=np.int64),
    )
    set_rev(2)
    with pytest.raises(StaleObsCacheError):
        load_rows_npz(path)
    set_rev(1)
    back = load_rows_npz(path)
    assert len(back) == 8 and back[0].value_mask is True and back[0].prov.root_id == ""


def test_training_stamps_the_revision_on_the_checkpoint(tmp_path: Path, set_rev):
    set_rev(2)
    ckpt = tmp_path / "p.pt"
    train_policy_net(
        _rows(), ckpt,
        cfg=TrainConfig(hidden_dim=16, num_layers=1, epochs=1, log_every=10**9),
        meta={"obs_rev": 99},  # derived, not asserted
    )
    _, meta = load_policy_checkpoint(ckpt)
    assert meta["obs_rev"] == 2


def test_loader_and_host_warn_loudly_on_a_revision_mismatch(
    tmp_path: Path, set_rev, caplog
):
    set_rev(2)
    legacy = tmp_path / "legacy.pt"  # no stamp ⇒ trained on revision-1 obs
    save_policy_checkpoint(
        legacy, build_policy_net(hidden_dim=16, num_layers=1), meta={"source": "rust_cfr"}
    )
    with caplog.at_level(logging.WARNING):
        host = load_policy_host(legacy)
    text = caplog.text
    assert "observation semantics revision mismatch" in text
    assert "PLO5BP_OBS_REV=1" in text
    badge = host.coverage_badge()
    assert badge["obs_rev"] == 1 and "mismatch" in badge["obs_rev_mismatch"]

    caplog.clear()
    current = tmp_path / "current.pt"
    save_policy_checkpoint(
        current, build_policy_net(hidden_dim=16, num_layers=1),
        meta={"source": "rust_cfr", "obs_rev": 2},
    )
    with caplog.at_level(logging.WARNING):
        ok = load_policy_host(current)
    assert "mismatch" not in caplog.text
    assert ok.coverage_badge()["obs_rev_mismatch"] is None
    # a host built directly from meta warns too
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        PolicyNetHost(
            model=build_policy_net(hidden_dim=16, num_layers=1),
            device=torch.device("cpu"),
            meta={"source": "rust_cfr", "obs_rev": 1},
        )
    assert "revision mismatch" in caplog.text


def test_node_dist_carries_the_unclamped_brackets():
    raw = {
        "head_version": 2, "gate_probs": [0.1, 0.6, 0.3], "rec_gate": 1, "rec_chips": 0,
        "value_bb": 0.0, "min_chips": 10_000, "max_chips": 200_000,
        "anchor_probs": [1.0] + [0.0] * 11, "anchor_chips": [10_000] * 12,
        "anchor_legal": [True] + [False] * 11, "anchor_lo": [10_000] * 12,
        "anchor_hi": [10_000] * 12, "refine_ok": [False] * 12,
        "refine_params": [[1.0, 1.0]] * 10, "rec_anchor": 0, "pot_ref_chips": 100_000,
        "anchor_lo_raw": list(range(12)), "anchor_hi_raw": list(range(100, 112)),
    }
    out = NodeDist.from_dict(raw).as_dict()
    assert out["anchor_lo_raw"] == list(range(12))
    assert out["anchor_hi_raw"] == list(range(100, 112))
    # absent on older producers → absent (not invented) on the way out
    legacy = {k: v for k, v in raw.items() if not k.endswith("_raw")}
    nd = NodeDist.from_dict(legacy)
    assert nd.anchor_lo_raw is None and nd.anchor_hi_raw is None
    assert "anchor_lo_raw" not in nd.as_dict()
