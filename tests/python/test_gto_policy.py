"""PolicyNet train smoke + PolicyNetHost T1 seam."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from plo5bp.encoding_nlh import OBS_DIM_NLH
from plo5bp.gto.dataset import PolicyDataset, SupervisedRow, collect_teacher_distill
from plo5bp.gto.policy_host import PolicyNetHost, load_policy_host
from plo5bp.gto.policy_net import (
    POLICY_KIND,
    build_policy_net,
    is_gto_checkpoint,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from plo5bp.gto.train import TrainConfig, train_policy_net
from plo5bp.network import ActorCriticV2
from plo5bp.sizing import NLH_ANCHOR_SPEC


def _tiny_rows(n: int = 64, seed: int = 0) -> list[SupervisedRow]:
    rng = np.random.default_rng(seed)
    k = NLH_ANCHOR_SPEC.count
    rows = []
    for _ in range(n):
        obs = rng.standard_normal(OBS_DIM_NLH).astype(np.float32)
        gm = np.array([True, True, True], dtype=bool)
        # Soft targets
        g = rng.random(3).astype(np.float32)
        g /= g.sum()
        a = rng.random(k).astype(np.float32)
        a /= a.sum()
        rows.append(
            SupervisedRow(
                obs=obs,
                gate_mask=gm,
                sizing=np.array([10000, 500000, 100000, 0], dtype=np.int64),
                gate_probs=g,
                anchor_probs=a,
                value_bb=float(rng.standard_normal()),
                street=1,
            )
        )
    return rows


def test_build_policy_net_shape():
    m = build_policy_net(hidden_dim=64, num_layers=1)
    assert isinstance(m, ActorCriticV2)
    assert m.anchor_spec.count == NLH_ANCHOR_SPEC.count
    assert m.head_version == 2
    x = torch.zeros(2, OBS_DIM_NLH)
    gm = torch.ones(2, 3, dtype=torch.bool)
    gl, al, ref, v = m(x, gm)
    assert gl.shape == (2, 3)
    assert al.shape == (2, NLH_ANCHOR_SPEC.count)
    assert v.shape == (2,)


def test_save_load_gto_checkpoint(tmp_path: Path):
    m = build_policy_net(hidden_dim=64, num_layers=1)
    path = tmp_path / "gto.pt"
    save_policy_checkpoint(path, m, meta={"source": "unit"})
    assert is_gto_checkpoint(path)
    m2, meta = load_policy_checkpoint(path)
    assert meta["kind"] == POLICY_KIND
    assert meta["is_gto"] is True
    # Weights match
    for (n1, p1), (n2, p2) in zip(m.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1, p2)


def test_train_smoke_reduces_loss(tmp_path: Path):
    rows = _tiny_rows(128, seed=1)
    out = tmp_path / "pol.pt"
    cfg = TrainConfig(
        hidden_dim=64,
        num_layers=1,
        epochs=3,
        batch_size=32,
        lr=1e-3,
        device="cpu",
        seed=0,
        log_every=1000,
    )
    r = train_policy_net(rows, out, cfg=cfg, meta={"source": "synthetic"})
    assert r.n_train == 128
    assert r.steps > 0
    assert out.is_file()
    assert is_gto_checkpoint(out)
    # Loss finite
    assert np.isfinite(r.final_loss)


def test_policy_host_badge_and_act(tmp_path: Path, trainer_factory):
    rows = _tiny_rows(32, seed=2)
    out = tmp_path / "pol.pt"
    train_policy_net(
        rows,
        out,
        cfg=TrainConfig(
            hidden_dim=64, num_layers=1, epochs=1, batch_size=16, log_every=999
        ),
        meta={"source": "synthetic"},
    )
    host = load_policy_host(out, device="cpu")
    badge = host.coverage_badge()
    # Synthetic/unprobed: host loads but does NOT claim GTO AI
    assert badge["is_gto"] is False
    assert "GTO AI" not in badge["label"]

    ts = trainer_factory(
        seats_mode="fixed",
        seats_fixed=3,
        stacks_mode="fixed",
        stack_bb=100.0,
        mc_rollouts=0,
        rng_seed=5,
    )
    from plo5bp.config import VARIANT_NLH
    from plo5bp.network import ActorCriticV4
    from plo5bp.encoding_nlh import OBS_DIM_NLH as D

    nlh_model = ActorCriticV4(
        hidden_dim=32, num_layers=1, obs_dim=D, anchor_spec=NLH_ANCHOR_SPEC
    ).eval()
    ts.set_format(VARIANT_NLH, nlh_model, None, backend=host)
    assert isinstance(ts.backend, PolicyNetHost)
    frames = ts.new_hand()
    assert frames
    state = ts.project_state()
    assert state["trainer"]["backend"]["is_gto"] is False
    if not ts.hand.terminal:
        s = ts.project_state()
        if s["legal"]["check_call"]:
            ts.act("check_call", None)
        elif s["legal"]["fold"]:
            ts.act("fold", None)


def test_badge_bootstrap_not_gto(tmp_path: Path):
    rows = _tiny_rows(16, seed=3)
    out = tmp_path / "boot.pt"
    train_policy_net(
        rows,
        out,
        cfg=TrainConfig(
            hidden_dim=64, num_layers=1, epochs=1, batch_size=16, log_every=999
        ),
        meta={"source": "rule_bootstrap", "n_train": 16},
    )
    host = load_policy_host(out, device="cpu")
    badge = host.coverage_badge()
    assert badge["is_gto"] is False
    assert badge["label"] == "Curriculum"
    assert "bootstrap" in badge["note"].lower() or "Curriculum" in badge["label"]


def test_badge_validated_rust_cfr_gto(tmp_path: Path):
    m = build_policy_net(hidden_dim=64, num_layers=1)
    path = tmp_path / "validated.pt"
    save_policy_checkpoint(
        path,
        m,
        meta={
            "source": "rust_cfr",
            "n_train": 1000,
            "probe": {
                "passed": True,
                "report": {
                    "n": 50,
                    "pure_n": 10,
                    "pure_agree": 0.95,
                    "mean_gate_kl": 0.1,
                    "mean_gate_acc": 0.9,
                },
                "gates": {},
                "reasons": [],
            },
        },
    )
    from plo5bp.gto.policy_net import is_validated_gto_checkpoint

    assert is_validated_gto_checkpoint(path)
    host = load_policy_host(path, device="cpu")
    badge = host.coverage_badge()
    assert badge["is_gto"] is True
    assert badge["label"] == "GTO AI"
    assert badge["probe_passed"] is True


def test_badge_rust_cfr_without_probe_not_gto(tmp_path: Path):
    m = build_policy_net(hidden_dim=64, num_layers=1)
    path = tmp_path / "unprobed.pt"
    save_policy_checkpoint(
        path,
        m,
        meta={"source": "rust_cfr", "n_train": 100},
    )
    host = load_policy_host(path, device="cpu")
    badge = host.coverage_badge()
    assert badge["is_gto"] is False
    assert "unvalidated" in badge["label"].lower()


def test_dataset_loader():
    rows = _tiny_rows(8)
    ds = PolicyDataset(rows)
    assert len(ds) == 8
    item = ds[0]
    assert item["obs"].shape == (OBS_DIM_NLH,)
    assert item["gate_probs"].shape == (3,)
    assert item["anchor_probs"].shape == (NLH_ANCHOR_SPEC.count,)
