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


def _teacher_provenance() -> dict:
    """``label_provenance`` as derive_training_provenance writes it for
    verified rust_cfr labels under the 1.0 bb cap."""
    return {
        "derived_from_records": True,
        "sources": {"rust_cfr_river": 1000},
        "n_rows": 1000,
        "n_unverified_expl": 0,
        "max_label_expl_bb": 0.8,
        "teacher_max_expl_bb": 1.0,
    }


def test_badge_validated_rust_cfr_gto(tmp_path: Path):
    """rust_cfr source + record-derived provenance (train AND holdout) + probe.

    (review 2026-09-20 F6) The probe pass alone used to be enough; the
    negative cases live in test_review_gto_provenance.py.
    """
    m = build_policy_net(hidden_dim=64, num_layers=1)
    path = tmp_path / "validated.pt"
    save_policy_checkpoint(
        path,
        m,
        meta={
            "source": "rust_cfr",
            "n_train": 1000,
            "label_provenance": _teacher_provenance(),
            "probe": {
                "passed": True,
                "report": {
                    "n": 50,
                    "pure_n": 10,
                    "pure_agree": 0.95,
                    "mean_gate_kl": 0.1,
                    "mean_gate_acc": 0.9,
                    "holdout_provenance": {**_teacher_provenance(), "problem": None},
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


def _illegal_anchor_row(base: SupervisedRow) -> SupervisedRow:
    """A copy of ``base`` whose teacher parks raise mass on a grid-ILLEGAL anchor."""
    import dataclasses

    import torch

    from plo5bp.sizing import anchor_grid_torch

    sizing = np.array([10000, 60000, 100000, 0], dtype=np.int64)  # max raise 0.6 pot
    legal = anchor_grid_torch(torch.from_numpy(sizing[None]), NLH_ANCHOR_SPEC).legal[0].numpy()
    assert not legal.all()
    a = np.zeros(NLH_ANCHOR_SPEC.count, dtype=np.float32)
    a[int(np.flatnonzero(~legal)[0])] = 1.0
    return dataclasses.replace(
        base, sizing=sizing, anchor_probs=a,
        gate_probs=np.array([0.0, 0.2, 0.8], dtype=np.float32),
    )


def test_illegal_teacher_mass_is_refused_before_any_optimizer_step(tmp_path, monkeypatch):
    """(TOOL-049) The bad row sits LAST; it used to raise only when its batch came
    up — after every earlier batch had already stepped the optimizer."""
    import torch

    from plo5bp.gto.labels import IllegalTeacherMassError

    rows = _tiny_rows(256, seed=2)
    rows.append(_illegal_anchor_row(rows[0]))
    steps = []
    real_step = torch.optim.Adam.step

    def counting_step(self, *a, **k):
        steps.append(1)
        return real_step(self, *a, **k)

    monkeypatch.setattr(torch.optim.Adam, "step", counting_step)
    cfg = TrainConfig(hidden_dim=32, num_layers=1, epochs=1, batch_size=16, log_every=10**9)
    with pytest.raises(IllegalTeacherMassError):
        train_policy_net(rows, tmp_path / "bad.pt", cfg=cfg, meta={"source": "synthetic"})
    assert steps == [] and not (tmp_path / "bad.pt").exists()


def test_reported_metrics_are_epoch_means(tmp_path):
    """(TOOL-049) final_* = the last epoch's row-weighted means, not one batch."""
    rows = _tiny_rows(96, seed=3)
    cfg = TrainConfig(hidden_dim=32, num_layers=1, epochs=3, batch_size=16, log_every=10**9)
    r = train_policy_net(rows, tmp_path / "m.pt", cfg=cfg, meta={"source": "synthetic"})
    assert [e["epoch"] for e in r.epochs] == [0.0, 1.0, 2.0]
    last = r.epochs[-1]
    assert (r.final_loss, r.final_gate_kl, r.final_anchor_kl) == (
        last["loss"], last["gate_kl"], last["anchor_kl"]
    )
    assert r.epochs[-1]["gate_kl"] <= r.epochs[0]["gate_kl"] + 1e-6  # it learns
    from plo5bp.gto.policy_net import load_policy_checkpoint

    _, meta = load_policy_checkpoint(tmp_path / "m.pt")
    assert meta["final_loss"] == r.final_loss and len(meta["epoch_means"]) == 3
