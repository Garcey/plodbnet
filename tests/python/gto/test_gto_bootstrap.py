"""Rule-based bootstrap curriculum (no PPO teacher)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plo5bp.gto.bootstrap import collect_bootstrap, label_node
from plo5bp.gto.policy_net import is_gto_checkpoint
from plo5bp.gto.roots import CLUBGG_NLH_ROOT
from plo5bp.gto.train import TrainConfig, train_policy_net
from plo5bp.env import BombPotEnv


def test_label_node_returns_simplex():
    cfg = CLUBGG_NLH_ROOT.game_config(num_seats=2, stack_bb=100)
    env = BombPotEnv(cfg)
    obs, info = env.reset(1, 0)
    g, ap, v = label_node(info)
    assert g.shape == (3,)
    assert pytest.approx(float(g.sum()), abs=1e-5) == 1.0
    assert ap.shape[0] >= 11
    assert pytest.approx(float(ap.sum()), abs=1e-5) == 1.0
    assert np.isfinite(v)


def test_collect_bootstrap_rows():
    rows = collect_bootstrap(n_decisions=64, seed=0, seats=(2, 3))
    assert len(rows) >= 32
    assert rows[0].obs.shape[0] == 995
    pure = sum(1 for r in rows if float(r.gate_probs.max()) >= 0.85)
    # At least some pure nodes from the rule set
    assert pure >= 1


def test_train_from_bootstrap(tmp_path: Path):
    rows = collect_bootstrap(n_decisions=256, seed=3, seats=(2, 6))
    out = tmp_path / "boot.pt"
    r = train_policy_net(
        rows,
        out,
        cfg=TrainConfig(
            hidden_dim=128,
            num_layers=1,
            epochs=3,
            batch_size=64,
            lr=1e-3,
            log_every=999,
            value_coef=0.05,
        ),
        meta={"source": "rule_bootstrap"},
    )
    assert is_gto_checkpoint(out)
    assert r.final_gate_kl < 1.0  # should fit bootstrap somewhat
