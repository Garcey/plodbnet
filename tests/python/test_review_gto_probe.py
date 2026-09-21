"""Review 2026-09-20 D4 — the holdout probe must be able to say NO.

The old probe scored gates only, passed a uniform-over-legal policy (KL ≈ 0.35
under a 0.50 cap), passed NaN (``nan > max`` is False), skipped the pure-node
gate when ``pure_n == 0``, put both step-7 holdout roots at SPR 3, and let a
checkpoint be "validated" on its own training labels.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from plo5bp.gto.cfr_api import RootSpec, SolveConfig, rust_cfr_available, solve
from plo5bp.gto.cfr_batch import expand_river_grid, expand_river_spr_grid
from plo5bp.gto.cfr_export import strategy_to_labels
from plo5bp.gto.labels import make_smoke_label, write_jsonl
from plo5bp.gto.obs_from_label import labels_to_supervised_rows
from plo5bp.gto.policy_net import build_policy_net, save_policy_checkpoint
from plo5bp.gto.probe import (
    ProbeGates,
    ProbeReport,
    _node_key,
    evaluate_probe_gates,
    probe_checkpoint,
    probe_policy_vs_labels,
    root_disjointness_problem,
)
from plo5bp.gto.teacher import root_stratum, split_root_ids, spr_bucket
from plo5bp.gto.train import TrainConfig, train_policy_net

REPO = Path(__file__).resolve().parents[2]
K = 12


class _LookupNet(torch.nn.Module):
    """Policy defined by a table obs → (gate probs, anchor probs)."""

    def __init__(self, table: dict[bytes, tuple[np.ndarray, np.ndarray]]):
        super().__init__()
        self.table = table

    def forward(self, obs, gm):
        gates, anchors = [], []
        for row in obs.cpu().numpy():
            g, a = self.table[row.tobytes()]
            gates.append(np.log(np.clip(g, 1e-9, None)))
            anchors.append(np.log(np.clip(a, 1e-9, None)))
        gl = torch.tensor(np.stack(gates), dtype=torch.float32).masked_fill(~gm, -1e9)
        al = torch.tensor(np.stack(anchors), dtype=torch.float32)
        return gl, al, None, torch.zeros(obs.shape[0])


class _UniformNet(torch.nn.Module):
    def forward(self, obs, gm):
        b = obs.shape[0]
        return torch.zeros(b, 3).masked_fill(~gm, -1e9), torch.zeros(b, K), None, torch.zeros(b)


@pytest.fixture(scope="module")
def river_holdout():
    if not rust_cfr_available():
        pytest.skip("extension not built")
    root = RootSpec(
        street=3, pot_bb=10.0, effective_stack_bb=20.0, board=[5, 7, 30, 44, 47],
        raise_sizes_pm=[500, 1000], root_id="probe_holdout",
    )
    rep = solve(
        root,
        SolveConfig.teacher(max_iterations=1200, seed=3, target_exploitability_bb=0.0,
                            thread_num=2, poll_every=10**6),
    )
    labels = strategy_to_labels(rep.as_dict(), min_visit_mass=1.0, teacher_max_expl_bb=1.0)
    rows = labels_to_supervised_rows(labels)
    assert len(rows) == len(labels) > 500
    return labels, rows


def _card_blind_table(labels, rows):
    groups: dict[tuple, list[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(_node_key(lab), []).append(i)
    table = {}
    for idx in groups.values():
        g = np.mean([rows[i].gate_probs for i in idx], axis=0)
        w = np.array([rows[i].gate_probs[2] for i in idx])
        a = np.sum([rows[i].anchor_probs * w[j] for j, i in enumerate(idx)], axis=0)
        a = a / a.sum() if a.sum() > 0 else np.full(K, 1.0 / K)
        for i in idx:
            table[rows[i].obs.tobytes()] = (g, a)
    return table


def test_uniform_over_legal_policy_fails(river_holdout):
    labels, _rows = river_holdout
    rep = probe_policy_vs_labels(_UniformNet(), labels)
    assert rep.mean_gate_kl == pytest.approx(rep.uniform_gate_kl, abs=1e-6)
    assert 0.2 < rep.mean_gate_kl < 0.5  # review: 0.34–0.36 — UNDER the old 0.50 cap
    res = evaluate_probe_gates(rep, ProbeGates(min_n=50))
    assert not res.passed
    assert any("mean_gate_kl" in r for r in res.reasons)
    assert any("mean_anchor_kl" in r for r in res.reasons)
    # the old gates (KL <= 0.50 only, with no pure canaries) would have passed it
    assert rep.mean_gate_kl < 0.50


def test_card_blind_per_node_policy_fails(river_holdout):
    labels, rows = river_holdout
    rep = probe_policy_vs_labels(_LookupNet(_card_blind_table(labels, rows)), labels)
    assert rep.mean_gate_kl == pytest.approx(rep.card_blind_gate_kl, rel=1e-3)
    assert rep.pure_agree == pytest.approx(rep.card_blind_pure_agree, abs=1e-9)
    assert rep.mean_gate_kl < 0.50  # also under the old cap
    res = evaluate_probe_gates(rep, ProbeGates(min_n=50))
    assert not res.passed
    assert any("card-blind" in r for r in res.reasons)


def test_a_policy_that_matches_the_teacher_passes(river_holdout):
    labels, rows = river_holdout
    oracle = {r.obs.tobytes(): (r.gate_probs, r.anchor_probs) for r in rows}
    rep = probe_policy_vs_labels(_LookupNet(oracle), labels)
    res = evaluate_probe_gates(rep, ProbeGates(min_n=50))
    assert res.passed, res.reasons
    assert rep.mean_gate_kl < 1e-4 and rep.mean_anchor_kl < 1e-4
    assert rep.jam_freq_gap < 1e-4 and rep.pure_agree == 1.0
    assert rep.jam_freq_target > 0.05  # jams exist in this tree


def test_right_gates_wrong_sizing_fails(river_holdout):
    """The pre-D1 net: raise FREQUENCY exactly right, every raise the MINIMUM
    size (its jam targets had been masked away). Anchor 0 is always legal; it
    is the jam only where min == max, i.e. where there is no size to choose."""
    labels, rows = river_holdout
    min_raise = np.zeros(K, dtype=np.float32)
    min_raise[0] = 1.0
    table = {r.obs.tobytes(): (r.gate_probs, min_raise) for r in rows}
    rep = probe_policy_vs_labels(_LookupNet(table), labels)
    assert rep.mean_gate_kl < 1e-4  # the gates-only probe saw nothing wrong
    assert rep.pure_agree == 1.0
    res = evaluate_probe_gates(rep, ProbeGates(min_n=50))
    assert not res.passed
    assert any("mean_anchor_kl" in r for r in res.reasons)
    assert any("jam_freq_gap" in r for r in res.reasons)
    assert rep.jam_freq_model < rep.jam_freq_target - 0.05


def test_nan_never_passes():
    rep = ProbeReport(n=5000, pure_n=0, pure_agree=None, mean_gate_kl=float("nan"),
                      mean_gate_acc=0.0)
    assert not evaluate_probe_gates(rep, ProbeGates(min_n=50)).passed
    rep2 = ProbeReport(n=5000, pure_n=300, pure_agree=1.0, mean_gate_kl=float("nan"),
                       mean_gate_acc=0.0, n_raise_rows=10, mean_anchor_kl=float("nan"),
                       mean_jam_gap=float("nan"), jam_freq_gap=float("nan"))
    res = evaluate_probe_gates(rep2, ProbeGates(min_n=50))
    assert not res.passed and len(res.reasons) >= 3
    # end to end: a net with NaN weights fails instead of "agreeing" on FOLD
    net = build_policy_net(hidden_dim=16, num_layers=1)
    with torch.no_grad():
        net.gate_head.weight.fill_(float("nan"))
    labels = [make_smoke_label(seed=i, trash_fold=True) for i in range(6)]
    rep3 = probe_policy_vs_labels(net, labels)
    assert rep3.n_nonfinite == 6 and rep3.pure_agree == 0.0
    assert math.isnan(rep3.mean_gate_kl)
    res3 = evaluate_probe_gates(rep3, ProbeGates())
    assert not res3.passed and any("non-finite" in r for r in res3.reasons)


def test_no_pure_nodes_is_a_failure_not_a_skip():
    base = dict(n=200, pure_agree=None, mean_gate_kl=0.05, mean_gate_acc=0.9,
                n_raise_rows=50, mean_anchor_kl=0.02, mean_jam_gap=0.01, jam_freq_gap=0.0)
    res = evaluate_probe_gates(ProbeReport(pure_n=0, **base), ProbeGates(min_pure_n=0))
    assert not res.passed and any("UNVERIFIED" in r for r in res.reasons)
    assert evaluate_probe_gates(
        ProbeReport(pure_n=0, **base), ProbeGates(allow_no_pure=True)
    ).passed
    no_sizing = {**base, "n_raise_rows": 0, "mean_anchor_kl": None}
    res2 = evaluate_probe_gates(
        ProbeReport(pure_n=5, **{**no_sizing, "pure_agree": 1.0}), ProbeGates()
    )
    assert not res2.passed and any("sizing gates UNVERIFIED" in r for r in res2.reasons)


# --- holdout must be a holdout ---------------------------------------------------


def _teacher(root: str, n: int = 16):
    out = []
    for i in range(n):
        lab = make_smoke_label(seed=i, trash_fold=bool(i % 2))
        lab.source, lab.root_name = "rust_cfr", root
        lab.notes = {**lab.notes, "exploitability_bb": 0.4, "expl_verified": True,
                     "teacher_max_expl_bb": 1.0}
        out.append(lab)
    return out


GATES = ProbeGates(min_n=4, min_pure_agree=0.75, max_mean_gate_kl=0.8, min_pure_n=2)


def test_self_fit_is_refused(tmp_path: Path):
    labels = _teacher("train_root")
    ckpt = tmp_path / "fit.pt"
    train_policy_net(
        labels_to_supervised_rows(labels), ckpt,
        cfg=TrainConfig(hidden_dim=64, num_layers=1, epochs=40, batch_size=16,
                        lr=3e-3, log_every=10**9, seed=0),
    )
    same = tmp_path / "same.jsonl"
    write_jsonl(same, labels)
    res = probe_checkpoint(ckpt, same, gates=GATES)
    assert not res.passed
    assert any("intersects the training roots" in r for r in res.reasons)
    other = tmp_path / "other.jsonl"
    write_jsonl(other, _teacher("holdout_root"))
    assert probe_checkpoint(ckpt, other, gates=GATES).passed


def test_checkpoint_without_recorded_roots_cannot_be_verified(tmp_path: Path):
    ckpt = tmp_path / "legacy.pt"  # what every pre-2026-09-20 checkpoint looks like
    save_policy_checkpoint(
        ckpt, build_policy_net(hidden_dim=32, num_layers=1), meta={"source": "rust_cfr"}
    )
    hold = tmp_path / "h.jsonl"
    write_jsonl(hold, _teacher("holdout_root"))
    res = probe_checkpoint(ckpt, hold, gates=GATES)
    assert not res.passed
    assert any("no verifiable train_root_ids" in r for r in res.reasons)
    assert root_disjointness_problem(
        {"train_root_ids": ["a"], "train_roots_known": True}, ["b"]
    ) is None
    assert "carry no root ids" in root_disjointness_problem(
        {"train_root_ids": ["a"], "train_roots_known": True}, []
    )


# --- stratified, PYTHONHASHSEED-independent split ---------------------------------


def test_spr_buckets_separate_the_step7_grid_points():
    assert len({spr_bucket(s) for s in (1.0, 2.0, 3.0, 5.0)}) == 4
    assert spr_bucket(None) == spr_bucket(0.0) == "spr?"


def test_every_spr_stratum_with_two_roots_gets_a_holdout_root():
    jobs = expand_river_spr_grid(n_boards=6, seed=7)
    for j in jobs:  # the ids the real campaign directory uses
        j.job_id = j.legacy_job_id
    ids = [j.job_id for j in jobs]
    strata = {j.job_id: root_stratum(j.root) for j in jobs}
    _old_train, old_hold = split_root_ids(ids)
    # the unstratified split the review flagged: both holdout roots at SPR 3
    assert old_hold == ["s3_spr3p0_s7_b0", "s3_spr3p0_s7_b1"]
    train, hold = split_root_ids(ids, strata=strata)
    assert set(train).isdisjoint(hold) and set(train) | set(hold) == set(ids)
    for key in set(strata.values()):
        members = {i for i in ids if strata[i] == key}
        assert len(members) == 6
        assert members & set(hold), f"no holdout root in {key}"
        assert members & set(train), f"no train root in {key}"
    assert (train, hold) == split_root_ids(list(reversed(ids)), strata=strata)
    # singleton strata fall back to the plain hash rule (nothing to force)
    solo = split_root_ids(["only"], strata={"only": "s3_n2_spr0"})
    assert sorted(solo[0] + solo[1]) == ["only"]


def test_split_does_not_depend_on_pythonhashseed():
    code = (
        "import json;"
        "from plo5bp.gto.cfr_batch import expand_river_spr_grid;"
        "from plo5bp.gto.teacher import root_stratum, split_root_ids;"
        "jobs = expand_river_spr_grid(n_boards=6, seed=7);"
        "print(json.dumps(split_root_ids([j.job_id for j in jobs],"
        " strata={j.job_id: root_stratum(j.root) for j in jobs})))"
    )
    outs = set()
    for seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(REPO / "python")}
        got = subprocess.run(
            [sys.executable, "-W", "ignore", "-c", code], env=env, capture_output=True,
            text=True, check=True,
        )
        outs.add(got.stdout.strip().splitlines()[-1])
    assert len(outs) == 1


def test_grid_split_is_stable_across_solve_configs():
    a = expand_river_grid(n_roots=8, seed=3, iters=100, size_preset="micro")
    b = expand_river_grid(n_roots=8, seed=3, iters=40_000, size_preset="micro")
    assert split_root_ids([j.job_id for j in a]) == split_root_ids([j.job_id for j in b])
