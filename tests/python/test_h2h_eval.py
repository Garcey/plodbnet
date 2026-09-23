"""scripts/h2h_eval.py (the network-size sweep's strength test, 2026-09-23).

Duplicate deals with the seats swapped between the two passes make a match
of a model against ITSELF zero-sum per deal pair only in expectation (both
passes sample their own actions), but the estimate must be unbiased and the
report well-formed. Pinned: a self-match stays within a few standard errors
of zero, the JSONL line carries the per-tier breakdown, and a checkpoint
stamped with another observation revision is refused.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from plo5bp import encoding
from plo5bp.encoding import OBS_DIM_MINIMAL
from plo5bp.network import ActorCriticV5

REPO = Path(__file__).resolve().parents[2]


def _h2h():
    spec = importlib.util.spec_from_file_location("_h2h", REPO / "scripts" / "h2h_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_ckpt(path: Path, seed: int, obs_rev: int) -> None:
    torch.manual_seed(seed)
    model = ActorCriticV5(
        hidden_dim=16, obs_dim=OBS_DIM_MINIMAL, num_layers=3, torso_layernorm=True
    )
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"hidden_dim": 16, "num_layers": 3, "obs_mode": "minimal"},
            "obs_rev": obs_rev,
            "update_counter": seed,
            "variant": "plo5_double_bomb",
        },
        path,
    )


def test_self_match_is_unbiased_and_reported(tmp_path, monkeypatch) -> None:
    ck = tmp_path / "a.pt"
    _tiny_ckpt(ck, 3, int(encoding.OBS_SEMANTICS_REV))
    out = tmp_path / "h2h.jsonl"
    monkeypatch.setattr(
        sys, "argv",
        ["h2h_eval.py", str(ck), str(ck), "--deals", "256", "--configs-per-tier", "2",
         "--device", "cpu", "--seed", "5", "--out", str(out)],
    )
    _h2h().main()
    rep = json.loads(out.read_text().strip().splitlines()[-1])
    assert set(rep["tiers"]) == {"clubgg", "clubgg_deep", "deep"}
    assert rep["pairs"] > 0 and rep["se"] > 0
    assert abs(rep["edge_bb"]) < 4 * rep["se"], rep


def test_refuses_other_obs_revision(tmp_path, monkeypatch) -> None:
    ck = tmp_path / "old.pt"
    _tiny_ckpt(ck, 1, int(encoding.OBS_SEMANTICS_REV) + 7)
    monkeypatch.setattr(
        sys, "argv",
        ["h2h_eval.py", str(ck), str(ck), "--deals", "8", "--configs-per-tier", "1",
         "--device", "cpu", "--out", str(tmp_path / "x.jsonl")],
    )
    with pytest.raises(SystemExit, match="obs rev"):
        _h2h().main()
