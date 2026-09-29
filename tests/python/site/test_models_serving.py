"""plo5bp.ui.models — how the server reads and serves checkpoints:

- safe unpickling: a checkpoint that needs code execution is refused in the
  public build, loaded with a loud warning in the (trusted) local build
  (SEC-023);
- each checkpoint is read ONCE for its actor and critic, and the trainer
  shares the server's GTO host instead of loading it again (PERF-023).
"""

from __future__ import annotations

import logging

import pytest
import torch

from plo5bp.network import ActorCriticV2, CentralCritic
from plo5bp.ui import models


class NotData:
    """A pickled class: loading it needs code execution."""

    def __init__(self) -> None:
        self.x = 1


def _plain_ckpt(path, **extra):
    torch.manual_seed(0)
    net = ActorCriticV2(hidden_dim=8, num_layers=1)
    critic = CentralCritic(hidden_dim=8, num_blocks=1)
    torch.save({"model": net.state_dict(), "critic": critic.state_dict(), "head_version": 2,
                "config": {"hidden_dim": 8, "num_layers": 1}, **extra}, path)
    return path


def test_plain_checkpoints_load_with_weights_only(tmp_path):
    ckpt = models.read_checkpoint(_plain_ckpt(tmp_path / "ok.pt"), trust=False)
    assert "model" in ckpt


def test_code_executing_checkpoints_are_refused_unless_trusted(tmp_path, monkeypatch, caplog):
    path = _plain_ckpt(tmp_path / "evil.pt", payload=NotData())
    with pytest.raises(models.UnsafeCheckpoint):
        models.read_checkpoint(path, trust=False)
    # Public build: refused by default -> the format serves a flagged placeholder.
    monkeypatch.setenv("PLO5BP_PUBLIC", "1")
    monkeypatch.delenv("PLO5BP_TRUST_CHECKPOINTS", raising=False)
    model, loaded = models.load_model("plo5_double_bomb", path)
    assert loaded is False
    # Explicitly trusted: loads, loudly.
    monkeypatch.setenv("PLO5BP_TRUST_CHECKPOINTS", "1")
    with caplog.at_level(logging.WARNING, logger="plo5bp.ui"):
        ckpt = models.read_checkpoint(path)
    assert isinstance(ckpt["payload"], NotData)
    assert "FULL unpickling" in caplog.text


def test_an_entry_reads_its_checkpoint_once(tmp_path, monkeypatch):
    path = _plain_ckpt(tmp_path / "once.pt")
    reads: list = []
    real = models.read_checkpoint
    monkeypatch.setattr(models, "read_checkpoint", lambda p, **k: reads.append(p) or real(p, **k))
    entry = models.build_entry("plo5_double_bomb", path)
    assert entry["loaded"] and entry["critic"] is not None
    assert reads == [path]
    assert entry["sha256"] == models.file_facts(path)["sha256"]


def test_the_trainer_router_reuses_the_servers_gto_host(monkeypatch):
    from plo5bp.ui import trainer as T

    def boom(*a, **k):
        raise AssertionError("the teacher was loaded a second time")

    monkeypatch.setattr(T, "try_load_gto_host", boom)
    monkeypatch.setenv("PLO5BP_GTO_CHECKPOINT", "some/teacher.pt")
    monkeypatch.delenv("PLO5BP_PUBLIC", raising=False)
    host = object()
    router = T.create_trainer_router(ActorCriticV2(hidden_dim=8, num_layers=1),
                                     torch.device("cpu"), gto_host=host)
    assert router is not None
