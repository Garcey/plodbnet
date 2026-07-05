"""Shared fixtures for trainer-mode tests (tiny random-init network)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Isolate trainer-stats persistence from the user's REAL
# checkpoints/trainer_stats.json for the whole test session. This must run
# before plo5bp.ui.server is imported (its module-global TrainerSession
# resolves PLO5BP_TRAINER_STATS at import time). Without it the trainer/UI
# tests read AND overwrite the real lifetime-stats file — e.g.
# test_trainer_settings_are_per_format persists a 200bb NLH stack into it and
# then self-fails on the next run's "== 100" default assertion.
_TEST_TRAINER_STATS = Path(tempfile.gettempdir()) / "plo5bp_test_trainer_stats.json"
try:
    _TEST_TRAINER_STATS.unlink()
except FileNotFoundError:
    pass
os.environ["PLO5BP_TRAINER_STATS"] = str(_TEST_TRAINER_STATS)


@pytest.fixture()
def trainer_factory(tmp_path):
    """Build a TrainerSession around a tiny random-init ActorCritic with
    stats persisted under tmp_path. Pass TrainerSettings overrides as
    kwargs; `rng_seed` pins the session's deal RNG."""
    from plo5bp.network import ActorCritic
    from plo5bp.ui.trainer import TrainerSession, TrainerSettings

    def make(rng_seed: int = 123, stats_name: str = "stats.json",
             model_cls=ActorCritic, **settings):
        torch.manual_seed(0)
        model = model_cls(hidden_dim=32, num_layers=1).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        ts = TrainerSession(
            model, torch.device("cpu"), stats_path=tmp_path / stats_name
        )
        if settings:
            # set_settings (not a bare `ts.settings =`) so the override also
            # lands in settings_by_variant — that's what _persist serializes,
            # so a bare assignment round-trips the DEFAULT and defeats any test
            # that persists then reloads (test_settings_persist_with_stats).
            ts.set_settings(
                TrainerSettings(**{**ts.settings.model_dump(), **settings})
            )
        ts.rng = np.random.default_rng(rng_seed)
        return ts

    return make


@pytest.fixture()
def play_to_terminal():
    """Drive a TrainerSession hand to terminal with simple hero choices
    (prefer check/call, then fold, then min-raise)."""

    def run(ts, max_steps: int = 80) -> None:
        steps = 0
        while ts.hand is not None and not ts.hand.terminal and steps < max_steps:
            s = ts.project_state()
            legal = s["legal"]
            if legal["check_call"]:
                ts.act("check_call", None)
            elif legal["fold"]:
                ts.act("fold", None)
            else:
                ts.act("raise", s["raise_bounds"]["min_chips"])
            steps += 1
        assert ts.hand is not None and ts.hand.terminal, "hand did not finish"

    return run
