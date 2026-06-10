"""Shared fixtures for trainer-mode tests (tiny random-init network)."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture()
def trainer_factory(tmp_path):
    """Build a TrainerSession around a tiny random-init ActorCritic with
    stats persisted under tmp_path. Pass TrainerSettings overrides as
    kwargs; `rng_seed` pins the session's deal RNG."""
    from plo5bp.network import ActorCritic
    from plo5bp.ui.trainer import TrainerSession, TrainerSettings

    def make(rng_seed: int = 123, stats_name: str = "stats.json", **settings):
        torch.manual_seed(0)
        model = ActorCritic(hidden_dim=32, num_layers=1).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        ts = TrainerSession(
            model, torch.device("cpu"), stats_path=tmp_path / stats_name
        )
        if settings:
            ts.settings = TrainerSettings(
                **{**ts.settings.model_dump(), **settings}
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
