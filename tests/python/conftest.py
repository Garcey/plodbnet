"""Shared fixtures for trainer-mode tests (tiny random-init network)."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# --- UI module (re)import isolation (review 2026-09-20 J3) -------------------
# The public-build test modules must set env vars BEFORE `plo5bp.ui.server`
# imports (public.py/homegame.py read PLO5BP_* at import time), so their
# module-scoped fixtures purge and reimport the ui modules. Popping
# `sys.modules` alone is NOT enough: `from plo5bp.ui import public` resolves
# through the *package attribute* first, so a second reimport in the same
# pytest process silently got the STALE `public` module (old DB, old
# FREE_HANDS, old middleware) while `server` was fresh — 7 cross-file
# failures. Both fixtures call this helper on setup AND teardown.
UI_MODULES = (
    "plo5bp.ui.server",
    "plo5bp.ui.public",
    "plo5bp.ui.trainer",
    "plo5bp.ui.homegame",
)


def purge_ui_modules(names: tuple[str, ...] = UI_MODULES) -> None:
    """Forget the env-sensitive ui modules so the next import re-executes them.

    Pops each module from ``sys.modules`` AND deletes the matching attribute
    from its parent package. Also quiesces the outgoing instance: stops the
    home-game shot-clock thread and closes the public sqlite connection so a
    purged module cannot keep ticking against (or locking) a temp DB."""
    if "plo5bp.ui.homegame" in names:
        hg = sys.modules.get("plo5bp.ui.homegame")
        shutdown = getattr(hg, "shutdown", None)
        if callable(shutdown):
            shutdown()  # joins the clock thread before the DB goes away
    if "plo5bp.ui.public" in names:
        pub = sys.modules.get("plo5bp.ui.public")
        db = getattr(pub, "DB", None)
        if db is not None:
            try:
                with db._lock:
                    db._conn.close()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
    for name in names:
        sys.modules.pop(name, None)
        parent_name, _, attr = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and attr in vars(parent):
            delattr(parent, attr)


@pytest.fixture(scope="session")
def ui_purge():
    """The purge helper as a fixture (test modules can't reliably
    ``import conftest`` — tests/ocr has its own conftest.py)."""
    return purge_ui_modules


PUBLIC_TEST_ADMIN = "themilesgarcia@icloud.com"
# The site is free-for-all by default (public.FREE_FOR_ALL, 2026-09-22). The
# paywall / quota / Stripe code is still there and still has to work, so the
# test session runs with the paywall ON unless a test opts into free mode
# (tests/python/test_public_free_mode.py).
os.environ.setdefault("PLO5BP_FREE_FOR_ALL", "0")

_PUBLIC_TEST_ENV = {
    "PLO5BP_PUBLIC": "1",
    "PLO5BP_DEV_LOGIN": "1",
    # Starlette's TestClient is client host "testclient" / Host "testserver";
    # the dev login only accepts those when a test opts in (review F4).
    "PLO5BP_DEV_LOGIN_TESTCLIENT": "1",
    "PLO5BP_BASE_URL": "http://127.0.0.1:8770",
    "PLO5BP_ADMIN_EMAILS": PUBLIC_TEST_ADMIN,
    "PLO5BP_HOMEGAME_MAX_TABLES": "1000",
}


@pytest.fixture(scope="module")
def boot_public_server(tmp_path_factory, ui_purge):
    """Factory: ``boot(**env) -> plo5bp.ui.server`` imported fresh in PUBLIC
    mode against a temp sqlite DB (never ``data/``). Env is restored and the
    ui modules purged again when the requesting test module finishes.

    ONE boot per test module: booting again purges (and closes the DB of)
    the previous app, which module-scoped fixtures may still be holding."""
    import importlib

    saved: dict[str, str | None] = {}
    booted: list[bool] = []

    def boot(**extra_env: str):
        assert not booted, "boot_public_server: one boot per test module"
        booted.append(True)
        tmp = tmp_path_factory.mktemp("public_app")
        env = {
            **_PUBLIC_TEST_ENV,
            "PLO5BP_DB": str(tmp / "public.db"),
            "PLO5BP_TRAINER_STATS": str(tmp / "default_stats.json"),
            **extra_env,
        }
        for k, v in env.items():
            saved.setdefault(k, os.environ.get(k))
            os.environ[k] = v
        ui_purge()
        return importlib.import_module("plo5bp.ui.server")

    yield boot
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    ui_purge()


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
