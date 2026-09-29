"""Shared fixtures for trainer-mode tests (tiny random-init network)."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# The tests live in area folders (engine/, training/, site/, homegame/, gto/,
# ops/ — TEST-039); the helpers they share (bash_tools, cfr_fixtures,
# hg_client_tools + hg_mini_dom.js) stay here, importable from every folder.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# --- UI module (re)import isolation (review 2026-09-20 J3) -------------------
# A fresh app no longer needs a fresh import: `server.create_app()` (BE-007)
# builds one from the environment as it is at that moment — the public layer
# and the home games read their settings when they are installed, and every
# app gets its own database, caches and home-games context. `boot_public_server`
# below does exactly that. This helper stays for a test that genuinely needs a
# module's IMPORT re-executed (something still read at import: the home games'
# lock checks / verifiable-shuffle switch / API rate spec, the trainer's session
# lock timeout …). Popping `sys.modules` alone is NOT enough: `from plo5bp.ui
# import public` resolves through the *package attribute* first, so a second
# reimport in the same pytest process silently got the STALE `public` module
# (old DB, old FREE_HANDS, old middleware) while `server` was fresh — 7
# cross-file failures. Call it on setup AND teardown.
UI_MODULES = (
    "plo5bp.ui.server",
    "plo5bp.ui.public",
    "plo5bp.ui.trainer",
    "plo5bp.ui.homegame",
    # Local-build live capture: its modules bind the server's session /
    # rebuild at import, so a reimported server must get fresh ones too.
    "plo5bp.ui.live",
    "plo5bp.ui.live.state",
    "plo5bp.ui.live.tracking",
    "plo5bp.ui.live.clubgg",
    "plo5bp.ui.live.pokernow",
    "plo5bp.ui.live.routes",
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
                db.close()  # the writer AND the per-thread read connections
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
# (tests/python/site/test_public_free_mode.py).
os.environ.setdefault("PLO5BP_FREE_FOR_ALL", "0")
# Home-game hands are graded by a background thread (one model forward per
# decision). Off for the test session so hundreds of scripted hands do not
# queue work behind the tests; test_homegame_tracking.py turns it on.
os.environ.setdefault("PLO5BP_HOMEGAME_GRADING", "0")

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
def boot_public_server(tmp_path_factory):
    """Factory: ``boot(**env) -> plo5bp.ui.server`` with a fresh PUBLIC app
    built by ``server.create_app()`` (BE-007) against a temp sqlite DB (never
    ``data/``), ``env`` on top of the public test environment. Until the
    requesting test module finishes, that app is the current site:
    ``server.app``, ``server.MODEL`` … name it, and ``plo5bp.ui.public`` /
    ``plo5bp.ui.homegame`` serve it. Then it is closed (home-game workers
    stopped, database closed), the environment restored and the previous site
    — the LOCAL app the other test modules use — made current again. Nothing is
    purged or re-imported.

    ONE boot per test module: the module-scoped fixtures hold on to the app."""
    import importlib

    # Imported BEFORE the environment changes, so its import-time app — the
    # previous site restored at the end — is the local build.
    server = importlib.import_module("plo5bp.ui.server")
    saved: dict[str, str | None] = {}
    booted: list = []

    def boot(**extra_env: str):
        assert not booted, "boot_public_server: one boot per test module"
        booted.append(server.current_site())
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
        booted.append(server.create_app().state.site)
        return server

    yield boot
    try:
        if len(booted) == 2:
            booted[1].close()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if booted:
            server.use_site(booted[0])


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
