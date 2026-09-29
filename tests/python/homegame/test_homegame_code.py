"""Home games — code structure (improvements backlog, 2026-09-28).

HGB-023: the home games use the env's public accessors, never the engine's
private ``_rs`` handle, and those accessors are exact pass-throughs.
HGB-024: the game-specific bits live in one place each — one "does this sealed
deck fit the table" check, one runout-equity loop for every game.
HGB-006: the parts split out of homegame.py keep ``homegame.X`` working —
importing, calling and patching it — and share one ``HomeGames`` context.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pytest

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.config import VARIANT_PLO5, VARIANT_PLO6, VARIANT_PLO67, GameConfig
from plo5bp.env import BombPotEnv

UI = Path(__file__).resolve().parents[3] / "python" / "plo5bp" / "ui"


def _home_game_sources() -> list[Path]:
    """homegame.py and whatever is split out of it (``homegame_*.py``, a package)."""
    out = [p for p in UI.glob("homegame*.py")]
    pkg = UI / "homegame"
    if pkg.is_dir():
        out += list(pkg.glob("*.py"))
    return out


def test_the_home_games_never_reach_into_the_engine_handle():
    for path in _home_game_sources():
        src = path.read_text(encoding="utf-8")
        assert not re.search(r"\._rs\b", src), f"{path.name} uses env._rs (HGB-023: use the env's accessors)"


def _play(env: BombPotEnv, rng: random.Random, limit: int = 200) -> None:
    """Random legal actions until the hand is over."""
    for _ in range(limit):
        if env.is_terminal():
            return
        gates = [GATE_FOLD, GATE_CHECK_CALL, GATE_RAISE]
        rng.shuffle(gates)
        for g in gates:
            try:
                env.step_hybrid(g, 10_000 if g == GATE_RAISE else 0)
                break
            except Exception:  # noqa: BLE001 — that gate was not legal here
                continue
    raise AssertionError("hand did not end")


@pytest.mark.parametrize("variant", [VARIANT_PLO5, VARIANT_PLO6, VARIANT_PLO67])
def test_the_env_accessors_are_exact_pass_throughs(variant):
    try:
        env = BombPotEnv(GameConfig(num_seats=4, starting_stack=0, starting_stacks=(40_000,) * 4,
                                    ante=3_000, bb=10_000, variant=variant),
                         ev_runout_samples=0, obs_mode="minimal")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"engine cannot deal {variant}: {e}")
    rng = random.Random(7)
    for seed in range(6):
        env.reset(1000 + seed, seed % 4)
        assert env.observation_dict() == dict(env._rs.observation_dict(skip_outcome_mc=True))
        _play(env, rng)
        pays = env.payouts()
        assert pays == [int(x) for x in env._rs.payouts()] and sum(pays) == 0
        assert env.observation_dict() == dict(env._rs.observation_dict(skip_outcome_mc=True))
        if variant == VARIANT_PLO67:
            burns = env.all_burns()
            assert len(burns) == 3 and burns == [int(c) for c in env._rs.all_burns()]
            for seat in range(4):
                counts = [env.hole_count_on(seat, st) for st in (1, 2, 3)]
                assert counts == [int(env._rs.hole_count_on(seat, st)) for st in (1, 2, 3)]
                assert 4 <= counts[0] <= counts[1] <= counts[2] <= 7
        else:
            assert env.all_burns() == []


@pytest.fixture(scope="module")
def hg(boot_public_server):
    boot_public_server()
    return sys.modules["plo5bp.ui.homegame"]


@pytest.fixture(scope="module")
def cast_code(hg):
    from starlette.testclient import TestClient

    app = sys.modules["plo5bp.ui.server"].app

    def login(email):
        c = TestClient(app, raise_server_exceptions=False)
        assert c.get("/auth/dev", params={"email": email}).status_code == 200
        return c

    adm = login("themilesgarcia@icloud.com")
    players = [login(f"code{i}@example.com") for i in range(2)]
    ids = {u["email"]: u["id"] for u in adm.get("/admin/api/users").json()["users"]}
    for i in range(2):
        assert adm.post("/admin/api/games_access", json={"user_id": ids[f"code{i}@example.com"],
                                                         "action": "grant"}).status_code == 200
    return {"hg": hg, "p": players}


def _ok_json(r):
    assert r.status_code == 200, r.text
    return r.json()


def test_one_check_says_whether_a_sealed_deck_fits_the_table(hg):
    from plo5bp.ui import fairdeal

    t = hg.LiveTable(game_id="g", host_user_id=1, name="x", num_seats=6, sb_cents=50, bb_cents=100,
                     ante_cents=300, default_buyin_cents=4000, status="open", button=0, hand_no=0,
                     seats=[None] * 6)
    assert hg._seal_fits(t, hg._fair_seal(t, 1, 1))
    assert not hg._seal_fits(t, fairdeal.SealedDeck.create("g:1:1:x", 5, hole=5))  # resized since
    assert not hg._seal_fits(t, fairdeal.SealedDeck.create("g:1:1:x", 6, hole=6))  # another game's slots
    t.variant = "plo6"
    assert hg._seal_fits(t, fairdeal.SealedDeck.create("g:1:1:x", 6, hole=6))
    src = "\n".join(p.read_text(encoding="utf-8") for p in _home_game_sources())
    assert src.count("sealed.num_seats") + src.count(".sealed.hole") <= 2, "the check lives in _seal_fits only"


# --- HGB-002: every /games/api route is a module-level function on ONE router ------------------


def test_every_api_route_is_on_the_router_and_table_routes_check_access(hg):
    import inspect

    from fastapi.routing import APIRoute

    routes = [r for r in hg.router.routes if isinstance(r, APIRoute)]
    assert len(routes) > 60
    assert all(r.path.startswith("/games/api/") for r in routes)
    assert any(getattr(d.dependency, "__name__", "") == "_api_guard" for d in hg.router.dependencies), \
        "the signed-in / rate-limit guard runs for every API request"
    for r in routes:
        fn = r.endpoint
        assert "<locals>" not in fn.__qualname__, f"{r.path}: a closure again (HGB-002)"
        assert getattr(hg, fn.__name__) is fn, f"{r.path}: not importable as homegame.{fn.__name__}"
        if r.path.startswith("/games/api/tables/{game_id}"):
            src = inspect.getsource(fn)
            assert "_in_table(" in src or "_table_for(" in src, f"{r.path} skips the club's access check"
    # the installed app serves them (install includes the router — FastAPI keeps it
    # as one included-router entry)
    app = sys.modules["plo5bp.ui.server"].app
    assert any(getattr(r, "original_router", None) is hg.router for r in app.router.routes)


# --- HGB-004: one HandState per hand -------------------------------------------------------------


def test_every_per_hand_field_lives_in_the_hand_and_starts_over(hg):
    from dataclasses import fields

    table_fields = {f.name for f in fields(hg.LiveTable)}
    assert not table_fields & set(hg.HAND_FIELDS), "a per-hand field is back on the table itself"
    t = hg.LiveTable(game_id="g", host_user_id=1, name="x", num_seats=6, sb_cents=50, bb_cents=100,
                     ante_cents=300, default_buyin_cents=4000, status="open", button=0, hand_no=0,
                     seats=[None] * 6)
    # the old names still work, and they ARE the hand's
    t.runout_active = True
    t.pots = [{"label": "Main pot"}]
    assert t.hand.runout_active is True and t.hand.pots == [{"label": "Main pot"}]
    # a fresh hand forgets every one of them (resize and close use the same)
    for name in hg.HAND_FIELDS:
        setattr(t.hand, name, object())
    t.hand = hg.HandState()
    fresh = hg.HandState()
    assert all(getattr(t, n) == getattr(fresh, n) for n in hg.HAND_FIELDS)


def test_a_deal_and_a_resize_leave_nothing_of_the_hand_before(cast_code):
    hg, p = cast_code["hg"], cast_code["p"]
    gid = _ok_json(p[0].post("/games/api/tables", json={"name": "fresh", "bb_cents": 100, "ante_cents": 300,
                                                          "default_buyin_cents": 4000}))["id"]
    assert p[1].post(f"/games/api/tables/{gid}/sit", json={"seat": 1, "buyin_cents": 4000}).status_code == 200
    assert p[0].post(f"/games/api/tables/{gid}/run", json={"running": True}).status_code == 200
    t = hg.HUB.get(gid)
    with t.lock:
        for _ in range(20):
            if t.phase != "in_hand":
                break
            hg._host_fold_locked(t, t.host_user_id)
        if t.runout_started_mono is not None:  # (a checked-down showdown: skip its award animation)
            t.runout_started_mono -= 600.0
        old = t.hand
        old.pots = [{"label": "left over"}]
        old.equity_by_len = {(3, 3): {}}
        old.rabbit_burns = [1, 2, 3]
        hg._deal_now_locked(t)
        assert t.hand is not old and t.pots == [] and t.equity_by_len == {} and t.rabbit_burns == []
        for _ in range(20):
            if t.phase != "in_hand":
                break
            hg._host_fold_locked(t, t.host_user_id)
        if t.runout_started_mono is not None:
            t.runout_started_mono -= 600.0
        t.hand.terminal_pot = 123
        with hg._mutation(t):
            hg._resize_locked(t, 6)
        assert t.terminal_pot == 0 and t.env is None and t.phase == "waiting"


# --- HGB-006: the parts split out of homegame.py -------------------------------------------------


def _parts(hg) -> list:
    return [sys.modules[f"plo5bp.ui.{part}"] for part in hg.SPLIT_MODULES]


def test_every_name_a_part_defines_is_homegame_x(hg):
    import ast

    assert len(hg.SPLIT_MODULES) >= 6
    for mod in _parts(hg):
        assert mod.hg is hg, f"{mod.__name__} is bound to another import of homegame"
        defined = set()
        for n in ast.parse(Path(mod.__file__).read_text(encoding="utf-8")).body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(n.name)
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                for tgt in n.targets if isinstance(n, ast.Assign) else [n.target]:
                    defined |= {x.id for x in ast.walk(tgt) if isinstance(x, ast.Name)}
        assert set(mod.__all__) == defined - {"hg", "logger", "__all__"}, \
            f"{mod.__name__}: __all__ lists exactly what it defines (else homegame.X misses it)"
        for name in mod.__all__:
            there = getattr(hg, name)
            assert there is getattr(mod, name) or getattr(there, "__wrapped__", None) is getattr(mod, name), \
                f"homegame.{name} is not {mod.__name__}.{name}"


def test_the_parts_reach_every_home_games_name_through_homegame(hg):
    """A part that called its own (or homegame's) functions by their bare names
    would slip past ``monkeypatch.setattr(homegame, ...)``: the tests that patch
    them would still pass, testing nothing (HGB-006). Inside a function every
    home-games name is ``hg.X``, looked up when it runs."""
    import symtable

    for mod in _parts(hg):
        own = set(mod.__all__)
        imported = {n for n in vars(mod) if n not in own}  # its imports, hg, logger
        home_names = (own | set(vars(hg))) - imported
        bad = []

        def walk(tab):
            if str(tab.get_type()) == "function":
                for sym in tab.get_symbols():
                    if sym.is_global() and sym.is_referenced() and sym.get_name() in home_names:
                        bad.append(f"{tab.get_name()}: {sym.get_name()}")
            for child in tab.get_children():
                walk(child)

        walk(symtable.symtable(Path(mod.__file__).read_text(encoding="utf-8"), mod.__file__, "exec"))
        assert not bad, f"{mod.__name__} names these directly (write hg.X): {bad}"


def test_patching_homegame_reaches_the_parts_and_use_context_swaps_their_state(hg, monkeypatch):
    user = {"id": 987_654, "name": "Account Name", "email": "someone@example.com"}
    # homegame_people calling its own helper: the patch on homegame reaches it
    monkeypatch.setattr(hg, "_chosen_name", lambda uid: "Chosen Name")
    assert hg._display_name(user) == "Chosen Name"
    monkeypatch.undo()
    ctx = hg.HomeGames()
    old = hg.use_context(ctx)
    try:
        assert hg.HUB is ctx.hub and hg.CTX is ctx  # (the old module-level names follow)
        assert hg._club_role("no-such-club", 987_654) is None  # homegame_clubs caches the answer ...
        assert ("no-such-club", 987_654) in ctx.role_cache  # ... in the CURRENT context
        assert ("no-such-club", 987_654) not in old.role_cache
    finally:
        assert hg.use_context(old) is ctx
    assert hg.CTX is old


def test_a_part_imported_on_its_own_still_serves_one_homegame(hg):
    """The parts carry no import-time side effects (the router they register on is
    their own; the clubs' DB listener is registered by homegame), so importing one
    directly returns the part homegame already uses."""
    import importlib

    for part in hg.SPLIT_MODULES:
        assert importlib.import_module(f"plo5bp.ui.{part}") is sys.modules[f"plo5bp.ui.{part}"]
    listeners = [fn for fn in getattr(hg.pub.DB, "write_listeners", []) if fn is hg._on_clubs_write]
    assert len(listeners) == 1
