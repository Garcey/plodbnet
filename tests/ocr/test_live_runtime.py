"""Runtime behaviour of the live-capture runners (no pixels, no real capture).

* TOOL-004 — one live lock: study requests wait for an in-flight tick, and
  the tick holds the lock while it snapshots/rebuilds/commits.
* TOOL-010 — the OCR loop ticks on fixed deadlines (a slow tick no longer
  stretches the period).
* TOOL-036 — the WGC callback drops frames that arrive within
  ``min_interval_s`` of the last one kept, BEFORE the ~8 MB copy.
* SEC-024 — no websocket ingest (any web page could have connected to it).
* TOOL-044 — ``/ocr/save_frame {"to_fixtures": true}`` writes into the
  tracked fixtures folder, with the extracted FrameState next to the PNG.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from starlette.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from plo5bp.ocr.types import FrameState, SeatObs  # noqa: E402
from plo5bp.ui import server  # noqa: E402
from plo5bp.ui.live import clubgg, routes as live_routes, tracking  # noqa: E402


@pytest.fixture(autouse=True)
def _idle_runner():
    r = clubgg.ocr_runner
    saved = (r.running, r.latest_frame, r._clock, r._sleep)
    yield
    r.running, r.latest_frame, r._clock, r._sleep = saved
    r.task = None


# --- TOOL-004: the live lock ---------------------------------------------------


def test_study_requests_wait_for_an_inflight_tick():
    order: list[str] = []

    async def fake_app(scope, receive, send):
        order.append("handler " + scope["path"])

    mw = live_routes._LiveLockMiddleware(fake_app, core_app=server.app)

    async def scenario():
        lock = tracking.live_lock()
        async with lock:  # a tick is mid-rebuild
            click = asyncio.create_task(mw({"type": "http", "path": "/action"}, None, None))
            status = asyncio.create_task(mw({"type": "http", "path": "/ocr/status"}, None, None))
            await asyncio.sleep(0.02)
            order.append("tick committed")
        await asyncio.gather(click, status)

    asyncio.run(scenario())
    # The status poll is not session state and never waits; the click runs
    # only after the tick committed.
    assert order == ["handler /ocr/status", "tick committed", "handler /action"]


def test_every_study_route_is_serialized():
    mw = live_routes._LiveLockMiddleware(None, core_app=server.app)
    paths = mw._locked_paths()
    for p in ("/state", "/action", "/undo", "/cards", "/seats", "/config",
              "/reset", "/format", "/ocr/rescan", "/ocr/simple", "/pokernow/ingest"):
        assert p in paths, p
    for p in ("/ocr/status", "/pokernow/status", "/ocr/save_frame"):
        assert p not in paths, p
    assert not any(p.startswith(("/trainer", "/static")) for p in paths)


def test_ocr_tick_runs_under_the_live_lock(monkeypatch):
    from plo5bp.ocr import live as ocr_live

    monkeypatch.setattr(ocr_live, "is_window_alive", lambda hwnd: True)
    r = clubgg.ocr_runner
    seen: list[bool] = []

    async def scenario():
        lock = tracking.live_lock()

        def fake_tick():
            seen.append(lock.locked())
            r.running = False

        monkeypatch.setattr(r, "_tick", fake_tick)
        r.running, r.poll_ms, r._capture_control, r._hwnd = True, 50, None, 1
        await r._loop()

    asyncio.run(scenario())
    assert seen == [True]


# --- TOOL-010: fixed cadence ------------------------------------------------------


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    async def sleep(self, d: float) -> None:
        self.t += d
        await asyncio.sleep(0)


@pytest.mark.parametrize("cost, expected", [
    (0.06, [0.1, 0.2, 0.3, 0.4]),     # on the grid: period stays poll_ms
    (0.15, [0.1, 0.25, 0.4, 0.55]),   # overrun: next tick at once, no burst
])
def test_ocr_loop_ticks_on_fixed_deadlines(monkeypatch, cost, expected):
    from plo5bp.ocr import live as ocr_live

    monkeypatch.setattr(ocr_live, "is_window_alive", lambda hwnd: True)
    r = clubgg.ocr_runner
    clock = _FakeClock()
    starts: list[float] = []

    def fake_tick():
        starts.append(round(clock.t, 6))
        clock.t += cost
        if len(starts) == 4:
            r.running = False

    monkeypatch.setattr(r, "_tick", fake_tick)
    r._clock, r._sleep = clock.now, clock.sleep
    r.running, r.poll_ms, r._capture_control, r._hwnd = True, 100, None, 1
    asyncio.run(r._loop())
    assert starts == expected
    assert r.status()["tick_ms"] == pytest.approx(cost * 1000.0)


# --- TOOL-036: frame throttle ---------------------------------------------------


def test_wgc_callback_skips_frames_before_copying(monkeypatch):
    from plo5bp.ocr import live as ocr_live

    handlers: dict = {}

    class FakeCapture:
        def __init__(self, **kw):
            pass

        def event(self, fn):
            handlers[fn.__name__] = fn
            return fn

        def start_free_threaded(self):
            return "control"

    monkeypatch.setitem(sys.modules, "windows_capture",
                        types.SimpleNamespace(WindowsCapture=FakeCapture))
    t = [0.0]
    monkeypatch.setattr(ocr_live.time, "monotonic", lambda: t[0])
    kept: list[int] = []
    assert ocr_live.start_wgc_capture(1, lambda bgr: kept.append(int(bgr[0, 0, 0])),
                                      min_interval_s=0.1) == "control"

    class FakeBuffer:
        def __init__(self, v):
            self.v = v

        @property
        def frame_buffer(self):
            # Reading the buffer IS the copy's source; a dropped frame must
            # never get here.
            kept_reads.append(self.v)
            return np.full((2, 2, 4), self.v, dtype=np.uint8)

    kept_reads: list[int] = []
    for i, ts in enumerate([0.0, 0.016, 0.05, 0.099, 0.1, 0.12, 0.25]):
        t[0] = ts
        handlers["on_frame_arrived"](FakeBuffer(i), None)
    assert kept == [0, 4, 6]
    assert kept_reads == [0, 4, 6]


# --- SEC-024: no websocket ingest ------------------------------------------------


def test_pokernow_websocket_ingest_is_gone():
    c = TestClient(server.app)
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/pokernow/ingest") as ws:
            ws.send_json({})
            ws.receive_json()


# --- TOOL-044: save a frame straight into the tracked fixtures ------------------


def test_save_frame_to_fixtures_writes_png_and_state(monkeypatch, tmp_path):
    written: dict = {}
    fake_cv2 = types.SimpleNamespace(
        imwrite=lambda path, img: written.setdefault(path, img.shape) is not None
        or True
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    fs = FrameState(
        board_a=(None,) * 5, board_b=(None,) * 5, hero_hole=(None,) * 5,
        button_seat=2, pot_total_chips=1200,
        seats=tuple(SeatObs(seat=i, stack_chips=5000, committed_chips=0,
                            folded=False) for i in range(6)),
    )
    fake_extract = types.ModuleType("plo5bp.ocr.extract")
    fake_extract.extract_frame_state = lambda img, num_seats=6, cache=None: fs
    fake_extract.fit_to_calibration = lambda img: (img, None)
    monkeypatch.setitem(sys.modules, "plo5bp.ocr.extract", fake_extract)
    monkeypatch.setattr(live_routes, "_FIXTURE_FRAMES_DIR", tmp_path / "fixtures")
    monkeypatch.setattr(live_routes, "_DEBUG_FRAMES_DIR", tmp_path / "debug")
    clubgg.ocr_runner.latest_frame = np.zeros((1391, 1927, 3), dtype=np.uint8)
    c = TestClient(server.app)

    out = c.post("/ocr/save_frame", json={"to_fixtures": True}).json()
    assert out["tracked"] is True
    assert out["path"].startswith(str(tmp_path / "fixtures"))
    state = json.loads(open(out["state_path"], encoding="utf-8").read())
    assert FrameState.from_dict(state) == fs

    # No body (the old call) = a local-only debug capture, and it says so.
    out = c.post("/ocr/save_frame").json()
    assert out["tracked"] is False and "note" in out
    assert out["path"].startswith(str(tmp_path / "debug"))


# --- TOOL-011: no needless work per tick -----------------------------------------


@pytest.fixture()
def fresh_session():
    s = server.session
    s.variant = server.VARIANT_PLO5
    s.num_seats, s.button_seat, s.hero_seat = 6, 0, 0
    server._new_session_defaults()
    tracking._reset_live_tracking()
    server._rebuild_env()
    yield s
    server._new_session_defaults()
    tracking._reset_live_tracking()
    server._rebuild_env()


def test_engine_view_skips_the_outcome_monte_carlo(fresh_session):
    env = fresh_session.env
    real_rs = env._rs
    seen: list[bool] = []

    class Spy:
        def __getattr__(self, name):
            attr = getattr(real_rs, name)
            if name != "observation_dict":
                return attr

            def wrapped(*a, **k):
                seen.append(bool(k.get("skip_outcome_mc")))
                return attr(*a, **k)
            return wrapped

    env._rs = Spy()
    try:
        view = tracking._engine_view_from_session()
    finally:
        env._rs = real_rs
    assert seen == [True] and view.num_seats == 6


def test_idle_ticks_reuse_the_env(monkeypatch, fresh_session):
    from plo5bp.ocr.events import EventReconstructor
    from plo5bp.ocr.types import Card
    from plo5bp.ui.live.state import live_state

    fs = FrameState(
        board_a=(None,) * 5, board_b=(None,) * 5,
        hero_hole=(Card(12, 0), Card(11, 1), Card(10, 2), Card(9, 3), Card(8, 0)),
        button_seat=0, pot_total_chips=None,
        seats=tuple(SeatObs(seat=i, stack_chips=None, committed_chips=None,
                            folded=True) for i in range(6)),
    )
    fake = types.ModuleType("plo5bp.ocr.extract")
    fake.extract_frame_state = lambda img, num_seats=6, cache=None: fs
    fake.fit_to_calibration = lambda img: (img, None)
    monkeypatch.setitem(sys.modules, "plo5bp.ocr.extract", fake)
    builds: list[int] = []
    real = tracking._rebuild_env
    monkeypatch.setattr(tracking, "_rebuild_env", lambda: (builds.append(1), real())[1])
    r = clubgg.ocr_runner
    monkeypatch.setattr(r, "_reconstructor", EventReconstructor(num_seats=6))
    r.latest_frame = object()
    live_state.simple_ocr_mode = True
    for _ in range(5):
        r._tick()
    # Tick 1 builds (nothing built by the live path yet), tick 3 commits the
    # hero's cards after the 3-read debounce and rebuilds; ticks 2, 4 and 5
    # change nothing the replay reads.
    assert len(builds) == 2
    assert fresh_session.hero_hole == [48, 45, 42, 39, 32]
    assert r.last_error is None


# --- TOOL-003: a wrong-shaped capture is refused loudly -------------------------------


def test_wrong_shaped_capture_is_refused_before_extraction(monkeypatch):
    extracted: list = []
    fake = types.ModuleType("plo5bp.ocr.extract")
    fake.fit_to_calibration = lambda img: (None, "the ClubGG window is 1920x1080 ...")
    fake.extract_frame_state = lambda *a, **k: extracted.append(1)
    monkeypatch.setitem(sys.modules, "plo5bp.ocr.extract", fake)
    from plo5bp.ocr.events import EventReconstructor

    r = clubgg.ocr_runner
    monkeypatch.setattr(r, "_reconstructor", EventReconstructor(num_seats=6))
    r.latest_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    r._tick()
    assert extracted == []
    assert r.last_error.startswith("the ClubGG window is 1920x1080")
    assert r.status()["frame_note"] == r.last_error


# --- TOOL-016: the userscript installs and updates itself from the local server ---------


def test_the_collector_is_served_for_one_click_install():
    r = TestClient(server.app).get("/pokernow/pokernow.user.js")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/javascript")
    assert "// @updateURL    http://127.0.0.1:8765/pokernow/pokernow.user.js" in r.text
