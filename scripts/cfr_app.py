#!/usr/bin/env python
"""CFR Solver — standalone desktop app (Monker/Pio-like).

Double-click the Desktop shortcut, or::

    .venv/Scripts/pythonw scripts/cfr_app.py          # silent desktop window
    .venv/Scripts/python  scripts/cfr_app.py --desktop
    .venv/Scripts/python  scripts/cfr_app.py --browser # dev / browser only

Creates no visible console when launched via pythonw / the .lnk shortcut.
The local FastAPI server binds 127.0.0.1 on an ephemeral free port and dies
with the window — no permanent "dev server" left running.
"""

from __future__ import annotations

import argparse
import atexit
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

_ICON = _ROOT / "python" / "plo5bp" / "cfr_app" / "static" / "app.ico"
_LOG = _ROOT / "runs" / "cfr_app.log"

# pythonw.exe leaves stdout/stderr as None. uvicorn's colourised formatter
# calls sys.stdout.isatty() at Config init and crashes with
# "Unable to configure formatter 'default'". Redirect to the app log first.
if sys.stdout is None or sys.stderr is None:
    try:
        _LOG.parent.mkdir(parents=True, exist_ok=True)
        _stdio_sink = open(_LOG, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        if sys.stdout is None:
            sys.stdout = _stdio_sink  # type: ignore[assignment]
        if sys.stderr is None:
            sys.stderr = _stdio_sink  # type: ignore[assignment]
    except OSError:
        # Last resort: /dev/null equivalent so isatty() never hits None.
        import io

        _null = io.StringIO()
        if sys.stdout is None:
            sys.stdout = _null  # type: ignore[assignment]
        if sys.stderr is None:
            sys.stderr = _null  # type: ignore[assignment]


def _log(msg: str) -> None:
    """Append a line to runs/cfr_app.log (safe when console is hidden)."""
    try:
        _LOG.parent.mkdir(parents=True, exist_ok=True)
        with _LOG.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")
    except OSError:
        pass


# Plain uvicorn log config — no ColourizedFormatter / isatty dependency.
_UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(levelname)s:     %(message)s"},
        "access": {"format": "%(message)s"},
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        "uvicorn.error": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        "uvicorn.access": {"handlers": ["access"], "level": "WARNING", "propagate": False},
    },
}


def _pick_port(host: str = "127.0.0.1") -> int:
    """Bind an ephemeral free port so we never collide with a leftover server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _port_listening(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def _start_server(host: str, port: int):
    """Start uvicorn in a daemon thread; return (server, thread)."""
    import uvicorn

    config = uvicorn.Config(
        "plo5bp.cfr_app.server:app",
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        use_colors=False,
        log_config=_UVICORN_LOG_CONFIG,
    )
    server = uvicorn.Server(config)

    def _run() -> None:
        try:
            server.run()
        except Exception:
            _log("uvicorn crashed:\n" + traceback.format_exc())

    t = threading.Thread(target=_run, name="cfr-uvicorn", daemon=True)
    t.start()

    # Imports take well under a second now that the API no longer loads torch
    # (TOOL-020); the generous bound covers a cold disk / antivirus scan.
    for _ in range(300):  # ≤30s
        if _port_listening(host, port):
            return server, t
        if not t.is_alive():
            raise RuntimeError("CFR server thread died before binding a port")
        time.sleep(0.1)
    raise RuntimeError(f"CFR server did not become ready on {host}:{port}")


# ---------------------------------------------------------------------------
# Closing the window while a solve runs (TOOL-031)
# ---------------------------------------------------------------------------

_ACTIVE = ("queued", "running", "paused")
ANSWER_SAVE, ANSWER_CLOSE, ANSWER_CANCEL = "save", "close", "cancel"


def _ask_close(job: dict) -> str:
    """Native Yes / No / Cancel box: stop-and-save, close now, or keep solving."""
    iters = job.get("iterations_run") or 0
    msg = (
        f"A solve is still {job.get('status', 'running')} ({int(iters):,} iterations so far).\n\n"
        "Yes — stop it and save the strategy, then close (a few seconds)\n"
        "No — close now (its last live snapshot stays in the Library)\n"
        "Cancel — keep solving"
    )
    if sys.platform == "win32":
        import ctypes

        # MB_YESNOCANCEL | MB_ICONWARNING | MB_TOPMOST
        r = ctypes.windll.user32.MessageBoxW(0, msg, "CFR Solver", 0x3 | 0x30 | 0x40000)
        return {6: ANSWER_SAVE, 7: ANSWER_CLOSE}.get(r, ANSWER_CANCEL)
    return ANSWER_CLOSE  # other platforms: the pywebview dialog below decides


def close_decision(session, ask=_ask_close, *, on_saved=None, wait_secs: float = 60.0) -> bool:
    """Whether the window may close NOW (True) — the closing-event handler.

    Idle → close. A live solve → ask: Cancel keeps it running; No closes at once
    (the spawn child dies with the app; its last snapshot is listed in the Library
    as "interrupted"); Yes requests a graceful stop, keeps the window open while
    the solver exports its strategy (the UI shows "stopping"), and ``on_saved``
    closes it from a background thread once the job is final (or after
    ``wait_secs``).
    """
    job = session.active_job()
    if not job or job.get("status") not in _ACTIVE:
        return True
    answer = ask(job)
    if answer == ANSWER_CANCEL:
        return False
    if answer == ANSWER_CLOSE:
        return True
    job_id = job.get("job_id")
    try:
        session.stop(job_id)
    except KeyError:
        return True

    def _wait_then_close() -> None:
        deadline = time.time() + wait_secs
        while time.time() < deadline:
            j = session.get_job(job_id, full=False)
            if not j or j.get("status") not in _ACTIVE:
                break
            time.sleep(0.25)
        if on_saved is not None:
            on_saved()

    threading.Thread(target=_wait_then_close, name="cfr-close-after-save", daemon=True).start()
    return False


def _stop_server(server) -> None:
    try:
        server.should_exit = True
    except Exception:
        pass


def _run_desktop(host: str, port: int | None) -> int:
    """Native OS window. Server lives only while the window is open."""
    if port is None:
        port = _pick_port(host)
    url = f"http://{host}:{port}"
    _log(f"desktop launch {url}")

    server, _thread = _start_server(host, port)
    atexit.register(_stop_server, server)

    try:
        import webview  # type: ignore
    except ImportError:
        _log("pywebview missing — falling back to default browser")
        import webbrowser

        webbrowser.open(url)
        # Keep the server alive until the user kills the process.
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 0

    icon = str(_ICON) if _ICON.is_file() else None
    # confirm_close stays False: pywebview's box would ask on EVERY close. The
    # closing handler below asks only while a solve is running (TOOL-031).
    kwargs = dict(
        title="CFR Solver — NLH",
        url=url,
        width=1440,
        height=900,
        min_size=(960, 640),
        confirm_close=False,
        background_color="#0e1116",
    )
    # pywebview versions differ on the icon kwarg name / support.
    if icon:
        try:
            window = webview.create_window(**kwargs, icon=icon)
        except TypeError:
            window = webview.create_window(**kwargs)
    else:
        window = webview.create_window(**kwargs)

    def _on_closing() -> bool:
        try:
            from plo5bp.cfr_app import server as cfr_server  # the module uvicorn serves
        except Exception:
            return True
        ask = _ask_close
        if sys.platform != "win32":
            def ask(job):  # noqa: E306 — pywebview's own two-button dialog
                ok = window.create_confirmation_dialog(
                    "CFR Solver", "A solve is still running. Close anyway? "
                    "Its last live snapshot stays in the Library."
                )
                return ANSWER_CLOSE if ok else ANSWER_CANCEL
        try:
            return close_decision(cfr_server.session, ask, on_saved=window.destroy)
        except Exception:
            _log("close handler failed:\n" + traceback.format_exc())
            return True

    try:
        window.events.closing += _on_closing
    except AttributeError:
        _log("pywebview without window events — closing will not confirm a running solve")

    try:
        webview.start()
    finally:
        _stop_server(server)
        _log("desktop window closed")
    return 0


def _run_browser(host: str, port: int, reload: bool) -> int:
    """Classic long-lived uvicorn (dev / browser mode)."""
    import uvicorn

    url = f"http://{host}:{port}"
    print(f"[cfr_app] {url}")
    print("[cfr_app] Tree builder · Ranges · Solve · Strategy lines · Library")
    print(f"[cfr_app] open {url}  (or pass --desktop / use the Desktop shortcut)")
    uvicorn.run(
        "plo5bp.cfr_app.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
    return 0


def _is_loopback(host: str) -> bool:
    h = (host or "").strip().strip("[]").lower()
    if h == "localhost":
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    # Default to desktop when launched via pythonw (no console) or when
    # CFR_APP_DESKTOP=1 is set by the shortcut. Explicit --browser wins.
    launched_via_pythonw = sys.executable.lower().endswith("pythonw.exe")
    env_desktop = os.environ.get("CFR_APP_DESKTOP", "").strip() in ("1", "true", "yes")

    p = argparse.ArgumentParser(description="CFR Solver desktop app")
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. Loopback only unless --allow-remote (TOOL-058)",
    )
    p.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow a non-loopback --host: ANYONE who can reach this machine can then "
        "drive the solver and read its files. Same-origin requests only.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port (default: ephemeral free port in desktop mode, 8766 in browser mode)",
    )
    p.add_argument("--reload", action="store_true", help="uvicorn --reload (browser mode only)")
    p.add_argument(
        "--desktop",
        action="store_true",
        default=launched_via_pythonw or env_desktop,
        help="Native OS window (default when launched via pythonw / Desktop shortcut)",
    )
    p.add_argument(
        "--browser",
        action="store_true",
        help="Long-lived browser/dev server instead of desktop window",
    )
    p.add_argument(
        "--no-browser-hint",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = p.parse_args(argv)

    desktop = bool(args.desktop) and not bool(args.browser)
    # (TOOL-058) The server's Host-header guard is no defence once the socket is
    # on the network (a LAN client can send "Host: 127.0.0.1"), so binding a
    # non-loopback address is refused unless the user asks for it by name.
    if not _is_loopback(args.host):
        if not args.allow_remote:
            print(
                f"[cfr_app] refusing --host {args.host}: it would expose the solver to the "
                "network. Use 127.0.0.1, or add --allow-remote if you really mean it.",
                file=sys.stderr,
            )
            return 2
        os.environ["CFR_APP_ALLOWED_HOSTS"] = "*"  # same-origin only, see server.py
        print(
            f"[cfr_app] WARNING: listening on {args.host} — anyone who can reach this "
            "machine can drive the solver.",
            file=sys.stderr,
        )
    try:
        if desktop:
            port = args.port  # None → pick free
            return _run_desktop(args.host, port)
        port = args.port if args.port is not None else 8766
        return _run_browser(args.host, port, args.reload)
    except Exception as exc:
        _log(f"fatal: {exc}\n{traceback.format_exc()}")
        # Surface a MessageBox when there is no console (pythonw).
        if launched_via_pythonw or env_desktop:
            try:
                import ctypes

                ctypes.windll.user32.MessageBoxW(
                    0,
                    f"CFR Solver failed to start:\n\n{exc}\n\nSee runs\\cfr_app.log",
                    "CFR Solver",
                    0x10,  # MB_ICONERROR
                )
            except Exception:
                pass
        else:
            print(f"[cfr_app] fatal: {exc}", file=sys.stderr)
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
