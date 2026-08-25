#!/usr/bin/env python
"""Install a double-click Desktop shortcut for the CFR Solver app.

Creates (or refreshes)::

    <Desktop>/CFR Solver.lnk

that launches the app with pythonw (no console window) and the brand icon.
Also drops a Start Menu entry under Programs\\CFR Solver.

Usage::

    .venv/Scripts/python scripts/install_cfr_desktop_shortcut.py
    .venv/Scripts/python scripts/install_cfr_desktop_shortcut.py --uninstall
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "cfr_app.py"
_ICON = _ROOT / "python" / "plo5bp" / "cfr_app" / "static" / "app.ico"
_VENV_PYTHONW = _ROOT / ".venv" / "Scripts" / "pythonw.exe"
_VENV_PYTHON = _ROOT / ".venv" / "Scripts" / "python.exe"


def _desktop_dir() -> Path:
    # Prefer the real Desktop (handles OneDrive redirection on Windows).
    try:
        import win32com.client  # type: ignore

        shell = win32com.client.Dispatch("WScript.Shell")
        return Path(shell.SpecialFolders("Desktop"))
    except Exception:
        pass
    for key in ("OneDrive", "USERPROFILE"):
        base = os.environ.get(key)
        if base:
            cand = Path(base) / "Desktop"
            if cand.is_dir():
                return cand
    return Path.home() / "Desktop"


def _start_menu_dir() -> Path:
    try:
        import win32com.client  # type: ignore

        shell = win32com.client.Dispatch("WScript.Shell")
        programs = Path(shell.SpecialFolders("Programs"))
        d = programs / "CFR Solver"
        d.mkdir(parents=True, exist_ok=True)
        return d
    except Exception:
        appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        d = appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "CFR Solver"
        d.mkdir(parents=True, exist_ok=True)
        return d


def _pythonw() -> Path:
    if _VENV_PYTHONW.is_file():
        return _VENV_PYTHONW
    if _VENV_PYTHON.is_file():
        return _VENV_PYTHON
    # Last resort: whatever is running us, but prefer pythonw sibling.
    exe = Path(sys.executable)
    sibling = exe.with_name("pythonw.exe")
    return sibling if sibling.is_file() else exe


def _create_shortcut(path: Path, target: Path, args: str, workdir: Path, icon: Path | None) -> None:
    import win32com.client  # type: ignore

    shell = win32com.client.Dispatch("WScript.Shell")
    sc = shell.CreateShortCut(str(path))
    sc.Targetpath = str(target)
    sc.Arguments = args
    sc.WorkingDirectory = str(workdir)
    sc.Description = "CFR Solver — NLH native (Monker/Pio-like)"
    sc.WindowStyle = 7  # minimized — pythonw has no window anyway
    if icon is not None and icon.is_file():
        sc.IconLocation = f"{icon},0"
    sc.save()


def install() -> list[Path]:
    if not _SCRIPT.is_file():
        raise SystemExit(f"missing launcher script: {_SCRIPT}")
    target = _pythonw()
    if not target.is_file():
        raise SystemExit(f"missing pythonw: {target}")

    # Force desktop mode even if someone swaps pythonw → python later.
    args = f'"{_SCRIPT}" --desktop'
    icon = _ICON if _ICON.is_file() else None

    created: list[Path] = []
    desktop = _desktop_dir()
    desktop.mkdir(parents=True, exist_ok=True)
    desk_lnk = desktop / "CFR Solver.lnk"
    _create_shortcut(desk_lnk, target, args, _ROOT, icon)
    created.append(desk_lnk)

    start = _start_menu_dir()
    start_lnk = start / "CFR Solver.lnk"
    _create_shortcut(start_lnk, target, args, _ROOT, icon)
    created.append(start_lnk)

    return created


def uninstall() -> list[Path]:
    removed: list[Path] = []
    for p in (
        _desktop_dir() / "CFR Solver.lnk",
        _start_menu_dir() / "CFR Solver.lnk",
    ):
        if p.is_file():
            p.unlink()
            removed.append(p)
    # Remove empty Start Menu folder.
    sm = _start_menu_dir()
    try:
        if sm.is_dir() and not any(sm.iterdir()):
            sm.rmdir()
    except OSError:
        pass
    return removed


def main() -> int:
    p = argparse.ArgumentParser(description="Install CFR Solver Desktop shortcut")
    p.add_argument("--uninstall", action="store_true")
    args = p.parse_args()

    try:
        import win32com.client  # noqa: F401
    except ImportError:
        print("pywin32 is required:  .venv/Scripts/pip install pywin32", file=sys.stderr)
        return 1

    if args.uninstall:
        removed = uninstall()
        if not removed:
            print("nothing to remove")
        for r in removed:
            print(f"removed  {r}")
        return 0

    created = install()
    print("CFR Solver installed as a desktop app:")
    for c in created:
        print(f"  {c}")
    print()
    print("Double-click 'CFR Solver' on your Desktop — no terminal, no browser tab.")
    print("To remove later:  .venv/Scripts/python scripts/install_cfr_desktop_shortcut.py --uninstall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
