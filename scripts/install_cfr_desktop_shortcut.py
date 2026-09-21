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
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "cfr_app.py"
_ICON = _ROOT / "python" / "plo5bp" / "cfr_app" / "static" / "app.ico"
_VENV_PYTHONW = _ROOT / ".venv" / "Scripts" / "pythonw.exe"
_VENV_PYTHON = _ROOT / ".venv" / "Scripts" / "python.exe"


def _have_pywin32() -> bool:
    try:
        import win32com.client  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


def _powershell(script: str, env: dict[str, str] | None = None) -> str:
    """Run an inline PowerShell script; return stdout. Raises on failure.

    Values reach the script through ENVIRONMENT VARIABLES (``$env:NAME``), never
    by string interpolation, so a path containing quotes / ``$`` / backticks
    cannot break — or inject into — the command.
    """
    exe = shutil.which("powershell") or shutil.which("pwsh")
    if not exe:
        raise RuntimeError("neither pywin32 nor PowerShell is available to create the shortcut")
    proc = subprocess.run(
        [exe, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, **(env or {})},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"PowerShell failed ({proc.returncode}): {proc.stderr.strip()[:400]}")
    return proc.stdout.strip()


def _special_folder(name: str) -> Path | None:
    """Real shell folder (handles OneDrive redirection) via pywin32, else PowerShell."""
    try:
        import win32com.client  # type: ignore

        return Path(win32com.client.Dispatch("WScript.Shell").SpecialFolders(name))
    except Exception:
        pass
    try:
        out = _powershell(
            "[Environment]::GetFolderPath([Environment+SpecialFolder]$env:CFR_FOLDER)",
            {"CFR_FOLDER": name},
        )
        return Path(out) if out else None
    except Exception:
        return None


def _desktop_dir() -> Path:
    # Prefer the real Desktop (handles OneDrive redirection on Windows).
    found = _special_folder("Desktop")
    if found is not None:
        return found
    for key in ("OneDrive", "USERPROFILE"):
        base = os.environ.get(key)
        if base:
            cand = Path(base) / "Desktop"
            if cand.is_dir():
                return cand
    return Path.home() / "Desktop"


def _start_menu_dir() -> Path:
    programs = _special_folder("Programs")
    if programs is None:
        appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        programs = appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    d = programs / "CFR Solver"
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


_DESCRIPTION = "CFR Solver — NLH native (Monker/Pio-like)"

# Same WScript.Shell COM object pywin32 drives, from PowerShell (ships with Windows).
_PS_CREATE_SHORTCUT = """
$ErrorActionPreference = 'Stop'
$sc = (New-Object -ComObject WScript.Shell).CreateShortcut($env:CFR_LNK_PATH)
$sc.TargetPath = $env:CFR_LNK_TARGET
$sc.Arguments = $env:CFR_LNK_ARGS
$sc.WorkingDirectory = $env:CFR_LNK_WORKDIR
$sc.Description = $env:CFR_LNK_DESC
$sc.WindowStyle = 7
if ($env:CFR_LNK_ICON) { $sc.IconLocation = $env:CFR_LNK_ICON }
$sc.Save()
"""


def _create_shortcut(path: Path, target: Path, args: str, workdir: Path, icon: Path | None) -> str:
    """Create a .lnk; returns which backend did it (``pywin32`` / ``powershell``).

    (review 2026-09-20) pywin32 was a hard requirement that no install step
    declared, so the one-click installer failed on a fresh venv. It is now
    optional: without it the same COM object is driven from PowerShell.
    """
    icon_loc = f"{icon},0" if icon is not None and icon.is_file() else ""
    if _have_pywin32():
        import win32com.client  # type: ignore

        shell = win32com.client.Dispatch("WScript.Shell")
        sc = shell.CreateShortCut(str(path))
        sc.Targetpath = str(target)
        sc.Arguments = args
        sc.WorkingDirectory = str(workdir)
        sc.Description = _DESCRIPTION
        sc.WindowStyle = 7  # minimized — pythonw has no window anyway
        if icon_loc:
            sc.IconLocation = icon_loc
        sc.save()
        return "pywin32"
    _powershell(
        _PS_CREATE_SHORTCUT,
        {
            "CFR_LNK_PATH": str(path),
            "CFR_LNK_TARGET": str(target),
            "CFR_LNK_ARGS": args,
            "CFR_LNK_WORKDIR": str(workdir),
            "CFR_LNK_DESC": _DESCRIPTION,
            "CFR_LNK_ICON": icon_loc,
        },
    )
    if not path.is_file():
        raise RuntimeError(f"PowerShell reported success but {path} was not created")
    return "powershell"


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


def _warn_if_app_deps_missing() -> None:
    """A shortcut to an app that cannot start helps nobody — say what is missing.

    The shortcut launches with pythonw (no console), so a missing dependency
    would otherwise surface only as a message box / a line in runs/cfr_app.log.
    """
    import importlib.util

    missing = [m for m in ("fastapi", "uvicorn", "multipart") if importlib.util.find_spec(m) is None]
    if missing:
        print(f'WARNING: missing packages {missing} — run:  .venv\\Scripts\\pip install -e ".[ui,dev]"')
    if importlib.util.find_spec("webview") is None:
        print("note: pywebview is not installed — the app will open in your browser instead "
              'of its own window (pip install -e ".[ui]" adds it).')
    # Look for the built file rather than importing plo5bp (that pulls in torch,
    # and an unrelated import error would masquerade as "extension not built").
    pkg = _ROOT / "python" / "plo5bp"
    if not (list(pkg.glob("_engine*.pyd")) or list(pkg.glob("_engine*.so"))):
        print("WARNING: the Rust solver extension is not built — from the repo root run:\n"
              "  .venv\\Scripts\\maturin develop --release")


def main() -> int:
    p = argparse.ArgumentParser(description="Install CFR Solver Desktop shortcut")
    p.add_argument("--uninstall", action="store_true")
    args = p.parse_args()

    if sys.platform != "win32":
        print("This installer creates Windows .lnk shortcuts. On other systems run:\n"
              "  python scripts/cfr_app.py --desktop", file=sys.stderr)
        return 1

    if args.uninstall:
        removed = uninstall()
        if not removed:
            print("nothing to remove")
        for r in removed:
            print(f"removed  {r}")
        return 0

    # pywin32 is optional (review 2026-09-20): PowerShell drives the same COM object.
    backend = "pywin32" if _have_pywin32() else "PowerShell (pywin32 not installed — that is fine)"
    try:
        created = install()
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        print(f"could not create the shortcut: {e}", file=sys.stderr)
        return 1
    print(f"Shortcuts created via {backend}.")
    _warn_if_app_deps_missing()
    print("CFR Solver installed as a desktop app:")
    for c in created:
        print(f"  {c}")
    print()
    print("Double-click 'CFR Solver' on your Desktop — no terminal, no browser tab.")
    print("To remove later:  .venv/Scripts/python scripts/install_cfr_desktop_shortcut.py --uninstall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
