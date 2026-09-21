"""Regression tests for the 2026-09-20 review — CFR desktop shortcut installer (E / J6).

``scripts/install_cfr_desktop_shortcut.py`` hard-required pywin32, which no
install step declared, so "Install CFR Solver.bat" failed on a fresh venv. It
now falls back to driving the same WScript.Shell COM object from PowerShell.

Nothing here touches the real Desktop / Start Menu.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "install_cfr_desktop_shortcut.py"


@pytest.fixture()
def inst():
    spec = importlib.util.spec_from_file_location("install_cfr_desktop_shortcut", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_without_pywin32_the_shortcut_is_created_through_powershell(inst, tmp_path, monkeypatch):
    calls = []
    lnk = tmp_path / "CFR Solver.lnk"

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        lnk.write_bytes(b"lnk")  # what PowerShell would have produced

        class Done:
            returncode, stdout, stderr = 0, "", ""

        return Done()

    monkeypatch.setattr(inst, "_have_pywin32", lambda: False)
    monkeypatch.setattr(inst.shutil, "which", lambda name: "powershell.exe")
    monkeypatch.setattr(inst.subprocess, "run", fake_run)

    hostile = tmp_path / "it's \"quoted\" $env & `tick`"
    backend = inst._create_shortcut(lnk, Path("C:/py/pythonw.exe"), '"C:/x y/cfr_app.py" --desktop', hostile, None)

    assert backend == "powershell"
    (cmd, kw), = calls
    assert cmd[0] == "powershell.exe" and "-NoProfile" in cmd and "-NonInteractive" in cmd
    script = cmd[-1]
    env = kw["env"]
    assert env["CFR_LNK_PATH"] == str(lnk)
    assert env["CFR_LNK_TARGET"] == str(Path("C:/py/pythonw.exe"))
    assert env["CFR_LNK_ARGS"] == '"C:/x y/cfr_app.py" --desktop'
    assert env["CFR_LNK_WORKDIR"] == str(hostile)
    assert env["CFR_LNK_ICON"] == ""
    # Values travel by environment variable ONLY — never pasted into the script.
    assert "$env:CFR_LNK_PATH" in script and "WScript.Shell" in script
    for value in (str(lnk), str(hostile), "cfr_app.py", "pythonw.exe"):
        assert value not in script


def test_powershell_failure_is_a_clear_error_not_a_silent_success(inst, tmp_path, monkeypatch):
    class Failed:
        returncode, stdout, stderr = 1, "", "Unable to save shortcut"

    monkeypatch.setattr(inst, "_have_pywin32", lambda: False)
    monkeypatch.setattr(inst.shutil, "which", lambda name: "powershell.exe")
    monkeypatch.setattr(inst.subprocess, "run", lambda *a, **k: Failed())
    with pytest.raises(RuntimeError, match="Unable to save shortcut"):
        inst._create_shortcut(tmp_path / "x.lnk", Path("a"), "", tmp_path, None)

    class LiedAboutIt:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(inst.subprocess, "run", lambda *a, **k: LiedAboutIt())
    with pytest.raises(RuntimeError, match="was not created"):
        inst._create_shortcut(tmp_path / "x.lnk", Path("a"), "", tmp_path, None)

    monkeypatch.setattr(inst.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="neither pywin32 nor PowerShell"):
        inst._create_shortcut(tmp_path / "x.lnk", Path("a"), "", tmp_path, None)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows shortcuts")
def test_main_no_longer_requires_pywin32(inst, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(inst, "_have_pywin32", lambda: False)
    monkeypatch.setattr(inst, "install", lambda: [tmp_path / "CFR Solver.lnk"])  # no real shortcuts
    monkeypatch.setattr(inst, "_warn_if_app_deps_missing", lambda: None)
    monkeypatch.setattr(sys, "argv", ["install_cfr_desktop_shortcut.py"])
    assert inst.main() == 0  # used to print "pywin32 is required" and return 1
    out = capsys.readouterr().out
    assert "PowerShell" in out and "CFR Solver.lnk" in out

    def boom():
        raise RuntimeError("no COM")

    monkeypatch.setattr(inst, "install", boom)
    assert inst.main() == 1
    assert "could not create the shortcut: no COM" in capsys.readouterr().err


@pytest.mark.skipif(
    sys.platform != "win32" or not (shutil.which("powershell") or shutil.which("pwsh")),
    reason="needs Windows PowerShell",
)
def test_real_powershell_shortcut_in_a_temp_dir(inst, tmp_path, monkeypatch):
    """The fallback end to end — into tmp_path, in a directory with a quote in its name."""
    monkeypatch.setattr(inst, "_have_pywin32", lambda: False)
    out_dir = tmp_path / "it's a dir"
    out_dir.mkdir()
    lnk = out_dir / "CFR Solver.lnk"
    args = f'"{inst._SCRIPT}" --desktop'

    assert inst._create_shortcut(lnk, Path(sys.executable), args, REPO, inst._ICON) == "powershell"
    assert lnk.is_file() and lnk.stat().st_size > 100

    back = inst._powershell(
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:P); "
        "'{0}|{1}|{2}|{3}' -f $s.TargetPath,$s.Arguments,$s.WorkingDirectory,$s.WindowStyle",
        {"P": str(lnk)},
    )
    target, got_args, workdir, style = back.split("|")
    assert Path(target) == Path(sys.executable)
    assert got_args == args
    assert Path(workdir) == REPO and style == "7"
