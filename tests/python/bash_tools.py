"""Running the repo's bash scripts from tests (the deploy, the ops scripts, hooks).

On Windows only Git for Windows' bash qualifies — never WSL's bash.exe, which
would run the script in another filesystem namespace."""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def find_bash() -> str | None:
    if os.name == "nt":
        for c in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
            if Path(c).exists():
                return c
        found = shutil.which("bash")
        return found if found and "git" in found.lower() else None
    return shutil.which("bash")


BASH = find_bash()


def posix(p: Path | str) -> str:
    """A path as bash sees it (a Windows drive path becomes /c/... under Git Bash)."""
    s = str(p)
    if os.name == "nt" and len(s) > 1 and s[1] == ":":
        s = "/" + s[0].lower() + s[2:].replace("\\", "/")
    return s
