"""Dead imports don't creep back into the web modules (BE-016).

A tiny stand-in for a linter's unused-import rule (no new dependency): every
name a module imports must be used in it, or be a deliberate re-export listed
here (other modules / tests read it through this module)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[3] / "python" / "plo5bp" / "ui"

#: module -> names it imports on purpose without using them itself.
REEXPORTS = {
    "server.py": {"OBS_DIM_MINIMAL", "_spec_anchor_label", "_encoding", "encode_observation"},
}


def _unused_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported[(a.asname or a.name).split(".")[0]] = node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            for a in node.names:
                imported[a.asname or a.name] = node.lineno
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                used.add(root.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            used.update(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", node.value))  # string annotations
    return set(imported) - used - REEXPORTS.get(path.name, set())


@pytest.mark.parametrize("name", [
    "server.py", "trainer.py", "public.py", "common.py", "ranges.py",
    "models.py", "middleware.py", "ratelimit.py",
])
def test_no_unused_imports(name):
    assert _unused_imports(UI / name) == set()
