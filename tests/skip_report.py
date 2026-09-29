"""Skipped tests are listed at the end of every run (TEST-031).

``pytest -q`` used to hide skips entirely, so on a PC without OpenCV every pixel-OCR
test skipped unnoticed. This plugin (registered by tests/conftest.py) prints a short
"skipped" section after every run — the count per reason, each reason tagged with
its category from ``tests/expected_skips.txt`` — and flags a reason that matches no
pattern there as UNEXPECTED. With ``--strict-skips``, ``PLO5BP_STRICT_SKIPS=1`` or on
CI an unexpected skip fails the run.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

import pytest

POLICY = Path(__file__).resolve().parent / "expected_skips.txt"


def load_policy(path: Path = POLICY) -> list[tuple[str, re.Pattern[str]]]:
    rules = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        category, _, pattern = line.partition("|")
        rules.append((category.strip(), re.compile(pattern.strip(), re.I)))
    return rules


def skip_reason(rep) -> str:
    """The human reason of a skipped report ("Skipped: " prefix removed)."""
    lr = getattr(rep, "longrepr", None)
    text = str(lr[2]) if isinstance(lr, tuple) and len(lr) == 3 else str(lr or "")
    return re.sub(r"^Skipped:\s*", "", text).strip()


def classify(reason: str, rules) -> str | None:
    for category, rx in rules:
        if rx.search(reason):
            return category
    return None


def _strict(config) -> bool:
    return bool(
        config.getoption("strict_skips", default=False)
        or os.environ.get("PLO5BP_STRICT_SKIPS", "").strip() in ("1", "true", "yes")
        or os.environ.get("CI")
    )


class SkipReport:
    def __init__(self, config) -> None:
        self.config = config
        try:
            self.rules = load_policy()
        except OSError:
            self.rules = []

    def _collect(self, stats) -> tuple[Counter, list[str]]:
        groups: Counter = Counter()
        unexpected: list[str] = []
        for rep in stats.get("skipped", []):
            reason = skip_reason(rep)
            cat = classify(reason, self.rules)
            groups[(cat or "UNEXPECTED", reason)] += 1
            if cat is None:
                unexpected.append(f"{getattr(rep, 'nodeid', '?')}: {reason}")
        return groups, unexpected

    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        groups, unexpected = self._collect(terminalreporter.stats)
        if not groups:
            return
        total = sum(groups.values())
        terminalreporter.section(f"{total} skipped (tests/expected_skips.txt)", sep="-")
        for (cat, reason), n in sorted(groups.items(), key=lambda kv: (kv[0][0] == "UNEXPECTED", -kv[1])):
            line = f"{n:4d}  [{cat}] {reason[:110]}"
            terminalreporter.write_line(line, red=cat == "UNEXPECTED", yellow=cat != "UNEXPECTED")
        if unexpected:
            terminalreporter.write_line(
                f"!!! {len(unexpected)} UNEXPECTED skip(s) — fix the cause, or add a pattern to "
                "tests/expected_skips.txt if the skip is legitimate on some machines"
                + ("" if _strict(config) else " (run with --strict-skips to fail on them)"),
                red=True,
            )

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        if not _strict(session.config) or exitstatus != 0:
            return
        tr = session.config.pluginmanager.get_plugin("terminalreporter")
        if tr is None:
            return
        _groups, unexpected = self._collect(tr.stats)
        if unexpected:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
