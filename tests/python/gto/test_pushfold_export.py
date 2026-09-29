"""(TOOL-061) scripts/export_pushfold_14_charts.py checks what it exported.

It used to return 0 whatever it found (``0 if n == 14 else 0``) and stamped every
INDEX with the text of one particular solve. Synthetic solve reports keep this
fast: the exporter only reads infoset ids + probs and the root.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "export_pushfold_14_charts.py"


def _mod():
    return runpy.run_path(str(SCRIPT), run_name="export14_test")


def _report(tmp: Path, *, drop: str | None = None, root_extra: dict | None = None) -> Path:
    mod = _mod()
    infosets = []
    for seat, path in sorted(mod["expected_nodes"]()):
        if path == drop:
            continue
        for cid in range(169):
            infosets.append({
                "infoset_id": f"mwpf_p{seat}_path{path}_c{cid}",
                "actions": ["FOLD", "ALLIN"],
                "probs": [0.25, 0.75] if cid % 2 else [1.0, 0.0],
            })
    root = {"num_seats": 4, "bb_chips": 10_000, "sb_chips": 5_000, "ante_chips": 0,
            "stacks_bb": [10.0] * 4, "raise_sizes_pm": [], "allin_atom": True,
            **(root_extra or {})}
    rep = {"status": "ok", "root": root, "strategy": {"infosets": infosets},
           "iterations_run": 20_000, "exploitability_bb": 0.3, "notes": []}
    p = tmp / "solve.json"
    p.write_text(json.dumps(rep), encoding="utf-8")
    return p


def _run(src: Path, out: Path, monkeypatch) -> int:
    monkeypatch.setattr(sys, "argv", ["export_pushfold_14_charts.py", str(src), str(out)])
    return _mod()["main"]()


def test_full_tree_exports_14_charts_and_describes_the_real_root(tmp_path, monkeypatch):
    out = tmp_path / "charts"
    assert _run(_report(tmp_path), out, monkeypatch) == 0
    index = json.loads((out / "INDEX.json").read_text(encoding="utf-8"))
    assert len(index["nodes"]) == 14 and index["missing_nodes"] == []
    assert index["spot"] == (
        "4-handed, 10bb stacks, blinds 0.5bb/1bb, no ante, no rake, push/fold only"
    )
    chart = json.loads((out / "00_CO_open.json").read_text(encoding="utf-8"))
    assert chart["num_hands"] == 169 and chart["hands"][0]["hand"] == "22"


def test_a_missing_node_fails(tmp_path, monkeypatch):
    out = tmp_path / "charts"
    assert _run(_report(tmp_path, drop="F,AI,AI"), out, monkeypatch) == 1
    index = json.loads((out / "INDEX.json").read_text(encoding="utf-8"))
    assert index["missing_nodes"] == ["BB:F,AI,AI"]


def test_the_spot_text_follows_the_input(tmp_path, monkeypatch):
    out = tmp_path / "charts"
    extra = {"stacks_bb": [15.0, 15.0, 12.0, 20.0], "ante_chips": 1_000}
    assert _run(_report(tmp_path, root_extra=extra), out, monkeypatch) == 0
    spot = json.loads((out / "INDEX.json").read_text(encoding="utf-8"))["spot"]
    assert spot == ("4-handed, stacks 15bb/15bb/12bb/20bb, blinds 0.5bb/1bb, ante 0.1bb, "
                    "no rake, push/fold only")


@pytest.mark.parametrize("extra", [{"num_seats": 3}, {"raise_sizes_pm": [500]}])
def test_not_a_4_handed_push_fold_solve_is_refused(tmp_path, monkeypatch, extra):
    out = tmp_path / "charts"
    assert _run(_report(tmp_path, root_extra=extra), out, monkeypatch) == 2
    assert not (out / "INDEX.json").exists()
