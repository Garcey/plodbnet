"""(TOOL-048) One set of JSON / label helpers for the CFR + GTO code, and the
Python constants that mirror the Rust dump schema pinned against the solver."""

from __future__ import annotations

import json
import math
import re
import runpy
from pathlib import Path

import numpy as np
import pytest

from plo5bp.gto import cfr_export
from plo5bp.gto.jsonio import atomic_write_json, atomic_write_text, jsonable, sanitize_json
from plo5bp.gto.preflop_class import preflop_class_label

REPO = Path(__file__).resolve().parents[3]
CFR_SRC = REPO / "rust_engine" / "src" / "cfr"


def test_atomic_write_json_is_strict_compact_and_leaves_no_temp(tmp_path: Path):
    p = tmp_path / "a" / "r.json"
    atomic_write_json(p, {"x": float("nan"), "y": [1.5, float("inf")], "z": "ok"})
    text = p.read_text(encoding="utf-8")
    assert text == '{"x":null,"y":[1.5,null],"z":"ok"}\n'
    atomic_write_json(p, {"k": 1}, indent=2)
    assert p.read_text(encoding="utf-8") == '{\n  "k": 1\n}\n'
    atomic_write_text(tmp_path / "t.txt", "hi")
    assert (tmp_path / "t.txt").read_text(encoding="utf-8") == "hi"
    assert not list(tmp_path.rglob("*.tmp"))


def test_failed_write_keeps_the_old_file_and_no_temp(tmp_path: Path):
    p = tmp_path / "r.json"
    atomic_write_json(p, {"v": 1})

    class Boom:
        pass

    with pytest.raises(TypeError):
        atomic_write_json(p, {"v": Boom()})
    assert json.loads(p.read_text(encoding="utf-8")) == {"v": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_sanitize_and_jsonable():
    assert sanitize_json({"a": [float("-inf"), 2.0]}) == {"a": [None, 2.0]}
    assert jsonable({1: np.float32(0.5), "b": (np.int64(3), b"\x01")}) == {
        "1": 0.5, "b": [3, [1]]
    }


def test_scripts_use_the_one_class_label_mapping():
    for script in ("archive/gto_campaigns/summarize_pushfold.py", "export_pushfold_14_charts.py"):
        mod = runpy.run_path(str(REPO / "scripts" / script), run_name="helpers_test")
        assert [mod["class_label"](i) for i in range(169)] == [
            preflop_class_label(i) for i in range(169)
        ]


def _rust_const(path: Path, name: str) -> str:
    """A `pub const` value: a string literal's text, or a number without `_`s."""
    m = re.search(rf"pub const {name}: [^=]+= ([^;]+);", path.read_text(encoding="utf-8"))
    assert m, f"{name} not found in {path.name}"
    value = m.group(1).strip()
    return value[1:-1] if value.startswith('"') else value.replace("_", "")


def test_mirrored_constants_match_the_rust_source():
    infoset = CFR_SRC / "infoset.rs"
    assert int(_rust_const(infoset, "DUMP_SCHEMA_VERSION")) == cfr_export.DUMP_SCHEMA_VERSION
    assert _rust_const(infoset, "PRIV_COMBO") == cfr_export.PRIV_COMBO
    assert _rust_const(infoset, "PRIV_CLASS") == cfr_export.PRIV_CLASS
    assert _rust_const(infoset, "PRIV_OCHS_BUCKET") == cfr_export.PRIV_OCHS_BUCKET
    assert int(_rust_const(CFR_SRC / "card_abs.rs", "OCHS_BUCKET_BASE")) == cfr_export.OCHS_BUCKET_BASE


def test_mirrored_constants_match_what_the_solver_dumps():
    from plo5bp.gto.cfr_api import RootSpec, SolveConfig, rust_cfr_available, solve

    if not rust_cfr_available():
        pytest.skip("native CFR not built")
    cfg = dict(max_iterations=5, target_exploitability_bb=0.0, seed=1)
    river = solve(RootSpec(street=3, pot_bb=10, effective_stack_bb=10, board=[0, 5, 10, 15, 20],
                           raise_sizes_pm=[1000]), SolveConfig(**cfg))
    iset = river.strategy["infosets"][0]
    assert iset["schema_version"] == cfr_export.DUMP_SCHEMA_VERSION
    assert iset["private_kind"] == cfr_export.PRIV_COMBO
    flop = solve(RootSpec(street=1, pot_bb=6, effective_stack_bb=6, board=[12, 28, 38],
                          raise_sizes_pm=[1000]), SolveConfig(card_abstraction="ochs", **cfg))
    kinds = {i["private_kind"] for i in flop.strategy["infosets"]}
    assert kinds == {cfr_export.PRIV_OCHS_BUCKET}
    assert min(i["private_id"] for i in flop.strategy["infosets"]) >= cfr_export.OCHS_BUCKET_BASE
    pre = solve(RootSpec(street=0, pot_bb=2.0, effective_stack_bb=10, board=[], ante_chips=0,
                         raise_sizes_pm=[], allin_atom=True),
                SolveConfig(algorithm="mccfr_es", max_iterations=50, target_exploitability_bb=0.0))
    assert {i["private_kind"] for i in pre.strategy["infosets"]} == {cfr_export.PRIV_CLASS}
    assert all(math.isfinite(p) for i in pre.strategy["infosets"] for p in i["probs"])
