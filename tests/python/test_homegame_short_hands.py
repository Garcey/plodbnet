"""Home games felt: the showdown labels' short forms (2026-09-25).

On a phone the seats on the table's long sides sit level with the boards, and
their showdown labels reached over the boards' end cards. games.table.js now
shortens the server's made-hand phrases on the felt ("a full house, Js full of
4s" -> "Js full of 4s"); the award caption, the dock and the hand history keep
the full wording. Pinned against every phrase hand_describe produces, so a
wording change on the server cannot silently fall back to the long form.
Evaluated in Node; skipped when Node is not installed."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from plo5bp.ui import hand_describe

TABLE_JS = Path(__file__).resolve().parents[2] / "python" / "plo5bp" / "ui" / "static" / "games.table.js"

HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const ctx = { console, Math, JSON, Number, String, Object, Array, Set, Map, RegExp };
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx, { filename: "games.table.js" });
const phrases = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
console.log(JSON.stringify(phrases.map((p) => ctx.HG.table.shortHand(p))));
"""

# (hand_describe info, the felt's short form)
CASES = [
    (("pair", 12), "Pair of As"),
    (("twopair", 8, 0), "Two pair 10s & 2s"),
    (("trips", 11), "Three Ks"),
    (("straight", 12, False), "Straight 10-A"),
    (("straight", 3, True), "Straight A-5"),
    (("flush", 11), "K-high flush"),
    (("fh", 9, 2), "Js full of 4s"),
    (("quads", 0), "Four 2s"),
    (("sf", 8, False), "Straight flush 6-10"),
    (("sf", 3, True), "Straight flush A-5"),
    (("high", 12), "A high"),
]


@pytest.fixture(scope="module")
def node():
    exe = shutil.which("node")
    if exe is None:
        pytest.skip("node is not installed")
    return exe


def test_every_made_hand_phrase_has_a_short_form_on_the_felt(node, tmp_path):
    phrases = [hand_describe._fmt(info) for info, _ in CASES]
    h = tmp_path / "h.js"
    h.write_text(HARNESS, encoding="utf-8")
    p = tmp_path / "phrases.json"
    p.write_text(json.dumps(phrases), encoding="utf-8")
    r = subprocess.run([node, str(h), str(TABLE_JS), str(p)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got == [short for _, short in CASES], list(zip(phrases, got))
    # every one of them is shorter than the server's phrase (bar "A high")
    for phrase, short in zip(phrases, got):
        assert len(short) < len(phrase) or phrase.endswith(" high") and len(short) == len(phrase), (phrase, short)
