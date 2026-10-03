"""The board focus (2026-10-03): hovering a board — or its made-hand label — lights up the
two hole cards you play on it (owner: "hover over each board and have it highlight the two
hole cards that I'm playing on that board", the home games, the Trainer and Study alike).

The felt picks them in the browser (``HG.table.bestPlay`` in games.table.js, which draws all
three tables). For PLO it must pick exactly the cards ``hand_describe.best_combo`` picks —
exactly two of yours and three of the board, a tie taking the lowest cards — so the light
never disagrees with the label beside it. One board (NLH): the best five of all seven, with
as few of your cards as the hand needs. Evaluated in Node; skipped without it."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from itertools import combinations
from pathlib import Path

import pytest

from plo5bp.ui.hand_describe import _best_nlh, _eval5, best_combo

TABLE_JS = Path(__file__).resolve().parents[3] / "python" / "plo5bp" / "ui" / "static" / "games.table.js"

HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const ctx = { console, Math, JSON, Number, String, Object, Array, Set, Map, RegExp };
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx, { filename: "games.table.js" });
const cases = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
console.log(JSON.stringify(cases.map((c) => ctx.HG.table.bestPlay(c.hole, c.board, c.any))));
"""


@pytest.fixture(scope="module")
def node():
    exe = shutil.which("node")
    if exe is None:
        pytest.skip("node is not installed")
    return exe


def _run(node, tmp_path, cases):
    h = tmp_path / "h.js"
    h.write_text(HARNESS, encoding="utf-8")
    p = tmp_path / "cases.json"
    p.write_text(json.dumps(cases), encoding="utf-8")
    r = subprocess.run([node, str(h), str(TABLE_JS), str(p)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def _deals(rng, n, holes, boards, deck=range(52)):
    out = []
    for _ in range(n):
        hn, bn = rng.choice(holes), rng.choice(boards)
        cards = rng.sample(list(deck), hn + bn)
        out.append((cards[:hn], cards[hn:]))
    return out


def test_plo_lights_exactly_the_cards_the_label_is_made_of(node, tmp_path):
    rng = random.Random(20261003)
    deals = _deals(rng, 1200, (4, 5, 6, 7), (3, 4, 5))
    # tens to aces only: straights, flushes, full houses, quads and royal flushes galore
    deals += _deals(rng, 600, (4, 5), (3, 4, 5), deck=range(32, 52))
    # one suit's run of cards: straight flushes and their ties
    deals += _deals(rng, 200, (4, 5), (3, 5), deck=[r * 4 + 2 for r in range(13)] + list(range(0, 52, 4)))
    got = _run(node, tmp_path, [{"hole": h, "board": b, "any": False} for h, b in deals])
    for (hole, board), play in zip(deals, got):
        want = best_combo(hole, board)
        assert play == {"hole": want["hole"], "board": want["board"]}, (hole, board, want["label"])
    # too few cards known (and face-down / empty places): nothing to light
    none = _run(node, tmp_path, [{"hole": [1, 2, 3, 4, 5], "board": [9, 10], "any": False},
                                 {"hole": [1], "board": [9, 10, 11], "any": False},
                                 {"hole": [-1, -1, 3, None, 5], "board": [9, 10, 11], "any": False}])
    assert none[:2] == [None, None] and none[2] == {"hole": [3, 5], "board": [9, 10, 11]}


def test_one_board_plays_the_best_five_with_the_fewest_of_your_cards(node, tmp_path):
    rng = random.Random(7)
    deals = _deals(rng, 800, (2,), (3, 4, 5)) + _deals(rng, 400, (2,), (5,), deck=range(32, 52))
    got = _run(node, tmp_path, [{"hole": h, "board": b, "any": True} for h, b in deals])
    for (hole, board), play in zip(deals, got):
        five = play["hole"] + play["board"]
        assert len(five) == 5 and set(play["hole"]) <= set(hole) and set(play["board"]) <= set(board)
        value = _eval5(five)[:2]
        assert value == _best_nlh(hole, board)[:2]
        for n in range(len(play["hole"])):  # (fewer of your cards never makes the same hand)
            for h in combinations(sorted(hole), n):
                for b in combinations(sorted(board), 5 - n):
                    assert _eval5([*h, *b])[:2] < value
