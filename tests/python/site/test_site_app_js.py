"""The Study / Trainer client's pure helpers, exercised in Node (site FE-018),
and the shape of the split client (FE-019).

The client is plain browser scripts (app.core.js … app.js) that index.html
loads in order into one global scope. They are evaluated the same way — in
index.html's order, in one `vm` context with a small stub page — so a file
that uses a later one while it loads fails here. Then the money / label /
card-order / spot-link helpers are checked on fixed inputs: number
formatting in both units, bet-vs-raise labels as raise-TO totals, the
street-commit walk (incl. preflop blinds), next-empty-slot order, preset
chip labels, share-link encoding, error wording, and the Trainer → Study
spot conversion. Skipped without Node.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[3] / "python" / "plo5bp" / "ui" / "static"


def app_scripts() -> list[Path]:
    """The client's scripts, in the order index.html loads them."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    names = re.findall(r'<script src="/static/(app(?:\.[a-z]+)?\.js)"></script>', html)
    return [STATIC / n for n in names]

HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const files = process.argv.slice(2);
const noop = () => {};
const el = () => ({
  style: {}, dataset: {}, classList: { add: noop, remove: noop, toggle: noop, contains: () => false },
  addEventListener: noop, setAttribute: noop, removeAttribute: noop, appendChild: noop,
  querySelector: () => null, querySelectorAll: () => [], children: [], hidden: false,
});
const store = {};
const ctx = {
  console, setTimeout, clearTimeout, setInterval, clearInterval, URL, URLSearchParams,
  TextEncoder, TextDecoder, btoa, atob, Promise, JSON, Math, Number, String, Array, Object, Set, Map,
  localStorage: { getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); }, removeItem: (k) => { delete store[k]; } },
  location: { search: "", href: "http://x/", origin: "http://x", pathname: "/", hash: "" },
  history: { replaceState: noop, state: null },
  navigator: {},
  matchMedia: () => ({ matches: false, addEventListener: noop }),
  document: {
    addEventListener: noop, getElementById: () => null, querySelector: () => null,
    querySelectorAll: () => [], createElement: el, createElementNS: el, body: el(),
    documentElement: { dataset: {} }, contains: () => false,
  },
};
ctx.window = ctx;
vm.createContext(ctx);
for (const f of files) {
  vm.runInContext(fs.readFileSync(f, "utf8"), ctx, { filename: f.split(/[\\/]/).pop() });
}
const run = (code) => vm.runInContext(code, ctx);
const out = {};
const S = (fmt, extra) => Object.assign({ format: fmt || "plo5_double_bomb",
  chip_scale: { bb_chips: 10000, ante_chips: 30000, dollars_per_bb: 2 } }, extra || {});
ctx.S = S;

// --- formatting (CPY-018) and units (ST-019)
out.fmt_bb = run(`UI.unit = "bb"; [formatUnit(200000, S()), formatUnit(25000, S()), formatUnit(33333, S()), formatUnit(12500000, S()), formatUnit(0, S())]`);
out.fmt_usd = run(`UI.unit = "$"; UI.rates = {}; [formatUnit(200000, S()), formatUnit(25000, S()), formatUnit(12500000, S()), formatUnit(15, S())]`);
out.rate_shared = run(`UI.unit = "$"; UI.rates = {plo5_double_bomb: 20}; [formatUnit(20000, S()), formatUnit(20000, S(null, {trainer: {}, chip_scale: {bb_chips: 10000, ante_chips: 0, dollars_per_bb: 2}}))]`);
out.signed = run(`UI.unit = "bb"; [fmtSignedValue(1.234, S()), fmtSignedValue(-2, S()), fmtSignedValue(0, S()), fmtSignedValue(null, S())]`);
out.signed_usd = run(`UI.unit = "$"; UI.rates = {plo5_double_bomb: 2}; [fmtSignedValue(-12, S()), fmtSignedValue(0.5, S())]`);
out.input_value = run(`UI.unit = "bb"; [unitInputValue(33333, S()), unitInputValue(200000, S())]`);
out.parse = run(`UI.unit = "$"; UI.rates = {plo5_double_bomb: 2}; [parseToChips("4", S()), parseToChips("-1", S()), parseToChips("x", S())]`);

// --- labels (ST-024 / F11): raise-TO totals, bet vs raise
out.labels = run(`UI.unit = "bb"; [
  gateActionLabel("raise", 40000, 0, S(), 0, "flop"),
  gateActionLabel("raise", 60000, 20000, S(), 20000, "flop"),
  gateActionLabel("raise", 60000, 20000, S(), undefined, "flop"),
  gateActionLabel("check_call", 0, 20000, S(), 0, "flop"),
  gateActionLabel("check_call", 0, 0, S(), 0, "flop"),
  gateActionLabel("fold", 0, 0, S(), 0, "flop")]`);
out.presets = run(`[presetChipLabel(25), presetChipLabel(33), presetChipLabel(100), presetChipLabel(150)]`);

// --- street-commit walk with NLH blinds seeded from the payload
out.walk = run(`JSON.stringify(streetCommitWalk(
  { chip_scale: { bb_chips: 10000, ante_chips: 0 }, starting_stacks_chips: [1e6, 1e6, 1e6],
    seats: [{seat: 0, committed_total_bb: 0}, {seat: 1, committed_total_bb: 0.5}, {seat: 2, committed_total_bb: 1}],
    history: [] },
  [{seat: 0, street: "preflop", chips: 30000}, {seat: 1, street: "preflop", chips: 25000}, {seat: 0, street: "flop", chips: 0}],
  (e) => e.chips))`);

// --- card entry order
out.next_slot = run(`const cs = {card_spec: {hero_hole: [1,2,null,4,5], flop_a: [null,null,null], flop_b: [null,null,null], turn: [null,null], river: [null,null]}};
  [JSON.stringify(nextEmptySlot(cs, null)), JSON.stringify(nextEmptySlot(cs, {key: "hero_hole", index: 2})),
   JSON.stringify(nextEmptySlot({card_spec: {hero_hole: [1], flop_a: [2], flop_b: [], turn: [3], river: [4]}}, null))]`);

// --- share links (FEAT-016)
out.spot = run(`const sp = {format: "plo5_double_bomb", num_seats: 3, button_seat: 1, starting_stacks: [200000, 300000, 400000],
  ante_chips: 30000, bb_chips: 10000, hero_hole: [51, 47, null, 39, 35], flop_a: [0, 4, 8], flop_b: [null, null, null],
  turn: [null, null], river: [null, null], actions: [{gate: "check_call"}, {gate: "raise", chips: 45000}, {gate: "fold"}]};
  const code = encodeSpot(sp); JSON.stringify({ok: /^[A-Za-z0-9_-]+$/.test(code), back: decodeSpot(code)})`);

// --- error wording (FE-013)
out.errors = run(`[friendlyDetail(400, "chips 34000 out of raise range [1, 2]"), friendlyDetail(502, ""), friendlyDetail(500, "Traceback"),
  friendlyDetail(400, "Nothing to see"), friendlyDetail(0, "")]`);

// --- trainer review -> study spot (FEAT-017)
out.trainer_spot = run(`JSON.stringify(trainerSpotAt({
  format: "plo5_double_bomb", num_seats: 4, hero_seat: 2, button_seat: 3,
  starting_stacks_chips: [100, 200, 300, 400], chip_scale: {ante_chips: 30, bb_chips: 10},
  card_spec: {hero_hole: [1,2,3,4,5], flop_a: [6,7,8], flop_b: [9,10,11], turn: [12,13], river: [14,15]},
  trainer: {review: {nodes: [
    {actual_gate: "check_call", actual_chips: 0, street: "flop"},
    {actual_gate: "raise", actual_chips: 50, street: "flop"},
    {actual_gate: "fold", actual_chips: 0, street: "flop"}]}}}, 2))`);

// --- trainer end-of-hand line (ST-013 / CPY-013)
out.terminal = run(`UI.unit = "bb"; [
  terminalText({terminal: "showdown", hero_seat: 0, seats: [{folded: false}], trainer: {rewards_bb: [12.5]}, chip_scale: {bb_chips: 10000}}),
  terminalText({terminal: "fold_out", hero_seat: 0, seats: [{folded: true}], trainer: {rewards_bb: [-3]}, chip_scale: {bb_chips: 10000}}),
  terminalText({terminal: "fold_out", hero_seat: 1, seats: [{}, {folded: false}], trainer: {rewards_bb: [0, 7]}, chip_scale: {bb_chips: 10000}})]`);
// --- bet-size curve numbers (FEAT-022)
out.curve = run(`UI.unit = "bb"; const anchors = [0,1,2,3,4,5,6,7,8,9,10].map((k) => ({k, frac: k / 10, label: k === 0 ? "min" : k === 10 ? "pot" : (k * 10) + "%",
  chips: 10000 + k * 20000, prob: [0.1, 0, 0, 0.3, 0, 0, 0.4, 0, 0, 0, 0.2][k]}));
  const svg = betCurveSVG({anchors, rec_anchor: 6, rec_chips: 130000, pot_ref_chips: 210000, gate_distribution: [0, 0.4, 0.6]}, S({actor: null, seats: []}), true);
  JSON.stringify({hits: (svg.match(/class="bc-hit"/g) || []).length, top: (svg.match(/data-top="([^"]*)"/) || [])[1]})`);
process.stdout.write(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def js():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    proc = subprocess.run(
        [node, "-", *map(str, app_scripts())], input=HARNESS, capture_output=True, text=True,
        encoding="utf-8", timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_number_formatting(js):
    assert js["fmt_bb"] == ["20bb", "2.5bb", "3.33bb", "1,250bb", "0bb"]
    assert js["fmt_usd"] == ["$40", "$5", "$2,500", "$0"]
    assert js["input_value"] == ["3.33", "20"]
    assert js["parse"] == [20000, None, None]


def test_one_rate_for_study_and_trainer(js):
    # the viewer's rate wins over each mode's server default
    assert js["rate_shared"] == ["$40", "$40"]


def test_signed_values(js):
    assert js["signed"] == ["+1.23bb", "−2bb", "0bb", None]
    assert js["signed_usd"] == ["−$24", "+$1"]


def test_action_labels_are_raise_to_totals(js):
    assert js["labels"] == [
        "Bet 4bb", "Raise 8bb", "Raise +6bb", "Call", "Check", "Fold",
    ]
    assert js["presets"] == ["25%", "33%", "Pot", "150%"]


def test_street_commit_walk_seeds_blinds(js):
    walk = json.loads(js["walk"])
    assert walk[0] == {"before": 0, "after": 30000, "level": 10000}
    assert walk[1] == {"before": 5000, "after": 30000, "level": 30000}
    assert walk[2] == {"before": 0, "after": 0, "level": 0}


def test_next_empty_slot(js):
    a, b, c = (json.loads(x) for x in js["next_slot"])
    assert a == {"key": "hero_hole", "index": 2}
    assert b == {"key": "flop_a", "index": 0}
    assert c is None


def test_share_link_round_trip(js):
    got = json.loads(js["spot"])
    assert got["ok"] is True
    back = got["back"]
    assert back["hero_hole"] == [51, 47, None, 39, 35]
    assert back["actions"] == [
        {"gate": "check_call"}, {"gate": "raise", "chips": 45000}, {"gate": "fold"},
    ]
    assert back["starting_stacks"] == [200000, 300000, 400000]


def test_error_wording(js):
    rewrite, restart, crash, plain, network = js["errors"]
    assert "pick one inside the range" in rewrite and "34000" not in rewrite
    assert "restarting" in restart
    assert "Traceback" not in crash and "went wrong" in crash
    assert plain == "Nothing to see"
    assert "again" in network


def test_trainer_spot_rotates_hero_to_seat_zero(js):
    spot = json.loads(js["trainer_spot"])
    assert spot["num_seats"] == 4
    assert spot["starting_stacks"] == [300, 400, 100, 200]   # hero (seat 2) first
    assert spot["button_seat"] == 1                           # (3 - 2) mod 4
    assert spot["actions"] == [{"gate": "check_call"}, {"gate": "raise", "chips": 50}]
    # a flop decision: the turn and river aren't dealt yet
    assert spot["turn"] == [None, None] and spot["river"] == [None, None]
    assert spot["flop_a"] == [6, 7, 8]


def test_trainer_result_line(js):
    assert js["terminal"] == [
        "Showdown — you won 12.5bb.",
        "You folded — you lost 3bb.",
        "Everyone else folded — you won 7bb.",
    ]


def test_bet_curve_reads_out_its_numbers(js):
    curve = json.loads(js["curve"])
    assert curve["hits"] == 11
    assert curve["top"] == "Top sizes: 60% pot 40% · 30% pot 30% · Pot 20%"


# --- the split client (FE-019) ------------------------------------------------------

_TOP_NAME = re.compile(r"^(?:async\s+)?(?:function\*?|const|let|var|class)\s+([A-Za-z_$][\w$]*)", re.M)


def test_index_loads_the_whole_client_in_order():
    names = [p.name for p in app_scripts()]
    assert names == [
        "app.core.js", "app.table.js", "app.play.js", "app.study.js",
        "app.trainer.js", "app.topbar.js", "app.js",
    ]
    on_disk = sorted(p.name for p in STATIC.glob("app*.js"))
    assert sorted(names) == on_disk, "an app*.js file index.html doesn't load"


def test_each_part_is_strict_and_names_are_defined_once():
    seen: dict[str, str] = {}
    for path in app_scripts():
        text = path.read_text(encoding="utf-8")
        code = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("//")]
        assert code[0] == '"use strict";', f"{path.name}: \"use strict\" must come first"
        for name in _TOP_NAME.findall(text):
            # one global scope: a second declaration silently replaces the first
            assert name not in seen, f"{name} is declared in {seen.get(name)} and {path.name}"
            seen[name] = path.name


def test_no_part_builds_inline_handlers():
    for path in app_scripts():
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"""<[^>]+\son[a-z]+\s*=\s*\\?["']""", text), path.name
        assert "eval(" not in text and "new Function(" not in text, path.name
