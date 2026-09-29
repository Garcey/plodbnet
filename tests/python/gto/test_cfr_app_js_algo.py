"""CFR desktop app — the algorithm menu and Validate's answer, driven under Node.

Runs the REAL static/app.js with a tiny DOM + fetch stub (the approach of
test_cfr_app_js_ui.py):

- TOOL-008: heads-up river / turn roots pick full-range DCFR; options a root
  cannot use are disabled; the card abstraction follows.
- TOOL-017: the "threads" box says what it means for the chosen algorithm
  (threads / deals per iteration / not used — disabled, and 1 is sent).
- TOOL-030: the live exploitability-check interval is sent with the config.
- TOOL-032: Validate shows the ranges and what the solve will need (memory
  against the budget, infosets, tree), or why the solver would refuse it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[3] / "python" / "plo5bp" / "cfr_app" / "static" / "app.js"

HARNESS = r"""
const path = process.argv[2];
const results = {};
function makeEl(name) {
  const classes = new Set();
  const el = {
    _name: name, _handlers: {}, _children: [], value: "", checked: false, disabled: false, open: false,
    textContent: "", title: "", className: "", dataset: {}, style: {}, files: null, min: "",
    classList: {
      add: (...c) => c.forEach((x) => classes.add(x)),
      remove: (...c) => c.forEach((x) => classes.delete(x)),
      toggle: (c, on) => { const want = on === undefined ? !classes.has(c) : !!on; want ? classes.add(c) : classes.delete(c); return want; },
      contains: (c) => classes.has(c),
    },
    addEventListener(ev, fn) { (el._handlers[ev] = el._handlers[ev] || []).push(fn); },
    appendChild(c) { el._children.push(c); return c; },
    remove() {}, click() {}, focus() {},
    querySelector: (sel) => (sel.startsWith(".") ? null : makeEl(name + " " + sel)),
    querySelectorAll: () => [],
    fire(ev, arg) { return Promise.all((el._handlers[ev] || []).map((fn) => fn(arg || {}))); },
  };
  Object.defineProperty(el, "innerHTML", { get() { return el._html || ""; }, set(v) { el._html = v; if (v === "") el._children = []; } });
  return el;
}
const els = new Map();
const $ = (sel) => { if (!els.has(sel)) els.set(sel, makeEl(sel)); return els.get(sel); };
let bootFn = null;
global.document = {
  hidden: false, querySelector: $,
  querySelectorAll: (sel) => (sel === ".invalid" ? [...els.values()].filter((e) => e.classList.contains("invalid")) : []),
  createElement: (tag) => makeEl("<" + tag + ">"),
  addEventListener: (ev, fn) => { if (ev === "DOMContentLoaded") bootFn = fn; },
};
global.window = { CFR_TOKEN: "tok" };
global.FormData = class { append() {} };
global.setInterval = () => 1; global.clearInterval = () => {};
global.setTimeout = () => 0; global.clearTimeout = () => {};

const calls = [];
let estimate = { ok: true, algorithm: "dcfr_vector", est_mb: 412, budget_mb: 19491, est_infosets: 281060,
                 public_nodes: 260, refuse_reason: null };
const reply = (body, status = 200) => ({ ok: status < 400, status, statusText: "x",
  headers: { get: () => "application/json" }, json: async () => body, text: async () => JSON.stringify(body) });
const INFO = [
  { id: "dcfr_vector", label: "Full-range DCFR", for: "Heads-up river and turn", streets: [2, 3], hu_only: true },
  { id: "dcfr", label: "Sampled DCFR", for: "Postflop", streets: [1, 2, 3], hu_only: false },
  { id: "mccfr_es", label: "External-sampling MCCFR", for: "Preflop and multiway", streets: [0, 1, 2, 3], hu_only: false },
];
global.fetch = (url, opts = {}) => {
  calls.push({ url, body: opts.body ? JSON.parse(opts.body) : null });
  if (url === "/api/health") return Promise.resolve(reply({ ok: true, rust_cfr: true, version: "1" }));
  if (url === "/api/meta") return Promise.resolve(reply({ size_presets: { standard: [500] }, presets: [], preflop_labels: [],
                                                          algorithms: INFO.map((a) => a.id), algorithm_info: INFO }));
  if (url === "/api/jobs") return Promise.resolve(reply({ jobs: [], active: null }));
  if (url === "/api/validate_root") return Promise.resolve(reply({ ok: true, root: { root_id: "r1" },
                                                                   ranges: { oop: { full: true }, ip: { full: false, combos: 6 } } }));
  if (url === "/api/estimate") return Promise.resolve(reply(estimate));
  if (url === "/api/solve") return Promise.resolve(reply({ job_id: "j1", status: "queued", config: { max_iterations: 200 } }));
  return Promise.resolve(reply({ ok: true }));
};
const flush = async () => { for (let i = 0; i < 8; i++) await new Promise((r) => setImmediate(r)); };
const opt = (v) => ({ value: v, disabled: false, title: "" });

function snap() {
  const sel = $("#f-algo");
  return {
    algo: sel.value, abs: $("#f-abs").value, hint: $("#f-algo-hint").textContent,
    threadsLabel: $("#c-threads-label").textContent, threadsHint: $("#c-threads-hint").textContent,
    threadsDisabled: $("#c-threads").disabled, threads: $("#c-threads").value,
    disabled: sel.options.filter((o) => o.disabled).map((o) => o.value),
  };
}

(async () => {
  require(path);
  $("#f-algo").options = [opt("dcfr_vector"), opt("dcfr"), opt("mccfr_es")];
  for (const [sel, v] of [["#f-street", "3"], ["#f-size-preset", "standard"], ["#f-sizes", "500"], ["#f-pot", "10"],
                          ["#f-stack", "50"], ["#f-seats", "2"], ["#f-bb", "10000"], ["#f-sb", "5000"], ["#f-ante", "5000"],
                          ["#c-iters", "200"], ["#c-threads", "4"], ["#c-seed", "0"], ["#c-poll", "50"],
                          ["#c-expl-check", "10"], ["#f-algo", "dcfr_vector"], ["#f-abs", "none"]]) $(sel).value = v;
  $("#validate-result").classList.add("hidden");
  await bootFn(); await flush();
  results.river = snap();

  $("#f-street").value = "2"; await $("#f-street").fire("change"); await flush();
  results.turn = snap();
  $("#f-street").value = "1"; await $("#f-street").fire("change"); await flush();
  results.flop = snap();
  $("#f-street").value = "0"; await $("#f-street").fire("change"); await flush();
  results.preflop = snap();

  // MCCFR: the box is not used — sent as 1 even when it is empty.
  $("#c-threads").value = "";
  await $("#btn-solve").fire("click"); await flush();
  const solve = calls.filter((c) => c.url === "/api/solve").pop();
  results.solveCfg = solve ? solve.body.config : null;

  $("#f-street").value = "3"; $("#f-seats").value = "3"; await $("#f-seats").fire("change"); await flush();
  results.multiway = snap();
  $("#f-seats").value = "2"; await $("#f-seats").fire("change"); await flush();
  results.backToHu = snap();

  // The user's pick stays while it applies; an inapplicable one is replaced.
  $("#f-algo").value = "dcfr"; await $("#f-algo").fire("change"); await flush();
  results.userPick = snap();

  $("#c-threads").value = "4";
  await $("#btn-validate").fire("click"); await flush();
  const est = calls.filter((c) => c.url === "/api/estimate").pop();
  results.estimateSent = est ? { algo: est.body.config.algorithm, check: est.body.config.expl_check_secs, street: est.body.root.street } : null;
  results.validateOk = { html: $("#validate-result").innerHTML, hidden: $("#validate-result").classList.contains("hidden"),
                         bad: $("#validate-result").classList.contains("bad"), toast: $("#toast").textContent };

  // Any edit makes the answer stale: it goes away.
  await $("#panel-builder").fire("input"); await flush();
  results.hiddenAfterEdit = $("#validate-result").classList.contains("hidden");

  estimate = { ok: true, algorithm: "dcfr_vector", est_mb: 30000, budget_mb: 19491, est_infosets: 9e7,
               public_nodes: 1128, refuse_reason: "dcfr_vector: the full tree and its report need ~29.3 GB" };
  await $("#btn-validate").fire("click"); await flush();
  results.validateBad = { html: $("#validate-result").innerHTML, bad: $("#validate-result").classList.contains("bad"),
                          toast: $("#toast").textContent };
  // A number the user typed stays when the algorithm changes.
  $("#c-threads").value = "6"; await $("#c-threads").fire("input");
  $("#f-street").value = "1"; await $("#f-street").fire("change"); await flush();
  results.userThreads = snap().threads;
  results.defaultThreads = String(Math.max(1, Math.min(8,
    (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 4)));
  console.log("RESULTS:" + JSON.stringify(results));
})().catch((e) => { console.log("HARNESS_ERROR:" + (e && e.stack || e)); process.exit(1); });
"""


@pytest.fixture(scope="module")
def results(tmp_path_factory) -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    harness = tmp_path_factory.mktemp("cfr_js_algo") / "harness.js"
    harness.write_text(HARNESS, encoding="utf-8")
    proc = subprocess.run([node, str(harness), str(APP_JS)], capture_output=True,
                          encoding="utf-8", timeout=60)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULTS:")), None)
    assert line, f"harness failed (rc={proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    return json.loads(line[len("RESULTS:"):])


def test_heads_up_river_and_turn_pick_full_range_dcfr(results):
    for spot in ("river", "turn"):
        s = results[spot]
        assert s["algo"] == "dcfr_vector" and s["abs"] == "none" and s["disabled"] == []
        assert s["threadsLabel"] == "Threads" and not s["threadsDisabled"]
        assert "every runout" in s["hint"]
        assert s["threads"] == results["defaultThreads"]  # real threads by default (cores, at most 8)
    assert "rivers in parallel" in results["turn"]["threadsHint"]
    assert "one thread" in results["river"]["threadsHint"]


def test_flop_uses_sampled_dcfr_with_buckets(results):
    s = results["flop"]
    assert s["algo"] == "dcfr" and s["abs"] == "ochs"
    assert s["disabled"] == ["dcfr_vector"]
    assert s["threadsLabel"] == "Deals per iteration" and "not in parallel" in s["threadsHint"]
    assert s["threads"] == "1"  # one deal per iteration by default


def test_preflop_uses_mccfr_and_the_threads_box_is_off(results):
    s = results["preflop"]
    assert s["algo"] == "mccfr_es" and s["disabled"] == ["dcfr_vector", "dcfr"]
    assert s["threadsDisabled"] and "Not used" in s["threadsHint"]
    cfg = results["solveCfg"]
    assert cfg["thread_num"] == 1 and cfg["algorithm"] == "mccfr_es"
    assert cfg["expl_check_secs"] == 10  # TOOL-030


def test_seat_count_changes_what_applies(results):
    assert results["multiway"]["algo"] == "dcfr" and results["multiway"]["disabled"] == ["dcfr_vector"]
    assert results["backToHu"]["algo"] == "dcfr_vector" and results["backToHu"]["disabled"] == []
    assert results["userPick"]["algo"] == "dcfr"  # a pick that applies is kept
    assert results["userPick"]["threadsLabel"] == "Deals per iteration"
    assert results["userThreads"] == "6"


def test_validate_shows_ranges_and_what_the_solve_needs(results):
    assert results["estimateSent"] == {"algo": "dcfr", "check": 10, "street": 3}
    ok = results["validateOk"]
    assert not ok["hidden"] and not ok["bad"]
    html = ok["html"]
    assert "Root OK" in html and "100%" in html and "6 combos" in html
    assert "≈ 412 MB of 19 GB" in html and "≈ 281k" in html and "Full-range DCFR" in html
    assert 'class="mem-meter"' in html
    assert ok["toast"] == "Root OK: r1"
    assert results["hiddenAfterEdit"] is True


def test_validate_shows_a_refusal(results):
    bad = results["validateBad"]
    assert bad["bad"] and "would refuse" in bad["html"] and "29.3 GB" in bad["html"]
    assert "≈ 29 GB of 19 GB" in bad["html"]
    assert "refuse" in bad["toast"]
