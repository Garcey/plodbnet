"""CFR desktop app — UI behaviour added in the improvements pass, driven under Node.

Runs the REAL static/app.js with a tiny DOM + fetch stub (same approach as
test_review_cfr_app_js.py):

- TOOL-053: a 422 reads "Pot (bb): …" (not a JSON blob); an emptied number box
  is refused client-side and marked; a server that stops answering shows the
  banner after a few failed requests and clears it when it answers again.
- TOOL-021: every exploitability number says what kind it is.
- TOOL-054: "Compare with" is filled from the Library; the Kuhn self-test runs
  from the Diagnostics menu.
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
    textContent: "", innerHTML: "", title: "", className: "", dataset: {}, style: {}, files: null, min: "",
    classList: {
      add: (...c) => c.forEach((x) => classes.add(x)),
      remove: (...c) => c.forEach((x) => classes.delete(x)),
      toggle: (c, on) => { const want = on === undefined ? !classes.has(c) : !!on; want ? classes.add(c) : classes.delete(c); return want; },
      contains: (c) => classes.has(c),
    },
    addEventListener(ev, fn) { (el._handlers[ev] = el._handlers[ev] || []).push(fn); },
    appendChild(c) { el._children.push(c); return c; },
    remove() {}, click() {}, focus() { el._focused = true; },
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
const intervals = [];
global.setInterval = (fn, ms) => { intervals.push({ fn, ms, live: true }); return intervals.length; };
global.clearInterval = (id) => { if (intervals[id - 1]) intervals[id - 1].live = false; };
global.setTimeout = () => 0; global.clearTimeout = () => {};

let offline = false;
const calls = [];
const reply = (body, status = 200) => ({ ok: status < 400, status, statusText: "x",
  headers: { get: () => "application/json" }, json: async () => body, text: async () => JSON.stringify(body) });
const LIB = { items: [
  { path: "C:/d/a.json", name: "a.json", kind: "solve_report", street: 3, board_str: "Ac Kd 2h 7s 9c", exploitability_bb: 0.42, expl_kind: "exact_infoset" },
  { path: "C:/d/b.json", name: "b.json", kind: "solve_report", street: 0, exploitability_bb: 3.1, expl_kind: "mc_br_proxy" },
  { path: "C:/d/c.json", name: "00_CO_open.json", kind: "chart", street: 0 },
], count: 3 };
global.fetch = (url, opts = {}) => {
  calls.push({ url, method: opts.method || "GET", body: opts.body || null });
  if (offline) return Promise.reject(new TypeError("Failed to fetch"));
  if (url === "/api/health") return Promise.resolve(reply({ ok: true, rust_cfr: true, version: "9.9.9" }));
  if (url === "/api/meta") return Promise.resolve(reply({ size_presets: { standard: [500] }, presets: [], preflop_labels: [] }));
  if (url === "/api/jobs") return Promise.resolve(reply({ jobs: [], active: null }));
  if (url === "/api/library") return Promise.resolve(reply(LIB));
  if (url === "/api/solve") return Promise.resolve(reply({ detail: [
    { loc: ["body", "root", "pot_bb"], msg: "Input should be greater than 0", type: "x" },
    { loc: ["body", "config", "seed"], msg: "Input should be a valid integer", type: "x" }] }, 422));
  if (url === "/api/solve/kuhn") return Promise.resolve(reply({ job_id: "kuhn00000000", status: "queued", notes: ["kuhn"], root: { game: "kuhn" } }));
  return Promise.resolve(reply({ ok: true }));
};
const flush = async () => { for (let i = 0; i < 8; i++) await new Promise((r) => setImmediate(r)); };

(async () => {
  require(path);
  for (const [sel, v] of [["#f-street", "3"], ["#f-size-preset", "standard"], ["#f-sizes", "500"], ["#f-pot", "10"],
                          ["#f-stack", "50"], ["#f-seats", "2"], ["#f-bb", "10000"], ["#f-sb", "5000"], ["#f-ante", "5000"],
                          ["#c-iters", "200"], ["#c-threads", "1"], ["#c-seed", "0"], ["#c-poll", "50"]]) $(sel).value = v;
  $("#conn-banner").classList.add("hidden");  // as index.html ships it
  await bootFn(); await flush();
  results.version = $("#app-version").textContent;

  // TOOL-053: a 422 is "field: message", fields named like the form.
  await $("#btn-solve").fire("click"); await flush();
  results.fmt422 = $("#toast").textContent;

  // TOOL-053: an emptied box is refused before any request, and marked.
  $("#f-pot").value = "";
  const n0 = calls.filter((c) => c.url === "/api/solve").length;
  await $("#btn-solve").fire("click"); await flush();
  results.empty = { sent: calls.filter((c) => c.url === "/api/solve").length - n0,
                    msg: $("#toast").textContent, marked: $("#f-pot").classList.contains("invalid"),
                    focused: !!$("#f-pot")._focused };
  $("#f-pot").value = "10";

  // TOOL-053: the idle heartbeat notices a dead server after a few misses.
  const hb = intervals.filter((t) => t.live && t.ms === 4000)[0].fn;
  offline = true;
  await hb(); await flush(); await hb(); await flush();
  results.bannerAfter2 = !$("#conn-banner").classList.contains("hidden");
  await hb(); await flush();
  results.bannerAfter3 = !$("#conn-banner").classList.contains("hidden");
  results.bannerText = $("#conn-banner").textContent;
  results.badge = $("#job-badge").textContent;
  offline = false; await hb(); await flush();
  results.bannerCleared = $("#conn-banner").classList.contains("hidden");

  // TOOL-021 + TOOL-055: library rows carry kind labels and the expl kind.
  await $("#btn-refresh-lib").fire("click"); await flush();
  const rows = $("#lib-table tbody")._children.map((tr) => tr.innerHTML);
  results.libRows = rows;

  // TOOL-054: the compare list comes from the Library (charts left out).
  await $("#cmp-path").fire("focus"); await flush();
  results.cmpOptions = $("#cmp-path")._children.map((o) => [o.value, o.textContent]);

  // TOOL-054: Kuhn runs from the Diagnostics menu and the Solve tab shows it.
  $("#diag-menu").open = true;
  await $("#btn-kuhn").fire("click"); await flush();
  results.kuhn = { posted: calls.some((c) => c.url === "/api/solve/kuhn"), menuClosed: $("#diag-menu").open === false };
  console.log("RESULTS:" + JSON.stringify(results));
})().catch((e) => { console.log("HARNESS_ERROR:" + (e && e.stack || e)); process.exit(1); });
"""


@pytest.fixture(scope="module")
def results(tmp_path_factory) -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    harness = tmp_path_factory.mktemp("cfr_js_ui") / "harness.js"
    harness.write_text(HARNESS, encoding="utf-8")
    proc = subprocess.run([node, str(harness), str(APP_JS)], capture_output=True,
                          encoding="utf-8", timeout=60)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULTS:")), None)
    assert line, f"harness failed (rc={proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    return json.loads(line[len("RESULTS:"):])


def test_validation_errors_read_like_the_form(results):
    assert results["fmt422"] == (
        "Pot (bb): Input should be greater than 0 · Seed: Input should be a valid integer"
    )


def test_an_empty_number_box_is_refused_and_marked(results):
    assert results["empty"] == {"sent": 0, "msg": "Pot (bb) needs a number",
                                "marked": True, "focused": True}


def test_a_dead_server_shows_the_banner_and_recovers(results):
    assert results["bannerAfter2"] is False and results["bannerAfter3"] is True
    assert "stopped responding" in results["bannerText"]
    assert results["badge"] == "offline"
    assert results["bannerCleared"] is True


def test_library_rows_label_the_kind_of_every_number(results):
    a, b, c = results["libRows"]
    assert "0.420 bb · exact" in a and ">Solve<" in a
    assert "3.100 bb · proxy" in b and "not a Nash certificate" in b
    assert ">Chart<" in c


def test_compare_choices_come_from_the_library(results):
    assert results["cmpOptions"] == [
        ["", "Choose a solution…"],
        ["C:/d/a.json", "a.json · River · Ac Kd 2h 7s 9c"],
        ["C:/d/b.json", "b.json · Preflop"],
    ]


def test_kuhn_runs_from_the_diagnostics_menu(results):
    assert results["kuhn"] == {"posted": True, "menuClosed": True}
    assert results["version"] == "CFR Solver 9.9.9"
