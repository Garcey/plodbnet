"""Regression tests for the 2026-09-20 review, CFR desktop app — JavaScript races.

``static/app.js`` is plain browser JS with no build step and no test runner, so
this drives the REAL, unmodified file under Node with a tiny DOM + fetch stub
(no npm packages). The stub lets the test decide the ORDER in which responses
arrive — which is the whole point of a race test.

Skipped when ``node`` is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[2] / "python" / "plo5bp" / "cfr_app" / "static" / "app.js"

HARNESS = r"""
const path = process.argv[2];
const results = {};

// ---- minimal DOM -----------------------------------------------------------
function makeEl(name) {
  const classes = new Set();
  const el = {
    _name: name, _handlers: {}, _children: [], value: "", checked: false, disabled: false,
    textContent: "", innerHTML: "", title: "", className: "", dataset: {}, style: {}, files: null, min: "",
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
  return el;
}
const els = new Map();
const $ = (sel) => { if (!els.has(sel)) els.set(sel, makeEl(sel)); return els.get(sel); };
let bootFn = null;
global.document = {
  hidden: false,
  querySelector: $,
  querySelectorAll: () => [],
  createElement: (tag) => makeEl("<" + tag + ">"),
  addEventListener: (ev, fn) => { if (ev === "DOMContentLoaded") bootFn = fn; },
};
global.window = { CFR_TOKEN: "tok-123" };
global.FormData = class { append() {} };

// Timers are captured, never run on their own: the test fires them by hand.
const intervals = [];
global.setInterval = (fn, ms) => { intervals.push({ fn, ms, live: true }); return intervals.length; };
global.clearInterval = (id) => { if (intervals[id - 1]) intervals[id - 1].live = false; };
global.setTimeout = () => 0;
global.clearTimeout = () => {};
const liveTimers = (ms) => intervals.filter((t) => t.live && t.ms === ms);

// ---- fetch: /view is resolved by hand, everything else from a route table ----
const calls = [];
const pendingViews = [];
let holdJobs = false;
const pendingJobs = [];
const reply = (body) => ({ ok: true, status: 200, statusText: "OK",
  headers: { get: () => "application/json" }, json: async () => body, text: async () => JSON.stringify(body) });
const JOB = { job_id: "job123456789", status: "running", notes: [], root: {}, config: { max_iterations: 0 },
              report: null, iterations_run: 10, num_infosets: 5 };
const META = { size_presets: { standard: [330, 500, 750, 1000, 1500], micro: [500, 1000] }, presets: [], preflop_labels: [] };
const viewBody = (tag) => ({ summary: { root: { root_id: tag }, status: "ok", street: 3, num_infosets: 1, num_nodes: 1, kind: "solve_report" },
  nodes: [{ seat: 0, path: "open", label: "P0 · Open", num_hands: 1, aggregate: { mean_mix: {}, num_hands: 1 } }],
  matrix: { empty: true, cells: [] }, line_nav: { navigable: true, root_path: "open", root_seat: 0, by_path: {} },
  page: { rows: [], total: 0, offset: 0, limit: 150 }, runout: { selected: "", label: "", options: [], total: 0 },
  job: { job_id: JOB.job_id, status: "running", live: true } });

global.fetch = (url, opts = {}) => {
  calls.push({ url, method: opts.method || "GET", headers: opts.headers || {}, body: opts.body || null });
  if (url.includes("/view")) return new Promise((resolve) => pendingViews.push({ url, resolve }));
  if (url === "/api/jobs" && holdJobs) return new Promise((resolve) => pendingJobs.push(resolve));
  if (url === "/api/jobs") return Promise.resolve(reply({ jobs: [JOB], active: JOB }));
  if (url === "/api/health") return Promise.resolve(reply({ ok: true, rust_cfr: true }));
  if (url === "/api/meta") return Promise.resolve(reply(META));
  if (url.endsWith("/progress")) return Promise.resolve(reply({ ...JOB, has_live_strategy: true, elapsed_secs: 1 }));
  if (url === "/api/validate_root") return Promise.resolve(reply({ ok: true, root: { root_id: "r" }, ranges: {} }));
  if (url === "/api/solve") return Promise.resolve(reply(JOB));
  return Promise.resolve(reply({ ok: true }));
};
const flush = async () => { for (let i = 0; i < 8; i++) await new Promise((r) => setImmediate(r)); };
const count = (pred) => calls.filter(pred).length;

(async () => {
  require(path);                       // the real app.js (an IIFE that registers boot)
  $("#f-street").value = "3"; $("#f-size-preset").value = "standard"; $("#f-sizes").value = "330,500,750,1000,1500";
  await bootFn(); await flush();

  // A. boot() re-attaches to the job that is still running in the server.
  results.reattach = {
    polling: liveTimers(800).length === 1,
    playDisabled: $("#btn-solve").disabled === true,
    stopEnabled: $("#btn-stop").disabled === false,
    badge: $("#st-job").textContent,
  };

  // (Only needed when this harness is pointed at the pre-fix app.js, where a
  // reload leaves the page idle: press Play so the remaining checks can run.)
  if (!liveTimers(800).length) { await $("#btn-solve").fire("click"); await flush(); }

  // C. /api/jobs poller: a tick is skipped while the previous one is in flight.
  const poll = liveTimers(800)[0].fn;
  await flush();
  holdJobs = true;
  const before = count((c) => c.url === "/api/jobs");
  poll(); poll(); poll(); await flush();
  results.pollInFlight = { requestsWhileBusy: count((c) => c.url === "/api/jobs") - before };
  holdJobs = false; pendingJobs.splice(0).forEach((r) => r(reply({ jobs: [JOB], active: JOB }))); await flush();
  const b2 = count((c) => c.url === "/api/jobs"); poll(); await flush();
  results.pollInFlight.resumesAfterwards = count((c) => c.url === "/api/jobs") - b2 === 1;

  // Open the job in the viewer (first view request, resolved normally).
  $("#panel-viewer").classList.add("active");
  const opening = $("#btn-view-job").fire("click"); await flush();
  pendingViews.shift().resolve(reply(viewBody("INIT"))); await opening; await flush();
  results.opened = $("#view-title").textContent;

  // B. Two filter clicks; the SECOND response arrives first, the first one last.
  $("#v-hand").value = "AKs";
  const c1 = $("#btn-filter").fire("click"); await flush();
  const c2 = $("#btn-filter").fire("click"); await flush();
  const [first, second] = pendingViews.splice(0);
  second.resolve(reply(viewBody("NEWER"))); await flush();
  first.resolve(reply(viewBody("STALE"))); await Promise.all([c1, c2]); await flush();
  results.staleGuard = { title: $("#view-title").textContent, sentFilter: first.url.includes("hand_query=AKs") };

  // E. live /view ticker: in-flight guard + nothing fetched while the viewer is hidden.
  const live = liveTimers(1500)[0].fn;
  const v0 = count((c) => c.url.includes("/view"));
  live(); live(); live(); await flush();
  results.liveView = { requestsWhileBusy: count((c) => c.url.includes("/view")) - v0,
                       carriesHandFilter: pendingViews[0].url.includes("hand_query=AKs") };
  pendingViews.splice(0).forEach((p) => p.resolve(reply(viewBody("LIVE")))); await flush();
  $("#panel-viewer").classList.remove("active");
  const v1 = count((c) => c.url.includes("/view")); live(); await flush();
  results.liveView.requestsWhileTabHidden = count((c) => c.url.includes("/view")) - v1;
  $("#panel-viewer").classList.add("active"); document.hidden = true;
  live(); await flush();
  results.liveView.requestsWhileWindowHidden = count((c) => c.url.includes("/view")) - v1;
  document.hidden = false;

  // D. An edited "Raise sizes" box is what gets sent; typing flips the preset to custom.
  $("#f-sizes").value = "250, 400 oops";
  await $("#btn-validate").fire("click"); await flush();
  const sent = JSON.parse(calls.filter((c) => c.url === "/api/validate_root").pop().body);
  await $("#f-sizes").fire("input");
  results.sizes = { sent: sent.raise_sizes_pm, presetAfterTyping: $("#f-size-preset").value };

  // Every state-changing request is JSON + carries the per-launch token.
  const posts = calls.filter((c) => c.method === "POST");
  results.auth = { posts: posts.length,
    allJson: posts.every((c) => c.headers["Content-Type"] === "application/json"),
    allToken: posts.every((c) => c.headers["X-CFR-Token"] === "tok-123") };

  console.log("RESULTS:" + JSON.stringify(results));
})().catch((e) => { console.log("HARNESS_ERROR:" + (e && e.stack || e)); process.exit(1); });
"""


@pytest.fixture(scope="module")
def results(tmp_path_factory) -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    harness = tmp_path_factory.mktemp("cfr_js") / "harness.js"
    harness.write_text(HARNESS, encoding="utf-8")
    proc = subprocess.run(
        [node, str(harness), str(APP_JS)],
        capture_output=True,
        encoding="utf-8",  # Node writes UTF-8; the Windows locale default (cp1252) garbles "·"
        timeout=60,
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULTS:")), None)
    assert line, f"harness failed (rc={proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    return json.loads(line[len("RESULTS:"):])


def test_boot_reattaches_to_a_job_that_is_still_running(results: dict):
    # A page reload used to come back "idle": Play enabled (→ 409), no Stop button.
    assert results["reattach"] == {
        "polling": True, "playDisabled": True, "stopEnabled": True, "badge": "job123456789",
    }


def test_job_poller_skips_ticks_while_a_request_is_in_flight(results: dict):
    # Three timer ticks during one slow /api/jobs round trip → still ONE request.
    assert results["pollInFlight"] == {"requestsWhileBusy": 1, "resumesAfterwards": True}


def test_a_slow_response_cannot_overwrite_a_newer_selection(results: dict):
    assert results["opened"] == "Strategy · INIT"
    # The older request resolved LAST; without the ticket guard it won ("STALE").
    assert results["staleGuard"] == {"title": "Strategy · NEWER", "sentFilter": True}


def test_live_view_ticker_is_guarded_and_idle_while_hidden(results: dict):
    assert results["liveView"] == {
        "requestsWhileBusy": 1,  # three ticks, one request
        "carriesHandFilter": True,  # a live tick no longer wipes the hand filter
        "requestsWhileTabHidden": 0,
        "requestsWhileWindowHidden": 0,
    }


def test_an_edited_sizes_box_is_honoured(results: dict):
    # Preset still said "standard"; the box said 250,400 (+ junk, which is dropped).
    assert results["sizes"] == {"sent": [250, 400], "presetAfterTyping": "custom"}


def test_every_post_is_json_and_carries_the_launch_token(results: dict):
    assert results["auth"]["posts"] >= 1
    assert results["auth"]["allJson"] and results["auth"]["allToken"]
