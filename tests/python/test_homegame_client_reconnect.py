"""Home games client (games.js) after a server restart (2026-09-25): the live
stream gives up after a few errors and the client polls — it used to poll for the
rest of the night. Once a poll gets through again it re-opens the stream (with a
growing back-off). Evaluated in Node with a stub DOM, a fake EventSource and a
controllable clock; skipped when Node is not installed."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

GAMES_JS = Path(__file__).resolve().parents[2] / "python" / "plo5bp" / "ui" / "static" / "games.js"

HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
function el() {
  return { style: {}, dataset: {}, hidden: false, value: "", innerHTML: "", textContent: "",
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    setAttribute() {}, getAttribute() { return null; }, appendChild(c) { return c; },
    addEventListener() {}, querySelectorAll() { return []; }, querySelector() { return null; },
    closest() { return null; }, focus() {} };
}
let clock = 0;
const sources = [];
class FakeES { constructor(url) { this.url = url; this.readyState = 0; sources.push(this); } close() { this.readyState = 2; this.closed = true; } }
const timers = [];
const ctx = {
  console, JSON, Math, Number, String, Object, Array, Set, Map, Promise, Error,
  performance: { now: () => clock },
  EventSource: FakeES,
  document: { getElementById: () => el(), createElement: () => el(), createElementNS: () => el(), hidden: false, body: el() },
  location: { pathname: "/games" }, history: { replaceState() {}, pushState() {} },
  fetch: async (url) => ({ ok: true, status: 200, statusText: "200", headers: { get: () => "application/json" },
    json: async () => ({ id: "T1", epoch: "e1", rev: 1, hand_no: 0, phase: "waiting", seats: [], events: [] }),
    text: async () => "{}" }),
  setInterval: (fn) => { timers.push(fn); return timers.length; }, clearInterval: (i) => { timers[i - 1] = null; },
  setTimeout: (fn) => 0, clearTimeout() {},
};
ctx.globalThis = ctx;
vm.createContext(ctx);
let src = fs.readFileSync(process.argv[2], "utf8").replace(/\r\n/g, "\n").replace(/\ninit\(\);\s*$/, "\n");
src += "\n;globalThis.__api = { G, startLive };";
vm.runInContext(src, ctx, { filename: "games.js" });
const { G, startLive } = ctx.__api;
G.gameId = "T1";
const out = {};
(async () => {
  startLive();
  const first = sources[0];
  for (let k = 0; k < 4; k++) await first.onerror();
  out.gaveUp = first.closed === true && G.live === null;
  out.polling = timers.some((t) => t);
  out.retryAt = G.streamRetryAt;
  // a poll before the retry time: still polling, no new stream
  clock = 10000;
  await timers.filter(Boolean).pop()();
  out.streamsBefore = sources.length;
  // after 30 s a poll that gets through re-opens the push
  clock = 31000;
  await timers.filter(Boolean).pop()();
  out.streamsAfter = sources.length;
  out.liveAgain = G.live === sources[sources.length - 1];
  // it opens: the back-off is cleared
  sources[sources.length - 1].onopen();
  out.retryCleared = G.streamRetryAt === null;
  console.log(JSON.stringify(out));
})();
"""


@pytest.fixture(scope="module")
def node():
    exe = shutil.which("node")
    if exe is None:
        pytest.skip("node is not installed")
    return exe


def test_the_client_goes_back_to_the_live_stream_after_a_restart(node, tmp_path):
    h = tmp_path / "h.js"
    h.write_text(HARNESS, encoding="utf-8")
    r = subprocess.run([node, str(h), str(GAMES_JS)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["gaveUp"] and out["polling"], out
    assert out["retryAt"] == 30000, out
    assert out["streamsBefore"] == 1, "no new stream before the retry time"
    assert out["streamsAfter"] == 2 and out["liveAgain"], out
    assert out["retryCleared"], out
