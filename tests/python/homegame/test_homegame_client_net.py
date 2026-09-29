"""Home-games client: amounts, errors and the connection (2026-09-28 improvements).

- one amount reader for every box ("2,50" is 2.50, "1,000" a thousand) and one money
  style (a true minus sign; presets drop ".00") — HGT-022 / CPY-010;
- every request gives up after 10 s, and failures that never reached the server speak
  plain words instead of "Failed to fetch" or a proxy's HTML page — HGT-032 / HGT-031;
- start-up retries a hiccup instead of showing "Not Found"; a signed-out user is told so —
  HGT-013 / HGT-014;
- the lobby has an offline state — HGT-034; the red "Connection lost" bar waits out a
  blip — HGT-033; toasts: errors stay longer, a tap dismisses, a phone table shows one.

Runs games.js / games.ui.js in Node (a tiny DOM + fake timers); skipped without Node."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hg_client_tools import MINI_DOM, STATIC, node_exe, run_node  # noqa: E402


@pytest.fixture(scope="module")
def node():
    if node_exe() is None:
        pytest.skip("node is not installed")
    return node_exe()


CORE = MINI_DOM + r"""
const fs = require("fs"), vm = require("vm");
function loadCore(fetchImpl, extra) {
  const W = makeWorld();
  const ctx = Object.assign({
    console, JSON, Math, Number, String, Object, Array, Set, Map, Promise, Error, AbortController,
    document: W.doc, location: { pathname: "/games", reload() { ctx.__reloads = (ctx.__reloads || 0) + 1; } },
    history: { replaceState() {}, pushState() {} },
    setTimeout: W.setTimeout, clearTimeout: W.clearTimeout, setInterval: () => 1, clearInterval() {},
    performance: { now: () => W.now() }, fetch: fetchImpl,
  }, extra || {});
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  let src = fs.readFileSync(process.argv[2], "utf8").replace(/\r\n/g, "\n").replace(/\ninit\(\);\s*$/, "\n");
  src += "\n;globalThis.__t = { G, j, readAmount, toCents, dollars, fmtAmt, whoAmI, lobbyTick, setConn };";
  vm.runInContext(src, ctx);
  return { ctx, W, t: ctx.__t };
}
const flush = async () => { for (let i = 0; i < 20; i++) await Promise.resolve(); };
const reply = (status, body, ct) => ({ ok: status < 400, status, statusText: String(status),
  headers: { get: () => ct || "application/json" }, json: async () => body, text: async () => (typeof body === "string" ? body : JSON.stringify(body)) });
"""


def test_one_amount_reader_and_one_money_style(node, tmp_path):
    script = CORE + r"""
const { t } = loadCore(async () => reply(200, {}));
const reads = ["2,50", "1,000", "1,250.50", "$40", " 12 ", "0,5", "7.25", "3bb", "abc", "", "-5", "−5", "1,2345"];
const out = { reads: reads.map((x) => t.readAmount(x)), cents: ["2,50", "", "x", "$1,000.10"].map((x) => t.toCents(x)),
  fmt: [t.dollars(-500), t.dollars(500), t.dollars(4000, true), t.dollars(4050, true), t.dollars(-4000, true)] };
t.G.prefs.unit = "bb";
out.bb = t.fmtAmt(-250, { stakes: { bb_cents: 100 } });
console.log(JSON.stringify(out));
"""
    got = run_node(script, STATIC / "games.js", tmp=tmp_path)
    assert got["reads"] == [2.5, 1000, 1250.5, 40, 12, 0.5, 7.25, 3, None, None, -5, -5, 12345]
    assert got["cents"] == [250, 0, None, 100010]
    assert got["fmt"] == ["−$5.00", "$5.00", "$40", "$40.50", "−$40"]
    assert got["bb"] == "−2.5 bb"


def test_requests_time_out_and_speak_plain_words(node, tmp_path):
    script = CORE + r"""
(async () => {
  const out = {};
  const grab = async (p) => { try { await p; return null; } catch (e) { return { msg: e.message, network: !!e.network, status: e.status }; } };
  // offline: fetch itself fails
  let L = loadCore(async () => { throw new TypeError("Failed to fetch"); });
  out.offline = await grab(L.t.j("/x"));
  // a proxy's HTML page during a restart
  L = loadCore(async () => reply(502, "<html>Bad gateway</html>", "text/html"));
  out.proxy = await grab(L.t.j("/x"));
  // the server's own answers keep their words
  L = loadCore(async () => reply(400, { detail: "buy-in must be at least $10.00" }));
  out.server = await grab(L.t.j("/x"));
  L = loadCore(async () => reply(404, "Not Found", "text/plain"));
  out.missing = await grab(L.t.j("/x"));
  // a request that hangs gives up after 10 s
  L = loadCore((url, o) => new Promise((_, rej) => o.signal.addEventListener("abort", () => rej(Object.assign(new Error("aborted"), { name: "AbortError" })))));
  const p = grab(L.t.j("/x"));
  L.W.advance(9990); await flush();
  let settled = false; p.then(() => { settled = true; });
  await flush();
  out.before = settled;
  L.W.advance(20); await flush();
  out.timeout = await p;
  console.log(JSON.stringify(out));
})();
"""
    got = run_node(script, STATIC / "games.js", tmp=tmp_path)
    assert got["offline"]["network"] and "Can't reach the server" in got["offline"]["msg"]
    assert got["proxy"]["network"] and "restarting" in got["proxy"]["msg"] and "html" not in got["proxy"]["msg"].lower()
    assert got["server"] == {"msg": "buy-in must be at least $10.00", "network": False, "status": 400}
    assert got["missing"]["status"] == 404 and not got["missing"]["network"]
    assert got["before"] is False
    assert got["timeout"]["network"] and "too long" in got["timeout"]["msg"]


def test_start_up_retries_a_hiccup_and_says_signed_out(node, tmp_path):
    script = CORE + r"""
(async () => {
  const out = {};
  // /me fails twice (a deploy's restart), then answers
  let calls = 0;
  let L = loadCore(async () => { calls++; if (calls <= 2) throw new TypeError("Failed to fetch"); return reply(200, { signed_in: true, homegame: true, name: "A" }); });
  const seen = [];
  L.ctx.HG.ui = { bootProblem: (msg) => seen.push(msg) };
  const p = L.t.whoAmI();
  await flush(); L.W.advance(1000); await flush(); L.W.advance(2000); await flush();
  const me = await p;
  out.retry = { calls, me: me && me.name, seen, body: L.W.doc.body.innerHTML };
  // signed out: say so (never "Not Found")
  L = loadCore(async () => reply(200, { signed_in: false }));
  let told = 0;
  L.ctx.HG.ui = { signedOut: () => { told++; } };
  out.signedOut = { me: await L.t.whoAmI(), told, body: L.W.doc.body.innerHTML };
  // an account without home games: Not Found
  L = loadCore(async () => reply(200, { signed_in: true, homegame: false }));
  L.ctx.HG.ui = {};
  out.none = { me: await L.t.whoAmI(), body: L.W.doc.body.innerHTML };
  console.log(JSON.stringify(out));
})();
"""
    got = run_node(script, STATIC / "games.js", tmp=tmp_path)
    r = got["retry"]
    assert r["calls"] == 3 and r["me"] == "A"
    assert len(r["seen"]) == 3 and "Can't reach the server" in r["seen"][0] and r["seen"][-1] is None
    assert "Not Found" not in r["body"]
    assert got["signedOut"] == {"me": None, "told": 1, "body": ""}
    assert got["none"]["me"] is None and "Not Found" in got["none"]["body"]


def test_the_lobby_says_when_it_cannot_reach_the_server(node, tmp_path):
    script = CORE + r"""
(async () => {
  let up = false;
  const L = loadCore(async (url) => { if (!up) throw new TypeError("Failed to fetch"); return reply(200, { clubs: [], tables: [] }); });
  const log = [];
  L.ctx.HG.ui = { lobbyOffline: (on) => log.push(on), renderLobby() {}, renderConn() {} };
  L.t.G.clubId = null;
  await L.t.lobbyTick(); await flush();
  const first = L.t.G.conn;
  await L.t.lobbyTick(); await flush();
  const second = L.t.G.conn;
  up = true;
  await L.t.lobbyTick(); await flush();
  console.log(JSON.stringify({ log, first, second, after: L.t.G.conn }));
})();
"""
    got = run_node(script, STATIC / "games.js", tmp=tmp_path)
    assert got["log"] == [True, True, False]
    assert got["first"] == "ok" and got["second"] == "off" and got["after"] == "ok"


UI = MINI_DOM + r"""
const fs = require("fs"), vm = require("vm");
function loadUi(width) {
  const W = makeWorld();
  const conn = W.el("span", "conn"); conn.appendChild(W.doc.createElement("i")); conn.appendChild(W.doc.createElement("span"));
  W.el("div", "toast-root");
  const ctx = { console, JSON, Math, Number, String, Object, Array, Set, Map, Promise, Error, Symbol, TypeError,
    document: W.doc, setTimeout: W.setTimeout, clearTimeout: W.clearTimeout, innerWidth: width || 1200, location: {} };
  ctx.globalThis = ctx;
  ctx.HG = { avatar: { hueOf: () => 0, initials: () => "" } };
  vm.createContext(ctx);
  // the real core (its markup helpers: html / put — FE-003), without its start-up
  vm.runInContext(fs.readFileSync(process.argv[3], "utf8").replace(/\r\n/g, "\n").replace(/\ninit\(\);\s*$/, "\n"), ctx);
  const core = ctx.HG.core;
  core.G.conn = "ok"; core.G.gameId = "T1";
  vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx);
  return { W, ui: ctx.HG.ui, core, toasts: () => W.doc.getElementById("toast-root").children.map((t) => t.textContent) };
}
"""


def test_the_connection_bar_waits_out_a_blip(node, tmp_path):
    script = UI + r"""
const { W, ui, core, toasts } = loadUi();
const bar = () => W.doc.getElementById("connbar");
const out = {};
core.G.conn = "off"; ui.renderConn();
out.label = W.doc.getElementById("conn").lastChild.textContent;
W.advance(2000); out.blipHidden = bar().hidden;
core.G.conn = "ok"; ui.renderConn(); W.advance(3000);
out.afterBlip = { hidden: bar().hidden, toasts: toasts() };
core.G.conn = "off"; ui.renderConn(); W.advance(2600);
out.outage = bar().hidden;
core.G.conn = "ok"; ui.renderConn();
out.back = { hidden: bar().hidden, toasts: toasts(), label: W.doc.getElementById("conn").lastChild.textContent };
console.log(JSON.stringify(out));
"""
    got = run_node(script, STATIC / "games.ui.js", STATIC / "games.js", tmp=tmp_path)
    assert got["label"] == "Reconnecting…"
    assert got["blipHidden"] is True
    assert got["afterBlip"] == {"hidden": True, "toasts": []}  # (no false alarm, no "Back online")
    assert got["outage"] is False
    assert got["back"] == {"hidden": True, "toasts": ["Back online"], "label": "Connected"}


def test_toasts_errors_stay_longer_a_tap_dismisses_and_a_phone_table_shows_one(node, tmp_path):
    script = UI + r"""
const out = {};
let L = loadUi(1200);
L.ui.toast("saved", "ok"); L.ui.toast("that buy-in is too small", "err");
L.W.advance(3900); out.desk = L.toasts();  // (3.4 s + its fade-out)
L.W.advance(4000); out.later = L.toasts();
L.ui.toast("tap me", "");
L.W.doc.getElementById("toast-root").children[0].click(); L.W.advance(450); out.tapped = L.toasts();
L = loadUi(390);
L.ui.toast("one", ""); L.ui.toast("two", ""); out.phone = L.toasts();
L.core.G.gameId = null; L.ui.toast("three", ""); out.phoneLobby = L.toasts();
console.log(JSON.stringify(out));
"""
    got = run_node(script, STATIC / "games.ui.js", STATIC / "games.js", tmp=tmp_path)
    assert got["desk"] == ["that buy-in is too small"]
    assert got["later"] == []
    assert got["tapped"] == []
    assert got["phone"] == ["two"]
    assert got["phoneLobby"] == ["two", "three"]
