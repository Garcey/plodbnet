"""Home-games dock (games.play.js), driven in Node with the real games.js as its core
and a tiny DOM (hg_mini_dom.js). 2026-09-28 improvements:

- one decision, one request: the pressed button shows it is sending, a second tap does
  nothing (HGT-011);
- "Confirm all-in" also asks on a call that puts the whole stack in, and a confirm that
  lands after the decision moved on sends nothing (HGT-020);
- a typed raise outside the window is shown adjusted on the first Enter, bet on the
  second (HGT-022); the box speaks big blinds in bb mode (HGT-021);
- pre-actions are one radio group, updated in place (A11Y-008);
- the between-hands strip: "Final ledger" on a closed table (HGT-026), "Deal now" during
  the countdown for the host only (HGT-024), the new-version refresh (OPS-039).
Skipped when Node is not installed."""
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


HARNESS = MINI_DOM + r"""
const fs = require("fs"), vm = require("vm");
function load() {
  const W = makeWorld();
  for (const id of ["actbar", "stage-box", "dock-left", "dock-right", "banner", "center", "hero-hand-labels", "toast-root"]) W.el("div", id);
  const acts = [], dialogs = [], toasts = [];
  const ctx = { console, JSON, Math, Number, String, Object, Array, Set, Map, Promise, Error,
    document: W.doc, location: { pathname: "/games/t/T1" }, history: { replaceState() {}, pushState() {} },
    setTimeout: W.setTimeout, clearTimeout: W.clearTimeout, setInterval: () => 1, clearInterval() {},
    performance: { now: () => W.now() }, matchMedia: () => ({ matches: false }), fetch: async () => { throw new Error("no network"); } };
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  let core = fs.readFileSync(process.argv[2], "utf8").replace(/\r\n/g, "\n").replace(/\ninit\(\);\s*$/, "\n");
  vm.runInContext(core, ctx);
  vm.runInContext(fs.readFileSync(process.argv[3], "utf8"), ctx);
  const HG = ctx.HG;
  let release = null;
  HG.core.act = (body) => { acts.push(body); return new Promise((res) => { release = () => res(null); }); };
  HG.ui = { toast: (m) => toasts.push(m), confirmDialog: (o) => new Promise((res) => dialogs.push({ o, res })),
    openModal() {}, openTopUp() {}, openLeave() {}, openHand() {}, openMenu() {}, copyInvite() {}, setRail(o, t) { ctx.__rail = t; },
    closeTop: () => false, renderDock: (st) => HG.play.render(st, st) };
  HG.uiState = { modals: [], drawer: null, menu: null };
  HG.table = { EMOTES: {} };
  HG.play.init();
  const $ = (id) => W.doc.getElementById(id);
  return { W, HG, $, acts, dialogs, toasts, release: () => release && release(), ctx };
}
function state(over) {
  const seats = [0, 1].map((i) => ({ seat: i, empty: false, name: "p" + i, user_id: i + 1, in_hand: true, folded: false,
    all_in: false, stack_cents: 4000, stack_chips: 400000, committed_this_street_cents: 0 }));
  return Object.assign({ id: "T1", epoch: "e", rev: 1, hand_no: 3, action_seq: 5, phase: "in_hand", status: "open",
    my_seat: 0, my_user_id: 1, actor: 0, is_host: false, is_member: true, running: true, seats,
    stakes: { bb_cents: 100, bb_chips: 10000, ante_cents: 300 }, legal: { fold: true, check_call: true, raise: true },
    to_call_cents: 200, to_call_chips: 20000, street_commit_chips: 0, pot_cents: 900, pot_chips: 90000,
    raise_bounds: { min_chips: 40000, max_chips: 110000 }, runout: { blocking: false }, settings: {},
    eligible_count: 2, last_hand_no: 2 }, over || {});
}
const flush = async () => { for (let i = 0; i < 20; i++) await Promise.resolve(); };
"""


def _run(node, tmp_path, body):
    return run_node(HARNESS + body, STATIC / "games.js", STATIC / "games.play.js", tmp=tmp_path)


def test_one_decision_one_request(node, tmp_path):
    got = _run(node, tmp_path, r"""
(async () => {
  const L = load(), s = state();
  L.HG.core.G.state = s; L.HG.play.render(s, s);
  L.$("fold-btn").click(); L.$("fold-btn").click(); L.$("check-btn").click();
  const during = ["fold-btn", "check-btn", "raise-go"].map((id) => [L.$(id).classList.contains("sending"), L.$(id).disabled]);
  L.release(); await flush();
  const after = ["fold-btn", "check-btn", "raise-go"].map((id) => [L.$(id).classList.contains("sending"), L.$(id).disabled]);
  console.log(JSON.stringify({ acts: L.acts, during, after }));
})();
""")
    assert got["acts"] == [{"gate": "fold"}]
    assert got["during"] == [[True, True], [False, True], [False, True]]
    assert got["after"] == [[False, False], [False, False], [False, False]]


def test_confirm_all_in_asks_on_a_call_too_and_never_acts_on_a_moved_decision(node, tmp_path):
    got = _run(node, tmp_path, r"""
(async () => {
  const out = {};
  let L = load();
  L.HG.core.G.prefs.confirmAllIn = true;
  let s = state({ to_call_cents: 4000 });
  L.HG.core.G.state = s; L.HG.play.render(s, s);
  L.$("check-btn").click(); await flush();
  out.asked = L.dialogs.map((d) => d.o.okLabel);
  L.dialogs[0].res(true); await flush();
  out.acts = L.acts.slice();
  // the clock acted for me while the dialog was open: nothing is sent
  L = load(); L.HG.core.G.prefs.confirmAllIn = true;
  s = state({ to_call_cents: 4000 });
  L.HG.core.G.state = s; L.HG.play.render(s, s);
  L.$("check-btn").click(); await flush();
  L.HG.core.G.state = state({ action_seq: 6, to_call_cents: 4000 });
  L.dialogs[0].res(true); await flush();
  out.moved = L.acts.slice();
  // a small call does not ask
  L = load(); L.HG.core.G.prefs.confirmAllIn = true;
  s = state(); L.HG.core.G.state = s; L.HG.play.render(s, s);
  L.$("check-btn").click(); await flush();
  out.small = { dialogs: L.dialogs.length, acts: L.acts.slice() };
  console.log(JSON.stringify(out));
})();
""")
    assert got["asked"] == ["Call all-in"]
    assert got["acts"] == [{"gate": "check_call"}]
    assert got["moved"] == []
    assert got["small"] == {"dialogs": 0, "acts": [{"gate": "check_call"}]}


def test_a_typed_raise_outside_the_window_is_shown_before_it_is_bet(node, tmp_path):
    got = _run(node, tmp_path, r"""
(async () => {
  const out = {};
  const L = load(), s = state();
  L.HG.core.G.state = s; L.HG.play.render(s, s);
  const inp = L.$("raise-input");
  inp.value = "999"; inp.dispatch("input");
  inp.dispatch("keydown", { key: "Enter" }); await flush();
  out.first = { acts: L.acts.length, box: inp.value, toast: L.toasts[0] || "" };
  inp.dispatch("keydown", { key: "Enter" }); await flush();
  out.second = L.acts.slice();
  // bb mode: the box shows and reads big blinds
  const B = load(); B.HG.core.G.prefs.unit = "bb";
  B.HG.core.G.state = s; B.HG.play.render(s, s);
  const bi = B.$("raise-input");
  out.bb = { box: bi.value, unit: bi.parentNode.classList.contains("unit-bb") };
  bi.value = "6"; bi.dispatch("input");
  bi.dispatch("keydown", { key: "Enter" }); await flush();
  out.bbAct = B.acts.slice();
  console.log(JSON.stringify(out));
})();
""")
    assert got["first"]["acts"] == 0 and got["first"]["box"] == "11.00"
    assert "The most you can bet is $11.00" in got["first"]["toast"]
    assert got["second"] == [{"gate": "raise", "raise_to_chips": 110000}]
    assert got["bb"] == {"box": "4", "unit": True}
    assert got["bbAct"] == [{"gate": "raise", "raise_to_chips": 60000}]


def test_pre_actions_are_one_radio_group_updated_in_place(node, tmp_path):
    got = _run(node, tmp_path, r"""
const L = load(), s = state({ actor: 1, legal: { fold: false, check_call: false, raise: false }, to_call_cents: 0 });
L.HG.core.G.state = s; L.HG.play.render(s, s);
const row = L.$("pre-row");
const boxes = () => row.querySelectorAll("input[data-pre]");
const first = boxes()[1];
const kinds = boxes().map((b) => [b.dataset.pre, b.type, b.getAttribute("name")]);
row.querySelectorAll("label")[1].click();
const picked = [L.HG.core.G.preAction, boxes().map((b) => b.checked), boxes()[1] === first];
row.querySelectorAll("label")[1].click();
const cleared = [L.HG.core.G.preAction, boxes().map((b) => b.checked), boxes()[1] === first];
console.log(JSON.stringify({ role: row.getAttribute("role"), kinds, picked, cleared }));
""")
    assert got["role"] == "radiogroup"
    assert got["kinds"] == [["check_fold", "radio", "pre-action"], ["check", "radio", "pre-action"], ["call_any", "radio", "pre-action"]]
    assert got["picked"] == ["check", [False, True, False], True]
    assert got["cleared"] == [None, [False, False, False], True]


def test_the_between_hands_strip(node, tmp_path):
    got = _run(node, tmp_path, r"""
const out = {};
const strip = (over, core) => {
  const L = load(); Object.assign(L.HG.core.G, core || {});
  const s = state(Object.assign({ phase: "waiting", actor: null, legal: { fold: false, check_call: false, raise: false } }, over));
  L.HG.core.G.state = s; L.HG.play.render(s, s);
  return { text: L.$("status-strip").textContent, L };
};
let r = strip({ status: "closed" });
out.closed = r.text;
r.L.$("status-strip").querySelectorAll("button")[0].click();
out.ledger = r.L.ctx.__rail;
out.guest = strip({ can_deal: true, next_deal_in_secs: 4 }).text;
out.host = strip({ can_deal: true, next_deal_in_secs: 4, is_host: true }).text;
out.update = strip({}, { newBuild: "abc" }).text;
console.log(JSON.stringify(out));
""")
    assert "Final ledger" in got["closed"] and got["ledger"] == "ledger"
    assert "Deal now" not in got["guest"]
    assert "Deal now" in got["host"]
    assert "New version · Refresh" in got["update"]


def test_the_host_sees_one_start_button(node, tmp_path):
    """HGT-004: before the first hand the felt's banner has it, later the dock's strip, and
    only a host with neither (not seated) gets the top bar's."""
    got = _run(node, tmp_path, r"""
const L = load();
const idle = (over) => state(Object.assign({ phase: "waiting", actor: null, running: false, is_host: true, legal: { fold: false, check_call: false, raise: false } }, over));
const strip = (s) => { L.HG.core.G.state = s; L.HG.play.render(s, s); return L.$("status-strip").textContent; };
const first = idle({ hand_no: 0 });
const later = idle({ hand_no: 5 });
const away = idle({ hand_no: 5, my_seat: null });
console.log(JSON.stringify({
  first: [L.HG.play.startOnFelt(first), strip(first).includes("Start game")],
  later: [L.HG.play.startOnFelt(later), strip(later).includes("Start game")],
  away: L.HG.play.startOnFelt(away),
}));
""")
    assert got["first"] == [True, False]  # (the banner's Start; the strip doesn't repeat it)
    assert got["later"] == [True, True]
    assert got["away"] is False  # (the top bar keeps it)
