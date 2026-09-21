"""Review 2026-09-20 H5 — static/ranges.js behaviour, driven under Node.

ranges.js is plain browser JS (no bundler, no test runner), so this loads it
into a `vm` context with a minimal DOM / fetch stub and asserts on the `RG`
state machine:

- a FAILED rebranch restores the viewed node (not just the line), so the next
  click still truncates there;
- the custom box is a raise-TO total and the wire entry is the raise-BY delta
  (typed total minus the actor's street commit);
- size buttons show `to_bb` but send the server's exact `chips_bb` delta;
- the checkpoint name is escaped before it reaches innerHTML.

Skipped when `node` is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

RANGES_JS = (
    Path(__file__).resolve().parents[2]
    / "python" / "plo5bp" / "ui" / "static" / "ranges.js"
)

HARNESS = r"""
const fs = require("fs");
const vm = require("vm");

class El {
  constructor(id) {
    this.id = id; this.children = []; this.listeners = {}; this._q = new Map();
    this._cls = new Set(); this.style = {}; this.dataset = {};
    this.innerHTML = ""; this.textContent = ""; this.value = "";
    this.hidden = false; this.disabled = false; this.title = "";
    const self = this;
    this.classList = {
      add: (c) => self._cls.add(c), remove: (c) => self._cls.delete(c),
      contains: (c) => self._cls.has(c),
      toggle: (c, on) => {
        const want = on === undefined ? !self._cls.has(c) : !!on;
        if (want) self._cls.add(c); else self._cls.delete(c);
      },
    };
  }
  appendChild(c) { this.children.push(c); return c; }
  addEventListener(t, fn) { (this.listeners[t] = this.listeners[t] || []).push(fn); }
  querySelector(sel) {
    if (!this._q.has(sel)) this._q.set(sel, new El(sel));
    return this._q.get(sel);
  }
  click() {
    for (const f of this.listeners.click || []) f({ target: this, currentTarget: this });
  }
}

const els = new Map();
const documentStub = {
  getElementById(id) { if (!els.has(id)) els.set(id, new El(id)); return els.get(id); },
  createElement() { return new El(); },
  querySelectorAll() { return []; },
  body: new El("body"),
};

const toasts = [];
const sent = [];
const queue = [];   // responses, consumed in order
function fetchStub(_url, opts) {
  sent.push(JSON.parse(opts.body));
  const r = queue.shift();
  if (!r) throw new Error("fetch queue empty");
  return Promise.resolve({
    ok: r.ok, status: r.status || (r.ok ? 200 : 400), json: async () => r.body,
  });
}

const ctx = vm.createContext({
  document: documentStub, window: {}, fetch: fetchStub, console,
  showToast: (m) => toasts.push(m),
  setInterval: () => 0, clearInterval: () => {}, setTimeout: () => 0,
  JSON, Math, Number, String, Object, Array, Set, Map, Promise, Error, parseFloat, parseInt,
});
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx, { filename: "ranges.js" });
const RG = vm.runInContext("RG", ctx);
const call = (src) => vm.runInContext(src, ctx);
const flush = () => new Promise((r) => setImmediate(r));

function okBody(n, extra) {
  const seq = [];
  for (let i = 0; i < n; i++) {
    seq.push({ i, t: "a", gate: "check_call", position: "P" + i, street: "preflop",
               chips_bb: null, to_bb: null });
  }
  return Object.assign({
    node: n, num_entries: n, sequence: seq, terminal: false, awaiting: null,
    model: { loaded: true, checkpoint: "tiny.pt" },
    state: {
      position: "BB", street: "preflop", pot_bb: 8, to_call_bb: 1.5,
      legal: { fold: true, check_call: true, raise: true },
      min_raise_bb: 3, max_raise_bb: 98.5,
      actor_commit_bb: 1, min_raise_to_bb: 4, max_raise_to_bb: 99.5,
      anchors: [
        { k: 0, label: "min", allin: false, chips_bb: 3, to_bb: 4 },
        { k: 11, label: "ALL-IN", allin: true, chips_bb: 98.5, to_bb: 99.5 },
      ],
    },
  }, extra || {});
}

(async () => {
  const out = {};

  // --- build a 3-action line, then VIEW node 1 -------------------------------------
  queue.push({ ok: true, body: okBody(0) });
  call("rgQuery()"); await flush();
  for (let i = 1; i <= 3; i++) {
    queue.push({ ok: true, body: okBody(i) });
    call('rgMutate(() => RG.line.push({ t: "a", gate: "check_call" }))');
    await flush();
  }
  out.built_len = RG.line.length;
  queue.push({ ok: true, body: okBody(3) });
  call("RG.node = 1; rgQuery()"); await flush();
  out.viewing = RG.node;

  // --- a rebranch the server REJECTS -----------------------------------------------
  queue.push({ ok: false, status: 400, body: { detail: "line entry 1: nope" } });
  call('rgMutate(() => RG.line.push({ t: "a", gate: "raise", chips_bb: 999 }))');
  await flush();
  out.after_fail_node = RG.node;
  out.after_fail_len = RG.line.length;
  out.after_fail_gates = RG.line.map((e) => e.gate);
  out.toast = toasts[toasts.length - 1];
  out.busy_after_fail = RG.busy;

  // --- the NEXT click must still truncate at the viewed node -------------------------
  queue.push({ ok: true, body: okBody(2) });
  call('rgMutate(() => RG.line.push({ t: "a", gate: "fold" }))');
  await flush();
  out.next_line = RG.line.map((e) => e.gate);
  out.next_sent_line = sent[sent.length - 1].line.map((e) => e.gate);
  out.next_node = RG.node;

  // --- a failed NAVIGATION restores the node too ---------------------------------------
  queue.push({ ok: false, status: 500, body: {} });
  call("RG.node = 0; rgQuery()"); await flush();
  out.nav_fail_node = RG.node;

  // --- a failed table-config change rolls seats + line back ----------------------------
  const before = { seats: RG.seats, len: RG.line.length };
  queue.push({ ok: false, status: 422, body: { detail: [{ msg: "bad seats" }] } });
  call("RG.seats = 3; RG.line = []; RG.node = null; rgQuery()"); await flush();
  out.cfg_restored = RG.seats === before.seats && RG.line.length === before.len;
  out.toast_422 = toasts[toasts.length - 1];
  out.seats_control = documentStub.getElementById("rg-seats").value;

  // --- raise-TO box -> raise-BY wire entry ---------------------------------------------
  const actions = documentStub.getElementById("rg-actions");
  const sizesRow = actions.children[2];
  const custom = actions.children[3];
  out.custom_html = custom.innerHTML;
  out.size_label = sizesRow.children[0].innerHTML;
  out.allin_label = sizesRow.children[1].innerHTML;
  queue.push({ ok: true, body: okBody(3) });
  custom.querySelector("#rg-custom-bb").value = "9";
  custom.querySelector("#rg-custom-go").click();
  await flush();
  out.custom_entry = RG.line[RG.line.length - 1];
  // float dust: 4.3 - 1 must go out as 3.3, not 3.3000000000000003
  const custom2 = documentStub.getElementById("rg-actions").children[3];
  queue.push({ ok: true, body: okBody(4) });
  custom2.querySelector("#rg-custom-bb").value = "4.3";
  custom2.querySelector("#rg-custom-go").click();
  await flush();
  out.dust_entry = RG.line[RG.line.length - 1];
  // size buttons send the server's exact delta
  const sizes2 = documentStub.getElementById("rg-actions").children[2];
  queue.push({ ok: true, body: okBody(5) });
  sizes2.children[0].click();
  await flush();
  out.size_entry = RG.line[RG.line.length - 1];

  // --- checkpoint name is escaped ----------------------------------------------------------
  queue.push({ ok: true, body: okBody(6, {
    model: { loaded: false, checkpoint: '<img src=x onerror="alert(1)">.pt' },
  }) });
  call("rgQuery()"); await flush();
  out.model_html = documentStub.getElementById("rg-model").innerHTML;

  // --- which network served the node ---------------------------------------------------------
  queue.push({ ok: true, body: okBody(6, { model: {
    loaded: true, checkpoint: "ppo_nlh.pt", backend: "ppo_fallback",
    reason: 'policy_net was not trained on <6-handed> "preflop" nodes', obs_form: "live",
  } }) });
  call("rgQuery()"); await flush();
  out.fallback_html = documentStub.getElementById("rg-model").innerHTML;
  queue.push({ ok: true, body: okBody(6, { model: {
    loaded: true, checkpoint: "gto.pt", backend: "policy_net", reason: null,
    obs_form: "canonical",
  } }) });
  call("rgQuery()"); await flush();
  out.gto_html = documentStub.getElementById("rg-model").innerHTML;
  queue.push({ ok: true, body: okBody(6, { model: {
    loaded: true, checkpoint: "ppo_nlh.pt", backend: "ppo", reason: null, obs_form: "live",
  } }) });
  call("rgQuery()"); await flush();
  out.ppo_html = documentStub.getElementById("rg-model").innerHTML;

  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.fixture(scope="module")
def js(tmp_path_factory):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    harness = tmp_path_factory.mktemp("rangesjs") / "harness.js"
    harness.write_text(HARNESS, encoding="utf-8")
    proc = subprocess.run(
        [node, str(harness), str(RANGES_JS)],
        capture_output=True, text=True, timeout=120, encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return json.loads(proc.stdout)


def test_failed_rebranch_restores_line_and_viewed_node(js):
    assert js["built_len"] == 3 and js["viewing"] == 1
    # rollback: the WHOLE prior state — line and the node still on screen
    assert js["after_fail_len"] == 3
    assert js["after_fail_gates"] == ["check_call"] * 3
    assert js["after_fail_node"] == 1, "viewed node lost on rollback"
    assert js["toast"] == "line entry 1: nope"
    assert js["busy_after_fail"] is False


def test_next_click_after_a_failed_rebranch_still_truncates(js):
    # viewing node 1 -> the new action replaces everything from entry 1 on
    assert js["next_line"] == ["check_call", "fold"]
    assert js["next_sent_line"] == ["check_call", "fold"]
    assert js["next_node"] is None


def test_failed_navigation_and_config_change_roll_back(js):
    assert js["nav_fail_node"] is None  # back to the live end that is rendered
    assert js["cfg_restored"] is True
    assert js["seats_control"] == "6"
    assert js["toast_422"] == "bad seats"  # pydantic detail list -> readable


def test_custom_box_is_raise_to_and_wire_is_raise_by(js):
    html = js["custom_html"]
    assert 'min="4"' in html and 'max="99.5"' in html
    assert "4–99.5bb" in html and ">Raise to<" in html
    # typed 9 (raise TO 9bb) with 1bb already in -> 8bb delta on the wire
    assert js["custom_entry"] == {"t": "a", "gate": "raise", "chips_bb": 8}
    assert js["dust_entry"]["chips_bb"] == 3.3


def test_size_buttons_show_totals_but_send_the_server_delta(js):
    assert "<i>4bb</i>" in js["size_label"] and "min" in js["size_label"]
    assert "ALL-IN" in js["allin_label"] and "<i>99.5bb</i>" in js["allin_label"]
    assert js["size_entry"] == {"t": "a", "gate": "raise", "chips_bb": 3}


def test_checkpoint_name_is_escaped(js):
    html = js["model_html"]
    assert "<img" not in html and "&lt;img" in html
    assert "onerror=&quot;alert(1)&quot;" in html
    assert "untrained" in html


def test_model_line_says_which_network_served_the_node(js):
    fb = js["fallback_html"]
    assert "PPO fallback" in fb and "ppo_nlh.pt" in fb
    # the reason rides in a title attribute: escaped, never raw markup
    assert "&lt;6-handed&gt;" in fb and "&quot;preflop&quot;" in fb
    assert "<6-handed>" not in fb
    assert "policy_net" in js["gto_html"] and "canonical obs" in js["gto_html"]
    assert "PPO fallback" not in js["gto_html"]
    # plain PPO build: nothing extra
    assert js["ppo_html"] == "model: <b>ppo_nlh.pt</b>"
