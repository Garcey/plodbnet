"""Review 2026-09-20 G3/G8/G14 — the games.js half, exercised in Node.

games.js is a plain browser script, so it is evaluated in a `vm` context with
a stub DOM + a scriptable `fetch`, and the real functions are driven from
there. Skipped when Node is not installed (the wiring is also pinned textually
in test_review_homegame_fixes.py::test_games_js_client_contract).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

GAMES_JS = (
    Path(__file__).resolve().parents[2]
    / "python" / "plo5bp" / "ui" / "static" / "games.js"
)

HARNESS = r"""
const fs = require("fs");
const vm = require("vm");

function el() {
  const e = {
    style: {}, dataset: {}, hidden: false, value: "", innerHTML: "", textContent: "",
    scrollTop: 0, scrollHeight: 0, clientHeight: 0, disabled: false, checked: false,
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    setAttribute() {}, getAttribute() { return null; }, appendChild(c) { return c; },
    addEventListener() {}, querySelectorAll() { return []; }, querySelector() { return null; },
    closest() { return null; }, insertAdjacentHTML() {}, contains() { return false; }, focus() {},
  };
  return e;
}
const els = {};
const calls = [];            // every fetch: {url, method, body}
const pending = [];          // unresolved fetches, in send order
function fetchStub(url, opts) {
  const rec = { url, method: (opts && opts.method) || "GET",
                body: opts && opts.body ? JSON.parse(opts.body) : null };
  calls.push(rec);
  return new Promise((resolve) => {
    pending.push({ rec, reply(status, payload) {
      resolve({ ok: status < 400, status, statusText: String(status),
                headers: { get: () => "application/json" },
                json: async () => payload, text: async () => JSON.stringify(payload) });
    } });
  });
}
const timers = [];
const ctx = {
  console, JSON, Math, Number, String, Object, Array, Set, Map, Promise, Error,
  performance: { now: () => 0 },
  document: { getElementById: (id) => (els[id] = els[id] || el()), createElementNS: () => el(),
              createElement: () => el(), activeElement: null, hidden: false, body: el() },
  location: { pathname: "/games" }, history: { replaceState() {} },
  confirm: () => true, fetch: fetchStub,
  setInterval: (fn, ms) => { timers.push(fn); return timers.length; }, clearInterval() {},
  setTimeout: (fn, ms) => { ctx.__timeouts.push(fn); return ctx.__timeouts.length; }, clearTimeout() {},
  __timeouts: [],
};
vm.createContext(ctx);
let src = fs.readFileSync(process.argv[2], "utf8").replace(/\r\n/g, "\n");
if (!/\ninit\(\);\s*$/.test(src)) throw new Error("games.js no longer ends with init();");
src = src.replace(/\ninit\(\);\s*$/, "\n");
src += "\n;globalThis.__api = { G, acceptState, render, act, deal, post, startPoll, maybePreAct, setPreAction };";
vm.runInContext(src, ctx, { filename: "games.js" });
const { G, acceptState, render, act, deal, startPoll } = ctx.__api;

function state(over) {
  const seats = [];
  for (let i = 0; i < 8; i++) seats.push({ seat: i, empty: i > 1, name: i > 1 ? null : "p" + i,
    user_id: i > 1 ? null : i + 1, is_hero: i === 0, in_hand: i < 2, folded: false,
    stack_cents: 4000, committed_this_street_cents: 0, hole: null, hand_desc: null });
  return Object.assign({
    id: "T1", epoch: "e1", rev: 1, hand_no: 1, action_seq: 0, name: "t", status: "open",
    phase: "in_hand", street: "flop", actor: 1, my_seat: 0, my_user_id: 1, hero_seat: 0,
    is_host: true, running: true, can_deal: false, can_rabbit: false, rabbit_shown: false,
    num_seats: 8, button_seat: 0, seats, history: [], ledger: [], chat: [],
    stakes: { sb_cents: 50, bb_cents: 100, ante_cents: 300, default_buyin_cents: 4000, bb_chips: 10000 },
    board: { a: { flop: [1, 2, 3], turn: null, river: null }, b: { flop: [4, 5, 6], turn: null, river: null } },
    legal: { fold: false, check_call: false, raise: false },
    raise_bounds: { min_chips: 0, max_chips: 0 }, to_call_chips: 0, to_call_cents: 0,
    street_commit_chips: 0, pot_cents: 600, pot_chips: 60000, hand_deltas_cents: [],
    auto_stack: { mode: "off", all_cents: 0 }, decision_secs: 30, turn_remaining_secs: 20,
    street_pause_secs: 1.5, eligible_count: 2,
    runout: { active: false, blocking: false, shown_len: 0, award_index: -1, award_step: null },
  }, over || {});
}
const tick = () => new Promise((r) => setImmediate(r));
const out = {};

(async () => {
  // --- acceptState: revision first, send order second ------------------------
  G.gameId = "T1";
  render(state({ rev: 5 }));
  G.appliedSeq = 10;
  out.older_rev = acceptState(state({ rev: 4 }), 99);
  out.same_rev_older_seq = acceptState(state({ rev: 5 }), 9);
  out.same_rev_newer_seq = acceptState(state({ rev: 5 }), 11);
  out.newer_rev = acceptState(state({ rev: 6 }), 3);
  out.new_epoch = acceptState(state({ rev: 0, epoch: "e2" }), 1);
  out.other_table = acceptState(state({ rev: 0, id: "T2" }), 1);

  // --- a slow poll must not roll the UI back behind the POST that answered me
  G.state = null; G.appliedSeq = 0; G.reqSeq = 0;
  render(state({ rev: 7, actor: 0, legal: { fold: true, check_call: true, raise: false }, to_call_cents: 100 }));
  startPoll();
  const pollTick = timers[timers.length - 1];
  pollTick();                                   // poll #1 goes out ...
  pollTick(); pollTick();                       // ... further ticks are skipped while it is in flight
  out.polls_in_flight = pending.length;
  const actP = act({ gate: "check_call" });     // my click, sent AFTER the poll
  out.act_body = calls[calls.length - 1].body;
  const [pollReq, actReq] = pending.splice(0, 2);
  actReq.reply(200, state({ rev: 8, action_seq: 1, actor: 1 }));
  await actP; await tick();
  pollReq.reply(200, state({ rev: 7, actor: 0 }));   // the stale poll lands LATE
  await tick(); await tick();
  out.after_stale_poll = { rev: G.state.rev, action_seq: G.state.action_seq };
  pollTick();
  out.poll_resumes = pending.length;
  pending.splice(0).forEach((p) => p.reply(200, state({ rev: 8, action_seq: 1, actor: 1 })));
  await tick(); await tick();

  // --- armed pre-action is cleared on hand / street / phase change ------------
  G.preAction = "fold";
  render(state({ rev: 9, action_seq: 2, actor: 1 }));               // same hand + street
  out.pre_kept_same_street = G.preAction;
  render(state({ rev: 10, action_seq: 3, actor: 1, street: "turn" }));
  out.pre_after_street_change = G.preAction;
  G.preAction = "fold";
  render(state({ rev: 11, phase: "showdown", actor: null }));
  out.pre_after_phase_change = G.preAction;
  G.preAction = "fold";
  const before = calls.length;
  render(state({ rev: 12, hand_no: 2, action_seq: 0, actor: 0,
                 legal: { fold: true, check_call: true, raise: false }, to_call_cents: 300 }));
  await tick();
  out.pre_after_new_hand = G.preAction;
  out.auto_fold_fired_next_hand = calls.slice(before).some((c) => c.url.endsWith("/act"));
  pending.splice(0);

  // --- a 409 is not an error banner, it just refreshes ---------------------------
  const p409 = act({ gate: "fold" });
  pending.shift().reply(409, { detail: "stale: the action has moved on" });
  await p409; await tick();
  out.err_hidden_after_409 = els["err"].hidden;
  out.refresh_after_409 = pending.length === 1 && pending[0].rec.method === "GET";
  pending.splice(0).forEach((p) => p.reply(200, state({ rev: 13, hand_no: 2, action_seq: 1 })));
  await tick(); await tick();

  // --- 2026-09-21: the SERVER deals; the client never schedules a deal itself ---
  ctx.__timeouts.length = 0;
  render(state({ rev: 14, phase: "waiting", actor: null, can_deal: true, hand_no: 2 }));
  out.client_auto_deal_timers = ctx.__timeouts.length;
  deal(G.state);
  out.deal_call = calls[calls.length - 1];
  pending.splice(0);

  // --- extended pre-actions: an armed price / free check must still hold ----------
  const base = { rev: 20, hand_no: 3, action_seq: 0, actor: 1 };
  const withBet = (mine, theirs) => {
    const st = state(base);
    st.seats[0].committed_this_street_cents = mine;
    st.seats[1].committed_this_street_cents = theirs;
    return st;
  };
  render(withBet(0, 500));
  ctx.__api.setPreAction("call");
  out.call_armed_at = G.preCallCents;
  render(Object.assign(withBet(0, 500), { rev: 21 }));
  out.call_kept_same_price = G.preAction;
  render(Object.assign(withBet(0, 1500), { rev: 22, action_seq: 1 }));
  out.call_after_raise = G.preAction;                 // the price moved -> disarmed
  render(Object.assign(withBet(0, 0), { rev: 23, action_seq: 2 }));
  ctx.__api.setPreAction("check");
  render(Object.assign(withBet(0, 700), { rev: 24, action_seq: 3 }));
  out.check_after_bet = G.preAction;                  // a bet appeared -> disarmed
  ctx.__api.setPreAction("call_any");
  const n0 = calls.length;
  render(Object.assign(withBet(0, 700), { rev: 25, action_seq: 4, actor: 0,
    legal: { fold: true, check_call: true, raise: false }, to_call_cents: 700 }));
  await tick();
  out.call_any_fired = calls.slice(n0).filter((c) => c.url.endsWith("/act")).map((c) => c.body);
  pending.splice(0);
  console.log("RESULT " + JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.fixture(scope="module")
def result():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    proc = subprocess.run(
        [node, "-", str(GAMES_JS)], input=HARNESS, capture_output=True,
        text=True, encoding="utf-8", timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(x for x in proc.stdout.splitlines() if x.startswith("RESULT "))
    return json.loads(line[len("RESULT "):])


def test_out_of_order_responses_are_dropped(result):
    assert result["older_rev"] is False
    assert result["same_rev_older_seq"] is False
    assert result["same_rev_newer_seq"] is True
    assert result["newer_rev"] is True
    assert result["new_epoch"] is True  # a reloaded table restarts `rev`
    assert result["other_table"] is True


def test_polls_are_single_flight_and_cannot_roll_back_a_post(result):
    assert result["polls_in_flight"] == 1  # three ticks, one request
    assert result["after_stale_poll"] == {"rev": 8, "action_seq": 1}
    assert result["poll_resumes"] == 1


def test_act_and_deal_name_the_decision(result):
    assert result["act_body"] == {"gate": "check_call", "hand_no": 1, "action_seq": 0}
    assert result["deal_call"]["url"].endswith("/games/api/tables/T1/deal")
    assert result["deal_call"]["body"] == {"hand_no": 2}


def test_armed_pre_action_does_not_survive_the_street_or_hand(result):
    assert result["pre_kept_same_street"] == "fold"
    assert result["pre_after_street_change"] is None
    assert result["pre_after_phase_change"] is None
    assert result["pre_after_new_hand"] is None
    assert result["auto_fold_fired_next_hand"] is False  # the reported bug


def test_409_refreshes_quietly(result):
    assert result["err_hidden_after_409"] is True
    assert result["refresh_after_409"] is True


def test_the_client_never_deals_on_its_own(result):
    # Dealing moved to the server (homegame._auto_deal_tick_locked).
    assert result["client_auto_deal_timers"] == 0


def test_extended_pre_actions_follow_the_price(result):
    assert result["call_armed_at"] == 500
    assert result["call_kept_same_price"] == "call"
    assert result["call_after_raise"] is None
    assert result["check_after_bet"] is None
    assert result["call_any_fired"] == [{"gate": "check_call", "hand_no": 3, "action_seq": 4}]
