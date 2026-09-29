"""Home-games client features of the 2026-09-28 second pass, driven in Node with the real
modules on the mini DOM (hg_mini_dom.js):

- FEAT-011: three kinds of sound, each with its own switch (your turn / chat / table) under
  the master mute; the turn notification brings the table forward when tapped and goes
  away with the decision; a phone vibrates on your turn (Preferences can turn it off);
- FEAT-009: "How a hand works" opens by itself the first time someone who isn't seated
  looks at a table of a game (once per browser and game), and from the welcome banner
  and the buy-in dialog;
- SEC-008: every club member may browse a table's hands — the History tab and "Last
  hand" follow the server's `can_browse_hands`, not "has sat here".
Skipped when Node is not installed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hg_client_tools import MINI_DOM, STATIC, UI_BOOT, UI_FILES, node_exe, run_node  # noqa: E402


@pytest.fixture(scope="module")
def node():
    if node_exe() is None:
        pytest.skip("node is not installed")
    return node_exe()


def _ui(tmp_path, body, *extra):
    return run_node(UI_BOOT + body, STATIC, json.dumps(UI_FILES), *extra, tmp=tmp_path)


# ------------------------------------------------------------------ FEAT-011
def test_each_kind_of_sound_has_its_own_switch_under_the_master(node, tmp_path):
    got = run_node(MINI_DOM + r"""
const fs = require("fs"), vm = require("vm");
const played = [];
class Osc { constructor() { this.frequency = { setValueAtTime() {}, exponentialRampToValueAtTime() {} }; } connect(x) { return x; } start() { played.push(1); } stop() {} }
class Gain { constructor() { this.gain = { value: 1, setValueAtTime() {}, exponentialRampToValueAtTime() {} }; } connect(x) { return x; } }
class AC { constructor() { this.state = "running"; this.currentTime = 0; this.sampleRate = 100; this.destination = {}; }
  createGain() { return new Gain(); } createOscillator() { return new Osc(); }
  createBuffer() { return { getChannelData: () => new Float32Array(60) }; }
  createBufferSource() { const s = new Osc(); s.buffer = null; return s; }
  createBiquadFilter() { return { frequency: {}, Q: {}, connect: (x) => x }; } }
const ctx = { console, AudioContext: AC, Math, Float32Array };
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx);
const S = ctx.HG.sound;
const hears = (name) => { played.length = 0; S.play(name); return played.length > 0; };
const out = { kinds: ["turn", "tick", "urgent", "ask", "msg", "chip", "win", "deal"].map(S.kindOf) };
out.all = ["turn", "msg", "chip"].map(hears);
S.setKinds({ chat: false, table: false });
out.turnOnly = ["turn", "tick", "ask", "msg", "chip", "win"].map(hears);
S.setKinds({ turn: false, chat: true });
out.chatOnly = ["turn", "msg", "chip"].map(hears);
S.setKinds({ turn: true, chat: true, table: true }); S.setEnabled(false);
out.muted = ["turn", "msg", "chip"].map(hears);
console.log(JSON.stringify(out));
""", STATIC / "games.sound.js", tmp=tmp_path)
    assert got["kinds"] == ["turn", "turn", "turn", "turn", "chat", "table", "table", "table"]
    assert got["all"] == [True, True, True]
    assert got["turnOnly"] == [True, True, True, False, False, False]  # only the turn chime (+ its clock, requests)
    assert got["chatOnly"] == [False, True, False]
    assert got["muted"] == [False, False, False]  # (the top bar's speaker still silences everything)


def test_turn_alerts_notify_vibrate_and_the_prefs_carry_the_switches(node, tmp_path):
    got = _ui(tmp_path, r"""
(async () => {
  const B = boot(() => ({}));
  const notes = [], buzz = [], kinds = [];
  let focused = 0;
  B.ctx.focus = () => { focused++; };
  B.ctx.Notification = class { constructor(t, o) { this.t = t; this.o = o; this.closed = false; notes.push(this); } close() { this.closed = true; } };
  B.ctx.Notification.permission = "granted";
  B.ctx.navigator = { vibrate: (p) => { buzz.push(p); return true; } };
  B.ctx.matchMedia = (q) => ({ matches: /pointer: coarse/.test(q) });
  B.HG.sound = { play() {}, setEnabled() {}, setVolume() {}, setKinds: (k) => kinds.push(k) };
  B.W.doc.hidden = true;
  const G = B.HG.core.G;
  G.prefs.notify = true; G.gameId = "T1";
  const turn = (seq) => table({ phase: "in_hand", hand_no: 9, action_seq: seq, actor: 0, legal: { fold: true, check_call: true, raise: true }, to_call_cents: 200 });
  const out = {};
  B.ctx.turnCue(turn(3));  // (games.js: runs on every rendered state)
  out.first = notes.map((n) => [n.t, n.o.body, n.o.tag]);
  out.buzz = buzz.length;
  notes[0].onclick();
  out.click = [focused, notes[0].closed];
  B.ctx.turnCue(turn(3));  // (the same decision again: no second alert)
  B.ctx.turnCue(turn(5));
  out.second = notes.length;
  B.ctx.turnCue(table({ phase: "in_hand", hand_no: 9, action_seq: 6, actor: 1 }));  // (someone else's turn now)
  out.goneWithTheDecision = notes[1].closed;
  G.prefs.vibrate = false; buzz.length = 0;
  B.ctx.turnCue(turn(8));
  out.noBuzz = buzz.length;
  // Preferences: the three kinds, the phone's vibration, the notification
  B.HG.ui.openPrefs();
  const m = B.W.doc.querySelector(".modal");
  out.switches = ["p-snd-turn", "p-snd-chat", "p-snd-table", "p-vibrate", "p-notify"].map((id) => !!m.querySelector("#" + id));
  const chat = m.querySelector("#p-snd-chat");
  chat.checked = false; chat.dispatch("change");
  out.saved = [G.prefs.sndChat, JSON.stringify(kinds[kinds.length - 1])];
  console.log(JSON.stringify(out));
})();
""")
    assert got["first"] == [["Your turn", "Friday", "hg-turn"]]
    assert got["buzz"] == 1
    assert got["click"] == [1, True]  # tapping it brings the table forward
    assert got["second"] == 2
    assert got["goneWithTheDecision"] is True
    assert got["noBuzz"] == 0
    assert got["switches"] == [True, True, True, True, True]
    assert got["saved"] == [False, json.dumps({"turn": True, "chat": False, "table": True}, separators=(",", ":"))]


# ------------------------------------------------------------------ FEAT-009
def test_the_guide_opens_once_for_a_first_look_and_from_the_links(node, tmp_path):
    got = _ui(tmp_path, r"""
(async () => {
  const B = boot(() => ({}));
  const out = {};
  const watching = table({ my_seat: null, is_host: false, variant: "plo67", game: { code: "plo67", label: "PLO67", name: "PLO67 double-board bomb pot", hole: 7, dealt: 4, burns: 3, max_seats: 5, graded: false } });
  B.HG.core.G.state = watching; B.HG.core.G.gameId = "T1";
  B.HG.ui.render(watching, null);
  B.W.advance(1000);
  const m = () => B.W.doc.querySelector("#modal-root .modal");
  out.auto = m() ? [m().querySelector("h3").textContent, m().querySelectorAll("ol.guide li").length] : null;
  out.burns = m() ? /face up/.test(m().textContent) && /4\+ cards/.test(m().textContent) : null;
  out.text = m() ? m().textContent.slice(0, 400) : null;
  out.remembered = JSON.parse(B.store["hg.guide.v1"] || "{}");
  B.HG.ui.closeTop(); B.W.advance(600);
  B.HG.ui.render(watching, null); B.W.advance(1000);  // (a second look: not again)
  out.second = !!m();
  const plo5 = table({ my_seat: null, is_host: false, id: "T2" });
  B.HG.core.G.state = plo5; B.HG.ui.render(plo5, watching); B.W.advance(1000);
  out.otherGame = !!m();  // (another game's first look: its own guide)
  B.HG.ui.closeTop(); B.W.advance(600);
  const seated = table({ id: "T3", my_seat: 0, variant: "plo6" });
  B.HG.core.G.state = seated; B.HG.ui.render(seated, plo5); B.W.advance(1000);
  out.seated = !!m();  // (someone playing is never interrupted)
  // the buy-in dialog's link
  B.HG.core.G.state = table({ my_seat: null, is_host: false });
  B.HG.ui.openSit(3);
  const link = B.W.doc.querySelector(".md-guide");
  out.sitLink = !!link;
  link.click();
  out.fromLink = B.W.doc.querySelectorAll("#modal-root .modal").length;
  console.log(JSON.stringify(out));
})();
""")
    assert got["auto"] == ["How a hand works", 4]
    assert got["burns"] is True, got["text"]  # (PLO67: the face-up burns are part of it)
    assert got["remembered"] == {"plo67": 1}
    assert got["second"] is False
    assert got["otherGame"] is True
    assert got["seated"] is False
    assert got["sitLink"] is True and got["fromLink"] == 2  # (on top of the buy-in dialog)


# ------------------------------------------------------------------ SEC-008
def test_every_club_member_may_browse_the_tables_hands(node, tmp_path):
    got = _ui(tmp_path, r"""
(async () => {
  const hands = [];
  const B = boot((url) => { if (/\/hands\?/.test(url)) { hands.push(url); return { hands: [{ hand_no: 3, board_a: [1, 2, 3], board_b: [4, 5, 6], winners: [{ name: "Dana" }], pot_cents: 900 }], stats: [], h2h: [] }; } return {}; });
  const out = {};
  // a clubmate who never sat here: the server says they may browse (SEC-008)
  const s = table({ my_seat: null, is_member: false, can_browse_hands: true, last_hand_no: 3 });
  B.HG.core.G.state = s; B.HG.core.G.gameId = "T1"; B.HG.core.G.prefs.rail = true;
  B.HG.ui.render(s, null);
  B.HG.ui.setRail(true, "hands");
  await flush(); B.W.advance(50); await flush();
  out.fetched = hands.length > 0;
  out.rows = B.W.doc.querySelectorAll("#hands-body .hand-row").length;
  // an older server (no flag) keeps the old rule
  const old = table({ my_seat: null, is_member: false, last_hand_no: 3 });
  delete old.can_browse_hands;
  B.HG.uiState.hands = null; B.HG.uiState.handsFor = null;
  B.HG.core.G.state = old;
  B.HG.ui.render(old, s);
  B.HG.ui.setRail(true, "hands");
  out.oldServer = B.$("hands-body").textContent;
  console.log(JSON.stringify(out));
})();
""")
    assert got["fetched"] is True and got["rows"] == 1
    assert "club" in got["oldServer"]  # (no flag, not a member: the note instead of a list)


# ------------------------------------------------------------------ FEAT-013
def test_a_seated_player_moves_to_an_empty_seat_now_or_after_the_hand(node, tmp_path):
    got = _ui(tmp_path, r"""
(async () => {
  const B = boot((url, body) => url.endsWith("/move") ? table({ my_seat: body.seat }) : {});
  const out = {};
  const G = B.HG.core.G;
  G.gameId = "T1";
  const ok = () => { const m = B.W.doc.querySelector("#modal-root .modal"); [...m.querySelectorAll(".m-foot button")].pop().click(); };
  // between hands: straight there
  G.state = table();
  B.HG.ui.openSit(3);
  let m = B.W.doc.querySelector("#modal-root .modal");
  out.ask = [m.querySelector("h3").textContent, [...m.querySelectorAll(".m-foot button")].map((b) => b.textContent)];
  ok(); await flush();
  out.now = B.posts.map((p) => [p.url.split("/").pop(), p.body]);
  B.W.advance(600); B.posts.length = 0;
  // holding cards: after the hand
  const inHand = table({ phase: "in_hand", hand_no: 5 });
  inHand.seats[0].in_hand = true;
  G.state = inHand;
  B.HG.ui.openSit(4);
  m = B.W.doc.querySelector("#modal-root .modal");
  out.later = [...m.querySelectorAll(".m-foot button")].pop().textContent;
  ok(); await flush();
  out.queued = [B.posts.length, JSON.stringify(B.HG.uiState.moveAfter)];
  B.HG.ui.render(inHand, inHand); await flush();
  out.stillHolding = B.posts.length;
  const over = table({ phase: "showdown", hand_no: 5, last_hand_no: 5 });
  B.HG.ui.render(over, inHand); await flush();
  out.after = B.posts.map((p) => [p.url.split("/").pop(), p.body]);
  out.cleared = B.HG.uiState.moveAfter;
  console.log(JSON.stringify(out));
})();
""")
    assert got["ask"] == ["Move to seat 4?", ["Cancel", "Move here"]]
    assert got["now"] == [["move", {"seat": 3}]]
    assert got["later"] == "Move after this hand"
    assert got["queued"] == [0, json.dumps({"table": "T1", "seat": 4}, separators=(",", ":"))]
    assert got["stillHolding"] == 0
    assert got["after"] == [["move", {"seat": 4}]]
    assert got["cleared"] is None


# ------------------------------------------------------------------ FEAT-012
def test_my_hands_draws_the_running_net_between_the_numbers_and_the_list(node, tmp_path):
    got = _ui(tmp_path, r"""
(async () => {
  let series = { points: [[0, 0], [1, 300], [2, -200], [3, 450]], hands: 3, net_cents: 450, best_cents: 450, worst_cents: -200, breaks: [2] };
  const asked = [];
  const B = boot((url) => {
    if (url.includes("/series")) { asked.push(url); return series; }
    if (url.includes("/stats")) return { hands: 3, net_cents: 450, wins: 2, accuracy: null, graded: 0, games: [], sessions: [{ id: "g1", name: "Friday", hands: 3, net_cents: 450 }], versus: [] };
    if (url.includes("/hands")) return { hands: [], total: 0, limit: 40 };
    return {};
  });
  const out = {};
  B.HG.core.G.clubId = "c1";
  B.HG.ui.openMyHands("");
  for (let i = 0; i < 6; i++) { await flush(); B.W.advance(20); }
  const g = B.W.doc.querySelector(".pgraph");
  out.graph = !!g;
  out.caption = g && [...g.querySelectorAll("figcaption > span")].map((x) => x.textContent.replace(/\s+/g, " "));
  out.lines = g ? g.querySelectorAll(".pg-line").length : 0;
  out.breaks = g ? g.querySelectorAll(".pg-break").length : 0;
  out.order = [...B.W.doc.querySelector(".db").children].map((c) => c.getAttribute("id") || c.className).slice(0, 4);
  out.asked = asked.map((u) => u.split("?")[0]);
  // a session picked from the list: that table's line
  const sel = B.W.doc.querySelector("#db-game");
  sel.value = "g1"; sel.dispatch("change");
  for (let i = 0; i < 6; i++) { await flush(); B.W.advance(20); }
  out.session = asked[asked.length - 1].includes("game=g1");
  // too few hands for a line: no graph at all
  series = { points: [[0, 0], [1, 100]], hands: 1, net_cents: 100, best_cents: 100, worst_cents: 0, breaks: [] };
  sel.value = ""; sel.dispatch("change");
  for (let i = 0; i < 6; i++) { await flush(); B.W.advance(20); }
  out.tiny = !!B.W.doc.querySelector(".pgraph");
  console.log(JSON.stringify(out));
})();
""")
    assert got["graph"] is True
    assert got["caption"] == ["Running net", "+$4.50 high · −$2.00 low · 3 hands"]
    assert got["lines"] == 2 and got["breaks"] == 1  # (green over zero, red under; one table change)
    assert got["order"] == ["db-head", "db-graph", "db-vs", "db-bar"]
    assert got["asked"] == ["/games/api/my/series"]
    assert got["session"] is True
    assert got["tiny"] is False
