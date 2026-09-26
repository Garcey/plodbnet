"use strict";
// Home games — client core: state, networking, polling, pre-actions, routing.
// Rendering lives in games.table.js (the felt) and games.ui.js (everything
// around it); sounds in games.sound.js. This file has no DOM building in it,
// so it can be driven headless (tests/python/test_review_homegame_client_js.py).
//
// The SERVER deals the next hand (table setting "next hand after"); this client
// never auto-deals — a host who switched tabs used to stall the whole table.
// State arrives by live push (SSE, `startLive`); `startPoll` is the fallback.

const HG = (globalThis.HG = globalThis.HG || {});
const POLL_MS = 450;
const LOBBY_POLL_MS = 5000;
const PREFS_KEY = "hg.prefs.v1";
const PREF_DEFAULTS = {
  sound: true, volume: 0.6, anim: "full", deck: "4c", cards: "bold", back: "blue",
  felt: "emerald", unit: "dollars", hotkeys: true, confirmAllIn: false, bubbles: true,
  notify: false, rail: true, railTab: "chat",
};

const G = {
  me: null,
  state: null,
  gameId: null,
  poll: null,
  live: null, // the EventSource while the table is pushed (null = polling fallback)
  lobbyPoll: null,
  pollTicks: 0,
  pollBusy: false, // a poll request is in flight — skip ticks until it lands
  reqSeq: 0, // request counter (send order)
  appliedSeq: 0, // send-order number of the newest response applied
  preAction: null, // null | check_fold | fold | check | call | call_any
  preCallCents: null, // the price an armed "call" was armed at
  preActed: false,
  raiseTouched: false,
  turnKey: null, // decision I was last alerted about
  conn: "ok", // ok | slow | off
  fails: 0,
  lastLatency: 0,
  prefs: Object.assign({}, PREF_DEFAULTS),
  baseTitle: "Home games",
};

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------- money
function dollars(cents) {
  const n = Number(cents || 0) / 100;
  const sign = n < 0 ? "-" : "";
  return `${sign}$${Math.abs(n).toFixed(2)}`;
}
function fmtAmt(cents, s) {
  if (G.prefs.unit === "bb" && s && s.stakes) {
    const bb = Number(cents || 0) / (s.stakes.bb_cents || 100);
    const t = Math.abs(bb);
    const body = Math.abs(t - Math.round(t)) < 0.05 ? String(Math.round(t)) : t.toFixed(1);
    return `${bb < 0 ? "-" : ""}${body} bb`;
  }
  return dollars(cents);
}
function toCents(v) {
  const n = Number(String(v == null ? "" : v).replace(/[$,\s]/g, ""));
  if (!Number.isFinite(n)) return null;
  return Math.round(n * 100);
}
function chipsToCents(chips, s) {
  return Math.round((Number(chips || 0) * s.stakes.bb_cents) / (s.stakes.bb_chips || 10000));
}
function centsToChips(cents, s) {
  return Math.round((Number(cents || 0) * (s.stakes.bb_chips || 10000)) / (s.stakes.bb_cents || 100));
}
function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function showErr(msg) {
  const el = $("err");
  if (!msg) { el.hidden = true; el.textContent = ""; return; }
  if (HG.ui && HG.ui.toast) { HG.ui.toast(msg, "err"); return; }
  el.hidden = false;
  el.textContent = msg;
}

async function j(url, opts) {
  const res = await fetch(url, { credentials: "same-origin", ...opts });
  const ct = res.headers.get("content-type") || "";
  const body = ct.includes("json") ? await res.json() : { detail: await res.text() };
  if (!res.ok) {
    const d = body.detail || body.error || res.statusText;
    // (a structured answer — e.g. a table of a club you are not in — keeps its fields)
    const err = new Error(typeof d === "string" ? d : (d && d.message) || JSON.stringify(d));
    err.status = res.status;
    err.detail = d;
    throw err;
  }
  return body;
}

// Every table-state response goes through here before it is rendered.
// A response is DROPPED when something newer is already on screen: by the
// server's state revision (`rev`, comparable within one `epoch` = one
// in-memory load of the table), and for equal revisions by send order.
// Without this a slow poll could land after the POST that answered my click
// and roll the UI back a decision — the stale buttons then acted on the
// wrong node (review 2026-09-20 G8).
function acceptState(s, seq) {
  const cur = G.state;
  if (cur && s && cur.id === s.id && cur.epoch === s.epoch) {
    if (s.rev < cur.rev) return false;
    if (s.rev === cur.rev && seq < G.appliedSeq) return false;
  }
  if (seq > G.appliedSeq) G.appliedSeq = seq;
  return true;
}

// -------------------------------------------------------------- turn logic
function myTurn(s) {
  return !!(s && s.legal && (s.legal.fold || s.legal.check_call || s.legal.raise));
}
function inHandAlive(s) {
  if (!s || s.phase !== "in_hand" || !Number.isInteger(s.my_seat)) return false;
  const me = s.seats[s.my_seat];
  return !!(me && !me.empty && !me.folded && me.in_hand && !me.all_in);
}
// What I would have to put in to continue, in cents — also when it is not my
// turn yet (the payload's to_call_* only describe the actor's own decision).
function heroToCallCents(s) {
  if (!s || !Number.isInteger(s.my_seat)) return 0;
  if (myTurn(s)) return s.to_call_cents || 0;
  const me = s.seats[s.my_seat];
  if (!me) return 0;
  let top = 0;
  for (const x of s.seats) if (!x.empty && x.in_hand) top = Math.max(top, x.committed_this_street_cents || 0);
  return Math.max(0, Math.min(top - (me.committed_this_street_cents || 0), me.stack_cents || 0));
}
function potBetTo(s, frac) {
  const ac = s.street_commit_chips || 0;
  const toCall = s.to_call_chips || 0;
  const extra = Math.round(frac * ((s.pot_chips || 0) + toCall));
  return ac + toCall + extra;
}
function raiseBoundsTo(s) {
  const ac = s.street_commit_chips || 0;
  const rb = s.raise_bounds || {};
  return { min: (rb.min_chips || 0) + ac, max: (rb.max_chips || 0) + ac };
}
function clampRaiseTo(s, chips) {
  const b = raiseBoundsTo(s);
  if (b.max <= 0) return chips;
  return Math.max(b.min, Math.min(b.max, chips));
}

function setPreAction(kind, s) {
  const st = s || G.state;
  if (!kind || G.preAction === kind) { G.preAction = null; G.preCallCents = null; }
  else { G.preAction = kind; G.preCallCents = kind === "call" ? heroToCallCents(st) : null; }
  if (HG.ui && HG.ui.renderDock && st) HG.ui.renderDock(st);
}

function maybePreAct(s) {
  const turn = myTurn(s);
  if (!turn) {
    G.preActed = false;
    return;
  }
  if (G.preActed || !G.preAction) return;
  const want = G.preAction;
  const armedAt = G.preCallCents;
  G.preActed = true;
  G.preAction = null;
  G.preCallCents = null;
  const free = !(s.to_call_cents > 0);
  if (want === "check_fold") {
    if (s.legal.check_call && free) act({ gate: "check_call" });
    else if (s.legal.fold) act({ gate: "fold" });
    else if (s.legal.check_call) act({ gate: "check_call" });
  } else if (want === "fold") {
    if (s.legal.fold) act({ gate: "fold" });
    else if (free && s.legal.check_call) act({ gate: "check_call" });
  } else if (want === "check") {
    if (free && s.legal.check_call) act({ gate: "check_call" });
  } else if (want === "call") {
    if (!free && s.legal.check_call && s.to_call_cents === armedAt) act({ gate: "check_call" });
  } else if (want === "call_any") {
    if (s.legal.check_call) act({ gate: "check_call" });
  }
}

// ------------------------------------------------------------------ render
function render(s) {
  const prev = G.state;
  if (prev && (prev.id !== s.id || prev.hand_no !== s.hand_no || prev.street !== s.street || prev.phase !== s.phase)) {
    // An armed pre-action belongs to ONE street of ONE hand: it used to
    // survive the hand, so an armed "fold" silently folded your first decision
    // of the next one (review 2026-09-20 G8).
    G.preCallCents = null;
    G.preAction = null;
    G.preActed = false;
    G.raiseTouched = false;
  }
  G.state = s;
  // the verifiable shuffle: this device takes part in cutting the next deck and
  // checks every card it is shown (games.fair.js)
  if (HG.fair && HG.fair.onState) HG.fair.onState(s);
  if (G.preAction && !myTurn(s)) {
    // The price moved under an armed "check" / "call $x": it no longer means
    // what the player agreed to.
    const owe = heroToCallCents(s);
    if ((G.preAction === "check" && owe > 0) || (G.preAction === "call" && owe !== G.preCallCents) || !inHandAlive(s)) {
      G.preAction = null;
      G.preCallCents = null;
    }
  }
  if (HG.ui && HG.ui.render) HG.ui.render(s, prev);
  maybePreAct(s);
  turnCue(s);
}

function turnCue(s) {
  const mine = myTurn(s);
  const key = mine ? `${s.id}:${s.hand_no}:${s.action_seq}` : null;
  if (typeof document !== "undefined" && "title" in document) {
    document.title = mine ? `▶ Your turn · ${s.name}` : `${s.name} · ${G.baseTitle}`;
  }
  if (key && key !== G.turnKey) {
    if (HG.sound) HG.sound.play("turn");
    if (HG.ui && HG.ui.onMyTurn) HG.ui.onMyTurn(s);
    if (G.prefs.notify && document.hidden && globalThis.Notification && Notification.permission === "granted") {
      try { new Notification("Your turn", { body: s.name, tag: "hg-turn" }); } catch (_) { /* optional */ }
    }
  }
  G.turnKey = key;
}

// ----------------------------------------------------------------- network
async function post(url, body) {
  showErr("");
  const seq = ++G.reqSeq;
  try {
    const s = await j(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    if (acceptState(s, seq)) render(s);
    return s;
  } catch (e) {
    if (e.status === 409) {
      // The table moved on while this request was in flight (someone acted,
      // the clock acted for me, the next hand was dealt). Nothing was done —
      // just catch up; no error banner.
      refreshNow();
      throw e;
    }
    showErr(e.message);
    if (HG.sound) HG.sound.play("error");
    throw e;
  }
}

// Every action names the decision it was composed for: the server answers
// 409 instead of applying "Call $1" to a different bet (review 2026-09-20 G8).
function act(body) {
  const s = G.state;
  if (!s) return Promise.resolve(null);
  return post(`/games/api/tables/${G.gameId}/act`, {
    ...body, hand_no: s.hand_no, action_seq: s.action_seq,
  }).catch(() => null);
}

// `s` = the finished hand this client saw; 409 if someone already dealt.
function deal(s) {
  return post(`/games/api/tables/${s.id}/deal`, { hand_no: s.hand_no }).catch(() => null);
}

function tablePost(what, body) {
  return post(`/games/api/tables/${G.gameId}/${what}`, body || {});
}

async function refreshNow() {
  if (!G.gameId) return;
  const seq = ++G.reqSeq;
  try {
    const s = await j(`/games/api/tables/${G.gameId}`);
    if (acceptState(s, seq)) render(s);
  } catch (_) { /* the poll will retry */ }
}

function setConn(state) {
  if (G.conn === state) return;
  G.conn = state;
  if (HG.ui && HG.ui.renderConn) HG.ui.renderConn();
}

function startPoll() {
  stopPoll();
  G.pollTicks = 0;
  G.poll = setInterval(async () => {
    if (!G.gameId) return;
    if (G.pollBusy) return; // one poll in flight
    // A hidden tab still polls (slowly): the turn alert has to reach it.
    G.pollTicks++;
    if (document.hidden && G.pollTicks % 4 !== 0) return;
    G.pollBusy = true;
    const seq = ++G.reqSeq;
    const t0 = performance.now();
    try {
      const s = await j(`/games/api/tables/${G.gameId}`);
      G.fails = 0;
      G.lastLatency = performance.now() - t0;
      setConn(G.lastLatency > 1200 ? "slow" : "ok");
      if (acceptState(s, seq)) render(s);
      // Polling is the fallback, not the way of life: once the server answers again
      // (after a restart the stream had given up), try the push again now and then.
      if (globalThis.EventSource && G.streamRetryAt && performance.now() >= G.streamRetryAt) {
        G.streamRetryMs = Math.min((G.streamRetryMs || 15000) * 2, 300000);
        G.streamRetryAt = performance.now() + G.streamRetryMs;
        startLive();
      }
    } catch (e) {
      if (e.status === 404) {
        stopPoll();
        showErr("That table no longer exists.");
        showLobby(true);
      } else if (e.status === 403 && e.detail && e.detail.error === "club") {
        // (no longer in the table's club — removed, or left in another tab): back to
        // the lobby, saying so — never a "connection lost" bar that cannot come back
        stopPoll();
        const gid = G.gameId;
        showErr(`You're no longer a member of ${e.detail.club.name}.`);
        showLobby(true);
        if (HG.ui && HG.ui.openClubGate) HG.ui.openClubGate(e.detail, gid);
      } else if (++G.fails >= 3) setConn("off");
    } finally {
      G.pollBusy = false;
    }
  }, POLL_MS);
}

function stopPoll() {
  if (G.poll) { clearInterval(G.poll); G.poll = null; }
  G.pollBusy = false;
}

// ------------------------------------------------------------- live push
// The table is PUSHED over Server-Sent Events (`…/stream`): the server sends
// this viewer's state the moment it changes, so an opponent's action shows up
// immediately instead of on the next 450 ms poll, and an idle table costs
// almost nothing. EventSource reconnects by itself; if the stream cannot be
// kept open (old proxy, corporate filter) the client falls back to polling.
function startLive() {
  stopLive();
  if (!globalThis.EventSource) { startPoll(); return; }
  const id = G.gameId;
  let errors = 0;
  const es = new EventSource(`/games/api/tables/${id}/stream`);
  G.live = es;
  es.onopen = () => { errors = 0; setConn("ok"); G.streamRetryAt = null; G.streamRetryMs = null; };
  es.onmessage = (ev) => {
    if (G.live !== es || G.gameId !== id) return;
    let s = null;
    try { s = JSON.parse(ev.data); } catch (_) { return; }
    errors = 0;
    setConn("ok");
    if (acceptState(s, ++G.reqSeq)) render(s);
  };
  es.onerror = async () => {
    if (G.live !== es) return;
    errors++;
    setConn("off");
    if (errors === 1) {
      // Is the table gone, or just the connection?
      try { await j(`/games/api/tables/${id}`); }
      catch (e) {
        if (e.status === 404 && G.live === es) { stopLive(); showErr("That table no longer exists."); showLobby(true); return; }
      }
    }
    if (errors >= 4 || es.readyState === 2) {
      es.close();
      if (G.live === es) {
        G.live = null;
        if (!G.streamRetryAt) { G.streamRetryMs = 30000; G.streamRetryAt = performance.now() + 30000; }
        startPoll();
      }
    }
  };
}

function stopLive() {
  if (G.live) { try { G.live.close(); } catch (_) { /* already closed */ } G.live = null; }
  stopPoll();
}

// ----------------------------------------------------------------- routing
async function openTable(id, push) {
  const seq = ++G.reqSeq;
  const s = await j(`/games/api/tables/${id}`);
  G.gameId = s.id;
  G.state = null;
  G.preAction = null;
  G.preActed = false;
  if (G.lobbyPoll) { clearInterval(G.lobbyPoll); G.lobbyPoll = null; }
  if (push && location.pathname !== `/games/t/${s.id}`) history.pushState(null, "", `/games/t/${s.id}`);
  if (HG.ui) HG.ui.showTable();
  if (acceptState(s, seq)) render(s);
  startLive();
  return s;
}

// ------------------------------------------------------------------- clubs
// The lobby shows ONE club at a time: its tables, its players, its numbers.
// Which one is remembered per browser (a convenience — the server decides who
// may see what).
const CLUB_KEY = "hg.club.v1";
function savedClub() {
  try { return (globalThis.localStorage && localStorage.getItem(CLUB_KEY)) || null; } catch (_) { return null; }
}
function setClub(id) {
  G.clubId = id || null;
  try { if (globalThis.localStorage) { if (id) localStorage.setItem(CLUB_KEY, id); else localStorage.removeItem(CLUB_KEY); } } catch (_) { /* private mode */ }
}

async function loadLobby() {
  if (G.clubId === undefined) G.clubId = savedClub();
  let data = await j("/games/api/tables" + (G.clubId ? `?club=${encodeURIComponent(G.clubId)}` : "")).catch((e) => {
    if (e.status === 404 && G.clubId) return null;  // (left or removed from that club)
    throw e;
  });
  const clubs = (data && data.clubs) || [];
  if (!data || !G.clubId || !clubs.some((c) => c.id === G.clubId)) {
    const pick = clubs.length ? clubs[0].id : null;
    if (!data || pick !== G.clubId) {
      setClub(pick);
      data = await j("/games/api/tables" + (pick ? `?club=${encodeURIComponent(pick)}` : ""));
    }
  }
  if (HG.ui) HG.ui.renderLobby(data);
  return data;
}

function showLobby(replace) {
  stopLive();
  G.gameId = null;
  G.state = null;
  G.turnKey = null;
  G.fails = 0;
  setConn("ok");  // (the "connection lost" bar is about a table: the lobby has its own poll)
  if (typeof document !== "undefined") document.title = G.baseTitle;
  if (location.pathname !== "/games") {
    if (replace) history.replaceState(null, "", "/games");
    else history.pushState(null, "", "/games");
  }
  if (HG.ui) HG.ui.showLobby();
  loadLobby().catch((e) => showErr(e.message));
  if (G.lobbyPoll) clearInterval(G.lobbyPoll);
  G.lobbyPoll = setInterval(() => {
    if (!G.gameId && !document.hidden) loadLobby().catch(() => {});
  }, LOBBY_POLL_MS);
}

async function route() {
  const m = location.pathname.match(/^\/games\/t\/([^/]+)$/);
  if (m) {
    try { await openTable(m[1], false); return; }
    catch (e) {
      // a table of a club I am not in: the lobby, with "ask to join the club" on top
      if (e.status === 403 && e.detail && e.detail.error === "club") {
        showLobby(true);
        if (HG.ui && HG.ui.openClubGate) HG.ui.openClubGate(e.detail, m[1]);
        return;
      }
      showErr(e.status === 404 ? "That table no longer exists." : e.message); showLobby(true); return;
    }
  }
  const inv = location.pathname.match(/^\/games\/join\/([A-Za-z0-9_-]+)$/);
  if (inv) {
    showLobby(true);
    if (HG.ui && HG.ui.openInvite) HG.ui.openInvite(inv[1]);
    return;
  }
  showLobby(true);
}

// ------------------------------------------------------------------- prefs
function loadPrefs() {
  try {
    const raw = globalThis.localStorage ? localStorage.getItem(PREFS_KEY) : null;
    if (raw) Object.assign(G.prefs, JSON.parse(raw));
  } catch (_) { /* private mode */ }
}
function savePrefs(patch) {
  Object.assign(G.prefs, patch || {});
  try { if (globalThis.localStorage) localStorage.setItem(PREFS_KEY, JSON.stringify(G.prefs)); } catch (_) { /* private mode */ }
  applyPrefs();
}
function applyPrefs() {
  const root = document.documentElement;
  if (root && root.dataset) {
    root.dataset.felt = G.prefs.felt;
    root.dataset.cards = G.prefs.cards;
    root.dataset.deck = G.prefs.deck;
    root.dataset.back = G.prefs.back;
    root.dataset.anim = G.prefs.anim;
  }
  if (HG.sound) { HG.sound.setEnabled(!!G.prefs.sound); HG.sound.setVolume(G.prefs.volume); }
}

HG.core = {
  G, $, j, post, act, deal, tablePost, refreshNow, fmtAmt, dollars, toCents, esc, chipsToCents, centsToChips,
  potBetTo, clampRaiseTo, raiseBoundsTo, myTurn, inHandAlive, heroToCallCents, setPreAction, savePrefs,
  applyPrefs, openTable, showLobby, loadLobby, showErr, startLive, stopLive, setClub,
};

async function init() {
  loadPrefs();
  applyPrefs();
  try {
    G.me = await j("/me");
  } catch (_) {
    G.me = null;
  }
  if (!G.me || !G.me.signed_in || !G.me.homegame) {
    document.body.innerHTML = "<h1>Not Found</h1>";
    return;
  }
  const unlock = () => { if (HG.sound) HG.sound.unlock(); };
  globalThis.addEventListener("pointerdown", unlock, { passive: true });
  globalThis.addEventListener("keydown", unlock);
  globalThis.addEventListener("popstate", () => { route(); });
  document.addEventListener("visibilitychange", () => { if (!document.hidden && G.gameId) refreshNow(); });
  if (HG.ui) HG.ui.init();
  if (HG.fair) HG.fair.init();
  await route();
}

init();
