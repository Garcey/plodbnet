"use strict";
// Home games — client core: state, networking, polling, pre-actions, routing.
// Rendering lives in games.table.js (the felt) and games.ui.js (everything
// around it); sounds in games.sound.js. This file has no DOM building in it,
// so it can be driven headless (tests/python/homegame/test_review_homegame_client_js.py).
//
// The SERVER deals the next hand (table setting "next hand after"); this client
// never auto-deals — a host who switched tabs used to stall the whole table.
// State arrives by live push (SSE, `startLive`); `startPoll` is the fallback.

const HG = (globalThis.HG = globalThis.HG || {});
const POLL_MS = 450;
const LOBBY_POLL_MS = 5000;
const PREFS_KEY = "hg.prefs.v1";
const PREF_DEFAULTS = {
  // anim: "auto" follows the device's Reduce Motion setting; "full" / "off" are the player's
  // own choice and win over it (motionOn).
  // sound = the master switch (the top bar's speaker); sndTurn / sndChat / sndTable = the
  // three kinds of sound (FEAT-011: "your turn" alone used to mean muting everything)
  sound: true, volume: 0.6, anim: "auto", deck: "4c", cards: "bold", back: "blue",
  felt: "emerald", unit: "dollars", hotkeys: true, confirmAllIn: false, bubbles: true,
  notify: false, rail: true, railTab: "chat", sndTurn: true, sndChat: true, sndTable: true, vibrate: true,
};
const PREFS_VERSION = 2; // (v1 had no "auto": see loadPrefs)

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
// One money style everywhere: a true minus sign ("−$5.00", as the felt's own lines
// already wrote it), and `short` drops a whole amount's cents ("$40" — preset buttons).
const MINUS = "−";
function dollars(cents, short) {
  const n = Number(cents || 0) / 100;
  const sign = n < 0 ? MINUS : "";
  const body = Math.abs(n).toFixed(2);
  return `${sign}$${short && body.endsWith(".00") ? body.slice(0, -3) : body}`;
}
function fmtAmt(cents, s) {
  if (G.prefs.unit === "bb" && s && s.stakes) {
    const bb = Number(cents || 0) / (s.stakes.bb_cents || 100);
    const t = Math.abs(bb);
    const body = Math.abs(t - Math.round(t)) < 0.05 ? String(Math.round(t)) : t.toFixed(1);
    return `${bb < 0 ? MINUS : ""}${body} bb`;
  }
  return dollars(cents);
}
// The ONE reader for a typed amount (every money / bb box): "$1,250.50", "40", "2.5",
// and a decimal comma — "2,50" is 2.50 (a comma followed by one or two digits at the
// end, with no point), "1,000" is a thousand. It used to delete every comma ("2,50" →
// $250) while the ante box beside it read "0,5" as half. null = not a number.
function readAmount(v) {
  let t = String(v == null ? "" : v).replace(/[\s$]|bb/gi, "").replace(MINUS, "-");
  if (t.includes(",")) t = !t.includes(".") && /^-?\d+,\d{1,2}$/.test(t) ? t.replace(",", ".") : t.replace(/,/g, "");
  if (t === "" || t === "-") return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}
function toCents(v) {
  if (String(v == null ? "" : v).trim() === "") return 0;  // (an empty box is 0, as ever)
  const n = readAmount(v);
  return n == null ? null : Math.round(n * 100);
}
// What a recorded action was, in words — ONE reading of the engine's action ids for the
// felt's seat badges, the Action tab and the replayer (FE-002: it was written three times).
function actionKind(x) {
  return x.action === 0 ? "fold" : x.action === 1 ? (x.chips > 0 ? "call" : "check") : x.action === 7 ? "allin" : "raise";
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

// ------------------------------------------------------------------ markup
// Markup is written ONLY in html`` templates (FE-003). Every value put into one is
// escaped — a name, a chat line, a club name, a code the server sent — unless it is
// markup itself: another html`` result, or raw() around a snippet this code wrote (an
// icon). A list of markup goes in as an array (never .join("")). One forgotten esc()
// can no longer let a player's name run as code in everyone's browser: the safe way is
// the only way (tests/python/homegame/test_homegame_client_markup.py checks every file).
// The result is a SafeHTML: games.ui.js put() / h() take it as markup and a plain
// string as TEXT; assigning it to innerHTML still works (it turns into its text).
class SafeHTML {
  constructor(s) { this.s = s; }
  toString() { return this.s; }
  // `markup + "…"` would quietly turn it into a plain string (shown as text, tags and
  // all): put the pieces in one html`` template instead
  [Symbol.toPrimitive](hint) {
    if (hint === "default") throw new TypeError("markup can't be joined with + (use one html`` template)");
    return this.s;
  }
}
function raw(s) { return s instanceof SafeHTML ? s : new SafeHTML(String(s == null ? "" : s)); }
function markupOf(v) {
  if (v instanceof SafeHTML) return v.s;
  if (Array.isArray(v)) return v.map(markupOf).join("");
  return esc(v);  // (null / undefined: nothing; numbers and booleans: their text)
}
function html(strings, ...values) {
  let out = strings[0];
  for (let i = 0; i < values.length; i++) out += markupOf(values[i]) + strings[i + 1];
  return new SafeHTML(out);
}
const isHTML = (v) => v instanceof SafeHTML;
// The one way content goes into an element: markup (html``) as markup, a DOM node as
// that node, anything else as TEXT. A value a template needs in CSS (an avatar's hue, a
// seat's place on a mini felt) rides in data-vars="h:212;x:40%" and is set here as a CSS
// custom property through the CSSOM — the page's CSP allows no inline style (FE-011), and
// the CSS decides what each variable does.
function put(el, content) {
  if (content instanceof SafeHTML) { el.innerHTML = content.s; if (content.s.includes("data-vars")) applyVars(el); }
  else if (content && content.nodeType) { el.textContent = ""; el.appendChild(content); }
  else el.textContent = content == null ? "" : String(content);
  return el;
}
function applyVars(root) {
  const set = (e) => {
    for (const pair of String(e.getAttribute("data-vars") || "").split(";")) {
      const k = pair.indexOf(":");
      if (k > 0) e.style.setProperty("--" + pair.slice(0, k).trim(), pair.slice(k + 1).trim());
    }
  };
  if (root.getAttribute && root.getAttribute("data-vars") != null) set(root);
  root.querySelectorAll("[data-vars]").forEach(set);
}

function showErr(msg) {
  const el = $("err");
  if (!msg) { el.hidden = true; el.textContent = ""; return; }
  if (HG.ui && HG.ui.toast) { HG.ui.toast(msg, "err"); return; }
  el.hidden = false;
  el.textContent = msg;
}

// Every request gives up after REQUEST_TIMEOUT_MS (a phone switching networks could
// hang one for minutes, and the fallback poll waits for it). A request that never
// reached the server — offline, timed out, or answered by a proxy's error page during
// a restart — fails with `network` set and a plain-words message (the callers retry
// quietly); the server's own answers ({detail}) keep their message and status.
const REQUEST_TIMEOUT_MS = 10000;
function netError(kind, status) {
  const msg = kind === "timeout" ? "The server is taking too long to answer — check your connection"
    : kind === "offline" ? "Can't reach the server — check your connection"
      : status === 502 || status === 503 || status === 504 ? "The server is restarting — try again in a moment"
        : `Something went wrong on the server (error ${status})`;
  const err = new Error(msg);
  err.network = true;
  err.kind = kind;
  err.status = status || 0;
  return err;
}
async function j(url, opts) {
  const o = { credentials: "same-origin", ...opts };
  const ctl = globalThis.AbortController && !o.signal ? new AbortController() : null;
  const timer = ctl ? setTimeout(() => ctl.abort(), o.timeout || REQUEST_TIMEOUT_MS) : null;
  if (ctl) o.signal = ctl.signal;
  delete o.timeout;
  let res, body;
  try {
    res = await fetch(url, o);
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("json")) body = await res.json();
    else if (!res.ok) {
      // a page that isn't ours (a proxy's 502 while the server restarts) never reaches a toast
      if (res.status >= 500) throw netError("server", res.status);
      body = { detail: res.status === 404 ? "Not Found" : res.statusText || `Error ${res.status}` };
    } else body = { detail: await res.text() };
  } catch (e) {
    if (e && e.network) throw e;
    if (e && e.name === "AbortError") throw netError("timeout");
    throw res ? netError("server", res.status) : netError("offline");  // (a garbled answer / no answer)
  } finally {
    if (timer) clearTimeout(timer);
  }
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
  noteBuild(s && s.client_build);
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
      try {
        // tapping it brings the table back to the front (FEAT-011); it goes away by itself
        // once the decision is gone (you acted elsewhere, or the clock did)
        const n = new Notification("Your turn", { body: s.name, tag: "hg-turn" });
        n.onclick = () => { try { globalThis.focus(); } catch (_) { /* not allowed */ } n.close(); };
        G.turnNote = n;
      } catch (_) { /* optional */ }
    }
  }
  if (!key && G.turnNote) { try { G.turnNote.close(); } catch (_) { /* gone */ } G.turnNote = null; }
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
    // (every home-games call answers 404 to a signed-out user: say THAT, not "Not Found")
    if (e.status === 404 && await signedOut()) { showSignedOut(); throw e; }
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
        tableGone();
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
        if (e.status === 404 && G.live === es) { tableGone(); return; }
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

// ------------------------------------------------------------ signed out
// Every home-games call answers 404 to someone who isn't signed in, so a 404 alone
// can't tell "that table is gone" from "your sign-in ended" (signed out in another
// tab, an expired session) — /me can. Signed out, the page says so and signs you
// back in to the same table (a reload serves the sign-in page for this address).
async function signedOut() {
  try { const me = await j("/me"); return !(me && me.signed_in); } catch (_) { return false; }
}
function showSignedOut() {
  stopLive();
  if (G.lobbyPoll) { clearInterval(G.lobbyPoll); G.lobbyPoll = null; }
  if (HG.ui && HG.ui.signedOut) HG.ui.signedOut();
}
async function tableGone() {
  stopLive();
  if (await signedOut()) { showSignedOut(); return; }
  showErr("That table no longer exists.");
  showLobby(true);
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
  noteBuild(data && data.client_build);
  if (HG.ui) HG.ui.renderLobby(data);
  return data;
}

// ------------------------------------------------------- a new version (OPS-039)
// The page names the client build it was served with (<meta name="hg-build">, filled
// in by the server); the lobby and table views name the server's current one. After
// a deploy an open page keeps running the old code against the new server, so it
// offers a refresh while that costs nothing — in the lobby, or at a table when you
// are not holding cards (the dock's status strip, games.play.js) — and a tab in the
// background with nothing to lose just reloads itself: once per build, so a
// mismatch that survives a reload can never loop.
const PAGE_BUILD = (() => {
  try { const m = document.querySelector('meta[name="hg-build"]'); return (m && m.content) || ""; } catch (_) { return ""; }
})();
function noteBuild(build) {
  if (build && PAGE_BUILD && build !== PAGE_BUILD) G.newBuild = build;
  if (G.newBuild) offerUpdate();
}
function holdingCards(s) {
  if (!s || !Number.isInteger(s.my_seat)) return false;
  const me = s.seats[s.my_seat];
  return !!(me && me.in_hand && !me.folded && (s.phase === "in_hand" || (s.runout && s.runout.blocking)));
}
function offerUpdate() {
  if (!G.newBuild) return;
  const idle = !holdingCards(G.gameId ? G.state : null);
  const U = HG.uiState, a = document.activeElement;
  const busy = (U && (U.modals.length || U.drawer)) || (a && /^(INPUT|TEXTAREA|SELECT)$/.test(a.tagName) && a.value);
  if (idle && document.hidden && !busy && !reloadedFor(G.newBuild)) { reloadForUpdate(); return; }
  if (HG.ui && HG.ui.showUpdate) HG.ui.showUpdate(idle);
}
function reloadedFor(build) {
  try { return sessionStorage.getItem("hg.reloadedFor") === build; } catch (_) { return true; }
}
function reloadForUpdate() {
  try { sessionStorage.setItem("hg.reloadedFor", G.newBuild || ""); } catch (_) { /* private mode */ }
  location.reload();
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
  G.lobbyFails = 0;
  lobbyTick();
  if (G.lobbyPoll) clearInterval(G.lobbyPoll);
  G.lobbyPoll = setInterval(() => {
    if (!G.gameId && !document.hidden) lobbyTick();
  }, LOBBY_POLL_MS);
}
// One lobby load. A failure says so in the lobby ("Can't reach the server — retrying")
// and, from the second in a row, raises the same "Connection lost" bar as a table; the
// 5 s poll is the retry. A server's own refusal is said once, as a toast.
function lobbyTick() {
  return loadLobby().then(() => {
    G.lobbyFails = 0;
    setConn("ok");
    if (HG.ui && HG.ui.lobbyOffline) HG.ui.lobbyOffline(false);
  }).catch(async (e) => {
    if (G.gameId) return;
    if (e.status === 404 && await signedOut()) { showSignedOut(); return; }
    G.lobbyFails = (G.lobbyFails || 0) + 1;
    if (HG.ui && HG.ui.lobbyOffline) HG.ui.lobbyOffline(true);
    if (e.network) { if (G.lobbyFails >= 2) setConn("off"); }
    else if (G.lobbyFails === 1) showErr(e.message);
  });
}

// Hand review (2026-10-03, games.review.js): the page of the player's uploaded hand
// histories — no table, no lobby poll.
function showReview(push) {
  stopLive();
  G.gameId = null;
  G.state = null;
  if (G.lobbyPoll) { clearInterval(G.lobbyPoll); G.lobbyPoll = null; }
  if (push && location.pathname !== "/games/review") history.pushState(null, "", "/games/review");
  if (HG.ui && HG.ui.showReview) HG.ui.showReview();
}

async function route() {
  if (location.pathname === "/games/review") { showReview(false); return; }
  const m = location.pathname.match(/^\/games\/t\/([^/]+)$/);
  if (m) {
    // a link to one hand (the replayer's "Copy link"): the table, then that hand on top
    const hand = Number((String(location.search || "").match(/[?&]hand=(\d+)/) || [])[1]) || 0;
    try {
      await openTable(m[1], false);
      if (hand && HG.ui && HG.ui.openHand) { history.replaceState(null, "", `/games/t/${m[1]}`); HG.ui.openHand(m[1], hand); }
      return;
    }
    catch (e) {
      // a table of a club I am not in: the lobby, with "ask to join the club" on top
      if (e.status === 403 && e.detail && e.detail.error === "club") {
        showLobby(true);
        if (HG.ui && HG.ui.openClubGate) HG.ui.openClubGate(e.detail, m[1]);
        return;
      }
      if (e.status === 404 && await signedOut()) { showSignedOut(); return; }
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
    const p = raw ? JSON.parse(raw) : null;
    if (p && typeof p === "object") {
      // v1 saved its default anim "full" along with every other preference, so a stored
      // "full" is not a choice: it becomes "auto" (= full unless the device asks for reduced
      // motion). An explicit "off" stays.
      if (!(p.pv >= 2) && p.anim === "full") p.anim = "auto";
      Object.assign(G.prefs, p);
    }
  } catch (_) { /* private mode */ }
  G.prefs.pv = PREFS_VERSION;
}
const REDUCED_MOTION = globalThis.matchMedia ? globalThis.matchMedia("(prefers-reduced-motion: reduce)") : null;
// Whether the felt animates: the player's choice, or on "auto" the device's Reduce Motion.
function motionOn() {
  const a = G.prefs.anim;
  if (a === "off") return false;
  if (a === "full") return true;
  return !(REDUCED_MOTION && REDUCED_MOTION.matches);
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
    root.dataset.anim = motionOn() ? "full" : "off";  // (what applies — games.css keys on it)
  }
  if (HG.sound) {
    HG.sound.setEnabled(!!G.prefs.sound);
    HG.sound.setVolume(G.prefs.volume);
    if (HG.sound.setKinds) HG.sound.setKinds({ turn: G.prefs.sndTurn !== false, chat: G.prefs.sndChat !== false, table: G.prefs.sndTable !== false });
  }
}

HG.core = {
  G, $, j, post, act, deal, tablePost, refreshNow, fmtAmt, dollars, toCents, readAmount, esc, html, raw, isHTML, put, applyVars, chipsToCents, centsToChips, actionKind,
  potBetTo, clampRaiseTo, raiseBoundsTo, myTurn, inHandAlive, heroToCallCents, setPreAction, savePrefs,
  applyPrefs, motionOn, reloadForUpdate, openTable, showLobby, showReview, loadLobby, showErr, startLive, stopLive, setClub,
};

// Who is signed in. A hiccup — the server restarting during a deploy, a flaky phone
// connection — is not "Not Found": the page says it can't reach the server and tries
// again by itself (1 s, 2 s, 4 s … every 15 s; "Try now" skips the wait). Signed out,
// it offers to sign in again; "Not Found" is only for an account the server says has
// no home games.
async function whoAmI() {
  const notFound = () => { put(document.body, html`<h1>Not Found</h1>`); return null; };
  for (let wait = 1000; ; wait = Math.min(wait * 2, 15000)) {
    try {
      const me = await j("/me");
      if (HG.ui && HG.ui.bootProblem) HG.ui.bootProblem(null);
      if (!me || !me.signed_in) { showSignedOut(); return null; }
      return me.homegame ? me : notFound();
    } catch (e) {
      if (!e.network) return notFound();
      await new Promise((resolve) => {
        const t = setTimeout(resolve, wait);
        if (HG.ui && HG.ui.bootProblem) HG.ui.bootProblem(e.message, () => { clearTimeout(t); resolve(); });
      });
    }
  }
}

async function init() {
  loadPrefs();
  applyPrefs();
  if (REDUCED_MOTION) {  // (the device setting changed while the page is open)
    if (REDUCED_MOTION.addEventListener) REDUCED_MOTION.addEventListener("change", applyPrefs);
    else if (REDUCED_MOTION.addListener) REDUCED_MOTION.addListener(applyPrefs);  // (Safari < 14)
  }
  G.me = await whoAmI();
  if (!G.me) return;  // (the page says why)
  const unlock = () => { if (HG.sound) HG.sound.unlock(); };
  globalThis.addEventListener("pointerdown", unlock, { passive: true });
  globalThis.addEventListener("keydown", unlock);
  globalThis.addEventListener("popstate", () => { route(); });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && G.gameId) refreshNow();
    else if (document.hidden) offerUpdate();  // (a new version reloads while nobody looks)
  });
  if (HG.ui) HG.ui.init();
  if (HG.fair) HG.fair.init();
  await route();
}

init();
