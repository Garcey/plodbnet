// Study / Trainer client, part 1 of 7 — foundations: card constants,
// display units and number formatting, small helpers, toasts and dialogs,
// the shared UI state object, and every request to the server (fetching,
// posting and applying the state).
//
// Plain scripts, no build step: index.html loads app.core.js, app.table.js,
// app.play.js, app.study.js, app.trainer.js, app.topbar.js and app.js in
// that order, and they share one global scope — any file may call a
// function from any other. Code that runs while a file LOADS may use only
// the files before it; everything else starts from init() in app.js.
"use strict";

const RANK_STRINGS = "23456789TJQKA";
const SUIT_STRINGS = ["c", "d", "h", "s"];
const SUIT_GLYPHS = { c: "\u2663", d: "\u2666", h: "\u2665", s: "\u2660" };
const RED_SUITS = new Set(["d", "h"]);
const SUIT_COLOR_VAR = {
  h: "var(--card-hearts)",
  d: "var(--card-diamonds)",
  c: "var(--card-clubs)",
  s: "var(--card-spades)",
};

// (Table geometry — seat ellipse, slot sizes, board positions — lives in
// TABLE_LAYOUTS: one drawing for wide screens, one for upright phones.)

function cardToString(c) {
  if (c === null || c === undefined) return null;
  const rank = RANK_STRINGS[Math.floor(c / 4)];
  const suit = SUIT_STRINGS[c % 4];
  return {
    rank,
    suit,
    glyph: SUIT_GLYPHS[suit],
    red: RED_SUITS.has(suit),
    color: SUIT_COLOR_VAR[suit],
  };
}

function chipsToBB(chips, state) {
  const bb = state?.chip_scale?.bb_chips || 10000;
  return chips / bb;
}
function bbToChips(bb, state) {
  const bbChips = state?.chip_scale?.bb_chips || 10000;
  return Math.round(bb * bbChips);
}

// --- Display units (ST-010 / ST-019 / ST-013 / ST-023) -----------------------
// Amounts show in big blinds unless the viewer picked "$"; the choice is
// remembered in this browser. "$" converts at ONE rate per format that Study
// and Trainer share (they used to convert at $2/bb and $20/bb, so the same
// 100bb stack read $200 in one tab and $2,000 in the other). Chips stay the
// unit of every computation — conversion and formatting happen at the edge.
const UNIT_PREF_KEY = "plo5bp-unit";
const RATE_PREF_KEY = "plo5bp-dpb";   // {format id: dollars per bb}

function loadRatePrefs() {
  try {
    const r = JSON.parse(lsGet(RATE_PREF_KEY));
    return r && typeof r === "object" && !Array.isArray(r) ? r : {};
  } catch (_) { return {}; }
}
function validRate(r) { return typeof r === "number" && isFinite(r) && r > 0; }

// Dollars per big blind for this state's format: the viewer's rate when set,
// otherwise the server's default for the mode (only reachable before the
// viewer ever switched to "$" — see setUnit).
function dollarsPerBB(state) {
  const f = state && state.format;
  const pref = f && UI.rates ? UI.rates[f] : undefined;
  if (validRate(pref)) return pref;
  const srv = state && state.chip_scale ? state.chip_scale.dollars_per_bb : undefined;
  return validRate(srv) ? srv : 1;
}
function chipsToCurrentUnit(chips, state) {
  const bb = chipsToBB(chips, state);
  return UI.unit === "bb" ? bb : bb * dollarsPerBB(state);
}
function parseToChips(val, state) {
  const n = parseFloat(val);
  if (!isFinite(n) || n < 0) return null;
  return bbToChips(UI.unit === "bb" ? n : n / dollarsPerBB(state), state);
}

// --- Number formatting (CPY-018) ---------------------------------------------
// "20bb", "2.5bb", "1,250bb", "$1,250", "$12.50": thousands separators, no
// ".00" on whole amounts, big blinds to at most two decimals, dollars with
// cents only when there are cents. Input boxes use unitInputValue instead
// (a plain number the field can parse).
const MINUS = "−";
function fmtNum(v, maxFrac = 2, minFrac = 0) {
  const n = Number(v);
  if (!isFinite(n)) return "—";
  const r = Math.abs(n) < 0.5 * 10 ** -maxFrac ? 0 : n;   // no "-0"
  return r.toLocaleString("en-US", {
    minimumFractionDigits: minFrac, maximumFractionDigits: maxFrac,
  }).replace("-", MINUS);
}
// `value` is already in `unit` ("bb" | "$").
function fmtAmountIn(value, unit) {
  const n = Number(value) || 0;
  if (unit === "bb") return `${fmtNum(n, 2)}bb`;
  const whole = Math.abs(Math.round(n * 100)) % 100 === 0;
  const abs = fmtNum(Math.abs(n), whole ? 0 : 2, whole ? 0 : 2);
  return `${n < 0 && abs !== "0" ? MINUS : ""}$${abs}`;
}
function formatUnit(chips, state) {
  return fmtAmountIn(chipsToCurrentUnit(chips, state), UI.unit);
}
// A big-blind quantity (EV, result, loss) in the active unit.
// " ±1.2bb": the Monte-Carlo EV-loss estimate's standard error (BE-006).
function evBand(se, state) {
  const n = Number(se);
  return (se === null || se === undefined || !isFinite(n) || n <= 0) ? "" : ` ±${fmtBBValue(n, state)}`;
}
function fmtBBValue(bb, state) {
  const n = Number(bb) || 0;
  return fmtAmountIn(UI.unit === "bb" ? n : n * dollarsPerBB(state), UI.unit);
}
// Signed big-blind quantity: "+1.25bb", "−$24".
function fmtSignedValue(bb, state) {
  if (bb === null || bb === undefined || !isFinite(Number(bb))) return null;
  const n = Number(bb);
  const body = fmtBBValue(Math.abs(n), state);
  if (body === "0bb" || body === "$0") return body;
  return `${n >= 0 ? "+" : MINUS}${body}`;
}
// The value an input box shows for `chips`: plain digits, at most 2 decimals.
function unitInputValue(chips, state) {
  const v = chipsToCurrentUnit(chips, state);
  return String(Math.round(v * 100) / 100);
}
function unitSuffix() { return UI.unit === "bb" ? "bb" : "$"; }
// localStorage can throw (blocked site data, private mode, quota). These run
// at script load, so an unguarded access killed the whole UI
// (review 2026-09-20 F17). Function declarations: hoisted above `UI`.
function lsGet(key) {
  try { return localStorage.getItem(key); } catch (_) { return null; }
}
function lsSet(key, val) {
  try { localStorage.setItem(key, val); } catch (_) { /* storage unavailable */ }
}
// Public build flag: the server injects `window.PLO5BP_PUBLIC`; a
// `data-public="1"` on <html> is accepted too (for a CSP without inline
// scripts).
function isPublicBuild() {
  return window.PLO5BP_PUBLIC === true
    || (document.documentElement && document.documentElement.dataset.public === "1");
}
// Escape server/user-supplied text before it goes through innerHTML
// (review 2026-09-20 F17).
function escapeHTML(v) {
  return String(v ?? "").replace(/[&<>"']/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
  ));
}
// "Raise" vs "Bet". Any preflop aggression is a raise — the big blind is
// already a live bet even when nothing is owed (BB option) or no raise is in
// the history yet (an open) (review 2026-09-20 F17).
function aggVerb(toCall, street) {
  return (toCall > 0 || street === "preflop") ? "Raise" : "Bet";
}
// True when a raise-BY delta puts the actor all-in. Engine raise chips are
// deltas on top of the street commit, so the comparison is against the
// REMAINING stack — not stack + commit (review 2026-09-20 F8).
function isAllInDelta(actorSeat, deltaChips) {
  return !!actorSeat && deltaChips > 0 && deltaChips >= actorSeat.stack_chips;
}
// Action label in the ACTIVE display unit. Server-built *_label strings
// are bb-only; prefer this whenever raw gate/chips are in the payload.
// `chips` is the engine's raise-BY delta; `commit` is the actor's street
// commit before acting, so the label shows the raise-TO total — the same
// convention as the action buttons / raise input (review 2026-09-20 F11).
// An unknown commit only matters for a raise (an opening bet has commit 0),
// and then the label shows "+chips added" instead of passing a delta off as
// a total.
function gateActionLabel(gateSlug, chips, toCall, state, commit, street) {
  if (gateSlug === "fold") return "Fold";
  if (gateSlug === "check_call") return toCall > 0 ? "Call" : "Check";
  const verb = aggVerb(toCall, street);
  const known = typeof commit === "number" && isFinite(commit);
  if (!known && verb === "Raise") return `Raise +${formatUnit(chips || 0, state)}`;
  return `${verb} ${formatUnit((chips || 0) + (known ? commit : 0), state)}`;
}
// Street commits around every action of a hand: [{ before, after, level }]
// per entry, where `level` is the street's bet level before the action.
// `deltaOf(entry, before, level)` returns the chips the action added. Commits
// reset at each street change; a preflop first street is seeded with the
// live blind posts (blinds are never history records).
function streetCommitWalk(s, entries, deltaOf) {
  const out = new Array(entries.length);
  let street = null, commits = {}, level = 0;
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    if (i === 0 || e.street !== street) {
      street = e.street;
      commits = (i === 0 && street === "preflop") ? liveBlindSeed(s) : {};
      level = 0;
      for (const k of Object.keys(commits)) level = Math.max(level, commits[k]);
    }
    const before = commits[e.seat] || 0;
    const after = before + Math.max(0, Number(deltaOf(e, before, level)) || 0);
    commits[e.seat] = after;
    out[i] = { before, after, level };
    level = Math.max(level, after);
  }
  return out;
}
// Blind posts per seat, derived from what the payload already carries:
// total commit − ante − every recorded action's chips. Empty for ante-only
// (bomb pot) formats. Only consulted for hands with a preflop street.
function liveBlindSeed(s) {
  const seed = {};
  if (!s || !s.seats || !s.chip_scale) return seed;
  const bbChips = s.chip_scale.bb_chips || 10000;
  const ante = s.chip_scale.ante_chips || 0;
  const acted = {};
  for (const h of s.history || []) acted[h.seat] = (acted[h.seat] || 0) + (Number(h.chips) || 0);
  for (const seat of s.seats) {
    if (typeof seat.committed_total_bb !== "number") continue;
    const total = Math.round(seat.committed_total_bb * bbChips);
    const start = s.starting_stacks_chips ? s.starting_stacks_chips[seat.seat] : null;
    const antePaid = typeof start === "number" ? Math.min(ante, start) : ante;
    const blind = total - antePaid - (acted[seat.seat] || 0);
    if (blind > 0 && blind <= bbChips) seed[seat.seat] = blind;
  }
  return seed;
}
// Walk over state.history (every entry's chips = chips actually added).
function historyCommits(s) {
  return streetCommitWalk(s, (s && s.history) || [], (h) => h.chips);
}
// Chips the current actor has already committed THIS street. Engine raise
// bounds / recommendation are DELTAS on top of this; the UI displays totals,
// where total = delta + actorCommitChips. Returns 0 (-> total == delta, a safe
// no-op) for opening bets or when there is no actor.
function actorCommitChips(s) {
  return (s && s.actor !== null && s.actor !== undefined && s.seats[s.actor])
    ? s.seats[s.actor].committed_this_street_chips : 0;
}
// Sentinel for auth/paywall errors already handled by an overlay/modal —
// callers' generic `catch (e) { showToast(e.message); }` stays quiet.
const GATE_HANDLED = "__gate_handled__";

// Toasts (A11Y-014): errors are announced as alerts, notices politely; each
// has a close button and stays while hovered or focused. The same message
// already on screen restarts its timer instead of stacking a copy.
const TOAST_MS = { error: 6000, info: 4000 };
function showToast(msg, kind = "error") {
  // Also quiet when a caller wrapped the sentinel ("X failed: __gate…").
  if (typeof msg === "string" && msg.includes(GATE_HANDLED)) return;
  const container = document.getElementById("toast-container");
  if (!container) return;
  const text = msg instanceof Error ? msg.message : String(msg ?? "");
  if (!text) return;
  for (const t of container.children) {
    if (t.dataset.msg === text && typeof t._restart === "function") { t._restart(); return; }
  }
  const toast = document.createElement("div");
  toast.className = `toast toast-${kind}`;
  toast.dataset.msg = text;
  toast.setAttribute("role", kind === "error" ? "alert" : "status");
  const body = document.createElement("span");
  body.className = "toast-msg";
  body.textContent = text;
  const close = document.createElement("button");
  close.type = "button";
  close.className = "toast-x";
  close.setAttribute("aria-label", "Dismiss");
  close.textContent = "×";
  toast.append(body, close);
  let timer = null;
  const stop = () => { if (timer) { clearTimeout(timer); timer = null; } };
  const start = () => { stop(); timer = setTimeout(() => toast.remove(), TOAST_MS[kind] || 5000); };
  toast._restart = start;
  close.addEventListener("click", () => { stop(); toast.remove(); });
  toast.addEventListener("mouseenter", stop);
  toast.addEventListener("mouseleave", start);
  toast.addEventListener("focusin", stop);
  toast.addEventListener("focusout", start);
  container.appendChild(toast);
  start();
}

// --- Dialogs (A11Y-015) --------------------------------------------------------
// Every pop-up is a native <dialog> opened with showModal(): focus moves in
// and returns to where it was, Tab stays inside, Escape closes, the page
// behind is inert, and a click on the dimmed backdrop closes it.
function openDialog(dlg) {
  if (!dlg || dlg.open) return;
  dlg._returnFocus = document.activeElement;
  if (!dlg._wired) {
    dlg._wired = true;
    // Backdrop = the dialog element itself (its content fills it). Only a
    // press that STARTED there closes it, so a text drag out of a field
    // doesn't.
    dlg.addEventListener("pointerdown", (e) => { dlg._downOnBackdrop = e.target === dlg; });
    dlg.addEventListener("click", (e) => {
      if (e.target === dlg && dlg._downOnBackdrop && !dlg.hasAttribute("data-modal-only")) {
        closeDialog(dlg, "cancel");
      }
    });
    dlg.addEventListener("close", () => {
      const back = dlg._returnFocus;
      dlg._returnFocus = null;
      const cb = dlg._onClose;
      dlg._onClose = null;
      if (back && typeof back.focus === "function" && document.contains(back)) {
        try { back.focus({ preventScroll: true }); } catch (_) { /* detached */ }
      }
      if (typeof cb === "function") cb(dlg.returnValue);
    });
  }
  dlg.returnValue = "";
  if (typeof dlg.showModal === "function") dlg.showModal();
  else dlg.setAttribute("open", "");
}
function closeDialog(dlg, value) {
  if (!dlg || !dlg.open) return;
  if (typeof dlg.close === "function") dlg.close(value);
  else { dlg.removeAttribute("open"); dlg.dispatchEvent(new Event("close")); }
}
// In-app confirmation (ST-018) — resolves true when confirmed.
function confirmDialog({ title, body, ok = "Continue", cancel = "Cancel", danger = false }) {
  const dlg = document.getElementById("confirm-dlg");
  if (!dlg) return Promise.resolve(true);
  document.getElementById("confirm-title").textContent = title || "";
  document.getElementById("confirm-body").textContent = body || "";
  const okBtn = document.getElementById("confirm-ok");
  const cancelBtn = document.getElementById("confirm-cancel");
  okBtn.textContent = ok;
  okBtn.classList.toggle("danger", !!danger);
  cancelBtn.textContent = cancel;
  return new Promise((resolve) => {
    dlg._onClose = (v) => resolve(v === "ok");
    openDialog(dlg);
    (danger ? cancelBtn : okBtn).focus();
  });
}

const UI = {
  // Display unit: big blinds unless this browser chose "$" (ST-019).
  unit: lsGet(UNIT_PREF_KEY) === "$" ? "$" : "bb",
  rates: loadRatePrefs(),
  // ?mode= wins, then the last tab used here; a first visit opens the Trainer,
  // which deals a hand straight away (ST-016) — Study starts from an empty
  // table that needs a whole spot entered by hand.
  mode: (() => {
    const q = new URLSearchParams(location.search).get("mode");
    if (q === "trainer" || q === "study") return q;
    return lsGet("plo5bp-mode") === "study" ? "study" : "trainer";
  })(),
  selectedSlot: null,
  lastState: null,
  lastStateKey: null,
  cardsPostPending: false,
  cardsPostInFlight: false,
  draggingButton: false,
  // Public build: /me payload (null = not fetched / signed out).
  me: null,
  // GET /formats payload: list of {id, label, model_loaded}. null until fetched.
  formats: null,
  raiseUserSet: false,
  raiseLastActor: null,
  raiseNodeKey: null,     // decision node the raise input's value belongs to
  // Mode/format context epoch: bumped on every mode or format switch. A
  // response requested under an older epoch is stale and must not be applied
  // (review 2026-09-20 F4).
  ctxSeq: 0,
  resyncing: false,       // a quiet state refetch (409 / failed POST) is queued
  // Trainer
  actCommit: null,        // {handNo, commit, street} at the hero's last POSTed action
  feedbackShownIdx: -1,
  feedbackTimer: null,
  reviewNode: null,       // node (action_log) index currently shown, or null
  trainerPick: false,     // card grid open for a what-if swap
  settingsOpen: false,
  animSeq: 0,             // bumped to cancel an in-flight frame animation
  animating: false,
  // In-flight guard for state-mutating action / new-hand / raise POSTs.
  // Set true BEFORE the POST is issued and cleared in a finally after the
  // response is fully handled (incl. 401/402/500). Blocks double-clicks
  // (double-graded decisions, wrong-seat folds, burned free hands) while
  // one request is outstanding. Composes with `animating`: this covers the
  // network round-trip, `animating` covers the subsequent frame playback.
  actionInFlight: false,
  // Study: actions taken off by Undo / a history click, replayable by Redo.
  redo: [],
  workingText: null,      // "Dealing…" while a deal / graded move is in flight
};

// Opponent-action playback prefs (client-only, global across formats; set in
// the trainer settings modal). animMs = pause between opponent action frames
// (0 = instant). ffFold = once the hero folds, skip watching opponents finish.
const TRAINER_ANIM_DEFAULT_MS = 1200;
const TRAINER_ANIM_MAX_MS = 8000;
const trainerPrefs = (() => {
  let ms = parseInt(lsGet("plo5bp-trainer-anim-ms"), 10);
  if (!Number.isFinite(ms) || ms < 0) ms = TRAINER_ANIM_DEFAULT_MS;
  ms = Math.min(ms, TRAINER_ANIM_MAX_MS);
  const ff = lsGet("plo5bp-trainer-ff-fold");
  return { animMs: ms, ffFold: ff === null ? true : ff === "true" };
})();
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function apiBase() {
  return UI.mode === "trainer" ? "/trainer" : "/study";
}

// Public build: track remaining free hands from the response header and
// intercept auth (401) / paywall (402) responses centrally.
function noteFreeHands(res) {
  const left = res.headers.get("X-Free-Hands-Left");
  if (left !== null && UI.me && UI.me.free) {
    UI.me.free.left = parseInt(left, 10);
    UI.me.free.used = Math.max(0, UI.me.free.limit - UI.me.free.left);
    renderAccountChip();
  }
}

// Quietly re-pull the authoritative state (no toast). Deferred a tick so the
// failing caller's finally{} (in-flight flag / busy state) unwinds first;
// `resyncing` keeps a failing refetch from re-queueing itself.
function resyncQuietly() {
  if (UI.resyncing) return;
  UI.resyncing = true;
  setTimeout(async () => {
    UI.lastStateKey = null;  // repaint even when the server state is unchanged
    try { await fetchState(); } finally { UI.resyncing = false; }
  }, 0);
}

// Table routes (Study + Trainer) as opposed to account / billing ones.
function isGameRoute(url) {
  return !/^\/(billing|auth|admin|me)(\/|\?|$)/.test(String(url));
}

async function gateIntercept(res, url) {
  // 409 on a table route: the server has nothing to apply this request to —
  // e.g. POST /trainer/act with no active hand. The client's picture is
  // stale, so resync instead of toasting a raw error (review 2026-09-20).
  // Other routes' 409s carry a message for the user (e.g. checkout while the
  // site is free), so they surface like any other error (FE-013).
  if (res.status === 409 && isGameRoute(url)) {
    resyncQuietly();
    return true;
  }
  if (!isPublicBuild()) return false;
  if (res.status === 401) {
    // Only signed-in pages talk to these routes (a signed-out visitor gets
    // the landing page from the server), so a 401 here means the session
    // ended while the page was open (ACC-026).
    showSessionExpired();
    return true;
  }
  if (res.status === 402) {
    let body = {};
    try { body = await res.clone().json(); } catch (_) {}
    showPaywall(body);
    return true;
  }
  return false;
}

// States whose request was issued under an older mode/format context
// (UI.ctxSeq). applyState / the frame player drop them, so a slow response
// can't clobber the screen after a mode or format switch
// (review 2026-09-20 F4).
const STALE_STATES = new WeakSet();
function tagIfStale(ctx, data) {
  if (ctx !== UI.ctxSeq && data && data.state && typeof data.state === "object") {
    STALE_STATES.add(data.state);
  }
  return data;
}

// --- Requests (FE-013 / FE-014) ------------------------------------------------
// Every call times out: a request hung by a server restart used to keep the
// action buttons disabled until the browser gave up (~100 s behind
// Cloudflare), with no message. Every failure — GETs included, which used to
// show only the statusText that HTTP/2 leaves empty ("502 ") — becomes one
// plain-English sentence without a status code.
const REQUEST_TIMEOUT_MS = 20000;

class ApiError extends Error {
  constructor(message, status, code) {
    super(message);
    this.status = status;
    this.code = code || "";
  }
}

// Server messages that are too technical to show as they are.
const ERROR_REWRITES = [
  [/out of raise range/i, "That amount isn't a legal size here — pick one inside the range shown."],
  [/hero hole cards required/i, "Place your hole cards first."],
  [/\b(flop|turn|river) cards required/i, (m) => `Place the ${m[1].toLowerCase()} cards first.`],
  [/gate '\w+' not legal/i, "That action isn't available right now."],
  [/no actor/i, "The hand is over — start a new one."],
  [/nothing to undo/i, "Nothing to undo."],
  [/action rejected by the engine/i, "That action isn't allowed here."],
  [/duplicate card/i, "That card is already on the table."],
];

function friendlyDetail(status, detail) {
  if (typeof detail === "string" && detail.trim()) {
    for (const [re, out] of ERROR_REWRITES) {
      const m = detail.match(re);
      if (m) return typeof out === "function" ? out(m) : out;
    }
    if (status > 0 && status < 500) return detail.trim();
  }
  if ([502, 503, 504, 520, 521, 522, 523, 524].includes(status)) {
    return "WrapGTO is restarting — try again in a moment.";
  }
  if (status >= 500) return "Something went wrong on our side — please try again.";
  if (status === 404) return "That isn't available any more — reload the page.";
  if (status === 413) return "That request is too large.";
  if (status === 429) return "Too many requests — wait a moment and try again.";
  return "That didn't work — please try again.";
}

async function readDetail(res) {
  try {
    const body = await res.clone().json();
    const d = body ? body.detail : undefined;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      return d.map((x) => (x && typeof x.msg === "string" ? x.msg.replace(/^Value error, /, "") : ""))
        .filter(Boolean).join("; ");
    }
    if (d && typeof d === "object" && typeof d.message === "string") return d.message;
  } catch (_) { /* not JSON */ }
  return "";
}

async function apiFetch(url, init) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), REQUEST_TIMEOUT_MS);
  try {
    return await fetch(url, { ...(init || {}), signal: ctl.signal });
  } catch (e) {
    if (e && e.name === "AbortError") {
      throw new ApiError("WrapGTO is taking too long to answer — please try again.", 0, "timeout");
    }
    throw new ApiError("Can't reach WrapGTO — check your connection and try again.", 0, "network");
  } finally {
    clearTimeout(timer);
  }
}

async function requestJSON(method, url, body) {
  const ctx = UI.ctxSeq;
  let res;
  try {
    res = await apiFetch(url, method === "GET" ? undefined : {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
    });
  } catch (e) {
    // A POST that timed out may still have landed server-side: re-pull the
    // authoritative state once the caller has re-enabled its controls.
    if (method !== "GET" && isGameRoute(url)) resyncQuietly();
    throw e;
  }
  noteFreeHands(res);
  if (!res.ok) {
    if (await gateIntercept(res, url)) throw new Error(GATE_HANDLED);
    throw new ApiError(friendlyDetail(res.status, await readDetail(res)), res.status);
  }
  let data;
  try { data = await res.json(); }
  catch (_) {
    throw new ApiError("WrapGTO sent an unreadable answer — please try again.", res.status, "parse");
  }
  return tagIfStale(ctx, data);
}
function postJSON(url, body) { return requestJSON("POST", url, body); }
function getJSON(url) { return requestJSON("GET", url); }

async function fetchState() {
  // A newer authoritative state (or a frame animation) that landed while this
  // GET was in flight wins: applying the older snapshot would roll the screen
  // back and abort the playback (review 2026-09-20 F6). Every applyState and
  // animation start bumps animSeq, so it doubles as the state sequence.
  // Resolves true only when the fetched state was handed to applyState.
  const seq = UI.animSeq;
  try {
    const data = await getJSON(`${apiBase()}/state`);
    if (UI.animSeq !== seq) return false;
    applyState(data.state);
    return true;
  } catch (e) {
    if (e && e.message === GATE_HANDLED) return false;
    // Nothing on screen yet: an empty felt with a toast looked broken — say
    // so on the table itself, with a Retry (ST-028).
    if (!UI.lastState) showLoadError(e.message);
    else showToast(e.message);
    return false;
  }
}
// A study card slot was just mutated locally (UI.lastState.card_spec, in
// place): paint it right away, then sync. The screen no longer matches the
// keyed server state, so the dedupe key is dropped (review 2026-09-20 F1).
function commitLocalCards() {
  UI.lastStateKey = null;
  if (UI.lastState) render(UI.lastState);
  postCards();
}
async function postCards() {
  UI.cardsPostPending = true;
  if (UI.cardsPostInFlight) return;
  UI.cardsPostInFlight = true;
  try {
    while (UI.cardsPostPending) {
      UI.cardsPostPending = false;
      const s = UI.lastState;
      // Study-only endpoint: never post a trainer spec after a mode switch.
      if (!s || s.trainer || UI.mode !== "study") break;
      try {
        const data = await postJSON("/study/cards", {
          hero_hole: s.card_spec.hero_hole,
          flop_a: s.card_spec.flop_a,
          flop_b: s.card_spec.flop_b,
          turn: s.card_spec.turn,
          river: s.card_spec.river,
        });
        // Clicks that landed while this POST was in flight mutated the local
        // spec and queued another post. The response predates them — keep the
        // local spec (the next loop pass posts it) instead of letting the
        // response erase those cards (review 2026-09-20 F1).
        const cur = UI.lastState;
        if (UI.cardsPostPending && cur && !cur.trainer && cur.card_spec
            && data.state && data.state.card_spec
            && sameCardSpecShape(cur.card_spec, data.state.card_spec)) {
          data.state.card_spec = cur.card_spec;
        }
        applyState(data.state);
      } catch (e) {
        showToast(e.message);
        // The optimistic paint shows a spec the server refused: resync,
        // unless a queued post is about to supersede it anyway.
        if (!UI.cardsPostPending && e.message !== GATE_HANDLED) resyncQuietly();
      }
    }
  } finally {
    UI.cardsPostInFlight = false;
  }
}
function sameCardSpecShape(a, b) {
  return ["hero_hole", "flop_a", "flop_b", "turn", "river"].every(
    (k) => Array.isArray(a[k]) && Array.isArray(b[k]) && a[k].length === b[k].length
  );
}
async function postSeats(body) {
  try { const data = await postJSON("/study/seats", body); UI.redo = []; applyState(data.state); }
  catch (e) { showToast(e.message); }
}
// Visually disable the action controls while a state-mutating POST is in
// flight, so a double-click both no-ops (via UI.actionInFlight) AND looks
// disabled. Purely cosmetic — the guard is the flag; this just mirrors it.
// Best-effort: elements may be absent/re-rendered, so guard every lookup.
function setActionsBusy(busy) {
  const ids = [
    "raise-submit",
    "trainer-new-hand-btn", "trainer-repeat-btn",
    "review-next-hand", "review-repeat-hand",
  ];
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el) el.disabled = busy;
  }
  const gate = document.getElementById("gate-buttons");
  if (gate) {
    gate.classList.toggle("busy", busy);
    // Un-busy must not resurrect buttons that were rendered disabled (illegal
    // gate / hero blocked on cards): renderActions records each button's
    // legality in data-legal (review 2026-09-20 F2).
    for (const b of gate.querySelectorAll("button")) {
      b.disabled = busy || b.dataset.legal === "0";
    }
  }
}
// A typed/preset raise total belongs to ONE decision. Drop it (and the input
// focus that keeps render from refreshing the field) on every action and on
// mode/format switches, so a second Enter can't re-submit a stale total at
// the next node (review 2026-09-20 F15).
function resetRaiseEntry() {
  UI.raiseUserSet = false;
  const input = document.getElementById("raise-input");
  if (input && document.activeElement === input) input.blur();
}
// Mode/format switch: the previous context's controls must not stay live
// (or visible) while the new state is still loading (review 2026-09-20 F4).
function clearActionControls() {
  const gate = document.getElementById("gate-buttons");
  if (gate) gate.innerHTML = "";
  const raiseSection = document.getElementById("raise-section");
  if (raiseSection) raiseSection.hidden = true;
  closeBetPresetEditor();
}
async function postAction(body) {
  // In-flight guard: a second click while the first POST is outstanding is
  // a no-op (prevents double-graded trainer decisions / wrong-seat study
  // folds / double raise-submits). `animating` still gates the subsequent
  // opponent frame playback; this gates the network round-trip before it.
  if (UI.actionInFlight) return;
  // No state yet (a mode/format switch is loading) or a state from the other
  // mode: whatever control fired this is stale DOM (review 2026-09-20 F4).
  const cur = UI.lastState;
  if (!cur || !!cur.trainer !== (UI.mode === "trainer")) return;
  if (UI.mode === "trainer" && UI.animating) return;
  resetRaiseEntry();
  if (UI.mode === "trainer") {
    // Hero's street commit + street at this decision: the graded-verdict
    // flash needs them to show raise-TO totals (review 2026-09-20 F11).
    UI.actCommit = {
      handNo: cur.trainer.hand_no,
      commit: actorCommitChips(cur),
      street: cur.street,
    };
    UI.actionInFlight = true;
    setActionsBusy(true);
    // Grading (and EV-loss run-outs) can take a moment: say so (ST-028).
    startWorking("Grading your move…");
    try {
      const data = await postJSON("/trainer/act", body);
      // Once the hero folds, fast-forward past the opponents finishing the
      // hand (unless the user turned that off).
      const instant = body.gate === "fold" && trainerPrefs.ffFold;
      await animateTrainerResponse(data, { instant });
    } catch (e) { showToast(e.message); }
    finally {
      stopWorking();
      UI.actionInFlight = false;
      setActionsBusy(false);
    }
    return;
  }
  UI.actionInFlight = true;
  setActionsBusy(true);
  try {
    const data = await postJSON("/study/action", body);
    UI.redo = [];
    applyState(data.state);
  }
  catch (e) { showToast(e.message); }
  finally { UI.actionInFlight = false; setActionsBusy(false); }
}
async function postTrainer(path, body) {
  try {
    const data = await postJSON(`/trainer/${path}`, body ?? {});
    await animateTrainerResponse(data);
    return true;
  } catch (e) { showToast(e.message); return false; }
}

// Play the per-action frames the trainer returns (one snapshot per
// opponent action), then settle on the authoritative final state. Any
// applyState from elsewhere bumps animSeq and cancels the playback.
async function animateTrainerResponse(data, opts) {
  stopWorking();   // the answer is here: the banner narrates from now on
  const frames = data.frames || [];
  const final = data.state;
  const instant = !!(opts && opts.instant);
  // Requested under another mode/format context: drop it whole — frames
  // included — rather than play an old-format hand over the new state
  // (review 2026-09-20 F4). applyState repeats the check for `final`.
  const ctx = UI.ctxSeq;
  if (!final || STALE_STATES.has(final)) return;
  if (UI.mode !== "trainer" || frames.length === 0) {
    applyState(final);
    return;
  }
  // Fast-forward (e.g. after the hero folds): still flash the graded verdict,
  // but skip watching the opponents play the hand out.
  if (instant) {
    ++UI.animSeq; // cancel any in-flight playback
    if (final.trainer && final.trainer.feedback) renderFeedbackFlash(final, true);
    applyState(final);
    return;
  }
  const seq = ++UI.animSeq;
  UI.animating = true;
  // Frames are painted without going through applyState, so from here the
  // screen no longer matches the dedupe key. Without this reset an applyState
  // of an already-keyed state (a poll that beat this response) aborted the
  // playback and then skipped its own render — stranding the table on
  // "Opponents acting…" (review 2026-09-20 F6).
  UI.lastStateKey = null;
  // A completed hand's `final` is authoritative and MUST settle, even if a
  // stray background applyState (e.g. init's late /trainer/state on a cold
  // server) bumps animSeq mid-animation and trips the abort below. Without
  // this, the abort skipped the settle and left UI.lastState stuck at the
  // pre-action hand (hand_active:true, no review) while the screen showed the
  // run-out — the "post-hand review never appears on the first hand after a
  // promote+refresh" bug. Mid-hand frames (hero to act again) still yield to a
  // genuinely newer state; only the terminal settle overrides the abort.
  const terminalFinal = !!(final.trainer && final.trainer.hand_active === false);
  let aborted = false;
  try {
    // Hero's verdict flashes immediately, while opponents play out.
    if (final.trainer && final.trainer.feedback) {
      renderFeedbackFlash(final, true);
    }
    for (let i = 0; i < frames.length; i++) {
      if (UI.animSeq !== seq) { aborted = true; break; }
      render(frames[i]);
      if (i < frames.length - 1) await sleep(trainerPrefs.animMs);
    }
    if (UI.animSeq !== seq) aborted = true;
  } finally {
    UI.animating = false;
  }
  // The terminal override yields to a mode/format switch: that is a new
  // context, not a stray background state (review 2026-09-20 F4).
  if (ctx !== UI.ctxSeq) return;
  if (!aborted || terminalFinal) applyState(final);
}
// Step the unified review cursor to any decision NODE (hero or villain).
async function trainerReviewGotoNode(node) {
  try {
    const data = await getJSON(`/trainer/review?node=${node}`);
    UI.reviewNode = node;
    applyState(data.state);
  } catch (e) { showToast(e.message); }
}
// Undo = rewind by one, so the undone action can be redone (FEAT-020).
async function postUndo() {
  const s = UI.lastState;
  const n = s && !s.trainer ? (s.history || []).length : 0;
  if (n > 0) await rewindTo(n - 1);
}
async function postReset() {
  try { const data = await postJSON("/study/reset", {}); UI.redo = []; applyState(data.state); }
  catch (e) { showToast(e.message); }
}
async function postConfig(body) {
  try { const data = await postJSON("/study/config", body); applyState(data.state); }
  catch (e) { showToast(e.message); }
}

function applyState(s) {
  if (!s) return;
  // Cross-mode / cross-format clobber guard (review 2026-09-20 F4): ignore a
  // state requested under an older context, and a trainer-shaped state in
  // study mode (or the reverse) — e.g. a slow POST that resolves after the
  // user switched tabs.
  if (STALE_STATES.has(s)) return;
  if (!!s.trainer !== (UI.mode === "trainer")) return;
  UI.animSeq++;  // an authoritative state cancels any frame animation
  UI.lastState = s;
  // A fresh authoritative state ends a what-if card pick; the grid (and the
  // body class that swaps it in) was otherwise left open with nothing
  // selected (review 2026-09-20 F17).
  if (UI.trainerPick) {
    UI.selectedSlot = null;
    cancelTrainerPick();
  }
  if (UI.selectedSlot) {
    const arr = (s.card_spec && s.card_spec[UI.selectedSlot.key]) || [];
    // A format switch shrinks card_spec arrays — drop an out-of-range pick.
    if (UI.selectedSlot.index >= arr.length) {
      UI.selectedSlot = null;
    } else {
      const cur = arr[UI.selectedSlot.index];
      if (cur !== null && cur !== undefined) UI.selectedSlot = null;
    }
  }
  if (window.__wgLiveState) window.__wgLiveState(s);
  // Skip the full re-render when nothing visible changed. Each render
  // does innerHTML = "" on action / board / card-grid containers,
  // which detaches buttons mid-click during high-frequency polling and eats
  // the click event. Most polls return identical state.
  const stateKey = JSON.stringify(s);
  if (stateKey === UI.lastStateKey) return;
  UI.lastStateKey = stateKey;
  render(s);
}

function collectUsedCards(s) {
  const used = new Set();
  for (const k of ["hero_hole", "flop_a", "flop_b", "turn", "river"]) {
    for (const c of s.card_spec[k]) {
      if (c !== null && c !== undefined) used.add(c);
    }
  }
  return used;
}
