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

const SLOT_GEOMETRY = {
  hero_hole: { count: 5, width: 32, height: 44, gap: 4 },
  flop_a: { count: 3, width: 44, height: 60, gap: 6 },
  flop_b: { count: 3, width: 44, height: 60, gap: 6 },
  turn: { count: 2, width: 44, height: 60, gap: 6 },
  river: { count: 2, width: 44, height: 60, gap: 6 },
};

const TABLE_CENTER = { x: 400, y: 260 };
const SEAT_RX = 310;
const SEAT_RY = 170;

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
function chipsToCurrentUnit(chips, state) {
  const bb = chipsToBB(chips, state);
  if (UI.unit === "bb") return bb;
  return bb * (state?.chip_scale?.dollars_per_bb ?? 2);
}
function parseToChips(val, state) {
  const n = parseFloat(val);
  if (!isFinite(n) || n < 0) return null;
  if (UI.unit === "bb") return bbToChips(n, state);
  const dpb = state?.chip_scale?.dollars_per_bb ?? 2;
  return bbToChips(n / dpb, state);
}
function formatUnit(chips, state) {
  const val = chipsToCurrentUnit(chips, state);
  const str = val.toFixed(2);
  return UI.unit === "bb" ? `${str}bb` : `$${str}`;
}
// localStorage can throw (blocked site data, private mode, quota). These run
// at script load, so an unguarded access killed the whole UI
// (review 2026-09-20 F17). Function declarations: hoisted above `UI`.
function lsGet(key) {
  try { return localStorage.getItem(key); } catch (_) { return null; }
}
function lsSet(key, val) {
  try { localStorage.setItem(key, val); } catch (_) { /* storage unavailable */ }
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

function showToast(msg, kind = "error") {
  // Also quiet when a caller wrapped the sentinel ("X failed: __gate…").
  if (typeof msg === "string" && msg.includes(GATE_HANDLED)) return;
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = `toast toast-${kind}`;
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

const UI = {
  unit: "$",
  mode: (() => {
    const q = new URLSearchParams(location.search).get("mode");
    if (q === "trainer" || q === "study") return q;
    return lsGet("plo5bp-mode") === "trainer" ? "trainer" : "study";
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
  reviewDecision: null,   // hero decision index currently shown, or null
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
  return UI.mode === "trainer" ? "/trainer" : "";
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

async function gateIntercept(res) {
  // 409 (both builds): the server has nothing to apply this request to — e.g.
  // POST /trainer/act with no active hand. The client's picture is stale, so
  // resync instead of toasting a raw error (review 2026-09-20).
  if (res.status === 409) {
    resyncQuietly();
    return true;
  }
  if (!window.PLO5BP_PUBLIC) return false;
  if (res.status === 401) {
    showLoginOverlay(true);
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

async function postJSON(url, body) {
  const ctx = UI.ctxSeq;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  noteFreeHands(res);
  if (!res.ok) {
    if (await gateIntercept(res)) throw new Error(GATE_HANDLED);
    let detail = res.statusText;
    try {
      const d = (await res.json()).detail;
      if (typeof d === "string") detail = d;
      else if (Array.isArray(d)) detail = d.map(x => x.msg ?? JSON.stringify(x)).join("; ");
      else if (d !== undefined) detail = JSON.stringify(d);
    } catch (_) {}
    throw new Error(`${res.status} ${detail}`);
  }
  return tagIfStale(ctx, await res.json());
}
async function getJSON(url) {
  const ctx = UI.ctxSeq;
  const res = await fetch(url);
  noteFreeHands(res);
  if (!res.ok) {
    if (await gateIntercept(res)) throw new Error(GATE_HANDLED);
    throw new Error(`${res.status} ${res.statusText}`);
  }
  return tagIfStale(ctx, await res.json());
}

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
  } catch (e) { showToast(e.message); return false; }
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
        const data = await postJSON("/cards", {
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
  try { const data = await postJSON("/seats", body); applyState(data.state); }
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
    try {
      const data = await postJSON("/trainer/act", body);
      // Once the hero folds, fast-forward past the opponents finishing the
      // hand (unless the user turned that off).
      const instant = body.gate === "fold" && trainerPrefs.ffFold;
      await animateTrainerResponse(data, { instant });
    } catch (e) { showToast(e.message); }
    finally { UI.actionInFlight = false; setActionsBusy(false); }
    return;
  }
  UI.actionInFlight = true;
  setActionsBusy(true);
  try { const data = await postJSON("/action", body); applyState(data.state); }
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
async function trainerReviewGoto(decision) {
  try {
    const data = await getJSON(`/trainer/review?decision=${decision}`);
    UI.reviewDecision = decision;
    applyState(data.state);
  } catch (e) { showToast(e.message); }
}
// Step the unified review cursor to any decision NODE (hero or villain).
async function trainerReviewGotoNode(node) {
  try {
    const data = await getJSON(`/trainer/review?node=${node}`);
    UI.reviewNode = node;
    applyState(data.state);
  } catch (e) { showToast(e.message); }
}
async function postUndo() {
  try { const data = await postJSON("/undo", {}); applyState(data.state); }
  catch (e) { showToast(e.message); }
}
async function postReset() {
  try { const data = await postJSON("/reset", {}); applyState(data.state); }
  catch (e) { showToast(e.message); }
}
async function postConfig(body) {
  try { const data = await postJSON("/config", body); applyState(data.state); }
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

// --- Rendering --------------------------------------------------------------

function renderTableChrome(s) {
  document.getElementById("undo-btn").disabled = !s.can_undo;
  const insertIcon = document.getElementById("insert-icon");
  if (s.num_seats >= 6 || s.trainer) {
    insertIcon.setAttribute("hidden", "");
    insertIcon.style.display = "none";
  } else {
    insertIcon.style.display = "";
  }
}

// Each panel paints independently: one throwing sub-render (e.g. a payload
// shape a chart didn't expect) must not abort the rest and leave the table
// half-drawn or the action buttons missing (review 2026-09-20 F5).
const RENDER_STEPS = [
  renderTopBar, renderSeats, renderBoards, renderHeroHole,
  renderHeroHandLabels, renderDealerButton, renderPotLabel, renderActorBanner,
  renderActions, syncPresetPop, renderRecommendation, renderHistory,
  renderCardGrid, renderTrainer, renderTableChrome, syncCardPicker,
];

function render(s) {
  if (!s) return;  // nothing to draw before the first state (review 2026-09-20 F17)
  for (const step of RENDER_STEPS) {
    try { step(s); }
    catch (e) { console.error(`render: ${step.name} failed`, e); }
  }
}

// --- Mobile card-picker sheet ------------------------------------------------
// On small screens the study card grid is not a sidebar; while a slot is
// selected (study entry, or a trainer what-if pick) the #card-grid node is
// re-parented into a bottom sheet. Same node, same render path, same click
// handlers — only the container moves.

const MOBILE_MQ = window.matchMedia("(max-width: 860px)");
function isMobile() { return MOBILE_MQ.matches; }

const SLOT_LABELS = {
  hero_hole: "Hero hole", flop_a: "Board A flop", flop_b: "Board B flop",
  turn: "Turn", river: "River",
};

// --- Format helpers ----------------------------------------------------------
// Card-slot shapes follow the per-format card_spec lengths the server sends
// (PLO5: 5 hole, two flops, paired turn/river; NLH: 2 hole, one flop, single
// turn/river). An empty flop_b marks a single-board format.

function isSingleBoard(s) {
  return !!(s && s.card_spec) && (s.card_spec.flop_b || []).length === 0;
}

const SLOT_LABELS_SINGLE = {
  hero_hole: "Hero hole", flop_a: "Flop", flop_b: "Flop",
  turn: "Turn", river: "River",
};

function slotLabel(s, key) {
  return isSingleBoard(s) ? SLOT_LABELS_SINGLE[key] : SLOT_LABELS[key];
}

// False when the active format is serving the untrained random-init
// placeholder (no checkpoint promoted yet). Study states carry the flag;
// trainer states carry only `format`, so fall back to the /formats payload.
function formatModelLoaded(s) {
  if (!s) return true;
  if (s.format_model_loaded !== undefined && s.format_model_loaded !== null) {
    return !!s.format_model_loaded;
  }
  if (s.format && UI.formats) {
    const f = UI.formats.find((x) => x.id === s.format);
    if (f) return !!f.model_loaded;
  }
  return true;
}

// True when a strategy host other than the per-format PPO checkpoint produced
// this policy (study rec: `is_gto` / `mode`; trainer: the backend badge). The
// format's "no checkpoint promoted" flag doesn't describe such output.
function servedByStrategyHost(rec, s) {
  if (rec && (rec.is_gto === true || (rec.mode && rec.mode !== "ppo"))) return true;
  const b = s && s.trainer && s.trainer.backend;
  return !!(b && b.mode && b.mode !== "ppo");
}

function untrainedBadgeHTML(rec, s) {
  const untrained = (rec && rec.model_loaded === false) || !formatModelLoaded(s);
  if (!untrained || servedByStrategyHost(rec, s)) return "";
  // Name the ACTIVE format — the text used to hardcode one format's name
  // (review 2026-09-20 F13).
  const f = s && s.format && UI.formats ? UI.formats.find((x) => x.id === s.format) : null;
  const name = (s && s.format_label) || (f && f.label) || "";
  return '<div class="rec-untrained">untrained placeholder — no '
    + `${name ? `${escapeHTML(name)} ` : ""}checkpoint promoted yet</div>`;
}

function syncCardPicker(s) {
  const modal = document.getElementById("card-picker-modal");
  if (!modal || !s) return;
  const sel = UI.selectedSlot;
  const wantOpen = isMobile() && !!sel && (!s.trainer || UI.trainerPick);
  const grid = document.getElementById("card-grid");
  const host = document.getElementById("picker-grid-host");
  const home = document.getElementById("card-grid-panel");
  if (wantOpen) {
    if (grid && grid.parentElement !== host) host.appendChild(grid);
    document.getElementById("picker-title").textContent = s.trainer
      ? `What-if: swap ${slotLabel(s, sel.key)} [${sel.index + 1}]`
      : `${slotLabel(s, sel.key)} — slot ${sel.index + 1}`;
    const filled = !s.trainer && s.card_spec[sel.key] &&
      s.card_spec[sel.key][sel.index] !== null &&
      s.card_spec[sel.key][sel.index] !== undefined;
    document.getElementById("picker-clear-btn").style.display = filled ? "" : "none";
    modal.style.display = "";
  } else {
    modal.style.display = "none";
    if (grid && home && grid.parentElement !== home) home.appendChild(grid);
  }
}

function closeCardPicker() {
  UI.selectedSlot = null;
  cancelTrainerPick();
  if (UI.lastState) render(UI.lastState);
}

function setupCardPicker() {
  document.getElementById("picker-close-btn").addEventListener("click", closeCardPicker);
  document.querySelector("#card-picker-modal .picker-backdrop")
    .addEventListener("click", closeCardPicker);
  document.getElementById("picker-clear-btn").addEventListener("click", () => {
    const s = UI.lastState, sel = UI.selectedSlot;
    if (!s || !sel || s.trainer) return;
    s.card_spec[sel.key][sel.index] = null;
    commitLocalCards();  // slot stays selected; picker stays open for the re-place
  });
  MOBILE_MQ.addEventListener("change", () => { if (UI.lastState) render(UI.lastState); });
}

function renderTrainer(s) {
  const reviewPanel = document.getElementById("review-panel");
  if (!s.trainer) {
    reviewPanel.hidden = true;
    hideFeedbackFlash();
    return;
  }
  renderFeedbackFlash(s);
  renderTrainerStats(s);
  renderReviewPanel(s);
}

function renderTopBar(s) {
  document.getElementById("seats-count").textContent = String(s.num_seats);
  document.getElementById("seats-dec").disabled = s.num_seats <= 2;
  document.getElementById("seats-inc").disabled = s.num_seats >= 6;
  document.getElementById("unit-toggle").textContent = UI.unit;
  const dpbInput = document.getElementById("dpb-input");
  if (document.activeElement !== dpbInput) {
    dpbInput.value = s.chip_scale.dollars_per_bb.toFixed(2);
  }
  const anteInput = document.getElementById("ante-input");
  if (document.activeElement !== anteInput) {
    anteInput.value = chipsToCurrentUnit(s.chip_scale.ante_chips, s).toFixed(2);
  }
  const fmtSel = document.getElementById("format-select");
  if (fmtSel && s.format && fmtSel.value !== s.format
      && document.activeElement !== fmtSel) {
    fmtSel.value = s.format;
  }
}

function seatPositions(numSeats, heroSeat) {
  const positions = new Array(numSeats);
  for (let i = 0; i < numSeats; i++) {
    const rel = (i - heroSeat + numSeats) % numSeats;
    // Physical CW from hero: increasing seat index moves visually CW on
    // screen, matching engine's (actor + 1) % n advancement and real-
    // poker action order (SB is one CW step from BTN, etc.). In SVG
    // (Y-down), visual CW corresponds to INCREASING theta from π/2.
    const theta = Math.PI / 2 + (rel * 2 * Math.PI / numSeats);
    const x = TABLE_CENTER.x + SEAT_RX * Math.cos(theta);
    const y = TABLE_CENTER.y + SEAT_RY * Math.sin(theta);
    positions[i] = { x, y, theta };
  }
  return positions;
}

// Committed-bet marker: a poker chip in front of the seat (toward the
// table center) with the amount labeled beside it, GTO-Wizard style.
// `p` is the seat's absolute table position; the returned group uses
// seat-local coordinates (the caller's node is translated to `p`).
function makeBetChip(chips, s, p) {
  const dx = TABLE_CENTER.x - p.x;
  const dy = TABLE_CENTER.y - p.y;
  const len = Math.hypot(dx, dy) || 1;
  const offset = 64; // dealer button sits at 38 on the same ray
  let ax = p.x + (dx / len) * offset;
  let ay = p.y + (dy / len) * offset;

  const label = formatUnit(chips, s);
  // Label goes on the side of the chip facing the table center so it
  // never runs back over the seat plate / dealer button.
  let labelLeft = dx < -10;
  const textW = label.length * 6.6;

  // Keep-out around the pot badge (rect 338-462 × 154-186, padded): the
  // top-center seat's ray lands on it — slide the block sideways past
  // the badge edge, label facing away from it.
  const POT = { x1: 326, y1: 142, x2: 474, y2: 198 };
  const bx1 = labelLeft ? ax - 12 - textW : ax - 10;
  const bx2 = labelLeft ? ax + 10 : ax + 12 + textW;
  if (ay > POT.y1 && ay < POT.y2 && bx2 > POT.x1 && bx1 < POT.x2) {
    if (p.x >= TABLE_CENTER.x) {
      labelLeft = false;
      ax = POT.x2 + 18;
    } else {
      labelLeft = true;
      ax = POT.x1 - 18;
    }
  }

  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
  g.setAttribute("class", "bet-chip");
  g.setAttribute("transform", `translate(${ax - p.x} ${ay - p.y})`);

  const under = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  under.setAttribute("cy", 2.6); under.setAttribute("r", 8);
  under.setAttribute("class", "bet-chip-under");
  g.appendChild(under);

  const base = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  base.setAttribute("r", 8);
  base.setAttribute("class", "bet-chip-base");
  g.appendChild(base);

  const stripes = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  stripes.setAttribute("r", 8);
  stripes.setAttribute("class", "bet-chip-stripes");
  g.appendChild(stripes);

  const inner = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  inner.setAttribute("r", 4.2);
  inner.setAttribute("class", "bet-chip-inner");
  g.appendChild(inner);

  const amount = document.createElementNS("http://www.w3.org/2000/svg", "text");
  amount.setAttribute("x", labelLeft ? -13 : 13);
  amount.setAttribute("y", 4);
  amount.setAttribute("text-anchor", labelLeft ? "end" : "start");
  amount.setAttribute("class", "bet-chip-amount");
  amount.textContent = label;
  g.appendChild(amount);

  return g;
}

function renderSeats(s) {
  const g = document.getElementById("seats");
  g.innerHTML = "";
  const positions = seatPositions(s.num_seats, s.hero_seat);
  for (const seat of s.seats) {
    if (seat.participant === false) continue;
    const p = positions[seat.seat];
    const node = document.createElementNS("http://www.w3.org/2000/svg", "g");
    node.classList.add("seat-node");
    if (seat.is_actor) node.classList.add("actor");
    if (seat.is_hero) node.classList.add("hero");
    if (seat.folded) node.classList.add("folded");
    if (seat.all_in) node.classList.add("all-in");
    if (s.trainer && s.trainer.anim_action && s.trainer.anim_action.seat === seat.seat) {
      node.classList.add("acted");
    }
    node.setAttribute("transform", `translate(${p.x} ${p.y})`);

    const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    bg.setAttribute("x", -46); bg.setAttribute("y", -26);
    bg.setAttribute("width", 92); bg.setAttribute("height", 52);
    bg.setAttribute("rx", 10); bg.setAttribute("class", "seat-bg");
    node.appendChild(bg);

    const pos = document.createElementNS("http://www.w3.org/2000/svg", "text");
    pos.setAttribute("y", -10); pos.setAttribute("class", "seat-position");
    pos.textContent = seat.position + (seat.is_hero ? " (hero)" : "");
    node.appendChild(pos);

    const stack = document.createElementNS("http://www.w3.org/2000/svg", "text");
    stack.setAttribute("y", 6);
    const editable = !seat.all_in && !s.trainer;
    stack.setAttribute("class", editable ? "seat-stack editable" : "seat-stack");
    stack.textContent = seat.all_in ? "all-in" : formatUnit(seat.stack_chips, s);
    if (editable) {
      stack.style.cursor = "pointer";
      stack.addEventListener("click", (e) => {
        e.stopPropagation();
        openStackEditor(seat, s);
      });
    }
    node.appendChild(stack);

    if (seat.committed_this_street_chips > 0) {
      node.appendChild(makeBetChip(seat.committed_this_street_chips, s, p));
    }

    // Trainer: revealed opponent hole cards. Drawn on the OUTSIDE of the
    // table (above the plate for top-half seats, below for bottom-half)
    // so they never collide with the dealer button / bet chips, which
    // live on the inside ray.
    if (seat.hole && !seat.is_hero) {
      const mini = document.createElementNS("http://www.w3.org/2000/svg", "g");
      mini.setAttribute("class", "seat-hole");
      const cw = 24, ch = 33, gap = 3;
      const total = seat.hole.length * cw + (seat.hole.length - 1) * gap;
      // Keep the row on-canvas for far-left/right seats.
      const rowCenterX = Math.max(total / 2 + 4,
        Math.min(800 - total / 2 - 4, p.x)) - p.x;
      const above = p.y < TABLE_CENTER.y;
      const rowY = above ? -(26 + 7 + ch) : 26 + 7;
      const x0 = rowCenterX - total / 2;
      for (let i = 0; i < seat.hole.length; i++) {
        const card = cardToString(seat.hole[i]);
        const x = x0 + i * (cw + gap);
        const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        r.setAttribute("x", x); r.setAttribute("y", rowY);
        r.setAttribute("width", cw); r.setAttribute("height", ch);
        r.setAttribute("rx", 3.5);
        r.setAttribute("class", "seat-hole-card");
        r.style.fill = card.color;
        mini.appendChild(r);
        const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
        t.setAttribute("x", x + cw / 2); t.setAttribute("y", rowY + 15);
        t.setAttribute("text-anchor", "middle");
        t.setAttribute("class", "seat-hole-rank");
        t.textContent = card.rank;
        mini.appendChild(t);
        const gl = document.createElementNS("http://www.w3.org/2000/svg", "text");
        gl.setAttribute("x", x + cw / 2); gl.setAttribute("y", rowY + 28);
        gl.setAttribute("text-anchor", "middle");
        gl.setAttribute("class", "seat-hole-suit");
        gl.textContent = card.glyph;
        mini.appendChild(gl);
      }
      node.appendChild(mini);
    }

    if (!seat.is_hero && s.num_seats > 2 && !s.trainer) {
      const rm = document.createElementNS("http://www.w3.org/2000/svg", "g");
      rm.setAttribute("class", "seat-remove");
      rm.setAttribute("transform", "translate(38 -18)");
      const rmBg = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      rmBg.setAttribute("r", 8);
      rmBg.setAttribute("class", "seat-remove-bg");
      rm.appendChild(rmBg);
      const rmX = document.createElementNS("http://www.w3.org/2000/svg", "text");
      rmX.setAttribute("text-anchor", "middle");
      rmX.setAttribute("dy", 4);
      rmX.setAttribute("class", "seat-remove-x");
      rmX.textContent = "\u00d7";
      rm.appendChild(rmX);
      rm.addEventListener("click", (e) => { e.stopPropagation(); removeSeat(seat.seat, s); });
      node.appendChild(rm);
    }

    g.appendChild(node);
  }
}

function removeSeat(idx, s) {
  if (idx === s.hero_seat) return;
  if (s.num_seats <= 2) return;
  const stacks = s.seats.map(x => x.stack_chips);
  stacks.splice(idx, 1);
  const newN = s.num_seats - 1;
  let newButton = s.button_seat;
  if (newButton === idx) newButton = idx % newN;
  else if (newButton > idx) newButton = newButton - 1;
  postSeats({
    num_seats: newN,
    button_seat: newButton,
    starting_stacks: stacks,
  });
}

function insertSeat(a, s) {
  if (s.num_seats >= 6) return;
  const N = s.num_seats;
  const aIsLast = (a + 1) % N === 0;
  const ins = aIsLast ? N : a + 1;
  const stacks = s.seats.map(x => x.stack_chips);
  const defaultChips = stacks[a];
  stacks.splice(ins, 0, defaultChips);
  let newButton = s.button_seat;
  if (!aIsLast && newButton >= ins) newButton = newButton + 1;
  postSeats({
    num_seats: N + 1,
    button_seat: newButton,
    starting_stacks: stacks,
  });
}

function openStackEditor(seat, s) {
  const existing = document.getElementById("stack-edit-input");
  if (existing) existing.remove();

  const svg = document.getElementById("table-svg");
  const wrap = document.getElementById("table-wrap");
  const positions = seatPositions(s.num_seats, s.hero_seat);
  const p = positions[seat.seat];
  const pt = svg.createSVGPoint();
  pt.x = p.x; pt.y = p.y + 6;
  const screen = pt.matrixTransform(svg.getScreenCTM());
  const wrapBox = wrap.getBoundingClientRect();

  const input = document.createElement("input");
  input.type = "number";
  input.step = UI.unit === "bb" ? "0.1" : "1";
  input.min = "0";
  input.id = "stack-edit-input";
  input.className = "stack-edit-input";
  input.value = chipsToCurrentUnit(seat.stack_chips, s).toFixed(2);
  const width = 80;
  input.style.width = `${width}px`;
  input.style.left = `${screen.x - wrapBox.left - width / 2}px`;
  input.style.top = `${screen.y - wrapBox.top - 10}px`;
  wrap.style.position = "relative";
  wrap.appendChild(input);
  input.focus();
  input.select();

  let committed = false;
  const commit = async () => {
    if (committed) return;
    committed = true;
    const chips = parseToChips(input.value, s);
    input.remove();
    if (chips === null || chips <= 0) {
      showToast("Invalid stack value");
      return;
    }
    // Build from the LATEST state, not the one captured when the editor
    // opened — re-sending that snapshot would roll back stacks that changed
    // in the meantime (review 2026-09-20 H6, client half).
    const live = UI.lastState && !UI.lastState.trainer ? UI.lastState : s;
    if (seat.seat >= live.seats.length) return;  // that seat is gone
    const stacks = live.seats.map(x => x.stack_chips);
    stacks[seat.seat] = chips;
    await postConfig({ starting_stacks: stacks });
  };
  const cancel = () => {
    if (committed) return;
    committed = true;
    input.remove();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); commit(); }
    else if (e.key === "Escape") { e.preventDefault(); cancel(); }
  });
  input.addEventListener("blur", cancel);
}

function renderBoards(s) {
  const boardA = document.getElementById("board-a");
  const boardB = document.getElementById("board-b");
  boardA.innerHTML = "";
  boardB.innerHTML = "";
  // Single-board formats (flop_b === []) draw one vertically-centered row
  // and skip board B entirely; the double-board layout is unchanged.
  const single = isSingleBoard(s);
  boardA.setAttribute("transform", single ? "translate(280 246)" : "translate(280 207)");

  renderSlotStrip(boardA, "flop_a", s.card_spec.flop_a, s, 0);
  if (!single) renderSlotStrip(boardB, "flop_b", s.card_spec.flop_b, s, 0);

  const turnX0 = 3 * (44 + 6);
  renderSingleSlot(boardA, "turn", 0, s.card_spec.turn[0], s, turnX0);
  if (s.card_spec.turn.length > 1) {
    renderSingleSlot(boardB, "turn", 1, s.card_spec.turn[1], s, turnX0);
  }

  const riverX0 = 4 * (44 + 6);
  renderSingleSlot(boardA, "river", 0, s.card_spec.river[0], s, riverX0);
  if (s.card_spec.river.length > 1) {
    renderSingleSlot(boardB, "river", 1, s.card_spec.river[1], s, riverX0);
  }
}

function renderSlotStrip(parent, key, values, s, x0) {
  const geom = SLOT_GEOMETRY[key];
  for (let i = 0; i < values.length; i++) {
    const x = x0 + i * (geom.width + geom.gap);
    renderSlotRect(parent, key, i, values[i], s, x, 0, geom.width, geom.height);
  }
}
function renderSingleSlot(parent, key, index, value, s, x) {
  const geom = SLOT_GEOMETRY[key];
  renderSlotRect(parent, key, index, value, s, x, 0, geom.width, geom.height);
}

function renderSlotRect(parent, key, index, value, s, x, y, w, h) {
  const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  rect.setAttribute("x", x); rect.setAttribute("y", y);
  rect.setAttribute("width", w); rect.setAttribute("height", h);
  rect.setAttribute("rx", 5);
  let cls = "slot-rect" + (value === null ? " empty" : "");
  const isSelected = UI.selectedSlot && UI.selectedSlot.key === key && UI.selectedSlot.index === index;
  if (isSelected) cls += " selected";
  rect.setAttribute("class", cls);
  rect.addEventListener("click", () => onSlotClick(key, index));
  rect.addEventListener("dblclick", (e) => { e.preventDefault(); onSlotDoubleClick(key, index); });
  parent.appendChild(rect);

  if (value !== null && value !== undefined) {
    const card = cardToString(value);
    rect.style.fill = card.color;

    const smallRankSize = Math.max(8, Math.round(h * 0.227));
    const smallSuitSize = Math.max(8, Math.round(h * 0.25));
    const bigRankSize = Math.max(14, Math.round(h * 0.50));
    const pad = Math.max(3, Math.round(w * 0.12));

    const tlRank = document.createElementNS("http://www.w3.org/2000/svg", "text");
    tlRank.setAttribute("x", x + pad);
    tlRank.setAttribute("y", y + smallRankSize + 2);
    tlRank.setAttribute("class", "slot-corner-rank");
    tlRank.setAttribute("font-size", smallRankSize);
    tlRank.setAttribute("fill", "#ffffff");
    tlRank.textContent = card.rank;
    parent.appendChild(tlRank);

    const tlSuit = document.createElementNS("http://www.w3.org/2000/svg", "text");
    tlSuit.setAttribute("x", x + pad);
    tlSuit.setAttribute("y", y + smallRankSize + smallSuitSize + 3);
    tlSuit.setAttribute("class", "slot-corner-suit");
    tlSuit.setAttribute("font-size", smallSuitSize);
    tlSuit.setAttribute("fill", "#ffffff");
    tlSuit.textContent = card.glyph;
    parent.appendChild(tlSuit);

    const brRank = document.createElementNS("http://www.w3.org/2000/svg", "text");
    brRank.setAttribute("x", x + w - pad);
    brRank.setAttribute("y", y + h - Math.max(3, Math.round(h * 0.08)));
    brRank.setAttribute("text-anchor", "end");
    brRank.setAttribute("class", "slot-corner-bigrank");
    brRank.setAttribute("font-size", bigRankSize);
    brRank.setAttribute("fill", "#ffffff");
    brRank.textContent = card.rank;
    parent.appendChild(brRank);
  }

  const modified = (s.modified_cards || []).some(
    (m) => m.slot_key === key && m.index === index
  );
  if (modified) {
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("cx", x + w - 4);
    dot.setAttribute("cy", y + 4);
    dot.setAttribute("r", 3);
    dot.setAttribute("class", "slot-modified-dot");
    parent.appendChild(dot);
  }
}

function renderHeroHole(s) {
  const g = document.getElementById("hero-hole");
  g.innerHTML = "";
  const geom = SLOT_GEOMETRY.hero_hole;
  const count = s.card_spec.hero_hole.length;  // per-format: 5 (PLO) / 2 (NLH)
  const totalWidth = count * geom.width + (count - 1) * geom.gap;
  const x0 = -totalWidth / 2;
  for (let i = 0; i < count; i++) {
    const x = x0 + i * (geom.width + geom.gap);
    renderSlotRect(g, "hero_hole", i, s.card_spec.hero_hole[i], s, x, 0, geom.width, geom.height);
  }
}

// Two made-hand labels ("#1 a pair of 8s" / "#2 ...") in the
// gap between the hero plate (y~430) and the hero cards (y~500), so a
// stealth set/straight is hard to miss while deciding. Both boards are
// always dealt together postflop, so normally both rows render.
function renderHeroHandLabels(s) {
  const g = document.getElementById("hero-hand-labels");
  g.innerHTML = "";
  const desc = s.hero_hand_desc || [];
  const rows = [];
  if (isSingleBoard(s)) {
    // One board — a single unnumbered label (descB is always null).
    if (desc[0]) rows.push([null, desc[0]]);
  } else {
    if (desc[0]) rows.push(["#1", desc[0]]);
    if (desc[1]) rows.push(["#2", desc[1]]);
  }
  if (!rows.length) return;
  const svgNS = "http://www.w3.org/2000/svg";
  rows.forEach(([tag, text], i) => {
    const t = document.createElementNS(svgNS, "text");
    t.setAttribute("x", 400);
    t.setAttribute("y", 472 + i * 16);
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("class", "hero-hand-label");
    if (tag) {
      const tg = document.createElementNS(svgNS, "tspan");
      tg.setAttribute("class", "hhl-tag");
      tg.textContent = tag + " ";
      t.appendChild(tg);
    }
    t.appendChild(document.createTextNode(text));
    g.appendChild(t);
  });
}

function renderDealerButton(s) {
  const node = document.getElementById("dealer-button");
  if (UI.draggingButton) return;
  const positions = seatPositions(s.num_seats, s.hero_seat);
  const p = positions[s.button_seat];
  const dx = TABLE_CENTER.x - p.x;
  const dy = TABLE_CENTER.y - p.y;
  const len = Math.hypot(dx, dy) || 1;
  const offset = 38;
  const bx = p.x + (dx / len) * offset;
  const by = p.y + (dy / len) * offset;
  node.setAttribute("transform", `translate(${bx} ${by})`);
  node.dataset.seat = String(s.button_seat);
}

function renderPotLabel(s) {
  // Top badge = grand Total Pot (incl. this street's live bets); the
  // badge below the boards = settled Pot (gathered from prior streets).
  document.getElementById("pot-label").textContent = `Total Pot ${formatUnit(s.pot_chips, s)}`;
  const settled = s.settled_pot_chips ?? s.pot_chips;
  document.getElementById("pot-settled-label").textContent = `Pot ${formatUnit(settled, s)}`;
}

// --- Trainer review labels ---------------------------------------------------
// Review payloads carry raise-BY deltas plus bb-only server strings, while the
// table / raise input show raise-TO totals in the active unit. Every review
// label is rebuilt with gateActionLabel so the two agree
// (review 2026-09-20 F11).

// Street commits around every node of the reviewed hand, from review.nodes
// (one row per action: seat, street, gate, raise-BY chips). A call's chips are
// not in the row, so a call is taken as matching the street's bet level.
function reviewNodeCommits(s, rv) {
  return streetCommitWalk(s, (rv && rv.nodes) || [], (nd, before, level) => (
    nd.actual_gate === "raise" ? nd.actual_chips
      : nd.actual_gate === "check_call" ? level - before : 0
  ));
}

// Actor's street commit before review node `idx` acted; null = unknown.
// Order: a server-provided `actor_commit_chips`, the replayed node state
// itself (exact, straight from the engine), then the walk over review.nodes.
function reviewCommitBefore(s, rv, idx, seat, payload, walk) {
  if (payload && typeof payload.actor_commit_chips === "number") {
    return payload.actor_commit_chips;
  }
  if (s.actor === seat && (s.history || []).length === idx) return actorCommitChips(s);
  const nd = rv && rv.nodes ? rv.nodes[idx] : null;
  if (walk && walk[idx] && nd && nd.seat === seat) return walk[idx].before;
  return null;
}

// A review node with no graded decision behind it: every villain node, and a
// hero moot auto-check (all-in run-out), which the backend sends with
// `category: null` (+ `auto: true`). Such nodes have no score / marks /
// user_label and no what-if (review 2026-09-20 F12).
function reviewNodeUngraded(nc) {
  return !!nc && (!nc.is_hero || nc.auto === true
    || nc.category === null || nc.category === undefined);
}

// Same, for the node detail payload (`node_current`, or the legacy hero-only
// `current`, which carries neither seat nor node index).
function reviewCurCommit(s, rv, cur, walk) {
  const seat = cur.seat !== null && cur.seat !== undefined ? cur.seat : s.hero_seat;
  const idx = cur.node_idx !== null && cur.node_idx !== undefined ? cur.node_idx : rv.node;
  return reviewCommitBefore(s, rv, idx, seat, cur, walk);
}

// Frames come from every engine step between hero decisions, and that
// includes HERO's own engine-forced checks at a moot node (all-in run-out):
// the backend tags those `is_hero` + `auto`. They are not opponent actions.
function animIsHeroAuto(a) {
  return !!a && a.is_hero === true && a.auto === true;
}

function animActionText(s) {
  const a = s.trainer.anim_action;
  const seat = s.seats[a.seat];
  if (animIsHeroAuto(a)) return "You check (automatic)";
  if (a.gate === "fold") return `${a.position} folds`;
  if (a.gate === "check_call") {
    return a.to_call > 0 ? `${a.position} calls` : `${a.position} checks`;
  }
  const committed = seat ? seat.committed_this_street_chips : a.chips;
  const verb = seat && seat.all_in ? "is all-in"
    : aggVerb(a.to_call, a.street) === "Raise" ? "raises to" : "bets";
  const amt = committed > 0 ? ` ${formatUnit(committed, s)}` : "";
  return `${a.position} ${verb}${amt}`;
}

function renderActorBanner(s) {
  const banner = document.getElementById("actor-banner");
  // Trainer animation frame: narrate the action that just landed — an
  // opponent's, or the hero's own automatic check (styled as the hero's).
  if (s.trainer && s.trainer.anim_action) {
    banner.hidden = false;
    banner.classList.toggle("hero", animIsHeroAuto(s.trainer.anim_action));
    banner.textContent = animActionText(s);
    return;
  }
  // Trainer review: the state is a mid-hand reconstruction, not a live turn.
  if (s.trainer && !s.trainer.hand_active && s.trainer.review) {
    const rv = s.trainer.review;
    const nc = rv.node_current;
    banner.hidden = false;
    banner.classList.toggle("hero", nc ? nc.is_hero : true);
    const walk = reviewNodeCommits(s, rv);
    if (nc) {
      const who = `${nc.position}${nc.is_hero ? " (hero)" : ""}`;
      const did = nc.actual_gate
        ? gateActionLabel(nc.actual_gate, nc.actual_chips, nc.to_call_chips, s,
                          reviewCurCommit(s, rv, nc, walk), nc.street)
        : nc.actual_label;
      // A hero moot auto-check (ungraded: category null) was never a choice.
      const what = !nc.is_hero ? `acted ${did}`
        : reviewNodeUngraded(nc) ? "auto-check (betting closed)" : `you chose ${did}`;
      banner.textContent =
        `Node ${rv.node + 1} / ${rv.num_nodes} — ${nc.street} · ${who} ${what}`;
    } else if (rv.current) {
      const c = rv.current;
      const did = c.user_gate
        ? gateActionLabel(c.user_gate, c.user_chips, c.to_call_chips, s,
                          reviewCurCommit(s, rv, c, walk), c.street)
        : c.user_label;
      banner.textContent =
        `Reviewing decision ${rv.decision + 1} / ${rv.num_decisions} — ${c.street}` +
        ` · you chose ${did}`;
    } else {
      banner.textContent = "Reviewing hand";
    }
    return;
  }
  if (s.actor === null || s.actor === undefined) {
    banner.hidden = true;
    return;
  }
  const seat = s.seats[s.actor];
  const isHero = s.actor === s.hero_seat;
  banner.hidden = false;
  banner.classList.toggle("hero", isHero);
  if (isHero) {
    banner.textContent = `Hero to act (${seat.position})`;
  } else {
    banner.textContent = `Seat ${s.actor} to act — ${seat.position}`;
  }
}

function renderActions(s) {
  const gate = document.getElementById("gate-buttons");
  gate.innerHTML = "";
  const raiseSection = document.getElementById("raise-section");
  const terminalPane = document.getElementById("terminal-pane");

  if (s.terminal) {
    raiseSection.hidden = true;
    terminalPane.hidden = false;
    document.getElementById("terminal-message").textContent = s.terminal_message ?? "Hand complete.";
    return;
  }
  terminalPane.hidden = true;

  // Trainer review reconstruction: the hand is over; this state is a
  // replayed decision node. Show the choice, don't allow acting.
  if (s.trainer && !s.trainer.hand_active) {
    raiseSection.hidden = true;
    gate.innerHTML = '<p class="muted">Reviewing — actions disabled. Use Next Hand to continue.</p>';
    return;
  }

  // Trainer animation frame: an opponent is acting.
  if (s.trainer && s.actor !== null && s.actor !== undefined && s.actor !== s.hero_seat) {
    raiseSection.hidden = true;
    gate.innerHTML = '<p class="muted">Opponents acting…</p>';
    return;
  }

  if (s.actor === null || s.actor === undefined) {
    raiseSection.hidden = true;
    gate.innerHTML = '<p class="muted">Hand complete or awaiting cards.</p>';
    return;
  }

  const isHero = s.actor === s.hero_seat;
  const holeCount = s.card_spec ? s.card_spec.hero_hole.length : 5;
  const BLOCK_TOOLTIPS = {
    hole:  `Place hero's ${holeCount} hole cards to act`,
    flop:  "Place the flop cards to act",
    turn:  "Place the turn cards to act",
    river: "Place the river cards to act",
  };
  const heroBlocked = isHero && s.hero_blocking_reason != null;
  const tooltip = heroBlocked ? BLOCK_TOOLTIPS[s.hero_blocking_reason] : "";

  const mkBtn = (label, gateName, enabled, cls) => {
    const b = document.createElement("button");
    b.textContent = label;
    if (cls) b.classList.add(cls);
    // data-legal lets setActionsBusy(false) restore exactly this state; a
    // button painted while a POST is outstanding starts out busy-disabled
    // (review 2026-09-20 F2).
    b.dataset.legal = enabled ? "1" : "0";
    b.disabled = !enabled || UI.actionInFlight;
    if (tooltip) b.title = tooltip;
    b.addEventListener("click", () => postAction({ gate: gateName }));
    return b;
  };

  const actorSeat = s.seats[s.actor];
  const toCallLabel = s.to_call_chips > 0 ? `Call ${formatUnit(s.to_call_chips, s)}` : "Check";

  gate.appendChild(mkBtn("Fold", "fold", s.legal.fold && !heroBlocked, "fold"));
  gate.appendChild(mkBtn(toCallLabel, "check_call", s.legal.check_call && !heroBlocked, "call"));

  if (s.legal.raise && !heroBlocked) {
    raiseSection.hidden = false;
    renderRaiseSection(s, actorSeat);
  } else {
    raiseSection.hidden = true;
  }
}

function renderRaiseSection(s, actorSeat) {
  const minChips = s.raise_bounds.min_chips;
  const maxChips = s.raise_bounds.max_chips;
  // Engine bounds are raise-BY deltas; the UI shows raise-TO totals.
  // total = delta + ac. Arithmetic stays in chips; format only at the edge.
  const ac = actorCommitChips(s);
  const input = document.getElementById("raise-input");
  const inputRow = input.parentElement;
  const unitLabel = document.getElementById("raise-unit");
  const boundsLabel = document.getElementById("raise-bounds-label");
  const submit = document.getElementById("raise-submit");
  const shortcuts = document.getElementById("raise-shortcuts");
  unitLabel.textContent = UI.unit === "bb" ? "bb" : "$";

  // Degenerate range (min == max): only one legal raise amount — render as
  // a single button. Hides the slider/input/shortcuts. Two regimes:
  //  - actor-is-short: maxChips == actor's full stack → "All-in $X"
  //  - cover-short: actor is deep, max collapses to short opp's reach →
  //    "Bet/Raise $X" (actor still has chips left)
  if (minChips === maxChips && maxChips > 0) {
    boundsLabel.textContent = "";
    inputRow.querySelectorAll("input, .raise-unit").forEach(el => el.style.display = "none");
    shortcuts.innerHTML = "";
    // maxChips is a raise-BY delta → all-in iff it takes the remaining stack
    // (review 2026-09-20 F8).
    const isAllIn = isAllInDelta(actorSeat, maxChips);
    const verb = aggVerb(s.to_call_chips, s.street);
    // Display the raise-TO total (delta + ac); still post the DELTA.
    submit.textContent = isAllIn
      ? `All-in ${formatUnit(maxChips + ac, s)}`
      : `${verb} ${formatUnit(maxChips + ac, s)}`;
    submit.onclick = () => postAction({ gate: "raise", chips: maxChips });
    return;
  }
  inputRow.querySelectorAll("input, .raise-unit").forEach(el => el.style.display = "");
  submit.textContent = "Raise";

  // Bounds shown as raise-TO totals (delta + ac).
  const minDisp = chipsToCurrentUnit(minChips + ac, s);
  const maxDisp = chipsToCurrentUnit(maxChips + ac, s);
  boundsLabel.textContent = `Raise to: ${minDisp.toFixed(2)} – ${maxDisp.toFixed(2)} ${UI.unit === "bb" ? "bb" : "$"}`;
  input.min = minDisp.toFixed(2);
  input.max = maxDisp.toFixed(2);
  input.step = UI.unit === "bb" ? "0.1" : "0.01";

  // A typed/preset total belongs to ONE decision node. A new node (hand,
  // street, history length or actor changed) drops it even while the input
  // still has focus — a focused field used to keep its old number, so a
  // second Enter re-submitted the previous street's total
  // (review 2026-09-20 F15).
  const nodeKey = `${s.trainer ? s.trainer.hand_no : "study"}|${s.street}|`
    + `${(s.history || []).length}|${s.actor}`;
  const newNode = UI.raiseNodeKey !== nodeKey;
  if (newNode || UI.raiseLastActor !== s.actor) {
    UI.raiseUserSet = false;
    UI.raiseLastActor = s.actor;
    UI.raiseNodeKey = nodeKey;
  }

  const userTyping = document.activeElement === input && !newNode;
  const userActive = userTyping || UI.raiseUserSet;
  if (!userActive) {
    let preset = minChips;
    const rec = s.recommendation;
    if (rec && rec.gate === "raise" && rec.chips !== null && rec.chips !== undefined) {
      preset = Math.max(minChips, Math.min(maxChips, rec.chips));
    }
    // preset is a DELTA; display it as a raise-TO total.
    input.value = chipsToCurrentUnit(preset + ac, s).toFixed(2);
  } else if (!userTyping) {
    // The input holds a TOTAL; clamp in delta space, redisplay as total.
    const curTotal = parseToChips(input.value, s);
    if (curTotal !== null) {
      const curDelta = curTotal - ac;
      if (curDelta < minChips || curDelta > maxChips) {
        const clampedDelta = Math.max(minChips, Math.min(maxChips, curDelta));
        input.value = chipsToCurrentUnit(clampedDelta + ac, s).toFixed(2);
      }
    }
  }

  submit.onclick = () => {
    // The user typed a raise-TO total; convert to the engine's raise-BY delta.
    const total = parseToChips(input.value, s);
    if (total === null) { showToast("invalid raise amount"); return; }
    const delta = total - ac;
    const clamped = Math.max(minChips, Math.min(maxChips, delta));
    UI.raiseUserSet = false;
    postAction({ gate: "raise", chips: clamped });
  };

  shortcuts.innerHTML = "";
  // Returns the UNCLAMPED raise-TO total (matches the now-total-space input). A
  // pot-fraction bet means: call (toCall) then raise BY mult*(pot+toCall) on
  // top, so the final commitment is ac + toCall + extra. (The input is total
  // and submit subtracts ac, so the commitment equals this exactly — this also
  // fixes the old over-commit where a total was posted as a delta.)
  const potSize = (mult) => {
    const toCall = s.to_call_chips;
    const extra = Math.round(mult * (s.pot_chips + toCall));
    return ac + toCall + extra;
  };
  const potLimit = isPotLimit(s);
  // "b33" has always meant a THIRD of pot (mult 1/3, not 0.33) — keep exact.
  const presetMult = (n) => (n === 33 ? 1 / 3 : n / 100);
  // A chip's label must describe the chips it prefills: a preset that the
  // legal range clamps is labelled by where it lands — "min", or "all-in"
  // ("max" when the cap is a cover-short clamp and the actor keeps chips
  // behind) — never by its nominal pot-%. Presets collapsing onto the same
  // amount dedupe to one chip (review 2026-09-20 F9).
  const minTotal = minChips + ac;
  const maxTotal = maxChips + ac;
  const maxIsAllIn = isAllInDelta(actorSeat, maxChips);
  const items = [];
  const seenTotals = new Set();
  const addItem = (label, chips, cls) => {
    if (seenTotals.has(chips)) return;
    seenTotals.add(chips);
    items.push({ label, chips, cls });
  };
  for (const n of betPresets(potLimit)) {
    const raw = potSize(presetMult(n));
    if (raw <= minTotal) addItem("min", minTotal);
    else if (raw > maxTotal || (raw === maxTotal && maxIsAllIn)) {
      addItem(maxIsAllIn ? "all-in" : "max", maxTotal, "raise-shortcut-allin");
    } else addItem(presetChipLabel(n, potLimit), raw);
  }
  if (!potLimit) {
    // No-limit only: an all-in prefill chip (in pot-limit the pot chip IS the
    // cap). Prefills the max raise-TO total; the user still clicks Raise.
    addItem(maxIsAllIn ? "all-in" : "max", maxTotal, "raise-shortcut-allin");
  }
  for (const it of items) {
    const b = document.createElement("button");
    b.className = "raise-shortcut" + (it.cls ? ` ${it.cls}` : "");
    b.textContent = it.label;
    b.addEventListener("click", () => {
      input.value = chipsToCurrentUnit(it.chips, s).toFixed(2);
      UI.raiseUserSet = true;
    });
    shortcuts.appendChild(b);
  }
  const plus = document.createElement("button");
  plus.id = "preset-edit-btn";
  plus.type = "button";
  plus.className = "raise-shortcut raise-shortcut-edit";
  plus.textContent = "+";
  plus.title = "Edit bet-size presets";
  plus.addEventListener("click", () => toggleBetPresetEditor(plus, potLimit));
  shortcuts.appendChild(plus);
}

// --- Bet-size preset chips ---------------------------------------------------
// The raise shortcuts row is driven by a per-cap-class preset list (numbers =
// % of pot after call, the potSize formula above). Pot-limit and no-limit
// formats keep SEPARATE localStorage lists so preferences don't collide; the
// "+" chip opens a popover editor. The all-in chip (NL) is not part of the
// list — always present in NL, never in PL.

const BET_PRESET_DEFAULTS = { pl: [25, 33, 50, 75, 100], nl: [25, 33, 50, 75, 100, 150] };
const BET_PRESET_CAP = { pl: 100, nl: 1000 };

// Cap class of the active format, from state.format + the /formats cache
// (nothing hardcodes format ids). Unknown → pot-limit, the legacy behavior.
function isPotLimit(s) {
  if (s && s.format && UI.formats) {
    const f = UI.formats.find((x) => x.id === s.format);
    if (f && f.pot_limit !== undefined && f.pot_limit !== null) return !!f.pot_limit;
  }
  return true;
}

function betPresetKey(potLimit) {
  return potLimit ? "plo5bp-bet-presets-pl" : "plo5bp-bet-presets-nl";
}

// Load the preset list: numbers > 0 within the cap, one decimal place,
// deduped, ascending. Malformed storage falls back to the defaults; a valid
// but emptied list stays empty (the user removed every preset on purpose).
function betPresets(potLimit) {
  const cls = potLimit ? "pl" : "nl";
  let raw = null;
  try { raw = JSON.parse(localStorage.getItem(betPresetKey(potLimit))); }
  catch (_) { raw = null; }
  if (!Array.isArray(raw)) return BET_PRESET_DEFAULTS[cls].slice();
  const clean = [...new Set(
    raw.filter((n) => typeof n === "number" && isFinite(n)
                      && n > 0 && n <= BET_PRESET_CAP[cls])
       .map((n) => Math.round(n * 10) / 10),
  )].sort((a, b) => a - b);
  return clean;
}

function saveBetPresets(potLimit, list) {
  try { localStorage.setItem(betPresetKey(potLimit), JSON.stringify(list)); }
  catch (_) { /* storage unavailable — presets stay session-default */ }
}

// Chip label: "b25", "b33.3", … — except 100% of pot in a pot-limit format,
// which IS the cap and keeps its historical "pot" label.
function presetChipLabel(n, potLimit) {
  return potLimit && n === 100 ? "pot" : `b${n}`;
}

// --- Preset editor popover ---------------------------------------------------

function closeBetPresetEditor() {
  const pop = document.getElementById("preset-pop");
  if (pop) pop.remove();
  document.removeEventListener("pointerdown", onPresetPopOutside, true);
  document.removeEventListener("keydown", onPresetPopKey, true);
}

function onPresetPopOutside(e) {
  const pop = document.getElementById("preset-pop");
  if (!pop || pop.contains(e.target)) return;
  const plus = document.getElementById("preset-edit-btn");
  if (plus && plus.contains(e.target)) return;  // the "+" click toggles
  closeBetPresetEditor();
}

function onPresetPopKey(e) {
  if (e.key !== "Escape") return;
  e.preventDefault();
  e.stopPropagation();
  closeBetPresetEditor();
}

function presetPopHint(msg) {
  const el = document.getElementById("preset-hint");
  if (!el) return;
  el.textContent = msg || "";
  el.classList.toggle("error", !!msg);
}

function renderPresetPopContent(pop) {
  const potLimit = pop.dataset.cap === "pl";
  const list = betPresets(potLimit);
  const pills = pop.querySelector(".preset-pill-list");
  pills.innerHTML = "";
  for (const n of list) {
    const pill = document.createElement("span");
    pill.className = "preset-pill";
    pill.appendChild(document.createTextNode(presetChipLabel(n, potLimit)));
    const x = document.createElement("button");
    x.type = "button";
    x.className = "preset-pill-x";
    x.setAttribute("aria-label", `Remove b${n}`);
    x.textContent = "×";
    x.addEventListener("click", () => {
      saveBetPresets(potLimit, betPresets(potLimit).filter((v) => v !== n));
      presetPopHint("");
      if (UI.lastState) render(UI.lastState);  // refreshes row + this popover
    });
    pill.appendChild(x);
    pills.appendChild(pill);
  }
  if (!list.length) {
    const none = document.createElement("span");
    none.className = "muted";
    none.textContent = "No presets";
    pills.appendChild(none);
  }
}

function presetPopAdd(pop) {
  const potLimit = pop.dataset.cap === "pl";
  const inp = document.getElementById("preset-add-input");
  const v = parseFloat(inp.value);
  if (!isFinite(v) || v <= 0) { presetPopHint("Enter a size above 0"); return; }
  const n = Math.round(v * 10) / 10;  // up to one decimal place
  if (potLimit && n > 100) { presetPopHint("pot-limit caps at pot (b100)"); return; }
  if (!potLimit && n > 1000) { presetPopHint("no-limit presets cap at b1000"); return; }
  const list = betPresets(potLimit);
  if (list.includes(n)) { presetPopHint(`b${n} is already a preset`); return; }
  list.push(n);
  list.sort((a, b) => a - b);
  saveBetPresets(potLimit, list);
  inp.value = "";
  presetPopHint("");
  if (UI.lastState) render(UI.lastState);
}

// Fixed-position near the "+" chip, clamped into the viewport (mobile-safe);
// flips above the chip when there is no room below.
function positionPresetPop(pop, plus) {
  const r = plus.getBoundingClientRect();
  const popW = pop.offsetWidth, popH = pop.offsetHeight;
  const left = Math.max(8, Math.min(r.left, window.innerWidth - popW - 8));
  let top = r.bottom + 6;
  if (top + popH > window.innerHeight - 8) top = Math.max(8, r.top - popH - 6);
  pop.style.left = `${left}px`;
  pop.style.top = `${top}px`;
}

function toggleBetPresetEditor(plus, potLimit) {
  if (document.getElementById("preset-pop")) { closeBetPresetEditor(); return; }
  const pop = document.createElement("div");
  pop.id = "preset-pop";
  pop.className = "preset-pop";
  pop.dataset.cap = potLimit ? "pl" : "nl";
  pop.innerHTML = `
    <div class="preset-pop-title">Bet-size presets <span class="muted">% of pot</span></div>
    <div class="preset-pill-list"></div>
    <div class="preset-add-row">
      <input id="preset-add-input" type="number" min="0" step="0.1"
             max="${potLimit ? 100 : 1000}" placeholder="% of pot" />
      <button id="preset-add-btn" type="button">Add</button>
    </div>
    <div id="preset-hint" class="preset-hint muted"></div>
    <button id="preset-reset-btn" class="preset-reset" type="button">Reset to defaults</button>
  `;
  document.body.appendChild(pop);
  renderPresetPopContent(pop);
  positionPresetPop(pop, plus);
  document.getElementById("preset-add-btn").addEventListener("click", () => presetPopAdd(pop));
  document.getElementById("preset-add-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); presetPopAdd(pop); }
  });
  document.getElementById("preset-reset-btn").addEventListener("click", () => {
    try { localStorage.removeItem(betPresetKey(pop.dataset.cap === "pl")); } catch (_) {}
    presetPopHint("");
    if (UI.lastState) render(UI.lastState);
  });
  document.addEventListener("pointerdown", onPresetPopOutside, true);
  document.addEventListener("keydown", onPresetPopKey, true);
  document.getElementById("preset-add-input").focus();
}

// Called from render(): keeps an open popover in sync — refresh its pills,
// track the (rebuilt) "+" chip, and close it when the raise row is gone or
// the active cap class changed (format switch).
function syncPresetPop(s) {
  const pop = document.getElementById("preset-pop");
  if (!pop) return;
  const plus = document.getElementById("preset-edit-btn");
  const section = document.getElementById("raise-section");
  const capNow = isPotLimit(s) ? "pl" : "nl";
  if (!plus || !section || section.hidden || pop.dataset.cap !== capNow) {
    closeBetPresetEditor();
    return;
  }
  renderPresetPopContent(pop);
  positionPresetPop(pop, plus);
}

function distRowsHTML(dist, callName) {
  const distNames = ["Fold", callName, "Raise"];
  const distClasses = ["fold", "call", "raise"];
  return (dist || []).map((p, i) => `
    <div class="rec-dist-row">
      <span class="rec-dist-name">${distNames[i] ?? "?"}</span>
      <div class="rec-dist-track">
        <div class="rec-dist-fill ${distClasses[i] ?? ""}" style="width:${(p * 100).toFixed(1)}%"></div>
      </div>
      <span class="rec-dist-pct">${(p * 100).toFixed(0)}%</span>
    </div>`).join("");
}

const ANCHOR_AXIS_LABELS = ["min", "10", "20", "30", "40", "50", "60", "70", "80", "90", "pot"];

function smoothPath(pts) {
  // Catmull-Rom → cubic Bézier so the body curve passes through every
  // anchor point smoothly (tension 1/6).
  if (!pts.length) return "";
  if (pts.length === 1) return `M${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
  let d = `M${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] || pts[i];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2] || p2;
    const c1x = p1[0] + (p2[0] - p0[0]) / 6;
    const c1y = p1[1] + (p2[1] - p0[1]) / 6;
    const c2x = p2[0] - (p3[0] - p1[0]) / 6;
    const c2y = p2[1] - (p3[1] - p1[1]) / 6;
    d += ` C${c1x.toFixed(1)} ${c1y.toFixed(1)} ${c2x.toFixed(1)} ${c2y.toFixed(1)} ${p2[0].toFixed(1)} ${p2[1].toFixed(1)}`;
  }
  return d;
}

// v4 sizing recommendation: a smooth min→pot density curve with discrete
// "wall" bars at the lowest and highest legal sizes (the absorbing atoms —
// min and pot, or min and all-in when the stack caps the range). The white
// dot is the EXACT recommended size (refined chips), never snapped to a 10%
// anchor. Stack-capped spots grey out the unreachable zone past all-in.
function betCurveSVG(rec, s, raiseActive) {
  const anchors = (rec.anchors || []).slice().sort((a, b) => a.k - b.k);
  if (!raiseActive || !anchors.length) return "";
  const lo = anchors[0];
  const hi = anchors[anchors.length - 1];
  // Ladders ending in an ALL-IN atom (frac === null; the 12-anchor NLH spec)
  // can overshoot the pot, so chips-space is a poor axis there: those ladders
  // space anchors evenly by INDEX and tick labels come from the anchors
  // themselves. The PLO ladder keeps the original chips-space axis.
  const idxMode = hi.frac === null || hi.frac === undefined;
  const n1 = Math.max(1, anchors.length - 1);
  // A short-shove node has ONE legal anchor (the all-in atom): index mode
  // centers it instead of dividing by a zero-length ladder
  // (review 2026-09-20 F5).
  const single = anchors.length === 1;
  // (A strategy host that doesn't know the pot reference sends 0, not null.)
  const potRef = rec.pot_ref_chips > 0 ? rec.pot_ref_chips : hi.chips;
  const recA = anchors.find((a) => a.k === rec.rec_anchor) || null;
  let recChips = rec.rec_chips != null ? rec.rec_chips : rec.chips;
  if (recChips == null) recChips = recA ? recA.chips : hi.chips;
  // The ring marks the size actually played: hero review nodes carry
  // `user_chips`, villain nodes only `actual_chips` (+ `user_anchor`) — keying
  // on user_chips alone never drew a villain's marker (review 2026-09-20).
  const userChips = rec.user_chips != null ? rec.user_chips
    : (rec.actual_chips != null ? rec.actual_chips : null);

  const axisLo = lo.chips;
  const axisHi = Math.max(potRef, hi.chips);
  const span = Math.max(1, axisHi - axisLo);
  const pmax = Math.max(1e-9, ...anchors.map((a) => a.prob));
  // chips → axis fraction. Index mode interpolates between the anchors' even
  // positions so the exact-size dot still lands between its neighbors.
  const chipsFrac = (c) => Math.max(0, Math.min(1, (c - axisLo) / span));
  const idxFrac = (c) => {
    if (single) return 0.5;
    if (c <= anchors[0].chips) return 0;
    if (c >= anchors[anchors.length - 1].chips) return 1;
    for (let i = 0; i < anchors.length - 1; i++) {
      const a = anchors[i].chips, b = anchors[i + 1].chips;
      if (c <= b) {
        const t = b > a ? (c - a) / (b - a) : 1;
        return (i + t) / n1;
      }
    }
    return 1;
  };
  const xf = idxMode ? idxFrac : chipsFrac;
  const yf = (p) => Math.max(0, p / pmax);

  const W = 340, padL = 22, padR = 22, padT = 9, plotH = 68;
  const plotW = W - padL - padR, baseY = padT + plotH;
  const px = (f) => padL + f * plotW;
  const py = (v) => baseY - v * plotH;

  const pts = idxMode
    ? anchors.map((a, i) => [px(single ? 0.5 : i / n1), py(yf(a.prob))])
    : anchors.map((a) => [px(xf(a.chips)), py(yf(a.prob))]);
  const body = smoothPath(pts);
  const loX = px(xf(lo.chips)), hiX = px(xf(hi.chips));
  const area = `${body} L${hiX.toFixed(1)} ${baseY} L${loX.toFixed(1)} ${baseY} Z`;
  // Index mode's axis ends at the ladder top (the all-in atom) — nothing past
  // it is representable, so the unreachable-zone hatch never applies.
  const capped = !idxMode && hi.chips < potRef - 1;

  const bar = (cx, p) => {
    const h = yf(p) * plotH;
    return `<rect x="${(cx - 4).toFixed(1)}" y="${(baseY - h).toFixed(1)}" width="8" height="${Math.max(0, h).toFixed(1)}" rx="2" fill="url(#betgrad)"/>`;
  };

  const dotX = px(xf(recChips)), dotY = py(yf(recA ? recA.prob : pmax));

  let userMark = "";
  if (userChips != null) {
    const uA = rec.user_anchor != null ? anchors.find((a) => a.k === rec.user_anchor) : null;
    userMark = `<circle cx="${px(xf(userChips)).toFixed(1)}" cy="${py(yf(uA ? uA.prob : 0)).toFixed(1)}" r="5" fill="none" stroke="var(--text-bright)" stroke-width="2"/>`;
  }

  let grey = "";
  if (capped) {
    const gw = Math.max(0, (W - padR) - hiX);
    grey =
      `<rect x="${hiX.toFixed(1)}" y="${padT}" width="${gw.toFixed(1)}" height="${plotH}" fill="var(--muted)" opacity="0.05"/>` +
      `<rect x="${hiX.toFixed(1)}" y="${padT}" width="${gw.toFixed(1)}" height="${plotH}" fill="url(#bethatch)"/>`;
  }
  const allinY = Math.max(10, baseY - yf(hi.prob) * plotH - 7);
  const allin = capped
    ? `<text x="${hiX.toFixed(1)}" y="${allinY.toFixed(1)}" class="bc-allin" text-anchor="middle">all-in</text>`
    : "";

  const tickText = (frac, lbl, dim) =>
    `<text x="${px(frac).toFixed(1)}" y="${(baseY + 13).toFixed(1)}" ` +
    `class="bc-tick${dim ? " dim" : ""}" text-anchor="middle">${escapeHTML(lbl)}</text>`;
  let ticks = "";
  if (idxMode) {
    // Ticks are a spread of the anchors' own labels (always both ends, so
    // "min" and "ALL-IN" show); all 12 labels would collide at this width.
    // Indices are bounded by the LAST anchor: a one-anchor ladder used to read
    // anchors[1] and throw, aborting the whole render (review 2026-09-20 F5).
    const lastIdx = anchors.length - 1;
    const tickIdx = new Set([0, lastIdx]);
    for (let j = 1; j <= 3; j++) tickIdx.add(Math.round((j * lastIdx) / 4));
    for (const i of [...tickIdx].sort((a, b) => a - b)) {
      ticks += tickText(single ? 0.5 : i / n1,
                        anchors[i].label != null ? anchors[i].label : "");
    }
  } else {
    // Endpoints are always labelled; "pot" dims when the stack caps the range.
    // A no-limit ladder whose top fraction anchor is clamped to the stack
    // ends ABOVE pot: the right end is then the all-in ("max" when the cap is
    // a cover-short clamp and the actor keeps chips behind), and "pot" gets
    // its own tick at its true position (review 2026-09-20 F10).
    const overPot = hi.chips > potRef + 1;
    const nodeActor = s && s.actor !== null && s.actor !== undefined ? s.seats[s.actor] : null;
    const endName = nodeActor && !isAllInDelta(nodeActor, hi.chips) ? "max" : "all-in";
    ticks += tickText(0, "min") + tickText(1, overPot ? endName : "pot", capped);
    // Interior labels annotate the HUMPS OF THE DRAWN CURVE — the local
    // maxima of the anchor distribution the spline passes through — never the
    // head's latent parameters. (Mixture component centers routinely sit off
    // the blended marginal's peaks: components overlap, the marginal is
    // discretized onto the anchor grid, and the ε weight floor lifts the
    // whole baseline. Labelling μ's put text on flat stretches and left
    // visible humps unlabelled.) Works identically for every anchor head
    // (v2/v4/v5+). Anchors that clamp to the same chips (min-raise / all-in
    // dupes) collapse to one point first. The white dot — the recommended
    // size — always gets the first interior label, at its exact pot-%, so the
    // rec reads straight off the axis; the hump it sits on then yields to it.
    // Remaining humps are labelled tallest-first with a pixel-space gap so
    // nothing overlaps at this width. When the stack caps the ladder, the
    // all-in column already carries the "all-in" text, so no %-label lands
    // under it.
    const MINGAP = 28;  // min px between label centers ("100%" ≈ 25px @ 11px)
    const PROM = 0.02;  // min hump prominence, as a fraction of the tallest anchor
    const placed = [px(0), px(1)];
    if (capped) placed.push(hiX);
    const free = (x) => placed.every((q) => Math.abs(q - x) >= MINGAP);
    const put = (x, frac) => {
      placed.push(x);
      ticks += tickText((x - padL) / plotW, `${Math.round(frac * 100)}%`);
    };
    if (overPot) {
      const potX = px(xf(potRef));
      if (free(potX)) { placed.push(potX); ticks += tickText(xf(potRef), "pot"); }
    }
    // Collapse clamped dupes to distinct axis positions (max prob wins).
    const dx = [];
    for (const a of anchors) {
      const x = px(xf(a.chips));
      const last = dx[dx.length - 1];
      if (last && x - last.x < 0.75) {
        if (a.prob > last.p) { last.p = a.prob; last.frac = a.frac; }
      } else dx.push({ x, p: a.prob, frac: a.frac != null ? a.frac : a.k / n1 });
    }
    // Pot-% at an arbitrary chips position. Exact when the node's to-call is
    // known: chips = toCall + frac·(pot + toCall) and pot_ref = pot + 2·toCall.
    // (The piecewise fallback below interpolates the anchors' NOMINAL fracs,
    // which is off inside a clamped end bracket — "min" is nominally 0% but
    // sits at the min-raise — so a rec just above the min-raise read far too
    // small.) Falls back to piecewise-linear between distinct anchors — exact
    // wherever the ladder isn't clamped, since raise-to chips are linear in
    // frac there. (review 2026-09-20 — found while fixing F10)
    const tcRef = rec.to_call_chips != null ? rec.to_call_chips
      : (nodeActor ? s.to_call_chips : null);
    const baseRef = rec.pot_ref_chips != null && tcRef != null
      ? rec.pot_ref_chips - tcRef : 0;
    const fracAt = (c) => {
      if (baseRef > 0) return Math.max(0, (c - tcRef) / baseRef);
      const cx = px(xf(c));
      if (cx <= dx[0].x) return dx[0].frac;
      for (let i = 0; i + 1 < dx.length; i++) {
        const a = dx[i], b = dx[i + 1];
        if (cx <= b.x) return a.frac + ((cx - a.x) / (b.x - a.x)) * (b.frac - a.frac);
      }
      return dx[dx.length - 1].frac;
    };
    if (free(dotX)) put(dotX, fracAt(recChips));
    // Interior local maxima, with valley-to-valley prominence: immediate
    // neighbors understate a broad hump (they sit on its shoulders), so walk
    // outward to the nearest higher ground and measure against the deepest
    // valley on the way. The absolute floor keeps near-uniform ripple quiet.
    const promMin = Math.max(PROM * pmax, 0.004);
    const humps = [];
    for (let i = 1; i + 1 < dx.length; i++) {
      const c = dx[i].p;
      if (c < dx[i - 1].p || c < dx[i + 1].p) continue;
      if (c === dx[i - 1].p && c === dx[i + 1].p) continue; // plateau interior
      let lv = c, rv = c;
      for (let j = i - 1; j >= 0 && dx[j].p <= c; j--) lv = Math.min(lv, dx[j].p);
      for (let j = i + 1; j < dx.length && dx[j].p <= c; j++) rv = Math.min(rv, dx[j].p);
      if (c - Math.max(lv, rv) >= promMin) humps.push(dx[i]);
    }
    humps.sort((a, b) => b.p - a.p);
    for (const h of humps) if (free(h.x)) put(h.x, h.frac);
  }

  return `<svg class="bet-curve" viewBox="0 0 ${W} 113" xmlns="http://www.w3.org/2000/svg">`
    + `<defs>`
    + `<linearGradient id="betgrad" gradientUnits="userSpaceOnUse" x1="${px(0).toFixed(1)}" y1="0" x2="${px(1).toFixed(1)}" y2="0">`
    + `<stop offset="0" stop-color="#5FE5D9"/><stop offset="1" stop-color="#18D2C3"/></linearGradient>`
    + `<pattern id="bethatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">`
    + `<line x1="0" y1="0" x2="0" y2="8" stroke="var(--muted)" stroke-width="1" stroke-opacity="0.2"/></pattern>`
    + `</defs>`
    + grey
    + `<line x1="${padL}" y1="${baseY}" x2="${W - padR}" y2="${baseY}" stroke="var(--border)" stroke-width="1"/>`
    + `<path d="${area}" fill="url(#betgrad)" fill-opacity="0.16"/>`
    + `<path d="${body}" fill="none" stroke="url(#betgrad)" stroke-width="2.4" stroke-linecap="round"/>`
    + bar(loX, lo.prob) + bar(hiX, hi.prob)
    + `<line x1="${dotX.toFixed(1)}" y1="${dotY.toFixed(1)}" x2="${dotX.toFixed(1)}" y2="${baseY}" stroke="var(--text-bright)" stroke-opacity="0.4" stroke-width="1.5"/>`
    + `<circle cx="${dotX.toFixed(1)}" cy="${dotY.toFixed(1)}" r="9" fill="var(--text-bright)" opacity="0.16"/>`
    + `<circle cx="${dotX.toFixed(1)}" cy="${dotY.toFixed(1)}" r="5.5" fill="var(--text-bright)"/>`
    + userMark + allin + ticks
    + `<text x="${padL}" y="106" class="bc-end" fill="var(--muted)">smaller bets</text>`
    + `<text x="${(W - padR)}" y="106" class="bc-end" fill="var(--muted)" text-anchor="end">larger bets</text>`
    + `</svg>`;
}

function recDetailHTML(rec, userAnchor, s) {
  // v2 payloads carry `anchors`; v1 carries the single Beta's (α, β).
  // raiseActive mirrors the displayed RAISE % — when it rounds to 0 the
  // network never raises here, so the whole sizing detail is blanked.
  const gateDist = rec.gate_distribution || rec.gate_probs || [];
  const raiseActive = Math.round((gateDist[2] || 0) * 100) > 0;
  const anchors = recAnchorRows(rec, s);
  if (anchors) {
    return betCurveSVG(rec.anchors ? rec : { ...rec, anchors }, s, raiseActive);
  }
  if (!raiseActive) return "";
  // Neither an anchor ladder nor a Beta: nothing to show — not "β(0.0, 0.0)".
  if (typeof rec.beta_alpha !== "number" || typeof rec.beta_beta !== "number") return "";
  return `<div class="rec-detail">β(${rec.beta_alpha.toFixed(1)}, ${rec.beta_beta.toFixed(1)})</div>`;
}

// Anchor rows for the bet curve. v2+ PPO payloads carry ready-made `anchors`;
// strategy-host recommendations carry the raw parallel arrays
// (anchor_probs / anchor_chips / anchor_legal, spec-length) instead — build
// the same legal-only rows from whichever is present (review 2026-09-20 F13).
function recAnchorRows(rec, s) {
  if (Array.isArray(rec.anchors)) return rec.anchors;
  const probs = rec.anchor_probs, chips = rec.anchor_chips;
  if (!Array.isArray(probs) || !Array.isArray(chips) || !probs.length) return null;
  const legal = Array.isArray(rec.anchor_legal) ? rec.anchor_legal : null;
  const top = probs.length - 1;
  // A no-limit ladder ends in the ALL-IN atom (chips = max raise, no pot
  // fraction); a pot-limit ladder's top anchor is 100% pot.
  const atomTop = !isPotLimit(s);
  const toCall = rec.to_call_chips != null ? rec.to_call_chips : ((s && s.to_call_chips) || 0);
  const base = rec.pot_ref_chips > 0 ? rec.pot_ref_chips - toCall : 0;
  const rows = [];
  for (let k = 0; k <= top; k++) {
    if (legal && !legal[k]) continue;
    const c = Number(chips[k]) || 0;
    const isAtom = atomTop && k === top;
    // chips = toCall + frac·(pot + toCall)  →  the anchor's true pot fraction.
    const frac = isAtom ? null
      : k === 0 ? 0
      : base > 0 ? Math.max(0, (c - toCall) / base) : k / Math.max(1, top);
    // Without a pot reference the index-based frac only spaces the axis — it
    // is not a real pot-%, so it gets no label.
    const label = k === 0 ? "min" : isAtom ? "ALL-IN"
      : base > 0 ? `${Math.round(frac * 100)}%` : "";
    rows.push({ k, frac, label, prob: Number(probs[k]) || 0, chips: c });
  }
  return rows.length ? rows : null;
}

function fmtSignedValue(bb, s) {
  if (bb === null || bb === undefined) return null;
  const sign = bb >= 0 ? "+" : "-";
  const abs = Math.abs(bb);
  return UI.unit === "bb"
    ? `${sign}${abs.toFixed(2)}bb`
    : `${sign}$${(abs * (s?.chip_scale?.dollars_per_bb ?? 2)).toFixed(2)}`;
}

function renderTrainerReviewRecommendation(s, el) {
  const rv = s.trainer.review;
  const cur = rv.node_current || rv.current;
  const whatif = rv.whatif;
  const callName = cur.to_call_chips > 0 ? "Call" : "Check";
  // Raise sizes are raise-BY deltas; show raise-TO totals on top of the
  // acting seat's street commit at this node (review 2026-09-20 F11).
  const commit = reviewCurCommit(s, rv, cur, reviewNodeCommits(s, rv));
  let actionText, dist, valueBB, valueTrueBB = null, tag = "", detailHTML;
  if (whatif) {
    const rec = whatif.recommendation;
    actionText = gateActionLabel(rec.gate, rec.chips, cur.to_call_chips, s, commit, cur.street);
    dist = rec.gate_distribution;
    valueBB = rec.value_bb;
    tag = `<span class="whatif-tag">what-if</span> `;
    detailHTML = recDetailHTML(rec, null, s);
  } else {
    actionText = cur.rec_gate
      ? gateActionLabel(cur.rec_gate, cur.rec_chips, cur.to_call_chips, s, commit, cur.street)
      : cur.rec_label;
    dist = cur.gate_probs;
    valueBB = cur.value_bb;
    valueTrueBB = cur.value_true_bb;       // null on v1 / when no critic
    detailHTML = recDetailHTML(cur, cur.user_anchor, s);
  }
  const ownV = fmtSignedValue(valueBB, s);
  const trueV = fmtSignedValue(valueTrueBB, s);
  // "own" = the acting seat's blind value head; "true" = the all-cards
  // critic's EV for that seat. Show both side by side when available.
  const valueHTML = trueV
    ? `<span class="rec-value">value (own) ${ownV} · ` +
      `<span class="rec-value-true">true ${trueV}</span></span>`
    : `<span class="rec-value">value ${ownV}</span>`;
  const recPayload = whatif ? whatif.recommendation : cur;
  el.innerHTML = `
    ${untrainedBadgeHTML(recPayload, s)}
    <div class="rec-line">
      ${tag}<span class="rec-action">${actionText}</span>
      ${valueHTML}
    </div>
    <div class="rec-dist">${distRowsHTML(dist, callName)}</div>
    ${detailHTML}
  `;
}

function renderRecommendation(s) {
  const el = document.getElementById("recommendation");
  if (s.trainer) {
    if (s.trainer.review && (s.trainer.review.node_current || s.trainer.review.current)) {
      renderTrainerReviewRecommendation(s, el);
      return;
    }
    el.innerHTML = '<p class="muted">Hidden during play — revealed in the post-hand review.</p>';
    return;
  }
  const holeCount = s.card_spec ? s.card_spec.hero_hole.length : 5;
  const REC_PROMPTS = {
    hole:  `Place hero's ${holeCount} hole cards to see network output.`,
    flop:  "Place the flop cards to see network output.",
    turn:  "Place the turn cards to see network output.",
    river: "Place the river cards to see network output.",
  };
  if (s.hero_blocking_reason != null) {
    el.innerHTML = `<p class="muted">${REC_PROMPTS[s.hero_blocking_reason]
      ?? "Place cards to see network output."}</p>`;
    return;
  }
  const rec = s.recommendation;
  if (!rec) {
    el.innerHTML = '<p class="muted">Hero is not the current actor.</p>';
    return;
  }
  let actionText = rec.gate_name;
  if (rec.gate === "fold") {
    actionText = "Fold";
  } else if (rec.gate === "check_call") {
    actionText = s.to_call_chips > 0 ? `Call ${formatUnit(s.to_call_chips, s)}` : "Check";
  } else if (rec.gate === "raise" && rec.chips !== null && rec.chips !== undefined) {
    const verb = aggVerb(s.to_call_chips, s.street);
    const actorSeat = s.actor !== null && s.actor !== undefined ? s.seats[s.actor] : null;
    // rec.chips is a raise-BY delta: all-in iff it takes the remaining stack
    // (review 2026-09-20 F8).
    const isAllIn = isAllInDelta(actorSeat, rec.chips);
    const suffix = isAllIn ? " (all-in)" : "";
    // rec.chips is a raise-BY delta; display the raise-TO total (delta + ac).
    const ac = actorCommitChips(s);
    actionText = `${verb} ${formatUnit(rec.chips + ac, s)}${suffix}`;
  }
  const distRows = distRowsHTML(
    rec.gate_distribution || [],
    s.to_call_chips > 0 ? "Call" : "Check",
  );
  const vBB = rec.value_bb;
  const sign = vBB >= 0 ? "+" : "-";
  const absBB = Math.abs(vBB);
  const vDisp = UI.unit === "bb"
    ? `${sign}${absBB.toFixed(2)}bb`
    : `${sign}$${(absBB * (s?.chip_scale?.dollars_per_bb ?? 2)).toFixed(2)}`;
  el.innerHTML = `
    ${untrainedBadgeHTML(rec, s)}
    <div class="rec-line">
      <span class="rec-action">${actionText}</span>
      <span class="rec-value">value ${vDisp}</span>
    </div>
    <div class="rec-dist">${distRows}</div>
    ${recDetailHTML(rec, null, s)}
  `;
}

// `walk` = historyCommits(s). History chips are what the action ADDED; bets,
// raises and all-ins are shown as the seat's street total after the action
// (raise-TO) — the convention of the bet chips on the table and the raise
// input (review 2026-09-20 F11). Calls keep the call amount, like the Call
// button.
function actionLabel(h, s, idx, walk) {
  const name = h.action;
  const chips = Number(h.chips) || 0;
  if (name === "Fold") return "Fold";
  if (name === "CheckCall") return chips > 0 ? `Call ${formatUnit(chips, s)}` : "Check";
  const w = walk ? walk[idx] : null;
  const total = w ? w.after : chips;
  if (name === "AllIn") return `All-in ${formatUnit(total, s)}`;
  // A raise when the street already has a live bet — incl. the preflop
  // blinds, which never appear in the history (review 2026-09-20 F17).
  const raised = (w && w.level > 0) || h.street === "preflop";
  return `${raised ? "Raise" : "Bet"} ${formatUnit(total, s)}`;
}

function renderHistory(s) {
  const el = document.getElementById("history");
  el.innerHTML = "";
  if (!s.history.length) {
    el.innerHTML = '<p class="muted">No actions yet.</p>';
    return;
  }
  const walk = historyCommits(s);
  for (let i = 0; i < s.history.length; i++) {
    const h = s.history[i];
    const row = document.createElement("div");
    row.className = "history-entry";
    const streetClass = `h-${String(h.street).toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
    row.innerHTML = `
      <span class="h-street ${streetClass}">${h.street}</span>
      <span class="h-pos">${h.position}</span>
      <span class="h-action">${actionLabel(h, s, i, walk)}</span>
    `;
    el.appendChild(row);
  }
}

function renderCardGrid(s) {
  const grid = document.getElementById("card-grid");
  grid.innerHTML = "";
  const used = collectUsedCards(s);
  // 13 rows × 4 cols. Rows = ranks (A top to 2 bottom). Cols = suits c,d,h,s.
  for (let rank = 12; rank >= 0; rank--) {
    for (let suit = 0; suit < 4; suit++) {
      const cardInt = rank * 4 + suit;
      const info = cardToString(cardInt);
      const b = document.createElement("button");
      b.className = "card-btn suit-" + info.suit;
      if (used.has(cardInt)) b.classList.add("used");
      b.dataset.card = String(cardInt);
      b.innerHTML = `
        <span class="corner-tl">
          <span class="corner-rank">${info.rank}</span>
          <span class="corner-suit">${info.glyph}</span>
        </span>
        <span class="corner-br">${info.rank}</span>
      `;
      b.disabled = used.has(cardInt);
      b.addEventListener("click", () => onGridCardClick(cardInt));
      grid.appendChild(b);
    }
  }
  const hint = document.getElementById("slot-hint");
  if (UI.selectedSlot) {
    const { key, index } = UI.selectedSlot;
    hint.textContent = `Selected: ${slotLabel(s, key)} [${index + 1}]`;
  } else {
    hint.textContent = "Click a slot to place cards";
  }
}

// --- Slot / grid interactions ----------------------------------------------

function onSlotClick(key, index) {
  const s = UI.lastState;
  if (s && s.trainer) {
    // Trainer: card slots are display-only mid-hand; in review a click
    // selects the card for a what-if swap.
    const rv = s.trainer.review;
    if (!rv) return;
    // What-if is hero-only (the /whatif replay must reach hero's node) —
    // villain-node card clicks are inert. So are hero moot auto-checks: no
    // graded decision sits behind them to re-score (review 2026-09-20 F12).
    if (reviewNodeUngraded(rv.node_current)) return;
    const spec = s.card_spec[key];
    if (!spec || spec[index] === null || spec[index] === undefined) return;
    // The hand-end view shows the FINAL board, but a what-if re-runs the
    // reviewed decision: a street dealt after it isn't part of that node.
    const streetRank = reviewStreetRank(rv);
    if (streetRank !== null && SLOT_MIN_STREET[key] > streetRank) {
      showToast(`${slotLabel(s, key)} isn't dealt yet at the reviewed decision — `
        + "step to a later decision to swap it.", "info");
      return;
    }
    if (UI.selectedSlot && UI.selectedSlot.key === key && UI.selectedSlot.index === index) {
      UI.selectedSlot = null;
      cancelTrainerPick();
    } else {
      UI.selectedSlot = { key, index };
      UI.trainerPick = true;
      document.body.classList.add("trainer-pick");
    }
    render(s);
    return;
  }
  if (UI.selectedSlot && UI.selectedSlot.key === key && UI.selectedSlot.index === index) {
    UI.selectedSlot = null;
  } else {
    UI.selectedSlot = { key, index };
  }
  render(UI.lastState);
}

function onSlotDoubleClick(key, index) {
  const s = UI.lastState;
  if (!s || s.trainer) return;
  const spec = s.card_spec[key];
  if (spec[index] === null) return;
  spec[index] = null;
  UI.selectedSlot = { key, index };
  commitLocalCards();
}

// Street order + the first street on which each card slot is dealt. A what-if
// re-runs ONE reviewed decision, so slots of later streets don't exist at
// that node (review 2026-09-20 F3).
const STREET_RANK = { preflop: 0, flop: 1, turn: 2, river: 3, showdown: 4 };
const SLOT_MIN_STREET = { hero_hole: 0, flop_a: 1, flop_b: 1, turn: 2, river: 3 };

// Street rank of the decision a what-if would re-run (`review.decision`, whose
// detail is `review.current`); null when the payload doesn't say.
function reviewStreetRank(rv) {
  const cur = rv && (rv.current || rv.node_current);
  const rank = cur && cur.street ? STREET_RANK[String(cur.street).toLowerCase()] : undefined;
  return rank === undefined ? null : rank;
}

function onGridCardClick(cardInt) {
  const s = UI.lastState;
  if (!s) return;
  if (s.trainer) {
    const rv = s.trainer.review;
    const sel = UI.selectedSlot;
    if (!rv || !sel || !UI.trainerPick) return;
    if (reviewNodeUngraded(rv.node_current)) return;  // hero's graded decisions only
    const used = collectUsedCards(s);
    if (used.has(cardInt)) return;
    // Send the full current spec as absolute overrides so earlier
    // what-if swaps survive; the server re-validates against originals.
    const base = rv.whatif ? rv.whatif.card_spec : s.card_spec;
    const body = { decision: rv.decision };
    // From the hand-end view `base` is the FINAL board, but the server only
    // accepts overrides for cards revealed at the reviewed decision ("turn[0]
    // is not revealed at this decision" → the what-if failed unless hero's
    // last decision was on the river). Blank the later streets
    // (review 2026-09-20 F3).
    const streetRank = reviewStreetRank(rv);
    if (streetRank !== null && SLOT_MIN_STREET[sel.key] > streetRank) return;
    for (const k of ["hero_hole", "flop_a", "flop_b", "turn", "river"]) {
      body[k] = (base[k] || []).slice();
      if (streetRank !== null && SLOT_MIN_STREET[k] > streetRank) {
        body[k] = body[k].map(() => null);
      }
    }
    if (body[sel.key][sel.index] === undefined) return;  // slot not in this format
    body[sel.key][sel.index] = cardInt;
    UI.selectedSlot = null;
    cancelTrainerPick();
    postTrainer("whatif", body);
    return;
  }
  placeStudyCard(cardInt);
}

// --- Continuous card entry (study) -------------------------------------------
// Cards are entered in dealing order: hero hole -> board A flop -> board B flop
// -> turn -> river. Placing a card moves the selection to the NEXT EMPTY slot —
// across groups, not just inside one (it used to stop after the fifth hole
// card, so every street needed another click on the table). With nothing
// selected a card goes to the first empty slot, so a whole spot can be entered
// by clicking (or typing) cards one after another.
const SLOT_ORDER = ["hero_hole", "flop_a", "flop_b", "turn", "river"];

function nextEmptySlot(s, after) {
  let started = !after;
  for (const key of SLOT_ORDER) {
    const arr = (s.card_spec && s.card_spec[key]) || [];
    for (let i = 0; i < arr.length; i++) {
      if (!started) {
        if (key === after.key && i === after.index) started = true;
        continue;
      }
      if (arr[i] === null || arr[i] === undefined) return { key, index: i };
    }
  }
  return null;
}

function placeStudyCard(cardInt) {
  const s = UI.lastState;
  if (!s || s.trainer || UI.mode !== "study") return false;
  const sel = UI.selectedSlot || nextEmptySlot(s, null);
  if (!sel) { showToast("Every card slot is filled — double-click a card to clear it"); return false; }
  if (collectUsedCards(s).has(cardInt)) { showToast("That card is already on the table"); return false; }
  const spec = s.card_spec[sel.key];
  if (!spec || sel.index >= spec.length) return false;
  spec[sel.index] = cardInt;
  UI.selectedSlot = nextEmptySlot(s, sel);
  commitLocalCards();
  return true;
}

// Keyboard entry: a rank (2-9 T J Q K A) then a suit (c d h s) places that card;
// Backspace takes back the most recent card; Escape drops the selection.
const KEY_RANKS = "23456789TJQKA";
const KEY_SUITS = "cdhs";
let pendingRankKey = null;

function lastFilledSlot(s) {
  let last = null;
  for (const key of SLOT_ORDER) {
    const arr = (s.card_spec && s.card_spec[key]) || [];
    arr.forEach((v, i) => { if (v !== null && v !== undefined) last = { key, index: i }; });
  }
  return last;
}

function setupCardKeyboard() {
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const t = e.target;
    const tag = t && t.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || (t && t.isContentEditable)) return;
    const s = UI.lastState;
    if (!s || s.trainer || UI.mode !== "study" || UI.settingsOpen) return;
    if (document.body.classList.contains("ranges-mode")) return;
    const k = e.key;
    const rank = KEY_RANKS.indexOf(k.toUpperCase());
    if (k.length === 1 && rank >= 0) {  // (no letter is both a rank and a suit)
      pendingRankKey = rank;
      const hint = document.getElementById("slot-hint");
      if (hint) hint.textContent = `${KEY_RANKS[rank]}… now a suit: c d h s`;
      e.preventDefault();
      return;
    }
    const suit = k.length === 1 ? KEY_SUITS.indexOf(k.toLowerCase()) : -1;
    if (suit >= 0 && pendingRankKey !== null) {
      const cardInt = pendingRankKey * 4 + suit;
      pendingRankKey = null;
      placeStudyCard(cardInt);
      e.preventDefault();
      return;
    }
    if (e.key === "Backspace") {
      pendingRankKey = null;
      const last = lastFilledSlot(s);
      if (last) {
        s.card_spec[last.key][last.index] = null;
        UI.selectedSlot = last;
        commitLocalCards();
      }
      e.preventDefault();
      return;
    }
    if (e.key === "Escape" && (UI.selectedSlot || pendingRankKey !== null)) {
      pendingRankKey = null;
      UI.selectedSlot = null;
      render(s);
    }
  });
}

// --- Dealer-button drag -----------------------------------------------------

const UI_INSERT = { pairA: null, pairB: null };

function svgPoint(svg, clientX, clientY) {
  const pt = svg.createSVGPoint();
  pt.x = clientX; pt.y = clientY;
  return pt.matrixTransform(svg.getScreenCTM().inverse());
}

function ellipseAngle(px, py) {
  return Math.atan2((py - TABLE_CENTER.y) / SEAT_RY,
                    (px - TABLE_CENTER.x) / SEAT_RX);
}

function ellipseRadius(px, py) {
  return Math.hypot((px - TABLE_CENTER.x) / SEAT_RX,
                    (py - TABLE_CENTER.y) / SEAT_RY);
}

function setupInsertHover() {
  const svg = document.getElementById("table-svg");
  const icon = document.getElementById("insert-icon");
  if (!icon) return;
  const TWO_PI = 2 * Math.PI;
  const THETA0 = Math.PI / 2;

  svg.addEventListener("mousemove", (e) => {
    const s = UI.lastState;
    if (!s || s.num_seats >= 6 || s.trainer) { icon.setAttribute("hidden", ""); return; }
    const pt = svgPoint(svg, e.clientX, e.clientY);
    const r = ellipseRadius(pt.x, pt.y);
    if (r < 0.55 || r > 1.25) { icon.setAttribute("hidden", ""); return; }

    const N = s.num_seats;
    const seatStep = TWO_PI / N;
    const ma = ellipseAngle(pt.x, pt.y);
    // seatPositions() lays seats out at THETA0 + rel·step (INCREASING theta =
    // visual clockwise), so the gap index under the cursor must be measured
    // the same way. Measuring THETA0 − ma walked the table the other way
    // round: the "+" inserted into the mirror-image gap
    // (review 2026-09-20 F16).
    let rel = ma - THETA0;
    while (rel < 0) rel += TWO_PI;
    while (rel >= TWO_PI) rel -= TWO_PI;
    const rawIdx = Math.min(N - 1, Math.floor(rel / seatStep));
    const hero = s.hero_seat;
    const pairA = (rawIdx + hero) % N;
    const pairB = (rawIdx + 1 + hero) % N;

    const midTheta = THETA0 + (rawIdx + 0.5) * seatStep;
    const mx = TABLE_CENTER.x + SEAT_RX * Math.cos(midTheta);
    const my = TABLE_CENTER.y + SEAT_RY * Math.sin(midTheta);
    icon.setAttribute("transform", `translate(${mx} ${my})`);
    icon.removeAttribute("hidden");
    UI_INSERT.pairA = pairA;
    UI_INSERT.pairB = pairB;
  });

  svg.addEventListener("mouseleave", () => { icon.setAttribute("hidden", ""); });

  icon.addEventListener("click", (e) => {
    e.stopPropagation();
    const s = UI.lastState;
    if (!s || s.num_seats >= 6) return;
    if (UI_INSERT.pairA == null) return;
    insertSeat(UI_INSERT.pairA, s);
    icon.setAttribute("hidden", "");
  });
}

function setupDealerDrag() {
  const button = document.getElementById("dealer-button");
  const svg = document.getElementById("table-svg");
  const getSvgPoint = (evt) => {
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX; pt.y = evt.clientY;
    const ctm = svg.getScreenCTM();
    return ctm ? pt.matrixTransform(ctm.inverse()) : { x: evt.clientX, y: evt.clientY };
  };
  button.addEventListener("pointerdown", (e) => {
    if (UI.mode === "trainer") return;
    e.preventDefault();
    UI.draggingButton = true;
    button.classList.add("dragging");
    button.setPointerCapture(e.pointerId);
  });
  button.addEventListener("pointermove", (e) => {
    if (!UI.draggingButton) return;
    const p = getSvgPoint(e);
    button.setAttribute("transform", `translate(${p.x} ${p.y})`);
  });
  const endDrag = (e) => {
    if (!UI.draggingButton) return;
    UI.draggingButton = false;
    button.classList.remove("dragging");
    try { button.releasePointerCapture(e.pointerId); } catch (_) {}
    const p = getSvgPoint(e);
    const s = UI.lastState;
    if (!s) return;
    const positions = seatPositions(s.num_seats, s.hero_seat);
    let bestSeat = 0, bestDist = Infinity;
    for (let i = 0; i < s.num_seats; i++) {
      const dx = positions[i].x - p.x;
      const dy = positions[i].y - p.y;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestDist) { bestDist = d2; bestSeat = i; }
    }
    if (bestSeat !== s.button_seat) {
      postSeats({ button_seat: bestSeat });
    } else {
      render(s);
    }
  };
  button.addEventListener("pointerup", endDrag);
  button.addEventListener("pointercancel", () => {
    UI.draggingButton = false;
    button.classList.remove("dragging");
    if (UI.lastState) render(UI.lastState);
  });
}

// --- Top-bar handlers -------------------------------------------------------

function setupTopBar() {
  document.getElementById("seats-dec").addEventListener("click", () => {
    const s = UI.lastState;
    if (!s || s.num_seats <= 2) return;
    postSeats({ num_seats: s.num_seats - 1 });
  });
  document.getElementById("seats-inc").addEventListener("click", () => {
    const s = UI.lastState;
    if (!s || s.num_seats >= 6) return;
    postSeats({ num_seats: s.num_seats + 1 });
  });
  document.getElementById("unit-toggle").addEventListener("click", () => {
    UI.unit = UI.unit === "bb" ? "$" : "bb";
    if (UI.lastState) render(UI.lastState);
  });
  document.getElementById("dpb-input").addEventListener("change", (e) => {
    const v = parseFloat(e.target.value);
    if (!isFinite(v) || v <= 0) return;
    postConfig({ dollars_per_bb: v });
  });
  document.getElementById("ante-input").addEventListener("change", (e) => {
    const s = UI.lastState;
    if (!s) return;
    const v = parseFloat(e.target.value);
    if (!isFinite(v) || v < 0) return;
    let anteChips;
    if (UI.unit === "bb") anteChips = bbToChips(v, s);
    else {
      const dpb = s.chip_scale.dollars_per_bb || 2;
      anteChips = bbToChips(v / dpb, s);
    }
    postConfig({ ante_chips: anteChips });
  });
  document.getElementById("format-select").addEventListener("change", async (e) => {
    const sel = e.target;
    const prev = UI.lastState && UI.lastState.format ? UI.lastState.format : null;
    // New context (review 2026-09-20 F4/F15): responses still in flight for
    // the old format go stale, any frame playback stops, and the old format's
    // action controls / typed raise must not stay live while /format loads.
    UI.ctxSeq++;
    UI.animSeq++;
    UI.lastStateKey = null;  // the screen was just blanked: force the next paint
    resetRaiseEntry();
    clearActionControls();
    try {
      const data = await postJSON("/format", { format: sel.value });
      // The server switches BOTH tabs' sessions — drop per-hand client
      // cursors, then show the fresh state for whichever mode is active.
      UI.selectedSlot = null;
      UI.reviewDecision = null;
      UI.reviewNode = null;
      cancelTrainerPick();
      if (UI.mode === "trainer") {
        UI.lastState = null;
        UI.lastStateKey = null;
        await fetchState();
      } else {
        applyState(data.state);
      }
    } catch (err) {
      showToast(err.message);
      if (prev) sel.value = prev;  // revert the visible selection
      // The switch didn't happen: bring back the controls blanked above, then
      // resync — a response dropped as stale may have advanced the server.
      if (UI.lastState) render(UI.lastState);
      resyncQuietly();
    }
  });
  document.getElementById("undo-btn").addEventListener("click", () => postUndo());
  document.getElementById("new-hand-btn").addEventListener("click", () => postReset());
  // Optional integration hook (defined only in some builds).
  if (window.__wgLiveTopBar) window.__wgLiveTopBar();
}

// Populate the top-bar format dropdown from GET /formats. Untrained formats
// (no promoted checkpoint) get a muted "(untrained)" suffix.
async function initFormats() {
  const sel = document.getElementById("format-select");
  if (!sel) return;
  let data;
  try {
    data = await getJSON("/formats");
  } catch (_) {
    // Endpoint unavailable — hide the control rather than show an empty box.
    const wrap = sel.closest(".top-bar-item");
    if (wrap) wrap.style.display = "none";
    return;
  }
  UI.formats = data.formats || [];
  sel.innerHTML = "";
  for (const f of UI.formats) {
    const opt = document.createElement("option");
    opt.value = f.id;
    if (f.locked) {
      // Server-gated for this account (POST /format would 403): shown but
      // not selectable. "coming soon!" replaces the untrained marker.
      opt.disabled = true;
      opt.textContent = `${f.label} — coming soon!`;
      opt.className = "opt-locked";
    } else {
      opt.textContent = f.model_loaded ? f.label : `${f.label} (untrained)`;
      if (!f.model_loaded) opt.className = "opt-untrained";
    }
    sel.appendChild(opt);
  }
  const active = (UI.lastState && UI.lastState.format) || data.active;
  if (active) sel.value = active;
  // The raise-preset row derives its cap class (pot-limit vs no-limit) from
  // this payload — refresh a state rendered before it arrived.
  if (UI.lastState) render(UI.lastState);
}

// --- Public build: account, sign-in gate, paywall -------------------------

async function refreshMe() {
  try { UI.me = await getJSON("/me"); }
  catch (_) { UI.me = null; }
  renderAccountChip();
  return UI.me;
}

function isEntitled() {
  return !window.PLO5BP_PUBLIC || !!(UI.me && UI.me.sub && UI.me.sub.active);
}

// Admin-only "safe to deploy?" signal: distinct non-admin users who made an
// authenticated request inside the server's activity window (default 5 min).
const ACTIVE_POLL_MS = 10000;
let ACTIVE_POLL_TIMER = null;

async function refreshActiveCount() {
  if (document.hidden) return; // resync on visibilitychange instead
  const pill = document.getElementById("acct-active-pill");
  if (!pill) return;
  try {
    const d = await getJSON("/admin/api/active");
    pill.textContent = `Active: ${d.active_users}`;
    const mins = Math.max(1, Math.round(d.window_seconds / 60));
    const lines = d.emails && d.emails.length
      ? [`Active in the last ${mins} min:`, ...d.emails]
      : [`No users active in the last ${mins} min — safe to deploy`];
    const admins = (d.active_total || 0) - d.active_users;
    if (admins > 0) {
      lines.push(`+${admins} admin${admins > 1 ? "s" : ""} online (not counted)`);
    }
    pill.title = lines.join("\n");
    pill.classList.toggle("busy", d.active_users > 0);
  } catch (_) {
    pill.textContent = "Active: ?";
    pill.title = "Couldn't reach /admin/api/active";
  }
}

function startActivePoll() {
  refreshActiveCount(); // re-render wiped the pill node; fill it now
  if (!ACTIVE_POLL_TIMER) {
    ACTIVE_POLL_TIMER = setInterval(refreshActiveCount, ACTIVE_POLL_MS);
  }
}

function stopActivePoll() {
  if (ACTIVE_POLL_TIMER) {
    clearInterval(ACTIVE_POLL_TIMER);
    ACTIVE_POLL_TIMER = null;
  }
}

// Hidden tabs skip polls; catch up the instant the admin looks back.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && ACTIVE_POLL_TIMER) refreshActiveCount();
});

function renderAccountChip() {
  const el = document.getElementById("account-chip");
  if (!el) return;
  if (!window.PLO5BP_PUBLIC || !UI.me || !UI.me.signed_in) {
    el.style.display = "none";
    stopActivePoll();
    syncAccountExtraTab(null);
    return;
  }
  const me = UI.me;
  el.style.display = "flex";
  // While the models are in development the whole site is free (/me.free_for_all):
  // no quota pill, no Upgrade button — just say so.
  const pill = me.is_admin
    ? `<span class="acct-pill pro">ADMIN</span>`
    : me.free_for_all
      ? `<span class="acct-pill pro" title="WrapGTO is free while the models are in development">FREE ACCESS</span>`
      : me.sub.active
        ? `<span class="acct-pill pro">${me.sub.source === "comp" ? "COMP" : "PRO"}</span>`
        : `<span class="acct-pill free">${me.free.left}/${me.free.limit} free today</span>`;
  // /me fields are account data (OAuth profile): escape before innerHTML, and
  // only ever load an http(s) avatar (review 2026-09-20 F17).
  const avatar = typeof me.picture === "string" && /^https?:\/\//i.test(me.picture)
    ? `<img class="acct-avatar" src="${escapeHTML(me.picture)}" alt="" referrerpolicy="no-referrer" />`
    : "";
  const email = escapeHTML(me.email);
  const who = me.name ? `${escapeHTML(me.name)} — ${email}` : email;
  const upgrade = me.sub.active
    ? ""
    : `<button class="acct-btn upgrade" id="acct-upgrade-btn" type="button">Upgrade</button>`;
  const manage = me.sub.active && me.sub.source === "stripe"
    ? `<button class="acct-btn" id="acct-manage-btn" type="button">Billing</button>`
    : "";
  const admin = me.is_admin
    ? `<span class="acct-pill activity" id="acct-active-pill" title="">Active: –</span><button class="acct-btn" id="acct-admin-btn" type="button">Admin</button>`
    : "";
  // Per-account extra section: label + destination come entirely from /me,
  // so nothing about it ships to accounts that weren't granted it.
  const extra = accountExtraLink(me);
  const extraBtn = extra
    ? `<button class="acct-btn" id="acct-extra-btn" type="button">${escapeHTML(extra.label)}</button>`
    : "";
  el.innerHTML = `${pill}${avatar}<span class="acct-email" title="${who}">${email}</span>${upgrade}${manage}${admin}${extraBtn}<button class="acct-btn" id="acct-signout-btn" type="button">Sign out</button>`;
  const up = document.getElementById("acct-upgrade-btn");
  if (up) up.addEventListener("click", () => startCheckout());
  const mg = document.getElementById("acct-manage-btn");
  if (mg) mg.addEventListener("click", async () => {
    try { const d = await postJSON("/billing/portal"); window.location.href = d.url; }
    catch (e) { showToast(e.message); }
  });
  const ad = document.getElementById("acct-admin-btn");
  if (ad) ad.addEventListener("click", () => window.open("/admin", "_blank"));
  const ex = document.getElementById("acct-extra-btn");
  if (ex) ex.addEventListener("click", () => { window.location.href = extra.href; });
  document.getElementById("acct-signout-btn").addEventListener("click", () => signOut());
  if (me.is_admin) startActivePoll();
  else stopActivePoll();
  syncAccountExtraTab(me);
}

// Sign-out is state-changing, so POST it (review 2026-09-20, public F-minor:
// "/auth/logout is a state-changing GET"); a server that only routes the GET
// answers 405, so any failure falls back to the plain navigation.
async function signOut() {
  try {
    const res = await fetch("/auth/logout", { method: "POST" });
    if (res.ok) { window.location.href = "/"; return; }
  } catch (_) { /* fall through */ }
  window.location.href = "/auth/logout";
}

// /me carries an optional {href, label} link object for accounts granted an
// extra section (absent otherwise). The label and destination are
// SERVER-supplied on purpose — this file ships to every visitor and must not
// name the section or its path (review 2026-09-20, public F1). The legacy
// boolean `true` carries no destination, so it renders nothing. Only a
// same-origin absolute path is accepted as a destination.
function accountExtraLink(me) {
  const hg = me && me.homegame;
  if (!hg || typeof hg !== "object") return null;
  const href = typeof hg.href === "string" ? hg.href : "";
  if (!/^\/(?!\/)/.test(href) || href.includes("\\")) return null;
  const label = typeof hg.label === "string" && hg.label.trim() ? hg.label.trim() : "More";
  return { href, label };
}

function syncAccountExtraTab(me) {
  const tabs = document.getElementById("mode-tabs");
  if (!tabs) return;
  let tab = document.getElementById("tab-extra");
  const extra = accountExtraLink(me);
  if (!extra) {
    if (tab) tab.remove();
    return;
  }
  if (!tab) {
    tab = document.createElement("button");
    tab.id = "tab-extra";
    tab.className = "mode-tab";
    tab.type = "button";
    tab.addEventListener("click", () => {
      if (tab.dataset.href) window.location.href = tab.dataset.href;
    });
    tabs.appendChild(tab);
  }
  tab.textContent = extra.label;
  tab.dataset.href = extra.href;
}

function showLoginOverlay(show) {
  const ov = document.getElementById("login-overlay");
  if (!ov) return;
  ov.style.display = show ? "block" : "none";
  if (!show) return;
  const me = UI.me || {};
  const g = document.getElementById("google-signin-btn");
  g.style.display = me.auth_configured === false ? "none" : "flex";
  const devRow = document.getElementById("dev-login-row");
  devRow.style.display = me.dev_login ? "flex" : "none";
  const freeHands = document.getElementById("gate-free-hands");
  if (freeHands && UI.me && UI.me.free) freeHands.textContent = UI.me.free.limit;
}

function showPaywall(body) {
  const modal = document.getElementById("paywall-modal");
  if (!modal) return;
  const title = document.getElementById("paywall-title");
  const msg = document.getElementById("paywall-msg");
  if (body && body.error === "free_limit") {
    title.textContent = "Daily free limit reached";
    const resets = body.resets_at ? new Date(body.resets_at) : null;
    const inH = resets ? Math.max(1, Math.round((resets - Date.now()) / 3600000)) : null;
    msg.textContent =
      `You've used your ${body.limit ?? 5} free trainer hands for today` +
      (inH ? ` — more in ~${inH}h.` : ".") +
      " Subscribe for unlimited hands plus full Study mode.";
  } else {
    title.textContent = "Subscription required";
    msg.textContent =
      (body && body.detail) ||
      "Study mode is part of the subscription. The trainer stays free for 5 hands a day.";
  }
  modal.style.display = "flex";
}

async function startCheckout() {
  try {
    const d = await postJSON("/billing/checkout");
    window.location.href = d.url;
  } catch (e) {
    showToast(e.message);
  }
}

function setupPublicUI() {
  const g = document.getElementById("google-signin-btn");
  if (g) g.addEventListener("click", () => { window.location.href = "/auth/login"; });
  // Landing page has additional sign-in CTAs (top bar, pricing card).
  for (const btn of document.querySelectorAll(".landing-signin")) {
    btn.addEventListener("click", () => { window.location.href = "/auth/login"; });
  }
  const devBtn = document.getElementById("dev-login-btn");
  if (devBtn) devBtn.addEventListener("click", () => {
    const email = document.getElementById("dev-login-email").value.trim();
    if (email) window.location.href = `/auth/dev?email=${encodeURIComponent(email)}`;
  });
  const sub = document.getElementById("paywall-subscribe-btn");
  if (sub) sub.addEventListener("click", () => startCheckout());
  const close = document.getElementById("paywall-close-btn");
  if (close) close.addEventListener("click", () => {
    document.getElementById("paywall-modal").style.display = "none";
  });
}

// The OAuth callback bounces a failed sign-in to /?login=failed. Say so —
// the landing page used to reload with no explanation (review 2026-09-20
// F17). The inline notice sits by the sign-in button; the toast covers a
// build without that markup.
function handleLoginReturn() {
  const params = new URLSearchParams(location.search);
  if (params.get("login") !== "failed") return;
  const msg = "Sign-in didn't go through — Google didn't return a verified "
    + "email for that account. Please try again.";
  const note = document.getElementById("login-error");
  if (note) {
    note.textContent = msg;
    note.hidden = false;
  }
  showToast(msg);
  params.delete("login");
  const qs = params.toString();
  history.replaceState(null, "", location.pathname + (qs ? `?${qs}` : ""));
}

async function handleCheckoutReturn() {
  const params = new URLSearchParams(location.search);
  const state = params.get("checkout");
  if (!state) return;
  if (state === "success" && params.get("session_id")) {
    try {
      const d = await getJSON(`/billing/confirm?session_id=${encodeURIComponent(params.get("session_id"))}`);
      if (d.active) {
        showToast("Subscription active — welcome aboard!", "info");
        await refreshMe();
      } else {
        showToast("Payment not confirmed yet; refresh in a moment.");
      }
    } catch (e) { showToast(e.message); }
  } else if (state === "cancel") {
    showToast("Checkout canceled.", "info");
  }
  params.delete("checkout"); params.delete("session_id");
  const qs = params.toString();
  history.replaceState(null, "", location.pathname + (qs ? `?${qs}` : ""));
}

// --- Trainer mode -------------------------------------------------------

function applyModeUI() {
  const trainer = UI.mode === "trainer";
  document.body.classList.toggle("trainer-mode", trainer);
  document.getElementById("tab-study").classList.toggle("active", !trainer);
  document.getElementById("tab-trainer").classList.toggle("active", trainer);
}

function setMode(mode) {
  if (UI.mode === mode) {
    // Coming back from the Ranges tab to the mode we were already in: the
    // tab bar still needs its highlight restored — Ranges cleared it
    // (review 2026-09-20 F17).
    applyModeUI();
    return;
  }
  // Public build: Study is subscriber-only — offer the upgrade instead of
  // switching into a tab whose routes will 402.
  if (mode === "study" && window.PLO5BP_PUBLIC && UI.me && UI.me.signed_in && !isEntitled()) {
    showPaywall({ error: "subscription_required" });
    return;
  }
  UI.mode = mode;
  lsSet("plo5bp-mode", mode);
  // New context (review 2026-09-20 F4): in-flight responses of the old mode
  // go stale, a running frame animation stops, and the old mode's controls /
  // panels must not stay clickable while the new state loads.
  UI.ctxSeq++;
  UI.animSeq++;
  UI.lastState = null;
  UI.lastStateKey = null;
  UI.selectedSlot = null;
  UI.reviewDecision = null;
  UI.reviewNode = null;
  cancelTrainerPick();
  resetRaiseEntry();
  clearActionControls();
  const reviewPanel = document.getElementById("review-panel");
  if (reviewPanel) reviewPanel.hidden = true;
  hideFeedbackFlash();
  applyModeUI();
  fetchState();
}

function cancelTrainerPick() {
  UI.trainerPick = false;
  document.body.classList.remove("trainer-pick");
}

function hideFeedbackFlash() {
  const el = document.getElementById("feedback-flash");
  if (el) el.hidden = true;
}

function renderFeedbackFlash(s, force = false) {
  // During frame playback the frames carry the previous decision's
  // feedback — only the explicit (forced) call may flash.
  if (UI.animating && !force) return;
  const fb = s.trainer.feedback;
  if (!fb) return;
  const key = `${s.trainer.hand_no}:${fb.decision_idx}`;
  if (key === UI.feedbackShownIdx) return;
  UI.feedbackShownIdx = key;
  const el = document.getElementById("feedback-flash");
  document.getElementById("feedback-marks").textContent = fb.marks;
  // Raise sizes are raise-BY deltas; show raise-TO totals like the raise
  // input did (review 2026-09-20 F11). Hero's street commit at the decision:
  // a server-sent `actor_commit_chips` when present, else what postAction
  // recorded for this hand just before POSTing.
  const rec0 = UI.actCommit && UI.actCommit.handNo === s.trainer.hand_no ? UI.actCommit : null;
  const commit = typeof fb.actor_commit_chips === "number" ? fb.actor_commit_chips
    : (rec0 ? rec0.commit : null);
  const street = fb.street || (rec0 ? rec0.street : null);
  const userLabel = fb.user_gate
    ? gateActionLabel(fb.user_gate, fb.user_chips, fb.to_call_chips, s, commit, street)
    : fb.label;
  document.getElementById("feedback-text").textContent =
    `${userLabel} · ${Math.round(fb.score)}%`;
  const bits = [];
  const recLabel = fb.rec_gate
    ? gateActionLabel(fb.rec_gate, fb.rec_chips, fb.to_call_chips, s, commit, street)
    : fb.rec_label;
  if (fb.category !== "best" && recLabel) bits.push(`best: ${recLabel}`);
  if (fb.ev_loss_hidden === true) {
    // Mid-hand the backend withholds the number (it is computed from cards
    // the hero can't see yet) and sends ev_loss_bb: null — never render that
    // as "NaN" / "0.00".
    bits.push("EV loss shown at hand end");
  } else if (typeof fb.ev_loss_bb === "number" && isFinite(fb.ev_loss_bb) && fb.ev_loss_bb > 0) {
    bits.push(`EV −${fb.ev_loss_bb.toFixed(2)}bb`);
  }
  document.getElementById("feedback-sub").textContent = bits.join(" · ");
  el.className = `flash-${fb.category}`;
  el.hidden = false;
  if (UI.feedbackTimer) clearTimeout(UI.feedbackTimer);
  UI.feedbackTimer = setTimeout(() => { el.hidden = true; }, 2000);
}

const CAT_LABELS = [
  ["best", "Best move"],
  ["correct", "Correct"],
  ["inaccuracy", "Inaccuracy"],
  ["wrong", "Wrong move"],
  ["blunder", "Blunder"],
];

function statsBlockHTML(title, st, scope) {
  const moves = st.moves || 0;
  const rows = CAT_LABELS.map(([k, label]) => {
    const c = (st.cat_counts && st.cat_counts[k]) || 0;
    const pct = moves > 0 ? (100 * c) / moves : 0;
    return `<div class="stat-cat-row">
      <span class="stat-cat-count">${c}</span>
      <div class="rec-dist-track"><div class="rec-dist-fill cat-${k}" style="width:${pct.toFixed(1)}%"></div></div>
      <span class="stat-cat-label">${label}</span>
    </div>`;
  }).join("");
  const score = (st.gto_score !== null && st.gto_score !== undefined)
    ? `${st.gto_score.toFixed(1)}%` : "—";
  const evTotal = st.ev_loss_total_bb ?? 0;
  const evHand = st.ev_loss_per_hand_bb;
  const evLine = `EV loss ${evTotal.toFixed(2)}bb total` +
    ((evHand !== null && evHand !== undefined) ? ` · ${evHand.toFixed(2)}bb / hand` : "");
  return `
    <div class="stats-title"><span class="stats-chev">&#9662;</span>${title}
      <button class="stats-reset" data-scope="${scope}" type="button">Reset</button>
    </div>
    <div class="stats-top">
      <div><span class="stats-num">${st.hands ?? 0}</span><span class="stats-cap">hands</span></div>
      <div><span class="stats-num">${moves}</span><span class="stats-cap">moves</span></div>
      <div><span class="stats-num stats-score">${score}</span><span class="stats-cap">GTO score</span></div>
    </div>
    ${rows}
    <div class="stats-ev muted">${evLine}</div>
  `;
}

// Collapse state for the SESSION/LIFETIME blocks. Default: expanded on
// desktop, collapsed on small screens (they'd otherwise push the whole
// column down); user toggles persist either way.
function statsCollapsePrefs() {
  try { return JSON.parse(localStorage.getItem("plo5bp-stats-collapsed")) || {}; }
  catch (_) { return {}; }
}

function statsCollapsed(scope) {
  const prefs = statsCollapsePrefs();
  if (typeof prefs[scope] === "boolean") return prefs[scope];
  return isMobile();
}

function toggleStatsBlock(scope) {
  const prefs = statsCollapsePrefs();
  prefs[scope] = !statsCollapsed(scope);
  lsSet("plo5bp-stats-collapsed", JSON.stringify(prefs));
  const el = document.getElementById(scope === "session" ? "stats-session" : "stats-lifetime");
  if (el) el.classList.toggle("collapsed", prefs[scope]);
}

function renderTrainerStats(s) {
  const stats = s.trainer.stats || {};
  const se = document.getElementById("stats-session");
  const lt = document.getElementById("stats-lifetime");
  se.innerHTML = statsBlockHTML("Session", stats.session || {}, "session");
  lt.innerHTML = statsBlockHTML("Lifetime", stats.lifetime || {}, "lifetime");
  se.classList.toggle("collapsed", statsCollapsed("session"));
  lt.classList.toggle("collapsed", statsCollapsed("lifetime"));
  for (const btn of document.querySelectorAll("#trainer-stats-panel .stats-reset")) {
    btn.addEventListener("click", () => {
      const scope = btn.dataset.scope;
      if (scope === "lifetime" &&
          !confirm("Reset lifetime stats? This clears the saved stats file.")) {
        return;
      }
      postTrainer("stats/reset", { scope });
    });
  }
}

function ringClass(pct) {
  return pct >= 80 ? "ring-good" : pct >= 50 ? "ring-mid" : "ring-bad";
}

function renderReviewPanel(s) {
  const panel = document.getElementById("review-panel");
  const rv = s.trainer.review;
  if (!rv) {
    panel.hidden = true;
    UI.reviewDecision = null;
    return;
  }
  panel.hidden = false;
  UI.reviewDecision = rv.decision;
  UI.reviewNode = rv.node;

  const C = 2 * Math.PI * 26;
  const pct = rv.hand_score ?? 0;
  const fill = document.getElementById("review-ring-fill");
  fill.style.strokeDasharray = `${((C * pct) / 100).toFixed(1)} ${C.toFixed(1)}`;
  fill.setAttribute("class", `ring-fill ${ringClass(pct)}`);
  document.getElementById("review-score-num").textContent =
    (rv.hand_score !== null && rv.hand_score !== undefined)
      ? `${Math.round(rv.hand_score)}%` : "—";

  // Arrows + label step the unified NODE cursor (every seat's decision).
  const nc = rv.node_current;
  document.getElementById("review-step-label").textContent = nc
    ? `Node ${rv.node + 1} / ${rv.num_nodes} — ${nc.position}${nc.is_hero ? " (hero)" : ""}`
    : `Node ${rv.node + 1} / ${rv.num_nodes}`;
  const atStart = rv.node <= 0;
  const atEnd = rv.node >= rv.num_nodes - 1;
  document.getElementById("review-first").disabled = atStart;
  document.getElementById("review-prev").disabled = atStart;
  document.getElementById("review-next").disabled = atEnd;
  document.getElementById("review-last").disabled = atEnd;

  // Pills stay one-per-hero-decision; clicking one drives the shared node
  // cursor. Highlight the pill whose node is the current cursor.
  const chips = document.getElementById("review-chips");
  chips.innerHTML = "";
  // Labels are rebuilt client-side as raise-TO totals in the active unit; the
  // server's *_label strings (bb-only, raise-BY) are only the fallback
  // (review 2026-09-20 F11).
  const walk = reviewNodeCommits(s, rv);
  rv.decisions.forEach((d) => {
    const b = document.createElement("button");
    b.type = "button";
    const isCurrent = d.node_idx === rv.node;
    b.className = `review-chip cat-border-${d.category}` + (isCurrent ? " current" : "");
    const nd = rv.nodes ? rv.nodes[d.node_idx] : null;
    const w = walk[d.node_idx];
    const label = nd && w && nd.actual_gate
      ? gateActionLabel(
          nd.actual_gate, nd.actual_chips,
          typeof d.to_call_chips === "number" ? d.to_call_chips : Math.max(0, w.level - w.before),
          s, reviewCommitBefore(s, rv, d.node_idx, nd.seat, d, walk), nd.street)
      : d.user_label;
    b.innerHTML = `<span class="rc-street">${d.street}</span>${label}` +
      ` <span class="rc-score">${Math.round(d.score)}%</span>`;
    b.addEventListener("click", () => trainerReviewGotoNode(d.node_idx));
    chips.appendChild(b);
  });

  document.getElementById("review-whatif-bar").hidden = !rv.whatif;

  const cur = rv.node_current || rv.current;
  const detail = document.getElementById("review-detail");
  if (!cur) { detail.innerHTML = ""; return; }
  const commit = reviewCurCommit(s, rv, cur, walk);
  const curLabel = (gate, gateChips, fallback) => (gate
    ? gateActionLabel(gate, gateChips, cur.to_call_chips, s, commit, cur.street)
    : fallback);
  const recLabel = curLabel(cur.rec_gate, cur.rec_chips, cur.rec_label);
  if (reviewNodeUngraded(rv.node_current)) {
    // Ungraded node — an opponent's decision, or a hero moot auto-check
    // (`category: null`; reading cur.category.toUpperCase() there threw and
    // blanked the panel — review 2026-09-20 F12). Show what actually happened
    // vs the network's deterministic pick (policy + EVs ride in the
    // recommendation panel). No EV-loss, no what-if hint.
    const who = `${cur.position}${cur.is_hero ? " (hero)" : ""}`;
    const moves = cur.is_hero
      ? "Automatic check — betting was closed (all-in run-out), so there is nothing to grade."
      : `Actual: <b>${curLabel(cur.actual_gate, cur.actual_chips, cur.actual_label)}</b>` +
        ` · Network: <b>${recLabel}</b>`;
    detail.innerHTML = `
      <div class="review-villain">${who} · ${cur.street} decision</div>
      <div class="review-moves">${moves}</div>
    `;
    return;
  }
  const num = (v) => typeof v === "number" && isFinite(v);
  let evRow = "";
  if (num(cur.ev_loss_bb)) {
    const detailBit = (num(cur.ev_user_bb) && num(cur.ev_best_bb))
      ? ` <span class="muted">(you ${cur.ev_user_bb.toFixed(2)} vs best ${cur.ev_best_bb.toFixed(2)})</span>`
      : "";
    evRow = `<div class="review-ev">EV loss <b>${cur.ev_loss_bb.toFixed(2)}bb</b>${detailBit}</div>`;
  }
  const rescored = rv.whatif
    ? `<div class="review-rescored">What-if rescore: <b class="cat-text-${rv.whatif.rescored.category}">` +
      `${rv.whatif.rescored.category}</b> ${Math.round(rv.whatif.rescored.score)}%</div>`
    : "";
  detail.innerHTML = `
    <div class="review-verdict cat-text-${cur.category}">${cur.marks} ${String(cur.category).toUpperCase()} · ${Math.round(cur.score)}%</div>
    <div class="review-moves">You: <b>${curLabel(cur.user_gate, cur.user_chips, cur.user_label)}</b> · Network: <b>${recLabel}</b></div>
    ${evRow}${rescored}
    <div class="muted review-hint">Click a board or hero card to try a what-if swap.</div>
  `;
}

// --- Trainer settings modal ----------------------------------------------

function _tsVal(id) { return document.getElementById(id).value; }
function _tsNum(id) { return parseFloat(document.getElementById(id).value); }
function _tsInt(id) { return parseInt(document.getElementById(id).value, 10); }
function _tsShow(id, on) { document.getElementById(id).style.display = on ? "" : "none"; }

// Server-advertised ceiling for settings.mc_rollouts (null when the payload
// doesn't carry one — older servers).
function trainerMcRolloutsMax(s) {
  const m = s && s.trainer ? s.trainer.mc_rollouts_max : null;
  return typeof m === "number" && isFinite(m) && m >= 0 ? Math.floor(m) : null;
}
// Clamp a typed rollout count into [0, ceiling]; a non-number passes through
// untouched so the server's validation message still reaches the user.
function clampMcRollouts(n, s) {
  if (!Number.isFinite(n)) return n;
  const mcMax = trainerMcRolloutsMax(s);
  const lo = Math.max(0, n);
  return mcMax !== null ? Math.min(lo, mcMax) : lo;
}

function syncSettingsVisibility() {
  const seatsMode = _tsVal("ts-seats-mode");
  _tsShow("ts-seats-fixed-wrap", seatsMode === "fixed");
  _tsShow("ts-seats-range-wrap", seatsMode === "random");
  const stacksMode = _tsVal("ts-stacks-mode");
  _tsShow("ts-stack-fixed-wrap", stacksMode === "fixed");
  _tsShow("ts-stack-range-wrap", stacksMode === "random");
  _tsShow("ts-per-seat-wrap", stacksMode === "per_seat");
  _tsShow("ts-hero-kth-wrap", _tsVal("ts-hero-mode") === "kth");
}

function openTrainerSettings() {
  const s = UI.lastState;
  const t = s && s.trainer ? s.trainer.settings : null;
  if (!t) return;
  document.getElementById("ts-seats-mode").value = t.seats_mode;
  document.getElementById("ts-seats-fixed").value = t.seats_fixed;
  document.getElementById("ts-seats-min").value = t.seats_min;
  document.getElementById("ts-seats-max").value = t.seats_max;
  document.getElementById("ts-stacks-mode").value = t.stacks_mode;
  document.getElementById("ts-stack-bb").value = t.stack_bb;
  document.getElementById("ts-stack-min-bb").value = t.stack_min_bb;
  document.getElementById("ts-stack-max-bb").value = t.stack_max_bb;
  document.getElementById("ts-hero-mode").value = t.hero_position_mode;
  document.getElementById("ts-hero-kth").value = String(t.hero_kth);
  document.getElementById("ts-ante-bb").value = t.ante_bb;
  // The build's ceiling on EV rollouts (`trainer.mc_rollouts_max`, e.g. 32 on
  // the public build) bounds the field; the server clamps too, this just
  // keeps the form honest about what will be used.
  const mcInput = document.getElementById("ts-mc-rollouts");
  const mcMax = trainerMcRolloutsMax(s);
  if (mcMax !== null) {
    mcInput.max = String(mcMax);
    mcInput.title = "Monte-Carlo rollouts per EV-loss estimate; 0 disables EV loss"
      + ` (max ${mcMax})`;
  }
  mcInput.value = mcMax !== null ? Math.min(t.mc_rollouts, mcMax) : t.mc_rollouts;
  document.getElementById("ts-dollars-bb").value = t.dollars_per_bb;
  document.getElementById("ts-anim-ms").value = String(trainerPrefs.animMs);
  document.getElementById("ts-anim-ms-range").value = String(
    Math.min(trainerPrefs.animMs, 4000)
  );
  document.getElementById("ts-ff-fold").checked = trainerPrefs.ffFold;
  const wrap = document.getElementById("ts-per-seat");
  wrap.innerHTML = "";
  for (let i = 0; i < 6; i++) {
    const [lo, hi] = t.stacks_per_seat_bb[i] || [20, 20];
    const row = document.createElement("div");
    row.className = "ts-seat-row";
    row.innerHTML = `<span>Seat ${i + 1}</span>
      <input type="number" class="ts-ps-lo" data-i="${i}" min="1" max="1000" step="0.5" value="${lo}" /> –
      <input type="number" class="ts-ps-hi" data-i="${i}" min="1" max="1000" step="0.5" value="${hi}" />`;
    wrap.appendChild(row);
  }
  syncSettingsVisibility();
  UI.settingsOpen = true;
  document.getElementById("trainer-settings-modal").hidden = false;
}

function closeTrainerSettings() {
  UI.settingsOpen = false;
  document.getElementById("trainer-settings-modal").hidden = true;
}

async function saveTrainerSettings() {
  const perSeat = [];
  for (let i = 0; i < 6; i++) {
    const lo = parseFloat(document.querySelector(`.ts-ps-lo[data-i="${i}"]`).value);
    const hi = parseFloat(document.querySelector(`.ts-ps-hi[data-i="${i}"]`).value);
    perSeat.push([lo, hi]);
  }
  const body = {
    seats_mode: _tsVal("ts-seats-mode"),
    seats_fixed: _tsInt("ts-seats-fixed"),
    seats_min: _tsInt("ts-seats-min"),
    seats_max: _tsInt("ts-seats-max"),
    stacks_mode: _tsVal("ts-stacks-mode"),
    stack_bb: _tsNum("ts-stack-bb"),
    stack_min_bb: _tsNum("ts-stack-min-bb"),
    stack_max_bb: _tsNum("ts-stack-max-bb"),
    stacks_per_seat_bb: perSeat,
    hero_position_mode: _tsVal("ts-hero-mode"),
    hero_kth: _tsInt("ts-hero-kth"),
    ante_bb: _tsNum("ts-ante-bb"),
    mc_rollouts: clampMcRollouts(_tsInt("ts-mc-rollouts"), UI.lastState),
    dollars_per_bb: _tsNum("ts-dollars-bb"),
  };
  // Playback prefs are client-only (global, format-independent) — persist to
  // localStorage and apply live, independent of the server settings POST.
  let animMs = parseInt(document.getElementById("ts-anim-ms").value, 10);
  if (!Number.isFinite(animMs) || animMs < 0) animMs = TRAINER_ANIM_DEFAULT_MS;
  animMs = Math.min(animMs, TRAINER_ANIM_MAX_MS);
  trainerPrefs.animMs = animMs;
  trainerPrefs.ffFold = document.getElementById("ts-ff-fold").checked;
  lsSet("plo5bp-trainer-anim-ms", String(trainerPrefs.animMs));
  lsSet("plo5bp-trainer-ff-fold", trainerPrefs.ffFold ? "true" : "false");

  const ok = await postTrainer("settings", body);
  if (ok) closeTrainerSettings();
}

function setupTrainerControls() {
  document.getElementById("tab-study").addEventListener("click", () => setMode("study"));
  document.getElementById("tab-trainer").addEventListener("click", () => setMode("trainer"));
  // Both share the in-flight guard: the public build's free-tier middleware
  // counts every POST /trainer/new_hand, so an unguarded double-click burns
  // 2 of 5 daily hands while showing one. `postTrainer` never rejects
  // (its own try/catch), so clearing in `.finally` is always reached.
  const dealGuarded = (path) => {
    if (UI.actionInFlight) return;
    cancelTrainerPick();
    UI.reviewDecision = null;
    UI.reviewNode = null;
    UI.selectedSlot = null;
    UI.actionInFlight = true;
    setActionsBusy(true);
    postTrainer(path).finally(() => {
      UI.actionInFlight = false;
      setActionsBusy(false);
    });
  };
  const newHand = () => dealGuarded("new_hand");
  const repeatHand = () => dealGuarded("repeat");
  document.getElementById("trainer-new-hand-btn").addEventListener("click", newHand);
  document.getElementById("trainer-repeat-btn").addEventListener("click", repeatHand);
  document.getElementById("review-next-hand").addEventListener("click", newHand);
  document.getElementById("review-repeat-hand").addEventListener("click", repeatHand);
  document.getElementById("review-first").addEventListener("click", () => {
    const rv = UI.lastState?.trainer?.review;
    if (rv && rv.node > 0) trainerReviewGotoNode(0);
  });
  document.getElementById("review-prev").addEventListener("click", () => {
    const rv = UI.lastState?.trainer?.review;
    if (rv && rv.node > 0) trainerReviewGotoNode(rv.node - 1);
  });
  document.getElementById("review-next").addEventListener("click", () => {
    const rv = UI.lastState?.trainer?.review;
    if (rv && rv.node < rv.num_nodes - 1) trainerReviewGotoNode(rv.node + 1);
  });
  document.getElementById("review-last").addEventListener("click", () => {
    const rv = UI.lastState?.trainer?.review;
    if (rv && rv.node < rv.num_nodes - 1) trainerReviewGotoNode(rv.num_nodes - 1);
  });
  document.getElementById("review-whatif-reset").addEventListener("click", () => {
    const rv = UI.lastState?.trainer?.review;
    if (rv) trainerReviewGotoNode(rv.node);
  });
  document.getElementById("trainer-settings-btn").addEventListener("click", openTrainerSettings);
  document.getElementById("ts-cancel").addEventListener("click", closeTrainerSettings);
  document.getElementById("ts-save").addEventListener("click", saveTrainerSettings);
  for (const id of ["ts-seats-mode", "ts-stacks-mode", "ts-hero-mode"]) {
    document.getElementById(id).addEventListener("change", syncSettingsVisibility);
  }
  // Opponent-speed slider and text box mirror each other. The slider caps at
  // 4000ms; the box accepts up to 8000 for the patient.
  const animRange = document.getElementById("ts-anim-ms-range");
  const animNum = document.getElementById("ts-anim-ms");
  animRange.addEventListener("input", () => { animNum.value = animRange.value; });
  animNum.addEventListener("input", () => {
    const v = parseInt(animNum.value, 10);
    if (Number.isFinite(v)) animRange.value = String(Math.min(Math.max(v, 0), 4000));
  });
  document.getElementById("trainer-settings-modal").addEventListener("pointerdown", (e) => {
    if (e.target === e.currentTarget) closeTrainerSettings();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (UI.settingsOpen) { closeTrainerSettings(); return; }
    if (UI.trainerPick) {
      UI.selectedSlot = null;
      cancelTrainerPick();
      if (UI.lastState) render(UI.lastState);
    }
  });
}

setupCardKeyboard();

function setupRaiseInput() {
  const input = document.getElementById("raise-input");
  input.addEventListener("input", () => { UI.raiseUserSet = true; });
  input.addEventListener("focus", () => {
    setTimeout(() => input.select(), 0);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    e.preventDefault();
    const submit = document.getElementById("raise-submit");
    const section = document.getElementById("raise-section");
    if (submit.disabled || section.hidden) return;
    submit.click();
  });
}

async function init() {
  // Signed-out public visitors receive landing-only markup (the app chrome
  // between the WGAPP markers is stripped server-side). Wire just the gate
  // and bail BEFORE the app setups below, which query app-chrome elements
  // that don't exist in that markup. setupPublicUI is null-guarded and only
  // touches landing/paywall elements, so it's safe to run first.
  if (window.PLO5BP_PUBLIC) {
    setupPublicUI();
    handleLoginReturn();
    await refreshMe();
    if (!UI.me || !UI.me.signed_in) {
      showLoginOverlay(true);
      return;
    }
  }
  setupTopBar();
  setupDealerDrag();
  setupInsertHover();
  setupRaiseInput();
  setupTrainerControls();
  setupCardPicker();
  // Stats collapse: delegated (block innerHTML is replaced every render)
  document.getElementById("trainer-stats-panel").addEventListener("click", (e) => {
    if (e.target.closest(".stats-reset")) return;
    const title = e.target.closest(".stats-title");
    if (!title) return;
    const block = title.closest(".stats-block");
    toggleStatsBlock(block && block.id === "stats-lifetime" ? "lifetime" : "session");
  });
  if (window.PLO5BP_PUBLIC) {
    await handleCheckoutReturn();
    // Free accounts land in the trainer (Study is subscriber-only).
    if (!isEntitled() && UI.mode === "study") UI.mode = "trainer";
  }
  applyModeUI();
  initFormats();
  fetchState();
  if (window.__wgLiveInit) window.__wgLiveInit();
}

// WGLIVE:START
// Live-capture client code: ClubGG pixel-OCR controls + PokerNow DOM-bridge
// status (full LOCAL build only). The public server strips everything
// between the WGLIVE markers before serving this file, so none of this —
// names, selectors, endpoints — exists in what a public visitor downloads.
Object.assign(UI, {
  ocrRunning: false,
  ocrPollTimer: null,
  ocrLastStatus: null,
  ocrToggleBusy: false,
  ocrPollMs: 200,
  ocrWindowMatch: "",
  ocrMenuOpen: false,
  simpleOcrMode: true,
  simpleOcrToggleBusy: false,
  // Live-capture source: "clubgg" (pixel OCR) or "pokernow" (browser DOM bridge).
  liveSource: (() => {
    const s = lsGet("plo5bp-live-source");
    return s === "pokernow" ? "pokernow" : "clubgg";
  })(),
  pokernowPollTimer: null,
  // In-flight guards: setInterval keeps firing while a slow tick is still
  // awaiting, which stacked overlapping status+state fetches whose responses
  // could land out of order (review 2026-09-20 F6).
  ocrPollBusy: false,
  pokernowPollBusy: false,
});

// Live capture drives the STUDY session. In trainer mode fetchState() would
// hit the trainer's state route instead: pointless, and every such applyState
// cancelled the opponent-action playback (review 2026-09-20 F6).
async function livePollFetchState() {
  if (UI.mode !== "study") return false;
  return fetchState();
}

async function postRescan(target) {
  try {
    const data = await postJSON("/ocr/rescan", { target });
    applyState(data.state);
  } catch (e) {
    showToast(`Rescan ${target} failed: ${e.message}`);
  }
}

// --- OCR ---------------------------------------------------------------

function setOcrStatusError(msg) {
  const el = document.getElementById("ocr-status");
  el.textContent = msg;
  el.classList.add("error");
  el.classList.remove("muted");
}

function clearOcrStatusError() {
  const el = document.getElementById("ocr-status");
  el.classList.remove("error");
  el.classList.add("muted");
}

// Custom window picker (replaces the native <select> + Refresh button).
// Clicking the button fetches a *fresh* window list and shows a popup menu;
// there is no background polling — the only fetch is the one this click
// triggers. Mirrors the openStackEditor overlay idiom (DOM-mutation popup,
// Esc / click-outside dismissal).

const PICK_WINDOW_LABEL = "— pick window —";

function setOcrWindowSelection(match) {
  UI.ocrWindowMatch = match || "";
  const btn = document.getElementById("ocr-window-button");
  if (btn) {
    btn.textContent = UI.ocrWindowMatch || PICK_WINDOW_LABEL;
    btn.title = UI.ocrWindowMatch || "Pick the window to screen-read";
  }
}

function closeOcrWindowPicker() {
  UI.ocrMenuOpen = false;
  const picker = document.querySelector(".ocr-window-picker");
  const menu = picker ? picker.querySelector(".ocr-window-menu") : null;
  if (menu) menu.remove();
  const btn = document.getElementById("ocr-window-button");
  if (btn) btn.setAttribute("aria-expanded", "false");
  document.removeEventListener("pointerdown", onOcrPickerOutside, true);
  document.removeEventListener("keydown", onOcrPickerKey, true);
}

function onOcrPickerOutside(e) {
  const picker = document.querySelector(".ocr-window-picker");
  if (picker && !picker.contains(e.target)) closeOcrWindowPicker();
}

function onOcrPickerKey(e) {
  if (e.key === "Escape") { e.preventDefault(); closeOcrWindowPicker(); }
}

async function openOcrWindowPicker() {
  if (UI.ocrMenuOpen) { closeOcrWindowPicker(); return; }
  const picker = document.querySelector(".ocr-window-picker");
  const btn = document.getElementById("ocr-window-button");
  if (!picker || !btn) return;

  let titles = [];
  try {
    const data = await getJSON("/ocr/windows");
    titles = data.windows || [];
    clearOcrStatusError();
  } catch (e) {
    setOcrStatusError(`Window list failed: ${e.message}`);
    return;
  }
  // A late click that resolved after another open/close — bail if stale.
  if (UI.ocrMenuOpen) return;

  const menu = document.createElement("div");
  menu.className = "ocr-window-menu";
  menu.setAttribute("role", "listbox");

  const addItem = (label, value, cls) => {
    const row = document.createElement("div");
    row.className = "item" + (cls ? ` ${cls}` : "");
    row.textContent = label;
    if (cls !== "empty") {
      row.title = label;
      row.addEventListener("click", () => {
        setOcrWindowSelection(value);
        closeOcrWindowPicker();
      });
    }
    menu.appendChild(row);
  };

  addItem(PICK_WINDOW_LABEL, "", "placeholder");
  if (titles.length === 0) {
    addItem("(no windows found)", "", "empty");
  } else {
    for (const t of titles) addItem(t, t, null);
  }

  picker.appendChild(menu);
  UI.ocrMenuOpen = true;
  btn.setAttribute("aria-expanded", "true");
  document.addEventListener("pointerdown", onOcrPickerOutside, true);
  document.addEventListener("keydown", onOcrPickerKey, true);
}

async function toggleOcr() {
  if (UI.ocrToggleBusy) return;
  UI.ocrToggleBusy = true;
  try {
    if (UI.ocrRunning) await stopOcr();
    else await startOcr();
  } finally {
    UI.ocrToggleBusy = false;
  }
}

async function startOcr() {
  const match = UI.ocrWindowMatch;
  if (!match) {
    setOcrStatusError("Pick a window first");
    return;
  }
  clearOcrStatusError();
  try {
    const data = await postJSON("/ocr/start", {
      window_match: match,
      poll_ms: UI.ocrPollMs,
    });
    UI.ocrRunning = true;
    UI.ocrLastStatus = data.status;
    setOcrToggleUI();
    renderOcrStatus(data.status);
    startOcrPolling();
  } catch (e) {
    setOcrStatusError(`Start failed: ${e.message}`);
    showToast(`OCR start failed: ${e.message}`);
  }
}

async function stopOcr() {
  try {
    const data = await postJSON("/ocr/stop", {});
    UI.ocrLastStatus = data.status;
  } catch (e) {
    showToast(`OCR stop failed: ${e.message}`);
  } finally {
    UI.ocrRunning = false;
    setOcrToggleUI();
    renderOcrStatus(UI.ocrLastStatus);
    stopOcrPolling();
  }
}

function setOcrToggleUI() {
  const btn = document.getElementById("ocr-toggle");
  btn.textContent = UI.ocrRunning ? "On" : "Off";
  btn.setAttribute("aria-pressed", UI.ocrRunning ? "true" : "false");
  btn.classList.toggle("active", UI.ocrRunning);
}

async function saveOcrFrame() {
  const btn = document.getElementById("ocr-save-frame");
  if (btn) btn.disabled = true;
  try {
    const data = await postJSON("/ocr/save_frame", {});
    const name = (data.path || "").split(/[\\/]/).pop() || "frame";
    const sz = data.frame_size ? ` (${data.frame_size.width}x${data.frame_size.height})` : "";
    showToast(`Saved ${name}${sz}`, "info");
  } catch (e) {
    showToast(`Save frame failed: ${e.message}`);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function toggleSimpleOcr() {
  if (UI.simpleOcrToggleBusy) return;
  UI.simpleOcrToggleBusy = true;
  try {
    const next = !UI.simpleOcrMode;
    const data = await postJSON("/ocr/simple", { enabled: next });
    applyState(data.state);
  } catch (e) {
    showToast(`Simple OCR toggle failed: ${e.message}`);
  } finally {
    UI.simpleOcrToggleBusy = false;
  }
}

function setSimpleOcrToggleUI() {
  const btn = document.getElementById("ocr-simple-toggle");
  if (!btn) return;
  btn.textContent = UI.simpleOcrMode ? "Simple: On" : "Simple: Off";
  btn.setAttribute("aria-pressed", UI.simpleOcrMode ? "true" : "false");
  btn.classList.toggle("active", UI.simpleOcrMode);
  const grp = document.getElementById("ocr-rescan-group");
  if (grp) grp.style.display = UI.simpleOcrMode ? "" : "none";
}

function renderOcrStatus(st) {
  const el = document.getElementById("ocr-status");
  if (!st) { el.textContent = ""; clearOcrStatusError(); return; }
  if (st.running) {
    const title = st.window_title ? ` · ${st.window_title}` : "";
    const err = st.last_error ? ` · ${st.last_error}` : "";
    el.textContent = `frames ${st.frames_seen} · events ${st.events_applied}${title}${err}`;
    if (st.last_error) setOcrStatusError(el.textContent);
    else clearOcrStatusError();
  } else if (st.last_error) {
    setOcrStatusError(st.last_error);
  } else {
    el.textContent = "";
    clearOcrStatusError();
  }
}

function startOcrPolling() {
  stopOcrPolling();
  UI.ocrPollTimer = setInterval(async () => {
    if (UI.ocrPollBusy) return;  // previous tick still in flight
    UI.ocrPollBusy = true;
    try {
      const st = await getJSON("/ocr/status");
      const wasRunning = UI.ocrRunning;
      UI.ocrLastStatus = st;
      UI.ocrRunning = !!st.running;
      setOcrToggleUI();
      renderOcrStatus(st);
      if (st.running) {
        await livePollFetchState();
      } else {
        // Auto-off because the captured window was closed: reset the picker
        // back to the placeholder (a manual Off leaves the selection intact).
        if (wasRunning && st.stopped_reason === "window_closed") {
          setOcrWindowSelection("");
          showToast("OCR stopped — window closed", "info");
        }
        stopOcrPolling();
      }
    } catch (_) { /* ignore transient polling errors */ }
    finally { UI.ocrPollBusy = false; }
  }, 500);
}

function stopOcrPolling() {
  if (UI.ocrPollTimer !== null) {
    clearInterval(UI.ocrPollTimer);
    UI.ocrPollTimer = null;
  }
}

async function refreshOcrStatusOnLoad() {
  try {
    const st = await getJSON("/ocr/status");
    UI.ocrLastStatus = st;
    UI.ocrRunning = !!st.running;
    setOcrToggleUI();
    renderOcrStatus(st);
    // Reflect an already-running session in the picker label (e.g. after a
    // browser reload while OCR is on).
    if (st.running && st.window_match) setOcrWindowSelection(st.window_match);
    if (st.running) startOcrPolling();
  } catch (_) { /* ocr endpoints may be unavailable; ignore */ }
}

// --- Live-capture source (ClubGG OCR vs PokerNow browser bridge) --------

function setSourceToggleUI() {
  const btn = document.getElementById("source-toggle");
  if (btn) btn.textContent = UI.liveSource === "pokernow" ? "Source: PokerNow" : "Source: ClubGG";
  const clubgg = document.getElementById("clubgg-controls");
  const pokernow = document.getElementById("pokernow-controls");
  const isPn = UI.liveSource === "pokernow";
  if (clubgg) clubgg.style.display = isPn ? "none" : "";
  if (pokernow) pokernow.style.display = isPn ? "" : "none";
}

// Apply the selected source: show its controls and run only its status poll.
// ClubGG and PokerNow are mutually exclusive (the server refuses a PokerNow
// connection while OCR is running), so switching to PokerNow stops any live
// OCR capture first.
async function applySourceUI() {
  setSourceToggleUI();
  if (UI.liveSource === "pokernow") {
    if (UI.ocrRunning) await stopOcr();
    stopOcrPolling();
    startPokernowPolling();
  } else {
    stopPokernowPolling();
    refreshOcrStatusOnLoad();
  }
}

function toggleLiveSource() {
  UI.liveSource = UI.liveSource === "pokernow" ? "clubgg" : "pokernow";
  lsSet("plo5bp-live-source", UI.liveSource);
  applySourceUI();
}

function showPokerNowHelp() {
  alert(
    "PokerNow browser bridge\n" +
    "\n" +
    "1. Install the Tampermonkey extension in your browser.\n" +
    "2. Create a new userscript and paste the contents of\n" +
    "   tools/pokernow/pokernow.user.js (in the repo).\n" +
    "3. Make sure THIS study server is running on port 8765.\n" +
    "4. Open your PokerNow table. The badge in the page corner\n" +
    "   turns green ('connected') and this status shows live frames.\n" +
    "\n" +
    "The bridge only reads the table DOM — it never acts for you."
  );
}

function renderPokernowStatus(st) {
  const el = document.getElementById("pokernow-status");
  if (!el) return;
  el.classList.remove("error");
  el.classList.add("muted");
  if (!st || !st.connected) {
    el.textContent = "○ waiting for browser…";
    return;
  }
  const seats = st.table_seats ? ` · ${st.table_seats}-handed` : "";
  el.textContent = `● connected · frames ${st.frames_seen} · actions ${st.events_applied}${seats}`;
  if (st.last_error) {
    el.textContent += ` · ${st.last_error}`;
    el.classList.add("error");
    el.classList.remove("muted");
  }
}

function startPokernowPolling() {
  stopPokernowPolling();
  let lastFrames = -1;
  UI.pokernowPollTimer = setInterval(async () => {
    if (UI.pokernowPollBusy) return;  // previous tick still in flight
    UI.pokernowPollBusy = true;
    try {
      const st = await getJSON("/pokernow/status");
      renderPokernowStatus(st);
      // Pull fresh session state only when the bridge advanced a frame. The
      // frame counter is only acknowledged once the state was really applied:
      // in trainer mode, or when the fetch lost the race to another state, the
      // next tick tries again instead of sitting on a stale table.
      if (st.connected && st.frames_seen !== lastFrames) {
        if (await livePollFetchState()) lastFrames = st.frames_seen;
      }
    } catch (_) { /* endpoint unavailable; ignore transient errors */ }
    finally { UI.pokernowPollBusy = false; }
  }, 500);
}

function stopPokernowPolling() {
  if (UI.pokernowPollTimer !== null) {
    clearInterval(UI.pokernowPollTimer);
    UI.pokernowPollTimer = null;
  }
}

window.__wgLiveTopBar = function () {
  document.getElementById("source-toggle").addEventListener("click", () => toggleLiveSource());
  document.getElementById("pokernow-help").addEventListener("click", () => showPokerNowHelp());
  document.getElementById("ocr-toggle").addEventListener("click", () => toggleOcr());
  document.getElementById("ocr-simple-toggle").addEventListener("click", () => toggleSimpleOcr());
  document.getElementById("ocr-save-frame").addEventListener("click", () => saveOcrFrame());
  document.getElementById("ocr-window-button").addEventListener("click", () => openOcrWindowPicker());
  document.getElementById("ocr-rescan-hole-btn").addEventListener("click", () => postRescan("hole"));
  document.getElementById("ocr-rescan-board-btn").addEventListener("click", () => postRescan("board"));
};
window.__wgLiveState = function (s) {
  if (typeof s.simple_ocr_mode === "boolean") {
    UI.simpleOcrMode = s.simple_ocr_mode;
    setSimpleOcrToggleUI();
  }
};
window.__wgLiveInit = function () { applySourceUI(); };
// WGLIVE:END

document.addEventListener("DOMContentLoaded", init);
