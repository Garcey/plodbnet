// Study / Trainer client, part 4 of 7 — Study tools: undo / redo and
// rewinding to any action, share links for a spot, opening a Trainer hand
// in Study, and the "How Study works" guide.
//
// Plain scripts, no build step: index.html loads app.core.js, app.table.js,
// app.play.js, app.study.js, app.trainer.js, app.topbar.js and app.js in
// that order, and they share one global scope — any file may call a
// function from any other. Code that runs while a file LOADS may use only
// the files before it; everything else starts from init() in app.js.
"use strict";

// --- Undo / Redo / rewind (FEAT-020) --------------------------------------------------
// Going back keeps what was taken off, so Redo can replay it — a stray click
// on an early history line costs nothing. Any other change clears Redo.
function renderRedo(s) {
  const btn = document.getElementById("redo-btn");
  if (!btn) return;
  const n = (UI.redo || []).length;
  btn.hidden = !n || !!s.trainer;
  btn.disabled = !n;
  btn.title = n ? `Redo ${n === 1 ? "the action" : `${n} actions`} you went back over` : "";
}

async function rewindTo(length) {
  const s = UI.lastState;
  if (!s || s.trainer || UI.actionInFlight) return;
  if (length >= (s.history || []).length) return;
  UI.actionInFlight = true;
  setActionsBusy(true);
  try {
    const data = await postJSON("/study/rewind", { length });
    UI.redo = [...(data.removed || []), ...(UI.redo || [])];
    applyState(data.state);
  } catch (e) { showToast(e.message); }
  finally { UI.actionInFlight = false; setActionsBusy(false); }
}

async function redoOne() {
  const next = (UI.redo || [])[0];
  if (!next || UI.actionInFlight) return;
  UI.actionInFlight = true;
  setActionsBusy(true);
  try {
    const body = next.gate === "raise" ? { gate: "raise", chips: next.chips } : { gate: next.gate };
    const data = await postJSON("/study/action", body);
    UI.redo = UI.redo.slice(1);
    applyState(data.state);
  } catch (e) {
    UI.redo = [];
    showToast(e.message);
    if (UI.lastState) renderRedo(UI.lastState);
  } finally { UI.actionInFlight = false; setActionsBusy(false); }
}

function setupHistoryRewind() {
  const hist = document.getElementById("history");
  hist.addEventListener("click", (e) => {
    const row = e.target.closest("[data-rewind]");
    if (row) rewindTo(parseInt(row.dataset.rewind, 10));
  });
  // Redo sits next to Undo, in the dock (2026-10-01)
  const undo = document.getElementById("undo-btn");
  if (undo && !document.getElementById("redo-btn")) {
    const redo = document.createElement("button");
    redo.id = "redo-btn";
    redo.type = "button";
    redo.className = "btn sm study-only";
    redo.textContent = "Redo";
    redo.hidden = true;
    redo.addEventListener("click", redoOne);
    undo.after(redo);
  }
}

// --- Share a Study spot (FEAT-016 / FEAT-026) -------------------------------------------
// The link carries the whole spot — table, cards and actions — as compact
// base64url JSON; opening it loads the spot through POST /spot, which
// validates it on the server exactly like hand-entered actions.
function b64urlEncode(str) {
  const bytes = new TextEncoder().encode(str);
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
function b64urlDecode(code) {
  const b64 = String(code).replace(/-/g, "+").replace(/_/g, "/");
  const bin = atob(b64 + "===".slice((b64.length + 3) % 4));
  return new TextDecoder().decode(Uint8Array.from(bin, (c) => c.charCodeAt(0)));
}
const GATE_CODE = { fold: "f", check_call: "c", raise: "r" };
const CODE_GATE = { f: "fold", c: "check_call", r: "raise" };
const nullToNeg = (arr) => (arr || []).map((c) => (c === null || c === undefined ? -1 : c));
const negToNull = (arr) => (Array.isArray(arr) ? arr.map((c) => (c === -1 ? null : c)) : undefined);

function encodeSpot(sp) {
  return b64urlEncode(JSON.stringify([
    1, sp.format, sp.num_seats, sp.button_seat, sp.starting_stacks, sp.ante_chips, sp.bb_chips,
    nullToNeg(sp.hero_hole), nullToNeg(sp.flop_a), nullToNeg(sp.flop_b),
    nullToNeg(sp.turn), nullToNeg(sp.river),
    (sp.actions || []).map((a) => (a.gate === "raise" ? `r${a.chips}` : GATE_CODE[a.gate])).join(","),
  ]));
}
function decodeSpot(code) {
  const a = JSON.parse(b64urlDecode(code));
  if (!Array.isArray(a) || a[0] !== 1) throw new Error("unknown spot version");
  const acts = String(a[12] || "").split(",").filter(Boolean).map((t) => (
    t[0] === "r" ? { gate: "raise", chips: parseInt(t.slice(1), 10) } : { gate: CODE_GATE[t] }
  ));
  return {
    format: a[1], num_seats: a[2], button_seat: a[3], starting_stacks: a[4],
    ante_chips: a[5], bb_chips: a[6],
    hero_hole: negToNull(a[7]), flop_a: negToNull(a[8]), flop_b: negToNull(a[9]),
    turn: negToNull(a[10]), river: negToNull(a[11]), actions: acts,
  };
}

async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch (_) { /* fall back */ }
  return false;
}

async function shareSpot() {
  const s = UI.lastState;
  if (!s || s.trainer) return;
  let spot;
  try { spot = (await getJSON("/study/spot")).spot; }
  catch (e) { showToast(e.message); return; }
  const url = `${location.origin}/?mode=study&spot=${encodeSpot(spot)}`;
  if (await copyText(url)) {
    showToast("Link copied — anyone signed in can open this spot in their Study.", "info");
    return;
  }
  // No clipboard (plain http, old browser): show the link to copy by hand.
  const dlg = document.getElementById("help-dlg");
  document.getElementById("help-title").textContent = "Share this spot";
  document.getElementById("help-body").innerHTML =
    '<p>Copy this link. Anyone signed in to WrapGTO can open the spot in their Study.</p>'
    + `<input class="share-url" type="text" readonly value="${escapeHTML(url)}" aria-label="Link to this spot" />`;
  openDialog(dlg);
  const input = dlg.querySelector(".share-url");
  if (input) { input.focus(); input.select(); }
}

// Loads a spot into Study (share link or the Trainer's "Open in Study").
// Asks before replacing a Study hand that has work in it.
async function loadStudySpot(spot, { confirmReplace = true } = {}) {
  let cur = UI.mode === "study" ? UI.lastState : null;
  if (!cur || cur.trainer) {
    try { cur = (await getJSON("/study/state")).state; } catch (_) { cur = null; }
  }
  if (confirmReplace && cur && !cur.trainer) {
    const n = (cur.history || []).length;
    const cards = collectUsedCards(cur).size;
    if (n || cards) {
      const ok = await confirmDialog({
        title: "Replace your Study hand?",
        body: "Opening this spot replaces the hand you have in Study.",
        ok: "Open the spot",
      });
      if (!ok) return false;
    }
  }
  if (spot.format && cur && cur.format && spot.format !== cur.format) {
    const f = UI.formats ? UI.formats.find((x) => x.id === spot.format) : null;
    if (f && f.locked) { showToast("That spot is in a format your account can't open yet."); return false; }
    await postJSON("/study/format", { format: spot.format });
    UI.lastState = null;
    const sel = document.getElementById("format-select");
    if (sel) sel.value = spot.format;
  }
  const data = await postJSON("/study/spot", spot);
  UI.redo = [];
  UI.selectedSlot = null;
  if (UI.mode !== "study") {
    setMode("study");
  } else {
    UI.lastStateKey = null;
    applyState(data.state);
  }
  return true;
}

async function openSharedSpot(code) {
  // The spot is applied once; drop it from the address bar so a reload
  // shows whatever the spot turned into, not the original again.
  try {
    const url = new URL(location.href);
    url.searchParams.delete("spot");
    history.replaceState(history.state, "", url.pathname + url.search + url.hash);
  } catch (_) { /* fine */ }
  let spot;
  try { spot = decodeSpot(code); }
  catch (_) {
    showToast("That spot link is damaged — ask for a new one.");
    await fetchState();
    return;
  }
  if (UI.mode !== "study") {
    UI.mode = "study";
    lsSet("plo5bp-mode", "study");
    applyModeUI();
  }
  try {
    if (!(await loadStudySpot(spot))) await fetchState();
    else showToast("Spot loaded from the link.", "info");
  } catch (e) {
    if (e.message !== GATE_HANDLED) showToast(`Couldn't open that spot: ${e.message}`);
    await fetchState();
  }
}

// --- Trainer hand -> Study (FEAT-017) ----------------------------------------------------
// The spot at the reviewed action: the trainer's seats rotated so the hero
// is Study's seat 0, the cards dealt by then, and every action before it.
function trainerSpotAt(s, nodeIdx) {
  const rv = s.trainer && s.trainer.review;
  if (!rv || !Array.isArray(rv.nodes)) return null;
  const n = s.num_seats;
  const h = s.hero_seat;
  const rot = (arr) => Array.from({ length: n }, (_, i) => arr[(h + i) % n]);
  const cs = s.card_spec || {};
  const upTo = Math.max(0, Math.min(nodeIdx, rv.nodes.length));
  // Only the cards dealt by that decision: the review of a finished hand
  // shows the final board, and Study shouldn't start with the future.
  const at = rv.nodes[upTo] || rv.nodes[rv.nodes.length - 1] || {};
  const rank = STREET_RANK[String(at.street || "").toLowerCase()];
  const upToStreet = (key) => (Array.isArray(cs[key]) && rank !== undefined && SLOT_MIN_STREET[key] > rank
    ? cs[key].map(() => null) : cs[key]);
  return {
    format: s.format,
    num_seats: n,
    button_seat: ((s.button_seat - h) % n + n) % n,
    starting_stacks: rot(s.starting_stacks_chips || []),
    ante_chips: s.chip_scale.ante_chips,
    bb_chips: s.chip_scale.bb_chips,
    hero_hole: cs.hero_hole, flop_a: upToStreet("flop_a"), flop_b: upToStreet("flop_b"),
    turn: upToStreet("turn"), river: upToStreet("river"),
    actions: rv.nodes.slice(0, upTo).map((nd) => (
      nd.actual_gate === "raise" ? { gate: "raise", chips: nd.actual_chips } : { gate: nd.actual_gate }
    )),
  };
}

async function openReviewInStudy() {
  const s = UI.lastState;
  const rv = s && s.trainer && s.trainer.review;
  if (!rv) return;
  const spot = trainerSpotAt(s, rv.node);
  if (!spot) return;
  const btn = document.getElementById("review-open-study");
  if (btn) btn.disabled = true;
  try {
    if (await loadStudySpot(spot)) {
      showToast("Opened in Study at this decision — try other lines from here.", "info");
    }
  } catch (e) {
    if (e.message !== GATE_HANDLED) showToast(`Couldn't open it in Study: ${e.message}`);
  } finally {
    if (btn) btn.disabled = false;
  }
}

// --- "How Study works" (ST-015) ---------------------------------------------------
// Shown until dismissed; the "?" in the work bar brings it back.
const GUIDE_PREF_KEY = "plo5bp-study-guide";
function toggleStudyGuide(show) {
  const g = document.getElementById("study-guide");
  const btn = document.getElementById("study-help-btn");
  if (!g) return;
  const on = typeof show === "boolean" ? show : g.hidden;
  g.hidden = !on;
  if (btn) btn.setAttribute("aria-expanded", on ? "true" : "false");
  lsSet(GUIDE_PREF_KEY, on ? "shown" : "hidden");
  if (on && show === undefined) g.scrollIntoView({ block: "nearest", behavior: "smooth" });
}
