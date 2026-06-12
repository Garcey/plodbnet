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
// Action label in the ACTIVE display unit. Server-built *_label strings
// are bb-only; prefer this whenever raw gate/chips are in the payload.
function gateActionLabel(gateSlug, chips, toCall, state) {
  if (gateSlug === "fold") return "Fold";
  if (gateSlug === "check_call") return toCall > 0 ? "Call" : "Check";
  const verb = toCall > 0 ? "Raise" : "Bet";
  return `${verb} ${formatUnit(chips || 0, state)}`;
}
// Chips the current actor has already committed THIS street. Engine raise
// bounds / recommendation are DELTAS on top of this; the UI displays totals,
// where total = delta + actorCommitChips. Returns 0 (-> total == delta, a safe
// no-op) for opening bets or when there is no actor.
function actorCommitChips(s) {
  return (s && s.actor !== null && s.actor !== undefined && s.seats[s.actor])
    ? s.seats[s.actor].committed_this_street_chips : 0;
}
function showToast(msg, kind = "error") {
  const container = document.getElementById("toast-container");
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
    return localStorage.getItem("plo5bp-mode") === "trainer" ? "trainer" : "study";
  })(),
  selectedSlot: null,
  lastState: null,
  lastStateKey: null,
  cardsPostPending: false,
  cardsPostInFlight: false,
  draggingButton: false,
  ocrRunning: false,
  ocrPollTimer: null,
  ocrLastStatus: null,
  ocrToggleBusy: false,
  ocrPollMs: 200,
  ocrWindowMatch: "",
  ocrMenuOpen: false,
  simpleOcrMode: true,
  simpleOcrToggleBusy: false,
  raiseUserSet: false,
  raiseLastActor: null,
  // Trainer
  feedbackShownIdx: -1,
  feedbackTimer: null,
  reviewDecision: null,   // decision index currently shown in review, or null
  trainerPick: false,     // card grid open for a what-if swap
  settingsOpen: false,
  animSeq: 0,             // bumped to cancel an in-flight frame animation
  animating: false,
};

const TRAINER_ANIM_MS = 1200;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function apiBase() {
  return UI.mode === "trainer" ? "/trainer" : "";
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const d = (await res.json()).detail;
      if (typeof d === "string") detail = d;
      else if (Array.isArray(d)) detail = d.map(x => x.msg ?? JSON.stringify(x)).join("; ");
      else if (d !== undefined) detail = JSON.stringify(d);
    } catch (_) {}
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}
async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function fetchState() {
  try { const data = await getJSON(`${apiBase()}/state`); applyState(data.state); }
  catch (e) { showToast(e.message); }
}
async function postCards() {
  UI.cardsPostPending = true;
  if (UI.cardsPostInFlight) return;
  UI.cardsPostInFlight = true;
  while (UI.cardsPostPending) {
    UI.cardsPostPending = false;
    const s = UI.lastState;
    if (!s) break;
    try {
      const data = await postJSON("/cards", {
        hero_hole: s.card_spec.hero_hole,
        flop_a: s.card_spec.flop_a,
        flop_b: s.card_spec.flop_b,
        turn: s.card_spec.turn,
        river: s.card_spec.river,
      });
      applyState(data.state);
    } catch (e) { showToast(e.message); }
  }
  UI.cardsPostInFlight = false;
}
async function postSeats(body) {
  try { const data = await postJSON("/seats", body); applyState(data.state); }
  catch (e) { showToast(e.message); }
}
async function postAction(body) {
  if (UI.mode === "trainer") {
    if (UI.animating) return;
    try {
      const data = await postJSON("/trainer/act", body);
      await animateTrainerResponse(data);
    } catch (e) { showToast(e.message); }
    return;
  }
  try { const data = await postJSON("/action", body); applyState(data.state); }
  catch (e) { showToast(e.message); }
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
async function animateTrainerResponse(data) {
  const frames = data.frames || [];
  const final = data.state;
  if (UI.mode !== "trainer" || frames.length === 0) {
    applyState(final);
    return;
  }
  const seq = ++UI.animSeq;
  UI.animating = true;
  try {
    // Hero's verdict flashes immediately, while opponents play out.
    if (final.trainer && final.trainer.feedback) {
      renderFeedbackFlash(final, true);
    }
    for (let i = 0; i < frames.length; i++) {
      if (UI.animSeq !== seq) return;
      render(frames[i]);
      if (i < frames.length - 1) await sleep(TRAINER_ANIM_MS);
    }
    if (UI.animSeq !== seq) return;
  } finally {
    UI.animating = false;
  }
  applyState(final);
}
async function trainerReviewGoto(decision) {
  try {
    const data = await getJSON(`/trainer/review?decision=${decision}`);
    UI.reviewDecision = decision;
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
async function postRescan(target) {
  try {
    const data = await postJSON("/ocr/rescan", { target });
    applyState(data.state);
  } catch (e) {
    showToast(`Rescan ${target} failed: ${e.message}`);
  }
}
async function postConfig(body) {
  try { const data = await postJSON("/config", body); applyState(data.state); }
  catch (e) { showToast(e.message); }
}

function applyState(s) {
  UI.animSeq++;  // an authoritative state cancels any frame animation
  UI.lastState = s;
  if (UI.selectedSlot) {
    const cur = s.card_spec[UI.selectedSlot.key][UI.selectedSlot.index];
    if (cur !== null && cur !== undefined) UI.selectedSlot = null;
  }
  if (typeof s.simple_ocr_mode === "boolean") {
    UI.simpleOcrMode = s.simple_ocr_mode;
    setSimpleOcrToggleUI();
  }
  // Skip the full re-render when nothing visible changed. Each render
  // does innerHTML = "" on action / board / card-grid containers,
  // which detaches buttons mid-click during 2 Hz OCR polling and eats
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

function render(s) {
  renderTopBar(s);
  renderSeats(s);
  renderBoards(s);
  renderHeroHole(s);
  renderDealerButton(s);
  renderPotLabel(s);
  renderActorBanner(s);
  renderActions(s);
  renderRecommendation(s);
  renderHistory(s);
  renderCardGrid(s);
  renderTrainer(s);
  document.getElementById("undo-btn").disabled = !s.can_undo;
  const insertIcon = document.getElementById("insert-icon");
  if (s.num_seats >= 6 || s.trainer) {
    insertIcon.setAttribute("hidden", "");
    insertIcon.style.display = "none";
  } else {
    insertIcon.style.display = "";
  }
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
    const stacks = s.seats.map(x => x.stack_chips);
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

  renderSlotStrip(boardA, "flop_a", s.card_spec.flop_a, s, 0);
  renderSlotStrip(boardB, "flop_b", s.card_spec.flop_b, s, 0);

  const turnX0 = 3 * (44 + 6);
  renderSingleSlot(boardA, "turn", 0, s.card_spec.turn[0], s, turnX0);
  renderSingleSlot(boardB, "turn", 1, s.card_spec.turn[1], s, turnX0);

  const riverX0 = 4 * (44 + 6);
  renderSingleSlot(boardA, "river", 0, s.card_spec.river[0], s, riverX0);
  renderSingleSlot(boardB, "river", 1, s.card_spec.river[1], s, riverX0);
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
  const totalWidth = geom.count * geom.width + (geom.count - 1) * geom.gap;
  const x0 = -totalWidth / 2;
  for (let i = 0; i < geom.count; i++) {
    const x = x0 + i * (geom.width + geom.gap);
    renderSlotRect(g, "hero_hole", i, s.card_spec.hero_hole[i], s, x, 0, geom.width, geom.height);
  }
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
  document.getElementById("pot-label").textContent = `Pot ${formatUnit(s.pot_chips, s)}`;
}

function animActionText(s) {
  const a = s.trainer.anim_action;
  const seat = s.seats[a.seat];
  if (a.gate === "fold") return `${a.position} folds`;
  if (a.gate === "check_call") {
    return a.to_call > 0 ? `${a.position} calls` : `${a.position} checks`;
  }
  const committed = seat ? seat.committed_this_street_chips : a.chips;
  const verb = seat && seat.all_in ? "is all-in"
    : a.to_call > 0 ? "raises to" : "bets";
  const amt = committed > 0 ? ` ${formatUnit(committed, s)}` : "";
  return `${a.position} ${verb}${amt}`;
}

function renderActorBanner(s) {
  const banner = document.getElementById("actor-banner");
  // Trainer animation frame: narrate the opponent action that just landed.
  if (s.trainer && s.trainer.anim_action) {
    banner.hidden = false;
    banner.classList.toggle("hero", false);
    banner.textContent = animActionText(s);
    return;
  }
  // Trainer review: the state is a mid-hand reconstruction, not a live turn.
  if (s.trainer && !s.trainer.hand_active && s.trainer.review) {
    const rv = s.trainer.review;
    const cur = rv.current;
    banner.hidden = false;
    banner.classList.toggle("hero", true);
    banner.textContent =
      `Reviewing decision ${rv.decision + 1} / ${rv.num_decisions} — ${cur.street}` +
      ` · you chose ${cur.user_label}`;
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
  const BLOCK_TOOLTIPS = {
    hole:  "Place hero's 5 hole cards to act",
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
    b.disabled = !enabled;
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
    const isAllIn = actorSeat
      && maxChips >= actorSeat.stack_chips + actorSeat.committed_this_street_chips;
    const verb = s.to_call_chips > 0 ? "Raise" : "Bet";
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

  if (UI.raiseLastActor !== s.actor) {
    UI.raiseUserSet = false;
    UI.raiseLastActor = s.actor;
  }

  const userTyping = document.activeElement === input;
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
  // Returns a raise-TO total (matches the now-total-space input). A pot-fraction
  // bet means: call (toCall) then raise BY mult*(pot+toCall) on top, so the
  // final commitment is ac + toCall + extra. Clamp in total space. (The input is
  // total and submit subtracts ac, so the commitment equals this exactly — this
  // also fixes the old over-commit where a total was posted as a delta.)
  const potSize = (mult) => {
    const toCall = s.to_call_chips;
    const extra = Math.round(mult * (s.pot_chips + toCall));
    const total = ac + toCall + extra;
    return Math.max(minChips + ac, Math.min(maxChips + ac, total));
  };
  const items = [
    { label: "b25", chips: potSize(0.25) },
    { label: "b33", chips: potSize(1 / 3) },
    { label: "b50", chips: potSize(0.5) },
    { label: "b75", chips: potSize(0.75) },
    { label: "pot", chips: potSize(1.0) },
  ];
  for (const it of items) {
    const b = document.createElement("button");
    b.className = "raise-shortcut";
    b.textContent = it.label;
    b.addEventListener("click", () => {
      input.value = chipsToCurrentUnit(it.chips, s).toFixed(2);
      UI.raiseUserSet = true;
    });
    shortcuts.appendChild(b);
  }
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

function anchorHeatmapHTML(anchors, recAnchor, userAnchor, s) {
  // v2 sizing EQ: one column per anchor spanning min → pot, like a
  // stereo equalizer — bar height carries the network's preference
  // (more bulk = more bet at that size). ★ = network's pick; ring +
  // ● = the anchor the user's size snapped to (review).
  const byK = new Map((anchors || []).map((a) => [a.k, a]));
  const pmax = Math.max(1e-9, ...(anchors || []).map((a) => a.prob));
  let cells = "";
  let labels = "";
  for (let k = 0; k <= 10; k++) {
    const a = byK.get(k);
    if (a) {
      const isUser = userAnchor !== null && userAnchor !== undefined && k === userAnchor;
      const isRec = k === recAnchor;
      const marks = `${isRec ? "★" : ""}${isUser ? "●" : ""}`;
      const tip = `${a.label} pot — ${(a.prob * 100).toFixed(0)}% — ${formatUnit(a.chips, s)}`;
      // Floor nonzero bars at 6% so a rarely-used size still shows a
      // sliver instead of reading as illegal.
      const hpct = Math.max(6, 100 * a.prob / pmax);
      cells += `
        <div class="anchor-eq-cell${isUser ? " is-user" : ""}${isRec ? " is-rec" : ""}" title="${tip}">
          <div class="anchor-eq-bar" style="height:${hpct.toFixed(1)}%"></div>
          <span class="anchor-eq-marks">${marks}</span>
        </div>`;
    } else {
      cells += `<div class="anchor-eq-cell dead" title="${ANCHOR_AXIS_LABELS[k]} — not a distinct legal size here"></div>`;
    }
    labels += `<span>${ANCHOR_AXIS_LABELS[k]}</span>`;
  }
  return `
    <div class="anchor-heat">
      <div class="anchor-eq-row">${cells}</div>
      <div class="anchor-heat-labels">${labels}</div>
    </div>`;
}

function recDetailHTML(rec, userAnchor, s) {
  // v2 payloads carry `anchors`; v1 carries the single Beta's (α, β).
  if (rec.anchors) {
    let html = anchorHeatmapHTML(rec.anchors, rec.rec_anchor, userAnchor, s);
    if (rec.refine) {
      html += `<div class="rec-detail">slider β(${rec.refine.alpha.toFixed(1)}, ${rec.refine.beta.toFixed(1)})</div>`;
    }
    return html;
  }
  return `<div class="rec-detail">β(${(rec.beta_alpha ?? 0).toFixed(1)}, ${(rec.beta_beta ?? 0).toFixed(1)})</div>`;
}

function renderTrainerReviewRecommendation(s, el) {
  const rv = s.trainer.review;
  const cur = rv.current;
  const whatif = rv.whatif;
  const callName = cur.to_call_chips > 0 ? "Call" : "Check";
  let actionText, dist, valueBB, tag = "", detailHTML;
  if (whatif) {
    const rec = whatif.recommendation;
    actionText = rec.chips !== null && rec.chips !== undefined
      ? `${cur.to_call_chips > 0 ? "Raise" : "Bet"} ${formatUnit(rec.chips, s)}`
      : (rec.gate === "fold" ? "Fold" : callName);
    dist = rec.gate_distribution;
    valueBB = rec.value_bb;
    tag = `<span class="whatif-tag">what-if</span> `;
    detailHTML = recDetailHTML(rec, null, s);
  } else {
    actionText = cur.rec_gate
      ? gateActionLabel(cur.rec_gate, cur.rec_chips, cur.to_call_chips, s)
      : cur.rec_label;
    dist = cur.gate_probs;
    valueBB = cur.value_bb;
    detailHTML = recDetailHTML(cur, cur.user_anchor, s);
  }
  const sign = valueBB >= 0 ? "+" : "-";
  const absBB = Math.abs(valueBB);
  const vDisp = UI.unit === "bb"
    ? `${sign}${absBB.toFixed(2)}bb`
    : `${sign}$${(absBB * (s?.chip_scale?.dollars_per_bb ?? 2)).toFixed(2)}`;
  el.innerHTML = `
    <div class="rec-line">
      ${tag}<span class="rec-action">${actionText}</span>
      <span class="rec-value">value ${vDisp}</span>
    </div>
    <div class="rec-dist">${distRowsHTML(dist, callName)}</div>
    ${detailHTML}
  `;
}

function renderRecommendation(s) {
  const el = document.getElementById("recommendation");
  if (s.trainer) {
    if (s.trainer.review && s.trainer.review.current) {
      renderTrainerReviewRecommendation(s, el);
      return;
    }
    el.innerHTML = '<p class="muted">Hidden during play — revealed in the post-hand review.</p>';
    return;
  }
  const REC_PROMPTS = {
    hole:  "Place hero's 5 hole cards to see network output.",
    flop:  "Place the flop cards to see network output.",
    turn:  "Place the turn cards to see network output.",
    river: "Place the river cards to see network output.",
  };
  if (s.hero_blocking_reason != null) {
    el.innerHTML = `<p class="muted">${REC_PROMPTS[s.hero_blocking_reason]}</p>`;
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
    const verb = s.to_call_chips > 0 ? "Raise" : "Bet";
    const actorSeat = s.actor !== null && s.actor !== undefined ? s.seats[s.actor] : null;
    const isAllIn = actorSeat
      && rec.chips >= actorSeat.stack_chips + actorSeat.committed_this_street_chips;
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
    <div class="rec-line">
      <span class="rec-action">${actionText}</span>
      <span class="rec-value">value ${vDisp}</span>
    </div>
    <div class="rec-dist">${distRows}</div>
    ${recDetailHTML(rec, null, s)}
  `;
}

function actionLabel(h, s, idx) {
  const name = h.action;
  const chips = Number(h.chips) || 0;
  if (name === "Fold") return "Fold";
  if (name === "CheckCall") return chips > 0 ? `Call ${formatUnit(chips, s)}` : "Check";
  if (name === "AllIn") return `All-in ${formatUnit(chips, s)}`;
  let priorAggression = false;
  for (let i = 0; i < idx; i++) {
    const p = s.history[i];
    if (p.street === h.street && Number(p.chips) > 0 && p.action !== "CheckCall") {
      priorAggression = true;
      break;
    }
  }
  const verb = priorAggression ? "Raise" : "Bet";
  return `${verb} ${formatUnit(chips, s)}`;
}

function renderHistory(s) {
  const el = document.getElementById("history");
  el.innerHTML = "";
  if (!s.history.length) {
    el.innerHTML = '<p class="muted">No actions yet.</p>';
    return;
  }
  for (let i = 0; i < s.history.length; i++) {
    const h = s.history[i];
    const row = document.createElement("div");
    row.className = "history-entry";
    const streetClass = `h-${String(h.street).toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
    row.innerHTML = `
      <span class="h-street ${streetClass}">${h.street}</span>
      <span class="h-pos">${h.position}</span>
      <span class="h-action">${actionLabel(h, s, i)}</span>
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
    const labels = {
      hero_hole: "Hero hole", flop_a: "Board A flop", flop_b: "Board B flop",
      turn: "Turn", river: "River",
    };
    hint.textContent = `Selected: ${labels[key]} [${index + 1}]`;
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
    const spec = s.card_spec[key];
    if (!spec || spec[index] === null || spec[index] === undefined) return;
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
  postCards();
}

function onGridCardClick(cardInt) {
  const s = UI.lastState;
  if (!s) return;
  if (s.trainer) {
    const rv = s.trainer.review;
    const sel = UI.selectedSlot;
    if (!rv || !sel || !UI.trainerPick) return;
    const used = collectUsedCards(s);
    if (used.has(cardInt)) return;
    // Send the full current spec as absolute overrides so earlier
    // what-if swaps survive; the server re-validates against originals.
    const base = rv.whatif ? rv.whatif.card_spec : s.card_spec;
    const body = { decision: rv.decision };
    for (const k of ["hero_hole", "flop_a", "flop_b", "turn", "river"]) {
      body[k] = (base[k] || []).slice();
    }
    body[sel.key][sel.index] = cardInt;
    UI.selectedSlot = null;
    cancelTrainerPick();
    postTrainer("whatif", body);
    return;
  }
  const sel = UI.selectedSlot;
  if (!sel) { showToast("Click a slot first"); return; }
  const used = collectUsedCards(s);
  if (used.has(cardInt)) return;
  const spec = s.card_spec[sel.key];
  spec[sel.index] = cardInt;
  const nextIndex = spec.findIndex((v, i) => v === null && i > sel.index);
  if (nextIndex >= 0) {
    UI.selectedSlot = { key: sel.key, index: nextIndex };
  } else {
    UI.selectedSlot = null;
  }
  postCards();
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
    let rel = THETA0 - ma;
    while (rel < 0) rel += TWO_PI;
    while (rel >= TWO_PI) rel -= TWO_PI;
    const rawIdx = Math.floor(rel / seatStep);
    const hero = s.hero_seat;
    const pairA = (rawIdx + hero) % N;
    const pairB = (rawIdx + 1 + hero) % N;

    const midTheta = THETA0 - (rawIdx + 0.5) * seatStep;
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
  document.getElementById("undo-btn").addEventListener("click", () => postUndo());
  document.getElementById("new-hand-btn").addEventListener("click", () => postReset());
  document.getElementById("ocr-toggle").addEventListener("click", () => toggleOcr());
  document.getElementById("ocr-simple-toggle").addEventListener("click", () => toggleSimpleOcr());
  document.getElementById("ocr-save-frame").addEventListener("click", () => saveOcrFrame());
  document.getElementById("ocr-window-button").addEventListener("click", () => openOcrWindowPicker());
  document.getElementById("ocr-rescan-hole-btn").addEventListener("click", () => postRescan("hole"));
  document.getElementById("ocr-rescan-board-btn").addEventListener("click", () => postRescan("board"));
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
    try {
      const st = await getJSON("/ocr/status");
      const wasRunning = UI.ocrRunning;
      UI.ocrLastStatus = st;
      UI.ocrRunning = !!st.running;
      setOcrToggleUI();
      renderOcrStatus(st);
      if (st.running) {
        await fetchState();
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

// --- Trainer mode -------------------------------------------------------

function applyModeUI() {
  const trainer = UI.mode === "trainer";
  document.body.classList.toggle("trainer-mode", trainer);
  document.getElementById("tab-study").classList.toggle("active", !trainer);
  document.getElementById("tab-trainer").classList.toggle("active", trainer);
}

function setMode(mode) {
  if (UI.mode === mode) return;
  UI.mode = mode;
  localStorage.setItem("plo5bp-mode", mode);
  UI.lastState = null;
  UI.lastStateKey = null;
  UI.selectedSlot = null;
  UI.reviewDecision = null;
  cancelTrainerPick();
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
  const userLabel = fb.user_gate
    ? gateActionLabel(fb.user_gate, fb.user_chips, fb.to_call_chips, s)
    : fb.label;
  document.getElementById("feedback-text").textContent =
    `${userLabel} · ${Math.round(fb.score)}%`;
  const bits = [];
  const recLabel = fb.rec_gate
    ? gateActionLabel(fb.rec_gate, fb.rec_chips, fb.to_call_chips, s)
    : fb.rec_label;
  if (fb.category !== "best" && recLabel) bits.push(`best: ${recLabel}`);
  if (fb.ev_loss_bb !== null && fb.ev_loss_bb !== undefined && fb.ev_loss_bb > 0) {
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
    <div class="stats-title">${title}
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

function renderTrainerStats(s) {
  const stats = s.trainer.stats || {};
  const se = document.getElementById("stats-session");
  const lt = document.getElementById("stats-lifetime");
  se.innerHTML = statsBlockHTML("Session", stats.session || {}, "session");
  lt.innerHTML = statsBlockHTML("Lifetime", stats.lifetime || {}, "lifetime");
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

  const C = 2 * Math.PI * 26;
  const pct = rv.hand_score ?? 0;
  const fill = document.getElementById("review-ring-fill");
  fill.style.strokeDasharray = `${((C * pct) / 100).toFixed(1)} ${C.toFixed(1)}`;
  fill.setAttribute("class", `ring-fill ${ringClass(pct)}`);
  document.getElementById("review-score-num").textContent =
    (rv.hand_score !== null && rv.hand_score !== undefined)
      ? `${Math.round(rv.hand_score)}%` : "—";

  document.getElementById("review-step-label").textContent =
    `Decision ${rv.decision + 1} / ${rv.num_decisions}`;
  document.getElementById("review-prev").disabled = rv.decision <= 0;
  document.getElementById("review-next").disabled = rv.decision >= rv.num_decisions - 1;

  const chips = document.getElementById("review-chips");
  chips.innerHTML = "";
  rv.decisions.forEach((d, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = `review-chip cat-border-${d.category}` + (i === rv.decision ? " current" : "");
    b.innerHTML = `<span class="rc-street">${d.street}</span>${d.user_label}` +
      ` <span class="rc-score">${Math.round(d.score)}%</span>`;
    b.addEventListener("click", () => trainerReviewGoto(i));
    chips.appendChild(b);
  });

  document.getElementById("review-whatif-bar").hidden = !rv.whatif;

  const cur = rv.current;
  const detail = document.getElementById("review-detail");
  let evRow = "";
  if (cur.ev_loss_bb !== null && cur.ev_loss_bb !== undefined) {
    const detailBit = (cur.ev_user_bb !== null && cur.ev_user_bb !== undefined)
      ? ` <span class="muted">(you ${cur.ev_user_bb.toFixed(2)} vs best ${cur.ev_best_bb.toFixed(2)})</span>`
      : "";
    evRow = `<div class="review-ev">EV loss <b>${cur.ev_loss_bb.toFixed(2)}bb</b>${detailBit}</div>`;
  }
  const rescored = rv.whatif
    ? `<div class="review-rescored">What-if rescore: <b class="cat-text-${rv.whatif.rescored.category}">` +
      `${rv.whatif.rescored.category}</b> ${Math.round(rv.whatif.rescored.score)}%</div>`
    : "";
  detail.innerHTML = `
    <div class="review-verdict cat-text-${cur.category}">${cur.marks} ${cur.category.toUpperCase()} · ${Math.round(cur.score)}%</div>
    <div class="review-moves">You: <b>${cur.user_label}</b> · Network: <b>${cur.rec_label}</b></div>
    ${evRow}${rescored}
    <div class="muted review-hint">Click a board or hero card to try a what-if swap.</div>
  `;
}

// --- Trainer settings modal ----------------------------------------------

function _tsVal(id) { return document.getElementById(id).value; }
function _tsNum(id) { return parseFloat(document.getElementById(id).value); }
function _tsInt(id) { return parseInt(document.getElementById(id).value, 10); }
function _tsShow(id, on) { document.getElementById(id).style.display = on ? "" : "none"; }

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
  document.getElementById("ts-mc-rollouts").value = t.mc_rollouts;
  document.getElementById("ts-dollars-bb").value = t.dollars_per_bb;
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
    mc_rollouts: _tsInt("ts-mc-rollouts"),
    dollars_per_bb: _tsNum("ts-dollars-bb"),
  };
  const ok = await postTrainer("settings", body);
  if (ok) closeTrainerSettings();
}

function setupTrainerControls() {
  document.getElementById("tab-study").addEventListener("click", () => setMode("study"));
  document.getElementById("tab-trainer").addEventListener("click", () => setMode("trainer"));
  const newHand = () => {
    cancelTrainerPick();
    UI.reviewDecision = null;
    UI.selectedSlot = null;
    postTrainer("new_hand");
  };
  const repeatHand = () => {
    cancelTrainerPick();
    UI.reviewDecision = null;
    UI.selectedSlot = null;
    postTrainer("repeat");
  };
  document.getElementById("trainer-new-hand-btn").addEventListener("click", newHand);
  document.getElementById("trainer-repeat-btn").addEventListener("click", repeatHand);
  document.getElementById("review-next-hand").addEventListener("click", newHand);
  document.getElementById("review-repeat-hand").addEventListener("click", repeatHand);
  document.getElementById("review-prev").addEventListener("click", () => {
    const rv = UI.lastState?.trainer?.review;
    if (rv && rv.decision > 0) trainerReviewGoto(rv.decision - 1);
  });
  document.getElementById("review-next").addEventListener("click", () => {
    const rv = UI.lastState?.trainer?.review;
    if (rv && rv.decision < rv.num_decisions - 1) trainerReviewGoto(rv.decision + 1);
  });
  document.getElementById("review-whatif-reset").addEventListener("click", () => {
    const rv = UI.lastState?.trainer?.review;
    if (rv) trainerReviewGoto(rv.decision);
  });
  document.getElementById("trainer-settings-btn").addEventListener("click", openTrainerSettings);
  document.getElementById("ts-cancel").addEventListener("click", closeTrainerSettings);
  document.getElementById("ts-save").addEventListener("click", saveTrainerSettings);
  for (const id of ["ts-seats-mode", "ts-stacks-mode", "ts-hero-mode"]) {
    document.getElementById(id).addEventListener("change", syncSettingsVisibility);
  }
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
  setupTopBar();
  setupDealerDrag();
  setupInsertHover();
  setupRaiseInput();
  setupTrainerControls();
  applyModeUI();
  fetchState();
  refreshOcrStatusOnLoad();
}

document.addEventListener("DOMContentLoaded", init);
