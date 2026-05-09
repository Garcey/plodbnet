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
function showToast(msg) {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

const UI = {
  unit: "$",
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
  simpleOcrMode: true,
  simpleOcrToggleBusy: false,
  raiseUserSet: false,
  raiseLastActor: null,
};

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch (_) {}
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
  try { const data = await getJSON("/state"); applyState(data.state); }
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
  try { const data = await postJSON("/action", body); applyState(data.state); }
  catch (e) { showToast(e.message); }
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
  document.getElementById("undo-btn").disabled = !s.can_undo;
  const insertIcon = document.getElementById("insert-icon");
  if (s.num_seats >= 6) {
    insertIcon.setAttribute("hidden", "");
    insertIcon.style.display = "none";
  } else {
    insertIcon.style.display = "";
  }
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
    if (seat.folded) node.classList.add("folded");
    if (seat.all_in) node.classList.add("all-in");
    node.setAttribute("transform", `translate(${p.x} ${p.y})`);

    const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    bg.setAttribute("x", -46); bg.setAttribute("y", -26);
    bg.setAttribute("width", 92); bg.setAttribute("height", 52);
    bg.setAttribute("rx", 6); bg.setAttribute("class", "seat-bg");
    node.appendChild(bg);

    const pos = document.createElementNS("http://www.w3.org/2000/svg", "text");
    pos.setAttribute("y", -10); pos.setAttribute("class", "seat-position");
    pos.textContent = seat.position + (seat.is_hero ? " (hero)" : "");
    node.appendChild(pos);

    const stack = document.createElementNS("http://www.w3.org/2000/svg", "text");
    stack.setAttribute("y", 6);
    stack.setAttribute("class", seat.all_in ? "seat-stack" : "seat-stack editable");
    stack.textContent = seat.all_in ? "all-in" : formatUnit(seat.stack_chips, s);
    if (!seat.all_in) {
      stack.style.cursor = "pointer";
      stack.addEventListener("click", (e) => {
        e.stopPropagation();
        openStackEditor(seat, s);
      });
    }
    node.appendChild(stack);

    if (seat.committed_this_street_chips > 0) {
      const commit = document.createElementNS("http://www.w3.org/2000/svg", "text");
      commit.setAttribute("y", 20); commit.setAttribute("class", "seat-commit");
      commit.textContent = `+${formatUnit(seat.committed_this_street_chips, s)}`;
      node.appendChild(commit);
    }

    if (!seat.is_hero && s.num_seats > 2) {
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
  rect.setAttribute("rx", 4);
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

function renderActorBanner(s) {
  const banner = document.getElementById("actor-banner");
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
  gate.appendChild(mkBtn(toCallLabel, "check_call", s.legal.check_call && !heroBlocked));

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
    submit.textContent = isAllIn
      ? `All-in ${formatUnit(maxChips, s)}`
      : `${verb} ${formatUnit(maxChips, s)}`;
    submit.onclick = () => postAction({ gate: "raise", chips: maxChips });
    return;
  }
  inputRow.querySelectorAll("input, .raise-unit").forEach(el => el.style.display = "");
  submit.textContent = "Raise";

  const minDisp = chipsToCurrentUnit(minChips, s);
  const maxDisp = chipsToCurrentUnit(maxChips, s);
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
    input.value = chipsToCurrentUnit(preset, s).toFixed(2);
  } else if (!userTyping) {
    const cur = parseToChips(input.value, s);
    if (cur !== null && (cur < minChips || cur > maxChips)) {
      const clamped = Math.max(minChips, Math.min(maxChips, cur));
      input.value = chipsToCurrentUnit(clamped, s).toFixed(2);
    }
  }

  submit.onclick = () => {
    const chips = parseToChips(input.value, s);
    if (chips === null) { showToast("invalid raise amount"); return; }
    const clamped = Math.max(minChips, Math.min(maxChips, chips));
    UI.raiseUserSet = false;
    postAction({ gate: "raise", chips: clamped });
  };

  shortcuts.innerHTML = "";
  const potSize = (mult) => {
    const actorCommit = actorSeat.committed_this_street_chips;
    const toCall = s.to_call_chips;
    const extra = Math.round(mult * (s.pot_chips + toCall));
    return Math.max(minChips, Math.min(maxChips, actorCommit + toCall + extra));
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

function renderRecommendation(s) {
  const el = document.getElementById("recommendation");
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
    actionText = `${verb} ${formatUnit(rec.chips, s)}${suffix}`;
  }
  const dist = rec.gate_distribution || [];
  const distFmt = dist.map((p, i) => {
    const names = ["F", "C", "R"];
    return `${names[i]} ${(p * 100).toFixed(0)}%`;
  }).join("  ");
  const vBB = rec.value_bb;
  const sign = vBB >= 0 ? "+" : "-";
  const absBB = Math.abs(vBB);
  const vDisp = UI.unit === "bb"
    ? `${sign}${absBB.toFixed(2)}bb`
    : `${sign}$${(absBB * (s?.chip_scale?.dollars_per_bb ?? 2)).toFixed(2)}`;
  el.innerHTML = `
    <div class="rec-line">Network: <span class="rec-action">${actionText}</span>
      <span class="muted">· value ${vDisp}</span>
    </div>
    <div class="rec-detail">${distFmt} · \u03B2(${rec.beta_alpha.toFixed(1)}, ${rec.beta_beta.toFixed(1)})</div>
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
    row.innerHTML = `
      <span class="h-street">${h.street}</span>
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
  if (UI.selectedSlot && UI.selectedSlot.key === key && UI.selectedSlot.index === index) {
    UI.selectedSlot = null;
  } else {
    UI.selectedSlot = { key, index };
  }
  render(UI.lastState);
}

function onSlotDoubleClick(key, index) {
  const s = UI.lastState;
  if (!s) return;
  const spec = s.card_spec[key];
  if (spec[index] === null) return;
  spec[index] = null;
  UI.selectedSlot = { key, index };
  postCards();
}

function onGridCardClick(cardInt) {
  const s = UI.lastState;
  if (!s) return;
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
    if (!s || s.num_seats >= 6) { icon.setAttribute("hidden", ""); return; }
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
  document.getElementById("ocr-refresh-windows").addEventListener("click", () => refreshOcrWindows());
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

async function refreshOcrWindows() {
  const select = document.getElementById("ocr-window-match");
  const prev = select.value;
  try {
    const data = await getJSON("/ocr/windows");
    const titles = data.windows || [];
    select.innerHTML = '<option value="">— pick window —</option>';
    for (const t of titles) {
      const opt = document.createElement("option");
      opt.value = t;
      opt.textContent = t;
      select.appendChild(opt);
    }
    if (prev && titles.includes(prev)) select.value = prev;
  } catch (e) {
    setOcrStatusError(`Window list failed: ${e.message}`);
  }
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
  const match = document.getElementById("ocr-window-match").value;
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
      UI.ocrLastStatus = st;
      UI.ocrRunning = !!st.running;
      setOcrToggleUI();
      renderOcrStatus(st);
      if (st.running) await fetchState();
      else stopOcrPolling();
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
    if (st.running) startOcrPolling();
  } catch (_) { /* ocr endpoints may be unavailable; ignore */ }
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
  fetchState();
  refreshOcrWindows();
  refreshOcrStatusOnLoad();
}

document.addEventListener("DOMContentLoaded", init);
