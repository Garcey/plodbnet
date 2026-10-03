// Study / Trainer client, part 2 of 7 — the table: the home games' table
// renderer fed a view of the Study / Trainer state (seats, boards, hero cards,
// pot, bets, dealer button); the seat menu and stack editor; adding seats and
// dragging the dealer button; the card picker and card grid; card entry by
// click and by keyboard.
//
// Plain scripts, no build step: index.html loads app.core.js, app.table.js,
// app.play.js, app.study.js, app.trainer.js, app.topbar.js and app.js in
// that order, and they share one global scope — any file may call a
// function from any other. Code that runs while a file LOADS may use only
// the files before it; everything else starts from init() in app.js.
"use strict";

// --- Table chrome: the undo button --------------------------------------------

function renderTableChrome(s) {
  document.getElementById("undo-btn").disabled = !s.can_undo;
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

// --- Card picker sheet (phones) and dialog ---------------------------------

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
      ? `What-if: another card for ${slotWords(s, sel.key, sel.index)}`
      : `Card for ${slotWords(s, sel.key, sel.index)}`;
    const filled = !s.trainer && s.card_spec[sel.key] &&
      s.card_spec[sel.key][sel.index] !== null &&
      s.card_spec[sel.key][sel.index] !== undefined;
    document.getElementById("picker-clear-btn").hidden = !filled;
    if (!modal.open) openDialog(modal);
  } else {
    if (modal.open) {
      modal._syncClose = true;
      closeDialog(modal);
    }
    if (grid && home && grid.parentElement !== home) home.appendChild(grid);
  }
}

// Closing the sheet (✕, Escape, a tap on the dimmed table) always drops the
// selected slot, so nothing reopens it (ST-006).
function closeCardPicker() {
  UI.selectedSlot = null;
  UI.pickerClosedAt = Date.now();
  cancelTrainerPick();
  if (UI.lastState) render(UI.lastState);
}

// MOB-007: on a phone the picker opens from one big button at the next
// empty slot (it then walks from slot to slot by itself).
function openPickerAtNextSlot() {
  const s = UI.lastState;
  if (!s || s.trainer) return;
  const next = nextEmptySlot(s, null) || { key: "hero_hole", index: 0 };
  UI.selectedSlot = next;
  render(s);
}

function setupCardPicker() {
  const modal = document.getElementById("card-picker-modal");
  document.getElementById("picker-close-btn").addEventListener("click", closeCardPicker);
  modal.addEventListener("close", () => {
    if (modal._syncClose) { modal._syncClose = false; return; }
    closeCardPicker();   // Escape or a backdrop tap
  });
  document.getElementById("picker-clear-btn").addEventListener("click", () => {
    const s = UI.lastState, sel = UI.selectedSlot;
    if (!s || !sel || s.trainer) return;
    s.card_spec[sel.key][sel.index] = null;
    commitLocalCards();  // slot stays selected; picker stays open for the re-place
  });
  document.addEventListener("click", (e) => {
    if (e.target.closest("[data-open-picker]")) openPickerAtNextSlot();
  });
  MOBILE_MQ.addEventListener("change", () => {
    closeSeatMenu();
    const ed = document.getElementById("stack-edit-input");
    if (ed) ed.remove();
    if (UI.lastState) render(UI.lastState);
  });
}

// --- Your seat (ST-015 / MOB-013) -----------------------------------------------
// The hero always sits at the bottom; "Your seat" picks the position by moving
// the D button — the same thing dragging the button does, but visible,
// keyboard-reachable and usable on a touch screen.
function heroSeatOptions(s) {
  const n = s.num_seats;
  const names = {};
  for (const seat of s.seats) {
    names[((seat.seat - s.button_seat) % n + n) % n] = seat.position;
  }
  const out = [];
  // Turn order: the seat after the button acts first, the button last.
  for (let k = 1; k <= n; k++) {
    const off = k % n;
    out.push({ off, name: names[off] || `Seat ${k}`, button: ((s.hero_seat - off) % n + n) % n });
  }
  return out;
}

function renderHeroSeatSelect(s) {
  const sel = document.getElementById("hero-pos-select");
  if (!sel) return;
  const opts = heroSeatOptions(s);
  const key = opts.map((o) => `${o.button}:${o.name}`).join("|");
  if (sel.dataset.key !== key) {
    sel.innerHTML = opts.map((o) => `<option value="${o.button}">${escapeHTML(o.name)}</option>`).join("");
    sel.dataset.key = key;
  }
  if (document.activeElement !== sel) sel.value = String(s.button_seat);
}

// ST-018: seat, button, ante and New hand changes restart the hand. Ask first
// when there is work to lose (Undo can't bring it back after the restart).
async function confirmHandReset(what, { clearsCards = false } = {}) {
  const s = UI.lastState;
  if (!s || s.trainer) return true;
  const n = (s.history || []).length;
  const cards = clearsCards ? collectUsedCards(s).size : 0;
  if (!n && !cards) return true;
  const parts = [];
  if (cards) parts.push(`the ${cards} card${cards === 1 ? "" : "s"}`);
  if (n) parts.push(`the ${n} action${n === 1 ? "" : "s"}`);
  return confirmDialog({
    title: `${what}?`,
    body: `This clears ${parts.join(" and ")} entered so far${!clearsCards && n ? " (the cards stay)" : ""}.`,
    ok: what,
    danger: true,
  });
}

// --- The table (2026-10-01) ------------------------------------------------------
// Study and Trainer draw on the home games' table — the owner: "bring the home
// games' look to Study and Trainer", one card face everywhere. games.table.js
// (loaded before this file) is that renderer: persistent seats, cards, chips
// and pots that animate from one state to the next. It reads a home-games
// table state, so feltView() turns the Study / Trainer state into one (money
// in chips: its "cents" are chips here), and it calls a few helpers of the
// home games' core (games.js) — FELT_HG.core below speaks Study's units and
// markup instead. Study's own controls ride on the same elements: card places
// (markSlots), the seat menu, the stack editor, the dealer button, "+" seats.
const FELT_HG = (globalThis.HG = globalThis.HG || {});

class FeltMarkup {
  constructor(s) { this.s = s; }
  toString() { return this.s; }
}
function feltMarkupOf(v) {
  if (v instanceof FeltMarkup) return v.s;
  if (Array.isArray(v)) return v.map(feltMarkupOf).join("");
  return escapeHTML(v);
}
// The renderer writes markup ONLY through html`` (every value escaped) and put()
// (markup as markup, anything else as text) — the home games' rules (FE-003).
function feltHTML(strings, ...values) {
  let out = strings[0];
  for (let i = 0; i < values.length; i++) out += feltMarkupOf(values[i]) + strings[i + 1];
  return new FeltMarkup(out);
}
function feltPut(el, content) {
  if (content instanceof FeltMarkup) el.innerHTML = content.s;
  else if (content && content.nodeType) { el.textContent = ""; el.appendChild(content); }
  else el.textContent = content == null ? "" : String(content);
  return el;
}
const FELT_REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)");
if (!FELT_HG.core) {
  FELT_HG.core = {
    G: { state: null, prefs: { anim: "auto", bubbles: false } },
    html: feltHTML,
    put: feltPut,
    // money on this table is in chips; amounts read in the viewer's unit (bb / $)
    fmtAmt: (chips, v) => formatUnit(chips, (v && v._src) || UI.lastState),
    chipsToCents: (chips) => Number(chips) || 0,
    actionKind: (x) => (x.action === 0 ? "fold" : x.action === 1 ? (x.chips > 0 ? "call" : "check") : x.action === 7 ? "allin" : "raise"),
    motionOn: () => !FELT_REDUCED_MOTION.matches,
  };
}
if (!FELT_HG.ui) {
  FELT_HG.ui = {
    openPlayer: (i) => onFeltSeat(i),
    openSit() {}, openRequest() {}, onClock() {},
    noteFor: () => ({ tag: "none" }),
  };
}

// The seats' discs show their positions, each in its own colour; the name line
// spells it out ("Cutoff"), and says Hero (Study) / You (Trainer) at the hero.
const POSITION_HUES = { BTN: 42, SB: 205, BB: 262, UTG: 150, "UTG+1": 122, MP: 95, LJ: 178, HJ: 318, CO: 12 };
const POSITION_NAMES = {
  BTN: "Button", SB: "Small blind", BB: "Big blind", UTG: "Under the gun", "UTG+1": "UTG+1",
  MP: "Middle", LJ: "Lojack", HJ: "Hijack", CO: "Cutoff",
};
function seatHue(position, seat) {
  return POSITION_HUES[position] ?? (seat * 57 + 20) % 360;
}

// One board as the table draws it. Study: its five PLACES (null = no card
// entered yet — a card taken back leaves its own place empty). Trainer: the
// cards dealt so far, flop then turn then river (they come in order, and the
// renderer deals them in with a flip).
function feltBoard(s, b) {
  const spec = s.card_spec || {};
  const flop = (b === 0 ? spec.flop_a : spec.flop_b) || [];
  const street = (arr) => (arr && arr.length > b && arr[b] !== undefined ? arr[b] : null);
  if (!s.trainer) {
    return { slots: [flop[0] ?? null, flop[1] ?? null, flop[2] ?? null, street(spec.turn), street(spec.river)] };
  }
  return { flop: flop.filter((c) => c !== null && c !== undefined), turn: street(spec.turn), river: street(spec.river) };
}

function feltSeat(x, s, holeN) {
  const isHero = x.seat === s.hero_seat;
  const spec = s.card_spec || {};
  let hole;
  if (isHero) {
    // Study: the places of a hand being entered (null = still empty);
    // Trainer: the hand dealt to you
    hole = (spec.hero_hole || []).map((c) => (c === null || c === undefined ? (s.trainer ? -1 : null) : c));
  } else if (Array.isArray(x.hole) && x.hole.length && x.hole.every((c) => c !== null && c >= 0)) {
    hole = x.hole.slice();  // (Trainer: tabled at the end of a hand, and in the review)
  } else {
    hole = new Array(holeN).fill(-1);
  }
  const position = x.position || `Seat ${x.seat + 1}`;
  return {
    seat: x.seat,
    empty: x.participant === false,
    user_id: x.seat,
    name: isHero ? (s.trainer ? "You" : "Hero") : (POSITION_NAMES[position] || position),
    av_text: position,
    hue: seatHue(position, x.seat),
    position: "",
    stack_cents: x.stack_chips,
    in_hand: x.participant !== false,
    folded: !!x.folded,
    all_in: !!x.all_in,
    is_actor: !!x.is_actor,
    is_hero: isHero,
    hole,
    hand_desc: isHero && Array.isArray(s.hero_hand_desc) ? s.hero_hand_desc : null,
    committed_this_street_cents: Number(x.committed_this_street_chips) || 0,
    present: true,
  };
}

// The Study / Trainer state as the home games' table reads it.
function feltView(s) {
  const holeN = s.card_spec ? s.card_spec.hero_hole.length : 5;
  const bb = (s.chip_scale && s.chip_scale.bb_chips) || 10000;
  const walk = historyCommits(s);
  const history = (s.history || []).map((h, i) => ({
    seat: h.seat,
    street: h.street,
    action: h.action === "Fold" ? 0 : h.action === "CheckCall" ? 1 : h.action === "AllIn" ? 7 : 2,
    chips: Number(h.chips) || 0,
    cents: Number(h.chips) || 0,
    to_cents: walk[i] ? walk[i].after : Number(h.chips) || 0,
  }));
  const t = s.trainer;
  const over = !!s.terminal;
  const rewards = t && over && Array.isArray(t.rewards_bb) ? t.rewards_bb : null;
  return {
    id: `${t ? "trainer" : "study"}:${s.format || ""}`,
    num_seats: s.num_seats,
    hero_seat: s.hero_seat,
    my_seat: s.hero_seat,
    my_user_id: -1,
    hole_count: holeN,
    game: { dealt: holeN, burns: 0 },
    status: "open",
    phase: over ? "showdown" : "in_hand",
    hand_no: t ? t.hand_no : 1,
    street: s.street,
    actor: s.actor,
    button_seat: s.button_seat,
    seats: s.seats.map((x) => feltSeat(x, s, holeN)),
    history,
    pot_cents: Number(s.pot_chips) || 0,
    settled_pot_chips: s.settled_pot_chips ?? s.pot_chips,
    stakes: { bb_cents: bb, bb_chips: bb, ante_cents: (s.chip_scale && s.chip_scale.ante_chips) || 0 },
    board: { a: feltBoard(s, 0), b: isSingleBoard(s) ? null : feltBoard(s, 1) },
    runout: { active: false, blocking: false },
    hand_deltas_cents: rewards ? rewards.map((r) => Math.round((Number(r) || 0) * bb)) : [],
    decision_secs: 0,
    _src: s,
  };
}

let FELT_PREV = null;     // the view drawn last (the renderer animates prev -> next)
let FELT_UNIT_KEY = null; // bb / $ and the rate it was drawn in

function renderTable(s) {
  if (!FELT_HG.table || !s.card_spec) return;
  const v = feltView(s);
  FELT_HG.core.G.state = v;
  const unitKey = `${UI.unit}|${UI.unit === "$" ? dollarsPerBB(s) : ""}`;
  const prev = FELT_PREV && FELT_PREV.id === v.id ? FELT_PREV : null;
  const wrap = document.getElementById("stage-wrap");
  wrap.classList.remove("loading");
  wrap.classList.toggle("study-table", !s.trainer);
  FELT_HG.table.render(v, prev, { unitChanged: FELT_UNIT_KEY !== null && FELT_UNIT_KEY !== unitKey });
  FELT_PREV = v;
  FELT_UNIT_KEY = unitKey;
  const focus = focusedSlotKey();
  markSlots(s);
  restoreSlotFocus(focus);
  markSeats(s);
  placeSeatAdds(s);
  const dealer = document.getElementById("dealer-btn");
  dealer.classList.toggle("draggable", !s.trainer);
  dealer.title = s.trainer ? "The dealer button" : "The dealer button — drag it to another seat, or choose Your seat above";
  renderTableSummary(s);
}

// What a click on a seat does: Study opens its menu; the Trainer's seats are
// the table's (nothing to change there).
function onFeltSeat(i) {
  const s = UI.lastState;
  if (!s || s.trainer) return;
  openSeatMenu(i);
}

// Study: every seat is a button (its menu); the stacks say they can be changed.
function markSeats(s) {
  const study = !s.trainer;
  for (const seat of s.seats) {
    const el = document.querySelector(`#seats .seat[data-seat="${seat.seat}"]`);
    if (!el) continue;
    el.classList.toggle("study-seat", study);
    const stack = el.querySelector(".seat-stack");
    if (stack) {
      const editable = study && !seat.all_in;
      stack.classList.toggle("editable", editable);
      stack.title = editable ? "Change this stack" : "";
    }
    const main = el.querySelector(".seat-main");
    if (main && study) {
      main.setAttribute("aria-haspopup", "menu");
      main.setAttribute("aria-label", `${seat.position}${seat.is_hero ? " (Hero)" : ""}, ${seat.all_in ? "all-in" : `stack ${formatUnit(seat.stack_chips, s)}`}. Seat options`);
    } else if (main) main.removeAttribute("aria-haspopup");
  }
}

// --- Card places on the table -------------------------------------------------------
// The hero's cards and the boards' places are buttons (A11Y-017): a click (or
// Enter) selects one — Study: the card grid fills it; Trainer review: a what-if
// swap — a double click (or Delete) empties it (Study). Each says what it holds.
// With none selected, Study marks the place a card goes next (.slot-next).
function slotAt(s, el, key, index, next) {
  if (!el) return;
  const spec = s.card_spec || {};
  const arr = spec[key];
  const value = arr ? arr[index] : undefined;
  if (value === undefined) {
    delete el.dataset.slot; delete el.dataset.index;
    el.classList.remove("slot-pick", "slot-sel", "slot-next", "slot-mod");
    el.removeAttribute("tabindex"); el.removeAttribute("role"); el.removeAttribute("aria-label");
    return;
  }
  const has = value !== null;
  const rv = s.trainer && s.trainer.review;
  // (the Trainer's review: a what-if swap, at your own graded decisions only — onSlotClick)
  const interactive = !s.trainer || (!!rv && has && !reviewNodeUngraded(rv.node_current));
  const selected = !!UI.selectedSlot && UI.selectedSlot.key === key && UI.selectedSlot.index === index;
  el.dataset.slot = key;
  el.dataset.index = String(index);
  el.classList.toggle("slot-pick", interactive);
  el.classList.toggle("slot-sel", selected);
  el.classList.toggle("slot-next", !!next && next.key === key && next.index === index);
  el.classList.toggle("slot-mod", (s.modified_cards || []).some((m) => m.slot_key === key && m.index === index));
  if (interactive) { el.tabIndex = 0; el.setAttribute("role", "button"); }
  else { el.removeAttribute("tabindex"); el.removeAttribute("role"); }
  const words = slotWords(s, key, index);
  const Words = `${words[0].toUpperCase()}${words.slice(1)}`;
  el.setAttribute("aria-label", `${Words}: ${has ? cardName(value) : "empty"}${selected ? ", selected" : ""}`);
  if (selected) el.setAttribute("aria-pressed", "true"); else el.removeAttribute("aria-pressed");
}

function markSlots(s) {
  const next = !s.trainer && !UI.selectedSlot ? nextEmptySlot(s, null) : null;
  document.querySelectorAll("#hero-hole > .card").forEach((el, k) => slotAt(s, el, "hero_hole", k, next));
  const boards = isSingleBoard(s) ? [["a", 0]] : [["a", 0], ["b", 1]];
  for (const [k, b] of boards) {
    document.querySelectorAll(`#board-${k} > .slot-card`).forEach((el, j) => {
      if (j < 3) slotAt(s, el, b === 0 ? "flop_a" : "flop_b", j, next);
      else slotAt(s, el, j === 3 ? "turn" : "river", b, next);
    });
  }
}

function focusedSlotKey() {
  const a = document.activeElement;
  return a && a.dataset && a.dataset.slot ? `${a.dataset.slot}:${a.dataset.index}` : null;
}
function restoreSlotFocus(key) {
  if (!key) return;
  const [slot, index] = key.split(":");
  const el = document.querySelector(`#stage [data-slot="${slot}"][data-index="${index}"]`);
  if (el) el.focus();
}

// Two made-hand labels ("1 a pair of 8s" / "2 ...") under the table, like the
// home games' — so a stealth set or straight is hard to miss while deciding.
function renderHeroHandLabels(s) {
  const el = document.getElementById("hero-hand-labels");
  if (!el) return;
  const desc = s.hero_hand_desc || [];
  const rows = [];
  if (isSingleBoard(s)) {
    if (desc[0]) rows.push([null, desc[0]]);  // one board — one label, no number
  } else {
    if (desc[0]) rows.push(["1", desc[0]]);
    if (desc[1]) rows.push(["2", desc[1]]);
  }
  const html = rows.map(([tag, text]) => `<span class="hero-hand-label" title="${escapeHTML(text)}">${tag ? `<span class="hhl-tag">${tag}</span>` : ""}<span class="hhl-txt">${escapeHTML(text)}</span></span>`).join("");
  if (el.dataset.k !== html) { el.dataset.k = html; el.innerHTML = html; }
}

// --- Seat menu (ST-009 / MOB-007 / ST-015) ------------------------------------------
function closeSeatMenu() {
  const pop = document.getElementById("seat-pop");
  if (pop) pop.remove();
  document.removeEventListener("pointerdown", onSeatMenuOutside, true);
}
function onSeatMenuOutside(e) {
  const pop = document.getElementById("seat-pop");
  if (pop && !pop.contains(e.target)) closeSeatMenu();
}

function seatMain(seatIdx) {
  return document.querySelector(`#seats .seat[data-seat="${seatIdx}"] .seat-main`);
}

function openSeatMenu(seatIdx) {
  const already = document.getElementById("seat-pop");
  closeSeatMenu();
  const s = UI.lastState;
  if (!s || s.trainer || (already && already.dataset.seat === String(seatIdx))) return;
  const seat = s.seats[seatIdx];
  const node = seatMain(seatIdx);
  if (!seat || !node) return;
  const n = s.num_seats;
  const off = ((seat.seat - s.button_seat) % n + n) % n;
  const heroButton = ((s.hero_seat - off) % n + n) % n;   // button that seats Hero here
  const items = [];
  if (!seat.all_in) {
    items.push({ act: "stack", label: `Change stack (${formatUnit(seat.stack_chips, s)})` });
    // FEAT-020: one click instead of six stack edits.
    if (s.seats.some((x) => x.participant !== false && x.stack_chips !== seat.stack_chips)) {
      items.push({ act: "all", label: `Give everyone ${formatUnit(seat.stack_chips, s)}` });
    }
  }
  if (!seat.is_hero) {
    items.push({ act: "sit", label: `Make this Hero's seat (${seat.position})` });
    if (n > 2) items.push({ act: "remove", label: "Remove this player", danger: true });
  }
  if (!items.length) return;
  const pop = document.createElement("div");
  pop.id = "seat-pop";
  pop.className = "preset-pop seat-pop";
  pop.dataset.seat = String(seatIdx);
  pop.setAttribute("role", "menu");
  pop.setAttribute("aria-label", `${seat.position} options`);
  pop.innerHTML = `<div class="preset-pop-title">${escapeHTML(seat.position)}${seat.is_hero ? " · Hero" : ""}</div>`
    + items.map((it) => `<button type="button" role="menuitem" class="seat-pop-item${it.danger ? " danger" : ""}" data-act="${it.act}">${escapeHTML(it.label)}</button>`).join("");
  document.body.appendChild(pop);
  const r = node.getBoundingClientRect();
  const popW = pop.offsetWidth, popH = pop.offsetHeight;
  const left = Math.max(8, Math.min(r.left + r.width / 2 - popW / 2, window.innerWidth - popW - 8));
  let top = r.bottom + 6;
  if (top + popH > window.innerHeight - 8) top = Math.max(8, r.top - popH - 6);
  pop.style.left = `${left}px`;
  pop.style.top = `${top}px`;
  pop.addEventListener("click", async (e) => {
    const b = e.target.closest("[data-act]");
    if (!b) return;
    closeSeatMenu();
    const cur = UI.lastState;
    if (!cur || cur.trainer) return;
    if (b.dataset.act === "stack") openStackEditor(cur.seats[seatIdx], cur);
    else if (b.dataset.act === "all") {
      const chips = cur.seats[seatIdx].stack_chips;
      postConfig({ starting_stacks: cur.seats.map(() => chips) });
    }
    else if (b.dataset.act === "sit") {
      if (await confirmHandReset("Change Hero's seat")) postSeats({ button_seat: heroButton });
    } else if (b.dataset.act === "remove") {
      if (await confirmHandReset("Remove this player")) removeSeat(seatIdx, cur);
    }
  });
  pop.addEventListener("keydown", (e) => {
    const list = [...pop.querySelectorAll("[role=menuitem]")];
    const i = list.indexOf(document.activeElement);
    if (e.key === "Escape") { e.preventDefault(); closeSeatMenu(); node.focus(); }
    else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      list[(i + (e.key === "ArrowDown" ? 1 : -1) + list.length) % list.length].focus();
    } else if (e.key === "Tab") closeSeatMenu();
  });
  document.addEventListener("pointerdown", onSeatMenuOutside, true);
  const first = pop.querySelector("[role=menuitem]");
  if (first) first.focus();
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

// The stack editor: a box over the seat's stack (Enter / clicking away saves,
// Escape cancels).
function openStackEditor(seat, s) {
  const existing = document.getElementById("stack-edit-input");
  if (existing) existing.remove();
  closeSeatMenu();
  const box = document.getElementById("stage-box");
  const stackEl = document.querySelector(`#seats .seat[data-seat="${seat.seat}"] .seat-stack`);
  if (!box || !stackEl) return;
  const r = stackEl.getBoundingClientRect(), b = box.getBoundingClientRect();

  const input = document.createElement("input");
  input.type = "number";
  input.inputMode = "decimal";
  input.step = UI.unit === "bb" ? "0.1" : "1";
  input.min = "0";
  input.id = "stack-edit-input";
  input.className = "stack-edit-input";
  input.setAttribute("aria-label", `${seat.position} stack, in ${UI.unit === "bb" ? "big blinds" : "dollars"}. Enter to save, Escape to cancel`);
  const original = unitInputValue(seat.stack_chips, s);
  input.value = original;
  const width = 96;
  input.style.width = `${width}px`;
  input.style.left = `${r.left + r.width / 2 - b.left - width / 2}px`;
  input.style.top = `${r.top + r.height / 2 - b.top - 16}px`;
  box.appendChild(input);
  input.focus();
  input.select();

  let committed = false;
  const commit = async () => {
    if (committed) return;
    committed = true;
    const unchanged = input.value.trim() === original;
    const chips = parseToChips(input.value, s);
    input.remove();
    if (unchanged) return;
    if (chips === null || chips <= 0) {
      showToast("Enter a stack above zero.");
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
  // ST-022: clicking away SAVES a changed, valid value (it used to throw
  // the typed number away); Escape still cancels.
  input.addEventListener("blur", () => {
    const chips = parseToChips(input.value, s);
    if (chips !== null && chips > 0) commit();
    else cancel();
  });
}

// A plain-text summary of the table for screen readers (A11Y-017).
function renderTableSummary(s) {
  const el = document.getElementById("table-summary");
  if (!el) return;
  const parts = [`${s.num_seats} players, pot ${formatUnit(s.pot_chips, s)}.`];
  for (const seat of s.seats) {
    if (seat.participant === false) continue;
    const who = seat.is_hero ? `${seat.position} (${s.trainer ? "you" : "Hero"})` : seat.position;
    const state = seat.folded ? "folded" : seat.all_in ? "all-in" : formatUnit(seat.stack_chips, s);
    parts.push(`${who}: ${state}${seat.committed_this_street_chips > 0 ? `, bet ${formatUnit(seat.committed_this_street_chips, s)}` : ""}.`);
  }
  const cards = (arr) => (arr || []).filter((c) => c !== null && c !== undefined).map(cardName);
  const spec = s.card_spec || {};
  const hole = cards(spec.hero_hole);
  if (hole.length) parts.push(`${s.trainer ? "Your" : "Hero's"} cards: ${hole.join(", ")}.`);
  const single = isSingleBoard(s);
  const boardA = [...cards(spec.flop_a), ...cards([spec.turn && spec.turn[0], spec.river && spec.river[0]])];
  const boardB = [...cards(spec.flop_b), ...cards([spec.turn && spec.turn[1], spec.river && spec.river[1]])];
  if (boardA.length) parts.push(`${single ? "Board" : "Board A"}: ${boardA.join(", ")}.`);
  if (!single && boardB.length) parts.push(`Board B: ${boardB.join(", ")}.`);
  if (s.actor !== null && s.actor !== undefined && s.seats[s.actor]) {
    parts.push(`${s.seats[s.actor].position} to act.`);
  }
  el.textContent = parts.join(" ");
}

// --- Card grid ----------------------------------------------------------------
// 13 x 4 card matrix. Built ONCE (A11Y-016) and updated in place: rebuilding
// 52 buttons every render dropped keyboard focus after each pick. Names are
// spoken as "Ace of spades" (A11Y-021), not "A black spade suit A".
const RANK_NAMES = {
  "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
  "8": "Eight", "9": "Nine", T: "Ten", J: "Jack", Q: "Queen", K: "King", A: "Ace",
};
const SUIT_NAMES = { c: "clubs", d: "diamonds", h: "hearts", s: "spades" };
function cardName(cardInt) {
  const c = cardToString(cardInt);
  return c ? `${RANK_NAMES[c.rank]} of ${SUIT_NAMES[c.suit]}` : "";
}

function ensureCardGrid() {
  const grid = document.getElementById("card-grid");
  if (grid.childElementCount === 52) return grid;
  grid.innerHTML = "";
  grid.setAttribute("role", "group");
  grid.setAttribute("aria-label", "Cards");
  // 13 rows × 4 cols in DOM order (ranks A→2, suits c,d,h,s); CSS lays the
  // desktop rail and the phone picker out as ranks across, suits down.
  for (let rank = 12; rank >= 0; rank--) {
    for (let suit = 0; suit < 4; suit++) {
      const cardInt = rank * 4 + suit;
      const info = cardToString(cardInt);
      const b = document.createElement("button");
      b.type = "button";
      b.className = "card-btn suit-" + info.suit;
      b.dataset.card = String(cardInt);
      b.setAttribute("aria-label", cardName(cardInt));
      b.innerHTML = `
        <span class="corner-tl" aria-hidden="true">
          <span class="corner-rank">${info.rank}</span>
          <span class="corner-suit">${info.glyph}</span>
        </span>
        <span class="corner-br" aria-hidden="true">${info.rank}</span>
      `;
      grid.appendChild(b);
    }
  }
  if (!grid._wired) {
    grid._wired = true;
    grid.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-card]");
      if (!b || b.disabled) return;
      onGridCardClick(parseInt(b.dataset.card, 10));
    });
  }
  return grid;
}

// Words for a card slot: "your hole card 2", "flop A card 1", "the turn (board B)".
function slotWords(s, key, index) {
  const single = isSingleBoard(s);
  if (key === "hero_hole") return `${s.trainer ? "your" : "Hero's"} hole card ${index + 1}`;
  if (key === "flop_a" || key === "flop_b") {
    return single ? `flop card ${index + 1}` : `flop ${key === "flop_a" ? "A" : "B"} card ${index + 1}`;
  }
  const street = key === "turn" ? "turn" : "river";
  return single ? `the ${street}` : `the ${street} (board ${index === 0 ? "A" : "B"})`;
}

function renderCardGrid(s) {
  const grid = ensureCardGrid();
  const used = collectUsedCards(s);
  const keepFocus = grid.contains(document.activeElement);
  for (const b of grid.children) {
    const c = parseInt(b.dataset.card, 10);
    const isUsed = used.has(c);
    b.classList.toggle("used", isUsed);
    b.disabled = isUsed;
    b.setAttribute("aria-label", `${cardName(c)}${isUsed ? " (on the table)" : ""}`);
  }
  // A used card can't hold focus once disabled: move to the next free one.
  if (keepFocus && document.activeElement && document.activeElement.disabled) {
    const next = [...grid.children].find((b) => !b.disabled);
    if (next) next.focus();
  }
  // CPY-015: say what a click does now — cards fill the next empty slot.
  const hint = document.getElementById("slot-hint");
  if (!hint) return;
  if (s.trainer) {
    const sel = UI.selectedSlot;
    hint.textContent = sel ? `Pick a card for ${slotWords(s, sel.key, sel.index)}` : "";
    return;
  }
  const target = UI.selectedSlot || nextEmptySlot(s, null);
  hint.textContent = target
    ? `Next: ${slotWords(s, target.key, target.index)}`
    : "All cards placed — click a card on the table to change it";
}

// --- Slot / grid interactions ----------------------------------------------

function onSlotClick(key, index) {
  const s = UI.lastState;
  // A tap that closed the phone card picker must not land on the slot under
  // the finger and open it again (ST-006).
  if (UI.pickerClosedAt && Date.now() - UI.pickerClosedAt < 400) return;
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
      const w = slotWords(s, key, index);
      showToast(`${w[0].toUpperCase()}${w.slice(1)} isn't dealt yet at this decision — `
        + "step to a later one to change it.", "info");
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
  if (!sel) { showToast("All the cards are placed — click a card on the table to change it.", "info"); return false; }
  if (collectUsedCards(s).has(cardInt)) { showToast("That card is already on the table.", "info"); return false; }
  const spec = s.card_spec[sel.key];
  if (!spec || sel.index >= spec.length) return false;
  spec[sel.index] = cardInt;
  UI.selectedSlot = nextEmptySlot(s, sel);
  commitLocalCards();
  return true;
}

// Keyboard entry: a rank (2-9 T J Q K A) then a suit (c d h s) places that card;
// Backspace takes back the most recent card; Escape drops the selection.
const KEY_RANKS = RANK_STRINGS;   // one copy of the rank order (FE-021)
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

// --- The table's own clicks and keys -----------------------------------------------
// Card places: click / Enter selects one, a double click / Delete empties it
// (Study). A Study stack: a click changes it (before the seat's own menu).
function setupTableKeyboard() {
  const stage = document.getElementById("stage");
  stage.addEventListener("click", (e) => {
    const slot = e.target.closest("[data-slot]");
    if (slot && slot.classList.contains("slot-pick")) onSlotClick(slot.dataset.slot, parseInt(slot.dataset.index, 10));
  });
  stage.addEventListener("dblclick", (e) => {
    const slot = e.target.closest("[data-slot]");
    if (!slot || !slot.classList.contains("slot-pick")) return;
    e.preventDefault();
    onSlotDoubleClick(slot.dataset.slot, parseInt(slot.dataset.index, 10));
  });
  stage.addEventListener("keydown", (e) => {
    const slot = e.target.closest && e.target.closest("[data-slot]");
    if (!slot || !slot.classList.contains("slot-pick")) return;
    const key = slot.dataset.slot;
    const index = parseInt(slot.dataset.index, 10);
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      e.stopPropagation();
      onSlotClick(key, index);
    } else if (e.key === "Delete" || e.key === "Backspace") {
      const s = UI.lastState;
      if (!s || s.trainer) return;
      e.preventDefault();
      e.stopPropagation();
      onSlotDoubleClick(key, index);
    }
  });
  // (capturing: it runs before the seat's own click, which opens its menu)
  document.getElementById("seats").addEventListener("click", (e) => {
    const stack = e.target.closest(".seat-stack.editable");
    if (!stack) return;
    const s = UI.lastState;
    const seatEl = stack.closest(".seat");
    if (!s || s.trainer || !seatEl) return;
    e.stopPropagation();
    const seat = s.seats[parseInt(seatEl.dataset.seat, 10)];
    if (seat && !seat.all_in) openStackEditor(seat, s);
  }, true);
}

// --- "+": add a player between two seats (Study, a mouse, fewer than 6) ------------
// One "+" sits on the rail halfway between each pair of neighbouring seats; it
// shows while the pointer is over the table (the Players control in the work
// bar does the same on a touch screen — MOB-013).
const FINE_POINTER = window.matchMedia("(hover: hover) and (pointer: fine)");
function placeSeatAdds(s) {
  const host = document.getElementById("seat-adds");
  if (!host) return;
  const want = !s.trainer && s.num_seats < 6 && FINE_POINTER.matches && FELT_HG.table && FELT_HG.table.seatCenter;
  if (!want) { host.innerHTML = ""; host.dataset.k = ""; return; }
  const stage = document.getElementById("stage");
  const w = stage.offsetWidth, h = stage.offsetHeight;
  if (!w || !h) return;
  const n = s.num_seats;
  const centers = [];
  for (let i = 0; i < n; i++) centers.push(FELT_HG.table.seatCenter(i));
  if (centers.some((c) => !c)) return;
  const cx = w / 2, cy = h / 2;
  const key = `${n}|${w}x${h}|${centers.map((c) => c.map(Math.round).join(",")).join(";")}`;
  if (host.dataset.k === key) return;
  host.dataset.k = key;
  host.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const a = centers[i], b = centers[(i + 1) % n];
    // halfway round the rail: the midpoint, pushed out to the seats' distance
    const mx = (a[0] + b[0]) / 2 - cx, my = (a[1] + b[1]) / 2 - cy;
    const ra = Math.hypot(a[0] - cx, a[1] - cy), rb = Math.hypot(b[0] - cx, b[1] - cy);
    const len = Math.hypot(mx, my) || 1, r = (ra + rb) / 2;
    const x = cx + (mx / len) * r, y = cy + (my / len) * r;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "seat-add";
    btn.dataset.after = String(i);
    btn.textContent = "+";
    btn.title = "Add a player here";
    btn.setAttribute("aria-label", "Add a player between these seats");
    btn.style.left = `${(x / w) * 100}%`;
    btn.style.top = `${(y / h) * 100}%`;
    host.appendChild(btn);
  }
}

function setupInsertHover() {
  const host = document.getElementById("seat-adds");
  if (!host) return;
  host.addEventListener("click", async (e) => {
    const btn = e.target.closest(".seat-add");
    if (!btn) return;
    e.stopPropagation();
    const s = UI.lastState;
    if (!s || s.trainer || s.num_seats >= 6) return;
    if (!(await confirmHandReset("Add a player"))) return;
    insertSeat(parseInt(btn.dataset.after, 10), s);
  });
  // the felt lays itself out again when its box changes size: the "+" follow
  if (window.ResizeObserver) {
    new ResizeObserver(() => requestAnimationFrame(() => {
      host.dataset.k = "";
      if (UI.lastState && UI.lastState.card_spec) placeSeatAdds(UI.lastState);
    })).observe(document.getElementById("stage-box"));
  }
}

// --- Dealer-button drag (Study) ---------------------------------------------------
function setupDealerDrag() {
  const button = document.getElementById("dealer-btn");
  const stage = document.getElementById("stage");
  const at = (e) => {
    const r = stage.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  button.addEventListener("pointerdown", (e) => {
    if (UI.mode === "trainer" || !UI.lastState || UI.lastState.trainer) return;
    e.preventDefault();
    UI.draggingButton = true;
    button.classList.add("dragging");
    button.setPointerCapture(e.pointerId);
  });
  button.addEventListener("pointermove", (e) => {
    if (!UI.draggingButton) return;
    const [x, y] = at(e);
    button.style.left = `${(x / stage.offsetWidth) * 100}%`;
    button.style.top = `${(y / stage.offsetHeight) * 100}%`;
  });
  const endDrag = (e) => {
    if (!UI.draggingButton) return;
    UI.draggingButton = false;
    button.classList.remove("dragging");
    try { button.releasePointerCapture(e.pointerId); } catch (_) {}
    const s = UI.lastState;
    if (!s || !FELT_HG.table) return;
    const [x, y] = at(e);
    let bestSeat = s.button_seat, bestDist = Infinity;
    for (let i = 0; i < s.num_seats; i++) {
      if (s.seats[i] && s.seats[i].participant === false) continue;
      const c = FELT_HG.table.seatCenter(i);
      if (!c) continue;
      const d2 = (c[0] - x) ** 2 + (c[1] - y) ** 2;
      if (d2 < bestDist) { bestDist = d2; bestSeat = i; }
    }
    if (bestSeat !== s.button_seat) postSeats({ button_seat: bestSeat });
    else FELT_HG.table.layout();  // back to its place
  };
  button.addEventListener("pointerup", endDrag);
  button.addEventListener("pointercancel", () => {
    UI.draggingButton = false;
    button.classList.remove("dragging");
    if (FELT_HG.table) FELT_HG.table.layout();
  });
}
