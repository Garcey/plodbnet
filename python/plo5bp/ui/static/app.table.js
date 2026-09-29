// Study / Trainer client, part 2 of 7 — the table: seat geometry (wide and
// phone layouts), seats, boards, hero cards, pot and dealer button; the
// seat menu; the card picker and card grid; card entry by click and by
// keyboard; dragging the dealer button and adding seats.
//
// Plain scripts, no build step: index.html loads app.core.js, app.table.js,
// app.play.js, app.study.js, app.trainer.js, app.topbar.js and app.js in
// that order, and they share one global scope — any file may call a
// function from any other. Code that runs while a file LOADS may use only
// the files before it; everything else starts from init() in app.js.
"use strict";

// --- Table chrome: the undo button and the "add a seat" marker --------------

function renderTableChrome(s) {
  document.getElementById("undo-btn").disabled = !s.can_undo;
  const insertIcon = document.getElementById("insert-icon");
  if (s.num_seats >= 6 || s.trainer || tableLayout().name !== "wide") {
    insertIcon.setAttribute("hidden", "");
    insertIcon.style.display = "none";
  } else {
    insertIcon.style.display = "";
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
  MOBILE_MQ.addEventListener("change", () => { if (UI.lastState) render(UI.lastState); });
  TALL_MQ.addEventListener("change", () => {
    closeSeatMenu();
    const ed = document.getElementById("stack-edit-input");
    if (ed) ed.remove();
    if (UI.lastState) { UI.lastStateKey = null; render(UI.lastState); }
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

// --- Table geometry (MOB-006 / ST-004 / MOB-007) ----------------------------------
// Two drawings of the same table. "wide" is the original 800 x 560 oval for
// desktops, tablets and phones held sideways. "tall" is a portrait table for
// phones held upright: the wide oval shrank to ~42% there, so stacks rendered
// at ~5px and card slots at 13 x 18px. The tall table draws at ~90% of its
// units on a 375px phone, with larger type, cards you can tap, and the
// dealer button pinned to its seat.
const TABLE_LAYOUTS = {
  wide: {
    name: "wide", w: 800, h: 560,
    center: { x: 400, y: 260 }, seatRx: 310, seatRy: 170,
    felt: {
      rim: [400, 260, 352, 202], edge: [400, 260, 340, 190],
      line: [400, 260, 300, 153], glow: [400, 222, 266, 116],
    },
    plate: { w: 92, h: 52, rx: 10, posY: -10, stackY: 6 },
    board: { w: 44, h: 60, gap: 6 },
    hole: { w: 32, h: 44, gap: 4 },
    boardA: [280, 207], boardB: [280, 285], boardSingle: [280, 246],
    heroHole: [400, 500],
    labels: { y: 472, dy: 16 },
    pot: [400, 170], potW: 160,
    potKeepOut: { x1: 326, y1: 142, x2: 474, y2: 198 },
    dealerOffset: 38, betOffset: 64,
    mini: { w: 24, h: 33, gap: 3, rankY: 15, suitY: 28 },
    menuDot: [38, -17],
  },
  tall: {
    name: "tall", w: 400, h: 536,
    center: { x: 200, y: 250 }, seatRx: 150, seatRy: 222,
    felt: {
      rim: [200, 250, 184, 236], edge: [200, 250, 176, 228],
      line: [200, 250, 148, 196], glow: [200, 220, 122, 164],
    },
    plate: { w: 104, h: 48, rx: 10, posY: -6, stackY: 15 },
    // Boards stay 182 units wide so seats at mid-height (4-handed) clear them.
    board: { w: 34, h: 47, gap: 3 },
    hole: { w: 36, h: 50, gap: 3 },
    boardA: [109, 172], boardB: [109, 225], boardSingle: [109, 198],
    heroHole: [200, 392],
    // Made-hand labels go UNDER the hero's plate: there is no free band
    // between the seats on a phone.
    labels: { y: 514, dy: 16 },
    pot: [200, 92], potLow: [200, 144], potW: 150,
    potKeepOut: null,
    dealerOffset: 0, betOffset: 0,
    mini: { w: 18, h: 25, gap: 2, rankY: 12, suitY: 22 },
    menuDot: [42, -12],
  },
};
const TALL_MQ = window.matchMedia("(max-width: 560px) and (orientation: portrait)");
function tableLayout() { return TALL_MQ.matches ? TABLE_LAYOUTS.tall : TABLE_LAYOUTS.wide; }
const SVG_NS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs, text) {
  const el = document.createElementNS(SVG_NS, tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text !== undefined) el.textContent = text;
  return el;
}

// Applies the active layout to the static parts of the SVG (viewBox, felt).
function applyTableLayout() {
  const L = tableLayout();
  const svg = document.getElementById("table-svg");
  if (!svg || svg.dataset.layout === L.name) return;
  svg.dataset.layout = L.name;
  svg.setAttribute("viewBox", `0 0 ${L.w} ${L.h}`);
  svg.classList.toggle("tall", L.name === "tall");
  document.body.classList.toggle("table-tall", L.name === "tall");
  const set = (id, [cx, cy, rx, ry]) => {
    const e = document.getElementById(id);
    if (e) { e.setAttribute("cx", cx); e.setAttribute("cy", cy); e.setAttribute("rx", rx); e.setAttribute("ry", ry); }
  };
  set("felt-rim", L.felt.rim);
  set("felt-edge", L.felt.edge);
  set("felt-line", L.felt.line);
  set("felt-glow", L.felt.glow);
  const pot = document.getElementById("pot-badge");
  if (pot) {
    pot.setAttribute("transform", `translate(${L.pot[0]} ${L.pot[1]})`);
    const r = pot.querySelector("rect");
    if (r) { r.setAttribute("x", -L.potW / 2); r.setAttribute("width", L.potW); }
  }
}

function seatPositions(numSeats, heroSeat) {
  const L = tableLayout();
  const positions = new Array(numSeats);
  for (let i = 0; i < numSeats; i++) {
    const rel = (i - heroSeat + numSeats) % numSeats;
    // Physical CW from hero: increasing seat index moves visually CW on
    // screen, matching engine's (actor + 1) % n advancement and real-
    // poker action order (SB is one CW step from BTN, etc.). In SVG
    // (Y-down), visual CW corresponds to INCREASING theta from π/2.
    const theta = Math.PI / 2 + (rel * 2 * Math.PI / numSeats);
    const x = L.center.x + L.seatRx * Math.cos(theta);
    const y = L.center.y + L.seatRy * Math.sin(theta);
    positions[i] = { x, y, theta };
  }
  return positions;
}

// Where a seat's bet chip sits, in seat-local coordinates, and which side
// its amount goes. Wide: toward the table centre (GTO-Wizard style), sliding
// past the pot badge. Tall: straight inward from the plate's side (the
// top seat's goes under its plate, the hero's beside it).
function betAnchor(L, p, label) {
  if (L.name === "tall") {
    const W = L.plate.w, H = L.plate.h;
    const dxC = L.center.x - p.x;
    const inward = dxC >= 0 ? 1 : -1;               // +1: the centre is to the right
    if (p.y > L.center.y + L.seatRy * 0.8) {          // bottom (hero): beside the plate
      return { x: W / 2 + 14, y: 0, labelLeft: false };
    }
    if (Math.abs(dxC) < 10) {                          // top centre: under the plate
      return { x: -24, y: H / 2 + 14, labelLeft: false };
    }
    if (p.y < L.center.y - L.seatRy * 0.55 || Math.abs(p.y - L.center.y) < L.seatRy * 0.3) {
      // the top pair (5-handed) and mid-height seats (4-handed): under the plate
      return { x: inward * 10, y: H / 2 + 14, labelLeft: inward < 0 };
    }
    return { x: inward * (W / 2 + 14), y: 0, labelLeft: inward < 0 };   // sides: inward
  }
  const dx = L.center.x - p.x;
  const dy = L.center.y - p.y;
  const len = Math.hypot(dx, dy) || 1;
  let ax = p.x + (dx / len) * L.betOffset;
  const ay = p.y + (dy / len) * L.betOffset;
  // Label goes on the side of the chip facing the table center so it
  // never runs back over the seat plate / dealer button.
  let labelLeft = dx < -10;
  const textW = label.length * 6.6;
  // Keep-out around the pot badge: the top-center seat's ray lands on it —
  // slide the block sideways past the badge edge, label facing away from it.
  const POT = L.potKeepOut;
  const bx1 = labelLeft ? ax - 12 - textW : ax - 10;
  const bx2 = labelLeft ? ax + 10 : ax + 12 + textW;
  if (POT && ay > POT.y1 && ay < POT.y2 && bx2 > POT.x1 && bx1 < POT.x2) {
    if (p.x >= L.center.x) {
      labelLeft = false;
      ax = POT.x2 + 18;
    } else {
      labelLeft = true;
      ax = POT.x1 - 18;
    }
  }
  return { x: ax - p.x, y: ay - p.y, labelLeft };
}

// Committed-bet marker: a poker chip with the amount labeled beside it.
// `p` is the seat's absolute table position; the returned group uses
// seat-local coordinates (the caller's node is translated to `p`).
function makeBetChip(chips, s, p) {
  const L = tableLayout();
  const label = formatUnit(chips, s);
  const a = betAnchor(L, p, label);
  const g = svgEl("g", { class: "bet-chip", transform: `translate(${a.x} ${a.y})` });
  g.appendChild(svgEl("circle", { cy: 2.6, r: 8, class: "bet-chip-under" }));
  g.appendChild(svgEl("circle", { r: 8, class: "bet-chip-base" }));
  g.appendChild(svgEl("circle", { r: 8, class: "bet-chip-stripes" }));
  g.appendChild(svgEl("circle", { r: 4.2, class: "bet-chip-inner" }));
  g.appendChild(svgEl("text", {
    x: a.labelLeft ? -13 : 13, y: 4,
    "text-anchor": a.labelLeft ? "end" : "start",
    class: "bet-chip-amount",
  }, label));
  return g;
}

function renderSeats(s) {
  const L = tableLayout();
  const g = document.getElementById("seats");
  g.innerHTML = "";
  const positions = seatPositions(s.num_seats, s.hero_seat);
  const study = !s.trainer;
  for (const seat of s.seats) {
    if (seat.participant === false) continue;
    const p = positions[seat.seat];
    const node = svgEl("g", { transform: `translate(${p.x} ${p.y})` });
    node.classList.add("seat-node");
    node.dataset.seat = String(seat.seat);
    if (seat.is_actor) node.classList.add("actor");
    if (seat.is_hero) node.classList.add("hero");
    if (seat.folded) node.classList.add("folded");
    if (seat.all_in) node.classList.add("all-in");
    if (s.trainer && s.trainer.anim_action && s.trainer.anim_action.seat === seat.seat) {
      node.classList.add("acted");
    }
    const W = L.plate.w, H = L.plate.h;
    const bg = svgEl("rect", { x: -W / 2, y: -H / 2, width: W, height: H, rx: L.plate.rx, class: "seat-bg" });
    node.appendChild(bg);

    const heroTag = seat.is_hero ? (s.trainer ? " · you" : " · hero") : "";
    node.appendChild(svgEl("text", { y: L.plate.posY, class: "seat-position" }, seat.position + heroTag));

    const editable = !seat.all_in && study;
    const stackText = seat.all_in ? "all-in" : formatUnit(seat.stack_chips, s);
    const stack = svgEl("text", { y: L.plate.stackY, class: editable ? "seat-stack editable" : "seat-stack" }, stackText);
    if (editable) {
      stack.addEventListener("click", (e) => {
        e.stopPropagation();
        openStackEditor(seat, s);
      });
    }
    node.appendChild(stack);

    // Study: the whole plate opens the seat menu (ST-009 / MOB-007) — a
    // big tap target instead of a 7px red "x" nobody could identify.
    if (study) {
      node.classList.add("menu-seat");
      node.setAttribute("tabindex", "0");
      node.setAttribute("role", "button");
      node.setAttribute("aria-haspopup", "menu");
      node.setAttribute("aria-label",
        `${seat.position}${seat.is_hero ? " (Hero)" : ""}, ${seat.all_in ? "all-in" : `stack ${stackText}`}. Seat options`);
      const dot = svgEl("text", {
        x: L.menuDot[0], y: L.menuDot[1], "text-anchor": "middle", class: "seat-more", "aria-hidden": "true",
      }, "⋯");
      node.appendChild(dot);
      node.addEventListener("click", () => openSeatMenu(seat.seat));
      node.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openSeatMenu(seat.seat); }
      });
    }

    if (seat.committed_this_street_chips > 0) {
      node.appendChild(makeBetChip(seat.committed_this_street_chips, s, p));
    }

    // Trainer: revealed opponent hole cards. Wide: drawn OUTSIDE the table
    // (above the plate for top-half seats, below for bottom-half) so they
    // never collide with the dealer button / bet chips on the inside ray.
    // Tall: over the plate's lower half (there is no outside on a phone).
    if (seat.hole && !seat.is_hero) {
      const mini = svgEl("g", { class: "seat-hole" });
      const { w: cw, h: ch, gap } = L.mini;
      const total = seat.hole.length * cw + (seat.hole.length - 1) * gap;
      let rowCenterX = 0;
      let rowY;
      if (L.name === "tall") {
        rowY = -3;
      } else {
        // Keep the row on-canvas for far-left/right seats.
        rowCenterX = Math.max(total / 2 + 4, Math.min(L.w - total / 2 - 4, p.x)) - p.x;
        rowY = p.y < L.center.y ? -(H / 2 + 7 + ch) : H / 2 + 7;
      }
      const x0 = rowCenterX - total / 2;
      for (let i = 0; i < seat.hole.length; i++) {
        const card = cardToString(seat.hole[i]);
        const x = x0 + i * (cw + gap);
        const r = svgEl("rect", { x, y: rowY, width: cw, height: ch, rx: 3.5, class: "seat-hole-card" });
        r.style.fill = card.color;
        mini.appendChild(r);
        mini.appendChild(svgEl("text", { x: x + cw / 2, y: rowY + L.mini.rankY, "text-anchor": "middle", class: "seat-hole-rank" }, card.rank));
        mini.appendChild(svgEl("text", { x: x + cw / 2, y: rowY + L.mini.suitY, "text-anchor": "middle", class: "seat-hole-suit" }, card.glyph));
      }
      const t = svgEl("title");
      t.textContent = `${seat.position}: ${seat.hole.map(cardName).join(", ")}`;
      mini.appendChild(t);
      node.appendChild(mini);
    }

    g.appendChild(node);
  }
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

function openSeatMenu(seatIdx) {
  const already = document.getElementById("seat-pop");
  closeSeatMenu();
  const s = UI.lastState;
  if (!s || s.trainer || (already && already.dataset.seat === String(seatIdx))) return;
  const seat = s.seats[seatIdx];
  const node = document.querySelector(`#seats .seat-node[data-seat="${seatIdx}"]`);
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

function openStackEditor(seat, s) {
  const existing = document.getElementById("stack-edit-input");
  if (existing) existing.remove();
  closeSeatMenu();

  const svg = document.getElementById("table-svg");
  const wrap = document.getElementById("table-wrap");
  const L = tableLayout();
  const positions = seatPositions(s.num_seats, s.hero_seat);
  const p = positions[seat.seat];
  const pt = svg.createSVGPoint();
  pt.x = p.x; pt.y = p.y + L.plate.stackY - 4;
  const screen = pt.matrixTransform(svg.getScreenCTM());
  const wrapBox = wrap.getBoundingClientRect();

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
  input.style.left = `${screen.x - wrapBox.left - width / 2}px`;
  input.style.top = `${screen.y - wrapBox.top - 16}px`;
  wrap.style.position = "relative";
  wrap.appendChild(input);
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

function renderBoards(s) {
  const L = tableLayout();
  const boardA = document.getElementById("board-a");
  const boardB = document.getElementById("board-b");
  const focus = focusedSlotKey();
  boardA.innerHTML = "";
  boardB.innerHTML = "";
  // Single-board formats (flop_b === []) draw one vertically-centered row
  // and skip board B entirely; the double-board layout is unchanged.
  const single = isSingleBoard(s);
  const [ax, ay] = single ? L.boardSingle : L.boardA;
  boardA.setAttribute("transform", `translate(${ax} ${ay})`);
  boardB.setAttribute("transform", `translate(${L.boardB[0]} ${L.boardB[1]})`);

  renderSlotStrip(boardA, "flop_a", s.card_spec.flop_a, s, 0);
  if (!single) renderSlotStrip(boardB, "flop_b", s.card_spec.flop_b, s, 0);

  const step = L.board.w + L.board.gap;
  renderSingleSlot(boardA, "turn", 0, s.card_spec.turn[0], s, 3 * step);
  if (s.card_spec.turn.length > 1) {
    renderSingleSlot(boardB, "turn", 1, s.card_spec.turn[1], s, 3 * step);
  }
  renderSingleSlot(boardA, "river", 0, s.card_spec.river[0], s, 4 * step);
  if (s.card_spec.river.length > 1) {
    renderSingleSlot(boardB, "river", 1, s.card_spec.river[1], s, 4 * step);
  }
  restoreSlotFocus(focus);
}

function renderSlotStrip(parent, key, values, s, x0) {
  const geom = tableLayout().board;
  for (let i = 0; i < values.length; i++) {
    const x = x0 + i * (geom.w + geom.gap);
    renderSlotRect(parent, key, i, values[i], s, x, 0, geom.w, geom.h);
  }
}
function renderSingleSlot(parent, key, index, value, s, x) {
  const geom = tableLayout().board;
  renderSlotRect(parent, key, index, value, s, x, 0, geom.w, geom.h);
}

// Slots are keyboard buttons too (A11Y-017): Tab reaches them, Enter picks
// one, Delete/Backspace clears it, and each says what it holds.
function focusedSlotKey() {
  const a = document.activeElement;
  return a && a.classList && a.classList.contains("slot-rect") ? `${a.dataset.slot}:${a.dataset.index}` : null;
}
function restoreSlotFocus(key) {
  if (!key) return;
  const [slot, index] = key.split(":");
  const el = document.querySelector(`#table-svg .slot-rect[data-slot="${slot}"][data-index="${index}"]`);
  if (el) el.focus();
}

function renderSlotRect(parent, key, index, value, s, x, y, w, h) {
  const rect = svgEl("rect", { x, y, width: w, height: h, rx: 5 });
  let cls = "slot-rect" + (value === null ? " empty" : "");
  const isSelected = UI.selectedSlot && UI.selectedSlot.key === key && UI.selectedSlot.index === index;
  if (isSelected) cls += " selected";
  rect.setAttribute("class", cls);
  rect.dataset.slot = key;
  rect.dataset.index = String(index);
  const interactive = !s.trainer || (s.trainer.review && value !== null && value !== undefined);
  if (interactive) {
    rect.setAttribute("tabindex", "0");
    rect.setAttribute("role", "button");
  }
  const has = value !== null && value !== undefined;
  const words = slotWords(s, key, index);
  rect.setAttribute("aria-label", has
    ? `${words[0].toUpperCase()}${words.slice(1)}: ${cardName(value)}${isSelected ? ", selected" : ""}`
    : `${words[0].toUpperCase()}${words.slice(1)}: empty${isSelected ? ", selected" : ""}`);
  if (isSelected) rect.setAttribute("aria-pressed", "true");
  rect.addEventListener("click", () => onSlotClick(key, index));
  rect.addEventListener("dblclick", (e) => { e.preventDefault(); onSlotDoubleClick(key, index); });
  parent.appendChild(rect);

  if (has) {
    const card = cardToString(value);
    rect.style.fill = card.color;

    const smallRankSize = Math.max(8, Math.round(h * 0.227));
    const smallSuitSize = Math.max(8, Math.round(h * 0.25));
    const bigRankSize = Math.max(14, Math.round(h * 0.50));
    const pad = Math.max(3, Math.round(w * 0.12));
    const common = { fill: "#ffffff", "aria-hidden": "true" };
    parent.appendChild(svgEl("text", {
      ...common, x: x + pad, y: y + smallRankSize + 2, class: "slot-corner-rank", "font-size": smallRankSize,
    }, card.rank));
    parent.appendChild(svgEl("text", {
      ...common, x: x + pad, y: y + smallRankSize + smallSuitSize + 3, class: "slot-corner-suit", "font-size": smallSuitSize,
    }, card.glyph));
    parent.appendChild(svgEl("text", {
      ...common, x: x + w - pad, y: y + h - Math.max(3, Math.round(h * 0.08)),
      "text-anchor": "end", class: "slot-corner-bigrank", "font-size": bigRankSize,
    }, card.rank));
  }

  const modified = (s.modified_cards || []).some(
    (m) => m.slot_key === key && m.index === index
  );
  if (modified) {
    parent.appendChild(svgEl("circle", { cx: x + w - 4, cy: y + 4, r: 3, class: "slot-modified-dot" }));
  }
}

function renderHeroHole(s) {
  const L = tableLayout();
  const g = document.getElementById("hero-hole");
  const focus = focusedSlotKey();
  g.innerHTML = "";
  g.setAttribute("transform", `translate(${L.heroHole[0]} ${L.heroHole[1]})`);
  const geom = L.hole;
  const count = s.card_spec.hero_hole.length;  // per-format: 5 (PLO) / 2 (NLH)
  const totalWidth = count * geom.w + (count - 1) * geom.gap;
  const x0 = -totalWidth / 2;
  for (let i = 0; i < count; i++) {
    const x = x0 + i * (geom.w + geom.gap);
    renderSlotRect(g, "hero_hole", i, s.card_spec.hero_hole[i], s, x, 0, geom.w, geom.h);
  }
  restoreSlotFocus(focus);
}

// Two made-hand labels ("#1 a pair of 8s" / "#2 ...") in the
// gap between the hero plate and the hero cards (wide) or under the boards
// (tall), so a stealth set/straight is hard to miss while deciding.
function renderHeroHandLabels(s) {
  const L = tableLayout();
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
  rows.forEach(([tag, text], i) => {
    const t = svgEl("text", {
      x: L.center.x, y: L.labels.y + i * L.labels.dy, "text-anchor": "middle", class: "hero-hand-label",
    });
    if (tag) t.appendChild(svgEl("tspan", { class: "hhl-tag" }, tag + " "));
    t.appendChild(document.createTextNode(text));
    g.appendChild(t);
  });
}

function renderDealerButton(s) {
  const L = tableLayout();
  const node = document.getElementById("dealer-button");
  node.style.display = "";
  if (UI.draggingButton) return;
  const positions = seatPositions(s.num_seats, s.hero_seat);
  const p = positions[s.button_seat];
  let bx, by;
  if (L.name === "tall") {
    // On the plate's outer top corner, like a badge: nothing else fits
    // beside a seat on a phone. The hero's goes left of the plate (its bet
    // sits on the right, its cards just above).
    if (p.y > L.center.y + L.seatRy * 0.8) {
      bx = p.x - L.plate.w / 2 - 18;
      by = p.y;
    } else {
      const side = p.x < L.center.x - 10 ? -1 : 1;
      bx = p.x + side * (L.plate.w / 2 - 4);
      by = p.y - L.plate.h / 2 + 4;
    }
  } else {
    const dx = L.center.x - p.x;
    const dy = L.center.y - p.y;
    const len = Math.hypot(dx, dy) || 1;
    bx = p.x + (dx / len) * L.dealerOffset;
    by = p.y + (dy / len) * L.dealerOffset;
  }
  node.setAttribute("transform", `translate(${bx} ${by})`);
  node.dataset.seat = String(s.button_seat);
}

// CPY-019: ONE pot badge. It shows the pot gathered from earlier streets and,
// while bets are out on this street, how much more is in front of the
// players — the two separate "Total Pot" / "Pot" badges confused newcomers.
function renderPotLabel(s) {
  const badge = document.getElementById("pot-badge");
  const label = document.getElementById("pot-label");
  badge.removeAttribute("hidden");
  const L = tableLayout();
  if (L.potLow) {
    // Tall table: a pair of seats across the top (5-handed) sits where the
    // pot normally goes — drop the pot to just above the boards.
    const topPair = seatPositions(s.num_seats, s.hero_seat).some(
      (p) => Math.abs(p.x - L.center.x) >= 10 && p.y < L.center.y - L.seatRy * 0.55);
    const [px, py] = topPair ? L.potLow : L.pot;
    badge.setAttribute("transform", `translate(${px} ${py})`);
  }
  const total = Number(s.pot_chips) || 0;
  const settled = s.settled_pot_chips ?? total;
  const live = Math.max(0, total - settled);
  label.textContent = live > 0
    ? `Pot ${formatUnit(settled, s)} + ${formatUnit(live, s)}`
    : `Pot ${formatUnit(total, s)}`;
  const t = badge.querySelector("title") || badge.appendChild(svgEl("title"));
  t.textContent = live > 0
    ? `${formatUnit(settled, s)} in the pot, plus ${formatUnit(live, s)} bet on this street (${formatUnit(total, s)} in all)`
    : `${formatUnit(total, s)} in the pot`;
  renderTableSummary(s);
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

// --- Table keyboard (A11Y-017) ---------------------------------------------------
// Card slots are focusable buttons: Enter/Space selects one, Delete or
// Backspace empties it (Study).
function setupTableKeyboard() {
  const svg = document.getElementById("table-svg");
  svg.addEventListener("keydown", (e) => {
    const slot = e.target.closest && e.target.closest(".slot-rect");
    if (!slot) return;
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
}

// --- Dealer-button drag -----------------------------------------------------

const UI_INSERT = { pairA: null, pairB: null };

function svgPoint(svg, clientX, clientY) {
  const pt = svg.createSVGPoint();
  pt.x = clientX; pt.y = clientY;
  return pt.matrixTransform(svg.getScreenCTM().inverse());
}

function ellipseAngle(px, py) {
  const L = tableLayout();
  return Math.atan2((py - L.center.y) / L.seatRy, (px - L.center.x) / L.seatRx);
}

function ellipseRadius(px, py) {
  const L = tableLayout();
  return Math.hypot((px - L.center.x) / L.seatRx, (py - L.center.y) / L.seatRy);
}

function setupInsertHover() {
  const svg = document.getElementById("table-svg");
  const icon = document.getElementById("insert-icon");
  if (!icon) return;
  const TWO_PI = 2 * Math.PI;
  const THETA0 = Math.PI / 2;

  svg.addEventListener("pointermove", (e) => {
    const s = UI.lastState;
    // Mouse-only affordance ("+" between two seats); touch screens use the
    // Players control in the work bar (MOB-013).
    if (!s || s.num_seats >= 6 || s.trainer || e.pointerType === "touch") {
      icon.setAttribute("hidden", "");
      return;
    }
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
    const L = tableLayout();
    const mx = L.center.x + L.seatRx * Math.cos(midTheta);
    const my = L.center.y + L.seatRy * Math.sin(midTheta);
    icon.setAttribute("transform", `translate(${mx} ${my})`);
    icon.removeAttribute("hidden");
    UI_INSERT.pairA = pairA;
    UI_INSERT.pairB = pairB;
  });

  svg.addEventListener("mouseleave", () => { icon.setAttribute("hidden", ""); });

  icon.addEventListener("click", async (e) => {
    e.stopPropagation();
    const s = UI.lastState;
    if (!s || s.num_seats >= 6) return;
    if (UI_INSERT.pairA == null) return;
    const pairA = UI_INSERT.pairA;
    icon.setAttribute("hidden", "");
    if (!(await confirmHandReset("Add a player"))) return;
    insertSeat(pairA, s);
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
