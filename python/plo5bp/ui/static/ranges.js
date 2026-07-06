// NLH range grid ("Ranges" tab) — GTO-Wizard-style 169-hand strategy view.
// LOCAL BUILD ONLY: the tab stays hidden and /ranges routes 404 on the
// public build until the feature is validated and deliberately shipped.
//
// Self-contained on purpose: app.js is untouched. The tab toggles
// body.ranges-mode (CSS hides the study/trainer layout); clicking the
// Study/Trainer tabs clears it. State is one line of interleaved actions
// and street cards; every mutation re-queries POST /ranges/query, which
// is stateless and server-cached per node.

"use strict";

const RG = {
  line: [],        // [{t:"a",gate,chips_bb?} | {t:"cards",cards:[..]}]
  node: null,      // viewed prefix length; null = live end
  seats: 6,
  stackBb: 100,
  data: null,      // last response
  busy: false,
  pick: null,      // {need, street, chosen:[]} while the card modal is open
  enabled: false,  // format == nlh_single && local build
};

const RG_RANKS = "AKQJT98765432";
const RG_SUIT_GLYPH = { c: "♣", d: "♦", h: "♥", s: "♠" };
const RG_GATE_LABEL = { fold: "Fold", check_call: "Call", raise: "Raise" };

// --- data ------------------------------------------------------------------

let RG_SEQ = 0;

async function rgQuery(prevLine) {
  // Never drop a query: every call fires, the LATEST response wins
  // (superseded ones are discarded). Dropping-when-busy desyncs the
  // line from the rendered node when clicks arrive mid-flight.
  const seq = ++RG_SEQ;
  RG.busy = true;
  document.getElementById("rg-view").classList.add("busy");
  try {
    const body = {
      seats: RG.seats,
      stack_bb: RG.stackBb,
      line: RG.line,
    };
    if (RG.node !== null) body.node = RG.node;
    const res = await fetch("/ranges/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      let msg = `ranges query failed (${res.status})`;
      try { msg = (await res.json()).detail || msg; } catch (_) {}
      throw new Error(msg);
    }
    const data = await res.json();
    if (seq !== RG_SEQ) return; // superseded by a newer query
    RG.data = data;
    rgRender();
  } catch (e) {
    if (seq !== RG_SEQ) return;
    if (typeof showToast === "function") showToast(e.message);
    if (prevLine) { RG.line = prevLine; RG.node = null; }
  } finally {
    if (seq === RG_SEQ) {
      RG.busy = false;
      document.getElementById("rg-view").classList.remove("busy");
    }
  }
}

function rgMutate(fn) {
  // Line mutations are serialized: while a mutation's query is in flight
  // the action buttons are inert (busy guard here + pointer-events CSS),
  // so the line can never race its own rendering. Any action taken while
  // viewing a past node truncates the line there (GTO-Wizard-style
  // rebranching), then appends.
  if (RG.busy) return;
  const prev = JSON.parse(JSON.stringify(RG.line));
  if (RG.node !== null && RG.node < RG.line.length) {
    RG.line = RG.line.slice(0, RG.node);
  }
  fn();
  RG.node = null;
  rgQuery(prev);
}

// --- rendering -------------------------------------------------------------

function rgCellName(r, c) {
  if (r === c) return RG_RANKS[r] + RG_RANKS[r];
  const hi = Math.min(r, c);
  const lo = Math.max(r, c);
  return RG_RANKS[hi] + RG_RANKS[lo] + (c > r ? "s" : "o");
}

function rgBuildGrid() {
  const grid = document.getElementById("rg-grid");
  grid.innerHTML = "";
  for (let r = 0; r < 13; r++) {
    for (let c = 0; c < 13; c++) {
      const name = rgCellName(r, c);
      const cell = document.createElement("div");
      cell.className = "rg-cell";
      cell.id = `rg-cell-${name}`;
      cell.innerHTML = `<span class="rg-cell-name">${name}</span>`;
      cell.addEventListener("mouseenter", () => rgRenderHover(name));
      grid.appendChild(cell);
    }
  }
}

function rgCellGradient(cd) {
  // Bands left→right: all-in (dark red), raise (red), call (green),
  // fold (blue) — matching the reference layout.
  const ai = Math.max(cd.ai, 0);
  const r = Math.max(cd.r - cd.ai, 0);
  const c = Math.max(cd.c, 0);
  const p1 = ai * 100;
  const p2 = p1 + r * 100;
  const p3 = p2 + c * 100;
  return (
    `linear-gradient(to right,` +
    ` var(--rg-allin) 0% ${p1}%,` +
    ` var(--rg-raise) ${p1}% ${p2}%,` +
    ` var(--rg-call) ${p2}% ${p3}%,` +
    ` var(--rg-fold) ${p3}% 100%)`
  );
}

function rgRenderGrid() {
  const cells = RG.data.cells || {};
  for (let r = 0; r < 13; r++) {
    for (let c = 0; c < 13; c++) {
      const name = rgCellName(r, c);
      const el = document.getElementById(`rg-cell-${name}`);
      const cd = cells[name];
      el.classList.toggle("empty", !cd);
      el.classList.toggle("dead", !!(cd && cd.dead));
      if (!cd) {
        el.style.background = "";
        el.title = `${name} — blocked by the board`;
        continue;
      }
      el.style.background = rgCellGradient(cd);
      el.title =
        `${name} · raise ${(cd.r * 100).toFixed(1)}%` +
        (cd.ai > 0.0005 ? ` (all-in ${(cd.ai * 100).toFixed(1)}%)` : "") +
        ` · call ${(cd.c * 100).toFixed(1)}% · fold ${(cd.f * 100).toFixed(1)}%` +
        ` · value ${cd.v > 0 ? "+" : ""}${cd.v}bb`;
    }
  }
}

function rgComboHtml(name) {
  // "AsKs" -> glyph markup with 4-color suits.
  const parts = [];
  for (let i = 0; i < name.length; i += 2) {
    const suit = name[i + 1];
    parts.push(
      `<span class="rg-card suit-${suit}">${name[i]}${RG_SUIT_GLYPH[suit]}</span>`
    );
  }
  return parts.join("");
}

function rgRenderHover(cellName) {
  const box = document.getElementById("rg-hover");
  const d = RG.data;
  if (!d || !d.combos) { box.innerHTML = ""; return; }
  const rows = Object.entries(d.combos)
    .filter(([, v]) => v.cell === cellName)
    .sort((a, b) => a[0].localeCompare(b[0]));
  const cd = (d.cells || {})[cellName];
  let html = `<div class="rg-hover-head">${cellName}`;
  if (cd) {
    html += `<span class="muted"> · ${rows.length} combos · ` +
      `${cd.reach.toFixed(1)} in range · value ${cd.v > 0 ? "+" : ""}${cd.v}bb</span>`;
  }
  html += `</div>`;
  for (const [name, v] of rows) {
    const bar = rgCellGradient(v);
    html +=
      `<div class="rg-hover-row">` +
      `<span class="rg-hover-combo">${rgComboHtml(name)}</span>` +
      `<span class="rg-hover-bar" style="background:${bar}"></span>` +
      `<span class="rg-hover-nums">` +
      `<b class="rg-r">${(v.r * 100).toFixed(1)}</b>/` +
      `<b class="rg-c">${(v.c * 100).toFixed(1)}</b>/` +
      `<b class="rg-f">${(v.f * 100).toFixed(1)}</b>` +
      ` · ${(v.reach * 100).toFixed(0)}%</span>` +
      `</div>`;
  }
  box.innerHTML = html;
}

function rgChip(label, cls, onClick) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = `rg-chip ${cls || ""}`;
  b.innerHTML = label;
  if (onClick) b.addEventListener("click", onClick);
  return b;
}

function rgCardsLabel(cards) {
  return cards
    .map((idx) => {
      const r = "23456789TJQKA"[idx >> 2];
      const s = "cdhs"[idx & 3];
      return `<span class="rg-card suit-${s}">${r}${RG_SUIT_GLYPH[s]}</span>`;
    })
    .join("");
}

function rgRenderStrips() {
  const d = RG.data;
  const wrap = document.getElementById("rg-strips");
  wrap.innerHTML = "";
  const viewed = RG.node === null ? d.num_entries : RG.node;
  wrap.appendChild(
    rgChip(`${RG.seats}-max · ${RG.stackBb}bb`, "cfg", () => {
      RG.node = 0;
      rgQuery();
    })
  );
  for (const e of d.sequence || []) {
    if (e.t === "a") {
      const size = e.chips_bb !== null && e.chips_bb !== undefined
        ? ` ${e.chips_bb}bb` : "";
      const chip = rgChip(
        `<b>${e.position}</b> ${RG_GATE_LABEL[e.gate] || e.gate}${size}`,
        `act gate-${e.gate}`,
        () => { RG.node = e.i; rgQuery(); }
      );
      if (viewed === e.i) chip.classList.add("viewing");
      wrap.appendChild(chip);
    } else {
      const chip = rgChip(
        `<b>${e.street}</b> ${rgCardsLabel(e.cards)}`,
        "cards",
        () => { RG.node = e.i + 1; rgQuery(); }
      );
      if (viewed === e.i + 1) chip.classList.add("viewing");
      wrap.appendChild(chip);
    }
  }
  const live = rgChip("live", "live", () => { RG.node = null; rgQuery(); });
  if (RG.node === null || RG.node === d.num_entries) live.classList.add("viewing");
  wrap.appendChild(live);
}

function rgRenderActions() {
  const d = RG.data;
  const box = document.getElementById("rg-actions");
  box.innerHTML = "";

  if (d.terminal) {
    box.innerHTML =
      `<div class="rg-banner">Hand over — everyone folded or showdown.` +
      ` Click a strip above to rewind, or Reset.</div>`;
    return;
  }
  if (d.awaiting) {
    const btn = rgChip(
      `Pick the ${d.awaiting} (${d.need} card${d.need > 1 ? "s" : ""})`,
      "pick-cards",
      () => rgOpenCardModal(d.awaiting, d.need, d.board || [])
    );
    box.appendChild(btn);
    return;
  }

  const st = d.state;
  const head = document.createElement("div");
  head.className = "rg-node-head";
  const toCall = st.to_call_bb > 0 ? ` · to call ${st.to_call_bb}bb` : "";
  head.innerHTML =
    `<b>${st.position}</b> to act · ${st.street} · pot ${st.pot_bb}bb${toCall}`;
  box.appendChild(head);

  const row = document.createElement("div");
  row.className = "rg-action-row";
  if (st.legal.fold) {
    row.appendChild(rgChip("Fold", "gate-fold", () =>
      rgMutate(() => RG.line.push({ t: "a", gate: "fold" }))));
  }
  if (st.legal.check_call) {
    const label = st.to_call_bb > 0 ? `Call ${st.to_call_bb}bb` : "Check";
    row.appendChild(rgChip(label, "gate-check_call", () =>
      rgMutate(() => RG.line.push({ t: "a", gate: "check_call" }))));
  }
  box.appendChild(row);

  if (st.legal.raise) {
    const sizes = document.createElement("div");
    sizes.className = "rg-action-row rg-sizes-row";
    for (const a of st.anchors || []) {
      sizes.appendChild(
        rgChip(`${a.label} <i>${a.chips_bb}bb</i>`, "gate-raise", () =>
          rgMutate(() =>
            RG.line.push({ t: "a", gate: "raise", chips_bb: a.chips_bb })
          ))
      );
    }
    box.appendChild(sizes);

    const custom = document.createElement("div");
    custom.className = "rg-action-row rg-custom";
    custom.innerHTML =
      `<input id="rg-custom-bb" type="number" step="0.1"` +
      ` min="${st.min_raise_bb}" max="${st.max_raise_bb}"` +
      ` placeholder="${st.min_raise_bb}–${st.max_raise_bb}bb" />` +
      `<button id="rg-custom-go" type="button" class="rg-chip gate-raise">Raise to</button>`;
    box.appendChild(custom);
    custom.querySelector("#rg-custom-go").addEventListener("click", () => {
      const v = parseFloat(custom.querySelector("#rg-custom-bb").value);
      if (!Number.isFinite(v)) return;
      rgMutate(() => RG.line.push({ t: "a", gate: "raise", chips_bb: v }));
    });
  }
}

function rgRenderSummary() {
  const d = RG.data;
  const bar = document.getElementById("rg-summary");
  const sizesBox = document.getElementById("rg-sizes");
  if (!d.summary) { bar.innerHTML = ""; sizesBox.innerHTML = ""; return; }
  const parts = [
    ["allin", "ALL-IN", "var(--rg-allin)"],
    ["raise", "Raise", "var(--rg-raise)"],
    ["check_call", d.state && d.state.to_call_bb > 0 ? "Call" : "Check", "var(--rg-call)"],
    ["fold", "Fold", "var(--rg-fold)"],
  ];
  let seg = `<div class="rg-sumbar">`;
  let leg = `<div class="rg-sumleg">`;
  for (const [key, label, color] of parts) {
    const s = d.summary[key];
    if (!s) continue;
    if (s.freq > 0.0005) {
      seg += `<span style="width:${s.freq * 100}%;background:${color}"></span>`;
    }
    leg +=
      `<span class="rg-leg-item"><i style="background:${color}"></i>` +
      `${label} <b>${(s.freq * 100).toFixed(1)}%</b>` +
      ` <span class="muted">${s.combos}</span></span>`;
  }
  seg += `</div>`;
  leg += `</div>`;
  bar.innerHTML = seg + leg;

  let sh = "";
  if (d.sizes && d.sizes.length) {
    sh = `<div class="rg-sizes-hist">`;
    for (const s of d.sizes) {
      if (s.frac < 0.005) continue;
      sh +=
        `<span class="rg-size-item">${s.label}` +
        ` <i>${s.chips_bb}bb</i> <b>${(s.frac * 100).toFixed(0)}%</b></span>`;
    }
    sh += `</div>`;
  }
  sizesBox.innerHTML = sh;
}

function rgRenderModel() {
  const el = document.getElementById("rg-model");
  const m = (RG.data && RG.data.model) || {};
  el.innerHTML =
    `model: <b>${m.checkpoint || "?"}</b>` +
    (m.loaded ? "" : ` <span class="rg-untrained">untrained</span>`);
}

function rgRender() {
  const d = RG.data;
  if (!d) return;
  rgRenderStrips();
  rgRenderActions();
  rgRenderModel();
  const gridWrap = document.getElementById("rg-grid-wrap");
  if (d.cells) {
    gridWrap.classList.remove("inactive");
    rgRenderGrid();
    rgRenderSummary();
  } else {
    gridWrap.classList.add("inactive");
    document.getElementById("rg-summary").innerHTML = "";
    document.getElementById("rg-sizes").innerHTML = "";
    document.getElementById("rg-hover").innerHTML = "";
  }
  // Auto-open the picker when the live end awaits street cards.
  if (d.awaiting && RG.node === null && !RG.pick) {
    rgOpenCardModal(d.awaiting, d.need, d.board || []);
  }
}

// --- card picker -------------------------------------------------------------

function rgOpenCardModal(street, need, board) {
  RG.pick = { street, need, chosen: [] };
  const modal = document.getElementById("rg-card-modal");
  document.getElementById("rg-card-title").textContent =
    `Pick the ${street} — ${need} card${need > 1 ? "s" : ""}`;
  const grid = document.getElementById("rg-card-grid");
  grid.innerHTML = "";
  const used = new Set(board);
  for (const e of RG.line) if (e.t === "cards") e.cards.forEach((c) => used.add(c));
  for (let s = 3; s >= 0; s--) {
    for (let r = 12; r >= 0; r--) {
      const idx = r * 4 + s;
      const b = document.createElement("button");
      b.type = "button";
      const suit = "cdhs"[s];
      b.className = `rg-pick-card suit-${suit}`;
      b.innerHTML = `${"23456789TJQKA"[r]}${RG_SUIT_GLYPH[suit]}`;
      if (used.has(idx)) b.disabled = true;
      b.addEventListener("click", () => {
        const i = RG.pick.chosen.indexOf(idx);
        if (i >= 0) RG.pick.chosen.splice(i, 1);
        else if (RG.pick.chosen.length < RG.pick.need) RG.pick.chosen.push(idx);
        b.classList.toggle("chosen", RG.pick.chosen.includes(idx));
        document.getElementById("rg-card-ok").disabled =
          RG.pick.chosen.length !== RG.pick.need;
      });
      grid.appendChild(b);
    }
  }
  document.getElementById("rg-card-ok").disabled = true;
  modal.hidden = false;
}

function rgCloseCardModal() {
  RG.pick = null;
  document.getElementById("rg-card-modal").hidden = true;
}

// --- tab / mode wiring --------------------------------------------------------

function rgEnterMode() {
  document.body.classList.add("ranges-mode");
  document.getElementById("tab-ranges").classList.add("active");
  document.getElementById("tab-study").classList.remove("active");
  document.getElementById("tab-trainer").classList.remove("active");
  if (!RG.data) rgQuery();
}

function rgLeaveMode() {
  document.body.classList.remove("ranges-mode");
  document.getElementById("tab-ranges").classList.remove("active");
}

function rgSyncTabVisibility() {
  const sel = document.getElementById("format-select");
  const tab = document.getElementById("tab-ranges");
  const nlh = !!sel && sel.value === "nlh_single";
  RG.enabled = nlh && !window.PLO5BP_PUBLIC;
  tab.hidden = !RG.enabled;
  if (!RG.enabled && document.body.classList.contains("ranges-mode")) {
    document.getElementById("tab-trainer").click();
  }
}

function rgInit() {
  if (window.PLO5BP_PUBLIC) return; // local-only feature
  const tab = document.getElementById("tab-ranges");
  if (!tab) return;
  rgBuildGrid();
  tab.addEventListener("click", rgEnterMode);
  document.getElementById("tab-study").addEventListener("click", rgLeaveMode);
  document.getElementById("tab-trainer").addEventListener("click", rgLeaveMode);

  const sel = document.getElementById("format-select");
  if (sel) {
    sel.addEventListener("change", () => setTimeout(rgSyncTabVisibility, 50));
  }
  // The format dropdown fills asynchronously at app boot.
  const poll = setInterval(() => {
    if (sel && sel.options.length) {
      clearInterval(poll);
      rgSyncTabVisibility();
    }
  }, 300);
  setTimeout(() => clearInterval(poll), 15000);

  document.getElementById("rg-seats").addEventListener("change", (e) => {
    RG.seats = parseInt(e.target.value, 10);
    RG.line = []; RG.node = null; RG.data = null;
    rgQuery();
  });
  document.getElementById("rg-stack").addEventListener("change", (e) => {
    const v = parseFloat(e.target.value);
    if (Number.isFinite(v) && v > 1) {
      RG.stackBb = v;
      RG.line = []; RG.node = null; RG.data = null;
      rgQuery();
    }
  });
  document.getElementById("rg-reset").addEventListener("click", () => {
    RG.line = []; RG.node = null;
    rgQuery();
  });
  document.getElementById("rg-card-ok").addEventListener("click", () => {
    const cards = RG.pick ? RG.pick.chosen.slice() : [];
    rgCloseCardModal();
    if (cards.length) {
      rgMutate(() => RG.line.push({ t: "cards", cards }));
    }
  });
  document.getElementById("rg-card-cancel").addEventListener("click", rgCloseCardModal);
  document.getElementById("rg-card-modal").addEventListener("pointerdown", (e) => {
    if (e.target === e.currentTarget) rgCloseCardModal();
  });
}

rgInit();
