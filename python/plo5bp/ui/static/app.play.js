// Study / Trainer client, part 3 of 7 — acting and advice: the actor
// banner, action buttons, raise sizing and bet-size presets, the
// recommendation panel (action mix, bet-size curve, EV, candidate model)
// and the action history.
//
// Plain scripts, no build step: index.html loads app.core.js, app.table.js,
// app.play.js, app.study.js, app.trainer.js, app.topbar.js and app.js in
// that order, and they share one global scope — any file may call a
// function from any other. Code that runs while a file LOADS may use only
// the files before it; everything else starts from init() in app.js.
"use strict";

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
  banner.classList.remove("error");
  // A deal or a graded move in flight keeps its "Dealing…" line (ST-028).
  if (UI.workingText) {
    banner.hidden = false;
    banner.classList.add("working");
    banner.textContent = UI.workingText;
    return;
  }
  banner.classList.remove("working", "villain");
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
      const did = nc.actual_gate
        ? gateActionLabel(nc.actual_gate, nc.actual_chips, nc.to_call_chips, s,
                          reviewCurCommit(s, rv, nc, walk), nc.street)
        : nc.actual_label;
      // A hero moot auto-check (ungraded: category null) was never a choice.
      const what = !nc.is_hero ? `${nc.position}: ${did}`
        : reviewNodeUngraded(nc) ? `You (${nc.position}): automatic check`
          : `You (${nc.position}): ${did}`;
      banner.textContent = `Review · action ${rv.node + 1} of ${rv.num_nodes} · ${nc.street} · ${what}`;
    } else if (rv.current) {
      const c = rv.current;
      const did = c.user_gate
        ? gateActionLabel(c.user_gate, c.user_chips, c.to_call_chips, s,
                          reviewCurCommit(s, rv, c, walk), c.street)
        : c.user_label;
      banner.textContent =
        `Review · your decision ${rv.decision + 1} of ${rv.num_decisions} · ${c.street} · ${did}`;
    } else {
      banner.textContent = "Reviewing the hand";
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
  // ST-017: say whose action this is, by position — never an internal seat
  // number that appears nowhere on the table.
  if (s.trainer) {
    banner.textContent = isHero ? `Your turn (${seat.position})` : `${seat.position} is thinking…`;
  } else if (isHero) {
    banner.textContent = `Hero to act (${seat.position}) — the network's answer is in Recommendation`;
  } else {
    banner.classList.add("villain");
    banner.textContent = `${seat.position} to act — enter ${seat.position}'s action below`;
  }
}

// Fold / Check-Call are built ONCE and updated in place (A11Y-016): rebuilding
// them on every render threw keyboard focus back to the top of the page
// after each action and could eat a click that landed mid-rebuild.
function ensureGateButtons() {
  const gate = document.getElementById("gate-buttons");
  if (gate.querySelector('button[data-gate="fold"]')) return gate;
  gate.innerHTML = "";
  const note = document.createElement("p");
  note.className = "muted gate-note";
  note.hidden = true;
  gate.appendChild(note);
  for (const [slug, cls] of [["fold", "fold"], ["check_call", "call"]]) {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.gate = slug;
    b.className = cls;
    b.addEventListener("click", () => postAction({ gate: slug }));
    gate.appendChild(b);
  }
  return gate;
}

// End-of-hand line. The Trainer builds it from the hand's result in the
// active unit, in the second person (ST-013 / ST-023 / CPY-013); it also
// can't claim "All opponents folded" when it was you who folded.
function terminalText(s) {
  const t = s.trainer;
  const r = t && Array.isArray(t.rewards_bb) ? t.rewards_bb : null;
  if (!t || !r || typeof r[s.hero_seat] !== "number") {
    return s.terminal_message ?? "Hand complete.";
  }
  const net = r[s.hero_seat];
  const amt = fmtBBValue(Math.abs(net), s);
  const res = net > 0 ? `you won ${amt}` : net < 0 ? `you lost ${amt}` : "you broke even";
  const hero = s.seats[s.hero_seat];
  if (hero && hero.folded) return `You folded — ${net < 0 ? `you lost ${amt}` : res}.`;
  if (s.terminal === "fold_out") return `Everyone else folded — ${res}.`;
  return `Showdown — ${res}.`;
}

function renderActions(s) {
  const gate = ensureGateButtons();
  const note = gate.querySelector(".gate-note");
  const btns = [...gate.querySelectorAll("button[data-gate]")];
  const raiseSection = document.getElementById("raise-section");
  const terminalPane = document.getElementById("terminal-pane");
  const title = document.getElementById("action-title");
  const panel = document.getElementById("action-panel");
  panel.classList.remove("villain-turn");
  const onlyNote = (text) => {
    note.textContent = text;
    note.hidden = !text;
    for (const b of btns) b.hidden = true;
    raiseSection.hidden = true;
  };

  if (s.terminal) {
    onlyNote("");
    terminalPane.hidden = false;
    title.textContent = "Hand over";
    document.getElementById("terminal-message").textContent = terminalText(s);
    return;
  }
  terminalPane.hidden = true;

  // Trainer review reconstruction: the hand is over; this state is a
  // replayed decision node. Show the choice, don't allow acting.
  if (s.trainer && !s.trainer.hand_active) {
    title.textContent = "Review";
    onlyNote("The hand is over — step through it above, or deal the next hand.");
    return;
  }

  // Trainer animation frame: an opponent is acting.
  if (s.trainer && s.actor !== null && s.actor !== undefined && s.actor !== s.hero_seat) {
    title.textContent = "Your action";
    onlyNote("Opponents are acting…");
    return;
  }

  if (s.actor === null || s.actor === undefined) {
    title.textContent = "Actions";
    const st = s.awaiting_next_street;
    onlyNote(st ? `Betting is closed — enter the ${st} cards to continue.` : "No action to take right now.");
    return;
  }

  const isHero = s.actor === s.hero_seat;
  const actorSeat = s.seats[s.actor];
  // ST-017: say whose action the buttons enter. Study records every player's
  // action, so an opponent's turn gets its own title and tint.
  if (s.trainer) title.textContent = "Your action";
  else if (isHero) title.textContent = `Hero's action (${actorSeat.position})`;
  else {
    title.textContent = `Enter ${actorSeat.position}'s action`;
    panel.classList.add("villain-turn");
  }
  const holeCount = s.card_spec ? s.card_spec.hero_hole.length : 5;
  const BLOCK_TOOLTIPS = {
    hole:  `Place your ${holeCount} hole cards to act`,
    flop:  "Place the flop cards to act",
    turn:  "Place the turn cards to act",
    river: "Place the river cards to act",
  };
  const heroBlocked = isHero && s.hero_blocking_reason != null;
  const tooltip = heroBlocked ? BLOCK_TOOLTIPS[s.hero_blocking_reason] : "";
  note.hidden = !heroBlocked;
  note.textContent = heroBlocked ? `${tooltip}.` : "";

  const setBtn = (b, label, enabled, hint) => {
    b.hidden = false;
    b.textContent = label;
    // data-legal lets setActionsBusy(false) restore exactly this state; a
    // button painted while a POST is outstanding starts out busy-disabled
    // (review 2026-09-20 F2).
    b.dataset.legal = enabled ? "1" : "0";
    b.disabled = !enabled || UI.actionInFlight;
    b.title = tooltip || hint || "";
  };
  const trainerKeys = !!s.trainer;
  setBtn(btns[0], "Fold", s.legal.fold && !heroBlocked,
    trainerKeys ? "Fold (F)" : (s.legal.fold ? "" : "Nothing to fold to — checking is free"));
  setBtn(btns[1], s.to_call_chips > 0 ? `Call ${formatUnit(s.to_call_chips, s)}` : "Check",
    s.legal.check_call && !heroBlocked, trainerKeys ? "Check or call (C)" : "");

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
  unitLabel.textContent = unitSuffix();
  // ST-024: "Bet" when nothing is in front, "Raise" otherwise — the same verb
  // the history and the recommendation use.
  const verb = aggVerb(s.to_call_chips, s.street);
  input.setAttribute("aria-label", `${verb} size, in ${UI.unit === "bb" ? "big blinds" : "dollars"}`);

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
    // Display the raise-TO total (delta + ac); still post the DELTA.
    submit.textContent = isAllIn
      ? `All-in ${formatUnit(maxChips + ac, s)}`
      : `${verb} ${formatUnit(maxChips + ac, s)}`;
    submit.onclick = () => postAction({ gate: "raise", chips: maxChips });
    return;
  }
  inputRow.querySelectorAll("input, .raise-unit").forEach(el => el.style.display = "");
  submit.textContent = verb;

  // Bounds shown as raise-TO totals (delta + ac).
  const minDisp = chipsToCurrentUnit(minChips + ac, s);
  const maxDisp = chipsToCurrentUnit(maxChips + ac, s);
  boundsLabel.textContent = `${verb === "Bet" ? "Bet" : "Raise to"} `
    + `${formatUnit(minChips + ac, s)} – ${formatUnit(maxChips + ac, s)}`;
  input.min = String(Math.floor(minDisp * 100) / 100);
  input.max = String(Math.ceil(maxDisp * 100) / 100);
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
    input.value = unitInputValue(preset + ac, s);
  } else if (!userTyping) {
    // The input holds a TOTAL; clamp in delta space, redisplay as total.
    const curTotal = parseToChips(input.value, s);
    if (curTotal !== null) {
      const curDelta = curTotal - ac;
      if (curDelta < minChips || curDelta > maxChips) {
        const clampedDelta = Math.max(minChips, Math.min(maxChips, curDelta));
        input.value = unitInputValue(clampedDelta + ac, s);
      }
    }
  }

  submit.onclick = () => {
    // The user typed a raise-TO total; convert to the engine's raise-BY delta.
    const total = parseToChips(input.value, s);
    if (total === null) { showToast("Enter a bet size."); input.focus(); return; }
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
  // The 33% preset has always meant a THIRD of pot (mult 1/3, not 0.33) — keep exact.
  const presetMult = (n) => (n === 33 ? 1 / 3 : n / 100);
  // A chip's label must describe the chips it prefills: a preset that the
  // legal range clamps is labelled by where it lands — "Min", or "All-in"
  // ("Max" when the cap is a cover-short clamp and the actor keeps chips
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
    if (raw <= minTotal) addItem("Min", minTotal);
    else if (raw > maxTotal || (raw === maxTotal && maxIsAllIn)) {
      addItem(maxIsAllIn ? "All-in" : "Max", maxTotal, "raise-shortcut-allin");
    } else addItem(presetChipLabel(n), raw);
  }
  if (!potLimit) {
    // No-limit only: an all-in prefill chip (in pot-limit the pot chip IS the
    // cap). Prefills the max raise-TO total; the user still clicks Raise.
    addItem(maxIsAllIn ? "All-in" : "Max", maxTotal, "raise-shortcut-allin");
  }
  items.forEach((it, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "raise-shortcut" + (it.cls ? ` ${it.cls}` : "");
    b.textContent = it.label;
    const amount = formatUnit(it.chips, s);
    b.title = `${verb === "Bet" ? "Bet" : "Raise to"} ${amount}${s.trainer && i < 9 ? ` (${i + 1})` : ""}`;
    b.setAttribute("aria-label", `${it.label}: ${verb === "Bet" ? "bet" : "raise to"} ${amount}`);
    b.addEventListener("click", () => {
      input.value = unitInputValue(it.chips, s);
      UI.raiseUserSet = true;
    });
    shortcuts.appendChild(b);
  });
  const plus = document.createElement("button");
  plus.id = "preset-edit-btn";
  plus.type = "button";
  plus.className = "raise-shortcut raise-shortcut-edit";
  plus.textContent = "Edit";
  plus.title = "Edit the bet-size buttons";
  plus.setAttribute("aria-haspopup", "dialog");
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

// Chip label: "25%", "33.3%", … of the pot (ST-005: it used to read "b25") —
// except 100% of pot, which reads "Pot" (in pot-limit it IS the cap).
function presetChipLabel(n) {
  return n === 100 ? "Pot" : `${n}%`;
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
    pill.appendChild(document.createTextNode(presetChipLabel(n)));
    const x = document.createElement("button");
    x.type = "button";
    x.className = "preset-pill-x";
    x.setAttribute("aria-label", `Remove the ${presetChipLabel(n)} button`);
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
  if (!isFinite(v) || v <= 0) { presetPopHint("Enter a size above 0%."); return; }
  const n = Math.round(v * 10) / 10;  // up to one decimal place
  if (potLimit && n > 100) { presetPopHint("Pot-limit bets stop at the pot (100%)."); return; }
  if (!potLimit && n > 1000) { presetPopHint("Presets go up to 1,000% of the pot."); return; }
  const list = betPresets(potLimit);
  if (list.includes(n)) { presetPopHint(`${presetChipLabel(n)} is already a preset.`); return; }
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
  // The third row names the move the way the buttons do: a bet when there
  // is nothing to call, a raise otherwise (ST-024).
  const distNames = ["Fold", callName, callName === "Call" ? "Raise" : "Bet"];
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

  // Hover / tap columns, one per distinct rung (FEAT-022).
  const psum = anchors.reduce((acc, a) => acc + Math.max(0, a.prob), 0) || 1;
  const rungs = [];
  for (const a of anchors) {
    const x = idxMode ? px(single ? 0.5 : anchors.indexOf(a) / n1) : px(xf(a.chips));
    const last = rungs[rungs.length - 1];
    if (last && Math.abs(x - last.x) < 0.75) { last.p += a.prob; continue; }
    rungs.push({ x, p: a.prob, name: anchorName(a, n1, idxMode) });
  }
  const hits = rungs.map((r, i) => {
    const left = i === 0 ? padL : (rungs[i - 1].x + r.x) / 2;
    const right = i === rungs.length - 1 ? W - padR : (r.x + rungs[i + 1].x) / 2;
    const text = `${r.name} — ${pctOf(r.p / psum)} of its bets`;
    return `<rect class="bc-hit" x="${left.toFixed(1)}" y="${padT}" width="${Math.max(2, right - left).toFixed(1)}"`
      + ` height="${plotH + 16}" fill="transparent" data-bc="${escapeHTML(text)}"><title>${escapeHTML(text)}</title></rect>`;
  }).join("");
  const topLine = "Top sizes: " + rungs.slice().sort((a, b) => b.p - a.p).slice(0, 3)
    .filter((r) => r.p > 0).map((r) => `${r.name} ${pctOf(r.p / psum)}`).join(" · ");

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
    + hits
    + `</svg>`
    + `<p class="bc-readout" data-top="${escapeHTML(topLine)}">${escapeHTML(topLine)}</p>`;
}

// FEAT-022: the numbers behind the curve. Each size the network considers
// gets an invisible hover / tap column (with a native tooltip), and the line
// under the curve reads out the size under the pointer — or, by default, the
// sizes it likes most. `anchorName` names a ladder rung in plain words.
function anchorName(a, n1, idxMode) {
  if (a.label === "ALL-IN" || (idxMode && a.frac == null)) return "All-in";
  if (a.k === 0 || a.label === "min") return "Min bet";
  const frac = a.frac != null ? a.frac : a.k / n1;
  const pct = Math.round(frac * 100);
  return pct >= 100 ? "Pot" : `${pct}% pot`;
}
function pctOf(p) {
  const v = Math.max(0, Number(p) || 0) * 100;
  return v > 0 && v < 1 ? "<1%" : `${Math.round(v)}%`;
}

function setupCurveReadout() {
  const box = document.getElementById("recommendation");
  if (!box) return;
  const readoutFor = (el) => {
    const svg = el.closest("svg.bet-curve");
    const r = svg && svg.nextElementSibling;
    return r && r.classList.contains("bc-readout") ? r : null;
  };
  const show = (e) => {
    const hit = e.target.closest && e.target.closest(".bc-hit");
    if (!hit) return;
    const r = readoutFor(hit);
    if (r) { r.textContent = hit.dataset.bc; r.classList.add("live"); }
  };
  box.addEventListener("pointerover", show);
  box.addEventListener("click", show);   // a tap on a phone
  box.addEventListener("pointerout", (e) => {
    const svg = e.target.closest && e.target.closest("svg.bet-curve");
    if (!svg || (e.relatedTarget && svg.contains(e.relatedTarget))) return;
    const r = readoutFor(svg);
    if (r && e.pointerType === "mouse") { r.textContent = r.dataset.top; r.classList.remove("live"); }
  });
}

function recDetailHTML(rec, s) {
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

// EV chip(s) for the recommendation line (CPY-012: plain words, not
// "value (own) · true").
function evChipsHTML(ownBB, trueBB, s) {
  const own = fmtSignedValue(ownBB, s);
  const tru = fmtSignedValue(trueBB, s);
  if (!own && !tru) return "";
  if (!tru) {
    return `<span class="rec-value" title="What the network thinks this spot is worth to the player acting">EV ${escapeHTML(own)}</span>`;
  }
  return `<span class="rec-value" title="EV from what the acting player can see · EV using every player's cards">`
    + `EV ${escapeHTML(own ?? "—")} · <span class="rec-value-true">all cards ${escapeHTML(tru)}</span></span>`;
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
    detailHTML = recDetailHTML(rec, s);
  } else {
    actionText = cur.rec_gate
      ? gateActionLabel(cur.rec_gate, cur.rec_chips, cur.to_call_chips, s, commit, cur.street)
      : cur.rec_label;
    dist = cur.gate_probs;
    valueBB = cur.value_bb;
    valueTrueBB = cur.value_true_bb;       // null on v1 / when no critic
    detailHTML = recDetailHTML(cur, s);
  }
  const recPayload = whatif ? whatif.recommendation : cur;
  const who = trainerWho(cur.position, cur.is_hero);
  el.innerHTML = `
    ${untrainedBadgeHTML(recPayload, s)}
    <div class="rec-for muted">${who ? `For ${escapeHTML(who)} · ${escapeHTML(cur.street || "")}` : ""}</div>
    <div class="rec-line">
      ${tag}<span class="rec-action">${escapeHTML(actionText)}</span>
      ${evChipsHTML(valueBB, valueTrueBB, s)}
    </div>
    <div class="rec-dist">${distRowsHTML(dist, callName)}</div>
    ${detailHTML}
  `;
}

// Seats that act before the hero on this street, in turn order, starting
// with the current actor (folded and all-in seats don't act).
function seatsBeforeHero(s) {
  const n = s.num_seats;
  const out = [];
  if (s.actor === null || s.actor === undefined) return out;
  for (let i = 0, seat = s.actor; i < n && seat !== s.hero_seat; i++, seat = (seat + 1) % n) {
    const x = s.seats[seat];
    if (x && !x.folded && !x.all_in && x.participant !== false) out.push(x.position);
  }
  return out;
}

function listJoin(names) {
  if (names.length <= 1) return names.join("");
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

function renderRecommendation(s) {
  const el = document.getElementById("recommendation");
  if (s.trainer) {
    if (s.trainer.review && (s.trainer.review.node_current || s.trainer.review.current)) {
      renderTrainerReviewRecommendation(s, el);
      return;
    }
    el.innerHTML = '<p class="muted">Hidden while you play — after the hand, the review shows the network’s answer at every decision.</p>';
    return;
  }
  const holeCount = s.card_spec ? s.card_spec.hero_hole.length : 5;
  const pickerBtn = '<button type="button" class="enter-cards-btn" data-open-picker>Enter cards</button>';
  const REC_PROMPTS = {
    hole:  `Place Hero's ${holeCount} hole cards to see the network's answer — click cards in the grid, or type a rank then a suit (<kbd>A</kbd><kbd>s</kbd>).`,
    flop:  "Place the flop cards to see the network's answer.",
    turn:  "Place the turn cards to see the network's answer.",
    river: "Place the river cards to see the network's answer.",
  };
  if (s.hero_blocking_reason != null) {
    el.innerHTML = `<p class="muted">${REC_PROMPTS[s.hero_blocking_reason]
      ?? "Place the cards to see the network's answer."}</p>${pickerBtn}`;
    return;
  }
  const rec = s.recommendation;
  if (!rec) {
    // ST-001 / ST-017: say why there is nothing yet, and what to do.
    if (s.terminal) {
      el.innerHTML = '<p class="muted">The hand is over. Click a history line (or Undo) to go back to a decision, or start a new hand.</p>';
      return;
    }
    if (s.actor === null || s.actor === undefined) {
      const st = s.awaiting_next_street;
      el.innerHTML = `<p class="muted">${st ? `Enter the ${escapeHTML(st)} cards to continue.` : "Recommendations appear on Hero's turns."}</p>`;
      return;
    }
    const before = seatsBeforeHero(s);
    const canCheck = s.legal && s.legal.check_call && s.to_call_chips === 0;
    const names = escapeHTML(listJoin(before));
    el.innerHTML = `
      <p class="rec-wait">${before.length > 1 ? `${names} act` : `${names} acts`} before Hero on this street.
        Enter ${before.length > 1 ? "their actions" : "the action"} with the buttons under the table
        — the network answers on Hero's turn.</p>
      ${canCheck ? `<button type="button" class="check-to-hero" id="check-to-hero">Everyone checks to Hero</button>` : ""}`;
    const btn = document.getElementById("check-to-hero");
    if (btn) btn.addEventListener("click", checkToHero);
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
  el.innerHTML = `
    ${untrainedBadgeHTML(rec, s)}
    <div class="rec-line">
      <span class="rec-action">${escapeHTML(actionText)}</span>
      ${evChipsHTML(rec.value_bb, null, s)}
    </div>
    <div class="rec-dist">${distRows}</div>
    ${recDetailHTML(rec, s)}
    ${candidateHTML(s)}
  `;
  const toggle = el.querySelector(".rec-candidate-toggle");
  if (toggle) toggle.addEventListener("click", () => toggleCandidate(toggle.dataset.compare === "1"));
}

// --- Candidate model side by side (FEAT-025; admins) ---------------------------
// When an admin has configured a candidate checkpoint (the "experimental"
// format, PLO5BP_CHECKPOINT_CANDIDATE), Study can show its answer to the same
// spot under the live one — a new model is judged on real spots before it is
// promoted. /formats lists the slot only for users allowed to use it.
function candidateAvailable() {
  return (UI.formats || []).some((f) => f.id === "experimental" && !f.locked && f.model_loaded);
}
function candidateHTML(s) {
  if (!candidateAvailable() || s.format !== "plo5_double_bomb") return "";
  const on = !!s.compare_candidate;
  const c = s.candidate_recommendation;
  let line = "";
  if (on && c) {
    let act = "Check";
    if (c.gate === "fold") act = "Fold";
    else if (c.gate === "check_call") act = s.to_call_chips > 0 ? `Call ${formatUnit(s.to_call_chips, s)}` : "Check";
    else if (c.chips !== null && c.chips !== undefined) {
      act = `${aggVerb(s.to_call_chips, s.street)} ${formatUnit(c.chips + actorCommitChips(s), s)}`;
    }
    const names = ["Fold", s.to_call_chips > 0 ? "Call" : "Check", "Raise"];
    const dist = (c.gate_distribution || [])
      .map((p, i) => `${names[i]} ${Math.round(Number(p) * 100)}%`).join(" · ");
    line = `<div class="rec-candidate-line"><span class="rec-candidate-name">${escapeHTML(c.label || "Candidate")}</span>`
      + ` <b>${escapeHTML(act)}</b><div class="muted">${escapeHTML(dist)}</div></div>`;
  }
  return `<div class="rec-candidate">${line}<button type="button" class="rec-candidate-toggle" `
    + `data-compare="${on ? "0" : "1"}">${on ? "Hide candidate" : "Compare with candidate"}</button></div>`;
}
async function toggleCandidate(on) {
  try {
    const data = await postJSON("/study/compare", { on });
    applyState(data.state);
  } catch (e) { if (e.message !== GATE_HANDLED) showToast(e.message); }
}

// --- Check to Hero, and the action history -----------------------------------

// ST-001: in a spot where nobody has bet yet, every player before Hero can
// check in one click instead of five.
async function checkToHero() {
  if (UI.actionInFlight) return;
  UI.actionInFlight = true;
  setActionsBusy(true);
  try {
    for (let i = 0; i < 12; i++) {
      const s = UI.lastState;
      if (!s || s.trainer || UI.mode !== "study" || s.terminal) break;
      if (s.actor === null || s.actor === undefined || s.actor === s.hero_seat) break;
      if (!(s.legal && s.legal.check_call) || s.to_call_chips > 0) break;
      const data = await postJSON("/study/action", { gate: "check_call" });
      UI.redo = [];
      applyState(data.state);
    }
  } catch (e) { showToast(e.message); }
  finally { UI.actionInFlight = false; setActionsBusy(false); }
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

// History rows: in Study each row is a button that goes back to that
// decision (FEAT-020 — one click instead of Undo, Undo, Undo…).
function renderHistory(s) {
  const el = document.getElementById("history");
  const study = !s.trainer;
  if (!s.history.length) {
    el.innerHTML = `<p class="muted">${study ? "No actions yet — they appear here as you enter them." : "No actions yet."}</p>`;
    renderRedo(s);
    return;
  }
  const walk = historyCommits(s);
  const rows = [];
  for (let i = 0; i < s.history.length; i++) {
    const h = s.history[i];
    const streetClass = `h-${String(h.street).toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
    const isHero = h.seat === s.hero_seat;
    const who = s.trainer && isHero ? "You" : h.position;
    const inner = `
      <span class="h-street ${streetClass}">${escapeHTML(h.street)}</span>
      <span class="h-pos${isHero ? " hero" : ""}">${escapeHTML(who)}</span>
      <span class="h-action">${escapeHTML(actionLabel(h, s, i, walk))}</span>`;
    rows.push(study
      ? `<button type="button" class="history-entry" data-rewind="${i}" title="Go back to this decision">${inner}<span class="h-back" aria-hidden="true">↺</span></button>`
      : `<div class="history-entry">${inner}</div>`);
  }
  el.innerHTML = rows.join("");
  el.scrollTop = el.scrollHeight;
  renderRedo(s);
}

// --- Raise input -------------------------------------------------------------
// Typing a size marks it as the player's own (re-renders keep it); Enter
// bets it.
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
