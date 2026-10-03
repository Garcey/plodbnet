// Study / Trainer client, part 5 of 7 — the Trainer: grading feedback,
// stats, recent hands, the review panel, the settings dialog, in-app help
// and keyboard shortcuts.
//
// Plain scripts, no build step: index.html loads app.core.js, app.table.js,
// app.play.js, app.study.js, app.trainer.js, app.topbar.js and app.js in
// that order, and they share one global scope — any file may call a
// function from any other. Code that runs while a file LOADS may use only
// the files before it; everything else starts from init() in app.js.
"use strict";

// --- Trainer panels: grading feedback and stats --------------------------------

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
  renderDrill(s);
  renderMyTablesToggle(s);
}

function cancelTrainerPick() {
  UI.trainerPick = false;
  document.body.classList.remove("trainer-pick");
}

function hideFeedbackFlash() {
  const el = document.getElementById("feedback-flash");
  if (el) el.hidden = true;
}

const CAT_LABELS = [
  ["best", "Best move"],
  ["correct", "Correct"],
  ["inaccuracy", "Inaccuracy"],
  ["wrong", "Wrong move"],
  ["blunder", "Blunder"],
];
const CAT_NAME = Object.fromEntries(CAT_LABELS);

// The graded verdict of hero's last decision, in the active unit: raise sizes
// are raise-BY deltas, shown as raise-TO totals like the raise input
// (review 2026-09-20 F11). Hero's street commit at the decision: the
// server-sent `actor_commit_chips` when present, else what postAction
// recorded for this hand just before POSTing.
function verdictParts(s, fb) {
  const rec0 = UI.actCommit && UI.actCommit.handNo === s.trainer.hand_no ? UI.actCommit : null;
  const commit = typeof fb.actor_commit_chips === "number" ? fb.actor_commit_chips
    : (rec0 ? rec0.commit : null);
  const street = fb.street || (rec0 ? rec0.street : null);
  const userLabel = fb.user_gate
    ? gateActionLabel(fb.user_gate, fb.user_chips, fb.to_call_chips, s, commit, street)
    : fb.label;
  const recLabel = fb.rec_gate
    ? gateActionLabel(fb.rec_gate, fb.rec_chips, fb.to_call_chips, s, commit, street)
    : fb.rec_label;
  let ev = "";
  if (fb.ev_loss_hidden === true) {
    // Mid-hand the backend withholds the number (it is computed from cards
    // the hero can't see yet) and sends ev_loss_bb: null.
    ev = "EV loss shown at hand end";
  } else if (typeof fb.ev_loss_bb === "number" && isFinite(fb.ev_loss_bb) && fb.ev_loss_bb > 0) {
    ev = `EV ${MINUS}${fmtBBValue(fb.ev_loss_bb, s)}${evBand(fb.ev_loss_se_bb, s)}`;
  }
  return {
    marks: fb.marks || "",
    cat: fb.category,
    catLabel: CAT_NAME[fb.category] || String(fb.category || ""),
    score: Math.round(Number(fb.score) || 0),
    userLabel: userLabel || "",
    best: fb.category !== "best" && recLabel ? recLabel : "",
    ev,
  };
}

function renderFeedbackFlash(s, force = false) {
  // During frame playback the frames carry the previous decision's
  // feedback — only the explicit (forced) call may flash.
  if (UI.animating && !force) return;
  renderLastVerdict(s);
  const fb = s.trainer.feedback;
  if (!fb) return;
  const key = `${s.trainer.hand_no}:${fb.decision_idx}`;
  if (key === UI.feedbackShownIdx) return;
  UI.feedbackShownIdx = key;
  const v = verdictParts(s, fb);
  const el = document.getElementById("feedback-flash");
  document.getElementById("feedback-marks").textContent = v.marks;
  document.getElementById("feedback-text").textContent = `${v.userLabel} · ${v.catLabel} ${v.score}%`;
  document.getElementById("feedback-sub").textContent =
    [v.best ? `best: ${v.best}` : "", v.ev].filter(Boolean).join(" · ");
  el.className = `flash-${fb.category}`;
  el.hidden = false;
  if (UI.feedbackTimer) clearTimeout(UI.feedbackTimer);
  UI.feedbackTimer = setTimeout(() => { el.hidden = true; }, 2600);
}

// ST-020: the flash is brief (it sits on the table while the opponents
// play), so the same verdict also stays in the action panel until your next
// decision — readable, and announced to screen readers.
function renderLastVerdict(s, force = false) {
  const el = document.getElementById("last-verdict");
  if (!el) return;
  if (UI.animating && !force) return;
  const t = s && s.trainer;
  const fb = t && t.feedback;
  if (!fb || !t.hand_active) {
    el.hidden = true;
    el.innerHTML = "";
    return;
  }
  const v = verdictParts(s, fb);
  el.hidden = false;
  el.className = `last-verdict cat-edge-${escapeHTML(v.cat)}`;
  el.setAttribute("role", "status");
  el.innerHTML = `<span class="lv-label">Your last move</span>`
    + `<span class="lv-main"><b class="cat-text-${escapeHTML(v.cat)}">${escapeHTML(v.marks)} ${escapeHTML(v.catLabel)} · ${v.score}%</b>`
    + ` — ${escapeHTML(v.userLabel)}</span>`
    + (v.best || v.ev ? `<span class="lv-sub">${escapeHTML([v.best ? `best: ${v.best}` : "", v.ev].filter(Boolean).join(" · "))}</span>` : "");
}

function statsBlockHTML(title, st, scope, s) {
  const moves = st.moves || 0;
  const rows = CAT_LABELS.map(([k, label]) => {
    const c = (st.cat_counts && st.cat_counts[k]) || 0;
    const pct = moves > 0 ? (100 * c) / moves : 0;
    return `<div class="stat-cat-row">
      <span class="stat-cat-count">${fmtNum(c, 0)}</span>
      <div class="rec-dist-track"><div class="rec-dist-fill cat-${k}" style="width:${pct.toFixed(1)}%"></div></div>
      <span class="stat-cat-label">${label}</span>
    </div>`;
  }).join("");
  const score = (st.gto_score !== null && st.gto_score !== undefined)
    ? `${fmtNum(st.gto_score, 1)}%` : "—";
  const evTotal = Number(st.ev_loss_total_bb) || 0;
  const evHand = st.ev_loss_per_hand_bb;
  const evLine = `EV lost ${fmtBBValue(evTotal, s)}` +
    ((evHand !== null && evHand !== undefined) ? ` · ${fmtBBValue(evHand, s)} a hand` : "");
  // the hands' real results, opposite the EV lost (owner, 2026-10-03: "results oriented …
  // fun to see"); absent until a hand with a result is counted
  const net = st.net_total_bb, netHand = st.net_per_hand_bb;
  let netLine = "";
  if (net !== null && net !== undefined && Number.isFinite(Number(net))) {
    const n = Number(net);
    const cls = n > 0 ? "pos" : n < 0 ? "neg" : "muted";
    const verb = n > 0 ? "Won" : n < 0 ? "Lost" : "Even";
    const each = (n !== 0 && netHand !== null && netHand !== undefined)
      ? ` · ${fmtBBValue(Math.abs(Number(netHand)), s)} a hand` : "";
    // (a lifetime from before 2026-10-03 has hands without a result: say how many count)
    const counted = Number(st.net_hands) || 0;
    const over = counted ? `over ${fmtNum(counted, 0)} hand${counted === 1 ? "" : "s"} ` : "";
    netLine = `<span class="stats-net ${cls}" title="Your results ${over}— the cards that came, not the EV">`
      + `${verb}${n !== 0 ? ` ${fmtBBValue(Math.abs(n), s)}` : ""}${each}</span>`;
  }
  return `
    <div class="stats-title" role="button" tabindex="0" aria-expanded="${statsCollapsed(scope) ? "false" : "true"}">
      <span><span class="stats-chev" aria-hidden="true">&#9662;</span>${title}</span>
      <button class="stats-reset" data-scope="${scope}" type="button" aria-label="Reset ${title.toLowerCase()} stats">Reset</button>
    </div>
    <div class="stats-top">
      <div><span class="stats-num">${fmtNum(st.hands ?? 0, 0)}</span><span class="stats-cap">hands</span></div>
      <div><span class="stats-num">${fmtNum(moves, 0)}</span><span class="stats-cap">moves</span></div>
      <div><span class="stats-num stats-score">${score}</span><span class="stats-cap">accuracy</span></div>
    </div>
    ${rows}
    <div class="stats-foot"><span class="stats-ev muted">${evLine}</span>${netLine}</div>
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
  if (el) {
    el.classList.toggle("collapsed", prefs[scope]);
    const t = el.querySelector(".stats-title");
    if (t) t.setAttribute("aria-expanded", prefs[scope] ? "false" : "true");
  }
}

function renderTrainerStats(s) {
  const stats = s.trainer.stats || {};
  const se = document.getElementById("stats-session");
  const lt = document.getElementById("stats-lifetime");
  se.innerHTML = statsBlockHTML("Session", stats.session || {}, "session", s);
  lt.innerHTML = statsBlockHTML("Lifetime", stats.lifetime || {}, "lifetime", s);
  se.classList.toggle("collapsed", statsCollapsed("session"));
  lt.classList.toggle("collapsed", statsCollapsed("lifetime"));
  renderRecentHands(s);
}

async function resetStats(scope) {
  const ok = await confirmDialog(scope === "lifetime"
    ? { title: "Reset lifetime stats?", body: "This permanently clears your lifetime hands, moves, accuracy and EV-loss totals.", ok: "Reset", danger: true }
    : { title: "Reset session stats?", body: "This clears this session's hands, moves, accuracy and EV-loss totals.", ok: "Reset", danger: true });
  if (ok) postTrainer("stats/reset", { scope });
}

// --- Recent hands (FEAT-017 / FEAT-027) -------------------------------------------
// The last finished hands, newest first; one click reopens a hand's review
// (graded decisions, what-if, Open in Study, Repeat). Stored server-side
// with your stats, so they survive a reload.
function relTime(t) {
  const sec = Math.max(0, Date.now() / 1000 - Number(t || 0));
  if (sec < 60) return "just now";
  if (sec < 3600) return `${Math.round(sec / 60)} min ago`;
  if (sec < 86400) return `${Math.round(sec / 3600)} h ago`;
  return `${Math.round(sec / 86400)} d ago`;
}

function renderRecentHands(s) {
  const el = document.getElementById("recent-hands");
  if (!el) return;
  const list = (s.trainer && Array.isArray(s.trainer.recent)) ? s.trainer.recent : [];
  el.hidden = !list.length;
  // the tab under the pinned stats (a desktop): how many there are to look back at
  const tab = document.getElementById("ph-tab");
  if (tab) {
    tab.hidden = !list.length;
    document.getElementById("ph-count").textContent = list.length ? `(${list.length})` : "";
  }
  if (!list.length) { el.innerHTML = ""; setPrevHands(false); return; }
  const rows = list.map((r) => {
    const net = typeof r.net_bb === "number" ? fmtSignedValue(r.net_bb, s) : "";
    const cls = r.net_bb > 0 ? "pos" : r.net_bb < 0 ? "neg" : "";
    const score = typeof r.score === "number" ? `${Math.round(r.score)}%` : "—";
    const other = r.format && r.format !== s.format;
    const title = other ? "Played in another format — switch format to open it"
      : `Open the review of this hand (${relTime(r.t)})`;
    return `<button type="button" class="recent-row${other ? " other" : ""}" data-recent="${escapeHTML(r.id)}" title="${escapeHTML(title)}"${other ? " disabled" : ""}>`
      + `<span class="rh-pos">${escapeHTML(r.position || "?")}${r.repeat ? " ↻" : ""}</span>`
      + `<span class="rh-seats muted">${Number(r.seats) || "?"} players</span>`
      + `<span class="rh-net ${cls}">${escapeHTML(net)}</span>`
      + `<span class="rh-score" title="Accuracy">${score}</span></button>`;
  });
  el.innerHTML = `<div class="stats-title stats-title-static"><span>Previous hands</span></div>${rows.join("")}`;
}

// --- Previous hands over the side rail (2026-10-03, owner: "pin the session and lifetime
// stats to the bottom of the right panel … a little tab at the bottom … 'Previous hands'
// … pressing that little tab or scrolling down pulls the hand history up into the full
// right panel"). A desktop's rail is the recommendation, the hand's actions (they fill the
// room and scroll inside it) and the stats pinned at the bottom, so nothing moves as a
// hand goes on; the tab — or scrolling down past the stats — slides the list up over the
// whole rail. Back, Esc, or scrolling up at the list's top slides it away. (A phone lists
// the hands under the stats: style.css.)
const RAIL_SHEET_MQ = window.matchMedia("(min-width: 861px)");
function prevHandsOpen() { return document.body.classList.contains("hands-open"); }
function setPrevHands(open, focus = false) {
  const want = !!open && RAIL_SHEET_MQ.matches && document.body.classList.contains("trainer-mode");
  if (want === prevHandsOpen()) return;
  document.body.classList.toggle("hands-open", want);
  UI.handsToggledAt = performance.now();
  const tab = document.getElementById("ph-tab");
  if (tab) tab.setAttribute("aria-expanded", want ? "true" : "false");
  if (focus) (want ? document.getElementById("ph-back") : tab)?.focus();
}

function setupPreviousHands() {
  const $ = (id) => document.getElementById(id);
  const tab = $("ph-tab"), sheet = $("prev-hands"), rail = $("side-rail"), list = $("recent-hands");
  if (!tab || !sheet || !rail || !list) return;
  tab.addEventListener("click", () => setPrevHands(true, true));
  $("ph-back").addEventListener("click", () => setPrevHands(false, true));
  sheet.addEventListener("click", (e) => {
    const row = e.target.closest("[data-recent]");
    if (!row) return;
    setPrevHands(false);
    onRecentHandClick(row);
  });
  // one wheel gesture flips the rail once (a trackpad's flick sends many events)
  const settled = () => performance.now() - (UI.handsToggledAt || 0) > 500;
  rail.addEventListener("wheel", (e) => {
    if (e.deltaY <= 0 || prevHandsOpen() || tab.hidden || !settled()) return;
    if (e.target.closest("#history")) return;  // (reading the hand's actions never flips it)
    if (rail.scrollTop + rail.clientHeight < rail.scrollHeight - 2) return;  // (a short window scrolls first)
    setPrevHands(true);
  }, { passive: true });
  sheet.addEventListener("wheel", (e) => {
    if (e.deltaY >= 0 || !prevHandsOpen() || list.scrollTop > 0 || !settled()) return;
    setPrevHands(false);
  }, { passive: true });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && prevHandsOpen() && !document.querySelector("dialog[open]")) {
      e.preventDefault();
      setPrevHands(false, true);
    }
  });
  RAIL_SHEET_MQ.addEventListener("change", () => { if (!RAIL_SHEET_MQ.matches) setPrevHands(false); });
}

async function onRecentHandClick(btn) {
  const id = btn.dataset.recent;
  const s = UI.lastState;
  if (!id || !s || !s.trainer || UI.actionInFlight || btn.disabled) return;
  if (s.trainer.hand_active) {
    const ok = await confirmDialog({
      title: "Leave this hand?",
      body: "Opening an earlier hand ends the one you're playing. An unfinished hand isn't counted in your stats.",
      ok: "Open it",
      danger: true,
    });
    if (!ok) return;
  }
  UI.actionInFlight = true;
  setActionsBusy(true);
  try {
    const data = await postJSON("/trainer/hands/open", { id });
    const st = data.state;
    // Its last verdict was shown when it was played — don't flash it again.
    if (st && st.trainer && st.trainer.feedback) {
      UI.feedbackShownIdx = `${st.trainer.hand_no}:${st.trainer.feedback.decision_idx}`;
    }
    UI.reviewNode = null;
    cancelTrainerPick();
    UI.selectedSlot = null;
    applyState(st);
    const panel = document.getElementById("review-panel");
    if (panel && !panel.hidden) panel.scrollIntoView({ block: "nearest", behavior: "smooth" });
  } catch (e) { showToast(e.message); }
  finally { UI.actionInFlight = false; setActionsBusy(false); }
}

function ringClass(pct) {
  return pct >= 80 ? "ring-good" : pct >= 50 ? "ring-mid" : "ring-bad";
}

// "You (BTN)" for the hero, the position otherwise (CPY-013: the Trainer
// speaks to the player; "Hero" belongs to Study, where you enter a hand).
function trainerWho(position, isHero) {
  return isHero ? `You (${position})` : String(position || "");
}

function renderReviewPanel(s) {
  const panel = document.getElementById("review-panel");
  const rv = s.trainer.review;
  if (!rv) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  UI.reviewNode = rv.node;

  const C = 2 * Math.PI * 26;
  const pct = rv.hand_score ?? 0;
  const fill = document.getElementById("review-ring-fill");
  fill.style.strokeDasharray = `${((C * pct) / 100).toFixed(1)} ${C.toFixed(1)}`;
  fill.setAttribute("class", `ring-fill ${ringClass(pct)}`);
  document.getElementById("review-score-num").textContent =
    (rv.hand_score !== null && rv.hand_score !== undefined)
      ? `${Math.round(rv.hand_score)}%` : "—";

  // Arrows + label step the unified cursor over every action of the hand.
  const nc = rv.node_current;
  document.getElementById("review-step-label").textContent = nc
    ? `Action ${rv.node + 1} of ${rv.num_nodes} · ${trainerWho(nc.position, nc.is_hero)}`
    : `Action ${rv.node + 1} of ${rv.num_nodes}`;
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
    b.className = `review-chip cat-border-${escapeHTML(d.category)}` + (isCurrent ? " current" : "");
    if (isCurrent) b.setAttribute("aria-current", "step");
    const nd = rv.nodes ? rv.nodes[d.node_idx] : null;
    const w = walk[d.node_idx];
    const label = nd && w && nd.actual_gate
      ? gateActionLabel(
          nd.actual_gate, nd.actual_chips,
          typeof d.to_call_chips === "number" ? d.to_call_chips : Math.max(0, w.level - w.before),
          s, reviewCommitBefore(s, rv, d.node_idx, nd.seat, d, walk), nd.street)
      : d.user_label;
    b.innerHTML = `<span class="rc-street">${escapeHTML(d.street)}</span>${escapeHTML(label)}` +
      ` <span class="rc-score">${Math.round(d.score)}%</span>`;
    b.title = `${CAT_NAME[d.category] || d.category} · ${Math.round(d.score)}%`;
    b.addEventListener("click", () => trainerReviewGotoNode(d.node_idx));
    chips.appendChild(b);
  });

  document.getElementById("review-whatif-bar").hidden = !rv.whatif;

  const cur = rv.node_current || rv.current;
  const detail = document.getElementById("review-detail");
  if (!cur) { detail.innerHTML = ""; return; }
  const commit = reviewCurCommit(s, rv, cur, walk);
  const curLabel = (gate, gateChips, fallback) => escapeHTML(gate
    ? gateActionLabel(gate, gateChips, cur.to_call_chips, s, commit, cur.street)
    : fallback);
  const recLabel = curLabel(cur.rec_gate, cur.rec_chips, cur.rec_label);
  if (reviewNodeUngraded(rv.node_current)) {
    // Ungraded node — an opponent's decision, or a hero moot auto-check
    // (`category: null`; reading cur.category.toUpperCase() there threw and
    // blanked the panel — review 2026-09-20 F12). Show what actually happened
    // vs the network's pick (policy + EVs ride in the recommendation panel).
    const who = escapeHTML(trainerWho(cur.position, cur.is_hero));
    const moves = cur.is_hero
      ? "An automatic check — betting was over (everyone left was all-in), so there was nothing to decide."
      : `Played <b>${curLabel(cur.actual_gate, cur.actual_chips, cur.actual_label)}</b>` +
        ` · the network plays <b>${recLabel}</b>`;
    detail.innerHTML = `
      <div class="review-villain">${who} · ${escapeHTML(cur.street)}</div>
      <div class="review-moves">${moves}</div>
    `;
    return;
  }
  const num = (v) => typeof v === "number" && isFinite(v);
  let evRow = "";
  if (num(cur.ev_loss_bb)) {
    const detailBit = (num(cur.ev_user_bb) && num(cur.ev_best_bb))
      ? ` <span class="muted">(your move ${escapeHTML(fmtSignedValue(cur.ev_user_bb, s))} vs the network's ${escapeHTML(fmtSignedValue(cur.ev_best_bb, s))})</span>`
      : "";
    evRow = `<div class="review-ev">EV loss <b>${escapeHTML(fmtBBValue(cur.ev_loss_bb, s))}</b>${escapeHTML(evBand(cur.ev_loss_se_bb, s))}${detailBit}</div>`;
  }
  const rescored = rv.whatif
    ? `<div class="review-rescored">With these cards: <b class="cat-text-${escapeHTML(rv.whatif.rescored.category)}">` +
      `${escapeHTML(CAT_NAME[rv.whatif.rescored.category] || rv.whatif.rescored.category)}</b> ${Math.round(rv.whatif.rescored.score)}%</div>`
    : "";
  detail.innerHTML = `
    <div class="review-verdict cat-text-${escapeHTML(cur.category)}">${escapeHTML(cur.marks)} ${escapeHTML(CAT_NAME[cur.category] || cur.category)} · ${Math.round(cur.score)}%</div>
    <div class="review-moves">You: <b>${curLabel(cur.user_gate, cur.user_chips, cur.user_label)}</b> · Network: <b>${recLabel}</b></div>
    ${evRow}${rescored}
    <div class="muted review-hint">Tip: click your cards or the board to see how other cards change the answer.</div>
  `;
}

// --- Trainer settings dialog -------------------------------------------------

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
  // My tables (2026-10-03): the table's players, stacks and ante come from your own
  // tables, so their rows step aside
  const mine = _tsVal("ts-tables") === "mine";
  for (const id of ["ts-row-seats", "ts-row-stacks", "ts-row-ante"]) _tsShow(id, !mine);
  _tsShow("ts-tables-help", mine);
  if (mine) fillMyTablesHelp();
  const seatsMode = _tsVal("ts-seats-mode");
  _tsShow("ts-seats-fixed-wrap", seatsMode === "fixed");
  _tsShow("ts-seats-range-wrap", seatsMode === "random");
  const stacksMode = _tsVal("ts-stacks-mode");
  _tsShow("ts-stack-fixed-wrap", stacksMode === "fixed");
  _tsShow("ts-stack-range-wrap", stacksMode === "random");
  _tsShow("ts-per-seat-wrap", stacksMode === "per_seat" && !mine);
  _tsShow("ts-hero-kth-wrap", _tsVal("ts-hero-mode") === "kth");
}

function ordinal(n) {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return `${n}${s[(v - 20) % 10] || s[v] || s[0]}`;
}

// Per-seat stack ranges are counted from YOUR seat (ST-030): "You", then the
// players to your left in turn order. They used to be engine seats "Seat 1-6",
// which moved with the random seating and meant nothing on the table.
function perSeatLabel(i) {
  return i === 0 ? "You" : `${ordinal(i)} to your left`;
}

function openTrainerSettings() {
  const s = UI.lastState;
  const t = s && s.trainer ? s.trainer.settings : null;
  if (!t) return;
  document.getElementById("ts-tables").value = t.tables === "mine" ? "mine" : "custom";
  _tsShow("ts-row-tables", myTablesFormat(s));
  UI.myTables = null;  // (asked again: an upload may have changed it)
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
  document.getElementById("ts-spot").value = t.spot || "any";
  document.getElementById("ts-ante-bb").value = t.ante_bb;
  // The build's ceiling on EV rollouts (`trainer.mc_rollouts_max`, e.g. 32 on
  // the public build) bounds the field; the server clamps too, this just
  // keeps the form honest about what will be used.
  const mcInput = document.getElementById("ts-mc-rollouts");
  const mcMax = trainerMcRolloutsMax(s);
  if (mcMax !== null) mcInput.max = String(mcMax);
  mcInput.value = mcMax !== null ? Math.min(t.mc_rollouts, mcMax) : t.mc_rollouts;
  // (say the ceiling: a bigger number used to come back as the cap without a word)
  document.getElementById("ts-mc-max").textContent = mcMax !== null ? ` At most ${mcMax}.` : "";
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
    const name = perSeatLabel(i);
    row.innerHTML = `<span class="ts-seat-name">${escapeHTML(name)}</span>
      <input type="number" class="ts-ps-lo" data-i="${i}" min="1" max="1000" step="0.5" inputmode="decimal" value="${escapeHTML(lo)}" aria-label="${escapeHTML(name)}: smallest stack (bb)" /> –
      <input type="number" class="ts-ps-hi" data-i="${i}" min="1" max="1000" step="0.5" inputmode="decimal" value="${escapeHTML(hi)}" aria-label="${escapeHTML(name)}: largest stack (bb)" />`;
    wrap.appendChild(row);
  }
  settingsError(null);
  syncSettingsVisibility();
  UI.settingsOpen = true;
  const dlg = document.getElementById("trainer-settings-modal");
  dlg._onClose = () => { UI.settingsOpen = false; };
  openDialog(dlg);
}

function closeTrainerSettings() {
  UI.settingsOpen = false;
  closeDialog(document.getElementById("trainer-settings-modal"));
}

// Inline form errors (ST-030): caught here, in plain words, before the
// server's validation would answer with "stacks_per_seat_bb[2]: …".
function settingsError(msg, input) {
  const el = document.getElementById("ts-error");
  for (const x of document.querySelectorAll("#trainer-settings-modal [aria-invalid]")) {
    x.removeAttribute("aria-invalid");
  }
  if (!el) return;
  el.hidden = !msg;
  el.textContent = msg || "";
  if (input) {
    input.setAttribute("aria-invalid", "true");
    input.focus();
  }
}

function validateTrainerSettings(body) {
  const q = (id) => document.getElementById(id);
  const inRange = (v, lo, hi) => Number.isFinite(v) && v >= lo && v <= hi;
  const mine = body.tables === "mine";  // (its players / stacks / ante rows are unused, and hidden)
  if (!mine && body.seats_mode === "fixed" && !inRange(body.seats_fixed, 2, 6)) {
    return ["Players must be between 2 and 6.", q("ts-seats-fixed")];
  }
  if (!mine && body.seats_mode === "random") {
    if (!inRange(body.seats_min, 2, 6)) return ["Players must be between 2 and 6.", q("ts-seats-min")];
    if (!inRange(body.seats_max, 2, 6)) return ["Players must be between 2 and 6.", q("ts-seats-max")];
    if (body.seats_min > body.seats_max) return ["The fewest players can't be more than the most.", q("ts-seats-min")];
  }
  if (!mine && body.stacks_mode === "fixed" && !inRange(body.stack_bb, 1, 1000)) {
    return ["Stacks must be between 1 and 1,000bb.", q("ts-stack-bb")];
  }
  if (!mine && body.stacks_mode === "random") {
    if (!inRange(body.stack_min_bb, 1, 1000)) return ["Stacks must be between 1 and 1,000bb.", q("ts-stack-min-bb")];
    if (!inRange(body.stack_max_bb, 1, 1000)) return ["Stacks must be between 1 and 1,000bb.", q("ts-stack-max-bb")];
    if (body.stack_min_bb > body.stack_max_bb) return ["The smallest stack can't be larger than the largest.", q("ts-stack-min-bb")];
  }
  if (!mine && body.stacks_mode === "per_seat") {
    for (let i = 0; i < 6; i++) {
      const [lo, hi] = body.stacks_per_seat_bb[i];
      const loEl = document.querySelector(`.ts-ps-lo[data-i="${i}"]`);
      const hiEl = document.querySelector(`.ts-ps-hi[data-i="${i}"]`);
      if (!inRange(lo, 1, 1000)) return [`${perSeatLabel(i)}: stacks must be between 1 and 1,000bb.`, loEl];
      if (!inRange(hi, 1, 1000)) return [`${perSeatLabel(i)}: stacks must be between 1 and 1,000bb.`, hiEl];
      if (lo > hi) return [`${perSeatLabel(i)}: the smallest stack can't be larger than the largest.`, loEl];
    }
  }
  if (!mine && !inRange(body.ante_bb, 0, 100)) return ["The ante must be between 0 and 100bb.", q("ts-ante-bb")];
  if (!Number.isFinite(body.mc_rollouts) || body.mc_rollouts < 0) {
    return ["EV-loss samples must be 0 or more.", q("ts-mc-rollouts")];
  }
  return null;
}

async function saveTrainerSettings() {
  const perSeat = [];
  for (let i = 0; i < 6; i++) {
    const lo = parseFloat(document.querySelector(`.ts-ps-lo[data-i="${i}"]`).value);
    const hi = parseFloat(document.querySelector(`.ts-ps-hi[data-i="${i}"]`).value);
    perSeat.push([lo, hi]);
  }
  const cur = UI.lastState && UI.lastState.trainer ? UI.lastState.trainer.settings : null;
  const body = {
    tables: _tsVal("ts-tables") === "mine" && myTablesFormat(UI.lastState) ? "mine" : "custom",
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
    spot: _tsVal("ts-spot"),
    ante_bb: _tsNum("ts-ante-bb"),
    mc_rollouts: clampMcRollouts(_tsInt("ts-mc-rollouts"), UI.lastState),
    // The $ rate is a display preference now (top bar); keep the stored one.
    ...(cur && validRate(cur.dollars_per_bb) ? { dollars_per_bb: cur.dollars_per_bb } : {}),
  };
  const bad = validateTrainerSettings(body);
  if (bad) { settingsError(bad[0], bad[1]); return; }
  settingsError(null);
  // Playback prefs are client-only (global, format-independent) — persist to
  // localStorage and apply live, independent of the server settings POST.
  let animMs = parseInt(document.getElementById("ts-anim-ms").value, 10);
  if (!Number.isFinite(animMs) || animMs < 0) animMs = TRAINER_ANIM_DEFAULT_MS;
  animMs = Math.min(animMs, TRAINER_ANIM_MAX_MS);
  trainerPrefs.animMs = animMs;
  trainerPrefs.ffFold = document.getElementById("ts-ff-fold").checked;
  lsSet("plo5bp-trainer-anim-ms", String(trainerPrefs.animMs));
  lsSet("plo5bp-trainer-ff-fold", trainerPrefs.ffFold ? "true" : "false");

  const save = document.getElementById("ts-save");
  save.disabled = true;
  try {
    const data = await postJSON("/trainer/settings", body);
    await animateTrainerResponse(data);
    closeTrainerSettings();
  } catch (e) {
    if (e.message !== GATE_HANDLED) settingsError(e.message);
  } finally {
    save.disabled = false;
  }
}

// --- In-app help (FEAT-015 / ACC-010 / CPY-014 / ST-021) --------------------------
// Short, plain explanations next to the thing they explain: how moves are
// graded, how to read the network's output, and the Trainer's shortcuts.
const HELP = {
  grading: {
    title: "How moves are graded",
    html: `
      <p>Each of your moves is compared with what the network does in the same spot: how often it plays your move next to its favourite. The score counts that on a log scale, the way the network weighs its choices &mdash; <b>100%</b> is its favourite, a move it plays half as often scores about 84%, a third as often 75%, a tenth as often 47%. When it mixes several moves the spot is close, so every move in its mix grades well. For bets and raises it also counts how close your size is to the sizes it prefers (the right move with an odd size is an inaccuracy at worst).</p>
      <ul class="help-bands">
        <li><b class="cat-text-best">Best move</b> its favourite, or a move it plays at least &frac34; as often (93% or more)</li>
        <li><b class="cat-text-correct">Correct</b> at least a quarter as often &mdash; part of its mix (68% or more)</li>
        <li><b class="cat-text-inaccuracy">Inaccuracy</b> at least a tenth as often (47&ndash;68%)</li>
        <li><b class="cat-text-wrong">Wrong move</b> at least a fiftieth as often (10&ndash;47%)</li>
        <li><b class="cat-text-blunder">Blunder</b> rarer than that, or an action the network takes less than 2% of the time</li>
      </ul>
      <p><b>Accuracy</b> is your average score. <b>EV loss</b> estimates what a move cost against the network's choice by simulating run-outs; it is noisy on one hand and settles over many.</p>
      <p class="muted">The network approximates game-theory-optimal play through self-play &mdash; it is not a solver, so read it as a very strong player's opinion.</p>`,
  },
  reading: {
    title: "Reading the recommendation",
    html: `
      <p><b>The big line</b> is the network's preferred action. Right now this top choice is its most reliable signal.</p>
      <p><b>The bars</b> show how often it folds, checks or calls, and bets or raises here. While the models are still training these mixes are soft, so second-choice lines can show more weight than they should.</p>
      <p><b>The curve</b> shows the bet sizes it likes, from the minimum to the pot (or all-in when stacks are short). The white dot is the size it picks; a ring marks the size that was played.</p>
      <p><b>EV</b> is the network's estimate of what the spot is worth to the player acting. In a Trainer review, <b>EV (all cards)</b> also uses every player's cards.</p>`,
  },
  shortcuts: {
    title: "Trainer shortcuts",
    html: `
      <table class="help-keys">
        <tr><td><kbd>F</kbd></td><td>Fold</td></tr>
        <tr><td><kbd>C</kbd> or <kbd>Space</kbd></td><td>Check or call</td></tr>
        <tr><td><kbd>R</kbd> or <kbd>B</kbd></td><td>Bet or raise &mdash; type a size, then <kbd>Enter</kbd></td></tr>
        <tr><td><kbd>1</kbd>&ndash;<kbd>9</kbd></td><td>Pick a bet-size preset</td></tr>
        <tr><td><kbd>N</kbd></td><td>Next hand (once the hand is over)</td></tr>
        <tr><td><kbd>&larr;</kbd> <kbd>&rarr;</kbd></td><td>Step through the review</td></tr>
        <tr><td><kbd>Home</kbd> <kbd>End</kbd></td><td>First / last action of the review</td></tr>
        <tr><td><kbd>?</kbd></td><td>This list</td></tr>
      </table>
      <p class="muted"><button type="button" class="linklike" data-help-go="grading">How moves are graded</button></p>`,
  },
};

function showHelp(topic) {
  const h = HELP[topic];
  const dlg = document.getElementById("help-dlg");
  if (!h || !dlg) return;
  document.getElementById("help-title").textContent = h.title;
  document.getElementById("help-body").innerHTML = h.html;
  openDialog(dlg);
}

// --- Trainer keyboard (ST-021) -------------------------------------------------
function typingInField(t) {
  const tag = t && t.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || !!(t && t.isContentEditable);
}
function clickIfEnabled(el) {
  if (el && !el.disabled && !el.hidden && el.offsetParent !== null) { el.click(); return true; }
  return false;
}

function onTrainerKey(e) {
  if (UI.mode !== "trainer" || e.ctrlKey || e.metaKey || e.altKey || e.defaultPrevented) return;
  if (document.querySelector("dialog[open]") || UI.settingsOpen) return;
  if (document.body.classList.contains("ranges-mode")) return;
  const s = UI.lastState;
  if (!s || !s.trainer) return;
  const t = e.target;
  if (typingInField(t)) {
    if (e.key === "Escape" && t.id === "raise-input") { t.blur(); e.preventDefault(); }
    return;
  }
  const k = e.key;
  const lower = k.length === 1 ? k.toLowerCase() : k;
  const done = () => e.preventDefault();
  if (k === "?") { done(); showHelp("shortcuts"); return; }
  const tr = s.trainer;
  if (!tr.hand_active) {
    if (lower === "n") { done(); clickIfEnabled(document.getElementById("review-next-hand")) || clickIfEnabled(document.getElementById("trainer-new-hand-btn")); return; }
    if (tr.review) {
      const map = { ArrowLeft: "review-prev", ArrowRight: "review-next", Home: "review-first", End: "review-last" };
      if (map[k]) { done(); clickIfEnabled(document.getElementById(map[k])); }
    }
    return;
  }
  if (UI.animating || UI.actionInFlight) return;
  const heroTurn = s.actor !== null && s.actor !== undefined && s.actor === s.hero_seat;
  if (!heroTurn) return;
  if (lower === "f") { done(); clickIfEnabled(document.getElementById("fold-btn")); return; }
  if (lower === "c" || (k === " " && !(t && t.tagName === "BUTTON"))) {
    done(); clickIfEnabled(document.getElementById("check-btn")); return;
  }
  const raise = document.getElementById("raise-go");
  const sizing = document.getElementById("sizing");
  if ((lower === "r" || lower === "b") && raise && !raise.hidden) {
    done();
    // the size box when it shows (type a size, Enter bets it); otherwise the
    // button itself — on a phone its first press opens the sizing panel
    const input = document.getElementById("raise-input");
    if (sizing && !sizing.hidden && input && input.offsetParent !== null) input.focus();
    else clickIfEnabled(raise);
    return;
  }
  if (/^[1-9]$/.test(k) && raise && !raise.hidden) {
    const chip = document.querySelectorAll("#raise-presets button[data-i]")[Number(k) - 1];
    if (chip) {
      done();
      if (sizing && sizing.hidden) clickIfEnabled(raise);  // (a phone: open the panel)
      chip.click();
      const input = document.getElementById("raise-input");
      if (input && input.offsetParent !== null) input.focus();
    }
  }
}

// A deal (new hand, repeat). One in-flight guard for all of them: the public build's
// free-tier middleware counts every POST /trainer/new_hand, so an unguarded
// double-click burns 2 of 5 daily hands while showing one. `postTrainer` never
// rejects (its own try/catch), so clearing in `.finally` is always reached.
function dealTrainer(path) {
  if (UI.actionInFlight) return Promise.resolve(false);
  cancelTrainerPick();
  UI.reviewNode = null;
  UI.selectedSlot = null;
  UI.actionInFlight = true;
  setActionsBusy(true);
  startWorking("Dealing…");
  return postTrainer(path).finally(() => {
    stopWorking();
    UI.actionInFlight = false;
    setActionsBusy(false);
  });
}

// --- The mistakes drill (Hand review, 2026-10-03) ------------------------------------
// `/?mode=trainer&drill=1` (Hand review's "Start the drill"): the Trainer deals your own
// mistakes back — the exact spots of your uploaded hands the network graded a wrong move
// or a blunder — one decision at a time (POST /trainer/drill/next). Worst first (the
// default, remembered here): the worst come up most often, a spot you fix comes up less
// and one you miss again more; off, every spot equally often. Next spot / Try again
// (practice: it changes nothing) / Exit drill. The server keeps the round and each
// spot's priority (handreview_store); a spot ends with your decision.
const DRILL_PRIO_KEY = "plo5bp-drill-prio";
// A spot replays every action of the hand before it: at most this pause between them.
const DRILL_REPLAY_MS = 700;
UI.drill = null;         // {prioritize} while drilling
UI.drillOffAt = 0;       // the hand number the drill was left at (an older state can't re-enter it)

function drillRequested() {
  try { return new URLSearchParams(location.search).get("drill") === "1"; } catch (_) { return false; }
}
function drillPrioritize() { return lsGet(DRILL_PRIO_KEY) !== "off"; }
function syncDrillUrl(on) {
  try {
    const url = new URL(location.href);
    if ((url.searchParams.get("drill") === "1") === on) return;
    if (on) url.searchParams.set("drill", "1"); else url.searchParams.delete("drill");
    history.replaceState(history.state, "", url.pathname + url.search + url.hash);
  } catch (_) { /* old browser: the drill still works */ }
}

async function startDrill() {
  UI.drill = { prioritize: drillPrioritize() };
  syncDrillUrl(true);
  renderDrill(UI.lastState);
  return dealDrill(true);
}

// The next spot (`resume`: the one dealt before, if it wasn't played — a reload).
async function dealDrill(resume) {
  const body = { prioritize: UI.drill ? UI.drill.prioritize : true, resume: !!resume };
  if (UI.actionInFlight) return false;
  cancelTrainerPick();
  UI.reviewNode = null;
  UI.selectedSlot = null;
  UI.actionInFlight = true;
  setActionsBusy(true);
  startWorking("Finding your next mistake…");
  try {
    let data;
    try {
      data = await postJSON("/trainer/drill/next", body);
    } catch (e) {
      if (!(e && e.status === 409 && /PLO5/.test(e.message))) throw e;
      // the drill deals PLO5 spots: switch the format (both tabs follow one game), again
      await postJSON("/study/format", { format: "plo5_double_bomb" });
      const sel = document.getElementById("format-select");
      if (sel) sel.value = "plo5_double_bomb";
      data = await postJSON("/trainer/drill/next", body);
    }
    await animateTrainerResponse(data, { ms: Math.min(trainerPrefs.animMs, DRILL_REPLAY_MS), hold: true });
    return true;
  } catch (e) {
    if (e && e.message === GATE_HANDLED) return false;
    if (e && e.status === 409 && /No mistakes/i.test(e.message)) {
      exitDrill(false);
      showToast(e.message, "info");
      if (!UI.lastState) fetchState();
      return false;
    }
    if (!UI.lastState) showLoadError(e.message);
    else showToast(e.message);
    return false;
  } finally {
    stopWorking();
    UI.actionInFlight = false;
    setActionsBusy(false);
  }
}

function exitDrill(deal) {
  UI.drillOffAt = (UI.lastState && UI.lastState.trainer && UI.lastState.trainer.hand_no) || 0;
  UI.drill = null;
  syncDrillUrl(false);
  renderDrill(UI.lastState);
  if (deal) dealTrainer("new_hand");
}

// The drill bar (in the workbar) and the deal buttons' words. A state holding a drill
// spot puts the page in the drill (a reload without ?drill=1, the Study tab and back).
function renderDrill(s) {
  const d = s && s.trainer ? s.trainer.drill : null;
  if (d && !UI.drill && s.trainer.hand_no > UI.drillOffAt) {
    UI.drill = { prioritize: d.prioritize !== false };
    syncDrillUrl(true);
  }
  const on = !!UI.drill && UI.mode === "trainer";
  document.body.classList.toggle("drill-mode", on);
  const nb = document.getElementById("trainer-new-hand-btn");
  const rb = document.getElementById("trainer-repeat-btn");
  if (nb) {
    nb.textContent = on ? "Next spot" : "New hand";
    nb.title = on ? "Deal the next of your mistakes (N)" : "Deal a new hand (N)";
  }
  if (rb) {
    rb.textContent = on ? "Try again" : "Repeat";
    rb.title = on ? "Play this spot again — practice: it doesn't change how often the spot comes up"
      : "Deal the same hand again (not counted in your stats)";
  }
  const prio = document.getElementById("drill-prioritize");
  if (prio) prio.checked = UI.drill ? !!UI.drill.prioritize : drillPrioritize();
  const where = document.getElementById("drill-where");
  if (!where) return;
  if (!on || !d) { where.textContent = ""; return; }
  const parts = [];
  if (d.pos && d.size) parts.push(`Spot ${d.pos} of ${d.size}`);
  if (d.played_at) parts.push(`your hand of ${String(d.played_at).slice(0, 10)}`);
  if (d.bb_cents) parts.push(`${fmtCents(d.sb_cents || d.bb_cents / 2)}/${fmtCents(d.bb_cents)}`);
  if (d.practice) parts.push("practice");
  where.textContent = parts.join(" · ");
}
function fmtCents(c) {
  const v = Number(c) / 100;
  return "$" + (Number.isInteger(v) ? String(v) : v.toFixed(2));
}

// The dock's line once a drill spot is played: how it went, the network's play (and how
// often it makes it), what you did in the hand. On a phone or a short window only the
// first two fit beside the button (`.dn-extra` hides: style.css) — the table never
// changes size with the dock.
function drillNoteHTML(s) {
  const t = s.trainer, d = t.drill, fb = t.feedback;
  if (!fb) return "";
  const v = verdictParts(s, fb);
  const kind = (d.result && d.result.outcome)
    || ({ best: "fixed", correct: "fixed", wrong: "missed", blunder: "missed" }[fb.category] || "close");
  const head = { fixed: "✓ Fixed", missed: "✗ Still a mistake", close: "~ Close" }[kind];
  const cls = { fixed: "best", missed: "blunder", close: "inaccuracy" }[kind];
  const p = d.probs ? d.probs[{ fold: "fold", check_call: "call", raise: "raise" }[fb.rec_gate]] : null;
  const netPart = v.best
    ? `The network: ${v.best}${typeof p === "number" ? ` (${Math.round(p * 100)}%)` : ""}`
    : "The network plays it the same way";
  const o = d.orig;
  const was = o && o.gate
    ? gateActionLabel(o.gate, o.chips, fb.to_call_chips, s, fb.actor_commit_chips, fb.street)
    : (o && o.label) || "";
  const extra = `${was ? ` · in the hand: ${was}` : ""}${d.practice ? " · practice" : ""}`;
  return `<b class="cat-text-${cls}">${head}</b> ${escapeHTML(netPart)}`
    + (extra ? `<span class="dn-extra">${escapeHTML(extra)}</span>` : "");
}

// --- My tables (2026-10-03) ------------------------------------------------------------
// One click in the work bar: the next hands are dealt like the tables in YOUR hand
// histories — players, the ante, your own stack and your opponents' stacks, each from its
// own distribution (the profile Hand review keeps of your latest hands; GET
// /trainer/my_tables says which). Without enough hands of yours: typical ClubGG tables.
// The setting lives in the Trainer settings ("Tables"); PLO5 only.
function myTablesFormat(s) {
  return !!s && s.format !== "nlh_single";
}
function renderMyTablesToggle(s) {
  const b = document.getElementById("trainer-mytables-btn");
  if (!b) return;
  const t = s && s.trainer ? s.trainer.settings : null;
  b.hidden = !t || !myTablesFormat(s) || !!UI.drill;
  const on = !!t && t.tables === "mine";
  b.setAttribute("aria-pressed", on ? "true" : "false");
  b.classList.toggle("on", on);
}
async function fetchMyTables() {
  if (!UI.myTables) UI.myTables = await getJSON("/trainer/my_tables");
  return UI.myTables;
}
const mtBB = (v) => `${fmtNum(v, v >= 100 ? 0 : 1)}bb`;
const mtPct = (p) => `${Math.round(p * 100)}%`;
function myTablesText(v) {
  const typical = "typical ClubGG tables (players and stacks from real $10/$20 games, a 3bb ante)";
  if (!v) return "Players, stacks and the ante like the tables in your own hand histories.";
  if (v.source === "mine") {
    // (one line each; #ts-tables-help keeps the line breaks)
    const seats = v.seats.slice().sort((a, b) => b[0] - a[0]).filter(([, p]) => p >= 0.005)
      .map(([n, p]) => `${n}: ${mtPct(p)}`).join(" · ");
    const stacks = (d) => {
      const top = (d.atoms || []).slice().sort((a, b) => b[1] - a[1])[0];
      return `usually ${mtBB(d.p10).slice(0, -2)}–${mtBB(d.p90)}, median ${mtBB(d.median)}`
        + (top ? `; exactly ${mtBB(top[0])} in ${mtPct(top[1])} of hands` : "");
    };
    const ante = v.antes.map(([a]) => mtBB(a)).join(" or ");
    return [
      `Dealt like your last ${v.hands.toLocaleString()} hands:`,
      `Players — ${seats}`,
      `Your stack — ${stacks(v.hero)}`,
      `Opponents — ${stacks(v.opponents)}`,
      `Ante ${ante}. Each seat is drawn on its own (no real table comes back); new uploads update it.`,
    ].join("\n");
  }
  if (!v.review) return `For now, ${typical}. Your own tables come from your hand histories in Hand review on the site.`;
  if (!v.paid) {
    return `For now, ${typical}. With a subscription, upload your ClubGG hand histories in Hand review `
      + "and My tables deals tables like yours: your stack and your opponents'.";
  }
  return `You have ${v.hands.toLocaleString()} hand${v.hands === 1 ? "" : "s"} in Hand review; My tables uses `
    + `your own tables from ${v.min_hands}. Until then, ${typical}.`;
}
async function fillMyTablesHelp() {
  const el = document.getElementById("ts-tables-help");
  if (!el) return;
  try {
    el.textContent = myTablesText(await fetchMyTables());
  } catch (_e) {
    el.textContent = myTablesText(null);
  }
}
async function toggleMyTables() {
  const s = UI.lastState;
  const cur = s && s.trainer ? s.trainer.settings : null;
  if (!cur || UI.actionInFlight) return;
  const on = cur.tables !== "mine";
  try {
    const data = await postJSON("/trainer/settings", { ...cur, tables: on ? "mine" : "custom" });
    await animateTrainerResponse(data);
    if (!on) {
      showToast("My tables off: players and stacks from your settings, from the next hand.", "info");
      return;
    }
    UI.myTables = null;
    let v = null;
    try { v = await fetchMyTables(); } catch (_e) { /* (the toast says it plainly) */ }
    showToast(v && v.source === "mine"
      ? `My tables on: from the next hand, tables like your last ${v.hands.toLocaleString()} hands.`
      : (v && v.review && v.paid
        ? `My tables on: typical ClubGG tables until Hand review has ${v.min_hands} of your hands (${v.hands} so far).`
        : "My tables on: typical ClubGG tables for now — Hand review reads your own (Settings says more)."), "info");
  } catch (e) {
    if (e.message !== GATE_HANDLED) showToast(e.message);
  }
}

function setupTrainerControls() {
  document.getElementById("tab-study").addEventListener("click", () => setMode("study"));
  document.getElementById("tab-trainer").addEventListener("click", () => setMode("trainer"));
  const newHand = () => (UI.drill ? dealDrill(false) : dealTrainer("new_hand"));
  const repeatHand = () => dealTrainer("repeat");
  document.getElementById("trainer-new-hand-btn").addEventListener("click", newHand);
  document.getElementById("trainer-repeat-btn").addEventListener("click", repeatHand);
  document.getElementById("review-next-hand").addEventListener("click", newHand);
  // the dock's "Next hand" at the end of a hand (app.play.js renderActions); in the
  // drill, "Next spot" and "Try again"
  document.getElementById("status-strip").addEventListener("click", (e) => {
    if (UI.actionInFlight) return;
    if (e.target.closest("[data-trainer-next]")) newHand();
    else if (e.target.closest("[data-drill-again]")) repeatHand();
  });
  const prio = document.getElementById("drill-prioritize");
  if (prio) prio.addEventListener("change", () => {
    lsSet(DRILL_PRIO_KEY, prio.checked ? "on" : "off");
    if (UI.drill) UI.drill.prioritize = prio.checked;
    showToast(prio.checked ? "Worst first: your worst mistakes come up most, the ones you've fixed less."
      : "Equal priority: every spot comes up as often as the others.", "info");
  });
  const myTables = document.getElementById("trainer-mytables-btn");
  if (myTables) myTables.addEventListener("click", toggleMyTables);
  document.getElementById("ts-tables").addEventListener("change", syncSettingsVisibility);
  const exit = document.getElementById("drill-exit");
  if (exit) exit.addEventListener("click", () => { if (!UI.actionInFlight) exitDrill(true); });
  document.getElementById("review-repeat-hand").addEventListener("click", repeatHand);
  document.getElementById("review-open-study").addEventListener("click", () => openReviewInStudy());
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
  document.getElementById("trainer-help-btn").addEventListener("click", () => showHelp("shortcuts"));
  document.getElementById("ts-cancel").addEventListener("click", closeTrainerSettings);
  document.getElementById("ts-form").addEventListener("submit", (e) => {
    e.preventDefault();   // saving is async and can fail: close only on success
    saveTrainerSettings();
  });
  for (const id of ["ts-seats-mode", "ts-stacks-mode", "ts-hero-mode"]) {
    document.getElementById(id).addEventListener("change", syncSettingsVisibility);
  }
  // Pause slider and number box mirror each other. The slider caps at
  // 4000ms; the box accepts up to 8000 for the patient.
  const animRange = document.getElementById("ts-anim-ms-range");
  const animNum = document.getElementById("ts-anim-ms");
  animRange.addEventListener("input", () => { animNum.value = animRange.value; });
  animNum.addEventListener("input", () => {
    const v = parseInt(animNum.value, 10);
    if (Number.isFinite(v)) animRange.value = String(Math.min(Math.max(v, 0), 4000));
  });
  // Help buttons anywhere on the page ("?" next to Accuracy / Recommendation,
  // links inside a help text).
  document.addEventListener("click", (e) => {
    const q = e.target.closest("[data-help]");
    if (q) { e.preventDefault(); showHelp(q.dataset.help); return; }
    const go = e.target.closest("[data-help-go]");
    if (go) { e.preventDefault(); showHelp(go.dataset.helpGo); }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && UI.trainerPick && !document.querySelector("dialog[open]")) {
      UI.selectedSlot = null;
      cancelTrainerPick();
      if (UI.lastState) render(UI.lastState);
      return;
    }
    onTrainerKey(e);
  });
}
