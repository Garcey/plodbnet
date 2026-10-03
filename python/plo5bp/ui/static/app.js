// Study / Trainer client, part 7 of 7 — the render pipeline, loading
// states and init(); then the local build's live-capture client, which the
// public build strips from this file.
//
// Plain scripts, no build step: index.html loads app.core.js, app.table.js,
// app.play.js, app.study.js, app.trainer.js, app.topbar.js and app.js in
// that order, and they share one global scope — any file may call a
// function from any other. Code that runs while a file LOADS may use only
// the files before it; everything else starts from init() in app.js.
"use strict";

// --- Rendering --------------------------------------------------------------
// render(state) repaints everything from the last server state. Each panel
// paints independently: one throwing sub-render (e.g. a payload
// shape a chart didn't expect) must not abort the rest and leave the table
// half-drawn or the action buttons missing (review 2026-09-20 F5).
const RENDER_STEPS = [
  renderTopBar, renderTable, renderHeroHandLabels, renderActorBanner,
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

// --- Working / loading states (ST-028) -----------------------------------------------
// While a deal or a graded move is being worked out the banner says so, and
// the table shows "Loading…" until the first state arrives (the static
// "Total Pot 0bb" and card prompt used to show even in the Trainer).
function setWorking(text) {
  UI.workingText = text || null;
  const banner = document.getElementById("actor-banner");
  if (!banner) return;
  if (text) {
    banner.hidden = false;
    banner.classList.remove("hero");
    banner.classList.add("working");
    banner.textContent = text;
  } else {
    banner.classList.remove("working");
    if (UI.lastState) renderActorBanner(UI.lastState);
  }
}

// A "working" line appears only if the request takes more than a moment,
// so fast answers don't flicker.
function startWorking(text, delay = 250) {
  stopWorking();
  UI.workingTimer = setTimeout(() => { UI.workingTimer = null; setWorking(text); }, delay);
}
function stopWorking() {
  if (UI.workingTimer) { clearTimeout(UI.workingTimer); UI.workingTimer = null; }
  if (UI.workingText) setWorking(null);
}

// (the table keeps what it showed, dimmed, until the new state is drawn over it)
function showLoadingState() {
  const rec = document.getElementById("recommendation");
  if (rec) rec.innerHTML = '<p class="muted">Loading the table…</p>';
  const wrap = document.getElementById("stage-wrap");
  if (wrap) wrap.classList.add("loading");
  const labels = document.getElementById("hero-hand-labels");
  if (labels) { labels.innerHTML = ""; labels.dataset.k = ""; }
  const lv = document.getElementById("last-verdict");
  if (lv) lv.hidden = true;
}

// The first state couldn't be loaded: say so on the table, with a Retry.
function showLoadError(message) {
  const banner = document.getElementById("actor-banner");
  if (!banner) return;
  banner.hidden = false;
  banner.classList.remove("hero", "working");
  banner.classList.add("error");
  banner.innerHTML = `<span>${escapeHTML(message || "Couldn't load the table.")}</span>`
    + '<button type="button" id="load-retry" class="banner-btn">Retry</button>';
  document.getElementById("load-retry").addEventListener("click", () => {
    banner.classList.remove("error");
    banner.textContent = "Loading…";
    fetchState();
  });
  const rec = document.getElementById("recommendation");
  if (rec) rec.innerHTML = '<p class="muted">Nothing to show until the table loads.</p>';
}

async function init() {
  // A signed-out public visitor gets the landing page only: the server strips
  // the app (and this script) from the page, so this never runs for them.
  if (!document.getElementById("top-bar")) return;
  // Safety net: never leave the workspace hidden if a setup step throws.
  setTimeout(() => document.body.classList.add("app-ready"), 2500);
  setupCardKeyboard();
  if (isPublicBuild()) {
    setupPublicUI();
    await refreshMe();
  }
  setupTopBar();
  setupDealerDrag();
  setupInsertHover();
  setupRaiseInput();
  setupTrainerControls();
  setupCardPicker();
  setupTableKeyboard();
  setupHistoryRewind();
  setupCurveReadout();
  // Stats collapse + reset: delegated (block innerHTML is replaced every render)
  const statsPanel = document.getElementById("trainer-stats-panel");
  statsPanel.addEventListener("click", (e) => {
    const reset = e.target.closest(".stats-reset");
    if (reset) { resetStats(reset.dataset.scope); return; }
    const recent = e.target.closest("[data-recent]");
    if (recent) { onRecentHandClick(recent); return; }
    const title = e.target.closest(".stats-title");
    if (!title) return;
    const block = title.closest(".stats-block");
    toggleStatsBlock(block && block.id === "stats-lifetime" ? "lifetime" : "session");
  });
  statsPanel.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && e.target.classList.contains("stats-title")) {
      e.preventDefault();
      const block = e.target.closest(".stats-block");
      toggleStatsBlock(block && block.id === "stats-lifetime" ? "lifetime" : "session");
    }
  });
  if (isPublicBuild()) {
    await handleCheckoutReturn();
    // Accounts without Study land in the trainer (Study is subscriber-only
    // when the paywall is on).
    if (!isEntitled() && UI.mode === "study") UI.mode = "trainer";
  }
  const guide = document.getElementById("study-guide");
  if (guide) {
    guide.hidden = lsGet(GUIDE_PREF_KEY) === "hidden";
    const btn = document.getElementById("study-help-btn");
    if (btn) btn.setAttribute("aria-expanded", guide.hidden ? "false" : "true");
  }
  applyModeUI();
  showLoadingState();
  // Until now the workspace is hidden (CSS), so the wrong tab's controls and
  // the desktop felt never flash while the script loads.
  document.body.classList.add("app-ready");
  initFormats();
  const spot = new URLSearchParams(location.search).get("spot");
  if (spot) await openSharedSpot(spot);
  else if (UI.mode === "trainer" && drillRequested()) await startDrill();  // (Hand review's mistakes drill)
  else await fetchState();
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

async function saveOcrFrame(asFixture) {
  const btn = document.getElementById("ocr-save-frame");
  if (btn) btn.disabled = true;
  try {
    const data = await postJSON("/ocr/save_frame", { to_fixtures: !!asFixture });
    const name = (data.path || "").split(/[\\/]/).pop() || "frame";
    const sz = data.frame_size ? ` (${data.frame_size.width}x${data.frame_size.height})` : "";
    const where = data.tracked ? " to the test fixtures" : "";
    showToast(`Saved ${name}${sz}${where}`, "info");
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
    showToast(`Action tracking toggle failed: ${e.message}`);
  } finally {
    UI.simpleOcrToggleBusy = false;
  }
}

// "Simple" mode = automatic action tracking OFF. The button names what the
// user actually gets (it used to read "Simple: On", which hid that no action
// was being tracked).
function setSimpleOcrToggleUI() {
  const btn = document.getElementById("ocr-simple-toggle");
  if (!btn) return;
  const tracking = !UI.simpleOcrMode;
  btn.textContent = tracking ? "Track actions: On" : "Track actions: Off";
  btn.setAttribute("aria-pressed", tracking ? "true" : "false");
  btn.classList.toggle("active", tracking);
  const grp = document.getElementById("ocr-rescan-group");
  if (grp) grp.style.display = UI.simpleOcrMode ? "" : "none";
}

function renderOcrStatus(st) {
  const el = document.getElementById("ocr-status");
  if (!st) { el.textContent = ""; clearOcrStatusError(); return; }
  if (st.running) {
    const title = st.window_title ? ` · ${st.window_title}` : "";
    const err = st.last_error ? ` · ${st.last_error}` : "";
    // Sync warnings (a refused action, the pot out of step) — red, like errors.
    const warn = (st.warnings || []).map((w) => ` · ⚠ ${w}`).join("");
    const mode = UI.simpleOcrMode ? "you enter actions · " : "tracking actions · ";
    el.textContent = `${mode}frames ${st.frames_seen} · events ${st.events_applied}${title}${err}${warn}`;
    if (st.last_error || warn) setOcrStatusError(el.textContent);
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
    "2. With THIS study server running on port 8765, open\n" +
    "   http://127.0.0.1:8765/pokernow/pokernow.user.js\n" +
    "   and click Install. Tampermonkey keeps it up to date\n" +
    "   from there by itself.\n" +
    "3. Open your PokerNow table. The badge in the page corner\n" +
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
  const warn = (st.warnings || []).map((w) => ` · ⚠ ${w}`).join("");
  if (st.last_error || warn) {
    if (st.last_error) el.textContent += ` · ${st.last_error}`;
    el.textContent += warn;
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
  document.getElementById("ocr-save-frame").addEventListener("click", (e) => saveOcrFrame(e.shiftKey));
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
