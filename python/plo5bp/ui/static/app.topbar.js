// Study / Trainer client, part 6 of 7 — the top bar: Study / Trainer tabs,
// the format picker, units and $ rate; and, in the public build, the
// account menu, sign-in / paywall notices and the maintenance heads-up.
//
// Plain scripts, no build step: index.html loads app.core.js, app.table.js,
// app.play.js, app.study.js, app.trainer.js, app.topbar.js and app.js in
// that order, and they share one global scope — any file may call a
// function from any other. Code that runs while a file LOADS may use only
// the files before it; everything else starts from init() in app.js.
"use strict";

// --- Top bar: format badges and the Study work bar ----------------------------

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
  return '<div class="rec-untrained">Placeholder model — '
    + `${name ? `${escapeHTML(name)} has` : "this format has"} no trained model on this server yet</div>`;
}

function renderTopBar(s) {
  if (UI.unit === "$") ensureRatePinned(s);
  syncUnitControl(s);
  const fmtSel = document.getElementById("format-select");
  if (fmtSel && s.format && fmtSel.value !== s.format
      && document.activeElement !== fmtSel) {
    fmtSel.value = s.format;
  }
  syncFormatStatic(s);
  if (s.trainer) return;
  document.getElementById("seats-count").textContent = String(s.num_seats);
  document.getElementById("seats-dec").disabled = s.num_seats <= 2;
  document.getElementById("seats-inc").disabled = s.num_seats >= 6;
  const anteLabel = document.getElementById("ante-label");
  if (anteLabel) anteLabel.innerHTML = `Ante <span class="tb-unit">(${escapeHTML(unitSuffix())})</span>`;
  const anteInput = document.getElementById("ante-input");
  anteInput.step = UI.unit === "bb" ? "0.5" : "0.01";
  if (document.activeElement !== anteInput) {
    anteInput.value = unitInputValue(s.chip_scale.ante_chips, s);
  }
  renderHeroSeatSelect(s);
}

// --- Units control (ST-010): a two-way "bb | $" switch; in $ a chip shows the
// rate and edits it. --------------------------------------------------------
function syncUnitControl(s) {
  for (const b of document.querySelectorAll("#unit-seg button")) {
    const on = b.dataset.unit === UI.unit;
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", on ? "true" : "false");
  }
  const rateBtn = document.getElementById("rate-btn");
  if (!rateBtn) return;
  rateBtn.hidden = UI.unit !== "$" || !s;
  if (s) {
    rateBtn.textContent = `${fmtAmountIn(dollarsPerBB(s), "$")} = 1bb`;
    rateBtn.setAttribute("aria-label", `One big blind is ${fmtAmountIn(dollarsPerBB(s), "$")}. Change the rate`);
  }
}

// Pin the rate the moment dollars are first shown for a format, so Study and
// Trainer convert with the same number from then on.
function ensureRatePinned(s) {
  if (!s || !s.format || validRate(UI.rates[s.format])) return;
  const srv = s.chip_scale ? s.chip_scale.dollars_per_bb : undefined;
  if (validRate(srv)) setRate(s.format, srv, false);
}

function setRate(format, rate, syncServer = true) {
  if (!format || !validRate(rate)) return;
  UI.rates[format] = rate;
  lsSet(RATE_PREF_KEY, JSON.stringify(UI.rates));
  // Local build: the Study session's rate also converts live-capture reads
  // (cents -> chips), so keep the server's copy equal to what the screen shows.
  if (syncServer) syncStudyRate();
}

// Local build only (there is no live capture in the public build).
function syncStudyRate() {
  if (isPublicBuild() || UI.rateSyncing) return;
  const s = UI.lastState;
  if (!s || s.trainer || UI.mode !== "study") return;
  const want = UI.rates[s.format];
  if (!validRate(want) || Math.abs(want - (s.chip_scale.dollars_per_bb || 0)) < 1e-9) return;
  UI.rateSyncing = true;
  postJSON("/study/config", { dollars_per_bb: want })
    .then((d) => applyState(d.state))
    .catch((e) => showToast(e.message))
    .finally(() => { UI.rateSyncing = false; });
}

function setUnit(unit) {
  if (unit !== "bb" && unit !== "$") return;
  UI.unit = unit;
  lsSet(UNIT_PREF_KEY, unit);
  const s = UI.lastState;
  if (unit === "$" && s) ensureRatePinned(s);
  UI.raiseUserSet = false;   // the raise box refills in the new unit
  if (s) { UI.lastStateKey = null; render(s); }
  else syncUnitControl(null);
}

function closeRatePop() {
  const pop = document.getElementById("rate-pop");
  if (pop) pop.remove();
  document.removeEventListener("pointerdown", onRatePopOutside, true);
}
function onRatePopOutside(e) {
  const pop = document.getElementById("rate-pop");
  const btn = document.getElementById("rate-btn");
  if (pop && !pop.contains(e.target) && !(btn && btn.contains(e.target))) closeRatePop();
}
function openRatePop() {
  if (document.getElementById("rate-pop")) { closeRatePop(); return; }
  const s = UI.lastState;
  const btn = document.getElementById("rate-btn");
  if (!s || !btn) return;
  const f = UI.formats ? UI.formats.find((x) => x.id === s.format) : null;
  const pop = document.createElement("div");
  pop.id = "rate-pop";
  pop.className = "preset-pop rate-pop";
  pop.setAttribute("role", "dialog");
  pop.setAttribute("aria-label", "Dollar value of one big blind");
  pop.innerHTML = `
    <label class="preset-pop-title" for="rate-input">One big blind is worth</label>
    <div class="preset-add-row">
      <span class="rate-cur">$</span>
      <input id="rate-input" type="number" min="0.01" step="0.01" inputmode="decimal"
             value="${escapeHTML(String(Math.round(dollarsPerBB(s) * 100) / 100))}" />
      <button id="rate-save" type="button">Save</button>
    </div>
    <div class="preset-hint muted" id="rate-hint">Used in Study and Trainer${f ? ` for ${escapeHTML(f.label)}` : ""}.</div>`;
  document.body.appendChild(pop);
  positionPresetPop(pop, btn);
  const input = document.getElementById("rate-input");
  const save = () => {
    const v = parseFloat(input.value);
    if (!(v > 0) || !isFinite(v) || v > 1e6) {
      const hint = document.getElementById("rate-hint");
      hint.textContent = "Enter an amount above $0.";
      hint.classList.add("error");
      input.focus();
      return;
    }
    setRate(s.format, Math.round(v * 100) / 100);
    closeRatePop();
    UI.raiseUserSet = false;
    if (UI.lastState) { UI.lastStateKey = null; render(UI.lastState); }
    btn.focus();
  };
  document.getElementById("rate-save").addEventListener("click", save);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); save(); }
    else if (e.key === "Escape") { e.preventDefault(); closeRatePop(); btn.focus(); }
  });
  document.addEventListener("pointerdown", onRatePopOutside, true);
  input.focus();
  input.select();
}

// --- Format picker (ST-008 / ST-027) --------------------------------------------
// Only the formats this account can play are offered; with a single one the
// picker becomes plain text (nothing to choose, and no internal test slots
// or "coming soon" rows shown to the public).
function syncFormatStatic(s) {
  const wrap = document.getElementById("format-wrap");
  const sel = document.getElementById("format-select");
  const stat = document.getElementById("format-static");
  if (!wrap || !sel || !stat || !UI.formats) return;
  const usable = UI.formats.filter((f) => !f.locked);
  const single = usable.length <= 1;
  sel.hidden = single;
  stat.hidden = !single;
  document.body.classList.toggle("multi-format", !single);
  const cur = UI.formats.find((f) => f.id === (s && s.format)) || usable[0];
  if (single && cur) {
    stat.textContent = cur.label;
    stat.title = UI.formats.length > usable.length ? "More formats are on the way" : "";
  }
  const label = wrap.querySelector("label");
  if (label) label.hidden = single;
}

// --- Top-bar handlers -------------------------------------------------------

function setupTopBar() {
  document.getElementById("seats-dec").addEventListener("click", async () => {
    const s = UI.lastState;
    if (!s || s.num_seats <= 2) return;
    if (!(await confirmHandReset("Remove a player"))) return;
    postSeats({ num_seats: s.num_seats - 1 });
  });
  document.getElementById("seats-inc").addEventListener("click", async () => {
    const s = UI.lastState;
    if (!s || s.num_seats >= 6) return;
    if (!(await confirmHandReset("Add a player"))) return;
    postSeats({ num_seats: s.num_seats + 1 });
  });
  for (const b of document.querySelectorAll("#unit-seg button")) {
    b.addEventListener("click", () => setUnit(b.dataset.unit));
  }
  document.getElementById("rate-btn").addEventListener("click", openRatePop);
  const heroSel = document.getElementById("hero-pos-select");
  heroSel.addEventListener("change", async () => {
    const s = UI.lastState;
    if (!s || s.trainer) return;
    const button = parseInt(heroSel.value, 10);
    if (!Number.isFinite(button) || button === s.button_seat) return;
    if (!(await confirmHandReset("Change your seat"))) {
      heroSel.value = String(s.button_seat);
      return;
    }
    postSeats({ button_seat: button });
  });
  const anteInput = document.getElementById("ante-input");
  anteInput.addEventListener("change", async () => {
    const s = UI.lastState;
    if (!s) return;
    const chips = parseToChips(anteInput.value, s);
    if (chips === null || chips === s.chip_scale.ante_chips) {
      anteInput.value = unitInputValue(s.chip_scale.ante_chips, s);
      return;
    }
    if (!(await confirmHandReset("Change the ante"))) {
      anteInput.value = unitInputValue(s.chip_scale.ante_chips, s);
      return;
    }
    postConfig({ ante_chips: chips });
  });
  document.getElementById("format-select").addEventListener("change", async (e) => {
    const sel = e.target;
    if (sel.closest("#acct-menu")) toggleAccountMenu(false, false);   // phones: picked from the menu
    const prev = UI.lastState && UI.lastState.format ? UI.lastState.format : null;
    if (UI.mode === "study" && !(await confirmHandReset("Switch format", { clearsCards: true }))) {
      if (prev) sel.value = prev;
      return;
    }
    // New context (review 2026-09-20 F4/F15): responses still in flight for
    // the old format go stale, any frame playback stops, and the old format's
    // action controls / typed raise must not stay live while /format loads.
    UI.ctxSeq++;
    UI.animSeq++;
    UI.lastStateKey = null;  // the screen was just blanked: force the next paint
    resetRaiseEntry();
    clearActionControls();
    try {
      const data = await postJSON("/study/format", { format: sel.value });
      document.dispatchEvent(new CustomEvent("wg:format", { detail: { format: sel.value } }));
      // The server switches BOTH tabs' sessions — drop per-hand client
      // cursors, then show the fresh state for whichever mode is active.
      UI.selectedSlot = null;
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
  document.getElementById("new-hand-btn").addEventListener("click", async () => {
    if (!(await confirmHandReset("Start a new hand", { clearsCards: true }))) return;
    postReset();
  });
  document.getElementById("study-help-btn").addEventListener("click", () => toggleStudyGuide());
  document.getElementById("study-guide-close").addEventListener("click", () => toggleStudyGuide(false));
  document.getElementById("share-spot-btn").addEventListener("click", () => shareSpot());
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
  // Formats this account can't play (POST /format would 403) aren't offered
  // at all — the public used to see internal test slots as "coming soon!"
  // rows (ST-027); syncFormatStatic says more are on the way instead.
  for (const f of UI.formats) {
    if (f.locked) continue;
    const opt = document.createElement("option");
    opt.value = f.id;
    opt.textContent = f.model_loaded ? f.label : `${f.label} (untrained)`;
    if (!f.model_loaded) opt.className = "opt-untrained";
    sel.appendChild(opt);
  }
  const active = (UI.lastState && UI.lastState.format) || data.active;
  if (active) sel.value = active;
  syncFormatStatic(UI.lastState || { format: active });
  // Other scripts (ranges.js, local build) follow the formats list and the
  // active format from these events instead of polling the dropdown.
  document.dispatchEvent(new CustomEvent("wg:formats", { detail: { formats: UI.formats, active } }));
  // The raise-preset row derives its cap class (pot-limit vs no-limit) from
  // this payload — refresh a state rendered before it arrived.
  if (UI.lastState) render(UI.lastState);
}

// --- Public build: account, sign-in gate, paywall -------------------------

// /me for the account menu. A failed call is NOT "signed out" (ACC-026): the
// server already decided that when it sent this page (a signed-out visitor
// gets the landing page, without the app), and a 502 while a deploy restarts
// used to bounce signed-in players to a sign-in page for nothing. Retries a
// few times; a real `signed_in: false` means the session ended.
async function refreshMe(attempt = 0) {
  try {
    const me = await getJSON("/me");
    UI.me = me;
    syncMaintenanceBanner(me);
    startMePoll();
    if (me && me.signed_in === false) showSessionExpired();
  } catch (e) {
    if (e && e.message === GATE_HANDLED) return UI.me;
    if (attempt < 4) {
      setTimeout(() => refreshMe(attempt + 1), 1500 * (attempt + 1));
    }
  }
  renderAccountChip();
  return UI.me;
}

// --- Maintenance heads-up (FEAT-028) ------------------------------------------------
// The admin announces a restart from /admin; /me carries {message, at} and
// this thin bar shows it (the home-game tables read the same /me field).
// /me is re-read every 2 minutes while the tab is visible so a notice set
// after the page loaded still arrives in time.
let ME_POLL_TIMER = null;
function startMePoll() {
  if (ME_POLL_TIMER || !isPublicBuild()) return;
  ME_POLL_TIMER = setInterval(() => { if (!document.hidden) refreshMe(); }, 120000);
}
function syncMaintenanceBanner(me) {
  const m = me && me.maintenance;
  let bar = document.getElementById("maint-banner");
  if (!m || !m.message) {
    if (bar) bar.remove();
    return;
  }
  if (!bar) {
    bar = document.createElement("div");
    bar.id = "maint-banner";
    bar.setAttribute("role", "status");
    document.body.prepend(bar);
  }
  let when = "";
  if (m.at) {
    const t = new Date(m.at);
    if (!isNaN(t)) {
      when = ` · ${t.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}`;
    }
  }
  bar.textContent = `${m.message}${when}`;
}

function isEntitled() {
  return !isPublicBuild() || !UI.me || !!(UI.me.sub && UI.me.sub.active);
}

// --- Account menu (ACC-015 / ACC-020 / ST-011 / ST-007 / ACC-027 / FEAT-021) --
// One avatar button opens a small menu: who you are, your plan, Home games,
// Admin (admins), help and legal links, Sign out. The email no longer sits on
// screen all the time (players stream and share screenshots), and the bar no
// longer runs off the side of a laptop screen with "Sign out" hidden behind
// an invisible scrollbar.

// Admin-only "safe to deploy?" signal: distinct non-admin users who made an
// authenticated request inside the server's activity window (default 5 min).
// A count badge on the avatar + a line in the menu (was two top-bar chips).
const ACTIVE_POLL_MS = 10000;
let ACTIVE_POLL_TIMER = null;
const SUPPORT_EMAIL = "support@wrapgto.com";

async function refreshActiveCount() {
  if (document.hidden) return; // resync on visibilitychange instead
  try {
    UI.active = await getJSON("/admin/api/active");
  } catch (_) {
    UI.active = { error: true };
  }
  syncActiveBadge();
}

function syncActiveBadge() {
  const d = UI.active;
  if (!d) return;
  const n = d.error ? 0 : (Number(d.active_users) || 0);
  const badge = document.getElementById("acct-active-badge");
  if (badge) {
    badge.hidden = !n;
    badge.textContent = n ? String(n) : "";
  }
  const line = document.getElementById("acct-active-line");
  if (!line) return;
  if (d.error) {
    line.textContent = "Active players: unavailable";
    line.title = "";
    line.classList.remove("busy");
    return;
  }
  const mins = Math.max(1, Math.round((Number(d.window_seconds) || 300) / 60));
  line.textContent = n
    ? `${n} player${n > 1 ? "s" : ""} active in the last ${mins} min`
    : `Nobody active in the last ${mins} min — safe to deploy`;
  const who = Array.isArray(d.emails) ? d.emails.slice() : [];
  const admins = (Number(d.active_total) || 0) - n;
  if (admins > 0) who.push(`+${admins} admin${admins > 1 ? "s" : ""} online (not counted)`);
  line.title = who.join("\n");
  line.classList.toggle("busy", n > 0);
}

function startActivePoll() {
  refreshActiveCount();
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

// Same colour-from-name and initials as the home-games avatars, so a player
// looks the same on both sides of the site.
function hueOf(key) {
  let h = 0;
  const str = String(key);
  for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) >>> 0;
  return (h * 47) % 360;
}
function initialsOf(name) {
  const parts = String(name || "?").trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "?";
  const a = parts[0][0] || "?";
  const b = parts.length > 1 ? parts[parts.length - 1][0] : (parts[0][1] || "");
  return (a + b).toUpperCase();
}
function avatarHTML(name, pic, cls) {
  // /me fields are account data (OAuth profile): escape before innerHTML, and
  // only ever load an https avatar (review 2026-09-20 F17).
  const img = pic
    ? `<img src="${escapeHTML(pic)}" alt="" referrerpolicy="no-referrer" loading="lazy" />`
    : "";
  return `<span class="av ${cls || ""}" style="--h:${hueOf(name)}" aria-hidden="true">`
    + `${escapeHTML(initialsOf(name))}${img}</span>`;
}

function fmtPrice(cents) {
  const n = Number(cents);
  return isFinite(n) && n > 0 ? fmtAmountIn(n / 100, "$") : "";
}

// The plan line under the name. Follows /me — never hard-coded (ACC-017).
function planLine(me) {
  if (me.is_admin) return { text: "Admin", cls: "admin" };
  if (me.free_for_all) return { text: "Free while the models are in development", cls: "free" };
  if (me.sub && me.sub.active) {
    return { text: me.sub.source === "comp" ? "Complimentary access" : "Subscribed", cls: "pro" };
  }
  const f = me.free || {};
  const left = Number(f.left) || 0;
  return {
    text: `${left} of ${Number(f.limit) || 0} free trainer hands left today`,
    cls: left > 0 ? "quota" : "quota empty",
  };
}

function renderAccountChip() {
  const el = document.getElementById("account-chip");
  if (!el) return;
  parkFormatPicker();   // the menu is rebuilt below: bring the picker home first
  const me = UI.me;
  if (!isPublicBuild() || !me || !me.signed_in) {
    el.hidden = true;
    el.innerHTML = "";
    stopActivePoll();
    syncHomeGamesTab(null);
    syncReviewTab(null);
    return;
  }
  const wasOpen = !!document.querySelector("#acct-menu:not([hidden])");
  el.hidden = false;
  const email = String(me.email || "");
  const name = String(me.name || "").trim() || email.split("@")[0] || "You";
  const pic = typeof me.picture === "string" && /^https:\/\//i.test(me.picture) ? me.picture : "";
  const plan = planLine(me);
  const games = homeGamesLink(me);
  const price = fmtPrice(me.price_cents);
  const items = [];
  if (!me.sub.active && !me.free_for_all) {
    items.push(`<button type="button" role="menuitem" class="acct-item accent" data-act="upgrade">`
      + `Upgrade${price ? ` — ${escapeHTML(price)} a month` : ""}</button>`);
  }
  if (me.sub.active && me.sub.source === "stripe") {
    items.push('<button type="button" role="menuitem" class="acct-item" data-act="billing">Billing</button>');
  }
  if (games) {
    items.push(`<a role="menuitem" class="acct-item" href="${escapeHTML(games.href)}">${escapeHTML(games.label)}</a>`);
  }
  if (me.is_admin) {
    items.push('<a role="menuitem" class="acct-item" href="/admin" target="_blank" rel="noopener">Admin</a>'
      + '<div class="acct-active" id="acct-active-line">Active players: …</div>');
  }
  // Self-service account actions, offered when the server has them (/me.account).
  const acc = me.account && typeof me.account === "object" ? me.account : {};
  const same = (u) => typeof u === "string" && /^\/(?!\/)/.test(u);
  if (same(acc.export) || same(acc.delete) || same(acc.signout_everywhere)) {
    items.push('<hr class="acct-sep" />');
    if (same(acc.export)) {
      items.push(`<a role="menuitem" class="acct-item" href="${escapeHTML(acc.export)}" download>Download my data</a>`);
    }
    if (same(acc.signout_everywhere)) {
      items.push('<button type="button" role="menuitem" class="acct-item" data-act="signout-all">Sign out other devices</button>');
    }
    if (same(acc.delete)) {
      items.push('<button type="button" role="menuitem" class="acct-item danger" data-act="delete">Delete account…</button>');
    }
  }
  items.push('<hr class="acct-sep" />');
  items.push(`<a role="menuitem" class="acct-item" href="mailto:${SUPPORT_EMAIL}">Contact support</a>`);
  items.push('<div class="acct-legal"><a role="menuitem" class="acct-item sm" href="/terms">Terms</a>'
    + '<a role="menuitem" class="acct-item sm" href="/privacy">Privacy</a></div>');
  items.push('<hr class="acct-sep" />');
  items.push('<button type="button" role="menuitem" class="acct-item" data-act="signout">Sign out</button>');
  el.innerHTML = `
    <button id="acct-btn" class="acct-trigger" type="button" aria-haspopup="menu"
            aria-expanded="false" aria-controls="acct-menu" aria-label="Account: ${escapeHTML(name)}">
      ${avatarHTML(name, pic, "sm")}
      <span class="acct-active-badge" id="acct-active-badge" hidden></span>
      <svg class="acct-chev" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 4.5 6 7.5 9 4.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
    </button>
    <div id="acct-menu" class="acct-menu" role="menu" aria-label="Account" hidden>
      <div class="acct-head">
        ${avatarHTML(name, pic, "")}
        <div class="acct-who">
          <div class="acct-name">${escapeHTML(name)}</div>
          <div class="acct-email" title="${escapeHTML(email)}">${escapeHTML(email)}</div>
        </div>
      </div>
      <div class="acct-plan plan-${plan.cls}">${escapeHTML(plan.text)}</div>
      <div class="acct-settings" id="acct-settings"></div>
      ${items.join("")}
    </div>`;
  const btn = document.getElementById("acct-btn");
  const menu = document.getElementById("acct-menu");
  btn.addEventListener("click", () => toggleAccountMenu(!menu.hidden ? false : true));
  menu.addEventListener("click", (e) => {
    const it = e.target.closest("[data-act]");
    if (!it) return;
    const act = it.dataset.act;
    toggleAccountMenu(false);
    if (act === "upgrade") startCheckout();
    else if (act === "billing") openBillingPortal();
    else if (act === "signout") signOut();
    else if (act === "signout-all") signOutOtherDevices();
    else if (act === "delete") openDeleteAccount();
  });
  menu.addEventListener("keydown", onAccountMenuKey);
  for (const img of el.querySelectorAll(".av > img")) {
    img.addEventListener("error", () => img.remove());   // initials stay underneath
  }
  if (me.is_admin) { startActivePoll(); syncActiveBadge(); }
  else stopActivePoll();
  syncHomeGamesTab(me);
  syncReviewTab(me);
  placeFormatPicker();
  if (wasOpen) toggleAccountMenu(true, false);
}

// Phones held upright: the top bar is one row that must fit a 360px screen —
// brand, Study, Trainer, Games, the bb/$ switch and the avatar. The format
// picker (rarely changed) moves into the account menu there; moving the node
// keeps its listeners and what syncFormatStatic() set on it.
const PHONE_BAR_MQ = window.matchMedia("(max-width: 560px)");

function parkFormatPicker() {
  const wrap = document.getElementById("format-wrap");
  const bar = document.getElementById("top-bar");
  if (!wrap || !bar || wrap.parentElement === bar) return;
  bar.insertBefore(wrap, bar.querySelector(".unit-ctl"));
}

function placeFormatPicker() {
  const slot = document.getElementById("acct-settings");
  const wrap = document.getElementById("format-wrap");
  if (PHONE_BAR_MQ.matches && slot && wrap) {
    if (wrap.parentElement !== slot) slot.appendChild(wrap);
  } else {
    parkFormatPicker();
  }
}
PHONE_BAR_MQ.addEventListener("change", placeFormatPicker);

function accountMenuItems() {
  const menu = document.getElementById("acct-menu");
  return menu ? [...menu.querySelectorAll('[role="menuitem"]')] : [];
}

function toggleAccountMenu(open, focusFirst = true) {
  const btn = document.getElementById("acct-btn");
  const menu = document.getElementById("acct-menu");
  if (!btn || !menu) return;
  menu.hidden = !open;
  btn.setAttribute("aria-expanded", open ? "true" : "false");
  if (open) {
    document.addEventListener("pointerdown", onAccountMenuOutside, true);
    if (focusFirst) {
      const first = accountMenuItems()[0];
      if (first) first.focus();
    }
    if (UI.me && UI.me.is_admin) refreshActiveCount();
  } else {
    document.removeEventListener("pointerdown", onAccountMenuOutside, true);
  }
}

function onAccountMenuOutside(e) {
  const chip = document.getElementById("account-chip");
  if (chip && chip.contains(e.target)) return;
  toggleAccountMenu(false);
}

function onAccountMenuKey(e) {
  const items = accountMenuItems();
  const i = items.indexOf(document.activeElement);
  if (e.key === "Escape") {
    e.preventDefault();
    toggleAccountMenu(false);
    const btn = document.getElementById("acct-btn");
    if (btn) btn.focus();
  } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    if (!items.length) return;
    const step = e.key === "ArrowDown" ? 1 : -1;
    items[(i + step + items.length) % items.length].focus();
  } else if (e.key === "Home" || e.key === "End") {
    e.preventDefault();
    if (items.length) items[e.key === "Home" ? 0 : items.length - 1].focus();
  } else if (e.key === "Tab") {
    toggleAccountMenu(false);
  }
}

async function signOutOtherDevices() {
  const url = UI.me && UI.me.account && UI.me.account.signout_everywhere;
  if (!url) return;
  try {
    await postJSON(url, {});
    showToast("Signed out on every other device. This one stays signed in.", "info");
  } catch (e) { showToast(e.message); }
}

function openDeleteAccount() {
  const dlg = document.getElementById("delete-dlg");
  if (!dlg || !UI.me || !UI.me.account) return;
  const input = document.getElementById("delete-confirm");
  const err = document.getElementById("delete-error");
  input.value = "";
  input.placeholder = UI.me.email || "";
  err.hidden = true;
  // Offer the export first, when the server has one (same-origin path only).
  const exp = UI.me.account.export;
  const expLine = document.getElementById("delete-export-line");
  if (expLine) {
    const ok = typeof exp === "string" && /^\/(?!\/)/.test(exp);
    expLine.hidden = !ok;
    if (ok) document.getElementById("delete-export").setAttribute("href", exp);
  }
  if (!dlg._wiredDelete) {
    dlg._wiredDelete = true;
    document.getElementById("delete-cancel").addEventListener("click", () => closeDialog(dlg));
    document.getElementById("delete-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const url = UI.me && UI.me.account && UI.me.account.delete;
      const typed = input.value.trim();
      if (!typed || typed.toLowerCase() !== String(UI.me.email || "").toLowerCase()) {
        err.textContent = "That isn't the email address on this account.";
        err.hidden = false;
        input.focus();
        return;
      }
      const go = document.getElementById("delete-go");
      go.disabled = true;
      try {
        await postJSON(url, { confirm: typed });
        window.location.href = "/";
      } catch (ex) {
        err.textContent = ex.message;
        err.hidden = false;
      } finally {
        go.disabled = false;
      }
    });
  }
  openDialog(dlg);
  input.focus();
}

async function openBillingPortal() {
  try { const d = await postJSON("/billing/portal"); window.location.href = d.url; }
  catch (e) { showToast(e.message); }
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

// /me carries {href, label} for the Home games page (every signed-in user
// since clubs, 2026-09-25). The tab is a real link — middle-click opens it
// in a new tab — with a small arrow saying it leaves the Study/Trainer
// workspace. Only a same-origin absolute path is accepted as a destination.
function homeGamesLink(me) {
  const hg = me && me.homegame;
  if (!hg || typeof hg !== "object") return null;
  const href = typeof hg.href === "string" ? hg.href : "";
  if (!/^\/(?!\/)/.test(href) || href.includes("\\")) return null;
  const label = typeof hg.label === "string" && hg.label.trim() ? hg.label.trim() : "Home games";
  return { href, label };
}

function syncHomeGamesTab(me) {
  const tabs = document.getElementById("mode-tabs");
  if (!tabs) return;
  let tab = document.getElementById("tab-games");
  const link = homeGamesLink(me);
  if (!link) {
    if (tab) tab.remove();
    return;
  }
  if (!tab) {
    tab = document.createElement("a");
    tab.id = "tab-games";
    tab.className = "mode-tab mode-link";
    tabs.appendChild(tab);
  }
  tab.href = link.href;
  // Phones show the short name ("Games") so the top bar stays one row.
  const short = /^home games$/i.test(link.label) ? "Games" : link.label;
  tab.innerHTML = `<span class="tab-long">${escapeHTML(link.label)}</span>`
    + `<span class="tab-short">${escapeHTML(short)}</span><svg class="ext-arrow" viewBox="0 0 12 12" aria-hidden="true">`
    + '<path d="M4 3h5v5M9 3 3.5 8.5" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
}

// Hand review (2026-10-03): /me carries {href, label} for the paid page of uploaded
// hand histories (beside Home games, a real link like it).
function syncReviewTab(me) {
  const tabs = document.getElementById("mode-tabs");
  if (!tabs) return;
  let tab = document.getElementById("tab-review");
  const info = me && me.review;
  const href = info && typeof info.href === "string" ? info.href : "";
  if (!/^\/(?!\/)/.test(href) || href.includes("\\")) {
    if (tab) tab.remove();
    return;
  }
  const label = typeof info.label === "string" && info.label.trim() ? info.label.trim() : "Hand review";
  if (!tab) {
    tab = document.createElement("a");
    tab.id = "tab-review";
    tab.className = "mode-tab mode-link";
    tabs.appendChild(tab);
  }
  tab.href = href;
  tab.innerHTML = `<span class="tab-long">${escapeHTML(label)}</span>`
    + `<span class="tab-short">Review</span><svg class="ext-arrow" viewBox="0 0 12 12" aria-hidden="true">`
    + '<path d="M4 3h5v5M9 3 3.5 8.5" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
}

// The session ended while the page was open (a 401 from any route, or /me
// saying signed-out): a small dialog, not the whole marketing page over the
// table (ACC-026). Signing in comes back to this tab.
function showSessionExpired() {
  const dlg = document.getElementById("session-dlg");
  if (!dlg || dlg.open) return;
  const link = document.getElementById("session-signin");
  if (link) {
    const here = location.pathname + (UI.mode ? `?mode=${UI.mode}` : "");
    link.href = `/auth/login?next=${encodeURIComponent(here)}`;
  }
  stopActivePoll();
  openDialog(dlg);
}

function showPaywall(body) {
  const modal = document.getElementById("paywall-modal");
  if (!modal) return;
  const me = UI.me || {};
  const title = document.getElementById("paywall-title");
  const msg = document.getElementById("paywall-msg");
  const sub = document.getElementById("paywall-subscribe-btn");
  const fine = document.getElementById("paywall-fine");
  // Price and daily limit come from the server (/me, the 402 body) — they
  // are settings, not constants (ACC-021).
  const price = fmtPrice(me.price_cents);
  const limit = Number((body && body.limit) ?? (me.free && me.free.limit));
  const hands = isFinite(limit) && limit > 0 ? `${limit} free trainer hand${limit === 1 ? "" : "s"}` : "your free trainer hands";
  if (body && body.error === "free_limit") {
    title.textContent = "That's today's free hands";
    const resets = body.resets_at ? new Date(body.resets_at) : null;
    const inH = resets ? Math.max(1, Math.round((resets - Date.now()) / 3600000)) : null;
    msg.textContent = `You've played ${hands} for today`
      + (inH ? ` — more in about ${inH} hour${inH === 1 ? "" : "s"}.` : ".")
      + " Subscribe for unlimited hands and full Study mode.";
  } else {
    title.textContent = "Study is part of the subscription";
    msg.textContent = (body && typeof body.detail === "string" && body.detail)
      || `Subscribe for full Study mode and unlimited trainer hands. The trainer stays free for ${hands} a day.`;
  }
  sub.textContent = price ? `Subscribe — ${price} a month` : "Subscribe";
  sub.hidden = me.billing_configured === false;
  if (fine) fine.hidden = !price;
  openDialog(modal);
}

async function startCheckout() {
  try {
    const d = await postJSON("/billing/checkout");
    window.location.href = d.url;
  } catch (e) {
    showToast(e.message, e.status === 409 ? "info" : "error");
  }
}

function setupPublicUI() {
  const sub = document.getElementById("paywall-subscribe-btn");
  if (sub) sub.addEventListener("click", () => startCheckout());
  const close = document.getElementById("paywall-close-btn");
  if (close) close.addEventListener("click", () => closeDialog(document.getElementById("paywall-modal")));
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

// --- Study / Trainer tabs -------------------------------------------------

function applyModeUI() {
  const trainer = UI.mode === "trainer";
  document.body.classList.toggle("trainer-mode", trainer);
  if (!trainer) document.body.classList.remove("hands-open");  // (the Trainer's previous hands)
  for (const [id, on] of [["tab-study", !trainer], ["tab-trainer", trainer]]) {
    const tab = document.getElementById(id);
    tab.classList.toggle("active", on);
    tab.setAttribute("aria-pressed", on ? "true" : "false");
  }
  const ranges = document.getElementById("tab-ranges");
  if (ranges) ranges.setAttribute("aria-pressed", ranges.classList.contains("active") ? "true" : "false");
  syncModeInUrl();
}

// The address bar names the open tab (ST-029), so a bookmark or a shared
// link opens what was on screen. replaceState: switching tabs doesn't add
// Back-button stops.
function syncModeInUrl() {
  try {
    const url = new URL(location.href);
    if (url.searchParams.get("mode") === UI.mode) return;
    url.searchParams.set("mode", UI.mode);
    history.replaceState(history.state, "", url.pathname + url.search + url.hash);
  } catch (_) { /* old browser: the tab still works */ }
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
  if (mode === "study" && isPublicBuild() && UI.me && UI.me.signed_in && !isEntitled()) {
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
  UI.reviewNode = null;
  cancelTrainerPick();
  resetRaiseEntry();
  clearActionControls();
  const reviewPanel = document.getElementById("review-panel");
  if (reviewPanel) reviewPanel.hidden = true;
  hideFeedbackFlash();
  applyModeUI();
  showLoadingState();
  fetchState();
}
