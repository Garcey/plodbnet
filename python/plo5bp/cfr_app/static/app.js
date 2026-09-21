/* CFR Solver desktop — Solve / Strategy / Library */
(() => {
  "use strict";

  const RANKS = "23456789TJQKA";
  const SUITS = ["c", "d", "h", "s"];
  const SUIT_SYM = { c: "♣", d: "♦", h: "♥", s: "♠" };
  const STREET_NEED = { 0: 0, 1: 3, 2: 4, 3: 5 };
  const STREET_NAME = { 0: "Preflop", 1: "Flop", 2: "Turn", 3: "River" };
  const RANK_DISP = "AKQJT98765432";

  const state = {
    meta: null,
    board: [],
    activeSlot: 0,
    pollTimer: null,
    liveViewTimer: null,
    currentJobId: null,
    viewSource: null,
    viewData: null,
    viewMode: "matrix",
    pageOffset: 0,
    pageLimit: 150,
    selectedNodePath: null,
    selectedSeat: null,
    selectedClassId: null,
    selectedHandMix: null,
    lineNav: null,
    chartPack: null,
    liveView: false,
    liveViewSeq: 0,
    loadedPath: null,
    rangeWhich: "oop",
    rangeInfo: { oop: null, ip: null }, // last /api/range/parse reply per side
    rangeSeq: { oop: 0, ip: 0 },
    rangeTimers: { oop: null, ip: null },
    libItems: [],
  };

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  function toast(msg, kind = "") {
    const el = $("#toast");
    el.textContent = msg;
    el.className = "toast" + (kind ? " " + kind : "");
    clearTimeout(el._t);
    el._t = setTimeout(() => el.classList.add("hidden"), 3800);
  }

  // (review 2026-09-20 local API) The server only accepts state-changing requests
  // that are JSON or carry the per-launch token (injected into index.html, which
  // a cross-origin page cannot read) — so a random web page can't drive the app.
  function authHeaders() {
    return window.CFR_TOKEN ? { "X-CFR-Token": window.CFR_TOKEN } : {};
  }

  async function api(path, opts = {}) {
    const res = await fetch(path, {
      ...opts,
      headers: { "Content-Type": "application/json", ...authHeaders(), ...(opts.headers || {}) },
    });
    let body = null;
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) body = await res.json();
    else body = await res.text();
    if (!res.ok) {
      const detail = (body && body.detail) || body || res.statusText;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return body;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function cardLabel(c) {
    if (c == null || c < 0) return "";
    return RANKS[(c / 4) | 0] + SUIT_SYM[SUITS[c % 4]];
  }

  function cardIsRed(c) {
    const s = SUITS[c % 4];
    return s === "d" || s === "h";
  }

  // ---------- tabs ----------
  function initTabs() {
    $$("#tabs .tab").forEach((btn) => {
      btn.addEventListener("click", () => switchTab(btn.dataset.panel));
    });
  }

  function switchTab(name) {
    $$("#tabs .tab").forEach((b) => b.classList.toggle("active", b.dataset.panel === name));
    $$(".panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${name}`));
    if (name === "library") loadLibrary();
  }

  // ---------- board picker ----------
  function boardNeed() {
    return STREET_NEED[parseInt($("#f-street").value, 10)] || 0;
  }

  function ensureBoardLen() {
    const n = boardNeed();
    while (state.board.length < n) state.board.push(null);
    state.board = state.board.slice(0, n);
    if (state.activeSlot >= n) state.activeSlot = Math.max(0, n - 1);
  }

  function renderBoardSlots() {
    ensureBoardLen();
    // Board cards block combos, so what a range text means depends on the board
    // (incl. "no board" preflop). Debounced: a burst of clicks = one request.
    scheduleRangeRefresh("oop");
    scheduleRangeRefresh("ip");
    const wrap = $("#board-slots");
    wrap.innerHTML = "";
    const n = boardNeed();
    if (n === 0) {
      wrap.innerHTML = '<span class="muted">No board (preflop)</span>';
      $("#card-picker").innerHTML = "";
      return;
    }
    for (let i = 0; i < n; i++) {
      const slot = document.createElement("div");
      slot.className =
        "slot" +
        (state.board[i] != null ? " filled" : "") +
        (state.activeSlot === i ? " active" : "");
      const c = state.board[i];
      if (c != null) {
        const span = document.createElement("span");
        span.className = "suit-" + SUITS[c % 4];
        span.textContent = cardLabel(c);
        slot.appendChild(span);
      } else {
        slot.textContent = "·";
      }
      slot.addEventListener("click", () => {
        state.activeSlot = i;
        renderBoardSlots();
      });
      wrap.appendChild(slot);
    }
    renderCardPicker();
  }

  function renderCardPicker() {
    const wrap = $("#card-picker");
    wrap.innerHTML = "";
    if (boardNeed() === 0) return;
    const used = new Set(state.board.filter((x) => x != null));
    for (let suit = 0; suit < 4; suit++) {
      for (let ri = 0; ri < 13; ri++) {
        const rank = 12 - ri;
        const c = rank * 4 + suit;
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "pcard" + (cardIsRed(c) ? " red" : "");
        btn.textContent = RANKS[rank] + SUIT_SYM[SUITS[suit]];
        btn.disabled = used.has(c);
        btn.addEventListener("click", () => pickCard(c));
        wrap.appendChild(btn);
      }
    }
  }

  function pickCard(c) {
    ensureBoardLen();
    if (boardNeed() === 0) return;
    const existing = state.board.indexOf(c);
    if (existing >= 0) {
      state.board[existing] = null;
      state.activeSlot = existing;
      renderBoardSlots();
      return;
    }
    state.board[state.activeSlot] = c;
    let next = state.activeSlot + 1;
    while (next < state.board.length && state.board[next] != null) next++;
    if (next < state.board.length) state.activeSlot = next;
    renderBoardSlots();
  }

  function randomBoard() {
    const n = boardNeed();
    const deck = Array.from({ length: 52 }, (_, i) => i);
    for (let i = deck.length - 1; i > 0; i--) {
      const j = (Math.random() * (i + 1)) | 0;
      [deck[i], deck[j]] = [deck[j], deck[i]];
    }
    state.board = deck.slice(0, n);
    state.activeSlot = 0;
    renderBoardSlots();
  }

  function onStreetChange() {
    const street = parseInt($("#f-street").value, 10);
    if (street === 0) {
      $("#f-algo").value = "mccfr_es";
      $("#f-abs").value = "none";
    } else if (street === 1) {
      $("#f-algo").value = "dcfr";
      $("#f-abs").value = "ochs";
    } else {
      $("#f-algo").value = "dcfr";
      $("#f-abs").value = "none";
    }
    updateStackMin();
    renderBoardSlots();
  }

  // (review 2026-09-20 E1) A PREFLOP stack that does not cover the big blind +
  // ante crashed the native solver, and the spinner (min 0.5) walked straight
  // into it. The server enforces the exact rule (validate_root_for_app); this
  // just keeps the spinner out of the rejected zone. Postflop keeps 0.5 — a 1bb
  // river stack is a legitimate root.
  function updateStackMin() {
    const stack = $("#f-stack");
    if (!stack) return;
    const street = parseInt($("#f-street").value, 10);
    const bb = parseFloat($("#f-bb").value) || 0;
    const ante = parseFloat($("#f-ante").value) || 0;
    if (street === 0 && bb > 0) {
      const need = (bb + ante) / bb; // must be strictly exceeded
      stack.min = String(Math.floor(need * 2) / 2 + 0.5);
    } else {
      stack.min = "0.5";
    }
  }

  // ---------- range 13×13 ----------
  // (review 2026-09-20 E11) The grid has NO range grammar of its own. It used to
  // light a cell only when the cell's label was literally a token in the box, so
  // "QQ+" / "KK-TT" / "AA:0.5" selected nothing here while the solver did
  // something else again. The server's strict parser (cfr_app/ranges.py — the
  // same one /api/validate_root and /api/solve use) now says what the text
  // means; the text box stays the single source of truth.
  function rangeBox(which) {
    return $(which === "ip" ? "#f-range-ip" : "#f-range-oop");
  }

  function boardForRanges() {
    return state.board.filter((c) => c != null).map(Number);
  }

  async function refreshRangeInfo(which, toggle) {
    const box = rangeBox(which);
    if (!box) return;
    state.rangeSeq[which] = (state.rangeSeq[which] || 0) + 1;
    const seq = state.rangeSeq[which];
    let info;
    try {
      info = await api("/api/range/parse", {
        method: "POST",
        body: JSON.stringify({ text: box.value, board: boardForRanges(), toggle: toggle || null }),
      });
    } catch (e) {
      info = { ok: false, error: String(e.message || e) };
    }
    if (seq !== state.rangeSeq[which]) return; // a newer edit superseded this reply
    if (info.ok && toggle) box.value = info.normalized || "";
    state.rangeInfo[which] = info;
    renderRangeGrid();
  }

  function refreshAllRanges() {
    refreshRangeInfo("oop");
    refreshRangeInfo("ip");
  }

  function scheduleRangeRefresh(which) {
    clearTimeout(state.rangeTimers[which]);
    state.rangeTimers[which] = setTimeout(() => refreshRangeInfo(which), 250);
  }

  function syncRangeFromTextareas() {
    refreshAllRanges();
  }

  function renderRangeStatus() {
    const el = $("#range-status");
    if (!el) return;
    const bits = ["oop", "ip"].map((which) => {
      const info = state.rangeInfo[which];
      const name = which.toUpperCase();
      if (!info) return `${name}: …`;
      if (!info.ok) return `${name}: ${info.error}`;
      if (info.full) return `${name}: 100% (${info.combos} combos)`;
      return `${name}: ${info.combos} combos · weight ${info.weight} · ${info.pct}%`;
    });
    el.textContent = bits.join("   |   ");
    const bad = ["oop", "ip"].some((w) => state.rangeInfo[w] && !state.rangeInfo[w].ok);
    el.classList.toggle("range-error", bad);
  }

  function renderRangeGrid() {
    const wrap = $("#range-grid");
    if (!wrap) return;
    wrap.innerHTML = "";
    renderRangeStatus();
    const info = state.rangeInfo[state.rangeWhich];
    const classes = (info && info.ok && info.classes) || {};
    const full = !!(info && info.ok && info.full);
    // A high at top-left. Suited above diagonal, pairs on diag, offsuit below.
    for (let ri = 0; ri < 13; ri++) {
      for (let ci = 0; ci < 13; ci++) {
        const rHi = 12 - ri;
        const rLo = 12 - ci;
        const label =
          rHi === rLo
            ? RANKS[rHi] + RANKS[rLo]
            : rHi > rLo
              ? RANKS[rHi] + RANKS[rLo] + "s"
              : RANKS[rLo] + RANKS[rHi] + "o";
        const btn = document.createElement("button");
        btn.type = "button";
        // Mean weight of the class's live combos: 1 = all in, 0.25 = e.g. one of
        // four suited combos, or the whole class at weight 0.25.
        const w = full ? 1 : floatOr(classes[label], 0);
        btn.className = "rg-cell" + (w > 0 ? " on" : "");
        if (w > 0 && w < 1) btn.style.opacity = String(0.35 + 0.65 * w);
        btn.textContent = label;
        btn.title = w > 0 && w < 1 ? `${label} · ${Math.round(w * 100)}%` : label;
        btn.addEventListener("click", () => refreshRangeInfo(state.rangeWhich, label));
        wrap.appendChild(btn);
      }
    }
  }

  function setRangeTab(which) {
    state.rangeWhich = which;
    $("#btn-range-oop-tab").classList.toggle("active", which === "oop");
    $("#btn-range-ip-tab").classList.toggle("active", which === "ip");
    renderRangeGrid();
  }

  // ---------- form / presets ----------
  function applyPreset(p) {
    $("#f-street").value = String(p.street);
    $("#f-seats").value = String(p.num_seats || 2);
    $("#f-pot").value = String(p.pot_bb);
    $("#f-stack").value = String(p.effective_stack_bb);
    if (p.size_preset) $("#f-size-preset").value = p.size_preset;
    if (p.raise_sizes_pm) {
      $("#f-sizes").value = p.raise_sizes_pm.join(",");
      if (!p.raise_sizes_pm.length) $("#f-size-preset").value = "custom";
    }
    if (p.algorithm) $("#f-algo").value = p.algorithm;
    if (p.card_abstraction) $("#f-abs").value = p.card_abstraction;
    if (p.allin_atom != null) $("#f-allin").checked = !!p.allin_atom;
    if (p.ante_chips != null) $("#f-ante").value = String(p.ante_chips);
    if (p.bb_chips != null) $("#f-bb").value = String(p.bb_chips);
    if (p.sb_chips != null) $("#f-sb").value = String(p.sb_chips);
    $("#f-stacks").value = p.stacks_bb && p.stacks_bb.length ? p.stacks_bb.join(",") : "";
    updateStackMin(); // street / bb / ante were set programmatically (no change event)
    state.board = (p.board || []).slice();
    renderBoardSlots();
    $$(".preset-btn").forEach((b) => b.classList.toggle("active", b.dataset.id === p.id));
  }

  function sizesFromBox() {
    const raw = $("#f-sizes").value.trim();
    if (!raw) return [];
    return raw
      .split(/[,\s]+/)
      .filter(Boolean)
      .map((x) => parseInt(x, 10))
      .filter((x) => Number.isFinite(x) && x > 0);
  }

  // (review 2026-09-20) The "Raise sizes" box is what the user sees, so it is
  // what gets solved. The old code returned the PRESET's list whenever a named
  // preset was selected and silently ignored an edited box. Editing the box
  // now flips the preset to "custom" (see boot), and as a backstop a box that
  // no longer matches its preset wins here too.
  function parseSizes() {
    const preset = $("#f-size-preset").value;
    const box = sizesFromBox();
    const named = preset !== "custom" && state.meta && state.meta.size_presets[preset];
    if (named && named.join(",") === box.join(",")) return named.slice();
    return box;
  }

  function collectRoot() {
    const stacksRaw = $("#f-stacks").value.trim();
    const stacks_bb = stacksRaw
      ? stacksRaw.split(/[,\s]+/).filter(Boolean).map(Number)
      : [];
    const board = state.board.filter((c) => c != null).map(Number);
    return {
      street: parseInt($("#f-street").value, 10),
      pot_bb: parseFloat($("#f-pot").value),
      effective_stack_bb: parseFloat($("#f-stack").value),
      board,
      num_seats: parseInt($("#f-seats").value, 10),
      bb_chips: parseInt($("#f-bb").value, 10),
      sb_chips: parseInt($("#f-sb").value, 10),
      ante_chips: parseInt($("#f-ante").value, 10),
      raise_sizes_pm: parseSizes(),
      size_preset: $("#f-size-preset").value,
      allin_atom: $("#f-allin").checked,
      stacks_bb,
      range_oop: ($("#f-range-oop") && $("#f-range-oop").value.trim()) || "",
      range_ip: ($("#f-range-ip") && $("#f-range-ip").value.trim()) || "",
      algorithm: $("#f-algo").value,
      card_abstraction: $("#f-abs").value,
    };
  }

  async function previewTree() {
    try {
      const tree = await api("/api/tree/preview", {
        method: "POST",
        body: JSON.stringify(collectRoot()),
      });
      const el = $("#tree-preview");
      el.classList.remove("muted");
      el.textContent = formatTreeText(tree.tree, 0);
      toast(`Tree: ${tree.num_nodes_built} nodes · ${tree.street_name}`, "ok");
    } catch (e) {
      toast(String(e.message || e), "error");
    }
  }

  function formatTreeText(node, indent) {
    if (!node) return "(empty)";
    const pad = "  ".repeat(indent);
    if (node.truncated) return pad + node.label + "\n";
    let line = pad;
    if (node.terminal) {
      line += `■ ${node.label} [${node.terminal_kind}] pot=${node.pot_bb}\n`;
      return line;
    }
    // Seat index + position in the SOLVER's seat order (HU preflop: P1 = SB acts
    // first, P0 = BB) so the preview reads like the solved strategy.
    const who = `P${node.seat}` + (node.seat_label ? ` ${node.seat_label}` : "");
    line += `● ${who} pot=${node.pot_bb} stack=${node.stack_bb} to_call=${node.to_call_bb}\n`;
    line += pad + `  menu: ${(node.actions || []).join(" | ")}\n`;
    (node.children || []).forEach((ch) => {
      line += formatTreeText(ch, indent + 1);
    });
    return line;
  }

  function collectConfig() {
    const unlimited = $("#c-unlimited") && $("#c-unlimited").checked;
    let iters = parseInt($("#c-iters").value, 10);
    if (unlimited || !Number.isFinite(iters) || iters < 0) iters = 0;
    let poll = parseInt(($("#c-poll") && $("#c-poll").value) || "50", 10);
    if (!Number.isFinite(poll) || poll < 1) poll = 50;
    if (iters === 0) poll = Math.min(poll, 100);
    return {
      max_iterations: iters,
      unlimited: iters === 0,
      thread_num: parseInt($("#c-threads").value, 10),
      target_exploitability_bb: parseFloat($("#c-expl").value) || 0,
      time_budget_secs: parseFloat($("#c-time").value) || 0,
      seed: parseInt($("#c-seed").value, 10),
      use_isomorphism: $("#c-iso").checked,
      algorithm: $("#f-algo").value,
      card_abstraction: $("#f-abs").value,
      poll_every: poll,
    };
  }

  function setTransportButtons(status) {
    const running = status === "running" || status === "queued";
    const paused = status === "paused";
    const live = running || paused;
    const btnSolve = $("#btn-solve");
    const btnPause = $("#btn-pause");
    const btnResume = $("#btn-resume");
    const btnStop = $("#btn-stop");
    if (btnSolve) btnSolve.disabled = live;
    if (btnPause) btnPause.disabled = !running;
    if (btnResume) btnResume.disabled = !paused;
    if (btnStop) btnStop.disabled = !live;
  }

  // ---------- solve ----------
  async function startSolve() {
    const root = collectRoot();
    const config = collectConfig();
    try {
      const job = await api("/api/solve", {
        method: "POST",
        body: JSON.stringify({ root, config, save: true }),
      });
      state.currentJobId = job.job_id;
      setTransportButtons("running");
      updateJobBadges(job);
      const lim = config.unlimited || config.max_iterations === 0 ? "∞" : config.max_iterations;
      toast(`Solve started (${lim} iters)`, "ok");
      startPolling();
    } catch (e) {
      toast(String(e.message || e), "error");
    }
  }

  async function stopSolve() {
    try {
      await api("/api/solve/stop", { method: "POST" });
      const btnPause = $("#btn-pause");
      const btnResume = $("#btn-resume");
      if (btnPause) btnPause.disabled = true;
      if (btnResume) btnResume.disabled = true;
      $("#st-status").textContent = "stopping";
      // (review 2026-09-20 E10) The solver only sees the stop file BETWEEN
      // iterations, and one iteration of a deep tree can run for minutes. The
      // solve now runs in a child process that the server kills if it has not
      // stopped within ~10 s — say so, instead of implying an instant stop.
      toast("Stop requested — saving at the next iteration; force-stopped after ~10 s if stuck", "ok");
    } catch (e) {
      toast(String(e.message || e), "error");
    }
  }

  async function pauseSolve() {
    try {
      const j = await api("/api/solve/pause", { method: "POST" });
      setTransportButtons("paused");
      updateJobBadges(j);
      toast("Paused", "ok");
    } catch (e) {
      toast(String(e.message || e), "error");
    }
  }

  async function resumeSolve() {
    try {
      const j = await api("/api/solve/resume", { method: "POST" });
      setTransportButtons("running");
      updateJobBadges(j);
      toast("Resumed", "ok");
      startPolling();
    } catch (e) {
      toast(String(e.message || e), "error");
    }
  }

  async function startKuhn() {
    try {
      const job = await api("/api/solve/kuhn", {
        method: "POST",
        body: JSON.stringify({ iterations: 8000 }),
      });
      state.currentJobId = job.job_id;
      setTransportButtons("running");
      updateJobBadges(job);
      toast("Kuhn solve started — result stays on this tab", "ok");
      startPolling();
    } catch (e) {
      toast(String(e.message || e), "error");
    }
  }

  function startPolling() {
    stopPolling();
    state.pollTimer = setInterval(pollJobs, 800);
    pollJobs();
  }

  function stopPolling() {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }

  function isKuhnJob(job) {
    return job && ((job.root && job.root.game === "kuhn") || (job.notes || []).includes("kuhn"));
  }

  async function pollJobs() {
    // (review 2026-09-20 E4) setInterval fires every 800 ms whether or not the
    // previous round-trip finished; slow responses used to stack up without
    // bound. One poll in flight at a time — a busy tick is simply skipped.
    if (state.pollBusy) return;
    state.pollBusy = true;
    try {
      await pollJobsOnce();
    } finally {
      state.pollBusy = false;
    }
  }

  async function pollJobsOnce() {
    try {
      const data = await api("/api/jobs");
      const active = data.active;
      if (!active) return;
      if (!state.currentJobId) state.currentJobId = active.job_id;
      updateJobBadges(active);
      setTransportButtons(active.status);
      try {
        const prog = await api(`/api/jobs/${active.job_id}/progress`);
        showProgress(prog);
        if (prog.has_live_strategy || (prog.iterations_run && prog.iterations_run > 0)) {
          const btnView = $("#btn-view-job");
          if (btnView && !isKuhnJob(active)) btnView.disabled = false;
        }
      } catch (_) {
        showJobStats(active);
      }
      if (["done", "error", "stopped", "cancelled"].includes(active.status)) {
        stopPolling();
        setTransportButtons(active.status);
        const bar = $("#st-progress-bar");
        if (bar) {
          bar.classList.remove("indeterminate");
          bar.style.width = "100%";
        }
        if (active.status === "done" || active.status === "stopped") {
          const full = await api(`/api/jobs/${active.job_id}`);
          showJobStats(full);
          if (isKuhnJob(full)) {
            const r = full.report || {};
            $("#st-notes").textContent = [
              "Kuhn NE gate",
              r.value_p0 != null ? `value_p0=${Number(r.value_p0).toFixed(6)}` : null,
              r.nash_value != null ? `nash=${Number(r.nash_value).toFixed(6)}` : null,
              r.exploitability != null ? `expl=${Number(r.exploitability).toFixed(6)}` : null,
            ]
              .filter(Boolean)
              .join("\n");
            toast("Kuhn finished", "ok");
          } else {
            const btnView = $("#btn-view-job");
            if (btnView) btnView.disabled = false;
            if (state.liveView && state.viewSource && state.viewSource.type === "job") {
              await loadJobView(active.job_id, { quiet: true });
            }
          }
          stopLiveView();
        } else if (active.status === "error") {
          showJobStats(active);
          toast(active.error || "solve error", "error");
          stopLiveView();
        }
      }
    } catch (_) {
      /* ignore transient */
    }
  }

  function showProgress(prog) {
    updateJobBadges(prog);
    $("#st-elapsed").textContent =
      prog.elapsed_secs != null ? Number(prog.elapsed_secs).toFixed(1) + "s" : "—";
    const iters = prog.iterations_run != null ? prog.iterations_run : "—";
    const unlimited =
      prog.unlimited || (prog.config && Number(prog.config.max_iterations) === 0);
    $("#st-iters").textContent = unlimited ? `${iters} / ∞` : String(iters);
    $("#st-expl").textContent =
      prog.exploitability_bb != null
        ? Number(prog.exploitability_bb).toFixed(4) + " bb"
        : "—";
    $("#st-ninfo").textContent =
      prog.num_infosets != null ? String(prog.num_infosets) : "—";
    const notes = []
      .concat(prog.notes || [])
      .concat(prog.error ? ["error: " + prog.error] : [])
      .concat(prog.progress_message ? [prog.progress_message] : []);
    $("#st-notes").textContent = notes.filter(Boolean).join("\n");
    const bar = $("#st-progress-bar");
    if (!bar) return;
    const st = prog.status;
    const cfgIters = prog.config && prog.config.max_iterations;
    if (st === "running" || st === "queued" || st === "paused") {
      if (cfgIters && prog.iterations_run) {
        bar.classList.remove("indeterminate");
        bar.style.width = Math.min(99, (100 * prog.iterations_run) / cfgIters) + "%";
      } else {
        bar.classList.add("indeterminate");
      }
    } else if (st === "done" || st === "stopped") {
      bar.classList.remove("indeterminate");
      bar.style.width = "100%";
    }
  }

  function updateJobBadges(job) {
    const jb = $("#job-badge");
    const st = job.status || "idle";
    jb.textContent = st + (job.job_id ? " · " + job.job_id.slice(0, 8) : "");
    jb.className =
      "badge " +
      (st === "running" || st === "queued"
        ? "badge-run"
        : st === "paused"
          ? "badge-paused"
          : st === "done"
            ? "badge-ok"
            : st === "error"
              ? "badge-bad"
              : "badge-idle");
    $("#st-status").textContent = st;
    $("#st-job").textContent = job.job_id || "—";
  }

  function showJobStats(job) {
    updateJobBadges(job);
    const rep = job.report || {};
    $("#st-iters").textContent = rep.iterations_run != null ? rep.iterations_run : "—";
    $("#st-expl").textContent =
      rep.exploitability_bb != null
        ? Number(rep.exploitability_bb).toFixed(4) + " bb"
        : "—";
    const strat = rep.strategy || {};
    const n =
      strat.num_infosets != null
        ? strat.num_infosets
        : (strat.infosets && strat.infosets.length) || "—";
    $("#st-ninfo").textContent = String(n);
    const notes = []
      .concat(job.notes || [])
      .concat(rep.notes || [])
      .concat(job.error ? ["error: " + job.error] : [])
      .concat(job.progress_message ? [job.progress_message] : []);
    $("#st-notes").textContent = notes.filter(Boolean).join("\n");
  }

  async function openCurrentJob(opts = {}) {
    if (!state.currentJobId) return;
    switchTab("viewer");
    const wantLive = opts.live !== false;
    try {
      await loadJobView(state.currentJobId, {
        quiet: !!opts.live,
        live: wantLive,
        firstOpen: true,
      });
    } catch (e) {
      if (!opts.live) toast(String(e.message || e), "error");
    }
    if (wantLive) startLiveView();
  }

  function startLiveView() {
    stopLiveView();
    state.liveView = true;
    state.liveViewSeq = (state.liveViewSeq || 0) + 1;
    const seq = state.liveViewSeq;
    state.liveViewTimer = setInterval(async () => {
      if (!state.currentJobId || !state.liveView) return;
      if (!state.viewSource || state.viewSource.type !== "job") return;
      if (seq !== state.liveViewSeq) return;
      // (review 2026-09-20 E4) /view is the expensive poll (the server parses the
      // whole live snapshot). Never stack requests, and don't fetch what nobody
      // can see: skip while another tick is in flight, while the Strategy tab is
      // not the active panel, or while the window is hidden. Ticks resume on
      // their own when the viewer is visible again; a finished job stops the
      // timer (loadJobView / pollJobs call stopLiveView on a terminal status).
      if (state.liveViewBusy || !viewerVisible()) return;
      state.liveViewBusy = true;
      try {
        await loadJobView(state.currentJobId, { quiet: true, keepSelection: true, seq });
      } catch (_) {
        /* 409 until first snapshot */
      } finally {
        state.liveViewBusy = false;
      }
    }, 1500);
  }

  function viewerVisible() {
    const panel = $("#panel-viewer");
    return !document.hidden && !!panel && panel.classList.contains("active");
  }

  function stopLiveView() {
    state.liveView = false;
    state.liveViewSeq = (state.liveViewSeq || 0) + 1;
    if (state.liveViewTimer) {
      clearInterval(state.liveViewTimer);
      state.liveViewTimer = null;
    }
  }

  // ---------- strategy view ----------
  // (review 2026-09-20 JS races) Every view fetch takes a ticket. A response is
  // rendered only if its ticket is still the newest — otherwise a slow reply
  // for an OLD node/filter/page could land after a newer one and overwrite it.
  function nextViewTicket() {
    state.viewReqSeq = (state.viewReqSeq || 0) + 1;
    return state.viewReqSeq;
  }

  function isStaleView(ticket) {
    return ticket !== state.viewReqSeq;
  }

  // One param builder for every view request, so a live tick carries the same
  // seat / hand filter as "Apply" (live ticks used to drop the hand filter and
  // wipe a filtered table every 1.5 s).
  function buildViewParams() {
    const params = new URLSearchParams({
      limit: String(state.pageLimit),
      offset: String(state.pageOffset),
    });
    const seatSel = $("#v-seat");
    const seat =
      seatSel && seatSel.value !== ""
        ? seatSel.value
        : state.selectedSeat != null
          ? String(state.selectedSeat)
          : "";
    if (seat !== "") params.set("seat", seat);
    const handEl = $("#v-hand");
    const hand = handEl ? handEl.value.trim() : "";
    if (hand) params.set("hand_query", hand);
    // (review 2026-09-20 E6) explicit runout pick; absent = server default
    // (the most-visited runout for the selected node).
    if (state.selectedRunout) params.set("runout", state.selectedRunout);
    return params;
  }

  // Runout picker: only shown when the selected node exists on several boards.
  function renderRunoutPicker(info) {
    const wrap = $("#runout-wrap");
    const sel = $("#v-runout");
    if (!wrap || !sel) return;
    const opts = (info && info.options) || [];
    if (opts.length < 2) {
      wrap.classList.add("hidden");
      sel.innerHTML = "";
      return;
    }
    wrap.classList.remove("hidden");
    sel.innerHTML = "";
    opts.forEach((o) => {
      const el = document.createElement("option");
      el.value = o.key;
      el.textContent = `${o.label} · ${o.num_hands} hands`;
      sel.appendChild(el);
    });
    sel.value = info.selected || opts[0].key;
    const shown = opts.length;
    wrap.title =
      info.total > shown
        ? `Showing the ${shown} most-visited of ${info.total} runouts`
        : `${shown} runouts, most visited first`;
  }

  // A newly opened solution starts unfiltered: selection AND the seat / hand
  // filter controls (buildViewParams reads them) are reset together.
  function resetViewSelection() {
    state.pageOffset = 0;
    state.selectedNodePath = null;
    state.selectedSeat = null;
    state.selectedHandMix = null;
    state.selectedClassId = null;
    state.selectedRunout = null;
    const seatSel = $("#v-seat");
    if (seatSel) seatSel.value = "";
    const handEl = $("#v-hand");
    if (handEl) handEl.value = "";
  }

  async function loadJobView(jobId, opts = {}) {
    state.viewSource = { type: "job", id: jobId };
    if (!opts.keepSelection && !opts.quiet) resetViewSelection();
    const params = buildViewParams();
    if (state.selectedNodePath != null) params.set("path", state.selectedNodePath);
    const ticket = nextViewTicket();
    const data = await api(`/api/jobs/${jobId}/view?` + params.toString());
    if (isStaleView(ticket)) return data;
    if (opts.seq != null && opts.seq !== state.liveViewSeq) return data;
    state.viewData = data;
    renderView(data, {
      keepDetail: !!opts.quiet && state.selectedNodePath != null,
      keepSelection: !!opts.keepSelection,
    });
    applyLiveBadge(data);
    const st = data.job && data.job.status;
    if (st && ["done", "stopped", "error", "cancelled"].includes(st)) stopLiveView();
    if (!opts.quiet) toast("Loaded job " + jobId.slice(0, 8), "ok");
    return data;
  }

  function applyLiveBadge(data) {
    const titleEl = $("#view-title");
    if (!titleEl) return;
    const live =
      data && data.job && ["running", "paused", "queued"].includes(data.job.status);
    let b = titleEl.querySelector(".live-badge");
    if (live) {
      if (!b) {
        b = document.createElement("span");
        b.className = "live-badge";
        titleEl.appendChild(b);
      }
      b.textContent = data.job.status === "paused" ? "paused" : "live";
    } else if (b) {
      b.remove();
    }
  }

  async function loadFileView(path) {
    state.viewSource = { type: "file", path };
    resetViewSelection();
    const q = buildViewParams();
    q.set("path", path);
    const ticket = nextViewTicket();
    const data = await api("/api/view?" + q.toString());
    if (isStaleView(ticket)) return;
    state.viewData = data;
    renderView(data);
    toast("Loaded " + path.split(/[/\\]/).pop(), "ok");
  }

  async function refreshViewPage() {
    if (!state.viewSource) return;
    const params = buildViewParams();
    const source = state.viewSource;
    const ticket = nextViewTicket();
    try {
      let data;
      if (source.type === "job") {
        if (state.selectedNodePath != null) params.set("path", state.selectedNodePath);
        data = await api(`/api/jobs/${source.id}/view?` + params.toString());
      } else {
        params.set("path", source.path);
        if (state.selectedNodePath != null) params.set("path_filter", state.selectedNodePath);
        data = await api("/api/view?" + params.toString());
      }
      // A newer click / filter / page / live tick superseded this request.
      if (isStaleView(ticket)) return;
      state.viewData = data;
      renderView(data, { keepDetail: true });
    } catch (e) {
      if (!isStaleView(ticket)) toast(String(e.message || e), "error");
    }
  }

  function prettyPath(path) {
    const p = String(path || "root");
    if (p === "root" || p === "open" || p === "") return "Open";
    if (/^\d{8,}$/.test(p)) return "Line " + p.slice(-4);
    // (review 2026-09-20) Tokenize first, then shorten each token (mirrors
    // humanize_path / action_short in strategy_view.py). Replacing "_" before
    // CHECK_CALL / RAISE_500 turned them into "CHECK → CALL" / "RAISE → 500".
    return p
      .split(",")
      .map((tok) => tok.trim())
      .filter(Boolean)
      .map((tok) => {
        const up = tok.toUpperCase();
        if (up === "FOLD") return "F";
        if (up === "CHECK_CALL") return "X/C";
        if (up === "ALLIN") return "AI";
        const m = /^RAISE_(\d+)$/.exec(up);
        if (m) return "R" + parseInt(m[1], 10) / 10 + "%";
        return tok.replace(/_/g, " → "); // legacy "AI_F"-style separators
      })
      .join(" → ");
  }

  function renderView(data, opts = {}) {
    const sum = data.summary || {};
    const title =
      (sum.kind === "chart" ? "Chart · " : "Strategy · ") +
      (sum.root && sum.root.root_id
        ? sum.root.root_id
        : (sum.source || "loaded").toString().split(/[/\\]/).pop());
    const titleEl = $("#view-title");
    if (titleEl) {
      const badge = titleEl.querySelector(".live-badge");
      titleEl.textContent = title;
      if (badge) titleEl.appendChild(badge);
    }
    applyLiveBadge(data);

    const streetLab =
      sum.street != null && sum.street >= 0 ? STREET_NAME[sum.street] || `street ${sum.street}` : null;
    const chips = [
      streetLab,
      sum.board_str || null,
      sum.num_infosets != null ? `${sum.num_infosets} infosets` : null,
      sum.iterations_run != null ? `${sum.iterations_run} iters` : null,
      sum.exploitability_bb != null ? `expl ${Number(sum.exploitability_bb).toFixed(3)} bb` : null,
      data.matrix && data.matrix.aggregated_from_combos ? "class avg" : null,
    ]
      .filter(Boolean)
      .join(" · ");
    const chipsEl = $("#view-meta-chips");
    if (chipsEl) {
      chipsEl.textContent = chips || "loaded";
      chipsEl.classList.remove("muted");
    }

    const lines = [
      `status: ${sum.status}`,
      `street: ${streetLab || sum.street}  board: ${sum.board_str || "—"}`,
      `infosets: ${sum.num_infosets}  nodes: ${sum.num_nodes}`,
      sum.iterations_run != null ? `iters: ${sum.iterations_run}` : null,
      sum.exploitability_bb != null ? `expl: ${Number(sum.exploitability_bb).toFixed(4)} bb` : null,
      sum.range_oop ? `range_oop: ${sum.range_oop}` : null,
      sum.range_ip ? `range_ip: ${sum.range_ip}` : null,
    ].filter(Boolean);
    $("#view-summary").textContent = lines.join("\n");
    $("#view-summary").classList.remove("muted");

    renderQuality(sum.quality);
    state.lineNav = data.line_nav || null;
    state.chartPack = data.chart_pack || null;
    renderRunoutPicker(data.runout);
    renderLineNav(data);
    renderLineTree(data.solution_tree);
    state.loadedPath = sum.source || (state.viewSource && state.viewSource.path) || null;
    const exp = $("#btn-export");
    if (exp) exp.disabled = !state.loadedPath || state.loadedPath === "<memory>";

    const nl = $("#node-list");
    nl.innerHTML = "";
    (data.nodes || []).forEach((n) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className =
        "node-btn" +
        (String(n.path) === String(state.selectedNodePath) && n.seat === state.selectedSeat
          ? " active"
          : "");
      const lab = n.label || `P${n.seat} · ${prettyPath(n.path)}`;
      b.textContent = `${lab} (${n.num_hands})`;
      b.addEventListener("click", () => selectDecisionNode(n));
      nl.appendChild(b);
    });

    const seats = new Set((data.nodes || []).map((n) => n.seat));
    const seatSel = $("#v-seat");
    if (seatSel) {
      const cur = seatSel.value;
      seatSel.innerHTML = '<option value="">all</option>';
      [...seats].sort().forEach((s) => {
        const o = document.createElement("option");
        o.value = String(s);
        o.textContent = "seat " + s;
        seatSel.appendChild(o);
      });
      if (cur) seatSel.value = cur;
    }

    if (state.selectedNodePath == null && data.nodes && data.nodes.length) {
      const nav = data.line_nav;
      let first = data.nodes[0];
      if (nav && nav.root_path != null) {
        const hit = data.nodes.find(
          (n) => normalizeLinePath(n.path) === normalizeLinePath(nav.root_path)
        );
        if (hit) first = hit;
        state.selectedNodePath = String(nav.root_path);
        state.selectedSeat = nav.root_seat != null ? nav.root_seat : first.seat;
      } else {
        state.selectedNodePath = String(first.path);
        state.selectedSeat = first.seat;
      }
      showNodeAgg(first);
    }

    renderMatrix(data.matrix);
    renderTable(data.page);

    const matrixEmpty = !data.matrix || data.matrix.empty || !data.matrix.cells || !data.matrix.cells.length;
    if (matrixEmpty && state.viewMode === "matrix") setViewMode("table");
    else setViewMode(state.viewMode);

    if (!opts.keepDetail) {
      $("#hand-detail").innerHTML = '<span class="muted">Select a cell or row.</span>';
    }
    updateBreadcrumb();
  }

  function normalizeLinePath(path) {
    const p = String(path == null ? "" : path).trim();
    if (p === "" || p === "root") return "open";
    return p;
  }

  function pathTokenList(path) {
    const p = normalizeLinePath(path);
    if (p === "open") return [];
    if (/^\d{8,}$/.test(p)) return [];
    return p.split(/[,]/).filter(Boolean);
  }

  function joinLinePath(tokens) {
    return tokens.length ? tokens.join(",") : "open";
  }

  function tokenPretty(tok) {
    const t = String(tok || "");
    // Chart packs use short tokens; native dumps use the solver's full labels.
    if (t === "F" || t === "FOLD") return "Fold";
    if (t === "AI" || t === "ALLIN") return "All-in";
    if (t === "XC" || t === "X" || t === "C" || t === "CHECK_CALL") return "Check/Call";
    const m = /^(?:RAISE_|R)(\d+)$/.exec(t);
    if (m) return `Bet ${parseInt(m[1], 10) / 10}%`;
    return prettyPath(t);
  }

  function currentNavEntry() {
    const nav = state.lineNav;
    if (!nav || !nav.by_path) return null;
    const key = normalizeLinePath(state.selectedNodePath || nav.root_path || "open");
    return nav.by_path[key] || null;
  }

  function packEntry(path) {
    const pack = state.chartPack && state.chartPack.by_path;
    if (!pack) return null;
    return pack[normalizeLinePath(path)] || null;
  }

  function actorLabel(entry, path) {
    const pe = packEntry(path);
    if (pe && pe.seat) return String(pe.seat);
    if (entry && entry.label) return entry.label;
    if (state.selectedSeat != null) return "P" + state.selectedSeat;
    return "Hero";
  }

  function selectDecisionNode(n) {
    state.selectedNodePath = String(n.path);
    state.selectedSeat = n.seat;
    state.selectedHandMix = null;
    state.selectedClassId = null;
    state.selectedRunout = null; // a runout belongs to one node; new node → its default
    const seatSel = $("#v-seat");
    if (seatSel) seatSel.value = String(n.seat);
    state.pageOffset = 0;
    showNodeAgg(n);
    updateBreadcrumb();
    refreshViewPage();
  }

  function updateBreadcrumb() {
    const el = $("#tree-breadcrumb");
    if (el) {
      if (state.selectedNodePath == null && state.selectedSeat == null) el.textContent = "—";
      else el.textContent = `P${state.selectedSeat ?? "?"} · ${prettyPath(state.selectedNodePath)}`;
    }
    renderLineNav(state.viewData || {});
  }

  function renderLineNav(data) {
    const strip = $("#action-tree-strip");
    const crumbsEl = $("#line-breadcrumb");
    const actorEl = $("#line-actor");
    if (!strip) return;
    const nav = (data && data.line_nav) || state.lineNav;
    const nodes = (data && data.nodes) || [];
    const path = normalizeLinePath(
      state.selectedNodePath || (nav && nav.root_path) || (nodes[0] && nodes[0].path) || "open"
    );
    const entry = nav && nav.by_path ? nav.by_path[path] : null;
    const tokens = pathTokenList(path);

    if (actorEl) {
      const hands = entry ? entry.num_hands : nodes.find((n) => normalizeLinePath(n.path) === path);
      const nHands = entry ? entry.num_hands : hands && hands.num_hands;
      // (review 2026-09-20 E6) name the dealt card(s) this node is shown for.
      const dealt = data && data.runout && data.runout.label;
      actorEl.textContent =
        `${actorLabel(entry, path)} to act` +
        (nHands ? " · " + nHands + " hands" : "") +
        (dealt ? " · dealt " + dealt : "");
      actorEl.classList.remove("muted");
    }

    if (crumbsEl) {
      crumbsEl.classList.remove("muted");
      crumbsEl.innerHTML = "";
      const bits = [{ label: "Root", tokens: [], path: "open" }];
      tokens.forEach((tok, i) => {
        bits.push({
          label: tokenPretty(tok),
          tokens: tokens.slice(0, i + 1),
          path: joinLinePath(tokens.slice(0, i + 1)),
        });
      });
      bits.forEach((b, i) => {
        if (i) {
          const sep = document.createElement("span");
          sep.className = "line-sep";
          sep.textContent = "›";
          crumbsEl.appendChild(sep);
        }
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "line-crumb" + (i === bits.length - 1 ? " active" : "");
        btn.textContent = b.label;
        btn.addEventListener("click", () => goToLinePath(b.path));
        crumbsEl.appendChild(btn);
      });
    }

    strip.innerHTML = "";
    const actions = mixActionsForDisplay(entry);
    if (!actions.length) {
      strip.innerHTML =
        '<div class="muted tree-empty">No actions at this node. Load a solution or pick a different line.</div>';
      return;
    }
    const navigable = !nav || nav.navigable !== false;
    actions.forEach((a) => {
      const btn = document.createElement("button");
      btn.type = "button";
      const pct = Math.round((a.freq || 0) * 1000) / 10;
      btn.className = "line-act " + (a.css || "") + (a.terminal ? " terminal" : "");
      btn.disabled = !a.has_next && !a.terminal ? false : false;
      if (!a.has_next && !a.terminal && !navigable) btn.disabled = true;
      btn.innerHTML = `
        <span class="la-name">${escapeHtml(a.short || a.action)}</span>
        <span class="la-freq">${pct}%</span>
        <div class="la-bar"><i class="${a.css || ""}" style="width:${Math.min(100, pct)}%;background:currentColor"></i></div>
      `;
      btn.title = a.has_next
        ? `Play ${a.short} → next player`
        : a.terminal
          ? `${a.short} ends the hand`
          : `${a.short} (no node in this dump)`;
      btn.addEventListener("click", () => takeLineAction(a));
      strip.appendChild(btn);
    });
    if (nav && nav.navigable === false) {
      const hint = document.createElement("div");
      hint.className = "line-hint";
      hint.textContent =
        "This dump has no action-line keys (history hashes only). Re-solve or open a chart pack to walk the tree.";
      strip.appendChild(hint);
    }
  }

  function mixActionsForDisplay(entry) {
    const base = (entry && entry.actions) || [];
    const mix = state.selectedHandMix;
    if (!mix || !mix.length) return base;
    const byAct = {};
    mix.forEach((s) => {
      byAct[String(s.action || "").toUpperCase()] = floatOr(s.prob, 0);
    });
    return base.map((a) => {
      const p = byAct[String(a.action || "").toUpperCase()];
      return p == null ? a : { ...a, freq: p };
    });
  }

  function floatOr(v, d) {
    const n = Number(v);
    return Number.isFinite(n) ? n : d;
  }

  async function goToLinePath(path) {
    const key = normalizeLinePath(path);
    const nav = state.lineNav;
    const entry = nav && nav.by_path && nav.by_path[key];
    const pack = packEntry(key);
    if (pack && pack.file && (!entry || !state.viewSource || state.viewSource.type === "file")) {
      // Jump to sibling chart when this file only holds one node
      const onlyOne = nav && nav.by_path && Object.keys(nav.by_path).length <= 1;
      if (onlyOne && pack.file) {
        state.selectedNodePath = key;
        state.selectedSeat = pack.seat_index != null ? pack.seat_index : state.selectedSeat;
        state.selectedHandMix = null;
        try {
          await api("/api/library/load", {
            method: "POST",
            body: JSON.stringify({ path: pack.file }),
          });
          await loadFileView(pack.file);
        } catch (e) {
          toast(String(e.message || e), "error");
        }
        return;
      }
    }
    const node = ((state.viewData && state.viewData.nodes) || []).find(
      (n) => normalizeLinePath(n.path) === key
    );
    if (node) {
      selectDecisionNode(node);
      return;
    }
    state.selectedNodePath = key;
    state.selectedHandMix = null;
    state.selectedRunout = null;
    state.pageOffset = 0;
    refreshViewPage();
  }

  async function takeLineAction(act) {
    if (act.terminal && !act.has_next) {
      toast((act.short || act.action) + " ends the hand", "ok");
      return;
    }
    if (act.chart_file) {
      const nav = state.lineNav;
      const onlyOne = nav && nav.by_path && Object.keys(nav.by_path).length <= 1;
      if (onlyOne || act.chart_file) {
        state.selectedHandMix = null;
        state.selectedClassId = null;
        state.selectedNodePath = act.next_path;
        if (act.next_seat != null) state.selectedSeat = act.next_seat;
        try {
          await api("/api/library/load", {
            method: "POST",
            body: JSON.stringify({ path: act.chart_file }),
          });
          await loadFileView(act.chart_file);
        } catch (e) {
          toast(String(e.message || e), "error");
        }
        return;
      }
    }
    if (act.has_next && act.next_path) {
      const node = ((state.viewData && state.viewData.nodes) || []).find(
        (n) => normalizeLinePath(n.path) === normalizeLinePath(act.next_path)
      );
      if (node) {
        selectDecisionNode(node);
        return;
      }
      state.selectedNodePath = act.next_path;
      if (act.next_seat != null) state.selectedSeat = act.next_seat;
      state.selectedHandMix = null;
      state.selectedRunout = null;
      state.pageOffset = 0;
      refreshViewPage();
      return;
    }
    toast("No next decision for " + (act.short || act.action), "error");
  }

  function renderQuality(q) {
    const el = $("#quality-panel");
    if (!el) return;
    if (!q) {
      el.textContent = "No quality metrics.";
      el.classList.add("muted");
      return;
    }
    el.classList.remove("muted");
    el.textContent = [
      q.exploitability_bb != null ? `expl: ${Number(q.exploitability_bb).toFixed(4)} bb` : null,
      q.iterations_run != null ? `iters: ${q.iterations_run}` : null,
      `infosets: ${q.num_infosets}`,
      `H(π) mean: ${q.mean_entropy}`,
      `mix F/C/R/AI: ${q.mean_fold} / ${q.mean_call} / ${q.mean_raise} / ${q.mean_allin}`,
      `pure (≥99%): ${((q.pure_strategy_frac || 0) * 100).toFixed(1)}%`,
    ]
      .filter(Boolean)
      .join("\n");
  }

  function showNodeAgg(n) {
    const el = $("#node-agg");
    if (!el) return;
    const agg = n.aggregate;
    if (!agg) {
      el.classList.add("hidden");
      return;
    }
    el.classList.remove("hidden");
    const chips = Object.entries(agg.mean_mix || {})
      .map(([a, p]) => `<span class="mix-chip">${escapeHtml(a)}: ${(p * 100).toFixed(1)}%</span>`)
      .join("");
    el.innerHTML = `<strong>${escapeHtml(n.label || "P" + n.seat)} · ${escapeHtml(prettyPath(n.path))}</strong> · ${agg.num_hands} hands
      <div class="mix-row">${chips}</div>`;
  }

  function renderLineTree(st) {
    const el = $("#line-tree");
    if (!el) return;
    el.innerHTML = "";
    if (!st || !st.forest) return;
  }

  function setViewMode(mode) {
    state.viewMode = mode;
    $("#matrix-wrap").classList.toggle("hidden", mode !== "matrix");
    $("#table-wrap").classList.toggle("hidden", mode !== "table");
    $("#btn-show-matrix").classList.toggle("active", mode === "matrix");
    $("#btn-show-table").classList.toggle("active", mode === "table");
  }

  function cellColor(cell) {
    if (!cell) return "#1a2030";
    const f = cell.fold || 0;
    const c = cell.call || 0;
    const a = cell.agg || 0;
    const r = Math.round(45 * f + 50 * c + 200 * a);
    const g = Math.round(70 * f + 110 * c + 55 * a);
    const b = Math.round(120 * f + 200 * c + 55 * a);
    return `rgb(${r},${g},${b})`;
  }

  function renderMatrix(matrix) {
    const wrap = $("#matrix");
    wrap.innerHTML = "";
    const legend = $("#matrix-legend");
    legend.innerHTML =
      '<span class="leg-fold">Fold</span><span class="leg-call">Call/Check</span><span class="leg-raise">Raise/Agg</span><span class="leg-allin">All-in (in mix)</span>';

    if (!matrix || matrix.empty || !matrix.cells || !matrix.cells.length) {
      wrap.innerHTML =
        '<div class="matrix-empty">No 13×13 matrix for this node. Use the Table view for combo strategies.</div>';
      return;
    }
    const labels = matrix.rank_labels || RANK_DISP.split("");
    const corner = document.createElement("div");
    corner.className = "mcell head";
    wrap.appendChild(corner);
    labels.forEach((lab) => {
      const h = document.createElement("div");
      h.className = "mcell head";
      h.textContent = lab;
      wrap.appendChild(h);
    });
    for (let ri = 0; ri < 13; ri++) {
      const rh = document.createElement("div");
      rh.className = "mcell head";
      rh.textContent = labels[ri];
      wrap.appendChild(rh);
      for (let ci = 0; ci < 13; ci++) {
        const cell = matrix.cells[ri][ci];
        const el = document.createElement("div");
        el.className = "mcell" + (cell ? "" : " empty");
        if (cell) {
          const f = cell.fold || 0;
          const c = cell.call || 0;
          const a = cell.agg || 0;
          el.innerHTML = `
            <div class="mix-stack" aria-hidden="true">
              <div class="ms" style="flex-basis:${Math.round(a * 100)}%;background:#c83737"></div>
              <div class="ms" style="flex-basis:${Math.round(c * 100)}%;background:#3270c8"></div>
              <div class="ms" style="flex-basis:${Math.round(f * 100)}%;background:#2d4678"></div>
            </div>
            <span class="lab">${escapeHtml(cell.label)}</span>
            <span class="pct">${Math.round(a * 100)}%</span>`;
          el.style.background = cellColor(cell);
          el.title = [
            cell.label,
            `F ${Math.round(f * 100)}%  C ${Math.round(c * 100)}%  R ${Math.round(a * 100)}%`,
            cell.n_combos ? `${cell.n_combos} combos averaged` : "",
          ]
            .filter(Boolean)
            .join("\n");
          el.addEventListener("click", () => {
            $$(".mcell.selected").forEach((x) => x.classList.remove("selected"));
            el.classList.add("selected");
            state.selectedClassId = cell.class_id;
            state.selectedHandMix = cell.strategy || null;
            showHandDetail({
              hand_label: cell.label,
              seat: matrix.seat != null ? matrix.seat : state.selectedSeat,
              path: matrix.path != null ? matrix.path : state.selectedNodePath,
              strategy: cell.strategy,
              private: cell.class_id,
              private_kind: "class",
            });
            renderLineNav(state.viewData || {});
          });
          if (state.selectedClassId != null && cell.class_id === state.selectedClassId) {
            el.classList.add("selected");
          }
        }
        wrap.appendChild(el);
      }
    }
  }

  function renderTable(page) {
    const tbody = $("#hand-table tbody");
    tbody.innerHTML = "";
    const rows = (page && page.rows) || [];
    rows.forEach((r) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${escapeHtml(r.hand_label || "—")}</strong></td>
        <td>P${r.seat}</td>
        <td class="muted">${escapeHtml(prettyPath(r.path))}</td>
        <td>${stratBarHtml(r.strategy)}</td>
        <td class="muted">${escapeHtml(String(r.primary_action || ""))} ${
          r.primary_prob != null ? (r.primary_prob * 100).toFixed(0) + "%" : ""
        }</td>
      `;
      tr.addEventListener("click", () => {
        $$("#hand-table tbody tr.selected").forEach((x) => x.classList.remove("selected"));
        tr.classList.add("selected");
        state.selectedHandMix = r.strategy || null;
        showHandDetail(r);
        renderLineNav(state.viewData || {});
      });
      tbody.appendChild(tr);
    });
    const total = (page && page.total) || 0;
    const off = (page && page.offset) || 0;
    const lim = (page && page.limit) || state.pageLimit;
    $("#page-info").textContent = total ? `${off + 1}–${Math.min(off + lim, total)} of ${total}` : "0 rows";
  }

  function stratBarHtml(strategy) {
    if (!strategy || !strategy.length) return "—";
    return (
      `<div class="strat-bars">` +
      strategy
        .map((s) => {
          const w = Math.max(0, Math.round(floatOr(s.prob, 0) * 100));
          if (w <= 0) return "";
          // (review 2026-09-20 XSS) `short` is the raw action label for anything
          // the server doesn't recognise, i.e. attacker-controlled text from an
          // uploaded / loaded JSON — it was the one unescaped innerHTML sink.
          const tip = `${escapeHtml(s.short || s.action || "")}: ${floatOr(s.pct, w)}%`;
          return `<div class="seg ${escapeHtml(s.css || "act-other")}" style="width:${w}%" title="${tip}"></div>`;
        })
        .join("") +
      `</div>`
    );
  }

  function showHandDetail(r) {
    const el = $("#hand-detail");
    const strat = r.strategy || [];
    let html = `<h3>${escapeHtml(r.hand_label || r.infoset_id || "hand")}</h3>`;
    html += `<div class="muted" style="font-family:var(--mono);font-size:0.75rem;margin-bottom:0.6rem">P${
      r.seat ?? "?"
    } · ${escapeHtml(prettyPath(r.path))}</div>`;
    if (!strat.length) {
      html += '<div class="muted">No action mix.</div>';
    } else {
      strat.forEach((s) => {
        const pct = s.pct != null ? s.pct : Math.round((s.prob || 0) * 1000) / 10;
        const col =
          s.css === "act-fold"
            ? "var(--fold)"
            : s.css === "act-call"
              ? "var(--call)"
              : s.css === "act-allin"
                ? "var(--allin)"
                : "var(--raise)";
        html += `<div class="bar-row">
          <span class="act-name">${escapeHtml(s.short || s.action)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${col}"></div></div>
          <span class="act-pct">${pct}%</span>
        </div>`;
      });
    }
    el.innerHTML = html;
    el.classList.remove("muted");
  }

  async function uploadSolutionFile(file) {
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file, file.name);
    try {
      toast("Uploading " + file.name + "…");
      // multipart cannot be application/json → the token is what authorizes it
      const res = await fetch("/api/upload", { method: "POST", body: fd, headers: authHeaders() });
      let body = null;
      const ct = res.headers.get("content-type") || "";
      if (ct.includes("application/json")) body = await res.json();
      else body = await res.text();
      if (!res.ok) {
        const detail = (body && body.detail) || body || res.statusText;
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      state.pageOffset = 0;
      state.selectedNodePath = null;
      state.selectedSeat = null;
      if (body.path) await loadFileView(body.path);
      else if (body.job_id) await loadJobView(body.job_id);
      switchTab("viewer");
      toast(`Loaded ${body.kind || "solution"} · ${body.num_infosets || "?"} infosets`, "ok");
    } catch (e) {
      toast(String(e.message || e), "error");
    }
  }

  function wireUpload() {
    const input = $("#upload-file");
    const open = () => input && input.click();
    if (input)
      input.addEventListener("change", () => {
        if (!input.files || !input.files[0]) return;
        uploadSolutionFile(input.files[0]);
        input.value = "";
      });
    const btn = $("#btn-upload");
    if (btn) btn.addEventListener("click", open);
    const btnLib = $("#btn-upload-lib");
    if (btnLib) btnLib.addEventListener("click", open);
    const panel = $("#panel-viewer");
    if (panel) {
      ["dragenter", "dragover"].forEach((ev) => {
        panel.addEventListener(ev, (e) => {
          e.preventDefault();
          e.stopPropagation();
          panel.classList.add("drag-over");
        });
      });
      ["dragleave", "drop"].forEach((ev) => {
        panel.addEventListener(ev, (e) => {
          e.preventDefault();
          e.stopPropagation();
          panel.classList.remove("drag-over");
        });
      });
      panel.addEventListener("drop", (e) => {
        const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (f) uploadSolutionFile(f);
      });
    }
  }

  // ---------- library ----------
  function streetLabel(s) {
    if (s == null || s === "") return "—";
    return STREET_NAME[s] || String(s);
  }

  function renderLibrary(filter) {
    const tbody = $("#lib-table tbody");
    tbody.innerHTML = "";
    const q = (filter || "").trim().toLowerCase();
    const items = (state.libItems || []).filter((it) => {
      if (!q) return true;
      const blob = `${it.name} ${it.rel || ""} ${it.kind || ""} ${it.root_id || ""}`.toLowerCase();
      return blob.includes(q);
    });
    items.forEach((it) => {
      const tr = document.createElement("tr");
      const expl =
        it.exploitability_bb != null ? Number(it.exploitability_bb).toFixed(3) : "—";
      tr.innerHTML = `
        <td><strong>${escapeHtml(it.name)}</strong><div class="muted" style="font-size:0.7rem">${escapeHtml(it.rel || "")}</div></td>
        <td class="muted">${escapeHtml(it.kind || "—")}</td>
        <td>${escapeHtml(streetLabel(it.street))}</td>
        <td class="muted">${expl}</td>
        <td class="muted">${it.num_infosets != null ? it.num_infosets : "—"}</td>
        <td class="muted">${it.size_kb} KB</td>
        <td><button type="button" class="btn btn-ghost btn-sm">Open</button></td>
      `;
      const open = async () => {
        try {
          await api("/api/library/load", {
            method: "POST",
            body: JSON.stringify({ path: it.path }),
          });
          await loadFileView(it.path);
          switchTab("viewer");
        } catch (e) {
          toast(String(e.message || e), "error");
        }
      };
      tr.querySelector("button").addEventListener("click", (e) => {
        e.stopPropagation();
        open();
      });
      tr.addEventListener("click", open);
      tbody.appendChild(tr);
    });
  }

  async function loadLibrary() {
    try {
      const data = await api("/api/library");
      state.libItems = data.items || [];
      renderLibrary($("#lib-filter") ? $("#lib-filter").value : "");
    } catch (e) {
      toast(String(e.message || e), "error");
    }
  }

  // ---------- boot ----------
  async function boot() {
    initTabs();
    wireUpload();
    $("#f-street").addEventListener("change", onStreetChange);
    $("#btn-clear-board").addEventListener("click", () => {
      state.board = [];
      renderBoardSlots();
    });
    $("#btn-random-board").addEventListener("click", randomBoard);
    $("#f-size-preset").addEventListener("change", () => {
      const p = $("#f-size-preset").value;
      if (state.meta && state.meta.size_presets[p]) {
        $("#f-sizes").value = state.meta.size_presets[p].join(",");
      }
    });
    ["#f-bb", "#f-ante"].forEach((sel) => $(sel).addEventListener("input", updateStackMin));
    // Typing in the sizes box means "custom" — make the UI say so (see parseSizes).
    $("#f-sizes").addEventListener("input", () => {
      $("#f-size-preset").value = "custom";
    });
    $("#btn-validate").addEventListener("click", async () => {
      try {
        const r = await api("/api/validate_root", {
          method: "POST",
          body: JSON.stringify(collectRoot()),
        });
        if (r.ok) {
          // Say what the range text MEANS, not just that it parsed (E11).
          const rg = r.ranges || {};
          const bit = (n, x) => (x ? ` · ${n} ${x.full ? "100%" : x.combos + " combos"}` : "");
          toast("Root OK: " + (r.root.root_id || "valid") + bit("OOP", rg.oop) + bit("IP", rg.ip), "ok");
        } else toast(r.error || "invalid", "error");
      } catch (e) {
        toast(String(e.message || e), "error");
      }
    });
    $("#btn-solve").addEventListener("click", startSolve);
    $("#btn-stop").addEventListener("click", stopSolve);
    $("#btn-pause").addEventListener("click", pauseSolve);
    $("#btn-resume").addEventListener("click", resumeSolve);
    const unl = $("#c-unlimited");
    if (unl) {
      unl.addEventListener("change", () => {
        const iters = $("#c-iters");
        if (unl.checked) {
          iters.dataset.prev = iters.value;
          iters.value = "0";
          iters.disabled = true;
        } else {
          iters.disabled = false;
          iters.value = iters.dataset.prev || "200";
        }
      });
    }
    $("#btn-kuhn").addEventListener("click", startKuhn);
    $("#btn-view-job").addEventListener("click", () => openCurrentJob({ live: true }));
    $("#btn-filter").addEventListener("click", () => {
      state.pageOffset = 0;
      refreshViewPage();
    });
    $("#btn-prev").addEventListener("click", () => {
      state.pageOffset = Math.max(0, state.pageOffset - state.pageLimit);
      refreshViewPage();
    });
    $("#btn-next").addEventListener("click", () => {
      state.pageOffset += state.pageLimit;
      refreshViewPage();
    });
    const runoutSel = $("#v-runout");
    if (runoutSel) {
      runoutSel.addEventListener("change", () => {
        state.selectedRunout = runoutSel.value || null;
        state.selectedHandMix = null;
        state.selectedClassId = null;
        state.pageOffset = 0;
        refreshViewPage();
      });
    }
    $("#btn-show-matrix").addEventListener("click", () => setViewMode("matrix"));
    $("#btn-show-table").addEventListener("click", () => setViewMode("table"));
    $("#btn-refresh-lib").addEventListener("click", loadLibrary);
    const libFilter = $("#lib-filter");
    if (libFilter) libFilter.addEventListener("input", () => renderLibrary(libFilter.value));
    $("#btn-tree-preview").addEventListener("click", previewTree);
    $("#btn-range-oop-tab").addEventListener("click", () => setRangeTab("oop"));
    $("#btn-range-ip-tab").addEventListener("click", () => setRangeTab("ip"));
    $("#btn-range-aa").addEventListener("click", () => {
      $("#f-range-oop").value = "AA,KK";
      $("#f-range-ip").value = "random";
      syncRangeFromTextareas();
    });
    $("#btn-range-clear").addEventListener("click", () => {
      $("#f-range-oop").value = "";
      $("#f-range-ip").value = "";
      syncRangeFromTextareas();
    });
    ["oop", "ip"].forEach((which) => {
      const box = rangeBox(which);
      box.addEventListener("input", () => scheduleRangeRefresh(which)); // live, debounced
      box.addEventListener("change", () => refreshRangeInfo(which));
    });

    $("#btn-export").addEventListener("click", async () => {
      if (!state.loadedPath || state.loadedPath === "<memory>") {
        toast("No file path to export", "error");
        return;
      }
      try {
        const r = await api("/api/export", {
          method: "POST",
          body: JSON.stringify({ path: state.loadedPath }),
        });
        toast("Exported → " + r.rel, "ok");
      } catch (e) {
        toast(String(e.message || e), "error");
      }
    });
    $("#btn-compare").addEventListener("click", async () => {
      const pathB = $("#cmp-path").value.trim();
      const pathA =
        state.loadedPath && state.loadedPath !== "<memory>"
          ? state.loadedPath
          : state.viewSource && state.viewSource.path;
      if (!pathA || !pathB) {
        toast("Need current file + path B", "error");
        return;
      }
      try {
        const r = await api("/api/compare", {
          method: "POST",
          body: JSON.stringify({ path_a: pathA, path_b: pathB }),
        });
        const d = r.diff || {};
        $("#cmp-out").textContent = [
          `common hands: ${d.num_common}`,
          `mean L1: ${d.mean_l1}`,
          "top diffs:",
          ...(d.top_diffs || [])
            .slice(0, 8)
            .map((x) => `  ${x.hand}: L1=${x.l1}  ${x.a_primary} vs ${x.b_primary}`),
        ].join("\n");
      } catch (e) {
        toast(String(e.message || e), "error");
      }
    });

    renderRangeGrid();

    try {
      const health = await api("/api/health");
      const rb = $("#rust-badge");
      if (health.rust_cfr) {
        rb.textContent = "CFR ready";
        rb.className = "badge badge-ok";
      } else {
        rb.textContent = "CFR missing";
        rb.className = "badge badge-bad";
      }
      state.meta = await api("/api/meta");
      const list = $("#preset-list");
      (state.meta.presets || []).forEach((p) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "preset-btn";
        b.dataset.id = p.id;
        b.textContent = p.label;
        b.addEventListener("click", () => applyPreset(p));
        list.appendChild(b);
      });
      if (state.meta.presets && state.meta.presets[2]) applyPreset(state.meta.presets[2]);
      else renderBoardSlots();
    } catch (e) {
      toast("Failed to load meta: " + e.message, "error");
      renderBoardSlots();
    }
    await reattachActiveJob();
  }

  // (review 2026-09-20 JS races) The solve lives in the server, not the page. A
  // reload (F5, webview refresh) used to come back "idle" with Play enabled
  // while a job was still running — no Stop button, and Play just 409'd.
  async function reattachActiveJob() {
    try {
      const data = await api("/api/jobs");
      const active = data && data.active;
      if (!active) return;
      state.currentJobId = active.job_id;
      updateJobBadges(active);
      setTransportButtons(active.status);
      if (["running", "queued", "paused"].includes(active.status)) {
        startPolling();
      } else {
        showJobStats(active);
        const btnView = $("#btn-view-job");
        if (btnView && !isKuhnJob(active) && active.num_infosets > 0) btnView.disabled = false;
      }
    } catch (_) {
      /* server not ready — stay idle */
    }
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
