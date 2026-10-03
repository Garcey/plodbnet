"use strict";
// Hand review (2026-10-03) — the site's one paid page (it keeps your hand histories on
// the server): drop the .zip ClubGG exports (no unpacking), see your profit and loss
// beside your all-in EV (side pots included), every hand in the replayer with Open in
// Study, and every decision you made graded by the network — the worst-played first.
// The store and its API: python/plo5bp/ui/handreview_store.py.
(function () {
  const HG = globalThis.HG;
  const UI = HG.ui;
  const { $, C, html, put, icon, h, d2, toast, confirmDialog, loading, segHtml, segWire, miniCards, fillMiniCards } = HG.uikit;
  const API = "/games/api/review";
  const POLL_MS = 1000;
  const GRADE_POLL_MS = 5000;
  // one page's state; `epoch` changes whenever the view is (re)opened, so a poll from an
  // earlier visit never paints over this one
  const R = { epoch: 0, sort: "time", dir: "desc", filter: "", offset: 0, rows: [], total: 0, summary: null, bb: 2000, unit: "usd" };
  const UNIT_KEY = "hg.review.unit.v1";
  try { R.unit = localStorage.getItem(UNIT_KEY) === "bb" ? "bb" : "usd"; } catch (_) { /* private mode */ }

  const signed = (c) => (c > 0 ? "+" : "") + d2(c);
  const tone = (c) => (c > 0 ? "pos" : c < 0 ? "neg" : "muted");
  const bbOf = (c) => `${c > 0 ? "+" : ""}${(c / (R.bb || 2000)).toFixed(1)} bb`;
  const amount = (c) => (R.unit === "bb" ? bbOf(c) : signed(c));
  const plural = (n, w) => `${Number(n).toLocaleString()} ${w}${n === 1 ? "" : "s"}`;
  const REASONS = {
    other_game: "not PLO5 bomb pots", not_a_bomb_pot: "not bomb pots", not_double_board: "single-board hands",
    no_hero_cards: "hands without your cards", hero_not_dealt: "hands you weren't dealt into",
    storage_full: "over your storage limit", duplicate_in_upload: "repeated inside the upload",
  };

  function showReview() {
    $("lobby").hidden = true;
    $("table-view").hidden = true;
    $("review-view").hidden = false;
    document.body.classList.remove("in-lobby");
    document.title = "Hand review · WrapGTO";
    R.epoch++;
    load();
  }

  // ------------------------------------------------------------------ loading
  async function load() {
    const ep = R.epoch;
    const main = $("rv-main");
    put(main, loading());
    await checkoutReturn();
    let s;
    try { s = await C().j(`${API}/summary`); }
    catch (e) {
      if (ep !== R.epoch) return;
      if (e.status === 402) return paintPaywall(e.detail || {});
      put(main, html`<div class="rv-note err">${e.message}</div>`);
      return;
    }
    if (ep !== R.epoch) return;
    R.summary = s;
    R.bb = (s.stakes && s.stakes[0] && s.stakes[0].bb_cents) || 2000;
    paint(s);
    if (s.grading_pending) gradePoll(ep);
    const busy = (s.uploads || []).find((u) => u.status === "queued" || u.status === "reading");
    if (busy) pollUpload(busy.id, ep);
  }

  // Back from Stripe's checkout (`?checkout=success&session_id=…`): confirm it, then
  // load the page as a subscriber. (The webhook does the same if it is configured.)
  async function checkoutReturn() {
    const q = new URLSearchParams(location.search || "");
    const st = q.get("checkout");
    if (!st) return;
    history.replaceState(null, "", "/games/review");
    if (st !== "success" || !q.get("session_id")) { toast("Checkout cancelled — nothing was charged.", ""); return; }
    try {
      const d = await C().j(`/billing/confirm?session_id=${encodeURIComponent(q.get("session_id"))}`);
      toast(d.active ? "Subscribed — welcome to Hand review" : "Your payment hasn't gone through yet — try again in a moment", d.active ? "ok" : "err", 7000);
    } catch (e) { toast(e.message, "err"); }
  }

  // ------------------------------------------------------------------ paywall
  function paintPaywall(d) {
    const price = d.price_cents ? `$${(d.price_cents / 100).toFixed(d.price_cents % 100 ? 2 : 0)}` : "$10";
    put($("rv-main"), html`<section class="rv-hero"><div>
        <h1>Hand <em>review</em></h1>
        <p>Your ClubGG PLO5 double-board bomb pots, back in WrapGTO: what you won beside what you should have won, every hand replayed, every decision checked.</p>
      </div></section>
      <section class="rv-pay">
        <ul class="rv-feats">
          <li>${icon("i-chart")}<span><b>Profit and loss beside your all-in EV</b> — the luck taken out at every all-in with cards to come, side pots included.</span></li>
          <li>${icon("i-list")}<span><b>Every hand in the replayer</b> — step through it, and send any spot to Study with one click.</span></li>
          <li>${icon("i-bolt")}<span><b>Every decision you made checked against the network</b> — sort by your worst-played hands and review those first.</span></li>
          <li>${icon("i-upload")}<span><b>Drop the zip ClubGG exports</b> — no unpacking. Hands you've already uploaded are skipped, so the same export can go in twice.</span></li>
        </ul>
        <div class="rv-pay-cta">
          <button class="btn gold lg" id="rv-subscribe" type="button" ${d.billing_configured === false ? "disabled" : ""}>Subscribe — ${price}/month</button>
          <p class="muted small">Hand review is the one paid part of WrapGTO: it keeps your hand histories on our server. Study, the Trainer and home games stay free.${d.billing_configured === false ? " (Billing isn't switched on yet.)" : ""}</p>
        </div>
      </section>`);
    const b = $("rv-subscribe");
    if (b) b.addEventListener("click", async () => {
      b.disabled = true;
      try {
        const r = await C().j("/billing/checkout", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ next: "/games/review" }) });
        location.href = r.url;
      } catch (e) { b.disabled = false; toast(e.message, "err"); }
    });
  }

  // ------------------------------------------------------------------ the page
  function paint(s) {
    const main = $("rv-main");
    const empty = !s.hands;
    put(main, html`<section class="rv-hero"><div>
        <h1>Hand <em>review</em></h1>
        <p>${empty ? "Drop the zip of hand histories ClubGG exports and see how you really ran: your results beside your all-in EV, and every decision checked against the network."
          : html`${plural(s.hands, "hand")} of PLO5 double-board bomb pots${s.first_ts ? html` · ${dateOf(s.first_ts)} – ${dateOf(s.last_ts)}` : ""}.`}</p>
      </div><span class="spacer"></span>
      <div class="lb-actions">${empty ? "" : html`<button class="btn sm ghost" id="rv-unit" type="button" title="Show money in dollars or big blinds">${R.unit === "bb" ? "Show $" : "Show bb"}</button><button class="btn sm ghost" id="rv-delete" type="button" title="Delete every hand you uploaded">${icon("i-trash", "sm")}<span>Delete all</span></button>`}</div></section>
      <label class="rv-drop" id="rv-drop" for="rv-file">
        <input type="file" id="rv-file" accept=".zip,.txt,application/zip,text/plain" class="sr-only"/>
        <span class="rv-drop-ico">${icon("i-upload", "lg")}</span>
        <span class="rv-drop-txt"><b>${empty ? "Drop your ClubGG hand-history zip here" : "Add more hands"}</b><small>or click to choose the file — no need to unzip it · up to ${s.max_upload_mb || 25} MB · hands already here are skipped</small></span>
      </label>
      <div id="rv-progress"></div>
      ${empty ? "" : html`<div id="rv-stats"></div><div id="rv-graph"></div>
      <section class="rv-list-sec">
        <div class="db-bar">${segHtml("rv-sort", [["time", "Date"], ["worst", "Worst played"], ["net", "Profit / loss"], ["luck", "Luck"], ["pot", "Pot size"]], R.sort)}
          <button type="button" class="btn sm" id="rv-dir"></button></div>
        <div class="db-bar">${segHtml("rv-filter", [["", "All hands"], ["allin", "All-ins"], ["showdown", "Showdowns"], ["mistakes", "Mistakes"]], R.filter)}</div>
        <div id="rv-list"></div><button class="btn block" id="rv-more" type="button" hidden>Load more</button>
      </section>`}`);
    wireDrop();
    if (empty) return;
    paintStats(s);
    segWire(main);
    $("rv-sort").addEventListener("pick", (e) => { R.sort = e.detail; R.dir = "desc"; loadList(true); });
    $("rv-filter").addEventListener("pick", (e) => { R.filter = e.detail; loadList(true); });
    $("rv-dir").addEventListener("click", () => { R.dir = R.dir === "desc" ? "asc" : "desc"; loadList(true); });
    $("rv-more").addEventListener("click", () => loadList(false));
    $("rv-delete").addEventListener("click", deleteAll);
    $("rv-unit").addEventListener("click", () => {
      R.unit = R.unit === "bb" ? "usd" : "bb";
      try { localStorage.setItem(UNIT_KEY, R.unit); } catch (_) { /* private mode */ }
      paint(R.summary);
    });
    loadGraph();
    loadList(true);
  }

  function dateOf(ts) {
    const d = new Date(Number(ts) * 1000);
    return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: d.getUTCFullYear() === new Date().getFullYear() ? undefined : "numeric", timeZone: "UTC" });
  }

  function paintStats(s) {
    const luck = s.net_cents - s.ev_net_cents;
    const accTip = s.graded ? `${plural(s.graded, "decision")} graded` : s.grading_pending ? "being checked…" : "no decisions graded yet";
    put($("rv-stats"), html`<div class="pcard-stats rv-stats">
      <div><b>${Number(s.hands).toLocaleString()}</b><small>Hands</small></div>
      <div><b class="${tone(s.net_cents)}">${amount(s.net_cents)}</b><small>Net won${R.unit === "bb" ? "" : html` · ${bbOf(s.net_cents)}`}</small></div>
      <div><b class="${tone(s.ev_net_cents)} ev">${amount(s.ev_net_cents)}</b><small>All-in EV result</small></div>
      <div><b class="${tone(luck)}">${amount(luck)}</b><small>${luck >= 0 ? "Above" : "Below"} EV · ${plural(s.allin_hands, "all-in")}</small></div>
      <div><b>${s.accuracy == null ? "–" : Math.round(s.accuracy) + "%"}</b><small>Accuracy · ${accTip}</small></div>
      <div><b class="${s.mistakes ? "neg" : ""}">${Number(s.mistakes).toLocaleString()}</b><small>Mistakes (wrong moves and blunders)</small></div>
    </div>${s.grading_pending ? html`<div class="rv-note">${icon("i-bolt", "sm")} Checking your decisions against the network — ${plural(s.grading_pending, "hand")} to go.</div>` : ""}`);
  }

  // ------------------------------------------------------------------ upload
  function wireDrop() {
    const drop = $("rv-drop"), input = $("rv-file");
    if (!drop || !input) return;
    input.addEventListener("change", () => { if (input.files && input.files[0]) upload(input.files[0]); input.value = ""; });
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "dragend"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      drop.classList.remove("over");
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) upload(f);
    });
  }

  async function upload(file) {
    const max = ((R.summary && R.summary.max_upload_mb) || 25) * 1024 * 1024;
    if (!/\.(zip|txt)$/i.test(file.name)) return toast("Choose the .zip of hand histories (or one of its .txt files)", "err");
    if (file.size > max) return toast(`That file is over ${Math.round(max / 1048576)} MB — split the export into smaller zips`, "err");
    const ep = R.epoch;
    progress(html`<div class="rv-up"><span class="spin"></span><span>Uploading <b>${file.name}</b> (${(file.size / 1048576).toFixed(1)} MB)…</span></div>`);
    try {
      const r = await C().j(`${API}/upload?name=${encodeURIComponent(file.name)}`, {
        method: "POST", headers: { "Content-Type": /\.zip$/i.test(file.name) ? "application/zip" : "text/plain" }, body: file, timeout: 300000,
      });
      pollUpload(r.upload_id, ep);
    } catch (e) {
      progress("");
      if (e.status === 402) return load();
      toast(e.message, "err", 8000);
    }
  }

  function progress(markup) { const el = $("rv-progress"); if (el) put(el, markup); }

  async function pollUpload(id, ep) {
    let u;
    try { u = await C().j(`${API}/uploads/${id}`); } catch (e) { if (ep === R.epoch) setTimeout(() => pollUpload(id, ep), 3 * POLL_MS); return; }
    if (ep !== R.epoch) return;
    if (u.status === "queued" || u.status === "reading") {
      const pct = u.total ? Math.round((100 * u.read) / u.total) : 0;
      progress(html`<div class="rv-up"><span class="spin"></span><span>${u.status === "queued" ? "Waiting for its turn…" : html`Reading <b>${u.filename}</b> — ${u.read.toLocaleString()} / ${u.total.toLocaleString()} hands · ${u.added.toLocaleString()} new · ${u.duplicates.toLocaleString()} already here`}</span><i class="rv-bar" data-vars="w:${pct}%"></i></div>`);
      setTimeout(() => pollUpload(id, ep), POLL_MS);
      return;
    }
    if (u.status === "failed") {
      progress(html`<div class="rv-note err">${icon("i-warn", "sm")} ${u.filename}: ${u.error || "it couldn't be read"}</div>`);
      return;
    }
    const why = Object.entries(u.skipped_detail || {}).map(([k, n]) => `${n.toLocaleString()} ${REASONS[k] || k.replace(/_/g, " ")}`);
    const report = html`<div class="rv-note ok">${icon("i-check", "sm")} <span><b>${plural(u.added, "new hand")}</b> from ${u.filename}${u.duplicates ? ` · ${plural(u.duplicates, "duplicate")} skipped` : ""}${why.length ? ` · not read: ${why.join(", ")}` : ""}</span></div>`;
    progress(report);
    toast(u.added ? `${plural(u.added, "hand")} added` : "No new hands in that file", u.added ? "ok" : "");
    await load();
    if (ep === R.epoch) progress(report);  // (the report stays after the repaint)
  }

  // While hands wait for their grades, the numbers refresh every few seconds.
  function gradePoll(ep) {
    setTimeout(async () => {
      if (ep !== R.epoch || document.hidden) { if (ep === R.epoch) gradePoll(ep); return; }
      let s;
      try { s = await C().j(`${API}/summary`); } catch (_) { gradePoll(ep); return; }
      if (ep !== R.epoch) return;
      const was = R.summary ? R.summary.grading_pending : 0;
      R.summary = s;
      paintStats(s);
      if (s.grading_pending) gradePoll(ep);
      if (!s.grading_pending || was - s.grading_pending >= 25) loadList(true);
    }, GRADE_POLL_MS);
  }

  async function deleteAll() {
    const n = R.summary ? R.summary.hands : 0;
    const ok = await confirmDialog({ title: "Delete every hand?", text: `All ${plural(n, "hand")} you uploaded and their grades are deleted from our server. Your ClubGG export is untouched — you can upload it again.`, okLabel: "Delete them", danger: true });
    if (!ok) return;
    try {
      const r = await C().j(`${API}/delete`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirm: true }) });
      toast(`${plural(r.deleted, "hand")} deleted`, "ok");
      load();
    } catch (e) { toast(e.message, "err"); }
  }

  // ------------------------------------------------------------------ the graph
  // Two running lines from the first hand to the last: what you won (green) and your
  // all-in EV result (gold) — the gap between them is luck. Point at it (or drag a
  // finger along it) for the hand and both numbers there.
  let gseq = 0;
  async function loadGraph() {
    const seq = ++gseq, ep = R.epoch;
    let sr;
    try { sr = await C().j(`${API}/series`); } catch (_) { return; }
    if (seq !== gseq || ep !== R.epoch) return;
    drawGraph($("rv-graph"), sr);
  }
  function drawGraph(host, sr) {
    const pts = (sr && sr.points) || [];
    if (!host) return;
    if (pts.length < 3) { put(host, ""); return; }
    const bbMode = R.unit === "bb";
    const val = (p, k) => (bbMode ? p[k + 2] * 100 : p[k]);  // (bb x 100 keeps one axis scale)
    const W = 900, H = 210, top = 10, bottom = 10;
    const n = pts[pts.length - 1][0] || 1;
    const all = pts.flatMap((p) => [val(p, 1), val(p, 2)]);
    const hi = Math.max(0, ...all), lo = Math.min(0, ...all), span = hi - lo || 1;
    const x = (k) => (k / n) * W;
    const y = (v) => top + (1 - (v - lo) / span) * (H - top - bottom);
    const path = (k) => pts.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)} ${y(val(p, k)).toFixed(1)}`).join(" ");
    const zero = y(0);
    const fmt = (p, k) => (bbMode ? `${p[k + 2] > 0 ? "+" : ""}${p[k + 2].toFixed(1)} bb` : signed(p[k]));
    const last = pts[pts.length - 1];
    put(host, html`<figure class="pgraph rv-graph" aria-label="Over ${n} hands: net ${fmt(last, 1)}, all-in EV ${fmt(last, 2)}">
      <figcaption><span>Results over ${plural(n, "hand")}</span><span class="rv-legend"><i class="lg-net"></i>Net won <b class="num ${tone(last[1])}">${fmt(last, 1)}</b><i class="lg-ev"></i>All-in EV <b class="num ${tone(last[2])}">${fmt(last, 2)}</b></span></figcaption>
      <div class="pg-box rv-box"><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">
        <line class="pg-zero" x1="0" x2="${W}" y1="${zero.toFixed(1)}" y2="${zero.toFixed(1)}"/>
        <path class="rv-line ev" d="${path(2)}"/><path class="rv-line net" d="${path(1)}"/>
      </svg><i class="pg-dot" hidden></i><i class="pg-dot ev" hidden></i><span class="pg-tip num" hidden></span></div></figure>`);
    const box = host.querySelector(".pg-box"), dots = host.querySelectorAll(".pg-dot"), tip = host.querySelector(".pg-tip");
    const show = (e) => {
      const r = box.getBoundingClientRect();
      if (!r.width) return;
      const k = Math.max(0, Math.min(n, ((e.clientX - r.left) / r.width) * n));
      let best = pts[0];
      for (const p of pts) if (Math.abs(p[0] - k) < Math.abs(best[0] - k)) best = p;
      const px = (best[0] / n) * 100;
      [1, 2].forEach((j, i) => { dots[i].hidden = false; dots[i].style.left = px + "%"; dots[i].style.top = (y(val(best, j)) / H) * 100 + "%"; });
      tip.hidden = false;
      tip.style.left = px + "%";
      tip.classList.toggle("flip", px > 66);
      tip.textContent = `${best[0] ? `Hand ${best[0].toLocaleString()}` : "Start"} · net ${fmt(best, 1)} · EV ${fmt(best, 2)}`;
    };
    box.addEventListener("pointermove", show);
    box.addEventListener("pointerdown", show);
    box.addEventListener("pointerleave", () => { dots.forEach((d) => { d.hidden = true; }); tip.hidden = true; });
  }

  // ------------------------------------------------------------------ the hand list
  let lseq = 0;
  async function loadList(reset) {
    const seq = ++lseq, ep = R.epoch;
    if (reset) { R.offset = 0; R.rows = []; }
    let d;
    try { d = await C().j(`${API}/hands?sort=${R.sort}&dir=${R.dir}&filter=${R.filter}&limit=40&offset=${R.offset}`); }
    catch (e) { if (e.status === 402) return load(); toast(e.message, "err"); return; }
    if (seq !== lseq || ep !== R.epoch) return;
    R.rows = R.rows.concat(d.hands);
    R.total = d.total;
    R.offset += d.limit;
    drawList();
  }
  function gradeCell(x) {
    if (x.grading) return html`<span class="muted small">checking…</span>`;
    if (x.accuracy == null) return x.decisions ? html`<span class="muted small">–</span>` : html`<span class="muted small">no decisions</span>`;
    const cls = x.mistakes ? "g-blunder" : x.worst != null && x.worst < 70 ? "g-inaccuracy" : "g-correct";
    const label = x.mistakes ? plural(x.mistakes, "mistake") : `${Math.round(x.accuracy)}%`;
    return html`<span class="grade sm ${cls}" title="Accuracy ${Math.round(x.accuracy)}% · worst decision ${Math.round(x.worst)}/100">${label}</span>`;
  }
  function drawList() {
    const host = $("rv-list");
    if (!host) return;
    const worstFirst = R.sort === "worst";
    $("rv-dir").textContent = R.sort === "time" ? (R.dir === "desc" ? "Newest first" : "Oldest first")
      : worstFirst ? (R.dir === "desc" ? "Worst first" : "Best first") : (R.dir === "desc" ? "↓ High to low" : "↑ Low to high");
    put(host, R.rows.length ? "" : html`<div class="muted empty-note">${R.filter ? "No hands match." : "No hands yet."}</div>`);
    R.rows.forEach((x) => {
      const luck = x.net_cents - x.ev_net_cents;
      const row = h("button", { class: "hand-row rv-row", type: "button" },
        html`<span class="hr-cards">${miniCards(x.my_hole, "mine")}<span class="boards2 gap board">${miniCards(x.board_a)}${miniCards(x.board_b)}</span></span>
          <span class="rv-res"><b class="net ${tone(x.net_cents)}">${amount(x.net_cents)}</b>${x.allin ? html`<small class="num" title="Your all-in EV result: the luck taken out">EV ${amount(x.ev_net_cents)}${luck ? html` · <span class="${tone(luck)}">${luck > 0 ? "ran good" : "ran bad"}</span>` : ""}</small>` : ""}</span>
          <span class="rv-grade">${gradeCell(x)}</span>
          <span class="who">${x.played_at} · ${x.players} players · pot ${d2(x.pot_cents)}${x.showdown ? " · showdown" : ""}${x.allin ? " · all in" : ""}</span>`);
      row.addEventListener("click", () => UI.openHand(null, null, { url: `${API}/hands/${encodeURIComponent(x.key)}`, noLink: true }));
      host.appendChild(row);
    });
    fillMiniCards(host);
    $("rv-more").hidden = R.rows.length >= R.total;
  }

  Object.assign(UI, { showReview });
})();
