"use strict";
// Home games — UI chrome: toasts, dialogs, menus, the lobby, the side rail
// (chat / hand log / ledger / history) and the host's "Manage table" drawer.
// The action dock lives in games.play.js; the felt in games.table.js.
(function () {
  const HG = (globalThis.HG = globalThis.HG || {});
  const $ = (id) => document.getElementById(id);
  const C = () => HG.core;
  const esc = (x) => C().esc(x);
  const icon = (id, cls) => `<svg class="ico ${cls || ""}"><use href="#${id}"/></svg>`;
  const U = (HG.uiState = {
    eventSeen: null, chatSig: "", logSig: "", ledgerSig: "", handsFor: null, hands: null, handsLoading: false,
    unread: 0, drawer: null, drawerTab: "game", modals: [], menu: null, lobbySig: "", railTab: "chat",
  });

  function h(tag, attrs, html) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") e.className = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else if (v === true) e.setAttribute(k, "");
      else if (v !== false && v != null) e.setAttribute(k, v);
    }
    if (html != null) e.innerHTML = html;
    return e;
  }
  function avatar(name, key, cls) {
    const A = HG.avatar;
    return `<span class="av ${cls || ""}" style="--h:${A.hueOf(key != null ? key : name)}">${esc(A.initials(name))}</span>`;
  }
  function moneyInput(id, cents, attrs) {
    return `<div class="money"><input class="input" id="${id}" inputmode="decimal" autocomplete="off" value="${(cents / 100).toFixed(2)}" ${attrs || ""}/></div>`;
  }
  const d2 = (c) => C().dollars(c);

  // ------------------------------------------------------------------ toasts
  function toast(msg, kind, ms) {
    const root = $("toast-root");
    if (!root) return;
    const t = h("div", { class: "toast " + (kind || "") }, esc(msg));
    root.appendChild(t);
    while (root.children.length > 4) root.firstChild.remove();
    setTimeout(() => { t.classList.add("out"); setTimeout(() => t.remove(), 260); }, ms || 3400);
  }

  // ------------------------------------------------------------------ modals
  function openModal(opts) {
    const root = $("modal-root");
    const layer = h("div", { class: "layer", style: "position:absolute;inset:0" });
    const scrim = h("div", { class: "scrim" });
    const wrap = h("div", { class: "modal-wrap" });
    const modal = h("div", { class: "modal" + (opts.wide ? " wide" : ""), role: "dialog", "aria-modal": "true" });
    const head = h("div", { class: "m-head" }, `<h3>${esc(opts.title || "")}</h3>`);
    const x = h("button", { class: "icon-btn", type: "button", "aria-label": "Close" }, icon("i-x"));
    head.appendChild(x);
    modal.appendChild(head);
    if (opts.sub) modal.appendChild(h("div", { class: "m-sub" }, esc(opts.sub)));
    const body = h("div", { class: "m-body" });
    if (typeof opts.body === "string") body.innerHTML = opts.body; else if (opts.body) body.appendChild(opts.body);
    modal.appendChild(body);
    const foot = h("div", { class: "m-foot" });
    const close = (val) => {
      layer.classList.remove("open");
      U.modals = U.modals.filter((m) => m !== api);
      setTimeout(() => layer.remove(), 240);
      if (opts.onClose) opts.onClose(val);
    };
    const api = { close, body, modal, foot };
    (opts.buttons || []).forEach((b) => {
      const btn = h("button", { class: "btn " + (b.cls || ""), type: "button" }, esc(b.label));
      btn.addEventListener("click", async () => {
        if (!b.onClick) return close(b.value);
        btn.disabled = true;
        try { const keep = await b.onClick(api); if (keep !== false) close(b.value); }
        catch (e) { if (e && e.status !== 409) { /* post() already toasted */ } }
        finally { btn.disabled = false; }
      });
      foot.appendChild(btn);
    });
    if (foot.children.length) modal.appendChild(foot);
    wrap.appendChild(modal);
    layer.appendChild(scrim);
    layer.appendChild(wrap);
    root.appendChild(layer);
    x.addEventListener("click", () => close(null));
    scrim.addEventListener("click", () => close(null));
    requestAnimationFrame(() => layer.classList.add("open"));
    U.modals.push(api);
    const first = body.querySelector("input,select,button");
    if (first && opts.autofocus !== false) setTimeout(() => first.focus({ preventScroll: true }), 60);
    return api;
  }
  function confirmDialog(o) {
    return new Promise((resolve) => {
      let done = false;
      openModal({
        title: o.title, sub: o.text, body: o.body || "",
        onClose: (v) => { if (!done) { done = true; resolve(v === true); } },
        buttons: [{ label: o.cancelLabel || "Cancel", cls: "ghost", value: false }, { label: o.okLabel || "Confirm", cls: o.danger ? "danger" : "primary", value: true }],
      });
    });
  }
  function closeTop() {
    if (U.menu) { closeMenu(); return true; }
    if (U.modals.length) { U.modals[U.modals.length - 1].close(null); return true; }
    if (U.drawer) { closeDrawer(); return true; }
    return false;
  }

  // ------------------------------------------------------------------- menus
  function openMenu(anchor, items) {
    closeMenu();
    const m = h("div", { class: "menu" });
    items.forEach((it) => {
      if (it === "-") return m.appendChild(h("hr"));
      if (it.header) return m.appendChild(h("div", { class: "mh" }, esc(it.header)));
      const b = h("button", { type: "button", class: it.danger ? "danger" : "" }, (it.icon ? icon(it.icon) : "") + `<span>${esc(it.label)}</span>`);
      b.disabled = !!it.disabled;
      if (it.hint) b.title = it.hint;
      b.addEventListener("click", () => { closeMenu(); it.onClick(); });
      m.appendChild(b);
    });
    document.body.appendChild(m);
    const r = anchor.getBoundingClientRect();
    const mw = m.offsetWidth, mh = m.offsetHeight;
    let left = Math.min(globalThis.innerWidth - mw - 8, Math.max(8, r.right - mw));
    let top = r.bottom + 6;
    if (top + mh > globalThis.innerHeight - 8) top = Math.max(8, r.top - mh - 6);
    m.style.left = left + "px";
    m.style.top = top + "px";
    U.menu = m;
    setTimeout(() => document.addEventListener("pointerdown", onDocDown, true), 0);
  }
  function onDocDown(e) { if (U.menu && !U.menu.contains(e.target)) closeMenu(); }
  function closeMenu() {
    if (U.menu) { U.menu.remove(); U.menu = null; }
    document.removeEventListener("pointerdown", onDocDown, true);
  }

  // ------------------------------------------------------------------- lobby
  const STAKES = [
    { l: "Micro", sb: 10, bb: 25, ante: 75, buy: 1000 },
    { l: "Low", sb: 25, bb: 50, ante: 150, buy: 2000 },
    { l: "Medium", sb: 50, bb: 100, ante: 300, buy: 4000 },
    { l: "High", sb: 100, bb: 200, ante: 600, buy: 8000 },
  ];

  function miniFelt(t) {
    const n = t.num_seats, by = {};
    (t.players || []).forEach((p) => (by[p.seat] = p));
    let out = "";
    for (let i = 0; i < n; i++) {
      const th = Math.PI / 2 + (i * 2 * Math.PI) / n;
      const x = 50 + 46 * Math.cos(th), y = 50 + 40 * Math.sin(th);
      const p = by[i];
      out += p
        ? `<span class="av ${p.is_me ? "me" : ""}" style="left:${x}%;top:${y}%;--h:${HG.avatar.hueOf(p.name)}" title="${esc(p.name)}">${esc(HG.avatar.initials(p.name))}</span>`
        : `<span class="slot" style="left:${x}%;top:${y}%"></span>`;
    }
    return `<div class="tcard-felt">${out}<div class="mid"><small>Ante</small>${d2(t.ante_cents)}</div></div>`;
  }
  function tableCard(t) {
    const full = t.seated >= t.num_seats;
    const cta = t.is_seated ? "Return to table" : full ? "Watch" : "Join table";
    const card = h("div", { class: "tcard" });
    card.innerHTML =
      `<div class="tcard-top"><div style="min-width:0;flex:1"><div class="tcard-name">${esc(t.name)}</div>` +
      `<div class="tcard-host">Hosted by ${esc(t.host_name)}${t.is_host ? " (you)" : ""}</div></div>` +
      `<span class="pill ${t.running ? "live" : "paused"}">${t.running ? "Live" : "Paused"}</span></div>` +
      miniFelt(t) +
      `<div class="tcard-meta"><span class="pill gold num">${d2(t.sb_cents)}/${d2(t.bb_cents)}</span>` +
      `<span class="pill">${icon("i-users", "sm")}${t.seated}/${t.num_seats}</span>` +
      (t.hand_no ? `<span class="pill num">Hand #${t.hand_no}</span>` : "") +
      (t.listed ? "" : `<span class="pill">${icon("i-lock", "sm")}Link only</span>`) + "</div>";
    const row = h("div", { class: "tcard-cta" });
    const go = h("button", { class: "btn " + (t.is_seated ? "primary" : ""), type: "button" }, esc(cta));
    go.addEventListener("click", () => C().openTable(t.id, true).catch((e) => toast(e.message, "err")));
    const copy = h("button", { class: "btn", type: "button", title: "Copy invite link" }, icon("i-link", "sm"));
    copy.addEventListener("click", () => copyInvite(t.id));
    row.appendChild(go); row.appendChild(copy);
    card.appendChild(row);
    return card;
  }
  function renderLobby(data) {
    const tables = data.tables || [];
    const sig = JSON.stringify(data);
    if (sig === U.lobbySig) return;
    U.lobbySig = sig;
    const mine = tables.filter((t) => t.is_host || t.is_seated);
    const open = tables.filter((t) => !(t.is_host || t.is_seated));
    $("lb-mine-sec").hidden = !mine.length;
    const mineEl = $("lb-mine"), openEl = $("lb-open");
    mineEl.innerHTML = ""; openEl.innerHTML = "";
    mine.forEach((t) => mineEl.appendChild(tableCard(t)));
    open.forEach((t) => openEl.appendChild(tableCard(t)));
    $("lb-open-count").textContent = open.length ? `${open.length} running` : "";
    if (!open.length) {
      const empty = h("div", { class: "lb-empty", style: "grid-column:1/-1" },
        mine.length ? "<b>No other tables right now</b>When a friend hosts one it shows up here." : "<b>No tables yet</b>Host one and send your friends the invite link — it takes ten seconds.");
      openEl.appendChild(empty);
    }
    const sess = data.sessions || [];
    $("lb-sess-sec").hidden = !sess.length;
    $("lb-sessions").innerHTML = sess.map((x) =>
      `<div class="sess" data-id="${esc(x.id)}" role="button" tabindex="0" title="Open this session: final ledger and every hand"><div><b>${esc(x.name)}</b><br><small>${d2(x.sb_cents)}/${d2(x.bb_cents)} · ante ${d2(x.ante_cents)}</small></div>` +
      `<small class="opt">${x.hands} hand${x.hands === 1 ? "" : "s"}</small><small class="opt">in for ${d2(x.buyin_cents)}</small>` +
      `<b class="num ${x.net_cents >= 0 ? "pos" : "neg"}">${x.net_cents >= 0 ? "+" : ""}${d2(x.net_cents)}</b></div>`).join("");
    wireSessions();
    loadClub(false); // (rides the lobby poll, at most every 30 s)
  }
  function wireSessions() {
    document.querySelectorAll("#lb-sessions .sess[data-id]").forEach((row) => row.addEventListener("click", () => C().openTable(row.dataset.id, true).catch((e) => toast(e.message, "err"))));
  }
  async function copyInvite(id) {
    const url = `${location.origin}/games/t/${id}`;
    try { await navigator.clipboard.writeText(url); toast("Invite link copied", "ok"); }
    catch (_) { openModal({ title: "Invite link", sub: "Copy this link and send it to your friends.", body: `<input class="input" readonly value="${esc(url)}" onfocus="this.select()"/>`, buttons: [{ label: "Done", cls: "primary" }] }); }
  }

  function openCreate() {
    let pick = 2;
    const me = C().G.me || {};
    const body = h("div", { style: "display:flex;flex-direction:column;gap:16px" });
    body.innerHTML =
      `<label class="field"><span>Table name</span><input type="text" id="c-name" maxlength="60" value="${esc((me.name || "My").split(" ")[0])}'s game"/></label>` +
      `<div class="field"><span>Stakes</span><div class="stake-grid" id="c-stakes">` +
      STAKES.map((s, i) => `<button type="button" class="stake ${i === pick ? "on" : ""}" data-i="${i}"><b>${d2(s.sb)}/${d2(s.bb)}</b><small>${s.l} · ante ${d2(s.ante)}</small></button>`).join("") +
      `</div><small>Blinds set the chip unit and the minimum bet. Nobody posts them — every hand is a bomb pot: all players ante and the action starts on the flop.</small></div>` +
      `<div class="row3" style="grid-template-columns:minmax(0,1fr) minmax(0,1fr) auto"><label class="field"><span>Ante</span>${moneyInput("c-ante", STAKES[pick].ante)}</label>` +
      `<label class="field"><span>Your buy-in</span>${moneyInput("c-buyin", STAKES[pick].buy)}</label>` +
      `<div class="field"><span>Seats</span><div class="seg" id="c-seats">${[2, 4, 6, 8].map((n) => `<button type="button" data-v="${n}" class="${n === 8 ? "on" : ""}">${n}</button>`).join("")}</div></div></div>` +
      `<details class="adv"><summary>More options</summary><div>` +
      `<div class="row2"><label class="field"><span>Small blind</span>${moneyInput("c-sb", STAKES[pick].sb)}</label><label class="field"><span>Big blind</span>${moneyInput("c-bb", STAKES[pick].bb)}</label></div>` +
      `<div class="row2"><label class="field"><span>Min buy-in</span>${moneyInput("c-min", 0)}<small>0 = no minimum</small></label><label class="field"><span>Max buy-in</span>${moneyInput("c-max", 0)}<small>0 = no maximum</small></label></div>` +
      `<div class="field"><span>Decision time</span><div class="seg" id="c-clock">${[[0, "Off"], [15, "15s"], [20, "20s"], [30, "30s"], [45, "45s"], [60, "60s"]].map(([v, l]) => `<button type="button" data-v="${v}" class="${v === 30 ? "on" : ""}">${l}</button>`).join("")}</div></div>` +
      `<div class="field"><span>Time bank</span><div class="seg" id="c-bank">${[[0, "Off"], [30, "30s"], [60, "60s"], [120, "2 min"]].map(([v, l]) => `<button type="button" data-v="${v}" class="${v === 30 ? "on" : ""}">${l}</button>`).join("")}</div></div>` +
      `<div class="field"><span>Next hand</span><div class="seg" id="c-deal">${[[0, "Manual"], [3, "3s"], [5, "5s"], [8, "8s"], [12, "12s"]].map(([v, l]) => `<button type="button" data-v="${v}" class="${v === 5 ? "on" : ""}">${l}</button>`).join("")}</div></div>` +
      `<div class="setrow"><div><b>I approve every buy-in</b><small>Sit-downs and top-ups wait for your OK — you can trust regulars so they never wait</small></div><label class="switch"><input type="checkbox" id="c-approve"/><i></i></label></div>` +
      `<div class="setrow"><div><b>List in the lobby</b><small>Off = only people with the link can find it</small></div><label class="switch"><input type="checkbox" id="c-listed" checked/><i></i></label></div>` +
      `</div></details>`;
    segWire(body);
    const setMoney = (id, c) => { body.querySelector("#" + id).value = (c / 100).toFixed(2); };
    body.querySelector("#c-stakes").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-i]");
      if (!b) return;
      pick = Number(b.dataset.i);
      body.querySelectorAll(".stake").forEach((x) => x.classList.toggle("on", x === b));
      const s = STAKES[pick];
      setMoney("c-sb", s.sb); setMoney("c-bb", s.bb); setMoney("c-ante", s.ante); setMoney("c-buyin", s.buy);
    });
    openModal({
      title: "Host a table", sub: "You can change almost everything later from Manage table.", body,
      buttons: [
        { label: "Cancel", cls: "ghost" },
        {
          label: "Create table", cls: "gold",
          onClick: async () => {
            const q = (id) => body.querySelector("#" + id);
            const payload = {
              name: q("c-name").value.trim() || "Home game",
              sb_cents: C().toCents(q("c-sb").value), bb_cents: C().toCents(q("c-bb").value),
              ante_cents: C().toCents(q("c-ante").value), default_buyin_cents: C().toCents(q("c-buyin").value),
              min_buyin_cents: C().toCents(q("c-min").value) || 0, max_buyin_cents: C().toCents(q("c-max").value) || 0,
              num_seats: segVal(body, "c-seats"), decision_secs: segVal(body, "c-clock"),
              time_bank_secs: segVal(body, "c-bank"), deal_delay_secs: segVal(body, "c-deal"), listed: q("c-listed").checked,
              approve_buyins: q("c-approve").checked,
            };
            try {
              const s = await C().j("/games/api/tables", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
              await C().openTable(s.id, true);
              HG.sound && HG.sound.play("sit");
            } catch (e) { toast(e.message, "err"); return false; }
          },
        },
      ],
    });
  }
  function segWire(root) {
    root.querySelectorAll(".seg").forEach((seg) => seg.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-v]");
      if (!b || b.disabled) return;
      seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      seg.dispatchEvent(new CustomEvent("pick", { detail: b.dataset.v }));
    }));
  }
  function segVal(root, id) {
    const on = root.querySelector(`#${id} button.on`);
    return on ? Number(on.dataset.v) : null;
  }

  // ------------------------------------------------------- seat money dialogs
  function buyinLimits(s, stackCents) {
    const st = s.stakes, set = s.settings || {};
    const floor = Math.max(st.bb_cents, st.ante_cents + st.bb_cents);
    const lo = stackCents > 0 ? st.bb_cents : Math.max(floor, set.min_buyin_cents || 0);
    let hi = set.max_buyin_cents ? set.max_buyin_cents - stackCents : Math.max(st.default_buyin_cents * 5, lo * 4);
    hi = Math.max(lo, hi);
    return { lo, hi, capped: !!set.max_buyin_cents };
  }
  function moneyDialog(o) {
    const { lo, hi } = o;
    let cents = Math.max(lo, Math.min(hi, o.start));
    const body = h("div", { style: "display:flex;flex-direction:column;gap:14px" });
    body.innerHTML =
      `<div class="bigmoney"><span id="md-big"></span><small id="md-antes"></small></div>` +
      `<input type="range" id="md-range" min="${lo}" max="${hi}" step="${Math.max(1, Math.round(o.step || 100))}" value="${cents}"/>` +
      `<div class="sz-row"><div class="sz-presets" id="md-presets"></div><div class="sz-amt">${moneyInput("md-input", cents)}</div></div>` +
      `<small class="muted">${esc(o.hint || "")}</small>`;
    const big = body.querySelector("#md-big"), antes = body.querySelector("#md-antes"), range = body.querySelector("#md-range"), input = body.querySelector("#md-input");
    const presets = body.querySelector("#md-presets");
    const sync = (from) => {
      cents = Math.max(lo, Math.min(hi, cents));
      big.textContent = d2(cents);
      antes.textContent = o.ante ? `${Math.floor(cents / o.ante)} antes` : "";
      if (from !== "range") range.value = String(cents);
      if (from !== "input") input.value = (cents / 100).toFixed(2);
      range.style.setProperty("--fill", (hi > lo ? ((cents - lo) / (hi - lo)) * 100 : 100) + "%");
    };
    (o.presets || []).filter((p) => p.cents >= lo && p.cents <= hi).slice(0, 4).forEach((p) => {
      const b = h("button", { type: "button" }, esc(p.label));
      b.addEventListener("click", () => { cents = p.cents; sync(); });
      presets.appendChild(b);
    });
    range.addEventListener("input", () => { cents = Number(range.value); sync("range"); });
    input.addEventListener("input", () => { const c = C().toCents(input.value); if (c != null) { cents = c; big.textContent = d2(Math.max(lo, Math.min(hi, c))); } });
    input.addEventListener("change", () => sync());
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); sync(); go(); } });
    sync();
    let api = null;
    const go = async () => { sync(); try { await o.onOk(cents); if (api) api.close(true); } catch (_) { /* toasted */ } };
    api = openModal({ title: o.title, sub: o.sub, body, buttons: [{ label: "Cancel", cls: "ghost" }, { label: o.okLabel, cls: "primary", onClick: async () => { sync(); await o.onOk(cents); } }] });
  }
  function openSit(seat) {
    const s = C().G.state;
    if (!s) return;
    if (s.my_seat != null) return toast("You're already seated", "err");
    const lim = buyinLimits(s, 0), dflt = s.stakes.default_buyin_cents;
    moneyDialog({
      title: `Take seat ${seat + 1}`, sub: `${s.name} · ${d2(s.stakes.sb_cents)}/${d2(s.stakes.bb_cents)} · ante ${d2(s.stakes.ante_cents)}`,
      lo: lim.lo, hi: lim.hi, start: dflt, ante: s.stakes.ante_cents, step: s.stakes.bb_cents, okLabel: s.needs_approval ? "Request seat" : "Sit down",
      hint: (s.needs_approval ? "The host approves buy-ins here — your seat is held while they decide. " : "") + (lim.capped ? `Buy-in ${d2(lim.lo)} – ${d2(lim.hi)}. ` : "") + "Real money is settled between you — the ledger just keeps score.",
      presets: [{ label: "Min", cents: lim.lo }, { label: d2(dflt), cents: dflt }, { label: d2(dflt * 2), cents: dflt * 2 }, { label: "Max", cents: lim.hi }],
      onOk: async (cents) => {
        const out = await C().tablePost("sit", { seat, buyin_cents: cents });
        if (out && out.my_request) toast("Request sent — waiting for the host", "ok");
        else HG.sound && HG.sound.play("sit");
      },
    });
  }
  function openTopUp() {
    const s = C().G.state;
    if (!s || s.my_seat == null) return;
    const me = s.seats[s.my_seat];
    // Table stakes: chips asked for while you hold cards land when the hand ends.
    const holding = me.in_hand && (s.phase === "in_hand" || (s.runout && s.runout.blocking));
    const lim = buyinLimits(s, me.stack_cents);
    if (lim.capped && lim.hi < s.stakes.bb_cents) return toast(`You're at the table maximum (${d2(s.settings.max_buyin_cents)})`);
    const dflt = Math.max(lim.lo, Math.min(lim.hi, s.stakes.default_buyin_cents - me.stack_cents > 0 ? s.stakes.default_buyin_cents - me.stack_cents : s.stakes.default_buyin_cents));
    moneyDialog({
      title: "Add chips", sub: `Your stack is ${d2(me.stack_cents)}.` + (holding ? " They are added when this hand ends." : ""), lo: lim.lo, hi: lim.hi, start: dflt,
      ante: s.stakes.ante_cents, step: s.stakes.bb_cents, okLabel: s.needs_approval ? "Request chips" : "Add chips",
      hint: (s.needs_approval ? "The host approves buy-ins here. " : "") + (lim.capped ? `You can top up to ${d2(s.settings.max_buyin_cents)} in total.` : ""),
      presets: [{ label: d2(dflt), cents: dflt }, { label: d2(s.stakes.default_buyin_cents), cents: s.stakes.default_buyin_cents }, { label: "Max", cents: lim.hi }],
      onOk: async (cents) => {
        const out = await C().tablePost("rebuy", { amount_cents: cents, queue: true });
        if (out && out.my_request) toast("Request sent — waiting for the host", "ok");
        else if (holding) toast(`${d2(cents)} lands when this hand ends`, "ok");
        else HG.sound && HG.sound.play("chips");
      },
    });
  }
  function openAutoChips() {
    const s = C().G.state;
    if (!s || !Number.isInteger(s.my_seat)) return;
    const me = s.seats[s.my_seat];
    const topMode = s.auto_topup.mode, setMode = s.auto_stack.mode;
    const kinds = [["off", "Off"]];
    if (topMode === "player") kinds.push(["topup", "Auto top-up"]);
    if (setMode === "player") kinds.push(["set", "Set stack"]);
    const cur = setMode !== "off" && me.auto_stack_cents > 0 ? "set" : topMode !== "off" && me.topup_target_cents > 0 ? "topup" : "off";
    const hostLines = [];
    if (topMode === "host") hostLines.push(me.topup_target_cents ? `The host tops you up to ${d2(me.topup_target_cents)} whenever you drop below ${d2(me.topup_below_cents || me.topup_target_cents)}.` : "The host controls auto top-up (off for you).");
    if (setMode === "host") hostLines.push(me.auto_stack_cents ? `The host resets your stack to ${d2(me.auto_stack_cents)} before every hand.` : "The host controls set-stack (off for you).");
    const body = h("div", { style: "display:flex;flex-direction:column;gap:14px" });
    body.innerHTML =
      (hostLines.length ? `<div class="grp"><h4>Set by the host</h4><p style="margin:0">${hostLines.map(esc).join("<br>")}</p></div>` : "") +
      (kinds.length > 1
        ? `<div class="field"><span>Automatic chips</span><div class="seg" id="ac-kind">${kinds.map(([k, l]) => `<button type="button" data-v="${k}" class="${(kinds.some((x) => x[0] === cur) ? cur : "off") === k ? "on" : ""}">${l}</button>`).join("")}</div></div>` +
          `<div id="ac-top" hidden><div class="row2"><label class="field"><span>Top up to</span>${moneyInput("ac-top-target", me.topup_target_cents || s.stakes.default_buyin_cents)}</label>` +
          `<label class="field"><span>When below</span>${moneyInput("ac-top-below", me.topup_below_cents || me.topup_target_cents || s.stakes.default_buyin_cents)}</label></div>` +
          `<small class="muted">Before each hand, if your stack has dropped below the second amount it is topped back up to the first. It never takes chips off the table.</small></div>` +
          `<div id="ac-set" hidden><label class="field"><span>Stack every hand</span>${moneyInput("ac-set-target", me.auto_stack_cents || s.stakes.default_buyin_cents)}</label>` +
          `<small class="muted">Before EVERY hand your stack is reset to this amount — short stacks are topped up and anything above it goes back to your ledger. This table allows it.</small></div>`
        : (hostLines.length ? "" : `<p class="muted" style="margin:0">The host hasn't enabled automatic chips at this table.</p>`)) +
      (s.needs_approval ? `<small class="muted">The host approves buy-ins here: automatic chips only run once the host trusts you.</small>` : "");
    const showKind = (k) => { const a = body.querySelector("#ac-top"), b = body.querySelector("#ac-set"); if (a) a.hidden = k !== "topup"; if (b) b.hidden = k !== "set"; };
    segWire(body);
    const segEl = body.querySelector("#ac-kind");
    if (segEl) { segEl.addEventListener("pick", (e) => showKind(e.detail)); showKind(kinds.some((x) => x[0] === cur) ? cur : "off"); }
    openModal({
      title: "Automatic chips", body, autofocus: false,
      buttons: kinds.length > 1 ? [{ label: "Cancel", cls: "ghost" }, {
        label: "Save", cls: "primary",
        onClick: async () => {
          const on = body.querySelector("#ac-kind button.on");
          const kind = on ? on.dataset.v : "off";
          const payload = { kind };
          if (kind === "topup") { payload.target_cents = C().toCents(body.querySelector("#ac-top-target").value) || 0; payload.below_cents = C().toCents(body.querySelector("#ac-top-below").value) || 0; }
          if (kind === "set") payload.target_cents = C().toCents(body.querySelector("#ac-set-target").value) || 0;
          await C().tablePost("auto_chips_self", payload);
          toast(kind === "off" ? "Automatic chips off" : kind === "topup" ? `Topping up to ${d2(payload.target_cents)}` : `Stack resets to ${d2(payload.target_cents)} every hand`, "ok");
        },
      }] : [{ label: "Done", cls: "primary" }],
    });
  }
  // ------------------------------------------------------------ preferences
  function openPrefs() {
    const p = C().G.prefs;
    const seg = (id, opts, cur) => `<div class="seg" id="${id}">${opts.map(([v, l]) => `<button type="button" data-v="${v}" class="${String(v) === String(cur) ? "on" : ""}">${l}</button>`).join("")}</div>`;
    const sw = (id, on) => `<label class="switch"><input type="checkbox" id="${id}" ${on ? "checked" : ""}/><i></i></label>`;
    const body = h("div", { style: "display:flex;flex-direction:column;gap:14px" });
    body.innerHTML =
      `<div class="grp"><h4>Sound</h4><div class="setrow"><div><b>Table sounds</b><small>Cards, chips, your-turn chime</small></div>${sw("p-sound", p.sound)}</div>` +
      `<label class="field"><span>Volume</span><input type="range" id="p-vol" min="0" max="100" value="${Math.round(p.volume * 100)}" style="--fill:${Math.round(p.volume * 100)}%"/></label>` +
      `<div class="setrow"><div><b>Desktop notification on my turn</b><small>Only when this tab is in the background</small></div>${sw("p-notify", p.notify)}</div></div>` +
      `<div class="grp"><h4>Table</h4><div class="field"><span>Felt</span>${seg("p-felt", [["emerald", "Emerald"], ["royal", "Royal"], ["crimson", "Crimson"], ["violet", "Violet"], ["graphite", "Graphite"]], p.felt)}</div>` +
      `<div class="row2"><div class="field"><span>Cards</span>${seg("p-cards", [["bold", "Bold"], ["classic", "Classic"]], p.cards)}</div><div class="field"><span>Deck</span>${seg("p-deck", [["4c", "4-colour"], ["2c", "2-colour"]], p.deck)}</div></div>` +
      `<div class="row2"><div class="field"><span>Card backs</span>${seg("p-back", [["blue", "Blue"], ["red", "Red"], ["green", "Green"], ["black", "Black"]], p.back)}</div><div class="field"><span>Amounts</span>${seg("p-unit", [["dollars", "$"], ["bb", "BB"]], p.unit)}</div></div>` +
      `<div class="field"><span>Animations</span>${seg("p-anim", [["full", "Full"], ["off", "Off"]], p.anim)}</div>` +
      `<div class="setrow"><div><b>Chat bubbles over seats</b></div>${sw("p-bubbles", p.bubbles !== false)}</div></div>` +
      `<div class="grp"><h4>Playing</h4><div class="setrow"><div><b>Keyboard shortcuts</b><small>F fold · C check/call · R bet/raise · 1–5 sizes · ↑↓ adjust</small></div>${sw("p-hot", p.hotkeys)}</div>` +
      `<div class="setrow"><div><b>Confirm all-in</b><small>Ask before putting your whole stack in</small></div>${sw("p-allin", p.confirmAllIn)}</div></div>`;
    segWire(body);
    const save = (patch) => { C().savePrefs(patch); const s = C().G.state; if (s) HG.ui.render(s, s, { unitChanged: true }); renderSound(); };
    [["p-felt", "felt"], ["p-cards", "cards"], ["p-deck", "deck"], ["p-back", "back"], ["p-unit", "unit"], ["p-anim", "anim"]].forEach(([id, key]) =>
      body.querySelector("#" + id).addEventListener("pick", (e) => save({ [key]: e.detail })));
    [["p-sound", "sound"], ["p-bubbles", "bubbles"], ["p-hot", "hotkeys"], ["p-allin", "confirmAllIn"]].forEach(([id, key]) =>
      body.querySelector("#" + id).addEventListener("change", (e) => save({ [key]: e.target.checked })));
    body.querySelector("#p-vol").addEventListener("input", (e) => { e.target.style.setProperty("--fill", e.target.value + "%"); save({ volume: Number(e.target.value) / 100 }); });
    body.querySelector("#p-vol").addEventListener("change", () => HG.sound && HG.sound.play("chip"));
    body.querySelector("#p-notify").addEventListener("change", async (e) => {
      if (e.target.checked && globalThis.Notification && Notification.permission !== "granted") {
        const r = await Notification.requestPermission().catch(() => "denied");
        if (r !== "granted") { e.target.checked = false; toast("Notifications are blocked by the browser", "err"); }
      }
      save({ notify: e.target.checked });
    });
    openModal({ title: "Preferences", sub: "Saved on this device.", body, buttons: [{ label: "Done", cls: "primary" }], autofocus: false });
  }
  function renderSound() {
    const b = $("tb-sound"), on = !!C().G.prefs.sound;
    b.classList.toggle("on", on);
    b.innerHTML = icon(on ? "i-vol" : "i-mute");
  }

  // -------------------------------------------------------------------- rail
  function setRail(open, tab) {
    const p = C().G.prefs;
    if (tab) U.railTab = tab;
    C().savePrefs({ rail: open, railTab: U.railTab });
    $("rail").classList.toggle("closed", !open);
    $("tb-rail").classList.toggle("on", open);
    document.querySelectorAll("#rail-tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === U.railTab));
    for (const k of ["chat", "log", "ledger", "hands"]) $("panel-" + k).hidden = k !== U.railTab;
    if (open && U.railTab === "chat") { U.unread = 0; renderUnread(); const sc = $("chat-log").parentNode; sc.scrollTop = sc.scrollHeight; }
    if (p.rail !== open) setTimeout(() => HG.table.layout(), 300);
    const s = C().G.state;
    if (s && open) renderRail(s, true);
  }
  function renderUnread() {
    const n = U.unread;
    $("chat-badge").hidden = !n; $("chat-badge").textContent = n > 9 ? "9+" : String(n);
    $("rail-dot").hidden = !n || C().G.prefs.rail;
  }
  function timeOf(x) { const t = Date.parse(x); return Number.isFinite(t) ? t : 0; }
  function hhmm(ms) { const d = new Date(ms); return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`; }

  function renderRail(s, force) {
    const open = C().G.prefs.rail;
    // chat (+ dealer lines), always tracked so the unread badge works
    // clock auto-actions would flood the chat; the seat badges already show them
    const chat = s.chat || [], events = (s.events || []).filter((e) => e.kind !== "timeout");
    const sig = `${chat.length ? chat[chat.length - 1].id : 0}:${events.length ? events[events.length - 1].id : 0}:${chat.length}`;
    if (sig !== U.chatSig || force) {
      const firstPaint = U.chatSig === "";
      const prevLast = Number(U.chatSig.split(":")[0] || 0);
      U.chatSig = sig;
      const myName = (s.seats[s.my_seat] || {}).name;
      const items = chat.map((m) => ({ t: timeOf(m.created_at), chat: m })).concat(events.map((e) => ({ t: e.ts * 1000, ev: e })));
      items.sort((a, b) => a.t - b.t);
      const log = $("chat-log"), sc = log.parentNode;
      const stick = sc.scrollHeight - sc.scrollTop - sc.clientHeight < 40;
      log.innerHTML = items.length ? items.map((it) => it.chat
        ? `<div class="msg ${it.chat.name === myName ? "me" : ""}">${avatar(it.chat.name, it.chat.name, "sm")}<div class="body"><div class="who">${esc(it.chat.name)}<time>${it.t ? hhmm(it.t) : ""}</time></div><div class="txt">${esc(it.chat.text)}</div></div></div>`
        : `<div class="msg sys ${it.ev.kind === "win" ? "win" : ""}">${esc(it.ev.text)}</div>`).join("")
        : `<div class="muted" style="text-align:center;padding:28px 10px">Say hi — the dealer posts joins, rebuys and results here too.</div>`;
      if (stick || firstPaint) sc.scrollTop = sc.scrollHeight;
      if (!firstPaint) {
        const fresh = chat.filter((m) => m.id > prevLast && m.name !== myName).length;
        if (fresh && !(open && U.railTab === "chat")) { U.unread += fresh; HG.sound && HG.sound.play("msg"); }
      }
      renderUnread();
    }
    if (!open) return;
    if (U.railTab === "log") renderLog(s);
    if (U.railTab === "ledger") renderLedger(s, force);
    if (U.railTab === "hands") renderHands(s, force);
  }

  function renderLog(s) {
    const hist = s.history || [];
    const sig = `${s.hand_no}:${hist.length}:${C().G.prefs.unit}`;
    if (sig === U.logSig) return;
    U.logSig = sig;
    const names = {};
    s.seats.forEach((x) => { if (!x.empty) names[x.seat] = x.name; });
    let html = "", street = null;
    hist.forEach((x) => {
      if (x.street !== street) { street = x.street; html += `<div class="log-street">${esc(street)}</div>`; }
      const k = x.action === 0 ? "fold" : x.action === 1 ? (x.chips > 0 ? "call" : "check") : x.action === 7 ? "allin" : "raise";
      const lb = k === "fold" ? "Fold" : k === "check" ? "Check" : k === "call" ? "Call " + C().fmtAmt(x.cents, s) : (k === "allin" ? "All-in " : "Raise to ") + C().fmtAmt(x.to_cents, s);
      html += `<div class="log-row k-${k}">${avatar(names[x.seat] || "?", names[x.seat], "sm")}<span class="nm">${esc(names[x.seat] || "Seat " + (x.seat + 1))}</span><span class="lb">${lb}</span></div>`;
    });
    $("log-body").innerHTML = (s.hand_no ? `<div class="muted num" style="margin-bottom:10px">Hand #${s.hand_no} · ante ${d2(s.stakes.ante_cents)}</div>` : "") +
      (html || `<div class="muted" style="text-align:center;padding:28px 10px">${s.phase === "in_hand" ? "No action yet — everyone anted." : "The action of the current hand shows up here."}</div>`);
    const sc = $("log-body"); sc.scrollTop = sc.scrollHeight;
  }

  function settleUp(ledger) {
    const debt = ledger.filter((r) => r.net_cents < 0).map((r) => ({ n: r.name, c: -r.net_cents })).sort((a, b) => b.c - a.c);
    const cred = ledger.filter((r) => r.net_cents > 0).map((r) => ({ n: r.name, c: r.net_cents })).sort((a, b) => b.c - a.c);
    const out = [];
    let i = 0, k = 0;
    while (i < debt.length && k < cred.length) {
      const amt = Math.min(debt[i].c, cred[k].c);
      if (amt > 0) out.push({ from: debt[i].n, to: cred[k].n, cents: amt });
      debt[i].c -= amt; cred[k].c -= amt;
      if (!debt[i].c) i++;
      if (!cred[k].c) k++;
    }
    return out;
  }
  function renderLedger(s, force) {
    const led = s.ledger || [];
    const sig = JSON.stringify(led) + C().G.prefs.unit + (U.hands ? U.hands.statsSig : "");
    if (sig === U.ledgerSig && !force) return;
    U.ledgerSig = sig;
    const rows = led.map((r) =>
      `<tr class="${r.user_id === s.my_user_id ? "me" : ""} ${r.seated ? "" : "gone"}"><td title="${esc(r.name)}">${esc(r.name)}</td><td>${d2(r.buyin_cents)}</td><td>${r.seated ? d2(r.stack_cents) : d2(r.leftover_cents)}</td><td class="${r.net_cents > 0 ? "pos" : r.net_cents < 0 ? "neg" : ""}">${r.net_cents > 0 ? "+" : ""}${d2(r.net_cents)}</td></tr>`).join("");
    const pays = settleUp(led);
    const stats = (U.hands && U.hands.stats) || [];
    $("ledger-body").innerHTML =
      `<table class="ledger"><thead><tr><th>Player</th><th>Buy-in</th><th>Stack</th><th>Net</th></tr></thead><tbody>${rows}</tbody></table>` +
      `<div class="muted" style="font-size:11.5px;margin-top:8px">Stacks settle at the end of each hand. Players who left show what they cashed out.</div>` +
      `<div class="rsec"><h4>Settle up</h4>${pays.length ? pays.map((p) => `<div class="settle-row"><span>${esc(p.from)} pays ${esc(p.to)}</span><b>${d2(p.cents)}</b></div>`).join("") : '<div class="muted">Everyone is even.</div>'}` +
      (pays.length ? `<button class="btn sm block" id="settle-copy" style="margin-top:10px">${icon("i-copy", "sm")}Copy summary</button>` : "") + `</div>` +
      (stats.length ? `<div class="rsec"><h4>Session stats</h4><table class="ledger"><thead><tr><th>Player</th><th>Hands</th><th>Won</th><th>Best</th><th title="Average score of their decisions against the network (0-100)">Acc.</th></tr></thead><tbody>` +
        stats.map((r) => `<tr class="${r.is_me ? "me" : ""}"><td>${esc(r.name)}</td><td>${r.hands}</td><td>${r.wins}</td><td>${d2(r.biggest_win_cents)}</td><td>${r.accuracy == null ? "–" : Math.round(r.accuracy) + "%"}</td></tr>`).join("") + `</tbody></table>` +
        `<div class="muted" style="font-size:11.5px;margin-top:6px">Accuracy = how closely each decision matched the network, graded in the background after every hand.</div></div>` : "") +
      (((U.hands && U.hands.h2h) || []).length ? `<div class="rsec"><h4>Head to head — this table</h4>` + U.hands.h2h.map((x) => `<div class="settle-row"><span>${esc(x.to)} is up on ${esc(x.from)}</span><b>${d2(x.cents)}</b></div>`).join("") + `</div>` : "") +
      `<button class="btn sm block" id="ledger-myhands" style="margin-top:14px">${icon("i-chart", "sm")}My hands &amp; lifetime stats</button>`;
    const mh = $("ledger-myhands");
    if (mh) mh.addEventListener("click", () => openMyHands(""));
    const cp = $("settle-copy");
    if (cp) cp.addEventListener("click", async () => {
      const text = `${s.name} — settle up\n` + led.map((r) => `${r.name}: ${r.net_cents >= 0 ? "+" : ""}${d2(r.net_cents)}`).join("\n") + "\n\n" + pays.map((p) => `${p.from} pays ${p.to} ${d2(p.cents)}`).join("\n");
      try { await navigator.clipboard.writeText(text); toast("Summary copied", "ok"); } catch (_) { toast("Couldn't copy", "err"); }
    });
    if (!U.hands && !U.handsLoading && s.is_member) loadHands(s);
  }

  async function loadHands(s) {
    if (U.handsLoading) return;
    U.handsLoading = true;
    try {
      const data = await C().j(`/games/api/tables/${s.id}/hands?limit=40`);
      data.statsSig = JSON.stringify([data.stats || [], data.h2h || []]);
      U.hands = data; U.handsFor = `${s.id}:${s.last_hand_no}`;
    } catch (_) { U.hands = { hands: [], stats: [], statsSig: "", denied: true }; U.handsFor = `${s.id}:${s.last_hand_no}`; }
    U.handsLoading = false;
    const cur = C().G.state;
    if (cur && cur.id === s.id) renderRail(cur, true);
  }
  function miniCards(list, extra) {
    return `<span class="mini-cards ${extra || ""}" data-cards="${(list || []).join(",")}"></span>`;
  }
  function fillMiniCards(root) {
    root.querySelectorAll(".mini-cards[data-cards]").forEach((m) => {
      if (m.childElementCount) return;
      (m.dataset.cards ? m.dataset.cards.split(",") : []).forEach((c) => m.appendChild(HG.cards.cardEl(c === "" || c === "x" ? -1 : Number(c))));
    });
  }
  function renderHands(s, force) {
    const key = `${s.id}:${s.last_hand_no}`;
    const body = $("hands-body");
    if (!s.is_member) { body.innerHTML = `<div class="muted" style="text-align:center;padding:28px 10px">Hand history is for players at this table. Take a seat to see it.</div>`; return; }
    if (U.handsFor !== key && !U.handsLoading) { loadHands(s); if (!U.hands) body.innerHTML = `<div class="muted" style="text-align:center;padding:28px 10px">Loading hands…</div>`; return; }
    if (!U.hands || (!force && body.dataset.k === key + C().G.prefs.unit)) return;
    body.dataset.k = key + C().G.prefs.unit;
    const list = U.hands.hands || [];
    body.innerHTML = list.length ? "" : `<div class="muted" style="text-align:center;padding:28px 10px">No finished hands yet. Every hand you play is saved here — your cards, the boards and who won.</div>`;
    list.forEach((x) => {
      const net = x.my_delta_cents;
      const row = h("button", { class: "hand-row", type: "button" },
        `<span class="no">#${x.hand_no}</span><span style="display:flex;align-items:center;min-width:0">${x.my_hole ? miniCards(x.my_hole) : ""}${miniCards(x.board_a, "gap")}</span>` +
        `<span class="net ${net > 0 ? "pos" : net < 0 ? "neg" : "muted"}">${net == null ? "—" : (net > 0 ? "+" : "") + d2(net)}</span>` +
        `<span></span><span class="who">${esc((x.winners || []).map((w) => w.name).join(", ") || "Split pot")} · pot ${d2(x.pot_cents)}</span><span class="muted num" style="font-size:11px">${x.my_accuracy == null ? (x.showdown ? "Showdown" : "") : Math.round(x.my_accuracy) + "%"}</span>`);
      row.addEventListener("click", () => openHand(s.id, x.hand_no));
      body.appendChild(row);
    });
    fillMiniCards(body);
    const all = h("button", { class: "btn sm block", type: "button", style: "margin-top:10px" }, icon("i-chart", "sm") + "All my hands (every table)");
    all.addEventListener("click", () => openMyHands(""));
    body.appendChild(all);
  }
  // ------------------------------------------------------------ hand replayer
  // A CLICK-THROUGH, not a video: the hand opens on the flop with the first
  // player to act; forward plays one action, back takes one away. Every player
  // decision carries the network's verdict (same marks as the Trainer), and any
  // position can be sent to the Study tab as a spot.
  const MARKS = { best: "✓✓", correct: "✓", inaccuracy: "~", wrong: "✗", blunder: "✗✗" };
  const MARK_LABEL = { best: "Best move", correct: "Correct", inaccuracy: "Inaccuracy", wrong: "Wrong move", blunder: "Blunder" };
  const kindOfAction = (x) => (x.action === 0 ? "fold" : x.action === 1 ? (x.chips > 0 ? "call" : "check") : x.action === 7 ? "allin" : "raise");
  function gradeChip(g, small) {
    if (!g) return "";
    return `<span class="grade g-${g.cat}${small ? " sm" : ""}" title="${MARK_LABEL[g.cat] || g.cat} · ${Math.round(g.score)}/100 vs the network">${MARKS[g.cat] || "?"}${small ? "" : ` <em>${MARK_LABEL[g.cat] || g.cat}</em> <b>${Math.round(g.score)}</b>`}</span>`;
  }

  function replayState(rec, k) {
    const seats = {};
    rec.seats.forEach((x) => { seats[x.seat] = { stack: x.start_cents - rec.ante_cents, bet: 0, folded: false }; });
    let pot = rec.ante_cents * rec.seats.length, street = "flop";
    const clearBets = () => Object.values(seats).forEach((p) => { p.bet = 0; });
    for (let i = 0; i < k; i++) {
      const a = rec.actions[i];
      if (String(a.street).toLowerCase() !== street) { street = String(a.street).toLowerCase(); clearBets(); }
      const p = seats[a.seat];
      if (!p) continue;
      if (a.action === 0) p.folded = true;
      else { p.stack -= a.cents; p.bet += a.cents; pot += a.cents; }
    }
    const next = rec.actions[k] || null;
    if (next && String(next.street).toLowerCase() !== street) { street = String(next.street).toLowerCase(); clearBets(); }
    const over = !next;
    if (over) clearBets();
    const boardN = over ? Math.max(3, (rec.board_a || []).length) : street === "river" ? 5 : street === "turn" ? 4 : 3;
    return { seats, pot, street, next, over, boardN, last: k > 0 ? rec.actions[k - 1] : null };
  }

  async function openHand(gid, no) {
    let rec;
    try { rec = await C().j(`/games/api/tables/${gid}/hands/${no}`); } catch (e) { return toast(e.message, "err"); }
    const N = (rec.actions || []).length;
    const gradeAt = {};
    (rec.grades || []).forEach((g) => { gradeAt[g.i] = g; });
    const names = {};
    rec.seats.forEach((x) => (names[x.seat] = x.is_me ? "You" : x.name));
    const hero = (rec.seats.find((x) => x.is_me) || rec.seats[0] || {}).seat || 0;
    const order = rec.seats.map((x) => x.seat);
    const n = rec.num_seats || 8;
    let k = 0;
    const body = h("div", { class: "rp" });
    body.innerHTML =
      `<div class="rp-main"><div class="rp-felt" id="rp-felt"></div>` +
      `<div class="rp-banner" id="rp-banner"></div>` +
      `<div class="rp-ctl"><button class="btn sm" id="rp-first" title="Start of the hand (Home)">⏮</button><button class="btn" id="rp-prev" title="Back one action (←)">◀</button>` +
      `<span class="rp-step num" id="rp-step"></span><button class="btn primary" id="rp-next" title="Play the next action (→)">▶</button><button class="btn sm" id="rp-last" title="End of the hand (End)">⏭</button>` +
      `<span class="spacer"></span>${rec.fair && HG.fair ? `<button class="btn sm" id="rp-fair" title="Re-check this hand's sealed deck, its cut and every card you can see">${icon("i-shield", "sm")}<span>Check shuffle</span></button>` : ""}` +
      `<button class="btn gold sm" id="rp-study" title="Send this exact spot to the Study tab">${icon("i-chart", "sm")}Open in Study</button></div></div>` +
      `<div class="rp-side"><div class="rsec" style="margin-top:0"><h4>Action</h4><div id="rp-list" class="rp-list"></div></div><div id="rp-result"></div></div>`;
    const q = (id) => body.querySelector("#" + id);

    function paint() {
      const st = replayState(rec, k);
      // seats around the felt, hero at the bottom
      let html = "";
      rec.seats.forEach((x) => {
        const rel = (x.seat - hero + n) % n;
        const th = Math.PI / 2 + (rel * 2 * Math.PI) / n;
        const px = 50 + 44 * Math.cos(th), py = 50 + 40 * Math.sin(th);
        const p = st.seats[x.seat];
        const acting = st.next && st.next.seat === x.seat;
        const known = x.hole && x.hole.length && x.hole[0] >= 0;
        const cards = p.folded && !known ? "" : `<span class="mini-cards" data-cards="${known ? x.hole.join(",") : "x,x,x,x,x"}"></span>`;
        const res = st.over ? `<b class="num ${x.delta_cents > 0 ? "pos" : x.delta_cents < 0 ? "neg" : "muted"}">${x.delta_cents > 0 ? "+" : ""}${d2(x.delta_cents)}</b>` : "";
        html += `<div class="rp-seat ${p.folded ? "folded" : ""} ${acting ? "acting" : ""}" style="left:${px}%;top:${py}%">${cards}` +
          `<div class="rp-plate">${avatar(x.name, x.name, "sm")}<div><b>${esc(names[x.seat])}${x.seat === rec.button ? ' <i class="rp-d">D</i>' : ""}</b><span class="num">${d2(Math.max(0, p.stack))}</span></div></div>` +
          (p.bet > 0 ? `<span class="rp-bet num">${d2(p.bet)}</span>` : "") + res + `</div>`;
      });
      html += `<div class="rp-center"><span class="rp-pot num">Pot ${d2(st.pot)}</span>` +
        `<span class="mini-cards" data-cards="${(rec.board_a || []).slice(0, st.boardN).join(",")}"></span>` +
        `<span class="mini-cards" data-cards="${(rec.board_b || []).slice(0, st.boardN).join(",")}"></span></div>`;
      const felt = q("rp-felt");
      felt.innerHTML = html;
      fillMiniCards(felt);
      // banner: the action just played (with its verdict) and who is up
      const last = st.last, g = last ? gradeAt[k - 1] : null;
      q("rp-banner").innerHTML =
        (last ? `<span class="rp-act k-${kindOfAction(last)}"><b>${esc(names[last.seat] || "?")}</b> ${esc(last.label)}${last.auto ? ' <small class="muted">(clock)</small>' : ""}</span>${gradeChip(g)}` : `<span class="muted">Flop dealt — everyone anted ${d2(rec.ante_cents)}.</span>`) +
        `<span class="spacer"></span>` +
        (st.next ? `<span class="muted">${esc(String(st.street).toUpperCase())} · <b style="color:var(--tx)">${esc(names[st.next.seat] || "?")}</b> to act</span>` : `<span class="pill gold">Hand over</span>`);
      q("rp-step").textContent = `${k} / ${N}`;
      q("rp-prev").disabled = q("rp-first").disabled = k === 0;
      q("rp-next").disabled = q("rp-last").disabled = k === N;
      body.querySelectorAll("#rp-list .log-row").forEach((r) => r.classList.toggle("on", Number(r.dataset.i) === k - 1));
      const cur = body.querySelector("#rp-list .log-row.on");
      if (cur) cur.scrollIntoView({ block: "nearest" });
      q("rp-result").hidden = !st.over;
    }
    function go(to) { k = Math.max(0, Math.min(N, to)); paint(); }

    // action list (click = jump to just after that action)
    let list = "", street = null;
    (rec.actions || []).forEach((a2, i) => {
      if (a2.street !== street) { street = a2.street; list += `<div class="log-street">${esc(street)}</div>`; }
      list += `<button type="button" class="log-row k-${kindOfAction(a2)}" data-i="${i}"><span class="nm">${esc(names[a2.seat] || "?")}</span><span class="lb">${esc(a2.label)}</span>${gradeChip(gradeAt[i], true)}</button>`;
    });
    q("rp-list").innerHTML = list || `<div class="muted">No betting — everyone was all-in from the ante.</div>`;
    q("rp-list").addEventListener("click", (e) => { const r = e.target.closest(".log-row"); if (r) go(Number(r.dataset.i) + 1); });
    const awards = (rec.awards || []).map((w) => {
      const who = w.winners.map((x) => names[x] || "?").join(" & ");
      const lab = w.winners.length === 1 && w.labels && w.labels[String(w.winners[0])] ? ` with ${w.labels[String(w.winners[0])]}` : "";
      return `<div class="settle-row"><span>${w.uncontested ? "Uncontested" : "Board " + (w.board === "b" ? 2 : 1)} · ${esc(who)}${esc(lab)}</span><b>${d2(w.cents)}</b></div>`;
    }).join("");
    const flows = (rec.flows || []).map((f) => `<div class="settle-row"><span>${esc(names[f.from] || "?")} → ${esc(names[f.to] || "?")}</span><b>${d2(f.cents)}</b></div>`).join("");
    q("rp-result").innerHTML = (awards ? `<div class="rsec"><h4>Pots</h4>${awards}</div>` : "") + (flows ? `<div class="rsec"><h4>Who paid whom</h4>${flows}</div>` : "");
    q("rp-first").addEventListener("click", () => go(0));
    q("rp-prev").addEventListener("click", () => go(k - 1));
    q("rp-next").addEventListener("click", () => go(k + 1));
    q("rp-last").addEventListener("click", () => go(N));
    q("rp-study").addEventListener("click", () => openInStudy(rec, k));
    const fairBtn = q("rp-fair");
    if (fairBtn) fairBtn.addEventListener("click", async () => {
      fairBtn.disabled = true;
      try {
        const r = await HG.fair.checkPast(gid, no);
        fairBtn.classList.add("ok");
        fairBtn.lastElementChild.textContent = r.contributors ? `Verified · cut by ${r.contributors} device${r.contributors === 1 ? "" : "s"}${r.mine ? " (yours too)" : ""}` : "Sealed · nobody's device cut it";
        toast(`Sealed deck, cut and ${r.cards} card${r.cards === 1 ? "" : "s"} check out`, "ok");
      } catch (e) {
        fairBtn.classList.add("bad");
        fairBtn.lastElementChild.textContent = "CHECK FAILED";
        toast("This hand's shuffle does not check out: " + e.message, "err", 9000);
      }
      fairBtn.disabled = false;
    });
    const onKey = (e) => {
      if (U.modals[U.modals.length - 1] !== api) return;
      if (e.key === "ArrowRight") { e.preventDefault(); go(k + 1); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); go(k - 1); }
      else if (e.key === "Home") { e.preventDefault(); go(0); }
      else if (e.key === "End") { e.preventDefault(); go(N); }
    };
    document.addEventListener("keydown", onKey);
    const graded = (rec.grades || []).length;
    const api = openModal({
      title: `Hand #${rec.hand_no}${rec.table_name ? " · " + rec.table_name : ""}`,
      sub: `Pot ${d2(rec.pot_cents)} · ante ${d2(rec.ante_cents)} · ${rec.showdown ? "showdown" : "won without showdown"}` +
        (rec.grades == null ? " · accuracy is still being worked out" : graded ? "" : " · no graded decisions") + " · use ← → to step",
      body, wide: true, autofocus: false, buttons: [{ label: "Close", cls: "primary" }],
      onClose: () => document.removeEventListener("keydown", onKey),
    });
    api.modal.classList.add("xwide");
    paint();
  }

  // The Study tab works on the signed-in user's own server-side session, so a
  // spot is "copied" by driving Study's normal API: reset, stakes, seats + stacks
  // (hero = seat 0, clockwise), cards dealt so far, then the actions up to here.
  async function openInStudy(rec, k) {
    const dealt = rec.seats.map((x) => x.seat);
    if (dealt.length > 6) return toast(`Study handles up to 6 players — this hand had ${dealt.length}`, "err");
    const N = rec.actions.length;
    const known = (seat) => { const x = rec.seats.find((y) => y.seat === seat); return x && x.hole && x.hole[0] >= 0 ? x : null; };
    const actor = (rec.actions[Math.min(k, N - 1)] || {}).seat;
    const me = rec.seats.find((x) => x.is_me);
    const heroSeat = actor != null && known(actor) ? actor : me && known(me.seat) ? me.seat : (rec.seats.find((x) => known(x.seat)) || { seat: actor != null ? actor : dealt[0] }).seat;
    const n = rec.num_seats || 8;
    const order = dealt.slice().sort((x, y) => ((x - heroSeat + n) % n) - ((y - heroSeat + n) % n));
    const st = replayState(rec, k);
    const heroRec = known(heroSeat);
    const pad = (arr, len) => { const out = arr.slice(0, len); while (out.length < len) out.push(null); return out; };
    const cards = {
      hero_hole: heroRec ? heroRec.hole.slice(0, 5) : [null, null, null, null, null],
      flop_a: pad(rec.board_a || [], 3), flop_b: pad(rec.board_b || [], 3),
      turn: st.boardN >= 4 ? [(rec.board_a || [])[3] ?? null, (rec.board_b || [])[3] ?? null] : [null, null],
      river: st.boardN >= 5 ? [(rec.board_a || [])[4] ?? null, (rec.board_b || [])[4] ?? null] : [null, null],
    };
    const post = (url, b) => C().j(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b || {}) });
    const btn = document.getElementById("rp-study");
    if (btn) btn.disabled = true;
    // Study opens in a NEW TAB so the replayer keeps its place. The tab has to be
    // opened right here, inside the click — after the awaits below a browser
    // treats window.open as a pop-up and blocks it.
    let tab = null;
    try {
      tab = globalThis.open("", "_blank");
      if (tab) {
        tab.document.title = "Opening in Study…";
        tab.document.body.style.cssText = "margin:0;height:100vh;display:grid;place-items:center;background:#070b11;color:#8794a6;font:14px system-ui,sans-serif";
        tab.document.body.textContent = "Copying the spot into Study…";
      }
    } catch (_) { /* a blocked or cross-origin handle: fall through to the link below */ }
    try {
      await post("/format", { format: "plo5_double_bomb" }).catch(() => null);
      await post("/reset");
      // (hands recorded before 2026-09-23 carry cents only)
      const bbChips = rec.bb_chips || 10000;
      const toChips = (cents) => Math.round((cents * bbChips) / (rec.bb_cents || 100));
      await post("/config", { bb_chips: bbChips, ante_chips: rec.ante_chips != null ? rec.ante_chips : toChips(rec.ante_cents), dollars_per_bb: rec.bb_cents / 100 });
      await post("/seats", {
        num_seats: order.length, button_seat: Math.max(0, order.indexOf(rec.button)),
        starting_stacks: order.map((seat) => { const x = rec.seats.find((y) => y.seat === seat); return x.start_chips != null ? x.start_chips : toChips(x.start_cents); }), stacks_are_starting: true,
      });
      await post("/cards", cards);
      for (let i = 0; i < k; i++) {
        const a2 = rec.actions[i];
        const gate = a2.action === 0 ? "fold" : a2.action === 1 ? "check_call" : "raise";
        await post("/action", gate === "raise" ? { gate, chips: a2.chips } : { gate });
      }
      if (btn) btn.disabled = false;
      if (tab && !tab.closed) { try { tab.opener = null; } catch (_) { /* fine */ } tab.location.replace("/?mode=study"); toast("Opened in Study (new tab)", "ok"); }
      else {
        // pop-ups blocked: the spot IS loaded — a plain link click is always allowed
        openModal({ title: "Spot copied to Study", sub: "Your browser blocked the new tab. The spot is loaded — open Study with this link.", body: `<a class="btn gold block" href="/?mode=study" target="_blank" rel="noopener">Open Study in a new tab</a>`, buttons: [{ label: "Done", cls: "primary" }] });
      }
    } catch (e) {
      if (btn) btn.disabled = false;
      if (tab && !tab.closed) tab.close();
      toast(e.status === 402 ? "Study needs a subscription on this account" : "Couldn't copy the spot: " + e.message, "err");
    }
  }

  // --------------------------------------------------- lifetime hand database
  // `player` = {user_id, name} opens somebody else's database (the club is private:
  // everyone may browse everyone — cards still follow the table's reveal rule).
  async function openMyHands(gameId, player) {
    const other = player && !player.is_me ? player : null;
    const base = other ? `/games/api/players/${other.user_id}` : "/games/api/my";
    let stats;
    try { stats = await C().j(base + "/stats"); } catch (e) { return toast(e.message, "err"); }
    const F = { sort: "time", dir: "desc", game: gameId || "", offset: 0, rows: [], total: 0 };
    const acc = (v) => (v == null ? "–" : Math.round(v) + "%");
    const body = h("div", { class: "db" });
    body.innerHTML =
      `<div class="pcard-stats"><div><b>${stats.hands}</b><small>Hands</small></div><div><b class="${stats.net_cents > 0 ? "pos" : stats.net_cents < 0 ? "neg" : ""}">${stats.net_cents > 0 ? "+" : ""}${d2(stats.net_cents)}</b><small>Lifetime net</small></div>` +
      `<div><b>${acc(stats.accuracy)}</b><small>Accuracy (${stats.graded} decisions)</small></div><div><b>${stats.hands ? Math.round((100 * stats.wins) / stats.hands) + "%" : "–"}</b><small>Hands won</small></div></div>` +
      ((stats.versus || []).length ? `<div class="rsec"><h4>Head to head — lifetime</h4><div class="vs">${stats.versus.map((v) => `<span class="vs-chip"><span>${esc(v.name)}</span><b class="num ${v.net_cents > 0 ? "pos" : v.net_cents < 0 ? "neg" : ""}">${v.net_cents > 0 ? "+" : ""}${d2(v.net_cents)}</b></span>`).join("")}</div></div>` : "") +
      `<div class="db-bar"><select class="input" id="db-game"><option value="">All sessions (${(stats.sessions || []).length})</option>${(stats.sessions || []).map((x) => `<option value="${esc(x.id)}" ${x.id === F.game ? "selected" : ""}>${esc(x.name)} · ${x.hands} hands · ${x.net_cents >= 0 ? "+" : ""}${d2(x.net_cents)}${x.accuracy == null ? "" : " · " + acc(x.accuracy)}</option>`).join("")}</select>` +
      `<div class="seg" id="db-sort">${[["time", "Date"], ["pot", "Pot size"], ["net", "Profit / loss"], ["accuracy", "Accuracy"]].map(([v, l]) => `<button type="button" data-v="${v}" class="${v === "time" ? "on" : ""}">${l}</button>`).join("")}</div>` +
      `<button class="btn sm" id="db-dir" title="Reverse the order">↓ High to low</button></div>` +
      `<div id="db-list"></div><button class="btn block" id="db-more" hidden>Load more</button>`;
    const q = (id) => body.querySelector("#" + id);
    const draw = () => {
      const host = q("db-list");
      host.innerHTML = F.rows.length ? "" : `<div class="muted" style="text-align:center;padding:26px">${other ? "No hands here yet." : "No hands yet. Every hand you play at any table lands here."}</div>`;
      F.rows.forEach((x) => {
        const net = x.net_cents;
        const row = h("button", { class: "hand-row db-row", type: "button" },
          `<span class="no">#${x.hand_no}</span><span style="display:flex;align-items:center;min-width:0">${x.my_hole ? miniCards(x.my_hole) : ""}${miniCards(x.board_a, "gap")}${miniCards(x.board_b, "gap")}</span>` +
          `<span class="net ${net > 0 ? "pos" : net < 0 ? "neg" : "muted"}">${net > 0 ? "+" : ""}${d2(net)}</span>` +
          `<span></span><span class="who">${esc(x.table_name)} · ${esc(String(x.ended_at || "").slice(0, 10))} · pot ${d2(x.pot_cents)}${x.showdown ? " · showdown" : ""}</span><span class="muted num" style="font-size:11.5px">${x.accuracy == null ? "" : acc(x.accuracy)}</span>`);
        row.addEventListener("click", () => openHand(x.game_id, x.hand_no));
        host.appendChild(row);
      });
      fillMiniCards(host);
      q("db-more").hidden = F.rows.length >= F.total;
      q("db-dir").textContent = F.dir === "desc" ? "↓ High to low" : "↑ Low to high";
    };
    const load = async (reset) => {
      if (reset) { F.offset = 0; F.rows = []; }
      try {
        const d = await C().j(`${base}/hands?sort=${F.sort}&dir=${F.dir}&limit=40&offset=${F.offset}` + (F.game ? `&game=${encodeURIComponent(F.game)}` : ""));
        F.rows = F.rows.concat(d.hands); F.total = d.total; F.offset += d.limit;
      } catch (e) { toast(e.message, "err"); }
      draw();
    };
    segWire(body);
    q("db-sort").addEventListener("pick", (e) => { F.sort = e.detail; load(true); });
    q("db-dir").addEventListener("click", () => { F.dir = F.dir === "desc" ? "asc" : "desc"; load(true); });
    q("db-game").addEventListener("change", (e) => { F.game = e.target.value; load(true); });
    q("db-more").addEventListener("click", () => load(false));
    const api = openModal({
      title: other ? `${other.name} — hands & stats` : "My hands",
      sub: other ? "Every hand they have played. You see the cards you saw at the table: your own, and hands that were shown." : "Every hand you have played at any table, with the network's accuracy rating.",
      body, wide: true, autofocus: false, buttons: [{ label: "Close", cls: "primary" }],
    });
    api.modal.classList.add("xwide");
    load(true);
  }

  // ------------------------------------------------------------ the club: everyone
  const MIN_RANKED = 20; // graded decisions before an accuracy counts for the podium
  const accTxt = (v) => (v == null ? "–" : Math.round(v) + "%");
  const signed = (c) => (c > 0 ? "+" : c < 0 ? "−" : "") + d2(Math.abs(c));
  const tone = (c) => (c > 0 ? "pos" : c < 0 ? "neg" : "");

  async function loadClub(force) {
    if (!force && U.clubAt && Date.now() - U.clubAt < 30000) return;
    U.clubAt = Date.now();
    try { renderClub(await C().j("/games/api/community")); } catch (_) { /* the lobby works without it */ }
  }
  function renderClub(data) {
    U.club = data;
    const players = data.players || [];
    $("lb-club-sec").hidden = !players.length;
    if (!players.length) return;
    // podium: accuracy, among players with enough graded decisions to mean something
    const rated = players.filter((p) => p.accuracy != null);
    const ranked = rated.filter((p) => p.graded >= MIN_RANKED).sort((a, b) => b.accuracy - a.accuracy || b.graded - a.graded);
    const early = rated.filter((p) => p.graded < MIN_RANKED).sort((a, b) => b.accuracy - a.accuracy || b.graded - a.graded);
    const top = ranked.concat(early).slice(0, 3);
    const pod = $("lb-podium");
    pod.hidden = !top.length;
    const step = (p, place) => !p ? `<div class="pod-col p${place} empty"><div class="pod-step"><b>${place}</b></div></div>` :
      `<button type="button" class="pod-col p${place}" data-uid="${p.user_id}" title="Open ${esc(p.name)}'s hands">` +
      `${place === 1 ? `<span class="pod-crown">${icon("i-crown")}</span>` : ""}${avatar(p.name, p.name, "lg")}` +
      `<span class="pod-name">${esc(p.name)}${p.is_me ? " <i>you</i>" : ""}</span>` +
      `<span class="pod-acc num">${accTxt(p.accuracy)}</span>` +
      `<span class="pod-sub">${p.graded} decision${p.graded === 1 ? "" : "s"}${p.graded < MIN_RANKED ? " · provisional" : ""}</span>` +
      `<span class="pod-step"><b>${place}</b></span></button>`;
    pod.innerHTML = step(top[1], 2) + step(top[0], 1) + step(top[2], 3);
    // one card per player, biggest winner first
    $("lb-players").innerHTML = players.map((p) => {
      const pct = p.accuracy == null ? 0 : Math.max(0, Math.min(100, p.accuracy));
      return `<button type="button" class="plcard ${p.is_me ? "me" : ""}" data-uid="${p.user_id}">` +
        `<span class="plcard-top">${avatar(p.name, p.name)}<span class="plcard-name"><b>${esc(p.name)}</b><small>${p.hands} hand${p.hands === 1 ? "" : "s"} · ${p.sessions} session${p.sessions === 1 ? "" : "s"}</small></span>` +
        `<b class="num plcard-net ${tone(p.net_cents)}">${signed(p.net_cents)}</b></span>` +
        `<span class="plcard-acc"><span class="plcard-meter"><i style="width:${pct}%"></i></span><b class="num">${accTxt(p.accuracy)}</b></span>` +
        `<span class="plcard-foot"><span>Accuracy${p.graded ? ` · ${p.graded} decisions` : " · not rated yet"}</span><span>Won ${p.hands ? Math.round((100 * p.wins) / p.hands) : 0}% · best ${d2(p.best_cents)}</span></span></button>`;
    }).join("");
    $("lb-club-sub").textContent = `${players.length} player${players.length === 1 ? "" : "s"} · accuracy is the network's rating of every decision`;
    document.querySelectorAll("#lb-podium [data-uid], #lb-players [data-uid]").forEach((b) => b.addEventListener("click", () => {
      const p = players.find((x) => String(x.user_id) === b.dataset.uid);
      if (p) openMyHands("", { user_id: p.user_id, name: p.name, is_me: p.is_me });
    }));
  }

  // who is up on whom: row = the player, column = the opponent, cell = what the
  // row player has won from (+) or lost to (−) that opponent, all sessions
  function openMatrix() {
    const data = U.club;
    if (!data || !(data.players || []).length) return toast("No hands played yet", "");
    const ps = data.players.filter((p) => (data.pairs || []).some((x) => x.from === p.user_id || x.to === p.user_id));
    if (ps.length < 2) return toast("No money has changed hands yet", "");
    const net = {};
    (data.pairs || []).forEach((x) => { net[x.to + ":" + x.from] = x.cents; net[x.from + ":" + x.to] = -x.cents; });
    const peak = Math.max(1, ...Object.values(net).map(Math.abs));
    const cell = (v) => {
      if (!v) return `<td class="mx-zero">·</td>`;
      const a = 0.1 + 0.34 * Math.min(1, Math.abs(v) / peak);
      return `<td class="num ${tone(v)}" style="background:rgba(${v > 0 ? "53,200,120" : "242,86,106"},${a.toFixed(2)})">${signed(v)}</td>`;
    };
    const body = h("div", { class: "mx-wrap" });
    body.innerHTML = `<table class="mx"><thead><tr><th class="mx-corner">won from →</th>${ps.map((p) => `<th title="${esc(p.name)}">${avatar(p.name, p.name, "sm")}<span>${esc(p.name)}</span></th>`).join("")}<th class="mx-total">Total</th></tr></thead><tbody>` +
      ps.map((r) => {
        const total = ps.reduce((acc, c) => acc + (net[r.user_id + ":" + c.user_id] || 0), 0);
        return `<tr><th data-uid="${r.user_id}" title="Open ${esc(r.name)}'s hands">${avatar(r.name, r.name, "sm")}<span>${esc(r.name)}${r.is_me ? " (you)" : ""}</span></th>` +
          ps.map((c) => (c.user_id === r.user_id ? `<td class="mx-self"></td>` : cell(net[r.user_id + ":" + c.user_id] || 0))).join("") +
          `<td class="num mx-total ${tone(total)}">${signed(total)}</td></tr>`;
      }).join("") + `</tbody></table>` +
      `<p class="muted" style="font-size:12px;margin:12px 0 0">Read across: a green cell is what that player has won from the player in the column. Every pot is traced layer by layer — side pots, split boards and quartered pots each go to who really paid for them.</p>`;
    body.querySelectorAll("th[data-uid]").forEach((th) => th.addEventListener("click", () => {
      const p = ps.find((x) => String(x.user_id) === th.dataset.uid);
      if (p) openMyHands("", { user_id: p.user_id, name: p.name, is_me: p.is_me });
    }));
    const api = openModal({ title: "Head to head", sub: "Who has won what from whom, across every recorded session.", body, wide: true, autofocus: false, buttons: [{ label: "Close", cls: "primary" }] });
    api.modal.classList.add("xwide");
  }

  // every session on record; the site admin can take test tables out of the stats
  function openSessions() {
    const body = h("div", { class: "db" });
    const paint = () => {
      const data = U.club || {}, rows = data.sessions || [];
      body.innerHTML = (data.is_admin ? `<p class="muted" style="font-size:12.5px;margin:0 0 12px">Excluding a session takes it out of everyone's stats, hand lists and head-to-head. Nothing is deleted — you can restore it here any time. An open table is closed first.</p>` : "") +
        (rows.length ? rows.map((x) =>
          `<div class="sess ${x.excluded ? "excluded" : ""}" data-id="${esc(x.id)}"><div><b>${esc(x.name)}</b>${x.open ? ` <span class="pill live">Open</span>` : ""}${x.excluded ? ` <span class="pill">Excluded</span>` : ""}<br>` +
          `<small>${esc(String(x.created_at || "").slice(0, 10))} · ${d2(x.sb_cents)}/${d2(x.bb_cents)} · ante ${d2(x.ante_cents)}</small></div>` +
          `<small class="opt">${x.hands} hand${x.hands === 1 ? "" : "s"}</small><small class="opt">${x.players} player${x.players === 1 ? "" : "s"}</small>` +
          `<span class="sess-act"><button class="btn sm" data-open="${esc(x.id)}" type="button">Open</button>` +
          (data.is_admin ? `<button class="btn sm ${x.excluded ? "" : "danger"}" data-ex="${esc(x.id)}" data-on="${x.excluded ? "0" : "1"}" type="button">${x.excluded ? "Restore" : "Exclude"}</button>` : "") + `</span></div>`).join("")
          : `<div class="muted" style="text-align:center;padding:26px">No sessions yet.</div>`);
      body.querySelectorAll("[data-open]").forEach((b) => b.addEventListener("click", () => { api.close(null); C().openTable(b.dataset.open, true).catch((e) => toast(e.message, "err")); }));
      body.querySelectorAll("[data-ex]").forEach((b) => b.addEventListener("click", async () => {
        const on = b.dataset.on === "1", row = rows.find((x) => x.id === b.dataset.ex);
        if (on) {
          const ok = await confirmDialog({ title: `Exclude “${row.name}”?`, text: `Its ${row.hands} hand${row.hands === 1 ? "" : "s"} stop counting toward anyone's profit, accuracy and head-to-head${row.open ? ", and the table is closed (everyone is cashed out)" : ""}. You can restore it later.`, okLabel: "Exclude from stats", danger: true });
          if (!ok) return;
        }
        b.disabled = true;
        try {
          await C().j(`/games/api/tables/${encodeURIComponent(b.dataset.ex)}/exclude`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ on }) });
          toast(on ? "Session excluded from the stats" : "Session restored", "ok");
          await loadClub(true); U.lobbySig = ""; C().loadLobby().catch(() => {});
          paint();
        } catch (e) { b.disabled = false; toast(e.message, "err"); }
      }));
    };
    const api = openModal({ title: "All sessions", sub: "Every table on record, newest first.", body, wide: true, autofocus: false, buttons: [{ label: "Close", cls: "primary" }] });
    paint();
  }

  // --------------------------------------------------------- manage drawer
  function openDrawer(tab) {
    const s = C().G.state;
    if (!s) return;
    if (tab) U.drawerTab = tab;
    if (U.drawer) { paintDrawer(s, true); return; }
    const root = $("drawer-root");
    const layer = h("div", { style: "position:absolute;inset:0" });
    const scrim = h("div", { class: "scrim" });
    const dr = h("aside", { class: "drawer", role: "dialog", "aria-label": "Manage table" });
    layer.appendChild(scrim); layer.appendChild(dr);
    root.appendChild(layer);
    scrim.addEventListener("click", closeDrawer);
    U.drawer = { layer, dr, sig: "" };
    paintDrawer(s, true);
    requestAnimationFrame(() => layer.classList.add("open"));
  }
  function closeDrawer() {
    const d = U.drawer;
    if (!d) return;
    U.drawer = null;
    d.layer.classList.remove("open");
    setTimeout(() => d.layer.remove(), 300);
  }
  const segHtml = (id, opts, cur) => `<div class="seg" id="${id}">${opts.map(([v, l]) => `<button type="button" data-v="${v}" class="${Number(v) === Number(cur) ? "on" : ""}">${l}</button>`).join("")}</div>`;

  function paintDrawer(s, force) {
    const d = U.drawer;
    if (!d) return;
    if (!s.is_host) { closeDrawer(); return; }
    // Forms are only rebuilt when the drawer opens / the tab changes / a save
    // lands — never under the host's cursor. The Players tab follows the table.
    const sig = U.drawerTab === "players"
      ? JSON.stringify(s.seats.map((x) => [x.user_id, x.name, x.stack_cents, x.sitting_out, x.pending_remove, x.auto_stack_cents, x.trusted, x.present])) + s.phase + s.auto_stack.mode
      : U.drawerTab === "chips" ? JSON.stringify([s.requests, s.settings.approve_buyins, s.auto_stack, s.auto_topup])
      : U.drawerTab === "table" ? `${s.running}:${s.phase}:${s.actor}:${s.can_deal}:${s.runout.blocking}` : "form";
    if (!force && sig === d.sig) return;
    d.sig = sig;
    const set = s.settings, st = s.stakes;
    const nReq = (s.requests || []).length;
    const tabs = [["game", "Game"], ["chips", "Chips" + (nReq ? ` (${nReq})` : "")], ["pace", "Pace"], ["players", "Players"], ["table", "Table"]];
    let html = `<div class="dr-head"><h3>${icon("i-crown")}Manage table</h3><button class="icon-btn" id="dr-x" aria-label="Close">${icon("i-x")}</button></div>` +
      `<div class="dr-tabs">${tabs.map(([k, l]) => `<button data-t="${k}" class="${k === U.drawerTab ? "on" : ""}">${l}</button>`).join("")}</div><div class="dr-body">`;
    if (U.drawerTab === "game") {
      html += `<div class="grp"><h4>Table</h4><label class="field"><span>Name</span><input type="text" id="m-name" maxlength="60" value="${esc(s.name)}"/></label>` +
        `<div class="row2"><label class="field"><span>Ante</span>${moneyInput("m-ante", st.ante_cents)}<small>Applies from the next hand</small></label>` +
        `<div class="field"><span>Blinds (chip unit)</span><input type="text" class="input num" disabled value="${d2(st.sb_cents)} / ${d2(st.bb_cents)}"/><small>Fixed once a table is created</small></div></div>` +
        `<div class="field"><span>Seats</span>${segHtml("m-seats", [2, 3, 4, 5, 6, 7, 8].map((n) => [n, String(n)]), s.num_seats)}<small>Between hands only — the higher seats must be empty to shrink</small></div></div>` +
        `<div class="grp"><h4>Buy-ins</h4><div class="row3"><label class="field"><span>Minimum</span>${moneyInput("m-min", set.min_buyin_cents)}</label><label class="field"><span>Default</span>${moneyInput("m-dflt", st.default_buyin_cents)}</label><label class="field"><span>Maximum</span>${moneyInput("m-max", set.max_buyin_cents)}</label></div><p>0 = no limit. The maximum also caps top-ups.</p></div>` +
        `<div class="grp"><h4>Privacy &amp; extras</h4><div class="setrow"><div><b>List in the lobby</b><small>Off = link only</small></div><label class="switch"><input type="checkbox" id="m-listed" ${set.listed ? "checked" : ""}/><i></i></label></div>` +
        `<div class="setrow"><div><b>Accuracy marks on everyone's actions</b><small>The replayer rates every decision against the network. Off = players only see marks on their own</small></div><label class="switch"><input type="checkbox" id="m-grades" ${set.show_grades ? "checked" : ""}/><i></i></label></div>` +
        `<div class="setrow"><div><b>Rabbit hunting</b><small>Let players peek at the undealt streets after a fold-out</small></div><label class="switch"><input type="checkbox" id="m-rabbit" ${set.allow_rabbit ? "checked" : ""}/><i></i></label></div></div>`;
    } else if (U.drawerTab === "chips") {
      const modeSeg = (id, cur) => `<div class="seg as-modes" id="${id}">${[["off", "Off"], ["host", "Host sets"], ["player", "Players choose"]].map(([v, l]) => `<button type="button" data-v="${v}" class="${cur === v ? "on" : ""}">${l}</button>`).join("")}</div>`;
      html += `<div class="grp"><h4>Buy-in approval</h4><div class="setrow"><div><b>I approve every buy-in</b><small>Sit-downs and top-ups wait for your OK. Players you trust never wait.</small></div><label class="switch"><input type="checkbox" id="m-approve" ${set.approve_buyins ? "checked" : ""}/><i></i></label></div>` +
        (nReq ? (s.requests || []).map((r) =>
          `<div class="prow req" data-req="${r.id}">${avatar(r.name, r.name)}<div class="who"><b>${esc(r.name)}</b><small>${r.kind === "sit" ? `wants seat ${r.seat + 1} with ${d2(r.amount_cents)}` : `wants to add ${d2(r.amount_cents)}`}</small></div>` +
          `<div class="acts"><button class="btn sm primary" data-ok="${r.id}">Approve</button><button class="btn sm gold" data-okt="${r.id}" title="Approve, and never ask again for this player">+ Trust</button><button class="icon-btn" data-no="${r.id}" title="Decline" style="color:#ff9aa6">${icon("i-x")}</button></div></div>`).join("")
          : (set.approve_buyins ? `<p>No one is waiting. Trust regulars from the Players tab so the game never stops for them.</p>` : "")) + `</div>` +
        `<div class="grp"><h4>Auto top-up</h4><p>When a stack drops below a threshold it is topped back up before the next hand. Winnings stay on the table — no ratholing.</p>${modeSeg("m-top", s.auto_topup.mode)}` +
        (s.auto_topup.mode === "host" ? `<div class="row2"><label class="field"><span>Top up to</span>${moneyInput("m-top-target", s.auto_topup.all_target_cents || st.default_buyin_cents)}</label><label class="field"><span>When below</span>${moneyInput("m-top-below", s.auto_topup.all_below_cents || s.auto_topup.all_target_cents || st.default_buyin_cents)}</label></div><button class="btn sm" id="m-top-apply">Apply to everyone</button><small class="muted">Per-player amounts: tap a player's seat.</small>` : "") + `</div>` +
        `<div class="grp"><h4>Set stack every hand</h4><p>Every stack is reset to one amount before EVERY deal — short stacks top up, big stacks bank the difference. For high-action games where ratholing is fine.</p>${modeSeg("m-auto", s.auto_stack.mode)}` +
        (s.auto_stack.mode === "host" ? `<div class="sz-row"><label class="field" style="flex:1"><span>Stack for everyone</span>${moneyInput("m-auto-all", s.auto_stack.all_cents || st.default_buyin_cents)}</label><button class="btn sm" id="m-auto-apply" style="align-self:flex-end;height:40px">Apply</button></div><small class="muted">Per-player amounts: tap a player's seat. Set-stack wins when a player has both.</small>` : "") + `</div>` +
        (set.approve_buyins ? `<small class="muted">While you approve buy-ins, automatic chips only run for you and the players you trust.</small>` : "");
    } else if (U.drawerTab === "pace") {
      html += `<div class="grp"><h4>Shot clock</h4><div class="field"><span>Decision time</span>${segHtml("m-clock", [[0, "Off"], [10, "10s"], [15, "15s"], [20, "20s"], [30, "30s"], [45, "45s"], [60, "60s"]], s.decision_secs)}<small>When it runs out the player checks if that is free, otherwise folds.</small></div>` +
        `<div class="field"><span>Time bank per player</span>${segHtml("m-bank", [[0, "Off"], [15, "15s"], [30, "30s"], [60, "60s"], [120, "2 min"]], set.time_bank_secs)}<small>Burned automatically after the base clock; a couple of seconds come back every hand.</small></div></div>` +
        `<div class="grp"><h4>Dealing</h4><div class="field"><span>Next hand after</span>${segHtml("m-deal", [[0, "Manual"], [2, "2s"], [3, "3s"], [5, "5s"], [8, "8s"], [12, "12s"]], set.deal_delay_secs)}<small>The server deals — the game keeps running even if you switch tabs.</small></div>` +
        `<div class="field"><span>All-in runout, per street</span>${segHtml("m-pause", [[0.5, "0.5s"], [1, "1s"], [1.5, "1.5s"], [2.5, "2.5s"], [4, "4s"]], s.street_pause_secs)}</div></div>` +
        ``;
    } else if (U.drawerTab === "players") {
      const seated = s.seats.filter((x) => !x.empty);
      html += `<div class="grp"><h4>${seated.length} seated</h4>` + seated.map((x) =>
        `<div class="prow" data-uid="${x.user_id}">${avatar(x.name, x.name)}<div class="who"><b>${esc(x.name)}${x.is_host ? ' <span class="pill host" style="height:18px;font-size:10px">Host</span>' : ""}</b><small>${d2(x.stack_cents)}${x.sitting_out ? " · sitting out" : ""}${x.pending_remove ? " · leaving" : ""}</small></div>` +
        `<div class="acts">` +
        (x.user_id !== s.my_user_id ? `<button class="btn sm ${x.trusted ? "gold" : ""}" data-trust="${x.user_id}" data-on="${x.trusted ? 0 : 1}" title="${x.trusted ? "Trusted: buys in without asking. Click to stop trusting." : "Trust: let them buy in and top up without your approval"}">${icon("i-check", "sm")}${x.trusted ? "Trusted" : "Trust"}</button>` : "") +
        (x.pending_remove ? "" : `<button class="btn sm" data-away="${x.user_id}" data-on="${x.sitting_out ? 0 : 1}">${x.sitting_out ? (x.user_id === s.my_user_id ? "I'm back" : "Sit in") : "Sit out"}</button>`) +
        (x.user_id !== s.my_user_id && !x.pending_remove ? `<button class="icon-btn" data-host="${x.user_id}" title="Make host">${icon("i-swap")}</button><button class="icon-btn" data-kick="${x.user_id}" title="Remove from table" style="color:#ff9aa6">${icon("i-x")}</button>` : "") +
        `</div></div>`).join("") + `</div>` +
        ((s.spectators || []).length ? `<div class="grp"><h4>${s.spectators.length} watching</h4><p style="margin:0">${s.spectators.map(esc).join(", ")}</p></div>` : "") +
        `<small class="muted">Tap a player's seat for their card: trust, per-player automatic chips, notes.</small>`;
    } else {
      const busy = s.phase === "in_hand" || s.runout.blocking;
      const actor = s.actor != null ? s.seats[s.actor] : null;
      html += `<div class="grp"><h4>Game</h4><div class="setrow"><div><b>${s.running ? (busy ? "Game is running" : "Game is running") : "Game is paused"}</b><small>${s.running ? "Pausing lets the current hand finish first." : s.eligible_count < 2 ? "Needs two players with more than the ante." : "Everyone is waiting on you."}</small></div>` +
        `<button class="btn ${s.running ? "" : "primary"}" id="m-run" ${!s.running && s.eligible_count < 2 ? "disabled" : ""}>${icon(s.running ? "i-pause" : "i-play", "sm")}${s.running ? (busy ? "Pause after hand" : "Pause") : "Start game"}</button></div>` +
        `<div class="setrow"><div><b>Deal now</b><small>Skip the wait between hands</small></div><button class="btn" id="m-deal" ${s.can_deal ? "" : "disabled"}>${icon("i-bolt", "sm")}Deal</button></div></div>` +
        `<div class="grp danger"><h4>Careful</h4><div class="setrow"><div><b>Fold the player on the clock</b><small>${actor && s.phase === "in_hand" ? esc(actor.name) + " is up. Checks instead when checking is free." : "Nobody is on the clock."}</small></div><button class="btn danger" id="m-hostfold" ${actor && s.phase === "in_hand" ? "" : "disabled"}>Fold</button></div>` +
        `<div class="setrow"><div><b>Close the table</b><small>Cashes everyone out and ends the session${busy ? " — after this hand" : ""}.</small></div><button class="btn danger" id="m-close" ${busy ? "disabled" : ""}>Close table</button></div></div>`;
    }
    html += `</div>`;
    if (U.drawerTab === "game") html += `<div class="dr-foot"><button class="btn ghost" id="m-cancel">Cancel</button><button class="btn primary" id="m-save">Save changes</button></div>`;
    d.dr.innerHTML = html;
    wireDrawer(s, d.dr);
  }

  function wireDrawer(s, root) {
    const q = (id) => root.querySelector("#" + id);
    const gid = s.id;
    const saveSet = async (patch, okMsg) => { try { await C().tablePost("settings", patch); if (okMsg) toast(okMsg, "ok"); } catch (_) { const cur = C().G.state; if (cur) paintDrawer(cur, true); } };
    q("dr-x").addEventListener("click", closeDrawer);
    root.querySelectorAll(".dr-tabs button").forEach((b) => b.addEventListener("click", () => { U.drawerTab = b.dataset.t; paintDrawer(C().G.state, true); }));
    segWire(root);
    if (U.drawerTab === "game") {
      q("m-cancel").addEventListener("click", closeDrawer);
      q("m-save").addEventListener("click", async () => {
        const patch = {
          name: q("m-name").value, ante_cents: C().toCents(q("m-ante").value),
          min_buyin_cents: C().toCents(q("m-min").value) || 0, default_buyin_cents: C().toCents(q("m-dflt").value),
          max_buyin_cents: C().toCents(q("m-max").value) || 0, listed: q("m-listed").checked, allow_rabbit: q("m-rabbit").checked,
          show_grades: q("m-grades").checked,
        };
        const seats = segVal(root, "m-seats");
        if (seats !== s.num_seats) patch.num_seats = seats;
        try { await C().tablePost("settings", patch); toast("Table updated", "ok"); closeDrawer(); } catch (_) { /* toasted */ }
      });
    } else if (U.drawerTab === "chips") {
      q("m-approve").addEventListener("change", (e) => saveSet({ approve_buyins: e.target.checked }));
      const resolve = (id, action, trust) => C().tablePost("request", { id: Number(id), action, trust: !!trust }).catch(() => {});
      root.querySelectorAll("[data-ok]").forEach((b) => b.addEventListener("click", () => resolve(b.dataset.ok, "approve", false)));
      root.querySelectorAll("[data-okt]").forEach((b) => b.addEventListener("click", () => resolve(b.dataset.okt, "approve", true)));
      root.querySelectorAll("[data-no]").forEach((b) => b.addEventListener("click", () => resolve(b.dataset.no, "deny", false)));
      q("m-top").addEventListener("pick", async (e) => { try { await C().tablePost("auto_topup", { mode: e.detail }); } catch (_) { /* toasted */ } paintDrawer(C().G.state, true); });
      q("m-auto").addEventListener("pick", async (e) => { try { await C().tablePost("auto_stack", { mode: e.detail }); } catch (_) { /* toasted */ } paintDrawer(C().G.state, true); });
      const ta = q("m-top-apply");
      if (ta) ta.addEventListener("click", async () => { try { await C().tablePost("auto_topup", { all_target_cents: C().toCents(q("m-top-target").value) || 0, all_below_cents: C().toCents(q("m-top-below").value) || 0 }); toast("Auto top-up set for everyone", "ok"); } catch (_) { /* toasted */ } });
      const ap = q("m-auto-apply");
      if (ap) ap.addEventListener("click", async () => { try { await C().tablePost("auto_stack", { all_cents: C().toCents(q("m-auto-all").value) || 0 }); toast("Everyone resets to that stack each hand", "ok"); } catch (_) { /* toasted */ } });
    } else if (U.drawerTab === "pace") {
      q("m-clock").addEventListener("pick", (e) => saveSet({ decision_secs: Number(e.detail) }));
      q("m-bank").addEventListener("pick", (e) => saveSet({ time_bank_secs: Number(e.detail) }));
      q("m-deal").addEventListener("pick", (e) => saveSet({ deal_delay_secs: Number(e.detail) }));
      q("m-pause").addEventListener("pick", (e) => C().tablePost("street_pause", { secs: Number(e.detail) }).catch(() => {}));
    } else if (U.drawerTab === "players") {
      root.querySelectorAll("[data-trust]").forEach((b) => b.addEventListener("click", () => C().tablePost("trust", { user_id: Number(b.dataset.trust), on: b.dataset.on === "1" }).catch(() => {})));
      root.querySelectorAll("[data-away]").forEach((b) => b.addEventListener("click", () => {
        const uid = Number(b.dataset.away), on = b.dataset.on === "1";
        (uid === s.my_user_id ? C().tablePost("sit_out", { on }) : C().tablePost("sit_out_player", { user_id: uid, on })).catch(() => {});
      }));
      root.querySelectorAll("[data-kick]").forEach((b) => b.addEventListener("click", async () => {
        const who = s.seats.find((x) => x.user_id === Number(b.dataset.kick));
        const ok = await confirmDialog({ title: `Remove ${who ? who.name : "player"}?`, text: "They are cashed out at the end of the current hand and their seat opens up. They can sit back down later.", okLabel: "Remove", danger: true });
        if (ok) C().tablePost("kick", { user_id: Number(b.dataset.kick) }).catch(() => {});
      }));
      root.querySelectorAll("[data-host]").forEach((b) => b.addEventListener("click", async () => {
        const who = s.seats.find((x) => x.user_id === Number(b.dataset.host));
        const ok = await confirmDialog({ title: `Make ${who ? who.name : "them"} the host?`, text: "They get the Manage button; you keep your seat.", okLabel: "Hand over" });
        if (ok) C().tablePost("transfer_host", { user_id: Number(b.dataset.host) }).then(() => closeDrawer()).catch(() => {});
      }));
    } else {
      q("m-run").addEventListener("click", () => C().tablePost("run", { running: !s.running }).catch(() => {}));
      q("m-deal").addEventListener("click", () => C().deal(C().G.state));
      q("m-hostfold").addEventListener("click", async () => {
        const cur = C().G.state, a = cur && cur.actor != null ? cur.seats[cur.actor] : null;
        if (!a) return;
        const ok = await confirmDialog({ title: `Fold ${a.name}?`, text: a.user_id === cur.my_user_id ? "That's you — this folds your own hand." : "Use this when someone is away and the table is waiting. It can't be undone.", okLabel: "Fold them", danger: true });
        if (ok) C().tablePost("host_fold").catch(() => {});
      });
      q("m-close").addEventListener("click", async () => {
        const cur = C().G.state;
        const lines = (cur.ledger || []).map((r) => `<div class="settle-row"><span>${esc(r.name)}</span><b class="${r.net_cents >= 0 ? "pos" : "neg"}">${r.net_cents >= 0 ? "+" : ""}${d2(r.net_cents)}</b></div>`).join("");
        const ok = await confirmDialog({ title: "Close this table?", text: "Everyone is cashed out and the session ends. The final ledger stays available from the lobby.", body: `<div>${lines}</div>`, okLabel: "Close table", danger: true });
        if (ok) { try { await C().tablePost("close"); closeDrawer(); } catch (_) { /* toasted */ } }
      });
    }
    void gid;
  }

  function openInfo() {
    const s = C().G.state;
    if (!s) return;
    const set = s.settings, st = s.stakes;
    const row = (k, v) => `<div class="settle-row"><span class="muted">${k}</span><b style="color:var(--tx)">${v}</b></div>`;
    openModal({
      title: s.name, sub: "PLO5 double-board bomb pot", autofocus: false,
      body: `<div>${row("Blinds (chip unit)", `${d2(st.sb_cents)} / ${d2(st.bb_cents)}`)}${row("Ante", d2(st.ante_cents))}` +
        row("Buy-in", set.min_buyin_cents || set.max_buyin_cents ? `${set.min_buyin_cents ? d2(set.min_buyin_cents) : "any"} – ${set.max_buyin_cents ? d2(set.max_buyin_cents) : "any"}` : "No limits") +
        row("Seats", s.num_seats) + row("Decision time", s.decision_secs ? s.decision_secs + "s" : "No clock") + row("Time bank", set.time_bank_secs ? set.time_bank_secs + "s per player" : "Off") +
        row("Next hand", set.deal_delay_secs ? `dealt automatically after ${set.deal_delay_secs}s` : "dealt manually") + row("Rabbit hunt", set.allow_rabbit ? "Allowed" : "Off") + row("Lobby", set.listed ? "Listed" : "Link only") + `</div>` +
        `<div class="grp"><h4>How a hand works</h4><p style="margin:0;color:var(--tx-2);font-size:13px;line-height:1.5">Everyone dealt in posts the ante — no blinds. You get five cards and the hand starts on the flop with <b>two boards</b>. Betting is pot-limit. At showdown each board awards half the pot to the best hand using exactly two hole cards and three board cards.</p></div>`,
      buttons: [{ label: "Copy invite link", cls: "", onClick: () => { copyInvite(s.id); return false; } }, { label: "Done", cls: "primary" }],
    });
  }

  // ------------------------------------------------------------ player card
  // Notes + colour tags are PRIVATE: they live in this browser only.
  const TAGS = { none: "transparent", red: "#f2566a", amber: "#f5a742", green: "#35c878", blue: "#5b95ff", violet: "#a67bf0" };
  function notesAll() { try { return JSON.parse(localStorage.getItem("hg.notes.v1") || "{}"); } catch (_) { return {}; } }
  function noteFor(uid) { return notesAll()[String(uid)] || { tag: "none", text: "" }; }
  function saveNote(uid, patch) {
    const all = notesAll();
    all[String(uid)] = Object.assign({ tag: "none", text: "" }, all[String(uid)], patch);
    if (all[String(uid)].tag === "none" && !all[String(uid)].text) delete all[String(uid)];
    try { localStorage.setItem("hg.notes.v1", JSON.stringify(all)); } catch (_) { /* private mode */ }
    const s = C().G.state;
    if (s) HG.table.render(s, s);
  }
  function openPlayer(seatIdx) {
    const s = C().G.state;
    const x = s && s.seats[seatIdx];
    if (!x || x.empty) return;
    const mine = x.user_id === s.my_user_id;
    const row = (s.ledger || []).find((r) => r.user_id === x.user_id) || {};
    const st = ((U.hands && U.hands.stats) || []).find((r) => r.name === x.name) || {};
    const note = noteFor(x.user_id);
    const body = h("div", { style: "display:flex;flex-direction:column;gap:16px" });
    body.innerHTML =
      `<div class="pcard-head">${avatar(x.name, x.name)}<div><b>${esc(x.name)}${mine ? " (you)" : ""}</b><span class="muted">Seat ${seatIdx + 1}${x.is_host ? " · host" : ""}${x.sitting_out ? " · sitting out" : ""}</span></div></div>` +
      `<div class="pcard-stats"><div><b>${d2(x.stack_cents)}</b><small>Stack</small></div><div><b class="${row.net_cents > 0 ? "pos" : row.net_cents < 0 ? "neg" : ""}">${row.net_cents > 0 ? "+" : ""}${d2(row.net_cents || 0)}</b><small>Net</small></div>` +
      `<div><b>${st.hands != null ? st.hands : "–"}</b><small>Hands</small></div><div><b>${st.wins != null ? st.wins : "–"}</b><small>Won</small></div></div>` +
      (mine ? "" : `<div class="field"><span>Colour tag</span><div class="tagrow">${Object.entries(TAGS).map(([k, col]) => `<button type="button" data-tag="${k}" class="${note.tag === k ? "on" : ""}" style="--tc:${k === "none" ? "rgba(255,255,255,.12)" : col}" title="${k}"></button>`).join("")}</div></div>` +
        `<label class="field"><span>Private note</span><textarea class="input" id="pc-note" maxlength="500" placeholder="Only you can see this — it stays in this browser.">${esc(note.text)}</textarea></label>`);
    if (s.is_host && s.status === "open") {
      const hostBox = h("div", { class: "grp" });
      hostBox.innerHTML = `<h4>Host</h4>` +
        (mine ? "" : `<div class="setrow"><div><b>Trusted</b><small>Buys in, tops up and auto-tops-up without your approval</small></div><label class="switch"><input type="checkbox" id="pc-trust" ${x.trusted ? "checked" : ""}/><i></i></label></div>`) +
        (s.auto_topup.mode === "host" ? `<div class="row2"><label class="field"><span>Top up to</span>${moneyInput("pc-top-target", x.topup_target_cents || 0)}</label><label class="field"><span>When below</span>${moneyInput("pc-top-below", x.topup_below_cents || 0)}</label></div>` : "") +
        (s.auto_stack.mode === "host" ? `<label class="field"><span>My stack each hand (host)</span>${moneyInput("pc-set", x.auto_stack_cents || 0)}<small>0 = off for this player</small></label>` : "") +
        (s.auto_topup.mode === "host" || s.auto_stack.mode === "host" ? `<button class="btn sm" id="pc-save">Save automatic chips</button>` : "");
      if (hostBox.children.length > 1) body.appendChild(hostBox);
      const tr = hostBox.querySelector("#pc-trust");
      if (tr) tr.addEventListener("change", (e) => C().tablePost("trust", { user_id: x.user_id, on: e.target.checked }).then(() => toast(e.target.checked ? `${x.name} is trusted` : `${x.name} needs approval again`, "ok")).catch(() => { e.target.checked = !e.target.checked; }));
      const sv = hostBox.querySelector("#pc-save");
      if (sv) sv.addEventListener("click", async () => {
        try {
          if (s.auto_topup.mode === "host") await C().tablePost("auto_topup", { players: [{ user_id: x.user_id, target_cents: C().toCents(hostBox.querySelector("#pc-top-target").value) || 0, below_cents: C().toCents(hostBox.querySelector("#pc-top-below").value) || 0 }] });
          if (s.auto_stack.mode === "host") await C().tablePost("auto_stack", { players: [{ user_id: x.user_id, cents: C().toCents(hostBox.querySelector("#pc-set").value) || 0 }] });
          toast("Saved", "ok");
        } catch (_) { /* toasted */ }
      });
    }
    const buttons = [];
    if (s.is_host && !mine && s.status === "open" && !x.pending_remove) {
      buttons.push({ label: x.sitting_out ? "Sit them in" : "Sit them out", cls: "", onClick: () => C().tablePost("sit_out_player", { user_id: x.user_id, on: !x.sitting_out }) });
      buttons.push({ label: "Remove", cls: "danger", onClick: async () => {
        const ok = await confirmDialog({ title: `Remove ${x.name}?`, text: "They are cashed out at the end of the current hand and their seat opens up.", okLabel: "Remove", danger: true });
        if (!ok) return false;
        await C().tablePost("kick", { user_id: x.user_id });
      } });
    }
    buttons.push({ label: "Done", cls: "primary" });
    if (!mine) {
      body.querySelector(".tagrow").addEventListener("click", (e) => {
        const b = e.target.closest("button[data-tag]");
        if (!b) return;
        body.querySelectorAll(".tagrow button").forEach((k) => k.classList.toggle("on", k === b));
        saveNote(x.user_id, { tag: b.dataset.tag });
      });
      body.querySelector("#pc-note").addEventListener("change", (e) => saveNote(x.user_id, { text: e.target.value.trim() }));
    }
    openModal({ title: "Player", body, buttons, autofocus: false });
    if (!U.hands && s.is_member) loadHands(s);
  }

  function seatMenu(anchor) {
    const s = C().G.state;
    if (!s) return;
    const seated = Number.isInteger(s.my_seat);
    const me = seated ? s.seats[s.my_seat] : null;
    const busy = s.phase === "in_hand" || s.runout.blocking;
    const items = [{ header: seated ? `${me.name} · ${d2(me.stack_cents)}` : "Not seated" }];
    if (seated && s.status === "open") {
      items.push({ icon: "i-pluscircle", label: s.needs_approval ? "Request chips" : "Add chips", onClick: openTopUp });
      if (s.auto_stack.mode !== "off" || s.auto_topup.mode !== "off") items.push({ icon: "i-wallet", label: "Automatic chips…", onClick: openAutoChips });
      if (me.sitting_out) items.push({ icon: "i-play", label: "I'm back", onClick: () => C().tablePost("sit_out", { on: false }).catch(() => {}) });
      else {
        items.push({ icon: "i-coffee", label: me.sit_out_next ? "Cancel sit-out" : "Sit out next hand", onClick: () => C().tablePost("sit_out", me.sit_out_next ? { on: false } : { on: true, next_hand: true }).catch(() => {}) });
        items.push({ icon: "i-clock", label: "Step away now", hint: "You are checked or folded until you come back", onClick: () => C().tablePost("sit_out", { on: true }).catch(() => {}) });
      }
      items.push("-");
      items.push({
        icon: "i-door", label: me.pending_remove ? "Leaving after this hand" : "Leave seat", danger: true, disabled: !!me.pending_remove,
        onClick: async () => {
          const ok = await confirmDialog({ title: "Leave your seat?", text: busy ? "You're out of this hand right away (checked or folded for you) and cashed out when it ends." : `You cash out ${d2(me.stack_cents)} and your seat opens up.`, okLabel: "Leave seat", danger: true });
          if (ok) C().tablePost("leave").catch(() => {});
        },
      });
    } else if (s.status === "open") items.push({ icon: "i-user", label: "Pick an open seat on the table", disabled: true, onClick: () => {} });
    items.push({ icon: "i-back", label: "Back to lobby", onClick: () => C().showLobby(false) });
    openMenu(anchor, items);
  }

  // -------------------------------------------------------------------- init
  function init() {
    const me = C().G.me || {};
    $("userchip").innerHTML = `${avatar(me.name || me.email, me.name || me.email, "sm")}<span>${esc(me.name || me.email || "")}</span>`;
    $("c-open").addEventListener("click", openCreate);
    $("lb-myhands").addEventListener("click", () => openMyHands(""));
    $("lb-h2h").addEventListener("click", openMatrix);
    $("lb-allsess").addEventListener("click", openSessions);
    $("join-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const raw = $("join-input").value.trim();
      const m = raw.match(/\/games\/t\/([A-Za-z0-9_-]+)/) || raw.match(/^([A-Za-z0-9_-]{4,})$/);
      if (!m) return toast("Paste a table link or its code", "err");
      C().openTable(m[1], true).catch((err) => toast(err.status === 404 ? "No table with that code" : err.message, "err"));
    });
    $("brand-link").addEventListener("click", (e) => { e.preventDefault(); C().showLobby(false); });
    $("tb-back").addEventListener("click", () => C().showLobby(false));
    $("tb-invite").addEventListener("click", () => copyInvite(C().G.gameId));
    $("tb-manage").addEventListener("click", () => openDrawer(((C().G.state || {}).requests || []).length ? "chips" : null));
    $("tb-watch").addEventListener("click", (e) => {
      const s = C().G.state, names = (s && s.spectators) || [];
      openMenu(e.currentTarget, [{ header: `${names.length} watching` }].concat(names.map((n) => ({ icon: "i-eye", label: n, disabled: true, onClick: () => {} }))));
    });
    $("tb-info").addEventListener("click", openInfo);
    $("tb-prefs").addEventListener("click", openPrefs);
    $("tb-seat").addEventListener("click", (e) => seatMenu(e.currentTarget));
    $("tb-more").addEventListener("click", (e) => {
      const s = C().G.state, on = !!C().G.prefs.sound;
      openMenu(e.currentTarget, [
        { header: s ? s.name : "Table" },
        { icon: "i-link", label: "Copy invite link", disabled: !s || s.status !== "open", onClick: () => copyInvite(C().G.gameId) },
        { icon: "i-info", label: "Table info", onClick: openInfo },
        { icon: "i-clock", label: "Last hand", disabled: !s || !s.last_hand_no || !s.is_member, onClick: () => openHand(s.id, C().G.state.last_hand_no) },
        "-",
        { icon: on ? "i-vol" : "i-mute", label: on ? "Sound on" : "Sound off", onClick: () => { C().savePrefs({ sound: !on }); renderSound(); } },
        { icon: "i-sliders", label: "Preferences", onClick: openPrefs },
      ]);
    });
    $("tb-sound").addEventListener("click", () => { C().savePrefs({ sound: !C().G.prefs.sound }); renderSound(); if (C().G.prefs.sound) { HG.sound.unlock(); HG.sound.play("chip"); } });
    $("tb-rail").addEventListener("click", () => setRail(!C().G.prefs.rail));
    document.querySelectorAll("#rail-tabs button").forEach((b) => b.addEventListener("click", () => setRail(true, b.dataset.tab)));
    $("chat-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const input = $("chat-input"), text = (input.value || "").trim();
      if (!text || !C().G.gameId) return;
      input.value = "";
      try { await C().tablePost("chat", { text }); } catch (_) { input.value = text; }
    });
    const tray = $("emote-tray");
    tray.innerHTML = Object.entries(HG.table.EMOTES).map(([k, g]) => `<button type="button" data-e="${k}" title="${k}">${g}</button>`).join("");
    tray.addEventListener("click", (e) => { const b = e.target.closest("button[data-e]"); if (b) { C().tablePost("react", { emote: b.dataset.e }).catch(() => {}); tray.hidden = true; } });
    $("emote-btn").addEventListener("click", () => { tray.hidden = !tray.hidden; });
    U.railTab = C().G.prefs.railTab || "chat";
    const narrow = globalThis.matchMedia && matchMedia("(max-width: 1080px)").matches;
    if (narrow) C().G.prefs.rail = false;
    setRail(!!C().G.prefs.rail, U.railTab);
    renderSound();
    if (HG.play) HG.play.init();
  }

  function showLobby() { $("lobby").hidden = false; $("table-view").hidden = true; closeDrawer(); U.lobbySig = ""; loadClub(true); }
  function showTable() {
    $("lobby").hidden = true; $("table-view").hidden = false;
    U.eventSeen = null; U.chatSig = ""; U.logSig = ""; U.ledgerSig = ""; U.hands = null; U.handsFor = null; U.unread = 0;
    $("hands-body").dataset.k = "";
  }
  function renderConn() {
    const c = $("conn"), st = C().G.conn;
    c.className = "conn " + (st === "ok" ? "" : st);
    c.lastChild.textContent = st === "ok" ? "Live" : st === "slow" ? "Slow" : "Reconnecting…";
  }

  function renderTop(s) {
    $("table-title").textContent = s.name;
    $("table-sub").textContent = `PLO5 bomb pot · ${d2(s.stakes.sb_cents)}/${d2(s.stakes.bb_cents)} · ante ${d2(s.stakes.ante_cents)}`;
    const st = $("tb-status");
    const label = s.status !== "open" ? "Closed" : s.running ? "Live" : "Paused";
    st.textContent = label;
    st.className = "pill " + (label === "Live" ? "live" : label === "Paused" ? "paused" : "");
    $("tb-hand").textContent = s.hand_no ? `Hand #${s.hand_no}` : "";
    $("tb-hand").hidden = !s.hand_no;
    $("tb-hostpill").hidden = !s.is_host;
    $("tb-manage").hidden = !(s.is_host && s.status === "open");
    const nReq = s.is_host ? (s.requests || []).length : 0;
    $("tb-req").hidden = !nReq; $("tb-req").textContent = String(nReq);
    const nw = (s.spectators || []).length;
    $("tb-watch").hidden = !nw; $("tb-watch-n").textContent = String(nw);
    $("tb-invite").hidden = s.status !== "open";
  }

  function handleEvents(s, prev) {
    const evs = s.events || [];
    const last = evs.length ? evs[evs.length - 1].id : 0;
    if (U.eventSeen != null && prev && prev.id === s.id) {
      evs.filter((e) => e.id > U.eventSeen).slice(-3).forEach((e) => {
        if (e.kind === "timeout" && e.seat === s.my_seat) { toast("You ran out of time — " + (e.text.includes("folded") ? "your hand was folded" : "you were checked"), "err", 5000); return; }
        if (e.kind === "request") { if (s.is_host && /asks to/.test(e.text)) { toast(e.text + " — open Manage › Chips", "gold", 6000); HG.sound && HG.sound.play("msg"); } return; }
        if (e.kind === "fair") { toast(e.text, "gold", 6000); return; } // a redone shuffle is always said out loud
        if (["join", "leave", "rebuy", "host", "settings", "run"].includes(e.kind)) toast(e.text, e.kind === "join" ? "ok" : "");
        if (e.kind === "join") HG.sound && HG.sound.play("sit");
      });
    }
    U.eventSeen = last;
    if (prev && prev.id === s.id && prev.last_hand_no !== s.last_hand_no) { U.handsFor = null; }
  }

  function render(s, prev, opts) {
    renderTop(s);
    HG.table.render(s, prev, opts);
    if (HG.play) HG.play.render(s, prev, opts);
    handleEvents(s, prev);
    renderRail(s, !!(opts && opts.unitChanged));
    if (U.drawer) paintDrawer(s, false);
  }

  HG.ui = {
    init, render, renderLobby, showLobby, showTable, renderConn, toast, openModal, confirmDialog, openMenu, closeTop,
    openSit, openTopUp, openAutoChips, openPlayer, openMyHands, noteFor, TAGS, openDrawer, openInfo, openPrefs, openHand, copyInvite, setRail, renderDock: (s) => HG.play && HG.play.render(s, s),
    onClock: (left, tm) => HG.play && HG.play.onClock(left, tm),
  };
})();
