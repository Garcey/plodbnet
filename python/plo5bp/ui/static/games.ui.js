"use strict";
// Home games — UI shell: the building blocks every window uses (toasts, dialogs,
// menus, segmented pickers), the side rail (chat / hand log / ledger / history),
// preferences, the profile picture, the top bar and the render entry point.
// The windows live in feature modules that load after this file and add their
// functions to HG.ui (they call each other through HG.ui at call time, so their
// load order doesn't matter):
//   games.lobby.js    the lobby, clubs, hosting a table, the club's numbers
//   games.history.js  the hand replayer, Open in Study, the lifetime hand database
//   games.seat.js     sitting down, chips, automatic chips, requests, the player card, leaving
//   games.manage.js   the host's Manage drawer and the Table info card
// The action dock lives in games.play.js; the felt in games.table.js.
(function () {
  const HG = (globalThis.HG = globalThis.HG || {});
  const $ = (id) => document.getElementById(id);
  const C = () => HG.core;
  // Markup (FE-003): html`` escapes every value it is given; put() puts markup in as
  // markup and anything else as text (games.js; this file loads before it, hence the
  // call-time wrappers). Styling lives in games.css — a value the CSS needs rides in
  // data-vars (FE-011: the page allows no inline style).
  const html = (strings, ...values) => C().html(strings, ...values);
  const put = (el, content) => C().put(el, content);
  const icon = (id, cls) => html`<svg class="ico ${cls || ""}"><use href="#${id}"/></svg>`;
  const UI = (HG.ui = HG.ui || {});
  UI.onInit = UI.onInit || [];  // (each feature module's own wiring, run by init)
  const U = (HG.uiState = {
    eventSeen: null, chatSig: "", logSig: "", ledgerSig: "", handsFor: null, hands: null, handsLoading: false,
    unread: 0, drawer: null, drawerTab: "game", modals: [], menu: null, lobbySig: "", railTab: "chat",
  });

  // An element with its attributes (on* = a listener) and its content: markup (html``),
  // a node, or text — a plain string is always text (put).
  function h(tag, attrs, content) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") e.className = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else if (v === true) e.setAttribute(k, "");
      else if (v !== false && v != null) e.setAttribute(k, v);
    }
    if (attrs && attrs["data-vars"]) C().applyVars(e);
    if (content != null) put(e, content);
    return e;
  }
  // markup added at the end of `el` (new chat lines), its data-vars applied like put's
  function append(el, markup) {
    const box = put(document.createElement("div"), markup);
    while (box.firstChild) el.appendChild(box.firstChild);
  }
  // `url` = the person's profile picture (2026-09-26); the initials stay under it
  // and show again if the picture can't load (the error listener in init drops it).
  // The hue is a CSS variable (data-vars: the page allows no inline style).
  function avatar(name, key, cls, url) {
    const A = HG.avatar;
    const img = url ? html`<img src="${url}" alt="" loading="lazy" decoding="async"/>` : "";
    return html`<span class="av ${cls || ""}" data-vars="h:${A.hueOf(key != null ? key : name)}">${A.initials(name)}${img}</span>`;
  }
  // `label` names the box for a screen reader (optional)
  function moneyInput(id, cents, label) {
    return html`<div class="money"><input class="input" id="${id}" inputmode="decimal" autocomplete="off" value="${(cents / 100).toFixed(2)}"${label ? html` aria-label="${label}"` : ""}/></div>`;
  }
  const CONN_GRACE_MS = 2500;  // (renderConn: how long a lost connection may last before the bar shows)
  const d2 = (c) => C().dollars(c);
  const d0 = (c) => C().dollars(c, true);  // "$100", "$12.50": preset buttons are narrow
  // The games a table can deal (2026-09-26: PLO6 next to PLO5; 2026-09-27: PLO67). The
  // SERVER's list is the source (FE-004): the lobby data and every table view carry it
  // (setGames) — labels, seats, graded, and whether this server can deal it at all (PLO67
  // needs its engine). This copy only covers the moment before the first answer, and an
  // older server. Seat limits: one deck (7 x 6 + 10 = 52; PLO67 reserves seven cards a
  // player + 10 + 3 face-up burns = 48 at 5).
  const WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven"];
  const GAMES = {
    plo5: { label: "PLO5", name: "PLO5 double-board bomb pot", hole: 5, dealt: 5, burns: 0, word: "five", maxSeats: 8, graded: true, available: true },
    plo6: { label: "PLO6", name: "PLO6 double-board bomb pot", hole: 6, dealt: 6, burns: 0, word: "six", maxSeats: 7, graded: false, available: true },
    plo67: { label: "PLO67", name: "PLO67 double-board bomb pot", hole: 7, dealt: 4, burns: 3, word: "four", maxSeats: 5, graded: false, available: true },
  };
  function setGames(list) {
    for (const g of Array.isArray(list) ? list : [list]) {
      if (!g || !g.code) continue;
      const cur = GAMES[g.code] || {};
      GAMES[g.code] = {
        label: g.label || cur.label || g.code, name: g.name || cur.name || g.code, hole: g.hole, dealt: g.dealt, burns: g.burns || 0,
        word: WORDS[g.dealt] || String(g.dealt), maxSeats: g.max_seats, graded: !!g.graded,
        available: g.available != null ? !!g.available : cur.available !== false,
      };
    }
  }
  const gameOf = (v) => GAMES[v] || GAMES.plo5;
  const gameNote = (v) => {
    const G = gameOf(v), cards = G.word[0].toUpperCase() + G.word.slice(1);
    const deal = G.burns ? `${cards} cards each + one more for every red burn (the burns are dealt face up)` : `${cards} cards each`;
    return `${deal}, up to ${G.maxSeats} players · ${G.graded ? "every decision graded by the network" : `not graded yet (there is no ${G.label} network)`}`;
  };
  // PLO67 in one paragraph (the table's info card and the Game guide)
  const burnRules = () => html`The three burn cards are dealt <b>face up</b>: one before the flops, one before the turns, one before the rivers.
    Every <b>red</b> burn (diamond or heart) deals everyone still in the hand — all-in players too — one more hole card:
    4-5 cards on the flop, 4-6 on the turn, up to 7 on the river.`;
  // The choices for a table's settings — ONE list for "Host a table" and Manage (they
  // used to offer different ones: no 10 s clock or 5 seats when hosting).
  const OPTIONS = {
    clock: [[0, "Off"], [10, "10s"], [15, "15s"], [20, "20s"], [30, "30s"], [45, "45s"], [60, "60s"]],
    bank: [[0, "Off"], [15, "15s"], [30, "30s"], [60, "60s"], [120, "2 min"]],
    deal: [[0, "Manual"], [2, "2s"], [3, "3s"], [5, "5s"], [8, "8s"], [12, "12s"]],
    pause: [[0.5, "0.5s"], [1, "1s"], [1.5, "1.5s"], [2.5, "2.5s"], [4, "4s"]],
    seats: (max) => Array.from({ length: Math.max(1, max - 1) }, (_, i) => [i + 2, String(i + 2)]),
  };
  const secsLabel = (v) => (Number(v) ? `${v}s` : "Off");
  // "Sat 27 Sep, 21:40" in the viewer's own time zone (the server's dates are UTC, so a
  // late game read "tomorrow" as a bare "2026-09-27"; HGH-004). The year only when it isn't this one.
  function fmtWhen(iso, withTime) {
    const t = Date.parse(iso);
    if (!Number.isFinite(t)) return String(iso || "").slice(0, 10);
    const d = new Date(t);
    const date = d.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short", year: d.getFullYear() === new Date().getFullYear() ? undefined : "numeric" });
    return withTime === false ? date : `${date}, ${d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}`;
  }

  // ------------------------------------------------------------------ toasts
  // Errors stay 7 s (a buy-in rule takes a moment to read) and are announced at once
  // (role=alert; the root is a polite live region for the rest); any toast goes away
  // when tapped. A phone at the table shows one at a time: they sit over the far seats.
  function toast(msg, kind, ms) {
    const root = $("toast-root");
    if (!root) return;
    // the same line again while it is still up is noise (stacked copies hid the table)
    if ([...root.children].some((x) => x.textContent === String(msg) && !x.classList.contains("out"))) return;
    const err = kind === "err";
    const t = h("div", { class: "toast " + (kind || ""), role: err ? "alert" : null, title: "Tap to dismiss" }, String(msg));
    const bye = () => {
      if (t.classList.contains("out")) return;
      t.classList.add("out");
      t.addEventListener("animationend", () => t.remove());  // (its fade-out; the timer covers no animation)
      setTimeout(() => t.remove(), 400);
    };
    t.addEventListener("click", bye);
    root.appendChild(t);
    const max = !C().G.gameId || globalThis.innerWidth > 760 ? 3 : 1;
    while (root.children.length > max) root.firstChild.remove();
    setTimeout(bye, ms || (err ? 7000 : 3400));
  }
  // A page-wide notice over everything (no network at start-up, signed out): title, a
  // line, one button. `null` takes it away.
  function pageNotice(o) {
    let el = $("pagenote");
    if (!o) { if (el) el.remove(); return; }
    if (!el) { el = h("div", { id: "pagenote", role: "alertdialog", "aria-modal": "true", "aria-labelledby": "pn-title" }); document.body.appendChild(el); }
    put(el, html`<div class="pn-card"><span class="pn-ico">${icon(o.icon || "i-info")}</span><h2 id="pn-title">${o.title}</h2><p>${o.text}</p></div>`);
    if (o.button) {
      const b = h("button", { class: "btn gold", type: "button", onclick: o.onClick }, o.button);
      el.firstChild.appendChild(b);
      setTimeout(() => b.focus({ preventScroll: true }), 30);
    }
  }
  function bootProblem(msg, retry) {
    if (msg == null) { if ($("pagenote") && $("pagenote").dataset.k === "boot") pageNotice(null); return; }
    pageNotice({ icon: "i-bolt", title: "Can't reach WrapGTO", text: `${msg}. Trying again by itself…`, button: "Try now", onClick: () => retry && retry() });
    $("pagenote").dataset.k = "boot";
  }
  function signedOut() {
    pageNotice({ icon: "i-user", title: "You've been signed out", text: "Your sign-in ended (signed out in another tab, or it expired). Sign in again to pick up where you were.", button: "Sign in again", onClick: () => location.reload() });
    $("pagenote").dataset.k = "signin";
  }
  // the lobby's first load failed: its "Loading…" says why (the 5 s poll retries)
  function lobbyOffline(on) {
    const lb = $("lobby");
    if (lb) lb.classList.toggle("offline", !!on);
  }

  // ------------------------------------------------------------------ modals
  // Dialogs (A11Y-004): named by their title (aria-labelledby); focus moves in — to the
  // first box, or with `autofocus: false` (no phone keyboard popping up) to the dialog
  // itself — Tab stays inside (trapFocus), and focus goes back to what opened it. A
  // dialog that fills in later (a slow load) opens at once with "Loading…" and gets its
  // content through setBody / setButtons / setTitle (HGH-002). `then(fn)` runs fn once
  // the dialog has finished closing — the next dialog opens from there (FE-007: this
  // used to be copies of the CSS timings in setTimeouts).
  function openModal(opts) {
    const root = $("modal-root");
    const layer = h("div", { class: "layer" });
    const scrim = h("div", { class: "scrim" });
    const wrap = h("div", { class: "modal-wrap" });
    U.modalSeq = (U.modalSeq || 0) + 1;
    const tid = "mt-" + U.modalSeq;
    const modal = h("div", { class: "modal" + (opts.wide ? " wide" : ""), role: "dialog", "aria-modal": "true", "aria-labelledby": tid, tabindex: "-1" });
    const head = h("div", { class: "m-head" }, h("h3", { id: tid }));
    const x = h("button", { class: "icon-btn", type: "button", "aria-label": "Close" }, icon("i-x"));
    head.appendChild(x);
    modal.appendChild(head);
    const sub = h("div", { class: "m-sub" });
    modal.appendChild(sub);
    const body = h("div", { class: "m-body" });
    modal.appendChild(body);
    const foot = h("div", { class: "m-foot" });
    modal.appendChild(foot);
    const opener = document.activeElement;
    const after = [];
    let closed = false, gone = false;
    const finish = () => {
      if (gone) return;
      gone = true;
      layer.remove();
      after.splice(0).forEach((f) => f());
    };
    const close = (val) => {
      if (closed) return;
      closed = true;
      layer.classList.remove("open");
      U.modals = U.modals.filter((m) => m !== api);
      afterTransition(modal, finish, 400);  // (gone when its fade-out ends)
      if (opts.onClose) opts.onClose(val);
      if (!U.modals.length && !U.drawer && opener && opener.isConnected && opener.focus) opener.focus({ preventScroll: true });
    };
    const setTitle = (title, subText) => {
      head.firstChild.textContent = title || "";
      if (subText !== undefined) { sub.textContent = subText || ""; sub.hidden = !subText; }
    };
    const setBody = (b) => put(body, b);  // (markup, a node, or a line of text)
    const setButtons = (list) => {
      foot.innerHTML = "";
      (list || []).forEach((b) => {
        const btn = h("button", { class: "btn " + (b.cls || ""), type: "button" }, b.label);
        btn.addEventListener("click", async () => {
          if (!b.onClick) return close(b.value);
          btn.disabled = true;
          try { const keep = await b.onClick(api); if (keep !== false) close(b.value); }
          catch (e) { if (e && e.status !== 409) { /* post() already toasted */ } }
          finally { btn.disabled = false; }
        });
        foot.appendChild(btn);
      });
      foot.hidden = !foot.children.length;
    };
    const api = {
      close, body, modal, foot, setBody, setButtons, setTitle,
      get closed() { return closed; },
      then(fn) { if (gone) fn(); else after.push(fn); return api; },
    };
    setTitle(opts.title, opts.sub || "");
    setBody(opts.body);
    setButtons(opts.buttons);
    wrap.appendChild(modal);
    layer.appendChild(scrim);
    layer.appendChild(wrap);
    root.appendChild(layer);
    x.addEventListener("click", () => close(null));
    scrim.addEventListener("click", () => close(null));
    requestAnimationFrame(() => layer.classList.add("open"));
    U.modals.push(api);
    setTimeout(() => {
      if (closed) return;
      const first = opts.autofocus !== false && body.querySelector("input:not([type=hidden]):not([disabled]),select,textarea");
      (first || modal).focus({ preventScroll: true });
    }, 60);
    return api;
  }
  // Run `fn` once `el`'s own CSS transition has ended — a timer covers one that never
  // runs (reduced motion, a background tab). The chained dialogs, the drawer and the rail
  // used to copy the CSS durations into setTimeouts (FE-007).
  function afterTransition(el, fn, fallback) {
    let done = false, t = 0;
    const end = (e) => {
      if ((e && e.target !== el) || done) return;
      done = true;
      el.removeEventListener("transitionend", end);
      clearTimeout(t);
      fn();
    };
    el.addEventListener("transitionend", end);
    t = setTimeout(end, fallback || 450);
  }
  // "Loading…" for a dialog whose content is on its way (openModal's setBody)
  const loading = () => html`<div class="m-loading" role="status"><span class="spin"></span>Loading…</div>`;
  // Tab / Shift+Tab stay inside the top dialog (or the Manage drawer): the page behind
  // it can't be reached until it closes.
  function trapFocus(e) {
    if (e.key !== "Tab") return;
    const box = U.modals.length ? U.modals[U.modals.length - 1].modal : U.drawer ? U.drawer.dr : null;
    if (!box) return;
    const list = [...box.querySelectorAll("a[href],button:not([disabled]),input:not([disabled]):not([type=hidden]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex='-1'])")]
      .filter((el) => !el.closest("[hidden]"));
    if (!list.length) { e.preventDefault(); box.focus(); return; }
    const first = list[0], last = list[list.length - 1], a = document.activeElement;
    if (!box.contains(a)) { e.preventDefault(); first.focus(); }
    else if (e.shiftKey && (a === first || a === box)) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && a === last) { e.preventDefault(); first.focus(); }
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
    if (U.menu) { closeMenu(true); return true; }
    if (U.modals.length) { U.modals[U.modals.length - 1].close(null); return true; }
    if (U.drawer && UI.closeDrawer) { UI.closeDrawer(); return true; }
    return false;
  }
  // close the top dialog, then run `fn` once it is gone (the next dialog in a chain)
  function closeTopThen(fn) {
    const top = U.modals[U.modals.length - 1];
    if (!top) { fn(); return; }
    top.then(fn);
    top.close(null);
  }

  // ------------------------------------------------------------------- menus
  // Menus (the seat menu, ≡, the club switcher, React) work from the keyboard (A11Y-013):
  // role=menu with menuitems, focus on the first item, ↑ ↓ Home End move, Tab or Escape
  // (games.play.js closeTop) close it, focus goes back to the button that opened it — and
  // the table's hotkeys are paused while it is open (games.play.js).
  function openMenu(anchor, items) {
    closeMenu();
    const m = h("div", { class: "menu", role: "menu" });
    items.forEach((it) => {
      if (it === "-") return m.appendChild(h("hr", { role: "separator" }));
      if (it.header) return m.appendChild(h("div", { class: "mh", role: "presentation" }, it.header));
      const b = h("button", { type: "button", role: "menuitem", tabindex: "-1", class: it.danger ? "danger" : "" }, html`${it.icon ? icon(it.icon) : ""}<span>${it.label}</span>`);
      b.disabled = !!it.disabled;
      if (it.hint) b.title = it.hint;
      b.addEventListener("click", () => { closeMenu(true); it.onClick(); });
      m.appendChild(b);
    });
    m.addEventListener("keydown", (e) => {
      const list = [...m.querySelectorAll("button[role=menuitem]:not([disabled])")];
      if (!list.length) return;
      const i = list.indexOf(document.activeElement);
      const go = (k) => { e.preventDefault(); e.stopPropagation(); list[(k + list.length) % list.length].focus(); };
      if (e.key === "ArrowDown") go(i + 1);
      else if (e.key === "ArrowUp") go(i < 0 ? list.length - 1 : i - 1);
      else if (e.key === "Home") go(0);
      else if (e.key === "End") go(list.length - 1);
      else if (e.key === "Tab") closeMenu(true);
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
    U.menuAnchor = anchor;
    anchor.setAttribute("aria-expanded", "true");
    const first = m.querySelector("button[role=menuitem]:not([disabled])");
    if (first) first.focus({ preventScroll: true });
    setTimeout(() => document.addEventListener("pointerdown", onDocDown, true), 0);
  }
  function onDocDown(e) { if (U.menu && !U.menu.contains(e.target)) closeMenu(); }
  // `refocus`: the menu was used from inside (an item, Escape, Tab) — focus goes back to its button
  function closeMenu(refocus) {
    const a = U.menuAnchor;
    if (U.menu) { U.menu.remove(); U.menu = null; }
    U.menuAnchor = null;
    if (a) { a.setAttribute("aria-expanded", "false"); if (refocus && a.isConnected && a.focus) a.focus({ preventScroll: true }); }
    document.removeEventListener("pointerdown", onDocDown, true);
  }

  // ------------------------------------------------------ segmented pickers
  // ONE builder for every segmented picker (create, Manage, Preferences, …; FE-002).
  // `cur` compares as text (numbers and codes alike); `keep(v)` labels the current value
  // when it isn't one of the options (a value set elsewhere), so it still shows as chosen.
  // The chosen button says so to screen readers (aria-pressed; A11Y-009). A label is text
  // or markup; `cls` adds classes to the picker ("wide", "as-modes").
  function segHtml(id, opts, cur, keep, cls) {
    const list = opts.slice();
    if (keep && cur != null && !list.some(([v]) => String(v) === String(cur))) {
      list.push([cur, keep(cur)]);
      list.sort((a, b) => Number(a[0]) - Number(b[0]));
    }
    return html`<div class="seg${cls ? " " + cls : ""}" id="${id}" role="group">${list.map(([v, l]) => {
      const on = String(v) === String(cur);
      return html`<button type="button" data-v="${v}" class="${on ? "on" : ""}" aria-pressed="${on}">${l}</button>`;
    })}</div>`;
  }
  function segWire(root) {
    root.querySelectorAll(".seg").forEach((seg) => seg.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-v]");
      if (!b || b.disabled) return;
      seg.querySelectorAll("button").forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-pressed", String(x === b)); });
      seg.dispatchEvent(new CustomEvent("pick", { detail: b.dataset.v }));
    }));
  }
  function segVal(root, id) {
    const on = root.querySelector(`#${id} button.on`);
    return on ? Number(on.dataset.v) : null;
  }
  // "Saved" beside a setting that saves on its own (Manage; games.css .saved)
  function savedFlash(node) {
    const row = node && node.closest(".field, .setrow");
    if (!row) return;
    row.classList.remove("saved");
    void row.offsetWidth;
    row.classList.add("saved");
    clearTimeout(row._savedT);
    row._savedT = setTimeout(() => row.classList.remove("saved"), 1600);
  }

  // --------------------------------------------------------- profile picture
  // (2026-09-26) The browser crops the middle square of the photo and re-encodes
  // it at 256 px (small, and free of the photo's metadata) before it is sent.
  function squareDataUrl(file) {
    return new Promise((resolve, reject) => {
      if (!/^image\//.test(file.type || "")) return reject(new Error("That file isn't a picture"));
      if (file.size > 30e6) return reject(new Error("That picture is too large"));
      const fr = new FileReader();
      fr.onerror = () => reject(new Error("Couldn't read that file"));
      fr.onload = () => {
        const img = new Image();
        img.onerror = () => reject(new Error("Couldn't open that picture"));
        img.onload = () => {
          const side = Math.min(img.naturalWidth, img.naturalHeight);
          if (!side) return reject(new Error("Couldn't open that picture"));
          const c = document.createElement("canvas");
          c.width = c.height = 256;
          const g = c.getContext("2d");
          g.imageSmoothingQuality = "high";
          g.drawImage(img, (img.naturalWidth - side) / 2, (img.naturalHeight - side) / 2, side, side, 0, 0, 256, 256);
          let out = c.toDataURL("image/webp", 0.86);
          if (!out.startsWith("data:image/webp")) out = c.toDataURL("image/jpeg", 0.88);  // (no WebP encoder)
          resolve(out);
        };
        img.src = fr.result;
      };
      fr.readAsDataURL(file);
    });
  }
  function renderMe() {
    const me = C().G.me || {};
    const nm = myName();
    put($("userchip"), html`${avatar(nm, nm, "sm", me.avatar)}<span>${nm}</span>`);
  }
  function setMyAvatar(url) {  // the lobby and the table views carry it (my_avatar)
    const me = C().G.me || {};
    if (url === undefined || (me.avatar || null) === (url || null)) return;
    me.avatar = url || null;
    renderMe();
  }
  async function saveAvatar(dataUrl) {
    try {
      const out = await C().j("/games/api/me/avatar", dataUrl
        ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ data_url: dataUrl }) }
        : { method: "DELETE" });
      setMyAvatar(out.avatar || null);
      toast(out.avatar ? "Picture saved" : "Picture removed", "ok");
    } catch (e) { toast(e.message, "err"); return false; }
  }
  // Your picture AND the name the tables show (FEAT-008: friends were stuck with their
  // full Google name, cut off on the seat plate — or an email's first half).
  function openAvatar(opts) {
    const me = (C().G.me = C().G.me || {});  // (one object: setMyName fills in the name's state)
    const nm = myName();
    let picked = null;
    const body = h("div", { class: "avup" });
    const typed = () => { const b = body.querySelector("#avup-name"); return b ? b.value.trim() : null; };
    let touched = false;  // (the box keeps what was typed; until then it follows the name's state)
    const paint = () => {
      const st = me.name_state || {};
      const value = touched ? typed() : st.chosen || (opts && opts.focusName && st.account_name) || "";
      put(body, html`<div class="avup-pic">${avatar(nm, nm, "xl", picked || me.avatar)}</div>
        <div class="avup-side"><label class="btn" for="avup-file">${icon("i-user", "sm")}Choose a photo</label>
        <input type="file" id="avup-file" accept="image/*" hidden/>
        <small class="muted">We use a square from the middle of it. Everyone in your clubs sees it at the table and on the club page.</small></div>
        <label class="field avup-name"><span>Name at the table</span><input class="input" id="avup-name" maxlength="20" autocomplete="nickname" placeholder="${st.account_name || "e.g. Sam"}" value="${value || ""}"/>
        <small class="muted">What your seat, the chat and the club show — short names fit the seat best. Empty = ${st.account_name ? `your account's name (${st.account_name})` : "\u201cPlayer\u201d and a number"}. A club can also give you a nickname of its own.</small></label>`);
      body.querySelector("#avup-name").addEventListener("input", () => { touched = true; });
      body.querySelector("#avup-file").addEventListener("change", async (e) => {
        const f = e.target.files && e.target.files[0];
        if (!f) return;
        try { picked = await squareDataUrl(f); paint(); } catch (err) { toast(err.message, "err"); }
      });
    };
    paint();
    const saveName = async () => {
      const v = typed();
      const cur = (me.name_state && me.name_state.chosen) || "";
      if (v == null || v === cur) return true;
      try {
        const st = await C().j("/games/api/me/name", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: v }) });
        setMyName(st);
        toast(v ? `The tables call you ${st.name}` : "Name reset", "ok");
        return true;
      } catch (e) { toast(e.message, "err"); return false; }
    };
    const buttons = [];
    if (me.avatar) buttons.push({ label: "Remove photo", cls: "ghost", onClick: () => saveAvatar(null) });
    buttons.push({ label: "Cancel", cls: "ghost" }, {
      label: "Save", cls: "gold",
      onClick: async () => {
        if (!(await saveName())) return false;
        return picked ? saveAvatar(picked) : true;
      },
    });
    const api = openModal({ title: (opts && opts.title) || "Your picture and name", sub: opts && opts.sub, body, buttons, autofocus: !!(opts && opts.focusName) });
    if (opts && opts.focusName) setTimeout(() => { const b = body.querySelector("#avup-name"); if (b) b.focus({ preventScroll: true }); }, 80);
    // (opened from a table link, before any lobby: the name's state comes from here)
    if (!me.name_state) C().j("/games/api/me/name").then((st) => { setMyName(st); if (!api.closed) paint(); }).catch(() => {});
  }
  // FEAT-008: the name the tables show for you — the lobby and table views carry it
  function myName() {
    const me = C().G.me || {};
    return (me.name_state && me.name_state.name) || me.name || me.email || "?";
  }
  function setMyName(st) {
    if (!st || typeof st !== "object") return;
    const me = (C().G.me = C().G.me || {});
    const was = JSON.stringify(me.name_state || null);
    me.name_state = st;
    if (JSON.stringify(st) !== was) renderMe();
  }
  // No name of your own yet ("Player 12", or your email's first half): asked once per
  // visit, the first time you open a table, with the picture dialog focused on the name.
  // (asked at most once a week in this browser: a Save with the name that is there already
  // settles it for good; a dismissal only for the week)
  const ASK_NAME_KEY = "hg.askname.v1";
  function askForName(s) {
    if (!s || !s.my_name_default || U.askedName) return;
    U.askedName = true;
    try {
      const last = Number(localStorage.getItem(ASK_NAME_KEY) || 0);
      if (Date.now() - last < 7 * 86400e3) return;
      localStorage.setItem(ASK_NAME_KEY, String(Date.now()));
    } catch (_) { /* private mode: once per visit */ }
    setTimeout(() => openAvatar({ title: "What should the table call you?", sub: "Your seat and the chat show this name. You can change it any time from your picture.", focusName: true }), 400);
  }

  // ------------------------------------------------------------ preferences
  function openPrefs() {
    const p = C().G.prefs;
    const seg = segHtml;
    const sw = (id, on) => html`<label class="switch"><input type="checkbox" id="${id}" ${on ? "checked" : ""}/><i></i></label>`;
    const row = (id, on, title, hint) => html`<div class="setrow"><div><b>${title}</b>${hint ? html`<small>${hint}</small>` : ""}</div>${sw(id, on)}</div>`;
    const vol = Math.round(p.volume * 100);
    const body = h("div", { class: "stack-14" });
    // Sounds come in three kinds (FEAT-011): someone who only wants the your-turn chime
    // used to have to mute everything. The top bar's speaker still mutes them all.
    put(body, html`<div class="grp"><h4>Sound</h4>${p.sound ? "" : html`<p class="flush">All sounds are muted — the speaker button in the top bar turns them back on.</p>`}
      ${row("p-snd-turn", p.sndTurn !== false, "Your turn", "The chime, your clock's last seconds, and a buy-in or join request waiting for you")}
      ${row("p-snd-chat", p.sndChat !== false, "Chat", "A soft ping for a new message while the chat is closed")}
      ${row("p-snd-table", p.sndTable !== false, "Table", "Cards, chips, bets and wins")}
      <label class="field"><span>Volume</span><input type="range" id="p-vol" min="0" max="100" value="${vol}" data-vars="fill:${vol}%"/></label>
      ${canVibrate() ? row("p-vibrate", p.vibrate !== false, "Vibrate on my turn") : ""}
      ${row("p-notify", p.notify, "Notification on my turn", "When this tab is in the background — tap it to come back to the table")}</div>
      <div class="grp"><h4>Table</h4><div class="field"><span>Felt</span>${seg("p-felt", [["emerald", "Emerald"], ["royal", "Royal"], ["crimson", "Crimson"], ["violet", "Violet"], ["graphite", "Graphite"]], p.felt)}</div>
      <div class="row2"><div class="field"><span>Cards</span>${seg("p-cards", [["bold", "Bold"], ["classic", "Classic"]], p.cards)}</div><div class="field"><span>Deck</span>${seg("p-deck", [["4c", "4-colour"], ["2c", "2-colour"]], p.deck)}</div></div>
      <div class="row2"><div class="field"><span>Card backs</span>${seg("p-back", [["blue", "Blue"], ["red", "Red"], ["green", "Green"], ["black", "Black"]], p.back)}</div><div class="field"><span>Amounts</span>${seg("p-unit", [["dollars", "$"], ["bb", "BB"]], p.unit)}</div></div>
      <div class="field"><span>Animations</span>${seg("p-anim", [["auto", "Auto"], ["full", "Full"], ["off", "Off"]], p.anim)}<small>Auto follows your device's Reduce Motion setting.</small></div>
      ${row("p-bubbles", p.bubbles !== false, "Chat bubbles over seats")}</div>
      <div class="grp"><h4>Playing</h4>${row("p-hot", p.hotkeys, "Keyboard shortcuts", "F fold · C check/call · R or B bet/raise · 1–6 sizes · arrows adjust · Esc cancels a pre-action · ? lists them all")}
      ${row("p-allin", p.confirmAllIn, "Confirm all-in", "Ask before putting your whole stack in (a call too)")}</div>
      <div class="grp"><h4>Private notes &amp; tags</h4><div class="setrow"><div><b>Back up or bring back</b><small>Your notes on players stay in this browser. Save them to a file before clearing it or changing phone.</small></div>
      <span class="row-inline"><button type="button" class="btn sm" id="p-notes-out">Save</button><label class="btn sm" for="p-notes-in">Restore</label><input type="file" id="p-notes-in" accept="application/json,.json" hidden/></span></div></div>`);
    // (FEAT-014 above: the notes live in this browser only — a file keeps them safe, and still private)
    segWire(body);
    const save = (patch) => { C().savePrefs(patch); const s = C().G.state; if (s) HG.ui.render(s, s, { unitChanged: true }); renderSound(); };
    [["p-felt", "felt"], ["p-cards", "cards"], ["p-deck", "deck"], ["p-back", "back"], ["p-unit", "unit"], ["p-anim", "anim"]].forEach(([id, key]) =>
      body.querySelector("#" + id).addEventListener("pick", (e) => save({ [key]: e.detail })));
    [["p-snd-turn", "sndTurn"], ["p-snd-chat", "sndChat"], ["p-snd-table", "sndTable"], ["p-vibrate", "vibrate"], ["p-bubbles", "bubbles"], ["p-hot", "hotkeys"], ["p-allin", "confirmAllIn"]].forEach(([id, key]) => {
      const box = body.querySelector("#" + id);
      if (box) box.addEventListener("change", (e) => save({ [key]: e.target.checked }));
    });
    body.querySelector("#p-vol").addEventListener("input", (e) => { e.target.style.setProperty("--fill", e.target.value + "%"); save({ volume: Number(e.target.value) / 100 }); });
    body.querySelector("#p-vol").addEventListener("change", () => HG.sound && HG.sound.play("chip"));
    body.querySelector("#p-notes-out").addEventListener("click", () => UI.exportNotes && UI.exportNotes());
    body.querySelector("#p-notes-in").addEventListener("change", (e) => { const f = e.target.files && e.target.files[0]; if (f && UI.importNotes) UI.importNotes(f); e.target.value = ""; });
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
    put(b, icon(on ? "i-vol" : "i-mute"));
    b.title = on ? "Mute sounds" : "Turn sounds on";
    b.setAttribute("aria-label", b.title);
  }

  // -------------------------------------------------------------------- rail
  function setRail(open, tab) {
    const p = C().G.prefs;
    if (tab) U.railTab = tab;
    C().savePrefs({ rail: open, railTab: U.railTab });
    $("rail").classList.toggle("closed", !open);
    $("tb-rail").classList.toggle("on", open);
    document.querySelectorAll("#rail-tabs button").forEach((b) => { b.classList.toggle("on", b.dataset.tab === U.railTab); b.setAttribute("aria-selected", String(b.dataset.tab === U.railTab)); });
    for (const k of ["chat", "log", "ledger", "hands"]) $("panel-" + k).hidden = k !== U.railTab;
    if (open && U.railTab === "chat") { U.unread = 0; renderUnread(); const sc = $("chat-log").parentNode; sc.scrollTop = sc.scrollHeight; }
    if (p.rail !== open) afterTransition($("rail"), () => HG.table.layout());  // (the felt re-fits once the rail has moved)
    const s = C().G.state;
    if (s && open) renderRail(s, true);
  }
  // It just became my turn. On a phone the side panel covers the whole table —
  // someone reading the chat never saw the buttons and the clock folded them: close
  // it (a half-typed message stays in its box). A dialog or the Manage drawer may hold
  // unsaved edits, so those stay open and a toast says it instead.
  // A phone buzzes too (FEAT-011; Preferences › Vibrate on my turn — Android: iPhones have
  // no vibration for web pages).
  const canVibrate = () => !!(globalThis.navigator && navigator.vibrate && globalThis.matchMedia && matchMedia("(pointer: coarse)").matches);
  function onMyTurn(s) {
    if (C().G.prefs.vibrate !== false && canVibrate()) { try { navigator.vibrate([90, 70, 90]); } catch (_) { /* not allowed yet */ } }
    if (C().G.prefs.rail && getComputedStyle($("rail")).position === "absolute") setRail(false);
    if (U.drawer || document.querySelector("#modal-root .modal")) toast("It's your turn", "gold", 4000);
    // screen readers hear it too (the chime and the tab title were all there was)
    const live = $("sr-live");
    if (live && s) {
      const owe = s.to_call_cents || 0;
      live.textContent = "";
      setTimeout(() => { live.textContent = owe > 0 ? `Your turn — ${C().fmtAmt(owe, s)} to call` : "Your turn — check or bet"; }, 60);
    }
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
      // mine = by user id (two members can share a name — HGT-027); an older server sends names only
      const myName = (s.seats[s.my_seat] || {}).name;
      const isMine = (m) => (m.user_id != null ? m.user_id === s.my_user_id : m.name === myName);
      const items = chat.map((m) => ({ t: timeOf(m.created_at), chat: m })).concat(events.map((e) => ({ t: e.ts * 1000, ev: e })));
      items.sort((a, b) => a.t - b.t);
      const log = $("chat-log"), sc = log.parentNode;
      const stick = sc.scrollHeight - sc.scrollTop - sc.clientHeight < 40;
      const line = (it) => (it.chat
        ? html`<div class="msg ${isMine(it.chat) ? "me" : ""}">${avatar(it.chat.name, it.chat.name, "sm", it.chat.avatar)}<div class="body"><div class="who">${it.chat.name}<time>${it.t ? hhmm(it.t) : ""}</time></div><div class="txt">${it.chat.text}</div></div></div>`
        : html`<div class="msg sys ${it.ev.kind === "win" ? "win" : ""}">${it.ev.text}</div>`);
      // New lines are ADDED (FE-008: the whole chat used to be redrawn for every line);
      // lines the server's window dropped at the top are removed. Anything else redraws.
      const keys = items.map((it) => (it.chat ? "c" + it.chat.id : "e" + it.ev.id));
      const old = U.chatKeys || [];
      const from = keys.length ? old.indexOf(keys[0]) : -1;
      const kept = from < 0 ? [] : old.slice(from);
      if (!force && items.length && kept.length && kept.length <= keys.length && kept.every((k, i) => k === keys[i]) && log.childElementCount === old.length) {
        for (let i = 0; i < from; i++) log.firstElementChild.remove();
        if (keys.length > kept.length) append(log, html`${items.slice(kept.length).map(line)}`);
      } else {
        put(log, items.length ? html`${items.map(line)}`
          : html`<div class="muted empty-note">Say hi — the dealer posts joins, rebuys and results here too.</div>`);
      }
      U.chatKeys = items.length ? keys : [];
      if (stick || firstPaint) sc.scrollTop = sc.scrollHeight;
      if (!firstPaint) {
        const fresh = chat.filter((m) => m.id > prevLast && !isMine(m)).length;
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
    const names = {}, pics = {};
    s.seats.forEach((x) => { if (!x.empty) { names[x.seat] = x.name; pics[x.seat] = x.avatar; } });
    const rows = [];
    let street = null;
    hist.forEach((x) => {
      if (x.street !== street) { street = x.street; rows.push(html`<div class="log-street">${street}</div>`); }
      const k = C().actionKind(x);
      const lb = k === "fold" ? "Fold" : k === "check" ? "Check" : k === "call" ? "Call " + C().fmtAmt(x.cents, s) : (k === "allin" ? "All-in " : "Raise to ") + C().fmtAmt(x.to_cents, s);
      rows.push(html`<div class="log-row k-${k}">${avatar(names[x.seat] || "?", names[x.seat], "sm", pics[x.seat])}<span class="nm">${names[x.seat] || "Seat " + (x.seat + 1)}</span><span class="lb">${lb}</span></div>`);
    });
    put($("log-body"), html`${s.hand_no ? html`<div class="muted num log-head">Hand #${s.hand_no} · ante ${d2(s.stakes.ante_cents)}</div>` : ""}${rows.length ? rows
      : html`<div class="muted empty-note">${s.phase === "in_hand" ? "No action yet — everyone anted." : "The action of the current hand shows up here."}</div>`}`);
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
  // Who pays whom (FEAT-001): the server's fewest payments (`settle_up`, your own lines
  // marked); an older server only sent the ledger, settled here greedily.
  function payments(s) {
    if (Array.isArray(s.settle_up)) return s.settle_up.map((p) => ({ from: p.from_name, to: p.to_name, cents: p.cents, you: p.you }));
    return settleUp(s.ledger || []);
  }
  const payLine = (p) => p.you === "pay" ? html`<b class="you">You</b> pay ${p.to}` : p.you === "get" ? html`${p.from} pays <b class="you">you</b>` : html`${p.from} pays ${p.to}`;
  function renderLedger(s, force) {
    const led = s.ledger || [];
    const sig = JSON.stringify(led) + JSON.stringify(s.settle_up || null) + C().G.prefs.unit + (U.hands ? U.hands.statsSig : "");
    if (sig === U.ledgerSig && !force) return;
    U.ledgerSig = sig;
    // your own row opens your receipt (FEAT-002); the host's, anyone's
    const tappable = (r) => r.user_id === s.my_user_id || s.is_host;
    const rows = led.map((r) =>
      html`<tr class="${r.user_id === s.my_user_id ? "me" : ""} ${r.seated ? "" : "gone"} ${tappable(r) ? "tap" : ""}"${tappable(r) ? html` data-uid="${Number(r.user_id)}" tabindex="0" role="button" aria-label="${r.name}: every buy-in and cash-out"` : ""}><td title="${r.name}">${r.name}</td><td>${d2(r.buyin_cents)}</td><td>${r.seated ? d2(r.stack_cents) : d2(r.leftover_cents)}</td><td class="${r.net_cents > 0 ? "pos" : r.net_cents < 0 ? "neg" : ""}">${r.net_cents > 0 ? "+" : ""}${d2(r.net_cents)}</td></tr>`);
    const pays = payments(s);
    const stats = (U.hands && U.hands.stats) || [];
    const h2h = (U.hands && U.hands.h2h) || [];
    const G = gameOf(s.variant);
    put($("ledger-body"), html`<table class="ledger"><thead><tr><th>Player</th><th>Buy-in</th><th>Stack</th><th>Net</th></tr></thead><tbody>${rows}</tbody></table>
      <div class="muted fine-print">Stacks settle at the end of each hand. Players who left show what they cashed out. ${UI.openReceipt ? (s.is_host ? "Tap a row for every buy-in and cash-out." : "Tap your row for every buy-in and cash-out.") : ""}</div>
      <div class="rsec"><h4>Settle up</h4>${pays.length ? html`<div class="muted fine-print lead">The fewest payments that square everyone${s.status === "open" ? " if the game ended now" : ""}.</div>${pays.map((p) => html`<div class="settle-row ${p.you ? "mine" : ""}"><span>${payLine(p)}</span><b>${d2(p.cents)}</b></div>`)}` : html`<div class="muted">Everyone is even.</div>`}
      ${pays.length ? html`<button class="btn sm block gap-top" id="settle-copy">${icon("i-copy", "sm")}Copy summary</button>` : ""}</div>
      ${stats.length ? html`<div class="rsec"><h4>Session stats</h4><table class="ledger"><thead><tr><th>Player</th><th>Hands</th><th title="Hands won">Hands won</th><th title="Biggest pot won">Best</th><th title="Average score of their decisions against the network (0-100)">Acc.</th></tr></thead><tbody>${stats.map((r) =>
        html`<tr class="${r.is_me ? "me" : ""}"><td>${r.name}</td><td>${r.hands}</td><td>${r.wins}</td><td>${d2(r.biggest_win_cents)}</td><td>${r.accuracy == null ? "–" : Math.round(r.accuracy) + "%"}</td></tr>`)}</tbody></table>
        <div class="muted fine-print">${G.graded ? "Accuracy = how closely each decision matched the network, graded in the background after every hand." : `${G.label} decisions are not graded — there is no ${G.label} network yet.`}</div></div>` : ""}
      ${h2h.length ? html`<div class="rsec"><h4>Head to head — this table</h4>${h2h.map((x) => html`<div class="settle-row"><span>${x.to} is up on ${x.from}</span><b>${d2(x.cents)}</b></div>`)}</div>` : ""}
      <button type="button" class="btn sm block gap-top" id="ledger-myhands">${icon("i-chart", "sm")}My hands &amp; stats</button>`);
    const mh = $("ledger-myhands");
    if (mh) mh.addEventListener("click", () => UI.openMyHands(""));
    $("ledger-body").querySelectorAll("tr.tap[data-uid]").forEach((tr) => {
      const open = () => UI.openReceipt && UI.openReceipt(Number(tr.dataset.uid));
      tr.addEventListener("click", open);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
    });
    const cp = $("settle-copy");
    if (cp) cp.addEventListener("click", async () => {
      const text = `${s.name} — settle up\n` + led.map((r) => `${r.name}: ${r.net_cents >= 0 ? "+" : ""}${d2(r.net_cents)}`).join("\n") + "\n\n" + pays.map((p) => `${p.from} pays ${p.to} ${d2(p.cents)}`).join("\n");
      try { await navigator.clipboard.writeText(text); toast("Summary copied", "ok"); } catch (_) { toast("Couldn't copy", "err"); }
    });
    if (!U.hands && !U.handsLoading && canBrowse(s)) loadHands(s);
  }

  // Who may browse a table's hands: every member of its club (SEC-008 — the server says
  // so with `can_browse_hands`; an older server let only the table's own players).
  const canBrowse = (s) => !!(s && (s.can_browse_hands != null ? s.can_browse_hands : s.is_member));

  // The table's hands, newest first, 40 at a time. `older` asks for the page before the
  // oldest one shown (HGH-001: the History tab used to stop at the last 40); a refresh
  // after a new hand keeps the older pages already loaded.
  async function loadHands(s, older) {
    if (U.handsLoading) return;
    U.handsLoading = true;
    const have = U.hands && U.hands.table === s.id ? U.hands : null;
    try {
      const before = older && have && have.hands.length ? have.hands[have.hands.length - 1].hand_no : null;
      const data = await C().j(`/games/api/tables/${s.id}/hands?limit=40` + (before ? `&before=${before}` : ""));
      if (before) {
        have.hands = have.hands.concat(data.hands || []);
        have.more = !!data.more;
      } else {
        data.table = s.id;
        data.statsSig = JSON.stringify([data.stats || [], data.h2h || []]);
        if (have && (data.hands || []).length) {
          const newest = data.hands[data.hands.length - 1].hand_no;
          const kept = have.hands.filter((x) => x.hand_no < newest);
          if (kept.length) { data.hands = data.hands.concat(kept); data.more = have.more; }
        }
        U.hands = data; U.handsFor = `${s.id}:${s.last_hand_no}`;
      }
    } catch (_) {
      if (!older) { U.hands = { hands: [], stats: [], statsSig: "", denied: true, table: s.id }; U.handsFor = `${s.id}:${s.last_hand_no}`; }
    }
    U.handsLoading = false;
    const cur = C().G.state;
    if (cur && cur.id === s.id) renderRail(cur, true);
  }
  function miniCards(list, extra) {
    // (a PLO67 hand holds up to seven: its row of mini cards tucks together to fit — games.css .many)
    const many = (list || []).length > 5 ? " many" : "";
    return html`<span class="mini-cards ${extra || ""}${many}" data-cards="${(list || []).join(",")}"></span>`;
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
    if (!canBrowse(s)) { put(body, html`<div class="muted empty-note">Hand history is for the players of this table's club.</div>`); return; }
    if (U.handsFor !== key && !U.handsLoading) { loadHands(s); if (!U.hands) put(body, html`<div class="muted empty-note">Loading hands…</div>`); return; }
    if (!U.hands || (!force && body.dataset.k === key + C().G.prefs.unit)) return;
    body.dataset.k = key + C().G.prefs.unit;
    const list = U.hands.hands || [];
    put(body, list.length ? "" : html`<div class="muted empty-note">No finished hands yet. Every hand played here is saved — the cards you may see, the boards and who won.</div>`);
    list.forEach((x) => {
      const net = x.my_delta_cents;
      const row = h("button", { class: "hand-row", type: "button", "aria-label": `Hand ${x.hand_no}` },
        // your hand framed in gold, then BOTH boards stacked (HGH-005: board 1 alone could
        // show the board you lost while your money came from board 2)
        html`<span class="no">#${Number(x.hand_no)}</span><span class="hr-cards">${x.my_hole ? miniCards(x.my_hole, "mine") : ""}<span class="boards2${x.my_hole ? " gap board" : ""}">${miniCards(x.board_a)}${miniCards(x.board_b)}</span></span>
        <span class="net ${net > 0 ? "pos" : net < 0 ? "neg" : "muted"}">${net == null ? "—" : (net > 0 ? "+" : "") + d2(net)}</span>
        <span></span><span class="who">${(x.winners || []).map((w) => w.name).join(", ") || "Split pot"} · pot ${d2(x.pot_cents)}</span><span class="muted num small">${x.my_accuracy == null ? (x.showdown ? "Showdown" : "") : Math.round(x.my_accuracy) + "%"}</span>`);
      row.addEventListener("click", () => UI.openHand(s.id, x.hand_no));
      body.appendChild(row);
    });
    fillMiniCards(body);
    if (U.hands.more) {
      const older = h("button", { class: "btn sm block gap-top", type: "button" }, U.handsLoading ? "Loading…" : "Load older hands");
      older.addEventListener("click", () => { older.disabled = true; older.textContent = "Loading…"; loadHands(s, true); });
      body.appendChild(older);
    }
    const all = h("button", { class: "btn sm block gap-top", type: "button" }, html`${icon("i-chart", "sm")}My hands &amp; stats`);
    all.addEventListener("click", () => UI.openMyHands(""));
    body.appendChild(all);
    // FEAT-003: the whole session's hands as a file — only the cards you could see
    if (list.length) {
      const base = `/games/api/tables/${encodeURIComponent(s.id)}/hands/export?format=`;
      body.appendChild(h("div", { class: "dl-row" },
        html`<span class="muted">Download the session</span><a class="btn sm ghost" href="${base}txt" download>Text</a><a class="btn sm ghost" href="${base}json" download>JSON</a>`));
    }
  }

  // -------------------------------------------------------------------- init
  // The page chrome: the top bar, the rail, chat. Each feature module wires its own
  // part of the page from its entry in UI.onInit (games.lobby.js: the lobby).
  function init() {
    renderMe();
    document.addEventListener("keydown", trapFocus);
    $("userchip").addEventListener("click", openAvatar);
    // a profile picture that can't load leaves the initials under it
    document.addEventListener("error", (e) => {
      const t = e.target;
      if (t && t.tagName === "IMG" && t.parentElement && t.parentElement.classList.contains("av")) t.remove();
    }, true);
    $("brand-link").addEventListener("click", (e) => { e.preventDefault(); C().showLobby(false); });
    $("tb-back").addEventListener("click", () => C().showLobby(false));
    $("tb-invite").addEventListener("click", () => UI.copyInvite(C().G.gameId));
    $("tb-manage").addEventListener("click", () => UI.openDrawer(((C().G.state || {}).requests || []).length ? "chips" : null));
    $("tb-run").addEventListener("click", () => { const s = C().G.state; if (s) C().tablePost("run", { running: !s.running }).catch(() => {}); });
    $("rabbit-btn").addEventListener("click", () => C().tablePost("rabbit").catch(() => {}));
    $("tb-watch").addEventListener("click", (e) => {
      const s = C().G.state, names = (s && s.spectators) || [];
      openMenu(e.currentTarget, [{ header: `${names.length} watching` }].concat(names.map((n) => ({ icon: "i-eye", label: n, disabled: true, onClick: () => {} }))));
    });
    $("tb-info").addEventListener("click", () => UI.openInfo());
    $("tb-prefs").addEventListener("click", openPrefs);
    $("tb-seat").addEventListener("click", (e) => UI.seatMenu(e.currentTarget));
    $("tb-more").addEventListener("click", (e) => {
      const s = C().G.state, on = !!C().G.prefs.sound, anchor = e.currentTarget;
      // (a phone has no React button: the dock's right side is hidden there)
      const react = () => setTimeout(() => openMenu(anchor, [{ header: "React" }].concat(Object.entries(HG.table.EMOTES).map(([k, g]) => ({
        label: `${g}  ${(HG.table.EMOTE_NAMES || {})[k] || k}`, onClick: () => C().tablePost("react", { emote: k }).catch(() => {}),
      })))), 0);
      // A phone's top bar keeps room for the table's name (HGT-003): the seat menu and the
      // shuffle's shield live here instead (games.css hides their buttons ≤ 560 px).
      const phone = !!(globalThis.matchMedia && matchMedia("(max-width: 560px)").matches);
      const fair = $("tb-fair");
      const seat = phone && s && UI.seatItems ? UI.seatItems(s) : [];
      openMenu(anchor, [
        ...seat, ...(seat.length ? ["-"] : []),
        { header: s ? s.name : "Table" },
        { icon: "i-link", label: "Copy invite link", disabled: !s || s.status !== "open", onClick: () => UI.copyInvite(C().G.gameId) },
        { icon: "i-info", label: "Table info", onClick: () => UI.openInfo() },
        { icon: "i-clock", label: "Last hand", disabled: !s || !s.last_hand_no || !canBrowse(s), onClick: () => UI.openHand(s.id, C().G.state.last_hand_no) },
        { icon: "i-smile", label: "React", disabled: !s || !Number.isInteger(s.my_seat), onClick: react },
        ...(phone && fair && !fair.hidden && HG.fair ? [{ icon: "i-shield", label: `Shuffle: ${fair.lastElementChild.textContent}`, onClick: () => HG.fair.openPanel() }] : []),
        "-",
        // (an action, not a status: "Sound on" read like either)
        { icon: on ? "i-mute" : "i-vol", label: on ? "Mute sounds" : "Turn sounds on", onClick: () => { C().savePrefs({ sound: !on }); renderSound(); } },
        { icon: "i-sliders", label: "Preferences", onClick: openPrefs },
        { icon: "i-user", label: "Your picture…", onClick: openAvatar },
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
    put(tray, html`${Object.entries(HG.table.EMOTES).map(([k, g]) => { const nm = (HG.table.EMOTE_NAMES || {})[k] || k; return html`<button type="button" data-e="${k}" title="${nm}" aria-label="${nm}">${g}</button>`; })}`);
    tray.addEventListener("click", (e) => { const b = e.target.closest("button[data-e]"); if (b) { C().tablePost("react", { emote: b.dataset.e }).catch(() => {}); tray.hidden = true; } });
    $("emote-btn").addEventListener("click", () => { tray.hidden = !tray.hidden; });
    U.railTab = C().G.prefs.railTab || "chat";
    const narrow = globalThis.matchMedia && matchMedia("(max-width: 1080px)").matches;
    if (narrow) C().G.prefs.rail = false;
    setRail(!!C().G.prefs.rail, U.railTab);
    renderSound();
    UI.onInit.forEach((f) => f());
    if (HG.play) HG.play.init();
  }

  // A new version of the client is live (games.js offerUpdate; OPS-039). At a table the
  // dock's status strip offers the refresh between hands (games.play.js); the lobby has
  // this card. "Later" hides it until the page next comes back to the lobby.
  function showUpdate() {
    let card = $("updcard");
    const inLobby = !C().G.gameId;
    if (!card) {
      if (!inLobby) return;
      card = h("div", { id: "updcard", class: "joinreq updcard", role: "status" },
        html`<div class="jr-txt"><b>A new version is ready</b><small>Refresh to get it — it only takes a second.</small></div>`);
      const btns = h("div", { class: "jr-btns" });
      btns.appendChild(h("button", { class: "btn sm ghost", type: "button", onclick: () => { U.updLater = true; card.hidden = true; } }, "Later"));
      btns.appendChild(h("button", { class: "btn sm gold", type: "button", onclick: () => C().reloadForUpdate() }, "Refresh"));
      card.appendChild(btns);
      document.body.appendChild(card);
    }
    card.hidden = !inLobby || !!U.updLater;
  }

  function showLobby() {
    $("lobby").hidden = false; $("table-view").hidden = true; document.body.classList.add("in-lobby");
    if ($("review-view")) $("review-view").hidden = true;
    U.updLater = false;
    if (UI.closeDrawer) UI.closeDrawer();
    U.lobbySig = "";
    if (UI.loadClub) UI.loadClub(true);
  }
  function showTable() {
    $("lobby").hidden = true; $("table-view").hidden = false;
    if ($("review-view")) $("review-view").hidden = true;
    if ($("updcard")) $("updcard").hidden = true;  // (the table offers it in the dock)
    document.body.classList.remove("in-lobby");
    U.eventSeen = null; U.chatSig = ""; U.logSig = ""; U.ledgerSig = ""; U.hands = null; U.handsFor = null; U.unread = 0;
    $("hands-body").dataset.k = "";
  }
  function renderConn() {
    const c = $("conn"), st = C().G.conn;
    c.className = "conn " + (st === "ok" ? "" : st);
    // ("Connected", not "Live": the game's own Live / Paused pill sits next to it)
    c.lastChild.textContent = st === "ok" ? "Connected" : st === "slow" ? "Slow connection" : "Reconnecting…";
    // A lost connection must be SEEN — on a phone the top bar has no room for the
    // indicator above, and the frozen table looked perfectly normal. But the live feed
    // reconnects by itself: a blip under CONN_GRACE_MS raises no bar and no "Back online".
    let bar = $("connbar");
    if (!bar) {
      bar = h("div", { id: "connbar", role: "status", "aria-live": "polite" }, html`<i></i><span><b>Connection lost</b> — reconnecting…</span>`);
      bar.hidden = true;
      document.body.appendChild(bar);
    }
    clearTimeout(U.connTimer);
    if (st === "off") {
      if (bar.hidden) U.connTimer = setTimeout(() => { if (C().G.conn === "off") { bar.hidden = false; U.connShown = true; } }, CONN_GRACE_MS);
    } else {
      bar.hidden = true;
      if (U.connShown && st === "ok") toast("Back online", "ok");
      U.connShown = false;
    }
  }

  function renderTop(s) {
    $("table-title").textContent = s.name;
    $("table-sub").textContent = `${gameOf(s.variant).label} bomb pot · ${d2(s.stakes.bb_cents)} bb · ante ${d2(s.stakes.ante_cents)}${s.club ? " · " + s.club.name : ""}`;
    const st = $("tb-status");
    const label = s.status !== "open" ? "Closed" : s.running ? "Live" : "Paused";
    st.textContent = label;
    st.title = label;  // (a small phone shows "Live" as its dot only)
    st.className = "pill " + (label === "Live" ? "live" : label === "Paused" ? "paused" : "");
    $("tb-hand").textContent = s.hand_no ? `Hand #${s.hand_no}` : "";
    $("tb-hand").hidden = !s.hand_no;
    $("tb-hostpill").hidden = !s.is_host;
    $("tb-manage").hidden = !(s.is_host && s.status === "open");
    const run = $("tb-run");
    // (not when the felt or the dock already offers "Start game": one Start — HGT-004)
    run.hidden = !(s.is_host && s.status === "open") || !!(HG.play && HG.play.startOnFelt && HG.play.startOnFelt(s));
    if (!run.hidden) {
      const busy = s.phase === "in_hand" || (s.runout && s.runout.blocking);
      const label = s.running ? (busy ? "Pause after hand" : "Pause") : "Start game";
      const k = `${s.running}:${busy}:${s.eligible_count < 2}`;
      if (run.dataset.k !== k) {
        run.dataset.k = k;
        run.className = "btn sm " + (s.running ? "" : "start");
        // (a phone gets the short label: the long one squeezed the status pill off the bar)
        put(run, html`${icon(s.running ? "i-pause" : "i-play", "sm")}<span class="lbl-l">${label}</span><span class="lbl-s">${s.running ? "Pause" : "Start"}</span>`);
        run.disabled = !s.running && s.eligible_count < 2;
        run.title = s.running ? "Pause the game (the current hand finishes first)" : s.eligible_count < 2 ? "Needs two players with more than the ante" : "Start dealing";
        run.setAttribute("aria-label", s.running ? (busy ? "Pause after this hand" : "Pause the game") : "Start the game");  // (a phone shows it as its icon)
      }
    }
    const nReq = s.is_host ? (s.requests || []).length : 0;
    $("tb-req").hidden = !nReq; $("tb-req").textContent = String(nReq);
    const nw = (s.spectators || []).length;
    $("tb-watch").hidden = !nw; $("tb-watch-n").textContent = String(nw);
    $("tb-invite").hidden = s.status !== "open";
  }

  function handleEvents(s, prev) {
    const evs = s.events || [];
    const last = evs.length ? evs[evs.length - 1].id : 0;
    // the server restarted (new epoch): its event ids start again at 1 — show them
    if (prev && prev.id === s.id && prev.epoch && s.epoch && prev.epoch !== s.epoch) U.eventSeen = 0;
    if (U.eventSeen != null && prev && prev.id === s.id) {
      // Pop-ups are for what concerns YOU (2026-09-28): joins, rebuys, auto top-ups, leaves
      // and the host's settings used to be toasted too — on a phone they sat over the far
      // seats most hands. They are all in the chat (the dealer's lines).
      evs.filter((e) => e.id > U.eventSeen).slice(-3).forEach((e) => {
        if (e.kind === "timeout" && e.seat === s.my_seat) { toast("You ran out of time — " + (e.text.includes("folded") ? "your hand was folded" : "you were checked"), "err", 5000); return; }
        // (a request waiting for YOU sounds like your turn — its "ask" is in that group of sounds, FEAT-011)
        if (e.kind === "request") { if (s.is_host && /asks to/.test(e.text)) { toast(e.text + " — tap their seat to answer", "gold", 6000); HG.sound && HG.sound.play("ask"); } return; }
        if (e.kind === "joinreq") { if ((s.join_requests || []).length) HG.sound && HG.sound.play("ask"); return; }  // the card says it
        if (e.kind === "fair") {  // a redone shuffle is always said out loud — and explained to the one it names
          if (Number.isInteger(s.my_seat) && (e.seats || []).includes(s.my_seat)) toast("Your device didn't confirm the shuffle in time, so it was redone. Keep this page open and in front between hands.", "err", 9000);
          else toast(e.text, "gold", 6000);
          return;
        }
        if (e.kind === "join") { HG.sound && HG.sound.play("sit"); return; }
        if (e.kind === "host") { if (s.is_host && prev && !prev.is_host) toast("You're the host now — Manage is in the top bar", "gold", 5000); return; }
        // the game stopping on its own (an error, a hand cut short) is news; start / pause show in the top bar
        if (e.kind === "run" && !/^Game (started|paused|pauses after)/.test(e.text)) toast(e.text, "", 6000);
      });
    }
    U.eventSeen = last;
    if (prev && prev.id === s.id && prev.last_hand_no !== s.last_hand_no) { U.handsFor = null; }
  }

  function render(s, prev, opts) {
    if (s.game) setGames(s.game);  // (the server's word on this table's game)
    setMyAvatar(s.my_avatar);
    askForName(s);  // (FEAT-008: no name of your own yet — asked once)
    if (UI.renderJoinReqs) UI.renderJoinReqs(s.join_requests);
    renderTop(s);
    HG.table.render(s, prev, opts);
    if (HG.play) HG.play.render(s, prev, opts);
    handleEvents(s, prev);
    renderRail(s, !!(opts && opts.unitChanged));
    if (U.drawer && UI.paintDrawer) UI.paintDrawer(s, false);
    if (UI.pendingMove) UI.pendingMove(s);  // (a change of seats asked for during the hand — FEAT-013)
    // A first look at a game's table: how a hand works (FEAT-009; games.manage.js). A
    // moment later, so a hand opened by a link (the replayer) comes first and wins.
    if (!prev || prev.id !== s.id) {
      clearTimeout(U.guideTimer);
      U.guideTimer = setTimeout(() => { const cur = C().G.state; if (cur && cur.id === s.id && UI.maybeGuide) UI.maybeGuide(cur); }, 900);
    }
  }

  // Shared building blocks for the feature modules (games.lobby.js, games.history.js,
  // games.seat.js, games.manage.js): each takes what it needs from HG.uikit at load.
  HG.uikit = {
    $, C, html, put, append, icon, U, h, avatar, moneyInput, d2, d0, GAMES, gameOf, gameNote, setGames, burnRules, OPTIONS, secsLabel, fmtWhen,
    toast, openModal, confirmDialog, closeTop, closeTopThen, afterTransition, loading, openMenu, closeMenu, segHtml, segWire, segVal, savedFlash,
    setMyAvatar, setMyName, miniCards, fillMiniCards, loadHands, canBrowse,
  };
  Object.assign(UI, {
    init, render, showLobby, showTable, renderConn, toast, openModal, confirmDialog, openMenu, closeTop,
    openPrefs, setRail, onMyTurn, openAvatar, setMyAvatar, setMyName, showUpdate, bootProblem, signedOut, lobbyOffline,
    renderDock: (s) => HG.play && HG.play.render(s, s),
    onClock: (left, tm) => HG.play && HG.play.onClock(left, tm),
  });
})();
