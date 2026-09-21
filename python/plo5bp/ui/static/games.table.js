"use strict";
// Home games — the table renderer.
//
// Every seat, card, bet and chip is a PERSISTENT node that is updated in
// place; `render(s, prev)` diffs the new server state against the previous
// one and turns the differences into motion (deal, bet -> pot, board flips,
// mucks, awards). The old client rebuilt the whole table from scratch on every
// 450 ms poll, which is why nothing could animate.
(function () {
  const HG = (globalThis.HG = globalThis.HG || {});
  const RANK = "23456789TJQKA";
  const SUIT = ["c", "d", "h", "s"];
  const GLYPH = { c: "♣", d: "♦", h: "♥", s: "♠" };
  const RING = 2 * Math.PI * 46;
  const EMOTES = {
    gg: "🤝", nh: "👏", ty: "🙏", gl: "🍀", lol: "😂", wow: "😮",
    cry: "😭", angry: "😡", fire: "🔥", clap: "👌", think: "🤔", ship: "🚢",
  };

  const $ = (id) => document.getElementById(id);
  const T = {
    ready: false, tableId: null, n: 0, hero: 0, seated: false,
    seats: [], bets: [], geom: null, boards: { a: [], b: [] },
    boardCards: { a: [], b: [] }, heroCards: [], potCents: null,
    awardKey: null, foldoutKey: null, handNo: null, chatSeen: null, reactSeen: 0,
    timer: { key: null, deadline: 0, total: 1, seat: null, bank: false, lastTick: null }, raf: 0,
    ro: null,
  };

  const el = (tag, cls, html) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  };
  const icon = (id) => `<svg class="ico"><use href="#${id}"/></svg>`;
  const anim = () => (HG.core && HG.core.G.prefs.anim) !== "off";
  const play = (n) => HG.sound && HG.sound.play(n);
  const fmt = (c, s) => HG.core.fmtAmt(c, s);

  function hueOf(key) {
    let h = 0;
    const str = String(key);
    for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) >>> 0;
    return (h * 47) % 360;
  }
  function initials(name) {
    const parts = String(name || "?").trim().split(/\s+/).filter(Boolean);
    if (!parts.length) return "?";
    const a = parts[0][0] || "?";
    const b = parts.length > 1 ? parts[parts.length - 1][0] : (parts[0][1] || "");
    return (a + b).toUpperCase();
  }
  HG.avatar = { hueOf, initials };

  // ------------------------------------------------------------------ cards
  function cardEl(c, extra) {
    const e = el("div", "card" + (extra ? " " + extra : ""), '<div class="card-in"><div class="card-f"></div><div class="card-b"></div></div>');
    setCard(e, c);
    return e;
  }
  function setCard(e, c) {
    const prev = e.dataset.c;
    if (prev === String(c)) return false;
    e.dataset.c = String(c);
    e.classList.remove("s-c", "s-d", "s-h", "s-s");
    if (c != null && c >= 0) {
      const suit = SUIT[c % 4];
      const r = RANK[(c / 4) | 0];
      e.classList.add("s-" + suit);
      e.firstChild.firstChild.innerHTML = `<b class="hg-card-bigrank">${r === "T" ? "10" : r}</b><i>${GLYPH[suit]}</i>`;
      e.classList.remove("is-down");
    } else {
      e.classList.add("is-down");
    }
    return true;
  }
  HG.cards = { cardEl, setCard };

  // --------------------------------------------------------------- geometry
  function stadiumPoint(d, W, H) {
    // Distance `d` along a stadium (pill) W x H, from the bottom centre,
    // clockwise on screen (bottom -> left -> top -> right).
    if (W >= H) {
      const r = H / 2, L = W - 2 * r, P = 2 * L + 2 * Math.PI * r;
      d = ((d % P) + P) % P;
      if (d <= L / 2) return [-d, r];
      d -= L / 2;
      if (d <= Math.PI * r) { const th = Math.PI / 2 + d / r; return [-L / 2 + r * Math.cos(th), r * Math.sin(th)]; }
      d -= Math.PI * r;
      if (d <= L) return [-L / 2 + d, -r];
      d -= L;
      if (d <= Math.PI * r) { const th = -Math.PI / 2 + d / r; return [L / 2 + r * Math.cos(th), r * Math.sin(th)]; }
      d -= Math.PI * r;
      return [L / 2 - d, r];
    }
    const r = W / 2, L = H - 2 * r, P = 2 * L + 2 * Math.PI * r;
    d = ((d % P) + P) % P;
    const q = (Math.PI * r) / 2;
    if (d <= q) { const th = Math.PI / 2 + d / r; return [r * Math.cos(th), L / 2 + r * Math.sin(th)]; }
    d -= q;
    if (d <= L) return [-r, L / 2 - d];
    d -= L;
    if (d <= Math.PI * r) { const th = Math.PI + d / r; return [r * Math.cos(th), -L / 2 + r * Math.sin(th)]; }
    d -= Math.PI * r;
    if (d <= L) return [r, -L / 2 + d];
    d -= L;
    const th = d / r;
    return [r * Math.cos(th), L / 2 + r * Math.sin(th)];
  }
  function perimeter(W, H) {
    const r = Math.min(W, H) / 2;
    return 2 * (Math.max(W, H) - 2 * r) + 2 * Math.PI * r;
  }

  function computeGeom() {
    const box = $("stage-box").getBoundingClientRect();
    const bw = Math.max(240, box.width), bh = Math.max(200, box.height);
    const portrait = bw / bh < 0.92;
    // WIDE = a phone on its side. Height is the scarce thing, so the table
    // goes flat and wide, the two boards sit SIDE BY SIDE, the hero's plate
    // moves beside the hero's cards, and nobody is seated along the bottom
    // edge (the action buttons overlay that corner — see games.css).
    const wide = !portrait && !!(globalThis.matchMedia && matchMedia("(max-height: 480px) and (orientation: landscape)").matches);
    // The table takes the shape of the space it is given (within reason): a
    // squarer window gets a rounder table instead of letterbox bars.
    const aspect = portrait ? Math.max(0.52, Math.min(0.74, bw / bh))
      : wide ? Math.max(1.9, Math.min(2.5, bw / bh))
      : Math.max(1.25, Math.min(1.5, bw / bh));
    let w = Math.min(bw, bh * aspect, 1680);
    let h = w / aspect;
    if (h > bh) { h = bh; w = h * aspect; }
    const u = portrait ? w / 58 : w / 100;
    // Felt rectangle = the CSS insets of #felt (keep the two in step).
    const ins = portrait ? { t: 0.075, r: 0.1, b: 0.16, l: 0.1 } : { t: 0.105, r: 0.085, b: 0.14, l: 0.085 };
    // Seats along the top edge table their cards ABOVE the avatar at showdown:
    // keep that much air over them however flat the window is.
    ins.t = Math.max(ins.t, (u * 9.3) / h);
    if (wide) { ins.t = (u * 7) / h; ins.b = (u * 3) / h; ins.l = ins.r = 0.07; }
    const fx = w * ins.l, fy = h * ins.t, fw = w * (1 - ins.l - ins.r), fh = h * (1 - ins.t - ins.b);
    return { w, h, u, portrait, wide, fx, fy, fw, fh, cx: fx + fw / 2, cy: fy + fh / 2 };
  }

  function layout() {
    if (!T.ready) return;
    const g = (T.geom = computeGeom());
    const stage = $("stage");
    stage.style.width = g.w + "px";
    stage.style.height = g.h + "px";
    stage.style.setProperty("--u", g.u + "px");
    stage.classList.toggle("portrait", g.portrait);
    stage.classList.toggle("wide", g.wide);
    const felt = $("felt");
    felt.style.inset = `${(g.fy / g.h) * 100}% ${(1 - (g.fx + g.fw) / g.w) * 100}% ${(1 - (g.fy + g.fh) / g.h) * 100}% ${(g.fx / g.w) * 100}%`;
    $("center").style.top = ((g.fy + g.fh * (g.portrait ? 0.43 : g.wide ? 0.47 : 0.465)) / g.h) * 100 + "%";
    $("banner").style.top = $("center").style.top;
    const n = T.n;
    if (!n) return;
    const pad = g.u * 0.4;
    const W = g.fw + 2 * pad, H = g.fh + 2 * pad;
    const P = perimeter(W, H);
    for (let i = 0; i < n; i++) {
      const rel = (i - T.hero + n) % n;
      let [px, py] = stadiumPoint((rel * P) / n, W, H);
      if (g.wide) {
        // opponents share the top and the two ends; the bottom edge is the hero's
        const r0 = Math.min(W, H) / 2, d0 = (Math.max(W, H) - 2 * r0) / 2 + Math.PI * r0 * 0.5;
        if (rel === 0) { px = -g.u * 25.2; py = g.h - g.u * 8.6 - g.cy; }
        else [px, py] = stadiumPoint(n > 2 ? d0 + ((rel - 1) / (n - 2)) * (P - 2 * d0) : P / 2, W, H);
      }
      const x = g.cx + px, y = g.cy + py;
      const sv = T.seats[i];
      sv.x = x; sv.y = y; sv.rel = rel;
      sv.el.style.left = (x / g.w) * 100 + "%";
      sv.el.style.top = (y / g.h) * 100 + "%";
      // bet spot: toward the centre, clear of the pot / boards / hero cards
      const dx = g.cx - x, dy = g.cy - y;
      const dist = Math.hypot(dx, dy) || 1;
      // far enough along the line to the centre to clear this seat's own box
      // (fan above the avatar, plate + badge below it, half a pill of margin)
      const ux = dx / dist, uy = dy / dist;
      const rx = Math.abs(ux) > 0.05 ? 10.4 / Math.abs(ux) : 99;
      const ry = Math.abs(uy) > 0.05 ? (uy > 0 ? 10.3 : 7.4) / Math.abs(uy) : 99;
      const r = Math.max(9, Math.min(16, Math.min(rx, ry)));
      let bx = x + ux * g.u * r, by = y + uy * g.u * r;
      const central = Math.abs(dx) < g.u * 9;
      sv.central = central;
      if (central && dy > 0) { bx = g.cx; by = y + g.u * 9.6; } // straight under the top seat's badge, above the pot
      if (central && dy < 0) { bx = g.cx + g.u * (g.portrait ? 17.5 : 20.5); by = y - g.u * 8; }
      if (g.wide) {
        const potY = g.fy + g.fh * 0.47 - g.u * 4;
        if (rel === 0) { bx = g.cx - g.u * 22; by = g.h - g.u * 12.6; }
        else if (central) { bx = g.cx + g.u * 13; by = potY; }
      }
      sv.bx = bx; sv.by = by;
      // dealer button: beside the bet line; next to the avatar for the seats
      // on the centre line (the hero's cards / the pot sit on that line)
      if (g.wide && rel === 0) { sv.dx = x - g.u * 7.6; sv.dy = y + g.u * 0.4; }
      else if (central) { sv.dx = x + (dy > 0 ? 8.2 : -8.2) * g.u; sv.dy = y + (dy > 0 ? 0.6 : -0.6) * g.u; }
      else {
        const ang = Math.atan2(dy, dx) + 0.5;
        sv.dx = x + Math.cos(ang) * g.u * 10.4;
        sv.dy = y + Math.sin(ang) * g.u * 10.4;
      }
      // opened (face-up) cards sit above the avatar; keep the row on the stage
      const half = g.u * 7.2;
      let shift = 0;
      if (x - half < g.u) shift = g.u - (x - half);
      if (x + half > g.w - g.u) shift = (g.w - g.u) - (x + half);
      sv.openShift = shift;
      placeCards(sv);
    }
    // two bet spots must never sit on top of each other (corner seat vs the
    // seat on the centre line): slide the centre one toward the pot
    for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) {
      const a = T.seats[i], b = T.seats[k];
      if (i === k || !a.central || b.central) continue;
      if (Math.abs(a.bx - b.bx) < g.u * 9.5 && Math.abs(a.by - b.by) < g.u * 3.4) a.by += (a.y < g.cy ? 1 : -1) * g.u * 3.8;
    }
    for (let i = 0; i < n; i++) {
      const sv = T.seats[i], b = T.bets[i];
      b.el.style.left = (sv.bx / g.w) * 100 + "%";
      b.el.style.top = (sv.by / g.h) * 100 + "%";
    }
    const heroV = T.seats[T.hero];
    const hz = $("hero-zone");
    if (heroV) hz.style.top = ((g.wide ? g.h - g.u * (1 + 6 * 1.38) : heroV.y - g.u * (3.5 + 6.6 * 1.38)) / g.h) * 100 + "%";
    // wide: the hero's cards sit left of centre so the action buttons fit beside them
    hz.style.left = g.wide ? ((g.cx - g.u * 5.5) / g.w) * 100 + "%" : "50%";
    placeDealer(T.lastButton);
  }

  function placeCards(sv) {
    const c = sv.cards;
    if (c.classList.contains("open")) {
      c.style.setProperty("--cx", (sv.openShift || 0) + "px");
      // (wide: no air above the top seats — tabled cards sit ON the avatar)
      c.style.setProperty("--cy", T.geom && T.geom.wide ? "calc(var(--u) * -0.4)" : "calc(var(--u) * -5.5)");
    } else {
      c.style.setProperty("--cx", "0px");
      c.style.setProperty("--cy", "calc(var(--u) * -3.1)");
    }
  }

  function placeDealer(button) {
    T.lastButton = button;
    const d = $("dealer-btn");
    const sv = button != null ? T.seats[button] : null;
    if (!sv || !T.geom || sv.dx == null) { d.hidden = true; return; }
    d.hidden = false;
    d.style.left = (sv.dx / T.geom.w) * 100 + "%";
    d.style.top = (sv.dy / T.geom.h) * 100 + "%";
  }

  // ------------------------------------------------------------------ build
  function build(s) {
    T.tableId = s.id; T.n = s.num_seats; T.hero = s.hero_seat || 0;
    T.seated = s.my_seat != null;
    const seatsEl = $("seats"), betsEl = $("bets");
    seatsEl.innerHTML = ""; betsEl.innerHTML = ""; $("fx").innerHTML = "";
    T.seats = []; T.bets = [];
    for (let i = 0; i < T.n; i++) {
      const e = el("div", "seat");
      e.dataset.seat = String(i);
      e.innerHTML =
        '<div class="seat-cards"></div>' +
        '<div class="seat-main">' +
        '<div class="seat-av"><svg class="actor-timer" viewBox="0 0 100 100"><circle class="trk" cx="50" cy="50" r="46"/><circle class="arc" cx="50" cy="50" r="46"/></svg>' +
        `<span class="av-txt"></span><span class="av-count"></span><span class="av-crown">${icon("i-crown")}</span></div>` +
        '<div class="seat-plate"><span class="seat-pos"></span><div class="seat-name"></div><div class="seat-stack num"></div></div>' +
        "</div>" +
        '<div class="seat-badge"></div><div class="seat-hand"></div><div class="seat-eq"></div>';
      const sit = el("button", "seat-sit", `${icon("i-plus")}<span>Sit</span>`);
      sit.type = "button";
      sit.addEventListener("click", () => HG.ui && HG.ui.openSit(i));
      e.querySelector(".seat-main").addEventListener("click", () => HG.ui && HG.ui.openPlayer(i));
      e.appendChild(sit);
      seatsEl.appendChild(e);
      const arc = e.querySelector(".arc");
      arc.style.strokeDasharray = String(RING);
      T.seats.push({
        el: e, cards: e.querySelector(".seat-cards"), main: e.querySelector(".seat-main"), av: e.querySelector(".seat-av"),
        avTxt: e.querySelector(".av-txt"), count: e.querySelector(".av-count"), arc,
        name: e.querySelector(".seat-name"), stack: e.querySelector(".seat-stack"), pos: e.querySelector(".seat-pos"),
        badge: e.querySelector(".seat-badge"), hand: e.querySelector(".seat-hand"), eq: e.querySelector(".seat-eq"), sit,
        stackCents: null, cardEls: [], nameKey: null, folded: false,
      });
      const b = el("div", "bet", '<span class="chips"></span><span class="amt"></span>');
      betsEl.appendChild(b);
      T.bets.push({ el: b, chips: b.firstChild, amt: b.lastChild, cents: 0 });
    }
    for (const k of ["a", "b"]) {
      const host = $("board-" + k);
      host.querySelectorAll(".slot-card").forEach((x) => x.remove());
      T.boards[k] = [];
      T.boardCards[k] = [];
      for (let j = 0; j < 5; j++) {
        const slot = el("div", "slot-card");
        host.appendChild(slot);
        T.boards[k].push(slot);
      }
    }
    $("hero-hole").innerHTML = "";
    T.heroCards = [];
    T.potCents = null; T.awardKey = null; T.foldoutKey = null; T.handNo = null;
    T.chatSeen = null; T.reactSeen = 0;
    layout();
  }

  // -------------------------------------------------------------- fx helpers
  function centerOf(node) {
    const st = $("stage").getBoundingClientRect();
    const r = node.getBoundingClientRect();
    return [r.left + r.width / 2 - st.left, r.top + r.height / 2 - st.top];
  }
  function chipColor(cents, s) {
    const bb = cents / ((s.stakes && s.stakes.bb_cents) || 100);
    if (bb < 1) return "c-white";
    if (bb < 5) return "c-red";
    if (bb < 25) return "c-blue";
    if (bb < 100) return "c-green";
    if (bb < 500) return "c-black";
    return "c-purple";
  }
  function chipStack(host, cents, s) {
    const bb = cents / ((s.stakes && s.stakes.bb_cents) || 100);
    const count = bb < 2 ? 1 : bb < 10 ? 2 : bb < 50 ? 3 : 4;
    const cls = chipColor(cents, s);
    const key = cls + count;
    if (host.dataset.k === key) return;
    host.dataset.k = key;
    host.innerHTML = "";
    for (let i = 0; i < count; i++) host.appendChild(el("i", cls));
  }
  function flyChips(from, to, opts) {
    if (!anim() || !T.geom) return;
    const o = opts || {};
    const fx = $("fx");
    const n = o.count || 3;
    for (let i = 0; i < n; i++) {
      const c = el("div", "fx-chip chip " + (o.cls || "c-red"));
      c.style.left = from[0] + "px";
      c.style.top = from[1] + "px";
      fx.appendChild(c);
      const jx = (Math.random() - 0.5) * T.geom.u * 1.6, jy = (Math.random() - 0.5) * T.geom.u * 1.6;
      const a = c.animate(
        [
          { transform: "translate(0,0) scale(0.7)", opacity: 0.2 },
          { transform: `translate(${(to[0] - from[0]) * 0.5 + jx}px, ${(to[1] - from[1]) * 0.5 + jy - T.geom.u * 2}px) scale(1.05)`, opacity: 1, offset: 0.5 },
          { transform: `translate(${to[0] - from[0]}px, ${to[1] - from[1]}px) scale(0.8)`, opacity: 0.9 },
        ],
        { duration: o.duration || 480, delay: (o.delay || 0) + i * 45, easing: "cubic-bezier(.3,.6,.25,1)", fill: "backwards" },
      );
      a.onfinish = a.oncancel = () => c.remove();
    }
  }
  function tween(node, from, to, s, ms) {
    if (!anim() || from == null || from === to) { node.textContent = fmt(to, s); return; }
    const t0 = performance.now(), dur = ms || 500;
    const token = (node._tw = (node._tw || 0) + 1);
    const step = (now) => {
      if (node._tw !== token) return;
      const k = Math.min(1, (now - t0) / dur);
      const e = 1 - Math.pow(1 - k, 3);
      node.textContent = fmt(Math.round(from + (to - from) * e), s);
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  function floatDelta(i, cents, s) {
    const sv = T.seats[i];
    if (!sv || !anim()) return;
    const d = el("div", "seat-delta " + (cents >= 0 ? "up" : "down"));
    d.textContent = (cents >= 0 ? "+" : "−") + fmt(Math.abs(cents), s);
    sv.el.appendChild(d);
    setTimeout(() => d.remove(), 2500);
  }
  function bubble(i, text, emote) {
    const sv = T.seats[i];
    if (!sv) return;
    sv.el.querySelectorAll(".seat-bubble" + (emote ? ".emote" : ":not(.emote)")).forEach((x) => x.remove());
    const b = el("div", "seat-bubble" + (emote ? " emote" : ""));
    b.textContent = text;
    sv.el.appendChild(b);
    setTimeout(() => b.remove(), emote ? 2700 : 4100);
  }

  // -------------------------------------------------------------- seat update
  function kindOf(h) {
    if (h.action === 0) return "fold";
    if (h.action === 1) return h.chips > 0 ? "call" : "check";
    return h.action === 7 ? "allin" : "raise";
  }
  function shortLabel(h, s) {
    const k = kindOf(h);
    if (k === "fold") return "Fold";
    if (k === "check") return "Check";
    if (k === "call") return "Call " + fmt(h.cents, s);
    if (k === "allin") return "All-in " + fmt(h.to_cents, s);
    return (h.first_bet ? "Bet " : "Raise ") + fmt(h.to_cents, s);
  }

  function updateSeat(i, seat, s, ctx) {
    const sv = T.seats[i];
    const e = sv.el;
    const showdown = s.phase === "showdown";
    const empty = !!seat.empty;
    e.classList.toggle("is-empty", empty);
    sv.main.hidden = empty;
    sv.sit.hidden = !empty;
    const held = empty && !!seat.reserved_by;
    e.classList.toggle("locked", empty && !held && (s.my_seat != null || s.status !== "open"));
    e.classList.toggle("reserved", held);
    if (empty) {
      const label = held ? seat.reserved_by : "Sit";
      if (sv.sit.dataset.l !== label) { sv.sit.dataset.l = label; sv.sit.lastChild.textContent = label; sv.sit.title = held ? `Reserved for ${seat.reserved_by} — waiting for the host` : ""; }
      sv.sit.disabled = held;
    }
    if (empty) {
      setSeatCards(sv, null, false, ctx);
      sv.badge.className = "seat-badge"; sv.badge.textContent = "";
      sv.hand.innerHTML = ""; sv.eq.innerHTML = "";
      e.classList.remove("is-hero", "is-actor", "is-folded", "is-away", "is-winner", "is-out", "is-host", "t-warn", "t-crit", "t-bank");
      sv.stackCents = null; sv.nameKey = null; sv.folded = false;
      return;
    }
    const key = seat.user_id + "|" + seat.name;
    if (sv.nameKey !== key) {
      sv.nameKey = key;
      sv.name.textContent = seat.name || "Player";
      sv.name.title = seat.name || "";
      sv.avTxt.textContent = initials(seat.name);
      sv.av.style.setProperty("--h", String(hueOf(seat.name)));  // by NAME: history / lobby rows carry no ids
    }
    // my private colour tag for this player (games.ui.js, localStorage)
    const tag = HG.ui && seat.user_id !== s.my_user_id ? HG.ui.noteFor(seat.user_id).tag : "none";
    if (tag && tag !== "none") { sv.av.dataset.tag = tag; sv.av.style.setProperty("--tagc", HG.ui.TAGS[tag]); }
    else delete sv.av.dataset.tag;
    const dealtIn = !!seat.in_hand && (s.phase === "in_hand" || showdown);
    const folded = dealtIn && !!seat.folded;
    e.classList.toggle("is-hero", !!seat.is_hero);
    e.classList.toggle("is-host", !!seat.is_host);
    e.classList.toggle("is-actor", !!seat.is_actor && s.phase === "in_hand");
    e.classList.toggle("is-folded", folded);
    e.classList.toggle("is-away", !!seat.sitting_out);
    e.classList.toggle("is-offline", seat.present === false);
    e.classList.toggle("is-out", !dealtIn && s.phase === "in_hand");
    if (sv.stackCents !== seat.stack_cents) {
      tween(sv.stack, ctx.animate ? sv.stackCents : null, seat.stack_cents, s, 600);
      sv.stackCents = seat.stack_cents;
    } else if (ctx.unitChanged) sv.stack.textContent = fmt(seat.stack_cents, s);
    // Bomb pots have no blinds, so only the button is a position worth a tag.
    sv.pos.textContent = dealtIn && seat.position && !/^S\d+$/i.test(seat.position) ? seat.position : "";
    sv.pos.hidden = !sv.pos.textContent;

    // status / last action (one line under the plate)
    let label = "", cls = "";
    const last = ctx.lastAction[i];
    if (seat.pending_remove) { label = "Leaving"; cls = "k-away"; }
    else if (dealtIn && seat.all_in && !folded) { label = "All-in"; cls = "k-allin"; }
    else if (folded) { label = "Fold"; cls = "k-fold"; }
    else if (last && s.phase === "in_hand") { label = shortLabel(last, s); cls = "k-" + kindOf(last); }
    else if (ctx.settled && dealtIn && ctx.deltas[i]) {
      // the hand is over: what it was worth to each player who was in it
      const d = ctx.deltas[i];
      label = (d > 0 ? "+" : "−") + fmt(Math.abs(d), s); cls = d > 0 ? "k-win" : "k-loss";
    }
    else if (seat.sitting_out) { label = "Sitting out"; cls = "k-away"; }
    else if (seat.sit_out_next) { label = "Out next hand"; cls = "k-away"; }
    if (sv.badge.textContent !== label || !sv.badge.classList.contains("show") !== !label) {
      sv.badge.textContent = label;
      sv.badge.className = "seat-badge" + (label ? " show " + cls : "");
    }

    // cards
    const hole = seat.hole;
    const isHeroCards = !!seat.is_hero && s.my_seat === i;
    if (isHeroCards) {
      setSeatCards(sv, null, false, ctx);
    } else {
      const open = !!(hole && hole.length && hole[0] >= 0);
      setSeatCards(sv, folded && !open ? null : hole, open, ctx, folded && !sv.folded);
    }
    sv.folded = folded;

    // showdown labels + equities
    let handHtml = "";
    if (!isHeroCards && seat.hand_desc && (showdown || s.runout.active)) {
      handHtml = seat.hand_desc.map((d, k) => (d ? `<span><b>${k + 1}</b>${HG.core.esc(d.charAt(0).toUpperCase() + d.slice(1))}</span>` : "")).join("");
    }
    if (sv.hand.dataset.h !== handHtml) { sv.hand.dataset.h = handHtml; sv.hand.innerHTML = handHtml; }
    let eqHtml = "";
    // equities matter while cards are still to come — not once the board is out
    if ((seat.equity_a != null || seat.equity_b != null) && s.runout.active && (s.runout.shown_len || 0) < 5) {
      const p = (x) => (x == null ? "–" : Math.round(x * 100) + "%");
      eqHtml = `<span data-b="1">${p(seat.equity_a)}</span><span data-b="2">${p(seat.equity_b)}</span>`;
    }
    if (sv.eq.dataset.h !== eqHtml) { sv.eq.dataset.h = eqHtml; sv.eq.innerHTML = eqHtml; }
  }

  function setSeatCards(sv, hole, open, ctx, mucking) {
    const host = sv.cards;
    if (!hole || !hole.length) {
      if (sv.cardEls.length) {
        const old = sv.cardEls;
        sv.cardEls = [];
        if (mucking && ctx.animate && T.geom) {
          old.forEach((c, k) => {
            c.style.setProperty("--fx", (T.geom.cx - sv.x) * 0.55 + "px");
            c.style.setProperty("--fy", (T.geom.cy - sv.y) * 0.55 + "px");
            c.style.animationDelay = k * 25 + "ms";
            c.classList.add("muck");
          });
          setTimeout(() => old.forEach((c) => c.remove()), 600);
        } else old.forEach((c) => c.remove());
      }
      host.classList.remove("open");
      return;
    }
    const wasOpen = host.classList.contains("open");
    if (wasOpen !== open) { host.classList.toggle("open", open); placeCards(sv); }
    // a NEW hand always gets freshly dealt cards (never last hand's, flipped back)
    if (ctx.dealing || sv.cardEls.length !== hole.length) {
      sv.cardEls.forEach((c) => c.remove());
      sv.cardEls = hole.map((c, k) => {
        const ce = cardEl(-1);
        ce.style.setProperty("--i", String(k));
        if (ctx.dealing && T.geom) {
          ce.style.setProperty("--fx", T.geom.cx - sv.x + "px");
          ce.style.setProperty("--fy", T.geom.cy - sv.y + "px");
          ce.style.animationDelay = (sv.rel * 5 + k) * 24 + "ms";
          ce.classList.add("deal-in");
        }
        host.appendChild(ce);
        return ce;
      });
    }
    hole.forEach((c, k) => {
      const ce = sv.cardEls[k];
      if (c >= 0 && ce.dataset.c !== String(c) && ctx.animate) {
        setTimeout(() => setCard(ce, c), 60 + k * 70);
        if (k === 0) play("flip");
      } else setCard(ce, c);
    });
  }

  function updateHero(s, ctx) {
    const hz = $("hero-zone"), host = $("hero-hole");
    const me = s.my_seat != null ? s.seats[s.my_seat] : null;
    const hole = me && me.hole && me.hole.length && me.hole[0] >= 0 && me.in_hand && s.phase !== "waiting" ? me.hole : null;
    hz.hidden = !hole;
    if (!hole) {
      if (T.heroCards.length) { T.heroCards.forEach((c) => c.remove()); T.heroCards = []; }
      return;
    }
    const key = hole.join(",");
    if (host.dataset.k !== key) {
      host.dataset.k = key;
      T.heroCards.forEach((c) => c.remove());
      T.heroCards = hole.map((c, k) => {
        const ce = cardEl(ctx.dealing ? -1 : c);
        if (ctx.dealing) {
          ce.style.setProperty("--fx", "0px");
          ce.style.setProperty("--fy", `calc(var(--u) * -22)`);
          ce.style.animationDelay = k * 70 + "ms";
          ce.classList.add("deal-in");
          setTimeout(() => setCard(ce, c), 380 + k * 80);
        }
        host.appendChild(ce);
        return ce;
      });
    }
    host.classList.toggle("folded", !!me.folded);
  }

  // ------------------------------------------------------------------ boards
  function boardList(spec) {
    const out = [];
    for (const c of [...(spec.flop || []), spec.turn, spec.river]) if (c != null) out.push(c);
    return out;
  }
  function updateBoards(s, ctx) {
    let delayBase = ctx.collected ? 380 : 0;
    let flipped = 0;
    for (const k of ["a", "b"]) {
      const cards = s.phase === "waiting" ? [] : boardList(s.board[k]);
      const cur = T.boardCards[k];
      const same = cards.length >= cur.length && cur.every((c, j) => c === cards[j]);
      if (!same) {
        T.boards[k].forEach((slot) => (slot.innerHTML = ""));
        T.boardCards[k] = [];
      }
      const have = T.boardCards[k].length;
      for (let j = have; j < cards.length; j++) {
        const ce = cardEl(ctx.animate ? -1 : cards[j]);
        T.boards[k][j].appendChild(ce);
        if (ctx.animate) {
          const d = delayBase + (j - have) * 150 + (k === "b" ? 90 : 0);
          ce.style.animationDelay = d + "ms";
          ce.classList.add("pop-in");
          setTimeout(() => { setCard(ce, cards[j]); play("flip"); }, d + 140);
          flipped++;
        }
      }
      T.boardCards[k] = cards.slice();
    }
    const tag = $("street-tag");
    // "Showdown" only when hands were actually tabled (a fold-out has no runout)
    const label = s.phase === "waiting" ? "" : s.phase === "showdown" ? (s.runout.active ? (s.runout.blocking && (s.runout.shown_len || 0) < 5 ? (s.street || "") : "Showdown") : "") : (s.street || "");
    if (tag.textContent !== label) tag.textContent = label;
    return flipped;
  }

  // ------------------------------------------------------------- pot + bets
  function chipsToCents(chips, s) {
    return Math.round((chips * s.stakes.bb_cents) / (s.stakes.bb_chips || 10000));
  }
  function updateMoney(s, prev, ctx) {
    const inHand = s.phase === "in_hand";
    const potEl = $("pot"), amt = $("pot-amt"), total = $("pot-total");
    const potCenter = () => centerOf(potEl);
    let streetSum = 0;
    const newBets = [];
    for (let i = 0; i < T.n; i++) {
      const seat = s.seats[i];
      const cents = inHand && !seat.empty ? seat.committed_this_street_cents || 0 : 0;
      newBets.push(cents);
      streetSum += cents;
    }
    // 1. bets that went away -> collected into the pot
    let collected = false;
    for (let i = 0; i < T.n; i++) {
      const b = T.bets[i];
      if (b.cents > 0 && newBets[i] < b.cents && newBets[i] === 0) {
        if (ctx.animate) flyChips([T.seats[i].bx, T.seats[i].by], potCenter(), { cls: chipColor(b.cents, s), count: 3 });
        collected = true;
      }
    }
    // 2. new / bigger bets -> chips slide out from the seat
    for (let i = 0; i < T.n; i++) {
      const b = T.bets[i], cents = newBets[i];
      if (cents > b.cents && ctx.animate) {
        flyChips([T.seats[i].x, T.seats[i].y], [T.seats[i].bx, T.seats[i].by], { cls: chipColor(cents, s), count: cents - b.cents > s.stakes.bb_cents * 10 ? 4 : 2, duration: 380 });
        b.el.classList.remove("pop");
        void b.el.offsetWidth;
        b.el.classList.add("pop");
      }
      if (cents !== b.cents || ctx.unitChanged) {
        b.cents = cents;
        b.amt.textContent = cents > 0 ? fmt(cents, s) : "";
        if (cents > 0) chipStack(b.chips, cents, s);
        b.el.classList.toggle("on", cents > 0);
      }
    }
    // 3. antes at the start of a hand
    if (ctx.dealing && ctx.animate) {
      s.seats.forEach((seat, i) => {
        if (seat.in_hand) flyChips([T.seats[i].x, T.seats[i].y], potCenter(), { cls: chipColor(s.stakes.ante_cents, s), count: 1, delay: 120 + T.seats[i].rel * 40 });
      });
      play("chips");
    }
    const settled = inHand ? chipsToCents(s.settled_pot_chips || 0, s) : s.pot_cents || 0;
    const shown = s.phase === "waiting" ? 0 : settled;
    if (T.potCents !== shown || ctx.unitChanged) {
      if (ctx.animate && T.potCents != null && shown > T.potCents) {
        const from = T.potCents;
        setTimeout(() => { tween(amt, from, shown, s, 450); potEl.classList.remove("bump"); void potEl.offsetWidth; potEl.classList.add("bump"); }, collected || ctx.dealing ? 380 : 0);
      } else amt.textContent = fmt(shown, s);
      T.potCents = shown;
      if (shown > 0) chipStack($("pot-chips"), shown, s); else { $("pot-chips").innerHTML = ""; $("pot-chips").dataset.k = ""; }
    }
    potEl.style.visibility = s.phase === "waiting" || (s.phase === "showdown" && !shown) ? "hidden" : "visible";
    const totalTxt = inHand && streetSum > 0 ? "Total " + fmt(s.pot_cents, s) : "";
    total.hidden = !totalTxt;
    if (total.textContent !== totalTxt) total.textContent = totalTxt;
    if (collected && ctx.animate) play("pot");
    return collected;
  }

  // ------------------------------------------------------------------ awards
  function updateAwards(s, prev, ctx) {
    const cap = $("award-caption");
    const step = s.runout && s.runout.award_step;
    const played = new Set(), winners = new Set();
    if (step) {
      for (const combo of Object.values(step.combos || {})) {
        (combo.hole || []).forEach((c) => played.add(c));
        (combo.board || []).forEach((c) => played.add(c));
      }
      (step.winners || []).forEach((w) => winners.add(Number(w)));
    }
    // fold-out: nobody to show down against — the pot slides to the winner
    const foldout = s.phase === "showdown" && !s.runout.active;
    const over = s.phase === "showdown" && !s.runout.blocking;
    if (foldout || (over && !step)) {
      (s.hand_deltas_cents || []).forEach((d, i) => { if (d > 0) winners.add(i); });
    }
    T.seats.forEach((sv, i) => sv.el.classList.toggle("is-winner", winners.has(i) && (!!step || foldout || over)));
    const markCards = (list) => list.forEach((ce) => {
      const c = Number(ce.dataset.c);
      ce.classList.toggle("win", !!step && played.has(c));
      ce.classList.toggle("lose", !!step && played.size > 0 && c >= 0 && !played.has(c));
    });
    for (const k of ["a", "b"]) {
      const host = $("board-" + k);
      host.classList.toggle("win", !!step && step.board === k);
      host.classList.toggle("dim", !!step && step.board !== k);
      markCards(Array.from(host.querySelectorAll(".card")));
    }
    T.seats.forEach((sv) => markCards(sv.cardEls));
    markCards(T.heroCards);

    const names = {};
    s.seats.forEach((x) => { if (!x.empty) names[x.seat] = x.is_hero && s.my_seat === x.seat ? "You" : x.name; });
    let line = "";
    if (step) {
      const amt = fmt(chipsToCents(step.chips, s), s);
      const who = (step.winners || []).map((i) => names[i] || "Seat " + (i + 1));
      const b = step.board === "b" ? "2" : "1";
      if (step.uncontested) line = `${who.join(" & ")} ${who[0] === "You" && who.length === 1 ? "take" : "takes"} ${amt} uncontested`;
      else if (who.length > 1) line = `Board ${b} · ${who.join(" & ")} chop ${amt}`;
      else {
        const c = (step.combos || {})[step.winners[0]] || (step.combos || {})[String(step.winners[0])];
        line = `Board ${b} · ${who[0]} ${who[0] === "You" ? "win" : "wins"} ${amt}${c && c.label ? " with " + c.label : ""}`;
      }
    } else if (foldout && winners.size) {
      const i = [...winners][0];
      line = `${names[i] || "Winner"} ${names[i] === "You" ? "win" : "wins"} ${fmt((s.hand_deltas_cents || [])[i] || 0, s)}`;
    }
    cap.hidden = !line;
    if (cap.textContent !== line) { cap.textContent = line; if (line) { cap.style.animation = "none"; void cap.offsetWidth; cap.style.animation = ""; } }

    // chips: pot -> winners, once per award step / fold-out
    if (!ctx.animate) { T.awardKey = step ? `${s.hand_no}:${s.runout.award_index}` : T.awardKey; if (foldout) T.foldoutKey = s.hand_no; return; }
    const pc = centerOf($("pot"));
    if (step) {
      const key = `${s.hand_no}:${s.runout.award_index}`;
      if (T.awardKey !== key) {
        T.awardKey = key;
        let mine = false;
        for (const [k, v] of Object.entries(step.shares || {})) {
          const i = Number(k), sv = T.seats[i];
          if (!sv) continue;
          const cents = chipsToCents(v, s);
          flyChips(pc, [sv.x, sv.y], { cls: chipColor(cents, s), count: 5, duration: 620 });
          floatDelta(i, cents, s);
          if (i === s.my_seat) mine = true;
        }
        play(mine ? "win" : "pot");
      }
    } else if (foldout && T.foldoutKey !== s.hand_no && prev && prev.hand_no === s.hand_no && prev.phase === "in_hand") {
      T.foldoutKey = s.hand_no;
      let mine = false;
      winners.forEach((i) => {
        const sv = T.seats[i];
        flyChips(pc, [sv.x, sv.y], { cls: "c-gold", count: 5, duration: 620 });
        floatDelta(i, (s.hand_deltas_cents || [])[i] || 0, s);
        if (i === s.my_seat) mine = true;
      });
      play(mine ? "win" : "pot");
    } else if (foldout) T.foldoutKey = s.hand_no;
  }

  // ------------------------------------------------------------------- timer
  function syncTimer(s) {
    const tm = T.timer;
    const running = s.phase === "in_hand" && s.actor != null && (s.decision_secs || 0) > 0 && s.turn_remaining_secs != null;
    if (!running) { tm.key = null; tm.seat = null; clearRings(); return; }
    const bank = !!(s.time_bank && s.time_bank.active);
    const left = bank ? s.time_bank.remaining_secs : s.turn_remaining_secs;
    const key = `${s.hand_no}:${s.action_seq}:${bank ? 1 : 0}`;
    const deadline = performance.now() + left * 1000;
    if (tm.key !== key) {
      tm.key = key; tm.seat = s.actor; tm.bank = bank; tm.deadline = deadline; tm.lastTick = null;
      tm.total = bank ? Math.max(left, (s.time_bank.secs || left)) : s.decision_secs;
      tm.mine = s.actor === s.my_seat;
      if (bank && tm.mine) play("urgent");
      clearRings();
    } else if (Math.abs(tm.deadline - deadline) > 700) tm.deadline = deadline;
    if (!T.raf) T.raf = requestAnimationFrame(tickTimer);
  }
  function clearRings() {
    T.seats.forEach((sv) => { sv.el.classList.remove("t-warn", "t-crit", "t-bank"); sv.arc.style.strokeDashoffset = "0"; sv.count.textContent = ""; });
  }
  function tickTimer() {
    T.raf = 0;
    const tm = T.timer;
    if (tm.key == null || tm.seat == null) return;
    const sv = T.seats[tm.seat];
    if (!sv) return;
    const left = Math.max(0, (tm.deadline - performance.now()) / 1000);
    const frac = Math.max(0, Math.min(1, left / (tm.total || 1)));
    sv.arc.style.strokeDashoffset = String(RING * (1 - frac));
    const crit = left <= 5.05;
    sv.el.classList.toggle("t-bank", tm.bank);
    sv.el.classList.toggle("t-crit", !tm.bank && crit);
    sv.el.classList.toggle("t-warn", !tm.bank && !crit && frac < 0.5);
    const secs = Math.ceil(left);
    if (tm.bank || crit) {
      if (sv.count.textContent !== String(secs)) sv.count.textContent = String(secs);
      if (tm.mine && secs !== tm.lastTick && secs <= 5 && secs > 0) { tm.lastTick = secs; play("tick"); }
    }
    if (HG.ui && HG.ui.onClock) HG.ui.onClock(left, tm);
    T.raf = requestAnimationFrame(tickTimer);
  }

  // ----------------------------------------------------------------- render
  function render(s, prev, opts) {
    const o = opts || {};
    if (!T.ready) {
      T.ready = true;
      if (globalThis.ResizeObserver) { T.ro = new ResizeObserver(() => layout()); T.ro.observe($("stage-box")); }
      else globalThis.addEventListener("resize", layout);
    }
    const rebuilt = T.tableId !== s.id || T.n !== s.num_seats || T.hero !== (s.hero_seat || 0) || T.seated !== (s.my_seat != null);
    if (rebuilt) build(s);
    const samePrev = !!prev && prev.id === s.id && !rebuilt;
    const animate = samePrev && anim() && !document.hidden;
    const newHand = samePrev && prev.hand_no !== s.hand_no && s.phase !== "waiting";
    const ctx = {
      animate, dealing: animate && newHand, unitChanged: !!o.unitChanged, lastAction: {}, collected: false,
      settled: s.phase === "showdown" && !s.runout.blocking, deltas: s.hand_deltas_cents || [],
    };
    if (T.handNo !== s.hand_no) { T.handNo = s.hand_no; T.awardKey = null; }

    // last action per seat on the CURRENT street (+ sounds for fresh actions)
    const hist = s.history || [];
    let seenBet = false, curStreet = null;
    hist.forEach((h) => {
      if (h.street !== curStreet) { curStreet = h.street; seenBet = false; }
      const k = kindOf(h);
      h.first_bet = (k === "raise" || k === "allin") && !seenBet;
      if (k === "raise" || k === "allin") seenBet = true;
      if (h.street === s.street || k === "fold") ctx.lastAction[h.seat] = h;
    });
    if (animate && !newHand) {
      const fresh = hist.slice(prev.history ? prev.history.length : 0);
      fresh.slice(-3).forEach((h, j) => {
        const k = kindOf(h);
        setTimeout(() => play(k === "fold" ? "fold" : k === "check" ? "check" : k === "allin" ? "allin" : "chip"), j * 120);
      });
    }
    if (ctx.dealing) { for (let j = 0; j < 5; j++) setTimeout(() => play("deal"), j * 95); }

    ctx.collected = updateMoney(s, prev, ctx);
    s.seats.forEach((seat, i) => updateSeat(i, seat, s, ctx));
    updateHero(s, ctx);
    updateBoards(s, ctx);
    placeDealer(s.phase === "waiting" && !s.hand_no ? null : s.button_seat);
    updateAwards(s, prev, ctx);
    syncTimer(s);

    // chat bubbles + reactions over the seats
    const chat = s.chat || [];
    const lastId = chat.length ? chat[chat.length - 1].id : 0;
    if (T.chatSeen != null && HG.core.G.prefs.bubbles !== false) {
      chat.filter((m) => m.id > T.chatSeen).slice(-3).forEach((m) => {
        const seat = s.seats.find((x) => !x.empty && x.name === m.name);
        if (seat) bubble(seat.seat, m.text, false);
      });
    }
    T.chatSeen = lastId;
    (s.reactions || []).forEach((r) => {
      if (r.id > T.reactSeen) { T.reactSeen = r.id; if (samePrev) { bubble(r.seat, EMOTES[r.emote] || "👍", true); } }
    });
  }

  HG.table = { render, layout, EMOTES, seatCenter: (i) => (T.seats[i] ? [T.seats[i].x, T.seats[i].y] : null) };
})();
