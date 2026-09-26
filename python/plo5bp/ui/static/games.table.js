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

  // Inward normal of the rail at a perimeter point (relative to the table centre):
  // "straight in front of the player". Flat edges point straight across, the
  // round ends point at their own circle's centre.
  function railNormal(px, py, W, H) {
    if (W >= H) {
      const r = H / 2, L = W - 2 * r;
      if (Math.abs(px) <= L / 2) return [0, py > 0 ? -1 : 1];
      const c = px > 0 ? L / 2 : -L / 2, d = Math.hypot(px - c, py) || 1;
      return [-(px - c) / d, -py / d];
    }
    const r = W / 2, L = H - 2 * r;
    if (Math.abs(py) <= L / 2) return [px > 0 ? -1 : 1, 0];
    const c = py > 0 ? L / 2 : -L / 2, d = Math.hypot(px, py - c) || 1;
    return [-px / d, -(py - c) / d];
  }
  const rOver = (a, b, m) => Math.max(0, Math.min(a[2], b[2] + m) - Math.max(a[0], b[0] - m)) * Math.max(0, Math.min(a[3], b[3] + m) - Math.max(a[1], b[1] - m));
  // The award caption's height at this scale (font floors), measured on the real node.
  function captionHeight(g) {
    const cap = $("award-caption");
    const was = { hidden: cap.hidden, text: cap.textContent, vis: cap.style.visibility };
    cap.style.visibility = "hidden"; cap.hidden = false; cap.textContent = "Side pot 1 · Board 1 · Somebody wins $100.00 with a full house, As full of Qs";
    const h = cap.getBoundingClientRect().height || g.u * 3;
    cap.hidden = was.hidden; cap.textContent = was.text; cap.style.visibility = was.vis;
    return h;
  }
  // A bet pill's real size at this scale (font floors make it relatively bigger
  // on a small phone), measured once per layout on a throwaway twin.
  function pillSize(g) {
    const probe = el("div", "bet on", '<span class="chips"><i class="c-red"></i><i class="c-red"></i></span><span class="amt">$188.50</span>');
    probe.style.visibility = "hidden";
    $("bets").appendChild(probe);
    const r = probe.getBoundingClientRect();
    probe.remove();
    return { hw: Math.max(r.width, g.u * 8) / 2, hh: Math.max(r.height, g.u * 2.2) / 2 };
  }
  // how far a seat reaches below its anchor: the plate, plus the action badge
  // ("Bet $6") that hangs under it exactly when a bet is out
  // (the name + stack lines stop shrinking at their font floors, hence the second term)
  const seatBottom = (u) => Math.max(u * 7.04, u * 3.81 + 25.8) - u * 0.75 + Math.max(9.5, u * 1.08) * 1.45 + u * 0.36 + 2;
  const OWN_GAP = 0.25, OBS_GAP = 0.3; // units of air: bet <-> its own seat, bet <-> anything else

  // Where each seat's bet (and the dealer button) goes. A bet belongs in FRONT
  // of its player: on the rail's normal, just clear of the seat's own box. When
  // something is in the way — the boards on a narrow phone, the hero's cards,
  // a neighbour's bet — the spot swings round the seat a few degrees at a time
  // and takes the first free place, so it always stays nearest its own seat.
  // (It used to aim at the table CENTRE, which on a wide table put a corner
  // seat's chips — and the hero's — in front of the player next door.)
  function placeBetSpots(g, pill) {
    const u = g.u, n = T.n, sr = $("stage").getBoundingClientRect();
    const rectOf = (node) => { const r = node.getBoundingClientRect(); return [r.left - sr.left, r.top - sr.top, r.right - sr.left, r.bottom - sr.top]; };
    const bottom = seatBottom(u);
    // what a bet must never cover
    const fixed = [];
    // the pot pill — or the pots in its place mid-hand (a hidden element's box is empty)
    const potNode = ["live-pots", "pot", "pots"].map($).find((e) => e && !e.hidden && e.offsetWidth) || $("pot");
    const pot = rectOf(potNode), boards = rectOf($("boards"));
    const potHalf = Math.max((pot[2] - pot[0]) / 2, u * 8);
    fixed.push([g.cx - potHalf, pot[1], g.cx + potHalf, pot[3]]);
    fixed.push([g.cx - potHalf - u * 8.4, pot[1], g.cx - potHalf, pot[3]]); // the street total, left of the pot
    if (!g.portrait && !g.wide) fixed.push([g.cx + potHalf, pot[1], g.cx + potHalf + u * 9, pot[3]]); // the street tag, right of it
    fixed.push(boards);
    if (g.portrait) fixed.push([g.cx - u * 5, boards[3] + u * 0.5, g.cx + u * 5, boards[3] + u * 2.6]); // street tag
    const heroV = T.seated ? T.seats[T.hero] : null;
    let heroCards = null;
    if (heroV) {
      const cw = u * g.heroCw, half = (cw * 4.04) / 2, hx = g.wide ? g.cx - u * 5.5 : g.cx;
      const top = g.wide ? g.h - u * (1 + g.heroCw * 1.38) : heroV.y - u * (3.5 + g.heroCw * 1.38);
      heroCards = [hx - half, top, hx + half, top + cw * 1.38];
      fixed.push(heroCards);
    }
    const boxes = T.seats.map((sv) => [sv.x - u * 5.9, sv.y - u * (sv === heroV ? 3 : 5.1), sv.x + u * 5.9, sv.y + bottom]);
    const inStage = (r) => r[0] >= u * 0.4 && r[1] >= u * 0.4 && r[2] <= g.w - u * 0.4 && r[3] <= g.h - u * 0.4;
    // distance along (dx, dy) at which a (hw x hh) box is just clear of `own`
    const reach = (ox, oy, own, dx, dy, half, gap) => Math.min(
      Math.abs(dx) > 1e-6 ? ((dx > 0 ? own[2] - ox : ox - own[0]) + half.hw + gap) / Math.abs(dx) : Infinity,
      Math.abs(dy) > 1e-6 ? ((dy > 0 ? own[3] - oy : oy - own[1]) + half.hh + gap) / Math.abs(dy) : Infinity);
    const placed = [];
    const order = [];
    for (let i = 0; i < n; i++) order.push(i);
    // the hero first (the big hero cards leave the fewest options), then the seat
    // across the table (it shares the centre line with the pot), then the rest
    const rank = (i) => { const rel = T.seats[i].rel; return rel === 0 ? 0 : n % 2 === 0 && rel === n / 2 ? 1 : 2 + rel; };
    order.sort((a, b) => rank(a) - rank(b));
    for (const i of order) {
      const sv = T.seats[i];
      // The hero's bet goes out from the hero's CARDS. Upright and desktop they sit
      // over the hero's plate (one box); on a phone on its side they sit beside it.
      const bigHero = sv === heroV;
      const own = !bigHero ? boxes[i] : g.wide ? heroCards
        : [Math.min(boxes[i][0], heroCards[0]), heroCards[1], Math.max(boxes[i][2], heroCards[2]), boxes[i][3]];
      const ox = bigHero ? (heroCards[0] + heroCards[2]) / 2 : sv.x, oy = bigHero && g.wide ? (heroCards[1] + heroCards[3]) / 2 : sv.y;
      const others = fixed.filter((r) => !(bigHero && r === heroCards)).concat(boxes.filter((_, k) => k !== i || (bigHero && g.wide)), placed);
      const base = Math.atan2(sv.normal[1], sv.normal[0]);
      const lean = Math.sin(Math.atan2(g.cy - sv.y, g.cx - sv.x) - base);
      const turn = Math.abs(lean) < 0.02 ? (sv.normal[0] > 0 ? -1 : 1) : lean > 0 ? 1 : -1; // toward the table centre first (dead ahead: up-screen)
      const cands = [];
      for (let a = 0; a <= 96; a += 8) for (const sgn of a ? [turn, -turn] : [1]) for (const extra of [0, 1.6, 3.2]) {
        const th = base + (sgn * a * Math.PI) / 180, dx = Math.cos(th), dy = Math.sin(th);
        const r = reach(ox, oy, own, dx, dy, pill, u * OWN_GAP) + extra * u;
        cands.push({ cost: a + extra * 5 + (sgn === turn ? 0 : 0.5), x: ox + dx * r, y: oy + dy * r });
      }
      cands.sort((p, q) => p.cost - q.cost);
      let best = cands[0], bestOver = Infinity;
      for (const c of cands) {
        const r = [c.x - pill.hw, c.y - pill.hh, c.x + pill.hw, c.y + pill.hh];
        let over = inStage(r) ? 0 : 1e6;
        // (dead ahead wins whenever it fits at all; only a swing insists on the full air gap)
        for (const o of others) over += rOver(r, o, u * (c.cost === 0 ? 0.06 : OBS_GAP));
        if (over < bestOver) { bestOver = over; best = c; }
        if (over === 0) break;
      }
      sv.bx = best.x; sv.by = best.y;
      placed.push([best.x - pill.hw, best.y - pill.hh, best.x + pill.hw, best.y + pill.hh]);
    }
    // dealer button: beside the seat, on whichever side of its bet is free
    const disc = { hw: u * 1.35, hh: u * 1.35 };
    for (let i = 0; i < n; i++) {
      const sv = T.seats[i];
      const others = fixed.concat(boxes.filter((_, k) => k !== i), placed);
      // The button clears the plate's REAL width: on a small phone the name / stack
      // stop shrinking at their font floors, and the plate grows up to its max-width
      // (7.5u a side) — the 5.9u model put the disc over the stack's last digits.
      const plate = sv.el.querySelector(".seat-plate");
      const halfW = Math.max(u * 7.6, plate && plate.offsetWidth ? plate.offsetWidth / 2 + u * 0.3 : 0);
      const own = [sv.x - halfW, boxes[i][1], sv.x + halfW, boxes[i][3]];
      // (the hero's bet leaves from the hero's CARDS — the button stays by the hero's plate)
      const ang = sv === heroV ? Math.atan2(sv.normal[1], sv.normal[0]) : Math.atan2(sv.by - sv.y, sv.bx - sv.x);
      let pick = [sv.x, sv.y], pickOver = Infinity;
      for (const off of sv === heroV || (n % 2 === 0 && sv.rel === n / 2) ? [-90, 90, -66, 66, -112, 112] : [48, -48, 66, -66, 90, -90, 112, -112]) {
        const th = ang + (off * Math.PI) / 180, dx = Math.cos(th), dy = Math.sin(th);
        const r = reach(sv.x, sv.y, own, dx, dy, disc, u * 0.2), x = sv.x + dx * r, y = sv.y + dy * r;
        const rc = [x - disc.hw, y - disc.hh, x + disc.hw, y + disc.hh];
        let over = inStage(rc) ? 0 : 1e6;
        for (const o of others) over += rOver(rc, o, u * 0.2);
        if (over < pickOver) { pickOver = over; pick = [x, y]; }
        if (over === 0) break;
      }
      sv.dx = pick[0]; sv.dy = pick[1];
    }
  }

  function computeGeom() {
    const box = $("stage-box").getBoundingClientRect();
    const bw = Math.max(240, box.width), bh = Math.max(200, box.height);
    // (a phone keeps the upright table up to a squarish box — a short screen,
    // e.g. an iPhone SE with Safari's bars: the flat one there drew every card
    // about a third smaller)
    const portrait = bw / bh < 0.92 || (bw < 520 && bw / bh < 1.3);
    // WIDE = a phone on its side. Height is the scarce thing, so the table
    // goes flat and wide, the two boards sit SIDE BY SIDE, the hero's plate
    // moves beside the hero's cards, and nobody is seated along the bottom
    // edge (the action buttons overlay that corner — see games.css).
    const wide = !portrait && !!(globalThis.matchMedia && matchMedia("(max-height: 480px) and (orientation: landscape)").matches);
    // The table takes the shape of the space it is given (within reason): a
    // squarer window gets a rounder table instead of letterbox bars.
    // (upright: never wider than 0.74 per unit of height — a squarer table lifts
    // the seats on its long sides onto the boards)
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
    // hero card width in units — keep in step with #hero-hole in games.css
    const heroCw = wide ? 6 : portrait ? 6.6 : 6.3;
    return { w, h, u, portrait, wide, heroCw, fx, fy, fw, fh, cx: fx + fw / 2, cy: fy + fh / 2 };
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
    const n = T.n;
    const pad = g.u * 0.4;
    const W = g.fw + 2 * pad, H = g.fh + 2 * pad;
    // The pot + boards block keeps a bet's height of felt clear on both sides
    // of it: under the seat across the table, and (seated) over the hero's
    // cards — so both of those bets sit dead ahead of their player.
    const pill = pillSize(g);
    let centerY = g.fy + g.fh * (g.portrait ? 0.43 : g.wide ? 0.47 : 0.465);
    if (n) {
      const ch = $("center").offsetHeight || g.u * 21.5;
      const betRoom = 2 * pill.hh + g.u * (OWN_GAP + OBS_GAP);
      // (a phone on its side seats every opponent along the top, whatever the count)
      const lo = (n % 2 === 0 || g.wide ? g.cy - H / 2 + seatBottom(g.u) + betRoom : g.fy + g.u * 2) + ch / 2;
      const cardsTop = g.wide ? g.h - g.u * (1 + g.heroCw * 1.38) : g.cy + H / 2 - g.u * (3.5 + g.heroCw * 1.38);
      // under the boards: the hero's bet, or (at showdown) the award caption — whichever is taller
      const underRoom = Math.max(betRoom, captionHeight(g) + g.u * (g.portrait ? 3.4 : 1.4));
      const hi = (T.seated ? cardsTop - underRoom : g.fy + g.fh - g.u * 2) - ch / 2;
      centerY = lo <= hi ? Math.max(lo, Math.min(hi, centerY)) : (lo + hi) / 2;
    }
    $("center").style.top = (centerY / g.h) * 100 + "%";
    $("banner").style.top = $("center").style.top;
    if (!n) return;
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
      const sv = T.seats[i];
      // the side seats sit ON the rail's ends: on a narrow phone half their plate
      // used to hang off the screen — keep the (measured) plate on the stage
      const plate = sv.el.querySelector(".seat-plate");
      const ph = Math.max(g.u * 3, plate && plate.offsetWidth ? plate.offsetWidth / 2 : 0);
      const x = Math.max(ph + 2, Math.min(g.w - ph - 2, g.cx + px)), y = g.cy + py;
      sv.x = x; sv.y = y; sv.rel = rel;
      sv.el.style.left = (x / g.w) * 100 + "%";
      sv.el.style.top = (y / g.h) * 100 + "%";
      sv.normal = g.wide && rel === 0 ? [0, -1] : railNormal(px, py, W, H);
    }
    placeBetSpots(g, pill);
    for (let i = 0; i < n; i++) {
      const sv = T.seats[i], b = T.bets[i];
      b.el.style.left = (sv.bx / g.w) * 100 + "%";
      b.el.style.top = (sv.by / g.h) * 100 + "%";
    }
    const heroV = T.seats[T.hero];
    const hz = $("hero-zone");
    if (heroV) hz.style.top = ((g.wide ? g.h - g.u * (1 + g.heroCw * 1.38) : heroV.y - g.u * (3.5 + g.heroCw * 1.38)) / g.h) * 100 + "%";
    // wide: the hero's cards sit left of centre so the action buttons fit beside them
    hz.style.left = g.wide ? ((g.cx - g.u * 5.5) / g.w) * 100 + "%" : "50%";
    placeDealer(T.lastButton);
    fitPots($("pots"));
    fitPots($("live-pots"));
    fitSeats();
  }

  // Tabled (face-up) cards sit over the avatar and a seat's showdown labels hang
  // under it. On a phone the seats on the table's long sides are level with the
  // boards: a five-card row reached over the end card of a board, and a label
  // over a board's corner. So a row that would touch the boards / pot block fans
  // its cards tighter (down to a little under half a card each: the rank and
  // suit in the corner still show) and moves to the stage edge, and a label
  // slides outward — neither ever leaves the stage. Runs after every layout and
  // render (the pot row changes width at a showdown); every read comes first.
  function fitSeats() {
    const g = T.geom;
    if (!g || !T.seats.length) return;
    const st = $("stage").getBoundingClientRect();
    const blocks = [];
    for (const id of ["boards", "pots", "pot-row", "street-tag", "award-caption"]) {
      const e = $(id);
      if (!e || e.hidden || !e.offsetWidth) continue;
      const r = e.getBoundingClientRect();
      blocks.push([r.left - st.left, r.top - st.top, r.right - st.left, r.bottom - st.top]);
    }
    const box = (e) => { const r = e.getBoundingClientRect(); return [r.left - st.left, r.top - st.top, r.right - st.left, r.bottom - st.top]; };
    const labels = T.seats.map((sv) => {
      const w = sv.x == null ? 0 : sv.hand.offsetWidth;
      if (!w) return null;
      const r = sv.hand.getBoundingClientRect();
      return { w, top: r.top - st.top, bot: r.bottom - st.top };
    });
    const mains = T.seats.map((sv) => (sv.x == null || !sv.main.offsetWidth ? null : box(sv.main)));
    const hz = $("hero-zone");
    const heroBox = hz && !hz.hidden && hz.offsetWidth ? box(hz) : null;
    const cw = g.u * (g.wide ? 3 : 3.5), gap = g.u * 0.4, edge = g.u * 0.6;
    const labelBox = [], rowBox = [];
    T.seats.forEach((sv, i) => {
      if (sv.x == null) return;
      const left = sv.x < g.w / 2;
      // the label: on the stage, then off the block
      const lb = labels[i];
      let hx = 0;
      if (lb) {
        const m = 4, l = sv.x - lb.w / 2, r = sv.x + lb.w / 2;
        hx = l < m ? m - l : r > g.w - m ? g.w - m - r : 0;
        const hit = blocks.filter((b) => b[1] < lb.bot && b[3] > lb.top && b[0] < r + hx + gap && b[2] > l + hx - gap);
        if (hit.length && left) {
          const bl = Math.min(...hit.map((b) => b[0])) - gap;
          if (sv.x < bl) hx = Math.max(m - l, Math.min(hx, bl - r));
        } else if (hit.length) {
          const br = Math.max(...hit.map((b) => b[2])) + gap;
          if (sv.x > br) hx = Math.min(g.w - m - r, Math.max(hx, br - l));
        }
        labelBox[i] = [l + hx, lb.top, r + hx, lb.bot];
      }
      hx = Math.round(hx);
      if (sv.labelShift !== hx) { sv.labelShift = hx; sv.hand.style.setProperty("--hx", hx + "px"); }
      // the tabled cards
      const n = sv.cardEls.length || 5;
      const width = (ov) => (n - (n - 1) * ov) * cw;  // (keep in step with .seat-cards.open in games.css)
      const cy = sv.y + g.u * (g.wide ? -0.4 : -5.5);
      const top = cy - cw * 0.87, bot = cy + cw * 0.69;  // (a winning card lifts 0.18 of a card)
      let ov = 0.3, hw = width(ov) / 2;
      let c = Math.max(g.u + hw, Math.min(g.w - g.u - hw, sv.x));
      const hits = blocks.filter((b) => b[1] < bot && b[3] > top && b[0] < c + hw + gap && b[2] > c - hw - gap);
      const inner = !hits.length ? null : left ? Math.min(...hits.map((b) => b[0])) - gap : Math.max(...hits.map((b) => b[2])) + gap;
      if (inner != null && n > 1 && (left ? sv.x < inner : sv.x > inner)) {  // (beside the block, not over it)
        const room = left ? inner - edge : g.w - edge - inner;
        ov = Math.min(0.56, Math.max(0.3, (n - room / cw) / (n - 1)));
        hw = width(ov) / 2;
        c = left ? Math.max(edge + hw, Math.min(inner - hw, sv.x)) : Math.min(g.w - edge - hw, Math.max(inner + hw, sv.x));
      }
      const shift = Math.round((c - sv.x) * 10) / 10, ovr = Math.round(ov * 1000) / 1000;
      if (sv.openShift !== shift || sv.openOv !== ovr) { sv.openShift = shift; sv.openOv = ovr; placeCards(sv); }
      if (sv.cards.classList.contains("open") && sv.cardEls.length) rowBox[i] = [c - hw, top, c + hw, bot];
    });
    // On a short phone a label can still land on the next seat down (its
    // avatar or tabled cards), the hero's cards or a board: then it steps back
    // (hidden) — the cards and the award caption still say what everyone had.
    // (a board or the pots only when it is more than a corner: the label was already slid clear of them as far as the stage allows)
    const meets = (a, b, mx) => !!b && Math.min(a[2], b[2]) - Math.max(a[0], b[0]) > mx && Math.min(a[3], b[3]) - Math.max(a[1], b[1]) > 3;
    T.seats.forEach((sv, i) => {
      const lb = labelBox[i];
      const crowded = !!lb && (T.seats.some((_, j) => j !== i && (meets(lb, mains[j], 3) || meets(lb, rowBox[j], 3))) || meets(lb, heroBox, 3) || blocks.some((b) => meets(lb, b, 8)));
      if (sv.crowded !== crowded) { sv.crowded = crowded; sv.hand.classList.toggle("crowded", crowded); }
    });
    // The dealer disc sits beside its seat, which is where a showdown's labels
    // and tabled cards can land: it steps aside (fades) while they cover it.
    const taken = [...labelBox.filter((b, i) => b && !T.seats[i].crowded), ...rowBox.filter(Boolean)];
    const d = $("dealer-btn"), bs = T.lastButton != null ? T.seats[T.lastButton] : null;
    let covered = false;
    if (bs && bs.dx != null && !d.hidden) {
      const rad = Math.max(g.u * 1.25, 8.5);
      covered = taken.some((t) => t[0] < bs.dx + rad && t[2] > bs.dx - rad && t[1] < bs.dy + rad && t[3] > bs.dy - rad);
    }
    d.classList.toggle("covered", covered);
  }

  function placeCards(sv) {
    const c = sv.cards;
    if (c.classList.contains("open")) {
      c.style.setProperty("--cx", (sv.openShift || 0) + "px");
      c.style.setProperty("--ov", String(sv.openOv || 0.3));
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
        '<div class="seat-badge"></div><div class="seat-hand"></div>';
      const sit = el("button", "seat-sit", `${icon("i-plus")}<span>Sit</span>`);
      sit.type = "button";
      sit.hidden = true;  // the render shows it on empty seats (else a bare button flashes on load)
      sit.addEventListener("click", () => {
        const st = HG.core.G.state;
        const held = st && st.seats[i] && st.seats[i].reserved_by;
        if (held && st.is_host && HG.ui && HG.ui.openRequest) HG.ui.openRequest(i);
        else if (HG.ui) HG.ui.openSit(i);
      });
      e.querySelector(".seat-main").addEventListener("click", () => HG.ui && HG.ui.openPlayer(i));
      e.appendChild(sit);
      seatsEl.appendChild(e);
      const arc = e.querySelector(".arc");
      arc.style.strokeDasharray = String(RING);
      T.seats.push({
        el: e, cards: e.querySelector(".seat-cards"), main: e.querySelector(".seat-main"), av: e.querySelector(".seat-av"),
        avTxt: e.querySelector(".av-txt"), count: e.querySelector(".av-count"), arc,
        name: e.querySelector(".seat-name"), stack: e.querySelector(".seat-stack"), pos: e.querySelector(".seat-pos"),
        badge: e.querySelector(".seat-badge"), hand: e.querySelector(".seat-hand"), sit,
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
    e.classList.toggle("host-review", held && !!s.is_host);
    e.classList.toggle("has-request", !empty && !!seat.request);
    if (empty) {
      const label = held ? seat.reserved_by : "Sit";
      if (sv.sit.dataset.l !== label) { sv.sit.dataset.l = label; sv.sit.lastChild.textContent = label; sv.sit.title = held ? (s.is_host ? `${seat.reserved_by} asks to buy in — tap to review` : `Reserved for ${seat.reserved_by} — waiting for the host`) : ""; }
      sv.sit.disabled = held && !s.is_host;  // (the host taps it to approve)
    }
    if (empty) {
      setSeatCards(sv, null, false, ctx);
      sv.badge.className = "seat-badge"; sv.badge.textContent = ""; sv.badge.dataset.k = "";
      sv.hand.innerHTML = ""; sv.hand.dataset.h = "";
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
    // All-in runout: each player's chance to win each board takes the badge
    // line while cards are still to come (over the seat, where it used to hang,
    // the tabled cards covered it).
    let eqHtml = "";
    if (dealtIn && !folded && (seat.equity_a != null || seat.equity_b != null) && s.runout.active && (s.runout.shown_len || 0) < 5) {
      const p = (x) => (x == null ? "–" : Math.round(x * 100) + "%");
      eqHtml = `<span data-b="1">${p(seat.equity_a)}</span><span data-b="2">${p(seat.equity_b)}</span>`;
    }
    const bkey = eqHtml ? "eq|" + eqHtml : label ? cls + "|" + label : "";
    if (sv.badge.dataset.k !== bkey) {
      sv.badge.dataset.k = bkey;
      if (eqHtml) sv.badge.innerHTML = eqHtml; else sv.badge.textContent = label;
      sv.badge.className = "seat-badge" + (eqHtml ? " show k-eq" : label ? " show " + cls : "");
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

    // showdown labels (the short form: "Js full of 4s" — the caption, the dock
    // and the hand history keep the full wording)
    let handHtml = "";
    if (!isHeroCards && seat.hand_desc && (showdown || s.runout.active)) {
      handHtml = seat.hand_desc.map((d, k) => (d ? `<span><b>${k + 1}</b>${HG.core.esc(shortHand(d))}</span>` : "")).join("");
    }
    if (sv.hand.dataset.h !== handHtml) { sv.hand.dataset.h = handHtml; sv.hand.innerHTML = handHtml; }  // (placed by fitSeats)
  }

  // A made hand as the felt labels it: short enough to sit beside the boards on
  // a phone ("a full house, Js full of 4s" -> "Js full of 4s"). Anything this
  // does not recognise is shown as it came.
  const SHORT_HANDS = [
    [/^a pair of (\S+)$/i, "Pair of $1"],
    [/^two pair, (\S+) and (\S+)$/i, "Two pair $1 & $2"],
    [/^three of a kind, (\S+)$/i, "Three $1"],
    [/^a straight flush, (\S+)$/i, "Straight flush $1"],
    [/^a straight (\S+)$/i, "Straight $1"],
    [/^a flush (\S+) high$/i, "$1-high flush"],
    [/^a full house, (.+)$/i, "$1"],
    [/^four of a kind, (\S+)$/i, "Four $1"],
  ];
  function shortHand(d) {
    const t = String(d || "");
    for (const [re, out] of SHORT_HANDS) if (re.test(t)) return t.replace(re, out);
    return t.charAt(0).toUpperCase() + t.slice(1);
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
    const reshaped = renderLivePots(s);
    const potCenter = () => centerOf(potAnchor());
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
    // (beside the pot, not under it: a line of its own used to appear with the
    // first bet and push both boards down)
    const totalTxt = inHand && streetSum > 0 ? fmt(s.pot_cents, s) : "";
    total.hidden = !totalTxt;
    const totalAmt = $("pot-total-amt");
    if (totalAmt.textContent !== totalTxt) totalAmt.textContent = totalTxt;
    fitPots($("live-pots"));
    // (a wider or narrower pot row: the bet spots are placed around it again)
    if (reshaped) layout();
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
    const pots = renderPots(s, step);
    let line = "";
    if (step) {
      const amt = fmt(chipsToCents(step.chips, s), s);
      const who = (step.winners || []).map((i) => names[i] || "Seat " + (i + 1));
      const b = step.board === "b" ? "2" : "1";
      const potName = pots.length > 1 && s.pots[step.pot] ? s.pots[step.pot].label + " · " : "";
      if (step.uncontested) line = `${potName}${who.join(" & ")} ${who[0] === "You" && who.length === 1 ? "take" : "takes"} ${amt} uncontested`;
      else if (who.length > 1) line = `${potName}Board ${b} · ${who.join(" & ")} chop ${amt}`;
      else {
        const c = (step.combos || {})[step.winners[0]] || (step.combos || {})[String(step.winners[0])];
        line = `${potName}Board ${b} · ${who[0]} ${who[0] === "You" ? "win" : "wins"} ${amt}${c && c.label ? " with " + c.label : ""}`;
      }
    } else if (foldout && winners.size) {
      const i = [...winners][0];
      line = `${names[i] || "Winner"} ${names[i] === "You" ? "win" : "wins"} ${fmt((s.hand_deltas_cents || [])[i] || 0, s)}`;
    }
    cap.hidden = !line;
    if (cap.textContent !== line) { cap.textContent = line; if (line) { cap.style.animation = "none"; void cap.offsetWidth; cap.style.animation = ""; } }

    // chips: pot -> winners, once per award step / fold-out
    if (!ctx.animate) { T.awardKey = step ? `${s.hand_no}:${s.runout.award_index}` : T.awardKey; if (foldout) T.foldoutKey = s.hand_no; return; }
    // the chips leave the pot they belong to (side pots first, main pot last)
    const potNode = step && pots.length > 1 && pots[step.pot] ? pots[step.pot] : potAnchor();
    const pc = centerOf(potNode);
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

  // The pots of a showdown, side by side: "Side pot 2 · Side pot 1 · Main pot",
  // each shrinking as its halves are paid (ClubGG-style). One pot = the plain
  // pot pill does the job. Returns the pot nodes by index (empty when hidden).
  function renderPots(s, step) {
    const host = $("pots");
    const pots = (s.runout && s.runout.active && s.runout.blocking && s.pots) || [];
    if (pots.length < 2) { host.hidden = true; host.innerHTML = ""; host.dataset.k = ""; $("pot-row").classList.remove("replaced"); return []; }
    const paid = {};
    (s.pot_awards || []).forEach((a) => { paid[a.pot] = (paid[a.pot] || 0) + (a.chips || 0); });
    const cur = step ? step.pot : -1;
    const key = pots.map((p, k) => `${p.label}:${p.chips - (paid[k] || 0)}:${k === cur ? 1 : 0}`).join("|") + "|" + (s.hand_no || 0);
    if (host.dataset.k !== key) {
      host.dataset.k = key;
      host.innerHTML = "";
      pots.forEach((p, k) => {
        const left = Math.max(0, p.chips - (paid[k] || 0));
        const d = potPill(p, chipsToCents(left, s), s, "final:" + k);
        if (k === cur) d.classList.add("on");
        if (left <= 0) d.classList.add("paid");
        host.appendChild(d);
      });
    }
    host.hidden = false;
    $("pot-row").classList.add("replaced");  // the split pots ARE the pot
    fitPots(host);  // (before the chips fly from them)
    return Array.from(host.children);
  }

  // One pot as a pill: its name ("Side pot 1", on a phone "Side 1"), a chip
  // stack and what is in it. The title names the players who can win it;
  // hovering it lights them up.
  function potPill(p, cents, s, id) {
    const d = el("div", "potc");
    d.dataset.pot = id;
    const short = p.label === "Main pot" ? "Main" : String(p.label).replace(/^Side pot /, "Side ");
    d.innerHTML = `<small class="lab-l">${HG.core.esc(p.label)}</small><small class="lab-s">${HG.core.esc(short)}</small>` +
      `<span class="chips"></span><b class="num">${fmt(cents, s)}</b>`;
    chipStack(d.querySelector(".chips"), Math.max(cents, 1), s);
    const names = (p.eligible || []).map((i) => s.seats[i] && (s.seats[i].is_hero && s.my_seat === i ? "You" : s.seats[i].name)).filter(Boolean);
    d.title = `${p.label}: ${names.join(", ")}`;
    return d;
  }

  // The pots WHILE the hand is played (server `live_pots`, 2026-09-25 — side
  // pots used to show only at the showdown): two or more take the pot pill's
  // place in its row, with the same names and order as the showdown's, so the
  // runout takes over without a pot moving. A pot that grew bumps like the pot.
  function renderLivePots(s) {
    const host = $("live-pots"), pot = $("pot");
    const pots = (s.phase === "in_hand" && s.live_pots) || [];
    if (pots.length < 2) {
      if (!host.hidden || pot.classList.contains("split")) {
        host.hidden = true; host.innerHTML = ""; host.dataset.k = ""; pot.classList.remove("split");
        T.livePots = null;
        return true;  // (the pot row changed shape)
      }
      return false;
    }
    const key = pots.map((p) => `${p.label}:${p.chips}:${p.eligible.join(",")}`).join("|") + "|" + (s.hand_no || 0);
    if (host.dataset.k === key) return false;
    const was = T.livePots || {};
    host.dataset.k = key;
    host.innerHTML = "";
    T.livePots = {};
    pots.forEach((p, k) => {
      const d = potPill(p, chipsToCents(p.chips, s), s, "live:" + k);
      if (was[p.label] != null && p.chips > was[p.label] && anim()) d.classList.add("bump");
      T.livePots[p.label] = p.chips;
      host.appendChild(d);
    });
    host.hidden = false;
    pot.classList.add("split");
    return true;
  }
  // (bets fly into — and a fold-out's chips out of — whichever holds the pot)
  const potAnchor = () => ["live-pots", "pot"].map($).find((e) => e && !e.hidden && e.offsetWidth) || $("pot");

  // Hover a pot — tap it on a phone — and its players light up while everyone
  // else dims: a side pot shows who can still win it, the main pot everyone
  // still in the hand. The focus survives the pills being rebuilt (by index).
  function potEligible(node) {
    const s = HG.core.G.state;
    if (!s || !node || !node.offsetWidth) return null;
    if (node.id === "pot") return s.seats.filter((x) => !x.empty && x.in_hand && !x.folded).map((x) => x.seat);
    const [kind, k] = String(node.dataset.pot || "").split(":");
    const list = kind === "live" ? s.live_pots : kind === "final" ? s.pots : null;
    const p = list && list[Number(k)];
    return p ? p.eligible : null;
  }
  function focusPot(node) {
    T.potFocus = node ? (node.id === "pot" ? "pot" : node.dataset.pot) : null;
    applyPotFocus();
  }
  function applyPotFocus() {
    const f = T.potFocus;
    const node = !f ? null : f === "pot" ? $("pot") : document.querySelector(`.potc[data-pot="${f}"]`);
    const elig = potEligible(node);
    if (!elig) T.potFocus = null;
    const set = new Set(elig || []);
    $("stage").classList.toggle("pot-focus", !!elig);
    T.seats.forEach((sv, i) => sv.el.classList.toggle("pot-in", !!elig && set.has(i)));
    document.querySelectorAll("#pot.hi, .potc.hi").forEach((x) => { if (x !== node || !elig) x.classList.remove("hi"); });
    if (node && elig) node.classList.add("hi");
  }
  function wirePotFocus() {
    const center = $("center");
    const potOf = (e) => (e.target && e.target.closest ? e.target.closest("#pot, .potc") : null);
    center.addEventListener("pointerover", (e) => { if (e.pointerType === "touch") return; const n = potOf(e); if (n) focusPot(n); });
    center.addEventListener("pointerout", (e) => {
      if (e.pointerType === "touch") return;
      const n = potOf(e);
      if (n && !(e.relatedTarget && n.contains(e.relatedTarget))) focusPot(null);
    });
    // a phone has no hover: a tap shows the pot's players for a few seconds (tap again to hide)
    center.addEventListener("pointerup", (e) => {
      if (e.pointerType !== "touch") return;
      const n = potOf(e);
      if (!n) return;
      clearTimeout(T.potFocusTimer);
      const id = n.id === "pot" ? "pot" : n.dataset.pot;
      if (T.potFocus === id) { focusPot(null); return; }
      focusPot(n);
      T.potFocusTimer = setTimeout(() => focusPot(null), 3500);
    });
  }

  // Four or more pots are wider than the gap between the seats beside them on
  // a phone (the outer pills went under those seats' plates): drop the chip
  // icons, then shrink the row until it fits. (The pots of a live hand share
  // the pot's row with the street total: they need its room on BOTH sides.)
  function fitPots(host) {
    if (!host || host.hidden || !T.geom) return;
    const total = $("pot-total");
    const flank = host.id === "live-pots" && !total.hidden ? total.offsetWidth + T.geom.u * 1.1 : 0;
    const key = `${host.dataset.k}|${T.geom.w}x${T.geom.h}|${Math.round(flank)}`;
    if (host.dataset.fit === key) return;
    host.dataset.fit = key;
    host.classList.remove("tight");
    host.style.transform = "";
    const st = $("stage").getBoundingClientRect(), pr = host.getBoundingClientRect();
    const mid = (pr.left + pr.right) / 2, gap = T.geom.u * 0.5;
    let room = 2 * Math.min(mid - st.left, st.right - mid) - 2 * gap;
    for (const sv of T.seats) {
      for (const node of [sv.el.querySelector(".seat-plate"), sv.badge]) {
        if (!node || !node.offsetWidth) continue;
        const r = node.getBoundingClientRect();
        if (r.bottom <= pr.top || r.top >= pr.bottom) continue;
        if (r.right <= mid) room = Math.min(room, 2 * (mid - r.right - gap));
        else if (r.left >= mid) room = Math.min(room, 2 * (r.left - mid - gap));
      }
      // a showdown label slides out as far as the stage edge (fitSeats), no further
      const lw = sv.x == null ? 0 : sv.hand.offsetWidth;
      if (lw) {
        const r = sv.hand.getBoundingClientRect();
        if (r.bottom <= pr.top || r.top >= pr.bottom) continue;
        if (st.left + sv.x < mid) room = Math.min(room, 2 * (mid - (st.left + 4 + lw) - gap));
        else room = Math.min(room, 2 * (st.right - 4 - lw - mid - gap));
      }
    }
    room -= 2 * flank;
    if (host.offsetWidth <= room) return;
    host.classList.add("tight");
    const w = host.offsetWidth;
    if (w > room) host.style.transform = `scale(${Math.max(0.7, room / w).toFixed(3)})`;
  }

  // The rabbit hunt: a small button in the gap the turn and river would fill.
  function placeRabbit(s) {
    const btn = $("rabbit-btn");
    const me = Number.isInteger(s.my_seat);
    const show = !!(s.can_rabbit && me && s.phase === "showdown");
    btn.hidden = !show;
    if (!show) return;
    const bx = $("boards").getBoundingClientRect();
    const empties = { a: [], b: [] };
    for (const k of ["a", "b"]) T.boards[k].forEach((slot, j) => { if (!(T.boardCards[k] || [])[j] && j >= (T.boardCards[k] || []).length) empties[k].push(slot.getBoundingClientRect()); });
    const box = (rs) => rs.length ? { l: Math.min(...rs.map((r) => r.left)), t: Math.min(...rs.map((r) => r.top)), r: Math.max(...rs.map((r) => r.right)), b: Math.max(...rs.map((r) => r.bottom)) } : null;
    const A = box(empties.a), B = box(empties.b);
    let target = A || B;
    if (A && B && !(A.r < B.l || B.r < A.l)) target = { l: Math.min(A.l, B.l), t: Math.min(A.t, B.t), r: Math.max(A.r, B.r), b: Math.max(A.b, B.b) }; // stacked boards: centre the 2 x 2 gap
    if (!target) { btn.hidden = true; return; }
    btn.style.left = ((target.l + target.r) / 2 - bx.left) + "px";
    btn.style.top = ((target.t + target.b) / 2 - bx.top) + "px";
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
      wirePotFocus();
      if (globalThis.ResizeObserver) { T.ro = new ResizeObserver(() => { layout(); if (HG.core.G.state) placeRabbit(HG.core.G.state); }); T.ro.observe($("stage-box")); }
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
    placeRabbit(s);
    fitSeats();
    applyPotFocus();  // (fresh players for the pot under the pointer; rebuilt pills keep the focus)
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

  HG.table = { render, layout, EMOTES, shortHand, seatCenter: (i) => (T.seats[i] ? [T.seats[i].x, T.seats[i].y] : null) };
})();
