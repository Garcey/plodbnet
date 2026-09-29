// Debug-only harness for the home-games felt (paste into the page, then call
// __measure()). Shows a bet pill on EVERY seat and reports, per seat: where the
// bet sits relative to the seat (units), its sideways offset from "straight in
// front" (the rail normal), the nearest seat, and anything it overlaps.
window.__measure = (amount) => {
  // (a background tab: its ResizeObserver never fires and its animation clock does not run —
  // lay the felt out for the window, then finish every finite transition before measuring)
  if (globalThis.HG && HG.table) HG.table.layout();
  const stage = document.getElementById("stage");
  const u = parseFloat(getComputedStyle(stage).getPropertyValue("--u"));
  const sr = stage.getBoundingClientRect();
  const seats = [...document.querySelectorAll("#seats > *")];
  const bets = [...document.querySelectorAll("#bets > .bet")];
  bets.forEach((b) => { b.classList.add("on"); b.querySelector(".amt").textContent = amount || "$188.50"; if (!b.querySelector(".chips").children.length) b.querySelector(".chips").innerHTML = '<i class="c-red"></i><i class="c-red"></i>'; });
  const tot = document.getElementById("pot-total"); if (tot) { tot.hidden = false; const a = document.getElementById("pot-total-amt"); if (a && !a.textContent) a.textContent = "$1,288.50"; }
  document.getAnimations().forEach((a) => { try { if (a.effect && a.effect.getTiming().iterations !== Infinity) a.finish(); } catch (e) { /* (idle) */ } });
  const rect = (el) => { const r = el.getBoundingClientRect(); return { l: (r.left - sr.left) / u, t: (r.top - sr.top) / u, r: (r.right - sr.left) / u, b: (r.bottom - sr.top) / u }; };
  const ok = (r) => r && r.r > r.l && r.b > r.t;
  const union = (els) => els.filter(Boolean).map(rect).filter(ok).reduce((a, r) => a ? { l: Math.min(a.l, r.l), t: Math.min(a.t, r.t), r: Math.max(a.r, r.r), b: Math.max(a.b, r.b) } : r, null);
  const hit = (a, b, m = 0) => ok(a) && ok(b) && a.l < b.r + m && a.r > b.l - m && a.t < b.b + m && a.b > b.t - m;
  const felt = rect(document.getElementById("felt"));
  const cx = (felt.l + felt.r) / 2, cy = (felt.t + felt.b) / 2;
  const W = felt.r - felt.l + 0.8, H = felt.b - felt.t + 0.8;
  const wide = stage.classList.contains("wide");
  const normal = (px, py) => {
    if (W >= H) { const r = H / 2, L = W - 2 * r; if (Math.abs(px) <= L / 2) return [0, py > 0 ? -1 : 1]; const c = px > 0 ? L / 2 : -L / 2; const d = Math.hypot(px - c, py) || 1; return [-(px - c) / d, -py / d]; }
    const r = W / 2, L = H - 2 * r; if (Math.abs(py) <= L / 2) return [px > 0 ? -1 : 1, 0]; const c = py > 0 ? L / 2 : -L / 2; const d = Math.hypot(px, py - c) || 1; return [-px / d, -(py - c) / d];
  };
  const named = { pot: rect(document.getElementById("pot")), total: tot ? rect(tot) : null, boards: rect(document.getElementById("boards")), tag: rect(document.getElementById("street-tag")), heroCards: union([document.getElementById("hero-hole")]), dealer: rect(document.getElementById("dealer-btn")) };
  // (a phone on its side: the dock floats over the felt's bottom corners and covers what is under it)
  const shown = (e) => (e && e.offsetWidth && e.offsetHeight ? rect(e) : null);
  named.dockL = wide ? shown(document.getElementById("dock-left")) : null;
  named.dockR = wide ? shown(document.getElementById("actbar")) : null;
  // (the hero's cards are part of the hero's box where they sit over the plate; on a phone on its
  // side they sit BESIDE it, and the union of the two would swallow the felt between them)
  const boxes = seats.map((e) => union([e.querySelector(".seat-main"), e.querySelector(".seat-av"), e.querySelector(".seat-badge"), e.querySelector(".seat-cards"), e.classList.contains("is-hero") && !wide ? document.getElementById("hero-hole") : null]));
  const pos = seats.map((e) => [parseFloat(e.style.left) / 100 * (sr.width / u), parseFloat(e.style.top) / 100 * (sr.height / u)]);
  let worst = 0, bad = 0;
  const out = seats.map((e, i) => {
    const [x, y] = pos[i];
    const br = rect(bets[i]);
    const bx = (br.l + br.r) / 2, by = (br.t + br.b) / 2;
    let [nx, ny] = normal(x - cx, y - cy);
    if (wide && Math.abs(x - cx + 25.2) < 0.5 && y > cy) { nx = 0; ny = -1; }
    const vx = bx - x, vy = by - y;
    const along = vx * nx + vy * ny, side = vx * -ny + vy * nx;
    worst = Math.max(worst, Math.abs(side));
    const hits = [];
    for (const k in named) if (hit(br, named[k], 0.05)) hits.push(k);
    boxes.forEach((bb, k) => { if (hit(br, bb, 0.05)) hits.push("seat" + k); });
    bets.forEach((o, k) => { if (k !== i && hit(br, rect(o), 0.1)) hits.push("bet" + k); });
    if (br.l < 0 || br.t < 0 || br.r > sr.width / u || br.b > sr.height / u) hits.push("offstage");
    // "whose bet is it?" = the seat whose BOX is nearest the pill (not the anchor)
    let near = -1, nd = 1e9;
    const gapTo = (bb) => { if (!ok(bb)) return 1e9; const gx = Math.max(0, bb.l - br.r, br.l - bb.r), gy = Math.max(0, bb.t - br.b, br.t - bb.b); return Math.hypot(gx, gy); };
    boxes.forEach((bb, k) => { let d = gapTo(bb); if (!ok(bb)) { const [ox, oy] = pos[k]; d = gapTo({ l: ox - 5.8, t: oy - 5, r: ox + 5.8, b: oy + 7 }); } if (d < nd - 0.01) { nd = d; near = k; } });
    if (near !== i || hits.length) bad++;
    return `seat${i} @(${(x - cx).toFixed(1)},${(y - cy).toFixed(1)}) bet +(${vx.toFixed(1)},${vy.toFixed(1)}) ahead=${along.toFixed(1)} side=${side.toFixed(1)}${near !== i ? " NEAREST=seat" + near + " !!" : ""} ${hits.length ? "HITS " + hits.join(",") : "clear"}`;
  });
  const dl = named.dealer, dhits = [];
  // (its OWN seat is allowed: with nowhere free it tucks half onto its own plate, under it — HGT-005;
  // on a phone on its side the dock floats over the felt's bottom corners: dockL / dockR)
  // (whose button: window.__btn when a script renders states of its own, else the table's)
  let btn = globalThis.__btn != null ? globalThis.__btn : globalThis.HG && HG.core.G.state ? HG.core.G.state.button_seat : null, bd = 1e9;
  if (btn == null && ok(dl)) boxes.forEach((bb, k) => { if (!ok(bb)) return; const d = Math.hypot(Math.max(0, bb.l - dl.r, dl.l - bb.r), Math.max(0, bb.t - dl.b, dl.t - bb.b)); if (d < bd) { bd = d; btn = k; } });
  if (ok(dl)) { boxes.forEach((bb, k) => { if (k !== btn && hit(dl, bb, 0)) dhits.push("seat" + k); }); bets.forEach((o, k) => { if (hit(dl, rect(o), 0)) dhits.push("bet" + k); }); for (const k of ["pot", "total", "boards", "heroCards", "dockL", "dockR"]) if (hit(dl, named[k], 0)) dhits.push(k); }  // (the pot row's "Total" too: HGT-005)
  const f = (r) => ok(r) ? `${r.t.toFixed(1)}..${r.b.toFixed(1)}` : "-";
  return `${innerWidth}x${innerHeight} u=${u.toFixed(2)} stage=${(sr.width / u).toFixed(0)}x${(sr.height / u).toFixed(1)}u [${stage.className}] pot y ${f(named.pot)} boards y ${f(named.boards)} heroCards y ${f(named.heroCards)} | worst side=${worst.toFixed(1)}u, problems=${bad}, dealer ${dhits.length ? "HITS " + dhits.join(",") : "clear"}\n` + out.join("\n");
};
