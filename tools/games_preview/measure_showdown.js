// Debug-only harness for the home-games felt at a SHOWDOWN / all-in runout
// (2026-09-25; paste into a seated player's page on the preview server). It
// records, for every runout street and award step it sees, every pair of felt
// elements that overlap (plates, avatars, badges, labels, tabled cards, board
// cards and tags, pot pills, bets, caption, street tag, hero cards) — the check
// used to tune fitSeats / fitPots in games.table.js at every screen size.
//
//   __showdownWatch()          start logging into window.__log
//   __log.filter(x => x.n)     snapshots that had overlaps (x.o = the pairs)
//   __freezeNextShowdown()     stop the table re-rendering at the next showdown
//                              (for a screenshot); window.__freeze = false resumes
//
// The dealer disc is left out on purpose: it sits under the seats and fades
// while a label or tabled row covers it.
(() => {
  const R = (e) => { const r = e.getBoundingClientRect(); return [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)]; };
  const vis = (e) => {
    if (!e) return false;
    const cs = getComputedStyle(e);
    if (cs.display === "none" || cs.visibility === "hidden" || +cs.opacity === 0) return false;
    const r = e.getBoundingClientRect();
    return r.width > 1 && r.height > 1;
  };
  // A background tab's animation clock does not run (the tab is never painted), so a row or
  // pill in the middle of its transition would be measured where it STARTED: finish every
  // finite animation / transition first — the settled felt is what gets measured.
  const settle = () => document.getAnimations().forEach((a) => { try { if (a.effect && a.effect.getTiming().iterations !== Infinity) a.finish(); } catch (e) { /* (idle) */ } });
  window.__measure = () => {
    settle();
    const items = [];
    document.querySelectorAll(".seat").forEach((s) => {
      const i = s.dataset.seat;
      const add = (sel, kind) => s.querySelectorAll(sel).forEach((e) => { if (vis(e)) items.push({ who: "s" + i, kind, r: R(e) }); });
      add(".seat-plate", "plate"); add(".seat-av", "av"); add(".seat-badge.show", "badge"); add(".seat-hand span", "label"); add(".seat-cards .card", "card");
    });
    document.querySelectorAll("#board-a .slot-card, #board-b .slot-card").forEach((e, k) => items.push({ who: "board", kind: "b" + k, r: R(e) }));
    document.querySelectorAll(".board-tag").forEach((e, k) => vis(e) && items.push({ who: "board", kind: "tag" + k, r: R(e) }));
    document.querySelectorAll("#pots .potc, #live-pots .potc, #pot, #pot-total").forEach((e, k) => vis(e) && items.push({ who: "pot", kind: "pot" + k, r: R(e) }));
    document.querySelectorAll(".bet").forEach((e, k) => vis(e) && e.offsetWidth && items.push({ who: "bet" + k, kind: "bet", r: R(e) }));
    ["award-caption", "hero-hole", "street-tag", "rabbit-btn", "hero-tag", "burns"].forEach((id) => { const e = document.getElementById(id); if (vis(e)) items.push({ who: id, kind: id, r: R(e) }); });
    const ov = (a, b) => Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const out = [];
    for (let i = 0; i < items.length; i++) for (let j = i + 1; j < items.length; j++) {
      const a = items[i], b = items[j];
      if (a.who === b.who) continue;
      const o = ov(a.r, b.r);
      if (o > 12) out.push(`${a.who}.${a.kind}${JSON.stringify(a.r)} x ${b.who}.${b.kind}${JSON.stringify(b.r)} = ${o}`);
    }
    return out;
  };
  window.__showdownWatch = () => {
    window.__log = [];
    if (window.__watch) clearInterval(window.__watch);
    window.__watch = setInterval(() => {
      const s = HG.core.G.state;
      if (!s || !(s.phase === "showdown" || (s.runout && s.runout.active))) return;
      const key = (x) => `${innerWidth}x${innerHeight}:${x.hand_no}:${x.runout.shown_len || 0}:${x.runout.award_index || 0}`;
      const k = key(s);
      if (window.__lastK === k) return;
      window.__lastK = k;
      setTimeout(() => {
        // (a background tab is throttled: the table may have moved on — and its ResizeObserver
        // never fires, so the layout is brought up to the window's size first)
        if (!HG.core.G.state || key(HG.core.G.state) !== k) return;
        HG.table.layout();
        const o = window.__measure();
        window.__log.push({ k, n: o.length, o, hidden: document.querySelectorAll(".seat-hand.crowded").length });
      }, 450);  // (after the JS-driven parts of the step: pot amounts, chips)
    }, 350);
  };
  window.__freezeNextShowdown = () => {
    if (!window.__origRender) {
      window.__origRender = HG.table.render;
      HG.table.render = function (...a) { if (window.__freeze) return; return window.__origRender.apply(this, a); };
    }
    window.__freeze = false;
    const t = setInterval(() => {
      const s = HG.core.G.state;
      if (s && s.phase === "showdown" && document.querySelectorAll(".seat-cards.open .card:not(.is-down)").length >= 10) {
        clearInterval(t);
        setTimeout(() => { window.__freeze = true; }, 900);
      }
    }, 100);
  };
})();
