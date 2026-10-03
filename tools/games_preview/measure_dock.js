// Does the TABLE keep its size whatever the dock shows? (2026-10-02 — owner: "when it's your
// turn vs not your turn, the addition and removal of the betting options slightly resizes the
// whole table"). Paste into a Study / Trainer page (/?mode=trainer, /?mode=study) or a seated
// player's home-games table, then:
//
//   __dockWatch()                 start logging every size change of #stage-box (the felt's box)
//   __autoTrainer(ms)             Trainer: check / call every decision, deal the next hand
//   __autoStudy(ms)               Study: deal random cards, check / call for every seat, add the
//                                 turn and river when Study asks for them
//   __autoHome(ms)                home games: check / call your turns, "I'm back" if the clock
//                                 sat you out, deal when it's yours to deal (run
//                                 `bot.py loop <table>` beside it for the other seats)
//   __dockReport()                every size the box had, with what the dock showed then;
//                                 `statesSeen` = how many different dock states went by
//
// The drivers run in the background (start one for minutes, then re-run __dockWatch() after
// each window resize). ONE size per window size is the goal. Sizes used on 2026-10-02:
// 1920x1080, 1304x923, 1100x800, 1366x650 (a short window: the sizing panel floats),
// 820x1180, 390x844, 375x667 and 844x390 (the dock floats over the felt).
(function () {
  const W = window;
  const vis = (el) => !!el && !el.hidden && el.getClientRects().length > 0 && getComputedStyle(el).visibility !== "hidden";
  const h = (el) => Math.round(el.getBoundingClientRect().height);
  // what the dock shows: the visible parts of the action bar (and its slot), the side
  // clusters, the hand labels, Study's work bar / guide
  W.__dockState = function () {
    const bar = document.getElementById("actbar");
    const slot = document.getElementById("act-slot");
    const parts = bar ? [...bar.children, ...(slot ? slot.children : [])]
      .filter((el) => el !== slot).filter(vis).map((el) => `${el.id || el.className}:${h(el)}`) : [];
    const side = ["act-slot", "dock-left", "dock-right", "hero-hand-labels", "workbar", "study-guide"]
      .map((id) => document.getElementById(id)).filter(vis).map((el) => `${el.id}:${h(el)}`);
    const dock = document.getElementById("dock");
    return `${parts.join(" ")} | ${side.join(" ")} | dock:${dock ? h(dock) : "-"}`;
  };
  W.__seen = new Set();
  W.__note = () => W.__seen.add(W.__dockState());
  // Polled, not a ResizeObserver: a hidden tab (the Browser pane in the background) runs no
  // rendering steps, so an observer never fires there — a forced layout every 100 ms does.
  W.__dockWatch = function () {
    W.__dockLog = [];
    W.__seen = new Set();
    if (W.__dockTimer) clearInterval(W.__dockTimer);
    const box = document.getElementById("stage-box");
    let last = "";
    const look = () => {
      const r = box.getBoundingClientRect();
      const k = `${Math.round(r.width)}x${Math.round(r.height)}`;
      if (k === last) return;
      last = k;
      W.__dockLog.push({ t: Math.round(performance.now()), w: Math.round(r.width), h: Math.round(r.height), dock: W.__dockState() });
    };
    look();
    W.__dockTimer = setInterval(look, 100);
    return `watching #stage-box at ${innerWidth}x${innerHeight}`;
  };
  W.__dockReport = function () {
    const sizes = {};
    for (const x of W.__dockLog || []) {
      const k = `${x.w}x${x.h}`;
      (sizes[k] = sizes[k] || new Set()).add(x.dock);
    }
    const out = {};
    for (const [k, v] of Object.entries(sizes)) out[k] = [...v].slice(0, 6);
    return { viewport: `${innerWidth}x${innerHeight}`, changes: (W.__dockLog || []).length, sizes: out, statesSeen: W.__seen.size };
  };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const clickable = (el) => !!el && !el.disabled && vis(el);
  // (the page's UI / postJSON / applyState are top-level declarations of its classic
  // scripts: global bindings, not properties of window)
  /* eslint-disable no-undef */
  W.__autoTrainer = async function (ms) {
    const end = performance.now() + ms;
    let acts = 0, hands = 0;
    while (performance.now() < end) {
      W.__note();
      const s = UI.lastState;
      const next = document.querySelector("[data-trainer-next]");
      if (s && s.terminal && clickable(next)) { await sleep(300); W.__note(); next.click(); hands++; await sleep(1500); continue; }
      const b = document.getElementById("check-btn");
      if (s && !s.terminal && s.actor === s.hero_seat && clickable(b)) { await sleep(400); W.__note(); b.click(); acts++; await sleep(900); continue; }
      await sleep(200);
    }
    return (W.__autoRes = { acts, hands });
  };
  W.__autoStudy = async function (ms) {
    const end = performance.now() + ms;
    let acts = 0, hands = 0, d = null;
    const shuffle = () => {
      const deck = [...Array(52).keys()];
      for (let i = deck.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [deck[i], deck[j]] = [deck[j], deck[i]]; }
      return deck;
    };
    const post = async (url, body) => { const r = await postJSON(url, body); if (r && r.state) applyState(r.state); return r; };
    const newHand = async () => {
      d = shuffle();
      await post("/reset", {});
      await sleep(600); W.__note();
      await post("/cards", { hero_hole: d.slice(0, 5), flop_a: d.slice(5, 8), flop_b: d.slice(8, 11), turn: [null, null], river: [null, null] });
    };
    await newHand();
    while (performance.now() < end) {
      W.__note();
      const s = UI.lastState;
      // (on Hero's turn Study says "Place the turn cards to act"; otherwise "Betting is closed")
      const need = s && (s.awaiting_next_street || (["turn", "river"].includes(s.hero_blocking_reason) ? s.hero_blocking_reason : null));
      if (s && s.terminal) { await sleep(700); W.__note(); hands++; await newHand(); await sleep(500); continue; }
      if (need === "turn") { await sleep(600); W.__note(); await post("/cards", { turn: [d[11], d[12]] }); await sleep(400); continue; }
      if (need === "river") { await sleep(600); W.__note(); await post("/cards", { river: [d[13], d[14]] }); await sleep(400); continue; }
      const b = document.getElementById("check-btn");
      if (s && s.actor != null && clickable(b) && !document.getElementById("act-btns").hidden) { await sleep(300); W.__note(); b.click(); acts++; await sleep(700); continue; }
      await sleep(250);
    }
    return (W.__autoRes = { acts, hands });
  };
  /* eslint-enable no-undef */
  W.__autoHome = async function (ms) {
    const end = performance.now() + ms;
    let acts = 0, deals = 0;
    while (performance.now() < end) {
      W.__note();
      const b = document.getElementById("check-btn");
      if (clickable(b) && !document.getElementById("act-btns").hidden) { await sleep(400); W.__note(); b.click(); acts++; await sleep(900); continue; }
      const strip = [...document.querySelectorAll("#status-strip button")];
      const back = strip.find((x) => /I'm back/.test(x.textContent));
      if (clickable(back)) { back.click(); await sleep(1200); continue; }
      const deal = strip.find((x) => /^Deal next/.test(x.textContent));
      if (clickable(deal)) { deal.click(); deals++; await sleep(1200); continue; }
      await sleep(200);
    }
    return (W.__autoRes = { acts, deals });
  };
})();
