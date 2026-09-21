/*
 * Node harness for tools/pokernow/pokernow.user.js (review 2026-09-20).
 *
 * Runs the REAL userscript inside a `vm` sandbox with a fake DOM, fake timers,
 * a fake MutationObserver and a fake GM_xmlhttpRequest, then drives the
 * delivery scenarios the review flagged:
 *
 *   - onerror / ontimeout dropped the in-flight frame and never drained again
 *   - the "nothing changed" early return skipped drain()
 *   - the heartbeat re-sent a remembered payload after the observer's root
 *     had been replaced (stale state presented as live)
 *   - all-in was inferred from a missing stack number
 *
 * Usage:  node pokernow_userscript_harness.js <path-to-userscript>
 * Exit 0 + "OK <n> scenarios" on success; throws (exit 1) on the first failure.
 * Driven by tests/ocr/test_pokernow_userscript.py (skipped when node is absent).
 */
'use strict';

const assert = require('assert');
const fs = require('fs');
const vm = require('vm');

const SRC = fs.readFileSync(process.argv[2], 'utf8');
const HEARTBEAT_MS = 2000;
const RETRY_MS = 1000;

// ---- fake DOM ---------------------------------------------------------------

function textEl(text) {
  return { textContent: text };
}

function cardEl(card) {
  // card: "Ts" | null (face-down)
  return {
    querySelector(sel) {
      if (!card) return null;
      if (sel === '.value') return textEl(card[0] === 'T' ? '10' : card[0]);
      if (sel === '.suit:not(.sub-suit)') return textEl(card[1]);
      return null;
    },
  };
}

// seat model: {seat, name, hero, actor, folded, stack (number|null), stackLabel
// (string|undefined), noStackEl (bool), bet (number|string|null), cards, extraClass}
function seatEl(m) {
  const classes = ['table-player', `table-player-${m.seat}`];
  if (m.hero) classes.push('you-player');
  if (m.actor) classes.push('decision-current');
  if (m.folded) classes.push('fold');
  if (m.extraClass) classes.push(m.extraClass);
  const stackText = m.stack != null ? String(m.stack) : m.stackLabel || '';
  const stackEl = m.noStackEl
    ? null
    : {
        textContent: stackText,
        querySelector: (sel) =>
          sel === '.normal-value' && m.stack != null ? textEl(String(m.stack)) : null,
      };
  const betText = m.bet == null ? '' : String(m.bet);
  return {
    className: classes.join(' '),
    classList: { contains: (c) => classes.includes(c) },
    textContent: `${m.name || ''} ${stackText} ${betText}`,
    querySelector(sel) {
      if (sel === '.table-player-stack') return stackEl;
      if (sel === '.table-player-stack .normal-value') return stackEl && stackEl.querySelector('.normal-value');
      if (sel === '.table-player-bet-value') return m.bet == null ? null : textEl(betText);
      if (sel === '.table-player-name') return textEl(m.name || '');
      return null;
    },
    querySelectorAll: (sel) => (sel === '.card' ? (m.cards || []).map(cardEl) : []),
    getBoundingClientRect: () => ({ left: m.x || 0, top: m.y || 0, width: 10, height: 10 }),
  };
}

function makeWorld() {
  const world = {
    now: 0,
    timers: [],       // {id, at, fn, every}
    nextTimer: 1,
    requests: [],     // GM_xmlhttpRequest option objects, in call order
    observers: [],
    tableRoot: { id: 'table#1', isConnected: true, getBoundingClientRect: () => ({ left: 0, top: 0, width: 1000, height: 600 }) },
    seats: [],
    boards: [
      { run: '1', cards: ['4c', 'Jc', 'Ad'] },
      { run: '2', cards: ['9h', '2s', 'Kd'] },
    ],
    pot: 12,
    badge: { style: {}, textContent: '' },
  };

  const body = {
    isConnected: true,
    appendChild() {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 1000, height: 600 }),
  };

  const document = {
    body,
    createElement: () => world.badge,
    querySelector(sel) {
      if (sel === '.table') return world.tableRoot;
      if (sel === '.dealer-button-ctn') return { className: 'dealer-button-ctn dealer-position-6' };
      if (sel === '.five-cards') return {};
      if (sel === '.table-pot-size') return textEl(String(world.pot));
      return null;
    },
    querySelectorAll(sel) {
      if (sel === '.table-player') return world.seats.map(seatEl);
      if (sel === '.table-cards') {
        return world.boards.map((b) => ({
          className: `table-cards run-${b.run}`,
          querySelectorAll: (s) => (s === '.card' ? b.cards.map(cardEl) : []),
        }));
      }
      return [];
    },
  };

  class MutationObserver {
    constructor(cb) {
      this.cb = cb;
      this.root = null;
      world.observers.push(this);
    }
    observe(root) { this.root = root; }
    disconnect() { this.root = null; }
  }

  const sandbox = {
    document,
    location: { pathname: '/games/pgl-harness' },
    MutationObserver,
    GM_xmlhttpRequest: (opts) => { world.requests.push(opts); },
    setInterval: (fn, ms) => { world.timers.push({ id: world.nextTimer, at: world.now + ms, fn, every: ms }); return world.nextTimer++; },
    setTimeout: (fn, ms) => { world.timers.push({ id: world.nextTimer, at: world.now + ms, fn, every: 0 }); return world.nextTimer++; },
    clearTimeout: (id) => { world.timers = world.timers.filter((t) => t.id !== id); },
    Date: { now: () => world.now },
    console,
  };

  // Fire a DOM mutation the way the browser would: only an observer still
  // bound to the LIVE root hears it.
  world.mutate = () => {
    for (const o of world.observers) if (o.root && o.root === world.tableRoot) o.cb([]);
  };
  // Move the clock WITHOUT running timers (isolates code paths from the retry pump).
  world.setNow = (t) => { world.now = t; };
  // Run every timer due up to `now + ms`, in time order.
  world.advance = (ms) => {
    const until = world.now + ms;
    for (;;) {
      const due = world.timers.filter((t) => t.at <= until).sort((a, b) => a.at - b.at)[0];
      if (!due) break;
      world.now = due.at;
      if (due.every) due.at += due.every;
      else world.timers = world.timers.filter((t) => t.id !== due.id);
      due.fn();
    }
    world.now = until;
  };
  world.body = (i) => JSON.parse(world.requests[i].data);
  world.start = () => vm.runInNewContext(SRC, sandbox, { filename: 'pokernow.user.js' });
  return world;
}

const HERO = ['Ts', 'As', '4h', '3d', '2d'];
function headsUp(world, { heroStack = 86, villStack = 86, heroBet = null, villBet = null } = {}) {
  world.seats = [
    { seat: 1, name: 'Miles', hero: true, actor: true, stack: heroStack, bet: heroBet, cards: HERO, x: 500, y: 550 },
    { seat: 6, name: 'JJ', stack: villStack, bet: villBet, cards: [null, null, null, null, null], x: 500, y: 20 },
  ];
}

// ---- scenarios --------------------------------------------------------------

const scenarios = [];
const scenario = (name, fn) => scenarios.push([name, fn]);

scenario('onerror requeues the in-flight frame and retries it', () => {
  const w = makeWorld();
  headsUp(w);
  w.start();
  assert.strictEqual(w.requests.length, 1, 'initial frame posted');
  w.requests[0].onerror();
  assert.strictEqual(w.requests.length, 1, 'no tight retry loop while backing off');
  w.advance(RETRY_MS);
  assert.strictEqual(w.requests.length, 2, 'frame re-sent after the backoff');
  assert.deepStrictEqual(w.body(1), w.body(0), 'the SAME frame, not a dropped one');
  w.requests[1].onload({ status: 200 });
  assert.match(w.badge.textContent, /frames sent: 1/);
});

scenario('ontimeout keeps order: failed frame first, then the frames queued behind it', () => {
  const w = makeWorld();
  headsUp(w);
  w.start();                                   // F1 in flight
  headsUp(w, { heroStack: 76, heroBet: 10 });  // F2
  w.mutate();
  headsUp(w, { heroStack: 76, heroBet: 10, villStack: 76, villBet: 10 }); // F3
  w.mutate();
  assert.strictEqual(w.requests.length, 1, 'one request at a time');
  w.requests[0].ontimeout();
  w.advance(RETRY_MS);
  w.requests[1].onload({ status: 200 });
  w.requests[2].onload({ status: 200 });
  w.requests[3].onload({ status: 200 });
  const bets = [1, 2, 3].map((i) => w.body(i).seats.map((s) => s.betDollars));
  assert.deepStrictEqual(bets, [[null, null], [10, null], [10, 10]], 'F1, F2, F3 in order');
  assert.strictEqual(w.requests.length, 4);
});

scenario('a no-change mutation still drains a stranded queue', () => {
  const w = makeWorld();
  headsUp(w);
  w.start();
  w.requests[0].onerror();          // F1 requeued, backing off
  w.setNow(RETRY_MS + 1);           // backoff over — but do NOT run the retry timer
  w.mutate();                       // DOM unchanged => same fingerprint
  assert.strictEqual(w.requests.length, 2, 'the early "nothing changed" exit must still drain');
});

scenario('an HTTP error drops that frame and keeps draining', () => {
  const w = makeWorld();
  headsUp(w);
  w.start();
  headsUp(w, { heroStack: 76, heroBet: 10 });
  w.mutate();
  w.requests[0].onload({ status: 500 });
  assert.strictEqual(w.requests.length, 2, 'next frame goes out');
  assert.deepStrictEqual(w.body(1).seats.map((s) => s.betDollars), [10, null]);
  w.requests[1].onload({ status: 200 });
  w.advance(HEARTBEAT_MS - 1);      // past RETRY_MS, short of the first heartbeat
  assert.strictEqual(w.requests.length, 2, 'the rejected frame is not retried');
});

scenario('heartbeat never resends a remembered payload when the table is gone', () => {
  const w = makeWorld();
  headsUp(w);
  w.start();
  w.requests[0].onload({ status: 200 });
  w.seats = [];                     // left the table: nothing on screen
  w.advance(HEARTBEAT_MS * 3);
  assert.strictEqual(w.requests.length, 1, 'no table => no heartbeat (old script re-sent F1 forever)');
});

scenario('heartbeat re-attaches a detached observer and reports FRESH state', () => {
  const w = makeWorld();
  headsUp(w);
  w.start();
  w.requests[0].onload({ status: 200 });
  const oldRoot = w.tableRoot;
  // SPA re-render: `.table` replaced; the observer is still bound to the old node.
  oldRoot.isConnected = false;
  w.tableRoot = { id: 'table#2', isConnected: true, getBoundingClientRect: oldRoot.getBoundingClientRect };
  headsUp(w, { heroStack: 76, heroBet: 10 });
  w.mutate();                       // nobody hears it: observer root is stale
  assert.strictEqual(w.requests.length, 1);
  w.advance(HEARTBEAT_MS);
  assert.strictEqual(w.observers[0].root, w.tableRoot, 'observer re-bound to the live root');
  assert.strictEqual(w.requests.length, 2);
  assert.deepStrictEqual(w.body(1).seats.map((s) => s.betDollars), [10, null], 'fresh state, not stale F1');
  w.requests[1].onload({ status: 200 });
  assert.match(w.badge.textContent, /frames sent: 2/, 'missed state counts as a real frame');
  // ... and mutations are heard again.
  headsUp(w, { heroStack: 76, heroBet: 10, villStack: 76, villBet: 10 });
  w.mutate();
  assert.strictEqual(w.requests.length, 3);
});

scenario('idle heartbeat sends a fresh (identical) snapshot and is not counted as a frame', () => {
  const w = makeWorld();
  headsUp(w);
  w.start();
  w.requests[0].onload({ status: 200 });
  w.advance(HEARTBEAT_MS);
  assert.strictEqual(w.requests.length, 2);
  assert.deepStrictEqual(w.body(1), w.body(0));
  w.requests[1].onload({ status: 200 });
  assert.match(w.badge.textContent, /frames sent: 1/);
});

scenario('a failed heartbeat is not requeued', () => {
  const w = makeWorld();
  headsUp(w);
  w.start();
  w.requests[0].onload({ status: 200 });
  w.advance(HEARTBEAT_MS);
  w.requests[1].onerror();
  w.advance(RETRY_MS);
  assert.strictEqual(w.requests.length, 2, 'nothing to retry: a heartbeat carries no new state');
});

scenario('all-in only when the DOM says so', () => {
  const w = makeWorld();
  w.seats = [
    { seat: 1, name: 'Miles', hero: true, stack: 56, bet: 30, cards: HERO, x: 500, y: 550 },
    { seat: 2, name: 'Shover', stack: null, stackLabel: 'All In', bet: 20, cards: [null, null, null, null, null], x: 0, y: 300 },
    { seat: 3, name: 'MidRender', stack: null, noStackEl: true, cards: [null, null, null, null, null], x: 250, y: 20 },
    { seat: 4, name: 'All In Annie', stack: null, noStackEl: true, cards: [null, null, null, null, null], x: 500, y: 20 },
    { seat: 5, name: 'Classy', stack: null, extraClass: 'all-in', cards: [null, null, null, null, null], x: 750, y: 20 },
    { seat: 6, name: 'Folder', folded: true, stack: null, stackLabel: 'All In', cards: [], x: 1000, y: 300 },
  ];
  w.start();
  const bySeat = Object.fromEntries(w.body(0).seats.map((s) => [s.seat, s]));
  assert.strictEqual(bySeat[1].allIn, false);
  assert.strictEqual(bySeat[2].allIn, true, 'stack label reads "All In"');
  assert.strictEqual(bySeat[2].stackDollars, null);
  assert.strictEqual(bySeat[2].stackText, 'All In');
  assert.strictEqual(bySeat[3].allIn, false, 'missing stack number alone is NOT all-in');
  assert.strictEqual(bySeat[4].allIn, false, 'the player NAME must not count as the marker');
  assert.strictEqual(bySeat[5].allIn, true, 'all-in class on the seat');
  assert.strictEqual(bySeat[6].allIn, false, 'a folded seat is never all-in');
});

scenario('payload keeps the pokernow.v1 shape the server validates', () => {
  const w = makeWorld();
  headsUp(w, { heroStack: 76, heroBet: 10 });
  w.start();
  const p = w.body(0);
  assert.strictEqual(p.schema, 'pokernow.v1');
  assert.strictEqual(p.gameId, 'pgl-harness');
  assert.deepStrictEqual(p.button, { seat: 6 });
  assert.strictEqual(p.potDollars, 12);
  assert.strictEqual(w.requests[0].url, 'http://127.0.0.1:8765/pokernow/ingest');
  assert.strictEqual(w.requests[0].method, 'POST');
  for (const s of p.seats) {
    for (const k of ['seat', 'cards', 'stackDollars', 'betDollars', 'betText', 'allIn', 'folded', 'isHero', 'isActor', 'angleCW', 'name']) {
      assert.ok(k in s, `seat key ${k}`);
    }
  }
  assert.deepStrictEqual(p.seats[0].cards, HERO);
});

let ran = 0;
const only = process.argv[3];
for (const [name, fn] of scenarios) {
  if (only && !name.includes(only)) continue;
  try {
    fn();
  } catch (e) {
    console.error(`FAIL: ${name}\n${e && e.stack ? e.stack : e}`);
    process.exit(1);
  }
  ran += 1;
}
console.log(`OK ${ran} scenarios`);
