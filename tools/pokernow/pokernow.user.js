// ==UserScript==
// @name         plodbbot PokerNow bridge
// @namespace    plodbbot
// @version      1.2.0
// @description  Streams PokerNow PLO5 double-board bomb-pot table state from the DOM to the local plodbbot study server.
// @match        https://www.pokernow.com/games/*
// @match        https://www.pokernow.club/games/*
// @run-at       document-idle
// @grant        GM_xmlhttpRequest
// @connect      127.0.0.1
// @connect      localhost
// ==/UserScript==

/*
 * Browser-side collector for the PokerNow live-capture source.
 *
 * PokerNow renders the entire table as DOM elements, so — unlike ClubGG, which
 * needs pixel OCR — we read game state directly and exactly. This script
 * snapshots the table on every change (MutationObserver, debounced), normalizes
 * it into the `pokernow.v1` payload the server expects, and POSTs it to
 * http://127.0.0.1:8765/pokernow/ingest. The server maps each payload to a
 * FrameState and runs it through the same reconstructor/Session pipeline ClubGG
 * OCR uses.
 *
 * Transport note: we use GM_xmlhttpRequest, NOT fetch/WebSocket. An https
 * PokerNow tab cannot reach 127.0.0.1 from page context — Chrome's Private
 * Network Access + mixed-content rules block it (a page-context ws:// to
 * localhost just hangs in CONNECTING). GM_xmlhttpRequest runs in Tampermonkey's
 * privileged context and bypasses those restrictions, which is why @connect is
 * declared above.
 *
 * Nothing here mutates the page or automates play — it only reads. Keep it that
 * way: this is a study aid, not a bot.
 */
(function () {
  'use strict';

  const INGEST_URL = 'http://127.0.0.1:8765/pokernow/ingest';
  const HEARTBEAT_MS = 2000;

  // ---- DOM extraction (validated against a live PLO5 double-board table) ----

  const clean = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const RANK = (v) => (v === '10' ? 'T' : v); // engine rank char

  // ".card" → "Ts" / "Ad" / null (face-down or absent).
  function cardStr(el) {
    const v = clean(el.querySelector('.value')?.textContent);
    const s = clean(el.querySelector('.suit:not(.sub-suit)')?.textContent);
    if (!v || !s) return null;
    return RANK(v) + s.toLowerCase();
  }

  // First number in a text blob, as a float (or null).
  function num(t) {
    const m = clean(t || '').replace(/[, ]/g, '').match(/-?\d+(\.\d+)?/);
    return m ? parseFloat(m[0]) : null;
  }

  function tableCenter() {
    const table = document.querySelector('.table') || document.body;
    const r = table.getBoundingClientRect();
    return { cx: r.left + r.width / 2, cy: r.top + r.height / 2 };
  }

  // Clockwise angle (deg, 0..360 from 12 o'clock) of a seat around the table.
  function seatAngle(el, center) {
    const r = el.getBoundingClientRect();
    const px = r.left + r.width / 2;
    const py = r.top + r.height / 2;
    let a = (Math.atan2(px - center.cx, -(py - center.cy)) * 180) / Math.PI;
    if (a < 0) a += 360;
    return Math.round(a);
  }

  // "All In" marker. PokerNow replaces the stack NUMBER with this label when a
  // player has 0 behind.
  const ALL_IN_RE = /\ball[\s-]*in\b/i;

  function readSeat(el, center) {
    const seat = parseInt((el.className.match(/table-player-(\d+)/) || [])[1], 10);
    const cards = [...el.querySelectorAll('.card')].map(cardStr);
    const stackEl = el.querySelector('.table-player-stack');
    const stackText = clean(stackEl?.textContent);
    const stack = num(stackEl?.querySelector('.normal-value')?.textContent);
    // Bet-value: a numeric amount (bet/raise/call/ante) OR a verb ("check").
    const betRaw = clean(el.querySelector('.table-player-bet-value')?.textContent);
    const betNum = num(betRaw);
    const folded = /\bfold\b/.test(el.className);
    const name = clean(el.querySelector('.table-player-name')?.textContent);
    // All-in ONLY when the DOM says so (review 2026-09-20). This used to be
    // `stack === null && !folded`, but a missing `.normal-value` is just as
    // often a mid-render snapshot or some other non-numeric seat state, and the
    // server turns all-in into "stack 0" — a full-stack drop that corroborates
    // any bet shown and routes the seat through the all-in paths. Evidence, in
    // order: the stack label's text, an all-in class on the seat, or the label
    // anywhere in the seat EXCEPT the player's name (a "Mr All In" must not
    // count). `stackText` rides along so a capture log shows what was read.
    const seatTextSansName = name
      ? clean(el.textContent).split(name).join(' ')
      : clean(el.textContent);
    const allIn =
      !folded &&
      stack === null &&
      (ALL_IN_RE.test(stackText) ||
        /\ball-?in\b/i.test(el.className) ||
        ALL_IN_RE.test(seatTextSansName));
    return {
      seat,
      name,
      isHero: el.classList.contains('you-player'),
      isActor: el.classList.contains('decision-current'),
      angleCW: seatAngle(el, center),
      stackDollars: stack,
      stackText: stackText || null,
      allIn,
      folded,
      betDollars: betNum,
      betText: betNum === null && betRaw ? betRaw.toLowerCase() : null,
      cards,
    };
  }

  function snapshot() {
    const center = tableCenter();
    const seatEls = [...document.querySelectorAll('.table-player')].filter((e) =>
      /table-player-\d+/.test(e.className)
    );
    if (seatEls.length === 0) return null;

    const boards = [...document.querySelectorAll('.table-cards')].map((b) => ({
      run: (b.className.match(/run-(\d)/) || [])[1],
      cards: [...b.querySelectorAll('.card')].map(cardStr),
    }));

    const seats = seatEls.map((el) => readSeat(el, center));
    const hero = seats.find((s) => s.isHero);

    const dealer = document.querySelector('.dealer-button-ctn');
    const button = dealer
      ? parseInt((dealer.className.match(/dealer-position-(\d+)/) || [])[1], 10)
      : null;

    return {
      schema: 'pokernow.v1',
      // Identifies which PokerNow game this frame came from. The server locks
      // onto one game and ignores frames from any other tab/window, so a second
      // open table can't interleave its state into your live hand.
      gameId: (location.pathname.match(/\/games\/([^/?#]+)/) || [])[1] || location.pathname,
      variant: document.querySelector('.five-cards') ? 'plo5' : 'unknown',
      bombPot: !!document.querySelector('.bomb-pot-signal, .bomb-pot-banner'),
      potDollars: num(document.querySelector('.table-pot-size')?.textContent),
      button: button != null && !Number.isNaN(button) ? { seat: button } : null,
      boards,
      heroCards: hero ? hero.cards : null,
      seats,
    };
  }

  // ---- Status badge ----------------------------------------------------------

  const badge = document.createElement('div');
  Object.assign(badge.style, {
    position: 'fixed', bottom: '8px', right: '8px', zIndex: 2147483647,
    font: '12px/1.4 monospace', padding: '4px 8px', borderRadius: '6px',
    background: 'rgba(0,0,0,0.75)', color: '#fff', pointerEvents: 'none',
    whiteSpace: 'pre',
  });
  document.body.appendChild(badge);
  let sent = 0;
  function setBadge(state, color) {
    badge.style.borderLeft = `4px solid ${color}`;
    badge.textContent = `plodbbot · ${state}\nframes sent: ${sent}`;
  }
  setBadge('starting…', '#e0a000');

  // ---- POST transport (GM_xmlhttpRequest) + change-dedupe --------------------
  //
  // Delivery contract (review 2026-09-20): the server rebuilds the hand from
  // the ORDERED stream of distinct frames, so a frame may be delayed but never
  // silently lost or reordered while the table is live.
  //   * transport failure (offline / timeout / abort): the frame goes back on
  //     the FRONT of the queue and is retried after RETRY_MS. It used to be
  //     dropped, and nothing drained again until a later mutation happened to
  //     change the fingerprint — one hiccup cost an action for the whole hand.
  //   * HTTP error status: the server saw and rejected THIS frame; resending
  //     it cannot help and would block everything behind it, so it is dropped
  //     and draining continues.
  //   * every path out of enqueueCurrent() ends in drain().

  let lastKey = '';
  let queue = [];         // distinct frames awaiting send, in order (FIFO)
  let inFlight = false;
  let nextAttemptAt = 0;  // backoff gate (ms epoch) after a transport failure
  let retryTimer = null;
  const MAX_QUEUE = 60;   // safety bound if the server stalls
  const RETRY_MS = 1000;

  function payloadKey(p) {
    // Cheap structural fingerprint so identical states don't re-send.
    return JSON.stringify([
      p.button, p.potDollars,
      p.boards.map((b) => b.cards),
      p.seats.map((s) => [s.seat, s.stackDollars, s.betDollars, s.betText, s.cards, s.isActor, s.folded, s.allIn]),
    ]);
  }

  // Bound memory by dropping the OLDEST frames (only reachable if the server is
  // badly stalled or offline for a long stretch).
  function trimQueue() {
    while (queue.length > MAX_QUEUE) queue.shift();
  }

  function scheduleRetry() {
    if (retryTimer !== null) return;
    retryTimer = setTimeout(() => {
      retryTimer = null;
      drain();
    }, RETRY_MS);
  }

  function transportFailed(payload, isHeartbeat, label) {
    inFlight = false;
    nextAttemptAt = Date.now() + RETRY_MS;
    // A heartbeat carries no new state — nothing to keep. A real frame goes
    // back where it came from so order is preserved.
    if (!isHeartbeat) {
      queue.unshift(payload);
      trimQueue();
    }
    setBadge(`${label} (q${queue.length})`, '#c0392b');
    scheduleRetry();
  }

  function doPost(payload, isHeartbeat) {
    inFlight = true;
    GM_xmlhttpRequest({
      method: 'POST',
      url: INGEST_URL,
      headers: { 'Content-Type': 'application/json' },
      data: JSON.stringify(payload),
      timeout: 4000,
      onload: (res) => {
        inFlight = false;
        nextAttemptAt = 0;
        if (res.status >= 200 && res.status < 300) {
          if (!isHeartbeat) sent += 1;
          setBadge(`connected (q${queue.length})`, '#2faf4f');
        } else if (res.status === 409) {
          setBadge('ClubGG OCR active in app', '#c0392b');
        } else {
          setBadge(`server error ${res.status}`, '#c0392b');
        }
        drain(); // immediately send the next queued frame
      },
      onerror: () => transportFailed(payload, isHeartbeat, 'server offline (start on :8765)'),
      ontimeout: () => transportFailed(payload, isHeartbeat, 'server timeout'),
      onabort: () => transportFailed(payload, isHeartbeat, 'request aborted'),
    });
  }

  function canSend() {
    return !inFlight && Date.now() >= nextAttemptAt;
  }

  // Send queued frames one at a time, in order. We send EVERY distinct frame
  // (not just the latest) so the actor-transition frames the reconstructor
  // needs to detect checks/closes are never dropped. With the server's
  // per-frame work kept light (no model inference on ingest) the queue drains
  // in ~ms, so this stays near-empty under live play.
  function drain() {
    if (queue.length === 0) return;
    if (!canSend()) {
      // Backing off after a failure: make sure something wakes us up again
      // even if the table goes quiet.
      if (!inFlight) scheduleRetry();
      return;
    }
    doPost(queue.shift(), false);
  }

  // Snapshot the table, or null when there is nothing truthful to report
  // (no table on the page, or a transient mid-render DOM that threw).
  function safeSnapshot() {
    try {
      return snapshot();
    } catch (e) {
      return null; // next mutation / heartbeat retries
    }
  }

  // Snapshot the table; enqueue it if it's an observable change. Driven by the
  // MutationObserver (which Chrome does NOT throttle in a background tab, so the
  // table stays live regardless of which tab is focused).
  //
  // ALWAYS ends in drain(): the "nothing changed" / "no table" exits used to
  // `return` first, so after a failed send the queue sat untouched through
  // every mutation that did not alter the fingerprint (timers, animations).
  function enqueueCurrent() {
    const p = safeSnapshot();
    if (p) {
      const key = payloadKey(p);
      if (key !== lastKey) {
        lastKey = key;
        queue.push(p);
        trimQueue();
      }
    }
    drain();
  }

  // ---- Observer (re-attachable) ---------------------------------------------
  //
  // PokerNow is a single-page app and can REPLACE the `.table` node. An
  // observer bound to the old, now-detached node never fires again, so the
  // stream silently stopped while the heartbeat kept re-sending the last
  // snapshot it had — stale state presented as live, and replayed as a fresh
  // frame into a restarted server. Re-resolve the root on every heartbeat.
  const OBSERVE_OPTS = { subtree: true, childList: true, attributes: true, characterData: true };
  const obs = new MutationObserver(enqueueCurrent);
  let root = null;

  function ensureObserver() {
    const want = document.querySelector('.table') || document.body;
    if (want === root && root.isConnected !== false) return false;
    obs.disconnect();
    root = want;
    obs.observe(root, OBSERVE_OPTS);
    return true;
  }

  // Heartbeat keeps the server's "connected" status warm during idle stretches
  // (no DOM mutations between hands). It sends a snapshot taken NOW — never a
  // remembered payload — so it cannot assert a table state that is no longer on
  // screen: unchanged state is deduped by the server, changed state (mutations
  // we missed) is enqueued as a real frame, and no table means no heartbeat
  // (the server's recency-based "connected" then lapses, which is the truth).
  function heartbeat() {
    ensureObserver();
    if (queue.length > 0) {
      drain(); // doubles as the retry pump while backing off
      return;
    }
    if (!canSend()) return;
    const p = safeSnapshot();
    if (!p) return;
    const key = payloadKey(p);
    if (key !== lastKey) {
      lastKey = key;
      queue.push(p);
      drain();
    } else {
      doPost(p, true);
    }
  }
  setInterval(heartbeat, HEARTBEAT_MS);

  // Observe the table subtree; every change enqueues a fresh snapshot.
  ensureObserver();

  enqueueCurrent();
})();
