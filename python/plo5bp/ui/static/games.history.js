"use strict";
// Home games — hand history: the click-through replayer (with the network's marks and
// Check shuffle), Open in Study, and the lifetime hand database (yours or a club
// member's). Part of the UI split out of games.ui.js (FE-005).
(function () {
  const HG = globalThis.HG;
  const UI = HG.ui;
  const { C, html, put, icon, U, h, avatar, d2, gameOf, toast, openModal, loading, segHtml, segWire, miniCards, fillMiniCards, fmtWhen } = HG.uikit;
  // arrow-key hints only where there is a keyboard to press them (CPY-008)
  const hasKeys = () => !!(globalThis.matchMedia && matchMedia("(hover: hover) and (pointer: fine)").matches);

  // ------------------------------------------------------------ hand replayer
  // A CLICK-THROUGH, not a video: the hand opens on the flop with the first
  // player to act; forward plays one action, back takes one away. Every player
  // decision carries the network's verdict (same marks as the Trainer), and any
  // position can be sent to the Study tab as a spot.
  const MARKS = { best: "✓✓", correct: "✓", inaccuracy: "~", wrong: "✗", blunder: "✗✗" };
  const MARK_LABEL = { best: "Best move", correct: "Correct", inaccuracy: "Inaccuracy", wrong: "Wrong move", blunder: "Blunder" };
  const kindOfAction = (x) => C().actionKind(x);
  function gradeChip(g, small) {
    if (!g) return "";
    const label = MARK_LABEL[g.cat] || g.cat;
    return html`<span class="grade g-${g.cat}${small ? " sm" : ""}" title="${label} · ${Math.round(g.score)}/100 vs the network">${MARKS[g.cat] || "?"}${small ? "" : html` <em>${label}</em> <b>${Math.round(g.score)}</b>`}</span>`;
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
  // PLO67: a seat's cards on the street being replayed (0 flop, 1 turn, 2 river): the
  // first `counts[i]` of its cards in the order they came, shown high to low — or that
  // many face down for a hand the viewer may not see. Other games: the stored hand.
  function replayHole(x, rec, boardN) {
    const i = Math.max(0, Math.min(2, boardN - 3));
    const counts = x.counts || null, n = counts ? counts[i] : (x.hole ? x.hole.length : rec.hole_count || 5);
    if (x.hole_seq && x.hole_seq.length) return x.hole_seq.slice(0, n).sort((a, b) => b - a);
    if (x.hole && x.hole.length && x.hole[0] >= 0 && !counts) return x.hole;
    return Array(n).fill("x");
  }

  // "turn · burn 7♦, everyone in gets a card" — PLO67's face-up burn of a street (else "")
  const cardName = (c) => "23456789TJQKA"[Math.floor(c / 4)].replace("T", "10") + "♣♦♥♠"[c % 4];
  function burnNote(rec, street) {
    const j = { flop: 0, turn: 1, river: 2 }[String(street).toLowerCase()];
    const b = j == null ? null : (rec.burns || [])[j];
    if (b == null) return "";
    const red = b % 4 === 1 || b % 4 === 2;
    return html` <span class="burn-note ${red ? "red" : ""}">· burn ${cardName(b)}${red ? " — everyone in gets a card" : ""}</span>`;
  }
  // Opens at once with "Loading…" and fills in when the hand lands (HGH-002: on a slow
  // phone the button looked dead, and a second tap opened a second copy).
  async function openHand(gid, no) {
    const key = `${gid}:${no}`;
    if (U.openingHand && U.openingHand.key === key && !U.openingHand.api.closed) return;
    const shell = openModal({ title: `Hand #${Number(no)}`, body: loading(), wide: true, autofocus: false, buttons: [{ label: "Close", cls: "primary" }] });
    shell.modal.classList.add("xwide");
    U.openingHand = { key, api: shell };
    let rec;
    try { rec = await C().j(`/games/api/tables/${gid}/hands/${no}`); } catch (e) { shell.close(null); return toast(e.message, "err"); }
    if (shell.closed) return;
    const N = (rec.actions || []).length;
    // An all-in runout (record v2: `equities` per street, `runout_from`): after the last
    // action the replay goes on street by street — each street's equities on the felt,
    // then the result (owner, 2026-09-29).
    const FROM = Number(rec.runout_from) || 0;
    const R = rec.equities && FROM >= 3 ? Math.max(0, 5 - FROM) : 0;
    const END = N + R;
    const STREETS = ["flop", "turn", "river"];
    const pct = (x) => (x == null ? "–" : Math.round(x * 100) + "%");
    const gradeAt = {};
    (rec.grades || []).forEach((g) => { gradeAt[g.i] = g; });
    const names = {};
    rec.seats.forEach((x) => (names[x.seat] = x.is_me ? "You" : x.name));
    const hero = (rec.seats.find((x) => x.is_me) || rec.seats[0] || {}).seat || 0;
    const order = rec.seats.map((x) => x.seat);
    const n = rec.num_seats || 8;
    let k = 0;
    const body = h("div", { class: "rp" });
    put(body, html`<div class="rp-main"><div class="rp-felt" id="rp-felt"></div>
      <div class="rp-banner" id="rp-banner"></div>
      <div class="rp-ctl"><button type="button" class="btn sm" id="rp-first" title="Start of the hand (Home)" aria-label="Start of the hand">⏮</button><button type="button" class="btn" id="rp-prev" title="Back one action (←)" aria-label="Back one action">◀</button><span class="rp-step num" id="rp-step" aria-live="polite"></span><button type="button" class="btn primary" id="rp-next" title="Play the next action (→)" aria-label="Next action">▶</button><button type="button" class="btn sm" id="rp-last" title="End of the hand (End)" aria-label="End of the hand">⏭</button>
      <span class="spacer"></span><button class="btn sm" id="rp-link" type="button" title="Copy a link to this hand — club members open it with the same reveal rules">${icon("i-link", "sm")}<span>Copy link</span></button>${rec.fair && HG.fair ? html`<button class="btn sm" id="rp-fair" title="Re-check this hand's sealed deck, its cut and every card you can see">${icon("i-shield", "sm")}<span>Check shuffle</span></button>` : ""}${studyable(rec) ? html`<button class="btn gold sm" id="rp-study" title="Send this exact spot to the Study tab">${icon("i-chart", "sm")}Open in Study</button>` : ""}</div></div>
      <div class="rp-side"><div class="rsec flush-top"><h4>Action</h4><div id="rp-list" class="rp-list"></div></div><div id="rp-result"></div></div>`);
    const q = (id) => body.querySelector("#" + id);

    function paint() {
      const st = replayState(rec, Math.min(k, N));
      // all in: the street being run out and its equities (none on the result)
      const j = R && k >= N ? k - N : -1;
      let eqMap = null;
      if (j >= 0) {
        st.boardN = FROM + j;
        st.over = j === R;
        if (!st.over) eqMap = rec.equities[String(st.boardN)] || null;
      }
      // seats around the felt, hero at the bottom (each one's place rides in data-vars:
      // --x / --y, games.css .rp-seat)
      const seats = rec.seats.map((x) => {
        const rel = (x.seat - hero + n) % n;
        const th = Math.PI / 2 + (rel * 2 * Math.PI) / n;
        const px = (50 + 44 * Math.cos(th)).toFixed(2), py = (50 + 40 * Math.sin(th)).toFixed(2);
        const p = st.seats[x.seat];
        const acting = st.next && st.next.seat === x.seat;
        const known = x.hole && x.hole.length && x.hole[0] >= 0;
        const cards = p.folded && !known ? "" : html`<span class="mini-cards" data-cards="${replayHole(x, rec, st.boardN).join(",")}"></span>`;
        const res = st.over ? html`<b class="num ${x.delta_cents > 0 ? "pos" : x.delta_cents < 0 ? "neg" : "muted"}">${x.delta_cents > 0 ? "+" : ""}${d2(x.delta_cents)}</b>` : "";
        const e = eqMap && eqMap[String(x.seat)];
        const eq = e ? html`<span class="rp-eq num" title="${names[x.seat]}'s chance to win board 1 · board 2 from here"><span data-b="1">${pct(e[0])}</span><span data-b="2">${pct(e[1])}</span></span>` : "";
        return html`<div class="rp-seat ${p.folded ? "folded" : ""} ${acting ? "acting" : ""}" data-vars="x:${px}%;y:${py}%">${cards}<div class="rp-plate">${avatar(x.name, x.name, "sm", x.avatar)}<div><b>${names[x.seat]}${x.seat === rec.button ? html` <i class="rp-d">D</i>` : ""}</b><span class="num">${d2(Math.max(0, p.stack))}</span></div></div>${eq}${p.bet > 0 ? html`<span class="rp-bet num">${d2(p.bet)}</span>` : ""}${res}</div>`;
      });
      // PLO67: the burns turned up by this street (a red one dealt everyone still in a card)
      const burns = (rec.burns || []).slice(0, Math.max(0, st.boardN - 2));
      const felt = q("rp-felt");
      put(felt, html`${seats}<div class="rp-center"><span class="rp-pot num">Pot ${d2(st.pot)}</span><span class="mini-cards" data-cards="${(rec.board_a || []).slice(0, st.boardN).join(",")}"></span><span class="mini-cards" data-cards="${(rec.board_b || []).slice(0, st.boardN).join(",")}"></span>${burns.length ? html`<span class="rp-burns" title="Burn cards, face up: a red one deals everyone still in the hand another card"><small>Burns</small><span class="mini-cards" data-cards="${burns.join(",")}"></span></span>` : ""}</div>`);
      fillMiniCards(felt);
      // banner: the action just played (with its verdict) and who is up
      const last = st.last, g = last && j < 1 ? gradeAt[k - 1] : null;
      const dealtNow = j >= 1 ? STREETS[st.boardN - 3] : null;  // (a street the runout just dealt)
      const left = dealtNow
        ? html`<span class="rp-act k-runout"><b>${dealtNow[0].toUpperCase() + dealtNow.slice(1)}</b> dealt${burnNote(rec, dealtNow)}</span>`
        : last ? html`<span class="rp-act k-${kindOfAction(last)}"><b>${names[last.seat] || "?"}</b> ${last.label}${last.auto ? html` <small class="muted">(clock)</small>` : ""}</span>${gradeChip(g)}`
        : html`<span class="muted">Flop dealt — everyone anted ${d2(rec.ante_cents)}.</span>`;
      const right = st.next ? html`<span class="muted">${String(st.street).toUpperCase()} · <b class="tx">${names[st.next.seat] || "?"}</b> to act</span>`
        : !st.over ? html`<span class="pill">All in · running it out</span>`
        : html`<span class="pill gold">Hand over</span>`;
      put(q("rp-banner"), html`${left}<span class="spacer"></span>${right}`);
      q("rp-step").textContent = `${k} / ${END}`;
      q("rp-prev").disabled = q("rp-first").disabled = k === 0;
      q("rp-next").disabled = q("rp-last").disabled = k === END;
      body.querySelectorAll("#rp-list .log-row").forEach((r) => r.classList.toggle("on", Number(r.dataset.i) === k - 1));
      const cur = body.querySelector("#rp-list .log-row.on");
      if (cur) cur.scrollIntoView({ block: "nearest" });
      q("rp-result").hidden = !st.over;
    }
    function go(to) { k = Math.max(0, Math.min(END, to)); paint(); }

    // action list (click = jump to just after that action)
    const list = [];
    let street = null;
    (rec.actions || []).forEach((a2, i) => {
      if (a2.street !== street) { street = a2.street; list.push(html`<div class="log-street">${street}${burnNote(rec, street)}</div>`); }
      list.push(html`<button type="button" class="log-row k-${kindOfAction(a2)}" data-i="${i}"><span class="nm">${names[a2.seat] || "?"}</span><span class="lb">${a2.label}</span>${gradeChip(gradeAt[i], true)}</button>`);
    });
    if (R) {
      // all in: the runout's streets, each a step of its own (its equities on the felt)
      for (let j = 1; j <= R; j++) {
        const stName = STREETS[FROM - 3 + j];
        list.push(html`<div class="log-street">${stName}${burnNote(rec, stName)}</div>`);
        list.push(html`<button type="button" class="log-row k-runout" data-i="${N + j - 1}"><span class="nm">All in</span><span class="lb">${j < R ? "run out — the equities" : "run out — the result"}</span></button>`);
      }
    } else {
      // PLO67: a street run out with nobody to act still turned its burn up (and dealt the cards)
      const seenStreets = new Set((rec.actions || []).map((x) => String(x.street).toLowerCase()));
      STREETS.slice(0, (rec.burns || []).length).forEach((st) => {
        if (!seenStreets.has(st)) list.push(html`<div class="log-street">${st}${burnNote(rec, st)}</div>`);
      });
    }
    put(q("rp-list"), list.length ? html`${list}` : html`<div class="muted">No betting — everyone was all-in from the ante.</div>`);
    q("rp-list").addEventListener("click", (e) => { const r = e.target.closest(".log-row"); if (r) go(Number(r.dataset.i) + 1); });
    const awards = (rec.awards || []).map((w) => {
      const who = w.winners.map((x) => names[x] || "?").join(" & ");
      const lab = w.winners.length === 1 && w.labels && w.labels[String(w.winners[0])] ? ` with ${w.labels[String(w.winners[0])]}` : "";
      return html`<div class="settle-row"><span>${w.uncontested ? "Uncontested" : "Board " + (w.board === "b" ? 2 : 1)} · ${who}${lab}</span><b>${d2(w.cents)}</b></div>`;
    });
    const flows = (rec.flows || []).map((f) => html`<div class="settle-row"><span>${names[f.from] || "?"} → ${names[f.to] || "?"}</span><b>${d2(f.cents)}</b></div>`);
    put(q("rp-result"), html`${awards.length ? html`<div class="rsec"><h4>Pots</h4>${awards}</div>` : ""}${flows.length ? html`<div class="rsec"><h4>Who paid whom</h4>${flows}</div>` : ""}`);
    q("rp-first").addEventListener("click", () => go(0));
    q("rp-prev").addEventListener("click", () => go(k - 1));
    q("rp-next").addEventListener("click", () => go(k + 1));
    q("rp-last").addEventListener("click", () => go(END));
    if (q("rp-study")) q("rp-study").addEventListener("click", () => askStudy(rec, Math.min(k, N)));
    // a link to this hand (FEAT-010): the table's page opens it for any club member,
    // with the usual reveal rules for the cards
    q("rp-link").addEventListener("click", async () => {
      const url = `${location.origin}/games/t/${encodeURIComponent(gid)}?hand=${Number(no)}`;
      try { await navigator.clipboard.writeText(url); toast("Link to this hand copied — send it to the club", "ok"); }
      catch (_) { toast(url, "", 9000); }
    });
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
      else if (e.key === "End") { e.preventDefault(); go(END); }
    };
    document.addEventListener("keydown", onKey);
    const graded = (rec.grades || []).length;
    const api = shell;
    api.setTitle(`Hand #${rec.hand_no}${rec.table_name ? " · " + rec.table_name : ""}`,
      `${gameOf(rec.variant).label} · Pot ${d2(rec.pot_cents)} · ante ${d2(rec.ante_cents)} · ${rec.showdown ? "showdown" : "won without showdown"}` +
        (!gameOf(rec.variant).graded ? ` · ${gameOf(rec.variant).label} isn't graded yet` : rec.grades == null ? " · accuracy is still being worked out" : graded ? "" : " · no graded decisions") +
        (hasKeys() ? " · use ← → to step" : " · tap ◀ ▶ to step"));
    api.setBody(body);
    api.then(() => document.removeEventListener("keydown", onKey));
    paint();
  }

  // Sending a spot to Study replaces what is set up there (and switches Study to PLO5):
  // ask first (HGH-003), with "don't ask again". The confirm button's click is the one
  // that opens the new tab — a browser only allows that inside a click.
  const STUDY_ASK_KEY = "hg.studyask.v1";
  function askStudy(rec, k) {
    let skip = false;
    try { skip = localStorage.getItem(STUDY_ASK_KEY) === "no"; } catch (_) { /* private mode */ }
    if (skip) return openInStudy(rec, k);
    const body = h("div", { class: "stack-12" },
      html`<p class="flush">This spot replaces the hand set up in your Study tab${rec.variant && rec.variant !== "plo5" ? "" : " (and switches Study to PLO5)"}. Study opens in a new tab.</p>
      <label class="chk-line"><input type="checkbox" id="st-noask"/> Don't ask me again</label>`);
    openModal({
      title: "Open this spot in Study?", body, autofocus: false,
      buttons: [{ label: "Cancel", cls: "ghost" }, {
        label: "Open in Study", cls: "gold",
        onClick: () => {
          if (body.querySelector("#st-noask").checked) { try { localStorage.setItem(STUDY_ASK_KEY, "no"); } catch (_) { /* private mode */ } }
          openInStudy(rec, k);  // (synchronously, inside this click)
        },
      }],
    });
  }
  // The Study tab works on the signed-in user's own server-side session, so a
  // spot is "copied" by driving Study's normal API: reset, stakes, seats + stacks
  // (hero = seat 0, clockwise), cards dealt so far, then the actions up to here.
  // (Study deals PLO5 — five hole cards; a hand of another game has no spot to copy there)
  function studyable(rec) { return (rec.variant || "plo5") === "plo5"; }
  async function openInStudy(rec, k) {
    if (!studyable(rec)) return toast(`Study works with PLO5 hands — this was ${gameOf(rec.variant).label}`, "err");
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
        openModal({ title: "Spot copied to Study", sub: "Your browser blocked the new tab. The spot is loaded — open Study with this link.", body: html`<a class="btn gold block" href="/?mode=study" target="_blank" rel="noopener">Open Study in a new tab</a>`, buttons: [{ label: "Done", cls: "primary" }] });
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
  // `variant` = one game's numbers (PLO5 / PLO6 are different games: the club section
  // opens this on the game it shows); a switch changes it once more than one was played.
  async function openMyHands(gameId, player, clubId, variant) {
    const other = player && !player.is_me ? player : null;
    const base = other ? `/games/api/players/${other.user_id}` : "/games/api/my";
    // (one club's numbers — the club's lobby, or the club of the table it was opened from)
    const club = clubId || C().G.clubId;
    const cc = UI.currentClub(), clubName = (cc && cc.id === club && cc.name) || "";
    const F = { sort: "time", dir: "desc", game: gameId || "", variant: variant || "", offset: 0, rows: [], total: 0 };
    // FEAT-004: the period the club's numbers show (All time / This month / 30 days)
    const since = UI.statsSince ? UI.statsSince() : "";
    const qs = () => [club ? `club=${encodeURIComponent(club)}` : "", F.variant ? `variant=${encodeURIComponent(F.variant)}` : "", since ? `since=${encodeURIComponent(since)}` : ""].filter(Boolean).join("&");
    let stats;
    const title = (other ? `${other.name} — hands & stats` : "My hands & stats") + (clubName ? ` · ${clubName}` : "") + (since && UI.periodLabel ? ` · ${UI.periodLabel()}` : "");
    const key = `${base}:${club}`;
    if (U.openingDb && U.openingDb.key === key && !U.openingDb.api.closed) return;  // (a second tap)
    const api = openModal({
      title,
      sub: other ? `Every hand they played${clubName ? " in " + clubName : ""}. You see the cards you saw at the table: your own, and hands that were shown.` : `Every hand you played${clubName ? " in " + clubName : ""}, with the network's accuracy rating (PLO5).`,
      body: loading(), wide: true, autofocus: false, buttons: [{ label: "Close", cls: "primary" }],
    });
    api.modal.classList.add("xwide");
    U.openingDb = { key, api };
    const loadStats = async () => { stats = await C().j(base + "/stats" + (qs() ? "?" + qs() : "")); };
    try { await loadStats(); } catch (e) { api.close(null); return toast(e.message, "err"); }
    if (api.closed) return;
    const acc = (v) => (v == null ? "–" : Math.round(v) + "%");
    const played = () => (stats.games || []).filter((g) => g.hands > 0);
    const body = h("div", { class: "db" });
    put(body, html`<div id="db-head"></div><div id="db-graph"></div><div id="db-vs"></div>
      <div class="db-bar"><select class="input" id="db-game"></select>${segHtml("db-sort", [["time", "Date"], ["pot", "Pot size"], ["net", "Profit / loss"], ["accuracy", "Accuracy"]], "time")}<button type="button" class="btn sm" id="db-dir" title="Reverse the order">Newest first</button></div>
      <div id="db-list"></div><button class="btn block" id="db-more" hidden>Load more</button>`);
    const q = (id) => body.querySelector("#" + id);
    segWire(body);  // (the sort: wired once — the game switch is wired with each paint of the head)
    const tone = (c) => (c > 0 ? "pos" : c < 0 ? "neg" : "");
    const paintHead = () => {
      const games = played().map((g) => g.code);
      if (F.variant && !games.includes(F.variant)) games.push(F.variant);
      const ungraded = !!F.variant && !gameOf(F.variant).graded;
      const sw = games.length > 1 ? segHtml("db-v", [["", "All games"]].concat(games.map((c) => [c, gameOf(c).label])), F.variant, null, "db-games") : "";
      put(q("db-head"), html`${sw}<div class="pcard-stats"><div><b>${stats.hands}</b><small>${F.variant ? gameOf(F.variant).label + " hands" : "Hands"}</small></div><div><b class="${tone(stats.net_cents)}">${stats.net_cents > 0 ? "+" : ""}${d2(stats.net_cents)}</b><small>${F.variant ? "Net" : "Lifetime net"}</small></div>${ungraded ? html`<div><b>–</b><small>Not graded (no ${gameOf(F.variant).label} network)</small></div>` : html`<div><b>${acc(stats.accuracy)}</b><small>Accuracy (${stats.graded} decisions)</small></div>`}<div><b>${stats.hands ? Math.round((100 * stats.wins) / stats.hands) + "%" : "–"}</b><small>Hands won</small></div></div>`);
      // (the profit graph sits between the numbers and the head to head: #db-graph, loadGraph)
      put(q("db-vs"), (stats.versus || []).length ? html`<div class="rsec flush-top"><h4>Head to head${F.variant ? " · " + gameOf(F.variant).label : ""}${clubName ? " — " + clubName : ""}</h4><div class="vs">${stats.versus.map((v) => html`<span class="vs-chip"><span>${v.name}</span><b class="num ${tone(v.net_cents)}">${v.net_cents > 0 ? "+" : ""}${d2(v.net_cents)}</b></span>`)}</div></div>` : "");
      const sess = stats.sessions || [], tag = !F.variant && played().length > 1;
      put(q("db-game"), html`<option value="">All sessions (${sess.length})</option>${sess.map((x) => html`<option value="${x.id}" ${x.id === F.game ? "selected" : ""}>${x.name}${tag ? " · " + gameOf(x.variant).label : ""} · ${x.hands} hands · ${x.net_cents >= 0 ? "+" : ""}${d2(x.net_cents)}${x.accuracy == null ? "" : " · " + acc(x.accuracy)}</option>`)}`);
      if (!sw) return;
      segWire(q("db-head"));
      q("db-v").addEventListener("pick", async (e) => {
        F.variant = e.detail; F.game = "";
        try { await loadStats(); } catch (err) { toast(err.message, "err"); return; }
        paintHead();
        load(true);
        loadGraph();
      });
    };
    // FEAT-012: how it went — the running net, hand by hand, for what the window shows
    // (all sessions, or the one picked; the game; the period). An older server has no
    // series: the graph just isn't there.
    let graphSeq = 0;
    const loadGraph = async () => {
      const seq = ++graphSeq;
      let sr = null;
      try { sr = await C().j(`${base}/series?` + [qs(), F.game ? `game=${encodeURIComponent(F.game)}` : ""].filter(Boolean).join("&")); } catch (_) { sr = null; }
      if (seq !== graphSeq || api.closed) return;
      drawGraph(q("db-graph"), sr, F.game ? ((stats.sessions || []).find((x) => x.id === F.game) || {}).name : "");
    };
    const draw = () => {
      const host = q("db-list");
      put(host, F.rows.length ? "" : html`<div class="muted empty-note">${other ? "No hands here yet." : "No hands yet. Every hand you play at this club's tables lands here."}</div>`);
      const tag = !F.variant && played().length > 1;  // (every game in one list: say which each hand was)
      F.rows.forEach((x) => {
        const net = x.net_cents;
        const row = h("button", { class: "hand-row db-row", type: "button" },
          html`<span class="no">#${x.hand_no}</span><span class="hr-cards">${x.my_hole ? miniCards(x.my_hole) : ""}${miniCards(x.board_a, "gap")}${miniCards(x.board_b, "gap")}</span><span class="net ${net > 0 ? "pos" : net < 0 ? "neg" : "muted"}">${net > 0 ? "+" : ""}${d2(net)}</span><span></span><span class="who">${x.table_name}${tag ? " · " + gameOf(x.variant).label : ""} · ${fmtWhen(x.ended_at)} · pot ${d2(x.pot_cents)}${x.showdown ? " · showdown" : ""}</span><span class="muted num small">${x.accuracy == null ? "" : acc(x.accuracy)}</span>`);
        row.addEventListener("click", () => openHand(x.game_id, x.hand_no));
        host.appendChild(row);
      });
      fillMiniCards(host);
      q("db-more").hidden = F.rows.length >= F.total;
      // (dates read "Newest first", not "High to low")
      q("db-dir").textContent = F.sort === "time" ? (F.dir === "desc" ? "Newest first" : "Oldest first") : (F.dir === "desc" ? "↓ High to low" : "↑ Low to high");
    };
    const load = async (reset) => {
      if (reset) { F.offset = 0; F.rows = []; }
      try {
        const d = await C().j(`${base}/hands?sort=${F.sort}&dir=${F.dir}&limit=40&offset=${F.offset}` + (F.game ? `&game=${encodeURIComponent(F.game)}` : "") + (qs() ? "&" + qs() : ""));
        F.rows = F.rows.concat(d.hands); F.total = d.total; F.offset += d.limit;
      } catch (e) { toast(e.message, "err"); }
      draw();
    };
    paintHead();
    q("db-sort").addEventListener("pick", (e) => { F.sort = e.detail; load(true); });
    q("db-dir").addEventListener("click", () => { F.dir = F.dir === "desc" ? "asc" : "desc"; load(true); });
    q("db-game").addEventListener("change", (e) => { F.game = e.target.value; load(true); loadGraph(); });
    q("db-more").addEventListener("click", () => load(false));
    api.setBody(body);
    load(true);
    loadGraph();
  }

  // --------------------------------------------------------- the profit graph
  // (FEAT-012) The running net as a line from the first hand to the last: green above
  // zero, red below, faint ticks where the next table's hands begin; point at it (or
  // drag a finger along it) for the hand and the running net there.
  let graphIds = 0;
  function drawGraph(host, sr, sessionName) {
    const pts = (sr && sr.points) || [];
    if (!host) return;
    if (pts.length < 3) { put(host, ""); return; }  // (fewer than two hands: no line to draw)
    const W = 600, H = 132, top = 8, bottom = 8;
    const n = pts[pts.length - 1][0] || 1;
    const vals = pts.map((p) => p[1]);
    const hi = Math.max(0, ...vals), lo = Math.min(0, ...vals), span = hi - lo || 1;
    const x = (k) => (k / n) * W;
    const y = (v) => top + (1 - (v - lo) / span) * (H - top - bottom);
    const line = pts.map(([k, v], i) => `${i ? "L" : "M"}${x(k).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");
    const zero = y(0), id = "pg" + ++graphIds;
    const area = `${line} L${W} ${zero.toFixed(1)} L0 ${zero.toFixed(1)} Z`;
    const breaks = (sr.breaks || []).length <= 40 ? sr.breaks || [] : [];
    const signed = (c) => (c > 0 ? "+" : "") + d2(c);
    put(host, html`<figure class="pgraph" aria-label="Running net over ${n} hands: from ${d2(0)} to ${signed(sr.net_cents)}, highest ${signed(sr.best_cents)}, lowest ${signed(sr.worst_cents)}">
      <figcaption><span>${sessionName ? `Running net · ${sessionName}` : "Running net"}</span><span class="num"><b class="pos">${signed(sr.best_cents)}</b> high · <b class="${sr.worst_cents < 0 ? "neg" : ""}">${signed(sr.worst_cents)}</b> low · ${n} hands</span></figcaption>
      <div class="pg-box"><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">
        <defs><clipPath id="${id}-up"><rect x="0" y="0" width="${W}" height="${zero.toFixed(1)}"/></clipPath><clipPath id="${id}-dn"><rect x="0" y="${zero.toFixed(1)}" width="${W}" height="${(H - zero).toFixed(1)}"/></clipPath></defs>
        ${breaks.map((b) => html`<line class="pg-break" x1="${x(b).toFixed(1)}" x2="${x(b).toFixed(1)}" y1="0" y2="${H}"/>`)}
        <path class="pg-area up" d="${area}" clip-path="url(#${id}-up)"/><path class="pg-area dn" d="${area}" clip-path="url(#${id}-dn)"/>
        <line class="pg-zero" x1="0" x2="${W}" y1="${zero.toFixed(1)}" y2="${zero.toFixed(1)}"/>
        <path class="pg-line up" d="${line}" clip-path="url(#${id}-up)"/><path class="pg-line dn" d="${line}" clip-path="url(#${id}-dn)"/>
      </svg><i class="pg-dot" hidden></i><span class="pg-tip num" hidden></span></div></figure>`);
    const box = host.querySelector(".pg-box"), dot = host.querySelector(".pg-dot"), tip = host.querySelector(".pg-tip");
    const show = (e) => {
      const r = box.getBoundingClientRect();
      if (!r.width) return;
      const k = Math.max(0, Math.min(n, ((e.clientX - r.left) / r.width) * n));
      let best = pts[0];
      for (const p of pts) if (Math.abs(p[0] - k) < Math.abs(best[0] - k)) best = p;
      const px = (best[0] / n) * 100, py = (y(best[1]) / H) * 100;
      dot.hidden = tip.hidden = false;
      dot.style.left = tip.style.left = px + "%";
      dot.style.top = py + "%";
      tip.classList.toggle("flip", px > 70);
      tip.textContent = `${best[0] ? `Hand ${best[0]}` : "Start"} · ${signed(best[1])}`;
    };
    box.addEventListener("pointermove", show);
    box.addEventListener("pointerdown", show);
    box.addEventListener("pointerleave", () => { dot.hidden = tip.hidden = true; });
  }

  Object.assign(UI, { openHand, openMyHands });
})();
