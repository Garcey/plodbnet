"use strict";
// Home games — the player's dock: action buttons, bet sizing, pre-actions,
// the between-hands status strip, the on-felt banner, keyboard shortcuts.
//
// The dock's DOM is built ONCE and updated in place, so a poll never resets
// the amount a player is typing or dragging.
(function () {
  const HG = (globalThis.HG = globalThis.HG || {});
  const $ = (id) => document.getElementById(id);
  const C = () => HG.core;
  const icon = (id, cls) => `<svg class="ico ${cls || ""}"><use href="#${id}"/></svg>`;
  const P = { built: false, key: null, raiseTo: 0, presets: [], stripSig: "", leftSig: "", rightSig: "", bannerSig: "", labelSig: "", sizingOpen: false };
  // Phones / short windows: the sizing panel floats above the buttons and only
  // opens when the player reaches for Bet/Raise (tap once to size, again to confirm).
  const compact = () => !!(globalThis.matchMedia && matchMedia("(max-width: 760px), (max-height: 700px)").matches);

  function build() {
    if (P.built) return;
    P.built = true;
    $("actbar").innerHTML =
      `<div id="sizing" hidden><div class="sz-row"><div class="sz-presets" id="raise-presets"></div></div>` +
      `<div class="sz-row"><button class="sz-step" id="sz-minus" type="button" aria-label="Smaller">−</button>` +
      `<input type="range" id="sz-slider" min="0" max="1000" value="0" aria-label="Bet size"/>` +
      `<button class="sz-step" id="sz-plus" type="button" aria-label="Bigger">+</button>` +
      `<div class="money sz-amt"><input class="input" id="raise-input" inputmode="decimal" autocomplete="off" aria-label="Raise to"/></div></div>` +
      `<div class="sz-row" style="justify-content:space-between"><span class="sz-range" id="sz-range"></span><span class="sz-range" id="sz-pot"></span></div></div>` +
      `<div id="act-btns" hidden><button class="act fold" id="fold-btn" type="button"><kbd>F</kbd><span>Fold</span></button>` +
      `<button class="act call" id="check-btn" type="button"><kbd>C</kbd><span id="call-lbl">Check</span><small id="call-amt"></small></button>` +
      `<button class="act raise" id="raise-go" type="button"><kbd>R</kbd><span id="raise-lbl">Bet</span><small id="raise-amt"></small></button></div>` +
      `<div id="pre-row" hidden></div><div id="status-strip" hidden></div>`;
    $("fold-btn").addEventListener("click", () => doFold());
    $("check-btn").addEventListener("click", () => doCall());
    $("raise-go").addEventListener("click", () => {
      const s = C().G.state;
      if (compact() && !P.sizingOpen && s && s.legal.raise) { P.sizingOpen = true; render(s, s); return; }
      doRaise();
    });
    $("stage-box").addEventListener("pointerdown", () => { if (P.sizingOpen && compact()) { P.sizingOpen = false; const s = C().G.state; if (s) render(s, s); } });
    $("sz-slider").addEventListener("input", (e) => {
      const s = C().G.state; if (!s) return;
      const b = C().raiseBoundsTo(s), k = Number(e.target.value) / 1000;
      setRaise(s, snapCents(s, b.min + (b.max - b.min) * k), "slider");
    });
    $("sz-minus").addEventListener("click", () => nudge(-1));
    $("sz-plus").addEventListener("click", () => nudge(1));
    const inp = $("raise-input");
    inp.addEventListener("input", () => {
      const s = C().G.state, c = C().toCents(inp.value);
      if (!s || c == null) return;
      C().G.raiseTouched = true;
      setRaise(s, C().centsToChips(c, s), "input", true);
    });
    inp.addEventListener("blur", () => { const s = C().G.state; if (s) setRaise(s, P.raiseTo, "blur"); });
    inp.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); doRaise(); } });
    $("raise-presets").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-i]");
      const s = C().G.state;
      if (!b || !s) return;
      C().G.raiseTouched = true;
      setRaise(s, P.presets[Number(b.dataset.i)].to, "preset");
    });
    $("pre-row").addEventListener("change", (e) => {
      const box = e.target.closest("input[data-pre]");
      if (box) C().setPreAction(box.dataset.pre);
    });
  }

  // ------------------------------------------------------------- bet sizing
  function snapCents(s, chips) {
    // whole cents when the stake allows it; the ends of the window stay exact
    const b = C().raiseBoundsTo(s);
    if (chips <= b.min) return b.min;
    if (chips >= b.max) return b.max;
    const snapped = C().centsToChips(C().chipsToCents(chips, s), s);
    return Math.max(b.min, Math.min(b.max, snapped));
  }
  function isAllIn(s, to) {
    const me = s.seats[s.my_seat];
    const rb = s.raise_bounds || {};
    return !!me && to >= C().raiseBoundsTo(s).max && (rb.max_chips || 0) >= (me.stack_chips || 0);
  }
  function setRaise(s, to, from, soft) {
    const b = C().raiseBoundsTo(s);
    const clamped = Math.max(b.min, Math.min(b.max, Math.round(to)));
    P.raiseTo = soft ? Math.round(to) : clamped;
    const shown = clamped;
    if (from !== "slider") $("sz-slider").value = String(b.max > b.min ? Math.round(((shown - b.min) / (b.max - b.min)) * 1000) : 1000);
    $("sz-slider").style.setProperty("--fill", (b.max > b.min ? ((shown - b.min) / (b.max - b.min)) * 100 : 100) + "%");
    if (from !== "input") $("raise-input").value = (C().chipsToCents(shown, s) / 100).toFixed(2);
    const allin = isAllIn(s, shown);
    const opening = !(s.to_call_chips > 0) && !(s.street_commit_chips > 0);
    $("raise-lbl").textContent = allin ? "All-in" : opening ? "Bet" : "Raise to";
    $("raise-amt").textContent = C().fmtAmt(C().chipsToCents(shown, s), s);
    $("raise-go").classList.toggle("allin", allin);
    document.querySelectorAll("#raise-presets button").forEach((x) => x.classList.toggle("on", P.presets[Number(x.dataset.i)].to === shown));
  }
  function nudge(dir) {
    const s = C().G.state;
    if (!s) return;
    C().G.raiseTouched = true;
    setRaise(s, snapCents(s, P.raiseTo + dir * (s.stakes.bb_chips || 10000)), "step");
  }
  function buildPresets(s) {
    const b = C().raiseBoundsTo(s);
    const raw = [["Min", b.min], ["⅓ pot", C().potBetTo(s, 1 / 3)], ["½ pot", C().potBetTo(s, 0.5)], ["¾ pot", C().potBetTo(s, 0.75)], ["Pot", C().potBetTo(s, 1)], ["Max", b.max]];
    const out = [];
    raw.forEach(([label, to]) => {
      const v = snapCents(s, to);
      const hit = out.find((p) => p.to === v);
      // A size that clamps onto another one is the SAME bet: keep one button,
      // named for what it really is.
      if (hit) { if (label === "Max" || (label === "Pot" && hit.label !== "Min")) hit.label = label; return; }
      out.push({ label, to: v });
    });
    out.forEach((p) => { if (p.to === b.max) p.label = isAllIn(s, p.to) ? "All-in" : p.label === "Max" ? "Pot" : p.label; });
    P.presets = out;
    $("raise-presets").innerHTML = out.map((p, i) => `<button type="button" data-i="${i}" title="${C().fmtAmt(C().chipsToCents(p.to, s), s)}">${p.label}</button>`).join("");
  }

  // ---------------------------------------------------------------- actions
  function doFold() {
    const s = C().G.state;
    if (s && C().myTurn(s) && s.legal.fold) C().act({ gate: "fold" });
  }
  function doCall() {
    const s = C().G.state;
    if (s && C().myTurn(s) && s.legal.check_call) C().act({ gate: "check_call" });
  }
  async function doRaise() {
    const s = C().G.state;
    if (!s || !C().myTurn(s) || !s.legal.raise) return;
    const to = C().clampRaiseTo(s, P.raiseTo || C().raiseBoundsTo(s).min);
    if (C().G.prefs.confirmAllIn && isAllIn(s, to)) {
      const ok = await HG.ui.confirmDialog({ title: "Go all-in?", text: `This puts your whole stack (${C().fmtAmt(C().chipsToCents(to, s), s)} total this street) in the middle.`, okLabel: "All-in", danger: true });
      if (!ok) return;
    }
    C().act({ gate: "raise", raise_to_chips: to });
  }

  // ------------------------------------------------------------------ render
  function preRow(s) {
    const owe = C().heroToCallCents(s), cur = C().G.preAction;
    const opts = owe > 0
      ? [["fold", "Fold"], ["call", "Call " + C().fmtAmt(owe, s)], ["call_any", "Call any"]]
      : [["check_fold", "Check / Fold"], ["check", "Check"], ["call_any", "Call any"]];
    const sig = opts.map((o) => o.join(":")).join("|") + "#" + cur;
    const row = $("pre-row");
    if (row.dataset.sig === sig) return;
    row.dataset.sig = sig;
    row.innerHTML = opts.map(([k, l]) => `<label class="chk ${cur === k ? "on" : ""}"><input type="checkbox" data-pre="${k}" ${cur === k ? "checked" : ""}/>${l}</label>`).join("");
  }

  function strip(s) {
    const el = $("status-strip");
    const me = Number.isInteger(s.my_seat) ? s.seats[s.my_seat] : null;
    const busy = s.phase === "in_hand" || s.runout.blocking;
    const btns = [];
    let msg = "";
    if (s.status !== "open") { msg = "<b>This table is closed.</b> The final ledger is in the side panel."; btns.push(["lobby", "Back to lobby", "primary"]); }
    else if (s.my_request) { msg = `Waiting for the host to approve your <b>${C().fmtAmt(s.my_request.amount_cents, s)}</b> ${s.my_request.kind === "sit" ? "buy-in" : "top-up"}…`; btns.push(["cancelreq", "Cancel request", ""]); }
    else if (!me) msg = s.seats.some((x) => x.empty) ? "<b>You're watching.</b> Pick an open seat on the table to join." : "<b>You're watching.</b> The table is full right now.";
    else if (me.pending_remove) msg = "You're leaving — you'll be cashed out when this hand ends.";
    else if (me.sitting_out) { msg = "<b>You're sitting out.</b>"; btns.push(["back", "I'm back", "primary"]); }
    else if (!(busy && me.in_hand) && me.stack_cents <= s.stakes.ante_cents) { msg = "<b>You're out of chips.</b> Reload to be dealt into the next hand."; btns.push(["topup", "Add chips", "gold"]); }
    else if (s.phase === "in_hand") msg = !me.in_hand ? "You'll be dealt in next hand." : me.folded ? "You folded — watching the rest of the hand." : me.all_in ? "<b>You're all in.</b> Good luck." : "";
    else if (s.runout.blocking) msg = (s.runout.shown_len || 0) >= 5 ? "<b>Showdown</b>" : "Running it out…";
    else if (!s.running) {
      if (s.is_host) { msg = s.eligible_count >= 2 ? "<b>Everyone's ready.</b>" : "Waiting for a second player — send the invite link."; btns.push(s.eligible_count >= 2 ? ["start", "Start game", "gold"] : ["invite", "Copy invite link", ""]); }
      else msg = "Waiting for the host to start the game.";
    } else if (s.eligible_count < 2) msg = "Waiting for another player with chips…";
    else if (s.fair && s.fair.next && s.fair.next.pending) msg = s.fair.next.attempt > 1 ? "<b>Reshuffling…</b> a device did not confirm the last shuffle." : "<b>Shuffling…</b> the players' devices are cutting the deck.";
    else if (s.next_deal_in_secs != null) msg = `Next hand in <b class="num" id="deal-count">${Math.max(1, Math.ceil(s.next_deal_in_secs))}s</b>`;
    else if (s.can_deal) { msg = "Ready for the next hand."; btns.push(["deal", "Deal next hand", "primary"]); }
    if (s.can_show) btns.push(["show", "Show my cards", ""]);
    if (s.can_rabbit && me) btns.push(["rabbit", "Rabbit hunt", ""]);
    if (s.can_deal && s.next_deal_in_secs != null && (s.is_host || me)) btns.push(["deal", "Deal now", ""]);
    if (me && me.queued_topup_cents) msg += (msg ? " · " : "") + `<span class="pos">+${C().fmtAmt(me.queued_topup_cents, s)} after this hand</span>`;
    // the countdown is ticked locally (the push only arrives when something changes)
    P.dealAt = s.next_deal_in_secs != null ? performance.now() + s.next_deal_in_secs * 1000 : null;
    const sig = msg.replace(/id="deal-count">\d+s/, "") + "|" + btns.map((b) => b[0]).join(",");
    if (sig === P.stripSig) return;
    P.stripSig = sig;
    el.innerHTML = `<span>${msg}</span>`;
    btns.forEach(([k, label, cls]) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "btn sm " + cls; b.textContent = label;
      b.addEventListener("click", () => stripAction(k));
      el.appendChild(b);
    });
  }
  function stripAction(k) {
    const s = C().G.state;
    if (!s) return;
    if (k === "lobby") C().showLobby(false);
    else if (k === "back") C().tablePost("sit_out", { on: false }).catch(() => {});
    else if (k === "topup") HG.ui.openTopUp();
    else if (k === "start") C().tablePost("run", { running: true }).catch(() => {});
    else if (k === "invite") HG.ui.copyInvite(s.id);
    else if (k === "deal") C().deal(s);
    else if (k === "show") C().tablePost("show", { hand_no: s.hand_no }).catch(() => {});
    else if (k === "rabbit") C().tablePost("rabbit").catch(() => {});
    else if (k === "cancelreq") C().tablePost("request", { action: "cancel" }).catch(() => {});
  }

  function sides(s) {
    const me = Number.isInteger(s.my_seat) ? s.seats[s.my_seat] : null;
    const left = $("dock-left"), right = $("dock-right");
    const busy = s.phase === "in_hand" || s.runout.blocking;
    const lsig = me && s.status === "open" ? `${me.sitting_out}:${me.sit_out_next}:${s.needs_approval}:${me.pending_remove}` : "none";
    if (lsig !== P.leftSig) {
      P.leftSig = lsig;
      left.innerHTML = "";
      if (me && s.status === "open" && !me.pending_remove) {
        const lab = document.createElement("label");
        const on = me.sitting_out || me.sit_out_next;
        lab.className = "chk" + (on ? " on" : "");
        lab.innerHTML = `<input type="checkbox" ${on ? "checked" : ""}/>${me.sitting_out ? "Sitting out" : "Sit out next hand"}`;
        lab.firstChild.addEventListener("change", (e) => {
          C().tablePost("sit_out", e.target.checked ? { on: true, next_hand: true } : { on: false }).catch(() => { P.leftSig = ""; });
        });
        left.appendChild(lab);
        const tu = document.createElement("button");
        tu.type = "button"; tu.className = "btn sm"; tu.title = "Add chips to your stack (they land after the hand if you are in one)";
        tu.innerHTML = icon("i-pluscircle", "sm") + (s.needs_approval ? "Request chips" : "Add chips");
        tu.addEventListener("click", () => HG.ui.openTopUp());
        left.appendChild(tu);
      }
    }
    const rsig = `${s.last_hand_no}:${!!me}:${s.is_member}`;
    if (rsig !== P.rightSig) {
      P.rightSig = rsig;
      right.innerHTML = "";
      if (s.last_hand_no && s.is_member) {
        const lh = document.createElement("button");
        lh.type = "button"; lh.className = "btn sm"; lh.innerHTML = icon("i-clock", "sm") + "Last hand";
        lh.addEventListener("click", () => HG.ui.openHand(s.id, C().G.state.last_hand_no));
        right.appendChild(lh);
      }
      if (me) {
        const em = document.createElement("button");
        em.type = "button"; em.className = "btn sm"; em.innerHTML = icon("i-smile", "sm") + "React";
        em.addEventListener("click", (e) => HG.ui.openMenu(e.currentTarget, Object.entries(HG.table.EMOTES).map(([k, g]) => ({ label: `${g}  ${k.toUpperCase()}`, onClick: () => C().tablePost("react", { emote: k }).catch(() => {}) }))));
        right.appendChild(em);
      }
    }
  }

  function banner(s) {
    const el = $("banner");
    let html = "";
    if (s.status !== "open") html = `<b>Table closed</b><span>Thanks for playing. The final ledger and hand history stay available.</span>`;
    else if (s.phase === "waiting" && !s.hand_no) {
      const seated = s.seats.filter((x) => !x.empty).length;
      if (s.is_host) html = `<b>${seated < 2 ? "Invite your friends" : "Ready when you are"}</b><span>${seated < 2 ? "Send them the link — they pick a seat and buy in. You need at least two players." : `${seated} players seated. Start the game and the server deals every hand.`}</span>`;
      else html = `<b>${Number.isInteger(s.my_seat) ? "You're in" : "Pick a seat"}</b><span>${Number.isInteger(s.my_seat) ? "Waiting for the host to start the game." : "Tap any open seat around the table to buy in."}</span>`;
    }
    if (html !== P.bannerSig) {
      P.bannerSig = html;
      el.innerHTML = html;
      if (html && s.status === "open" && s.is_host) {
        const row = document.createElement("div");
        row.className = "row";
        row.innerHTML = `<button class="btn sm" data-a="invite">${icon("i-link", "sm")}Copy invite link</button>` + (s.eligible_count >= 2 ? `<button class="btn sm gold" data-a="start">${icon("i-play", "sm")}Start game</button>` : "");
        row.addEventListener("click", (e) => { const b = e.target.closest("button[data-a]"); if (b) stripAction(b.dataset.a); });
        el.appendChild(row);
      }
    }
    el.hidden = !html;
    $("center").style.visibility = html ? "hidden" : "visible";
  }

  function heroLabels(s) {
    const me = Number.isInteger(s.my_seat) ? s.seats[s.my_seat] : null;
    const desc = me && me.in_hand && me.hand_desc && s.phase !== "waiting" ? me.hand_desc : null;
    const cap = (d) => d.charAt(0).toUpperCase() + d.slice(1);
    const html = desc ? desc.map((d, k) => (d ? `<span class="hero-hand-label"><span class="hhl-tag">${k + 1}</span>${C().esc(cap(d))}</span>` : "")).join("") : "";
    if (html !== P.labelSig) { P.labelSig = html; $("hero-hand-labels").innerHTML = html; }
  }

  function render(s, prev, opts) {
    build();
    const mine = C().myTurn(s), alive = C().inHandAlive(s);
    const bar = $("actbar");
    bar.classList.toggle("my-turn", mine);
    $("act-btns").hidden = !mine;
    if (!mine) P.sizingOpen = false;
    $("sizing").hidden = !(mine && s.legal.raise && (!compact() || P.sizingOpen));
    $("pre-row").hidden = !(alive && !mine);
    $("status-strip").hidden = mine || (alive && !mine);
    if (mine) {
      const key = `${s.id}:${s.hand_no}:${s.action_seq}`;
      const fresh = key !== P.key;
      if (fresh) { P.sizingOpen = false; $("sizing").hidden = !(s.legal.raise && !compact()); }
      P.key = key;
      $("fold-btn").disabled = !s.legal.fold;
      $("check-btn").disabled = !s.legal.check_call;
      $("raise-go").disabled = !s.legal.raise;
      const owe = s.to_call_cents || 0;
      const me = s.seats[s.my_seat];
      $("call-lbl").textContent = owe > 0 ? (me && owe >= me.stack_cents ? "Call all-in" : "Call") : "Check";
      $("call-amt").textContent = owe > 0 ? C().fmtAmt(owe, s) : "";
      if (s.legal.raise) {
        if (fresh || (opts && opts.unitChanged)) buildPresets(s);
        if (fresh || !C().G.raiseTouched) setRaise(s, C().raiseBoundsTo(s).min, "init");
        else if (opts && opts.unitChanged) setRaise(s, P.raiseTo, "init");
        const b = C().raiseBoundsTo(s);
        $("sz-range").textContent = `${C().fmtAmt(C().chipsToCents(b.min, s), s)} – ${C().fmtAmt(C().chipsToCents(b.max, s), s)}`;
        $("sz-pot").textContent = `pot ${C().fmtAmt(s.pot_cents, s)}`;
      }
    } else {
      P.key = null;
      if (alive) preRow(s);
      else { $("pre-row").dataset.sig = ""; strip(s); }
    }
    if (mine || alive) P.stripSig = "";
    sides(s);
    banner(s);
    heroLabels(s);
  }

  function onClock() { /* the seat ring carries the clock; hook kept for the dock */ }

  // ---------------------------------------------------------------- hotkeys
  function onKey(e) {
    if (e.key === "Escape") { if (HG.ui.closeTop()) e.preventDefault(); else if (C().G.preAction) C().setPreAction(null); return; }
    if (!C().G.prefs.hotkeys || e.ctrlKey || e.metaKey || e.altKey) return;
    const t = e.target, tag = t && t.tagName;
    const typing = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || (t && t.isContentEditable);
    if (typing && !(t.id === "raise-input" && (e.key === "ArrowUp" || e.key === "ArrowDown"))) return;
    if (HG.uiState.modals.length || HG.uiState.drawer) return;
    const s = C().G.state;
    if (!s || !C().myTurn(s)) return;
    const k = e.key.toLowerCase();
    if (k === "f") { e.preventDefault(); doFold(); }
    else if (k === "c") { e.preventDefault(); doCall(); }
    else if (k === "r" || k === "b") { e.preventDefault(); doRaise(); }
    else if (s.legal.raise && (e.key === "ArrowUp" || e.key === "ArrowRight")) { e.preventDefault(); nudge(1); }
    else if (s.legal.raise && (e.key === "ArrowDown" || e.key === "ArrowLeft")) { e.preventDefault(); nudge(-1); }
    else if (s.legal.raise && /^[1-6]$/.test(e.key) && P.presets[Number(e.key) - 1]) { e.preventDefault(); C().G.raiseTouched = true; setRaise(s, P.presets[Number(e.key) - 1].to, "preset"); }
  }

  function tickDeal() {
    const el = $("deal-count");
    if (el && P.dealAt != null) el.textContent = Math.max(1, Math.ceil((P.dealAt - performance.now()) / 1000)) + "s";
  }

  HG.play = { init() { build(); document.addEventListener("keydown", onKey); setInterval(tickDeal, 250); }, render, onClock };
})();
