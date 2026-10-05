"use strict";
// Home games — your seat and the people at the table: sitting down, adding / taking off
// chips, automatic chips, the host's buy-in requests, the player card (private notes
// and tags), leaving, and the seat menu. Part of the UI split out of games.ui.js (FE-005).
(function () {
  const HG = globalThis.HG;
  const UI = HG.ui;
  const { C, html, put, U, h, avatar, moneyInput, d2, d0, toast, openModal, confirmDialog, closeTopThen, openMenu, segHtml, segWire, loadHands, loading, canBrowse } = HG.uikit;

  // ------------------------------------------------------- seat money dialogs
  // A buy-in / top-up window. With no table maximum there is no real top: the slider
  // stops at a sensible one (5x the default buy-in, `soft`) but a typed amount may go
  // past it — it used to be cut to that invented cap without a word (HGT-010).
  function buyinLimits(s, stackCents) {
    const st = s.stakes, set = s.settings || {};
    const floor = Math.max(st.bb_cents, st.ante_cents + st.bb_cents);
    const lo = stackCents > 0 ? st.bb_cents : Math.max(floor, set.min_buyin_cents || 0);
    let hi = set.max_buyin_cents ? set.max_buyin_cents - stackCents : Math.max(st.default_buyin_cents * 5, lo * 4);
    hi = Math.max(lo, hi);
    return { lo, hi, capped: !!set.max_buyin_cents };
  }
  // the top preset: the table's maximum, or on a table with none just an amount
  const topPreset = (lim) => ({ label: lim.capped ? "Max" : d0(lim.hi), cents: lim.hi });

  // ONE amount picker (FE-002): the big number, a slider, presets and a box — for
  // sitting down, adding or taking off chips, and the host's request dialog (`note`
  // there says "as asked"). Renders into `host`; `.cents` is the amount chosen.
  function moneyPanel(host, o) {
    const { lo, hi } = o, soft = !!o.soft;
    const clamp = (c) => Math.max(lo, soft ? c : Math.min(hi, c));
    let cents = clamp(o.start);
    put(host, html`<div class="bigmoney"><span class="md-big"></span><small class="md-note"></small></div>
      <input type="range" class="md-range" min="${lo}" max="${hi}" step="${Math.max(1, Math.round(o.step || 100))}" value="${Math.min(hi, cents)}" aria-label="${o.label || "Amount"}"/>
      <div class="sz-row"><div class="sz-presets"></div><div class="sz-amt">${moneyInput(o.id || "md-input", cents, o.label || "Amount")}</div></div>
      ${o.hint ? html`<small class="muted">${o.hint}</small>` : ""}${o.guide && UI.openGuide ? html`<button type="button" class="linkish md-guide">New to double-board bomb pots? How a hand works</button>` : ""}`);
    const guide = host.querySelector(".md-guide");
    if (guide) guide.addEventListener("click", () => UI.openGuide());
    const big = host.querySelector(".md-big"), note = host.querySelector(".md-note"), range = host.querySelector(".md-range"), input = host.querySelector("input.input");
    const show = (c) => {
      big.textContent = d2(c);
      note.textContent = o.note ? o.note(c) : o.ante ? `${Math.floor(c / o.ante)} antes` : "";
    };
    const sync = (from) => {
      cents = clamp(cents);
      show(cents);
      if (from !== "range") range.value = String(Math.min(hi, cents));
      if (from !== "input") input.value = (cents / 100).toFixed(2);
      range.style.setProperty("--fill", (hi > lo ? ((Math.min(hi, cents) - lo) / (hi - lo)) * 100 : 100) + "%");
      range.setAttribute("aria-valuetext", d2(cents));
    };
    const seen = new Set();  // two presets on the same amount read as a bug: keep the first
    (o.presets || []).filter((p) => p.cents >= lo && (soft || p.cents <= hi) && !seen.has(p.cents) && seen.add(p.cents)).slice(0, 4).forEach((p) => {
      const b = h("button", { type: "button" }, p.label);
      b.addEventListener("click", () => { cents = p.cents; sync(); });
      host.querySelector(".sz-presets").appendChild(b);
    });
    range.addEventListener("input", () => { cents = Number(range.value); sync("range"); });
    input.addEventListener("input", () => { const c = C().toCents(input.value); if (c != null) { cents = c; show(clamp(c)); } });
    input.addEventListener("change", () => sync());
    input.addEventListener("keydown", (e) => { if (e.key === "Enter" && o.onEnter) { e.preventDefault(); sync(); o.onEnter(); } });
    sync();
    return { get cents() { return clamp(cents); }, sync };
  }
  function moneyDialog(o) {
    const body = h("div", { class: "stack-14" });
    const panel = h("div", { class: "stack-14" });
    body.appendChild(panel);
    if (o.footer) body.appendChild(o.footer);
    let api = null;
    const go = async () => { try { await o.onOk(pick.cents); if (api) api.close(true); } catch (_) { /* toasted */ } };
    const pick = moneyPanel(panel, Object.assign({}, o, { onEnter: go }));
    api = openModal({ title: o.title, sub: o.sub, body, buttons: [{ label: "Cancel", cls: "ghost" }, { label: o.okLabel, cls: "primary", onClick: async () => { pick.sync(); await o.onOk(pick.cents); } }] });
  }
  // Change seats (FEAT-013): an empty seat tapped by someone already seated. Holding
  // cards, the move waits for the hand to end (games.ui.js sends it then).
  const holdingCards = (s) => { const me = Number.isInteger(s.my_seat) ? s.seats[s.my_seat] : null; return !!(me && me.in_hand && !me.folded && (s.phase === "in_hand" || (s.runout && s.runout.blocking))); };
  async function moveTo(seat) {
    const s = C().G.state;
    try {
      await C().tablePost("move", { seat });
      HG.sound && HG.sound.play("sit");
    } catch (e) {
      const d = String((e && e.message) || "");
      if (/being dealt/.test(d)) { U.moveAfter = { table: s.id, seat }; toast(`The next hand was already being dealt — you move to seat ${seat + 1} after it`, "gold", 5000); }
      else if (e && e.status === 409) toast(/reserved|taken/.test(d) ? `Seat ${seat + 1} was just taken` : d, "err");  // (post() says nothing on a 409)
    }
  }
  function openMove(seat) {
    const s = C().G.state;
    if (!s || !Number.isInteger(s.my_seat)) return;
    const me = s.seats[s.my_seat];
    if (me.leaving || me.pending_remove) return toast("You're leaving this table", "err");
    const later = holdingCards(s);
    confirmDialog({
      title: `Move to seat ${seat + 1}?`,
      text: later ? "You're in this hand — you move as soon as it's over. Your chips come with you." : "Your chips and everything else at this table come with you.",
      okLabel: later ? "Move after this hand" : "Move here",
    }).then((ok) => {
      if (!ok) return;
      if (!later || !holdingCards(C().G.state)) return moveTo(seat);
      U.moveAfter = { table: s.id, seat };
      toast(`You move to seat ${seat + 1} when this hand ends`, "ok");
    });
  }
  // (games.ui.js render: a move asked for during a hand, once the hand is over)
  function pendingMove(s) {
    const mv = U.moveAfter;
    if (!mv || !s) return;
    if (mv.table !== s.id || !Number.isInteger(s.my_seat) || s.status !== "open") { U.moveAfter = null; return; }
    if (holdingCards(s) || U.moveBusy) return;
    U.moveAfter = null; U.moveBusy = true;
    if (!s.seats[mv.seat] || !s.seats[mv.seat].empty) { U.moveBusy = false; toast(`Seat ${mv.seat + 1} was taken while you played the hand`, "err"); return; }
    moveTo(mv.seat).finally(() => { U.moveBusy = false; });
  }
  function openSit(seat) {
    const s = C().G.state;
    if (!s) return;
    if (s.my_seat != null) return openMove(seat);
    const lim = buyinLimits(s, 0), dflt = s.stakes.default_buyin_cents;
    moneyDialog({
      title: `Take seat ${seat + 1}`, sub: `${s.name} · ${d2(s.stakes.bb_cents)} bb · ante ${d2(s.stakes.ante_cents)}`,
      lo: lim.lo, hi: lim.hi, soft: !lim.capped, start: dflt, ante: s.stakes.ante_cents, step: s.stakes.bb_cents, label: "Buy-in",
      okLabel: s.needs_approval ? "Request seat" : "Sit down",
      hint: (s.needs_approval ? "The host approves buy-ins here — your seat is held while they decide. " : "") + (lim.capped ? `Buy-in ${d2(lim.lo)} – ${d2(lim.hi)}. ` : "") + "Real money is settled between you — the ledger just keeps score.",
      guide: true,  // (a "How a hand works" link: FEAT-009)
      presets: [{ label: "Min", cents: lim.lo }, { label: d0(dflt), cents: dflt }, { label: d0(dflt * 2), cents: dflt * 2 }, topPreset(lim)],
      onOk: async (cents) => {
        const out = await C().tablePost("sit", { seat, buyin_cents: cents });
        if (out && out.my_request) toast("Request sent — waiting for the host", "ok");
        else HG.sound && HG.sound.play("sit");
      },
    });
  }

  // Chips: ONE dialog (HGT-016). Where the host allows taking chips off, an Add / Take off
  // switch sits on top and swaps the amount in place — adding, the common case, used to
  // cost an extra dialog and a click.
  function addConfig(s, me, holding) {
    const lim = buyinLimits(s, me.stack_cents), dflt0 = s.stakes.default_buyin_cents;
    if (lim.capped && lim.hi < s.stakes.bb_cents) return { blocked: `You're at the table maximum (${d2(s.settings.max_buyin_cents)}).` };
    const dflt = Math.max(lim.lo, Math.min(lim.hi, dflt0 - me.stack_cents > 0 ? dflt0 - me.stack_cents : dflt0));
    return {
      sub: `Your stack is ${d2(me.stack_cents)}.` + (holding ? " They are added when this hand ends." : ""),
      lo: lim.lo, hi: lim.hi, soft: !lim.capped, start: dflt, ante: s.stakes.ante_cents, step: s.stakes.bb_cents, label: "Chips to add",
      okLabel: s.needs_approval ? "Request chips" : "Add chips",
      hint: (s.needs_approval ? "The host approves buy-ins here. " : "") + (lim.capped ? `You can top up to ${d2(s.settings.max_buyin_cents)} in total.` : ""),
      // "To $100" tops the stack up to a buy-in; "+$100" adds one on top
      presets: [
        ...(me.stack_cents > 0 && dflt0 > me.stack_cents ? [{ label: `To ${d0(dflt0)}`, cents: dflt0 - me.stack_cents }] : []),
        { label: `+${d0(dflt0)}`, cents: dflt0 }, { label: `+${d0(dflt0 * 2)}`, cents: dflt0 * 2 }, topPreset(lim),
      ],
      onOk: async (cents) => {
        const out = await C().tablePost("rebuy", { amount_cents: cents, queue: true });
        if (out && out.my_request) toast("Request sent — waiting for the host", "ok");
        else if (holding) toast(`${d2(cents)} lands when this hand ends`, "ok");
        else HG.sound && HG.sound.play("chips");
      },
    };
  }
  function removeConfig(s, me, holding) {
    const floor = s.stakes.ante_cents + s.stakes.bb_cents;  // what stays: an ante and a bet
    const hi = me.stack_cents - floor, lo = s.stakes.bb_cents;
    if (hi < lo) return { blocked: `Nothing to take off — you keep at least ${d2(floor)} to stay seated. Leave the table to cash out.` };
    return {
      sub: `Your stack is ${d2(me.stack_cents)}.` + (holding ? " They come off when this hand ends." : ""),
      lo, hi, start: Math.min(hi, Math.max(lo, Math.round(hi / 2 / s.stakes.bb_cents) * s.stakes.bb_cents)),
      ante: s.stakes.ante_cents, step: s.stakes.bb_cents, label: "Chips to take off", okLabel: "Take off",
      hint: `They go back to your ledger as if you had cashed them out. You keep at least ${d2(floor)} on the table.`,
      presets: [
        { label: "Half", cents: Math.round(hi / 2) },
        ...(me.stack_cents - s.stakes.default_buyin_cents >= lo ? [{ label: `Keep ${d0(s.stakes.default_buyin_cents)}`, cents: me.stack_cents - s.stakes.default_buyin_cents }] : []),
        { label: "Max", cents: hi },
      ],
      onOk: async (cents) => {
        await C().tablePost("remove_chips", { amount_cents: cents, queue: holding });
        toast(holding ? `${d2(cents)} comes off when this hand ends` : `${d2(cents)} taken off the table`, "ok");
        HG.sound && HG.sound.play("chips");
      },
    };
  }
  function openTopUp(mode) {
    const s = C().G.state;
    if (!s || s.my_seat == null) return;
    const me = s.seats[s.my_seat];
    // Table stakes: chips asked for while you hold cards land when the hand ends.
    const holding = me.in_hand && (s.phase === "in_hand" || (s.runout && s.runout.blocking));
    const canRemove = !!(s.settings && s.settings.allow_rathole);
    const cfg = { add: addConfig(s, me, holding), remove: canRemove ? removeConfig(s, me, holding) : null };
    let cur = mode === "remove" && canRemove ? "remove" : "add";
    if (!canRemove) {  // (the plain Add chips dialog)
      if (cfg.add.blocked) return toast(cfg.add.blocked);
      return moneyDialog(Object.assign({ title: "Add chips", footer: autoChipsRow(s, me) }, cfg.add));
    }
    const body = h("div", { class: "stack-14" });
    put(body, segHtml("ch-mode", [["add", "Add chips"], ["remove", "Take chips off"]], cur, null, "wide"));
    const panel = h("div", { class: "stack-14" });
    body.appendChild(panel);
    const auto = autoChipsRow(s, me);
    if (auto) body.appendChild(auto);
    let pick = null, api = null;
    const go = async () => { const c = cfg[cur]; if (c.blocked) return; try { await c.onOk(pick.cents); if (api) api.close(true); } catch (_) { /* toasted */ } };
    const paint = () => {
      const c = cfg[cur];
      if (c.blocked) { put(panel, html`<p class="muted flush">${c.blocked}</p>`); pick = null; }
      else pick = moneyPanel(panel, Object.assign({}, c, { onEnter: go }));
      if (api) {
        api.setTitle("Chips", c.sub || `Your stack is ${d2(me.stack_cents)}.`);
        api.setButtons(buttons());
      }
    };
    const buttons = () => {
      const c = cfg[cur];
      return c.blocked ? [{ label: "Close", cls: "primary" }]
        : [{ label: "Cancel", cls: "ghost" }, { label: c.okLabel, cls: "primary", onClick: async () => { pick.sync(); await c.onOk(pick.cents); } }];
    };
    segWire(body);
    body.querySelector("#ch-mode").addEventListener("pick", (e) => { cur = e.detail === "remove" ? "remove" : "add"; paint(); });
    paint();
    api = openModal({ title: "Chips", sub: cfg[cur].sub || `Your stack is ${d2(me.stack_cents)}.`, body, autofocus: false, buttons: buttons() });
  }
  // What automatic chips do for this player here, for the places people look
  // (2026-09-26): the choice used to live only behind the top bar's person icon,
  // so a table set to "Players choose" looked like it had no such option.
  function autoChipsNow(s, me) {
    const top = s.auto_topup.mode, set = s.auto_stack.mode;
    const setOn = set !== "off" && me.auto_stack_cents > 0;
    const topOn = !setOn && top !== "off" && me.topup_target_cents > 0;
    let text = setOn ? `Your stack resets to ${d2(me.auto_stack_cents)} before every hand.`
      : topOn ? `Topped back up to ${d2(me.topup_target_cents)} when you drop below ${d2(me.topup_below_cents || me.topup_target_cents)}.`
      : top === "player" || set === "player" ? "Off. This table lets you choose." : "Off for you.";
    if ((setOn && set === "host") || (topOn && top === "host")) text += " Set by the host.";
    if ((setOn || topOn) && s.needs_approval) text += " Runs once the host trusts you.";
    return { any: top !== "off" || set !== "off", choose: top === "player" || set === "player", text };
  }
  function autoChipsRow(s, me) {
    const a = autoChipsNow(s, me);
    if (!a.any) return null;
    const row = h("div", { class: "setrow" }, html`<div><b>Automatic chips</b><small>${a.text}</small></div>`);
    row.appendChild(h("button", {
      type: "button", class: "btn sm",
      onclick: () => closeTopThen(openAutoChips),  // (this dialog closes, then that one opens)
    }, a.choose ? "Change" : "Details"));
    const box = h("div", { class: "grp" });
    box.appendChild(row);
    return box;
  }
  function openAutoChips() {
    const s = C().G.state;
    if (!s || !Number.isInteger(s.my_seat)) return;
    const me = s.seats[s.my_seat];
    const topMode = s.auto_topup.mode, setMode = s.auto_stack.mode;
    const kinds = [["off", "Off"]];
    if (topMode === "player") kinds.push(["topup", "Auto top-up"]);
    if (setMode === "player") kinds.push(["set", "Set stack"]);
    const cur = setMode !== "off" && me.auto_stack_cents > 0 ? "set" : topMode !== "off" && me.topup_target_cents > 0 ? "topup" : "off";
    const hostLines = [];
    if (topMode === "host") hostLines.push(me.topup_target_cents ? `The host tops you up to ${d2(me.topup_target_cents)} whenever you drop below ${d2(me.topup_below_cents || me.topup_target_cents)}.` : "The host controls auto top-up (off for you).");
    if (setMode === "host") hostLines.push(me.auto_stack_cents ? `The host resets your stack to ${d2(me.auto_stack_cents)} before every hand.` : "The host controls set-stack (off for you).");
    const body = h("div", { class: "stack-14" });
    put(body, html`${hostLines.length ? html`<div class="grp"><h4>Set by the host</h4><p class="flush">${hostLines.map((l, i) => html`${i ? html`<br>` : ""}${l}`)}</p></div>` : ""}
      ${kinds.length > 1
        ? html`<div class="field"><span>Automatic chips</span>${segHtml("ac-kind", kinds, kinds.some((x) => x[0] === cur) ? cur : "off")}</div>
          <div id="ac-top" hidden><div class="row2"><label class="field"><span>Top up to</span>${moneyInput("ac-top-target", me.topup_target_cents || s.stakes.default_buyin_cents)}</label><label class="field"><span>When below</span>${moneyInput("ac-top-below", me.topup_below_cents || me.topup_target_cents || s.stakes.default_buyin_cents)}</label></div>
          <small class="muted">Before each hand, if your stack has dropped below the second amount it is topped back up to the first. It never takes chips off the table.</small></div>
          <div id="ac-set" hidden><label class="field"><span>Stack every hand</span>${moneyInput("ac-set-target", me.auto_stack_cents || s.stakes.default_buyin_cents)}</label>
          <small class="muted">Before EVERY hand your stack is reset to this amount — short stacks are topped up and anything above it goes back to your ledger. This table allows it.</small></div>`
        : (hostLines.length ? "" : html`<p class="muted flush">The host hasn't enabled automatic chips at this table.</p>`)}
      ${s.needs_approval ? html`<small class="muted">The host approves buy-ins here: automatic chips only run once the host trusts you.</small>` : ""}`);
    const showKind = (k) => { const a = body.querySelector("#ac-top"), b = body.querySelector("#ac-set"); if (a) a.hidden = k !== "topup"; if (b) b.hidden = k !== "set"; };
    segWire(body);
    const segEl = body.querySelector("#ac-kind");
    if (segEl) { segEl.addEventListener("pick", (e) => showKind(e.detail)); showKind(kinds.some((x) => x[0] === cur) ? cur : "off"); }
    openModal({
      title: "Automatic chips", body, autofocus: false,
      buttons: kinds.length > 1 ? [{ label: "Cancel", cls: "ghost" }, {
        label: "Save", cls: "primary",
        onClick: async () => {
          const on = body.querySelector("#ac-kind button.on");
          const kind = on ? on.dataset.v : "off";
          const payload = { kind };
          if (kind === "topup") { payload.target_cents = C().toCents(body.querySelector("#ac-top-target").value) || 0; payload.below_cents = C().toCents(body.querySelector("#ac-top-below").value) || 0; }
          if (kind === "set") payload.target_cents = C().toCents(body.querySelector("#ac-set-target").value) || 0;
          await C().tablePost("auto_chips_self", payload);
          toast(kind === "off" ? "Automatic chips off" : kind === "topup" ? `Topping up to ${d2(payload.target_cents)}` : `Stack resets to ${d2(payload.target_cents)} every hand`, "ok");
        },
      }] : [{ label: "Done", cls: "primary" }],
    });
  }

  // ------------------------------------------------------------ the network
  // The site's owner only (the view carries `bot` for them alone; the server checks too —
  // homegame_bot.py): the network plays their seat, or suggests each move, its favourite
  // move or its full strategy. Everyone at the table sees a chip on the seat while it is on.
  function openBot() {
    const s = C().G.state;
    if (!s || !s.bot || !Number.isInteger(s.my_seat)) return;
    const b = s.bot;
    const body = h("div", { class: "stack-14" });
    put(body, html`${b.available ? "" : html`<p class="flush">${b.why}</p>`}
      <div class="field"><span>Your seat</span>${segHtml("bot-mode", [["", "You play"], ["assist", "It suggests"], ["auto", "It plays"]], b.mode || "")}</div>
      <small class="muted" id="bot-mode-help"></small>
      <div class="field"><span>How it plays</span>${segHtml("bot-mix", [["best", "Its favourite move"], ["mix", "Its full strategy"]], b.mix ? "mix" : "best")}</div>
      <small class="muted" id="bot-mix-help"></small>
      <small class="muted">Everyone at the table sees a chip on your seat while it is on, and the hand history marks every move it made or suggested. Those moves are never graded.</small>`);
    const HELP = {
      "": "You play your own hand.",
      assist: "On your turn its move lights up and its bet size is set — you still press the button.",
      auto: "It bets, calls and folds for you, about a second into each turn. Switch it off any time.",
      best: "The same move every time: the one it rates best.",
      mix: "It mixes its moves the way it was trained: each turn one is drawn from its odds.",
    };
    const pickOf = (id) => { const x = body.querySelector(`#${id} button.on`); return x ? x.dataset.v : ""; };
    const help = () => {
      body.querySelector("#bot-mode-help").textContent = HELP[pickOf("bot-mode")] || "";
      body.querySelector("#bot-mix-help").textContent = HELP[pickOf("bot-mix") || "best"];
    };
    segWire(body);
    ["bot-mode", "bot-mix"].forEach((id) => body.querySelector("#" + id).addEventListener("pick", help));
    help();
    openModal({
      title: "The network", sub: "Let the network play your seat, or suggest your moves.", body, autofocus: false,
      buttons: [{ label: "Cancel", cls: "ghost" }, {
        label: "Save", cls: "primary",
        onClick: async () => {
          const mode = pickOf("bot-mode"), mix = pickOf("bot-mix") === "mix";
          if (mode && !b.available) { toast(b.why || "The network can't play here", "err"); return false; }
          await C().tablePost("bot", { mode: mode || "off", mix });
          toast(mode === "auto" ? "The network plays your seat" : mode === "assist" ? "The network suggests your moves" : "The network is off", "ok");
        },
      }],
    });
  }

  // ------------------------------------------------------------ player card
  // Notes + colour tags are PRIVATE: they live in this browser only. Read once and kept
  // in memory (FE-008: the felt asks for every seat on every update); another tab of
  // this browser that changes them is picked up through the storage event.
  const NOTES_KEY = "hg.notes.v1";
  // the tags; their colours are games.css's ([data-tag] on a seat and on a tag button)
  const TAGS = { none: "No tag", red: "Red", amber: "Amber", green: "Green", blue: "Blue", violet: "Violet" };
  let NOTES = null;
  function notesAll() {
    if (NOTES) return NOTES;
    try { NOTES = JSON.parse(localStorage.getItem(NOTES_KEY) || "{}") || {}; } catch (_) { NOTES = {}; }
    return NOTES;
  }
  globalThis.addEventListener && globalThis.addEventListener("storage", (e) => { if (e.key === NOTES_KEY) NOTES = null; });
  function noteFor(uid) { return notesAll()[String(uid)] || { tag: "none", text: "" }; }
  function saveNote(uid, patch) {
    const all = notesAll(), k = String(uid);
    all[k] = Object.assign({ tag: "none", text: "" }, all[k], patch);
    if (all[k].tag === "none" && !all[k].text) delete all[k];
    try { localStorage.setItem(NOTES_KEY, JSON.stringify(all)); } catch (_) { /* private mode */ }
    const s = C().G.state;
    if (s) HG.table.render(s, s);
  }
  // Back up / bring back the notes (FEAT-014): a file only you keep — they stay private.
  function exportNotes() {
    const all = notesAll(), n = Object.keys(all).length;
    if (!n) return toast("No notes or tags to back up yet", "");
    const blob = new Blob([JSON.stringify({ kind: "wrapgto-home-games-notes", v: 1, saved: new Date().toISOString(), notes: all }, null, 1)], { type: "application/json" });
    const a = h("a", { href: URL.createObjectURL(blob), download: `home-games-notes-${new Date().toISOString().slice(0, 10)}.json` });
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
    toast(`${n} note${n === 1 ? "" : "s"} saved to a file`, "ok");
  }
  function importNotes(file) {
    return new Promise((resolve) => {
      const fr = new FileReader();
      fr.onerror = () => { toast("Couldn't read that file", "err"); resolve(false); };
      fr.onload = () => {
        let data = null;
        try { data = JSON.parse(String(fr.result)); } catch (_) { /* below */ }
        const notes = data && data.kind === "wrapgto-home-games-notes" && data.notes && typeof data.notes === "object" ? data.notes : null;
        if (!notes) { toast("That isn't a notes backup", "err"); return resolve(false); }
        const all = notesAll();
        let n = 0;
        for (const [uid, v] of Object.entries(notes)) {
          if (!/^\d+$/.test(uid) || !v || typeof v !== "object") continue;
          const tag = TAGS[v.tag] !== undefined ? v.tag : "none", text = String(v.text || "").slice(0, 500);
          if (tag === "none" && !text) continue;
          all[uid] = { tag, text };  // (a backup's note replaces this browser's note for the same player)
          n++;
        }
        try { localStorage.setItem(NOTES_KEY, JSON.stringify(all)); } catch (_) { /* private mode */ }
        const s = C().G.state;
        if (s) HG.table.render(s, s);
        toast(`${n} note${n === 1 ? "" : "s"} restored`, "ok");
        resolve(true);
      };
      fr.readAsText(file);
    });
  }
  // The host taps a reserved seat (or a seated player's "$" badge) and gets
  // the request right there: approve as asked, approve a DIFFERENT amount
  // ("asked $150 — seated with $80"), approve + trust, or decline.
  function openRequest(seatIdx) {
    const s = C().G.state;
    if (!s || !s.is_host) return;
    const seat = s.seats[seatIdx];
    const r = (s.requests || []).find((q) => (q.kind === "sit" ? q.seat === seatIdx : seat && !seat.empty && q.user_id === seat.user_id));
    if (!r) return toast("Nothing is waiting for you at that seat", "");
    openRequestDialog(r);
  }
  function openRequestDialog(r) {
    const s = C().G.state;
    const st = s.stakes, set = s.settings || {};
    const stackNow = r.kind === "rebuy" ? ((s.seats.find((x) => !x.empty && x.user_id === r.user_id) || {}).stack_cents || 0) : 0;
    // (the same window the player's own dialog uses — the two used to cap differently)
    const lim = buyinLimits(s, stackNow);
    const lo = Math.min(lim.lo, r.amount_cents), hi = Math.max(lim.hi, lim.capped ? 0 : r.amount_cents);
    const body = h("div", { class: "stack-14" });
    put(body, html`<div class="req-head">${avatar(r.name, r.name, "", r.avatar)}<div><b>${r.name}</b><small>${r.kind === "sit" ? `wants seat ${Number(r.seat) + 1} with ${d2(r.amount_cents)}` : `wants to add ${d2(r.amount_cents)} (stack ${d2(stackNow)})`}</small></div></div>`);
    const panel = h("div", { class: "stack-14" });
    body.appendChild(panel);
    const pick = moneyPanel(panel, {
      id: "rq-input", lo, hi, soft: !lim.capped, start: r.amount_cents, step: st.bb_cents, label: "Amount to approve",
      note: (c) => (c === r.amount_cents ? "as asked" : `asked ${d2(r.amount_cents)}`),
      hint: `Change the amount if you like — they are told what you approved.${set.max_buyin_cents ? ` Table maximum ${d2(set.max_buyin_cents)}.` : ""}`,
      presets: [{ label: "As asked", cents: r.amount_cents }, { label: d0(st.default_buyin_cents), cents: st.default_buyin_cents }, { label: "Half", cents: Math.round(r.amount_cents / 2) }, topPreset(lim)],
    });
    const resolve = async (action, trust) => {
      pick.sync();
      const cents = pick.cents;
      const body2 = { id: r.id, action, trust: !!trust };
      if (action === "approve" && cents !== r.amount_cents) body2.amount_cents = cents;
      await C().tablePost("request", body2);
      if (action === "approve") { toast(`${r.name} ${r.kind === "sit" ? "is seated" : "topped up"} with ${d2(cents)}${trust ? " — trusted from now on" : ""}`, "ok"); HG.sound && HG.sound.play("sit"); }
    };
    openModal({
      title: r.kind === "sit" ? "Buy-in request" : "Top-up request", body, autofocus: false,
      buttons: [
        { label: "Decline", cls: "ghost", onClick: () => resolve("deny", false) },
        { label: "Approve & trust", cls: "gold", onClick: () => resolve("approve", true) },
        { label: "Approve", cls: "primary", onClick: () => resolve("approve", false) },
      ],
    });
  }

  function openPlayer(seatIdx) {
    const s = C().G.state;
    const x = s && s.seats[seatIdx];
    if (!x || x.empty) return;
    if (s.is_host && x.request) return openRequestDialog(Object.assign({ user_id: x.user_id, name: x.name, avatar: x.avatar, seat: seatIdx }, x.request));
    const mine = x.user_id === s.my_user_id;
    const row = (s.ledger || []).find((r) => r.user_id === x.user_id) || {};
    // (by user id: two members can share a name — HGT-027; an older server sends names only)
    const st = ((U.hands && U.hands.stats) || []).find((r) => (r.user_id != null ? r.user_id === x.user_id : r.name === x.name)) || {};
    const note = noteFor(x.user_id);
    const body = h("div", { class: "stack-16" });
    put(body, html`<div class="pcard-head">${avatar(x.name, x.name, "", x.avatar)}<div><b>${x.name}${mine ? " (you)" : ""}</b><span class="muted">Seat ${seatIdx + 1}${x.is_host ? " · host" : ""}${x.sitting_out ? " · sitting out" : ""}</span></div></div>
      <div class="pcard-stats"><div><b>${d2(x.stack_cents)}</b><small>Stack</small></div><div><b class="${row.net_cents > 0 ? "pos" : row.net_cents < 0 ? "neg" : ""}">${row.net_cents > 0 ? "+" : ""}${d2(row.net_cents || 0)}</b><small>Net</small></div><div><b>${st.hands != null ? st.hands : "–"}</b><small>Hands</small></div><div><b>${st.wins != null ? st.wins : "–"}</b><small>Hands won</small></div></div>
      <button type="button" class="btn sm block" id="pc-all">${mine ? "All my hands & stats" : `All ${x.name}'s hands & stats`}</button>
      ${mine || s.is_host ? html`<button type="button" class="btn sm block" id="pc-receipt">${mine ? "My buy-ins & cash-outs" : "Their buy-ins & cash-outs"}</button>` : ""}
      ${mine ? "" : html`<div class="field"><span id="pc-tag-l">Colour tag</span><div class="tagrow" role="group" aria-labelledby="pc-tag-l">${Object.entries(TAGS).map(([k, name]) => html`<button type="button" data-tag="${k}" class="${note.tag === k ? "on" : ""}" title="${name}" aria-label="${name}" aria-pressed="${note.tag === k}"></button>`)}</div></div>
        <label class="field"><span>Private note</span><textarea class="input" id="pc-note" maxlength="500" placeholder="Only you can see this — it stays in this browser.">${note.text}</textarea></label>`}`);
    // your own automatic chips (the host's own fields sit in the Host box below)
    if (mine && s.status === "open" && seatIdx === s.my_seat && !(s.is_host && !autoChipsNow(s, x).choose)) {
      const auto = autoChipsRow(s, x);
      if (auto) body.appendChild(auto);
    }
    if (s.is_host && s.status === "open") {
      const hostBox = h("div", { class: "grp" });
      put(hostBox, html`<h4>Host</h4>${mine ? "" : html`<div class="setrow"><div><b>Trusted</b><small>Buys in, tops up and auto-tops-up without your approval</small></div><label class="switch"><input type="checkbox" id="pc-trust" ${x.trusted ? "checked" : ""}/><i></i></label></div>`}${s.auto_topup.mode === "host" ? html`<div class="row2"><label class="field"><span>Top up to</span>${moneyInput("pc-top-target", x.topup_target_cents || 0)}</label><label class="field"><span>When below</span>${moneyInput("pc-top-below", x.topup_below_cents || 0)}</label></div>` : ""}${s.auto_stack.mode === "host" ? html`<label class="field"><span>${mine ? "Your stack every hand (set by you)" : "Stack every hand (set by you)"}</span>${moneyInput("pc-set", x.auto_stack_cents || 0)}<small>0 = off for ${mine ? "you" : "this player"}</small></label>` : ""}${s.auto_topup.mode === "host" || s.auto_stack.mode === "host" ? html`<button type="button" class="btn sm" id="pc-save">Save automatic chips</button>` : ""}`);
      if (hostBox.children.length > 1) body.appendChild(hostBox);
      const tr = hostBox.querySelector("#pc-trust");
      if (tr) tr.addEventListener("change", (e) => C().tablePost("trust", { user_id: x.user_id, on: e.target.checked }).then(() => toast(e.target.checked ? `${x.name} is trusted` : `${x.name} needs approval again`, "ok")).catch(() => { e.target.checked = !e.target.checked; }));
      const sv = hostBox.querySelector("#pc-save");
      if (sv) sv.addEventListener("click", async () => {
        try {
          if (s.auto_topup.mode === "host") await C().tablePost("auto_topup", { players: [{ user_id: x.user_id, target_cents: C().toCents(hostBox.querySelector("#pc-top-target").value) || 0, below_cents: C().toCents(hostBox.querySelector("#pc-top-below").value) || 0 }] });
          if (s.auto_stack.mode === "host") await C().tablePost("auto_stack", { players: [{ user_id: x.user_id, cents: C().toCents(hostBox.querySelector("#pc-set").value) || 0 }] });
          toast("Saved", "ok");
        } catch (_) { /* toasted */ }
      });
    }
    const buttons = [];
    if (s.is_host && !mine && s.status === "open" && !x.pending_remove) {
      buttons.push({ label: x.sitting_out ? "Sit them in" : "Sit them out", cls: "", onClick: () => C().tablePost("sit_out_player", { user_id: x.user_id, on: !x.sitting_out }) });
      buttons.push({ label: "Remove", cls: "danger", onClick: async () => {
        const ok = await confirmDialog({ title: `Remove ${x.name}?`, text: "They are cashed out at the end of the current hand and their seat opens up.", okLabel: "Remove", danger: true });
        if (!ok) return false;
        await C().tablePost("kick", { user_id: x.user_id });
      } });
    }
    buttons.push({ label: "Done", cls: "primary" });
    let flushNote = () => {};
    if (!mine) {
      body.querySelector(".tagrow").addEventListener("click", (e) => {
        const b = e.target.closest("button[data-tag]");
        if (!b) return;
        body.querySelectorAll(".tagrow button").forEach((k) => { k.classList.toggle("on", k === b); k.setAttribute("aria-pressed", String(k === b)); });
        saveNote(x.user_id, { tag: b.dataset.tag });
      });
      // the note saves as you type (a moment after the last key) and when the card
      // closes — Escape closes it with no change event, and the note used to be lost
      const box = body.querySelector("#pc-note");
      let t = null, last = note.text;
      flushNote = () => { clearTimeout(t); const v = box.value.trim(); if (v !== last) { last = v; saveNote(x.user_id, { text: v }); } };
      box.addEventListener("input", () => { clearTimeout(t); t = setTimeout(flushNote, 300); });
      box.addEventListener("change", flushNote);
    }
    // the club's player page (their hands and numbers), opened from the table (HGH-006)
    body.querySelector("#pc-all").addEventListener("click", () => closeTopThen(() => UI.openMyHands("", { user_id: x.user_id, name: x.name, is_me: mine }, s.club && s.club.id, s.variant)));
    const rc = body.querySelector("#pc-receipt");
    if (rc) rc.addEventListener("click", () => closeTopThen(() => openReceipt(x.user_id)));
    openModal({ title: "Player", body, buttons, autofocus: false, onClose: () => flushNote() });
    if (!U.hands && canBrowse(s)) loadHands(s);
  }

  // ------------------------------------------------------------- receipts
  // FEAT-002: a player's money at this table, movement by movement — every buy-in,
  // top-up, automatic top-up, set-stack move, withdrawal and cash-out, with the hand it
  // came after and its time — and the totals the ledger shows. Your own; the host may
  // open anyone's (the ledger rows and the player card open it).
  const hhmm = (iso) => { const t = Date.parse(iso); if (!Number.isFinite(t)) return ""; const d = new Date(t); return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`; };
  function receiptBody(r) {
    const items = r.items || [];
    const when = (x) => [x.hand_no == null ? "" : x.hand_no ? `after hand #${x.hand_no}` : "before the first hand", hhmm(x.at)].filter(Boolean).join(" · ");
    const sign = (x) => (x.direction === "in" ? "+" : "−");
    const net = r.net_cents;
    return html`<div class="receipt">${items.length
      ? items.map((x) => html`<div class="rc-row ${x.direction}"><span class="rc-what"><b>${x.label}</b><small>${when(x)}</small></span><b class="rc-amt num">${sign(x)}${d2(x.cents)}</b></div>`)
      : html`<div class="muted">No money has moved yet.</div>`}</div>
      <div class="pcard-stats rc-tot"><div><b>${d2(r.in_cents)}</b><small>Money in</small></div><div><b>${d2(r.out_cents)}</b><small>Money out</small></div>
      <div><b>${d2(r.stack_cents)}</b><small>${r.seated ? "On the table" : "Left"}</small></div>
      <div><b class="${net > 0 ? "pos" : net < 0 ? "neg" : ""}">${net > 0 ? "+" : ""}${d2(net)}</b><small>Net</small></div></div>`;
  }
  function receiptText(r) {
    return `${r.table_name} — ${r.name}\n` + (r.items || []).map((x) => `${x.direction === "in" ? "+" : "-"}${d2(x.cents)}  ${x.label}${x.hand_no ? ` (after hand #${x.hand_no})` : ""}`).join("\n") +
      `\n\nIn ${d2(r.in_cents)} · Out ${d2(r.out_cents)} · ${r.seated ? "On the table" : "Left"} ${d2(r.stack_cents)} · Net ${r.net_cents >= 0 ? "+" : ""}${d2(r.net_cents)}`;
  }
  function openReceipt(userId) {
    const s = C().G.state;
    if (!s) return;
    const mine = !userId || userId === s.my_user_id;
    const row = (s.ledger || []).find((r) => r.user_id === (mine ? s.my_user_id : userId));
    const m = openModal({ title: mine ? "Your money here" : `${row ? row.name : "Player"}'s money`, sub: "Every buy-in and cash-out at this table.", body: loading(), buttons: [{ label: "Done", cls: "primary" }], autofocus: false });
    C().j(`/games/api/tables/${s.id}/ledger/${mine ? "me" : Number(userId)}`).then((r) => {
      if (m.closed) return;
      m.setBody(receiptBody(r));
      m.setButtons([
        { label: "Copy", cls: "ghost", onClick: async () => { try { await navigator.clipboard.writeText(receiptText(r)); toast("Copied", "ok"); } catch (_) { toast("Couldn't copy", "err"); } return false; } },
        { label: "Done", cls: "primary" },
      ]);
    }).catch((e) => { if (!m.closed) m.setBody((e && e.message) || "Couldn't load it — try again."); });  // (text)
  }

  function openLeave() {
    const s = C().G.state;
    if (!s || !Number.isInteger(s.my_seat)) return;
    const me = s.seats[s.my_seat];
    if (me.leaving) return C().tablePost("stay").catch(() => {});
    const busy = me.in_hand && (s.phase === "in_hand" || (s.runout && s.runout.blocking));
    if (!busy) {
      confirmDialog({ title: "Leave your seat?", text: `You cash out ${d2(me.stack_cents)} and your seat opens up.`, okLabel: "Leave seat", danger: true })
        .then((ok) => { if (ok) C().tablePost("leave").catch(() => {}); });
      return;
    }
    openModal({
      title: "Leave your seat?", sub: "You're in a hand. Play it out and leave when it ends — or leave right now (you are checked or folded for the rest of it).",
      body: "", autofocus: false,
      buttons: [
        { label: "Cancel", cls: "ghost" },
        { label: "Leave now (fold)", cls: "danger", onClick: () => C().tablePost("leave", { now: true }) },
        { label: "Leave after this hand", cls: "primary", onClick: () => C().tablePost("leave").then(() => toast("You leave when this hand ends", "ok")) },
      ],
    });
  }

  function seatMenu(anchor) {
    const s = C().G.state;
    if (!s) return;
    openMenu(anchor, seatItems(s).concat([{ icon: "i-back", label: "Back to lobby", onClick: () => C().showLobby(false) }]));
  }
  // What you can do with your seat — the person menu's items, and on a phone a section of
  // the ≡ menu (HGT-003: the top bar keeps room for the table's name there).
  function seatItems(s) {
    const seated = Number.isInteger(s.my_seat);
    const me = seated ? s.seats[s.my_seat] : null;
    const items = [{ header: seated ? `${me.name} · ${d2(me.stack_cents)}` : "Not seated" }];
    if (seated && s.status === "open") {
      items.push({ icon: "i-pluscircle", label: s.needs_approval ? "Request chips" : (s.settings && s.settings.allow_rathole) ? "Add or take off chips" : "Add chips", onClick: () => openTopUp() });
      if (s.auto_stack.mode !== "off" || s.auto_topup.mode !== "off") items.push({ icon: "i-wallet", label: "Automatic chips…", onClick: openAutoChips });
      // (the site's owner only: the view carries `bot` for them alone)
      if (s.bot) items.push({ icon: "i-cpu", label: me.bot === "auto" ? "The network plays your seat…" : me.bot === "assist" ? "The network suggests your moves…" : "The network…", onClick: openBot });
      if (me.sitting_out) items.push({ icon: "i-play", label: "I'm back", onClick: () => C().tablePost("sit_out", { on: false }).catch(() => {}) });
      else {
        items.push({ icon: "i-coffee", label: me.sit_out_next ? "Cancel sit-out" : "Sit out next hand", onClick: () => C().tablePost("sit_out", me.sit_out_next ? { on: false } : { on: true, next_hand: true }).catch(() => {}) });
        items.push({ icon: "i-clock", label: "Step away now", hint: "You are checked or folded until you come back", onClick: () => C().tablePost("sit_out", { on: true }).catch(() => {}) });
      }
      items.push("-");
      if (me.leaving) items.push({ icon: "i-play", label: "Stay — cancel leaving", onClick: () => C().tablePost("stay").catch(() => {}) });
      else items.push({
        icon: "i-door", label: me.pending_remove ? "Leaving after this hand" : "Leave seat", danger: true, disabled: !!me.pending_remove,
        onClick: openLeave,
      });
    } else if (s.status === "open") items.push({ icon: "i-user", label: "Pick an open seat on the table", disabled: true, onClick: () => {} });
    return items;
  }

  Object.assign(UI, { openSit, openMove, pendingMove, openTopUp, openAutoChips, openPlayer, openRequest, openRequestDialog, openLeave, openBot, seatMenu, seatItems, noteFor, TAGS, exportNotes, importNotes, openReceipt });
})();
