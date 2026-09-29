"use strict";
// Home games — the table's settings: the host's Manage drawer (settings, chips, pace,
// players, controls) and everyone's Table info card. Part of the UI split out of
// games.ui.js (FE-005).
//
// Every control in the drawer saves on its own (2026-09-28, HGT-015): switches and
// pickers when flipped, a box when you leave it (a "Saved" tick says so). The Settings
// tab used to need a "Save changes" button while the other tabs saved at once — and
// switching tab or tapping outside silently threw its edits away.
(function () {
  const HG = globalThis.HG;
  const UI = HG.ui;
  const { $, C, html, put, icon, U, h, avatar, moneyInput, d2, gameOf, burnRules, OPTIONS, toast, openModal, confirmDialog,
    segHtml, segWire, savedFlash, afterTransition } = HG.uikit;

  // --------------------------------------------------------- manage drawer
  // (internal tab keys stay "game" / "table": the top bar opens "chips" by name)
  const TABS = [["game", "Settings"], ["chips", "Chips"], ["pace", "Pace"], ["players", "Players"], ["table", "Controls"]];

  function openDrawer(tab) {
    const s = C().G.state;
    if (!s) return;
    if (tab) U.drawerTab = tab;
    if (U.drawer) { paintDrawer(s, true); return; }
    const root = $("drawer-root");
    const layer = h("div", { class: "layer" });
    const scrim = h("div", { class: "scrim" });
    const dr = h("aside", { class: "drawer", role: "dialog", "aria-modal": "true", "aria-label": "Manage table" });
    layer.appendChild(scrim); layer.appendChild(dr);
    root.appendChild(layer);
    scrim.addEventListener("click", closeDrawer);
    U.drawer = { layer, dr, sig: "", opener: document.activeElement };
    paintDrawer(s, true);
    requestAnimationFrame(() => layer.classList.add("open"));
    const first = dr.querySelector(".dr-tabs button.on");
    if (first) setTimeout(() => first.focus({ preventScroll: true }), 60);
  }
  function closeDrawer() {
    const d = U.drawer;
    if (!d) return;
    // a box still being typed in saves as it loses focus (its change event) — before it goes
    const a = document.activeElement;
    if (a && d.dr.contains(a) && a.blur) a.blur();
    U.drawer = null;
    d.layer.classList.remove("open");
    afterTransition(d.dr, () => d.layer.remove());
    if (d.opener && d.opener.isConnected && d.opener.focus) d.opener.focus({ preventScroll: true });
  }

  // What a tab's form is built from. It is rebuilt only when this changes (or on open /
  // a tab change / a failed save) — never under the host's pointer: stacks and the actor
  // change with every bet, so they are updated in place instead (liveBits; HGT-009).
  function drawerSig(s) {
    if (U.drawerTab === "players") {
      return JSON.stringify(s.seats.map((x) => [x.user_id, x.name, x.sitting_out, x.pending_remove, x.auto_stack_cents, x.trusted, x.present, x.is_host]))
        + (s.spectators || []).join(",") + s.auto_stack.mode;
    }
    if (U.drawerTab === "chips") return JSON.stringify([s.requests, s.settings.approve_buyins, s.auto_stack, s.auto_topup]);
    if (U.drawerTab === "table") return `${s.running}:${s.phase}:${s.can_deal}:${s.runout.blocking}:${s.eligible_count < 2}`;
    return "form";
  }
  function liveBits(s, root) {
    root.querySelectorAll("[data-stk]").forEach((x) => {
      const seat = s.seats.find((y) => !y.empty && String(y.user_id) === x.dataset.stk);
      if (seat) { const t = d2(seat.stack_cents); if (x.textContent !== t) x.textContent = t; }
    });
    const fold = root.querySelector("#m-hostfold");
    if (fold) {
      const actor = s.phase === "in_hand" && s.actor != null ? s.seats[s.actor] : null;
      fold.disabled = !actor;
      const t = actor ? `${actor.name} is up. Checks instead when checking is free.` : "Nobody is on the clock.";
      const who = root.querySelector("#m-fold-who");
      if (who && who.textContent !== t) who.textContent = t;
    }
  }

  function paintDrawer(s, force) {
    const d = U.drawer;
    if (!d) return;
    if (!s.is_host) { closeDrawer(); return; }
    const sig = drawerSig(s);
    if (!force && sig === d.sig) { liveBits(s, d.dr); return; }
    d.sig = sig;
    const set = s.settings, st = s.stakes, G = gameOf(s.variant);
    const nReq = (s.requests || []).length;
    const sw = (id, on) => html`<label class="switch"><input type="checkbox" id="${id}" ${on ? "checked" : ""}/><i></i></label>`;
    let tab;
    if (U.drawerTab === "game") {
      const maxSeats = (s.game && s.game.max_seats) || G.maxSeats;
      tab = html`<div class="grp"><h4>Table</h4><label class="field"><span>Name</span><input type="text" class="input" id="m-name" maxlength="60" value="${s.name}"/></label>
        <div class="field"><span>Game</span><input type="text" class="input" disabled value="${G.name}"/><small>Fixed once a table is created — host another table for another game</small></div>
        <div class="row2"><label class="field"><span>Ante</span>${moneyInput("m-ante", st.ante_cents)}<small>Applies from the next hand</small></label>
        <div class="field"><span>Big blind (chip unit)</span><input type="text" class="input num" disabled value="${d2(st.bb_cents)}"/><small>Fixed once a table is created</small></div></div>
        <div class="field"><span>Seats</span>${segHtml("m-seats", OPTIONS.seats(maxSeats), s.num_seats)}<small>Between hands only — the higher seats must be empty to shrink${maxSeats < 8 ? ` · ${G.label} seats up to ${maxSeats} (one deck)` : ""}</small></div></div>
        <div class="grp"><h4>Buy-ins</h4><div class="row3"><label class="field"><span>Minimum</span>${moneyInput("m-min", set.min_buyin_cents)}</label><label class="field"><span>Default</span>${moneyInput("m-dflt", st.default_buyin_cents)}</label><label class="field"><span>Maximum</span>${moneyInput("m-max", set.max_buyin_cents)}</label></div><p>0 = no limit. The maximum also caps top-ups.</p></div>
        <div class="grp"><h4>Privacy &amp; extras</h4><div class="setrow"><div><b>Show in the club lobby</b><small>Off = link only: only club members with the link find it</small></div>${sw("m-listed", set.listed)}</div>
        ${G.graded ? html`<div class="setrow"><div><b>Accuracy marks on shown hands</b><small>Everyone sees the network's marks on their own decisions, and on hands tabled at showdown — never on a mucked hand. Off = only their own</small></div>${sw("m-grades", set.show_grades)}</div>` : ""}
        <div class="setrow"><div><b>Rabbit hunting</b><small>Let players peek at the undealt streets after a fold-out</small></div>${sw("m-rabbit", set.allow_rabbit)}</div>
        <div class="setrow"><div><b>Players may take chips off the table</b><small>Ratholing allowed: anyone can pocket part of their stack between hands (a player always keeps an ante + 1 bb)</small></div>${sw("m-rathole", set.allow_rathole)}</div></div>
        <p class="dr-note">Changes save as you make them.</p>`;
      // (a game the network doesn't grade has no marks to show)
    } else if (U.drawerTab === "chips") {
      const modeSeg = (id, cur) => segHtml(id, [["off", "Off"], ["host", "Host sets"], ["player", "Players choose"]], cur, null, "as-modes");
      const pick = html`<small class="muted">Each player picks their own: they tap their seat (or Add chips) › Automatic chips.</small>`;
      tab = html`<div class="grp"><h4>Buy-in approval</h4><div class="setrow"><div><b>I approve every buy-in</b><small>Sit-downs and top-ups wait for your OK. Players you trust never wait.</small></div>${sw("m-approve", set.approve_buyins)}</div>
        ${nReq ? (s.requests || []).map((r) =>
          html`<div class="prow req" data-req="${r.id}">${avatar(r.name, r.name, "", r.avatar)}<div class="who"><b>${r.name}</b><small>${r.kind === "sit" ? `wants seat ${Number(r.seat) + 1} with ${d2(r.amount_cents)}` : `wants to add ${d2(r.amount_cents)}`}</small></div><div class="acts"><button type="button" class="btn sm primary" data-ok="${r.id}">Approve</button><button type="button" class="btn sm gold" data-okt="${r.id}" title="Approve, and never ask again for this player">+ Trust</button><button type="button" class="btn sm" data-edit="${r.id}" title="Change the amount, then approve">Edit</button><button type="button" class="icon-btn danger-ico" data-no="${r.id}" title="Decline" aria-label="Decline ${r.name}">${icon("i-x")}</button></div></div>`)
          : (set.approve_buyins ? html`<p>No one is waiting. Trust regulars from the Players tab so the game never stops for them.</p>` : "")}</div>
        <div class="grp"><h4>Auto top-up</h4><p>When a stack drops below a threshold it is topped back up before the next hand. Winnings stay on the table — no ratholing.</p>${modeSeg("m-top", s.auto_topup.mode)}
        ${s.auto_topup.mode === "player" ? pick : ""}
        ${s.auto_topup.mode === "host" ? html`<div class="row2"><label class="field"><span>Top up to</span>${moneyInput("m-top-target", s.auto_topup.all_target_cents || st.default_buyin_cents)}</label><label class="field"><span>When below</span>${moneyInput("m-top-below", s.auto_topup.all_below_cents || s.auto_topup.all_target_cents || st.default_buyin_cents)}</label></div><button type="button" class="btn sm" id="m-top-apply">Apply to everyone</button><small class="muted">Per-player amounts: tap a player's seat.</small>` : ""}</div>
        <div class="grp"><h4>Set stack every hand</h4><p>Every stack is reset to one amount before EVERY deal — short stacks top up, big stacks bank the difference. For high-action games where ratholing is fine.</p>${modeSeg("m-auto", s.auto_stack.mode)}
        ${s.auto_stack.mode === "player" ? pick : ""}
        ${s.auto_stack.mode === "host" ? html`<div class="sz-row"><label class="field grow"><span>Stack for everyone</span>${moneyInput("m-auto-all", s.auto_stack.all_cents || st.default_buyin_cents)}</label><button type="button" class="btn sm field-btn" id="m-auto-apply">Apply</button></div><small class="muted">Per-player amounts: tap a player's seat. Set-stack wins when a player has both.</small>` : ""}</div>
        ${set.approve_buyins ? html`<small class="muted">While you approve buy-ins, automatic chips only run for you and the players you trust.</small>` : ""}`;
    } else if (U.drawerTab === "pace") {
      tab = html`<div class="grp"><h4>Shot clock</h4><div class="field"><span>Decision time</span>${segHtml("m-clock", OPTIONS.clock, s.decision_secs)}<small>When it runs out the player checks if that is free, otherwise folds.</small></div>
        <div class="field"><span>Time bank per player</span>${segHtml("m-bank", OPTIONS.bank, set.time_bank_secs)}<small>Burned automatically after the base clock; a couple of seconds come back every hand.</small></div></div>
        <div class="grp"><h4>Dealing</h4><div class="field"><span>Next hand after</span>${segHtml("m-deal", OPTIONS.deal, set.deal_delay_secs)}<small>The server deals — the game keeps running even if you switch tabs.</small></div>
        <div class="field"><span>All-in runout, per street</span>${segHtml("m-pause", OPTIONS.pause, s.street_pause_secs)}</div></div>`;
    } else if (U.drawerTab === "players") {
      const seated = s.seats.filter((x) => !x.empty);
      tab = html`<div class="grp"><h4>${seated.length} seated</h4>${seated.map((x) =>
        html`<div class="prow" data-uid="${x.user_id}">${avatar(x.name, x.name, "", x.avatar)}<div class="who"><b>${x.name}${x.is_host ? html` <span class="pill host pill-xs">Host</span>` : ""}</b><small><span data-stk="${x.user_id}">${d2(x.stack_cents)}</span>${x.sitting_out ? " · sitting out" : ""}${x.pending_remove ? " · leaving" : ""}</small></div><div class="acts">${x.user_id !== s.my_user_id ? html`<button type="button" class="btn sm ${x.trusted ? "gold" : ""}" data-trust="${x.user_id}" data-on="${x.trusted ? 0 : 1}" aria-pressed="${!!x.trusted}" title="${x.trusted ? "Trusted: buys in without asking. Click to stop trusting." : "Trust: let them buy in and top up without your approval"}">${icon("i-check", "sm")}${x.trusted ? "Trusted" : "Trust"}</button>` : ""}${x.pending_remove ? "" : html`<button type="button" class="btn sm" data-away="${x.user_id}" data-on="${x.sitting_out ? 0 : 1}">${x.sitting_out ? (x.user_id === s.my_user_id ? "I'm back" : "Sit in") : "Sit out"}</button>`}${x.user_id !== s.my_user_id && !x.pending_remove ? html`<button type="button" class="icon-btn" data-host="${x.user_id}" title="Make host" aria-label="Make ${x.name} the host">${icon("i-swap")}</button><button type="button" class="icon-btn danger-ico" data-kick="${x.user_id}" title="Remove from table" aria-label="Remove ${x.name} from the table">${icon("i-x")}</button>` : ""}</div></div>`)}</div>
        ${(s.spectators || []).length ? html`<div class="grp"><h4>${s.spectators.length} watching</h4><p class="flush">${s.spectators.join(", ")}</p></div>` : ""}
        <small class="muted">Tap a player's seat for their card: trust, per-player automatic chips, notes.</small>`;
    } else {
      const busy = s.phase === "in_hand" || s.runout.blocking;
      // (the Close button can't close a table mid-hand: the line says when it can)
      tab = html`<div class="grp"><h4>Game</h4><div class="setrow"><div><b>${s.running ? "Game is running" : "Game is paused"}</b><small>${s.running ? "Pausing lets the current hand finish first." : s.eligible_count < 2 ? "Needs two players with more than the ante." : "Everyone is waiting on you."}</small></div><button type="button" class="btn ${s.running ? "" : "primary"}" id="m-run" ${!s.running && s.eligible_count < 2 ? "disabled" : ""}>${icon(s.running ? "i-pause" : "i-play", "sm")}${s.running ? (busy ? "Pause after hand" : "Pause") : "Start game"}</button></div>
        <div class="setrow"><div><b>Deal now</b><small>Skip the wait between hands</small></div><button type="button" class="btn" id="m-deal" ${s.can_deal ? "" : "disabled"}>${icon("i-bolt", "sm")}Deal</button></div></div>
        <div class="grp danger"><h4>Careful</h4><div class="setrow"><div><b>Fold the player on the clock</b><small id="m-fold-who"></small></div><button type="button" class="btn danger" id="m-hostfold" disabled>Fold</button></div>
        <div class="setrow"><div><b>Close the table</b><small>Cashes everyone out and ends the session.${busy ? " Available between hands." : ""}</small></div><button type="button" class="btn danger" id="m-close" ${busy ? "disabled" : ""}>Close table</button></div></div>`;
    }
    const focusedTab = document.activeElement && document.activeElement.closest && document.activeElement.closest(".dr-tabs") ? U.drawerTab : null;
    put(d.dr, html`<div class="dr-head"><h3>${icon("i-crown")}Manage table</h3><button class="icon-btn" id="dr-x" type="button" aria-label="Close">${icon("i-x")}</button></div>
      <div class="dr-tabs" role="tablist">${TABS.map(([k, l]) => html`<button type="button" role="tab" id="dr-t-${k}" aria-selected="${k === U.drawerTab}" data-t="${k}" class="${k === U.drawerTab ? "on" : ""}">${l}${k === "chips" && nReq ? ` (${nReq})` : ""}</button>`)}</div>
      <div class="dr-body" role="tabpanel" aria-labelledby="dr-t-${U.drawerTab}">${tab}</div>`);
    wireDrawer(s, d.dr);
    liveBits(s, d.dr);
    if (focusedTab) { const b = d.dr.querySelector(`#dr-t-${focusedTab}`); if (b) b.focus({ preventScroll: true }); }
  }

  function wireDrawer(s, root) {
    const q = (id) => root.querySelector("#" + id);
    // one saver for every control: a "Saved" tick on success; on a refusal (the server
    // toasts why) the tab is rebuilt from the table's real values
    const saveSet = async (patch, node) => {
      try { await C().tablePost("settings", patch); if (node) savedFlash(node); return true; }
      catch (_) { const cur = C().G.state; if (cur) paintDrawer(cur, true); return false; }
    };
    q("dr-x").addEventListener("click", closeDrawer);
    const tabs = [...root.querySelectorAll(".dr-tabs button")];
    tabs.forEach((b, i) => {
      b.addEventListener("click", () => { U.drawerTab = b.dataset.t; paintDrawer(C().G.state, true); });
      b.addEventListener("keydown", (e) => {  // (a tab list: ← → move between tabs)
        const k = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
        if (!k) return;
        e.preventDefault();
        const nb = tabs[(i + k + tabs.length) % tabs.length];
        U.drawerTab = nb.dataset.t;
        paintDrawer(C().G.state, true);
        const f = root.querySelector(`#dr-t-${U.drawerTab}`);
        if (f) f.focus({ preventScroll: true });
      });
    });
    segWire(root);
    if (U.drawerTab === "game") {
      const onChange = (id, patch) => { const el = q(id); if (el) el.addEventListener("change", () => saveSet(patch(el), el)); };
      onChange("m-name", (el) => ({ name: el.value.trim() || s.name }));
      onChange("m-ante", (el) => ({ ante_cents: C().toCents(el.value) }));
      // (the three buy-in amounts are checked together: they go together)
      const buyins = () => ({ min_buyin_cents: C().toCents(q("m-min").value) || 0, default_buyin_cents: C().toCents(q("m-dflt").value), max_buyin_cents: C().toCents(q("m-max").value) || 0 });
      ["m-min", "m-dflt", "m-max"].forEach((id) => onChange(id, buyins));
      onChange("m-listed", (el) => ({ listed: el.checked }));
      onChange("m-grades", (el) => ({ show_grades: el.checked }));
      onChange("m-rabbit", (el) => ({ allow_rabbit: el.checked }));
      onChange("m-rathole", (el) => ({ allow_rathole: el.checked }));
      root.querySelectorAll("#m-name, .money input").forEach((el) => el.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); el.blur(); } }));
      q("m-seats").addEventListener("pick", (e) => { const n = Number(e.detail); if (n !== s.num_seats) saveSet({ num_seats: n }, q("m-seats")); });
    } else if (U.drawerTab === "chips") {
      q("m-approve").addEventListener("change", (e) => saveSet({ approve_buyins: e.target.checked }, e.target));
      const resolve = (id, action, trust) => C().tablePost("request", { id: Number(id), action, trust: !!trust }).catch(() => {});
      root.querySelectorAll("[data-ok]").forEach((b) => b.addEventListener("click", () => resolve(b.dataset.ok, "approve", false)));
      root.querySelectorAll("[data-okt]").forEach((b) => b.addEventListener("click", () => resolve(b.dataset.okt, "approve", true)));
      root.querySelectorAll("[data-no]").forEach((b) => b.addEventListener("click", () => resolve(b.dataset.no, "deny", false)));
      root.querySelectorAll("[data-edit]").forEach((b) => b.addEventListener("click", () => { const r = (s.requests || []).find((x) => String(x.id) === b.dataset.edit); if (r) UI.openRequestDialog(r); }));
      q("m-top").addEventListener("pick", async (e) => { try { await C().tablePost("auto_topup", { mode: e.detail }); } catch (_) { /* toasted */ } paintDrawer(C().G.state, true); });
      q("m-auto").addEventListener("pick", async (e) => { try { await C().tablePost("auto_stack", { mode: e.detail }); } catch (_) { /* toasted */ } paintDrawer(C().G.state, true); });
      const ta = q("m-top-apply");
      if (ta) ta.addEventListener("click", async () => { try { await C().tablePost("auto_topup", { all_target_cents: C().toCents(q("m-top-target").value) || 0, all_below_cents: C().toCents(q("m-top-below").value) || 0 }); toast("Auto top-up set for everyone", "ok"); } catch (_) { /* toasted */ } });
      const ap = q("m-auto-apply");
      if (ap) ap.addEventListener("click", async () => { try { await C().tablePost("auto_stack", { all_cents: C().toCents(q("m-auto-all").value) || 0 }); toast("Everyone resets to that stack each hand", "ok"); } catch (_) { /* toasted */ } });
    } else if (U.drawerTab === "pace") {
      q("m-clock").addEventListener("pick", (e) => saveSet({ decision_secs: Number(e.detail) }, q("m-clock")));
      q("m-bank").addEventListener("pick", (e) => saveSet({ time_bank_secs: Number(e.detail) }, q("m-bank")));
      q("m-deal").addEventListener("pick", (e) => saveSet({ deal_delay_secs: Number(e.detail) }, q("m-deal")));
      q("m-pause").addEventListener("pick", (e) => C().tablePost("street_pause", { secs: Number(e.detail) }).then(() => savedFlash(q("m-pause"))).catch(() => {}));
    } else if (U.drawerTab === "players") {
      root.querySelectorAll("[data-trust]").forEach((b) => b.addEventListener("click", () => C().tablePost("trust", { user_id: Number(b.dataset.trust), on: b.dataset.on === "1" }).catch(() => {})));
      root.querySelectorAll("[data-away]").forEach((b) => b.addEventListener("click", () => {
        const uid = Number(b.dataset.away), on = b.dataset.on === "1";
        (uid === s.my_user_id ? C().tablePost("sit_out", { on }) : C().tablePost("sit_out_player", { user_id: uid, on })).catch(() => {});
      }));
      root.querySelectorAll("[data-kick]").forEach((b) => b.addEventListener("click", async () => {
        const who = s.seats.find((x) => x.user_id === Number(b.dataset.kick));
        const ok = await confirmDialog({ title: `Remove ${who ? who.name : "player"}?`, text: "They are cashed out at the end of the current hand and their seat opens up. They can sit back down later.", okLabel: "Remove", danger: true });
        if (ok) C().tablePost("kick", { user_id: Number(b.dataset.kick) }).catch(() => {});
      }));
      root.querySelectorAll("[data-host]").forEach((b) => b.addEventListener("click", async () => {
        const who = s.seats.find((x) => x.user_id === Number(b.dataset.host));
        const ok = await confirmDialog({ title: `Make ${who ? who.name : "them"} the host?`, text: "They get the Manage button; you keep your seat.", okLabel: "Hand over" });
        if (ok) C().tablePost("transfer_host", { user_id: Number(b.dataset.host) }).then(() => closeDrawer()).catch(() => {});
      }));
    } else {
      q("m-run").addEventListener("click", () => C().tablePost("run", { running: !s.running }).catch(() => {}));
      q("m-deal").addEventListener("click", () => C().deal(C().G.state));
      q("m-hostfold").addEventListener("click", async () => {
        const cur = C().G.state, a = cur && cur.actor != null ? cur.seats[cur.actor] : null;
        if (!a) return;
        const ok = await confirmDialog({ title: `Fold ${a.name}?`, text: a.user_id === cur.my_user_id ? "That's you — this folds your own hand." : "Use this when someone is away and the table is waiting. It can't be undone.", okLabel: "Fold them", danger: true });
        if (ok) C().tablePost("host_fold").catch(() => {});
      });
      q("m-close").addEventListener("click", async () => {
        const cur = C().G.state;
        const lines = (cur.ledger || []).map((r) => html`<div class="settle-row"><span>${r.name}</span><b class="${r.net_cents >= 0 ? "pos" : "neg"}">${r.net_cents >= 0 ? "+" : ""}${d2(r.net_cents)}</b></div>`);
        const ok = await confirmDialog({ title: "Close this table?", text: "Everyone is cashed out and the session ends. The final ledger stays available from the lobby.", body: html`<div>${lines}</div>`, okLabel: "Close table", danger: true });
        if (ok) { try { await C().tablePost("close"); closeDrawer(); } catch (_) { /* toasted */ } }
      });
    }
  }

  function openInfo() {
    const s = C().G.state;
    if (!s) return;
    const set = s.settings, st = s.stakes, G = gameOf(s.variant);
    const row = (k, v) => html`<div class="settle-row"><span class="muted">${k}</span><b class="tx">${v}</b></div>`;
    openModal({
      title: s.name, sub: G.name, autofocus: false,
      body: html`<div>${row("Big blind (chip unit)", d2(st.bb_cents))}${row("Ante", `${d2(st.ante_cents)} (${(st.ante_cents / st.bb_cents).toFixed(st.ante_cents % st.bb_cents ? 1 : 0)} bb)`)}${row("Buy-in", set.min_buyin_cents || set.max_buyin_cents ? `${set.min_buyin_cents ? d2(set.min_buyin_cents) : "any"} – ${set.max_buyin_cents ? d2(set.max_buyin_cents) : "any"}` : "No limits")}${row("Seats", s.num_seats)}${row("Decision time", s.decision_secs ? s.decision_secs + "s" : "No clock")}${row("Time bank", set.time_bank_secs ? set.time_bank_secs + "s per player" : "Off")}${row("Next hand", set.deal_delay_secs ? `dealt automatically after ${set.deal_delay_secs}s` : "dealt manually")}${row("Rabbit hunt", set.allow_rabbit ? "Allowed" : "Off")}${row("Club lobby", set.listed ? "Shown" : "Link only")}</div>
        <div class="grp"><h4>How a hand works</h4>${guideSteps(s)}</div>`,
      buttons: [{ label: "Copy invite link", cls: "", onClick: () => { UI.copyInvite(s.id); return false; } }, { label: "Done", cls: "primary" }],
    });
  }

  // ------------------------------------------------------ how a hand works
  // FEAT-009: a friend opening their first link never saw the rules (they lived behind
  // the ⓘ). The same four steps are the Table info card's, the guide's that opens by
  // itself the first time someone who isn't seated looks at a table of a game (once per
  // browser and game), and the "How it works" links in the welcome banner and the
  // buy-in dialog.
  function guideSteps(s) {
    const G = gameOf(s.variant), ante = d2(s.stakes.ante_cents);
    const cards = G.burns ? html`You start with <b>${G.dealt} cards</b>, and the hand starts on the flop with <b>two boards</b>. ${burnRules()}`
      : html`You get <b>${G.hole} cards</b>, and the hand starts on the flop with <b>two boards</b>, side by side.`;
    const steps = [
      [html`Everyone antes, no blinds`, html`Every player dealt in puts in the ante (${ante}) — that is the pot the hand starts with.`],
      [html`${G.burns ? `${G.dealt}+ cards` : `${G.hole} cards`}, two boards`, cards],
      [html`Pot-limit betting`, html`The most you can bet is the size of the pot. Bet or Raise shows the sizes.`],
      [html`Each board wins half`, html`At the showdown the best hand on each board takes half the pot — always exactly <b>two</b> of your cards and <b>three</b> from that board. Win both to scoop.`],
    ];
    return html`<ol class="guide">${steps.map(([t, d]) => html`<li><b>${t}</b><span>${d}</span></li>`)}</ol>${G.graded ? "" : html`<p class="rules">${G.label} decisions are not graded — there is no ${G.label} network yet.</p>`}`;
  }
  const GUIDE_KEY = "hg.guide.v1";
  const guideSeen = () => { try { return JSON.parse(localStorage.getItem(GUIDE_KEY) || "{}") || {}; } catch (_) { return {}; } };
  function markGuide(variant) {
    const all = guideSeen();
    all[variant || "plo5"] = 1;
    try { localStorage.setItem(GUIDE_KEY, JSON.stringify(all)); } catch (_) { /* private mode: every visit */ }
  }
  function openGuide() {
    const s = C().G.state;
    if (!s) return;
    const G = gameOf(s.variant);
    markGuide(s.variant);
    openModal({
      title: "How a hand works", sub: `${G.name} — the whole game in four steps.`, autofocus: false,
      body: guideSteps(s), buttons: [{ label: "Got it", cls: "primary" }],
    });
  }
  // (games.ui.js render): the first look at a game's table, when nothing else is going on
  function maybeGuide(s) {
    if (!s || s.status !== "open" || Number.isInteger(s.my_seat) || U.modals.length || U.drawer) return;
    const v = s.variant || "plo5", shown = (U.guideShown = U.guideShown || {});
    if (guideSeen()[v] || shown[v]) return;
    shown[v] = true;  // (once per game and page load even where the browser keeps nothing)
    openGuide();
  }

  Object.assign(UI, { openDrawer, closeDrawer, paintDrawer, openInfo, openGuide, maybeGuide });
})();
