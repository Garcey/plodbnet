"use strict";
// Home games — the lobby: the club's tables and sessions, clubs (switch, start, join,
// invite, members), hosting a table, and the club's numbers (podium, players,
// head to head, all sessions). Part of the UI split out of games.ui.js (FE-005).
(function () {
  const HG = globalThis.HG;
  const UI = HG.ui;
  const { $, C, html, put, icon, U, h, avatar, moneyInput, d2, GAMES, gameOf, gameNote, setGames, OPTIONS, secsLabel, toast, openModal, confirmDialog,
    closeTopThen, loading, openMenu, segHtml, segWire, segVal, setMyAvatar, fmtWhen } = HG.uikit;

  // --------------------------------------- join requests (club owner / admins)
  // Someone asked to join the club (from a table link, or the invite link of an
  // ask-first club). A small card, never a modal — it must not get in the way of
  // a hand.
  function renderJoinReqs(list) {
    let el = $("joinreq");
    if (!list || !list.length) {
      if (el && !el.hidden) { el.hidden = true; el.dataset.sig = ""; }
      document.body.classList.remove("has-joinreq");
      return;
    }
    if (!el) {
      el = h("div", { id: "joinreq", class: "joinreq", role: "status" });
      document.body.appendChild(el);
      el.addEventListener("click", async (e) => {
        const b = e.target.closest("button[data-j]");
        const r = U.joinReq;
        if (!b || b.disabled || !r) return;
        el.querySelectorAll("button").forEach((x) => { x.disabled = true; });
        const allow = b.dataset.j === "yes";
        try {
          await C().j(`/games/api/clubs/${encodeURIComponent(r.club_id)}/requests/decide`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ user_id: r.user_id, allow }) });
          toast(allow ? `${r.name || "They"} joined ${r.club_name || "the club"}` : "Request declined", allow ? "ok" : "");
        } catch (err) { toast(err.message, "err"); }
        el.dataset.sig = "";  // the next state (or lobby poll) brings the rest
        el.hidden = true;
        document.body.classList.remove("has-joinreq");
        if (C().G.gameId) C().refreshNow().catch(() => {}); else C().loadLobby().catch(() => {});
      });
    }
    const r = list[0];
    U.joinReq = r;
    const sig = list.map((x) => `${x.club_id}:${x.user_id}`).join(",");
    if (el.dataset.sig !== sig) {
      el.dataset.sig = sig;
      put(el, html`<div class="jr-txt"><b>${r.name || "Someone"}</b> wants to join ${r.club_name || "the club"}<small>${r.email}${list.length > 1 ? ` · +${list.length - 1} more` : ""}</small></div>
        <div class="jr-btns"><button class="btn sm ghost" type="button" data-j="no">Not now</button><button class="btn sm gold" type="button" data-j="yes">Let in</button></div>`);
    }
    el.hidden = false;
    document.body.classList.add("has-joinreq");
  }

  // ------------------------------------------------------------------- lobby
  // (each seat's place rides in data-vars — --x / --y, games.css places it)
  function miniFelt(t) {
    const n = t.num_seats, by = {};
    (t.players || []).forEach((p) => (by[p.seat] = p));
    const seats = [];
    for (let i = 0; i < n; i++) {
      const th = Math.PI / 2 + (i * 2 * Math.PI) / n;
      const x = (50 + 46 * Math.cos(th)).toFixed(2), y = (50 + 40 * Math.sin(th)).toFixed(2);
      const p = by[i];
      seats.push(p
        ? html`<span class="av ${p.is_me ? "me" : ""}" data-vars="x:${x}%;y:${y}%;h:${HG.avatar.hueOf(p.name)}" title="${p.name}">${HG.avatar.initials(p.name)}</span>`
        : html`<span class="slot" data-vars="x:${x}%;y:${y}%"></span>`);
    }
    return html`<div class="tcard-felt">${seats}<div class="mid"><small>Ante</small>${d2(t.ante_cents)}</div></div>`;
  }
  // One table in the lobby. The whole card opens it (HGL-004: it lifted on hover but only
  // its button answered); the copy-link button is its own. "Hand #N" sits in the header
  // beside Live / Paused (HGT-006: four pills wrapped on a phone).
  function tableCard(t) {
    const full = t.seated >= t.num_seats;
    const cta = t.is_seated ? "Return to table" : full ? "Watch" : "Join table";
    const card = h("div", { class: "tcard" });
    // (your seat at a table of ANOTHER club: say which club it is in)
    const other = t.club_id && t.club_id !== C().G.clubId ? ` · ${t.club_name || "another club"}` : "";
    put(card, html`<div class="tcard-top"><div class="tcard-id"><div class="tcard-name">${t.name}</div>
      <div class="tcard-host">Hosted by ${t.host_name}${t.is_host ? " (you)" : ""}${other}</div></div>
      <span class="tcard-state"><span class="pill num tcard-hand"${t.hand_no ? "" : html` hidden`}>#${Number(t.hand_no) || 0}</span><span class="pill ${t.running ? "live" : "paused"}">${t.running ? "Live" : "Paused"}</span></span></div>
      ${miniFelt(t)}
      <div class="tcard-meta"><span class="pill game">${gameOf(t.variant).label}</span><span class="pill gold num">${d2(t.bb_cents)} bb</span><span class="pill">${icon("i-users", "sm")}${Number(t.seated)}/${Number(t.num_seats)}</span>${t.listed ? "" : html`<span class="pill">${icon("i-lock", "sm")}Link only</span>`}</div>`);
    const row = h("div", { class: "tcard-cta" });
    const open = () => C().openTable(t.id, true).catch((e) => toast(e.message, "err"));
    const go = h("button", { class: "btn " + (t.is_seated ? "primary" : ""), type: "button" }, cta);
    go.addEventListener("click", (e) => { e.stopPropagation(); open(); });
    const copy = h("button", { class: "btn", type: "button", title: "Copy invite link", "aria-label": `Copy the invite link to ${t.name}` }, icon("i-link", "sm"));
    copy.addEventListener("click", (e) => { e.stopPropagation(); copyInvite(t.id); });
    row.appendChild(go); row.appendChild(copy);
    card.appendChild(row);
    card.addEventListener("click", open);
    return card;
  }
  // The grids are kept card by card (HGL-008): a card is rebuilt only when ITS table
  // changed, a new hand just updates its "#N" — the 5 s refresh used to wipe and redraw
  // every card and session row (a focused "Join table" lost focus, hovers flickered).
  const cardSig = (t) => JSON.stringify([C().G.clubId, Object.assign({}, t, { hand_no: 0 })]);
  function renderGrid(host, tables, emptyNote) {
    const old = new Map([...host.children].filter((c) => c.dataset.id).map((c) => [c.dataset.id, c]));
    const want = tables.map((t) => {
      const sig = cardSig(t);
      let card = old.get(t.id);
      old.delete(t.id);
      if (!card || card.dataset.sig !== sig) {
        const fresh = tableCard(t);
        fresh.dataset.id = t.id;
        fresh.dataset.sig = sig;
        if (card) host.replaceChild(fresh, card);
        card = fresh;
      }
      const pill = card.querySelector(".tcard-hand");
      if (pill) { pill.hidden = !t.hand_no; const txt = `#${Number(t.hand_no) || 0}`; if (pill.textContent !== txt) pill.textContent = txt; pill.title = `Hand ${Number(t.hand_no) || 0}`; }
      return card;
    });
    old.forEach((c) => c.remove());
    const cur = [...host.children].filter((c) => c.dataset.id);
    if (cur.length !== want.length || cur.some((c, i) => c !== want[i])) want.forEach((c) => host.appendChild(c));
    let empty = host.querySelector(".lb-empty");
    if (emptyNote && !tables.length) {
      if (!empty) { empty = h("div", { class: "lb-empty span-all" }); host.appendChild(empty); }
      const k = String(emptyNote);  // (markup; drawn again only when it changes)
      if (empty.dataset.k !== k) { empty.dataset.k = k; put(empty, emptyNote); }
    } else if (empty) empty.remove();
  }
  function renderLobby(data) {
    const tables = data.tables || [];
    const clubs = data.clubs || [];
    U.lobbyData = data;
    setGames(data.games);  // (the server's list of games: FE-004)
    setMyAvatar(data.my_avatar);
    if (UI.setMyName) UI.setMyName(data.my_name);  // (FEAT-008: the name the tables show for you)
    renderJoinReqs(data.join_requests);
    // a request to join was answered while this page was open: straight into the club
    // (and back to the table the request came from)
    const pend = U.pendingClub;
    if (pend && clubs.some((c) => c.id === pend.id)) {
      U.pendingClub = null;
      toast(`You're in ${pend.name}!`, "ok", 5000);
      HG.sound && HG.sound.play("sit");
      if (pend.table) { C().openTable(pend.table, true).catch(() => {}); return; }
      if (C().G.clubId !== pend.id) { switchClub(pend.id); return; }
    }
    const sig = JSON.stringify(data);
    if (sig === U.lobbySig) return;
    U.lobbySig = sig;
    $("lobby").classList.remove("loading");  // (until the first answer the page does not guess: no half-empty lobby)
    renderClubBar(data);
    const panel = U.clubPanel, shown = clubs.find((c) => c.id === data.club);
    if (panel && shown && panel.cid === shown.id) {
      const psig = `${shown.members}:${shown.requests}`;
      if (psig !== panel.sig && !(panel.typing && panel.typing())) { panel.sig = psig; panel.refresh(); }
    }
    const none = !clubs.length;
    $("lb-welcome").hidden = !none;
    $("lb-hero").querySelector(".lb-actions").hidden = none;  // (hosting needs a club: the welcome offers one)
    // Someone in a club came for their tables: the intro shrinks to its buttons, and the
    // join box lives in the club menu (HGL-005 / HGT-001 — on a phone it filled the screen)
    $("lobby").classList.toggle("has-club", !none);
    $("lb-open").closest(".lb-sec").hidden = none;
    renderPending(clubs);
    const mine = tables.filter((t) => t.is_host || t.is_seated);
    const open = tables.filter((t) => !(t.is_host || t.is_seated));
    $("lb-mine-sec").hidden = !mine.length;
    renderGrid($("lb-mine"), mine, "");
    renderGrid($("lb-open"), open, mine.length ? html`<b>No other tables right now</b>When someone in the club hosts one, it shows up here.` : html`<b>No tables yet</b>Host one — everyone in the club sees it here, or send them its link.`);
    // (a paused table isn't "running": say which are live — HGL-003)
    const live = open.filter((t) => t.running).length, paused = open.length - live;
    $("lb-open-count").textContent = open.length ? [live ? `${live} live` : "", paused ? `${paused} paused` : ""].filter(Boolean).join(" · ") : "";
    const sess = data.sessions || [];
    $("lb-sess-sec").hidden = !sess.length;
    const ssig = JSON.stringify(sess);
    if (U.sessSig !== ssig) {
      U.sessSig = ssig;
      // real buttons (A11Y-012: rows that took focus but not Enter). FEAT-001: your part
      // of settling each session up ("You pay Sam $42.10"), from the server's fewest payments.
      const settle = (x) => (x.settle || []).length
        ? html`<small class="sess-settle">${x.settle.map((p, i) => html`${i ? " · " : ""}${p.you === "pay" ? html`<span class="pay">You pay ${p.name} ${d2(p.cents)}</span>` : html`<span class="get">${p.name} pays you ${d2(p.cents)}</span>`}`)}</small>` : "";
      put($("lb-sessions"), html`${sess.map((x) =>
        html`<button type="button" class="sess" data-id="${x.id}" title="Open this session: final ledger and every hand"><span class="sess-name"><b>${x.name}</b><small>${gameOf(x.variant).label} · ${d2(x.bb_cents)} bb · ante ${d2(x.ante_cents)}</small>${settle(x)}</span><small class="opt">${Number(x.hands)} hand${x.hands === 1 ? "" : "s"}</small><small class="opt">in for ${d2(x.buyin_cents)}</small><b class="num ${x.net_cents >= 0 ? "pos" : "neg"}">${x.net_cents >= 0 ? "+" : ""}${d2(x.net_cents)}</b></button>`)}`);
      document.querySelectorAll("#lb-sessions .sess[data-id]").forEach((row) => row.addEventListener("click", () => C().openTable(row.dataset.id, true).catch((e) => toast(e.message, "err"))));
    }
    loadClub(false); // (rides the lobby poll, at most every 30 s)
  }
  // A request to join a club is waiting (HGL-001): said in the lobby for as long as it
  // waits — the dialog's note used to close with it, and nothing said to keep the page open.
  function renderPending(clubs) {
    const p = U.pendingClub;
    let el = $("lb-pending");
    const show = !!p && !(clubs || []).some((c) => c.id === p.id);
    if (!show) { if (el) el.hidden = true; return; }
    if (!el) {
      el = h("section", { id: "lb-pending", class: "lb-pending", role: "status" });
      const main = document.querySelector("#lobby .lb-main");
      main.insertBefore(el, main.firstChild);
    }
    const txt = html`<span class="spin"></span><div><b>Waiting for ${p.name} to let you in</b><small>Keep this page open — you'll be taken in${p.table ? " (and to the table)" : ""} as soon as the owner or an admin says yes.</small></div>`;
    if (el.dataset.k !== String(txt)) { el.dataset.k = String(txt); put(el, txt); }  // (the spinner keeps turning)
    el.hidden = false;
  }
  function copyInvite(id) {  // (copyText has the fallback: one copy of it — FE-002)
    return copyText(`${location.origin}/games/t/${id}`, "Invite link copied", "Invite link");
  }

  // ------------------------------------------------------------------- clubs
  // The lobby shows ONE club: its tables, its players, its numbers. A club's
  // owner (and admins) invite people with its link and let in whoever asks.
  const ROLE_WORD = { owner: "you run it", admin: "you're an admin", member: "you're a member" };
  function clubBadge(name, cls) {
    const A = HG.avatar;
    return html`<span class="club-badge ${cls || ""}" data-vars="h:${A.hueOf("club:" + name)}">${A.initials(name)}</span>`;
  }
  function inviteUrl(code) { return `${location.origin}/games/join/${code}`; }
  async function copyText(text, okMsg, title) {
    try { await navigator.clipboard.writeText(text); toast(okMsg, "ok"); }
    catch (_) {
      const pick = (e) => e.target.select();
      const box = h("input", { class: "input", readonly: true, value: text, onfocus: pick, onclick: pick });
      openModal({ title, sub: "Copy this link and send it to your friends.", body: box, buttons: [{ label: "Done", cls: "primary" }] });
    }
  }
  function currentClub() {
    const d = U.lobbyData || {};
    return (d.clubs || []).find((c) => c.id === C().G.clubId) || null;
  }
  function renderClubBar(data) {
    const club = (data.clubs || []).find((c) => c.id === data.club) || null;
    const bar = $("lb-clubbar");
    bar.hidden = !club;
    if (!club) return;
    const A = HG.avatar, badge = $("club-badge");
    badge.textContent = A.initials(club.name);
    badge.style.setProperty("--h", A.hueOf("club:" + club.name));
    $("club-name").textContent = club.name;
    // (not "1 of 3": the point is "you have more clubs — tap to switch")
    $("club-kicker").textContent = data.clubs.length > 1 ? `One of your ${data.clubs.length} clubs` : "Club";
    $("club-meta").textContent = `${club.members} member${club.members === 1 ? "" : "s"} · ${ROLE_WORD[club.role] || club.role}`;
    const manage = club.role === "owner" || club.role === "admin";
    $("club-invite").hidden = !manage;
    const req = $("club-req");
    req.hidden = !club.requests; req.textContent = String(club.requests || 0);
  }
  function switchClub(id) {
    C().setClub(id);
    U.lobbySig = ""; U.clubAt = 0; U.club = null;
    C().loadLobby().catch((e) => toast(e.message, "err"));
    loadClub(true);
  }
  function openClubMenu(anchor) {
    const clubs = (U.lobbyData && U.lobbyData.clubs) || [];
    const archived = (U.lobbyData && U.lobbyData.archived_clubs) || [];
    const cur = C().G.clubId;
    openMenu(anchor, [{ header: "Your clubs" }]
      .concat(clubs.map((c) => ({
        icon: c.id === cur ? "i-check" : "i-users",
        label: c.name + (c.requests ? ` · ${c.requests} waiting` : ""),
        hint: `${c.members} member${c.members === 1 ? "" : "s"} · ${ROLE_WORD[c.role] || c.role}`,
        onClick: () => { if (c.id !== cur) switchClub(c.id); },
      })))
      .concat(["-", { icon: "i-plus", label: "Start a new club", onClick: openCreateClub },
        { icon: "i-link", label: "Join a club with a link", onClick: openJoinClub }])
      // FEAT-007: the clubs you archived — back with one tap
      .concat(archived.length ? ["-", { header: "Archived" }].concat(archived.map((c) => ({
        icon: "i-play", label: `Restore ${c.name}`, hint: "Back in everyone's lobby, invite link and all",
        onClick: () => restoreClub(c),
      }))) : []));
  }
  async function restoreClub(c) {
    try {
      await C().j(`/games/api/clubs/${encodeURIComponent(c.id)}/archive`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ on: false }) });
      toast(`${c.name} is back`, "ok");
      switchClub(c.id);
    } catch (e) { toast(e.message, "err", 5000); }
  }
  async function createClub(name) {
    try {
      const club = await C().j("/games/api/clubs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
      switchClub(club.id);
      toast(`${club.name} is ready — send your friends its invite link`, "ok", 5000);
      return true;
    } catch (e) { toast(e.message, "err"); return false; }
  }
  function openCreateClub() {
    const body = h("div", { class: "stack-14" },
      html`<label class="field"><span>Club name</span><input type="text" class="input" id="cc-name" maxlength="40" placeholder="Friday night"/></label>
      <p class="muted note">You run it: invite people with the club's link and decide who's in. Its tables, leaderboard and everyone's numbers stay inside the club.</p>`);
    openModal({
      title: "Start a club", body,
      buttons: [{ label: "Cancel", cls: "ghost" }, {
        label: "Create club", cls: "gold",
        onClick: async () => {
          const name = body.querySelector("#cc-name").value.trim();
          if (!name) { toast("Give the club a name", "err"); return false; }
          return createClub(name);
        },
      }],
    });
  }
  function openJoinClub() {
    const body = h("div", { class: "stack-12" },
      html`<label class="field"><span>Invite link</span><input type="text" class="input" id="jc-link" placeholder="Paste the invite link" autocomplete="off"/><small>A table link works too: you can ask its club to let you in.</small></label>`);
    // (a link that doesn't work leaves the dialog open with the text in it, to fix one character)
    const go = async () => (await joinByLink(body.querySelector("#jc-link").value)) ? true : false;
    body.querySelector("#jc-link").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); api.foot.lastChild.click(); } });
    const api = openModal({
      title: "Join a club", body,
      buttons: [{ label: "Cancel", cls: "ghost" }, { label: "Continue", cls: "gold", onClick: go }],
    });
  }
  // a pasted link: a club invite, a table link (joins or asks its club) or a bare code.
  // true = it led somewhere (a table, an invite, a club gate): the box can be cleared.
  async function joinByLink(raw) {
    const s = String(raw || "").trim();
    let m = s.match(/\/games\/join\/([A-Za-z0-9_-]+)/);
    if (m) return openInvite(m[1]);
    m = s.match(/\/games\/t\/([A-Za-z0-9_-]+)/) || s.match(/^([A-Za-z0-9_-]{4,})$/);
    if (!m) { toast("Paste a table link or a club's invite link", "err"); return false; }
    try { await C().openTable(m[1], true); return true; }
    catch (err) {
      if (err.status === 403 && err.detail && err.detail.error === "club") { openClubGate(err.detail, m[1]); return true; }
      if (err.status === 404 && !/\/games\/t\//.test(s)) return openInvite(m[1], true);  // (a bare code: a club's?)
      toast(err.status === 404 ? "No table or club with that link" : err.message, "err");
      return false;
    }
  }
  // someone else's club: its invite link (join now, or ask when the club asks first)
  async function openInvite(code, quiet404) {
    let info;
    try { info = await C().j(`/games/api/invites/${encodeURIComponent(code)}`); }
    catch (e) { toast(e.status === 404 ? (quiet404 ? "No table or club with that link" : "That invite link is no longer valid — ask for a new one") : e.message, "err", 5000); return false; }
    if (info.member) { if (C().G.clubId !== info.club.id) switchClub(info.club.id); toast(`You're in ${info.club.name}`, "ok"); return true; }
    const c = info.club, body = h("div", { class: "invite-box" });
    const paint = () => {
      put(body, html`<div class="inv-head">${clubBadge(c.name, "lg")}<div><b>${c.name}</b><small>${c.members} member${c.members === 1 ? "" : "s"} · run by ${c.owner_name}</small></div></div>
        ${info.request === "pending" ? html`<p class="inv-note">Your request is in. ${c.owner_name} or a club admin lets you in — you'll be taken into the club as soon as they do.</p>`
          : info.request === "declined" && info.retry_in > 0 ? html`<p class="inv-note">The club didn't let you in this time. You can ask again in a minute.</p>`
            : html`<p>Join to see the club's tables, sit down with its members and show up on its leaderboard. The club's numbers stay inside the club.</p>`}`);
    };
    paint();
    const waiting = info.request === "pending" || (info.request === "declined" && info.retry_in > 0);
    const buttons = [{ label: waiting ? "Close" : "Not now", cls: "ghost" }];
    if (!waiting) buttons.push({
      label: info.approve ? "Ask to join" : "Join club", cls: "gold",
      onClick: async () => {
        try {
          const out = await C().j(`/games/api/invites/${encodeURIComponent(code)}/join`, { method: "POST" });
          if (out.member) { switchClub(out.club.id); toast(`Welcome to ${out.club.name}!`, "ok", 5000); HG.sound && HG.sound.play("sit"); return true; }
          info = out; paint();
          U.pendingClub = { id: c.id, name: c.name, table: null };
          renderPending((U.lobbyData || {}).clubs);
          api.setButtons([{ label: "Close", cls: "primary" }]);  // (the waiting note stays up — HGL-001)
          return false;
        } catch (e) { toast(e.message, "err"); return false; }
      },
    });
    const api = openModal({ title: "Club invite", body, buttons, autofocus: false });
    if (info.request === "pending") { U.pendingClub = { id: c.id, name: c.name, table: null }; renderPending((U.lobbyData || {}).clubs); }
    return true;
  }
  // a table of a club I am not in (a table link): ask to join the club
  function openClubGate(d, tableId) {
    const c = d.club, body = h("div", { class: "invite-box" });
    let st = d.request;
    const retry = d.retry_in || 0;
    const paint = () => {
      put(body, html`<div class="inv-head">${clubBadge(c.name, "lg")}<div><b>${c.name}</b><small>This table belongs to the club</small></div></div>
        ${st === "pending" ? html`<p class="inv-note">Your request is in — the club's owner or an admin lets you in. The table opens by itself as soon as they do (keep this page open).</p>`
          : st === "declined" && retry > 0 ? html`<p class="inv-note">The club didn't let you in this time. You can ask again in a minute.</p>`
            : html`<p>Only the club's members can sit at its tables or watch them. Ask to join, and the club's owner or an admin lets you in.</p>`}`);
    };
    paint();
    if (st === "pending") U.pendingClub = { id: c.id, name: c.name, table: tableId || null };
    const waiting = st === "pending" || (st === "declined" && retry > 0);
    const buttons = [{ label: waiting ? "Close" : "Not now", cls: "ghost" }];
    if (!waiting) buttons.push({
      label: "Ask to join", cls: "gold",
      onClick: async () => {
        try {
          const out = await C().j(`/games/api/clubs/${encodeURIComponent(c.id)}/request`, { method: "POST" });
          if (out.member) { if (tableId) await C().openTable(tableId, true); return true; }
          st = out.request; paint();
          U.pendingClub = { id: c.id, name: c.name, table: tableId || null };
          renderPending((U.lobbyData || {}).clubs);
          api.setButtons([{ label: "Close", cls: "primary" }]);  // (the waiting note stays up — HGL-001)
          return false;
        } catch (e) { toast(e.message, "err"); return false; }
      },
    });
    const api = openModal({ title: "Members only", body, buttons, autofocus: false });
  }
  // the club's members, invite link and settings (owner / admins manage, members look)
  async function openClubSettings() {
    const cid = C().G.clubId;
    if (!cid) return;
    if (U.clubPanel && U.clubPanel.cid === cid && !U.clubPanel.api.closed) return;  // (a second tap while it opens)
    // opens at once and fills in when the club's data lands (HGH-002)
    const body = h("div", { class: "club-set" }, loading());
    const api = openModal({
      title: "Members", body, wide: true, autofocus: false,
      buttons: [{ label: "Done", cls: "primary" }],
      onClose: () => { if (U.clubPanel && U.clubPanel.api === api) U.clubPanel = null; },
    });
    // (someone joins or asks while it is open: the lobby poll notices and it repaints —
    // never while a box in it is being typed in: the rename box used to lose its text, HGL-006)
    const typing = () => { const a = document.activeElement; return !!a && body.contains(a) && /^(INPUT|TEXTAREA)$/.test(a.tagName) && a.type !== "checkbox"; };
    const summary0 = currentClub();
    let v = null, refreshFn = null;
    U.clubPanel = { api, cid, sig: summary0 ? `${summary0.members}:${summary0.requests}` : "", refresh: () => refreshFn && refreshFn(), typing };
    try { v = await C().j(`/games/api/clubs/${encodeURIComponent(cid)}`); } catch (e) { api.close(null); return toast(e.message, "err"); }
    if (api.closed) return;
    const post = async (path, payload) => {
      try {
        const out = await C().j(`/games/api/clubs/${encodeURIComponent(cid)}/${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload || {}) });
        if (out && out.members) v = out;
        U.lobbySig = ""; C().loadLobby().catch(() => {});
        return true;
      } catch (e) { toast(e.message, "err", 5000); return false; }
    };
    const refresh = async () => { try { v = await C().j(`/games/api/clubs/${encodeURIComponent(cid)}`); } catch (_) { /* keep */ } paint(); };
    refreshFn = refresh;
    const canAct = (m) => !m.is_me && (v.role === "owner" || (v.role === "admin" && m.role === "member"));
    // FEAT-006: your own name in this club; the owner and admins may name anyone (two Mikes)
    const canName = (m) => m.is_me || v.role === "owner" || v.role === "admin";
    const openNickname = (m) => {
      const box = h("div", { class: "stack-16" }, html`<label class="field"><span>${m.is_me ? "Your name in this club" : `${m.base_name || m.name}'s name in this club`}</span>
        <input class="input" id="nick-in" maxlength="20" autocomplete="off" value="${m.nickname || ""}" placeholder="${m.base_name || m.name}"/>
        <small class="muted">Shown at this club's tables and on its page, instead of ${m.is_me ? "your" : "their"} name everywhere else${m.base_name ? ` (${m.base_name})` : ""}. Empty = no nickname.</small></label>`);
      openModal({
        title: m.is_me ? "Your name in this club" : "Nickname", body: box,
        buttons: [{ label: "Cancel", cls: "ghost" }, { label: "Save", cls: "gold", onClick: async () => {
          const nickname = box.querySelector("#nick-in").value.trim();
          if (!(await post("nickname", { user_id: m.user_id, nickname }))) return false;
          toast(nickname ? `Named ${nickname} in ${v.name}` : "Nickname removed", "ok");
          paint();
        } }],
      });
    };
    const paint = () => {
      const owner = v.role === "owner", manage = owner || v.role === "admin";
      put(body, html`${owner ? html`<div class="grp"><label class="field"><span>Club name</span><span class="row-inline"><input class="input" id="cs-name" maxlength="40" value="${v.name}"/><button class="btn sm" id="cs-save" type="button">Save</button></span></label></div>` : ""}
        ${manage ? html`<div class="grp"><h4>Invite link</h4><span class="row-inline"><input class="input" id="cs-link" readonly value="${inviteUrl(v.invite_code)}"/><button class="btn sm gold" id="cs-copy" type="button">${icon("i-copy", "sm")}Copy</button></span>
          <small class="muted">Anyone with this link can ${v.approve_joins ? "ask to join" : "join the club"}. <button class="linkish" id="cs-reset" type="button">Make a new link</button> — the old one stops working.</small>
          ${owner ? html`<div class="setrow"><div><b>Ask me first</b><small>New people ask to join; you or an admin let them in</small></div><label class="switch"><input type="checkbox" id="cs-approve" ${v.approve_joins ? "checked" : ""}/><i></i></label></div>` : ""}</div>` : ""}
        ${(v.requests || []).length ? html`<div class="grp"><h4>Waiting to join · ${v.requests.length}</h4>${v.requests.map((q) =>
          html`<div class="mem"><span class="mem-who">${avatar(q.name, q.name, "sm", q.avatar)}<span><b>${q.name}</b><small>${q.email}</small></span></span><span class="mem-act"><button class="btn sm ghost" type="button" data-deny="${q.user_id}">Not now</button><button class="btn sm gold" type="button" data-allow="${q.user_id}">Let in</button></span></div>`)}</div>` : ""}
        <div class="grp"><h4>Members · ${v.members.length}</h4>${v.members.map((m) =>
          html`<div class="mem"><span class="mem-who">${avatar(m.name, m.name, "sm", m.avatar)}<span><b>${m.name}${m.is_me ? html` <i>you</i>` : ""}</b><small>${m.role === "owner" ? "Runs the club" : m.role === "admin" ? "Admin: lets people in" : "Member"}${m.nickname && m.base_name && m.base_name !== m.name ? ` · ${m.base_name} elsewhere` : ""}</small></span></span><span class="mem-act"><span class="pill role-${m.role}">${m.role}</span>${canAct(m) || canName(m) ? html`<button class="icon-btn" type="button" data-mem="${m.user_id}" aria-label="${m.is_me ? "Your name in this club" : `Manage ${m.name}`}">${icon("i-menu")}</button>` : ""}</span></div>`)}</div>
        ${owner ? html`<p class="muted note">You run this club. To step down, hand it to another member (the ☰ next to their name)${v.is_main ? "" : " — or archive the club when its games are over"}.</p>
          ${v.is_main ? "" : html`<button class="btn sm danger" id="cs-archive" type="button">${icon("i-door", "sm")}Archive club</button>`}`
          : html`<button class="btn sm danger" id="cs-leave" type="button">${icon("i-door", "sm")}Leave club</button>`}`);
      const q = (id) => body.querySelector("#" + id);
      if (q("cs-save")) q("cs-save").addEventListener("click", async () => { const name = q("cs-name").value.trim(); if (!name) return toast("Give the club a name", "err"); if (await post("settings", { name })) { toast("Renamed", "ok"); paint(); } });
      if (q("cs-copy")) q("cs-copy").addEventListener("click", () => copyText(inviteUrl(v.invite_code), "Invite link copied", "Invite link"));
      if (q("cs-link")) { const pick = (e) => e.target.select(); q("cs-link").addEventListener("focus", pick); q("cs-link").addEventListener("click", pick); }
      if (q("cs-reset")) q("cs-reset").addEventListener("click", async () => {
        const ok = await confirmDialog({ title: "Make a new invite link?", text: "The current link stops working — anyone who has it but hasn't joined yet will need the new one.", okLabel: "New link" });
        if (ok && await post("invite")) { toast("New invite link ready", "ok"); paint(); }
      });
      if (q("cs-approve")) q("cs-approve").addEventListener("change", async (e) => { if (await post("settings", { approve_joins: e.target.checked })) { toast(e.target.checked ? "New people ask first now" : "The link lets people straight in", "ok"); paint(); } else e.target.checked = !e.target.checked; });
      body.querySelectorAll("[data-allow],[data-deny]").forEach((b) => b.addEventListener("click", async () => {
        const allow = !!b.dataset.allow, uid = Number(b.dataset.allow || b.dataset.deny);
        b.disabled = true;
        if (await post("requests/decide", { user_id: uid, allow })) { toast(allow ? "Welcome aboard — they're in" : "Request declined", allow ? "ok" : ""); await refresh(); }
        else b.disabled = false;
      }));
      body.querySelectorAll("[data-mem]").forEach((b) => b.addEventListener("click", () => {
        const m = v.members.find((x) => String(x.user_id) === b.dataset.mem);
        if (!m) return;
        const items = [{ header: m.name }];
        if (canName(m)) {
          items.push({ icon: "i-user", label: m.is_me ? "Your name in this club…" : "Nickname in this club…", onClick: () => openNickname(m) });
          if (canAct(m)) items.push("-");
        }
        if (!canAct(m)) return openMenu(b, items);
        if (v.role === "owner") {
          items.push(m.role === "admin"
            ? { icon: "i-user", label: "Make a member", onClick: async () => { if (await post("members", { user_id: m.user_id, role: "member" })) paint(); } }
            : { icon: "i-shield", label: "Make an admin", hint: "Admins let people in and remove members", onClick: async () => { if (await post("members", { user_id: m.user_id, role: "admin" })) paint(); } });
          items.push({ icon: "i-crown", label: "Hand the club over", onClick: async () => {
            const ok = await confirmDialog({ title: `Hand ${v.name} to ${m.name}?`, text: `${m.name} runs the club from now on; you stay on as an admin.`, okLabel: "Hand over" });
            if (ok && await post("members", { user_id: m.user_id, role: "owner" })) { toast(`${m.name} runs ${v.name} now`, "ok"); paint(); }
          } });
          items.push("-");
        }
        items.push({ icon: "i-door", label: "Remove from the club", danger: true, onClick: async () => {
          const ok = await confirmDialog({ title: `Remove ${m.name}?`, text: "They lose the club's tables and numbers. Their hands stay in the club's history.", okLabel: "Remove", danger: true });
          if (ok && await post("members", { user_id: m.user_id, remove: true })) { toast(`${m.name} is out of the club`, ""); paint(); }
        } });
        openMenu(b, items);
      }));
      // FEAT-007: the owner retires the club — nothing is deleted, and it comes back from the club menu
      if (q("cs-archive")) q("cs-archive").addEventListener("click", async () => {
        const ok = await confirmDialog({ title: `Archive ${v.name}?`, text: "It leaves everyone's lobby and its invite link stops working. Its hands, ledgers and numbers are all kept, its old tables still open by link, and you can restore it any time from the club menu.", okLabel: "Archive club", danger: true });
        if (!ok) return;
        try {
          await C().j(`/games/api/clubs/${encodeURIComponent(cid)}/archive`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ on: true }) });
          api.close(null);
          toast(`${v.name} is archived — restore it from the club menu`, "ok", 5000);
          C().setClub(null); U.lobbySig = ""; C().loadLobby().catch(() => {});
        } catch (e) { toast(e.message, "err", 5000); }
      });
      if (q("cs-leave")) q("cs-leave").addEventListener("click", async () => {
        const ok = await confirmDialog({ title: `Leave ${v.name}?`, text: "Its tables and numbers disappear from your lobby. You can come back with an invite link.", okLabel: "Leave club", danger: true });
        if (!ok) return;
        try {
          await C().j(`/games/api/clubs/${encodeURIComponent(cid)}/leave`, { method: "POST" });
          api.close(null);
          toast(`You left ${v.name}`, "");
          C().setClub(null); U.lobbySig = ""; C().loadLobby().catch(() => {});
        } catch (e) { toast(e.message, "err", 5000); }
      });
    };
    api.setTitle(v.name, `${v.members.length} member${v.members.length === 1 ? "" : "s"} · ${ROLE_WORD[v.role] || v.role}`);
    paint();
    const summary = currentClub();
    U.clubPanel.sig = summary ? `${summary.members}:${summary.requests}` : "";
  }

  // "Host a table". Opens at once and fills in when your settings from last time land
  // (HGH-002); the games are the server's list (FE-004: a server that can't deal PLO67
  // doesn't offer it); the choices are the same as Manage's (HGT-029). A table that was
  // created but couldn't be opened is never created twice (HGT-030).
  async function openCreate(fresh) {
    const me = C().G.me || {};
    const clubs = (U.lobbyData && U.lobbyData.clubs) || [];
    const cur = clubs.find((c) => c.id === C().G.clubId) || clubs[0];
    if (!cur) return openCreateClub();  // (a table lives in a club)
    if (U.creating && !U.creating.closed) return;  // (a second tap while it opens)
    const sub = `In ${cur.name}: its members see the table in the lobby. You can change almost everything later from Manage table.`;
    const api = openModal({ title: "Host a table", sub, body: loading(), buttons: [{ label: "Cancel", cls: "ghost" }] });
    U.creating = api;
    // The settings of the last table you hosted (2026-09-26); amounts come in big blinds.
    let prefs = null;
    if (!fresh) { try { prefs = (await C().j("/games/api/host_prefs")).prefs || null; } catch (_) { prefs = null; } }
    if (api.closed) return;
    const P = prefs || {};
    const num = (v, dflt) => (Number.isFinite(Number(v)) && v !== null && v !== undefined ? Number(v) : dflt);
    const bb0 = num(P.bb_cents, 0) > 0 ? num(P.bb_cents, 100) : 100;
    const anteBB0 = num(P.ante_bb, 0) > 0 ? num(P.ante_bb, 3) : 3;
    const buyinBB0 = num(P.buyin_bb, 0) > 0 ? num(P.buyin_bb, 40) : 40;
    // the game: fixed for the table's life; the seat choices follow it
    const games = Object.entries(GAMES).filter(([, G]) => G.available !== false);
    const game0 = games.some(([k]) => k === P.variant) ? P.variant : "plo5";
    const seatsFor = (g, want) => { const G = gameOf(g); return segHtml("c-seats", OPTIONS.seats(G.maxSeats), Math.min(G.maxSeats, num(want, G.maxSeats))); };
    const first = String(me.name || "").trim().split(/\s+/)[0];
    const body = h("div", { class: "stack-16" });
    const sw = (id, on) => html`<label class="switch"><input type="checkbox" id="${id}" ${on ? "checked" : ""}/><i></i></label>`;
    put(body, html`${prefs ? html`<div class="setrow"><div><b>Your settings from last time</b><small>Stakes, seats, clock and buy-ins — and Manage's too: automatic chips, grades, rabbit</small></div><button type="button" class="btn sm" id="c-reset">Use defaults</button></div>` : ""}
      ${clubs.length > 1 ? html`<label class="field"><span>Club</span><select class="input" id="c-club">${clubs.map((c) => html`<option value="${c.id}" ${c.id === cur.id ? "selected" : ""}>${c.name}</option>`)}</select><small>Only this club's members can see and join the table</small></label>` : ""}
      <label class="field"><span>Table name</span><input type="text" class="input" id="c-name" maxlength="60" value="${first ? `${first}'s game` : "Home game"}"/></label>
      <div class="field"><span>Game</span>${segHtml("c-game", games.map(([k, G]) => [k, G.label]), game0)}<small id="c-game-note">${gameNote(game0)}</small>
      <small>Every hand is a bomb pot: no blinds — everyone antes and the action starts on the flop.</small></div>
      <div class="row2"><label class="field"><span>Big blind</span>${moneyInput("c-bb", bb0)}<small>The chip unit and the minimum bet</small></label>
      <label class="field"><span>Ante, in big blinds</span><div class="money unit-bb"><input class="input" id="c-ante-bb" inputmode="decimal" autocomplete="off" value="${anteBB0}"/></div><small id="c-ante-eq">= $3.00 per player, every hand</small></label></div>
      <div class="field" id="c-seats-f"><span>Seats</span>${seatsFor(game0, P.num_seats)}</div>
      <label class="field"><span>Your buy-in</span>${moneyInput("c-buyin", Math.round(bb0 * buyinBB0))}<small>What you sit down with — you can add chips later.</small></label>
      <details class="adv"><summary>More options</summary><div>
      <div class="row2"><label class="field"><span>Min buy-in</span>${moneyInput("c-min", Math.round(bb0 * num(P.min_buyin_bb, 0)))}<small>0 = no minimum</small></label><label class="field"><span>Max buy-in</span>${moneyInput("c-max", Math.round(bb0 * num(P.max_buyin_bb, 0)))}<small>0 = no maximum</small></label></div>
      <div class="field"><span>Decision time</span>${segHtml("c-clock", OPTIONS.clock, num(P.decision_secs, 30), secsLabel)}</div>
      <div class="field"><span>Time bank</span>${segHtml("c-bank", OPTIONS.bank, num(P.time_bank_secs, 30), secsLabel)}</div>
      <div class="field"><span>Next hand</span>${segHtml("c-deal", OPTIONS.deal, num(P.deal_delay_secs, 5), (v) => (Number(v) ? `${v}s` : "Manual"))}</div>
      <div class="setrow"><div><b>I approve every buy-in</b><small>Sit-downs and top-ups wait for your OK — you can trust regulars so they never wait</small></div>${sw("c-approve", P.approve_buyins)}</div>
      <div class="setrow"><div><b>Players may take chips off the table</b><small>Ratholing allowed: anyone can pocket part of their stack between hands</small></div>${sw("c-rathole", P.allow_rathole)}</div>
      <div class="setrow"><div><b>Show in the club lobby</b><small>Off = link only: only club members with the link find it</small></div>${sw("c-listed", P.listed !== false)}</div>
      </div></details>`);
    segWire(body);
    const q = (id) => body.querySelector("#" + id);
    const gameVal = () => { const on = body.querySelector("#c-game button.on"); return on ? on.dataset.v : "plo5"; };
    q("c-game").addEventListener("pick", (e) => {
      const f = q("c-seats-f"), want = segVal(body, "c-seats");
      put(f, html`<span>Seats</span>${seatsFor(e.detail, want)}`);
      segWire(f);
      q("c-game-note").textContent = gameNote(e.detail);
    });
    const reset = q("c-reset");
    if (reset) reset.addEventListener("click", () => closeTopThen(() => openCreate(true)));
    const bbCents = () => Math.max(1, C().toCents(q("c-bb").value) || 0);
    const anteBB = () => Math.max(0, C().readAmount(q("c-ante-bb").value) || 0);  // (the one amount reader: "0,5" = half)
    const anteCents = () => Math.round(bbCents() * anteBB());
    let buyinTouched = false;
    const sync = () => {
      q("c-ante-eq").textContent = `= ${d2(anteCents())} per player, every hand`;
      if (!buyinTouched) q("c-buyin").value = ((bbCents() * buyinBB0) / 100).toFixed(2);  // your usual buy-in in bb (40 bb by default)
    };
    q("c-bb").addEventListener("input", sync);
    q("c-ante-bb").addEventListener("input", sync);
    q("c-buyin").addEventListener("input", () => { buyinTouched = true; });
    sync();
    api.setBody(body);
    api.setButtons([
      { label: "Cancel", cls: "ghost" },
      {
        label: "Create table", cls: "gold",
        onClick: async () => {
          const bb = bbCents();
          if (anteBB() <= 0) { toast("Set an ante above 0", "err"); return false; }
          const payload = {
            name: q("c-name").value.trim() || "Home game", variant: gameVal(),
            bb_cents: bb, sb_cents: Math.max(1, Math.round(bb / 2)),
            ante_cents: anteCents(), default_buyin_cents: C().toCents(q("c-buyin").value),
            min_buyin_cents: C().toCents(q("c-min").value) || 0, max_buyin_cents: C().toCents(q("c-max").value) || 0,
            num_seats: segVal(body, "c-seats"), decision_secs: segVal(body, "c-clock"),
            time_bank_secs: segVal(body, "c-bank"), deal_delay_secs: segVal(body, "c-deal"), listed: q("c-listed").checked,
            approve_buyins: q("c-approve").checked, allow_rathole: q("c-rathole").checked,
            club_id: clubs.length > 1 ? q("c-club").value : cur.id,
            remembered: !!prefs,  // also bring Manage's settings from last time
          };
          let s;
          try { s = await C().j("/games/api/tables", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); }
          catch (e) { toast(e.message, "err"); return false; }
          // created: the dialog goes whatever happens next, so a second click can't make a second table
          api.close(true);
          HG.sound && HG.sound.play("sit");
          C().openTable(s.id, true).catch(() => {
            toast("Your table is ready, but it couldn't be opened just now — it's in the lobby under Your tables", "err", 8000);
            U.lobbySig = "";
            C().loadLobby().catch(() => {});
          });
        },
      },
    ]);
    const nm = q("c-name");
    if (nm && !("ontouchstart" in globalThis)) setTimeout(() => nm.focus({ preventScroll: true }), 30);
  }

  // ------------------------------------------------------------ the club: everyone
  const MIN_RANKED = 20; // graded decisions before an accuracy counts for the podium
  const accTxt = (v) => (v == null ? "–" : Math.round(v) + "%");
  const signed = (c) => (c > 0 ? "+" : c < 0 ? "−" : "") + d2(Math.abs(c));
  const tone = (c) => (c > 0 ? "pos" : c < 0 ? "neg" : "");

  // PLO5 and PLO6 are different games: the club's numbers are shown one game at a time
  // (2026-09-26). The server picks the club's most-played game; a switch picks another,
  // remembered per club in this browser.
  const CLUB_GAME_KEY = "hg.clubgame.v1";
  function clubGames() { try { return JSON.parse(localStorage.getItem(CLUB_GAME_KEY) || "{}") || {}; } catch (_) { return {}; } }
  function clubGameFor(cid) { const v = clubGames()[cid]; return GAMES[v] ? v : ""; }
  function setClubGame(cid, v) {
    const all = clubGames();
    all[cid] = v;
    try { localStorage.setItem(CLUB_GAME_KEY, JSON.stringify(all)); } catch (_) { /* private mode: this visit only */ }
  }
  // FEAT-004: the club's numbers for a period — all time, this month (the viewer's OWN
  // month: the start is sent as a date), the last 30 days — remembered in this browser;
  // a player's hands & stats window opens on the same period.
  const PERIOD_KEY = "hg.clubperiod.v1";
  const PERIODS = [["all", "All time"], ["month", "This month"], ["30d", "30 days"]];
  function clubPeriod() {
    try { const v = localStorage.getItem(PERIOD_KEY); return PERIODS.some((p) => p[0] === v) ? v : "all"; } catch (_) { return "all"; }
  }
  function setClubPeriod(v) { try { localStorage.setItem(PERIOD_KEY, v); } catch (_) { /* private mode: this visit only */ } }
  // the API's `since` for a period ("" = all time), and how the windows say it
  function statsSince(v) {
    const p = v || clubPeriod();
    if (p === "month") { const d = new Date(); return new Date(d.getFullYear(), d.getMonth(), 1).toISOString(); }
    return p === "all" ? "" : p;
  }
  const periodLabel = (v) => (PERIODS.find((p) => p[0] === (v || clubPeriod())) || PERIODS[0])[1];
  function renderPeriod() {
    let pd = $("lb-period");
    if (!pd) {  // (next to the game switch: the two choose what the numbers are about)
      pd = h("div", { class: "seg lb-game lb-period", id: "lb-period", role: "group", "aria-label": "Period" });
      $("lb-game").after(pd);
      pd.addEventListener("click", (e) => {
        const b = e.target.closest("button[data-p]");
        if (!b || b.classList.contains("on")) return;
        setClubPeriod(b.dataset.p);
        renderPeriod();
        loadClub(true);
      });
    }
    const cur = clubPeriod();
    put(pd, html`${PERIODS.map(([k, label]) => html`<button type="button" data-p="${k}" class="${k === cur ? "on" : ""}" aria-pressed="${k === cur}">${label}</button>`)}`);
  }
  async function loadClub(force) {
    const cid = C().G.clubId;
    if (!cid) { renderClub({ players: [] }); return; }  // (no club (yet): no numbers to show)
    // (the 30 s throttle is per club: a switch, or the first load after the club is known, always fetches)
    if (!force && U.clubAt && U.clubFor === cid && Date.now() - U.clubAt < 30000) return;
    U.clubAt = Date.now(); U.clubFor = cid;
    const v = clubGameFor(cid), since = statsSince();
    try {
      const data = await C().j(`/games/api/community?club=${encodeURIComponent(cid)}` + (v ? `&variant=${encodeURIComponent(v)}` : "") + (since ? `&since=${encodeURIComponent(since)}` : ""));
      if (C().G.clubId === cid) renderClub(data);  // (a switch in the meantime: the newer answer wins)
    } catch (_) { /* the lobby works without it */ }
  }
  function renderClub(data) {
    U.club = data;
    const players = data.players || [];
    const G = gameOf(data.variant), played = (data.games || []).filter((g) => g.hands > 0);
    // (a period with no hands keeps the section: its switch is how you get back)
    $("lb-club-sec").hidden = !players.length && !played.length && clubPeriod() === "all";
    if ($("lb-club-sec").hidden) return;
    renderPeriod();
    // the game switch: once the club has played more than one game
    const codes = played.map((g) => g.code);
    if (data.variant && !codes.includes(data.variant)) codes.push(data.variant);
    const sw = $("lb-game");
    sw.hidden = codes.length < 2;
    put(sw, html`${codes.map((c) => {
      const n = ((data.games || []).find((g) => g.code === c) || { hands: 0 }).hands;
      return html`<button type="button" data-v="${c}" class="${c === data.variant ? "on" : ""}" aria-pressed="${c === data.variant}" title="${n} hand${n === 1 ? "" : "s"} played">${gameOf(c).label}</button>`;
    })}`);
    $("lb-club-sub").textContent = `${G.label} · ${players.length} player${players.length === 1 ? "" : "s"} · ${G.graded ? "accuracy is the network's rating of every decision" : `${G.label} isn't graded yet (there is no ${G.label} network)`}`;
    if (!players.length) {
      $("lb-podium").hidden = true;
      put($("lb-players"), clubPeriod() !== "all"
        ? html`<div class="lb-empty span-all"><b>No ${G.label} hands — ${periodLabel().toLowerCase()}</b>Nobody in the club has played ${G.label} in this period. Switch to All time for everything.</div>`
        : html`<div class="lb-empty span-all"><b>No ${G.label} hands yet</b>The club's ${G.label} numbers show up here after its first ${G.label} hand.</div>`);
      return;
    }
    // podium: accuracy, among players with enough graded decisions to mean something
    // (a game the network doesn't grade has no podium)
    const rated = G.graded ? players.filter((p) => p.accuracy != null) : [];
    const ranked = rated.filter((p) => p.graded >= MIN_RANKED).sort((a, b) => b.accuracy - a.accuracy || b.graded - a.graded);
    const early = rated.filter((p) => p.graded < MIN_RANKED).sort((a, b) => b.accuracy - a.accuracy || b.graded - a.graded);
    const top = ranked.concat(early).slice(0, 3);
    const pod = $("lb-podium");
    pod.hidden = !top.length;
    const step = (p, place) => !p ? html`<div class="pod-col p${place} empty"><div class="pod-step"><b>${place}</b></div></div>` :
      html`<button type="button" class="pod-col p${place}" data-uid="${p.user_id}" title="Open ${p.name}'s hands">${place === 1 ? html`<span class="pod-crown">${icon("i-crown")}</span>` : ""}${avatar(p.name, p.name, "lg", p.avatar)}<span class="pod-name">${p.name}${p.is_me ? html` <i>you</i>` : ""}</span><span class="pod-acc num">${accTxt(p.accuracy)}</span><span class="pod-sub">${p.graded} decision${p.graded === 1 ? "" : "s"}${p.graded < MIN_RANKED ? " · provisional" : ""}</span><span class="pod-step"><b>${place}</b></span></button>`;
    // 1st, 2nd, 3rd in the page (screen readers and Tab); games.css draws them 2 · 1 · 3
    put(pod, html`${step(top[0], 1)}${step(top[1], 2)}${step(top[2], 3)}`);
    // one card per player, biggest winner first (an ungraded game's meter: hands won)
    put($("lb-players"), html`${players.map((p) => {
      const won = p.hands ? Math.round((100 * p.wins) / p.hands) : 0;
      const pct = !G.graded ? won : p.accuracy == null ? 0 : Math.max(0, Math.min(100, p.accuracy));
      return html`<button type="button" class="plcard ${p.is_me ? "me" : ""}" data-uid="${p.user_id}"><span class="plcard-top">${avatar(p.name, p.name, "", p.avatar)}<span class="plcard-name"><b>${p.name}</b><small>${p.hands} hand${p.hands === 1 ? "" : "s"} · ${p.sessions} session${p.sessions === 1 ? "" : "s"}</small></span><b class="num plcard-net ${tone(p.net_cents)}">${signed(p.net_cents)}</b></span>
        <span class="plcard-acc"><span class="plcard-meter"><i data-vars="pct:${pct}%"></i></span><b class="num">${G.graded ? accTxt(p.accuracy) : won + "%"}</b></span>
        ${G.graded
          ? html`<span class="plcard-foot"><span>Accuracy${p.graded ? ` · ${p.graded} decisions` : " · not rated yet"}</span><span>Won ${won}% · best ${d2(p.best_cents)}</span></span>`
          : html`<span class="plcard-foot"><span>Hands won · not graded</span><span>best ${d2(p.best_cents)}</span></span>`}</button>`;
    })}`);
    document.querySelectorAll("#lb-podium [data-uid], #lb-players [data-uid]").forEach((b) => b.addEventListener("click", () => {
      const p = players.find((x) => String(x.user_id) === b.dataset.uid);
      if (p) UI.openMyHands("", { user_id: p.user_id, name: p.name, is_me: p.is_me }, null, data.variant);
    }));
  }

  // who is up on whom: row = the player, column = the opponent, cell = what the
  // row player has won from (+) or lost to (−) that opponent, all sessions
  function openMatrix() {
    const data = U.club;
    if (!data || !(data.players || []).length) return toast("No hands played yet", "");
    const ps = data.players.filter((p) => (data.pairs || []).some((x) => x.from === p.user_id || x.to === p.user_id));
    if (ps.length < 2) return toast(`No money has changed hands at ${gameOf(data.variant).label} yet`, "");
    const net = {};
    (data.pairs || []).forEach((x) => { net[x.to + ":" + x.from] = x.cents; net[x.from + ":" + x.to] = -x.cents; });
    const peak = Math.max(1, ...Object.values(net).map(Math.abs));
    // (a cell's tint grows with the amount: its strength rides in data-vars, games.css .mx td.tint)
    const cell = (v) => {
      if (!v) return html`<td class="mx-zero">·</td>`;
      const a = 0.1 + 0.34 * Math.min(1, Math.abs(v) / peak);
      return html`<td class="num tint ${tone(v)}" data-vars="a:${a.toFixed(2)}">${signed(v)}</td>`;
    };
    const body = h("div", { class: "mx-wrap" });
    put(body, html`<table class="mx"><thead><tr><th class="mx-corner">won from →</th>${ps.map((p) => html`<th title="${p.name}">${avatar(p.name, p.name, "sm", p.avatar)}<span>${p.name}</span></th>`)}<th class="mx-total">Total</th></tr></thead><tbody>${ps.map((r) => {
        const total = ps.reduce((acc, c) => acc + (net[r.user_id + ":" + c.user_id] || 0), 0);
        return html`<tr><th data-uid="${r.user_id}" title="Open ${r.name}'s hands">${avatar(r.name, r.name, "sm", r.avatar)}<span>${r.name}${r.is_me ? " (you)" : ""}</span></th>${ps.map((c) => (c.user_id === r.user_id ? html`<td class="mx-self"></td>` : cell(net[r.user_id + ":" + c.user_id] || 0)))}<td class="num mx-total ${tone(total)}">${signed(total)}</td></tr>`;
      })}</tbody></table>
      <p class="muted note gap-top">Read across: a green cell is what that player has won from the player in the column. Every pot is traced layer by layer — side pots, split boards and quartered pots each go to who really paid for them.</p>`);
    body.querySelectorAll("th[data-uid]").forEach((th) => th.addEventListener("click", () => {
      const p = ps.find((x) => String(x.user_id) === th.dataset.uid);
      if (p) UI.openMyHands("", { user_id: p.user_id, name: p.name, is_me: p.is_me }, null, data.variant);
    }));
    const G = gameOf(data.variant);
    const api = openModal({ title: `Head to head · ${G.label}`, sub: `Who has won what from whom at ${G.label}, across every recorded session.`, body, wide: true, autofocus: false, buttons: [{ label: "Close", cls: "primary" }] });
    api.modal.classList.add("xwide");
  }

  // every session on record; the club's owner can take test tables out of the stats
  function openSessions() {
    const body = h("div", { class: "db" });
    const paint = () => {
      const data = U.club || {}, rows = data.sessions || [];
      const manage = !!data.can_manage;  // (the club's owner — FE-006; the old "is_admin" name is gone, HGB-011)
      put(body, html`${manage ? html`<p class="muted note">Excluding a session takes it out of everyone's stats, hand lists and head-to-head. Nothing is deleted — you can restore it here any time. An open table is closed first.</p>` : ""}
        ${rows.length ? rows.map((x) =>
          html`<div class="sess ${x.excluded ? "excluded" : ""}" data-id="${x.id}"><div><b>${x.name}</b>${x.open ? html` <span class="pill live">Open</span>` : ""}${x.excluded ? html` <span class="pill">Excluded</span>` : ""}<br><small>${fmtWhen(x.created_at)} · ${gameOf(x.variant).label} · ${d2(x.bb_cents)} bb · ante ${d2(x.ante_cents)}</small></div><small class="opt">${x.hands} hand${x.hands === 1 ? "" : "s"}</small><small class="opt">${x.players} player${x.players === 1 ? "" : "s"}</small><span class="sess-act"><button class="btn sm" data-open="${x.id}" type="button">Open</button>${manage ? html`<button class="btn sm ${x.excluded ? "" : "danger"}" data-ex="${x.id}" data-on="${x.excluded ? "0" : "1"}" type="button">${x.excluded ? "Restore" : "Exclude"}</button>` : ""}</span></div>`)
          : html`<div class="muted empty-note">No sessions yet.</div>`}`);
      body.querySelectorAll("[data-open]").forEach((b) => b.addEventListener("click", () => { api.close(null); C().openTable(b.dataset.open, true).catch((e) => toast(e.message, "err")); }));
      body.querySelectorAll("[data-ex]").forEach((b) => b.addEventListener("click", async () => {
        const on = b.dataset.on === "1", row = rows.find((x) => x.id === b.dataset.ex);
        if (on) {
          const ok = await confirmDialog({ title: `Exclude “${row.name}”?`, text: `Its ${row.hands} hand${row.hands === 1 ? "" : "s"} stop counting toward anyone's profit, accuracy and head-to-head${row.open ? ", and the table is closed (everyone is cashed out)" : ""}. You can restore it later.`, okLabel: "Exclude from stats", danger: true });
          if (!ok) return;
        }
        b.disabled = true;
        try {
          await C().j(`/games/api/tables/${encodeURIComponent(b.dataset.ex)}/exclude`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ on }) });
          toast(on ? "Session excluded from the stats" : "Session restored", "ok");
          await loadClub(true); U.lobbySig = ""; C().loadLobby().catch(() => {});
          paint();
        } catch (e) { b.disabled = false; toast(e.message, "err"); }
      }));
    };
    const api = openModal({ title: "All sessions", sub: "Every table on record, newest first.", body, wide: true, autofocus: false, buttons: [{ label: "Close", cls: "primary" }] });
    paint();
  }

  // ------------------------------------------------------------ the lobby's wiring
  UI.onInit.push(() => {
    $("c-open").addEventListener("click", () => openCreate());
    $("lb-myhands").addEventListener("click", () => UI.openMyHands(""));
    $("lb-h2h").addEventListener("click", openMatrix);
    $("lb-game").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-v]"), cid = C().G.clubId;
      if (!b || !cid || b.classList.contains("on")) return;
      setClubGame(cid, b.dataset.v);
      $("lb-game").querySelectorAll("button").forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-pressed", String(x === b)); });
      loadClub(true);
    });
    $("lb-allsess").addEventListener("click", openSessions);
    // (the pasted text stays in the box until it leads somewhere — HGL-007)
    const tryLink = async (id) => { const box = $(id); if (await joinByLink(box.value)) { box.value = ""; box.blur(); } };
    $("join-form").addEventListener("submit", (e) => { e.preventDefault(); tryLink("join-input"); });
    $("club-switch").addEventListener("click", (e) => openClubMenu(e.currentTarget));
    $("club-settings").addEventListener("click", openClubSettings);
    $("club-invite").addEventListener("click", async () => {
      const cid = C().G.clubId;
      try {
        const v = await C().j(`/games/api/clubs/${encodeURIComponent(cid)}`);
        if (v.invite_code) copyText(inviteUrl(v.invite_code), `Invite link to ${v.name} copied`, "Club invite link");
      } catch (err) { toast(err.message, "err"); }
    });
    $("wel-create").addEventListener("submit", (e) => {
      e.preventDefault();
      const name = $("wel-name").value.trim();
      if (!name) { toast("Give the club a name", "err"); $("wel-name").focus(); return; }
      createClub(name);
    });
    $("wel-join").addEventListener("submit", (e) => { e.preventDefault(); tryLink("wel-link"); });
  });

  Object.assign(UI, { renderLobby, renderJoinReqs, loadClub, currentClub, copyInvite, openInvite, openClubGate, openClubSettings, statsSince, periodLabel, clubPeriod });
})();
