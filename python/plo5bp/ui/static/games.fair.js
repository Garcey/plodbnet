"use strict";
// Home games — the verifiable shuffle, the PLAYER'S half.
//
// The spec (and a second, independent implementation) is plo5bp/ui/fairdeal.py:
// the server SEALS a shuffled deck before anyone contributes; every seated
// device commits to a secret random number and reveals it only after it has seen
// the frozen list of everyone's commitments under that same seal; the revealed
// numbers re-permute the sealed deck; and every card this player is ever shown
// arrives with a proof that it is the card the seal + the cut put in that slot.
//
// So this file does three things, all automatically:
//   1. takes part   — commit, check, reveal (never reveal under a changed seal);
//   2. checks       — the transcript of each hand and EVERY card on this screen;
//   3. tells        — a shield in the top bar; anything that fails is loud.
// Nothing here is trusted by the server and nothing here trusts the server.
(function () {
  const HG = (globalThis.HG = globalThis.HG || {});
  const SPEC = "wrapgto-fair-v1";
  const DECK = 52;

  // ------------------------------------------------------------ SHA-256 (sync)
  // (crypto.subtle is async and missing on plain-http LAN addresses; every input
  // in the spec is a short ASCII string, so a small synchronous one is simpler)
  // (literal tables: Math.cbrt / Math.sqrt are not guaranteed correctly rounded across engines)
  const H0 = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);
  const K = new Uint32Array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ]);
  function sha(text) {
    const len = text.length, total = ((len + 9 + 63) >> 6) << 6;
    const b = new Uint8Array(total);
    for (let i = 0; i < len; i++) {
      const c = text.charCodeAt(i);
      if (c > 127) throw new Error("fair: non-ASCII input");
      b[i] = c;
    }
    b[len] = 0x80;
    const bits = len * 8;
    b[total - 4] = (bits >>> 24) & 255; b[total - 3] = (bits >>> 16) & 255; b[total - 2] = (bits >>> 8) & 255; b[total - 1] = bits & 255;
    const h = new Uint32Array(H0), w = new Uint32Array(64);
    const rotr = (x, r) => (x >>> r) | (x << (32 - r));
    for (let off = 0; off < total; off += 64) {
      for (let i = 0; i < 16; i++) w[i] = (b[off + 4 * i] << 24) | (b[off + 4 * i + 1] << 16) | (b[off + 4 * i + 2] << 8) | b[off + 4 * i + 3];
      for (let i = 16; i < 64; i++) {
        const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
        const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
        w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
      }
      let [a, bb, c, d, e, f, g, hh] = h;
      for (let i = 0; i < 64; i++) {
        const t1 = (hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i]) >>> 0;
        const t2 = ((rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & bb) ^ (a & c) ^ (bb & c))) >>> 0;
        hh = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = bb; bb = a; a = (t1 + t2) >>> 0;
      }
      h[0] += a; h[1] += bb; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    let out = "";
    for (let i = 0; i < 8; i++) out += ("00000000" + h[i].toString(16)).slice(-8);
    return out;
  }

  // ------------------------------------------------------------------ the spec
  const isHex64 = (x) => typeof x === "string" && /^[0-9a-f]{64}$/.test(x);
  const cardCommitment = (hid, pos, salt, card) => sha(`${SPEC}|card|${hid}|${pos}|${salt}|${card}`);
  const sealOf = (hid, comm) => sha(`${SPEC}|seal|${hid}|` + comm.join(""));
  const nonceCommitment = (hid, seal, seat, nonce) => sha(`${SPEC}|nonce|${hid}|${seal}|${seat}|${nonce}`);
  const lockOf = (hid, seal, locked) => sha(`${SPEC}|lock|${hid}|${seal}|` + locked.map(([s, c]) => `${s}:${c}`).join(","));
  const cutOf = (hid, lock, reveals) => sha(`${SPEC}|cut|${hid}|${lock}|` + reveals.map(([s, n]) => `${s}:${n}`).join(","));
  function permutation(cut) {
    const perm = [];
    for (let i = 0; i < DECK; i++) perm.push(i);
    let words = [], block = 0;
    const next = () => {
      if (!words.length) {
        const h = sha(`${SPEC}|stream|${cut}|${block++}`);
        for (let i = 0; i < 64; i += 8) words.push(parseInt(h.slice(i, i + 8), 16));
      }
      return words.shift();
    };
    for (let j = DECK - 1; j > 0; j--) {
      const span = j + 1, limit = Math.floor(4294967296 / span) * span;
      let w;
      do { w = next(); } while (w >= limit);
      const r = w % span, tmp = perm[j];
      perm[j] = perm[r]; perm[r] = tmp;
    }
    return perm;
  }
  class FairError extends Error {}
  // `mine` = what THIS device remembers from before it revealed: {seat, nonce, seal, lock}
  function verifyTranscript(tr, mine) {
    if (!tr || tr.spec !== SPEC) throw new FairError("unknown spec");
    const hid = String(tr.hand_id), comm = tr.commitments || [];
    if (comm.length !== DECK || !comm.every(isHex64)) throw new FairError("a sealed deck has 52 commitments");
    if (sealOf(hid, comm) !== tr.seal) throw new FairError("the commitments do not add up to the seal");
    const locked = (tr.locked || []).map(([s, c]) => [Number(s), String(c)]);
    for (let i = 1; i < locked.length; i++) if (locked[i][0] <= locked[i - 1][0]) throw new FairError("the lock list must name each seat once, in order");
    if (lockOf(hid, tr.seal, locked) !== tr.lock) throw new FairError("the lock does not match its list");
    const reveals = (tr.reveals || []).map(([s, n]) => [Number(s), String(n)]);
    if (reveals.length !== locked.length || reveals.some((r, i) => r[0] !== locked[i][0])) throw new FairError("every locked seat must have revealed");
    reveals.forEach(([s, n], i) => {
      if (!isHex64(n) || nonceCommitment(hid, tr.seal, s, n) !== locked[i][1]) throw new FairError(`seat ${s}: the revealed number does not open its commitment`);
    });
    if (cutOf(hid, tr.lock, reveals) !== tr.cut) throw new FairError("the cut does not follow from the revealed numbers");
    if (mine) {
      if (mine.seal && mine.seal !== tr.seal) throw new FairError("the seal changed after this device committed");
      if (mine.lock && mine.lock !== tr.lock) throw new FairError("the lock list changed after this device revealed");
      if (mine.nonce && !reveals.some(([s, n]) => s === mine.seat && n === mine.nonce)) throw new FairError("this device's number is not in the cut");
    }
    return permutation(tr.cut);
  }
  function verifyOpening(tr, perm, card, op, expectSlots) {
    const slot = Number(op && op.slot), pos = Number(op && op.pos), salt = String(op && op.salt);
    if (!(card >= 0 && card < DECK && slot >= 0 && slot < DECK)) throw new FairError("no such card or slot");
    if (perm[slot] !== pos) throw new FairError(`card ${card}: slot ${slot} is not where the cut put position ${pos}`);
    if (expectSlots && !expectSlots.includes(slot)) throw new FairError(`card ${card} was dealt from slot ${slot}, which is not its place`);
    if (cardCommitment(String(tr.hand_id), pos, salt, card) !== tr.commitments[pos]) throw new FairError(`card ${card} does not open the sealed position ${pos}`);
  }
  // every card on a state payload, with the slots it is allowed to come from
  function visibleCards(s) {
    const out = [], n = s.num_seats;
    (s.seats || []).forEach((seat, i) => (seat.hole || []).forEach((c) => {
      if (Number.isInteger(c) && c >= 0) out.push([c, [5 * i, 5 * i + 1, 5 * i + 2, 5 * i + 3, 5 * i + 4]]);
    }));
    [["a", 0], ["b", 5]].forEach(([k, off]) => {
      const bd = (s.board || {})[k] || {};
      (bd.flop || []).concat([bd.turn, bd.river]).forEach((c, m) => { if (Number.isInteger(c) && c >= 0) out.push([c, [5 * n + off + m]]); });
    });
    return out;
  }

  // ------------------------------------------------------------ device memory
  const F = (HG.fairState = {
    mem: {},      // hand_id -> {seat, nonce, seal, commit, lock, state}
    hands: {},    // hand_id -> {status, tr, perm, checked:{card:true}, loading, error, names, contributors, voids}
    current: null, alarmed: {}, tally: { mine: 0, others: 0, none: 0, failed: 0 },
  });
  const store = (() => { try { return globalThis.sessionStorage || null; } catch (_) { return null; } })();
  const MEM_KEY = "hg.fair.mem.v1";
  function loadMem() {
    try { const raw = store && store.getItem(MEM_KEY); if (raw) F.mem = JSON.parse(raw) || {}; } catch (_) { F.mem = {}; }
  }
  function saveMem() {
    const keys = Object.keys(F.mem);
    keys.slice(0, Math.max(0, keys.length - 60)).forEach((k) => delete F.mem[k]); // newest 60
    try { if (store) store.setItem(MEM_KEY, JSON.stringify(F.mem)); } catch (_) { /* private mode: memory only */ }
  }
  function randomHex32() {
    const a = new Uint8Array(32);
    globalThis.crypto.getRandomValues(a);
    let out = "";
    for (let i = 0; i < 32; i++) out += ("0" + a[i].toString(16)).slice(-2);
    return out;
  }
  const api = (gid, path) => `/games/api/tables/${encodeURIComponent(gid)}/fair/${path}`;
  const post = (gid, path, body) => HG.core.j(api(gid, path), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

  // ---------------------------------------------------------------- take part
  function commit(s, nx) {
    const m = (F.mem[nx.hand_id] = { seat: s.my_seat, nonce: randomHex32(), seal: nx.seal, state: "committing" });
    m.commit = nonceCommitment(nx.hand_id, nx.seal, m.seat, m.nonce);
    saveMem();
    return post(s.id, "commit", { hand_id: nx.hand_id, commit: m.commit }).then(
      () => { m.state = "committed"; saveMem(); },
      (e) => { if (e && e.status) { m.state = "refused"; saveMem(); } else delete F.mem[nx.hand_id]; }, // (network blip: try again on the next state)
    );
  }
  function reveal(s, nx) {
    const m = F.mem[nx.hand_id];
    if (!m || m.state !== "committed") return null;
    const locked = (nx.locked || []).map(([a, b]) => [Number(a), String(b)]);
    const inList = locked.some(([a, b]) => a === m.seat && b === m.commit);
    if (!inList) { m.state = "left-out"; saveMem(); return null; } // locked without us (we were not dealt in): nothing to reveal
    // NEVER reveal under a seal, or into a list, other than the ones committed to / shown.
    if (nx.seal !== m.seal || lockOf(nx.hand_id, nx.seal, locked) !== nx.lock) {
      m.state = "withheld"; saveMem();
      alarm(nx.hand_id, "The sealed deck or the lock list changed after this device committed. Its number was NOT revealed.");
      return null;
    }
    m.lock = nx.lock; m.state = "revealing"; saveMem();
    return post(s.id, "reveal", { hand_id: nx.hand_id, nonce: m.nonce }).then(
      () => { m.state = "revealed"; saveMem(); },
      (e) => { m.state = e && e.status === 409 ? "closed" : "committed"; saveMem(); },
    );
  }

  // -------------------------------------------------------------------- check
  function checkCards(s, h, rec) {
    if (!rec.perm) return;
    for (const [card, slots] of visibleCards(s)) {
      if (rec.checked[card]) continue;
      const op = (h.open || {})[String(card)];
      if (!op) { fail(h.hand_id, rec, "a card was shown without its proof"); return; }
      try { verifyOpening(rec.tr, rec.perm, card, op, slots); rec.checked[card] = true; }
      catch (e) { fail(h.hand_id, rec, e.message); return; }
    }
  }
  function fail(hid, rec, why) {
    if (rec.status !== "failed") F.tally.failed++;
    rec.status = "failed"; rec.error = why;
    alarm(hid, why);
  }
  function checkHand(s, h) {
    let rec = F.hands[h.hand_id];
    if (!rec) {
      rec = F.hands[h.hand_id] = { status: "pending", checked: {}, hand_no: h.hand_no, names: h.names || [], contributors: h.contributors || [], voids: h.voids || [] };
      const old = Object.keys(F.hands);
      old.slice(0, Math.max(0, old.length - 40)).forEach((k) => delete F.hands[k]);
    }
    F.current = h.hand_id;
    if (rec.status === "failed") return;
    if (!rec.tr && !rec.loading && (rec.tries || 0) < 6) {
      rec.loading = true; rec.tries = (rec.tries || 0) + 1;
      HG.core.j(api(s.id, String(h.hand_no))).then((tr) => {
        rec.loading = false;
        const m = F.mem[h.hand_id];
        const mine = m && (m.state === "revealed" || m.state === "revealing") ? m : null;
        try {
          if (tr.hand_id !== h.hand_id || tr.seal !== h.seal) throw new FairError("the transcript is for a different sealed deck");
          rec.perm = verifyTranscript(tr, mine);
          rec.tr = tr;
          rec.status = mine ? "mine" : (tr.locked || []).length ? "others" : "none";
          F.tally[rec.status]++;
          const cur = HG.core.G.state;
          if (cur && cur.fair && cur.fair.hand && cur.fair.hand.hand_id === h.hand_id) checkCards(cur, cur.fair.hand, rec);
        } catch (e) { fail(h.hand_id, rec, e.message); }
        paint(HG.core.G.state);
      }, () => { rec.loading = false; });
    }
    checkCards(s, h, rec);
  }

  function onState(s) {
    const f = s && s.fair;
    if (!f || !f.supported) { paint(s); return; }
    const nx = f.next;
    if (nx && Number.isInteger(s.my_seat) && nx.you && !nx.you.barred) {
      // (a seal this device has not committed to — new hand, redone shuffle, or the
      // same id under a different seal: always a fresh number, never a reused one)
      if (nx.stage === "commit" && (!F.mem[nx.hand_id] || F.mem[nx.hand_id].seal !== nx.seal)) commit(s, nx);
      else if (nx.stage === "reveal") reveal(s, nx);
    }
    if (f.hand) checkHand(s, f.hand);
    paint(s);
  }

  // --------------------------------------------------------------------- tell
  const LABEL = {
    mine: ["ok", "Verified shuffle", "Your device helped cut this deck, and every card you have been shown checks out."],
    others: ["mid", "Shuffle cut by others", "Other players' devices cut this deck; yours did not take part in this hand. Every card you have been shown checks out."],
    none: ["off", "Unverified shuffle", "Nobody's device took part in this hand's shuffle. The deck was sealed before the deal (no card could be swapped), but its order was the server's alone."],
    pending: ["mid", "Checking…", "Checking this hand's shuffle."],
    failed: ["bad", "SHUFFLE CHECK FAILED", ""],
  };
  function alarm(hid, why) {
    if (F.alarmed[hid]) return;
    F.alarmed[hid] = true;
    if (HG.ui && HG.ui.openModal) {
      HG.ui.openModal({
        title: "This hand's shuffle did not check out",
        sub: "Your device could not confirm that this hand was dealt fairly. Tell the table and keep this message.",
        body: `<div class="fair-alarm"><b>${HG.core.esc(why)}</b><small>Hand ${HG.core.esc(hid)}</small></div>`,
        buttons: [{ label: "Details", onClick: () => { openPanel(); } }, { label: "OK", cls: "primary" }],
      });
    }
  }
  function paint(s) {
    if (typeof document === "undefined") return;
    const btn = document.getElementById("tb-fair");
    if (!btn) return;
    const f = s && s.fair;
    const h = f && f.hand, rec = h && F.hands[h.hand_id];
    const show = !!(f && f.supported && (h || f.next));
    btn.hidden = !show;
    if (!show) return;
    const status = rec ? rec.status : (f.next && f.next.pending ? "shuffling" : "idle");
    const [tone, text] = status === "shuffling" ? ["mid", "Shuffling…"] : status === "idle" ? ["off", "Verified shuffle"] : LABEL[status];
    btn.className = "pill fair " + tone;
    btn.lastElementChild.textContent = text;
    btn.title = status === "idle" ? "The next deck is sealed; your device takes part in cutting it." : (LABEL[status] || ["", "", "Confirming the shuffle with the players' devices…"])[2] || (rec && rec.error) || "";
  }
  function openPanel() {
    const s = HG.core.G.state, f = s && s.fair;
    if (!f || !HG.ui) return;
    const esc = HG.core.esc;
    const h = f.hand, rec = h && F.hands[h.hand_id];
    const short = (x) => (x ? `${String(x).slice(0, 10)}…${String(x).slice(-6)}` : "–");
    const st = rec ? rec.status : "pending";
    const names = (h && h.names ? h.names : []).map(([, nm]) => esc(nm)).join(", ");
    const voids = Object.entries(f.void_counts || {});
    const body = document.createElement("div");
    body.className = "fair-panel";
    body.innerHTML =
      (h ? `<div class="fair-now ${LABEL[st][0]}"><b>${esc(LABEL[st][1])}</b><span>${esc(st === "failed" ? (rec.error || "") : LABEL[st][2])}</span></div>` +
        `<dl class="fair-facts"><dt>Hand</dt><dd class="num">#${h.hand_no}</dd><dt>Sealed deck</dt><dd class="num" title="${esc(h.seal)}">${esc(short(h.seal))}</dd>` +
        `<dt>Cut by</dt><dd>${names || "nobody's device"}</dd><dt>Cards checked</dt><dd class="num">${rec ? Object.keys(rec.checked).length : 0} of ${visibleCards(s).length} on your screen</dd>` +
        ((h.voids || []).length ? `<dt>Redone</dt><dd>${h.voids.map((v) => esc((v.names || []).join(", ") || v.reason)).join(" · ")}</dd>` : "") + `</dl>`
        : `<div class="fair-now mid"><b>The next deck is sealed</b><span>Your device takes part in cutting it when the hand is dealt.</span></div>`) +
      `<div class="fair-how"><h4>How it works</h4><ol>` +
      `<li>Before anyone contributes, the server <b>seals</b> a shuffled deck: it publishes a fingerprint of every card position that it cannot change afterwards.</li>` +
      `<li>Every seated device picks a secret random number and publishes only a fingerprint of it.</li>` +
      `<li>Only once it has seen everyone's fingerprints under that same seal does your device <b>reveal</b> its number.</li>` +
      `<li>The revealed numbers <b>re-shuffle the sealed deck</b>. Nobody — not the server, not the other players together — could know or choose the result, as long as your own number was random.</li>` +
      `<li>Every card you are shown comes with a proof that it is the card the seal and the cut put there. Cards nobody is shown stay sealed: a mucked hand stays mucked.</li></ol>` +
      `<p>A device that commits and then does not reveal forces a fresh deck. That is a visible re-roll: it is announced at the table with the player's name.</p>` +
      `<p class="muted">What this cannot do: stop the site's operator from looking at cards on the server. It proves the deal was random and unaltered — not that nobody peeked.</p></div>` +
      `<div class="fair-tally"><span>This session on this device</span><b class="num">${F.tally.mine}</b> cut by you · <b class="num">${F.tally.others}</b> by others · <b class="num">${F.tally.none}</b> unverified${F.tally.failed ? ` · <b class="num neg">${F.tally.failed} FAILED</b>` : ""}` +
      (voids.length ? `<br><span>Shuffles redone because a device did not confirm</span>${voids.map(([nm, n]) => `${esc(nm)} <b class="num">${n}</b>`).join(" · ")}` : "") + `</div>`;
    const buttons = [];
    if (rec && rec.tr) buttons.push({ label: "Copy transcript", onClick: async () => { await copy(JSON.stringify({ transcript: rec.tr, my_number: (F.mem[h.hand_id] || {}).nonce || null, my_seat: (F.mem[h.hand_id] || {}).seat }, null, 1)); return false; } });
    buttons.push({ label: "Close", cls: "primary" });
    HG.ui.openModal({ title: "Verified shuffle", sub: "Sealed deck + the players' cut — checked by your own device, every hand.", body, wide: true, autofocus: false, buttons });
  }
  async function copy(text) {
    try { await navigator.clipboard.writeText(text); HG.ui.toast("Copied — anyone can re-check it with the published method", "ok"); }
    catch (_) { HG.ui.openModal({ title: "Transcript", body: `<textarea class="input" style="width:100%;height:260px" readonly>${HG.core.esc(text)}</textarea>`, buttons: [{ label: "Done", cls: "primary" }] }); }
  }
  // history: check a finished hand's transcript + the cards this viewer may see
  async function checkPast(gid, handNo) {
    const tr = await HG.core.j(api(gid, String(handNo)));
    const m = F.mem[tr.hand_id];
    const perm = verifyTranscript(tr, m && m.state === "revealed" ? m : null);
    let n = 0;
    for (const [card, op] of Object.entries(tr.open || {})) { verifyOpening(tr, perm, Number(card), op, null); n++; }
    return { tr, cards: n, mine: !!(m && m.state === "revealed"), contributors: (tr.locked || []).length };
  }

  function init() {
    loadMem();
    if (typeof document === "undefined") return;
    const btn = document.getElementById("tb-fair");
    if (btn) btn.addEventListener("click", openPanel);
  }

  HG.fair = { init, onState, openPanel, checkPast, SPEC,
    __api: { sha, permutation, verifyTranscript, verifyOpening, visibleCards, nonceCommitment, lockOf, cutOf, sealOf, cardCommitment, FairError, F, commit, reveal } };
})();
