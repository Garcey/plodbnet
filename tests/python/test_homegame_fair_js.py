"""games.fair.js (the player's half of the verifiable shuffle), run in Node against
the Python implementation: same hashes, same permutation, same verdicts — and the
one behaviour that carries the guarantee: a device never reveals its number under
a seal or a lock list other than the ones it committed to / was shown."""
from __future__ import annotations

import hashlib
import json
import secrets
import shutil
import subprocess
from pathlib import Path

import pytest

from plo5bp.ui import fairdeal as fd

FAIR_JS = Path(__file__).resolve().parents[2] / "python" / "plo5bp" / "ui" / "static" / "games.fair.js"

HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const input = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const posts = [];
const ctx = { console, JSON, Math, Number, String, Object, Array, Set, Map, Promise, Error, Uint8Array, Uint32Array,
  parseInt, encodeURIComponent, crypto: require("crypto").webcrypto, setTimeout, };
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx, { filename: "games.fair.js" });
const HG = ctx.HG, A = HG.fair.__api;
HG.core = { G: { state: null }, esc: (x) => String(x),
  j: (url, opts) => { posts.push({ url, body: opts && opts.body ? JSON.parse(opts.body) : null }); return input.transcript && !opts ? Promise.resolve(input.transcript) : Promise.resolve({ ok: true }); } };
const out = { sha: input.strings.map((s) => A.sha(s)), perm: A.permutation("00".repeat(32)) };
const bad = (fn) => { try { fn(); return null; } catch (e) { return e instanceof A.FairError ? e.message : "WRONG ERROR " + e; } };

// 1. a Python-made transcript checks out, card by card
const tr = input.transcript, mine = input.mine;
const perm = A.verifyTranscript(tr, mine);
out.dealt_ok = input.cards.every(([card, op, slots]) => bad(() => A.verifyOpening(tr, perm, card, op, slots)) === null);
const [c0, op0] = input.cards[0];
out.caught = {
  other_card: bad(() => A.verifyOpening(tr, perm, (c0 + 1) % 52, op0, null)),
  wrong_slot: bad(() => A.verifyOpening(tr, perm, c0, op0, [51])),
  swapped_deck: bad(() => A.verifyTranscript(Object.assign({}, tr, { commitments: [tr.commitments[1], tr.commitments[0]].concat(tr.commitments.slice(2)) }), null)),
  not_my_number: bad(() => A.verifyTranscript(tr, Object.assign({}, mine, { nonce: "cd".repeat(32) }))),
  seal_replaced: bad(() => A.verifyTranscript(tr, Object.assign({}, mine, { seal: "ef".repeat(32) }))),
  chosen_cut: bad(() => A.verifyTranscript(Object.assign({}, tr, { cut: A.sha("chosen by the server") }), null)),
  dropped_player: bad(() => A.verifyTranscript(Object.assign({}, tr, { locked: tr.locked.slice(0, 1), reveals: tr.reveals.slice(0, 1) }), null)),
};

// 2. taking part: commit once, reveal only under the SAME seal and a list that has us
(async () => {
  const nx = { hand_no: 9, attempt: 1, hand_id: "T:9:1", seal: "aa".repeat(32), stage: "commit", pending: false, locked: [], lock: null, you: { seat: 2, barred: false } };
  const s = (next) => ({ id: "T", my_seat: 2, num_seats: 6, seats: [], board: {}, fair: { supported: true, next } });
  HG.fair.onState(s(nx)); HG.fair.onState(s(nx));
  await new Promise((r) => setTimeout(r, 5));
  const m = A.F.mem["T:9:1"];
  out.commits_posted = posts.filter((p) => p.url.endsWith("/fair/commit")).length;
  out.commit_matches = posts[0].body.commit === A.nonceCommitment("T:9:1", nx.seal, 2, m.nonce) && /^[0-9a-f]{64}$/.test(m.nonce);
  // (a) the server swaps the sealed deck before asking for the numbers
  const locked = [[0, "bb".repeat(32)], [2, m.commit]];
  const swapped = Object.assign({}, nx, { stage: "reveal", seal: "cc".repeat(32), locked, lock: A.lockOf("T:9:1", "cc".repeat(32), locked) });
  HG.fair.onState(s(swapped));
  await new Promise((r) => setTimeout(r, 5));
  out.revealed_under_swapped_seal = posts.some((p) => p.url.endsWith("/fair/reveal"));
  out.state_after_swap = m.state;
  // (b) an honest lock: reveal exactly the committed number, once
  m.state = "committed";
  const honest = Object.assign({}, nx, { stage: "reveal", locked, lock: A.lockOf("T:9:1", nx.seal, locked) });
  HG.fair.onState(s(honest)); HG.fair.onState(s(honest));
  await new Promise((r) => setTimeout(r, 5));
  const rv = posts.filter((p) => p.url.endsWith("/fair/reveal"));
  out.reveals_posted = rv.length;
  out.reveal_is_the_committed_number = rv.length === 1 && rv[0].body.nonce === m.nonce && rv[0].body.hand_id === "T:9:1";
  // (c) a lock list that leaves this device out: nothing to reveal, no alarm
  delete A.F.mem["T:9:1"]; posts.length = 0;
  HG.fair.onState(s(nx));
  await new Promise((r) => setTimeout(r, 5));
  const others = [[0, "bb".repeat(32)]];
  HG.fair.onState(s(Object.assign({}, nx, { stage: "reveal", locked: others, lock: A.lockOf("T:9:1", nx.seal, others) })));
  await new Promise((r) => setTimeout(r, 5));
  out.left_out_reveals = posts.filter((p) => p.url.endsWith("/fair/reveal")).length;
  // (d) a bad lock hash is refused too
  delete A.F.mem["T:9:1"]; posts.length = 0;
  HG.fair.onState(s(nx));
  await new Promise((r) => setTimeout(r, 5));
  const m2 = A.F.mem["T:9:1"], l2 = [[2, m2.commit]];
  HG.fair.onState(s(Object.assign({}, nx, { stage: "reveal", locked: l2, lock: "dd".repeat(32) })));
  await new Promise((r) => setTimeout(r, 5));
  out.bad_lock_reveals = posts.filter((p) => p.url.endsWith("/fair/reveal")).length;
  process.stdout.write(JSON.stringify(out));
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_browser_verifier_agrees_with_the_python_one(tmp_path):
    sd = fd.SealedDeck.create("game:41:2", 6)
    nonces = {1: secrets.token_hex(32), 4: secrets.token_hex(32)}
    sd.set_lock({s: fd.nonce_commitment(sd.hand_id, sd.seal, s, n) for s, n in nonces.items()})
    deck = sd.finish(nonces)
    cards = []
    for k in range(5):  # seat 4's hole cards, then board A
        c = deck[fd.hole_slot(4, k)]
        cards.append([c, sd.opening(c), list(range(20, 25))])
    for m in range(5):
        c = deck[fd.board_slot(6, "a", m)]
        cards.append([c, sd.opening(c), [30 + m]])
    strings = ["", "abc", "a" * 55, "a" * 56, "a" * 63, "a" * 64, "a" * 119, "a" * 120, "wrapgto-fair-v1|" * 70]
    payload = {
        "strings": strings, "transcript": sd.public(), "cards": cards,
        "mine": {"seat": 4, "nonce": nonces[4], "seal": sd.seal, "lock": sd.lock},
    }
    (tmp_path / "in.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "harness.js").write_text(HARNESS, encoding="utf-8")
    run = subprocess.run(["node", str(tmp_path / "harness.js"), str(FAIR_JS), str(tmp_path / "in.json")],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    assert out["sha"] == [hashlib.sha256(s.encode()).hexdigest() for s in strings], "SHA-256 (padding edges included)"
    # (the Python side of this known answer is pinned as a literal in test_homegame_fair.py)
    assert out["perm"] == fd.permutation("00" * 32), "the same cut must give the same deck in every implementation"
    assert out["dealt_ok"] is True
    assert all(isinstance(v, str) and "WRONG" not in v for v in out["caught"].values()), out["caught"]
    # taking part
    assert out["commits_posted"] == 1 and out["commit_matches"] is True
    assert out["revealed_under_swapped_seal"] is False and out["state_after_swap"] == "withheld"
    assert out["reveals_posted"] == 1 and out["reveal_is_the_committed_number"] is True
    assert out["left_out_reveals"] == 0 and out["bad_lock_reveals"] == 0
