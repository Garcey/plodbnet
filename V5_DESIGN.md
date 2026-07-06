# V5 Design — PLO5 Double-Board Bomb Pot

**Status**: **IMPLEMENTED 2026-07-06** (same day as the research; training not
yet launched). Landed: the K=3 mixture head (`--sizing-head mixture`,
head_version 4), obs v2 core +29 dims (OBS_DIM 1020; P1 per-board outcome, P3
blockers, P4 effective price, SPR-log1p — P2/P5 deferred per user), bug fixes
B1/B2/B3(v5-only)/B5/B8/B9 + `--value-clip` flag (B4), the EMA-magnet
persistence (`model_ema`), the critic Q-aux dueling head + `--q-aux-coef`,
per-tier entropy/F/T/R under mix-configs, and `scripts/convert_v4_to_v5.py`
(function-preservation verified against the real stub.pt). B6 reclassified
not-a-bug (deliberate gentle-restart warmup; comment fixed). Rust obs encoder
force-disabled pending its obs-v2 port. See CLAUDE.md "v5" section for the
operator surface; tests: test_sizing_mixture.py, test_obs_v2.py,
test_per_tier_mix.py (+ updated parity suites).

**Post-implementation verification (2026-07-06)**: a 15-check numerical
harness (sampling-frequency==marginal across all-legal/short-shove/capped,
act↔evaluate parity, exact marginal entropy, converter decomposition,
self-KL==0, Q-head consistency) all pass, plus three adversarial subsystem
reviewers (network+ppo math / obs encoding / train wiring). Network/ppo:
NO bugs (mixture normalization, nan-safety, KL, Q-aux isolation, gradient
paths all confirmed). Train wiring: **two real narrow-trigger bugs FIXED** —
(1) the converter didn't strip `model_ema`, so a magnet-on v4 source would
crash a magnet-on v5 warm-start on a shape mismatch; (2) the `entropy_coef`
broadcast under mix-configs clobbered a same-write per-tier `tier_ent`
override. Also added: bf16-autocast mixture parity test (the pod's train
dtype; the CPU harness is f32), stale-comment fix. `check_mixture_usage.py`
added — the v5 success metric (weight-entropy / effective-K / μ-spread /
multi-modal% census; run it during training the way check_sizing_dist.py
tracked v4). Real 2048×4 v5 (converted from stub.pt) served end-to-end
through the UI (head_version 4, mixture payload, model_loaded True).
Known latent (not fixed, no trigger today): `build_actor_from_state_dict`
doesn't restore `mix_weight_floor` — moot while it's a fixed 0.03 constant
with no CLI flag; if a floor flag is ever added, store it in the checkpoint.

Originally: proposal produced from five parallel research
lanes: engine/game-dynamics, network/sizing-head, training pipeline, observation
encoding, and external literature. Every code claim below was verified against
HEAD by at least one lane; the highest-impact claims (entropy config, value_clip,
kl-anchor crash, entropy-gradient paths) were independently re-verified.

**The goal** (user): a near-perfect study model for serious players. Live-play
critiques of v4: (1) sizings are "face up" (strength-revealing), (2) decisions
imperfect, (3) in check-or-bet spots it mixes significant betting with hands
that should be pure checks.

---

## 0. TL;DR

v4's two symptoms have **identified, mostly-verified mechanical causes**, and
several of them are fixable without a new architecture:

1. **The entropy picture (corrected 2026-07-06 against the pod log).** vFour4
   WAS annealed — manually, via `runs/anneal_control.json` `tier_ent` edits
   (the correct key): 0.45 → 0.30 → 0.27 (u263) → 0.24 (u310) → 0.21 → 0.18 →
   0.15 → 0.12 (u466) → **0.10 at u550**, then ~370 of its 917 total updates
   at 0.10. An earlier draft of this doc claimed "flat 0.45 lifelong" — wrong.
   Implications: (a) the promoted checkpoints your testers played reflect a
   long 0.10-coef stretch, so **entropy financing is a residual suspect for
   the check-bet over-mixing, not the primary one — pool best-response
   cycling moves to #1** (§1.2); (b) remaining pure-entropy headroom to v1's
   mature floors (0.04–0.085) is modest, and pushing below 0.10 without a
   replacement mixing pressure risks the documented passivity failure — which
   is exactly the case for the EMA magnet (§4.1). What remains true
   config-side: under `--mix-configs` the F/T/R-gated AUTO-anneal is
   structurally off (train.py:911/1420/1503), only `tier_ent["clubgg"]` is
   consumed so all tiers move in lockstep (the per-tier "deep needs more
   entropy" design silently doesn't exist in mix mode), and the
   `entropy_coef` control key is a silent no-op in mix mode (consumed only on
   the legacy single-config branch, train.py:1400-1406).
2. **The entropy bonus actively subsidizes end-anchor (pot/min) sizing**
   (verified gradient path, network.py:505-509): Beta differential entropy is
   ≤ 0 for α,β ≥ 1, and the `beta_h_eff` term weights it by **non-detached**
   `anchor_probs` — so maximizing the bonus pays (μ,s) to move anchor mass off
   refinable interior anchors onto the atoms. Second mechanism: under the s-cap
   the maximum-entropy anchor shape is a **U** (29% on each end anchor), so raw
   anchor-entropy pressure also points at min/pot. Both feed pot-heavy,
   face-up sizing.
3. **The v4 head makes size a monotone function of one latent (μ)** — size →
   strength is invertible almost by construction. It provably cannot mix an
   interior size with anything: best achievable joint mass on {33%, pot}
   without flooding between ≈ 0.008. (It CAN express a {min, pot} two-point mix
   via tail absorption — a U-shape — but nothing involving interior anchors.)
4. **The network's only strength oracle misranks draws and hides splits.** The
   12 opp-outcome dims are current-rank-only (engine.rs:1255-1257): a wrap +
   nut-flush-draw monster reads "behind almost everything" in the strongest
   input features; win-one-lose-one (the most common double-board outcome) is
   excluded as residual; and there is **no runout equity anywhere in the obs**.
   Plus: the SPR feature clips at 4.0, so the *entire deep tier* (true flop SPR
   5.4–13.9 at 6-max) is constant exactly where depth matters
   (encoding.py:779).
5. **Nothing anchors self-play toward equilibrium.** The pool is 8 snapshots
   spanning only ~40 updates of near-identical selves; the literature (ICLR
   2026 large-scale study) says pools alone are insufficient and regularized
   policy gradient (PPO + KL-to-slow-reference, i.e. MMD) is the consensus fix
   — which is exactly our dormant `--kl-anchor-coef` flag, except it **crashes
   on v4 heads** (verified, reproduced) and has a second latent bug (p_raise
   not detached).

**Recommended v5 scope** (detail in §2–§5, sequencing in §7):

| Piece | What | Attacks |
|---|---|---|
| Eval stack | duplicate/seat-rotated matches + exploiter-probe time series + LBR-lite | makes "x% quality" measurable at all (prerequisite) |
| W1 training fixes, v4 arch | repair per-tier anneal machinery under mix-configs, fixed kl-anchor EMA magnet, shift mixing pressure entropy→magnet then push temperature below 0.10, value_clip A/B | symptom 3, pool cycling, sample efficiency |
| v5 sizing head | K=3 mixture of discretized logistics over the same 11 anchors | symptom 1 (face-up), solver-style size menus |
| Obs v2 (+~30 core dims) | per-board A/T/B + split/double-tie decomposition, blockers-to-nuts, effective-price block, log1p companions; MC runout equity DEFERRED (user call 2026-07-06, §3.2) | symptom 2 (imperfect decisions) |
| Q-critic (VRPO) | CentralCritic → Q(s, gate/anchor) + Expected-SARSA(λ) advantages | mixed-node gradient noise = both symptoms' variance floor |

---

## 1. Diagnosis: why v4 shows these symptoms

### 1.1 What the game actually rewards (engine lane, rules verified sound)

No rule-level bugs found: PL math, TDA-style reopen rules, side-pot layering,
per-board exactly-2-of-5 eval, auto-runout, zero-sum splits all check out
(engine.rs, double_board.rs; the module's own tests cover scoop/chop/side-pot
cases). Key verified structure:

- Ante-only (3bb × seats), both 5-card boards pre-dealt, action starts at the
  flop, one betting round per street covering both boards. No preflop betting
  → **symmetric random ranges at the root**: no range advantage, no "range
  betting" — position and realized hand strength are everything.
- **Flop SPR by tier** (pot = 3n bb at n seats): clubgg bulk (20–40bb) 6-max ≈
  **0.9–2.1**; clubgg_deep bulk (40–80bb) ≈ 2.1–4.3; deep (100–250bb) 6-max ≈
  **5.4–13.7**, HU 16–41. At clubgg depth the game is nearly a
  one-decision/geometric game ({check, ⅓–½, pot} suffices); real multi-size
  menus and raise trees live at SPR > 4 — i.e. the deep tier and HU branches.
- **Split-pot economics**: HU, betting with a one-board lock and no
  second-board equity is ~EV-neutral on the bet itself — profit is the *scoop
  differential*. Multiway it flips: a one-board lock bet returns (m+1)/2 per
  unit vs m callers — thin one-board value bets multiway are correct, the same
  hands HU are the game's canonical indifference class. Raising wars are
  scoop-vs-scoop; quarter/freeroll spots (nut-on-A + live-on-B) are the key
  raise class.
- **Air is rare, fold equity multiway is terrible**: 10 two-card combos ×
  2 boards per hand; in a 6-way pot ~everyone connects with something, so five
  simultaneous folds vs a pot bet happen ~1–3% of the time. **Theoretical
  verdict on symptom 3: wide bet-mixing with both-board-weak hands in multiway
  check-or-bet nodes is NOT plausibly equilibrium** — this game's structure
  argues for *less* weak-hand betting than NLH intuition. Legitimate mixing is
  confined mostly to: HU/3-way polar river nodes, the HU one-board-lock class,
  SPR 1–2 bet-now-vs-shove-turn equivalences, and IP flop stabs with
  backdoor-scoop hands. The user's read matches theory; the mixing is most
  plausibly a training artifact (§1.2).
- Caveats for study-tool calibration: the model plays **rake-free** (thin
  margins slightly optimistic vs raked live games). Training seats are
  **uniform 2–6 — an intentional user choice** (2026-07-06): the target is a
  rounded model that also covers 3–4-handed, deeper home games, not a
  ClubGG-tuned specialist. Keep uniform; the `_CLUBGG_SEAT_WEIGHTS` table
  stays available if a ClubGG-specialist stem is ever wanted.

### 1.2 Symptom 3 (bets hands that should be pure checks) — cause ranking

1. **[likely — now the primary suspect] Pool best-response cycling.** 8
   snapshots × every-5-updates = a 40-update window of near-identical selves;
   nothing anchors toward equilibrium (kl-anchor off, no long-run averaging).
   If the population over-folds vs bets, betting air genuinely profits *vs
   the pool* — and that profit persists at ANY entropy coef, which fits the
   corrected fact that over-mixing survived ~370 updates at coef 0.10. Not
   equilibrium — exploitable by calling down. Diagnostic:
   `scripts/exploitability.py` exploiter probe vs frozen vFour4 (a
   calling-station-flavored exploit confirms it); plus measure pool
   fold-to-bet frequencies via `eval.run_match` between snapshots.
2. **[residual, was primary in an earlier draft — corrected §0.1] Entropy
   financed.** vFour4's coef was walked to 0.10 by u550. 0.10 is still 1.2–
   2.6× v1's mature floors and softmax gates can't hit exact purity, so a
   floor residue remains plausible but can no longer carry the whole
   symptom. The discriminating diagnostic is free either way: pull
   `gate_distribution` from the study API across many known-pure-check hands
   — broadly flat bet mass (~0.1–0.3 everywhere) implicates the bonus/floor;
   concentration on specific blocker/air classes implicates learned
   pool-exploitation (cause 1).
3. **[verified mechanism, magnitude unknown] Weak per-node gradient.** The
   check-vs-bet EV gap for weak hands multiway is small in bb while outcome
   variance is enormous (6 players, split pots); PPO's signal for "pure check"
   is slow. This is what the Q-critic (§5.3) attacks.
4. **Not the cause**: critic optimism (no structural evidence — CentralCritic
   conditions on all holes; display head is detached), serving mismatch (study
   recs are argmax — that *hides* mixing rather than creating it).

Note the counter-lesson from project history (2026-06-07): a behavior that
looks wrong vs live rec-players can be *correct* vs a GTO-ish field. Here the
direction is reversed (the model is MORE aggressive than the user's read, in
spots where theory sides with the user), so the artifact explanation stands —
but run diagnostic 1 before concluding any specific spot is broken.

### 1.3 Symptom 1 (face-up sizings) — three compounding mechanisms

1. **Structural**: one (μ,s) logistic per node → μ is a smooth monotone
   function of strength; size leaks strength by construction (§0.3).
2. **Entropy-shape artifact**: the bonus's end-anchor subsidy (§0.2) purifies
   pot-bets further.
3. **Presentation**: the UI recommendation is deterministic argmax (gate
   argmax + anchor argmax + Beta mean; server.py:1268-1285) — even a genuinely
   mixing policy *presents* as a single face-up size unless the user reads the
   anchors histogram. Training opponents do sample stochastically
   (rollout.py:612/1033), so self-play could in principle punish tells, but
   40-update-recent clones exert weak pressure.

Fixes map: mixture head (representation, §2) + EMA-magnet/pool diversity
(pressure, §4) + menu-first UI presentation (§6.3).

---

## 2. v5 sizing head: K=3 mixture of discretized logistics

**Why K=3**: user's read — most spots want ≤2 sizes, err toward too many.
Collapse to fewer components is graceful (w→one-hot IS v4), so K=3 costs only
4 extra head outputs vs K=2 and covers the three-size nodes deep play needs.

### 2.1 Architecture (network lane, worked against HEAD)

The v2 `act`/`evaluate` plumbing is head-agnostic through `_anchor_dist`
(network.py:365-374) — v4 exploited exactly this, and v5 does too. **v5 is a
~30-line subclass overriding `__init__`/`forward`/`_anchor_dist`:**

```python
size_head = nn.Linear(hidden, 3*K)              # K×(mu_raw, s_raw) + K mix logits
mu_k = c + (c+2)*tanh(raw[..., 0:K])            # c = (count-1)/2, same as v4
s_k  = 0.3 + (5.0-0.3)*sigmoid(raw[..., K:2K])  # keep v4's floor/cap per component
w    = eps + (1 - K*eps)*softmax(raw[..., 2K:3K])   # eps ≈ 0.02-0.05 weight floor
P_k  = _discretized_logistic_probs(mu_k, s_k, legal[..., None, :])  # (B, K, 11)
probs = (w[..., :, None] * P_k).sum(-2)
return Categorical(probs=probs)
```

Because anchors are **discrete**, the mixture marginal is just an 11-way
Categorical: log-prob of a stored anchor = `log Σ_k w_k P_k(a)` and entropy =
exact marginal entropy over ≤11 legal anchors — **both exact closed forms, no
MC, no PPO-machinery changes**. No component index is sampled or stored
(sampling the marginal ≡ sample-component-then-anchor), so the rollout buffer
schema is untouched and `evaluate`'s replay contract (ratio = 1.0 at epoch
start) holds with zero changes.

**Refine slider**: keep the single per-anchor Beta refine conditioned on the
sampled anchor (inherit `_gather_refine` untouched). Per-component refine
would break the exact-marginal property, triple params for ±5%-pot brackets,
and break `score_move_v2`/UI payloads.

**Entropy terms**: `Ha` automatically becomes the marginal mixture entropy —
component-weight uncertainty enters exactly insofar as it spreads the anchor
pmf, which is the correct generative quantity. **Do NOT add an H(w) bonus**
(that is v2's flat-collapse analog: a maximized H(w) pins w uniform and blurs
the menu into a fat unimodal average). Log `Hw = H(softmax(mix_logits))` as a
pure diagnostic next to Hg/Ha/Hb. The `p_raise.detach()` isolation
(network.py:509) and `sizing_entropy_scale` algebra (ppo.py:310-316) carry
over unchanged.

### 2.2 Failure modes and mitigations (v2-postmortem lens)

| Mode | Verdict | Mitigation |
|---|---|---|
| w collapses to one component | graceful degradation to exactly v4 — acceptable | none needed |
| Dead components (w_k→0, gradient starvation) | real risk | the ε weight floor (inside `_anchor_dist`, so act/evaluate agree); passive — doesn't fight advantages like an Hw bonus would |
| μ's coalescing (mode collapse to unimodal) | likely default early | **init spread**: zero weights, biases → μ ≈ (−0.3, 5, 10.3) (min / half-pot / pot region), mix-logit biases (2.0, −1.0, −1.0) → w ≈ (0.91, 0.045, 0.045); optionally freeze mix logits + μ spread first ~50 updates |
| Weight-entropy pinning w uniform | only if Hw is bonused | Hw coef = 0 (diagnostic only). Fallback if components die despite ε-floor: tiny Hw floor (external-lit suggestion), but try the floor first |
| Permutation/identifiability | harmless for PG training | log components sorted by μ |
| Heavier importance ratios than v4 | expected: between v4's 0.01–0.10 and v2's 1–3, much nearer v4 (per-component gradients weighted by responsibilities are smooth and self-attenuating; the ε-floor bounds `P(a) ≥ ε·P_k(a)`) | existing `target_kl 0.5` soft-stop + `kl_hard` rollback; keep per-head anchor-KL logging (works unchanged — `anchor_log_prob` stores the marginal) |
| Per-component s floor | keep 0.3 | 68% max interior commit per component is sharper than any equilibrium mix needs; the floor is the anti-spike property that made v4 trainable |

### 2.3 Compat surface

- `head_version = 4`; **name the new tensor distinctly** (e.g. `mix_head.weight`)
  and sniff it FIRST in `model_class_for_state_dict` (network.py:647-661) — if
  v5 reuses the literal name `size_head.weight` with shape (3K, hidden), the
  sniffer returns V4, `load_state_dict` shape-fails, and the UI silently
  serves a random-init placeholder (server.py:166-175).
- UI: nearly free. `_recommendation_v2` and trainer's distribution path
  already go through `model._anchor_dist(...).probs`; the 11-bin histogram
  renders any shape; `score_move_v2` consumes `anchor_probs` head-agnostically
  and actually gets FAIRER (a second menu size scores ~100 instead of being
  penalized against the lone hump). Add an optional `mixture` payload block
  (per-component μ/s/w) so the UI can annotate the menu (§6.3).
- Bundle with the head change (same file, same reasons): fix W1/W2 kl-anchor
  bugs (§6.1) and **detach the `anchor_probs` weight in `beta_h_eff`**
  (network.py:505-507) to kill the end-anchor subsidy.

### 2.4 Warm-start v4→v5 (recommend WARM, needs user sign-off)

Offline converter script (train.py's head_version guard stays strict): copy
torso/gate/refine/value/critic verbatim; copy v4's `size_head` rows into
component 0; spread components 1–2 per §2.2; emit `head_version: 4`.
Resulting policy ≈ v4 with a ~9% smear from the minor components (push w₀ to
0.98 if a smaller smear is wanted) — reads as a modest entropy re-warm, not a
policy break. **This composes with obs growth**: appended input dims warm-start
by zero-padding the first-layer columns of the actor AND critic
(function-preserving — new columns contribute exactly 0 at step 0, §3.3), so
"new dims + new sizing head" together still reproduce v4's policy at step 0;
the combination is mechanically routine, not impossible. The 2026-07-03
mandate forbids cross-**variant** warm-starts (game-equity rationale); a
same-variant head/obs upgrade is a different axis — **flagged as a user
decision, not assumed**.
Cold-start fallback if the warm run shows early anchor-KL thrash or immediate
μ coalescing. Note: warm-start pool seeding will skip v4 siblings
(selfplay.py:182-184 filters on head_version) — either run the converter over
pool members too, or accept pure self-play until the first snapshot.

---

## 3. Observation v2

### 3.1 What exists today (991 dims) and the three gaps

Digests that already exist: made-hand category ×2 boards, pair-position
counts, straight/flush/SF outs + nut distances (38×2), cross-board block (28,
incl. the JT-freeroll detector), 12 opp-outcome fractions. The gaps
(verified):

- **Gap 1 — no runout equity anywhere.** opp-outcome is current-rank
  dominance only ("No runout sampling on flop/turn", engine.rs:1257). Draws
  are systematically misranked by the net's strongest features; outs counts
  exist but the outs→equity(street, SPR, two boards) conversion is left to
  the MLP. Most plausible single cause of "imperfect decisions" and face-up
  draw sizing.
- **Gap 2 — no per-board marginals / split probability.** The 12 dims encode
  joint scoop/quarter vs one opponent; win-one-lose-one (the modal outcome)
  is lumped into residual. "A locked, B coinflip" (freeroll — bet huge) vs
  "55/45 both" (thin) can read the same.
- **Gap 3 — blockers-to-nuts are raw-only.** `nut_flush_draw_outs` requires
  holding the draw; `flush_nut_distance` requires the made flush. Bare-ace
  bluff selection has no digest.

On the opp-outcome opponent universe (k∈{2,3,4}) — resolved 2026-07-06 after
user review: these are **combo-dominance densities by design**, not opponent
simulations. The exhaustive k=2 row answers "what fraction of 2-card combos
scoop/quarter us" (e.g. double 3-heart unpaired boards without two hearts →
every heart-heart combo is in the opp-scoops fraction, a crisp fold signal;
nut straight on A tied by a combo with better on B → the opp-quarters
bucket). That reading matches the code exactly, and phase-7 observations
credited this encoding for correct quartering/freeroll discipline — the
density signals demonstrably work. A *different* quantity — field
probability, P(an actual best-2-of-5-per-board opponent scoops) — is not in
the obs and differs from the density by a texture-dependent selection factor
(3-flush board: ~5.5% of 2-card combos are flushes vs ~35% of real 5-card
holdings, a ~6× gap that varies by hand class), which the net must learn
texture-conditionally from rewards — and the quartering evidence suggests it
substantially has. Consequently **P5 (k=5 field-probability row) is
ablation-tier, same treatment as P2**: test only if post-v5 evals still show
multiway thin-spot imprecision. Note P1 below is the k=2 density feature
*decomposed per board* (same exhaustive loop, per-board A/T/B + explicit
win-one-lose-one/double-tie counters) — it sharpens exactly the
quartering-structure reads the joint buckets compress.
Separately: the prior "cards contribute little" attribution artifact does
**not** exist in-repo (treat its numbers as unverified; the qualitative claim
is expected given digests — but the gaps above are exactly the categories
where low card attribution IS diagnostic of missing extraction).

### 3.2 The v5 package (append-only; core ≈ +40 dims, full ≈ +130)

| # | Feature | Dims | Cost/decision | Note |
|---|---|---|---|---|
| **P1** | Per-board ahead/tied/behind vs exhaustive k=2 combos + joint split & double-tie fractions | 8 | **~0** (counter increments inside the existing k=2 loop, engine.rs:1327-1355) | per-board "behind" IS the nut-rank percentile; exact, deterministic |
| ~~P2~~ | **DEFERRED (user call 2026-07-06)** — MC runout equity decomposition vs 1 random 5-card opp. User's case: the model demonstrably gets money in with strong unmade draws already (the learning signal contains draw equity via 64-runout `payouts_ev` chip deltas), and vs-random showdown equity overstates realizable equity at range-narrowed later-street nodes. Noted for the record: the same vs-random + MC caveats apply to the existing 12 opp-outcome dims, so the philosophical line doesn't exclude it — but the marginal value is unproven and it was the only expensive item. Revisit ONLY as an ablation if post-v5 evals still show draw-node imprecision | (8) | ~650µs @S=96 | — |
| **P3** | Blockers-to-nuts per board, unconditional (nut-flush card on 3+-suit boards, nut-straight rank blockers, paired-board trips/boat blockers) | 8–10 | <2µs | bluff selection |
| **P4** | Effective-price block: eff_to_call = min(to_call, eff stack), effective pot odds, all-in-if-call, commit fraction, log1p pot/eff-stack | 5–6 | ~0 | fixes the uncapped-price bug as append-only companions |
| ~~P5~~ | **ABLATION-TIER (downgraded 2026-07-06, §3.1)** — k=5 field-probability row (best-2-of-5), the calibrated complement to the k-density rows | (4–10) | +131µs @384, fund by cutting k=3 to ~128 | densities + reward calibration already yield correct quartering discipline; test only if multiway thin spots stay imprecise post-v5 |
| P6–P10 (optional) | players-yet-to-act + raises-this-street; per-seat aggression aggregates from history; log1p money companions (incl. **unclipped SPR log1p** — fixes deep-tier saturation); hole-pattern class; board texture | ~60 | ~0 | P8's SPR fix is the one I'd promote into core |

**Core recommendation: P1 + P3 + P4 + SPR-log1p (from P8), ~+30 dims — all
exact, deterministic, near-free, and squarely "objective game-state facts"
(P1 is counters inside the existing exhaustive k=2 loop; P3 is board+hole bit
math; P4 is arithmetic).** P2 and P5 are both ablation-tier follow-ups, not
launch scope (§3.1) — revisit only if post-v5 evals show draw-node or
multiway thin-spot imprecision respectively.

### 3.3 Migration

Zero-pad warm-start is exactly function-preserving here (torso[0] is a plain
Linear; new columns zero-init → identical policy at step 0). Required work:
both scalar + batched encoders (+ Rust encoder: update it or gate it off
explicitly — it's default-OFF and would silently diverge, env_batched.py:76-89);
parity tests; `obs_adapter` gains a `991 → slice` entry (trivial — tail
append); **train.py needs a pad-shim for BOTH the actor AND CentralCritic**
(its input = obs + opp multi-hots, so its first layer grows too; today
warm-start would hard-fail on shape); pool seeding needs the same shim (or
accept a thin pool). Scale new dims to ≈[0,1] (outs/13 etc.) so fresh columns
don't get outsized gradients.

---

## 4. Training: equilibrium anchoring + regimen

### 4.1 Equilibrium anchoring (the "face-up/over-bluff root cause" program)

Literature verdict (ICLR 2026 large-scale study, 7 algorithms × exact
exploitability): **regularized PG (PPO/MMD) is the strongest family** — NFSP,
PSRO, ESCHER, R-NaD all failed to beat it; but unregularized self-play cycles,
and a snapshot pool is a heuristic that stabilizes without converging (its
fictitious-play average is never computed, and pool members are stale weaker
selves — best-responding to them is exactly the air-betting mechanism).
Consensus fix = PPO + KL-to-slow-reference (MMD; the 2026 GARIP result says
the reference should be an **EMA of past policies** — which is literally our
dormant `--kl-anchor-coef` design). Interpretation shift: (entropy coef, KL
coef) jointly define a **QRE temperature** — mixing becomes a *dial we own and
anneal*, not an artifact of an entropy floor fighting the optimizer.

Plan: fix the two kl-anchor bugs (§6.1), then trial coef 0.05–0.2 during the
W1 anneal phase, watching the existing `klanc` log. Cost ≈ one frozen forward
per minibatch. Note the EMA reference is process-lifetime only (not
checkpointed) — persist it as `ckpt["model_ema"]` while at it; the same
tensor doubles as a smoother, less-exploitable serving actor (EMA serving).

### 4.2 Anneal machinery repair + the entropy→magnet handoff (v4 arch)

Corrected history (§0.1): vFour4 was manually annealed to 0.10 via
anneal_control's `tier_ent` keys and spent ~370 updates there — so "just
anneal more" is not the plan. Two things follow:

1. **Repair the machinery anyway** (it constrains every future run): have
   `collect_rollout_multiconfig` return per-tier F/T/R (it holds per-sub
   batches at rollout.py:1566-1570 — today they're summed, so the per-tier
   stop-loss can't run and tier coefs can't diverge), apply per-tier coefs per
   sub-rollout instead of `tier_ent[mix_tiers[0]]` for the whole update
   (train.py:1394 — also fixes the live-tune traps: `{"tier_ent": {"deep":
   X}}` accepted-and-ignored, and the `entropy_coef` key a silent no-op in
   mix mode). This restores the deliberate per-tier design (deep historically
   needs the higher floor) that mix-configs silently flattened.
2. **The remaining anneal headroom (0.10 → v1's mature 0.04–0.085 band) is
   modest, and pushing below 0.10 on entropy alone risks the documented
   passivity failure.** The handoff is the point of §4.1: bring the fixed
   kl-anchor EMA magnet up (0.05–0.2) FIRST so mixing is held by the
   reference term, then walk the entropy coef down under the F/T/R +
   exploiter-probe stop-loss, with the check-bet probe (§1.2 diagnostic 2)
   as the acceptance test. Temperature = (entropy coef, magnet coef) as a
   pair.

### 4.3 Regimen verdict (the user's 10×3 question)

Verified: the 30 configs are **fresh random draws each update** (10 per tier,
not a fixed catalog); each fixes one seat count + one per-seat stack vector
for a ~333k-decision sub-rollout; advantage normalization is effectively
per-config (each sub is unit-normalized before the near-identity global
renorm, rollout.py:442-447/1501-1520), so deep pots do NOT dominate the policy
gradient; reward is bb-units everywhere. **Keep mix-configs** — it exists
because temporal block rotation demonstrably killed stems (rollout.py:1481-1483)
— with these changes:

1. **Seats stay uniform 2–6** — retracted an earlier `--seats-dist clubgg`
   recommendation: the uniform mix is intentional (rounded model for
   3–4-handed deeper home games as well as ClubGG, user 2026-07-06).
2. **Hoist the frozen-opponent rebuild to per-update scope** (today ≤8 × 2048×4
   models are rebuilt from state dicts 30×/update, rollout.py:852-864; the
   pool is frozen within an update, peak stays 8 models — this is NOT the
   reverted cross-update opponent-cache OOM). Then **raise configs/tier to
   15–20** — more stack-vector diversity per update at ≥150k rows/config
   (per-config norm still solid).
3. **Coverage gap**: 80–150bb is the thinnest band (clubgg_deep caps at 120
   with 7% on 80–120) while the study tool serves arbitrary stacks and the
   home games play deep — add a fourth draw band or widen clubgg_deep. Keep
   equal tier thirds otherwise.
4. Don't touch: bb-reward + per-config advantage norm (the 2026-06-23
   "per-config destabilized" precedent was actually a near-null delta —
   removing the global renorm — treat that attribution as noise, and leave
   the current scheme alone either way).

### 4.4 Time-to-quality levers, ranked

1. **value_clip A/B — cheapest potentially-large lever.** `value_clip = 0.2`
   (config.py:104) is applied in **raw bb** against returns spanning ±250bb+
   (ppo.py:286-291): once the critic's prediction moves >0.2bb from its
   rollout-time value toward the target, the gradient dies — critic tracking
   is rate-limited to ~0.2bb/state/epoch-visit. Warm runs partially mask it;
   distribution shifts and cold starts pay heavily; slow critic = biased
   advantages = slower policy. A/B ∈ {2, 10, ∞} on a short stem, watching the
   collapse telemetry (it may be an accidental stabilizer given the history —
   don't just flip it on the production stem).
2. **VRPO / Q-critic (biggest single upgrade, published at poker scale).**
   Fan & Farina 2026: in self-play, GAE carries irreducible variance from
   sampling future actions of *mixed* policies — worst exactly at mixed
   check/bet nodes; replacing GAE with Expected-SARSA(λ≈0.95) on a
   centralized Q(s,a) analytically averages it out; first search-free PPO to
   beat Slumbot (+33±19 mbb/h). Our CentralCritic already sees all holes;
   extend to Q over (gate × anchor). **Clean migration**: build the v5 critic
   WITH the Q head from day one, zero-init so Q(s,a) ≈ V(s), train it as an
   auxiliary regression under GAE first, flip the advantage estimator once
   Q-loss converges — no checkpoint break, clean attribution.
3. **Opponent-cache hoist + configs/tier 15–20** (§4.3.2) — pure throughput.
4. **KL-adaptive inner loop**: no KLSTOP and mean |kl| < 0.25 for k updates →
   ppo_epochs 3; any trip → LR ×0.7 next update. ~10–30% fewer updates to a
   given strength, gentle and reversible.
5. **Rust obs encoder A/B on the idle pod** (built, bit-exact, default-OFF
   purely as an instant-revert switch; prior measurement inconclusive under
   contention).
6. **EV-runout samples 64 → 256** (engine time is a few % of update; small
   critic-variance win). Exact turn+river enumeration is ~170× for marginal
   residual — skip.
7. **Duplicate/antithetic dealing in training**: plumbing mostly exists
   (per-env seed+button arrays), but CentralCritic + EV-runouts + per-config
   norm already harvest most of the variance; modest expected gain, mind
   within-minibatch correlation. Below the levers above.
8. **Pool-quality softmax sampling + wider snapshot spacing** — more an
   anti-cycling lever (symptom 3) than a wall-clock one. Exploiter league =
   W3.

Non-levers: rollout 10M→11.5M (linear samples-for-time), trinal-clip (policy
side duplicates target_kl+adv_clip; value side subsumed by fixing value_clip).

---

## 5. Eval stack (prerequisite — how "x% quality" becomes measurable)

Accepted practice (LBR/AIVAT/duplicate lineage + DeepNash/Pluribus reporting):

1. **Duplicate + seat-rotated matches with CRN** between checkpoints/stems —
   ~10× fewer hands per decision (infra: batched reset already takes per-env
   seed+button arrays).
2. **Trained-exploiter probe time series** (exploitability lower bound):
   `scripts/exploitability.py` exists — schedule it per promotion candidate
   and TREND it; this is the arbiter for "is check-bet mixing entropy residue
   or genuine pool exploitation", and the acceptance gate for every v5 change.
3. **LBR-lite** (fold/call/pot/jam lookahead using engine equities) — cheap
   incremental floor, weekend-scale port given `payouts_ev`.
4. **AIVAT-lite** (critic-based luck correction) for any human/live sessions.
5. **External spot-checks**: no solver supports this format (verified — closest:
   MonkerSolver single-board PLO5 ante-pot trees; DinoPoker double-board
   bomb-pot equity calculator; CardQuant sells double-board *strategy*
   heuristics, PLO4-oriented). Use MonkerSolver single-board PLO5 ante trees
   to sanity-check one-board-dominant nodes; equity-calc parity for
   settlement. Neural search (ReBeL/SoG) is intractable here (flop PLO5
   double-board beliefs ≈ C(46,5) ≈ 1.4M holdings/player, ~10³× HUNL, and
   multiway voids the two-player theory) — regularized model-free self-play
   is the on-trend architecture; invest in eval, not search.

---

## 6. Consolidated verified bug/oddity list

### 6.1 Fix in W0 (no retrain required to land the code)

| # | Bug | Where | Effect |
|---|---|---|---|
| B1 | `--kl-anchor-coef` crashes v4 heads: (B,2) size_params masked_fill vs (B,11) grid.legal — **reproduced** | ppo.py:79-104 | blocks the MMD magnet entirely; fix via `model._anchor_dist(...)` on both sides |
| B2 | `_kl_to_reference` doesn't detach p_raise (the 2026-06-11 lesson, mirrored nowhere here); minimized regularizer → fold-bias pressure | ppo.py:103 | latent until B1 is fixed; fix together |
| B3 | `beta_h_eff` weights negative Beta entropies by **non-detached** anchor_probs → entropy bonus pays mass onto end anchors | network.py:505-507 | pot-heavy face-up sizing; detach at the v5 head bump (behavior change on v4 if hotfixed — prefer bundling) |
| B4 | value_clip 0.2 **bb** vs ±250bb returns | config.py:104, ppo.py:286-291 | critic rate-limit; A/B first (§4.4.1) |
| B5 | mix-configs consumes only `tier_ent[mix_tiers[0]]`; `{"tier_ent":{"deep"/"clubgg_deep": X}}` and `{"entropy_coef": X}` are accepted/printed but silently ignored in mix mode; F/T/R pooled across tiers → per-tier design flattened, auto-anneal can't run (manual anneal via the clubgg key DID work — pod log verified) | train.py:1394, 1400-1406, 911, 1420; rollout.py:1522-1534 | live-tune no-op traps; per-tier coefs impossible |
| B6 | LR warmup runs on the LOCAL update counter → every guardian relaunch silently re-runs 75 warmup updates | train.py:1264-1266, 1413 | hidden tax on routine resumes |
| B7 | Stale-anneal_control guard (7c7ec63) is at HEAD but was deliberately not deployed while vFour4 ran — ship on next deploy; inspect pod `runs/anneal_control.json` before ANY relaunch | train.py:1198-1209 | the stale-override incident class |
| B8 | Serial pool sampling uses unseeded global `random` | selfplay.py:64 | serial/eval reproducibility only |
| B9 | `_save_mid` stamps `game_config` = first clubgg draw under mix-configs | train.py:1226, 1358 | checkpoint metadata misrepresents regimen |

### 6.2 Obs-encoding items (fold into obs v2, append-only)

Uncapped pot-odds/bet-faced vs hero stack (encoding.py:783-792 — same bug
class as NLH); SPR clip[0,4] saturating the whole deep tier at the flop
(encoding.py:779); dead 8-dim rel-pos block (188-196); ~33 structurally dead
preflop dims (reclaim only on a cold layout); stale docstrings ("886",
bet-faced comment); train-vs-serve MC budget delta (384 vs 1024) —
documented, and inherited by any future MC feature. (The opp-outcome k≤4
universe was listed here in an earlier draft — reclassified NOT-a-bug: it's
combo-dominance density by design, §3.1.)

### 6.3 Product/serving notes

- UI rec is argmax — with the mixture head, present the **menu** (per-size
  frequencies from the histogram + optional per-component μ/s/w block); a
  study tool for serious players should show solver-style mixed strategies as
  the primary artifact, argmax as "primary line" only.
- EMA-actor serving (persist `model_ema`, promote it to stub) — smoother and
  less exploitable than the last iterate; pairs with §4.1.
- Min atom = 1bb open (~5.6% of an 18bb pot) burns one of 11 anchors on a
  size humans read as a misclick; revisit the ladder only if a layout break
  happens anyway (it costs a checkpoint break on its own).
- Training is rake-free; if study users play raked bomb pots, thin-margin
  recommendations are slightly optimistic.

---

## 7. Wave plan

- **W0 (no retrain, this week)**: land B1–B9 fixes; build the eval stack
  (§5.1–5.3); run the two symptom diagnostics on the current stub (free
  gate-distribution probe + exploiter probe) — they decide how much of
  symptom 3 the W1 anneal is expected to fix and set the baseline every later
  wave is judged against.
- **W1 (vFour4 continuation or vFour5, v4 arch unchanged)**: per-tier
  F/T/R + per-tier coefs under mix-configs (B5); **magnet on first, then walk
  temperature below 0.10** (entropy→magnet handoff, §4.2; stop-loss = F/T/R +
  exploiter probe); opponent-cache hoist (+ configs/tier 15–20); value_clip
  A/B on a side stem. Gate: exploiter-probe exploitability drops; check-bet
  probe shows bet mass collapsing on pure-check classes.
- **W2 (the v5 stem — one deliberate break)**: mixture sizing head (§2) +
  obs v2 core (§3.2) + B3 detach + Q-head-in-critic (trained as auxiliary,
  GAE still driving) + zero-pad/head-converter warm-start from the best W1
  checkpoint (pending user sign-off; cold fallback). Entropy seeds: do NOT
  cold-start at W1's annealed floors — re-seed near 0.45 and re-anneal (the
  one-way-down lesson).
- **W2.5**: flip advantages GAE → Expected-SARSA(λ) once Q-loss converges;
  A/B 20 updates.
- **W3 (compute programs)**: exploiter stems feeding the pool (league-lite),
  pool-quality sampling, external validation campaign (MonkerSolver
  single-board spot checks), serving EMA promotion.

---

## 8. Open decisions for Miles

1. **Warm-start v4→v5** — mechanically feasible despite head + obs changes
   (component-0 head converter + zero-pad input columns compose; §2.4/§3.3
   — the v5 net reproduces v4's policy at step 0). Warm preserves ~10 GPU-days
   of vFour4; cold restarts the exploration/collapse-hardening saga but gives
   clean lineage. (Recommend warm; your call.)
2. **K=3 confirmed?** (Recommend yes; collapse is graceful.)
3. **Obs v2 scope**: core +~30 dims (P1 + P3 + P4 + SPR-log1p; P2 and P5
   both ablation-tier per your calls) — add any of P6-P10? (Recommend core
   as-is.)
4. **W1 first or straight to W2?** W1 is pure harvest on the existing stem
   (magnet + temperature push + machinery repair) and de-risks W2's baseline,
   at the cost of pod-weeks before the architecture work lands. (Recommend
   W1 first — it also cleanly answers how much headroom the magnet+anneal
   path has left in v4.)
5. **value_clip**: A/B on a throwaway stem before adopting anywhere.

---

*Full agent reports (file:line evidence for every claim here) live in the
session transcript of 2026-07-06. External references: Sokota et al. ICLR
2023 (MMD, arXiv:2206.05825); Rudolph et al. ICLR 2026 (arXiv:2502.08938);
Perolat et al. Science 2022 (R-NaD); GARIP arXiv:2606.22688; Fan & Farina
2026 (VRPO, arXiv:2605.19235); Lisy & Bowling AAAI-17 (LBR,
arXiv:1612.07547); Burch et al. AAAI-18 (AIVAT); Salimans et al. ICLR 2017
(PixelCNN++ discretized logistic mixtures); Brown & Sandholm Science 2019
(Pluribus, multiway reporting practice).*
