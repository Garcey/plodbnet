# V7 design — decided workstreams

*Started 2026-07-12, in parallel with the vSix2 cold start (the corrected
v6 recipe run end-to-end from update 0). v7 is the next architecture
generation; this document holds its four workstreams: each is now a
DECIDED design with implementation status, not a sketch. Items marked
`[shipped]` exist in the codebase today (dormant unless flagged);
`[protocol]` items are specified and waiting on their trigger.*

## Posture

- v6's substrate (LayerNorm+l2-init, HL-Gauss value head, prob-dependent
  clip with live rooms, pooled+anchored Q head, magnet-off cold starts)
  carries forward unless a workstream explicitly replaces a piece.
- Stems are disposable; recipes are the product. vSix2 is both a training
  run and the validation harness for everything v7 inherits.
- User-set shape (2026-07-10): if the sizing revisit shows capacity is
  the binder, v7 = bigger actor and/or critic, possibly with new features
  alongside. Oversized is acceptable; undersized silently caps skill.
- Nothing here touches the running vSix2 recipe. New telemetry rides to
  the pod on the next NATURAL restart; semantics flags are default-off
  and refuse warm-starts across a flip.

## Workstream 1 — advantage & Q-head integrity

*Motivated by the July incident chain: Q-surface family offsets →
terminal-action boundary term paying a fold subsidy → post-fix
expectation-inflation transient consolidating fold-at-locks.*

### 1.1 Hard-coded fold column `[shipped: --q-fold-zero, default OFF]`

`Q[FOLD] ≡ 0` by construction (`CentralCritic(q_fold_zero=True)` pins the
column in the dueling composer; the fold adv params receive zero gradient
and stay at zero-init). Kills the fold-subsidy class permanently — no
supervision to drown, no offset to develop.

**Decision: implemented but NOT the default recommendation.** Two reasons:

1. **It trades one boundary artifact for an init transient.** VRPO mixes
   `V^π(s′) = Σ_a π(a|s′)·Q(s′,a)`. With the fold column pinned at 0 and
   the sibling columns starting at V (zero-init), the mix under-reads by
   `π_fold(s′)·V(s′)` until the siblings learn their conditionals —
   mild fake pessimism on bootstrapped continues, i.e. the same *smell*
   as the bug being fixed, though it decays with head training instead of
   persisting. (The legacy Q≡V init makes VRPO≡GAE exactly; the pin
   deliberately gives up that identity in exchange for truth on folds.)
2. **The anchor may already be sufficient.** vSix2 runs the coef-15 fold
   anchor from update 0 — offsets can never develop, so the subsidy class
   is dead by supervision alone if qF holds ≈0 for the whole run.

**Adoption gate:** if vSix2's qF canary stays |qF| < ~1bb through the run,
the pin stays off in v7 (unnecessary complexity). If qF drifts despite
coef 15 (or the coef needs escalating), flip the pin on the v7 fresh stem
— it is exactly the nuclear option for that failure. Warm-starts across a
flip are refused by train.py (the surface reinterprets).

### 1.2 Raw-space dueling base `[shipped: --q-base-raw, default OFF — recommended ON for the first v7 stem]`

**Root cause being fixed:** the dueling base is the display V =
`symexp(Σ p_i·c_i)` — symexp of the *symlog-space* mean — while the Q
targets are *raw-space* returns. That Jensen-type gap between the two
means is width-scaled (deep tiers ≫ shallow), and the shared adv rows
absorbed it as the July family offsets (−3/−16bb by tier), which the
boundary term then paid out as the fold subsidy.

**Chosen design:** compose the base from the SAME HL-Gauss categorical,
raw-space centers: `base = Σ p_i·symexp(c_i)` (`_raw_value_centers`, a
non-persisted buffer — state dicts stay interchangeable both directions).
Base and target now share units; zero new parameters; the display V and
every V consumer (GAE, UI, value loss) are unchanged; zero-init still
gives Q ≡ base, so VRPO starts as GAE-with-raw-baseline.

**Alternatives considered and rejected:**
- *Plain Q head (no V base)* — loses the dueling variance reduction and
  the Q≈V init; the head would relearn state value from scratch per column.
- *Separate learned raw-space scalar head* — works, but adds parameters,
  a second value loss to balance (raw-bb² scale, the exact drowning
  hazard from the coef-1.0 era), and AGC/exemption questions. Fallback if
  the readout residual proves material.
- *Regress Q in symlog space, symexp at read* — moves the same Jensen gap
  INTO Q (E[symlog] ≠ symlog(E)); VRPO needs expectation-of-raw units.

**Known residual:** the HL-Gauss target smear is Gaussian in symlog space;
symexp-ing bin centers turns that into a small width-scaled skew (~2-3% of
|V| at σ=0.75·bin — an order below the 15bb offsets). The qF/qT canaries
measure what's left; if it matters, the separate raw head is the fallback.

**Adoption:** ON for the first v7 stem (fresh stem — no warm-start
concern). Optionally A/B-able earlier on a throwaway v6.x stem since the
flag needs no checkpoint surgery, only a cold start or a deliberate
convert. train.py refuses warm-starts across a flip.

### 1.3 Terminal-boundary telemetry `[shipped: Batch.is_terminal + qT log stat]`

Terminal actions (folds, hand-ending calls, successful steals) take
`δ = r − Q(s,a)` with NO next-state term to cancel estimator bias — any Q
bias becomes a per-action subsidy/tax exactly there. July's fold subsidy
was this mechanism through the fold column; the qF canary watches only
fold. The new canary generalizes to every terminal action:

- The batched collector now flags each seat's last decision row
  (`Batch.is_terminal`; at those rows the stored return IS the raw
  realized reward — free ground-truth labels, the same identity the fold
  probe used).
- ppo.py logs **`qT` = mean(return − Q(s,a)) over NON-FOLD terminal
  rows** per update, next to qF. Reading: persistent positive = hand-
  ending actions collect fake advantage (subsidy); negative = taxed;
  healthy = hovers near 0 with sampling noise. This is the successor lie
  detector if 1.1's pin ever makes qF read a trivial 0.
- Cost: one bool per row (~11MB per 11.5M-row update), no RNG or ordering
  changes; serial collectors keep `is_terminal=None` and the canary skips.
- **Deployment: rides to the pod at the next natural vSix2 restart** (do
  not restart a healthy run for telemetry).

### 1.4 Q-head training data — RESOLVED: keep gather + pooling

Each column learns only from taken-action rows; the 2026-07-09 starvation
(11 anchor columns × ~3% of rows each) was a *pooling* problem and pooling
fixed it (3 columns, ~⅓ of rows each; deep |E_π[Q]−V| p95 53→13bb).
Expected-SARSA-style all-column targets need off-action returns we don't
have (counterfactual continuations = a rollout per (state, action) —
prohibitive), and importance-weighted off-action regression adds exactly
the variance VRPO exists to remove. Standing check instead of redesign:
the probe suite's |E_π[Q]−V| metric (WS3) catches starvation regressions.

## Workstream 2 — capacity & sizing

*Standing plan (memory: v7-network-sizing-revisit): the clock starts after
~1 healthy vSix2 week. Order: cheap probe → targeted A/B → gated sweep.*

### 2.1 Utilization probe `[shipped: scripts/utilization_probe.py]`

Per-layer, on a probe batch of real facing-a-bet nodes across three tiers:
dead / near-dead unit %, activation effective rank (participation ratio)
+ rank@99% variance, and weight stable rank, for every Linear in actor and
critic; JSON history to `runs/utilization_history.jsonl`. Reading guide:
eff-rank floor ≪ width and rising dead% ⇒ oversized (fine); eff-rank
saturating toward width across depth ⇒ capacity may bind ⇒ run 2.2/2.3
before concluding.

**Dry-run on vSix1_485 (2026-07-12, logged — tainted history, methodology
validation only):** the torso is near rank-COLLAPSED, not oversized-idle:
actor eff-rank ~1.5–1.9 of 2048 per layer, 93–98% dead units, critic
similar (~1.2–2.3 of 1536); first-layer pre-activations mean −18.9 with
99.4% ≤ 0 (a learned dying-ReLU regime — the input projection, which has
NO LayerNorm in front of it, unlike the residual blocks); cross-checked by
hand, consistent with the checkpoint's known fold-heavy collapse. Two
takeaways: (a) the probe detects representation collapse loudly, so it
doubles as a health monitor — **run it on vSix2 EARLY (~u50) and weekly**,
not just at the sizing decision; (b) if healthy vSix2 also trends toward
input-block death, that's an architecture bug to fix in v7, not a sizing
signal (see 2.4 candidate #1). **Canonical capacity read: a vSix2
checkpoint after a healthy week.**

### 2.2 Critic 2× A/B `[protocol]`

Question: is the converged `v` loss a floor (irreducible outcome variance)
or a ceiling (capacity)? Protocol: freeze a mature vSix2 policy; collect
ONE update's rollout (~9M rows) with it; train two fresh critics on
identical data/schedule — A = 1536×2 (current), B = 3072×2 (or 1536×4) —
16 minibatches × 3 epochs, 90/10 train/held-out split; compare held-out
HL-Gauss loss + |E_π[Q]−V| calibration. If B beats A materially on
held-out, critic capacity binds → v7 widens the critic.
**Compute rule: NEVER beside the live trainer** (63.8/96 GiB peak leaves
no headroom) — run in a deliberate pause window on the pod, or locally on
a ≤2M-row subsample (fits the 3070 at small batch). Caveat carried from
the sizing memo: this is a *conditional* read (value-fitting capacity at
that policy), not proof about the policy ceiling — treat as one vote.

### 2.3 Eval-gated size sweep `[protocol, blocked on WS3.4]`

Grid: actor {1024, 2048, 4096} × {3, 4, 6} blocks (critic scaled by the
2.2 verdict), short cold runs on the vSix2 recipe, scored by the WS3.4
eval stack (duplicate-deal + probes), not by training-loss aesthetics.
Promotion rule: a size wins only if it beats the incumbent on eval at
equal wall-clock budget (not equal updates — bigger nets get fewer).

### 2.4 Obs v3 / architecture candidates `[collecting]`

Standing note: every obs candidate must name the decision it should
change and the existing dims that fail to carry it (the obs-v2 tail set
the bar).

1. **Input-block pre-norm** (from the 2.1 dry-run): the v6 LayerNorm sits
   only inside the residual blocks; the 1020→2048 input projection is
   bare Linear+ReLU with no skip path — the one place a dying-ReLU regime
   can strangle the whole network (vSix1_485 measured 96.5% dead there).
   Candidate: LayerNorm on the input block too (or on the concatenated
   obs before it). Gate on the vSix2 probe series: architecture change
   only if input-block death shows up on a HEALTHY run.

## Workstream 3 — eval & probe suite (first-class)

*The week's meta-lesson: the user's hand reviews out-diagnosed aggregate
metrics twice. Automate the eyeballs.*

### 3.1–3.3 Per-checkpoint probe suite `[shipped: scripts/probe_suite.py]`

One CLI run per checkpoint (node banks generated once, reused across
checkpoints), appending one JSON line to `runs/probe_history.jsonl`:

1. **Lock-fold probe** (3.1): P(fold)/P(raise) at locked boards facing a
   pot bet, shallow + deep banks; P(fold) at double-board trash as the
   contrast. THE never-fold-the-nuts curve; deep-lock P(fold) must trend
   flat/down. This gates entropy cuts (suspended until proven) and any
   promote.
2. **Q-calibration audit** (3.2): fold-column stats vs truth-0,
   |E_π[Q]−V| median/p95, per-column advantage sanity, by node family —
   the 2026-07-09/11 audit families A–F, same seeds, now tracked (port
   verified bit-exact against the original scratchpad probes).
3. **Gate-sharpness panel** (3.3): median max-gate-prob + mean gate
   entropy per family — commitment-without-collapse, trend across
   checkpoints.

**vSix1 baseline series (logged 2026-07-12):** deep-tier calibration
|E_π[Q]−V| p95: u200 53–61bb → u460 12.7–12.9 → u485 2.3–3.2bb, fold-col
std down to 0.29bb — the anchor-15 fix converging the SURFACE — while
deep-lock P(fold) went 33.8 → 35.6 → **54.3%** (raise 22 → 1.2%) over the
same span: the transient consolidating the wrong POLICY. The suite exists
precisely to show that split-screen (calibration ≠ behavior); vSix2's
curves get judged against this series. Known bank quirk: the "trash"
contrast split (behind ≥90% on both boards) is empty at fold-legal
facing-pot-bet nodes — threshold to revisit, metric kept.

**Cadence & wiring:** every ~50 vSix2 updates (pod-side over ssh — CPU,
8 threads, no GPU contention — or locally on a fetched checkpoint), plus
MANDATORY before any UI promote: no promote without a suite line in the
history file. vSix1_200/350/460/485 are logged as the baseline series (the
known-bad → post-fix arc, so future curves have context).

### 3.4 Real scoreboard `[roadmap — the WS2.3 blocker]`

Duplicate-deal eval (mirrored seats/cards, CRN) → AIVAT-style variance
reduction → LBR-style exploitability probes. Build order is duplicate-deal
first: it alone turns "looks better in review" into a paired, low-variance
match result and unblocks the size sweep. (Design exists in the NLH
roadmap memo; PLO5 port is the same machinery.)

## Workstream 4 — process rules (carried + new tooling)

- One variable at a time; pre-committed revert criteria; probes before
  interventions; learn/MASTER.md patched same-day as mechanics change.
- Live-tunable knobs preferred over restart-bound ones (anneal_control:
  lr, tier_ent, clip rooms, q_fold_sup_coef — extend as needed).
- Guardian discipline: kill guardian BEFORE trainer; anneal_control is a
  startup baseline — bake live values into flags before any restart;
  rewrite the whole control file on edit (stale keys re-apply).
- `[shipped]` **scripts/check_restart_sync.py**: pre-restart gate that
  diffs a guardian script's flags against anneal_control.json and exits
  nonzero on mismatch — the "guardian silently reverts live values"
  failure (bit twice in the vSix1 era) is now a 2-second check. Known
  limit: preset-derived values (e.g. --v6's fold-sup 15) aren't visible
  as flags; it errs permissive when a knob is absent on one side.
- Pod deploys while a run is live: single-file additive only (new
  scripts), never a full-tree overwrite; full deploys wait for a restart
  window.
- Fresh-stem semantics flags (q_fold_zero, q_base_raw, and successors)
  must: stamp into ckpt config, refuse warm-starts across a flip, and be
  invisible to V-only consumers. (All three properties are how the ε-floor
  class of bugs stays dead.)

## Incident ledger (what motivated all this)

| date | incident | root cause | v7 answer |
|---|---|---|---|
| 07-09 | 200-update cold-start freeze | magnet EMA tethered policy to its init | magnet late-stage only, short horizon (WS4) |
| 07-09 | VRPO advantages noise | 13-col Q starvation (~3%/col) | pooled head (shipped); WS1.4 resolved: keep gather |
| 07-11 | KL pinned vs 22× LR | clip mid-band = per-update quota | live clip rooms (shipped) |
| 07-11 | Q fold column −3/−16bb offsets | drowned anchor + symlog/raw base gap | anchor 15 (shipped); raw base (WS1.2, shipped dormant) |
| 07-12 | fold subsidy + lock-fold consolidation | terminal boundary term; expectation transient | fold pin (WS1.1, shipped dormant); qT canary (WS1.3, shipped); lock probe (WS3.1, shipped) |

## Status

- 2026-07-12: all four workstreams executed to decided-design +
  implementation. Shipped this pass: `--q-fold-zero` / `--q-base-raw`
  (dormant, warm-guarded), `Batch.is_terminal` + the `qT` boundary canary
  (live at next natural restart), `scripts/probe_suite.py` (+ vSix1
  baseline series), `scripts/utilization_probe.py` (+ vSix1_485 dry run),
  `scripts/check_restart_sync.py` (validated against the live pod).
- Open decisions for the brainstorm: v7 stem recipe (raw base on, pin
  gated on vSix2 qF evidence), WS2 sizing verdict (waits on the healthy
  week), WS3.4 duplicate-deal build slot, obs v3 candidates.
