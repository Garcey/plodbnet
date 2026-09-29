# NLH Neural GTO Estimator — Design Options & Ranked Architectures

**Scope:** training-data + inference pipeline for a study-tool "neural estimator of GTO" (GTO Wizard AI / DeepStack lineage), built on plodbnet's existing Rust NLH engine, study UI, and PPO stack — **not** a continuation of nlh5 PPO anneal alone.

**Honest premise:** current `nlh1–4` is multiway self-play PPO. That is a useful *substrate* (engine, obs, anchors, UI, adversary probe) but is **not** a GTO estimator. Multiway self-play has no unique Nash; the known fold-collapse and soft preflop frequencies are expected under EV-maximizing PPO with sunk blinds. A GTO product needs **equilibrium labels or equilibrium search**, not more entropy knobs.

---

## What you already have (reuse map)

| Asset | Role in GTO system |
|-------|-------------------|
| Rust `nlh_single` engine + study APIs (`reset_study_nlh`, board setters, `pack_range_nlh`) | Exact game dynamics + UI entry; **must** remain ground truth for legal actions / pots / SPR |
| `encoding_nlh.py` OBS 995 (log1p SPR, pot-frac history, hole-class, masks pad-to-8) | Strong starting **public-state encoder**; extend, don't rewrite |
| `NLH_ANCHOR_SPEC` 12-anchor ladder + v4/v5 heads | Ready **action representation** for policy targets and re-solve menus |
| Study UI + Ranges tab | Serving surface for pure-net and later re-solve modes |
| `exploit.py` LBR-ish adversary + future probe bank | Measurement stack (upgrade to true exploitability on HU) |
| nlh PPO checkpoints | Optional **encoder warm-start** / sampling policy; **not** GTO labels |

---

## A) Offline solution generation

### Practical abstractions for 6-max NLH ~100bb + ante

Full 6-max NLH is ~10¹⁶⁵ histories; nobody solves it raw. Practical decomposition:

| Layer | Practical choice (small team) | Notes |
|-------|-------------------------------|-------|
| **Players** | **HU first**, then **3-way postflop**, then 4–6 via preflop-only or reduced postflop | GTOW caps 3 to flop for the same reason; multiway NE is non-unique |
| **Action abstraction** | Pot fractions aligned with `NLH_ANCHOR_SPEC` (min, 25–275%, all-in) + **raise cap 3–4/street** | Same menu as the net → labels land on the head without remapping. Optional "dynamic size" later at re-solve time |
| **Card abstraction** | River: equity / hand-strength buckets or **exact** (1326 combos tractable HU). Turn/flop: potential-aware or EHS² buckets (e.g. 200–2000). Preflop: 169 lossless classes | Suit isomorphism is free and lossless |
| **Preflop** | Blueprint MCCFR on abstract tree + **weighted flop subsets** (25 / 49 / 85 / 184 of 1,755 iso flops) | Matches industry practice (Pio/GTOW subset literature) |
| **Postflop subgames** | Exact or lightly abstracted HU trees from fixed root (SRP / 3bet / 4bet, SPR bins) | Native rust_cfr problem size |
| **Ante** | Bake into pot/stack root of every subgame | Engine already has ante; roots must match `GameConfig.nlh_default` |

**6-max 100bb "full" offline solve is not a day-1 goal.** Day-1 data = library of **postflop subgames** + a **preflop blueprint**. Day-N = denser library + multiway river/turn.

### Label teacher (locked)

**Native rust_cfr on `rust_engine` is the only production teacher.** Same chip rules as live UI; DCFR/MCCFR; export `source=rust_cfr*`.

| Source | Fit | Verdict |
|--------|-----|---------|
| **Custom MCCFR/DCFR in Rust** | Same rules as live UI | **Only teacher** |
| **OpenSpiel** `universal_poker` | Research-grade; slow | Algorithm prototypes only, not production data |
| **PokerRL / Deep CFR papers** | Research reference | Learn from; don't depend on |

No external commercial or AGPL solver is in the train path.

### HU first vs multiway day 1

**HU first — non-negotiable for a GTO claim.**

- Unique NE (2p0s); exploitability is well-defined (mbb/hand or % pot).
- Labels are coherent; you can gate model quality with best-response / local re-solve.
- Multiway day 1 produces "a" equilibrium of many; frequencies won't match any single commercial solver and you cannot score Nash distance honestly.

Multiway later: **3-way river → 3-way turn → restricted 3-way flop**, with clear product language ("equilibrium under shared blueprint / population play"), not "the GTO."

### How to sample situations for training

| Scheme | When | Weight |
|--------|------|--------|
| **Reach-weighted** | Default for Deep CFR / value nets | ∝ π_{-i}(I) under average strategy — puts mass where hands actually arrive |
| **Strategy-weighted / regret-priority** | Hard spots, mixing nodes | Upweight high-regret or high-entropy infosets |
| **Uniform over nodes** | Bad as sole scheme | Over-samples ghost lines (zero-reach); GTOW's QRE story exists partly because of this |
| **Curriculum by SPR / street / pot type** | Always | Explicit strata: SPR∈{1,3,6,10,15,20,30}, streets, SRP vs 3bet, IP/OOP |
| **Public-state + range samples** | For re-solve value nets | Sample (board, pot, stacks, betting line, range pair) not just hero hole |

**Concrete recipe:** store solver trajectories as `(public_features, hero_combo or bucket, legal_mask, π*, v*, reach)`. Train with **reach-weighted loss** + **stratum oversampling** so rare SPR/line cells still get coverage. Do **not** train only on uniform tree dumps.

---

## B) What the network predicts

### Target heads (pick a bundle per architecture)

| Head | Target | Use |
|------|--------|-----|
| **Policy π** | Softmax / logistic over gate × anchors (optionally Beta refine) | Pure-net study recommendations (UI today) |
| **Action CFVs / Q** | Counterfactual values per legal action (bb or pot-normalized) | Depth-limited re-solve leaf values (DeepStack / GTOW AI) |
| **State value V** | Range-weighted EV under equilibrium | Critic, baselines, AIVAT probes |
| **Ranges / reach** | 1326 (or bucket) weights for each seat | Belief input to re-solve; hardest; often *derived* by Bayes from blueprint instead of predicted |
| **Equity / strength aux** | Showdown equity, EHS | Cheap aux; stabilizes early training |

### Representation of continuous / flexible sizing

Keep **one** sizing language end-to-end:

1. **Training labels:** map solver sizes → nearest legal anchor + optional refine `u` in bracket (reuse `sizing.py`).
2. **Policy head:** current v4 logistic (or v5 mixture) over `NLH_ANCHOR_SPEC` — already overbet + all-in capable.
3. **Re-solve menu:** same anchors, or "dynamic K sizes" chosen by 1-ply EV using the value net (GTOW dynamic sizing analogue) — **inference only**, doesn't require continuous policy.

Avoid a second parallel continuous Beta-only head that diverges from solver menus.

### Policy-only vs value-only vs both

| Mode | Pros | Cons | Verdict |
|------|------|------|---------|
| **Policy-only supervised** | Fast UI; simple loss; reuses act path | No re-solve; weak OOD on weird lines/stacks | Good **Phase 1 product** |
| **Value-only (DeepStack)** | Enables re-solve; better theoretical grounding | Needs search at serve; range tracking | Required for **Phase 2 quality** |
| **Both (actor-critic supervised)** | UI can show π instantly; re-solve uses V/Q when user waits | Two losses to balance; must keep π ≈ argmax re-solve | **Recommended end state** |

Supervised both ≠ PPO actor-critic. Losses are **KL/cross-entropy to π\*** and **MSE/Huber (or HL-Gauss) to v\***/CFVs, optionally reach-weighted. Entropy bonuses are unnecessary if labels are already mixed.

---

## C) Generalization targets (encoding)

You already encode the right *kinds* of features; the gap is **training distribution + a few fields**, not a new paradigm.

| Target | Encoding strategy | Train distribution |
|--------|-------------------|--------------------|
| **Arbitrary stacks** | Keep log1p SPR per seat, stacks/bb, pot/bb; add **effective SPR** and **commit/stack** ratios | Roots at SPR grid 0.5–40+, asymmetric stacks, not only 100bb |
| **Arbitrary bet sizes in history** | Keep log1p(chips/pot) history (already); **never** one-hot abstract size IDs as the only channel | Solver trees with varied size menus; at train time inject off-grid sizes mapped into history scalars |
| **Multiway 2–6** | Existing hero-rotated masks (active/all-in/exists/commits) pad-to-8 | HU-heavy early; gradually mix 3-way labeled data; 4–6 only when labels exist or as self-play regularizer |
| **Position / blinds / ante** | hero-is-SB/BB, street commits (blinds visible), pot includes ante | Match live ClubGG stake in roots |
| **Ranges as input (re-solve)** | Separate **belief encoder**: 1326×n_players sparse or bucket histogram + blockers | Not in current OBS; new module for search mode only |

**Interpolation principle:** all chip features **relative** (bb, pot frac, SPR); absolute chip counts only as secondary. Condition policy on **public state + hero hole** (current UI path); condition value net for re-solve on **public state + ranges** (DeepStack-style).

Do **not** rely on PPO multiway self-play alone to "generalize to GTO multiway" — without multiway equilibrium labels or search, it generalizes to *self-play equilibrium*, which is a different object.

---

## D) Inference-time options

```
Mode 0  Pure network          (ms)     study default
Mode 1  River exact re-solve  (0.1–2s) when street=river & ≤3-way
Mode 2  Street-limited CFR    (1–10s)  NN values at street boundary
Mode 3  Blueprint preflop     (ms)     table lookup / small net
        + Mode 0/2 postflop
```

| Mode | When to ship | Quality | Cost |
|------|--------------|---------|------|
| **Pure net** | First product | Good on-distribution; soft OOD | GPU/CPU ms — fits current UI |
| **Net + depth-limited re-solve** | After value net calibrated | GTOW-AI class on HU/3-way postflop | Need range tracking + CFR worker |
| **Blueprint preflop + resolve postflop** | After preflop MCCFR blueprint | Best preflop coherence | Blueprint storage + root matching |

**UI UX:** default Mode 0 for instant recs; "Deep solve" button → Mode 1/2 with progress; show both π_net and π_resolve when available (trust calibration metric).

---

## E) Phased roadmap from current NLH PPO

### Is PPO pretraining still useful?

| Use | Yes/No |
|-----|--------|
| GTO strategy target | **No** — wrong objective in multiway; HU PPO still ≠ CFR average strategy without regret machinery |
| Encoder / torso warm-start | **Maybe** — transfer public-state features; freeze early layers briefly |
| Sampling policy for data collection | **Yes** — visit diverse lines while solvers label offline |
| Exploiter / LBR probe opponent | **Yes** — keep `exploit.py` lineage |
| Production study recommendations long-term | **No** as sole source — replace with supervised π* / re-solve |

**nlh5 PPO anneal** remains a *parallel* track for "strong self-play agent" if you want a bot; it is **orthogonal** to the GTO estimator track. Do not block GTO work on nlh5 entropy.

### Shared Phase 0 (all candidates) — CPU / no GPU contention

1. **HU postflop label pipeline**
   - Root generator: stacks, pot, ranges (or full random), board, line template (SRP/3bet), ante-aware
   - Solver backend: native rust_cfr (`cfr_solve`)
   - Normalize → `NLH_ANCHOR_SPEC` targets + CFVs
2. **Dataset schema** (`data/gto_nlh/…`): parquet/arrow with public obs, hero combo, π*, v*, reach, meta
3. **Metrics:** HU local exploitability proxy (re-solve best response on held-out roots); KL(π_net ‖ π*); action agreement on pure nodes; EV loss vs solver
4. **NLH probe bank** (from nlh5 design): still useful, but scored **against solver labels**, not vibes
5. **Do not** prune `nlh*` / `vFour*` checkpoints; do not touch live PLO pod

---

## Ranked architecture candidates

### Candidate 1 — **Supervised GTO Net (library → imitate)**  ★ recommended first

**Idea:** Offline CFR library (HU postflop → expand) trains a policy+value net; UI serves pure net. Optional cheap river exact re-solve later without full DeepStack.

```
[Root sampler] → [native rust_cfr] → [π*, v* dataset]
        → [Supervised train: CE(π)+Huber(v) on encoding_nlh + small head delta]
        → [UI Mode 0]
        → [Optional Mode 1 river exact]
```

| | |
|--|--|
| **Feasibility** | Highest — reuses engine, encoding, anchors, UI; no live CFR loop required for v1 |
| **Team fit** | 1 eng solver pipeline + 1 eng train/serve |
| **Risk** | OOD on unseen lines/stacks; multiway absent until labeled |
| **GTO honesty** | High on covered roots; must show coverage badges in UI |

**Phase gates**

| Gate | Success metric |
|------|----------------|
| G0 Label | 10k+ HU river/turn roots solved; π* folds pure trash, jams pure nuts on smoke tests |
| G1 Imitation | Held-out KL < threshold; pure-node agreement ≥ 90%; mean EV loss ≤ 0.5% pot vs solver on river |
| G2 Stack gen | Same metrics on SPR not in train grid (interpolation) |
| G3 Product | UI "GTO net" mode; latency < 50ms; coverage map documented |
| G4 Multiway | 3-way river labels; separate metrics; UI flags 3-way as experimental |

**PPO role:** encoder warm-start optional; exploiter for residual gaps.

---

### Candidate 2 — **Hybrid DeepStack / GTOW-AI (value net + depth-limited re-solve)**  ★ quality endgame

**Idea:** Train **range-conditioned value/CFV nets** on self-play + solver subgames; at inference run **one-street (or depth-limited) CFR** with NN leaves. Blueprint for preflop.

```
Offline: MCCFR subgames + self-play CFV targets → ValueNet(public, ranges)
Online:  track ranges via Bayes(blueprint/π) → build street tree → CFR → π_resolve
Serve:   Mode 0 = PolicyNet; Mode 2 = re-solve (3s target)
```

| | |
|--|--|
| **Feasibility** | Medium — need range tracking, tree builder, CFR core, value net API |
| **Team fit** | Serious Rust CFR investment; matches long-term product ceiling |
| **Risk** | Belief error compounds; multiway search cost; more moving parts |
| **GTO honesty** | Best of the three when search budget adequate |

**Phase gates**

| Gate | Success metric |
|------|----------------|
| G0 | Candidate 1 G1 passed (value head already useful) |
| G1 CFV | Value net predicts river CFVs within X bb of exact on held-out |
| G2 Search | HU flop re-solve Nash distance ≤ ~0.5% pot vs long CFR (local) |
| G3 Speed | p50 solve ≤ 5s/street on study hardware |
| G4 Blueprint | Preflop blueprint + postflop re-solve coherent (no huge line EV gaps) |

**PPO role:** obsolete as primary trainer; may generate diverse reach for CFV targets (Deep CFR-style).

---

### Candidate 3 — **Deep CFR / ReBeL end-to-end**  (research ceiling, slowest)

**Idea:** Neural regret or continual re-solving from scratch (Brown et al. Deep CFR; ReBeL). Built on native rust_cfr; maximum research surface.

```
External-sampling MCCFR with neural regret/advantage nets
  and/or ReBeL: DRL + search in belief space
```

| | |
|--|--|
| **Feasibility** | Lowest for small team wanting a study tool this year |
| **Team fit** | Needs dedicated research time; OpenSpiel for prototypes |
| **Risk** | High variance; long until UI-quality strategies; multiway worse |
| **GTO honesty** | Theoretically clean if converged; hard to verify multiway |

**Phase gates:** toy game (Leduc) exploitability → HU LHE → abstracted HUNL → only then product. **Not recommended as primary** unless Candidate 1/2 stall on principle.

---

## Comparison (decision matrix)

| Criterion | C1 Supervised net | C2 Hybrid re-solve | C3 Deep CFR/ReBeL |
|-----------|-------------------|--------------------|-------------------|
| Time to useful UI | **Weeks–months** | Months+ | Many months |
| Reuse of plodbnet | **Excellent** | Excellent + new CFR | Partial |
| Arbitrary stacks/sizes | Train dist + features | **Search handles OOD** | If trained broadly |
| Multiway path | Labeled 3-way later | **3-way search (hard)** | Hardest |
| Measurable GTO | On library support | **On any root (budget)** | Exploitability if 2p |
| Ops complexity | Low | Medium-high | High |
| Small-team rank | **#1** | **#2** | #3 |

**Recommended path:** **C1 → grow library → C2**, with C3 only as research spikes (e.g. Deep CFR value targets feeding C2).

---

## Concrete Phase plan (C1→C2)

```
Phase 0  (now, CPU)     Label factory + dataset schema + solver smoke tests
Phase 1  (GPU small)    Supervised Policy+Value on HU postflop library
Phase 2  (product)      UI GTO mode + coverage + metrics dashboard
Phase 3  (expand)       Stack/line density; preflop blueprint; 3-way river
Phase 4  (C2)           Street-limited re-solve + range tracker
Phase 5  (optional)     Dynamic sizing at re-solve; QRE-style soft response
```

**Parallel (optional):** nlh5 PPO for self-play bot / encoder pretrain — **separate stem, separate success metrics**, never marketed as GTO.

---

## Success metrics (product-level)

1. **HU river EV loss** vs exact CFR ≤ 0.3–0.5% pot on held-out roots  
2. **Pure-node agreement** (fold/jam clearly pure) ≥ 95%  
3. **Interpolation:** SPR and bet sizes unseen in train, EV loss < 2× in-dist  
4. **Latency:** Mode 0 < 50 ms; Mode 2 p50 < 5 s  
5. **Honesty:** UI never labels multiway output "Nash" without search+caveat  
6. **Regression:** probe suite fails build if fold-canary / card-blind policy returns  

---

## Explicit non-goals (near term)

- Claiming 6-max full-tree GTO  
- Replacing PLO training on the pod  
- Using multiway PPO frequencies as supervised targets  
- Building nodelocking/ICM before HU postflop quality is real  

---

## Open decisions for you

1. **Primary candidate?** Default recommendation: **C1 then C2**.  
2. **Label backend:** locked — native rust_cfr only.  
3. **Product priority:** pure-net study speed vs willingness to wait for re-solve quality?  
4. **Keep nlh5 PPO track alive in parallel**, or freeze NLH PPO until GTO Phase 1 ships?  
5. **Ante structure:** lock to ClubGG $5/$10 + $5 ante as the canonical root family?  

No code or training launches in this design step; next implementation plan should start at Phase 0 label factory only after you pick the candidate and backend.
