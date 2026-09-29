# NLH GTO — "Strong AI You Can Train Against"

**Focus:** real-time **play-against** (Trainer tab), not one-shot study recs.  
**Premise:** GTO Wizard’s gap is real — their Trainer drills against **precomputed libraries**, not a live custom AI. Our edge is a **serving architecture that always lets you play**, with strength that climbs without breaking latency.

Existing substrate (keep):
- Rust `nlh_single` + study/ranges UI
- `python/plo5bp/ui/trainer.py` already deals hands, **samples** `model_policy(deterministic=False)` for non-hero seats, scores hero vs net dist, MC EV-loss rollouts, deterministic reseed by `(hand seed, action prefix)`
- C1/C2 design freeze in `docs/plans/nlh-neural-gto-solver-ground-up-architecture.md`

---

## 1. Real-time play vs one-shot study (public lineage)

| System | Core idea | Decision cost | Best fit |
|--------|-----------|---------------|----------|
| **DeepStack** | Continual re-solving every decision; CVN at depth limit | ~seconds / decision (HU) | Strongest HU **play**, heavy for multi-seat trainer + MC rollouts |
| **Libratus / Pluribus** | Offline **blueprint** + **depth-limited nested search** at act time | ~0.1–few s / key decision | Multiway / full-hand coherence; search budget controllable |
| **ReBeL / SoG** | PBS value (+ policy) nets trained so search queries generalize | Search + NN leaves | Long-term quality ceiling; not day-1 product |
| **GTOW AI (public)** | Street-limited solve + NN future values (~3s/street); multiway ≤3 postflop | Study-oriented | One-shot / per-street study, **not** snappy ring trainer |
| **Pure policy net** | Forward pass → sample π | **&lt;50 ms** | Only path that supports **many sequential acts + MC rollouts** |

**Research consensus for product strength:**

```
Blueprint + nested/street search  ≈  Continual re-solving + CVN
    >> pure supervised π* net
      >> Deep CFR average-strategy alone
        >> multiway PPO self-play
```

**Critical product split** (not in prior plans as a first-class serving rule):

| Surface | Latency budget | Decisions / hand | Right default |
|---------|----------------|------------------|---------------|
| **Study rec** | User can wait 1–5s | 1 (current node) | Mode 0 instant + optional Mode 1/2 “Deep solve” |
| **Trainer opponent** | Must feel live | 5–20+ AI acts + MC rollouts | **Pure PolicyNet sample** as the spine |
| **Trainer scoring / review** | Can be async or post-hand | 1–N hero nodes | Net score now; re-solve polish later |

**Implication:** architectures optimized only for study (re-solve every click) **fail** as trainer opponents unless you pay seconds × N acts × MC paths. Superhuman play AIs (DeepStack/Pluribus) re-solve because **one** high-stakes match can wait; a drill loop cannot.

---

## 2. Trainer opponent tradeoffs

| Pattern | Strength | Latency | Coherence | Verdict for us |
|---------|----------|---------|-----------|----------------|
| **A. Pure policy every decision** | Soft OOD; strong on-dist if C1 is good | Best (ms) | Good if net trained full-hand | **Default spine always** |
| **B. Re-solve every decision (DeepStack)** | Highest HU quality | Worst (s × acts × MC) | Excellent | Only “Pro spar” HU, optional |
| **C. Blueprint + re-solve hero only; pure net for opp** | Strong review; opp still fast | Opp ms; hero optional s | Opp/hero may disagree slightly | **Best hybrid for Trainer+Study** |
| **D. Cached street strategy** (solve once at street root, play out) | Stronger than pure net on that street | 1–5s once / street, then ms | Excellent within street | **Best “Strong” tier (HU)** |
| **E. Full library trainer (GTOW)** | Exact on covered trees | Lookup ms | Brittle OOD; no custom AI | **Skip** — this is their limitation |

**Recommended serving architecture (Trainer):**

```
                    ┌──────────────────────────────────────┐
                    │           Trainer Session            │
                    │  deal → advance → score → MC EV-loss │
                    └───────────────┬──────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
     Opp seats (always)     Hero at decision        Post-hand / Review
     ─────────────────      ────────────────        ──────────────────
     PolicyNet SAMPLE       PolicyNet score         Optional:
     (Mode 0, ms)           (same π, instant)       Mode 1 river exact
     optional StreetCache   optional "Exam" re-solve  Mode 2 street re-solve
     if Strong+HU           for hero only             for marked nodes
```

**Rules of thumb:**
1. **Opponent path never blocks on CFR** in default tiers (T0–T3).
2. **Hero scoring** uses the **same** π the opp samples from (fair drill), not a stronger hidden solver — unless user opts into “Exam mode” (hero graded vs re-solve, opp still net).
3. **MC EV-loss rollouts** always pure-net (or street-cache lookup); never re-solve inside MC.
4. **Determinism** stays as today: sample reseeded from `(hand_seed, prefix)` so Repeat hand works.

---

## 3. Trainer Strength Ladder (ship while always playable)

Every tier **must** support Play-against on day of ship. Study quality can lag one tier behind.

| Tier | Name | Opp acts | Hero score | When | What user feels |
|------|------|----------|------------|------|-----------------|
| **T0** | **Self-play bot** | nlh PPO sample | vs PPO dist | **Already shipped** | Playable multiway; soft / not GTO |
| **T1** | **GTO Net** | C1 PolicyNet sample | vs π\* net | After C1 Phase 1–2 | Instant GTO-flavored drill; HU postflop first, full-hand as labels expand |
| **T2** | **Net + River Judge** | PolicyNet | Instant net + **async river exact** on river hero nodes | C1 + Mode 1 | Play stays fast; river grades tighten |
| **T3** | **Street Cache (HU)** | At street start: one Mode 2 solve → cache π; sample from cache | Same cache π | C2 early, **HU only** | Stronger, still live after 1–5s “thinking…” per street |
| **T4** | **Exam mode** | PolicyNet (or T3 cache) | Hero graded vs **re-solve** (Mode 1/2) | C2 mature | Hardest honest feedback; opp still responsive |
| **T5** | **Pro spar (HU)** | Continual re-solve both sides (budgeted) | Same | Late polish | DeepStack-class; slow; optional flag |

**UI strength picker (simple):**
- `Fast` → T1 (always)
- `Strong` → T3 when HU + ValueNet ready; else T1 + badge “Strong needs HU”
- `Exam` → T4
- `Pro` → T5 (HU, confirm latency)

**Coverage badges (required honesty):**
- seats, street coverage (preflop blueprint? postflop library?), SPR band, “equilibrium estimate” vs “Nash (HU re-solve)”

**Incremental ship order (always playable):**
1. Keep T0 wired for NLH (already).
2. Swap trainer backend to C1 PolicyNet → **T1** (biggest product jump).
3. Add river exact scorer path → **T2**.
4. Street-cache solver worker → **T3**.
5. Exam / Pro as opt-in.

---

## 4. Multiway × play-against

| Fact | Product consequence |
|------|---------------------|
| Multiway NE is **non-unique** | Never label ring trainer “the GTO”; use “equilibrium under shared blueprint / population play” |
| Pluribus multiway = blueprint + short search, opponents modeled as **same strategy class** | Trainer should use **one PolicyNet / one blueprint** for all AI seats (already true in `trainer.py`) |
| Re-solve cost explodes with seats | **T3–T5 hard-cap:** re-solve / street-cache **HU only** (optionally 3-way river later). Ring tables stay **T0/T1 pure net** |
| User wants ring eventually | Train PolicyNet with **HU-heavy + gradual 3-way labels**; 4–6 seats = net generalization + honesty flags, **not** full multiway search day-1 |
| Belief / ranges multiway | Bayes range track for C2 search is HU/3-way only; ring trainer does **not** need per-seat range CFR |

**Practical multiway ladder:**
- Ring play-against: **T1 forever** as the workhorse (ms, N seats).
- HU: climb T1→T5.
- 3-way: T1 + optional T2 river; T3 only if search budget proven.

Do **not** block ring trainer on multiway CFR. That is how products die.

---

## 5. What NOT to build (GTOW-shaped distractions)

Skip so strength + trainer stay primary:

| Skip | Why |
|------|-----|
| **Library-only trainer** (precomputed solution packs as sole opp) | Exactly GTOW’s limit; we already have live sampling |
| **Full 6-max offline GTO claim** | Impossible honestly; multiway NE non-unique |
| **Node locking / custom tree editor / ICM / MTT** | Study power-user surface; zero trainer strength |
| **Massive precomputed solution browser** as core UX | Storage + stale; re-solve/net generalize better for custom stacks |
| **QRE / “human-like” mixing first** | Cosmetic; build after real π\* |
| **OCR / PokerNow for NLH** | Orthogonal; PLO path exists |
| **nlh5 PPO anneal as GTO path** | Parallel bot only; wrong objective |
| **Re-solve inside MC EV-loss** | Latency bomb |
| **Per-opponent exploit modeling** | Different product (exploiter); dilutes GTO trainer |
| **Dynamic continuous sizing head day-1** | Anchors + off-tree insert later; one sizing language |
| **PLO dual-board GTO** | Out of scope for this ladder |

**Do build (thin):** coverage badges, strength picker, HU re-solve worker, range Bayes for search modes, probe bank vs solver labels.

---

## 6. How this changes C1 → C2 phase priorities

Prior plans ordered for **study quality**. Trainer-first reorders **serving** and **data**, not the scientific spine (still C1→C2, not pure PPO).

### Keep
- C1 supervised PolicyNet (+ value head) from HU postflop library
- C2 street re-solve + ValueNet leaves as quality ceiling
- Rust engine as ground truth; AGPL-safe label boundary
- nlh PPO frozen as T0 only / optional encoder warm-start

### Change (trainer-first)

| Old emphasis | New emphasis |
|--------------|--------------|
| Mode 0 study rec as first product | **T1 trainer play-against** as co-equal first product (same PolicyNet) |
| Mode 2 “Deep solve” button early | Mode 2 first used as **T3 street-cache** and **T4 exam**, not default every click |
| ValueNet only for study re-solve | ValueNet still for C2; **PolicyNet latency + sampling quality** is the trainer critical path |
| Preflop blueprint “Phase 3 expand” | **Promote blueprint earlier** — trainer hands start preflop; incoherent preflop π ruins “play against” feel |
| Full-hand labels secondary | Add **full-hand reach-weighted trajectories** (not only isolated postflop roots) so Mode 0 plays coherent streets |
| MC EV-loss after strong solve | Keep MC on pure net; never gate trainer ship on re-solve |
| Multiway search ambition | Explicit **ring = pure net**; search budget reserved HU (3-way later) |

### Revised phase priorities

```
Phase 0   Label factory (HU postflop) + metrics + smoke  [unchanged, CPU]
Phase 1   Supervised PolicyNet (+V)                      [unchanged train]
Phase 2a  ★ TRAINER T1: wire PolicyNet into trainer.py
          (sample opp, score hero, badges, strength=Fast)
          Study Mode 0 same net
Phase 2b  Preflop blueprint (or preflop-labeled net)
          so full-hand trainer is coherent — pull forward from old Phase 3
Phase 2c  Mode 1 river exact → Trainer T2 (score/review only)
Phase 3   Denser library, stack OOD, 3-way river labels (net only)
Phase 4   C2 ValueNet + street CFR worker
          → Trainer T3 street-cache (HU), Study Deep solve, T4 Exam
Phase 5   T5 Pro spar optional; dynamic K-size; 3-way search experiments
```

**Success metrics (add trainer-specific):**
1. Opp act p99 **&lt; 50 ms** on T1 (CPU or GPU).
2. Full 6-max hand wall-clock “feels live” (no multi-second stalls between acts).
3. Repeat hand: identical opp lines until hero deviation (keep current reseed contract).
4. HU river EV loss / pure-node gates from prior plan (quality).
5. Strength picker never offers T3+ on multiway without auto-downgrade + reason.

### Open product rules (locked by this design)
1. **Default trainer path = pure PolicyNet sample** for all AI seats.
2. **Re-solve is budgeted compute**, not the default act loop.
3. **Asymmetric modes OK:** stronger grading for hero than opp latency path.
4. **Ring tables = T0/T1 forever** until multiway search is proven; no fake “Strong GTO 6-max.”
5. **C1 PolicyNet is dual-use** (Study + Trainer); C2 raises ceiling without replacing the spine.

---

## 7. One-line recommendation

**Serve the trainer as a pure GTO PolicyNet opponent (T1) from day one of C1, climb strength via street-cached / hero-only re-solve (T3–T4) on HU only, and never make continual multiway re-solve a dependency of “play against.”**

This is the architecture that beats GTOW’s library-trainer limitation while matching DeepStack/Pluribus/GTOW-AI science for the **quality ceiling**.
