# NLH Neural GTO Solver — Ground-Up Architecture

**Status:** design frozen (v2 — Trainer-first product). Phase 0 + StrategyBackend seam **started 2026-07-16**.  
**Not** a continuation of nlh5 PPO anneal.

### Implementation progress

| Piece | Status | Location |
|-------|--------|----------|
| Frozen decisions §10 | Done | this file |
| `StrategyBackend` + `PpoSolverHost` (T0) | **Shipped** | `python/plo5bp/gto/backend.py` |
| Trainer wired through backend | **Shipped** | `python/plo5bp/ui/trainer.py` (`act` / `_advance` / review / what-if) |
| Coverage badge in trainer state | **Shipped** | `trainer.backend` payload |
| ClubGG 5/10($5) root sampler | **Shipped** | `python/plo5bp/gto/roots.py` |
| Label schema + JSONL IO | **Shipped** | `python/plo5bp/gto/labels.py` |
| Native rust_cfr solver | **Shipped** | `rust_engine/src/cfr/`, `gto/cfr_api.py` |
| Label factory CLI (smoke only) | **Shipped** | `scripts/gto_label_factory.py` |
| Metrics + river exact hook | **Shipped** (`cfr_solve`) | `python/plo5bp/gto/metrics.py` |
| PolicyNet + supervised train | **Shipped** | `gto/policy_net.py`, `gto/train.py`, `scripts/gto_train.py` |
| Teacher distill dataset | **Shipped** | `gto/dataset.py` (2–6 seats, ClubGG stakes) |
| `PolicyNetHost` T1 swap | **Shipped** | `gto/policy_host.py`; `PLO5BP_GTO_CHECKPOINT` |
| Smoke ckpt | `checkpoints/gto_policy.pt` (**rule bootstrap**, not GTO) | `PLO5BP_GTO_CHECKPOINT` |
| Rule bootstrap data | **Shipped** | `gto/bootstrap.py` |
| Study Mode 0 via GTO host | **Shipped** | `server.py` + `PLO5BP_GTO_CHECKPOINT` |
| nlh1–nlh4 PPO checkpoints | **Deleted** (failed lineage; not GTO path) | — |
| CFR batch / overnight | **Shipped** | `scripts/cfr_batch.py`, `scripts/cfr_overnight.py` |
| CFR → LabelRecord export | **Shipped** | `gto/cfr_export.py`, `scripts/cfr_export_labels.py` |
| Obs from solver labels | **Shipped** | `gto/obs_from_label.py` |
| Train from rust_cfr JSONL | **Shipped** | `scripts/gto_train_from_labels.py`, `scripts/train_policy_from_cfr.py` |
| Probe vs labels | **Shipped** | `gto/probe.py` |
| Scale teacher batch | Next (overnight local CPU) | HU river SPR grid, expl-gated |

```powershell
# river batch (local CPU; leave PLO pod alone)
.venv/Scripts/python scripts/cfr_batch.py --streets 3 --resume
.venv/Scripts/python scripts/cfr_export_labels.py `
  --strategies data/cfr/verify/batch/strategies `
  --out data/gto_nlh/cfr_labels.jsonl
.venv/Scripts/python scripts/gto_train_from_labels.py `
  --labels data/gto_nlh/cfr_labels.jsonl `
  --epochs 12 --out checkpoints/gto_policy.pt
set PLO5BP_GTO_CHECKPOINT=checkpoints/gto_policy.pt
```

PLO5 runpod training is **left running** (user mandate 2026-07-16).

### How to train & serve T1

```bash
# Default: rule-based pure-node curriculum (NO PPO teacher)
.venv/Scripts/python scripts/gto_train.py \
  --bootstrap --n-decisions 8192 --epochs 10 \
  --hidden-dim 512 --num-layers 2 \
  --out checkpoints/gto_policy.pt

# Serve Trainer + Study T1
set PLO5BP_GTO_CHECKPOINT=checkpoints/gto_policy.pt
# launch UI; switch format to NLH
```

**Honesty:** bootstrap is a curriculum prior (trash folds / nuts jams / free checks), **not** equilibrium. Badge `"GTO AI"` only for `source=rust_cfr*` **and** holdout probe pass. **Do not** reintroduce the retired nlh PPO stem as a teacher.

---

## 0. Product north star (user mandate)

### What we are building

A **strong NLH AI solver** you can study with **and play against**.

| Goal | Priority |
|------|----------|
| Strong enough that the **Trainer tab is the main practice surface** | **#1 — ships with first GTO product** |
| Study recs + Ranges grids from the same AI | #2 — same backend, co-ship |
| Approach GTO Wizard pure strength / feature depth | Ideal ceiling, **not** a day-1 requirement |

### The differentiator vs GTO Wizard

GTO Wizard’s Trainer only drills against **pre-solved solution libraries**. Even their highest tier with custom multiway AI solving does **not** let you **play** against those custom solutions in the trainer — browse/solve only.

**Our tool must let the user play full hands in Trainer against the live AI strategy** (custom stacks, seats, lines the net generalizes to), not only against a frozen library of trees.

That single product fact drives serving architecture:

> **Play-against requires a fast sampleable policy at every decision.**  
> Re-solve is a strength/quality layer, not the default opponent clock.

### What “done” means for v1 product

A user can:

1. Switch format to NLH, open **Trainer**
2. Play 2–6 seat hands with arbitrary stack settings
3. Opponents act from a **GTO PolicyNet** (mixed strategy, seeded for Repeat)
4. Hero is scored against the **same** π (fair drill)
5. Post-hand review shows frequencies / EV feedback
6. Optionally open Study/Ranges on the same backend for analysis

**Not required for v1:** beating GTOW Nash distance, multiway re-solve, full preflop library, nodelocking, ICM.

---

## 1. Premise (why not more PPO)

PLO5DBBP PPO does not transfer as a GTO estimator:

1. **No-limit sizing** is min→all-in. SOTA uses sparse abstract ladders + **off-tree nested re-solve**, not pot-capped Beta alone.
2. **NLH is CFR-solvable at subgame scale.** The net’s job is to **generalize equilibrium** across stacks, sizes, multiway.

Public systems (DeepStack, ReBeL, SoG, GTOW AI):

> Train **value/CFV (+ policy) nets** so a **real-time depth-limited CFR** can re-solve for arbitrary stacks/pots/ranges/sizes.  
> Library imitation is a **bootstrap**; search is the **quality ceiling**.  
> For **trainer play**, pure policy sampling is the **latency spine**.

Strength hierarchy:

```
Blueprint + nested/street search  ≈  Continual re-solving + CVN
    >> pure supervised π* net          ← Trainer default opponent
      >> Deep CFR average-strategy alone
        >> multiway PPO self-play
```

---

## 2. Serving architecture (Trainer is first-class)

### 2.1 Surfaces and latency budgets

| Surface | Decisions / hand | Budget | Default |
|---------|------------------|--------|---------|
| **Trainer opp seats** | 5–30+ AI acts | **p95 ≤ 50ms** per act | Mode 0 PolicyNet **sample** |
| **Trainer hero score** | 1 per hero act | ≤ 300–500ms interactive; ≤1s hard | Same π (fair); optional Exam = re-solve |
| **Trainer MC EV-loss** | 16 continuations today | Keep net-only | **Never** CFR inside MC |
| **Study rec** | 1 node | &lt;50ms instant + optional 1–5s deep | Mode 0 + “Deep solve” |
| **Ranges grid** | 1 batch forward | &lt;200ms | Mode 0 over combos |
| **Post-hand review** | N nodes | Can be slower / async | Mode 0; optional Mode 1/2 polish |

### 2.2 Mode table (with Trainer column)

| Mode | Latency | Quality | **Trainer opponent** | Study / Ranges | Ship |
|------|---------|---------|----------------------|----------------|------|
| **0 Pure PolicyNet** | &lt;50ms | Strong on-distribution GTO estimate | **Default all seats 2–6** | Default rec / grid | **Phase 2a MVP** |
| **1 River exact** | 0.1–2s | Near-exact HU/≤3 river | Hero score / review only | Optional deep | Phase 2c |
| **2 Street re-solve** | ~1–5s once / street | GTOW-AI class postflop | **HU Strong tier:** solve once/street, cache π, then sample; never every-seat multiway | “Deep solve” button | Phase 4 |
| **3 Blueprint preflop + 0/2 postflop** | ms + above | Coherent full-hand openers | Preflop sample blueprint; postflop 0/2 | Preflop study | Phase 3–4 |

### 2.3 Trainer Strength Ladder (always playable)

| Tier | Opp acts | Hero score | When | User-facing name |
|------|----------|------------|------|------------------|
| **T0** | nlh PPO sample | vs PPO | Now (existing) | “Self-play (untrained GTO)” |
| **T1** | **GTO PolicyNet sample** | vs same π | **First GTO ship** | **“GTO AI”** — killer MVP |
| **T2** | PolicyNet sample | + async river exact polish on hero | After Mode 1 | “GTO AI + river exam” |
| **T3** | **Street-cache re-solve** (HU) then sample | same cache / net | After Mode 2 | “Strong (HU)” |
| **T4** | PolicyNet (or T3) | **Exam:** grade vs re-solve, opp still net | Mode 2 mature | “Exam mode” |
| **T5** | Continual re-solve both sides | re-solve | Optional Pro, HU only | “Pro spar” — never default multiway |

**Invariant:** every tier above T0 must support **full multiway 2–6 play** via Mode 0. Search only upgrades HU (then ≤3).

### 2.4 StrategyBackend contract (single host for Trainer + Study + Ranges)

Today trainer hardwires PPO (`model_policy`, `compute_node_distribution` in `python/plo5bp/ui/trainer.py`). Abstract once:

```text
StrategyBackend
├── act(obs|public, seat, *, deterministic, rng_seed) -> (gate, chips)
├── node_distribution(obs|public, seat) -> NodeDist
│     gate_probs[3], anchor_*, rec_*, value_bb?, mode, coverage, think_ms
├── supports(seats, street, mode) -> bool
└── (Mode 2+) update_beliefs / street_cache_get
```

| Hook today | Backend method |
|------------|----------------|
| `_advance` → `self._policy` | `act(..., deterministic=False, rng_seed=_opp_seed)` |
| `act` scoring → `compute_node_distribution` | `node_distribution` → keep `score_move_v2` |
| `_rollout_ev` | `act` only (Mode 0); later host EV delta replaces MC |
| Study `_compute_recommendation` | `node_distribution` + deterministic rec |
| `ranges.py` grid | batch `node_distribution` / PolicyNet forward |

Adapters: `PpoSolverHost` (prove seam) → `PolicyNetHost` (T1) → `StreetCacheHost` (T3) → `ResolveHost` (T4/T5).

**Determinism:** keep `(hand_seed, action_log prefix)` seeding so **Repeat hand** works. Mode 2 must be deterministic given `(budget, seed)` or fall back Mode 0.

### 2.5 What already works (do not rewrite)

`TrainerSession` is already a GTOW-Practice-shaped shell:

- Random deal → multiway 2–6 → sample mixed opp → score hero vs node π → review freqs
- NLH format switch, per-format stacks/ante, 12-anchor scoring, seeded opp lines
- Routes `/trainer/*` and client animation stay

**Gap is the backend seam + GTO PolicyNet**, not the UX loop.

Optional later product (not MVP): custom-line drills (borrow Ranges `line` model), live freqs toggle while hero decides (today `recommendation: null` deliberately).

---

## 3. Target solver architecture (strength ceiling)

```
        ┌─────────────────────────────────────────────────┐
        │  Trainer  │  Study  │  Ranges  │  (future Exam) │
        └─────────────────────┬───────────────────────────┘
                              │ StrategyBackend
                    ┌─────────▼─────────┐
                    │   Solver Host     │
                    │  Mode 0 PolicyNet │
                    │  Mode 1 river CFR │
                    │  Mode 2 street+NN │
                    │  Mode 3 blueprint │
                    └────┬─────────┬────┘
           chips/legal   │         │ CFV / prior
                ┌────────▼──┐ ┌────▼──────────────┐
                │ Rust eng  │ │ Neural (new)      │
                │ KEEP      │ │ PolicyNet         │
                │ rules/pot │ │ ValueNet (ranges) │
                │ study API │ │ optional PriorNet │
                └───────────┘ └────▲──────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │ Offline Label Factory       │
                    │ RootSampler → MCCFR/TS     │
                    │ → (π*, v*/CFV*, reach)      │
                    │ Prefer full-hand trajectories│
                    └─────────────────────────────┘
```

### 3.1 Three neural objects

| Net | Input | Output | Trainer use |
|-----|-------|--------|-------------|
| **PolicyNet** | Public + hero hole (adapt `encoding_nlh`) | Gate × `NLH_ANCHOR_SPEC` (+ refine optional) | **Opp sample + hero score (T1+)** |
| **ValueNet / CVN** | Public belief + **ranges** | CFVs per hand/bucket | Mode 2 leaves; Exam scoring |
| **PriorNet** | optional | CFR warm-start | Mode 2 speed only |

Hero-obs-only PolicyNet **cannot** be the re-solve value function. Range-conditioned ValueNet is required for C2.

### 3.2 Action / sizing (NL-native)

One language end-to-end:

- Abstract menu = **`NLH_ANCHOR_SPEC`** (min … 275% + ALL-IN)
- Offline CFR labels map onto that menu
- History keeps **continuous pot-frac / log1p** (already in encoding)
- Off-tree sizes at re-solve: **insert into tree** (Libratus/Pluribus)
- Dynamic K-size later at inference only

### 3.3 Offline labels

| Choice | Recommendation |
|--------|----------------|
| Day-1 | **HU postflop** (unique NE, measurable) |
| Day-N | Preflop blueprint **earlier than pure study plans** (full-hand trainer coherence); 3-way river experimental |
| Teacher | **Native rust_cfr** on `rust_engine` (only production π*) |
| Curriculum | Rule bootstrap (not badge-eligible) |
| Sampling | **Reach-weighted full-hand trajectories** + SPR/street/pot strata — not isolated postflop roots only |
| Stacks | Randomize SPR 0.5–40+, asymmetric stacks |

---

## 4. KEEP / ADAPT / DROP

### KEEP

| Asset | Why |
|-------|-----|
| Rust NLH engine, blinds, min/max raise, short-shove, `apply_raise_chips` | Solver chip primitive |
| Study APIs + `pack_range_nlh` | Study + ranges |
| `NLH_ANCHOR_SPEC` | Action abstraction |
| `encoding_nlh` feature blocks | PolicyNet encoder ingredients |
| **Trainer UX** (`trainer.py`, `/trainer/*`, client) | **Primary product surface** |
| Rule tests (`test_nlh_*`, short-allin, ranges, chip conservation) | Regression wall |

### ADAPT

| Asset | Change |
|-------|--------|
| `trainer.py` | Inject `StrategyBackend`; drop hardwired PPO-only path for GTO format |
| `server.py` FORMATS | Host per format; Study + Trainer + Ranges share it |
| `encoding_nlh` | Public/private split; ValueNet gets range encoder |
| `compute_node_distribution` / scoring | Backend NodeDist; migrate EV-loss to host EVs when available |
| Probe suite | NLH GTO bank scored vs labels; plus **trainer playability** canary |

### DROP as GTO spine (PLO product stays)

PPO/rollout/selfplay as GTO trainer; ActorCritic as production GTO; multiway PPO freqs as labels; marketing PPO as GTO.

**PPO residual:** T0 trainer until T1 ships; optional encoder warm-start; never the long-term opp.

---

## 5. Phase plan (Trainer-first reorder)

```
Phase 0  (CPU, PLO holds GPU)
  Label factory + schema + metrics (KL, pure-node, EV loss % pot)
  Solver: native rust_cfr (`cfr_solve` / batch / export)
  Smoke: trash folds / nuts jam
  Do not touch live PLO pod

Phase 1  (small GPU / local)
  Supervised PolicyNet (+ value head) on HU postflop + growing full-hand traj
  Train dist: SPR grid, asymmetric stacks, varied history sizes
  Gate: pure-node ≥90%, river EV loss ≤0.5% pot held-out

Phase 2a  ★ PRODUCT MVP — KILLER DIFFERENTIATOR
  StrategyBackend seam (PpoSolverHost → PolicyNetHost)
  NLH Trainer T1: play 2–6 seats vs PolicyNet; score vs same π
  Coverage / “GTO AI” badges; refuse silent PPO-as-GTO when GTO host loaded
  Success: user completes multiway trainer hands; p95 opp act ≤50ms

Phase 2b
  Study Mode 0 + Ranges on same PolicyNetHost (browse parity with play)

Phase 2c
  Mode 1 river exact for hero review / Exam polish (not multiway opp)

Phase 3
  Preflop blueprint (Mode 3) — full-hand openers for trainer
  Denser library; 3-way river labels (experimental flags)
  Stack/size interpolation stress

Phase 4  (C2 hybrid)
  Range tracker + street-limited CFR + ValueNet leaves
  Study “Deep solve”
  Trainer T3: HU street-cache re-solve then sample
  Trainer T4: Exam mode (hero vs re-solve)
  Off-tree size insertion

Phase 5
  Dynamic K-size; multiway search cap ≤3 postflop; optional T5 Pro spar HU
```

**Key reorder vs study-only plans:** Phase **2a Trainer-vs-Mode0** is co-equal with (and ships **before or with**) Ranges — not after Mode 2 search.

---

## 6. Success metrics

### Strength (solver quality)

1. HU river EV loss vs exact CFR ≤ 0.3–0.5% pot held-out  
2. Pure-node agreement ≥ 95%  
3. SPR/size OOD EV loss &lt; 2× in-distribution  
4. Mode 2 p50 ≤ 5s/street on study hardware  

### Trainer / differentiator (product)

5. **Playability:** full 2–6 seat hand vs Mode 0; **p95 opp decision ≤ 50ms**  
6. **Fair drill:** Study argmax ≈ trainer score target at same node (agreement canary)  
7. **Repeat determinism:** same seed → same opp line until user deviation  
8. **Marketing-safe claim:** “Play Trainer against the live AI strategy (custom stacks/seats), not only pre-solved libraries.”  
9. **No silent PPO-as-GTO:** NLH GTO format requires PolicyNet load + badge  

### Honesty

10. Multiway never labeled plain “Nash” without search + caveat  
11. Collapse canaries fail the probe suite  

---

## 7. Explicit non-goals (near term)

| Skip | Why |
|------|-----|
| Beat GTOW on every feature | Ceiling, not gate |
| **Library-only trainer** (GTOW model) | **Opposite of our differentiator** |
| 6-max full-tree GTO claim | Impossible honestly |
| Mode 2 as multiway opp default | Latency kills UX |
| Re-solve inside MC EV-loss | Impossible in budget |
| Nodelock / ICM / MTT / PKO | Distraction before T1–T3 solid |
| Massive precomputed solution browser | Nice later; not required to play-against |
| OCR for NLH | Deliberate |
| Replace live PLO pod training | One family per pod |

---

## 8. Multiway policy

| Seats | Trainer opp | Search |
|-------|-------------|--------|
| 2 HU | Mode 0 default; T3 street-cache optional | Full C2 path |
| 3 | Mode 0 default | Experimental Mode 2 later |
| 4–6 | **Mode 0 only** (required forever until proven otherwise) | No interactive re-solve |

Multiway NE is non-unique — UI: “equilibrium-style AI / population play,” not “the GTO.”

---

## 9. Engineering layout (proposed)

```
rust_engine/src/
  cfr/              # NEW Phase 4: public tree, CFR+, reach, terminals under ranges
  abstract/         # NEW: anchors ↔ chips
python/plo5bp/
  gto/              # NEW: dataset, train, PolicyNet/ValueNet, StrategyBackend hosts
  encoding_nlh.py   # ADAPT public/private
  ui/trainer.py     # ADAPT backend injection (Phase 2a)
  ui/server.py      # ADAPT FORMATS → host
  ui/ranges.py      # ADAPT same host
scripts/
  gto_label_factory.py
  gto_train.py
  probe_suite_nlh.py
data/gto_nlh/       # gitignored parquet
```

---

## 10. Frozen decisions (2026-07-16)

| # | Decision | Lock | Rationale |
|---|----------|------|-----------|
| **1** | **Trainer-first C1→C2** | **YES** — T1 (Mode 0 PolicyNet play-against) is the first GTO product ship; C2 (Mode 2 street re-solve / T3 HU) is the strength ceiling, not the MVP gate | Differentiator is play-against live AI; study re-solve alone is GTOW-shaped and fails multiway trainer latency |
| **2** | **Phase 0 labels** | **Native rust_cfr only** (`source=rust_cfr*`) for train labels; river DCFR also scores EV-loss canaries | One teacher, one chip rule, no external solver |
| **3** | **T1 ship bar** | **Multiway 2–6 required on day one** of GTO trainer | v1 done criteria already list 2–6 seats; Mode 0 sample is free for N seats. Labels stay **HU-heavy** (unique NE, measurable); ring = net generalization + honesty badges (“equilibrium-style / population play”), never “the Nash” |
| **4** | **Canonical roots** | **ClubGG 5/10($5)** locked for Phase 0–2 train / probe dist | User lock. SPR strata and asymmetric stacks still randomize *around* this root (0.5–40+ SPR); root anchors realism and ClubGG study continuity |
| **5** | **nlh PPO** | **Keep as T0 until T1 ships, then freeze stem** | Playable multiway shell during C1 build. After T1: no further PPO as GTO path; keep frozen checkpoints as fallback “Self-play (untrained GTO)” if PolicyNet fails to load. Do **not** prune `nlh*` files; do **not** resume nlh5 anneal as the GTO plan |
| **6** | **Live freqs while hero decides** | **Post-act only for v1** — no spoil toggle in T1 MVP | Today’s `recommendation: null` is deliberate drill pedagogy. Spoil-toggle is Phase 2b+ optional polish; does not block play-against MVP |

### Implications for next work

1. **Phase 0** = native rust_cfr batch/export + JSONL schema + ClubGG 5/10($5) root sampler + river DCFR scorer for metrics.
2. **Phase 2a** = `StrategyBackend` seam + `PolicyNetHost` + multiway trainer T1; refuse silent PPO-as-GTO when GTO host is selected.
3. **PPO** stays wired as T0 only; guardian/stem freeze after T1 badge ships.
4. Trainer client: no live-freq UI work until after T1 playability green.

---

## 11. What I will not do without explicit go

- Touch the live PLO pod run  
- Prune `nlh*` / `vFour*` / PLO checkpoints  
- Ship NLH trainer labeled “GTO” while still on PPO  
- Claim multiway Nash without honesty layer  
- Block multiway trainer play on Mode 2 search  

---

## Research anchors

- DeepStack (2017) — continual re-solving + CVN (play-strong, latency-heavy)  
- Libratus / Pluribus (2017/2019) — blueprint + nested search  
- ReBeL (2020) / Student of Games (2023) — PBS / GT-CFR + nets  
- GTO Wizard AI (public) — street-limited + NN leaves; **trainer limited to libraries** (our gap)  
- Deep CFR / DREAM — references only; not product spine without search  

Supporting docs:  
`docs/plans/nlh-neural-gto-estimator-design.md` (pipeline options)  
`docs/plans/nlh-trainer-strength-architecture.md` (trainer ladder detail)
