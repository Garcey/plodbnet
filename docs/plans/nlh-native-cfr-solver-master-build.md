# NLH Native CFR Solver — Master Build Plan

**Status:** multi-agent plan merge (2026-07-16)  
**Priority:** build this while **PLO5 keeps training**; Mode 0 teacher is native rust_cfr only  

**Product vision:** **MonkerSolver-class capability** (preflop → river, multiway-capable) + **our wedge** (batch solves, scripting, ClubGG/`rust_engine` parity)

### Sub-plans (detail)

| Topic | File |
|-------|------|
| Algorithms + game tree | `docs/plans/nlh-cfr-algorithm-game-tree-architecture.md` |
| Engine integration | `docs/plans/nlh-cfr-engine-integration-phase-1-3.md` |
| Card / size abstraction | `docs/plans/nlh-cfr-card-bet-size-abstraction.md` |
| Batch / script / export | `docs/plans/cfr-batch-solving-scripting-export-ops-layer.md` |
| Original scaffold plan | `docs/plans/nlh-preflop-river-cfr-solver.md` |

---

## 0. North star (one sentence)

A **scriptable, batchable** NLH solver that can open a **preflop or multiway root**, solve with **abstractions**, induce ranges street-to-street, export strategies/labels on **our chip rules**, and later feed Trainer/PolicyNet — inspired by Monker’s *scope*, not by shipping Monker.

```
Config / CLI
    → Root grid (batch)
    → Rust CFR core (DCFR | MCCFR)
    → Strategy JSON
    → (later) LabelRecord → PolicyNet / re-solve
```

---

## 1. Algorithm defaults (decisive)

| Regime | Algorithm | Notes |
|--------|-----------|--------|
| Toy (Kuhn/Leduc) | Vanilla / Linear CFR | Exploitability → 0 proof |
| **HU postflop subgame** | **DCFR** full tree | Phase 1 default |
| **HU preflop blueprint** | **External-sampling MCCFR** | Phase 2 |
| **Multiway 3–6** | MCCFR only | Phase 3; population eq language |
| Depth-limited resolve | Later optional | Not Phase 1 |

**Not day-1:** DeepCFR, GPU CFR, continuous bet sizes.

**Full-hand glue (not one mega-tree):**

```
Preflop MCCFR blueprint
    → path / action Bayes update on ranges
    → postflop DCFR (or MCCFR) subgame with induced ranges
    → river exact when cards + combos fit
```

---

## 2. Tree + engine integration

### 2.1 Layers

| Layer | Role |
|-------|------|
| **Public tree** | Board, pot, stacks, commits, to_act, abstract actions |
| **Infosets** | (player, public history, private view) → regrets + average σ |
| **Chance** | Deal holes / streets (iso-compressed or sampled) |
| **Terminal** | Fold win or showdown EV (`evaluate_nlh` / side pots) |

### 2.2 Engine

| Use engine for | Do **not** |
|----------------|------------|
| Chip rules: min/max raise, short all-in reopen, antes/blinds | Clone `GameState` per tree node |
| `apply_raise_chips` parity tests | Reuse PPO 8-action enum as CFR menu |
| Showdown payouts | Study-mode zero payouts as solve path |

**Critical gap:** mid-street **solver root** (fixed board + pot + stacks, no re-post blinds).  
**Fix:** CFR `PublicState` owns tree transitions; thin `GameState::from_solver_root` (or equivalent) for parity.

### 2.3 Target module tree (`rust_engine/src/cfr/`)

```
cfr/
  mod.rs, types.rs          # Phase 0 ✅
  public_state.rs           # compact public node
  actions.rs                # FOLD|XC|RAISE_pm|ALLIN → chips
  engine_bridge.rs          # parity apply / from_solver_root
  range.rs                  # 1326 HU ranges, blockers
  tree.rs                   # build / traverse
  infoset.rs                # regret tables
  showdown.rs               # EV leaves
  dcfr.rs                   # Phase 1
  br.rs                     # best-response exploitability
  export.rs                 # strategy JSON
  toy.rs                    # Kuhn/Leduc
  preflop.rs, mccfr.rs      # Phase 2
  induce.rs                 # range induction
  multiway.rs, card_abs.rs  # Phase 3
```

---

## 3. Abstraction (solve vs serve)

### Hard rules

1. **Solve ladder ≠ serve ladder** — CFR uses coarse sizes; labels/UI use full `NLH_ANCHOR_SPEC` (12 atoms) at **export only**.  
2. One chip formula: pot-after-call + per-mille (same as `sizing.py` / engine).  
3. **Refuse, don’t OOM** — estimate memory; fail with coarser suggestion.

### Size presets

| Preset | Interior raises (pm) | Use |
|--------|----------------------|-----|
| `micro` | 500, 1000 | 6-max preflop; multiway crisis |
| `coarse` | 330, 500, 1000, 1500 | HU flop; 3-way; batch default |
| `standard` | 330, 500, 750, 1000, 1500 | HU river/turn (Phase 0 default) |
| `fine` | + more fracs | High-value HU re-solve only |

### Card abstraction

| Regime | Cards | Iso |
|--------|-------|-----|
| HU river | **exact** | **on by default** |
| HU turn | exact if RAM else buckets | on |
| HU flop | **OCHS-style ~200 buckets** (exact refused) | on |
| Preflop | **169** classes | n/a / suit classes |
| Multiway | heavier buckets + `micro` sizes | on |

### Memory ballparks (order of magnitude)

| Solve | RAM | Notes |
|-------|-----|--------|
| HU river exact + standard sizes | ~0.1–2 GB | Phase 1 target |
| HU flop OCHS@200 + coarse | ~2–15 GB | Phase 1b/2 |
| HU flop exact | 100+ GB | **refuse** |
| 3-way river abstract | ~2–20 GB | Phase 3 |
| 6-max preflop 169 + micro | ~0.5–10 GB table | MCCFR hours–day |

---

## 4. Ops layer (our wedge)

| Script | Role |
|--------|------|
| `scripts/cfr_solve.py` | One root (exists Phase 0 — extend) |
| `scripts/cfr_batch.py` | Grid + resume + manifest |
| `scripts/cfr_export_labels.py` | Strategy → `LabelRecord` JSONL |

**Parallelism:** outer Python workers (roots) × inner Rayon (`thread_num`); cap `workers × threads ≤ CPUs`.

**Outputs:** atomic strategy JSON per root; batch manifest; labels with `source=rust_cfr_*` only for production GTO teacher.

**No re-poison:** never mix TS + rust_cfr in one train shard without explicit tag; badge allowlist later for `rust_cfr_*` + probe.

**Ops metrics:** roots/hour, seed determinism, resume-on-failure, CI smoke on stub then real solves.

---

## 5. Phased build (merge of all agents)

### Phase 0 — Scaffold ✅

- Types, validate, stub `solve`, Python CLI, tests

### Phase 1 — HU postflop vertical slice (next)

**Goal:** one real river solve that is engine-legal and measurable.

1. `PublicState` + abstract actions → chips  
2. Tree build for river (exact cards, given ranges)  
3. DCFR iterations + average strategy  
4. Best-response exploitability in bb  
5. Engine parity tests (short all-in, min-raise, ante-free postflop root)  
6. PyO3 bind `cfr_solve`  
7. Extend CLI + **batch** shell for N river roots  

**Exit criteria:** exploitability trending down on toy + small real river; strategy JSON dump; batch resumes.

### Phase 1b — Toy gates + turn

- Kuhn/Leduc unit tests (known values)  
- Turn (exact or light abs)

### Phase 2 — Preflop + flop abs

- 169 preflop MCCFR blueprint  
- Flop with OCHS-style buckets + `coarse` sizes  
- **Range induction** API: blueprint action → updated ranges  

### Phase 3 — Full-hand + multiway

- Scripted: preflop → postflop resolve pipeline  
- 3-way then 4–6 with `micro` sizes + heavy card abs  
- Honest language: population equilibrium / non-unique NE  

### Phase 4 — Product consumers

- `cfr_export_labels` → PolicyNet train optional  
- Trainer Mode 1 re-solve with path ranges  
- Badge: `source=rust_cfr_*` + probe only  

---

## 6. Explicit non-goals (until phase says so)

| Item | Until |
|------|--------|
| Mode 0 200-root TS train as main path | User asks / Phase 4 |
| Pause PLO5 pod | Never for this work |
| Multiway “unique GTO” marketing | Never without honesty note |
| GPU / DeepCFR | After tabular works |
| Reverse-engineer Monker binary | Never |
| Exact HU flop full tree | Memory gate refuse |

---

## 7. Correctness checklist (every phase)

- [ ] Chip apply matches engine on ClubGG blinds/ante/short-all-in  
- [ ] Deterministic given seed + config  
- [ ] Exploitability (or MCCFR proxy) logged  
- [ ] Ranges renormalized with blockers  
- [ ] Solve→export size map documented  
- [ ] Batch resume + manifest  
- [ ] No silent multiway Nash claim  

---

## 8. Immediate next step (when user says go)

**Phase 1 start:** implement `public_state` + `actions` + river tree + DCFR + BR exploitability + parity tests — still no preflop/multiway until that slice is green.

```
rust_engine/src/cfr/public_state.rs
rust_engine/src/cfr/actions.rs
rust_engine/src/cfr/dcfr.rs
rust_engine/src/cfr/br.rs
tests (Rust) + extend scripts/cfr_solve.py when bound
```

---

## 9. How this relates to Mode 0 work already done

| Keep | Defer |
|------|--------|
| Label schema, PolicyNet, probe, badge | Workstream C scale train until teacher expl is gated |
| ClubGG roots, NLH anchors | Mode 2 ValueNet / street re-solve |
| StrategyBackend / Trainer seam | GTO AI promote until rust_cfr labels + probe |

Teacher is native rust_cfr only.
