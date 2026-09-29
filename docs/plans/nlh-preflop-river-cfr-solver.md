# NLH Preflop→River CFR Solver

> **Master build plan (merged multi-agent design):**  
> **`docs/plans/nlh-native-cfr-solver-master-build.md`**

**Priority (user 2026-07-16):** Do NLH **correctly** via a **native full-hand CFR stack** (preflop through river), with batch solves + scripting. PolicyNet trains on **our** rust_cfr labels only. PLO5 pod keeps training undisturbed.

---

## Product inspiration: MonkerSolver + our ops layer

**User intent (explicit):** MonkerSolver already shows that **preflop + multiway** solving is the right *capability* target for a serious solver product. We should **take architectural inspiration from that class of tool** (abstractions, multiway trees, preflop→river), **not** clone the GUI or depend on Monker as a library.

| Borrow from Monker-class solvers | Our differentiator |
|----------------------------------|--------------------|
| Preflop → river in one product story | **Batch solves** (grid of roots unattended) |
| Multiway (not HU-only forever) | **Scripting / automation** (CLI, config, JSON export, CI) |
| Card + size abstraction so trees fit memory | **ClubGG / our engine rules** bit-exact (ante, stacks, NLH menu) |
| Subgame roots with ranges | **Trainer / PolicyNet pipeline** later (labels from *our* π*) |

**Not goals:** reverse-engineer Monker binaries; AGPL/third-party link risk; “Monker but free” marketing.  
**Yes goals:** same *problem space* Monker occupies (multiway + preflop), engineered for **plodbnet** (scriptable factory → labels → play).

---

## What “full preflop→river” means (honest engineering)

River ranges must come from earlier streets. The deliverable is **not** one naive unabstracted NLH tree (impossible at real sizes). It **is**:

1. **One solver codebase** that can represent and solve **subgames from any street root**, including **preflop roots** and (later) **multiway**.
2. **Abstractions** where required (card bucketing, size abstraction, isomorphism) — the Monker-style tradeoff that makes multiway/preflop feasible.
3. **Range induction** under the solved strategy (forward along paths / blueprint), so river is never “orphan” ranges.
4. **Ops tooling first-class:** script API, batch root grid, JSON export of π*, deterministic seeds — **this is the product wedge**, on our engine rules.

```
Preflop root (abstract) ──► flop/turn/river subgames
         │                        │
         └──── same CFR core, same public state, same chip rules ────┘
         (+ multiway seats when abstraction + tree builder support it)
```

---

## Why this fits the repo

| Asset | Use |
|--------|-----|
| `rust_engine` NLH path | Already posts blinds, preflop, 2 hole cards, single board (`has_preflop`) |
| ClubGG roots | Locked stakes (`gto/roots.py`) — solver configs must match |
| `NLH_ANCHOR_SPEC` | Size abstraction for betting (or a coarser solve ladder that maps to anchors for labels) |
| PyO3 | Batch/script from Python; heavy CFR loops stay Rust |
| Mode 0 host later | Export `LabelRecord` / train PolicyNet from *our* π* |

CFR lives in `rust_engine/src/cfr/` on the same chip rules as the NLH engine.

---

## Architecture (phased, but one product)

### Core (Rust crate module, e.g. `rust_engine/src/cfr/`)

| Component | Responsibility |
|-----------|----------------|
| **Public state** | Pot, stacks, commits, street, board, to_act, legal actions — derived from / parity with `engine.rs` |
| **Range** | Reach probs over combos (1326 HU; abstract buckets when needed) |
| **Action abstraction** | Fold / check-call / raise-to discrete sizes (+ all-in); map to engine chips |
| **Tree builder** | Chance (deal) + action nodes; isomorphism optional |
| **CFR variant** | Start **DCFR or Linear CFR** on abstract tree; external sampling MCCFR for large preflop |
| **Solve API** | `solve(root_spec) -> Strategy` (π per infoset, optional CFVs) |
| **Export** | JSON/JSONL compatible with future `LabelRecord` path |

### Script / batch (Python)

| Piece | Role |
|--------|------|
| `scripts/cfr_solve.py` | One root (preflop or postflop board) |
| `scripts/cfr_batch.py` | Grid of SPR / boards / seeds |
| `plo5bp/gto/cfr_host.py` (later) | Process or in-process bindings |
| Config TOML/JSON | Sizes, iters, buckets, threads |

### Correctness gates (non-negotiable)

1. **Toy games**: Kuhn / Leduc (or tiny NLH) — known values, exploitability → 0  
2. **Engine parity**: every CFR action apply matches `BombPotEnv` / Rust engine chip rules on ClubGG blinds+ante  
3. **Range consistency**: after preflop solve, flop ranges = Bayesian update of blueprint; river ranges from full path  
4. **Determinism**: same seed + config → same π (within float tol)  
5. **Multiway is in the product vision** (Monker-class) but **ship HU correctness first** — no silent multiway Nash claims until multiway trees are implemented and tested

---

## Implementation phases (build order)

### Phase 0 — Scaffold + contracts ✅

- Types + CLI stub (done 2026-07-16)
- Doc: ClubGG units; **vision = Monker-class scope + batch/script ops**

### Phase 1 — HU postflop CFR on engine rules (foundation)

- Flop/turn/river **subgame** with **given** ranges
- Card abstraction toggle; size abstraction
- Exploitability + known-spot gates on HU roots
- **Batch + script** for many postflop roots (ops muscle early)

### Phase 2 — Preflop abstract tree (Monker-class spine)

- Preflop MCCFR/DCFR + hand/size abstraction  
- Blueprint → ranges into postflop  
- Scripted preflop solves

### Phase 3 — Full-hand pipeline + multiway path

- Glue: preflop → postflop resolve (range induction)  
- **Multiway tree builder** (start 3-way, then 4–6 with heavier abstraction)  
- Batch grids: seats × SPR × boards  

### Phase 4 — Labels + Trainer (after solver trustworthy)

- Export `LabelRecord` / PolicyNet optional  
- Trainer re-solve with path-tracked ranges  
- Badge only from `source=rust_cfr_*` + probe

---

## Explicit non-goals (until later)

- Multiway Nash (2–6 equilibrium)
- GPU CFR day-1
- Pausing PLO5 training
- Replacing Mode 0 UI path immediately
- Claiming “GTO AI” from unvalidated nets

---

## Success criteria for “we have a real P→R solver”

- [ ] HU preflop blueprint solves and dumps ranges  
- [ ] Postflop subgame solve from those ranges on ClubGG rules  
- [ ] Batch CLI runs N roots unattended  
- [ ] Exploitability / EV checks on toy + small real roots documented  
- [ ] Scriptable config (sizes, iters, threads, seed)  
- [x] Path from solve → strategy JSON via native rust_cfr  

---

## First concrete step after plan approval

**Phase 0 scaffold:** Rust `cfr` module + `RootSpec`/`SolveConfig` + Python CLI stub + architecture note in `docs/plans/`. No PLO pod touch. No Mode 0 200-root train.

Then Phase 1 HU postflop CFR core on engine parity tests.

---

## Phase 0 status (implemented 2026-07-16)

| Piece | Path |
|--------|------|
| Rust types + `solve` stub | `rust_engine/src/cfr/` |
| Python API | `python/plo5bp/gto/cfr_api.py` |
| CLI | `scripts/cfr_solve.py` |
| Tests | `tests/python/gto/test_cfr_api.py` + Rust `cfr` unit tests |

```powershell
.venv/Scripts/python scripts/cfr_solve.py --preflop --stack-bb 100
.venv/Scripts/python -m pytest tests/python/gto/test_cfr_api.py -q
cargo test --manifest-path rust_engine/Cargo.toml cfr
```

**Next:** teacher-quality rust_cfr batch → PolicyNet train → probe.

---

## Relationship to Mode 0 work already done

| Keep | Defer |
|------|--------|
| Label schema, PolicyNet, probe, badge honesty | Workstream C scale train until expl is gated |
| ClubGG roots, NLH anchors | Mode 2 ValueNet / street re-solve |
| Trainer StrategyBackend seam | GTO AI promote until rust_cfr labels + probe |

Teacher is native rust_cfr only.
