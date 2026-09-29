# Algorithm & Game-Tree Architecture (merge into `nlh-preflop-river-cfr-solver.md`)

**Status:** design for full preflop→river build  
**Scope:** algorithms, tree representation, blueprint↔subgame glue, multiway honesty, exploitability, phased path  
**Constraint:** one Rust CFR core; ClubGG / `rust_engine` chip rules; HU correctness before multiway Nash claims

---

## 1. CFR family — what to use when

| Algorithm | Memory model | Best use | Avoid for |
|-----------|--------------|----------|-----------|
| **Vanilla CFR** | Full tree, tabular regrets | Toy gates only (Kuhn / Leduc / 2-card river) | Any real NLH tree |
| **Linear CFR** | Full tree | Medium HU postflop that fits RAM; clean average strategy | Huge preflop / multiway |
| **DCFR** (Discounted CFR) | Full tree | **Default tabular HU postflop** (flop/turn/river subgames) | When tree does not fit memory |
| **External-sampling MCCFR** | Path sampling + infoset table | **Preflop blueprint**, large multiway, any tree too big for full traversal | Tiny games (higher variance than full CFR) |
| **Outcome-sampling MCCFR** | Path sampling | Research / secondary | Primary product path (noisier than external) |
| **CFR+** | Full tree | Optional A/B vs DCFR on same tree | Not required day-1 |

### Decisive defaults (do not bike-shed)

| Regime | Algorithm | Why |
|--------|-----------|-----|
| **Toy / unit** | Vanilla CFR (+ optional Linear) | Known values; exploitability → 0 proof |
| **HU river / turn / flop subgame** (given ranges, size-abstracted) | **DCFR**, full public-tree traversal | Fast early convergence; deterministic with fixed seed order; easy BR exploitability |
| **HU preflop blueprint** | **External-sampling MCCFR** + Linear/DCFR-style regret discount on sampled paths | Tree + chance branching too large for full iterate; industry-standard for abstract preflop |
| **Multiway (3–6)** | External-sampling MCCFR only | Full-tree multiway is memory-impossible under useful size menus |
| **Depth-limited resolve** (later, optional) | DCFR on short tree + leaf CFV (net or blueprint continuation) | Libratus/Pluribus pattern; **not** Phase 1 |

**Config tag** (already in `SolveConfig.algorithm`):

- `"vanilla"` | `"linear"` | `"dcfr"` | `"mccfr_es"`  
- Default: `"dcfr"` for postflop roots; force `"mccfr_es"` when `street == Preflop` or `num_seats > 2` (or when estimated infoset count exceeds a memory budget).

**Not day-1:** DeepCFR / neural regrets, GPU CFR. Tabular first; neural only if abstraction ceilings hurt product strength.

---

## 2. Tree structure

### 2.1 Two layers: public tree vs infosets

```
PublicTree  = chance + betting sequences + board (what everyone sees)
Infoset     = (player, public_history_id, private_view)
              private_view = hole combo  OR  card-bucket id under abstraction
```

- **Public node** stores: street, board (or chance-to-deal), pot/stacks/commits, to_act, legal abstract actions, children.
- **Infoset** stores: regret vector `R[a]`, cumulative strategy `σ̄[a]` (or DCFR accumulators), action list.
- Many hole cards / buckets share one public node; each has its own infoset under that history.

**Key invariant:** apply-action chip math is **identical** for all private hands at a public node (only legality of all-in/min-raise depends on stacks, which are public). Card-dependent branching only at **chance** and at **showdown evaluation**.

### 2.2 Node kinds

| Kind | Contents | Branching |
|------|----------|-----------|
| **Root** | `RootSpec` public params + ranges | → first actor or chance |
| **Chance (deal)** | Deal flop / turn / river (or preflop holes) | Iso-compressed or sampled |
| **Action** | One seat to act | Fold / check-call / raise sizes / all-in |
| **Terminal** | Fold win or showdown | Leaf EV in bb |

Preflop root: chance deals 2+2 holes (or abstract hand classes), then SB/BB action.  
Postflop root: board fixed; holes drawn from **range distributions** (no re-deal of board).

### 2.3 Action abstraction (shared with product)

Canonical abstract menu (maps to engine chips via pot-fraction):

```
FOLD | CHECK_CALL | RAISE_pm[i]... | ALLIN
```

- `raise_sizes_pm`: pot-fraction per-mille (Phase 0 default: `330, 500, 750, 1000, 1500`).
- Coarser solve ladder allowed for speed; **map to `NLH_ANCHOR_SPEC` only at label export**, not by forking chip math.
- Illegal sizes pruned by pot-limit / stack / min-raise rules from `engine` (parity tests).
- `allin_atom`: always available when remaining stack > largest discrete raise (and when jam is the only legal raise).

**Do not** use continuous bet sizes inside CFR. Discrete atoms only.

### 2.4 Card abstraction levels (toggle, not rewrite)

| Level | Private view | When |
|-------|--------------|------|
| **none** | Full 1326 combos (minus blocked) | HU river exact; small turn; correctness tests |
| **iso** | Suit isomorphism on public board + relative suits | Always-on compression when `use_isomorphism` |
| **bucket** | EHS / OCHS / custom clusters per street | Flop multiway, preflop hand classes, deep SPR |
| **preflop class** | 169 (or coarser 50–100 clusters) | Preflop blueprint only |

`SolveConfig.card_abstraction`: `"none" | "iso" | "ochs" | "preflop169" | ...`

### 2.5 Range representation

```
Range[seat] : ComboId → reach_weight ≥ 0
ComboId     : 0..1325 (C(52,2)) or BucketId under abstraction
```

- Normalize after blocking board cards.
- Empty range string in `RootSpec` = uniform over unblocked combos (Phase 0 behavior).
- Multiway: one range vector per seat; product measure for chance of private deals in MCCFR.

### 2.6 Storage layout (Rust)

```
PublicTree { nodes: Vec<PublicNode>, root: NodeId }
PublicNode {
  kind, street, board_key, pot_chips, stacks[], to_act,
  actions: Vec<AbstractAction>,
  children: Vec<NodeId>,  // or chance outcomes
}
InfosetTable {
  // key: hash(player, public_history, private_view)
  regrets: ..., strategy_sum: ...,  // DCFR discounts applied in-place
}
```

- Build tree **once** per solve (or rebuild when SPR/size menu changes).
- Parallelism: lock-free per-infoset updates for MCCFR (Pluribus-style); DCFR full-tree can shard by infoset after traverse collects regrets.

---

## 3. Preflop blueprint ↔ postflop subgames (range induction)

### 3.1 Product model (not one giant tree)

```
                    ┌─────────────────────────────┐
                    │  Preflop blueprint solve     │
                    │  (abstract hands + sizes)    │
                    └──────────────┬──────────────┘
                                   │ π_blueprint
                                   ▼
              Path / line filter (open, 3bet, call, …)
                                   │
                                   ▼
                    Bayesian range update per seat
                                   │
                                   ▼
              ┌────────────────────────────────────┐
              │ Postflop subgame root              │
              │ board + pot + stacks + ranges      │
              │ DCFR (finer card abs / none river) │
              └────────────────────────────────────┘
```

Full-hand “GTO” in product language = **blueprint strategy on preflop nodes** + **resolved postflop strategies** under induced ranges — same as Monker-class / Pio workflow.

### 3.2 Range induction (exact procedure)

Given blueprint average strategy `σ̄` and a public action sequence `h` from preflop start to a postflop root:

For each seat `i`, each combo `c`:

```
reach_i(c | h) ∝ prior_i(c) * ∏_{decision points of i on path h} σ̄_i(a_t | infoset_i(c, h_t))
```

Then:

1. Zero combos that conflict with the **known board** (and optionally known villain cards in study mode).
2. Renormalize each seat’s range.
3. Attach to `RootSpec.range_ip` / `range_oop` (and multiway range list later).
4. Solve postflop subgame **from those ranges** with DCFR.

**Critical tests:**

- Mass conservation: sum of induced range weights equals path probability mass (within float tol).
- Blockers: removing a board card zeros intersecting combos only.
- Round-trip: pure-strategy path → ranges match manual Bayes on a 2-combo toy.

### 3.3 When to re-solve vs trust blueprint continuation

| Street / SPR | Policy |
|--------------|--------|
| Preflop decisions | Use blueprint π |
| Flop / turn / river at known board | **Always prefer subgame re-solve** with induced ranges (Phase 2+) |
| Memory blow-up | Coarser card abs on flop; exact on river |
| Later strength layer | Depth-limited CFR + leaf values (optional Phase 4+) |

Do **not** ship “orphan river ranges” (user-supplied ranges with no path). Product API may allow them for study, but label the export `source=orphan_range`, not `full_hand_path`.

### 3.4 Blueprint abstraction → postflop refinement

- Preflop private view: 169 (or coarser).
- At flop root: **expand** abstract class → combo distribution (uniform within class, or stored class→combo map), then apply board blockers, then solve with finer abs.
- Document expansion rule in config so solves are reproducible.

---

## 4. Multiway (3–6) — honest design

### 4.1 Theory honesty (product language)

- **HU:** unique Nash value; exploitability well-defined (mbb/hand or % pot).
- **Multiway:** Nash equilibria are **not unique**; different equilibria can differ a lot on off-path and some on-path mixes.
- Ship language: **“equilibrium under shared strategy class / population play”**, never “the GTO” for 3–6.

### 4.2 Practical algorithm

- Same public-tree + infoset machinery; `num_seats ∈ 3..6`.
- **External-sampling MCCFR** only (sample one trajectory per iteration; update regrets for the traverser).
- All seats share the **same algorithm instance** (symmetric Pluribus-style population): one strategy table per infoset key that includes seat position / relative stack role as needed.
- Heavier **action** abstraction (e.g. 2–3 bet sizes + all-in) and **card** buckets from day-1 of multiway.

### 4.3 Tree size pressure (order-of-magnitude)

| Setting | Dominant cost | Mitigation |
|---------|---------------|------------|
| HU river exact, 5 sizes | Combos × histories | Fits; DCFR |
| HU flop, no card abs, 5 sizes | Huge but known | Iso + optional buckets |
| 3-way river, 5 sizes | Histories × 3 ranges | Coarse sizes; MCCFR |
| 6-max preflop 100bb | Impossible unabstracted | 169 hands + tiny size menu + MCCFR blueprint only |

**Rule:** if estimated public nodes × avg infosets/node > memory budget, refuse solve with clear error and suggest coarser abs — do not silent OOM.

### 4.4 Multiway range induction

Same Bayes product along path; each seat’s reach multiplies only on **that seat’s** action nodes. Chance deals remaining holes consistently with blockers across all seats (rejection or sequential deal without replacement).

### 4.5 Ship order for multiway

1. 3-way **river** with given ranges (validate infrastructure).  
2. 3-way turn/flop with buckets.  
3. 3–6 preflop blueprint (very coarse).  
4. Full-hand multiway glue only after HU P→R path is green.

---

## 5. Exploitability & convergence criteria

### 5.1 HU (primary metric)

**Exploitability** (two-player zero-sum):

```
expl(σ) = (BR₁(σ₂) + BR₂(σ₁)) / 2     # in bb/hand or mbb
```

Implementation:

1. Fix opponent average strategy `σ̄`.
2. Compute best response value for each player via tree DP (max over actions; expectation over chance and opponent σ).
3. Report `exploitability_bb` on `SolveReport` (field already exists).

**Stop when** any of:

- `expl ≤ target_exploitability_bb` (config; default 0.5 bb is loose — tighten per street), or
- `max_iterations` reached, or
- optional: relative pot metric `expl / pot_bb ≤ ε`.

**Street-typical targets** (guidance, not hard locks):

| Subgame | Practical target |
|---------|------------------|
| Toy | `< 1e-3` bb |
| HU river small | `≤ 0.05–0.1` bb |
| HU flop abstract | `≤ 0.3–0.5` bb |
| Preflop MCCFR blueprint | track **avg. strategy stability** + sampled BR proxy; full expl expensive |

### 5.2 Preflop / MCCFR convergence (proxy metrics)

Full BR on abstract preflop is possible but slow. Log:

- Local BR / sampled exploitability every N iters.
- Strategy L1 change of average π.
- Head-to-head EV of current average vs frozen snapshot (self-play).

Do not claim Nash without either full BR expl (abstract game) or documented proxy.

### 5.3 Multiway convergence

- **No unique expl.** Report:
  - Local best-response gain for one seat vs fixed others (one-seat exploitability).
  - Average strategy stability.
  - Optional: n-agent local Nash gap under shared class.
- Product export: `equilibrium_kind = "population_mccfr"` vs `"hu_nash_approx"`.

### 5.4 Correctness gates (must stay green)

1. Kuhn / Leduc: value + expl → known.  
2. Engine parity: every abstract action → chip delta matches `BombPotEnv` / Rust NLH apply.  
3. Range induction unit tests.  
4. Determinism: same seed + config → same π (MCCFR: same RNG stream).  
5. Known-spot + Kuhn/Leduc + engine-parity gates (native only).

---

## 6. Phased algorithm path (no dead ends)

Each phase **extends** the same core; nothing is thrown away.

### Phase 0 — Scaffold ✅
Types, validate, stub `solve`. Done.

### Phase 1 — HU postflop foundation (**implement first**)
**Goal:** real DCFR on engine-legal trees.

| Build | Detail |
|-------|--------|
| Public tree builder | River first → turn → flop |
| Action abstraction | Fold / X-C / raise_pm / all-in → engine chips |
| Ranges | 1326 weights; block board |
| Algorithm | **DCFR** full traverse |
| Card abs | `none` + suit iso |
| Metrics | True HU exploitability |
| Ops | CLI + JSON strategy dump + batch N roots |

**Exit criteria:** river subgame expl→target; chip parity tests; optional TS delta on shared HU roots.

*Why not dead-end:* every later phase reuses public nodes, infosets, action map, ranges, export.

### Phase 1b — Toy game harness (parallel, cheap)
Kuhn + Leduc in same CFR update code (or thin game trait). Proves regret matching before NLH complexity.

### Phase 2 — Preflop blueprint spine
| Build | Detail |
|-------|--------|
| Preflop tree | Blinds/antes ClubGG; abstract sizes |
| Card abs | 169 (expandable to clusters) |
| Algorithm | **External-sampling MCCFR** |
| Output | Blueprint π + dump of common line ranges |
| Glue | `induce_ranges(blueprint, path) → RootSpec` |

**Exit criteria:** open/3bet/call ranges look sane; induced flop ranges feed Phase 1 solver unchanged.

### Phase 3 — Full-hand pipeline + multiway path
| Build | Detail |
|-------|--------|
| Scripted P→R | Blueprint path → induce → postflop DCFR |
| Batch grids | seats × SPR × boards × lines |
| Multiway | Start 3-way river MCCFR; then turn/flop with buckets |
| Honesty flags | `hu_nash_approx` vs `population_mccfr` |

### Phase 4 — Labels / Trainer (after solver trustworthy)
Export `LabelRecord` with `source=rust_cfr_*`; PolicyNet distill; badge only from native solves + probe.

---

## 7. Module map (Rust `cfr/`, target layout)

```
cfr/
  mod.rs          // solve() dispatch by algorithm + street
  types.rs        // RootSpec, SolveConfig, Strategy (exists)
  public_state.rs // pot/stacks/legal actions ↔ engine parity
  actions.rs      // abstract menu → chip amounts
  tree.rs         // PublicTree build (chance + action + terminal)
  range.rs        // 1326 / buckets, block, normalize, induce
  infoset.rs      // keying, regret tables, average strategy
  dcfr.rs         // full-tree DCFR / Linear / Vanilla
  mccfr.rs        // external sampling
  br.rs           // best response + exploitability (HU)
  showdown.rs     // NLH eval via hand_eval::evaluate_nlh
  export.rs       // Strategy → JSON
```

Python stays thin: `cfr_api.solve` → PyO3; batch/CLI unchanged contract.

---

## 8. Non-negotiable design rules

1. **One chip truth:** abstract actions always lower to `engine` apply; no parallel pot math.  
2. **Subgame roots are first-class** — full-hand = composition, not one unabstracted mega-tree.  
3. **HU metrics before multiway claims.**  
4. **DCFR tabular postflop + MCCFR-ES preflop/multiway** — fixed defaults until measured otherwise.  
5. **Range induction is part of the solver product**, not a Python afterthought.  
6. **Refuse** oversize trees with actionable abs suggestions.  
7. PLO5 pod untouched; all work local / separate process.

---

## 9. Immediate next implementation slice (after this design merge)

Phase 1 minimal vertical slice:

1. `public_state` + `actions` parity tests vs NLH engine (HU river, fixed board).  
2. Build river public tree with size menu.  
3. Tabular DCFR + average strategy.  
4. BR exploitability on report.  
5. Bind `cfr_solve` in PyO3; flip `rust_cfr_available()`.  
6. Keep Kuhn/Leduc as algorithm unit tests.

No preflop, no multiway, no neural — until river expl and parity are green.
