# NLH CFR — Card & Bet-Size Abstraction

**Status:** design (merge into master preflop→river plan)  
**Scope:** solve-time size ladders, serving/label map to `NLH_ANCHOR_SPEC`, card abs levels, suit iso policy, memory ballparks, `SolveConfig` surface  
**Constraint:** ClubGG 5/10($5) (`bb=10_000`); HU first; discrete atoms only; **never** fork chip math — same pot-after-call convention as `sizing.py` / engine

---

## 0. Hard rules (do not re-litigate)

1. **Solve ladder ≠ serve ladder.** CFR trees use a **coarse** discrete size menu. Labels / PolicyNet / UI use **full `NLH_ANCHOR_SPEC`** (12 atoms: min + 10 fracs + all-in). Mapping is **export-only**.
2. **Chip truth is one place:** `chips = clamp(to_call + round_half_up(pm * (pot+to_call) / 1000), min_raise, max_raise)`; all-in = `max_raise`. Same as `NLH_ANCHOR_SPEC` / `apply_raise_chips`.
3. **No continuous bets inside CFR.** Atoms only; legality dedupe (strictly greater chips than previous legal atom).
4. **Refuse, don't OOM:** if estimated infoset-bytes > budget, fail with suggested coarser abs.
5. **Multiway ≠ unique Nash** — coarser abs + population language from day one of multiway.

---

## 1. Size abstraction ladder

### 1.1 Full serve ladder (labels / UI / PolicyNet)

Canonical product menu — **do not shrink this for training/serving**:

| Index | Label | `fracs_pm` | Role |
|------:|-------|------------|------|
| 0 | min | 0 | min-raise atom |
| 1–10 | 25…275% | 250,330,500,660,800,1000,1250,1600,2000,2750 | near-geo overbets |
| 11 | ALLIN | — | always `max_raise` |

Source of truth: `python/plo5bp/sizing.py` → `NLH_ANCHOR_SPEC` (count **12**).

### 1.2 Solve ladders (named presets)

Named presets live as constants on the Rust CFR side + CLI strings. `RootSpec.raise_sizes_pm` holds the **interior raise fracs only** (fold / check-call / all-in are structural).

| Preset | `raise_sizes_pm` | Typical actions/node | Use |
|--------|------------------|----------------------|-----|
| **`micro`** | `500, 1000` | F / XC / 50% / pot / AI ≈ **5** | 6-max preflop blueprint; 3–6 multiway; memory crisis |
| **`coarse`** (Phase 1 default) | `330, 500, 1000, 1500` | ≈ **6** | HU flop/turn; 3-way river; batch grids |
| **`standard`** | `330, 500, 750, 1000, 1500` | ≈ **7** | Phase 0 default (`DEFAULT_RAISE_SIZES_PM`); HU river/turn when RAM OK |
| **`fine`** | `250, 330, 500, 660, 800, 1000, 1500, 2000` | ≈ **10** | High-value HU river re-solves; not multiway |
| **`full_anchor`** | all 10 NLH fracs (no min — min is XC-side raise floor) | ≈ **12** | Research / tiny SPR only; **not** production trees |

**Decisive defaults by regime:**

| Regime | Size preset | Why |
|--------|-------------|-----|
| HU river exact cards | `standard` (or `fine` if SPR≤20 and expl budget tight) | Branching dominated by combos, not sizes |
| HU turn | `standard` | Balance |
| HU flop abstract | `coarse` | Public histories explode with sizes |
| HU preflop blueprint | `coarse` open/3bet; `micro` for 4bet+ depth | MCCFR variance + depth |
| 3-way river | `coarse` → `micro` if OOM | Multiway histories × seats |
| 6-max preflop | **`micro` only** | Only feasible menu |

**Street-adaptive menu (recommended, config-driven):** allow different presets per street in one full-hand solve:

```
size_ladder_preflop = micro|coarse
size_ladder_flop    = coarse
size_ladder_turn    = standard
size_ladder_river   = standard|fine
```

Phase 1 can ship a single global ladder; Phase 2+ should support per-street.

### 1.3 Legal menu construction (every public action node)

```
menu = []
if to_call > 0:  menu += [FOLD]
menu += [CHECK_CALL]          # check if to_call==0 else call
for pm in raise_sizes_pm:
    chips = pot_frac_chips(pm, pot, to_call, min_r, max_r)
    if legal & chips > last_menu_raise_chips: append RAISE_pm
if allin_atom and max_r > last_menu_raise_chips: append ALLIN
# short-shove lockout / engine min_raise=0: only ALLIN (or XC if already all-in call)
```

### 1.4 Solve → serve label map (export only)

When writing `LabelRecord` / UI recommendations on the **full** 12-anchor head:

| Solve action | Map to `NLH_ANCHOR_SPEC` |
|--------------|--------------------------|
| FOLD | fold (gate) |
| CHECK_CALL | check/call (gate) |
| RAISE_pm | nearest anchor by `|log(pm_s) − log(pm_a)|` among **legal** anchors; ties → lower pm |
| ALLIN | all-in atom (index 11) |
| min-raise-only situations | anchor 0 (min) |

**Probability mass:** put solve π mass on the **mapped** anchor; if two solve sizes collapse to one legal chip amount, **sum** mass. Never invent mass on unmapped anchors (leave 0) — PolicyNet learns the support of the teacher.

**Do not** re-solve on full 12-size menu just to label; map is enough for distill. Optional later: river `fine` re-solve for study quality.

### 1.5 What Phase 0 already has vs change

- `DEFAULT_RAISE_SIZES_PM = [330,500,750,1000,1500]` → keep as **`standard`**.
- Add named presets + `size_ladder: String` (or enum) on config; CLI `--size-ladder coarse`.
- `RootSpec.raise_sizes_pm` remains the ground-truth menu for that tree (presets expand into it).

---

## 2. Card abstraction — options & phased recommendation

### 2.1 Levels (private view)

| Level | Private view | Domain size | When |
|-------|--------------|-------------|------|
| **`none`** | Full combo `0..1325` (board-blocked) | ≤1326 | HU river; unit tests; tiny turn |
| **`iso`** | Suit-isomorphic combo id (canonical suit relabel) | ~2–5× fewer than raw on many boards | **Always on for build** when `use_isomorphism=true` (orthogonal to buckets) |
| **`preflop169`** | 169 hand classes (AA…72o) | 169 | Preflop blueprint only |
| **`ehs`** | Equity vs uniform / vs range → 1-D quantiles | B buckets/street | Cheap baseline; weak multiway / blocker play |
| **`ochs`** | Opponent-Cluster Hand Strength (equity vs K opponent archetypes → K-D → cluster) | B buckets/street | **Default postflop abstract** (Monker-class) |
| **`custom`** | Precomputed map path | file-defined | Production bucket files versioned on disk |

**Infoset key:**

```
infoset = hash(player_role, public_history_id, private_view)
private_view ∈ ComboId | IsoComboId | Class169 | BucketId(street)
```

### 2.2 Bucket counts (decisive defaults)

| Street | HU default B | Multiway default B | Notes |
|--------|-------------:|-------------------:|-------|
| Preflop | 169 classes | 169 (or 50–80 clusters later) | No EHS preflop |
| Flop | **200** | **100** | OCHS; raise if RAM free |
| Turn | **500** | **200** | More resolution before river |
| River | **none** (exact) HU; **200** multiway | Exact preferred when seats=2 | River is where value lines pay for precision |

Store **separate** bucket maps per street (standard). Flop buckets ≠ turn buckets.

### 2.3 OCHS pipeline (implementation sketch)

1. Offline (or first-run cache): sample opponent range archetypes (K≈8–16 clusters of hands by equity/hand-class).  
2. For each hero combo on a board: vector `e[k] = P(hero wins | opp ~ archetype_k)`.  
3. k-means / hierarchical cluster into B buckets; assign each combo→bucket.  
4. Version file: `buckets/{variant}/{street}_b{B}_v{N}.bin` + config hash.  
5. Solve path only looks up `combo → bucket`; never recomputes EHS mid-CFR.

**Phase 1:** `card_abstraction=none` (+ iso).  
**Phase 1.5:** load precomputed OCHS maps; no online clustering in the hot loop.

### 2.4 Phased recommendation (build order)

| Phase | Card abs | Rationale |
|------:|----------|-----------|
| **1** HU river | `none` + iso | Correctness + real expl; tree fits |
| **1** HU turn | `none` + iso first; fall back `ochs@500` if estimate > budget | Prefer exact |
| **1** HU flop | `ochs@200` + iso **default**; `none` only for toy SPR / smoke | Unabstracted flop is the classic memory bomb |
| **2** Preflop blueprint | `preflop169` + iso on deal | MCCFR spine |
| **2** Blueprint→postflop | Expand 169 → uniform combos in class, block board, then postflop abs | Document expansion in config |
| **3** 3-way river | `ochs@200` or `none` if ranges sparse | Validate multiway infra |
| **3** 3-way flop/turn | `ochs@100/200` | Required |
| **3** 6-max preflop | `preflop169` + `micro` sizes only | Blueprint only |

**Product quality ladder (same codebase):**

```
Study "exact river"  → card=none, sizes=fine/standard
Batch factory flop   → card=ochs@200, sizes=coarse
Preflop blueprint    → card=preflop169, sizes=micro/coarse
Multiway             → card=ochs low-B, sizes=micro
```

---

## 3. Suit isomorphism — when to enable

### 3.1 What it does

Canonical relabeling of suits so that strategically identical boards/holes share nodes:

- **Public iso:** map board suits to a canonical suit pattern (e.g. first suit → A, second → B…).  
- **Private iso:** express hole suits **relative** to the public canonical map.  
- Chance nodes deal **canonical boards** only; multiply by suit-orbit weight.

### 3.2 Policy (decisive)

| Setting | `use_isomorphism` | Notes |
|---------|-------------------|--------|
| Default (all production solves) | **`true`** | Free 2–10× public-tree compression on many textures; Phase 0 default already `true` |
| Unit tests of raw combo indexing | `false` | Compare against non-iso oracle |
| Bucket builds | Build maps **in iso space** (or store both) | Avoid double-counting orbits |
| Multiway | `true` | Even more valuable |
| Preflop 169 | N/A for ranks; still iso on **future** board deals when expanding to postflop | 169 already suit-abstracted for holes |

**Enable always for tree build unless debugging.** Do not treat iso as a “phase 2 nice-to-have” — it is Phase 1 tree-builder core (`cfr/iso.rs`).

**Correctness tests:** same strategy probs for suit-isomorphic boards after remapping; orbit weights sum to true deal probability.

---

## 4. Memory / compute ballparks

Assumptions (order-of-magnitude, ClubGG-ish 100bb lines, f32 regrets + f32 strategy_sum ≈ **8 bytes × |A| per infoset**, |A|≈6, plus public tree ~100–200 B/node). **Not** peak RSS of a full product binary — **solver table scale**.

### 4.1 Formula sketch

```
infosets ≈ num_public_action_nodes × avg_private_views_per_seat × num_seats_acting
bytes   ≈ infosets × |A| × 8 × 2   (+ strategy dump)
```

Public action nodes scale roughly as `O( (|A_bet|)^depth × chance_branches )` with depth = streets remaining × bets per street (cap ~3–4 bets/street in practice with NL).

### 4.2 Regime table

| Regime | Cards | Sizes | Public nodes (ord.) | Private views | Infosets (ord.) | RAM (ord.) | Compute |
|--------|-------|-------|---------------------|---------------|-----------------|------------|---------|
| **HU river exact** | none + iso | standard (~6) | 10²–10³ | ~1e3 combos | **10⁶–10⁷** | **0.1–2 GB** | DCFR, seconds–minutes / root on desktop CPU |
| **HU turn exact** | none + iso | standard | 10³–10⁴ (+ river chance) | ~1e3 | **10⁷–10⁸** | **2–20 GB** | DCFR minutes; may need iso+budget gate |
| **HU flop abstract** | ochs@200 + iso | coarse (~5–6) | 10⁴–10⁵ | 200 buckets | **10⁷–10⁸** | **2–15 GB** | DCFR minutes–tens of min; batch on workstation |
| **HU flop exact** | none | standard | 10⁵–10⁶ | ~1e3 | **10⁹–10¹⁰** | **100+ GB** | **Refuse** in v1 unless research machine |
| **3-way river abstract** | ochs@200 | coarse/micro | 10³–10⁴ | 200 × 3 seats | **10⁷–10⁸** | **2–20 GB** | **MCCFR** 10⁷–10⁸ iters; hours |
| **3-way river exact** | none | coarse | 10³–10⁴ | 1e3³ joint deal* | explodes | **often impossible** | Sample joint holes in MCCFR; no full tabular |
| **6-max preflop blueprint** | 169 | micro (~4–5) | 10⁴–10⁶ abstract | 169 × positions | **10⁶–10⁸** | **0.5–10 GB** abstract table | **MCCFR** 10⁸–10⁹ iters; hours–day on multi-core |

\*Multiway exact joint ranges are not stored as a full product table; MCCFR samples. Tabular regrets stay per-seat infoset (private view × public history), still large with |A| and history count.

### 4.3 Practical gates (encode in solver)

| Estimate | Action |
|----------|--------|
| `< 4 GB` | Solve freely |
| `4–24 GB` | Warn; require explicit `--allow-large-tree` |
| `> 24 GB` | **Hard refuse** + print suggested preset (`coarse`/`micro`, raise B, enable iso) |

ClubGG 5/10($5) does not change combinatorial scale (bb only rescales EV units).

### 4.4 Iteration budgets (guidance)

| Regime | Algorithm | Iters (ord.) | Stop |
|--------|-----------|--------------|------|
| HU river exact | DCFR | 200–2k | expl ≤ 0.05–0.1 bb |
| HU flop ochs | DCFR | 500–5k | expl ≤ 0.3–0.5 bb |
| Preflop 169 | MCCFR-ES | 1e7–1e9 | strategy L1 stability + sampled BR |
| 3-way river | MCCFR-ES | 1e7–1e8 | one-seat local BR gap (not unique Nash) |

---

## 5. Config surface (`SolveConfig` + friends)

### 5.1 Keep / extend Phase 0 `SolveConfig`

Already present:

| Field | Keep | Notes |
|-------|------|-------|
| `max_iterations` | yes | |
| `target_exploitability_bb` | yes | Street-specific defaults in CLI profiles |
| `thread_num` | yes | |
| `seed` | yes | Determinism |
| `use_isomorphism` | yes | **Default true** |
| `algorithm` | yes | `dcfr` \| `mccfr_es` \| `linear` \| `vanilla` |
| `card_abstraction` | **extend values** | see below |

### 5.2 Add fields (Phase 1–2)

```rust
// SolveConfig extensions
pub size_ladder: String,           // "micro"|"coarse"|"standard"|"fine"|"custom"
// if custom: RootSpec.raise_sizes_pm is authoritative
pub size_ladder_by_street: Option<SizeLadderByStreet>, // Phase 2+
pub bucket_count_flop: u32,        // default 200
pub bucket_count_turn: u32,        // default 500
pub bucket_count_river: u32,       // default 0 = exact when card_abstraction allows
pub bucket_map_path: Option<String>, // precomputed OCHS maps
pub memory_budget_gb: f64,         // default 24.0; hard refuse above
pub allow_large_tree: bool,        // default false
pub export_anchor_map: bool,       // default true → map π to NLH_ANCHOR_SPEC on export
pub preflop_classes: u32,          // default 169
```

**`card_abstraction` allowed strings:**

```
"none" | "preflop169" | "ehs" | "ochs" | "custom"
```

Iso is **not** a card_abstraction value — it is the separate `use_isomorphism` flag (compositional).

### 5.3 `RootSpec` (action side)

| Field | Role |
|-------|------|
| `raise_sizes_pm` | Concrete solve menu (expanded from preset) |
| `allin_atom` | Default **true** |
| `num_seats` | 2 now; 3–6 later |
| ranges / board / pot / stacks | unchanged |

CLI examples:

```text
scripts/cfr_solve.py --street river --size-ladder standard --card-abstraction none
scripts/cfr_solve.py --street flop  --size-ladder coarse   --card-abstraction ochs --buckets-flop 200
scripts/cfr_solve.py --preflop --size-ladder micro --card-abstraction preflop169 --algorithm mccfr_es
```

### 5.4 Profiles (sugar, not new engines)

| Profile name | Expands to |
|--------------|------------|
| `hu_river_exact` | sizes=standard, card=none, iso=true, algo=dcfr, target_expl=0.1 |
| `hu_flop_batch` | sizes=coarse, card=ochs@200, iso=true, algo=dcfr, target_expl=0.5 |
| `hu_preflop_blueprint` | sizes=coarse, card=preflop169, iso=true, algo=mccfr_es |
| `mw3_river` | sizes=micro, card=ochs@200, iso=true, algo=mccfr_es, seats=3 |
| `mtt6_preflop` | sizes=micro, card=preflop169, algo=mccfr_es, seats=6 |

---

## 6. Implementation touchpoints (no code now — plan only)

| Piece | Path |
|-------|------|
| Size presets + legal menu | `rust_engine/src/cfr/actions.rs` |
| Iso | `rust_engine/src/cfr/iso.rs` |
| Card abs lookup | `rust_engine/src/cfr/card_abs.rs` |
| Memory estimate before build | `rust_engine/src/cfr/tree.rs` |
| Export → NLH anchors | `rust_engine/src/cfr/export.rs` + Python label path |
| Types | extend `SolveConfig` / validate in `types.rs` |
| Tests | menu chip parity vs engine; iso orbit; 169 expand; map-to-anchor mass conservation |

---

## 7. One-page decision summary

| Question | Decision |
|----------|----------|
| Solve sizes vs serve sizes | **Coarse solve presets**; **full `NLH_ANCHOR_SPEC` at export** |
| Phase 1 default sizes | **`coarse` for flop**, **`standard` for turn/river** |
| River cards HU | **Exact (`none`)** |
| Flop cards HU | **OCHS ~200 + iso** |
| Preflop | **169 + MCCFR** |
| Iso | **On by default always** |
| Multiway sizes | **`micro` / `coarse` only** |
| Full 12-size CFR tree | **No** (except research `fine` HU river) |
| OOM policy | **Estimate → refuse** with preset suggestion |

This section is ready to merge under the master plan’s “Action abstraction / Card abstraction” headings and supersedes the vague “5 sizes / optional buckets” bullets with named presets and memory gates.
