# NLH CFR — Engine Integration Plan (Phase 1–3)

**Status:** design / audit (read-only)  
**Depends on:** Phase 0 scaffold (`rust_engine/src/cfr/{mod,types}.rs`, `python/plo5bp/gto/cfr_api.py`)  
**Related plans:** `nlh-preflop-river-cfr-solver.md`, `nlh-cfr-algorithm-game-tree-architecture.md`  
**Constraint:** ClubGG chip rules bit-exact via `GameState`; HU first; PLO5 pod untouched

---

## 1. What `GameState` already provides that CFR needs

| CFR need | Engine API / field | Notes |
|----------|-------------------|--------|
| Legal voluntary acts | `legal_action_mask()` | 8-slot PPO menu (Fold / CheckCall / 5 pot% / AllIn). **Not** the CFR size ladder — use as *reference* for fold/x-c legality + short-shove rules |
| Discrete apply | `apply(Action)` | Full street/actor/run-out machine |
| Continuous raise (CFR sizes) | `apply_raise_chips(chips)` + `min_raise_chips()` / `max_raise_chips()` | **Primary apply path for abstract RAISE_pm** |
| Chip query | `action_to_chips`, pot-fraction via `compute_sizing_chips` | Same pot-after-call convention as PPO |
| Public chips | `pot`, `stacks`, `street_commit`, `total_commit`, `bet_to_call` | All `u64` engine chips |
| Raise floor / reopen | `last_raise_size`, `street_level_acted`, `last_aggression_was_full_raise`, `short_shove_lockout` | TDA-style short all-in — CFR **must** preserve |
| Street / actor / terminal | `street`, `actor`, `is_terminal()`, `current_actor()` | Preflop→Showdown |
| Blind structure (NLH) | `GameConfig::{sb,bb,ante}`, `Variant::NlhSingle`, `nlh_blind_seats`, live blinds into `street_commit` | Antes dead; blinds live; `bet_to_call = bb` preflop |
| Deal / study roots | `new_hand`, `new_study_nlh`, `set_flop/turn/river_nlh` | Full hands + UI study — **not** mid-street solver roots |
| Showdown EV | `payouts()` → `single_board_payout` + `evaluate_nlh` | Side pots already correct for multi-commit |
| MC runout EV | `payouts_ev(num_samples, seed)` | Useful for all-in before river; exact CFR prefers full board at terminal |
| Clone | `#[derive(Clone)]` on `GameState` | Safe for parity probes; too heavy as tree node payload |
| Config helpers | `GameConfig::new_nlh_uniform` | ClubGG: bb=10_000, sb=5_000, ante=5_000 (Python `CLUBGG_NLH_ROOT`) |

**NLH blinds/preflop (already implemented):**

1. Dead ante per in-hand seat → pot / `total_commit`.  
2. SB/BB live into `street_commit`; short posts all-in; nominal `bet_to_call = bb`.  
3. Street = Preflop, boards empty until round closes.  
4. First actor = first non-folded/non-all-in clockwise after BB (HU: button/SB).  
5. Round close → reveal flop (production auto; study waits for `set_flop_nlh`).

**PPO vs CFR action spaces (do not conflate):**

| Layer | Actions | Apply |
|-------|---------|--------|
| PPO / training | Fixed 8 `Action` enum | `apply` / env `step` |
| CFR abstract menu | `FOLD \| CHECK_CALL \| RAISE_pm[i]… \| ALLIN` from `RootSpec.raise_sizes_pm` | Map to chips → `apply_raise_chips` (or Fold/CheckCall via `apply`) |
| Label export later | Map solve ladder → `NLH_ANCHOR_SPEC` | Export only; **never** fork chip math |

---

## 2. Gaps (what CFR must add / engine must expose)

### 2.1 Mid-street public root constructor — **critical**

There is **no** API to open a hand already on flop/turn/river with:

- fixed board (3/4/5 cards),
- given pot + remaining stacks,
- `street_commit = 0`, `bet_to_call = 0`, `last_raise_size = bb`,
- first postflop actor = left of button,
- holes **not** dealt (or placeholders), no re-post of blinds/antes.

`new_hand` always deals + posts; study paths still start preflop (NLH) or flop (PLO) with ante logic.

**Required:** either

- `GameState::from_solver_root(...)` (production-mode, `study_mode=false`, holes set later for showdown), **or**
- CFR-owned `PublicState` that owns betting transitions and only materializes `GameState` for parity tests.

Recommendation: **both** — pure `PublicState` for the tree; thin engine bridge for tests and optional “oracle apply”.

### 2.2 Range representation

None in engine (holes are concrete `Vec<Card>`). CFR needs:

- HU: `Range = [f32; 1326]` (or sparse) per seat, board-blocked + renormalized  
- Parse hooks for `RootSpec.range_ip` / `range_oop` (empty = uniform unblocked)  
- Multiway later: one vector per seat + joint blocker handling

### 2.3 Infoset IDs

None. Need stable keys:

```
infoset_key = hash(player_role, public_history_id, private_view)
private_view = ComboId (0..1325) | BucketId | Preflop169
```

Public history = sequence of abstract actions + chance outcomes (board cards), not raw `ActionRecord` labels.

### 2.4 Abstract actions → engine chips

`RootSpec.raise_sizes_pm` (default 330/500/750/1000/1500) must lower to chip **deltas** with the **same** convention as `compute_sizing_chips` / `sizing.anchor_grid_np`:

- Opening: `raise_over = pot * pm / 1000`  
- Facing bet: `raise_over = (pot + to_call) * pm / 1000`, target total = `bet_to_call + raise_over`  
- Clamp to `[min_raise_chips, max_raise_chips]`; drop illegal / duplicate chips; keep ALLIN when `allin_atom` and stack > call

**Do not** call `apply(BetPct50)` for “50% pot” when CFR menu uses 500‰ — always compute chips then `apply_raise_chips`.

### 2.5 Chance + terminal without full engine clone

- Chance: deal remaining board from unseen deck (iso-compressed or sampled).  
- Terminal fold: pot to survivor (card-agnostic).  
- Terminal showdown: `evaluate_nlh` + side-pot layers (`single_board_payout` or pure reimplementation of same layers).  
- Preflop blueprint: chance deals hole classes / combos under ranges.

### 2.6 RootSpec → chips semantics (clarify in Phase 1)

| Field | Meaning at root | Engine mapping |
|-------|-----------------|----------------|
| `bb_chips` | 1 bb unit | `config.bb` |
| `pot_bb` | pot already in middle | `pot = round(pot_bb * bb_chips)` |
| `effective_stack_bb` | chips each player can still put in | `stacks[i] = round(eff * bb)` (HU equal for v1) |
| Preflop | pot should equal blinds+antes | `sb+bb+2*ante` / bb; stacks = starting − ante − own blind (or define “starting stack” and post via engine) |

**Ambiguity to lock in Phase 1 docs/tests:** preflop `effective_stack_bb` = starting stack **before** blinds, or remaining after posting? ClubGG product language is usually starting stack (100bb). Prefer: **starting stack pre-post**, and build root by calling a controlled deal/post path so pot/street_commit match `new_hand` bit-exactly.

---

## 3. Clone `GameState` per node vs pure abstract public state

| Approach | Pros | Cons |
|----------|------|------|
| **Clone `GameState` at every public node** | Zero reimplementation risk | Huge RAM (history, hole_cards, full_board×2, study fields); cloning on every expand kills tree build |
| **Pure `PublicState` + shared chip math** | Compact tree; MCCFR-friendly | Must reimplement transitions carefully |
| **Hybrid (recommended)** | Best of both | Small bridge layer |

**Decision: Hybrid**

1. **`PublicState`** — only fields that affect betting / public history:

   ```
   street, pot, stacks[N], street_commit[N], total_commit[N],
   bet_to_call, last_raise_size, last_aggression_was_full_raise,
   street_level_acted[N], acted_this_street[N], actor, folded[N], all_in[N],
   board[0..5], button, bb, num_seats, last_aggressor
   ```

2. **Transitions** implemented in `cfr/public_state.rs` by porting the *chip* and *round-close* rules from `engine.rs` (same integer ops). No `history` Vec, no hole cards on the public node.

3. **Parity harness:** materialize a `GameState` (via `from_solver_root` + set holes if needed), apply the same sequence with `apply` / `apply_raise_chips`, assert public fields match after every action.

4. **Showdown:** call `hand_eval::evaluate_nlh` + `single_board_payout` with concrete combos (from range sampling or full enumeration), not via a live `GameState` if avoidable.

5. **When to use real `GameState` in the solve path:** optional debug flag only; never as the default tree payload.

---

## 4. Module layout under `rust_engine/src/cfr/`

```
rust_engine/src/cfr/
  mod.rs              # solve() dispatch: algorithm × street; re-exports
  types.rs            # RootSpec, SolveConfig, Strategy, SolveReport (Phase 0 ✅)

  # --- Phase 1 foundation ---
  public_state.rs     # PublicState; from_root; apply_abstract; round close / street advance
  actions.rs          # AbstractAction enum; menu from raise_sizes_pm; chips via pot math; legal filter
  engine_bridge.rs    # RootSpec/PublicState ↔ GameState for parity; ClubGG unit helpers
  range.rs            # ComboId 1326; block board; normalize; parse range strings; sample
  tree.rs             # PublicTree / PublicNode (Root|Chance|Action|Terminal); build river→turn→flop
  infoset.rs          # InfosetKey, regret + strategy_sum tables (DCFR accumulators)
  showdown.rs         # Terminal EV in chips/bb; fold + NLH showdown via hand_eval + side pots
  dcfr.rs             # Full-tree Vanilla / Linear / DCFR traverse + average strategy
  br.rs               # HU best response + exploitability_bb
  export.rs           # Strategy → JSON (infoset_id, actions, probs)

  # --- Phase 1b (parallel, cheap) ---
  toy.rs              # Kuhn + Leduc games sharing regret update trait

  # --- Phase 2 ---
  preflop.rs          # 169 classes; preflop public tree under abstraction
  mccfr.rs            # External-sampling MCCFR
  iso.rs              # Suit isomorphism (board + relative hole suits)

  # --- Phase 3 ---
  induce.rs           # Range induction: blueprint path → postflop RootSpec ranges
  multiway.rs         # num_seats 3..6 tree builder hooks + honesty flags
  card_abs.rs         # OCHS / bucket maps (toggle via SolveConfig.card_abstraction)
```

**Engine touchpoints (minimal, not a rewrite):**

| Change | Where | Why |
|--------|-------|-----|
| `GameState::from_solver_root(...)` (or free fn) | `engine.rs` or `cfr/engine_bridge.rs` | Mid-street / controlled preflop root without full deal RNG |
| Optionally extract pure `raise_chips_from_pot_frac` | shared by `compute_sizing_chips` + `cfr/actions` | One chip truth |
| PyO3: `solve` binding | `bindings.rs` + `lib.rs` | Flip `rust_cfr_available()` in Python |

**Do not put CFR tables inside `GameState`.** Keep `engine` = rules; `cfr` = solver.

---

## 5. Parity test strategy vs `BombPotEnv` / engine

Layers (all must stay green):

### A. Unit — abstract menu chips

For fixed public situations (HU, known pot/stacks/to_call):

1. Build `PublicState` / bridge `GameState`.  
2. For each `raise_sizes_pm` + ALLIN + fold/x-c, compute chips.  
3. Assert equal to engine: `min_raise_chips`/`max_raise_chips` bounds; apply via `apply_raise_chips`; resulting `pot`/`stacks`/`bet_to_call`/`last_raise_size` match.  
4. Cross-check pot-fraction math against `python/plo5bp/sizing.py` `anchor_grid_np` for shared pm values (e.g. 500, 1000) on NLH.

### B. Sequence parity — multi-action lines

Scripted lines (e.g. check–bet ½–raise ¾–call) on river:

- Path A: `BombPotEnv` / `PyGameState` with `VARIANT_NLH` + `apply_raise_chips`.  
- Path B: CFR `PublicState` transitions.  
- Assert after each act: pot, stacks, street_commit, bet_to_call, actor, folded, all_in, terminal flag.  
- On showdown with fixed holes+board: payouts match `env` / `payouts()`.

### C. Short all-in / reopen

Cases from existing engine tests (short shove below min-raise; lockout of already-acted seat; cover-short clamp):

- Abstract ALLIN that is a short raise must **not** reopen; legal menu after must match `legal_action_mask` raise availability (raise illegal while call legal).

### D. Ante + blinds preflop

HU ClubGG: ante 5k, sb 5k, bb 10k, stack 100bb:

- After root build: pot = 5k+5k+10k+5k = 25k (2.5 bb), street_commit SB/BB correct, first actor = SB/button.  
- Bit-exact vs `GameState::new_hand(GameConfig::new_nlh_uniform(...), seed, button)` public fields (ignore holes/boards if not yet dealt).

### E. Side pots

Unequal stacks all-in: terminal EV via CFR showdown module == `single_board_payout` (reuse engine tests’ commit patterns).

### F. Algorithm gates (not env, but required)

- Kuhn/Leduc: known value + expl → 0.  
- Tiny HU river (2–3 combos, 2 sizes): DCFR expl under target.  
- Determinism: same seed/config → same π (float tol).

### G. Native correctness (no external oracle)

Kuhn NE, jam/check known spot, engine-parity apply, and `scripts/cfr_verify.py` certificate.

**Python:** `tests/python/gto/test_cfr_api.py` + `tests/python/gto/test_cfr_solver.py`; Rust `#[cfg(test)]` in `cfr/*` for fast loops.

---

## 6. Risks (short)

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Short all-in reopen** | Wrong legal raises → wrong Nash | Port `short_shove_lockout` + `street_level_acted` into `PublicState`; parity tests from engine suite |
| **Ante dead vs blind live** | Wrong preflop pot/SPR | Always build preflop via same post order as `new_hand`; never “pot_bb only” without commit split when blinds matter for min-raise |
| **Multiway pots / side pots** | Wrong terminal EV | Reuse `single_board_payout`; multiway trees Phase 3 only; product language = population eq, not unique Nash |
| **Duplicate abstract sizes after clamp** | Inflated action dim / broken regrets | Dedup by chip amount; keep one ALLIN atom |
| **Dust / sub-chip** | Off-by-one vs `apply_raise_chips` dust snap | Call same dust rule (bb/100) or disable dust in solver mode with documented flag + parity |
| **Clone-heavy tree** | OOM on flop | PublicState only; refuse solve if estimated nodes > budget |
| **Range / board blockers** | Mass leakage | Normalize after block; induction tests (Phase 2) |
| **Study mode zeros** | False payouts if bridge uses study | Solver roots: `study_mode=false` always |
| **PPO Action enum confusion** | Silent wrong sizes | CFR never uses BetPct* for solve menu; only Fold/CheckCall + raise_chips |

---

## 7. Integration plan by phase

### Phase 1 — HU postflop foundation (implement first)

**Goal:** Real DCFR on engine-legal river (then turn/flop) with given ranges.

1. `actions.rs` + `public_state.rs` + `engine_bridge.rs`  
2. Parity tests A–C, E (river focus)  
3. `range.rs` (1326, block, uniform default)  
4. `tree.rs` river public tree (no chance if board fixed)  
5. `showdown.rs` + `infoset.rs` + `dcfr.rs` + `br.rs`  
6. Wire `solve()` → real report (`status=ok`, strategy dump, `exploitability_bb`)  
7. `export.rs` + PyO3 bind; CLI `scripts/cfr_solve.py` works end-to-end  
8. Phase 1b: `toy.rs` Kuhn/Leduc sharing update kernel  
9. Batch: `scripts/cfr_batch.py` N river roots (ops muscle)

**Exit:** river expl ≤ target on small roots; chip parity green; no GameState-per-node.

### Phase 2 — Preflop abstract spine

1. Preflop root via engine-compatible blind/ante post  
2. `preflop.rs` (169) + `mccfr.rs` external sampling  
3. Size abstraction (coarse menu)  
4. `induce.rs`: path → flop ranges → existing Phase 1 solver  
5. Scripted preflop solves + range dump JSON

**Exit:** open/3bet/call ranges sane; induced flop solves without API change.

### Phase 3 — Full-hand pipeline + multiway path

1. Glue CLI: preflop blueprint → line filter → induce → postflop DCFR  
2. Batch grids: seats × SPR × boards × lines  
3. `multiway.rs`: 3-way river first (MCCFR); honesty flag `population_mccfr`  
4. Heavier card abs (`card_abs.rs`) as needed  
5. Refuse oversize trees with abs suggestions

**Exit:** scripted P→R path; 3-way river infrastructure; still no false “the GTO” multiway claims.

### Phase 4 (out of this integration plan)

LabelRecord export / PolicyNet distill — only after Phase 1–2 trustworthy.

---

## 8. Concrete first implementation slice (after approval)

Order of commits (small, testable):

1. **`cfr/actions.rs` + `public_state.rs`**: abstract menu → chips; apply fold/x-c/raise/all-in; street not advanced yet (single-street river).  
2. **`engine_bridge` + parity tests** vs `GameState`/`BombPotEnv` NLH.  
3. **`from_solver_root`**: fixed board, pot, stacks, postflop actor.  
4. **River tree + DCFR + BR** on 2-combo toy then full 1326 with 2 sizes.  
5. **PyO3 + CLI** flip from `not_implemented`.

No preflop/multiway/neural until river parity + expl are green.

---

## 9. Summary decision table

| Question | Answer |
|----------|--------|
| Use `legal_action_mask` as CFR action set? | **No** — legality oracle + short-shove rules only |
| How to bet abstract sizes? | Chips from pot-frac → **`apply_raise_chips` semantics** |
| Store full `GameState` in tree? | **No** — compact `PublicState` + parity bridge |
| Showdown truth? | `evaluate_nlh` + `single_board_payout` |
| Preflop pot truth? | Same ante→blind post order as `new_hand` |
| Multiway day-1? | **No** — types allow later; MCCFR Phase 3 |
| Chip unit | Engine chips; report bb via `/ bb_chips` |
