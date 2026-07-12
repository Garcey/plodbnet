# V7 obs — batch-2 implementation plan (stack + board + dual)

Branch `v7-obs`. Batch 2 = the 25 kept stack/board/dual dims, ruled
2026-07-12. Implemented as a **pure tail append after 1019** (no
middle-of-layout surgery), so every downgrade projection and old-checkpoint
UI serving keeps working by tail-slice, exactly as the obs-v2 append did.
Position + history (batch 1) and the 32→50 history-window widening are a
SEPARATE later pass; they reshuffle the middle and will renumber the tail
then — fine, cold-start v7, nothing trained yet.

`OBS_DIM: 1020 → 1171` (+151).

## Dependency order (the "be mindful of order" answer)

Two dependency classes:

1. **Shared engine exposure** — `acted_this_street` (STK-1). Also wanted by
   POS-3/HIST-3 in batch 1, so expose it cleanly now.
2. **Own engine computation** — BRD-7 (boat-outs fn), BRD-12 (improve-outs
   fn), DUAL-2 (winning-pair identity), DUAL-4 (k=2 g_min/g_max in the fused
   pass). Each is independent of the others.

Everything else is pure-encoder from already-exposed arrays (`total_commit`,
`street_commit`, `folded`, `all_in`, effective stacks, `pot`, `bet_to_call`,
`min/max_bet`, board card-index arrays, `per_board_outcome` 991..999).

**Two green commits:**

- **Chunk A (pure encoder, NO rebuild):** bump OBS_DIM, add all offset
  constants (incl. reserved ranges for the 5 engine dims), implement the 20
  pure-encoder dims in BOTH `encode_observation` (serial) and
  `encode_observation_batch`, hard-disable the Rust obs encoder for the new
  width (project precedent — re-port later), bump the batched packer width.
  Engine-dim columns stay zero (reserved) → serial/batched parity holds.
  Parity + sanity tests. **Tree green.**
- **Chunk B (engine dims, ONE rebuild):** expose `acted_this_street`; add
  BRD-7/BRD-12 Rust free fns; DUAL-2 winning-pair; DUAL-4 g_min/g_max; wire
  observation_dict + batched packer arrays; fill the reserved columns in
  both encoders; parity tests. **Tree green.**

## Tail offset map (1020 → 1171)

Parity invariant for every dim: the batched encoder must reproduce the
scalar path's f64-intermediate-then-cast-to-f32 arithmetic bit-exactly
(the encoder does all scalar math in f64 and casts only on assignment).

### Stack geometry — 1020..1061 (41)
| dim | off | n | engine? |
|---|---|---|---|
| STK-1 money/raise behind | 1020 | 4 | **acted_this_street** |
| STK-2 raise-ladder envelope | 1024 | 6 | no |
| STK-4 seat commitment ratio | 1030 | 8 | no |
| STK-5 spr-after-action | 1038 | 4 | no |
| STK-6 geometric jam plan | 1042 | 2 | no |
| STK-7 pot-ceiling implied odds | 1044 | 2 | no |
| STK-8 side-pot eligibility | 1046 | 3 | no |
| STK-9 call-risk fraction | 1049 | 2 | no |
| STK-10 ante-pot bloat | 1051 | 2 | no |
| STK-11 per-seat price-to-continue | 1053 | 8 | no |

### Board texture — 1061..1139 (78)
| dim | off | n | engine? |
|---|---|---|---|
| BRD-1 board rank ladder | 1061 | 10 | no |
| BRD-2 board suit census | 1071 | 12 | no |
| BRD-4 arrival volatility | 1083 | 6 | no |
| BRD-5 hero vulnerability outs | 1089 | 6 | no |
| BRD-6 straight out union | 1095 | 4 | no |
| BRD-7 boat+ outs | 1099 | 2 | **Rust fn** |
| BRD-8 fd rank quality | 1101 | 4 | no |
| BRD-9 backdoor draw census | 1105 | 4 | no |
| BRD-10 future nut-flush blocker | 1109 | 2 | no |
| BRD-11 turn/river card identity | 1111 | 20 | no |
| BRD-12 hero improve outs | 1131 | 4 | **Rust fn** |
| BRD-13 board nut ceiling | 1135 | 4 | no |

### Double-board — 1139..1171 (32)
| dim | off | n | engine? |
|---|---|---|---|
| DUAL-1 split-adjusted price | 1139 | 2 | no |
| DUAL-2 best-hand card usage | 1141 | 10 | **winning-pair** |
| DUAL-3 nut-lock/freeroll flags | 1151 | 6 | no |
| DUAL-4 guaranteed pot share | 1157 | 5 | **k=2 g_min/max** |
| DUAL-5 villain cross-board cover | 1162 | 9 | no |

Reserved-for-Chunk-B (zero in Chunk A): STK-1, BRD-7, BRD-12, DUAL-2,
DUAL-4 = 25 dims. Pure-encoder Chunk A = 126 dims.

## Status (2026-07-12)

- **Scaffolding: DONE, green (commit b70024f).** OBS_DIM 1171, all tail
  constants, 6 no-op helper stubs wired into both encoders, Rust obs-encoder
  width-gated off, test pins updated. Reserved columns zero → parity holds.
- **Chunk A (20 pure-encoder dims, 126 cols): DONE, green (commit 5db3071).**
  Both encoders filled for STK-2/4/5/6/7/8/9/10/11, BRD-1/2/4/5/6/8/9/10/11/13,
  DUAL-1/3/5. Gates: serial↔batched bit-exact parity (test_encoding_batch.py
  sweep), correctness (test_obs_v3_batch2.py + a 3-agent adversarial review
  that re-derived every dim vs spec, ZERO bugs), full suite 684 pass / 65 skip.
  One parity bug caught pre-commit (BRD-11 op order). obs_adapter generalized
  to prefix-slice any width in [991, OBS_DIM).
- **Chunk B (5 engine dims): IMPLEMENTED 2026-07-12 — verification in flight.**
  Rust: hand_eval.rs gains boat_plus_outs / improve_outs / best_pair_mask
  (+ best_rank_using_added incremental enumeration + pair_best_cks shared
  core; 8 hand-computed unit tests); engine.rs gains GameState::hero_board_v3
  (NLH/no-actor/short-board guarded) and outcome_features_mc N_OUT 20→22
  (k=2 g_min/g_max share trackers — no new evals/RNG; P1 pin updated: dims
  0..20 still byte-identical vs the frozen reference, quarter-grid check on
  20/21); bindings.rs: acted_this_street + hero_board_v3 + share_bounds
  through PackedObservation, the unsafe pack loop, the [f32;22] outcome
  cache, all 5 dict sites, and the serial observation_dict ([12..20] slice
  fix). Python: STK-1/BRD-7/BRD-12/DUAL-2/DUAL-4 filled in BOTH encoders
  (None-guarded kwargs). Gates green so far: cargo 153/153, parity sweep
  50/50 (engine values flow byte-identical), batch-2 semantics 10/10
  (two-bits-per-board, quarter grid, DUAL-3↔DUAL-4 consistency, dict-vs-
  encoder recompute). Build gotcha hit + handled: maturin can't replace
  _engine.pyd while the UI holds it — stop :8765 first, rebuild, restart.
  In flight: full pytest suite + a single adversarial reviewer over the
  Chunk B diff; commit follows both.
  Original plumbing map (for reference): Each needs:
  a `PackedObservation` field (bindings.rs:1866), a fill in the parallel
  raw-pointer packing loop (~bindings.rs:2116), dict entries in ~5 batched
  bundle variants (observation_arrays + encoded variants) AND the scalar
  `observation_dict` (~bindings.rs:328), then the reserved encoder columns
  filled (serial+batched) + reserved-zero test updated. Build in buildable
  increments (maturin develop --release from repo root; verify PyInit symbol).
  The fused pass returns `[f32;20]` via `outcome_features_mc`, cached as
  `[f32;20]` (bindings.rs:2009/2022) — DUAL-4 extends this to `[f32;22]`
  (g_min,g_max appended; existing 20 byte-identical, cache type bumps).
  Specs (V7_OBS_CANDIDATES.md, verified 2026-07-12):
  - STK-1: `acted_this_street` already a `GameState` field (state.rs:245) —
    expose in observation_dict + batched packer (`observation_arrays` and the
    3 bundle variants in bindings.rs) + Rust encoder input.
  - BRD-7: Rust free fn `boat_plus_outs(hole, board)->u8` per board.
  - BRD-12: Rust free fn `improve_outs(hole, board)->(u8,u8)` per board
    (distinct improve cards / actual unseen size; best-cat combo count /10).
  - DUAL-2: `evaluate_plo5_partial` records the argmax hole-pair (tie-break =
    lexicographically smallest (lo,hi)); expose per-board 5-bit mask.
  - DUAL-4: append g_min/g_max to the fused pass output (N_OUT 20→22, existing
    dims byte-identical); expose; encoder derives 5 dims.
  - One maturin rebuild covers all five.

## Test strategy

- `tests/python/test_obs_v3_batch2.py`: serial vs batched bit-exact parity
  on a seeded battery of nodes across tiers (reuse the probe gen pattern);
  per-dim sanity (ranges, zero-on-terminal, hero-rotation, HU/all-in edges);
  reserved columns assert exactly 0.0 until Chunk B.
- Existing `test_encoding.py` / `test_env_batched.py` OBS_DIM pins updated.
- Rust obs-encoder parity test (`test_encoding_rust.py`) gated off for the
  new width with a skip + note (re-port is deferred v7 work).
