# Rollout / obs performance findings (2026-07-12)

Diagnosis + optimizations for the ~3× rollout wall-clock regression after the
v7 obs batch-2 tail (OBS_DIM 1020 → 1171). User reported 9M-step rollout
~8 min → ~25 min; suspected `opp_outcome` but that was **not** the new cost.

## Root cause (confirmed by microbench)

- **`opp_outcome` / `outcome_features_mc`**: still ~40–55 µs/decision (train MC
  384). Already paid before v7; fused pass + outcome_seed cache in batch path.
  **Not** the 3×.
- **New cost = Python packing of the 151 v7 dims**, especially:
  - **BRD-5 / BRD-6** (hero vulnerability + straight out union) — worst
  - **DUAL-5** scoop-pair count
  - **STK-2** full `anchor_grid_np` just for legal-anchor fraction
- Batched BRD-5/6 was **slower per-row than serial** (nested Python
  `for rc × for w` over `(N,13)` bool matrices).
- Engine `hero_board_v3` residual ~15–20 µs — not the problem.
- NN width 1020→1171 is negligible vs encode.

### Pre-fix serial flop microbench (approx)

| Stage | µs |
|---|---|
| outcome_features_mc(384) | ~42 |
| observation_dict residual (incl hero_board) | ~15–20 |
| encode_observation full | **~450** |
| encode without all v3 blocks | ~103 (v3 ≈ **77%** of encode) |
| board_v3 alone | ~180–200 (41–44% of encode) |
| stack_v3 / dual_v3 | ~120 / ~75–110 |
| batch encode N=128 | **~131 µs/row** (board alone ~101 µs/row) |

## What was implemented (2026-07-12)

### Round 1 — pure Python (parity-preserving)

1. **Bitmask BRD-5/6** serial helpers (`_danger_straight_outs`,
   `_straight_out_union`) + rewritten batched path (`u16` + `np.bitwise_count`).
2. **Bitmask DUAL-5** serial (`_scoop_pair_count`); batch already bitmask.
3. **`n_legal_anchors_np`** in `sizing.py` — STK-2 dim 5 without full grid.
4. Diagnostic: `scripts/bench_obs_blocks.py`.

After R1: serial encode ~316 µs; batch ~104 µs/row; board batch ~72 µs/row.

### Round 2 — Rust hot paths + fuse

1. **`hero_board_one`** fused path (one `pair_best_cks` + one unseen scan →
   boat, improve, combos, mask). Free fns call fused; `GameState::hero_board_v3`
   uses it. Unit: `hero_board_one_matches_free_fns`.
2. **Rust BRD-5/6/DUAL-5**: `danger_straight_outs`, `straight_out_union`,
   `scoop_pair_count`, fused `board_draw_v3` → `[ds_a, ds_b, u_a, n_a, u_b,
   n_b, scoop]`. Wired through `GameState::board_draw_v3`, serial
   `observation_dict`, batched packer (`PackedObservation.board_draw_v3`).
3. Python encoder consumes `board_draw_v3` when present; pure-Python fallback
   for pre-v7 fixtures.
4. **Board-static cache deferred** — low ROI once hot loops were in Rust.

After R2 (same flop / N=128 bench):

| Path | Original v7 | After R1 | **After R2** |
|---|---|---|---|
| Serial encode | ~450 µs | 316 | **~245 µs** |
| Batch encode / row | ~131 µs | 104 | **~35 µs** |
| Board batch / row | ~101 µs | 72 | **~12 µs** |
| `_refresh` / row | ~146 µs | 110 | **~39 µs** |
| Engine pack / row | ~5 µs | — | **~3.8 µs** |

~3.7× faster batched encode/refresh vs post-v7 baseline. Full
`tests/python/` green: **686 passed, 65 skipped**.

### Key files

- `python/plo5bp/encoding.py` — v7 helpers, bitmask, board_draw consumer
- `python/plo5bp/sizing.py` — `n_legal_anchors_np`
- `rust_engine/src/hand_eval.rs` — fused hero_board + board_draw
- `rust_engine/src/engine.rs` — `hero_board_v3`, `board_draw_v3`
- `rust_engine/src/bindings.rs` — pack + dict export of `board_draw_v3`
- `scripts/bench_obs_blocks.py` — re-measure after further changes

Build note: maturin can't replace `_engine.pyd` while UI holds it — stop
:8765 first (`maturin develop --release` from repo root).

## Remaining training-loop recommendations (not yet implemented)

Rollout still ~95% of update wall-clock; PPO ~3–5% (already compile'd on CUDA).
Estimates = **end-to-end update** vs post-R2 baseline.

| # | Item | Est. wall-clock | Notes |
|---|---|---|---|
| 1 | Skip **encode** for newly-terminal envs on post-apply full `_refresh` (still need post-apply pack for commits) | **+1–5%** (typ ~2–3%) | Full refresh then subset-refresh of terminals wastes encode |
| 2 | Subset `all_hole_cards` (only re-dealt envs) | **+0.5–2%** (typ ~1%) | Today full `all_hole_cards_batch` every terminal flush |
| 3 | Reuse `BatchedBombPotEnv` / Rust engine across sub-rollouts | **+0–1%** single-config; **+2–8%** multi-config | Constructed every `collect_rollout_batched` |
| 4 | **Full Rust obs encoder** (history/SF/stack/multi-hots; width gate off at 1171) | **+15–30%** (typ ~20%) | Only remaining double-digit lever |
| 5 | Pinned **step-batch** H2D only (not multi-GB finalize slab) | **+0–3%** CUDA | Pin tax on big slab correctly avoided |

### Medium / situational

- EV runouts 64→0: **+2–8%** shove-heavy, **harms reward quality** (default 64 is intentional).
- `TRAIN_OPP_OUTCOME_MC` 384→256: **+0–5%** often small with outcome cache; 256 once destabilized gate.
- Obs storage 1171 vs 991: ~1% H2D/RAM only.
- Bulk traj writes / vectorize terminal pool-mix / keep opp snaps on GPU: **+0–4%** each, messy.

### Stacked rough scenarios

- Quick safe (#1+#2): **~+2–6%**
- + multi-config reuse (#3): **~+5–12%**
- + full Rust encoder (#4): **~+20–40%** vs post-R2
- Not another 3× without redesign; PPO tweaks &lt;5% of update

### Suggested implement order

1. #1 + #2 (small, parity-testable)
2. #3 if mix-configs heavy
3. #4 for next real step
4. #5 only if profile shows H2D/sync high

### Already good (don't chase first)

- Subset refresh after terminal reset
- Outcome MC cache (`outcome_seed`)
- Coalesced action D2H; critic reuses learner `o_t`
- Snapshot model cache across multi-config sub-rollouts
- No pin of multi-GB finalize obs slab
- PPO `torch.compile` on CUDA

## How to re-measure

```bash
.venv/Scripts/python scripts/bench_obs_blocks.py --n 600 --batch 128
.venv/Scripts/python scripts/train.py --profile-one-update --batched ...
# Chrome trace: runs/profile_update0.json + scripts/summarize_profile.py
```

Constants of interest: `rollout.TRAIN_OPP_OUTCOME_MC = 384`,
`train.EV_RUNOUT_SAMPLES = 64`, `env_batched` `_RUST_ENCODER_OBS_DIM` width gate,
`PLO5_RUST_ENCODER` env (force-off at OBS_DIM 1171).
