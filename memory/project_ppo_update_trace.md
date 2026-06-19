---
name: PPO update code trace and bottleneck map
description: empirical torch.profiler attribution of a PPO update — encoder ported to Rust dropped step1/encoder 312→49ms/call (6.4×); wall 395s→234s (k=3 MC) + encoder ports (v6)
type: project
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
## 2026-05-11 — v6 on runpod Blackwell, heads-up clean run

Pushed v6 to runpod (RTX Pro 6000 Blackwell, 96GB), rebuilt extension,
ran single seats=2 update at the same full-scale config v5 used
(24576 envs / 2088960 rollout / 32 minibatches):

- **Clean wall: 179.3s** (no profiler overhead). v5 baseline 234.4s
  had `--profile-one-update` overhead, so not perfectly apples-to-apples,
  but indicates ~24% reduction from changes shipped this session.
- py-spy flamegraph (10Hz, 1670 samples, `runs/profile_pyspy_v6_runpod.svg`):
  - `encode_observation_batch` (Python-side): ~9% of samples,
    down from v5 baseline ~32% (SF + cross-board + draw-flags +
    pair-features all in Rust now).
  - `_refresh` / `collect_rollout_batched` (bundle path):
    ~48% of samples — `obs_features_batch` bundle is now the bulk
    of CPU work, and is dominated internally by `opp_outcome_fractions`.
  - `_forward` (PPO inner forward in collect): ~13% (network forward
    + numpy take).

The encoder ports' impact is clearly visible in flamegraph: encoder
Python frames collapsed from 32% to 9% of samples. Bundle is now
the dominant attribution and the next ROI target.

## 2026-05-11 — bundle writeback parallelized + 3 encoder ports (v6 trace)

Shipped two changes in one bundle:
1. **Bundle writeback parallelization** in `pack_observation` (`bindings.rs`):
   `RowData` intermediate struct and serial 25-field writeback loop replaced
   with `(0..n).into_par_iter().for_each` writing directly into ndarray
   outputs via raw pointers (`OutPtrs` struct with `unsafe impl Send + Sync`).
   Zero per-env `Vec<u64>`/`Vec<bool>` allocations. Rust 2021 disjoint
   capture rules required `let ptrs = &ptrs;` inside the closure to force
   whole-struct capture (per-field `*mut T` is `!Sync` independent of the
   wrapper's Sync impl).
2. **Three Python encoder hotspots ported to Rust** (`bindings.rs`):
   `cross_board_straight_batch` (u128 bitset over 78 pair-bits × 10 SF
   windows, collapsing the Python (N, 78) intermediate to 4 scalars/env),
   `draw_flags_batch` (stack-local `[u8; 4]` suit counts + `u16` rank
   masks; both boards in one call), `pair_features_batch` (per-rank
   counts + 5-element insertion-sort for valid_ranks; both boards in
   one call). All three use rayon `par_iter` + `py.allow_threads`.

**Half-scale measurement** (12288 envs / 1044480 rollout, seats=2) — full-scale
24576-env profile failed pinned-memory OOM (system pinned-pool cap ~8GB,
slab needs ~11GB):

- `step1/encoder`: 176 × **48.99 ms/call** (was 312 in v5, **-84%, ~6.4× speedup**)
- `step1a_bundle/obs_features_batch`: 176 × 991 ms/call
- `step1a/refresh`: 88 × 1011 ms/call
- `step9f/reset_terminal`: 87 × 1081 ms/call
- **Wall: 227.1s** at half scale. Trace: `runs/profile_update0_v6_seats2.json`.

**Apples-to-apples encoder savings:** at full scale (v5 baseline 24576 envs),
the 312→49ms/call encoder shrinkage would save ~45-50s wall directly. Bundle
TOTAL at half-scale (174s for 176 calls) vs v5 (146s for 172 calls at 2× the
envs) cannot be directly compared because half-scale has fixed overheads and
worse rayon parallelism (12288 envs / 48 vCPUs = 256 per thread is small).
The bundle is now dominated by `opp_outcome_fractions` Monte Carlo work
(`engine.rs:951`) which this change did NOT touch — that's the next ROI.

**Next-optimization candidates** (post-v6):
1. Cache `opp_outcome_fractions` per (board_a, board_b, hero_hole) tuple
   within a hand — board doesn't change within a street.
2. Reduce `MC_SAMPLES_K4` from 1024 → 256 (feature noise probably below
   action-policy granularity).
3. Skip k=3,k=4 buckets at seats≤2 (already 0 contribution, but the Rust
   code may still run the MC loops).

## 2026-05-11 — k=3 forced to MC (v5 trace, seats=2)

Changed `opp_outcome_fractions` in `engine.rs:951` to always MC=1024 for k=3,4 (only k=2 stays exhaustive). Previously k=3 was exhaustive at turn (C(39,3)=9139) and river (C(37,3)=7770) which dominated bundle cost — ~18k evals/env vs ~2k for MC.

- `step1a_bundle/obs_features_batch`: 172 calls, **848 ms/call** mean (was 1829ms in v4, -53.6%)
- `step1a/refresh`: 88 × 1143ms (was 2123, -46%)
- `step9f/reset_terminal`: 83 × 1203ms (was 2151, -44%)
- `step1/encoder`: 172 × 312ms (basically unchanged from 291ms in v4)
- **Wall: 234.4s (was 395.3s in v4, -40.7%).** Trace: `runs/profile_update0_v5.json`.

**Feature accuracy implication:** turn/river k=3 fractions go from exact to MC-estimated. SE ≈ 1.4% for a 0.25 proportion at n=1024. Deterministic (seeded from observable state), so still reproducible across encoder passes. Training-obs behavior change — model receives slightly noisier k=3 features at turn/river than before. Should be flagged on next retrain.

## 2026-05-11 — after annotation split (v4 trace, seats=2)

Split refresh into `step1a_bundle/obs_features_batch` (Rust `observation_and_features_batch` call) and `step1a_unpack/*` (numpy unpack). Result attributes the previously-mysterious refresh non-encoder portion:

- `step1a_bundle/obs_features_batch`: **172 calls, 1828.8 ms/call mean, 314.6s total.** New dominant single cost in update.
- `step1/encoder`: 172 calls, 290.8 ms/call mean, 50.0s total. Down from 905ms baseline (-68%); from 527ms v3 (-45%).
- `step1a/refresh` (wrapper): 88 calls × 2123ms; `step9f/reset_terminal` (includes inner refresh): 83 × 2151ms.
- Wall: 395.3s. Trace: `runs/profile_update0_v4.json`.

**The "refresh non-encoder doubled" puzzle is now solved:** bundle cost was always ~1100ms/call (v1 estimate); it's actually ~1830ms/call (v4 measurement, ~70% higher than estimated). The encoder shrinkage exposes bundle as the new bottleneck. Inside the bundle, the prime suspect is `opp_outcome_fractions` in `engine.rs:951`: per-env runs C(41,2)=820 exhaustive enumerations + 1024 MC samples for k=3 + 1024 MC samples for k=4, every refresh call, regardless of seat count (the 12-dim feature is fixed-shape). With 24576 envs × ~3000 hand evals each × 172 calls, that's ~12.7B evals per update — almost certainly the dominant Rust cost.

**Next optimization options** (ordered by ROI):
1. **Reduce `MC_SAMPLES_K4` from 1024 to 256** — feature noise probably below action-policy granularity. Should ~3x speedup of k=4 portion.
2. **Cache `opp_outcome_fractions` within a hand** — board doesn't change within a street; only deck changes when board cards revealed. Memoize per (board_a, board_b, hero_hole) tuple. Per-env cache hit-rate inside a hand should be high.
3. **Drop k=3, k=4 entirely for heads-up** — those features are constant zero for heads-up (only 1 opp). But fixed obs shape complicates this without breaking parity.

## 2026-05-11 — after Rust SF port

Re-profiled with same half-scale config (24576 envs / 2088960 rollout / 32 minibatches) but seats=5 (sampler vs prior 2-seat). Apples-to-apples per-call:

- `step1/encoder`: **177 calls, 486 ms/call mean** — down from 905 ms (-46%, ~1.86× speedup).
- `_straight_flush_features_batch` no longer appears in py-spy flamegraph; top encoder hotspot is now `_cross_board_straight_batch` (14 sample frames, was ~1% before).
- Trace: `runs/profile_update0_v2.json`; flamegraph `runs/profile_pyspy_v2.svg`.

Wall went 390.5s → 520.7s but that's not the optimization regressing — 5-seat workload makes refresh's *non-encoder* path 25-28% slower per call. Need a heads-up rerun (seats=2) for a clean wall comparison.

## 2026-05-10 — prior baseline

Fact: a single PPO update on RTX Pro 6000 Blackwell is **encoder-bound, not GPU-bound or terminal-flush-bound**. Empirical attribution from torch.profiler trace 2026-05-10 (half-scale: 24576 envs, 2088960 rollout, 32 minibatches, batch=65280):

- Wall: 390.5s update. Self CPU: 384s. Self CUDA: 5.6s. **CUDA used ~1.5% of wall time.**
- `step1/encoder` (encode_observation_batch): **181 calls, 164s CPU, 0 CUDA, 905ms/call mean.** Called from both env._refresh() and reset_terminal_batch.
- `step1a/refresh` (env._refresh, includes encoder): 92 calls, 182s, 1.98s/call.
- `step9f/reset_terminal` (reset + second refresh + encoder): 88 calls, 179s, 2.03s/call.
- PPO inner loop (step12, all sub-spans): 4.6s wall, 3.2s CUDA. **<2% of update time.**
- Everything else: <5s combined (Rust apply 1.3s, slab copies 4.3s, opp forwards via step4, GAE 0.7s, pool_mix 0.3s, aggression bonus 0.6s, finalize_h2d 0.15s, pool_snapshot 42ms).

**Why:** user reported persistent ~30s CPU=2% / GPU=0% windows "a few times per update" — interpreted as terminal-flush dips. The empirical trace shows it isn't dips. It's steady-state: 384s of single-threaded CPU encoding on a 48-vCPU box = ~2% mean utilization. The GPU only does meaningful work during the 4.6s PPO inner loop (~1% of wall). The "few times per update" cadence the user perceived was probably the PPO inner-loop bursts between long encoder-bound rollout stretches.

**How to apply:** the next optimization must target encode_observation_batch — nothing else moves the needle. The terminal-flush suspects from the prior speculative version of this memory (GAE scan, slab copies, pool-mix, per-street reductions) are all <5s and not worth touching. The H100 NVL upgrade decision is moot until the encoder is parallelized: the bigger GPU would still idle 98% of the time. Phase 3B (overlap collect/update) only helps after the encoder is fast enough that the collect phase isn't itself dominated by single-threaded work.

---

## Trace table (sorted by CPU)

```
step                              n_calls       cpu_ms      cuda_ms
step1a/refresh                         92     182456.2          0.0
step9f/reset_terminal                  88     178679.7          0.0
step1/encoder                         181     163897.1          0.0   <-- nested in both above
step12/inner_loop                       1       4586.9       3160.2
step9d/slab_copies                     88       4288.1          0.0
step6+7/rust_apply                     92       1319.1          0.0
step12a/evaluate                      132       1293.4        151.7
step5/action_d2h                       92       1272.9       1115.6
step2/learner_h2d                      92        922.4          0.0
step3/learner_forward                  92        765.2        297.5
step9c/gae_scan                        88        689.9          0.0
step12c/backward                      132        681.3        543.9
step8/aggression_bonus                 92        592.9          0.0
step9a/payouts                         88        468.8          0.0
step9e/pool_mix                        88        261.7          0.0
step9b/retroactive_bonus               88        207.1          0.0
step12d/optimizer_step                132        160.9         96.8
step11/finalize_h2d                     1        150.2          0.0
step12b/loss                          132        116.8        215.4
step14/pool_snapshot                    1         42.5          0.0
step12e/kl                            132         10.3          3.6
step13/stats_sync                       1          4.7          3.7
```

Trace: `runs/profile_update0.json` (66 MB). Re-parse with `scripts/summarize_profile.py`.

## What this rules out

- **PPO inner loop as the dip cause** — 4.6s wall total, GPU=85% during it, contradicts user's "few times per update" and GPU=0% observation.
- **Terminal-flush GAE / slab copies / pool-mix / per-street reductions** — all combined <6s. Not worth optimizing.
- **`non_blocking=True` at rollout.py:782-784** — fix would save <1s; encoder dominates by 2 orders of magnitude.
- **Pool-snapshot cost** — 42ms total per update; not a factor.

## What's next

py-spy flamegraph 2026-05-10 (10306 samples) localizes the encoder cost to **`_straight_flush_features_batch` at `encoding.py:1010-1145`**. The two calls at `encoding.py:1354-1355` (one per board) account for ~19% of all samples; inner numpy `_sum` at lines 1079-1081 contribute another ~13%. **Combined: ~32% of total update wall time.** It's pure Python: a 10-iteration loop over straight windows, each allocating four `(N, 13, 4)` bool arrays and reducing them.

In samples, secondary hot spots in encoding.py:
- `_cross_board_straight_batch` (called at 1457): ~1% — similar shape, much smaller.
- Card-bit one-hot scatter at 1188: ~1%.

The rest of encode_observation_batch is small contributions from many features (rank histograms, pair-with-board, etc.).

**Phase 2's claim that `pack_observation` was parallelized doesn't apply here** — `_straight_flush_features_batch` is Python+numpy (not in Rust). The current Phase 2 work parallelized the *Rust* side of encoding, but the Python-side straight/flush feature compute (added 2026-05-06) became the new bottleneck and was never moved to Rust.

Optimization options, ordered by ROI:
1. **Stack both boards into one call.** Lines 1354-1355 run the same kernel on board_a and board_b. Stack hole/board into shape (2N, ...), call once, split. ~halves the function cost.
2. **Move `_straight_flush_features_batch` to Rust with rayon par_iter.** Per-env work is small and independent — ideal fit. Expected 10-20× speedup vs Python.
3. **Pre-allocate workspace.** Each call allocates ~48 MB of bool arrays per invocation × 2 boards × 181 calls/update = ~17 GB churn. Pre-allocated buffers cut allocator pressure significantly even without other changes.
