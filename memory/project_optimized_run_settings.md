---
name: optimized.pt / optimized2.pt / optimized3.pt run lineage (2026-05-11 → ongoing)
description: Hyperparameters and runtime context for the long RunPod training run; phase 1 cold-start (u0→u1488), phase 2 crashed at u12 (CUDA OOM), phase 3 warm-restart with phase-1 rollout + lowered entropy
metadata:
  type: project
---

## Phase 3 (active, since 2026-05-25 ~22:30 UTC, PID 21358)

Warm-started from `checkpoints/optimized_1485.pt`, output stem
`checkpoints/optimized3.pt`. Launched via
`/workspace/plodbnet/launch_optimized3.sh` (local copy at repo root).

Phase-1 rollout/minibatches envelope (proven-fit), keeping phase-2's
lowered entropy floors. Only knob changed vs phase 1 is the schedule:
ent 0.09/0.11/0.15 → **0.07/0.09/0.12**.

```
.venv/bin/python -u scripts/train.py \
  --batched --device cuda \
  --hidden-dim 2048 --num-layers 4 \
  --num-envs 49134 \
  --rollout-length 6266880 \
  --num-minibatches 48 \
  --block-rotation clubgg:0.07,clubgg_deep:0.09,deep:0.12 \
  --block-size 50 \
  --num-updates 100000000 \
  --load-checkpoint checkpoints/optimized_1485.pt \
  --checkpoint checkpoints/optimized3.pt
```

Expected cadence: ~10-11 min/update (same as phase 1).

`--load-checkpoint` only restores weights (`scripts/train.py:494, 512`)
— Adam moments reset. First 1-3 updates may be noisier than phase-1's
tail; not a schedule failure.

`rollout_length` is total transitions across all envs, not per-env
(see `rollout.py:801` — `while wcursor < rollout_target`). At
6,266,880 / 49,134 ≈ 127 transitions/env average.

## Phase 2 (crashed, 2026-05-23 04:50 UTC, PID 20786 at u12)

Warm-started from `optimized_1485.pt`, attempted rollout ×1.5
(9.4M) + minibatches 48→72. Cmdline lived in `launch_optimized2.sh`.

Crashed at u12 with **CUDA OOM in `rollout.py:310`** trying to
allocate **33.66 GiB** for the rollout obs slab during finalize H2D.
**Root cause: my VRAM math only modeled the per-minibatch peak.**
The full rollout obs tensor (~36 GB at OBS_DIM 959, 4B, rollout 9.4M)
is transferred to GPU as one contiguous block at finalize time
(`rollout.py:310`) and stays resident through the entire PPO update;
`iter_minibatches` (`rollout.py:1115-1127`) just indexes into it.
Number of minibatches does NOT shrink this footprint.

**Lesson for sizing future rollout-scaling changes:** GPU-resident
ceiling = rollout-slab (scales with rollout_length) + minibatch peak
(scales with rollout_length / num_minibatches) + model + optimizer +
fragmentation overhead. At ~95 GB total, keep the obs slab ≲ 28-30 GB
→ rollout ≲ 7.5-8M for OBS_DIM=959.

Only `optimized2_5.pt` and `optimized2_10.pt` exist; orphaned, not
materially better than `optimized_1485.pt` (Adam reset, 10 updates of
clubgg-only at the new ent=0.07).

## Phase 1 (cold-start, ended 2026-05-23 ~01:30 UTC at u1488)

PID 12418 ran 2026-05-11 → 2026-05-23 (~11d 19h), producing
`checkpoints/optimized_<update>.pt`. Killed via SIGKILL (SIGTERM did
not take inside 60s — likely mid-PPO-step). None of the
`launch_f2.sh`/`launch_f3.sh`/`launch_profile*.sh` scripts in the repo
match its cmdline; it was launched by hand.

Cmdline (as captured from `/proc/12418/cmdline`):

```
.venv/bin/python -u scripts/train.py \
  --batched --device cuda \
  --hidden-dim 2048 --num-layers 4 \
  --num-envs 49134 \
  --rollout-length 6266880 \
  --num-minibatches 48 \
  --block-rotation clubgg:0.09,clubgg_deep:0.11,deep:0.15 \
  --block-size 50 \
  --num-updates 100000000 \
  --checkpoint checkpoints/optimized.pt
```

Reached u1488 before kill; cadence ~10-11 min/update.

Derived: batch_size = 6266880 / 48 = **130560** (logged at startup).

Cadence: ~10-11 min/update; reached u1487 by 2026-05-22 22:30 UTC
(~11d 16h elapsed, ~1490 updates total).

**Why these values (vs the project standards in
[[feedback_training_envs_rollout]]):**

- `--num-envs 49134` is ~32× the standard 1536; `--rollout-length
  6266880` is ~24× the standard 262144. The host GPU is an
  RTX PRO 6000 Blackwell (97 GiB VRAM), and the run allocates
  ~94 GiB — the large batch is what justifies that hardware.
- `--num-minibatches 48` (vs default 32) keeps per-update gradient
  granularity in line with the 4× rollout step. Aligns with
  [[feedback_ratio_defaults]] — scale minibatches with rollout.
- 3-tier `--block-rotation clubgg:0.09,clubgg_deep:0.11,deep:0.15`
  with `--block-size 50`: cycles 50-update blocks through three
  curriculum tiers, each with its own entropy floor. The deep tier
  matches the [[project_late_stage_entropy_plan]] anneal target.
- **Cold start** (no `--load-checkpoint`) — first 12 updates show
  warm-up H swings (0.30-0.70) before stabilizing. By u50 it's
  rotating into block 2/3 (clubgg_deep) cleanly.

**How to apply:**

- Treat any `optimized_<update>.pt` as a 2048×4 cold-start run, not
  a warm extension of the f2 series. Don't compare its early-update
  metrics against f2/f3.
- The run is the source of every `optimized_*.pt` in `checkpoints/`.
  Latest promoted to UI is u1240 → `stub.pt` (2026-05-20); newer
  checkpoints (1245 .. 1470+) accumulate every 5 updates on RunPod.
- If a future invocation tries to clone these settings, copy the
  cmdline above verbatim — note the unusual `49134` env count and
  the per-tier entropy on `--block-rotation`.
- Remote repo state: at commit `5dc9f2b` with 11 modified files
  (engine + ppo + encoding + train), matching local working tree.
  Encode changes that produced new OBS_DIM bits ([[project_pair_with_board_added]],
  [[project_hero_rank_histogram_added]], etc.) are present in the
  running engine.
