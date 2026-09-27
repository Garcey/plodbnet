#!/bin/bash
# recipe_run.sh STEM NODE UPDATES [train.py flags...] -- one candidate of the
# 2026-09-26 recipe search (the regression diagnosis found the vSix5 line
# converged to the fixed point of its recipe; candidates change the recipe).
# Base = vSix5's recipe exactly (v6 preset, full obs at obs rev 1, actor 2048x4,
# critic 1536x2, 30 mixed configs, host batch + 200k-row micro-batches, entropy
# 0.10, sizing-entropy scale 1.0, lr 1.5e-4) at a search scale (NUM_ENVS 220k,
# ROLLOUT_LENGTH 15M), warm-started from WARM (default checkpoints/vSix5.pt =
# u1290, whose Adam sidecar vSix5.optim.pt and pool siblings are restored), CPUs
# pinned to NUMA node NODE. Flags after UPDATES override the recipe (argparse
# keeps the LAST value), e.g.
#   bash scripts/recipe_run.sh r1e06 1 8 --entropy-coef 0.06 --sizing-entropy-scale 0.3
# Every candidate of a wave starts from the same weights, optimizer state, pool
# and random stream (the resume seed is (seed, 1290)): common random numbers.
# Log: runs/STEM.log; checkpoints STEM_1292.pt, ...; stop: touch runs/STEM.stop
# and kill -TERM the trainer (it finishes its update and saves).
set -uo pipefail
cd /workspace/plodbnet || exit 1
STEM=$1; NODE=$2; UPDATES=$3; shift 3
WARM=${WARM:-checkpoints/vSix5.pt}
LOG=runs/${STEM}.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PLO5_RUST_ENCODER=1
export PLO5BP_OBS_REV=1
export PLO5BP_STEP_TIMERS=1
export PLO5BP_ANNEAL_CONTROL=runs/${STEM}.control.json   # per-run live control file
export NUMPY_MADVISE_HUGEPAGE=0
export MALLOC_MMAP_THRESHOLD_=33554432
export MALLOC_TRIM_THRESHOLD_=17179869184
export MALLOC_TOP_PAD_=67108864
CPUS=$(cat "/sys/devices/system/node/node${NODE}/cpulist")
echo "[recipe $(date -u '+%m-%d %H:%M:%S')] $STEM node $NODE, $UPDATES updates, warm $WARM, overrides: $*" >> "$LOG"
setsid nohup taskset -c "$CPUS" .venv/bin/python -u scripts/train.py \
  --variant plo5_double_bomb --v6 \
  --hidden-dim 2048 --num-layers 4 --critic-hidden-dim 1536 --critic-num-blocks 2 \
  --batched --device cuda \
  --num-envs "${NUM_ENVS:-220000}" --rollout-length "${ROLLOUT_LENGTH:-15000000}" \
  --batch-on-host --micro-batch-rows 200000 \
  --num-minibatches 16 --ppo-epochs 2 \
  --mix-configs --configs-per-tier 10 --mix-tiers clubgg,clubgg_deep,deep \
  --entropy-coef 0.10 --sizing-entropy-scale 1.0 \
  --lr 1.5e-4 --lr-warmup-updates 0 --clip-room-mid 0.07 \
  --target-kl 0.5 --kl-hard 10.0 --adv-clip 8 --cpu-threads "${CPU_THREADS:-12}" \
  --snapshot-every 5 --checkpoint-every 1 \
  --load-checkpoint "$WARM" --checkpoint "checkpoints/${STEM}.pt" \
  --num-updates "$UPDATES" "$@" >> "$LOG" 2>&1 < /dev/null &
echo "$!"
