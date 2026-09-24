#!/bin/bash
# tune_run.sh STEM NODE UPDATES [train.py flags...] -- one hyperparameter-tuning
# candidate (2026-09-24): the vMin3 recipe (actor 32x3 / critic 128x2, v6 preset,
# minimal obs, 30 mixed configs, host-resident batch + 1M-row micro-batches) at a
# tuning scale (1.76M envs, 44M-row target -> ~58M rows / update), warm-started
# from WARM (default checkpoints/swA32_40.pt, whose Adam sidecar swA32.optim.pt is
# restored) and run for UPDATES updates, CPUs pinned to NUMA node NODE. Flags after
# UPDATES override the recipe (argparse keeps the LAST value), e.g.
#   bash scripts/tune_run.sh t1lr500 1 20 --lr 5e-4
# Every candidate of a wave starts from the same weights, optimizer state, opponent
# pool and random stream (the resume seed is (seed, 40)), so the waves compare the
# overridden coefficients alone. Log: runs/STEM.log; checkpoints STEM_41.pt, ...
# Stop early: touch runs/STEM.stop, then kill -TERM the trainer.
set -uo pipefail
cd /workspace/plodbnet || exit 1
STEM=$1; NODE=$2; UPDATES=$3; shift 3
WARM=${WARM:-checkpoints/swA32_40.pt}
LOG=runs/${STEM}.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PLO5_RUST_ENCODER=1
export PLO5BP_STEP_TIMERS=1
export NUMPY_MADVISE_HUGEPAGE=0
export MALLOC_MMAP_THRESHOLD_=33554432
export MALLOC_TRIM_THRESHOLD_=17179869184
export MALLOC_TOP_PAD_=67108864
CPUS=$(cat "/sys/devices/system/node/node${NODE}/cpulist")
echo "[tune $(date -u '+%m-%d %H:%M:%S')] $STEM node $NODE, $UPDATES updates, warm $WARM, overrides: $*" >> "$LOG"
setsid nohup taskset -c "$CPUS" .venv/bin/python -u scripts/train.py \
  --variant plo5_double_bomb --v6 --obs-mode minimal \
  --hidden-dim 32 --num-layers 3 --critic-hidden-dim 128 --critic-num-blocks 2 \
  --batched --device cuda \
  --num-envs "${NUM_ENVS:-1760000}" --rollout-length "${ROLLOUT_LENGTH:-44000000}" \
  --batch-on-host --micro-batch-rows 1000000 \
  --num-minibatches 16 --ppo-epochs 2 \
  --mix-configs --configs-per-tier 10 --mix-tiers clubgg,clubgg_deep,deep \
  --entropy-coef 0.25 --sizing-entropy-scale 1.0 \
  --lr 1.5e-4 --lr-warmup-updates 0 --clip-room-mid 0.07 \
  --target-kl 0.5 --kl-hard 10.0 --adv-clip 8 --cpu-threads 24 \
  --snapshot-every 5 --checkpoint-every 1 \
  --load-checkpoint "$WARM" --checkpoint "checkpoints/${STEM}.pt" \
  --num-updates "$UPDATES" "$@" >> "$LOG" 2>&1 < /dev/null &
echo "$!"
