#!/bin/bash
# vMin3 guardian — the full run on the settings the 2026-09-23/24 sweep and
# measurements picked (RunPod RTX PRO 6000, 233 GiB container, 27-core quota):
#   - network: actor 32x3, critic 128x2. Every actor from 32 to 128 matched a
#     128x3/128x2 control over updates 30-39 (h2h, duplicate deals); 16x3 lagged
#     badly. The critic's own loss rises as it shrinks (128: 3.27-3.30, 64:
#     3.36-3.40, 32: 3.42-3.45), so it stays 128. 32/128 had the best window.
#   - num_envs 2,640,000: the most rows/s (457k at a 44M-row target, vs 201k at
#     vMin2's 220k envs; 1.3M-3.5M are all within ~5%).
#   - rollout ROLLOUT_LENGTH rows (default below; owner: longer rollouts are
#     always better). --batch-on-host keeps the batch in host RAM (~570 B per
#     row + ~7 GiB) and --micro-batch-rows keeps the GPU at ~13 GiB, so host RAM
#     is the only bound. RUN IT ALONE: another trainer's ~30 GiB would push the
#     container past its memory limit -> this guardian refuses to start while any
#     scripts/train.py is running.
#   - everything else = the vMin2 recipe (v6 preset, minimal obs rev 2, 30 mixed
#     configs, entropy 0.25, lr 1.5e-4, drain on, checkpoint every update).
#
#   bash scripts/vMin3_guardian.sh                      # cold start
#   WARM=checkpoints/swA32_40.pt bash scripts/vMin3_guardian.sh   # warm start
#   ROLLOUT_LENGTH=200000000 bash scripts/vMin3_guardian.sh       # shorter rollout
#
# Clean stop: touch runs/vMin3.stop (the trainer finishes its update and saves).
set -uo pipefail
cd /workspace/plodbnet || exit 1

ROLLOUT_LENGTH=${ROLLOUT_LENGTH:-320000000}
NUM_ENVS=${NUM_ENVS:-2640000}
GLOG=runs/vMin3_guardian.log
STOPFLAG=runs/vMin3.stop
LOG=runs/vMin3.log
MAX_RESTARTS=4
POLL=300
restarts=0

log(){ echo "[vMin3-guardian $(date -u '+%m-%d %H:%M:%S')] $*" >> "$GLOG"; }
train_pid(){ pgrep -f "python -u scripts/[t]rain.py .*--checkpoint checkpoints/vMin3.pt" | head -1; }
other_trainer(){ pgrep -f "python -u scripts/[t]rain.py" | grep -v "^$(train_pid)\$" | head -1; }

launch(){
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PATH="$HOME/.cargo/bin:$PATH"
  export PLO5_RUST_ENCODER=1
  export PLO5BP_STEP_TIMERS=1
  export NUMPY_MADVISE_HUGEPAGE=0
  export MALLOC_MMAP_THRESHOLD_=33554432
  export MALLOC_TRIM_THRESHOLD_=17179869184
  export MALLOC_TOP_PAD_=67108864
  local load=""
  [ -n "${1:-}" ] && load="--load-checkpoint $1"
  log "launch: ${ROLLOUT_LENGTH} rows, ${NUM_ENVS} envs, ${load:-cold}"
  # No CPU pinning: the batch (~190 GiB at the default) is bigger than one
  # NUMA node's free memory.
  setsid nohup .venv/bin/python -u scripts/train.py \
    --variant plo5_double_bomb \
    --v6 \
    --obs-mode minimal \
    --hidden-dim 32 --num-layers 3 \
    --critic-hidden-dim 128 --critic-num-blocks 2 \
    --batched --device cuda \
    --num-envs "$NUM_ENVS" --rollout-length "$ROLLOUT_LENGTH" \
    --batch-on-host --micro-batch-rows 1000000 \
    --num-minibatches 16 --ppo-epochs 2 \
    --mix-configs --configs-per-tier 10 --mix-tiers clubgg,clubgg_deep,deep \
    --entropy-coef 0.25 --sizing-entropy-scale 1.0 \
    --lr 1.5e-4 --lr-warmup-updates 0 --clip-room-mid 0.07 \
    --target-kl 0.5 --kl-hard 10.0 --adv-clip 8 --cpu-threads 24 \
    --snapshot-every 5 --checkpoint-every 1 \
    $load --checkpoint checkpoints/vMin3.pt \
    --num-updates 100000000 >> "$LOG" 2>&1 < /dev/null &
  disown 2>/dev/null || true
}

compatible_ckpt(){
  [ -f "$1" ] || return 1
  .venv/bin/python - "$1" <<'PY'
import sys, torch
try:
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
except Exception:
    sys.exit(1)
cfg = ck.get("config") or {}
ok = (int(cfg.get("hidden_dim") or 0) == 32 and int(cfg.get("num_layers") or 0) == 3
      and int(cfg.get("critic_hidden_dim") or 0) == 128)
sys.exit(0 if ok else 1)
PY
}

pick_warm(){
  local f
  for f in $(ls -t checkpoints/vMin3_*.pt 2>/dev/null); do
    compatible_ckpt "$f" && { echo "$f"; return 0; }
  done
  [ -n "${WARM:-}" ] && compatible_ckpt "$WARM" && { echo "$WARM"; return 0; }
  return 1
}

if [ -f "$STOPFLAG" ]; then
  log "stop flag present at start -> not launching (rm $STOPFLAG to run)"
  echo "vMin3 guardian: $STOPFLAG exists -> not launching (remove it to run)" >&2
  exit 0
fi
if [ -n "$(other_trainer)" ]; then
  log "another scripts/train.py is running -> not launching (vMin3 needs the pod's memory to itself)"
  echo "vMin3 guardian: stop the other trainer first (e.g. touch runs/vMin2.stop)" >&2
  exit 1
fi

if [ -z "$(train_pid)" ]; then
  WARM_CKPT=$(pick_warm) || WARM_CKPT=""
  launch "$WARM_CKPT"; sleep 45
fi

log "started; watching pid=$(train_pid)"
while true; do
  [ -f "$STOPFLAG" ] && { log "stop flag present -> exiting"; exit 0; }
  sleep "$POLL"
  PID=$(train_pid)
  if [ -z "$PID" ]; then
    sleep 15; PID=$(train_pid)
    if [ -z "$PID" ]; then
      [ -f "$STOPFLAG" ] && { log "process gone and stop flag present -> exiting"; exit 0; }
      if [ "$restarts" -ge "$MAX_RESTARTS" ]; then log "DEAD; restart cap hit -> STOP"; touch "$STOPFLAG"; exit 0; fi
      restarts=$((restarts+1))
      WARM_CKPT=$(pick_warm) || WARM_CKPT=""
      log "process DEAD; relaunch #$restarts warm=${WARM_CKPT:-COLD}"
      launch "$WARM_CKPT"; sleep 45
    fi
  fi
done
