#!/bin/bash
# vSix5 guardian (2026-09-26) — the LIVE SITE's lineage (vSix4_1240: full obs,
# actor 2048x4, critic 1536x2, obs rev 1) continued on the efficient pipeline.
#
# Why: the 2026-09-26 deep dive put the minimal-obs line far behind it —
# scripts/h2h_cross.py (each model served its observation the way the site
# serves it) scored vMin3 u239 -2.06 +- 0.04 bb/seat-hand sampled and -2.42
# +- 0.03 argmax-vs-argmax against vSix4_1240 (vMin3 u200: -2.25 / -2.67),
# although vMin3 had seen ~5x more rows (~55B vs ~11B). The minimal obs drops
# the engineered hand-strength features (made-hand categories, opp-outcome MC
# equities, blockers, draws) and its 32-wide actor had reached rank99 28/32.
#
# Recipe = vSix4's (vSix4_guardian.sh): v6 preset, 2048x4 / 1536x2, lr 1.5e-4,
# 2 epochs x 16 minibatches, 30 mixed configs, sizing-entropy scale 1.0, and
# entropy 0.16 = where vSix4's anneal had brought every tier by u1240. Obs rev 1
# (PLO5BP_OBS_REV=1) like the live site, so a checkpoint here is a drop-in
# promotion. What changes is only the pipeline: ~20x more envs, rollouts of
# ROLLOUT_LENGTH rows (vSix4: 9M), the batch in host RAM with micro-batched PPO,
# the drain on (a new stem), CPUs pinned to the GPU's NUMA node, a checkpoint
# every update.
#
#   WARM=checkpoints/vSix4_1240.pt bash scripts/vSix5_guardian.sh    # first launch
#   ROLLOUT_LENGTH=40000000 bash scripts/vSix5_guardian.sh           # shorter rollout
#
# Clean stop: touch runs/vSix5.stop (the trainer finishes its update and saves).
set -uo pipefail
cd /workspace/plodbnet || exit 1

ROLLOUT_LENGTH=${ROLLOUT_LENGTH:-60000000}
NUM_ENVS=${NUM_ENVS:-880000}
MICRO_ROWS=${MICRO_ROWS:-200000}
GLOG=runs/vSix5_guardian.log
STOPFLAG=runs/vSix5.stop
LOG=runs/vSix5.log
MAX_RESTARTS=4          # crashes allowed within RESTART_WINDOW seconds
RESTART_WINDOW=21600    # 6 h
POLL=300
restart_times=()

log(){ echo "[vSix5-guardian $(date -u '+%m-%d %H:%M:%S')] $*" >> "$GLOG"; }
train_pid(){ pgrep -f "python -u scripts/[t]rain.py .*--checkpoint checkpoints/vSix5.pt" | head -1; }
other_trainer(){ pgrep -f "python -u scripts/[t]rain.py" | grep -v "^$(train_pid)\$" | head -1; }

numa_prefix(){
  command -v taskset >/dev/null 2>&1 || return 0
  local bus node
  bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -1 | tr 'A-F' 'a-f')
  node=$(cat "/sys/bus/pci/devices/${bus: -12}/numa_node" 2>/dev/null || echo -1)
  if [ "$node" -ge 0 ] 2>/dev/null && [ -r "/sys/devices/system/node/node${node}/cpulist" ]; then
    echo "taskset -c $(cat "/sys/devices/system/node/node${node}/cpulist")"
  fi
}

launch(){
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PATH="$HOME/.cargo/bin:$PATH"
  export PLO5_RUST_ENCODER=1
  export PLO5BP_OBS_REV=1
  export PLO5BP_STEP_TIMERS=1
  export NUMPY_MADVISE_HUGEPAGE=0
  export MALLOC_MMAP_THRESHOLD_=33554432
  export MALLOC_TRIM_THRESHOLD_=17179869184
  export MALLOC_TOP_PAD_=67108864
  local load=""
  [ -n "${1:-}" ] && load="--load-checkpoint $1"
  local numa
  numa=$(numa_prefix)
  log "launch: ${ROLLOUT_LENGTH} rows, ${NUM_ENVS} envs, micro ${MICRO_ROWS}, ${load:-cold}, ${numa:-unpinned}"
  setsid nohup $numa .venv/bin/python -u scripts/train.py \
    --variant plo5_double_bomb \
    --v6 \
    --hidden-dim 2048 --num-layers 4 \
    --critic-hidden-dim 1536 --critic-num-blocks 2 \
    --batched --device cuda \
    --num-envs "$NUM_ENVS" --rollout-length "$ROLLOUT_LENGTH" \
    --batch-on-host --micro-batch-rows "$MICRO_ROWS" \
    --num-minibatches 16 --ppo-epochs 2 \
    --mix-configs --configs-per-tier 10 --mix-tiers clubgg,clubgg_deep,deep \
    --entropy-coef 0.16 --sizing-entropy-scale 1.0 \
    --lr 1.5e-4 --lr-warmup-updates 0 --clip-room-mid 0.07 \
    --target-kl 0.5 --kl-hard 10.0 --adv-clip 8 --cpu-threads 24 \
    --snapshot-every 5 --checkpoint-every 1 \
    $load --checkpoint checkpoints/vSix5.pt \
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
ok = (int(cfg.get("hidden_dim") or 0) == 2048 and int(cfg.get("num_layers") or 0) == 4
      and int(cfg.get("critic_hidden_dim") or 0) == 1536
      and str(cfg.get("obs_mode") or "full") == "full" and int(ck.get("obs_rev") or 1) == 1)
sys.exit(0 if ok else 1)
PY
}

pick_warm(){
  # The rolling vSix5.pt counts too: the first update after any (re)launch
  # writes no numbered checkpoint (train.py numbers from the 2nd update on).
  local f
  for f in $(ls -t checkpoints/vSix5_*.pt checkpoints/vSix5.pt 2>/dev/null); do
    compatible_ckpt "$f" && { echo "$f"; return 0; }
  done
  [ -n "${WARM:-}" ] && compatible_ckpt "$WARM" && { echo "$WARM"; return 0; }
  return 1
}

if [ -f "$STOPFLAG" ]; then
  log "stop flag present at start -> not launching (rm $STOPFLAG to run)"
  echo "vSix5 guardian: $STOPFLAG exists -> not launching (remove it to run)" >&2
  exit 0
fi
if [ -n "$(other_trainer)" ]; then
  log "another scripts/train.py is running -> not launching (vSix5 needs the pod's memory to itself)"
  echo "vSix5 guardian: stop the other trainer first (e.g. touch runs/vMin3.stop)" >&2
  exit 1
fi

if [ -z "$(train_pid)" ]; then
  WARM_CKPT=$(pick_warm) || WARM_CKPT=""
  if [ -z "$WARM_CKPT" ]; then
    log "no compatible checkpoint (set WARM=checkpoints/vSix4_1240.pt) -> not launching"
    echo "vSix5 guardian: no warm start found (WARM=checkpoints/vSix4_1240.pt)" >&2
    exit 1
  fi
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
      now=$(date +%s); keep=()
      for t in ${restart_times[@]+"${restart_times[@]}"}; do
        [ $((now - t)) -lt "$RESTART_WINDOW" ] && keep+=("$t")
      done
      restart_times=(${keep[@]+"${keep[@]}"})
      if [ "${#restart_times[@]}" -ge "$MAX_RESTARTS" ]; then
        log "DEAD; $MAX_RESTARTS restarts within $((RESTART_WINDOW / 3600)) h -> STOP"; touch "$STOPFLAG"; exit 0
      fi
      restart_times+=("$now")
      WARM_CKPT=$(pick_warm) || WARM_CKPT=""
      log "process DEAD; relaunch #${#restart_times[@]} in $((RESTART_WINDOW / 3600)) h warm=${WARM_CKPT:-NONE}"
      [ -n "$WARM_CKPT" ] && launch "$WARM_CKPT"; sleep 45
    fi
  fi
done
