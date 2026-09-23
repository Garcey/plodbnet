#!/bin/bash
# Network-size sweep guardian (2026-09-23). One training run of the vMin2
# recipe (minimal obs, v6 preset, 220k envs, 44M-row rollout, 30 mixed
# configs, entropy 0.25, lr 1.5e-4) with ONLY the network size changed, run
# to a fixed number of global updates so every size is compared at the same
# training budget. Several sweep runs (and vMin2) share the pod: every run
# passes --gpu-lock runs/gpu_ppo.lock, so only one at a time holds a batch +
# PPO working set on the GPU (rollouts, which use little GPU, overlap).
#
#   [NUMA_NODE=n] bash scripts/sweep_guardian.sh STEM ACTOR_HD ACTOR_LAYERS CRITIC_HD CRITIC_BLOCKS TARGET_UPDATES
#   e.g. NUMA_NODE=1 bash scripts/sweep_guardian.sh sw64 64 3 64 2 60
#
# Resumes from the newest <STEM>_*.pt (same sizes only) and exits once
# TARGET_UPDATES updates exist. Clean stop: touch runs/<STEM>.stop
set -uo pipefail
cd /workspace/plodbnet || exit 1

STEM=$1; HD=$2; NL=$3; CHD=$4; CNB=$5; TARGET=$6
GLOG=runs/${STEM}_guardian.log
STOPFLAG=runs/${STEM}.stop
LOG=runs/${STEM}.log
MAX_RESTARTS=4
POLL=120
restarts=0

log(){ echo "[${STEM}-guardian $(date -u '+%m-%d %H:%M:%S')] $*" >> "$GLOG"; }
train_pid(){ pgrep -f "python -u scripts/[t]rain.py .*--checkpoint checkpoints/${STEM}.pt" | head -1; }

# Same placement rule as vMin2_guardian.sh (see the comment there), unless
# NUMA_NODE=<n> pins this run to node n -- concurrent sweep runs each get
# their own socket (their own cores, cache and memory bandwidth).
numa_prefix(){
  command -v taskset >/dev/null 2>&1 || return 0
  if [ -n "${NUMA_NODE:-}" ] && [ -r "/sys/devices/system/node/node${NUMA_NODE}/cpulist" ]; then
    echo "taskset -c $(cat /sys/devices/system/node/node${NUMA_NODE}/cpulist)"; return 0
  fi
  local bus gpu_node d n free best="" bestfree=0 need=$((48 * 1024 * 1024))
  bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -1 | tr 'A-F' 'a-f')
  gpu_node=$(cat "/sys/bus/pci/devices/${bus: -12}/numa_node" 2>/dev/null || echo -1)
  for d in /sys/devices/system/node/node[0-9]*; do
    [ -r "$d/cpulist" ] || continue
    n=${d##*node}
    free=$(awk '/MemFree/ {print $4}' "$d/meminfo")
    if [ "$n" = "$gpu_node" ] && [ "$free" -ge "$need" ]; then
      echo "taskset -c $(cat "$d/cpulist")"; return 0
    fi
    if [ "$free" -gt "$bestfree" ]; then best=$d; bestfree=$free; fi
  done
  [ -n "$best" ] && echo "taskset -c $(cat "$best/cpulist")"
}

# Updates already trained = 1 + the highest numbered checkpoint (0 if none).
done_updates(){
  local f n best=-1
  for f in checkpoints/${STEM}_*.pt; do
    [ -f "$f" ] || continue
    n=${f##*_}; n=${n%.pt}
    [[ "$n" =~ ^[0-9]+$ ]] || continue
    [ "$n" -gt "$best" ] && best=$n
  done
  echo $((best + 1))
}

compatible_ckpt(){
  [ -f "$1" ] || return 1
  .venv/bin/python - "$1" "$HD" "$NL" "$CHD" <<'PY'
import sys, torch
p, hd, nl, chd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
try:
    ck = torch.load(p, map_location="cpu", weights_only=False)
except Exception:
    sys.exit(1)
cfg = ck.get("config") or {}
ok = (int(cfg.get("hidden_dim") or 0) == hd and int(cfg.get("num_layers") or 0) == nl
      and int(cfg.get("critic_hidden_dim") or 0) == chd)
sys.exit(0 if ok else 1)
PY
}

pick_warm(){
  local f
  for f in $(ls -t checkpoints/${STEM}_*.pt 2>/dev/null); do
    compatible_ckpt "$f" && { echo "$f"; return 0; }
  done
  return 1
}

launch(){
  local numa remaining load=""
  remaining=$(( TARGET - $(done_updates) ))
  if [ "$remaining" -le 0 ]; then
    log "target of $TARGET updates reached -> done"; touch "$STOPFLAG"; return 1
  fi
  numa=$(numa_prefix)
  log "numa placement: ${numa:-none}; ${remaining} updates to go"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PATH="$HOME/.cargo/bin:$PATH"
  export PLO5_RUST_ENCODER=1
  export PLO5BP_STEP_TIMERS=1
  export NUMPY_MADVISE_HUGEPAGE=0
  export MALLOC_MMAP_THRESHOLD_=33554432
  export MALLOC_TRIM_THRESHOLD_=17179869184
  export MALLOC_TOP_PAD_=67108864
  [ -n "${1:-}" ] && load="--load-checkpoint $1"
  setsid nohup $numa .venv/bin/python -u scripts/train.py \
    --variant plo5_double_bomb \
    --v6 \
    --obs-mode minimal \
    --hidden-dim "$HD" --num-layers "$NL" \
    --critic-hidden-dim "$CHD" --critic-num-blocks "$CNB" \
    --batched --device cuda \
    --num-envs 220000 --rollout-length 44000000 --num-minibatches 16 --ppo-epochs 2 \
    --mix-configs --configs-per-tier 10 --mix-tiers clubgg,clubgg_deep,deep \
    --entropy-coef 0.25 --sizing-entropy-scale 1.0 \
    --lr 1.5e-4 --lr-warmup-updates 0 --clip-room-mid 0.07 \
    --target-kl 0.5 --kl-hard 10.0 --adv-clip 8 --cpu-threads 24 \
    --snapshot-every 5 --checkpoint-every 1 \
    --gpu-lock runs/gpu_ppo.lock \
    $load --checkpoint "checkpoints/${STEM}.pt" \
    --num-updates "$remaining" >> "$LOG" 2>&1 < /dev/null &
  disown 2>/dev/null || true
}

if [ -f "$STOPFLAG" ]; then
  log "stop flag present at start -> not launching (rm $STOPFLAG to run)"
  echo "$STEM guardian: $STOPFLAG exists -> not launching" >&2
  exit 0
fi

if [ -z "$(train_pid)" ]; then
  if WARM=$(pick_warm); then
    log "initial launch warm-loading ${WARM} (${HD}x${NL} / critic ${CHD}x${CNB})"
  else
    WARM=""
    log "initial launch COLD (${HD}x${NL} / critic ${CHD}x${CNB}, target ${TARGET} updates)"
  fi
  launch "$WARM" || exit 0
  sleep 45
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
      if [ "$(done_updates)" -ge "$TARGET" ]; then
        log "process finished: $(done_updates)/${TARGET} updates -> done"; touch "$STOPFLAG"; exit 0
      fi
      if [ "$restarts" -ge "$MAX_RESTARTS" ]; then log "DEAD; restart cap hit -> STOP"; touch "$STOPFLAG"; exit 0; fi
      restarts=$((restarts+1))
      WARM=$(pick_warm) || WARM=""
      log "process DEAD at $(done_updates) updates; relaunch #$restarts warm=${WARM:-COLD}"
      launch "$WARM" || exit 0
      sleep 45
    fi
  fi
done
