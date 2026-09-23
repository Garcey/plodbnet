#!/bin/bash
# vMin2 guardian — fresh minimal-obs stem (2026-09-23). The vMin1 recipe (v6
# preset, --obs-mode minimal, actor 128x3 / critic 128x2, 30 mixed configs)
# cold-started with:
#   - the CORRECTED observation values (rev 2 — no PLO5BP_OBS_REV pin; vMin1's
#     pre-fix checkpoints are gone, so nothing needs the old values),
#   - compact observation storage + batched opponents (be910f9),
#   - in-flight hands drained (the new-stem default; no --no-drain-inflight),
#   - the rollout doubled to 44M rows (owner: longer rollouts are always
#     better; 22M peaked at 30 GiB of the RTX PRO 6000's 96),
#   - a numbered checkpoint EVERY update (small net = cheap; a restart loses
#     at most one update).
# PLO5BP_STEP_TIMERS=1 prints the per-step timer table each update — kept on
# while the intermittent rollout stall seen on the first RunPod host is being
# fixed. NUMPY_MADVISE_HUGEPAGE=0: numpy stops requesting transparent huge
# pages (that host's memory was badly fragmented; the kernel can stall an
# allocation to compact it).
# Does NOT touch other stems. Clean stop: touch runs/vMin2.stop
set -uo pipefail
cd /workspace/plodbnet || exit 1

GLOG=runs/vMin2_guardian.log
STOPFLAG=runs/vMin2.stop
LOG=runs/vMin2.log
MAX_RESTARTS=4
POLL=300
restarts=0

log(){ echo "[vMin2-guardian $(date -u '+%m-%d %H:%M:%S')] $*" >> "$GLOG"; }
# Match only THIS stem's train.py.
train_pid(){ pgrep -f "python -u scripts/[t]rain.py .*--checkpoint checkpoints/vMin2.pt" | head -1; }

# NUMA placement: run the trainer on ONE socket's CPUs. The first RunPod host
# had automatic NUMA balancing on and it migrated 4M of the trainer's pages
# (16 GB) between sockets in 40 min — the Rust (rayon) worker threads spent
# ~half their CPU in the kernel taking those faults. With every thread on one
# node, first-touch puts the memory there and nothing is ever accessed
# remotely, so there is nothing to migrate. CPU affinity only (taskset): the
# container's seccomp profile refuses set_mempolicy, so `numactl --preferred`
# fails there. Node = the GPU's (fastest host<->GPU copies) when it has
# >= 48 GB free, else the node with the most free memory; no NUMA info ->
# unpinned, as before.
numa_prefix(){
  command -v taskset >/dev/null 2>&1 || return 0
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

launch(){
  local numa
  numa=$(numa_prefix)
  log "numa placement: ${numa:-none (no NUMA info / taskset)}"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PATH="$HOME/.cargo/bin:$PATH"
  export PLO5_RUST_ENCODER=1
  export PLO5BP_STEP_TIMERS=1
  export NUMPY_MADVISE_HUGEPAGE=0
  # glibc malloc: keep freed memory for reuse instead of handing it back to
  # the kernel and faulting it in again on the next step (the per-step numpy
  # temporaries cost ~6M page faults per update). Fixed mmap threshold (32 MB,
  # glibc's max) = no dynamic threshold; never trim; grow in 64 MB steps.
  # Allocation policy only -- no effect on any computed value.
  export MALLOC_MMAP_THRESHOLD_=33554432
  export MALLOC_TRIM_THRESHOLD_=17179869184
  export MALLOC_TOP_PAD_=67108864
  local load=""
  [ -n "${1:-}" ] && load="--load-checkpoint $1"
  setsid nohup $numa .venv/bin/python -u scripts/train.py \
    --variant plo5_double_bomb \
    --v6 \
    --obs-mode minimal \
    --hidden-dim 128 --num-layers 3 \
    --critic-hidden-dim 128 --critic-num-blocks 2 \
    --batched --device cuda \
    --num-envs 220000 --rollout-length 44000000 --num-minibatches 16 --ppo-epochs 2 \
    --mix-configs --configs-per-tier 10 --mix-tiers clubgg,clubgg_deep,deep \
    --entropy-coef 0.25 --sizing-entropy-scale 1.0 \
    --lr 1.5e-4 --lr-warmup-updates 0 --clip-room-mid 0.07 \
    --target-kl 0.5 --kl-hard 10.0 --adv-clip 8 --cpu-threads 24 \
    --snapshot-every 5 --checkpoint-every 1 \
    --gpu-lock runs/gpu_ppo.lock \
    $load --checkpoint checkpoints/vMin2.pt \
    --num-updates 100000000 >> "$LOG" 2>&1 < /dev/null &
  disown 2>/dev/null || true
}

# Only warm-start from checkpoints that match the 128-wide architecture.
compatible_ckpt(){
  local f="$1"
  [ -f "$f" ] || return 1
  .venv/bin/python - "$f" <<'PY'
import sys, torch
p = sys.argv[1]
try:
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
except Exception:
    sys.exit(1)
cfg = ckpt.get("config") or {}
hd = int(cfg.get("hidden_dim") or 0)
nl = int(cfg.get("num_layers") or 0)
ch = int(cfg.get("critic_hidden_dim") or 0)
if hd <= 0:
    w = ckpt.get("model", {}).get("torso.0.0.weight")
    if w is not None:
        hd = int(w.shape[0])
if ch <= 0:
    w = (ckpt.get("critic") or {}).get("torso.0.0.weight")
    if w is not None:
        ch = int(w.shape[0])
ok = (hd == 128 and (nl in (0, 3)) and ch in (0, 128))
sys.exit(0 if ok else 1)
PY
}

pick_warm(){
  local f
  for f in $(ls -t checkpoints/vMin2_*.pt 2>/dev/null); do
    if compatible_ckpt "$f"; then
      echo "$f"
      return 0
    fi
  done
  if [ -f checkpoints/vMin2.pt ] && compatible_ckpt checkpoints/vMin2.pt; then
    echo "checkpoints/vMin2.pt"
    return 0
  fi
  return 1
}

# A stop flag left over from the last clean stop must be cleared by hand:
# launching and THEN exiting on it would leave a trainer with no guardian.
if [ -f "$STOPFLAG" ]; then
  log "stop flag present at start -> not launching (rm $STOPFLAG to run)"
  echo "vMin2 guardian: $STOPFLAG exists -> not launching (remove it to run)" >&2
  exit 0
fi

# train.py's rolling optimizer sidecar is checkpoints/vMin2.optim.pt —
# deliberately NOT matched by pick_warm's vMin2_*.pt glob. Checkpoints are
# written atomically (<name>.pt.tmp + rename), so every match is complete.
if [ -z "$(train_pid)" ]; then
  if WARM=$(pick_warm); then
    log "initial launch warm-loading ${WARM} (128-wide)"
  else
    WARM=""
    log "initial launch COLD (minimal obs rev 2 + 128x3/128x2 + 44M/220k)"
  fi
  launch "$WARM"; sleep 45
fi

log "started; watching pid=$(train_pid) (vMin2 = minimal-obs 796 + 128x3/128x2 + 220k envs + 44M rollout)"
while true; do
  [ -f "$STOPFLAG" ] && { log "stop flag present -> exiting"; exit 0; }
  sleep "$POLL"
  PID=$(train_pid)
  if [ -z "$PID" ]; then
    sleep 15; PID=$(train_pid)
    if [ -z "$PID" ]; then
      if [ "$restarts" -ge "$MAX_RESTARTS" ]; then log "DEAD; restart cap hit -> STOP"; touch "$STOPFLAG"; exit 0; fi
      # Re-check the stop flag RIGHT before relaunching: a clean stop (touch
      # the flag, kill the trainer) usually lands inside the POLL sleep above.
      [ -f "$STOPFLAG" ] && { log "process DEAD and stop flag present -> NOT relaunching; exiting"; exit 0; }
      restarts=$((restarts+1))
      if WARM=$(pick_warm); then
        log "process DEAD; relaunch #$restarts warm=${WARM}"
      else
        WARM=""
        log "process DEAD; relaunch #$restarts COLD"
      fi
      launch "$WARM"; sleep 45; continue
    fi
  fi
done
