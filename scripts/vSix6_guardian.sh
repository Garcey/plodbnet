#!/bin/bash
# vSix6 guardian (2026-09-26/27) — the redesigned recipe after the regression
# diagnosis (CLAUDE.md "Regression diagnosis + redesign"). vSix5 had converged:
# its updates were ~87% noise around a fixed point, its critic read values ~43%
# low (a log-space average) with 77% of its input layer dead, and every visible
# gain since u940 came from lowering entropy. vSix6 changes the fixed point and
# the signal, not only the step count:
#   - a NEW critic (1536x2): SiLU, LayerNorm'd input block, V read as the
#     raw-space mean of its return distribution, the fold value pinned to 0 and
#     the Q losses normalized by the return variance (without both a fresh
#     critic's value estimates collapse online), 1 extra critic-only epoch of 128
#     small minibatches per rollout (the PPO epochs give it only 16 huge steps);
#     started from a critic trained offline on u1290's rollouts (CRITIC_INIT);
#     no gradient checkpointing (micro-batching bounds the memory; -25% compute);
#   - entropy 0.06 and sizing-entropy scale 0.3 (round 1: sampled play +0.36
#     over 0.10 within two updates, argmax kept; 0.03 hurt argmax);
#   - ACTOR_SIZE (default 1024x3): a student distilled from u1290 holds the
#     2048x4 policy (held-out gate KL .0034 vs the same-size copy's .0031); the
#     round-2 size study decides — see CLAUDE.md;
#   - pipeline: rollouts of ROLLOUT_LENGTH rows with the batch in host RAM as
#     float16 real columns (--obs-real-f16: ~1.1 KB per row instead of ~2.1),
#     micro-batched PPO, the engine's shared MC board tables and packed-only
#     full-layout encoding, CPUs pinned to the GPU's NUMA node; NUM_ENVS from
#     the throughput probe.
# Obs rev 1 (PLO5BP_OBS_REV=1) like the live site, so a checkpoint is a drop-in
# promotion (the site reads the actor's size from the checkpoint; its old code
# cannot rebuild the new critic, which only disables the review's "true EV").
#
#   WARM=checkpoints/r2b_1299.pt bash scripts/vSix6_guardian.sh     # first launch
#   ROLLOUT_LENGTH=120000000 NUM_ENVS=880000 bash scripts/vSix6_guardian.sh
#
# Clean stop: touch runs/vSix6.stop (the trainer finishes its update and saves).
set -uo pipefail
cd /workspace/plodbnet || exit 1

ACTOR_HD=${ACTOR_HD:-1024}
ACTOR_NL=${ACTOR_NL:-3}
ROLLOUT_LENGTH=${ROLLOUT_LENGTH:-150000000}
NUM_ENVS=${NUM_ENVS:-880000}
MICRO_ROWS=${MICRO_ROWS:-500000}
CRITIC_INIT=${CRITIC_INIT:-checkpoints/critic_silu1536x2_u1290.pt}
GLOG=runs/vSix6_guardian.log
STOPFLAG=runs/vSix6.stop
LOG=runs/vSix6.log
MAX_RESTARTS=4          # crashes allowed within RESTART_WINDOW seconds
RESTART_WINDOW=21600    # 6 h
POLL=300
restart_times=()

log(){ echo "[vSix6-guardian $(date -u '+%m-%d %H:%M:%S')] $*" >> "$GLOG"; }
train_pid(){ pgrep -f "python -u scripts/[t]rain.py .*--checkpoint checkpoints/vSix6.pt" | head -1; }
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
  export PLO5BP_ANNEAL_CONTROL=runs/vSix6.control.json   # its own live-control file
  export NUMPY_MADVISE_HUGEPAGE=0
  export MALLOC_MMAP_THRESHOLD_=33554432
  export MALLOC_TRIM_THRESHOLD_=17179869184
  export MALLOC_TOP_PAD_=67108864
  local load="" first=""
  [ -n "${1:-}" ] && load="--load-checkpoint $1"
  # A launch from a file of ANOTHER stem (the round-2 winner, whose critic has
  # another size): install the 1536x2 critic from CRITIC_INIT (trained offline
  # on u1290's rollouts; the actor's Adam moments still restore) and give it
  # one critic-only warm-up update. vSix6's own files resume normally.
  case "$(basename "${1:-}")" in
    vSix6*) ;;
    "") ;;
    *) first="--critic-init ${CRITIC_INIT} --actor-freeze-updates 1" ;;
  esac
  local numa
  numa=$(numa_prefix)
  log "launch: ${ACTOR_HD}x${ACTOR_NL}, ${ROLLOUT_LENGTH} rows, ${NUM_ENVS} envs, micro ${MICRO_ROWS}, ${load:-cold} ${first}, ${numa:-unpinned}"
  setsid nohup $numa .venv/bin/python -u scripts/train.py \
    --variant plo5_double_bomb \
    --v6 \
    --hidden-dim "$ACTOR_HD" --num-layers "$ACTOR_NL" \
    --critic-hidden-dim 1536 --critic-num-blocks 2 \
    --critic-act silu --critic-in-norm --critic-v-raw --q-base-raw --q-fold-zero --critic-q-norm \
    --critic-extra-epochs 1 --critic-minibatches 128 --no-grad-checkpoint \
    --batched --device cuda \
    --num-envs "$NUM_ENVS" --rollout-length "$ROLLOUT_LENGTH" \
    --batch-on-host --obs-real-f16 --micro-batch-rows "$MICRO_ROWS" \
    --num-minibatches 16 --ppo-epochs 2 \
    --mix-configs --configs-per-tier 10 --mix-tiers clubgg,clubgg_deep,deep \
    --entropy-coef 0.06 --sizing-entropy-scale 0.3 \
    --lr 1.5e-4 --lr-warmup-updates 0 --clip-room-mid 0.07 \
    --target-kl 0.5 --kl-hard 10.0 --adv-clip 8 --cpu-threads 24 \
    --snapshot-every 5 --checkpoint-every 1 \
    $load $first --checkpoint checkpoints/vSix6.pt \
    --num-updates 100000000 >> "$LOG" 2>&1 < /dev/null &
  disown 2>/dev/null || true
}

compatible_ckpt(){
  [ -f "$1" ] || return 1
  ACTOR_HD="$ACTOR_HD" ACTOR_NL="$ACTOR_NL" .venv/bin/python - "$1" <<'PY'
import os, sys, torch
try:
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
except Exception:
    sys.exit(1)
cfg = ck.get("config") or {}
ok = (int(cfg.get("hidden_dim") or 0) == int(os.environ["ACTOR_HD"])
      and int(cfg.get("num_layers") or 0) == int(os.environ["ACTOR_NL"])
      and (not os.path.basename(sys.argv[1]).startswith("vSix6")
           or (int(cfg.get("critic_hidden_dim") or 0) == 1536
               and str(cfg.get("critic_act") or "relu") == "silu"))
      and str(cfg.get("obs_mode") or "full") == "full" and int(ck.get("obs_rev") or 1) == 1)
sys.exit(0 if ok else 1)
PY
}

pick_warm(){
  # The rolling vSix6.pt counts too: the first update after any (re)launch
  # writes no numbered checkpoint (train.py numbers from the 2nd update on).
  local f
  for f in $(ls -t checkpoints/vSix6_*.pt checkpoints/vSix6.pt 2>/dev/null); do
    compatible_ckpt "$f" && { echo "$f"; return 0; }
  done
  [ -n "${WARM:-}" ] && compatible_ckpt "$WARM" && { echo "$WARM"; return 0; }
  return 1
}

if [ -f "$STOPFLAG" ]; then
  log "stop flag present at start -> not launching (rm $STOPFLAG to run)"
  echo "vSix6 guardian: $STOPFLAG exists -> not launching (remove it to run)" >&2
  exit 0
fi
if [ -n "$(other_trainer)" ]; then
  log "another scripts/train.py is running -> not launching (vSix6 needs the pod's memory to itself)"
  echo "vSix6 guardian: stop the other trainer first" >&2
  exit 1
fi

if [ -z "$(train_pid)" ]; then
  WARM_CKPT=$(pick_warm) || WARM_CKPT=""
  if [ -z "$WARM_CKPT" ]; then
    log "no compatible checkpoint (set WARM=<round-2 winner>) -> not launching"
    echo "vSix6 guardian: no warm start found (WARM=checkpoints/<round-2 winner>.pt)" >&2
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
