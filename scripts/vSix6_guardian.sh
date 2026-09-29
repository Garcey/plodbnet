#!/bin/bash
# vSix6 guardian (2026-09-26/27) — the redesigned recipe after the regression
# diagnosis (docs/training-log.md "Regression diagnosis + redesign"). vSix5 had converged:
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
#     round-2 size study decides — see docs/training-log.md;
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
# 2026-09-28 (docs/training-log.md "Plateau check + recipe rounds 3-5"): the run had stalled
# since ~u1340 at entropy 0.06 / lambda 0.95 / lr 1.5e-4 / 10 setups per tier. Its
# updates are noise-dominated: bigger or faster steps hurt, and every noise cut
# helped (sampled / argmax vs the u1390 start after 10 search-scale updates, old
# recipe -0.001 / +0.034): entropy 0.045 +0.057 / +0.017, + lambda 0.8 +0.103 /
# +0.046, + lr 7.5e-5 +0.085 / +0.047, both +0.117 / +0.051, and with 30 setups
# per tier +0.119 / +0.072 (setups alone did nothing: once the other noise is cut,
# WHICH setups an update drew dominates). 90 setups cost ~20% rollout time here.
ENTROPY=${ENTROPY:-0.045}
LAMBDA=${LAMBDA:-0.8}
LR=${LR:-7.5e-5}
CONFIGS_PER_TIER=${CONFIGS_PER_TIER:-30}
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
# A trainer whose heartbeat (runs/vSix6.heartbeat, written after every update
# since 2026-09-28) is older than this is HUNG: killed, then relaunched like a crash.
STALE_SECS=${STALE_SECS:-10800}
restart_times=()

STEM=vSix6
PY=.venv/bin/python
# Ships together with scripts/guardian_lib.sh and python/plo5bp/train/ (one pull):
# the lib refuses to run without its helper module.
# shellcheck source=scripts/guardian_lib.sh
. scripts/guardian_lib.sh

log(){ echo "[vSix6-guardian $(date -u '+%m-%d %H:%M:%S')] $*" >> "$GLOG"; }
train_pid(){ pgrep -f "python -u scripts/[t]rain.py .*--checkpoint checkpoints/vSix6.pt" | head -1; }
other_trainer(){ pgrep -f "python -u scripts/[t]rain.py" | grep -v "^$(train_pid)\$" | head -1; }

numa_prefix(){ gl_numa_prefix; }

launch(){
  # This stem's own Inductor autotune cache (a cache shared with evaluators or
  # sweeps can change the compiled kernels' numerics between relaunches).
  export TORCHINDUCTOR_CACHE_DIR="$(gl_inductor_cache)"
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
  log "launch: ${ACTOR_HD}x${ACTOR_NL}, ${ROLLOUT_LENGTH} rows, ${NUM_ENVS} envs, micro ${MICRO_ROWS}, entropy ${ENTROPY}, lambda ${LAMBDA}, lr ${LR}, ${CONFIGS_PER_TIER} setups/tier, ${load:-cold} ${first}, ${numa:-unpinned}"
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
    --mix-configs --configs-per-tier "$CONFIGS_PER_TIER" --mix-tiers clubgg,clubgg_deep,deep \
    --entropy-coef "$ENTROPY" --sizing-entropy-scale 0.3 --gae-lambda "$LAMBDA" \
    --lr "$LR" --lr-warmup-updates 0 --clip-room-mid 0.07 \
    --target-kl 0.5 --kl-hard 10.0 --adv-clip 8 --cpu-threads 24 \
    --snapshot-every 5 --checkpoint-every 1 \
    $load $first --checkpoint checkpoints/vSix6.pt \
    --num-updates 100000000 >> "$LOG" 2>&1 < /dev/null &
  disown 2>/dev/null || true
}

pick_warm(){
  # The HIGHEST-numbered compatible vSix6_<N>.pt -- or the rolling vSix6.pt when
  # its counter says it is newer (the first update after any (re)launch writes
  # no numbered checkpoint) -- else $WARM. By update number, not modification
  # time (plo5bp/train/guardian.py). The critic requirements apply to vSix6's
  # own files only: a launch from another stem installs CRITIC_INIT.
  gl_pick_warm --need hidden_dim="$ACTOR_HD" --need num_layers="$ACTOR_NL" \
    --need obs_mode=full --need obs_rev=1 \
    --need-own critic_hidden_dim=1536 --need-own critic_act=silu \
    ${WARM:+--also "$WARM"}
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
  # Alive but hung (heartbeat older than STALE_SECS): kill it, relaunch below.
  [ -n "$PID" ] && gl_check_heartbeat "$PID" && PID=""
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
