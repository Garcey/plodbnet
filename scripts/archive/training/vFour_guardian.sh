#!/bin/bash
# vFour guardian — runs ON THE POD, survives the Claude app closing.
#
# V4 = the ordinal discretized-logistic sizing head (the anti-collapse fix), a
# COLD-START run on the 30-config mix-configs regime. Recovers from:
#  (a) process DEATH (crash/OOM) -> relaunch from the latest vFour_*.pt ckpt, or
#      COLD if none yet (bounded restarts).
#  (b) SUSTAINED gate collapse (>=5 of last 8 updates Hg<0.15) -> STOP the run +
#      write runs/collapsed.flag, NO auto-restart. A V4 collapse would be a
#      SURPRISING result to investigate (the head is meant to prevent it), and
#      blind restarts misfired historically.
# NOTE: the V4-specific early failure is the ANCHOR entropy (Ha) crashing while
# the gate lags — the agent's watcher trips on Ha; this Hg guard is the backstop.
#
# Stop cleanly:  touch /workspace/plodbnet/runs/guardian.stop
set -uo pipefail
cd /workspace/plodbnet || exit 1

GLOG=runs/guardian.log
STOPFLAG=runs/guardian.stop
LOG=runs/vFour.log
MAX_RESTARTS=4
POLL=300
restarts=0

log(){ echo "[guardian $(date -u '+%m-%d %H:%M:%S')] $*" >> "$GLOG"; }
train_pid(){ pgrep -f "python -u scripts/train.py" | head -1; }

launch(){  # $1 = checkpoint to warm-load, or "" for a cold start
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PATH="$HOME/.cargo/bin:$PATH"
  local load=""
  [ -n "${1:-}" ] && load="--load-checkpoint $1"
  setsid nohup .venv/bin/python -u scripts/train.py \
    --batched --device cuda --hidden-dim 2048 --num-layers 4 \
    --critic-hidden-dim 1536 --critic-num-blocks 2 \
    --num-envs 49134 --rollout-length 10000000 --num-minibatches 16 --ppo-epochs 2 \
    --mix-configs --configs-per-tier 10 --mix-tiers clubgg,clubgg_deep,deep \
    --sizing-head logistic \
    --entropy-coef 0.3 --sizing-entropy-scale 1.0 \
    --target-kl 2.0 --kl-hard 10.0 --adv-clip 8 --lr-warmup-updates 75 \
    --snapshot-every 5 \
    $load --checkpoint checkpoints/vFour.pt \
    --num-updates 100000000 >> "$LOG" 2>&1 < /dev/null &
  disown 2>/dev/null || true
}

log "started; watching pid=$(train_pid) (max_restarts=$MAX_RESTARTS, poll=${POLL}s)"
while true; do
  [ -f "$STOPFLAG" ] && { log "stop flag present -> exiting"; exit 0; }
  sleep "$POLL"
  PID=$(train_pid)
  NUPD=$(grep -cE "update +[0-9]" "$LOG" 2>/dev/null || echo 0)
  LASTHG=$(grep -E "update +[0-9]" "$LOG" 2>/dev/null | tail -1 | grep -oE "Hg/Ha/Hb=[0-9.]+" | head -1 | sed 's#.*=##')
  COLL=$(grep -E "update +[0-9]" "$LOG" 2>/dev/null | tail -8 | grep -oE "Hg/Ha/Hb=[0-9.]+" | sed 's#.*=##' | awk '{if($1+0<0.15)c++} END{print c+0}')

  # --- process death (crash / OOM) -------------------------------------------
  if [ -z "$PID" ]; then
    sleep 15; PID=$(train_pid)
    if [ -z "$PID" ]; then
      if [ "$restarts" -ge "$MAX_RESTARTS" ]; then log "DEAD; restart cap ($MAX_RESTARTS) hit -> STOP"; touch "$STOPFLAG"; exit 0; fi
      restarts=$((restarts+1))
      L=$(ls -t checkpoints/vFour_*.pt 2>/dev/null | head -1)
      log "process DEAD (crash/OOM); relaunch #$restarts from ${L:-COLD}"
      launch "${L:-}"; sleep 45; continue
    fi
  fi

  # --- sustained collapse: STOP + FLAG (no auto-restart) ---------------------
  if [ "${NUPD:-0}" -ge 10 ] && [ "${COLL:-0}" -ge 5 ]; then
    GP=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$GP" ]; then kill "$GP" 2>/dev/null; sleep 6; kill -9 "$GP" 2>/dev/null; fi
    log "SUSTAINED COLLAPSE (Hg=${LASTHG:-?}, $COLL/8 <0.15) -> run STOPPED, no auto-restart. flag=runs/collapsed.flag"
    date -u '+%Y-%m-%d %H:%M:%S' > runs/collapsed.flag
    touch "$STOPFLAG"; exit 0
  fi

  log "ok: pid=$PID updates=$NUPD Hg=${LASTHG:-none} collapsed_last8=${COLL:-0} restarts=$restarts"
done
