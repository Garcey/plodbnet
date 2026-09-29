#!/bin/bash
# vFour3 guardian — corrected warm-restart of V4.
#
# vFour collapsed at u36 (gate-side, as the LR ramp + loose target_kl=2.0 let
# the gate over-move). vFour2 was meant to fix it but I DISABLED the LR warmup
# (--lr-warmup-updates 0) on a wrong hunch that the ramp was the stressor — it
# is the PROTECTION: warm-loading resets Adam's moments, so the first steps at
# full LR are huge without a ramp. vFour2 blew up at minibatch 1 (kl=9.72) and
# the gate was gone by u2.
#
# vFour3 = vFour2 with the warmup RE-ENABLED (the actual fix). The loop counter
# starts at 0 (anneal off), so warmup runs fresh from x0.013 and eases the
# reset Adam in. Config:
#   - warm-load vFour_30 (last clean pre-turbulence snapshot)
#   - lr 1.5e-4 with --lr-warmup-updates 75 (gentle fresh ramp; below v1/v2's
#     proven-stable 2e-4)
#   - target_kl 0.5 (was 2.0 in vFour — the likely real cause of the u36 gate
#     over-move)
# Everything else identical. If THIS collapses for a real (non-Adam) reason,
# the next lever is lr 1e-4.
#
# Stop cleanly: touch runs/vFour3.stop
set -uo pipefail
cd /workspace/plodbnet || exit 1

GLOG=runs/vFour3_guardian.log
STOPFLAG=runs/vFour3.stop
LOG=runs/vFour3.log
COLLFLAG=runs/vFour3_collapsed.flag
SEED_CKPT=checkpoints/vFour_30.pt
MAX_RESTARTS=4
POLL=300
restarts=0

log(){ echo "[vFour3-guardian $(date -u '+%m-%d %H:%M:%S')] $*" >> "$GLOG"; }
train_pid(){ pgrep -f "python -u scripts/[t]rain.py" | head -1; }

launch(){  # $1 = checkpoint to warm-load
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
    --lr 1.5e-4 --lr-warmup-updates 75 --target-kl 0.5 --kl-hard 10.0 --adv-clip 8 \
    --snapshot-every 5 \
    $load --checkpoint checkpoints/vFour3.pt \
    --num-updates 100000000 >> "$LOG" 2>&1 < /dev/null &
  disown 2>/dev/null || true
}

if [ -z "$(train_pid)" ]; then
  L=$(ls -t checkpoints/vFour3_*.pt 2>/dev/null | head -1)
  WARM="${L:-$SEED_CKPT}"
  log "initial launch warm-loading ${WARM}"
  launch "$WARM"; sleep 45
fi

log "started; watching pid=$(train_pid) (lr=1.5e-4 warmup=75, target_kl=0.5, max_restarts=$MAX_RESTARTS, poll=${POLL}s)"
while true; do
  [ -f "$STOPFLAG" ] && { log "stop flag present -> exiting"; exit 0; }
  sleep "$POLL"
  PID=$(train_pid)
  NUPD=$(grep -cE "update +[0-9]" "$LOG" 2>/dev/null || echo 0)
  LASTHG=$(grep -E "update +[0-9]" "$LOG" 2>/dev/null | tail -1 | grep -oE "Hg/Ha/Hb=[0-9.]+" | head -1 | sed 's#.*=##')
  COLL=$(grep -E "update +[0-9]" "$LOG" 2>/dev/null | tail -8 | grep -oE "Hg/Ha/Hb=[0-9.]+" | sed 's#.*=##' | awk '{if($1+0<0.15)c++} END{print c+0}')

  if [ -z "$PID" ]; then
    sleep 15; PID=$(train_pid)
    if [ -z "$PID" ]; then
      if [ "$restarts" -ge "$MAX_RESTARTS" ]; then log "DEAD; restart cap ($MAX_RESTARTS) hit -> STOP"; touch "$STOPFLAG"; exit 0; fi
      restarts=$((restarts+1))
      L=$(ls -t checkpoints/vFour3_*.pt 2>/dev/null | head -1)
      WARM="${L:-$SEED_CKPT}"
      log "process DEAD (crash/OOM); relaunch #$restarts warm-loading ${WARM}"
      launch "$WARM"; sleep 45; continue
    fi
  fi

  if [ "${NUPD:-0}" -ge 10 ] && [ "${COLL:-0}" -ge 5 ]; then
    GP=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$GP" ]; then kill "$GP" 2>/dev/null; sleep 6; kill -9 "$GP" 2>/dev/null; fi
    log "SUSTAINED COLLAPSE (Hg=${LASTHG:-?}, $COLL/8 <0.15) -> run STOPPED, no auto-restart. flag=$COLLFLAG"
    date -u '+%Y-%m-%d %H:%M:%S' > "$COLLFLAG"
    touch "$STOPFLAG"; exit 0
  fi

  log "ok: pid=$PID updates=$NUPD Hg=${LASTHG:-none} collapsed_last8=${COLL:-0} restarts=$restarts"
done
