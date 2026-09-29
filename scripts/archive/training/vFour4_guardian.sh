#!/bin/bash
# vFour4 guardian — warm-restart after vFour3's gate collapse at u47.
#
# vFour3 (lr 1.5e-4 warmup, target_kl 0.5, entropy 0.30) held Hg ~0.78 through
# u43, then dipped u44-46 and collapsed at u47 (Hg 0.12) as the LR ramped to
# lr×0.64 (~0.96e-4). The tight guard FIRED every step (KLSTOP u45/46/47) but
# couldn't hold it — the per-minibatch gate moves before each trip compounded.
# Root cause: entropy-coef 0.30 was BELOW the floor. The proven-stable vTwo runs
# held at 0.45 (and at a HIGHER lr, 2e-4). As the LR climbed, the 0.30 bonus lost
# grip on the gate.
#
# vFour4 = vFour3 with **entropy-coef 0.30 -> 0.45** (the only deliberate change;
# the proven floor + the documented gentle-resume value). Warm-load the clean
# vFour3_40 (Hg 0.78, pre-dip). Also runs the b35c199 code fixes now on the pod
# (actor/critic grad-clip split; sizing legal-edge tail absorption; target_kl
# default 0.5). Keep lr 1.5e-4 + warmup 75 + target_kl 0.5.
# If THIS collapses, the next lever is lr 1e-4.
#
# Stop cleanly: touch runs/vFour4.stop
set -uo pipefail
cd /workspace/plodbnet || exit 1

GLOG=runs/vFour4_guardian.log
STOPFLAG=runs/vFour4.stop
LOG=runs/vFour4.log
COLLFLAG=runs/vFour4_collapsed.flag
SEED_CKPT=checkpoints/vFour3_40.pt
MAX_RESTARTS=4
POLL=300
restarts=0

log(){ echo "[vFour4-guardian $(date -u '+%m-%d %H:%M:%S')] $*" >> "$GLOG"; }
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
    --entropy-coef 0.45 --sizing-entropy-scale 1.0 \
    --lr 1.5e-4 --lr-warmup-updates 75 --target-kl 0.5 --kl-hard 10.0 --adv-clip 8 \
    --snapshot-every 5 \
    $load --checkpoint checkpoints/vFour4.pt \
    --num-updates 100000000 >> "$LOG" 2>&1 < /dev/null &
  disown 2>/dev/null || true
}

if [ -z "$(train_pid)" ]; then
  L=$(ls -t checkpoints/vFour4_*.pt 2>/dev/null | head -1)
  WARM="${L:-$SEED_CKPT}"
  log "initial launch warm-loading ${WARM}"
  launch "$WARM"; sleep 45
fi

log "started; watching pid=$(train_pid) (entropy=0.45, lr=1.5e-4 warmup=75, target_kl=0.5, max_restarts=$MAX_RESTARTS)"
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
      L=$(ls -t checkpoints/vFour4_*.pt 2>/dev/null | head -1)
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
