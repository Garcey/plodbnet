#!/bin/bash
# vSix4 guardian — warm-start from vSix3_60 with rollout perf #1-#5
# (Rust obs encoder @1171 via PLO5_RUST_ENCODER=1, skip terminal encode,
# hole subset, env reuse, pinned step H2D). Same training recipe as vSix3.
# Compare wall-clock/update vs vSix3 (see runs/vSix3.log ~800s/update).
# Clean stop: touch runs/vSix4.stop, kill THIS guardian first, then trainer.
set -uo pipefail
cd /workspace/plodbnet || exit 1

GLOG=runs/vSix4_guardian.log
STOPFLAG=runs/vSix4.stop
LOG=runs/vSix4.log
COLLFLAG=runs/vSix4_collapsed.flag
MAX_RESTARTS=4
POLL=300
restarts=0

log(){ echo "[vSix4-guardian $(date -u '+%m-%d %H:%M:%S')] $*" >> "$GLOG"; }
train_pid(){ pgrep -f "python -u scripts/[t]rain.py" | head -1; }

launch(){  # $1 = optional checkpoint to warm-load (empty = newest vSix4_*)
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PATH="$HOME/.cargo/bin:$PATH"
  # Full Rust obs encoder (OBS_DIM 1171) — main expected wall-clock win.
  export PLO5_RUST_ENCODER=1
  local load=""
  [ -n "${1:-}" ] && load="--load-checkpoint $1"
  setsid nohup .venv/bin/python -u scripts/train.py \
    --variant plo5_double_bomb \
    --v6 \
    --hidden-dim 2048 --num-layers 4 \
    --critic-hidden-dim 1536 --critic-num-blocks 2 \
    --batched --device cuda \
    --num-envs 49134 --rollout-length 9000000 --num-minibatches 16 --ppo-epochs 2 \
    --mix-configs --configs-per-tier 10 --mix-tiers clubgg,clubgg_deep,deep \
    --entropy-coef 0.25 --sizing-entropy-scale 1.0 \
    --lr 1.5e-4 --lr-warmup-updates 0 --clip-room-mid 0.07 \
    --target-kl 0.5 --kl-hard 10.0 --adv-clip 8 --cpu-threads 32 \
    --snapshot-every 5 \
    $load --checkpoint checkpoints/vSix4.pt \
    --num-updates 100000000 >> "$LOG" 2>&1 < /dev/null &
  disown 2>/dev/null || true
}

if [ -z "$(train_pid)" ]; then
  L=$(ls -t checkpoints/vSix4_*.pt 2>/dev/null | head -1)
  if [ -n "${L:-}" ]; then
    WARM="$L"
  else
    # First launch: warm from vSix3 update 60 (perf A/B baseline).
    WARM="checkpoints/vSix3_60.pt"
  fi
  if [ -n "$WARM" ] && [ -f "$WARM" ]; then
    log "initial launch warm-loading ${WARM}"
  else
    log "initial launch COLD (missing warm ckpt)"
    WARM=""
  fi
  launch "$WARM"; sleep 45
fi

log "started; watching pid=$(train_pid) (vSix4 = vSix3 recipe + rollout #1-#5 + PLO5_RUST_ENCODER=1; warm from vSix3_60)"
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
      L=$(ls -t checkpoints/vSix4_*.pt 2>/dev/null | head -1)
      WARM="${L:-checkpoints/vSix3_60.pt}"
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
