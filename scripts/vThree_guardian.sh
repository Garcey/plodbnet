#!/bin/bash
# vThree autonomous guardian — runs ON THE POD, survives the Claude app closing.
#
# The Claude agent cannot act while the user's app is closed (no agent runs).
# This script is the unattended safety net, granted by the user 2026-06-21:
# recover vThree from (a) process DEATH (crash/OOM) -> relaunch from the latest
# checkpoint; (b) SUSTAINED gate collapse (>=6 of the last 8 updates Hg<0.10)
# -> kill + warm-restart from the known-clean vTwo10_445 at HALVED LR. Bounded
# restarts, a stop flag, and full logging so the morning review is easy.
#
# Stop it cleanly:  touch /workspace/plodbnet/runs/guardian.stop
set -uo pipefail
cd /workspace/plodbnet || exit 1

GLOG=runs/guardian.log
STOPFLAG=runs/guardian.stop
LOG=runs/vThree.log
CLEAN=checkpoints/vTwo10_445.pt   # known-clean fallback: deep block, Hg ~0.8
MAX_RESTARTS=4
POLL=300                          # seconds between checks
restarts=0

log(){ echo "[guardian $(date -u '+%m-%d %H:%M:%S')] $*" >> "$GLOG"; }

train_pid(){ pgrep -f "python -u scripts/train.py" | head -1; }

launch(){  # $1 = checkpoint to warm-load
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PATH="$HOME/.cargo/bin:$PATH"
  setsid nohup .venv/bin/python -u scripts/train.py \
    --batched --device cuda --hidden-dim 2048 --num-layers 4 \
    --critic-hidden-dim 1536 --critic-num-blocks 2 \
    --num-envs 49134 --rollout-length 9000000 --num-minibatches 16 --ppo-epochs 2 \
    --mix-configs --configs-per-tier 10 --mix-tiers clubgg,clubgg_deep,deep \
    --entropy-coef 0.45 --target-kl 2.0 --kl-hard 10.0 --adv-clip 8 --lr-warmup-updates 75 \
    --snapshot-every 5 \
    --load-checkpoint "$1" --checkpoint checkpoints/vThree.pt \
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
  COLL=$(grep -E "update +[0-9]" "$LOG" 2>/dev/null | tail -8 | grep -oE "Hg/Ha/Hb=[0-9.]+" | sed 's#.*=##' | awk '{if($1+0<0.10)c++} END{print c+0}')

  # --- process death (crash / OOM) -------------------------------------------
  if [ -z "$PID" ]; then
    sleep 15; PID=$(train_pid)              # re-check: ignore a transient miss
    if [ -z "$PID" ]; then
      if [ "$restarts" -ge "$MAX_RESTARTS" ]; then log "DEAD; restart cap ($MAX_RESTARTS) hit -> STOP"; touch "$STOPFLAG"; exit 0; fi
      restarts=$((restarts+1))
      L=$(ls -t checkpoints/vThree_*.pt 2>/dev/null | head -1); L=${L:-$CLEAN}
      log "process DEAD (crash/OOM); relaunch #$restarts from $L"
      launch "$L"; sleep 45; continue
    fi
  fi

  # --- sustained gate collapse -----------------------------------------------
  if [ "${NUPD:-0}" -ge 8 ] && [ "${COLL:-0}" -ge 6 ]; then
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then log "COLLAPSE; restart cap hit -> STOP"; touch "$STOPFLAG"; exit 0; fi
    restarts=$((restarts+1))
    GP=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$GP" ]; then kill "$GP" 2>/dev/null; sleep 6; kill -9 "$GP" 2>/dev/null; fi
    CUR_LR=$(grep -oE '"lr"[: ]+[0-9.]+' runs/anneal_control.json 2>/dev/null | grep -oE '[0-9.]+' | head -1)
    NEW_LR=$(awk "BEGIN{printf \"%.6g\", (${CUR_LR:-0.00015})/2}")
    printf '{"lr": %s, "tier_ent": {"clubgg": 0.45, "clubgg_deep": 0.45, "deep": 0.45}}\n' "$NEW_LR" \
      > runs/anneal_control.json.tmp && mv runs/anneal_control.json.tmp runs/anneal_control.json
    log "SUSTAINED COLLAPSE (Hg=${LASTHG:-?}, $COLL/8 <0.10); warm-restart #$restarts from $CLEAN at LR=$NEW_LR (mix-configs should make this rare — review!)"
    sleep 8; launch "$CLEAN"; sleep 45; continue
  fi

  log "ok: pid=$PID updates=$NUPD Hg=${LASTHG:-none} collapsed_last8=${COLL:-0} restarts=$restarts"
done
