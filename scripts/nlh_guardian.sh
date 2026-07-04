#!/bin/bash
# nlh<N> guardian (STEM below) — NLH training runs (cold start lineage).
#
# nlh1 (2026-07-03, ~4 updates): first launch, --stack-dist deep. Replaced
# same night by nlh2 when the user specified the REAL table curriculum:
# --stack-dist nlh_topoff (40% pinned at the 100bb auto-top-off, 25%
# short 30-100bb, 20% at 100-150bb, 15% winners 150-400bb) +
# --seats-dist nlh_ring (5-6 handed slightly favored, 2-4 even).
# nlh1's Hg trend before replacement: 0.17 -> 0.52 (cold-init sharpness
# recovering under the 0.30 entropy bonus — healthy).
#
# vFour4 (PLO5DBBP) was PAUSED gracefully for this GPU — NOT collapsed.
# It resumes later via warm-start; its pool rebuilds from
# checkpoints/vFour4_*.pt (warm-start pool seeding) — do NOT prune those.
#
# Config (USER-DIRECTED 2026-07-03):
#   - COLD START (user mandate: no cross-variant warm-starts — equities
#     and made-hand strength differ too much by game; the guard refuses
#     them anyway).
#   - entropy 0.30 (user's pick for the NLH cold start. CAVEAT ON
#     RECORD: 0.30 is the value vFour3's PLO gate collapsed at under
#     the same v4 head — different game, but if Hg slides toward the
#     0.15 detector below, the next lever is 0.45, the proven PLO
#     floor). lr 1.5e-4 with 75-update warmup, target_kl 0.5,
#     kl_hard 10, adv-clip 8 — vFour4's proven guard rails.
#   - rollout 11.6M / 16 minibatches / 2 epochs (user's pick — the
#     computed max at ~90 GiB on the 96 GB card with the minibatch
#     growing to ~725k; watch nvidia-smi on the first updates).
#   - --stack-dist nlh_topoff + --seats-dist nlh_ring (see above), seats
#     2-6 sampled per update. NO --mix-configs (its tiers are PLO stack
#     blocks; NLH resamples a config every update instead).
#   - batched collector (NLH packer + vectorized encoder shipped
#     2026-07-03); NLH is byte-equivalent to PLO5 per decision in VRAM.
#   - 5/10 with $5/player ante falls out of the defaults at bb=10000
#     (sb=bb/2, ante=bb/2).
#
# Relaunches warm-load the newest ${STEM}_*.pt (same-variant warm start;
# the opponent pool auto-reseeds from siblings). Stop cleanly:
#   touch runs/<stem>.stop
set -uo pipefail
cd /workspace/plodbnet || exit 1

STEM=nlh2
GLOG=runs/${STEM}_guardian.log
STOPFLAG=runs/${STEM}.stop
LOG=runs/${STEM}.log
COLLFLAG=runs/${STEM}_collapsed.flag
MAX_RESTARTS=4
POLL=300
restarts=0

log(){ echo "[${STEM}-guardian $(date -u '+%m-%d %H:%M:%S')] $*" >> "$GLOG"; }
train_pid(){ pgrep -f "python -u scripts/[t]rain.py" | head -1; }

launch(){  # $1 = checkpoint to warm-load ("" = cold start)
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PATH="$HOME/.cargo/bin:$PATH"
  local load=""
  [ -n "${1:-}" ] && load="--load-checkpoint $1"
  setsid nohup .venv/bin/python -u scripts/train.py \
    --variant nlh_single --sizing-head logistic \
    --batched --device cuda --hidden-dim 2048 --num-layers 4 \
    --critic-hidden-dim 1536 --critic-num-blocks 2 \
    --num-envs 49134 --rollout-length 11600000 --num-minibatches 16 --ppo-epochs 2 \
    --stack-dist nlh_topoff --seats-dist nlh_ring --num-seats-range "2,3,4,5,6" \
    --entropy-coef 0.30 --sizing-entropy-scale 1.0 \
    --lr 1.5e-4 --lr-warmup-updates 75 --target-kl 0.5 --kl-hard 10.0 --adv-clip 8 \
    --snapshot-every 5 \
    $load --checkpoint checkpoints/${STEM}.pt \
    --num-updates 100000000 >> "$LOG" 2>&1 < /dev/null &
  disown 2>/dev/null || true
}

if [ -z "$(train_pid)" ]; then
  L=$(ls -t checkpoints/${STEM}_*.pt 2>/dev/null | head -1)
  WARM="${L:-}"
  if [ -n "$WARM" ]; then
    log "initial launch warm-loading ${WARM}"
  else
    log "initial launch COLD (first NLH run)"
  fi
  launch "$WARM"; sleep 45
fi

log "started; watching pid=$(train_pid) (nlh_single cold, entropy=0.30, rollout=11.6M, lr=1.5e-4 warmup=75, target_kl=0.5, max_restarts=$MAX_RESTARTS)"
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
      L=$(ls -t checkpoints/${STEM}_*.pt 2>/dev/null | head -1)
      WARM="${L:-}"
      log "process DEAD (crash/OOM); relaunch #$restarts warm-loading ${WARM:-COLD}"
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
