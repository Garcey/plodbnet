#!/bin/bash
# vFive1 guardian — first v5 stem (K=3 mixture sizing head + obs v2 +
# kl-anchor EMA magnet), warm-started from vFour4_915 via
# scripts/convert_v4_to_v5.py (component 0 = v4, w0~=0.92, so update 0
# plays ~vFour4's policy; the mixture menu emerges from there).
#
# Config rationale (V5_DESIGN.md):
#   - entropy-coef 0.45 (vFour4's proven floor; re-seeded HIGH, not an
#     annealed floor — the mixture head is a harder exploration problem;
#     anneal is one-way down so err high).
#   - kl-anchor-coef 0.05 (the MMD magnet; EMA ref starts as a clone so
#     KL~=0 early and ramps in over ~1/(1-0.999) updates — negligible
#     disturbance to the warm-start bake-in, present for later annealing).
#   - value-clip 0.2 (vFour4-proven; deliberately NOT loosened here — do
#     the value-clip A/B on a throwaway stem, not the real launch).
#   - q-aux-coef 0 (the critic's dueling Q head is built + zero-init so
#     the VRPO flip is not a checkpoint break, but left UNTRAINED so the
#     critic forward stays torch.compiled/fast). Enable later for VRPO.
#   - rollout 11M (comfortable VRAM point; v5's +29 obs dims add ~1.25GiB
#     over vFour4 — watch the first `[cuda] peak` line and bump toward
#     11.5M only if there's headroom).
#   - uniform 2-6 seats (INTENTIONAL — rounded model incl. deep 3-4-handed
#     home games; do NOT add --seats-dist).
#
# Health watch: same as vFour4 (Hg collapse < 0.15 for 5/8 blocks -> stop;
# process death -> warm relaunch up to cap). Watch ALSO klA (mixture can
# spike heavier than v4's single logistic — target_kl 0.5 + kl_hard 10
# guard it) and the mixture menu via scripts/check_mixture_usage.py.
#
# Stop cleanly: touch runs/vFive1.stop
set -uo pipefail
cd /workspace/plodbnet || exit 1

GLOG=runs/vFive1_guardian.log
STOPFLAG=runs/vFive1.stop
LOG=runs/vFive1.log
COLLFLAG=runs/vFive1_collapsed.flag
SEED_CKPT=checkpoints/vFive1_seed.pt
MAX_RESTARTS=4
POLL=300
restarts=0

log(){ echo "[vFive1-guardian $(date -u '+%m-%d %H:%M:%S')] $*" >> "$GLOG"; }
train_pid(){ pgrep -f "python -u scripts/[t]rain.py" | head -1; }

launch(){  # $1 = checkpoint to warm-load
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PATH="$HOME/.cargo/bin:$PATH"
  local load=""
  [ -n "${1:-}" ] && load="--load-checkpoint $1"
  setsid nohup .venv/bin/python -u scripts/train.py \
    --variant plo5_double_bomb \
    --sizing-head mixture --mixture-k 3 \
    --batched --device cuda --hidden-dim 2048 --num-layers 4 \
    --critic-hidden-dim 1536 --critic-num-blocks 2 \
    --num-envs 49134 --rollout-length 11000000 --num-minibatches 16 --ppo-epochs 2 \
    --mix-configs --configs-per-tier 10 --mix-tiers clubgg,clubgg_deep,deep \
    --entropy-coef 0.45 --sizing-entropy-scale 1.0 \
    --kl-anchor-coef 0.05 --value-clip 0.2 \
    --lr 1.5e-4 --lr-warmup-updates 75 --target-kl 0.5 --kl-hard 10.0 --adv-clip 8 \
    --snapshot-every 5 \
    $load --checkpoint checkpoints/vFive1.pt \
    --num-updates 100000000 >> "$LOG" 2>&1 < /dev/null &
  disown 2>/dev/null || true
}

if [ -z "$(train_pid)" ]; then
  L=$(ls -t checkpoints/vFive1_*.pt 2>/dev/null | grep -v '_seed' | head -1)
  WARM="${L:-$SEED_CKPT}"
  log "initial launch warm-loading ${WARM}"
  launch "$WARM"; sleep 45
fi

log "started; watching pid=$(train_pid) (mixture K=3, entropy=0.45, kl-anchor=0.05, lr=1.5e-4 warmup=75, target_kl=0.5, max_restarts=$MAX_RESTARTS)"
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
      L=$(ls -t checkpoints/vFive1_*.pt 2>/dev/null | grep -v '_seed' | head -1)
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
