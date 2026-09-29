#!/bin/bash
# Phase-3: revert to phase-1 rollout/minibatches (proven-fit), keep
# phase-2's lowered entropy floors (0.07/0.09/0.12). Warm-start from
# the latest optimized_<N>.pt.
set -euo pipefail
cd /workspace/plodbnet

# --- preflight: refuse to launch if any train.py is still writing optimized*.pt ---
if pgrep -af 'scripts/train\.py.*checkpoints/optimized[0-9]*\.pt' >/dev/null; then
  echo "ERROR: an existing training process is still writing checkpoints/optimized*.pt:"
  pgrep -af 'scripts/train\.py'
  echo
  echo "Stop it first (kill <PID>), then re-run this script."
  exit 1
fi

# --- pick the most recent optimized_<N>.pt as warm-start source (ignores optimized2_*.pt etc.) ---
WARM=$(ls -1 checkpoints/optimized_*.pt 2>/dev/null \
       | sed -E 's|.*/optimized_([0-9]+)\.pt|\1 &|' \
       | sort -n \
       | tail -1 \
       | awk '{print $2}')
if [[ -z "${WARM:-}" ]]; then
  echo "ERROR: no checkpoints/optimized_*.pt found to warm-start from"
  exit 1
fi
echo "[launcher] warm-start from $WARM"

mkdir -p runs
LOG=runs/optimized3.log
if [[ -e "$LOG" ]]; then
  mv "$LOG" "${LOG}.$(date +%Y%m%d-%H%M%S).bak"
fi
echo "[launcher] log -> $LOG"

# --- launch (nohup + disown so SSH disconnect doesn't kill it) ---
nohup .venv/bin/python -u scripts/train.py \
  --batched --device cuda \
  --hidden-dim 2048 --num-layers 4 \
  --num-envs 49134 \
  --rollout-length 6266880 \
  --num-minibatches 48 \
  --block-rotation 'clubgg:0.07,clubgg_deep:0.09,deep:0.12' \
  --block-size 50 \
  --num-updates 100000000 \
  --load-checkpoint "$WARM" \
  --checkpoint checkpoints/optimized3.pt \
  >> "$LOG" 2>&1 &

PID=$!
disown $PID || true
echo "[launcher] training PID=$PID"

# Wait briefly and verify it didn't die at startup.
sleep 5
if ! kill -0 "$PID" 2>/dev/null; then
  echo "ERROR: training died within 5s of launch. Tail of $LOG:"
  tail -60 "$LOG"
  exit 1
fi

echo "[launcher] training alive."
echo "[launcher] monitor: tail -f $LOG"
echo "[launcher] checkpoints land at checkpoints/optimized3_<update>.pt every 5 updates."
