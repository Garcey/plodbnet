#!/bin/bash
# Phase-9 launcher. AUTOMATED ENTROPY ANNEAL (--anneal-entropy): the trainer now
# lowers each tier's entropy coef by 0.002 whenever that tier's F/T/R held vs its
# previous same-tier block, floored at 0. Annealed floors + per-tier F/T/R
# baselines + the update counter are persisted IN the checkpoint and restored on
# warm-start, so a relaunch resumes the annealed floors AND the block position
# (no reset-to-block-1 → deep is no longer under-trained by restarts). No more
# manual entropy restarts; runs indefinitely, pool grows between pod deaths.
#
# Starting floors (the anneal walks down from here): clubgg 0.04 / clubgg_deep
# 0.06 / deep 0.085 — i.e. where phase 8 was. Everything else identical to
# phase 8 (rollout 7,833,600 / num-minibatches 48; block-size 50). Warm-starts
# from highest optimized9_<N>.pt if any (watchdog relaunch), else highest
# optimized8_<N>.pt (initial transition; optimized8 ckpts have no anneal keys, so
# anneal state seeds fresh from the CLI floors, counter 0).
#
# Checkpoint selection is by highest UPDATE NUMBER. With --anneal-entropy the
# counter is restored (not reset to 0), so mid-checkpoint numbers keep climbing
# monotonically within the stem.
set -uo pipefail
cd /workspace/plodbnet

if pgrep -af 'scripts/train\.py' >/dev/null; then
  echo "[launch $(date -u +%H:%M:%S)] training already running; not starting another."
  exit 0
fi

# Wait for the GPU to free (a crashed process can briefly hold memory).
for _ in $(seq 1 18); do
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [[ "${MEM:-0}" -lt 2000 ]]; then break; fi
  echo "[launch $(date -u +%H:%M:%S)] GPU busy (${MEM} MiB); waiting..."
  sleep 10
done

WARM=$(ls checkpoints/optimized9_*.pt 2>/dev/null \
        | sed -E 's|.*/optimized9_([0-9]+)\.pt|\1 &|' \
        | sort -n | tail -1 | awk '{print $2}')
if [[ -z "${WARM:-}" ]]; then
  WARM=$(ls checkpoints/optimized8_*.pt 2>/dev/null \
          | sed -E 's|.*/optimized8_([0-9]+)\.pt|\1 &|' \
          | sort -n | tail -1 | awk '{print $2}')
fi
if [[ -z "${WARM:-}" || ! -e "$WARM" ]]; then
  echo "[launch $(date -u +%H:%M:%S)] ERROR: no warm-start checkpoint found"
  exit 1
fi
echo "[launch $(date -u +%H:%M:%S)] warm-start from $WARM"

mkdir -p runs
LOG=runs/optimized9.log
[[ -e "$LOG" ]] && mv "$LOG" "${LOG}.$(date +%Y%m%d-%H%M%S).bak"

export PATH="$HOME/.cargo/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

setsid nohup .venv/bin/python -u scripts/train.py \
  --batched --device cuda \
  --hidden-dim 2048 --num-layers 4 \
  --num-envs 49134 \
  --rollout-length 7833600 \
  --num-minibatches 48 \
  --block-rotation 'clubgg:0.04,clubgg_deep:0.06,deep:0.085' \
  --block-size 50 \
  --anneal-entropy \
  --anneal-step 0.002 \
  --anneal-floor 0.0 \
  --anneal-tolerance 0.5 \
  --num-updates 100000000 \
  --load-checkpoint "$WARM" \
  --checkpoint checkpoints/optimized9.pt \
  >> "$LOG" 2>&1 < /dev/null &
PID=$!
disown "$PID" || true
echo "[launch $(date -u +%H:%M:%S)] training PID=$PID (log: $LOG)"

sleep 6
if ! kill -0 "$PID" 2>/dev/null; then
  echo "[launch $(date -u +%H:%M:%S)] ERROR: died within 6s. Tail of $LOG:"
  tail -30 "$LOG"
  exit 1
fi
echo "[launch $(date -u +%H:%M:%S)] training alive."
