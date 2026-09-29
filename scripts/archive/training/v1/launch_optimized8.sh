#!/bin/bash
# Phase-8 launcher. ENTROPY ANNEAL STEP 1: lower the clubgg (shallow) entropy
# floor 0.05 -> 0.04; clubgg_deep (0.06) and deep (0.085) UNCHANGED. Everything
# else identical to phase 7 (rollout-length 7,833,600 / num-minibatches 48 ->
# batch_size 163,200; block-size 50). New stem so the before/after F/T/R
# comparison stays clean and the 0.04-floor checkpoints don't mix with phase-7.
# Warm-starts from the highest optimized8_<N>.pt if any exist (watchdog
# relaunch), else from the highest optimized7_<N>.pt (initial transition).
#
# WATCH (entropy-anneal guardrail): clubgg bonus%(F/T/R) must HOLD or RISE vs
# the phase-7 baseline ~20/28/35. If clubgg F/T/R drops >~2-3 pts (esp. flop
# toward single digits) over its next 1-2 blocks, REVERT this step (relaunch
# phase 7 / clubgg:0.05). See memory: project_entropy_anneal_protocol.
#
# Checkpoint selection is by highest UPDATE NUMBER (train.py resets its counter
# to 0 on warm-start, re-writing low numbers with newer mtimes; mtime-pick
# would resume from an EARLIER point). Highest-number is monotonic WITHIN a
# stem; the stem bump keeps phases cleanly separated.
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

WARM=$(ls checkpoints/optimized8_*.pt 2>/dev/null \
        | sed -E 's|.*/optimized8_([0-9]+)\.pt|\1 &|' \
        | sort -n | tail -1 | awk '{print $2}')
if [[ -z "${WARM:-}" ]]; then
  WARM=$(ls checkpoints/optimized7_*.pt 2>/dev/null \
          | sed -E 's|.*/optimized7_([0-9]+)\.pt|\1 &|' \
          | sort -n | tail -1 | awk '{print $2}')
fi
if [[ -z "${WARM:-}" || ! -e "$WARM" ]]; then
  echo "[launch $(date -u +%H:%M:%S)] ERROR: no warm-start checkpoint found"
  exit 1
fi
echo "[launch $(date -u +%H:%M:%S)] warm-start from $WARM"

mkdir -p runs
LOG=runs/optimized8.log
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
  --num-updates 100000000 \
  --load-checkpoint "$WARM" \
  --checkpoint checkpoints/optimized8.pt \
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
