#!/bin/bash
# Phase-5 launcher (lower entropy: clubgg 0.05 / clubgg_deep 0.06 / deep 0.085).
# Continuation of phase 4. Warm-starts from the HIGHEST-NUMBERED optimized5_<N>.pt
# if any exist (watchdog relaunch case), else from the highest optimized4_<N>.pt
# (initial transition from phase 4). Sets the CUDA allocator anti-fragmentation
# flag and launches detached. Used by the initial start and by
# watchdog_optimized5.sh for auto-relaunch.
#
# Checkpoint selection is by highest UPDATE NUMBER, not file mtime: train.py
# resets its update counter to 0 on warm-start, so a resumed run re-writes
# low-numbered checkpoints with newer mtimes — picking by mtime would resume
# from an EARLIER point and lose progress. Highest-number is monotonic.
set -uo pipefail
cd /workspace/plodbnet

# Refuse to launch a second trainer.
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

# Prefer the highest-numbered phase-5 checkpoint (resume); on the very first
# launch there are none, so fall back to the highest phase-4 checkpoint.
WARM=$(ls checkpoints/optimized5_*.pt 2>/dev/null \
        | sed -E 's|.*/optimized5_([0-9]+)\.pt|\1 &|' \
        | sort -n | tail -1 | awk '{print $2}')
if [[ -z "${WARM:-}" ]]; then
  WARM=$(ls checkpoints/optimized4_*.pt 2>/dev/null \
          | sed -E 's|.*/optimized4_([0-9]+)\.pt|\1 &|' \
          | sort -n | tail -1 | awk '{print $2}')
fi
if [[ -z "${WARM:-}" || ! -e "$WARM" ]]; then
  echo "[launch $(date -u +%H:%M:%S)] ERROR: no warm-start checkpoint found"
  exit 1
fi
echo "[launch $(date -u +%H:%M:%S)] warm-start from $WARM"

mkdir -p runs
LOG=runs/optimized5.log
[[ -e "$LOG" ]] && mv "$LOG" "${LOG}.$(date +%Y%m%d-%H%M%S).bak"

export PATH="$HOME/.cargo/bin:$PATH"
# Anti-fragmentation: lets the caching allocator's segments grow/shrink so the
# big per-rollout obs batch can't be blocked by a fragmented reserved pool.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

setsid nohup .venv/bin/python -u scripts/train.py \
  --batched --device cuda \
  --hidden-dim 2048 --num-layers 4 \
  --num-envs 49134 \
  --rollout-length 6266880 \
  --num-minibatches 48 \
  --block-rotation 'clubgg:0.05,clubgg_deep:0.06,deep:0.085' \
  --block-size 50 \
  --num-updates 100000000 \
  --load-checkpoint "$WARM" \
  --checkpoint checkpoints/optimized5.pt \
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
