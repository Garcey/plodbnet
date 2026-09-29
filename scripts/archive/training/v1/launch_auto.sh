#!/bin/bash
# Stem-bumping launcher (replaces per-phase launch_optimizedN.sh).
#
# Finds the highest checkpoint stem optimized<X> in checkpoints/, then
# trains as optimized<X+1>, warm-started from the newest LOADABLE
# checkpoint of stem X (newest by mtime across optimizedX_<N>.pt and the
# final optimizedX.pt; a checkpoint truncated by a mid-write death is
# skipped in favor of the next older one). Each death/relaunch therefore
# begins a new stem — the lineage records every restart.
#
# If the new stem dies before writing any checkpoint, the next launch
# recomputes X from files on disk, so it re-uses the same target stem
# instead of inflating numbers.
#
# Entropy anneal: tier floors, F/T/R baselines, block accumulator and
# the update counter are persisted IN the checkpoint and restored by
# --anneal-entropy (scripts/train.py:599-622), so annealed floors carry
# across stems. The --block-rotation values below only seed a checkpoint
# that has no anneal state (e.g. pre-anneal optimized8 files).
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

# Highest stem number with at least one checkpoint on disk.
X=$(ls checkpoints/ 2>/dev/null \
     | sed -nE 's/^optimized([0-9]+)(_[0-9]+)?\.pt$/\1/p' \
     | sort -n | tail -1)
if [[ -z "${X:-}" ]]; then
  echo "[launch $(date -u +%H:%M:%S)] ERROR: no optimized<N> checkpoints found"
  exit 1
fi
NEXT=$((X + 1))

# Newest loadable checkpoint of stem X.
WARM=""
for f in $(ls -t checkpoints/optimized${X}_*.pt checkpoints/optimized${X}.pt 2>/dev/null); do
  if .venv/bin/python -c "import sys,torch; torch.load(sys.argv[1], map_location='cpu', weights_only=False)" "$f" >/dev/null 2>&1; then
    WARM="$f"
    break
  fi
  echo "[launch $(date -u +%H:%M:%S)] checkpoint $f failed to load; trying older"
done
if [[ -z "${WARM:-}" ]]; then
  echo "[launch $(date -u +%H:%M:%S)] ERROR: no loadable stem-${X} checkpoint"
  exit 1
fi
echo "[launch $(date -u +%H:%M:%S)] stem optimized${NEXT}, warm-start from $WARM"

mkdir -p runs
LOG=runs/optimized${NEXT}.log
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
  --checkpoint "checkpoints/optimized${NEXT}.pt" \
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
