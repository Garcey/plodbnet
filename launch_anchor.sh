#!/bin/bash
# Stem-bumping launcher for the v2 anchor-head family (anchor<N>).
#
# Same contract as launch_auto.sh, but for the cold-started v2
# architecture (11-anchor sizing head + centralized critic,
# head_version=2 checkpoints):
#   - No anchor<N> checkpoints on disk -> NEXT=1 and a COLD start
#     (no --load-checkpoint).
#   - Otherwise trains as anchor<X+1>, warm-started from the newest
#     LOADABLE checkpoint of stem X. v1 optimized<N> checkpoints are
#     never considered; train.py also refuses them via the
#     head_version check.
#
# NOTE: this family and the optimized<N> family both match
# 'scripts/train.py' in pgrep — run ONE watchdog family per pod.
# Stop the v1 family first:  touch runs/watchdog_auto.stop
#
# Entropy coefs are seeded at HALF the v1 values: the anchor
# categorical adds up to log(11) ≈ 2.4 nats on top of the v1
# gate+Beta entropy scale. Retune from the bake-off's Hg/Ha/Hb log
# fields once real readings exist.
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

# Highest anchor-stem number with at least one checkpoint on disk.
X=$(ls checkpoints/ 2>/dev/null \
     | sed -nE 's/^anchor([0-9]+)(_[0-9]+)?\.pt$/\1/p' \
     | sort -n | tail -1)
WARM=""
if [[ -z "${X:-}" ]]; then
  NEXT=1
  echo "[launch $(date -u +%H:%M:%S)] no anchor<N> checkpoints; COLD start as anchor1"
else
  NEXT=$((X + 1))
  # Newest loadable checkpoint of stem X.
  for f in $(ls -t checkpoints/anchor${X}_*.pt checkpoints/anchor${X}.pt 2>/dev/null); do
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
  echo "[launch $(date -u +%H:%M:%S)] stem anchor${NEXT}, warm-start from $WARM"
fi

mkdir -p runs
LOG=runs/anchor${NEXT}.log
[[ -e "$LOG" ]] && mv "$LOG" "${LOG}.$(date +%Y%m%d-%H%M%S).bak"

export PATH="$HOME/.cargo/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WARM_ARGS=()
[[ -n "$WARM" ]] && WARM_ARGS=(--load-checkpoint "$WARM")

setsid nohup .venv/bin/python -u scripts/train.py \
  --batched --device cuda \
  --hidden-dim 2048 --num-layers 4 \
  --critic-hidden-dim 1536 --critic-num-blocks 2 \
  --num-envs 49134 \
  --rollout-length 7833600 \
  --num-minibatches 48 \
  --block-rotation 'clubgg:0.02,clubgg_deep:0.03,deep:0.0425' \
  --block-size 50 \
  --anneal-entropy \
  --anneal-step 0.002 \
  --anneal-floor 0.0 \
  --anneal-tolerance 0.5 \
  --num-updates 100000000 \
  "${WARM_ARGS[@]}" \
  --checkpoint "checkpoints/anchor${NEXT}.pt" \
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
