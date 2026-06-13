#!/bin/bash
# Stem-bumping launcher for the v2 anchor-head family (vTwo<N>).
#
# Same contract as launch_auto.sh, but for the cold-started v2
# architecture (11-anchor sizing head + centralized critic,
# head_version=2 checkpoints):
#   - No vTwo<N> checkpoints on disk -> NEXT=1 and a COLD start
#     (no --load-checkpoint).
#   - Otherwise trains as vTwo<X+1>, warm-started from the newest
#     LOADABLE checkpoint of stem X. v1 optimized<N> checkpoints are
#     never considered; train.py also refuses them via the
#     head_version check.
#
# NOTE: this family and the optimized<N> family both match
# 'scripts/train.py' in pgrep — run ONE watchdog family per pod.
# Stop the v1 family first:  touch runs/watchdog_auto.stop
#
# Entropy coefs seed at 0.5 across all blocks (user decision
# 2026-06-11): a deliberate high-exploration phase. At the collapsed
# fixed point the suppressing and restoring forces on a gate both
# scale with its probability, so the coef directly decides whether
# collapse is escapable — 0.1 was borderline, and cold starts at
# 0.10/0.18 still pinned fold ~0 by update 5. Step the coefs DOWN
# MANUALLY via runs/anneal_control.json {"tier_ent": {...}} once the
# strategy matures (don't wait for the slow auto-anneal); anneal
# decisions begin after --anneal-start-update updates.
#
# --lr-warmup-updates 75: full-LR cold-start Adam steps moved the
# policy KL 1-20 per minibatch; the warmup eases them in.
#
# KL guard is now SPLIT (the single-threshold full-rollback froze a
# run — 317/318 updates rolled back to no-ops once the policy
# sharpened, 2026-06-12):
#   --target-kl 2.0  SOFT early-stop: stop the inner loop but KEEP the
#                    minibatches already applied (standard PPO).
#   --kl-hard 10.0   HARD rollback: revert the WHOLE update; reserved
#                    for catastrophe (vTwo2 hit approx_kl ≈ +2417 at
#                    update 173). Both live-tunable via anneal_control.
#
# Base LR is the train.py default (3e-4) — live-tunable down/up via
# runs/anneal_control.json {"lr": X} (decay it as the policy sharpens
# and KLSTOP frequency climbs, no restart, opponent pool preserved).
#
# 16 minibatches x 2 epochs (was 48 x 4): bigger minibatches = better
# game-tree coverage per gradient step; the 7.83M batch doesn't need
# 4x reuse.
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

# Highest vTwo-stem number with at least one checkpoint on disk.
X=$(ls checkpoints/ 2>/dev/null \
     | sed -nE 's/^vTwo([0-9]+)(_[0-9]+)?\.pt$/\1/p' \
     | sort -n | tail -1)
WARM=""
if [[ -z "${X:-}" ]]; then
  NEXT=1
  echo "[launch $(date -u +%H:%M:%S)] no vTwo<N> checkpoints; COLD start as vTwo1"
else
  NEXT=$((X + 1))
  # Newest loadable checkpoint of stem X.
  for f in $(ls -t checkpoints/vTwo${X}_*.pt checkpoints/vTwo${X}.pt 2>/dev/null); do
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
  echo "[launch $(date -u +%H:%M:%S)] stem vTwo${NEXT}, warm-start from $WARM"
fi

mkdir -p runs
LOG=runs/vTwo${NEXT}.log
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
  --num-minibatches 16 \
  --ppo-epochs 2 \
  --block-rotation 'clubgg:0.5,clubgg_deep:0.5,deep:0.5' \
  --block-size 50 \
  --target-kl 2.0 \
  --kl-hard 10.0 \
  --adv-clip 8 \
  --lr-warmup-updates 75 \
  --anneal-entropy \
  --anneal-step 0.002 \
  --anneal-floor 0.0 \
  --anneal-tolerance 1.0 \
  --anneal-start-update 900 \
  --num-updates 100000000 \
  "${WARM_ARGS[@]}" \
  --checkpoint "checkpoints/vTwo${NEXT}.pt" \
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
