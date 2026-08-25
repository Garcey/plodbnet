#!/bin/bash
# vMin1 guardian — bare-visibility obs (796) + small-net experiment.
# Same v6 recipe / mix tiers / entropy start as vSix4, but:
#   --obs-mode minimal (direct 796 encode; opp_mc=0; no engineered tails)
#   actor 128x3, critic 128x2
#   rollout 22M; num_envs 220000
# Does NOT touch vSix4. Clean stop: touch runs/vMin1.stop
set -uo pipefail
cd /workspace/plodbnet || exit 1

GLOG=runs/vMin1_guardian.log
STOPFLAG=runs/vMin1.stop
LOG=runs/vMin1.log
MAX_RESTARTS=4
POLL=300
restarts=0

log(){ echo "[vMin1-guardian $(date -u '+%m-%d %H:%M:%S')] $*" >> "$GLOG"; }
# Match only THIS stem's train.py (avoid killing/seeing vSix4).
train_pid(){ pgrep -f "python -u scripts/[t]rain.py .*--checkpoint checkpoints/vMin1.pt" | head -1; }

launch(){
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PATH="$HOME/.cargo/bin:$PATH"
  # Lean minimal Rust encoder (observation_encoded_minimal_batch): emits
  # 796 directly, skips opp MC / v7 pack scans / engineered tails. opp_mc=0
  # is still forced by env_batched under obs_mode=minimal.
  export PLO5_RUST_ENCODER=1
  local load=""
  [ -n "${1:-}" ] && load="--load-checkpoint $1"
  setsid nohup .venv/bin/python -u scripts/train.py \
    --variant plo5_double_bomb \
    --v6 \
    --obs-mode minimal \
    --hidden-dim 128 --num-layers 3 \
    --critic-hidden-dim 128 --critic-num-blocks 2 \
    --batched --device cuda \
    --num-envs 220000 --rollout-length 22000000 --num-minibatches 16 --ppo-epochs 2 \
    --mix-configs --configs-per-tier 10 --mix-tiers clubgg,clubgg_deep,deep \
    --entropy-coef 0.25 --sizing-entropy-scale 1.0 \
    --lr 1.5e-4 --lr-warmup-updates 0 --clip-room-mid 0.07 \
    --target-kl 0.5 --kl-hard 10.0 --adv-clip 8 --cpu-threads 32 \
    --snapshot-every 5 \
    $load --checkpoint checkpoints/vMin1.pt \
    --num-updates 100000000 >> "$LOG" 2>&1 < /dev/null &
  disown 2>/dev/null || true
}

# Only warm-start from checkpoints that match the 128-wide architecture.
compatible_ckpt(){
  local f="$1"
  [ -f "$f" ] || return 1
  .venv/bin/python - "$f" <<'PY'
import sys, torch
p = sys.argv[1]
try:
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
except Exception:
    sys.exit(1)
cfg = ckpt.get("config") or {}
hd = int(cfg.get("hidden_dim") or 0)
nl = int(cfg.get("num_layers") or 0)
ch = int(cfg.get("critic_hidden_dim") or 0)
if hd <= 0:
    w = ckpt.get("model", {}).get("torso.0.0.weight")
    if w is not None:
        hd = int(w.shape[0])
if ch <= 0:
    w = (ckpt.get("critic") or {}).get("torso.0.0.weight")
    if w is not None:
        ch = int(w.shape[0])
ok = (hd == 128 and (nl in (0, 3)) and ch in (0, 128))
sys.exit(0 if ok else 1)
PY
}

pick_warm(){
  local f
  for f in $(ls -t checkpoints/vMin1_*.pt 2>/dev/null); do
    if compatible_ckpt "$f"; then
      echo "$f"
      return 0
    fi
  done
  if [ -f checkpoints/vMin1.pt ] && compatible_ckpt checkpoints/vMin1.pt; then
    echo "checkpoints/vMin1.pt"
    return 0
  fi
  return 1
}

if [ -z "$(train_pid)" ]; then
  if WARM=$(pick_warm); then
    log "initial launch warm-loading ${WARM} (128-wide)"
  else
    WARM=""
    log "initial launch COLD (minimal obs + 128x3/128x2 + 22M/220k)"
  fi
  launch "$WARM"; sleep 45
fi

log "started; watching pid=$(train_pid) (vMin1 = minimal-obs 796 + 128x3/128x2 + 220k envs + 22M rollout)"
while true; do
  [ -f "$STOPFLAG" ] && { log "stop flag present -> exiting"; exit 0; }
  sleep "$POLL"
  PID=$(train_pid)
  if [ -z "$PID" ]; then
    sleep 15; PID=$(train_pid)
    if [ -z "$PID" ]; then
      if [ "$restarts" -ge "$MAX_RESTARTS" ]; then log "DEAD; restart cap hit -> STOP"; touch "$STOPFLAG"; exit 0; fi
      restarts=$((restarts+1))
      if WARM=$(pick_warm); then
        log "process DEAD; relaunch #$restarts warm=${WARM}"
      else
        WARM=""
        log "process DEAD; relaunch #$restarts COLD"
      fi
      launch "$WARM"; sleep 45; continue
    fi
  fi
done
