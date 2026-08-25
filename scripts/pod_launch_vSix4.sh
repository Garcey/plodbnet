#!/bin/bash
set -euo pipefail
cd /workspace/plodbnet
export PATH=/workspace/.cargo/bin:$PATH
export RUSTUP_HOME=/workspace/.rustup
export CARGO_HOME=/workspace/.cargo

chmod +x scripts/vSix4_guardian.sh

# Smoke: new symbols + rust encoder gate
.venv/bin/python - <<'PY'
import os
os.environ["PLO5_RUST_ENCODER"] = "1"
from plo5bp.env_batched import BatchedBombPotEnv, _RUST_ENCODER_OBS_DIM
from plo5bp.encoding import OBS_DIM
from plo5bp.config import GameConfig
e = BatchedBombPotEnv(2, GameConfig(num_seats=2), opp_outcome_mc=64)
print("OBS_DIM", OBS_DIM, "rust_gate", _RUST_ENCODER_OBS_DIM, "use_rust", e._use_rust_encoder)
print("has_subset", hasattr(e._be, "all_hole_cards_subset_batch"))
print("has_reconfigure", hasattr(e._be, "reconfigure"))
assert OBS_DIM == 1171
assert e._use_rust_encoder is True
assert hasattr(e._be, "all_hole_cards_subset_batch")
assert hasattr(e._be, "reconfigure")
print("smoke OK")
PY

# Ensure vSix3 stays stopped; clear any stale vSix4 stop
test -f runs/vSix3.stop && echo "vSix3.stop present"
rm -f runs/vSix4.stop
test -f checkpoints/vSix3_60.pt || { echo "MISSING vSix3_60.pt"; exit 1; }

# Launch guardian (it warm-starts from vSix3_60 if no vSix4_* yet)
nohup bash scripts/vSix4_guardian.sh >> runs/vSix4_guardian_boot.log 2>&1 &
sleep 55

echo "=== processes ==="
pgrep -af 'vSix4_guardian|scripts/train.py' || true
echo "=== vSix4 log (tail) ==="
tail -30 runs/vSix4.log 2>/dev/null || echo "(no log yet)"
echo "=== guardian log (tail) ==="
tail -15 runs/vSix4_guardian.log 2>/dev/null || echo "(no glog yet)"
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
