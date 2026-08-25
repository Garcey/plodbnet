from pathlib import Path

# permanent small unit test
test = Path("tests/python/test_obs_minimal.py")
test.write_text('''"""Bare-visibility (minimal) observation layout."""
from __future__ import annotations

import numpy as np
import pytest

from plo5bp.config import GameConfig
from plo5bp.encoding import (
    OBS_DIM,
    OBS_DIM_MINIMAL,
    _MINIMAL_INDEX,
    _SPR_OFF,
    _OPP_OUTCOME_OFF,
    project_obs_minimal,
)
from plo5bp.env import BombPotEnv
from plo5bp.env_batched import BatchedBombPotEnv


def test_minimal_dim_and_index():
    assert OBS_DIM_MINIMAL == 796
    assert _MINIMAL_INDEX.shape == (796,)
    # cards block contiguous
    assert list(_MINIMAL_INDEX[:156]) == list(range(156))
    kept = set(int(x) for x in _MINIMAL_INDEX)
    assert _SPR_OFF not in kept
    assert _OPP_OUTCOME_OFF not in kept
    assert 990 not in kept  # bet_pct_pot


def test_serial_minimal_matches_project():
    cfg = GameConfig()
    full = BombPotEnv(cfg, obs_mode="full")
    mini = BombPotEnv(cfg, obs_mode="minimal")
    obs_f, _ = full.reset(seed=7, button=1)
    obs_m, _ = mini.reset(seed=7, button=1)
    assert obs_f.shape == (OBS_DIM,)
    assert obs_m.shape == (OBS_DIM_MINIMAL,)
    np.testing.assert_array_equal(project_obs_minimal(obs_f), obs_m)


def test_batched_minimal_matches_project():
    cfg = GameConfig()
    full = BatchedBombPotEnv(4, cfg, obs_mode="full", opp_outcome_mc=1)
    mini = BatchedBombPotEnv(4, cfg, obs_mode="minimal", opp_outcome_mc=1)
    seeds = np.arange(4, dtype=np.uint64) + 100
    buttons = np.array([0, 1, 2, 0], dtype=np.uint8)
    st_f = full.reset_batch(seeds, buttons)
    st_m = mini.reset_batch(seeds, buttons)
    assert st_m.obs.shape == (4, OBS_DIM_MINIMAL)
    np.testing.assert_array_equal(project_obs_minimal(st_f.obs), st_m.obs)


def test_minimal_rejects_nlh():
    from plo5bp.config import VARIANT_NLH
    cfg = GameConfig(variant=VARIANT_NLH)
    with pytest.raises(ValueError, match="PLO-only"):
        BombPotEnv(cfg, obs_mode="minimal")
''', encoding="utf-8")
print("wrote", test)

# guardian script
g = Path("scripts/vMin1_guardian.sh")
g.write_text(r'''#!/bin/bash
# vMin1 guardian — bare-visibility obs (796) + smaller net experiment.
# Same v6 recipe / mix tiers / entropy start as vSix4, but:
#   --obs-mode minimal (no engineered strength / no opp-MC features)
#   actor 1024x3, critic 1024x2 (capacity underuse study)
#   smaller env count so it can share a GPU or run lean
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
  # Minimal path uses numpy project; rust full encoder is irrelevant / skipped.
  export PLO5_RUST_ENCODER=0
  local load=""
  [ -n "${1:-}" ] && load="--load-checkpoint $1"
  setsid nohup .venv/bin/python -u scripts/train.py \
    --variant plo5_double_bomb \
    --v6 \
    --obs-mode minimal \
    --hidden-dim 1024 --num-layers 3 \
    --critic-hidden-dim 1024 --critic-num-blocks 2 \
    --batched --device cuda \
    --num-envs 16384 --rollout-length 4000000 --num-minibatches 16 --ppo-epochs 2 \
    --mix-configs --configs-per-tier 10 --mix-tiers clubgg,clubgg_deep,deep \
    --entropy-coef 0.25 --sizing-entropy-scale 1.0 \
    --lr 1.5e-4 --lr-warmup-updates 0 --clip-room-mid 0.07 \
    --target-kl 0.5 --kl-hard 10.0 --adv-clip 8 --cpu-threads 32 \
    --snapshot-every 5 \
    $load --checkpoint checkpoints/vMin1.pt \
    --num-updates 100000000 >> "$LOG" 2>&1 < /dev/null &
  disown 2>/dev/null || true
}

if [ -z "$(train_pid)" ]; then
  L=$(ls -t checkpoints/vMin1_*.pt 2>/dev/null | head -1)
  if [ -n "${L:-}" ]; then
    WARM="$L"
    log "initial launch warm-loading ${WARM}"
  else
    WARM=""
    log "initial launch COLD (minimal obs + smaller net)"
  fi
  launch "$WARM"; sleep 45
fi

log "started; watching pid=$(train_pid) (vMin1 = minimal-obs 796 + 1024x3/1024x2)"
while true; do
  [ -f "$STOPFLAG" ] && { log "stop flag present -> exiting"; exit 0; }
  sleep "$POLL"
  PID=$(train_pid)
  if [ -z "$PID" ]; then
    sleep 15; PID=$(train_pid)
    if [ -z "$PID" ]; then
      if [ "$restarts" -ge "$MAX_RESTARTS" ]; then log "DEAD; restart cap hit -> STOP"; touch "$STOPFLAG"; exit 0; fi
      restarts=$((restarts+1))
      L=$(ls -t checkpoints/vMin1_*.pt 2>/dev/null | head -1)
      WARM="${L:-}"
      log "process DEAD; relaunch #$restarts warm=${WARM:-COLD}"
      launch "$WARM"; sleep 45; continue
    fi
  fi
done
''', encoding="utf-8", newline="\n")
print("wrote", g)

# memory note
Path("memory/project_vmin1_minimal_obs_experiment.md").write_text('''---
name: vMin1 bare-visibility obs experiment
description: Overnight experiment — obs_mode=minimal (796) + smaller net; compare to vSix4
type: project
date: 2026-07-20
---

# vMin1 — bare-visibility observation experiment

## Goal
Test whether a policy trained on **only table-visible state** (no engineered
strength / opp-outcome MC / SPR-odds crutches) can match or beat full-obs v6
late in training, while running faster per update.

## Locked keep set (OBS_DIM_MINIMAL = 796)
- hole (52) + board A (52) + board B (52)
- street (4)
- active (8) + all-in (8)
- stacks/bb (8)
- scalars: pot, to_call, min_bet, max_bet (4)
- history 32×18 (576)
- seat-exists (8)
- total commit (8) + street commit (8)
- hero–button distance (8)

**Dropped:** SPR, pot odds, bet% pot, categories, draws, pair features,
SF/flush blocks, blockers, opp-outcome MC, per-board outcome, last aggressor,
rel-pos, all v2/v7 engineered tails.

## Stem config (scripts/vMin1_guardian.sh)
- `--obs-mode minimal`
- actor **1024×3** + torso LN; critic **1024×2**
- v6 preset (mixture, q_aux, etc.)
- cold start (no warm from vSix4 — obs width mismatch guarded)
- entropy start 0.25 (same ladder plan as vSix4; manual anneal_control)
- num-envs 16384, rollout 4M (leaner VRAM than vSix4)

## Code entry points
- `encoding.OBS_DIM_MINIMAL` / `project_obs_minimal`
- `BombPotEnv(..., obs_mode=)` / `BatchedBombPotEnv(..., obs_mode=)`
- `TrainingConfig.obs_mode`, `train.py --obs-mode`
- tests: `tests/python/test_obs_minimal.py`

## Compare later vs vSix4
- wall-clock / update
- matchup EV at matched entropy floors (0.16, 0.10)
- study trash-raise% and value calibration (see project_v6_value_gap_anneal_revisit.md)
''', encoding="utf-8")
print("memory written")
