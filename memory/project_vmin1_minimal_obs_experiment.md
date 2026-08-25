---
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
