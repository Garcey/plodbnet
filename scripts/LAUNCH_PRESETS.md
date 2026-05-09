# Training launch presets

Quick-copy commands for the curriculum stages we've actually run.
Constants for distribution-shape work (clubgg bands and seat
weights) live in `scripts/train.py` (`_CLUBGG_STACK_BANDS`,
`_CLUBGG_SEAT_WEIGHTS`) — change those at the source, not via
flags.

All commands assume repo root and `.venv` already built.

Use `python -u` (unbuffered) when redirecting stdout to a log
file, otherwise Python's full-buffer mode hides progress lines
until kilobytes accumulate.

## clubgg — realistic table mix (2-6 seats, weighted bands)

Per-seat stack bands: 5% Short (1-20bb), 50% Hover (20-40bb),
18% Warm (40-75bb), 17% Big (75-150bb), 10% Monster (150-300bb).
Seat-count weights: 6:25% / 5:25% / 4:25% / 3:15% / 2:10%.

## clubgg_deep — $0.80-ante / $80-buy-in game

~2x deeper than the $0.60-ante game. Per-seat stack bands:
2% (1-20bb), 6% (20-30bb), 16% (30-40bb), 22% (40-50bb),
25% (50-65bb), 22% (65-80bb), 7% (80-120bb). Concentrated on
30-65bb (63%); E[seats 65-80bb in 6-max] ≈ 1.3. Same clubgg
seat-count weights. Per-seat mean ≈ 54.5bb.

```bash
.venv/Scripts/python -u scripts/train.py --batched \
  --hidden-dim 2048 --num-layers 4 --device cuda \
  --num-seats-range "2,3,4,5,6" \
  --stack-range "1:120" \
  --stack-dist clubgg_deep \
  --seats-dist clubgg \
  --load-checkpoint checkpoints/<prior_stage>.pt \
  --checkpoint checkpoints/run_clubgg_deep.pt \
  --num-updates 1000000 \
  --checkpoint-every-sec 300 \
  > runs/run_clubgg_deep.log 2>&1 &
```

```bash
.venv/Scripts/python -u scripts/train.py --batched \
  --num-seats-range "2,3,4,5,6" \
  --stack-range "1:300" \
  --stack-dist clubgg \
  --seats-dist clubgg \
  --load-checkpoint checkpoints/<prior_stage>.pt \
  --checkpoint checkpoints/run_forward_ev_clubgg.pt \
  --num-updates 1000000 \
  --checkpoint-every-sec 300 \
  > runs/forward_ev_clubgg.log 2>&1 &
```

## 2to6 deepstack — uniform seats, uniform 100-150bb stacks

Heavier deepstack training. Uniform 2-6 seat sampling, per-seat
uniform(100, 150) bb stacks. Warm-starts from the last
`forward_ev_2to6_20bb` checkpoint.

```bash
.venv/Scripts/python -u scripts/train.py --batched \
  --num-seats-range "2,3,4,5,6" \
  --stack-range "100:150" \
  --stack-dist uniform \
  --seats-dist uniform \
  --load-checkpoint checkpoints/run_forward_ev_2to6_20bb_3020.pt \
  --checkpoint checkpoints/run_forward_ev_2to6_100_150bb.pt \
  --num-updates 1000000 \
  --checkpoint-every-sec 300 \
  > runs/forward_ev_2to6_100_150bb.log 2>&1 &
```

## 2to6 shallow — uniform seats, fixed 20bb stacks

Earlier curriculum stage. Uniform 2-6 seats, every seat 20bb.

```bash
.venv/Scripts/python -u scripts/train.py --batched \
  --num-seats-range "2,3,4,5,6" \
  --stack-range "20:20" \
  --stack-dist uniform \
  --seats-dist uniform \
  --load-checkpoint checkpoints/run_forward_ev_hu20_6353.pt \
  --checkpoint checkpoints/run_forward_ev_2to6_20bb.pt \
  --num-updates 1000000 \
  --checkpoint-every-sec 300 \
  > runs/forward_ev_2to6_20bb.log 2>&1 &
```

## Heads-up shallow — bootstrap stage

Curriculum bootstrap: 2-seat 20bb, from-scratch (no warm-start).

```bash
.venv/Scripts/python -u scripts/train.py --batched \
  --num-seats-range "2" \
  --stack-range "20:20" \
  --stack-dist uniform \
  --seats-dist uniform \
  --checkpoint checkpoints/run_forward_ev_hu20.pt \
  --num-updates 1000000 \
  --checkpoint-every-sec 300 \
  > runs/forward_ev_hu20.log 2>&1 &
```
