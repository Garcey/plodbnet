---
name: Project-standard training defaults are baked into train.py
description: train.py defaults match project standard (1536 envs, 262144 rollout, 2048×4, --batched, --device cuda, --log-every 1, --checkpoint-every 5, block-rotation cycle); no need to override these on every launch
type: feedback
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
As of 2026-05-09 the train.py argparse defaults match the project's
canonical training config:

- `--num-updates 100_000_000` (train indefinitely)
- `--hidden-dim 2048 --num-layers 4`
- `--num-envs 1536 --rollout-length 262144`
- `--num-seats-range "2,3,4,5,6" --stack-range "1:300"`
- `--block-rotation "clubgg:0.1,clubgg_deep:0.1,deep:0.2"` `--block-size 50`
- `--entropy-coef 0.1` (already default)
- `--device cuda`
- `--batched` (use `--no-batched` to opt out)
- `--log-every 1 --checkpoint-every 5`

**Why:** Earlier this session the user pointed out that the prior
defaults were wrong; rather than continue passing identical flags on
every launch, the user asked to bake the run's settings as the
defaults. This memory replaces the prior note that demanded
`--num-envs 1536 --rollout-length 16384` on every launch. The new
canonical rollout-length is **262144**, not 16384.

**How to apply:**
- A bare `python -u scripts/train.py --checkpoint <name>` is now a
  valid project-standard launch.
- Only pass flags when the run *deviates* from the standard
  (e.g. `--no-batched`, a different stack range, a warm-start, a
  custom block-rotation, anneal-phase entropy override).
- If you find yourself re-typing all the standard flags from
  memory, you're working off stale guidance — trust the train.py
  defaults.
