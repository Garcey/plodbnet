---
name: Prefer ratio-based defaults that auto-scale with rollout size
description: For knobs like minibatch size, default to a count-per-update (e.g. 32 minibatches/update) that derives from rollout-length, not an absolute number that becomes wrong at scale
type: feedback
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
For tunables that scale with rollout size (minibatch size, etc.), default to a **ratio / count-per-update** rather than an absolute integer.

**Why**: An absolute `--batch-size 256` default becomes effectively unused — at production scale (rollout_length 262 144 – 4 194 304) it's three orders of magnitude too small, and the user always overrides. Defaults that auto-scale stay correct as the workload grows. Specific instance: `scripts/train.py` had `--batch-size 256` default; user pointed out 32 minibatches per update would scale across all rollout sizes.

**How to apply**: When proposing a new default for a per-update tunable, ask "does this still make sense at 1k× the current rollout size?" If not, derive it from rollout-length (or whatever it scales with). Keep the absolute-value flag as an override, not the default.
