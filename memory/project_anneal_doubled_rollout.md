---
name: Anneal phase uses doubled rollout-length (32768)
description: Late-stage entropy-anneal training uses --rollout-length 32768 (vs 16384 standard) — tighter per-update gradients reduce sample variance during low-entropy lock-in
type: project
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
For late-stage entropy annealing (clubgg/clubgg_deep ent ≤ 0.05),
training uses `--rollout-length 32768` instead of the project
standard 16384. Standard config: 1536 envs, 16384 rollout. Anneal
config: 1536 envs, 32768 rollout.

**Why:** User confirmed on 2026-05-08 (anneal stage 1, u0-u20)
that doubling rollout halved the visible per-update sample
variance: bonus% buckets cluster more tightly across consecutive
ticks of the same tier, and v_loss is less all-over-the-place.
At low entropy the gradient signal is what tips the policy toward
specific bet sizes; tighter gradients reduce the chance of a noisy
batch shoving the policy into a bad local minimum during lock-in.

**How to apply:**
- When launching anneal-phase runs (any time `entropy_coef ≤ 0.05`
  on at least one tier), pass `--rollout-length 32768`.
- Standard pre-anneal training stays at `--rollout-length 16384`.
- If the user explicitly says "use the standard rollout" or asks
  to revert, drop back to 16384 — this is a knob, not a hard rule.
