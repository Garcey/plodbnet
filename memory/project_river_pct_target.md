---
name: River bonus% target — 40% floor across seat counts
description: User's eyeball target for river bonus% — 40% minimum across all seat counts; persistent shortfall is the trigger for re-enabling the aggression bonus
type: project
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
User's target equilibrium: **river bonus% ≥ 40% across every
seat-count bucket**. If a run sits below 40% on river for any
seat count over a sustained window, it's a candidate for
re-enabling the aggression bonus (currently disabled per
`project_aggression_bonus_disabled.md`).

**Why:** Stated 2026-05-11 mid-`optimized` run. At u80, the
per-seat river% landed ~46/38/33/26/23 for 2/3/4/5/6 seats —
only 2seats clears the bar. User views this as the model
sitting at a near-passive equilibrium rather than playing
target-aggressive poker.

**How to apply:**
- When summarizing training trends, call out per-seat river% vs
  the 40% floor explicitly.
- Don't reflexively re-enable the bonus on a single update or
  short window — match the "wait for sustained pathology"
  patience norm (`feedback_training_patience.md`) and the
  "passivity may be correct" caveat (`feedback_passivity_may_be_correct.md`).
- The decision point is the user's, not mine: surface the data
  cleanly and let them call it.
