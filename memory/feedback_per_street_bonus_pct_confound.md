---
name: Per-street bonus% has a structural denominator confound
description: F/T/R bonus% can't be compared cross-street; river bucket is structurally inflated because fewer seats remain. Track per-bucket trend over time, not cross-bucket levels.
type: feedback
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
The training log's `bonus%(F/T/R)` field is per-street
`bonus_steps / learner_steps`. The denominator (total learner steps
on that street) is **not the same shape across streets**:

- River sees fewer learner steps because fold-outs reduce seat
  count by river.
- The retroactive-bonus eligibility gate (hero pot-share ≥ 50%) is
  also more likely to fire when fewer seats remain to split the
  pot, compounding the effect.

So a higher river% than flop% at any single update **does not** by
itself mean the policy is more aggressive on the river. The metric
is structurally river-biased.

**How to apply.**
- Don't read `river > flop` at one timestamp as evidence of
  policy aggression shape. The shape was already there from the
  metric's denominator.
- Compare each bucket against itself across updates. "River bucket
  rose from 34% to 55% between u10 and u500" is a real signal of
  policy adapting; "river > flop at u10" is mostly the metric.
- For an interpretable cross-street aggression signal, the right
  metric would normalize by hands-where-the-seat-was-still-in
  rather than learner-step count, or simply track per-street raise
  frequency independent of the bonus gate. That isn't logged
  today — flag if the user asks for cross-street comparison.
