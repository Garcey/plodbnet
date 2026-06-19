---
name: Retroactive aggression bonus is pot-relative
description: 2026-05-07 reward-shaping change — `_apply_retroactive_bonus` scales bonus per qualifying step by pot-at-decision in bb. `c` semantics changed from absolute bb-per-step to bb-per-bb-of-pot.
type: feedback
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
`_apply_retroactive_bonus` (`python/plo5bp/rollout.py`) now applies
**`bonus_t = c * pot_at_decision_bb`** per qualifying step instead
of a flat `c` bb. Pot-at-decision is the total committed across all
seats just before the actor moves, captured per-step in a parallel
`pot_trajs` list alongside `cost_trajs`.

**Why:** the full_mix_c2.5 run (u900) showed the policy was
aggressive on the flop but **passive on the river**, including
checking back nutted-on-both-boards holdings. Hypothesis: river
chip-delta variance is large vs the flat bonus, so the bonus was
loud at the flop (small pots, small variance) and inaudible at the
river (big pots, big variance). Pot-relative scales the bonus to
the variance it competes against — louder where the policy is
under-betting.

**How to apply.**
- `--retroactive-bonus-c` semantics changed: it's now bb-of-bonus
  per bb-of-pot per qualifying step. **Old absolute-c values must
  be scaled down by ~30–100× depending on stack depth.** The prior
  c=2.5 absolute would translate to roughly c=0.05–0.1 here.
- Aim for printed `bonus` field ~0.3–0.5 bb mean (matches prior
  healthy magnitude). The 10-update smoke at c=0.05 produced
  ~0.10–0.26 on shallow stages — c=0.10 is a reasonable next try
  for a full curriculum stage.
- Eligibility logic is unchanged: hero pot share ≥ 50% required;
  RAISE always qualifies, CHECK_CALL only at exact 50% and only
  when chips were committed (`costs[t] < 0`).
- The `bonus_pct` field was replaced by `bonus%(F/T/R)` — three
  per-street percentages (flop / turn / river). Numerator: learner
  steps that received a bonus on that street; denominator: total
  learner steps on that street. Bomb pots have no preflop action so
  3 buckets cover everything. Diagnostic intent: confirm the river
  bucket rises and the flop bucket drops as the policy adapts to
  the pot-relative scaling. Magnitude still comes from `bonus`
  (mean across all qualifying-or-not steps).
