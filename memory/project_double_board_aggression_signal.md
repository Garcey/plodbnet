---
name: Why double-board chops weaken the aggression signal
description: Theoretical reason the PPO signal-to-noise for aggression in PLO5 double-board is structurally weak — chops dominate, scoops are rare and self-eroding, so persistent low aggression under a directional bonus may be near-optimal
type: project
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
User's reasoning (2026-05-08, ~u4070 of block-rotation widerange
run with `c=0.15` pot-relative bonus): the PPO gradient for
aggression in PLO5 double-board is structurally weak.

In double-board, aggression with marginal-to-medium hands lands in
three buckets:
1. **Chop** (most common) — bet $x into $x = win half the pot;
   check-down = win half the pot. Aggression EV ≈ 0. No gradient.
2. **Lose** (semi-often) — opponent had it, hero pays off bigger.
   Aggression EV negative. Gradient pushes *away*.
3. **Scoop** (rare, and self-eroding) — opponent folds scoopable
   hands against good play, so the upside collapses against
   competent opponents in self-play. Aggression EV positive but
   small in expected magnitude.

Net: the gradient for aggression is small and noisy, while the
gradient for passivity is small and noisy in the *opposite*
direction. The bonus (`c * pot_at_decision_bb` per qualifying
RAISE step, gated on hero pot share ≥ 50%) is what tips the scale.

**Implication:** if even with the bonus + many envs + long
rollouts the network's learned aggression rate remains modest,
this is plausible evidence that **modest aggression is near-
optimal for PLO5 double-board**, not a training pathology. The
bonus is already gated on winning hands (share ≥ 50%), so it's
directionally correct; persistent low R% under it is a stronger
signal than low R% under a blind bonus would be.

**How to apply:**
- When the user or I observe "low" river aggression, do NOT
  default to bumping `c`. The structural argument above suggests
  that might just be moving the policy off-optimal in the wrong
  direction.
- The right test for "is aggression actually correct here" is
  live play vs. competent humans (the user's plan), not raising
  the bonus until R% hits some eyeball target.
- If live play exposes specific spots where the network checks
  back clearly +EV bets, that's a targeted signal — not a license
  to globally bump aggression.
