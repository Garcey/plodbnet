---
name: High aggression-bonus runs have volatile entropy by design
description: Don't read swinging H during c≥2 runs as collapse — the bonus pushes aggressive exploration and PPO sorts out good-vs-bad aggression on its own
type: feedback
originSessionId: 7b62bece-8e86-454f-89e8-27566d72af24
---
When training with a high pot-fraction aggression bonus (c≥2, especially
c=3+), entropy will swing volatilely across rollouts (e.g., 0.4 → 0.01
→ 0.04 over 30 updates). This is expected, not pathological.

**Why:** PLO5 double-board bombs are complex; the c-bonus deliberately
pushes the policy into aggressive lines so it gets enough samples of
each to learn which earn EV and which lose. The volatility is the
sorting process — bonus pulls toward raise, value head punishes the
bad raises, policy oscillates while it figures out the conditional
structure. A flat-stable H here would mean the bonus isn't doing its
job.

**How to apply:** Don't kill the run or propose remediation (entropy
coef bumps, bonus reductions) on the first H drop. Only flag if the
collapse is *sustained* with no oscillation back up — same ≥100-update
window from `feedback_training_patience.md`, but here you should
specifically expect the swings, not be surprised by them. Bonus values
near zero in a single rollout don't mean the policy stopped raising
overall — it's a per-rollout snapshot that depends on the sampled
seats/stacks.
