---
name: Aggression-bonus curriculum strategy
description: User's training philosophy — escape passive local equilibrium with high aggression-bonus c, then anneal to 0 to settle into aggressive equilibrium. No solver exists, so user defines "true equilibrium" as the aggressive local minimum aligned with their poker beliefs and player pool.
type: project
originSessionId: 5fea7462-79e9-4e59-936c-fb00309f5ed1
---
User's stated training philosophy for the bomb-pot PPO model:

PLO5 double-board bomb-pot has no equilibrium solver, so "true
equilibrium" is unreachable analytically. The vanilla PPO self-play
trajectory tends to settle into a *passive* local equilibrium
(checking/calling spots that the user judges to be clear bets or
check-shoves). User's goal is to push the policy out of that local
min and into an *aggressive* local equilibrium that better matches
the player pool (clubgg) and his own beliefs about optimal play.

Method: ratchet `--aggression-bonus-c` upward as a forcing
function (0.10 → 0.50 → 2.00 escalation observed 2026-04-29), then
anneal back to 0 once aggression is established. The user accepts
that intermediate stages may be "way over aggressive" — the
expectation is that c=0 retraining from the aggressive checkpoint
settles into a stable aggressive basin.

**Why:** Two reasons:
1. No solver, so equilibrium is policy-relative. User's
   aggressive-pool target is the meaningful one for live play.
2. **Node coverage**. Wider aggression forces wider defense, which
   forces self-play to explore deep-stack multi-way nodes that the
   passive equilibrium never visits — e.g. 3-handed pot-flop-bet,
   two callers, turn check-shove. The current passive policy is
   so locked in that it folds nut draws with board coverage on
   double boards because its prior is "betting range = nuts on
   both boards." That value-head knowledge of rarely-explored
   nodes should persist when c is annealed back, even if the
   aggression itself softens.

**How to apply:**
- Don't push back on "aggressive" reward shaping as if it's
  bias-toward-EV-leak. The user knows c>0 is biasing; the bias is
  the point.
- When the user asks to bump c, propose the launch immediately;
  flag risks once but don't litigate the strategy.
- After the high-c phases, expect a c=0.0 anneal phase
  warm-started from the aggressive checkpoint. That is the
  intended endpoint, not the high-c training itself.
- Promotion-to-UI checkpoints during high-c phases are for
  *behavior comparison* (does the model now bet/raise the spot?),
  not for "production play."
