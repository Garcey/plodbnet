---
name: Curriculum training — start simple, scale up
description: Bootstrap fresh training runs from heads-up shallow before extending to multi-seat / deep-stack so the network learns the basics before facing high variance
type: feedback
originSessionId: 5fea7462-79e9-4e59-936c-fb00309f5ed1
---
When starting a fresh training run, begin with the simplest format
(2-seat, 20bb stacks, 3bb ante → 17bb effective on flop) so the
network can learn fundamentals quickly. Only after that's stable
should we widen to multi-seat and deeper stacks.

**Why:** Heads-up shallow play has the smallest action space, the
narrowest variance, and the cleanest signal. The network learns
positional & equity-driven decisions before facing the combinatorial
complexity of 6-handed deep-stack pots. Wider distributions
training-from-scratch (1-300 BB ClubGG bands × 2-6 seats) overwhelm
a random-init value head — which is what blew up the Apr 26 run
(v_loss bouncing 100×, joint entropy going negative).

**How to apply:** For any from-scratch training launch, default to
`--num-seats-range 2 --stack-range "20:20"` first. Only widen
after the run shows stable v_loss and bounded approx_kl across
≥1000 updates. Save the wider production distributions
(`_CLUBGG_STACK_BANDS`, `_CLUBGG_SEAT_WEIGHTS` in `scripts/train.py`)
for later curriculum stages — don't delete them.
