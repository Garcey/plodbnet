---
name: Don't call training collapse on <500 updates
description: 200 updates is too few to diagnose a training run; PPO normally has noisy early v_loss / entropy that settles after several hundred updates
type: feedback
originSessionId: 5fea7462-79e9-4e59-936c-fb00309f5ed1
---
Don't declare a training run "collapsed" or "unhealthy" based on the
first 100-200 update lines. Early-update noise can include big v_loss
swings and entropy oscillation as the value head finds scale and the
policy explores. Wait at least ~500 updates before calling it broken.

**Why:** On the Apr 26 forward-EV launch I called collapse at 200
updates because v_loss was bouncing 90 → 44033 and joint entropy went
negative. The user pushed back that this was premature — early PPO
is naturally noisy; the trend matters more than the spread. We don't
have a baseline of what stub.pt's first 200 updates looked like.

**How to apply:** Health-check criteria from CLAUDE.md (`v_loss
stable, entropy not collapsing/blowing up, approx_kl < 0.05`) apply
to the converged tail, not the first 200 updates. When monitoring,
watch for sustained pathology over a window (≥100 updates) before
concluding the run is unhealthy. If unsure, ask before killing.

**Specifically for entropy:** H dropping (even sharply, e.g.,
0.63 → 0.30 in one update) is normal — it means the model is
starting to converge on something. **H=0.3 is not low.** Don't
flag entropy decay as concerning unless it's much more severe
(approaching 0) AND sustained across many updates. Even a 50%
single-update drop is within normal PPO behavior. Confirmed
2026-05-11 on the `optimized` run at u12.
