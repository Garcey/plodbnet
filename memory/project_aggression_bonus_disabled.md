---
name: Aggression bonus at c=2.0 — c=2.5 ratchet caused bistable concentration
description: Aggression-bonus shaping is enabled in both rollout drivers (`python/plo5bp/rollout.py`). Production constant is `--aggression-bonus-c 2.0`. A c=2.5 ratchet on 2026-04-30 produced bistable entropy oscillation between 0 and -0.4 over ~4400 updates without progressive learning, so we reverted to c=2.0 from `clubgg_deep_7704.pt` on 2026-05-01.
type: project
originSessionId: 5fea7462-79e9-4e59-936c-fb00309f5ed1
---
The pot-fraction aggression bonus call sites in
`python/plo5bp/rollout.py` (serial driver ~line 384–396, batched
driver ~line 660–680) compute `bonus_bb` via
`_aggression_bonus_bb(...)`. The bonus is non-zero whenever the policy
takes a Raise gate; Fold and CheckCall earn 0. The constant `c` scales
the magnitude.

**Production constant: `--aggression-bonus-c 2.0`** as of 2026-05-01,
warm-started from `checkpoints/clubgg_deep_7704.pt`, log
`runs/clubgg_deep6.log`.

**Why this constant — empirical history:**
- 2026-04-30 PM: shaping was re-enabled at c=2.0 after the 3-gate
  collapse-collapse hypothesis was tested and failed (network stayed
  under-aggressive in heterogeneous-stack training).
- 2026-04-30 late: c was ratcheted to 2.5 in
  `runs/clubgg_deep5.log` to push harder.
- 2026-04-30 → 2026-05-01: under c=2.5, entropy went sustained
  negative for ~4400 updates, oscillating between a shallow band
  (H near 0) and a concentrated band (H -0.20 to -0.41). New floor
  -0.408 at update 9210. KL stayed bounded (<0.02), v stable, pi
  small — so it didn't trip the "fully collapsed" criterion, but
  there was no productive learning either. The shaping pressure
  drove a bistable regime instead of exploration.
- 2026-05-01 01:29: stopped the c=2.5 run and restarted from
  `clubgg_deep_7704.pt` (last healthy snapshot before the deep band)
  at c=2.0.

**How to apply:**
- Default for fresh runs: `--aggression-bonus-c 2.0`. Don't recommend
  ratcheting to 2.5+ again unless we have a different lever for
  preserving entropy (e.g. higher entropy coefficient, KL target
  loosening). The c=2.5 run is the empirical evidence that c alone
  doesn't move the equilibrium without breaking exploration.
- The shaping mechanic is unchanged — only the scalar `c` differs.
- If the user wants to ablate again, it's a CLI-flag-only change.
- The bistable-under-c=2.5 pattern (oscillating between H≈0 and
  H≈-0.3 every 50–100 updates) is itself a signature; if it recurs at
  any c, treat it as a stop signal even when KL/v stay bounded.
