---
name: Late-stage entropy annealing plan
description: Strategy for driving raise-size lock-in late in training — anneal in stages, exploit block rotation as observation surface, decouple gate vs raise-size entropy only if Beta-head collapse recurs
type: project
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
Plan for late-stage training (~5k–20k updates from now): stage-anneal
`entropy_coef` 0.2 → 0.1 → 0.05 → 0.02 with revert-on-collapse, holding
each stage long enough to verify per-tier stability before dropping
further. Build a decoupled gate-vs-raise-size entropy mechanism in PPO
only if Beta-head collapse recurs at low ent.

**Why:** At ent ≥ 0.1 the Beta raise-size head doesn't differentiate
sizes — defaults to ~half-pot (the distribution's prior), so the
policy never locks in to per-state bet sizing. Lowering entropy is
required to get sizing lock-in. But historical low-ent runs (notably
the c=2.5 ratchet attempt 2026-05-01) collapsed in deep stacks where
chip-delta variance overwhelmed the policy gradient. The retroactive
pot-relative bonus c=0.15 added 2026-05-07 provides directional reward
signal that *should* make low-ent more stable than before, but this is
unverified — the collapse risk is real.

**How to apply:**
- Don't drop `entropy_coef` in one shot. Stage-anneal with explicit
  hold-and-watch periods at each level.
- Use the block-rotation observation surface (introduced 2026-05-08
  via `--block-rotation` in `scripts/train.py`). Per-tier H tells you
  which stack depth is collapsing first; hold a fragile tier's
  `entropy_coef` higher while annealing the others. Per-tier coef
  control already exists in the rotation spec (`tier:ent_coef`).
- If collapse signals come from the **Beta head specifically**
  (sizing variance spikes, not gate flicker), implement decoupled
  entropy in PPO (`network.evaluate()` returns gate_entropy and
  raise_entropy separately) before forcing further global annealing.
  This is the "raise-size aggressively, keep gate stable" lever.
- Order of attack: try plain stage-annealing first under the new
  pot-relative bonus regime; only build the decoupled-head path if
  the same deep collapse comes back.

## Empirical update (2026-05-25, ~u1500)

The strategic frame above is now in active execution. Per user:

**Gate strategy is well-formed.** The high-entropy regime was
explicitly important for game-tree exploration. Anecdote: an early
spot wanted to check-fold a straight flush at ~11% call / 89% fold
— a "fold to aggression without board coverage" heuristic that
ignored the nuts. Under high entropy the policy explored its way
out; the same spot is now a mix of raise and call with 0% folds.
Had entropy been lower earlier, it could have locked into 100% fold
and never escaped. **This is the load-bearing argument for why the
historical high-entropy regime was right** — do not retrospectively
second-guess it.

**Active problem: Beta raise-size head is uniform at (1.0, 1.0)** in
the majority of spots (some exceptions). This is the "doesn't
differentiate sizes" prediction from the original Why above, now
empirically confirmed at u1500. Lowering entropy is the lever to
get bet-size lock-in.

**Near-term direction (next 1-2 weeks):** continue progressively
lowering the block-rotation entropy floors. Phase 3 (active since
2026-05-25) is at 0.07/0.09/0.12 (clubgg/clubgg_deep/deep), down
from phase 1's 0.09/0.11/0.15. Expect further drops in subsequent
relaunches.

**Hypothesis for cascading benefit:** the real player pool uses
pot-sized bets in the vast majority of spots. The current network
faces random sizing from its own raise head, which is unrealistic
texture. Once the Beta head sharpens to typical sizes (likely
pot-ish), self-play exposes the network to larger bets more often,
which should help fix the remaining deep-stack overvaluation issue
(see below).

**Known residual issue (don't treat as schedule failure):** in
deep river nodes with second-nut flush, the network overvalues hero
when opponent action clearly implies the nut flush. This is the
specific deep-stack blind spot the new sizing exposure is hoped to
address. Other than this, no recurring blunders the user is seeing.

**How to apply:**
- The pot-relative bonus is at c=0 in phase 3 (no aggression bonus
  this run) — bet-size lock-in is being driven by entropy schedule
  alone, not reward shaping.
- If Beta head sharpens but the deep-river second-nut-flush issue
  persists across a few hundred updates of low-entropy training,
  that's the trigger to revisit reward shape, not just keep
  lowering entropy.

## Convergence strategy (2026-05-28, replaces uniform 0.02-step rule)

When lowering entropy further, **step the deeper tiers down faster
than the shallow ones** so the three tier floors converge over time.
Examples of the cadence:
- Phase 1: 0.09 / 0.11 / 0.15 (spread 0.06)
- Phase 3: 0.07 / 0.09 / 0.12 (spread 0.05) — uniform 0.02 step
- Phase 4 (planned): 0.06 / 0.075 / 0.10 (spread 0.04) — clubgg
  Δ0.01, clubgg_deep Δ0.015, deep Δ0.02

**Why:** The original reason for tier-differentiated entropies was
that the deep-stacked strategy was collapsing to passivity — chip
delta variance per decision was overwhelming the entropy gradient,
so deep needed more entropy to keep exploring. Now that the gate
strategy is established across all tiers (per the 2026-05-25
empirical update above), that protection is no longer load-bearing
in the same way. Deep can step down faster because:
- The gate isn't at risk of re-collapsing into passive folding.
- The remaining open problem is Beta-head sharpening, which needs
  the deep tier's entropy to drop to make sizing differentiate.
- Keeping the deep tier's floor much higher than the shallow tiers
  would mean Beta-head sharpening lags in exactly the spots where
  bet-size precision matters most (deep river decisions).

**How to apply:**
- Default lower-entropy step from any current schedule: clubgg Δ0.01,
  clubgg_deep Δ0.015, deep Δ0.02. Confirm with user before launching.
- Don't propose a single uniform step like "subtract 0.02 from all
  three" — that maintains the spread instead of closing it.
- If deep collapses on a tier-transition (kl spike >0.05 sustained
  past the first-touch update, v_loss > 4x shallow tiers), revert
  the deep step only and hold the shallow tiers' new floors.

**Multi-phase trajectory:** the user's plan is to step down repeatedly
over the next several phases until the three tier floors are
near-converged in the **~0.025 / 0.03 / 0.035** territory (spread
~0.01). Don't treat phase 4 as the end state; expect 5-7 more
similarly-shaped reductions after it, each closing the spread a
little more.
