---
name: Opp-vs-hero outcome combo features added
description: 2026-05-09 production obs change — 12 opp-vs-hero scoop/quarter rates per k=2,3,4 hole sizes appended; OBS_DIM 946→958; triggers retrain
type: project
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
12 new observation scalars in `[0, 1]` appended to the encoder, indexed
`[k][outcome]` for `k ∈ {2, 3, 4}` opp-hand sizes and outcomes
{SCOOP_OPP=0, QUARTER_OPP=1, SCOOP_HERO=2, QUARTER_HERO=3}. Each value
is the fraction of unseen-deck k-card subsets where the opponent
produces that outcome, evaluated under PLO5 rules at the **current
board rank** (no runout sampling). **Sampling (current, verified
2026-06-19): k=2 exhaustive; k=3 AND k=4 MC=1024** — k=3 was
downgraded from exhaustive to MC for speed (`engine.rs:1068`). The MC
PRNG is **seeded deterministically from (hero_seat, street, hero_hole,
both boards)** (`engine.rs:1012`), so the estimate is REPRODUCIBLE —
it does NOT break the bit-exact obs contract. Now at **OBS_DIM 991,
feature at offset 978** (`bindings.rs:1882 OPP_OUTCOME_OFF`); computed
in parallel across envs in the batched encoder (`bindings.rs:1561`,
flagged there as "the expensive per-env work"). **Measured cost:
~430µs/call on the flop = 94% of the engine `observation_dict` build**
(local microbench `scripts/bench_opp_outcome.py`, 2026-06-19) — by far
the dominant per-decision CPU cost. (Old ~0.27ms note was pre-k=3-MC.)

**Why:** PPO signal for marginal hero-CALLS in PLO5 double-board is
structurally noisy under self-play (see
`project_double_board_aggression_signal`): chops dominate, scoops
self-erode under competent folds, hero needs ~2/3 board win rate to
call profitably for half-pot ≈ chop. Network was inferring combo-
domination from category + draws + boards alone. These features
collapse that inference into a sharp learnable signal. Conceptually
distinct from the equity-vs-random features removed 2026-05-06: those
were 2-scalar MC win-rate over full runouts (fuzzy, mixing made-hand
strength with runout variance); these are combinatorial cardinalities
at the **current** board, broken out per hand size and per outcome.

k=5 dropped intentionally — opp uses exactly 2 cards per board in
PLO5, so across both boards opp ever plays at most 4 unique
hole-cards; k=5 features add redundant compute on the most expensive
bucket.

**How to apply:** triggers retrain. After retrain, watch live play
for whether marginal-call frequency rises (the policy currently folds
strong hands rather than calling) and whether river R% on bet-call
lines moves. If marginal-call frequency does NOT increase materially,
the feature isn't paying for itself and should be reconsidered.

**Removing/resizing the 12 dims is an OBS-SCHEMA change →
checkpoint-incompatible → cold start.** Mid-campaign that throws away
the matured v2 model + the whole entropy-stability campaign, which
dwarfs any per-update wall-clock saving. To cut its (dominant) cost
WITHOUT a schema change: lower the k=3/k=4 MC sample count
(1024→~256 ≈ 2× faster, ~1–3% noisier, fine-tune-safe, re-bless the
encoder golden tests) and/or add exact closed-forms for the common
textures (monotone/paired). Removal is the right *experiment* only at
a deliberate cold-start, run as an A/B (with vs without the dims) on
the aggression metrics — it attacks a structural signal-to-noise
problem (see [[double-board-aggression-signal]]), not a sample-size
one, so more RunPod rollout likely does NOT fully substitute.
