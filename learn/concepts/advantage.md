# Advantage

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.
See also [→ TD, bootstrapping & GAE](td-bootstrapping-gae.md) and
[→ advantage normalization](advantage-normalization.md).*

**Advantage** answers one question about a single decision:

> "How much better or worse was this action than what I usually get from
> this kind of spot?"

Formally `A(s,a) = Q(s,a) − V(s)` — the action's value minus the state's
value. Positive advantage: the action beat the baseline; raise its
probability. Negative: it underperformed; lower it. Zero: no lesson.

This is the anti-results-oriented number. Winning a 5% cooler can still
carry a *negative* advantage if the call was −EV; losing a cooler can
carry a *positive* advantage if the shove was +EV. The critic's job is to
strip luck and position-baseline out so the policy learns the decision,
not the runout.

## How this project builds A

Two estimators, same λ-recursion shape (`γ = 1`, `λ = 0.95`):

- **GAE** — temporal differences on the critic's V:
  `δ_t = r_t + V(s_{t+1}) − V(s_t)`, then
  `A_t = δ_t + λ·A_{t+1}`.
- **VRPO** (v6 default) — same recursion on
  `δ⁺_t = r_t + V^π(s_{t+1}) − Q(s_t, a_t)`, using the dueling Q head.
  At a zero-init Q head, Q ≡ V and VRPO is bit-identical to GAE (warm-start
  safe). As Q calibrates, it replaces "sampled next-action luck" with an
  expectation at mixed nodes.

Value-head *targets* are always `returns = A_GAE + V` (even under VRPO).
The policy sees the VRPO advantages when that estimator is on.

## Post-processing

Raw advantages are normalized to mean 0 / std 1 per sub-rollout, clamped
to ±8σ, then **re-normalized across the full 30-config batch**. That
global step couples tiers: a noisy deep tier inflates σ for everyone and
shrinks every tier's signal — the coupling the freeze postmortem named.

## In this project

- Collector: `python/plo5bp/rollout.py` (GAE / VRPO backward scan on
  terminal flush).
- Consumer: `ratio * advantages` inside the clipped surrogate
  ([→ clipped surrogate](clipped-surrogate-loss.md)).
- The centralized critic peeks at all hole cards so V/Q are sharper than
  anything the actor could estimate alone
  ([→ centralized critic](centralized-critic.md)).
