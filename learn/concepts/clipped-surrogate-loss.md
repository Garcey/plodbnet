# The clipped surrogate loss

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.
Read with [→ importance ratios & the PPO clip](ppo-clip.md) and
[→ log-probability](log-probability.md).*

When people say "the clipped surrogate loss," they mean the single number
the policy half of PPO is trying to drive down each minibatch. Three words,
three jobs:

1. **Surrogate** — a stand-in objective. You cannot directly maximize "win
   more chips under the new policy," because that would require replaying
   every hand with the new weights. Instead PPO maximizes a cheap proxy
   built from *old* hands: raise the probability of actions that graded
   well, lower the probability of actions that graded badly. That proxy is
   the surrogate. It is not the true poker EV; it is a local, differentiable
   estimate of "how much better would this batch look if I nudged π?"
2. **Clipped** — the seatbelt on that proxy. The importance ratio
   `r = π_new(a|s) / π_old(a|s)` is allowed to help the objective only
   inside a band around 1. Past the band, further movement in the *rewarded*
   direction earns nothing. (Movement in the punished direction is never
   clipped — mistakes stay fully expensive.)
3. **Loss** — the training code minimizes, so the surrogate is negated:
   `policy_loss = −mean(min(r·A, clip(r)·A))`. The log field `pi=` is this
   number. Negative `pi` means the update is fitting real signal; near-zero
   flip-flopping means there is nothing left to fit inside the clip band.

## The formula, one line at a time

For each decision in the minibatch:

```
r     = exp(log π_new − log π_old)     # importance ratio
surr1 = r · A                          # unclipped surrogate
surr2 = clamp(r, lo, hi) · A           # clipped surrogate
L     = −min(surr1, surr2)             # pessimistic pick, then negate
```

`A` is the advantage ([→ advantage](advantage.md)): positive means "better
than expected," negative means worse. The `min` is the pessimistic pick —
when A > 0 the optimizer only gets credit up to the upper clip; when A < 0
it only gets credit (for reducing the action) down to the lower clip, and
can be punished without bound if it *increases* a bad action.

## Why "surrogate" and not "policy gradient"

A pure policy-gradient step would be `∇ log π · A` on the *current* policy
with *fresh* rollouts every step. PPO reuses one big rollout for many
optimizer steps (2 epochs × 16 minibatches here). Reusing data without a
trust region is how you get catastrophic policy collapse — the grades were
earned under π_old, and after a few large steps they stop describing the
player you have become. The clipped surrogate *is* that trust region,
written as a loss instead of a hard constraint.

## In this project

- Code: `python/plo5bp/ppo.py`, inside `PPOTrainer.update` — `ratio`,
  `surr1`, `surr2`, `policy_loss = -min(surr1,surr2).mean()`.
- v6 does **not** use a flat ±0.2 band. It uses a probability-dependent
  gate clip: rare gates get ~10 points of movement room, 50/50 gates get
  a tight mid band (live `clip_room_mid=0.07`). See MASTER Part 6.2.
- The joint log-prob is gate + (if raise) anchor + refine, so `r` is a
  ratio over the full factored action, not just the gate.
- `pi=` on the log is this loss averaged over the update. Consistently
  negative at a few thousandths is healthy fitting; `|pi| ≲ 0.001` with
  sign flips means the clip band is full and/or advantages carry no
  usable signal.
