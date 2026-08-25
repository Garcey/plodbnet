# Reward accounting

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

Training never sees dollars. Engine chips are integers with
`cfg.bb = 10_000` chips per big blind; every reward and value target is a
**bb float**.

## When money moves

- **Mid-hand.** The environment's raw step reward is 0. The collector
  turns `commit_delta` (chips the actor put in this decision) into a
  per-step *cost* in bb — negative numbers as you put money in.
- **Terminal.** At hand end, each seat receives its **gross pot share**
  (showdown or folds). A folder's terminal payout is 0; the chips they
  already put in were charged at commit time.
- **Early all-in.** If the hand freezes with ≥2 live seats before river,
  terminal rewards come from `payouts_ev(64, seed)` — 64 seeded runouts
  averaged — so runout luck is replaced by its expectation
  ([→ Monte-Carlo & variance](monte-carlo-variance.md)). Fold-outs and
  river closes use exact payouts.

## The fold-is-zero identity

Because sunk chips were already charged as costs, the *forward* value of
folding is exactly 0 at every node. That is not a modeling choice; it is
an accounting identity. Consequences:

- `Q(s, fold)` has a free perfect label (the Q-head fold supervision).
- Advantages on fold rows teach "was folding better than the alternative
  baseline," not "did I lose the chips I already put in."

## What is *not* reward

- **`bonus%(F/T/R)`** on the log is an outcome-dependent *diagnostic*
  (raises-that-won + calls-that-won per street). Reward for that path is
  off (`bonus=+0.0000`); do not treat bonus% as the training signal.
- Display-head EV in the UI is a network estimate, not a payout.

## In this project

- Collector cost path: `python/plo5bp/rollout.py` from `commit_delta`.
- Units invariant is Part 0 of MASTER — nothing in training ever sees $.
