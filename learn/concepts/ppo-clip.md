# Importance ratios and the PPO clip

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

Training runs in a loop: collect a big batch of hands, then update the policy on what they revealed. By update time there's a wrinkle — the hands were played by yesterday's policy, and the network being updated is already drifting away from it.

PPO handles this with bookkeeping. When a hand is collected, every action is stored with a "receipt": the probability the policy assigned to that action at the moment it acted. At update time the network re-asks the question: how likely would I be to make that same choice now? The new probability divided by the receipt is the importance ratio. Above one, the action has become more likely; below one, less. The update's whole job is to push ratios up on actions that graded well and down on actions that graded badly.

Left alone, that goes wrong fast. If one action drew a glowing grade, the optimizer would happily crank its probability toward certainty in a single update — overcommitting to what may be one lucky sample. Worse, the grades were earned under the old policy; drift too far from it and they stop describing the player the network has become.

## The clip

So PPO caps how far a ratio can profitably move in one update. Past a set distance from one, further movement earns no extra credit — and since pushing harder gains nothing, the optimizer stops pushing. Think of it as a seatbelt: it doesn't steer, and on a smooth drive you never feel it. It only locks when the update tries to lurch, holding the policy close to the data it was graded on.

One subtlety is worth keeping: the clip is one-sided pessimistic. Movement past the cap in the rewarded direction stops earning credit, but movement in the wrong direction keeps getting punished without limit. The update is quick to correct mistakes and deliberately slow to pile onto wins.

**In this project:**

- v6 uses a probability-dependent clip, keyed on the gate's old probability: rare, near-certain actions (probability close to 0 or 1) get about 10 percentage points of movement room, while 50/50 actions get a tight 5 points.
- The `pi=` number on the log line is this clipped objective, negated. Consistently negative `pi` at a few thousandths means the update is actively fitting real signal.
- `pi` hovering around ±0.001 with flipping signs means the opposite: there is nothing to fit.

## The loss line

```
ratio = exp(log π_new − log π_old)
loss  = −mean( min(ratio·A, clamp(ratio, lo, hi)·A) )
```

That whole expression is the **clipped surrogate loss** — full teaching
card: [→ clipped surrogate loss](clipped-surrogate-loss.md). The log
field `pi=` is exactly this loss (averaged over the update's minibatches).

Live rooms (vSix4): `clip_room_ext≈0.10`, `clip_room_mid` started 0.07 and
is live-tunable via `anneal_control` (widened when the mid-band KL quota
was the bottleneck).
