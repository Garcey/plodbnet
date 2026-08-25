# Learning rate and warmup

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

The learning rate is one number that scales every weight movement in every update. After the optimizer works out the direction and relative size of each weight's step, the learning rate multiplies all of it. Double it and every step doubles; halve it and everything moves half as far.

Both extremes fail in characteristic ways. Too high, and the network overshoots: each update yanks weights past the point the evidence supports, the next update yanks back, and instead of settling, the policy thrashes — in the worst case it collapses into something degenerate that later updates can't rescue. Too low, and nothing goes wrong except everything: learning is glacial, and compute is spent standing nearly still. The right rate is the fastest pace that stays stable.

## Why warmup exists

Warmup addresses a specific danger zone: the first updates after a start or a restart. The optimizer's per-weight running averages — its sense of each gradient's trend and typical size — begin as blind guesses built from no evidence at all. Full-size steps taken on those guesses are confident motion in possibly wrong directions. So warmup starts the learning rate at a fraction of its full value and ramps it linearly over the first N updates: tiny steps while the optimizer gathers evidence, full speed once its averages actually mean something.

It's how a professional handles an unfamiliar lineup. You don't fire triple-barrel bluffs in the first orbit; you deliberately play smaller, more careful poker while the reads accumulate, then widen your game once you trust them. Warmup is that first orbit — caution proportional to ignorance, and strictly temporary.

One more thing worth knowing: warmup length should match how blind the system actually is. A fresh network needs a long ramp, because the weights and the optimizer are both starting from nothing. A restart from a trained checkpoint needs far less — the weights already know how to play, and only the optimizer's averages need rebuilding.

**In this project:**

- The base learning rate is 0.00015 (written 1.5e-4 in configs and logs).
- During the ramp, the training log line carries a suffix in the style of "lr×0.90" showing the current fraction of full speed; the suffix disappears once warmup completes.
- Cold-start guardians historically warm up over 75 updates. **Live vSix4 / vMin1 ship `--lr-warmup-updates 0`** (full LR immediately on warm restart — Adam moments rebuild under full step size; acceptable because the weights are already trained). A mid-run warm restart that still wants a short ramp can set 5–10.
