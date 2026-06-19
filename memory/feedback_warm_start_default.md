---
name: Default to warm-start when launching training
description: Always assume the user wants `--load-checkpoint` from the most recent healthy snapshot of the prior curriculum stage; from-scratch only when explicitly requested
type: feedback
originSessionId: 5fea7462-79e9-4e59-936c-fb00309f5ed1
---
When launching a training run that extends a prior stage of the
curriculum, default to warm-starting from the prior stage's latest
healthy snapshot via `--load-checkpoint`. Don't ask whether to
warm-start or train from scratch — assume warm-start.

**Why:** The user told me on 2026-04-26 to "always assume I want a
warm start unless stated otherwise." Curriculum bootstrapping is the
intended workflow — each stage builds on the prior policy.

**How to apply:**
- New seat-count / stack-depth / format expansion → warm-start from
  the prior stage's last good snapshot.
- Use the numbered snapshot (e.g. `..._6353.pt`) over the live
  `..._<run>.pt` if the live one was clobbered or doesn't exist.
- From-scratch only when the user says so explicitly, or when the
  prior stage was itself unhealthy.
