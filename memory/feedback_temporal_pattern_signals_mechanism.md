---
name: Temporal pattern shape constrains mechanism — warmup-only is not contention
description: When a slowdown follows a clean temporal pattern (every-time-at-start, every-N-minutes, etc.), use the shape to rule out mechanisms whose own behavior doesn't match. CPU contention is persistent or sporadic, not warmup-only
type: feedback
originSessionId: 78983dc4-2873-4db3-a0a7-cbda35e7b613
---
When a perf bug has a *shape* in time (only at start, only after N
hours, only every N minutes), match candidate mechanisms against
that shape before pitching one. Generic "noise" or "load
contention" theories carry their own implicit timing — and if the
implicit timing doesn't match, the theory is wrong even when the
mechanism is plausible in the abstract.

**Why:** On 2026-04-25 the user reported OCR ticks taking ~5s/frame
for the first 1-2 min of every session, then snapping to normal.
PPO training was running at 70% CPU. I pitched CPU contention as
the lead hypothesis. User pushed back: contention would slow
things at random times or persistently, not produce a clean
warmup-shaped curve that always resolves at the same point. The
shape argued for a per-session cold-start cost (e.g. tesseract.exe
+ eng.traineddata page-in to OS file cache), not steady-state
contention.

**How to apply:** Before proposing a slowdown cause, write down the
shape the user observed (only-at-start, intermittent, monotonic
ramp, periodic, post-deploy). Then write down the shape your
candidate mechanism would itself produce in isolation. If they
don't match, drop the candidate. Specifically: contention has no
"warmup curve" — it tracks the contender's load, not session
elapsed time. Cold-cache costs DO have a warmup curve and resolve
once the working set is paged in. Use the temporal shape as a
positive constraint, not just a sanity check.

This is a sibling to `feedback_specificity_signals_mechanism.md`
(which entity is hit) — same idea on the time axis (when it hits).
