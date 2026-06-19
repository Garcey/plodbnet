---
name: When upstream guards don't fix a persistent bug, the bug is downstream
description: After a plausible upstream fix ships and the symptom persists, stop adding more upstream guards — trace every consumer of the signal, including implicit ones like auto-advance loops
type: feedback
originSessionId: 78983dc4-2873-4db3-a0a7-cbda35e7b613
---
When Fix N (corroboration guard), Fix P (zero-on-zero guard), and
Fix Q (sub-facing-bet raise guard) each plausibly addressed the
phantom-fold bug by tightening the reconstructor's event
derivation in events.py but the bug kept reproducing, the real
cause was downstream — in server.py's `_auto_fold_sitting_out`,
which used `sitting_out_seats` (including mid-hand folds) and
advanced `current_actor` past the seat whose FOLD was about to
replay. Three fix cycles were wasted because each iteration
stayed in events.py.

**Why:** On 2026-04-24, after three reconstructor-side fixes
failed to eliminate the phantom BTN fold, only full-flow
instrumentation revealed the defect lived past the reconstructor.
`action_log` entries carry `{gate, chips}` with NO seat
identifier — `step_hybrid` applies to whoever the engine says is
current_actor. An auto-advance loop that silently pre-folded the
intended target seat caused a valid upstream signal (Castor's
real fold) to land on the wrong seat (BTN). No upstream guard
could catch a bug downstream of a valid signal.

**How to apply:** When a plausible fix ships and the bug
persists, don't reach for a tighter upstream guard first.
Instrument the full pipeline: derivation → storage → replay →
effect. Log at each boundary, or at minimum trace every reader
of the signal — especially **implicit consumers** like
auto-advance loops, replay drivers, or retirement/skip
mechanisms that read shared state. If the upstream signal looks
correct but downstream state is wrong, the defect is in a
consumer, not the producer. Map every place the signal is
read before hypothesizing more upstream causes.
