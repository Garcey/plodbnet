---
name: Prefer inference / gap-fill over erroring on missing data
description: When a pipeline could plausibly infer what was missed, don't halt on the first gap — reconstruct from surrounding state.
type: feedback
originSessionId: 229c3164-a12b-4cc5-885b-789d2200234b
---
When a real-time pipeline (OCR, event stream, diff-based reconstruction)
misses an observation, don't throw an error, drop to display-only, or
stop the loop. Try to infer what happened from surrounding state deltas
and proceed.

**Why:** User stated directly when discussing OCR event reconstruction:
"if it misses someone betting and then the next player calls, the first
bet is almost certainly still showing. Or if it misses someone checking,
gets to the next person who bets, it should deduce that the previous
player checked." They also preferred increasing poll rate (500ms → 200ms)
over tolerating lost events.

**How to apply:**
- For frame-diff or event-stream reconstruction, walk the expected
  actor/item queue and explain state deltas by filling in the gaps
  (CHECK if no facing bet, CALL/BET/FOLD inferred from committed/stack
  deltas, etc.).
- Surface a warning rather than raising, and let the next observation
  correct the state.
- Suggest tightening the polling cadence when gaps are frequent, not
  relaxing the invariants.
- This is specifically about observational/inference pipelines, not
  user-input paths where strict validation is appropriate.
