---
name: Hero-rotated seat-exists mask added
description: 2026-05-06 production obs change — 8-dim structural seat-presence mask appended at offset 878; OBS_DIM 878→886. Disambiguates micro-stacked seats from padded slots ahead of heterogeneous-stack curriculum stages. Triggers retrain.
type: project
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
On 2026-05-06, immediately after the history-slot rewrite
(`project_history_chips_features.md`), an 8-dim hero-rotated
**seat-exists** mask was appended at the encoder tail.

**Problem this closes.** All per-seat blocks (`_ACTIVE_OFF`,
`_ALLIN_OFF`, `_STACKS_OFF`, `_SPR_OFF`) pad to 8 with trailing
zeros. Under uniform stacks the network can infer `num_seats` from
`stacks > 0`, but once heterogeneous stacks ship a micro-stacked
seat would look identical to a padded slot. The mask is **structural,
not stack-driven**, so a 0-chip seat still reads as "this slot is a
real seat."

```
878..886  seat_exists: slot k = 1 iff (hero + k) % num_seats < num_seats
                       (i.e. slots 0..num_seats-1 = 1, rest 0)
OBS_DIM = 886
```

**Implementation.** Pure-Python; no Rust binding change.
- Scalar `encode_observation`: `for k in range(num_seats): out[_SEAT_EXISTS_OFF + k] = 1.0`.
- Batched `encode_observation_batch`:
  `out[live_mask, _SEAT_EXISTS_OFF : _SEAT_EXISTS_OFF + num_seats] = 1.0`.
  Constant per config — same value for every live env.
- Tests: `test_seat_exists_mask_matches_num_seats` (2-seat and
  6-seat configs), `test_seat_exists_independent_of_stack` (0-chip
  hero still flags slot 0).

**How to apply.**
- Production observation-semantics change. Retrain from scratch on
  the new 886-dim encoder; do not warm-start from any pre-2026-05-06
  checkpoint (none exist on disk by design).
- The mask is technically redundant under uniform stacks (every
  slot 0..num_seats-1 already has positive `_STACKS_OFF`), so this
  feature only starts paying off once the heterogeneous-stack stages
  begin (see `project_direction.md`). Surfacing it explicitly lets
  the network learn position/seat-count semantics without entangling
  them with stack-positivity.
