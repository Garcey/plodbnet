---
name: Cross-board interaction features added
description: 2026-05-06 production obs change — +28 dims (13 shared-rank mask + 12 hero-involved per-suit cross-board flush + 3 hero-rank-pair-aggregated cross-board straight). OBS_DIM 918→946. Triggers retrain.
type: project
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
On 2026-05-06, immediately after the per-seat commit / aggressor /
button-distance pass (`project_commit_aggressor_button_features.md`),
28 cross-board interaction dims were appended at the encoder tail.

```
918..931  shared rank mask: slot r = 1 iff rank r appears on BOTH
          boards (board-only, no hero gating)
931..935  per-suit flush MADE on both: hero ≥2-of-s AND ba ≥3-of-s
          AND bb ≥3-of-s
935..939  per-suit flush DRAW on both: hero ≥2-of-s AND ba 2-of-s
          AND bb 2-of-s
939..943  per-suit flush MIXED: hero ≥2-of-s AND (one ≥3, other 2)
943       straight made_both: ∃ hero rank-pair {r1,r2} with cov=5
          on A AND cov=5 on B (windows may differ)
944       straight draw_both: ∃ pair with cov=4 on A AND cov=4 on B
          (made wins over draw per-board, then intersect)
945       straight mixed: ∃ pair made on one and drawing on the other
OBS_DIM = 946
```

**What this closes.** Every prior board feature was computed in
isolation. PLO5 double-board has structural cross-board interactions
the network couldn't see:

- *Shared ranks* — when a rank appears on both boards, equity
  collisions (pair-the-board, full house dynamics) shift. Combined
  with `_HERO_RANK_HIST_OFF` the net can read "did I pair a rank
  that's on both boards." Board-only mask, no hero gating.
- *Cross-board flush coupling* — same suit on both boards is the
  classic PLO5 cooler axis. Hero ≥2-of-s gates each block so the
  feature only fires when hero participates; the three blocks are
  mutually exclusive per suit. Made-both is the nut-flush-on-both
  case; draw-both is the freeroll-flush case; mixed covers
  one-board-made-one-board-draw.
- *Cross-board straight (hero-pair-aggregated)* — JT on KQx + Q9x
  uses windows {7..11} and {6..10} respectively. Per-window
  indicators would miss this; aggregating per-pair across windows
  (pair = `frozenset({r1, r2})`) lets the same pair fire on both
  boards regardless of which window each board uses. The user-named
  failure case: hero flops a set, opponent jams happily because they
  hold a 2-card straight draw on both boards (freeroll).

**Implementation (encoding.py only).**
- `_cross_board_straight(hole_idx, ba_idx, bb_idx) -> (m, d, x)`:
  iterates 10 windows × ≤10 hero pairs/window × set-ops; collects
  pair_made_a/b, pair_draw_a/b sets; computes
  `made_both = pair_made_a ∩ pair_made_b`,
  `draw_both = (pair_draw_a − pair_made_a) ∩ (pair_draw_b − pair_made_b)`,
  `mixed = (pair_made_a ∩ (pair_draw_b − pair_made_b)) ∪
   (pair_made_b ∩ (pair_draw_a − pair_made_a))`.
- `_cross_board_features(...)` returns a `(28,)` float32 vector
  covering shared-rank mask + flush blocks + straight indicators.
  Reused by the scalar path verbatim.
- Vectorized `encode_observation_batch`: shared rank mask + per-suit
  flush blocks fully vectorized via `np.add.at` over (rank/suit,
  slot) per board. Cross-board straight (3 dims) falls back to a
  per-row Python loop calling `_cross_board_straight`; bit-exact
  with scalar by construction (same helper).
- No Rust binding change — everything derives from hole/board card
  indices already exposed.

**How to apply.**
- Production observation-semantics change. Retrain from scratch on
  the new 946-dim encoder; do not warm-start from any pre-2026-05-06
  checkpoint (none exist on disk by design).
- Cross-board straight aggregation is the load-bearing trick:
  per-window indicators (30 dims) would have missed the JT freeroll
  across different windows. Aggregating per-pair via `frozenset`
  merges windows naturally and keeps the feature compact (3 dims).
- Flush blocks gate on hero involvement — boards-only flush
  coupling is invisible to the network unless hero can play it.
  Intentional: opponents' flushes are still implicit in pot/action.
