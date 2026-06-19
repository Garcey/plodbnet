---
name: Pair-with-board features added to observation
description: 2026-05-06 production obs change — 5 pair-count slots + 4 board-pair-structure bits per board appended; OBS_DIM 739→757. Subsumes pair/set/full-house/quads via slot value. Triggers retrain.
type: project
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
On 2026-05-06, after the equity-vs-random removal, the encoder grew
+18 dims of explicit pair-with-board signal. New tail layout:

```
739..744  _PAIR_COUNT_A_OFF   (5 slots, board A)
744..749  _PAIR_COUNT_B_OFF   (5 slots, board B)
749..753  _BOARD_STRUCT_A_OFF (paired, double_paired, tripled, quadded)
753..757  _BOARD_STRUCT_B_OFF (same)
OBS_DIM = 757
```

**Slot semantics.** Slots 0..4 follow the board cards sorted
*descending by rank* (one entry per board card — paired boards
repeat ranks across consecutive slots). Slot value = count of hero
hole cards sharing that slot's rank. Active slot count = number of
board cards visible (3 flop / 4 turn / 5 river); trailing slots
stay 0. Examples:
- top pair: K-Q-J-T-2 on K-9-4 → `[1,0,0,0,0]`
- top set: K-K-Q-J-T on K-9-4 → `[2,0,0,0,0]` (slot value 2 is the
  pocket-pair-on-board signal)
- three-pair: K-Q-J-9-4 on K-9-4 → `[1,1,1,0,0]`
- paired flop: one K on K-K-9 → `[1,1,0,0,0]` (slots 0+1 share rank)
- full-house board K-K-K-9-9 with K9 in hand → `[1,1,1,1,1]`
- quadded river K-K-K-K-9: hero can't hold a K so slots 0..3=0;
  slot 4 = count of 9s.

**Pair-structure bits** (monotonic):
- unpaired: `(0,0,0,0)`
- paired: `(1,0,0,0)`
- double-paired (K-K-9-9-x): `(1,1,0,0)`
- tripled: `(1,0,1,0)`
- full-house board (K-K-K-9-9): `(1,1,1,0)`
- quadded (K-K-K-K-x): `(1,0,1,1)`

**Known blind spot (acceptable).** Pocket pairs whose rank isn't
on the board read all-zeros in the count slots — overpair AAxxx on
K-9-4, KKQQT on 9-9-4, etc. The Pair / TwoPair category one-hot
still fires, giving the strength bucket. If this gap shows up as a
real weakness during training, the fix is a separate "hero in-hand
pair count" feature — don't reach for equity-vs-range as the answer.

**Why:** After removing equity-vs-random, the only hand-strength
signals beyond raw card multi-hots were the 9-bucket category
one-hot and 2 binary draw flags. Category distinguishes Pair from
Trips but says nothing about *which* pair / set — top set vs
bottom set, top three-pair vs bottom three-pair. That's the
within-category granularity that matters most in narrow-range
bomb-pot play. User: "Let's start first with pairs. The network
should be told which pairs it has — top pair, second pair, third
pair, fourth pair, fifth pair and any combination of pairs."

Implementation lives entirely in `python/plo5bp/encoding.py`:
- `_pair_features(hole_idx, board_idx)` — scalar helper.
- `_pair_features_batch(hole, board)` — vectorized counterpart.
  Builds `(N, 13)` rank-count tensors via `np.add.at`, sorts board
  ranks descending with `-np.sort(-sort_key)` (sentinels go to the
  tail), gathers hero counts at sorted-rank indices.
- Both are wired into `encode_observation` / `encode_observation_batch`
  after the draw-flag block. The vectorized parity test
  (`test_encoder_batch_parity`) catches scalar-vs-batch divergence
  byte-exactly.
- No Rust changes — feature is computed entirely from existing
  `hero_hole` and `board_a/b` arrays already in the obs bundle.

**How to apply:**
- This is a production observation-semantics change. Retrain from
  scratch on the new 757-dim encoder; do not warm-start from any
  pre-2026-05-06 checkpoint (none exist on disk, by design).
- If a future obs-feature lands and breaks this layout, update the
  offset constants and the module docstring at `encoding.py`
  lines 5–28 in lockstep.
- Keep features added "one by one" (user's framing): each new
  hand-strength signal should be evaluated on its own training run
  before piling on the next.
