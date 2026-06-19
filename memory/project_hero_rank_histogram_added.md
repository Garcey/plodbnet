---
name: Hero rank histogram added to observation
description: 2026-05-06 production obs change — 13-dim board-agnostic histogram of hero hole-card ranks appended; OBS_DIM 757→770. Closes the pocket-pair-not-on-board blind spot from the pair-with-board feature. Triggers retrain.
type: project
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
On 2026-05-06, immediately after the pair-with-board feature
(`project_pair_with_board_added.md`), a 13-dim board-agnostic hero
rank histogram was appended at the encoder tail.

```
757..770  _HERO_RANK_HIST_OFF (13 floats)
          slot r = count of hero hole cards at rank r (0=2 ... 12=A)
          values are integer-valued in [0, 4], cast to f32, no normalization
OBS_DIM = 770
```

**What it surfaces.**
- Pocket pair of any rank not on the board: AA on K-9-4 → slot 12 = 2.
- Pocket trips: AAA on 9-7-3 → slot 12 = 3.
- Pocket quads (rare): AAAA → slot 12 = 4.
- Two pocket pairs: AAKK on 9-7-3 → slot 12 = 2, slot 11 = 2.
- Naked overcards: AKQJT (no pair) → five slots = 1.

**Why.** The pair-with-board feature reads zeros for any pocket
pair whose rank isn't on the board — the blind spot called out in
the pair-with-board memory note. The histogram closes it
completely. Strategically the overpair / underpair distinction
isn't huge in PLO5 bomb-pots (nut-peddling dominates and pocket
aces is usually just two outs against the actual continuing
range), but the user's framing is "give the network as much
*objective* hand-structure information as possible" — the
representation problem with equity-vs-random was that it was
non-objective, not that there was too much information density.

**Redundancy is intentional.** For any rank that appears on either
board, `_HERO_RANK_HIST_OFF + r` value equals the corresponding
pair-with-board slot value. The histogram only adds *net new* info
for ranks not on either board. We chose redundancy over masking
to keep the feature board-agnostic and simple.

Implementation lives entirely in `python/plo5bp/encoding.py`:
- Scalar: a single loop incrementing
  `out[_HERO_RANK_HIST_OFF + (c // 4)]` for each hero hole card.
  Reuses the existing `hole_list` already built for draw / pair
  features.
- Vectorized: `np.add.at` over an `(N, 13)` zero array, identical
  pattern to `_pair_features_batch`'s hero count tensor. Inlined
  in `encode_observation_batch` rather than refactored out of
  `_pair_features_batch` to avoid cross-feature coupling.
- Batch parity is enforced by the existing
  `test_encoder_batch_parity` sweep — no new parity test needed.
- No Rust changes; feature is computed entirely from `hero_hole`
  already in the obs bundle.

**How to apply:**
- This is a production observation-semantics change. Retrain from
  scratch on the new 770-dim encoder; do not warm-start from any
  pre-2026-05-06 checkpoint (none exist on disk).
- If a future hand-strength feature wants to *replace* this
  histogram with something board-relative (ladder slots, etc.),
  remember the design rationale: the histogram is the
  minimum-sufficient board-agnostic view; explicit ladder
  features go *on top of* it, not in place of it.
- Layout offsets at `encoding.py` lines 5–32 are the canonical
  reference. Update the docstring in lockstep with any future
  layout change.
