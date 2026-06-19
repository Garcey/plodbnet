---
name: Straight / flush / SF features added to observation
description: 2026-05-06 production obs change — 38 dims/board (76 total) of straight outs, flush draws, made-flush nut distance, and SF outs appended; OBS_DIM 770→846. Triggers retrain.
type: project
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
On 2026-05-06, immediately after the hero rank histogram
(`project_hero_rank_histogram_added.md`), 76 dims of explicit
straight / flush / straight-flush structure were appended at the
encoder tail (38 dims per board, two boards). New tail layout:

```
770..808  board A SF block (38 dims)
  770       flush_nut_distance         (uncapped count of unaccounted higher suit-s ranks; 0 if no made flush)
  771       straight_nut_distance      (uncapped count of higher board-feasible windows; 0 if hero hasn't made any straight)
  772..782  straight_outs_per_window   (10 dims; slot 0=wheel, 9=broadway; raw card count, not normalized)
  782..792  straight_possible_per_window (10 dims; board-only ≥3 distinct ranks of W)
  792..796  flush_possible_per_suit    (4 dims; board-only ≥3 of suit)
  796..800  flush_draw_outs[s]         (4 dims; needs 2-2 split; 13 - visible_s)
  800..804  nut_flush_draw_outs[s]     (4 dims; produces nut hit; 0/1/all-outs cases)
  804..808  straight_flush_draw_outs[s] (4 dims; (rank,suit) hits completing both)
808..846  board B SF block (38 dims, same sub-layout)
OBS_DIM = 846
```

**Design principles (closes the gaps left by category one-hot + rank histogram).**
- **No clamping.** Nut distance is uncapped — let the network see the
  exact count of higher cards still possible.
- **Cross-board visibility.** `visible_count[r][s]` is built once per
  encode from hero hole + board A + board B, so straight outs and
  flush nut distance on board A correctly subtract cards seen on
  board B.
- **Per-suit, not per-flush.** Multi-suit flush draws (1 nut + 1
  blocker-locked) read independently — `flush_draw_outs[s]` and
  `nut_flush_draw_outs[s]` are 4 separate slots.
- **PLO 2+3 rule baked in.** Hero must use exactly 2 distinct hole
  ranks. `H_W = hole_ranks_set & W` is the *set* of distinct hero
  ranks in window W; pocket pair both-cards-in-W counts as one.
- **SF is suit-restricted, not flush ∩ straight.** SF outs use
  per-suit board / hero rank sets `B_s`, `H_s` — a 5-card same-suit
  sequence — not the full straight's board ranks. Each (rank, suit)
  is filtered through the per-suit visibility table.
- **Board-only "possible" signals are symmetric.** Both
  `straight_possible_per_window` and `flush_possible_per_suit` use
  the ≥3-board-cards-in-W / ≥3-of-suit criterion, surfacing "the
  straight/flush is already on the board structurally" alongside
  hero-relative draw quality.

**Algorithm summary (per board).**
- Hero makes straight in W: `(W \ B_W) ⊆ H_W AND |H_W| ≥ 2 AND |W \ B_W| ≤ 2`.
- `straight_outs_per_window[W]`: 0 if hero already makes or |H_W| < 2;
  if |L| = 3 and |M| = 0, sum over r ∈ L of (4 - vct[r]); if |M| = 1
  and 1 ≤ |L| ≤ 3, return 4 - vct[r*] for the single missing rank;
  else 0.
- `straight_nut_distance`: 0 if hero hasn't made any straight; else
  sum of `straight_possible_per_window[w]` for w > h_max (the highest
  window hero makes).
- `flush_draw_outs[s]`: requires hero ≥2 of suit AND board exactly 2
  of suit; value is 13 − visible_count[:, s].sum().
- `nut_flush_draw_outs[s]`: blockers (unseen suit-s ranks > hero top
  suit-s rank). 0 → all draw outs are nut. 1 → exactly 1 (the
  specific blocker card itself). ≥2 → 0 (no single hit produces nut).
- `flush_nut_distance`: 0 if hero has no made flush; else count of
  unseen suit-s ranks above hero's top suit-s hole card.
- `sf_draw_outs[s]`: needs flush draw on s; per window, compute SF
  candidate ranks via the suit-restricted L_s / M_s / H_s sets;
  filter by visibility[(rank, s)] == 0.

**Implementation lives entirely in `python/plo5bp/encoding.py`.**
- Scalar: `_straight_flush_features(hole_idx, board_idx, visible_count)`
  computes the 38-dim block in pure Python (10-window loop with set
  operations). Called twice per encode for board A and B.
- Vectorized: `_straight_flush_features_batch` mirrors the scalar
  semantics with (N, 13, 4) presence tensors and broadcasts the
  per-suit gates over the rank axis. SF candidates accumulate via
  `|=` across the 10 windows then filter through `unseen` and the
  `flush_draw_mask` per (env, suit).
- `encode_observation` builds `visible_count` (13, 4) once from the
  three card lists, then calls the helper for each board.
- `encode_observation_batch` builds `visible_count_batch` (N, 13, 4)
  via per-card writes (deduped — set to 1 vs `add`).
- Bit-exact parity with the scalar path is enforced by the existing
  `test_encoder_batch_parity` sweep — no new parity test added.
- No Rust changes: features are computed entirely from `hero_hole`,
  `board_a`, `board_b` already in the obs bundle.

**How to apply:**
- This is a production observation-semantics change. Retrain from
  scratch on the new 846-dim encoder; do not warm-start from any
  pre-2026-05-06 checkpoint (none exist on disk by design).
- If a future hand-strength feature wants to *replace* one of these
  blocks with a more compact representation, remember the design
  rationale: each block is the minimum-sufficient board-agnostic
  view of one geometric structure. Pair-with-board, hero rank
  histogram, and the SF block are intentionally redundant in
  trivial cases (e.g., `nut_flush_draw_outs[s] == flush_draw_outs[s]`
  when hero holds the ace) — redundancy was chosen over masking to
  keep features composable and simple.
- Layout offsets at `encoding.py` lines 5–41 are the canonical
  reference. Update the docstring in lockstep with any future
  layout change.
