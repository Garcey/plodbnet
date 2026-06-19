---
name: Encoder dead-chips invariance — production observation change
description: As of 2026-04-30 the encoder subtracts `max(0, starting - eff_cap)` from each seat's stack so the network is invariant to chips above max-other-reachable; affects `_STACKS_OFF` and `_SPR_OFF` for any seat with a starting stack above the second-deepest. Triggers retrain.
type: project
originSessionId: 5fea7462-79e9-4e59-936c-fb00309f5ed1
---
On 2026-04-30 the encoder (`python/plo5bp/encoding.py`, both scalar
and batched paths) was changed so that the per-seat effective remaining
fed to `_STACKS_OFF` and `_SPR_OFF` is
`max(0, stacks[i] - max(0, starting_stacks[i] - eff_cap[i]))` instead
of `min(stacks[i], eff_cap[i])`.

For any seat whose starting stack exceeds the second-deepest at hand
start, this changes the encoded value by the size of the dead portion
plus any committed chips that previously leaked through the cap. In
uniform-stack training (current curriculum) the change is a no-op
because every seat has `starting == eff_cap` and `dead == 0`. The
divergence only fires once heterogeneous stack distributions are
introduced (e.g., ClubGG-weighted stage from
`_CLUBGG_STACK_BANDS`).

**Why:** User screenshot showed the network's bet recommendation
shifting from $69.90 → $69.92 between two HU bomb-pot frames that
differed only in villain's stack ($340 vs $440). The previous encoder
formula `min(remaining, cap)` (a) didn't subtract the deep seat's
already-committed ante from the cap, and (b) the SPR feature didn't
clamp at all. User invariant: "the network should be told that both
players have $340 behind."

**How to apply:**
- This is a production observation-semantics change. Once heterogeneous
  stacks enter training, retrain on the new encoder rather than
  warm-starting from a uniform-stack checkpoint that never saw this
  signal differ.
- `tests/python/test_encoding.py::test_encoding_invariant_to_unreachable_chips_above_eff_cap`
  asserts the invariant; do not weaken it.
- The UI continues to display raw chips; the change is purely at the
  encoder boundary. Engine state and reward semantics are unchanged.
