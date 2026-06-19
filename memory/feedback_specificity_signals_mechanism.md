---
name: Bug specificity tells you the mechanism — don't hand-wave "transient noise" when only ONE seat/case is affected
description: When a bug consistently hits one specific seat/entity and not others, reject generic noise theories; investigate what's structurally unique about that seat (ROI placement, neighbor sitting-out state, rendering overlap)
type: feedback
originSessionId: 78983dc4-2873-4db3-a0a7-cbda35e7b613
---
When a bug is CONSISTENTLY reproducible on one specific seat/
object but not others, "transient noise" or "motion blur" theories
are suspect. Noise hits uniformly; structural issues hit
specifically. Prefer theories that match the observed selectivity.

**Why:** On 2026-04-24 the user reported BTN (seat 4) consistently
shows folded in the UI before actually folding on-screen — every
hand, only that seat. I proposed a "transient has_cards_back dip
during fold animation" theory as the root cause. User corrected:
"I would also be hesitant to say that there's an issue where the
BTN's cards fall below the 0.15 ROI threshold because it's
consistently happening for the BTN but no one else. Perhaps
consider the impact of the player who is sat out (username
Gunner53) that sits in between Castor876 and the BTN on the bug?"

The consistency-to-one-seat was a signal I skipped past. A
transient-noise theory would need a specific story for why the
noise only lands on one seat — e.g., ROI overlap with a
specifically-rendered neighbor (sitting-out overlay, button
chip), or a miscalibrated ROI post-renumber.

**How to apply:** When the user says "this always happens to X,
never to Y", treat that as a direct hint about mechanism. Look
for structural differences between X and Y: ROI coordinates,
neighbor state, rendering order, Z-ordering of overlays, seat-
position-specific logic. If a noise theory survives, you must
explain the selectivity.
