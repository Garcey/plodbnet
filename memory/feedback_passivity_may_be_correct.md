---
name: Apparent passivity may be correct play
description: Don't reflexively diagnose lower-than-expected river bonus% as a regression — the network may be finding genuinely correct play that contradicts user intuition; live-test before tuning
type: feedback
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
When river bonus% comes in below the user's intuitive target
(e.g., <30% on river), do NOT default to "the network is too
passive, bump c". The user's mental model of optimal aggression
may itself be wrong, and PPO under a directional reward signal can
land on play that's correct-but-counterintuitive.

**Why:** User explicitly flagged on 2026-05-08 (mid-block-rotation
widerange run, ~u4015) that they want to live-test the network
against friends before deciding whether observed passivity is a
real problem. Quote: "It's possible my understanding of what
optimal play might not actually be optimal and the network is
finding somewhat optimal play."

**How to apply:**
- Report the metric (e.g., "R=22% on 6-seat clubgg_deep") without
  attaching a "too low" verdict unless the trend itself is
  pathological (e.g., monotonic decline over hundreds of updates).
- When the user notes apparent passivity, offer the live-test path
  before recommending a `c` bump or other reward-shaping change.
- The bar for "regression" is sustained collapse signals (H/kl/v),
  not a single bonus% bucket missing the user's eyeball target.
- If live testing confirms the network plays well, that's a green
  light to begin the late-stage entropy annealing plan — not a
  trigger to change the reward.
