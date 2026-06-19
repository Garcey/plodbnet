---
name: Bet-faced as fraction-of-pot feature added
description: 2026-05-09 production obs change — 1 scalar at offset 958 of bet-faced / pot-bet-into; OBS_DIM 958→959; triggers retrain
type: project
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
One new observation scalar at offset 958: bet hero is facing as a
fraction of the pot the bet was made into. Computed as
`to_call / max(pot - to_call, 1)`, clipped `[0, 4]`. 0 when no bet
to face. Half-pot bet reads as ~0.5; pot bet ~1.0; 2x pot overbet
~2.0. OBS_DIM 958 → 959.

**Why:** Pot-odds (offset 748) is `to_call / (pot + to_call)` which
is a non-linear transform of the same underlying ratio. Adding the
raw bet-sizing fraction stretches granularity at the upper end
where pot-odds saturates near 1 (e.g., a 2x overbet vs 4x overbet
read 0.67 vs 0.80 in pot-odds but 2.0 vs 4.0 here), and gives the
network a number that maps directly onto the action-space sizing
labels (`BetPct50`, etc.) without it having to invert the
pot-odds expression.

**How to apply:** triggers retrain. After retrain, watch whether
sizing-aware decisions improve — particularly facing overbets and
facing min-bets, where pot-odds compression hurts the most.
