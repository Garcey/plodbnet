---
name: UI action-button labels must match chip amounts
description: Never label a UI raise button as "10%" if its chip amount was clamped to min-raise or stack — show MinRaise / AllIn instead
type: feedback
originSessionId: 0f4de9c9-28ef-47dd-90d1-414a1df544da
---
UI action buttons should display truthfully. A button's label (e.g.
"Raise 10%") must correspond to the actual chip amount the click will
send. If the engine's `compute_sizing_chips` floor-clamps a pct
sizing up to min-raise, the button should read "MinRaise", not "10%".
Symmetrically on the high side: if a pct sizing clamps to stack /
all-in, it should read "All-in" or be suppressed in favor of the
AllIn button.

**Why:** user specified this on 2026-04-20 when reviewing the UI
action menu plan. The engine's internal dedup (`engine.rs:359-377`)
already handles high-side clamping (B100 → AllIn). The low-side
case (B10/B25 → min-raise) was the gap; rendering a "10%" button that
actually sends min-raise chips is misleading.

**How to apply:**
- When building the UI action menu facing a bet, compute the
  *natural* pct amount (ignoring the `.max(min_total)` clamp). If
  natural < min_bet_total AND the player can make a full min-raise,
  emit at most one "MinRaise" button instead of any floor-clamped
  pct labels.
- If the player's stack is too short to reach min_bet_total even
  with a shove, do NOT emit a MinRaise button. AllIn is the only
  raise-style option (symmetric to B100 clamping to AllIn on the
  high side).
- Engine legal mask is unchanged — this is a display-layer rule only.
  Training determinism and the model's action space are untouched.
- Full spec lives in Phase 0 of the approved plan at
  `.claude/plans/i-m-starting-a-new-kind-dragonfly.md`.
