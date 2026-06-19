---
name: Flag production behavior changes explicitly
description: When a change affects previously trained models or past-run semantics, call it out loudly — don't bury it as cleanup or style
type: feedback
originSessionId: 0f4de9c9-28ef-47dd-90d1-414a1df544da
---
When a "fix" or refactor changes the observation distribution, reward shape, action space, or any other invariant that past training runs depended on, the plan must say so **explicitly and prominently** — not bury it as "a small, correctness-preserving change" or "cleanup."

**Why**: Caught me calling a hero_equity_mc fix "correctness-preserving" when it was actually a future-peek leak in Phase 2 training obs. The fix was right, but framing it as a minor study-mode tweak hid the fact that stub.pt's training distribution had been contaminated. User flagged this explicitly — they want to make retraining/evaluation decisions with full awareness of what shifted.

**How to apply**: For any change that touches training-visible state (observations, rewards, action mask shape, env dynamics, RNG seeding), the plan should (a) label it "production behavior change" in the section header, (b) list what breaks for existing checkpoints, (c) call out whether retraining is implied. Apply even when the fix seems "obviously correct" — the user values the honesty about impact more than the tidy framing.
