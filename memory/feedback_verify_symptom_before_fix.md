---
name: Verify the symptom before proposing a fix
description: Don't jump to a fix theory until you've actually reproduced/confirmed what the user is seeing; map the user's exact numbers to the code paths before diagnosing
type: feedback
originSessionId: 78983dc4-2873-4db3-a0a7-cbda35e7b613
---
When the user reports a numeric bug ("pot shows $720, should be $700"),
don't immediately diagnose root cause and propose an engine-level fix.
First: do the arithmetic end-to-end, find which code path produces the
reported number, and confirm the user's expected value matches what
the fixed path would produce. If the math doesn't add up cleanly
(e.g. off-by-$20 where the obvious fix would be off-by-$120), stop and
ask the user for clarification rather than hand-waving.

**Why:** On 2026-04-24 I proposed a Rust engine `excluded_seats` fix
for a $720-vs-$700 pot display. User corrected me: "You are mistaken
in your diagnosis. The pot on the flop is accurately 240 (60*4)." My
fix assumed the flop display was wrong and the engine pot needed
correcting, when the actual issue was elsewhere and the flop display
was already correct.

**How to apply:** For "display shows X, should be Y" bugs, trace the
specific display path that produces X, compare to what Y would require,
and confirm the user's mental model of Y before proposing a fix.
Especially important when the discrepancy is small (1 BB) and doesn't
match the obvious structural cause (e.g. 2 extra antes).
