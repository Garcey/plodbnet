---
name: Don't elevate user speculation (or your own hypothesis) to fact in diagnoses
description: When the user offers a "perhaps consider X" hint, treat it as a hypothesis to test, not a premise to build on. Same for your own structural theories — verify before asserting.
type: feedback
originSessionId: 78983dc4-2873-4db3-a0a7-cbda35e7b613
---
When the user says "perhaps consider X" or "maybe Y is involved",
that's a hypothesis from them, not a confirmed fact. Do not write
the plan's root-cause section as if their speculation were
established. Similarly, don't treat structural theories you
generated (ROI overlap, preprocessor behavior, Tesseract output)
as fact until you've verified them by running the actual code
against the actual frame.

Also: don't overgeneralize from one observation. "Consistently
happening in this specific hand" is not the same as "button-
specific across hands" — one hand is one hand.

**Why:** On 2026-04-24 I wrote a Fix P plan that confidently
asserted: (a) Tesseract reads the gold D disc as "0" via the
chip-crop fallback, (b) Gunner53's sitting-out state is the
mechanism by which the walk reaches BTN. Both were plausible
but neither verified. The user had only observed the bug in one
hand and had framed the Gunner53 involvement as speculation:
"Perhaps consider the impact of the player who is sat out". I
promoted that speculation to a confirmed mechanism in the plan.
User correction: "Ensure that your reasoning for the cause of
the bug is fact, not speculation."

**How to apply:** Before writing a "root cause" section, separate
observed facts (user saw X) from code facts (file at :line does Y)
from hypotheses (therefore the mechanism is likely Z). Label them
distinctly in the plan. If a hypothesis is load-bearing, verify
it by tracing code — read the exact functions/branches involved,
compute numeric values end-to-end, or run diagnostics (e.g.
extract_frame_state on the canonical frame). Don't assume engine
seat numbers from screen position; don't assume a user's
"perhaps X" maps to any particular variable. If you can't verify
a hypothesis in plan mode, either drop it from the plan or label
it as unverified and make the fix robust to it being wrong —
the fix itself should be justifiable on code-fact grounds alone.
