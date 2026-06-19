---
name: Limit image reads — they consume context budget
description: PNG frames are ~1MB each; reading too many in one turn hits the 32MB cap mid-output
type: feedback
originSessionId: 6609b555-9a44-45b5-aaec-c1043b436873
---
When inspecting captured PNG frames from `screenrecords/frames/`, each image is roughly 1MB. Reading more than ~30 in a single turn risks truncating output around the 32MB tool-result cap.

**Why:** The user noted on 2026-04-26: "Be careful sampling too many images though, because sampling over 32mb will stop you mid output. Each image is roughly 1mb."

**How to apply:** When a burst produces many frames (e.g., 100+ saves), do NOT read them all looking for a transition. Instead:
1. Ask the user to triage and point you at the candidate frame timestamp(s) — the user can scrub through file explorer / image viewer faster than you can read.
2. Or read only 2–4 specific frames per turn (start/middle/end, or around timestamps the user names).
3. Or write a small Python script that diffs frames on disk (e.g., MSE between consecutive frames) to find the transition moments, then read only those.

Don't burn context loading every frame to "be thorough" — it actively breaks tool output mid-response.
