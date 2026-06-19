---
name: Keep plans minimal — write the change, not the analysis
description: User found a multi-section plan with a comparison table and tradeoff analysis hard to follow; they want the actual edit described concisely
type: feedback
originSessionId: 7b62bece-8e86-454f-89e8-27566d72af24
---
When planning a fix, lead with what the change actually does and where it
goes. Don't bury the change under analysis tables, alternative-considered
sections, or per-field encoding diffs.

**Why:** The user said "I'm having difficulty following along with this
plan" after I produced a plan with a 7-row encoding-difference table, a
3-option tradeoff analysis, and a multi-step compressed-obs synthesis
helper. Their actual fix description was one sentence: "instead of
auto-folding sat out players in OCR mode, continue doing the same things
in the UI but call the network as if the number of seats was the correct
number."

**How to apply:**
- Lead with: what file, what function, what changes. The minimum viable
  description of the edit.
- Cut "Why not also do X?" sections unless the user asked about an
  alternative — they're pre-emptive defense, not signal.
- Cut multi-row comparison tables unless the user is choosing between
  options. If the bug is one mechanism, describe that one mechanism.
- Code blocks in the plan should be short — function signatures or
  the 1-3 lines that change, not full helper implementations.
- If the user described the fix in plain English, the plan should
  recognizably match that English. Don't transmute their description
  into a different shape.