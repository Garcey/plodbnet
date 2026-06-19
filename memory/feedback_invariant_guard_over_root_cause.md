---
name: Prefer cheap invariant guards over targeted root-cause patches
description: For state-machine bugs, prefer a simple consumer-side invariant check over a fix that targets one specific mechanism
type: feedback
originSessionId: 5fea7462-79e9-4e59-936c-fb00309f5ed1
---
For bugs in state-machines or event pipelines (OCR triggers,
hand-start detection, debouncers, etc.), prefer a single cheap
invariant guard at the consumer side over a targeted patch that
addresses one specific upstream mechanism.

Concrete example: false `_begin_new_hand` triggers were caused by
the ClubGG anti-collusion hero-hole reveal tripping a stale
`hero_hole_rotated` baseline. My initial plan threaded a new
`_prev_hero_hole_indices` field through the rotation predicate to
suppress the None→cards transition. User pushed back: simpler to
just check `fs.button_seat == session.button_seat` at the trigger
site — a real hand boundary always moves the button. One guard,
catches any false-trigger source, no upstream surgery.

**Why:** Targeted fixes are narrow — they address the failure we
identified but miss adjacent ones (e.g., button-OCR flickers that
also produce false triggers). A single invariant captures the
real hand-boundary semantic and short-circuits ALL false
triggers, including ones we haven't diagnosed yet. Less code, less
state, less risk of regressing the upstream signal flow.

**How to apply:** When proposing a fix, before adding new fields
or thread-through state, ask: "Is there a cheap observable
invariant that any real instance of this transition must satisfy?
Can I check that at the consumer instead of patching the source?"
If yes, propose the invariant-guard form first; defer the
upstream surgery to "Out of scope" unless the simpler form
genuinely can't cover the case. Don't lead with the more-complex
fix when a one-line consumer check would do.
