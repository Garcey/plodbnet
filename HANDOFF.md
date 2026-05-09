# Handoff — hero's silent CHECK on the flop undetected by UI

## Status

**Active bug**: Hero CHECKs on the flop in heads-up bomb-pot
(first-to-act, OOP). The check goes undetected by the UI; engine
state stays stuck on hero as `current_actor` until either fastaf
acts (existing walk ladder infers hero's CHECK via
`_any_remaining_delta`) or the next street deals (StreetReveal
CHECK reconciler back-fills via `_reconcile_missed_checks_on_street_reveal`).

**Tasks A/B shipped earlier (working). Task C shipped but is dead
code — see "Confirmed wrong" below.** No banner-detection fix has
been written yet.

## Confirmed wrong this session (2026-04-26)

### H1 — anti-collusion cards-hide rule (Task C's premise)

**The rule documented in `extract.py:96-103` is misstated.** The
correct ClubGG behavior, per user (and corroborated by burst
frames where hero's cards stayed face-up the entire 16s while it
was fastaf's turn):

> Cards are hidden from hand-start UNTIL it becomes hero's first
> turn on the flop. Once it becomes hero's first flop turn, cards
> turn face-up and STAY face-up for the rest of the hand —
> through every subsequent action and street.

The visible→hidden transition only occurs on a NEW HAND boundary,
NOT on hero's mid-hand action. Therefore Task C's
`_hero_hole_just_hid` helper never fires mid-hand and is dead code
in production.

**Saved to memory**: `project_clubgg_anti_collusion.md` (loaded
into MEMORY.md). Future sessions should rely on the memory, not
the stale `extract.py:96-103` comment. Updating that comment is
follow-up cleanup but not on the critical path.

### Action: leave Task C in place but plan around it

The Task C branch in `events.py:_infer_seat_actions` is harmless
(never fires) but adds noise. Roll back only AFTER a working
replacement is verified.

## Current direction — yellow "Check" banner detection

`screenrecords/frames/debug_1777131926.png` shows a YELLOW "Check"
banner overlaid on hero's hole-cards area on the TURN, immediately
after a CHECK. This is the analog of `has_bet_banner` for CHECK —
both are HSV-detectable bands over the seat's `cards_back` ROI,
disjoint hue ranges (blue ~100-115 vs yellow ~20-35).

**Open question**: does the same banner fire on hero's FLOP CHECK?
We have not yet confirmed. The plan
(`.claude/plans/this-is-a-continuation-vectorized-diffie.md`)
gates the implementation on this verification.

## Verification status this session

### Server change — ms-precision filenames for save_frame

`python/plo5bp/ui/server.py:1593-1594` — changed
`debug_<int(time.time())>.png` → `debug_<int(time.time()*1000)>.png`
so within-second saves no longer overwrite. Previously, a 30-save
burst over 1.5s collapsed to ≤2 unique files. UI server
restarted to pick up the change.

### Session config — bumped num_seats 5 → 6

User's session was at `num_seats=5` but `rois.py:seats()` only
supports 6. Bumped via `POST /seats {"num_seats": 6}`. The 6-seat
layout works fine for heads-up — the empty seats just read empty.
User may want to revert post-debug.

### Burst captures

Three bursts attempted:
1. `debug_1777141137-141.png` (5 unique frames, sec-precision):
   missed CHECK entirely; static fastaf-thinking state.
2. `debug_1777141614367-1777141624836.png` (60 frames, ms-prec,
   10s, 100ms cadence): static across all 60 — hero already
   CHECKed before burst started (user confirmed timing problem).
   Crucial corroboration: hero's cards stayed FACE-UP the entire
   10s while it was fastaf's turn → H1 wrong.
3. `debug_1777183332658-1777183348761.png` (~120 frames,
   ms-prec, ~16s, 80ms cadence): captured around a real CHECK
   click after a 3s delay-then-GO message. **User confirmed a
   ~10s delay between hitting Enter and the GO message firing,
   and is going to triage the frames manually to identify the
   CHECK moment** (image-read budget too tight for me to scan
   all 120).

### Memory: image read budget

PNG frames are ~1MB each. Reading >30 in a turn truncates around
the 32MB tool-result cap. Saved guidance to
`feedback_image_read_budget.md` (in MEMORY.md). Prefer:
- User triages, names candidate timestamps, I read those only.
- Or write a Python script that diffs frames on disk.

## Recommended starting point for the next session

1. **Get the CHECK-moment frame.** Ask the user which frame in
   `debug_1777183332658-1777183348761` shows hero just after the
   CHECK click (or one frame before/after). Read 2–4 specific
   frames named by the user, NOT the full burst.
2. **If yellow banner is visible**: proceed to Phase 2 of the
   plan (`this-is-a-continuation-vectorized-diffie.md` Steps
   5-8): add `has_check_banner` to `cards.py`, `check_banner`
   field to `SeatObs`, populate in `extract.py:_build_seat_obs`,
   and add a walk-ladder branch in
   `events.py:_infer_seat_actions` BEFORE the Task C branch and
   AFTER the bet-banner branch. Tune HSV bounds from the live
   frame.
3. **If yellow banner does NOT appear on the flop**: pivot to H5
   (active-actor highlight ring detection) — separate plan
   needed. The yellow underline on fastaf's stack plate observed
   in the bursts is the candidate signal — verify whether it's
   actually a turn indicator (animates with countdown) vs. a
   static decoration.
4. **If unclear**: write a Python diff script that walks the
   burst frames in chronological order and flags any that differ
   substantially in the hero `cards_back` ROI. That isolates the
   transition without requiring image reads.

## What was tried this session and previously (recap)

### Task A — seat 3 phantom in-hand bug (DONE, working)

`python/plo5bp/ocr/rois.py` — moved seat 3's `committed_label`
ROI from y=0.3098 to y=0.2500 to clear `POT_BANNER` (top=0.3087).
Verified: 82 OCR tests pass.

### Task B — `_reconcile_missed_checks_on_street_reveal` over-counting (DONE, working)

`python/plo5bp/ui/server.py:513-567` — replaced unconditional
`n = len(active) * streets_to_fill` push with a loop-fill that
delegates to the engine. Helps the after-the-fact recovery path,
not the live-detection path.

### Task C — hero hole-card visibility transition (SHIPPED, dead code)

`python/plo5bp/ocr/events.py:159-176, 459-471` — added
`_hero_hole_just_hid(last, fs)` and a hero-specific walk branch.
Premise (H1) confirmed wrong this session. Branch never fires.
Leave in place until a working replacement is verified.

### Plan files

`.claude/plans/this-is-a-continuation-vectorized-diffie.md` — the
banner-detection plan. Phase 1 (verify) is in progress: the
ms-precision save_frame is shipped, three bursts captured, user
is triaging the third for the CHECK moment. Phase 2 (implement)
contingent on banner being visible on the flop.

`.claude/plans/in-the-last-session-mossy-hedgehog.md` — Task C's
plan. Premise wrong. Reference for "what NOT to do."

## Files modified this session

```
python/plo5bp/ocr/rois.py          Task A (prior session): seat 3 committed_label y=0.2500
python/plo5bp/ui/server.py         Task B (prior session): reconciler loop-fill (513-567)
                                   Task C (prior session): no change (lives in events.py)
                                   THIS SESSION: ms-precision save_frame filenames (1593-1594)
python/plo5bp/ocr/events.py        Task C (prior session): _hero_hole_just_hid + walk branch
                                   (159-176, 459-471) — DEAD CODE, leave in place for now
HANDOFF.md                         this file
~/.claude/projects/.../memory/MEMORY.md      added 2 entries
~/.claude/projects/.../memory/project_clubgg_anti_collusion.md   new
~/.claude/projects/.../memory/feedback_image_read_budget.md      new
```

## Test status

Tests not re-run this session. Last known: 82 OCR tests pass + 2
pre-existing fixture failures unchanged. The ms-precision filename
change should not affect tests (no test reads `debug_*.png`
filenames). Next session should run
`.venv/Scripts/python -m pytest tests/ocr/ -q` after any code
change.

## Live UI state at handoff

- UI server: started in background this session (PID 661311 at
  start, may have been replaced; check `netstat -ano | grep :8765`).
- OCR runner: running, `window_match=testing`, `poll_ms=100`,
  `num_seats=6`.
- Last known frame range: `debug_1777183332658-1777183348761.png`
  in `screenrecords/frames/` (user triaging).

## Diagnostic capture for timer-bar regressions (2026-04-26)

Two regressions reported on the shipped timer-bar CHECK signal:

1. fastaf checks → hero checks back → turn deals → UI still on hero.
2. Hero first-to-act CHECK on flop is never registered.

To capture the underlying signal, restart the UI with the env var
`PLO5BP_OCR_DEBUG_TIMER=1`:

```bash
PLO5BP_OCR_DEBUG_TIMER=1 .venv/Scripts/python -m uvicorn \
  plo5bp.ui.server:app --port 8765
```

Then play one hand reproducing each scenario. Filter the uvicorn
stdout for these prefixes:

- `ocr.timer:` — per-tick `prev_active`/`now_active`/`raw_actors`/
  `is_actor_per_seat`. Shows directly whether the timer-bar HSV
  detector caught the bar each tick and whether multiple seats
  briefly registered during animation.
- `ocr.walk: timer_bar_branch_fired` — the active-actor transition
  branch in the walk ladder (events.py:548) emitted a CHECK.
- `ocr.walk: stalled` — walk hit `else: break` at this seat with no
  signal from any branch (commit/banner/stack-drop/timer-bar).
- `ocr.reconcile.checks:` — StreetReveal-driven CHECK reconciler
  ran; shows `view.street`, `view.bet_to_call`, `committed`.

Also capture one frame mid-thinking for each seat with the bar
visible via `POST /ocr/save_frame` — overlay each frame's
`timer_bar_left` ROI to validate live-capture geometry vs the
1927×1391 reference.
