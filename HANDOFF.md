# Handoff — session of 2026-06-08/09 (training + OCR)

This replaces the prior hero-silent-CHECK handoff (that older investigation is
in git history of this file; it was NOT this session's focus and may still be
open — see "Older open issue" at the bottom).

## TL;DR for the next session
- **Training (optimized9, auto-entropy-anneal) is running on the pod, healthy.**
  Just keep checking it and promoting checkpoints on request.
- **All OCR + trainer work this session is committed and pushed to `main`**
  (`github.com/Garcey/plodbnet`, HEAD `1afa5f9`). Working tree only has the
  intentional exclusions (`plo5bp_engine.pdb` build artifact, `memory/` personal).
- **One OCR issue is open:** intermittent chip **decimal-drop** at the top-center
  seat (see below). A robustness fix is proposed but not built (awaiting user go).

## Training — optimized9 (automated F/T/R-gated entropy anneal)
- **Pod:** `ssh root@205.196.144.42 -p 11231 -i ~/.ssh/id_ed25519`
  (network volume US-MO; persists across pod death). Training PID was **15713**.
- **Check health (one-liner pattern):** `pgrep -f scripts/train`, tail
  `runs/optimized9.log`, `nvidia-smi`, latest `checkpoints/optimized9_<N>.pt`.
  Watchdog: `watchdog_optimized9.sh` (stop file `runs/watchdog9.stop`,
  log `runs/watchdog9.log`). Relaunch is automatic on crash.
- **State at handoff (~update 385, ~44h, 0 errors, GPU ~88GB stable):** anneal
  floors **clubgg 0.038 / clubgg_deep 0.060 / deep 0.085** (one cut so far, on
  clubgg). All three tiers currently `drop:hold` — annealing paused because each
  tier's recent block F/T/R came in below its baseline (the stop-loss working).
- **How the anneal works:** in `scripts/train.py` behind `--anneal-entropy`;
  lowers a tier's entropy coef by 0.002 when its per-street aggression (F/T/R)
  holds vs its previous same-tier block; state persisted in the checkpoint. See
  memory `project_entropy_anneal_protocol.md`. **Open design choice (user not yet
  decided):** the no-ratchet-down baseline makes resumption slow; offered to
  switch to compare-vs-immediately-prior-block for faster annealing if they want.
- **Watch:** deep tier dipped (16.0/24.5/31.7 vs 18.2/28.1/35.7 baseline) at the
  unchanged floor — training variance for now; flag if it keeps sliding.
- **Promote a checkpoint to the UI (on request):** read the ACTUAL latest
  `optimized9_<N>` number (don't guess), `scp` it down (aws CLI not on this pod;
  use scp with keepalives — connection resets intermittently, SHA-verify), load-
  check, `cp` over `checkpoints/stub.pt`, restart the UI. Last promoted:
  **optimized9_205** (sha `9469c5805ffd182a`); pod is ahead — promote latest if
  asked. See memory `feedback_promote_read_dont_guess.md`.

## Local services (these survive a Claude-Code update / new session)
- **Study UI:** `uvicorn plo5bp.ui.server:app --port 8765` — was PID **33584**,
  serving `stub.pt` = optimized9_205. Re-find via `netstat -ano | grep :8765`.
  Static files are no-store (JS/HTML/CSS reload on browser refresh); Python
  changes need a UI restart (start it yourself per CLAUDE.md). OCR was OFF at
  handoff — user re-enables via the window picker → On.

## OCR — shipped this session (all committed)
Card-read accuracy (fanned-card per-slot multi-angle de-rotation + adaptive
crop), ~3× tick speedup (per-crop Tesseract read cache), card-commit debounce,
hand-start button-decouple, window picker + auto-off + **Save frame** button,
raise-to-total UI display. Details + diagnostic workflow in memory
`project_ocr_card_reads_and_perf.md`. Debug env vars:
`PLO5BP_OCR_DEBUG_HANDSTART=1`, `PLO5BP_OCR_DEBUG_TIMER=1`.

### OPEN: intermittent chip decimal-drop
A chip reads occasionally drop the decimal (70.05→7050, 450.5→4505; 100× too
big) at the **top-center seat (internal seat 3; user calls it "seat 4")**.
Per-frame deterministic but intermittent; saved frames so far read correctly, so
it's likely the **seeded starting stack** from a past hand-start anchor frame.
**Proposed fix (not built, pending go):** guard stack-seeding against a transient
drop — debounce anchor stack reads and/or reject implausible-magnitude jumps
(`_begin_new_hand` only guards too-small via `_MIN_PLAUSIBLE_STACK_CENTS`). Two
clarifying Qs are outstanding: which surface (ClubGG window vs study-UI panel)
and which seat.

## Deferred (noted, not done)
- Anneal: switch to compare-vs-prior-block (faster resumption) — user's call.
- OCR perf: `tesserocr` in-process Tesseract (marginal after cache; not installed).
- Hand-start: board-clear/pot-reset fallback trigger (only if debug logs show
  button detection itself misses).
- Lower card-commit debounce N (only if speed still feels slow).
- `_CARD_STABLE_TICKS`, `_BUTTON_STABLE_TICKS_LOCKED`, `_anneal_*` are all tunable.

## Git / commit habits
- `main` == `origin/main` at `1afa5f9`. Repo-local git identity is set to
  `Garcey <themilesgarcia@icloud.com>` (it was unset; commits failed without it).
- Excluded from commits by design: `plo5bp_engine.pdb` (build artifact),
  `memory/` (personal). Checkpoints/`*.pt`, `screenrecords/`, `runs/` gitignored.

## Older open issue (pre-this-session, possibly still unresolved)
Hero's silent CHECK on the flop / new-hand timer-bar reconstruction — the prior
focus of this file. Not worked on this session; the hand-start decouple + card
debounce may have helped adjacent symptoms. Full prior detail is in
`git log -p HANDOFF.md` and `.claude/plans/`.
