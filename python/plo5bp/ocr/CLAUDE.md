# Live capture (ClubGG OCR + PokerNow) — notes

Loaded when you work under `python/plo5bp/ocr/` (pointer from `python/plo5bp/ui/live/`). Local build only: the public build never imports any of it.

## OCR integration architecture

The OCR stack turns a polled ClubGG screenshot into mutations on the
UI's `Session` so action_log, participant mask, and card spec stay in
sync with the live table. The pipeline is layered so that each layer
can be unit-tested in isolation:

```
capture_by_match (live.py)
    ↓  BGR np.ndarray
extract_frame_state (extract.py)
    ↓  FrameState (cards, seats, button, pot)
EventReconstructor.step(fs, engine_view) (events.py)
    ↓  list[OcrEvent]  (HeroHoleRevealed, StreetReveal, SeatAction, OcrWarning)
OcrRunner._tick (ui/live/clubgg.py)  →  mutates session, calls _rebuild_env
```

### Key files

- `python/plo5bp/ocr/live.py` — window-match capture. Resolves target
  window by title substring; returns BGR frame via pywin32/MSS.
- `python/plo5bp/ocr/rois.py` — normalized ROI rectangles (`x, y, w, h`
  in fraction of frame). `SeatROIs` per seat bundles: `name_plate`,
  `stack_label`, `committed_label`, `button_anchor`, `cards_back`.
  Seat 0 is hero (south-center); 1-5 clockwise.
- `python/plo5bp/ocr/cards.py` — per-card classifier: `classify_suit`
  (HSV bands), `classify_rank` (template match against pre-extracted
  rank glyphs in `templates/`), `has_cards_back` (silver-diamond
  pattern ratio > 0.15), `has_bet_banner` (blue-banner HSV mask >
  0.03 of ROI).
- `python/plo5bp/ocr/text.py` — Tesseract wrappers:
  `read_chip_amount` (stack labels, returns cents),
  `read_seat_commit` (chip-oval "180"-style amount),
  `read_pot_amount`. Cyan-bbox pre-crop improves digit segmentation.
- `python/plo5bp/ocr/extract.py` — stateless `extract_frame_state`
  entrypoint. `_read_seat` uses **multi-signal "in hand"** detection
  (cards_back OR banner OR committed>0). Button detection scores
  amber-disc largest connected component per seat's `button_anchor`.
- `python/plo5bp/ocr/events.py` — `EventReconstructor`. Diffs new
  FrameState against `last_fs`, consults `EngineView` for current
  actor, emits `SeatAction`s via a fallback ladder. `last_fs` is
  advanced only on accepted reads; warnings don't rebaseline.
- `python/plo5bp/ocr/types.py` — `Card`, `SeatObs`, `FrameState`
  dataclasses. `FrameState.to_dict` / `from_dict` round-trip through
  the fixture JSON format in `tests/ocr/fixtures/`.
- `python/plo5bp/ocr/tools/` — `label_cards` (assisted labeling) and
  other dev tools for producing fixtures and bootstrapping rank
  templates.
- `python/plo5bp/ui/server.py` — FastAPI study backend.
  - `Session`: big mutable state bag (cards, action_log, button,
    participant mask, config; `session.live` = the live tracker state).
  - `_rebuild_env`: canonical path. Pads card spec with unused deck
    indices, reset_study, replays action_log with
    `_auto_fold_sitting_out` for retired seats.
- `python/plo5bp/ui/live/` — live capture, LOCAL BUILD ONLY (server.py
  calls `routes.install(app)` unless `PLO5BP_PUBLIC`; the public build
  never imports the package). `state.py` (`LiveState`: debounce counters,
  pending card reads, observed stacks/pot — `live_state.x` proxies the
  current session's), `tracking.py` (shared by both sources: card-slot
  writes, units, `_mirror_observable_state` + `_begin_new_hand`, the
  reveal reconcilers, `_engine_view_from_session`), `clubgg.py`
  (`OcrRunner._tick`: capture → extract → mirror → reconstruct → apply
  events → rebuild env; one task per process), `pokernow.py`
  (`PokerNowRunner`), `routes.py` (`/ocr/*`, `/pokernow/*`, `install`).
  The core resets live state through `server._SESSION_RESET_HOOKS` and
  takes live keys into `_state_dict` through `_STATE_EXTRAS_HOOKS`. Tests
  patch names where they are USED (e.g. `pokernow._begin_new_hand`).

### Unit system

OCR reads are in **cents** (ClubGG's dollar display × 100, so "$180"
is 18000). Engine carries **chips** where `cfg.bb` chips = 1 big blind
= `dollars_per_bb` dollars. At the Study default (bb=10000, $2/bb —
`common.FORMAT_DEFAULTS`, shared with the Trainer) `chips_per_cent = 50.0`,
so 18000 cents → 900000 engine-chips (at $20/bb it would be 5.0). **Every
arithmetic step inside `EventReconstructor` runs in engine-chips**;
conversion happens at the boundary via
`_ocr_cents_to_engine_chips` on the server and `_to_engine` /
`chips_per_cent` inside events.py. Don't mix units — it was the
source of a prior bug where raw cents got added to `cfg.ante`.

### The multi-signal "in hand" rule (extract.py)

A non-hero seat is `folded=False` when **any** of:
- `has_cards_back(cards_back_crop)` — silver-diamond pattern > 15%;
- `has_bet_banner(cards_back_crop)` — blue "Bet" overlay > 3%;
- `committed_chips` reads positive.

The banner signal is load-bearing: while ClubGG's chip-settle
animation renders a blue overlay, `has_cards_back` drops below
threshold and would otherwise evict the betting seat from the hand.
Do NOT add an `in_hand` gate in front of `has_bet_banner` — each
signal fires independently.

### The debounced hand-start machine (server.py)

`_mirror_observable_state` owns hand-boundary detection. Every tick:

1. Mirror card spec (hero hole, flop_a/b, turn, river) and observed
   stacks/pot.
2. Build `observed_sitting_out` (non-hero seats with `folded=True`)
   and `observed_button`.
3. Debounce: if `(observed_button, observed_sitting_out)` matches
   the pending snapshot, increment `_pending_stable_ticks`;
   otherwise reset and record the current frame as
   `_pending_anchor_fs` (the pre-commit baseline).
4. After 2 stable ticks, `committed_ready=True`.
5. Fire `_begin_new_hand(anchor_fs, button, hero_hole_indices)` when
   any of: button rotated, hero hole rotated, or first-ever commit
   (`hand_in_hand_mask` empty AND someone reads in hand — ten all-folded
   ticks fire it once, not every tick). The hero-hole trigger NEVER fires
   on a frame whose `fs.button_seat is None`.
6. `_begin_new_hand`: resets defaults (incl. `last_hero_hole`, which is
   then adopted from hero's first full read of the hand — a stale baseline
   used to latch `hero_hole_rotated` and let one unreadable-button frame
   wipe a live hand), locks
   `hand_in_hand_mask = {i : anchor.seats[i].folded is False}` (hero CAN
   join later through mask expansion and is never auto-folded), seeds
   `cfg.starting_stacks` from the anchor frame's OCR reads via
   `dataclasses.replace` (variant/sb carried), sets `session.env = None` so
   the `EngineView` built on that tick reflects the NEW hand, and calls
   `_active_reconstructor().rebaseline(anchor_fs)` so diff-based
   inference starts from the pre-commit frame.
7. After any hand-start, refresh
   `sitting_out_seats = (all_seats - hand_in_hand_mask) |
   folded_this_hand`. Mid-hand SeatAction(gate="fold") events grow
   `folded_this_hand` without triggering a new hand-start.

**Invariants the debouncer MUST preserve:**
- `session.button_seat` is not updated pre-commit. (An earlier
  `elif fs.button_seat is not None` branch silently clobbered it
  during the stability window and defeated `button_changed`
  detection on the commit tick. Don't re-introduce.)
- `session.sitting_out_seats` is not written pre-commit. Initial
  state is `frozenset()` until `hand_in_hand_mask` is populated.
- `folded_this_hand` is cleared by `_new_session_defaults` (which
  `_begin_new_hand` calls) and, in live mode, RE-DERIVED from the engine
  after every `_rebuild_env` (`_sync_folds_with_engine`): a FOLD the
  engine rejected — or one the user `/undo`es — must not leave the seat
  skipped while the engine still waits on it. The ClubGG reveal-frame
  fold reconcile needs two consecutive ticks; PokerNow is immediate.
- Live capture is PLO5-only: `/ocr/start`, `/ocr/rescan` and
  `/pokernow/ingest` return 409 under any other format before touching
  state. `/reset`, `/format`, `OcrRunner.start` and a live-source switch
  clear the debounce state (`_reset_live_tracking`).
- A hand-start anchor must show the antes in its pot read (pot >= 90% of
  in-hand seats x ante; unreadable = ok): the debouncer waits at most
  `_ANTE_WAIT_TICKS` (5) and upgrades to the first frame that does — a
  pre-ante anchor seeded every stack one ante high.
- Sync health (`tracking.py`, shown in `/ocr/status` + `/pokernow/status`
  `warnings`, red in the live status line): walk actions are validated
  against the engine BEFORE they enter the log, as one batch
  (`_record_seat_actions`); a refused one voids the step and restores the
  reconstructor (`snapshot`/`restore`) so the next frame re-derives it, and
  after 3 identical refusals the legal prefix is kept and the user is asked
  to enter that seat's action (PokerNow: no retries, reads are exact). The
  on-screen pot is compared with the engine's (with or without this
  street's bets); a disagreement lasting 3 ticks (2 PokerNow payloads)
  raises "out of sync" (ClubGG only while action tracking is on).
- One live lock (`tracking.live_lock`) serializes ticks, PokerNow payloads
  and every study route (`routes._LiveLockMiddleware`); the OCR loop ticks
  on fixed `poll_ms` deadlines.

### The reconstructor fallback ladder (events.py)

`_infer_seat_actions` walks seats starting from
`engine_view.current_actor`. For each actor, it derives
`new_commit` via (in priority order):

1. `primary_read` = `committed_chips` OCR, in engine-chips, **only
   if it differs from `base_commit[actor]`**. (A zero / unchanged
   read falls through.)
2. Stack drop from `prev_obs.stack_chips` (via
   `reconstructor.last_fs`) gated by either bet banner visibility
   **or** drop ≥ `min_bet_cents` (1 bb, ~2000 cents at defaults).
3. Banner alone with no chip amount → emit `OcrWarning`, break the
   walk (retry next tick).
4. Primary read that matches `base_commit` → trust it (CHECK or no
   action) — non-current seats only.
5. Timer-bar transition off the actor with a READABLE, UNCHANGED stack →
   CHECK (an unreadable stack holds the timer lock one more tick). Also
   when the turn PASSED: the next live seat holds the bar on two positive
   reads (no lock on the actor needed); hero's cards turning face up on
   hero's turn set the lock on hero.
6. Downstream evidence (`_any_remaining_delta`) → CHECK and continue.
7. Nothing → break.

Rules the ladder obeys (review 2026-09-20 — don't regress them):
- **Street gate**: the server advances streets with padded cards as soon
  as betting closes, so the engine can be a street AHEAD of the screen.
  The walk is skipped while `engine_view.street` exceeds the street implied
  by the visible boards (bounded at 25 ticks), and while no board card is
  readable at all (antes are not actions). Stale ovals after the closing
  action used to produce phantom CHECK cascades / a phantom raise.
- **All-in CALL is `check_call`**: stack 0 with `new_commit <= facing_bet`
  is a call (the engine rejects it as a raise and the action is lost);
  only `new_commit > facing_bet` with stack 0 is a (short) raise. A ±1
  engine-chip mismatch between a bet and its call counts as an exact call
  (`cents_to_engine_chips` is the ONE conversion, shared with the server).
- **Folds come only from `obs.folded`** — present in BOTH `last` and `fs`
  for pixel OCR (one frame for PokerNow's exact folds). A seat with chips
  in, facing a raise, with no new chips is NEVER fold-inferred: the walk
  waits. Frames where no seat reads in-hand are dropped entirely. The
  Task-C "hero hole hid" CHECK branch is deleted (ClubGG never re-hides
  hero's cards mid-hand; it only fired on glitches).
- **Baseline**: `last_fs` is replaced on every processed frame; only
  `stack_chips` is conservative — a carried-forward None stack is reduced
  by what the walk already explained, and an UNEXPLAINED drop is held (not
  absorbed) for seats that showed chip evidence this street.

`_any_remaining_delta` decides whether to continue the walk past a
CHECK. It checks two signals for any downstream seat — commit change
(with the same banner/stack-drop corroboration Fix N requires) and stack
drop > 0 AND ≥ `max(1, min_bet_cents)` — skipping the current actor,
sitting-out/folded/all-in seats and seats already explained this pass.
Require drop > 0 explicitly — `min_bet_cents == 0` in tests would
otherwise make zero-drop look like a hit.

### Session field cheat sheet

| Field | Owner | Semantics |
|---|---|---|
| `action_log` | server | List of `{gate, chips, seat}` entries; replayed positionally by `_rebuild_env` (`seat` is recorded and a mismatch with the engine's actor is WARNED, replay semantics unchanged) |
| `hand_in_hand_mask` | `_begin_new_hand` | Seats dealt into current hand (locked at hand-start; grows through expansion, hero included) |
| `folded_this_hand` | `OcrRunner._tick` | Seats folded this hand; re-derived from the engine after each rebuild in live mode |
| `sitting_out_seats` | refreshed every tick | `(all_seats - mask) \| folded_this_hand` |
| `_pending_button`, `_pending_sitting_out`, `_pending_stable_ticks`, `_pending_anchor_fs` | `_mirror_observable_state` | 2-tick debounce state |
| `observed_stacks`, `observed_pot` | `_mirror_observable_state` | Last OCR read, refreshed every tick regardless of hand state |
| `last_hero_hole` | `_begin_new_hand` | For rewind-proof detection of a new hand |
| `game_config` | `_begin_new_hand` (stacks), `/config` (bb/ante/dpb) | `GameConfig` with resolved_stacks |

### Running / testing the OCR loop

Record / replay (TEST-028): `PLO5BP_LIVE_RECORD=<dir>` makes each capture
session write `<dir>/live_<source>_<time>.jsonl` (header = session config,
then per tick the input FrameState or PokerNow payload, the EngineView,
events, action log, warnings; a PNG of every ClubGG frame that warned —
`ui/live/record.py`). `python -m plo5bp.ui.live.replay FILE` feeds it back
through `OcrRunner.process_frame` / `PokerNowRunner.handle_payload` and
reports the first tick that differs; recordings kept in
`tests/ocr/fixtures/sessions/` are replayed by `tests/ocr/test_live_replay.py`.

```bash
# UI + OCR runner
.venv/Scripts/python -m uvicorn plo5bp.ui.server:app --port 8765

# Kick off OCR
curl -X POST http://127.0.0.1:8765/ocr/start \
  -H 'Content-Type: application/json' \
  -d '{"window_match": "ClubGG", "poll_ms": 200}'

# Keep the current frame as a TEST FIXTURE (tracked folder, with the
# FrameState the extractor reads from it next to the PNG)
curl -X POST http://127.0.0.1:8765/ocr/save_frame \
  -H 'Content-Type: application/json' -d '{"to_fixtures": true}'
# → tests/ocr/fixtures/frames/frame_<epoch>.png + .state.json
# (no body → screenrecords/frames/debug_<epoch>.png: gitignored, local-only)

# OCR tests (synthetic FrameStates + server mirror + labeled fixtures)
.venv/Scripts/python -m pytest tests/ocr/ -q
```

### Known OCR pitfalls

- **ClubGG must use the 4-colour deck**: suits are told by colour (green
  clubs, blue diamonds, red hearts, black spades). On the 2-colour deck
  diamonds read as hearts and clubs as spades; after 3 hands / 15 cards
  without a club or diamond the live status warns (`tracking._check_deck_colours`).
- **The table window's shape matters**: ROIs are calibrated on 1927x1391;
  another aspect is refused, another size rescaled (`rois.frame_geometry`).
- **Tesseract misreads on the chip oval**: the "180" badge is small
  and sits on a noisy background. Expect `committed_chips` to come
  back `None` or `0` sometimes. Both cases are handled by the
  fallback ladder — do NOT paper over by tightening the ROI or
  assuming the OCR read is authoritative.
- **Banner only shows for ~300-500ms** during the chip-settle
  animation. Poll at ≤200ms to catch at least one banner frame.
- **`has_cards_back` threshold is close to noise** on ClubGG's
  silver-diamond backs (0.21-0.23 in-hand vs 0.15 threshold). If the
  banner covers ≥30% of the ROI, ratio drops to ~0.13. Fix 1's
  multi-signal OR routes around this, but don't tighten the
  threshold without simulating banner overlap first.
- **Gap-fill over error**: if an intermediate action is missed, the
  reconstructor walks forward from `engine_view.current_actor` and
  explains as many deltas as it can. Prefer adding fallback signals
  over raising.
- **Labeled ground-truth frames are LOCAL-ONLY and were lost
  (2026-06-26)**: `tests/ocr/fixtures/labels.json` referenced five
  frame PNGs under gitignored `screenrecords/frames/`; a disk cleanup
  deleted them (unrecoverable), so the two card-accuracy tests SKIP.
  New/recovered labeled frames go in **tracked**
  `tests/ocr/fixtures/frames/` (see its README) — never only in
  `screenrecords/`. Do not "fix" the skips by loosening accuracy
  thresholds.

## PokerNow live source (DOM, not OCR)

A second live-capture source for **PokerNow** (browser web app) sits alongside
the ClubGG pixel-OCR path. PokerNow renders the whole table as DOM elements, so
state is read directly — no Tesseract/HSV/ROIs. A Tampermonkey userscript
(`tools/pokernow/pokernow.user.js`) snapshots the table DOM on change and POSTs
a `pokernow.v1` JSON payload to `POST /pokernow/ingest`; the server's
`PokerNowRunner` maps it (`python/plo5bp/ocr/pokernow.py: map_payload`) into the
**same** `FrameState` → `EventReconstructor` → `Session` → `_rebuild_env`
pipeline ClubGG uses. Select the source in the UI top bar ("Live: ClubGG /
PokerNow"). Setup + transport rationale: `tools/pokernow/README.md`.

Things that differ from the ClubGG path (don't "fix" them to match):

- **Transport is HTTP POST via `GM_xmlhttpRequest`, not a websocket.** An https
  PokerNow tab can't reach `127.0.0.1` from page context (Chrome PNA +
  mixed-content); `GM_xmlhttpRequest` runs privileged and bypasses it. There
  is no websocket endpoint (it had no Origin check — any open page could
  inject snapshots; removed). `connected` status is recency-based (heartbeat
  every ~2s).
- **Exact data, so the reconstructor's OCR fallbacks never fire.** PokerNow
  gives the exact per-seat committed amount (`.table-player-bet-value`),
  explicit actor (`.decision-current`), and folds (`fold` class). The mapper
  emits authoritative `committed_chips` / `is_actor`, so `primary_read` always
  wins. No 2-tick debounce, no banner/timer-bar inference.
- **Variable seat count.** ClubGG is fixed 6; PokerNow tables vary, so the
  runner reconfigures `session.num_seats` per hand. Seats are ordered
  geometrically (hero = engine seat 0, then clockwise) from the userscript's
  angle reads — PokerNow's physical seat numbers don't encode CW order.
- **Hand-start triggers on hero's hole cards changing** (order-insensitive set
  compare), NOT on button rotation or card-disjointness. Hero's cards are dealt
  at hand start, always visible, exact, and re-dealt every hand — the dealer
  button DOM can lag the flop deal, and consecutive deals often share a card
  (so a ClubGG-style disjoint guard would suppress the trigger and the flop +
  recommendation wouldn't appear until the first action). `button_changed` is a
  secondary signal; `first_commit` bootstraps.
- **It still fires at the flop, not earlier.** Antes post as a per-seat bet
  *before* the flop; gating on flop-present means the anchor frame carries
  post-ante stacks + cleared street commits (right baseline), and the
  engine-posted antes aren't read as actions. No stack-plausibility gate (the
  OCR path has one) — PokerNow reads are exact and `_begin_new_hand` guards
  per-seat, so a villain who ante'd all-in doesn't block the hand-start.
- Both live runners register their reconstructor via `_set_active_reconstructor`
  so the shared `_begin_new_hand` rebaselines whichever source is driving —
  PokerNow re-registers on EVERY payload and `OcrRunner.stop()` retires its
  own (a ClubGG session used to leave the wrong reconstructor active, logging
  phantom ante RAISE/CALLs on every later PokerNow hand).
- Review 2026-09-20: a bare button change with an unchanged hero card set
  corrects the button and rebuilds — it does NOT restart the hand (button
  DOM lag); mid-street seeding is `stack + committed + ante`; tables with
  more than 8 seats are refused gracefully (the obs layout has 8 seat
  slots; reason shown in `/pokernow/status`); `map_payload` validates the
  schema and a malformed payload is a 400, never a 500 loop; a null stack is
  all-in ONLY with the explicit `allIn` flag (userscript ≥ 1.2.0 — re-paste
  it into Tampermonkey; its all-in DOM marker is still unverified against a
  live table).

## Current open issue

**Hero's silent CHECK — fixed in code (2026-09-28), awaiting a live
check.** Hero first to act postflop with no bet to face (heads-up bomb pot
OOP) checks and no chips move. The only CHECK signal used to be "the timer
bar was on hero and moved off" (it needs hero's own bar read as the lit one;
a hand-start clears that lock), so the UI sat on hero until villain put chips
in or the next card came. Now (`events.py`, TOOL-002): the turn PASSED —
the next live seat holds the bar on two positive reads — reads as the check,
and hero's cards turning face up (ClubGG flips them when hero's turn comes)
lock the turn on hero. Pinned by `tests/ocr/test_hero_silent_check.py`. The
old hypotheses: (1) Simple mode meant no action tracking at all — the toggle
now says "Track actions: Off/On"; (2) poll drift — ticks run on fixed
deadlines now; (3) the 1-2 px villain timer bars — see TOOL-012. What is left
is the owner's live check: record a session (`PLO5BP_LIVE_RECORD`, see
"Running / testing the OCR loop") with "Track actions: On" and replay it.
(The second open OCR item — anchoring a hand before ClubGG deducts the antes
— is handled by the ante-aware anchor whenever the pot label reads.) The
2026-09-20 review (findings, fixes, deferrals, repro scripts) lives in
`docs/reviews/`; the Tasks A/B/C investigation is in the git history of
`HANDOFF.md`.
