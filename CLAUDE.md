# plodbbot — working notes

PLO5 double-board bomb-pot self-play PPO + live OCR study tool.

- **Engine**: Rust (PyO3) under `rust_engine/`.
- **Trainer**: Python PPO under `python/plo5bp/` (`env.py`,
  `env_batched.py`, `ppo.py`, `rollout.py`, `selfplay.py`).
- **Study UI**: FastAPI + HTML/JS single-page app at
  `python/plo5bp/ui/` (server.py + static/). Replays action_log
  through the engine and shows model recommendations.
- **OCR integration**: `python/plo5bp/ocr/` — captures the live ClubGG
  window, extracts `FrameState`, and feeds an `EventReconstructor`
  that emits action events the UI applies to its session.

Correctness-first throughout: observation encoding and engine state
are bit-exact reproducible; tests lean heavily on parity.

## Environment

- Windows 11, Python via `.venv/Scripts/python` (bash uses Unix paths
  — `/dev/null`, forward slashes).
- Torch supports both CPU and CUDA. Default training device is CPU
  (`scripts/train.py --device cpu`); pass `--device cuda` for GPU.
  RTX 3070 / Ampere uses cu128 wheels:
  `.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128`.
  UI inference reads `PLO5BP_DEVICE` (default `cpu`); auto-falls back to
  CPU if cuda is requested but unavailable. The 2048×4 architecture with
  residual connections needs GPU; the legacy 128×2 net still trains fine
  on CPU.
- Rust extension module: `plo5bp._engine` (defined in
  `rust_engine/src/lib.rs`, wired via root `pyproject.toml`).
- Tesseract binary is required for chip-amount / pot OCR. If it isn't
  installed, `plo5bp.ocr.text` helpers return `None` and downstream
  code relies on the stack-delta / banner fallbacks.
- OpenCV (`cv2`) and `pytesseract` are both required by the OCR
  extras; tests `importorskip` them.

## Build & test

```bash
# Rebuild Rust extension into python/plo5bp/_engine.pyd (must run from
# repo ROOT — the root pyproject.toml has `module-name =
# "plo5bp._engine"` and `python-source = "python"`. Running maturin
# from rust_engine/ installs to the wrong place.)
.venv/Scripts/maturin develop --release

# Python tests (250 as of 2026-04-24: 208 engine/trainer + 42 OCR)
.venv/Scripts/python -m pytest tests/python/ tests/ocr/ -q

# Rust tests
cargo test --manifest-path rust_engine/Cargo.toml

# Profile batched vs serial rollout
.venv/Scripts/python scripts/profile_rollout.py --num-envs 64 --rollout-length 2048
```

If maturin fails with "Couldn't find the symbol `PyInit_plo5bp_engine`",
it's being run from `rust_engine/` instead of repo root.

## Training

**Always use the 2048×4 network: pass `--hidden-dim 2048 --num-layers 4`
on every training invocation.** The script defaults to 128×2 (legacy
size); training at the default silently produces a smaller, weaker
model — already cost a multi-day run mistaken for a 2048×4 result.
There is no scenario in this project where 128×2 is the right
architecture; if a flag is missing, add it. The same rule applies to
`launch_vtwo.sh` (it hardcodes 2048×4).

`scripts/train.py` is **v2-only** (anchor sizing head + centralized
critic, `head_version: 2` checkpoints; the critic's state rides in the
checkpoint under `"critic"`). It refuses to warm-start v1 checkpoints.
v2 specifics:

- Sizing head: 11 pot-fraction anchors (0=min,10%,…,100%=pot) with
  per-anchor Beta refinement sliders; canonical chips/legality math in
  `python/plo5bp/sizing.py` (shared by network/rollout/UI — don't fork it).
- `CentralCritic` sees all hole cards during training only
  (`--critic-hidden-dim 1536 --critic-num-blocks 2` defaults); the
  actor keeps its own observation-only value head for the UI display.
- Log line: `v` is the critic loss, `vd` the display-head loss, and
  `Hg/Ha/Hb` decompose entropy into gate/anchor/beta.
- Entropy coefs seed at clubgg:0.10/clubgg_deep:0.12/deep:0.18
  (raised 2026-06-11 from v1's cold-start values; the v2 anchor head
  is a harder exploration problem) — NOT the annealed floors v1 later
  earned. The anneal only walks down, so err high: a 0.02 cold start
  collapsed gate entropy within 10 updates (2026-06-10).
- `--target-kl` (default 0.5) is the KL guard: aborts the PPO inner
  loop before the optimizer step when a minibatch's |approx_kl|
  exceeds it (logged as `KLSTOP@mbN`). vTwo2 collapsed at update 173
  (approx_kl ≈ +2417 → entropy pinned at 0) without it; the v2
  discrete anchor head has heavier-tailed importance ratios than v1's
  continuous Beta, which is why v1 never needed this. 0 disables.
- Anneal: no decisions (no baselines, no lowering) until
  `--anneal-start-update` (default 600) updates; tolerance default 1.0
  (30/30/30→29/29/29 still lowers — absorbs seat/stack block variance).
  Live-tune WITHOUT pausing training via `runs/anneal_control.json`:
  `{"step": 0.003}` changes the decrement, `{"tier_ent": {"deep":
  0.08}}` manually sets a tier's coef (one-shot; anneal continues from
  there). Applied whenever file content changes.
- `--kl-anchor-coef` (default 0 = off) enables the KL-to-EMA-reference
  regularizer; the reference is not persisted in checkpoints.

```bash
# Serial rollout
.venv/Scripts/python scripts/train.py --hidden-dim 2048 --num-layers 4

# Batched rollout (Phase A-D speedup; 1.3× self-play, 2.06× pool-mix on CPU)
.venv/Scripts/python scripts/train.py --batched --hidden-dim 2048 --num-layers 4
```

Pod stem families: `optimized<N>` (v1, retired — `launch_auto.sh` /
`watchdog_auto.sh`) and `vTwo<N>` (v2 — `launch_vtwo.sh` /
`watchdog_vtwo.sh`; cold-starts vTwo1 when no vTwo checkpoints
exist). Both watchdogs pgrep the same `scripts/train.py` — run ONE
family per pod; stop the other via its `runs/watchdog_*.stop` file.

## Promote good checkpoints to the UI

After a training run finishes, if the checkpoint looks good, push it to
the UI automatically — no need to ask. "Looks good" means the log tail
shows: finite losses throughout, entropy either dropping or flat (not
blowing up toward log(NUM_ACTIONS)≈2.2, not collapsing to 0 too fast),
`approx_kl` bounded under ~0.05, and `v_loss` stable. If any of those
are off, flag it and do NOT promote.

Mechanism: the UI loads `$PLO5BP_CHECKPOINT` or falls back to
`checkpoints/stub.pt` (see `python/plo5bp/ui/server.py:_load_model`).
Promote by copying over `stub.pt`:

```bash
cp checkpoints/<run_name>.pt checkpoints/stub.pt
```

The server loads `MODEL` once at import, so a running UI needs a
restart to pick up the new weights.

The UI serves BOTH checkpoint generations: `_load_model` sniffs the
head class (`anchor_head.weight` → v2, `raise_head.weight` → v1) and
the trained obs width (959-era v1 checkpoints get the exact
`downgrade_obs_to_v1` projection via `network.obs_adapter`; the
encoder always emits 991). v2 recommendations carry an `anchors`
histogram + `rec_anchor` + `refine` instead of `beta_alpha/beta_beta`;
trainer scoring snaps the user's raise size to the nearest legal
anchor (`score_move_v2`).

## Restart services yourself

If an action you just took (promoting a checkpoint, editing
`server.py`, etc.) requires a service restart to take effect, do the
restart — don't ask. This applies to the UI server specifically and to
any other local dev service we own in this repo. Check for an existing
instance (`netstat -ano | grep :8765` for the UI), stop it if found,
and start the new one in the background. Report the new URL/PID so the
user can find it.

Start the UI with:

```bash
.venv/Scripts/python -m uvicorn plo5bp.ui.server:app --port 8765
```

(Drop `--reload` in a scripted background start — it spawns a reloader
subprocess that complicates clean shutdown.)

## Trainer mode (GTO-Wizard-style practice)

`python/plo5bp/ui/trainer.py` — a second UI mode (tab next to Study, or
`/?mode=trainer`) that deals random bomb-pot hands and drills the user
against network opponents. Backend rides on the same app/MODEL under
`/trainer/*` (`create_trainer_router`), state fully separate from the
study `Session`. Shared pure helpers live in `python/plo5bp/ui/common.py`.

Key invariants (documented in the module docstring — don't break):

- A hand is fully determined by `(config, seed, button)`; review /
  repeat / EV-loss all rebuild by replaying the recorded `action_log`
  through a fresh `env.reset(seed, button)`. Nothing snapshots live
  engine objects.
- Trainer terminal state is synthesized from the `done` flag —
  `study_terminal` / `awaiting_next_street` are study-mode-only engine
  fields and stay `None` in random-deal mode.
- Opponents sample the mixed strategy via `eval.model_policy(...,
  deterministic=False)`, re-seeded per node from `(hand seed, action
  prefix length)` — behavior is a pure function of the action prefix.
  trainer.py is the ONLY consumer of torch's global RNG in the UI
  process (study path is always `deterministic=True`).
- Scoring (`score_move` + the `SCORING` dict): gate-probability ratio ×
  Beta-PDF size quality → 0-100 score → best/correct/inaccuracy/wrong/
  blunder. EV loss = paired Monte-Carlo rollouts (common random
  numbers) of user action vs the deterministic rec; `mc_rollouts`
  default 16 keeps a deviating `/trainer/act` under ~1s on CPU with the
  2048×4 net (matching actions skip MC entirely).
- What-if card swaps replay through `reset_study` — an unmodified
  what-if reproduces the original node's observation bit-exactly
  (pinned by `test_trainer_review.py`).
- `all_hole_cards()` (engine accessor added for this) reveals opponent
  cards — projection exposes them only at terminal/review.
- Lifetime stats + settings persist to `checkpoints/trainer_stats.json`
  (override with `PLO5BP_TRAINER_STATS`); session stats are in-memory.

Tests: `tests/python/test_trainer_*.py`, `test_all_hole_cards.py`
(shape parity with the study `_state_dict` is pinned — if you add a key
to the study projection, mirror it in `_trainer_state_dict`).

## Rollout paths

Two drivers in `python/plo5bp/rollout.py`:

- `collect_rollout` — serial, one env at a time. Used by UI, exploit
  probe, eval. Must stay bit-exact; don't refactor for speed.
- `collect_rollout_batched` — Phase A-D: `BatchedBombPotEnv` wraps
  `PyBatchedEngine`, encoder is vectorized, opponents are grouped by
  pool-snapshot index. Bit-exact parity vs serial is *not* asserted
  (RNG-consumption order differs); parity is at the env level (see
  `tests/python/test_env_batched.py`) and the training smoke.

## Determinism contracts (don't break these)

- EV runout seed: hand base seed XOR `0x9E3779B97F4A7C15`.
- Canonical orderings for multi-sets (hole cards, boards) are fixed in
  the encoder — see memory note on engine design preferences.

## Config surface

`GameConfig(num_seats, starting_stack, ante, bb)`. Default is 6-seat,
20bb, 3bb ante. Seats and stacks are meant to vary across training
runs — see the project-direction memory note. Action space keeps full
pot-fraction enum even when some sizes dupe AllIn at shallow stacks.

---

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
OcrRunner._tick (server.py)  →  mutates session, calls _rebuild_env
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
- `python/plo5bp/ui/server.py` — FastAPI backend.
  - `Session`: big mutable state bag (cards, action_log, button,
    participant mask, anchor_fs, config).
  - `_rebuild_env`: canonical path. Pads card spec with unused deck
    indices, reset_study, replays action_log with
    `_auto_fold_sitting_out` for retired seats.
  - `_mirror_observable_state(fs)`: mirrors card spec + runs the
    debounced hand-start machine (see below).
  - `OcrRunner._tick`: capture → extract → mirror → reconstruct → apply
    events → rebuild env. One task per process.

### Unit system

OCR reads are in **cents** (ClubGG's dollar display × 100, so "$180"
is 18000). Engine carries **chips** where `cfg.bb` chips = 1 big blind
= `dollars_per_bb` dollars. At defaults (bb=10000, $20/bb),
`chips_per_cent = 5.0`, so 18000 cents → 90000 engine-chips. **Every
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
   (`hand_in_hand_mask` empty).
6. `_begin_new_hand`: resets defaults, locks
   `hand_in_hand_mask = {i : anchor.seats[i].folded is False}`, seeds
   `cfg.starting_stacks` from the anchor frame's OCR reads, and calls
   `ocr_runner._reconstructor.rebaseline(anchor_fs)` so diff-based
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
- `folded_this_hand` only grows during a hand; it's cleared by
  `_new_session_defaults` which `_begin_new_hand` calls.

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
   action).
5. Nothing → break.

`_any_remaining_delta` decides whether to continue the walk past a
CHECK. It checks the same three independent signals (commit change,
stack drop > 0 AND ≥ `max(1, min_bet_cents)`, banner) for any
downstream seat. Require drop > 0 explicitly — `min_bet_cents == 0`
in tests would otherwise make zero-drop look like a hit.

### Session field cheat sheet

| Field | Owner | Semantics |
|---|---|---|
| `action_log` | server | List of `{gate, chips}` entries; replayed by `_rebuild_env` |
| `hand_in_hand_mask` | `_begin_new_hand` | Seats dealt into current hand (locked at hand-start) |
| `folded_this_hand` | `OcrRunner._tick` | Seats the reconstructor has emitted FOLD for this hand |
| `sitting_out_seats` | refreshed every tick | `(all_seats - mask) \| folded_this_hand` |
| `_pending_button`, `_pending_sitting_out`, `_pending_stable_ticks`, `_pending_anchor_fs` | `_mirror_observable_state` | 2-tick debounce state |
| `observed_stacks`, `observed_pot` | `_mirror_observable_state` | Last OCR read, refreshed every tick regardless of hand state |
| `last_hero_hole` | `_begin_new_hand` | For rewind-proof detection of a new hand |
| `game_config` | `_begin_new_hand` (stacks), `/config` (bb/ante/dpb) | `GameConfig` with resolved_stacks |

### Running / testing the OCR loop

```bash
# UI + OCR runner
.venv/Scripts/python -m uvicorn plo5bp.ui.server:app --port 8765

# Kick off OCR
curl -X POST http://127.0.0.1:8765/ocr/start \
  -H 'Content-Type: application/json' \
  -d '{"window_match": "ClubGG", "poll_ms": 200}'

# Debug: grab a single frame for fixtures
curl -X POST http://127.0.0.1:8765/ocr/save_frame \
  -H 'Content-Type: application/json' \
  -d '{"window_match": "ClubGG", "poll_ms": 200}'
# → screenrecords/frames/debug_<epoch>.png

# OCR tests (synthetic FrameStates + server mirror + labeled fixtures)
.venv/Scripts/python -m pytest tests/ocr/ -q
```

### Known OCR pitfalls

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

## Current open issue

**Hero's silent CHECK on the flop is not detected by the UI.** When
hero is first to act on a postflop street with no facing bet (e.g.,
heads-up bomb-pot OOP) and clicks CHECK, the existing walk ladder
in `events.py:_infer_seat_actions` has no positive signal to emit
`SeatAction(gate="check_call")` until either fastaf acts (allowing
`_any_remaining_delta` to fire) or the next street deals (allowing
the StreetReveal CHECK reconciler to back-fill). Mid-street the UI
sits stuck on hero as `current_actor`.

See `HANDOFF.md` at repo root for the latest symptom, the three
fixes shipped this session (Tasks A/B/C), and the open hypotheses
for why Task C did not resolve the bug. Plan files live in
`.claude/plans/`; the most recent is
`in-the-last-session-mossy-hedgehog.md`. Treat every hypothesis in
HANDOFF.md as speculation, not fact — Task C shipped on a documented
ClubGG behavior assumption that may not hold empirically.

## Layout

```
rust_engine/src/              engine + PyO3 bindings
  PyGameState (serial), PyBatchedEngine (batched)

python/plo5bp/
  env.py, env_batched.py      Gym-style envs
  encoding.py                 scalar + encode_observation_batch
  rollout.py                  serial + batched rollout drivers
  ppo.py, selfplay.py         training loop
  network.py                  ActorCritic (gate head + Beta raise head)
  actions.py, config.py, masking.py
  ocr/                        capture → extract → events pipeline
    live.py, rois.py, cards.py, text.py, extract.py,
    events.py, types.py, tools/, templates/
  ui/
    server.py                 FastAPI app, Session, OcrRunner
    static/                   index.html, app.js, style.css

tests/
  python/                     engine / encoder / env / rollout parity
  ocr/                        extract / events / server_mirror / rois / cards
    fixtures/                 labeled frames + JSON state

scripts/
  train.py, profile_rollout.py, evaluate.py,
  exploitability.py, smoke_test.py

.claude/plans/                approved plan files
checkpoints/                  trained weights; stub.pt is UI default
screenrecords/frames/         debug captures from /ocr/save_frame
HANDOFF.md                    latest unresolved-issue handoff
```
