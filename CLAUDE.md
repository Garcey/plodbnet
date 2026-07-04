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
- `CentralCritic` sees all hole cards: training advantages, and (read-
  only) the trainer review's all-cards "true EV"
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
- **Warm-start pool seeding** (default ON with `--load-checkpoint`;
  `--no-warmstart-pool` disables): the opponent pool is ephemeral (too
  heavy for checkpoints), so a resume used to start EMPTY (pure
  self-play until the first snapshot tick). Now the pool is rebuilt
  from the loaded checkpoint's numbered siblings (`<stem>_<N>.pt`) —
  the exact prior membership when the checkpoint carries
  `pool_member_updates` (saved since 2026-07-03), otherwise the nearest
  files to the natural snapshot grid (`snapshot_every` spacing, oldest
  evicted first), walking further back when disk cadence is coarser so
  the pool still fills. Incompatible files (variant/head/shape) are
  skipped with a `[pool] skip` line. Logic + tests:
  `selfplay.select_warmstart_pool_updates` /
  `seed_pool_from_checkpoints`, `tests/python/test_warmstart_pool.py`.
  Pod watchdog relaunches get this automatically (they warm-start from
  the highest stem file).

```bash
# Serial rollout
.venv/Scripts/python scripts/train.py --hidden-dim 2048 --num-layers 4

# Batched rollout (Phase A-D speedup; 1.3× self-play, 2.06× pool-mix on CPU)
.venv/Scripts/python scripts/train.py --batched --hidden-dim 2048 --num-layers 4
```

Pod stem families: `optimized<N>` (v1, retired — `launch_auto.sh` /
`watchdog_auto.sh`), `vTwo<N>`/`vFour<N>` (PLO5 v2/v4 — guardian
scripts per stem, e.g. `vFour4_guardian.sh`), and `nlh<N>` (NLH v4 —
`nlh_guardian.sh`, cold-started nlh1 on 2026-07-03 after vFour4 was
PAUSED gracefully; vFour4 resumes later via warm-start + pool
seeding — do NOT prune `checkpoints/vFour4_*.pt` on the pod). Every
watchdog/guardian pgreps the same `scripts/train.py` — run ONE family
per pod; stop the other via its `runs/*.stop` file.

## PLO4/PLO6 variants (`plo4_double_bomb`, `plo6_double_bomb`)

PLO5 double-board bomb pot with **4 or 6 hole cards**; every other rule
is identical (two boards, pot-limit, ante-only, starts at the flop,
exactly-2-hole + 3-board eval). PLO6 built 2026-07-03, PLO4 built
2026-07-03; **neither is trained** — engines-first strategy, training
later.

- **Dims are deliberately identical to PLO5**: OBS_DIM 991 (the hero hole
  is a 52-dim multi-hot — 4 or 6 cards is just fewer/more bits), same
  11-anchor PL sizing head, critic input unchanged (opp multi-hot
  absorbs width).
- **NO cross-variant warm-start — every variant trains from scratch.**
  Identical dims made plo5→plo6 warm-starts *possible*, but the user
  vetoed them (2026-07-03): equities and minimum made-hand strengths
  differ drastically by hole count (PLO6 needs far stronger hands than
  PLO5; PLO4 needs weaker ones and rarely flops dual-board draws), so
  transferred weights are a confused prior, not a head start.
  `--allow-cross-variant-warmstart` was REMOVED; `train.py` now refuses
  any checkpoint/`--variant` mismatch unconditionally. Do not re-add
  the flag without an explicit user decision.
- **Batched IS supported**: the packers/encoder are
  hole-width-generic; `BatchedBombPotEnv` forwards `variant` (it didn't
  pre-2026-07-03 — a config with a non-default variant silently dealt
  PLO5). Batched==serial obs parity is pinned bit-exact in
  `tests/python/test_plo6_env.py` and `test_plo4_env.py`.
- Eval cost scales with C(hole,2) pairs × 10 board triples per
  hand-board: PLO4 60 combos (0.6× PLO5), PLO6 150 (1.5×).
- Encoding gotcha for future variants: several batch-encoder loops
  iterate hole and board columns; hole loops are `hole.shape[1]`-driven,
  board loops stay `range(5)` — do NOT merge them (hole width 4/6 ≠
  board width 5).
- NOT ported (deliberately, same as NLH): study mode / what-if
  (`new_study` is 5-card and variant-guarded — trainer UI + study are the
  product phase), OCR (N/A). Entropy seeds untuned; training ops TBD.

## NLH variant (`nlh_single`)

The engine/trainer serve a second game: single-board no-limit hold'em —
2 hole cards, best-5-of-7 any-combo eval, SB/BB blinds + per-player
ante, preflop betting round. Selected by
`GameConfig(variant="nlh_single", sb=...)` (or `GameConfig.nlh_default()`)
and `train.py --variant nlh_single`. The default stake mirrors the
target table: 5/10 with a 5 ante at bb=10000 chips → sb 5000,
ante 5000; 6-max preflop pot = 45000 ("$45").

- One `Variant` enum in Rust (`state.rs`) gates hole count, board
  count, PL-vs-NL cap, and the preflop round. The PLO path is the
  default variant and byte-identical to pre-variant behavior (all
  prior tests pass unchanged).
- Blinds post LIVE into `street_commit` (antes stay dead);
  `bet_to_call` = the NOMINAL bb (short posts go all-in, callers owe
  the full blind, side pots absorb it); `acted_this_street` stays
  false at posting so the BB option falls out of the existing
  round-close logic. Blinds are never history records (mirror of
  antes). Heads-up: button = SB acts first preflop; the generic
  first-alive-left-of-button rule already seats BB first postflop.
  Blind seats are STORED on the state and exposed via
  `observation_dict` (`sb_seat`/`bb_seat`) — never re-derive them
  (the assignment walk skips sitting-out seats).
- Sizing: v4 (μ,s) head over `NLH_ANCHOR_SPEC` — 12 anchors: min atom,
  25/33/50/66/80/100/125/160/200/275% pot, then an ALL-IN atom whose
  chips are always `max_raise` (the logistic's upper tail lands on it,
  so μ high = jam). `sizing.py` is spec-parameterized; the PLO spec
  reproduces the pre-spec closed forms bit-exactly (pinned by
  `test_anchor_grid_nlh.py`); refine brackets are neighbor-min-gap
  symmetric so u=0.5 still lands exactly on the anchor.
- Observations: `encoding_nlh.py`, `OBS_DIM_NLH = 995` — single board,
  history depth 40 (preflop adds actions; depth can't grow post-hoc),
  log1p scaling for SPR / history pot-frac / bet-faced (clips saturate
  at deep-NL scales), 3-dim exhaustive opp-outcome (ahead/tied/behind,
  Rust `nlh_opp_outcome_fractions`), hole-class block + SB/BB flags.
- Training: batched AND serial (batched is the default, same as PLO,
  since 2026-07-03). The batched path: `PyBatchedEngine` accepts
  `nlh_single`; the packer adds `sb_seat`/`bb_seat`/`nlh_opp_outcome`
  arrays and a variant-dependent history window (`history_cap`: PLO 32,
  NLH 40 — the packer keeps the NEWEST cap records, so the width MUST
  equal the encoder's depth or every history feature shifts);
  `encode_observation_batch_nlh` (encoding_nlh.py) is the vectorized
  encoder, bit-exact vs serial (pinned by
  `tests/python/test_nlh_env_batched.py` incl. preflop, >40-action
  histories, and terminal rewards). The Rust obs encoder
  (`PLO5_RUST_ENCODER=1`) stays PLO-only — the engine refuses it for
  NLH and env_batched gates it off. Block-rotation's PLO-tier default
  auto-disables under NLH; use `--stack-dist` (`deep` = 100-250bb
  matches the reference table) + `--entropy-coef`. Checkpoints carry
  `variant` + `anchor_count`; cross-variant warm-starts are refused;
  pool snapshots rebuild via state-dict sniffing (class + obs width +
  anchor spec).
- UI: PORTED 2026-07-03 — the study + trainer tabs serve NLH behind a
  format dropdown. Study enters at the PREFLOP via the NLH study path
  (`new_study_nlh` in Rust: 2-card hole, blinds posted; streets via
  `set_flop_nlh`/`set_turn_nlh`/`set_river_nlh` — the PLO dual-board
  setters refuse NLH states). Server: `FORMATS` registry (PLO5 model
  from stub.pt, NLH from `$PLO5BP_CHECKPOINT_NLH` /
  `checkpoints/nlh_stub.pt`; missing → random-init v4 placeholder
  flagged `model_loaded: false`), `POST /format` + `GET /formats`,
  per-variant card-spec shapes (NLH: hole 2, one flop, single
  turn/river, `flop_b` []), spec-aware 12-anchor recommendations
  (ALL-IN atom, `frac: null`). Trainer follows the format via
  `router.set_format` (fresh stake defaults: 100-250bb 5/10($5));
  scoring/EV/review are anchor-spec-generic; `describe_made_hand_nlh`
  = any-combo labels. Tests: `tests/python/test_nlh_ui.py`. NOT
  ported: OCR/PokerNow for NLH (deliberate — study/trainer only).
- Entropy seeds / (μ,s) floor-cap for the 12-anchor ladder are
  untuned — nlh1 launched 2026-07-03 with vFour4's proven hypers
  (0.45 ent, lr 1.5e-4 warmup 75, target-kl 0.5) via
  `scripts/nlh_guardian.sh`; watch the same health signals.

```bash
# NLH training (2048×4 rule applies to real runs, same as PLO)
.venv/Scripts/python scripts/train.py --variant nlh_single \
  --sizing-head logistic --hidden-dim 2048 --num-layers 4 \
  --stack-dist deep --num-seats-range "2,3,4,5,6"
```

## Promote good checkpoints to the UI

After a training run finishes, if the checkpoint looks good, push it to
the UI automatically — no need to ask. "Looks good" means the log tail
shows: finite losses throughout, entropy either dropping or flat (not
blowing up toward log(NUM_ACTIONS)≈2.2, not collapsing to 0 too fast),
`approx_kl` bounded under ~0.05, and `v_loss` stable. If any of those
are off, flag it and do NOT promote.

Mechanism: the UI loads one checkpoint PER FORMAT —
`$PLO5BP_CHECKPOINT` / `checkpoints/stub.pt` for PLO5 and
`$PLO5BP_CHECKPOINT_NLH` / `checkpoints/nlh_stub.pt` for NLH (see
`server.py:_load_model` + the `FORMATS` registry; a missing NLH stub
serves a random-init placeholder flagged "untrained" in the UI).
Promote by copying over the format's stub:

```bash
cp checkpoints/<run_name>.pt checkpoints/stub.pt       # PLO5
cp checkpoints/nlh1_<u>.pt  checkpoints/nlh_stub.pt    # NLH
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
  (pinned by `test_trainer_review.py`). What-if stays hero-only.
- Review steps EVERY decision node (hero + villain): the arrows drive a
  shared node cursor (`review_at_node` / `?node=` → `review.node_current`,
  one `_node_view` per click); pills still jump to hero decisions. Each
  node shows the acting seat's policy + DUAL EV — the actor's own (blind)
  value head AND the critic's all-cards "true EV" (UI loads `ckpt['critic']`
  via `server._load_critic`; opp multi-hot built with the canonical
  `rollout._rotate_opp_holes` so it matches training). Villain nodes have
  no graded score; hero nodes overlay the stored `DecisionRecord`.
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
  mixed-content); `GM_xmlhttpRequest` runs privileged and bypasses it. The
  `/pokernow/ingest` websocket endpoint also exists but is for non-userscript
  clients. `connected` status is recency-based (heartbeat every ~2s).
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
  so the shared `_begin_new_hand` rebaselines whichever source is driving.

## Public build (`PLO5BP_PUBLIC`)

Set `PLO5BP_PUBLIC=1` to serve the trainer + study tabs WITHOUT any live
capture — the public / monetizable variant (model served server-side, no
real-time table reading, so nothing that enables RTA). Single codebase,
one flag; unset (the default) is the full local build with OCR + PokerNow
intact. Run it with:

```bash
PLO5BP_PUBLIC=1 .venv/Scripts/python -m uvicorn plo5bp.ui.server:app --port 8765
```

- Server (`server.py`): the flag strips every `/ocr` and `/pokernow` route
  after registration (an `app.router.routes[:]` filter — the handlers stay
  defined, they're just unmounted, so those paths 404). The `/` route
  injects `window.PLO5BP_PUBLIC` into the served HTML so the client knows
  its mode before first paint (no `/config` fetch race, no flash of the
  live controls).
- Frontend (`app.js`): when `window.PLO5BP_PUBLIC`, `setupTopBar` skips the
  live-control wiring and hides `.ocr-group`, `init` skips `applySourceUI`
  (the OCR/PokerNow status polling), and the per-render `simple_ocr_mode`
  sync is gated off (it was what revealed the `#ocr-rescan-group` buttons).
- Trainer + Study are 100% live-independent: every study route rebuilds
  from user input via `_rebuild_env`. Study = manual hand entry → replay →
  recommendation; trainer = random deals. Nothing in either path calls the
  live routes, so gating is purely additive.

The public build also mounts the **service layer** (`python/plo5bp/ui/public.py`,
installed at the end of server.py only under the flag): Google sign-in
(Authlib; loopback-only dev login via `PLO5BP_DEV_LOGIN=1`), SQLite user DB
(`data/public.db`), free tier (5 trainer hands/UTC-day, middleware-enforced on
`POST /trainer/new_hand`; Study routes are subscriber-only → 402), Stripe
$10/mo subscriptions (checkout + success-redirect confirm + optional webhook +
lazy revalidation — no public URL needed), and `/admin` (users, comp
grant/revoke, revenue) allowlisted to `PLO5BP_ADMIN_EMAILS` (default
themilesgarcia@icloud.com). Setup/runbook: `PUBLIC_SETUP.md`. Tests:
`tests/python/test_public_service.py` (sets env + reimports ui modules).

Per-user state plumbing (matters when touching server.py/trainer.py):

- server.py's `session` is a **proxy** (`_SessionProxy`) over
  `_current_session()` — resolver installed by public.py returns the
  signed-in user's own `Session`; local build falls through to the single
  `_DEFAULT_SESSION`. Don't reassign `session`; mutate attributes (as all
  existing code does).
- trainer.py routes resolve their `TrainerSession` via `_ts()` +
  `set_session_resolver` the same way (default = the router's own instance;
  per-user stats persist to `data/trainer_stats/u<id>.json`).
- Trainer sampling seeds the GLOBAL torch RNG — every seed→sample region
  must hold `trainer._TORCH_RNG_LOCK` (opponent sampling + `_rollout_ev` MC
  block do). Keep that invariant if adding sampling paths.

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
