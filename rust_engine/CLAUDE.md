# Rust engine & game variants — notes

Loaded when you work under `rust_engine/`. The determinism contracts are in the root CLAUDE.md; build/test commands too. The CFR solver (`src/cfr/`) has its own notes in `python/plo5bp/gto/CLAUDE.md` — read them before changing the solver.

Keep non-source files OUT of `src/`: the engine's `SOURCE_HASH` (build.rs, and its twins in tests/conftest.py and ops/deploytool.py) covers every file there, so a notes file under `src/` makes the built engine look stale (tests skip, the deploy refuses to ship).

## PLO4/PLO6 variants (`plo4_double_bomb`, `plo6_double_bomb`)

PLO5 double-board bomb pot with **4 or 6 hole cards**; every other rule
is identical (two boards, pot-limit, ante-only, starts at the flop,
exactly-2-hole + 3-board eval). PLO6 built 2026-07-03, PLO4 built
2026-07-03; **neither is trained** — engines-first strategy, training
later.

- **Dims are deliberately identical to PLO5** (OBS_DIM 1171 today; the hero hole
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
  `tests/python/engine/test_plo6_env.py` and `test_plo4_env.py`.
- Eval cost scales with C(hole,2) pairs × 10 board triples per
  hand-board: PLO4 60 combos (0.6× PLO5), PLO6 150 (1.5×).
- Encoding gotcha for future variants: several batch-encoder loops
  iterate hole and board columns; hole loops are `hole.shape[1]`-driven,
  board loops stay `range(5)` — do NOT merge them (hole width 4/6 ≠
  board width 5).
- NOT ported (deliberately, same as NLH): study mode / what-if
  (`new_study` is 5-card and variant-guarded — trainer UI + study are the
  product phase), OCR (N/A). Entropy seeds untuned; training ops TBD.
- PLO6 IS dealt in the home games (2026-09-26, "PLO6 tables" under Home games):
  untrained, so those hands are never graded; a trained PLO6 net would be wired
  in through `homegame.GAMES["plo6"]["graded"]` + a PLO6 `grade_hand` model.

### PLO67 (`plo67_double_bomb`, 2026-09-27 — the owner's friends' format)

PLO double-board bomb pot with FOUR hole cards and the three burn cards dealt
FACE UP (one before the flops, turns, rivers). Every RED burn (diamond or heart)
deals every seat still in the hand — all-in seats too, folded / sitting-out seats
not — one more hole card: 4-5 on the flop, 4-6 on the turn, 4-7 on the river.
Exactly 2 hole + 3 board per board, as ever. Home games only (serial engine).

- Engine (`state.rs` / `engine.rs`): `Variant::hole_count` = cards at the DEAL (4),
  `hole_slots` = the most a seat can hold (7 — every other variant: `hole_count`),
  `burn_count` (3, else 0), `max_seats` = (52 - 10 - burns) / slots = **5**,
  `cards_needed(n)`. Deal order extends the contract (unchanged for the others):
  `hole_slots` per seat INDEX (a seat's slots past `hole_count` are its reserved
  extras, `GameState.extra_holes`, handed out front first), board A, board B, then
  the burns (`full_burns`). `reveal_burn` runs before a street's board cards —
  at the deal for the flop, in `close_round_or_run_out` for the turn / river (so
  run-outs deal extras too); a fold-out turns up no more burns (`burns` = those
  seen). `hole_count_on(seat, street)` = 4 + min(red burns up to that street,
  extras received) — a seat's extras are a prefix of the red burns. `payouts_ev`
  returns the actual deal for PLO67 (undealt burns change the HANDS). Every PLO
  evaluator takes 4..=7 hole cards (`PAIRS_7`, `plo_pairs`, `MAX_PLO_HOLE`) —
  exact for the other variants (pinned vs brute force). The batched engine
  refuses PLO67 (fixed-width packers; nothing trains it).
- Bindings: `observation_dict()["burns"]`, `all_burns()` (reveal
  accessor — the rabbit), `hole_count_on(seat, street)`, and the pyfunction
  `plo67_runout_equities(holes, board_a, board_b, dead, samples, seed)`: Monte
  Carlo whole runouts dealt the game's way (burn, a card to every hand if red,
  a card per board), ~10 ms / 3000 samples; pinned against the engine's own
  deals (`runout_equities_match_the_engine_deal_distribution`).

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
  `tests/python/engine/test_nlh_env_batched.py` incl. preflop, >40-action
  histories, and terminal rewards). The Rust obs encoder
  (`BatchedEngine.encode`) stays PLO-only — the engine refuses it for
  NLH and env_batched gates it off. Use `--stack-dist` (`deep` =
  100-250bb matches the reference table) + `--entropy-coef`. Checkpoints carry
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
  = any-combo labels. Tests: `tests/python/site/test_nlh_ui.py`. NOT
  ported: OCR/PokerNow for NLH (deliberate — study/trainer only).
- Entropy seeds / (μ,s) floor-cap for the 12-anchor ladder are
  untuned. The NLH PPO lineage (nlh1–nlh4, launched 2026-07-03 with
  vFour4's hypers) was ABANDONED 2026-07-16 and its checkpoints deleted;
  `scripts/nlh_guardian.sh` is a retired stub that exits 1. NLH strategy
  now comes from the CFR teacher pipeline below; the PPO path still
  trains (tests cover it) but nothing runs it.
- Engine (review 2026-09-20): a seat that already covers everything any
  live opponent can still put in gets NO decision node (the nominal
  `bet_to_call = bb` used to offer a covering SB a fold vs a short all-in
  BB and forfeit uncalled chips) — the hand runs out; uncalled/orphan
  layers are refunded to their contributors. Hands can therefore be
  TERMINAL AT DEAL when fewer than two seats can act: `env.reset` returns
  `info.terminal=True, actor=None`; use `env.terminal_rewards()`.

## Config surface

`GameConfig(num_seats, starting_stack, ante, bb)`. Default is 6-seat,
20bb, 3bb ante. Seats and stacks vary across training runs (per-seat
stacks, 2-6 seats — the multi-config tiers). The sizing head keeps its full
anchor grid even when some sizes collapse to all-in at shallow stacks (the
engine's dup-masking drops them there; they are distinct deeper).

`GameConfig.reach_cap` (2026-10-02; Python `GameConfig(reach_cap=...)`, the
serial `PyGameState(reach_cap=)` kwarg + getter) picks the bet ceiling.
`true` (every constructor's default) = the rule every network is trained on and
Study / the Trainer / the CFR solver play: `max_bet_total` is also capped at what
the deepest alive opponent can still put in, and `min_raise_chips` clamps the floor
down to cover a shorter opponent. `false` = the home games' rule (owner: "capped at
the pot or your own stack size, whichever is smaller"): pot limit (NL: own stack)
only, the floor stays the floor, and what nobody matches comes back through the
payout's one-seat layer. Either way a bet needs an alive opponent able to put in
more than `bet_to_call` (else min = max = 0 and the mask is fold / call), so the
legal mask, AllIn and `max_raise` stay consistent. The two rules are strategically
equivalent (an overbet's excess is refunded), which is why the home games' grader
replays a hand under the capped rule with the raise chips clamped into
`[min_raise_chips, max_raise_chips]`. The batched engine (`env_batched`) refuses
`false`; default-path numerics, golden digests and `exactness_check --recipe all`
are unchanged. Tests: `reach_cap_off_tests` (engine.rs),
`tests/python/engine/test_reach_cap.py`.

---
