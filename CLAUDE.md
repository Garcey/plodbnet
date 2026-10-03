# WrapGTO — working notes

WrapGTO (the product, wrapgto.com) = this repo `plodbnet` = the Python package
`plo5bp` (older notes say "plodbbot"). Four parts:

- **Engine** — Rust (PyO3) under `rust_engine/`: the game rules, observation
  encoders, payouts, the NLH CFR solver.
- **Training** — Python PPO self-play under `python/plo5bp/` (+ `train/`,
  `evaluation/`, `scripts/`), run on a RunPod GPU pod.
- **Website** — the public build (`PLO5BP_PUBLIC=1`) of the FastAPI app in
  `python/plo5bp/ui/`: Study, Trainer, accounts, admin, and the private home
  games (clubs, tables, verified shuffle). One server behind a Cloudflare tunnel.
- **Local tools** — live capture (ClubGG OCR, PokerNow) that feeds Study, the
  CFR Solver desktop app, the GTO label pipeline. Never in the public build.

Correctness first throughout: observation encoding and engine state are
bit-exact reproducible; tests lean heavily on parity.

## Hard rules

- **Production is the owner's.** Live changes — code or model — only with the
  owner's explicit OK, every time; the owner runs `scripts/deploy_prod.sh`
  (runbook `docs/ops/PRODUCTION.md`). Never ssh/scp to the server or the pod on
  your own. Commit or push only when asked; work directly on `main`.
- **The GitHub repo is PUBLIC** until the owner makes it private: never commit a
  secret; `docs/improvements/` and `docs/reviews/` are local-only on purpose
  (their own `.gitignore`).
- **Training**: name the network size explicitly (train.py requires
  `--hidden-dim`, `--num-layers`, `--critic-hidden-dim`, `--critic-num-blocks`).
  Default training paths stay BIT-EXACT — prove it with
  `scripts/exactness_check.py --recipe all`; anything that changes numerics goes
  behind a default-OFF flag. Longer rollouts are always better (owner). Never
  change a guardian's recipe lines unasked. Details: `docs/training.md`.
- **Observations and determinism** (below) never change in place: a new feature
  value is a new `PLO5BP_OBS_REV`; the deal order and the verified-shuffle spec
  are public contracts.
- **Local services you affect, restart them yourself** (the UI on :8765); promote
  a good checkpoint to the LOCAL UI only (`python/plo5bp/ui/CLAUDE.md`).
- The engine binary must match its sources: rebuild with
  `.venv/Scripts/maturin develop --release` from the repo root after any Rust
  change (a stale build refuses to import; tests warn "STALE ENGINE").

## Where the notes live — read them before working in an area

| Area | Notes |
|---|---|
| Website: public build, sign-in, accounts, admin, Study, Trainer, Ranges, home games | `python/plo5bp/ui/CLAUDE.md` (loads by itself in `ui/`) |
| Live capture: ClubGG OCR, PokerNow | `python/plo5bp/ocr/CLAUDE.md` |
| Rust engine, variants (PLO4/5/6/67, NLH), config surface | `rust_engine/CLAUDE.md` |
| NLH CFR solver, GTO labels / PolicyNet, CFR Solver app | `python/plo5bp/gto/CLAUDE.md` |
| Training: PPO, rollout, env, encoding, flags, runs, guardians | **`docs/training.md`** — read it before touching training code |
| Training history: runs, sweeps, tuning, diagnoses | `docs/training-log.md` |
| Training: what has and hasn't worked, the current best recipe, how to test a change | `docs/training-what-works.md` — read it before planning a training change |
| Production: deploy, promote a model, roll back, restore, alerts | `docs/ops/PRODUCTION.md`, `ops/SERVER_SETUP.md` |
| Setting up this PC / the training pod | `SETUP.md` |
| Improvement backlog (local only) | `docs/improvements/README.md` |

## Current state (2026-09-30)

- **Main run: `vSix7`** (started 2026-09-29 on the owner's /goal) = r8c's recipe
  (all-actions gate term + KL magnet + entropy 0.025, actor 1024x3, SiLU critic
  1536x2) + the new obs-X inputs `line,run` (1246 dims; +~0.02 bb/seat-hand over
  r8c, `docs/design/OBS_X_DIMS_2026-09-29.md`). Since 2026-09-30 08:09 UTC (u1645,
  owner: "absolute maximum rollout") it runs at the host-RAM maximum: 1.76M envs,
  136M-row target = **~148.6M rows/update**, ~36.5 min/update, a checkpoint every
  update, peak RSS 207 GiB of the 233.8 GiB container (sizing in
  `docs/training-what-works.md`, "Pod lessons"). It runs from the
  pod's LOCAL disk (`/root/plx`, code = the side worktree
  `C:\Users\themi\plodbnet-exp`: all-actions + obs-X, NOT the main code): guardian
  `/root/plx/vx_guardian_local.sh` (STEM=vSix7; v2 backs the rollout off 8% if the
  trainer is ever OOM-killed), memory every 10 s in `/root/plx/runs/vSix7_mem.log`.
  To stop: `touch /root/plx/runs/vSix7.stop`, then kill the trainer right after a
  numbered save (`switch_vsix7_max.sh` shows how). The local disk is wiped if the pod
  restarts; a desktop loop copies checkpoints to `checkpoints/vSix7/`. The pod's `/workspace` (~20 GB
  quota) has hung on reads since 2026-09-28 ~16:40 UTC; the old `vSix6` guardian
  there is stopped. A vSix7 checkpoint can't go live until the site's encoder
  and the main code get the obs-X tail + all-actions option.
- **Live site model: `vSix6_1300`** (since 2026-09-26). Candidate `r8c_1460`
  (vs live +0.34–0.38 sampled / +0.23 argmax on two deal seeds; supersedes
  `avg_1380_1389`) awaits the owner's OK. Full detail: `docs/training.md`.

## Environment

- Windows 11, Python via `.venv/Scripts/python` (bash uses Unix paths
  — `/dev/null`, forward slashes).
- Torch supports both CPU and CUDA. `scripts/train.py` defaults to
  `--device cuda` (this PC's torch is CPU-only: pass `--device cpu`).
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
  code relies on the stack-delta / banner fallbacks. Once digit templates
  exist (`python/plo5bp/ocr/templates/digits.npz`, harvested from golden
  frames by `python -m plo5bp.ocr.tools.harvest_digits`), amounts are read
  in-process by `ocr/digits.py` (numpy only, ~2-5 ms) and Tesseract only
  reads what that reader is unsure of (TOOL-033; `PLO5BP_OCR_DIGITS=0` = off).
- OpenCV (`cv2`) and `pytesseract` are required only by the PIXEL half
  of the OCR stack (`extract`, `cards`, `live`). `plo5bp.ocr` imports
  lazily, so `events`, `types`, `text` parsing and the PokerNow mapper
  work — and their tests run — without OpenCV; the pixel test modules
  `importorskip` cv2 individually (never from `conftest.py`: a
  module-level skip there aborts the whole pytest session).

## Build & test

```bash
# Rebuild Rust extension into python/plo5bp/_engine.pyd (must run from
# repo ROOT — the root pyproject.toml has `module-name =
# "plo5bp._engine"` and `python-source = "python"`. Running maturin
# from rust_engine/ installs to the wrong place.)
.venv/Scripts/maturin develop --release

# Python tests (~3,140 as of 2026-09-28; ~12 min on this PC). Every run ends with the
# skipped tests grouped by reason; --strict-skips fails on one not listed in
# tests/expected_skips.txt (CI is always strict).
.venv/Scripts/python -m pytest tests/python/ tests/ocr/ -q --strict-skips
# One area: tests/python/{engine,training,site,homegame,gto,ops}/ (or tests/ocr/)
.venv/Scripts/python -m pytest tests/python/homegame/ -q
# Subsets by marker (applied per file by tests/conftest.py): -m training,
# -m "homegame or public or ui or ops", -m "not slow". Everything CI runs:
bash scripts/check.sh            # CI itself: .github/workflows/ci.yml

# Rust tests (~320). pyo3-build-config needs an interpreter: if cargo
# says "no Python 3.x interpreter found", export
# PYO3_PYTHON=<repo>\.venv\Scripts\python.exe and put the BASE python
# dir (python3.dll) on PATH first. `--profile fasttest` instead of
# `--release` skips the fat-LTO link (much faster rebuilds) and adds
# debug assertions + overflow checks; outputs are identical.
cargo test --manifest-path rust_engine/Cargo.toml --release --lib
# After ANY engine change: the golden digests of every engine output
# (bindings/golden_tests.rs, ~1 s) must not move unless the change is meant to.
cargo test --manifest-path rust_engine/Cargo.toml --profile fasttest --lib golden
# The Rust gate before committing (= CI's Rust job): cargo fmt --check, clippy
# -D warnings (policy: [lints.clippy] in rust_engine/Cargo.toml), the tests.
# Sets PYO3_PYTHON / PATH itself; `--fix` formats first. The compiler is pinned
# by rust-toolchain.toml (1.93.1); builds on the server and in CI use --locked.
bash scripts/rust_check.sh
# Criterion benchmarks of the hot kernels (rust_engine/bench, one core):
# `-- --save-baseline x` then `-- --baseline x` to compare a change. The default
# build targets any x86-64; RUSTFLAGS="-C target-cpu=native" is an opt-in for a
# machine that builds AND runs its own engine (the pod): ~3-4% faster full-obs
# encoding, bit-identical (SETUP.md; never for the server or CI's wheel).
bash scripts/rust_bench.sh [regex]

# Profile batched vs serial rollout
.venv/Scripts/python scripts/profile_rollout.py --num-envs 64 --rollout-length 2048
```

If maturin fails with "Couldn't find the symbol `PyInit_plo5bp_engine`",
it's being run from `rust_engine/` instead of repo root.

## Determinism contracts (don't break these)

- EV runout seed: hand base seed XOR `0x9E3779B97F4A7C15`.
- Canonical orderings for multi-sets (hole cards, boards) are fixed in
  the encoder. The design rules behind the observations: a multi-set (hole
  cards, a board) is a multi-hot or is sorted canonically before any
  per-position encoding — never deal order; every per-seat field is
  hero-rotated (slot 0 = hero, slot k = seat (hero + k) mod n), history
  included; all-in is its own mask, never inferred from stacks; seeded
  code instantiates a PINNED RNG from a u64 (no generic `impl Rng`); sizes
  that clamp to the same chips are masked by one rule (mask any sizing whose
  chips equal a lower-index legal action's).
- Opp-outcome MC seed (`engine.rs: outcome_mc_seed`, shared by
  `outcome_seed()` and `outcome_features_mc`): street + SORTED hero hole +
  each visible board as a SORTED set, through the hand-written `SeedMixer`
  (FNV-1a 64 → splitmix64 finalizer). No hero seat, no deal order, no
  `std` `DefaultHasher` (unspecified across Rust releases) — so dims
  982-989 are invariant to card order / table rotation and stable across
  toolchains. Study placeholder deals use the same mixer.

- Explicit-deck deal (`GameState::new_hand_from_deck`, `reset_with_deck`,
  `BombPotEnv.reset_with_deck`): the seeded deal IS this with
  `Deck::new_shuffled(seed)` — bit-identical observations and payouts, pinned
  by `tests/python/engine/test_reset_with_deck.py` and a Rust test. Deal order is a
  public contract the home games' verifiable shuffle depends on: `hole_slots`
  cards per seat INDEX (dealt in or not; = `hole_count` for every variant but
  PLO67, whose slots past the first four are the extras red burns hand out),
  seat 0 first, then full board A, then full board B, then the burns (PLO67
  only). Do not reorder it.

## Layout

```
rust_engine/src/              engine + PyO3 bindings
  engine.rs, state.rs, double_board.rs, hand_eval.rs, cards.rs
  bindings.rs                 shared helpers (variant, OBS_REV, validation) + the
                              module wiring of bindings/:
  bindings/serial.rs          PyGameState (serial) + plo67_runout_equities
  bindings/batched.rs         PyBatchedEngine: the one #[pymethods] block;
                              `encode(layout, indices, out, ...)` = the one
                              Rust-encoder entry (observation_encoded_* = aliases)
  bindings/pack.rs            packers (PackedCore shared by both layouts)
  bindings/encode*.rs         encode_core (blocks both layouts share), the full
                              1171 layout (+ v7 tail) and the minimal 796 one
  bindings/features.rs, compact.rs, rollout_ops.rs   feature pyfunctions,
                              compact obs rows, aggression kernels
  bindings/golden_tests.rs    golden digests of every engine output (run after
                              ANY engine change: cargo test ... golden)
  cfr/                        native NLH CFR solver (DCFR / ES-MCCFR), py_api

python/plo5bp/
  env.py, env_batched.py      Gym-style envs
  encoding.py, encoding_nlh.py  scalar + batch encoders, OBS_SEMANTICS_REV
  sizing.py                   canonical anchor-grid math (network/rollout/UI)
  rollout.py                  serial + batched + multiconfig rollout drivers
  compact_obs.py              compact (bit-packed) rollout observation storage
  ppo.py, selfplay.py         training loop, opponent pool
  network.py                  ActorCritic v1/v2/v4/v5 + CentralCritic
  actions.py, config.py, masking.py, eval.py, exploit.py
  gto/                        CFR → labels → PolicyNet teacher pipeline + host
  cfr_app/                    desktop "CFR Solver" app (FastAPI + pywebview)
  ocr/                        capture → extract → events pipeline
    live.py, rois.py, cards.py, text.py, extract.py,
    events.py, pokernow.py, types.py, tools/, templates/
  ui/
    server.py                 app factory (create_app, Site), Session, study routes
    live/                     live capture (local only): OcrRunner,
                              PokerNowRunner, hand-start machine, routes
    trainer.py, ranges.py     trainer mode, NLH range grid (local only)
    public.py, homegame.py    public service layer, private home games
    homegame_*.py             the home games' parts (schema, people, clubs,
                              stats, pages, routes, export) — see HGB-006
    handreview.py, handreview_store.py  Hand review (the paid page): ClubGG
                              hand-history zips -> records, all-in EV, grading
    runout.py, hand_describe.py, common.py
    static/                   index.html, app{.core,.table,.play,.study,.trainer,
                              .topbar,}.js, ranges.js, landing.js, admin.html,
                              games.html/.css + games{,.table,.play,.ui,.sound,.review}.js

tests/
  conftest.py                 markers per file, STALE ENGINE check, skip report
  python/                     one folder per area (each has a README):
    engine/ training/         engine rules + encoders + envs; PPO, rollout, checkpoints
    site/ homegame/           the website (public build, Study, Trainer); home games
    gto/ ops/                 CFR solver / GTO / CFR app; deploy + test tooling
    *.py, hg_mini_dom.js      shared helpers (bash_tools, cfr_fixtures, hg_client_tools)
                              (test_review_*.py = 2026-09-20 review regressions)
  ocr/                        extract / events / server_mirror / rois / cards
    fixtures/                 labeled frames + JSON state
  expected_skips.txt          the skips a machine may legitimately have

scripts/
  train.py, profile_rollout.py, evaluate.py, exploitability.py,
  probe_suite.py, smoke_test.py, *_guardian.sh (pod), cfr_*/gto_*/step*.py

tools/pokernow/               Tampermonkey userscript + README
tools/games_preview/          local preview harness (launcher, bots, felt measurements)
docs/plans/, docs/reviews/    plan files; code reviews + repros (docs/reviews is
                              local-only until the repo is private)
checkpoints/                  trained weights; stub.pt is UI default;
                              <stem>.optim.pt = rolling optimizer sidecar
screenrecords/frames/         debug captures from /ocr/save_frame
SETUP.md                      setting up a Windows PC / the training pod
docs/                         ops/PRODUCTION.md (runbook), models.md (what is live),
                              design/ (design + research history), learn/-style guides
ops/                          the production server: unit, backup, alerts, deploy half
requirements/                 exact package versions per machine
.github/workflows/ci.yml      CI (tests, Rust, shell, secrets, the Linux engine wheel)
```
