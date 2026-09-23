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

# Python tests (2,100 as of 2026-09-20; ~3.5 min on CPU)
.venv/Scripts/python -m pytest tests/python/ tests/ocr/ -q

# Rust tests (~290). pyo3-build-config needs an interpreter: if cargo
# says "no Python 3.x interpreter found", export
# PYO3_PYTHON=<repo>\.venv\Scripts\python.exe and put the BASE python
# dir (python3.dll) on PATH first.
cargo test --manifest-path rust_engine/Cargo.toml --release --lib

# Profile batched vs serial rollout
.venv/Scripts/python scripts/profile_rollout.py --num-envs 64 --rollout-length 2048
```

If maturin fails with "Couldn't find the symbol `PyInit_plo5bp_engine`",
it's being run from `rust_engine/` instead of repo root.

## Training

**Always name the network size explicitly: pass `--hidden-dim` and
`--num-layers` on every training invocation.** The script defaults to
128×2 (legacy); an unflagged run once trained the default for days and
was mistaken for a 2048×4 result — the lesson is "never train a size by
accident", not "only 2048×4". Sizes by lineage: the full-obs `vSix` stems
use 2048×4 (`launch_vtwo.sh` hardcodes it too); the minimal-obs `vMin1`
stem DELIBERATELY uses a small net (actor 128×3, critic 128×2 — see
`scripts/vMin1_guardian.sh`). A network-size study (owner, 2026-08/09)
found many unused neurons; 128×3 is the last size tested and the owner
believes it is STILL too big (dead neurons/connections). The small net is
what buys vMin1's much longer rollout (more data per update). Do not
"correct" vMin1 up to 2048×4.

**Longer rollouts are always better (owner, 2026-09-23).** PLO5
double-board bomb pots stack an enormous space of hole/board card
combinations on an effectively unbounded game tree, on top of a wide
spread of seat/stack configurations, on top of massive variance. More rows
per update means each update sees more near-identical spots AND how small
differences between them shift the strategy — the precise, solver-like
strategy this project wants. So memory/time saved anywhere in the loop
should be spent on `--rollout-length`; prefer designs that let it grow.

Roadmap (2026-09-23): (1) make the training loop more efficient (compact
observation storage etc.), (2) real timing runs on RunPod, (3) a size
sweep for BOTH actor and critic to find the optimal network size, (4) a
full training run.

`scripts/train.py` is **v2-family-only** (centralized critic; the
critic's state rides in the checkpoint under `"critic"`). It refuses to
warm-start v1 checkpoints, and head families are strict: `--sizing-head
anchor` = v2 (head_version 2), `logistic` = v4 (3), `mixture` = v5 (4).

### Current state (2026-09-20 — read this before the older v5 notes)

- **Obs**: PLO OBS_DIM is **1171** (v7-obs tail 1020..1171: STK/BRD/DUAL
  blocks, V7_DESIGN.md / V7_OBS_CANDIDATES.md), minimal layout 796
  (`--obs-mode minimal`, vMin1), NLH 995. The fused **Rust obs encoder is
  live again** at width 1171 (`PLO5_RUST_ENCODER=1`, width-gated in
  `env_batched.py`; the pod guardians set it). Production stems: `vSix<N>`
  (`--v6` preset) and `vMin1`; guardians `scripts/vSix4_guardian.sh`,
  `scripts/vMin1_guardian.sh`.
- **Observation-SEMANTICS revision (`PLO5BP_OBS_REV`)**. The 2026-09-20
  review fixed feature VALUES without moving any dim: STK-2 / STK-5[2:4]
  (1024-1029, 1040-1041) and the min/max-bet scalars (PLO 186/187, minimal
  slots 2/3, NLH 134/135) now describe the LEGAL raise window (engine
  `min_raise`/`max_raise`, zeros when Raise is illegal, dead-chip
  invariant); the straight-draw flags (800/802) put the ace-low shadow
  below the deuce; flush blockers (999-1006) see the other board; NLH
  flush-nut distance (906) on five-flush boards. Unset/`2` = fixed values
  (default); **`PLO5BP_OBS_REV=1` reproduces the pre-fix values exactly**
  in all three encoders — use it to serve or resume any checkpoint trained
  before 2026-09-20. Read once at import (`encoding.OBS_SEMANTICS_REV`);
  the Rust engines are pinned to the same rev at construction. `train.py`
  stamps `obs_rev` into checkpoints and REFUSES a warm start across a rev
  change unless `--allow-obs-rev-change` (deliberate migration — expect a
  transient); pool seeding skips other-rev siblings; the UI logs
  `OBS-REV MISMATCH` and exposes `obs_rev_mismatch` in `GET /formats`; GTO
  PolicyNet checkpoints and `.npz` row caches are stamped too. The live
  guardians export `PLO5BP_OBS_REV=1`. NOT gated (same distribution, new
  bits): the opp-outcome MC seed (dims 982-989) — see Determinism contracts.
- **Rollout**: in-flight hands are DRAINED at rollout end by default
  (`TrainingConfig.drain_inflight`; `--no-drain-inflight` restores the
  legacy truncation that under-sampled long hands by ~len/W). Drain adds
  ~2.5-8% rows (and VRAM) per update — the live guardians pass
  `--no-drain-inflight`; for a new stem drop it and lower
  `--rollout-length` ~5-8%. Configs where fewer than two seats can act
  after posting are resampled by `_sample_game_config`, and both collectors
  re-deal hands that are terminal at deal (bounded) instead of spinning.
  Advantage normalization under `--mix-configs` is PER CONFIG (each
  sub-rollout is unit-normalized first; the pooled renorm is a no-op).
- **Checkpoints**: writes are atomic (`<name>.pt.tmp` + `os.replace`). Adam
  moments + the L2-init reference live in ONE rolling sidecar
  `<stem>.optim.pt` (never prune it; `--no-optimizer-sidecar` opts out);
  it is restored only when its update counter matches the loaded
  checkpoint — every outcome prints a line, a cold Adam start is never
  silent. Default `--checkpoint` is `checkpoints/train_run.pt`; writing
  `stub.pt` / `nlh_stub.pt` needs `--allow-overwrite-stub`.
- **Live control file**: `runs/anneal_control.json` content is stamped into
  checkpoints (`anneal_control_applied`); on relaunch a pre-existing file
  is RE-APPLIED only if it equals the loaded checkpoint's stamp, otherwise
  ignored with a loud line. Read as `utf-8-sig`; malformed/UTF-16 content
  is logged once, never fatal; invalid values make the edit a logged
  no-op; unknown keys/tiers are logged.
- **PPO guards**: a non-finite KL or loss is a hard trip (never applied);
  `--value-clip <= 0` disables value clipping; the v4/v5 sizing head
  upcasts to fp32 inside `_anchor_dist` (bf16 autocast quantized the
  upper-tail CDF differences — an asymmetric, passive-sizing bias).
  OPEN design question (unchanged): the probability-dependent clip is
  keyed on the gate prob but applied to the JOINT ratio.
- **NLH**: the NLH PPO lineage (`nlh1`-`nlh4`) is RETIRED
  (`scripts/nlh_guardian.sh` exits 1). The NLH path is the native Rust CFR
  solver (`rust_engine/src/cfr/`) → label export → supervised PolicyNet
  (`python/plo5bp/gto/`), served through `PolicyNetHost`; the desktop
  "CFR Solver" app is `python/plo5bp/cfr_app/`. See "NLH GTO teacher".
- **Rollout efficiency pass (2026-09-23)** — exact unless noted:
  - **Compact observation storage** (`python/plo5bp/compact_obs.py`; Rust
    `pack_obs_rows` / `unpack_obs_rows`): the batched rollout keeps each
    observation's exact-0/1 columns (cards, one-hots, seat masks, history
    one-hots — `encoding.FLAG_MASK_MINIMAL` / `FLAG_MASK_FULL`) as bits and
    the rest verbatim f32: 456 B per row on the minimal layout (dense 3,184,
    7.0x), 1,956 B on full (2.4x) — host staging, the end-of-rollout H2D and
    the GPU-resident batch all shrink, which is what lets `--rollout-length`
    grow. `Batch.obs` is then a `PackedObs`; `iter_minibatches` unpacks each
    minibatch on the learner device, bit-exact (pinned: compact vs dense
    rollouts AND PPO updates are identical, `test_compact_obs.py`). Feeding a
    whole `Batch.obs` to a network fails loudly — tests/diagnostics use
    `compact_obs.as_dense`. The packer REJECTS any flag-column value other
    than exactly 0.0/1.0, so an encoder change can't silently corrupt rows.
    NLH stays dense. `--no-compact-obs` = the old dense rows. An engine built
    before this has no packer: the rollout stores dense and says so once.
  - **Batched opponents** (`rollout._StackedOpponents`): all pool snapshots'
    opponent rows go through ONE vmapped forward + ONE sampling pass per step
    (`ActorCriticV2._act_from_heads` = `act` minus `forward`; the learner's
    `act` is bit-identical) instead of one `act()` per snapshot — each call is
    mostly fixed GPU launch overhead with a small net. RNG stream changes
    (same per-row policy; ~1e-6 logit rounding). `--no-batched-opponents`.
  - **Pool-mix draws batched** (`rollout._draw_pool_mix`, was a Python loop
    per finished hand): same distribution, different RNG stream.
  - Trajectory arrays start at 32 slots per seat and double on demand up to
    192 (was a fixed 192: ~735 MB zero-filled per vMin1 sub-rollout).
  - **Rollout buffers are REUSED** across sub-rollouts and updates
    (`rollout._TRAJ_BUFFERS` / `_OBS_POOL_BUFFERS` / `_STAGING_BUFFERS`):
    the trajectory arrays, the per-step obs pool, and multiconfig's shared
    staging (CUDA learners only — on a CPU learner the returned batch IS a
    view of the staging; `_reuse_staging`). Fresh allocations 30x per update
    stalled the first RunPod host (fragmented memory: one update ran 4x its
    neighbors, all of it inside those allocations). Exact — every read is
    confined to what the current collection wrote (pinned by
    `test_buffer_reuse.py`, dirty vs fresh buffers). `_clear_rollout_buffers()`
    for tests that must start fresh.
  - Step timers also cover `step0/setup`, `step0/prestep`,
    `step3a/opp_holes_rot` and the per-step snapshot split into
    `step3c/obs_pack`, `step3c/traj_writes` (+ `step3c/pool_grow` /
    `step3c/traj_grow` when a buffer grows); every region also reports the
    kernel CPU time (`sys_s`) and minor page faults (`faults_k`) spent in it
    (Linux; zeros on Windows) — a kernel stall shows up there.
- **Second efficiency pass (2026-09-23, all BIT-EXACT)** — every change is
  verified by training 3 small updates (CPU, and CUDA on the pod) and hashing
  every tensor of the checkpoints + optimizer sidecar against the previous
  commit: identical. From here on the rule is exactness (owner: "remains bit
  exact") — no more RNG-stream changes. vMin2 went ~500-555 s -> ~342 s per
  update (RTX PRO 6000 pod) before the last batch below.
  - Engine: `payouts_ev` ranks double-board runouts through
    `double_board::RunoutRanker` — hole pairs encoded once per hand, the
    triples of cards already out scored once, and (flop/turn all-ins)
    per-next-card tables for the triples holding one new card, so a sample
    evaluates only triples with 2+ new cards (`hand_eval::plo_best_ck` /
    `one_new_card_table`; same min over the same combos). 2.3-2.6x faster
    all-in payouts. Pinned by `plo_best_ck_matches_evaluate_plo5` and
    `runout_ranker_matches_the_plain_evaluator`.
  - Engine: `payouts_ev_subset` (only the newly-finished rows — in the drain
    phase the whole-batch call re-ran every earlier-finished hand's runouts
    each step; one hand per rayon task), `observation_encoded_minimal_into` /
    `_subset_into` (rows written straight into the env's obs buffer),
    in-place parallel `reset_terminal_batch`, parallel `reset_batch`,
    `apply_hybrid_batch` validation in parallel + mask-free Fold/CheckCall
    legality (`GameState::fold_is_legal` / `check_call_is_legal`, pinned to
    the mask in `play_random_hand`), no per-action heap allocation.
  - Rollout: act-time rows cross PCIe PACKED (`_PinnedStepH2D.upload_rows`:
    Rust-packed from `env._obs` into pinned memory, unpacked on the GPU —
    ~7x fewer bytes, no dense host gather); the learner's packed rows (slot
    0; opponents use slots 1/2) are copied into the trajectory obs pool
    instead of packed twice; trajectory reads/writes use one flat slot index
    (`_flat_traj_view`); the critic's rotated opponent holes come from the
    per-hand `holes_rot_cache`.
  - Guardian: glibc malloc keeps freed memory (`MALLOC_*_` env in
    `vMin2_guardian.sh`). The ~6M page faults per update left in
    `step9d/slab_copies` are the host's AutoNUMA hinting faults on the
    reused 28 GB staging buffer (RSS is flat) — host setting, not fixable
    from the container.
  - Verify an exactness claim the same way: `scripts/train.py --device
    {cpu,cuda} --hidden-dim 32 --num-layers 3 --critic-hidden-dim 32
    --critic-num-blocks 1 --num-envs 480 --rollout-length 24000
    --mix-configs --configs-per-tier 2 --seed 1234 --num-updates 3`, then
    SHA-256 every tensor of every checkpoint (+ `.optim.pt`), old vs new.
    **On CUDA give both runs ONE private `TORCHINDUCTOR_CACHE_DIR`**:
    Inductor autotunes the compiled PPO kernels by timing and caches the
    choice, so a shared cache written by other runs (e.g. under GPU
    contention) changes the numerics — the untouched old code hashed
    differently before and after a same-shaped sweep run compiled. Within
    one cache state the runs are deterministic (pod: `/root/gpu_digest.sh`).
  - Env keeps a PACKED copy of its observations (`BatchedBombPotEnv.
    enable_packed_obs` / `packed_obs`): the in-place encoders pack each row
    right after encoding it (`MINIMAL_INTO_PACKED`), and the rollout's GPU
    uploads gather those 456-byte rows instead of re-reading and packing the
    3.2 KB dense ones (packing ~5k rows cost ~1.8 ms/step on the pod). Only
    passed when the upload's source array IS `env._obs`; any refresh path
    that can't keep the copy in step drops it (then rows are packed on the
    fly). `pack_obs_rows` also packs run-by-run now (same bytes).

### v5 (2026-07-06, IMPLEMENTED, not yet trained — V5_DESIGN.md canonical)

- **Head**: `--sizing-head mixture` (+ `--mixture-k`, default 3) =
  `ActorCriticV5`, tensor `mix_head` (3K rows: K mu_raw, K s_raw, K mix
  logits) — K-component mixture of discretized logistics over the same
  11 anchors. The marginal is a plain Categorical → exact closed-form
  log-prob/entropy; act/evaluate/PPO/UI inherit through `_anchor_dist`
  unchanged. ε weight floor 0.03 is a FIXED constant (not a flag — it
  isn't in the state dict, so serving must match training). NO H(w)
  bonus (v2 flat-collapse analog). v5 also detaches the anchor-prob
  weight in `beta_h_eff` (kills the verified end-anchor entropy
  subsidy); v2/v4 keep legacy behavior for byte-identical resumes.
- **Obs v2**: OBS_DIM 991 → **1020**, pure tail append: per-board
  ahead/tie/behind + win-one/tie-both (8, free counters inside the fused
  Rust `outcome_features_mc` pass), unconditional blockers-to-nuts (8),
  effective-price block (5, capped by the EFFECTIVE stack — dead-chip
  invariance), log1p SPR (8, unclipped — the legacy [0,4] clip saturated
  the whole deep tier at the flop). 991-era checkpoints serve via the
  `downgrade_obs_to_v2` slice. (HISTORICAL: at v5 time the Rust obs
  encoder was force-disabled; it was ported and re-enabled at width 1171
  in 2026-07 — see "Current state".)
- **Warm-start v4→v5**: `scripts/convert_v4_to_v5.py <v4.pt> <out.pt>` —
  component 0 = the v4 head (w₀≈0.92), zero-pads obs columns on actor AND
  critic, adds the zero-init critic `adv_head`, strips anneal/counter/
  pool metadata; function-preservation verified in-script (f32 kernel
  noise allowance). Convert pool siblings with the same script if
  prior-pool seeding is wanted. The head/variant guards stay strict —
  the converter is the only v4→v5 bridge.
- **Q-aux**: mixture runs build the critic WITH the zero-init dueling
  `adv_head` (q_actions = 2 + anchors) so the later VRPO advantage flip
  is not a checkpoint break; `--q-aux-coef` (default 0) trains it as an
  auxiliary regression (log column `q=`).
- **KL-anchor magnet is usable now**: the v4/v5 crash in
  `_kl_to_reference` is fixed (head-agnostic `_anchor_dist`, manual KL
  over probs, p_raise detached), and the EMA reference persists as
  `ckpt["model_ema"]` (restored on warm-start).
- **EMA serving** (`PLO5BP_SERVE_EMA=1`): the UI/study path serves the
  `model_ema` actor (smoother, less exploitable — what a study tool
  wants) instead of the last iterate. Env-gated, reversible, falls back
  to the last iterate when the key is absent/None (magnet-off runs). The
  critic is never EMA'd.
- **`--ev-runout-samples`** (default 64, unchanged): MC runout samples
  for the terminal-reward EV. Measured 64→256 ≈ +23% of ROLLOUT
  wall-clock on CPU (a few % of a full pod update); left at 64 by
  default since the extra variance reduction is marginal — opt in per run.
- **Per-tier control under --mix-configs**: sub-rollouts attach per-row
  entropy coefs (`Batch.ent_coef_rows`) so `{"tier_ent": {...}}` control
  edits genuinely apply per tier; `{"entropy_coef": X}` broadcasts to all
  tiers in mix mode (was a silent no-op); per-tier F/T/R prints as
  `[ftr-tier]` per log line; checkpoints stamp `mix_configs`/`mix_tiers`/
  `configs_per_tier`.
- `--value-clip` exposed (default 0.2 RAW bb — see V5_DESIGN.md B4; A/B
  on a throwaway stem before changing production). `OpponentPool` serial
  sampling is seeded (`seed=` arg; train.py passes the run seed).
- v5 entropy seeds are UNTUNED — the stem re-seeds high (~0.45), never
  at an annealed floor (anneal is one-way down). UI serves v5
  automatically (sniffs `mix_head.weight`); recommendations gain a
  `mixture` payload block (per-component mu/s/w).

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
  regularizer; the reference IS persisted (`ckpt["model_ema"]`, restored
  on warm-start — see the v5 notes).
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
`watchdog_auto.sh`), `vTwo<N>`/`vFour<N>`/`vFive<N>` (PLO5 v2/v4/v5 —
guardian scripts per stem), the current `vSix<N>` (`--v6`,
`vSix4_guardian.sh`), `vMin1` (minimal obs, `vMin1_guardian.sh`), `vMin2`
(fresh minimal-obs stem on rev-2 values, compact storage, 44M rows,
`vMin2_guardian.sh` — started 2026-09-23 as the size sweep's 128x3 baseline),
and
`nlh<N>` (NLH v4 PPO — RETIRED 2026-07-16, `nlh_guardian.sh` now exits 1;
do NOT prune `checkpoints/vFour4_*.pt` on the pod). Guardians resume from
the newest `<stem>_*.pt`; the `<stem>.optim.pt` sidecar and `.pt.tmp`
files never match that glob. A guardian refuses to start while its stop
flag exists and re-checks it right before every relaunch. vSix4's guardian
pgreps any `scripts/train.py`; vMin1's matches only its own
`--checkpoint` — still run ONE family per pod unless you know the GPU
fits both; stop via the stem's `runs/*.stop` file.

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

### NLH GTO teacher (`rust_engine/src/cfr/`, `python/plo5bp/gto/`, `cfr_app/`)

Native Rust CFR (DCFR for HU postflop, external-sampling MCCFR for
preflop/multiway) → label export → supervised PolicyNet → `PolicyNetHost`
(env `PLO5BP_GTO_CHECKPOINT`). Invariants after the 2026-09-20 review:

- **Solver**: ES-MCCFR accumulates the average strategy at the OPPONENT's
  sampled nodes (own-reach weighting); deals are sampled from the true
  joint (rejection on card collision) and the BR weights hero combos by
  their true marginal; the FINAL exploitability estimator runs on every
  exit path and notes carry an honest `expl_kind=` (`exact_infoset`,
  `hero_enum`, `sampled_runout_br`, `mc_poll`, `mc_br_proxy`);
  `Range::parse` errors on unknown tokens (`#44` = combo id, `44` = pocket
  fours); ALLIN raises to `max_raise_chips()` and a jam-sized `RAISE_x` is
  merged into it; invalid roots (sub-blind stacks, 0-chip pots,
  `max_iterations=0` with no stop condition, over-budget trees) are
  `ValueError`s, never panics. Strategies solved BEFORE the review for
  preflop/multiway (or with ranges) should be re-solved.
- **Export/labels**: CFR `ALLIN` (and any raise that clamps to the stack)
  maps to the grid-LEGAL jam anchor via `labels.jam_anchor_index`, never
  blindly to the top atom; facing a jam, `ALLIN` with `max_raise == 0` is a
  CALL; teacher mass on an illegal action raises
  `IllegalTeacherMassError`. Only reports with a final estimator kind and
  no `early_stop=`/promoted marker are exploitability-VERIFIED; everything
  else lands in `unverified/` and is skipped by teacher export
  (`--allow-unverified-expl` to override). Pre-review label shards and
  PolicyNet checkpoints are invalid — re-export and retrain.
- **Serve**: `PolicyNetHost` re-encodes postflop nodes to the canonical
  obs the labels were built from (current-street history only,
  `total_commit := street_commit`, blind seats None), serves GRID chips
  (no refine head) and snaps near-stack sizes to exactly `max_raise`;
  `supports(seats=, street=)` answers from recorded coverage and the UI
  falls back to PPO (`gto_unsupported`) outside it. The probe scores gates
  AND sizing, fails NaN, uses a stratified SHA-256 holdout and refuses
  train/holdout overlap; the GTO badge requires recorded provenance.
- **Desktop app** (`cfr_app/`): solves run in a spawned child process
  (`CFR_APP_INPROCESS=1` to disable), directories resolve through
  `cfr_app/paths.py` (`CFR_APP_DATA_DIR`), the local API is
  loopback-Host/Origin checked (`CFR_APP_ALLOWED_HOSTS`), range text goes
  through one strict parser (`/api/range/parse`), node views are keyed by
  (seat, path, runout) and weighted by `visit_mass`.

```bash
# NLH training (name the size explicitly, same as PLO — see Training)
.venv/Scripts/python scripts/train.py --variant nlh_single \
  --sizing-head logistic --hidden-dim 2048 --num-layers 4 \
  --stack-dist deep --num-seats-range "2,3,4,5,6"
```

### NLH range grid ("Ranges" tab, LOCAL BUILD ONLY)

GTO-Wizard-style 169-hand strategy grid (built 2026-07-06). The tab
appears next to Study/Trainer only when format=NLH AND not
`PLO5BP_PUBLIC`; the public build strips every `/ranges` route (same
filter as `/ocr`) — do NOT un-strip without an explicit user decision
(and admin-gate it first when that day comes).

- Rust: `PyGameState.pack_range_nlh(holes)` packs ONE decision node once
  per candidate hole in the exact batched-packer layout (the obs is
  villain-blind; only the hole multi-hot + `hero_cat_a` +
  `nlh_opp_outcome` vary per row — the latter two via
  `nlh_category_for` / `nlh_opp_outcome_for` free fns, rayon).
  **Bit-exact vs serial study obs**, pinned by
  `tests/python/test_ranges.py` — keep that parity.
- Backend: `plo5bp/ui/ranges.py`, stateless `POST /ranges/query`
  (seats, stack_bb, `line` = ONE ordered list interleaving actions and
  street cards, optional node = prefix length). Ephemeral study replay
  (never the user's study Session), one batched forward per node
  (in-process cache per line-prefix), reach = per-seat gate-level Π of
  the seat's own action probs (no size-conditional reach in v1), fixed
  5/10($5) stake with button=seat 0 so positions are canonical. A study
  street boundary sets the env's done flag — check awaiting FIRST.
- Frontend: `static/ranges.js`, self-contained (app.js untouched — the
  tab toggles `body.ranges-mode`); strips builder with GTO-Wizard
  rebranching (acting at a viewed past node truncates the line there),
  anchor-size raise buttons, street card popup, hover combo breakdown.
  Client queries NEVER drop-when-busy: latest-response-wins via RG_SEQ
  (dropping desyncs the line from the render — bug found in validation).
  A failed query rolls back the line AND the viewed node.
- Sizes: the wire format (`chips_bb`) is the engine's raise-BY delta; the
  payload also carries raise-TO totals (`to_bb`, `actor_commit_bb`,
  `min/max_raise_to_bb`) and the client shows/enters raise-TO. ANY legal
  anchor whose chips equal `max_raise` is all-in (the ALL-IN atom is
  deduped away whenever a fraction anchor already reaches the stack) — it
  is summed into `allin_p` and labelled ALL-IN.

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
  2048×4 net (matching actions skip MC entirely). v2+ scoring inverts the
  refinement `u` over the UNCLAMPED anchor bracket (`anchor_lo_raw/
  hi_raw`) — the policy maps `u` over the unclamped bracket and clamps
  chips afterwards, so inverting over the clamped grid graded the exact
  recommendation as an "inaccuracy" whenever min/max-raise clipped the
  bracket; chips equal to `rec_chips` always score "best" with no MC.
- The EV-loss estimate replays the REAL deal and future board, so it is
  hidden while the hand is live (`feedback.ev_loss_bb: null`,
  `ev_loss_hidden: true`) and revealed at terminal / in review; stats are
  committed once per COMPLETED hand (abandoned hands add nothing) and
  accumulate the SIGNED estimate, clamping only the displayed aggregate.
  `_rollout_ev` holds `_TORCH_RNG_LOCK` only around seed→sample regions
  (env work happens outside it); the public build caps `mc_rollouts` at
  32. `POST /trainer/act` with no live hand is a 409 (never auto-deals).
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
  pool-snapshot index (one stacked call for all of them since 2026-09-23).
  Bit-exact parity vs serial is *not* asserted (RNG-consumption order
  differs); parity is at the env level (see
  `tests/python/test_env_batched.py`) and the training smoke. Its RNG stream
  changed on 2026-09-23 (batched pool-mix draws + batched opponents): the
  same seed does not reproduce pre-2026-09-23 batches — distributions are
  unchanged. Stored observations may be compact (`compact_obs.PackedObs`).

## Determinism contracts (don't break these)

- EV runout seed: hand base seed XOR `0x9E3779B97F4A7C15`.
- Canonical orderings for multi-sets (hole cards, boards) are fixed in
  the encoder — see memory note on engine design preferences.
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
  by `tests/python/test_reset_with_deck.py` and a Rust test. Deal order is a
  public contract the home games' verifiable shuffle depends on: `hole_count`
  cards per seat INDEX (dealt in or not), seat 0 first, then full board A,
  then full board B. Do not reorder it.

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
   CHECK (an unreadable stack holds the timer lock one more tick).
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

## Public build (`PLO5BP_PUBLIC`)

Set `PLO5BP_PUBLIC=1` to serve the trainer + study tabs WITHOUT any live
capture — the public / monetizable variant (model served server-side, no
real-time table reading, so nothing that enables RTA). Single codebase,
one flag; unset (the default) is the full local build with OCR + PokerNow
intact. Run it with:

```bash
PLO5BP_PUBLIC=1 .venv/Scripts/python -m uvicorn plo5bp.ui.server:app --port 8765
```

- Server (`server.py`): the flag strips every `/ocr`, `/pokernow` and
  `/ranges` route after registration (`_public_route_kept` — handles
  FastAPI's `_IncludedRouter` wrapper too; the handlers stay defined, they're
  just unmounted, so those paths 404). The `/` route injects
  `window.PLO5BP_PUBLIC` into the served HTML so the client knows its mode
  before first paint (no `/config` fetch race, no flash of the live
  controls). Static files are policed on the RESOLVED file inside
  `NoCacheStaticFiles` (so `/static//app.js`, trailing `/`, `..`, case
  variants can't dodge it): `app.js`/`style.css` are served WGLIVE-stripped;
  `index.html`, `ranges.js`, `admin.html` and the `games.*` assets 404 from
  the mount (each has its own gated route). The public app is built with
  `docs_url=None, redoc_url=None, openapi_url=None`.
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
`tests/python/test_public_service.py` (sets env + reimports ui modules —
ALWAYS through `conftest.purge_ui_modules` / the `ui_purge` fixture: popping
`sys.modules` alone leaves the stale module bound as an attribute of the
`plo5bp.ui` package and `from plo5bp.ui import public` silently reuses it).

**Free while the models are in development (2026-09-22):** `public.FREE_FOR_ALL`
(`PLO5BP_FREE_FOR_ALL`, default ON) makes every SIGNED-IN user entitled — no
daily trainer quota, Study unlocked, `/billing/checkout` answers 409, `/me`
carries `free_for_all`, the account chip says FREE ACCESS and the landing page
says so. Sign-in stays. The paywall / quota / Stripe code is untouched and still
tested: `tests/python/conftest.py` defaults the test session to
`PLO5BP_FREE_FOR_ALL=0`; `test_public_free_mode.py` boots production's way. Set
the env var to `0` in `/etc/wrapgto/env` to bring the paywall back.

**Workspace UI (2026-09-22):** Study / Trainer were re-skinned to sit next to the
home games but stay a TOOL (flat panels, dense type, one accent). The card FACES
are deliberately the old ones (owner's call) — do not swap in the home-games
cards. Layout: slim `#top-bar` (mode tabs, format, units, account) + per-mode
`#workbar` above the table (seats / $ per bb / ante / live controls / New Hand,
trainer Settings / Repeat / New Hand) + `#side-rail` (Recommendation on top, the
13 x 4 card matrix or the trainer stats under it). The redesign is the
"WORKSPACE THEME v2" layer APPENDED to `style.css` — it wins by cascade order, so
add new workspace styles after it. Card entry is continuous
(`placeStudyCard` / `nextEmptySlot` in `app.js`): a placed card selects the next
empty slot across groups (hole -> flop A -> flop B -> turn -> river), a card
clicked with nothing selected fills the first empty slot, and cards can be typed
(rank then suit, Backspace undoes).

Service-layer rules (review 2026-09-20 — keep them):

- The access middleware authorizes on a NORMALIZED `scope["path"]`
  (`//`, trailing `/`, `.`/`..`, backslashes resolved). `/format` is a
  subscriber route. Implicit deals are metered: for a non-entitled user
  whose `TrainerSession` has `hand_no > 0` and `hand is None`, a request to
  `/trainer/state|settings|act|stats/reset` counts against the quota like
  `POST /trainer/new_hand` (the first implicit hand of a fresh session
  stays free).
- Stripe: re-validation never runs on the event loop (threadpool, 8 s
  timeout); `resource_missing`/"No such subscription" ⇒ INACTIVE; other
  errors fail open only until `current_period_end` + 3 days, re-checked at
  most every 15 min (`PLO5BP_STRIPE_TIMEOUT`, `PLO5BP_STRIPE_GRACE_DAYS`,
  `PLO5BP_STRIPE_RETRY_S`). Activation (confirm AND webhook) requires
  `mode == "subscription"`, a subscription id, `payment_status` paid or
  `no_payment_required`, and a subscription that is active/trialing NOW —
  replaying an old session cannot re-activate.
- Dev login exists only when `PLO5BP_DEV_LOGIN=1` AND the `BASE_URL` host is
  loopback; it rejects any forwarding header (XFF, CF-Connecting-IP,
  Forwarded, …) and needs a loopback client + Host. The test client is
  accepted only with `PLO5BP_DEV_LOGIN_TESTCLIENT=1` (fixtures set it).
  `email_verified` defaults to False when the claim is absent.
- `/me.homegame` is `{"href", "label"}` for granted users (absent
  otherwise) so `app.js` ships no home-games strings to everyone else.

**Home games** (`ui/homegame.py`, `static/games.*`, admin-granted
`homegame_access`, 404 for everyone else): PokerNow-style private PLO5
double-board tables over `BombPotEnv` (minimal obs, actual payouts, HTTP
polling, per-table RLock + clock watchdog). Invariants: hole cards are
revealed only when ≥ 2 hands are live at terminal (an uncontested winner
stays face-down); "own" cards are shown against the user id DEALT into the
seat this hand (a seated-but-not-dealt viewer or a seat taken over later
sees nothing); all-in equities are computed once per (hand, street) from
the ALIVE seats (`runout.board_equities` = one exact enumeration per
board) — never per viewer under the lock; money is exact integers and the
ledger always sums to zero (largest-remainder apportionment at cash-out);
mutations persist before (or roll back with) in-memory state; `/act` and
`/deal` carry `hand_no`/`action_seq` and 409 when stale; `/deal` refuses
while a runout is still revealing and the payload releases deltas/awards
progressively (no unrevealed card anywhere); chat/rabbit need table
membership; new tables default to a 30 s clock and the watchdog auto-acts
Away actors even on clock-less tables; a mid-hand leave folds/checks now
and cashes out at hand end.

Premium tables pass (2026-09-21 — `tests/python/test_homegame_premium.py`):

- **The SERVER deals** (`_auto_deal_tick_locked`, table setting
  `deal_delay_secs`; 0 = manual). An API create that omits it gets MANUAL
  (scripted callers/tests stay deterministic); the create dialog sends 5 and
  pre-existing rows migrate to 5 s. It deals only to a table somebody is AT
  (`PRESENCE_WINDOW_S`: ≥ 2 eligible players polled recently) — an abandoned
  table must not keep posting antes. The client never auto-deals.
- **Time bank**: after the base clock the actor burns `Seat.time_bank_left`
  (in-memory; only the seconds used are charged, `TIME_BANK_REFILL_S` comes
  back per hand, capped by `time_bank_secs`). Two timeouts in a row sit the
  player out.
- **Table settings** (`POST /settings`, host): name, ante, buy-in min/default/
  max, seats (between hands; shrink needs the high seats empty), clock, bank,
  deal delay, runout pause, `listed` (link-only tables), `allow_rabbit`.
  Blinds are fixed at creation (they are the chip unit — changing them would
  re-value every stack). The hand in progress keeps the config it was dealt
  with. Also `/transfer_host`, `/show` (table your own cards after the hand —
  the ONLY way a fold-out winner or a folded hand is ever revealed), `/react`
  (whitelisted emotes), `/hands` + `/hands/{n}`.
- **Hand history** stores every dealt hand's cards (`homegame_hands`) and
  filters per viewer with the live rule (own cards + hands tabled at showdown
  or shown); members only; nothing about a hand — history, stats, the `win`
  event line — is served while its runout is still revealing.
- **Sitting / reloading mid-hand** is allowed for a seat that is NOT in the
  hand (`_dealt_in`); `_sync_idle_stack` keeps `hand_start_stacks` in step so
  the hand's end and the runout-time ledger stay zero-sum. Players holding
  cards still wait (table stakes).
- `events` / `reactions` are small in-memory feeds (dealer lines, toasts,
  emotes). Lobby rows carry seated NAMES only (no emails / user ids).
- **Client** = five gated modules (`homegame.GAMES_ASSETS`, each also "deny" in
  `server._PUBLIC_STATIC_POLICY`): `games.js` (core: state, ordering guards,
  polling, pre-actions, routing — no DOM building, driven headless by
  `test_review_homegame_client_js.py`), `games.table.js` (persistent-node
  felt renderer: every animation is a DIFF of prev→next state, so never
  rebuild the table with innerHTML), `games.play.js` (dock: actions, sizing,
  pre-actions, status strip, hotkeys — built once, updated in place),
  `games.ui.js` (lobby, rail, Manage drawer, dialogs, toasts, player card),
  `games.sound.js` (WebAudio synth, no audio files). On-felt sizes are
  multiples of `--u` (set by `layout()`); the felt insets in `computeGeom`
  and the seat geometry must stay in step. Player notes/tags and preferences
  are localStorage-only.
- **Chips in (2026-09-22 — `tests/python/test_homegame_chips.py`)**:
  `approve_buyins` turns a sit / top-up by anyone but the host or a TRUSTED
  player (`homegame_players.trusted`, `/trust`) into a pending request
  (`LiveTable.requests`, in memory; `/request` approve|deny|cancel; a sit
  request holds its seat; it lapses when the requester's browser is gone;
  switching approval off or trusting the player lets it through). Two
  DIFFERENT automatic-chip features, each off | host | player:
  **auto top-up** (`topup_mode`, `/auto_topup`, per-seat target + below —
  tops UP only, and only once the stack is under the threshold: no rathole)
  and **set stack** (`auto_stack_mode`, `/auto_stack`, the older feature —
  resets the stack to the target before EVERY deal, up or down: ratholing is
  the point). Set-stack wins when a seat has both; players choose through
  `/auto_chips_self {kind: off|topup|set}` and only touch the knobs the host
  left to them; targets are capped by `max_buyin_cents`; while approval is on
  neither runs for an untrusted player (`_auto_chips_allowed`). A top-up sent
  with `queue: true` while holding cards is queued (`queued_topup_cents`) and
  lands when the hand is over; without the flag it is still a 400.
- **Live push**: `GET …/stream` (SSE, async generator — never a threadpool
  thread per viewer; the view is built under the lock via
  `run_in_threadpool`) pushes the viewer's state whenever `_stream_sig`
  changes (rev + the clock-driven runout/award steps) and at least every
  `STREAM_HEARTBEAT_S`; `max_events` exists for tests. The client
  (`startLive`) falls back to the 450 ms poll after repeated stream errors.
  Presence (`seen`, fed by polls AND stream pushes) drives server dealing,
  the `spectators` name list and each seat's `present` flag.
- Phone landscape = the `wide` geometry in `games.table.js` (boards side by
  side, hero plate beside the hero cards, no seats along the bottom edge) plus
  the `(max-height: 480px) and (orientation: landscape)` block in `games.css`
  that floats the dock over the felt's bottom corners — keep the two in step.
- **Tracking (2026-09-23 — `tests/python/test_homegame_tracking.py`)**: every
  hand record (`homegame_hands.summary`) is REPLAYABLE (`actions` carry the
  engine action id + chips + `auto` = the clock decided; seats carry
  `start_chips`; `ante_chips`, `flows`, `grades`). The client replayer is a
  click-through (`openHand` in `games.ui.js`: position k = k actions played;
  `replayState` rebuilds stacks / bets / pot / boards) and `openInStudy` copies
  any position into Study by driving Study's own API (`/reset`, `/config`,
  `/seats` with `stacks_are_starting`, `/cards`, `/action` x k; hero = the
  actor when their cards are visible to the viewer; Study caps at 6 players).
  **Who paid whom** = `runout.money_flows`: per pot LAYER, each contributor's
  chips go to that layer's winners in proportion to what each took (self-flows
  dropped) — fold-outs, scoops, chops, quartering, side pots and dead money all
  follow; checked against the engine's payouts in a randomized sweep. Stored net
  per hand in `homegame_flows` (user ids, chips). **AI grading**: a background
  thread (`_grader_loop`, `PLO5BP_HOMEGAME_GRADING`, OFF in the test session)
  replays each finished hand in a FULL-observation env from an in-memory job
  (deal seed + exact engine inputs — the seed is NEVER persisted or served) and
  scores every PLAYER decision with the Trainer's `compute_node_distribution` +
  `score_move(_v2)` from the actor's own seat; clock/away/host actions are not
  graded. Grades land in the record and as `acc_sum`/`acc_n` in
  `homegame_hand_results` (session + lifetime accuracy are SQL sums). Table
  setting `show_grades` (default on) = marks on everyone's actions; off = own
  actions only (a mark on a mucked hand leaks a little about it). Lifetime:
  `GET /games/api/my/hands` (sort time|pot|net|accuracy, filter by table,
  paged) and `/games/api/my/stats` (net, accuracy, sessions, head-to-head in
  cents across tables); neither serves a hand its table is still revealing.
  `openInStudy` opens Study in a NEW TAB: the tab is opened synchronously
  inside the click (after the awaits a browser blocks it as a pop-up) and
  pointed at `/?mode=study` once the spot is loaded; blocked pop-ups fall back
  to a modal with a plain `target=_blank` link — the replayer never navigates.
- **The club (2026-09-24)**: home games are a private circle, so stats are OPEN
  inside it — `GET /games/api/community` (every player's hands / net / accuracy,
  the pairwise `pairs` = "`to` is up `cents` on `from`", all sessions),
  `/games/api/players/{id}/stats|hands` (= the `my/*` pair for any player;
  `_my_hands(viewer, …, player_id=)`). What stays PRIVATE is unchanged: hole
  cards follow the table's reveal rule for the VIEWER (own + tabled/shown) even
  when browsing someone else's history, and no email is ever served. The lobby's
  Players section (`renderClub` in `games.ui.js`) = podium of the top three by
  accuracy (needs `MIN_RANKED` = 20 graded decisions, else "provisional"), a
  card per player, the head-to-head matrix (`openMatrix`) and All sessions.
  **Excluded sessions**: `homegames.excluded` is a SOFT, reversible flag set by
  the SITE ADMIN only (`POST …/exclude {on}`, not the host — a host must not be
  able to erase a losing night); an open table is closed first (cash-out), a
  busy hand is a 400. Every stats query joins `homegames` and filters
  `excluded=0` (`_my_hands`, `_my_stats`, `_community`, lobby sessions) — a new
  aggregate MUST do the same. Nothing is deleted; the table still opens by link.
- **Bet spots (`placeBetSpots` in `games.table.js`)**: a bet sits on the rail's
  inward NORMAL at its seat (`railNormal`: flat edges straight across, round
  ends toward their own circle's centre), just clear of the seat's own box —
  NOT "toward the table centre" (that put a corner seat's chips, and the
  hero's, in front of the neighbour on a wide table). Blocked spots swing round
  the seat 8° at a time (toward the centre first) and take the first place free
  of the pot row, boards, hero cards, other seats and already-placed bets; dead
  ahead wins whenever it physically fits. The hero's bet goes out from the hero's
  CARDS in every layout. `layout()` slides the pot+boards block (bounded) so one
  bet's height of felt stays clear under the far seat and over the hero's cards;
  desktop compacts the block (`.board` 5.3u, `#hero-hole` 6.3u = `g.heroCw` —
  keep JS and CSS in step). The street TOTAL and the street tag live on the
  pot's ROW (`#pot-row` grid flanks), not on a line of their own: that line
  appeared with the first bet and shoved both boards down. Sizes that have font
  floors are modelled (`seatBottom`) or measured (`pillSize`) per layout — a
  pure `u` constant is wrong on a small phone. The measuring harness used to
  tune this is `.claude/tools/games_preview/measure_bets.js`.
- **Verified shuffle (2026-09-25 — `ui/fairdeal.py` is the SPEC, `static/games.fair.js`
  the player's half; `tests/python/test_homegame_fair*.py`)**: the operator also
  PLAYS in these games, so the threat model is "server + one player together".
  Per hand: the server SEALS a shuffled deck (52 salted per-position SHA-256
  commitments -> `seal`) before anyone contributes; each seated browser commits
  to a 32-byte random number bound to `hand_id|seal|seat`; at the deal the list
  is LOCKED (dealt-in, PRESENT devices only) and published; a browser reveals
  ONLY after it has seen its own commitment in that list under the seal it
  committed to; `cut` = hash of the reveals drives a Fisher-Yates permutation
  (SHA-256 counter stream, rejection sampling) and the engine deals
  `F[slot] = D[perm[slot]]`. Every card a viewer is shown carries
  `{slot, pos, salt}` in `fair.hand.open` — built FROM the viewer's visible
  cards, so it can never open a card the reveal rule hides; mucked hands stay
  sealed. Guarantee: if YOUR device contributed, the deal was uniform and
  unaltered whatever everybody else did. It does NOT stop the operator from
  looking at cards server-side (only mental poker can), and deck COMPOSITION is
  proven only for opened cards — say so, never oversell it. Invariants:
  the slot map is public and mask-independent (seat s hole k = `5s+k` for EVERY
  seat index, board A `5n..`, board B `5n+5..`; pinned in Rust + Python tests);
  a committed device that does not reveal within `FAIR_REVEAL_S` VOIDS the
  attempt — new deck, new seal, the absentee barred for that hand, ALWAYS
  announced (`fair` event + `voids` in the transcript + `void_counts`): a
  withheld number is a visible re-roll, never a silent one (two in a row =
  `FAIR_PENALTY_HANDS` out of the shuffle). A sealed deck whose cut may be known
  is never dealt later (pause / close / resize void it). Tables nobody's browser
  takes part in (scripts, tests, bots) deal AT ONCE, exactly as before: the wait
  exists only for users in `fair_capable`. Transcripts persist compactly in
  `homegame_fair` (key + deck; commitments are recomputed) — the key and deck
  are secrets at rest, like the hole cards already in `homegame_hands`. The
  grader replays `hand_deck` (memory only), not a seed. Kill switch
  `PLO5BP_HOMEGAME_FAIR=0`; an engine built before `reset_with_deck` turns
  `FAIR_ON` off by itself (old dealing, client shows "unverified") — so a
  production ship MUST rebuild the engine (`scripts/deploy_prod.sh`). The spec
  is pinned by a known-answer permutation and a Node run of the browser verifier
  against Python transcripts; changing either side is a PUBLIC spec change.
- **Table UX round 2 (2026-09-22 — `tests/python/test_homegame_table_ux.py`)**:
  the create dialog takes a BIG BLIND and an ANTE IN BB (no stake presets, no
  small blind — `sb_cents` is kept in the wire format/DB for the old rows and
  defaults to half a bb; displays say "$1.00 bb · ante $3.00"). The host TAPS a
  reserved seat (or a seated player's "$" badge / the Chips tab's Edit) and gets
  the request dialog: approve as asked, approve a DIFFERENT amount
  (`/request {amount_cents}` — validated like a buy-in, announced "approved for
  $80 (asked $150)"), approve + trust, decline; the host's view carries
  `seat.request`. `allow_rathole` (host switch, default off) lets a player TAKE
  CHIPS OFF the table: `/remove_chips {amount_cents, queue}` — whole cents to
  leftover + a `cashout` ledger row (the move set-stack makes), never below an
  ante + 1 bb (that is leaving), queued while holding cards
  (`queued_remove_cents`, re-checked when it lands). LEAVING mid-hand plays the
  hand out — `leave_after_hand`, `seat.leaving`, `/stay` to cancel — and cashes
  out when it ends (`_apply_leaves_locked` runs with the deferred work); the old
  fold-now behaviour is `/leave {now: true}` (kicks/Remove use it); a leaver
  whose browser is gone counts as away so the clock never waits on them. The
  rabbit button lives ON THE FELT in the 2x2 gap of the undealt cards
  (`placeRabbit`, "Click to reveal"), the host has Start/Pause in the top bar
  (`#tb-run`). Award animation (ClubGG-style): `_capture_rabbit` names the pot
  layers deepest-first ("Side pot N" … "Main pot", `t.pots`, served as `pots`
  while the runout blocks) and tags each award step with its `pot`
  (`_assign_award_pots`); the client shows the pots as inline pills that REPLACE
  the pot pill (same height — the boards must not move at showdown), highlights
  the active pot, counts each one down as its halves are paid, flies the chips
  from THAT pot, prefixes the caption with the pot name, and the layout reserves
  room under the boards for the caption (`captionHeight`) so it never lands on
  the hero's cards (it wraps on portrait phones).
- Preview harness (gitignored): `.claude/tools/games_preview/` — launch
  entry `games_preview` (public build + dev login + temp DB on :8772) and
  `bot.py` (scripted guests).

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

The original investigation (Tasks A/B/C) is in the git history of
`HANDOFF.md`; plan files live in `.claude/plans/`. Task C's "hero hole
hid" branch was DELETED in the 2026-09-20 review (ClubGG never re-hides
hero's cards mid-hand, so it only ever fired on glitches). Still open —
treat all of these as hypotheses, not facts: (1) `simple_ocr_mode`
defaults to True and then the reconstructor is never stepped — confirm it
was off when validating; (2) the OCR loop sleeps `poll_ms` AFTER each
tick, so the real period exceeds 200 ms; (3) the timer-transition CHECK
branch needs a unique positive read on a villain timer bar whose ROI is
only 1-2 px tall. Second open OCR item: `OcrRunner` can anchor a hand
before ClubGG deducts antes, leaving seeded stacks one ante high. The
full review (findings, fixes, deferrals, repro scripts) lives in
`.claude/reviews/`.

## Layout

```
rust_engine/src/              engine + PyO3 bindings
  engine.rs, state.rs, double_board.rs, hand_eval.rs, cards.rs
  bindings.rs                 PyGameState (serial), PyBatchedEngine (batched),
                              fused obs encoder + feature pyfunctions
  obs_v7_inc.rs               v7 obs tail (include!d by bindings.rs)
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
    server.py                 FastAPI app, Session, OcrRunner, PokerNowRunner
    trainer.py, ranges.py     trainer mode, NLH range grid (local only)
    public.py, homegame.py    public service layer, private home games
    runout.py, hand_describe.py, common.py
    static/                   index.html, app.js, ranges.js, admin.html,
                              games.html/.css + games{,.table,.play,.ui,.sound}.js

tests/
  python/                     engine / encoder / env / rollout / ui / gto
                              (test_review_*.py = 2026-09-20 review regressions)
  ocr/                        extract / events / server_mirror / rois / cards
    fixtures/                 labeled frames + JSON state

scripts/
  train.py, profile_rollout.py, evaluate.py, exploitability.py,
  probe_suite.py, smoke_test.py, *_guardian.sh (pod), cfr_*/gto_*/step*.py

tools/pokernow/               Tampermonkey userscript + README
.claude/plans/, .claude/reviews/   plan files; code-review findings + repros
checkpoints/                  trained weights; stub.pt is UI default;
                              <stem>.optim.pt = rolling optimizer sidecar
screenrecords/frames/         debug captures from /ocr/save_frame
HANDOFF.md                    latest session handoff
```
