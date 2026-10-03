# Training reference (PPO, rollout, env, encoding, runs)

Read this BEFORE changing anything under `python/plo5bp/` that trains or encodes (ppo, rollout, config, network, selfplay, compact_obs, env, env_batched, encoding*, eval, exploit, sizing, `train/`, `evaluation/`) or any training script / guardian. Moved out of CLAUDE.md on 2026-09-28 (REPO-002 / ML-036); the dated results live in `docs/training-log.md`.

## Training

**Always name the network size explicitly.** Since 2026-09-28 train.py
REQUIRES `--hidden-dim`, `--num-layers`, `--critic-hidden-dim` and
`--critic-num-blocks` (it used to default silently to 2048×4 / 1536×2,
while TrainingConfig says 128×2); an unflagged run once trained a default for
days and was mistaken for a 2048×4 result — the lesson is "never train a size
by accident", not "only 2048×4". Sizes by lineage: the full-obs `vSix` stems
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

### Current state (updated 2026-09-28 — read this before the older v5 notes)

- **HEADLINE 2026-09-26 evening: `vSix6` is the main run** (see the "Regression
  diagnosis + redesign" bullets below): the vSix5 line had converged (not
  regressed); vSix6 = actor 1024x3 + rebuilt 1536x2 SiLU critic, entropy 0.06,
  ~164M rows/update at 1.76M envs, half-precision storage, faster engine.
  Guardian `scripts/vSix6_guardian.sh`, stop `runs/vSix6.stop`, its own control
  file `runs/vSix6.control.json`. **The live site serves vSix6_1300 since
  2026-09-26 23:42 UTC** (its 1024x3 actor + the vSix5_1248 critic, because the
  deployed site code cannot rebuild the new SiLU critic, which only feeds the
  review's true EV; backup `stub.pt.bak-pre-vSix6_1300`). vs vSix5_1248: sampled
  +0.34 (z 13), argmax-vs-argmax +0.08 (z 4), top action vs its sampled play
  +0.895 (vSix5_1248's own +0.979 — the one metric it does not win).
  **2026-09-28: vSix6 had stalled at ~u1340; after 5 recipe rounds it RESUMED on
  entropy 0.045 / lambda 0.8 / lr 7.5e-5 / 30 setups per tier** (see "Plateau
  check + recipe rounds 3-5"). Promotion candidate `avg_1380_1389` (+0.14 sampled,
  +0.16 argmax over the live 1300) awaits the owner's OK. The pod's /workspace
  is a ~20 GB quota — a full volume silently kills trainers.
- **Obs**: PLO OBS_DIM is **1171** (v7-obs tail 1020..1171: STK/BRD/DUAL
  blocks, docs/design/V7_DESIGN.md / V7_OBS_CANDIDATES.md), minimal layout 796
  (`--obs-mode minimal`, vMin1), NLH 995. The fused **Rust obs encoder**
  (`BatchedEngine.encode`) is the batched env's DEFAULT since 2026-09-28
  (`PLO5_RUST_ENCODER=0` = the numpy batch encoders, the reference; NLH always
  numpy; a Python OBS_DIM the Rust port lacks is an error, not a fallback).
  The compiled engine must be current: `plo5bp.engine_abi` refuses a stale
  build at import (no more hasattr fallbacks). The SERIAL env (Study /
  Trainer / eval) encodes PLO4/5/6 with the same engine encoder too, one row
  at a time (`_engine.encode_game_state`, ML-008: fed the raw dict's
  opp-outcome block, so the MC still runs once per decision and
  `info.raw_obs` is unchanged; encodes at the revision Python reads at call
  time, like the numpy encoders did; bit-exact with the scalar numpy
  encoders -- now oracles -- pinned by `test_serial_encode.py`, ~3-4x
  faster); NLH and PLO67 stay numpy. The main run is `vSix6`
  (`scripts/vSix6_guardian.sh`, see the headline); earlier stems: `vSix4` /
  `vSix5` (full obs, `--v6` preset) and the paused minimal-obs `vMin1`-`vMin3`.
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
- **Table tiers** (`plo5bp/train/tiers.py`; the evaluation tools draw from the same
  functions): production mixes `clubgg`, `clubgg_deep` and `deep` (1/3 each, seats
  uniform 2-6 — `--seats-dist` is not passed). **`clubgg_real` (2026-10-03, default
  off)** = the owner's own ClubGG tables measured from their hand histories (953 hands,
  $10/$20, 3bb ante): per-seat stack bands with the 20bb buy-in spike AND the real seat
  counts (6/5/4/3/2-handed 41.5/31.2/20.9/5.2/1.2%, carried by the tier). The real
  tables differ from the production mix: median stack 43bb (mix 71), median flop SPR
  2.5 (5.3), 58% of hands mix a <=25bb and a >=100bb seat (10%), 2-3 handed 6% (40%).
  Use it with `--mix-tiers clubgg_real` (training) / `h2h_x.py --tiers clubgg_real`
  (scoring, side worktree). Pinned by `test_tiers_clubgg_real.py` (the old tiers'
  draws are digest-identical). Round R (2026-10-03, on the pod from vSix7_1767) tests
  it; its result goes in training-log.md. The site's
  Trainer deals it too ("My tables" for a player without a profile of their own:
  `python/plo5bp/ui/CLAUDE.md`), so a change to the tier changes those deals.
- **Checkpoints**: writes are atomic (`<name>.pt.tmp` + `os.replace`). Adam
  moments + the L2-init reference live in ONE rolling sidecar
  `<stem>.optim.pt` (never prune it; `--no-optimizer-sidecar` opts out);
  it is restored only when its update counter matches the loaded
  checkpoint — every outcome prints a line, a cold Adam start is never
  silent. Default `--checkpoint` is `checkpoints/train_run.pt`; writing
  `stub.pt` / `nlh_stub.pt` needs `--allow-overwrite-stub`.
- **Live control file**: per run since 2026-09-28 —
  `<run-dir>/<stem>.control.json` (e.g. `runs/vSix6.control.json`; env
  `PLO5BP_ANNEAL_CONTROL` still overrides; it used to default to the pod-wide
  `runs/anneal_control.json`). Its content is stamped into
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
  keyed on the gate prob but applied to the JOINT ratio. Since 2026-09-28 a
  non-finite GRADIENT norm is a hard trip too, the critic-only passes refuse
  non-finite steps (`crit=... SKIPPED=n` on the `[health]` line), and no
  checkpoint or sidecar with a NaN/Inf tensor is ever written
  (`train/checkpoint.assert_finite_for_save`).
- **train.py is a thin wrapper (2026-09-28)** around `python/plo5bp/train/`
  (cli, tiers, control, checkpoint, diagnostics, metrics, loop — moved
  verbatim, digest-identical). Per-run files are named after the
  `--checkpoint` stem under `--run-dir` (default `runs/`):
  `<stem>.metrics.jsonl` (one JSON object per update: every PPOStats field,
  the critic's value health, timings, rows, lr, per-tier aggr%, pool, VRAM —
  read it with `train.metrics.read_metrics` or `scripts/plot_metrics.py`),
  `<stem>.heartbeat` (rewritten after every update), `<stem>.launches.jsonl`
  (provenance of every launch: argv, git commit/dirty, engine hash, versions,
  env), `<stem>.threads.txt` (live thread count; was the shared
  `runs/threads.txt`), `<stem>.resources.jsonl` (`--resource-sampler`, no
  longer started by PLO5BP_STEP_TIMERS). Checkpoints are `schema` 2: new keys
  `arch` (every constructor argument; `network.build_actor_from_checkpoint`
  / `build_critic_from_checkpoint`) and `provenance`; nothing was renamed or
  removed, so old files load/resume/serve unchanged
  (`test_checkpoint_compat.py`). The log line's `H=` is now the POLICY'S
  entropy (`Hbonus=` = the old bonus-weighted number, shown when
  sizing_entropy_scale != 1), `bonus%` is printed `aggr%`, and a `[health]`
  line follows: critic EV/bias (overall, per street), clip fraction, k3 KL,
  `kl0` (the pre-step rollout-vs-PPO mismatch), gradient norms (actor /
  critic / display head) and clipped-step shares. `--max-consecutive-
  rollbacks` (default 20) exits a rollback livelock with code 3. Silently
  ignored flag combinations are refused (`cli.validate_flag_combinations`);
  `--spec FILE` (TOML/JSON of flag values; the command line wins) holds a
  stem's recipe. RETIRED 2026-09-28 (ML-030; the flags are gone, old
  checkpoints still load/resume): block rotation + the F/T/R auto-anneal
  (`--block-rotation/--block-size/--anneal-*`, the control file's `step`), the
  aggression / retroactive bonuses (`--aggression-bonus-c`,
  `--retroactive-bonus-c`, `agro_deep`) and the NLH-PPO tiers (`nlh_topoff`,
  `nlh_ring`); F/T/R stays as telemetry (`aggr%`). New default-off
  numerics flags (enable at a relaunch, with the owner): `--critic-q-norm-
  minibatch`, `--compile-critic-train`; `--weight-decay` (default 0.01 =
  what every stem used, now explicit). For a NEW stem (not bit-exact with
  the current numbering / step counts): `--minibatches-from-rows` (an epoch is
  exactly `--num-minibatches` steps of the COLLECTED rows; vSix6 takes ~17,
  not 16, from the drain overshoot) and `--number-by-count` (checkpoint files,
  update_counter, pool tags, heartbeat and metrics all count updates done: the
  first update after a relaunch is saved and numbers never fall behind; stamped
  `numbering`, a relaunch follows its file, a stem cannot switch midway), and
  `--crn-streams` for paired comparisons (recipe / tuning waves): the update's
  configs and PPO shuffles and, per (env, hand), the deals / buttons / opponent
  assignments come from streams keyed by (seed, update[, sub-rollout]), so
  candidates play the same hands whatever their hand lengths
  (`rollout._CrnDeals`; the shared stream pairs only the first update), and
  `--pool-anchors 25,50,100,200` (the numbered checkpoints nearest those ages
  back join the FIFO opponent pool as anchors against self-play drift;
  `selfplay.refresh_pool_anchors`; checkpoints record only the FIFO).
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
    all-in payouts. Pinned by `plo_best_ck_matches_evaluate_plo` and
    `runout_ranker_matches_the_plain_evaluator`.
  - Engine: `payouts_ev_subset` (only the newly-finished rows — in the drain
    phase the whole-batch call re-ran every earlier-finished hand's runouts
    each step; one hand per rayon task), `observation_encoded_minimal_into` /
    `_subset_into` (rows written straight into the env's obs buffer),
    in-place parallel `reset_terminal_batch`, parallel `reset_batch`,
    `apply_hybrid_batch` validation in parallel + mask-free Fold/CheckCall/
    AllIn legality (`GameState::fold_is_legal` / `check_call_is_legal` /
    `all_in_is_legal`, pinned to the mask in `play_random_hand`), no
    per-action heap allocation. Re-deals reuse the finished hand's buffers
    (`GameState::redeal`, PERF-034; pinned equal to `new_hand`).
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
  - Verify an exactness claim the same way: **`scripts/exactness_check.py`**
    (library `plo5bp/exactness.py`) trains tiny recipes with a git revision's
    code (default HEAD, exported read-only with `git archive`) and with the
    working tree, then compares the SHA-256 of every tensor of every
    checkpoint + `.optim.pt` (`--recipe tiny|v6|minimal|smoke|all`, `--same` =
    determinism, `--device cuda` on the pod; both sides share this tree's
    engine binary). `tests/python/training/test_exactness.py` runs the `smoke` recipe
    (the vSix6 flag set at toy size) on every pytest session: HEAD vs the
    working tree when training sources changed, else the tree twice. The
    `tiny` recipe is the original manual one: `scripts/train.py --device
    {cpu,cuda} --hidden-dim 32 --num-layers 3 --critic-hidden-dim 32
    --critic-num-blocks 1 --num-envs 480 --rollout-length 24000
    --mix-configs --configs-per-tier 2 --seed 1234 --num-updates 3`.
    **On CUDA give both runs ONE private `TORCHINDUCTOR_CACHE_DIR`** (the
    script does, and runs the two sides one after the other):
    Inductor autotunes the compiled PPO kernels by timing and caches the
    choice, so a shared cache written by other runs (e.g. under GPU
    contention) changes the numerics — the untouched old code hashed
    differently before and after a same-shaped sweep run compiled. Within
    one cache state the runs are deterministic (pod: `/root/gpu_digest.sh`).
  - Env keeps a PACKED copy of its observations (`BatchedBombPotEnv.
    enable_packed_obs` / `packed_obs`): the in-place encoders pack each row
    right after encoding it (`encode(..., out_bits=, out_real=)`), and the rollout's GPU
    uploads gather those 456-byte rows instead of re-reading and packing the
    3.2 KB dense ones (packing ~5k rows cost ~1.8 ms/step on the pod). Only
    passed when the upload's source array IS `env._obs`; any refresh path
    that can't keep the copy in step drops it (then rows are packed on the
    fly). `pack_obs_rows` also packs run-by-run now (same bytes).
  - The per-finished-hand flush (winning-aggression qualification + the
    retired retroactive bonus, passed as 0,
    GAE and VRPO backward scans, gathers into the output slabs) is ONE Rust
    pass (`rust_engine/src/flush.rs::flush_trajectories`) reproducing
    numpy's exact f32 operation order (no FMA); ~4 ms -> ~1 ms per step.
    Dense storage (NLH, `--no-compact-obs`) goes through the same kernel
    (each float32 row passed as raw bytes). The numpy originals of all three
    per-step kernels live in `plo5bp/rollout_reference.py` (same signatures;
    `PLO5BP_NUMPY_FLUSH=1` makes the collector call them); `test_rust_flush.py`
    pins every Batch field bitwise for GAE, GAE + bonus and VRPO, compact and
    dense. (2026-09-28: `collect_rollout_batched` is a `_BatchedCollector`
    class, one method per step phase; the act prefetch
    `PLO5BP_ROLLOUT_OVERLAP` is deleted.) Only the logged bonus TOTAL is
    summed in another order (a diagnostic; exactly 0 with the bonus off).
    When a branch of the flush is added, remember `traj_lengths[term_envs]
    = 0` — skipping it silently grows every trajectory across hands.
- **Resumed runs draw their own random stream (2026-09-23)**: numpy, torch
  and the pool are seeded from `SeedSequence((seed, resume update))` on
  `--load-checkpoint`; before, every guardian relaunch re-seeded with the
  bare `--seed` and replayed the fresh run's first updates — the same table
  configs AND card deals. Fresh runs unchanged (`test_resume_seed.py`).
- **`--micro-batch-rows N`** (default 0 = off, the exact old path): each PPO
  minibatch is gathered/forwarded/backpropagated in chunks of <= N rows with
  accumulated gradients (per-row means weighted by chunk share; the fold-
  supervision term keeps its minibatch-wide denominator; L2-to-init and the
  KL anchor sum once). Same step up to float order; lets the rollout grow
  past one minibatch's GPU working set (~12 KB/row) — then the stored batch
  (~590 B/row compact) bounds it. `test_ppo_microbatch.py`.
- **Network-size sweep tooling (2026-09-23)**: `scripts/sweep_guardian.sh
  STEM HD NL CHD CNB TARGET` (vMin2 recipe, only the size changed, stops at
  TARGET updates; `NUMA_NODE=n` pins a socket), `train.py --gpu-lock FILE`
  (several runs share one GPU: only one holds a batch + PPO working set at a
  time; rollouts overlap), `scripts/h2h_eval.py A.pt B.pt` (duplicate deals,
  seats swapped, EV payouts, train-tier configs; edge in bb/seat-hand +- se),
  `scripts/utilization_probe.py --states selfplay` (dead units / effective
  rank over the checkpoint's own self-play states — the original fixed-flop
  probe calls units "dead" that are merely idle in that one spot: 76/384 vs
  4/384 for the same 128-wide actor), `scripts/sweep_eval.py` (runs both
  every N updates at equal update counts), `scripts/sweep_report.py` (one
  table of every comparison + probe so far).
- **Third efficiency pass (2026-09-24, BIT-EXACT, CPU + CUDA digests)**:
  - **num_envs is the big lever.** The production shape (220k envs / 30
    mixed configs = 7.3k envs per sub-rollout, ~8,600 steps per update) is
    bound by FIXED per-step overhead (Python, kernel launches, and ~0.3 ms
    per thread-pool wake-up on the pod). Uncontended pod, 128x3/128x2, 44M-row
    target: 220k envs 227 s/update (201k rows/s), 440k 175 s (272k), 880k
    145 s (353k), 1.76M 154 s (379k; 58M rows — the drain overshoot grows
    ~8 rows per env). `scripts/bench_subrollout.py` (per-step cost by region
    vs env count) and `scripts/throughput_probe.py` (full updates, peak GPU
    memory + peak host RSS) measure it.
  - Per-step host row copies run through ONE engine call
    (`gather_rows_multi`: packed obs + gate-mask + sizing rows into the pinned
    upload slot; the uploaded rows into the trajectory pool), on the calling
    thread unless the copy is >= 4 MB — the first cut woke the thread pool per
    array and made the production shape ~2 ms/step SLOWER. Rule for any new
    engine call on the per-step path: a pool wake-up costs ~0.2-0.4 ms on the
    pod, so small work stays on the calling thread and calls are fused.
    `record_learner_steps` goes parallel from 8,192 rows.
  - Sub-rollout setup: `reset_batch(snapshot=False)` (the discarded snapshot
    copied the whole dense obs buffer — 186 MB per sub at 58k envs, 9 s per
    update at 1.76M envs), `reconfigure(clear_obs=False)`, reused
    hole-rotation buffer.
  - **Packed-only observations** on the CUDA upload path: the in-place
    minimal encoders take `out=None` (each row encoded into a per-task scratch
    row and only PACKED), and `enable_packed_obs(layout, dense=False)` keeps
    only the packed copy — `env._obs` then holds NaN as a tripwire, snapshots
    unpack, `_drop_packed_obs` refuses. The collector uses it whenever every
    upload gathers packed rows (the CUDA upload path). At 58k
    envs per sub the dense (N, 796) write was ~10 ms of an 11 ms refresh.
  - `aggression_record_batch` (engine) = the (retired, 0) aggression bonus AND its
    trajectory record (cost / pre-step pot / street at the acting seat's slot,
    slot count + 1) in one call; its numpy reference is in
    `rollout_reference.py` (`PLO5BP_NUMPY_FLUSH=1`), pinned equal in
    `test_aggression_record.py`.
  - **`--batch-on-host`** (TrainingConfig.batch_on_host, multiconfig only):
    the rollout batch stays in host RAM; each PPO minibatch / micro-batch
    chunk is gathered on the CPU into pinned staging and copied over
    (`rollout.HostBatchLoader`), compact obs unpacked on the GPU. The CUDA
    digest equals the device-resident batch's (also with
    `--micro-batch-rows`), so the rollout is bounded by host RAM (~573 B/row
    stored) instead of GPU memory. `test_host_batch.py`. Each chunk's ~17
    fields are gathered in ONE parallel pass, and the fold denominator's
    gate-mask rows through `HostBatchLoader.gate_mask_rows` (field-by-field
    single-threaded gathers made host PPO ~3x slower).
### v5 (2026-07-06, IMPLEMENTED, not yet trained — docs/design/V5_DESIGN.md canonical)

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
  earned. Err high: a 0.02 cold start collapsed gate entropy within 10
  updates (2026-06-10).
- `--target-kl` (default 0.5) is the KL guard: aborts the PPO inner
  loop before the optimizer step when a minibatch's |approx_kl|
  exceeds it (logged as `KLSTOP@mbN`). vTwo2 collapsed at update 173
  (approx_kl ≈ +2417 → entropy pinned at 0) without it; the v2
  discrete anchor head has heavier-tailed importance ratios than v1's
  continuous Beta, which is why v1 never needed this. 0 disables.
- Live-tune WITHOUT pausing training via the run's control file
  (`<run-dir>/<stem>.control.json`, or `PLO5BP_ANNEAL_CONTROL`):
  `{"tier_ent": {"deep": 0.08}}` sets a tier's coef, `{"entropy_coef": X}`
  every tier's, plus lr / target_kl / kl_hard / clip rooms /
  sizing_entropy_scale / q_fold_sup_coef. Applied whenever file content
  changes. (The v2-era auto-anneal and its `step` key are retired.)
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
  `seed_pool_from_checkpoints`, `tests/python/training/test_warmstart_pool.py`.
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
`vSix4_guardian.sh` ... `vSix6_guardian.sh` = the 2026-09-26 redesign, the
main run; the round stems `r1*`/`r2*` = its recipe search), `vMin1` (minimal obs, `vMin1_guardian.sh`), `vMin2`
(fresh minimal-obs stem on rev-2 values, compact storage, 44M rows,
`vMin2_guardian.sh` — started 2026-09-23 as the size sweep's 128x3 baseline),
and
`nlh<N>` (NLH v4 PPO — RETIRED 2026-07-16, `nlh_guardian.sh` now exits 1;
do NOT prune `checkpoints/vFour4_*.pt` on the pod). Since 2026-09-28 every
retired launcher/guardian (the v1 root `launch_*`/`watchdog_*`, vThree..vFive1,
vSix4/5, vMin1/2, nlh, the curriculum orchestrators, `pod_prune_disk.sh`) lives in
`scripts/archive/training/` (read, don't run); the live guardians (vSix6, vMin3)
share `scripts/guardian_lib.sh` + `plo5bp/train/guardian.py`: they resume from
the HIGHEST numbered compatible `<stem>_<N>.pt` (or the rolling `<stem>.pt` when
its counter is newer) — no longer the newest by mtime — kill a trainer whose
`runs/<stem>.heartbeat` is older than `STALE_SECS` (3 h), and give each stem its
own `TORCHINDUCTOR_CACHE_DIR` (`$HOME/.cache/plo5bp-inductor/<stem>`). The
`<stem>.optim.pt` sidecar and `.pt.tmp` files never match `<stem>_<N>.pt`.
The guardians, `guardian_lib.sh` and `python/plo5bp/train/` SHIP TOGETHER (one
pull; the lib refuses to run without its helper). A trainer that writes no
heartbeat of its own (an 80048a0-era train.py) is never killed for it: the
check is OFF for that PID (logged once) and the PID watch rules.
`scripts/README.md` lists every script. A guardian refuses to start while its stop
flag exists and re-checks it right before every relaunch. vSix4's guardian
pgreps any `scripts/train.py`; vMin1's matches only its own
`--checkpoint` — still run ONE family per pod unless you know the GPU
fits both; stop via the stem's `runs/*.stop` file.

## Rollout paths

Two drivers in `python/plo5bp/rollout.py`:

- `collect_rollout` — serial, one env at a time. Used by UI, exploit
  probe, eval. Must stay bit-exact; don't refactor for speed.
- `collect_rollout_batched` — Phase A-D: `BatchedBombPotEnv` wraps
  `PyBatchedEngine`, encoder is vectorized, opponents are grouped by
  pool-snapshot index (one stacked call for all of them since 2026-09-23).
  Bit-exact parity vs serial is *not* asserted (RNG-consumption order
  differs); parity is at the env level (see
  `tests/python/engine/test_env_batched.py`) and the training smoke. Its RNG stream
  changed on 2026-09-23 (batched pool-mix draws + batched opponents): the
  same seed does not reproduce pre-2026-09-23 batches — distributions are
  unchanged. Stored observations may be compact (`compact_obs.PackedObs`).
