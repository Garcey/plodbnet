# Training log — dated results, runs and diagnoses

History, newest last. The rules and the reference these runs produced are in `docs/training.md`; this file is the evidence behind them. (Moved out of CLAUDE.md on 2026-09-28, REPO-002 / ML-036.)

## From the "Current state" notes

- **Size sweep + rollout/num_envs results (2026-09-24, RunPod)** — the
  `sw*` stems, `scripts/sweep_report.py`:
  - Every actor from 32 to 128 wide (x3) and every critic from 32 to 128
    (x2) matched a 128x3/128x2 control over updates 30-39 (h2h means within
    +-0.06 bb/seat-hand; two 128 runs differ by 0.03-0.07 — run-to-run
    noise). Smaller nets LAG for ~20-30 updates, then catch up; 16x3/16x2
    was -0.86 at update 9. The critic's own loss rises monotonically as it
    shrinks (HL-Gauss CE over u30-39: 128 3.27-3.30, 64 3.36-3.40, 32
    3.42-3.45, 16 far worse), the actor shows no such penalty. Chosen:
    **actor 32x3, critic 128x2**. A 40-update sweep cannot see capacity
    limits that only bind late in training (entropy still 0.25 here).
  - Network size barely moves speed or memory: the rollout is CPU-bound and
    GPU memory is the stored batch — shrinking the net does NOT buy a longer
    rollout any more.
  - **num_envs**: 2.64M = most rows/s at a 44M-row target (457k rows/s, 142 s
    per update for 65M rows; 1.3M-3.5M all within ~5%; 220k = 201k rows/s).
    The drain adds ~8 rows per env to every update.
  - **Maximum rollout** (host RAM, 233.8 GiB container): RSS ~= 6.7 GiB +
    570 B/row with `--batch-on-host --micro-batch-rows 1000000` (GPU flat at
    12.8 GiB). Confirmed: a 349M-row target collected 362M rows in one update
    at 212 GiB peak (rollout 621k rows/s). Long rollouts collect faster
    (fixed per-step costs amortize).
  - At long rollouts PIN the trainer's CPUs to one NUMA node (taskset):
    unpinned, automatic NUMA balancing tripled the rollout's kernel time
    (2.64M envs / 340M rows: 464k rows/s vs 621k pinned at 1.76M / 362M).
    Host-batch PPO: 161 s for 340M rows after the fused gather (433 s before).
  - `scripts/vMin3_guardian.sh` = the full run on these settings (32x3 /
    128x2, 1.76M envs pinned to the GPU's node, 330M-row default via
    `ROLLOUT_LENGTH`, host batch, micro-batching, entropy 0.07 from the tuning
    below, sizing-entropy scale 0.1; warm start `WARM=checkpoints/t3ent07.pt`);
    it refuses to start while any other trainer runs (their RAM would push the
    container over its limit) — also stop leftover evaluators first (each holds
    a few GiB). It relaunches a dead trainer unless 4 restarts fall within 6 h
    (a crash loop), so rare one-off crashes over weeks never end the run.
    `scripts/run_watch.py --stem vMin3` runs beside it: every 5 updates one line
    in `runs/vMin3_watch.log` = argmax vs argmax against the start
    (`t3ent07.pt`) and vMin2 u150, the collapse/sizing probe, and the
    trainer's latest v/H/KL.
- **Hyperparameter tuning (2026-09-24, RunPod) — the vMin3 settings** (`t1*`-
  `t4*` stems; the KL-anchor magnet was deliberately NOT tuned — it stays off
  until late-stage training):
  - Method: `scripts/tune_run.sh STEM NODE UPDATES [flags]` = the vMin3 recipe
    at a tuning scale (1.76M envs, 44M-row target ~ 58M rows/update, host
    batch) warm-started from `swA32_40` WITH its Adam state, opponent pool and
    resume seed, ONE knob overridden, 20 updates, one NUMA node each, four at a
    time. Runs are compared at EQUAL update counts.
  - Measures: stochastic h2h (how the policy plays); **argmax vs argmax**
    (`h2h_eval.py --greedy-a --greedy-b`, `sweep_eval.py` passes both): what a
    run has LEARNED to prefer, apart from how much it still randomizes — half
    the noise of a stochastic h2h, and it keeps separating runs where a
    candidate's argmax vs a SAMPLING reference saturates (+2.27 from u49 on);
    `scripts/policy_sharpness.py` (one fixed set of self-play states): gate
    entropy and the share of decisions whose least-likely legal action is
    below 1% / 0.1% (the collapse indicator). A single checkpoint's h2h swings
    +-0.1-0.3, so read runs of 3+ checkpoints, never one.
  - **lr stays 1.5e-4**: 5e-4 and 1.5e-3 swing up to +-0.3 bb/seat-hand
    between 5-update checkpoints (self-play cycling) and are worse on average
    (argmax vs argmax means -0.01 / -0.24); 5e-3 blew up the critic (v 3.2 ->
    18). `--critic-lr` (the critic in its own AdamW group) gained nothing, so
    one group.
  - **Entropy 0.25 -> 0.07.** At 0.25 the policy is very random (raises 44% of
    the time where legal, near-uniform raise sizes; against the same sampling
    opponent, playing its most likely action instead of sampling is worth ~2
    bb/seat-hand). Argmax vs the 0.25 run at u49/u54/u59: 0.40 -0.25/-0.21/
    -0.16, 0.15 +0.15/+0.34/+0.22, 0.10 +0.18/+0.43/+0.33, 0.07 +0.29/+0.45/
    +0.44 — every tier, the deep tier most; the same order against vMin2 u150's
    argmax (a different, longer-trained 128x3 lineage: 0.40 -0.15, 0.25 +0.05,
    0.15 +0.27, 0.10 +0.42, 0.07 +0.50). Continued to u79 (40 updates), the
    order held with steady leads (means u64-u79 vs 0.25: 0.15 +0.24, 0.10
    +0.29, 0.07 +0.38; 0.10 vs 0.15 +0.09, 0.07 vs 0.10 +0.10). Sampled play vs
    0.25: 0.15 +0.6-0.7, 0.10 +1.0-1.1 bb/seat-hand. Collapse indicator
    (least-likely legal gate action < 0.1%): 0.25 ~3% of decisions, 0.15 9%,
    0.10 13%, 0.07 17% — each level settles within ~15-20 updates of the switch
    and then holds flat through u79 (no collapse, 0.07 included). The 2026-05
    reason for a high exploration entropy was a 16k-row rollout; at 58M-345M
    rows per update even a 0.1% action is sampled many times per update. **The
    owner chose 0.07 for vMin3 (2026-09-24)** — an explicit exception to the
    usual 0.1 floor, backed by these runs. Watch a long run with
    `policy_sharpness.py` on its checkpoints (same cached states).
  - No gain (stay as they were): `--ppo-epochs 4` (vs 2 at entropy 0.15),
    `--gae-lambda` 0.9 and 1.0 (flag added; 1.0's sampled play was worse).
  - **Sizing-entropy scale 1.0 -> 0.1** (`t5sz*`, at entropy 0.07). With the
    full sizing bonus raise sizes NEVER differentiated: size entropy rose from
    ~1.8 at init to ~1.98 (of a 2.40 max) by u20 and stayed there — vMin2
    through u150 included; the average spread was min-raise ~17%, every other
    size 6-11%, pot ~6%. The size-specific advantage is a few % of the pot
    against a whole-hand advantage scale (and the pooled raise Q column gives
    sizes signal only through the lambda-TD terms), so the bonus wins. Scale
    0.5 / 0.2 / 0.1 / 0: size entropy at u59 1.95 / 1.89 / 1.82 / 1.68 (still
    falling), every tier moving toward SMALL bets (at 0: min-raise ~29%, pot
    ~4.5%); argmax vs vMin2 u150's argmax: 1.0 +0.43, 0.1 +0.57, 0 +0.55
    bb/seat-hand (deep tier +1.11 -> +1.42). At entropy 0.25 a 0.5 scale had
    shown nothing (the sizing bonus was still 0.125). vMin3 uses 0.1 —
    live-tunable (`{"sizing_entropy_scale": X}`); watch the size spread.
  - Not tested, kept (reasoned): gamma 1.0 (chips, no discount), v6 clip room
    (at KL ~0.0005/update the ratio band rarely binds; `clip` 0.2 is unused
    under v6), target_kl 0.5 / kl_hard 10 (guards that never trip at this lr),
    adv_clip 8, q_aux 0.5 / fold-sup 15 / value 0.5 (critic-internal balance;
    critic stable at v ~3.2), HL-Gauss 51 bins / 1500 / 0.75, AGC 0.1, l2-init
    1e-4, AdamW b2 0.999 / wd 0.01, pool 8 x every 5 updates at mix 0.5,
    ev_runout_samples 64, bonuses 0, 16 minibatches.
- **Deep dive 2026-09-26: the minimal-obs line is FAR behind the live model ->
  vMin3 paused, vSix5 (the live lineage on the new pipeline) is the main run.**
  - `scripts/h2h_cross.py A.pt B.pt` = h2h_eval for two checkpoints that read
    DIFFERENT observations (layout and/or obs rev): each model is served the full
    obs encoded at ITS rev, then `obs_adapter` — exactly how the site serves it
    (numpy encoders read `encoding.OBS_SEMANTICS_REV` at call time; the env's own
    pack is captured and re-encoded per model). `--selfcheck` proves the re-encode
    equals the Rust encoder of an engine BUILT at rev 1 and at rev 2 (max diff 0).
    Sanity: a model vs itself ~0 (vMin3 -0.01 +- 0.04, vSix4 -0.03 +- 0.03);
    vMin3 u239 vs u200 +0.12 +- 0.05 (h2h_eval said +0.07 +- 0.03).
  - vs the live `vSix4_1240` (full obs rev 1, 2048x4 / 1536x2, 1240 updates of
    9M rows ~ 11B rows): vMin3 u239 (32x3 / 128x2 minimal, ~55B rows) -2.06 +-
    0.04 bb/seat-hand sampled, -2.42 +- 0.03 argmax-vs-argmax; u200 -2.25 /
    -2.67; vMin2 u150 (128x3 minimal) -4.35. Every tier, deep worst. The minimal
    obs drops the engineered hand-strength features (made-hand categories,
    opp-outcome MC equities, blockers, draws) — 5x more data did not make up for
    them. The size sweep (all minimal) could never see this.
  - Inside vMin3: head to head it improved strongly to ~u200 (u200 vs u150 +0.61
    argmax) and was flat after (u239 vs u200 -0.02 argmax, +0.07 sampled) while
    its edge over OLDER references fell (vs t3ent07 +1.50 at u200 -> +0.93 at
    u235) = self-play drift; raise sizing collapsed toward min-raise (20% ->
    54%, sizeH 1.91 -> 1.23 under sizing scale 0.1); KL/update 10x (gate); actor
    rank99 18 -> 28 of 32 (u40 -> u239: near saturation). Paused cleanly at u242
    (`runs/vMin3.stop`; resumable with its guardian).
  - `scripts/vSix5_guardian.sh`: vSix4's exact recipe (v6, 2048x4 / 1536x2,
    obs rev 1, entropy 0.16 = vSix4's annealed level, sizing scale 1.0, lr
    1.5e-4) on the new pipeline: 880k envs, 70M-row target (~77M with the drain;
    vSix4 used 9M), host batch + micro 200k, pinned CPUs, drain on, checkpoint
    every update; warm from `vSix4_1240` (pool seeded 940/1215/1240; cold Adam).
    Measured (before the in-place encoder): 68.7k rows/s (vMin3 ~620k — the full
    obs costs 384-sample opp-outcome MC per row), 61 GiB RSS at 27.5M rows, GPU
    13.6 GiB. A vSix5 checkpoint is a drop-in promotion for the live site
    (same obs rev 1, same shapes).
  - `observation_encoded_into` (engine) = the full layout's in-place encoder,
    like the minimal ones: no fresh (N, 1171) array + copy per step (~137 MB of
    page faults per step at 29k envs); bit-identical (`test_full_encoder_into.py`).
    On the pod: rollout 864 s -> 541-688 s per ~77M rows (~106k rows/s).
  - vSix5 vs the live vSix4_1240 (h2h_cross, fixed seed 2026; sampled / argmax):
    at entropy 0.16 u1242 -0.01 / +0.08, u1244 -0.05 / +0.07, u1245 +0.00 /
    +0.09; entropy -> 0.10 (anneal_control, after u1246's update) u1246 +0.38 /
    +0.07, u1248 +0.50 / +0.13 (z 16 / 7), u1249 +0.56 / +0.20 (z 18 / 11).
    Later checkpoints are also scored against vSix5_1248 (the live model since
    the promotion) — runs/vsix5_eval.log on the desktop. vSix4's own 1240 vs 1215 was +0.27 /
    -0.07: its late "gains" were entropy sharpening, not learning.
    **vSix5_1248 PROMOTED to the live site 2026-09-26 08:04 UTC** (stub.pt;
    backup `stub.pt.bak-pre-vSix5_1248`; no OBS_REV change — rev 1). Grades on
    the site now come from a sharper policy (entropy 0.10 vs 0.16).
  - The live model's own utilization (self-play probe): actor 2048x4 rank99
    248-598 of 2048, 372/8192 dead (16% of the input layer); critic 1536x2
    64% dead, rank99 23-69 — the full-obs nets have lots of slack: a full-obs
    size sweep (e.g. distilled from vSix5 for a warm start) is the next size study.
- **Regression diagnosis + redesign (2026-09-26 evening, owner /goal "it stopped
  improving and regresses — redesign it").** vSix5 stopped at u1290 (18:06 UTC).
  - NOT a regression: a 9-checkpoint round robin (`scripts/h2h_league.py`:
    shared deals, least-squares ratings, residuals = non-transitivity; the
    standing version is `scripts/panel_eval.py`: a FIXED panel of references
    + baseline policies, `scripts/panels/*.json`, round robin cached, one
    rating per checkpoint on the panel's scale -- ML-039) is
    transitive (residual RMS ~ pair se); late checkpoints sit +-0.1 around 1248's
    level (1288 rated = 1248). The u1248-1258 "peak" was a lucky high right after
    the entropy drop. The real problem is a STALL: argmax ratings of u940..u1288
    all within +-0.1 bb/seat-hand — every sampled gain since u940 was entropy.
  - Updates are noise around a converged fixed point: weights random-walk
    (displacement ~ sqrt(k) steps, consecutive deltas cos -0.35); two one-update
    runs from u1290 differing only in --seed (`scripts/update_snr.py`) agree on
    their gate log-prob changes at corr +0.13 (params cos 0.53 actor / 0.87
    critic). Weight averaging (`scripts/average_checkpoints.py`) is neutral. A
    single update moves argmax-vs-argmax h2h by up to +-0.18, so judge by several
    checkpoints, never one.
  - The critic was broken: `PLO5BP_DUMP_BATCH=<file>` (train.py: dump a rollout
    sample, obs as <file>.obs16.npy, then exit) + `--gae-lambda 1.0` +
    `scripts/critic_calibration.py`: V = symexp(E[symlog]) reads 43% LOW (2-2.5x
    low in its upper deciles, +7.9 bb on the river) — a log-space average; the
    raw-space mean of the same distribution is within ~1 bb. EV only 0.29. The
    utilization probe: critic input layer 77% dead. `scripts/critic_offline.py`
    (held-out CE on the dump): a FRESH SiLU critic trained ~1,500 small steps
    beats the 15B-row critic (2.60 vs 2.755). Online, the fresh critic first
    FAILED (r1crit): 16 huge minibatches per epoch = too few steps, and the Q
    loss (raw bb^2, ~1e3) drowned the value cross-entropy (~3) in the shared
    torso (q ran away to 1e4) -> `--critic-q-norm` and `--critic-minibatches`.
    Then a second trap: with a zero-init adv_head the fold column starts at
    Q_fold = V (~+8 bb) and the 15x fold supervision drags the shared torso to
    fix it — the value CE went 2.57 -> 3.64 within 100 steps (offline repro;
    the online warm-up showed 3.65). A new critic needs `--q-fold-zero` (Q_fold
    pinned to its exact truth 0: folding ends the seat's future rewards) — with
    it the CE holds at 2.55.
  - Actor capacity is NOT binding (`scripts/distill_size.py`: students distilled
    from u1290 on 2M states — held-out gate KL 256x3 .0062, 512x3 .0043, 1024x3
    .0034, same-size 2048x4 .0031; h2h vs the teacher -0.05..-0.09 for all, the
    same-size copy -0.05 = the method's floor).
  - New flags (all default OFF = the old behavior): `--critic-act {relu,silu,
    gelu}`, `--critic-in-norm`, `--critic-v-raw` (V = raw-space mean),
    `--critic-q-norm` (Q losses / (return var + 1)), `--critic-extra-epochs N`,
    `--critic-minibatches N` (critic-only passes, many small steps),
    `--critic-fresh` / `--critic-init PATH` (a new critic with the ACTOR's Adam
    moments + l2-init refs still restored), `--actor-freeze-updates N` (critic-only
    warm-up), `--obs-real-f16` (compact rows' real columns stored float16: 1,022
    vs 1,956 B per full-layout row; the Rust flush converts == numpy's cast;
    pinned by `test_obs_real_f16.py`; NOT bit-exact), `PLO5BP_ANNEAL_CONTROL`
    (a per-run live-control file — runs sharing the pod read each other's
    `runs/anneal_control.json` otherwise). The critic's choices ride in an
    `_arch` buffer so `build_critic_from_state_dict` rebuilds it (the old site
    code fails to load such a critic -> it only disables the review's true EV).
    `scripts/convert_offline_critic.py` turns an offline candidate into a
    `--critic-init` file; `scripts/recipe_run.sh STEM NODE UPDATES [flags]` = the
    vSix5 recipe at a search scale (220k envs, 15M rows) for parallel candidates.
  - Engine (bit-exact, pinned): the actor's hand categories in
    `pack_full_with_cats` run in parallel (was a serial loop, up to ~100 ms/step
    on the river); the opp-outcome MC's k=2 pair ranks come from a per-env,
    per-street `BoardPairTable` shared by every seat that acts on the street
    (`outcome_features_mc_shared`; MC cost -60% at 6 seats, test
    `shared_pair_table_matches_outcome_features_mc`); the full layout's in-place
    encoders write the packed copy too, or ONLY it (`encode("full", out=None,
    ...)`; the CUDA rollout then never writes the
    4.7 KB dense rows — digest-identical, `PLO5BP_NO_FULL_PACKED=1` = old path);
    `plo_board_strength_batch` (every player's made-hand strength).
  - Recipe rounds (`scripts/recipe_run.sh`, 15M rows/update, all from u1290
    with the same random stream; `scripts/round_summary.py` = per-run means
    over checkpoints; vs u1290, sampled / argmax-vs-argmax):
    round 1 (old critic, 2048x4): entropy 0.10 +0.05 / +0.14; 0.06 (+ sizing
    0.3) +0.34 / -0.03; 0.03 +0.50 / -0.10 (the argmax loss sits in the DEEP
    tier). Round 2 (entropy 0.06, new critic): 2048x4 + pool +0.32 / +0.01
    (= the old critic's run), 2048x4 no pool +0.27 / 0.00, distilled 1024x3
    +0.37 / +0.14, 512x3 +0.34 / +0.13 — the small students win, and keep
    their deep-tier argmax (+0.2 vs -0.1). The new critic's loss kept falling
    (2.55 -> 2.44) while the old one's rose (2.70 -> 2.74).
  - **vSix6** (`scripts/vSix6_guardian.sh`) = the result: actor 1024x3 (warm
    from r2b = the distilled student after round 2), critic 1536x2 SiLU
    (offline 1536x2 ~ 2048x3 at 2.2x less compute; first launch installs
    `checkpoints/critic_silu1536x2_u1290.pt` + one critic-only update), entropy
    0.06, sizing-entropy 0.3, 1 extra critic epoch of 128 minibatches,
    `--no-grad-checkpoint`, `--obs-real-f16`, host batch, obs rev 1 (drop-in for
    the live site; its old code cannot rebuild the new critic -> only the
    review's true EV is off until a deploy). Metrics: sampled h2h and TOP action
    vs sampled play against the live vSix5_1248 (argmax-vs-argmax flips on
    genuinely mixed spots — a weak signal).
- **Plateau check + recipe rounds 3-5 (2026-09-28, owner /goal "ensure it is
  still improving, test ways to make it stronger, resume")**:
  - vSix6 STALLED after ~u1340: 1389 vs 1346 sampled +0.03 +- 0.02, argmax
    +0.006 +- 0.013 (43 updates, ~7B rows); 1389 vs the live 1300 +0.13 / +0.15.
    Paused at u1390 (`runs/vSix6.stop`). Weight averages help a little and do not
    need a wider window: `avg_1380_1389` (and `avg_1370_1389`) vs 1389 argmax
    +0.025 (z 2.4); avg_1380_1389 vs the live vSix6_1300 sampled +0.140 (z 6),
    argmax +0.163 (z 11) — the promotion candidate (site code loads it).
  - **The pod's /workspace is a ~20 GB network-volume QUOTA** (`df` shows the
    whole MooseFS cluster, useless — measure `du -sb /workspace`). Full = every
    trainer dies SILENTLY at its next checkpoint write (the traceback cannot be
    written either); round 3's first launch died that way. Old checkpoints are
    moved to the desktop (`checkpoints/pod_archive/`, SHA-256 checked both sides,
    then removed from the pod). Keep the volume under ~15 GB.
  - Rounds = `scripts/recipe_run.sh` candidates from vSix6_1390 with its Adam
    state, pool and random stream (common random numbers — update 0 of an
    unchanged recipe is bit-identical), 10 updates at 220k envs / 15M rows,
    scored vs vSix6_1390 (h2h_cross seed 7, sampled + argmax) at +4/+6/+8/+10
    and averaged (single checkpoints swing +-0.05). Numbered file `<stem>_<1389+N>`
    = N updates after the start (train.py numbers from the 2nd update).
  - Round 3 (means, sampled / argmax): control -0.001 / +0.034; 4x optimizer
    steps (64 minibatches) -0.021 / -0.035; lr 3e-4 -0.028 / -0.004; both -0.031
    / -0.013; entropy 0.045 +0.057 / +0.017 (deep-tier argmax -0.023 vs +0.084).
    Bigger or faster steps HURT: the updates are noise-dominated.
  - Round 4 (all at entropy 0.045): lambda 0.8 +0.103 / +0.046 (every tier
    positive); lambda 0.6 +0.141 / +0.023 (deep argmax negative again); 256
    all-in runouts +0.022 / -0.064 (deep argmax -0.18, unexplained); lr 7.5e-5
    +0.085 / +0.047 (every tier positive). Less noise = stronger model.
  - Why production stalls while search-scale runs improve (hypothesis): each
    update draws only 30 (seats, stacks) setups and splits the rows among them;
    164M rows make every one of the 32 steps point the same way — toward THOSE
    30 setups — and the next update pulls toward 30 others. More rows cannot
    reduce that; more setups per update, a lower lr or a lower lambda can.
  - Round 5 (entropy 0.045 + lambda 0.8 + lr 7.5e-5 = "r5a"): +0.117 / +0.051;
    + 30 setups per tier (`--configs-per-tier 30`) +0.119 / +0.072 (best of all
    18 runs; +4/+6/+8 only — stopped early); r5a with the deep tier at 0.06
    +0.105 / +0.057 (not needed once lambda/lr are lower); entropy 0.045 + 30
    setups alone +0.063 / +0.016 (= without) — setups matter only once the other
    noise is cut. Every good variant jumps in ~6 small updates, then levels:
    the recipe moves the fixed point; whether production keeps refining from
    there is what the resumed run's checkpoints (scored vs 1390) will show.
    90 setups at 220k envs = 2.4k envs per sub-rollout ran 1.8x slower
    (fixed per-step costs); at 1.76M envs (19.5k per sub) expect ~+20%.
  - **vSix6 RESUMED 2026-09-28 11:52 UTC** from u1391 (= the u1390 weights, Adam
    + pool restored) on entropy 0.045, lambda 0.8, lr 7.5e-5; relaunched after
    its first update with 30 setups per tier (guardian defaults `ENTROPY`,
    `LAMBDA`, `LR`, `CONFIGS_PER_TIER` — env-overridable). Launch:
    `NUM_ENVS=1760000 setsid nohup bash scripts/vSix6_guardian.sh`. First
    read: vSix6_1392 (2 updates on the new recipe, 10 setups) vs 1390 sampled
    +0.121 +- 0.022, argmax +0.040 +- 0.010 — every tier positive; the stalled
    recipe's last 43 updates gave +0.03 / +0.006. vSix6_1395 (6 new-recipe
    updates, the last 3 with 90 setups; 24 min per update vs 21) +0.091 +-
    0.022 / +0.049 +- 0.013, every tier positive.
