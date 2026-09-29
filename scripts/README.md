# scripts/

One line per script, grouped by what it is for. Library code lives in
`python/plo5bp/` (training: `plo5bp/train/`, evaluation: `plo5bp/evaluation/`);
these are entry points. Retired scripts are in `archive/` (training: `archive/training/`,
OCR diagnostics: `archive/ocr/`) — kept to read, not to run.

## Training (the pod)

| Script | What it does |
|---|---|
| `train.py` | the PPO trainer (a thin wrapper around `plo5bp.train.loop.main`; `--help` lists every flag) |
| `vSix6_guardian.sh` | **the main run**: launches vSix6, resumes it after crashes, kills it when hung (heartbeat) |
| `vMin3_guardian.sh` | the minimal-observation run (paused) |
| `guardian_lib.sh` | what the guardians share: NUMA pinning, resume-file choice, heartbeat check, Inductor cache |
| `recipe_run.sh` | one recipe-search candidate at search scale (from a fixed start, 10 updates) |
| `tune_run.sh` | one hyperparameter-tuning candidate (one knob changed) |
| `sweep_guardian.sh` | one network-size sweep run |
| `check_restart_sync.py` | before a relaunch: do the guardian's flags still match the live-control file? |
| `exactness_check.py` | is a code change still bit-exact? (trains tiny recipes at HEAD and in the working tree, compares every tensor) |
| `plot_metrics.py` | charts `runs/<stem>.metrics.jsonl` as one HTML page (or `--table`) |
| `pod_disk_report.sh` | disk usage of the pod's volumes (the /workspace quota) |

## Measuring checkpoints

| Script | What it does |
|---|---|
| `h2h_eval.py` | head to head of two checkpoints (same layout and obs revision), duplicate deals |
| `h2h_cross.py` | head to head across observation layouts / revisions (how the site serves each) |
| `h2h_league.py` | round robin of N checkpoints → ratings with intervals + a non-transitivity test |
| `panel_eval.py` | the standing robustness track: a checkpoint (or every Nth of a stem) rated against a FIXED panel of references + baselines (`panels/*.json`; the panel round robin cached) — one rating on a stable scale, interval, non-transitivity |
| `round_summary.py` | a recipe round's mean edge per run over its checkpoints (per opponent) |
| `run_watch.py` | runs beside a training run: strength vs fixed references, sharpness, trainer health |
| `sweep_eval.py` | the size sweep's evaluator (paired h2h at equal update counts + utilization) |
| `sweep_report.py` | one table of every sweep comparison |
| `policy_sharpness.py` | how deterministic a policy is on one cached set of states (collapse watch) |
| `utilization_probe.py` | dead units / effective rank of actor and critic |
| `update_snr.py` | signal-to-noise of one training update (two seeds from one start) |
| `probe_suite.py` | fixed-spot probe families (fold/raise frequencies, Q canaries) |
| `critic_calibration.py`, `critic_offline.py`, `convert_offline_critic.py` | critic diagnostics on a rollout dump (`PLO5BP_DUMP_BATCH`) and offline critic training |
| `distill_size.py` | actor size study by distillation (students of a teacher) |
| `average_checkpoints.py` | weight average of several checkpoints of one run |
| `evaluate.py` | a checkpoint against scripted baselines (serial env) |
| `exploitability.py` | a PPO-trained exploiter against a frozen checkpoint |
| `check_mixture_usage.py`, `check_sizing_dist.py` | v5 / v4 sizing-head inspections |
| `convert_v4_to_v5.py` | the v4 → v5 warm-start converter (historical, still valid) |
| `bankroll_sim.py` | bankroll-requirement simulation |

## Performance measurements

| Script | What it does |
|---|---|
| `throughput_probe.py` | full updates at several num_envs / rollout lengths: rows/s, peak memory |
| `bench_subrollout.py` | per-step cost by region vs env count |
| `rust_bench.sh` | Criterion benchmarks of the engine's hot kernels, one core (`rust_engine/bench`; `-- --save-baseline x` / `--baseline x` to compare) |
| `bench_opp_act.py`, `bench_upload.py`, `bench_obs_blocks.py`, `bench_opp_outcome.py` | micro-benchmarks of rollout pieces |
| `profile_rollout.py`, `summarize_profile.py` | serial vs batched rollout; summarize a `--profile-one-update` trace |
| `smoke_test.py` | random legal hands through the engine, invariants checked |
| `diag_pot_inflation_value.py`, `diag_side_pots.py` | older value-loss / side-pot diagnostics |

## NLH solver and GTO tools (CFR)

`cfr_app.py`, `install_cfr_desktop_shortcut.py` (the desktop CFR Solver), `cfr_solve.py`,
`cfr_batch.py`, `cfr_overnight.py`, `cfr_verify.py`, `cfr_export_labels.py`,
`gto_label_factory.py`, `gto_train.py`, `gto_train_from_labels.py`, `gto_probe.py`,
`train_policy_from_cfr.py` and `export_pushfold_14_charts.py`. Which of them make a
servable, badge-eligible PolicyNet, in what order: `docs/ops/GTO_PIPELINE.md`. The Step 6 /
Step 7 teacher campaigns and the push/fold study scripts are in
`archive/gto_campaigns/` (with a README). See also CLAUDE.md, "NLH GTO teacher".

## Live capture (OCR)

`ocr_collect_fixtures.py` (labeled frames into the tracked fixtures),
`window_affinity_monitor.py` (ClubGG window diagnostic). The older OCR diagnostics are in
`archive/ocr/`.

## Site and repository

| Script | What it does |
|---|---|
| `deploy_prod.sh` | ships code or a model to production, and undoes either (the owner runs it) |
| `check.sh` | runs CI's checks locally |
| `rust_check.sh` | the Rust gate: `cargo fmt --check`, clippy `-D warnings`, the Rust tests (`--fix` formats first); `rust_env.sh` = its shared cargo environment |
| `hooks/` | git hooks (secret scan) |
