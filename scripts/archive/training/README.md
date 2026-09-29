# Retired training scripts — do not run

Moved here on 2026-09-28 (ML-035 / ML-058 / REPO-011 / REPO-012) so the repository
root and `scripts/` show only what is in use. They are kept to be READ (old recipes,
old stems' flags); git history has every version. None of them should be launched:
they name retired stems, warm-start checkpoints the trainer refuses, or rely on
defaults that changed (block rotation, 2048×4 sizes).

| Files | What they were |
|---|---|
| `v1/launch_optimized{2..9}.sh`, `v1/launch_auto.sh`, `v1/watchdog_*.sh` | pod launchers / watchdogs of the v1 `optimized<N>` Beta-head stems (May–June 2026) |
| `v1/launch_vtwo.sh`, `v1/watchdog_vtwo.sh` | the first v2 anchor-head stems (`vTwo<N>`) |
| `v1/analyze_ftr.py`, `v1/analyze_hv.py` | v1 log parsers (read hard-coded %TEMP% logs) |
| `vThree_guardian.sh`, `vFour*_guardian.sh`, `vFive1_guardian.sh` | guardians of the v3/v4/v5 PLO stems |
| `vSix4_guardian.sh`, `vSix5_guardian.sh` | the full-obs lineage before vSix6 (vSix5 converged at u1290) |
| `vMin1_guardian.sh`, `vMin2_guardian.sh` | the first minimal-obs stems (vMin3 supersedes them) |
| `nlh_guardian.sh` | the retired NLH PPO lineage (exits 1) |
| `run_curriculum.sh`, `orchestrate_*.sh`, `watch_curriculum_finish.sh` | the May v1 stack curriculum (ran in an old checkout; Windows `wmic`/`taskkill`) |
| `wait_vSix3_60.sh`, `pod_launch_vSix4.sh`, `pod_compare_wallclock.py` | one-off pod helpers of the vSix3/vSix4 era |
| `pod_prune_disk.sh` | DELETES checkpoints (keeps the latest 8 of optimized7/9): dangerous, and its stems are gone |

What runs today: `scripts/vSix6_guardian.sh` (the main run), `scripts/vMin3_guardian.sh`
(paused), both on `scripts/guardian_lib.sh`; searches with `scripts/recipe_run.sh`,
`scripts/tune_run.sh`, `scripts/sweep_guardian.sh`. See `scripts/README.md`.
