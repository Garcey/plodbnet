# Documentation map

Start with the [README](../README.md). The always-current working notes for Claude
sessions are [CLAUDE.md](../CLAUDE.md); these are the longer documents.

## Running things

| Document | What it is |
|---|---|
| [ops/PRODUCTION.md](ops/PRODUCTION.md) | the production runbook — deploy, put a model live, roll back, restore, alerts |
| [models.md](models.md) | which model is live on wrapgto.com, and the evidence for each change |
| [ops/GTO_PIPELINE.md](ops/GTO_PIPELINE.md) | the NLH GTO pipeline: CFR solves → labels → PolicyNet → probe → badge |
| [../SETUP.md](../SETUP.md) | setting up a Windows PC or the training pod |
| [../ops/SERVER_SETUP.md](../ops/SERVER_SETUP.md) | the production server from scratch, backups, monitoring |
| [../PUBLIC_SETUP.md](../PUBLIC_SETUP.md) | running the public website locally; Google / Stripe accounts |
| [../requirements/README.md](../requirements/README.md) | which package versions each machine runs |

## Learning how the training works

| Document | Status |
|---|---|
| [../learn/](../learn/README.md) | the course (master document + plain-English concepts), checked against the code on 2026-07-23 — see its "what changed since" box |
| [TRAINING_UPDATE_WALKTHROUGH.md](TRAINING_UPDATE_WALKTHROUGH.md) | one PPO update step by step (memory, CPU vs GPU) as of July — see its "what changed since" box |

## Design history (newest first)

Kept for the reasoning behind today's code. Code comments cite them by file name.

| Document | Date | Status |
|---|---|---|
| [design/REDESIGN_2026-09-26.md](design/REDESIGN_2026-09-26.md) | 2026-09-26 | **current** — why vSix5 stalled and the vSix6 redesign (the later recipe rounds are in CLAUDE.md) |
| [design/V7_DESIGN.md](design/V7_DESIGN.md) | 2026-07-12 | implemented — the v7 workstreams (obs tail, critic/Q-head work) |
| [design/V7_OBS_IMPL_PLAN.md](design/V7_OBS_IMPL_PLAN.md) | 2026-07-12 | implemented — the obs batch 2 (OBS_DIM 1171), merged 2026-09-22 |
| [design/V7_OBS_CANDIDATES.md](design/V7_OBS_CANDIDATES.md) | 2026-07-12 | reference — every observation idea reviewed for obs v3, kept or not |
| [design/V6_RESEARCH.md](design/V6_RESEARCH.md) | 2026-07 | implemented in part — the upgrade catalog behind the `--v6` preset |
| [design/V5_DESIGN.md](design/V5_DESIGN.md) | 2026-07-06 | implemented — the mixture sizing head and obs v2 (OBS_DIM 1020) |
| [history/AUTOMATION.md](history/AUTOMATION.md) | 2026-07-05 | history — an automation catalog that was never scheduled |

The improvements backlog (`docs/improvements/`) is a local working folder and is not
in git.
