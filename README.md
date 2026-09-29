# WrapGTO

An AI for **PLO5 double-board bomb pots**, trained by self-play, and the website it
powers — **[wrapgto.com](https://wrapgto.com)**: Study (enter any spot, see what the
model plays and why), the Trainer (play hands against it and get graded), and private
**home games** for clubs of friends (PLO5 / PLO6 / PLO67 tables with a verifiable
shuffle and every decision graded after the hand).

**One project, several names:** WrapGTO is the product; this repository is `plodbnet`;
the Python package is `plo5bp`; older notes call it `plodbbot`.

## The parts

| Part | Where | What |
|---|---|---|
| Engine | `rust_engine/` | the game in Rust — dealing, betting, double-board payouts, observation features, an NLH CFR solver — imported as `plo5bp._engine` |
| Training | `python/plo5bp/`, `scripts/train.py` | self-play PPO (actor + centralized critic) on a RunPod GPU pod |
| Website | `python/plo5bp/ui/` | the FastAPI app: Study, Trainer, home games; with `PLO5BP_PUBLIC=1` it is wrapgto.com (sign-in, admin) |
| Local tools | `python/plo5bp/ocr/`, `tools/pokernow/`, `python/plo5bp/cfr_app/` | live-table study aids (ClubGG screen capture, a PokerNow userscript — never on the website) and the CFR Solver desktop app |
| Operations | `ops/`, `scripts/deploy_prod.sh`, `.github/` | the production server's files, the deploy, CI |

## Getting started

```powershell
py -3 -m venv .venv
.venv\Scripts\pip install -e ".[dev,desktop]"
.venv\Scripts\maturin develop --release            # builds the engine (from the repo root)
.venv\Scripts\python -m uvicorn plo5bp.ui.server:app --port 8765    # the study tool
bash scripts/check.sh                              # what CI runs (in Git Bash)
```

Full setup — a Windows PC, the training pod — is in [SETUP.md](SETUP.md). Training
always names the network size: `scripts/train.py --hidden-dim … --num-layers …`
(see CLAUDE.md, "Training").

## Documentation

- [docs/README.md](docs/README.md) — the map of every document
- [docs/ops/PRODUCTION.md](docs/ops/PRODUCTION.md) — running wrapgto.com (deploy, models, rollback, alerts)
- [CLAUDE.md](CLAUDE.md) — the detailed working notes (current training run, invariants, subsystems)
- [learn/](learn/README.md) — a course on how the model is trained, written for the owner

## Conventions worth knowing

- **Chips**: 1 big blind = 10,000 chips (cent precision at $20/bb); the default table
  is 6 seats, 20 bb stacks (200,000), a 3 bb ante (30,000). All chip math is integer
  and payouts sum to zero.
- **Correctness first**: observations and engine state are bit-exact reproducible,
  and the tests lean on parity and exactness.
- **Live changes need the owner's OK** — code or model, every time.

All rights reserved — see [LICENSE](LICENSE). Security issues: [SECURITY.md](SECURITY.md).
