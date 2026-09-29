# scripts/ — which notes to read

- Training, evaluation, guardians, pod runs: **`docs/training.md`** first (never train a size by accident; bit-exact defaults; `exactness_check.py`); run history in `docs/training-log.md`. The owner's recipes live in the guardians — never change a guardian's recipe lines unasked.
- `deploy_prod.sh`: `docs/ops/PRODUCTION.md` (the owner runs it; never deploy on your own).
- CFR / GTO scripts: `python/plo5bp/gto/CLAUDE.md`. One line per script: `scripts/README.md`.
