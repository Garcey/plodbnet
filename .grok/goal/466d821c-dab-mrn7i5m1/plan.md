# CFR Desktop App — Plan (v2 after skeptic gaps)

## Objective
Standalone desktop app (Monker/Pio-like) for the native NLH CFR solver:
build game trees, run solutions, view solutions for quality inspection before training.

## Architecture
- **Package**: `python/plo5bp/cfr_app/`
- **Solver**: `plo5bp.gto.cfr_api.solve` / `RootSpec` / `SolveConfig`
- **Jobs**: background thread + kill-safe `stop_file` / `time_budget_secs`
- **Launch**:
  - Browser: `scripts/cfr_app.py` → http://127.0.0.1:8766
  - Desktop window: `scripts/cfr_app.py --desktop` (pywebview)

## Features (addresses skeptic gaps)
| Feature | Implementation |
|---------|----------------|
| Tree builder | Abstract bet-size tree via `tree_model.build_abstract_tree` + UI Preview |
| Ranges | OOP/IP textareas → `range_oop` / `range_ip` on RootSpec |
| Solve | Start/stop, elapsed/iters/expl progress bar |
| Line browser | Hierarchical `solution_tree` from path tokens / hash buckets |
| Quality | Entropy, F/C/R/AI mass, pure-strategy frac, expl, node aggregates |
| Compare / export | `/api/compare` L1, `/api/export` → `data/cfr/app_export/` |
| Library | Scans `data/cfr/**` |
| Matrix | 13×13 preflop/chart; postflop table |

Honest limit: native strategy JSON is **frequency-only** (no per-hand CFV in dump). Quality panel states this; exploitability comes from the solver report when present.

## Files
- `python/plo5bp/cfr_app/{server,session,strategy_view,tree_model}.py`
- `python/plo5bp/cfr_app/static/{index.html,app.js,style.css}`
- `scripts/cfr_app.py`
- `tests/python/test_cfr_app.py` (28 tests)

## Verification plan
1. `pytest tests/python/test_cfr_app.py -q` → 28 passed
2. Smoke: tree preview, ranges validate, chart quality, real range solve → view
3. E2E: `test_e2e_solve_with_ranges_then_inspect_quality` drives real Rust solve with ranges then inspects quality/export
4. HTML contains range fields, tree preview, line-tree, quality panel
5. `--desktop` entry uses pywebview when available

## Progress log
- v1 shell shipped; skeptics rejected (tree/ranges/quality/desktop gaps)
- v2 closed gaps: tree_model, ranges UI, quality, compare/export, desktop host, e2e
