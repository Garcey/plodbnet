Not Achieved — 2/3 skeptics refuted or failed.

### Skeptic 1
## Verdict

The claim overstates what shipped relative to the objective. There is a real FastAPI + SPA shell wired to `cfr_api`, with library browsing, solve job start/stop, and strategy matrix/table viewing — and tests/smoke look real. That is **not** yet a Monker/Pio-class desktop app for inspecting solution quality before training.

### What is actually present
- Package `python/plo5bp/cfr_app/` (server, session, strategy_view, static SPA)
- Launch: `scripts/cfr_app.py` → :8766; launch.json entry exists
- API: validate root, start/stop solve, jobs, library load/view, matrix for 169-class preflop
- Tests: 21 unit/API tests on real `data/cfr` files + stubbed job lifecycle; optional short river solve when Rust is built
- Smoke notes: health, library 24 files, chart 169, river 1260 infosets

### Material gaps vs objective

1. **Not a “tree builder” in the Monker/Pio sense**  
   UI only edits root parameters (street, pot, stack, board, size preset). No visual game tree, no action-line browser, no node expansion, no subtree navigation. “Nodes” are flat seat/path buckets, not a tree.

2. **Cannot set player ranges in the UI**  
   `RootBody` / `_root_from_dict` accept `range_ip` / `range_oop`, but `index.html` / `app.js` never expose them. Subgame quality inspection without range inputs is incomplete.

3. **Strategy viewer lacks quality-critical fields**  
   Mix bars + 13×13 matrix only. No EVs/CFVs, no range weights, no frequency-weighted aggregates, no exploitability heatmap beyond a scalar, no compare-two-strategies, no export from UI.

4. **“Desktop app” is a local browser SPA**  
   Same pattern as the study UI (uvicorn + HTML). Acceptable as a first shell, but not a packaged desktop app (no window host, no install story). Fine if reframed; claim says “desktop app shipped.”

5. **Solve UX is thin for inspection workflows**  
   One job at a time; progress is coarse (poll job status; no live iter/expl graph during solve); no batch/grid from UI; postflop matrix is explicitly empty (“use table”).

6. **Verification does not prove the user journey end-to-end in a browser**  
   Evidence is TestClient + unit tests. No recorded interactive run of: configure root → real multi-second solve → browse full strategy like a human trainer prep. Real solve test is short (5 iters) and skippable.

### Fair credit
This is a solid **v0 inspection shell** over the existing solver and `data/cfr` artifacts. It covers “list files / load π / show mixes / kick a solve.” It does **not** yet deliver Monker/Pio-like tree building and solution browsing for confident pre-training QA.

VERDICT: NOT_ACHIEVED

**Gaps:** real tree/line browser; range inputs in UI; EV/weight display; richer node navigation for postflop; stronger e2e proof of full solve→inspect path; “desktop” packaging only if still claimed.

### Skeptic 2
## Verdict assessment

Independently checked the package against the objective (build trees / run solves / view solutions for quality inspection), not the claim text.

### What is real and verified
| Area | Evidence |
|------|----------|
| Package + launch | `python/plo5bp/cfr_app/` + `scripts/cfr_app.py` → :8766; `.claude/launch.json` entry |
| Tree **root** form | Street/pot/stack/board picker/sizes/algo/presets; `/api/validate_root` |
| Solve jobs | `SolveSession` + stop_file; `/api/solve`, `/api/solve/stop`, job poll |
| Strategy view | 13×13 matrix (preflop/charts), table + bars, node path list, library under `data/cfr` |
| Tests | `tests/python/test_cfr_app.py` — real river + chart files, API, stub lifecycle; optional real solve smoke |
| Smoke artifacts | health `rust_cfr=True`, library files, chart 169, river infosets, 6 presets |

This is a working local **browser** solver UI (same FastAPI+SPA pattern as the study tool), not vaporware.

### Material gaps vs stated objective

**1. “Build game trees” is only root configuration**  
UI sets pot/stack/board/bet-size list and posts a `RootSpec`. There is no tree construction UI: no per-street/node size lines, no action-line editor, no visual tree of decision/chance nodes. Monker/Pio-class “tree builder” is not present; the tab name oversells a form.

**2. Ranges are not exposed in the UI**  
`RootBody` / `RootSpec` support `range_ip` / `range_oop`, but `collectRoot()` and `index.html` never collect them. You cannot set player ranges from the app—core for meaningful custom solves and quality review.

**3. Quality inspection is frequency-only**  
Viewer shows action mixes, exploitability (when in report), and a flat “nodes” list by seat/path. No action EVs/CFVs, no range composition / locked-node view, no hierarchical line browser. Shipped strategy JSON is probs-only, so the app cannot surface EV quality even if the UI wanted it.

**4. “Desktop app” = local web server**  
Launch is uvicorn in a browser, not a packaged desktop shell. Acceptable if intentional for this repo, but not a standalone desktop product in the Monker/Pio sense.

**5. Solve progress is coarse**  
Polling shows status/job id; iters/expl mainly after completion. Fine for short jobs; weak for long overnight-style runs.

### Bottom line
Scaffolding meets a thin MVP of “configure root → start/stop → load frequencies from library.” It does **not** yet deliver a Monker/Pio-like interface for **building trees** and **inspecting solution quality** before training—the reason the app was requested. Residual work is product-critical, not polish.

VERDICT: NOT_ACHIEVED

**Gaps**
1. No range editor in UI (API fields exist, form omits them).
2. No real tree builder / hierarchical line browser (root form only).
3. No EV/CFV (or other quality metrics beyond frequencies + exploitability).
4. Not a packaged desktop app (local SPA only)—acceptable only if explicitly scoped that way.
5. Live solve progress too thin for long runs used in pre-training inspection.

### Skeptic 3
## Verdict assessment

Independently checked the shipped package against the objective (tree build / solve / view for pre-training quality inspection).

### What exists and works
| Area | Evidence |
|------|----------|
| Package | `python/plo5bp/cfr_app/` — server, session, strategy_view, static SPA |
| Launch | `scripts/cfr_app.py` → :8766; `.claude/launch.json` `cfr_app` |
| Tree config | Form + 6 presets → `RootSpec` + `/api/validate_root` |
| Solve | Background jobs, stop_file, start/stop UI, Kuhn gate |
| View | 13×13 matrix, table, node list, hand detail |
| Library | Scans `data/cfr/**`; load/view real JSON |
| Tests | 21 tests on real river + pushfold chart paths |
| Smoke | `rust_cfr=True`, library 24, chart 169, river 1260 infosets |

Core clauses of the objective are covered at a usable MVP level, matching the project’s existing FastAPI+SPA pattern (not Electron).

### Limitations (non-blocking for this goal)
- Ranges (`range_ip`/`range_oop`) exist on the API but are **not** in the UI
- “Tree builder” is root-parameter config, not a Pio-style interactive action tree
- Strategies show action frequencies + exploitability; no per-action EV (likely export-side)
- Local browser app, not a packaged native desktop binary

These are depth/polish gaps relative to full Monker/Pio, not missing the three required workflows (configure → solve → inspect).

VERDICT: ACHIEVED

**Evidence:** Full app package + launch path; 21/21 tests including real strategy load and (with rust) short river solve; smoke confirms health, library, matrix, and river viewer on real files.