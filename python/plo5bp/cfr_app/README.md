# CFR Solver (desktop app)

A local, offline solver for heads-up and multiway NLH spots on top of the native
Rust CFR solver: build a root (street, board, pot, stacks, bet sizes, ranges), solve
it, and browse the strategy as a 13×13 matrix or a table, node by node. It never runs
on the website.

## Install

Double-click **`Install CFR Solver.bat`** in the repo folder. If the folder has no
Python environment yet, it checks for Python 3.11+ and the Rust toolchain, offers to
build one (venv → `pip install -e ".[dev,desktop]"` → `maturin develop --release`,
5–10 minutes), then puts **CFR Solver** on the Desktop and in the Start Menu. By hand:

```bat
py -3 -m venv .venv
.venv\Scripts\pip install -e ".[dev,desktop]"
.venv\Scripts\maturin develop --release
.venv\Scripts\python scripts\install_cfr_desktop_shortcut.py
```

Remove the shortcuts with `scripts\install_cfr_desktop_shortcut.py --uninstall`.
The whole-PC setup (GPU torch, tests) is in `SETUP.md`, "A Windows PC".

## Run

- the shortcut, or `.venv\Scripts\pythonw scripts\cfr_app.py` — its own window, no console;
  the local server lives on a random 127.0.0.1 port only while the window is open;
- `.venv\Scripts\python scripts\cfr_app.py --browser` — a normal server on
  http://127.0.0.1:8766 for a browser (development).

Closing the window while a solve runs asks first: stop and save, close now, or keep
solving. A solve that dies anyway (closed, killed, a crash) leaves its last live
snapshot, which the Library lists as **Interrupted** and opens like any solution.

## Which algorithm

The Algorithm menu offers only what the root can use and picks the best one when the
street or seat count changes:

| algorithm | for | "Threads" box |
|---|---|---|
| **Full-range DCFR** (`dcfr_vector`) | heads-up river and turn | real threads: a turn root solves its rivers in parallel (same result for any count) |
| **Sampled DCFR** (`dcfr`) | flops (hand buckets), multiway postflop | deals per iteration, sampled one after another |
| **MCCFR** (`mccfr_es`) | preflop and multiway | not used (disabled) |

Full-range DCFR updates every hand and every runout in each iteration: a standard river
tree (5 sizes, 50 bb, full ranges) reaches 0.05 bb in ~100 iterations (2–3 s), where
sampled DCFR needed ~200k iterations (70 s) for 1 bb. It builds the whole tree before
the first iteration, so **Validate** first: it shows the memory the solve needs against
the budget below, how many infosets (rows) the report will have and the tree size — or
why the solver would refuse the root. Narrow ranges shrink a turn report a lot.

"Check expl every (s)" measures exploitability while the solve runs (never more than
~20% of the run), so the status shows a live number without stopping the solve.

## Where things go

Everything lives under `data/cfr/` (git-ignored), or under `CFR_APP_DATA_DIR` if set:
`app_jobs/` (solves — `<job>.json` when finished), `uploads/` (opened files),
`app_export/` (Export copy). The Library scans that folder. Problems at start-up are
logged to `runs/cfr_app.log`.

## Settings (environment variables)

| variable | effect |
|---|---|
| `CFR_APP_DATA_DIR` | put every app file under this folder instead of `data/cfr/` |
| `CFR_APP_INPROCESS=1` | solve in the app process instead of a child process (debugging; a native crash then takes the app down) |
| `CFR_RAM_BUDGET_MB` | the most memory a solve's table may use (default: 60% of the PC's RAM, at least 1 GB) |

`--host` binds loopback only; `--allow-remote` is needed for anything else, because
anyone who can reach the port could then drive the solver.

## What the numbers mean

"Expl" is how much a perfect opponent could win against the strategy, in big blinds
per hand, labelled by kind: **exact** (a real certificate for this tree), **sampled**
(best response over sampled runouts — an upper bound), **estimate** (an in-solve
sample the final number replaces), **proxy** (preflop / multiway: a
perfect-information stand-in that does not shrink with more iterations — not a Nash
certificate).

Heads-up river and turn solves also carry every hand's **EV** (bb) and **equity** at
every node (the solver's final best-response pass, average strategies): the matrix
tooltip and the hand detail show them per class (reach-weighted over its combos), the
Table view per combo. EV is the hand's expected share of the final pot minus the chips
it still puts in from that node; equity is its chance to win at showdown against the
range that reaches the node (ties count half).
