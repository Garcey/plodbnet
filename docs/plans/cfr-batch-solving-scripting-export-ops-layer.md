# CFR Ops Layer: Batch, Scripts, Export

**Status:** design (ops product wedge)  
**Depends on:** Phase 0 scaffold (`cfr_api.py`, `scripts/cfr_solve.py`, Rust `cfr/` types)  
**Master plan:** `docs/plans/nlh-preflop-river-cfr-solver.md`  
**Related:** engine integration, algorithm/tree, card/size abstraction plans  
**Constraint:** PLO5 pod untouched; HU first; teacher is native rust_cfr only

---

## Product role

Monker-class **capability** lives in the Rust CFR core. Our **wedge** is ops:

| Monker-class | plodbnet ops |
|--------------|--------------|
| One GUI solve | **Unattended grid** of roots |
| Manual export | **JSON / JSONL** pipelines |
| Closed binary | **Scriptable** config + CI |

This layer is first-class from Phase 1 (postflop HU), not a Phase 4 afterthought. Target native `solve()` and `source=rust_cfr_*` (`scripts/cfr_batch.py`, `scripts/cfr_export_labels.py`).

```
Config (TOML/JSON)
    │
    ▼
Root grid ──► cfr_batch (Python orchestrator)
    │              │
    │              ├─ per root: solve()  [Rust hot loop, Rayon threads]
    │              │
    │              ▼
    │         strategy JSON (per root, atomic write)
    │              │
    └──────────────┼──► cfr_export_labels
                   │         │
                   ▼         ▼
            LabelRecord JSONL  →  PolicyNet / Trainer (filtered by source)
```

---

## 1. CLI surface

Three entrypoints under `scripts/`. Shared library code under `python/plo5bp/gto/` (not logic-in-scripts).

### 1.1 `scripts/cfr_solve.py` (exists — extend, don't rewrite)

**Role:** one root, interactive / debug / CI smoke.

| Flag | Meaning | Default |
|------|---------|---------|
| `--preflop` | HU preflop blueprint root | off |
| `--street` | 0..3 | 3 if not preflop |
| `--board` | comma card indices 0..51 | required by street |
| `--pot-bb` / `--stack-bb` | public chips in bb | 10 / 100 |
| `--range-ip` / `--range-oop` | range strings (empty = uniform) | `""` |
| `--iters` / `--threads` / `--seed` | SolveConfig | 200 / 1 / 0 |
| `--algorithm` | `dcfr` \| `linear` \| `mccfr_es` \| `vanilla` | `dcfr` |
| `--size-preset` | `micro` \| `coarse` \| `standard` \| `fine` | `standard` |
| `--card-abs` | `none` \| … | `none` |
| `--target-expl` | stop early if exploitability ≤ X bb/hand | 0.5 |
| `--config` | TOML/JSON override file (merges over flags) | none |
| `--out` | strategy report JSON path | stdout only |
| `--export-labels` | optional path: also emit LabelRecord JSONL for this root | none |

**Exit codes:** `0` ok · `1` solver unavailable / runtime fail · `2` invalid root/config · `3` non-converged (if `--require-expl`).

Phase 0 today returns `status=not_implemented` with exit 0 (scaffold). Phase 1+ treats `not_implemented` as fail when `rust_cfr_available()`.

### 1.2 `scripts/cfr_batch.py` (new)

**Role:** unattended grid. Analog of `gto_batch_solve.py` but native CFR.

```text
.venv/Scripts/python scripts/cfr_batch.py --config configs/cfr/river_grid.toml
.venv/Scripts/python scripts/cfr_batch.py --config ... --resume
.venv/Scripts/python scripts/cfr_batch.py --n-roots 64 --streets 3 --seed 0
```

| Flag | Meaning |
|------|---------|
| `--config` | batch TOML/JSON (primary) |
| `--n-roots` / `--seed` / `--streets` | CLI grid without full config (smoke) |
| `--out-dir` | root for strategies + manifest + labels |
| `--workers` | Python process-pool width (outer) |
| `--threads-per-worker` | Rayon threads inside each solve (inner) |
| `--resume` | skip roots with completed marker |
| `--fail-fast` | stop on first hard error |
| `--dry-run` | expand grid, write plan JSON, no solve |
| `--timeout-s` | per-root wall clock (kill + mark failed) |

**Library:** `python/plo5bp/gto/cfr_batch.py` with `BatchJob`, `run_batch()`, `BatchManifest` (mirror `batch_factory.BatchConfig` / `BatchReport`).

### 1.3 `scripts/cfr_export_labels.py` (new)

**Role:** strategy JSON (or batch dir) → `LabelRecord` JSONL. Decouples solve from train so re-export can change mapping without re-solving.

```text
.venv/Scripts/python scripts/cfr_export_labels.py \
  --in data/cfr/batch_xxx/strategies/ \
  --out data/gto_nlh/rust_cfr/labels_seed0.jsonl \
  --max-combos-per-node 40 \
  --source rust_cfr_dcfr
```

| Flag | Meaning |
|------|---------|
| `--in` | single strategy JSON **or** directory of `*.strategy.json` |
| `--out` | JSONL path |
| `--source` | stamp on every record (must be `rust_cfr_*`) |
| `--max-combos-per-node` | cap per-infoset hole rows (same spirit as TS batch) |
| `--map-sizes` | `nearest_anchor` (default) — map solve ladder → `NLH_ANCHOR_SPEC` |
| `--include-cfv` | if strategy has CFVs, fill `value_bb` / `cfv_bb` |
| `--root-only` | only root decision node (fast train shards) vs full tree walk |

**Library:** `python/plo5bp/gto/cfr_export.py` — pure Python walk of strategy tree → `LabelRecord` + `write_jsonl`.

---

## 2. Config file format

**Primary: TOML** (human-editable grids). **JSON accepted** for machine-generated plans (`--dry-run` emits JSON).

### 2.1 Single-root config (`configs/cfr/examples/river_hu.toml`)

```toml
[root]
street = 3
pot_bb = 20.0
effective_stack_bb = 50.0
board = [48, 44, 40, 36, 32]
raise_sizes_pm = [330, 500, 750, 1000, 1500]
allin_atom = true
range_ip = ""          # empty = uniform unblocked
range_oop = ""
root_id = "river_example"
bb_chips = 10000
sb_chips = 5000
ante_chips = 5000

[solve]
max_iterations = 200
target_exploitability_bb = 0.5
thread_num = 4
seed = 0
algorithm = "dcfr"
card_abstraction = "none"
use_isomorphism = true
size_preset = "standard"   # optional; overrides raise_sizes_pm if set

[export]
enabled = false
source = "rust_cfr_dcfr"
max_combos_per_node = 40
```

Maps 1:1 onto existing `RootSpec` / `SolveConfig` fields in `cfr_api.py` + Rust `types.rs`.

### 2.2 Batch grid config (`configs/cfr/river_grid.toml`)

```toml
[meta]
name = "river_hu_clubgg_v1"
clubgg_root = "clubgg_5_10_5"
schema_version = 1

[solve]                    # defaults for every root
max_iterations = 150
target_exploitability_bb = 0.75
thread_num = 2             # inner Rayon; keep low when workers > 1
seed = 0                   # base seed; per-root seed = hash(base, root_id)
algorithm = "dcfr"
card_abstraction = "none"
size_preset = "standard"

[grid]
# Either sample like Mode 0 roots, or explicit product
mode = "sample"            # "sample" | "product" | "list"
n_roots = 64
streets = [3]
seats = [2]
pot_bb_range = [5.0, 30.0]
spr_bands = ["short", "mid", "deep"]   # subset of SPR_STRATA names
board_seed_mode = "from_root_seed"     # deterministic board from RootSample.seed

# mode = "product" alternative:
# spr_points = [1.0, 3.0, 8.0, 15.0, 30.0]
# pot_bb = [10.0]
# boards_file = "configs/cfr/boards_river.txt"  # one board per line

[parallel]
workers = 4                # Python ProcessPool
threads_per_worker = 2     # Rayon; workers * threads ≲ logical CPUs
timeout_s = 600.0

[output]
dir = "data/cfr/river_grid_v1"
strategy_subdir = "strategies"
labels_subdir = "labels"   # optional live export; or post-hoc cfr_export_labels
write_labels = true
source = "rust_cfr_dcfr"
resume = true
atomic_writes = true

[quality]
min_iterations = 50
max_exploitability_bb = 2.0   # soft flag in manifest; hard fail if require_expl
require_expl = false
```

### 2.3 Loader API

```text
plo5bp/gto/cfr_config.py
  load_solve_config(path) -> (RootSpec | None, SolveConfig, export opts)
  load_batch_config(path) -> BatchJob
  expand_grid(job) -> list[RootSpec]   # uses roots.sample_train_roots / iter_spr_grid
  config_fingerprint(job) -> str       # sha256 of normalized JSON for resume keys
```

ClubGG chips always default from `CLUBGG_NLH_ROOT` (`roots.py`); config may not silently change bb/sb/ante without an explicit override + warning.

---

## 3. Output formats

### 3.1 Strategy JSON (solver native)

One file per root: `{out}/strategies/{root_id}.strategy.json`

```json
{
  "schema_version": 1,
  "status": "ok",
  "root": { "...RootSpec as_dict..." },
  "config": { "...SolveConfig as_dict..." },
  "solve_id": "sha256_16chars",
  "iterations_run": 150,
  "exploitability_bb": 0.42,
  "elapsed_ms": 12345,
  "strategy": {
    "root_id": "...",
    "algorithm": "dcfr",
    "size_ladder_pm": [330, 500, 750, 1000, 1500],
    "allin_atom": true,
    "infosets": [
      {
        "key": "hex_or_stable_string",
        "player": 0,
        "street": 3,
        "public_id": "...",
        "private": { "kind": "combo", "id": 312 },
        "actions": [
          { "a": "fold", "p": 0.12 },
          { "a": "check_call", "p": 0.55 },
          { "a": "raise_pm", "pm": 500, "p": 0.20 },
          { "a": "allin", "p": 0.13 }
        ],
        "cfv_bb": null
      }
    ],
    "public_tree_note": "optional compact history encoding for export walk"
  },
  "notes": []
}
```

**Rules:**

- Extend Phase 0 `SolveReport` — same top-level shape `cfr_api.SolveReport` already writes.
- **Atomic write:** write `*.strategy.json.tmp` → fsync → rename (Windows: replace). Resume checks final path existence + valid JSON + `status==ok`.
- **Determinism stamp:** `solve_id = hash(root_dict, config_dict, engine_git_or_version)`; same inputs → same π within float tol (asserted in tests).

### 3.2 Batch manifest

`{out}/manifest.json` (+ append-only `manifest_events.jsonl` for live progress):

```json
{
  "name": "river_hu_clubgg_v1",
  "config_fingerprint": "...",
  "n_roots": 64,
  "n_ok": 61,
  "n_fail": 2,
  "n_skipped_resume": 1,
  "n_labels": 2400,
  "seconds": 1800.5,
  "roots_per_hour": 122.0,
  "failures": [{"root_id": "...", "error": "..."}],
  "clubgg_root": "clubgg_5_10_5",
  "source": "rust_cfr_dcfr",
  "solver": "rust_cfr"
}
```

Per-root marker: `{out}/strategies/{root_id}.done` (or embed completion only in strategy JSON) so `--resume` is O(1) path check.

### 3.3 LabelRecord JSONL bridge

Reuse `labels.py` unchanged at schema level:

| Field | CFR export fill |
|-------|-----------------|
| `source` | `rust_cfr_dcfr` / `rust_cfr_mccfr_es` / `rust_cfr_linear` (prefix **required**) |
| `root_name` | `clubgg_5_10_5` |
| `solve_id` | from strategy JSON |
| `gate_probs` | fold / check_call / sum(raises+allin) |
| `action_probs` | map each raise_pm → nearest `NLH_ANCHOR_SPEC` via existing `map_size_to_anchor` |
| `hero_hole` / `board` / chips | from public node + combo |
| `notes` | `{ "size_ladder_pm", "algorithm", "exploitability_bb", "root" }` |

**Size map is export-only** (abstraction plan hard rule): solve ladder stays coarse; PolicyNet always sees full anchor menu.

Shard layout (mirror TS batch):

```text
data/cfr/<job>/strategies/*.strategy.json
data/cfr/<job>/labels/labels_seed{S}_n{N}.jsonl
data/cfr/<job>/manifest.json
```

Optional symlink/copy into `data/gto_nlh/rust_cfr/` for the existing `dataset.py` / `gto_train_from_labels.py` path.

---

## 4. Parallelism model

### 4.1 Two levels (decisive defaults)

| Level | Where | What | When |
|-------|-------|------|------|
| **Inner** | Rust Rayon (`SolveConfig.thread_num`) | Parallelize **within** one tree (infoset updates / public-node traversal / MCCFR batches) | Single root, large tree (flop/turn) |
| **Outer** | Python `ProcessPoolExecutor` (`workers`) | Many **independent** roots | Batch grids (river × SPR × boards) |

**Do not** use threads for outer parallelism: PyO3 + Rust Rayon already owns CPU; processes isolate GIL and let OS schedule.

### 4.2 Budget rule

```text
workers * threads_per_worker ≤ logical_cpus
prefer: river batch → high workers, threads=1..2
        flop single → workers=1, threads=all
```

Config validation warns if product > 1.25 × CPU count.

### 4.3 Why not Rayon-only batch?

- Batch orchestration needs **resume, timeout, JSON IO, labels export** — Python is the right shell.
- Process crash on one root must not kill the grid (`try/except` + failure list, same as `batch_factory.run_batch`).
- Future: remote workers can swap ProcessPool for a queue without changing strategy schema.

### 4.4 Why not Python multiprocessing for the CFR loop?

- Hot path is tabular regret over millions of infosets — belongs in Rust.
- Phase 0 contract: `solve()` → later `_engine.cfr_solve`; Python only validates, dispatches, serializes.

### 4.5 Worker entrypoint

```text
cfr_batch._worker(root_dict, config_dict, out_path) -> status_dict
  # fresh interpreter, import plo5bp, call solve(), write strategy JSON
```

Pass plain dicts (picklable). No shared memory strategy tables across roots.

---

## 5. PolicyNet / Trainer feed — no re-poisoning

### 5.1 Provenance contract

| Source prefix | Meaning | May train PolicyNet for "GTO" path? |
|---------------|---------|--------------------------------------|
| `synthetic_smoke` | canaries only | **No** (metrics only) |
| `rust_cfr*` | **our** π* | **Yes** (after probe) |
| `rule_bootstrap` | curriculum prior | **No** |

`policy_net._meta_claims_gto` / `source_is_gto_teacher` accept **only** `rust_cfr*`.

```text
ALLOWED_GTO_SOURCES = ("rust_cfr",)           # production teacher
```

- Badge **"GTO AI"** only when: `source` starts with `rust_cfr` **and** holdout probe pass **and** probe labels also `rust_cfr*`.
- Train scripts: default filter `source.startswith("rust_cfr")`; refuse bootstrap/smoke mix for the badge.

### 5.2 Pipeline (after solver trustworthy)

```text
cfr_batch  →  strategies/
cfr_export_labels  →  labels/*.jsonl  (source=rust_cfr_*)
gto_train_from_labels  →  PolicyNet ckpt  (meta.source=rust_cfr_*)
gto_probe  →  holdout rust_cfr JSONL  →  is_gto_validated
Trainer StrategyBackend / UI  →  badge only if validated
```

### 5.3 Anti-poison checklist

1. **Never** write a non-`rust_cfr` source from CFR export.  
2. **Never** mix bootstrap/smoke shards into a production rust_cfr train glob.  
3. Manifest + every LabelRecord carry `solve_id` + `config_fingerprint` for audit.  
4. Probe holdout must be **disjoint roots** (seed split documented in batch config).  
5. Smoke labels stay in separate shards; factory canaries never enter production train globs.  
6. Size mapping loss stays in `notes` if needed later; do not silently drop mass when remapping raise atoms.

### 5.4 Trainer re-solve (later, Phase 4)

Path-tracked ranges: live hand → open subgame root with Bayesian ranges from blueprint → `cfr_solve` → labels for that node. Same export path; same `source=rust_cfr_resolve`. Ops layer already supports one-root `cfr_solve` for this.

---

## 6. Success metrics (ops)

### 6.1 Throughput

| Metric | How measured | Phase 1 target (indicative) |
|--------|--------------|------------------------------|
| **Roots/hour** | `n_ok / (seconds/3600)` in manifest | River HU standard ladder, ~100bb pot mid-SPR: **≥ 30 roots/h** on 8-core desktop (tune after first real solves) |
| **Labels/hour** | `n_labels / hours` | Secondary; depends on max_combos |
| **CPU efficiency** | busy time / wall | outer workers saturated without thrashing |

Log line per batch: `[cfr_batch] roots/h=… ok=… fail=… resume_skip=…`.

### 6.2 Determinism

| Gate | Test |
|------|------|
| Same `RootSpec` + `SolveConfig` + seed → same strategy (L1 / TV distance on π < 1e-5 after fixed iters) | `tests/python/test_cfr_determinism.py` + Rust unit |
| Config fingerprint stable under key reorder | loader test |
| Resume does not re-solve completed roots | batch integration test with pre-seeded `.strategy.json` |

MCCFR: determinism = same seed stream → same samples; document that `algorithm=mccfr_es` is seed-deterministic, not order-of-thread deterministic unless `thread_num=1`.

### 6.3 Resume-on-failure

| Behavior | Spec |
|----------|------|
| Crash mid-root | no final `.strategy.json` (tmp discarded) → retry on `--resume` |
| Crash mid-batch | completed roots kept; manifest rewrite on next run |
| Soft fail (timeout, non-converge) | write `{root_id}.failed.json` with error; count in `n_fail`; continue |
| Hard fail + `--fail-fast` | exit 1 after flushing manifest |
| Corrupt strategy JSON | treat as incomplete; re-solve |

### 6.4 Quality gates (ops-visible)

- Per-root: `iterations_run`, `exploitability_bb` (when computable).  
- Batch: fraction with `expl ≤ target`; fraction failed.  
- Export: label count, gate mass sum checks, no empty `action_probs`.  
- CI smoke: `cfr_solve --preflop` + tiny river board + `cfr_batch --n-roots 2 --dry-run` + export round-trip.

### 6.5 What "done" means for ops wedge

- [ ] `cfr_solve` produces real strategy JSON (Phase 1+) with ClubGG chip fields  
- [ ] `cfr_batch` runs N roots unattended with resume  
- [ ] Config TOML expands via ClubGG `roots` sampling  
- [ ] `cfr_export_labels` → JSONL loadable by `read_jsonl` / `dataset.py`  
- [ ] Manifest reports roots/hour + failures  
- [ ] Determinism test green at `thread_num=1`  
- [ ] Documented source filter so TS shards cannot silently train "GTO" nets once rust_cfr is teacher  

---

## 7. File layout (implementation map)

| Path | Role |
|------|------|
| `python/plo5bp/gto/cfr_api.py` | RootSpec / SolveConfig / solve (extend) |
| `python/plo5bp/gto/cfr_config.py` | **new** TOML/JSON load + grid expand |
| `python/plo5bp/gto/cfr_batch.py` | **new** process pool + resume + manifest |
| `python/plo5bp/gto/cfr_export.py` | **new** strategy → LabelRecord |
| `scripts/cfr_solve.py` | extend flags |
| `scripts/cfr_batch.py` | **new** |
| `scripts/cfr_export_labels.py` | **new** |
| `configs/cfr/*.toml` | **new** example grids |
| `data/cfr/` | default output root (gitignored) |
| `tests/python/test_cfr_batch.py` | grid expand, resume, fingerprint |
| `tests/python/test_cfr_export.py` | map_size_to_anchor + JSONL round-trip |

Reuse patterns from `scripts/cfr_batch.py` / `gto/cfr_export.py`.

---

## 8. Phased delivery (ops relative to CFR core)

| When | Ops deliverable |
|------|-----------------|
| **Phase 0 (now)** | CLI stubs + this plan; config schema frozen as types |
| **Phase 1 (HU postflop CFR)** | `cfr_solve` real; `cfr_batch` river grid; strategy JSON; resume; roots/hour baseline |
| **Phase 1b** | `cfr_export_labels` + train smoke on rust_cfr JSONL (badge still unvalidated until probe) |
| **Phase 2 (preflop)** | batch configs for blueprint; MCCFR seed policy in config |
| **Phase 4** | GTO teacher allowlist is `rust_cfr*` only; badge after probe |

---

## 9. Explicit non-goals (ops)

- GUI solve browser (CLI/JSON only for v1)  
- Distributed cluster scheduler (ProcessPool local first)  
- GPU CFR batching  
- Auto-merge of TS + rust label corpora  
- Changing PLO5 training launch scripts  

---

## 10. First implementation steps (after approval)

1. Freeze strategy JSON schema + `SolveReport` fields (document in `cfr_api` docstring).  
2. `cfr_config.py` load/expand + unit tests (no Rust needed).  
3. `cfr_batch.py` skeleton: dry-run + resume over **stub** `solve()` (Phase 0) to prove ops before Phase 1 CFR lands.  
4. `cfr_export.py` skeleton: empty infosets → zero labels; fixture strategy JSON → LabelRecords once Phase 1 dumps π.  
5. Wire real `_engine.cfr_solve` when Phase 1 core is ready — ops shell already stable.
