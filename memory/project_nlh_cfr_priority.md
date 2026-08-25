# NLH priority pivot (user 2026-07-16)

PLO5 keeps running. NLH teacher is **native rust_cfr only**.

## Master plan

**`.claude/plans/nlh-native-cfr-solver-master-build.md`** (merged from 4 sub-plans)

| Detail plan | Topic |
|-------------|--------|
| `nlh-cfr-algorithm-game-tree-architecture.md` | DCFR / MCCFR / tree / multiway honesty |
| `nlh-cfr-engine-integration-phase-1-3.md` | PublicState, engine parity, modules |
| `nlh-cfr-card-bet-size-abstraction.md` | Size presets, card abs, memory gates |
| `cfr-batch-solving-scripting-export-ops-layer.md` | CLI batch/export ops |

## Phase order

0 scaffold ✅ → 1 HU river DCFR ✅ → 2 preflop MCCFR + flop abs ✅ → 3 multiway + P→R glue ✅ → **4 labels/Trainer (rust_cfr → PolicyNet)**

## Next go target

Workstream C: teacher-quality rust_cfr batch → export → PolicyNet train → probe.
