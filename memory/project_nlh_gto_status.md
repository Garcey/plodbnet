# NLH GTO stack status (Mode 0)

**Product:** Trainer-first sampleable PolicyNet (not library-only GTO Wizard). Labels from **native rust_cfr** → PolicyNet → `PLO5BP_GTO_CHECKPOINT`.

**Plan:** `.claude/plans/nlh-gto-complete-before-confident-training.md`  
**Architecture:** `.claude/plans/nlh-neural-gto-solver-ground-up-architecture.md`

## Done

- StrategyBackend / PolicyNetHost / Trainer badge seam; ClubGG roots; CFR batch/export/train/probe CLIs; rule bootstrap (prior only)
- Native CFR solver (HU + multiway, preflop→river, pipeline)
- rust_cfr → LabelRecord export + PolicyNet train path
- **Badge GTO AI only:** `source` starts with `rust_cfr` **and** `probe.passed`
- Bootstrap → "Curriculum"; unprobed rust_cfr → "Policy net (unvalidated)"

## Teacher (locked)

Native `rust_engine` CFR only. No external solver.

## Next (default go target)

1. Teacher-quality HU river / full-hand rust_cfr batch (expl-gated)
2. Export → train PolicyNet → probe → serve
3. Mode 2 ValueNet + street re-solve after Mode 0 net is playable

## Constraints

- PLO5 runpod stays running (do not pause for NLH)
- NLH/CFR work local CPU unless user frees GPU
- Do not claim GTO on bootstrap-only checkpoints
