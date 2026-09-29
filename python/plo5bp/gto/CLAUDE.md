# NLH GTO teacher, CFR solver & the CFR Solver app — notes

Loaded when you work under `python/plo5bp/gto/` (pointers from `cfr_app/` and `rust_engine/src/cfr/`).

### NLH GTO teacher (`rust_engine/src/cfr/`, `python/plo5bp/gto/`, `cfr_app/`)

Native Rust CFR (DCFR for HU postflop, external-sampling MCCFR for
preflop/multiway) → label export → supervised PolicyNet → `PolicyNetHost`
(env `PLO5BP_GTO_CHECKPOINT`). The command sequence (cfr_batch →
train_policy_from_cfr → gto_probe --stamp → serve) is `docs/ops/GTO_PIPELINE.md`;
the step-6/7 campaign scripts are in `scripts/archive/gto_campaigns/`.
HU river / turn roots can also run `algorithm="dcfr_vector"` (`cfr/vector.rs`,
2026-09-28): full-range DCFR over the whole public tree — every hand and every
runout each iteration, < 0.05 bb in ~100 iterations on a standard river tree
(sampled DCFR: ~1 bb after 200k); it builds the whole tree up front, so its
memory check (`cfr_estimate_memory(algorithm=, range_oop=, range_ip=)`) counts
the tree and the report rows. Its best response is pinned to `dcfr.rs`'s.
Invariants after the 2026-09-20 review:

- **Solver**: ES-MCCFR accumulates the average strategy at the OPPONENT's
  sampled nodes (own-reach weighting); deals are sampled from the true
  joint (rejection on card collision) and the BR weights hero combos by
  their true marginal; the FINAL exploitability estimator runs on every
  exit path and notes carry an honest `expl_kind=` (`exact_infoset`,
  `hero_enum`, `sampled_runout_br`, `mc_poll`, `mc_br_proxy`);
  `Range::parse` errors on unknown tokens (`#44` = combo id, `44` = pocket
  fours); ALLIN raises to `max_raise_chips()` and a jam-sized `RAISE_x` is
  merged into it; invalid roots (sub-blind stacks, 0-chip pots,
  `max_iterations=0` with no stop condition, over-budget trees) are
  `ValueError`s, never panics. Strategies solved BEFORE the review for
  preflop/multiway (or with ranges) should be re-solved.
- **Export/labels**: CFR `ALLIN` (and any raise that clamps to the stack)
  maps to the grid-LEGAL jam anchor via `labels.jam_anchor_index`, never
  blindly to the top atom; facing a jam, `ALLIN` with `max_raise == 0` is a
  CALL; teacher mass on an illegal action raises
  `IllegalTeacherMassError`. Only reports with a final estimator kind and
  no promoted marker are exploitability-VERIFIED — an `early_stop=` counts
  only with `expl_kind=exact_infoset`/`hero_enum` (re-computed on exit) and
  never for `stop_file`; everything
  else lands in `unverified/` and is skipped by teacher export
  (`--allow-unverified-expl` to override). Pre-review label shards and
  PolicyNet checkpoints are invalid — re-export and retrain.
- **Serve**: `PolicyNetHost` re-encodes postflop nodes to the canonical
  obs the labels were built from (current-street history only,
  `total_commit := street_commit`, blind seats None), serves GRID chips
  (no refine head) and snaps near-stack sizes to exactly `max_raise`;
  `supports(seats=, street=)` answers from recorded coverage and the UI
  falls back to PPO (`gto_unsupported`) outside it. The probe scores gates
  AND sizing, fails NaN, uses a stratified SHA-256 holdout and refuses
  train/holdout overlap; the GTO badge requires recorded provenance.
- **Desktop app** (`cfr_app/`): solves run in a spawned child process
  (`CFR_APP_INPROCESS=1` to disable), directories resolve through
  `cfr_app/paths.py` (`CFR_APP_DATA_DIR`), the local API is
  loopback-Host/Origin checked (`CFR_APP_ALLOWED_HOSTS`), range text goes
  through one strict parser (`/api/range/parse`), node views are keyed by
  (seat, path, runout) and weighted by `visit_mass`.

```bash
# NLH training (name the size explicitly, same as PLO — see Training)
.venv/Scripts/python scripts/train.py --variant nlh_single \
  --sizing-head logistic --hidden-dim 2048 --num-layers 4 \
  --stack-dist deep --num-seats-range "2,3,4,5,6"
```
