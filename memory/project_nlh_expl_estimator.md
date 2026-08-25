# HU river exploitability estimator (2026-08-12)

**Root cause (two stacked issues):**

1. **`br_value` was perfect-info BR** — maxed using the sampled opponent
   combo. That overestimates NashConv by the value of seeing villain's
   hole. Kept as `ExplKind::DealBr` for tests.
2. **Paired 48-sample (BR−π) MC is extremely high-variance.** Even after
   switching to infoset BR, 128 paired deals on a 4×4 tree still read
   ~2 bb when exact enumeration is < 1 bb. That is estimator noise, not
   a Nash floor.

**Fix:**
- Infoset BR (`infoset_br_value`) against the opponent range.
- Small support (n0*n1 ≤ 10k): exact deal enumeration.
- Full-range final report: subsampled hero-enum BR + MC π (not paired).
- Poll (every 50 iters): 24 paired MC deals, early-stop only.
- Units still NashConv/2 in bb/hand.

- Final report: 128 deals (`EXPL_DEAL_SAMPLES`)
- In-loop early-stop poll: 24 deals
- Old deal-BR kept as `ExplKind::DealBr` for the regression test
- `TEACHER_MAX_EXPL_BB` stays 1.0

**Before / after (same seed, board, iters):**

| Root | Before (deal-BR, 48) | After (infoset BR) |
|---|---|---|
| s3_s3_i2 20k | 4.098 bb | **0.673 bb** |
| quads-on-board 10k | 4.82 bb | **1.000 bb** |
| tiny 4×4 jam/check 3k | n/a (was ~2 bb paired-MC) | **0.015 bb** |

**1.0 bb is now a reachable teacher bar** on a 20k-iter HU river.

**Badge:** `checkpoints/gto_step6.pt` unstamped. Holdout probe lives
under `holdout_probe` (net matched that teacher). `is_gto_validated`
is false — not a 1.0 bb Nash teacher.
