# NLH GTO pipeline — from CFR solves to the "GTO AI" badge

The one sequence that produces a servable, badge-eligible NLH PolicyNet. Run every
command from the repo root. Names like `teacher_r1` are yours to pick — use the same one
in every step.

```bash
# 1. Solve teacher roots (resume-safe: re-running skips finished roots)
.venv/Scripts/python scripts/cfr_batch.py --out-dir data/cfr/teacher_r1 --n-roots 40 --iters 20000 --size-preset micro --workers 4

# 2. Export labels + train (writes a held-out split and never trains on it)
.venv/Scripts/python scripts/train_policy_from_cfr.py --strategies data/cfr/teacher_r1/strategies --labels-out data/gto_nlh/teacher_r1.jsonl --ckpt checkpoints/gto_teacher_r1.pt --hidden-dim 512 --num-layers 2 --epochs 10

# 3. Probe the held-out roots and stamp the result into the checkpoint
.venv/Scripts/python scripts/gto_probe.py --ckpt checkpoints/gto_teacher_r1.pt --holdout data/gto_nlh/teacher_r1_holdout.jsonl --stamp

# 4. Serve it in the local UI (stop a running UI first)
PLO5BP_GTO_CHECKPOINT=checkpoints/gto_teacher_r1.pt .venv/Scripts/python -m uvicorn plo5bp.ui.server:app --port 8765
```

River / turn teacher roots solve far faster with full-range DCFR: add
`--algorithm dcfr_vector` to step 1 with a few hundred iterations instead of tens of
thousands (e.g. `--iters 300`: a standard river tree reaches < 0.05 bb in ~100). Flop
roots in the same grid stay on bucketed sampled DCFR. The reports are the usual ones
(same rows, verified exact exploitability) plus per-hand EV / equity.

What each step leaves behind:

| step | output |
|---|---|
| 1 | `data/cfr/teacher_r1/strategies/` (accepted: FINAL exploitability ≤ 1 bb), `rejected/` (over the cap), `unverified/` (number not a final estimate), `manifest.json` |
| 2 | `data/gto_nlh/teacher_r1.jsonl` (train labels), `teacher_r1_holdout.jsonl`, `teacher_r1_split.json`, `checkpoints/gto_teacher_r1.pt` |
| 3 | the probe result + `is_gto_validated` inside the checkpoint; `gto_badge_note` says why when the badge is refused |

## What the badge needs (checked when the checkpoint is loaded, not asserted by a script)

1. every training label came from the native solver (`source=rust_cfr*`) with a verified
   exploitability under the 1 bb teacher cap (step 1 + 2 enforce it;
   `--allow-unverified-expl` / `--no-expl-floor` produce a checkpoint that can never get it);
2. a probe PASS on held-out roots the net never trained on, over at least 50 rows
   (`scripts/gto_probe.py` defaults `--min-n 50`);
3. the holdout labels are verified teacher labels too.

Train and serve with the same observation revision (`PLO5BP_OBS_REV`; production runs 1).
The checkpoint is stamped with the revision it was trained at and the UI warns on a
mismatch.

## The other scripts

| script | use it for |
|---|---|
| `cfr_solve.py` | one root, to look at or to time |
| `cfr_verify.py` | an end-to-end correctness certificate for the solver (Kuhn + batch metrics into `data/cfr/verify/`) |
| `cfr_export_labels.py` | step 2's export alone (e.g. to inspect labels before training) |
| `gto_train_from_labels.py` | training from label files you already have, optionally mixed with rule-bootstrap rows or warm-started — a bootstrap mix is never badge-eligible |
| `gto_train.py` | the rule-based bootstrap curriculum (smoke tests, day-1 nets) — never badge-eligible |
| `cfr_overnight.py` | the full-hand preflop → postflop pipeline grid (research) |
| `cfr_app.py` | the desktop CFR Solver (one root at a time, with a strategy viewer) |

The 2026 step-6 / step-7 campaign drivers are archived in
`scripts/archive/gto_campaigns/`.
