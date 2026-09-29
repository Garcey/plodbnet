# NLH GTO — Complete Implementation Before Confident Training

**Status:** Mode-0 **plumbing is built**; **confident GTO training** waits on teacher-quality **native rust_cfr** labels + holdout probe — not on missing Trainer wiring.

**Teacher:** native CFR only (`rust_engine/src/cfr/`, `source=rust_cfr*`). No external solver.

**Constraint (locked):** PLO5 runpod stays running until the full NLH implementation is done. All NLH/CFR work is **local CPU** unless you later free GPU.

---

## Verdict

| Question | Answer |
|----------|--------|
| Is the train→serve code path present? | **Yes** (`cfr_solve` / `cfr_batch` → JSONL → PolicyNet → `PLO5BP_GTO_CHECKPOINT`) |
| Is it safe to start “real” GTO training now? | **Not at product scale** |
| Why? | Postflop smoke solves still have high exploitability; overnight full-hand shard is empty; train obs still reconstructs empty history |

---

## What is already done (do not rebuild)

- Frozen decisions (Trainer-first, ClubGG roots, multiway T1 play, PPO frozen, no live freqs v1)
- `StrategyBackend` / `PpoSolverHost` / `PolicyNetHost` + Trainer act/score/badge
- Study + Ranges host when `PLO5BP_GTO_CHECKPOINT` set
- ClubGG roots, label schema, native CFR core, batch/export/train/probe CLIs
- Rule bootstrap curriculum (honest as **prior**, not GTO)
- First real teacher artifact: 4-handed 10bb push/fold, 300k iters

---

## Definition: “Ready to train confidently”

All of the following:

1. **rust_cfr strategy → LabelRecord → SupervisedRow** correctness (unit tests on live/export JSON).
2. **Held-out probe gate** vs **rust_cfr** labels (not self-KL on train): pure-node agree + gate KL thresholds documented.
3. **Minimum data scale:** ≥200 river roots (SPR grid), reach-weighted or full in-range combos, train/holdout split, teacher expl gated.
4. **Obs fidelity:** rows from real engine node **or** synthetic obs that matches live encoding on pot/to_call/stacks/history minimum.
5. **Badge policy:** `"GTO AI"` only if ckpt meta has `source=rust_cfr*` **and** probe gate passed; bootstrap = different label.
6. **Ship script:** one documented command sequence (batch → export → train → probe → promote).

Mode 2 ValueNet + street re-solve are **not** required for this bar (HU postflop Mode 0 only).

---

## Implementation plan (ordered)

### Workstream A — Label pipeline (done)

`cfr_export.py` / `obs_from_label.py` / `labels.py` + `test_cfr_export_train.py`:

- Path reconstruction of pot / to_call / stacks
- Solve-ladder RAISE_pm / ALLIN → `NLH_ANCHOR_SPEC`
- Per-infoset rows use that combo/class strategy vector
- Fold legal iff `to_call > 0`

### Workstream B — Quality gates + honesty (done, retargeted)

`probe.py` / `policy_net.py` / `policy_host.py` / `scripts/gto_probe.py`:

- `ProbeGates` + `evaluate_probe_gates` (pure_agree ≥ 0.90, mean_gate_kl ≤ 0.50 defaults)
- `scripts/gto_probe.py` exit 0/1; optional `--stamp` writes `probe` + `is_gto_validated`
- Badge **"GTO AI"** only if `source` starts with `rust_cfr` **and** probe.passed
- Bootstrap → **"Curriculum"**; rust_cfr without probe → **"Policy net (unvalidated)"**

### Workstream C — Scale data + train loop (next)

**No PLO pod.** Local CPU overnight.

1. `cfr_batch.py` / `cfr_overnight.py` — HU river SPR grid + longer iters; gate expl.
2. Holdout 10–20% roots by seed.
3. `cfr_export_labels.py` then `gto_train_from_labels.py` / `train_policy_from_cfr.py`.
4. Probe holdout; promote only on pass → `checkpoints/gto_policy.pt`.

### Workstream D — Later (explicitly after confident Mode 0)

| Phase | Work |
|-------|------|
| Preflop | Blueprint + induced postflop (pipeline already exists) |
| Mode 1 | River exact for trainer Exam |
| C2 | ValueNet + street re-solve T3 |
| Multiway GTO | Honesty only until labels/search exist |

---

## Execution order

```
A      rust_cfr export → labels → obs          [done]
B      Probe gates + rust_cfr badge honesty    [done]
C      Scale batch → train → probe → promote   [ops, local]
--- TRAIN CONFIDENTLY (HU river Mode 0) ---
D      Preflop density / Mode1 / C2            [later]
```

---

## Success criteria before you start the “real” training run

- [x] Export tests prove to_call, size→anchor, per-combo/class rows
- [x] `pytest tests/python/test_gto_*.py tests/python/test_cfr_*.py` green
- [ ] ≥200 river roots labeled at teacher-quality expl
- [x] Holdout probe script exists and documents pass/fail
- [x] Badge cannot claim GTO on bootstrap-only ckpt
- [x] Badge cannot claim GTO unless `source=rust_cfr*`
- [ ] One README/plan section: exact commands for batch → train → probe → serve

---

## Out of scope until Mode 0 is validated

- Pausing PLO5
- Full-hand multiway Nash claims
- Live spoil freqs (decision #6 frozen off)

---

## Immediate next step

Workstream C: HU river SPR grid at teacher-quality exploitability → export → train PolicyNet → probe.
