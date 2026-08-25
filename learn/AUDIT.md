# learn/ course audit & repair log

*Repaired 2026-07-23 against live `python/plo5bp` + `scripts/*_guardian.sh`.*

## Verdict (pre-repair → post-repair)

| axis | before | after |
|---|---|---|
| Structure | A | A |
| Mechanisms | A− | A |
| Numbers / current-run | C | A− |
| Path to "clipped surrogate loss" | C+ | A |
| Completeness | B− | A− |

## Critical facts corrected

| claim (old) | live truth | where fixed |
|---|---|---|
| `OBS_DIM = 1020` | **1171** (v7 batch-2 tail after 1019) | MASTER Part 2, ch 01/02/20, concepts, glossary, README |
| (missing) minimal mode | **OBS_DIM_MINIMAL = 796**, stem vMin1 | MASTER Part 2, ch 02, glossary |
| critic input 1020+260=1280 | **1171+260=1431** | MASTER 4.3, centralized-critic.md |
| actor params ~14.76M / total ~21.5M | actor **15,065,119**; joint **22,069,333** | MASTER 4.1, ch 20 |
| first linear 2048×1020 | **2048×1171** (2,398,208 w) | linear-layers.md, ch 20 |
| legal mask "14-action" | **NUM_ACTIONS = 8** legacy discrete; training gate is 3-way | MASTER Part 1 |
| LR warmup "currently 10" | **vSix4/vMin1: `--lr-warmup-updates 0`** | MASTER 6.10, learning-rate-warmup.md |
| ent / agenda "0.20 now" | live **ent=0.25** flat, anneal not armed | MASTER Part 10, entropy.md |
| stamp 2026-07-10 vSix1 | **2026-07-23 vSix4 + vMin1** | MASTER header, README |

## P0 concept pack added

| file | role |
|---|---|
| `concepts/clipped-surrogate-loss.md` | the phrase, formula, why surrogate, project hooks |
| `concepts/log-probability.md` | receipts, joint log-prob, bit-exact duty |
| `concepts/advantage.md` | A = Q−V, GAE vs VRPO, normalization |
| `concepts/value-function.md` | V/Q/display heads, HL-Gauss, fold identity |
| `concepts/minibatch-epochs.md` | 2×16 loop, KL guards |
| `concepts/reward-accounting.md` | commit_delta, gross terminal, fold≡0 |
| `concepts/game-shape.md` | PLO5 DB bomb, pot-limit, multi-config |

## MASTER structural adds

- Part **6.0** teaching spine (six-beat story → formula → what else is in the loss)
- Part **6.0b** log-line literacy table
- Part **6.2** expanded with the exact `surr1/surr2/policy_loss` snippet
- Part **2** v7 tail rows + obs-mode section
- Part **10** live-stem blurb + entropy-walk agenda

## Still intentionally historical

- Generational lineage (v1→v5 "obs → 1020") keeps the date-stamped story;
  the 1171 append is noted inline on the v5/v6 bullet.
- vSix1 investigation narrative in Part 10 is provenance, not live config.

## How to verify

```text
.venv/Scripts/python -c "from plo5bp.encoding import OBS_DIM, OBS_DIM_MINIMAL; \
 from plo5bp.actions import NUM_ACTIONS; print(OBS_DIM, OBS_DIM_MINIMAL, NUM_ACTIONS)"
# → 1171 796 8
```

Reading path for the target jargon: README → MASTER 6.0 →
`concepts/clipped-surrogate-loss.md` (+ linked cards).
