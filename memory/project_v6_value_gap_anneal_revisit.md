---
name: v6 actor/critic value gaps — revisit at ent≈0.10
description: User study-tool observations on negative V, own vs true EV gaps, soft mixes on trash; revisit when vSix4 entropy ~0.10
type: project
originSessionId: e3088dd7-0e2d-4a0f-8fb5-4ba33be69140
date: 2026-07-20
---

# v6 value-display disappointment — revisit at entropy ≈ 0.10

## Trigger
When **vSix4 (or successor) entropy is around 0.10**, re-open this conversation with the user. Do **not** wait to be reminded — if they ask about training health, promote, study quality, or v7, surface this note and compare current spots/metrics to the baseline below.

At save time (2026-07-20): vSix4 soaking at **ent=0.16** (cut after u1144; first live u1145). Prior floors: 0.25→0.22→0.20→0.18→0.16. Magnet still off. Public/UI promoted through vSix4_1240-era weights.

## User stance
- Still somewhat **disappointed with v6** as a study product, even while agreeing:
  - anneal is **not finished**
  - model already **plays very strongly**
- Wants colder entropy (~0.10) before judging whether these issues are “still soft π” vs structural/v7-worthy.

## Concrete spots (trainer review screenshots, ~2026-07-20)

### Spot A — weak multiway BB (negative V complaint)
- **Seat:** BB, flop decision after SB check. Node “BB acted Check”; Actual=Check, Network=Check.
- **Setup:** 6-max bomb pot, stacks ~$340, pot ~$360. Boards A♦7♣8♠ / 8♥5♦Q♣.
- **BB hand:** J♥ T♣ 9♣ 6♥ 5♥ — user: combo draw junk (gutshot to non-nut + bad fd on one board; underpair on other). Dominated fd (two players with better hearts possible) and dominated gutshot (JT out).
- **Policy:** Check ~74%, Raise ~26%, Fold 0%. GTO score high (~96%) because preferred action matched.
- **Values shown:**
  - **value(own) ≈ −$64** (actor blind display head) — user finds this absurd
  - **true ≈ −$4.46** (central critic, all cards) — user: almost noise vs pot, or ambiguous
- **User claim:** A hand that loses money continuing should **check-fold** and have **V≈0**. CFR/NLH solvers don’t show hands with large negative EV when fold/check-down floor is 0. Gap own vs true this late in training is surprising.

### Spot B — strong BTN (own overvalues, critic calmer)
- **Seat:** BTN vs HJ $20 open; CO folded. Actual=Call, Network=Raise ~2bb ($40).
- **Boards:** 7♠4♦Q♣ / T♥J♠6♦. **BTN:** J♠ T♦ 9♠ 9♥ 4♣ — user: best/near-best on top (two pair), big wrap bottom; strong even vs BB set of tens if UTG folds queens denied, etc.
- **Policy:** ~Call 49% / Raise 50% / Fold 2% — soft between continue lines.
- **Values:**
  - **value(own) ≈ +$140**
  - **true ≈ +$18.80**
- **User concern:** Huge actor–critic gap again. Also: residual entropy should *help* strong hands (opponents randomly stack off light), so critic +$19 may still be low; actor +$140 feels wild.

## What the numbers actually are (agent analysis — keep when revisiting)
| UI label | Source | Sees opp holes? | Training weight |
|---|---|---|---|
| value(own) | actor `value_head` | No | aux `display_value_coef=0.125`; log `vd` still huge (~600–1000) |
| true | `CentralCritic` V | Yes | main value / VRPO path |

- Both are **\(V^\pi\)** under **current soft policy**, not equilibrium pure-strategy EV and not Q(best action).
- Fold forward return is **exactly 0** under project reward (sunk excluded) — see `project_ev_semantics.md` + fold-sup / qF canary.
- Soft mix (e.g. 26% raise on trash) ⇒ \(V^\pi\) can be **negative** even though fold=0. That is policy value, not a proof fold costs money.
- Own-value is the **least reliable** of {policy bars, true, own}. Product guidance at mid-anneal: **trust true + action mix; discount own.**

Related code: `python/plo5bp/ui/trainer.py` (`value_bb` vs `value_true_bb`), `ppo.py` qF/q_fold_sup, display head loss.

## User’s product/quality worries to re-check at ent≈0.10
1. **Soft raises on obvious multiway trash** (26% raise BB junk) — does mix concentrate onto check/fold?
2. **Own-value magnitude** still absurd (tens of $ off) vs pot — display head still broken for UI?
3. **|own − true| gaps** still huge on both weak and strong hands?
4. **True V on pure trash** — closer to **0** (check-fold lines) or still clearly negative?
5. **True V on monsters** — still modest vs own’s huge positives; any change with colder π?
6. Still disappointed enough to prioritize **v7**, or anneal fixed the study feel?

## v7 ideas already floated in same arc (do not auto-implement)
- Smaller actor/critic stem (utilization probe: low effective rank, lots of dead ReLUs — bigger not clearly needed; smaller may help speed/VRAM/rollout and maybe generalization).
- Hide or relabel own-value; or train display harder; or show Q(check)/Q(fold)≈0 in UI.
- Finish anneal on v6 before judging; magnet still off until more concentration needed.
- Network size analysis + dead-weight study done on vSix4_1215 (actor ~15M 2048×4, critic ~7M 1536×2; final torso PCA ~2% dims for 95% var).

## How to revisit (checklist)
When ent≈0.10 (or user says “revisit value gaps”):
1. Confirm live `ent=` and Hg/Ha vs this note’s 0.16 / Hg~0.67–0.68 baseline.
2. Re-find similar spots in trainer (weak multiway BB OOP; strong BTN facing small open) on **current** promoted ckpt.
3. Compare: raise% on trash, own vs true magnitudes, whether true trash ≈0.
4. Recall this file + `project_ev_semantics.md` + `feedback_negative_ev_recommendations.md`.
5. Only then advise: keep annealing / UI-only fix / v7 stem.
