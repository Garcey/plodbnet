# NLH next-version training design (nlh5)

PLO5 holds the pod; this is a **design-only** track so nlh5 is ready to launch the moment GPU frees. No code until design is locked.

## Where we are (nlh1 → nlh4)

| Stem | Entropy | Curriculum | Outcome |
|------|---------|------------|---------|
| nlh1 | 0.30 | `--stack-dist deep` | ~3 updates; replaced same night (wrong table culture) |
| nlh2 | 0.30 | `nlh_topoff` + `nlh_ring` | **Collapsed u21–25** into card-blind all-fold (`p_raise ~1e-8`). Not a code bug — NLH-specific trap: fold is the only zero-variance zero-return action under forward-EV (blinds sunk); cold overbet raises realize terrible EV; 4.5bb steal drowns in 100–400bb clash variance under global adv norm |
| nlh3 | **0.45**, `sizing_entropy_scale=0.65` | same | Warm from clean `nlh2_20`. Hg 0.75–0.92 through u121. **Correct prefs** (AA/72o ordering, min-click opens) but **soft frequencies** (62o UTG fold only ~38%) |
| nlh4 | **0.40** (~11% cut; known cliff is 0.30) | same | Mild sharpening. Local peak `nlh4_200.pt` → `nlh_stub.pt` (2026-07-05). Guardian still refuses cold-start at 0.40 |

**Hard constraints carried forward**
- NEVER cold-start at ≤0.40 (nlh2 collapse config).
- NEVER jump entropy to ≤0.35 in one step (guardian mandate).
- Next mild step if soft AND Hg stable ≥0.6: **~0.36**.
- Guardian Hg floor: 12/12 last updates Hg < 0.45 → stop-and-look (tighter than the 0.15 collapse trip).
- No cross-variant warm-start (PLO→NLH refused by design; equities differ).
- One family per pod; do not prune `vFour4_*.pt` when NLH returns.

**Architecture frozen on nlh1–4**
- Head: v4 logistic (`--sizing-head logistic`), 12-anchor `NLH_ANCHOR_SPEC` (min … 275% pot + ALL-IN atom).
- Net: 2048×4 residual actor + CentralCritic 1536×2.
- Obs: `OBS_DIM_NLH=995` (`encoding_nlh.py`) — history depth 40, log1p SPR/overbets, 3-dim opp-outcome, hole-class + SB/BB.
- Rollout: 11.6M / 16 mb / 2 epochs, lr 1.5e-4 warmup 75, target_kl 0.5, kl_hard 10, adv-clip 8.
- Live knobs already work for flat NLH: `runs/anneal_control.json` → `{"entropy_coef": X}`, `sizing_entropy_scale`, `target_kl`, `lr`.

## What "next version" can mean (decision axes)

Three axes are independent. Pick any combination; the recommended default is **A then B, architecture later**.

### Axis A — Hypers / anneal (lowest risk, pure continuation)

**nlh5 = warm from `nlh4_200` (or newest healthy nlh4), entropy 0.40 → 0.36**, keep everything else.

- Pre-launch gate (before GPU spend): probe `nlh4_200` on fixed nodes
  - Prefs still correct: AA-BTN raise ≫ 72o-UTG fold
  - Softness: 62o UTG fold rate, min-click open share, Hg band
  - If already sharp (Hg ~0.55–0.65 and frequencies look committed) → **skip anneal**, run longer at 0.40 instead
- If soft + Hg ≥ 0.6: step to **0.36**, hold 50–100 updates, re-probe
- Floor discipline: if Hg dips toward 0.50 sustained → revert via `anneal_control` to 0.38/0.40; do not chase
- Optional: raise `sizing_entropy_scale` slightly (0.65 → 0.75) so gate takes more of the entropy budget as we lower the global coef (scale <1 already tilts pressure onto gate via detached `p_raise` — re-read network math before changing)

### Axis B — NLH-native eval / probes (build while PLO trains — **no GPU**)

PLO's `scripts/probe_suite.py` is dual-board lock-fold / Q-calibration. NLH needs its own node bank or the next anneal is still vibes.

Minimum NLH probe bank (CPU, local or pod spare cores):

1. **Preflop identity panel** (the collapse surface)
   - AA/KK/QQ BTN open %; 72o/83o UTG fold %; AKs vs AKo BTN
   - SB complete / 3-bet vs BTN min-open (steal defense)
   - BB defend vs BTN 2.5x (or min) across 100bb
2. **Gate sharpness** — median max-gate-prob + mean Hg by street (preflop vs river)
3. **Sizing sanity** — open size histogram (min-click vs pot vs jam); overbet share on polar river spots
4. **All-in atom reachability** — short-stack shove nodes land on last anchor, not clipped interior
5. **Collapse canaries** — p_raise floor on any legal-raise node; card-blind fold rate (same action across AA vs 72o)

Wire as `scripts/probe_suite_nlh.py` (or a `--variant nlh_single` mode on the existing suite) writing `runs/probe_history_nlh.jsonl`. **Mandatory before promote** (same rule as PLO V7).

Ranges tab already exists for qualitative review — use it to eyeball 169-grids after each probe line; do not replace the automated bank.

### Axis C — Architecture / objective upgrades (higher risk; only after A+B baseline)

Transfer **selectively** from PLO v5–v7. Several PLO wins are bomb-pot-specific and can **hurt** NLH.

| Candidate | Fit for NLH | Notes |
|-----------|-------------|--------|
| Mild entropy anneal (live `anneal_control`) | **Yes — primary** | Already the guardian plan |
| `sizing_entropy_scale` gate tilt | **Yes** | Proven on nlh3/4 |
| KL-anchor magnet (`--kl-anchor-coef`, EMA) | **Careful, late only** | V7 lesson: cold magnet freezes at init. Only after Hg stable and prefs locked; short horizon. V6 notes NashPG family is 2p0s — **HU-NLH A/B is the honest testbed**, multiway ring is not |
| v5 mixture sizing head | **Maybe later** | NL already has geometric overbet ladder + all-in atom; mixture helps multi-modal sizes. Cold-layout change → converter or cold start; do **not** pair with first anneal step |
| Q-aux / VRPO / q_fold_zero | **Maybe** | Fold column is load-bearing in NLH (the collapse mode). If Q-aux lands, fold-sup must be correct day-1 or we re-buy the PLO lock-fold pathology |
| Obs append (169-class, more preflop) | **Deferred** | Hole-class block already present; 169 one-hot was explicitly deferred. Cold layout = new stem family, not warm nlh5 |
| Tsallis / sparsemax gate | **No for nlh5** | PPO hazard (exact zeros → NaN ratios). Research only |
| PFSP / pool mix bias to current policy | **Yes, cheap** | V6 finding: current-policy self-play beats stale pool under fixed budget. Bias snapshot mix toward recent members without architecture change |
| Per-street / preflop-boosted entropy | **Interesting** | Collapse is preflop-local; a higher ent weight on preflop rows (or lower on river) could sharpen postflop without re-opening the fold trap. Needs `Batch.ent_coef_rows` plumbing (PLO mix-configs already has per-row ent) — design spike, not day-1 |
| Advantage normalization | **Investigate** | nlh2 postmortem cites global adv norm drowning the 4.5bb steal. Options: per-street adv scale, pot-relative adv, or clip more aggressively on deep multiway. Measure first via probe, don't guess |

**Explicit non-goals for nlh5**
- OCR / PokerNow for NLH
- Cross-variant warm-start from PLO
- Obs schema change on the first launch
- Jumping entropy ≤0.35
- Shipping v5 mixture + anneal + magnet in one stem (one variable at a time)

## Recommended default plan

```
Phase 0  (now, CPU, while PLO runs)
  - Design freeze on this doc + user decisions below
  - Build NLH probe suite (Axis B) + run it on nlh4_10/50/85/200
  - Write baseline series into runs/probe_history_nlh.jsonl
  - Optional: qualitative Ranges-tab review of nlh_stub

Phase 1  (first GPU window = nlh5)
  - Warm nlh4_200 (or best probe winner)
  - entropy 0.36, sizing_entropy_scale 0.65 (hold), same rollout/guards
  - Guardian: copy nlh_guardian.sh → STEM=nlh5, warm from nlh4_*, floor detectors unchanged
  - Live anneal_control only; bake values into guardian flags before any restart
  - Probe every ~25–50 updates; promote only with a green probe line

Phase 2  (only if Phase 1 plateaus healthy but soft)
  - Second mild cut 0.36 → 0.33 (still above the 0.30 cliff; one step)
  - OR raise sizing_entropy_scale 0.65 → 0.80 at fixed ent (pick ONE)

Phase 3  (separate stem if architecture wanted)
  - nlhMix1 or nlhV5: mixture head and/or obs append and/or Q-aux
  - Fresh cold start at ent ≥ 0.45, never warm across head flip without a converter
```

## Open decisions (need your call)

1. **Scope of "next version"** — pure anneal continuation (nlh5 @ 0.36), or do you also want architecture (mixture / Q-aux / obs) in the same design doc as a later phase?
2. **Probe suite first?** — recommend yes (CPU work now). Confirm or skip.
3. **Curriculum hold** — keep `nlh_topoff` + `nlh_ring` as-is, or bias more HU/short for sharper steal learning?
4. **Pool policy** — keep current snapshot pool, or bias toward recent/current policy (PFSP-lite)?
5. **Success criteria for promote** — propose: prefs correct + 72o UTG fold ≥ ~70% + AA BTN open ≥ ~80% + Hg in 0.55–0.80 + no collapse canary. Tune numbers after Phase 0 baseline.

## Files that will change when we implement (not now)

- `scripts/nlh_guardian.sh` → `nlh5` stem + warm path from nlh4
- New or extended: `scripts/probe_suite_nlh.py` + `tests/python/test_probe_suite_nlh.py`
- Possibly `runs/anneal_control.json` live edits only (no code)
- Later phases only: `network.py` / converter / `encoding_nlh.py` / train flags

## What I will not do without explicit go

- Touch the live PLO pod run
- Launch NLH training (no GPU contention)
- Change obs dim, head family, or reward
- Prune any `nlh*` or `vFour4*` checkpoints
