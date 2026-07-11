# V7 design — living skeleton

*Started 2026-07-12, in parallel with the vSix2 cold start (the corrected
v6 recipe run end-to-end from update 0). v7 is the next architecture
generation; this document accumulates its workstreams, the incidents
motivating them, and the gates each must pass. Nothing here is committed
design until marked so.*

## Posture

- v6's substrate (LayerNorm+l2-init, HL-Gauss value head, prob-dependent
  clip with live rooms, pooled+anchored Q head, magnet-off cold starts)
  carries forward unless a workstream explicitly replaces a piece.
- Stems are disposable; recipes are the product. vSix2 is both a training
  run and the validation harness for everything v7 inherits.
- User-set shape (2026-07-10): if the sizing revisit shows capacity is
  the binder, v7 = bigger actor and/or critic, possibly with new features
  alongside. Oversized is acceptable; undersized silently caps skill.

## Workstream 1 — advantage & Q-head integrity (bug-class-proof design)

*Motivated by the July incident chain: Q-surface family offsets →
terminal-action boundary term paying a fold subsidy → post-fix
expectation-inflation transient consolidating fold-at-locks.*

1. **Hard-coded fold column**: `Q[FOLD] ≡ 0` by construction (identity,
   not estimate). Kills the fold-subsidy class permanently. Precondition
   understood: only safe when sibling columns don't carry a shared offset
   (cancellation asymmetry) — on a fresh v7 head with the anchor from
   birth, or after an audit-verified surface. Replaces the `qF` canary —
   needs a successor canary (see 3).
2. **Raw-space dueling base**: the current `Q = V.detach() + A` straddles
   value spaces (V = symexp of a symlog-space mean; targets = raw-space
   returns) — a Jensen-type gap that A absorbs as family offsets. Options:
   plain Q head (no V base), a learned raw-space baseline, or regressing
   Q in symlog space and symexp-ing at read (rejected once for VRPO's
   unit requirements — revisit with the boundary term in view).
3. **Boundary-term audit of the advantage estimator**: terminal actions
   (folds, showdown calls, successful steals) take `δ = r − Q(s,a)` with
   no next-state term to cancel estimator bias. Any Q bias becomes a
   terminal-action subsidy/tax. Candidate mitigations: hard-coded fold
   (kills the largest case), a terminal-residual canary (mean
   `r − Q(s,a)` over showdown rows — the successor to qF), bias-corrected
   targets.
4. **Q-head training data**: today each column learns only from
   taken-action rows. Consider expected-SARSA-style all-column targets or
   importance-weighted off-action learning — density was the original
   starvation cause.

## Workstream 2 — capacity & sizing

*Standing plan (memory: v7-network-sizing-revisit): clock starts after
~1 healthy week of vSix2.*

1. Utilization probe on a mature checkpoint (dead units, effective rank)
   — cheap, anytime.
2. 2× critic A/B on identical data (is v ≈ 3.0 a floor or a ceiling?).
3. Eval-gated width/depth sweep (1024/2048/4096 × 3/4/6 blocks) — needs
   WS3's scoreboard first.
4. If capacity binds: bigger actor and/or critic; new features ride the
   same redesign (obs v3 candidates TBD — collect during vSix2 reviews).

## Workstream 3 — eval & probe suite (promoted to first-class)

*The week's meta-lesson: the user's hand reviews out-diagnosed aggregate
metrics twice. Automate the eyeballs.*

Per-checkpoint automated suite (target: runs off-pod on promoted
checkpoints, wired into the promote flow):
1. **Lock-fold probe** — P(fold | locked board, facing bet) by tier; the
   direct never-fold-the-nuts curve. (Manual script exists:
   probe_lock_folds.py.)
2. **Q-calibration audit** — fold/terminal residuals, |E_π[Q]−V| by
   tier, column sanity. (Manual: q_head_audit_pooled.py.)
3. **Gate-sharpness panel** — fixed seeded node set; P(gate) trajectories
   across checkpoints (commitment without collapse).
4. Later (roadmap): duplicate-deal eval, AIVAT variance reduction, LBR
   exploitability probes — the real scoreboard for ship decisions and
   the WS2 sweep.

## Workstream 4 — carried process rules

- One variable at a time; pre-committed revert criteria; probes before
  interventions; the master doc (learn/MASTER.md) patched same-day as
  mechanics change.
- Live-tunable knobs preferred over restart-bound ones (anneal_control:
  lr, tier_ent, clip rooms, q_fold_sup_coef — extend as needed).
- Guardian discipline: kill guardian before trainer; anneal_control is a
  startup baseline — bake live values into flags before any restart;
  rewrite the whole control file on edit (stale keys re-apply).

## Incident ledger (what motivated all this)

| date | incident | root cause | v7 answer |
|---|---|---|---|
| 07-09 | 200-update cold-start freeze | magnet EMA tethered policy to its init | magnet late-stage only, short horizon (WS4) |
| 07-09 | VRPO advantages noise | 13-col Q starvation (~3%/col) | pooled head (shipped); WS1.4 |
| 07-11 | KL pinned vs 22× LR | clip mid-band = per-update quota | live clip rooms (shipped) |
| 07-11 | Q fold column −3/−16bb offsets | drowned anchor + symlog/raw gap | anchor 15 (shipped); WS1.2 |
| 07-12 | fold subsidy + lock-fold consolidation | terminal boundary term; expectation transient | WS1.1/1.3; lock probe (WS3.1) |

## Status

- 2026-07-12: skeleton created. vSix2 cold start running as the recipe
  validator. First deliverable: WS3.1+3.2 automation (probe cadence on
  vSix2 checkpoints), then WS2.1 utilization probe after a healthy week.
