# Plan: v4 sizing head — ordinal discretized-logistic over the existing anchors

## Objective
Replace the flat 11-way anchor **categorical** (the diagnosed root cause of every
vThree collapse) with an **ordinal discretized-logistic** parameterized by a single
location `μ` and scale `s`, laid *over the same 11 discrete anchors*. This makes the
size distribution smooth/low-dimensional (→ stable under PPO, no cross-stack-depth
gradient war) while **keeping the discrete anchors so exact-pot and exact-min stay
first-class, concentratable actions** (the property v2 was built for; a pure continuum
would regress on it). With the sizing head stable, the gate can finally be annealed to
decisive (~0.2) play instead of 60/40 mush.

Derived from a 4-agent design review (continuous-RL / ordinal / poker / optimization
angles) + the endpoint-hittability constraint. 3 of 4 agents independently converged on
location+scale; the discretized-over-anchors form (vs pure continuous) is required by the
exact-pot/min constraint.

## Decisions baked in (with rationale)
1. **Discretized logistic over the anchor INDEX axis** (0..10), NOT a continuum and NOT
   per-anchor free logits. `μ` slides along the ordered index; each anchor's probability
   is the slice of the logistic CDF over `[k-0.5, k+0.5]`.
2. **Tail-absorbing endpoints** — anchor 0 (min) gets `(-∞, 0.5]`, anchor 10 (pot) gets
   `[9.5, +∞)`. So min/pot soak up the whole tail and are reliably hittable + can carry
   most of the mass (set `μ` past the end + small `s`). **This is the load-bearing detail
   for the exact-pot/min requirement — do not drop it.**
3. **`μ` allowed beyond the endpoints** so an endpoint can dominate: `μ = 5 + 7·tanh(μ_raw)`
   → `μ ∈ [-2, 12]`.
4. **Scale floor** `s = softplus(s_raw) + s_min`, `s_min ≈ 0.3` index-units (≈⅓ of an
   anchor gap) so the distribution can never become a one-hot spike — the hard anti-collapse
   guarantee. Optional soft cap `s ≤ s_max ≈ 5`.
5. **Keep the per-anchor Beta "refine" (Option B — surgical).** The Beta is NOT the primary
   instability (collapse logs show klA, the anchor categorical, spiking — klB stayed small).
   Keeping it means the change is contained to `network.py` with zero ripple into the Batch
   schema / `ppo.py` / `rollout.py` / UI. (Dropping the Beta = Phase 2 if it later proves
   worth the ripple.)
6. **Phase-2 enhancements deferred** (documented below, not in v1): anchored location,
   explicit SPR conditioning, K=2 mixture, drop-Beta, log/pot-fraction spacing.

## The math (one helper does it all)
```python
# mu: (...,) location on the index axis;  s: (...,) scale >= s_min
# legal: (..., 11) bool mask from anchor_grid_torch(sizing).legal
def anchor_probs_from_logistic(mu, s, legal, K=ANCHOR_COUNT):
    idx = torch.arange(K, device=mu.device, dtype=mu.dtype)        # 0..10
    z_hi = (idx + 0.5 - mu[..., None]) / s[..., None]              # (...,11)
    z_lo = (idx - 0.5 - mu[..., None]) / s[..., None]
    cdf_hi, cdf_lo = torch.sigmoid(z_hi), torch.sigmoid(z_lo)      # logistic CDF
    p = cdf_hi - cdf_lo
    p[..., 0]    = cdf_hi[..., 0]            # min anchor absorbs left tail  (cdf_lo=0)
    p[..., K-1]  = 1.0 - cdf_lo[..., K-1]    # pot anchor absorbs right tail (cdf_hi=1)
    p = p * legal                            # mask illegal anchors
    return p / p.sum(-1, keepdim=True).clamp_min(1e-8)             # renormalize over legal
```
- For numerical safety compute the bin mass in **log space** via `logsigmoid` and a stable
  log-difference (`log(σ(b)-σ(a))`), then mask/renormalize in log space; the snippet above
  is the readable form.
- Sampling/log-prob/entropy then treat `p` exactly as today's categorical does — **so
  everything downstream of "the 11 anchor probabilities" is unchanged.**

## Code changes (file by file)

### `python/plo5bp/network.py` — the only substantive change
- `ActorCriticV2.__init__`: replace `self.anchor_head = nn.Linear(hidden_dim, ANCHOR_COUNT)`
  with `self.size_head = nn.Linear(hidden_dim, 2)` (outputs `μ_raw, s_raw`). Keep
  `gate_head`, `refine_head`, `value_head` unchanged.
- `forward`: emit `(gate_logits, mu_raw, s_raw, refine, value)`. (Optional torso-grad
  isolation: `z_size = z.detach() + λ·(z - z.detach())` feeding `size_head`, `λ` default 1.0
  = no-op; flag wired but inert unless dialed.)
- Add `anchor_probs_from_logistic(...)` (above) + `μ = 5 + 7·tanh(mu_raw)`,
  `s = softplus(s_raw) + s_min`.
- `act` / `evaluate`: replace `Categorical(logits=anchor_logits_m)` with
  `Categorical(probs=anchor_probs_from_logistic(μ, s, grid.legal))`. **Anchor index is still
  sampled, stored, gathered, mapped→chips, and refined-by-Beta exactly as today.**
- Return tuple shape is **unchanged** (`..., gate_entropy, anchor_entropy, beta_h_eff,
  gate_log_prob, anchor_log_prob`); `anchor_entropy` is now the discretized-logistic
  categorical entropy. The `entropy = gate_h + p_raise.detach()·(anchor_h + beta_h_eff)`
  line is unchanged → **`ppo.py` needs no changes** and `--sizing-entropy-scale` still works.
- Head-sniffing: `_load_model` / `head class` detection keys on `anchor_head.weight`. Add a
  `size_head.weight` → v4 branch (see UI).

### `python/plo5bp/sizing.py` — UNCHANGED
`anchor_grid_torch` (per-anchor chips, `legal`, refine brackets) is reused verbatim. Min
anchor = min-raise chips; pot anchor = `max_raise`/pot chips → exact-pot/min preserved.

### `python/plo5bp/ppo.py` — UNCHANGED
Consumes the same `(log_prob, entropy, gate_h, anchor_h, beta_h, ...)` interface; per-head
KL decomposition, `target_kl` guard, `sizing_entropy_scale` all keep working.

### `python/plo5bp/rollout.py` — UNCHANGED
`Batch` still stores `anchor_actions`, `refine_u`, `old_anchor_logp` (the anchor log-prob is
just computed from the new distribution). No schema change → `collect_rollout` /
`collect_rollout_batched` untouched.

### `python/plo5bp/ui/server.py` — minimal
`_load_model` head-sniff: add `size_head.weight → v4`. The recommendation still exposes an
11-way `anchors` histogram (now the discretized-logistic probs — same shape) + `rec_anchor`
(argmax) + `refine`. `score_move_v2` snaps to nearest legal anchor (unchanged). The UI's
existing v2 rendering path works as-is on the 11-vector.

### CLI / config / stem
- New stem family (e.g. `vFour`). New head ⇒ **checkpoint break** (`anchor_head` → `size_head`).
- Optional flags: `--sizing-scale-floor` (default 0.3), `--sizing-torso-grad-scale`
  (default 1.0). `--sizing-entropy-scale` already exists and applies unchanged.

## Tests (`tests/python/`)
New `test_sizing_logistic.py`:
1. **Sum/shape** — probs sum to 1 over legal anchors; shape (B,11).
2. **CDF correctness** — `anchor_probs_from_logistic` matches a hand-computed reference.
3. **Endpoint tail-absorption (the user's requirement)** — `μ` high ⇒ `P(pot)` → ~1;
   `μ` low ⇒ `P(min)` → ~1; with small `s`, an endpoint carries the large majority of mass.
4. **Legality** — illegal anchors get exactly 0, mass renormalized to legal set; the
   short-shove single-legal-anchor case → that anchor prob 1, entropy 0.
5. **Scale floor** — entropy is bounded below by a positive constant (no one-hot collapse).
6. **KL smoothness (the stability property)** — a small `Δμ` produces a small KL between
   old/new anchor distributions; moving the preferred size by one anchor is a small-KL move
   (contrast: a flat-categorical logit swing is not). Mirror the gradient-scaling test style
   already in `test_ppo_v2.py`.
7. **act/evaluate parity** — replaying a stored anchor gives matching log-prob (the
   1.0-ratio-at-epoch-start contract).
Extend `test_ppo_v2.py`: v4 `update()` produces finite stats incl. the entropy decomposition.
Update any tests that assert the old `anchor_head`/flat-logit shape.

## Validation / rollout sequence
1. Implement + all unit tests green; full `pytest tests/python/` green.
2. **Smoke** — short cold-start run (small net, ~30 updates): confirm it runs, the anchor KL
   (klA) stays bounded (no spike), entropy doesn't collapse.
3. **Stability run** — fuller cold-start, watch the u16–30 zone that killed every prior run:
   - klA bounded, no KLSTOP cascade, Ha holds (does NOT crash to 0).
   - Hg can be annealed *down* toward ~0.2 (decisive gate) without dragging.
   - sizing histogram shows real **variety incl. exact pot and exact min** (not pinned ~half-pot).
4. **Browse on UI** — does it play decisively and size sensibly across spots?
5. Iterate; add Phase-2 enhancements only if a specific failure shows.

## Checkpoint / warm-start
New head ⇒ can't load `anchor_head`. Two options:
- **Cold start (recommended for the honest architecture test)** — fresh everything; cleanest
  read on whether the new head trains stably + decisively, no inherited mushiness. Slower.
- **Partial warm-start** — load torso/gate/value/refine from a clean checkpoint (e.g.
  `vThree_10_clean`/`vTwo10_445`) with `strict=False`, re-init `size_head`. Faster, but
  inherits the mushy gate/torso; the new head must pull them out of it.
Recommendation: smoke + a short cold-start to validate stability, then decide cold-vs-warm
for the long run based on what the smoke shows.

## Phase-2 enhancements (deferred; add only if a specific need appears)
- **Anchored location** (agent-1): `μ = μ_ref(state) + bounded Δ`, where `μ_ref` is a
  parameter-free context size (e.g. SPR-appropriate anchor) → kills any residual center-bias
  and pre-aligns cross-depth gradients. Add if sizing shows a half-pot pull.
- **Explicit SPR conditioning** (agent-3): feed `log(SPR)` / FiLM into `size_head` so `μ`
  tracks stack depth by construction. Add if cross-depth resolution is still hard.
- **K=2 mixture of logistics** (all agents): for genuinely polarized (small-or-jam) spots.
  Add if review shows the single mode stuck between two sizes.
- **Drop the Beta refine** (agents 3/4): 11 exact sizes only; removes a second instability
  axis. Ripples into Batch/ppo/UI — do as a clean follow-up if the Beta misbehaves.
- **Finer grid / log spacing** (agents 1/4): more resolution or geometry-matched spacing.

## Risks + mitigations
- **Unimodality** (K=1 can't be bimodal at a node) → K=2 escape hatch (Phase 2); poker
  evidence says rarely needed.
- **Checkpoint break** → cold start or partial warm-start (above).
- **Numerical tails** (`σ→0/1`) → log-space computation + clamp `z` to e.g. `[-12, 12]`.
- **`μ` saturating at the rails** when pot/min is almost always right → tail absorption means
  `μ` need not reach the rail to put most mass on the corner; `s_min` keeps the corner < 1.
- **Parity bugs at edges/masking** → pinned parity test (#7) before any full-speed run.
- **Endpoint concentration vs smoothness** — by design it can't put literal 100% on pot
  (always a sliver on the 90% anchor); strategically fine (healthy mixing); lower `s_min` to
  approach a near-pure-pot strategy when truly optimal.

## Decisions — LOCKED (user, 2026-06-23)
1. **Keep the Beta refine** (Option B, surgical). One-file change in `network.py`.
2. **COLD START** — the current model hasn't converged beyond "slightly less random than
   random," so there is nothing worth warm-starting; train V4 fresh.
3. **Keep the 30-config mix-configs regime** (`--mix-configs`, 10 configs/tier × 3 tiers).
   The ordinal head is precisely what makes mixing stack depths stable for the sizing
   gradient, so head + mix-configs are complementary, not either/or.
4. **Name: V4** — new stem `vFour` (log `runs/vFour.log`, ckpts `vFour_N.pt`). New head
   architecture ⇒ `head_version = 3` internally (sniff key `size_head.weight`); the *run*
   is "V4". v1 (`raise_head`), v2 (`anchor_head`), v4-head (`size_head`) all loadable by the UI.

Implementation consequence of (2)+(3): the validation/stability run is a **cold-start
`--mix-configs` run** (the exact regime that collapsed under the old flat head) — the
cleanest possible test that the new head fixes it.
