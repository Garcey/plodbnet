# The plo5dbbp training system — master document

*The complete technical description of how this project trains the PLO5
double-board bomb-pot model: engine to encoder to networks to optimizer to
ops. Written dense and unsimplified — this is the reference spine of the
course. Wherever a concept deserves a gentler treatment, it links into the
[plain-English concept library](concepts/) like this: [→ ReLU](concepts/relu.md).
If a whole part feels steep, chapters 1–3 of the intro track cover Parts
0–3's territory at conversational pace.*

*Everything here is verified against the code as of 2026-07-23 (live
stems **vSix4** full-obs and **vMin1** minimal-obs; post Q-head
calibration, post v7 batch-2 obs tail). File references name the
function rather than the line where practical; line numbers drift,
functions don't.*

**Contents** — Part 0: invariants · 1: engine & environment · 2:
observation encoding · 3: action space & sizing · 4: networks · 5:
collection & advantage estimation · 6: the PPO update · 7: self-play &
the opponent pool · 8: multi-config training · 9: ops & lifecycle ·
10: provenance & current agenda.

**Reading path to the phrase "clipped surrogate loss":**
[→ game shape](concepts/game-shape.md) ·
[→ reward accounting](concepts/reward-accounting.md) ·
[→ log-probability](concepts/log-probability.md) ·
[→ advantage](concepts/advantage.md) ·
[→ value functions](concepts/value-function.md) ·
[→ minibatches & epochs](concepts/minibatch-epochs.md) ·
[→ importance ratios & clip](concepts/ppo-clip.md) ·
[→ clipped surrogate loss](concepts/clipped-surrogate-loss.md) ·
then Part 6.0–6.2 below.

---

## Part 0 — System shape and invariants

One training **update** = one lap of *collect* (play ~9×10⁶ decisions
across 49,134 simultaneous environments under 30 sampled table configs)
then *learn* (16 minibatches × 2 epochs = 32 optimizer steps over that
batch). Wall-clock ≈ 11–13 min on the production pod (H100-class, 96GB).
Everything below is the anatomy of that lap.

Three invariants govern the whole codebase; violating any of them is
treated as a correctness bug, not a style issue:

1. **Bit-exact reproducibility of the data path.** The observation
   encoder, the sizing math, and the serial environment are pinned
   byte-identical by tests (`tests/python/test_encoding*.py`,
   `test_anchor_grid.py`, env parity suites). Rationale: PPO compares the
   updated policy against *stored* action log-probabilities; if
   `act()`-time math and `evaluate()`-time math disagree by one ULP of
   rounding, every importance ratio is silently corrupted.
   [→ seeds & determinism](concepts/seeds-determinism.md)
2. **Deterministic seeding.** A hand is a pure function of
   `(GameConfig, seed, button)`; terminal-equity Monte-Carlo uses
   `reset_seed XOR 0x9E3779B97F4A7C15`; the k=4 opponent-outcome MC seeds
   from observation-visible state only. Replay (UI, trainer review, tests)
   depends on this.
3. **Units.** Engine chips are integers with `cfg.bb = 10_000` chips per
   big blind; all rewards and value targets are **bb floats**; the UI's
   dollar layer converts at the boundary. Nothing in training ever sees a
   dollar.

---

## Part 1 — Engine and environment layer

**Rust engine** (`rust_engine/src/`, PyO3 module `plo5bp._engine`). Owns:
dealing, the full betting state machine, legality, showdown evaluation,
and the heavy feature math. The `Variant` enum (`state.rs`) gates hole
count (5), board count (2), the pot-limit cap, and the absence of a
preflop round; `plo5_double_bomb` is the default variant and byte-stable
against the pre-variant engine.

Game shape: **double-board bomb pot**. All seats post a 3bb ante
(default `GameConfig(num_seats=6, starting_stack=20bb, ante=3bb,
bb=10_000)`), there are no blinds and no preflop decisions; both boards
deal a flop and play begins there. Showdown: each board is evaluated
independently under the PLO5 rule — best hand using **exactly two** of
five hole cards plus three board cards — and each board takes half the
pot; scooping both halves is the strategic object. Evaluation cost per
hand-board pair is C(5,2)=10 hole-pairs × 10 board-triples; the evaluator
(`hand_eval.rs`) resolves ranks through precomputed perfect-hash tables
(P1/P2 optimizations, 2026-07-10, bit-exact).

**Two Python-facing engines.** `PyGameState` — one table, the serial
reference; drives the UI, eval probes, and the parity tests.
`PyBatchedEngine` — N tables advanced in lockstep with rayon-parallel
feature fills; the training path. The training-side contract is *env-level
parity*: serial and batched must produce identical observations for
identical states (`test_env_batched.py`), while RNG consumption order may
differ between the drivers.

**Environment wrappers** (`python/plo5bp/env.py`, `env_batched.py`).
`BombPotEnv.reset(seed, button, in_hand_mask=None)` → `(obs, StepInfo)`;
`StepInfo` carries `legal_mask` (legacy **8-action** discrete enum:
Fold / CheckCall / five BetPct / AllIn — `NUM_ACTIONS = 8` in
`actions.py`; UI and old tests still use it), `gate_mask` (3-way training
gate), `min_raise_chips`, `max_raise_chips`, `actor`, `raw_obs`, and
per-step `commit_delta`/`total_commit`. Action application is
`step_hybrid(gate, raise_chips)`: FOLD/CHECK_CALL dispatch directly;
GATE_RAISE calls `apply_raise_chips(chips)` **except** in the short-shove
regime — engine zeroes `min_raise_chips` while `legal[ALL_IN]` holds — in
which case it redirects to `apply(ALL_IN)` and ignores the chip argument.

**Reward emission.** Mid-hand rewards are zero. Per-step training costs
are derived by the collector from `commit_delta` (chips the actor put in
this step, negative in bb). At `is_terminal()`, per-seat payouts are
**gross pot shares**: a folder receives 0; sunk chips were already
charged at commit time. Consequence used throughout the project: the
forward value of folding is exactly 0 at every node. When a hand closes
early with ≥2 live seats (all-in before river), rewards come from
`payouts_ev(64, seed)` — the engine deals 64 seeded runouts and averages
the payouts, replacing realized runout luck with its expectation
(variance reduction at the source,
[→ Monte-Carlo & variance](concepts/monte-carlo-variance.md)); fold-outs
and river closes short-circuit to exact payouts.

**Observation production.** `env._pack_obs` merges
`observation_dict()` (Rust) with Rust-computed hand categories and hands
the dict to the encoder (Part 2). The full **1,171-dim** encode can run
in Rust (`PLO5_RUST_ENCODER=1`, enabled on the vSix4 batched path; numpy
remains the pinned reference).

**Reward path in one line** ([→ reward accounting](concepts/reward-accounting.md)):
per-step costs from `commit_delta` (bb); terminal gross pot shares;
fold forward-value ≡ 0; early all-ins use 64-runout EV.

---

## Part 2 — Observation encoding

`python/plo5bp/encoding.py` (module docstring is the authoritative layout
spec; vectorized twin `encode_observation_batch` is pinned bit-exact
against the scalar path). **`OBS_DIM = 1171`** (live full layout) = the
991-dim v2 body + 29-dim obs-v2 tail (→ 1020) + **151-dim v7 batch-2
tail** (stack geometry 41 + board texture 78 + double-board 32, appended
2026-07-12). Historical widths still named in code: `OBS_DIM_V1 = 959`,
`OBS_DIM_V2 = 991`, and experimental **`OBS_DIM_MINIMAL = 796`**
(`--obs-mode minimal`, stem **vMin1** — table-visible dims only, no
engineered equity/blocker/SPR tails). All card sets are encoded
[→ multi-hot](concepts/multi-hot.md); all per-seat vectors are
**hero-rotated** (slot k = seat `(hero + k) mod num_seats`, padded to 8)
so position is relative and one policy serves every chair; all chip
quantities are /bb.

| dims | block |
|---|---|
| 0–52, 52–104, 104–156 | hero hole, board A, board B (52-slot multi-hots) |
| 156–160 | street one-hot (preflop slot always zero in this variant) |
| 160–176 | active mask; all-in mask (hero-rotated, 8-padded) |
| 176–184 | stacks /bb |
| 184–188 | pot, bet_to_call, min_bet, max_bet /bb |
| 188–196 | relative position one-hot of actor |
| 196–772 | **history**: last 32 actions, oldest-first, 18 dims each |
| 772–780 | SPR per seat, clip [0,4] *(saturates deep — see tail)* |
| 780–781 | pot odds `to_call/(pot+to_call)` |
| 781–799 | hero made-hand category one-hot per board (9 classes each) |
| 799–803 | flush/straight draw flags per board |
| 803–813 | pair-with-board counts per board card (rank-desc, 5 slots each) |
| 813–821 | board pair structure per board (paired/double/tripled/quadded) |
| 821–834 | hero rank histogram (13) — closes the unpaired-pocket-pair blind spot |
| 834–910 | straight/flush/SF block per board (38 each): nut distances, straight outs per 10 windows, board-possible masks, flush-draw / nut-flush-draw / SF-draw outs per suit |
| 910–918 | structural seat-exists mask |
| 918–934 | per-seat hand-total and street commits /bb |
| 934–942 | last-aggressor one-hot (all-zero pre-raise) |
| 942–950 | hero→button distance one-hot |
| 950–978 | **cross-board block**: shared ranks (13); per-suit flush made-both / draw-both / mixed; same-hole-pair straight made-both / draw-both / mixed |
| 978–990 | **opp-outcome fractions**: k∈{2,3,4} × {opp scoops, opp quarters, hero scoops, hero quarters} over unseen-deck k-card combos; k=2,3 exhaustive, k=4 MC-1024 with a state-derived seed |
| 990–991 | bet-faced / pot-bet-into, clip [0,4] |
| 991–1020 | **obs-v2 tail** (pure append): per-board ahead/tie/behind + win-one/tie-both (8); blockers-to-nuts per board (4+4); effective-price block (5) — price capped by the *effective* stack (dead-chip invariance) + commitment + log1p money; log1p effective SPR per seat (8), **unclipped** |
| 1020–1061 | **v7 stack geometry** (41): money/raise-exposure behind, raise-ladder envelope, per-seat commitment ratio, spr-after-action, jam plan, pot-ceiling implied odds, side-pot eligibility, call-risk, ante-pot bloat, price-to-continue |
| 1061–1139 | **v7 board texture** (78): rank ladder, suit census, arrival volatility, hero vulnerability outs, straight-out union, boat+ outs, flush-draw rank quality, backdoor census, future nut-flush blocker, turn/river card identity, hero improve outs, board nut ceiling |
| 1139–1171 | **v7 double-board** (32): split-adjusted price ladder, best-hand card usage/coverage, nut-lock/freeroll flags, guaranteed pot share, villain cross-board coverage |

History record layout (18 dims/slot): hero-relative seat one-hot (8),
encoder gate one-hot (4 — the encoder splits Check vs Call; the policy
gate does not), street one-hot (4), chips/bb (1), and chips as a fraction
of the **pot before that action**, clip [0,2] (1) — the same pot-fraction
language the sizing head emits, reconstructed exactly under the 32-slot
truncation window.

Design notes with teeth: the [0,4]-clipped SPR block **saturates for the
entire deep tier at the flop** (true SPR 5.4–13.9), which is why the
obs-v2 tail re-encodes SPR as unclipped `log1p`
([→ symlog & log1p](concepts/symlog-log1p.md)) — a case study in silent
information deletion by saturation. Tails are *pure appends*: 991-era
checkpoints serve through `downgrade_obs_to_v2` (slice to 991), 1020-era
through a slice to 1020, and v1-era 959-dim through the exact `_V1_INDEX`
projection; encoder upgrades in this project are append-only as policy.

**Obs modes.** `--obs-mode full` (default, 1171) vs `--obs-mode minimal`
(796 — cards, street, active/all-in, stacks, pot scalars, history,
commits, seat-exists, button; **no** SPR/categories/draws/blockers/
opp-outcome MC/v2–v7 tails). Minimal is a cold-start-only experiment
(stem **vMin1**: 128×3 actor / 128×2 critic, ~230k envs, ~23M rollout)
asking whether the engineered coaching block is load-bearing or
over-constraining.

---

## Part 3 — Action space and sizing

The policy action is a factored triple `(gate, anchor, u)`.

**Gate** — [→ Categorical](concepts/categorical-distribution.md) over
{FOLD=0, CHECK_CALL=1, RAISE=2}
(`actions.py`), masked by engine legality before normalization: illegal
logits are `masked_fill(-1e9)` so illegal probability is exactly 0
([→ softmax & logits](concepts/softmax-and-logits.md)). Check and call
share a gate because they are never simultaneously legal.

**Anchor** — one of 11 ladder positions. `sizing.py` is the single
source of truth: `PLO_ANCHOR_SPEC.fracs_pm = (0, 100, …, 1000)` per-mille
of `base = pot + to_call`. Chip mapping (integer, half-up):

```
chips_k = clip(to_call + (frac_pm_k · base + 500) // 1000,
               min_raise, max_raise)
```

`chips` are the **delta added** (engine `apply_raise_chips` interface);
`to_call + base` is the pot-limit maximum, so anchor 10 ("pot") is the PL
cap when deep and the jam when short. Anchors clamp monotonically;
**legality dedupe** keeps an anchor only if its chips strictly exceed the
previous anchor's (facing a pot-size bet, rungs 1–5 collapse into the min
atom and vanish). Anchor 0 and 10 are **atoms** (no refinement); the 9
interior anchors carry refinement brackets of half-width
`min(gap to prev, gap to next)//2` per-mille — ±50pm (±5% pot) on this
uniform ladder — chosen so `u = 0.5` lands exactly on the anchor and a
bracket can never swallow a neighbor. The grid
(`anchor_grid_np`/`anchor_grid_torch`) is a pure function of
`(min_raise, max_raise, pot, to_call, spec)`; numpy and torch twins are
pinned bit-identical (`test_anchor_grid.py`) because `evaluate()` never
reverse-engineers anchors from chips — it recomputes the grid and reuses
the stored `(anchor, u)`.

**Refine** — `u ∈ (0,1)` from the chosen anchor's
[→ Beta distribution](concepts/beta-distribution.md), parameterized
`(α, β) = softplus(refine_head(z)) + 1` (both ≥ 1 ⇒ unimodal, finite
log-density at the clamped endpoints `u ∈ [1e-4, 1-1e-4]`). Linear map
across the bracket; atoms and collapsed brackets set `refine_ok = False`
and contribute nothing to log-prob or entropy.

**Joint log-probability** (the PPO "receipt"):
`log π = log P(gate) + 𝟙[raise]·(log P(anchor | legal grid) + log f_Beta(u)·𝟙[refine_ok])`.
Deterministic mode (UI recommendations) replaces samples with
`argmax` gate, `argmax` anchor, and the Beta mean `α/(α+β)`.

---

## Part 4 — Networks

`python/plo5bp/network.py`. Two networks; the actor is the product, the
critic is training-only scaffolding.

### 4.1 Actor torso

For `num_layers = L ≥ 3` (production vSix4: `--hidden-dim 2048
--num-layers 4`): an input projection `Linear(1171 → 2048) + ReLU`
([→ linear layers](concepts/linear-layers.md), [→ ReLU](concepts/relu.md))
followed by `L−1 = 3` residual blocks
([→ residual connections](concepts/residual-connections.md)):

```python
# _ResidualBlock.forward — v6 form (torso_layernorm=True)
h = self.norm(x)           # LayerNorm(2048)  [pre-activation]
return x + F.relu(self.linear(h))   # Linear(2048→2048)
```

The docstring records the empirical constraint driving this: a plain
2048×4 MLP **fails to train** — the skip path is required for gradient
flow at depth. Under `--v6`, LayerNorm
([→ LayerNorm](concepts/layernorm.md)) pre-normalizes each block's input
— the plasticity change; it is **not function-preserving** (normalizes
even at init), hence v6 = fresh stem, and it ships paired with l2-init
(Part 6.7) because norm-solo removes the brake on weight-norm growth.
Legacy `L ≤ 2` builds a flat `Sequential` preserving 128×2 checkpoint
parameter names. The **vMin1** experiment uses 128×3 actor / 128×2 critic
on the 796-dim minimal obs — deliberately tiny, not a production shape.

Real parameter census (built at `OBS_DIM=1171`, mixture K=3, torso LN on):
input block **2,400,256** (`2048×1171 + 2048` bias); each residual block
4,194,304 + 4,096 (LayerNorm); actor total **15,065,119**; critic
(1431→1536 torso, 51-bin HL-Gauss, pooled Q=3) **7,004,214**; joint
**22,069,333**.

### 4.2 Actor heads

All heads are single `nn.Linear` readers of the final 2048-dim feature
`z`:

- `gate_head: 2048→3` (6,147 params) → masked logits → Categorical.
- `mix_head: 2048→3K` (K=3 → 18,441 params), rows = (K μ_raw, K s_raw,
  K mix-logits). This is the **v5 mixture sizing head**
  (`ActorCriticV5`, `head_version=4`): component weights
  `w = (1−Kε)·softmax(mix_logits) + ε` with a **fixed** floor ε=0.03
  (not in the state dict ⇒ serving must hardcode the same constant);
  each component is a discretized logistic over the anchor-index axis,
  and the anchor marginal is the w-weighted sum — an ordinary
  Categorical, so log-prob and entropy are closed-form
  ([→ logistic mixtures](concepts/logistic-mixtures.md)).
- `refine_head: 2048→18` (9 interior anchors × (α,β)).
- `value_head: 2048→1` — the observation-only "display" value the UI
  shows; **not** the training baseline (that's the critic).

`_discretized_logistic_probs(mu, s, legal)` implements the
discretization: anchor k owns the index interval [k−½, k+½];
`P(k) = σ((k+½−μ)/s) − σ((k−½−μ)/s)` with standardized edges clamped to
±12; the **lowest and highest *legal*** anchors absorb the outer tails
(exact-min and exact-pot stay concentratable — v1's continuous-density
failure — and a μ pinned beyond a capped range lands on the nearest legal
anchor, not spuriously on min); illegal anchors are zeroed and the
result renormalized. The v4 head (`head_version=3`) is the K=1 special
case with `s` bounded into [0.3, 5.0] — the scale *floor* is what makes a
one-hot spike impossible and tamed the v2 flat-categorical KL
instability.

`evaluate(obs, gate_mask, sizing, gate_a, anchor_a, u)` recomputes the
distributions, returns the joint log-prob, and decomposes entropy
([→ entropy](concepts/entropy.md)) as

```
H = H_gate + P(raise).detach() · (H_anchor + Σ_k w_k·H_Beta(k)·refine_ok_k)
```

with **two deliberate detachments**: the `P(raise)` weight (an entropy
*bonus* with that weight in-graph pays the gate to shift mass onto the
branch holding ~ln 11 extra nats — a raise bias; detached 2026-06-11) and,
v5-only, the anchor-prob weight inside the Beta term (the verified
"end-anchor entropy subsidy"). It also returns the raw head outputs so
the KL-anchor regularizer can reuse this forward (Part 6.6).

### 4.3 The centralized critic

`CentralCritic` ([→ centralized critic](concepts/centralized-critic.md)):
input = the same **1,171-dim** observation **concatenated with 260 dims**
of hero-rotated opponent hole multi-hots (5 slots × 52; `_rotate_opp_holes`
defines the canonical rotation used identically by training, audit, and
UI review) → **1,431-dim** critic input. Torso `Linear(1431→1536)+ReLU` +
2 residual blocks. Heads:

- **HL-Gauss distributional value head**
  ([→ distributional value](concepts/distributional-value.md)):
  `Linear(1536→51)` logits over 51 bins whose centers are `linspace` in
  **symlog** space spanning ±1500bb (`value_support`); the scalar value
  is `V = symexp(Σ softmax(logits)·centers)`. Training loss
  (`hlgauss_value_loss`) is cross-entropy
  ([→ cross-entropy](concepts/cross-entropy.md)) against a soft label:
  the probability mass a Gaussian centered at `symlog(return)` with
  σ = 0.75·bin-step places in each bin (computed via CDF differences over
  the persisted bin edges, fp32). No value clipping — the categorical
  support *is* the bound. Symlog packs 20bb pots and 1500bb six-way
  all-ins onto one grid with fine resolution near zero.
- **Dueling Q head** ([→ dueling Q](concepts/dueling-q.md)):
  `q(s,a) = V(s).detach() + adv_head(z)`, `adv_head` zero-initialized
  ([→ initialization](concepts/weight-initialization.md)) so Q ≡ V at
  step 0. Width: historically 13 (fold, check/call, raise@anchor 0–10);
  **3 post-2026-07-10** (`--q-pooled`: fold, check/call, raise) after the
  calibration audit showed per-anchor columns starving at ~3% data share
  each. The `V.detach()` stops the auxiliary Q regression from
  double-driving V; the trunk still receives the aux gradient through A.

Both `build_actor_from_state_dict` and `build_critic_from_state_dict`
sniff architecture (obs width, hidden, depth, head family, value bins,
q_actions) from tensor shapes, which is what lets the UI serve any
generation and lets pool rebuilds and the Q-width surgery work without
metadata.

### 4.4 The dueling Q head: mechanism, gradients, and failure history

*The most-audited component in the system, and the one whose training
mechanism is least like anything else here. Plain-English companions:
[→ dueling Q heads](concepts/dueling-q.md) (structure),
[→ how the Q head learns](concepts/q-head-learning.md) (mechanism).*

**Definition.** One critic torso pass yields `z` (1536-dim);
`Q(s,·) = V(s).detach() + adv_head(z)` where `adv_head` is a single
`Linear(1536 → 3)`, zero-initialized, columns = {0: fold, 1: check/call,
2: raise} (pooled 2026-07-10; formerly 2+k per anchor). `V` is the
HL-Gauss head's symexp-mean. The `.detach()` is load-bearing: the Q
regression must not double-drive V (V already has the value loss), but
its gradient still reaches the **shared trunk** through `A(z)` — the
coupling that makes everything below interesting.

**Training signal 1 — taken-action regression** (`ppo.py`, the q-loss
block). Per minibatch row: `q_idx = gate_actions` (pooled; width-13
checkpoints use `2 + anchor`), `q_taken = q_all.gather(q_idx)`,
`MSE(q_taken, mb.returns)`. The gather is the mechanism's defining
property: **each column receives gradient only from rows where its
action was taken**, so a column's training density is the policy's
action frequency. This is what starved the 13-column head (~3% of rows
per anchor column — audit #1, 2026-07-09) and what pooling fixed (the
raise column now sees every raise). Targets are the same GAE λ-returns
the value head trains on — outcome-anchored at γ=1, and **exactly 0 at
fold rows** (per-step-cost accounting; probe-verified 2026-07-11).

**Training signal 2 — dense fold supervision** (`_q_fold_sup_term`).
`q_all[:, FOLD]²` averaged over every fold-**legal** row (mask lifted to
f32 — half-million-row bf16 sums are garbage), weighted
`q_fold_sup_coef = 15.0`. The target is an identity, not an estimate:
fold's forward return ≡ 0, so this is free perfect supervision at ~3×
the row density of fold-taken rows alone. The coefficient history *is*
the lesson: at the original 1.0 the term was **~4% of the q gradient**
(fold-MSE ≈ tens–hundreds of bb² vs taken-MSE ≈ thousands — raw-scale
mismatch), and the known-truth anchor simply lost the tug-of-war (audit
#2's −3/−16bb family offsets). 15.0 ≈ gradient parity; live-tunable via
`anneal_control {"q_fold_sup_coef": X}`.

**Assembly and gradient pathways.**
`q_loss = taken_MSE + 15·fold_MSE`, entering the joint loss at
`q_aux_coef = 0.5`. Backward, the gradient reaches: the gathered
column's row of `adv_head` (per data row), the fold row of `adv_head`
(via supervision), and the **critic trunk through A** — where it
competes with the HL-Gauss value gradient for the same features. Guard
interactions, both deliberate: `adv_head` is **AGC-exempt**
(2026-07-10 — the 10%-of-weight-norm cap was strangling a
zero-initialized head that must grow; NFNet practice exempts final
layers), while the critic-side global clip (total norm 0.5) still
bounds it jointly with the trunk. Residual risk of that exemption:
raw-bb² MSE with an uncapped head can thrash — the local overfit probe
(400 updates on one fixed batch) oscillated rather than converging, and
the live `q=`/`qF=` fields are the watch.

**The moving-target problem.** `A` must represent `target − V(s)`, and
both sides move: V trains toward the same returns concurrently, so the
corrections chase a drifting baseline. Worse, the two live in different
spaces — V is a **symlog-space mean** (symexp of the HL-Gauss
expectation), targets are **raw-space returns** — and the Jensen-type
gap between those means grows with return-distribution width. The A
rows absorb that gap as family-level constants (≈ −3bb shallow, −16bb
deep at audit #2), invisible except against a known truth — which is
exactly what the fold column provides and why the `qF=` canary (mean
`Q[fold]` over fold-legal rows, truth 0, per update, in `PPOStats`) is
on every log line. Post-rebalance behavior: qF settled from a ±1.5bb
transition into a ±0.5bb band within ~6 updates.

**Why supervision, not hard-coding.** `Q[fold] ≡ 0` by construction is
the *correct endgame* (never estimate an identity) and costs one line.
It was deliberately deferred: (1) the fold column is the diagnostic
window onto the shared-offset disease — pin it by fiat and the call/
raise columns keep the bias, newly invisible; (2) the cancellation trap:
VRPO advantages difference Q-terms of adjacent states, so a
*uniform-across-columns* offset cancels exactly, but truth-pinning fold
alone converts the siblings' residual offset into a bias weighted by
the state's fold probability — state-dependent, non-cancelling;
(3) supervision routes the correction through the shared trunk, moving
all columns together, with qF as the progress meter. Migration path:
confirm via audit that the sibling columns' offsets fell with the
trunk, then hard-code fold, delete the supervision term, and pair with
a raw-space base in the v7 head redesign.

**v7 state (2026-07-12, V7_DESIGN.md WS1 — implemented, dormant):** both
endgame pieces now exist behind default-off flags. `--q-fold-zero` pins
`Q[fold] ≡ 0` in the dueling composer (adoption gated on vSix2's qF
evidence — if the anchor holds ≈0 all run, the pin stays unnecessary);
`--q-base-raw` re-composes the dueling base as `Σ p_i·symexp(c_i)` — the
raw-space mean of the SAME HL-Gauss categorical — so base and Q targets
finally share units (this kills the offset *source*; recommended ON for
the first v7 stem). Neither leaves a state-dict trace, so train.py
stamps them into the checkpoint config and refuses warm-starts across a
flip. A third piece is pure telemetry and rides to the pod at the next
natural restart: `Batch.is_terminal` + the **`qT=` canary** = mean
(return − Q(s,a)) over NON-fold terminal rows — the exact boundary
residual that paid the fold subsidy, now watched for every hand-ending
action (positive = subsidy, negative = tax, healthy ≈ 0).

**Consumption — VRPO** (Part 5): the collector stores `Q(s, a_taken)`
and `V^π(s) = Σ_a π(a)·Q(s,a)` per learner step (pooled: gate-probs ·
3 columns; the act-time marginal collapses since Σₖ p_raise·π(k) =
p_raise), feeding `δ⁺ = r + γ·V^π(s′) − Q(s,a)`. At the zero-init head
Q ≡ V and the trace is bit-identical to GAE — the property that makes
the head warm-start-safe (the train.py surgery drops a
width-mismatched `adv_head` back to zero-init on load, and the
estimator gracefully degrades to GAE while the head re-earns trust).

**Ledger.** Audit #1 (2026-07-09): per-anchor starvation, illegal-column
artifact lesson (always check legality masks before reading per-action
outputs) → pooling + fold supervision + AGC exemption (`9d7088a`).
Audit #2 (2026-07-11): drowned supervision + symlog/raw offset;
collector exonerated by probes (fold-row returns ≡ 0, alignment clean)
→ coef 15 + `qF=` canary + live key (`f2e6bf7`). 2026-07-12: the audit
DID graduate into an automated cadence — `scripts/probe_suite.py` runs
the same families A–F (port bit-exact) + the lock-fold probe per
checkpoint into `runs/probe_history.jsonl`; the vSix1 series is logged
as the baseline (deep calibration p95 53→2.3bb while deep-lock P(fold)
rose 34→54% — surface fixed, policy transient consolidating: the
split-screen the suite exists to show).

---

## Part 5 — Data collection and advantage estimation

`python/plo5bp/rollout.py`. Two drivers with different contracts:
`collect_rollout` (serial, bit-exact reference — UI, eval,
exploitability probes; float64 GAE) and `collect_rollout_batched` (the
training path; float32, RNG-consumption order deliberately unpinned
against serial). Production runs `collect_rollout_multiconfig`, which
calls the batched collector once per sampled config (Part 8) and
concatenates.

**Seat roles.** Per environment, a learner-seat mask marks which seats
the current policy plays *as the learner* (their decisions enter the
batch); remaining seats are played by the current policy or by frozen
pool snapshots (Part 7), grouped per snapshot index so each group is one
batched forward. Opponent decisions are never trained on.

**Per-step loop** (steady state, all 49,134 envs advancing together):
read the engine's packed state (obs slab, gate masks, min/max raise,
actors, dones); build the per-step sizing rows `(min_raise, max_raise,
pot, to_call)`; one learner forward `model.act(obs, mask, sizing)`
sampling `(gate, chips, anchor, u)` and returning the joint and per-head
log-probs, the display value, and — under VRPO — the 13-way (pooled:
3-way) **action marginal** π over the Q-head layout, computed from the
same forward with no extra RNG draw; one critic forward
(`q_values`) yielding `V(s)` and the Q row, from which the collector
stores `Q(s, a_taken)` (gate-indexed when the head is pooled) and
`V^π(s) = Σ_a π(a)Q(s,a)`; opponent-group forwards (marginal computation
gated off — they'd discard it); apply all actions through the batched
engine; charge per-step costs from `commit_delta`.

Trajectory storage is per (env, seat): index arrays into a preallocated
obs pool plus parallel arrays for gate/chips/anchor/u/log-probs/value/
q_taken/vpi/costs (capacity `MAX_STEPS_PER_SEAT = 192`).

**Terminal flush.** When hands end, each finished (env, seat) trajectory
is closed with its terminal reward (gross payout, EV-averaged when
early-all-in) and scanned backward:

- **GAE(λ)** ([→ TD, bootstrapping & GAE](concepts/td-bootstrapping-gae.md)):
  `δ_t = r_t + γ·V(s_{t+1}) − V(s_t)`, `A_t = δ_t + γλ·A_{t+1}`, with
  `γ = 1.0`, `λ = 0.95`, V from the critic. `returns_t = A_t + V(s_t)`
  are the value-head targets **always** (even under VRPO).
- **VRPO / Expected-SARSA(λ)** (v6, `advantage_estimator="vrpo"`): the
  same λ-recursion over `δ⁺_t = r_t + γ·V^π(s_{t+1}) − Q(s_t, a_t)`
  using the stored critic quantities. At a zero-init adv head Q ≡ V and
  V^π ≡ V, so the trace is **bit-identical to GAE** — the property that
  makes the Q-head a warm-startable scaffold. Its value proposition:
  `Q(s,a)` replaces sampled-future-action luck with its expectation
  (analytic variance reduction at mixed nodes); its risk — realized in
  the vSix1 incident — is that Q *error* is a fixed function of state
  and does **not** average out across a 9M-row batch the way outcome
  noise does.

**Advantage post-processing**: per-sub-rollout normalize to zero
mean/unit std and clamp to ±`adv_clip` = 8σ (on-device, fp32 reduction
order pinned), then a **global re-normalization across the concatenated
30-config batch** ([→ advantage normalization](concepts/advantage-normalization.md)).
The global step couples the tiers: any tier whose advantage estimates
carry excess variance inflates σ for everyone and shrinks all tiers'
normalized signal — the coupling mechanism identified in the freeze
postmortem.

The assembled `Batch` (≈4,076 bytes/decision) carries: obs (f32),
gate_masks, sizing rows, actions `(gate, anchor, u, chips)`, joint and
per-head old log-probs, critic values, advantages, returns, hero-rotated
opponent holes (u8, for the critic's training forwards), and optional
per-row entropy coefficients (`ent_coef_rows`, Part 8).

---

## Part 6 — The PPO update

`python/plo5bp/ppo.py`, `PPOTrainer.update(batch, rng, entropy_coef)`.

### 6.0 What "clipped surrogate loss" means (teaching spine)

The phrase names the policy half of one minibatch step. Read the concept
cards first if any word is cold:
[→ log-probability](concepts/log-probability.md),
[→ advantage](concepts/advantage.md),
[→ importance ratios & clip](concepts/ppo-clip.md),
[→ clipped surrogate](concepts/clipped-surrogate-loss.md),
[→ minibatches & epochs](concepts/minibatch-epochs.md).

**Story in six beats:**

1. **Collect** stores every learner decision with a *receipt*
   `log π_old(a|s)` — the joint log-prob of the factored action under the
   weights that acted.
2. **Grade** builds an advantage `A` per decision (GAE or VRPO) so each
   row knows whether it beat or missed the baseline
   ([→ advantage](concepts/advantage.md),
   [→ value functions](concepts/value-function.md)).
3. **Re-evaluate** under *current* weights yields `log π_new` for the
   *same* stored action (must be bit-exact with act-time math).
4. **Ratio** `r = exp(log π_new − log π_old)` says how much more/less
   likely the action became.
5. **Surrogate** `r · A` is the cheap proxy for "would this batch look
   better if I nudged π?" — not true poker EV, a local differentiable
   stand-in.
6. **Clip** replaces `r` with `clamp(r, lo, hi)` on the *rewarded* side
   only; the loss is
   `L_π = −mean(min(r·A, clip(r)·A))`. That is the clipped surrogate
   loss. The log field `pi=` *is* this number.

Why a surrogate at all: true on-policy policy gradient would need a
fresh rollout every optimizer step. PPO reuses one ~9M-row rollout for
32 Adam steps (2 epochs × 16 minibatches); without a trust region those
steps walk off the data that produced `A`. The clip *is* the trust
region, written as a loss. Why pessimistic `min`: good actions stop
earning credit past the band; bad actions keep getting punished without
bound if the policy *increases* them.

**Joint receipt** (factored action):
`log π = log P(gate) + 1[raise]·(log P(anchor|legal) + 1[refine_ok]·log f_β(u))`.
Fold/check have no sizing term; atoms and collapsed brackets skip Beta.

**v6 band is not ±0.2.** It is probability-dependent on the *old* gate
probability (6.2). Mid-band room is the per-update policy-KL *quota*; LR
fills it, only widening the band raises it.

**The rest of the loss** (same backward): distributional value + display
value + Q-aux (+ fold supervision) + entropy bonus + optional l2-init /
KL-magnet. Only the policy term is the "clipped surrogate"; the others
are separate desires summed into one scalar before `loss.backward()`.

### 6.0b Reading one log line

A production line looks like:

```
[t] update N  pi=...  v=...  vd=...  H=...  Hg/Ha/Hb=...  kl=...
  klG/klA/klB=...  q=...  qF=...  qT=...  bonus=...  bonus%(F/T/R)=...
  pool=...  seats=...  stacks_bb=[...]  ent=...
```

| field | meaning |
|---|---|
| `pi` | clipped surrogate loss (want consistently negative, few e-3) |
| `v` | HL-Gauss value loss |
| `vd` | display-head MSE (subordinate) |
| `H` / `Hg/Ha/Hb` | total / gate / anchor / beta entropy (nats) |
| `kl` / `klG/A/B` | approx KL this update; head decomposition (additive) |
| `q` / `qF` / `qT` | Q aux loss; mean Q[fold] over fold-legal (truth 0); terminal non-fold residual |
| `bonus% (F/T/R)` | diagnostic only — raises+calls that won per street; reward OFF |
| `pool` | opponent-pool size (cap 8) |
| `seats` / `stacks_bb` | **one sample config** (mix_cfgs[0]), not the aggregate |
| `ent` | effective entropy coefficient (flat or tier-mixed) |
| `lr×s` suffix | present only during LR warmup; vSix4 ships `--lr-warmup-updates 0` |

Per-tier F/T/R prints on the following `[ftr-tier]` line. Never explain
`bonus%` or `H` with the displayed `seats=` alone.

---

Loop: 2 epochs × 16 minibatches (shuffled by `iter_minibatches`), each
minibatch inside a bf16 autocast region
([→ mixed precision](concepts/mixed-precision.md)) with fp32 escapes for
precision-critical reductions. Per minibatch, in code order:

**6.1 Re-evaluation.** `evaluate()` (compiled via `torch.compile` on
CUDA) recomputes the joint log-prob and entropy decomposition of the
*stored* actions under the *current* weights, returning the raw head
outputs for reuse (6.6).

**6.2 Ratio and clip.** `ratio = exp(new_lp − old_lp)`
([→ importance ratios & the PPO clip](concepts/ppo-clip.md)). v6 uses the
**probability-dependent gate clip** (`_gate_clip_bounds`): target
absolute probability-movement room is a symmetric U in the *old* gate
probability p,

```
R(p) = room_ext − (room_ext − room_mid)·4p(1−p)
band = [1 − R/p, 1 + R/p],  p floored at 1e-3, lower bound clamped ≥ 0
```

Rooms shipped as 0.10/0.05/0.10; since 2026-07-11 the rooms are
**live-tunable** (`anneal_control {"clip_room_mid": X, "clip_room_ext":
Y}`) and the live run widened `room_mid` to **0.07** — the mid band is
the per-update policy-KL *quota* at mixed gates (≈0.01–0.02 at 0.05),
and a 22× LR ladder demonstrated that LR cannot raise it, only fill it
(overshoot past the band reads as persistently *positive* `pi`).

— rare gates get ~10 points of recovery room, 50/50 gates get the tight
5-point band; the band keys on the gate but applies to the joint ratio
(the parametric sizing menu isn't over-loosened; for fold/check the
joint ratio *is* the gate ratio). Non-v6 falls back to the flat
`clip = 0.2` band. Policy loss — the **clipped surrogate**
([→ full teaching card](concepts/clipped-surrogate-loss.md)):

```python
ratio = exp(new_lp - old_lp)
surr1 = ratio * advantages
surr2 = clamp(ratio, clip_lo, clip_hi) * advantages
policy_loss = -min(surr1, surr2).mean()   # -> log field pi=
```

Code: `PPOTrainer.update` in `python/plo5bp/ppo.py`.

**6.3 Value losses.** Distributional path (v6): one critic
`train_outputs` forward yields `(V, value_logits, Q)`;
`value_loss = hlgauss_value_loss(value_logits, returns)` (no MSE clip —
the support is the bound). Scalar path (pre-v6): clipped-MSE with
`value_clip`. Display head: plain MSE on the same returns,
`display_value_coef = 0.125` — subordinate by construction.

**6.4 Q-head auxiliary** (`q_aux_coef = 0.5` under v6).
`q_idx = gate` (pooled width 3) or `where(gate==RAISE, 2+anchor, gate)`
(width 13); `q_loss = MSE(q_all.gather(q_idx), returns)` **plus** dense
fold supervision (`q_fold_sup_coef = 15.0` — raised from 1.0 on
2026-07-11 after audit #2 showed the term drowned at ~4% of the q
gradient; live-tunable): `q_all[:, FOLD]²` averaged over every
fold-*legal* row (mask lifted to fp32 — half-million-row sums are
garbage in bf16). Ground truth: fold's forward return ≡ 0, so this is a
free perfect label on ~all facing-a-bet rows, anchoring the head's
hardest sub-task (`A_fold ≡ −V`). The per-update mean of `Q[fold]` over
fold-legal rows is logged as the **`qF=` canary** (truth 0). Full
mechanism, gradient pathways, and failure ledger: Part 4.4.

**6.5 Entropy bonus.** `entropy_for_loss` optionally rescales the
sizing share: `gate_h + sizing_entropy_scale·(entropy − gate_h)` (the
two gate_h terms cancel in gradient, leaving gate weight 1). Bonus =
`−coef · entropy`, where coef is the scalar effective coefficient or,
under mix-configs, the **per-row** `ent_coef_rows` (each transition pays
its own tier's rate). Current live value: 0.25 flat.

**6.6 KL-anchor magnet** (`kl_anchor_coef`, **0/off since 2026-07-09**).
When on: `loss += coef · KL(π_current ‖ π_ref)` where π_ref is a
per-update EMA of the actor (`kl_anchor_ema = 0.999/update` ⇒ ~1000-update
memory; [→ EMA](concepts/ema.md)), computed head-agnostically —
`kl_gate + P(raise).detach()·(kl_anchor + kl_refine)` — reusing 6.1's
forward (the ref forward is no-grad fp32; a second forward-with-grad
cost ~19GiB and OOM'd). The ref persists as `ckpt["model_ema"]` and
doubles as a smoother serving actor (`PLO5BP_SERVE_EMA=1`). Postmortem
(Part 10): on a cold start this term tethers the policy to its own
random init — per-update KL pinned LR-independently at ~0.005, gate
entropy frozen 110+ updates. Re-enable conditions: late-stage only,
eval-gated, EMA horizon shortened to ~0.98–0.99, fresh reference.

**6.7 l2-init** (`l2_init_coef = 1e-4`, v6):
`loss += coef · Σ_trunk ‖p − p₀‖²` over trunk weight matrices only
(`"torso" in name and p.dim() ≥ 2`; heads exempt), p₀ snapshotted at
trainer construction (per-process — a warm restart re-anchors to restart
weights; known, documented). Regenerative regularization: bounds
weight-norm growth so LayerNorm's scale-invariance can't silently decay
the effective LR.

**6.8 Diagnostics.** `approx_kl = mean(old_lp − new_lp)`
([→ KL divergence](concepts/kl-divergence.md)); per-head decomposition
`klG` (gate), `klA` (anchor, sum/B — the C4 normalization fix makes klG
+ klA + klB ≡ kl an additive identity), `klB` derived.

**6.9 Guards.** Checked **before** the optimizer step:
`|kl| > target_kl = 0.5` → soft early-stop — the offending minibatch is
never applied, already-applied minibatches stand (`KLSTOP@mbN`);
`|kl| > kl_hard = 10` → **hard rollback** — parameters *and* Adam moments
*and* the Adam step counters are restored from a snapshot cloned at the
top of `update()`; the whole update reads as if it never happened.
Provenance: vTwo2 died at update 173 with `approx_kl ≈ +2417` before
these existed.

**6.10 The gradient gauntlet.** `loss.backward()`
([→ gradients & backprop](concepts/gradients-and-backprop.md)); then
**AGC** ([→ gradient clipping](concepts/gradient-clipping.md)):
per-tensor `‖g‖ ≤ 0.1·max(‖p‖, 1e-3)`, applied to `_agc_params` — all
parameters **except `critic.adv_head`** (2026-07-10 exemption; the cap
was rate-limiting a zero-init head that must chase a moving trunk); then
**split global clips** — actor parameters to total norm 0.5 and critic
parameters separately to 0.5, split so chip-scale critic loss spikes
can't throttle the gate gradient; then `AdamW.step()`
([→ Adam](concepts/adam.md)) — β₁=0.9, β₂=`adam_b2` (default 0.999;
the "Adam-β2" v6 idea is *not* in the preset — sweep intent), fused on
CUDA, lr = 1.5e-4 × warmup scale
([→ LR & warmup](concepts/learning-rate-warmup.md), linear
`(u+1)/warmup_updates` when warmup > 0). **vSix4 / vMin1 ship
`--lr-warmup-updates 0`** (full LR from update 0 on warm restart);
cold-start guardians historically used 75. Adam moments are
**not checkpointed** — every restart re-estimates them, which is the
operational reason restarts re-ramp LR. Gradient checkpointing
([→ gradient checkpointing](concepts/gradient-checkpointing.md)) trades
torso activation memory for a backward-pass recompute; it's what lets
9M-row updates fit alongside the 2× batch residency.

After the minibatch loops: `_ema_update_ref()` (when the magnet is on),
and the accumulated `PPOStats` map 1:1 onto the log line: `pi v vd H
Hg/Ha/Hb kl klG/klA/klB [klanc] q bonus …`.

---

## Part 7 — Self-play and the opponent pool

`python/plo5bp/selfplay.py`. `OpponentPool` is a fixed-capacity FIFO of
frozen actor `state_dict`s (production capacity 8; snapshots pushed
every `snapshot_every = 5` updates, CPU-cloned). `tags` records each
member's update index; checkpoints persist `pool_member_updates` so
resumes rebuild the **exact** membership from numbered checkpoint
siblings (`seed_pool_from_checkpoints`; incompatible files skipped with
a `[pool] skip` line). Sampling is seeded (`random.Random(run_seed)`) on
the serial path; the batched collector assigns snapshot indices per env
and groups forwards by snapshot.

Mechanism, not folklore: with `pool_mix_prob = 0.5` and
`pool_opp_seats = 2`, roughly half the tables seat two frozen recent
policies against the learner. Pure self-play is prone to strategy
cycling — best-responding to your own latest quirks in circles
([→ self-play & exploitability](concepts/self-play-exploitability.md));
a trailing window of past selves dampens the cycle by making the
opponent distribution a short history average rather than a point. This
is a *stabilizer*, not an exploration device — the pool is the same
lineage, minutes older. Population diversity beyond the lineage is
explicitly out of scope for now (an eval-stack question).

---

## Part 8 — Multi-config training

`scripts/train.py --mix-configs`. Every update resamples **30 configs
= 10 per tier × {clubgg, clubgg_deep, deep}** via
`_sample_game_config(tier, rng)`: uniform seats 2–6 and per-tier
piecewise stack-depth bands (clubgg ≈ the $0.25-ante lineup around
20–40bb with a monster tail; clubgg_deep ≈ the $0.80 game's 20–120bb
bands; deep = 100–250bb uniform). The multiconfig collector runs one
sub-rollout per config (rollout_length/30 rows each) and concatenates.

Per-tier control: each sub-rollout's rows carry that tier's entropy
coefficient (`Batch.ent_coef_rows`), so `{"tier_ent": {...}}` live edits
genuinely apply per tier; per-tier F/T/R prints as the `[ftr-tier]` log
line. **The log's `seats=N stacks_bb=[…]` is `mix_cfgs[0]` — one display
sample, always clubgg-tier by construction, unrelated to any aggregate
on the line.** `bonus%(F/T/R)` semantics (reward OFF, `bonus=+0.0000`):
per street, the fraction of decisions that were a raise that *won*
(field folded, or showdown ≥ half pot) plus a call that won — an
outcome-dependent diagnostic carrying showdown variance; trend-read
only.

---

## Part 9 — Ops and lifecycle

**Driver flow** (`scripts/train.py`): parse args → apply the `--v6`
preset — `{sizing_head: mixture, advantage_estimator: vrpo, q_aux_coef:
0.5, q_pooled: True, q_fold_sup_coef: 15.0, torso_norm: True,
l2_init_coef: 1e-4, agc_clip: 0.1, grad_checkpoint: True, value_bins:
51, clip_prob_dependent: True}` — to every arg still at its parser
default, echoing the applied/kept split (`[v6] preset ON`). The old C2
sharp edge (explicit-at-default flags silently losing to the preset) is
FIXED: covered flags use `default=None` sentinels, so any explicitly
passed value — including `--no-<flag>` and values equal to the legacy
default — wins over the preset. → build model/critic (`critic_q_actions
= 3 if q_pooled else 2 + anchors`) → warm-start.

**Checkpoint anatomy**: `{model, critic, config (hidden_dim/num_layers/
variant/anchor_count/…), update_counter, anneal_tier_ent /
anneal_baseline / anneal_block_acc, pool_member_updates, model_ema
(magnet runs only), game_config}`. Warm-start guards: gate-head width
must match; cross-variant loads are refused unconditionally; the
Q-width surgery — a checkpoint `adv_head` whose shape mismatches the
built head is dropped to fresh zero-init (`[q-pooled] … dropped`) while
everything else loads strict; and the v7 Q-semantics flags
(`q_fold_zero` / `q_base_raw`, stamped in the config since 2026-07-12)
refuse a warm-start across a flip — they leave no state-dict trace but
reinterpret the whole Q surface. Numbered siblings `<stem>_<N>.pt` save on
the cumulative-update grid every `snapshot_every`; the update counter in
logs is session-local, checkpoint numbering is cumulative.

**Live tuning** (`runs/anneal_control.json`, content-change-triggered,
stale file = startup baseline): keys `step`, `tier_ent`, `entropy_coef`,
`entropy_coef_deep`, `lr`, `target_kl`, `kl_hard`,
`sizing_entropy_scale`, `clip_room_mid`, `clip_room_ext`,
`q_fold_sup_coef` (the last three added 2026-07-11). Two operational
sharp edges: any file edit re-applies **all** keys present, so stale
values are live hazards — rewrite the whole file to current-true values
when changing one; and the file's content at startup is a *baseline*
(never applied), so a live-tuned value survives a restart only if baked
into the guardian's flags. That second edge now has a 2-second gate:
`scripts/check_restart_sync.py <guardian.sh> <anneal_control.json>`
diffs the guardian flags against the control baseline and exits nonzero
on mismatch — run it before every restart (limit: preset-derived values
aren't visible as flags; it errs permissive when a knob is absent on one
side). Not live-tunable (restart required): `kl_anchor_coef`, anything
architectural.

**Guardians** (`scripts/<stem>_guardian.sh`, pod-side): pgrep-based
liveness, warm relaunch from the newest numbered sibling on death
(restart cap 4), auto-stop on sustained gate-entropy collapse (Hg < 0.15
in ≥5 of the last 8 updates), clean-stop via `runs/<stem>.stop`.
Operational gotchas, both live-fired: the running guardian holds its
launch flags **in memory** (edit the script → must restart the guardian,
killing it *before* the trainer or it relaunches with stale flags); and
a clean stop can race the 300s poll — the guardian may see the dead
trainer before the stop flag and "crash-relaunch" it (kill the guardian
first, always).

**Serving**: the UI loads one checkpoint per format
(`checkpoints/stub.pt` for PLO5); promotion = copy + server restart;
`_load_model` sniffs generation and serves the EMA actor iff
`PLO5BP_SERVE_EMA=1` *and* the checkpoint carries `model_ema`.
Production (wrapgto.com) is pinned to the v4 generation by decision
2026-07-07; v6 promotes to the local UI only.

---

## Part 10 — Provenance and the current agenda

**Generational lineage.** *v1*: Beta-distribution continuous sizing over
[min,pot], 959-dim obs — endpoints nearly unreachable (continuous
density → ~0 mass at exact-min/pot). *v2*: 11-anchor categorical +
per-anchor Beta refinement + the CentralCritic and display-head split
(991 dims) — representation fixed, optimization fragile: flat-categorical
KL spikes; vTwo2's `approx_kl ≈ +2417` collapse begat `target_kl`.
*v4*: ordinal discretized-logistic (μ, s) with a scale floor — the
optimization fix; vFour4 trained to production and still serves prod.
*v5*: K=3 mixture over the same discretization (multi-modal size menus),
obs → 1020 (later v7 batch-2 tail → **1171** on vSix3+), Q-head scaffold (zero-init, aux-trainable), magnet fixed and
persisted; vFive1 validated warm. *v6*: cold-start substrate kit
(Part 9's preset) — trains from scratch by design; failed stems just
don't ship.

**The vSix1 investigation (2026-07-09/10), compressed.** Symptom: ~200
updates with gate entropy pinned at 0.85–0.89, near-uniform play,
`pi ≈ 0`. (1) The carried-over magnet was tethering a cold start to its
own random init — klanc plateaued as Hg froze, per-update KL sat at
~0.005 **independent of a 20× LR ramp** (the restoring-force signature);
removed → freeze persisted → magnet was necessary-to-remove, not
sufficient. (2) The Q-head audit — exploiting `Q(s,fold) ≡ 0` as a free
ground truth — found the fold column *degrading* between checkpoints,
tens-of-bb self-inconsistency `|E_π[Q] − V|` at the deep tier, per-anchor
columns starving at ~3% data share, and (methodology lesson) the scary
"+28bb raise columns" to be an artifact of reading **illegal-action
columns** that VRPO provably never consumes — always check the legality
mask before reading per-action outputs. (3) Fix (commit `9d7088a`):
pooled Q columns 13→3, dense fold supervision, AGC exemption for the adv
head. Post-fix: advantages are exactly-GAE while the head re-earns
trust; `pi` came alive (−0.003, consistently signed) but Hg held ~0.87 —
the **two-binder** conclusion: advantage SNR (addressed) *and* the
entropy level (0.25 sits above the 0.10–0.18 range every cold start that
ever sharpened here used).

**Live stems (2026-07-23).** **vSix4** — full obs 1171, 2048×4,
~49k envs, ~9M rollout, ent 0.25 flat (no `--anneal-entropy` yet; live
`anneal_control` holds tier_ent at 0.25), warm-started through vSix3.
**vMin1** — minimal obs 796, 128×3/128×2, ~230k envs, ~23M rollout.

**Open agenda, in order**: entropy walk from 0.25 (small steps; co-anneal
with Q — do not wait for perfect qF before the first decrements); Q-head audit cadence — audit #2 ran 2026-07-11 (noise
PASS: deep |E_π[Q]−V| p95 53–61bb → ~13bb; calibration FAIL → the coef-15
fold rebalance + `qF=` canary now standing, Part 4.4) with the
audit-node re-check pending; magnet re-enable only late-stage under the
Part 6.6 conditions; the v7 head questions (hard-coded fold column,
raw-space dueling base) and the network-sizing revisit after a healthy
week; and the standing promotion policy — local UI freely, prod only on
eval-proven, user-approved parity with v4.

*Related deep-dives in the intro track: chapters 1–3 (Parts 0–3 at
conversational pace) and chapter 20 (Part 4 + 6.10 at conversational
pace).*

