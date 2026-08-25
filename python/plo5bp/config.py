"""Configuration dataclasses for the game and training."""

from __future__ import annotations

from dataclasses import dataclass

#: PLO5 double-board bomb pot: 5 hole cards, two boards, pot-limit,
#: ante-only (no blinds; hands start at the flop). The original format.
VARIANT_PLO5 = "plo5_double_bomb"
#: PLO4 double-board bomb pot: identical rules to PLO5 double-board
#: (two boards, pot-limit, ante-only, starts at the flop) with 4 hole cards.
VARIANT_PLO4 = "plo4_double_bomb"
#: PLO6 double-board bomb pot: identical rules to PLO5 double-board
#: (two boards, pot-limit, ante-only, starts at the flop) with 6 hole cards.
VARIANT_PLO6 = "plo6_double_bomb"
#: No-limit hold'em, single board: 2 hole cards, best-5-of-7, NL cap,
#: SB/BB blinds + per-player ante, preflop betting round.
VARIANT_NLH = "nlh_single"

_VARIANTS = (VARIANT_PLO5, VARIANT_PLO4, VARIANT_PLO6, VARIANT_NLH)


@dataclass(frozen=True)
class GameConfig:
    """Static hand configuration. Chip unit: 1 bb = 10000 chips (cent precision at $20/bb).

    Heterogeneous stacks: pass ``starting_stacks=(s0, s1, ...)`` with
    ``len == num_seats``. Otherwise ``starting_stack`` expands uniformly.

    ``variant`` selects the game; ``sb`` is the small blind in chips and
    only meaningful for blind variants (0 for bomb pots). ``ante`` is
    per player in both variants.
    """

    num_seats: int = 6
    starting_stack: int = 200000
    ante: int = 30000
    bb: int = 10000
    starting_stacks: tuple[int, ...] | None = None
    sb: int = 0
    variant: str = VARIANT_PLO5

    def __post_init__(self) -> None:
        if self.variant not in _VARIANTS:
            raise ValueError(
                f"unknown variant {self.variant!r} (expected one of {_VARIANTS})"
            )
        if self.starting_stacks is not None and len(self.starting_stacks) != self.num_seats:
            raise ValueError(
                f"starting_stacks length {len(self.starting_stacks)} != num_seats {self.num_seats}"
            )

    @classmethod
    def nlh_default(
        cls,
        num_seats: int = 6,
        starting_stack: int = 1_000_000,
        starting_stacks: tuple[int, ...] | None = None,
    ) -> "GameConfig":
        """The UI default NLH stake: $5/$10 with a $5 per-player ante at
        1bb = 10000 chips ($10) → sb 5000, bb 10000, ante 5000. Default
        stacks 100bb (the reference ClubGG table runs 100-250bb)."""
        return cls(
            num_seats=num_seats,
            starting_stack=starting_stack,
            ante=5_000,
            bb=10_000,
            starting_stacks=starting_stacks,
            sb=5_000,
            variant=VARIANT_NLH,
        )

    @property
    def hole_count(self) -> int:
        """Hole cards per seat for this variant (mirrors Rust
        ``Variant::hole_count``)."""
        return {
            VARIANT_PLO4: 4,
            VARIANT_PLO5: 5,
            VARIANT_PLO6: 6,
            VARIANT_NLH: 2,
        }[self.variant]

    @property
    def resolved_stacks(self) -> tuple[int, ...]:
        if self.starting_stacks is not None:
            return self.starting_stacks
        return (self.starting_stack,) * self.num_seats


@dataclass(frozen=True)
class TrainingConfig:
    """PPO training hyperparameters."""

    lr: float = 3e-4
    clip: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    rollout_length: int = 2048
    num_envs: int = 32
    ppo_epochs: int = 4
    batch_size: int = 256
    entropy_coef: float = 0.1
    value_clip: float = 0.2
    hidden_dim: int = 128
    # Observation layout: 'full' = OBS_DIM 1171; 'minimal' = bare
    # table-visible 796 (cards/history/stacks/commits/...). Cold-start only.
    obs_mode: str = "full"
    num_layers: int = 2
    num_updates: int = 1000
    opponent_pool_size: int = 8
    snapshot_every: int = 50
    seed: int = 0

    ev_runout_samples: int = 0
    pool_mix_prob: float = 0.5
    pool_opp_seats: int = 2

    # Pot-fraction aggression bonus (bb units). When > 0, each
    # GATE_RAISE step (covering both normal raises and short shoves
    # encoded at u=1) gets an additional
    # `c * min(1.0, aggressive_chips / pre_step_pot)` added to the
    # forward-EV per-step reward. Default 0.0 disables the bonus.
    aggression_bonus_c: float = 0.0

    # Retroactive aggression bonus (bb units). When > 0, at end-of-hand
    # each learner-seat trajectory receives a flat bonus on qualifying
    # steps based on hero's pot share:
    #   share > 50%  → bonus on GATE_RAISE steps only
    #   share == 50% → bonus on GATE_RAISE + GATE_CHECK_CALL(chips > 0)
    #   share < 50%  → no bonus
    # Folds and pure checks never get bonus. Independent of
    # `aggression_bonus_c`; both can be set but typical use is one or
    # the other.
    retroactive_bonus_c: float = 0.0

    # Auxiliary Q(s, a) regression coefficient for the critic's dueling
    # head (v5 stems; head exists zero-init regardless so the VRPO
    # advantage flip is not a checkpoint break). 0 = untrained.
    q_aux_coef: float = 0.0

    # Pool the dueling head's per-anchor raise columns into ONE raise
    # column (q_actions = 3: Fold/CheckCall/Raise, instead of 2+anchors).
    # 2026-07-09 Q-head audit: the 11 anchor columns each saw ~3% of the
    # rows and dominated the VRPO advantage noise (per-node |E_π[Q]−V|
    # p95 30-60bb at deep tiers); pooling gives the raise column 11× the
    # training density. The Q baseline's job is variance reduction —
    # size-specific credit still arrives through the reward trace.
    # Consumers (ppo q_idx, rollout marginal, UI loader) key off the Q
    # tensor's WIDTH, so 13-column checkpoints keep loading unchanged;
    # a warm-start across widths drops adv_head to fresh zero-init.
    q_pooled: bool = False

    # Dense supervision on the fold column: fold's forward return is
    # EXACTLY 0 under the reward convention (per-step costs, sunk chips
    # excluded, a folder wins nothing), so q[..., FOLD] regresses to 0
    # on EVERY fold-legal row — free perfect labels on all facing-a-bet
    # rows, not just the ~third where fold was taken. Anchors the head's
    # hardest sub-task (A_fold ≡ −V). Relative weight vs the taken-action
    # MSE inside the q_aux term; 0 = off.
    q_fold_sup_coef: float = 0.0

    # v7 WS1.1 (V7_DESIGN.md): pin Q[FOLD] ≡ 0 by construction instead of
    # learning it toward the supervision target. Kills the fold-subsidy
    # class outright, at the cost of an init-era transient: until the
    # sibling columns specialize away from V, E_π[Q(s')] under-reads
    # V^π(s') by ~π_fold(s')·V(s') — mild pessimism on bootstrapped
    # continues. Fresh-stem / deliberate-experiment flag; warm-starts
    # across a flip are refused (the whole Q surface reinterprets).
    q_fold_zero: bool = False

    # v7 WS1.2 (V7_DESIGN.md): compose the dueling base in RAW-return
    # space — base = Σ p_i·symexp(c_i) from the same HL-Gauss categorical —
    # instead of the display V = symexp(Σ p_i·c_i). The legacy base
    # straddles value spaces (symlog-space mean vs raw-space Q targets),
    # a Jensen-type gap the adv rows absorbed as the July family offsets
    # (−3/−16bb by tier, width-scaled). Raw base puts base and target in
    # the same units; zero new parameters. Requires value_bins>0;
    # warm-starts across a flip are refused.
    q_base_raw: bool = False

    # Advantage estimator (V5_DESIGN.md W2.5; VRPO, Fan & Farina 2026).
    #   "gae"  = V-based GAE(λ) (default; the pre-VRPO path, unchanged).
    #   "vrpo" = Expected-SARSA(λ) off the centralized dueling Q head:
    #            δ⁺ = r + γ·V^π(s') − Q(s,a), V^π(s') = Σ_a π(a|s')·Q(s',a),
    #            which analytically averages out the future-action-sampling
    #            variance GAE carries at mixed nodes. Requires the critic Q
    #            head AND q_aux_coef>0 (the head must be trained first). At the
    #            zero-init Q head Q≡V, so it reduces byte-exactly to GAE.
    #            `returns` (value-head target) stay GAE-based. Batched collector
    #            only (the training path); the serial collector rejects it.
    advantage_estimator: str = "gae"

    # v6 plasticity (V6_RESEARCH.md #4). `torso_layernorm` inserts pre-activation
    # LayerNorm into the residual torso of BOTH the actor and the critic
    # (x + ReLU(Linear(LayerNorm(x)))) to fight plasticity loss over long runs;
    # NOT function-preserving → fresh stem, num_layers>=3 only. `l2_init_coef` is
    # the REQUIRED companion: an L2-to-init penalty on the TRUNK weight matrices
    # that keeps weight-norm (hence effective LR) from decaying and counters the
    # generalization hit of norm-solo (Nauman 2024). Both default off = byte-
    # identical to pre-v6.
    torso_layernorm: bool = False
    l2_init_coef: float = 0.0

    # Optimizer hygiene (V6_RESEARCH.md internals). adam_b2 = AdamW's second-
    # moment β2 (sweep {0.98,0.99,0.999} against heavy-tailed policy-ratio
    # spikes; 0.999 = the current default). agc_clip = stateless adaptive
    # gradient clipping coefficient: per-tensor, clip each param's grad to
    # agc_clip*||param|| (NFNet AGC) — a per-tensor complement to the existing
    # per-group split clip; 0 = off, no running state (kl_hard-rollback-safe).
    adam_b2: float = 0.999
    agc_clip: float = 0.0

    # v6 probability-dependent PPO clip (Over-mixing §6; a generalization of
    # DAPO "Clip-Higher"). When True, the GATE's clip band widens for RARE gate
    # actions and narrows near 50/50, keyed on the sampled gate's OLD probability
    # p, so a suppressed-but-correct gate (e.g. a check that should recover)
    # climbs in a few updates instead of ~25 (the multiplicative clip freezes it
    # at 1.2× of a tiny base), while genuinely-mixed nodes take smaller, less-
    # thrashy steps. The band is set by a target ABSOLUTE probability-movement
    # room R(p) shaped as a symmetric U in p:
    #     R(p) = clip_room_ext − (clip_room_ext − clip_room_mid)·4p(1−p)
    # and the per-sample ratio band is [1 − R/p, 1 + R/p] (p floored by
    # clip_prob_floor, capping the max ratio at ~1 + clip_room_ext/floor).
    # Defaults 0.10 / 0.05 give ~10 points of room at the extremes and ~5 at the
    # middle. Scoped to the GATE (uses old_gate_logp) so the parametric sizing
    # menu is NOT over-loosened. clip_prob_dependent=False → the flat cfg.clip
    # band, byte-identical to pre-v6.
    clip_prob_dependent: bool = False
    clip_room_ext: float = 0.10
    clip_room_mid: float = 0.05
    clip_prob_floor: float = 1e-3

    # Gradient checkpointing (V6 internals): recompute torso activations in
    # backward instead of storing them — identical math, trades compute for
    # memory to buy back rollout headroom (the K=3 head OOM'd 11M→9M). Runtime
    # flag on the trainable model+critic only; rollout is no-grad (unaffected).
    grad_checkpoint: bool = False

    # Distributional / HL-Gauss critic value head (V6 internals, the keystone).
    # value_bins > 0 replaces the scalar critic value head with a categorical
    # head over a symlog-transformed support (±value_support bb), trained with
    # HL-Gauss cross-entropy (Gaussian σ = value_hlgauss_sigma bin-widths; →0 =
    # hard two-hot). V = symexp(E[bins]) stays scalar (dueling Q + serving
    # unchanged). 0 = scalar MSE head (default, byte-identical to pre-v6).
    value_bins: int = 0
    value_support: float = 1500.0
    value_hlgauss_sigma: float = 0.75

    # v2 (anchor head + centralized critic) hyperparameters. Ignored on
    # v1 runs — the critic is only built when train.py constructs one.
    critic_hidden_dim: int = 1536
    critic_num_blocks: int = 2
    # Weight on the actor's own value head ("display head" for the UI)
    # when a CentralCritic owns the GAE values. Plain regression, no
    # clipping; small so it stays subordinate to the policy loss.
    display_value_coef: float = 0.125
    # Weight on the centralized critic's value loss in the total loss. 0.5 = the
    # historical hardcoded value. Exposed for the distributional head (HL-Gauss
    # cross-entropy has a different magnitude than the old MSE, so the weight
    # needs re-tuning) and for the "critic-weight lift" A/B.
    value_loss_coef: float = 0.5
    # KL-to-EMA-reference regularizer. 0.0 = off (no EMA model built).
    # The reference re-initializes to current weights on every (re)start
    # — it is NOT persisted in checkpoints.
    kl_anchor_coef: float = 0.0
    kl_anchor_ema: float = 0.999

    # SOFT KL guard (early-stop): when a minibatch's |approx_kl| exceeds
    # this, break the PPO inner loop but KEEP the minibatches already
    # applied this update (the tripping minibatch's step is not applied).
    # Standard PPO early-stopping — bounds per-update drift without
    # discarding progress. 0.0 = off. NOTE: this used to do a full
    # rollback at this threshold, which froze a run (317/318 updates
    # rolled back to no-ops, 2026-06-12) once the policy sharpened enough
    # to cross it every update. Rollback now lives at kl_hard below.
    # 0.5 (not 2.0): the v2/v4 discrete heads have heavier-tailed importance
    # ratios than v1's Beta; vFour ran 2.0 and the gate over-moved → collapsed
    # at u36, vFour3 held at 0.5.
    target_kl: float = 0.5

    # HARD KL guard (full rollback): when a minibatch's |approx_kl|
    # exceeds this, restore params + optimizer moments captured at the
    # top of update() — the ENTIRE update is discarded. Reserved for
    # genuine catastrophe (vTwo2 hit approx_kl ≈ +2417 at update 173 and
    # collapsed entropy to 0). Should be >= target_kl. 0.0 = no hard
    # rollback (soft early-stop still applies).
    kl_hard: float = 10.0

    # Sizing-entropy scale (v2 only): multiplies the anchor+beta
    # (sizing-head) entropy bonus relative to the gate. 1.0 = off (gate and
    # sizing heads share entropy_coef). >1 gives the sizing heads a stronger
    # entropy bonus to resist the anchor/beta over-sharpening that drives v2
    # saturation collapse, WITHOUT loosening the gate. Live-tunable via
    # runs/anneal_control.json {"sizing_entropy_scale": X}.
    sizing_entropy_scale: float = 1.0

    # Clamp normalized advantages to ±this many σ before the PPO loss.
    # 0.0 = off. PPO's clip bounds the importance RATIO, not the
    # advantage weight, so one fat-tail sample (a 1500bb six-way
    # all-in pot lands at 30σ+ after unit normalization) carries 30x
    # gradient weight — deep-stack blocks generate exactly these and
    # they drove the update-52+ violence. Applied identically in the
    # serial and batched collectors.
    adv_clip: float = 8.0

    device: str = "cpu"
