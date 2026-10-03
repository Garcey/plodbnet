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
#: PLO67 double-board bomb pot (a home-game format, 2026-09-27): PLO5
#: double-board rules with FOUR hole cards and the three burn cards dealt
#: FACE UP (before the flops, turns, rivers); each red burn deals every seat
#: still in the hand one more hole card (4-5 on the flop, 4-6 on the turn,
#: 4-7 on the river). At most 5 seats: 7 reserved slots each + 10 + 3 burns.
VARIANT_PLO67 = "plo67_double_bomb"
#: No-limit hold'em, single board: 2 hole cards, best-5-of-7, NL cap,
#: SB/BB blinds + per-player ante, preflop betting round.
VARIANT_NLH = "nlh_single"

_VARIANTS = (VARIANT_PLO5, VARIANT_PLO4, VARIANT_PLO6, VARIANT_PLO67, VARIANT_NLH)

#: Hole cards per seat AT THE DEAL (mirrors Rust ``Variant::hole_count``).
_HOLE_COUNT = {VARIANT_PLO4: 4, VARIANT_PLO5: 5, VARIANT_PLO6: 6, VARIANT_PLO67: 4, VARIANT_NLH: 2}
#: Deck slots reserved per seat = the most hole cards a seat can hold
#: (mirrors Rust ``Variant::hole_slots``; only PLO67 grows mid-hand).
_HOLE_SLOTS = {**_HOLE_COUNT, VARIANT_PLO67: 7}
#: Burn cards dealt face up (mirrors Rust ``Variant::burn_count``).
_BURNS = {v: (3 if v == VARIANT_PLO67 else 0) for v in _VARIANTS}
#: Boards per hand (mirrors Rust ``Variant::num_boards``).
_NUM_BOARDS = {VARIANT_PLO4: 2, VARIANT_PLO5: 2, VARIANT_PLO6: 2, VARIANT_PLO67: 2, VARIANT_NLH: 1}

#: Table-size bounds every consumer is built for: the observation encoders
#: lay out 8 seat slots (``encoding._MAX_SEATS``) and the engine needs two
#: seats to play a hand.
MIN_SEATS = 2
MAX_SEATS = 8
_DECK_SIZE = 52


@dataclass(frozen=True)
class GameConfig:
    """Static hand configuration. Chip unit: 1 bb = 10000 chips (cent precision at $20/bb).

    Heterogeneous stacks: pass ``starting_stacks=(s0, s1, ...)`` with
    ``len == num_seats``. Otherwise ``starting_stack`` expands uniformly.

    ``variant`` selects the game; ``sb`` is the small blind in chips and
    only meaningful for blind variants (0 for bomb pots). ``ante`` is
    per player in both variants.

    ``reach_cap`` (default True — the rule every network trains on, and Study /
    the Trainer serve): a bet is also capped at what the deepest opponent can
    still put in. False = the home games' rule (owner, 2026-10-02): only the pot
    limit and the bettor's own stack cap it, and what nobody matches comes back
    (the engine's ``GameConfig::reach_cap``). The batched training engine plays
    the capped rule only.

    ``cover_short_bets`` (default False — training): when every opponent who can
    still put chips in has less than one big blind behind, the capped rule's only
    bet is the covering bet (min = max = what they have left — the engine's
    cover-short clamp), and the Raise gate's dust screen hides it
    (``actions.gate_mask_from_bounds``): the deep player could only check. True =
    the website (Study, the Trainer, the graders; owner, 2026-10-03: "in a real
    poker app I would be able to bet $20+ and the opponent just calls for their
    remaining chips"): that bet is offered; only true dust (under bb/100, the
    engine's dust guard) stays screened. The observation is unchanged either way.
    The batched training engine refuses it.
    """

    num_seats: int = 6
    starting_stack: int = 200000
    ante: int = 30000
    bb: int = 10000
    starting_stacks: tuple[int, ...] | None = None
    sb: int = 0
    variant: str = VARIANT_PLO5
    reach_cap: bool = True
    cover_short_bets: bool = False

    def __post_init__(self) -> None:
        if self.variant not in _VARIANTS:
            raise ValueError(
                f"unknown variant {self.variant!r} (expected one of {_VARIANTS})"
            )
        # Reject what the engine / encoders cannot represent HERE, as a
        # ValueError (review 2026-09-20 B8/C4). These used to surface as
        # Rust panics — pyo3's PanicException is a BaseException, so
        # `except Exception` never saw them (1 seat; PLO5 at 9 / PLO6 at 8
        # seats overrunning the deck) — or, worse, not at all: 9+ seats
        # silently corrupted the observation (seat 8's active flag lands in
        # the all-in block) and the critic only has 5 opponent slots.
        if not MIN_SEATS <= self.num_seats <= MAX_SEATS:
            raise ValueError(
                f"num_seats must be in {MIN_SEATS}..{MAX_SEATS}, got {self.num_seats}"
            )
        hole, boards = _HOLE_SLOTS[self.variant], _NUM_BOARDS[self.variant]
        burns = _BURNS[self.variant]
        cards_needed = self.num_seats * hole + 5 * boards + burns
        if cards_needed > _DECK_SIZE:
            raise ValueError(
                f"{self.variant} cannot deal {self.num_seats} seats from one deck: "
                f"{self.num_seats} x {hole} hole + {5 * boards} board"
                + (f" + {burns} burns" if burns else "")
                + f" = {cards_needed} cards (max {(_DECK_SIZE - 5 * boards - burns) // hole} seats)"
            )
        if self.bb <= 0:
            raise ValueError(f"bb must be positive, got {self.bb}")
        if self.ante < 0:
            raise ValueError(f"ante must be >= 0, got {self.ante}")
        if self.sb < 0:
            raise ValueError(f"sb must be >= 0, got {self.sb}")
        if self.starting_stacks is not None:
            if len(self.starting_stacks) != self.num_seats:
                raise ValueError(
                    f"starting_stacks length {len(self.starting_stacks)} != num_seats {self.num_seats}"
                )
            if any(s < 0 for s in self.starting_stacks):
                raise ValueError(
                    f"starting_stacks must be >= 0, got {tuple(self.starting_stacks)}"
                )
        elif self.starting_stack < 0:
            raise ValueError(f"starting_stack must be >= 0, got {self.starting_stack}")

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
        """Hole cards per seat at the deal for this variant (mirrors Rust
        ``Variant::hole_count``; PLO67 grows from here on red burns)."""
        return _HOLE_COUNT[self.variant]

    @property
    def hole_slots(self) -> int:
        """The most hole cards a seat can hold (PLO67 7, else ``hole_count``)."""
        return _HOLE_SLOTS[self.variant]

    @property
    def burn_count(self) -> int:
        """Burn cards dealt face up (PLO67 3, else 0)."""
        return _BURNS[self.variant]

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
    # PPO minibatches per epoch counted on the rows actually COLLECTED
    # (2026-09-28, ML-044): > 0 = each minibatch is ceil(rows / this), so an
    # epoch is exactly this many optimizer steps. 0 = `batch_size` (train.py
    # derives it from the rollout_length TARGET; the drain adds ~8-9% more
    # rows, so vSix6 takes ~17 steps per epoch, not 16, and the count moves
    # with the policy's hand lengths). Not bit-exact: switch at a stem
    # boundary (--minibatches-from-rows).
    num_minibatches: int = 0
    # Common random numbers (2026-09-28, ML-004; --crn-streams): each purpose
    # draws from its own stream keyed by (seed, update[, sub-rollout]) --
    # the update's table configs, its PPO shuffles, and per (env, hand) the
    # deals / buttons / opponent assignments (rollout._CrnDeals) -- so two
    # runs that differ only in a recipe knob see the same tables and hands.
    # False = the one shared stream (every stem so far). Not bit-exact with it.
    crn_streams: bool = False
    # Pool ANCHORS (--pool-anchors, 2026-09-28, ML-033): besides the FIFO of
    # `opponent_pool_size` recent snapshots, the numbered checkpoints nearest
    # these many updates back join the pool (selfplay.refresh_pool_anchors).
    # () = the FIFO alone (every stem so far).
    pool_anchor_ages: tuple = ()
    entropy_coef: float = 0.1
    value_clip: float = 0.2
    hidden_dim: int = 128
    # Observation layout: 'full' = OBS_DIM 1171; 'minimal' = bare
    # table-visible 796 (cards/history/stacks/commits/...). Cold-start only.
    obs_mode: str = "full"
    # Compact rollout-observation STORAGE (compact_obs.py): the exact-0/1
    # columns are kept as bits, the rest verbatim f32 — bit-exact on unpack,
    # so training is unchanged; rows take ~7x less memory on the minimal
    # layout (2.4x full), which is what lets rollout_length grow. False = the
    # old dense float32 rows (A/B, debugging). NLH always stores dense.
    compact_obs: bool = True
    # Half-precision storage of the compact rows' REAL columns in the
    # rollout's output (2026-09-26): ~1.9x the rows per GiB on the full
    # layout (1,022 B vs 1,956 B per observation), at IEEE float16 precision
    # (relative 5e-4; the largest real column, the pot in bb, stays far
    # below the 65,504 limit). NOT bit-exact: the PPO update sees the rows
    # rounded, the act-time forward saw them unrounded. False = float32.
    obs_real_f16: bool = False
    # PPO micro-batching (2026-09-23): split each minibatch into chunks of at
    # most this many rows, accumulating gradients (per-row means weighted by
    # each chunk's share of rows; the fold-supervision term keeps its
    # minibatch-wide denominator), so a minibatch -- and hence the rollout --
    # can outgrow GPU memory. The same gradient mathematically, not
    # bit-identical (float summation order). 0 = off (the exact old path).
    micro_batch_rows: int = 0
    # Keep the collected batch in HOST memory for PPO (2026-09-24): each
    # minibatch (or micro-batch chunk) is gathered on the CPU, in parallel,
    # into pinned staging and copied to the learner device -- compact
    # observations still cross packed and are unpacked there. The device
    # tensors are exactly the ones a device-resident batch yields, so the
    # update is bit-identical; only where the batch lives changes, which
    # bounds the rollout by host RAM instead of GPU memory (pair it with
    # micro_batch_rows for very long rollouts). Mixed-config (multiconfig)
    # collection only. False = the whole batch is copied to the device once.
    batch_on_host: bool = False
    # Batched rollout: act every pool snapshot's opponent rows in ONE stacked
    # forward + ONE sampling pass per step (rollout._StackedOpponents) instead
    # of one act() per snapshot — on the GPU each call is mostly fixed launch
    # overhead. Same per-row policy; the RNG stream differs from the
    # per-snapshot path. False = per-snapshot calls (A/B, debugging).
    batched_opponents: bool = True
    num_layers: int = 2
    num_updates: int = 1000
    opponent_pool_size: int = 8
    snapshot_every: int = 50
    seed: int = 0

    ev_runout_samples: int = 0
    pool_mix_prob: float = 0.5
    pool_opp_seats: int = 2

    # (The per-step and retroactive aggression bonuses -- aggression_bonus_c,
    # retroactive_bonus_c -- were retired 2026-09-28, ML-030: every stem since
    # vTwo ran them at 0. Their qualification still counts winning aggression
    # steps for the F/T/R telemetry, rollout._winning_aggression_steps.)

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
    # AdamW's decoupled weight decay, on every tensor. 0.01 is AdamW's own
    # default -- what every stem to date trained with (implicitly until
    # 2026-09-28, ML-051). Changing it is a stem-boundary decision.
    weight_decay: float = 0.01

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

    # Centralized critic hyperparameters (the critic is only built when
    # train.py constructs one).
    critic_hidden_dim: int = 1536
    critic_num_blocks: int = 2
    # Critic redesign (2026-09-26 regression diagnosis; network.CentralCritic):
    # torso activation, LayerNorm'd input block, and V read out as the mean of
    # the predicted return distribution. Defaults = the pre-redesign critic.
    critic_act: str = "relu"
    critic_in_norm: bool = False
    critic_v_raw: bool = False
    # Critic-only passes over each rollout AFTER the PPO epochs (value + Q
    # losses, the actor untouched). 0 = the pre-2026-09-26 update. A fresh
    # critic beat the 15B-row vSix5 critic after 3 passes over 1.6M rows, so
    # the critic gets more gradient steps per (expensive) rollout.
    critic_extra_epochs: int = 0
    # Minibatches per CRITIC-ONLY epoch (0 = the PPO's num_minibatches). The
    # PPO epochs take ~16 huge steps per rollout -- far too few optimizer
    # steps for a critic (a fresh one lagged the old at 16/epoch while ~1,500
    # small steps offline beat it), so the critic passes take many smaller ones.
    critic_minibatches: int = 0
    # Scale the Q regression's MSE terms (q_aux, fold supervision) by
    # 1 / (minibatch return variance + 1) -- see PPOTrainer._q_norm. For new
    # critics; False = the pre-2026-09-26 raw bb^2 losses.
    critic_q_norm: bool = False
    # With critic_q_norm AND micro-batching: take the return variance of the
    # WHOLE minibatch once (like the fold-supervision denominator) instead of
    # each chunk's own (2026-09-28, ML-011). Only then is a micro-batched step
    # the same gradient as the unchunked one. False = per chunk (vSix6's
    # recipe to date); unchunked minibatches are identical either way.
    critic_q_norm_minibatch: bool = False
    # torch.compile the critic's TRAINING forwards (train_outputs, q_values)
    # on CUDA, like evaluate()/forward already are (ML-013). Not bit-exact vs
    # eager kernels -> a relaunch decision. No effect on CPU / without Triton.
    compile_critic_train: bool = False
    # Weight on the actor's own value head ("display head" for the UI)
    # when a CentralCritic owns the GAE values. Plain regression, no
    # clipping; small so it stays subordinate to the policy loss.
    display_value_coef: float = 0.125
    # Learning rate of the centralized critic's own AdamW param group
    # (2026-09-24 tuning). 0 = the actor's `lr` in ONE param group -- the exact
    # path every stem to date trained with. The actor and critic share one
    # optimizer, so without this the critic's regression and the policy step
    # can only be tuned together.
    critic_lr: float = 0.0
    # Weight on the centralized critic's value loss in the total loss. 0.5 = the
    # historical hardcoded value. Exposed for the distributional head (HL-Gauss
    # cross-entropy has a different magnitude than the old MSE, so the weight
    # needs re-tuning) and for the "critic-weight lift" A/B.
    value_loss_coef: float = 0.5
    # KL-to-EMA-reference regularizer. 0.0 = off (no EMA model built).
    # The reference IS persisted: train.py saves it as ckpt["model_ema"] and
    # restores it on warm-start; absent that key it re-initializes to the
    # loaded weights and ramps in over ~1/(1-ema) updates.
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

    # Sizing-entropy scale (every anchor head, v2..v5): multiplies the
    # anchor+beta (sizing-head) entropy bonus relative to the gate. 1.0 = the
    # sizing heads share entropy_coef with the gate; < 1 lets raise sizes
    # differentiate (vMin3 0.1, vSix6 0.3); > 1 resists the anchor/beta
    # over-sharpening that drove the v2 saturation collapse, WITHOUT
    # loosening the gate. Live-tunable via the control file
    # {"sizing_entropy_scale": X}.
    sizing_entropy_scale: float = 1.0

    # Clamp normalized advantages to ±this many σ before the PPO loss.
    # 0.0 = off. PPO's clip bounds the importance RATIO, not the
    # advantage weight, so one fat-tail sample (a 1500bb six-way
    # all-in pot lands at 30σ+ after unit normalization) carries 30x
    # gradient weight — deep-stack blocks generate exactly these and
    # they drove the update-52+ violence. Applied identically in the
    # serial and batched collectors.
    adv_clip: float = 8.0

    # Drain in-flight hands at rollout end (review 2026-09-20 A6). When True the
    # collectors stop re-dealing once the row target is reached and keep stepping
    # until every live hand finishes, instead of discarding hands in flight (which
    # under-samples long hands by ~len/W). PRODUCTION BEHAVIOR CHANGE vs the
    # legacy truncation; `train.py --no-drain-inflight` restores it.
    drain_inflight: bool = True

    device: str = "cpu"
