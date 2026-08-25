# Glossary

One line per term, in the order the course introduces them. (ch#) = where
it's first explained properly.

## From chapter 1

- **update / lap** — one full cycle of collect-then-learn; ~9M decisions,
  ~11–13 min on the pod. The unit everything is counted in. (ch1)
- **rollout / collect phase** — playing ~49k simultaneous tables to gather
  decisions for one update. (ch1)
- **observation** — the 1,171 numbers (full mode; 796 minimal) describing
  one decision point; the model's entire input. (ch1, detail ch2)
- **actor / policy** — the network that plays: observation in,
  probabilities over legal actions out. (ch1)
- **gate** — stage one of an action: fold / check-call / raise. (ch1,
  detail ch3)
- **anchor** — stage two: one of 11 raise sizes (min-raise, 10%…100% pot).
  (ch1, detail ch3)
- **sampling** — actions are drawn from the probabilities, not always the
  max — the source of exploration. (ch1)
- **the receipt (log-prob)** — the probability the model gave the action
  it took, recorded at act time; PPO's reference point. (ch1, detail ch7)
- **opponent pool / snapshot** — frozen earlier copies of the model seated
  at some tables to damp self-play cycling. (ch1, detail ch12)
- **reward** — chips won/lost in big blinds at hand end; the only quality
  signal that exists. (ch1, detail ch4)
- **EV runouts** — all-ins before the river are graded by averaging 64
  dealt runouts instead of the one that happened. (ch1, detail ch4)
- **critic** — the training-only grading network; sees ALL hole cards;
  estimates the value of situations. (ch1, detail ch5)
- **value** — a network's estimate of "chips I expect to win from here."
  (ch1, detail ch5)
- **advantage** — result minus expectation; the anti-results-oriented
  number that decides whether an action's probability goes up or down.
  (ch1, detail ch6)
- **PPO clip** — the seatbelt limiting how far the policy can move from
  its receipts in one update. (ch1, detail ch7)
- **KL (kl= in the log)** — how far the policy actually moved this update;
  guards trip when it spikes. (ch1, detail ch8)
- **entropy (H, Hg/Ha/Hb)** — how mixed the play is (total / gate / size
  ladder / refine); the ent coefficient pays the model to stay mixed.
  (ch1, detail ch9)
- **pi / v / vd / q (log fields)** — the loss readouts: policy loss,
  critic error, display-head error, Q-head error. (ch1)
- **bonus%(F/T/R)** — per-street rate of raises-that-won plus
  calls-that-won; outcome-dependent diagnostic, reward is OFF. (ch1)
- **seats=N trap** — the log's seats/stacks show ONE of the 30 configs,
  for flavor; stats aggregate all 30. Never explain a stat with it. (ch1,
  detail ch14)
- **checkpoint / stem** — saved weights (every 5 updates numbered);
  a "stem" is a named run family like vSix1. (ch1, detail ch19)

## From chapter 2

- **multi-hot** — a 52-slot checklist with 1s for present cards; how all
  cards are encoded. (ch2)
- **hero-rotation** — every per-seat list starts at hero and goes
  clockwise, so position is relative and one strategy serves every chair.
  (ch2)
- **history block** — the last 32 actions, oldest first, 18 numbers each;
  over half the observation. (ch2)
- **pot-fraction language** — bet sizes expressed as a fraction of the
  pot at that moment — shared by the history encoding and the sizing
  head. (ch2, ch3)
- **coaching notes** — precomputed poker facts (SPR, categories, outs,
  textures, blockers) so learning is spent on strategy, not arithmetic.
  (ch2)
- **opp-outcome fractions** — how hero's hand ranks against the universe
  of unseen-deck hands (scoop/quarter each way); range-vs-universe math,
  not card peeking. (ch2)
- **saturation** — when a clipped/capped feature stops distinguishing
  values (the deep-tier SPR=4 bug); silently deletes information. (ch2)
- **dead-chip invariance** — prices capped by the EFFECTIVE stack: chips
  that can't be lost shouldn't change the price of a call. (ch2)
- **pure append** — encoding upgrades only ADD dims at the end; old
  checkpoints keep serving via a tail slice. (ch2)
- **bit-exact** — the requirement that the same spot encodes to the exact
  same bytes forever; enforced by tests; why encoder changes are scary.
  (ch2, detail ch11)

## From chapter 3

- **mask** — illegal gates/rungs get probability exactly 0 (score set to
  −∞ before probabilities form); rules live in the engine, not the net.
  (ch3)
- **base (pot after call)** — the sizing reference: a rung's chips =
  call + fraction × (pot + to-call); 100% = the pot-limit max. (ch3)
- **atom** — a rung with no slider: min and pot. Their meanings are
  absolute ends of the legal window. (ch3)
- **rung collapse / dedupe** — clamped rungs that land at or below the
  rung beneath them vanish from the menu (facing a pot bet, the bottom
  half of the ladder collapses into min). (ch3)
- **short-shove redirect** — too short to min-raise: the only rung left
  is the top one and it means all-in; the engine redirects. (ch3)
- **refine bracket** — ±5% pot around an interior rung; u ∈ (0,1) slides
  linearly across it; u = 0.5 is exactly the rung (an untrained net bets
  exactly the anchors). (ch3)
- **deterministic mode** — argmax gate + argmax rung + average slider;
  what the study tab's recommendation and the bet chart's white dot are.
  (ch3)

## From chapter 20 (read early)

- **neuron / weight / bias** — a weighted checklist over its inputs plus
  a constant, outputting one match-score. (ch20)
- **ReLU** — keep positives, zero negatives; the nonlinearity that makes
  stacked layers more than one big linear map. (ch20)
- **torso** — the shared body: input layer (1,171→2,048) + three
  residual blocks; builds the 2,048-number summary `z` all heads read.
  (ch20)
- **residual / skip connection** — blocks ADD a correction to their
  input instead of replacing it; also the gradient highway that makes
  depth trainable. (ch20)
- **LayerNorm** — re-centers/re-scales a block's inputs; keeps an aging
  network trainable (plasticity); not function-preserving → v6 cold
  start; must pair with l2-init. (ch20)
- **head** — a tiny final linear layer reading a decision off `z` (gate
  6,147 params; mix 18,441; refine 36,882; value 2,049). (ch20)
- **loss** — the single number summing every training desire; weights
  feel only the net force, never the individual terms. (ch20)
- **gradient / backprop** — each weight's personalized "would the loss
  rise or fall if I grew" answer, computed by blame flowing backward
  through the forward pass's recorded wiring. (ch20)
- **AdamW** — the optimizer: per-weight trend (momentum) ÷ typical
  magnitude → every weight steps at a reliability-adjusted pace; its
  running averages are NOT checkpointed (why restarts re-ramp LR).
  (ch20)
- **AGC** — per-tensor cap: gradient ≤ 10% of the tensor's own weight
  norm; the Q head is exempt post-audit. (ch20)
- **l2-init** — constant gentle pull of trunk weights toward their
  run-start values; LayerNorm's mandatory partner (stops weight-norm
  inflation from silently shrinking the effective LR). (ch20)
- **gradient checkpointing** — recompute activations during backward
  instead of storing them; same math, less memory. (ch20)

## From the live system (master doc)

- **qF canary** — the log line's per-update mean of Q[fold] over
  fold-legal rows; ground truth is exactly 0, so sustained drift = the
  Q surface acquiring a systematic lean. (MASTER 4.4)
- **fold anchor** — dense supervision pushing the fold column to its
  known-zero value on every fold-legal row; weighted 15× for gradient
  parity with the raw-outcome regression. (MASTER 4.4)
- **clip quota** — the per-update policy-KL ceiling set by the
  prob-dependent clip's mid band; LR fills it, only the band widens it.
  (MASTER 6.2)


## From the PPO teaching pack (MASTER 6.0)

- **receipt / log-prob** — `log π(a|s)` stored at act time; recomputed at
  update time; their difference drives the importance ratio.
- **importance ratio** — `r = π_new / π_old = exp(logπ_new − logπ_old)`.
- **advantage (A)** — how much better/worse an action was than the
  baseline; the grade multiplied by the ratio.
- **surrogate objective** — cheap proxy `r·A` standing in for true
  on-policy EV over a reused batch.
- **clipped surrogate loss** — `−mean(min(r·A, clip(r)·A))`; the policy
  term; log field `pi=`. (MASTER 6.0, concepts/clipped-surrogate-loss.md)
- **prob-dependent clip** — v6 band width depends on old gate probability
  (rare gates get more room than 50/50).
- **minibatch / epoch** — 16 shards × 2 passes = 32 Adam steps per update.
- **GAE / VRPO** — two advantage estimators; v6 default VRPO, collapses to
  GAE while Q is zero-init.
- **commit_delta cost** — per-step bb cost from chips put in; fold forward
  value ≡ 0 by accounting identity.
- **OBS_DIM / minimal** — full 1171 vs experimental bare-visibility 796
  (vMin1).
- **NUM_ACTIONS = 8** — legacy discrete menu (UI/tests); training gate is
  3-way + anchors.
- **vSix4 / vMin1** — live production full-obs stem / parallel minimal-obs
  experiment.
