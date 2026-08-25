# Minibatches and epochs

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

One training **update** collects a huge rollout (~9 million decisions),
then runs the optimizer over that fixed batch several times before
throwing it away and collecting fresh data. Two knobs name that reuse:

- **Epoch** — one full pass over every row in the rollout.
- **Minibatch** — one optimizer step on a random shard of the rollout.
  Production: **2 epochs × 16 minibatches = 32 Adam steps** per update.

## Why not one giant step?

A single gradient on 9M rows would be stable but slow to iterate and
awkward for GPU memory. Sharding into 16 minibatches lets each step see a
manageable slice, keeps activation memory in bounds (with gradient
checkpointing), and — because the data is shuffled each epoch — gives the
optimizer two slightly different looks at the same experience.

## Why not dozens of epochs?

Every extra epoch pushes π further from the π_old that generated the
advantages. The clipped surrogate and the KL guards exist exactly because
reuse is dangerous. Two epochs is the project's settled compromise:
enough to extract signal from an expensive rollout, not enough to walk
off the data distribution. If a minibatch's `|approx_kl|` exceeds
`target_kl` (0.5), that step is skipped (`KLSTOP@mbN`); if it exceeds
`kl_hard` (10), the entire update rolls back.

## Order inside one minibatch

1. Re-evaluate stored actions under current weights → new log-probs.
2. Build ratios, clipped surrogate, value/Q/entropy/l2 terms → `loss`.
3. Backward → AGC → split grad clips → AdamW step (unless a KL guard trips).

## In this project

- `PPOTrainer.update` in `python/plo5bp/ppo.py`; sharding via
  `iter_minibatches`.
- Rollout size and env count are launch flags (`--rollout-length`,
  `--num-envs`); minibatch count is effectively fixed by the trainer
  config used in production guardians.
- The log line reports *averages over the minibatches that actually
  stepped* (a KLSTOP shrinks the denominator).
