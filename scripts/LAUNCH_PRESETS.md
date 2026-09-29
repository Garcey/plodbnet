# Training launch presets

Rewritten 2026-09-28 (ML-027). The old page listed May-era single-tier curriculum
commands that no longer train what they say: block rotation used to be ON by default
and silently replaced their `--stack-dist`, only one named the network size, and none
used the v6 preset or `--mix-configs`. Everything below matches today's trainer.

All commands assume the repo root and a built `.venv` (`.venv/bin/python` on the pod,
`.venv/Scripts/python` on Windows). Use `python -u` when redirecting to a log.

## The production run: vSix6

Do not retype it: the guardian IS the recipe (flags, env, NUMA pinning, resume,
crash/hang handling). On the pod:

```bash
NUM_ENVS=1760000 setsid nohup bash scripts/vSix6_guardian.sh > /dev/null 2>&1 < /dev/null &
```

Its tunable defaults are env variables at the top of `scripts/vSix6_guardian.sh`
(`ENTROPY`, `LAMBDA`, `LR`, `CONFIGS_PER_TIER`, `ROLLOUT_LENGTH`, `NUM_ENVS`,
`MICRO_ROWS`, `ACTOR_HD`, `ACTOR_NL`). Clean stop: `touch runs/vSix6.stop`. Live tuning
without a restart: `runs/vSix6.control.json` (CLAUDE.md, "Live control file").

## Starting a NEW stem

Always name every size (the trainer refuses to run without them), and prefer the v6
preset with mixed configs — the combination every stem since vSix uses:

```bash
.venv/bin/python -u scripts/train.py --variant plo5_double_bomb --v6 \
  --hidden-dim 1024 --num-layers 3 --critic-hidden-dim 1536 --critic-num-blocks 2 \
  --batched --device cuda --num-envs 880000 --rollout-length 150000000 \
  --batch-on-host --micro-batch-rows 500000 --num-minibatches 16 --ppo-epochs 2 \
  --mix-configs --configs-per-tier 30 --mix-tiers clubgg,clubgg_deep,deep \
  --entropy-coef 0.045 --sizing-entropy-scale 0.3 --gae-lambda 0.8 --lr 7.5e-5 \
  --checkpoint checkpoints/<stem>.pt
```

Or keep the recipe in a file and pass `--spec runs/<stem>.toml` (flag names as keys;
the command line still wins). A new stem long-lived enough to need crash recovery gets
its own guardian: copy `scripts/vSix6_guardian.sh`, change `STEM`, the flags and the
compatibility requirements in `pick_warm`; the shared machinery is in
`scripts/guardian_lib.sh`.

## Search-scale runs

- `scripts/recipe_run.sh STEM NODE UPDATES [flags]` — one recipe candidate from a
  fixed start (220k envs, 15M rows), with its Adam state, pool and random stream.
- `scripts/tune_run.sh STEM NODE UPDATES [flags]` — one hyperparameter candidate.
- `scripts/sweep_guardian.sh STEM HD NL CHD CNB TARGET` — one network-size run.

Score them with `scripts/h2h_cross.py` (one seed for every checkpoint) and summarize
with `scripts/round_summary.py`.

## The stack tiers (for reference)

`--mix-tiers clubgg,clubgg_deep,deep` draws `--configs-per-tier` (seats, stacks) setups
per tier per update, all sampled in `python/plo5bp/train/tiers.py` — change the bands
there, not with flags:

- **clubgg** (the realistic table mix): per-seat stacks 5% Short 1-20bb, 50% Hover
  20-40bb, 18% Warm 40-75bb, 17% Big 75-150bb, 10% Monster 150-300bb; with
  `--seats-dist clubgg` the seat counts weigh 6:30% / 5:25% / 4:25% / 3:15% / 2:10%.
- **clubgg_deep** (the $0.80-ante game, ~2x deeper): 2% 1-20bb, 6% 20-30, 16% 30-40,
  22% 40-50, 25% 50-65, 22% 65-80, 7% 80-120bb.
- **deep**: every seat uniform in 100-250bb.

Seats default to uniform over `--num-seats-range 2,3,4,5,6`.
