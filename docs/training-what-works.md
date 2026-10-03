# Training — what has worked and what hasn't

Distilled from the training experiments of 2026-09-23 → 2026-09-29 (PLO5 double-board
bomb pots, PPO self-play on the RunPod pod). The dated evidence behind every line is in
[training-log.md](training-log.md); the flags and code are described in
[training.md](training.md). Update a line here when a new result changes it.

**How to read the numbers.** Edges are bb per seat per hand from `scripts/h2h_cross.py`
head-to-heads (61,440 paired hands). Each result has two numbers:

- **sampled**: both models mix their actions the way they were trained (how it plays);
- **argmax** (also called "top choice"): both always play their single most likely
  action (what it has learned to prefer, apart from how much it randomizes).

A single checkpoint swings ±0.03–0.05, and sometimes ±0.1. Every claim below rests on
means or trends over several checkpoints.

## The best recipe so far: r8c + obs-X `line,run` = vSix7 (2026-09-29)

**Update 2026-09-29 (obs-X rounds, `docs/design/OBS_X_DIMS_2026-09-29.md`):** on top of
r8c, two new observation groups (`--obs-x-groups line,run`, 75-dim tail after the 1171:
all-in equity with the turn / river dealt vs 1 and 2 random hands, and per-player
betting-line summaries) add about **+0.02 bb/seat-hand** at the same update (3 seeds,
both modes positive) at ~60% more time per update. Production **vSix7** runs r8c's recipe
with them from 2026-09-29 — since 2026-09-30 at the pod's maximum rollout, ~148.6M rows
per update (see "Pod lessons"). The rest of this section describes r8c itself.

The vSix6 recipe (actor 1024x3, SiLU critic 1536x2, `--v6`, lambda 0.8, lr 7.5e-5,
sizing-entropy scale 0.3, 15M rows per update at 220k envs, 10 setups per tier in the
tests; production used 30) **plus three changes**:

1. **All-actions gate term**, `--aa-coef 1.0`. At every decision the policy also follows
   the critic's value of fold, call and raise, not only the one action it took.
   (New code, default off: `.claude/tools/training_experiments/aa_coef.patch`, not yet
   in the main code.)
2. **The magnet**, `--kl-anchor-coef 0.1 --kl-anchor-ema 0.9`. Each update is pulled
   toward a running average of the recent policies, so steps are small and steady. The
   EMA moves once per update, so 0.9 means a reference about 10 updates behind; the
   default 0.999 is a nearly fixed anchor.
3. **Entropy stepped down**: 0.045 → 0.038 → 0.031 → 0.025 after 6, 12 and 18 updates
   (through the run's control file), then held at 0.025.

Results, from vSix6_1401 over 60 updates (the r8c continuations included):

| | sampled | argmax |
|---|---|---|
| mean over updates 4-20 / 24-40 / 44-60, vs the start | +0.08 / +0.16 / +0.19 | +0.10 / +0.13 / +0.16 |
| update 60 vs update 36, head-to-head on fresh deals | +0.055 (z 3.2) | +0.050 (z 4.1) |
| update 60 vs r7a (all-actions alone, leveled off) | +0.036 | +0.023 |
| updates 36 and 60 vs the live vSix6_1300, two deal seeds | +0.34 to +0.42 | +0.23 to +0.26 |

- It was **still climbing at update 60** (about +0.03 per 20 updates in both measures).
- The same recipe at **5M rows per update (r8g)** made the same progress per update in
  half the time.
- Health signals: gate entropy `Hg` about 0.37 and flat, KL per update about 0.0013.

## What works

| Change | Evidence | Notes |
|---|---|---|
| **All-actions gate term** `--aa-coef 1.0` | r7a: +0.13 / +0.11 within 4 updates, then flat (no slide) through 32; r7a_1416 vs live +0.34 / +0.28 | the gain is the critic's extra signal: as a pure noise cut (`--aa-cv`, same expected gradient) it gave nothing. **Use 1.0**; 2.0 is unstable (below) |
| **Magnet + entropy steps** | r6d (no all-actions): the only run still rising at update 24 (+0.09 / +0.10); with all-actions = r8c, the best | slow start. The magnet alone is too slow, and an entropy cut alone lowers argmax |
| **Cutting update noise** (rounds 3-5, vs the start; control −0.00 / +0.03) | entropy 0.06→0.045: +0.06 / +0.02; + lambda 0.95→0.8: +0.10 / +0.05; + lr 1.5e-4→7.5e-5: +0.12 / +0.05; + 30 setups per tier: +0.12 / +0.07 | "less noise = stronger". More setups per tier helps only once the other noise is cut |
| Much lower entropy than the old 0.25 (minimal-obs tuning, 2026-09-24) | argmax vs 0.25 over updates 64-79: 0.15 +0.24, 0.10 +0.29, 0.07 +0.38; no collapse at 0.07 | it learns better moves, not just sharper play. Only to a point: 0.03 hurt deep-tier argmax in the vSix line |
| Sizing-entropy scale below 1.0 | at 1.0 raise sizes never differentiated (size entropy ~1.98 of 2.40 max). Argmax vs vMin2 u150: scale 0.1 +0.57, scale 1.0 +0.43 | watch the size spread: 0.1 drifted toward min-raise over a long run (vMin3: 20% → 54%); vSix uses 0.3 |
| Full observations (engineered hand-strength features) | minimal-obs vMin3 was ~2 bb/seat-hand behind the full-obs live model despite 5x the rows | keep the full observations |
| Rebuilt critic | the old critic read values 43% low (a log-space mean) with 77% of its input units dead; a fresh SiLU critic beat it within ~1,500 small steps | needs `--q-fold-zero` (else the fold column drags the shared torso), plus `--critic-q-norm` and `--critic-v-raw` |
| Distilled smaller actor | a 1024x3 student of the 2048x4 held its strength and trained better (+0.37 / +0.14 vs +0.27–0.32 / 0.00) | actor capacity was not the limit |
| Averaging weights | avg_1380_1389 vs 1389: argmax +0.025; the average of 6 all-actions finals: argmax +0.147 vs its members' mean +0.125 | cheap and small, mostly argmax. Checkpoints from one start average fine |
| **Short updates** (5-15M rows) | 45M rows = the same progress per update as 15M (r7e); 5M ≈ 15M (r8e, r8g) at half the time. Production's 164M bought nothing per update | **contradicts the owner's "longer rollouts are always better" rule — the owner decides** |
| Pin each run to one NUMA node | unpinned runs spent 3x the kernel time | `taskset -c <that node's cpulist>` |
| **Obs-X `line,run`** (new inputs: all-in equity with runouts + betting-line summaries) | vs r8c at +32, seeds 7/13/17: +0.021 sampled / +0.025 top, all six positive; all four groups +0.014 / +0.030; `run` alone +0.011 / +0.025 (4 seeds) | `run` carries it (the critic uses it most); ~+60% time per update (its Monte Carlo); cheaper `line,range,pos` +0.007 / +0.023 at ~+20%. `docs/design/OBS_X_DIMS_2026-09-29.md` |

## What doesn't work — don't repeat without a new reason

| Tried | Result |
|---|---|
| Bigger or faster steps: 4x optimizer steps, lr 3e-4 / 5e-4 / 1.5e-3, a separate critic lr | worse, or swinging ±0.13–0.3 between checkpoints (self-play cycling); lr 5e-3 blew up the critic |
| More data per update (45M; production's 164M) | no more progress per update than 15M |
| More critic training (3 extra critic epochs) | nothing, with or without all-actions (r7c, r8f) |
| All-actions at 2.0 | same mean as 1.0, but it swung from −0.003 to +0.197 within 8 updates and gate entropy slid toward 0.30 (one-sided) |
| All-actions as a control variate (`--aa-cv`) | nothing |
| lr halving on top of all-actions | the same as all-actions alone. Without all-actions it only holds gains (r6b) |
| lr cuts + magnet + entropy steps without all-actions (r8a) | worse than each alone |
| Magnet alone; entropy cut alone | too slow; argmax falls |
| Production's old L2-init anchor (the round-2 student's weights) | a slight brake (sampled −0.03): re-anchor at restarts |
| 256 all-in runouts instead of 64 | nothing (deep-tier argmax −0.18, unexplained) |
| GAE lambda 0.6 / 1.0 | deep-tier argmax negative / worse |
| 4 PPO epochs instead of 2 | nothing |
| Entropy ≤ 0.03 (vSix) or ≥ 0.25 (minimal obs) | argmax hurt (deep tier) / far too random |
| Minimal observations | far behind the full observations (see above) |
| Shrinking the network to buy rollout length | size barely changes speed or memory (the rollout is CPU-bound; memory is the stored batch). Actor and critic widths 32-128 all matched after ~30 updates (smaller ones lag early); 16 was too small; the critic loss rises as the critic shrinks |
| Obs-X `line` alone / `pos` alone / `line,pos` (2026-09-29) | betting-line summaries alone −0.009 / −0.014 (3 seeds); who-acts-after ≈ 0; both −0.020 / +0.016 — `line` only helps together with `run` |
| Training on the site's 1024-sample opp-outcome features (M1) | +0.001 / +0.012 at the site's own conditions — within noise, +15-20% encode time |

## Why it stalls (the working theory)

Self-play PPO settles at a fixed point where each update is mostly noise:

- two one-update runs from the same weights, differing only in the seed, agree on their
  policy change at a correlation of about 0.13;
- consecutive updates point in opposite directions (cosine −0.35), so the weights
  random-walk.

More data per update doesn't help, because the noise isn't too few hands. It comes from
which setups and opponents each update sees, and from self-play chasing itself. Two
things do help:

- **smaller, damped steps**: lower lr, lambda and entropy, and the magnet;
- **more real signal per decision**: the all-actions term.

Each recipe change moves the fixed point: a jump within 4–8 updates, then a plateau.
Only the magnet runs kept climbing.

## How to test a training change (the method that held up)

1. **Start every candidate and a control from the same place**: same checkpoint, opponent
   pool and random stream (common random numbers). With no optimizer sidecar, use a cold
   Adam plus `--lr-warmup-updates 3`.
2. **Search scale**: 220k envs, 15M rows per update, 24 updates. Run 4–5 at once, one
   NUMA node each (about 10 minutes per update with 5 sharing, 3.6 alone).
3. **Score every 4 updates** against the start with `scripts/h2h_cross.py` (seed 7,
   `--deals 2048 --configs-per-tier 10`), sampled and `--greedy-a --greedy-b`. Compare
   means, trends and per-tier numbers. **For effects of a few hundredths, one seed is
   not enough (2026-09-29):** a matchup samples only 30 table setups, and that sample
   moves results by 0.02–0.05 between seeds — more than the per-hand se says, and
   averaging checkpoints scored on the SAME seed does not remove it. Confirm on 2–3
   seeds with 150 setups (`--configs-per-tier 50 --deals 512`; `h2h_x.py` prints the
   between-setup se). Seed 7 alone ranked the obs-X `line` group first and `run` flat;
   three more seeds reversed both.
4. **Prove a climb head-to-head**: a later checkpoint against an earlier one, on a fresh
   seed. Scores against a single fixed reference can flatten or mislead: r8c at update 60
   looked equal to update 36 against the live model, but won by +0.05 head-to-head.
5. **Re-score a promotion pick on a fresh deal seed**: the best of many on one seed is
   flattered by the choice.
6. **Read the train log**:
   - `Hg`, gate entropy: a steady slide means the policy is turning one-sided;
   - `kl` per update: magnet ~0.0013, all-actions ~0.002, all-actions at 2.0 ~0.004;
   - `v`: the critic loss;
   - `aa=`: the size of the all-actions signal.

   Sampled gains with a falling `Hg` are just sharpening; with a flat `Hg` they are real
   gains in how often it chooses each action.
7. **Numbered files `<stem>_<N>`**: train.py numbers from the 2nd update, and a warm start
   from a numbered file shifts the numbering by one. Compare a run's last numbered file
   with its main checkpoint's weights before calling it the final.

## Pod lessons

- **`/workspace` is a ~20 GB network-volume quota.** When it's full, trainers die
  silently. Measure it with `du -sb /workspace`; `df` shows the whole cluster.
- **Since 2026-09-28 ~16:40 UTC, reads from `/workspace` hang** and trainers stick in
  D-state. The fix is RunPod support or a new pod.
- **Workaround: a local-disk runtime at `/root/plx`**
  (`.claude/tools/training_experiments/setup_local.sh`). It is wiped if the pod
  restarts, so copy checkpoints and `.optim.pt` sidecars to the desktop with SHA-256
  checks.
- **ssh pitfalls**:
  - a `pgrep -f` / `pkill -f` pattern that also appears in the ssh command line matches
    your own shell, so write it as `[r]8c`;
  - `cd X && setsid … &` hangs the session; use `cd X; setsid nohup … &`.
- **Live control file**: one per run with
  `PLO5BP_ANNEAL_CONTROL=runs/<stem>.control.json`. It is read once per update, so a
  change lands one update late.
- **Maximum rollout = host RAM** (container limit 233.8 GiB; `--batch-on-host` keeps
  the GPU flat). Memory per stored row = the compact observation (1,172 B with
  `line,run`: 704 bits + 542 f16 — obs-X tail columns are all stored as f16, the unused
  groups included; 1,022 B without) + 113 B of other fields = **1,285 B/row**, plus the
  per-step pool, reused per setup (rollout / 30 setups x 2,264 B — float32 reals), plus
  ~19 GiB of baseline and PPO working memory. Measured on vSix7 (1.76M envs, 136M
  target -> 148.6M rows because each setup's in-flight hands finish past the target):
  RSS 201.5 GiB at the end of the rollout, **207 GiB peak** in the PPO phase, GPU 43.7 GiB
  reserved; rollout 30.3 min (82k rows/s, CPU-bound: 65% observation encode, 16% new
  hands), PPO 6.2 min. vSix6 (no obs-X) ran 150M there at 199 GiB first, then drifted to
  205 GiB over ~8 hours — allow ~6 GiB of drift, so ~92% of the limit is the ceiling.
  Levers not used (they change the recipe or the code): 30 setups per tier (pool share
  75 -> 25 B/row), an f16 pool, storing the X tail's 0/1 columns as bits and dropping its
  unused groups (~100 B/row).

## Untested ideas for the next plateau

- **Parallel copies**: several copies (one per NUMA node) with their weights averaged
  every N updates. Averaging has already helped the argmax a little.
- **A re-centering magnet**: a magnet whose reference re-centers on the current policy
  every N updates (magnet mirror descent / R-NaD, the family behind DeepMind's Stratego
  agent). It's the theory behind why the magnet keeps climbing.
- **Exploiters**: train a copy only to beat the main model, then add it to the main
  model's opponent pool so the main model must fix those leaks.
- **All-actions for bet sizes**: this needs a per-size Q; the critic's Q head pools all
  raise sizes into one column.
- **A long run of the r8c recipe at 5M rows per update**: r8g made the same progress
  per update in half the time.
- **Critic-only TRUE equity (E1)**: give the critic (which sees every hand) each seat's
  real runout equity given all hands. The critic latched onto even the random-hand
  `run` inputs hardest (column norm a third of a typical input's); with all-actions the
  critic's Q drives the policy directly. Needs a critic-only input path.

## Where things are

- **Obs-X code** (the 75-dim optional tail, `--obs-x-groups`, `scripts/pad_obs_x.py`,
  `scripts/h2h_x.py`): the worktree `C:\Users\themi\plodbnet-exp` (Rust + Python) and
  the pod's `/root/plx`; not in the main code or the site yet. Scores:
  `runs/roundx1_eval.jsonl`, `runs/roundx2_eval.jsonl`, `runs/roundx2m_eval.jsonl`,
  `runs/rr_x.jsonl`, `runs/wide_x.jsonl`; checkpoints `checkpoints/roundx1/`,
  `checkpoints/roundx2/`, `checkpoints/pod_archive/roundsx/`; production backups
  `checkpoints/vSix7/`.

- **`--aa-coef` code**: `.claude/tools/training_experiments/aa_coef.patch`, also in the
  worktree `C:\Users\themi\plodbnet-exp` and on the pod in `/root/plx`. It is not in the
  main code yet; port it default-off, then `scripts/exactness_check.py --recipe all`
  must stay IDENTICAL.
- **Scores**: `runs/round{3..8}_eval.jsonl`, `runs/round8_direct.jsonl`,
  `runs/promotion_eval.{log,jsonl}`. Summarize with
  `.claude/tools/training_experiments/summarize_round.py`.
- **Checkpoints**:
  - `checkpoints/round7/` and `checkpoints/round8/`: the scored ones;
  - `checkpoints/round8/resume/`: r8c and r8g with their optimizer sidecars;
  - `checkpoints/pod_archive/`: the earlier rounds.
- **Local helper scripts**: `.claude/tools/training_experiments/` (git-ignored; see its
  README).
