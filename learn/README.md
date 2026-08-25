# The plo5dbbp training course

A comprehensive course on how this project's PLO5 double-board bomb-pot
model is trained — written for the owner of this project, who plays the
game far better than the code does but didn't write the code.

The course now has an inverted structure, by design:

## 1 · The master doc — the spine

**[MASTER.md](MASTER.md)** is the complete technical description of the
training system, dense and unsimplified: engine → observation encoding →
action space → networks → collection & advantages → the PPO update →
self-play → multi-config → ops → provenance. An hour-plus of reading,
organized in 10 parts (each part plays the role of a chapter). Every
claim is verified against the code; functions are named so you can jump
from prose to source.

**How to read it:** don't fight the density. Wherever a term is doing
heavy lifting, it links into the concept library — follow the link, get
the plain-English version, come back. Two passes through the master doc
with liberal link-following beats one careful pass.

## 2 · The concept library — the escape hatches

**[concepts/](concepts/)** — ~30 plain-English explainers, one concept
each (~2–3 minutes), written for exactly your level: ReLU, LayerNorm,
softmax, Beta distributions, gradients and backprop, Adam, the PPO clip,
GAE, KL divergence, EMA, mixed precision, self-play and exploitability,
and the rest. Each ends with "In this project:" bullets tying the idea
to this codebase. They're reference cards — read them as the master doc
sends you, or browse freely.

## 3 · The intro track — the gentle on-ramp

Written first, kept on purpose. If the master doc feels steep, read
these four (~10 minutes each) and then go back — they cover the core
territory at conversational pace, with poker-native framing:

1. **[One lap around the training loop](01-one-lap-around-the-training-loop.md)**
   — the whole machine; ends with reading a real log line. *(≈ Master
   Parts 0 & 5–6 at map level.)*
2. **[What the model sees](02-what-the-model-sees.md)** — the 1,171
   observation numbers. *(≈ Part 2.)*
3. **[What the model says](03-what-the-model-says.md)** — gate → rungs →
   slider → chips. *(≈ Part 3.)*
4. **[Inside the network](20-inside-the-network.md)** — layers, heads,
   and how each weight decides its change. *(≈ Parts 4 & 6.10.)*

More gentle chapters can be added on request (rewards & accounting,
advantages, entropy, self-play — the old chapter plan), but the master
doc + concept library is now the primary path.

## 4 · Labs — do-together exercises

Ask for these in a session when the corresponding master-doc part feels
solid:

- **Lab A** *(after Parts 0–3)*: decode the newest pod log line live;
  pick one hand in the study UI and predict which observation features
  drive the recommendation.
- **Lab B** *(after Parts 5–6)*: re-run the Q-head calibration audit on
  a current checkpoint and read the results together.
- **Lab C** *(after Parts 7–8)*: pit two checkpoints against fixed probe
  nodes and read the policy differences.
- **Lab D** *(after Part 9)*: trace `--entropy-coef` from the command
  line through config into the loss term, in the real files.
- **Lab E** *(after Part 10)*: pick one open v7 design question and
  argue both sides.

## 5 · Reference

- **[GLOSSARY.md](GLOSSARY.md)** — one line per term, in the order the
  intro track introduces them.

---

*Status 2026-07-23: master doc refreshed to live code (OBS 1171 / minimal
796, vSix4 + vMin1); concept library ~38 docs including the clipped-surrogate
teaching pack; intro chapters 1–3 + 20 fact-swept. Start at MASTER Part 6.0
for the phrase "clipped surrogate loss." Questions improve the writeups.*
