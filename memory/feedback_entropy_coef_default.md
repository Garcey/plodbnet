---
name: Default entropy_coef is 0.1 — never 0.01 again
description: For the 2048x4 architecture with Beta raise-size head, entropy_coef must default to 0.1; 0.01 lets the Beta head saturate and collapse H to 0 / negative
type: feedback
originSessionId: 7b62bece-8e86-454f-89e8-27566d72af24
---
The PPO `entropy_coef` default is `0.1` in both `config.py` (TrainingConfig)
and `scripts/train.py` (--entropy-coef argparse default). Do not silently
revert this to 0.01 or any other lower value, even if older code, comments,
or memory snippets reference 0.01.

**Why:** During a 2026-05-03 from-scratch c=2 run on the 2048x4 architecture,
H repeatedly slammed to 0 and went deeply negative (-0.31 on 3-handed deep
stacks at u180; H=0.000 with kl=0.0000 on multiple HU updates). The Beta
raise-size head saturated despite an aggression bonus c=2 explicitly
designed to keep the policy active. Root cause: `entropy_coef = 0.01` was
the default, providing too little entropy regularization to counter Beta
concentration on the larger network. The user said "I don't want this
mistake made again" and asked to bake 0.1 in as the default.

**How to apply:**
- When proposing or launching a new training run, you do NOT need to pass
  `--entropy-coef 0.1` explicitly — it's the default. But if you want to
  go LOWER than 0.1, the user must explicitly request it.
- If you ever see a code edit, plan, or documentation that proposes
  `entropy_coef=0.01` (or 0.05) for the 2048x4 net, flag it as wrong and
  cite this memory.
- The default is in two places that must stay in sync:
  `python/plo5bp/config.py` (`TrainingConfig.entropy_coef`) and
  `scripts/train.py` (the `--entropy-coef` argparse default + help text).
