"""Shared evaluation tooling (2026-09-28, ML-017 / ML-020 / ML-034 / ML-002).

The head-to-head, sharpness, utilization, SNR, probe and exploit scripts used to
each rebuild actors their own way (some with a silent 128x2 size default), skip
the observation-revision check, re-implement the batched duplicate-deal table
loop and load `_sample_game_config` from scripts/train.py by file path. One
library now:

    loader.py   load_actor / load_critic / checkpoint_meta: sizes read from the
                checkpoint (ckpt["arch"], else the weights), the EMA actor on
                request, and a REFUSED observation-revision mismatch.
    tables.py   the training tiers' table sampler (with a version tag written
                into every result), BatchedBombPotEnv.sizing(), the duplicate-
                deal match (`play_duplicate`) and its summary with an honest
                config-level standard error.
    sharpness.py  the policy-sharpness probe's states and measures.
"""

from plo5bp.evaluation.loader import (  # noqa: F401
    ObsRevMismatch,
    checkpoint_meta,
    load_actor,
    load_checkpoint,
    load_critic,
)
