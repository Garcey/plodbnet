"""The PPO training driver, as a library (2026-09-28, ML-009).

`scripts/train.py` is a thin wrapper around `plo5bp.train.loop.main`. The code
used to live in that one 3,500-line script; it moved here VERBATIM, one
concern per module, so tests and tools import it instead of exec'ing the
script by file path:

    cli.py          the argument parser, the --v6 preset, flag parsing helpers
    tiers.py        the (seats, stacks) table-config samplers per stack tier
    control.py      the live control file (runs/<stem>.control.json) + the
                    block-rotation entropy anneal decisions
    checkpoint.py   atomic saves, the optimizer sidecar
    diagnostics.py  PLO5BP_DUMP_BATCH rollout dumps
    metrics.py      the 1 Hz resource sampler (--profile-one-update)
    loop.py         main(): setup, warm start, the update loop, the final save
"""
