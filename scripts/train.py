"""PPO training driver -- a thin wrapper (2026-09-28, ML-009).

The code lives in `python/plo5bp/train/` (cli, tiers, control, checkpoint,
diagnostics, metrics, loop); this script runs `plo5bp.train.loop.main`.
Budget: `--num-updates N`, or `--train-seconds S` (wall clock, wins when > 0).
Table configs: `--mix-configs` mixes `--configs-per-tier` (seats, stacks) draws
of every `--mix-tiers` tier per update (every production stem), else one
`--num-seats-range` / `--stack-dist` draw per update. Persistence:
`--checkpoint-every` / `--checkpoint-every-sec` numbered saves,
`--snapshot-every` / `--snapshot-every-sec` opponent-pool snapshots.

The names below stay importable from this file for tools that still load it
by path (prefer `from plo5bp.train... import ...`).
"""

from __future__ import annotations

from plo5bp.config import VARIANT_NLH, VARIANT_PLO4, VARIANT_PLO5, VARIANT_PLO6  # noqa: F401
from plo5bp.ppo import PPOTrainer  # noqa: F401
from plo5bp.train.checkpoint import (  # noqa: F401
    _UI_SERVED_CHECKPOINTS,
    _atomic_torch_save,
    _optimizer_sidecar_path,
    _restore_optimizer_sidecar,
    _save_optimizer_sidecar,
)
from plo5bp.train.cli import (  # noqa: F401
    EV_RUNOUT_SAMPLES,
    _V6_PRESET,
    _apply_v6_preset,
    _parse_mix_tiers,
    _parse_seats_range,
    _parse_stack_range,
    build_parser,
)
from plo5bp.train.control import (  # noqa: F401
    _CONTROL_WARNED,
    _apply_anneal_control,
    _control_number,
    _read_control_text,
    _warn_once,
)
from plo5bp.train.diagnostics import _dump_batch_diagnostics  # noqa: F401
from plo5bp.train.loop import _lr_warmup_scale, main  # noqa: F401
from plo5bp.train.tiers import (  # noqa: F401
    _KNOWN_STACK_DISTS,
    _VALID_STACK_DISTS,
    _sample_clubgg_seats,
    _sample_clubgg_stack_bb,
    _sample_game_config,
)

if __name__ == "__main__":
    main()
