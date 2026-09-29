"""Offline-diagnostics dumps of a collected rollout."""

from __future__ import annotations

import os

import torch


def _dump_batch_diagnostics(batch, path: str, max_rows: int = 2_000_000) -> None:
    """`PLO5BP_DUMP_BATCH=<file.npz>` (2026-09-26, the regression diagnosis): a
    random sample of the collected rollout -- the critic's value at sampling
    time, the returns (Monte-Carlo returns with --gae-lambda 1.0), the gate
    action + legality, the normalized advantage, and the street / pot / hero
    stack columns of the FULL obs layout -- for offline critic and policy
    diagnostics. train.py then exits without updating.

    The observation columns come from the encoder's own offsets (2026-09-28,
    ML-049: they were magic numbers, and the "full layout" guard let NLH's
    differently laid out rows through). The full and the minimal PLO layouts
    share them; any other width is refused. `bet_to_call_bb` is the street's
    highest commitment (what the BET is), not what the hero still owes -- it
    was mislabelled `to_call_bb`; `hero_stack_bb` is the hero's EFFECTIVE stack
    (chips no opponent can reach are not counted)."""
    import numpy as _np

    from plo5bp.encoding import (
        _NUM_STREET_ONEHOT,
        _SCALARS_OFF,
        _STACKS_OFF,
        _STREET_OFF,
        OBS_DIM,
        OBS_DIM_MINIMAL,
    )

    width = int(batch.obs.shape[1])
    if width not in (OBS_DIM, OBS_DIM_MINIMAL):
        raise SystemExit(
            f"PLO5BP_DUMP_BATCH: a {width}-wide observation is not a PLO layout "
            f"this dump describes (full {OBS_DIM} / minimal {OBS_DIM_MINIMAL})"
        )
    n = int(batch.values.shape[0])
    idx = _np.sort(_np.random.default_rng(0).choice(n, size=min(n, max_rows), replace=False))
    it = torch.as_tensor(idx)
    obs = batch.obs[it].float().cpu().numpy()
    street = obs[:, _STREET_OFF:_STREET_OFF + _NUM_STREET_ONEHOT].argmax(axis=1)
    out = {
        "values": batch.values[it].float().cpu().numpy(),
        "returns": batch.returns[it].float().cpu().numpy(),
        "advantages": batch.advantages[it].float().cpu().numpy(),
        "gate_actions": batch.gate_actions[it].cpu().numpy(),
        "gate_masks": batch.gate_masks[it].cpu().numpy(),
        "anchor_actions": batch.anchor_actions[it].cpu().numpy(),
        "street": street,  # 0 preflop, 1 flop, 2 turn, 3 river
        "pot_bb": obs[:, _SCALARS_OFF + 0],
        "bet_to_call_bb": obs[:, _SCALARS_OFF + 1],
        "hero_stack_bb": obs[:, _STACKS_OFF + 0],  # hero-rotated slot 0
        "rows_total": _np.asarray([n]),
    }
    if getattr(batch, "ent_coef_rows", None) is not None:
        out["ent_coef"] = batch.ent_coef_rows[it].float().cpu().numpy()
    if getattr(batch, "is_terminal", None) is not None:
        out["is_terminal"] = batch.is_terminal[it].cpu().numpy()
    out["opp_holes"] = batch.opp_holes[it].cpu().numpy()
    out["sizing"] = batch.sizing[it].cpu().numpy()
    out["old_gate_logp"] = batch.old_gate_logp[it].float().cpu().numpy()
    _np.savez_compressed(path, **out)
    # The observations themselves (float16, exact for the 0/1 columns) for
    # offline critic experiments: PLO5BP_DUMP_OBS_ROWS rows (default 3M) of the
    # same sample, in its order, as <path>.obs16.npy.
    k = min(len(idx), int(os.environ.get("PLO5BP_DUMP_OBS_ROWS", "3000000")))
    if k > 0:
        obs16 = _np.lib.format.open_memmap(
            path + ".obs16.npy", mode="w+", dtype=_np.float16, shape=(k, obs.shape[1])
        )
        obs16[:] = obs[:k].astype(_np.float16)
        obs16.flush()
