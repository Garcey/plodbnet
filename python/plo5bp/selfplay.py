"""Opponent pool for self-play. Holds frozen snapshots sampled uniformly.

The pool is deliberately ephemeral run state (full model state dicts are
too heavy to ride in checkpoints), which historically meant every
stop/resume threw the opponents away and resumed against an EMPTY pool
until the next snapshot tick. `select_warmstart_pool_updates` +
`seed_pool_from_checkpoints` reconstruct the pool a never-stopped run
would have had, from the mid-run checkpoint files already on disk
(`<stem>_<update>.pt`): exact prior members when the resumed checkpoint
recorded them (`pool_member_updates`), otherwise the nearest available
files to the natural snapshot grid, walking further back when the disk
cadence is coarser than the snapshot cadence.
"""

from __future__ import annotations

import copy
import random
import re
from pathlib import Path
from typing import Any

from torch.profiler import record_function

from plo5bp.network import ActorCritic


class OpponentPool:
    """Fixed-capacity FIFO buffer of frozen policy state_dicts.

    `tags` mirrors `snapshots` 1:1 with the update index each member was
    snapshotted at (-1 when unknown) — metadata only, so resumed runs can
    persist `pool_member_updates` in checkpoints and later rebuild the
    exact membership. Collectors read `snapshots` / `sample()` and never
    see tags.
    """

    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self.snapshots: list[dict[str, Any]] = []
        self.tags: list[int] = []

    def _push(self, sd: dict[str, Any], tag: int) -> None:
        if len(self.snapshots) >= self.capacity:
            self.snapshots.pop(0)
            self.tags.pop(0)
        self.snapshots.append(sd)
        self.tags.append(int(tag))

    def snapshot(self, model: ActorCritic, tag: int = -1) -> None:
        with record_function("step14/pool_snapshot"):
            sd = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            self._push(sd, tag)

    def seed(self, state_dict: dict[str, Any], tag: int = -1) -> None:
        """Append an already-materialized (CPU) state dict — used by the
        warm-start reconstruction. Caller is responsible for architecture
        compatibility; entries are indistinguishable from `snapshot` ones."""
        self._push(dict(state_dict), tag)

    def sample(self) -> dict[str, Any] | None:
        if not self.snapshots:
            return None
        return copy.deepcopy(random.choice(self.snapshots))

    def __len__(self) -> int:
        return len(self.snapshots)


def select_warmstart_pool_updates(
    available: "list[int]",
    target_update: int,
    capacity: int,
    snapshot_every: int,
    preferred: "list[int] | None" = None,
) -> "list[int]":
    """Choose which on-disk checkpoint updates to seed the pool with,
    emulating the pool a never-stopped run would hold at `target_update`.

    - `available`: update numbers with a checkpoint file on disk.
    - `preferred`: the exact `pool_member_updates` recorded in the resumed
      checkpoint, honored first when those files still exist.
    - Otherwise members come from the natural snapshot grid (the last
      `capacity` multiples of `snapshot_every` at or below
      `target_update`), each mapped to the nearest unused available file
      (ties prefer the newer file). When the disk cadence is coarser than
      the snapshot cadence the grid walk continues further back, so the
      pool still fills to capacity with the most-recent distinct files.

    Returns update numbers OLDEST-FIRST (FIFO order: the next natural
    snapshot evicts the oldest member, exactly as an uninterrupted run
    would).
    """
    pool_of = sorted({int(a) for a in available if int(a) <= int(target_update)})
    if not pool_of or capacity <= 0:
        return []
    chosen: list[int] = []

    if preferred:
        avail_set = set(pool_of)
        for u in sorted({int(p) for p in preferred}, reverse=True):
            if u in avail_set and u not in chosen and len(chosen) < capacity:
                chosen.append(u)

    s = max(1, int(snapshot_every))
    g = (int(target_update) // s) * s
    remaining = [a for a in pool_of if a not in set(chosen)]
    while len(chosen) < capacity and remaining:
        best = min(remaining, key=lambda a: (abs(a - g), -a))
        chosen.append(best)
        remaining.remove(best)
        g -= s

    return sorted(chosen)


_CKPT_NUM_RE = re.compile(r"^(?P<base>.+)_(?P<num>\d+)\.pt$")


def discover_checkpoint_family(
    ckpt_path: "Path", directory: "Path | None" = None
) -> "tuple[str, dict[int, Path]]":
    """Find the numbered siblings of a checkpoint file.

    `<base>_<N>.pt` files sharing the loaded checkpoint's base stem are
    the family (`vFour4_500.pt` → base `vFour4`); a plain `<base>.pt`
    checkpoint uses its whole stem as the base. Returns (base, {N: path}).
    """
    ckpt_path = Path(ckpt_path)
    m = _CKPT_NUM_RE.match(ckpt_path.name)
    base = m.group("base") if m else ckpt_path.stem
    directory = Path(directory) if directory is not None else ckpt_path.parent
    family: dict[int, Path] = {}
    if directory.is_dir():
        for p in directory.glob(f"{base}_*.pt"):
            pm = _CKPT_NUM_RE.match(p.name)
            if pm and pm.group("base") == base:
                family[int(pm.group("num"))] = p
    return base, family


def seed_pool_from_checkpoints(
    pool: OpponentPool,
    ckpt_path: "Path",
    target_update: int,
    snapshot_every: int,
    expected_variant: str,
    expected_head_version: int,
    reference_state_dict: "dict[str, Any]",
    preferred: "list[int] | None" = None,
    directory: "Path | None" = None,
) -> "list[int]":
    """Reconstruct the opponent pool from a warm-start checkpoint's
    on-disk siblings. Returns the update numbers seeded (oldest-first).

    Each candidate must match the run's variant + head_version and have a
    model state dict with exactly the reference's keys and shapes —
    incompatible files are skipped with a warning rather than poisoning
    the pool. Loads one file at a time (peak memory ≈ one checkpoint over
    the pool's normal steady-state footprint).
    """
    import torch

    base, family = discover_checkpoint_family(ckpt_path, directory)
    chosen = select_warmstart_pool_updates(
        list(family.keys()),
        target_update,
        pool.capacity,
        snapshot_every,
        preferred=preferred,
    )
    ref_shapes = {k: tuple(v.shape) for k, v in reference_state_dict.items()}
    seeded: list[int] = []
    for u in chosen:  # oldest-first → natural FIFO order
        path = family[u]
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, ValueError) as e:
            print(f"[pool] skip {path.name}: unreadable ({e})")
            continue
        variant = str(ckpt.get("variant", "plo5_double_bomb"))
        head = int(ckpt.get("head_version", 1))
        sd = ckpt.get("model")
        if variant != expected_variant or head != expected_head_version:
            print(
                f"[pool] skip {path.name}: variant/head mismatch "
                f"({variant}, v{head})"
            )
            continue
        if not isinstance(sd, dict) or {
            k: tuple(v.shape) for k, v in sd.items()
        } != ref_shapes:
            print(f"[pool] skip {path.name}: model state shape mismatch")
            continue
        pool.seed({k: v.detach().clone().cpu() for k, v in sd.items()}, tag=u)
        seeded.append(u)
    return seeded
