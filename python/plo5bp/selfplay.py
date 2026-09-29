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
    """Frozen policy state_dicts the learner plays against: a fixed-capacity
    FIFO of recent snapshots, plus -- optional, `set_anchors` -- ANCHORS: older
    members of the run (--pool-anchors, 2026-09-28, ML-033: the FIFO of 8
    snapshots every 5 updates only remembers ~40 updates, which lets
    self-play drift).

    `snapshots` = the FIFO members, then the anchors; `tags` mirrors it 1:1
    with the update each member was snapshotted at (-1 when unknown) —
    metadata only. `fifo_tags` (the FIFO part) is what checkpoints persist as
    `pool_member_updates`, so a resumed run rebuilds the exact FIFO; anchors
    are re-derived from the numbered checkpoints. Collectors read `snapshots`
    / `sample()` and never see tags.
    """

    def __init__(self, capacity: int = 16, seed: int | None = None):
        self.capacity = capacity
        self._fifo: list[dict[str, Any]] = []
        self._fifo_tags: list[int] = []
        self._anchors: list[dict[str, Any]] = []
        self._anchor_tags: list[int] = []
        # Serial-path sampling RNG, seeded per run (train.py passes the
        # run seed) so serial/eval rollouts reproduce. Previously sampled
        # via the GLOBAL unseeded `random` module (V5_DESIGN.md B8);
        # None preserves that OS-entropy behavior for ad-hoc callers.
        self._rng = random.Random(seed)

    @property
    def snapshots(self) -> list[dict[str, Any]]:
        return self._fifo + self._anchors

    @property
    def tags(self) -> list[int]:
        return self._fifo_tags + self._anchor_tags

    @property
    def fifo_tags(self) -> list[int]:
        return list(self._fifo_tags)

    @property
    def anchor_tags(self) -> list[int]:
        return list(self._anchor_tags)

    def set_anchors(self, members: "list[tuple[int, dict[str, Any]]]") -> None:
        """Replace the anchor members with `members` = [(tag, state_dict)]."""
        self._anchor_tags = [int(t) for t, _ in members]
        self._anchors = [sd for _, sd in members]

    def _push(self, sd: dict[str, Any], tag: int) -> None:
        if len(self._fifo) >= self.capacity:
            self._fifo.pop(0)
            self._fifo_tags.pop(0)
        self._fifo.append(sd)
        self._fifo_tags.append(int(tag))

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
        return copy.deepcopy(self._rng.choice(self.snapshots))

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
      pool still fills to capacity with the most-recent distinct files —
      but never past grid point 0: a run younger than
      `capacity * snapshot_every` updates gets the partially-filled pool it
      would really hold, not a back-fill of its earliest checkpoints.

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
    # Stop at grid point 0 (review 2026-09-20 A18): a never-stopped run has
    # no snapshot older than its first, so a YOUNG run's pool is simply not
    # full yet. The walk used to continue through negative grid points and
    # back-fill with the EARLIEST files on disk (files 5..100 every 5, target
    # 103, snapshot_every 50 -> the true pool {0, 50, 100} came back padded
    # with 10, 15, 20, 25, 30 — five near-random-init opponents). The
    # coarse-disk walk-back above is unaffected: it fills to capacity long
    # before the grid reaches 0 on any production cadence.
    while len(chosen) < capacity and remaining and g >= 0:
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
    expected_obs_rev: "int | None" = None,
) -> "list[int]":
    """Reconstruct the opponent pool from a warm-start checkpoint's
    on-disk siblings. Returns the update numbers seeded (oldest-first).

    Each candidate must match the run's variant + head_version and have a
    model state dict with exactly the reference's keys and shapes —
    incompatible files are skipped with a warning rather than poisoning
    the pool. Loads one file at a time (peak memory ≈ one checkpoint over
    the pool's normal steady-state footprint).

    `expected_obs_rev` (2026-09-20): the run's observation-SEMANTICS
    revision (`plo5bp.encoding.OBS_SEMANTICS_REV`). A sibling stamped with a
    different `obs_rev` (absent = 1, i.e. trained before the 2026-09-20
    feature fixes) was trained on other feature VALUES at the same dims —
    same shapes, so nothing else here would catch it — and is skipped. None
    = don't check (callers that predate the stamp).
    """
    import torch

    base, family = discover_checkpoint_family(ckpt_path, directory)
    if preferred:
        # A18: recorded members with no `<base>_<N>.pt` on disk (pruned, or
        # the checkpoint came from ANOTHER stem whose siblings live under a
        # different base) used to fall through to the grid without a word.
        gone = sorted({int(p) for p in preferred} - set(family))
        if gone:
            print(
                f"[pool] {len(gone)}/{len(set(preferred))} recorded pool "
                f"members have no {base}_<N>.pt on disk ({gone}) — filling "
                "from the snapshot grid instead"
            )
    chosen = select_warmstart_pool_updates(
        list(family.keys()),
        target_update,
        pool.capacity,
        snapshot_every,
        preferred=preferred,
    )
    seeded: list[int] = []
    for u in chosen:  # oldest-first → natural FIFO order
        sd = load_pool_member(
            family[u], expected_variant, expected_head_version,
            reference_state_dict, expected_obs_rev,
        )
        if sd is not None:
            pool.seed(sd, tag=u)
            seeded.append(u)
    return seeded


def load_pool_member(
    path: "Path",
    expected_variant: str,
    expected_head_version: int,
    reference_state_dict: "dict[str, Any]",
    expected_obs_rev: "int | None" = None,
) -> "dict[str, Any] | None":
    """The actor state dict of a checkpoint that may join this run's pool, or
    None (with a `[pool] skip` line) when it is unreadable or does not fit:
    another variant / head version / observation revision (absent = 1), or
    other parameter names or shapes than `reference_state_dict`."""
    import torch

    path = Path(path)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as e:
        print(f"[pool] skip {path.name}: unreadable ({e})")
        return None
    variant = str(ckpt.get("variant", "plo5_double_bomb"))
    head = int(ckpt.get("head_version", 1))
    sd = ckpt.get("model")
    if variant != expected_variant or head != expected_head_version:
        print(f"[pool] skip {path.name}: variant/head mismatch ({variant}, v{head})")
        return None
    obs_rev = int(ckpt.get("obs_rev", 1))
    if expected_obs_rev is not None and obs_rev != int(expected_obs_rev):
        print(
            f"[pool] skip {path.name}: obs_rev mismatch (file "
            f"{obs_rev} vs run {int(expected_obs_rev)})"
        )
        return None
    ref_shapes = {k: tuple(v.shape) for k, v in reference_state_dict.items()}
    if not isinstance(sd, dict) or {k: tuple(v.shape) for k, v in sd.items()} != ref_shapes:
        print(f"[pool] skip {path.name}: model state shape mismatch")
        return None
    return {k: v.detach().clone().cpu() for k, v in sd.items()}


def select_anchor_updates(
    available: "list[int]",
    current_update: int,
    ages: "list[int]",
    snapshot_every: int,
    exclude: "set[int] | None" = None,
) -> "list[int]":
    """The pool ANCHORS for `current_update` (--pool-anchors): for each age,
    the available numbered checkpoint nearest to current_update - age (the
    target rounded down to the snapshot grid, so the choice only moves every
    `snapshot_every` updates), never a future one, none twice, none already
    in the FIFO (`exclude`), none past the run's start (a young run simply
    has fewer anchors). Returns update numbers oldest-first."""
    s = max(1, int(snapshot_every))
    pool_of = sorted({int(a) for a in available if int(a) <= int(current_update)})
    skip = set(exclude or ())
    chosen: list[int] = []
    for age in sorted({int(a) for a in ages if int(a) > 0}):
        target = ((int(current_update) - age) // s) * s
        if not pool_of or target < pool_of[0]:
            continue
        cands = [a for a in pool_of if a not in skip and a not in chosen]
        if not cands:
            break
        chosen.append(min(cands, key=lambda a: (abs(a - target), a)))
    return sorted(chosen)


def refresh_pool_anchors(
    pool: OpponentPool,
    ckpt_path: "Path",
    current_update: int,
    ages: "list[int]",
    snapshot_every: int,
    expected_variant: str,
    expected_head_version: int,
    reference_state_dict: "dict[str, Any]",
    expected_obs_rev: "int | None" = None,
    directory: "Path | None" = None,
) -> "list[int]":
    """Point `pool`'s anchors at `select_anchor_updates` of the run's numbered
    checkpoints (`<stem>_<N>.pt` beside `ckpt_path`); an anchor that stays
    chosen keeps its loaded weights, only newly chosen files are read.
    Returns the anchor tags."""
    _, family = discover_checkpoint_family(ckpt_path, directory)
    chosen = select_anchor_updates(
        list(family), current_update, ages, snapshot_every, exclude=set(pool.fifo_tags)
    )
    if chosen == pool.anchor_tags:
        return chosen
    have = dict(zip(pool.anchor_tags, pool._anchors))
    members: list[tuple[int, dict[str, Any]]] = []
    for u in chosen:
        sd = have.get(u)
        if sd is None:
            sd = load_pool_member(
                family[u], expected_variant, expected_head_version,
                reference_state_dict, expected_obs_rev,
            )
        if sd is not None:
            members.append((u, sd))
    pool.set_anchors(members)
    return pool.anchor_tags
