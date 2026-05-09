"""Opponent pool for self-play. Holds frozen snapshots sampled uniformly."""

from __future__ import annotations

import copy
import random
from typing import Any

from plo5bp.network import ActorCritic


class OpponentPool:
    """Fixed-capacity buffer of frozen policy state_dicts."""

    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self.snapshots: list[dict[str, Any]] = []

    def snapshot(self, model: ActorCritic) -> None:
        sd = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        if len(self.snapshots) >= self.capacity:
            self.snapshots.pop(0)
        self.snapshots.append(sd)

    def sample(self) -> dict[str, Any] | None:
        if not self.snapshots:
            return None
        return copy.deepcopy(random.choice(self.snapshots))

    def __len__(self) -> int:
        return len(self.snapshots)
