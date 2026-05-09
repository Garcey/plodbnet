"""Action mask helpers. Thin wrapper so the network can mask invalid logits."""

from __future__ import annotations

import numpy as np
import torch


def apply_mask_to_logits(
    logits: torch.Tensor, mask: torch.Tensor, fill_value: float = -1e9
) -> torch.Tensor:
    """Replace entries where `mask` is False with `fill_value`.

    `logits` and `mask` must be broadcast-compatible. Mask dtype: bool.
    """
    return logits.masked_fill(~mask, fill_value)


def mask_from_list(mask: list[bool]) -> np.ndarray:
    return np.asarray(mask, dtype=bool)
