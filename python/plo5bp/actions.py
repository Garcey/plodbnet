"""Discrete 8-action space (Rust mirror) + 3-way gate space for the
hybrid continuous-sizing policy head.

The discrete BetPctN enum is kept for UI/legacy tests; the training-time
policy head now emits a 3-way gate (Fold/CheckCall/Raise) with a separate
Beta-parameterised continuous chip amount for the Raise gate. Short
shoves are routed through the Raise gate at u=1; the engine's
`max_raise_chips` already clamps to stack, so a stack-bound shove is
just a Raise to the clamped max.
"""

from __future__ import annotations

import numpy as np

NUM_ACTIONS: int = 8

FOLD: int = 0
CHECK_CALL: int = 1
BET_PCT_10: int = 2
BET_PCT_25: int = 3
BET_PCT_50: int = 4
BET_PCT_75: int = 5
BET_PCT_100: int = 6
ALL_IN: int = 7

ACTION_NAMES: tuple[str, ...] = (
    "Fold",
    "CheckCall",
    "BetPct10",
    "BetPct25",
    "BetPct50",
    "BetPct75",
    "BetPct100",
    "AllIn",
)

GATE_ACTIONS: int = 3

GATE_FOLD: int = 0
GATE_CHECK_CALL: int = 1
GATE_RAISE: int = 2

GATE_NAMES: tuple[str, ...] = ("Fold", "CheckCall", "Raise")


def gate_mask_from_bounds(
    legal_mask: np.ndarray, max_raise_chips: np.ndarray | int
) -> np.ndarray:
    """Derive the 3-wide gate-legality mask from the engine's 8-wide
    discrete mask and the continuous `max_raise_chips` upper bound.

    - Fold/CheckCall gates mirror the discrete mask at their slots.
    - Raise gate is legal iff `max_raise_chips > 0`. The engine
      contract:
        - Normal raise: `min_raise > 0`, `max_raise = stack - call`,
          `legal[ALL_IN] = True`. GATE_RAISE legal.
        - Sub-min-raise short shove: `min_raise = 0`,
          `max_raise = stack - call > 0`, `legal[ALL_IN] = True`.
          GATE_RAISE legal — env redirects the dispatch via
          `apply(AllIn)` because `apply_raise_chips` rejects when
          `min == 0`. Chip amount is moot; treat as degenerate-range.
        - Reopen lockout / no raise: `max_raise = 0`. GATE_RAISE
          illegal regardless of `legal[ALL_IN]` (which may still be
          True as an all-in-for-less *call*, not a raise).

    Using `max_raise > 0` cleanly separates "can raise" from "ALL_IN
    is a legal call". A stack < bet_to_call player has
    `legal[ALL_IN] = True` but `max_raise = 0`; only fold/call is
    appropriate, never the raise gate.

    Shapes:
    - scalar: legal_mask (8,), max_raise_chips int → (3,) bool
    - batched: legal_mask (N, 8), max_raise_chips (N,) → (N, 3) bool
    """
    legal = np.asarray(legal_mask, dtype=bool)
    max_raise = np.asarray(max_raise_chips)
    if legal.ndim == 1:
        return np.array(
            [
                bool(legal[FOLD]),
                bool(legal[CHECK_CALL]),
                int(max_raise) > 0,
            ],
            dtype=bool,
        )
    fold = legal[..., FOLD]
    call = legal[..., CHECK_CALL]
    raise_ = max_raise > 0
    return np.stack([fold, call, raise_], axis=-1).astype(bool)
