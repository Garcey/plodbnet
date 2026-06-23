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
    legal_mask: np.ndarray,
    max_raise_chips: np.ndarray | int,
    min_bet_chips: np.ndarray | int = 0,
) -> np.ndarray:
    """Derive the 3-wide gate-legality mask from the engine's 8-wide
    discrete mask, the continuous `max_raise_chips` upper bound, and the
    minimum full-size wager `min_bet_chips` (one big blind; pass it to
    enable the dust screen, below).

    - Fold/CheckCall gates mirror the discrete mask at their slots.
    - Raise gate is legal iff `max_raise_chips > 0` AND the raise is
      either a full-size wager (`max_raise >= min_bet_chips`, ≥ 1bb) or a
      genuine all-in shove (`legal[ALL_IN]`). The engine contract:
        - Normal raise: `min_raise > 0`, `max_raise = stack - call` ≥ 1bb,
          `legal[ALL_IN] = True`. GATE_RAISE legal.
        - Sub-min-raise short shove: `min_raise = 0`,
          `max_raise = stack - call > 0`, `legal[ALL_IN] = True`.
          GATE_RAISE legal via the all-in clause — env redirects the
          dispatch via `apply(AllIn)` because `apply_raise_chips` rejects
          when `min == 0`. Chip amount is moot; treat as degenerate-range.
        - Reopen lockout / no raise: `max_raise = 0`. GATE_RAISE illegal.
        - Cover-short DUST: the deepest alive opponent can be covered for
          only a sub-1bb amount, so the engine reports a tiny `max_raise`
          (the opponent's dust reach) with NO discrete bet/all-in action
          legal (`legal[ALL_IN] = False`). Without the dust screen this
          surfaced as a degenerate "bet $0.00" option. A sub-1bb,
          non-all-in cover is not a legal wager in standard poker anyway
          (you can only put in less than the min bet by going all-in), so
          GATE_RAISE is illegal here. NOTE: `min_bet_chips` defaults to 0
          (screen disabled, legacy behaviour) — callers that have the bb
          (the envs) pass it so the screen is active in training and UI.

    Using `max_raise > 0` still separates "can raise" from "ALL_IN is a
    legal call": a stack < bet_to_call player has `max_raise = 0`, so
    only fold/call is appropriate, never the raise gate.

    Shapes:
    - scalar: legal_mask (8,), max_raise_chips int, min_bet_chips int
      → (3,) bool
    - batched: legal_mask (N, 8), max_raise_chips (N,), min_bet_chips
      scalar or (N,) → (N, 3) bool
    """
    legal = np.asarray(legal_mask, dtype=bool)
    max_raise = np.asarray(max_raise_chips)
    min_bet = np.asarray(min_bet_chips)
    if legal.ndim == 1:
        can_raise = int(max_raise) > 0 and (
            int(max_raise) >= int(min_bet) or bool(legal[ALL_IN])
        )
        return np.array(
            [bool(legal[FOLD]), bool(legal[CHECK_CALL]), can_raise],
            dtype=bool,
        )
    fold = legal[..., FOLD]
    call = legal[..., CHECK_CALL]
    raise_ = (max_raise > 0) & ((max_raise >= min_bet) | legal[..., ALL_IN])
    return np.stack([fold, call, raise_], axis=-1).astype(bool)
