"""Session-free helpers shared by the study (`server.py`) and trainer
(`trainer.py`) state projections.

Everything here is pure: no module globals beyond constants, no
FastAPI imports. The study server keeps thin wrappers that close over
its `session` object; the trainer passes its own hand state explicitly.
"""

from __future__ import annotations

from typing import Any, Callable

POSITION_BY_SEAT_6: dict[int, str] = {
    0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "HJ", 5: "CO",
}
POSITION_BY_SEAT_SHORT = {
    2: ("SB", "BB"),
    3: ("BTN", "SB", "BB"),
    4: ("BTN", "SB", "BB", "CO"),
    5: ("BTN", "SB", "BB", "UTG", "CO"),
    6: ("BTN", "SB", "BB", "UTG", "HJ", "CO"),
}

STREET_NAMES = {0: "preflop", 1: "flop", 2: "turn", 3: "river", 4: "showdown"}
AWAITING_NAMES = {2: "turn", 3: "river"}

HISTORY_NAMES = (
    "Fold", "CheckCall", "BetPct10", "BetPct25", "BetPct50",
    "BetPct75", "BetPct100", "AllIn",
)


def position_name(
    seat: int,
    button: int | None,
    num_seats: int,
    in_hand: frozenset[int] | set[int] | None = None,
) -> str:
    """Resolve position label by walking physical CW from the button.

    Increasing seat index is physical-clockwise, matching the engine's
    `(actor + 1) % n` advancement, so position labels walk
    `(button + offset) % n`. Seats not in `in_hand` are skipped during
    the walk (and labeled "OUT") so sit-outs don't phantom-shift labels.
    """
    names = POSITION_BY_SEAT_SHORT.get(num_seats)
    if names is None or button is None:
        return f"S{seat}"
    members = in_hand if in_hand else frozenset(range(num_seats))
    if seat not in members:
        return "OUT"
    position_idx = 0
    for offset in range(num_seats):
        candidate = (button + offset) % num_seats
        if candidate not in members:
            continue
        if candidate == seat:
            return names[position_idx] if position_idx < len(names) else f"S{seat}"
        position_idx += 1
    return f"S{seat}"


def chips_to_bb(chips: int | float, bb: int) -> float:
    return float(chips) / float(bb)


def history_entries(
    obs: dict[str, Any],
    position_of: Callable[[int], str],
    bb: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for seat, action_idx, chips, street_idx in obs.get("history", []):
        entries.append({
            "seat": int(seat),
            "position": position_of(int(seat)),
            "action": HISTORY_NAMES[int(action_idx)],
            "chips": int(chips),
            "chips_bb": round(chips_to_bb(int(chips), bb), 4),
            "street": STREET_NAMES.get(int(street_idx), str(street_idx)),
        })
    return entries


def validate_card_list(
    xs: list[int | None], length: int, name: str
) -> list[int | None]:
    """Length/range-check a nullable card list. Raises ValueError (callers
    wrap into their transport's error type)."""
    if len(xs) != length:
        raise ValueError(f"{name} must be length {length}, got {len(xs)}")
    out: list[int | None] = []
    for x in xs:
        if x is None:
            out.append(None)
        else:
            xi = int(x)
            if not (0 <= xi < 52):
                raise ValueError(f"{name}: card {xi} out of range [0,51]")
            out.append(xi)
    return out
