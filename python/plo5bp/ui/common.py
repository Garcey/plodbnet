"""Session-free helpers shared by the study (`server.py`), trainer
(`trainer.py`), ranges and home-games state projections.

Everything here is pure: no module globals beyond constants, no FastAPI
imports. The study server keeps thin wrappers that close over its `session`
object; the trainer passes its own hand state explicitly.

One copy of each rule (review 2026-09-28, BE-010): gate slugs, the
amount-to-call, the anchor rows of a recommendation, card-list validation,
history rows, the UI-format -> engine-variant map and the per-format table
defaults all live here — they used to be copied between server.py and
trainer.py, and fixes had to be applied to every copy.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, Callable, Mapping

from pydantic import BaseModel, Field

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.config import GameConfig, VARIANT_NLH, VARIANT_PLO5

logger = logging.getLogger("plo5bp.ui.common")

# --- Environment flags ---------------------------------------------------------

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def env_flag(name: str, default: bool = False, environ: Mapping[str, str] | None = None) -> bool:
    """A boolean environment setting, parsed ONE way everywhere (BE-015).

    ``1/true/yes/on`` and ``0/false/no/off`` (any case, surrounding spaces
    ignored); unset or empty means ``default``. Anything else also means
    ``default`` and is logged once — ``PLO5BP_DEV_LOGIN=on`` used to be
    silently OFF because one module accepted "on" and another did not.
    ``environ`` = the mapping to read (default: the process environment; the
    app factory's settings parsers pass theirs — BE-007)."""
    raw = (os.environ if environ is None else environ).get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    if value:
        logger.warning(
            "environment %s=%r is not a boolean (use 1/0, true/false, yes/no,"
            " on/off) — using the default (%s)", name, raw, default,
        )
    return default


def model_slots(environ: Mapping[str, str] | None = None) -> int:
    """How many CPU-heavy model requests the public site runs at once
    (``PLO5BP_MODEL_SLOTS``; default 2-4 by core count). The access layer's
    work gate and the per-forward torch thread count both follow it."""
    raw = (os.environ if environ is None else environ).get("PLO5BP_MODEL_SLOTS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning("PLO5BP_MODEL_SLOTS=%r is not a number — using the default", raw)
    return max(2, min(4, os.cpu_count() or 2))


# --- Positions -------------------------------------------------------------------

#: Position labels by the number of players DEALT IN, in walk order from the
#: button (the engine's `(actor + 1) % n` direction). Heads-up the button posts
#: the small blind. Seven- and eight-handed tables exist in home games (PLO5 up
#: to 8 seats, PLO6 up to 7). (BE-005)
POSITIONS_BY_PLAYERS: dict[int, tuple[str, ...]] = {
    2: ("SB", "BB"),
    3: ("BTN", "SB", "BB"),
    4: ("BTN", "SB", "BB", "CO"),
    5: ("BTN", "SB", "BB", "UTG", "CO"),
    6: ("BTN", "SB", "BB", "UTG", "HJ", "CO"),
    7: ("BTN", "SB", "BB", "UTG", "MP", "HJ", "CO"),
    8: ("BTN", "SB", "BB", "UTG", "UTG+1", "MP", "HJ", "CO"),
}
#: Historical name of the table above (it used to be keyed by TABLE size).
POSITION_BY_SEAT_SHORT = POSITIONS_BY_PLAYERS

STREET_NAMES = {0: "preflop", 1: "flop", 2: "turn", 3: "river", 4: "showdown"}
AWAITING_NAMES = {1: "flop", 2: "turn", 3: "river"}

HISTORY_NAMES = (
    "Fold", "CheckCall", "BetPct10", "BetPct25", "BetPct50",
    "BetPct75", "BetPct100", "AllIn",
)


def effective_button(
    button: int,
    num_seats: int,
    in_hand: frozenset[int] | set[int] | None,
) -> int:
    """The seat that plays the button when the physical button is dead.

    The engine seats the first in-hand player clockwise of the button as
    first to act, so a button parked on a seat that was NOT dealt in (dead
    button — the player left / is sitting out) leaves the first in-hand
    seat COUNTER-clockwise from it acting last, i.e. in the button's
    position. Returns ``button`` unchanged when it is dealt in, or when
    there is no in-hand set to consult.
    """
    if not in_hand or button in in_hand:
        return button
    for offset in range(1, num_seats + 1):
        candidate = (button - offset) % num_seats
        if candidate in in_hand:
            return candidate
    return button


def position_name(
    seat: int,
    button: int | None,
    num_seats: int,
    in_hand: frozenset[int] | set[int] | None = None,
) -> str:
    """Position label of ``seat``: walk physical-clockwise from the button.

    Increasing seat index is physical-clockwise, matching the engine's
    `(actor + 1) % n` advancement. Only seats in ``in_hand`` (default: every
    seat) take part; any other seat is "OUT", so sit-outs never phantom-shift
    the labels. The label list is chosen by the number of players DEALT IN,
    not by the table size — four players read BTN/SB/BB/CO whether they sit
    at a 4-seat or a 6-seat table — and the walk starts at the seat that
    really plays the button (`effective_button`), so a dead button labels the
    last seat to act "BTN", not the first. (BE-005)
    """
    if button is None:
        return f"S{seat}"
    members = in_hand if in_hand else frozenset(range(num_seats))
    if seat not in members:
        return "OUT"
    names = POSITIONS_BY_PLAYERS.get(len(members))
    if names is None:
        return f"S{seat}"
    start = effective_button(button, num_seats, members)
    position_idx = 0
    for offset in range(num_seats):
        candidate = (start + offset) % num_seats
        if candidate not in members:
            continue
        if candidate == seat:
            return names[position_idx]
        position_idx += 1
    return f"S{seat}"


# --- Units / labels -------------------------------------------------------------------


def chips_to_bb(chips: int | float, bb: int) -> float:
    return float(chips) / float(bb)


def anchor_label(k: int) -> str:
    """Display label for sizing anchor k (pot fraction k/10). Anchor 0
    clamps to the min-raise floor — "min", never "0%"."""
    return "min" if k == 0 else f"{k * 10}%"


def anchor_label_spec(spec: Any, k: int) -> str:
    """Spec-aware anchor label: 'min' atom, pot-percent interior anchors,
    'ALL-IN' top atom (NLH). Reproduces `anchor_label` exactly on the PLO
    spec ('min', '10%', …, '100%')."""
    if k == 0:
        return "min"
    if spec.allin_atom and k == spec.count - 1:
        return "ALL-IN"
    pct = spec.fracs_pm[k] / 10.0
    return f"{pct:g}%"


def anchors_payload(
    spec: Any,
    anchor_probs: Any,
    anchor_chips: Any,
    anchor_legal: Any,
    bb: int,
) -> list[dict[str, Any]]:
    """Legal-only anchor rows for the client's bet curve, from spec-length
    parallel arrays (probability, raise-by chips, legality).

    ONE builder for every recommendation source (study, trainer, what-if,
    review) so the client never reconstructs rows (pot fractions, labels, the
    ALL-IN atom) from raw arrays. ``frac`` is None for the ALL-IN atom, whose
    chips are max_raise, not a pot fraction."""
    top = spec.count - 1
    return [
        {
            "k": int(k),
            "frac": (
                None
                if (spec.allin_atom and k == top)
                else spec.fracs_pm[k] / 1000.0
            ),
            "label": anchor_label_spec(spec, int(k)),
            "prob": round(float(anchor_probs[k]), 4),
            "chips": int(anchor_chips[k]),
            "chips_bb": round(chips_to_bb(int(anchor_chips[k]), bb), 4),
        }
        for k in range(spec.count)
        if bool(anchor_legal[k])
    ]


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


# --- Actions ----------------------------------------------------------------------------

GATE_SLUGS: dict[int, str] = {
    GATE_FOLD: "fold",
    GATE_CHECK_CALL: "check_call",
    GATE_RAISE: "raise",
}
GATE_NAME_TO_IDX: dict[str, int] = {v: k for k, v in GATE_SLUGS.items()}


class ActionRequest(BaseModel):
    """One action as the study and trainer APIs accept it."""

    gate: str = Field(..., pattern=r"^(fold|check_call|raise)$")
    chips: int | None = Field(default=None, ge=0)


def to_call_chips(obs: dict[str, Any], actor: int) -> int:
    """Chips ``actor`` needs to call, capped by their stack (an all-in call)."""
    current_commit = int(obs["street_commit"][actor])
    stack = int(obs["stacks"][actor])
    bet_to_call = int(obs["bet_to_call"])
    return min(max(0, bet_to_call - current_commit), stack)


# --- Table projection ---------------------------------------------------------------------
#
# (BE-008) The Study (`server._state_dict`) and the Trainer
# (`TrainerSession.project_state`) show the same table: seats, pot, the acting
# seat's buttons and raise window, the action history, the chip scale. That
# half of their state payload is built HERE, once; each mode then adds its
# own keys (cards, terminal, recommendation, the trainer block, ...).
# `test_trainer_session.py::test_state_shape_superset_of_study` still pins
# that the Trainer's payload carries every Study key.


def action_window(
    raw: dict[str, Any],
    actor: int | None,
    info: Any,
    bb: int,
    blocked_for: Callable[[int], bool] | None = None,
) -> tuple[dict[str, bool], dict[str, Any], int]:
    """``(legal, raise_bounds, to_call_chips)`` for the node a table shows.

    ``info`` is the engine's StepInfo for that node, or None when nobody can
    act there (every button off, an empty raise window). ``blocked_for(actor)``
    True greys every button out while keeping the window (Study: the hero's
    cards for the street are not entered yet)."""
    if actor is None or info is None:
        return (
            {"fold": False, "check_call": False, "raise": False},
            {"min_chips": 0, "max_chips": 0, "min_bb": 0.0, "max_bb": 0.0},
            0,
        )
    gm = info.gate_mask
    legal = {
        "fold": bool(gm[GATE_FOLD]),
        "check_call": bool(gm[GATE_CHECK_CALL]),
        "raise": bool(gm[GATE_RAISE]),
    }
    if blocked_for is not None and blocked_for(actor):
        legal = {k: False for k in legal}
    max_chips = int(info.max_raise_chips)
    min_chips = min(int(info.min_raise_chips), max_chips)
    # Short shove: the only legal raise is the all-in. Collapse the window to
    # one point so the UI renders an All-in button.
    if legal["raise"] and min_chips == 0 and max_chips > 0:
        min_chips = max_chips
    raise_bounds = {
        "min_chips": min_chips,
        "max_chips": max_chips,
        "min_bb": round(chips_to_bb(min_chips, bb), 4),
        "max_bb": round(chips_to_bb(max_chips, bb), 4),
    }
    return legal, raise_bounds, to_call_chips(raw, actor)


def seat_rows(
    raw: dict[str, Any],
    num_seats: int,
    bb: int,
    *,
    actor: int | None,
    hero_seat: int,
    position_of: Callable[[int], str],
    hole_of: Callable[[int], list[int] | None],
    participant_of: Callable[[int], bool] | None = None,
) -> list[dict[str, Any]]:
    """One row per seat: stack, commitments, flags and the cards the viewer
    may see (``hole_of``). ``participant_of`` (default: everyone) hides seats
    that were never dealt in (Study's live capture)."""
    rows: list[dict[str, Any]] = []
    for seat in range(num_seats):
        stack = int(raw["stacks"][seat])
        street_commit = int(raw["street_commit"][seat])
        rows.append({
            "seat": seat,
            "position": position_of(seat),
            "stack_chips": stack,
            "stack_bb": round(chips_to_bb(stack, bb), 4),
            "committed_this_street_bb": round(chips_to_bb(street_commit, bb), 4),
            "committed_total_bb": round(chips_to_bb(int(raw["total_commit"][seat]), bb), 4),
            "committed_this_street_chips": street_commit,
            "folded": bool(raw["folded"][seat]),
            "participant": True if participant_of is None else bool(participant_of(seat)),
            "all_in": bool(raw["all_in"][seat]),
            "is_actor": actor is not None and seat == actor,
            "is_hero": seat == hero_seat,
            "hole": hole_of(seat),
        })
    return rows


def table_state(
    raw: dict[str, Any],
    cfg: GameConfig,
    *,
    button_seat: int,
    hero_seat: int,
    info: Any,
    dollars_per_bb: float,
    position_of: Callable[[int], str],
    hole_of: Callable[[int], list[int] | None],
    participant_of: Callable[[int], bool] | None = None,
    blocked_for: Callable[[int], bool] | None = None,
) -> dict[str, Any]:
    """The shared table half of a Study / Trainer state payload.

    ``raw`` is the engine's ``observation_dict`` of the node shown (public
    fields are all it reads), ``info`` that node's StepInfo or None when
    nobody can act (see `action_window`)."""
    bb = int(cfg.bb)
    actor_raw = raw.get("actor")
    actor = int(actor_raw) if actor_raw is not None else None
    legal, raise_bounds, to_call = action_window(raw, actor, info, bb, blocked_for)
    pot = int(raw["pot"])
    # Settled pot ("Pot") = the pot gathered from finished streets: the total
    # minus this street's commits, which still sit in front of the seats.
    # `pot_chips` is the grand "Total Pot".
    settled = pot - sum(int(x) for x in raw["street_commit"])
    bet_to_call = int(raw["bet_to_call"])
    return {
        "num_seats": cfg.num_seats,
        "button_seat": button_seat,
        "hero_seat": hero_seat,
        "actor": actor,
        "seats": seat_rows(
            raw, cfg.num_seats, bb, actor=actor, hero_seat=hero_seat,
            position_of=position_of, hole_of=hole_of, participant_of=participant_of,
        ),
        "pot_chips": pot,
        "pot_bb": round(chips_to_bb(pot, bb), 4),
        "settled_pot_chips": settled,
        "settled_pot_bb": round(chips_to_bb(settled, bb), 4),
        "bet_to_call_chips": bet_to_call,
        "bet_to_call_bb": round(chips_to_bb(bet_to_call, bb), 4),
        "to_call_chips": int(to_call),
        "to_call_bb": round(chips_to_bb(int(to_call), bb), 4),
        "street": STREET_NAMES.get(int(raw["street"]), "flop"),
        "history": history_entries(raw, position_of, bb),
        "legal": legal,
        "raise_bounds": raise_bounds,
        "chip_scale": {
            "bb_chips": bb,
            "ante_chips": int(cfg.ante),
            "dollars_per_bb": float(dollars_per_bb),
        },
        "starting_stacks_chips": [int(s) for s in cfg.resolved_stacks],
        "starting_stacks_bb": [round(chips_to_bb(int(s), bb), 4) for s in cfg.resolved_stacks],
    }


# --- Formats ------------------------------------------------------------------------------

#: UI-only format id of the admin CANDIDATE model slot (historically the
#: "experimental" vMin1 ablation): PLO5 double-board bomb-pot rules, only the
#: served checkpoint differs. Not a GameConfig.variant.
FORMAT_EXPERIMENTAL = "experimental"
FORMAT_CANDIDATE = FORMAT_EXPERIMENTAL

#: UI format id -> engine GameConfig.variant for formats that are not an
#: engine variant themselves.
_FORMAT_ENGINE_VARIANT: dict[str, str] = {FORMAT_EXPERIMENTAL: VARIANT_PLO5}


def engine_variant(fmt_id: str) -> str:
    """The engine variant a UI format plays (the id itself for real variants)."""
    return _FORMAT_ENGINE_VARIANT.get(fmt_id, fmt_id)


#: The ONE table of per-game defaults the Study and Trainer tabs start from
#: (BE-019 — Study used to open PLO5 at 40bb and $2/bb while the Trainer dealt
#: 20bb at $20/bb). Keyed by ENGINE variant. `dollars_per_bb` is only the
#: display rate of the "$" unit (and the live-capture cents conversion of the
#: local build, which keeps its historical $2/bb).
FORMAT_DEFAULTS: dict[str, dict[str, float]] = {
    # PLO5 double-board bomb pot: GameConfig's own defaults (6-max, 20bb, 3bb ante).
    VARIANT_PLO5: {"num_seats": 6, "stack_bb": 20.0, "ante_bb": 3.0, "dollars_per_bb": 2.0},
    # NLH: the reference 5/10 table with a $5 ante (GameConfig.nlh_default).
    VARIANT_NLH: {"num_seats": 6, "stack_bb": 100.0, "ante_bb": 0.5, "dollars_per_bb": 10.0},
}


def format_defaults(fmt_id: str) -> dict[str, float]:
    """The defaults row for a UI format (via its engine variant)."""
    return FORMAT_DEFAULTS[engine_variant(fmt_id)]


def default_game_config(fmt_id: str, bb: int = 10_000) -> GameConfig:
    """The table a fresh Study session opens for a format."""
    d = format_defaults(fmt_id)
    # (cover_short_bets: the site offers the covering bet into a short stack's last
    # chips — GameConfig; owner, 2026-10-03)
    if engine_variant(fmt_id) == VARIANT_NLH:
        return dataclasses.replace(GameConfig.nlh_default(
            num_seats=int(d["num_seats"]),
            starting_stack=int(round(d["stack_bb"] * bb)),
        ), cover_short_bets=True)
    return GameConfig(
        num_seats=int(d["num_seats"]),
        starting_stack=int(round(d["stack_bb"] * bb)),
        ante=int(round(d["ante_bb"] * bb)),
        bb=bb,
        cover_short_bets=True,
    )
