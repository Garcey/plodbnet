"""PokerNow DOM-snapshot → `FrameState` mapper.

PokerNow (pokernow.com / .club) is a browser web app, so — unlike ClubGG,
which needs pixel OCR because it's a WDA-protected Unity window — its entire
game state is readable straight from the DOM. A small Tampermonkey userscript
(``tools/pokernow/pokernow.user.js``) snapshots the table on every change and
streams a normalized JSON payload to the server's ``/pokernow/ingest``
websocket. This module turns one such payload into the *same* `FrameState`
the OCR pipeline produces, so it can ride the existing
`EventReconstructor` → `Session` → `_rebuild_env` machinery unchanged.

The payload is exact (PokerNow renders cards/stacks/bets/actor/fold as DOM
elements), so the resulting FrameState carries authoritative
``committed_chips`` / ``stack_chips`` and a single-seat ``is_actor`` — the
reconstructor's OCR-disambiguation fallbacks simply never fire.

Unit convention
---------------
Like the OCR path, ``FrameState`` chip fields are in **cents** (PokerNow's
dollar display × 100, so "$74.00" → 7400). The server converts cents →
engine-chips at the boundary via ``_ocr_cents_to_engine_chips``. Keep this
mapper in cents; don't reach into engine units here.

Seat ordering
-------------
PokerNow's physical seat numbers (``table-player-{N}``, N=1..10) do NOT encode
clockwise order. The userscript measures each seat's on-screen angle around
the table center; this mapper sorts those angles clockwise from hero to assign
the engine's ``hero=0, then clockwise`` seat convention. Only seats that are
part of the current hand (have cards, or have folded out of it) are included,
so ``num_seats`` reflects the actual table size for the hand.

Payload contract (``pokernow.v1``) — validated, review 2026-09-20
-----------------------------------------------------------------
``map_payload`` is the trust boundary for JSON that arrives over HTTP, so it
checks the shape up front and raises `PokerNowPayloadError` (a ``ValueError``)
with the offending path — the server should answer 400. It used to let
garbage through to fail deep inside (``int(None)`` → TypeError, ``inf`` →
OverflowError → a 500 on every heartbeat), and a payload in the pre-v1 key
spelling (``stack``/``bet``/``pot``) mapped "successfully" to a table where
every player was all-in for an unknown amount.

    seats:       list, required. Each seat is an object with
      seat          int (PokerNow physical seat), required, unique
      cards         list of card strings / nulls (``[]`` = none), required
      stackDollars  finite number >= 0, or null — key required
      betDollars    finite number >= 0, or null — key required
      betText       string or null (``"check"`` = 0 this street)
      folded, isHero, isActor, allIn   booleans (absent = false)
      angleCW       finite number (absent = 0)
      name          string or null
    boards:      list of {run, cards} (absent = no board yet)
    button:      null | {"seat": int} | int
    heroCards:   list or null
    potDollars:  finite number >= 0, or null
    schema:      "pokernow.v1" when present

Unparseable *card strings* stay lenient (→ face-down): PokerNow's card DOM is
the one place a mid-render snapshot produces junk, and a dropped frame is worse
than an unknown card.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from plo5bp.ocr.types import Card, FrameState, SeatObs

SCHEMA = "pokernow.v1"


class PokerNowPayloadError(ValueError):
    """The payload is not a well-formed ``pokernow.v1`` snapshot.

    Subclasses ``ValueError`` so callers can turn it into an HTTP 400
    without importing this module's types.
    """


@dataclass(frozen=True)
class PokerNowFrame:
    """Result of mapping a PokerNow payload.

    `frame` is the OCR-compatible `FrameState`. `num_seats` is the number of
    in-hand seats (the engine seat count for this hand). `seat_names` maps an
    engine seat index → the PokerNow player name (for the UI). `physical_to_engine`
    maps PokerNow's physical seat number → engine seat index. `bomb_pot` /
    `variant` echo the table flags.
    """

    frame: FrameState
    num_seats: int
    seat_names: dict[int, str]
    physical_to_engine: dict[int, int]
    bomb_pot: bool
    variant: str


def _bad(where: str, problem: str, value) -> PokerNowPayloadError:
    shown = repr(value)
    if len(shown) > 80:  # the message ends up in a 400 body / log line
        shown = shown[:77] + "..."
    return PokerNowPayloadError(f"pokernow payload: {where} {problem} (got {shown})")


def _req_int(v, where: str) -> int:
    # bool is an int subclass; JSON true/false is never a seat number.
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise _bad(where, "must be an integer", v)
    if isinstance(v, float) and (not math.isfinite(v) or v != int(v)):
        raise _bad(where, "must be an integer", v)
    return int(v)


def _opt_amount(v, where: str) -> float | None:
    """A dollar amount: finite number >= 0, or None."""
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise _bad(where, "must be a number or null", v)
    f = float(v)
    if not math.isfinite(f) or f < 0:
        raise _bad(where, "must be a finite number >= 0", v)
    return f


def _opt_cards(v, where: str) -> list:
    if v is None:
        return []
    if not isinstance(v, (list, tuple)):
        raise _bad(where, "must be a list of card strings/nulls", v)
    for i, c in enumerate(v):
        if c is not None and not isinstance(c, str):
            raise _bad(f"{where}[{i}]", "must be a card string or null", c)
    return list(v)


def _validate_seat(seat, idx: int) -> dict:
    where = f"seats[{idx}]"
    if not isinstance(seat, dict):
        raise _bad(where, "must be an object", seat)
    # Key PRESENCE is what tells a v1 payload from the pre-v1 spelling
    # (`stack` / `bet`): with `.get()` alone those read as "no stack, no bet".
    for key in ("seat", "cards", "stackDollars", "betDollars"):
        if key not in seat:
            raise PokerNowPayloadError(
                f"pokernow payload: {where} is missing required key {key!r} "
                f"(keys present: {sorted(map(str, seat))})"
            )
    _req_int(seat["seat"], f"{where}.seat")
    _opt_cards(seat["cards"], f"{where}.cards")
    _opt_amount(seat["stackDollars"], f"{where}.stackDollars")
    _opt_amount(seat["betDollars"], f"{where}.betDollars")
    text = seat.get("betText")
    if text is not None and not isinstance(text, str):
        raise _bad(f"{where}.betText", "must be a string or null", text)
    ang = seat.get("angleCW")
    if ang is not None and (
        isinstance(ang, bool)
        or not isinstance(ang, (int, float))
        or not math.isfinite(float(ang))
    ):
        raise _bad(f"{where}.angleCW", "must be a finite number", ang)
    return seat


def _validate_payload(payload) -> None:
    if not isinstance(payload, dict):
        raise _bad("payload", "must be an object", payload)
    schema = payload.get("schema")
    if schema is not None and schema != SCHEMA:
        raise _bad("schema", f"must be {SCHEMA!r}", schema)
    if not isinstance(payload.get("seats"), list):
        raise _bad("seats", "must be a list", payload.get("seats"))
    seen: set[int] = set()
    heroes = 0
    for idx, seat in enumerate(payload["seats"]):
        _validate_seat(seat, idx)
        phys = int(seat["seat"])
        if phys in seen:
            raise _bad(f"seats[{idx}].seat", "duplicates an earlier seat", phys)
        seen.add(phys)
        heroes += 1 if seat.get("isHero") else 0
    if heroes > 1:
        raise _bad("seats", "must flag at most one isHero seat", heroes)

    boards = payload.get("boards")
    if boards is not None:
        if not isinstance(boards, list):
            raise _bad("boards", "must be a list", boards)
        for i, b in enumerate(boards):
            if not isinstance(b, dict):
                raise _bad(f"boards[{i}]", "must be an object", b)
            _opt_cards(b.get("cards"), f"boards[{i}].cards")
    _opt_cards(payload.get("heroCards"), "heroCards")
    _opt_amount(payload.get("potDollars"), "potDollars")

    button = payload.get("button")
    if isinstance(button, dict):
        if button.get("seat") is not None:
            _req_int(button["seat"], "button.seat")
    elif button is not None:
        _req_int(button, "button")


def _dollars_to_cents(v) -> int | None:
    if v is None:
        return None
    return int(round(float(v) * 100))


def _parse_card(s: str | None) -> Card | None:
    if not s:
        return None
    try:
        return Card.parse(s)
    except (ValueError, KeyError):
        return None


def _parse_cards(seq, n: int = 5) -> tuple[Card | None, ...]:
    """Parse a card-string list into a fixed-width tuple, padded with None.

    A folded seat reports an empty list; a face-down seat reports n nulls;
    a revealed hand reports n card strings. All normalize to width n.
    """
    seq = list(seq or [])
    out: list[Card | None] = [_parse_card(seq[i]) if i < len(seq) else None for i in range(n)]
    return tuple(out)


def _is_in_hand(seat: dict) -> bool:
    """A seat is part of the current hand if it holds cards or has folded.

    Folded seats keep their engine index for the rest of the hand (PokerNow
    removes their card elements, so ``cards`` empties to ``[]`` but ``folded``
    is set). Empty / sitting-out seats have neither and are excluded so they
    don't inflate the engine seat count.
    """
    if seat.get("folded"):
        return True
    return len(seat.get("cards") or []) > 0


def _engine_order(seats: list[dict]) -> list[dict]:
    """Order in-hand seats hero=0 then clockwise, by on-screen angle.

    ``angleCW`` is degrees measured clockwise from 12 o'clock (supplied by the
    userscript's geometry read). Sorting by ``(angle - hero_angle) mod 360``
    puts hero first (offset 0) and walks clockwise around the table — matching
    the engine's ``(actor + 1) % n`` advancement direction.
    """
    hero = next((s for s in seats if s.get("isHero")), None)
    if hero is None:
        # No hero seat flagged — fall back to physical-seat order so the
        # caller still gets a deterministic (if unrotated) layout.
        return sorted(seats, key=lambda s: s.get("seat", 0))
    # `or 0.0`: the key may be present with an explicit null.
    hero_ang = float(hero.get("angleCW") or 0.0)
    return sorted(
        seats, key=lambda s: (float(s.get("angleCW") or 0.0) - hero_ang) % 360.0
    )


def _seat_committed_cents(seat: dict) -> int | None:
    """This-street committed chips, in cents.

    PokerNow shows the seat's cumulative street commitment in
    ``.table-player-bet-value`` — a numeric amount (bet/raise/call/ante) or the
    literal verb "check". A numeric value maps to cents; "check" means zero
    contributed this street; anything else (no bet element) is unknown (None),
    leaving the reconstructor to infer from the actor signal.
    """
    bet = seat.get("betDollars")
    if bet is not None:
        return _dollars_to_cents(bet)
    text = (seat.get("betText") or "").strip().lower()
    if text == "check":
        return 0
    return None


def map_payload(payload: dict) -> PokerNowFrame:
    """Map a ``pokernow.v1`` DOM snapshot to a `PokerNowFrame`.

    Raises `PokerNowPayloadError` (a ``ValueError``) on a malformed payload —
    see "Payload contract" in the module docstring.
    """
    _validate_payload(payload)
    raw_seats = [s for s in payload["seats"] if _is_in_hand(s)]
    ordered = _engine_order(raw_seats)

    physical_to_engine: dict[int, int] = {}
    seat_names: dict[int, str] = {}
    seat_obs: list[SeatObs] = []
    for eng_idx, s in enumerate(ordered):
        phys = int(s.get("seat"))
        physical_to_engine[phys] = eng_idx
        seat_names[eng_idx] = str(s.get("name") or "")
        stack_cents = _dollars_to_cents(s.get("stackDollars"))
        # All-in ONLY when the DOM says so (review 2026-09-20). This used to
        # also infer it from a null stack on a non-folded seat, but a missing
        # `.normal-value` is just as often a mid-render snapshot or a seat in
        # some other non-numeric state — and "all-in" is not a harmless
        # default: it becomes stack 0, i.e. a full-stack DROP that corroborates
        # whatever the bet oval shows, and it routes the action through the
        # short-shove / all-in-call paths. The userscript (>= 1.2.0) sets
        # `allIn` from the stack element's "All In" text; older scripts still
        # send their own `stack === null && !folded` inference in the same
        # field, so they behave exactly as before. A folded seat is never
        # all-in (its stack text is irrelevant to the hand).
        all_in = bool(s.get("allIn")) and not bool(s.get("folded"))
        # An all-in player has $0 behind — PokerNow shows "all in" text instead
        # of a number, which reads as a null stack. Map it to 0, NOT None: the
        # reconstructor's corroboration guard needs the stack DROP to back the
        # shove's committed amount, and it also routes a stack→0 raise through
        # the engine's short-shove path. Leaving it None makes the guard discard
        # the all-in bet, collapsing the hand to a phantom check-down/showdown.
        # Unknown (null, not all-in) stays None: the reconstructor carries the
        # previous stack forward instead of inventing a drop.
        stack_chips = 0 if all_in else stack_cents
        seat_obs.append(
            SeatObs(
                seat=eng_idx,
                stack_chips=stack_chips,
                committed_chips=_seat_committed_cents(s),
                folded=bool(s.get("folded")),
                all_in=all_in,
                is_actor=bool(s.get("isActor")),
                bet_banner=False,
            )
        )

    # Button: translate PokerNow's physical dealer seat to its engine index.
    # If the button seat isn't among the in-hand seats (rare timing), leave None.
    button = payload.get("button") or {}
    button_phys = button.get("seat") if isinstance(button, dict) else button
    button_seat = physical_to_engine.get(int(button_phys)) if button_phys is not None else None

    # Boards: run-1 → board_a, run-2 → board_b.
    boards = {str(b.get("run")): b.get("cards") for b in (payload.get("boards") or [])}
    board_a = _parse_cards(boards.get("1"))
    board_b = _parse_cards(boards.get("2"))

    # Hero hole: prefer the top-level heroCards, else the hero seat's cards.
    hero_cards = payload.get("heroCards")
    if hero_cards is None:
        hero_seat = next((s for s in ordered if s.get("isHero")), None)
        hero_cards = hero_seat.get("cards") if hero_seat else None
    hero_hole = _parse_cards(hero_cards)

    frame = FrameState(
        board_a=board_a,
        board_b=board_b,
        hero_hole=hero_hole,
        button_seat=button_seat,
        pot_total_chips=_dollars_to_cents(payload.get("potDollars")),
        seats=tuple(seat_obs),
    )

    return PokerNowFrame(
        frame=frame,
        num_seats=len(seat_obs),
        seat_names=seat_names,
        physical_to_engine=physical_to_engine,
        bomb_pot=bool(payload.get("bombPot")),
        variant=str(payload.get("variant") or "unknown"),
    )
