"""Phase-1 OCR data contracts.

`Card.suit` matches `rust_engine/src/cards.rs`: suit 0=clubs, 1=diamonds,
2=hearts, 3=spades. Rank 0=2 .. 12=A.
"""

from __future__ import annotations

from dataclasses import dataclass, field

RANK_CHARS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
SUIT_CHARS = ("c", "d", "h", "s")


@dataclass(frozen=True)
class Card:
    rank: int
    suit: int

    def __post_init__(self) -> None:
        if not (0 <= self.rank < 13):
            raise ValueError(f"rank out of range: {self.rank}")
        if not (0 <= self.suit < 4):
            raise ValueError(f"suit out of range: {self.suit}")

    def __str__(self) -> str:
        return f"{RANK_CHARS[self.rank]}{SUIT_CHARS[self.suit]}"

    @classmethod
    def parse(cls, s: str) -> "Card":
        s = s.strip()
        if len(s) != 2:
            raise ValueError(f"expected 2-char card like 'As', got {s!r}")
        r = RANK_CHARS.index(s[0].upper())
        suit_char = s[1].lower()
        if suit_char not in SUIT_CHARS:
            raise ValueError(f"unknown suit char in {s!r}")
        return cls(rank=r, suit=SUIT_CHARS.index(suit_char))


@dataclass(frozen=True)
class SeatObs:
    seat: int
    stack_chips: int | None = None
    committed_chips: int | None = None
    folded: bool = False
    all_in: bool = False
    is_actor: bool = False
    # Blue "Bet" overlay on the card-backs; cheap color-mask signal that
    # a seat just committed chips even when committed_chips OCR misses.
    bet_banner: bool = False


@dataclass(frozen=True)
class FrameState:
    board_a: tuple[Card | None, ...] = field(default_factory=lambda: (None,) * 5)
    board_b: tuple[Card | None, ...] = field(default_factory=lambda: (None,) * 5)
    hero_hole: tuple[Card | None, ...] = field(default_factory=lambda: (None,) * 5)
    button_seat: int | None = None
    pot_total_chips: int | None = None
    seats: tuple[SeatObs, ...] = ()

    def to_dict(self) -> dict:
        def card(c: Card | None) -> str | None:
            return None if c is None else str(c)

        return {
            "board_a": [card(c) for c in self.board_a],
            "board_b": [card(c) for c in self.board_b],
            "hero_hole": [card(c) for c in self.hero_hole],
            "button_seat": self.button_seat,
            "pot_total_chips": self.pot_total_chips,
            "seats": [
                {
                    "seat": s.seat,
                    "stack_chips": s.stack_chips,
                    "committed_chips": s.committed_chips,
                    "folded": s.folded,
                    "all_in": s.all_in,
                    "is_actor": s.is_actor,
                    "bet_banner": s.bet_banner,
                }
                for s in self.seats
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FrameState":
        def card(s: str | None) -> Card | None:
            return None if s is None else Card.parse(s)

        return cls(
            board_a=tuple(card(x) for x in d.get("board_a", [None] * 5)),
            board_b=tuple(card(x) for x in d.get("board_b", [None] * 5)),
            hero_hole=tuple(card(x) for x in d.get("hero_hole", [None] * 5)),
            button_seat=d.get("button_seat"),
            pot_total_chips=d.get("pot_total_chips"),
            seats=tuple(
                SeatObs(
                    seat=s["seat"],
                    stack_chips=s.get("stack_chips"),
                    committed_chips=s.get("committed_chips"),
                    folded=s.get("folded", False),
                    all_in=s.get("all_in", False),
                    is_actor=s.get("is_actor", False),
                    bet_banner=s.get("bet_banner", False),
                )
                for s in d.get("seats", [])
            ),
        )
