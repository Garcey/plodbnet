"""Verifiable shuffle for the home games: a SEALED deck plus the PLAYERS' cut.

The operator of the site also plays in these games, so "trust the server's
RNG" is not good enough. The guarantee this gives each player:

    If YOUR device contributed a random number to a hand, that hand's deal was
    uniformly random and was not altered afterwards — even if the server and
    every other player at the table worked together.

How (``SPEC`` = "wrapgto-fair-v1"; every hash is SHA-256 over an ASCII string,
every value hex — so a second implementation is a page of code in any language;
the browser's is ``static/games.fair.js``):

1.  SEAL.  Before anybody contributes anything the server shuffles a deck ``D``
    and publishes one salted commitment per position,
    ``c[i] = H("…|card|hand_id|i|salt[i]|D[i]")``, and the seal
    ``H("…|seal|hand_id|c[0]c[1]…c[51]")``. The salts make an unopened
    commitment useless for guessing its card.
2.  COMMIT.  Each seated device picks 32 random bytes and sends only
    ``H("…|nonce|hand_id|seal|seat|nonce")``.
3.  LOCK.  At the deal the server freezes the list of commitments (dealt-in
    seats only) and publishes it; ``lock = H("…|lock|hand_id|seal|seat:commit,…")``.
4.  REVEAL.  A device reveals its number only after it has seen the lock list
    with its own commitment in it, under the seal it committed to. So when the
    server (and any player working with it) fixed the deck and ITS numbers, the
    honest numbers were still unknown.
5.  CUT.  ``cut = H("…|cut|hand_id|lock|seat:nonce,…")`` drives a Fisher–Yates
    shuffle ``perm`` (SHA-256 counter stream, rejection sampling — no modulo
    bias). The deck that is dealt is ``F[slot] = D[perm[slot]]``.
6.  DEAL.  Public slot map, fixed before anything is known: seat ``s``'s k-th
    hole card is slot ``5s+k`` (every seat index, dealt in or not), board A is
    ``5n..5n+4``, board B ``5n+5..5n+9`` (``n`` = seats at the table).
7.  OPEN.  Every card a player is ever shown comes with ``(slot, pos, salt)``;
    the device checks ``pos == perm[slot]``, that the slot is the right one for
    where the card appeared, and that ``c[pos]`` opens to exactly that card.
    Cards nobody is shown are never opened: a mucked hand stays mucked.

A device that committed but does not reveal in time VOIDS the attempt: the
sealed deck is thrown away and a fresh one is sealed (the absentee may not
contribute to that hand again). Withholding is therefore a visible re-roll —
it is announced at the table with the player's name and kept in the hand's
transcript — never a silent one.

What this does NOT do: it cannot stop the operator from LOOKING at cards on the
server (only dealing the cards on the players' own devices — "mental poker" —
can), and it proves the sealed deck held 52 distinct cards only for the cards
that get opened.

This module is pure (no FastAPI, no table state): sealing, the permutation, the
slot map, openings and ``verify_transcript`` — an independent checker the tests
run against what the server serves and against the JavaScript implementation.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass, field
from typing import Any, Iterable

SPEC = "wrapgto-fair-v1"
DECK_SIZE = 52
HOLE = 5
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def is_hex64(x: Any) -> bool:
    return isinstance(x, str) and bool(_HEX64.match(x))


# --- the pieces of the spec ----------------------------------------------------------


def card_commitment(hand_id: str, pos: int, salt: str, card: int) -> str:
    return sha(f"{SPEC}|card|{hand_id}|{int(pos)}|{salt}|{int(card)}")


def seal_of(hand_id: str, commitments: Iterable[str]) -> str:
    return sha(f"{SPEC}|seal|{hand_id}|" + "".join(commitments))


def nonce_commitment(hand_id: str, seal: str, seat: int, nonce: str) -> str:
    return sha(f"{SPEC}|nonce|{hand_id}|{seal}|{int(seat)}|{nonce}")


def lock_of(hand_id: str, seal: str, locked: Iterable[tuple[int, str]]) -> str:
    return sha(f"{SPEC}|lock|{hand_id}|{seal}|" + ",".join(f"{int(s)}:{c}" for s, c in locked))


def cut_of(hand_id: str, lock: str, reveals: Iterable[tuple[int, str]]) -> str:
    return sha(f"{SPEC}|cut|{hand_id}|{lock}|" + ",".join(f"{int(s)}:{n}" for s, n in reveals))


def permutation(cut: str) -> list[int]:
    """Fisher–Yates over 52 positions from a SHA-256 counter stream. Each block
    ``H("…|stream|cut|k")`` yields eight big-endian 32-bit words; a word at or
    above the largest multiple of the range is skipped (no modulo bias)."""
    perm = list(range(DECK_SIZE))
    words: list[int] = []
    block = 0

    def next_word() -> int:
        nonlocal block
        if not words:
            h = sha(f"{SPEC}|stream|{cut}|{block}")
            block += 1
            words.extend(int(h[i:i + 8], 16) for i in range(0, 64, 8))
        return words.pop(0)

    for j in range(DECK_SIZE - 1, 0, -1):
        span = j + 1
        limit = (2 ** 32 // span) * span
        while True:
            w = next_word()
            if w < limit:
                break
        r = w % span
        perm[j], perm[r] = perm[r], perm[j]
    return perm


def hole_slot(seat: int, k: int) -> int:
    return HOLE * int(seat) + int(k)


def board_slot(num_seats: int, board: str, m: int) -> int:
    return HOLE * int(num_seats) + (0 if board == "a" else 5) + int(m)


# --- the server's side ---------------------------------------------------------------


@dataclass
class SealedDeck:
    """One attempt at one hand. ``key`` and ``deck`` are SECRET until opened card
    by card; everything else is public the moment it exists."""

    hand_id: str
    num_seats: int
    key: str                      # 32 random bytes (hex): salts derive from it
    deck: list[int]               # D — the sealed order
    commitments: list[str] = field(default_factory=list)
    seal: str = ""
    locked: list[tuple[int, str]] = field(default_factory=list)
    lock: str = ""
    reveals: list[tuple[int, str]] = field(default_factory=list)
    cut: str = ""
    perm: list[int] = field(default_factory=list)
    dealt: list[int] = field(default_factory=list)   # F — what the engine deals
    _slot_of: dict[int, int] = field(default_factory=dict, repr=False)

    @classmethod
    def create(cls, hand_id: str, num_seats: int, *, key: str | None = None,
               deck: list[int] | None = None) -> "SealedDeck":
        if deck is None:
            deck = list(range(DECK_SIZE))
            secrets.SystemRandom().shuffle(deck)
        if sorted(int(c) for c in deck) != list(range(DECK_SIZE)):
            raise ValueError("a sealed deck is a permutation of all 52 cards")
        sd = cls(hand_id=str(hand_id), num_seats=int(num_seats),
                 key=key or secrets.token_hex(32), deck=[int(c) for c in deck])
        sd.commitments = [card_commitment(sd.hand_id, i, sd.salt(i), sd.deck[i])
                          for i in range(DECK_SIZE)]
        sd.seal = seal_of(sd.hand_id, sd.commitments)
        return sd

    def salt(self, pos: int) -> str:
        """Per-position salt. Derived (HMAC) so the stored secret is one key;
        an opened salt says nothing about the key or about any other salt."""
        return hmac.new(bytes.fromhex(self.key), f"{SPEC}|salt|{int(pos)}".encode("ascii"),
                        hashlib.sha256).hexdigest()

    def set_lock(self, commits: dict[int, str]) -> None:
        self.locked = sorted((int(s), str(c)) for s, c in commits.items())
        self.lock = lock_of(self.hand_id, self.seal, self.locked)

    def accepts(self, seat: int, nonce: str) -> bool:
        want = dict(self.locked).get(int(seat))
        return (want is not None and is_hex64(nonce)
                and hmac.compare_digest(want, nonce_commitment(self.hand_id, self.seal, seat, nonce)))

    def finish(self, reveals: dict[int, str]) -> list[int]:
        """All locked seats revealed: fix the cut and return the deck to deal."""
        if sorted(reveals) != [s for s, _ in self.locked]:
            raise ValueError("every locked seat must reveal before the cut")
        for s, n in reveals.items():
            if not self.accepts(s, n):
                raise ValueError(f"seat {s}: the nonce does not open its commitment")
        self.reveals = sorted((int(s), str(n)) for s, n in reveals.items())
        self.cut = cut_of(self.hand_id, self.lock, self.reveals)
        self.perm = permutation(self.cut)
        self.dealt = [self.deck[self.perm[j]] for j in range(DECK_SIZE)]
        self._slot_of = {c: j for j, c in enumerate(self.dealt)}
        return list(self.dealt)

    def opening(self, card: int) -> dict[str, Any]:
        slot = self._slot_of[int(card)]
        pos = self.perm[slot]
        return {"slot": slot, "pos": pos, "salt": self.salt(pos)}

    def public(self) -> dict[str, Any]:
        """The transcript: everything except the unopened cards."""
        return {
            "spec": SPEC, "hand_id": self.hand_id, "num_seats": self.num_seats,
            "seal": self.seal, "commitments": list(self.commitments),
            "locked": [[s, c] for s, c in self.locked], "lock": self.lock,
            "reveals": [[s, n] for s, n in self.reveals], "cut": self.cut,
        }

    # compact form kept in the database (commitments are recomputed from it)
    def to_store(self) -> dict[str, Any]:
        return {"hand_id": self.hand_id, "n": self.num_seats, "key": self.key, "deck": list(self.deck),
                "locked": [[s, c] for s, c in self.locked], "reveals": [[s, n] for s, n in self.reveals]}

    @classmethod
    def from_store(cls, d: dict[str, Any]) -> "SealedDeck":
        sd = cls.create(d["hand_id"], int(d["n"]), key=d["key"], deck=list(d["deck"]))
        sd.set_lock({int(s): c for s, c in d.get("locked") or []})
        sd.finish({int(s): n for s, n in d.get("reveals") or []})
        return sd


# --- the independent checker ---------------------------------------------------------


class FairError(Exception):
    """A transcript, or a card shown to a player, does not check out."""


def verify_transcript(tr: dict[str, Any], *, my_seat: int | None = None,
                      my_nonce: str | None = None, seen_seal: str | None = None,
                      seen_lock: str | None = None) -> list[int]:
    """Check a served transcript and return ``perm``. ``my_*`` / ``seen_*`` are
    what a contributing device remembers from before it revealed — the part of
    the guarantee that does not rest on anything the server says afterwards."""
    if tr.get("spec") != SPEC:
        raise FairError("unknown spec")
    hid = str(tr["hand_id"])
    comm = list(tr["commitments"])
    if len(comm) != DECK_SIZE or not all(is_hex64(c) for c in comm):
        raise FairError("a sealed deck has 52 commitments")
    if seal_of(hid, comm) != tr["seal"]:
        raise FairError("the commitments do not add up to the seal")
    locked = [(int(s), str(c)) for s, c in tr["locked"]]
    if locked != sorted(locked) or len({s for s, _ in locked}) != len(locked):
        raise FairError("the lock list must name each seat once, in order")
    if lock_of(hid, tr["seal"], locked) != tr["lock"]:
        raise FairError("the lock does not match its list")
    reveals = [(int(s), str(n)) for s, n in tr["reveals"]]
    if [s for s, _ in reveals] != [s for s, _ in locked]:
        raise FairError("every locked seat must have revealed")
    for (s, n), (_, c) in zip(reveals, locked):
        if not is_hex64(n) or nonce_commitment(hid, tr["seal"], s, n) != c:
            raise FairError(f"seat {s}: the revealed number does not open its commitment")
    if cut_of(hid, tr["lock"], reveals) != tr["cut"]:
        raise FairError("the cut does not follow from the revealed numbers")
    if seen_seal is not None and seen_seal != tr["seal"]:
        raise FairError("the seal changed after this device committed")
    if seen_lock is not None and seen_lock != tr["lock"]:
        raise FairError("the lock list changed after this device revealed")
    if my_nonce is not None:
        if my_seat is None or (int(my_seat), my_nonce) not in reveals:
            raise FairError("this device's number is not in the cut")
    return permutation(tr["cut"])


def verify_opening(tr: dict[str, Any], perm: list[int], card: int, opening: dict[str, Any],
                   *, expect_slots: Iterable[int] | None = None) -> None:
    slot, pos, salt = int(opening["slot"]), int(opening["pos"]), str(opening["salt"])
    if not (0 <= int(card) < DECK_SIZE and 0 <= slot < DECK_SIZE):
        raise FairError("no such card or slot")
    if perm[slot] != pos:
        raise FairError(f"card {card}: slot {slot} is not where the cut put position {pos}")
    if expect_slots is not None and slot not in set(expect_slots):
        raise FairError(f"card {card} was dealt from slot {slot}, which is not its place")
    if card_commitment(str(tr["hand_id"]), pos, salt, card) != tr["commitments"][pos]:
        raise FairError(f"card {card} does not open the sealed position {pos}")
