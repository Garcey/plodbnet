"""Hand review: a player's own ClubGG hand histories, read back into the site.

ClubGG exports hand histories in the GG network's text format, one ``.txt`` per
session, inside a ``.zip``. This module turns its PLO5 double-board bomb pots
("PLO5: BP/DB") into the home games' hand RECORDS (the shape
``homegame._record_hand_locked`` writes), so the replayer, Open in Study and the
grader work on them unchanged. It is pure: no database, no request — the store
(``handreview_store``) calls it.

1. ``read_upload(data)`` — the ``.txt`` files of a ``.zip`` (or one plain text
   file), with limits on entries and sizes (a zip's declared sizes may lie, so
   every read is capped too). Nothing is ever written to disk.
2. ``split_hands`` / ``parse_hand`` — the GG text of one hand → ``ParsedHand``.
   A bomb pot's betting is printed once PER BOARD (the same actions under each
   board's street header); the two copies must agree.
3. ``build_hand(parsed)`` — replays the hand through the engine with the home
   games' bet rule (``reach_cap=False``) from a deck of the cards we know (the
   player's own, both boards, hands shown at the showdown) plus placeholders that
   never show, checks every actor against the history and that the engine's
   result is what ClubGG paid (to the cent, apart from dead money the engine has
   no slot for), then builds the record, the all-in EV (side pots included) and
   the grading job (the player's own decisions only).
"""

from __future__ import annotations

import io
import re
import zipfile
from calendar import timegm
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from plo5bp.actions import ALL_IN, GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.config import VARIANT_PLO5, GameConfig
from plo5bp.ui.runout import board_equities, money_flows, pot_layers

SITE = "clubgg"
#: The record shape the home games' replayer reads (``homegame.HAND_RECORD_VERSION``)
#: plus the review's own fields (``kind`` "review").
REVIEW_RECORD_VERSION = 1

# --- upload limits -------------------------------------------------------------------
#: The upload itself (the request body limit is set from this).
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_ZIP_ENTRIES = 2000
#: One session's text file, and every text file of one upload together (unzipped).
MAX_FILE_BYTES = 40 * 1024 * 1024
MAX_TOTAL_BYTES = 200 * 1024 * 1024
#: Hands read from one upload.
MAX_HANDS_PER_UPLOAD = 50_000

RANKS = "23456789TJQKA"
SUITS = "cdhs"  # (the engine's order: card = rank * 4 + suit)
STREETS = ("flop", "turn", "river")
#: Board length on each street (bomb pots start on the flop).
STREET_LEN = {"flop": 3, "turn": 4, "river": 5}


class UploadError(ValueError):
    """The upload as a whole can't be read (not a zip / text, too big, no hands)."""


class HandError(ValueError):
    """One hand can't be read or replayed: it is skipped and counted, with ``reason``."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# --- reading the upload ----------------------------------------------------------------


def read_upload(data: bytes, filename: str = "") -> list[tuple[str, str]]:
    """``[(file name, text)]`` of every hand-history text file in the upload.

    A ``.zip`` (what ClubGG exports; it does not need unpacking first) or a
    single text file. Nested folders inside the zip are fine; anything that is
    not a ``.txt`` is ignored."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError("too_big", f"the upload is over {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not data:
        raise UploadError("empty", "the file is empty")
    if data[:4] == b"PK\x03\x04" or data[:4] == b"PK\x05\x06":
        return _read_zip(data)
    # One plain text file (a session's .txt dropped on its own).
    return [(filename or "hands.txt", _decode(data[:MAX_FILE_BYTES]))]


def _read_zip(data: bytes) -> list[tuple[str, str]]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise UploadError("bad_zip", "that zip file can't be opened") from e
    infos = [i for i in zf.infolist() if not i.is_dir()]
    if len(infos) > MAX_ZIP_ENTRIES:
        raise UploadError("too_many_files", f"the zip holds more than {MAX_ZIP_ENTRIES} files")
    out: list[tuple[str, str]] = []
    total = 0
    for info in infos:
        name = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
        if not name.lower().endswith(".txt") or name.startswith("."):
            continue
        if info.flag_bits & 0x1:
            raise UploadError("encrypted", "the zip is password-protected")
        if info.file_size > MAX_FILE_BYTES:
            raise UploadError("too_big", f"{name} is over {MAX_FILE_BYTES // (1024 * 1024)} MB unzipped")
        try:
            with zf.open(info) as fh:
                raw = fh.read(MAX_FILE_BYTES + 1)  # (a declared size can lie: cap the read)
        except (zipfile.BadZipFile, NotImplementedError, OSError, EOFError) as e:
            raise UploadError("bad_zip", f"{name} can't be read from the zip") from e
        if len(raw) > MAX_FILE_BYTES:
            raise UploadError("too_big", f"{name} is over {MAX_FILE_BYTES // (1024 * 1024)} MB unzipped")
        total += len(raw)
        if total > MAX_TOTAL_BYTES:
            raise UploadError("too_big", f"the zip holds over {MAX_TOTAL_BYTES // (1024 * 1024)} MB of text")
        out.append((name, _decode(raw)))
    return out


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


_HAND_START = re.compile(r"^Poker Hand #", re.M)


def split_hands(text: str) -> list[str]:
    """The hands of one file, each from its ``Poker Hand #`` line."""
    starts = [m.start() for m in _HAND_START.finditer(text)]
    return [text[a:b].strip() for a, b in zip(starts, starts[1:] + [len(text)])]


# --- parsing one hand --------------------------------------------------------------------

_AMT = r"[^\d\s(]*?(\d[\d,]*(?:\.\d+)?)"  # an amount, any currency sign before it
_RE_HEADER = re.compile(
    r"^Poker Hand #(?P<id>[A-Za-z0-9_\-]+): (?P<game>.+?) \(" + _AMT + r"/" + _AMT
    + r"\)\s*-\s*(?P<date>\d{4}/\d\d/\d\d \d\d:\d\d:\d\d)"
)
_RE_TABLE = re.compile(r"^Table '(?P<name>.*)' (?P<max>\d+)-max Seat #(?P<btn>\d+) is the button")
_RE_SEAT = re.compile(r"^Seat (?P<no>\d+): (?P<name>.+?) \(" + _AMT + r" in chips\)(?P<rest>.*)$")
_RE_ANTE = re.compile(r"^(?P<name>.+?): posts the ante " + _AMT + r"(?P<ai> and is all-in)?\s*$")
_RE_DEAD = re.compile(r"^(?P<name>.+?): posts (?:missed blind|dead) " + _AMT + r"(?P<ai> and is all-in)?\s*$")
_RE_BLIND = re.compile(r"^(?P<name>.+?): posts (?:the )?(?:small|big) blind")
_RE_DEALT = re.compile(r"^Dealt to (?P<name>.+?)(?: \[(?P<cards>[^\]]*)\])?\s*$")
_RE_STREET = re.compile(r"^\*\*\* (?P<st>FLOP|TURN|RIVER) \*\*\* \[(?P<b1>[^\]]*)\](?: \[(?P<b2>[^\]]*)\])?\s*$")
_RE_UNCALLED = re.compile(r"^Uncalled bet \(" + _AMT + r"\) returned to (?P<name>.+?)\s*$")
_RE_COLLECTED = re.compile(r"^(?P<name>.+?) collected " + _AMT + r" from (?:the )?(?:main |side )?pot(?:-?\d+)?\s*$")
_RE_TOTAL = re.compile(r"^Total pot " + _AMT + r"(?P<rest>.*)$")
_RE_RAKE = re.compile(r"Rake " + _AMT)
_RE_JACKPOT = re.compile(r"Jackpot " + _AMT)
_RE_SHOWS = re.compile(r"^(?P<name>.+?): shows \[(?P<cards>[^\]]*)\]")
_RE_SUMMARY_BOARD = re.compile(r"^Board \[(?P<cards>[^\]]*)\]\s*$")
_ACTION_TAIL = {
    "checks": re.compile(r"^\s*$"),
    "folds": re.compile(r"^\s*$"),
    "bets": re.compile(r"^ " + _AMT + r"(?P<ai> and is all-in)?\s*$"),
    "calls": re.compile(r"^ " + _AMT + r"(?P<ai> and is all-in)?\s*$"),
    # "raises $160 to $340" — or, all in for less than a full raise, "raises to $340"
    "raises": re.compile(r"^ (?:" + _AMT + r" )?to " + _AMT + r"(?P<ai> and is all-in)?\s*$"),
}
_IGNORED = (
    re.compile(r"^\*\*\* (HOLE CARDS|SUMMARY) \*\*\*"),
    re.compile(r"^Seat \d+: "),  # (summary lines)
    re.compile(r"^.+?: (mucks|doesn't show|sits out|is sitting out|has timed out|is disconnected)"),
    re.compile(r"^.+? (joins|leaves) the table"),
    re.compile(r"^.+?: Chooses to"),
)


def cents(s: str) -> int:
    """'1,191.39' -> 119139 (exact)."""
    try:
        v = Decimal(str(s).replace(",", ""))
    except InvalidOperation as e:
        raise HandError("bad_amount", str(s)) from e
    c = v * 100
    if c != c.to_integral_value():
        raise HandError("bad_amount", f"{s} is not a whole number of cents")
    return int(c)


def card_id(tok: str) -> int:
    tok = tok.strip()
    if len(tok) != 2 or tok[0].upper() not in RANKS or tok[1].lower() not in SUITS:
        raise HandError("bad_card", tok)
    return RANKS.index(tok[0].upper()) * 4 + SUITS.index(tok[1].lower())


def cards_of(s: str | None) -> list[int]:
    return [card_id(t) for t in (s or "").split()] if s and s.strip() else []


def card_str(c: int) -> str:
    return RANKS[c // 4] + SUITS[c % 4]


@dataclass
class Act:
    """One betting action as the history prints it (amounts in cents)."""

    name: str
    verb: str  # checks | folds | bets | calls | raises
    amount: int = 0  # bets / calls: the amount; raises: the raise-to total
    all_in: bool = False

    def key(self) -> tuple:
        return (self.name, self.verb, self.amount, self.all_in)


@dataclass
class ParsedHand:
    hand_id: str  # "ring_9000000001"
    played_at: str  # "2026/10/02 00:03:26", as printed (the table's clock)
    ts: int  # the same, as epoch seconds (read as UTC — only used to order hands)
    game: str
    sb_cents: int
    bb_cents: int
    table_name: str
    max_seats: int
    button_seat: int  # ClubGG's seat number (may be an empty seat: a dead button)
    seats: list[tuple[int, str, int]]  # (seat number, name, chips at the start in cents)
    antes: dict[str, int] = field(default_factory=dict)
    dead: dict[str, int] = field(default_factory=dict)  # missed blinds: dead money
    hero: str | None = None
    hero_hole: list[int] = field(default_factory=list)
    board_a: list[int] = field(default_factory=list)
    board_b: list[int] = field(default_factory=list)
    streets: dict[str, list[Act]] = field(default_factory=dict)
    uncalled: tuple[str, int] | None = None
    shown: dict[str, list[int]] = field(default_factory=dict)
    collected: list[tuple[int, str, int]] = field(default_factory=list)  # (board 0/1, name, cents)
    total_pot: int = 0
    rake: int = 0
    text: str = ""

    @property
    def key(self) -> str:
        return f"{SITE}:{self.hand_id}"


def _known_name(line: str, names: list[str]) -> tuple[str, str] | None:
    """(name, rest of the line) when the line starts with a seated player's
    ``name: `` (longest first: one name can be another's prefix)."""
    for nm in names:
        if line.startswith(nm + ": "):
            return nm, line[len(nm) + 2:]
    return None


def parse_hand(text: str) -> ParsedHand:
    """One hand's GG text -> ``ParsedHand``. Raises ``HandError`` (with a short
    ``reason``) for a hand of another game or one that doesn't read cleanly."""
    lines = [ln.rstrip("\r") for ln in text.strip().splitlines()]
    if not lines:
        raise HandError("empty")
    m = _RE_HEADER.match(lines[0])
    if not m:
        raise HandError("not_a_hand", lines[0][:80])
    game = m.group("game").strip()
    if "PLO" not in game.upper() and "OMAHA" not in game.upper():
        raise HandError("other_game", game)
    if "POT LIMIT" not in game.upper():
        raise HandError("other_game", game)
    sb, bb = cents(m.group(3)), cents(m.group(4))
    if bb <= 0:
        raise HandError("bad_stakes", m.group(0))
    date = m.group("date")
    try:
        ts = timegm(datetime.strptime(date, "%Y/%m/%d %H:%M:%S").timetuple())
    except ValueError as e:
        raise HandError("bad_date", date) from e
    if len(lines) < 2 or not (mt := _RE_TABLE.match(lines[1])):
        raise HandError("no_table_line")
    p = ParsedHand(
        hand_id=m.group("id"), played_at=date, ts=ts, game=game, sb_cents=sb, bb_cents=bb,
        table_name=mt.group("name"), max_seats=int(mt.group("max")),
        button_seat=int(mt.group("btn")), seats=[], text=text.strip(),
    )
    i = 2
    while i < len(lines) and (ms := _RE_SEAT.match(lines[i])):
        rest = ms.group("rest").strip().lower()
        if "sitting out" not in rest:
            p.seats.append((int(ms.group("no")), ms.group("name"), cents(ms.group(3))))
        i += 1
    if not p.seats:
        raise HandError("no_seats")
    names = sorted({nm for _, nm, _ in p.seats}, key=len, reverse=True)
    street: str | None = None  # the street being read
    copy = 0  # 0 = board 1's copy of the street's betting, 1 = board 2's
    seen: dict[str, int] = {}  # street -> header count
    board2_acts: dict[str, list[Act]] = {}
    showdown = -1  # which SHOWDOWN block (board) the collected lines belong to
    in_summary = False
    summary_boards: list[list[int]] = []
    uncalled_seen: list[tuple[str, int]] = []
    for ln in lines[i:]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("*** SUMMARY ***"):
            in_summary = True
            continue
        if in_summary:
            if mt2 := _RE_TOTAL.match(s):
                p.total_pot = cents(mt2.group(1))
                if mr := _RE_RAKE.search(mt2.group("rest")):
                    p.rake += cents(mr.group(1))
                if mj := _RE_JACKPOT.search(mt2.group("rest")):
                    p.rake += cents(mj.group(1))
            elif mb := _RE_SUMMARY_BOARD.match(s):
                summary_boards.append(cards_of(mb.group("cards")))
            continue
        if s.startswith("*** SHOWDOWN ***"):
            showdown += 1
            street = None
            continue
        if ms2 := _RE_STREET.match(s):
            st = ms2.group("st").lower()
            n_seen = seen.get(st, 0)
            if n_seen >= 2:
                raise HandError("single_board", f"three {st} headers")
            seen[st] = n_seen + 1
            street, copy = st, n_seen
            cards = cards_of(ms2.group("b1")) + cards_of(ms2.group("b2"))
            board = p.board_a if copy == 0 else p.board_b
            if len(cards) != STREET_LEN[st] or cards[: len(board)] != board[: len(cards)]:
                raise HandError("bad_board", s)
            board[:] = cards
            if copy == 0:
                p.streets.setdefault(st, [])
            else:
                board2_acts.setdefault(st, [])
            continue
        if ma := _RE_ANTE.match(s):
            p.antes[ma.group("name")] = p.antes.get(ma.group("name"), 0) + cents(ma.group(2))
            continue
        if md := _RE_DEAD.match(s):
            p.dead[md.group("name")] = p.dead.get(md.group("name"), 0) + cents(md.group(2))
            continue
        if _RE_BLIND.match(s):
            raise HandError("not_a_bomb_pot", "the hand has blinds")
        if mdl := _RE_DEALT.match(s):
            cards = cards_of(mdl.group("cards"))
            if cards:
                if p.hero is not None and p.hero != mdl.group("name"):
                    raise HandError("two_heroes")
                p.hero, p.hero_hole = mdl.group("name"), cards
            continue
        if mu := _RE_UNCALLED.match(s):
            uncalled_seen.append((mu.group("name"), cents(mu.group(1))))
            continue
        if mc := _RE_COLLECTED.match(s):
            if showdown < 0:
                raise HandError("collected_outside_showdown", s)
            p.collected.append((min(showdown, 1), mc.group("name"), cents(mc.group(2))))
            continue
        if msh := _RE_SHOWS.match(s):
            p.shown[msh.group("name")] = cards_of(msh.group("cards"))
            continue
        kn = _known_name(s, names)
        if kn is not None:
            nm, rest = kn
            verb = rest.split(" ", 1)[0]
            tail_re = _ACTION_TAIL.get(verb)
            if tail_re is not None:
                tail = rest[len(verb):]
                mtail = tail_re.match(tail)
                if not mtail:
                    raise HandError("bad_action", s)
                if street is None:
                    raise HandError("action_outside_street", s)
                amt = 0
                if verb in ("bets", "calls"):
                    amt = cents(mtail.group(1))
                elif verb == "raises":
                    amt = cents(mtail.group(2))
                act = Act(nm, verb, amt, bool(mtail.groupdict().get("ai")))
                (p.streets[street] if copy == 0 else board2_acts[street]).append(act)
                continue
        if any(r.match(s) for r in _IGNORED):
            continue
        raise HandError("unknown_line", s[:100])
    # --- consistency of what was read ---
    if p.hero is None or len(p.hero_hole) != 5:
        raise HandError("no_hero_cards", "this hand doesn't show your five cards")
    if seen.get("flop", 0) != 2:
        raise HandError("not_double_board", "a double-board hand has two flops")
    for st, acts in board2_acts.items():
        if acts and [a.key() for a in acts] != [a.key() for a in p.streets.get(st, [])]:
            raise HandError("boards_disagree", f"board 2's {st} betting differs from board 1's")
    for st in STREETS:
        if seen.get(st, 0) not in (0, 2):
            raise HandError("not_double_board", f"one {st} header")
    if summary_boards and len(summary_boards) == 2:
        if summary_boards[0] != p.board_a or summary_boards[1] != p.board_b:
            raise HandError("bad_board", "the summary's boards differ")
    if uncalled_seen:
        # (printed once per board — the same bet)
        if len({u for u in uncalled_seen}) != 1:
            raise HandError("bad_uncalled", str(uncalled_seen))
        p.uncalled = uncalled_seen[0]
    if not p.collected:
        raise HandError("no_result", "nobody collected the pot")
    if showdown != 1:
        raise HandError("not_double_board", "a double-board hand has two showdown blocks")
    for nm in list(p.antes) + list(p.dead) + [a.name for acts in p.streets.values() for a in acts]:
        if nm not in names:
            raise HandError("unknown_player", nm)
    return p


# --- the money, from ClubGG's own numbers ---------------------------------------------------


@dataclass
class Ledger:
    """One hand's money as ClubGG dealt it (cents), the players in seat order."""

    seats: list[tuple[int, str, int]]  # (seat number, name, chips at the start in cents)
    idx: dict[str, int]
    hero: int
    button: int  # index into ``seats``
    put: list[int]  # what each put in: ante, dead money, bets (the uncalled bet excluded)
    dead: list[int]
    won: list[int]  # what each collected
    folded: list[bool]
    holes: dict[int, list[int]]  # the cards we know: the player's own, and hands shown
    last_street: str | None  # the street of the last action (None: no betting at all)

    @property
    def n(self) -> int:
        return len(self.seats)

    @property
    def alive(self) -> list[int]:
        return [i for i in range(self.n) if not self.folded[i]]

    def net(self, i: int) -> int:
        return self.won[i] - self.put[i]


def _engine_seats(p: ParsedHand) -> list[tuple[int, str, int]]:
    """The players dealt in (they posted the ante), in seat order."""
    dealt = [s for s in p.seats if s[1] in p.antes]
    if len(dealt) < 2:
        raise HandError("too_few_players")
    if len(dealt) > 6:
        raise HandError("too_many_players", "Study and the network play up to 6")
    return sorted(dealt, key=lambda s: s[0])


def _flat_actions(p: ParsedHand) -> list[tuple[str, Act]]:
    return [(st, a) for st in STREETS for a in p.streets.get(st, [])]


def ledger(p: ParsedHand) -> Ledger:
    """Who put in what, who folded, who collected — straight from the history."""
    seats = _engine_seats(p)
    n = len(seats)
    idx = {nm: i for i, (_, nm, _) in enumerate(seats)}
    hero = idx.get(p.hero or "")
    if hero is None:
        raise HandError("hero_not_dealt", "you weren't dealt in")
    flat = _flat_actions(p)
    for _st, a in flat:
        if a.name not in idx:
            raise HandError("unknown_player", a.name)
    # The button: ClubGG's button seat, or (a dead button) the last seat before it —
    # and in any case the seat just before the first to act.
    btn = next((i for i in range(n - 1, -1, -1) if seats[i][0] <= p.button_seat), n - 1)
    if flat and (btn + 1) % n != idx[flat[0][1].name]:
        btn = (idx[flat[0][1].name] - 1) % n
    dead = [p.dead.get(nm, 0) for _, nm, _ in seats]
    put = [p.antes.get(nm, 0) + d for (_, nm, _), d in zip(seats, dead)]
    folded = [False] * n
    last: str | None = None
    for st in STREETS:
        sc = [0] * n
        for a in p.streets.get(st, []):
            i = idx[a.name]
            last = st
            if a.verb == "folds":
                folded[i] = True
            elif a.verb in ("bets", "calls"):
                sc[i] += a.amount
            elif a.verb == "raises":
                sc[i] = a.amount  # (a raise names the street's new total)
        for i in range(n):
            put[i] += sc[i]
    if p.uncalled:
        if p.uncalled[0] not in idx:
            raise HandError("unknown_player", p.uncalled[0])
        put[idx[p.uncalled[0]]] -= p.uncalled[1]
    won = [0] * n
    for _b, nm, c in p.collected:
        if nm not in idx:
            raise HandError("unknown_player", nm)
        won[idx[nm]] += c
    if sum(won) + p.rake != sum(put):
        raise HandError("pot_mismatch", f"{sum(put) / 100:.2f} went in, {sum(won) / 100:.2f} was collected")
    for i, (_, nm, start) in enumerate(seats):
        if put[i] > start:
            raise HandError("over_stack", f"{nm} put in more than they had")
    holes = {hero: list(p.hero_hole)}
    for nm, cs in p.shown.items():
        if nm in idx and len(cs) == 5:
            holes[idx[nm]] = list(cs)
    known = [c for cs in holes.values() for c in cs] + p.board_a + p.board_b
    if len(set(known)) != len(known):
        raise HandError("duplicate_card", " ".join(card_str(c) for c in known))
    return Ledger(seats, idx, hero, btn, put, dead, won, folded, holes, last)


def allin_ev(L: Ledger, p: ParsedHand, cache: dict | None = None) -> tuple[int, dict[str, Any] | None]:
    """The player's result with the all-in luck taken out (cents), and how it was
    worked out — the actual result when the hand had no all-in with cards to come.

    The all-in point is where the betting ended: if nobody acted on the river (or
    later streets) and two or more hands went to the showdown, the cards still to
    come were dealt with nobody able to act. Every pot LAYER there (main pot, side
    pots: ``runout.pot_layers``) is split half per board, and on each board a layer
    goes to the best hand among the players eligible for it. So a layer is worth
    ``half x share`` on each board, ``share`` = the player's chance of winning that
    board AGAINST THAT LAYER'S PLAYERS ONLY (``runout.board_equities``: exact over
    the board's missing cards, ties split). A layer the player alone is in is
    theirs. EV result = the expected collection minus what went in."""
    hero = L.hero
    net = L.net(hero)
    alive = L.alive
    if L.folded[hero] or len(alive) < 2:
        return net, None
    start = STREET_LEN[L.last_street] if L.last_street else 3
    if start >= 5 or len(p.board_a) < 5 or len(p.board_b) < 5:
        return net, None
    if any(i not in L.holes for i in alive):
        return net, None  # (a hand at the showdown we can't see)
    total = 0.0
    parts = []
    for ly in pot_layers(L.put, L.folded):
        elig = [int(s) for s in ly["eligible"]]
        size = int(ly["chips"])
        if hero not in elig or size <= 0:
            continue
        if len(elig) == 1:
            total += size
            parts.append({"cents": size, "players": 1, "share": [1.0, 1.0]})
            continue
        out_of_race = [c for s in alive if s not in elig for c in L.holes[s]]
        eq = _equities(cache, {s: L.holes[s] for s in elig}, p.board_a[:start], p.board_b[:start], out_of_race)
        a, b = float(eq[hero]["a"]), float(eq[hero]["b"])
        half = size // 2
        total += half * a + (size - half) * b
        parts.append({"cents": size, "players": len(elig), "share": [round(a, 4), round(b, 4)]})
    if p.rake:  # (the pots are paid net of the rake)
        total *= (sum(L.put) - p.rake) / max(1, sum(L.put))
    ev_net = int(round(total)) - L.put[hero]
    return ev_net, {"from": start, "pots": parts, "luck_cents": int(net - ev_net)}


def _equities(
    cache: dict | None, holes: dict[int, list[int]], board_a: list[int], board_b: list[int],
    dead: list[int] | tuple[int, ...] = (),
) -> dict[int, dict[str, float]]:
    """``runout.board_equities`` (exact shares) once per race in a hand: the
    replayer's street equities and the EV's main pot ask for the same one.
    ``dead`` = known cards out of the race (a side pot: the all-in hand that can't
    win it still holds its cards — they can't come on the board)."""
    dead_t = tuple(sorted(int(c) for c in dead))
    key = (tuple(sorted((int(s), tuple(h)) for s, h in holes.items())), tuple(board_a), tuple(board_b), dead_t)
    if cache is not None and key in cache:
        return cache[key]
    out = board_equities(holes, board_a, board_b, dead=dead_t, digits=None)
    if cache is not None:
        cache[key] = out
    return out


# --- the record (the home games' shape) ------------------------------------------------------


def _fmt(c: int) -> str:
    return f"${c / 100:,.2f}"


def _chips_fn(bb_cents: int):
    from plo5bp.ui import homegame as hg

    per = hg.chips_per_cent(bb_cents)
    return (lambda c: int(c) * per) if per > 0 else (lambda c: hg.cents_to_chips(int(c), bb_cents))


def make_record(p: ParsedHand, L: Ledger, cache: dict | None = None) -> dict[str, Any]:
    """The hand as the replayer and Open in Study read it — every amount ClubGG's."""
    from plo5bp.actions import BET_PCT_100, CHECK_CALL, FOLD
    from plo5bp.ui import homegame as hg
    from plo5bp.ui.hand_describe import best_combo

    n = L.n
    chips = _chips_fn(p.bb_cents)
    actions: list[dict[str, Any]] = []
    for st in STREETS:
        sc = [0] * n
        for a in p.streets.get(st, []):
            i = L.idx[a.name]
            if a.verb == "folds":
                aid, add, label = FOLD, 0, "Fold"
            elif a.verb == "checks":
                aid, add, label = CHECK_CALL, 0, "Check"
            elif a.verb == "calls":
                aid, add = CHECK_CALL, a.amount
                label = f"Call {_fmt(add)}" + (" · all-in" if a.all_in else "")
            else:
                to = a.amount if a.verb == "raises" else sc[i] + a.amount
                add = to - sc[i]
                aid = ALL_IN if a.all_in else BET_PCT_100
                label = f"All-in {_fmt(add)}" if a.all_in else (f"Bet {_fmt(to)}" if a.verb == "bets" else f"Raise to {_fmt(to)}")
            sc[i] += add
            actions.append({
                "seat": i, "street": st, "action": int(aid), "chips": chips(add), "cents": int(add),
                "to_cents": int(sc[i]), "label": label, "auto": False,
            })
    alive = L.alive
    showdown = len(alive) >= 2
    seat_recs = []
    for i, (no, nm, start) in enumerate(L.seats):
        shown = showdown and not L.folded[i]
        hole = sorted(L.holes[i], reverse=True) if i in L.holes and (i == L.hero or shown) else []
        seat_recs.append({
            "seat": i, "clubgg_seat": no, "name": "You" if i == L.hero else _display_name(nm),
            "start_cents": int(start), "start_chips": chips(start - L.dead[i]),
            **({"dead_cents": int(L.dead[i])} if L.dead[i] else {}),
            "delta_cents": int(L.net(i)), "folded": bool(L.folded[i]), "shown": bool(shown),
            "hole": hole, "is_me": i == L.hero,
        })
    boards = {"a": p.board_a, "b": p.board_b}
    awards = []
    for b, nm, c in p.collected:
        i = L.idx[nm]
        key = "a" if b == 0 else "b"
        combo = best_combo(L.holes[i], boards[key]) if showdown and i in L.holes and len(boards[key]) == 5 else None
        awards.append({
            "board": key, "winners": [i], "cents": int(c), "uncontested": not showdown,
            "labels": {str(i): combo["label"]} if combo else {},
        })
    holes_alive: list[list[int] | None] = [L.holes.get(i) if not L.folded[i] else None for i in range(n)]
    flows = money_flows(L.put, L.folded, holes_alive, p.board_a, p.board_b, L.button) \
        if not showdown or all(holes_alive[i] for i in alive) else {}
    equities: dict[str, Any] = {}
    start = STREET_LEN[L.last_street] if L.last_street else 3
    if showdown and start < 5 and len(p.board_a) == 5 and all(i in L.holes for i in alive):
        hs = {i: L.holes[i] for i in alive}
        for ln in range(start, 5):
            eq = _equities(cache, hs, p.board_a[:ln], p.board_b[:ln])
            equities[str(ln)] = {str(s): [round(float(v["a"]), 4), round(float(v["b"]), 4)] for s, v in sorted(eq.items())}
    unc = None
    if p.uncalled:
        unc = {"seat": L.idx[p.uncalled[0]], "cents": int(p.uncalled[1])}
    ante = max(p.antes[nm] for _, nm, _ in L.seats)
    return {
        "v": hg.HAND_RECORD_VERSION,
        "kind": "review", "review_v": REVIEW_RECORD_VERSION, "site": SITE,
        "hand_key": p.key, "hand_id": p.hand_id,
        "hand_no": int(re.sub(r"\D", "", p.hand_id)[-12:] or 0),
        "played_at": p.played_at, "ended_at": p.played_at, "ts": int(p.ts),
        "table_name": p.table_name, "game_name": p.game,
        "variant": "plo5", "hole_count": 5,
        "button": int(L.button), "num_seats": n,
        "bb_cents": int(p.bb_cents), "sb_cents": int(p.sb_cents), "ante_cents": int(ante),
        "dead_cents": int(sum(L.dead)),
        "pot_cents": int(sum(L.put)),
        **({"rake_cents": int(p.rake)} if p.rake else {}),
        **({"uncalled": unc} if unc else {}),
        "showdown": bool(showdown),
        "board_a": list(p.board_a), "board_b": list(p.board_b), "burns": [],
        **({"equities": equities, "runout_from": int(start)} if equities else {}),
        "actions": actions,
        "ante_chips": chips(ante), "bb_chips": hg.BB_CHIPS,
        "flows": [{"from": int(a), "to": int(b2), "cents": int(v)} for (a, b2), v in sorted(flows.items())],
        "grades": None,
        "fair": None,
        "awards": awards,
        "seats": seat_recs,
    }


# --- the engine replay: the check, and the grading job ---------------------------------------


def engine_replay(p: ParsedHand, L: Ledger) -> tuple[dict[str, Any], int, list[str]]:
    """Replay the hand through the engine (the home games' bet rule) from a deck of
    the known cards + placeholders: ``(grading job, actions it followed, notes)``.

    The job holds the actions up to the first one the engine can't follow — its
    rules are the network's, and ClubGG differs in one corner: a player who checked
    may raise a short all-in there (the engine keeps them to call or fold, the TDA
    rule). The decisions before it are still graded. Dead money (a missed blind)
    has no slot in the engine: the poster's stack is cut by it so every stack
    matches, and the pot is that much short (noted). ``notes`` empty = the replay
    ended exactly where ClubGG's did, paying what ClubGG paid (a few cents of
    ClubGG's board-split rounding aside)."""
    from plo5bp.env import BombPotEnv
    from plo5bp.ui import homegame as hg

    n = L.n
    chips = _chips_fn(p.bb_cents)
    notes: list[str] = []
    if hg.chips_per_cent(p.bb_cents) <= 0:
        notes.append("stakes")  # (chips round to the nearest cent)
    dead_total = sum(L.dead)
    if dead_total:
        notes.append("dead money")
    # The deck: five cards a seat INDEX, then board A, then board B; placeholders
    # (never shown, never in an observation the network reads) where we don't know.
    known = set(c for cs in L.holes.values() for c in cs) | set(p.board_a) | set(p.board_b)
    spare = [c for c in range(52) if c not in known]
    deck: list[int] = []
    for i in range(n):
        deck += L.holes[i] if i in L.holes else [spare.pop(0) for _ in range(5)]
    for b in (p.board_a, p.board_b):
        deck += b + [spare.pop(0) for _ in range(5 - len(b))]
    deck += spare
    ante = max(p.antes[nm] for _, nm, _ in L.seats)
    stacks = [chips(start - L.dead[i]) for i, (_, _, start) in enumerate(L.seats)]
    cfg = GameConfig(
        num_seats=n, starting_stack=0, starting_stacks=tuple(stacks), ante=chips(ante),
        bb=hg.BB_CHIPS, variant=VARIANT_PLO5, reach_cap=False,
    )
    env = BombPotEnv(cfg, ev_runout_samples=0)
    _obs, info = env.reset_with_deck(deck, L.button, in_hand_mask=[True] * n)
    per_cent = max(1, hg.chips_per_cent(p.bb_cents))
    flat = _flat_actions(p)
    behind = [L.seats[i][2] - L.dead[i] - min(ante, L.seats[i][2]) for i in range(n)]  # (cents)
    job_actions: list[list[Any]] = []
    upto = 0
    cur = None
    sc = [0] * n
    folded = [False] * n
    for k, (st, a) in enumerate(flat):
        if st != cur:
            cur, sc = st, [0] * n
        seat = L.idx[a.name]
        if info is None or info.actor is None or env.is_terminal():
            notes.append(f"the engine's hand ended before action {k + 1}")
            break
        if int(info.actor) != seat:
            notes.append(f"the engine expected another player at action {k + 1}")
            break
        if a.verb == "folds":
            gate, add_c = GATE_FOLD, 0
        elif a.verb in ("checks", "calls"):
            gate, add_c = GATE_CHECK_CALL, a.amount
        else:
            to = a.amount if a.verb == "raises" else sc[seat] + a.amount
            gate, add_c = GATE_RAISE, to - sc[seat]
        add = chips(add_c)
        if gate == GATE_RAISE:
            lo, hi = int(info.min_raise_chips), int(info.max_raise_chips)
            short_shove = lo == 0 and bool(info.legal_mask[ALL_IN])
            if not short_shove and hi <= 0:
                others_in = any(not folded[j] and behind[j] - sc[j] > 0 for j in range(n) if j != seat)
                if others_in:
                    notes.append(f"action {k + 1}: ClubGG let {('you' if seat == L.hero else 'a player')} raise a short all-in after checking — the engine's rules don't")
                    break
                gate = GATE_CHECK_CALL  # (everyone else is all in: a raise is a call — the rest comes back)
            elif not short_shove and not lo <= add <= hi:
                if dead_total:
                    add = max(lo, min(add, hi))  # (a pot-sized bet that counted the dead money)
                else:
                    notes.append(f"action {k + 1}: a size the engine doesn't allow")
                    break
        try:
            _obs, _r, _done, info = env.step_hybrid(int(gate), int(add))
        except Exception as e:  # noqa: BLE001 — the engine refused it
            notes.append(f"action {k + 1}: the engine refused it ({e})")
            break
        put_in = int(info.commit_delta[seat]) if info.commit_delta is not None else add
        if a.verb == "folds":
            folded[seat] = True
        sc[seat] += add_c if gate != GATE_FOLD else 0
        if gate == GATE_CHECK_CALL and a.verb in ("checks", "calls") and abs(put_in - add) > per_cent and not dead_total:
            notes.append(f"action {k + 1}: the engine's call differs")
            break
        job_actions.append([seat, int(gate), int(put_in) if gate == GATE_RAISE else 0, seat == L.hero])
        upto = k + 1
    else:
        if not env.is_terminal():
            notes.append("the engine's hand went on after ClubGG's ended")
        else:
            tol = 10 + dead_total + p.rake  # (ClubGG's board-split rounding: a few cents)
            pay = env.payouts()
            for i in range(n):
                if abs(hg.chips_to_cents(int(pay[i]), p.bb_cents) - L.net(i)) > tol:
                    notes.append("the engine's result differs from ClubGG's")
                    break
    job = {
        "variant": "plo5", "num_seats": n, "stacks": stacks, "ante": chips(ante),
        "deck": deck, "button": int(L.button), "mask": [True] * n, "actions": job_actions,
    }
    return job, upto, notes


@dataclass
class BuiltHand:
    parsed: ParsedHand
    record: dict[str, Any]
    job: dict[str, Any]
    net_cents: int
    ev_net_cents: int
    allin: bool  # the player was in an all-in with cards to come
    pot_cents: int
    decisions: int  # the player's decisions in the hand
    gradable: int  # of them, the ones the grading replay reaches
    notes: list[str]  # where the engine replay differs (empty = exact)


def build_hand(p: ParsedHand) -> BuiltHand:
    """The record, the all-in EV and the grading job of one parsed hand."""
    L = ledger(p)
    cache: dict = {}
    rec = make_record(p, L, cache)
    ev_net, ev_detail = allin_ev(L, p, cache)
    job, upto, notes = engine_replay(p, L)
    decisions = sum(1 for _st, a in _flat_actions(p) if L.idx[a.name] == L.hero)
    gradable = sum(1 for a in job["actions"] if a[3])
    rec["net_cents"] = int(L.net(L.hero))
    rec["ev_net_cents"] = int(ev_net)
    rec["allin"] = ev_detail is not None
    if ev_detail:
        rec["allin_ev"] = ev_detail
    rec["study_upto"] = int(upto)
    if notes:
        rec["replay_notes"] = notes
    return BuiltHand(
        parsed=p, record=rec, job=job, net_cents=int(L.net(L.hero)), ev_net_cents=int(ev_net),
        allin=ev_detail is not None, pot_cents=int(sum(L.put)), decisions=decisions,
        gradable=gradable, notes=notes,
    )


def _display_name(nm: str) -> str:
    """ClubGG's export already hides the other players behind an id ("aa11bb22");
    the review shows a short one."""
    s = str(nm).strip()
    return ("Player " + s[:4].upper()) if re.fullmatch(r"[0-9a-fA-F]{6,}", s) else s[:24]


# --- one upload ---------------------------------------------------------------------------


@dataclass
class UploadResult:
    hands: list[BuiltHand]
    files: int
    skipped: dict[str, int]  # reason -> count
    other_games: int
    examples: dict[str, str]  # reason -> one example detail (for the import report)


def build_upload(data: bytes, filename: str = "") -> UploadResult:
    """Read every hand of an upload; duplicates INSIDE the upload count once."""
    files = read_upload(data, filename)
    if not files:
        raise UploadError("no_text_files", "the zip has no hand-history .txt files")
    built: dict[str, BuiltHand] = {}
    skipped: dict[str, int] = {}
    examples: dict[str, str] = {}
    other = 0
    count = 0
    for _name, text in files:
        for h in split_hands(text):
            count += 1
            if count > MAX_HANDS_PER_UPLOAD:
                raise UploadError("too_many_hands", f"over {MAX_HANDS_PER_UPLOAD} hands in one upload — split it")
            try:
                ph = parse_hand(h)
                if ph.key in built:
                    skipped["duplicate_in_upload"] = skipped.get("duplicate_in_upload", 0) + 1
                    continue
                built[ph.key] = build_hand(ph)
            except HandError as e:
                if e.reason in ("other_game", "not_a_bomb_pot", "not_double_board"):
                    other += 1
                skipped[e.reason] = skipped.get(e.reason, 0) + 1
                examples.setdefault(e.reason, e.detail[:160])
    if count == 0:
        raise UploadError("no_hands", "no hand histories were found in the upload")
    return UploadResult(
        hands=sorted(built.values(), key=lambda b: (b.parsed.ts, b.parsed.hand_id)),
        files=len(files), skipped=skipped, other_games=other, examples=examples,
    )
