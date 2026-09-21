"""Strict range-text parser for the CFR desktop app.

(review 2026-09-20 E11) The native parser (``rust_engine/src/cfr/range.rs``)
splits on commas only, reads the first two characters of a token, and falls
back to the UNIFORM range when nothing parses. So, silently:

- ``KK-TT``            → KK
- ``AA KK QQ``         → AA           (whitespace is not a separator there)
- ``AA:0.5`` / ``AsKs`` → 100% range  (nothing parsed → uniform fallback)
- ``AA,AA,AA,KK``      → AA at 3× the weight of KK

This module parses the text strictly — an unknown token is an ERROR, never a
fallback — and expands it to a canonical string built only from the two forms
the current native parser handles correctly (whole-class tokens ``AA``/``AKs``/
``AKo`` and ``combo_id:weight`` pairs), so solves are right today and stay right
when the native parser improves. It is the ONE implementation of the grammar:
``/api/validate_root``, ``/api/solve`` and the 13×13 range grid all go through it.

Grammar — tokens separated by commas and/or whitespace; each ``<hand>[:<weight>]``
with ``0 < weight <= 1``; later tokens override earlier ones per combo:

    AA  AKs  AKo  AK        a class (AK = suited + offsuit)
    QQ+                     pairs QQ and better
    A5s+                    same top card, kicker 5 and better (A5s..AKs)
    KK-TT                   pair range, either order
    A5s-A2s                 same top card, kicker range
    T9s-76s                 same gap, both cards step together
    AsKs  AdAh              one explicit combo
    0123:0.5  0123          a combo id — 3+ digits (canonical output pads to 4),
                            so "22".."99" are always PAIRS and "72" is 72s+72o
    random / 100% / empty   the full range
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from plo5bp.gto.preflop_class import (
    cards_to_combo,
    combo_to_cards,
    preflop_class_from_cards,
    preflop_class_label,
)

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
NUM_COMBOS = 1326

_FULL_RANGE = ("", "random", "100%", "any", "all")

_RE_SPLIT = re.compile(r"[,\s]+")
_RE_COMBO = re.compile(r"^([2-9TJQKA])([cdhs])([2-9TJQKA])([cdhs])$")
_RE_CLASS = re.compile(r"^([2-9TJQKA])([2-9TJQKA])([so]?)(\+?)$")
# A combo ID is 3+ digits — the canonical form zero-pads ids to 4 ("0022:1").
# One- and two-digit tokens are HAND TEXT: "22".."99" are pairs and "72" is the
# class 72s+72o. Treating "22" as combo id 22 (one arbitrary 32o hand) silently
# corrupts six of the thirteen pairs — the native parser has exactly that bug,
# which is also why canonical() never emits a bare numeric pair token.
_RE_ID = re.compile(r"^\d{3,}$")
_ID_WIDTH = 4


class RangeError(ValueError):
    """Range text that cannot be parsed. The message names the offending token."""


@dataclass
class ParsedRange:
    text: str
    #: combo id → weight in (0, 1]; board-blocked combos already removed.
    weights: dict[int, float] = field(default_factory=dict)
    #: True for empty / ``random`` — no restriction (native side: uniform unblocked).
    full: bool = False
    #: board cards; combos holding one are never part of the range.
    blocked: frozenset[int] = frozenset()

    @property
    def combos(self) -> int:
        return len(self.weights)

    @property
    def weight(self) -> float:
        return float(sum(self.weights.values()))

    def class_weights(self) -> dict[str, float]:
        """Class label → mean weight over the class's UNBLOCKED combos, in (0, 1]."""
        return _class_weights(self.weights, self.blocked)

    def canonical(self) -> str:
        """String for the native solver. ``""`` = full range."""
        if self.full:
            return ""
        return _canonical(self.weights)

    def normalized(self) -> str:
        """Readable, re-parseable text: class tokens where whole classes share a
        weight, explicit combos otherwise. What the grid writes back."""
        if self.full:
            return ""
        return _normalized(self.weights, self.blocked)

    def summary(self) -> dict[str, Any]:
        total = NUM_COMBOS if not self.blocked else sum(
            1 for cid in range(NUM_COMBOS) if not (set(combo_to_cards(cid)) & self.blocked)
        )
        n = total if self.full else self.combos
        return {
            "full": self.full,
            "combos": n,
            "weight": round(float(n) if self.full else self.weight, 4),
            "pct": round(100.0 * (float(n) if self.full else self.weight) / total, 2) if total else 0.0,
            "classes": {k: round(v, 4) for k, v in self.class_weights().items()} if not self.full else {},
            "normalized": self.normalized(),
        }


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def _canon_token(tok: str) -> str:
    """Ranks upper-case, suits / s,o suffix lower-case (``aks`` → ``AKs``)."""
    return "".join(ch.upper() if ch.upper() in _RANKS else ch.lower() for ch in tok)


def _rank(ch: str) -> int:
    return _RANKS.index(ch)


def _class_combos(hi: int, lo: int, suffix: str) -> list[int]:
    """Combo ids of one class. ``suffix``: ``s`` / ``o`` / ``""`` (both; pairs)."""
    out = []
    for s0 in range(4):
        for s1 in range(4):
            if hi == lo:
                if s0 >= s1:
                    continue
            elif suffix == "s" and s0 != s1:
                continue
            elif suffix == "o" and s0 == s1:
                continue
            out.append(cards_to_combo(hi * 4 + s0, lo * 4 + s1))
    return out


def _parse_class(tok: str, original: str) -> tuple[int, int, str, bool]:
    m = _RE_CLASS.match(tok)
    if not m:
        raise RangeError(_unknown(original))
    a, b, suffix, plus = _rank(m.group(1)), _rank(m.group(2)), m.group(3), bool(m.group(4))
    hi, lo = max(a, b), min(a, b)
    if hi == lo and suffix:
        raise RangeError(f"range token {original!r}: a pair cannot be suited/offsuit")
    return hi, lo, suffix, plus


def _expand_hand(tok: str, original: str) -> list[int]:
    """One ``<hand>`` (no weight) → combo ids."""
    if _RE_ID.match(tok):
        cid = int(tok)
        if not 0 <= cid < NUM_COMBOS:
            raise RangeError(f"range token {original!r}: combo id must be 0..{NUM_COMBOS - 1}")
        return [cid]

    m = _RE_COMBO.match(tok)
    if m:
        c0 = _rank(m.group(1)) * 4 + _SUITS.index(m.group(2))
        c1 = _rank(m.group(3)) * 4 + _SUITS.index(m.group(4))
        if c0 == c1:
            raise RangeError(f"range token {original!r}: the same card twice")
        return [cards_to_combo(c0, c1)]

    if "-" in tok:
        left, _, right = tok.partition("-")
        h1, l1, s1, p1 = _parse_class(left, original)
        h2, l2, s2, p2 = _parse_class(right, original)
        if p1 or p2:
            raise RangeError(f"range token {original!r}: '+' cannot be combined with a '-' range")
        if s1 != s2:
            raise RangeError(f"range token {original!r}: both ends must be suited, offsuit, or neither")
        out: list[int] = []
        if h1 == l1 and h2 == l2:  # KK-TT
            for r in range(min(h1, h2), max(h1, h2) + 1):
                out += _class_combos(r, r, "")
        elif h1 == l1 or h2 == l2:
            raise RangeError(f"range token {original!r}: cannot mix a pair with a non-pair")
        elif h1 == h2:  # A5s-A2s: same top card, kicker range
            for lo in range(min(l1, l2), max(l1, l2) + 1):
                out += _class_combos(h1, lo, s1)
        elif h1 - l1 == h2 - l2:  # T9s-76s: same gap, both cards step together
            gap = h1 - l1
            for hi in range(min(h1, h2), max(h1, h2) + 1):
                out += _class_combos(hi, hi - gap, s1)
        else:
            raise RangeError(
                f"range token {original!r}: ends must share the top card (A5s-A2s) "
                "or the gap between cards (T9s-76s)"
            )
        return out

    hi, lo, suffix, plus = _parse_class(tok, original)
    if not plus:
        return _class_combos(hi, lo, suffix)
    out = []
    if hi == lo:  # QQ+
        for r in range(hi, 13):
            out += _class_combos(r, r, "")
    else:  # A5s+: top card fixed, kicker and better
        for k in range(lo, hi):
            out += _class_combos(hi, k, suffix)
    return out


def _unknown(tok: str) -> str:
    return (
        f"unknown range token {tok!r} — expected e.g. AA, AKs, AKo, QQ+, A5s+, KK-TT, "
        "A5s-A2s, AsKs, optionally with a weight like AA:0.5"
    )


def parse_range(text: str | None, board: Sequence[int] | None = None) -> ParsedRange:
    """Parse range text strictly. Raises :class:`RangeError` on any bad token."""
    raw = (text or "").strip()
    blocked = frozenset(int(c) for c in (board or []))
    tokens = [t for t in _RE_SPLIT.split(raw) if t]
    if not tokens or (len(tokens) == 1 and tokens[0].lower() in _FULL_RANGE):
        return ParsedRange(text=raw, full=True, blocked=blocked)

    weights: dict[int, float] = {}
    for original in tokens:
        if original.lower() in _FULL_RANGE:
            raise RangeError(f"range token {original!r} means the full range and must stand alone")
        hand, sep, w_text = original.partition(":")
        w = 1.0
        if sep:
            try:
                w = float(w_text)
            except ValueError:
                raise RangeError(f"range token {original!r}: weight {w_text!r} is not a number") from None
            if not 0.0 < w <= 1.0:
                raise RangeError(f"range token {original!r}: weight must be > 0 and <= 1")
        if not hand:
            raise RangeError(_unknown(original))
        for cid in _expand_hand(_canon_token(hand), original):
            weights[cid] = w  # de-duplicated: a combo counts once; the last token wins

    live = {cid: w for cid, w in weights.items() if not (set(combo_to_cards(cid)) & blocked)}
    if not live:
        # The native parser would silently fall back to the 100% range here.
        raise RangeError(f"range {raw!r} has no combos left on this board")
    return ParsedRange(text=raw, weights=dict(sorted(live.items())), blocked=blocked)


# ---------------------------------------------------------------------------
# output forms
# ---------------------------------------------------------------------------


def _fmt_w(w: float) -> str:
    return f"{w:.6g}"


def _by_class(weights: dict[int, float]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for cid, w in weights.items():
        c0, c1 = combo_to_cards(cid)
        out.setdefault(preflop_class_label(preflop_class_from_cards(c0, c1)), {})[cid] = w
    return out


def _class_ids(label: str) -> list[int]:
    hi, lo = _rank(label[0]), _rank(label[1])
    return _class_combos(hi, lo, label[2:] if len(label) > 2 else "")


def _class_weights(weights: dict[int, float], blocked: frozenset[int]) -> dict[str, float]:
    out = {}
    for label, members in _by_class(weights).items():
        alive = [c for c in _class_ids(label) if not (set(combo_to_cards(c)) & blocked)]
        if alive:
            out[label] = sum(members.get(c, 0.0) for c in alive) / len(alive)
    return out


def _canonical(weights: dict[int, float]) -> str:
    """Whole classes at weight 1 → the class token; everything else → ``id:w``.

    Both forms are handled correctly by the CURRENT native parser. A class token
    is only used when ALL of the class's combos (blocked ones included — the
    native side drops those itself) are present at exactly 1.0, because the
    native expansion always means "every combo, weight 1".
    """
    parts: list[str] = []
    for label, members in _by_class(weights).items():
        ids = _class_ids(label)
        whole = len(members) == len(ids) and all(w == 1.0 for w in members.values())
        # A numeric pair label ("22".."99") is NEVER emitted: the native parser
        # reads an all-digit token as a combo id, so "22" would mean combo #22.
        if whole and not label.isdigit():
            parts.append(label)
        else:
            parts += [f"{cid:0{_ID_WIDTH}d}:{_fmt_w(w)}" for cid, w in sorted(members.items())]
    return ",".join(parts)


def _combo_text(cid: int) -> str:
    c0, c1 = combo_to_cards(cid)
    hi, lo = (c0, c1) if c0 >= c1 else (c1, c0)
    return "".join(_RANKS[c // 4] + _SUITS[c % 4] for c in (hi, lo))


def _normalized(weights: dict[int, float], blocked: frozenset[int]) -> str:
    parts: list[str] = []
    for label, members in _by_class(weights).items():
        alive = [c for c in _class_ids(label) if not (set(combo_to_cards(c)) & blocked)]
        ws = {members.get(c) for c in alive}
        if len(ws) == 1 and None not in ws:  # every live combo present at one weight
            w = ws.pop()
            parts.append(label if w == 1.0 else f"{label}:{_fmt_w(w)}")
        else:
            parts += [
                _combo_text(c) + ("" if w == 1.0 else f":{_fmt_w(w)}")
                for c, w in sorted(members.items())
            ]
    return ",".join(parts)


def toggle_class(text: str | None, label: str, board: Sequence[int] | None = None) -> ParsedRange:
    """Flip one 13×13 cell: drop the class if any of it is in the range, else add
    it at weight 1. Used by the range grid so the grid and the text box can never
    disagree about what ``QQ+`` or ``A5s-A2s`` covers."""
    blocked = frozenset(int(c) for c in (board or []))
    pr = parse_range(text, board)
    hi, lo, suffix, plus = _parse_class(_canon_token(label), label)
    if plus or (hi != lo and not suffix):
        raise RangeError(f"grid cell {label!r} must be one class (AA, AKs or AKo)")
    ids = [c for c in _class_combos(hi, lo, suffix) if not (set(combo_to_cards(c)) & blocked)]
    if pr.full:
        # Full range → everything except this class.
        weights = {
            cid: 1.0 for cid in range(NUM_COMBOS) if not (set(combo_to_cards(cid)) & blocked)
        }
    else:
        weights = dict(pr.weights)
    if any(c in weights for c in ids):
        for c in ids:
            weights.pop(c, None)
    else:
        for c in ids:
            weights[c] = 1.0
    return ParsedRange(text=text or "", weights=dict(sorted(weights.items())), blocked=blocked)


def attach_range_text(report: dict[str, Any], root_d: dict[str, Any]) -> None:
    """Copy ``range_*_text`` (the user's wording) onto a report's root, in place.

    The solver echoes back the canonical string it was given — up to ~12 KB of
    ``id:weight`` pairs. The viewer shows the text the user actually typed.
    """
    rep_root = report.get("root")
    if not isinstance(rep_root, dict):
        return
    for key in ("range_oop_text", "range_ip_text"):
        if root_d.get(key):
            rep_root[key] = str(root_d[key])


def apply_ranges(root_d: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate + canonicalize ``range_oop`` / ``range_ip`` on a root dict.

    Returns ``(new_root_dict, info)``. The new dict carries the canonical strings
    (what the solver must see) and keeps the user's wording under
    ``range_*_text`` for display. Raises :class:`RangeError` naming the field.
    """
    out = dict(root_d)
    info: dict[str, Any] = {}
    board = [int(c) for c in (root_d.get("board") or [])]
    for which in ("oop", "ip"):
        key = f"range_{which}"
        # Prefer the user's wording when this dict was already canonicalized.
        text = str(root_d.get(f"{key}_text") or root_d.get(key) or "")
        try:
            pr = parse_range(text, board)
        except RangeError as e:
            raise RangeError(f"{key}: {e}") from None
        out[key] = pr.canonical()
        out[f"{key}_text"] = "" if pr.full else text.strip()
        info[which] = pr.summary()
    return out, info
