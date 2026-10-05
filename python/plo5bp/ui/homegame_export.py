"""A home-games session's hand history as a file (FEAT-003).

Pure formatting — no database, no tables, no locks: ``homegame.py`` hands it the
stored hand records ALREADY filtered for the viewer by ``_hand_for_viewer`` (the
live table's reveal rule: your own cards, plus hands tabled at a showdown or
shown), so an export can never show a card the table would not.

Two formats:

- ``hand_history_text`` — a readable history in the spirit of the classic
  online-poker text logs (seats and stacks, the antes, your cards, each street's
  two boards and the action, the showdown, who won what), for reading, sharing
  or pasting into notes. Double-board bomb pots are not a format any tracker
  imports, so it is written for people, not parsers.
- ``hand_history_json`` — the same hands as data (the stored record format,
  ``HAND_RECORD_VERSION``), for anyone who wants to study them with their own tools.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

RANKS = "23456789TJQKA"
SUITS = "cdhs"  # (the engine's order: card = rank * 4 + suit)
RED_SUITS = (1, 2)  # diamonds, hearts
STREETS = ("flop", "turn", "river")
PLO67_DEALT = 4  # (PLO67: four cards at the deal, one more per red burn)

#: What an export file says it is (the JSON's "format" + its version).
EXPORT_FORMAT = "wrapgto.homegame.hands"
EXPORT_VERSION = 1


def card(c: Any) -> str:
    """``As``, ``Td``, ``2c`` — ``??`` for a face-down card."""
    try:
        c = int(c)
    except (TypeError, ValueError):
        return "??"
    if not 0 <= c < 52:
        return "??"
    return RANKS[c // 4] + SUITS[c % 4]


def cards(cs: list[Any] | None) -> str:
    return "[" + " ".join(card(c) for c in (cs or [])) + "]"


def money(cents: Any) -> str:
    v = int(cents or 0)
    return f"{'-' if v < 0 else ''}${abs(v) / 100:,.2f}"


def signed(cents: Any) -> str:
    v = int(cents or 0)
    return ("+" if v > 0 else "") + money(v)


def _when(iso: Any) -> str:
    """"2026-09-28 21:40 UTC" from a stored ISO time."""
    try:
        dt = datetime.fromisoformat(str(iso))
    except (TypeError, ValueError):
        return str(iso or "")
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _visible(hole: Any) -> bool:
    return bool(hole) and all(isinstance(c, int) and c >= 0 for c in hole)


def _equity_line(shares: dict[str, Any], name: dict[int, str]) -> str:
    """"All in · equity, board 1 / board 2: Ann 62% / 41%, Bo 38% / 59%" — an
    all-in runout's shares on one street (record v2 ``equities``)."""
    def pct(x: Any) -> str:
        return f"{round(float(x) * 100)}%" if x is not None else "–"

    parts = [
        f"{name.get(int(s), f'Seat {int(s) + 1}')} {pct(v[0])} / {pct(v[1])}"
        for s, v in sorted(shares.items(), key=lambda kv: int(kv[0]))
    ]
    return "All in · equity, board 1 / board 2: " + ", ".join(parts)


def hand_text(rec: dict[str, Any], game_label: str) -> str:
    """One hand, as the viewer may see it."""
    seats = sorted(rec.get("seats") or [], key=lambda s: int(s.get("seat", 0)))
    name = {int(s["seat"]): str(s.get("name") or f"Seat {int(s['seat']) + 1}") for s in seats}
    ba, bb = list(rec.get("board_a") or []), list(rec.get("board_b") or [])
    burns = list(rec.get("burns") or [])
    me = next((s for s in seats if s.get("is_me") and _visible(s.get("hole"))), None)
    seq = list(me.get("hole_seq") or []) if me is not None else []
    counts = list(me.get("counts") or []) if me is not None else []
    out: list[str] = [
        f"Hand #{rec.get('hand_no')} · {game_label} · {_when(rec.get('ended_at'))}",
        f"Big blind {money(rec.get('bb_cents'))} · ante {money(rec.get('ante_cents'))} · "
        f"button on seat {int(rec.get('button', 0)) + 1}",
    ]
    for s in seats:
        you = " (you)" if s.get("is_me") else ""
        out.append(f"Seat {int(s['seat']) + 1}: {name[int(s['seat'])]}{you} ({money(s.get('start_cents'))})")
    out.append(f"Everyone antes {money(rec.get('ante_cents'))}.")
    if me is not None:
        # PLO67: the four dealt first; each red burn's card is dealt on its street below
        first = seq[:PLO67_DEALT] if (seq and counts) else list(me["hole"])
        out.append(f"Dealt to {name[int(me['seat'])]}: {cards(first)}")
    actions = list(rec.get("actions") or [])
    # the bet nobody matched (record v3): back to its owner once the betting is over
    uncalled = rec.get("uncalled") or None
    last_street = actions[-1].get("street") if actions else None
    held = PLO67_DEALT
    equities = rec.get("equities") or {}
    runout_from = int(rec.get("runout_from") or 6)
    for k, street in enumerate(STREETS):
        n = 3 + k
        if len(ba) < n:
            break
        head = f"*** {street.upper()} ***"
        if k < len(burns):
            red = int(burns[k]) % 4 in RED_SUITS
            head += f" burn {card(burns[k])}{' (red: everyone still in gets a card)' if red else ''} ·"
        if street == "flop":
            head += f" board 1 {cards(ba[:3])} · board 2 {cards(bb[:3])}"
        else:
            head += (f" board 1 {cards(ba[: n - 1])} [{card(ba[n - 1])}] · "
                     f"board 2 {cards(bb[: n - 1])} [{card(bb[n - 1])}]")
        out.append(head)
        if me is not None and seq and k < len(counts) and counts[k] > held:
            out.append(f"Dealt to {name[int(me['seat'])]}: {cards(seq[held:counts[k]])}")
            held = counts[k]
        for a in actions:
            if a.get("street") != street:
                continue
            seat = int(a.get("seat", -1))
            label = str(a.get("label") or "?")
            # (the network's moves at a seat it played or assisted — homegame_bot)
            note = {"auto": " (played by the network)", "assist": " (with the network's suggestion)"}.get(
                a.get("bot") or "", " (on the clock)" if a.get("auto") else "")
            out.append(f"{name.get(seat, f'Seat {seat + 1}')}: {label[:1].lower() + label[1:]}{note}")
        if uncalled and street == last_street:
            who = int(uncalled.get("seat", -1))
            out.append(f"Uncalled bet ({money(uncalled.get('cents'))}) returned to "
                       f"{name.get(who, f'Seat {who + 1}')}")
        # everyone all in: each street still to be run out, the equities it showed
        shares = equities.get(str(n)) if n >= runout_from else None
        if shares:
            out.append(_equity_line(shares, name))
    tabled =[s for s in seats if s.get("shown") and _visible(s.get("hole"))]
    if rec.get("showdown") or tabled:
        out.append("*** SHOWDOWN ***" if rec.get("showdown") else "*** SHOWN ***")
        for s in tabled:
            out.append(f"{name[int(s['seat'])]}: shows {cards(s['hole'])}")
    out.append("*** SUMMARY ***")
    out.append(f"Total pot {money(rec.get('pot_cents'))} · board 1 {cards(ba)} · board 2 {cards(bb)}"
               + (f" · burns {cards(burns)}" if burns else ""))
    for a in rec.get("awards") or []:
        winners = [name.get(int(w), f"Seat {int(w) + 1}") for w in a.get("winners") or []]
        board = {"a": "Board 1", "b": "Board 2"}.get(str(a.get("board")), "The pot")
        labels = [v for v in (a.get("labels") or {}).values() if v]
        how = "" if a.get("uncontested") or not labels else f" with {labels[0]}"
        out.append(f"{board}: {' and '.join(winners) or '?'} "
                   f"{'win' if len(winners) > 1 else 'wins'} {money(a.get('cents'))}{how}")
    for s in seats:
        note = " (folded)" if s.get("folded") else ""
        out.append(f"{name[int(s['seat'])]}: {signed(s.get('delta_cents'))}{note}")
    return "\n".join(out)


def hand_history_text(table: dict[str, Any], hands: list[dict[str, Any]], viewer: str) -> str:
    """Every hand, oldest first, under a header naming the session."""
    head = [
        f"WrapGTO home game — {table.get('name')} ({table.get('game_name')})",
        f"Big blind {money(table.get('bb_cents'))} · ante {money(table.get('ante_cents'))} · "
        f"{len(hands)} hand{'s' if len(hands) != 1 else ''} · exported for {viewer} "
        f"on {_when(table.get('exported_at'))}",
        "Cards you could not see at the table are not in this file either.",
        "",
        "",
    ]
    body = [hand_text(h, str(table.get("game_label") or "")) for h in hands]
    return "\n".join(head) + ("\n\n".join(body) if body else "No hands were played.") + "\n"


def hand_history_json(table: dict[str, Any], hands: list[dict[str, Any]], viewer: str) -> str:
    return json.dumps({
        "format": EXPORT_FORMAT, "version": EXPORT_VERSION,
        "table": table, "exported_for": viewer, "hands": hands,
    }, ensure_ascii=False, indent=1)


def filename(table_name: str, when_iso: str, ext: str) -> str:
    """``wrapgto-friday-plo-2026-09-28.txt`` (ASCII only: it goes in a header)."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(table_name or "table").lower()).strip("-")[:40] or "table"
    day = str(when_iso or "")[:10]
    return f"wrapgto-{slug}{'-' + day if re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', day) else ''}.{ext}"
