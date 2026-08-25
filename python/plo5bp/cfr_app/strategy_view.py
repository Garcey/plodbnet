"""Transform raw CFR SolveReport / chart JSON into UI-friendly views.

Handles:
- Full ``SolveReport`` dumps (``status/root/strategy.infosets``)
- Push/fold chart nodes (``hands[]`` with class labels)
- Preflop 169-class → 13×13 matrix aggregation
- Combo (0..1325) hand labels for postflop
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from plo5bp.gto.preflop_class import (
    NUM_PREFLOP_CLASSES,
    combo_to_cards,
    preflop_class_from_cards,
    preflop_class_from_id,
    preflop_class_label,
)

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
_SUIT_SYMBOLS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}

_RE_PRIV = re.compile(r"_c(?P<priv>\d+)$")
_RE_SEAT = re.compile(r"(?:^|_)p(?P<seat>\d+)(?:_|$)")
_RE_PATH = re.compile(r"_path(?P<path>[^_]+(?:_[^_]+)*?)_c\d+$")
_RE_HIST = re.compile(r"_h(?P<hist>\d+)_c\d+$")


def card_to_str(card: int) -> str:
    c = int(card)
    if not 0 <= c < 52:
        return f"?{c}"
    return f"{_RANKS[c // 4]}{_SUITS[c % 4]}"


def card_to_pretty(card: int) -> str:
    c = int(card)
    if not 0 <= c < 52:
        return f"?{c}"
    r, s = _RANKS[c // 4], _SUITS[c % 4]
    return f"{r}{_SUIT_SYMBOLS[s]}"


def board_to_str(board: Sequence[int]) -> str:
    return " ".join(card_to_pretty(c) for c in board)


def combo_label(combo_id: int) -> str:
    try:
        c0, c1 = combo_to_cards(int(combo_id))
        return f"{card_to_str(c0)}{card_to_str(c1)}"
    except ValueError:
        return f"c{combo_id}"


def parse_infoset_id(iid: str) -> dict[str, Any]:
    """Extract seat / path / private view from an infoset id string."""
    s = str(iid or "")
    out: dict[str, Any] = {
        "infoset_id": s,
        "seat": 0,
        "path": "",
        "history_hash": None,
        "private": None,
        "private_kind": None,  # "class" | "combo"
        "hand_label": "",
    }
    m = _RE_SEAT.search(s)
    if m:
        out["seat"] = int(m.group("seat"))
    m = _RE_PATH.search(s)
    if m:
        out["path"] = m.group("path")
    m = _RE_HIST.search(s)
    if m:
        out["history_hash"] = int(m.group("hist"))
    m = _RE_PRIV.search(s)
    if m:
        priv = int(m.group("priv"))
        out["private"] = priv
        # Prefer id-prefix over numeric range: river combos are 0..1325 and
        # collide with preflop class ids 0..168 if we only check priv < 169.
        #   pf_ / mwpf_ / chart_  → 169-class
        #   p{seat}_h{hist}_c… / mw_ → combo (postflop)
        is_preflop_class = (
            s.startswith("pf_")
            or s.startswith("mwpf_")
            or s.startswith("chart_")
            or "chart_" in s
        )
        is_postflop_combo = (
            s.startswith("mw_")
            or ("_h" in s and "_c" in s and not s.startswith("pf_") and not s.startswith("mwpf_"))
        )
        if is_preflop_class and priv < NUM_PREFLOP_CLASSES:
            out["private_kind"] = "class"
            out["hand_label"] = preflop_class_label(priv)
        elif is_postflop_combo:
            out["private_kind"] = "combo"
            out["hand_label"] = combo_label(priv) if 0 <= priv < 1326 else f"c{priv}"
        elif priv < NUM_PREFLOP_CLASSES and not is_postflop_combo:
            # Charts / bare class ids without a known prefix
            out["private_kind"] = "class"
            out["hand_label"] = preflop_class_label(priv)
        elif 0 <= priv < 1326:
            out["private_kind"] = "combo"
            out["hand_label"] = combo_label(priv)
        else:
            out["private_kind"] = "combo"
            out["hand_label"] = f"c{priv}"
    # mwpf open path often uses path token "open" or empty
    if "mwpf" in s and not out["path"]:
        if "_path" not in s:
            out["path"] = "open"
    # Normalize path for history-hash-only nodes so UI filters match node keys
    if not out["path"] and out["history_hash"] is not None:
        out["path"] = str(out["history_hash"])
    return out


def action_short(action: str) -> str:
    a = str(action).upper()
    if a == "FOLD":
        return "F"
    if a == "CHECK_CALL":
        return "X/C"
    if a == "ALLIN":
        return "AI"
    if a.startswith("RAISE_"):
        try:
            pm = int(a.split("_", 1)[1])
            return f"R{pm / 10:g}%"
        except ValueError:
            return a
    return a


def action_color_class(action: str) -> str:
    a = str(action).upper()
    if a == "FOLD":
        return "act-fold"
    if a == "CHECK_CALL":
        return "act-call"
    if a == "ALLIN":
        return "act-allin"
    if a.startswith("RAISE_"):
        return "act-raise"
    return "act-other"


def _normalize_probs(actions: Sequence[str], probs: Sequence[float]) -> list[dict[str, Any]]:
    out = []
    for a, p in zip(actions, probs):
        out.append(
            {
                "action": str(a),
                "short": action_short(a),
                "prob": float(p),
                "pct": round(float(p) * 100.0, 2),
                "css": action_color_class(a),
            }
        )
    return out


def infoset_row(raw: dict[str, Any]) -> dict[str, Any]:
    """One UI row from a raw infoset dict."""
    meta = parse_infoset_id(str(raw.get("infoset_id", "")))
    # Dump schema v2 is authoritative (combo 0..168 is not a 169-class).
    if raw.get("private_kind"):
        meta["private_kind"] = str(raw["private_kind"])
    if raw.get("private_id") is not None:
        meta["private"] = int(raw["private_id"])
    if raw.get("actor") is not None:
        meta["seat"] = int(raw["actor"])
    if isinstance(raw.get("path"), list):
        meta["path"] = ",".join(str(x) for x in raw["path"])
    actions = list(raw.get("actions") or [])
    probs = [float(x) for x in (raw.get("probs") or [])]
    # pad / trim
    if len(probs) < len(actions):
        probs = probs + [0.0] * (len(actions) - len(probs))
    probs = probs[: len(actions)]
    return {
        **meta,
        "actions": actions,
        "probs": probs,
        "strategy": _normalize_probs(actions, probs),
        "primary_action": (
            actions[max(range(len(probs)), key=lambda i: probs[i])] if actions else ""
        ),
        "primary_prob": max(probs) if probs else 0.0,
    }


def load_report(path: Path | str | dict[str, Any]) -> dict[str, Any]:
    """Load a SolveReport JSON (or pass-through dict) into a normalized view."""
    if isinstance(path, dict):
        data = path
        source = "<memory>"
    else:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        source = str(p)

    # Chart node format (pushfold hands)
    if "hands" in data and "strategy" not in data:
        return _view_from_chart(data, source=source)

    root = data.get("root") or {}
    strategy = data.get("strategy") or {}
    infosets_raw = strategy.get("infosets") or data.get("infosets") or []
    rows = [infoset_row(x) for x in infosets_raw if isinstance(x, dict)]

    street = int(root.get("street", -1)) if root else -1
    class_rows_n = sum(1 for r in rows[:500] if r.get("private_kind") == "class")
    combo_rows_n = sum(1 for r in rows[:500] if r.get("private_kind") == "combo")
    is_preflop_class = class_rows_n > 0 and class_rows_n >= combo_rows_n
    if street > 0:
        # Postflop: never treat combo ids 0..168 as preflop classes.
        is_preflop_class = class_rows_n > combo_rows_n and class_rows_n >= 50

    matrix = matrix_for_rows(rows, street=street)

    nodes = group_by_node(rows)
    from plo5bp.cfr_app.tree_model import (
        aggregate_node,
        build_line_nav,
        build_solution_tree,
        try_load_chart_pack,
    )

    solution_tree = build_solution_tree(rows)
    quality = quality_summary(rows, data)
    chart_pack = try_load_chart_pack(source)
    line_nav = build_line_nav(
        nodes,
        chart_pack=chart_pack,
        num_seats=int(root.get("num_seats") or 0) or None,
        street=street,
    )
    summary = {
        "source": source,
        "status": data.get("status", "ok"),
        "root": root,
        "config": data.get("config") or {},
        "iterations_run": data.get("iterations_run"),
        "exploitability_bb": data.get("exploitability_bb"),
        "notes": list(data.get("notes") or []),
        "num_infosets": len(rows),
        "num_nodes": len(nodes),
        "board_str": board_to_str(root.get("board") or []),
        "street": street,
        "is_preflop_class": is_preflop_class,
        "kind": "solve_report",
        "range_ip": root.get("range_ip") or "",
        "range_oop": root.get("range_oop") or "",
        "quality": quality,
        "navigable": bool(line_nav.get("navigable")),
    }
    return {
        "summary": summary,
        "rows": rows,
        "nodes": nodes,
        "matrix": matrix,
        "solution_tree": solution_tree,
        "line_nav": line_nav,
        "chart_pack": chart_pack,
        "raw_keys": list(data.keys()),
    }


def _view_from_chart(data: dict[str, Any], *, source: str) -> dict[str, Any]:
    hands = data.get("hands") or []
    rows: list[dict[str, Any]] = []
    for h in hands:
        actions = list(h.get("actions") or [])
        probs = [float(x) for x in (h.get("probs") or [])]
        label = str(h.get("hand") or "")
        cid = h.get("class_id")
        if cid is None and label:
            # best-effort: leave private None
            private = None
            kind = "class"
        else:
            private = int(cid) if cid is not None else None
            kind = "class"
        row = {
            "infoset_id": f"chart_{data.get('node_id', '')}_{label}",
            "seat": int(data.get("seat_index", 0)),
            "path": str(data.get("path") or "open"),
            "history_hash": None,
            "private": private,
            "private_kind": kind,
            "hand_label": label or (preflop_class_label(private) if private is not None else ""),
            "actions": actions,
            "probs": probs,
            "strategy": _normalize_probs(actions, probs),
            "primary_action": (
                actions[max(range(len(probs)), key=lambda i: probs[i])] if actions else ""
            ),
            "primary_prob": max(probs) if probs else 0.0,
        }
        # inject allin/fold convenience if present
        if "allin" in h:
            row["allin"] = float(h["allin"])
        if "fold" in h:
            row["fold"] = float(h["fold"])
        rows.append(row)

    matrix = build_preflop_matrix(rows)
    from plo5bp.cfr_app.tree_model import (
        aggregate_node,
        build_line_nav,
        build_solution_tree,
        try_load_chart_pack,
    )

    chart_path = str(data.get("path") or "open")
    chart_seat = int(data.get("seat_index", 0))
    nodes = [
        {
            "node_key": f"s{chart_seat}_path{chart_path}",
            "seat": chart_seat,
            "path": chart_path,
            "label": f"P{chart_seat} · {humanize_path(chart_path)}",
            "path_pretty": humanize_path(chart_path),
            "num_hands": len(rows),
            "description": data.get("description") or "",
            "aggregate": aggregate_node(rows),
        }
    ]
    solution_tree = build_solution_tree(rows)
    quality = quality_summary(rows, data)
    chart_pack = try_load_chart_pack(source)
    line_nav = build_line_nav(
        nodes, chart_pack=chart_pack, num_seats=None, street=0
    )
    summary = {
        "source": source,
        "status": "ok",
        "root": {
            "root_id": data.get("node_id"),
            "street": 0,
            "chart": True,
        },
        "config": {},
        "iterations_run": None,
        "exploitability_bb": None,
        "notes": [
            f"chart node {data.get('node_id')}",
            f"seat={data.get('seat')} path={data.get('path')}",
        ],
        "num_infosets": len(rows),
        "num_nodes": 1,
        "board_str": "",
        "street": 0,
        "is_preflop_class": True,
        "kind": "chart",
        "quality": quality,
        "navigable": True,
        "chart_meta": {
            "node_id": data.get("node_id"),
            "seat": data.get("seat"),
            "path": data.get("path"),
            "description": data.get("description"),
            "mean_allin": data.get("mean_allin"),
        },
    }
    return {
        "summary": summary,
        "rows": rows,
        "nodes": nodes,
        "matrix": matrix,
        "solution_tree": solution_tree,
        "line_nav": line_nav,
        "chart_pack": chart_pack,
        "raw_keys": list(data.keys()),
    }


def quality_summary(rows: Sequence[dict[str, Any]], data: dict[str, Any]) -> dict[str, Any]:
    """Aggregate quality metrics available from strategy dumps (probs + report fields).

    Native dumps are frequency-only (no per-hand CFV in export). We surface:
    exploitability, mean entropy, fold/call/raise/allin mass, pure-strategy
    fraction, and top aggressive hands — enough to judge solve quality before training.
    """
    from plo5bp.cfr_app.tree_model import aggregate_node

    agg = aggregate_node(rows)
    pure = 0
    for r in rows:
        pp = float(r.get("primary_prob") or 0)
        if pp >= 0.99:
            pure += 1
    n = max(1, len(rows))
    # weight proxy: each hand equal mass (class/combo uniform prior)
    return {
        "exploitability_bb": data.get("exploitability_bb"),
        "iterations_run": data.get("iterations_run"),
        "num_infosets": len(rows),
        "mean_entropy": agg.get("entropy"),
        "mean_fold": agg.get("fold"),
        "mean_call": agg.get("call"),
        "mean_raise": agg.get("raise"),
        "mean_allin": agg.get("allin"),
        "pure_strategy_frac": round(pure / n, 4),
        "agg_ge_50pct": agg.get("agg_ge_50pct"),
        "fold_ge_50pct": agg.get("fold_ge_50pct"),
        "mean_mix": agg.get("mean_mix"),
        "notes": [
            "Strategy JSON is frequency-only; CFV/EV not in native dump.",
            "Hand weights = uniform prior over infoset private views.",
            "Exploitability from solver report when present.",
        ],
    }


def humanize_path(path: str | None) -> str:
    """Readable action-line label for the node strip (not a raw history hash)."""
    p = str(path or "").strip()
    if p in ("", "root", "open"):
        return "Open"
    if p.isdigit() and len(p) >= 8:
        return f"Line {p[-4:]}"
    pretty = (
        p.replace(",", " → ")
        .replace("_", " → ")
        .replace("RAISE_", "R")
        .replace("CHECK_CALL", "X/C")
        .replace("FOLD", "F")
        .replace("ALLIN", "AI")
    )
    return pretty or "Open"


def group_by_node(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    from plo5bp.cfr_app.tree_model import aggregate_node

    buckets: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for r in rows:
        # path already normalized to history_hash string in parse_infoset_id
        key = (int(r.get("seat", 0)), str(r.get("path") or r.get("history_hash") or "root"))
        buckets.setdefault(key, []).append(r)
    nodes = []
    # Largest nodes first within each seat so the strip is usable.
    items = sorted(
        buckets.items(),
        key=lambda x: (x[0][0], -len(x[1]), str(x[0][1])),
    )
    for (seat, path), rs in items:
        agg = aggregate_node(rs)
        label = humanize_path(path)
        nodes.append(
            {
                "node_key": f"s{seat}_{path}",
                "seat": seat,
                "path": path,
                "label": f"P{seat} · {label}",
                "path_pretty": label,
                "num_hands": len(rs),
                "mean_primary": (
                    sum(float(x.get("primary_prob") or 0) for x in rs) / len(rs) if rs else 0.0
                ),
                "aggregate": agg,
            }
        )
    return nodes


def matrix_for_rows(
    rows: Sequence[dict[str, Any]],
    *,
    street: int | None = None,
) -> dict[str, Any]:
    """13×13 grid: true preflop classes, or combo→class average for postflop."""
    class_n = sum(1 for r in rows[:800] if r.get("private_kind") == "class")
    combo_n = sum(1 for r in rows[:800] if r.get("private_kind") == "combo")
    want_class = class_n > 0 and (street in (None, 0) or class_n >= combo_n)
    if want_class:
        m = build_preflop_matrix(rows)
        if not m.get("empty"):
            return m
    if combo_n > 0:
        return build_class_matrix_from_combos(rows)
    return build_preflop_matrix(rows)


def build_class_matrix_from_combos(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Average combo strategies into the 169-class 13×13 (postflop display)."""
    combo_rows = [
        r
        for r in rows
        if r.get("private_kind") == "combo" and r.get("private") is not None
    ]
    empty = {
        "cells": [],
        "rank_labels": list(reversed(_RANKS)),
        "empty": True,
        "aggregated_from_combos": True,
    }
    if not combo_rows:
        return empty

    def _node_key(r: dict[str, Any]) -> tuple[int, str]:
        path = r.get("path")
        if path is None or path == "":
            path = r.get("history_hash")
        return (int(r.get("seat", 0)), str(path if path is not None else "root"))

    by_node: dict[tuple[int, str], list] = {}
    for r in combo_rows:
        by_node.setdefault(_node_key(r), []).append(r)
    best_key = max(by_node.keys(), key=lambda k: len(by_node[k]))
    selected = by_node[best_key]

    acc: dict[int, dict[str, float]] = {}
    counts: dict[int, int] = {}
    last_strat: dict[int, list] = {}
    for r in selected:
        priv = int(r["private"])
        if not 0 <= priv < 1326:
            continue
        try:
            c0, c1 = combo_to_cards(priv)
            cid = preflop_class_from_cards(c0, c1)
        except ValueError:
            continue
        fold = call = agg = allin = 0.0
        for s in r.get("strategy") or []:
            a = str(s.get("action", "")).upper()
            p = float(s.get("prob") or 0)
            if a == "FOLD":
                fold += p
            elif a in ("CHECK_CALL", "CHECK", "CALL"):
                call += p
            elif a == "ALLIN":
                allin += p
                agg += p
            else:
                agg += p
        if cid not in acc:
            acc[cid] = {"fold": 0.0, "call": 0.0, "agg": 0.0, "allin": 0.0}
            counts[cid] = 0
        acc[cid]["fold"] += fold
        acc[cid]["call"] += call
        acc[cid]["agg"] += agg
        acc[cid]["allin"] += allin
        counts[cid] += 1
        last_strat[cid] = list(r.get("strategy") or [])

    cells: list[list[dict[str, Any] | None]] = [[None] * 13 for _ in range(13)]
    placed = 0
    for cid, sums in acc.items():
        n = max(1, counts.get(cid, 1))
        try:
            hi, lo, suited = preflop_class_from_id(cid)
        except ValueError:
            continue
        if hi == lo:
            ri, ci = 12 - hi, 12 - lo
        elif suited:
            ri, ci = 12 - hi, 12 - lo
        else:
            ri, ci = 12 - lo, 12 - hi
        fold_p = sums["fold"] / n
        call_p = sums["call"] / n
        aggressive = sums["agg"] / n
        allin_p = sums["allin"] / n
        cells[ri][ci] = {
            "class_id": cid,
            "label": preflop_class_label(cid),
            "hi": hi,
            "lo": lo,
            "suited": suited if hi != lo else None,
            "strategy": last_strat.get(cid) or [],
            "fold": round(fold_p, 4),
            "call": round(call_p, 4),
            "agg": round(aggressive, 4),
            "allin": round(allin_p, 4),
            "primary": None,
            "primary_prob": max(fold_p, call_p, aggressive),
            "n_combos": n,
        }
        placed += 1

    return {
        "empty": placed == 0,
        "aggregated_from_combos": True,
        "rank_labels": list(reversed(_RANKS)),
        "seat": best_key[0],
        "path": best_key[1],
        "num_classes": len(acc),
        "num_filled": placed,
        "cells": cells,
    }


def build_preflop_matrix(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate class-labelled rows into a 13×13 Monker/GTOW-style grid.

    Display convention (rank labels A..2 on both axes):
      - diagonal = pairs
      - above diagonal (col > row) = suited
      - below diagonal (col < row) = offsuit

    Prefer rows that share the most common (seat, path) group when mixed.
    Path falls back to history_hash so HU preflop history buckets don't merge.
    """
    class_rows = [
        r for r in rows if r.get("private_kind") == "class" and r.get("private") is not None
    ]
    if not class_rows:
        # try hand_label → class
        for r in rows:
            lab = r.get("hand_label") or ""
            if lab and len(lab) >= 2:
                try:
                    cid = _label_to_class(lab)
                    r = dict(r)
                    r["private"] = cid
                    r["private_kind"] = "class"
                    class_rows.append(r)
                except ValueError:
                    pass
    if not class_rows:
        return {"cells": [], "rank_labels": list(reversed(_RANKS)), "empty": True}

    def _node_key(r: dict[str, Any]) -> tuple[int, str]:
        path = r.get("path")
        if path is None or path == "":
            path = r.get("history_hash")
        return (int(r.get("seat", 0)), str(path if path is not None else "root"))

    # Prefer largest single node (or the only one when already filtered)
    by_node: dict[tuple[int, str], list] = {}
    for r in class_rows:
        by_node.setdefault(_node_key(r), []).append(r)
    best_key = max(by_node.keys(), key=lambda k: len(by_node[k]))
    selected = by_node[best_key]

    by_class: dict[int, dict[str, Any]] = {}
    for r in selected:
        cid = int(r["private"])
        by_class[cid] = r  # last wins if duplicates

    cells: list[list[dict[str, Any] | None]] = [[None] * 13 for _ in range(13)]
    placed = 0
    for cid, r in by_class.items():
        try:
            hi, lo, suited = preflop_class_from_id(cid)
        except ValueError:
            continue
        # Pair on diagonal; suited ABOVE diagonal; offsuit BELOW.
        # Display row/col = 12 - rank (A=0 .. 2=12).
        if hi == lo:
            ri, ci = 12 - hi, 12 - lo
        elif suited:
            ri, ci = 12 - hi, 12 - lo  # hi > lo → ri < ci → above diagonal
        else:
            ri, ci = 12 - lo, 12 - hi  # below diagonal
        strat = r.get("strategy") or []
        aggressive = 0.0
        fold_p = 0.0
        call_p = 0.0
        allin_p = 0.0
        for s in strat:
            a = str(s.get("action", "")).upper()
            p = float(s.get("prob") or 0)
            if a == "FOLD":
                fold_p += p
            elif a == "CHECK_CALL":
                call_p += p
            elif a == "ALLIN":
                allin_p += p
                aggressive += p
            else:
                aggressive += p
        cells[ri][ci] = {
            "class_id": cid,
            "label": preflop_class_label(cid),
            "hi": hi,
            "lo": lo,
            "suited": suited if hi != lo else None,
            "strategy": strat,
            "fold": round(fold_p, 4),
            "call": round(call_p, 4),
            "agg": round(aggressive, 4),
            "allin": round(allin_p, 4),
            "primary": r.get("primary_action"),
            "primary_prob": r.get("primary_prob"),
        }
        placed += 1

    return {
        "empty": placed == 0,
        "rank_labels": list(reversed(_RANKS)),  # A..2
        "seat": best_key[0],
        "path": best_key[1],
        "num_classes": len(by_class),
        "num_filled": placed,
        "cells": cells,
    }


def _label_to_class(label: str) -> int:
    lab = label.strip()
    if len(lab) == 2 and lab[0] == lab[1]:
        # pair
        hi = _RANKS.index(lab[0].upper())
        return hi
    if len(lab) == 3:
        hi = _RANKS.index(lab[0].upper())
        lo = _RANKS.index(lab[1].upper())
        suited = lab[2].lower() == "s"
        # use representative cards
        if suited:
            c0, c1 = hi * 4, lo * 4  # same suit 0
        else:
            c0, c1 = hi * 4, lo * 4 + 1
        return preflop_class_from_cards(c0, c1)
    raise ValueError(f"bad hand label {label!r}")


def filter_rows(
    rows: Sequence[dict[str, Any]],
    *,
    seat: int | None = None,
    path: str | None = None,
    hand_query: str = "",
    limit: int = 500,
    offset: int = 0,
) -> dict[str, Any]:
    q = (hand_query or "").strip().lower()
    out = []
    for r in rows:
        if seat is not None and int(r.get("seat", 0)) != int(seat):
            continue
        if path is not None and path != "":
            # Match path OR history_hash string (postflop / HU preflop nodes)
            rpath = str(r.get("path") or "")
            rhist = r.get("history_hash")
            rhist_s = str(rhist) if rhist is not None else ""
            if path != rpath and path != rhist_s and path != f"h{rhist_s}":
                continue
        if q:
            lab = str(r.get("hand_label") or "").lower()
            iid = str(r.get("infoset_id") or "").lower()
            if q not in lab and q not in iid:
                continue
        out.append(r)
    total = len(out)
    page = out[offset : offset + max(1, limit)]
    return {"total": total, "offset": offset, "limit": limit, "rows": page}


def list_strategy_library(
    roots: Sequence[Path | str] | None = None,
    *,
    max_files: int = 500,
) -> list[dict[str, Any]]:
    """Scan known CFR data dirs for loadable JSON strategy files."""
    if roots is None:
        # repo-relative defaults
        repo = Path(__file__).resolve().parents[3]
        roots = [
            repo / "data" / "cfr" / "uploads",
            repo / "data" / "cfr" / "app_export",
            repo / "data" / "cfr" / "app_jobs",
            repo / "data" / "cfr" / "overnight" / "strategies",
            repo / "data" / "cfr" / "bench",
            repo / "data" / "cfr" / "verify" / "batch" / "strategies",
            repo / "data" / "cfr" / "pushfold_14_charts",
            repo / "data" / "cfr",
        ]
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for root in roots:
        r = Path(root)
        if not r.exists():
            continue
        paths = sorted(r.rglob("*.json") if r.is_dir() else [r], key=lambda p: p.stat().st_mtime, reverse=True)
        for p in paths:
            if not p.is_file():
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            # skip grid/status/index noise optionally — still include INDEX as meta
            name = p.name.lower()
            if name in ("status.json", "overnight_grid.json", "manifest.json", "plan.json", "certificate.json"):
                continue
            seen.add(key)
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            items.append(
                {
                    "path": str(p),
                    "name": p.name,
                    "rel": _rel_to_repo(p),
                    "size": size,
                    "size_kb": round(size / 1024.0, 1),
                    "mtime": p.stat().st_mtime,
                    "dir": str(p.parent),
                }
            )
            if len(items) >= max_files:
                return items
    return items


def _rel_to_repo(p: Path) -> str:
    repo = Path(__file__).resolve().parents[3]
    try:
        return str(p.resolve().relative_to(repo))
    except ValueError:
        return str(p)


def summarize_report_light(path: Path | str) -> dict[str, Any]:
    """Peek at a strategy file without loading all infosets into matrix."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if "hands" in data and "strategy" not in data:
        return {
            "path": str(p),
            "kind": "chart",
            "status": "ok",
            "root_id": data.get("node_id"),
            "num_infosets": len(data.get("hands") or []),
            "street": 0,
            "board_str": "",
            "iterations_run": None,
            "exploitability_bb": None,
            "description": data.get("description"),
        }
    root = data.get("root") or {}
    strat = data.get("strategy") or {}
    infos = strat.get("infosets") or []
    return {
        "path": str(p),
        "kind": "solve_report",
        "status": data.get("status"),
        "root_id": root.get("root_id"),
        "num_infosets": len(infos),
        "street": root.get("street"),
        "board_str": board_to_str(root.get("board") or []),
        "iterations_run": data.get("iterations_run"),
        "exploitability_bb": data.get("exploitability_bb"),
        "notes": (data.get("notes") or [])[:5],
    }
