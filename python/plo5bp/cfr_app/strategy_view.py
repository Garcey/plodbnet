"""Transform raw CFR SolveReport / chart JSON into UI-friendly views.

Handles:
- Full ``SolveReport`` dumps (``status/root/strategy.infosets``)
- Push/fold chart nodes (``hands[]`` with class labels)
- Preflop 169-class → 13×13 matrix aggregation
- Combo (0..1325) hand labels for postflop
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

from plo5bp.gto.preflop_class import (
    NUM_PREFLOP_CLASSES,
    cards_to_combo,
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


# (review 2026-09-20 E3) Dump schema v2 gives the root decision `path: []`, which
# joins to "" — while line_nav calls the root "open" and legacy ids say "root".
# Three spellings of one node meant: first load showed some other node, and the
# root request (`path=open`) matched zero rows. One spelling everywhere: "open".
_ROOT_PATHS = ("", "root", "open")


def normalize_path(path: Any) -> str:
    p = str(path if path is not None else "").strip()
    return "open" if p in _ROOT_PATHS else p


def runout_key(board: Sequence[int] | None, root_board_len: int | None) -> str:
    """Cards dealt AFTER the root board, as a stable key ("" on the root street).

    (review 2026-09-20 E6) On a flop/turn root the same (seat, action line)
    exists once per runout; without this in the node key the viewer averaged
    strategies across different river cards.
    """
    if not board or root_board_len is None:
        return ""
    return ",".join(str(int(c)) for c in list(board)[int(root_board_len):])


def runout_label(key: str) -> str:
    if not key:
        return ""
    return " ".join(card_to_pretty(int(c)) for c in key.split(","))


_CLASS_COMBOS = {"pair": 6.0, "suited": 4.0, "offsuit": 12.0}


def prior_weight(row: dict[str, Any]) -> float:
    """Weight of a row with no solver mass: combos in the class (6/4/12), else 1."""
    if row.get("private_kind") == "class" and row.get("private") is not None:
        try:
            hi, lo, suited = preflop_class_from_id(int(row["private"]))
        except (ValueError, TypeError):
            return 1.0
        return _CLASS_COMBOS["pair" if hi == lo else "suited" if suited else "offsuit"]
    return 1.0


def row_weights(rows: Sequence[dict[str, Any]]) -> list[float]:
    """Per-row weights for every aggregate shown in the UI.

    (review 2026-09-20 E7/E9) A node's action mix is a property of the RANGE
    that reaches it, not of the list of hands: the UI showed 52.7% where the
    reach-weighted truth was 36.6%, and preflop treated AKo (12 combos) like AA
    (6). ``visit_mass`` (the solver's accumulated own-reach at the infoset) is
    that weight; mass 0 means "never reached" — those rows still hold the 1/n
    default strategy and must not dilute the mix. With no mass anywhere (chart
    files, legacy dumps) fall back to combo counts.
    """
    mass = [float(r.get("visit_mass") or 0.0) for r in rows]
    if any(m > 0.0 for m in mass):
        return [m if m > 0.0 and math.isfinite(m) else 0.0 for m in mass]
    return [prior_weight(r) for r in rows]


def weighted_strategy(rows: Sequence[dict[str, Any]]) -> tuple[list[str], list[float]]:
    """Weighted mean action mix over ``rows``, in the SOLVER's action order."""
    order: list[str] = []
    sums: dict[str, float] = {}
    total = 0.0
    for r, w in zip(rows, row_weights(rows)):
        acts = r.get("actions") or [s.get("action") for s in (r.get("strategy") or [])]
        probs = r.get("probs") or [s.get("prob") for s in (r.get("strategy") or [])]
        for a in acts:
            if a not in sums:
                sums[a] = 0.0
                order.append(a)
        if w <= 0.0:
            continue
        total += w
        for a, p in zip(acts, probs):
            p = float(p or 0.0)
            if math.isfinite(p):
                sums[a] += w * p
    if total <= 0.0:
        return order, [0.0] * len(order)
    return order, [sums[a] / total for a in order]


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


def _finite(x: Any, default: float = 0.0) -> float:
    """float(x), with NaN/inf/None/garbage → ``default``.

    (review 2026-09-20) json.loads accepts bare ``NaN``; one NaN prob in an
    uploaded report became ``pct: NaN`` and Starlette refused to serialize the
    response — a 500 on every view of that file.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


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


def infoset_row(raw: dict[str, Any], *, root_board_len: int | None = None) -> dict[str, Any]:
    """One UI row from a raw infoset dict.

    ``root_board_len`` = number of board cards at the solve root; cards beyond
    it on a row's ``board`` are that row's runout (E6).
    """
    meta = parse_infoset_id(str(raw.get("infoset_id", "")))
    # Dump schema v2 is authoritative (combo 0..168 is not a 169-class).
    if raw.get("private_kind"):
        meta["private_kind"] = str(raw["private_kind"])
    if raw.get("private_id") is not None:
        meta["private"] = int(raw["private_id"])
    if raw.get("actor") is not None:
        meta["seat"] = int(raw["actor"])
    if isinstance(raw.get("path"), list):
        # (review 2026-09-20 E3) the root is `[]` → "open", never "".
        meta["path"] = normalize_path(",".join(str(x) for x in raw["path"]))
    else:
        meta["path"] = normalize_path(meta.get("path"))

    board = [int(c) for c in (raw.get("board") or []) if isinstance(c, (int, float))]
    kind = meta.get("private_kind")
    meta["combo"] = None
    if kind == "combo":
        # (review 2026-09-20 E5) With isomorphism on, `private_id` is the ISO id —
        # a suit relabelling, not a hand. Labelling it as a combo showed 3549 of
        # 4809 river rows as the wrong hand, 756 of them holding a board card.
        # `raw_combo` is the real hand; every label, the class matrix, the hand
        # filter and the compare tool key off this one field.
        rc = raw.get("raw_combo")
        combo = int(rc) if rc is not None else meta.get("private")
        if isinstance(combo, int) and 0 <= combo < 1326:
            if set(combo_to_cards(combo)) & set(board):
                # An iso id with no raw_combo (old dump): we cannot know the hand.
                # Never display an impossible one.
                meta["hand_label"] = f"iso#{meta.get('private')}"
            else:
                meta["combo"] = combo
                meta["hand_label"] = combo_label(combo)
        else:
            meta["hand_label"] = f"c{meta.get('private')}"
    elif kind and kind != "class":
        # e.g. "ochs_bucket": a bucket id is neither a combo nor a 169-class.
        meta["hand_label"] = f"{kind.replace('_', ' ')} {meta.get('private')}"

    vm = raw.get("visit_mass")
    meta["visit_mass"] = float(vm) if isinstance(vm, (int, float)) and math.isfinite(vm) else None
    meta["street"] = int(raw["street"]) if isinstance(raw.get("street"), int) else None
    meta["board"] = board
    meta["runout"] = runout_key(board, root_board_len)
    actions = list(raw.get("actions") or [])
    probs = [_finite(x) for x in (raw.get("probs") or [])]
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


def load_report(
    path: Path | str | dict[str, Any],
    *,
    source: str | None = None,
) -> dict[str, Any]:
    """Load a SolveReport JSON (or pass-through dict) into a normalized view.

    ``source`` names the file an already-parsed dict came from, so callers that
    parsed it once (review 2026-09-20 E4) keep Export + chart-pack lookup working.
    """
    if isinstance(path, dict):
        data = path
        source = str(source) if source else "<memory>"
    else:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        source = str(p)

    # Chart node format (pushfold hands)
    if "hands" in data and "strategy" not in data:
        return _view_from_chart(data, source=source)

    root = data.get("root") if isinstance(data.get("root"), dict) else {}
    strategy = data.get("strategy") if isinstance(data.get("strategy"), dict) else {}
    infosets_raw = strategy.get("infosets") or data.get("infosets") or []
    root_board_len = len(root.get("board") or [])
    rows = [
        infoset_row(x, root_board_len=root_board_len)
        for x in infosets_raw
        if isinstance(x, dict)
    ]

    street = int(root.get("street", -1)) if root else -1
    class_rows_n = sum(1 for r in rows[:500] if r.get("private_kind") == "class")
    combo_rows_n = sum(1 for r in rows[:500] if r.get("private_kind") == "combo")
    is_preflop_class = class_rows_n > 0 and class_rows_n >= combo_rows_n
    if street > 0:
        # Postflop: never treat combo ids 0..168 as preflop classes.
        is_preflop_class = class_rows_n > combo_rows_n and class_rows_n >= 50

    nodes = group_by_node(rows)
    from plo5bp.cfr_app.tree_model import (
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
    # (review 2026-09-20 E3) The first, unfiltered load shows the ROOT decision —
    # the node the UI then selects (line_nav.root_*). It used to show "the node
    # with the most rows", i.e. an arbitrary deep line.
    matrix = matrix_for_rows(
        node_rows(rows, seat=line_nav["root_seat"], path=line_nav["root_path"])["rows"] or rows,
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
        # The user's wording when the app recorded it (review 2026-09-20 E11); the
        # bare field holds the canonical id:weight expansion sent to the solver.
        "range_ip": root.get("range_ip_text") or root.get("range_ip") or "",
        "range_oop": root.get("range_oop_text") or root.get("range_oop") or "",
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
    # (review 2026-09-20) Tokenize FIRST, then shorten each token. The old chain
    # replaced "_" with " → " before it looked for CHECK_CALL / RAISE_500, so
    # those could never match: "CHECK_CALL,RAISE_500" rendered as
    # "CHECK → CALL → RAISE → 500".
    parts = []
    for tok in p.split(","):
        tok = tok.strip()
        if not tok:
            continue
        up = tok.upper()
        if up in ("FOLD", "CHECK_CALL", "ALLIN") or up.startswith("RAISE_"):
            parts.append(action_short(up))
        else:
            parts.append(tok.replace("_", " → "))  # legacy "AI_F"-style separators
    return " → ".join(parts) or "Open"


_MAX_RUNOUT_OPTIONS = 60  # picker size cap (a flop root can see ~2k turn+river runouts)


def _row_node_path(r: dict[str, Any]) -> str:
    """The node path a row belongs to: its action line, else its history hash."""
    p = normalize_path(r.get("path"))
    if p == "open" and r.get("history_hash") is not None and not r.get("path"):
        return str(r["history_hash"])  # legacy hash-only row with no path at all
    return p


def _path_matches(r: dict[str, Any], want: str) -> bool:
    if want == _row_node_path(r):
        return True
    rhist = r.get("history_hash")
    return rhist is not None and want in (str(rhist), f"h{rhist}")


def runout_options(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Distinct runouts among ``rows``, most visited first ([] on the root street)."""
    acc: dict[str, dict[str, Any]] = {}
    for r in rows:
        k = r.get("runout") or ""
        a = acc.setdefault(k, {"key": k, "label": runout_label(k), "num_hands": 0, "visit_mass": 0.0})
        a["num_hands"] += 1
        a["visit_mass"] += float(r.get("visit_mass") or 0.0)
    if set(acc) <= {""}:
        return []
    opts = sorted(acc.values(), key=lambda a: (-a["visit_mass"], -a["num_hands"], a["key"]))
    for o in opts:
        o["visit_mass"] = round(o["visit_mass"], 6)
    return opts


def node_rows(
    rows: Sequence[dict[str, Any]],
    *,
    seat: int | None = None,
    path: str | None = None,
    runout: str | None = None,
) -> dict[str, Any]:
    """Rows of ONE decision node: (seat, action line, board-so-far).

    (review 2026-09-20 E3/E6) The single definition of "this node" shared by the
    table filter, the matrix and the node list. ``open``/``root``/``""`` are the
    same root; a legacy history hash also matches. When the line exists on
    several runouts, exactly one is returned — the requested one, else the most
    visited — never a blend across different turn/river cards.
    """
    want = normalize_path(path)
    sel = [
        r
        for r in rows
        if (seat is None or int(r.get("seat", 0)) == int(seat)) and _path_matches(r, want)
    ]
    opts = runout_options(sel)
    chosen = ""
    if opts:
        chosen = runout if runout in {o["key"] for o in opts} else opts[0]["key"]
        sel = [r for r in sel if (r.get("runout") or "") == chosen]
    return {
        "rows": sel,
        "runout": chosen,
        "runout_label": runout_label(chosen),
        "runouts": opts[:_MAX_RUNOUT_OPTIONS],
        "num_runouts": len(opts),
    }


def group_by_node(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    from plo5bp.cfr_app.tree_model import aggregate_node, path_tokens

    buckets: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault((int(r.get("seat", 0)), _row_node_path(r)), []).append(r)
    nodes = []
    # Tree order: shallow lines first, then seat — the root leads the list.
    # (Was "largest node first", which put an arbitrary deep line on top.)
    def _depth(path: str) -> int:
        return len(path_tokens(path))  # "open" and legacy hash ids are depth 0

    items = sorted(buckets.items(), key=lambda x: (_depth(x[0][1]), x[0][0], x[0][1]))
    for (seat, path), bucket in items:
        # Aggregate over ONE runout (the most visited), not across runouts (E6).
        picked = node_rows(bucket, seat=seat, path=path)
        rs = picked["rows"]
        agg = aggregate_node(rs)
        label = humanize_path(path)
        dealt = picked["runout_label"]
        nodes.append(
            {
                "node_key": f"s{seat}_{path}",
                "seat": seat,
                "path": path,
                "label": f"P{seat} · {label}" + (f" · {dealt}" if dealt else ""),
                "path_pretty": label,
                "num_hands": len(rs),
                "mean_primary": (
                    sum(float(x.get("primary_prob") or 0) for x in rs) / len(rs) if rs else 0.0
                ),
                "aggregate": agg,
                "runout": picked["runout"],
                "runout_label": dealt,
                "runouts": picked["runouts"],
                "num_runouts": picked["num_runouts"],
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
    # (review 2026-09-20 E5) `combo` is the REAL hand (raw_combo); `private` may
    # be an isomorphism id, which put hands in the wrong class cell.
    combo_rows = [
        r for r in rows if r.get("private_kind") == "combo" and r.get("combo") is not None
    ]
    empty = {
        "cells": [],
        "rank_labels": list(reversed(_RANKS)),
        "empty": True,
        "aggregated_from_combos": True,
    }
    if not combo_rows:
        return empty

    best_key, selected = _single_node(combo_rows)

    by_class: dict[int, list[dict[str, Any]]] = {}
    for r in selected:
        c0, c1 = combo_to_cards(int(r["combo"]))
        by_class.setdefault(preflop_class_from_cards(c0, c1), []).append(r)

    cells: list[list[dict[str, Any] | None]] = [[None] * 13 for _ in range(13)]
    placed = 0
    for cid, class_rows in by_class.items():
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
        # (review 2026-09-20 E7/E9) ONE weighted per-action average drives the
        # cell colour, the tooltip AND the detail panel. The panel used to show
        # `last_strat` — whichever combo happened to be iterated last — so a
        # click could disagree with the cell it was clicked on by 10+ points.
        actions, probs = weighted_strategy(class_rows)
        strat = _normalize_probs(actions, probs)
        mix = _mix_buckets(strat)
        cells[ri][ci] = {
            "class_id": cid,
            "label": preflop_class_label(cid),
            "hi": hi,
            "lo": lo,
            "suited": suited if hi != lo else None,
            "strategy": strat,
            "fold": round(mix["fold"], 4),
            "call": round(mix["call"], 4),
            "agg": round(mix["agg"], 4),
            "allin": round(mix["allin"], 4),
            "primary": None,
            "primary_prob": max(mix["fold"], mix["call"], mix["agg"]),
            "n_combos": len(class_rows),
            "weighted": True,
        }
        placed += 1

    return {
        "empty": placed == 0,
        "aggregated_from_combos": True,
        "rank_labels": list(reversed(_RANKS)),
        "seat": best_key[0],
        "path": best_key[1],
        "runout": best_key[2],
        "runout_label": runout_label(best_key[2]),
        "num_classes": len(by_class),
        "num_filled": placed,
        "cells": cells,
    }


def _mix_buckets(strategy: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Fold / call / aggressive / all-in mass of one strategy (all-in ⊂ aggressive)."""
    out = {"fold": 0.0, "call": 0.0, "agg": 0.0, "allin": 0.0}
    for s in strategy:
        a = str(s.get("action", "")).upper()
        p = _finite(s.get("prob"))
        if a == "FOLD":
            out["fold"] += p
        elif a in ("CHECK_CALL", "CHECK", "CALL"):
            out["call"] += p
        else:
            out["agg"] += p
            if a == "ALLIN":
                out["allin"] += p
    return out


def _single_node(
    rows: Sequence[dict[str, Any]],
) -> tuple[tuple[int, str, str], list[dict[str, Any]]]:
    """Reduce rows to ONE (seat, path, runout) node for a 13×13 grid.

    Callers normally pass a single node already (``node_rows``). For a mixed
    list prefer the root decision, then the most-visited node — a grid must
    never silently blend nodes or runouts (review 2026-09-20 E3/E6).
    """
    by_node: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for r in rows:
        key = (int(r.get("seat", 0)), _row_node_path(r), r.get("runout") or "")
        by_node.setdefault(key, []).append(r)

    def _rank(k: tuple[int, str, str]) -> tuple[Any, ...]:
        rs = by_node[k]
        mass = sum(float(r.get("visit_mass") or 0.0) for r in rs)
        return (k[1] != "open", -mass, -len(rs), k)

    best = min(by_node, key=_rank)
    return best, by_node[best]


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

    # One node only (normally already filtered); root preferred over "largest".
    best_key, selected = _single_node(class_rows)

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
    runout: str | None = None,
) -> dict[str, Any]:
    """Filter + page rows. ``path=None`` = every node; any other value selects
    ONE decision node via :func:`node_rows` (``open``/``root``/``""`` = the root)."""
    info: dict[str, Any] = {"runout": "", "runout_label": "", "runouts": [], "num_runouts": 0}
    if path is None:
        out = [r for r in rows if seat is None or int(r.get("seat", 0)) == int(seat)]
    else:
        picked = node_rows(rows, seat=seat, path=path, runout=runout)
        out = picked.pop("rows")
        info = picked
    match = hand_matcher(hand_query)
    if match is not None:
        out = [r for r in out if match(r)]
    total = len(out)
    offset = max(0, int(offset))
    page = out[offset : offset + max(1, limit)]
    return {"total": total, "offset": offset, "limit": limit, "rows": page, **info}


_RANK_IDX = {ch: i for i, ch in enumerate(_RANKS)}
_SUIT_IDX = {ch: i for i, ch in enumerate(_SUITS)}
_RE_Q_COMBO = re.compile(r"^([2-9TJQKA])([cdhs])([2-9TJQKA])([cdhs])$")
_RE_Q_CLASS = re.compile(r"^([2-9TJQKA])([2-9TJQKA])([so]?)$")


def hand_matcher(query: str):
    """Row predicate for the "Hand / id" box, or None for an empty query.

    (review 2026-09-20) The old filter was a substring test on the label, so
    ``AsKs`` found nothing when the row was labelled ``KsAs``, and class tokens
    (``AKs``, ``AA``) never matched combo rows at all. Understood here:

    - ``AsKs`` / ``KsAs`` — that exact combo, either card order;
    - ``AKs`` / ``AKo`` / ``AA`` / ``AK`` — the class (``AK`` = suited + offsuit),
      matching class rows by label and combo rows by the class of their hand;
    - anything else — case-insensitive substring of the label or infoset id.
    """
    raw = (query or "").strip()
    if not raw:
        return None
    # Canonical spelling: ranks upper-case, suit / s,o suffix lower-case.
    canon = "".join(ch.upper() if ch.upper() in _RANK_IDX else ch.lower() for ch in raw)

    m = _RE_Q_COMBO.match(canon)
    if m:
        c_a = _RANK_IDX[m.group(1)] * 4 + _SUIT_IDX[m.group(2)]
        c_b = _RANK_IDX[m.group(3)] * 4 + _SUIT_IDX[m.group(4)]
        if c_a == c_b:
            return lambda r: False
        want = cards_to_combo(c_a, c_b)
        cls = preflop_class_label(preflop_class_from_cards(c_a, c_b))

        def _combo(r: dict[str, Any]) -> bool:
            if r.get("private_kind") == "class":
                return r.get("hand_label") == cls  # a class row contains this combo
            return r.get("combo") == want

        return _combo

    m = _RE_Q_CLASS.match(canon)
    if m:
        a, b, suf = m.group(1), m.group(2), m.group(3)
        hi, lo = (a, b) if _RANK_IDX[a] >= _RANK_IDX[b] else (b, a)
        if hi == lo:
            labels = {hi + lo} if not suf else set()  # "AAs" is not a hand
        else:
            labels = {hi + lo + s for s in ((suf,) if suf else ("s", "o"))}

        def _class(r: dict[str, Any]) -> bool:
            if r.get("private_kind") == "class":
                return r.get("hand_label") in labels
            combo = r.get("combo")
            if combo is None:
                return False
            c0, c1 = combo_to_cards(int(combo))
            return preflop_class_label(preflop_class_from_cards(c0, c1)) in labels

        return _class

    q = raw.lower()
    return lambda r: q in str(r.get("hand_label") or "").lower() or q in str(
        r.get("infoset_id") or ""
    ).lower()


def list_strategy_library(
    roots: Sequence[Path | str] | None = None,
    *,
    max_files: int = 500,
) -> list[dict[str, Any]]:
    """Scan known CFR data dirs for loadable JSON strategy files."""
    if roots is None:
        # (review 2026-09-20 J4) env-overridable defaults — see cfr_app/paths.py
        from plo5bp.cfr_app.paths import library_roots

        roots = library_roots()
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for root in roots:
        r = Path(root)
        if not r.exists():
            continue
        # (review 2026-09-20 E12) stat() once per file, tolerating files that
        # vanish mid-scan: a live solve rewrites/deletes its progress + tmp files
        # under app_jobs, and an unguarded stat() in the sort key 500'd /api/library.
        stamped: list[tuple[float, int, Path]] = []
        for p in r.rglob("*.json") if r.is_dir() else [r]:
            try:
                st = p.stat()
            except OSError:
                continue
            stamped.append((st.st_mtime, st.st_size, p))
        stamped.sort(key=lambda x: x[0], reverse=True)
        for mtime, size, p in stamped:
            if not p.is_file():
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            # skip grid/status/index noise optionally — still include INDEX as meta
            name = p.name.lower()
            if name in ("status.json", "overnight_grid.json", "manifest.json", "plan.json", "certificate.json"):
                continue
            # (review 2026-09-20 E12) live-solve snapshots are not strategies.
            if name.endswith(".progress.json"):
                continue
            seen.add(key)
            items.append(
                {
                    "path": str(p),
                    "name": p.name,
                    "rel": _rel_to_repo(p),
                    "size": size,
                    "size_kb": round(size / 1024.0, 1),
                    "mtime": mtime,
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


def summarize_report_light(
    path: Path | str, *, data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Peek at a strategy file without loading all infosets into matrix.

    Pass ``data`` when the caller already parsed the file (avoids a re-read).
    """
    p = Path(path)
    if data is None:
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
