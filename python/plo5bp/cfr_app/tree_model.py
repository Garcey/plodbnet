"""Abstract game-tree construction + strategy-line hierarchy for the desktop app.

Two layers:

1. **Builder tree** — full abstract decision tree implied by a ``RootSpec``
   (raise sizes, all-in atom, seats). Shown before/while solving so the user
   sees the tree they are about to solve (Monker/Pio-style size menu tree).

2. **Solution tree** — hierarchical nodes recovered from strategy infosets
   (path tokens for multiway preflop; history-hash buckets for postflop).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from plo5bp.gto.cfr_api import RootSpec

STREET_NAMES = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River"}


def _action_menu(root: RootSpec, *, facing_bet: bool) -> list[str]:
    acts: list[str] = []
    if facing_bet:
        acts.append("FOLD")
        acts.append("CHECK_CALL")
    else:
        acts.append("CHECK_CALL")  # check
    for pm in root.raise_sizes_pm:
        acts.append(f"RAISE_{int(pm)}")
    if root.allin_atom:
        acts.append("ALLIN")
    # pure push/fold: only FOLD + ALLIN
    if not root.raise_sizes_pm and root.allin_atom:
        return ["FOLD", "ALLIN"] if facing_bet else ["CHECK_CALL", "ALLIN"]
    return acts


@dataclass
class TreeNode:
    id: str
    label: str
    seat: int
    depth: int
    facing_bet: bool
    pot_bb: float
    stack_bb: float
    to_call_bb: float
    actions: list[str] = field(default_factory=list)
    children: list["TreeNode"] = field(default_factory=list)
    terminal: bool = False
    terminal_kind: str = ""  # fold | showdown | allin_runout

    def as_dict(self, *, max_depth: int | None = None) -> dict[str, Any]:
        if max_depth is not None and self.depth > max_depth:
            return {
                "id": self.id,
                "label": self.label + " …",
                "seat": self.seat,
                "depth": self.depth,
                "truncated": True,
                "num_children": len(self.children),
            }
        return {
            "id": self.id,
            "label": self.label,
            "seat": self.seat,
            "depth": self.depth,
            "facing_bet": self.facing_bet,
            "pot_bb": round(self.pot_bb, 3),
            "stack_bb": round(self.stack_bb, 3),
            "to_call_bb": round(self.to_call_bb, 3),
            "actions": list(self.actions),
            "terminal": self.terminal,
            "terminal_kind": self.terminal_kind,
            "children": [
                c.as_dict(max_depth=max_depth) for c in self.children
            ],
            "num_children": len(self.children),
        }


def build_abstract_tree(
    root: RootSpec | dict[str, Any],
    *,
    max_nodes: int = 400,
    max_depth: int = 8,
) -> dict[str, Any]:
    """Enumerate the abstract bet-size tree for a root (HU simplified).

    Multiway (>2) uses a push/fold-style or sequential simplified model:
    each active seat acts once per betting round with the same menu.
    """
    if isinstance(root, dict):
        from plo5bp.cfr_app.session import _root_from_dict

        r = _root_from_dict(root)
    else:
        r = root

    n = max(2, int(r.num_seats))
    pot = float(r.pot_bb)
    eff = float(r.effective_stack_bb)
    counter = {"n": 0}
    nodes_flat: list[dict[str, Any]] = []

    def new_id() -> str:
        counter["n"] += 1
        return f"n{counter['n']}"

    def build(
        *,
        seat: int,
        depth: int,
        pot_bb: float,
        stacks: list[float],
        invested: list[float],
        facing: float,
        active: list[bool],
        last_full_raise: float,
        path_label: str,
    ) -> TreeNode:
        nid = new_id()
        if counter["n"] > max_nodes or depth > max_depth:
            return TreeNode(
                id=nid,
                label="(truncated)",
                seat=seat,
                depth=depth,
                facing_bet=facing > 1e-9,
                pot_bb=pot_bb,
                stack_bb=stacks[seat],
                to_call_bb=max(0.0, facing - invested[seat]),
                terminal=True,
                terminal_kind="truncated",
            )

        # skip folded seats
        if not active[seat]:
            nxt = (seat + 1) % n
            return build(
                seat=nxt,
                depth=depth,
                pot_bb=pot_bb,
                stacks=stacks,
                invested=invested,
                facing=facing,
                active=active,
                last_full_raise=last_full_raise,
                path_label=path_label,
            )

        to_call = max(0.0, facing - invested[seat])
        facing_bet = to_call > 1e-9
        # can act?
        if stacks[seat] <= 1e-12:
            # all-in already — pass
            nxt = (seat + 1) % n
            # if everyone all-in or matched → terminal
            return build(
                seat=nxt,
                depth=depth + 1,
                pot_bb=pot_bb,
                stacks=stacks,
                invested=invested,
                facing=facing,
                active=active,
                last_full_raise=last_full_raise,
                path_label=path_label,
            )

        menu = _action_menu(r, facing_bet=facing_bet)
        # filter illegal: if to_call >= stack, only fold / allin call
        if to_call >= stacks[seat] - 1e-12 and facing_bet:
            menu = ["FOLD", "ALLIN"]

        node = TreeNode(
            id=nid,
            label=path_label or "root",
            seat=seat,
            depth=depth,
            facing_bet=facing_bet,
            pot_bb=pot_bb,
            stack_bb=stacks[seat],
            to_call_bb=to_call,
            actions=menu,
        )

        for act in menu:
            child = _apply_action(
                act=act,
                seat=seat,
                depth=depth,
                pot_bb=pot_bb,
                stacks=list(stacks),
                invested=list(invested),
                facing=facing,
                active=list(active),
                last_full_raise=last_full_raise,
                path_label=path_label,
                n=n,
                build_fn=build,
                new_id=new_id,
                root=r,
            )
            node.children.append(child)

        nodes_flat.append(
            {
                "id": node.id,
                "label": node.label,
                "seat": node.seat,
                "depth": node.depth,
                "actions": node.actions,
                "pot_bb": node.pot_bb,
                "to_call_bb": node.to_call_bb,
                "terminal": node.terminal,
            }
        )
        return node

    def _apply_action(
        *,
        act: str,
        seat: int,
        depth: int,
        pot_bb: float,
        stacks: list[float],
        invested: list[float],
        facing: float,
        active: list[bool],
        last_full_raise: float,
        path_label: str,
        n: int,
        build_fn,
        new_id,
        root: RootSpec,
    ) -> TreeNode:
        to_call = max(0.0, facing - invested[seat])
        short = act if len(act) < 12 else act[:10]
        child_label = f"{path_label}/{short}" if path_label else short

        if act == "FOLD":
            active[seat] = False
            alive = [i for i, a in enumerate(active) if a]
            if len(alive) <= 1:
                return TreeNode(
                    id=new_id(),
                    label=child_label + " → fold_win",
                    seat=alive[0] if alive else seat,
                    depth=depth + 1,
                    facing_bet=False,
                    pot_bb=pot_bb,
                    stack_bb=stacks[alive[0]] if alive else 0,
                    to_call_bb=0,
                    terminal=True,
                    terminal_kind="fold",
                )
            nxt = (seat + 1) % n
            return build_fn(
                seat=nxt,
                depth=depth + 1,
                pot_bb=pot_bb,
                stacks=stacks,
                invested=invested,
                facing=facing,
                active=active,
                last_full_raise=last_full_raise,
                path_label=child_label,
            )

        if act == "CHECK_CALL":
            put = min(stacks[seat], to_call)
            stacks[seat] -= put
            invested[seat] += put
            pot_bb += put
            # if check (to_call=0), advance; if call, check if round closes
            nxt = (seat + 1) % n
            if to_call <= 1e-12:
                # check — if back to aggressor / all checked close street
                if _round_closed_check(seat, n, active, invested, facing):
                    return TreeNode(
                        id=new_id(),
                        label=child_label + " → showdown/next",
                        seat=seat,
                        depth=depth + 1,
                        facing_bet=False,
                        pot_bb=pot_bb,
                        stack_bb=stacks[seat],
                        to_call_bb=0,
                        terminal=True,
                        terminal_kind="showdown",
                    )
            else:
                if _bets_matched(active, invested, facing):
                    return TreeNode(
                        id=new_id(),
                        label=child_label + " → showdown/next",
                        seat=seat,
                        depth=depth + 1,
                        facing_bet=False,
                        pot_bb=pot_bb,
                        stack_bb=stacks[seat],
                        to_call_bb=0,
                        terminal=True,
                        terminal_kind="showdown",
                    )
            return build_fn(
                seat=nxt,
                depth=depth + 1,
                pot_bb=pot_bb,
                stacks=stacks,
                invested=invested,
                facing=facing,
                active=active,
                last_full_raise=last_full_raise,
                path_label=child_label,
            )

        # raise or allin
        if act == "ALLIN":
            put = stacks[seat]
            stacks[seat] = 0.0
            invested[seat] += put
            pot_bb += put
            new_face = invested[seat]
            raise_size = max(0.0, new_face - facing)
            last_full_raise = max(last_full_raise, raise_size)
            facing = max(facing, new_face)
            if _all_others_allin_or_matched(seat, active, stacks, invested, facing):
                return TreeNode(
                    id=new_id(),
                    label=child_label + " → runout",
                    seat=seat,
                    depth=depth + 1,
                    facing_bet=False,
                    pot_bb=pot_bb,
                    stack_bb=0,
                    to_call_bb=0,
                    terminal=True,
                    terminal_kind="allin_runout",
                )
            nxt = (seat + 1) % n
            return build_fn(
                seat=nxt,
                depth=depth + 1,
                pot_bb=pot_bb,
                stacks=stacks,
                invested=invested,
                facing=facing,
                active=active,
                last_full_raise=last_full_raise,
                path_label=child_label,
            )

        # RAISE_pm
        pm = 1000
        if act.startswith("RAISE_"):
            try:
                pm = int(act.split("_", 1)[1])
            except ValueError:
                pm = 1000
        # pot-fraction raise sizing (approx): raise to pot*pm/1000 more after call
        call_amt = min(stacks[seat], to_call)
        pot_after_call = pot_bb + call_amt
        raise_extra = pot_after_call * (pm / 1000.0)
        want = call_amt + raise_extra
        put = min(stacks[seat], want)
        stacks[seat] -= put
        invested[seat] += put
        pot_bb += put
        new_face = invested[seat]
        raise_size = max(0.0, new_face - facing)
        last_full_raise = max(last_full_raise, raise_size) if raise_size > 0 else last_full_raise
        facing = max(facing, new_face)
        nxt = (seat + 1) % n
        return build_fn(
            seat=nxt,
            depth=depth + 1,
            pot_bb=pot_bb,
            stacks=stacks,
            invested=invested,
            facing=facing,
            active=active,
            last_full_raise=last_full_raise,
            path_label=child_label,
        )

    stacks0 = (
        list(r.stacks_bb)
        if r.stacks_bb and len(r.stacks_bb) == n
        else [eff] * n
    )
    invested0 = [0.0] * n
    # preflop blinds simplification
    if r.street == 0 and n >= 2:
        bb = 1.0
        sb = 0.5
        ante = (r.ante_chips / float(r.bb_chips)) if r.bb_chips else 0.5
        pot0 = n * ante + sb + bb
        # seat 0 = first to act (UTG / SB in HU)
        if n == 2:
            # HU: seat0 SB, seat1 BB; SB acts first
            invested0[0] = sb + ante
            invested0[1] = bb + ante
            stacks0[0] = max(0.0, stacks0[0] - sb - ante)
            stacks0[1] = max(0.0, stacks0[1] - bb - ante)
            facing0 = bb + ante
            first = 0
        else:
            for i in range(n):
                invested0[i] = ante
                stacks0[i] = max(0.0, stacks0[i] - ante)
            # SB = n-2, BB = n-1
            sb_i, bb_i = n - 2, n - 1
            invested0[sb_i] += sb
            invested0[bb_i] += bb
            stacks0[sb_i] = max(0.0, stacks0[sb_i] - sb)
            stacks0[bb_i] = max(0.0, stacks0[bb_i] - bb)
            facing0 = bb + ante
            first = 0  # UTG
            pot0 = sum(invested0)
        pot = pot0
        facing = facing0
    else:
        facing = 0.0
        first = 0  # OOP acts first postflop
        pot = float(r.pot_bb)

    active = [True] * n
    tree = build(
        seat=first,
        depth=0,
        pot_bb=pot,
        stacks=stacks0,
        invested=invested0,
        facing=facing,
        active=active,
        last_full_raise=1.0,
        path_label="",
    )

    return {
        "root_id": r.root_id,
        "street": r.street,
        "street_name": STREET_NAMES.get(r.street, str(r.street)),
        "num_seats": n,
        "pot_bb": float(r.pot_bb),
        "effective_stack_bb": eff,
        "raise_sizes_pm": list(r.raise_sizes_pm),
        "allin_atom": r.allin_atom,
        "board": list(r.board),
        "num_nodes_built": counter["n"],
        "max_nodes": max_nodes,
        "tree": tree.as_dict(max_depth=max_depth),
        "flat": nodes_flat[:max_nodes],
    }


def _round_closed_check(
    seat: int, n: int, active: list[bool], invested: list[float], facing: float
) -> bool:
    """After a check: if next active seats would complete a full orbit of checks."""
    # simplified: if facing==0 and we've gone around — treat single check from last as open
    return facing <= 1e-12 and seat == (n - 1) % n


def _bets_matched(active: list[bool], invested: list[float], facing: float) -> bool:
    for i, a in enumerate(active):
        if a and abs(invested[i] - facing) > 1e-9:
            # still can put more if stack remains — handled elsewhere
            if invested[i] + 1e-9 < facing:
                return False
    return True


def _all_others_allin_or_matched(
    seat: int,
    active: list[bool],
    stacks: list[float],
    invested: list[float],
    facing: float,
) -> bool:
    for i, a in enumerate(active):
        if i == seat or not a:
            continue
        if stacks[i] > 1e-12 and invested[i] + 1e-9 < facing:
            return False
    return True


# ---------------------------------------------------------------------------
# Solution hierarchy from infoset rows
# ---------------------------------------------------------------------------


def build_solution_tree(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Group strategy rows into a hierarchical line browser.

    Prefers explicit ``path`` tokens (``open``, ``F``, ``AI``, ``F,AI``…).
    Falls back to ``history_hash`` buckets for postflop hash ids.
    """
    # node_key → list of rows
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        path = str(r.get("path") or "")
        hist = r.get("history_hash")
        seat = int(r.get("seat", 0))
        if path and path not in ("", "root") and not path.isdigit():
            key = f"p{seat}|{path}"
            kind = "path"
        elif hist is not None:
            key = f"p{seat}|h{hist}"
            kind = "hash"
        else:
            key = f"p{seat}|root"
            kind = "root"
        buckets.setdefault(key, []).append(r)

    # Build path forest for tokenized paths
    forest: dict[str, Any] = {"name": "root", "children": {}, "nodes": []}

    def ensure_path(tokens: list[str]) -> dict[str, Any]:
        cur = forest
        acc = []
        for t in tokens:
            acc.append(t)
            ch = cur["children"]
            if t not in ch:
                ch[t] = {
                    "name": t,
                    "line": ",".join(acc),
                    "children": {},
                    "nodes": [],
                }
            cur = ch[t]
        return cur

    solution_nodes: list[dict[str, Any]] = []
    for key, rs in sorted(buckets.items(), key=lambda x: (-len(x[1]), x[0])):
        seat = int(rs[0].get("seat", 0))
        path = str(rs[0].get("path") or "")
        hist = rs[0].get("history_hash")
        agg = aggregate_node(rs)
        node = {
            "key": key,
            "seat": seat,
            "path": path or (f"h{hist}" if hist is not None else "root"),
            "history_hash": hist,
            "num_hands": len(rs),
            "aggregate": agg,
            "kind": "path" if path and path not in ("", "root") else ("hash" if hist is not None else "root"),
        }
        solution_nodes.append(node)

        if path and path not in ("", "root", "open"):
            tokens = [t for t in path.replace("-", ",").split(",") if t]
            if tokens:
                leaf = ensure_path(tokens)
                leaf["nodes"].append(node)
        elif path == "open":
            leaf = ensure_path(["open"])
            leaf["nodes"].append(node)
        else:
            forest["nodes"].append(node)

    def freeze(node: dict[str, Any]) -> dict[str, Any]:
        children = [freeze(c) for c in node["children"].values()]
        # sort: open first, then F, AI, etc.
        children.sort(key=lambda c: c.get("name") or "")
        return {
            "name": node["name"],
            "line": node.get("line", node["name"]),
            "nodes": node.get("nodes") or [],
            "children": children,
            "num_descendant_hands": _count_hands(node),
        }

    def _count_hands(node: dict[str, Any]) -> int:
        n = sum(x.get("num_hands", 0) for x in (node.get("nodes") or []))
        for c in (node.get("children") or {}).values():
            n += _count_hands(c)
        return n

    return {
        "nodes": solution_nodes,
        "forest": freeze(forest),
        "num_decision_points": len(solution_nodes),
    }


def aggregate_node(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Frequency-weighted (uniform hand weight) action mix at a node."""
    if not rows:
        return {
            "actions": [],
            "mean_mix": {},
            "fold": 0.0,
            "call": 0.0,
            "raise": 0.0,
            "allin": 0.0,
            "entropy": 0.0,
            "num_hands": 0,
        }
    totals: dict[str, float] = {}
    fold = call = raise_p = allin = 0.0
    ent_sum = 0.0
    for r in rows:
        strat = r.get("strategy") or []
        if not strat and r.get("actions"):
            strat = [
                {"action": a, "prob": float(p)}
                for a, p in zip(r.get("actions") or [], r.get("probs") or [])
            ]
        local_ent = 0.0
        for s in strat:
            a = str(s.get("action", "")).upper()
            p = float(s.get("prob") or 0.0)
            totals[a] = totals.get(a, 0.0) + p
            if a == "FOLD":
                fold += p
            elif a == "CHECK_CALL":
                call += p
            elif a == "ALLIN":
                allin += p
            elif a.startswith("RAISE"):
                raise_p += p
            if p > 1e-12:
                import math

                local_ent -= p * math.log(p + 1e-15)
        ent_sum += local_ent
    n = float(len(rows))
    mean_mix = {a: v / n for a, v in sorted(totals.items(), key=lambda x: -x[1])}
    return {
        "actions": list(mean_mix.keys()),
        "mean_mix": {k: round(v, 4) for k, v in mean_mix.items()},
        "fold": round(fold / n, 4),
        "call": round(call / n, 4),
        "raise": round(raise_p / n, 4),
        "allin": round(allin / n, 4),
        "entropy": round(ent_sum / n, 4),
        "num_hands": len(rows),
        # range weight proxy: fraction of hands with primary aggressive action ≥ 50%
        "agg_ge_50pct": round(
            sum(1 for r in rows if float(r.get("primary_prob") or 0) >= 0.5
                and str(r.get("primary_action") or "").upper() not in ("FOLD", "CHECK_CALL"))
            / n,
            4,
        ),
        "fold_ge_50pct": round(
            sum(
                1
                for r in rows
                if str(r.get("primary_action") or "").upper() == "FOLD"
                and float(r.get("primary_prob") or 0) >= 0.5
            )
            / n,
            4,
        ),
    }


def action_to_token(action: str) -> str:
    """Map solver action labels onto compact path tokens (F / AI / XC / R500)."""
    a = str(action).upper()
    if a == "FOLD":
        return "F"
    if a == "ALLIN":
        return "AI"
    if a in ("CHECK_CALL", "CHECK", "CALL"):
        return "XC"
    if a.startswith("RAISE_"):
        try:
            return "R" + str(int(a.split("_", 1)[1]))
        except ValueError:
            return a
    return a


def path_tokens(path: str | None) -> list[str]:
    p = str(path or "").strip()
    if p in ("", "root", "open"):
        return []
    # History hashes are not action sequences
    if p.isdigit() and len(p) >= 8:
        return []
    if p.startswith("h") and p[1:].isdigit() and len(p) >= 8:
        return []
    return [t for t in p.replace("-", ",").split(",") if t]


def join_path(tokens: Sequence[str]) -> str:
    ts = [str(t) for t in tokens if t]
    return "open" if not ts else ",".join(ts)


def try_load_chart_pack(source: str | None) -> dict[str, Any] | None:
    """If ``source`` sits next to INDEX.json, return path → file map (14-chart packs)."""
    if not source or source == "<memory>":
        return None
    folder = Path(source).parent
    idx = folder / "INDEX.json"
    if not idx.is_file():
        return None
    try:
        data = json.loads(idx.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    by_path: dict[str, Any] = {}
    for node in data.get("nodes") or []:
        pth = str(node.get("path") or "open")
        fname = str(node.get("file") or "")
        if not fname:
            continue
        fpath = folder / fname
        by_path[join_path(path_tokens(pth))] = {
            "path": pth,
            "file": str(fpath) if fpath.is_file() else fname,
            "name": fname,
            "seat": node.get("seat"),
            "seat_index": node.get("seat_index"),
            "description": node.get("description") or "",
            "num_hands": node.get("num_hands"),
        }
    if not by_path:
        return None
    return {
        "index": str(idx),
        "dir": str(folder),
        "spot": data.get("spot"),
        "seat_order": data.get("seat_order"),
        "by_path": by_path,
    }


def build_line_nav(
    nodes: Sequence[dict[str, Any]],
    *,
    chart_pack: dict[str, Any] | None = None,
    num_seats: int | None = None,
    street: int | None = None,
) -> dict[str, Any]:
    """GTOW-style line graph: current node → action → next player's node.

    Tokenized paths (``open`` / ``F`` / ``AI,F``) are navigable. History-hash
    dumps are marked ``navigable=False`` (action mix still listed).
    """
    pack_paths = (chart_pack or {}).get("by_path") or {}
    by_path: dict[str, dict[str, Any]] = {}
    for n in nodes:
        key = join_path(path_tokens(n.get("path")))
        # Prefer the node that actually owns this path (first wins; usually unique)
        by_path.setdefault(key, n)

    def _actions_for(node: dict[str, Any]) -> list[dict[str, Any]]:
        tokens = path_tokens(node.get("path"))
        mix = ((node.get("aggregate") or {}).get("mean_mix")) or {}
        # Preserve solver action order when present
        order = list((node.get("aggregate") or {}).get("actions") or mix.keys())
        out: list[dict[str, Any]] = []
        for act in order:
            freq = float(mix.get(act, 0.0))
            tok = action_to_token(act)
            nxt_path = join_path([*tokens, tok])
            nxt = by_path.get(nxt_path)
            pack = pack_paths.get(nxt_path)
            # Charts use F for fold / AI for all-in; XC may not appear
            if nxt is None and pack is None and tok == "XC":
                for alt in ("X", "C"):
                    alt_path = join_path([*tokens, alt])
                    nxt = by_path.get(alt_path)
                    pack = pack_paths.get(alt_path)
                    if nxt or pack:
                        nxt_path = alt_path
                        break
            a_up = str(act).upper()
            css = (
                "act-fold"
                if a_up == "FOLD"
                else "act-call"
                if a_up in ("CHECK_CALL", "CHECK", "CALL")
                else "act-allin"
                if a_up == "ALLIN"
                else "act-raise"
            )
            short = (
                "F"
                if a_up == "FOLD"
                else "X/C"
                if a_up in ("CHECK_CALL", "CHECK", "CALL")
                else "AI"
                if a_up == "ALLIN"
                else tok
            )
            out.append(
                {
                    "action": act,
                    "token": tok,
                    "short": short,
                    "freq": round(freq, 4),
                    "css": css,
                    "next_path": nxt_path,
                    "next_seat": int(nxt["seat"]) if nxt else None,
                    "has_next": nxt is not None or pack is not None,
                    "chart_file": (pack or {}).get("file"),
                    "chart_name": (pack or {}).get("name"),
                    "terminal": nxt is None and pack is None,
                }
            )
        return out

    entries: dict[str, Any] = {}
    for key, n in by_path.items():
        entries[key] = {
            "seat": int(n.get("seat", 0)),
            "path": key,
            "label": n.get("label") or f"P{n.get('seat', 0)}",
            "path_pretty": n.get("path_pretty") or key,
            "num_hands": int(n.get("num_hands") or 0),
            "aggregate": n.get("aggregate") or {},
            "actions": _actions_for(n),
        }

    # Root = shortest token path, then lowest seat
    if by_path:
        root_node = min(
            by_path.values(),
            key=lambda n: (len(path_tokens(n.get("path"))), int(n.get("seat", 0))),
        )
        root_path = join_path(path_tokens(root_node.get("path")))
        root_seat = int(root_node.get("seat", 0))
    else:
        root_path, root_seat = "open", 0

    tokenish = [
        n
        for n in nodes
        if path_tokens(n.get("path")) or str(n.get("path") or "") in ("", "open", "root")
    ]
    navigable = bool(entries) and (
        bool(pack_paths) or any(path_tokens(n.get("path")) or True for n in tokenish)
    )
    # Hash-only dumps: every path is a long digit string → not navigable
    if nodes and all(
        (str(n.get("path") or "").isdigit() and len(str(n.get("path") or "")) >= 8)
        for n in nodes
    ):
        navigable = False

    return {
        "navigable": navigable,
        "root_path": root_path,
        "root_seat": root_seat,
        "num_seats": num_seats,
        "street": street,
        "by_path": entries,
    }


def compare_strategies(
    rows_a: Sequence[dict[str, Any]],
    rows_b: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compare two strategy row lists by hand_label (or private id).

    Returns per-hand L1 strategy distance and mean L1.
    """
    def index(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out = {}
        for r in rows:
            key = str(r.get("hand_label") or r.get("private") or r.get("infoset_id"))
            # also seat+path for disambiguation
            full = f"{r.get('seat')}|{r.get('path')}|{key}"
            out[full] = r
        return out

    ia, ib = index(rows_a), index(rows_b)
    common = set(ia) & set(ib)
    diffs = []
    for k in sorted(common):
        sa = {s["action"]: float(s["prob"]) for s in (ia[k].get("strategy") or [])}
        sb = {s["action"]: float(s["prob"]) for s in (ib[k].get("strategy") or [])}
        acts = set(sa) | set(sb)
        l1 = sum(abs(sa.get(a, 0.0) - sb.get(a, 0.0)) for a in acts)
        diffs.append(
            {
                "key": k,
                "hand": ia[k].get("hand_label"),
                "l1": round(l1, 4),
                "a_primary": ia[k].get("primary_action"),
                "b_primary": ib[k].get("primary_action"),
            }
        )
    diffs.sort(key=lambda d: -d["l1"])
    mean_l1 = sum(d["l1"] for d in diffs) / len(diffs) if diffs else 0.0
    return {
        "num_a": len(rows_a),
        "num_b": len(rows_b),
        "num_common": len(common),
        "mean_l1": round(mean_l1, 4),
        "top_diffs": diffs[:50],
    }
