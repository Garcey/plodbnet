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
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from plo5bp.gto.cfr_api import RootSpec

STREET_NAMES = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River"}


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
    terminal_kind: str = ""  # fold | showdown | next_street | allin_runout | truncated
    seat_label: str = ""  # position in the SOLVER's seat order (SB / BB / UTG… / OOP / IP)

    def as_dict(self, *, max_depth: int | None = None) -> dict[str, Any]:
        if max_depth is not None and self.depth > max_depth:
            return {
                "id": self.id,
                "label": self.label + " …",
                "seat": self.seat,
                "seat_label": self.seat_label,
                "depth": self.depth,
                "truncated": True,
                "num_children": len(self.children),
            }
        return {
            "id": self.id,
            "label": self.label,
            "seat": self.seat,
            "seat_label": self.seat_label,
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


def _position_names(n: int, street: int) -> list[str]:
    """Seat index → position, in the SOLVER's seat order.

    Preflop HU: seat 0 = BB, seat 1 = SB/button (acts first). Preflop 3+ handed:
    seats 0..n-3 = UTG…BTN, n-2 = SB, n-1 = BB, seat 0 first
    (``solve_multiway_preflop_mccfr``). Postflop: seat 0 acts first (OOP).
    """
    if street != 0:
        return ["OOP", "IP"] if n == 2 else [f"seat{i}" for i in range(n)]
    if n == 2:
        return ["BB", "SB"]
    # Standard names for the non-blind seats, first to act → button (n ≤ 6).
    early = {3: ["BTN"], 4: ["CO", "BTN"], 5: ["HJ", "CO", "BTN"], 6: ["UTG", "HJ", "CO", "BTN"]}.get(
        n, [f"seat{i}" for i in range(n - 2)]
    )
    return early + ["SB", "BB"]


class _PreviewState:
    """One betting round in integer chips — a mirror of ``PublicState``.

    (review 2026-09-20 preview tree ≠ solver) The old preview worked in float bb
    with its own rules: no min-raise clamp, no de-dup of sizes that clamp to the
    same chips, a limp that ended the hand, antes folded into the amount to
    call, and P0 first preflop. Same arithmetic as the solver (integer chips,
    ``pot * pm / 1000`` floor division) means the same menus, to the chip.
    """

    def __init__(self, root: RootSpec) -> None:
        bb = max(1, int(root.bb_chips))
        n = max(2, int(root.num_seats))
        self.bb, self.n, self.street = bb, n, int(root.street)
        eff = int(round(float(root.effective_stack_bb) * bb))
        self.stacks = (
            [int(round(float(s) * bb)) for s in root.stacks_bb]
            if root.stacks_bb and len(root.stacks_bb) == n
            else [eff] * n
        )
        self.commit = [0] * n
        self.acted = [False] * n
        self.folded = [False] * n
        self.bet_to_call = 0
        self.last_raise = bb
        if self.street == 0:
            self.pot = 0
            ante = max(0, int(root.ante_chips))
            for i in range(n):  # antes are dead money: pot, not street commit
                put = min(self.stacks[i], ante)
                self.stacks[i] -= put
                self.pot += put
            sb_seat, bb_seat = (1, 0) if n == 2 else (n - 2, n - 1)
            for seat, blind in ((sb_seat, max(0, int(root.sb_chips))), (bb_seat, bb)):
                put = min(self.stacks[seat], blind)  # blinds are live
                self.stacks[seat] -= put
                self.commit[seat] += put
                self.pot += put
            self.bet_to_call = bb
            self.actor: int | None = 1 if n == 2 else 0
        else:
            self.pot = int(round(float(root.pot_bb) * bb))
            self.actor = 0
        if self.actor is not None and not self._can_act(self.actor):
            self.actor = self._next_actor(self.actor)

    def clone(self) -> "_PreviewState":
        other = object.__new__(type(self))  # keep the subclass (and its menu rules)
        other.__dict__ = dict(self.__dict__)
        for k in ("stacks", "commit", "acted", "folded"):
            setattr(other, k, list(getattr(self, k)))
        return other

    # --- queries (names follow public_state.rs) ---------------------------

    def alive(self) -> list[int]:
        return [i for i in range(self.n) if not self.folded[i]]

    def _can_act(self, i: int) -> bool:
        return not self.folded[i] and self.stacks[i] > 0

    def _next_actor(self, after: int) -> int | None:
        for step in range(1, self.n + 1):
            i = (after + step) % self.n
            if self._can_act(i) and (not self.acted[i] or self.commit[i] < self.bet_to_call):
                return i
        return None

    def to_call(self) -> int:
        a = self.actor
        return min(max(0, self.bet_to_call - self.commit[a]), self.stacks[a])

    def _max_other_total(self) -> int:
        a = self.actor
        return max(
            (self.commit[j] + self.stacks[j] for j in range(self.n) if j != a and not self.folded[j]),
            default=0,
        )

    def _min_bet_total(self) -> int:
        return self.bb if self.bet_to_call == 0 else self.bet_to_call + self.last_raise

    def min_raise(self) -> int:
        a = self.actor
        min_total = self._min_bet_total()
        if min_total <= self.commit[a]:
            return 0
        max_other = self._max_other_total()
        if max_other <= self.bet_to_call:
            return 0
        cap_delta = max_other - self.commit[a]
        if cap_delta <= 0:
            return 0
        clamped = min(min_total - self.commit[a], cap_delta)
        return 0 if clamped > self.stacks[a] else clamped

    def max_raise(self) -> int:
        a = self.actor
        cap_total = min(self._max_other_total(), self.commit[a] + self.stacks[a])
        if cap_total <= self.commit[a] or cap_total <= self.bet_to_call:
            return 0
        return cap_total - self.commit[a]

    def raise_chips_for_pm(self, pm: int) -> int | None:
        """Pot-fraction raise, clamped to [min raise, max raise] — or None."""
        a = self.actor
        to_call_raw = max(0, self.bet_to_call - self.commit[a])
        if self.bet_to_call == 0:
            target_total = self.pot * pm // 1000
        else:
            target_total = self.bet_to_call + (self.pot + to_call_raw) * pm // 1000
        clamped = min(max(target_total, self._min_bet_total()), self._max_other_total())
        want = min(max(0, clamped - self.commit[a]), self.stacks[a])
        min_r, max_r = self.min_raise(), self.max_raise()
        if min_r == 0 or want < min_r or want > max_r:
            return None
        return want

    def menu(self, sizes: Sequence[int], allin_atom: bool) -> list[str]:
        """``legal_actions`` in actions.rs: one physical action = one label."""
        to_call, stack = self.to_call(), self.stacks[self.actor]
        min_r, max_r = self.min_raise(), self.max_raise()
        can_raise = min_r > 0 and max_r >= min_r
        if not sizes and allin_atom and self.street == 0:  # preflop push/fold
            if to_call > 0:
                return ["FOLD", "ALLIN"] if stack > 0 else ["FOLD"]
            return ["ALLIN"] if can_raise and stack >= min_r else ["CHECK_CALL"]
        out = (["FOLD"] if to_call > 0 else []) + ["CHECK_CALL"]
        if can_raise:
            seen: list[int] = []
            for pm in sizes:
                chips = self.raise_chips_for_pm(int(pm))
                if chips is None or (allin_atom and chips == max_r) or chips in seen:
                    continue  # illegal, IS the all-in, or same chips as an earlier size
                seen.append(chips)
                out.append(f"RAISE_{int(pm)}")
            if allin_atom:
                out.append("ALLIN")
        return out

    # --- transitions ------------------------------------------------------

    def apply(self, action: str) -> None:
        a = self.actor
        if action == "FOLD":
            self.folded[a] = True
        else:
            if action == "CHECK_CALL":
                chips = self.to_call()
            elif action == "ALLIN":
                # The maximum legal raise; with no raise legal it is a call all-in.
                chips = self.max_raise() if self.min_raise() > 0 else self.to_call()
            else:
                chips = self.raise_chips_for_pm(int(action.split("_", 1)[1])) or self.to_call()
            chips = min(chips, self.stacks[a])
            self.stacks[a] -= chips
            self.commit[a] += chips
            self.pot += chips
            if self.commit[a] > self.bet_to_call:
                size = self.commit[a] - self.bet_to_call
                if size >= self.last_raise:
                    self.last_raise = size  # a full raise sets the next minimum
                self.bet_to_call = self.commit[a]
        self.acted[a] = True
        self.actor = None if len(self.alive()) <= 1 else self._next_actor(a)

    def terminal_kind(self) -> str:
        alive = self.alive()
        if len(alive) <= 1:
            return "fold"
        # Betting is over. With at most one player holding chips the rest of the
        # board is simply run out; otherwise play continues on the next street.
        if sum(1 for i in alive if self.stacks[i] > 0) <= 1:
            return "allin_runout"
        return "showdown" if self.street >= 3 else "next_street"


def build_abstract_tree(
    root: RootSpec | dict[str, Any],
    *,
    max_nodes: int = 400,
    max_depth: int = 8,
) -> dict[str, Any]:
    """Enumerate the abstract bet-size tree of the ROOT street's betting round.

    Mirrors the solver's public state (seat order, blinds/antes, min-raise
    clamp, size de-dup, all-in) — see :class:`_PreviewState`. Later streets are
    not expanded: a closed round ends in ``next_street`` / ``showdown`` /
    ``allin_runout`` / ``fold``.
    """
    if isinstance(root, dict):
        from plo5bp.cfr_app.session import _root_from_dict

        r = _root_from_dict(root)
    else:
        r = root

    bb = float(max(1, int(r.bb_chips)))
    names = _position_names(max(2, int(r.num_seats)), int(r.street))
    counter = {"n": 0}
    nodes_flat: list[dict[str, Any]] = []
    sizes = [int(pm) for pm in r.raise_sizes_pm]

    def new_id() -> str:
        counter["n"] += 1
        return f"n{counter['n']}"

    def build(state: _PreviewState, depth: int, path_label: str) -> TreeNode:
        nid = new_id()
        if state.actor is None:
            kind = state.terminal_kind()
            alive = state.alive()
            seat = alive[0] if alive else 0
            suffix = {
                "fold": "fold_win",
                "showdown": "showdown",
                "next_street": "next street",
                "allin_runout": "runout",
            }[kind]
            return TreeNode(
                id=nid,
                label=f"{path_label} → {suffix}" if path_label else suffix,
                seat=seat,
                depth=depth,
                facing_bet=False,
                pot_bb=state.pot / bb,
                stack_bb=state.stacks[seat] / bb,
                to_call_bb=0.0,
                terminal=True,
                terminal_kind=kind,
                seat_label=names[seat],
            )
        seat = state.actor
        to_call = state.to_call()
        if counter["n"] > max_nodes or depth > max_depth:
            # A genuine size cap — not an all-in chain (those terminate above).
            return TreeNode(
                id=nid,
                label="(truncated)",
                seat=seat,
                depth=depth,
                facing_bet=to_call > 0,
                pot_bb=state.pot / bb,
                stack_bb=state.stacks[seat] / bb,
                to_call_bb=to_call / bb,
                terminal=True,
                terminal_kind="truncated",
                seat_label=names[seat],
            )
        menu = state.menu(sizes, bool(r.allin_atom))
        node = TreeNode(
            id=nid,
            label=path_label or "root",
            seat=seat,
            depth=depth,
            facing_bet=to_call > 0,
            pot_bb=state.pot / bb,
            stack_bb=state.stacks[seat] / bb,
            to_call_bb=to_call / bb,
            actions=menu,
            seat_label=names[seat],
        )
        for act in menu:
            child_state = state.clone()
            child_state.apply(act)
            node.children.append(
                build(child_state, depth + 1, f"{path_label}/{act}" if path_label else act)
            )
        nodes_flat.append(
            {
                "id": node.id,
                "label": node.label,
                "seat": node.seat,
                "seat_label": node.seat_label,
                "depth": node.depth,
                "actions": node.actions,
                "pot_bb": node.pot_bb,
                "to_call_bb": node.to_call_bb,
                "terminal": node.terminal,
            }
        )
        return node

    tree = build(_PreviewState(r), 0, "")

    return {
        "root_id": r.root_id,
        "street": r.street,
        "street_name": STREET_NAMES.get(r.street, str(r.street)),
        "num_seats": max(2, int(r.num_seats)),
        "seat_labels": names,
        "pot_bb": float(r.pot_bb),
        "effective_stack_bb": float(r.effective_stack_bb),
        "raise_sizes_pm": list(r.raise_sizes_pm),
        "allin_atom": r.allin_atom,
        "board": list(r.board),
        "num_nodes_built": counter["n"],
        "max_nodes": max_nodes,
        "tree": tree.as_dict(max_depth=max_depth),
        "flat": nodes_flat[:max_nodes],
    }


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
    """Range-weighted action mix at a node.

    (review 2026-09-20 E7) Rows are weighted by ``visit_mass`` — the solver's
    accumulated reach at the infoset — falling back to combos-per-class (6/4/12)
    when a file has no mass. This used to be an unweighted mean over the LIST of
    hands, so never-reached hands (still at their 1/n default) and rarely-held
    hands counted as much as the core of the range: 52.7% shown where the
    reach-weighted mix was 36.6%. ``actions`` keeps the SOLVER's order (E8) —
    it was sorted by frequency, which reshuffled the action strip per node.
    """
    from plo5bp.cfr_app.strategy_view import row_weights, weighted_strategy

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
            "weight": 0.0,
        }
    actions, probs = weighted_strategy(rows)
    mean_mix = {str(a).upper(): float(p) for a, p in zip(actions, probs)}
    fold = call = raise_p = allin = 0.0
    for a, p in mean_mix.items():
        if a == "FOLD":
            fold += p
        elif a in ("CHECK_CALL", "CHECK", "CALL"):
            call += p
        elif a == "ALLIN":
            allin += p
        elif a.startswith("RAISE"):
            raise_p += p

    weights = row_weights(rows)
    total_w = sum(weights)
    ent = agg50 = fold50 = 0.0
    for r, w in zip(rows, weights):
        if w <= 0.0:
            continue
        for p in r.get("probs") or [s.get("prob") for s in (r.get("strategy") or [])]:
            p = float(p or 0.0)
            if p > 1e-12 and math.isfinite(p):
                ent -= w * p * math.log(p)
        if float(r.get("primary_prob") or 0) >= 0.5:
            primary = str(r.get("primary_action") or "").upper()
            if primary == "FOLD":
                fold50 += w
            elif primary not in ("CHECK_CALL", "CHECK", "CALL"):
                agg50 += w
    denom = total_w if total_w > 0 else 1.0
    return {
        "actions": list(mean_mix.keys()),  # solver order
        "mean_mix": {k: round(v, 4) for k, v in mean_mix.items()},
        "fold": round(fold, 4),
        "call": round(call, 4),
        "raise": round(raise_p, 4),
        "allin": round(allin, 4),
        "entropy": round(ent / denom, 4),
        "num_hands": len(rows),
        "weight": round(total_w, 6),
        # share of the range (by weight) whose primary action is ≥ 50% aggressive / fold
        "agg_ge_50pct": round(agg50 / denom, 4),
        "fold_ge_50pct": round(fold50 / denom, 4),
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


def _is_hash(path: str | None) -> bool:
    """A legacy history-hash node id (``1499570753115261709`` / ``h1499…``)."""
    p = str(path or "").strip()
    if p.isdigit() and len(p) >= 8:
        return True
    return p.startswith("h") and p[1:].isdigit() and len(p) >= 8  # same bounds as before


def path_tokens(path: str | None) -> list[str]:
    p = str(path or "").strip()
    if p in ("", "root", "open"):
        return []
    # History hashes are not action sequences
    if _is_hash(p):
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

    def _nav_key(path: Any) -> str:
        # A legacy history hash is a node id, not an action line: keep it as its
        # own key. (path_tokens() maps every hash to [] → all of them used to
        # collapse onto the single key "open".)
        p = str(path or "")
        if _is_hash(p):
            return p
        return join_path(path_tokens(p))

    by_path: dict[str, dict[str, Any]] = {}
    for n in nodes:
        # Prefer the node that actually owns this path (first wins; usually unique)
        by_path.setdefault(_nav_key(n.get("path")), n)

    # (review 2026-09-20 E8) Native dumps spell path tokens with the solver's full
    # action labels (CHECK_CALL, RAISE_500, ALLIN); chart packs use short tokens
    # (XC, R500, AI). next_path was always built from the SHORT token, so in a
    # native dump no child was ever found and every action showed as terminal.
    full_style = any(
        t in ("FOLD", "ALLIN", "CHECK_CALL") or t.startswith("RAISE_")
        for n in nodes
        for t in path_tokens(n.get("path"))
    )

    def _actions_for(node: dict[str, Any]) -> list[dict[str, Any]]:
        tokens = path_tokens(node.get("path"))
        mix = ((node.get("aggregate") or {}).get("mean_mix")) or {}
        # Solver action order (aggregate_node keeps it) — not frequency order.
        order = list((node.get("aggregate") or {}).get("actions") or mix.keys())
        out: list[dict[str, Any]] = []
        for act in order:
            freq = float(mix.get(act, 0.0))
            tok = action_to_token(act)
            full = str(act).upper()
            # Try the dump's own spelling first, then chart tokens (XC may be X / C).
            candidates = [full, tok] + (["X", "C"] if tok == "XC" else [])
            nxt = pack = None
            nxt_path = join_path([*tokens, full if full_style else tok])
            for cand in candidates:
                cand_path = join_path([*tokens, cand])
                nxt = by_path.get(cand_path)
                pack = pack_paths.get(cand_path)
                if nxt or pack:
                    nxt_path = cand_path
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

    # Root = the "open" node when the dump has one, else the shortest line.
    if by_path:
        root_node = min(
            by_path.values(),
            key=lambda n: (
                _nav_key(n.get("path")) != "open",
                len(path_tokens(n.get("path"))),
                int(n.get("seat", 0)),
            ),
        )
        root_path = _nav_key(root_node.get("path"))
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
            # hand_label is the REAL hand (raw_combo under isomorphism) — the same
            # label the viewer and the hand filter use (review 2026-09-20 E5).
            key = str(r.get("hand_label") or r.get("private") or r.get("infoset_id"))
            # seat + path + runout: the same line on two river cards is two nodes (E6).
            full = f"{r.get('seat')}|{r.get('path')}|{r.get('runout') or ''}|{key}"
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
