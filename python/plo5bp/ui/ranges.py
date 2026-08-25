"""GTO-Wizard-style range grid for NLH (`/ranges/*`) — LOCAL BUILD ONLY.

Serves the 169-hand strategy grid: for a node defined by (seats, stack,
action line, board), query the NLH model once per live hole combo and
aggregate to grid cells. Design invariants:

- **Stateless**: every query carries the full spec; the server replays
  EPHEMERAL study envs (never the user's study `Session`). `line` is one
  ordered list interleaving actions and street cards:
  ``{"t": "a", "gate": "fold|check_call|raise", "chips_bb": 2.5}`` /
  ``{"t": "cards", "cards": [12, 25, 38]}`` (cards apply when the engine
  awaits that street — 3 for the flop, then 1 and 1).
- **Bit-exact**: grid rows are `pack_range_nlh` + the training batch
  encoder, pinned bit-exact against serial study observations by
  tests/python/test_ranges.py. The observation is villain-blind, so one
  ephemeral env serves all 1326 combos.
- **Reach weighting** is per-seat and gate-level: the acting seat's range
  entering a node is the product of its own earlier action probabilities
  (fold / check-call / raise-as-one-bucket — no size-conditional reach in
  v1), with combos colliding with board cards excluded as streets deal.
- **Fixed stake**: bb=10000, sb=bb/2, ante=bb/2 per player (the trained
  5/10($5) table); button is seat 0, so positions are canonical.
- Node results are cached per (model, seats, stack, line-prefix); the
  cache lives in-process and dies with the server, so a promoted
  checkpoint never serves stale grids.

The public build (`PLO5BP_PUBLIC=1`) strips every `/ranges` route in
server.py — same mechanism as `/ocr` and `/pokernow` — until the feature
is validated locally and deliberately shipped.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.config import VARIANT_NLH, GameConfig
from plo5bp.encoding_nlh import encode_observation_batch_nlh
from plo5bp.env import BombPotEnv
from plo5bp.sizing import anchor_grid_np, anchor_grid_torch, sizing_from_info
from plo5bp.ui.common import (
    AWAITING_NAMES,
    STREET_NAMES,
    anchor_label_spec,
    position_name,
)

logger = logging.getLogger("plo5bp.ui.ranges")

BB_CHIPS = 10_000  # fixed 5/10($5) stake, matching training + trainer defaults

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"

#: Canonical combo order: all C(52,2) pairs (lo, hi), lo < hi.
ALL_COMBOS: list[tuple[int, int]] = [
    (i, j) for i in range(52) for j in range(i + 1, 52)
]
_COMBO_POS = {c: k for k, c in enumerate(ALL_COMBOS)}

_GATES = {"fold": GATE_FOLD, "check_call": GATE_CHECK_CALL, "raise": GATE_RAISE}


def combo_name(lo: int, hi: int) -> str:
    """Display name, higher rank first (ties broken by suit index)."""
    a, b = (hi, lo) if (hi >> 2, hi & 3) >= (lo >> 2, lo & 3) else (lo, hi)
    if (a >> 2) < (b >> 2):
        a, b = b, a
    return (
        f"{RANK_CHARS[a >> 2]}{SUIT_CHARS[a & 3]}"
        f"{RANK_CHARS[b >> 2]}{SUIT_CHARS[b & 3]}"
    )


def cell_key(lo: int, hi: int) -> str:
    """169-grid cell for a combo: 'AA' / 'AKs' / 'AKo' (high rank first)."""
    r_lo, r_hi = lo >> 2, hi >> 2
    hi_r, lo_r = max(r_lo, r_hi), min(r_lo, r_hi)
    if hi_r == lo_r:
        return RANK_CHARS[hi_r] * 2
    suited = (lo & 3) == (hi & 3)
    return f"{RANK_CHARS[hi_r]}{RANK_CHARS[lo_r]}{'s' if suited else 'o'}"


class LineEntry(BaseModel):
    t: str  # "a" | "cards"
    gate: str | None = None
    chips_bb: float | None = None
    cards: list[int] | None = None


class RangeQuery(BaseModel):
    seats: int = Field(6, ge=2, le=6)
    stack_bb: float = Field(100.0, gt=1.0, le=1000.0)
    line: list[LineEntry] = Field(default_factory=list)
    node: int | None = None  # view the state after this many entries


def _nlh_cfg(seats: int, stack_bb: float) -> GameConfig:
    return GameConfig(
        num_seats=seats,
        starting_stack=int(round(stack_bb * BB_CHIPS)),
        ante=BB_CHIPS // 2,
        bb=BB_CHIPS,
        variant=VARIANT_NLH,
        sb=BB_CHIPS // 2,
    )


def _placeholder_hole(line: list[LineEntry]) -> list[int]:
    """Two cards guaranteed absent from every cards-entry in the line (the
    street setters validate new cards against the placeholder hero hole)."""
    used: set[int] = set()
    for e in line:
        if e.t == "cards" and e.cards:
            used.update(int(c) for c in e.cards)
    out: list[int] = []
    for idx in range(51, -1, -1):
        if idx not in used:
            out.append(idx)
            if len(out) == 2:
                return out
    raise HTTPException(400, "line uses too many cards")


def _bad(i: int, msg: str) -> HTTPException:
    return HTTPException(400, f"line entry {i}: {msg}")


def _replay(
    cfg: GameConfig, line: list[LineEntry], upto: int
) -> tuple[BombPotEnv, Any, bool, list[dict[str, Any]]]:
    """Fresh env, `line[:upto]` applied. Returns (env, info, done, seq_meta):
    one meta dict per applied entry, annotated with the pre-action actor."""
    env = BombPotEnv(cfg)
    obs, info = env.reset_study_nlh(
        button=0, hero_seat=0, hero_hole=_placeholder_hole(line)
    )
    done = False
    seq: list[dict[str, Any]] = []
    pos = lambda seat: position_name(seat, 0, cfg.num_seats)  # noqa: E731
    for i, e in enumerate(line[:upto]):
        if e.t == "a":
            if info.actor is None:
                # Study boundaries set the done flag too — awaiting wins.
                if env.awaiting_next_street() is not None:
                    raise _bad(i, "action while the engine awaits street cards")
                raise _bad(i, "action after the hand ended")
            gate = _GATES.get(e.gate or "")
            if gate is None:
                raise _bad(i, f"unknown gate {e.gate!r}")
            if not bool(info.gate_mask[gate]):
                raise _bad(i, f"{e.gate} is not legal here")
            chips = 0
            if gate == GATE_RAISE:
                if e.chips_bb is None:
                    raise _bad(i, "raise needs chips_bb")
                chips = int(round(float(e.chips_bb) * cfg.bb))
                if not (info.min_raise_chips <= chips <= info.max_raise_chips):
                    raise _bad(
                        i,
                        f"raise to {e.chips_bb}bb outside "
                        f"[{info.min_raise_chips / cfg.bb:g}, "
                        f"{info.max_raise_chips / cfg.bb:g}]bb",
                    )
            actor = int(info.actor)
            street_idx = int(info.raw_obs.get("street", 0))
            obs, _r, done, info = env.step_hybrid(gate, chips)
            seq.append({
                "i": i,
                "t": "a",
                "seat": actor,
                "position": pos(actor),
                "street": STREET_NAMES.get(street_idx, "?"),
                "gate": e.gate,
                "chips_bb": (
                    round(chips / cfg.bb, 4) if gate == GATE_RAISE else None
                ),
            })
        elif e.t == "cards":
            awaiting = env.awaiting_next_street()
            if awaiting is None:
                raise _bad(i, "cards while an action is pending")
            cards = [int(c) for c in (e.cards or [])]
            want = 3 if awaiting == 1 else 1
            if len(cards) != want:
                raise _bad(i, f"{AWAITING_NAMES[awaiting]} needs {want} card(s)")
            try:
                if awaiting == 1:
                    obs, info = env.set_flop_nlh(*cards)
                elif awaiting == 2:
                    obs, info = env.set_turn_nlh(cards[0])
                else:
                    obs, info = env.set_river_nlh(cards[0])
            except Exception as exc:  # engine validation (dupes / range)
                raise _bad(i, f"bad street cards ({exc})") from exc
            done = False  # the round-closing action set it; play continues
            seq.append({
                "i": i,
                "t": "cards",
                "street": AWAITING_NAMES[awaiting],
                "cards": cards,
            })
        else:
            raise _bad(i, f"unknown entry type {e.t!r}")
    return env, info, done, seq


class _NodeResult:
    """Model outputs for one node, over its live combos."""

    __slots__ = (
        "actor", "live_idx", "gate_probs", "anchor_probs", "values",
        "board", "state_meta",
    )

    def __init__(self, actor, live_idx, gate_probs, anchor_probs, values,
                 board, state_meta):
        self.actor = actor
        self.live_idx = live_idx          # (M,) indices into ALL_COMBOS
        self.gate_probs = gate_probs      # (M, 3) float32
        self.anchor_probs = anchor_probs  # (M, A) float32 (zeros if no raise)
        self.values = values              # (M,) float32, bb
        self.board = board                # list[int]
        self.state_meta = state_meta      # dict for the client


class _NodeCache:
    def __init__(self, cap: int = 256):
        self._map: OrderedDict[str, Any] = OrderedDict()
        self._cap = cap
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            v = self._map.get(key)
            if v is not None:
                self._map.move_to_end(key)
            return v

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._map[key] = value
            self._map.move_to_end(key)
            while len(self._map) > self._cap:
                self._map.popitem(last=False)


def create_ranges_router(
    formats: dict[str, dict[str, Any]],
    device: torch.device,
    nlh_ckpt_name: str = "",
    gto_model: Any | None = None,
) -> APIRouter:
    """Ranges grid. When ``gto_model`` is set (PolicyNet), use it for
    NLH range queries so Study/Trainer/Ranges share one backend."""
    router = APIRouter()
    cache = _NodeCache()

    def _model():
        if gto_model is not None:
            return gto_model, True
        entry = formats[VARIANT_NLH]
        return entry["model"], bool(entry.get("loaded"))

    def _line_key(q: RangeQuery, k: int) -> str:
        model, _ = _model()
        return json.dumps(
            [
                id(model), q.seats, q.stack_bb,
                [e.model_dump(exclude_none=True) for e in q.line[:k]],
            ],
            separators=(",", ":"),
        )

    def _compute_node(q: RangeQuery, k: int):
        """NodeResult for the state after line[:k], or a dict describing a
        non-actionable state (terminal / awaiting street cards)."""
        key = _line_key(q, k)
        hit = cache.get(key)
        if hit is not None:
            return hit

        cfg = _nlh_cfg(q.seats, q.stack_bb)
        env, info, done, _seq = _replay(cfg, q.line, k)
        board = [int(c) for c in info.raw_obs.get("board_a", [])]

        # A study street boundary sets the done flag too (no action possible
        # without new cards) — the awaiting check must come first.
        awaiting = env.awaiting_next_street()
        if awaiting is not None:
            result: Any = {
                "awaiting": AWAITING_NAMES[awaiting],
                "need": 3 if awaiting == 1 else 1,
                "board": board,
            }
            cache.put(key, result)
            return result
        if info.actor is None:
            result = {"terminal": True, "board": board}
            cache.put(key, result)
            return result

        actor = int(info.actor)
        on_board = set(board)
        live_idx = np.array(
            [
                p for p, (lo, hi) in enumerate(ALL_COMBOS)
                if lo not in on_board and hi not in on_board
            ],
            dtype=np.int64,
        )
        holes = np.array(
            [ALL_COMBOS[p] for p in live_idx], dtype=np.uint8
        )

        model, _loaded = _model()
        spec = model.anchor_spec
        pack = env.pack_range_nlh(holes)
        obs = encode_observation_batch_nlh(pack, pack["hero_cat_a"], cfg)
        m = obs.shape[0]

        gm = np.broadcast_to(
            np.asarray(info.gate_mask, dtype=bool), (m, 3)
        ).copy()
        sizing = sizing_from_info(info)
        sizing_t = torch.from_numpy(
            np.broadcast_to(sizing, (m, sizing.shape[0])).copy()
        ).to(device)

        raise_legal = bool(info.gate_mask[GATE_RAISE])
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device)
            gm_t = torch.from_numpy(gm).to(device)
            gate_logits, anchor_out, _refine, value = model(obs_t, gm_t)
            gate_probs = (
                F.softmax(gate_logits, dim=-1).float().cpu().numpy()
            )
            if raise_legal:
                grid_t = anchor_grid_torch(sizing_t, spec)
                anchor_probs = (
                    model._anchor_dist(anchor_out, grid_t)
                    .probs.float().cpu().numpy()
                )
            else:
                anchor_probs = np.zeros((m, spec.count), dtype=np.float32)
            values = value.squeeze(-1).float().cpu().numpy().reshape(-1)

        grid = anchor_grid_np(sizing[0], sizing[1], sizing[2], sizing[3], spec)
        hero_commit = int(info.raw_obs["street_commit"][actor])
        to_call = max(int(info.raw_obs["bet_to_call"]) - hero_commit, 0)
        anchors = [
            {
                "k": int(j),
                "label": anchor_label_spec(spec, int(j)),
                "chips_bb": round(int(grid.chips[j]) / cfg.bb, 4),
            }
            for j in range(spec.count)
            if raise_legal and bool(grid.legal[j])
        ]
        state_meta = {
            "actor": actor,
            "position": position_name(actor, 0, cfg.num_seats),
            "street": STREET_NAMES.get(int(info.raw_obs.get("street", 0)), "?"),
            "board": board,
            "pot_bb": round(int(info.raw_obs["pot"]) / cfg.bb, 4),
            "to_call_bb": round(to_call / cfg.bb, 4),
            "legal": {
                "fold": bool(info.gate_mask[GATE_FOLD]),
                "check_call": bool(info.gate_mask[GATE_CHECK_CALL]),
                "raise": raise_legal,
            },
            "min_raise_bb": round(info.min_raise_chips / cfg.bb, 4),
            "max_raise_bb": round(info.max_raise_chips / cfg.bb, 4),
            "anchors": anchors,
            "allin_k": (spec.count - 1) if spec.allin_atom else None,
        }
        result = _NodeResult(
            actor, live_idx, gate_probs, anchor_probs, values, board,
            state_meta,
        )
        cache.put(key, result)
        return result

    def _actor_reach(q: RangeQuery, k: int, actor: int) -> np.ndarray:
        """Π of the actor's own earlier action probabilities, over the full
        canonical combo vector (blocked combos handled by exclusion at the
        display node)."""
        reach = np.ones(len(ALL_COMBOS), dtype=np.float64)
        applied = 0
        for i, e in enumerate(q.line[:k]):
            if e.t != "a":
                continue
            node = _compute_node(q, i)
            if not isinstance(node, _NodeResult) or node.actor != actor:
                continue
            gate = _GATES[e.gate or "check_call"]
            reach[node.live_idx] *= node.gate_probs[:, gate].astype(np.float64)
            applied += 1
        return reach

    @router.post("/ranges/query")
    def ranges_query(q: RangeQuery) -> dict[str, Any]:
        n_entries = len(q.line)
        k = n_entries if q.node is None else int(q.node)
        if not (0 <= k <= n_entries):
            raise HTTPException(400, f"node must be in [0, {n_entries}]")

        cfg = _nlh_cfg(q.seats, q.stack_bb)
        # Full-line replay validates every entry and yields strip metadata.
        _env, _info, _done, seq = _replay(cfg, q.line, n_entries)

        model, loaded = _model()
        out: dict[str, Any] = {
            "node": k,
            "num_entries": n_entries,
            "sequence": seq,
            "seats": q.seats,
            "stack_bb": q.stack_bb,
            "model": {"loaded": loaded, "checkpoint": nlh_ckpt_name},
        }

        node = _compute_node(q, k)
        if not isinstance(node, _NodeResult):
            out.update(node)  # terminal / awaiting + board
            return out

        reach_full = _actor_reach(q, k, node.actor)
        reach = reach_full[node.live_idx]
        gp = node.gate_probs.astype(np.float64)
        spec = model.anchor_spec
        allin_k = spec.count - 1 if spec.allin_atom else None
        raise_p = gp[:, GATE_RAISE]
        if allin_k is not None:
            allin_p = raise_p * node.anchor_probs[:, allin_k].astype(np.float64)
        else:
            allin_p = np.zeros_like(raise_p)

        # --- per-combo payload -------------------------------------------
        combos_out: dict[str, Any] = {}
        for row, pos_idx in enumerate(node.live_idx):
            lo, hi = ALL_COMBOS[pos_idx]
            combos_out[combo_name(lo, hi)] = {
                "cell": cell_key(lo, hi),
                "f": round(float(gp[row, GATE_FOLD]), 4),
                "c": round(float(gp[row, GATE_CHECK_CALL]), 4),
                "r": round(float(raise_p[row]), 4),
                "ai": round(float(allin_p[row]), 4),
                "v": round(float(node.values[row]), 3),
                "reach": round(float(reach[row]), 5),
            }

        # --- 169 cells (reach-weighted; dead cells fall back to plain mean)
        cells: dict[str, Any] = {}
        cell_rows: dict[str, list[int]] = {}
        for row, pos_idx in enumerate(node.live_idx):
            lo, hi = ALL_COMBOS[pos_idx]
            cell_rows.setdefault(cell_key(lo, hi), []).append(row)
        for key_, rows in cell_rows.items():
            rows_a = np.asarray(rows, dtype=np.int64)
            w = reach[rows_a]
            wsum = float(w.sum())
            dead = wsum <= 1e-12
            weights = (
                np.full(len(rows_a), 1.0 / len(rows_a)) if dead else w / wsum
            )
            cells[key_] = {
                "f": round(float(gp[rows_a, GATE_FOLD] @ weights), 4),
                "c": round(float(gp[rows_a, GATE_CHECK_CALL] @ weights), 4),
                "r": round(float(raise_p[rows_a] @ weights), 4),
                "ai": round(float(allin_p[rows_a] @ weights), 4),
                "v": round(float(node.values[rows_a] @ weights), 3),
                "reach": round(wsum, 4),
                "live": int(len(rows_a)),
                "dead": bool(dead),
            }

        # --- action summary (reach-weighted combo counts) ------------------
        total = float(reach.sum())
        def _sum(p: np.ndarray) -> float:
            return float((reach * p).sum())
        summary = {}
        for name_, p in (
            ("fold", gp[:, GATE_FOLD]),
            ("check_call", gp[:, GATE_CHECK_CALL]),
            ("raise", raise_p - allin_p),
            ("allin", allin_p),
        ):
            s = _sum(p)
            summary[name_] = {
                "freq": round(s / total, 4) if total > 0 else 0.0,
                "combos": round(s, 2),
            }
        summary["total_combos"] = round(total, 2)

        # --- aggregate raise-size histogram --------------------------------
        sizes = []
        if node.state_meta["legal"]["raise"]:
            w_raise = reach * raise_p
            denom = float(w_raise.sum())
            if denom > 1e-12:
                agg = (
                    w_raise[:, None] * node.anchor_probs.astype(np.float64)
                ).sum(axis=0) / denom
                for a in node.state_meta["anchors"]:
                    sizes.append({**a, "frac": round(float(agg[a["k"]]), 4)})

        out.update({
            "state": node.state_meta,
            "terminal": False,
            "awaiting": None,
            "cells": cells,
            "combos": combos_out,
            "summary": summary,
            "sizes": sizes,
        })
        return out

    return router
