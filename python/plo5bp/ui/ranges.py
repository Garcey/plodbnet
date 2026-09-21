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
- **Raise amounts on the wire are raise-BY deltas**: ``chips_bb`` is the
  engine's `apply_raise_chips` quantity — the chips the actor ADDS now
  (call portion included), NOT the street total. Every amount the UI shows
  or accepts is a raise-TO total (= delta + the actor's street commit); the
  payload carries both (``chips_bb`` + ``to_bb``, ``min/max_raise_bb`` +
  ``min/max_raise_to_bb``, ``actor_commit_bb``) and the client subtracts
  the commit before sending. (review 2026-09-20 H5: the "Raise to" box used
  to send the typed total as a delta — BB typing 4 raised to 5.)
- **All-in is any legal anchor whose chips equal ``max_raise``**, not just
  the ladder's ALL-IN atom: when a fraction anchor clamps to the stack the
  atom is deduped as illegal and the jam mass sits on that fraction anchor
  (always the case at low SPR). It is labelled ALL-IN and counted in
  ``ai`` / ``summary.allin``. (review 2026-09-20 H5)
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
- **Which network serves a node** (review 2026-09-20, GTO serve==train).
  With a GTO teacher configured (`gto_host`), each node is served by the
  teacher ONLY if `host.supports(seats=, street=)` says its training
  coverage includes that table shape + street (preflop never is); every
  other node uses the PPO NLH model and is tagged ``"ppo_fallback"`` — the
  same per-node rule as the trainer, so a river-only teacher never paints a
  preflop grid. A teacher trained on the solver's CANONICAL obs
  (`host.serves_canonical_obs`) gets the canonical form of the packed rows
  on postflop nodes (`canonical_range_pack` — the batched twin of
  `gto.obs_from_label.canonical_serve_raw`, pinned bit-exact against
  `canonical_serve_obs`), and its sizes follow the host's serving rule: grid
  chips, anything within `JAM_DUST_BB` of the stack is the jam. A bare
  `gto_model` (no host => no coverage / obs-form metadata) is NOT served:
  guessing either would reintroduce the train/serve mismatch, so Ranges
  stays on the PPO model and says so. `model.backend` / `model.reason` /
  `model.obs_form` in the payload report what served the viewed node.

The public build (`PLO5BP_PUBLIC=1`) strips every `/ranges` route in
server.py — same mechanism as `/ocr` and `/pokernow` — until the feature
is validated locally and deliberately shipped.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.config import VARIANT_NLH, GameConfig
from plo5bp.encoding_nlh import encode_observation_batch_nlh
from plo5bp.env import BombPotEnv
from plo5bp.gto.obs_from_label import cfr_root_config
from plo5bp.gto.policy_host import JAM_DUST_BB
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
    #: Raise-BY delta in bb (engine chips the actor adds), NOT the raise-to
    #: total — see the module docstring. Finiteness is checked in `_replay`
    #: so a NaN/inf answers 400 like every other bad line entry (a pydantic
    #: `allow_inf_nan=False` would answer 422 with a non-string detail).
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
            actor = int(info.actor)
            commit = int(info.raw_obs["street_commit"][actor])
            if gate == GATE_RAISE:
                if e.chips_bb is None:
                    raise _bad(i, "raise needs chips_bb")
                chips_f = float(e.chips_bb) * cfg.bb
                # NaN / ±inf (JSON accepts the literals) used to reach
                # int(round(...)) and 500. (review 2026-09-20 H5)
                if not math.isfinite(chips_f):
                    raise _bad(i, "chips_bb must be a finite number")
                chips = int(round(chips_f))
                if not (info.min_raise_chips <= chips <= info.max_raise_chips):
                    # Report in raise-TO totals — the unit the UI shows.
                    raise _bad(
                        i,
                        f"raise to {(chips + commit) / cfg.bb:g}bb outside "
                        f"[{(info.min_raise_chips + commit) / cfg.bb:g}, "
                        f"{(info.max_raise_chips + commit) / cfg.bb:g}]bb",
                    )
            street_idx = int(info.raw_obs.get("street", 0))
            obs, _r, done, info = env.step_hybrid(gate, chips)
            seq.append({
                "i": i,
                "t": "a",
                "seat": actor,
                "position": pos(actor),
                "street": STREET_NAMES.get(street_idx, "?"),
                "gate": e.gate,
                # raise-BY delta (wire unit) + raise-TO total (display unit)
                "chips_bb": (
                    round(chips / cfg.bb, 4) if gate == GATE_RAISE else None
                ),
                "to_bb": (
                    round((chips + commit) / cfg.bb, 4)
                    if gate == GATE_RAISE else None
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


BACKEND_PPO = "ppo"
BACKEND_PPO_FALLBACK = "ppo_fallback"  # a GTO teacher is configured but does not serve this node


def canonical_range_pack(
    pack: Mapping[str, np.ndarray], bb: int
) -> tuple[dict[str, np.ndarray], GameConfig]:
    """Canonical (solver-label) form of a `pack_range_nlh` result.

    The batched twin of `gto.obs_from_label.canonical_serve_raw` — a GTO
    PolicyNet is trained on the solver's synthetic street root, not on a
    played hand's obs, so a postflop node must be re-encoded before the
    forward (review 2026-09-20 D3):

    - history: CURRENT-street records only (the root has no past), re-packed
      oldest-first from slot 0;
    - ``total_commit := street_commit`` (chips since the root);
    - ``sb_seat`` / ``bb_seat`` := -1 (blind flags off);
    - ``eff_stack_cap`` recomputed from the STREET-START stacks, every seat
      in, and encoded under the root config (those stacks, ante 0).

    Everything else — and everything that varies per candidate hole
    (``hero_hole``, ``hero_cat_a``, ``nlh_opp_outcome``) — is untouched. All
    rows of a range pack share one public state, so the edits are computed
    once from row 0 and tiled. Returns ``(pack, root_config)``; pinned
    bit-exact against `canonical_serve_obs` by
    tests/python/test_review_trainer_ranges.py.
    """
    m = int(np.asarray(pack["actor"]).shape[0])
    street = int(pack["street"][0])
    stacks = np.asarray(pack["stacks"][0], dtype=np.int64)
    street_commit = np.asarray(pack["street_commit"][0], dtype=np.int64)
    start = [int(x) for x in (stacks + street_commit)]
    n = len(start)

    out = dict(pack)
    hist_len = int(pack["history_len"][0])
    keep = [
        k for k in range(hist_len) if int(pack["history_street"][0][k]) == street
    ]
    pad = {"history_seat": -1, "history_action": -1, "history_chips": 0,
           "history_street": -1}
    for key, fill in pad.items():
        src_row = np.asarray(pack[key][0])
        row = np.full_like(src_row, fill)
        row[: len(keep)] = src_row[keep]
        out[key] = np.broadcast_to(row, (m,) + row.shape).copy()
    out["history_len"] = np.full_like(np.asarray(pack["history_len"]), len(keep))
    out["total_commit"] = np.asarray(pack["street_commit"]).astype(
        np.asarray(pack["total_commit"]).dtype
    )
    out["sb_seat"] = np.full_like(np.asarray(pack["sb_seat"]), -1)
    out["bb_seat"] = np.full_like(np.asarray(pack["bb_seat"]), -1)
    cap_row = np.array(
        [
            min(start[i], max((start[j] for j in range(n) if j != i), default=start[i]))
            for i in range(n)
        ],
        dtype=np.asarray(pack["eff_stack_cap"]).dtype,
    )
    out["eff_stack_cap"] = np.broadcast_to(cap_row, (m, n)).copy()
    return out, cfr_root_config(start, int(bb))


class _NodeResult:
    """Model outputs for one node, over its live combos."""

    __slots__ = (
        "actor", "live_idx", "gate_probs", "anchor_probs", "values",
        "board", "state_meta", "allin_ks", "served",
    )

    def __init__(self, actor, live_idx, gate_probs, anchor_probs, values,
                 board, state_meta, allin_ks=(), served=None):
        self.actor = actor
        self.live_idx = live_idx          # (M,) indices into ALL_COMBOS
        self.gate_probs = gate_probs      # (M, 3) float32
        self.anchor_probs = anchor_probs  # (M, A) float32 (zeros if no raise)
        self.values = values              # (M,) float32, bb
        self.board = board                # list[int]
        self.state_meta = state_meta      # dict for the client
        self.allin_ks = tuple(allin_ks)   # legal anchors with chips == max_raise
        self.served = dict(served or {})  # which network served the node (payload)


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
    gto_host: Any | None = None,
) -> APIRouter:
    """Ranges grid over the NLH model(s).

    ``gto_host`` — the GTO teacher's StrategyBackend (`PolicyNetHost`): its
    network serves exactly the nodes `gto_host.supports(...)` covers, on the
    obs form it was trained on; all other nodes use the PPO NLH model (see
    the module docstring). ``gto_model`` alone (the legacy call) carries no
    coverage / obs-form metadata and is therefore NOT served — the payload
    says so (`model.backend == "ppo_fallback"` + `model.reason`)."""
    router = APIRouter()
    cache = _NodeCache()
    host_name = str(getattr(gto_host, "name", "gto")) if gto_host is not None else None

    def _served(seats: int, street: int) -> dict[str, Any]:
        """Network + obs form for a node with this table shape / street."""
        entry = formats[VARIANT_NLH]
        ppo: dict[str, Any] = {
            "model": entry["model"],
            "loaded": bool(entry.get("loaded")),
            "checkpoint": nlh_ckpt_name,
            "backend": BACKEND_PPO,
            "reason": None,
            "obs_form": "live",
            "gto": False,
        }
        if gto_host is None:
            if gto_model is not None:
                ppo["backend"] = BACKEND_PPO_FALLBACK
                ppo["reason"] = (
                    "a GTO model is configured without its host (no training "
                    "coverage / obs-form metadata): the PPO NLH model is served"
                )
            return ppo
        supports = getattr(gto_host, "supports", None)
        ok = True
        if callable(supports):
            try:
                ok = bool(supports(seats=int(seats), street=int(street)))
            except Exception:  # noqa: BLE001 — a broken host must not 500 the grid
                logger.exception("%s.supports() failed — serving the PPO model", host_name)
                ok = False
        if not ok:
            ppo["backend"] = BACKEND_PPO_FALLBACK
            ppo["reason"] = (
                f"{host_name} was not trained on {int(seats)}-handed "
                f"{STREET_NAMES.get(int(street), '?')} nodes: the PPO NLH "
                "model is served"
            )
            return ppo
        ckpt = getattr(gto_host, "ckpt_path", None)
        canonical = bool(getattr(gto_host, "serves_canonical_obs", False))
        return {
            "model": gto_host.model,
            "loaded": True,
            "checkpoint": Path(str(ckpt)).name if ckpt else host_name,
            "backend": host_name,
            "reason": None,
            # Preflop labels have no exact serve-side twin: live obs there.
            "obs_form": "canonical" if (canonical and int(street) >= 1) else "live",
            "gto": True,
        }

    def _line_key(q: RangeQuery, k: int) -> str:
        return json.dumps(
            [
                id(formats[VARIANT_NLH]["model"]), id(gto_host), id(gto_model),
                q.seats, q.stack_bb,
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

        street_idx = int(info.raw_obs.get("street", 0))
        served = _served(q.seats, street_idx)
        model = served["model"]
        spec = model.anchor_spec
        pack = env.pack_range_nlh(holes)
        if served["obs_form"] == "canonical":
            # The teacher's TRAINING obs form, not the played hand's.
            c_pack, c_cfg = canonical_range_pack(pack, cfg.bb)
            obs = encode_observation_batch_nlh(c_pack, c_pack["hero_cat_a"], c_cfg)
        else:
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
        max_raise = int(info.max_raise_chips)
        min_raise = int(info.min_raise_chips)
        chips_eff = [int(c) for c in grid.chips]
        if served["gto"] and max_raise > 0:
            # The GTO host serves GRID chips and treats anything within
            # JAM_DUST_BB of the stack as the jam (PolicyNetHost D5) — mirror
            # it so the grid's sizes are the ones the trainer host plays.
            dust = max(1, int(JAM_DUST_BB * cfg.bb))
            chips_eff = [
                max_raise if c >= max_raise - dust else c for c in chips_eff
            ]
        legal_ks = [
            int(j) for j in range(spec.count)
            if raise_legal and bool(grid.legal[j])
        ]
        # (review 2026-09-20 H5) ANY legal anchor whose chips reach max_raise
        # is an all-in. Legality dedupes equal chips, so when a fraction
        # anchor clamps to the stack IT is the legal one and the ALL-IN atom
        # is not — keying all-in on the atom's index reported 0% jams exactly
        # where jams matter (low SPR). On a PPO node at most one legal anchor
        # qualifies; the GTO jam snap can add a second, merged below.
        allin_ks = tuple(j for j in legal_ks if chips_eff[j] == max_raise)
        anchors = []
        for j in legal_ks:
            is_allin = j in allin_ks
            if is_allin and j != allin_ks[-1]:
                continue  # merged into the single ALL-IN entry
            anchors.append({
                "k": j,
                # Every ladder index this button stands for (its size-histogram
                # share is their sum).
                "ks": list(allin_ks) if is_allin else [j],
                "label": "ALL-IN" if is_allin else anchor_label_spec(spec, j),
                "allin": is_allin,
                # raise-BY delta (what a line entry carries) ...
                "chips_bb": round(chips_eff[j] / cfg.bb, 4),
                # ... and the raise-TO total the UI displays.
                "to_bb": round((chips_eff[j] + hero_commit) / cfg.bb, 4),
            })
        # Short-shove (engine zeroes min_raise, only the jam is legal):
        # collapse the displayed range to the single all-in point.
        short_shove = raise_legal and min_raise == 0
        min_to = (max_raise if short_shove else min_raise) + hero_commit
        state_meta = {
            "actor": actor,
            "position": position_name(actor, 0, cfg.num_seats),
            "street": STREET_NAMES.get(street_idx, "?"),
            "board": board,
            # Network that produced this node's grid ("ppo", the GTO host's
            # name, or "ppo_fallback" when the teacher does not cover it).
            "backend": served["backend"],
            "pot_bb": round(int(info.raw_obs["pot"]) / cfg.bb, 4),
            "to_call_bb": round(to_call / cfg.bb, 4),
            "legal": {
                "fold": bool(info.gate_mask[GATE_FOLD]),
                "check_call": bool(info.gate_mask[GATE_CHECK_CALL]),
                "raise": raise_legal,
            },
            # Engine raise-BY deltas (the unit of a line entry's chips_bb).
            "min_raise_bb": round(min_raise / cfg.bb, 4),
            "max_raise_bb": round(max_raise / cfg.bb, 4),
            # Raise-TO totals (display/input unit) = delta + street commit.
            "actor_commit_bb": round(hero_commit / cfg.bb, 4),
            "min_raise_to_bb": round(min_to / cfg.bb, 4),
            "max_raise_to_bb": round((max_raise + hero_commit) / cfg.bb, 4),
            "anchors": anchors,
            "allin_k": allin_ks[-1] if allin_ks else None,
        }
        result = _NodeResult(
            actor, live_idx, gate_probs, anchor_probs, values, board,
            state_meta, allin_ks,
            served={k_: v for k_, v in served.items() if k_ not in ("model", "gto")},
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

        out: dict[str, Any] = {
            "node": k,
            "num_entries": n_entries,
            "sequence": seq,
            "seats": q.seats,
            "stack_bb": q.stack_bb,
        }

        node = _compute_node(q, k)
        if not isinstance(node, _NodeResult):
            # No decision here, so no network served it: report the default.
            entry = formats[VARIANT_NLH]
            out["model"] = {
                "loaded": bool(entry.get("loaded")), "checkpoint": nlh_ckpt_name,
                "backend": None, "reason": None, "obs_form": None,
            }
            out.update(node)  # terminal / awaiting + board
            return out
        # What served the VIEWED node (earlier nodes in the reach product may
        # have been served by the other network — same rule, per node).
        out["model"] = dict(node.served)

        reach_full = _actor_reach(q, k, node.actor)
        reach = reach_full[node.live_idx]
        gp = node.gate_probs.astype(np.float64)
        raise_p = gp[:, GATE_RAISE]
        # All-in share of the raise mass: every legal anchor at max_raise
        # (the ALL-IN atom OR a fraction anchor clamped to the stack — H5).
        allin_p = np.zeros_like(raise_p)
        for allin_k in node.allin_ks:
            allin_p = allin_p + (
                raise_p * node.anchor_probs[:, allin_k].astype(np.float64)
            )

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
                    share = float(sum(agg[j] for j in a["ks"]))
                    sizes.append({**a, "frac": round(share, 4)})

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
