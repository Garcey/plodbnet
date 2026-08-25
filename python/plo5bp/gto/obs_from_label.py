"""Synthesize NLH obs vectors from LabelRecords for supervised training.

v2 dumps carry path labels + pot/to_call/stacks. When the label has a
solver-root (``root_pot_chips`` / ``root_stacks_chips``) we rebuild a
live ``GameState`` via ``reset_nlh_cfr_node`` and encode that node's
``observation_dict`` — bit-exact vs the Trainer/Study encode path.

Fallback (legacy labels, preflop, reconstruct fail): a synthetic raw_obs
with history filled from path tokens. ``last_aggressor`` is ``-1`` (never
``None``) so encode does not throw.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from plo5bp.actions import ALL_IN, BET_PCT_100, CHECK_CALL, FOLD
from plo5bp.config import VARIANT_NLH, GameConfig
from plo5bp.encoding_nlh import OBS_DIM_NLH, encode_observation_nlh
from plo5bp.gto.labels import LabelRecord
from plo5bp.gto.roots import CLUBGG_NLH_ROOT


def path_tokens_from_label(lab: LabelRecord) -> list[str]:
    notes = lab.notes or {}
    raw = notes.get("path_tokens")
    if isinstance(raw, list) and raw:
        return [str(x) for x in raw if str(x) and str(x) != "open"]
    p = notes.get("path")
    if isinstance(p, list):
        return [str(x) for x in p if str(x) and str(x) != "open"]
    s = str(p or "")
    if not s or s == "open":
        return []
    return [t for t in s.split(",") if t and t != "open"]


def _bb(lab: LabelRecord) -> int:
    return int((lab.notes or {}).get("bb_chips") or CLUBGG_NLH_ROOT.bb)


def _root_stacks(lab: LabelRecord) -> list[int]:
    notes = lab.notes or {}
    raw = notes.get("root_stacks_chips")
    if raw:
        return [int(x) for x in raw]
    return [int(x) for x in lab.stacks_chips]


def _root_pot(lab: LabelRecord) -> int:
    notes = lab.notes or {}
    rp = notes.get("root_pot_chips")
    if rp:
        return int(rp)
    if not path_tokens_from_label(lab):
        return int(lab.pot_chips)
    return 0


def live_config_from_label(lab: LabelRecord) -> GameConfig:
    """Config matching ``reset_nlh_cfr_node`` (ante already in the pot)."""
    n = max(2, int(lab.num_seats))
    bb = _bb(lab)
    stacks = _root_stacks(lab)
    while len(stacks) < n:
        stacks.append(stacks[-1] if stacks else bb * 100)
    stacks = stacks[:n]
    return GameConfig(
        num_seats=n,
        starting_stack=int(stacks[0]),
        starting_stacks=tuple(int(s) for s in stacks),
        ante=0,
        bb=bb,
        sb=max(bb // 2, 1),
        variant=VARIANT_NLH,
    )


def reconstruct_live_engine(lab: LabelRecord):
    """Return a ``GameState`` at the dump node, or None if not reconstructible.

    Postflop only (street 1..3). Requires a 2-card hole and a board.
    """
    street = int(lab.street)
    if street < 1 or street > 3:
        return None
    hole = list(lab.hero_hole or [])
    if len(hole) < 2:
        return None
    need = {1: 3, 2: 4, 3: 5}[street]
    board = [int(c) for c in lab.board]
    if len(board) < need:
        return None
    board = board[:need]
    tokens = path_tokens_from_label(lab)
    root_pot = _root_pot(lab)
    if root_pot <= 0:
        return None
    n = max(2, int(lab.num_seats))
    stacks = _root_stacks(lab)
    while len(stacks) < n:
        stacks.append(stacks[-1])
    stacks = [int(x) for x in stacks[:n]]
    if any(s <= 0 for s in stacks):
        return None
    bb = _bb(lab)
    hero = int(lab.hero_seat) % n
    try:
        from plo5bp._engine import GameState as _RustGameState  # type: ignore
    except Exception:
        return None
    try:
        gs = _RustGameState(
            num_seats=n,
            starting_stack=int(stacks[0]),
            ante=0,
            bb=bb,
            starting_stacks=np.asarray(stacks, dtype=np.uint64),
            variant="nlh_single",
            sb=max(bb // 2, 1),
        )
        gs.reset_nlh_cfr_node(
            int(root_pot),
            stacks,
            board,
            street,
            hero,
            [int(hole[0]), int(hole[1])],
            tokens,
            bb,
        )
    except Exception:
        return None
    if gs.current_actor() is None:
        return None
    return gs


def encode_live_engine(gs, lab: LabelRecord) -> np.ndarray:
    """Same pack as ``BombPotEnv._pack_obs`` (category + encode_observation_nlh)."""
    raw = dict(gs.observation_dict())
    actor = raw.get("actor")
    if actor is not None:
        raw["hero_category_a"] = int(gs.hero_category(int(actor), 0))
        raw["hero_category_b"] = int(gs.hero_category(int(actor), 1))
    return encode_observation_nlh(raw, live_config_from_label(lab))


def _raw_obs_from_label(lab: LabelRecord) -> dict:
    """Synthetic observation_dict when a live engine node cannot be built."""
    n = max(2, int(lab.num_seats))
    hero = int(lab.hero_seat) % n
    stacks = list(lab.stacks_chips)
    while len(stacks) < n:
        stacks.append(stacks[-1] if stacks else CLUBGG_NLH_ROOT.bb * 100)
    stacks = stacks[:n]
    pot = int(lab.pot_chips)
    to_call = max(0, int(lab.to_call_chips))
    street_commit = [0] * n
    last_aggressor = -1
    if to_call > 0:
        vill = (hero - 1) % n
        if vill == hero:
            vill = (hero + 1) % n
        street_commit[vill] = to_call
        last_aggressor = vill
    settled = max(0, pot - to_call)
    half = settled // n
    total_commit = [half + street_commit[i] for i in range(n)]
    folded = [False] * n
    tokens = path_tokens_from_label(lab)
    if tokens and int(lab.street) == 0:
        seat_i = 0
        for t in tokens:
            if seat_i >= n:
                break
            if seat_i == hero:
                seat_i += 1
                if seat_i >= n:
                    break
            if t.upper() in ("F", "FOLD") and seat_i != hero:
                folded[seat_i] = True
            seat_i += 1
    all_in = [s <= 0 for s in stacks]
    board = list(lab.board)
    hole = list(lab.hero_hole) if lab.hero_hole else []
    if len(hole) < 2:
        from plo5bp.gto.preflop_class import representative_hole

        cid = (lab.notes or {}).get("class_id")
        if cid is not None:
            hole = representative_hole(int(cid), blocked=board)
        else:
            hole = [0, 1]
    hole = (list(hole) + [0, 1])[:2]
    min_r = max(0, int(lab.min_raise_chips))
    max_r = max(0, int(lab.max_raise_chips))
    min_bet = to_call + min_r if to_call > 0 else min_r
    max_bet = to_call + max_r if max_r > 0 else to_call
    if n >= 3:
        sb_seat, bb_seat = n - 2, n - 1
    else:
        sb_seat, bb_seat = 1 % n, 0

    history = _synthetic_history(lab, tokens, n, pot, stacks)

    return {
        "actor": hero,
        "hero_hole": hole,
        "board_a": board,
        "board_b": [],
        "street": int(lab.street),
        "folded": folded,
        "all_in": all_in,
        "stacks": stacks,
        "eff_stack_cap": list(stacks),
        "pot": pot,
        "bet_to_call": to_call,
        "min_bet": min_bet,
        "max_bet": max_bet,
        "min_raise": min_r,
        "max_raise": max_r,
        "history": history,
        "button": int(lab.button) % n,
        "street_commit": street_commit,
        "total_commit": total_commit,
        "last_aggressor": last_aggressor,
        "sb_seat": sb_seat,
        "bb_seat": bb_seat,
        "hero_category_a": 0,
        "hero_category_b": 0,
        "nlh_opp_outcome": [0.0, 0.0, 0.0],
        "opp_outcome_fractions": [0.0, 0.0, 0.0],
        "acted_this_street": [False] * n,
        "awaiting_next_street": None,
        "study_terminal": None,
        "share_bounds": [],
        "per_board_outcome": [],
        "hero_board_v3": [],
        "board_draw_v3": [],
    }


def _synthetic_history(
    lab: LabelRecord,
    tokens: Sequence[str],
    n: int,
    pot_now: int,
    stacks: Sequence[int],
) -> list[tuple[int, int, int, int]]:
    """Best-effort (seat, action, chips, street) from path labels."""
    if not tokens:
        return []
    street = int(lab.street)
    bb = _bb(lab)
    pot = int(_root_pot(lab) or pot_now)
    remain = [int(s) for s in (_root_stacks(lab) or stacks)]
    while len(remain) < n:
        remain.append(remain[-1] if remain else bb * 100)
    remain = remain[:n]
    street_commit = [0] * n
    btc = 0
    # CFR HU postflop: seat 0 first. Multiway preflop: seat 0 first.
    actor = 0
    hist: list[tuple[int, int, int, int]] = []
    for tok in tokens:
        up = str(tok).upper()
        chips = 0
        action = CHECK_CALL
        if up in ("F", "FOLD"):
            action = FOLD
        elif up in ("XC", "CHECK_CALL", "CHECK", "CALL"):
            need = max(0, btc - street_commit[actor])
            chips = min(need, remain[actor])
            action = CHECK_CALL
            remain[actor] -= chips
            pot += chips
            street_commit[actor] += chips
        elif up in ("AI", "ALLIN"):
            chips = remain[actor]
            action = ALL_IN
            remain[actor] = 0
            pot += chips
            street_commit[actor] += chips
            if street_commit[actor] > btc:
                btc = street_commit[actor]
        else:
            pm = None
            if up.startswith("RAISE_"):
                try:
                    pm = int(up.split("_", 1)[1])
                except ValueError:
                    pm = None
            elif up.startswith("R") and up[1:].isdigit():
                pm = int(up[1:])
            if pm is None:
                continue
            to_call_raw = max(0, btc - street_commit[actor])
            if btc == 0:
                raise_over = pot * pm // 1000
                target = raise_over
            else:
                raise_over = (pot + to_call_raw) * pm // 1000
                target = btc + raise_over
            chips = max(0, target - street_commit[actor])
            chips = min(chips, remain[actor])
            if chips <= 0:
                continue
            action = BET_PCT_100
            remain[actor] -= chips
            pot += chips
            street_commit[actor] += chips
            if street_commit[actor] > btc:
                btc = street_commit[actor]
        hist.append((actor, action, int(chips), street))
        # next alive seat
        nxt = (actor + 1) % n
        actor = nxt
    return hist


def game_config_from_label(lab: LabelRecord) -> GameConfig:
    """Fallback config (ClubGG ante) for synthetic encode."""
    notes = lab.notes or {}
    if notes.get("root_pot_chips") and int(lab.street) in (1, 2, 3):
        return live_config_from_label(lab)
    stacks = tuple(int(s) for s in lab.stacks_chips) or (
        CLUBGG_NLH_ROOT.bb * 100,
        CLUBGG_NLH_ROOT.bb * 100,
    )
    n = max(2, int(lab.num_seats))
    if len(stacks) < n:
        stacks = stacks + (stacks[-1],) * (n - len(stacks))
    return CLUBGG_NLH_ROOT.game_config(
        num_seats=n,
        starting_stacks_bb=[s / CLUBGG_NLH_ROOT.bb for s in stacks[:n]],
    )


def obs_from_label(lab: LabelRecord) -> np.ndarray | None:
    """Return (OBS_DIM_NLH,) float32 or None if label is aggregate (no hole)."""
    if not lab.hero_hole or len(lab.hero_hole) < 2:
        return None
    if lab.notes.get("aggregate"):
        return None
    gs = reconstruct_live_engine(lab)
    if gs is not None:
        try:
            return encode_live_engine(gs, lab)
        except Exception:
            pass
    raw = _raw_obs_from_label(lab)
    cfg = game_config_from_label(lab)
    try:
        return encode_observation_nlh(raw, cfg)
    except Exception:
        out = np.zeros(OBS_DIM_NLH, dtype=np.float32)
        for c in lab.hero_hole[:2]:
            if 0 <= int(c) < 52:
                out[int(c)] = 1.0
        for c in lab.board:
            if 0 <= int(c) < 52:
                out[52 + int(c)] = 1.0
        st = int(lab.street)
        if 0 <= st < 4:
            out[104 + st] = 1.0
        inv = 1.0 / float(CLUBGG_NLH_ROOT.bb)
        out[132] = lab.pot_chips * inv
        out[133] = lab.to_call_chips * inv
        return out


def labels_to_supervised_rows(labels: Sequence[LabelRecord]):
    """LabelRecords with holes → SupervisedRows (skips aggregates)."""
    from plo5bp.gto.dataset import SupervisedRow
    from plo5bp.sizing import NLH_ANCHOR_SPEC

    k = NLH_ANCHOR_SPEC.count
    rows = []
    for lab in labels:
        obs = obs_from_label(lab)
        if obs is None:
            continue
        fold_legal = int(lab.to_call_chips) > 0
        raise_legal = int(lab.max_raise_chips) > 0
        gm = np.array([fold_legal, True, raise_legal], dtype=bool)
        g = np.asarray(lab.gate_probs, dtype=np.float32).copy()
        if len(g) != 3:
            g = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        g = g * gm.astype(np.float32)
        if float(g.sum()) <= 0:
            g = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            g /= g.sum()
        ap = np.zeros(k, dtype=np.float32)
        for a in lab.action_probs:
            if a.gate == "raise" and a.anchor_k is not None:
                if 0 <= int(a.anchor_k) < k:
                    ap[int(a.anchor_k)] += float(a.prob)
        s = float(ap.sum())
        if s > 0:
            ap /= s
        elif raise_legal:
            ap[-1] = 1.0
        rows.append(
            SupervisedRow(
                obs=obs.astype(np.float32),
                gate_mask=gm,
                sizing=np.array(
                    [
                        lab.min_raise_chips,
                        lab.max_raise_chips,
                        lab.pot_chips,
                        lab.to_call_chips,
                    ],
                    dtype=np.int64,
                ),
                gate_probs=g,
                anchor_probs=ap,
                value_bb=float(lab.value_bb or 0.0),
                street=int(lab.street),
            )
        )
    return rows
