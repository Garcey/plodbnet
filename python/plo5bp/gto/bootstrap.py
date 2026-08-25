"""Rule-based pure-node bootstrap for PolicyNet (no PPO teacher).

Generates supervised rows by rolling NLH hands and labeling each
decision with near-pure strategies that any correct GTO net must get
right (trash folds, nuts jams, free checks). Soft labels are slightly
mixed (ε floor) so the student learns a valid simplex.

This is a curriculum prior only. Production GTO labels come from
native rust_cfr export (``scripts/cfr_export_labels.py``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.env import BombPotEnv, StepInfo
from plo5bp.gto.dataset import SupervisedRow
from plo5bp.gto.roots import CLUBGG_NLH_ROOT
from plo5bp.sizing import NLH_ANCHOR_SPEC, sizing_from_info

# Engine hand categories (mirrors Rust / env hero_category): higher = stronger
# 0 high card … 8 straight flush (typical poker category order).
_CAT_HIGH = 0
_CAT_PAIR = 1
_CAT_TWO_PAIR = 2
_CAT_TRIPS = 3
_CAT_STRAIGHT = 4
_CAT_FLUSH = 5
_CAT_FULL_HOUSE = 6
_CAT_QUADS = 7
_CAT_SF = 8

_EPS = 0.02  # minimum mass on off-actions (keeps simplex valid)


def _renorm3(f: float, c: float, r: float) -> np.ndarray:
    xs = np.array(
        [max(_EPS, f), max(_EPS, c), max(_EPS, r)], dtype=np.float32
    )
    xs /= xs.sum()
    return xs


def _pure_gate(preferred: int, strength: float = 0.94) -> np.ndarray:
    """Near-pure gate vector; preferred in {0,1,2}."""
    mass = [ _EPS, _EPS, _EPS ]
    mass[preferred] = strength
    s = sum(mass)
    return np.array([m / s for m in mass], dtype=np.float32)


def _allin_anchor_probs(k: int | None = None) -> np.ndarray:
    k = NLH_ANCHOR_SPEC.count if k is None else k
    ap = np.full(k, _EPS / k, dtype=np.float32)
    ap[-1] = 1.0 - _EPS  # ALL-IN atom
    ap /= ap.sum()
    return ap


def _pot_anchor_probs(k: int | None = None) -> np.ndarray:
    """Mass on ~pot (100% = index of 1000 pm in NLH ladder)."""
    k = NLH_ANCHOR_SPEC.count if k is None else k
    fracs = NLH_ANCHOR_SPEC.fracs_pm
    # find closest to 1000 pm among fraction anchors (exclude all-in last)
    best = 0
    best_d = 10**9
    for i, f in enumerate(fracs):
        d = abs(int(f) - 1000)
        if d < best_d:
            best_d = d
            best = i
    ap = np.full(k, _EPS / k, dtype=np.float32)
    ap[best] = 1.0 - _EPS
    ap /= ap.sum()
    return ap


def _min_anchor_probs(k: int | None = None) -> np.ndarray:
    k = NLH_ANCHOR_SPEC.count if k is None else k
    ap = np.full(k, _EPS / k, dtype=np.float32)
    ap[0] = 1.0 - _EPS
    ap /= ap.sum()
    return ap


def label_node(info: StepInfo) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (gate_probs, anchor_probs, value_bb_proxy) for one node.

    Pure-node rules (architecture Phase 0 smoke + bootstrap curriculum):
      - facing bet + air on river → fold
      - free option + air → check
      - nuts-class made hand → jam / pot raise
      - facing tiny bet with pair+ → call
      - default: check/call mass, small raise mix
    """
    raw = info.raw_obs
    gm = info.gate_mask
    to_call = int(info.min_raise_chips)  # not to_call — use pot/stacks
    # bet to call from raw
    actor = int(info.actor) if info.actor is not None else 0
    street_commit = raw.get("street_commit") or [0] * 8
    bet_to = int(raw.get("bet_to_call") or 0)
    my_street = int(street_commit[actor]) if actor < len(street_commit) else 0
    to_call_chips = max(0, bet_to - my_street)
    pot = max(1, int(raw.get("pot") or 1))
    street = int(raw.get("street") or 0)
    cat = int(raw.get("hero_category_a") or 0)
    max_r = int(info.max_raise_chips)
    raise_ok = bool(gm[GATE_RAISE]) and max_r > 0
    fold_ok = bool(gm[GATE_FOLD])
    call_ok = bool(gm[GATE_CHECK_CALL])

    k = NLH_ANCHOR_SPEC.count
    pot_odds = to_call_chips / float(to_call_chips + pot) if to_call_chips > 0 else 0.0

    # --- Pure: trash facing bet on turn/river --------------------------------
    if (
        to_call_chips > 0
        and fold_ok
        and street >= 2
        and cat <= _CAT_HIGH
        and pot_odds > 0.15
    ):
        g = _pure_gate(GATE_FOLD, 0.95)
        # zero illegal
        if not fold_ok:
            g = _pure_gate(GATE_CHECK_CALL if call_ok else GATE_RAISE)
        return g, _min_anchor_probs(k), -float(to_call_chips) / 10000.0

    # --- Pure: nuts-class → jam ---------------------------------------------
    if cat >= _CAT_FULL_HOUSE and raise_ok:
        g = _pure_gate(GATE_RAISE, 0.92)
        if not raise_ok and call_ok:
            g = _pure_gate(GATE_CHECK_CALL, 0.9)
        return g, _allin_anchor_probs(k), float(pot) / 10000.0 * 0.5

    # strong made (straight+) facing no bet → pot bet often
    if cat >= _CAT_STRAIGHT and to_call_chips == 0 and raise_ok and street >= 1:
        g = _pure_gate(GATE_RAISE, 0.75)
        return g, _pot_anchor_probs(k), float(pot) / 10000.0 * 0.3

    # --- Free check when weak -----------------------------------------------
    if to_call_chips == 0 and call_ok and cat <= _CAT_PAIR:
        g = _pure_gate(GATE_CHECK_CALL, 0.88)
        return g, _min_anchor_probs(k), 0.0

    # --- Facing bet with strong made → call / raise mix ---------------------
    if to_call_chips > 0 and cat >= _CAT_TWO_PAIR:
        if raise_ok and cat >= _CAT_TRIPS:
            g = _renorm3(0.02, 0.35, 0.63)
            return g, _pot_anchor_probs(k), float(pot) / 20000.0
        if call_ok:
            g = _pure_gate(GATE_CHECK_CALL, 0.85)
            return g, _min_anchor_probs(k), 0.0

    # --- Default: check/call preferred, small raise mix ---------------------
    if to_call_chips == 0:
        if raise_ok:
            g = _renorm3(0.0 if not fold_ok else _EPS, 0.70, 0.28)
            return g, _min_anchor_probs(k), 0.0
        g = _pure_gate(GATE_CHECK_CALL if call_ok else GATE_FOLD)
        return g, _min_anchor_probs(k), 0.0

    # facing bet, medium: pot-odds style call bias
    if call_ok and pot_odds < 0.35 and cat >= _CAT_PAIR:
        g = _renorm3(0.15 if fold_ok else 0.0, 0.75, 0.10 if raise_ok else 0.0)
        return g, _min_anchor_probs(k), 0.0
    if fold_ok and pot_odds >= 0.35 and cat <= _CAT_PAIR:
        g = _renorm3(0.70, 0.25 if call_ok else 0.0, 0.05 if raise_ok else 0.0)
        return g, _min_anchor_probs(k), -float(to_call_chips) / 20000.0

    # fallback legal-uniform-ish
    mass = np.array(
        [
            0.3 if fold_ok else 0.0,
            0.5 if call_ok else 0.0,
            0.2 if raise_ok else 0.0,
        ],
        dtype=np.float32,
    )
    if mass.sum() <= 0:
        mass = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    mass /= mass.sum()
    return mass, _min_anchor_probs(k), 0.0


def _mask_gate_probs(g: np.ndarray, gm: np.ndarray) -> np.ndarray:
    out = g.astype(np.float32).copy()
    out = out * gm.astype(np.float32)
    s = float(out.sum())
    if s <= 1e-8:
        # first legal
        for i in range(3):
            if gm[i]:
                out[:] = 0.0
                out[i] = 1.0
                return out
        out[1] = 1.0
        return out
    out /= s
    return out


def collect_bootstrap(
    *,
    n_decisions: int = 4096,
    seed: int = 0,
    seats: Sequence[int] = (2, 3, 4, 5, 6),
    stack_bb_range: tuple[float, float] = (20.0, 250.0),
    max_hands: int = 80_000,
    pure_only: bool = False,
) -> list[SupervisedRow]:
    """Roll NLH hands; label every decision with bootstrap rules.

    Advance policy: sample from the bootstrap gate (mixed) so trajectories
    visit both check-down and jam lines.
    """
    rng = np.random.default_rng(int(seed))
    root = CLUBGG_NLH_ROOT
    rows: list[SupervisedRow] = []
    hands = 0
    k = NLH_ANCHOR_SPEC.count

    while len(rows) < int(n_decisions) and hands < int(max_hands):
        hands += 1
        n_seats = int(seats[int(rng.integers(0, len(seats)))])
        stacks_bb = [
            float(rng.uniform(stack_bb_range[0], stack_bb_range[1]))
            for _ in range(n_seats)
        ]
        cfg = root.game_config(num_seats=n_seats, starting_stacks_bb=stacks_bb)
        env = BombPotEnv(cfg)
        button = int(rng.integers(0, n_seats))
        hand_seed = int(rng.integers(0, 2**63 - 1))
        obs, info = env.reset(hand_seed, button)
        steps = 0
        while not info.terminal and steps < 200 and len(rows) < n_decisions:
            steps += 1
            if info.actor is None or not bool(info.gate_mask.any()):
                break
            g, ap, v = label_node(info)
            g = _mask_gate_probs(g, info.gate_mask)
            keep = (not pure_only) or float(g.max()) >= 0.85
            if keep:
                rows.append(
                    SupervisedRow(
                        obs=np.asarray(obs, dtype=np.float32).copy(),
                        gate_mask=np.asarray(info.gate_mask, dtype=bool).copy(),
                        sizing=sizing_from_info(info).astype(np.int64),
                        gate_probs=g,
                        anchor_probs=ap.astype(np.float32),
                        value_bb=float(v),
                        street=int(info.raw_obs.get("street", 0)),
                    )
                )

            # advance: sample gate from soft label
            gate = int(rng.choice(3, p=g / g.sum()))
            chips = 0
            if gate == GATE_RAISE and bool(info.gate_mask[GATE_RAISE]):
                ap_n = ap / max(float(ap.sum()), 1e-8)
                ak = int(rng.choice(k, p=ap_n))
                sizing = sizing_from_info(info)
                from plo5bp.sizing import anchor_grid_np

                grid = anchor_grid_np(
                    int(sizing[0]),
                    int(sizing[1]),
                    int(sizing[2]),
                    int(sizing[3]),
                    NLH_ANCHOR_SPEC,
                )
                chips = int(grid.chips[ak])
                if not bool(grid.legal[ak]):
                    for j in range(k):
                        if grid.legal[j]:
                            chips = int(grid.chips[j])
                            break
            elif gate == GATE_RAISE:
                gate = (
                    GATE_CHECK_CALL
                    if bool(info.gate_mask[GATE_CHECK_CALL])
                    else GATE_FOLD
                )
                chips = 0
            obs, _, done, info = env.step_hybrid(gate, chips)
            if done:
                break

    return rows
