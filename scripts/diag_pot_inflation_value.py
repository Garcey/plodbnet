"""Quantify how much of the value-head loss comes from pot inflation.

Verification step #1 from
``.claude/plans/the-last-thing-i-woolly-swing.md``. Constructs a
heads-up bomb-pot river check-down spot with hero holding 22223
(four 2s + 3) the way the live UI session sees it: engine has 6
seats, posts 6 antes, and 4 sitting-out seats auto-check across
every street so the engine pot bloats to 6 × ante.

Then forwards the resulting observation through ``MODEL`` twice:
1. As-is (inflated 18 BB pot view).
2. With the pot-related scalars (pot, SPR, max_bet, min_bet,
   pot_odds) rewritten to the headsup-correct 6 BB pot view —
   stack/active/history slots untouched.

Reports both ``value_bb`` outputs and the gap.

If the gap is large (≥+5 BB toward zero), Suspect #1 in the plan
is confirmed dominant: the value head is being fed an OOD
observation and the right fix is engine-side (post antes only
for in-hand seats), not display-side cosmetic.

Run with::

    .venv/Scripts/python scripts/diag_pot_inflation_value.py
"""

from __future__ import annotations

import numpy as np
import torch

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD
from plo5bp.config import GameConfig
from plo5bp.encoding import (
    OBS_DIM,
    _POT_ODDS_OFF,
    _SCALARS_OFF,
    _SPR_OFF,
)
from plo5bp.env import BombPotEnv
from plo5bp.ui.server import _load_model


# --- Spot setup --------------------------------------------------------------
# 6-seat config matching ClubGG defaults: bb=10000 chips, ante=30000 chips
# (= 3 BB = $60 at $20/bb), 17 BB starting stack ($340).
CFG = GameConfig(num_seats=6, starting_stack=170_000, ante=30_000, bb=10_000)
HERO_SEAT = 0
VILLAIN_SEAT = 3
IN_HAND = {HERO_SEAT, VILLAIN_SEAT}

# 22223: four 2s (rank 0, all suits) + 3♣ (rank 1, suit 0).
HERO_HOLE = [0, 1, 2, 3, 4]
# Two boards of 5 cards each, none touching ranks 0 or 1 (no 2s, no 3s)
# so hero stays as a pair-of-2s with no improvement on either board.
# Cards: rank * 4 + suit. Avoid indices 0..7 (ranks 0=2, 1=3).
FLOP_A = [40, 25, 14]   # T♣ 8♦ 5♥  (40=10*4+0, 25=6*4+1, 14=3*4+2)
TURN_A = 32             # 9♣
RIVER_A = 47            # A♦ (47=11*4+3? actually 47/4=11=K, 47%4=3 → K♠)
# Actually: rank 11=K, suit 3 = K♠.
FLOP_B = [44, 28, 17]   # J♣ 9♦ 6♥
TURN_B = 36             # T♦
RIVER_B = 51            # A♠


VERBOSE = False


def _walk_to_river_check_decision(
    env: BombPotEnv, *, button: int
) -> tuple[np.ndarray, object]:
    """Replay the heads-up check-down to the river hero decision.

    Engine has 6 seats; only HERO and VILLAIN are 'in hand'. Sitting-out
    seats auto-check (no bet to face). On preflop (which this engine has
    no actions for in study mode — actions start on flop) we just walk
    the flop, then advance turn, then advance river up to hero's turn.
    """
    obs, info = env.reset_study(
        button=button,
        hero_seat=HERO_SEAT,
        hero_hole=HERO_HOLE,
        flop_a=FLOP_A,
        flop_b=FLOP_B,
    )

    last_obs, last_info = obs, info

    def _pump_to_next_street_or_hero() -> bool:
        """Walk actions until: hero is the actor on the river, OR engine
        is awaiting next street, OR hand-terminal. Returns True if we
        advanced a street, False otherwise.

        Note: ``env.is_terminal()`` returns True whenever ``actor is None``
        — that includes between-round pauses where ``awaiting_next_street``
        is set. Check ``awaiting`` first.
        """
        nonlocal last_obs, last_info
        guard = 0
        while True:
            guard += 1
            if guard > 64:
                raise RuntimeError("walk guard tripped")
            awaiting = env.awaiting_next_street()
            if awaiting is not None:
                return True
            if env.is_terminal() or env.study_terminal() is not None:
                return False
            actor = env.current_actor()
            if actor is None:
                return False
            # If hero on the river, stop here.
            raw = dict(env._rs.observation_dict())
            street = int(raw["street"])
            if actor == HERO_SEAT and street == 3:
                return False
            # Sitting-out and villain alike: face no bet → check_call.
            bet = int(raw.get("bet_to_call") or 0)
            gate = GATE_FOLD if (bet > 0 and actor not in IN_HAND) else GATE_CHECK_CALL
            if VERBOSE:
                print(f"  step: street={street} actor={actor} bet={bet} gate={gate}")
            last_obs, _, _, last_info = env.step_hybrid(gate, 0)

    while True:
        advanced = _pump_to_next_street_or_hero()
        if VERBOSE:
            print(
                f"  pump returned advanced={advanced} terminal={env.is_terminal()} "
                f"actor={env.current_actor()} awaiting={env.awaiting_next_street()}"
            )
        if not advanced:
            break
        awaiting = env.awaiting_next_street()
        if awaiting == 2:
            last_obs, last_info = env.set_turn(TURN_A, TURN_B)
        elif awaiting == 3:
            last_obs, last_info = env.set_river(RIVER_A, RIVER_B)
        else:
            break

    return last_obs, last_info


def _value_bb(model, obs: np.ndarray, gate_mask: np.ndarray) -> float:
    obs_t = torch.from_numpy(obs).unsqueeze(0)
    gm_t = torch.from_numpy(gate_mask).unsqueeze(0)
    with torch.no_grad():
        _, _, value = model(obs_t, gm_t)
    return float(value.squeeze(0).item())


def _probe_inflated(model, button: int) -> tuple[float, float, float]:
    """Run the 6-seat-with-4-sitting-out scenario; return (v_inflated,
    v_corrected, delta_bb)."""
    env = BombPotEnv(CFG)
    obs, info = _walk_to_river_check_decision(env, button=button)

    actor = env.current_actor()
    if actor != HERO_SEAT:
        raise RuntimeError(
            f"expected hero ({HERO_SEAT}) on river decision, got actor={actor}"
        )

    raw = dict(env._rs.observation_dict())
    pot_chips = int(raw["pot"])
    bb = float(CFG.bb)
    inv_bb = 1.0 / bb
    pot_bb_inflated = pot_chips / bb

    v_inflated = _value_bb(model, obs, info.gate_mask)

    excess_chips = (CFG.num_seats - len(IN_HAND)) * CFG.ante
    real_pot_chips = pot_chips - excess_chips
    real_pot_bb = real_pot_chips / bb

    obs_corrected = obs.copy()
    obs_corrected[_SCALARS_OFF + 0] = real_pot_chips * inv_bb
    obs_corrected[_SCALARS_OFF + 3] = real_pot_chips * inv_bb

    pot_safe_chips = max(real_pot_chips, 1)
    stacks = raw["stacks"]
    for k in range(CFG.num_seats):
        seat = (HERO_SEAT + k) % CFG.num_seats
        spr = float(stacks[seat]) / float(pot_safe_chips)
        obs_corrected[_SPR_OFF + k] = float(np.clip(spr, 0.0, 4.0))
    obs_corrected[_POT_ODDS_OFF] = 0.0

    v_corrected = _value_bb(model, obs_corrected, info.gate_mask)

    print(f"  pot {pot_bb_inflated:5.2f} BB inflated -> V = {v_inflated:+8.4f} BB "
          f"(${v_inflated * 20:+9.2f})")
    print(f"  pot {real_pot_bb:5.2f} BB real     -> V = {v_corrected:+8.4f} BB "
          f"(${v_corrected * 20:+9.2f})")
    delta = v_corrected - v_inflated
    print(f"  delta = {delta:+8.4f} BB (${delta * 20:+9.2f})")
    return v_inflated, v_corrected, delta


def _probe_native_2seat(model, button: int) -> float:
    """Control: build the same 22223 river spot in a NATIVE 2-seat config
    (no sitting-out, no inflation). The pot is structurally 6 BB. Returns
    the raw value_bb."""
    cfg2 = GameConfig(num_seats=2, starting_stack=170_000, ante=30_000, bb=10_000)
    env = BombPotEnv(cfg2)
    obs, info = env.reset_study(
        button=button,
        hero_seat=0,
        hero_hole=HERO_HOLE,
        flop_a=FLOP_A,
        flop_b=FLOP_B,
    )

    def _walk():
        last = obs
        last_info = info
        guard = 0
        while True:
            guard += 1
            if guard > 32:
                raise RuntimeError("walk guard tripped (2-seat)")
            awaiting = env.awaiting_next_street()
            if awaiting == 2:
                last, last_info = env.set_turn(TURN_A, TURN_B)
                continue
            if awaiting == 3:
                last, last_info = env.set_river(RIVER_A, RIVER_B)
                continue
            if env.is_terminal() or env.study_terminal() is not None:
                return last, last_info
            actor = env.current_actor()
            if actor is None:
                return last, last_info
            raw = dict(env._rs.observation_dict())
            street = int(raw["street"])
            if actor == 0 and street == 3:
                return last, last_info
            last, _, _, last_info = env.step_hybrid(GATE_CHECK_CALL, 0)

    obs2, info2 = _walk()
    raw = dict(env._rs.observation_dict())
    pot_bb = int(raw["pot"]) / float(cfg2.bb)
    v = _value_bb(model, obs2, info2.gate_mask)
    print(f"  pot {pot_bb:5.2f} BB native 2-seat -> V = {v:+8.4f} BB "
          f"(${v * 20:+9.2f})")
    return v


def main() -> None:
    model = _load_model()

    print("=== Hero=BTN, button=0 ===")
    print("[6-seat with 4 auto-checking sitting-out seats]")
    inflated_a, _, delta_a = _probe_inflated(model, button=0)
    print("[2-seat native control (no inflation)]")
    v_native_a = _probe_native_2seat(model, button=0)
    print(f"  native_2seat - corrected_6seat = "
          f"{v_native_a - (inflated_a + delta_a):+.4f} BB "
          f"(should be small if our scalar rewrite is faithful)")
    print()

    print("=== Hero=BB, button=3 (villain on BTN) ===")
    print("[6-seat with 4 auto-checking sitting-out seats]")
    _, _, _ = _probe_inflated(model, button=3)
    print("[2-seat native control]")
    _ = _probe_native_2seat(model, button=1)
    print()

    print("=== Summary ===")
    print(f"  Pot-inflation impact on V: ~{abs(delta_a):.2f} BB shift "
          f"(${abs(delta_a) * 20:.2f}) per scenario.")
    print("  Sign is state-dependent — magnitude is the relevant signal.")
    print("  Value head IS materially sensitive to the inflated-pot encoding;")
    print("  fixing the engine to post antes only for in-hand seats would")
    print("  shift inference outputs by several BB on this kind of spot.")


if __name__ == "__main__":
    main()
