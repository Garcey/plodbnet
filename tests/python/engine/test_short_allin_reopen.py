"""Per-seat short-all-in reopen rule (TDA-style).

A sub-min-raise all-in advances `bet_to_call` without reopening seats
that already responded to the prior bet — but a seat whose last action
was at a lower level (e.g. a check at 0) IS reopened once the
cumulative increase since its action reaches a full raise. Reported
from a live trainer hand: SB checks, BB bets, UTG short-shoves, BTN
calls — SB must be able to raise while BB stays locked.
"""

from __future__ import annotations

from plo5bp.actions import GATE_CHECK_CALL, GATE_RAISE, gate_mask_from_bounds
from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv


def _make_env(stacks: tuple[int, ...], button: int) -> tuple[BombPotEnv, object]:
    cfg = GameConfig(
        num_seats=len(stacks),
        starting_stack=0,
        ante=300,
        bb=100,
        starting_stacks=stacks,
    )
    env = BombPotEnv(cfg)
    _, info = env.reset(seed=7, button=button)
    return env, info


def test_checker_reopened_bettor_locked_after_short_shove() -> None:
    # 4 seats, button=3 -> flop order 0,1,2,3. Ante 300 each -> pot 1200.
    # Behind after ante: s0=19_700, s1=19_700, s2=700, s3=19_700.
    env, info = _make_env((20_000, 20_000, 1_000, 20_000), button=3)

    assert info.actor == 0
    _, _, _, info = env.step_hybrid(GATE_CHECK_CALL)  # seat 0 checks (level 0)
    assert info.actor == 1
    _, _, _, info = env.step_hybrid(GATE_RAISE, 600)  # full bet (half pot)
    assert info.actor == 2
    # Seat 2 shoves 700 total: > 600 call, < 1200 min-raise -> short all-in.
    # min_raise_chips is 0 for the short stack; step_hybrid redirects to
    # AllIn via the gate-2 short-shove branch.
    assert info.min_raise_chips == 0
    _, _, _, info = env.step_hybrid(GATE_RAISE, 0)
    assert int(info.raw_obs["bet_to_call"]) == 700
    assert info.actor == 3
    _, _, _, info = env.step_hybrid(GATE_CHECK_CALL)  # BTN-analog flat-calls

    # Back to seat 0 (the checker): reopened — full raise rights.
    assert info.actor == 0
    gm = gate_mask_from_bounds(info.legal_mask, info.max_raise_chips)
    assert gm[GATE_RAISE], "checker must be reopened by full bet + short shove"
    # Min raise = call 700 + the preserved 600 full-raise increment.
    assert info.min_raise_chips == 1300
    assert info.max_raise_chips >= info.min_raise_chips

    # Seat 0 flat-calls instead; seat 1 (who bet 600, faces +100 < 600)
    # must stay locked to fold/call.
    _, _, _, info = env.step_hybrid(GATE_CHECK_CALL)
    assert info.actor == 1
    gm1 = gate_mask_from_bounds(info.legal_mask, info.max_raise_chips)
    assert not gm1[GATE_RAISE], "original bettor must stay locked"
    assert info.min_raise_chips == 0
    assert info.max_raise_chips == 0


def test_full_raise_by_reopened_checker_reopens_everyone() -> None:
    # Same line, but the checker re-raises full — the original bettor's
    # rights must come back (bet grew by a full raise since its action).
    env, info = _make_env((20_000, 20_000, 1_000, 20_000), button=3)
    _, _, _, info = env.step_hybrid(GATE_CHECK_CALL)          # s0 check
    _, _, _, info = env.step_hybrid(GATE_RAISE, 600)          # s1 bet 600
    _, _, _, info = env.step_hybrid(GATE_RAISE, 0)            # s2 short 700
    _, _, _, info = env.step_hybrid(GATE_CHECK_CALL)          # s3 call 700
    assert info.actor == 0
    _, _, _, info = env.step_hybrid(GATE_RAISE, info.min_raise_chips)  # to 1300
    assert info.actor == 1
    gm1 = gate_mask_from_bounds(info.legal_mask, info.max_raise_chips)
    assert gm1[GATE_RAISE], "full re-raise must reopen the original bettor"
