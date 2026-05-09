"""Sub-1bb opening shove gate-mask + dispatch smoke.

The engine zeroes `min_raise_chips` for sub-1bb stacks (engine.rs::
short_open_shove_preserves_min_raise_floor), but `legal[ALL_IN]` is
still set so the actor can shove. The 3-gate hybrid mask now ORs
those signals together — `gate_mask_from_bounds` returns
`GATE_RAISE = (min_raise > 0) | legal[ALL_IN]` — so the network's
gate space CAN reach the shove. The env redirects the dispatch via
`apply(Action::AllIn)` because `apply_raise_chips` rejects when
`min == 0`.
"""

from __future__ import annotations

from plo5bp.actions import ALL_IN, CHECK_CALL, FOLD, GATE_RAISE, gate_mask_from_bounds
from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv


def test_sub_1bb_open_shove_is_legal_at_engine_and_gate_layers() -> None:
    # Heads-up bomb-pot, deep button + short BB. After 3-chip ante:
    # seat 0 has 100 - 3 = 97 behind (~9.7 bb), seat 1 has 5 - 3 = 2
    # behind (~0.2 bb, below 1bb min). button=0 → first_to_act_postflop
    # is seat 1 (the short stack), opening-position with no facing bet.
    cfg = GameConfig(
        num_seats=2,
        starting_stack=0,
        ante=3,
        bb=10,
        starting_stacks=(100, 5),
    )
    env = BombPotEnv(cfg)
    _, info = env.reset(seed=0, button=0)
    assert info.actor == 1, "short seat (seat 1) should be first to act"
    legal = info.legal_mask
    assert legal[ALL_IN], "engine must mark sub-1bb open shove ALL_IN legal"
    gm = gate_mask_from_bounds(legal, info.max_raise_chips)
    # GATE_RAISE is now legal because legal[ALL_IN] is set even though
    # min_raise == 0. The env redirects the dispatch to apply(ALL_IN);
    # the network reaches the shove via the 3-gate space.
    assert gm[GATE_RAISE], "GATE_RAISE must be legal when legal[ALL_IN] is set"
    assert info.min_raise_chips == 0, (
        "engine still zeros min_raise_chips for sub-1bb stack — gate-layer "
        "legality comes from the legal[ALL_IN] OR-branch"
    )


def test_sub_1bb_open_shove_apply_preserves_floor() -> None:
    # Same setup as above, then commit the shove via the network's
    # GATE_RAISE path. The env's step_hybrid sees `min_raise == 0`
    # and `legal[ALL_IN]` and redirects to `apply(Action::AllIn)`,
    # mirroring the engine.rs::short_open_shove_preserves_min_raise_floor
    # test (the 1bb floor is preserved on the discrete dispatch path).
    cfg = GameConfig(
        num_seats=2,
        starting_stack=0,
        ante=3,
        bb=10,
        starting_stacks=(100, 5),
    )
    env = BombPotEnv(cfg)
    _, info = env.reset(seed=0, button=0)
    assert info.actor == 1
    # Chip amount is moot — the env ignores it on the AllIn redirect path.
    _, _, _, info_after = env.step_hybrid(GATE_RAISE, info.max_raise_chips)
    raw = info_after.raw_obs
    # Seat 1's 2-chip shove advanced bet_to_call but did not reset
    # the 1bb floor. Seat 0 (deep) is now next to act.
    assert int(raw["bet_to_call"]) == 2


def test_deep_actor_can_bet_to_cover_sub_1bb_short() -> None:
    # HU bomb-pot, flop. Cover-short regime — inverse of the actor-is-
    # short cases above. Deep BB faces a covered sub-1BB SB. The 1 BB
    # lead-bet floor (20) exceeds SB's effective stack reach (15). The
    # engine's cover-short clamp must collapse both min and max raise
    # bounds to SB's reach so BB can make the legal covering bet
    # (degenerate single-amount range). Mirrors the UI screenshot
    # scenario: BB $340 behind, SB $15 behind, pot $120, 1 BB = $20.
    cfg = GameConfig(
        num_seats=2,
        starting_stack=0,
        ante=60,
        bb=20,
        starting_stacks=(75, 400),
    )
    env = BombPotEnv(cfg)
    _, info = env.reset(seed=0, button=0)
    # button=0 → seat 1 (BB) acts first postflop in HU.
    assert info.actor == 1, "BB (deep) should act first postflop"
    # Post-ante: SB=15 (sub-1BB), BB=340. Cover-short clamp: bounds
    # collapse to 15 (SB's effective reach).
    assert info.min_raise_chips == info.max_raise_chips == 15, (
        "cover-short clamp must collapse 1bb floor to short opp's reach"
    )
    legal = info.legal_mask
    assert legal[CHECK_CALL]
    # AllIn is NOT legal — BB has 340 chips post-ante; a 15-chip
    # covering bet leaves BB deep. Covering flows through Raise.
    assert not legal[ALL_IN], "covering bet is not all-in"
    gm = gate_mask_from_bounds(legal, info.max_raise_chips)
    assert gm[GATE_RAISE], "GATE_RAISE legal in cover-short regime"
    # Dispatch the covering bet via the network's GATE_RAISE path.
    _, _, _, info_after = env.step_hybrid(GATE_RAISE, info.max_raise_chips)
    raw_after = info_after.raw_obs
    assert int(raw_after["bet_to_call"]) == 15
    # SB (seat 0, the short) is now next to act, facing the cover.
    assert info_after.actor == 0
    legal_sb = info_after.legal_mask
    assert legal_sb[FOLD]
    assert legal_sb[CHECK_CALL]


def test_deep_actor_can_cover_after_3way_short_shove() -> None:
    # 3-way bomb-pot, flop. The multi-way variant of the cover-short
    # regime: SB shoves all-in for a full open, BB's min-raise floor
    # (= 2 × shove) exceeds BB's stack, but the deepest non-allin opp
    # (BTN) can still be covered by less than BB's stack. Cover-short
    # clamp must collapse min/max to BTN's reach despite delta > stack.
    # Mirrors the UI scenario: SB shoves $170, BB $230 stack, BTN $210.
    cfg = GameConfig(
        num_seats=3,
        starting_stack=0,
        ante=60,
        bb=20,
        starting_stacks=(230, 290, 270),
    )
    env = BombPotEnv(cfg)
    _, info = env.reset(seed=0, button=2)
    # button=2 → flop order: SB (0), BB (1), BTN (2). SB acts first.
    assert info.actor == 0, "SB acts first postflop (button=2 in 3-way)"
    # SB shoves all-in. Stack post-ante is 170 chips; max_raise_chips
    # for SB equals 170 (full stack open). Dispatch the shove via the
    # network's GATE_RAISE path — env apply_hybrid_batch routes it.
    sb_shove = info.max_raise_chips
    assert sb_shove == 170
    _, _, _, info_bb = env.step_hybrid(GATE_RAISE, sb_shove)
    raw = info_bb.raw_obs
    assert int(raw["bet_to_call"]) == 170, "SB shoved 170"
    assert info_bb.actor == 1, "BB acts next"
    # BB's min/max collapse to BTN's reach (210 chips).
    assert info_bb.min_raise_chips == info_bb.max_raise_chips == 210, (
        "cover-short clamp must collapse to BTN's effective reach"
    )
    legal_bb = info_bb.legal_mask
    assert legal_bb[FOLD]
    assert legal_bb[CHECK_CALL]
    gm = gate_mask_from_bounds(legal_bb, info_bb.max_raise_chips)
    assert gm[GATE_RAISE], "GATE_RAISE legal in 3-way cover-short"
    # BB raises to cover BTN. 230-chip stack drops to 20 after committing 210.
    _, _, _, info_btn = env.step_hybrid(GATE_RAISE, info_bb.max_raise_chips)
    raw_btn = info_btn.raw_obs
    assert int(raw_btn["bet_to_call"]) == 210
    assert info_btn.actor == 2, "BTN faces the cover"
    legal_btn = info_btn.legal_mask
    assert legal_btn[FOLD]
    assert legal_btn[CHECK_CALL]
