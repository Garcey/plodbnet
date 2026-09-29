"""Review 2026-09-20 D3 / D5 / F11 — PolicyNetHost serves what the net trained on.

D3: the label obs is the solver's CANONICAL form (synthetic street root:
current-street history only, ``total_commit == street_commit``, blind flags
off). These tests PLAY real env hands to a postflop decision, build the label
the CFR export would emit for the same public state, and pin the host's
canonicalized obs bit-exactly to ``obs_from_label``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plo5bp.actions import GATE_CHECK_CALL, GATE_RAISE
from plo5bp.config import VARIANT_NLH, GameConfig
from plo5bp.env import BombPotEnv
from plo5bp.gto.labels import LABEL_SCHEMA_VERSION, LabelRecord
from plo5bp.gto.obs_from_label import (
    OBS_KIND_ENGINE,
    canonical_serve_obs,
    obs_from_label_detailed,
)
from plo5bp.gto.policy_host import JAM_DUST_BB, PolicyNetHost
from plo5bp.gto.policy_net import build_policy_net
from plo5bp.gto.roots import CLUBGG_NLH_ROOT
from plo5bp.sizing import NLH_ANCHOR_SPEC, anchor_grid_np, sizing_from_info


def _engine_has_cfr_node() -> bool:
    try:
        from plo5bp._engine import GameState

        return hasattr(GameState, "reset_nlh_cfr_node")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _engine_has_cfr_node(), reason="rebuild: reset_nlh_cfr_node"
)

BB = CLUBGG_NLH_ROOT.bb


def _host(meta: dict | None = None) -> PolicyNetHost:
    torch.manual_seed(0)
    return PolicyNetHost(
        model=build_policy_net(hidden_dim=32, num_layers=1).eval(),
        device=torch.device("cpu"),
        meta=meta,
    )


def _play_to_street(env: BombPotEnv, info, street: int):
    """Check / call until ``street`` is reached."""
    obs = None
    while int(info.raw_obs["street"]) < street and not info.terminal:
        obs, _, _, info = env.step_hybrid(GATE_CHECK_CALL, 0)
    assert int(info.raw_obs["street"]) == street and info.actor is not None
    return obs, info


def _label_from_live(info, path: list[str]) -> LabelRecord:
    """The LabelRecord ``cfr_export`` would emit for this live public state.

    CFR seat map (postflop): seat 0 acts first, button = last seat — i.e. the
    live seats rotated so the seat left of the button is 0.
    """
    raw = info.raw_obs
    n = len(raw["stacks"])
    button = int(raw["button"])
    live_of = [(button + 1 + k) % n for k in range(n)]  # label seat -> live seat
    hero = live_of.index(int(info.actor))
    commit = [int(x) for x in raw["street_commit"]]
    root_stacks = [int(raw["stacks"][s]) + commit[s] for s in live_of]
    sz = sizing_from_info(info)
    return LabelRecord(
        schema_version=LABEL_SCHEMA_VERSION,
        source="rust_cfr",
        root_name="parity_root",
        num_seats=n,
        street=int(raw["street"]),
        spr=0.0,
        pot_chips=int(raw["pot"]),
        to_call_chips=int(sz[3]),
        min_raise_chips=int(sz[0]),
        max_raise_chips=int(sz[1]),
        hero_seat=hero,
        button=n - 1 if n > 2 else 1,
        hero_hole=[int(c) for c in raw["hero_hole"]],
        board=[int(c) for c in raw["board_a"]],
        stacks_chips=[int(raw["stacks"][s]) for s in live_of],
        gate_probs=[0.0, 1.0, 0.0],
        action_probs=[],
        notes={
            "path_tokens": list(path),
            "path": ",".join(path),
            "bb_chips": BB,
            "root_pot_chips": int(raw["pot"]) - sum(commit),
            "root_stacks_chips": root_stacks,
        },
    )


def _assert_parity(obs, info, path: list[str]) -> None:
    lab = _label_from_live(info, path)
    want, kind = obs_from_label_detailed(lab)
    assert kind == OBS_KIND_ENGINE, "label must rebuild through the engine"
    got = _host().canonical_obs(np.asarray(obs, dtype=np.float32), info)
    np.testing.assert_array_equal(got, want)
    # ... and the live obs really is a DIFFERENT distribution (the D3 bug).
    assert np.any(np.asarray(obs, dtype=np.float32) != want)


@pytest.mark.parametrize("button", [0, 1])
def test_played_hand_river_root_matches_label_bit_exact(button: int):
    env = BombPotEnv(CLUBGG_NLH_ROOT.game_config(num_seats=2, stack_bb=25.0))
    _, info = env.reset(12345, button)
    obs, info = _play_to_street(env, info, 3)
    _assert_parity(obs, info, [])


def test_played_hand_river_facing_bet_matches_label_bit_exact():
    env = BombPotEnv(CLUBGG_NLH_ROOT.game_config(num_seats=2, stack_bb=25.0))
    _, info = env.reset(777, 0)
    _, info = _play_to_street(env, info, 3)
    half_pot = int(info.raw_obs["pot"]) * 500 // 1000  # the solver's RAISE_500
    obs, _, _, info = env.step_hybrid(GATE_RAISE, half_pot)
    assert sizing_from_info(info)[3] == half_pot
    _assert_parity(obs, info, ["RAISE_500"])


def test_played_hand_with_preflop_raise_flop_root_bit_exact():
    """Prior-street aggression + blind flags live in the played obs only."""
    env = BombPotEnv(CLUBGG_NLH_ROOT.game_config(num_seats=2, stack_bb=40.0))
    _, info = env.reset(4242, 1)
    _, _, _, info = env.step_hybrid(GATE_RAISE, int(info.min_raise_chips))
    obs, info = _play_to_street(env, info, 1)
    assert any(int(r[3]) == 0 for r in info.raw_obs["history"])
    _assert_parity(obs, info, [])


def test_played_hand_unequal_stacks_turn_bit_exact():
    cfg = GameConfig(
        num_seats=2,
        starting_stack=300_000,
        starting_stacks=(300_000, 1_200_000),
        ante=CLUBGG_NLH_ROOT.ante,
        bb=BB,
        sb=CLUBGG_NLH_ROOT.sb,
        variant=VARIANT_NLH,
    )
    env = BombPotEnv(cfg)
    _, info = env.reset(99, 0)
    obs, info = _play_to_street(env, info, 2)
    _assert_parity(obs, info, [])
    obs, _, _, info = env.step_hybrid(GATE_CHECK_CALL, 0)
    _assert_parity(obs, info, ["CHECK_CALL"])


def test_played_hand_three_handed_flop_bit_exact():
    env = BombPotEnv(CLUBGG_NLH_ROOT.game_config(num_seats=3, stack_bb=30.0))
    _, info = env.reset(31337, 2)
    obs, info = _play_to_street(env, info, 1)
    assert not any(info.raw_obs["folded"])
    _assert_parity(obs, info, [])


def test_preflop_obs_is_served_unchanged():
    env = BombPotEnv(CLUBGG_NLH_ROOT.game_config(num_seats=2, stack_bb=25.0))
    obs, info = env.reset(5, 0)
    assert canonical_serve_obs(info.raw_obs, bb=BB) is None
    np.testing.assert_array_equal(_host().canonical_obs(obs, info), obs)


def test_host_recovers_live_bb_without_a_config():
    cfg = GameConfig(
        num_seats=2, starting_stack=5_000, ante=25, bb=100, sb=50, variant=VARIANT_NLH
    )
    env = BombPotEnv(cfg)
    _, info = env.reset(3, 0)
    obs, info = _play_to_street(env, info, 3)
    host = _host()
    assert host._bb_chips(obs, info.raw_obs) == 100
    np.testing.assert_array_equal(
        host.canonical_obs(obs, info), canonical_serve_obs(info.raw_obs, bb=100)
    )


def test_live_trained_nets_keep_the_live_obs():
    """Legacy bootstrap / distill checkpoints were trained on LIVE obs."""
    env = BombPotEnv(CLUBGG_NLH_ROOT.game_config(num_seats=2, stack_bb=25.0))
    _, info = env.reset(12345, 0)
    obs, info = _play_to_street(env, info, 3)
    legacy = _host({"source": "rule_bootstrap"})
    assert legacy.serves_canonical_obs is False
    np.testing.assert_array_equal(legacy.canonical_obs(obs, info), obs)
    assert _host({"source": "rust_cfr"}).serves_canonical_obs is True
    assert _host({"source": "x", "obs_forms": {"canonical": 9}}).serves_canonical_obs


# --- D5: no refinement, jams are exact ----------------------------------------


def _river_root(stack_bb: float = 25.0, seed: int = 12345):
    env = BombPotEnv(CLUBGG_NLH_ROOT.game_config(num_seats=2, stack_bb=stack_bb))
    _, info = env.reset(seed, 0)
    obs, info = _play_to_street(env, info, 3)
    return env, obs, info


def test_host_act_serves_grid_chips_never_a_refined_size():
    _, obs, info = _river_root()
    host = _host()
    mn, mx, pot, tc = (int(x) for x in sizing_from_info(info))
    grid = anchor_grid_np(mn, mx, pot, tc, NLH_ANCHOR_SPEC)
    allowed = {int(c) for c, ok in zip(grid.chips, grid.legal) if ok}
    raises = 0
    for seed in range(200):
        gate, chips = host.act(obs, info, deterministic=False, rng_seed=seed)
        if gate == GATE_RAISE:
            raises += 1
            assert chips in allowed, (chips, sorted(allowed))
        else:
            assert chips == 0
    assert raises > 20


def test_host_interior_jam_anchor_is_exactly_all_in():
    """SPR 2 (pot 3bb, 6bb behind): the jam is the INTERIOR 200% anchor, whose
    live refinement bracket [180%, 237%] reaches below the stack — the untrained
    refine head used to land 53% of these jams short of all-in."""
    _, obs, info = _river_root(stack_bb=7.5)
    mn, mx, pot, tc = (int(x) for x in sizing_from_info(info))
    grid = anchor_grid_np(mn, mx, pot, tc, NLH_ANCHOR_SPEC)
    k_jam = next(k for k in range(NLH_ANCHOR_SPEC.count) if int(grid.chips[k]) == mx)
    assert 0 < k_jam < NLH_ANCHOR_SPEC.count - 1 and bool(grid.refine_ok[k_jam])
    host = _host()
    with torch.no_grad():  # force the net onto (raise, jam anchor)
        host.model.gate_head.bias.copy_(torch.tensor([-50.0, -50.0, 50.0]))
        host.model.gate_head.weight.zero_()
        host.model.anchor_head.weight.zero_()
        bias = torch.full((NLH_ANCHOR_SPEC.count,), -50.0)
        bias[k_jam] = 50.0
        host.model.anchor_head.bias.copy_(bias)
    for seed in range(50):
        assert host.act(obs, info, rng_seed=seed) == (GATE_RAISE, mx)
    nd = host.node_distribution(obs, info)
    assert (nd.rec_gate, nd.rec_anchor, nd.rec_chips) == (GATE_RAISE, k_jam, mx)
    assert not any(nd.refine_ok)  # every anchor is an atom for a GTO net


def test_host_snaps_near_stack_sizes_to_all_in():
    _, obs, info = _river_root(stack_bb=4.0)
    host = _host()
    mn, mx, pot, tc = (int(x) for x in sizing_from_info(info))
    dust = int(JAM_DUST_BB * BB)
    grid = anchor_grid_np(mn, mx, pot, tc, NLH_ANCHOR_SPEC)
    for k in range(NLH_ANCHOR_SPEC.count):
        chips = host._anchor_chips(k, obs, info)
        assert chips == mx or chips < mx - dust
        if int(grid.chips[k]) < mx - dust:
            assert chips == int(grid.chips[k])


# --- F11: supports() / value head ---------------------------------------------


def test_supports_reports_trained_coverage_only():
    hu_river = _host(
        {
            "source": "rust_cfr",
            "coverage": {"streets": [0, 3], "seats": [2], "streets_exact": [3]},
        }
    )
    assert hu_river.supports(seats=2, street=3)
    assert not hu_river.supports(seats=2, street=0)  # synthetic preflop obs
    assert not hu_river.supports(seats=2, street=1)
    assert not hu_river.supports(seats=6, street=3)
    assert hu_river.coverage_badge()["seats"] == "2"
    # No recorded coverage → the v1 teacher scope, not "2-6 seats, all streets".
    legacy = _host({"source": "rust_cfr"})
    assert legacy.supports(seats=2, street=3)
    assert not legacy.supports(seats=3, street=3)
    assert not legacy.supports(seats=2, street=2)


def test_untrained_value_head_is_not_displayed_as_an_ev():
    _, obs, info = _river_root()
    assert _host({"source": "rust_cfr", "n_value_targets": 0}).node_distribution(
        obs, info
    ).value_bb == 0.0
    assert _host({"source": "rust_cfr"}).value_head_trained is True
