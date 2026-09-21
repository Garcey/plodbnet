"""Review 2026-09-20 H1 — grading the recommended raise on CLIPPED brackets.

`score_move_v2` used to invert chips -> u over the CLAMPED anchor bracket
(`AnchorGrid.lo/hi`) while the policy maps `u` over the UNCLAMPED bracket
and clamps the chips afterwards (`sizing.refine_chips_*`). Whenever
min-raise landed inside the recommended anchor's bracket (or max-raise
inside the top one) the EXACT recommendation graded "inaccuracy".

Everything here runs on REAL engine nodes (`BombPotEnv` + `sizing`), with
the network heads pinned to a chosen (anchor, Beta) so the recommendation
is known in closed form. Mirrors the saved repros t16_clamp_common.py,
t6_clampu.py and t5_snap.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plo5bp.actions import GATE_CHECK_CALL, GATE_RAISE
from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv
from plo5bp.network import ActorCriticV2
from plo5bp.sizing import (
    PLO_ANCHOR_SPEC,
    anchor_grid_np,
    refine_chips_np,
    sizing_from_info,
)
from plo5bp.ui.trainer import (
    TrainerSession,
    TrainerSettings,
    WhatifRequest,
    attach_unclamped_brackets,
    compute_node_distribution,
    refine_u_interval,
    score_move_v2,
    snap_to_anchor,
    unclamped_brackets,
)

BETAS = [(3.0, 3.0), (5.0, 5.0), (10.0, 10.0), (3.0, 9.0)]
CPU = torch.device("cpu")


def _softplus_inv(y: float) -> float:
    return float(np.log(np.expm1(y)))


def _pinned_model(
    rec_anchor: int, alpha: float, beta: float, raise_gate: bool = True
) -> ActorCriticV2:
    """ActorCriticV2 whose heads ignore the observation: always raises (or
    always calls), always picks `rec_anchor`, every slider = Beta(a, b)."""
    torch.manual_seed(0)
    m = ActorCriticV2(hidden_dim=32, num_layers=1).eval()
    with torch.no_grad():
        m.gate_head.weight.zero_()
        m.gate_head.bias.copy_(
            torch.tensor([-3.0, 0.0, 3.0] if raise_gate else [-3.0, 3.0, 0.0])
        )
        m.anchor_head.weight.zero_()
        b = torch.full((11,), -4.0)
        b[rec_anchor] = 4.0
        m.anchor_head.bias.copy_(b)
        m.refine_head.weight.zero_()
        rb = torch.zeros(18)
        rb[0::2] = _softplus_inv(alpha - 1.0)
        rb[1::2] = _softplus_inv(beta - 1.0)
        m.refine_head.bias.copy_(rb)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# --- real engine nodes --------------------------------------------------------


def _node_min_clipped_open():
    """Unopened 4-handed bomb-pot flop (t16): pot 12bb, the 10% anchor's
    bracket [6000, 18000] is clipped by the 1bb min bet to [10000, 18000]."""
    cfg = GameConfig(num_seats=4, starting_stack=1_000_000, ante=30_000, bb=10_000)
    env = BombPotEnv(cfg)
    obs, info = env.reset(11, 0)
    return obs, info, 1


def _node_min_clipped_facing_bet(frac: float = 0.6667):
    """Heads-up, villain bets `frac` pot (t6): the min-RAISE lands inside
    the 30% anchor's bracket."""
    cfg = GameConfig(num_seats=2, starting_stack=2_000_000, ante=30_000, bb=10_000)
    env = BombPotEnv(cfg)
    obs, info = env.reset(5, 0)
    pot = int(info.raw_obs["pot"])
    obs, _, _, info = env.step_hybrid(GATE_RAISE, int(round(frac * pot)))
    return obs, info, 3


def _node_max_clipped():
    """Heads-up with 5.5bb behind into a 6bb pot: max_raise (= the stack)
    lands INSIDE the 90% anchor's bracket [51000, 57000]."""
    cfg = GameConfig(num_seats=2, starting_stack=85_000, ante=30_000, bb=10_000)
    env = BombPotEnv(cfg)
    obs, info = env.reset(7, 0)
    return obs, info, 9


NODES = {
    "min_clipped_open": _node_min_clipped_open,
    "min_clipped_facing_bet": _node_min_clipped_facing_bet,
    "min_clipped_facing_bet_60": lambda: _node_min_clipped_facing_bet(0.60),
    "max_clipped": _node_max_clipped,
}


def _grid(info):
    sz = sizing_from_info(info)
    return sz, anchor_grid_np(sz[0], sz[1], sz[2], sz[3], PLO_ANCHOR_SPEC)


@pytest.mark.parametrize("node_name", sorted(NODES))
def test_fixture_nodes_really_are_clipped(node_name):
    """Guard the fixtures: the rec anchor must be legal + refinable and its
    bracket genuinely clipped on the intended side (else the tests below
    would pass vacuously)."""
    _obs, info, k = NODES[node_name]()
    sz, grid = _grid(info)
    lo_raw, hi_raw = unclamped_brackets(
        int(sz[0]), int(sz[1]), int(sz[2]), int(sz[3]), PLO_ANCHOR_SPEC
    )
    assert bool(grid.legal[k]) and bool(grid.refine_ok[k])
    if node_name == "max_clipped":
        assert hi_raw[k] > int(grid.hi[k]) == int(sz[1])
    else:
        assert lo_raw[k] < int(grid.lo[k]) == int(sz[0])


def test_unclamped_brackets_match_the_policy_map():
    """`unclamped_brackets` is the chip axis `refine_chips_np` maps u over:
    u=0 / u=1 land on its ends wherever the clip is inactive, and the
    clipped grid is exactly its clip."""
    for make in NODES.values():
        _obs, info, _k = make()
        sz, grid = _grid(info)
        mn, mx, pot, tc = (int(x) for x in sz)
        lo_raw, hi_raw = unclamped_brackets(mn, mx, pot, tc, PLO_ANCHOR_SPEC)
        assert len(lo_raw) == len(hi_raw) == PLO_ANCHOR_SPEC.count
        for k in range(PLO_ANCHOR_SPEC.count):
            assert int(grid.lo[k]) == min(max(lo_raw[k], min(mn, mx)), mx)
            assert int(grid.hi[k]) == min(max(hi_raw[k], min(mn, mx)), mx)
            if not PLO_ANCHOR_SPEC.refinable[k]:
                continue
            for u, want in ((0.0, lo_raw[k]), (1.0, hi_raw[k])):
                got = int(refine_chips_np(k, u, mn, mx, pot, tc, PLO_ANCHOR_SPEC))
                assert got == min(max(want, min(mn, mx)), mx)


@pytest.mark.parametrize("node_name", sorted(NODES))
@pytest.mark.parametrize("ab", BETAS)
def test_exact_rec_is_best_on_clipped_brackets(node_name, ab):
    obs, info, k = NODES[node_name]()
    model = _pinned_model(k, *ab)
    dist = compute_node_distribution(model, CPU, obs, info)
    assert dist["rec_gate"] == GATE_RAISE and dist["rec_anchor"] == k
    sc = score_move_v2(dist, GATE_RAISE, dist["rec_chips"])
    assert sc["user_anchor"] == k
    assert sc["size_q"] == pytest.approx(1.0)
    assert sc["score"] == pytest.approx(100.0)
    assert sc["category"] == "best"


@pytest.mark.parametrize("node_name", sorted(NODES))
@pytest.mark.parametrize("ab", BETAS)
def test_policy_size_earns_full_credit_without_the_rec_shortcut(node_name, ab):
    """The inversion itself, not just the `chips == rec_chips` shortcut:
    when the recommended GATE is a call, a raise of exactly the size the
    policy's own (argmax anchor, Beta mean) produces must still read as
    u == mean on the unclamped axis (full size credit)."""
    obs, info, k = NODES[node_name]()
    model = _pinned_model(k, *ab, raise_gate=False)
    dist = compute_node_distribution(model, CPU, obs, info)
    assert dist["rec_gate"] == GATE_CHECK_CALL and dist["rec_chips"] == 0
    sz = sizing_from_info(info)
    mean = ab[0] / (ab[0] + ab[1])
    chips = int(refine_chips_np(k, mean, sz[0], sz[1], sz[2], sz[3], PLO_ANCHOR_SPEC))
    anchor, pdf_ratio = snap_to_anchor(dist, chips)
    assert anchor == k
    assert pdf_ratio == pytest.approx(1.0, abs=5e-3)
    sc = score_move_v2(dist, GATE_RAISE, chips)
    assert sc["user_anchor"] == k
    assert sc["size_q"] == pytest.approx(1.0, abs=5e-3)


@pytest.mark.parametrize("node_name", sorted(NODES))
def test_u_roundtrip_on_the_unclamped_axis(node_name):
    """chips = policy_map(k, u)  =>  u lies in the inverted interval (a
    point up to chip rounding; an interval where the clip collapsed it)."""
    obs, info, _k = NODES[node_name]()
    model = _pinned_model(5, 2.0, 2.0)
    dist = compute_node_distribution(model, CPU, obs, info)
    sz = sizing_from_info(info)
    rng = np.random.default_rng(0)
    checked = 0
    for k in range(PLO_ANCHOR_SPEC.count):
        if not (dist["anchor_legal"][k] and dist["refine_ok"][k]):
            continue
        span = dist["anchor_hi_raw"][k] - dist["anchor_lo_raw"][k]
        for u in rng.uniform(0.0, 1.0, size=25):
            chips = int(refine_chips_np(
                k, u, sz[0], sz[1], sz[2], sz[3], PLO_ANCHOR_SPEC
            ))
            u_lo, u_hi = refine_u_interval(dist, k, chips)
            tol = 1.0 / span + 1e-9  # one chip of rounding
            assert u_lo - tol <= u <= u_hi + tol, (k, u, chips, u_lo, u_hi)
            checked += 1
    assert checked > 0


def test_clipped_inversion_differs_from_the_old_clamped_one():
    """Regression pin for the bug itself: on the t16 node the old inversion
    (over the clamped bracket) put the 12000 rec at u=0.25; the policy's
    axis has it at u=0.5."""
    obs, info, k = _node_min_clipped_open()
    dist = compute_node_distribution(_pinned_model(k, 5.0, 5.0), CPU, obs, info)
    assert dist["rec_chips"] == 12_000
    lo_c, hi_c = dist["anchor_lo"][k], dist["anchor_hi"][k]
    assert (lo_c, hi_c) == (10_000, 18_000)
    assert (dist["rec_chips"] - lo_c) / (hi_c - lo_c) == pytest.approx(0.25)
    u_lo, u_hi = refine_u_interval(dist, k, dist["rec_chips"])
    assert u_lo == u_hi == pytest.approx(0.5)


def test_min_raise_credits_the_likelier_reading():
    """A min-raise is BOTH the min atom and the clipped low end of the first
    legal bracket. It must be credited with whichever the network prefers —
    never graded against the wrong one (the pre-fix nearest-by-chips snap
    always said "anchor 0")."""
    obs, info, k = _node_min_clipped_facing_bet(0.60)
    sz, grid = _grid(info)
    min_raise = int(sz[0])
    assert int(grid.chips[0]) == min_raise and int(grid.lo[k]) == min_raise

    # Network loves anchor k, Beta(3,9): its mean sits near the clipped low end.
    dist = compute_node_distribution(_pinned_model(k, 3.0, 9.0), CPU, obs, info)
    anchor, ratio = snap_to_anchor(dist, min_raise)
    assert anchor == k and ratio > 0.5

    # Network loves the min atom: a min-raise IS the top choice.
    dist0 = compute_node_distribution(_pinned_model(0, 3.0, 9.0), CPU, obs, info)
    assert dist0["rec_anchor"] == 0 and dist0["rec_chips"] == min_raise
    # ... even with the rec shortcut out of the picture (rec gate = call).
    dist0c = compute_node_distribution(
        _pinned_model(0, 3.0, 9.0, raise_gate=False), CPU, obs, info
    )
    anchor0, ratio0 = snap_to_anchor(dist0c, min_raise)
    assert anchor0 == 0 and ratio0 == 1.0
    sc = score_move_v2(dist0c, GATE_RAISE, min_raise)
    assert sc["size_q"] == pytest.approx(1.0)


def test_off_centre_size_is_still_penalised():
    """The fix must not hand out free credit: a size at the far end of the
    rec anchor's bracket still scores well below the recommendation."""
    obs, info, k = _node_min_clipped_open()
    dist = compute_node_distribution(_pinned_model(k, 10.0, 10.0), CPU, obs, info)
    far = dist["anchor_hi"][k] - 1
    sc = score_move_v2(dist, GATE_RAISE, far)
    assert sc["user_anchor"] == k
    assert sc["size_q"] < 0.05
    assert sc["category"] != "best"


def test_attach_is_noop_when_present_and_restores_when_dropped():
    obs, info, k = _node_min_clipped_open()
    dist = compute_node_distribution(_pinned_model(k, 5.0, 5.0), CPU, obs, info)
    want = (list(dist["anchor_lo_raw"]), list(dist["anchor_hi_raw"]))
    assert attach_unclamped_brackets(dist, info, PLO_ANCHOR_SPEC) is dist
    stripped = {
        key: v for key, v in dist.items()
        if key not in ("anchor_lo_raw", "anchor_hi_raw")
    }
    attach_unclamped_brackets(stripped, info, PLO_ANCHOR_SPEC)
    assert (stripped["anchor_lo_raw"], stripped["anchor_hi_raw"]) == want
    # v1 dicts and a mismatched ladder are left alone.
    assert "anchor_lo_raw" not in attach_unclamped_brackets(
        {"head_version": 1}, info, PLO_ANCHOR_SPEC
    )
    short = {"head_version": 2, "anchor_chips": [1, 2, 3]}
    assert "anchor_lo_raw" not in attach_unclamped_brackets(
        short, info, PLO_ANCHOR_SPEC
    )


# --- NLH ladder (12 anchors incl. the ALL-IN atom, v4 logistic head) -----------------


def _nlh_bb_node():
    """6-max 100bb NLH, UTG opens to 3.5bb, folds to the BB: the BB's
    min-raise (5bb more) lands inside the 25% anchor's bracket
    [4.705, 5.545]bb."""
    from plo5bp.actions import GATE_FOLD
    from plo5bp.config import VARIANT_NLH

    cfg = GameConfig(
        num_seats=6, starting_stack=1_000_000, ante=5_000, bb=10_000,
        sb=5_000, variant=VARIANT_NLH,
    )
    env = BombPotEnv(cfg)
    obs, info = env.reset(3, 0)
    obs, _, _, info = env.step_hybrid(GATE_RAISE, 35_000)
    for _ in range(4):
        obs, _, _, info = env.step_hybrid(GATE_FOLD, 0)
    assert int(info.actor) == int(info.raw_obs["bb_seat"])
    return obs, info, 1


@pytest.mark.parametrize("ab", BETAS)
def test_nlh_exact_rec_is_best_on_a_min_raise_clipped_bracket(ab):
    import math

    from plo5bp.encoding_nlh import OBS_DIM_NLH
    from plo5bp.network import ActorCriticV4
    from plo5bp.sizing import NLH_ANCHOR_SPEC

    obs, info, k = _nlh_bb_node()
    sz = sizing_from_info(info)
    grid = anchor_grid_np(sz[0], sz[1], sz[2], sz[3], NLH_ANCHOR_SPEC)
    lo_raw, hi_raw = unclamped_brackets(
        int(sz[0]), int(sz[1]), int(sz[2]), int(sz[3]), NLH_ANCHOR_SPEC
    )
    assert len(lo_raw) == len(hi_raw) == NLH_ANCHOR_SPEC.count == 12
    assert lo_raw[-1] == hi_raw[-1] == int(sz[1])  # ALL-IN atom = max_raise
    assert bool(grid.legal[k]) and bool(grid.refine_ok[k])
    assert lo_raw[k] < int(grid.lo[k]) == int(sz[0])  # genuinely clipped

    torch.manual_seed(0)
    m = ActorCriticV4(
        hidden_dim=32, obs_dim=OBS_DIM_NLH, num_layers=1,
        anchor_spec=NLH_ANCHOR_SPEC,
    ).eval()
    c = (NLH_ANCHOR_SPEC.count - 1) / 2.0
    with torch.no_grad():
        m.gate_head.weight.zero_()
        m.gate_head.bias.copy_(torch.tensor([-3.0, 0.0, 3.0]))
        m.size_head.weight.zero_()  # mu pinned ON anchor k, s at its floor
        m.size_head.bias.copy_(torch.tensor([math.atanh((k - c) / (c + 2.0)), -6.0]))
        m.refine_head.weight.zero_()
        rb = torch.zeros(20)
        rb[0::2] = _softplus_inv(ab[0] - 1.0)
        rb[1::2] = _softplus_inv(ab[1] - 1.0)
        m.refine_head.bias.copy_(rb)
    dist = compute_node_distribution(m, CPU, obs, info)
    assert dist["rec_gate"] == GATE_RAISE and dist["rec_anchor"] == k
    assert dist["anchor_lo_raw"] == lo_raw and dist["anchor_hi_raw"] == hi_raw
    sc = score_move_v2(dist, GATE_RAISE, dist["rec_chips"])
    assert sc["user_anchor"] == k and sc["category"] == "best"
    assert sc["score"] == pytest.approx(100.0)
    # and via the inversion alone (mean-u chips of anchor k, no rec shortcut)
    mean = ab[0] / (ab[0] + ab[1])
    chips = int(refine_chips_np(k, mean, sz[0], sz[1], sz[2], sz[3], NLH_ANCHOR_SPEC))
    anchor, ratio = snap_to_anchor({**dist, "rec_gate": GATE_CHECK_CALL}, chips)
    assert anchor == k and ratio == pytest.approx(1.0, abs=5e-3)


# --- end to end through the session ------------------------------------------------


@pytest.mark.parametrize("ab", BETAS)
def test_session_grades_exact_rec_best_with_zero_ev_loss_and_no_mc(tmp_path, ab):
    """t16 end-to-end: TrainerSession.act with exactly the recommended raise
    on a min-raise-clipped node -> "best", 100, ev_loss 0, MC never runs.
    Also pins that the unclamped brackets survive the StrategyBackend /
    NodeDist round trip, and that the what-if rescoring agrees."""
    model = _pinned_model(1, *ab)
    ts = TrainerSession(model, CPU, stats_path=tmp_path / "s.json")
    ts.set_settings(TrainerSettings(**{
        **ts.settings.model_dump(),
        "seats_mode": "fixed", "seats_fixed": 4, "mc_rollouts": 4,
        "hero_position_mode": "kth", "hero_kth": 1, "stack_bb": 100.0,
    }))
    ts.rng = np.random.default_rng(2)
    ts.new_hand()
    h = ts.hand
    assert not h.terminal and h.last_info.actor == h.hero_seat

    dist = ts._node_dist(h.last_obs, h.last_info)
    assert "anchor_lo_raw" in dist and "anchor_hi_raw" in dist
    k = dist["rec_anchor"]
    assert k == 1 and dist["anchor_lo_raw"][k] < dist["anchor_lo"][k]

    def _no_mc(*_a, **_k):  # playing the rec must never roll out
        raise AssertionError("MC ran for the exact recommendation")

    ts._rollout_ev = _no_mc  # type: ignore[method-assign]
    ts.act("raise", dist["rec_chips"])
    del ts._rollout_ev  # later (deviating) calls may roll out normally
    d = h.decisions[0]
    assert d.user_anchor == d.rec_anchor == 1
    assert d.score == pytest.approx(100.0)
    assert d.category == "best"
    assert d.ev_loss_bb == 0.0 and d.ev_loss_signed_bb == 0.0
    assert d.ev_user_bb is None

    # Finish the hand (the pinned model always raises, so just call down).
    steps = 0
    while not h.terminal and steps < 60:
        s = ts.project_state()
        if s["legal"]["check_call"]:
            ts.act("check_call", None)
        else:
            ts.act("fold", None)
        steps += 1
    assert h.terminal
    state = ts.whatif(WhatifRequest(decision=0))
    rescored = state["trainer"]["review"]["whatif"]["rescored"]
    assert rescored == {"score": 100.0, "category": "best"}
