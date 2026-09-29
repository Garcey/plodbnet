"""The observation-SEMANTICS revision switch (`PLO5BP_OBS_REV`).

The 2026-09-20 review fixes move observation VALUES at 15-78% of decision
nodes while the layout stays put. A checkpoint is only served / resumed exactly
on the semantics it was trained on, so:

  rev 2 (default)        the fixed values  — asserted by test_review_obs_features.py
  rev 1 (PLO5BP_OBS_REV=1) the pre-fix values, BIT-EXACT — asserted here

Gated: B1 STK-2 (1024-1029) + STK-5[2:4] (1040-1041), B2 draw flags (800/802),
B3 min/max scalars (PLO 186/187, minimal slots 2/3, NLH 134/135), B5 blocker
flush dims (999/1000, 1003/1004), B7 NLH 906. NOT gated: B6, STK-6, C4/C6/C7.

What pins "rev 1 == the old encoder":
  * `_GOLDEN_PLO` — values of every gated dim computed by git 289bf87's scalar
    encoder (the pre-fix HEAD) on six replayable nodes that between them hit
    every gated dim, every regime (short shove, cover-short dust, a window on an
    illegal raise, negative STK-5, Q-K-A false positive, wheel false negative,
    other-board blocker) and PLO4/5/6. Frozen as literals on purpose: the test
    must not depend on `git`, nor on what HEAD happens to be later.
  * hand-derived expectations for the review's own examples.
  * all three encoders (scalar / numpy batch / fused Rust, full and minimal)
    bit-exact with each other in BOTH revisions, and ungated dims identical
    across revisions.

Most tests pin the revision in-process (`pin_obs_rev`: module constant for the
Python encoders + env var for the Rust engines, which read it at construction).
`test_public_encoders_honour_the_env_var_at_import` is the end-to-end check in
fresh interpreters: env var -> import-time constant -> public encode functions.

Needs an `_engine` binary rebuilt from this tree.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import plo5bp._engine as _engine  # type: ignore[attr-defined]
from plo5bp import encoding as E
from plo5bp import encoding_nlh as EN
from plo5bp._engine import BatchedEngine, GameState, draw_flags_batch  # type: ignore[attr-defined]
from plo5bp.actions import GATE_CHECK_CALL, GATE_RAISE, gate_mask_from_bounds
from plo5bp.config import VARIANT_NLH, GameConfig
from plo5bp.env import BombPotEnv

BB = 10_000
LEGACY, CURRENT = E.OBS_REV_LEGACY, E.OBS_REV_CURRENT

GATED_PLO = (186, 187, 800, 802, 999, 1000, 1003, 1004, *range(1024, 1030), 1040, 1041)
GATED_MINIMAL = (E._M_SCALARS + 2, E._M_SCALARS + 3)
GATED_NLH = (134, 135, 906)

# `obs_semantics_rev` at module level needs one registration line in lib.rs;
# the same function is always reachable as a static method of both classes.
_rust_obs_semantics_rev = getattr(_engine, "obs_semantics_rev", None) or getattr(
    GameState, "obs_semantics_rev", None
)
if _rust_obs_semantics_rev is None:
    # Same convention as test_encoding_rust.py: a stale binary skips, a rebuilt
    # one runs. (encoding.py already warned at import if rev 2 was requested.)
    pytest.skip(
        "plo5bp._engine predates the PLO5BP_OBS_REV switch — rebuild with "
        "`maturin develop --release`",
        allow_module_level=True,
    )


def pin_obs_rev(monkeypatch: pytest.MonkeyPatch, rev: int) -> None:
    """Run the rest of the test under observation-semantics revision `rev`
    (set it BEFORE building any env / engine — Rust reads it at construction)."""
    monkeypatch.setenv(E.OBS_REV_ENV, str(rev))
    monkeypatch.setattr(E, "OBS_SEMANTICS_REV", rev)
    monkeypatch.setattr(EN, "OBS_SEMANTICS_REV", rev)


def _card(rank: int, suit: int) -> int:
    return rank * 4 + suit


def _f32(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# The switch itself
# ---------------------------------------------------------------------------


def test_revision_constants_and_import_time_cross_check() -> None:
    assert (LEGACY, CURRENT) == (1, 2)  # CURRENT is what an unset variable selects
    assert E.OBS_SEMANTICS_REV in (LEGACY, CURRENT)
    assert EN.OBS_SEMANTICS_REV == E.OBS_SEMANTICS_REV  # one constant, imported
    # Import already cross-checked the two sides; they still agree now.
    assert _rust_obs_semantics_rev() == E.OBS_SEMANTICS_REV
    # The layout does not depend on the revision.
    assert (E.OBS_DIM, EN.OBS_DIM_NLH, E.OBS_DIM_MINIMAL) == (1171, 995, 796)


def test_module_level_obs_semantics_rev_is_registered() -> None:
    if not hasattr(_engine, "obs_semantics_rev"):
        pytest.skip(
            "bindings::obs_semantics_rev is not registered in rust_engine/src/lib.rs "
            "yet (one wrap_pyfunction! line); GameState.obs_semantics_rev() serves "
            "the import-time cross-check meanwhile"
        )
    assert _engine.obs_semantics_rev() == GameState.obs_semantics_rev()
    assert _engine.obs_semantics_rev() == BatchedEngine.obs_semantics_rev()


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, 2),      # unset
        ("", 2),        # `PLO5BP_OBS_REV=` — common in .env files: same as unset
        ("   ", 2),
        ("\t\n", 2),
        ("2", 2),
        ("1", 1),
        (" 1 ", 1),     # `set PLO5BP_OBS_REV=1 && ...` keeps the trailing blank on Windows
        ("2\n", 2),
    ],
)
def test_env_value_parsing_python_and_rust_agree(monkeypatch, raw, expected) -> None:
    if raw is None:
        monkeypatch.delenv(E.OBS_REV_ENV, raising=False)
    else:
        monkeypatch.setenv(E.OBS_REV_ENV, raw)
    assert E._read_obs_semantics_rev() == expected
    assert _rust_obs_semantics_rev() == expected
    assert GameState().obs_rev() == expected
    assert BatchedEngine(1).obs_rev() == expected


@pytest.mark.parametrize("raw", ["3", "0", "-1", "1.0", "one", "v2", "1 2"])
def test_unknown_revisions_are_rejected_on_both_sides(monkeypatch, raw) -> None:
    monkeypatch.setenv(E.OBS_REV_ENV, raw)
    with pytest.raises(ValueError, match=E.OBS_REV_ENV):
        E._read_obs_semantics_rev()
    with pytest.raises(ValueError, match=E.OBS_REV_ENV):
        _rust_obs_semantics_rev()
    # ... and an engine cannot even be constructed under a bogus value.
    with pytest.raises(ValueError, match=E.OBS_REV_ENV):
        GameState()
    with pytest.raises(ValueError, match=E.OBS_REV_ENV):
        BatchedEngine(2)


def test_engines_fix_their_revision_at_construction(monkeypatch) -> None:
    monkeypatch.setenv(E.OBS_REV_ENV, "1")
    gs, be = GameState(), BatchedEngine(2)
    monkeypatch.setenv(E.OBS_REV_ENV, "2")
    assert (gs.obs_rev(), be.obs_rev()) == (1, 1)  # not re-read afterwards
    assert (GameState().obs_rev(), BatchedEngine(2).obs_rev()) == (2, 2)
    # An explicit constructor argument overrides the environment.
    assert GameState(obs_rev=1).obs_rev() == 1
    assert BatchedEngine(2, obs_rev=1).obs_rev() == 1
    for bad in (0, 3):
        with pytest.raises(ValueError, match="obs_rev"):
            GameState(obs_rev=bad)
        with pytest.raises(ValueError, match="obs_rev"):
            BatchedEngine(2, obs_rev=bad)


def test_draw_flags_pyfunction_takes_the_revision_explicitly() -> None:
    def pad(cards, width):
        out = np.full((1, width), 255, dtype=np.uint8)
        out[0, : len(cards)] = cards
        return out

    # Q-K-A + rags on A-3-8: rev 1 reads a 4-run (shadow bit above the ace).
    qka = pad([_card(10, 3), _card(11, 1), _card(0, 0), _card(5, 2), _card(5, 1)], 5)
    qka_board = pad([_card(12, 2), _card(1, 3), _card(6, 0)], 5)
    # A-2 + rags on 3-4-J: the wheel draw rev 1 can never see.
    wheel = pad([_card(12, 3), _card(0, 1), _card(7, 0), _card(7, 2), _card(11, 1)], 5)
    wheel_board = pad([_card(1, 2), _card(2, 3), _card(9, 0)], 5)
    empty = pad([], 5)
    for hole, board, by_rev in ((qka, qka_board, {1: 1.0, 2: 0.0}), (wheel, wheel_board, {1: 0.0, 2: 1.0})):
        h, b = [int(c) for c in hole[0]], [int(c) for c in board[0] if c < 52]
        for rev, want in by_rev.items():
            assert E._draw_flags(h, b, rev)[1] == want
            assert float(E._draw_flags_batch(hole, board, rev)[1][0]) == want
            assert float(draw_flags_batch(hole, board, empty, obs_rev=rev)[1][0]) == want
            assert float(E._draw_flags_boards(hole, board, empty, rev)[1][0]) == want
        # The pyfunction defaults to the CURRENT revision.
        assert float(draw_flags_batch(hole, board, empty)[1][0]) == by_rev[2]
    with pytest.raises(ValueError, match="obs_rev"):
        draw_flags_batch(qka, qka_board, empty, obs_rev=3)


# ---------------------------------------------------------------------------
# Three encoders on one replayed node
# ---------------------------------------------------------------------------


def _encode_node(monkeypatch, rev, *, variant, stacks, seed, button, actions) -> dict:
    """Replay `actions` on a serial env and a lock-step BatchedEngine under
    revision `rev`; return the node's raw obs and every encoder's vector."""
    pin_obs_rev(monkeypatch, rev)
    cfg = GameConfig(num_seats=len(stacks), starting_stacks=tuple(stacks), ante=30_000,
                     bb=BB, variant=variant)
    env = BombPotEnv(cfg)
    obs, info = env.reset(seed, button)
    be = BatchedEngine(1, num_seats=len(stacks), ante=30_000, bb=BB, variant=variant,
                       starting_stacks=np.asarray(stacks, dtype=np.uint64))
    assert be.obs_rev() == rev
    be.reset_batch(np.array([seed], dtype=np.uint64), np.array([button], dtype=np.uint8))
    for gate, chips in actions:
        short_shove = gate == GATE_RAISE and info.min_raise_chips == 0
        obs, _, done, info = env.step_hybrid(gate, chips)
        assert not done
        be.apply_hybrid_batch(
            np.array([3 if short_shove else gate], dtype=np.uint8),
            np.array([chips], dtype=np.uint64),
        )
    bundle = be.observation_and_features_batch()
    cat_a, cat_b = np.asarray(bundle["hero_cat_a"]), np.asarray(bundle["hero_cat_b"])
    return {
        "raw": info.raw_obs,
        "scalar": obs,
        "numpy": E.encode_observation_batch(bundle, cat_a, cat_b, cfg)[0],
        "rust": np.asarray(be.observation_encoded_batch()["obs"])[0],
        "min_scalar": E.encode_observation_minimal(info.raw_obs, cfg),
        "min_numpy": E.encode_observation_batch_minimal(bundle, cfg)[0],
        "min_rust": np.asarray(be.observation_encoded_minimal_batch()["obs"])[0],
    }


def _assert_three_way(enc: dict) -> None:
    for name in ("numpy", "rust"):
        assert np.array_equal(enc[name], enc["scalar"]), (
            f"{name} != scalar at dims {np.nonzero(enc[name] != enc['scalar'])[0][:10].tolist()}"
        )
        assert np.array_equal(enc["min_" + name], enc["min_scalar"]), f"minimal {name} != scalar"
    assert np.array_equal(E.project_obs_minimal(enc["scalar"]), enc["min_scalar"])


# Values of the gated dims from git 289bf87's scalar `encode_observation` (the
# pre-fix encoder), generated 2026-09-20. `state` is the engine state the node
# is expected to be in — if THAT assertion fails the engine's rules moved and
# the scenario needs regenerating; it says nothing about the encoder.
_GOLDEN_PLO = [
    dict(
        tags=["negative_stk5", "old_window_on_illegal_raise", "other_board_blocker", "qka_false_positive"],
        variant="plo6_double_bomb", stacks=(530000, 260000, 220000, 50000, 210000), seed=504035937, button=0,
        actions=[(2, 83454), (2, 184236), (1, 0), (1, 0), (1, 0), (1, 0), (2, 45516)],
        state={"actor": 2, "pot": 948224, "min_bet": 91032, "max_bet": 315764, "min_raise": 0, "max_raise": 0},
        head={
            186: 9.10319995880127, 187: 31.576400756835938, 800: 0.0, 802: 1.0,
            999: 0.0, 1000: 0.0, 1003: 0.0, 1004: 0.0,
            1024: 0.04580272361636162, 1025: 0.2719504237174988, 1026: 1.0, 1027: 1.0,
            1028: 1.0, 1029: 0.3636363744735718, 1040: -0.22571556270122528, 1041: 5.039699554443359,
        },
    ),
    dict(
        tags=["negative_stk5", "short_shove", "wheel_false_negative"],
        variant="plo5_double_bomb", stacks=(670000, 270000, 720000, 590000, 170000), seed=622557219, button=1,
        actions=[(1, 0), (2, 77866)],
        state={"actor": 4, "pot": 227866, "min_bet": 155732, "max_bet": 383598, "min_raise": 0, "max_raise": 140000},
        head={
            186: 15.573200225830078, 187: 38.359798431396484, 800: 0.0, 802: 1.0,
            999: 0.0, 1000: 0.0, 1003: 0.0, 1004: 0.0,
            1024: 0.25468710064888, 1025: 1.0, 1026: 1.0, 1027: 1.0,
            1028: 0.0, 1029: 0.8181818127632141, 1040: -0.308687686920166, 1041: 4.529580116271973,
        },
    ),
    dict(
        tags=["dust"],
        variant="plo4_double_bomb", stacks=(60000, 710000), seed=670327563, button=0,
        actions=[(2, 11321), (2, 27171), (1, 0)],
        state={"actor": 1, "pot": 114342, "min_bet": 10000, "max_bet": 2829, "min_raise": 2829, "max_raise": 2829},
        head={
            186: 1.0, 187: 0.28290000557899475, 800: 0.0, 802: 1.0,
            999: 0.0, 1000: 0.0, 1003: 0.0, 1004: 0.0,
            1024: 0.08745692670345306, 1025: 0.024741563946008682, 1026: 1.0, 1027: 1.0,
            1028: 1.0, 1029: 0.09090909361839294, 1040: 0.0, 1041: 2.5649492740631104,
        },
    ),
    dict(
        tags=["other_board_blocker"],
        variant="plo4_double_bomb", stacks=(870000, 140000), seed=1075082457, button=0,
        actions=[],
        state={"actor": 1, "pot": 60000, "min_bet": 10000, "max_bet": 60000, "min_raise": 10000, "max_raise": 60000},
        head={
            186: 1.0, 187: 6.0, 800: 1.0, 802: 0.0,
            999: 0.0, 1000: 0.0, 1003: 0.0, 1004: 0.0,
            1024: 0.1666666716337204, 1025: 1.0, 1026: 0.09090909361839294, 1027: 0.5454545617103577,
            1028: 0.0, 1029: 0.9090909361839294, 1040: 0.24512246251106262, 1041: 2.944438934326172,
        },
    ),
    dict(
        tags=["other_board_blocker", "qka_false_positive"],
        variant="plo4_double_bomb", stacks=(250000, 450000), seed=197242335, button=1,
        actions=[(2, 25811), (1, 0)],
        state={"actor": 0, "pot": 111622, "min_bet": 10000, "max_bet": 111622, "min_raise": 10000, "max_raise": 111622},
        head={
            186: 1.0, 187: 11.162199974060059, 800: 1.0, 802: 1.0,
            999: 0.0, 1000: 0.3333333432674408, 1003: 0.0, 1004: 0.0,
            1024: 0.08958807587623596, 1025: 1.0, 1026: 0.051496222615242004, 1027: 0.5748111605644226,
            1028: 0.0, 1029: 1.0, 1040: 0.2203935980796814, 1041: 3.5405707359313965,
        },
    ),
    dict(
        tags=["other_board_blocker", "qka_false_positive"],
        variant="plo6_double_bomb", stacks=(250000, 610000, 80000), seed=604440301, button=2,
        actions=[(1, 0), (2, 62497), (1, 0), (1, 0), (2, 144741), (1, 0), (2, 10344)],
        state={"actor": 1, "pot": 564820, "min_bet": 20688, "max_bet": 12762, "min_raise": 12762, "max_raise": 12762},
        head={
            186: 2.0687999725341797, 187: 1.2762000560760498, 800: 1.0, 802: 0.0,
            999: 0.0, 1000: 0.0, 1003: 0.0, 1004: 0.3333333432674408,
            1024: 0.017984434962272644, 1025: 0.004204018507152796, 1026: 1.0, 1027: 1.0,
            1028: 1.0, 1029: 0.09090909361839294, 1040: 0.0, 1041: 4.077537536621094,
        },
    ),
]


def test_goldens_cover_every_gated_dim_and_regime() -> None:
    assert set(GATED_PLO) == set().union(*(g["head"] for g in _GOLDEN_PLO))
    assert {t for g in _GOLDEN_PLO for t in g["tags"]} == {
        "short_shove", "dust", "old_window_on_illegal_raise", "negative_stk5",
        "qka_false_positive", "wheel_false_negative", "other_board_blocker",
    }
    assert {g["variant"] for g in _GOLDEN_PLO} == {
        "plo4_double_bomb", "plo5_double_bomb", "plo6_double_bomb"
    }


@pytest.mark.parametrize("golden", _GOLDEN_PLO, ids=lambda g: "+".join(g["tags"]))
def test_rev1_reproduces_the_pre_fix_encoder(monkeypatch, golden) -> None:
    scenario = {k: golden[k] for k in ("variant", "stacks", "seed", "button", "actions")}
    old = _encode_node(monkeypatch, LEGACY, **scenario)
    raw = old["raw"]
    assert {k: int(raw[k]) for k in golden["state"]} == golden["state"], "engine state drifted"

    # Rev 1 == the pre-fix encoder on every gated dim, in all three encoders.
    _assert_three_way(old)
    for dim, value in golden["head"].items():
        assert old["scalar"][dim] == np.float32(value), (dim, old["scalar"][dim], value)
    assert old["min_scalar"][GATED_MINIMAL[0]] == np.float32(golden["head"][186])
    assert old["min_scalar"][GATED_MINIMAL[1]] == np.float32(golden["head"][187])
    # Whatever else rev 1 says, 186/187 ARE min_bet_total()/max_bet_total().
    assert old["scalar"][186] == np.float32(int(raw["min_bet"]) / BB)
    assert old["scalar"][187] == np.float32(int(raw["max_bet"]) / BB)

    # Rev 2 on the same node: three-way exact too, differs from rev 1 on gated
    # dims ONLY (these nodes were picked because it does differ).
    new = _encode_node(monkeypatch, CURRENT, **scenario)
    _assert_three_way(new)
    moved = np.nonzero(new["scalar"] != old["scalar"])[0].tolist()
    assert moved and set(moved) <= set(GATED_PLO), moved
    assert set(np.nonzero(new["min_scalar"] != old["min_scalar"])[0].tolist()) <= set(GATED_MINIMAL)


# ---------------------------------------------------------------------------
# The review's own examples, hand-derived
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rev", [LEGACY, CURRENT])
def test_review_b1_example_hero_3bb_behind_in_a_15bb_pot(monkeypatch, rev) -> None:
    """Five seats x 3bb ante = 15bb pot; hero started with 6bb -> 3bb behind,
    first to act. The pre-fix encoder: "max raise = the pot, not stack-capped,
    11/11 anchors", a NEGATIVE stack-after-max-raise. Truth: 0.2 pot, capped,
    3/11."""
    enc = _encode_node(
        monkeypatch, rev, variant="plo5_double_bomb",
        stacks=(60_000, 500_000, 500_000, 500_000, 500_000), seed=7, button=4, actions=[],
    )
    raw = enc["raw"]
    assert (raw["actor"], raw["pot"], raw["stacks"][0]) == (0, 150_000, 30_000)
    assert (raw["min_bet"], raw["max_bet"]) == (10_000, 150_000)      # TOTALS: uncapped
    assert (raw["min_raise"], raw["max_raise"]) == (10_000, 30_000)   # what is legal
    _assert_three_way(enc)
    vec = enc["scalar"]
    min_scalars = enc["min_scalar"][list(GATED_MINIMAL)]
    if rev == LEGACY:
        # The lead's / review's pre-fix numbers: [0.067, 1.0, 0.333, 1.0, 0.0, 1.0].
        assert np.array_equal(vec[1024:1030], _f32(1 / 15, 1.0, 1 / 3, 1.0, 0.0, 1.0))
        assert np.array_equal(vec[186:188], _f32(1.0, 15.0))  # min/max_bet_total
        assert vec[1040] < 0.0
        assert np.array_equal(
            vec[1040:1042], _f32(np.log1p((30_000 - 150_000) / 450_000.0), np.log1p(45.0))
        )
        assert np.array_equal(min_scalars, _f32(1.0, 15.0))
    else:
        assert np.array_equal(vec[1024:1030], _f32(1 / 15, 0.2, 1 / 3, 1.0, 1.0, 3 / 11))
        assert np.array_equal(vec[186:188], _f32(1.0, 3.0))   # the legal window
        # All-in: nothing behind; pot after = 15 + 2*3 bb.
        assert np.array_equal(vec[1040:1042], _f32(0.0, np.log1p(21.0)))
        assert np.array_equal(min_scalars, _f32(1.0, 3.0))


@pytest.mark.parametrize("rev", [LEGACY, CURRENT])
def test_review_b5_example_through_the_scalar_encoder(monkeypatch, rev) -> None:
    pin_obs_rev(monkeypatch, rev)
    board_a = [_card(11, 3), _card(5, 3), _card(0, 3)]   # Ks 7s 2s
    board_b = [_card(12, 3), _card(8, 1), _card(3, 2)]   # As face-up on the OTHER board
    hole = [_card(10, 3), _card(6, 0), _card(6, 1), _card(2, 2), _card(1, 0)]  # Qs
    obs = {
        "actor": 0, "hero_hole": hole, "board_a": board_a, "board_b": board_b,
        "street": 1, "folded": [False, False], "all_in": [False, False],
        "stacks": [200_000, 200_000], "eff_stack_cap": [230_000, 230_000],
        "pot": 60_000, "bet_to_call": 0, "min_bet": 10_000, "max_bet": 60_000,
        "min_raise": 10_000, "max_raise": 60_000, "history": [],
        "street_commit": [0, 0], "total_commit": [30_000, 30_000],
        "button": 0, "last_aggressor": -1, "hero_category_a": 0, "hero_category_b": 0,
    }
    vec = E.encode_observation(obs, GameConfig(num_seats=2, starting_stack=230_000))
    blockers = vec[E._BLOCKER_A_OFF : E._BLOCKER_A_OFF + 2]
    if rev == LEGACY:
        # Board-local: the As is the top "missing" spade (hero lacks it), and
        # the top three As/Qs/Js include hero's Qs.
        assert np.array_equal(blockers, _f32(0.0, 1 / 3))
    else:
        # The As is face-up elsewhere: hero's Qs IS the top holdable spade, one
        # of Qs/Js/Ts.
        assert np.array_equal(blockers, _f32(1.0, 1 / 3))


@pytest.mark.parametrize("rev", [LEGACY, CURRENT])
def test_nlh_gated_dims_scalar_and_batch(monkeypatch, rev) -> None:
    """NLH HU study node on a five-spade river; hero (BB, the SHORT stack)
    holds the 2s. B7 (906): the 3s/4s below the board's 5s only tie — 7 in the
    pre-fix encoder, 5 fixed. B3 (134/135): villain's raw reach vs hero's legal
    window."""
    pin_obs_rev(monkeypatch, rev)
    cfg = GameConfig(num_seats=2, starting_stacks=(500_000, 1_000_000), ante=0, bb=BB,
                     sb=5_000, variant=VARIANT_NLH)
    env = BombPotEnv(cfg)
    hole = [_card(0, 3), _card(9, 1)]
    env.reset_study_nlh(1, 0, hole)           # button/SB = seat 1, hero = BB
    env.step_hybrid(GATE_CHECK_CALL, 0)       # SB completes
    env.step_hybrid(GATE_CHECK_CALL, 0)       # BB checks
    obs, info = env.set_flop_nlh(_card(12, 3), _card(11, 3), _card(6, 3))
    for setter, card in ((env.set_turn_nlh, _card(4, 3)), (env.set_river_nlh, _card(3, 3))):
        env.step_hybrid(GATE_CHECK_CALL, 0)
        env.step_hybrid(GATE_CHECK_CALL, 0)
        obs, info = setter(card)
    raw = info.raw_obs
    assert info.actor == 0 and len(raw["board_a"]) == 5

    pack = env._rs.pack_range_nlh(np.array([hole], dtype=np.uint8))
    batch = EN.encode_observation_batch_nlh(pack, np.asarray(pack["hero_cat_a"]), cfg)[0]
    assert np.array_equal(batch, obs)

    assert int(raw["max_bet"]) > int(raw["street_commit"][0]) + int(raw["max_raise"])
    if rev == LEGACY:
        assert obs[906] == 7.0
        assert obs[134] == np.float32(int(raw["min_bet"]) / BB)
        assert obs[135] == np.float32(int(raw["max_bet"]) / BB)   # villain's RAW reach
    else:
        assert obs[906] == 5.0
        sc = int(raw["street_commit"][0])
        assert obs[134] == np.float32((sc + int(raw["min_raise"])) / BB)
        assert obs[135] == np.float32((sc + int(raw["max_raise"])) / BB)  # hero's stack

    # The other revision moves these three dims and nothing else.
    pin_obs_rev(monkeypatch, LEGACY if rev == CURRENT else CURRENT)
    other = EN.encode_observation_nlh(raw, cfg)
    moved = set(np.nonzero(other != obs)[0].tolist())
    assert moved and moved <= set(GATED_NLH), moved
    assert np.array_equal(
        EN.encode_observation_batch_nlh(pack, np.asarray(pack["hero_cat_a"]), cfg)[0], other
    )


def test_nlh_b7_helper_in_both_revisions() -> None:
    board = [_card(12, 3), _card(11, 3), _card(6, 3), _card(4, 3), _card(3, 3)]
    hole = [_card(0, 3), _card(9, 1)]
    vc = np.zeros((13, 4), dtype=np.int32)
    for c in hole + board:
        vc[c // 4, c % 4] = 1
    b = np.array([board], dtype=np.uint8)
    h = np.array([hole], dtype=np.uint8)
    for rev, want in ((LEGACY, 7.0), (CURRENT, 5.0)):
        assert EN._sf_features_nlh(hole, board, vc, rev)[0] == want
        assert EN._sf_features_nlh_batch(h, b, vc[None], rev)[0, 0] == want


# ---------------------------------------------------------------------------
# Parity sweeps and the ungated fixes, in BOTH revisions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rev", [LEGACY, CURRENT])
def test_nlh_scalar_batch_parity_sweep(monkeypatch, rev) -> None:
    pin_obs_rev(monkeypatch, rev)
    rng = np.random.default_rng(31 + rev)
    nodes = 0
    for _ in range(12):
        ns = int(rng.integers(2, 7))
        stacks = [int(x) * BB for x in rng.integers(3, 200, size=ns)]
        cfg = GameConfig(num_seats=ns, starting_stacks=tuple(stacks), ante=5_000, bb=BB,
                         sb=5_000, variant=VARIANT_NLH)
        kw = dict(num_seats=ns, ante=5_000, bb=BB, sb=5_000, variant="nlh_single",
                  starting_stacks=np.asarray(stacks, dtype=np.uint64))
        n = 6
        be = BatchedEngine(n, **kw)
        seeds = rng.integers(0, 2**62, size=n).astype(np.uint64)
        buttons = rng.integers(0, ns, size=n).astype(np.uint8)
        be.reset_batch(seeds, buttons)
        serial = []
        for i in range(n):
            gs = GameState(**kw)
            gs.reset(int(seeds[i]), int(buttons[i]))
            serial.append(gs)
        for _step in range(60):
            if be.is_terminal_batch().all():
                break
            bundle = be.observation_and_features_batch()
            batch = EN.encode_observation_batch_nlh(bundle, np.asarray(bundle["hero_cat_a"]), cfg)
            for i in range(n):
                if serial[i].is_terminal():
                    assert not batch[i].any()
                    continue
                raw = dict(serial[i].observation_dict())
                raw["hero_category_a"] = int(serial[i].hero_category(raw["actor"], 0))
                assert np.array_equal(batch[i], EN.encode_observation_nlh(raw, cfg))
                nodes += 1
            legal = np.asarray(bundle["legal_mask"])
            mn, mx = np.asarray(bundle["min_raise"]), np.asarray(bundle["max_raise"])
            gate_mask = gate_mask_from_bounds(legal, mx, BB)
            gates = np.ones(n, dtype=np.uint8)
            chips = np.zeros(n, dtype=np.uint64)
            for i in range(n):
                if serial[i].is_terminal():
                    continue
                gate = int(rng.choice(np.nonzero(gate_mask[i])[0]))
                if gate == GATE_RAISE and mn[i] == 0:
                    gate = 3
                elif gate == GATE_RAISE:
                    chips[i] = int(rng.integers(int(mn[i]), int(mx[i]) + 1))
                gates[i] = gate
            be.apply_hybrid_batch(gates, chips)
            for i in range(n):
                if serial[i].is_terminal():
                    continue
                if gates[i] == GATE_RAISE:
                    serial[i].apply_raise_chips(int(chips[i]))
                else:
                    serial[i].apply_action({0: 0, 1: 1, 3: 7}[int(gates[i])])
    assert nodes > 200


@pytest.mark.parametrize("rev", [LEGACY, CURRENT])
def test_ungated_fixes_hold_in_both_revisions(monkeypatch, rev) -> None:
    pin_obs_rev(monkeypatch, rev)
    # B6: pot_at_flop sums the antes of the DEALT-IN seats.
    cfg = GameConfig(num_seats=6, starting_stack=200_000, ante=30_000, bb=BB)
    env = BombPotEnv(cfg)
    obs, info = env.reset(5, 0, in_hand_mask=[True, True, False, True, False, False])
    bet = int(info.max_raise_chips)
    obs, _, _, _ = env.step_hybrid(GATE_RAISE, bet)
    assert obs[E._STK10_OFF + 1] == np.float32(np.log1p(bet / 90_000.0))
    # STK-6: exact on the power-of-3 boundaries.
    for spr, expected in ((1, 1), (4, 2), (13, 3), (40, 4), (121, 5)):
        start = 30_000 + spr * 60_000
        enc = _encode_node(monkeypatch, rev, variant="plo5_double_bomb",
                           stacks=(start, start), seed=1, button=0, actions=[])
        _assert_three_way(enc)
        assert enc["scalar"][E._STK6_OFF] == float(expected)


# ---------------------------------------------------------------------------
# End to end in fresh interpreters: env var -> import -> PUBLIC encode functions
# ---------------------------------------------------------------------------

_CHILD = r"""
import json, sys
import numpy as np
from plo5bp import encoding as E, encoding_nlh as EN
from plo5bp._engine import BatchedEngine, GameState
from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv
stacks = (60_000, 500_000, 500_000, 500_000, 500_000)
cfg = GameConfig(num_seats=5, starting_stacks=stacks, ante=30_000, bb=10_000)
obs, info = BombPotEnv(cfg).reset(7, 4)
be = BatchedEngine(1, num_seats=5, ante=30_000, bb=10_000,
                   starting_stacks=np.asarray(stacks, dtype=np.uint64))
be.reset_batch(np.array([7], dtype=np.uint64), np.array([4], dtype=np.uint8))
bundle = be.observation_and_features_batch()
numpy_obs = E.encode_observation_batch(
    bundle, np.asarray(bundle["hero_cat_a"]), np.asarray(bundle["hero_cat_b"]), cfg)[0]
rust_obs = np.asarray(be.observation_encoded_batch()["obs"])[0]
print(json.dumps({
    "rev": E.OBS_SEMANTICS_REV, "rev_nlh": EN.OBS_SEMANTICS_REV,
    "engine_rev": [GameState().obs_rev(), be.obs_rev(), GameState.obs_semantics_rev()],
    "stk2": [float(x) for x in obs[1024:1030]], "scalars": [float(obs[186]), float(obs[187])],
    "three_way": bool(np.array_equal(obs, numpy_obs) and np.array_equal(obs, rust_obs)),
}))
"""


def _spawn(env_value: str | None) -> subprocess.Popen:
    env = dict(os.environ)
    env.pop(E.OBS_REV_ENV, None)
    env.pop("PLO5_RUST_ENCODER", None)
    if env_value is not None:
        env[E.OBS_REV_ENV] = env_value
    # The child must import THIS tree's package / binary.
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(E.__file__)))
    env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        [sys.executable, "-c", _CHILD], env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )


def test_public_encoders_honour_the_env_var_at_import() -> None:
    """No monkeypatching: four fresh interpreters (spawned together — the import
    alone costs ~2.5s), each reading PLO5BP_OBS_REV once at import."""
    cases = {"unset": None, "rev1": "1", "empty": "", "bogus": "3"}
    procs = {name: _spawn(value) for name, value in cases.items()}
    results = {name: (p.communicate(timeout=180), p.returncode) for name, p in procs.items()}

    (out, err), code = results["bogus"]
    assert code != 0 and "PLO5BP_OBS_REV='3'" in err and "ValueError" in err

    third = float(np.float32(1 / 3))
    fifteenth = float(np.float32(1 / 15))
    expected = {
        "unset": (2, [fifteenth, float(np.float32(0.2)), third, 1.0, 1.0, float(np.float32(3 / 11))], [1.0, 3.0]),
        "empty": (2, [fifteenth, float(np.float32(0.2)), third, 1.0, 1.0, float(np.float32(3 / 11))], [1.0, 3.0]),
        "rev1": (1, [fifteenth, 1.0, third, 1.0, 0.0, 1.0], [1.0, 15.0]),
    }
    for name, (rev, stk2, scalars) in expected.items():
        (out, err), code = results[name]
        assert code == 0, f"{name}: {err[-800:]}"
        got = json.loads(out.strip().splitlines()[-1])
        assert got["rev"] == got["rev_nlh"] == rev, (name, got)
        assert got["engine_rev"] == [rev, rev, rev], (name, got)
        assert got["stk2"] == stk2 and got["scalars"] == scalars, (name, got)
        assert got["three_way"], name
