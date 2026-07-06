"""NLH range-grid: pack_range_nlh parity + /ranges backend.

The range grid's contract is bit-exactness: for any node and any candidate
hole, the packed+batch-encoded observation row must equal the SERIAL study
observation of an env where the acting seat actually holds that hole. This
is the same discipline as the what-if and batched-vs-serial parity pins.
"""

from __future__ import annotations

import numpy as np
import pytest

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.config import GameConfig
from plo5bp.encoding_nlh import OBS_DIM_NLH, encode_observation_batch_nlh
from plo5bp.env import BombPotEnv


def _cfg(seats: int = 6, stack_bb: float = 100.0) -> GameConfig:
    bb = 10_000
    return GameConfig(
        num_seats=seats,
        starting_stack=int(round(stack_bb * bb)),
        ante=bb // 2,
        bb=bb,
        variant="nlh_single",
        sb=bb // 2,
    )


# A line is a list of entries: ("a", gate, chips) | ("cards", [idx, ...]).
# Streets are applied via the NLH setters when the engine awaits them.
def _replay(env: BombPotEnv, line) -> tuple[np.ndarray, object]:
    obs, info = None, None
    for entry in line:
        if entry[0] == "a":
            _, gate, chips = entry
            obs, _r, _d, info = env.step_hybrid(gate, chips)
        else:
            cards = entry[1]
            awaiting = env.awaiting_next_street()
            assert awaiting is not None, "cards entry while not awaiting a street"
            if len(cards) == 3:
                obs, info = env.set_flop_nlh(*cards)
            else:
                assert len(cards) == 1
                if awaiting == 2:
                    obs, info = env.set_turn_nlh(cards[0])
                else:
                    obs, info = env.set_river_nlh(cards[0])
    return obs, info


def _range_row(env: BombPotEnv, combo, cfg: GameConfig) -> np.ndarray:
    pack = env.pack_range_nlh(np.asarray([combo], dtype=np.uint8))
    return encode_observation_batch_nlh(pack, pack["hero_cat_a"], cfg)[0]


def _serial_obs(cfg: GameConfig, line, actor: int, combo) -> np.ndarray:
    """Fresh study env where the node's ACTOR genuinely holds `combo`."""
    env = BombPotEnv(cfg)
    obs, info = env.reset_study_nlh(button=0, hero_seat=actor, hero_hole=list(combo))
    if line:
        obs, info = _replay(env, line)
    assert info.actor == actor
    return obs


def _drive_to_street(env, line, gates_until_awaiting):
    """Append check/call actions until the round closes."""
    for _ in range(gates_until_awaiting):
        line.append(("a", GATE_CHECK_CALL, 0))


def test_pack_range_parity_across_streets():
    cfg = _cfg(seats=3)
    env = BombPotEnv(cfg)
    # Placeholder hero: high cards that never collide with the test board.
    env.reset_study_nlh(button=0, hero_seat=0, hero_hole=[50, 51])

    # Node schedule: (line-so-far, combos to check). Board cards are low
    # indices; combos avoid them.
    flop = [2, 7, 12]
    turn = [17]
    river = [22]

    nodes: list[tuple[list, list]] = []
    line: list = []
    nodes.append((list(line), [[0, 1], [44, 40], [33, 21]]))  # preflop root
    line.append(("a", GATE_CHECK_CALL, 0))  # UTG (=BTN seat 0 3-max) limps
    nodes.append((list(line), [[0, 1], [48, 44]]))  # SB node
    line.append(("a", GATE_CHECK_CALL, 0))  # SB completes
    line.append(("a", GATE_CHECK_CALL, 0))  # BB checks -> awaiting flop
    line.append(("cards", flop))
    nodes.append((list(line), [[0, 1], [30, 26], [48, 44]]))  # flop, SB first
    line.append(("a", GATE_CHECK_CALL, 0))
    line.append(("a", GATE_CHECK_CALL, 0))
    line.append(("a", GATE_CHECK_CALL, 0))
    line.append(("cards", turn))
    nodes.append((list(line), [[0, 1], [35, 31]]))
    line.append(("a", GATE_CHECK_CALL, 0))
    line.append(("a", GATE_CHECK_CALL, 0))
    line.append(("a", GATE_CHECK_CALL, 0))
    line.append(("cards", river))
    nodes.append((list(line), [[0, 1], [45, 41]]))

    for node_line, combos in nodes:
        probe = BombPotEnv(cfg)
        _obs, info = probe.reset_study_nlh(
            button=0, hero_seat=0, hero_hole=[50, 51]
        )
        if node_line:
            _obs, info = _replay(probe, node_line)
        actor = info.actor
        assert actor is not None
        for combo in combos:
            got = _range_row(probe, combo, cfg)
            want = _serial_obs(cfg, node_line, actor, combo)
            assert got.shape == (OBS_DIM_NLH,)
            assert np.array_equal(got, want), (
                f"range row != serial obs at node depth {len(node_line)} "
                f"combo {combo}"
            )


def test_pack_range_villain_blindness():
    """Querying a combo that IS a villain's actual (placeholder) hole must
    produce the identical row as any other env holding different villain
    cards — villain cards never enter the observation."""
    cfg = _cfg(seats=3)
    env = BombPotEnv(cfg)
    env.reset_study_nlh(button=0, hero_seat=0, hero_hole=[50, 51])
    holes = env.all_hole_cards()
    villain_combo = list(holes[1])  # seat 1's actual placeholder cards
    got = _range_row(env, villain_combo, cfg)
    want = _serial_obs(cfg, [], 0, villain_combo)
    assert np.array_equal(got, want)


def test_pack_range_multi_row_batch_matches_single():
    cfg = _cfg(seats=2)
    env = BombPotEnv(cfg)
    env.reset_study_nlh(button=0, hero_seat=0, hero_hole=[50, 51])
    combos = np.array([[0, 1], [12, 8], [44, 40], [26, 25]], dtype=np.uint8)
    pack = env.pack_range_nlh(combos)
    batch = encode_observation_batch_nlh(pack, pack["hero_cat_a"], cfg)
    for i, combo in enumerate(combos.tolist()):
        single = _range_row(env, combo, cfg)
        assert np.array_equal(batch[i], single)


# --- /ranges endpoint ---------------------------------------------------


@pytest.fixture(scope="module")
def client():
    from starlette.testclient import TestClient

    import plo5bp.ui.server as server

    return TestClient(server.app)


def _q(client, **kw):
    body = {"seats": 6, "stack_bb": 100.0, "line": [], **kw}
    r = client.post("/ranges/query", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _a(gate, chips_bb=None):
    e = {"t": "a", "gate": gate}
    if chips_bb is not None:
        e["chips_bb"] = chips_bb
    return e


def test_ranges_preflop_root(client):
    d = _q(client)
    assert d["state"]["position"] == "UTG" and d["state"]["street"] == "preflop"
    assert d["state"]["pot_bb"] == 4.5  # 6 antes(0.5) + sb 0.5 + bb 1
    assert len(d["cells"]) == 169
    assert len(d["combos"]) == 1326
    # Every cell's gate freqs sum to ~1.
    for k, c in d["cells"].items():
        assert abs((c["f"] + c["c"] + c["r"]) - 1.0) < 2e-3, k
    # Root reach is uniform: all 1326 combos counted.
    assert abs(d["summary"]["total_combos"] - 1326) < 1e-6
    # allin is a sub-share of raise mass.
    aa = d["cells"]["AA"]
    assert 0.0 <= aa["ai"] <= aa["r"] + 1e-9
    assert d["cells"]["AKs"]["live"] == 4
    assert d["cells"]["AKo"]["live"] == 12
    assert d["cells"]["AA"]["live"] == 6


def test_ranges_sequence_nodes_and_positions(client):
    line = [_a("fold"), _a("fold")]
    d0 = _q(client, line=line, node=0)
    d1 = _q(client, line=line, node=1)
    d2 = _q(client, line=line, node=2)
    assert (
        d0["state"]["position"],
        d1["state"]["position"],
        d2["state"]["position"],
    ) == ("UTG", "HJ", "CO")
    assert [e["position"] for e in d2["sequence"]] == ["UTG", "HJ"]


def test_ranges_awaiting_terminal_and_board_removal(client):
    # Everyone folds to BB -> terminal.
    folds = [_a("fold")] * 5
    d = _q(client, line=folds)
    assert d.get("terminal") is True and "cells" not in d

    # Limped pot to the flop: 5 calls + BB check -> awaiting flop.
    calls = [_a("check_call")] * 6
    d = _q(client, line=calls)
    assert d.get("awaiting") == "flop" and d.get("need") == 3

    flop = {"t": "cards", "cards": [2, 7, 12]}
    d = _q(client, line=calls + [flop])
    assert d["state"]["street"] == "flop"
    assert d["state"]["board"] == [2, 7, 12]
    assert len(d["combos"]) == 1176  # C(49,2)
    # Pair cells holding a board card lose combos: 3 of 6 remain.
    for cell in ("22", "33", "55"):  # cards 2,7,12 = ranks 2,3,5
        assert d["cells"][cell]["live"] == 3
    names = set(d["combos"])
    assert not any("2h" in n for n in names)  # card idx 2 = 2h


def test_ranges_reach_product(client):
    """BB's reach on the flop equals its preflop call probability,
    combo by combo (gate-level reach, single prior action)."""
    line = [
        _a("raise", 2.5),  # UTG opens
        _a("fold"), _a("fold"), _a("fold"), _a("fold"),  # HJ..SB
        _a("check_call"),  # BB defends
    ]
    d_bb_pre = _q(client, line=line, node=5)
    assert d_bb_pre["state"]["position"] == "BB"
    d_flop = _q(client, line=line + [{"t": "cards", "cards": [2, 7, 12]}])
    assert d_flop["state"]["position"] == "BB"
    for name in ("AsKs", "Qd2d", "7c6c"):
        pre_call = d_bb_pre["combos"][name]["c"]
        flop_reach = d_flop["combos"][name]["reach"]
        assert abs(pre_call - flop_reach) < 2e-4, name
    # And the UTG opener's reach at ITS flop node uses raise prob.
    line2 = line + [{"t": "cards", "cards": [2, 7, 12]}, _a("check_call")]
    d_utg_flop = _q(client, line=line2)
    assert d_utg_flop["state"]["position"] == "UTG"
    for name in ("AsKs", "7c6c"):
        pre_raise = d_bb_pre and _q(client, line=line, node=0)["combos"][name]["r"]
        assert abs(pre_raise - d_utg_flop["combos"][name]["reach"]) < 2e-4


def test_ranges_sizes_and_model_meta(client):
    d = _q(client)
    assert isinstance(d["model"]["loaded"], bool)
    if d["state"]["legal"]["raise"]:
        assert d["sizes"], "aggregate size histogram expected"
        s = sum(x["frac"] for x in d["sizes"])
        assert abs(s - 1.0) < 2e-3
        assert d["sizes"][-1]["label"] == "ALL-IN"


def test_ranges_validation_errors(client):
    r = client.post(
        "/ranges/query",
        json={"seats": 6, "stack_bb": 100.0, "line": [_a("raise", 10_000.0)]},
    )
    assert r.status_code == 400  # raise-to beyond max
    r = client.post(
        "/ranges/query",
        json={
            "seats": 6, "stack_bb": 100.0,
            "line": [{"t": "cards", "cards": [1, 2, 3]}],
        },
    )
    assert r.status_code == 400  # cards while an action is pending
    r = client.post(
        "/ranges/query",
        json={"seats": 6, "stack_bb": 100.0, "line": [], "node": 3},
    )
    assert r.status_code == 400  # node out of range


def test_pack_range_errors():
    cfg = _cfg(seats=3)
    env = BombPotEnv(cfg)
    env.reset_study_nlh(button=0, hero_seat=0, hero_hole=[50, 51])
    # Bad combos.
    with pytest.raises(Exception):
        env.pack_range_nlh(np.array([[5, 5]], dtype=np.uint8))
    with pytest.raises(Exception):
        env.pack_range_nlh(np.array([[52, 0]], dtype=np.uint8))
    # Board collision (reach the flop first).
    _replay(
        env,
        [
            ("a", GATE_CHECK_CALL, 0),
            ("a", GATE_CHECK_CALL, 0),
            ("a", GATE_CHECK_CALL, 0),
            ("cards", [2, 7, 12]),
        ],
    )
    with pytest.raises(Exception):
        env.pack_range_nlh(np.array([[2, 30]], dtype=np.uint8))
    # PLO refusal.
    plo = BombPotEnv(GameConfig(num_seats=6, starting_stack=200_000, ante=30_000, bb=10_000))
    plo.reset(1, 0)
    with pytest.raises(Exception):
        plo.pack_range_nlh(np.array([[0, 1]], dtype=np.uint8))
