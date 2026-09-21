"""Review 2026-09-20 H5 — NLH range grid (`plo5bp.ui.ranges`).

- all-in share when a FRACTION anchor clamps to the stack (the ALL-IN atom is
  then deduped as illegal and the jam mass sits on that fraction anchor);
- raise-BY (wire) vs raise-TO (display / input) amounts;
- NaN / ±inf `chips_bb` -> 400, not 500.

Self-contained app (tiny ActorCriticV4 + `create_ranges_router`): does not
import plo5bp.ui.server. Mirrors repros t11/t12 (all-in) and t2 (raise-to,
NaN).
"""

from __future__ import annotations

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from plo5bp.encoding_nlh import OBS_DIM_NLH
from plo5bp.network import ActorCriticV4
from plo5bp.sizing import NLH_ANCHOR_SPEC, anchor_grid_np
from plo5bp.ui.ranges import create_ranges_router

BB = 10_000


def _model(jam: bool) -> ActorCriticV4:
    torch.manual_seed(0)
    m = ActorCriticV4(
        hidden_dim=32, obs_dim=OBS_DIM_NLH, num_layers=1,
        anchor_spec=NLH_ANCHOR_SPEC,
    ).eval()
    if jam:
        with torch.no_grad():
            m.gate_head.weight.zero_()
            m.gate_head.bias.copy_(torch.tensor([-6.0, -6.0, 6.0]))  # always raise
            m.size_head.weight.zero_()
            m.size_head.bias.copy_(torch.tensor([5.0, -6.0]))  # mu past the top: JAM
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _client(jam: bool, ckpt: str = "tiny") -> TestClient:
    app = FastAPI()
    app.include_router(create_ranges_router(
        {"nlh_single": {"model": _model(jam), "loaded": True}},
        torch.device("cpu"), nlh_ckpt_name=ckpt,
    ))
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def jam():
    return _client(jam=True)


@pytest.fixture(scope="module")
def rnd():
    return _client(jam=False)


def _a(gate, chips_bb=None):
    e = {"t": "a", "gate": gate}
    if chips_bb is not None:
        e["chips_bb"] = chips_bb
    return e


def _q(client, stack=100.0, line=(), **kw):
    r = client.post(
        "/ranges/query",
        json={"seats": 6, "stack_bb": stack, "line": list(line), **kw},
    )
    assert r.status_code == 200, r.text
    return r.json()


# 20bb, UTG opens to 3bb, folds to the BB: pot 7.5bb, 2bb to call, BB has
# 18.5bb behind its blind. A 200%-pot raise would add 21bb > 18.5bb -> the 200%
# anchor clamps to max_raise and the ALL-IN atom behind it is deduped as illegal.
LOW_SPR = [_a("raise", 3.0)] + [_a("fold")] * 4


def test_fixture_low_spr_node_dedupes_the_allin_atom(jam):
    st = _q(jam, stack=20.0, line=LOW_SPR)["state"]
    assert st["position"] == "BB"
    mn = int(round(st["min_raise_bb"] * BB))
    mx = int(round(st["max_raise_bb"] * BB))
    pot = int(round(st["pot_bb"] * BB))
    tc = int(round(st["to_call_bb"] * BB))
    grid = anchor_grid_np(mn, mx, pot, tc, NLH_ANCHOR_SPEC)
    top = NLH_ANCHOR_SPEC.count - 1
    assert not bool(grid.legal[top]), "fixture: the ALL-IN atom must be deduped here"
    legal_at_max = [
        k for k in range(NLH_ANCHOR_SPEC.count)
        if grid.legal[k] and int(grid.chips[k]) == mx
    ]
    assert len(legal_at_max) == 1 and legal_at_max[0] != top
    assert st["allin_k"] == legal_at_max[0]


def test_jam_mass_on_a_clamped_fraction_anchor_counts_as_allin(jam):
    d = _q(jam, stack=20.0, line=LOW_SPR)
    st = d["state"]
    # the clamped fraction anchor is presented as the all-in
    last = st["anchors"][-1]
    assert last["label"] == "ALL-IN" and last["allin"] is True
    assert last["chips_bb"] == st["max_raise_bb"]
    assert [a["allin"] for a in st["anchors"][:-1]] == [False] * (len(st["anchors"]) - 1)
    assert sum(a["label"] == "ALL-IN" for a in st["anchors"]) == 1
    # summary / sizes / cells / combos all report the jam (was 0% all-in)
    assert d["summary"]["allin"]["freq"] > 0.99
    assert d["summary"]["raise"]["freq"] < 0.01
    assert d["sizes"][-1]["label"] == "ALL-IN" and d["sizes"][-1]["frac"] > 0.99
    aa = d["cells"]["AA"]
    assert aa["ai"] == pytest.approx(aa["r"], abs=1e-3) and aa["ai"] > 0.99
    combo = next(iter(d["combos"].values()))
    assert combo["ai"] == pytest.approx(combo["r"], abs=1e-3)


def test_deep_stack_allin_is_still_the_atom(jam):
    d = _q(jam, stack=100.0)
    st = d["state"]
    assert st["allin_k"] == NLH_ANCHOR_SPEC.count - 1
    assert st["anchors"][-1]["label"] == "ALL-IN"
    assert st["anchors"][-2]["label"] == "275%" and st["anchors"][-2]["allin"] is False
    assert d["summary"]["allin"]["freq"] > 0.99
    # allin stays a sub-share of the raise mass, never more
    for c in d["cells"].values():
        assert -1e-9 <= c["ai"] <= c["r"] + 1e-9


def test_raise_amounts_expose_raise_to_totals(rnd):
    # UTG opens to 2.5bb, folds to the BB (1bb already posted).
    line = [_a("raise", 2.5)] + [_a("fold")] * 4
    d = _q(rnd, line=line)
    st = d["state"]
    assert st["position"] == "BB" and st["to_call_bb"] == 1.5
    assert st["actor_commit_bb"] == 1.0
    # wire unit = raise-BY delta; display unit = raise-TO total
    assert st["min_raise_bb"] == 3.0 and st["min_raise_to_bb"] == 4.0
    assert st["max_raise_to_bb"] == pytest.approx(st["max_raise_bb"] + 1.0)
    assert st["max_raise_to_bb"] == 99.5  # the whole stack (100bb - 0.5bb ante)
    for a in st["anchors"]:
        assert a["to_bb"] == pytest.approx(a["chips_bb"] + 1.0)
    # the opener had nothing in: delta == total
    assert d["sequence"][0]["chips_bb"] == d["sequence"][0]["to_bb"] == 2.5

    # BB 3-bets TO 9bb == a delta of 8bb on the wire.
    d2 = _q(rnd, line=line + [_a("raise", 8.0)])
    ev = d2["sequence"][-1]
    assert ev["position"] == "BB" and ev["chips_bb"] == 8.0 and ev["to_bb"] == 9.0
    utg = d2["state"]
    assert utg["position"] == "UTG" and utg["to_call_bb"] == 6.5  # 9 - 2.5
    assert utg["actor_commit_bb"] == 2.5
    for s in d2["sizes"]:
        assert s["to_bb"] == pytest.approx(s["chips_bb"] + 2.5)


def test_sb_completes_and_raise_to_units(rnd):
    st = _q(rnd, line=[_a("fold")] * 4)["state"]
    assert st["position"] == "SB" and st["actor_commit_bb"] == 0.5
    assert st["min_raise_to_bb"] == pytest.approx(st["min_raise_bb"] + 0.5)
    assert st["min_raise_to_bb"] == 2.0  # a min-raise over the 1bb blind


def test_out_of_range_error_speaks_raise_to(rnd):
    line = [_a("raise", 2.5)] + [_a("fold")] * 4 + [_a("raise", 1.0)]
    r = rnd.post("/ranges/query", json={"seats": 6, "stack_bb": 100.0, "line": line})
    assert r.status_code == 400
    # 1bb delta from the BB == a raise TO 2bb; legal totals are [4, 99.5]bb
    assert "raise to 2bb outside [4, 99.5]bb" in r.json()["detail"]


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_non_finite_chips_bb_is_a_400(rnd, bad):
    body = (
        '{"seats":6,"stack_bb":100,"line":'
        '[{"t":"a","gate":"raise","chips_bb":%s}]}' % bad
    )
    r = rnd.post(
        "/ranges/query", content=body, headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 400, r.text
    assert "finite" in r.json()["detail"]


def test_huge_finite_chips_bb_is_a_400_not_a_500(rnd):
    r = rnd.post(
        "/ranges/query",
        json={"seats": 6, "stack_bb": 100.0, "line": [_a("raise", 1e300)]},
    )
    assert r.status_code == 400, r.text


# --- GTO teacher: per-node coverage, canonical obs, jam snap --------------------------
# (review 2026-09-20 follow-up: PolicyNetHost serves canonical obs / grid chips
# and answers supports() from recorded training coverage.)


def _cfg(seats: int, stack_bb: float):
    from plo5bp.config import GameConfig

    return GameConfig(
        num_seats=seats, starting_stack=int(round(stack_bb * BB)), ante=BB // 2,
        bb=BB, variant="nlh_single", sb=BB // 2,
    )


def _replay_line(env, line):
    """line entries: ("a", gate, chips) | ("cards", [..])."""
    obs = info = None
    for e in line:
        if e[0] == "a":
            obs, _r, _d, info = env.step_hybrid(e[1], e[2])
        else:
            aw = env.awaiting_next_street()
            if aw == 1:
                obs, info = env.set_flop_nlh(*e[1])
            elif aw == 2:
                obs, info = env.set_turn_nlh(e[1][0])
            else:
                obs, info = env.set_river_nlh(e[1][0])
    return obs, info


def _canonical_lines():
    from plo5bp.actions import GATE_CHECK_CALL as C, GATE_FOLD as F, GATE_RAISE as R

    # 3-handed: BTN opens, SB folds, BB calls -> flop bet/call -> turn -> river.
    flop3 = [("a", R, 25_000), ("a", F, 0), ("a", C, 0), ("cards", [2, 7, 12])]
    turn3 = flop3 + [("a", R, 30_000), ("a", C, 0), ("cards", [17])]
    river3 = turn3 + [("a", C, 0), ("a", R, 60_000), ("a", C, 0), ("cards", [22])]
    # heads-up: SB(button) raises, BB calls -> river with a bet to face.
    hu = [("a", R, 20_000), ("a", C, 0), ("cards", [3, 9, 14]),
          ("a", C, 0), ("a", C, 0), ("cards", [20]),
          ("a", C, 0), ("a", C, 0), ("cards", [27]), ("a", R, 40_000)]
    return [(3, flop3), (3, flop3 + [("a", R, 30_000)]), (3, turn3),
            (3, river3), (2, hu)]


def test_canonical_range_pack_is_bit_exact_vs_canonical_serve_obs():
    """The batched canonicalization == the serial one the GTO host serves
    (`gto.obs_from_label.canonical_serve_obs`), row for row — and it really
    differs from the live rows (prior-street history / hand-total commits /
    blind flags are in play on every fixture)."""
    import numpy as np

    from plo5bp.encoding_nlh import encode_observation_batch_nlh
    from plo5bp.env import BombPotEnv
    from plo5bp.gto.obs_from_label import canonical_serve_obs
    from plo5bp.ui.ranges import canonical_range_pack

    checked = 0
    for seats, line in _canonical_lines():
        cfg = _cfg(seats, 100.0)
        env = BombPotEnv(cfg)
        env.reset_study_nlh(button=0, hero_seat=0, hero_hole=[50, 51])
        _obs, info = _replay_line(env, line)
        actor = int(info.actor)
        assert int(info.raw_obs["street"]) >= 1
        board = {int(c) for c in info.raw_obs["board_a"]}
        combos = [c for c in ([0, 1], [44, 40], [33, 21], [48, 47], [30, 26])
                  if not (set(c) & board)]
        pack = env.pack_range_nlh(np.asarray(combos, dtype=np.uint8))
        c_pack, c_cfg = canonical_range_pack(pack, cfg.bb)
        got = encode_observation_batch_nlh(c_pack, c_pack["hero_cat_a"], c_cfg)
        live = encode_observation_batch_nlh(pack, pack["hero_cat_a"], cfg)
        # prior-street records were stripped; the input pack is not mutated
        assert int(pack["history_len"][0]) > int(c_pack["history_len"][0])
        for r, combo in enumerate(combos):
            e2 = BombPotEnv(cfg)
            e2.reset_study_nlh(button=0, hero_seat=actor, hero_hole=list(combo))
            _o2, i2 = _replay_line(e2, line)
            assert int(i2.actor) == actor
            want = canonical_serve_obs(i2.raw_obs, bb=cfg.bb)
            assert want is not None
            assert np.array_equal(got[r], want), (seats, len(line), combo)
            assert not np.array_equal(got[r], live[r])
            checked += 1
    assert checked >= 15


class _FakeGtoHost:
    """Duck-typed stand-in for PolicyNetHost (what ranges.py relies on)."""

    name = "fake_gto"
    ckpt_path = "checkpoints/fake_gto_policy.pt"

    def __init__(self, model, covered, canonical=True):
        self.model = model
        self.covered = covered
        self.serves_canonical_obs = canonical
        self.asked: list[tuple[int, int]] = []

    def supports(self, *, seats: int, street: int) -> bool:
        self.asked.append((int(seats), int(street)))
        return bool(self.covered(int(seats), int(street)))


def _gto_client(host=None, gto_model=None, ppo=None):
    ppo = ppo if ppo is not None else _model(jam=False)
    app = FastAPI()
    app.include_router(create_ranges_router(
        {"nlh_single": {"model": ppo, "loaded": True}}, torch.device("cpu"),
        nlh_ckpt_name="ppo_nlh.pt", gto_model=gto_model, gto_host=host,
    ))
    return TestClient(app, raise_server_exceptions=False), ppo


def _capture_inputs(model):
    seen: list = []
    model.register_forward_pre_hook(lambda _m, args: seen.append(args[0].clone()))
    return seen


HU_RIVER = [
    _a("raise", 2.0), _a("check_call"), {"t": "cards", "cards": [3, 9, 14]},
    _a("check_call"), _a("check_call"), {"t": "cards", "cards": [20]},
    _a("check_call"), _a("check_call"), {"t": "cards", "cards": [27]},
]


@pytest.mark.parametrize("canonical", [True, False])
def test_gto_host_serves_only_covered_nodes_on_its_training_obs(canonical):
    import numpy as np

    from plo5bp.actions import GATE_CHECK_CALL as C, GATE_RAISE as R
    from plo5bp.encoding_nlh import encode_observation_batch_nlh
    from plo5bp.env import BombPotEnv
    from plo5bp.ui.ranges import ALL_COMBOS, canonical_range_pack

    gto = _model(jam=True)  # jams everything: unmistakable vs the random PPO net
    host = _FakeGtoHost(
        gto, covered=lambda seats, street: seats == 2 and street == 3,
        canonical=canonical,
    )
    client, ppo = _gto_client(host=host)
    gto_in, ppo_in = _capture_inputs(gto), _capture_inputs(ppo)

    # --- 6-max preflop root: the river-only HU teacher must NOT paint it
    d = _q(client)
    assert d["model"]["backend"] == "ppo_fallback" == d["state"]["backend"]
    assert "6-handed preflop" in d["model"]["reason"]
    assert d["model"]["checkpoint"] == "ppo_nlh.pt" and d["model"]["obs_form"] == "live"
    assert (6, 0) in host.asked
    assert not gto_in and len(ppo_in) == 1
    assert d["summary"]["allin"]["freq"] < 0.9  # the PPO net, not the jam teacher

    # --- HU river: covered -> the teacher, on the obs form it was trained on
    ppo_in.clear()
    r = client.post(
        "/ranges/query", json={"seats": 2, "stack_bb": 100.0, "line": HU_RIVER}
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["state"]["street"] == "river"
    assert d["model"]["backend"] == "fake_gto" == d["state"]["backend"]
    assert d["model"]["checkpoint"] == "fake_gto_policy.pt" and d["model"]["loaded"] is True
    assert d["model"]["obs_form"] == ("canonical" if canonical else "live")
    assert d["model"]["reason"] is None
    assert d["summary"]["allin"]["freq"] > 0.99  # the teacher's policy
    assert len(gto_in) == 1
    # ... while the EARLIER nodes of the same line (the reach product) were
    # not covered, so they went to the PPO net.
    assert len(ppo_in) >= 1

    # what reached the teacher == the canonical (or live) rows, rebuilt here
    cfg = _cfg(2, 100.0)
    env = BombPotEnv(cfg)
    env.reset_study_nlh(button=0, hero_seat=0, hero_hole=[51, 50])
    _replay_line(env, [("a", R, 20_000), ("a", C, 0), ("cards", [3, 9, 14]),
                       ("a", C, 0), ("a", C, 0), ("cards", [20]),
                       ("a", C, 0), ("a", C, 0), ("cards", [27])])
    board = {3, 9, 14, 20, 27}
    holes = np.array(
        [c for c in ALL_COMBOS if not (set(c) & board)], dtype=np.uint8
    )
    pack = env.pack_range_nlh(holes)
    live = encode_observation_batch_nlh(pack, pack["hero_cat_a"], cfg)
    c_pack, c_cfg = canonical_range_pack(pack, cfg.bb)
    canon = encode_observation_batch_nlh(c_pack, c_pack["hero_cat_a"], c_cfg)
    assert not np.array_equal(live, canon)
    want = canon if canonical else live
    assert np.array_equal(gto_in[0].numpy(), want)


def test_bare_gto_model_without_its_host_is_not_served():
    gto = _model(jam=True)
    client, ppo = _gto_client(gto_model=gto)  # today's server.py call shape
    gto_in, ppo_in = _capture_inputs(gto), _capture_inputs(ppo)
    d = _q(client)
    assert not gto_in and len(ppo_in) == 1
    assert d["model"]["backend"] == "ppo_fallback"
    assert "without its host" in d["model"]["reason"]
    assert d["model"]["checkpoint"] == "ppo_nlh.pt"
    # no teacher configured at all: plain "ppo", no reason
    client2, _ = _gto_client()
    d2 = _q(client2)
    assert d2["model"]["backend"] == "ppo" and d2["model"]["reason"] is None
    # non-decision nodes report no serving network
    d3 = _q(client2, line=[_a("fold")] * 5)
    assert d3["terminal"] is True and d3["model"]["backend"] is None


def test_gto_nodes_use_grid_chips_with_the_hosts_jam_snap():
    """17bb UTG open: the 275% anchor (16.125bb) lands within JAM_DUST_BB of
    the 16.5bb stack. The GTO host plays that as the jam, so a GTO-served
    grid shows ONE all-in (both ladder indices merged); a PPO grid keeps the
    two sizes apart."""
    from plo5bp.ui.ranges import JAM_DUST_BB

    assert JAM_DUST_BB == 0.5
    host = _FakeGtoHost(
        _model(jam=False), covered=lambda seats, street: True, canonical=False
    )
    gto_client, _ = _gto_client(host=host)
    ppo_client, _ = _gto_client()

    g = _q(gto_client, stack=17.0)
    p = _q(ppo_client, stack=17.0)
    assert g["state"]["max_raise_bb"] == p["state"]["max_raise_bb"] == 16.5

    p_top = p["state"]["anchors"][-2:]
    assert [(a["label"], a["chips_bb"]) for a in p_top] == [
        ("275%", 16.125), ("ALL-IN", 16.5),
    ]
    assert all(len(a["ks"]) == 1 for a in p["state"]["anchors"])

    g_anchors = g["state"]["anchors"]
    assert [a["label"] for a in g_anchors].count("ALL-IN") == 1
    assert "275%" not in [a["label"] for a in g_anchors]
    top = g_anchors[-1]
    assert top["allin"] and top["chips_bb"] == 16.5 and top["ks"] == [10, 11]
    assert g["state"]["allin_k"] == 11
    # the merged button's histogram share is the sum of both indices, and the
    # all-in summary counts both
    assert abs(sum(s["frac"] for s in g["sizes"]) - 1.0) < 2e-3
    assert g["sizes"][-1]["ks"] == [10, 11]
    raise_mass = g["summary"]["raise"]["freq"] + g["summary"]["allin"]["freq"]
    assert g["summary"]["allin"]["freq"] == pytest.approx(
        g["sizes"][-1]["frac"] * raise_mass, abs=2e-3
    )


def test_real_policy_net_host_plugs_into_ranges():
    """The duck-typed contract above is the REAL PolicyNetHost's: a meta-less
    host claims only the legacy HU-river coverage and canonical obs."""
    from plo5bp.gto.policy_host import PolicyNetHost

    host = PolicyNetHost(model=_model(jam=True), device=torch.device("cpu"))
    assert host.supports(seats=2, street=3) and not host.supports(seats=6, street=0)
    assert host.serves_canonical_obs is True
    client, _ = _gto_client(host=host)

    d = _q(client)  # 6-max preflop
    assert d["model"]["backend"] == "ppo_fallback"

    r = client.post(
        "/ranges/query", json={"seats": 2, "stack_bb": 100.0, "line": HU_RIVER}
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["model"]["backend"] == host.name == "policy_net"
    assert d["model"]["obs_form"] == "canonical"
    assert d["model"]["checkpoint"] == "policy_net"  # no ckpt_path on this host
    assert d["summary"]["allin"]["freq"] > 0.99
    # HU flop of the same line: not covered -> PPO
    r = client.post(
        "/ranges/query",
        json={"seats": 2, "stack_bb": 100.0, "line": HU_RIVER[:3]},
    )
    assert r.json()["state"]["street"] == "flop"
    assert r.json()["model"]["backend"] == "ppo_fallback"
