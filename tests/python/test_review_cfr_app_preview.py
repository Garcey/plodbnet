"""Regression tests for the 2026-09-20 review, CFR desktop app — "preview tree ≠ solver".

The builder preview now mirrors the solver's public state in integer chips:
seat order, blinds/antes, BB option after a limp, min-raise clamp, de-dup of
sizes that clamp to the same chips, and all-in runouts.
"""

from __future__ import annotations

import pytest

from plo5bp.cfr_app.session import _config_from_dict, _root_from_dict
from plo5bp.cfr_app.tree_model import _position_names, _PreviewState, build_abstract_tree
from plo5bp.gto.cfr_api import rust_cfr_available

RIVER_BOARD = [12, 28, 38, 41, 45]
HU_PREFLOP = {"street": 0, "pot_bb": 2.5, "effective_stack_bb": 20, "board": [], "num_seats": 2,
              "raise_sizes_pm": [500, 1000]}
LOW_SPR = {"street": 3, "pot_bb": 10, "effective_stack_bb": 12, "board": RIVER_BOARD, "num_seats": 2,
           "raise_sizes_pm": [500, 1000, 1500]}


def child(node: dict, *actions: str) -> dict:
    for act in actions:
        node = next(c for c in node["children"] if c["label"].split(" ")[0].split("/")[-1] == act)
    return node


def all_nodes(node: dict):
    yield node
    for c in node.get("children") or []:
        yield from all_nodes(c)


def test_hu_preflop_first_actor_is_the_sb_in_the_solvers_seat_order():
    tree = build_abstract_tree(HU_PREFLOP, max_nodes=3000, max_depth=12)
    root = tree["tree"]
    assert (root["seat"], root["seat_label"]) == (1, "SB")  # was labelled p0
    assert tree["seat_labels"] == ["BB", "SB"]
    # Antes are dead money in the pot; only the blinds count toward the call.
    assert (root["pot_bb"], root["to_call_bb"], root["stack_bb"]) == (2.5, 0.5, 19.0)
    assert root["actions"] == ["FOLD", "CHECK_CALL", "RAISE_500", "RAISE_1000", "ALLIN"]


def test_hu_preflop_limp_goes_to_the_bb_option_not_to_showdown():
    root = build_abstract_tree(HU_PREFLOP, max_nodes=3000, max_depth=12)["tree"]
    option = child(root, "CHECK_CALL")
    assert option["terminal"] is False  # used to be a terminal "showdown/next"
    assert (option["seat"], option["seat_label"]) == (0, "BB")
    assert (option["pot_bb"], option["to_call_bb"]) == (3.0, 0.0)
    assert option["actions"] == ["CHECK_CALL", "RAISE_500", "RAISE_1000", "ALLIN"]  # no FOLD: nothing to call
    # BB checks behind → the round is closed; BB raises → back to the SB.
    assert child(option, "CHECK_CALL")["terminal_kind"] == "next_street"
    iso = child(option, "RAISE_1000")
    assert (iso["seat"], iso["pot_bb"], iso["to_call_bb"]) == (1, 6.0, 3.0)


def test_low_spr_sizes_that_clamp_to_the_same_chips_are_offered_once():
    root = build_abstract_tree(LOW_SPR, max_nodes=3000, max_depth=12)["tree"]
    # RAISE_1500 (15bb into a 12bb stack) IS the all-in: one action, one label.
    assert root["actions"] == ["CHECK_CALL", "RAISE_500", "RAISE_1000", "ALLIN"]
    facing = child(root, "RAISE_500")  # pot 15, 5 to call, 12 behind: any raise is a jam
    assert facing["actions"] == ["FOLD", "CHECK_CALL", "ALLIN"]
    for node in all_nodes(root):
        assert len(node.get("actions") or []) == len(set(node.get("actions") or []))


def test_min_raise_clamp_matches_the_solver_arithmetic():
    st = _PreviewState(_root_from_dict(LOW_SPR))
    st.apply("RAISE_500")  # bet 5bb
    bb = 10_000
    assert (st.to_call(), st.min_raise(), st.max_raise()) == (5 * bb, 10 * bb, 12 * bb)  # min raise = to 10bb
    assert st.raise_chips_for_pm(500) == 12 * bb  # 5 + 50% of 20 = 15bb → capped at the 12bb stack
    assert _PreviewState(_root_from_dict({**LOW_SPR, "raise_sizes_pm": [10]})).raise_chips_for_pm(10) == bb  # 1% pot → min bet 1bb


def test_facing_an_all_in_the_options_are_fold_or_call():
    root = build_abstract_tree(LOW_SPR, max_nodes=3000, max_depth=12)["tree"]
    facing = child(root, "ALLIN")
    assert facing["actions"] == ["FOLD", "CHECK_CALL"]  # was ["FOLD", "ALLIN"]
    assert child(facing, "CHECK_CALL")["terminal_kind"] == "allin_runout"
    assert child(facing, "FOLD")["terminal_kind"] == "fold"


def test_all_in_chains_end_in_a_runout_not_in_truncated():
    tree = build_abstract_tree(
        {"street": 0, "pot_bb": 1.5, "effective_stack_bb": 10, "board": [], "num_seats": 3,
         "raise_sizes_pm": [], "allin_atom": True, "ante_chips": 0, "stacks_bb": [10, 10, 10]},
        max_nodes=350, max_depth=7,
    )
    kinds = [n["terminal_kind"] for n in all_nodes(tree["tree"]) if n.get("terminal")]
    assert sorted(set(kinds)) == ["allin_runout", "fold"]  # used to bottom out in "(truncated)"
    assert kinds.count("allin_runout") == 4 and kinds.count("fold") == 3
    assert tree["num_nodes_built"] == 13
    root = tree["tree"]
    assert (root["seat"], root["seat_label"], root["actions"]) == (0, "BTN", ["FOLD", "ALLIN"])
    assert child(root, "ALLIN", "ALLIN", "ALLIN")["pot_bb"] == 30.0


def test_a_real_size_cap_is_still_reported_as_truncated():
    tree = build_abstract_tree(
        {"street": 3, "pot_bb": 2, "effective_stack_bb": 500, "board": RIVER_BOARD, "num_seats": 2,
         "size_preset": "fine"},
        max_nodes=40, max_depth=3,
    )
    assert any(n.get("terminal_kind") == "truncated" for n in all_nodes(tree["tree"]))


@pytest.mark.parametrize(
    "n,street,expected",
    [
        (2, 0, ["BB", "SB"]), (2, 3, ["OOP", "IP"]), (3, 0, ["BTN", "SB", "BB"]),
        (4, 0, ["CO", "BTN", "SB", "BB"]), (6, 0, ["UTG", "HJ", "CO", "BTN", "SB", "BB"]),
    ],
)
def test_position_names_follow_the_solvers_seat_order(n: int, street: int, expected: list[str]):
    assert _position_names(n, street) == expected


def test_clone_keeps_the_subclass():
    class Custom(_PreviewState):
        pass

    assert type(Custom(_root_from_dict(LOW_SPR)).clone()) is Custom


@pytest.mark.skipif(not rust_cfr_available(), reason="Rust CFR not built")
@pytest.mark.parametrize(
    "root_d,cfg_d",
    [
        (LOW_SPR, {"max_iterations": 60}),
        (HU_PREFLOP, {"max_iterations": 400, "algorithm": "mccfr_es", "seed": 1}),
    ],
    ids=["river_low_spr", "hu_preflop"],
)
def test_preview_state_matches_the_real_solver_node_for_node(root_d: dict, cfg_d: dict):
    """Replay every solver node's path; actor / pot / call / stacks / raise
    bounds must be identical to the chip, and the menus equal.

    One label rule changed solver-side in this same review (D11): a size that
    clamps to the maximum raise IS the all-in and is no longer listed next to
    ALLIN. The installed extension may predate that, so such "twin jam" labels
    are dropped from the SOLVER's menu before comparing, and solver paths that
    run through one are skipped. Everything else must match under either binary.
    """
    from plo5bp.gto.cfr_api import solve

    root = _root_from_dict(root_d)
    rep = solve(root, _config_from_dict({**cfg_d, "target_exploitability_bb": 0, "poll_every": 100000}))
    sizes = [int(pm) for pm in root.raise_sizes_pm]
    seen: dict[tuple[str, ...], dict] = {}
    for x in rep.strategy["infosets"]:
        if x["street"] == root.street:  # the preview covers the root street's round
            seen.setdefault(tuple(x["path"]), x)
    assert len(seen) >= 10

    def is_twin_jam(st: _PreviewState, action: str) -> bool:
        return action.startswith("RAISE_") and st.raise_chips_for_pm(int(action[6:])) == st.max_raise()

    compared = 0
    for path, x in seen.items():
        st = _PreviewState(root)
        through_twin = False
        for act in path:
            through_twin |= is_twin_jam(st, act)
            st.apply(act)
        if through_twin:
            continue
        compared += 1
        assert st.actor == x["actor"], path
        assert (st.pot, st.to_call(), st.stacks) == (x["pot_chips"], x["to_call_chips"], x["stacks_chips"]), path
        assert (st.min_raise(), st.max_raise()) == (x["min_raise_chips"], x["max_raise_chips"]), path
        solver_menu = [a for a in x["actions"] if not (root.allin_atom and is_twin_jam(st, a))]
        assert st.menu(sizes, root.allin_atom) == solver_menu, path
    assert compared >= 10
