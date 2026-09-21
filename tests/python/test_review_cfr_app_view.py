"""Regression tests for the 2026-09-20 review, CFR desktop app — strategy viewer.

E3 root node · E5 labels under isomorphism · E6 runouts · E7/E9 weighted
aggregates · E8 line navigation · hand filter · path labels · NaN.

Rows are synthetic dump-schema-v2 infosets (the shape py_api.rs emits), so none
of this needs a solve; one small real solve at the end pins the real shape.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from plo5bp.cfr_app.strategy_view import (
    build_class_matrix_from_combos,
    combo_label,
    filter_rows,
    hand_matcher,
    humanize_path,
    infoset_row,
    load_report,
    node_rows,
    normalize_path,
    row_weights,
)
from plo5bp.cfr_app.tree_model import aggregate_node, build_line_nav, compare_strategies
from plo5bp.gto.cfr_api import rust_cfr_available
from plo5bp.gto.preflop_class import cards_to_combo

RANKS, SUITS = "23456789TJQKA", "cdhs"
RIVER_BOARD = [12, 28, 38, 41, 45]  # 5c 9c Jh Qd Kd
TURN_BOARD = RIVER_BOARD[:4]


def card(s: str) -> int:
    return RANKS.index(s[0]) * 4 + SUITS.index(s[1])


def combo(hand: str) -> int:
    return cards_to_combo(card(hand[:2]), card(hand[2:]))


def v2(
    hand: str,
    path: list[str],
    probs: list[float],
    *,
    actor: int = 0,
    actions: list[str] | None = None,
    board: list[int] | None = None,
    mass: float = 1.0,
    iso_id: int | None = None,
) -> dict:
    """One dump-schema-v2 infoset. ``iso_id`` != raw combo mimics isomorphism on."""
    raw = combo(hand)
    priv = raw if iso_id is None else iso_id
    return {
        "infoset_id": f"p{actor}_h{7000000000 + len(path)}_c{priv}",
        "actions": actions or ["CHECK_CALL", "RAISE_500", "ALLIN"],
        "probs": probs,
        "schema_version": 2,
        "street": 3,
        "actor": actor,
        "board": list(board if board is not None else RIVER_BOARD),
        "path": list(path),
        "private_kind": "combo",
        "private_id": priv,
        "raw_combo": raw,
        "iso_id": priv,
        "visit_mass": mass,
    }


def report(infosets: list[dict], board: list[int] | None = None, street: int = 3) -> dict:
    return {
        "status": "ok",
        "root": {"street": street, "board": list(board if board is not None else RIVER_BOARD), "num_seats": 2},
        "config": {},
        "iterations_run": 10,
        "exploitability_bb": 0.5,
        "notes": [],
        "strategy": {"root_id": "t", "schema_version": 2, "infosets": infosets},
    }


def _river_tree() -> dict:
    """Root (P0) with two hands, a deep P1 node with MORE rows than the root."""
    infos = [
        v2("AhAs", [], [0.2, 0.3, 0.5]),
        v2("7h7s", [], [1.0, 0.0, 0.0]),
    ]
    for hand in ("AhAs", "7h7s", "KhQh", "2c2d", "8d8h"):
        infos.append(v2(hand, ["RAISE_500"], [0.5, 0.5], actor=1, actions=["FOLD", "CHECK_CALL"]))
    infos.append(v2("AhAs", ["CHECK_CALL"], [1.0, 0.0, 0.0], actor=1))
    return report(infos)


# --------------------------------------------------------------------------- E3


def test_e3_root_has_one_spelling():
    assert {normalize_path(p) for p in ("", "root", "open", None, "  ")} == {"open"}
    assert normalize_path("CHECK_CALL,RAISE_500") == "CHECK_CALL,RAISE_500"
    assert infoset_row(v2("AhAs", [], [1.0, 0.0, 0.0]))["path"] == "open"  # v2 `path: []`


def test_e3_first_load_shows_the_root_not_the_biggest_node():
    view = load_report(_river_tree())
    nav = view["line_nav"]
    assert (nav["root_path"], nav["root_seat"]) == ("open", 0)
    # The P1 node has 5 rows vs the root's 2 — "most rows" used to win.
    assert (view["matrix"]["seat"], view["matrix"]["path"]) == (0, "open")
    assert (view["nodes"][0]["seat"], view["nodes"][0]["path"]) == (0, "open")


@pytest.mark.parametrize("spelling", ["open", "root", ""])
def test_e3_every_root_spelling_selects_the_root_rows(spelling: str):
    view = load_report(_river_tree())
    page = filter_rows(view["rows"], seat=0, path=spelling)
    assert page["total"] == 2
    assert {r["hand_label"] for r in page["rows"]} == {combo_label(combo("AhAs")), combo_label(combo("7h7s"))}
    assert filter_rows(view["rows"])["total"] == 8  # path=None still means "every node"


def test_e3_api_first_load_then_root_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    from plo5bp.cfr_app import server
    from plo5bp.cfr_app.session import SolveSession

    monkeypatch.setenv("CFR_APP_DATA_DIR", str(tmp_path / "cfr_data"))
    monkeypatch.setenv("CFR_APP_ALLOWED_HOSTS", "testserver")  # TestClient's Host
    monkeypatch.setattr(server, "session", SolveSession(solve_fn=lambda r, c: None))
    monkeypatch.setitem(server._view_cache, "key", None)
    f = tmp_path / "cfr_data" / "app_export" / "tree.json"
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps(_river_tree()), encoding="utf-8")

    client = TestClient(server.app)
    first = client.get("/api/view", params={"path": str(f)}).json()
    assert (first["matrix"]["seat"], first["matrix"]["path"]) == (0, "open")
    nav = first["line_nav"]
    # What the JS sends next (it selects line_nav.root_*): used to return 0 rows.
    second = client.get(
        "/api/view", params={"path": str(f), "path_filter": nav["root_path"], "seat": nav["root_seat"]}
    ).json()
    assert second["page"]["total"] == 2
    assert second["matrix"]["empty"] is False


# --------------------------------------------------------------------------- E5


def test_e5_label_comes_from_raw_combo_not_the_iso_id():
    # iso id = the combo "KdTh", which holds a BOARD card (Kd); the real hand is KhTd.
    row = infoset_row(v2("KhTd", [], [1.0, 0.0, 0.0], iso_id=combo("KdTh")), root_board_len=5)
    assert row["hand_label"] == combo_label(combo("KhTd"))
    assert row["combo"] == combo("KhTd")
    assert row["private"] == combo("KdTh")  # infoset identity is untouched


def test_e5_never_shows_a_hand_holding_a_board_card():
    raw = v2("KhTd", [], [1.0, 0.0, 0.0], iso_id=combo("KdTh"))
    raw["raw_combo"] = None  # an old dump: only the iso id is known
    row = infoset_row(raw, root_board_len=5)
    assert row["combo"] is None
    assert row["hand_label"] == f"iso#{combo('KdTh')}"
    assert "Kd" not in row["hand_label"]


def test_e5_class_matrix_and_compare_use_the_real_hand():
    rows = [infoset_row(v2("AhAs", [], [0.0, 0.0, 1.0], iso_id=combo("7c2d")), root_board_len=5)]
    cells = [c for line in build_class_matrix_from_combos(rows)["cells"] for c in line if c]
    assert [c["label"] for c in cells] == ["AA"]  # not 72o

    a = [infoset_row(v2("AhAs", [], [1.0, 0.0, 0.0], iso_id=5), root_board_len=5)]
    b = [infoset_row(v2("AhAs", [], [0.0, 0.0, 1.0], iso_id=9), root_board_len=5)]  # other iso id, same hand
    diff = compare_strategies(a, b)
    assert diff["num_common"] == 1 and diff["top_diffs"][0]["l1"] == 2.0


def test_e5_non_combo_kinds_are_not_labelled_as_hands():
    raw = v2("AhAs", [], [1.0, 0.0, 0.0])
    raw.update(private_kind="ochs_bucket", private_id=17, raw_combo=None)
    row = infoset_row(raw, root_board_len=3)
    assert row["hand_label"] == "ochs bucket 17" and row["combo"] is None


# --------------------------------------------------------------------------- E6


def _turn_tree() -> dict:
    """Turn root; the river node after check-check exists on two river cards."""
    r3s, r5h = card("3s"), card("5h")
    line = ["CHECK_CALL", "CHECK_CALL"]
    return report(
        [
            v2("AhAs", [], [1.0, 0.0, 0.0], board=TURN_BOARD),
            # river 3s: visited more → the default runout
            v2("AhAs", line, [0.0, 0.0, 1.0], board=TURN_BOARD + [r3s], mass=9.0),
            v2("7h7s", line, [0.0, 0.0, 1.0], board=TURN_BOARD + [r3s], mass=1.0),
            # river 5h: same hand, opposite strategy
            v2("AhAs", line, [1.0, 0.0, 0.0], board=TURN_BOARD + [r5h], mass=2.0),
        ],
        board=TURN_BOARD,
        street=2,
    )


def test_e6_a_node_is_one_runout_never_a_blend():
    view = load_report(_turn_tree())
    line = "CHECK_CALL,CHECK_CALL"
    r3s, r5h = str(card("3s")), str(card("5h"))

    default = filter_rows(view["rows"], seat=0, path=line)
    assert default["runout"] == r3s and default["runout_label"] == "3♠"
    assert default["total"] == 2
    assert {tuple(r["board"]) for r in default["rows"]} == {tuple(TURN_BOARD + [card("3s")])}
    assert [o["key"] for o in default["runouts"]] == [r3s, r5h]  # most visited first

    other = filter_rows(view["rows"], seat=0, path=line, runout=r5h)
    assert other["total"] == 1 and other["rows"][0]["probs"] == [1.0, 0.0, 0.0]

    # The node list aggregates the default runout only: AA jams 100% on the 3s.
    node = next(n for n in view["nodes"] if n["path"] == line)
    assert node["aggregate"]["allin"] == 1.0  # a blend with the 5h row would be < 1
    assert node["num_runouts"] == 2 and "3♠" in node["label"]
    # Root street: no runout, no picker.
    root = filter_rows(view["rows"], seat=0, path="open")
    assert root["runout"] == "" and root["runouts"] == []
    # An unknown key falls back to the default instead of returning nothing.
    assert filter_rows(view["rows"], seat=0, path=line, runout="51")["runout"] == r3s


def test_e6_compare_does_not_collide_across_runouts():
    rows = load_report(_turn_tree())["rows"]
    assert compare_strategies(rows, rows)["num_common"] == 4  # was 3: the two AhAs river rows collided


# ------------------------------------------------------------------------ E7 / E9


def test_e7_node_mix_is_weighted_by_visit_mass():
    rows = [
        infoset_row(v2("AhAs", [], [0.0, 0.0, 1.0], mass=9.0), root_board_len=5),
        infoset_row(v2("7h7s", [], [1.0, 0.0, 0.0], mass=1.0), root_board_len=5),
        # never reached: still the 1/n default — must not dilute the mix
        infoset_row(v2("2c2d", [], [1 / 3, 1 / 3, 1 / 3], mass=0.0), root_board_len=5),
    ]
    agg = aggregate_node(rows)
    assert agg["allin"] == pytest.approx(0.9)  # uniform mean would say 0.4444
    assert agg["call"] == pytest.approx(0.1)
    assert agg["num_hands"] == 3


def test_e7_class_rows_without_mass_fall_back_to_combo_counts():
    def cls(cid: int, fold: float) -> dict:
        return infoset_row({"infoset_id": f"pf_p0_h1_c{cid}", "actions": ["FOLD", "RAISE_500"], "probs": [fold, 1 - fold]})

    rows = [cls(12, 0.0), cls(90, 0.0), cls(168, 1.0)]  # AA raise, AKs raise, AKo fold
    assert row_weights(rows) == [6.0, 4.0, 12.0]
    # aggregate_node rounds to 4 dp. Combo-weighted 12/22 = 0.5455; uniform mean was 0.3333.
    assert aggregate_node(rows)["fold"] == pytest.approx(12 / 22, abs=1e-4)


def test_e8_aggregate_keeps_the_solvers_action_order():
    rows = [infoset_row(v2("AhAs", [], [0.1, 0.2, 0.7]), root_board_len=5)]
    assert aggregate_node(rows)["actions"] == ["CHECK_CALL", "RAISE_500", "ALLIN"]  # not by frequency


def test_e9_cell_detail_is_the_weighted_cell_average_not_one_combo():
    rows = [
        infoset_row(v2("AhAs", [], [0.0, 0.0, 1.0], mass=3.0), root_board_len=5),
        infoset_row(v2("AcAs", [], [1.0, 0.0, 0.0], mass=1.0), root_board_len=5),  # iterated last
    ]
    cell = next(c for line in build_class_matrix_from_combos(rows)["cells"] for c in line if c)
    detail = {s["action"]: s["prob"] for s in cell["strategy"]}
    assert detail == pytest.approx({"CHECK_CALL": 0.25, "RAISE_500": 0.0, "ALLIN": 0.75})
    assert cell["agg"] == pytest.approx(0.75) and cell["call"] == pytest.approx(0.25)
    assert cell["n_combos"] == 2
    assert all({"short", "css", "pct"} <= set(s) for s in cell["strategy"])  # renderable as-is


# --------------------------------------------------------------------------- E8


def test_e8_native_dump_lines_are_navigable():
    nav = load_report(_river_tree())["line_nav"]
    acts = {a["action"]: a for a in nav["by_path"]["open"]["actions"]}
    assert list(acts) == ["CHECK_CALL", "RAISE_500", "ALLIN"]  # solver order in the strip
    assert (acts["RAISE_500"]["next_path"], acts["RAISE_500"]["has_next"]) == ("RAISE_500", True)
    assert acts["RAISE_500"]["next_seat"] == 1 and acts["RAISE_500"]["terminal"] is False
    assert acts["CHECK_CALL"]["next_path"] == "CHECK_CALL" and acts["CHECK_CALL"]["has_next"] is True
    assert acts["ALLIN"]["terminal"] is True  # no node after it in this dump
    assert acts["ALLIN"]["next_path"] == "ALLIN"  # full-label style, not "AI"


def test_e8_chart_files_still_use_short_tokens():
    nodes = [
        {"seat": 0, "path": "open", "aggregate": {"actions": ["FOLD", "ALLIN"], "mean_mix": {"FOLD": 0.4, "ALLIN": 0.6}}},
        {"seat": 1, "path": "AI", "aggregate": {"actions": ["FOLD", "ALLIN"], "mean_mix": {"FOLD": 0.7, "ALLIN": 0.3}}},
    ]
    acts = {a["token"]: a for a in build_line_nav(nodes, street=0)["by_path"]["open"]["actions"]}
    assert (acts["AI"]["next_path"], acts["AI"]["has_next"]) == ("AI", True)
    assert (acts["F"]["next_path"], acts["F"]["terminal"]) == ("F", True)


def test_e8_legacy_hash_nodes_do_not_collapse_onto_open():
    nodes = [
        {"seat": 0, "path": "12222850381629986102", "aggregate": {}},
        {"seat": 1, "path": "10001907875208514704", "aggregate": {}},
    ]
    nav = build_line_nav(nodes, street=3)
    assert set(nav["by_path"]) == {"12222850381629986102", "10001907875208514704"}
    assert nav["navigable"] is False and nav["root_path"] in nav["by_path"]


# ---------------------------------------------------------------- filter / labels


def _filter_rows() -> list[dict]:
    # All four are POSSIBLE on 5c 9c Jh Qd Kd. (A hand holding a board card — e.g.
    # AhKd here — is deliberately never labelled as a hand, see the E5 tests.)
    hands = ["AsKs", "AhKc", "AdAh", "7c2d"]
    return [infoset_row(v2(h, [], [1.0, 0.0, 0.0]), root_board_len=5) for h in hands]


@pytest.mark.parametrize(
    "query,expected",
    [
        ("AsKs", {"AsKs"}), ("KsAs", {"AsKs"}), ("kSaS", {"AsKs"}),  # either card order, any case
        ("AKs", {"AsKs"}), ("aks", {"AsKs"}), ("AKo", {"AhKc"}), ("KAo", {"AhKc"}),
        ("AK", {"AsKs", "AhKc"}),  # no suffix = suited + offsuit
        ("AA", {"AdAh"}), ("AhAd", {"AdAh"}), ("AdAh", {"AdAh"}),
        ("72o", {"7c2d"}), ("AAs", set()), ("QQ", set()), ("AsAs", set()),
    ],
)
def test_hand_filter_matches_both_card_orders_and_class_tokens(query: str, expected: set[str]):
    got = {r["hand_label"] for r in filter_rows(_filter_rows(), hand_query=query)["rows"]}
    assert got == {combo_label(combo(h)) for h in expected}


def test_hand_filter_on_class_rows_and_substring_fallback():
    cls = [
        infoset_row({"infoset_id": f"pf_p0_h1_c{cid}", "actions": ["FOLD"], "probs": [1.0]})
        for cid in (12, 90, 168)  # AA, AKs, AKo
    ]
    assert [r["hand_label"] for r in filter_rows(cls, hand_query="AK")["rows"]] == ["AKs", "AKo"]
    assert [r["hand_label"] for r in filter_rows(cls, hand_query="AsKs")["rows"]] == ["AKs"]
    assert filter_rows(cls, hand_query="pf_p0")["total"] == 3  # infoset-id substring still works
    assert hand_matcher("   ") is None


def test_path_labels_are_not_mangled():
    assert humanize_path("CHECK_CALL,RAISE_500") == "X/C → R50%"  # was "CHECK → CALL → RAISE → 500"
    assert humanize_path("RAISE_1000,ALLIN,FOLD") == "R100% → AI → F"
    assert humanize_path("AI_F") == "AI → F"  # legacy underscore-separated tokens
    assert humanize_path("") == humanize_path("root") == humanize_path("open") == "Open"


# ----------------------------------------------------------------------------- NaN


def test_nan_in_a_report_cannot_break_the_json_response():
    rep = json.loads(
        '{"status":"ok","root":{"street":3,"board":[0,5,10,15,20]},"exploitability_bb":NaN,'
        '"strategy":{"infosets":[{"infoset_id":"p0_h77777777_c100","actions":["CHECK_CALL","ALLIN"],'
        '"probs":[NaN,1.0],"visit_mass":Infinity}]}}'
    )
    from plo5bp.cfr_app.server import _view_payload
    from plo5bp.cfr_app.solve_worker import sanitize_json

    view = load_report(sanitize_json(rep))
    row = view["rows"][0]
    assert row["probs"] == [0.0, 1.0] and row["visit_mass"] is None
    json.dumps(_view_payload(view), allow_nan=False)  # what Starlette's JSONResponse enforces
    # Even an unsanitized dict must not leak NaN out of a row.
    assert all(math.isfinite(p) for p in load_report(rep)["rows"][0]["probs"])


# ------------------------------------------------------------------- real solve


@pytest.mark.skipif(not rust_cfr_available(), reason="Rust CFR not built")
def test_real_river_dump_with_isomorphism_end_to_end():
    from plo5bp.cfr_app.session import _config_from_dict, _root_from_dict
    from plo5bp.gto.cfr_api import solve

    rep = solve(
        _root_from_dict({"street": 3, "pot_bb": 10, "effective_stack_bb": 50, "board": RIVER_BOARD,
                         "num_seats": 2, "raise_sizes_pm": [500, 1000]}),
        _config_from_dict({"max_iterations": 40, "thread_num": 1, "target_exploitability_bb": 0,
                           "use_isomorphism": True, "poll_every": 100000}),
    ).as_dict()
    raw = {x["infoset_id"]: x for x in rep["strategy"]["infosets"]}
    view = load_report(rep)
    board = set(RIVER_BOARD)

    assert any(r["private"] != r["combo"] for r in view["rows"]), "iso should relabel some hands"
    for r in view["rows"]:
        assert r["hand_label"] == combo_label(raw[r["infoset_id"]]["raw_combo"])  # E5
        assert not ({card(r["hand_label"][:2]), card(r["hand_label"][2:])} & board)

    nav = view["line_nav"]
    assert (nav["root_path"], nav["root_seat"]) == ("open", 0)  # E3
    root = node_rows(view["rows"], seat=0, path="open")["rows"]
    assert root and all(raw[r["infoset_id"]]["path"] == [] for r in root)
    assert (view["matrix"]["seat"], view["matrix"]["path"]) == (0, "open")
    acts = nav["by_path"]["open"]["actions"]  # E8
    assert [a["action"] for a in acts] == raw[root[0]["infoset_id"]]["actions"]
    assert all(a["has_next"] for a in acts)
