"""Regression tests for the 2026-09-20 review, CFR desktop app — E11 range text.

The native parser silently mis-read most real-world range text (``KK-TT`` → KK,
whitespace-separated → first token, ``AA:0.5`` → dropped / 100% range) and —
found while fixing this — reads the PAIRS ``22``..``99`` as combo ids. The app
now parses strictly and hands the solver a canonical string.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from plo5bp.cfr_app.ranges import RangeError, apply_ranges, parse_range, toggle_class
from plo5bp.gto.cfr_api import rust_cfr_available
from plo5bp.gto.preflop_class import (
    combo_to_cards,
    preflop_class_from_cards,
    preflop_class_label,
)

BOARD = [0, 5, 10, 15, 20]  # 2c 3d 4h 5s 7c


def classes(pr) -> set[str]:
    return {preflop_class_label(preflop_class_from_cards(*combo_to_cards(c))) for c in pr.weights}


@pytest.fixture(autouse=True)
def _isolated_cfr_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Never touch the real data/cfr (review J4)."""
    from plo5bp.cfr_app import server
    from plo5bp.cfr_app.session import SolveSession

    monkeypatch.setenv("CFR_APP_DATA_DIR", str(tmp_path / "cfr_data"))
    monkeypatch.setenv("CFR_APP_ALLOWED_HOSTS", "testserver")  # TestClient's Host
    monkeypatch.setattr(server, "session", SolveSession())
    yield


# ------------------------------------------------------------------ grammar


@pytest.mark.parametrize(
    "text,expected,combos",
    [
        ("AA,KK,QQ", {"AA", "KK", "QQ"}, 18),
        ("AA KK QQ", {"AA", "KK", "QQ"}, 18),  # whitespace separates (native: AA only)
        ("AA\nKK\tQQ , JJ", {"AA", "KK", "QQ", "JJ"}, 24),
        ("KK-TT", {"KK", "QQ", "JJ", "TT"}, 24),  # native: KK only
        ("TT-KK", {"KK", "QQ", "JJ", "TT"}, 24),  # either order
        ("QQ+", {"QQ", "KK", "AA"}, 18),
        ("A5s-A2s", {"A5s", "A4s", "A3s", "A2s"}, 16),
        ("T9s-76s", {"T9s", "98s", "87s", "76s"}, 16),  # same gap: both cards step
        ("A5s+", {f"A{k}s" for k in "56789TJQK"}, 36),
        ("76s+", {"76s"}, 4),  # top card fixed, kicker and better
        ("AK", {"AKs", "AKo"}, 16),
        ("aks", {"AKs"}, 4),
        ("AsKs", {"AKs"}, 1),
        ("KsAs", {"AKs"}, 1),
    ],
)
def test_e11_grammar(text: str, expected: set[str], combos: int):
    pr = parse_range(text, [])
    assert classes(pr) == expected
    assert pr.combos == combos and pr.weight == combos


@pytest.mark.parametrize("pair", list("23456789TJQKA"))
def test_e11_numeric_pairs_are_pairs_not_combo_ids(pair: str):
    # "22" used to mean combo id 22 (one 32o hand) — natively it still does,
    # which is why canonical() must never emit a bare all-digit token.
    pr = parse_range(pair * 2, [])
    assert classes(pr) == {pair * 2} and pr.combos == 6
    assert not any(tok.isdigit() for tok in pr.canonical().split(","))
    assert parse_range(pr.canonical(), []).weights == pr.weights


def test_e11_combo_ids_are_three_plus_digits_so_nothing_is_ambiguous():
    assert classes(parse_range("0022", [])) == {"32o"} and parse_range("0022", []).combos == 1
    assert classes(parse_range("72", [])) == {"72s", "72o"}  # two digits = hand text
    for bad in ("5", "12", "9999"):
        with pytest.raises(RangeError):
            parse_range(bad, [])


def test_e11_weights_and_dedupe():
    pr = parse_range("AA:0.5,KK", [])
    assert pr.combos == 12 and pr.weight == pytest.approx(9.0)
    assert pr.class_weights() == {"KK": 1.0, "AA": 0.5}
    # De-duplicated: a combo counts once (native: AA ended up at 3x KK's weight).
    assert parse_range("AA,AA,AA,KK", []).weights == parse_range("AA,KK", []).weights
    # Later tokens refine earlier ones: the class at 0.5, one combo at 1.
    refined = parse_range("AA:0.5,AsAh", [])
    assert refined.combos == 6 and refined.weight == pytest.approx(3.5)


@pytest.mark.parametrize("text", ["", "   ", "random", "RANDOM", "100%"])
def test_e11_full_range_spellings(text: str):
    pr = parse_range(text, BOARD)
    assert pr.full and pr.canonical() == ""
    assert pr.summary()["combos"] == 1081  # C(47, 2) on a 5-card board


@pytest.mark.parametrize(
    "text,needle",
    [
        ("AKx", "unknown range token 'AKx'"),
        ("AA,banana", "unknown range token 'banana'"),
        ("AA:1.5", "weight must be > 0 and <= 1"),
        ("AA:0", "weight must be > 0 and <= 1"),
        ("AA:x", "is not a number"),
        ("AAs", "a pair cannot be suited/offsuit"),
        ("KK-AKs", "both ends must be"),
        ("KK-AK", "cannot mix a pair with a non-pair"),
        ("A5s-K2s", "must share the top card"),
        ("QQ+-TT", "cannot be combined"),
        ("AsAs", "the same card twice"),
        ("random,AA", "must stand alone"),
    ],
)
def test_e11_bad_text_is_an_error_never_a_silent_full_range(text: str, needle: str):
    with pytest.raises(RangeError, match=re.escape(needle)):
        parse_range(text, [])


def test_e11_board_blocks_combos_and_an_empty_range_is_an_error():
    pr = parse_range("22", BOARD)  # 2c is on the board
    assert pr.combos == 3
    assert all(0 not in combo_to_cards(c) for c in pr.weights)
    # Every combo blocked: the native side would fall back to 100% here.
    with pytest.raises(RangeError, match="no combos left on this board"):
        parse_range("2c3d", BOARD)


def test_e11_canonical_uses_only_forms_the_native_parser_handles():
    pr = parse_range("KK-TT,AKs,AA:0.5,AsQs", [])
    parts = pr.canonical().split(",")
    labels = [p for p in parts if ":" not in p]
    ids = [p for p in parts if ":" in p]
    assert set(labels) == {"KK", "QQ", "JJ", "TT", "AKs"}  # whole classes at weight 1
    assert len(ids) == 7  # 6x AA at 0.5 + AsQs
    assert all(len(p.split(":")[0]) == 4 and p.split(":")[0].isdigit() for p in ids)  # zero-padded
    assert parse_range(pr.canonical(), []).weights == pr.weights
    assert parse_range(pr.normalized(), []).weights == pr.weights
    assert "AA:0.5" in pr.normalized().split(",")  # readable form keeps class tokens


def test_e11_grid_toggle_uses_the_same_semantics():
    assert classes(toggle_class("QQ+", "KK", [])) == {"QQ", "AA"}  # KK removed from a '+' range
    assert toggle_class("QQ+", "KK", []).normalized() == "QQ,AA"
    assert classes(toggle_class("AA", "AKs", [])) == {"AA", "AKs"}
    assert toggle_class("AsKs", "AKs", []).combos == 0  # partly present → cleared
    off = toggle_class("", "AA", [])  # full range minus one class
    assert off.combos == 1326 - 6 and "AA" not in classes(off)
    with pytest.raises(RangeError):
        toggle_class("AA", "AK", [])  # a grid cell is ONE class


def test_e11_apply_ranges_is_idempotent_and_keeps_the_users_wording():
    once, info = apply_ranges({"board": BOARD, "range_oop": "KK-TT  AA:0.5", "range_ip": "random"})
    assert once["range_oop_text"] == "KK-TT  AA:0.5" and once["range_ip"] == "" and once["range_ip_text"] == ""
    assert info["oop"]["combos"] == 30 and info["ip"]["full"] is True
    twice, info2 = apply_ranges(once)  # server handler, then session.start
    assert twice == once and info2 == info
    with pytest.raises(RangeError, match="^range_ip: unknown range token"):
        apply_ranges({"board": BOARD, "range_oop": "AA", "range_ip": "nope"})


# ---------------------------------------------------------------------- API


def test_e11_api_validate_solve_and_grid_endpoint():
    from fastapi.testclient import TestClient

    from plo5bp.cfr_app import server

    client = TestClient(server.app)
    root = {"street": 3, "pot_bb": 10, "effective_stack_bb": 10, "board": BOARD, "num_seats": 2}

    ok = client.post("/api/validate_root", json={**root, "range_oop": "KK-TT", "range_ip": ""}).json()
    assert ok["ok"] is True
    assert ok["ranges"]["oop"]["combos"] == 24 and set(ok["ranges"]["oop"]["classes"]) == {"KK", "QQ", "JJ", "TT"}
    assert ok["ranges"]["ip"]["full"] is True

    bad = client.post("/api/validate_root", json={**root, "range_oop": "AKx"}).json()
    assert bad["ok"] is False and "range_oop: unknown range token 'AKx'" in bad["error"]

    s = client.post("/api/solve", json={"root": {**root, "range_ip": "AA:2"}, "config": {"max_iterations": 1}})
    assert s.status_code == 400 and "range_ip" in s.json()["detail"]
    assert client.get("/api/jobs").json()["jobs"] == []  # rejected before anything started

    grid = client.post("/api/range/parse", json={"text": "QQ+", "board": BOARD}).json()
    assert grid["ok"] and grid["classes"] == {"QQ": 1.0, "KK": 1.0, "AA": 1.0}
    flipped = client.post("/api/range/parse", json={"text": "QQ+", "board": BOARD, "toggle": "KK"}).json()
    assert flipped["ok"] and flipped["normalized"] == "QQ,AA"
    err = client.post("/api/range/parse", json={"text": "QQ+ zz", "board": BOARD}).json()
    assert err["ok"] is False and "'zz'" in err["error"]


def test_e11_session_sends_canonical_and_keeps_the_text(tmp_path: Path):
    from plo5bp.cfr_app.session import SolveSession
    from plo5bp.gto.cfr_api import SolveReport

    seen = {}

    def solve_fn(root, config):
        seen["oop"], seen["ip"] = root.range_oop, root.range_ip
        return SolveReport(
            status="ok", root=root.as_dict(), config=config.as_dict(),
            strategy={"root_id": root.root_id, "infosets": []},
            iterations_run=1, exploitability_bb=None, notes=[],
        )

    sess = SolveSession(work_dir=tmp_path, solve_fn=solve_fn)
    job = sess.start(
        {"street": 3, "pot_bb": 10, "effective_stack_bb": 10, "board": BOARD,
         "range_oop": "KK-TT 22", "range_ip": "random"},
        {"max_iterations": 1}, save=False,
    )
    deadline = time.time() + 10
    while sess.get_job(job.job_id, full=False)["status"] in ("queued", "running") and time.time() < deadline:
        time.sleep(0.02)
    final = sess.get_job(job.job_id, full=True)

    assert final["status"] == "done"
    assert seen["ip"] == ""  # "random" → the solver's own uniform range
    assert parse_range(seen["oop"], BOARD).weights == parse_range("KK-TT 22", BOARD).weights
    assert "22" not in seen["oop"].split(",")  # never the bare numeric pair
    assert final["root"]["range_oop_text"] == "KK-TT 22"
    assert final["report"]["root"]["range_oop_text"] == "KK-TT 22"  # the viewer shows this

    from plo5bp.cfr_app.strategy_view import load_report

    assert load_report(final["report"])["summary"]["range_oop"] == "KK-TT 22"


@pytest.mark.skipif(not rust_cfr_available(), reason="Rust CFR not built")
def test_e11_native_solver_deals_exactly_the_intended_range():
    """The canonical string, read by the REAL native parser."""
    from plo5bp.cfr_app.session import _config_from_dict, _root_from_dict
    from plo5bp.gto.cfr_api import solve

    for text in ("22", "KK-TT", "AA:0.5 KK"):
        pr = parse_range(text, BOARD)
        rep = solve(
            _root_from_dict({"street": 3, "pot_bb": 10, "effective_stack_bb": 10, "board": BOARD,
                             "raise_sizes_pm": [], "allin_atom": True, "range_oop": pr.canonical(), "range_ip": ""}),
            _config_from_dict({"max_iterations": 600, "target_exploitability_bb": 0.0,
                               "use_isomorphism": False, "seed": 5, "poll_every": 100000}),
        )
        dealt = {x["raw_combo"] for x in rep.strategy["infosets"] if x["actor"] == 0 and x["path"] == []}
        assert dealt and dealt <= set(pr.weights), text
        got = {preflop_class_label(preflop_class_from_cards(*combo_to_cards(c))) for c in dealt}
        assert got == set(pr.class_weights()), text  # "22" → deuces, not 32o; "KK-TT" → all four
