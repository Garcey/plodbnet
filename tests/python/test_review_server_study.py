"""Regression tests for the 2026-09-20 code review — study routes in
`plo5bp/ui/server.py` (local build, no live capture, no OpenCV needed).

Covers review items H6 (stack preservation), H7 (validate-then-commit),
B9 (`_network_obs` projection), F13 (`/format` on a session without an env),
F15 (seat-stamped action log), I10 (live capture is PLO5-only), F9/F11
(`folded_this_hand` follows the engine), the `_load_critic` kwargs contract
and the checkpoint observation-semantics revision check (`PLO5BP_OBS_REV`).
Each test names the finding it pins.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from starlette.testclient import TestClient

import plo5bp.ui.server as srv
from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv
from plo5bp.ui.common import effective_button

PLO5 = "plo5_double_bomb"
NLH = "nlh_single"


@pytest.fixture()
def client():
    """A TestClient over a pristine PLO5 study session; restores it after."""
    c = TestClient(srv.app, raise_server_exceptions=False)

    def pristine() -> None:
        s = srv.session
        s.variant = srv.VARIANT_PLO5
        s.game_config = GameConfig(starting_stack=400000)
        s.dollars_per_bb = 2.0
        s.num_seats = 6
        s.button_seat = 0
        s.hero_seat = 0
        srv._new_session_defaults()
        srv._reset_live_tracking()
        srv._rebuild_env()
        # Live-runner flags other test modules leave behind on the shared app.
        srv.ocr_runner.running = False
        srv.pokernow_runner.last_error = None

    pristine()
    yield c
    try:
        srv.trainer_router.set_format(PLO5)
    except Exception:
        pass
    pristine()


def _session_fingerprint() -> dict:
    """Everything a rejected request must leave untouched."""
    s = srv.session
    return {
        "cfg": s.game_config,
        "dpb": s.dollars_per_bb,
        "num_seats": s.num_seats,
        "button": s.button_seat,
        "hero_hole": list(s.hero_hole),
        "flop_a": list(s.flop_a),
        "flop_b": list(s.flop_b),
        "turn": list(s.turn_cards),
        "river": list(s.river_cards),
        "locks": {k: list(v) for k, v in s._card_slot_locked.items()},
        "log": [dict(e) for e in s.action_log],
        "env": s.env,
    }


FULL_CARDS = {
    "hero_hole": [0, 1, 2, 3, 4],
    "flop_a": [5, 6, 7],
    "flop_b": [8, 9, 10],
    "turn": [None, None],
    "river": [None, None],
}


# --- H7: validate-then-commit -------------------------------------------------


def test_h7_duplicate_cards_leave_session_untouched(client):
    assert client.post("/cards", json=FULL_CARDS).status_code == 200
    before = _session_fingerprint()

    dup = dict(FULL_CARDS, flop_a=[4, 6, 7])  # 4 is already in hero_hole
    r = client.post("/cards", json=dup)
    assert r.status_code == 400
    assert "duplicate" in r.json()["detail"]

    # Nothing was stored or locked, and the env object is the same one.
    assert _session_fingerprint() == before
    assert srv.session.env is before["env"]
    # ...so the session keeps working (it used to 400 on every later call).
    assert client.get("/state").json()["state"]["card_spec"]["flop_a"] == [5, 6, 7]
    assert client.post("/action", json={"gate": "check_call"}).status_code == 200
    assert client.post("/undo").status_code == 200


def test_h7_out_of_range_card_is_rejected_before_commit(client):
    before = _session_fingerprint()
    r = client.post("/cards", json=dict(FULL_CARDS, flop_b=[8, 9, 52]))
    assert r.status_code == 400
    assert _session_fingerprint() == before


@pytest.mark.parametrize(
    "stacks",
    [
        [-5, 100, 100, 100, 100, 100],
        [-50000] * 6,
        [2**70] * 6,
        [2**63] * 6,
        [2**62 + 1, 1, 1, 1, 1, 1],
        [2**61] * 6,  # each in range, table total wraps the engine's u64 pot
    ],
)
@pytest.mark.parametrize("route", ["/seats", "/config"])
def test_h7_absurd_stacks_rejected_and_session_survives(client, route, stacks):
    before = _session_fingerprint()
    r = client.post(route, json={"starting_stacks": stacks})
    assert r.status_code == 400
    assert _session_fingerprint() == before
    # Every later rebuild used to 500 (and /reset did not help).
    assert client.post("/action", json={"gate": "check_call"}).status_code == 200
    assert client.post("/reset").status_code == 200


def test_h7_seats_flagged_starting_stacks_are_bounded_too(client):
    before = _session_fingerprint()
    r = client.post(
        "/seats",
        json={"starting_stacks": [-1] * 6, "stacks_are_starting": True},
    )
    assert r.status_code == 400
    assert _session_fingerprint() == before


@pytest.mark.parametrize("raw", ["Infinity", "-Infinity", "NaN", "1e308", "0", "-3"])
def test_h7_non_finite_dollars_per_bb_is_a_clean_4xx(client, raw):
    r = client.post(
        "/config",
        content='{"dollars_per_bb": %s}' % raw,
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422  # Infinity/NaN used to be a 200 / 500
    assert srv.session.dollars_per_bb == 2.0
    assert client.get("/state").status_code == 200


def test_h7_bb_and_ante_are_bounded(client):
    for body in ({"bb_chips": 2**62 + 1}, {"ante_chips": 2**63}):
        assert client.post("/config", json=body).status_code == 422
    assert srv.session.game_config.bb == 10000


def test_h7_action_is_not_committed_when_the_build_fails(client, monkeypatch):
    from fastapi import HTTPException

    assert client.post("/action", json={"gate": "check_call"}).status_code == 200
    before = _session_fingerprint()

    def boom(spec):
        raise HTTPException(status_code=400, detail="engine said no")

    monkeypatch.setattr(srv, "_build_env", boom)
    assert client.post("/action", json={"gate": "check_call"}).status_code == 400
    assert client.post("/undo").status_code == 400
    monkeypatch.undo()
    assert _session_fingerprint() == before


def test_h7_ocr_card_commit_never_locks_a_duplicate(client):
    """A stable OCR misread that duplicates another slot's card used to lock
    and then 400 every rebuild (t15a)."""
    for _ in range(srv._CARD_STABLE_TICKS):
        srv._ocr_apply_card_slot("hero_hole", 0, 10, debounce=True)
        srv._ocr_apply_card_slot("flop_a", 1, 10, debounce=True)
    s = srv.session
    assert s.hero_hole[0] == 10 and s._card_slot_locked["hero_hole"][0] is True
    assert s.flop_a[1] is None and s._card_slot_locked["flop_a"][1] is False
    srv._rebuild_env()  # must not raise
    # The refused slot is still open for a later, correct read.
    for _ in range(srv._CARD_STABLE_TICKS):
        srv._ocr_apply_card_slot("flop_a", 1, 11, debounce=True)
    assert s.flop_a[1] == 11


# --- H6: stacks survive /config and seat ops ----------------------------------


def test_h6_config_without_stacks_keeps_per_seat_stacks(client):
    behind = [370000, 100000, 200000, 300000, 500000, 600000]
    r = client.post("/config", json={"starting_stacks": behind})
    start = r.json()["state"]["starting_stacks_chips"]
    assert start == [b + 30000 for b in behind]

    assert client.post("/action", json={"gate": "raise", "chips": 60000}).status_code == 200
    r = client.post("/config", json={"dollars_per_bb": 5.0})
    st = r.json()["state"]
    assert st["starting_stacks_chips"] == start  # used to reset to uniform
    assert len(srv.session.action_log) == 1      # log kept, replayed on SAME stacks
    assert st["seats"][1]["stack_chips"] == 100000 - 60000

    # bb/ante edits clear the log but still keep the per-seat stacks.
    r = client.post("/config", json={"ante_chips": 20000})
    assert r.json()["state"]["starting_stacks_chips"] == start
    assert srv.session.action_log == []


def _client_remove_seat(client, idx: int):
    """What app.js removeSeat() posts: spliced BEHIND stacks + new button."""
    s = client.get("/state").json()["state"]
    stacks = [q["stack_chips"] for q in s["seats"]]
    del stacks[idx]
    new_n = s["num_seats"] - 1
    button = s["button_seat"]
    if button == idx:
        button = idx % new_n
    elif button > idx:
        button -= 1
    return client.post(
        "/seats",
        json={"num_seats": new_n, "button_seat": button, "starting_stacks": stacks},
    )


def _client_insert_seat(client, after: int):
    """What app.js insertSeat() posts."""
    s = client.get("/state").json()["state"]
    n = s["num_seats"]
    ins = n if (after + 1) % n == 0 else after + 1
    stacks = [q["stack_chips"] for q in s["seats"]]
    stacks.insert(ins, stacks[after])
    button = s["button_seat"]
    if ins != n and button >= ins:
        button += 1
    return client.post(
        "/seats",
        json={"num_seats": n + 1, "button_seat": button, "starting_stacks": stacks},
    )


def test_h6_nlh_blinds_do_not_erode_on_seat_ops(client):
    s0 = client.post("/format", json={"format": NLH}).json()["state"]
    start0 = s0["starting_stacks_chips"]
    behind0 = [q["stack_chips"] for q in s0["seats"]]
    for _ in range(3):
        assert _client_remove_seat(client, 5).status_code == 200
        r = _client_insert_seat(client, 4)
        assert r.status_code == 200
    s1 = r.json()["state"]
    # SB/BB used to lose their blind on every op (99.0 → 96.0, 98.5 → 92.5).
    assert s1["starting_stacks_chips"] == start0
    assert [q["stack_chips"] for q in s1["seats"]] == behind0


def test_h6_seat_removal_splices_the_right_starting_stack(client):
    behind = [370000, 100000, 200000, 300000, 500000, 600000]
    client.post("/config", json={"starting_stacks": behind})
    # Put chips in play first: "behind" and "starting" now differ per seat.
    assert client.post("/action", json={"gate": "raise", "chips": 60000}).status_code == 200

    r = _client_remove_seat(client, 2)
    st = r.json()["state"]
    assert st["num_seats"] == 5
    assert st["starting_stacks_chips"] == [
        b + 30000 for b in (370000, 100000, 300000, 500000, 600000)
    ]
    # The hand restarted: seat 1 has its 60000 back (old code kept it lost).
    assert st["seats"][1]["stack_chips"] == 100000

    r = _client_insert_seat(client, 1)  # new seat copies seat 1
    st = r.json()["state"]
    assert st["starting_stacks_chips"] == [
        b + 30000 for b in (370000, 100000, 100000, 300000, 500000, 600000)
    ]


def test_h6_seat_count_buttons_keep_per_seat_stacks(client):
    behind = [370000, 100000, 200000, 300000, 500000, 600000]
    client.post("/config", json={"starting_stacks": behind})
    r = client.post("/seats", json={"num_seats": 5})  # the "-" button: no stacks
    assert r.json()["state"]["starting_stacks_chips"] == [
        b + 30000 for b in behind[:5]
    ]
    r = client.post("/seats", json={"num_seats": 6})  # "+": pad with the last seat
    assert r.json()["state"]["starting_stacks_chips"] == [
        b + 30000 for b in behind[:5] + [behind[4]]
    ]


def test_h6_button_move_keeps_starting_stacks(client):
    behind = [370000, 100000, 200000, 300000, 500000, 600000]
    start = client.post("/config", json={"starting_stacks": behind}).json()[
        "state"
    ]["starting_stacks_chips"]
    client.post("/action", json={"gate": "raise", "chips": 60000})
    live_behind = [
        q["stack_chips"] for q in client.get("/state").json()["state"]["seats"]
    ]
    r = client.post("/seats", json={"button_seat": 3, "starting_stacks": live_behind})
    st = r.json()["state"]
    assert st["button_seat"] == 3
    assert st["starting_stacks_chips"] == start
    assert srv.session.action_log == []


def test_h6_flagged_stacks_are_starting_stacks_verbatim(client):
    want = [111000, 222000, 333000, 444000, 555000]
    r = client.post(
        "/seats",
        json={"num_seats": 5, "starting_stacks": want, "stacks_are_starting": True},
    )
    assert r.status_code == 200
    assert r.json()["state"]["starting_stacks_chips"] == want


def test_h6_unflagged_stacks_without_a_seat_op_stay_behind_stacks(client):
    """No seat/button change ⇒ the list keeps its historical meaning."""
    client.post("/action", json={"gate": "raise", "chips": 60000})
    behind = [370000, 250000, 370000, 370000, 370000, 370000]
    r = client.post("/seats", json={"starting_stacks": behind})
    st = r.json()["state"]
    assert [q["stack_chips"] for q in st["seats"]] == behind
    assert len(srv.session.action_log) == 1  # not a seat op: log survives


# --- F13 ------------------------------------------------------------------------


def test_f13_same_format_post_rebuilds_a_missing_env(client):
    srv.session.env = None  # what a fresh per-user Session looks like
    r = client.post("/format", json={"format": PLO5})
    assert r.status_code == 200
    assert r.json()["state"]["format"] == PLO5
    assert srv.session.env is not None


# --- F15 ------------------------------------------------------------------------


def test_f15_manual_actions_are_stamped_with_their_seat(client):
    actor = client.get("/state").json()["state"]["actor"]
    client.post("/action", json={"gate": "check_call"})
    assert srv.session.action_log[-1] == {
        "gate": int(GATE_CHECK_CALL), "chips": 0, "seat": actor,
    }


def test_f15_replay_warns_once_when_attribution_diverges(client, caplog):
    first = client.get("/state").json()["state"]["actor"]
    wrong = (first + 2) % 6
    srv.session.action_log = [{"gate": int(GATE_CHECK_CALL), "chips": 0, "seat": wrong}]
    with caplog.at_level(logging.WARNING, logger="plo5bp.ui"):
        srv._rebuild_env()
        srv._rebuild_env()
    hits = [r for r in caplog.records if "recorded for seat" in r.getMessage()]
    assert len(hits) == 1
    assert f"seat {wrong}" in hits[0].getMessage()
    # Replay semantics unchanged: the entry still went to the engine's actor.
    raw = dict(srv.session.env._rs.observation_dict())
    assert [int(h[0]) for h in raw["history"]] == [first]
    assert len(srv.session.action_log) == 1


def test_f15_unstamped_entries_replay_silently(client, caplog):
    srv.session.action_log = [{"gate": int(GATE_CHECK_CALL), "chips": 0}]
    with caplog.at_level(logging.WARNING, logger="plo5bp.ui"):
        srv._rebuild_env()
    assert not [r for r in caplog.records if "recorded for seat" in r.getMessage()]


# --- F9 / F11: folded_this_hand follows the engine ------------------------------


def _live_hand(mask=frozenset(range(6)), button=2):
    s = srv.session
    s.button_seat = button
    s.hand_in_hand_mask = frozenset(mask)
    s.folded_this_hand = frozenset()
    s.sitting_out_seats = frozenset(range(6)) - frozenset(mask)


def test_f11_undo_of_a_recorded_fold_unfolds_the_seat(client):
    _live_hand()
    srv.session.action_log = [
        {"gate": int(GATE_RAISE), "chips": 60000, "seat": 3},
        {"gate": int(GATE_FOLD), "chips": 0, "seat": 4},
    ]
    srv.session.folded_this_hand = frozenset({4})
    srv.session.sitting_out_seats = frozenset({4})
    srv._rebuild_env()
    assert srv.session.folded_this_hand == frozenset({4})

    assert client.post("/undo").status_code == 200
    raw = dict(srv.session.env._rs.observation_dict())
    assert int(raw["actor"]) == 4 and not raw["folded"][4]
    assert srv.session.folded_this_hand == frozenset()
    assert srv.session.sitting_out_seats == frozenset()


def test_f9_a_fold_the_engine_rejected_does_not_leave_the_seat_skipped(client):
    _live_hand()
    # No bet to face ⇒ the engine refuses the FOLD and replay drops it.
    srv.session.action_log = [{"gate": int(GATE_FOLD), "chips": 0, "seat": 3}]
    srv.session.folded_this_hand = frozenset({3})
    srv.session.sitting_out_seats = frozenset({3})
    srv._rebuild_env()
    assert srv.session.action_log == []
    assert srv.session.folded_this_hand == frozenset()
    assert srv.session.sitting_out_seats == frozenset()


def test_f9_sync_is_live_mode_only(client):
    """Plain study (no hand-start mask) never touches the live sets."""
    client.post("/action", json={"gate": "raise", "chips": 60000})
    client.post("/action", json={"gate": "fold"})
    assert srv.session.folded_this_hand == frozenset()
    assert srv.session.sitting_out_seats == frozenset()


def test_f9_structurally_out_seats_are_not_counted_as_folds(client):
    _live_hand(mask={0, 2, 4})
    srv._rebuild_env()
    assert srv.session.folded_this_hand == frozenset()
    assert srv.session.sitting_out_seats == frozenset({1, 3, 5})


# --- I8: hero is never auto-acted ----------------------------------------------


def test_i8_auto_fold_never_acts_for_hero(client):
    # Anchor missed hero: mask without seat 0 ⇒ engine deals everyone in.
    _live_hand(mask={2, 4}, button=2)
    srv._rebuild_env()
    # First to act is seat 3 (structurally out → auto-checked), then seat 4.
    srv.session.action_log = [{"gate": int(GATE_RAISE), "chips": 60000, "seat": 4}]
    srv._rebuild_env()
    raw = dict(srv.session.env._rs.observation_dict())
    assert not raw["folded"][0], "hero must not be auto-folded"
    assert int(raw["actor"]) == 0, "the engine waits on hero"
    assert raw["folded"][5], "a real sitting-out seat is still retired"


# --- B9: _network_obs vs an honest n-seat env ------------------------------------

HOLE = [48, 44, 40, 36, 7]
FLOP_A = [51, 47, 2]
FLOP_B = [20, 25, 30]
STACKS6 = [400000, 400000, 300000, 400000, 500000, 400000]


def _compressed_vs_honest(mask, button, actions):
    s = srv.session
    s.game_config = GameConfig(
        num_seats=6, starting_stack=400000, ante=30000, bb=10000,
        starting_stacks=tuple(STACKS6),
    )
    s.num_seats, s.hero_seat, s.button_seat = 6, 0, button
    srv._new_session_defaults()
    s.hero_hole, s.flop_a, s.flop_b = list(HOLE), list(FLOP_A), list(FLOP_B)
    s.hand_in_hand_mask = frozenset(mask)
    s.action_log = [dict(a) for a in actions]
    srv._rebuild_env()
    assert int(dict(s.env._rs.observation_dict())["actor"]) == 0
    served = srv._network_obs()

    comp = [p for p in range(6) if p in mask]  # hero first, clockwise
    honest_cfg = GameConfig(
        num_seats=len(comp), starting_stack=400000, ante=30000, bb=10000,
        starting_stacks=tuple(STACKS6[p] for p in comp),
    )
    honest = BombPotEnv(honest_cfg)
    honest.reset_study(
        button=comp.index(effective_button(button, 6, frozenset(mask))),
        hero_seat=0, hero_hole=list(HOLE), flop_a=list(FLOP_A), flop_b=list(FLOP_B),
    )
    for a in actions:
        honest.step_hybrid(a["gate"], a["chips"])
    assert honest.current_actor() == 0
    want, _ = honest._pack_obs()
    return served, want


_BET = {"gate": int(GATE_RAISE), "chips": 60000}
_CHECK = {"gate": int(GATE_CHECK_CALL), "chips": 0}


@pytest.mark.parametrize(
    "mask,button,actions",
    [
        ({0, 2, 4}, 2, [_BET]),                 # 3-way, seat 4 bets
        ({0, 3}, 0, [_BET]),                    # heads-up, villain bets
        ({0, 1, 2, 3, 5}, 5, []),               # 5-way, hero first
        ({0, 2, 4}, 4, [_CHECK, _BET, _CHECK]),  # x/b/c back to hero
        ({0, 2, 4}, 3, [_CHECK]),               # DEAD button (seat 3 not dealt)
        ({0, 2, 4}, 5, []),                     # dead button right of hero
        ({0, 1, 4}, 2, [_BET]),                 # dead button, hero facing a bet
    ],
)
def test_b9_network_obs_matches_an_honest_short_handed_env(
    client, mask, button, actions
):
    served, want = _compressed_vs_honest(mask, button, actions)
    assert served.shape == want.shape
    diff = np.nonzero(np.abs(served - want) > 1e-6)[0]
    assert diff.size == 0, f"dims differ: {diff.tolist()[:12]}"


def test_b9_effective_button():
    assert effective_button(3, 6, frozenset({0, 2, 4})) == 2
    assert effective_button(5, 6, frozenset({0, 2, 4})) == 4
    assert effective_button(1, 6, frozenset({0, 3})) == 0
    assert effective_button(0, 6, frozenset({1, 2})) == 2  # wraps counter-clockwise
    assert effective_button(2, 6, frozenset({0, 2, 4})) == 2  # live button
    assert effective_button(4, 6, None) == 4
    assert effective_button(4, 6, frozenset()) == 4


# --- I10: live capture is PLO5-only ---------------------------------------------

_PN_PAYLOAD = {
    "schema": "pokernow.v1", "variant": "plo5", "bombPot": True,
    "potDollars": 12.0, "button": {"seat": 6},
    "boards": [
        {"run": "1", "cards": ["4c", "Jc", "Ad"]},
        {"run": "2", "cards": ["9h", "2s", "Kd"]},
    ],
    "heroCards": ["Ts", "As", "4h", "3d", "2d"],
    "seats": [
        {"seat": 1, "name": "Miles", "isHero": True, "isActor": True,
         "angleCW": 147, "stackDollars": 86.0, "folded": False,
         "betText": "check", "betDollars": None,
         "cards": ["Ts", "As", "4h", "3d", "2d"]},
        {"seat": 6, "name": "JJ", "isHero": False, "isActor": False,
         "angleCW": 328, "stackDollars": 62.0, "folded": False,
         "betText": None, "betDollars": None, "cards": [None] * 5},
    ],
}


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("post", "/pokernow/ingest", _PN_PAYLOAD),
        ("post", "/ocr/start", {"window_match": "ClubGG", "poll_ms": 200}),
        ("post", "/ocr/rescan", {"target": "hole"}),
    ],
)
def test_i10_live_entry_points_409_outside_plo5(client, method, path, body):
    assert client.post("/format", json={"format": NLH}).status_code == 200
    before = _session_fingerprint()
    frames_before = srv.pokernow_runner.frames_seen

    r = getattr(client, method)(path, json=body)
    assert r.status_code == 409
    assert "PLO5" in r.json()["detail"]

    # No state mutation: one PokerNow POST used to swap in a PLO config while
    # session.variant stayed NLH, bricking the session until a format flip.
    assert _session_fingerprint() == before
    assert srv.session.variant == NLH
    assert srv.session.game_config.variant == NLH
    assert srv.pokernow_runner.frames_seen == frames_before
    assert client.get("/state").status_code == 200
    assert client.post("/reset").status_code == 200


def test_i10_runner_ignores_frames_outside_plo5(client):
    """Direct callers (websocket loop, in-flight frames) are guarded too."""
    client.post("/format", json={"format": NLH})
    before = _session_fingerprint()
    srv.pokernow_runner.handle_payload(_PN_PAYLOAD)
    assert _session_fingerprint() == before
    assert "PLO5" in (srv.pokernow_runner.last_error or "")
    srv.pokernow_runner.last_error = None


def test_i10_pokernow_seat_reconfig_keeps_variant_and_sb(client):
    srv.session.game_config = GameConfig.nlh_default()
    assert srv._pokernow_set_seat_count(4) is True
    cfg = srv.session.game_config
    assert (cfg.variant, cfg.sb, cfg.num_seats) == (NLH, 5000, 4)


def test_b8_pokernow_seat_count_above_engine_limit_is_refused(client):
    before = _session_fingerprint()
    for n in (9, 10, 1, 0):
        assert srv._pokernow_set_seat_count(n) is False
    assert _session_fingerprint() == before


# --- _load_critic kwargs contract -------------------------------------------------


def test_critic_value_kwargs_are_signature_guarded(monkeypatch):
    ckpt = {"config": {"value_support": 900.0, "value_hlgauss_sigma": 0.5,
                       "hidden_dim": 2048}}

    def new_builder(state_dict, q_fold_zero=False, q_base_raw=False,
                    value_support=None, value_hlgauss_sigma=None):
        raise AssertionError("not called")

    def old_builder(state_dict, q_fold_zero=False, q_base_raw=False):
        raise AssertionError("not called")

    monkeypatch.setattr(srv, "build_critic_from_state_dict", new_builder)
    assert srv._critic_value_kwargs(ckpt) == {
        "value_support": 900.0, "value_hlgauss_sigma": 0.5,
    }
    # Builder that predates the kwargs: nothing is passed (no TypeError).
    monkeypatch.setattr(srv, "build_critic_from_state_dict", old_builder)
    assert srv._critic_value_kwargs(ckpt) == {}
    # Absent / None / malformed config blocks pass nothing.
    monkeypatch.setattr(srv, "build_critic_from_state_dict", new_builder)
    assert srv._critic_value_kwargs({"config": {"value_support": None}}) == {}
    assert srv._critic_value_kwargs({"config": None}) == {}
    assert srv._critic_value_kwargs({}) == {}
    assert srv._critic_value_kwargs(object()) == {}


def test_load_critic_passes_the_checkpoint_config(monkeypatch, tmp_path):
    import torch

    from plo5bp.network import CentralCritic

    critic = CentralCritic(hidden_dim=16, num_blocks=1)
    path = tmp_path / "ckpt.pt"
    torch.save(
        {"head_version": 2, "critic": critic.state_dict(),
         "config": {"value_support": 777.0, "value_hlgauss_sigma": 0.25}},
        path,
    )
    monkeypatch.setattr(srv, "_format_ckpt_path", lambda variant: path)
    seen = {}
    real = srv.build_critic_from_state_dict

    def spy(state_dict, q_fold_zero=False, q_base_raw=False,
            value_support=None, value_hlgauss_sigma=None):
        seen.update(value_support=value_support,
                    value_hlgauss_sigma=value_hlgauss_sigma)
        return real(state_dict)

    monkeypatch.setattr(srv, "build_critic_from_state_dict", spy)
    assert srv._load_critic(torch.device("cpu"), srv.VARIANT_PLO5) is not None
    assert seen == {"value_support": 777.0, "value_hlgauss_sigma": 0.25}


# --- observation-semantics revision (PLO5BP_OBS_REV) ---------------------------------


def _actor_ckpt(path, **extra):
    import torch

    from plo5bp.network import ActorCriticV2

    net = ActorCriticV2(hidden_dim=16, num_layers=1)
    torch.save(
        {"model": net.state_dict(), "head_version": 2,
         "config": {"hidden_dim": 16, "num_layers": 1}, **extra},
        path,
    )
    return path


@pytest.fixture()
def obs_rev(monkeypatch):
    """Pin the process revision and start with a clean warn-once ledger."""
    monkeypatch.setattr(srv, "_OBS_REV_WARNED", set())
    monkeypatch.setattr(srv, "_OBS_REV_INFO", dict(srv._OBS_REV_INFO))

    def set_process_rev(rev: int) -> None:
        monkeypatch.setattr(srv._encoding, "OBS_SEMANTICS_REV", rev, raising=False)

    set_process_rev(2)
    return set_process_rev


def _obs_rev_warnings(caplog):
    return [r for r in caplog.records if "OBS-REV MISMATCH" in r.getMessage()]


def test_obs_rev_unstamped_checkpoint_is_a_loud_mismatch_but_still_served(
    obs_rev, tmp_path, monkeypatch, caplog
):
    path = _actor_ckpt(tmp_path / "old_run.pt")  # no obs_rev key ⇒ trained on rev 1
    monkeypatch.setenv("PLO5BP_CHECKPOINT", str(path))
    with caplog.at_level(logging.WARNING, logger="plo5bp.ui"):
        model, loaded = srv._load_model(srv.VARIANT_PLO5)
        srv._load_model(srv.VARIANT_PLO5)  # same checkpoint again
    assert loaded is True and model is not None  # never refused
    assert srv._obs_rev_entry(srv.VARIANT_PLO5) == {
        "obs_rev": 1, "obs_rev_mismatch": True,
    }
    hits = _obs_rev_warnings(caplog)
    assert len(hits) == 1, "ONE warning per checkpoint, not one per load"
    msg = hits[0].getMessage()
    assert str(path) in msg and "rev 1" in msg and "rev 2" in msg
    assert "PLO5BP_OBS_REV=1" in msg  # the remedy


def test_obs_rev_matching_checkpoints_do_not_warn(obs_rev, tmp_path, monkeypatch, caplog):
    stamped = _actor_ckpt(tmp_path / "new_run.pt", obs_rev=2)
    monkeypatch.setenv("PLO5BP_CHECKPOINT", str(stamped))
    with caplog.at_level(logging.WARNING, logger="plo5bp.ui"):
        _, loaded = srv._load_model(srv.VARIANT_PLO5)
    assert loaded is True
    assert srv._obs_rev_entry(srv.VARIANT_PLO5) == {
        "obs_rev": 2, "obs_rev_mismatch": False,
    }

    # PLO5BP_OBS_REV=1 process + an unstamped (rev-1) checkpoint: exact match.
    obs_rev(1)
    monkeypatch.setenv("PLO5BP_CHECKPOINT", str(_actor_ckpt(tmp_path / "old.pt")))
    with caplog.at_level(logging.WARNING, logger="plo5bp.ui"):
        srv._load_model(srv.VARIANT_PLO5)
    assert srv._obs_rev_entry(srv.VARIANT_PLO5) == {
        "obs_rev": 1, "obs_rev_mismatch": False,
    }
    assert _obs_rev_warnings(caplog) == []

    # ...and the rev-2 checkpoint is the mismatch there, with its own remedy.
    monkeypatch.setenv("PLO5BP_CHECKPOINT", str(stamped))
    with caplog.at_level(logging.WARNING, logger="plo5bp.ui"):
        srv._load_model(srv.VARIANT_PLO5)
    assert srv._obs_rev_entry(srv.VARIANT_PLO5)["obs_rev_mismatch"] is True
    assert "PLO5BP_OBS_REV=2" in _obs_rev_warnings(caplog)[-1].getMessage()


def test_obs_rev_random_init_placeholder_never_mismatches(
    obs_rev, tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("PLO5BP_CHECKPOINT", str(tmp_path / "missing.pt"))
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"not a checkpoint")
    with caplog.at_level(logging.WARNING, logger="plo5bp.ui"):
        _, loaded = srv._load_model(srv.VARIANT_PLO5)
        assert loaded is False
        assert srv._obs_rev_entry(srv.VARIANT_PLO5) == {
            "obs_rev": 2, "obs_rev_mismatch": False,
        }
        monkeypatch.setenv("PLO5BP_CHECKPOINT", str(garbage))
        _, loaded = srv._load_model(srv.VARIANT_PLO5)
        assert loaded is False
        assert srv._obs_rev_entry(srv.VARIANT_PLO5)["obs_rev_mismatch"] is False
    assert _obs_rev_warnings(caplog) == []


def test_obs_rev_is_exposed_per_format(client, monkeypatch):
    body = client.get("/formats").json()
    assert {f["id"] for f in body["formats"]} == set(srv.FORMATS)
    for f in body["formats"]:
        assert f["obs_rev_mismatch"] is bool(srv.FORMATS[f["id"]]["obs_rev_mismatch"])
        assert isinstance(srv.FORMATS[f["id"]]["obs_rev"], int)

    monkeypatch.setitem(srv.FORMATS[NLH], "obs_rev_mismatch", True)
    flags = {f["id"]: f["obs_rev_mismatch"] for f in client.get("/formats").json()["formats"]}
    assert flags[NLH] is True and flags[PLO5] is bool(srv.FORMATS[PLO5]["obs_rev_mismatch"])


def test_obs_rev_follows_the_experimental_hot_reload(obs_rev, tmp_path, monkeypatch):
    import torch

    from plo5bp.network import ActorCriticV5
    from plo5bp.sizing import PLO_ANCHOR_SPEC

    net = ActorCriticV5(
        hidden_dim=16, obs_dim=srv.OBS_DIM_MINIMAL, num_layers=1,
        anchor_spec=PLO_ANCHOR_SPEC,
    )
    snap = tmp_path / "vMin1_7.pt"
    torch.save(
        {"model": net.state_dict(), "head_version": 4,
         "config": {"hidden_dim": 16, "num_layers": 1}},
        snap,
    )
    monkeypatch.delenv("PLO5BP_CHECKPOINT_EXPERIMENTAL", raising=False)
    real = srv._format_ckpt_path
    monkeypatch.setattr(
        srv, "_format_ckpt_path",
        lambda v: snap if v == srv.FORMAT_EXPERIMENTAL else real(v),
    )
    entry = srv.FORMATS[srv.FORMAT_EXPERIMENTAL]
    saved = dict(entry)
    try:
        srv._maybe_reload_experimental()
        assert entry["loaded"] is True and entry["_ckpt_path"] == str(snap)
        assert (entry["obs_rev"], entry["obs_rev_mismatch"]) == (1, True)
    finally:
        entry.clear()
        entry.update(saved)


# --- GTO strategy host: training-coverage gate + anchors rows (F11 / F13) -----------


class _FakeGtoHost:
    """Strategy-host stub with a recorded training coverage (default: heads-up
    rivers only, like the v1 teacher). Records every `supports` query."""

    def __init__(self, seats=(2,), streets=(3,)):
        import types

        from plo5bp.sizing import NLH_ANCHOR_SPEC

        self._seats, self._streets = set(seats), set(streets)
        self.model = types.SimpleNamespace(anchor_spec=NLH_ANCHOR_SPEC)
        self.queries: list[tuple[int, int]] = []
        self.served_streets: list[int] = []

    def supports(self, *, seats: int, street: int) -> bool:
        self.queries.append((int(seats), int(street)))
        return seats in self._seats and street in self._streets

    def node_distribution(self, obs, info):
        from plo5bp.gto.backend import NodeDist
        from plo5bp.sizing import NLH_ANCHOR_SPEC, anchor_grid_np, sizing_from_info

        self.served_streets.append(int(info.raw_obs["street"]))
        sz = sizing_from_info(info)
        grid = anchor_grid_np(sz[0], sz[1], sz[2], sz[3], NLH_ANCHOR_SPEC)
        legal = [bool(x) for x in grid.legal]
        n_legal = max(1, sum(legal))
        rec_anchor = max(k for k, ok in enumerate(legal) if ok)
        return NodeDist(
            head_version=2,
            gate_probs=[0.1, 0.2, 0.7],
            rec_gate=int(GATE_RAISE),
            rec_chips=int(grid.chips[rec_anchor]),
            value_bb=0.0,
            min_chips=int(info.min_raise_chips),
            max_chips=int(info.max_raise_chips),
            anchor_probs=[(1.0 / n_legal if ok else 0.0) for ok in legal],
            anchor_chips=[int(c) for c in grid.chips],
            anchor_legal=legal,
            rec_anchor=rec_anchor,
            pot_ref_chips=int(sz[2]) + 2 * int(sz[3]),
            mode="policy_net",
            backend_name="fake_gto",
        )


class _LegacyGtoHost(_FakeGtoHost):
    """A host that predates the coverage API: no callable `supports`."""

    supports = None


def _nlh_hand(client, num_seats: int):
    assert client.post("/format", json={"format": NLH}).status_code == 200
    if num_seats != 6:
        assert client.post("/seats", json={"num_seats": num_seats}).status_code == 200
    r = client.post("/cards", json={
        "hero_hole": [51, 47], "flop_a": [0, 5, 10], "flop_b": [],
        "turn": [15], "river": [20],
    })
    assert r.status_code == 200
    return r.json()["state"]


def _walk_to_hero(client, state, street=None):
    """Check/call until hero is to act (on ``street`` when given)."""
    for _ in range(40):
        if state["terminal"]:
            break
        if state["actor"] == state["hero_seat"] and street in (None, state["street"]):
            return state
        state = client.post("/action", json={"gate": "check_call"}).json()["state"]
    raise AssertionError(f"hero never to act on {street}: at {state['street']}")


def test_f11_gto_host_is_not_served_outside_its_training_coverage(client, monkeypatch):
    host = _FakeGtoHost()  # trained on heads-up rivers only
    monkeypatch.setattr(srv, "GTO_HOST", host)

    # 6-max preflop: neither the table shape nor the street is covered.
    s = _walk_to_hero(client, _nlh_hand(client, 6))
    assert s["street"] == "preflop"
    rec = s["recommendation"]
    assert rec["mode"] == "ppo" and rec["gto_unsupported"] is True
    assert "is_gto" not in rec and rec.get("backend") != "fake_gto"
    # It is the format's own PPO payload (v4 ladder, placeholder flag intact).
    assert rec["anchor_count"] == 12 and "model_loaded" in rec
    assert rec["anchors"][0]["label"] == "min"
    assert host.served_streets == []
    assert set(host.queries) == {(6, 0)}


def test_f11_gto_host_serves_exactly_the_covered_nodes(client, monkeypatch):
    host = _FakeGtoHost()
    monkeypatch.setattr(srv, "GTO_HOST", host)

    # Heads-up: the table shape is covered, but only the river street is.
    s = _nlh_hand(client, 2)
    seen = {}
    for street in ("preflop", "flop", "turn", "river"):
        s = _walk_to_hero(client, s, street)
        seen[street] = s["recommendation"]
        if street != "river":
            s = client.post("/action", json={"gate": "check_call"}).json()["state"]

    for street in ("preflop", "flop", "turn"):
        assert seen[street]["mode"] == "ppo", street
        assert seen[street]["gto_unsupported"] is True, street
    river = seen["river"]
    assert river["mode"] == "policy_net" and river["backend"] == "fake_gto"
    assert "gto_unsupported" not in river
    assert host.served_streets and set(host.served_streets) == {3}
    assert set(host.queries) == {(2, 0), (2, 1), (2, 2), (2, 3)}


def test_f13_host_recommendation_carries_ready_made_anchor_rows(client, monkeypatch):
    from plo5bp.sizing import NLH_ANCHOR_SPEC

    monkeypatch.setattr(srv, "GTO_HOST", _FakeGtoHost())
    s = _walk_to_hero(client, _nlh_hand(client, 2), "river")
    rec = s["recommendation"]
    assert rec["mode"] == "policy_net"

    rows = rec["anchors"]
    legal_ks = [k for k, ok in enumerate(rec["anchor_legal"]) if ok]
    assert rows and [a["k"] for a in rows] == legal_ks  # legal-only, in order
    assert rec["anchor_count"] == NLH_ANCHOR_SPEC.count == len(rec["anchor_probs"])
    for a in rows:
        k = a["k"]
        assert a["chips"] == rec["anchor_chips"][k]
        assert a["prob"] == rec["anchor_probs"][k]
        assert a["chips_bb"] == round(a["chips"] / srv.session.game_config.bb, 4)
        assert a["label"] == srv._spec_anchor_label(NLH_ANCHOR_SPEC, k)
    assert rows[0]["label"] == "min" and rows[0]["frac"] == 0.0
    top = rows[-1]
    assert top["label"] == "ALL-IN" and top["frac"] is None
    assert top["chips"] == s["raise_bounds"]["max_chips"]
    # Same row shape as a PPO recommendation's.
    ppo_rows = srv._format_model_recommendation(
        srv._fmt(), srv._network_obs(), srv.session.last_info
    )["anchors"]
    assert set(rows[0]) == set(ppo_rows[0])


def test_f13_mismatched_arrays_leave_the_raw_arrays_only():
    from plo5bp.gto.backend import NodeDist
    from plo5bp.sizing import NLH_ANCHOR_SPEC

    nd = NodeDist(
        head_version=2, gate_probs=[0.0, 0.5, 0.5], rec_gate=int(GATE_RAISE),
        rec_chips=100, value_bb=0.0, min_chips=100, max_chips=900,
        anchor_probs=[0.5, 0.5], anchor_chips=[100, 900], anchor_legal=[True, True],
        rec_anchor=1, pot_ref_chips=300,
    )
    out = srv._recommendation_from_nodedist(nd, None, spec=NLH_ANCHOR_SPEC)
    assert "anchors" not in out and out["anchor_probs"] == [0.5, 0.5]
    assert "anchors" not in srv._recommendation_from_nodedist(nd, None)


def test_f11_host_without_supports_keeps_serving(client, monkeypatch):
    host = _LegacyGtoHost()
    monkeypatch.setattr(srv, "GTO_HOST", host)
    s = _walk_to_hero(client, _nlh_hand(client, 6))
    rec = s["recommendation"]
    assert rec["mode"] == "policy_net" and "gto_unsupported" not in rec
    assert host.served_streets and set(host.served_streets) == {0}


def test_f11_a_failing_coverage_check_falls_back_to_ppo(client, monkeypatch):
    class Broken(_FakeGtoHost):
        def supports(self, *, seats, street):
            raise RuntimeError("corrupt coverage meta")

    host = Broken()
    monkeypatch.setattr(srv, "GTO_HOST", host)
    s = _walk_to_hero(client, _nlh_hand(client, 2), "river")
    rec = s["recommendation"]
    assert rec["mode"] == "ppo" and rec["gto_unsupported"] is True
    assert host.served_streets == []


def test_f11_plo_recommendations_never_consult_the_host(client, monkeypatch):
    host = _FakeGtoHost(seats=range(2, 9), streets=range(4))
    monkeypatch.setattr(srv, "GTO_HOST", host)
    client.post("/cards", json=FULL_CARDS)
    s = _walk_to_hero(client, client.get("/state").json()["state"])
    rec = s["recommendation"]
    assert rec is not None and "gto_unsupported" not in rec and "mode" not in rec
    assert host.queries == [] and host.served_streets == []
