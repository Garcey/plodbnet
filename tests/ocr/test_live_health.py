"""Live sync health: ante-aware hand-start anchors, refused walk actions, pot drift.

* TOOL-013 — a hand-start anchor taken before the antes were collected (pot
  shows no antes) is not used to seed stacks; the debouncer waits (bounded)
  and upgrades to the first frame whose pot holds them.
* TOOL-025 — the walk's actions are validated against the engine before they
  enter the log; a refused one voids the step (retry on the next frame, the
  reconstructor restored), then is left to the user — visibly, never silently.
* TOOL-026 — the pot on screen is compared with the engine's; a lasting
  disagreement shows as a warning in the live status.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from plo5bp.config import GameConfig  # noqa: E402
from plo5bp.ocr.events import EventReconstructor, SeatAction  # noqa: E402
from plo5bp.ocr.types import Card, FrameState, SeatObs  # noqa: E402
from plo5bp.ui import server  # noqa: E402
from plo5bp.ui.live import clubgg, tracking  # noqa: E402
from plo5bp.ui.live.state import live_state  # noqa: E402

H1 = (Card(8, 1), Card(7, 1), Card(7, 0), Card(3, 2), Card(1, 2))
# GameConfig() defaults: bb 10000 chips, ante 3 bb = 30000 chips; at $20/bb
# that is 5 engine chips per cent, so one ante = 6000 cents ($60).
ANTE_CENTS = 6000


def _fs(stacks, *, pot, button=2, in_hand=(0, 1, 2)):
    seats = tuple(
        SeatObs(seat=i, stack_chips=stacks if i in in_hand else None,
                committed_chips=None, folded=i not in in_hand)
        for i in range(6)
    )
    return FrameState(board_a=(None,) * 5, board_b=(None,) * 5, hero_hole=H1,
                      button_seat=button, pot_total_chips=pot, seats=seats)


@pytest.fixture(autouse=True)
def _clean():
    s = server.session
    s.variant = server.VARIANT_PLO5
    s.game_config = GameConfig()
    s.dollars_per_bb = 20.0
    s.num_seats, s.button_seat, s.hero_seat = 6, 0, 0
    server._new_session_defaults()
    tracking._reset_live_tracking()
    tracking._set_active_reconstructor(None)
    server._rebuild_env()
    yield s
    s.game_config = GameConfig(starting_stack=400000)
    s.dollars_per_bb = 2.0
    server._new_session_defaults()
    tracking._reset_live_tracking()
    tracking._set_active_reconstructor(None)
    server._rebuild_env()


# --- TOOL-013 -----------------------------------------------------------------------


def test_anchor_before_the_antes_is_replaced_by_the_first_frame_that_shows_them(_clean):
    behind = 200_000  # cents behind BEFORE the ante
    tracking._mirror_observable_state(_fs(behind, pot=0))
    tracking._mirror_observable_state(_fs(behind, pot=0))
    assert not _clean.hand_in_hand_mask  # held: the pot shows no antes
    tracking._mirror_observable_state(_fs(behind - ANTE_CENTS, pot=3 * ANTE_CENTS))
    assert _clean.hand_in_hand_mask == frozenset({0, 1, 2})
    # starting = behind-after-ante + ante = exactly what the player had. The
    # pre-ante anchor used to seed one ante more (200000*5 + 30000).
    stacks = _clean.game_config.resolved_stacks
    assert [stacks[i] for i in (0, 1, 2)] == [behind * 5] * 3


def test_a_misread_pot_never_blocks_the_hand_start(_clean):
    for _ in range(2 + tracking._ANTE_WAIT_TICKS):
        tracking._mirror_observable_state(_fs(200_000, pot=0))
        if _clean.hand_in_hand_mask:
            break
    assert _clean.hand_in_hand_mask == frozenset({0, 1, 2})


def test_an_unreadable_pot_does_not_delay_anything(_clean):
    tracking._mirror_observable_state(_fs(200_000, pot=None))
    tracking._mirror_observable_state(_fs(200_000, pot=None))
    assert _clean.hand_in_hand_mask == frozenset({0, 1, 2})


# --- TOOL-025 -----------------------------------------------------------------------


def _start_hand(s):
    # Three players in; the engine seeds and deals them in.
    fs = _fs(200_000, pot=3 * ANTE_CENTS)
    tracking._begin_new_hand(fs, button_seat=2, hero_hole_indices=None)
    server._rebuild_env()
    return s.env.current_actor()


def test_accepted_actions_are_committed_as_one_batch(_clean):
    actor = _start_hand(_clean)
    applied, retry = tracking._record_seat_actions(
        [SeatAction(seat=actor, gate="check_call", chips=0)]
    )
    assert (applied, retry) == (1, False)
    assert [e["gate"] for e in _clean.action_log] == [1]
    assert _clean.env.current_actor() != actor
    assert tracking.live_warnings() == []


def test_a_refused_action_voids_the_step_then_is_left_to_the_user(_clean):
    actor = _start_hand(_clean)
    nxt = (actor + 1) % 6
    recon = EventReconstructor(num_seats=6)
    recon.rebaseline(_fs(200_000, pot=3 * ANTE_CENTS))
    before = recon.snapshot()
    batch = [
        SeatAction(seat=actor, gate="check_call", chips=0),   # legal
        SeatAction(seat=nxt, gate="raise", chips=1),          # below the min raise
    ]
    for attempt in range(1, tracking._REJECTED_ACTION_RETRIES):
        recon.last_fs = None  # the step moved the baseline ...
        applied, retry = tracking._record_seat_actions(batch, recon, before)
        assert (applied, retry) == (0, True)
        assert recon.last_fs is before["last_fs"]  # ... and was undone
        assert _clean.action_log == []  # all or nothing: not even the check
        (msg,) = tracking.live_warnings()
        assert f"seat {nxt} raise by" in msg and "minimum raise" in msg
        assert "retrying" in msg
    applied, retry = tracking._record_seat_actions(batch, recon, before)
    assert (applied, retry) == (1, False)  # the legal prefix is kept
    assert [e["gate"] for e in _clean.action_log] == [1]
    (msg,) = tracking.live_warnings()
    assert f"enter seat {nxt}'s action by hand" in msg
    # The user enters it: once the engine has moved past that seat the
    # warning goes away.
    _clean.action_log.append({"gate": 1, "chips": 0, "seat": nxt})
    server._rebuild_env()
    tracking._clear_stale_refusal()
    assert tracking.live_warnings() == []


def test_ocr_status_carries_the_warnings(_clean):
    live_state.warnings["pot"] = "pot on screen $1 but $2 tracked"
    assert clubgg.ocr_runner.status()["warnings"] == ["pot on screen $1 but $2 tracked"]


# --- TOOL-026 -----------------------------------------------------------------------


def test_pot_drift_warns_only_when_it_lasts_and_clears_when_back_in_sync(_clean):
    _start_hand(_clean)
    raw = _clean.env._rs.observation_dict(skip_outcome_mc=True)
    pot_cents = int(raw["pot"]) // 5  # 5 engine chips per cent
    assert pot_cents == 3 * ANTE_CENTS
    for _ in range(tracking._POT_DRIFT_TICKS - 1):
        tracking._check_pot_sync(_fs(200_000, pot=pot_cents + 5000))
        assert tracking.live_warnings() == []  # a lag of a tick or two is fine
    tracking._check_pot_sync(_fs(200_000, pot=pot_cents + 5000))
    (msg,) = tracking.live_warnings()
    assert "pot on screen $230.00 but $180.00 tracked" in msg
    tracking._check_pot_sync(_fs(200_000, pot=pot_cents))
    assert tracking.live_warnings() == []


def test_pot_drift_needs_a_hand_and_a_readable_pot(_clean):
    for _ in range(5):
        tracking._check_pot_sync(_fs(200_000, pot=1))  # no hand yet
    _start_hand(_clean)
    for _ in range(5):
        tracking._check_pot_sync(_fs(200_000, pot=None))
    assert tracking.live_warnings() == []


# --- TOOL-042: the 2-colour deck is called out ------------------------------------


def test_two_colour_deck_is_called_out_and_cleared_by_a_coloured_suit(_clean):
    tracking._note_live_source("ocr")
    fs = _fs(200_000, pot=3 * ANTE_CENTS)
    for hand in range(tracking._DECK_CHECK_HANDS):
        tracking._begin_new_hand(fs, button_seat=2, hero_hole_indices=None)
        for k in range(5):
            tracking._note_suit_read(4 * (hand * 5 + k) % 52 + 2 + (k % 2))  # hearts / spades
    (msg,) = tracking.live_warnings()
    assert "2-colour deck" in msg
    tracking._note_suit_read(4 * 12 + 1)  # an ace of diamonds (blue)
    assert tracking.live_warnings() == []
    tracking._note_live_source(None)


def test_pokernow_hands_do_not_count_for_the_deck_check(_clean):
    tracking._note_live_source("pokernow")
    fs = _fs(200_000, pot=3 * ANTE_CENTS)
    for _ in range(5):
        tracking._begin_new_hand(fs, button_seat=2, hero_hole_indices=None)
    assert live_state.deck_hands == 0
    tracking._note_live_source(None)
