"""Trainer hand history and per-seat stacks (site FEAT-027 / FEAT-017 / ST-030).

- Every finished hand is kept (newest first, capped), persisted with the
  stats file, listed in the state, and can be reopened: the reopened hand's
  review is the one it had when it ended, and it is never counted twice.
- Per-seat stack ranges are counted from the hero's seat ("You", "1st to
  your left", …), not from engine seat indices.
"""

from __future__ import annotations

import sys

import pytest


def _mod(ts):
    """The trainer module `ts` was built from — never a copy bound at import
    time here, which a test that purges and re-imports plo5bp.ui.* (conftest
    `ui_purge`) would leave stale."""
    return sys.modules[type(ts).__module__]


def _play_out(ts):
    """Play the live hand to the end: check/call when possible, else fold."""
    for _ in range(40):
        h = ts.hand
        if h is None or h.terminal:
            return
        s = ts.project_state()
        if s["legal"]["check_call"]:
            ts.act("check_call", None)
        else:
            ts.act("fold", None)
    raise AssertionError("hand did not finish")


def test_finished_hands_are_listed_and_reopen(trainer_factory):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=3, mc_rollouts=0)
    ts.new_hand()
    _play_out(ts)
    first = ts.hand
    before = ts.project_state()
    recent = before["trainer"]["recent"]
    assert len(recent) == 1 and recent[0]["hand_no"] == first.hand_no
    assert recent[0]["net_bb"] == pytest.approx(first.rewards_bb[first.hero_seat])
    hands_counted = ts.lifetime_stats.hands

    ts.new_hand()                         # a new live hand...
    rid = recent[0]["id"]
    ts.open_recent(rid)                   # ...replaced by the old one
    again = ts.project_state()
    assert again["trainer"]["hand_active"] is False
    assert again["card_spec"] == before["card_spec"]
    assert [s["stack_chips"] for s in again["seats"]] == [s["stack_chips"] for s in before["seats"]]
    assert again["trainer"]["rewards_bb"] == before["trainer"]["rewards_bb"]
    rv0, rv1 = before["trainer"]["review"], again["trainer"]["review"]
    assert rv1["num_nodes"] == rv0["num_nodes"]
    assert [d["score"] for d in rv1["decisions"]] == [d["score"] for d in rv0["decisions"]]
    # reopening never counts the hand again
    assert ts.lifetime_stats.hands == hands_counted
    with pytest.raises(Exception):
        ts.open_recent("ffffffffffff")


def test_recent_hands_persist_and_are_capped(trainer_factory, monkeypatch):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=2, mc_rollouts=0, stats_name="h.json")
    monkeypatch.setattr(_mod(ts), "RECENT_HANDS_MAX", 3)
    for _ in range(4):
        ts.new_hand()
        _play_out(ts)
    assert len(ts.recent_hands) == 3
    newest = ts.recent_hands[0]["hand_no"]
    again = trainer_factory(seats_mode="fixed", seats_fixed=2, mc_rollouts=0, stats_name="h.json")
    assert [r["hand_no"] for r in again.recent_hands] == [r["hand_no"] for r in ts.recent_hands]
    assert again.recent_hands[0]["hand_no"] == newest
    again.open_recent(again.recent_hands[-1]["id"])
    assert again.hand is not None and again.hand.terminal


def test_per_seat_stacks_count_from_the_hero(trainer_factory):
    rows = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0), (40.0, 40.0), (50.0, 50.0), (60.0, 60.0)]
    ts = trainer_factory(seats_mode="fixed", seats_fixed=4, stacks_mode="per_seat",
                         stacks_per_seat_bb=rows, mc_rollouts=0)
    for hero in range(4):
        stacks = ts._draw_stacks(4, hero)
        bb = [x / _mod(ts).BB_CHIPS for x in stacks]
        for j in range(4):
            assert bb[(hero + j) % 4] == rows[j][0]
    ts.new_hand()
    h = ts.hand
    assert h.config.resolved_stacks[h.hero_seat] == 10 * _mod(ts).BB_CHIPS


# --- drill filter (site FEAT-018) -------------------------------------------------


def _hero_spot(ts):
    s = ts.project_state()
    street = s["street"]
    before = [h for h in s["history"] if h["street"] == street]
    return s, before


@pytest.mark.parametrize("spot", ["first", "checked_to", "multiway"])
def test_drill_filter_deals_the_asked_spot(trainer_factory, spot):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=5, mc_rollouts=0, spot=spot)
    matched = 0
    for _ in range(4):
        ts.new_hand()
        if ts.hand.terminal:
            continue
        s, before = _hero_spot(ts)
        assert s["actor"] == s["hero_seat"]
        ok = {
            "first": not before,
            "checked_to": s["to_call_chips"] == 0 and bool(before),
            "multiway": sum(1 for x in s["seats"] if not x["folded"]) >= 3,
        }[spot]
        matched += ok
    assert matched >= 3


def test_dropped_drill_deals_take_no_hand_number_and_count_nowhere(trainer_factory):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=4, mc_rollouts=0, spot="first")
    ts.new_hand()
    assert ts.hand.hand_no == 1 and ts.hand.trial is False
    _play_out(ts)
    ts.new_hand()
    assert ts.hand.hand_no == 2
    assert ts.lifetime_stats.hands == 1 and len(ts.recent_hands) == 1
