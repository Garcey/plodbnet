"""Per-session state of the live-capture hand tracker (local build only).

What the ClubGG OCR runner and the PokerNow ingest learn from successive
frames — the debounce counters of the hand-start machine, the pending
card reads, the last on-screen stacks/pot — used to sit as ~20 extra fields
on the study ``Session``, so every signed-in user of the public site carried
them. They live here now, on ``session.live`` (created on first use by
:func:`current_live_state`), and only the local build ever creates one.

The hand MEMBERSHIP the replay itself reads (``hand_in_hand_mask``,
``folded_this_hand``, ``sitting_out_seats``) and the per-slot card locks the
``/cards`` edit sets stay on the ``Session``: the study core's env build
consumes them.
"""

from __future__ import annotations

from typing import Any

from plo5bp.ui.server import _card_spec_attrs, session


def blank_card_pending() -> dict[str, list[tuple[int, int] | None]]:
    """Fresh per-slot card-read debounce state (all slots empty), shaped
    like the current format's card spec."""
    return {attr: [None] * n for attr, n in _card_spec_attrs()}


class LiveState:
    """Everything the live sources accumulate for one study session."""

    def __init__(self) -> None:
        # Directly-observed values from the most recent frame. Refreshed every
        # tick by `_mirror_observable_state` (and every PokerNow payload)
        # regardless of hand-boundary detection, so stacks/pot stay live
        # across rewinds and missed hand transitions.
        self.observed_stacks: tuple[int | None, ...] = ()
        self.observed_pot: int | None = None

        # Hero's hole cards at the most recent hand-start (or hero's first full
        # read after it). A different, disjoint set later means a new hand
        # (rewind-proof fallback when the button read is missed).
        self.last_hero_hole: tuple[int, ...] | None = None

        # Stability gate: a new (button, sitting-out) snapshot only commits
        # after holding for enough consecutive ticks, so one mid-animation
        # frame can't cascade into a bogus hand-start.
        self.pending_button: int | None = None
        self.pending_sitting_out: frozenset[int] | None = None
        self.pending_stable_ticks: int = 0
        # Frame captured the tick a new pending snapshot first appeared — the
        # pre-commit baseline for stack seeding and the reconstructor
        # rebaseline once the gate commits. Without it the 2nd stable tick
        # (post-bet) would clobber the pre-bet baseline and the stack-delta
        # fallback could no longer see the action.
        self.pending_anchor_fs: Any = None

        # Button-ONLY stability counter, decoupled from the snapshot above.
        # The button never moves mid-hand, so a stable button MOVE is a strong
        # new-hand signal that must not wait on sitting-out flicker (folds /
        # banners during the deal).
        self.last_observed_button: int | None = None
        self.button_stable_ticks: int = 0

        # Mid-hand lock: ticks since the last hand-start. Past
        # `_LOCK_AFTER_TICKS` the gates need `_STABILITY_TICKS_REQUIRED_LOCKED`
        # stable ticks, and a hero-hole rotation runs through its own
        # debouncer instead of firing on one disjoint frame — chip-settle /
        # banner flickers can't restart a live hand.
        self.ticks_since_hand_start: int = 0
        self.pending_hero_hole_rotation: tuple[int, ...] | None = None
        self.pending_hero_hole_rotation_ticks: int = 0

        # Mid-hand mask expansion: a seat reading in-hand for two ticks while
        # outside the locked mask (and not folded) joins it — the anchor frame
        # missed its card backs. Late rebuyers read folded, real folds sit in
        # `folded_this_hand`, so neither can be added.
        self.pending_mask_additions: frozenset[int] = frozenset()
        self.pending_mask_additions_ticks: int = 0

        # StreetReveal fold reconcile (OCR path only): seats that read folded
        # on the reveal tick but have no FOLD yet are reconciled only if they
        # STILL read folded on the next tick; `pending_reveal_target` carries
        # the reveal's street to that follow-up tick. (review 2026-09-20 F11)
        self.pending_reveal_folds: frozenset[int] = frozenset()
        self.pending_reveal_target: int | None = None

        # Simple mode: the OCR tick mirrors cards and runs the hand-start
        # machine but never infers actions — the user enters them.
        self.simple_ocr_mode: bool = True

        # Per-slot stability debounce for the AUTOMATIC card mirror:
        # (card_idx, consecutive identical reads) or None. A slot commits (and
        # locks) only after `_CARD_STABLE_TICKS` identical reads, so a
        # transient mid-reveal misread never latches. Manual rescan bypasses it.
        self.card_slot_pending: dict[str, list[tuple[int, int] | None]] = (
            blank_card_pending()
        )

        # The env the live path last built and the replay inputs it was built
        # from (`tracking._rebuild_env_if_stale`): a tick that changed none of
        # them reuses it instead of replaying the whole hand again.
        self.built_env: Any = None
        self.built_spec: Any = None

        # Hand-start anchors must show the antes in the pot; ticks spent
        # waiting for such a frame (bounded, see `tracking._ANTE_WAIT_TICKS`).
        self.ante_wait_ticks: int = 0

        # Sync health, shown in /ocr/status + /pokernow/status: kind ->
        # message ("rejected_action", "pot"). `rejected_key/streak` count how
        # often the same walk action was refused in a row; `pot_mismatch_ticks`
        # how many ticks the on-screen pot has disagreed with the engine's.
        self.warnings: dict[str, str] = {}
        self.rejected_key: tuple[int, str, int] | None = None
        self.rejected_streak: int = 0
        self.pot_mismatch_ticks: int = 0

        # 4-colour deck check (ClubGG pixel OCR tells suits by COLOUR:
        # green clubs, blue diamonds). Hands started and cards committed since
        # capture started, and whether a club or diamond was ever read.
        self.deck_hands: int = 0
        self.deck_cards: int = 0
        self.deck_colored_seen: bool = False

    def reset_hand(self) -> None:
        """Per-hand defaults — runs whenever the study core restores its own
        (`_new_session_defaults`: a new hand, /reset, /format). The
        cross-hand debounce (pending snapshot/anchor, hero-hole baseline,
        observed stacks) deliberately survives: `_begin_new_hand` resets
        mid-stream and the next hand is detected against it."""
        self.pending_mask_additions = frozenset()
        self.pending_mask_additions_ticks = 0
        self.ticks_since_hand_start = 0
        self.pending_hero_hole_rotation = None
        self.pending_hero_hole_rotation_ticks = 0
        self.last_observed_button = None
        self.button_stable_ticks = 0
        self.pending_reveal_folds = frozenset()
        self.pending_reveal_target = None
        self.card_slot_pending = blank_card_pending()
        self.ante_wait_ticks = 0
        self._clear_health()

    def reset_tracking(self) -> None:
        """Forget everything learned across hands too (user reset, format
        switch, capture (re)start, live-source switch). Card reads in flight
        and the simple-mode toggle are kept."""
        self.pending_button = None
        self.pending_sitting_out = None
        self.pending_stable_ticks = 0
        self.pending_anchor_fs = None
        self.last_hero_hole = None
        self.observed_stacks = ()
        self.observed_pot = None
        self.pending_mask_additions = frozenset()
        self.pending_mask_additions_ticks = 0
        self.ticks_since_hand_start = 0
        self.pending_hero_hole_rotation = None
        self.pending_hero_hole_rotation_ticks = 0
        self.last_observed_button = None
        self.button_stable_ticks = 0
        self.pending_reveal_folds = frozenset()
        self.pending_reveal_target = None
        self.built_env = None
        self.built_spec = None
        self.deck_hands = 0
        self.deck_cards = 0
        self.deck_colored_seen = False
        self.ante_wait_ticks = 0
        self._clear_health()

    def _clear_health(self) -> None:
        self.warnings = {}
        self.rejected_key = None
        self.rejected_streak = 0
        self.pot_mismatch_ticks = 0


def current_live_state() -> LiveState:
    """The current study session's live state, created on first use."""
    st = session.live
    if st is None:
        st = LiveState()
        session.live = st
    return st


class _LiveStateProxy:
    """``live_state.x`` reads/writes the CURRENT session's `LiveState` — the
    same late-binding the ``session`` proxy gives the study core."""

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(current_live_state(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(current_live_state(), name, value)


live_state: Any = _LiveStateProxy()
