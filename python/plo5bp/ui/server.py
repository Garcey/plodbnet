"""Study-tool FastAPI backend (single-screen edition).

Card spec is client-authoritative and supports partial entry: any slot
can be null. The server pads nulls with the lowest-index unused deck
cards, runs `reset_study`, and replays the action log through
`step_hybrid`. Hero recommendations are gated on `hero_info_complete`
(all 5 hole cards placed); non-hero actors can act freely.

Run with: `uvicorn plo5bp.ui.server:app --port 8765`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from plo5bp.actions import (
    GATE_ACTIONS,
    GATE_CHECK_CALL,
    GATE_FOLD,
    GATE_NAMES,
    GATE_RAISE,
)
from plo5bp.config import GameConfig
from plo5bp.encoding import encode_observation
from plo5bp.env import BombPotEnv
from plo5bp.network import (
    ActorCritic,
    CentralCritic,
    build_actor_from_state_dict,
    obs_adapter,
)
from plo5bp.sizing import (
    ANCHOR_COUNT,
    BRACKET_HALF,
    anchor_grid_np,
    sizing_from_info,
)

from plo5bp.ui.common import (
    AWAITING_NAMES,
    HISTORY_NAMES as _HISTORY_NAMES,
    POSITION_BY_SEAT_6,
    POSITION_BY_SEAT_SHORT,
    STREET_NAMES,
    anchor_label as _anchor_label,
    position_name as _common_position_name,
)

logger = logging.getLogger("plo5bp.ui")

TERMINAL_NAMES = {0: "fold_out", 1: "run_out", 2: "showdown"}
TERMINAL_MESSAGES = {
    "fold_out": "All opponents folded — uncontested pot.",
    "run_out": "All remaining players are all-in; turn/river cards were not entered.",
    "showdown": "River action closed with multiple players — opponent cards unknown, no showdown evaluated.",
}

STATIC_DIR = Path(__file__).parent / "static"


# --- Model loader -----------------------------------------------------------

def _resolve_device() -> str:
    """Pick the inference device. PLO5BP_DEVICE=cuda promotes when
    available; otherwise log and fall back to CPU."""
    requested = os.environ.get("PLO5BP_DEVICE", "cpu").strip().lower() or "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("PLO5BP_DEVICE=cuda but cuda unavailable — falling back to cpu")
    return "cpu"


def _load_model() -> ActorCritic:
    device = _resolve_device()
    ckpt_path = Path(os.environ.get("PLO5BP_CHECKPOINT", "checkpoints/stub.pt"))
    if not ckpt_path.exists():
        logger.warning("checkpoint %s not found — using random-init model", ckpt_path)
        return ActorCritic(hidden_dim=128, num_layers=2).to(device).eval()
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.warning("failed to load %s (%s) — using random init", ckpt_path, e)
        return ActorCritic(hidden_dim=128, num_layers=2).to(device).eval()
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        cfg_block = ckpt.get("config", {}) or {}
        hidden_dim = int(cfg_block.get("hidden_dim", 128))
        num_layers = int(cfg_block.get("num_layers", 2))
    else:
        state_dict = ckpt
        hidden_dim = 128
        num_layers = 2
    # Dual path: v2 anchor-head checkpoints carry 'anchor_head.weight',
    # v1 Beta-head ones 'raise_head.weight'; the trained obs width (959
    # v1-era vs 991 current) is sniffed from the first torso layer. The
    # bundled critic state (ckpt['critic']) is loaded separately by
    # _load_critic() for the trainer review's all-cards "true EV".
    try:
        model = build_actor_from_state_dict(state_dict, hidden_dim, num_layers)
    except Exception as e:
        logger.warning(
            "checkpoint %s is incompatible with current network (%s) — "
            "using random init", ckpt_path, e,
        )
        model = ActorCritic(hidden_dim=hidden_dim, num_layers=num_layers)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info(
        "loaded checkpoint %s (%s, hidden_dim=%d, num_layers=%d, device=%s)",
        ckpt_path, type(model).__name__, hidden_dim, num_layers, device,
    )
    return model


def _load_critic(device: torch.device) -> CentralCritic | None:
    """Load the centralized critic bundled in the checkpoint (v2 only;
    ckpt['critic'] + head_version>=2). Returns None for v1 / random-init
    / missing critic, in which case the trainer review shows only the
    actor's own (blind) value estimate. One extra torch.load at startup."""
    ckpt_path = Path(os.environ.get("PLO5BP_CHECKPOINT", "checkpoints/stub.pt"))
    if not ckpt_path.exists():
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.warning("failed to load critic from %s (%s)", ckpt_path, e)
        return None
    if not (
        isinstance(ckpt, dict)
        and int(ckpt.get("head_version", 1)) >= 2
        and "critic" in ckpt
    ):
        return None
    cfg_block = ckpt.get("config", {}) or {}
    hidden_dim = int(cfg_block.get("critic_hidden_dim", 1536))
    num_blocks = int(cfg_block.get("critic_num_blocks", 2))
    try:
        critic = CentralCritic(hidden_dim=hidden_dim, num_blocks=num_blocks)
        critic.load_state_dict(ckpt["critic"])
    except Exception as e:
        logger.warning(
            "checkpoint %s critic incompatible (%s) — true-EV disabled",
            ckpt_path, e,
        )
        return None
    critic.to(device).eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    logger.info(
        "loaded centralized critic (hidden_dim=%d, num_blocks=%d, device=%s)",
        hidden_dim, num_blocks, device,
    )
    return critic


MODEL = _load_model()
MODEL_DEVICE = next(MODEL.parameters()).device
MODEL_CRITIC = _load_critic(MODEL_DEVICE)
# v1-era checkpoints (trained at OBS_DIM 959) get the exact downgrade
# projection; current-width models get identity.
OBS_ADAPT = obs_adapter(MODEL)


# --- Session state ----------------------------------------------------------

class Session:
    env: BombPotEnv | None = None
    game_config: GameConfig = GameConfig(starting_stack=400000)
    dollars_per_bb: float = 2.0

    num_seats: int = 6
    button_seat: int = 0
    # Hero is always seat 0 internally; display rotates so it lands at south.
    hero_seat: int = 0

    # Nullable user-entered cards. Server pads nulls with unused deck cards.
    hero_hole: list[int | None] = [None] * 5
    flop_a: list[int | None] = [None] * 3
    flop_b: list[int | None] = [None] * 3
    turn_cards: list[int | None] = [None, None]   # [board_a_turn, board_b_turn]
    river_cards: list[int | None] = [None, None]

    last_obs: np.ndarray | None = None
    last_info: Any = None

    # Action log contains only action entries: {"gate": int, "chips": int}.
    # Street transitions are inferred during replay from engine state.
    action_log: list[dict[str, Any]] = []

    # Seats that weren't dealt into this hand (OCR saw no card-backs).
    # `_rebuild_env` auto-folds any of these seats as soon as the engine
    # makes them the current actor, since the engine has no native
    # "sitting out" state and only the current actor can be folded.
    sitting_out_seats: frozenset[int] = frozenset()

    # Participant mask locked at hand-start: seats the anchor frame saw
    # with cards-back (or banner / live commit). This is the source of
    # truth for "who was dealt in this hand". Per-frame OCR glitches
    # (banner flicker, occlusion) can transiently make `has_cards_back`
    # fail — but once a seat is in this mask, it stays in the hand until
    # the reconstructor emits a genuine FOLD event.
    hand_in_hand_mask: frozenset[int] = frozenset()

    # Seats the reconstructor has emitted a FOLD event for during the
    # current hand. Combined with `hand_in_hand_mask`, this gives the
    # live sitting-out set without needing per-frame detection.
    folded_this_hand: frozenset[int] = frozenset()

    # Directly-observed values from the most recent OCR frame. These
    # are refreshed every tick by `_mirror_observable_state` regardless
    # of hand-boundary detection, so stacks/pot stay live across
    # rewinds and missed hand transitions.
    observed_stacks: tuple[int | None, ...] = ()
    observed_pot: int | None = None

    # Hero's hole cards snapshot at the most recent hand-start trigger.
    # Used to detect a new hand when hero hole re-appears with
    # different cards (rewind-proof fallback).
    last_hero_hole: tuple[int, ...] | None = None

    # Stability gate state: we only commit a new button/participant
    # snapshot after the same value has held for two consecutive ticks,
    # so a single mid-animation frame can't cascade into a bogus
    # hand-start.
    _pending_button: int | None = None
    _pending_sitting_out: frozenset[int] | None = None
    _pending_stable_ticks: int = 0
    # Button-ONLY stability counter, decoupled from the (button, sitting_out)
    # snapshot above. The button is a strong, discrete new-hand signal that
    # never moves mid-hand, so the button-change trigger debounces on this
    # short counter — sitting_out flicker (folds/banners during the deal)
    # no longer resets it. See `_mirror_observable_state`.
    _last_observed_button: int | None = None
    _button_stable_ticks: int = 0
    # Frame captured the tick a new pending snapshot first appeared —
    # used as the pre-commit baseline for stack-seeding and reconstructor
    # rebaselining when the 2-tick stability gate finally commits. Without
    # this, the 2nd stable tick (post-bet) would clobber the pre-bet
    # baseline, making the stack-delta fallback unable to see the action.
    _pending_anchor_fs: Any = None

    # Mid-hand `_begin_new_hand` lock. Counts ticks since the last
    # hand-start fired; once past `_LOCK_AFTER_TICKS` the snapshot
    # debouncer requires `_STABILITY_TICKS_REQUIRED_LOCKED` stable
    # ticks (instead of the bootstrap 2) before button_changed /
    # first_commit can re-fire, and `hero_hole_rotated` runs through
    # its own debouncer instead of firing on a single disjoint frame.
    # Suppresses false hand-restarts from chip-settle / banner OCR
    # flickers without delaying real new-hand detection (which
    # persists for many seconds in practice).
    _ticks_since_hand_start: int = 0
    _pending_hero_hole_rotation: tuple[int, ...] | None = None
    _pending_hero_hole_rotation_ticks: int = 0

    # Mid-hand mask-expansion debouncer. If a non-hero seat reads
    # `folded=False` for two consecutive ticks while NOT in the locked
    # `hand_in_hand_mask` (and not already in `folded_this_hand`), we
    # treat that as "anchor frame missed this participant" and expand
    # the mask. Late rebuyers stay safe because they read folded=True
    # (no cards / banner / commit / timer-bar). Real folds stay safe
    # because the FOLD event puts them in `folded_this_hand`, which
    # the candidate filter excludes.
    _pending_mask_additions: frozenset[int] = frozenset()
    _pending_mask_additions_ticks: int = 0

    # Snapshots captured at first-time street reveal during replay.
    # Compared against current card spec to flag "modified since reveal".
    snapshot_at_turn: dict[str, list[int | None]] | None = None
    snapshot_at_river: dict[str, list[int | None]] | None = None

    # When True, OcrRunner._tick still mirrors cards + runs the
    # debounced hand-start machine, but skips action inference
    # entirely. The user enters all actions via the manual /action
    # endpoint. Workaround for unreliable action detection in the
    # reconstructor (silent CHECKs, banner timing).
    simple_ocr_mode: bool = True

    # Per-slot OCR write lock. Once a card slot has been filled (by
    # OCR or by the user via /cards), the lock for that slot latches
    # to True and OCR will skip it for the rest of the hand. Lets
    # the user override OCR misreads without the next tick clobbering
    # their edit. Reset to all-False by `_new_session_defaults`.
    _card_slot_locked: dict[str, list[bool]] = {
        "hero_hole": [False] * 5,
        "flop_a": [False] * 3,
        "flop_b": [False] * 3,
        "turn_cards": [False, False],
        "river_cards": [False, False],
    }
    # Per-slot stability debounce for the AUTO mirror path: (card_idx, count)
    # of consecutive identical OCR reads, or None. A slot only commits+locks
    # after `_CARD_STABLE_TICKS` identical reads, so a transient mid-reveal
    # misread (dark flipping card -> spurious spade) never latches. Manual
    # rescan bypasses this. Reset by `_new_session_defaults`.
    _card_slot_pending: dict[str, list[tuple[int, int] | None]] = {
        "hero_hole": [None] * 5,
        "flop_a": [None] * 3,
        "flop_b": [None] * 3,
        "turn_cards": [None, None],
        "river_cards": [None, None],
    }


session = Session()

# Consecutive identical auto-OCR reads required before a card slot commits and
# locks. ~600ms at 200ms poll / 300ms at 100ms. Raise if misreads still slip
# through; lower if the commit feels sluggish.
_CARD_STABLE_TICKS = 3


_CARD_SPEC_ATTRS: tuple[tuple[str, int], ...] = (
    ("hero_hole", 5),
    ("flop_a", 3),
    ("flop_b", 3),
    ("turn_cards", 2),
    ("river_cards", 2),
)


def _blank_card_pending() -> dict[str, list[tuple[int, int] | None]]:
    """Fresh per-slot debounce state (all slots empty)."""
    return {attr: [None] * n for attr, n in _CARD_SPEC_ATTRS}


def _lock_filled_card_slots() -> None:
    """Latch the OCR-skip lock on any slot currently holding a card."""
    for attr, _ in _CARD_SPEC_ATTRS:
        spec = getattr(session, attr)
        locks = session._card_slot_locked[attr]
        for i, c in enumerate(spec):
            if c is not None:
                locks[i] = True


def _ocr_apply_card_slot(
    attr: str, idx: int, ocr_card_idx: int | None, debounce: bool = False
) -> None:
    """Write an OCR-detected card into the spec slot if it isn't locked.

    With ``debounce=True`` (the automatic mirror path) the read must repeat
    identically for ``_CARD_STABLE_TICKS`` consecutive ticks before it commits
    and locks — so a transient mid-reveal misread can't latch. ``debounce=False``
    (manual rescan, explicit edits) commits immediately, as before.
    """
    if session._card_slot_locked[attr][idx]:
        return
    if not debounce:
        if ocr_card_idx is None:
            return
        getattr(session, attr)[idx] = ocr_card_idx
        session._card_slot_locked[attr][idx] = True
        return

    pending = session._card_slot_pending[attr]
    if ocr_card_idx is None:
        # No confident read this tick — break the run.
        pending[idx] = None
        return
    prev = pending[idx]
    count = prev[1] + 1 if (prev is not None and prev[0] == ocr_card_idx) else 1
    pending[idx] = (ocr_card_idx, count)
    if count >= _CARD_STABLE_TICKS:
        getattr(session, attr)[idx] = ocr_card_idx
        session._card_slot_locked[attr][idx] = True
        pending[idx] = None


def _new_session_defaults() -> None:
    """Reset all per-hand state. Keeps config + dollars_per_bb."""
    session.hero_hole = [None] * 5
    session.flop_a = [None] * 3
    session.flop_b = [None] * 3
    session.turn_cards = [None, None]
    session.river_cards = [None, None]
    session.action_log = []
    session.snapshot_at_turn = None
    session.snapshot_at_river = None
    session.last_obs = None
    session.last_info = None
    session.sitting_out_seats = frozenset()
    session.hand_in_hand_mask = frozenset()
    session.folded_this_hand = frozenset()
    session._pending_mask_additions = frozenset()
    session._pending_mask_additions_ticks = 0
    session._ticks_since_hand_start = 0
    session._pending_hero_hole_rotation = None
    session._pending_hero_hole_rotation_ticks = 0
    session._last_observed_button = None
    session._button_stable_ticks = 0
    session._card_slot_locked = {
        "hero_hole": [False] * 5,
        "flop_a": [False] * 3,
        "flop_b": [False] * 3,
        "turn_cards": [False, False],
        "river_cards": [False, False],
    }
    session._card_slot_pending = _blank_card_pending()


def _clear_hand_state_keep_cards() -> None:
    """Clear action log + snapshots but keep card spec intact."""
    session.action_log = []
    session.snapshot_at_turn = None
    session.snapshot_at_river = None
    session.last_obs = None
    session.last_info = None


def _current_total_commit(num_seats: int, ante: int) -> list[int]:
    """Per-seat total_commit at the current replay end, in engine chips.

    Used by /config and /seats to convert UI "current behind" inputs
    to engine starting_stack via `engine_starting = behind + commit`.
    Falls back to `[ante] * num_seats` when no env exists yet (cold
    start) or the seat-count just changed.
    """
    if session.env is not None:
        try:
            raw = dict(session.env._rs.observation_dict())
            tc = raw.get("total_commit") or []
            commits = [int(x) for x in tc]
            if len(commits) == num_seats:
                return commits
        except Exception:
            pass
    return [int(ante)] * num_seats


# --- Card padding -----------------------------------------------------------

def _pad_all() -> dict[str, list[int]]:
    """Resolve nullable card spec to a fully-padded 11+4 deck slice.

    User-entered cards go in first; nulls are filled by scanning 0..51
    and taking the lowest unused index. Raises HTTP 400 on duplicates.
    """
    user_cards: list[int] = []
    for spec in (
        session.hero_hole, session.flop_a, session.flop_b,
        session.turn_cards, session.river_cards,
    ):
        for c in spec:
            if c is not None:
                user_cards.append(int(c))
    if len(set(user_cards)) != len(user_cards):
        raise HTTPException(status_code=400, detail="duplicate card in spec")
    for c in user_cards:
        if not (0 <= c < 52):
            raise HTTPException(status_code=400, detail=f"card {c} out of range")
    used = set(user_cards)

    def pad(spec: list[int | None]) -> list[int]:
        out: list[int] = []
        for c in spec:
            if c is not None:
                out.append(int(c))
            else:
                for cand in range(52):
                    if cand not in used:
                        out.append(cand)
                        used.add(cand)
                        break
        return out

    return {
        "hero_hole": pad(session.hero_hole),
        "flop_a": pad(session.flop_a),
        "flop_b": pad(session.flop_b),
        "turn": pad(session.turn_cards),
        "river": pad(session.river_cards),
    }


# --- Rebuild env (canonical path) -------------------------------------------

def _rebuild_env() -> None:
    """Rebuild env from session state by padding + replaying action_log.

    Updates snapshot_at_turn / snapshot_at_river when replay reaches those
    streets for the first time. If a previous replay reached a later
    street but this one doesn't (e.g. after /undo), clears the stale
    snapshot so the "modified" indicator stops showing.
    """
    padded = _pad_all()
    cfg = session.game_config

    # Build the in-hand mask from the locked hand-start membership. When the
    # session has not yet committed a hand-start (mask is empty) we pass
    # None — the engine treats every seat as in-hand, matching the prior
    # default-stack behaviour. Once a real mask is available, sitting-out
    # seats no longer post antes (engine-side); this collapses the inflated
    # 6-seat pot down to the real heads-up/3-way pot the OCR sees.
    in_hand_mask: list[bool] | None
    if session.hand_in_hand_mask:
        in_hand_mask = [
            i in session.hand_in_hand_mask for i in range(session.num_seats)
        ]
        if int(session.hero_seat) not in session.hand_in_hand_mask:
            in_hand_mask = None
        elif sum(in_hand_mask) < 2:
            in_hand_mask = None
    else:
        in_hand_mask = None

    env = BombPotEnv(cfg)
    try:
        env.reset_study(
            button=int(session.button_seat),
            hero_seat=int(session.hero_seat),
            hero_hole=list(padded["hero_hole"]),
            flop_a=list(padded["flop_a"]),
            flop_b=list(padded["flop_b"]),
            in_hand_mask=in_hand_mask,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    reached_turn = False
    reached_river = False

    def _advance_streets() -> None:
        nonlocal reached_turn, reached_river
        while True:
            awaiting = env.awaiting_next_street()
            if awaiting == 2:
                env.set_turn(int(padded["turn"][0]), int(padded["turn"][1]))
                reached_turn = True
            elif awaiting == 3:
                env.set_river(int(padded["river"][0]), int(padded["river"][1]))
                reached_river = True
            else:
                break

    def _auto_fold_sitting_out() -> None:
        """Retire a structurally-sitting-out seat the moment it becomes
        the actor.

        Only seats NEVER dealt into this hand
        (``all_seats - hand_in_hand_mask``) are folded here. Seats that
        folded mid-hand already have a FOLD entry in
        ``session.action_log``; they'll be applied to the correct seat
        by the replay loop. Including them here would step_hybrid-fold
        them BEFORE their action_log entry runs, advancing current_actor
        past them so the log's FOLD lands on the next non-folded seat
        (phantom fold).

        Folding a free-check position is illegal in this engine, so when
        `bet_to_call == 0` we feed a check_call instead. That keeps the
        seat in the pot across that street but burns their turn with a
        benign action; when a real bet lands later, the next pass folds
        them for real. Streets may advance as a result, so we interleave
        with `_advance_streets`.
        """
        if not session.hand_in_hand_mask:
            return
        all_seats = frozenset(range(session.num_seats))
        structurally_sitting = all_seats - session.hand_in_hand_mask
        if not structurally_sitting:
            return
        for _ in range(session.num_seats * 4):
            _advance_streets()
            actor = env.current_actor()
            if actor is None or int(actor) not in structurally_sitting:
                return
            raw = dict(env._rs.observation_dict())
            bet_to_call = int(raw.get("bet_to_call") or 0)
            gate = int(GATE_FOLD) if bet_to_call > 0 else int(GATE_CHECK_CALL)
            env.step_hybrid(gate, 0)

    # Skip entries the engine rejects rather than stalling env rebuild.
    # A single bad entry (e.g. a spurious sub-1bb raise from an OCR
    # glitch that slipped past events.py's guard) would otherwise freeze
    # `session.env` on whatever the last successful rebuild produced,
    # leaving the UI showing stack/pot state from a prior hand. When we
    # encounter one, log it, drop it from `session.action_log` for
    # future ticks, and continue replaying the rest — the session stays
    # in sync with the current hand and the dropped event surfaces as a
    # warning.
    kept: list[dict[str, Any]] = []
    for entry in session.action_log:
        _advance_streets()
        _auto_fold_sitting_out()
        try:
            env.step_hybrid(int(entry["gate"]), int(entry["chips"]))
            kept.append(entry)
        except Exception as e:
            logger.warning(
                "rebuild_env skipping illegal action %s: %s", entry, e
            )
    if len(kept) != len(session.action_log):
        session.action_log = kept

    _advance_streets()
    _auto_fold_sitting_out()

    if reached_turn:
        if session.snapshot_at_turn is None:
            session.snapshot_at_turn = {
                "flop_a": list(session.flop_a),
                "flop_b": list(session.flop_b),
            }
    else:
        session.snapshot_at_turn = None

    if reached_river:
        if session.snapshot_at_river is None:
            session.snapshot_at_river = {
                "flop_a": list(session.flop_a),
                "flop_b": list(session.flop_b),
                "turn": list(session.turn_cards),
            }
    else:
        session.snapshot_at_river = None

    session.env = env
    _refresh_obs()


def _refresh_obs() -> None:
    env = session.env
    assert env is not None
    if env.current_actor() is None:
        session.last_obs = None
        session.last_info = None
        return
    obs_vec, info = env._pack_obs()
    session.last_obs = obs_vec
    session.last_info = info


# --- Request models ---------------------------------------------------------

class ActionRequest(BaseModel):
    gate: str = Field(..., pattern=r"^(fold|check_call|raise)$")
    chips: int | None = Field(default=None, ge=0)


class CardsRequest(BaseModel):
    hero_hole: list[int | None] = Field(default_factory=lambda: [None] * 5)
    flop_a: list[int | None] = Field(default_factory=lambda: [None] * 3)
    flop_b: list[int | None] = Field(default_factory=lambda: [None] * 3)
    turn: list[int | None] = Field(default_factory=lambda: [None, None])
    river: list[int | None] = Field(default_factory=lambda: [None, None])


class SeatsRequest(BaseModel):
    num_seats: int | None = Field(default=None, ge=2, le=6)
    button_seat: int | None = Field(default=None, ge=0, le=5)
    starting_stacks: list[int] | None = None


class ConfigRequest(BaseModel):
    bb_chips: int | None = Field(default=None, ge=1)
    ante_chips: int | None = Field(default=None, ge=0)
    dollars_per_bb: float | None = Field(default=None, gt=0)
    starting_stacks: list[int] | None = None


# --- Helpers ----------------------------------------------------------------

_GATE_NAME_TO_IDX = {
    "fold": GATE_FOLD,
    "check_call": GATE_CHECK_CALL,
    "raise": GATE_RAISE,
}


def _chips_to_bb(chips: int | float) -> float:
    return float(chips) / float(session.game_config.bb)


def _ocr_cents_to_engine_chips(cents: int) -> int:
    """Convert an OCR-cent amount (ClubGG dollar display × 100) to engine chips.

    OCR reads ClubGG stack text in cents (100 = $1). The engine carries
    chip counts where `cfg.bb` chips = 1 big blind = `dollars_per_bb`
    dollars. At the defaults (bb=10000, $20/bb) that works out to 5
    engine-chips per cent. Doing the mixed-unit arithmetic that used to
    live in ``_reset_for_new_hand`` (raw_cents + cfg.ante_chips) is a
    bug; always run cents through this helper first.
    """
    cfg = session.game_config
    dpb = float(session.dollars_per_bb)
    if dpb <= 0:
        return int(cents)
    return int(round(int(cents) * float(cfg.bb) / (100.0 * dpb)))


def _seat_action_to_log_entry(ev: Any) -> dict[str, int]:
    """Translate a ``SeatAction`` OCR event into an ``action_log`` entry.

    ``SeatAction.chips`` is already a raise-by delta in engine-chips
    (the reconstructor converts OCR cents at its boundary); no unit
    fixup needed here.
    """
    return {"gate": int(_GATE_NAME_TO_IDX[ev.gate]), "chips": int(ev.chips)}


def _reconcile_missed_folds_on_street_reveal(fs: Any) -> None:
    """Append FOLD entries for in-hand seats the walk missed (Fix K).

    Triggered only when a ``StreetReveal`` fires. The guard is that
    ``StreetReveal`` already requires both board_a and board_b to
    show 4+ cards (see ocr.events), which rules out the bomb-pot
    intro animation flicker.

    Attribution by count, not by seat: ``step_hybrid`` applies
    actions to the engine's own ``current_actor``, so appending N
    bare FOLD entries retires the next N actors. That's fine because
    the walk already handled any non-fold actions ahead of the
    missed folds — if a call had sat between them, the engine
    would have advanced past those seats before the missed folds
    reached the front of the queue.
    """
    in_hand = session.hand_in_hand_mask
    if not in_hand:
        return
    fs_by_seat = {s.seat: s for s in fs.seats}
    visually_folded = frozenset(
        seat
        for seat in in_hand
        if fs_by_seat.get(seat) is not None and fs_by_seat[seat].folded
    )
    missing = visually_folded - session.folded_this_hand
    if not missing:
        return
    for _ in missing:
        session.action_log.append({"gate": int(GATE_FOLD), "chips": 0})
    session.folded_this_hand = frozenset(
        session.folded_this_hand | missing
    )
    all_seats = frozenset(range(session.num_seats))
    session.sitting_out_seats = (
        (all_seats - session.hand_in_hand_mask) | session.folded_this_hand
    )
    logger.warning(
        "ocr: StreetReveal fold reconcile — added %d FOLD entries "
        "for seats %s (walk missed these folds)",
        len(missing),
        sorted(missing),
    )


def _reconcile_missed_checks_on_street_reveal(target_street: int) -> None:
    """Append CHECK_CALL entries when the walk missed a pure-check round.

    Loop-fills: appends one CHECK_CALL, rebuilds the engine, rechecks
    ``view.street`` — until the engine has advanced to ``target_street``.
    The engine's own street-advance logic decides when each fill is
    enough, so this is correct whether the walk emitted 0, 1, or N
    CHECKs for the current street before the reconciler ran. The old
    ``len(active) * streets_to_fill`` formula assumed the engine was at
    the start of the current street, but step 5 of the walk ladder
    (events.py ``_infer_seat_actions``) routinely emits a CHECK for the
    first actor on a quiet street — leaving the engine partially
    advanced and the formula over-counting by exactly that emission.

    Safety: only fills when ``bet_to_call == 0`` and no seat has
    committed chips on the current street; positive evidence means a
    real bet was missed and we leave the engine stuck rather than
    silently mis-attribute. A loop-count guard caps runaway in case
    the engine refuses to advance.
    """
    _rebuild_env()
    if session.env is None:
        return
    view = _engine_view_from_session()
    if os.environ.get("PLO5BP_OCR_DEBUG_TIMER"):
        logger.warning(
            "ocr.reconcile.checks: target=%d view.street=%d "
            "view.bet_to_call=%d committed=%s",
            target_street, int(view.street), int(view.bet_to_call),
            tuple(int(c) for c in view.committed_this_street),
        )
    if view.street >= target_street:
        return
    if view.bet_to_call > 0 or any(int(c) > 0 for c in view.committed_this_street):
        logger.warning(
            "ocr: StreetReveal check reconcile — skipping; "
            "bet_to_call=%d committed=%s (real bets must have been missed)",
            int(view.bet_to_call),
            tuple(int(c) for c in view.committed_this_street),
        )
        return
    if not session.hand_in_hand_mask:
        return
    if not (session.hand_in_hand_mask - session.folded_this_hand):
        return

    cfg = session.game_config
    start_street = int(view.street)
    safety_limit = cfg.num_seats * (target_street - start_street) + cfg.num_seats
    appended = 0
    while view.street < target_street and appended < safety_limit:
        session.action_log.append({"gate": int(GATE_CHECK_CALL), "chips": 0})
        appended += 1
        _rebuild_env()
        view = _engine_view_from_session()

    if view.street < target_street:
        logger.warning(
            "ocr: StreetReveal check reconcile — safety limit hit at "
            "appended=%d (start_street=%d target=%d view.street=%d)",
            appended, start_street, target_street, int(view.street),
        )
    else:
        logger.warning(
            "ocr: StreetReveal check reconcile — appended %d CHECK_CALL "
            "entries (street %d → %d)",
            appended, start_street, int(view.street),
        )


def _position_name(seat: int) -> str:
    """Resolve position label by walking physical CW from button.

    Under the post-Option-B ROI/UI convention, increasing seat index
    IS physical-clockwise (0=bottom-center, 1=bottom-left, 2=top-left,
    3=top-center, 4=top-right, 5=bottom-right). This matches the
    engine's `(actor + 1) % n` advancement direction, so CW position
    labels also walk `(button + offset) % n`.

    Sitting-out seats are skipped during the walk so a 6-seat session
    with two players sat out labels the remaining four as BTN, SB, BB,
    UTG (not BTN, SB, BB, HJ with UTG/CO phantom-assigned to empty
    seats). Uses `hand_in_hand_mask` (locked at hand-start) rather than
    `sitting_out_seats` so mid-hand folds don't shift labels.
    """
    return _common_position_name(
        seat,
        session.button_seat,
        session.num_seats,
        session.hand_in_hand_mask or None,
    )


def _history_entries(obs: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for seat, action_idx, chips, street_idx in obs.get("history", []):
        entries.append({
            "seat": int(seat),
            "position": _position_name(int(seat)),
            "action": _HISTORY_NAMES[int(action_idx)],
            "chips": int(chips),
            "chips_bb": round(_chips_to_bb(int(chips)), 4),
            "street": STREET_NAMES.get(int(street_idx), str(street_idx)),
        })
    return entries


def _hero_info_complete() -> bool:
    return all(c is not None for c in session.hero_hole)


def _hero_blocking_reason() -> str | None:
    """Return None if hero can act; else 'hole'|'flop'|'turn'|'river'."""
    if not all(c is not None for c in session.hero_hole):
        return "hole"
    env = session.env
    if env is None:
        return None
    raw = env._rs.observation_dict()
    street = int(raw["street"])
    if street >= 1:
        if not all(c is not None for c in session.flop_a):
            return "flop"
        if not all(c is not None for c in session.flop_b):
            return "flop"
    if street >= 2 and not all(c is not None for c in session.turn_cards):
        return "turn"
    if street >= 3 and not all(c is not None for c in session.river_cards):
        return "river"
    return None


def _network_obs() -> np.ndarray | None:
    """Encoded observation the network sees.

    OCR mode keeps `cfg.num_seats == 6` with sit-out seats auto-folded for
    the engine, but the network was trained on `num_seats ∈ {2..6}` configs
    where every sampled seat is dealt in — never "6 seats with N starting
    folded." Re-encode the obs against a `GameConfig` whose `num_seats`
    equals the actual in-hand count, so the network sees an in-distribution
    view. In manual mode this returns `session.last_obs` unchanged.
    """
    env = session.env
    if env is None or env.current_actor() is None:
        return None
    cfg = session.game_config
    in_hand = sorted(session.hand_in_hand_mask) if session.hand_in_hand_mask else None
    if not in_hand or len(in_hand) == cfg.num_seats:
        return session.last_obs

    hero = int(session.hero_seat)
    if hero not in session.hand_in_hand_mask:
        return session.last_obs
    comp_to_phys = [
        (hero + k) % cfg.num_seats
        for k in range(cfg.num_seats)
        if (hero + k) % cfg.num_seats in session.hand_in_hand_mask
    ]
    n = len(comp_to_phys)
    phys_to_comp = {p: i for i, p in enumerate(comp_to_phys)}

    raw = dict(env._rs.observation_dict())
    proj = dict(raw)
    for key in (
        "folded",
        "all_in",
        "stacks",
        "eff_stack_cap",
        "street_commit",
        "total_commit",
    ):
        proj[key] = [raw[key][p] for p in comp_to_phys]
    proj["actor"] = 0
    raw_agg = int(raw.get("last_aggressor", -1))
    proj["last_aggressor"] = phys_to_comp.get(raw_agg, -1) if raw_agg >= 0 else -1
    proj["button"] = phys_to_comp.get(int(raw["button"]), 0)
    proj["history"] = [
        (phys_to_comp[s], a, c, st)
        for (s, a, c, st) in raw["history"]
        if s in phys_to_comp
    ]
    # Mirror env._pack_obs: the Rust observation_dict() omits hero_category_*;
    # without these the encoder defaults both boards to high-card (cat 0).
    raw_actor = int(raw["actor"])
    proj["hero_category_a"] = int(env._rs.hero_category(raw_actor, 0))
    proj["hero_category_b"] = int(env._rs.hero_category(raw_actor, 1))

    compressed_cfg = GameConfig(
        num_seats=n,
        starting_stack=cfg.starting_stack,
        ante=cfg.ante,
        bb=cfg.bb,
        starting_stacks=tuple(int(cfg.resolved_stacks[p]) for p in comp_to_phys),
    )
    return encode_observation(proj, compressed_cfg)


def _compute_recommendation() -> dict[str, Any] | None:
    """Single deterministic recommendation: argmax gate + Beta-mean chips.

    Gated on hero_info_complete — if hero's 5 hole cards aren't all placed
    we return None so the UI shows a muted placeholder.
    """
    env = session.env
    if env is None:
        return None
    actor = env.current_actor()
    if actor is None or actor != session.hero_seat:
        return None
    if _hero_blocking_reason() is not None:
        return None
    if session.last_obs is None or session.last_info is None:
        return None
    info = session.last_info
    obs_np = _network_obs()
    if obs_np is None:
        return None
    obs_t = torch.from_numpy(OBS_ADAPT(obs_np)).unsqueeze(0).to(MODEL_DEVICE)
    gm_t = torch.from_numpy(info.gate_mask).unsqueeze(0).to(MODEL_DEVICE)
    if getattr(MODEL, "head_version", 1) >= 2:
        return _recommendation_v2(obs_t, gm_t, info)
    raise_max = int(info.max_raise_chips)
    raise_min = min(int(info.min_raise_chips), raise_max)
    bounds_t = torch.tensor(
        [[raise_min, raise_max]], dtype=torch.long, device=MODEL_DEVICE
    )
    with torch.no_grad():
        gate_logits, raise_params, value = MODEL(obs_t, gm_t)
        gate_probs = F.softmax(gate_logits, dim=-1).squeeze(0).tolist()
        _act_out = MODEL.act(
            obs_t, gm_t, bounds_t, deterministic=True
        )
        gate = int(_act_out.gate.item())
        chips = int(_act_out.chips.item())
        alpha = float(raise_params[0, 0].item())
        beta = float(raise_params[0, 1].item())
        value_bb = float(value.squeeze(0).item())

    chips_out = chips if gate == GATE_RAISE else None
    # Short-shove redirect: when min_raise==0 and raise gate legal, the env
    # dispatches apply(AllIn) regardless of the Beta sample. Surface the
    # actual chips that will be committed (the all-in amount) so the UI
    # doesn't lie about the network's recommendation.
    if chips_out is not None and raise_min == 0 and raise_max > 0:
        chips_out = raise_max
    chips_bb = round(_chips_to_bb(chips_out), 4) if chips_out is not None else None
    gate_slug = (
        "fold" if gate == GATE_FOLD else
        "check_call" if gate == GATE_CHECK_CALL else
        "raise"
    )
    return {
        "gate": gate_slug,
        "gate_name": GATE_NAMES[gate],
        "chips": chips_out,
        "chips_bb": chips_bb,
        "value_bb": round(value_bb, 4),
        "gate_distribution": [round(p, 4) for p in gate_probs],
        "beta_alpha": round(alpha, 4),
        "beta_beta": round(beta, 4),
    }


def _recommendation_v2(
    obs_t: torch.Tensor, gm_t: torch.Tensor, info: Any
) -> dict[str, Any]:
    """v2 (anchor head) recommendation: argmax gate + argmax legal anchor
    with its refinement Beta. The server computes every anchor's chips —
    the client never recomputes sizing math."""
    sizing = sizing_from_info(info)
    sizing_t = torch.from_numpy(sizing[None, :]).to(MODEL_DEVICE)
    with torch.no_grad():
        gate_logits, anchor_logits, refine, value = MODEL(obs_t, gm_t)
        gate_probs = F.softmax(gate_logits, dim=-1).squeeze(0).tolist()
        _act_out = MODEL.act(obs_t, gm_t, sizing_t, deterministic=True)
        gate = int(_act_out.gate.item())
        chips = int(_act_out.chips.item())
        rec_anchor = int(_act_out.anchor.item())
        value_bb = float(value.squeeze(0).item())
        anchor_np = anchor_logits.squeeze(0).float().cpu().numpy()
        refine_np = refine.squeeze(0).float().cpu().numpy()  # (9, 2)

    grid = anchor_grid_np(sizing[0], sizing[1], sizing[2], sizing[3])
    masked = np.where(grid.legal, anchor_np, -1e9)
    exps = np.exp(masked - masked.max())
    anchor_probs = exps / exps.sum()
    anchors = [
        {
            "k": int(k),
            "frac": k / 10.0,
            "label": _anchor_label(int(k)),
            "prob": round(float(anchor_probs[k]), 4),
            "chips": int(grid.chips[k]),
            "chips_bb": round(_chips_to_bb(int(grid.chips[k])), 4),
        }
        for k in range(ANCHOR_COUNT)
        if bool(grid.legal[k])
    ]
    refine_block = None
    if bool(grid.refine_ok[rec_anchor]):
        alpha, beta = refine_np[rec_anchor - 1]
        refine_block = {
            "alpha": round(float(alpha), 4),
            "beta": round(float(beta), 4),
            "frac_lo": rec_anchor / 10.0 - BRACKET_HALF,
            "frac_hi": rec_anchor / 10.0 + BRACKET_HALF,
        }

    chips_out = chips if gate == GATE_RAISE else None
    chips_bb = round(_chips_to_bb(chips_out), 4) if chips_out is not None else None
    gate_slug = (
        "fold" if gate == GATE_FOLD else
        "check_call" if gate == GATE_CHECK_CALL else
        "raise"
    )
    return {
        "head_version": 2,
        "gate": gate_slug,
        "gate_name": GATE_NAMES[gate],
        "chips": chips_out,
        "chips_bb": chips_bb,
        "value_bb": round(value_bb, 4),
        "gate_distribution": [round(p, 4) for p in gate_probs],
        "anchors": anchors,
        "rec_anchor": rec_anchor,
        "refine": refine_block,
    }


def _legal_block(info: Any, actor: int) -> dict[str, bool]:
    gm = info.gate_mask
    legal = {
        "fold": bool(gm[GATE_FOLD]),
        "check_call": bool(gm[GATE_CHECK_CALL]),
        "raise": bool(gm[GATE_RAISE]),
    }
    # Hero actions blocked until hero hole + cards through current street are entered.
    if actor == session.hero_seat and _hero_blocking_reason() is not None:
        legal = {k: False for k in legal}
    return legal


def _to_call_chips(obs: dict[str, Any], actor: int) -> int:
    current_commit = int(obs["street_commit"][actor])
    stack = int(obs["stacks"][actor])
    bet_to_call = int(obs["bet_to_call"])
    return min(max(0, bet_to_call - current_commit), stack)


def _modified_cards() -> list[dict[str, Any]]:
    """Compare current card spec against saved snapshots.

    Each flagged slot is a dict `{slot_key, index}` that the client uses
    to render a "modified since reveal" indicator.
    """
    out: list[dict[str, Any]] = []
    snap_turn = session.snapshot_at_turn
    snap_river = session.snapshot_at_river

    def compare(slot_key: str, current: list[int | None], snap: list[int | None]) -> None:
        for i, (cur, was) in enumerate(zip(current, snap)):
            if cur != was:
                out.append({"slot_key": slot_key, "index": i})

    if snap_turn is not None:
        compare("flop_a", session.flop_a, snap_turn["flop_a"])
        compare("flop_b", session.flop_b, snap_turn["flop_b"])
    if snap_river is not None:
        compare("flop_a", session.flop_a, snap_river["flop_a"])
        compare("flop_b", session.flop_b, snap_river["flop_b"])
        compare("turn", session.turn_cards, snap_river["turn"])

    # Deduplicate by (slot_key, index).
    seen = set()
    unique: list[dict[str, Any]] = []
    for m in out:
        key = (m["slot_key"], m["index"])
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


def _state_dict() -> dict[str, Any]:
    env = session.env
    assert env is not None
    raw = dict(env._rs.observation_dict())

    actor_raw = raw.get("actor")
    actor = int(actor_raw) if actor_raw is not None else None

    awaiting_idx = raw.get("awaiting_next_street")
    study_term_idx = raw.get("study_terminal")
    terminal = (
        TERMINAL_NAMES.get(int(study_term_idx)) if study_term_idx is not None else None
    )
    terminal_message = TERMINAL_MESSAGES.get(terminal) if terminal is not None else None
    awaiting = AWAITING_NAMES.get(int(awaiting_idx)) if awaiting_idx is not None else None

    cfg = session.game_config
    mask = session.hand_in_hand_mask
    seats: list[dict[str, Any]] = []
    hero_hole_shown = [c for c in session.hero_hole if c is not None] \
        if any(c is not None for c in session.hero_hole) else None
    for seat in range(cfg.num_seats):
        hole = hero_hole_shown if seat == session.hero_seat else None
        seats.append({
            "seat": seat,
            "position": _position_name(seat),
            "stack_chips": int(raw["stacks"][seat]),
            "stack_bb": round(_chips_to_bb(int(raw["stacks"][seat])), 4),
            "committed_this_street_bb": round(
                _chips_to_bb(int(raw["street_commit"][seat])), 4
            ),
            "committed_total_bb": round(
                _chips_to_bb(int(raw["total_commit"][seat])), 4
            ),
            "committed_this_street_chips": int(raw["street_commit"][seat]),
            "folded": bool(raw["folded"][seat]),
            # Mid-hand folds stay visible (they were dealt into this hand
            # — rendering them as `folded` rather than hiding them matches
            # the real table). Seats never dealt in are hidden as before.
            "participant": (not mask) or (seat in mask),
            "all_in": bool(raw["all_in"][seat]),
            "is_actor": actor is not None and seat == actor,
            "is_hero": seat == session.hero_seat,
            "hole": hole,
        })

    if actor is not None and session.last_info is not None:
        info = session.last_info
        legal = _legal_block(info, actor)
        max_chips = int(info.max_raise_chips)
        min_chips = min(int(info.min_raise_chips), max_chips)
        # Short-shove: only legal raise is the all-in shove. Collapse the
        # slider to a single point so the UI can render an All-in button.
        if legal["raise"] and min_chips == 0 and max_chips > 0:
            min_chips = max_chips
        raise_bounds = {
            "min_chips": min_chips,
            "max_chips": max_chips,
            "min_bb": round(_chips_to_bb(min_chips), 4),
            "max_bb": round(_chips_to_bb(max_chips), 4),
        }
        to_call = _to_call_chips(raw, actor)
    else:
        legal = {k: False for k in ("fold", "check_call", "raise")}
        raise_bounds = {"min_chips": 0, "max_chips": 0, "min_bb": 0.0, "max_bb": 0.0}
        to_call = 0

    recommendation = _compute_recommendation()

    return {
        "num_seats": cfg.num_seats,
        "button_seat": session.button_seat,
        "hero_seat": session.hero_seat,
        "actor": actor,
        "seats": seats,
        "card_spec": {
            "hero_hole": list(session.hero_hole),
            "flop_a": list(session.flop_a),
            "flop_b": list(session.flop_b),
            "turn": list(session.turn_cards),
            "river": list(session.river_cards),
        },
        "hero_info_complete": _hero_info_complete(),
        "hero_blocking_reason": _hero_blocking_reason(),
        "modified_cards": _modified_cards(),
        "pot_chips": int(raw["pot"]),
        "pot_bb": round(_chips_to_bb(int(raw["pot"])), 4),
        # Settled pot = the pot gathered from completed streets (pot minus
        # this street's live commits, which still sit in front of seats).
        # "Total Pot" = pot_chips; "Pot" = settled_pot_chips.
        "settled_pot_chips": int(raw["pot"]) - sum(int(x) for x in raw["street_commit"]),
        "settled_pot_bb": round(
            _chips_to_bb(int(raw["pot"]) - sum(int(x) for x in raw["street_commit"])), 4
        ),
        "bet_to_call_chips": int(raw["bet_to_call"]),
        "bet_to_call_bb": round(_chips_to_bb(int(raw["bet_to_call"])), 4),
        "to_call_chips": int(to_call),
        "to_call_bb": round(_chips_to_bb(int(to_call)), 4),
        "street": STREET_NAMES.get(int(raw["street"]), "flop"),
        "history": _history_entries(raw),
        "legal": legal,
        "raise_bounds": raise_bounds,
        "terminal": terminal,
        "terminal_message": terminal_message,
        "awaiting_next_street": awaiting,
        "recommendation": recommendation,
        "can_undo": len(session.action_log) > 0,
        "chip_scale": {
            "bb_chips": int(cfg.bb),
            "ante_chips": int(cfg.ante),
            "dollars_per_bb": float(session.dollars_per_bb),
        },
        "starting_stacks_chips": [int(s) for s in cfg.resolved_stacks],
        "starting_stacks_bb": [
            round(_chips_to_bb(int(s)), 4) for s in cfg.resolved_stacks
        ],
        "simple_ocr_mode": bool(session.simple_ocr_mode),
    }


# --- Validation helpers -----------------------------------------------------

def _validate_card_list(xs: list[int | None], length: int, name: str) -> list[int | None]:
    if len(xs) != length:
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be length {length}, got {len(xs)}",
        )
    out: list[int | None] = []
    for x in xs:
        if x is None:
            out.append(None)
        else:
            xi = int(x)
            if not (0 <= xi < 52):
                raise HTTPException(
                    status_code=400,
                    detail=f"{name}: card {xi} out of range [0,51]",
                )
            out.append(xi)
    return out


# --- FastAPI app ------------------------------------------------------------

app = FastAPI(title="PLO5 Bomb-Pot Study Tool")

# Trainer mode rides on the same app/model under /trainer/*. Import here
# (not at top) so trainer.py never needs to import server.py back.
from plo5bp.ui.trainer import create_trainer_router  # noqa: E402

trainer_router = create_trainer_router(MODEL, MODEL_DEVICE, critic=MODEL_CRITIC)
app.include_router(trainer_router)


@app.get("/state")
def state() -> dict[str, Any]:
    if session.env is None:
        _rebuild_env()
    return {"state": _state_dict()}


@app.post("/cards")
def cards(req: CardsRequest) -> dict[str, Any]:
    session.hero_hole = _validate_card_list(req.hero_hole, 5, "hero_hole")
    session.flop_a = _validate_card_list(req.flop_a, 3, "flop_a")
    session.flop_b = _validate_card_list(req.flop_b, 3, "flop_b")
    session.turn_cards = _validate_card_list(req.turn, 2, "turn")
    session.river_cards = _validate_card_list(req.river, 2, "river")
    _lock_filled_card_slots()
    _rebuild_env()
    return {"state": _state_dict()}


@app.post("/seats")
def seats(req: SeatsRequest) -> dict[str, Any]:
    cfg = session.game_config
    new_num_seats = req.num_seats if req.num_seats is not None else session.num_seats
    new_button = req.button_seat if req.button_seat is not None else session.button_seat
    if new_button >= new_num_seats:
        new_button = 0
    seats_or_button_changed = (
        new_num_seats != cfg.num_seats or new_button != session.button_seat
    )

    if req.starting_stacks is not None:
        behinds = tuple(int(s) for s in req.starting_stacks)
        if len(behinds) != new_num_seats:
            raise HTTPException(
                status_code=400,
                detail=f"starting_stacks length {len(behinds)} != num_seats {new_num_seats}",
            )
        # When seat count changes, the existing env's total_commit
        # has the wrong shape; the helper returns [ante]*new_num_seats
        # in that case, which matches a fresh-hand conversion.
        if seats_or_button_changed:
            commits = [int(cfg.ante)] * new_num_seats
        else:
            commits = _current_total_commit(new_num_seats, int(cfg.ante))
        engine_stacks = tuple(b + commits[i] for i, b in enumerate(behinds))
        session.game_config = GameConfig(
            num_seats=new_num_seats,
            starting_stack=engine_stacks[0],
            ante=cfg.ante,
            bb=cfg.bb,
            starting_stacks=engine_stacks,
        )
    elif new_num_seats != cfg.num_seats:
        session.game_config = GameConfig(
            num_seats=new_num_seats,
            starting_stack=cfg.starting_stack,
            ante=cfg.ante,
            bb=cfg.bb,
        )
    session.num_seats = new_num_seats
    session.button_seat = new_button
    # Hero stays at seat 0 internally; UI rotates so it lands at south.
    session.hero_seat = 0
    if seats_or_button_changed:
        _clear_hand_state_keep_cards()
    _rebuild_env()
    return {"state": _state_dict()}


@app.post("/action")
def action(req: ActionRequest) -> dict[str, Any]:
    if session.env is None:
        _rebuild_env()
    if session.last_info is None:
        raise HTTPException(status_code=400, detail="no actor — hand may be terminal")
    info = session.last_info
    env = session.env
    assert env is not None

    gate_idx = _GATE_NAME_TO_IDX[req.gate]
    actor = env.current_actor()
    assert actor is not None
    # Hero gate enforcement: refuse hero actions until hole cards and the
    # board cards needed for the current street are all entered.
    if actor == session.hero_seat:
        reason = _hero_blocking_reason()
        if reason is not None:
            detail = {
                "hole":  "hero hole cards required before hero can act",
                "flop":  "flop cards required before hero can act",
                "turn":  "turn cards required before hero can act",
                "river": "river cards required before hero can act",
            }[reason]
            raise HTTPException(status_code=400, detail=detail)
    if not info.gate_mask[gate_idx]:
        raise HTTPException(
            status_code=400, detail=f"gate {req.gate!r} not legal"
        )

    chips = 0
    if gate_idx == GATE_RAISE:
        if req.chips is None:
            raise HTTPException(
                status_code=400, detail="chips required for gate 'raise'"
            )
        chips = int(req.chips)
        lo = int(info.min_raise_chips)
        hi = int(info.max_raise_chips)
        if not (lo <= chips <= hi):
            raise HTTPException(
                status_code=400,
                detail=f"chips {chips} out of raise range [{lo}, {hi}]",
            )

    session.action_log.append(
        {"gate": int(gate_idx), "chips": int(chips)}
    )
    try:
        _rebuild_env()
    except Exception:
        session.action_log.pop()
        raise
    return {"state": _state_dict()}


@app.post("/undo")
def undo() -> dict[str, Any]:
    if not session.action_log:
        raise HTTPException(status_code=400, detail="nothing to undo")
    session.action_log.pop()
    _rebuild_env()
    return {"state": _state_dict()}


@app.post("/reset")
def reset() -> dict[str, Any]:
    _new_session_defaults()
    _rebuild_env()
    return {"state": _state_dict()}


@app.post("/config")
def config(req: ConfigRequest) -> dict[str, Any]:
    cfg = session.game_config
    bb = int(req.bb_chips) if req.bb_chips is not None else cfg.bb
    ante = int(req.ante_chips) if req.ante_chips is not None else cfg.ante
    if req.starting_stacks is not None:
        behinds = tuple(int(s) for s in req.starting_stacks)
        if len(behinds) != session.num_seats:
            raise HTTPException(
                status_code=400,
                detail=f"starting_stacks length {len(behinds)} != num_seats {session.num_seats}",
            )
        # `req.starting_stacks` carries "current chips behind" per
        # seat; convert to engine starting_stack via total_commit at
        # the current replay end. Hand-start collapses to behind +
        # ante (engine deducts the ante on hand init).
        commits = _current_total_commit(session.num_seats, ante)
        engine_stacks = tuple(b + commits[i] for i, b in enumerate(behinds))
        new_cfg = GameConfig(
            num_seats=session.num_seats,
            starting_stack=engine_stacks[0],
            ante=ante,
            bb=bb,
            starting_stacks=engine_stacks,
        )
    else:
        new_cfg = GameConfig(
            num_seats=session.num_seats,
            starting_stack=cfg.starting_stack,
            ante=ante,
            bb=bb,
        )
    session.game_config = new_cfg
    if req.dollars_per_bb is not None:
        session.dollars_per_bb = float(req.dollars_per_bb)
    # Stack-only edits preserve action_log; bb/ante changes invalidate
    # prior actions (chip math depends on the unit).
    bb_or_ante_changed = (
        (req.bb_chips is not None and int(req.bb_chips) != cfg.bb)
        or (req.ante_chips is not None and int(req.ante_chips) != cfg.ante)
    )
    if bb_or_ante_changed:
        _clear_hand_state_keep_cards()
    _rebuild_env()
    return {"state": _state_dict()}


# Initialize env at module load so /state works on first request.
_rebuild_env()


# --- OCR runner -------------------------------------------------------------

class OcrStartRequest(BaseModel):
    window_match: str = Field(..., min_length=1)
    poll_ms: int = Field(default=200, ge=50, le=5000)


class OcrSimpleRequest(BaseModel):
    enabled: bool


class OcrRescanRequest(BaseModel):
    target: Literal["hole", "board"]


_RESCAN_GROUPS: dict[str, tuple[str, ...]] = {
    "hole": ("hero_hole",),
    "board": ("flop_a", "flop_b", "turn_cards", "river_cards"),
}


class OcrRunner:
    """WGC-fed OCR session that mutates `session` once per polling tick.

    Frame *acquisition* happens on a free-threaded
    Windows.Graphics.Capture session against the picked window's HWND.
    The WGC callback runs on the binding's worker thread and publishes
    the latest BGR ndarray into ``self.latest_frame`` under
    ``self._frame_lock``. An asyncio task (``_loop``) wakes every
    ``poll_ms`` and runs ``_tick`` on a worker thread (extract +
    reconstruct + ``_rebuild_env`` is CPU-bound and the engine is not
    asyncio-aware).

    Why WGC and not Chrome's getDisplayMedia: ClubGG sets
    ``SetWindowDisplayAffinity`` on its tables. Chrome's per-window
    capture goes through GDI BitBlt / DXGI Desktop Duplication and
    respects WDA, so the captured frame shows whatever is behind the
    table. WGC reads from the DWM compositor surface and bypasses WDA
    in the same way OBS's "Windows 10 (1903 and up)" source does.
    """

    def __init__(self) -> None:
        self.running: bool = False
        self.poll_ms: int = 200
        self.window_match: str | None = None
        self.window_title: str | None = None
        self.candidates: list[str] = []
        # HWND of the captured window, persisted so `_loop` can poll its
        # liveness each tick. None when not running.
        self._hwnd: int | None = None
        # Why the runner last stopped: None for manual/never, "window_closed"
        # when it auto-stopped because the captured window was destroyed. The
        # frontend uses this to reset the picker only on auto-off.
        self.stopped_reason: str | None = None
        self.last_error: str | None = None
        self.last_tick_at: float | None = None
        self.frames_seen: int = 0
        self.events_applied: int = 0
        # Most recent BGR frame published by the WGC callback. Read by
        # `_tick` under `_frame_lock`.
        self.latest_frame: Any = None
        self._frame_lock: threading.Lock = threading.Lock()
        # Serializes the background `_tick` (run via `asyncio.to_thread`)
        # against handlers that mutate the same session state (cards,
        # locks, `session.env`). Acquire in the event loop before
        # dispatching the threadpool work.
        self._tick_lock: asyncio.Lock = asyncio.Lock()
        self._capture_control: Any = None
        self._reconstructor: Any = None
        self.task: asyncio.Task | None = None
        # Per-crop OCR read cache (stack/commit/pot). Lets `_tick` skip the
        # Tesseract subprocess for any chip ROI whose pixels are unchanged
        # since the last tick — most ticks then do ~0 reads. Cleared on start().
        self._ocr_read_cache: dict = {}

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "poll_ms": self.poll_ms,
            "window_match": self.window_match,
            "window_title": self.window_title,
            "candidates": list(self.candidates),
            "last_tick_at": self.last_tick_at,
            "last_error": self.last_error,
            "stopped_reason": self.stopped_reason,
            "frames_seen": self.frames_seen,
            "events_applied": self.events_applied,
        }

    async def start(self, window_match: str, poll_ms: int) -> None:
        from plo5bp.ocr import live as ocr_live
        from plo5bp.ocr.events import EventReconstructor

        if self.running:
            return

        self.window_match = window_match
        self.poll_ms = int(poll_ms)
        self.last_error = None
        self.stopped_reason = None
        self.frames_seen = 0
        self.events_applied = 0
        self.candidates = []
        self.window_title = None
        self._hwnd = None
        self.latest_frame = None
        self._ocr_read_cache = {}

        try:
            wm = await asyncio.to_thread(ocr_live.find_window, window_match)
        except ocr_live.NoWindowError as e:
            try:
                self.candidates = await asyncio.to_thread(ocr_live.list_window_titles)
            except Exception:
                self.candidates = []
            self.last_error = str(e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ocr_live.MultipleWindowsError as e:
            self.candidates = list(e.candidates)
            self.last_error = str(e)
            raise HTTPException(status_code=400, detail=str(e)) from e

        self.window_title = wm.title
        self._hwnd = int(wm.hwnd)

        def _on_frame(bgr: Any) -> None:
            with self._frame_lock:
                self.latest_frame = bgr

        try:
            self._capture_control = await asyncio.to_thread(
                ocr_live.start_wgc_capture, int(wm.hwnd), _on_frame
            )
        except Exception as e:
            self.last_error = f"WGC start failed: {type(e).__name__}: {e}"
            logger.exception("WGC start failed for hwnd=%s", wm.hwnd)
            raise HTTPException(status_code=500, detail=self.last_error) from e

        # ClubGG always renders 6 physical seat positions; the ROI table is
        # tied to those absolute screen coords. Empty seats are handled by
        # the multi-signal in-hand detector. Force the session to 6 seats
        # so OCR output shape matches the engine state regardless of how
        # many seats were configured for non-OCR study.
        if session.num_seats != 6:
            cfg = session.game_config
            session.game_config = GameConfig(
                num_seats=6,
                starting_stack=cfg.starting_stack,
                ante=cfg.ante,
                bb=cfg.bb,
            )
            session.num_seats = 6
            if session.button_seat >= 6:
                session.button_seat = 0
            session.hero_seat = 0
            _clear_hand_state_keep_cards()
            _rebuild_env()

        self._reconstructor = EventReconstructor(num_seats=session.num_seats)
        self.running = True
        self.task = asyncio.create_task(self._loop())

    async def _release_capture(self) -> None:
        """Stop the WGC capture session if one is live (idempotent).

        Shared by the manual `stop()` path and the auto-off
        `_handle_window_closed()` path; the `cc is None` guard makes a second
        call a no-op if a manual Off races a window close.
        """
        cc = self._capture_control
        self._capture_control = None
        if cc is not None:
            try:
                await asyncio.to_thread(cc.stop)
            except Exception as e:
                logger.warning("WGC stop raised: %s", e)

    async def stop(self) -> None:
        was_running = self.running
        self.running = False
        self.stopped_reason = None  # manual stop
        task = self.task
        self.task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        await self._release_capture()
        self._hwnd = None
        if was_running:
            self.window_match = None
            self.window_title = None

    async def _handle_window_closed(self) -> None:
        """Auto-stop path: the captured window was destroyed.

        Runs *inside* `_loop`, so it must NOT cancel its own task — it tears
        down the capture and lets `_loop` return normally. The frontend reads
        `stopped_reason == "window_closed"` (via `status()`) to reset the
        picker only on this path, not on a manual Off.
        """
        self.running = False
        self.stopped_reason = "window_closed"
        self.task = None
        await self._release_capture()
        self._hwnd = None
        self.window_match = None
        self.window_title = None

    async def _loop(self) -> None:
        from plo5bp.ocr import live as ocr_live

        try:
            while self.running:
                await asyncio.sleep(self.poll_ms / 1000.0)
                if not self.running:
                    return
                # Auto-stop if the captured window has been destroyed. Both
                # signals are sub-microsecond, so run them directly on the
                # asyncio side (no to_thread) before any per-tick work.
                # IsWindow is authoritative (stays True on minimize); the
                # WGC is_finished() is a defensive secondary signal.
                cc = self._capture_control
                cc_finished = False
                if cc is not None:
                    try:
                        cc_finished = bool(cc.is_finished())
                    except Exception:
                        cc_finished = False
                if cc_finished or not ocr_live.is_window_alive(self._hwnd):
                    await self._handle_window_closed()
                    return
                try:
                    async with self._tick_lock:
                        await asyncio.to_thread(self._tick)
                except BaseException as e:
                    # PyO3 panics inherit from BaseException; surface
                    # them via last_error rather than killing the loop.
                    self.last_error = f"{type(e).__name__}: {e}"
                    logger.exception("ocr tick failed")
        except asyncio.CancelledError:
            pass

    def _tick(self) -> None:
        from plo5bp.ocr.events import (
            HeroHoleRevealed,
            OcrWarning,
            SeatAction,
            StreetReveal,
        )
        from plo5bp.ocr.extract import extract_frame_state

        with self._frame_lock:
            img = self.latest_frame
        # WGC may not have delivered the first frame yet on the very
        # first tick after start; skip silently and try again next poll.
        if img is None:
            return

        fs = extract_frame_state(
            img, num_seats=session.num_seats, cache=self._ocr_read_cache
        )
        self.frames_seen += 1

        # Mirror directly-observable card / button state from the frame
        # before the reconstructor runs — event callbacks can assume
        # the session card spec already reflects the latest reveal.
        _mirror_observable_state(fs)

        if not session.simple_ocr_mode:
            engine_view = _engine_view_from_session()
            events = self._reconstructor.step(fs, engine_view)

            for ev in events:
                if isinstance(ev, HeroHoleRevealed):
                    # Already mirrored via _mirror_observable_state.
                    self.events_applied += 1
                elif isinstance(ev, StreetReveal):
                    # Already mirrored.
                    self.events_applied += 1
                elif isinstance(ev, SeatAction):
                    session.action_log.append(_seat_action_to_log_entry(ev))
                    if ev.gate == "fold":
                        # Sticky participant mask: once the reconstructor
                        # confirms a fold, retire that seat for the rest of
                        # the hand. Subsequent per-frame misses (banner /
                        # occlusion) can't un-fold them, and the UI's
                        # sitting_out display stays accurate.
                        session.folded_this_hand = frozenset(
                            session.folded_this_hand | {int(ev.seat)}
                        )
                        if session.hand_in_hand_mask:
                            all_seats = frozenset(range(session.num_seats))
                            session.sitting_out_seats = (
                                (all_seats - session.hand_in_hand_mask)
                                | session.folded_this_hand
                            )
                    self.events_applied += 1
                elif isinstance(ev, OcrWarning):
                    logger.warning("ocr: %s", ev.message)

            street_reveals = [ev for ev in events if isinstance(ev, StreetReveal)]
            if street_reveals:
                _reconcile_missed_folds_on_street_reveal(fs)
                target_street = max(
                    2 if ev.street == "turn" else 3 for ev in street_reveals
                )
                _reconcile_missed_checks_on_street_reveal(target_street)

        try:
            _rebuild_env()
            self.last_error = None
        except HTTPException as e:
            self.last_error = f"rebuild failed: {e.detail}"
            logger.warning("ocr rebuild failed: %s", e.detail)
        except Exception as e:
            self.last_error = f"rebuild failed: {type(e).__name__}: {e}"
            logger.exception("ocr rebuild failed")

        self.last_tick_at = time.time()


def _card_idx(c: Any) -> int | None:
    """FrameState Card → engine card index (rank * 4 + suit)."""
    if c is None:
        return None
    return int(c.rank) * 4 + int(c.suit)


_STABILITY_TICKS_REQUIRED = 2

# Mid-hand lock thresholds. After `_LOCK_AFTER_TICKS` ticks since
# the last `_begin_new_hand`, the snapshot debouncer (and the
# hero-hole rotation gate) require `_STABILITY_TICKS_REQUIRED_LOCKED`
# stable ticks before triggering another hand-start. Real new
# hands persist for many seconds and trivially clear this bar; a
# 1-2-tick chip-settle / banner OCR flicker doesn't.
_LOCK_AFTER_TICKS = 3
_STABILITY_TICKS_REQUIRED_LOCKED = 6
# Consecutive identical button reads required to trust a mid-hand button MOVE
# as a new-hand trigger. Decoupled from the 6-tick snapshot above: a button
# move persists all hand, so 3 reads reject a 1-2 frame glitch while keeping
# new-hand latency low and immune to sitting_out churn.
_BUTTON_STABLE_TICKS_LOCKED = 3

# Minimum plausible stack_chips read (cents) for an in-hand seat's
# anchor OCR. Live ClubGG frames captured during the chip-settle
# animation occasionally OCR the stack label as 0 or a tiny number
# because the blue bet-banner or chip oval covers the digits.
# `_begin_new_hand` refuses to seed from such reads — it keeps the
# existing seeded value instead — and `_mirror_observable_state`
# refuses to commit an anchor where any in-hand seat looks glitched,
# extending the debounce by one tick until a clean frame lands.
_MIN_PLAUSIBLE_STACK_CENTS = 100  # $1 — every in-hand seat has at least this


def _anchor_fs_stacks_plausible(fs: Any) -> bool:
    """Return True iff every non-folded seat has a plausible stack read.

    A glitched anchor is one where a bet-banner or chip-settle
    animation is covering a seat's stack label at capture time, so
    the OCR text read comes back as None or a near-zero value. Using
    such an anchor to seed ``cfg.starting_stacks`` causes the engine
    to wipe the seat's stack to zero after ante posting, which then
    renders in the UI as a phantom pre-bet "all-in".
    """
    for s in fs.seats:
        if s.folded:
            continue
        if s.stack_chips is None or int(s.stack_chips) < _MIN_PLAUSIBLE_STACK_CENTS:
            return False
    return True


def _mirror_observable_state(fs: Any) -> None:
    """Sync every directly-observed field from a FrameState into `session`.

    Cards, button, participant mask, and observed stack/pot numbers are
    all overwritten on every tick. The debounced button + participant
    commit is gated by a 2-tick stability check so one glitched frame
    can't trigger a false hand-start. When the debounced state advances
    (new button, new participant mask, or a hero-hole reshuffle), we
    call :func:`_begin_new_hand` to re-seed `cfg.starting_stacks` from
    the current OCR numbers and rebaseline the reconstructor.
    """
    for i, c in enumerate(fs.hero_hole):
        _ocr_apply_card_slot("hero_hole", i, _card_idx(c), debounce=True)
    for i, c in enumerate(fs.board_a[:3]):
        _ocr_apply_card_slot("flop_a", i, _card_idx(c), debounce=True)
    for i, c in enumerate(fs.board_b[:3]):
        _ocr_apply_card_slot("flop_b", i, _card_idx(c), debounce=True)
    _ocr_apply_card_slot("turn_cards", 0, _card_idx(fs.board_a[3]), debounce=True)
    _ocr_apply_card_slot("turn_cards", 1, _card_idx(fs.board_b[3]), debounce=True)
    _ocr_apply_card_slot("river_cards", 0, _card_idx(fs.board_a[4]), debounce=True)
    _ocr_apply_card_slot("river_cards", 1, _card_idx(fs.board_b[4]), debounce=True)

    session.observed_stacks = tuple(s.stack_chips for s in fs.seats)
    session.observed_pot = fs.pot_total_chips

    # Tick the mid-hand lock counter. Only counts while a hand is
    # active (mask non-empty); during bootstrap (empty mask) we
    # stay in the fast 2-tick path so initial hand-start isn't
    # delayed.
    if session.hand_in_hand_mask:
        session._ticks_since_hand_start += 1
    is_locked = (
        bool(session.hand_in_hand_mask)
        and session._ticks_since_hand_start >= _LOCK_AFTER_TICKS
    )
    threshold = (
        _STABILITY_TICKS_REQUIRED_LOCKED if is_locked
        else _STABILITY_TICKS_REQUIRED
    )

    observed_sitting_out = frozenset(
        i for i, s in enumerate(fs.seats) if i != session.hero_seat and s.folded
    )
    observed_button = (
        int(fs.button_seat) if fs.button_seat is not None else session.button_seat
    )

    # Button-ONLY stability, decoupled from the (button, sitting_out) snapshot
    # below. The button never moves mid-hand, so a stable button MOVE is a
    # strong new-hand signal that must not wait on sitting_out churn.
    if observed_button == session._last_observed_button:
        session._button_stable_ticks += 1
    else:
        session._last_observed_button = observed_button
        session._button_stable_ticks = 1

    hero_hole_indices: tuple[int, ...] | None = None
    if all(c is not None for c in fs.hero_hole):
        hero_hole_indices = tuple(int(_card_idx(c)) for c in fs.hero_hole)

    # Debounce the button + participant snapshot over 2 consecutive
    # ticks so a glitched frame can't trigger a false hand-start. The
    # anchor frame captures the first tick of a new snapshot and is
    # used downstream for stack seeding / reconstructor rebaseline.
    snapshot = (observed_button, observed_sitting_out)
    if (
        snapshot == (session._pending_button, session._pending_sitting_out)
    ):
        session._pending_stable_ticks += 1
        # Opportunistic anchor upgrade: if the stored anchor has
        # glitched stack reads but this tick's fs is clean, swap in
        # the cleaner anchor. Keeps the debounce counter intact so
        # we don't reset progress just because the original anchor
        # was captured during the chip-settle animation.
        if (
            session._pending_anchor_fs is not None
            and not _anchor_fs_stacks_plausible(session._pending_anchor_fs)
            and _anchor_fs_stacks_plausible(fs)
        ):
            session._pending_anchor_fs = fs
    else:
        session._pending_button = observed_button
        session._pending_sitting_out = observed_sitting_out
        session._pending_stable_ticks = 1
        session._pending_anchor_fs = fs

    committed_ready = session._pending_stable_ticks >= threshold
    # Delay the commit one more tick if the anchor looks glitched.
    # The opportunistic upgrade above will replace it as soon as a
    # clean tick lands.
    if (
        committed_ready
        and session._pending_anchor_fs is not None
        and not _anchor_fs_stacks_plausible(session._pending_anchor_fs)
    ):
        logger.warning(
            "hand-start delayed: anchor_fs has glitched stack reads"
        )
        committed_ready = False
    # A button MOVE (off its prior seat), confirmed by a short run of identical
    # reads, fires a new hand — gated on the button alone, not the 6-tick
    # snapshot, so sitting_out flicker during the deal no longer delays it.
    # Still require the seeding anchor to have plausible stacks (the same guard
    # `committed_ready` applies for `first_commit`).
    button_threshold = (
        _BUTTON_STABLE_TICKS_LOCKED if is_locked else _STABILITY_TICKS_REQUIRED
    )
    button_changed = (
        observed_button != int(session.button_seat)
        and session._button_stable_ticks >= button_threshold
        and _anchor_fs_stacks_plausible(session._pending_anchor_fs or fs)
    )

    # Rewind-proof fallback: hero hole re-appears with cards that differ
    # from the snapshot we recorded last hand-start. Pre-lock (bootstrap
    # or first few ticks of a hand) we trust the read on a single tick,
    # since hero_hole OCR uses static per-card ROIs and is normally
    # stable. Once locked we require the same rotated indices to persist
    # for `threshold` consecutive ticks — a one-frame OCR glitch during
    # a banner / chip-settle animation must not wipe the hand.
    hero_hole_rotated_now = (
        hero_hole_indices is not None
        and session.last_hero_hole is not None
        and session.last_hero_hole != hero_hole_indices
        and set(session.last_hero_hole).isdisjoint(hero_hole_indices)
    )
    if not is_locked:
        hero_hole_rotated = hero_hole_rotated_now
        session._pending_hero_hole_rotation = None
        session._pending_hero_hole_rotation_ticks = 0
    else:
        if hero_hole_rotated_now and (
            hero_hole_indices == session._pending_hero_hole_rotation
        ):
            session._pending_hero_hole_rotation_ticks += 1
        elif hero_hole_rotated_now:
            session._pending_hero_hole_rotation = hero_hole_indices
            session._pending_hero_hole_rotation_ticks = 1
        else:
            session._pending_hero_hole_rotation = None
            session._pending_hero_hole_rotation_ticks = 0
        hero_hole_rotated = (
            session._pending_hero_hole_rotation_ticks >= threshold
        )

    # First-time commit: when no hand has been committed yet (empty
    # in-hand mask) and the debounce is ready, seed the hand. This is
    # how we bootstrap when OCR starts mid-hand — there's no button
    # change to trigger on since session.button_seat was at its default.
    first_commit = committed_ready and not session.hand_in_hand_mask

    # Hand-start triggers: real hand-boundary events (button rotation,
    # hero-hole rotation) plus the very first commit after OCR start.
    # A mid-hand change in `observed_sitting_out` (a fold, or a banner
    # flicker) is NOT a hand-start — the reconstructor emits FOLD
    # events for real folds, and the `hand_in_hand_mask` / banner
    # signals defend against OCR flicker.
    trigger_fired = button_changed or hero_hole_rotated or first_commit

    # OCR-confirm guard: a real hand boundary always moves the button
    # to a different seat. If OCR still reads the button on the seat
    # we recorded at the last hand-start, the trigger source is
    # spurious (seen with ClubGG anti-collusion's mid-hand hero-hole
    # reveal tripping `hero_hole_rotated` against a stale baseline).
    # `first_commit` is exempt — bootstrap from an empty mask predates
    # any meaningful `session.button_seat`. We only check that the
    # button has moved off its prior seat, not that it landed on the
    # next clockwise seat: with <6 active players the button can skip.
    if trigger_fired and not first_commit:
        if (
            fs.button_seat is not None
            and int(fs.button_seat) == int(session.button_seat)
        ):
            trigger_fired = False

    if os.environ.get("PLO5BP_OCR_DEBUG_HANDSTART"):
        logger.info(
            "ocr.handstart: raw_btn=%s obs_btn=%s sess_btn=%s btn_ticks=%d "
            "locked=%s snap_ticks=%d | btn_chg=%s hole_rot=%s first=%s -> %s",
            fs.button_seat, observed_button, session.button_seat,
            session._button_stable_ticks, is_locked,
            session._pending_stable_ticks,
            button_changed, hero_hole_rotated, first_commit,
            "FIRED" if trigger_fired else "-",
        )

    if trigger_fired:
        anchor_fs = session._pending_anchor_fs or fs
        _begin_new_hand(
            anchor_fs,
            button_seat=observed_button,
            hero_hole_indices=hero_hole_indices,
        )
        session._pending_anchor_fs = None

    # Mid-hand mask expansion. Backstop for the case where the anchor
    # frame fired before a participant's cards-back rendered — that
    # seat reads `folded=True` at anchor (so they're missing from
    # `hand_in_hand_mask`) but `folded=False` continuously thereafter.
    # We add them after a 2-tick stability window so single-frame OCR
    # flickers can't trigger a false addition. Strictly additive: the
    # mask never shrinks here. Late-rebuy players are safe because
    # they read `folded=True` (no cards/banner/commit/timer-bar);
    # already-folded players are excluded via `folded_this_hand`.
    if session.hand_in_hand_mask:
        candidate_additions = frozenset(
            i for i, s in enumerate(fs.seats)
            if (i != session.hero_seat
                and not s.folded
                and i not in session.hand_in_hand_mask
                and i not in session.folded_this_hand)
        )
        if candidate_additions and candidate_additions == session._pending_mask_additions:
            session._pending_mask_additions_ticks += 1
        else:
            session._pending_mask_additions = candidate_additions
            session._pending_mask_additions_ticks = 1 if candidate_additions else 0
        if session._pending_mask_additions_ticks >= _STABILITY_TICKS_REQUIRED:
            session.hand_in_hand_mask = frozenset(
                session.hand_in_hand_mask | candidate_additions
            )
            session._pending_mask_additions = frozenset()
            session._pending_mask_additions_ticks = 0

    # Refresh the live sitting-out set every tick from the sticky mask
    # plus any folds the reconstructor has emitted. This makes a
    # banner-flicker or a transient `has_cards_back` miss invisible to
    # downstream consumers: once a seat is in the anchor mask it stays
    # in the hand until the reconstructor says otherwise. Before any
    # commit, leave `sitting_out_seats` as-is (default frozenset()) so
    # the stability gate alone decides when to trust OCR participants.
    if session.hand_in_hand_mask:
        all_seats = frozenset(range(session.num_seats))
        session.sitting_out_seats = (
            (all_seats - session.hand_in_hand_mask) | session.folded_this_hand
        )


def _begin_new_hand(
    fs: Any,
    *,
    button_seat: int,
    hero_hole_indices: tuple[int, ...] | None,
) -> None:
    """Seed session state for a newly-observed hand.

    Pulls ``cfg.starting_stacks`` from the anchor frame's OCR-cent reads
    (converted to engine chips + ante), locks the hand-start participant
    mask from the anchor frame (seats whose `folded=False` at hand-start
    are the ones dealt into this hand), rebaselines the reconstructor so
    diff-based action inference starts from a clean slate, and records
    the hero-hole snapshot so a future rewind can distinguish "same hand
    again" from "new hand".
    """
    _new_session_defaults()
    session.button_seat = int(button_seat)

    # Lock the participant mask from the anchor frame. Once a seat is
    # in this mask it stays in the hand until the reconstructor emits a
    # FOLD event (tracked in `folded_this_hand`). Banner flickers and
    # transient `has_cards_back` misses can't remove them.
    n = session.num_seats
    in_hand_mask = frozenset(
        i for i in range(n) if i < len(fs.seats) and not fs.seats[i].folded
    )
    session.hand_in_hand_mask = in_hand_mask
    session.folded_this_hand = frozenset()
    all_seats = frozenset(range(n))
    session.sitting_out_seats = all_seats - in_hand_mask

    cfg = session.game_config
    existing = list(cfg.resolved_stacks)
    merged = list(existing)
    for i in range(n):
        cents = fs.seats[i].stack_chips if i < len(fs.seats) else None
        # A glitched anchor OCR read — stack label covered by the blue
        # bet-banner or chip-settle animation — occasionally returns 0
        # or an absurdly small cent value. Seeding merged[i] from such a
        # read can make the engine's ante posting wipe the stack to 0
        # and flip the seat to all-in at hand-start, before any action
        # events have been applied. Skip the read and keep the prior
        # seeded value when it looks implausibly small. `None` already
        # falls through via the untouched branch below.
        if cents is not None and int(cents) < _MIN_PLAUSIBLE_STACK_CENTS:
            continue
        if cents is None:
            continue
        merged[i] = _ocr_cents_to_engine_chips(int(cents)) + int(cfg.ante)
    if tuple(merged) != tuple(existing):
        session.game_config = GameConfig(
            num_seats=cfg.num_seats,
            starting_stack=cfg.starting_stack,
            ante=cfg.ante,
            bb=cfg.bb,
            starting_stacks=tuple(merged),
        )

    if hero_hole_indices is not None:
        session.last_hero_hole = hero_hole_indices

    if ocr_runner._reconstructor is not None:
        ocr_runner._reconstructor.rebaseline(fs)


def _engine_view_from_session() -> Any:
    """Build an EngineView snapshot from the current session env.

    Imported lazily so server import doesn't pull in ocr deps when the
    OCR extras aren't installed.
    """
    from plo5bp.ocr.events import EngineView

    env = session.env
    if env is None:
        _rebuild_env()
        env = session.env
    assert env is not None
    raw = dict(env._rs.observation_dict())
    n = session.num_seats
    actor_raw = raw.get("actor")
    actor = int(actor_raw) if actor_raw is not None else None
    cfg = session.game_config
    dpb = float(session.dollars_per_bb)
    chips_per_cent = (
        float(cfg.bb) / (100.0 * dpb) if dpb > 0 else 1.0
    )
    # Noise floor: 1 bb in cents. Any stack-delta smaller than this must
    # be OCR jitter — no legal bet is under 1 bb.
    min_bet_cents = int(round(float(cfg.bb) / chips_per_cent)) if chips_per_cent > 0 else 0
    return EngineView(
        num_seats=n,
        current_actor=actor,
        street=int(raw.get("street", 0)),
        awaiting_next_street=raw.get("awaiting_next_street"),
        button_seat=int(session.button_seat),
        committed_this_street=tuple(int(x) for x in raw["street_commit"][:n]),
        stacks=tuple(int(x) for x in raw["stacks"][:n]),
        folded=tuple(bool(x) for x in raw["folded"][:n]),
        all_in=tuple(bool(x) for x in raw["all_in"][:n]),
        bet_to_call=int(raw.get("bet_to_call", 0)),
        chips_per_cent=chips_per_cent,
        min_bet_cents=min_bet_cents,
        sitting_out=tuple(i in session.sitting_out_seats for i in range(n)),
    )


ocr_runner = OcrRunner()


@app.get("/ocr/windows")
def ocr_windows() -> dict[str, Any]:
    """List visible top-level window titles for the dropdown.

    The frontend calls this on Refresh and on first focus of the
    window-match input. Backend matching is deliberately kept simple
    (substring + exact-match override in ``find_window``); see
    ``plo5bp.ocr.live`` for the exact semantics.
    """
    try:
        from plo5bp.ocr import live as ocr_live
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"ocr deps missing: {e}") from e
    try:
        titles = ocr_live.list_window_titles()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"windows": list(titles)}


@app.post("/ocr/start")
async def ocr_start(req: OcrStartRequest) -> dict[str, Any]:
    await ocr_runner.start(req.window_match, req.poll_ms)
    return {"ok": True, "status": ocr_runner.status()}


@app.post("/ocr/stop")
async def ocr_stop() -> dict[str, Any]:
    await ocr_runner.stop()
    return {"ok": True, "status": ocr_runner.status()}


@app.post("/ocr/simple")
def ocr_simple(req: OcrSimpleRequest) -> dict[str, Any]:
    session.simple_ocr_mode = bool(req.enabled)
    return {"state": _state_dict()}


@app.post("/ocr/rescan")
async def ocr_rescan(req: OcrRescanRequest) -> dict[str, Any]:
    """Re-OCR a single card group from the current frame; preserve hand state.

    Used in simple OCR mode when a card group was misread (capture
    landed mid-reveal animation). Clears the per-slot lock for the
    target group, applies a fresh OCR read, re-latches filled slots,
    and rebuilds the engine. `action_log`, `button_seat`, participant
    mask, observed stacks/pot all survive untouched.
    """
    if not ocr_runner.running:
        raise HTTPException(status_code=400, detail="OCR not running")
    async with ocr_runner._tick_lock:
        with ocr_runner._frame_lock:
            img = ocr_runner.latest_frame
        if img is None:
            raise HTTPException(
                status_code=400,
                detail="no frame received yet; let one WGC frame land first",
            )

        from plo5bp.ocr.extract import extract_frame_state

        groups = _RESCAN_GROUPS[req.target]
        prev_spec = {g: list(getattr(session, g)) for g in groups}
        prev_locks = {g: list(session._card_slot_locked[g]) for g in groups}

        try:
            fs = extract_frame_state(img, num_seats=session.num_seats)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"rescan {req.target} failed: extract: {e}",
            ) from e

        for g in groups:
            session._card_slot_locked[g] = [False] * len(prev_locks[g])
            session._card_slot_pending[g] = [None] * len(prev_locks[g])

        if "hero_hole" in groups:
            for i, c in enumerate(fs.hero_hole):
                _ocr_apply_card_slot("hero_hole", i, _card_idx(c))
        if "flop_a" in groups:
            for i, c in enumerate(fs.board_a[:3]):
                _ocr_apply_card_slot("flop_a", i, _card_idx(c))
            for i, c in enumerate(fs.board_b[:3]):
                _ocr_apply_card_slot("flop_b", i, _card_idx(c))
            _ocr_apply_card_slot("turn_cards", 0, _card_idx(fs.board_a[3]))
            _ocr_apply_card_slot("turn_cards", 1, _card_idx(fs.board_b[3]))
            _ocr_apply_card_slot("river_cards", 0, _card_idx(fs.board_a[4]))
            _ocr_apply_card_slot("river_cards", 1, _card_idx(fs.board_b[4]))

        _lock_filled_card_slots()

        try:
            _rebuild_env()
        except Exception as e:
            for g in groups:
                setattr(session, g, prev_spec[g])
                session._card_slot_locked[g] = prev_locks[g]
            try:
                _rebuild_env()
            except Exception:
                logger.exception("rescan rollback rebuild also failed")
            detail = e.detail if isinstance(e, HTTPException) else str(e)
            raise HTTPException(
                status_code=400, detail=f"rescan {req.target} failed: {detail}"
            )
    return {"state": _state_dict()}


@app.get("/ocr/status")
def ocr_status() -> dict[str, Any]:
    return ocr_runner.status()


@app.post("/ocr/save_frame")
def ocr_save_frame() -> dict[str, Any]:
    """Debug: write the most recent WGC frame to disk.

    Source is `OcrRunner.latest_frame`, the BGR ndarray published by
    the WGC callback on the most recent capture. Returns 400 if WGC
    hasn't delivered a frame yet.
    """
    try:
        import cv2
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"ocr deps missing: {e}") from e

    with ocr_runner._frame_lock:
        img = ocr_runner.latest_frame
    if img is None:
        raise HTTPException(
            status_code=400,
            detail="no frame received yet; start OCR and let one WGC frame land first",
        )

    out_dir = Path(__file__).resolve().parents[3] / "screenrecords" / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_ms = int(time.time() * 1000)
    path = out_dir / f"debug_{ts_ms}.png"
    if not cv2.imwrite(str(path), img):
        raise HTTPException(status_code=500, detail=f"cv2.imwrite failed for {path}")
    h, w = img.shape[:2]
    return {
        "path": str(path),
        "frame_size": {"width": int(w), "height": int(h)},
    }


# --- Static frontend --------------------------------------------------------


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp


if STATIC_DIR.exists():
    app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "index.html",
            headers={"Cache-Control": "no-store, must-revalidate"},
        )
