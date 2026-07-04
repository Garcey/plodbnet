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
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from plo5bp.actions import (
    GATE_ACTIONS,
    GATE_CHECK_CALL,
    GATE_FOLD,
    GATE_NAMES,
    GATE_RAISE,
)
from plo5bp.config import GameConfig, VARIANT_NLH, VARIANT_PLO5
from plo5bp.encoding import encode_observation
from plo5bp.env import BombPotEnv
from plo5bp.network import (
    ActorCritic,
    ActorCriticV4,
    CentralCritic,
    build_actor_from_state_dict,
    obs_adapter,
)
from plo5bp.encoding_nlh import OBS_DIM_NLH
from plo5bp.sizing import (
    ANCHOR_COUNT,
    BRACKET_HALF,
    NLH_ANCHOR_SPEC,
    PLO_ANCHOR_SPEC,
    anchor_grid_np,
    anchor_grid_torch,
    sizing_from_info,
)

from plo5bp.ui.common import (
    AWAITING_NAMES,
    HISTORY_NAMES as _HISTORY_NAMES,
    POSITION_BY_SEAT_6,
    POSITION_BY_SEAT_SHORT,
    STREET_NAMES,
    anchor_label as _anchor_label,
    anchor_label_spec as _spec_anchor_label,
    position_name as _common_position_name,
)

logger = logging.getLogger("plo5bp.ui")

# Public build flag. When truthy, the live-capture subsystems (ClubGG OCR
# and PokerNow DOM ingest) are NOT exposed: their routes are unmounted below
# and the frontend hides the live controls. Trainer + Study are fully
# functional without them — every study route rebuilds from user input via
# _rebuild_env, with no dependency on a live feed.
PLO5BP_PUBLIC = os.environ.get("PLO5BP_PUBLIC", "").strip().lower() in (
    "1", "true", "yes", "on",
)

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


#: Per-format checkpoint resolution. The PLO5 arm keeps the historical
#: env var + stub path; NLH gets its own pair so `cp checkpoints/nlh1_X.pt
#: checkpoints/nlh_stub.pt` is the NLH promote flow.
_FORMAT_CKPTS = {
    VARIANT_PLO5: ("PLO5BP_CHECKPOINT", "checkpoints/stub.pt"),
    VARIANT_NLH: ("PLO5BP_CHECKPOINT_NLH", "checkpoints/nlh_stub.pt"),
}


def _format_ckpt_path(variant: str) -> Path:
    env_key, default = _FORMAT_CKPTS[variant]
    return Path(os.environ.get(env_key, default))


def _random_init_model(variant: str) -> ActorCritic:
    """Placeholder actor when no checkpoint exists for the format. The
    PLO fallback keeps the historical v1 128×2 shape; NLH needs a
    correctly-shaped v4 (995-dim obs, 12-anchor ladder) so the format is
    still explorable before the first promote — flagged un-loaded so the
    UI can badge the recommendations as untrained."""
    if variant == VARIANT_NLH:
        return ActorCriticV4(
            hidden_dim=128,
            obs_dim=OBS_DIM_NLH,
            num_layers=2,
            anchor_spec=NLH_ANCHOR_SPEC,
        )
    return ActorCritic(hidden_dim=128, num_layers=2)


def _load_model(variant: str = VARIANT_PLO5) -> tuple[ActorCritic, bool]:
    """Load the format's promoted checkpoint. Returns (model, loaded) —
    loaded=False means a random-init placeholder is being served."""
    device = _resolve_device()
    ckpt_path = _format_ckpt_path(variant)
    if not ckpt_path.exists():
        logger.warning(
            "checkpoint %s not found — using random-init model (%s)",
            ckpt_path, variant,
        )
        return _random_init_model(variant).to(device).eval(), False
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.warning("failed to load %s (%s) — using random init", ckpt_path, e)
        return _random_init_model(variant).to(device).eval(), False
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
    # v1-era vs 991/995 current) and anchor spec are sniffed from the
    # state dict. The bundled critic state (ckpt['critic']) is loaded
    # separately by _load_critic() for the trainer review's all-cards
    # "true EV".
    try:
        model = build_actor_from_state_dict(state_dict, hidden_dim, num_layers)
        loaded = True
    except Exception as e:
        logger.warning(
            "checkpoint %s is incompatible with current network (%s) — "
            "using random init", ckpt_path, e,
        )
        model = _random_init_model(variant)
        loaded = False
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info(
        "loaded checkpoint %s (%s, hidden_dim=%d, num_layers=%d, device=%s)",
        ckpt_path, type(model).__name__, hidden_dim, num_layers, device,
    )
    return model, loaded


def _load_critic(device: torch.device, variant: str = VARIANT_PLO5) -> CentralCritic | None:
    """Load the centralized critic bundled in the format's checkpoint
    (v2+ only; ckpt['critic'] + head_version>=2). Returns None for v1 /
    random-init / missing critic, in which case the trainer review shows
    only the actor's own (blind) value estimate."""
    ckpt_path = _format_ckpt_path(variant)
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


MODEL, MODEL_LOADED = _load_model(VARIANT_PLO5)
MODEL_DEVICE = next(MODEL.parameters()).device
MODEL_CRITIC = _load_critic(MODEL_DEVICE, VARIANT_PLO5)
# v1-era checkpoints (trained at OBS_DIM 959) get the exact downgrade
# projection; current-width models get identity.
OBS_ADAPT = obs_adapter(MODEL)

NLH_MODEL, NLH_MODEL_LOADED = _load_model(VARIANT_NLH)
NLH_CRITIC = _load_critic(MODEL_DEVICE, VARIANT_NLH)

#: Per-format serving registry. `label` is what the UI dropdown shows;
#: `loaded=False` means the format serves a random-init placeholder (no
#: checkpoint promoted yet) and the client badges recommendations as
#: untrained. Promote flows: PLO5 `cp checkpoints/<run>.pt
#: checkpoints/stub.pt`; NLH `cp checkpoints/nlh<N>_<u>.pt
#: checkpoints/nlh_stub.pt` — restart to pick up.
FORMATS: dict[str, dict[str, Any]] = {
    VARIANT_PLO5: {
        "label": "PLO5 Double Board Bomb Pot",
        "model": MODEL,
        "critic": MODEL_CRITIC,
        "adapter": OBS_ADAPT,
        "loaded": MODEL_LOADED,
    },
    VARIANT_NLH: {
        "label": "NLH 5/10 ($5 ante)",
        "model": NLH_MODEL,
        "critic": NLH_CRITIC,
        "adapter": obs_adapter(NLH_MODEL),
        "loaded": NLH_MODEL_LOADED,
    },
}


def _fmt() -> dict[str, Any]:
    """The active format's serving entry (model/critic/adapter/loaded)."""
    return FORMATS[session.variant]


#: Optional per-request format gate, installed by the public build:
#: callable(format_id) -> True when the CURRENT user may not select the
#: format (rendered greyed-out "coming soon!" in the dropdown; POST
#: /format returns 403). None (the local build) = everything unlocked.
_FORMAT_GATE: Any = None


def set_format_gate(fn: Any) -> None:
    global _FORMAT_GATE
    _FORMAT_GATE = fn


def _format_locked(fmt_id: str) -> bool:
    if _FORMAT_GATE is None:
        return False
    try:
        return bool(_FORMAT_GATE(fmt_id))
    except Exception:
        logger.exception("format gate failed")
        # Fail closed for non-default formats; never lock the default.
        return fmt_id != VARIANT_PLO5


# --- Session state ----------------------------------------------------------

class Session:
    env: BombPotEnv | None = None
    game_config: GameConfig = GameConfig(starting_stack=400000)
    dollars_per_bb: float = 2.0

    # Active game format. Switching (POST /format) swaps the game config
    # to the format default, resets per-hand state, and re-shapes the
    # card spec (see _CARD_SPEC_BY_VARIANT). The served model/critic pair
    # follows via the FORMATS registry.
    variant: str = VARIANT_PLO5

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


# The study session. Locally there is exactly one (module-global semantics,
# OCR runner included). In the public build, `plo5bp.ui.public` installs a
# resolver that returns the signed-in user's own Session — every existing
# `session.x` read/write below transparently lands on the per-user object.
# The resolver returning None (or nothing installed) falls back to the
# default single session, which keeps the local build byte-identical.
_DEFAULT_SESSION = Session()
_SESSION_RESOLVER: Any = None


def set_session_resolver(fn) -> None:
    global _SESSION_RESOLVER
    _SESSION_RESOLVER = fn


def _current_session() -> Session:
    if _SESSION_RESOLVER is not None:
        s = _SESSION_RESOLVER()
        if s is not None:
            return s
    return _DEFAULT_SESSION


class _SessionProxy:
    __slots__ = ()

    def __getattr__(self, name: str):
        return getattr(_current_session(), name)

    def __setattr__(self, name: str, value) -> None:
        setattr(_current_session(), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(_current_session(), name)


session: Any = _SessionProxy()

# Consecutive identical auto-OCR reads required before a card slot commits and
# locks. ~600ms at 200ms poll / 300ms at 100ms. Raise if misreads still slip
# through; lower if the commit feels sluggish.
_CARD_STABLE_TICKS = 3


#: Per-format card-slot shapes. PLO5 double-board: 5-card hole, two
#: flops, dual turn/river. NLH single-board: 2-card hole, one flop,
#: single turn/river card, no board B.
_CARD_SPEC_BY_VARIANT: dict[str, tuple[tuple[str, int], ...]] = {
    VARIANT_PLO5: (
        ("hero_hole", 5),
        ("flop_a", 3),
        ("flop_b", 3),
        ("turn_cards", 2),
        ("river_cards", 2),
    ),
    VARIANT_NLH: (
        ("hero_hole", 2),
        ("flop_a", 3),
        ("flop_b", 0),
        ("turn_cards", 1),
        ("river_cards", 1),
    ),
}


def _card_spec_attrs() -> tuple[tuple[str, int], ...]:
    return _CARD_SPEC_BY_VARIANT[session.variant]


def _blank_card_pending() -> dict[str, list[tuple[int, int] | None]]:
    """Fresh per-slot debounce state (all slots empty)."""
    return {attr: [None] * n for attr, n in _card_spec_attrs()}


def _lock_filled_card_slots() -> None:
    """Latch the OCR-skip lock on any slot currently holding a card."""
    for attr, _ in _card_spec_attrs():
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
    """Reset all per-hand state. Keeps config + dollars_per_bb + format."""
    lens = dict(_card_spec_attrs())
    session.hero_hole = [None] * lens["hero_hole"]
    session.flop_a = [None] * lens["flop_a"]
    session.flop_b = [None] * lens["flop_b"]
    session.turn_cards = [None] * lens["turn_cards"]
    session.river_cards = [None] * lens["river_cards"]
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
        attr: [False] * n for attr, n in _card_spec_attrs()
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

    is_nlh = session.variant == VARIANT_NLH
    env = BombPotEnv(cfg)
    try:
        if is_nlh:
            # NLH study starts PREFLOP with a 2-card hole; blinds post in
            # the engine. No in-hand mask (live capture is PLO-only).
            env.reset_study_nlh(
                button=int(session.button_seat),
                hero_seat=int(session.hero_seat),
                hero_hole=list(padded["hero_hole"]),
            )
        else:
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
            if is_nlh:
                if awaiting == 1:
                    env.set_flop_nlh(
                        int(padded["flop_a"][0]),
                        int(padded["flop_a"][1]),
                        int(padded["flop_a"][2]),
                    )
                elif awaiting == 2:
                    env.set_turn_nlh(int(padded["turn"][0]))
                    reached_turn = True
                elif awaiting == 3:
                    env.set_river_nlh(int(padded["river"][0]))
                    reached_river = True
                else:
                    break
            elif awaiting == 2:
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
        gate = int(entry["gate"])
        chips = int(entry["chips"])
        # Clamp an over-effective-stack bet to the engine's max raise instead of
        # dropping it. A live site (PokerNow) lets a deep player bet more than a
        # short opponent can cover — e.g. $50 into a $15 effective stack. The
        # engine caps raises at the effective stack (`max_other_reachable`), so
        # the raw delta is rejected as illegal and the bet would be lost,
        # glitching the hand. Clamping treats the over-bet as the
        # all-in-equivalent the cap is meant to model ("any bet big enough to
        # cover everyone is the same"). Only triggers when the amount exceeds
        # what every opponent can call; deterministic, so each rebuild re-clamps
        # identically and the stored entry stays intact.
        if gate == int(GATE_RAISE) and chips > 0:
            try:
                mx = int(env._rs.max_raise_chips())
                if 0 < mx < chips:
                    chips = mx
            except Exception:
                pass
        try:
            env.step_hybrid(gate, chips)
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


class FormatRequest(BaseModel):
    format: str = Field(..., pattern=r"^(plo5_double_bomb|nlh_single)$")


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


def _reconcile_missed_checks_on_street_reveal(
    target_street: int, force: bool = False
) -> None:
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

    Safety (``force=False``, the OCR default): only fills when
    ``bet_to_call == 0`` and no seat has committed chips on the current
    street; positive evidence means a real bet was missed and we leave
    the engine stuck rather than silently mis-attribute.

    ``force=True`` (PokerNow): fill CHECK_CALL even when facing a bet.
    A PokerNow street reveal is authoritative proof the prior street's
    betting closed (PokerNow won't deal the next card otherwise), and the
    closing call's signal is unrecoverable — PokerNow sweeps the chips to
    the pot and deals the next card in the same instant the closing caller
    acts, so their committed amount is gone and their stack delta already
    landed in a (likely coalesced-away) earlier frame. Folds are
    reconciled first (``_reconcile_missed_folds_on_street_reveal`` reads
    the DOM fold flags), so every remaining in-hand seat that hasn't
    matched the bet must have CALLED — GATE_CHECK_CALL calls the engine's
    current ``bet_to_call`` for the right amount. A loop-count guard caps
    runaway in case the engine refuses to advance.
    """
    _rebuild_env()
    if session.env is None:
        return
    view = _engine_view_from_session()
    if os.environ.get("PLO5BP_OCR_DEBUG_TIMER"):
        logger.warning(
            "ocr.reconcile.checks: target=%d force=%s view.street=%d "
            "view.bet_to_call=%d committed=%s",
            target_street, force, int(view.street), int(view.bet_to_call),
            tuple(int(c) for c in view.committed_this_street),
        )
    if view.street >= target_street:
        return
    if not force and (
        view.bet_to_call > 0 or any(int(c) > 0 for c in view.committed_this_street)
    ):
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
    """Return None if hero can act; else 'hole'|'flop'|'turn'|'river'.

    Slot lists are variant-shaped (NLH: 2-card hole, no board B, single
    turn/river cards), so the same all()-checks cover both formats —
    NLH's empty flop_b list is vacuously complete. Preflop (street 0)
    needs only the hole.
    """
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
    fmt = _fmt()
    model = fmt["model"]
    obs_t = torch.from_numpy(fmt["adapter"](obs_np)).unsqueeze(0).to(MODEL_DEVICE)
    gm_t = torch.from_numpy(info.gate_mask).unsqueeze(0).to(MODEL_DEVICE)
    if getattr(model, "head_version", 1) >= 2:
        return _recommendation_v2(model, obs_t, gm_t, info)
    raise_max = int(info.max_raise_chips)
    raise_min = min(int(info.min_raise_chips), raise_max)
    bounds_t = torch.tensor(
        [[raise_min, raise_max]], dtype=torch.long, device=MODEL_DEVICE
    )
    with torch.no_grad():
        gate_logits, raise_params, value = model(obs_t, gm_t)
        gate_probs = F.softmax(gate_logits, dim=-1).squeeze(0).tolist()
        _act_out = model.act(
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
    model: Any, obs_t: torch.Tensor, gm_t: torch.Tensor, info: Any
) -> dict[str, Any]:
    """v2/v4 (anchor head) recommendation: argmax gate + argmax legal
    anchor with its refinement Beta, computed under the MODEL'S OWN
    anchor spec (PLO 11-anchor pot ladder or NLH 12-anchor overbet
    ladder with the ALL-IN atom). The server computes every anchor's
    chips — the client never recomputes sizing math."""
    spec = getattr(model, "anchor_spec", PLO_ANCHOR_SPEC)
    sizing = sizing_from_info(info)
    sizing_t = torch.from_numpy(sizing[None, :]).to(MODEL_DEVICE)
    with torch.no_grad():
        gate_logits, anchor_head_out, refine, value = model(obs_t, gm_t)
        gate_probs = F.softmax(gate_logits, dim=-1).squeeze(0).tolist()
        _act_out = model.act(obs_t, gm_t, sizing_t, deterministic=True)
        gate = int(_act_out.gate.item())
        chips = int(_act_out.chips.item())
        rec_anchor = int(_act_out.anchor.item())
        value_bb = float(value.squeeze(0).item())
        # Anchor histogram via the model's own (head-agnostic) anchor
        # distribution: flat masked softmax for v2, discretized-logistic for
        # v4. Avoids assuming the 2nd forward output is raw anchor logits.
        grid_t = anchor_grid_torch(sizing_t, spec)
        anchor_probs = (
            model._anchor_dist(anchor_head_out, grid_t)
            .probs.squeeze(0).float().cpu().numpy()
        )
        refine_np = refine.squeeze(0).float().cpu().numpy()  # (interior, 2)

    grid = anchor_grid_np(sizing[0], sizing[1], sizing[2], sizing[3], spec)
    is_allin_atom = lambda k: spec.allin_atom and k == spec.count - 1  # noqa: E731
    anchors = [
        {
            "k": int(k),
            # Pot fraction of the anchor; None for the ALL-IN atom (its
            # chips are max_raise, not a pot fraction).
            "frac": (
                None if is_allin_atom(k) else spec.fracs_pm[k] / 1000.0
            ),
            "label": _spec_anchor_label(spec, int(k)),
            "prob": round(float(anchor_probs[k]), 4),
            "chips": int(grid.chips[k]),
            "chips_bb": round(_chips_to_bb(int(grid.chips[k])), 4),
        }
        for k in range(spec.count)
        if bool(grid.legal[k])
    ]
    refine_block = None
    if bool(grid.refine_ok[rec_anchor]) and rec_anchor < len(spec.fracs_pm):
        alpha, beta = refine_np[rec_anchor - 1]
        refine_block = {
            "alpha": round(float(alpha), 4),
            "beta": round(float(beta), 4),
            "frac_lo": spec.bracket_lo_pm[rec_anchor] / 1000.0,
            "frac_hi": spec.bracket_hi_pm[rec_anchor] / 1000.0,
        }

    chips_out = chips if gate == GATE_RAISE else None
    chips_bb = round(_chips_to_bb(chips_out), 4) if chips_out is not None else None
    gate_slug = (
        "fold" if gate == GATE_FOLD else
        "check_call" if gate == GATE_CHECK_CALL else
        "raise"
    )
    return {
        "head_version": model.head_version,
        "pot_ref_chips": int(sizing[2]) + 2 * int(sizing[3]),
        "gate": gate_slug,
        "gate_name": GATE_NAMES[gate],
        "chips": chips_out,
        "chips_bb": chips_bb,
        "value_bb": round(value_bb, 4),
        "gate_distribution": [round(p, 4) for p in gate_probs],
        "anchors": anchors,
        "rec_anchor": rec_anchor,
        "refine": refine_block,
        "anchor_count": int(spec.count),
        "model_loaded": bool(_fmt()["loaded"]),
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
        "format": session.variant,
        "format_label": _fmt()["label"],
        "format_model_loaded": bool(_fmt()["loaded"]),
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
        # Live-capture toggle state; omitted in the public build so the JSON
        # payload carries no trace of the live subsystems.
        **({} if PLO5BP_PUBLIC else {"simple_ocr_mode": bool(session.simple_ocr_mode)}),
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

trainer_router = create_trainer_router(
    MODEL, MODEL_DEVICE, critic=MODEL_CRITIC, formats=FORMATS
)
app.include_router(trainer_router)


@app.get("/state")
def state() -> dict[str, Any]:
    if session.env is None:
        _rebuild_env()
    return {"state": _state_dict()}


@app.post("/cards")
def cards(req: CardsRequest) -> dict[str, Any]:
    lens = dict(_card_spec_attrs())
    session.hero_hole = _validate_card_list(
        req.hero_hole, lens["hero_hole"], "hero_hole"
    )
    session.flop_a = _validate_card_list(req.flop_a, lens["flop_a"], "flop_a")
    session.flop_b = _validate_card_list(req.flop_b, lens["flop_b"], "flop_b")
    session.turn_cards = _validate_card_list(req.turn, lens["turn_cards"], "turn")
    session.river_cards = _validate_card_list(
        req.river, lens["river_cards"], "river"
    )
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
            sb=cfg.sb,
            variant=cfg.variant,
        )
    elif new_num_seats != cfg.num_seats:
        session.game_config = GameConfig(
            num_seats=new_num_seats,
            starting_stack=cfg.starting_stack,
            ante=cfg.ante,
            bb=cfg.bb,
            sb=cfg.sb,
            variant=cfg.variant,
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


@app.get("/formats")
def formats() -> dict[str, Any]:
    """Formats the server can serve, for the UI dropdown. `model_loaded`
    False = a random-init placeholder answers (no checkpoint promoted).
    `locked` True = greyed out "coming soon!" for this user (public
    build gates non-default formats to admins while they train)."""
    return {
        "formats": [
            {
                "id": vid,
                "label": f["label"],
                "model_loaded": bool(f["loaded"]),
                "locked": _format_locked(vid),
                # Betting cap class: pot-limit formats cap raises at pot
                # (the client's b100 preset is "pot" and nothing larger
                # exists); no-limit formats allow overbets + all-in.
                "pot_limit": vid != VARIANT_NLH,
            }
            for vid, f in FORMATS.items()
        ],
        "active": session.variant,
    }


@app.post("/format")
def set_format(req: FormatRequest) -> dict[str, Any]:
    """Switch the study session's game format. Resets per-hand state and
    swaps the game config to the format default (PLO5: 6-max 200bb bomb
    pot; NLH: 6-max 100bb 5/10 with a $5/player ante). The trainer's
    format follows via its own setter so both tabs stay on one game."""
    if _format_locked(req.format):
        raise HTTPException(
            status_code=403,
            detail="This format isn't available on your account yet — coming soon!",
        )
    if req.format != session.variant:
        session.variant = req.format
        if req.format == VARIANT_NLH:
            session.game_config = GameConfig.nlh_default()
            session.dollars_per_bb = 10.0
        else:
            session.game_config = GameConfig(starting_stack=400000)
            session.dollars_per_bb = 2.0
        session.num_seats = session.game_config.num_seats
        session.button_seat = 0
        session.hero_seat = 0
        _new_session_defaults()
        try:
            trainer_router.set_format(req.format)
        except Exception:
            logger.exception("trainer format sync failed")
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
        # NLH keeps sb = bb/2 (the 5/10 structure scales with the unit).
        sb = bb // 2 if cfg.variant == VARIANT_NLH else cfg.sb
        new_cfg = GameConfig(
            num_seats=session.num_seats,
            starting_stack=engine_stacks[0],
            ante=ante,
            bb=bb,
            starting_stacks=engine_stacks,
            sb=sb,
            variant=cfg.variant,
        )
    else:
        sb = bb // 2 if cfg.variant == VARIANT_NLH else cfg.sb
        new_cfg = GameConfig(
            num_seats=session.num_seats,
            starting_stack=cfg.starting_stack,
            ante=ante,
            bb=bb,
            sb=sb,
            variant=cfg.variant,
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
        _set_active_reconstructor(self._reconstructor)
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

    # Rebaseline whichever live source is currently driving the session
    # (ClubGG OCR or PokerNow ingest). Both runners register their
    # reconstructor as the active one when they start.
    recon = _active_reconstructor()
    if recon is not None:
        recon.rebaseline(fs)


def _engine_view_from_session(exact_folds: bool = False) -> Any:
    """Build an EngineView snapshot from the current session env.

    ``exact_folds`` is set by the PokerNow path (the DOM exposes an exact
    ``fold`` class) so the walk never *infers* a fold from a swept closing
    call. Imported lazily so server import doesn't pull in ocr deps when the
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
        exact_folds=exact_folds,
    )


ocr_runner = OcrRunner()


# --- Active live-source reconstructor ---------------------------------------
# Both live sources (ClubGG WGC OCR and PokerNow DOM ingest) feed the same
# session through an EventReconstructor. `_begin_new_hand` rebaselines
# whichever one is currently driving; each runner registers its reconstructor
# here when it starts so the shared hand-start path stays source-agnostic.
_LIVE_RECONSTRUCTOR: Any = None


def _active_reconstructor() -> Any:
    # Prefer an explicitly-registered source reconstructor; fall back to the
    # OCR runner's own (the historical default, and what tests that wire
    # `ocr_runner._reconstructor` directly rely on).
    if _LIVE_RECONSTRUCTOR is not None:
        return _LIVE_RECONSTRUCTOR
    return ocr_runner._reconstructor


def _set_active_reconstructor(recon: Any) -> None:
    global _LIVE_RECONSTRUCTOR
    _LIVE_RECONSTRUCTOR = recon


# --- PokerNow DOM ingest ----------------------------------------------------

def _count_prefix(cards: tuple[Any, ...]) -> int:
    """Length of the leading run of non-None cards (board fill level)."""
    n = 0
    for c in cards:
        if c is None:
            break
        n += 1
    return n


def _pokernow_set_seat_count(n: int) -> None:
    """Reconfigure the session for an ``n``-seat PokerNow table.

    PokerNow tables vary in size hand-to-hand; unlike ClubGG (fixed 6),
    the engine seat count must track the players actually dealt in. Hero
    stays engine seat 0 (the mapper rotates the table so hero is first).
    """
    cfg = session.game_config
    session.game_config = GameConfig(
        num_seats=n,
        starting_stack=cfg.starting_stack,
        ante=cfg.ante,
        bb=cfg.bb,
    )
    session.num_seats = n
    session.hero_seat = 0
    if session.button_seat >= n:
        session.button_seat = 0
    _clear_hand_state_keep_cards()
    session.hand_in_hand_mask = frozenset()
    session.folded_this_hand = frozenset()
    session.sitting_out_seats = frozenset()
    session.last_hero_hole = None


def _pokernow_mirror_cards(fs: Any) -> None:
    """Mirror hero hole + both boards from a PokerNow FrameState.

    PokerNow card reads are exact (DOM text), so we commit immediately
    (``debounce=False``) — no multi-tick stability gate is needed. The
    per-slot lock still lets the user override a slot via ``/cards``.
    Villain cards stay hidden (PokerNow only reveals them at showdown),
    so only hero + community cards are mirrored, exactly like the OCR path.
    """
    for i, c in enumerate(fs.hero_hole):
        _ocr_apply_card_slot("hero_hole", i, _card_idx(c), debounce=False)
    for i, c in enumerate(fs.board_a[:3]):
        _ocr_apply_card_slot("flop_a", i, _card_idx(c), debounce=False)
    for i, c in enumerate(fs.board_b[:3]):
        _ocr_apply_card_slot("flop_b", i, _card_idx(c), debounce=False)
    _ocr_apply_card_slot("turn_cards", 0, _card_idx(fs.board_a[3]), debounce=False)
    _ocr_apply_card_slot("turn_cards", 1, _card_idx(fs.board_b[3]), debounce=False)
    _ocr_apply_card_slot("river_cards", 0, _card_idx(fs.board_a[4]), debounce=False)
    _ocr_apply_card_slot("river_cards", 1, _card_idx(fs.board_b[4]), debounce=False)


class PokerNowRunner:
    """Receives normalized DOM snapshots from the PokerNow userscript and
    drives the same session pipeline the OCR runner uses.

    Unlike ``OcrRunner`` there is no capture loop — the browser-side
    Tampermonkey collector pushes a ``pokernow.v1`` payload over the
    ``/pokernow/ingest`` websocket on every table change, and each payload
    is processed by ``handle_payload`` (run on a worker thread because
    ``_rebuild_env`` is synchronous CPU work).

    Because PokerNow data is exact, the reconstructor's OCR-disambiguation
    fallbacks never fire; the per-seat committed amount and explicit actor
    come straight from the DOM. The one PokerNow-specific wrinkle vs ClubGG
    is hand-start: bomb pots post antes (which PokerNow renders as a per-seat
    bet) before the flop, so we wait until the flop is on the felt to fire
    ``_begin_new_hand`` — that captures post-ante stacks and a clean
    zero-commit baseline, matching what ClubGG's debouncer lands on.
    """

    # A POST-based collector looks "connected" while snapshots keep arriving;
    # the userscript heartbeats every ~2s so an idle table stays green.
    _STALE_SECONDS = 6.0
    # Seconds the locked game can be silent before another game may take over.
    # Longer than the heartbeat so a brief gap mid-hand never hands the session
    # to a second open tab; short enough that intentionally switching tables
    # recovers within a few seconds.
    _GAME_SWITCH_STALE = 8.0

    def __init__(self) -> None:
        self._ws_connected: bool = False
        self.frames_seen: int = 0
        self.events_applied: int = 0
        self.last_error: str | None = None
        self.last_tick_at: float | None = None
        self.last_post_at: float | None = None
        self.bomb_pot: bool = False
        self.variant: str | None = None
        self.table_seats: int = 0
        self._reconstructor: Any = None
        self._last_payload_key: str | None = None
        # Game-lock: the bridge processes frames from exactly ONE PokerNow game
        # at a time. A second open tab (another table) posts a different
        # `gameId`; those frames are ignored so they can't interleave into the
        # live hand. Switches only after the locked game goes quiet.
        self.active_game_id: str | None = None
        self._active_game_at: float | None = None
        self.ignored_frames: int = 0
        self._tick_lock: asyncio.Lock = asyncio.Lock()

    def accept_game(self, game_id: Any) -> bool:
        """Return True if a frame from ``game_id`` should be processed.

        Untagged frames (older userscript) are always accepted. A tagged
        frame is accepted when it matches the locked game, no game is locked
        yet, or the locked game has been silent past ``_GAME_SWITCH_STALE``.
        """
        if game_id is None:
            return True
        now = time.time()
        if (
            self.active_game_id is None
            or game_id == self.active_game_id
            or self._active_game_at is None
            or (now - self._active_game_at) > self._GAME_SWITCH_STALE
        ):
            if game_id != self.active_game_id:
                logger.info("pokernow: locking onto game %s", game_id)
                # New game takes over: drop the prior hand's state cleanly.
                self._reconstructor = None
            self.active_game_id = game_id
            self._active_game_at = now
            return True
        self.ignored_frames += 1
        return False

    @property
    def connected(self) -> bool:
        if self._ws_connected:
            return True
        return (
            self.last_post_at is not None
            and (time.time() - self.last_post_at) < self._STALE_SECONDS
        )

    def status(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "frames_seen": self.frames_seen,
            "events_applied": self.events_applied,
            "last_error": self.last_error,
            "last_tick_at": self.last_tick_at,
            "bomb_pot": self.bomb_pot,
            "variant": self.variant,
            "table_seats": self.table_seats,
            "active_game_id": self.active_game_id,
            "ignored_frames": self.ignored_frames,
        }

    def note_post(self) -> None:
        """Record that a snapshot/heartbeat arrived (HTTP POST path)."""
        self.last_post_at = time.time()

    def on_connect(self) -> None:
        self._ws_connected = True
        self.frames_seen = 0
        self.events_applied = 0
        self.last_error = None
        self._reconstructor = None
        self._last_payload_key = None

    def on_disconnect(self) -> None:
        self._ws_connected = False

    def is_duplicate(self, payload: dict) -> bool:
        """Cheap server-side dedupe so heartbeats don't re-run the pipeline.

        The userscript already dedupes by content, but heartbeats resend the
        last snapshot to keep the connection fresh; skip reprocessing those.
        """
        import json as _json

        try:
            key = _json.dumps(payload, sort_keys=True, default=str)
        except Exception:
            return False
        if key == self._last_payload_key:
            return True
        self._last_payload_key = key
        return False

    def handle_payload(self, payload: dict) -> None:
        from plo5bp.ocr.events import (
            OcrWarning,
            SeatAction,
            StreetReveal,
        )
        from plo5bp.ocr.events import EventReconstructor
        from plo5bp.ocr.pokernow import map_payload

        pf = map_payload(payload)
        fs = pf.frame
        n = pf.num_seats
        # Between hands PokerNow can briefly show <2 in-hand seats; skip
        # those frames rather than collapsing the engine seat count.
        if n < 2:
            return

        self.frames_seen += 1
        self.bomb_pot = pf.bomb_pot
        self.variant = pf.variant
        self.table_seats = n

        _pn_reconf = n != session.num_seats
        _pn_raw_seats = [
            (
                s.get("seat"),
                s.get("stackDollars"),
                s.get("betDollars") if s.get("betDollars") is not None else s.get("betText"),
                "fold" if s.get("folded") else ("allin" if s.get("allIn") else ""),
                len(s.get("cards") or []),
            )
            for s in (payload.get("seats") or [])
        ]

        # Track the table size; reconfigure + rebuild the reconstructor
        # when it changes (player joined/left between hands).
        if n != session.num_seats:
            _pokernow_set_seat_count(n)
            self._reconstructor = None
        if self._reconstructor is None or self._reconstructor.num_seats != session.num_seats:
            self._reconstructor = EventReconstructor(num_seats=session.num_seats)
            _set_active_reconstructor(self._reconstructor)

        _pokernow_mirror_cards(fs)
        session.observed_stacks = tuple(s.stack_chips for s in fs.seats)
        session.observed_pot = fs.pot_total_chips

        flop_present = (
            _count_prefix(fs.board_a) >= 3 and _count_prefix(fs.board_b) >= 3
        )
        hero_indices: tuple[int, ...] | None = None
        if all(c is not None for c in fs.hero_hole):
            hero_indices = tuple(int(_card_idx(c)) for c in fs.hero_hole)

        button_changed = (
            fs.button_seat is not None
            and int(fs.button_seat) != int(session.button_seat)
        )
        # Hero's hole cards are the most reliable new-hand signal: dealt at
        # hand start, visible throughout, exact, and re-dealt every hand.
        # Compare as a SET (order-insensitive — PokerNow may re-sort hero's
        # cards mid-hand) and fire on ANY change. We deliberately do NOT
        # require the new hand to be card-disjoint from the last (that's a
        # ClubGG OCR-noise guard; consecutive PokerNow deals often share a
        # card, which would suppress the trigger). The dealer-button DOM can
        # lag the flop deal, so `button_changed` is only a secondary signal.
        hero_changed = hero_indices is not None and (
            session.last_hero_hole is None
            or set(hero_indices) != set(session.last_hero_hole)
        )
        first_commit = not session.hand_in_hand_mask
        new_hand = first_commit or hero_changed or button_changed

        # Defer the hand-start to the flop so the anchor frame carries
        # post-ante stacks + cleared street commits (see class docstring).
        # No stack-plausibility gate here (unlike the OCR path): PokerNow
        # reads are exact, and `_begin_new_hand` already skips per-seat
        # None/short stacks — gating the whole hand-start on it would wrongly
        # block a hand where a villain ante'd all-in (stack renders empty).
        if new_hand and flop_present:
            btn = (
                int(fs.button_seat)
                if fs.button_seat is not None
                else int(session.button_seat)
            )
            _begin_new_hand(fs, button_seat=btn, hero_hole_indices=hero_indices)
            # _begin_new_hand wiped the card spec via _new_session_defaults;
            # re-mirror this frame so the flop/hero cards survive the reset.
            _pokernow_mirror_cards(fs)

        _pn_dbg = os.environ.get("PLO5BP_PN_DEBUG")
        _street_before = None
        if _pn_dbg and session.env is not None:
            try:
                _street_before = int(dict(session.env._rs.observation_dict()).get("street", -1))
            except Exception:
                _street_before = -2

        # Action inference runs only once flop betting is live (the engine
        # posts antes itself; PokerNow's pre-flop ante bets are not actions).
        events: list = []
        street_reveals: list = []
        if flop_present and session.hand_in_hand_mask:
            engine_view = _engine_view_from_session(exact_folds=True)
            events = self._reconstructor.step(fs, engine_view)
            for ev in events:
                if isinstance(ev, SeatAction):
                    session.action_log.append(_seat_action_to_log_entry(ev))
                    if ev.gate == "fold":
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
                elif isinstance(ev, StreetReveal):
                    self.events_applied += 1
                elif isinstance(ev, OcrWarning):
                    logger.warning("pokernow: %s", ev.message)

            street_reveals = [ev for ev in events if isinstance(ev, StreetReveal)]
            if street_reveals:
                _reconcile_missed_folds_on_street_reveal(fs)
                target_street = max(
                    2 if ev.street == "turn" else 3 for ev in street_reveals
                )
                # force=True: a PokerNow reveal proves the prior street closed,
                # but the closing call's committed/stack signal is gone (chips
                # swept to pot + next card dealt in the same instant). Fill the
                # remaining calls to close the street rather than stalling.
                _reconcile_missed_checks_on_street_reveal(target_street, force=True)

        try:
            _rebuild_env()
            self.last_error = None
        except HTTPException as e:
            self.last_error = f"rebuild failed: {e.detail}"
            logger.warning("pokernow rebuild failed: %s", e.detail)
        except Exception as e:
            self.last_error = f"rebuild failed: {type(e).__name__}: {e}"
            logger.exception("pokernow rebuild failed")

        if _pn_dbg:
            try:
                # Read street/actor straight from the engine — do NOT call
                # _state_dict() here: it runs _compute_recommendation() (a model
                # forward pass), which on every ingest frame throttles the whole
                # bridge and forces the userscript to coalesce/drop actions.
                _raw = (
                    dict(session.env._rs.observation_dict())
                    if session.env is not None else {}
                )
                _dbg_street = STREET_NAMES.get(int(_raw.get("street", -1)), _raw.get("street"))
                _dbg_actor = _raw.get("actor")
                ev_kinds = ",".join(type(e).__name__ for e in events) or "-"
                _gate_ch = {int(GATE_FOLD): "F", int(GATE_CHECK_CALL): "C", int(GATE_RAISE): "R"}
                alog_str = "".join(
                    _gate_ch.get(int(e["gate"]), "?") + str(int(e["chips"]))
                    for e in session.action_log
                ) or "-"
                seats_str = " ".join(
                    f"{s.seat}:stk={s.stack_chips}bet={s.committed_chips}{'A' if s.is_actor else ''}{'F' if s.folded else ''}"
                    for s in fs.seats
                )
                line = (
                    f"f{self.frames_seen} b1={_count_prefix(fs.board_a)} "
                    f"b2={_count_prefix(fs.board_b)} n={n} reconf={_pn_reconf} "
                    f"mask={sorted(session.hand_in_hand_mask)} "
                    f"raw={_pn_raw_seats} actorDOM="
                    f"{[s.seat for s in fs.seats if s.is_actor]} "
                    f"new_hand={new_hand}(hc={hero_changed},bc={button_changed},fc={first_commit}) "
                    f"flop_present={flop_present} street_before={_street_before} "
                    f"events=[{ev_kinds}] reveals={len(street_reveals)} "
                    f"alog={alog_str} seats=[{seats_str}] -> street={_dbg_street} "
                    f"actor={_dbg_actor} pot={session.observed_pot} err={self.last_error}\n"
                )
                with open(_pn_dbg, "a", encoding="utf-8") as fh:
                    fh.write(line)
            except Exception:
                pass

        self.last_tick_at = time.time()


pokernow_runner = PokerNowRunner()


@app.websocket("/pokernow/ingest")
async def pokernow_ingest(ws: WebSocket) -> None:
    """Receive ``pokernow.v1`` DOM snapshots from the Tampermonkey collector.

    One active collector at a time. Refuses the connection while the ClubGG
    OCR runner is live so the two sources can't fight over the session.
    """
    await ws.accept()
    if ocr_runner.running:
        await ws.close(code=1008, reason="ClubGG OCR is active")
        return
    pokernow_runner.on_connect()
    try:
        while True:
            payload = await ws.receive_json()
            # Game-lock first (before dedup) so the active game's heartbeats
            # keep the lock fresh even when they dedupe away.
            if not pokernow_runner.accept_game(payload.get("gameId")):
                continue
            pokernow_runner.note_post()
            if pokernow_runner.is_duplicate(payload):
                continue
            async with pokernow_runner._tick_lock:
                await asyncio.to_thread(pokernow_runner.handle_payload, payload)
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001 — surface, don't crash the socket loop
        pokernow_runner.last_error = f"{type(e).__name__}: {e}"
        logger.exception("pokernow ingest error")
    finally:
        pokernow_runner.on_disconnect()


@app.post("/pokernow/ingest")
async def pokernow_ingest_http(payload: dict) -> dict[str, Any]:
    """HTTP ingest for the Tampermonkey collector (``GM_xmlhttpRequest``).

    This is the primary transport: a userscript can't open a page-context
    websocket to ``127.0.0.1`` from an ``https`` PokerNow tab (Chrome's
    Private Network Access + mixed-content rules block it), but
    ``GM_xmlhttpRequest`` runs in the extension's privileged context and
    POSTs here freely. The websocket endpoint remains for non-userscript
    clients (e.g. a packaged extension).
    """
    if ocr_runner.running:
        raise HTTPException(status_code=409, detail="ClubGG OCR is active")
    # Game-lock first (before dedup) so the active game's heartbeats keep the
    # lock fresh even when they dedupe away. Foreign frames (a second open
    # PokerNow tab) are ignored so they can't interleave into the live hand.
    if not pokernow_runner.accept_game(payload.get("gameId")):
        return {"ok": True, "ignored": "other_game", "status": pokernow_runner.status()}
    pokernow_runner.note_post()
    if pokernow_runner.is_duplicate(payload):
        return {"ok": True, "deduped": True, "status": pokernow_runner.status()}
    async with pokernow_runner._tick_lock:
        await asyncio.to_thread(pokernow_runner.handle_payload, payload)
    return {"ok": True, "status": pokernow_runner.status()}


@app.get("/pokernow/status")
def pokernow_status() -> dict[str, Any]:
    return pokernow_runner.status()


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


# --- Public build: unmount live-capture routes ------------------------------
# In a public deployment the trained model is served but real-time table
# reading is not. The OCR/PokerNow handlers above stay defined; here we drop
# their routes so every /ocr/* and /pokernow/* path 404s. Trainer + Study are
# untouched (they never call these). Stripping post-registration avoids
# wrapping ~1000 lines of handlers in a conditional.
if PLO5BP_PUBLIC:
    _n_before = len(app.router.routes)
    app.router.routes[:] = [
        r
        for r in app.router.routes
        if not str(getattr(r, "path", "")).startswith(("/ocr", "/pokernow"))
    ]
    logger.info(
        "PUBLIC build: dropped %d live route(s); /ocr and /pokernow disabled",
        _n_before - len(app.router.routes),
    )


# --- Static frontend --------------------------------------------------------


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp


def _strip_wglive(text: str) -> str:
    """Drop every marker-bounded live-capture region (WGLIVE:START..END).

    The full local build keeps ClubGG-OCR / PokerNow client code and markup
    inside these markers; the public build serves the assets with the whole
    region removed — a public visitor's copy contains no trace (names,
    selectors, endpoints) of the live subsystems, inspectable or otherwise."""
    import re as _re

    return _re.sub(r"[^\n]*WGLIVE:START.*?WGLIVE:END[^\n]*\n?", "", text, flags=_re.S)


if STATIC_DIR.exists():
    if PLO5BP_PUBLIC:
        # Serve stripped app.js / style.css via explicit routes that shadow
        # the mount below (Starlette matches routes in registration order).
        # Computed once at import; a code deploy restarts the service anyway.
        _APP_JS_PUBLIC = _strip_wglive(
            (STATIC_DIR / "app.js").read_text(encoding="utf-8")
        )
        _STYLE_PUBLIC = _strip_wglive(
            (STATIC_DIR / "style.css").read_text(encoding="utf-8")
        )
        _NO_CACHE = {"Cache-Control": "no-store, must-revalidate"}

        @app.get("/static/app.js")
        def _public_app_js() -> Response:
            return Response(
                _APP_JS_PUBLIC, media_type="text/javascript", headers=_NO_CACHE
            )

        @app.get("/static/style.css")
        def _public_style_css() -> Response:
            return Response(
                _STYLE_PUBLIC, media_type="text/css", headers=_NO_CACHE
            )

    app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> HTMLResponse:
        # Inject the build mode so the frontend knows its mode before first
        # paint; in the public build also strip the WGLIVE markup regions.
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        if PLO5BP_PUBLIC:
            html = _strip_wglive(html)
        flag = "true" if PLO5BP_PUBLIC else "false"
        html = html.replace(
            "</head>", f"<script>window.PLO5BP_PUBLIC={flag};</script></head>", 1
        )
        return HTMLResponse(
            html, headers={"Cache-Control": "no-store, must-revalidate"}
        )

    @app.get("/terms")
    def terms_page() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "terms.html",
            headers={"Cache-Control": "no-store, must-revalidate"},
        )

    @app.get("/privacy")
    def privacy_page() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "privacy.html",
            headers={"Cache-Control": "no-store, must-revalidate"},
        )


# --- Public service layer (auth / billing / admin / per-user state) ----------
# Installed last so its middleware wraps every route above. Local build
# (flag unset) never imports plo5bp.ui.public.
if PLO5BP_PUBLIC:
    from plo5bp.ui import public as _public
    from plo5bp.ui.trainer import (
        TrainerSession as _TrainerSession,
        set_session_resolver as _set_trainer_resolver,
    )

    _public.install(
        app,
        study_session_factory=Session,
        set_study_resolver=set_session_resolver,
        trainer_session_factory=lambda stats_path: _TrainerSession(
            MODEL, MODEL_DEVICE, critic=MODEL_CRITIC, stats_path=stats_path
        ),
        set_trainer_resolver=_set_trainer_resolver,
        static_dir=STATIC_DIR,
        set_format_gate=set_format_gate,
    )
