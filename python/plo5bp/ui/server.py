"""Study-tool FastAPI backend (single-screen edition).

Card spec is client-authoritative and supports partial entry: any slot
can be null. The server pads nulls with the lowest-index unused deck
cards, runs `reset_study`, and replays the action log through
`step_hybrid`. Hero recommendations are gated on `hero_info_complete`
(all 5 hole cards placed); non-hero actors can act freely.

Run with: `uvicorn plo5bp.ui.server:app --port 8765`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import re
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Callable, Mapping

import anyio.to_thread
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as _StarletteHTTPException
from pydantic import BaseModel, Field

from plo5bp.actions import (
    GATE_CHECK_CALL,
    GATE_FOLD,
    GATE_NAMES,
    GATE_RAISE,
)
from plo5bp.config import GameConfig, VARIANT_NLH, VARIANT_PLO5

import plo5bp.encoding as _encoding  # module handle: OBS_SEMANTICS_REV is read late
from plo5bp.encoding import OBS_DIM_MINIMAL, encode_observation  # noqa: F401 (re-export)
from plo5bp.env import BombPotEnv
from plo5bp.network import ActorCritic, CentralCritic
from plo5bp.sizing import (
    NLH_ANCHOR_SPEC,
    PLO_ANCHOR_SPEC,
    anchor_grid_np,
    anchor_grid_torch,
    sizing_from_info,
)
from plo5bp.ui.common import FORMAT_EXPERIMENTAL  # the admin candidate slot

from plo5bp.ui import middleware as _mw
from plo5bp.ui.common import (
    AWAITING_NAMES,
    anchor_label_spec as _spec_anchor_label,
    ActionRequest as _CommonActionRequest,
    GATE_NAME_TO_IDX as _COMMON_GATE_NAME_TO_IDX,
    GATE_SLUGS as _COMMON_GATE_SLUGS,
    anchors_payload as _common_anchors_payload,
    default_game_config as _common_default_game_config,
    engine_variant as _common_engine_variant,
    table_state as _common_table_state,
    validate_card_list as _common_validate_card_list,
    effective_button as _effective_button,
    env_flag as _env_flag,
    format_defaults as _common_format_defaults,
    model_slots as _model_slots,
    position_name as _common_position_name,
)

logger = logging.getLogger("plo5bp.ui")


def _configure_logging() -> None:
    """(OPS-023) Make the app's own log lines reach the journal.

    uvicorn configures only its own loggers, so `plo5bp.*` INFO lines ("loaded
    checkpoint …", "evicted runtime …", "PUBLIC service installed …") were
    dropped and WARNINGs printed bare. When nothing else has configured
    logging (no root handler — pytest, a --log-config and embedding apps all
    install one), the `plo5bp` logger gets a timestamped handler at
    ``PLO5BP_LOG_LEVEL`` (default INFO). Runs before the models load, so the
    lines naming the served checkpoints are kept."""
    pkg = logging.getLogger("plo5bp")
    if logging.getLogger().handlers or pkg.handlers:
        return
    level = os.environ.get("PLO5BP_LOG_LEVEL", "INFO").strip().upper() or "INFO"
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    pkg.addHandler(handler)
    pkg.setLevel(level if level in logging.getLevelNamesMapping() else "INFO")


_configure_logging()

TERMINAL_NAMES = {0: "fold_out", 1: "run_out", 2: "showdown"}
TERMINAL_MESSAGES = {
    "fold_out": "All opponents folded — uncontested pot.",
    "run_out": "All remaining players are all-in; turn/river cards were not entered.",
    "showdown": "River action closed with multiple players — opponent cards unknown, no showdown evaluated.",
}

STATIC_DIR = Path(__file__).parent / "static"


# --- Serving models ---------------------------------------------------------
# Loading (one read per checkpoint, safe unpickling), the obs-rev bookkeeping,
# verification and live swaps live in `plo5bp.ui.models`. An app's served
# models are its site's `formats` (see `Site`); this module keeps the
# historical names — `MODEL`, `FORMATS`, `_load_model`, … — that the routes,
# the trainer, the home-games grader, the deploy check and the tests use.

from plo5bp.ui import models as _models  # noqa: E402

_resolve_device = _models.resolve_device
_random_init_model = _models.random_init_model
_OBS_REV_INFO = _models.OBS_REV_INFO
_process_obs_rev = _models.process_obs_rev
_note_checkpoint_obs_rev = _models.note_checkpoint_obs_rev
_obs_rev_entry = _models.obs_rev_entry
_critic_value_kwargs = _models.critic_value_kwargs
_CRITIC_VALUE_KWARGS = _models.CRITIC_VALUE_KWARGS

#: The formats a site serves, in dropdown order: PLO5, NLH and the admin
#: candidate slot.
_SERVED_FORMATS = (VARIANT_PLO5, VARIANT_NLH, FORMAT_EXPERIMENTAL)


def _format_ckpt_path(variant: str) -> Path | None:
    """The checkpoint a format serves: PLO5 `$PLO5BP_CHECKPOINT` /
    checkpoints/stub.pt, NLH `$PLO5BP_CHECKPOINT_NLH` / checkpoints/
    nlh_stub.pt, the admin candidate slot `$PLO5BP_CHECKPOINT_CANDIDATE`
    (None when unset)."""
    return _models.format_ckpt_path(variant)


def _load_model(variant: str = VARIANT_PLO5) -> tuple[ActorCritic, bool]:
    """Load the format's promoted checkpoint. Returns (model, loaded) —
    loaded=False means a random-init placeholder is being served."""
    return _models.load_model(variant, _format_ckpt_path(variant))


def _load_critic(device: torch.device, variant: str = VARIANT_PLO5) -> CentralCritic | None:
    """The centralized critic bundled in the format's checkpoint (v2+), or
    None — then the trainer review shows only the actor's blind value."""
    ckpt_path = _format_ckpt_path(variant)
    if ckpt_path is None or not ckpt_path.exists():
        return None
    try:
        ckpt = _models.read_checkpoint(ckpt_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("failed to load critic from %s (%s)", ckpt_path, e)
        return None
    return _models.critic_from_checkpoint(ckpt, device, variant, ckpt_path)


# --- The site: one app and everything it serves from (BE-007) ----------------
# `create_app()` builds the FastAPI app from the environment together with a
# `Site` that holds what the app serves from: its settings, its models, the
# NLH teacher, the trainer router, the study sessions, the format gate, the
# static files and — in the public build — the service layer's database and
# the home games. Importing this module builds one (uvicorn's
# `plo5bp.ui.server:app`, the deploy's pre-flight); a test builds more — another
# configuration, a fresh database — without re-importing any module.
#
# The process has ONE current site: the last one built, or the one `use_site`
# picked. The study routes serve from it and the module's historical names
# read it — `app`, `FORMATS`, `MODEL`, `MODEL_LOADED`, `GTO_HOST`,
# `trainer_router`, `PLO5BP_PUBLIC` … (`_SITE_NAMES`) — like `homegame.CTX` for
# the home games. Production builds exactly one.


@dataclasses.dataclass(frozen=True)
class SiteSettings:
    """What `create_app` reads from the environment — once, when it builds the
    app (`from_env`). The layers read their own settings the same way when an
    app installs them (`public.install`, `homegame.install`); the trainer and
    the checkpoint loader read PLO5BP_PUBLIC when they run, so `public` must
    agree with the environment (`create_app` checks)."""

    #: PLO5BP_PUBLIC — the public build: sign-in, per-user state, the service
    #: layer and the home games. The live-capture subsystems (ClubGG OCR and
    #: PokerNow ingest, `plo5bp.ui.live`), /ranges and the API docs are never
    #: imported or mounted, and the frontend hides the live controls; Trainer +
    #: Study are fully functional without them (every study route rebuilds from
    #: user input via _rebuild_env, with no dependency on a live feed).
    public: bool = False
    #: PLO5BP_GTO_CHECKPOINT — the NLH teacher Study and the Trainer share.
    gto_checkpoint: str | None = None
    #: PLO5BP_BASE_URL without its trailing slash ("": the request's own
    #: address) — the sitemap's links, and HSTS when the public build is https.
    base_url: str = ""
    #: PLO5BP_TORCH_THREADS — CPU threads per forward (None: the build decides).
    torch_threads: int | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "SiteSettings":
        env = os.environ if environ is None else environ
        threads = env.get("PLO5BP_TORCH_THREADS", "").strip()
        return cls(
            public=_env_flag("PLO5BP_PUBLIC", environ=env),
            gto_checkpoint=env.get("PLO5BP_GTO_CHECKPOINT", "").strip() or None,
            base_url=env.get("PLO5BP_BASE_URL", "").strip().rstrip("/"),
            torch_threads=max(1, int(threads)) if threads else None,
        )


class Site:
    """Everything one app serves from. `create_app` builds it; it is the app's
    ``app.state.site`` and, while it is the current site, what the module's
    names read."""

    def __init__(self, settings: SiteSettings) -> None:
        self.settings = settings
        self.app: FastAPI | None = None
        #: Per-format serving registry (`models.FormatEntry` dicts). `label` is
        #: what the UI dropdown shows; `loaded=False` means a random-init
        #: placeholder answers (no checkpoint promoted) and the client badges
        #: recommendations as untrained. Each checkpoint is read ONCE for its
        #: actor and critic (PERF-023). Promote without a restart from /admin
        #: (System → Promote) after copying the new file next to the served one
        #: as `<name>.new` (`model_admin`, a `models.ModelAdmin` over this dict).
        self.formats: dict[str, dict[str, Any]] = {}
        #: Where the models run (the PLO5 model's device at startup).
        self.device = torch.device("cpu")
        self.model_admin: _models.ModelAdmin | None = None
        #: The optional NLH GTO PolicyNet (Phase 2a): Study recommendations and
        #: the Trainer share this one host.
        self.gto_host: Any = None
        self.trainer_router: Any = None
        #: The study session. Locally there is exactly one (module-global
        #: semantics, live capture included). In the public build
        #: `plo5bp.ui.public` installs a resolver that returns the signed-in
        #: user's own Session — every `session.x` read/write lands on the
        #: per-user object — and this one is never served (SEC-011).
        self.default_session = Session()
        self.session_resolver: Callable[[], Session | None] | None = None
        #: The public build's per-request trainer resolver (the trainer module
        #: holds the current site's).
        self.trainer_resolver: Callable[[], Any] | None = None
        #: Optional per-request format gate, installed by the public build:
        #: callable(format_id) -> True when the CURRENT user may not select the
        #: format (rendered greyed-out "coming soon!" in the dropdown; POST
        #: /format returns 403). None (the local build) = everything unlocked.
        self.format_gate: Callable[[str], bool] | None = None
        self.static: SiteStaticFiles | None = None
        self.pages: _PageRenderer | None = None
        #: GET /health's worker checks (`middleware.register_health_check`).
        self.health_checks: dict[str, Any] = {}
        #: Public build: the service layer's database and per-user registry, and
        #: the home-games context (`homegame.HomeGames`).
        self.db: Any = None
        self.registry: Any = None
        self.homegames: Any = None
        self.started_at = time.time()
        self.closed = False

    @property
    def public(self) -> bool:
        return self.settings.public

    def set_session_resolver(self, fn: Callable[[], Session | None] | None) -> None:
        self.session_resolver = fn

    def set_trainer_resolver(self, fn: Callable[[], Any] | None) -> None:
        self.trainer_resolver = fn
        if _SITE is self:
            _trainer.set_session_resolver(fn)

    def set_format_gate(self, fn: Callable[[str], bool] | None) -> None:
        self.format_gate = fn

    def close(self) -> None:
        """Stop what this site runs in the background — the home games' clock,
        grader and live streams — and close its database. (Tests; a stopping
        server runs the app's shutdown hooks instead.) Idempotent."""
        if self.closed:
            return
        self.closed = True
        if self.homegames is not None:
            hg = sys.modules.get("plo5bp.ui.homegame")
            if hg is not None:
                previous = hg.use_context(self.homegames)
                try:
                    hg.shutdown()
                finally:
                    hg.use_context(previous)
        if self.db is not None:
            self.db.close()


#: The current site (see above). Set by `create_app` / `use_site`.
_SITE: Site | None = None
#: The site the public layer (`plo5bp.ui.public`, whose state is its module's)
#: currently serves: one at a time.
_PUBLIC_SITE: Site | None = None


def current_site() -> Site:
    """The site the study routes and the module's names serve from."""
    if _SITE is None:
        raise RuntimeError("no app has been built yet (plo5bp.ui.server.create_app)")
    return _SITE


def use_site(site: Site | FastAPI) -> Site | None:
    """Make `site` (or an app's) the current one; returns the one it replaces.

    The trainer's per-request session resolver, the /health worker checks and —
    for a public site — the home-games context follow it. A public site can be
    current only while the public layer still serves it (the layer's state is
    its module's: a later public `create_app` retired this one)."""
    if not isinstance(site, Site):
        site = site.state.site
    if site.closed:
        raise RuntimeError("that app was closed (Site.close)")
    if site.public and site is not _PUBLIC_SITE:
        raise RuntimeError(
            "this public app was retired by a later create_app(): its service layer "
            "now serves the newer app — build a new one"
        )
    return _activate(site)


def _activate(site: Site) -> Site | None:
    global _SITE
    old, _SITE = _SITE, site
    _trainer.set_session_resolver(site.trainer_resolver)
    _mw.use_health_checks(site.health_checks)
    if site.homegames is not None:
        sys.modules["plo5bp.ui.homegame"].use_context(site.homegames)
    return old


def _plo5_entry(site: Site) -> dict[str, Any]:
    return site.formats[VARIANT_PLO5]


#: The module's historical names, read from the CURRENT site (`_ServerModule`).
_SITE_NAMES: dict[str, Callable[[Site], Any]] = {
    # uvicorn's `plo5bp.ui.server:app`: the app built at import (the current site's).
    "app": lambda s: s.app,
    "FORMATS": lambda s: s.formats,
    "MODEL": lambda s: _plo5_entry(s)["model"],
    "MODEL_LOADED": lambda s: bool(_plo5_entry(s)["loaded"]),
    "MODEL_CRITIC": lambda s: _plo5_entry(s)["critic"],
    # v1-era checkpoints (trained at OBS_DIM 959) get the exact downgrade
    # projection; current-width models get identity.
    "OBS_ADAPT": lambda s: _plo5_entry(s)["adapter"],
    "NLH_MODEL": lambda s: s.formats[VARIANT_NLH]["model"],
    "NLH_MODEL_LOADED": lambda s: bool(s.formats[VARIANT_NLH]["loaded"]),
    "NLH_CRITIC": lambda s: s.formats[VARIANT_NLH]["critic"],
    "EXP_MODEL": lambda s: s.formats[FORMAT_EXPERIMENTAL]["model"],
    "EXP_MODEL_LOADED": lambda s: bool(s.formats[FORMAT_EXPERIMENTAL]["loaded"]),
    "EXP_CRITIC": lambda s: s.formats[FORMAT_EXPERIMENTAL]["critic"],
    "MODEL_DEVICE": lambda s: s.device,
    #: Admin reload / promote / rollback (OPS-027) — see `models.ModelAdmin`.
    "MODEL_ADMIN": lambda s: s.model_admin,
    "GTO_HOST": lambda s: s.gto_host,
    "_GTO_CKPT": lambda s: s.settings.gto_checkpoint,
    "trainer_router": lambda s: s.trainer_router,
    "PLO5BP_PUBLIC": lambda s: s.settings.public,
    "_DEFAULT_SESSION": lambda s: s.default_session,
    "_SESSION_RESOLVER": lambda s: s.session_resolver,
    "_FORMAT_GATE": lambda s: s.format_gate,
    "_STATIC_MOUNT": lambda s: s.static,
    "_PAGES": lambda s: s.pages,
    "_STARTED_AT": lambda s: s.started_at,
}
#: The names that may be assigned (a test's monkeypatch): they write the site.
_SITE_SETTABLE = {
    "GTO_HOST": "gto_host",
    "_SESSION_RESOLVER": "session_resolver",
    "_FORMAT_GATE": "format_gate",
}


class _ServerModule(types.ModuleType):
    """``server.MODEL``, ``server.app``, ``server.GTO_HOST`` … read (and the
    settable ones write) the CURRENT site's (BE-007)."""

    def __getattr__(self, name: str) -> Any:
        get = _SITE_NAMES.get(name)
        if get is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        site = self.__dict__.get("_SITE")
        if site is None:
            raise AttributeError(f"{__name__}.{name}: no app has been built yet (create_app)")
        return get(site)

    def __setattr__(self, name: str, value: Any) -> None:
        attr = _SITE_SETTABLE.get(name)
        if attr is not None:
            setattr(current_site(), attr, value)
        elif name in _SITE_NAMES:
            raise AttributeError(
                f"{__name__}.{name} is the current site's (read-only here) — see Site"
            )
        else:
            super().__setattr__(name, value)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(_SITE_NAMES))


def _on_model_swap(fmt: str, entry: dict[str, Any]) -> None:
    """`entry` now serves `fmt` on the current site. The model names (`MODEL`,
    `MODEL_CRITIC`, `NLH_MODEL` …) are read from `FORMATS` whenever they are
    used, so nothing else needs refreshing: trainer sessions re-bind at their
    next hand (the entry's `version`), the home-games grader at its next job."""
    current_site().formats[fmt] = entry


def _maybe_reload_experimental() -> None:
    """The candidate slot follows its file: when the configured candidate
    checkpoint changed on disk since it was loaded (a new file dropped in
    place), load it and swap it in. No-op when the slot is unset, the file
    is unchanged, or the new file does not load (the old one keeps serving)."""
    path = _format_ckpt_path(FORMAT_EXPERIMENTAL)
    if path is None:
        return
    try:
        st = path.stat()
    except OSError:
        return
    entry = current_site().formats.get(FORMAT_EXPERIMENTAL) or {}
    if (
        entry.get("loaded")
        and entry.get("_ckpt_path") == str(path)
        and entry.get("mtime") == float(st.st_mtime)
        and entry.get("size") == int(st.st_size)
    ):
        return
    new = _models.build_entry(FORMAT_EXPERIMENTAL, path)
    if not new["loaded"]:
        logger.warning("candidate checkpoint %s does not load — keeping the previous one", path)
        return
    _on_model_swap(FORMAT_EXPERIMENTAL, new)
    logger.info("candidate format now serving %s", path)


def _fmt() -> dict[str, Any]:
    """The active format's serving entry (model/critic/adapter/loaded)."""
    return current_site().formats[session.variant]


def set_format_gate(fn: Any) -> None:
    """Install the current site's per-request format gate (`Site.format_gate`)."""
    current_site().set_format_gate(fn)


def _format_locked(fmt_id: str) -> bool:
    gate = current_site().format_gate
    if gate is None:
        return False
    try:
        return bool(gate(fmt_id))
    except Exception:
        logger.exception("format gate failed")
        # Fail closed for non-default formats; never lock the default.
        return fmt_id != VARIANT_PLO5


# --- Session state ----------------------------------------------------------

class Session:
    env: BombPotEnv | None = None
    game_config: GameConfig
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
    # (Created per-instance in __init__ — mutable, must not be shared.)
    hero_hole: list[int | None]
    flop_a: list[int | None]
    flop_b: list[int | None]
    turn_cards: list[int | None]   # [board_a_turn, board_b_turn]
    river_cards: list[int | None]

    last_obs: np.ndarray | None = None
    last_info: Any = None

    # Action log contains only action entries: {"gate": int, "chips": int}
    # plus an optional "seat" (the seat the entry was attributed to when it
    # was recorded; absent on count-attributed reconcile entries). Replay
    # still applies each entry to the ENGINE's current actor — "seat" is
    # diagnostic only (see _build_env). Street transitions are inferred
    # during replay from engine state.
    # (Created per-instance in __init__ — mutable, must not be shared.)
    action_log: list[dict[str, Any]]

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

    # (log index, recorded seat, engine actor) triples already warned about
    # by the replay seat-attribution check, so a persistent mismatch logs
    # once per hand instead of once per tick. (review 2026-09-20 F15)
    _replay_mismatch_warned: frozenset[tuple[int, int, int]] = frozenset()

    # (FEAT-025) Admins: show the candidate model's answer next to the live
    # one (`POST /study/compare`).
    compare_candidate: bool = False

    # Snapshots captured at first-time street reveal during replay.
    # Compared against current card spec to flag "modified since reveal".
    snapshot_at_turn: dict[str, list[int | None]] | None = None
    snapshot_at_river: dict[str, list[int | None]] | None = None

    # Per-slot OCR write lock. Once a card slot has been filled (by
    # OCR or by the user via /cards), the lock for that slot latches
    # to True and OCR will skip it for the rest of the hand. Lets
    # the user override OCR misreads without the next tick clobbering
    # their edit. Reset to all-False by `_new_session_defaults`.
    _card_slot_locked: dict[str, list[bool]]

    # Live-capture tracker state (`plo5bp.ui.live.state.LiveState`: the OCR /
    # PokerNow debounce counters, pending card reads, observed stacks). The
    # local build's live capture creates it on first use; the public build
    # never does.
    live: Any = None

    def __init__(self) -> None:
        # (BE-002) Serializes this session's Study requests (see
        # `_study_route`): two quick clicks or two tabs used to interleave
        # validate-then-append and lose or double actions.
        self.lock = threading.RLock()
        # Per-instance mutable state. These MUST be created here, not as
        # class-level defaults: in the multi-user public build every signed-in
        # user gets their own Session(), and a shared class-level list/dict
        # would alias one user's action_log / cards / slot-locks into
        # another's session (a cross-user state leak). Immutable defaults
        # (None / int / str / tuple / frozenset) stay as class attributes
        # above — reassignment can't leak. Sizes match the PLO5 default
        # variant; _new_session_defaults re-sizes per variant on reset.
        self.game_config = _common_default_game_config(VARIANT_PLO5)
        self.hero_hole = [None] * 5
        self.flop_a = [None] * 3
        self.flop_b = [None] * 3
        self.turn_cards = [None, None]
        self.river_cards = [None, None]
        self.action_log = []
        self._card_slot_locked = {
            "hero_hole": [False] * 5,
            "flop_a": [False] * 3,
            "flop_b": [False] * 3,
            "turn_cards": [False, False],
            "river_cards": [False, False],
        }


# The study session (the current site's, see `Site.default_session`). Locally
# there is exactly one (module-global semantics, live capture included). In the
# public build, `plo5bp.ui.public` installs a resolver that returns the
# signed-in user's own Session — every existing `session.x` read/write below
# transparently lands on the per-user object. The resolver returning None (or
# nothing installed) falls back to the default single session, which keeps the
# local build byte-identical.


def set_session_resolver(fn) -> None:
    current_site().set_session_resolver(fn)


def _current_session() -> Session:
    site = current_site()
    if site.session_resolver is not None:
        s = site.session_resolver()
        if s is not None:
            return s
        if site.public:
            # (SEC-011) Mirror the trainer: the default session is ONE object
            # shared by every caller, so in the public build a request (or a
            # background task) with no signed-in user must never read or
            # drive it — it fails closed instead.
            raise HTTPException(status_code=401, detail="sign in required")
    return site.default_session


class _SessionProxy:
    __slots__ = ()

    def __getattr__(self, name: str):
        return getattr(_current_session(), name)

    def __setattr__(self, name: str, value) -> None:
        setattr(_current_session(), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(_current_session(), name)


session: Any = _SessionProxy()


# --- Local-build extension hooks ---------------------------------------------
# The live-capture package (`plo5bp.ui.live`, local build only) keeps its own
# per-session state (`Session.live`) and plugs into the study core here. The
# public build registers nothing, so these are no-ops there.

#: fn(kind) callbacks. kind "hand": `_new_session_defaults` just restored the
#: per-hand defaults (a new hand, /reset, /format); "user": a user-level reset
#: (/reset, /format) that must also forget what was learned across hands.
_SESSION_RESET_HOOKS: list[Any] = []
#: fn() -> dict callbacks whose keys are merged into `_state_dict()`.
_STATE_EXTRAS_HOOKS: list[Any] = []


def _run_session_reset_hooks(kind: str) -> None:
    # (each hook once, even if a second local app registered it again)
    for hook in dict.fromkeys(_SESSION_RESET_HOOKS):
        hook(kind)


def _state_extras() -> dict[str, Any]:
    extras: dict[str, Any] = {}
    for hook in dict.fromkeys(_STATE_EXTRAS_HOOKS):
        extras.update(hook())
    return extras


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


def _engine_variant(fmt_id: str | None = None) -> str:
    """Map a UI format id to the engine GameConfig.variant string (the
    registry's `engine_variant`, else the shared `common.engine_variant`)."""
    fid = session.variant if fmt_id is None else fmt_id
    entry = current_site().formats.get(fid)
    if entry is not None:
        return str(entry.get("engine_variant", fid))
    return _common_engine_variant(fid)


def _card_spec_attrs() -> tuple[tuple[str, int], ...]:
    return _CARD_SPEC_BY_VARIANT[_engine_variant()]


def _lock_filled_card_slots() -> None:
    """Latch the live-capture skip lock on any slot currently holding a card
    (a user edit via /cards wins over later live reads)."""
    for attr, _ in _card_spec_attrs():
        spec = getattr(session, attr)
        locks = session._card_slot_locked[attr]
        for i, c in enumerate(spec):
            if c is not None:
                locks[i] = True


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
    session._replay_mismatch_warned = frozenset()
    session._card_slot_locked = {
        attr: [False] * n for attr, n in _card_spec_attrs()
    }
    _run_session_reset_hooks("hand")


def _clear_hand_state_keep_cards() -> None:
    """Clear action log + snapshots but keep card spec intact."""
    session.action_log = []
    session.snapshot_at_turn = None
    session.snapshot_at_river = None
    session.last_obs = None
    session.last_info = None
    session._replay_mismatch_warned = frozenset()


def _current_total_commit(num_seats: int, ante: int) -> list[int]:
    """Per-seat total_commit at the current replay end, in engine chips.

    Used by /config and /seats to convert UI "current behind" inputs
    to engine starting_stack via `engine_starting = behind + commit`.
    Falls back to `[ante] * num_seats` when no env exists yet (cold
    start) or the seat-count just changed.
    """
    commits = _current_engine_seat_array("total_commit")
    if commits is not None and len(commits) == num_seats:
        return commits
    return [int(ante)] * num_seats


def _current_behind_stacks() -> list[int] | None:
    """Per-seat chips behind at the current replay end (None if no env)."""
    return _current_engine_seat_array("stacks")


def _current_engine_seat_array(key: str) -> list[int] | None:
    if session.env is None:
        # A live hand-start invalidates the env and a fresh per-user Session
        # has none yet; rebuild so callers see real commits (NLH blinds
        # included) instead of their ante-only fallback. Best-effort: a
        # session whose current state can't rebuild falls through.
        try:
            _rebuild_env()
        except Exception:  # noqa: BLE001 — callers fall back, but say why (OPS-028)
            logger.warning(
                "could not rebuild the study env to read %r — callers fall back to"
                " ante-only commits", key, exc_info=True,
            )
    if session.env is not None:
        try:
            raw = session.env._rs.observation_dict(skip_outcome_mc=True)
            return [int(x) for x in (raw.get(key) or [])]
        except Exception:  # noqa: BLE001 (OPS-028)
            logger.warning("reading %r from the study engine failed", key, exc_info=True)
    return None


# --- Card padding -----------------------------------------------------------

def _pad_cards(
    hero_hole: list[int | None],
    flop_a: list[int | None],
    flop_b: list[int | None],
    turn_cards: list[int | None],
    river_cards: list[int | None],
) -> dict[str, list[int]]:
    """Resolve a nullable card spec to a fully-padded 11+4 deck slice.

    User-entered cards go in first; nulls are filled by scanning 0..51
    and taking the lowest unused index. Raises HTTP 400 on duplicates.
    Pure — reads nothing from the session, so handlers can run it on a
    CANDIDATE spec before committing it.
    """
    user_cards: list[int] = []
    for spec in (hero_hole, flop_a, flop_b, turn_cards, river_cards):
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
        "hero_hole": pad(hero_hole),
        "flop_a": pad(flop_a),
        "flop_b": pad(flop_b),
        "turn": pad(turn_cards),
        "river": pad(river_cards),
    }


def _pad_all() -> dict[str, list[int]]:
    """`_pad_cards` over the session's current card spec."""
    return _pad_cards(
        session.hero_hole, session.flop_a, session.flop_b,
        session.turn_cards, session.river_cards,
    )


# --- Rebuild env (canonical path) -------------------------------------------
#
# (review 2026-09-20 H7) The rebuild is split into a PURE build step and a
# commit step so request handlers can validate-then-commit: `_build_env`
# constructs + replays an env from an explicit `_EnvSpec` without touching
# the session, and only `_commit_env_build` writes. `/cards`, `/seats`,
# `/config`, `/action` and `/undo` build their CANDIDATE spec first, so a
# rejected request (duplicate card, engine refusal, absurd stacks) leaves the
# session exactly as it was — previously the bad state was stored first and
# every later rebuild 400/500'd. `_rebuild_env()` remains the one canonical
# path for everything else: build from the session as-is, then commit.


@dataclasses.dataclass(frozen=True)
class _EnvSpec:
    """The replayable state an env is built from: a snapshot of the session,
    or a handler's candidate for it."""

    cfg: GameConfig
    is_nlh: bool
    num_seats: int
    button_seat: int
    hero_seat: int
    hand_in_hand_mask: frozenset[int]
    hero_hole: list[int | None]
    flop_a: list[int | None]
    flop_b: list[int | None]
    turn_cards: list[int | None]
    river_cards: list[int | None]
    action_log: list[dict[str, Any]]


@dataclasses.dataclass
class _EnvBuild:
    """Result of `_build_env`, handed to `_commit_env_build`."""

    env: BombPotEnv
    # Entries the engine accepted, in order (same dict objects as the spec's).
    kept_log: list[dict[str, Any]]
    # True when replay dropped at least one entry the engine rejected.
    dropped_entries: bool
    reached_turn: bool
    reached_river: bool
    # (log index, recorded seat, engine actor) where an entry's recorded
    # "seat" differed from the actor it was actually applied to.
    seat_mismatches: list[tuple[int, int, int]]


def _session_env_spec(**overrides: Any) -> _EnvSpec:
    """Snapshot the session's replayable state; ``overrides`` swap in a
    handler's candidate values."""
    spec = _EnvSpec(
        cfg=session.game_config,
        is_nlh=_engine_variant() == VARIANT_NLH,
        num_seats=int(session.num_seats),
        button_seat=int(session.button_seat),
        hero_seat=int(session.hero_seat),
        hand_in_hand_mask=frozenset(session.hand_in_hand_mask),
        hero_hole=list(session.hero_hole),
        flop_a=list(session.flop_a),
        flop_b=list(session.flop_b),
        turn_cards=list(session.turn_cards),
        river_cards=list(session.river_cards),
        action_log=list(session.action_log),
    )
    return dataclasses.replace(spec, **overrides) if overrides else spec


def _build_env(spec: _EnvSpec) -> _EnvBuild:
    """Build an env from ``spec`` by padding cards + replaying its action log.

    Pure with respect to the session. Raises HTTP 400 when the spec cannot
    produce a legal engine state (duplicate/out-of-range card, engine refusal
    at reset, stacks the engine can't represent).
    """
    padded = _pad_cards(
        spec.hero_hole, spec.flop_a, spec.flop_b,
        spec.turn_cards, spec.river_cards,
    )
    cfg = spec.cfg
    hero_seat = int(spec.hero_seat)

    # Build the in-hand mask from the locked hand-start membership. When the
    # session has not yet committed a hand-start (mask is empty) we pass
    # None — the engine treats every seat as in-hand, matching the prior
    # default-stack behaviour. Once a real mask is available, sitting-out
    # seats no longer post antes (engine-side); this collapses the inflated
    # 6-seat pot down to the real heads-up/3-way pot the OCR sees.
    in_hand_mask: list[bool] | None
    if spec.hand_in_hand_mask:
        in_hand_mask = [
            i in spec.hand_in_hand_mask for i in range(spec.num_seats)
        ]
        if hero_seat not in spec.hand_in_hand_mask:
            in_hand_mask = None
        elif sum(in_hand_mask) < 2:
            in_hand_mask = None
    else:
        in_hand_mask = None

    is_nlh = spec.is_nlh
    try:
        env = BombPotEnv(cfg)
        if is_nlh:
            # NLH study starts PREFLOP with a 2-card hole; blinds post in
            # the engine. No in-hand mask (live capture is PLO-only).
            env.reset_study_nlh(
                button=int(spec.button_seat),
                hero_seat=hero_seat,
                hero_hole=list(padded["hero_hole"]),
            )
        else:
            env.reset_study(
                button=int(spec.button_seat),
                hero_seat=hero_seat,
                hero_hole=list(padded["hero_hole"]),
                flop_a=list(padded["flop_a"]),
                flop_b=list(padded["flop_b"]),
                in_hand_mask=in_hand_mask,
            )
    except (ValueError, OverflowError) as e:
        # OverflowError: stacks numpy can't pack into the engine's u64 array
        # (negative / >= 2**64). Handlers bound their inputs; this keeps an
        # out-of-band bad config a 400 rather than a 500.
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

        (review 2026-09-20 I8) Hero is NEVER auto-acted, even when the
        anchor frame missed hero (hero outside the mask ⇒ the engine deals
        every seat in). Auto-checking/folding hero silently burned hero's
        turn, so the UI never waited on hero and produced no recommendation
        all hand; hero's actions must always be explicit log entries.
        """
        if not spec.hand_in_hand_mask:
            return
        all_seats = frozenset(range(spec.num_seats))
        structurally_sitting = (
            all_seats - spec.hand_in_hand_mask - {hero_seat}
        )
        if not structurally_sitting:
            return
        for _ in range(spec.num_seats * 4):
            _advance_streets()
            actor = env.current_actor()
            if actor is None or int(actor) not in structurally_sitting:
                return
            # Only bet_to_call is read: skip the opp-outcome Monte Carlo.
            raw = env._rs.observation_dict(skip_outcome_mc=True)
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
    seat_mismatches: list[tuple[int, int, int]] = []
    for log_idx, entry in enumerate(spec.action_log):
        _advance_streets()
        _auto_fold_sitting_out()
        gate = int(entry["gate"])
        chips = int(entry["chips"])
        # (review 2026-09-20 F15) Entries are applied to the ENGINE's current
        # actor, whoever recorded them. When the recorder (OCR walk / manual
        # click) attributed the entry to a different seat, the log and the
        # table have diverged — surface it; replay semantics are unchanged.
        recorded_seat = entry.get("seat")
        if recorded_seat is not None:
            actor_now = env.current_actor()
            if actor_now is not None and int(actor_now) != int(recorded_seat):
                seat_mismatches.append(
                    (log_idx, int(recorded_seat), int(actor_now))
                )
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
            except Exception:  # noqa: BLE001 (OPS-028) — replay continues unclamped
                logger.warning("max_raise_chips failed replaying %s", entry, exc_info=True)
        try:
            env.step_hybrid(gate, chips)
            kept.append(entry)
        except Exception as e:
            logger.warning(
                "rebuild_env skipping illegal action %s: %s", entry, e
            )

    _advance_streets()
    _auto_fold_sitting_out()

    return _EnvBuild(
        env=env,
        kept_log=kept,
        dropped_entries=len(kept) != len(spec.action_log),
        reached_turn=reached_turn,
        reached_river=reached_river,
        seat_mismatches=seat_mismatches,
    )


def _commit_env_build(build: _EnvBuild) -> None:
    """Install a finished build as the session's env.

    The session's card spec / config / action log must already hold the
    state the build was made from (a handler commits its candidate fields
    first, then calls this).

    Updates snapshot_at_turn / snapshot_at_river when replay reaches those
    streets for the first time. If a previous replay reached a later
    street but this one doesn't (e.g. after /undo), clears the stale
    snapshot so the "modified" indicator stops showing.
    """
    # Entries the engine rejected are dropped for future ticks (see the
    # replay loop). Only replace the list when something was dropped.
    if build.dropped_entries:
        session.action_log = build.kept_log

    if build.reached_turn:
        if session.snapshot_at_turn is None:
            session.snapshot_at_turn = {
                "flop_a": list(session.flop_a),
                "flop_b": list(session.flop_b),
            }
    else:
        session.snapshot_at_turn = None

    if build.reached_river:
        if session.snapshot_at_river is None:
            session.snapshot_at_river = {
                "flop_a": list(session.flop_a),
                "flop_b": list(session.flop_b),
                "turn": list(session.turn_cards),
            }
    else:
        session.snapshot_at_river = None

    session.env = build.env
    _refresh_obs()

    fresh = [
        m for m in build.seat_mismatches
        if m not in session._replay_mismatch_warned
    ]
    if fresh:
        session._replay_mismatch_warned = frozenset(
            session._replay_mismatch_warned | set(fresh)
        )
        for log_idx, recorded_seat, actor_now in fresh:
            logger.warning(
                "replay: action_log[%d] was recorded for seat %d but the "
                "engine applied it to seat %d — log and table have diverged",
                log_idx, recorded_seat, actor_now,
            )

    _sync_folds_with_engine()


def _rebuild_env() -> None:
    """Rebuild env from session state by padding + replaying action_log.

    The canonical path: build from the session exactly as it stands, then
    commit. Raises (HTTP 400) without touching ``session.env`` when the
    current state can't produce a legal engine state.
    """
    _commit_env_build(_build_env(_session_env_spec()))


def _sync_folds_with_engine() -> None:
    """Make ``folded_this_hand`` agree with the engine after a rebuild.

    (review 2026-09-20 F9/F11) ``folded_this_hand`` is written when a FOLD is
    *recorded*, but the engine is what the walk, the UI and the next replay
    actually follow. They diverged two ways: (1) a recorded FOLD the engine
    rejected (no bet to face) was dropped from the log yet left the seat in
    ``folded_this_hand`` ⇒ ``EngineView.sitting_out`` skipped a seat the
    engine was still waiting on, and every later action landed one seat off;
    (2) ``/undo`` of an OCR fold popped the log entry but the seat stayed
    folded. So after every rebuild the set is re-derived from the engine's
    folded flags over the seats dealt into the hand. This is the ONE place
    the set may shrink mid-hand, and only when the engine says the seat is
    live. Live mode only — without a hand-start mask (plain study) the set
    and ``sitting_out_seats`` stay untouched.
    """
    mask = session.hand_in_hand_mask
    env = session.env
    if not mask or env is None:
        return
    info = session.last_info
    if info is not None and getattr(info, "raw_obs", None) is not None:
        flags = info.raw_obs["folded"]
    else:
        flags = env._rs.observation_dict(skip_outcome_mc=True)["folded"]
    engine_folded = frozenset(
        i for i in mask if i < len(flags) and bool(flags[i])
    )
    if engine_folded != session.folded_this_hand:
        session.folded_this_hand = engine_folded
    all_seats = frozenset(range(session.num_seats))
    session.sitting_out_seats = (all_seats - mask) | session.folded_this_hand


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

#: One action — shared with the Trainer API (BE-010).
ActionRequest = _CommonActionRequest


class CardsRequest(BaseModel):
    """The card spec. An omitted field KEEPS the session's cards for it
    (BE-014): the old PLO5-shaped defaults cleared them and made a partial
    NLH request fail with "must be length 2, got 5". Lists are bounded so a
    huge array is refused before any work."""

    hero_hole: list[int | None] | None = Field(default=None, max_length=8)
    flop_a: list[int | None] | None = Field(default=None, max_length=3)
    flop_b: list[int | None] | None = Field(default=None, max_length=3)
    turn: list[int | None] | None = Field(default=None, max_length=2)
    river: list[int | None] | None = Field(default=None, max_length=2)


#: Upper bound for any chip quantity a request may set (per-seat stack, the
#: table total, bb, ante). The engine carries chips as u64 and sums commits
#: into the pot: six seats at 2**62 each silently wrapped the pot mod 2**64,
#: and anything negative / above 2**64 raised OverflowError inside numpy ⇒
#: every later rebuild 500'd. (review 2026-09-20 H7)
_MAX_CHIPS = 2**62
#: Sanity ceiling for "$ per big blind". Finite-but-absurd values (1e308)
#: overflow the cents↔chips conversions the live path runs every tick.
_MAX_DOLLARS_PER_BB = 1_000_000.0


class SeatsRequest(BaseModel):
    num_seats: int | None = Field(default=None, ge=2, le=6)
    button_seat: int | None = Field(default=None, ge=0, le=5)
    starting_stacks: list[int] | None = None
    # `starting_stacks` historically carries the client's "chips behind" per
    # seat. True = the list is ENGINE STARTING stacks, used verbatim. See
    # `seats()` for how an unflagged list is treated on a seat/button change.
    stacks_are_starting: bool = False


class ConfigRequest(BaseModel):
    bb_chips: int | None = Field(default=None, ge=1, le=_MAX_CHIPS)
    ante_chips: int | None = Field(default=None, ge=0, le=_MAX_CHIPS)
    dollars_per_bb: float | None = Field(
        default=None, gt=0, le=_MAX_DOLLARS_PER_BB, allow_inf_nan=False
    )
    starting_stacks: list[int] | None = None


class FormatRequest(BaseModel):
    # Validated against the live `FORMATS` registry in the handler (BE-011):
    # a new format needs no second list of ids here.
    format: str = Field(..., min_length=1, max_length=64)


# --- Helpers ----------------------------------------------------------------

_GATE_NAME_TO_IDX = _COMMON_GATE_NAME_TO_IDX


def _chips_to_bb(chips: int | float) -> float:
    return float(chips) / float(session.game_config.bb)


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
    # Public fields only: skip the opponent Monte-Carlo (PERF-021).
    raw = env._rs.observation_dict(skip_outcome_mc=True)
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
    # Every per-seat array the encoder indexes by seat must be projected.
    # (review 2026-09-20 B9) `acted_this_street` (STK-1's pending-opponent
    # rule, obs ≥ 1024 wide) was left in PHYSICAL seat order, so the encoder
    # read another seat's acted bit for every short-handed live hand.
    for key in (
        "folded",
        "all_in",
        "stacks",
        "eff_stack_cap",
        "street_commit",
        "total_commit",
        "acted_this_street",
    ):
        if raw.get(key) is not None:
            proj[key] = [raw[key][p] for p in comp_to_phys]
    proj["actor"] = 0
    raw_agg = int(raw.get("last_aggressor", -1))
    proj["last_aggressor"] = phys_to_comp.get(raw_agg, -1) if raw_agg >= 0 else -1
    # (review 2026-09-20 B9) A dead button (parked on a seat that wasn't
    # dealt in) has no compressed index; the old `.get(..., 0)` fallback
    # encoded HERO as the button. Map it to the in-hand seat that actually
    # acts last — the first one counter-clockwise from the physical button.
    proj["button"] = phys_to_comp[
        _effective_button(
            int(raw["button"]), cfg.num_seats, session.hand_in_hand_mask
        )
    ]
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

    compressed_cfg = dataclasses.replace(
        cfg,
        num_seats=n,
        starting_stacks=tuple(int(cfg.resolved_stacks[p]) for p in comp_to_phys),
    )
    return encode_observation(proj, compressed_cfg)



def _anchors_payload(
    spec: Any, anchor_probs: Any, anchor_chips: Any, anchor_legal: Any
) -> list[dict[str, Any]]:
    """Legal-only anchor rows for the client's bet curve, built from
    spec-length parallel arrays (probability, raise-by chips, legality).

    ONE builder for every recommendation source — the format's PPO heads and
    the strategy host — so the client never reconstructs rows (pot fractions,
    labels, the ALL-IN atom) from raw arrays.
    """
    return _common_anchors_payload(
        spec, anchor_probs, anchor_chips, anchor_legal, int(session.game_config.bb)
    )


def _dealt_in_seat_count() -> int:
    """Seats dealt into the current hand — i.e. the seat count of the
    observation `_network_obs` serves (it compresses the table to the
    hand-start mask when one is locked and includes hero; NLH study never
    has a mask, so this is the table size there)."""
    mask = session.hand_in_hand_mask
    if mask and int(session.hero_seat) in mask:
        return len(mask)
    return int(session.game_config.num_seats)


def _gto_host_covers_node(host: Any, info: Any) -> bool:
    """Whether the strategy host was TRAINED on this node's table shape and
    street.

    (review 2026-09-20 F11) `PolicyNetHost.supports` answers from the
    checkpoint's recorded training coverage (e.g. heads-up rivers only;
    preflop never counts). Study used to serve the host on every NLH node, so
    a river-only teacher produced preflop / multiway "recommendations" from
    inputs it never saw. ``seats`` is the table shape of the observation the
    host would be fed, which is what its coverage records — NOT the number of
    players still live: a 6-max hand that got heads-up is still a 6-seat obs.
    A host without the method predates the coverage API and keeps the old
    behaviour; a check that fails is treated as "not covered".
    """
    supports = getattr(host, "supports", None)
    if not callable(supports):
        return True
    try:
        raw = getattr(info, "raw_obs", None) or {}
        street = raw.get("street")
        if street is None:
            street = session.env._rs.observation_dict(skip_outcome_mc=True)["street"]
        return bool(supports(seats=_dealt_in_seat_count(), street=int(street)))
    except Exception:
        logger.exception(
            "strategy host supports() failed — serving the format's PPO model"
        )
        return False


def _recommendation_from_nodedist(nd, info, spec: Any = None) -> dict[str, Any]:
    """Build study recommendation payload from StrategyBackend NodeDist.

    ``spec`` is the anchor spec of the host's model; with it the payload
    carries the same ready-made ``anchors`` rows as a PPO recommendation."""
    d = nd.as_dict()
    gate = int(d["rec_gate"])
    chips = int(d["rec_chips"]) if gate == GATE_RAISE else None
    chips_bb = round(_chips_to_bb(chips), 4) if chips is not None else None
    gate_slug = _COMMON_GATE_SLUGS[gate]
    out: dict[str, Any] = {
        "gate": gate_slug,
        "gate_name": GATE_NAMES[gate],
        "chips": chips,
        "chips_bb": chips_bb,
        "value_bb": round(float(d["value_bb"]), 4),
        "gate_distribution": [round(p, 4) for p in d["gate_probs"]],
        "backend": d.get("backend_name", "policy_net"),
        "mode": d.get("mode", "policy_net"),
        "is_gto": True,
    }
    if d.get("head_version", 1) >= 2 and d.get("anchor_probs"):
        out["head_version"] = 2
        out["rec_anchor"] = d.get("rec_anchor")
        out["anchor_probs"] = [round(float(p), 4) for p in d["anchor_probs"]]
        out["anchor_chips"] = list(d.get("anchor_chips") or [])
        out["anchor_legal"] = list(d.get("anchor_legal") or [])
        out["pot_ref_chips"] = d.get("pot_ref_chips")
        # Ready-made rows from the same builder as the PPO path, so the
        # client stops reconstructing them from the raw arrays above (which
        # stay, additively). Skipped when the arrays don't line up with the
        # spec — the client's own fallback still covers that.
        if spec is not None and (
            len(d["anchor_probs"])
            == len(out["anchor_chips"])
            == len(out["anchor_legal"])
            == spec.count
        ):
            out["anchors"] = _anchors_payload(
                spec, d["anchor_probs"], out["anchor_chips"], out["anchor_legal"]
            )
            out["anchor_count"] = int(spec.count)
    return out


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
    gto_host = current_site().gto_host
    # NLH + GTO host: serve PolicyNet (Mode 0) instead of PPO placeholder —
    # but only on nodes inside the checkpoint's recorded training coverage
    # (see `_gto_host_covers_node`). Anything else falls back to the format's
    # PPO model, and the payload says so: `mode: "ppo"` keeps the client from
    # treating it as strategy-host output (so an untrained placeholder is
    # still badged as one) and `gto_unsupported` tells it why.
    if (
        _engine_variant() == VARIANT_NLH
        and gto_host is not None
        and info is not None
    ):
        if _gto_host_covers_node(gto_host, info):
            nd = gto_host.node_distribution(obs_np, info)
            return _recommendation_from_nodedist(
                nd,
                info,
                spec=getattr(
                    getattr(gto_host, "model", None), "anchor_spec", NLH_ANCHOR_SPEC
                ),
            )
        rec = _format_model_recommendation(fmt, obs_np, info)
        rec["mode"] = "ppo"
        rec["gto_unsupported"] = True
        return rec
    return _format_model_recommendation(fmt, obs_np, info)


def _candidate_available() -> bool:
    """The admin candidate slot is loaded and this user may use it."""
    entry = current_site().formats.get(FORMAT_EXPERIMENTAL) or {}
    return bool(entry.get("loaded")) and bool(entry.get("available", True)) and not \
        _format_locked(FORMAT_EXPERIMENTAL)


def _candidate_recommendation() -> dict[str, Any] | None:
    """(FEAT-025) The candidate checkpoint's answer to the SAME Study spot, so
    an admin can judge a new model on real spots before promoting it. PLO5
    spots only (the candidate slot plays PLO5); None whenever the live
    recommendation would be None."""
    if not _candidate_available() or _engine_variant() != VARIANT_PLO5:
        return None
    if session.variant == FORMAT_EXPERIMENTAL:
        return None  # already serving the candidate
    env = session.env
    if env is None or env.current_actor() != session.hero_seat:
        return None
    if _hero_blocking_reason() is not None or session.last_info is None:
        return None
    obs_np = _network_obs()
    if obs_np is None:
        return None
    entry = current_site().formats[FORMAT_EXPERIMENTAL]
    rec = _format_model_recommendation(entry, obs_np, session.last_info)
    rec["label"] = entry.get("label")
    rec["checkpoint"] = entry.get("checkpoint")
    return rec


def _format_model_recommendation(
    fmt: dict[str, Any], obs_np: np.ndarray, info: Any
) -> dict[str, Any]:
    """Recommendation from the active format's own (PPO) model."""
    model = fmt["model"]
    device = current_site().device
    obs_t = torch.from_numpy(fmt["adapter"](obs_np)).unsqueeze(0).to(device)
    gm_t = torch.from_numpy(info.gate_mask).unsqueeze(0).to(device)
    if getattr(model, "head_version", 1) >= 2:
        return _recommendation_v2(model, obs_t, gm_t, info)
    raise_max = int(info.max_raise_chips)
    raise_min = min(int(info.min_raise_chips), raise_max)
    bounds_t = torch.tensor(
        [[raise_min, raise_max]], dtype=torch.long, device=device
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
    gate_slug = _COMMON_GATE_SLUGS[gate]
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
    sizing_t = torch.from_numpy(sizing[None, :]).to(current_site().device)
    with torch.inference_mode():
        gate_logits, anchor_head_out, refine, value = model(obs_t, gm_t)
        gate_probs = F.softmax(gate_logits, dim=-1).squeeze(0).tolist()
        # (PERF-016) `act` = forward + `_act_from_heads`: reuse these heads
        # instead of a second forward (bit-identical; deterministic).
        _act_out = model._act_from_heads(
            gate_logits, anchor_head_out, refine, value, sizing_t, deterministic=True,
        )
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
    anchors = _anchors_payload(spec, anchor_probs, grid.chips, grid.legal)
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
    gate_slug = _COMMON_GATE_SLUGS[gate]
    # v5 mixture heads: expose the per-component (mu, s, w) so the client
    # can annotate the multi-modal menu. The `anchors` histogram already
    # renders the mixture marginal — this block is purely additive.
    mixture_block = None
    if hasattr(model, "mixture_params"):
        with torch.no_grad():
            mu_t, s_t, w_t = model.mixture_params(anchor_head_out)
        mixture_block = {
            "mu": [round(float(x), 4) for x in mu_t.squeeze(0).tolist()],
            "s": [round(float(x), 4) for x in s_t.squeeze(0).tolist()],
            "w": [round(float(x), 4) for x in w_t.squeeze(0).tolist()],
        }

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
        "mixture": mixture_block,
        "model_loaded": bool(_fmt()["loaded"]),
    }


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
    if env is None:
        # No env yet: a fresh per-user Session (public build) or a live
        # hand-start that just invalidated it. Build on demand instead of
        # asserting — `POST /format` with the unchanged format on a fresh
        # session used to 500 here. (review 2026-09-20 F13)
        _rebuild_env()
        env = session.env
    assert env is not None
    # The table projection reads public fields only — no opponent MC (PERF-021).
    raw = dict(env._rs.observation_dict(skip_outcome_mc=True))

    awaiting_idx = raw.get("awaiting_next_street")
    study_term_idx = raw.get("study_terminal")
    terminal = (
        TERMINAL_NAMES.get(int(study_term_idx)) if study_term_idx is not None else None
    )
    terminal_message = TERMINAL_MESSAGES.get(terminal) if terminal is not None else None
    awaiting = AWAITING_NAMES.get(int(awaiting_idx)) if awaiting_idx is not None else None

    cfg = session.game_config
    mask = session.hand_in_hand_mask
    hero_hole_shown = [c for c in session.hero_hole if c is not None] \
        if any(c is not None for c in session.hero_hole) else None
    # The table half (seats, pot, buttons, raise window, history, chip scale)
    # is the one the Trainer shows too: `common.table_state` (BE-008).
    state = _common_table_state(
        raw,
        cfg,
        button_seat=session.button_seat,
        hero_seat=session.hero_seat,
        info=session.last_info,
        dollars_per_bb=session.dollars_per_bb,
        position_of=_position_name,
        hole_of=lambda seat: hero_hole_shown if seat == session.hero_seat else None,
        # Mid-hand folds stay visible (they were dealt into this hand —
        # rendering them as `folded` rather than hiding them matches the real
        # table). Seats never dealt in are hidden as before.
        participant_of=lambda seat: (not mask) or (seat in mask),
        # Hero's buttons stay off until hero's hole + the cards through the
        # current street are entered.
        blocked_for=lambda actor: (
            actor == session.hero_seat and _hero_blocking_reason() is not None
        ),
    )

    recommendation = _compute_recommendation()
    candidate = _candidate_recommendation() if session.compare_candidate else None

    state.update({
        "format": session.variant,
        "format_label": _fmt()["label"],
        "format_model_loaded": bool(_fmt()["loaded"]),
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
        "terminal": terminal,
        "terminal_message": terminal_message,
        "awaiting_next_street": awaiting,
        "recommendation": recommendation,
        # (FEAT-025) The admin candidate model on the same spot; only present
        # while comparing (additive keys).
        **({"compare_candidate": True, "candidate_recommendation": candidate}
           if session.compare_candidate else {}),
        "can_undo": len(session.action_log) > 0,
        # Keys of local-build extensions (live capture: "simple_ocr_mode").
        # The public build registers none, so its payload carries no trace of
        # the live subsystems.
        **_state_extras(),
    })
    return state


# --- Validation helpers -----------------------------------------------------

def _validate_card_list(xs: list[int | None], length: int, name: str) -> list[int | None]:
    """`common.validate_card_list` (the one copy, BE-010) as an HTTP 400."""
    try:
        return _common_validate_card_list(xs, length, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# --- Error handlers (every app: `create_app`) ----------------------------------


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats (JSON has no Infinity/NaN) by their repr."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


async def _request_validation_error(request: Request, exc: RequestValidationError):
    """FastAPI's stock 422, minus one crash: it echoes the offending input,
    and a body like `{"dollars_per_bb": Infinity}` (which `json.loads`
    accepts) made the 422 itself unserializable ⇒ 500. (review 2026-09-20 H7)
    Same shape as every other error (BE-026): ``detail`` (FastAPI's list of
    field errors, which clients already read) + ``code``."""
    return JSONResponse(
        status_code=422,
        content=_mw.error_body(422, _json_safe(jsonable_encoder(exc.errors()))),
    )


async def _http_error(request: Request, exc: _StarletteHTTPException):
    """Every HTTP error in ONE shape (BE-026): ``{"detail", "code"}`` for API
    clients — ``detail`` passes through untouched (a string, or the club
    gate's dict) — and, for a browser navigation, a small branded page with
    a way back instead of a line of JSON (ACC-014 / ACC-028). 5xx carry an
    ``error_id`` that is also in the log line (OPS-024)."""
    status = int(exc.status_code)
    headers = dict(getattr(exc, "headers", None) or {})
    if status < 200 or status in (204, 304):
        return Response(status_code=status, headers=headers)
    error_id = None
    if status >= 500:
        error_id = _mw.new_error_id()
        _note_error(request, status, error_id, str(exc.detail))
    if status >= 400 and _mw.wants_html(request.headers, request.method):
        message = exc.detail if isinstance(exc.detail, str) and status not in (404, 405) else None
        return HTMLResponse(
            _mw.error_page(status, message, error_id=error_id, sign_in=status == 401),
            status_code=status,
            headers={**headers, "Cache-Control": "no-store"},
        )
    extra = {"error_id": error_id} if error_id else {}
    return JSONResponse(
        _mw.error_body(status, exc.detail, **extra), status_code=status, headers=headers
    )


async def _unhandled_error(request: Request, exc: Exception):
    """A crash: logged with an id, the user and the request, kept for the
    admin System panel, and answered with that id (OPS-024)."""
    error_id = _mw.new_error_id()
    logger.error(
        "unhandled error %s on %s %s (user %s)",
        error_id, request.method, request.url.path, _mw.request_user(request.scope),
        exc_info=exc,
    )
    _note_error(request, 500, error_id, f"{type(exc).__name__}: {exc}", log=False)
    if _mw.wants_html(request.headers, request.method):
        return HTMLResponse(
            _mw.error_page(500, error_id=error_id), status_code=500,
            headers={"Cache-Control": "no-store"},
        )
    return JSONResponse(
        _mw.error_body(500, "Something went wrong on our side.", error_id=error_id),
        status_code=500,
    )


def _note_error(request: Request, status: int, error_id: str, what: str, *, log: bool = True) -> None:
    uid = _mw.request_user(request.scope)
    if log:
        logger.warning(
            "error %s: %s %s -> %s (user %s): %s",
            error_id, request.method, request.url.path, status, uid, what,
        )
    _mw.METRICS.record_error({
        "id": error_id,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": request.method,
        "path": request.url.path,
        "status": status,
        "user": uid,
        "error": what[:300],
    })


# Trainer mode rides on the same app/models under /trainer/* (each app gets
# its own router: `create_app`). trainer.py never imports server.py back.
from plo5bp.ui import trainer as _trainer  # noqa: E402

# Optional NLH GTO PolicyNet (Phase 2a, PLO5BP_GTO_CHECKPOINT): Study
# recommendations and the Trainer share one PolicyNetHost (`Site.gto_host`).
from plo5bp.gto.policy_host import try_load_gto_host  # noqa: E402


# --- The Study API ----------------------------------------------------------------
# (BE-009) The Study routes live on ONE router, mounted twice by `create_app`:
# under /study/* — like /trainer/*, /games/api/*, /admin/api/* — so the public
# build gates them by PREFIX (a new Study route can't be left open by a
# forgotten list entry; the client calls /study/*, site 2026-09-28), and at the
# site root, the aliases a page still running the previous script calls.
study_router = APIRouter()
#: The site's other routes on this module: the format list and the candidate
#: comparison (root / one spelling only) and /health.
site_router = APIRouter()


def _mount_flat(
    app: FastAPI, router: APIRouter, prefix: str = "", *, include_in_schema: bool = True
) -> None:
    """Add each of `router`'s routes to `app` as a top-level route of its own —
    the same `APIRoute` a decorator on the app makes. (`include_router` would
    wrap the router in ONE opaque route, which the live-capture lock and the
    route checks cannot see into: they look for the study routes one by one.)"""
    for route in router.routes:
        app.add_api_route(
            prefix + route.path, route.endpoint, methods=sorted(route.methods),
            include_in_schema=include_in_schema,
        )


#: How long a Study request waits for its session's lock before a polite 429.
_STUDY_LOCK_TIMEOUT_S = float(os.environ.get("PLO5BP_SESSION_LOCK_TIMEOUT_S", "20"))


def _study_route(fn: Any) -> Any:
    """(BE-002) Run a Study handler under ITS session's lock, like every
    trainer route: handlers validate against the current action log and
    append later, so two requests of one user (two tabs, a double click)
    could interleave — a second action slipping in unvalidated, an /undo
    dropping an action appended in between. A request that can't get the
    lock in time answers 429 instead of parking a worker thread (the public
    build already queues one user's requests before they reach a thread)."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        lock = session.lock
        if not lock.acquire(timeout=_STUDY_LOCK_TIMEOUT_S):
            raise HTTPException(
                status_code=429,
                detail="Still working on your previous change — try again in a moment.",
                headers={"Retry-After": "1"},
            )
        try:
            return fn(*args, **kwargs)
        finally:
            lock.release()

    return wrapper


@study_router.get("/state")
@_study_route
def state() -> dict[str, Any]:
    if session.env is None:
        _rebuild_env()
    return {"state": _state_dict()}


def _checked_stacks(stacks: tuple[int, ...], what: str) -> tuple[int, ...]:
    """Bound a per-seat chip tuple (see `_MAX_CHIPS`); HTTP 400 otherwise."""
    for s in stacks:
        if not (0 <= int(s) <= _MAX_CHIPS):
            raise HTTPException(
                status_code=400,
                detail=f"{what}: {s} out of range [0, {_MAX_CHIPS}]",
            )
    if sum(int(s) for s in stacks) > _MAX_CHIPS:
        raise HTTPException(
            status_code=400,
            detail=f"{what}: table total exceeds {_MAX_CHIPS} chips",
        )
    return stacks


def _request_stacks(raw: list[int], num_seats: int) -> tuple[int, ...]:
    stacks = tuple(int(s) for s in raw)
    if len(stacks) != num_seats:
        raise HTTPException(
            status_code=400,
            detail=f"starting_stacks length {len(stacks)} != num_seats {num_seats}",
        )
    return _checked_stacks(stacks, "starting_stacks")


def _make_config(cfg: GameConfig, **changes: Any) -> GameConfig:
    """`dataclasses.replace` on a GameConfig with its validation errors
    (seat-count / deck-feasibility bounds) surfaced as HTTP 400."""
    try:
        return dataclasses.replace(cfg, **changes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _splice_starting_stacks(
    starting: tuple[int, ...],
    behind_now: list[int] | None,
    client_behinds: tuple[int, ...] | None,
    new_n: int,
) -> tuple[int, ...]:
    """Resize per-seat STARTING stacks for a seat-count change.

    The client describes the edit only through its spliced "chips behind"
    list, so that list is used to LOCATE the seat (compare against the
    server's own behind list), never for its values. An inserted seat copies
    its clockwise-previous neighbour (the client's own default). When the
    seat can't be located (no list, stale client, multi-seat jump) the table
    is truncated / padded at the end.
    """
    old_n = len(starting)
    out = list(starting)
    located = (
        client_behinds is not None
        and behind_now is not None
        and len(behind_now) == old_n
    )
    if located and new_n == old_n - 1:
        # Hero (seat 0) can't be removed; first match wins on ties.
        for i in range(1, old_n):
            if tuple(behind_now[:i] + behind_now[i + 1:]) == client_behinds:
                del out[i]
                return tuple(out)
    if located and new_n == old_n + 1:
        # The client inserts the copy AFTER its source seat, so on a tie
        # (copy == source) the inserted slot is the LAST matching index.
        for i in range(new_n - 1, 0, -1):
            if list(client_behinds[:i] + client_behinds[i + 1:]) == list(behind_now):
                out.insert(i, out[i - 1])
                return tuple(out)
    if new_n < old_n:
        return tuple(out[:new_n])
    return tuple(out + [out[-1]] * (new_n - old_n))


@study_router.post("/cards")
@_study_route
def cards(req: CardsRequest) -> dict[str, Any]:
    lens = dict(_card_spec_attrs())
    given = {
        "hero_hole": (req.hero_hole, "hero_hole"),
        "flop_a": (req.flop_a, "flop_a"),
        "flop_b": (req.flop_b, "flop_b"),
        "turn_cards": (req.turn, "turn"),
        "river_cards": (req.river, "river"),
    }
    # An omitted field keeps the session's cards (BE-014).
    candidate = {
        attr: _validate_card_list(
            value if value is not None else list(getattr(session, attr)),
            lens[attr], name,
        )
        for attr, (value, name) in given.items()
    }
    # Validate-then-commit (review 2026-09-20 H7): a duplicate used to 400
    # AFTER the bad spec was stored and its slots locked, so every later
    # request 400'd too.
    build = _build_env(_session_env_spec(**candidate))
    for attr, value in candidate.items():
        setattr(session, attr, value)
    _lock_filled_card_slots()
    _commit_env_build(build)
    return {"state": _state_dict()}


@study_router.post("/seats")
@_study_route
def seats(req: SeatsRequest) -> dict[str, Any]:
    cfg = session.game_config
    new_num_seats = req.num_seats if req.num_seats is not None else session.num_seats
    new_button = req.button_seat if req.button_seat is not None else session.button_seat
    if new_button >= new_num_seats:
        new_button = 0
    seats_or_button_changed = (
        new_num_seats != cfg.num_seats or new_button != session.button_seat
    )
    client_stacks = (
        _request_stacks(req.starting_stacks, new_num_seats)
        if req.starting_stacks is not None
        else None
    )

    # (review 2026-09-20 H6) A seat/button change restarts the hand, so the
    # stacks that carry over are the STARTING stacks the session already
    # holds. The client's list is "chips behind" — what is left AFTER this
    # hand's antes, blinds and bets — and the old `behind + [ante] * n`
    # conversion silently lost every blind/bet on each seat op (NLH: SB/BB
    # eroded by 0.5/1bb per click). So on a seat op an unflagged list only
    # locates which seat was spliced; its values are ignored. A caller that
    # really wants to set stacks alongside a seat op sends
    # `stacks_are_starting: true`. Without a seat op the list keeps its
    # historical meaning (behind + this hand's commits), same as /config.
    if client_stacks is not None and req.stacks_are_starting:
        new_cfg = _make_config(
            cfg,
            num_seats=new_num_seats,
            starting_stack=client_stacks[0],
            starting_stacks=client_stacks,
        )
    elif seats_or_button_changed:
        if new_num_seats == cfg.num_seats:
            new_cfg = cfg
        elif cfg.starting_stacks is None:
            new_cfg = _make_config(cfg, num_seats=new_num_seats)
        else:
            new_cfg = _make_config(
                cfg,
                num_seats=new_num_seats,
                starting_stacks=_splice_starting_stacks(
                    cfg.starting_stacks,
                    _current_behind_stacks(),
                    client_stacks,
                    new_num_seats,
                ),
            )
    elif client_stacks is not None:
        commits = _current_total_commit(new_num_seats, int(cfg.ante))
        engine_stacks = _checked_stacks(
            tuple(b + commits[i] for i, b in enumerate(client_stacks)),
            "starting_stacks",
        )
        new_cfg = _make_config(
            cfg, starting_stack=engine_stacks[0], starting_stacks=engine_stacks
        )
    else:
        new_cfg = cfg

    # Validate-then-commit (review 2026-09-20 H7).
    build = _build_env(
        _session_env_spec(
            cfg=new_cfg,
            num_seats=new_num_seats,
            button_seat=new_button,
            hero_seat=0,
            **({"action_log": []} if seats_or_button_changed else {}),
        )
    )
    session.game_config = new_cfg
    session.num_seats = new_num_seats
    session.button_seat = new_button
    # Hero stays at seat 0 internally; UI rotates so it lands at south.
    session.hero_seat = 0
    if seats_or_button_changed:
        _clear_hand_state_keep_cards()
    _commit_env_build(build)
    return {"state": _state_dict()}


@study_router.post("/action")
@_study_route
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

    # "seat" = who this entry was recorded for (diagnostic; review F15).
    entry = {"gate": int(gate_idx), "chips": int(chips), "seat": int(actor)}
    # Validate-then-commit (review 2026-09-20 H7): replay the candidate log
    # first. The old append → rebuild → pop-on-Exception dance left the entry
    # behind when the rebuild died with a non-Exception (a PyO3 panic is a
    # BaseException), wedging every later request.
    build = _build_env(
        _session_env_spec(action_log=[*session.action_log, entry])
    )
    if not build.kept_log or build.kept_log[-1] is not entry:
        raise HTTPException(
            status_code=400, detail="action rejected by the engine"
        )
    session.action_log.append(entry)
    _commit_env_build(build)
    return {"state": _state_dict()}


@study_router.post("/undo")
@_study_route
def undo() -> dict[str, Any]:
    if not session.action_log:
        raise HTTPException(status_code=400, detail="nothing to undo")
    remaining = session.action_log[:-1]
    build = _build_env(_session_env_spec(action_log=remaining))
    session.action_log = remaining
    # `_commit_env_build` re-derives `folded_this_hand` from the engine, so
    # undoing an OCR-recorded FOLD un-folds the seat. (review 2026-09-20 F11)
    _commit_env_build(build)
    return {"state": _state_dict()}


@study_router.post("/reset")
@_study_route
def reset() -> dict[str, Any]:
    _new_session_defaults()
    # Also drop the live hand-start debounce state, or the next OCR tick
    # re-seeds the "new" hand from a stale anchor. (review 2026-09-20 F10)
    _run_session_reset_hooks("user")
    _rebuild_env()
    return {"state": _state_dict()}


@site_router.get("/formats")
def formats() -> dict[str, Any]:
    """Formats the server can serve, for the UI dropdown. `model_loaded`
    False = a random-init placeholder answers (no checkpoint promoted).
    `locked` True = greyed out "coming soon!" for this user (public
    build gates non-default formats to admins while they train). The
    admin candidate slot is listed only when a candidate is configured,
    and never to a user the gate locks out (it is not "coming soon")."""
    out = []
    for vid, f in list(current_site().formats.items()):
        if not f.get("available", True):
            continue
        locked = _format_locked(vid)
        if locked and f.get("admin_only"):
            continue
        out.append({
            "id": vid,
            "label": f["label"],
            "model_loaded": bool(f["loaded"]),
            # True = the served checkpoint was trained on a different
            # observation-semantics revision than this process encodes
            # (see `models.note_checkpoint_obs_rev`). Additive key.
            "obs_rev_mismatch": bool(f.get("obs_rev_mismatch", False)),
            "locked": locked,
            # Betting cap class: pot-limit formats cap raises at pot
            # (the client's b100 preset is "pot" and nothing larger
            # exists); no-limit formats allow overbets + all-in.
            "pot_limit": _engine_variant(vid) != VARIANT_NLH,
        })
    return {"formats": out, "active": session.variant}


@study_router.post("/format")
@_study_route
def set_format(req: FormatRequest) -> dict[str, Any]:
    """Switch the study session's game format. Resets per-hand state and
    swaps the game config to the format default (`common.FORMAT_DEFAULTS`:
    PLO5 6-max 20bb with a 3bb ante; NLH 6-max 100bb 5/10 with a $5/player
    ante — the Trainer starts from the same table). The trainer's format
    follows via its own setter so both tabs stay on one game."""
    formats = current_site().formats
    if req.format not in formats:
        raise HTTPException(status_code=400, detail=f"unknown format {req.format!r}")
    if _format_locked(req.format):
        raise HTTPException(
            status_code=403,
            detail="This format isn't available on your account yet — coming soon!",
        )
    if req.format != session.variant:
        if req.format == FORMAT_EXPERIMENTAL:
            _maybe_reload_experimental()
        if not formats[req.format].get("available", True):
            raise HTTPException(status_code=400, detail="no candidate model is configured")
        session.variant = req.format
        session.game_config = _common_default_game_config(req.format)
        session.dollars_per_bb = float(_common_format_defaults(req.format)["dollars_per_bb"])
        session.num_seats = session.game_config.num_seats
        session.button_seat = 0
        session.hero_seat = 0
        _new_session_defaults()
        # A format switch is a hard reset for the live hand-start machine
        # too (stale anchor / mask from the previous game). (review F10)
        _run_session_reset_hooks("user")
        try:
            current_site().trainer_router.set_format(req.format)
        except Exception:
            logger.exception("trainer format sync failed")
        _rebuild_env()
    # Unchanged format: `_state_dict` builds the env on demand, so a fresh
    # per-user Session (env None) no longer 500s here. (review F13)
    return {"state": _state_dict()}


@study_router.post("/config")
@_study_route
def config(req: ConfigRequest) -> dict[str, Any]:
    cfg = session.game_config
    bb = int(req.bb_chips) if req.bb_chips is not None else cfg.bb
    ante = int(req.ante_chips) if req.ante_chips is not None else cfg.ante
    # NLH keeps sb = bb/2 (the 5/10 structure scales with the unit).
    sb = bb // 2 if cfg.variant == VARIANT_NLH else cfg.sb
    if req.starting_stacks is not None:
        behinds = _request_stacks(req.starting_stacks, session.num_seats)
        # `req.starting_stacks` carries "current chips behind" per
        # seat; convert to engine starting_stack via total_commit at
        # the current replay end. Hand-start collapses to behind +
        # ante (engine deducts the ante on hand init).
        commits = _current_total_commit(session.num_seats, ante)
        engine_stacks = _checked_stacks(
            tuple(b + commits[i] for i, b in enumerate(behinds)),
            "starting_stacks",
        )
        starting_stack = engine_stacks[0]
    else:
        # (review 2026-09-20 H6) No stacks in the request ⇒ keep the ones the
        # session has. Rebuilding the config without them silently reset
        # every per-seat stack to the uniform default — changing only
        # "$ / bb" kept the action log and replayed it against different
        # stacks.
        engine_stacks = (
            cfg.starting_stacks
            if cfg.starting_stacks is not None
            and len(cfg.starting_stacks) == session.num_seats
            else None
        )
        starting_stack = cfg.starting_stack
    new_cfg = _make_config(
        cfg,
        num_seats=session.num_seats,
        starting_stack=starting_stack,
        ante=ante,
        bb=bb,
        starting_stacks=engine_stacks,
        sb=sb,
    )
    # Stack-only edits preserve action_log; bb/ante changes invalidate
    # prior actions (chip math depends on the unit).
    bb_or_ante_changed = (
        (req.bb_chips is not None and int(req.bb_chips) != cfg.bb)
        or (req.ante_chips is not None and int(req.ante_chips) != cfg.ante)
    )
    # Validate-then-commit (review 2026-09-20 H7).
    build = _build_env(
        _session_env_spec(
            cfg=new_cfg,
            **({"action_log": []} if bb_or_ante_changed else {}),
        )
    )
    session.game_config = new_cfg
    if req.dollars_per_bb is not None:
        session.dollars_per_bb = float(req.dollars_per_bb)
    if bb_or_ante_changed:
        _clear_hand_state_keep_cards()
    _commit_env_build(build)
    return {"state": _state_dict()}


# --- Whole-spot load + rewind (site FEAT-016/026, FEAT-017, FEAT-020) ----------
# A Study spot is fully defined by the table (seats, button, stacks, ante),
# the cards and the action log. Share links and the Trainer's "Open in Study"
# load one in ONE validated call (validate-then-commit, like every handler
# above: a bad spot leaves the session exactly as it was); clicking a history
# row rewinds the log in one call instead of N x /undo.


class SpotAction(BaseModel):
    gate: str = Field(..., pattern=r"^(fold|check_call|raise)$")
    chips: int | None = Field(default=None, ge=0, le=_MAX_CHIPS)


class SpotRequest(BaseModel):
    # The spot's format must be the session's (switch with /format first — a
    # format switch resets both tabs, so the client asks the user).
    format: str | None = Field(
        default=None, pattern=r"^(plo5_double_bomb|nlh_single|experimental)$"
    )
    num_seats: int = Field(..., ge=2, le=6)
    button_seat: int = Field(..., ge=0, le=5)
    # ENGINE starting stacks (before antes/blinds), hero first, clockwise.
    starting_stacks: list[int] = Field(..., min_length=2, max_length=6)
    ante_chips: int | None = Field(default=None, ge=0, le=_MAX_CHIPS)
    bb_chips: int | None = Field(default=None, ge=1, le=_MAX_CHIPS)
    hero_hole: list[int | None] | None = None
    flop_a: list[int | None] | None = None
    flop_b: list[int | None] | None = None
    turn: list[int | None] | None = None
    river: list[int | None] | None = None
    actions: list[SpotAction] = Field(default_factory=list, max_length=400)


class RewindRequest(BaseModel):
    # Keep this many entries of the action log (0 = back to the deal).
    length: int = Field(..., ge=0)


@study_router.post("/spot")
@_study_route
def load_spot(req: SpotRequest) -> dict[str, Any]:
    if req.format is not None and req.format != session.variant:
        raise HTTPException(
            status_code=409,
            detail="This spot is for another format — switch format first.",
        )
    if req.button_seat >= req.num_seats:
        raise HTTPException(status_code=400, detail="button_seat must be a seat at the table")
    cfg = session.game_config
    stacks = _request_stacks(req.starting_stacks, req.num_seats)
    bb = int(req.bb_chips) if req.bb_chips is not None else int(cfg.bb)
    ante = int(req.ante_chips) if req.ante_chips is not None else int(cfg.ante)
    new_cfg = _make_config(
        cfg,
        num_seats=req.num_seats,
        starting_stack=stacks[0],
        starting_stacks=stacks,
        bb=bb,
        ante=ante,
        # NLH keeps sb = bb/2 (same rule as /config).
        sb=bb // 2 if cfg.variant == VARIANT_NLH else cfg.sb,
    )
    lens = dict(_card_spec_attrs())
    given = {
        "hero_hole": req.hero_hole, "flop_a": req.flop_a, "flop_b": req.flop_b,
        "turn_cards": req.turn, "river_cards": req.river,
    }
    cards = {
        attr: _validate_card_list(
            value if value is not None else [None] * lens[attr], lens[attr], attr
        )
        for attr, value in given.items()
    }
    log = [
        {
            "gate": int(_GATE_NAME_TO_IDX[a.gate]),
            "chips": int(a.chips or 0) if a.gate == "raise" else 0,
        }
        for a in req.actions
    ]
    build = _build_env(
        _session_env_spec(
            cfg=new_cfg,
            num_seats=req.num_seats,
            button_seat=req.button_seat,
            hero_seat=0,
            hand_in_hand_mask=frozenset(),
            action_log=log,
            **cards,
        )
    )
    if build.dropped_entries:
        bad = next(
            (i for i, e in enumerate(log)
             if i >= len(build.kept_log) or build.kept_log[i] is not e),
            len(log) - 1,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Action {bad + 1} of this spot isn't legal at that point.",
        )
    _new_session_defaults()
    session.game_config = new_cfg
    session.num_seats = req.num_seats
    session.button_seat = req.button_seat
    session.hero_seat = 0
    for attr, value in cards.items():
        setattr(session, attr, value)
    session.action_log = log
    _run_session_reset_hooks("user")
    _lock_filled_card_slots()
    _commit_env_build(build)
    return {"state": _state_dict()}


_GATE_IDX_TO_NAME = {v: k for k, v in _GATE_NAME_TO_IDX.items()}


def _spot_actions(log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"gate": _GATE_IDX_TO_NAME[int(e["gate"])], "chips": int(e["chips"])}
        for e in log
    ]


@study_router.post("/rewind")
@_study_route
def rewind(req: RewindRequest) -> dict[str, Any]:
    """Keep the first `length` actions. `removed` carries what was taken off
    (gate + chips, /spot's action shape) so the client can offer Redo."""
    if req.length >= len(session.action_log):
        return {"state": _state_dict(), "removed": []}
    remaining = session.action_log[: req.length]
    removed = _spot_actions(session.action_log[req.length:])
    build = _build_env(_session_env_spec(action_log=remaining))
    session.action_log = remaining
    _commit_env_build(build)
    return {"state": _state_dict(), "removed": removed}


@study_router.get("/spot")
@_study_route
def get_spot() -> dict[str, Any]:
    """The current Study spot in /spot's request shape (for share links)."""
    cfg = session.game_config
    return {
        "spot": {
            "format": session.variant,
            "num_seats": int(session.num_seats),
            "button_seat": int(session.button_seat),
            "starting_stacks": [int(x) for x in cfg.resolved_stacks],
            "ante_chips": int(cfg.ante),
            "bb_chips": int(cfg.bb),
            "hero_hole": list(session.hero_hole),
            "flop_a": list(session.flop_a),
            "flop_b": list(session.flop_b),
            "turn": list(session.turn_cards),
            "river": list(session.river_cards),
            "actions": _spot_actions(session.action_log),
        }
    }


class CompareRequest(BaseModel):
    on: bool


@site_router.post("/study/compare")
@_study_route
def study_compare(req: CompareRequest) -> dict[str, Any]:
    """(FEAT-025) Turn the candidate-model comparison on/off for this
    Study session (admins; only while a candidate is configured)."""
    if req.on and not _candidate_available():
        raise HTTPException(status_code=403, detail="No candidate model is available to compare.")
    session.compare_candidate = bool(req.on)
    if session.env is None:
        _rebuild_env()
    return {"state": _state_dict()}


# --- Static frontend --------------------------------------------------------
#
# Caching (FE-015 / PERF-001 / PERF-017 / PERF-020). HTML pages are
# revalidated on every visit (`no-cache` + an ETag: an unchanged page is a
# tiny 304), and every asset a page links is linked WITH ITS CONTENT HASH
# (`/static/app.js?v=3f9c2a…`). Such a URL never changes meaning, so the
# browser and Cloudflare keep it for a year (`immutable`), and a deploy that
# changes a file changes its URL — nothing stale is ever shown and nothing
# unchanged is ever downloaded twice. A request without (or with an outdated)
# `v` gets `no-cache` + ETag. Rendered pages and served-text hashes are
# cached by file (mtime, size), so a request re-reads nothing.


_NO_CACHE = {"Cache-Control": "no-store, must-revalidate"}
_REVALIDATE = "no-cache"
_IMMUTABLE = "public, max-age=31536000, immutable"


def _strip_wglive(text: str) -> str:
    """Drop every marker-bounded live-capture region (WGLIVE:START..END).

    The full local build keeps ClubGG-OCR / PokerNow client code and markup
    inside these markers; the public build serves the assets with the whole
    region removed — a public visitor's copy contains no trace (names,
    selectors, endpoints) of the live subsystems, inspectable or otherwise."""
    return re.sub(r"[^\n]*WGLIVE:START.*?WGLIVE:END[^\n]*\n?", "", text, flags=re.S)


def _strip_wgapp(text: str) -> str:
    """Drop the app-chrome region (WGAPP:START..END) from the landing HTML.

    A signed-out public visitor only ever sees the landing overlay, so the
    trainer/study chrome (top bar + table + panels) is removed server-side
    rather than merely hidden client-side. This kills the first-paint flash
    of the empty app and means there is nothing to reveal by deleting the
    overlay in devtools — the markup simply isn't sent.
    """
    return re.sub(r"[^\n]*WGAPP:START.*?WGAPP:END[^\n]*\n?", "", text, flags=re.S)


def _select_pricing_copy(html: str, public: bool | None = None) -> str:
    """Keep the landing's free-period copy (WGFREE regions) or its paid-plan
    copy (WGPAID regions) to match `PLO5BP_FREE_FOR_ALL`, and fill the paid
    copy's {{PRICE}} / {{FREE_HANDS}} from the service settings — so turning
    the paywall back on can never leave the page promising "free" (site
    ACC-017). Marker lines are removed either way. The local build has no
    service layer (and never imports it): free copy, never shown there.
    `public`: the page's build (default: the current site's)."""
    free, price_cents, hands = True, 1000, 5
    if current_site().public if public is None else public:
        try:
            from plo5bp.ui import public as _pub

            free = bool(_pub.FREE_FOR_ALL)
            price_cents = int(_pub.PRICE_CENTS)
            hands = int(_pub.FREE_HANDS_PER_DAY)
        except Exception:  # noqa: BLE001 — keep the page up; free copy
            logger.exception("landing: pricing settings unavailable")
    drop, keep = ("WGPAID", "WGFREE") if free else ("WGFREE", "WGPAID")
    html = re.sub(rf"[^\n]*{drop}:START.*?{drop}:END[^\n]*\n?", "", html, flags=re.S)
    html = re.sub(rf"[^\n]*{keep}:(?:START|END)[^\n]*\n?", "", html)
    dollars = price_cents / 100
    price = f"${dollars:,.0f}" if price_cents % 100 == 0 else f"${dollars:,.2f}"
    return html.replace("{{PRICE}}", price).replace("{{FREE_HANDS}}", str(hands))


#: Self-hosted brand fonts (site FE-024 / PERF-002): (family, file in
#: static/fonts/). While every file is there, pages load the fonts from this
#: site; until then they keep the Google Fonts links (a render-blocking
#: stylesheet from another origin, which also tells Google about the visit).
_FONT_FILES = (("Geist", "Geist-Variable.woff2"), ("Geist Mono", "GeistMono-Variable.woff2"))


def _font_versions(versions: "_AssetVersions") -> tuple[str, ...] | None:
    """Content versions of the self-hosted font files as served — None until
    every one of them is in static/fonts/."""
    got = tuple(versions.version(f"fonts/{name}") for _, name in _FONT_FILES)
    return None if any(v is None for v in got) else got


def _font_links(html: str, versions: tuple[str, ...] | None) -> str:
    """Swap a page's WGFONTS region (the Google Fonts links) for the
    self-hosted fonts when `versions` (from `_font_versions`) says they are
    there; otherwise keep the region and drop only its marker lines. The
    @font-face rules are written inline with content-versioned URLs, so the
    font request starts as soon as the head is parsed (the main face is also
    preloaded — the same URL, one download) and is cached for a year like
    every other versioned asset. A WGFONTNOTE region (the privacy policy's
    line about Google serving the fonts) is kept only while Google does."""
    if not versions:
        return re.sub(r"[^\n]*WGFONT(?:S|NOTE):(?:START|END)[^\n]*\n?", "", html)
    urls = [f"/static/fonts/{name}?v={v}" for (_, name), v in zip(_FONT_FILES, versions)]
    faces = "".join(
        '@font-face{font-family:"' + family + '";src:url("' + url + '") format("woff2");'
        "font-weight:100 900;font-style:normal;font-display:swap}"
        for (family, _), url in zip(_FONT_FILES, urls)
    )
    block = (
        '    <link rel="preload" href="' + urls[0] + '" as="font" type="font/woff2" crossorigin />\n'
        "    <style>" + faces + "</style>\n"
    )
    html = re.sub(r"[^\n]*WGFONTNOTE:START.*?WGFONTNOTE:END[^\n]*\n?", "", html, flags=re.S)
    return re.sub(r"[^\n]*WGFONTS:START.*?WGFONTS:END[^\n]*\n?", lambda _m: block, html, flags=re.S)


def _strip_local_only_scripts(html: str) -> str:
    """Drop the <script> tags of assets the public static mount refuses, so
    a public page never requests (and console-404s on) them."""
    return re.sub(
        r"[^\n]*<script\s+src=\"/static/ranges\.js\"\s*>\s*</script>[^\n]*\n?",
        "",
        html,
    )


#: Public build (SEC-014): the /static mount is an ALLOW-LIST. "strip" = text
#: served with the WGLIVE regions removed; "serve" = served as is; every file
#: under a directory of `_PUBLIC_STATIC_DIRS` is served as is. ANY other file —
#: "deny" below, or not listed at all (a new file, or one left on the server
#: after it left the repo) — 404s. The "deny" files have their own gated
#: homes: index.html is `/` (which also applies the signed-out WGAPP strip),
#: the home-games client (every games.*) `/games/static/{name}` + `/games`,
#: admin.html / admin.js `/admin`, the legal pages `/terms` + `/privacy`, and
#: ranges.js belongs to the local-only /ranges feature. The policy is
#: enforced on the file a request RESOLVES to (see `SiteStaticFiles`).
_PUBLIC_STATIC_POLICY: dict[str, str] = {
    "app.js": "strip",
    # The rest of the Study / Trainer client, loaded before app.js (site FE-019).
    "app.core.js": "strip",
    "app.table.js": "strip",
    "app.play.js": "strip",
    "app.study.js": "strip",
    "app.trainer.js": "strip",
    "app.topbar.js": "strip",
    "style.css": "strip",
    "landing.js": "strip",
    "index.html": "deny",
    "ranges.js": "deny",
    "admin.html": "deny",
    "admin.js": "deny",
    "terms.html": "deny",
    "privacy.html": "deny",
}
# "fonts": the self-hosted Geist files, once the owner adds them (site FE-024).
_PUBLIC_STATIC_DIRS = ("brand", "fonts")
_TEXT_MEDIA_TYPES = {
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


def _static_rel(directory: str, full_path: str) -> str | None:
    """`full_path`'s location inside `directory` after resolving links, 8.3
    names and case (normcase), as a forward-slash path; None if outside."""
    base = os.path.normcase(os.path.realpath(directory))
    target = os.path.normcase(os.path.realpath(full_path))
    try:
        rel = os.path.relpath(target, base)
    except ValueError:  # another drive (Windows)
        return None
    if rel == os.curdir or rel.startswith(os.pardir):
        return None
    return rel.replace(os.sep, "/")


class _AssetVersions:
    """Content hash of each static file AS SERVED (stripped in the public
    build), cached by (mtime, size) — the `?v=` of versioned links."""

    def __init__(self, directory: Path, public: bool) -> None:
        self.directory = Path(directory)
        self.public = public
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[tuple[int, int], bytes, str]] = {}

    def policy(self, rel: str) -> str | None:
        """How the mount serves `rel` (normcased, forward slashes): "strip",
        "serve", or None = 404. The local build serves everything as is."""
        if not self.public:
            return "serve"
        for name, pol in _PUBLIC_STATIC_POLICY.items():
            if os.path.normcase(name) == rel:
                return pol if pol in ("strip", "serve") else None
        top = rel.split("/", 1)[0]
        if "/" in rel and any(os.path.normcase(d) == top for d in _PUBLIC_STATIC_DIRS):
            return "serve"
        return None

    def body(self, rel: str) -> tuple[bytes, str] | None:
        """(served bytes, version) — None when the file does not exist."""
        path = self.directory / rel
        try:
            st = path.stat()
        except OSError:
            return None
        key = (int(st.st_mtime_ns), int(st.st_size))
        with self._lock:
            hit = self._cache.get(rel)
            if hit is not None and hit[0] == key:
                return hit[1], hit[2]
        if self.public and self.policy(rel) == "strip":
            # Text read with universal newlines (served LF), then stripped.
            data = _strip_wglive(path.read_text(encoding="utf-8")).encode("utf-8")
        else:
            data = path.read_bytes()
        version = hashlib.sha256(data).hexdigest()[:16]
        with self._lock:
            self._cache[rel] = (key, data, version)
        return data, version

    def version(self, rel: str) -> str | None:
        got = self.body(os.path.normcase(rel).replace(os.sep, "/"))
        return got[1] if got is not None else None


def _query_param(scope: dict[str, Any], name: str) -> str | None:
    raw = scope.get("query_string", b"").decode("latin-1")
    for part in raw.split("&"):
        k, _, v = part.partition("=")
        if k == name:
            return v
    return None


def _not_modified(scope: dict[str, Any], etag: str) -> bool:
    for k, v in scope.get("headers") or ():
        if k == b"if-none-match":
            tags = {t.strip() for t in v.decode("latin-1").split(",")}
            return etag in tags or "*" in tags
    return False


class SiteStaticFiles(StaticFiles):
    """The /static mount: content-hash caching + the public allow-list.

    (review 2026-09-20 F1) The public build used to shadow the mount with
    exact-path routes and rely on the access middleware's exact-path
    blocklist — both compare the RAW url, while the mount normalizes it — so
    `/static//app.js` served the unstripped OCR/PokerNow client and
    `/static//games.js` the hidden home-games client. The policy lives HERE,
    on the file the request actually resolves to, so no spelling of the path
    (doubled or trailing slashes, `.` segments, case on a case-insensitive
    filesystem, 8.3 names, links) can reach a protected file around it — and
    since 2026-09-28 it is an allow-list (SEC-014)."""

    def __init__(self, *args: Any, public: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.public = public
        if public:
            # Every home-games client file (games.*) is served only from its
            # gated /games/static route (HGB-017) — denied here like any file
            # not on the allow-list; listed so tests can classify it.
            for f in Path(str(self.directory)).glob("games.*"):
                _PUBLIC_STATIC_POLICY.setdefault(f.name, "deny")
        self.versions = _AssetVersions(Path(str(self.directory)), public)

    async def get_response(self, path, scope):
        if scope["method"] not in ("GET", "HEAD"):
            return await super().get_response(path, scope)
        try:
            full_path, stat_result = await anyio.to_thread.run_sync(self.lookup_path, path)
        except (OSError, ValueError):
            # Unresolvable path (too long, NUL byte…): the stock 404.
            raise HTTPException(status_code=404)
        if stat_result is None or not os.path.isfile(full_path):
            raise HTTPException(status_code=404)
        rel = _static_rel(str(self.directory), full_path)
        policy = self.versions.policy(rel) if rel is not None else None
        if policy is None:
            raise HTTPException(status_code=404)
        got = await anyio.to_thread.run_sync(self.versions.body, rel)
        if got is None:
            raise HTTPException(status_code=404)
        data, version = got
        etag = f'"{version}"'
        cache = _IMMUTABLE if _query_param(scope, "v") == version else _REVALIDATE
        headers = {"Cache-Control": cache, "ETag": etag}
        if _not_modified(scope, etag):
            return Response(status_code=304, headers=headers)
        ext = os.path.splitext(rel)[1].lower()
        media = _TEXT_MEDIA_TYPES.get(ext)
        if media is None:
            import mimetypes

            media = mimetypes.guess_type(rel)[0] or "application/octet-stream"
        return Response(data, media_type=media, headers=headers)


#: Historical name (tests, docs).
NoCacheStaticFiles = SiteStaticFiles

_STATIC_REF_RE = re.compile(r'(\b(?:src|href)=")/static/([^"?#]+)(")')


class _PageRenderer:
    """Renders a server-side HTML page once per (file state, variant, asset
    versions): versioned asset links, and the page's CSP with the hashes of
    its (server-written) inline scripts."""

    def __init__(self, versions: _AssetVersions) -> None:
        self.versions = versions
        self._lock = threading.Lock()
        self._cache: dict[tuple, Any] = {}

    def render(self, name: str, variant: tuple, transform: Any) -> tuple[str, str, str]:
        """(html, etag, csp) of static `name` after `transform(html)`."""
        path = self.versions.directory / name
        st = path.stat()
        base_key = (name, int(st.st_mtime_ns), int(st.st_size), variant)
        with self._lock:
            src = self._cache.get(("src",) + base_key)
        if src is None:
            html = transform(path.read_text(encoding="utf-8"))
            refs = tuple(sorted({m.group(2) for m in _STATIC_REF_RE.finditer(html)}))
            src = (html, refs)
            with self._lock:
                self._cache[("src",) + base_key] = src
        html, refs = src
        versions = tuple(self.versions.version(r) for r in refs)
        key = base_key + versions
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        vmap = dict(zip(refs, versions))

        def link(m: "re.Match[str]") -> str:
            v = vmap.get(m.group(2))
            suffix = f"?v={v}" if v else ""
            return f"{m.group(1)}/static/{m.group(2)}{suffix}{m.group(3)}"

        out = _STATIC_REF_RE.sub(link, html)
        etag = 'W/"' + hashlib.sha256(out.encode("utf-8")).hexdigest()[:20] + '"'
        csp = _mw.site_csp(_mw.inline_script_hashes(out))
        result = (out, etag, csp)
        with self._lock:
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = result
        return result


def _page_response(request: Request, html: str, etag: str, csp: str, *, private: bool) -> Response:
    headers = {
        "Cache-Control": f"{_REVALIDATE}, private" if private else _REVALIDATE,
        "ETag": etag,
        "Content-Security-Policy": csp,
    }
    if private:
        headers["Vary"] = "Cookie"
    if _not_modified(request.scope, etag):
        return Response(status_code=304, headers=headers)
    return HTMLResponse(html, headers=headers)


def _site_base_url(request: Request, site: Site | None = None) -> str:
    base = (site or current_site()).settings.base_url
    return base or str(request.base_url).rstrip("/")


def _install_pages(app: FastAPI, site: Site) -> None:
    """The /static mount and the server-rendered pages of one app: `/`, the
    legal pages, the icons, robots.txt and the sitemap."""
    public = site.public
    site.static = SiteStaticFiles(directory=STATIC_DIR, public=public)
    app.mount("/static", site.static, name="static")
    pages = site.pages = _PageRenderer(site.static.versions)

    def _index_transform(signed_in: bool, fonts: tuple[str, ...] | None = None) -> Any:
        def transform(html: str) -> str:
            if public:
                html = _strip_local_only_scripts(_strip_wglive(html))
                if not signed_in:
                    html = _strip_wgapp(html)
            html = _select_pricing_copy(html, public)
            html = _font_links(html, fonts)
            # The build mode, known to the client before first paint (the
            # page's CSP allows exactly this inline script, by its hash).
            flag = "true" if public else "false"
            return html.replace(
                "</head>", f"<script>window.PLO5BP_PUBLIC={flag};</script></head>", 1
            )

        return transform

    @app.get("/")
    def index(request: Request) -> Response:
        signed_in = True
        if public:
            # Signed-out visitors get the landing page ONLY — the app chrome
            # is stripped server-side so it can't flash on load or be revealed
            # by deleting the overlay. session.uid is set at login (public.py).
            try:
                signed_in = bool(request.session.get("uid"))
            except (AssertionError, KeyError):
                # SessionMiddleware not installed (shouldn't happen in the
                # public build) — fall back to sending the full markup.
                signed_in = True
        fonts = _font_versions(pages.versions)
        html, etag, csp = pages.render(
            "index.html", (public, signed_in, fonts),
            _index_transform(signed_in, fonts),
        )
        return _page_response(request, html, etag, csp, private=public)

    def _legal_page(name: str) -> Any:
        def page(request: Request) -> Response:
            fonts = _font_versions(pages.versions)
            html, etag, csp = pages.render(name, (fonts,), lambda h: _font_links(h, fonts))
            return _page_response(request, html, etag, csp, private=False)

        return page

    app.add_api_route("/terms", _legal_page("terms.html"), methods=["GET"])
    app.add_api_route("/privacy", _legal_page("privacy.html"), methods=["GET"])

    # iPhones use this icon for the home screen, favorites and share sheets,
    # and fetch it from the site ROOT whenever a page doesn't name one. Square
    # and opaque on purpose: iOS rounds the corners itself and paints
    # see-through pixels black. public.OPEN_EXACT serves both paths signed out.
    @app.get("/apple-touch-icon.png")
    @app.get("/apple-touch-icon-precomposed.png")
    def apple_touch_icon() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "brand" / "apple-touch-icon.png",
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # Browsers, bookmark managers and link unfurlers still ask for the root
    # favicon (ACC-013): the 48 px brand icon (PNG works in every browser).
    @app.get("/favicon.ico")
    def favicon() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "brand" / "favicon-48.png",
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=604800"},
        )

    # Crawl rules + sitemap (ACC-006 / BE-022): the landing and legal pages
    # are public; the app, its APIs and the private home games are not.
    _ROBOTS_DISALLOW = (
        "/games", "/trainer", "/study", "/admin", "/auth", "/billing", "/account",
        "/state", "/formats", "/me", "/health",
    )

    @app.get("/robots.txt")
    def robots(request: Request) -> Response:
        lines = ["User-agent: *"]
        lines += [f"Disallow: {p}" for p in _ROBOTS_DISALLOW]
        lines += ["Allow: /", "", f"Sitemap: {_site_base_url(request, site)}/sitemap.xml", ""]
        return Response(
            "\n".join(lines), media_type="text/plain; charset=utf-8",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/sitemap.xml")
    def sitemap(request: Request) -> Response:
        base = _site_base_url(request, site)
        rows = []
        for loc, name, prio in (
            ("/", "index.html", "1.0"),
            ("/terms", "terms.html", "0.3"),
            ("/privacy", "privacy.html", "0.3"),
        ):
            try:
                day = time.strftime("%Y-%m-%d", time.gmtime((STATIC_DIR / name).stat().st_mtime))
            except OSError:
                continue
            rows.append(
                f"  <url><loc>{base}{loc}</loc><lastmod>{day}</lastmod>"
                f"<priority>{prio}</priority></url>"
            )
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            + "\n".join(rows) + "\n</urlset>\n"
        )
        return Response(
            body, media_type="application/xml",
            headers={"Cache-Control": "public, max-age=86400"},
        )


# --- Health (OPS-019) -------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BUILD_INFO: dict[str, Any] | None = None


def _build_info() -> dict[str, Any]:
    """The deployed code: ``PLO5BP_BUILD_COMMIT``, else ``BUILD_INFO.json`` at
    the repo root (written by the deploy's pack step: {"commit", "built_at",
    ...}), else the git checkout. Read once."""
    global _BUILD_INFO
    if _BUILD_INFO is not None:
        return _BUILD_INFO
    info: dict[str, Any] = {"commit": None, "built_at": None, "source": "unknown"}
    env_commit = os.environ.get("PLO5BP_BUILD_COMMIT", "").strip()
    path = _REPO_ROOT / "BUILD_INFO.json"
    if env_commit:
        info.update(commit=env_commit, source="env")
    elif path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                info.update({k: data.get(k) for k in ("commit", "built_at", "dirty", "branch")})
                info["source"] = "file"
        except (OSError, ValueError) as e:
            logger.warning("BUILD_INFO.json unreadable: %s", e)
    elif (_REPO_ROOT / ".git").exists():
        import subprocess

        try:
            out = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True,
                text=True, timeout=3,
            )
            if out.returncode == 0:
                info.update(commit=out.stdout.strip(), source="git")
        except (OSError, subprocess.SubprocessError):
            pass
    commit = info.get("commit")
    info["commit_short"] = str(commit)[:12] if commit else None
    _BUILD_INFO = info
    return info


def health_report(site: Site | None = None) -> tuple[int, dict[str, Any]]:
    """(HTTP status, body) of ``GET /health`` for `site` (default: the current).

    The deploy check, the uptime monitor and the admin panel read these
    names: ``model_loaded`` / ``critic_loaded`` / ``obs_rev_mismatch`` (the
    PLO5 product format), ``build.commit``, ``threads`` (registered worker
    checks, e.g. the home-games clock and grader). In the public build a
    BROKEN model — a random placeholder, or one fed another observation
    revision — answers 503, so a deploy or a checkpoint swap that breaks it
    rolls back / pages instead of quietly serving garbage. A missing critic
    or an unhealthy worker is ``degraded`` (200, ``ok: false``)."""
    site = site or current_site()
    plo5 = site.formats[VARIANT_PLO5]
    model_loaded = bool(plo5.get("loaded"))
    critic_loaded = bool(plo5.get("critic_loaded", plo5.get("critic") is not None))
    mismatch = bool(plo5.get("obs_rev_mismatch", False))
    problems: list[str] = []
    if not model_loaded:
        problems.append("PLO5 model not loaded — a random placeholder is serving")
    if mismatch:
        problems.append(
            "PLO5 model trained on obs rev %s, process encodes rev %s"
            % (plo5.get("obs_rev"), _process_obs_rev())
        )
    if not critic_loaded:
        problems.append("PLO5 critic not loaded — the review's true EV is off")
    threads: dict[str, Any] = {}
    for name, check in list(site.health_checks.items()):
        try:
            threads[name] = dict(check())
        except Exception as e:  # noqa: BLE001 — a broken check is itself a finding
            threads[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        if not threads[name].get("ok", True):
            problems.append(f"{name} unhealthy")
    broken = not model_loaded or mismatch
    body = {
        "ok": not problems,
        "status": "broken" if broken else ("degraded" if problems else "ok"),
        "public": site.public,
        "model_loaded": model_loaded,
        "critic_loaded": critic_loaded,
        "obs_rev_mismatch": mismatch,
        "obs_rev": plo5.get("obs_rev"),
        "process_obs_rev": _process_obs_rev(),
        "checkpoint": plo5.get("checkpoint"),
        "sha256": (plo5.get("sha256") or "")[:16] or None,
        "build": _build_info(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(site.started_at)),
        "uptime_s": int(time.time() - site.started_at),
        "formats": {
            fid: {
                "model_loaded": bool(f.get("loaded")),
                "critic_loaded": bool(f.get("critic_loaded", f.get("critic") is not None)),
                "obs_rev_mismatch": bool(f.get("obs_rev_mismatch", False)),
                "checkpoint": f.get("checkpoint"),
            }
            for fid, f in list(site.formats.items())
            if f.get("available", True)
        },
        "threads": threads,
        "problems": problems,
    }
    return (503 if (site.public and broken) else 200), body


#: Any of these means the request crossed a proxy / the Cloudflare tunnel.
_PROXY_HEADERS = ("x-forwarded-for", "x-forwarded-host", "x-real-ip", "forwarded",
                  "cf-connecting-ip", "cf-ray", "true-client-ip")


def _loopback_request(request: Request) -> bool:
    """A request made ON the server (the deploy's own check), never one through the
    tunnel: cloudflared connects from 127.0.0.1 too, so also require no forwarding
    header and a loopback Host (the tunnel forwards the public hostname)."""
    import ipaddress
    from urllib.parse import urlsplit

    def _lo(h: str | None) -> bool:
        h = (h or "").strip().strip("[]").lower()
        if h == "localhost":
            return True
        try:
            return ipaddress.ip_address(h).is_loopback
        except ValueError:
            return False

    if any(h in request.headers for h in _PROXY_HEADERS):
        return False
    try:
        host = urlsplit("//" + request.headers.get("host", "")).hostname
    except ValueError:
        return False
    return _lo(request.client.host if request.client else "") and _lo(host)


@site_router.get("/health")
def health(request: Request) -> Response:
    status, body = health_report(getattr(request.app.state, "site", None))
    # (OPS-031) `?deploy=1` from the server itself adds what a restart would
    # interrupt, from memory (ops/deploytool.py `status`); anyone else gets plain /health.
    if request.query_params.get("deploy") == "1" and _loopback_request(request):
        hg = sys.modules.get("plo5bp.ui.homegame")
        status_fn = getattr(hg, "deploy_status", None) if hg is not None else None
        body = {**body, "deploy": {"home_games": status_fn() if callable(status_fn) else None}}
    return JSONResponse(body, status_code=status, headers={"Cache-Control": "no-store"})


def _system_info(site: Site | None = None) -> dict[str, Any]:
    """Server-side facts for the admin System panel (FEAT-024)."""
    site = site or current_site()
    status, body = health_report(site)
    return {
        "health": {**body, "http_status": status},
        "formats": [_models.entry_summary(fid, f) for fid, f in list(site.formats.items())],
        "device": str(site.device),
        "torch_threads": torch.get_num_threads(),
        "gto_host": site.gto_host is not None,
    }


def _tune_torch_threads(settings: SiteSettings) -> None:
    """(PERF-022) CPU threads per forward pass. The public site runs at most
    `common.model_slots()` model requests at once (the access layer's work
    gate); giving each ``cores // slots`` threads keeps them from
    oversubscribing the box (a 2-vCPU server: 2 requests x 1 thread instead
    of 2 x 2 fighting over 2 cores). ``PLO5BP_TORCH_THREADS`` overrides; the
    local build keeps torch's own default unless it is set."""
    if settings.torch_threads is not None:
        n = settings.torch_threads
    elif settings.public:
        n = max(1, (os.cpu_count() or 2) // _model_slots())
    else:
        return
    try:
        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
    except RuntimeError:  # interop threads can only be set before first use
        pass
    logger.info("torch CPU threads per forward: %d", torch.get_num_threads())


# --- Public service layer (auth / billing / admin / per-user state) ----------


def _install_public(app: FastAPI, site: Site) -> None:
    """The public build's service layer and home games, on `site`'s app.

    Installed last so its middleware wraps every route above. The local build
    never imports plo5bp.ui.public. The layer's state is its module's, so it
    serves one app at a time: a site it served before is closed first (its home
    games stopped, its database closed) — the new app gets its own database,
    caches and home-games context (`homegame.use_context`)."""
    global _PUBLIC_SITE
    from plo5bp.ui import homegame as _homegame
    from plo5bp.ui import public as _public

    retired = _PUBLIC_SITE
    if retired is not None and retired is not site and not retired.closed:
        logger.warning("a new public app replaces the previous one: closing the old one")
        retired.close()
    _PUBLIC_SITE = site
    site.homegames = _homegame.HomeGames()
    _homegame.use_context(site.homegames)

    def _plo5() -> dict[str, Any]:
        return site.formats[VARIANT_PLO5]

    _public.install(
        app,
        study_session_factory=Session,
        set_study_resolver=site.set_session_resolver,
        # The PLO5 model this site serves NOW (a reloaded / promoted one too).
        trainer_session_factory=lambda stats_path: _trainer.TrainerSession(
            _plo5()["model"], site.device, critic=_plo5()["critic"],
            stats_path=stats_path, formats=site.formats,
        ),
        set_trainer_resolver=site.set_trainer_resolver,
        static_dir=STATIC_DIR,
        set_format_gate=site.set_format_gate,
        system_info=lambda: _system_info(site),
        model_admin=site.model_admin,
    )
    site.db, site.registry = _public.DB, _public._REGISTRY
    # Home-games grading scores with the PLO5 model this site serves NOW (a
    # reloaded / promoted checkpoint included). HGB-016: the grader used to
    # import this whole module to borrow MODEL. (OPS-021) A random
    # placeholder is never offered: no real model = no grades, the hands stay
    # "being worked out" instead of carrying permanent garbage marks.
    _homegame.set_model_provider(lambda: _plo5()["model"] if _plo5().get("loaded") else None)


# --- The app factory ----------------------------------------------------------


def create_app(settings: SiteSettings | None = None) -> FastAPI:
    """Build the study / trainer app — and, in the public build, the service
    layer and the home games — from `settings` (default: the environment, read
    now), and make its site the current one (see `Site`). The app's state is
    ``app.state.site``. uvicorn serves the one built at import
    (``plo5bp.ui.server:app``).

    A test builds an app per configuration without re-importing anything:
    set the environment, call this, and `use_site` the previous site back (a
    public app's `Site.close()` stops its home games and closes its database).
    The layers' own settings (`public.install`, `homegame.install`) are read
    from the environment as it is at that moment."""
    settings = settings if settings is not None else SiteSettings.from_env()
    if settings.public != _env_flag("PLO5BP_PUBLIC"):
        # The trainer's public guards and the checkpoint loader read the flag
        # when they run: a site that disagrees with them would serve a mix.
        raise ValueError(
            f"SiteSettings(public={settings.public}) disagrees with PLO5BP_PUBLIC in "
            "the environment — set the environment (the trainer and the checkpoint "
            "loader read it when they run)"
        )
    site = Site(settings)
    # The served models: actor + critic from ONE read of each checkpoint.
    site.formats = {fmt: _models.build_entry(fmt, _format_ckpt_path(fmt)) for fmt in _SERVED_FORMATS}
    site.device = next(site.formats[VARIANT_PLO5]["model"].parameters()).device
    site.model_admin = _models.ModelAdmin(site.formats)
    previous = _activate(site)
    try:
        return _compose(site)
    except BaseException:
        if previous is not None:
            _activate(previous)
        raise


def _compose(site: Site) -> FastAPI:
    """The app itself, around `site` (the current site while it is built)."""
    settings = site.settings
    # (review 2026-09-20 F1) The public build ships no interactive docs / schema:
    # `/openapi.json` listed every route — the hidden home-games API and the
    # admin API included — to any signed-in free user.
    app = FastAPI(
        title="PLO5 Bomb-Pot Study Tool",
        **(
            {"docs_url": None, "redoc_url": None, "openapi_url": None}
            if settings.public
            else {}
        ),
    )
    site.app = app
    app.state.site = site
    app.add_exception_handler(RequestValidationError, _request_validation_error)
    app.add_exception_handler(_StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unhandled_error)

    if settings.gto_checkpoint:
        site.gto_host = try_load_gto_host(settings.gto_checkpoint, device=site.device)
    if site.gto_host is not None:
        logger.info("GTO PolicyNet loaded from %s (Study+Trainer T1)", settings.gto_checkpoint)
    else:
        logger.info(
            "No PLO5BP_GTO_CHECKPOINT — NLH Study uses PPO/random placeholder"
        )

    plo5 = site.formats[VARIANT_PLO5]
    site.trainer_router = _trainer.create_trainer_router(
        plo5["model"], site.device, critic=plo5["critic"], formats=site.formats,
        gto_checkpoint=settings.gto_checkpoint,
        # The ONE loaded teacher, shared (it used to be loaded twice, PERF-023).
        gto_host=site.gto_host,
    )
    app.include_router(site.trainer_router)

    # NLH range grid (/ranges/*) — LOCAL BUILD ONLY for now: the public build
    # never mounts it (like live capture) until the feature is validated and
    # deliberately shipped.
    if not settings.public:
        from plo5bp.ui.ranges import create_ranges_router

        app.include_router(
            create_ranges_router(
                site.formats,
                site.device,
                nlh_ckpt_name=_format_ckpt_path(VARIANT_NLH).name,
                gto_model=(site.gto_host.model if site.gto_host is not None else None),
                # The HOST (not just its model) carries coverage + obs-form
                # metadata: Ranges serves the teacher only where `supports()`
                # says yes and on its canonical obs, else the PPO NLH model
                # (review 2026-09-20 D3).
                gto_host=site.gto_host,
            )
        )

    _mount_flat(app, study_router)
    _mount_flat(app, site_router)
    _mount_flat(app, study_router, prefix="/study", include_in_schema=False)

    if not settings.public:
        # Build the env now so /state works on the first request — local build
        # only: the public build has no shared default session (SEC-011).
        _rebuild_env()
        # Live capture (local build only): ClubGG OCR + PokerNow ingest live in
        # `plo5bp.ui.live` (runners, hand-start machine, /ocr/* + /pokernow/*
        # routes). The public build never imports that package, so none of its
        # code ships there.
        from plo5bp.ui.live.routes import install as _install_live_capture

        _install_live_capture(app)

    if STATIC_DIR.exists():
        _install_pages(app, site)

    site.started_at = time.time()
    _tune_torch_threads(settings)

    if settings.public:
        _install_public(app, site)

    # --- Site-wide HTTP layer (outermost) --------------------------------------
    # Security headers on every response (SEC-001 / SEC-013 / FE-016) and request
    # timing / error tracking (OPS-024) — see `plo5bp.ui.middleware`. Added LAST so
    # it wraps everything, the public build's session + access layers included.
    app.add_middleware(
        _mw.SiteMiddleware,
        hsts=settings.public and settings.base_url.lower().startswith("https://"),
    )
    return app


# The module's historical names (`app`, `MODEL`, `FORMATS` …) read the current site.
sys.modules[__name__].__class__ = _ServerModule

# uvicorn (`plo5bp.ui.server:app`) and the deploy's pre-flight import this
# module: build the app from the environment now. `app` names the current
# site's — this one, in production.
create_app()
