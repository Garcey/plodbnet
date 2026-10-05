"""The network at the home games (2026-10-05; owner: "integrate the model into home games
... This is something only I get access to do").

The site's owner can let the served PLO5 network play their seat — ``auto``: it bets,
calls and folds for them — or only suggest each move — ``assist``: their action button
lights up and its size is set, they still press it — and choose whether it plays its
favourite move every time or its full strategy (``mix``: an action drawn from its mix,
the way it was trained). Everybody at the table can see it: a chip on the seat
(``seat.bot``), a line in the table's feed when its mode changes — suggesting or playing,
never the strategy (owner: "more information than necessary") — and the hand history marks
every decision it made or suggested (``action.bot``). Those decisions are never graded —
the network would be grading itself — so the player's accuracy stays their own.

How it decides: the hand so far is replayed in a FULL-observation env exactly like the
grader's (``grade_hand``: the dealt deck, the hand's start stacks, the actions so far,
raises clamped into the bet rule it was trained on) and the node goes through the
Trainer's ``compute_node_distribution``. The observation is the actor's own — their cards,
the boards, the betting; its Monte-Carlo inputs draw the unknown cards from everything
the actor can't see — the same Study and the grader use. When it is played, the raise is
clamped into the table's own window (``_apply_action_locked``).

Autopilot runs in the clock thread: ``_bot_tick_locked`` (a watchdog step) hands the
decision to the bot worker (one thread, ``_bot_loop``: off every table's lock) and plays
it once ``BOT_DELAY_S`` has passed since the turn began; the owner's own view offers no
buttons meanwhile (``_action_view``). Assist asks for the same decision through ``GET
…/bot/suggest`` — one per decision (``bot_cache``), so a reload never draws a new sample.
PLO5 tables only (the network plays PLO5); settings are in memory, so a reloaded table
comes back with the network off.

Split out of ``homegame`` like the other parts (HGB-006): every name defined here is
re-exported as ``homegame.<name>`` and the code reaches every home-games name through
``hg``, looked up when it runs.
"""

from __future__ import annotations

import importlib
import logging
import queue
import secrets
import sys
import threading
import time
from typing import Any, TYPE_CHECKING

from fastapi import HTTPException

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.config import VARIANT_PLO5, GameConfig
from plo5bp.env import BombPotEnv
from plo5bp.ui import public as pub

if TYPE_CHECKING:  # (annotations only)
    from plo5bp.ui.homegame import LiveTable

#: The home games' main module: every home-games name is looked up there when used.
hg = sys.modules.get("plo5bp.ui.homegame") or importlib.import_module("plo5bp.ui.homegame")
logger = logging.getLogger("plo5bp.ui.homegame")

__all__ = (
    "BOT_DELAY_S", "BOT_MODES", "BOT_QUEUE_MAX", "OWNER_TTL_S", "_GATE_NAMES", "_bot_answer", "_bot_decide", "_bot_job_locked",
    "_bot_loop", "_bot_play_locked", "_bot_queue", "_bot_ready", "_bot_replay", "_bot_suggestion",
    "_bot_tick_locked", "_bot_view", "_is_owner", "_set_bot_locked", "_start_bot_worker",
)

#: "" = off, "assist" = it shows its move, "auto" = it plays.
BOT_MODES = ("", "assist", "auto")
#: Autopilot acts this long after the turn began (an instant move is hard to follow).
BOT_DELAY_S = 1.2
#: Decisions waiting for the bot worker, site-wide (more = that turn waits for the clock).
BOT_QUEUE_MAX = 64
#: How long "is this account the site's owner" is kept per account (the view asks on
#: every push to every viewer; the owner list is the server's settings).
OWNER_TTL_S = 60.0
#: The wire's names of the three actions.
_GATE_NAMES = {GATE_FOLD: "fold", GATE_CHECK_CALL: "check_call", GATE_RAISE: "raise"}


def _is_owner(uid: int) -> bool:
    """The site's owner (an admin): the only account the network plays for. From the
    sign-in check's user cache, else the database; kept ``OWNER_TTL_S`` per account."""
    uid = int(uid)
    now = time.monotonic()
    hit = hg.CTX.owner_cache.get(uid)
    if hit is not None and hit[0] > now:
        return hit[1]
    user = pub._cached_user(uid)
    if user is pub._MISS:  # (a live stream's viewer: the cache entry of their last request ran out)
        user = pub._user_by_id(uid)
    yes = bool(pub._is_admin(user))
    hg.CTX.owner_cache[uid] = (now + hg.OWNER_TTL_S, yes)
    return yes


def _bot_ready(t: LiveTable, uid: int) -> str | None:
    """Why the network can't take ``uid``'s seat at ``t`` (None: it can)."""
    if not hg._is_owner(uid):
        return "Only the site's owner can let the network play."
    if hg._norm_game(t.variant) != "plo5":
        return "The network plays PLO5 only."
    if hg._grading_model() is None:
        return "The network isn't loaded on the server right now."
    return None


def _bot_view(t: LiveTable, viewer_id: int) -> dict[str, Any] | None:
    """The viewer's own network settings for the table view — the site's owner only
    (None for everybody else: the others see ``seat.bot``)."""
    if not hg._is_owner(viewer_id):
        return None
    p = t.player(int(viewer_id))
    why = hg._bot_ready(t, viewer_id)
    return {"available": why is None, "why": why,
            "mode": p.bot_mode if p is not None else "", "mix": bool(p.bot_mix) if p is not None else False}


def _set_bot_locked(t: LiveTable, uid: int, mode: Any, mix: bool) -> None:
    """The owner switches the network at their seat: ``mode`` "off" / "assist" / "auto",
    ``mix`` its full strategy. Said to the table whenever it changes."""
    hg._require_open(t)
    mode = str(mode or "").strip().lower()
    mode = "" if mode in ("off", "none", "") else mode
    if mode not in hg.BOT_MODES:
        raise HTTPException(status_code=400, detail="mode must be off, assist or auto")
    i = t.seat_of(int(uid))
    if i is None:
        raise HTTPException(status_code=400, detail="Take a seat first.")
    if not hg._is_owner(uid):
        raise HTTPException(status_code=403, detail="Only the site's owner can let the network play.")
    if mode:
        why = hg._bot_ready(t, uid)
        if why is not None:
            raise HTTPException(status_code=400, detail=why)
    p = t.seats[i]
    before = (p.bot_mode, p.bot_mix)
    p.bot_mode, p.bot_mix = mode, bool(mix)
    if (p.bot_mode, p.bot_mix) == before:
        return
    # (a move drawn under the old setting is not the new one's)
    t.bot_cache.clear()
    t.bot_asked.clear()
    t.rev += 1
    if p.bot_mode == before[0]:
        # (only how it plays changed: the table is told the mode alone — suggesting or
        # playing — never the strategy; owner, 2026-10-05: "more information than necessary")
        return
    if mode == "auto":
        text = f"The network is playing {p.name}'s seat."
    elif mode == "assist":
        text = f"The network is suggesting {p.name}'s moves."
    else:
        text = f"{p.name} is playing on their own again — the network is off."
    hg._emit(t, "bot", text)


def _bot_job_locked(t: LiveTable) -> dict[str, Any]:
    """The live hand so far, as the grader's replay job (plain values: the worker never
    touches the live table), with the seat to act and the decision it is."""
    return {
        "deck": list(t.hand_deck or []), "seed": int(t.hand_seed or 0),
        "button": int(t.button), "num_seats": int(t.num_seats),
        "stacks": [int(x) for x in (t.hand_start_stacks or [])], "ante": int(t.ante_chips),
        "mask": [bool(x) for x in (t.in_hand_mask or [])],
        "actions": [list(a) for a in t.hand_actions],
        "seat": int(t.env.current_actor()), "key": (int(t.hand_no), int(t.action_seq)),
    }


def _bot_replay(job: dict[str, Any]) -> tuple[Any, Any, Any]:
    """``(env, obs, info)`` at the job's decision: the grader's replay (``grade_hand``) —
    a full observation, the trained bet rule, a bigger table raise replayed as the
    trained rule's biggest (its excess came back uncalled)."""
    cfg = GameConfig(
        num_seats=job["num_seats"], starting_stack=0, starting_stacks=tuple(job["stacks"]),
        ante=job["ante"], bb=hg.BB_CHIPS, variant=VARIANT_PLO5, cover_short_bets=True,
    )
    env = BombPotEnv(cfg, ev_runout_samples=0)  # FULL observation (not "minimal")
    if job.get("deck"):
        obs, info = env.reset_with_deck(job["deck"], job["button"], in_hand_mask=job["mask"])
    else:
        obs, info = env.reset(job["seed"], job["button"], in_hand_mask=job["mask"])
    for k, (seat, gate, chips, _own) in enumerate(job["actions"]):
        if env.is_terminal() or info is None or info.actor is None or int(info.actor) != int(seat):
            raise RuntimeError(f"replay diverged at action {k}")
        chips = int(chips)
        if int(gate) == GATE_RAISE and int(info.max_raise_chips) > 0:
            chips = max(int(info.min_raise_chips), min(chips, int(info.max_raise_chips)))
        obs, _r, _done, info = env.step_hybrid(int(gate), chips)
    if env.is_terminal() or info is None or info.actor is None or int(info.actor) != int(job["seat"]):
        raise RuntimeError("replay diverged at the decision")
    return env, obs, info


def _bot_decide(job: dict[str, Any], model: Any, mix: bool) -> dict[str, Any]:
    """The network's move at the job's decision (no table, no lock): its favourite —
    the Trainer's recommendation — or, ``mix``, one drawn from its full strategy (a
    fresh unpredictable seed; the global torch RNG is the Trainer's, hence its lock)."""
    import torch

    from plo5bp.eval import model_policy
    from plo5bp.ui import trainer as tr

    _env, obs, info = hg._bot_replay(job)
    dist = tr.compute_node_distribution(model, next(model.parameters()).device, obs, info)
    probs = [float(p) for p in dist["gate_probs"]]
    if mix:
        with tr._TORCH_RNG_LOCK:
            torch.manual_seed(secrets.randbits(63))
            gate, chips = model_policy(model, deterministic=False)(obs, int(info.actor), info)
    else:
        gate, chips = int(dist["rec_gate"]), int(dist["rec_chips"])
    return {"key": tuple(job["key"]), "seat": int(job["seat"]), "gate": int(gate),
            "chips": int(chips) if int(gate) == GATE_RAISE else 0, "probs": probs, "mix": bool(mix)}


def _bot_answer(t: LiveTable, dec: dict[str, Any]) -> dict[str, Any]:
    """A decision as the owner's dock shows it: the action, the raise as a raise-TO in
    the table's chips (their dock's unit), and how often the network takes each action."""
    raw = hg._obs_dict(t.env)
    out = {
        "hand_no": int(dec["key"][0]), "action_seq": int(dec["key"][1]),
        "gate": hg._GATE_NAMES[int(dec["gate"])], "raise_to_chips": None, "mix": bool(dec["mix"]),
        "probs": {"fold": round(dec["probs"][0], 4), "check_call": round(dec["probs"][1], 4),
                  "raise": round(dec["probs"][2], 4)},
    }
    if int(dec["gate"]) == GATE_RAISE:
        out["raise_to_chips"] = int(raw["street_commit"][int(dec["seat"])]) + int(dec["chips"])
    return out


def _bot_suggestion(t: LiveTable, uid: int) -> dict[str, Any]:
    """The network's move for ``uid``, whose turn it is at a seat it assists (or plays):
    one per decision — asked again, the same answer (a full-strategy draw is never
    redrawn). Worked out off the table's lock; 409 when the decision moved on meanwhile."""
    if not hg._is_owner(uid):
        raise HTTPException(status_code=403, detail="Only the site's owner can let the network play.")
    with t.lock:
        i = t.seat_of(int(uid))
        p = t.seats[i] if i is not None else None
        why = hg._bot_ready(t, uid)
        if why is not None:
            raise HTTPException(status_code=400, detail=why)
        if p is None or not p.bot_mode:
            raise HTTPException(status_code=400, detail="The network isn't on at your seat.")
        if t.phase != "in_hand" or t.env is None or t.env.current_actor() != i:
            raise HTTPException(status_code=409, detail="not your turn")
        key = (int(t.hand_no), int(t.action_seq))
        dec = t.bot_cache.get(key)
        if dec is not None and "gate" in dec:
            return hg._bot_answer(t, dec)
        job, mix = hg._bot_job_locked(t), bool(p.bot_mix)
    dec = hg._bot_decide(job, hg._grading_model(), mix)
    with t.lock:
        if (int(t.hand_no), int(t.action_seq)) != key or t.env is None:
            raise HTTPException(status_code=409, detail="the hand moved on")
        have = t.bot_cache.get(key)
        if have is not None and "gate" in have:
            dec = have  # (worked out twice at once: the table keeps ONE)
        else:
            t.bot_cache[key] = dec
        return hg._bot_answer(t, dec)


def _bot_tick_locked(t: LiveTable) -> None:
    """(a watchdog step) The turn of a seat the network plays: hand the decision to the
    bot worker, then play it once ``BOT_DELAY_S`` has passed since the turn began. If it
    can't be worked out, the seat checks or folds — a clock-less table never waits on it."""
    if t.phase != "in_hand" or t.env is None or t.info is None:
        return
    actor = t.env.current_actor()
    if actor is None:
        return
    actor = int(actor)
    p = t.seats[actor] if actor < len(t.seats) else None
    if p is None or p.bot_mode != "auto" or hg._actor_is_away(t, actor):
        return
    key = (int(t.hand_no), int(t.action_seq))
    now = time.monotonic()
    # (counted from the turn's start: switched on late in a turn, it plays at once — the
    # clock never runs out on it)
    began = t.turn_started_mono if t.turn_key == key and t.turn_started_mono is not None else now
    asked = t.bot_asked.setdefault(key, min(began, now))
    dec = t.bot_cache.get(key)
    if dec is None:
        if hg._grading_model() is None:
            t.bot_cache[key] = {"error": "no network"}
            return
        t.bot_cache[key] = {"pending": True}
        hg._bot_queue(t, hg._bot_job_locked(t), bool(p.bot_mix))
        return
    if dec.get("pending") or now - asked < hg.BOT_DELAY_S:
        return
    hg._bot_play_locked(t, dec)


def _bot_play_locked(t: LiveTable, dec: dict[str, Any]) -> None:
    """Play the network's move for the seat to act (marked, never graded)."""
    actor = int(t.env.current_actor())
    if "gate" in dec and bool(t.info.gate_mask[int(dec["gate"])]):
        gate, chips = int(dec["gate"]), int(dec.get("chips") or 0)
    else:
        # (it couldn't decide — or the table forbids what the trained rule allowed, which
        # it never does: the table's rule only allows more) check if free, else fold
        gate, chips = hg._passive_choice(t)
        logger.warning("homegame network: table %s seat %s checks / folds (%s)",
                       t.game_id, actor, dec.get("error") or "an action the table refused")
    idx = len(t.hand_actions)
    t.bot_marks[idx] = "auto"
    try:
        hg._apply_action_locked(t, gate, chips)  # (not by_player: never graded)
    except BaseException:
        t.bot_marks.pop(idx, None)
        raise
    p = t.seats[actor]
    if p is not None:
        p.timeouts = 0


def _bot_queue(t: LiveTable, job: dict[str, Any], mix: bool) -> None:
    """Hand a decision to the bot worker (a full queue: the turn falls to check / fold)."""
    if hg.CTX.bot_q.qsize() >= hg.BOT_QUEUE_MAX:
        t.bot_cache[tuple(job["key"])] = {"error": "the network is busy"}
        return
    hg.CTX.bot_q.put((t, job, mix))
    hg._start_bot_worker()


def _start_bot_worker() -> None:
    th = hg.CTX.bot_thread
    if th is not None and th.is_alive():
        return
    hg.CTX.bot_stop.clear()
    hg.CTX.bot_thread = threading.Thread(target=hg._bot_loop, args=(hg.CTX,), name="homegame-network", daemon=True)
    hg.CTX.bot_thread.start()


def _bot_loop(ctx: Any = None) -> None:
    """The bot worker: works out each queued decision off the table's lock, then leaves
    it for the clock thread to play (if the table is still on that decision)."""
    ctx = ctx or hg.CTX  # (a swapped context never steals a running worker)
    while not ctx.bot_stop.is_set():
        try:
            item = ctx.bot_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            return
        t, job, mix = item
        try:
            model = hg._grading_model()
            if model is None:
                raise RuntimeError("no network is loaded")
            dec = hg._bot_decide(job, model, mix)
        except Exception as e:  # noqa: BLE001 — the turn falls to check / fold
            logger.exception("homegame network: table %s, decision %s", t.game_id, job.get("key"))
            dec = {"error": str(e)[:200]}
        key = tuple(job["key"])
        with t.lock:
            have = t.bot_cache.get(key)
            if have is not None and have.get("pending"):
                t.bot_cache[key] = dec
