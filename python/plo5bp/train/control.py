"""The live control file: retune a running trainer without a restart.

(Its name, `anneal_control`, is from the retired block-rotation entropy anneal,
whose F/T/R-driven decisions also lived here until 2026-09-28, ML-030.)"""

from __future__ import annotations

import json
import math
from pathlib import Path


# ---- live control file hardening (review 2026-09-20 A13 / A20) -------------
# One message per DISTINCT problem: the control file is re-read every update,
# so anything keyed on its content would otherwise spam the log forever.
_CONTROL_WARNED: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _CONTROL_WARNED:
        _CONTROL_WARNED.add(msg)
        print(msg)


def _read_control_text(path: Path) -> str | None:
    """Text of a live-control file (anneal_control.json / threads.txt), or
    None when it cannot be read. NEVER raises: the read sites used to catch
    only OSError, so a UTF-16 file (what Windows PowerShell 5.1 `>` /
    Out-File writes) raised UnicodeDecodeError — a ValueError — killed the
    run, and crash-looped every guardian relaunch on the same file.
    `utf-8-sig` also strips a UTF-8 BOM, which used to make the JSON
    unparseable and the file silently ignored forever."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, ValueError) as e:  # UnicodeDecodeError is a ValueError
        _warn_once(
            f"[control] cannot read {path}: {e!r} — IGNORED (save it as "
            "UTF-8; PowerShell 5.1 `>`/Out-File writes UTF-16)"
        )
        return None


# key -> (lo, hi, lo_is_exclusive). Outside the range = a typo, not a tuning
# choice (`true` -> lr 1.0, a dropped exponent, a sign slip): reject it.
_CONTROL_BOUNDS: dict[str, tuple[float, float, bool]] = {
    "target_kl": (0.0, 1e6, False),            # 0 = guard off
    "kl_hard": (0.0, 1e6, False),              # 0 = guard off
    "lr": (0.0, 0.1, True),
    "sizing_entropy_scale": (0.0, 1e3, False),
    "entropy_coef": (0.0, 10.0, False),
    "entropy_coef_deep": (0.0, 10.0, False),
    "clip_room_mid": (0.0, 1.0, True),
    "clip_room_ext": (0.0, 1.0, True),
    "q_fold_sup_coef": (0.0, 1e6, False),
}
_TIER_ENT_BOUNDS = (0.0, 10.0, False)


def _control_number(
    key: str, value: object, bounds: tuple[float, float, bool]
) -> tuple[float | None, str | None]:
    """(validated float, None) or (None, why-not). JSON `true` is a Python
    bool — an int subclass, so float(True) == 1.0 used to become lr 1.0."""
    lo, hi, lo_open = bounds
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"{key}={value!r} is not a number"
    try:
        x = float(value)
    except OverflowError:  # a 400-digit int literal
        return None, f"{key} overflows a float"
    if not math.isfinite(x):
        return None, f"{key}={x} is not finite"
    if x < lo or x > hi or (lo_open and x == lo):
        rng = f"{'(' if lo_open else '['}{lo:g}, {hi:g}]"
        return None, f"{key}={x:g} is outside {rng}"
    return x, None


def _apply_anneal_control(
    raw: str | None,
    last_raw: str | None,
    tier_ent: dict[str, float],
    live_lr: float,
    live_ent: float,
    live_ent_deep: float,
    trainer=None,
    broadcast_entropy: bool = False,
) -> tuple[str | None, float, float, float]:
    """Apply a live control-file edit (`<run-dir>/<stem>.control.json`)
    without pausing training. Returns (applied_content, live_lr, live_ent,
    live_ent_deep); mutates `tier_ent` in place (and the trainer's live knobs
    through `PPOTrainer.apply_live_control`, when given). Read for EVERY run.
    Re-applies only when the file CONTENT changes:

      {"tier_ent": {"deep": 0.08}}        — set a tier's coef (--mix-configs)
      {"entropy_coef": 0.38}              — FLAT coef (non-tier runs: NLH /
                                            plain --stack-dist). Under
                                            --mix-configs
                                            (`broadcast_entropy`) it sets
                                            EVERY tier not named by a
                                            `tier_ent` in the same write
      {"entropy_coef_deep": 0.1}          — the deep-dist flat variant
      {"target_kl": 2.0}                  — retune the soft KL early-stop
      {"kl_hard": 12.0}                   — retune the hard rollback level
      {"lr": 1e-4}                        — retune the base learning rate
      {"sizing_entropy_scale": 2.5}       — scale the sizing-head entropy
      {"clip_room_mid": 0.07}             — prob-dependent clip: mid-band
                                            probability room (the per-update
                                            policy quota at ~50/50 gates)
      {"clip_room_ext": 0.12}             — same, the rare-gate ends of the U
      {"q_fold_sup_coef": 15.0}           — fold-column supervision weight
                                            inside the q-aux loss (qF canary)
      {"lr": 1e-4, "tier_ent": {...}}     — any combination

    The lr set is the BASE lr — the per-update warmup scale still multiplies
    it. (`step`, the retired auto-anneal's decrement, is now an unknown key:
    logged once and skipped.)

    An edit is applied ATOMICALLY or not at all, and never silently
    (review 2026-09-20 A13 / A20): malformed JSON / a non-object / a
    non-object `tier_ent` / any value that is a bool, non-numeric, negative,
    non-finite or out of `_CONTROL_BOUNDS` makes the WHOLE edit a no-op
    (returned content unchanged, so a half-written save is simply retried
    on the next loop) with ONE log line per distinct content. Unknown keys
    and unknown tier names are logged once and skipped; the rest applies."""
    if raw is None or raw == last_raw:
        return last_raw, live_lr, live_ent, live_ent_deep
    unchanged = (last_raw, live_lr, live_ent, live_ent_deep)
    snippet = raw.strip().replace("\n", " ")[:160]
    try:
        ctrl = json.loads(raw)
    except ValueError:
        _warn_once(f"[anneal-control] malformed JSON IGNORED: {snippet}")
        return unchanged
    tiers_raw = ctrl.get("tier_ent") if isinstance(ctrl, dict) else None
    if not isinstance(ctrl, dict) or not isinstance(tiers_raw, (dict, type(None))):
        # C1: valid JSON but not an object (a list, bare string, or number
        # from a live-tune typo), or a non-object tier_ent. Treat as
        # malformed and ignore, per the docstring.
        _warn_once(f"[anneal-control] not a JSON object — IGNORED: {snippet}")
        return unchanged

    errors: list[str] = []
    vals: dict[str, float] = {}
    for key, bounds in _CONTROL_BOUNDS.items():
        if key in ctrl:
            x, why = _control_number(key, ctrl[key], bounds)
            if why is not None:
                errors.append(why)
            else:
                vals[key] = x
    new_tiers: dict[str, float] = {}
    unknown_tiers: list[str] = []
    for tier, v in (tiers_raw or {}).items():
        if tier not in tier_ent:
            unknown_tiers.append(str(tier))
            continue
        x, why = _control_number(f"tier_ent[{tier}]", v, _TIER_ENT_BOUNDS)
        if why is not None:
            errors.append(why)
        else:
            new_tiers[tier] = x
    if errors:
        _warn_once(
            "[anneal-control] edit IGNORED, NOTHING applied — "
            + "; ".join(errors) + f" | content: {snippet}"
        )
        return unchanged
    unknown_keys = sorted(set(ctrl) - set(_CONTROL_BOUNDS) - {"tier_ent"})
    if unknown_keys:
        _warn_once(
            f"[anneal-control] unknown key(s) {unknown_keys} skipped "
            f"(valid: {sorted(_CONTROL_BOUNDS) + ['tier_ent']})"
        )
    if unknown_tiers:
        _warn_once(
            f"[anneal-control] unknown tier_ent tier(s) {sorted(unknown_tiers)} "
            f"skipped (this run's tiers: {sorted(tier_ent)})"
        )

    new_lr = vals.get("lr")
    new_ent = vals.get("entropy_coef")
    new_ent_deep = vals.get("entropy_coef_deep")
    for tier, v in new_tiers.items():
        if tier_ent[tier] != v:
            print(f"[anneal-control] tier_ent[{tier}] {tier_ent[tier]} -> {v}")
        tier_ent[tier] = v
    # Mix-configs consumes tier_ent (per-row coefs), not the flat coef — an
    # `entropy_coef` edit used to be a silent no-op there (V5_DESIGN.md B5).
    # Broadcast it to every tier so the natural key works in both modes; a
    # tier NAMED by `tier_ent` in this same write keeps its explicit value.
    # Keyed on the key's PRESENCE, not on the flat value changing (A20): the
    # flat coef goes stale under mixing (nothing reads it), so re-sending the
    # launch value to undo per-tier edits compared equal and did nothing.
    if broadcast_entropy and new_ent is not None:
        for tier in tier_ent:
            if tier in new_tiers:
                continue
            if tier_ent[tier] != new_ent:
                print(
                    f"[anneal-control] tier_ent[{tier}] {tier_ent[tier]} "
                    f"-> {new_ent} (entropy_coef broadcast)"
                )
            tier_ent[tier] = new_ent
    # The trainer's live-tunable knobs (PPOTrainer.LIVE_KEYS: target_kl,
    # kl_hard, sizing_entropy_scale, the prob-dependent clip rooms, the
    # fold-supervision weight) are read per minibatch, so setting them
    # retunes the next update without a restart (2026-07-11: the clip's mid
    # band IS the per-update KL ceiling at mixed gates; the qF canary reads
    # the fold weight). The clip rooms only matter with clip_prob_dependent.
    if trainer is not None:
        live = {k: vals[k] for k in trainer.LIVE_KEYS if k in vals}
        for key, (was, now) in trainer.apply_live_control(**live).items():
            print(f"[anneal-control] {key} {was} -> {now}")
    out_lr = live_lr
    if new_lr is not None:
        if live_lr != new_lr:
            print(f"[anneal-control] lr {live_lr} -> {new_lr}")
        out_lr = new_lr
    out_ent = live_ent
    if new_ent is not None:
        if live_ent != new_ent:
            print(f"[anneal-control] entropy_coef {live_ent} -> {new_ent}")
        out_ent = new_ent
    out_ent_deep = live_ent_deep
    if new_ent_deep is not None:
        if live_ent_deep != new_ent_deep:
            print(
                f"[anneal-control] entropy_coef_deep {live_ent_deep} -> {new_ent_deep}"
            )
        out_ent_deep = new_ent_deep
    return raw, out_lr, out_ent, out_ent_deep
