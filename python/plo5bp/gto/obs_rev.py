"""Observation SEMANTICS revision for the GTO pipeline.

The NLH encoder's VALUES changed on 2026-09-20 (dims 134/135 min/max bet now
describe the legal raise window; dim 906 flush-nut distance on five-flush
boards). ``plo5bp.encoding.OBS_SEMANTICS_REV`` (env ``PLO5BP_OBS_REV``:
unset / ``2`` = fixed, ``1`` = legacy) says which semantics this process
encodes. Anything that EMBEDS an obs or was FIT to one is tied to a revision:

- cached supervised datasets (``dataset.save_rows_npz``) store ``obs_rev`` and
  ``load_rows_npz`` refuses a bundle from another revision — a stale cache is
  rebuilt, never silently reused;
- PolicyNet checkpoints are stamped ``obs_rev`` at train time and the loader /
  ``PolicyNetHost`` warn loudly when it differs from the serving process.

An artifact with NO stamp predates the switch, i.e. revision 1.
LabelRecord JSONL carries no obs (it is re-synthesized at load), so labels are
revision-independent.
"""

from __future__ import annotations

from typing import Any, Mapping

import plo5bp.encoding as _encoding

#: Revision of every artifact written before the stamp existed.
UNSTAMPED_OBS_REV = 1


def current_obs_rev() -> int:
    """Revision this process encodes (read late: the switch is env-driven)."""
    return int(getattr(_encoding, "OBS_SEMANTICS_REV", 2))


def stamped_obs_rev(meta: Mapping[str, Any] | None) -> int:
    """Revision an artifact was built under (absent ⇒ ``UNSTAMPED_OBS_REV``)."""
    rev = (meta or {}).get("obs_rev")
    try:
        return UNSTAMPED_OBS_REV if rev is None else int(rev)
    except (TypeError, ValueError):
        return UNSTAMPED_OBS_REV


def obs_rev_mismatch(meta: Mapping[str, Any] | None) -> str | None:
    """Human-readable mismatch between an artifact and this process, or None."""
    have, want = stamped_obs_rev(meta), current_obs_rev()
    if have == want:
        return None
    return (
        f"observation semantics revision mismatch: artifact obs_rev={have}, "
        f"this process encodes obs_rev={want} (PLO5BP_OBS_REV). The inputs the "
        f"net was fit to are NOT the inputs it is being served — retrain, or "
        f"run with PLO5BP_OBS_REV={have} to serve it exactly as trained."
    )
