---
name: Per-seat commits + last aggressor + hero-button distance added
description: 2026-05-06 production obs change — +32 dims (8 each: total commit, street commit, last aggressor, hero-btn distance), all hero-rotated. OBS_DIM 886→918. Triggers retrain.
type: project
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
On 2026-05-06, immediately after the seat-exists mask
(`project_seat_exists_mask.md`), four hero-rotated 8-dim blocks were
appended at the encoder tail. All four close gaps the previous
representation either reduced away or never surfaced.

```
886..894  total_commit / cfg.bb (per seat, hero-rotated, raw scalar)
894..902  street_commit / cfg.bb (per seat, hero-rotated, raw scalar)
902..910  last_aggressor hero-rel one-hot (all-zero if None)
910..918  hero distance to button: one-hot of (button - hero) % num_seats
OBS_DIM = 918
```

**What each closes.**
- *Per-seat hand-total commit.* Pot/bb already captures total chips
  in the middle, but per-seat exposure (who is pot-committed, who is
  not) was invisible. Reconstructible from the 32-slot history block,
  but expensive to integrate from history; surfacing it explicitly
  removes the burden.
- *Per-seat street commit.* Hero's own street-commit leaks via the
  pot-odds scalar; opponents' did not. Now visible per seat.
- *Last aggressor.* Initiative was implicit — derivable from history
  by finding the last Raise — but unreduced. Hero-relative one-hot;
  all-zero when no raise has happened yet on this street/hand.
- *Hero distance to button.* Previous layout encoded `actor - hero`
  via `_REL_POS_OFF` (always 0 since obs is encoded from actor's POV),
  and `seat_exists` covers the seat count, but `(button - hero) %
  num_seats` was nowhere. Closes the positional gap.

**Implementation.**
- `rust_engine/src/bindings.rs`: exposes `last_aggressor` in scalar
  `observation_dict` (i64; `-1` sentinel = None) and in both batched
  `observation_arrays()` and `observation_and_features_batch()`
  emitters (i8). `PackedObservation` struct + `pack_observation`
  init/populate/return all updated. `total_commit` and `button` were
  already exposed.
- `python/plo5bp/encoding.py`: scalar path reuses the `street_commit`
  local already loaded for pot-odds. Batched path reuses the same
  `rot` index built for the active/all-in/stacks blocks.

**How to apply.**
- Production observation-semantics change. Retrain from scratch on
  the new 918-dim encoder; do not warm-start from any pre-2026-05-06
  checkpoint (none exist on disk by design).
- Three of the four features are technically reconstructible from
  history + button signal, but reduction was on the network's
  shoulders. Surfacing them as dedicated blocks removes per-step
  reconstruction effort.
- Last-aggressor signal becomes more informative once heterogeneous
  stacks ship — short-shove dynamics depend on who reopened betting
  vs who didn't.
