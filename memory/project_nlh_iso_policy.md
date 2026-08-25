# NLH teacher iso policy (v1)

**Decision (2026-08-12):** teacher solves use **`use_isomorphism=false`**.
Export writes **`raw_combo` as `hero_hole`**. Serve encodes **raw 52-hot**.
Do **not** train on iso-canonical cards.

## Why

CFR iso remaps hole suits to a board-canonical permutation (`iso_combo_id`).
If labels decode that id as the hole, PolicyNet sees AcKd-on-clubs-board
while live play encodes AdKh. Same strategy, different obs — silent poison.

Expanding every suit orbit at export is the other consistent option. v1
turns iso **off** instead: one row per dealt combo, serve path unchanged.

## Code

- Constant: `plo5bp.gto.iso.TEACHER_USE_ISOMORPHISM = False`
- Teacher configs: `SolveConfig.teacher()` / `apply_teacher_iso_policy`
- Batch: `expand_river_grid` + `run_batch` force the flag
- Overnight blueprints: `SolveConfig.teacher(...)`
- Export: never treat `iso_id` as hole; `iso_id` set and `raw_combo` missing
  → DROP `iso_without_raw`

Desktop / interactive solves may still set `use_isomorphism=True` (memory).
Those dumps are not teacher labels unless they carry `raw_combo`.

## Serve

`encode_observation_nlh` stays raw hole multi-hot. No suit map at inference.
