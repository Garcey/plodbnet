# Labeled ground-truth frames (tracked drop-in)

`fixtures/labels.json` labels five ClubGG frames by their original
repo-relative paths under `screenrecords/frames/` — a **gitignored,
local-only** directory. Those five PNGs (`frame_0060/0300/0840/1050/
1440.png`) were lost in a 2026-06-26 disk cleanup (not in git, no
backup), which orphaned the labels: `test_suit_accuracy_on_revealed_cards`
and `test_rank_accuracy_on_revealed_cards` now SKIP for lack of data.

To re-arm them, drop frames **into this directory** (it is tracked, so
they stay durable): the conftest `resolve_frame` helper falls back here
by basename when the `screenrecords/` path is missing.

- If the original five frames ever resurface, copy them here unchanged.
- For fresh ground truth: during a live ClubGG session run
  `POST /ocr/save_frame` with `{"to_fixtures": true}` — it writes
  `frame_<epoch>.png` straight into THIS directory plus
  `frame_<epoch>.state.json`, the `FrameState` the extractor read from it.
  Check the JSON against the screenshot (fix anything the extractor got
  wrong — the JSON is the expected answer), then label cards with
  `plo5bp.ocr.tools.label_cards` and append entries to `labels.json`
  (any `"frame"` path works — basename is what matters here). Without
  `to_fixtures` the frame lands in gitignored `screenrecords/` instead.

Golden frames also feed the chip-amount digit reader (TOOL-033): with a few
reviewed frames here (between them showing every digit 0-9 in a stack, bet or
pot), run

    .venv/Scripts/python -m plo5bp.ocr.tools.harvest_digits

It cuts every stack / bet / pot amount out of the frames, pairs each glyph
with the amount in the `.state.json`, prints a per-character count and a
self-check, and writes `python/plo5bp/ocr/templates/digits.npz` once all ten
digits are covered. From then on amounts are read in-process; Tesseract stays
the fallback for anything the reader is unsure of.
