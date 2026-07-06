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
- For fresh ground truth: capture frames via `POST /ocr/save_frame`
  during a live ClubGG session, label them with
  `plo5bp.ocr.tools.label_cards`, append entries to `labels.json`
  (any `"frame"` path works — basename is what matters here), and put
  the PNGs in this directory rather than `screenrecords/`.
