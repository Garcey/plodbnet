---
name: reference-tesseract-psm-for-clubgg
description: Use Tesseract --psm 8 (single word), not --psm 7 (single line), for ClubGG chip-amount crops; PSM 7's line-segmentation heuristic misreads short 2-digit cyan stacks like "74"→"14"
metadata:
  type: reference
---

`python/plo5bp/ocr/text.py:_TESSERACT_CFG` uses `--psm 8` (single word).
Earlier it used `--psm 7` (single text line), which silently misread
short 2-digit cyan stacks: "74" came back as "14" (and only on the
no-decimal hero plate; "72.25", "118.34" etc. read fine). PSM 8 fixes
"74" and does not regress any of the decimal / 3-digit / pot / commit
crops.

PSM 13 (raw line, bypass tesseract heuristics) also works on this set,
but PSM 8 is semantically truer to the crops: every stack/pot/commit
ROI contains exactly one short word.

**Diagnostic recipe** when a future "$X displays as $Y" misread shows
up: sweep PSMs 6/7/8/13 on the offending crop with a fixed scale and
threshold; pick the PSM that matches all on-screen reads without
regressing the others.

See also [[reference-tesseract-install-windows]].
