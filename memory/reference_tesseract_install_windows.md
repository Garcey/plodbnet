---
name: reference-tesseract-install-windows
description: Tesseract binary install command on Windows for plodbbot's OCR text path; pip extras don't include it
metadata:
  type: reference
---

`pip install -e ".[ocr]"` installs `pytesseract` (Python wrapper) but NOT
the Tesseract binary it shells out to. On Windows, install the binary
separately:

```powershell
winget install --id UB-Mannheim.TesseractOCR --accept-source-agreements --accept-package-agreements --silent
```

Installs to `C:\Program Files\Tesseract-OCR\tesseract.exe`, which is
the first entry in `_TESSERACT_PATH_CANDIDATES` in
`python/plo5bp/ocr/text.py:19`, so no PATH edit needed.

**Symptom when missing**: every `read_chip_amount` / `read_pot_amount` /
`read_seat_commit` call silently returns `None` (the
`TesseractNotFoundError` is caught at `text.py:113`). Downstream:
`observed_stacks` is `(None,)*num_seats`, `_begin_new_hand` skips
seeding any seat with `cents is None` (`server.py:1862`), and stacks
fall back to the session default `GameConfig(starting_stack=400000)`
(40bb each → $80 raw, $74 after 3bb ante). Uniform $74 across every
seat in the UI is the tell.

**After installing**: restart any running UI server. The
`_tesseract_configured` flag in `text.py:25` is cached per-process and
will not retry the candidate-path probe.

See also [[feedback-preflight-external-preconditions]] — same class of
bug as missing CUDA wheels; probe the system-level binary before
assuming the Python extras did the job.
