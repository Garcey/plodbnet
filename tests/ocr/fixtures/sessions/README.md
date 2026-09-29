# Recorded live sessions (replayed by `tests/ocr/test_live_replay.py`)

Each `*.jsonl` here is a live ClubGG or PokerNow session recorded by the
local UI and replayed on every test run: the test feeds the recorded frames
(or PokerNow payloads) through today's live pipeline and fails at the first
tick whose events or action log differ from the recording.

## Recording one

1. Start the local UI with recording on (PowerShell):

   ```
   $env:PLO5BP_LIVE_RECORD = "recordings"; .venv\Scripts\python -m uvicorn plo5bp.ui.server:app --port 8765
   ```

2. Play (or watch) the hands that show the problem, with "Track actions: On".
   Each capture session writes `recordings/live_<source>_<time>.jsonl` (and,
   for ClubGG, a PNG of every frame that raised a warning).
3. Replay it to see what the pipeline does tick by tick:

   ```
   .venv\Scripts\python -m plo5bp.ui.live.replay recordings\live_ocr_<time>.jsonl
   ```

4. To pin the behaviour, copy the `.jsonl` into this folder (trim it to the
   hand you care about — keep the first line, the header). Recordings are
   plain JSON, a few MB per hour at 5 frames a second.

A recording of a BUG pins the buggy behaviour until it is fixed: after the
fix, re-record (or re-run and accept) so the file shows the right actions.
