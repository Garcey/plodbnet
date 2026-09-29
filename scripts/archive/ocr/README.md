# Archived OCR diagnostics and capture experiments

One-off tools from the ClubGG OCR bring-up (2026-05/06). None is used by the
app or the tests; several depend on debug frames that were lost in the
2026-06-26 `screenrecords/` cleanup, on server internals that have since moved
to `plo5bp.ui.live`, or on capture paths that cannot see a ClubGG table. Kept
for reference; they are not maintained.

| script | what it was for |
|---|---|
| `diag_pixels.py` | per-ROI pixel signals on the "Butt2Butt" bug frame (Phase 1) |
| `diag_frame.py` | `extract_frame_state` dump of the post-bet bug frame (Phase 2) |
| `diag_reconstruct.py` | EventReconstructor forensics on a synthesized frame pair (Phase 3) |
| `diag_begin_hand.py` | `_begin_new_hand` + rebuild on the bug frame (Phase 6a) |
| `diag_ocr_frame.py` | cold-start "pot $0 + no recommendation" walk-through |
| `diag_roi_sweep.py`, `diag_roi_finetune.py` | 2-D / fine y sweeps of seat 1's commit ROI |
| `diag_sweep.py` | primitive signals across the 8 debug frames |
| `dxcam_capture.py` | DXGI desktop-duplication capture — cannot capture ClubGG (WDA) |
| `save_capture_with_rois.py`, `overlay_stack_rois.py` | mss captures with ROI overlays — mss can't see ClubGG either |
| `remap_rois.py`, `remap_rois_lsq.py` | ROI re-mapping from landmarks (the LSQ one needs a `rois.py.bak.wmp` that no longer exists) |

Live diagnostics that still work: `scripts/window_affinity_monitor.py`,
`POST /ocr/save_frame`, and record / replay (`PLO5BP_LIVE_RECORD`,
`python -m plo5bp.ui.live.replay`).
