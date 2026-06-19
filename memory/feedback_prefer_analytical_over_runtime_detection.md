---
name: Prefer analytical one-shot transforms over runtime detection
description: For calibration/layout problems with a fixed underlying mapping, propose a one-shot mathematical transform — not a runtime detector. User repeatedly rejects per-frame heuristics in favor of measure-once solutions.
type: feedback
originSessionId: 78983dc4-2873-4db3-a0a7-cbda35e7b613
---
When the underlying mapping between two layouts is fixed (same content,
different framing), prefer a **one-shot analytical transform** (anchor-pair
affine, fixed offset table, etc.) over **runtime detection** (per-frame
black-bar stripping, aspect-ratio sniffing, on-the-fly inference).

**Why:** During the WMP-vs-ClubGG ROI remap, I proposed three solutions in
order: (1) `_strip_letterbox` runtime bar detector, (2) ROI overlay tool +
manual hand-tuning, (3) anchor-pair affine remap. User rejected (1) and (2)
and only accepted (3), with the framing "There's gotta be a way to just
move every single coordinate mathematically." Their next message added the
geometric insight that WMP-fullscreen and ClubGG-fullscreen physically
overlap on the monitor — meaning the OLD frame is the NEW frame plus
chrome, and a fixed affine handles it. They think geometrically and want
the simplest fixed transform that captures the invariant.

**How to apply:** When the problem has the shape "things look different
between capture-source A and capture-source B and our coords are tuned
for A," resist proposing runtime detection (heuristic strip / sniff /
adaptive thresholds). Look for the fixed transform first: what's the
analytical relationship between A's frame and B's frame? Two anchor
landmarks usually pin it down. Runtime detection adds permanent
maintenance burden; a one-shot transform updates the constants once and
disappears from the runtime path.
