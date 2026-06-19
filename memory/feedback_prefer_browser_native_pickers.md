---
name: Prefer browser-native pickers over backend heuristics
description: For UX surfaces where the browser exposes a native unambiguous picker (window/screen/file/device), use that instead of building a backend dropdown driven by string matching or heuristics
type: feedback
originSessionId: 78983dc4-2873-4db3-a0a7-cbda35e7b613
---
When a UX surface lets the user identify a resource the browser
already has a native picker for — `getDisplayMedia` for window/screen
selection, `<input type="file">` for files, `enumerateDevices` for
audio/video — use the browser primitive. Don't build a custom
dropdown populated by backend window-title scraping or similar
heuristics.

**Why:** Backend pickers based on string matching (e.g.
`pygetwindow` substring) are brittle: ambiguous matches, transient
window state (minimized at click-time), OS-specific quirks, no
visual confirmation. The user explicitly cited Zoom / Discord /
Meet's screen-share dialog as the reference UX — visual, unambiguous,
no user-typed string to disambiguate.

**How to apply:** When asked to add or fix a picker for a
resource the browser exposes natively, default to the browser API.
Push the actual data (frame, file, etc.) to the backend over HTTP.
Keep backend-side helpers for CLI / non-UI use cases (the title-match
flow stayed available for `scripts/*.py` even after the UI moved
to `getDisplayMedia`), but don't surface them as the UI primary path.
