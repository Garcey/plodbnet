# PokerNow browser bridge

Live-capture source for **PokerNow** (pokernow.com / .club) tables, as an
alternative to the ClubGG pixel-OCR path. PokerNow is a browser web app, so we
read game state **directly from the DOM** — no Tesseract, no HSV tuning, no ROI
calibration. It's exact and resolution-independent.

## How it works

```
pokernow.user.js (Tampermonkey, in your browser)
    reads the table DOM on every change (MutationObserver)
    │  POST pokernow.v1 JSON  (GM_xmlhttpRequest)
    ▼
POST /pokernow/ingest        (plo5bp.ui.server)
    map_payload() → FrameState → EventReconstructor → Session → _rebuild_env
    ▼
Study UI shows the live hand + model recommendation
```

The same `FrameState` → reconstructor → `Session` pipeline the ClubGG OCR path
uses; only the front end (DOM read instead of pixel OCR) differs. See
`python/plo5bp/ocr/pokernow.py` for the mapper.

### Why a userscript and not fetch/WebSocket from the page

An `https://pokernow.com` tab **cannot** reach `127.0.0.1` from page context —
Chrome's Private Network Access + mixed-content rules block it (a page-context
`ws://localhost` just hangs in CONNECTING; `fetch` is blocked outright).
`GM_xmlhttpRequest` runs in Tampermonkey's privileged extension context and
bypasses those restrictions, which is why the script declares `@connect`.

## Install

1. Install the **Tampermonkey** extension (Chrome/Edge/Firefox).
2. Tampermonkey → *Create a new script* → paste the contents of
   [`pokernow.user.js`](pokernow.user.js) → save.
3. Start the study server: `.venv/Scripts/python -m uvicorn plo5bp.ui.server:app --port 8765`
4. In the study UI top bar, switch **Live** source to **PokerNow**.
5. Open your PokerNow table. The badge in the page corner turns green
   (`connected`) and the UI's PokerNow status shows live frames.

If the server runs on a non-default port, edit `INGEST_URL` at the top of the
userscript.

## Keep both windows visible

Chrome heavily throttles timers in **background** tabs. For the live table to
update the study UI the instant a card deals, keep the **PokerNow tab and the
study UI both visible** — ideally two browser windows side by side, not two tabs
in one window where one is hidden. (The collector sends via the table's
MutationObserver, which isn't throttled, so a backgrounded tab mostly works —
but a fully hidden tab can still lag. Side-by-side is the reliable setup.)

Also note: open the study UI consistently on **one** origin. `localhost:8765`
and `127.0.0.1:8765` are different origins to the browser, so the
ClubGG/PokerNow source toggle (saved in localStorage) won't carry between them.

## Scope / safety

The bridge **only reads** the table DOM. It never clicks, types, or automates
play — it's a study aid, not a bot.

## Supported

- PLO5 double-board **bomb pots** (the project's variant): 5-card hole + two
  boards (`run-1` / `run-2`), ante → flop (both boards) → turn → river.
- Variable table size (heads-up through full ring); seats are ordered
  geometrically (hero = engine seat 0, then clockwise).

Set the table's bb / ante / $-per-bb in the study UI's config so chip amounts
map correctly (same requirement as the ClubGG path).
