# Home-games felt measuring tools (development only)

These scripts prove that the tuned felt layout (`python/plo5bp/ui/static/games.table.js`:
`fitSeats`, `fitPots`, `placeBetSpots`, `#stage.h6/.h7`, phone landscape) has no overlaps.
They are kept in git (they used to live only in the git-ignored `.claude/tools/`, the way
the labelled OCR frames were lost in June 2026).

- `measure_showdown.js` — paste into a seated player's page on a preview server; logs every
  pair of felt elements that overlap at each runout street and award step
  (`__showdownWatch()`, then `__log.filter(x => x.n)`).
- `measure_bets.js` — the same for the bet spots during a hand.
- `measure_dock.js` — does the TABLE keep its size whatever the dock shows (your turn, the
  pre-actions, the status line, the Trainer's "last move" line…)? Paste into a home-games
  table, the Trainer or Study; `__dockWatch()`, drive it (`__autoTrainer(ms)`,
  `__autoHome(ms)` with `bot.py loop <table>` beside it), then `__dockReport()`: ONE size per
  window size (2026-10-02). It polls the box, so it works in a hidden tab too.

Both lay the felt out for the window and FINISH its running transitions before they measure:
a background tab is never painted, so its animation clock stands still and a tabled row or a
pot pill would otherwise be measured where its transition started (2026-09-28 — earlier runs
in a hidden preview tab could report overlaps that were never on screen, or miss real ones).

Run them at 360–430 px phones, 375x667, 812x375 (landscape), 768x1024 and desktop; overlaps
that were 0 must stay 0. The preview server and the scripted guests (`run_games_preview.py`,
`bot.py`, `drive.py`) stay in `tools/games_preview/` (local harness).
