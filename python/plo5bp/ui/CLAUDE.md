# The website (`python/plo5bp/ui/`) — notes

Loaded when you work under `python/plo5bp/ui/`: the public build and its service layer (sign-in, accounts, admin), Study, Trainer, the NLH range grid, and the private home games (tables, clubs, verified shuffle, history, grading). Live capture has its own notes: `python/plo5bp/ocr/CLAUDE.md`.

### NLH range grid ("Ranges" tab, LOCAL BUILD ONLY)

GTO-Wizard-style 169-hand strategy grid (built 2026-07-06). The tab
appears next to Study/Trainer only when format=NLH AND not
`PLO5BP_PUBLIC`; the public build never mounts the `/ranges` router (like
`/ocr`) — do NOT mount it there without an explicit user decision
(and admin-gate it first when that day comes).

- Rust: `PyGameState.pack_range_nlh(holes)` packs ONE decision node once
  per candidate hole in the exact batched-packer layout (the obs is
  villain-blind; only the hole multi-hot + `hero_cat_a` +
  `nlh_opp_outcome` vary per row — the latter two via
  `nlh_category_for` / `nlh_opp_outcome_for` free fns, rayon).
  **Bit-exact vs serial study obs**, pinned by
  `tests/python/site/test_ranges.py` — keep that parity.
- Backend: `plo5bp/ui/ranges.py`, stateless `POST /ranges/query`
  (seats, stack_bb, `line` = ONE ordered list interleaving actions and
  street cards, optional node = prefix length). Ephemeral study replay
  (never the user's study Session), one batched forward per node
  (in-process cache per line-prefix), reach = per-seat gate-level Π of
  the seat's own action probs (no size-conditional reach in v1), fixed
  5/10($5) stake with button=seat 0 so positions are canonical. A study
  street boundary sets the env's done flag — check awaiting FIRST.
- Frontend: `static/ranges.js`, self-contained (app.js untouched — the
  tab toggles `body.ranges-mode`); strips builder with GTO-Wizard
  rebranching (acting at a viewed past node truncates the line there),
  anchor-size raise buttons, street card popup, hover combo breakdown.
  Client queries NEVER drop-when-busy: latest-response-wins via RG_SEQ
  (dropping desyncs the line from the render — bug found in validation).
  A failed query rolls back the line AND the viewed node.
- Sizes: the wire format (`chips_bb`) is the engine's raise-BY delta; the
  payload also carries raise-TO totals (`to_bb`, `actor_commit_bb`,
  `min/max_raise_to_bb`) and the client shows/enters raise-TO. ANY legal
  anchor whose chips equal `max_raise` is all-in (the ALL-IN atom is
  deduped away whenever a fraction anchor already reaches the stack) — it
  is summed into `allin_p` and labelled ALL-IN.

## Promote good checkpoints to the LOCAL UI (never to wrapgto.com on your own)

**wrapgto.com — model or code — never without the owner's explicit OK, each
time.** Prepare it (the h2h evidence + the command) and hand it over:
`bash scripts/deploy_prod.sh promote checkpoints/<file>.pt` verifies the file with
the live code and stages it; the owner then clicks Promote in /admin → System (no
restart). A model on the other observation revision goes with `OBS_REV=<rev> RESTART=1`
in front (PLO5BP_OBS_REV changes with it; both go back if the site is unhealthy) —
never a hand edit of /etc/wrapgto/env. Every live change runs detached on the server
(`watch` follows it, `recover` finishes a crashed one). Runbook: docs/ops/PRODUCTION.md;
history: docs/models.md. (On 2026-09-26 a checkpoint went live unasked — this rule
exists because of it.)

The LOCAL UI (`checkpoints/stub.pt` in this repo, uvicorn :8765): promote a better
checkpoint without asking. "Better" means: it beats the one promoted now head to head
(`scripts/h2h_cross.py`, sampled AND argmax, over several checkpoints — a single one
swings ±0.1-0.2 bb/seat-hand); it loads with its critic and no OBS-REV mismatch under
the `PLO5BP_OBS_REV` you serve (the server log / `/health` say so); and the run's log
is healthy (finite losses, entropy not collapsing, no KL-guard trips). If any of those
fail, flag it and do NOT promote.

Mechanism: the UI loads one checkpoint PER FORMAT —
`$PLO5BP_CHECKPOINT` / `checkpoints/stub.pt` for PLO5 and
`$PLO5BP_CHECKPOINT_NLH` / `checkpoints/nlh_stub.pt` for NLH (see
`python/plo5bp/ui/models.py` + the `FORMATS` registry; a missing stub
serves a random-init placeholder flagged "untrained" in the UI).
Promote by copying over the format's stub:

```bash
cp checkpoints/<run_name>.pt checkpoints/stub.pt       # PLO5
```

The local UI loads its models at start, so restart it to pick up new weights
(the public build can also reload / promote from /admin → System).

The UI serves every checkpoint generation: the head class is sniffed
(`anchor_head.weight` → v2, `raise_head.weight` → v1, …) and so is the trained obs
width (older widths get the exact downgrade projection via `network.obs_adapter`;
the encoder always emits the current width, OBS_DIM 1171). v2+ recommendations carry
an `anchors` histogram + `rec_anchor` + `refine`; trainer scoring snaps the user's
raise size to the nearest legal anchor (`score_move_v2`).

## Restart services yourself (LOCAL services only)

If an action you just took (promoting a checkpoint, editing
`server.py`, etc.) requires a service restart to take effect, do the
restart — don't ask. This applies to the LOCAL UI server and to any other
local dev service in this repo — never to the production service
(`wrapgto.service` on the server is the owner's). Check for an existing
instance (`netstat -ano | grep :8765` for the UI), stop it if found,
and start the new one in the background. Report the new URL/PID so the
user can find it.

Start the UI with:

```bash
.venv/Scripts/python -m uvicorn plo5bp.ui.server:app --port 8765
```

(Drop `--reload` in a scripted background start — it spawns a reloader
subprocess that complicates clean shutdown.)

## Trainer mode (GTO-Wizard-style practice)

`python/plo5bp/ui/trainer.py` — a second UI mode (tab next to Study, or
`/?mode=trainer`) that deals random bomb-pot hands and drills the user
against network opponents. Backend rides on the same app/MODEL under
`/trainer/*` (`create_trainer_router`), state fully separate from the
study `Session`. Shared pure helpers live in `python/plo5bp/ui/common.py`.

Key invariants (documented in the module docstring — don't break):

- A hand is fully determined by `(config, seed, button)`; review /
  repeat / EV-loss all rebuild by replaying the recorded `action_log`
  through a fresh `env.reset(seed, button)`. Nothing snapshots live
  engine objects.
- Trainer terminal state is synthesized from the `done` flag —
  `study_terminal` / `awaiting_next_street` are study-mode-only engine
  fields and stay `None` in random-deal mode.
- Opponents sample the mixed strategy via `eval.model_policy(...,
  deterministic=False)`, re-seeded per node from `(hand seed, action
  prefix length)` — behavior is a pure function of the action prefix.
  trainer.py is the ONLY consumer of torch's global RNG in the UI
  process (study path is always `deterministic=True`).
- Scoring (`score_move(_v2)` → `_grade`, the `SCORING` dict; 2026-10-03, owner: a flop
  fold the network mixed at 23% next to 40/37 was graded an inaccuracy): r = P(your gate)
  / P(its favourite gate), on a LOG scale — an entropy-regularised policy plays about
  exp(EV / temperature), so log(1/r) is the EV-loss proxy, not 1 - r. score = 100 ×
  (1 - log(1/r) / SCORE_SPAN), 10 at r = 1/50; a raise's size (anchor ratio × Beta pdf
  ratio) adds its own log-ratio × 0.6, at most 1.9 (a good move with an odd size is an
  inaccuracy at worst). Categories: best r ≥ 3/4 (near-ties too, not only the argmax),
  correct ≥ 1/4, inaccuracy ≥ 1/10, wrong ≥ 1/50, blunder rarer or P(gate) < 2%; the
  score bands (`best_min` …) are derived and Hand review's mistake severity reads
  `inaccuracy_min`. The Trainer's "How moves are graded" help says the same in words.
  Every grader shares it (the Trainer, home games, Hand review). EV loss = paired Monte-Carlo rollouts (common random
  numbers) of user action vs the deterministic rec; `mc_rollouts`
  default 16 keeps a deviating `/trainer/act` under ~1s on CPU with the
  2048×4 net (matching actions skip MC entirely). v2+ scoring inverts the
  refinement `u` over the UNCLAMPED anchor bracket (`anchor_lo_raw/
  hi_raw`) — the policy maps `u` over the unclamped bracket and clamps
  chips afterwards, so inverting over the clamped grid graded the exact
  recommendation as an "inaccuracy" whenever min/max-raise clipped the
  bracket; chips equal to `rec_chips` always score "best" with no MC.
- The EV-loss estimate replays the REAL deal and future board, so it is
  hidden while the hand is live (`feedback.ev_loss_bb: null`,
  `ev_loss_hidden: true`) and revealed at terminal / in review; stats are
  committed once per COMPLETED hand (abandoned hands add nothing) and
  accumulate the SIGNED estimate, clamping only the displayed aggregate.
  `_rollout_ev` holds `_TORCH_RNG_LOCK` only around seed→sample regions
  (env work happens outside it); the public build caps `mc_rollouts` at
  32 for players — not for its admins (2026-10-03, owner: "as the site admin/owner, I
  should be able to set this to whatever I want for myself"): `public.install` plugs
  `current_user_is_admin` into `trainer.set_uncapped_user_hook`, so `mc_rollouts_cap()`
  is 256 (`MC_ROLLOUTS_MAX`, the local build's) for an admin's requests; a failing check
  keeps 32. The settings window shows the ceiling ("At most N."; `mc_rollouts_max`).
  `POST /trainer/act` with no live hand is a 409 (never auto-deals).
- What-if card swaps replay through `reset_study` — an unmodified
  what-if reproduces the original node's observation bit-exactly
  (pinned by `test_trainer_review.py`). What-if stays hero-only.
- Review steps EVERY decision node (hero + villain): the arrows drive a
  shared node cursor (`review_at_node` / `?node=` → `review.node_current`,
  one `_node_view` per click); pills still jump to hero decisions. Each
  node shows the acting seat's policy + DUAL EV — the actor's own (blind)
  value head AND the critic's all-cards "true EV" (UI loads `ckpt['critic']`
  via `server._load_critic`; opp multi-hot built with the canonical
  `rollout._rotate_opp_holes` so it matches training). Villain nodes have
  no graded score; hero nodes overlay the stored `DecisionRecord`.
- `all_hole_cards()` (engine accessor added for this) reveals opponent
  cards — projection exposes them only at terminal/review.
- Lifetime stats + settings persist to `checkpoints/trainer_stats.json`
  (override with `PLO5BP_TRAINER_STATS`); session stats are in-memory.

- **My tables (2026-10-03; owner: deal "based on the individual user's hand histories … typical
  stack sizes for opponents and for themselves", represented — "I would prefer that you
  represent over pulling exact configurations")**: `TrainerSettings.tables` = "custom" (the
  default, the seat / stack / ante fields) or "mine": `_my_table` draws from the player's
  PROFILE — Hand review's `my_tables` (`handreview_store`, plugged in by
  `set_my_tables_hook`): `handreview.table_profile` of their latest `PROFILE_HANDS` (5,000)
  hands = players (shares of 2-6), ante (shares) and TWO stack distributions, the player's
  own (with auto top-up it rarely starts below the buy-in) and everyone else's; each = the
  exact amounts that recur (>= 2% and >= 5 times: a buy-in, a top-up target) as point
  masses + 101 quantiles of the rest, drawn log-uniformly between neighbours (never outside
  what the hands held); `draw_table` draws every seat on its own, so no real table comes
  back. Cached per player until an upload / deletion (`_profile_stale`). Without a usable
  profile (no subscription — it is part of Hand review —, under `PROFILE_MIN_HANDS` (50)
  hands, the local build, a store error): TYPICAL ClubGG tables = the training tier
  `clubgg_real` (plo5bp/train/tiers.py, ~950 real hands at $10/$20, up to 350bb), ante 3bb,
  your stack drawn like everyone's. PLO5 only (NLH ignores it). `GET /trainer/my_tables`
  (`my_tables_view`) says which and why, with the numbers the settings dialog shows (never
  the quantiles). Client: the "Tables" row of the settings dialog (hides Players / Stacks /
  Ante; `#ts-tables-help` = the profile in five lines) and the work bar's "My tables" toggle
  (`toggleMyTables`, hidden in the mistakes drill; narrow phones slim the row's buttons to
  keep it one line at 360-400 px). `test_trainer_my_tables.py` + the store half in
  `test_public_hand_review.py`.

Tests: `tests/python/test_trainer_*.py`, `test_all_hole_cards.py`
(shape parity with the study `_state_dict` is pinned). The table half of
both payloads (seats, pot, buttons, raise window, history, chip scale) is
ONE builder, `common.table_state` (BE-008): a new table field goes there
and reaches both modes; a Study-only key the renderers need must also be
added to `TrainerSession.project_state`.

## Public build (`PLO5BP_PUBLIC`)

Set `PLO5BP_PUBLIC=1` to serve the trainer + study tabs WITHOUT any live
capture — the public / monetizable variant (model served server-side, no
real-time table reading, so nothing that enables RTA). Single codebase,
one flag; unset (the default) is the full local build with OCR + PokerNow
intact. Run it with:

```bash
PLO5BP_PUBLIC=1 .venv/Scripts/python -m uvicorn plo5bp.ui.server:app --port 8765
```

- Server (`server.py`): under the flag the live-capture package
  (`plo5bp.ui.live`: `/ocr`, `/pokernow`) and the Ranges router are never
  imported or mounted, so those paths 404 and none of that code ships
  (pinned by `tests/ocr/test_live_package.py`). The `/` route injects
  `window.PLO5BP_PUBLIC` into the served HTML so the client knows its mode
  before first paint (no `/config` fetch race, no flash of the live
  controls). Static files are policed on the RESOLVED file inside
  `NoCacheStaticFiles` (so `/static//app.js`, trailing `/`, `..`, case
  variants can't dodge it): `app*.js`/`style.css` are served WGLIVE-stripped;
  `index.html`, `ranges.js`, `admin.html` and the `games.*` assets 404 from
  the mount (each has its own gated route). The public app is built with
  `docs_url=None, redoc_url=None, openapi_url=None`.
- Frontend: the Study / Trainer client is seven plain scripts that index.html
  loads in order into one global scope (2026-09-28, FE-019): `app.core.js`
  (units, formatting, dialogs, the `UI` state, requests), `app.table.js`,
  `app.play.js` (actions, sizing, recommendation, history), `app.study.js`,
  `app.trainer.js`, `app.topbar.js` (tabs, format, units, account menu) and
  `app.js` (render pipeline, `init`, and the WGLIVE live client). Code that
  runs while a file loads may use only earlier files; a new file needs an
  entry in `_PUBLIC_STATIC_POLICY`. When `window.PLO5BP_PUBLIC`, `setupTopBar` skips the
  live-control wiring and hides `.ocr-group`, `init` skips `applySourceUI`
  (the OCR/PokerNow status polling), and the per-render `simple_ocr_mode`
  sync is gated off (it was what revealed the `#ocr-rescan-group` buttons).
- Trainer + Study are 100% live-independent: every study route rebuilds
  from user input via `_rebuild_env`. Study = manual hand entry → replay →
  recommendation; trainer = random deals. Nothing in either path calls the
  live routes, so gating is purely additive.
- iPhone home-screen icon (2026-09-26, `test_touch_icon.py`): every page links
  `static/brand/apple-touch-icon.png` and `/apple-touch-icon.png` +
  `-precomposed.png` (in `public.OPEN_EXACT`, iOS probes the ROOT) serve it
  signed out. It is SQUARE and OPAQUE on purpose — iOS rounds the corners
  itself and paints see-through pixels black; `apple-touch-icon-180.png` is the
  rounded brand original. `apple-mobile-web-app-title` gives the home-screen
  name ("Home games" — games.js retitles the tab per table).

The public build also mounts the **service layer** (`python/plo5bp/ui/public.py`,
installed at the end of server.py only under the flag): Google sign-in
(Authlib; loopback-only dev login via `PLO5BP_DEV_LOGIN=1`), SQLite user DB
(`data/public.db`), free tier (5 trainer hands/UTC-day, middleware-enforced on
`POST /trainer/new_hand`; Study routes are subscriber-only → 402), Stripe
$10/mo subscriptions (checkout + success-redirect confirm + optional webhook +
lazy revalidation — no public URL needed), and `/admin` (users, comp
grant/revoke, revenue) allowlisted to `PLO5BP_ADMIN_EMAILS` (default
themilesgarcia@icloud.com). Local setup + accounts: `PUBLIC_SETUP.md`; production
runbook: `docs/ops/PRODUCTION.md`; every env var: `ops/env.example`. Tests:
`tests/python/site/test_public_service.py`. A test gets a public app from conftest's
`boot_public_server(**env)`: it sets the environment and calls
`server.create_app()` (BE-007 — the app factory: every app gets its own
settings, database, caches and home-games context; nothing is re-imported), and
at the module's end closes it and makes the local app current again.
`conftest.purge_ui_modules` / `ui_purge` remain only for a test that must
re-execute an import (popping `sys.modules` alone leaves the stale module bound
as an attribute of the `plo5bp.ui` package).

**Free while the models are in development (2026-09-22):** `public.FREE_FOR_ALL`
(`PLO5BP_FREE_FOR_ALL`, default ON) makes every SIGNED-IN user entitled — no
daily trainer quota, Study unlocked, `/billing/checkout` answers 409, `/me`
carries `free_for_all`, the account chip says FREE ACCESS and the landing page
says so. Sign-in stays. The paywall / quota / Stripe code is untouched and still
tested: `tests/python/conftest.py` defaults the test session to
`PLO5BP_FREE_FOR_ALL=0`; `test_public_free_mode.py` boots production's way. Set
the env var to `0` in `/etc/wrapgto/env` to bring the paywall back. **One exception
(2026-10-03): Hand review** (below) is paid even while FREE_FOR_ALL — it keeps a
player's hand histories on the server — so `/billing/checkout` stays open (it 409'd
before) and `/me` carries `paid` (`public._paid`: admin / comp / live subscription;
`_entitled` = FREE_FOR_ALL or `_paid`).

**Workspace UI (2026-09-22; the home games' table since 2026-10-01):** Study / Trainer
sit next to the home games but stay a TOOL. Layout: slim `#top-bar` (mode tabs, format,
units, account) — on a phone (<= 480 px) the mode tabs scroll inside their own strip and
Home games / Hand review show short names (2026-10-03: the two links pushed the account chip
off a 375 px screen) + per-mode `#workbar` above the table (Study: players / your seat / ante
/ live controls / Enter cards / Share / New hand; Trainer: Settings / Repeat / New hand)
+ the table + `#side-rail` (desktop: the Trainer's review on top, Recommendation, then the
13 x 4 card matrix (Study) or History + stats (Trainer); a phone stacks them under the
table). The Trainer's desktop rail never moves as a hand goes on (owner, 2026-10-03): the
hand's History fills the room between the Recommendation and the Session / Lifetime stats
and scrolls inside it, the stats are pinned at the bottom (sticky) with a "Previous hands
(N) ▾" tab (`#ph-tab`) under them, and the tab — or scrolling down over anything but the
History — slides `#prev-hands` (a child of `<main>` laid over the rail's own grid cell;
`#recent-hands` lives in it) up over the whole rail; Back, Esc or scrolling up at the list's
top slides it away (`setPrevHands` / `setupPreviousHands`, app.trainer.js; body.hands-open).
A phone lists the previous hands under the stats. Pinned by `test_trainer_rail.py`.
**ONE card face on the whole site — Study's** (owner, 2026-10-01: "use the
study/trainer card faces in the home games for uniformity"; suit-colour fill, rank over
suit top-left, big rank bottom-right, ten = "T").
- **The table is the home games' (owner 2026-10-01: "bring the same look to the study and
  trainer", each keeping its features; no buy-ins / rebuys / sitting there).**
  `games.table.js` draws it and `games.felt.css` styles it (the table part of games.css,
  split out and shared: section headers 1-8, loaded BEFORE the page's own stylesheet on
  both pages). `app.table.js` is the adapter: `feltView(s)` turns a Study / Trainer state
  into a home-games view (money in chips standing in for cents; seats named by position,
  "Hero" / "You"; Study's boards as five PLACES `{slots}` — null = a card to enter — the
  Trainer's as dealt streets; opponents face down unless tabled) and `renderTable` calls
  `HG.table.render(view, prev)` (persistent nodes, diff animations — never rebuild it).
  A shim defines `HG.core` / `HG.ui` (html/put, `fmtAmt` = `formatUnit`, seat click =
  Study's seat menu). On top, Study's own: card places are `.slot-pick` (click / Enter
  selects, double-click / Delete empties; `.slot-sel`, `.slot-next` = where the next card
  goes), editable stacks, the draggable dealer button, "+" between seats (`#seat-adds`,
  a mouse and < 6 players), what-if swaps in the Trainer's review (hero decisions only).
  `#dock` (inside `#stage-wrap`, as on games.html) = the home games' action bar: actor
  line, the Trainer's last grade, sizing (presets + Edit, slider, amount; a phone's first
  Bet tap opens it), Fold / Check-Call / Bet-Raise (`.rec` + "Network" badge = the
  network's choice on Hero's turn in Study), the status strip (hand over — the Trainer's
  "Next hand" — cards missing, opponents acting), Undo / Redo (Study).
- **Board focus (2026-10-03; owner: "hover over each board and have it highlight the two hole
  cards that I'm playing on that board", all three tabs)**: hovering a board — or its made-hand
  label under your cards (a phone: tap the label, 3.5 s) — lights the two hole cards you play
  there and its three board cards (`.card.plays`), the rest of the felt's cards dim
  (`#stage.board-focus`); `games.table.js` `bestPlay` / `applyBoardFocus` (re-applied after
  every render). PLO picks exactly what `hand_describe.best_combo` picks (pinned by
  `test_homegame_board_focus_js.py`); one board (NLH): the best five of seven, fewest hole cards.
  The ace-high straight flush reads "a royal flush" (felt short form "Royal flush").
- **The table never changes size with the turn (owner, 2026-10-02: "the addition and
  removal of the betting options slightly resizes the whole table")** — on all three
  pages. The felt gets what `#stage-wrap` leaves after the dock, so every part of the
  dock holds ONE height per layout: `#act-slot` (both pages: sizing + buttons / pre-actions
  / status strip) has `min-height: var(--slot-h)` = the one-row sizing panel (`--sz-h`,
  forced to exactly that height) + 8 + the buttons on a desktop, the buttons' height on a
  compact layout (the sizing panel floats there), 0 where the dock floats (phone on its
  side); the compact status strip wraps its words BESIDE its buttons (two lines fit), never
  under them; `#hero-hand-labels` is one line (a long label ends in "…", `.hhl-txt`).
  Study / Trainer (style.css): the actor line and the Trainer's "Your last move" line are
  single lines that keep their place when empty (`[hidden]` → `visibility: hidden`, with
  `!important` against the page's global `[hidden]` rule); no-limit's sizing panel is
  ALWAYS two rows on a desktop (`body.fmt-nl`, set by `renderActions`: `--sz-h` 92 px),
  pot-limit's one; small buttons on the felt are 30 px (`.primary`'s old 40 px min-height
  made "Next hand" taller than the buttons). The status strip is one message
  (`.strip-msg`) + its buttons (`note(msg, btns)`). A new dock line or state must keep
  this: check with `tools/games_preview/measure_dock.js` (one `#stage-box` size per window
  size, every state) at desktop, short-window, tablet and phone sizes.
- **Public build**: the /static mount refuses every `games.*` file, so for a signed-in
  page `server._link_shared_table_assets` points index.html's two links at
  `/games/static/<name>?v=<hash>` (signed out: the WGAPP strip drops them with the app).
  `test_site_landing.py::test_signed_in_page_draws_the_home_games_table` pins the links,
  their order and that every `$("id")` games.table.js looks up exists in index.html;
  `test_site_app_js.py::test_the_table_reads_study_and_trainer_states` pins feltView.
- `style.css` layers: base, "WORKSPACE THEME v2", "WORKSPACE v3", then "WORKSPACE v4"
  (2026-10-01: the home games' palette on `:root`, the page around the table, the
  Study/Trainer additions on the felt and dock, the phone column) — it wins by cascade
  order, so add new workspace styles after it. A shared table rule goes in
  `games.felt.css` (both pages), never in style.css.
- The landing page's product picture (`.shot` in index.html, its rules in style.css) is
  the Trainer's review drawn in plain HTML + CSS in this look — signed-out visitors don't
  load games.felt.css, so it carries its own copy: keep it in step when the UI changes.
Card entry is continuous
(`placeStudyCard` / `nextEmptySlot` in `app.table.js`): a placed card selects the next
empty slot across groups (hole -> flop A -> flop B -> turn -> river), a card
clicked with nothing selected fills the first empty slot, and cards can be typed
(rank then suit, Backspace undoes).

Service-layer rules (review 2026-09-20 — keep them):

- The access middleware authorizes on a NORMALIZED `scope["path"]`
  (`//`, trailing `/`, `.`/`..`, backslashes resolved). `/format` is a
  subscriber route. Implicit deals are metered: for a non-entitled user
  whose `TrainerSession` has `hand_no > 0` and `hand is None`, a request to
  `/trainer/state|settings|act|stats/reset` counts against the quota like
  `POST /trainer/new_hand` (the first implicit hand of a fresh session
  stays free).
- Stripe: re-validation never runs on the event loop (threadpool, 8 s
  timeout); `resource_missing`/"No such subscription" ⇒ INACTIVE; other
  errors fail open only until `current_period_end` + 3 days, re-checked at
  most every 15 min (`PLO5BP_STRIPE_TIMEOUT`, `PLO5BP_STRIPE_GRACE_DAYS`,
  `PLO5BP_STRIPE_RETRY_S`). Activation (confirm AND webhook) requires
  `mode == "subscription"`, a subscription id, `payment_status` paid or
  `no_payment_required`, and a subscription that is active/trialing NOW —
  replaying an old session cannot re-activate.
- Dev login exists only when `PLO5BP_DEV_LOGIN=1` AND the `BASE_URL` host is
  loopback; it rejects any forwarding header (XFF, CF-Connecting-IP,
  Forwarded, …) and needs a loopback client + Host. The test client is
  accepted only with `PLO5BP_DEV_LOGIN_TESTCLIENT=1` (fixtures set it).
  `email_verified` defaults to False when the claim is absent.
- `/me.homegame` is `{"href", "label"}` for every SIGNED-IN user (clubs,
  2026-09-25; absent signed out) so the client (`app*.js`) ships no home-games strings to a
  signed-out visitor.
- **Schema changes go through `Db.migrate(component, [Migration(...)])`**
  (public.py; 2026-09-28, OPS-029/OPS-043): numbered, append-only steps per
  component (`public`, `homegame`, …), each in one transaction, recorded in
  `schema_migrations`. **Migrations only ADD** (tables, nullable/defaulted
  columns) so a code rollback keeps working on the migrated DB; a rename /
  drop / changed meaning must be a `breaking=True` step — older code then
  refuses to start on that DB ("restore the pre-deploy backup"), loudly.
- DB access: writes (and anything inside `DB.transaction()`) go through the
  one writer connection + lock; plain SELECTs run on a per-thread read-only
  WAL connection and never wait for the writer (`synchronous=NORMAL`,
  `foreign_keys=ON`). The access middleware caches user rows 30 s,
  invalidated by any committed write touching `users`.

**Home games** (`ui/homegame.py`, `static/games.*`, every signed-in user since
clubs — a club's tables, members and numbers are its members' only; signed out
= a sign-in page for a browser, the hidden 404 otherwise): PokerNow-style private PLO5 /
PLO6 / PLO67 double-board tables over `BombPotEnv` (minimal obs, actual payouts, HTTP
polling, per-table RLock + clock watchdog). Invariants: hole cards are
revealed only when ≥ 2 hands are live at terminal (an uncontested winner
stays face-down); "own" cards are shown against the user id DEALT into the
seat this hand (a seated-but-not-dealt viewer or a seat taken over later
sees nothing); all-in equities are computed once per (hand, street) from
the ALIVE seats (`runout.board_equities` = one exact enumeration per
board) — never per viewer under the lock; money is exact integers and the
ledger always sums to zero (largest-remainder apportionment at cash-out);
mutations persist before (or roll back with) in-memory state; `/act` and
`/deal` carry `hand_no`/`action_seq` and 409 when stale; `/deal` refuses
while a runout is still revealing and the payload releases deltas/awards
progressively (no unrevealed card anywhere); chat/rabbit need table
membership; new tables default to a 30 s clock and the watchdog auto-acts
Away actors even on clock-less tables; a mid-hand leave folds/checks now
and cashes out at hand end.

PLO6 tables (2026-09-26 — `tests/python/homegame/test_homegame_plo6.py`; owner request):

- A table deals **PLO5 or PLO6**, chosen at creation and fixed for its life like
  the blinds (`homegames.variant` = `plo5` | `plo6`, `homegame.GAMES`; host another
  table to switch). PLO6 = six hole cards, **7 seats max** (7 x 6 + 10 = 52: the
  whole deck, no burn cards) — `_valid_num_seats(n, variant)`, a create without
  `num_seats` gets the game's max. The engine plays it (`GameConfig(variant=
  GAMES[v]["variant"])`); `runout.py` / `hand_describe` are hole-count generic
  (pinned against the engine's PLO6 payouts by a randomized sweep).
- **Never graded**: only `GAMES[v]["graded"]` hands reach the grader
  (`_enqueue_grading`; `grade_hand` refuses another game too); a PLO6 record has
  `grades: []` (not `None` = "still being worked out") and no `acc_n`.
- Records carry `variant` + `hole_count`; the view `variant`, `game` {code, label,
  name, hole, max_seats, graded} and `hole_count`; lobby rows, sessions and hand
  lists carry `variant`. The replayer draws that many face-down cards and hides
  "Open in Study" (Study deals PLO5; `openInStudy` refuses another game).
- **Verified shuffle**: the slot map is `h·s+k`, boards `h·n..` with `h` =
  `SealedDeck.hole` (5 when absent — PLO5 transcripts unchanged; public
  transcripts and the store carry `hole`); a non-PLO5 `hand_id` ends in `:plo6`, so
  the seal names the game. `games.fair.js` maps slots by `s.hole_count` and refuses
  a transcript whose `hole` differs from the table's.
- **Felt**: `#stage.h6` (games.felt.css) + `ROW` (games.table.js) — a six-card row takes
  about a five-card row's room: hero cards 0.9x with 0.28 overlap, a tighter
  face-down fan (`--fov/--mid/--rot`), tabled rows from 0.4 overlap up to 0.6. Keep
  the two in step. Found with PLO6's 7-seat default and fixed for every table:
  `fitSeats` places rows by where the award caption LANDS (`captionLanding`,
  `--cap-dy`: capin slides it in from higher up, so its first-frame box was wrong),
  and `fitPots` keeps pots clear of tabled rows at their tightest fan (7 seats put
  the upper side seats level with a runout's pots). Measured with
  `measure_showdown.js` on 7-player PLO6 tables: no real overlap at 360-430 px
  phones (a few 1-3 px corner slivers — a label on a board's end card), phone
  landscape, tablet and desktop; an SE-size 375x667 multi-pot (3+) runout
  keeps 2-3 px touches between the two top seats' badges and the pot pills (an
  11 px gap for 16 px pills) — like PLO5's SE-size corner touches.
- **Club numbers per game**: `/games/api/community?variant=` (default = the club's
  most-played game; `games` = hands per game for the switch; players + pairs
  filtered; sessions all, each tagged); `my/stats|hands` and `players/{id}/stats|
  hands` take `variant` (absent = every game; `games` breakdown). Lobby: the
  `#lb-game` switch (remembered per club, localStorage `hg.clubgame.v1`); an
  ungraded game has no podium and its cards meter hands won; a player's stats
  window opens on the club's game with All games / per-game switch. A new
  per-game aggregate must filter `g.variant` (`_games_played`, `_games_summary`).

PLO67 tables (2026-09-27 — `tests/python/homegame/test_homegame_plo67.py`; owner /goal):

- `GAMES["plo67"]`: `hole` 7 (= the shuffle's slots per seat and the view's
  `hole_count`: the felt is laid out for the widest hand), `dealt` 4, `burns` 3,
  **5 seats max**, never graded; `PLO67_ON` (the engine has the pyfunction) gates
  creating one. `game` info carries `dealt` + `burns` for every game.
- View: `burns` = the burns turned up so far (one per street on the board: the
  rabbit / an all-in runout reveal them street by street), `burns_played` = how
  many belonged to the hand (a fold-out's rabbit turns up burns that dealt nobody
  anything), and per seat `hole_seq` = a hand the viewer may see in DEAL order
  (the felt animates the card a red burn dealt — display order is sorted). While an
  all-in runout reveals, every hand shows what it held on the street shown
  (`_hole_count_on`). Equities per street = `_street_equities` → the Rust sampler
  (whole runouts dealt the game's way, so the chance of the extra cards red burns
  will still deal IS in them — checked against an independent simulation
  2026-09-29; hands as they were then, shown burns dead) — `board_equities`'
  per-board marginal does not apply.
- **All-in runout timeline (every game; owner 2026-09-29: the showdown began before
  the rivers had landed, and PLO67's burns + extra cards were "impossible to
  track")**: `_make_runout_plan` (set in `_capture_rabbit`, `t.runout_plan`) =
  when each street is REVEALED and when it is SETTLED (all its cards down). Street n
  is revealed one host pause (`_street_pause`, 0.3–5 s) after street n−1 settles; a
  PLO67 street presents its burn (`RUNOUT_BURN_S` 2.0 s), then — red — one card per
  live hand, clockwise from the button (`RUNOUT_EXTRA_S` 0.6 s + 0.35 s a hand),
  then the boards (`RUNOUT_BOARD_S`); other games settle at the reveal. The
  showdown (`award_at`) starts `RUNOUT_AWARD_BEAT_S` after the river settles
  (other games: `RUNOUT_BOARD_S` after its reveal). The view's `runout.shown_len` /
  `settled_len` / `timing` (ms — PLO67 only; the felt animates with exactly these
  numbers, so client and server never drift) come from the plan; equities and
  made-hand labels follow `settled_len` — never ahead of the cards on the felt
  (`_shared_key` and `_stream_sig` carry it).
- Record: `burns` (the hand's), per seat `hole_seq` + `counts` [flop, turn,
  river]; `_hand_for_viewer` hides `hole_seq` with `hole` (counts are public).
  The replayer shows each seat's first `counts[street]` cards of `hole_seq` (face
  down: that many backs), the burns so far, and "turn · burn A♦ — everyone in gets
  a card" street lines (run-out streets included).
- **Record v2 (every game)**: an all-in runout's `equities` {board length: {seat:
  [board 1, board 2]}} for each street that still had cards to come + `runout_from`
  (`_record_equities`); `_hand_for_viewer` keeps only seats whose cards the viewer
  may see. The replayer (`openHand`) goes on after the last action — one step per
  runout street with the equities under the plates, then the result — and the text
  export prints an "All in · equity" line per street.
- **Verified shuffle**: `SealedDeck(hole=7, burns=3)`; burn j = slot `7n+10+j`
  (`fairdeal.burn_slot`; transcripts carry `burns` only when nonzero — PLO5/PLO6
  unchanged); `hand_id` ends `:plo67`. The browser (`games.fair.js`) allows a hand
  of m cards only its seat's FIRST m slots (a later extra shown early is refused —
  for every game, same result for PLO5/PLO6), checks burn slots, and
  `holeCountProblem`: a live hand holds exactly 4 + the red burns played, a folded
  one no more; a transcript whose `burns` differ from the table's is refused.
- **Felt** (`games.table.js` `planBurns` / `presentBurn` / `growCards`, `#burns` in
  games.html, the PLO67 section of games.css): the burn strip is a column
  left of the boards (desktop / tablet), a short row under them on an upright phone
  (a column took the side seats' tabled-row room), a row left of board 1 on a phone
  on its side. A new burn is shown big over the boards, flipped, captioned ("Red
  burn · everyone in gets a card" / "Black burn · no card"), then drops into its
  slot (`BURN_MS`; in an all-in runout `runout.timing.burn_ms`, slower); a red one's
  cards fly to every live hand after it (`extraAt`; in a runout one hand at a time,
  clockwise from the button — `extraSlot`) and the street's board cards wait for
  both (`boardAt`). The REAL felt renders in Node on `hg_mini_dom.js`
  (`FELT_RUNOUT` in `test_homegame_plo67.py`: re-run games.table.js after
  `boot()`, which stubs the renderer) — a crash in this path once passed every test. At a deal the four go out
  first, then the flop's burn, then (red) the fifth. A growing row keeps its cards
  (a face-up one slides them aside — FLIP with the translate property) instead of
  rebuilding. `#stage.h7`: hero cards 0.82x / 0.33 overlap, fans centred per seat
  (`--mid` inline). Measured with 7-card hands at 375x812 / 375x667 / 768x1024 /
  812x375 / 1366x768: 0 overlaps (`measure_showdown.js` + the burn strip).
  History rows tuck 6-7 card hands (`.mini-cards.many`).
- Club numbers: nothing new — `GAMES` drives the per-game switch / stats.

The bet cap and the uncalled bet (2026-10-02 — `test_homegame_tracking.py`,
`test_homegame_client_windows.py`, `tests/python/engine/test_reach_cap.py`; owner: "I don't
want the bet sizes clipped at all … capped at the pot or your own stack size, whichever
is smaller"):

- **Bet cap**: the home games deal `GameConfig(reach_cap=False)` (`_deal_now_locked`) —
  a bet or raise is capped by the pot (pot limit) and the bettor's stack only, never by
  what the shorter stacks can call, and the floor is never clamped down to cover a short
  stack (`rust_engine/CLAUDE.md`, "Config surface"). Study, the Trainer, training and the
  CFR solver keep the trained rule (`reach_cap=True`, the default everywhere else).
- **The uncalled bet goes back** (it used to be a one-player "Side pot" the bettor
  "won" at the showdown): `runout.uncalled_bet(total_commit)` = the top stake beyond the
  second one (folded stakes count — they were matched). `_capture_rabbit` returns it at
  once — the bettor's `leftover_stacks` (so `display_stacks` all runout long), out of
  `terminal_pot`, `t.uncalled` {seat: chips}, an "Uncalled $X returned to NAME" event
  line — and takes it out of the commits `build_awards` / `display_pots` see, so no pot
  and no award step carries it (the engine's payout already refunds it as a one-seat
  layer: deltas and flows are unchanged). It exists only once betting is over (nobody
  matched the top bet ⇒ everyone else still in is all in), so `live_pots` never sees it.
  The view's `returned` {seat: cents} (showdown phase only) makes `updateMoney` slide
  that part of the bettor's collected bet back to the seat instead of into the pot.
- **Record v3** (`HAND_RECORD_VERSION`): `uncalled` {seat, cents} and `pot_cents`
  without it; the text export prints "Uncalled bet ($X) returned to NAME" after the
  last street's actions; the replayer's `replayState` refunds it at the end (any record
  version — the stacks say what was put in).
- **Grading / Open in Study** stay on the trained rule: the grader clamps a raise's
  chips into the replay's `[min_raise_chips, max_raise_chips]` before scoring, and
  `openInStudy` clamps into Study's `raise_bounds` — a bet above what anyone can call is
  the same bet as the capped one (its excess comes back), so nothing is mis-graded.
- **The covering bet into a short stack's last chips (2026-10-03; owner: "in a real poker
  app, I would be able to bet $20+ and the opponent just calls for their remaining
  chips")**: under the trained rule, against opponents with less than 1bb behind the only
  bet is the COVER (min = max = what they have left), which the Raise gate's dust screen
  used to hide — Study / the Trainer offered only Check. Every table the site builds
  (`common.default_game_config` for Study, the Trainer's three, the home-game grader and
  Hand review's grader) sets `GameConfig.cover_short_bets=True`: the dock shows "Bet X"
  (titled "Puts them all in — a bigger bet would come back uncalled"); only dust under
  bb/100 stays screened. Training keeps the screen (the default; the batched engine refuses
  the flag) and the observation is identical either way, so the network's raise logit
  here is its generalization from nearby spots — training never offered it. Tests:
  `tests/python/engine/test_cover_short_bets.py`, `tests/python/site/test_site_cover_short_bets.py`.

Hand review (2026-10-03 — `tests/python/site/test_public_hand_review.py`; owner: "create a page,
and this should be the one thing that is paywalled because it uses server storage, where you
can drop the zip folder in … checks for duplicate hands … graph your profit and loss along side
your all-in ev … [which] needs to take side pots into account … a hand history list similar to
in the home games where you can pull up a spot into study mode … check all your decision nodes
against the network so that you can easily find your worst played hands"):

- **The page** `/games/review` (`handreview_store.install`, after the home games: the games
  page, client and `/games/api` guard) = a third view of games.html (`#review-view`,
  `games.review.js`, routed by `games.js route()`; links from the lobby's top bar and the
  Study / Trainer tabs via `/me.review`). Every `/games/api/review/*` route needs
  `public._paid` (402 `subscription_required` otherwise); the page itself shows what the
  subscription buys and starts checkout with `{"next": "/games/review"}` (the success page
  confirms it). Deleting your hands is allowed without a subscription.
- **Reading** (`handreview.py`, pure): ClubGG exports GG-network text, one `.txt` per
  session, in a `.zip` (`read_upload`: no extraction to disk, entry / size caps, every read
  capped too — a zip's declared sizes can lie). PLO5 double-board bomb pots only
  (`parse_hand`; other games are skipped with a reason). GG prints each street's betting
  ONCE PER BOARD under each board's header — one round, the copies must agree; the
  `*** SHOWDOWN ***` blocks are board 1 then board 2, one `collected` line per pot (main
  first); the summary's "won ($X)" leaves side pots out (never read it). Opponents are
  ClubGG's anonymous ids (shown as "Player AA11"); the uploader is `Hero`.
- **The money is ClubGG's**: `ledger()` (antes, missed blinds = dead money, bets, the
  uncalled bet, what each collected; a pot that doesn't add up is refused). **All-in EV**
  (`allin_ev`): when nobody acted on a later street and 2+ hands reached the showdown,
  every pot LAYER (`runout.pot_layers`) is worth half x share on each board, `share` = the
  hero's chance on that board against THAT LAYER'S players (`board_equities(dead=…,
  digits=None)` — the all-in hand that can't win a side pot keeps its cards dead; exact over
  the missing cards). Checked against a brute force of both boards' joint runouts paid by
  `build_awards` (2026-10-03: 65 real all-ins, every turn all-in to the cent).
- **The record** (`make_record`) is the home games' hand record built from ClubGG's
  numbers (`kind` "review", `dead_cents` per seat, `study_upto`, `net_cents`,
  `ev_net_cents`, `allin_ev`), so `openHand(…, {url, noLink})` replays it and
  `openInStudy` copies any spot. **The engine replay** (`engine_replay`, the home games' bet
  rule) checks it and becomes the grading job (the HERO's decisions only — other players'
  cards are unknown, and the actor's observation never uses them): dead money is cut from
  the poster's engine stack (the pot is that much short — noted), and it stops at the one
  rule ClubGG doesn't share — a player who CHECKED may raise a short all-in there (the
  engine keeps the TDA rule: call or fold) — so decisions before it are graded and Study
  stops there (`study_upto`; the replayer disables the button past it).
- **Store** (`handreview_store.py`, component "handreview"): `review_hands` (PRIMARY KEY
  (user_id, hand_key) = the duplicate check; the record, the sortable numbers, the job
  until graded, the hand's text zlib'd for re-reading later) and `review_uploads` (progress
  + report). ONE import worker site-wide (`queue_upload`: one upload per user at a time, a
  short queue, body limit `MAX_UPLOAD_BYTES` 25 MB, `MAX_HANDS_PER_USER`
  `$PLO5BP_REVIEW_MAX_HANDS` 200k) and one grader (`homegame.grade_hand` with the served
  model — a placeholder never grades; `PLO5BP_REVIEW_GRADING`, off in the tests). ~35 ms a
  hand to read (the exact equities), grading ~2 s for 350 hands. "Download my data" lists
  the hands; "Delete my account" deletes them (`ACCOUNT_HOOKS["hand_review"]`).
- **Order and dates (2026-10-04; owner: "make sure the hand histories graph is always in
  chronological order … filter to specific date ranges of hands")**: hands go in the order
  they were PLAYED — `CHRONO` = `played_ts` (the time printed in the hand, ClubGG's clock =
  the player's own, read as if UTC), then the hand number as a NUMBER (`length(hand_key)`,
  `hand_key`: ring_999 before ring_1000) — whatever order they were uploaded in; the graph
  (`series`) and the list's Date sort share it. `summary` / `series` / `hands` take `start` /
  `end` (YYYY-MM-DD, whole days on that same clock, both included, either optional; 400 on a
  bad date or start > end); the summary adds every hand's `all_hands` / `all_first_ts` /
  `all_last_ts` and the `range`; a series point is [hand, net, ev, net bb, ev bb,
  played_ts] and its sums start at 0 on the first day. Client: the date bar above the
  numbers (`paintRange`: All time / 7 days / 30 days / This month / Last month / This year,
  counted back from the browser's today, or From / to date boxes = "custom"; remembered in
  `hg.review.range.v1`) narrows the numbers, the graph (dated tooltip, first / last day under
  it) and the list; the hero line, Delete all and the drill card stay about every hand.
- **The network's choice in the replayer** (2026-10-03; owner: "click on the decision node and
  see the network's choice in the hand history without having to put it in the study mode"):
  `openHand` asks `GET …/choice?i=` about the decision just played — Hand review's
  `/games/api/review/hands/{key}/choice` (paid) or a home game's
  `/games/api/tables/{gid}/hands/{no}/choice` (club members with Study's access,
  `public._entitled`) — once per decision, and draws it in the side panel (`#rp-net`: the
  pick, the fold / call / raise mix, the three likeliest sizes, the move that was made).
  `handreview.spot_env` rebuilds the spot Study's way from the record (the actor's cards, the
  boards, every action before it, raises clamped like the grader) — only where the viewer may
  see the actor's cards (else 400) — and `network_choice` runs the served PLO5 model
  (`hg._grading_model()`; 503 without one). The observation is the grader's, bit for bit
  (`test_a_spot_rebuilt_from_the_record_is_the_node_the_grader_scored`).
- **The mistakes drill** (2026-10-03; owner: the Trainer puts you back in the spots you got
  wrong, the biggest blunders most likely first but in a semi-random order, a spot you play
  right comes up less, one you miss again more — and a switch for equal priority):
  `review_mistakes` (migration 2, backfilled from the stored grades; `store_grades` keeps it
  in step and a regrade keeps a spot's learning state while it is still a mistake) = one row
  per decision graded wrong / blunder, `weight` = its learning multiplier. Hand review's
  "Start the drill" = `/?mode=trainer&drill=1` → `POST /trainer/drill/next {prioritize,
  resume}` → the store's `drill_next` (the round, in memory: `CTX.drill`) →
  `TrainerSession.load_drill` (the spot via `spot_env`; the other hands are placeholders,
  never shown). Worst first: weight = severity (score 0 → 1, the top of "wrong" → 0.25) x
  multiplier; a round = a weighted shuffle (Efraimidis-Spirakis); a spot under 1 sits a round
  out with probability 1 − multiplier; best / correct halves it (down to 1/8), wrong /
  blunder → max(m, 1) x 1.5 (up to 4) and the spot comes back later in the same round,
  inaccuracy changes nothing. Equal priority: every spot every round, uniformly shuffled.
  Either way the next spot is from ANOTHER hand whenever one is left (`_spread`). A spot
  replays the hand's real actions up to the decision (owner: "replay all the actions up until
  your decision node"): one frame per action, narrated like the Trainer's opponents (yours
  too: "You check"), `drill_replay` on each, <= 700 ms apart (`DRILL_REPLAY_MS`). Then it is
  ONE decision: graded like any Trainer move, then over (`drill_done`) — the engine stays
  there, no EV estimate, no opponents ("opponent hands are unknown, the network cannot play
  them"), no review / what-if, nothing in the Trainer's stats or recent hands (all of those
  replay a deal that doesn't exist); `resume()` never moves it. Try again (`/trainer/repeat`)
  = practice, straight to the decision, never recorded. Client (`app.trainer.js`): `body.drill-mode`
  shows the drill bar in the work bar (Worst first, remembered per browser; Exit drill), New
  hand → "Next spot", Repeat → "Try again"; the dock's line (`drillNoteHTML`: how it went, the
  network's play and how often it makes it, what you did in the hand — a compact dock keeps
  two lines beside one button) and the Recommendation panel = the spot's node view.

Premium tables pass (2026-09-21 — `tests/python/homegame/test_homegame_premium.py`):

- **The SERVER deals** (`_auto_deal_tick_locked`, table setting
  `deal_delay_secs`; 0 = manual). An API create that omits it gets MANUAL
  (scripted callers/tests stay deterministic); the create dialog sends 5 and
  pre-existing rows migrate to 5 s. It deals only to a table somebody is AT
  (`PRESENCE_WINDOW_S`: ≥ 2 eligible players polled recently) — an abandoned
  table must not keep posting antes. The client never auto-deals.
- **Time bank**: after the base clock the actor burns `Seat.time_bank_left`
  (in-memory; only the seconds used are charged, `TIME_BANK_REFILL_S` comes
  back per hand, capped by `time_bank_secs`). Two timeouts in a row sit the
  player out.
- **Table settings** (`POST /settings`, host): name, ante, buy-in min/default/
  max, seats (between hands; shrink needs the high seats empty), clock, bank,
  deal delay, runout pause, `listed` (link-only tables), `allow_rabbit`.
  Blinds are fixed at creation (they are the chip unit — changing them would
  re-value every stack). The hand in progress keeps the config it was dealt
  with. Also `/transfer_host`, `/show` (table your own cards after the hand —
  the ONLY way a fold-out winner or a folded hand is ever revealed), `/react`
  (whitelisted emotes), `/hands` + `/hands/{n}`.
- **Hand history** stores every dealt hand's cards (`homegame_hands`) and
  filters per viewer with the live rule (own cards + hands tabled at showdown
  or shown); open to every member of the table's club (SEC-008 — the club's
  stats already opened the same hands); nothing about a hand — history, stats,
  the `win` event line — is served while its runout is still revealing
  (`_unpublished_guard`, a lock-free SQL guard every cross-table aggregate uses).
- **Sitting / reloading mid-hand** is allowed for a seat that is NOT in the
  hand (`_dealt_in`); `_sync_idle_stack` keeps `hand_start_stacks` in step so
  the hand's end and the runout-time ledger stay zero-sum. Players holding
  cards still wait (table stakes).
- **Changing seats** (2026-09-28, FEAT-013 — `test_homegame_move.py`): `POST /move
  {seat}` moves a seated player who holds no cards to an empty seat with the whole
  `Seat` (never while the next hand's shuffle is being confirmed — 409); a
  commitment made from the old seat is dropped and games.fair.js commits again from
  the new one. The felt shows a seated player's empty seats as "Move"; someone in the
  hand is moved by the client as soon as it ends (`UI.pendingMove`).
- `events` / `reactions` are small in-memory feeds (dealer lines, toasts,
  emotes). Lobby rows carry seated NAMES only (no emails / user ids).
- **Client** = gated modules (`homegame.GAMES_ASSETS`, each also "deny" in
  `server._PUBLIC_STATIC_POLICY` and a `<script>` in games.html — pinned by
  `test_homegame_client_ui.py`). Since 2026-09-28 (FE-005) `games.ui.js` is the UI
  SHELL (toasts, dialogs, menus, pickers, rail, prefs, top bar, render; shared helpers
  exported as `HG.uikit`) and its windows live in feature modules that load after it
  and add to `HG.ui` (cross-module calls go through `HG.ui` at call time; a module's
  own DOM wiring goes in `UI.onInit`): `games.lobby.js` (lobby, clubs, host a table,
  the club's numbers), `games.history.js` (replayer, Open in Study, hand database),
  `games.seat.js` (sit / chips / automatic chips / requests / player card / leave),
  `games.manage.js` (Manage drawer, Table info). The page links every file as
  `?v=<content hash>` (kept a year, `private`); the views' `client_build` vs the
  page's `<meta name="hg-build">` makes an old page offer a refresh between hands
  (OPS-039). Then: `games.js` (core: state, ordering guards,
  polling, pre-actions, routing — no DOM building, driven headless by
  `test_review_homegame_client_js.py`), `games.table.js` (persistent-node
  felt renderer: every animation is a DIFF of prev→next state, so never
  rebuild the table with innerHTML), `games.play.js` (dock: actions, sizing,
  pre-actions, status strip, hotkeys — built once, updated in place),
  `games.sound.js` (WebAudio synth, no audio files). `games.css` is laid out by
  component (FE-009, 2026-09-28): a CONTENTS index at the top, numbered sections,
  each with its own phone / landscape `@media` blocks right after its base rules, touch
  screens last — add a rule to its component's section, never a dated block at the
  end (the reorder was checked computed-style-identical in the browser). The TABLE's
  rules (stage, felt, cards, seats, bets, the dock) are `games.felt.css` since 2026-10-01,
  shared with Study / Trainer and linked BEFORE games.css (also checked computed-style
  identical); `test_homegame_client_layout.py` reads the two as one. On-felt sizes are
  multiples of `--u` (set by `layout()`); the felt insets in `computeGeom`
  and the seat geometry must stay in step. Player notes/tags and preferences
  are localStorage-only.
- **Page security headers (2026-09-25 — `test_homegame_page_headers.py`)**:
  `/games` and `/games/t/{id}` send `homegame.PAGE_HEADERS` — a CSP with
  `script-src 'self'` (NO inline `<script>`, NO inline event-handler attribute:
  attach listeners, e.g. `h(tag, {onclick: fn})`; no `javascript:` URL, no
  eval), styles/fonts from self + Google Fonts and NO inline style (2026-09-28,
  FE-011: a value the CSS needs — a hue, a position — rides in `data-vars="h:212"`
  and games.js `put()` sets it through the CSSOM), `img-src 'self' data:`,
  `connect-src 'self'`, `frame-ancestors 'none'` + `X-Frame-Options: DENY`,
  `nosniff` (also on the `/games/static` assets), `Referrer-Policy: same-origin`
  and COOP `same-origin-allow-popups` (`openInStudy` fills the tab it opens). A
  blocked script fails SILENTLY in production (a dead button), so the test scans
  the client for inline handlers; a new outside origin (CDN, image host) must be
  added to `PAGE_CSP` and to the test's allow-list. Markup is written ONLY in
  `html` tagged templates (FE-003: every value escaped; `put()`/`h()` take a plain
  string as text) — `test_homegame_client_markup.py` scans every client file for it.
- **Chips in (2026-09-22 — `tests/python/homegame/test_homegame_chips.py`)**:
  `approve_buyins` turns a sit / top-up by anyone but the host or a TRUSTED
  player (`homegame_players.trusted`, `/trust`) into a pending request
  (`LiveTable.requests`, in memory; `/request` approve|deny|cancel; a sit
  request holds its seat; it lapses when the requester's browser is gone;
  switching approval off or trusting the player lets it through). Two
  DIFFERENT automatic-chip features, each off | host | player:
  **auto top-up** (`topup_mode`, `/auto_topup`, per-seat target + below —
  tops UP only, and only once the stack is under the threshold: no rathole)
  and **set stack** (`auto_stack_mode`, `/auto_stack`, the older feature —
  resets the stack to the target before EVERY deal, up or down: ratholing is
  the point). Set-stack wins when a seat has both; players choose through
  `/auto_chips_self {kind: off|topup|set}` and only touch the knobs the host
  left to them; targets are capped by `max_buyin_cents`; while approval is on
  neither runs for an untrusted player (`_auto_chips_allowed`). A top-up sent
  with `queue: true` while holding cards is queued (`queued_topup_cents`) and
  lands when the hand is over; without the flag it is still a 400. Players
  FIND the choice (2026-09-26 — it used to sit only behind the top bar's person
  icon, so "Players choose" looked like no option): the Add chips dialog, the
  Chips chooser and your own seat card carry an "Automatic chips" row
  (`autoChipsRow` / `autoChipsNow` in `games.ui.js`, "Change" opens
  `openAutoChips`), the host's Chips tab says where players set it, and a MODE
  change is said to the table once (`_AUTO_MODE_LINES`, a "settings" event).
- **Live push**: `GET …/stream` (SSE, async generator — never a threadpool
  thread per viewer; the view is built under the lock via
  `run_in_threadpool`) pushes the viewer's state whenever `_stream_sig`
  changes (rev + the clock-driven runout/award steps) and at least every
  `STREAM_HEARTBEAT_S`; `max_events` exists for tests. Every push resolves the
  table through `HUB.get` like a poll (OPS-001: a table watched only through its
  stream counts as active, and a reloaded copy is followed). The client
  (`startLive`) falls back to the 450 ms poll after repeated stream errors.
  Presence (`seen`, fed by polls AND stream pushes) drives server dealing,
  the `spectators` name list and each seat's `present` flag.
- **Reliability (2026-09-28, `test_homegame_reliability.py` / `_grading.py`)**:
  schema = numbered migrations of component "homegame" (`HOMEGAME_MIGRATIONS` on
  `public.Db.migrate`; append only, additive); table settings are ONE list
  (`META`: load, save, create, rollback, migration); ONE process per database
  (`_take_process_lock`, an OS lock on `<db>.homegames.lock` — never run uvicorn
  with `--workers`); the watchdog isolates every step of every table, skips busy
  or idle tables, pauses a table that keeps failing (announced) and reports to
  admins at `GET /games/api/health`; a hand's record/results/flows/grade job and
  a shuffle transcript are OWED writes committed with the stacks by
  `_persist_safe` (one transaction, each in a SAVEPOINT); the app's shutdown hook
  saves what tables owe (run uvicorn with `--timeout-graceful-shutdown`, or open
  streams block the stop). Host-only actions answer 403, races with another
  player 409; every buy-in / window amount must buy MORE than the ante.
- Phone landscape = the `wide` geometry in `games.table.js` (boards side by
  side, hero plate beside the hero cards, no seats along the bottom edge) plus
  the `(max-height: 480px) and (orientation: landscape)` block in `games.felt.css`
  that floats the dock over the felt's bottom corners — keep the two in step.
- **Tracking (2026-09-23 — `tests/python/homegame/test_homegame_tracking.py`)**: every
  hand record (`homegame_hands.summary`) is REPLAYABLE (`actions` carry the
  engine action id + chips + `auto` = the clock decided; seats carry
  `start_chips`; `ante_chips`, `flows`, `grades`). The client replayer is a
  click-through (`openHand` in `games.ui.js`: position k = k actions played;
  `replayState` rebuilds stacks / bets / pot / boards) and `openInStudy` copies
  any position into Study by driving Study's own API (`/reset`, `/config`,
  `/seats` with `stacks_are_starting`, `/cards`, `/action` x k; hero = the
  actor when their cards are visible to the viewer; Study caps at 6 players).
  **Who paid whom** = `runout.money_flows`: per pot LAYER, each contributor's
  chips go to that layer's winners in proportion to what each took (self-flows
  dropped) — fold-outs, scoops, chops, quartering, side pots and dead money all
  follow; checked against the engine's payouts in a randomized sweep. Stored net
  per hand in `homegame_flows` (user ids, chips). **AI grading**: a background
  thread (`_grader_loop`, `PLO5BP_HOMEGAME_GRADING`, OFF in the test session)
  replays each finished hand in a FULL-observation env from its job (the dealt
  deck + exact engine inputs; a deck job is saved in `homegame_grade_jobs` with
  the hand until graded, so restarts lose nothing — OPS-016; a seed is NEVER
  persisted or served), with the model server.py provides
  (`set_model_provider`; none/a placeholder = jobs wait), and
  scores every PLAYER decision with the Trainer's `compute_node_distribution` +
  `score_move(_v2)` from the actor's own seat; clock/away/host actions are not
  graded. Grades land in the record and as `acc_sum`/`acc_n` in
  `homegame_hand_results` (session + lifetime accuracy are SQL sums). Marks
  FOLLOW THE CARDS (owner, 2026-09-26, `_hand_for_viewer`): a viewer sees the
  marks on their own decisions and on hands they can see (tabled at showdown or
  shown) — never on a mucked hand, in the replayer or in anyone's history (a
  player's per-hand accuracy in `_my_hands` is None for a hand you couldn't see,
  and an accuracy sort of someone else's hands only ranks their showdowns). The
  hidden marks still count in every total. Table setting `show_grades` (default
  on) = that rule; off = your own marks only. Lifetime:
  `GET /games/api/my/hands` (sort time|pot|net|accuracy, filter by table,
  paged) and `/games/api/my/stats` (net, accuracy, sessions, head-to-head in
  cents across tables); neither serves a hand its table is still revealing.
  `/games/api/my/series` (+ `/players/{id}/series`, FEAT-012) = the profit
  graph's running net in the same scope, summed like the stats (a session's
  hands in chips, a closed one's ledger): its last point IS the stats' net.
  `openInStudy` opens Study in a NEW TAB: the tab is opened synchronously
  inside the click (after the awaits a browser blocks it as a pop-up) and
  pointed at `/?mode=study` once the spot is loaded; blocked pop-ups fall back
  to a modal with a plain `target=_blank` link — the replayer never navigates.
- **The club (2026-09-24; one per CLUB since 2026-09-25)**: a club is a private
  circle, so stats are OPEN inside it — `GET /games/api/community?club=` (every
  player's hands / net / accuracy, the pairwise `pairs` = "`to` is up `cents` on
  `from`", all sessions — of that club's tables only), `/games/api/players/{id}/
  stats|hands?club=` (= the `my/*` pair for any player; `_my_hands(viewer, …,
  player_id=, clubs=)`; without a club: your OWN numbers = everything, someone
  else's = only the clubs you share). What stays PRIVATE is unchanged: hole
  cards follow the table's reveal rule for the VIEWER (own + tabled/shown) even
  when browsing someone else's history, and no email is ever served. The lobby's
  Players section (`renderClub` in `games.ui.js`) = podium of the top three by
  accuracy (needs `MIN_RANKED` = 20 graded decisions, else "provisional"), a
  card per player, the head-to-head matrix (`openMatrix`) and All sessions.
  **Excluded sessions**: `homegames.excluded` is a SOFT, reversible flag set by
  the CLUB'S OWNER only (`POST …/exclude {on}`, not the host — a host must not be
  able to erase a losing night); an open table is closed first (cash-out), a
  busy hand is a 400. Every stats query joins `homegames` and filters
  `excluded=0` (`_my_hands`, `_my_stats`, `_community`, lobby sessions) — a new
  aggregate MUST do the same — and filter `g.club_id` (`_club_clause`). Nothing
  is deleted; the table still opens by link.
- **Bet spots (`placeBetSpots` in `games.table.js`)**: a bet sits on the rail's
  inward NORMAL at its seat (`railNormal`: flat edges straight across, round
  ends toward their own circle's centre), just clear of the seat's own box —
  NOT "toward the table centre" (that put a corner seat's chips, and the
  hero's, in front of the neighbour on a wide table). Blocked spots swing round
  the seat 8° at a time (toward the centre first) and take the first place free
  of the pot row, boards, hero cards, other seats and already-placed bets; dead
  ahead wins whenever it physically fits. The hero's bet goes out from the hero's
  CARDS in every layout. `layout()` slides the pot+boards block (bounded) so one
  bet's height of felt stays clear under the far seat and over the hero's cards;
  desktop compacts the block (`.board` 5.3u, `#hero-hole` 6.3u = `g.heroCw` —
  `layout()` PUBLISHES the numbers the CSS draws with: `--felt-inset`, `--hero-cw`,
  `--hero-ov`, `--open-cw`; FE-010, 2026-09-28). The street TOTAL and the street tag live on the
  pot's ROW (`#pot-row` grid flanks), not on a line of their own: that line
  appeared with the first bet and shoved both boards down. Sizes that have font
  floors are modelled (`seatBottom`) or measured (`pillSize`) per layout — a
  pure `u` constant is wrong on a small phone. The measuring harness used to
  tune this is `tools/games_preview/measure_bets.js`.
- **Verified shuffle (2026-09-25 — `ui/fairdeal.py` is the SPEC, `static/games.fair.js`
  the player's half; `tests/python/test_homegame_fair*.py`)**: the operator also
  PLAYS in these games, so the threat model is "server + one player together".
  Per hand: the server SEALS a shuffled deck (52 salted per-position SHA-256
  commitments -> `seal`) before anyone contributes; each seated browser commits
  to a 32-byte random number bound to `hand_id|seal|seat`; at the deal the list
  is LOCKED (dealt-in, PRESENT devices only) and published; a browser reveals
  ONLY after it has seen its own commitment in that list under the seal it
  committed to; `cut` = hash of the reveals drives a Fisher-Yates permutation
  (SHA-256 counter stream, rejection sampling) and the engine deals
  `F[slot] = D[perm[slot]]`. Every card a viewer is shown carries
  `{slot, pos, salt}` in `fair.hand.open` — built FROM the viewer's visible
  cards, so it can never open a card the reveal rule hides; mucked hands stay
  sealed. Guarantee: if YOUR device contributed, the deal was uniform and
  unaltered whatever everybody else did. It does NOT stop the operator from
  looking at cards server-side (only mental poker can), and deck COMPOSITION is
  proven only for opened cards — say so, never oversell it. Invariants:
  the slot map is public and mask-independent (seat s hole k = `5s+k` for EVERY
  seat index, board A `5n..`, board B `5n+5..`; pinned in Rust + Python tests);
  a committed device that does not reveal within `FAIR_REVEAL_S` VOIDS the
  attempt — new deck, new seal, the absentee barred for that hand, ALWAYS
  announced (`fair` event + `voids` in the transcript + `void_counts`): a
  withheld number is a visible re-roll, never a silent one (two in a row =
  `FAIR_PENALTY_HANDS` out of the shuffle). A sealed deck whose cut may be known
  is never dealt later (pause / close / resize void it). Tables nobody's browser
  takes part in (scripts, tests, bots) deal AT ONCE, exactly as before: the wait
  exists only for users in `fair_capable`. Transcripts persist compactly in
  `homegame_fair` (key + deck; commitments are recomputed) — the key and deck
  are secrets at rest, like the hole cards already in `homegame_hands`. The
  grader replays `hand_deck` (memory only), not a seed. Kill switch
  `PLO5BP_HOMEGAME_FAIR=0`; an engine built before `reset_with_deck` turns
  `FAIR_ON` off by itself (old dealing, client shows "unverified") — so a
  production ship MUST rebuild the engine (`scripts/deploy_prod.sh`). The spec
  is pinned by a known-answer permutation and a Node run of the browser verifier
  against Python transcripts; changing either side is a PUBLIC spec change.
- **Table UX round 2 (2026-09-22 — `tests/python/homegame/test_homegame_table_ux.py`)**:
  the create dialog takes a BIG BLIND and an ANTE IN BB (no stake presets, no
  small blind — `sb_cents` is kept in the wire format/DB for the old rows and
  defaults to half a bb; displays say "$1.00 bb · ante $3.00"). The host TAPS a
  reserved seat (or a seated player's "$" badge / the Chips tab's Edit) and gets
  the request dialog: approve as asked, approve a DIFFERENT amount
  (`/request {amount_cents}` — validated like a buy-in, announced "approved for
  $80 (asked $150)"), approve + trust, decline; the host's view carries
  `seat.request`. `allow_rathole` (host switch, default off) lets a player TAKE
  CHIPS OFF the table: `/remove_chips {amount_cents, queue}` — whole cents to
  leftover + a `cashout` ledger row (the move set-stack makes), never below an
  ante + 1 bb (that is leaving), queued while holding cards
  (`queued_remove_cents`, re-checked when it lands). LEAVING mid-hand plays the
  hand out — `leave_after_hand`, `seat.leaving`, `/stay` to cancel — and cashes
  out when it ends (`_apply_leaves_locked` runs with the deferred work); the old
  fold-now behaviour is `/leave {now: true}` (kicks/Remove use it); a leaver
  whose browser is gone counts as away so the clock never waits on them. The
  rabbit button lives ON THE FELT in the 2x2 gap of the undealt cards
  (`placeRabbit`, "Click to reveal"), the host has Start/Pause in the top bar
  (`#tb-run`). Award animation (ClubGG-style): `_capture_rabbit` names the POTS
  deepest-first ("Side pot N" … "Main pot", `t.pots`, served as `pots` while the
  runout blocks) — `runout.display_pots` / `pot_groups`: pot LAYERS with the same
  eligible players are ONE pot (a folded player's chips are dead money in it,
  never a side pot of their own; 2026-09-25) — and `runout.build_awards` tags
  each award step with its `pot` (the chips are still split layer by layer,
  exactly like the engine; `test_merged_pots_pay_every_seat_exactly_what_the_layers_do`).
  The client shows the pots as inline pills that REPLACE
  the pot pill (same height — the boards must not move at showdown), highlights
  the active pot, counts each one down as its halves are paid, flies the chips
  from THAT pot, prefixes the caption with the pot name, and the layout reserves
  room under the boards for the caption (`captionHeight`) so it never lands on
  the hero's cards (it wraps on portrait phones).
- **Clubs (2026-09-25 — `test_homegame_clubs.py`; owner request)**: home games
  used to be admin-granted per account (`homegame_access`); now every signed-in
  user has them and a CLUB is the private circle. Tables: `homegame_clubs` (id,
  name, owner, `invite_code`, `approve_joins`, `is_main`), `homegame_club_members`
  (role owner | admin | member), `homegame_club_requests` (pending | approved |
  declined), `homegames.club_id`. EVERY table belongs to one club and every table
  endpoint goes through `_table_for` → `_table_access`: a non-member gets 403
  `{"error": "club", "club": {id, name}, "request", "retry_in"}` (the client offers
  "ask to join"); the stream re-checks on every push. Anyone may start a club
  (`MAX_CLUBS_OWNED` = 5) and any member may host in it (`club_id` on create; none
  given = the main club if you are in it, else your oldest). Joining: the invite
  link `/games/join/{code}` (`/games/api/invites/{code}[/join]` — straight in, or a
  request when the club asks first), or "ask to join" from a table link
  (`/games/api/clubs/{id}/request`). The owner and admins see requests (masked
  email) in the lobby and at the club's tables and decide
  (`/games/api/clubs/{id}/requests/decide`); a declined request may ask again
  after `JOIN_RETRY_S`. Roles (`/games/api/clubs/{id}/members`): the owner makes
  admins, hands the club over (they become an admin) and removes anyone; an admin
  removes members; nobody is removed (and nobody leaves) while they have a seat at
  one of the club's open tables or host one (`_busy_in_club`); the owner can't
  leave. The owner renames, switches "ask me first" and makes a new invite link
  (admins too). The MAIN club (`is_main`) = the site's original circle:
  `_migrate_clubs` (every start, idempotent) moves every table with no club into
  it with everyone who hosted, sat or had the old flag, named "<owner>'s club";
  the /admin switch (`public._GAMES_ACCESS_HOOK` / `_GAMES_MEMBER_HOOK`) now adds
  to / removes from it. Lobby: `GET /games/api/tables?club=` = the viewer's clubs,
  that club's tables (+ your seats elsewhere, tagged), its sessions and — managers
  — its requests; the client remembers the club per browser (`hg.club.v1`,
  `setClub`), shows a club bar (switcher, Invite, Members) and, in no club yet, a
  welcome (start one / join with a link). Signed out, `/games`, a table link or an
  invite link is a script-free sign-in page (`_invite_response`, `INVITE_HEADERS`)
  whose Google sign-in comes back to it (`_safe_next`: those three shapes only).
- **Profile pictures, the host's last settings, bubbles (2026-09-26 —
  `test_homegame_profile.py`)**: a player uploads a picture from the lobby's
  account chip or the table's ≡ menu (`openAvatar`): the BROWSER crops the middle
  square and re-encodes it at 256 px WebP/JPEG (`squareDataUrl` — small, and the
  photo's EXIF/GPS is gone), `POST /games/api/me/avatar {data_url}`. The server
  has no image library: `_image_info` reads the PNG / JPEG / WebP header only,
  and the upload must match its declared type, be 16-1024 px and <= 200 KB
  (`DELETE` removes it). Stored in SQLite (`homegame_avatars`, in the nightly
  backup); served by `GET /games/api/avatars/{uid}?v=<content hash>` to the user
  and anyone sharing a club (else 404) with nosniff + a `sandbox` CSP + immutable
  caching. Every person object carries `avatar` (seats, chat, hand records, club
  players / members / requests, `my_avatar` on the lobby + table views; cached
  per user in `_avatar_url`); the client's `avatar(name, key, cls, url)` keeps
  the initials under the picture and drops a picture that fails to load. A new
  picture bumps the rev of every table the user sits at. HOST SETTINGS are
  remembered per host (`homegame_host_prefs`, amounts in big blinds): saved at
  create and after every host setting change; the create dialog pre-fills from
  `GET /games/api/host_prefs` ("Use defaults" resets), and a create with
  `remembered: true` also applies Manage's settings (grades, rabbit, runout
  pause, automatic chips) through the host endpoints — scripts that don't ask
  get plain defaults. Chat and emote BUBBLES live on the `#fx` layer anchored at
  the seat — the viewer's own at the top of their hole cards, which sit above
  every seat and used to hide their own emotes.
- **Readiness pass (2026-09-25 — `test_homegame_clubs.py`,
  `test_homegame_client_reconnect.py`, `test_homegame_short_hands.py`, new
  cases in `test_homegame_table_ux.py`)**, after hours of bot play in the preview:
  - A server restart mid-hand voids that hand chip-neutrally (stacks persist at
    hand end only); `_load_table` now says so in the table's feed ("Hand #N was
    cut short …"). Identical pending chip requests are deduplicated
    (`_request_locked`); closing a table keeps its host
    (`_cash_out_seat(..., closing=True)` skips the host hand-over).
  - Client: `#connbar` "Connection lost — reconnecting…" + a "Back online"
    toast; after the live stream gives up (a restart) a successful poll re-opens
    it past `G.streamRetryAt` (30 s, doubling to 5 min); `HG.ui.onMyTurn` closes
    an overlay rail / toasts "It's your turn" when a drawer or modal hides the
    action buttons; toasts are deduplicated. On the COMPACT layout (≤ 760 px
    wide or ≤ 700 px tall — a 1366x768 laptop's browser is compact too) the R/B
    hotkey works like the Raise button: the first press opens the sizing panel,
    the second bets (it used to bet the minimum unseen); arrows / 1-6 open it
    too. The ≡ menu has "React" (a phone has no React button).
  - **Showdown layout (phones)** — `fitSeats()` runs after every layout and
    render, all reads first: a tabled row that would touch the boards / pots /
    caption block fans tighter (`--ov` 0.3 → 0.56 of a card; the first card has
    no negative margin, so a row is centred on its seat) and moves to the stage
    edge; a showdown label slides outward off that block, and a label that still
    lands on another seat, a tabled row, the hero's cards or more than a corner
    of a board is hidden (`.seat-hand.crowded` — the cards and the caption still
    tell). The dealer disc sits UNDER the seats (z 3) and fades (`.covered`)
    while a label or row covers it. All-in EQUITIES are on the badge line
    (`.seat-badge.k-eq`, in place of "All-in"): the old `.seat-eq` above the seat
    was always under the tabled cards. Felt labels are short (`shortHand`: "Js
    full of 4s", "Two pair 10s & 2s", "K-high flush"…; pinned against every
    `hand_describe._fmt` phrase); the caption, dock and history keep the full
    wording. `fitPots()`: pots wider than the gap between the seats (and labels)
    beside them drop their chip icons (`#pots.tight`), then scale down (≥ 0.7).
    A phone-sized box (< 520 px wide, ratio < 1.3) keeps the upright table (the
    flat one drew every card ~1/3 smaller on an SE with Safari's bars); the
    upright table stays ≤ 0.74 wide per unit of height (squarer lifts the side
    seats onto the boards — tried 0.9, worse). Short upright phones (≤ 700 px)
    get a tighter dock; ≤ 620 px hide the dock's Sit out / Add chips / Leave row
    (all three are in the seat menu, and the status strip offers I'm back / Add
    chips / Stay whenever they matter). The check behind all this:
    `tools/games_preview/measure_showdown.js` (overlap pairs per runout
    street / award step at every screen size; 0 at 360-430 px phones, tablet,
    desktop; an SE-sized 375x560 keeps a few corner touches).
  - **Side pots while the hand is played + pot hover (owner request)**: the
    view carries `live_pots` during `in_hand` (`homegame._live_pots` =
    `display_pots` of total − street commit: the antes and finished streets;
    this street's bets stay in front of the players until the round closes, as
    on any site). Same names and order as `t.pots` (`_named_pots`), so the
    runout takes over in place; an uncalled excess only exists once betting is
    over (and is returned then, never a pot — 2026-10-02), so it never shows mid-hand. Two or more pots take the pot pill's place
    in `#pot-row` (`#live-pots`, `renderLivePots`; `#pot.split` hides the pill;
    bets fly into `potAnchor()`; the bet spots are re-placed when the row changes
    shape; `fitPots` leaves room for the street total on both sides). Hovering
    any pot pill (a tap on a phone = a 3.5 s toggle) lights up its `eligible`
    seats and dims the rest (`#stage.pot-focus` / `.seat.pot-in`, `focusPot` /
    `applyPotFocus`, re-applied after every render by pill index); the single
    pot and the main pot = everyone still in the hand; a pill's title names its
    players; phones show short names ("Side 1", "Main").
- **Backend backlog pass (2026-09-28, docs/improvements `HGB-*`/`FEAT-*`; tests
  `test_homegame_code.py`, `_money.py`, `_features.py`, `_concurrency.py`)**:
  - **Code layout (HGB-006)**: `homegame.py` = tables + the hand (state, dealing,
    clock, shuffle, money, view, grading) + entry points; beside it
    `homegame_schema` / `_people` / `_clubs` / `_stats` / `_pages` / `_routes`
    (`SPLIT_MODULES`). A part reaches EVERY home-games name as `hg.X` inside its
    functions (looked up at call time) and `homegame` re-exports all it defines —
    patch `homegame.X`, never a part's copy (a bare name in a part would dodge the
    patch; `test_the_parts_reach_every_home_games_name_through_homegame` fails).
    A part has no import-time side effects. Process state is ONE `HomeGames`
    object (`hg.CTX`; `use_context(ctx)` swaps it; `HUB`, `_WATCHDOG_THREAD` … are
    aliases on the module class). `LiveTable` keeps per-hand fields in a
    `HandState` (`t.hand`, property aliases; `HAND_FIELDS`), replaced at each deal.
  - **Locks (HGB-018)**: order hub loading → hub → ONE table lock → DB; every
    `*_locked` function asserts its table lock under pytest (`LOCK_CHECKS`).
  - **Money**: ledger kinds `LEDGER_IN` / `LEDGER_OUT` (+ `hand_no`, schema v8);
    `reconcile_ledger` (checked at start + in the site's /health), stats sum
    CHIPS per session and use the ledger for closed ones (OPS-015); `settle_up` =
    fewest payments (exact subset search ≤ 12 players); receipts
    `GET …/ledger/me|{user_id}`; `…/hands/export` (text/JSON, `homegame_export.py`).
  - **Names**: a player's chosen table name (`/games/api/me/name`, `homegame_names`,
    v9) and per-club nicknames (`/clubs/{id}/nickname`) — `_display_name`; NEVER an
    email (masked `f•••@c•••.com` only on join requests). Clubs can be ARCHIVED
    (`/clubs/{id}/archive {on}`, owner, v10 `archived_at`; soft, restorable).
  - `since=` (all/7d/30d/90d/365d/month/year/ISO date) on community / my / player
    stats + hands. The table view says why no hand is coming (`deal_blocked_reason`);
    during the countdown only the HOST may `/deal` (403 for others); the view's
    `can_browse_hands` is the one browse rule (SEC-008).
  - **Limits**: `PLO5BP_GAMES_RATE` (per user req/s,burst; off under pytest) +
    `API_COST`, `JOIN_RATE`, `AVATAR_RATE`, ≤ `MAX_STREAMS_PER_USER` live streams;
    the stream's heartbeat is an SSE comment (`: ping`), the view is pushed only
    when `_stream_sig` changes (presence is part of it).
- Preview harness (tracked): `tools/games_preview/` — launch
  entry `games_preview` (public build + dev login + temp DB on :8772) and
  `bot.py` (scripted guests).

Per-user state plumbing (matters when touching server.py/trainer.py):

- The app factory (BE-007): `server.create_app(settings=None)` builds an app
  from the environment with a `server.Site` holding its state (models registry
  `formats`, `model_admin`, GTO host, trainer router, study sessions, format
  gate, static mount, health checks; public build: `db`, `registry`,
  `homegames`) — `app.state.site`. Importing server.py builds one
  (`plo5bp.ui.server:app`). ONE site is current (`current_site()` / `use_site()`); the
  module's old names (`app`, `MODEL`, `FORMATS`, `GTO_HOST`, `PLO5BP_PUBLIC`,
  `trainer_router` …) read it, like `homegame.CTX`. public.py re-reads its
  settings and opens its database in `install()` (importing it opens none);
  homegame.install re-reads its env settings and starts its own context.
- server.py's `session` is a **proxy** (`_SessionProxy`) over
  `_current_session()` — resolver installed by public.py returns the
  signed-in user's own `Session`; local build falls through to the current
  site's single default session. Don't reassign `session`; mutate attributes
  (as all existing code does).
- trainer.py routes resolve their `TrainerSession` via `_ts()` +
  `set_session_resolver` the same way (default = the router's own instance;
  per-user stats persist to `data/trainer_stats/u<id>.json`).
- Trainer sampling seeds the GLOBAL torch RNG — every seed→sample region
  must hold `trainer._TORCH_RNG_LOCK` (opponent sampling + `_rollout_ev` MC
  block do). Keep that invariant if adding sampling paths.
