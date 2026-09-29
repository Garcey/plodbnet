# Home games — ideas for what to build next

Written 2026-09-25, after the readiness pass and the clubs rebuild. Ordered within
each group by how much I think it would matter to a real night with friends.
★ = my top picks. ✅ = shipped (with the date), ◐ = partly.

Shipped since this list was written:
- ✅ **PLO6 tables** (2026-09-26) and ✅ **PLO67** — face-up burns, a red burn deals
  everyone one more card (2026-09-27): chosen per table at creation, club stats per game.
- ✅ **Remembered host settings** (2026-09-26): a new table starts from the settings of
  the last one you hosted.
- ◐ **Deploys that spare the hand in play** (2026-09-28): a deploy now refuses to
  restart while a hand is being played or a game is running (FORCE=1 overrides), a new
  model is swapped in from /admin without any restart, the grader drains its queue on
  shutdown, and /admin can post a maintenance notice first. Still open: pausing the
  tables automatically and restarting between hands.

## Keeping the night smooth (operations)

- ◐ ★ **Deploy without killing the hand in play.** A restart still voids the hand
  in progress (chips go back), but the deploy now waits for it (see above). The rest:
  pause every table first, wait for the hands in play to finish, then restart — so an
  update can go out even during a game.
- ★ **Resume a hand after a restart** instead of voiding it: persist the
  engine's action log per decision and replay it on load (the engine is
  deterministic, so this is mostly bookkeeping).
- **Install as an app (PWA).** "Add to Home Screen" with a manifest gives a
  full-screen table with no browser bars — the single biggest win for small
  phones (an iPhone SE with Safari's bars is the cramped case today).
- **Push notifications**: "it's your turn", "a club table just opened",
  "you were let into the club" — reaches a phone that is locked.

## Clubs

- ★ **Club leaderboard periods**: this month / this season / all time, with a
  season reset (keeping the history) — keeps the podium interesting.
- ★ **Cross-session settle-up**: one running balance per pair of players across
  every night in the club ("Dana owes you $42 overall"), with a "settled" button.
- **Club table defaults**: stakes, clock, rabbit, approval… set once for the
  club, used by every new table. (◐ Per HOST this exists since 2026-09-26: a new
  table starts from the settings of the last one you hosted.)
- **Scheduled games**: "Friday 8 pm" with RSVPs and a reminder.
- **A club wall**: announcements and chat outside the tables.
- **Invite options**: single-use or expiring links; let members (not only
  admins) share the invite link.
- **Club profile pages per player**: profit over time and accuracy trend charts.
- Disband / archive a club (today the owner can only hand it over).

## At the table

- ★ **Run it twice** on all-ins — a home-game favourite, and the engine already
  runs all-in boards out.
- ◐ **Other formats**, selectable when creating a table: ✅ PLO6 (2026-09-26) and ✅
  PLO67 (2026-09-27) are live; PLO4 (the engine exists and is tested) and NLH (the
  engine exists) are not offered yet.
- **Regular (non-bomb-pot) hands with blinds and a preflop**, alternating with
  bomb pots ("bomb pot every button").
- **Straddles / double-board-only-on-button variants** for house rules.
- **Waitlist** when a table is full.
- **Share a hand**: a link that opens the replayer on that hand for club members.
- **Hand history export** (PokerStars-style text) for other tools.

## Coaching (the network)

- ★ **After-session review**: each player's three biggest mistakes of the
  night by the grades, one click into Study.
- **"Hand of the night"**: biggest pot, best call, luckiest river — a fun recap.
- **Accuracy trend** per session and per street.
- **Optional live hints for practice tables** (off for real games — a club
  setting, never on by default).

## Polish

- A loading skeleton for the table page (the lobby has one now).
- Accessibility: switches and pills announced to screen readers, visible focus
  rings, larger touch targets on the smallest phones.
- Sound options per event (chips, turn alert, chat).
- Colour-blind friendly card faces (the four-colour deck exists; add patterns).
