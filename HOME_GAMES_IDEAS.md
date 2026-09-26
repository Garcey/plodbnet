# Home games — ideas for what to build next

Written 2026-09-25, after the readiness pass and the clubs rebuild. Nothing here
is built yet. Ordered within each group by how much I think it would matter to a
real night with friends. ★ = my top picks.

## Keeping the night smooth (operations)

- ★ **Deploy without killing the hand in play.** Today a restart voids the hand
  in progress (chips go back). A "graceful deploy" would pause every table
  first, wait for the hands in play to finish, then restart — so an update can
  go out even during a game.
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
  club, used by every new table. (Per HOST this exists since 2026-09-26: a new
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
- **Other formats**: PLO4 / PLO6 double-board bomb pots (engines exist and are
  tested) and NLH (engine exists), selectable when creating a table.
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
