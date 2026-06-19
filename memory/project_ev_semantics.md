---
name: EV semantics — from decision point forward, sunk costs excluded
description: Project-correct EV definition for plodbbot value-head outputs and recommendations
type: project
originSessionId: 5fea7462-79e9-4e59-936c-fb00309f5ed1
---
EV in this project is defined as the **expected chip change from the
decision point forward**. **All chips already in the pot are sunk
cost** — antes, prior calls, AND any prior bets the hero made
earlier in the same hand. The principle applies recursively at
every decision point:
- River decision: flop+turn bets/calls are all sunk.
- Turn decision: flop bet is sunk.
- Flop decision: ante is sunk.

Only chips that *could still go in* from this point forward count
toward EV.

Concrete examples the user gave:
- Hero on the flop with 0% chance to win → EV(check) = $0, not the
  negative ante share.
- 22223 in heads-up bomb pot, river check-down: tiny chance both
  remaining cards on a board pair the threes (trip 3s = best hand)
  yields ~half-pot win. EV(check) should be slightly positive
  (probably <$1 but >$0), not negative.

**Why:** This is the user's mental model of what the value display
should mean — what does this action *gain me from here*. The
training reward signal in code (`payouts = won − total_commit`) is
chip delta from hand-start; that's a constant offset (=
total_commit at decision point) below the user-correct EV. The
displayed `value_bb` therefore lags the user-correct EV by the
already-committed share. Worth flagging when proposing fixes — the
display might want to add back `total_commit_bb` to match the
user's definition, or training might want to exclude pre-decision
commits from the reward.

**How to apply:** When diagnosing value-head outputs or
recommendation scoring, frame "expected" outcomes against this
forward-looking EV. Don't claim that "-1.5 BB is right because the
ante is sunk" — under this definition the ante is excluded entirely
and the floor is $0 for any check-down terminal line.
