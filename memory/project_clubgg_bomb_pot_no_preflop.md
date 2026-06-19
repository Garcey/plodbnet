---
name: ClubGG bomb pot — no preflop betting, action starts at flop
description: Bomb-pot games skip preflop betting entirely; hands start at flop street with antes already posted. STREET_NAMES{0: "preflop"} exists in code but is not a live betting state for this game.
type: project
originSessionId: 7b62bece-8e86-454f-89e8-27566d72af24
---
This codebase models PLO5 **bomb-pot** double-board hands (the format ClubGG
runs). There is no preflop betting round — every player antes in and the hand
begins directly at flop street.

**Why:** It's the bomb-pot rule. Don't reason about hero "preflop decisions"
or design UX/messages around a preflop stage. The first decision point is
always on the flop.

**How to apply:**
- Don't include "preflop" rows in tables describing required cards / decisions.
- When writing user-facing copy ("place the X cards"), the first betting stage
  is flop, not preflop.
- `STREET_NAMES = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}` in
  server.py is engine boilerplate; engine street index will be ≥1 in any
  state that has a current actor in this game. Code that branches on
  `street == 0` is dead but harmless — don't remove without verifying.
- Hero hole cards are still required for *any* betting (used by the model
  to evaluate the flop), so the "hole cards missing" gate fires at flop.
