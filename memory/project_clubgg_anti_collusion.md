---
name: ClubGG anti-collusion card visibility rule
description: When ClubGG hides hero's hole cards postflop — corrects extract.py:96-103, kills Task C
type: project
originSessionId: 6609b555-9a44-45b5-aaec-c1043b436873
---
ClubGG hides hero's hole cards from hand start (preflop) until it first becomes hero's turn on the flop. Once it becomes hero's turn on the flop, the cards turn face-up and STAY face-up for the REST of the hand — through every subsequent action, banner animation, and street, including AFTER hero CHECKs / bets / etc.

The visible→hidden transition only occurs at hand boundary (cards hide for the next preflop). It does NOT fire on hero's mid-hand action.

**Why:** Empirical correction from user 2026-04-26 (verified across debug_1777141614367–debug_1777141624836 burst: cards stayed face-up the entire 10.5s while it was fastaf's turn, after hero had already acted earlier). The `extract.py:96-103` comment claiming "hero's hole cards are visible ONLY during hero's turn (postflop)" is WRONG — they stay visible AFTER hero's first turn through showdown.

**How to apply:** Do NOT use a hero hole-card visible→hidden diff as a positive signal for hero's CHECK or any mid-hand action. The Task C `_hero_hole_just_hid` helper in `python/plo5bp/ocr/events.py` is dead code in production — the diff never fires mid-hand. Look elsewhere for hero's action signal (e.g., the yellow CHECK banner observed in `screenrecords/frames/debug_1777131926.png`, analogous to `has_bet_banner`). When updating extract.py:96-103 comments, the correct rule is: hidden preflop → face-up on hero's first flop turn → face-up through showdown → hidden again at next hand start.
