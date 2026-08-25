# One-hot and multi-hot encodings

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

Neural networks eat numbers. Poker is full of facts that aren't numbers: which street we're on, which cards I hold, who has the button. Before the network can see any of it, someone has to decide how those facts become numbers — and the obvious way is a trap.

The obvious way is to number the categories. Flop is 1, turn is 2, river is 3. Clubs are 1, spades are 4. The problem is that numbers carry baggage: order and distance come built in. Encode the suits as 1 through 4 and the network inherits the claims that spades are "more" than hearts and that hearts sit "closer" to diamonds than to clubs. None of that is true. The network doesn't know the numbering was arbitrary; it will burn real capacity discovering that the relationships it was handed are lies, and it may never fully unlearn them.

The clean alternative: give every possible value its own slot, and light up the slots that apply. To say which street it is, lay out one slot per street and light exactly one. That's a one-hot encoding — one slot hot, the rest cold. To say which cards I hold, lay out all 52 slots, one per card in the deck, and light the ones in my hand. Several slots on at once: a multi-hot.

Think of a team sheet with a checkbox beside every name. "Who's captain today?" is one-hot — exactly one box ticked. "Who's on the pitch?" is multi-hot — eleven boxes ticked. Nobody would compress that sheet down to "player number 7," because 7 says nothing about who that is — and worse, it whispers that player 7 sits somewhere between 6 and 8.

The payoff is that slots assert nothing. A lit slot doesn't claim its card is bigger, closer, or later than any other — it's just its own switch. Whatever the ace of spades really means — the flushes it makes, the hands it blocks, how it plays on different boards — the network learns from millions of hands of experience, instead of inheriting a fake geometry it has to fight. The encoding costs more slots than a single number would, but every slot is honest.

**In this project:**

- The hero's hand and both boards are 52-slot multi-hots. The model meets the ace of spades as "slot 51 on" and learns its meaning entirely from experience.
- The current street is a 4-slot one-hot.
- The critic — the training-time evaluator — additionally receives a 5×52 multi-hot block spelling out every opponent's exact cards.
