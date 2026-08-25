# The centralized critic (training with x-ray vision)

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

Watch a televised final table and you have a luxury nobody at the table had: the hole-card graphics. With every hand face-up, you can see instantly that the hero's river call was hopeless, or that the villain's shove was pure air. You're not a better player than they are — you're just grading with more information. The centralized critic is exactly that trick, built into training.

Self-play training here uses two networks with two different jobs. The actor is the player: it sees only what a real player could see — its own cards, the boards, the betting so far — and it chooses actions. The critic is the grader: it estimates how good each situation truly was, and those estimates feed the "advantage" calculation, the learning signal that tells the actor which decisions to lean into and which to drop.

The key move is that the critic never plays a hand. Because it only judges, it is allowed to peek at what's hidden — every opponent's exact hole cards.

## Why peeking makes learning faster

Poker results are drenched in noise. From the hero's seat, "got it in great and got unlucky" and "got it in bad" can look identical; only the cards you can't see separate them. A grader restricted to hero's view has to average over all that hidden possibility, so its judgments come out blurry. A grader who sees every hand can say precisely how good the spot really was. Sharper judgments mean cleaner advantages, and cleaner advantages mean the actor learns real lessons faster instead of chasing variance.

## Why nothing leaks into live play

The obvious worry is that this teaches the bot to cheat. It can't. The hidden cards flow only into the critic, and the critic's output is only ever used for grading — it never picks an action, and it isn't consulted when the trained model plays or gives advice. The actor's weights are shaped by which of its own honest, information-legal decisions earned good grades. Once training ends, the critic can be set aside entirely and the actor stands alone, still seeing only what a player is entitled to see. Researchers call this pattern centralized training with decentralized execution: concentrate information where it's safe (the grader), keep execution honest (the player).

**In this project:**

- The critic receives the same 1,171-number observation the actor gets, plus an extra 260-number block encoding every opponent's exact hole cards (5 hero-rotated slots × 52) — **1,431 numbers** into the critic torso.
- The actor keeps its own smaller "display" value head — the observation-only estimate the UI shows.
- The critic's all-cards estimate appears in the trainer review as "true EV", alongside the actor's blind view.
