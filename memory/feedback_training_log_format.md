---
name: Training-monitor status format — canonical one-liner
description: When reporting a training log update, use exactly: `uN (Xseats <block>): F/T/R=f/t/r v=V H=H kl=K` followed by an optional one-line note if anything significant happens
type: feedback
originSessionId: 50714cf5-79d0-4cbd-99d6-23db3848d18e
---
Canonical format for every per-update acknowledgement when monitoring
a training log:

```
u<N> (<X>seats <block>): F/T/R=<f>/<t>/<r>  v=<V>  H=<H>  kl=<K>
<optional one-line note if something significant>
```

- `u<N>`: update index from the log line (e.g. `u0`, `u25`).
- `(<X>seats <block>)`: seat count and block name from the log
  (`clubgg`, `clubgg_deep`, or `deep`). Block already encodes the
  stack distribution — don't separately classify shallow/mid/deep.
- `F/T/R`: the `bonus%(F/T/R)= f/t/r` triple (flop/turn/river
  retroactive-bonus qualifier rates).
- `v`, `H`, `kl`: the three core stability metrics, in that order.
- Note line only when something is worth flagging — sudden H drop,
  KL spike, v explosion, block transition, anomalous F/T/R shape.
  No note is the default when metrics look healthy.

**Why:** The user explicitly specified this exact shape. Compact
enough to scan at glance across many updates; carries the four
signals (block-context, F/T/R, v, H, kl) needed to spot regressions
without scrolling back to the raw log line.

**How to apply.**
- Use the literal block name from the log (`clubgg`, `clubgg_deep`,
  `deep`) — not derived stack tiers.
- Example healthy: `u25 (6seats deep): F/T/R=8.7/8.4/7.4  v=12.3
  H=0.11  kl=+0.013`
- Example with note: `u80 (3seats clubgg): F/T/R=2.5/3.1/4.0
  v=48.2  H=0.04  kl=+0.041 — H near gate-collapse zone, watching`
- Skip the line entirely only if you've already acknowledged the
  same regime in the last few updates and nothing changed.
