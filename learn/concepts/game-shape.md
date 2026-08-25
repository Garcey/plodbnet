# Game shape: PLO5 double-board bomb pot

*Part of the [master doc](../MASTER.md) concept library. Plain-English explainer.*

What the model is trained to play, in one page.

## Structure

- **Variant:** `plo5_double_bomb` — Pot-Limit Omaha with **five** hole
  cards, **two** boards, bomb-pot rules.
- **Antes, no blinds.** Default `GameConfig`: 6 seats, starting stack
  often ~20 bb for the shallow tier, `ante = 3 bb`, `bb = 10_000` chips.
  Every seat posts the ante; there is no small/big blind and **no preflop
  decision**. Both boards deal a flop and action starts there.
- **Pot-limit.** A raise may add at most `pot + to_call` on top of the
  call. The sizing head's top anchor ("pot") is exactly that cap when
  deep, and a jam when short.
- **Hand construction.** Best hand uses **exactly two** of five hole
  cards plus three board cards, evaluated independently on each board.
  Each board takes half the pot; scooping both halves is the strategic
  object.
- **Showdown / folds.** Standard elimination; side pots when stacks
  differ. Early all-ins before river are graded by EV runouts in training
  ([→ reward accounting](reward-accounting.md)).

## What the policy outputs

Not the legacy 8-way discrete menu (Fold / CheckCall / five bet-% /
AllIn — still used for UI/tests as `NUM_ACTIONS = 8`). Training uses a
**factored hybrid**:

1. Gate ∈ {Fold, CheckCall, Raise}
2. If Raise: one of 11 pot-fraction anchors, optional Beta refine

Illegal options are hard-masked by the engine before softmax.

## Multi-config pressure

Every update mixes ~30 table configs across three stack tiers (clubgg,
clubgg_deep, deep) and seat counts 2–6, so one network must play short
and deep, heads-up and six-handed. The log line's `seats=` / `stacks_bb=`
show **one sample config** for flavor — never explain aggregate stats
with those fields alone.

## In this project

- Engine: `rust_engine/` via `plo5bp._engine` (`PyGameState`,
  `PyBatchedEngine`).
- Env wrappers: `python/plo5bp/env.py`, `env_batched.py`.
- Sibling variants `plo4_double_bomb` / `plo6_double_bomb` share dims but
  **never** cross-warm-start (different equities).
