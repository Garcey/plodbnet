# plo5bp — PLO5 Double-Board Bomb Pot Self-Play PPO

Phase 1 plumbing: Rust game engine (via PyO3) + Python PPO trainer for 6-max
PLO5 double-board bomb pots, 20bb stacks, 3bb antes (ClubGG format).

## Build

```bash
pip install maturin
maturin develop                                # builds Rust engine, installs plo5bp
cargo test --manifest-path rust_engine/Cargo.toml
pytest tests/python
python scripts/smoke_test.py
python scripts/train.py
```

## Layout

- `rust_engine/` — Rust game engine (cards, hand eval, double-board payouts,
  state machine, PyO3 bindings).
- `python/plo5bp/` — Python package: Gym-style env, observation encoding,
  actor-critic network, PPO, self-play, rollout, eval.
- `scripts/` — `smoke_test.py` (10k random hands, correctness assertions),
  `train.py` (stub PPO run), `evaluate.py` (checkpoint vs baselines).
- `tests/python/` — pytest suites.
- `rust_engine/tests/` — Rust integration tests.

## Chip units

1 bb = 100 chips. Starting stack = 2000 (20bb), ante = 300 (3bb).
All chip math is integer; payouts sum to zero by construction.

## Phase 1 scope

Correctness-first pipeline. Not a strength run. See
`.claude/plans/i-m-starting-a-new-kind-dragonfly.md` for the approved plan.
