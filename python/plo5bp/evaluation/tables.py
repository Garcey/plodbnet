"""Table configs and the duplicate-deal head-to-head match, shared by the h2h
tools (h2h_eval / h2h_cross / h2h_league / sweep_eval / run_watch)."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from plo5bp.config import GameConfig
from plo5bp.env_batched import BatchedBombPotEnv
from plo5bp.rollout import TRAIN_OPP_OUTCOME_MC
from plo5bp.train import tiers as _tiers

TIERS = ("clubgg", "clubgg_deep", "deep")
BB, ANTE = 10_000, 30_000
SEAT_CHOICES = (2, 3, 4, 5, 6)


def tier_spec_version() -> str:
    """A tag of the table-config sampler (plo5bp/train/tiers.py's code, not
    its comments), written into every result: evaluations run after the tiers
    changed say so instead of silently comparing different tables."""
    src = Path(_tiers.__file__).read_text(encoding="utf-8")
    return hashlib.sha256(ast.dump(ast.parse(src)).encode()).hexdigest()[:12]


def sample_table(tier: str, rng: np.random.Generator,
                 variant: str = "plo5_double_bomb") -> GameConfig:
    """One (seats, stacks) table of a training tier: 2-6 seats, the tier's
    stack bands, 1 bb = 10,000 chips, 3 bb ante."""
    cfg, _ = _tiers._sample_game_config(
        SEAT_CHOICES, 1.0, 300.0, BB, ANTE, rng,
        stack_dist=tier, seats_dist="uniform", variant=variant, sb=0,
    )
    return cfg


def default_env(n: int, game_cfg: GameConfig, ev_samples: int, obs_mode: str) -> BatchedBombPotEnv:
    return BatchedBombPotEnv(
        n, game_cfg, ev_runout_samples=ev_samples, obs_mode=obs_mode,
        # the full layout's opp-outcome features (dims 982-989) at the training
        # sample count; 0 fed full-obs models zeros there (minimal obs has none)
        opp_outcome_mc=TRAIN_OPP_OUTCOME_MC if obs_mode == "full" else 0,
    )


def play_duplicate(
    game_cfg: GameConfig,
    models: Sequence[torch.nn.Module],
    deals: int,
    device: torch.device,
    rng: np.random.Generator,
    ev_samples: int = 64,
    greedy: "tuple[bool, bool]" = (False, False),
    obs_mode: str = "full",
    env: "BatchedBombPotEnv | None" = None,
    observe: "Callable[[int, BatchedBombPotEnv, np.ndarray], np.ndarray] | None" = None,
) -> "tuple[np.ndarray, int, int]":
    """One table config in the duplicate format: `deals` deals x 2 passes on the
    same cards and button, A's seats in pass 0 are B's in pass 1 (a random
    per-deal alternating offset), every hand that ends all-in before the river
    pays its `ev_samples`-runout EV. Returns (A's net chips per kept deal pair,
    summed over A's seats in both passes; seats; steps). Pairs where either
    pass was over at the deal are dropped.

    `observe(k, env, rows)` gives player k's observation rows (default
    `env._obs[rows]`); `env` replaces the default env (built for `2 * deals`
    tables). The random draws happen in a fixed order (deal seeds, buttons,
    offsets, then the models' sampling on torch's global RNG, A before B each
    step), so a seed reproduces the pre-library tools' results exactly."""
    n_seats = game_cfg.num_seats
    n = 2 * deals
    if env is None:
        env = default_env(n, game_cfg, ev_samples, obs_mode)
    seeds = rng.integers(0, 2**63 - 1, size=deals, dtype=np.int64).astype(np.uint64)
    buttons = rng.integers(0, n_seats, size=deals).astype(np.uint8)
    env.reset_batch(np.concatenate([seeds, seeds]), np.concatenate([buttons, buttons]))
    offset = rng.integers(0, 2, size=deals)
    seat = np.arange(n_seats)
    a0 = ((seat[None, :] + offset[:, None]) % 2) == 0          # (deals, S)
    a_mask = np.concatenate([a0, ~a0], axis=0)                  # (n, S)
    live_at_deal = ~env._dones.copy()
    net = np.zeros(n, dtype=np.float64)
    rows_all = np.arange(n)
    steps = 0
    while not env._dones.all():
        steps += 1
        actors = env._actors
        live = ~env._dones
        safe = np.where(actors >= 0, actors, 0).astype(np.intp)
        is_a = live & a_mask[rows_all, safe]
        sizing = env.sizing()
        gates = np.zeros(n, dtype=np.uint8)
        chips = np.zeros(n, dtype=np.uint64)
        for k, (model, rows_mask, det) in enumerate(
            ((models[0], is_a, greedy[0]), (models[1], live & ~is_a, greedy[1]))
        ):
            rows = np.nonzero(rows_mask)[0]
            if rows.size == 0:
                continue
            obs = env._obs[rows] if observe is None else observe(k, env, rows)
            o = torch.from_numpy(np.ascontiguousarray(obs)).to(device)
            m = torch.from_numpy(env._gate_mask[rows]).to(device)
            b = torch.from_numpy(sizing[rows]).to(device)
            with torch.inference_mode():
                out = model.act(o, m, b, deterministic=det)
            gates[rows] = out.gate.cpu().numpy().astype(np.uint8)
            chips[rows] = np.maximum(out.chips.cpu().numpy(), 0).astype(np.uint64)
        st = env.step_hybrid_batch(gates, chips)
        term = np.nonzero(st.newly_terminal)[0]
        if term.size:
            r = st.rewards[term].astype(np.float64)               # chip deltas
            net[term] += (r * a_mask[term]).sum(axis=1)
    pair_net = net[:deals] + net[deals:]
    keep = live_at_deal[:deals] & live_at_deal[deals:]
    return pair_net[keep], n_seats, steps


def _stats(parts: "list[np.ndarray]") -> dict:
    """Edge (bb per A seat-hand) with two standard errors: `se` treats every
    deal pair as independent (the historical number); `se_config` resamples
    whole table CONFIGS (per-config means, SE across configs) -- it includes
    config-to-config variance, which `se` ignores, so it is the honest one when
    few configs are played (ML-002)."""
    x = np.concatenate(parts) if parts else np.zeros(0)
    out = {"pairs": int(x.size), "configs": len(parts)}
    if x.size < 2:
        return {**out, "edge_bb": float(x.mean()) if x.size else float("nan"),
                "se": float("nan"), "se_config": float("nan")}
    means = np.asarray([p.mean() for p in parts if p.size])
    out.update(
        edge_bb=float(x.mean()),
        se=float(x.std(ddof=1) / np.sqrt(x.size)),
        se_config=(float(means.std(ddof=1) / np.sqrt(means.size))
                   if means.size >= 2 else float("nan")),
    )
    return out


def summarize(per_tier: "dict[str, list[np.ndarray]]") -> dict:
    """{"edge_bb", "se", "se_config", "pairs", "configs", "tiers": {tier:
    same}} from per-config arrays of bb per A seat-hand."""
    everything = [p for parts in per_tier.values() for p in parts]
    report = _stats(everything)
    report["tiers"] = {t: _stats(parts) for t, parts in per_tier.items() if parts}
    return report
