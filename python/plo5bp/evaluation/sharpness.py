"""Policy sharpness / collapse probe (scripts/policy_sharpness.py, run_watch.py,
update_snr.py): one fixed, cached set of self-play decision states, measured
for every checkpoint.

The cache now records what the states ARE (2026-09-28, ML-053): the
observation revision, layout and width they were encoded at, the checkpoint
whose self-play made them and the table sampler's tag. A checkpoint is only
measured on states it can read: same revision, and a layout its obs_adapter
accepts (full-layout states serve a minimal-obs model through the projection;
not the reverse). A cache written before the stamps exists is refused unless
the caller says which revision it holds.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from plo5bp import encoding as _encoding
from plo5bp.actions import GATE_RAISE
from plo5bp.env_batched import BatchedBombPotEnv
from plo5bp.evaluation.tables import TIERS, sample_table, tier_spec_version
from plo5bp.network import obs_adapter
from plo5bp.rollout import TRAIN_OPP_OUTCOME_MC
from plo5bp.sizing import anchor_grid_torch

ANCHOR_NAMES = ("min", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "pot")


class StatesMismatch(ValueError):
    """The cached states cannot be read by this checkpoint."""


def build_states(actor, rows: int, obs_mode: str, seed: int = 0, tables: int = 256):
    """Self-play decision states of `actor` (sampling its policy) on 30 table
    configs, 10 per tier, drawn like training; a uniform sample of `rows`.
    Returns (obs, masks, sizing, tiers)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    obs_l, mask_l, siz_l, tier_l = [], [], [], []
    for ti, tier in enumerate(TIERS):
        for _ in range(10):
            cfg = sample_table(tier, rng)
            env = BatchedBombPotEnv(
                tables, cfg, obs_mode=obs_mode,
                opp_outcome_mc=TRAIN_OPP_OUTCOME_MC if obs_mode == "full" else 0,
            )
            env.reset_batch(
                rng.integers(0, 2**63 - 1, size=tables, dtype=np.int64).astype(np.uint64),
                rng.integers(0, cfg.num_seats, size=tables).astype(np.uint8),
            )
            while not env._dones.all():
                live = np.nonzero(~env._dones)[0]
                sizing = env.sizing()
                obs_l.append(env._obs[live].copy())
                mask_l.append(env._gate_mask[live].copy())
                siz_l.append(sizing[live].copy())
                tier_l.append(np.full(live.size, ti, dtype=np.int8))
                gates = np.zeros(tables, dtype=np.uint8)
                chips = np.zeros(tables, dtype=np.uint64)
                with torch.no_grad():
                    out = actor.act(
                        torch.from_numpy(env._obs[live]),
                        torch.from_numpy(env._gate_mask[live]),
                        torch.from_numpy(sizing[live]),
                    )
                gates[live] = out.gate.numpy().astype(np.uint8)
                chips[live] = np.maximum(out.chips.numpy(), 0).astype(np.uint64)
                env.step_hybrid_batch(gates, chips)
    obs = np.concatenate(obs_l)
    pick = np.sort(rng.permutation(obs.shape[0])[: int(rows)])
    print(f"  self-play: {obs.shape[0]} decision states over 30 configs; sampled {pick.size}")
    return (obs[pick], np.concatenate(mask_l)[pick], np.concatenate(siz_l)[pick],
            np.concatenate(tier_l)[pick])


def save_states(cache: "str | Path", states, *, states_from: str, obs_mode: str) -> None:
    obs, masks, sizing, tiers = states
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, obs=obs, masks=masks, sizing=sizing, tiers=tiers,
        states_from=str(states_from), obs_mode=str(obs_mode),
        obs_rev=int(_encoding.OBS_SEMANTICS_REV), obs_dim=int(obs.shape[1]),
        tier_spec=tier_spec_version(),
    )


def load_states(cache: "str | Path", assume_rev: "int | None" = None) -> "tuple[tuple, dict]":
    """((obs, masks, sizing, tiers), info) from a cache. `info["obs_rev"]` is
    the stamp, or `assume_rev` for a cache written before the stamps (else
    StatesMismatch)."""
    z = np.load(cache)
    states = (z["obs"], z["masks"], z["sizing"], z["tiers"])
    info = {
        "path": str(cache),
        "states_from": str(z["states_from"]) if "states_from" in z else None,
        "obs_dim": int(z["obs"].shape[1]),
        "obs_mode": str(z["obs_mode"]) if "obs_mode" in z else None,
        "obs_rev": int(z["obs_rev"]) if "obs_rev" in z else None,
        "tier_spec": str(z["tier_spec"]) if "tier_spec" in z else None,
    }
    if info["obs_rev"] is None:
        if assume_rev is None:
            raise StatesMismatch(
                f"{cache} was written before caches recorded their observation "
                "revision: pass the revision it was built at (--cache-rev N), or "
                "delete it to rebuild"
            )
        info["obs_rev"] = int(assume_rev)
    return states, info


def check_readable(info: dict, meta: dict, actor) -> "callable":
    """The obs adapter that lets `actor` (a checkpoint with `meta`) read the
    cached states, or StatesMismatch."""
    if int(meta["obs_rev"]) != int(info["obs_rev"]):
        raise StatesMismatch(
            f"{meta['path']} was trained on obs rev {meta['obs_rev']}, the cached "
            f"states ({info['path']}) are rev {info['obs_rev']}: measure with a cache "
            "built at the checkpoint's revision"
        )
    adapt = obs_adapter(actor)
    probe = np.zeros((1, info["obs_dim"]), dtype=np.float32)
    try:
        adapt(probe)
    except ValueError as e:
        raise StatesMismatch(f"{meta['path']} cannot read {info['path']}: {e}") from e
    return adapt


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.log(p.clamp_min(1e-30))).sum(-1)


def measure(actor, obs, masks, sizing, tiers, batch: int = 8192, adapt=None) -> dict:
    gate_h, rare2, rare3, p_raise, raise_ok = [], [], [], [], []
    anch_h, anch_rare, anch_legal, anch_p, anch_ok = [], [], [], [], []
    with torch.no_grad():
        for s in range(0, obs.shape[0], batch):
            chunk = obs[s:s + batch] if adapt is None else adapt(obs[s:s + batch])
            o = torch.from_numpy(np.ascontiguousarray(chunk))
            m = torch.from_numpy(masks[s:s + batch])
            z = torch.from_numpy(sizing[s:s + batch])
            gate_logits, head_out, _refine, _v = actor(o, m)
            gp = torch.softmax(gate_logits.float(), dim=-1) * m
            gp = gp / gp.sum(-1, keepdim=True)
            gate_h.append(_entropy(gp))
            legal_min = torch.where(m, gp, torch.ones_like(gp)).min(-1).values
            n_legal = m.sum(-1)
            rare2.append((legal_min < 1e-2) & (n_legal > 1))
            rare3.append((legal_min < 1e-3) & (n_legal > 1))
            ok = m[:, GATE_RAISE]
            raise_ok.append(ok)
            p_raise.append(gp[:, GATE_RAISE])
            grid = anchor_grid_torch(z, actor.anchor_spec)
            ap = actor._anchor_dist(head_out, grid).probs.float()
            anch_h.append(_entropy(ap))
            anch_rare.append(((ap < 1e-3) & grid.legal).sum(-1).float())
            anch_legal.append(grid.legal.sum(-1).float())
            anch_p.append(ap)
            anch_ok.append(grid.legal)
    gate_h = torch.cat(gate_h).numpy()
    rare2 = torch.cat(rare2).numpy()
    rare3 = torch.cat(rare3).numpy()
    raise_ok = torch.cat(raise_ok).numpy()
    p_raise = torch.cat(p_raise).numpy()
    anch_h = torch.cat(anch_h).numpy()
    anch_rare = torch.cat(anch_rare).numpy()
    anch_legal = torch.cat(anch_legal).numpy()
    anch_p = torch.cat(anch_p).numpy()
    anch_ok = torch.cat(anch_ok).numpy()
    multi = (masks.sum(-1) > 1)

    def block(sel):
        g = sel & multi
        r = sel & raise_ok & (anch_legal > 1)
        return {
            "n": int(sel.sum()),
            "gate_h": float(gate_h[g].mean()) if g.any() else None,
            "gate_h_p10_p50_p90": [float(x) for x in np.percentile(gate_h[g], [10, 50, 90])] if g.any() else None,
            "rare_gate_1e2": float(rare2[g].mean()) if g.any() else None,
            "rare_gate_1e3": float(rare3[g].mean()) if g.any() else None,
            "p_raise": float(p_raise[sel & raise_ok].mean()) if (sel & raise_ok).any() else None,
            "anchor_h": float(anch_h[r].mean()) if r.any() else None,
            "rare_anchor_1e3": float((anch_rare[r] / anch_legal[r]).mean()) if r.any() else None,
            # mean probability of each anchor (min, 10%, ..., pot) where raising
            # with 2+ legal sizes, and how often each anchor is legal there
            "anchor_mean": [float(x) for x in anch_p[r].mean(0)] if r.any() else None,
            "anchor_legal": [float(x) for x in anch_ok[r].mean(0)] if r.any() else None,
            "anchor_top": float(anch_p[r].max(-1).mean()) if r.any() else None,
        }

    out = {"ALL": block(np.ones_like(multi))}
    for ti, tier in enumerate(TIERS):
        out[tier] = block(tiers == ti)
    return out


def ensure_states(cache: "str | Path", states_from: str, rows: int = 32768,
                  assume_rev: "int | None" = None) -> "tuple[tuple, dict]":
    """Load the cache, or build it from `states_from`'s self-play first."""
    from plo5bp.evaluation.loader import load_actor

    cache = Path(cache)
    if not cache.exists():
        t0 = time.time()
        ref, meta = load_actor(states_from)
        print(f"building {rows} states from {states_from} (obs_mode={meta['obs_mode']}) ...")
        states = build_states(ref, rows, meta["obs_mode"])
        save_states(cache, states, states_from=states_from, obs_mode=meta["obs_mode"])
        print(f"  cached -> {cache} ({time.time() - t0:.0f}s)")
    return load_states(cache, assume_rev)
