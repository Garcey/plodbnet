"""Ratings for a round robin of checkpoints (scripts/h2h_league.py), with
uncertainty (2026-09-28, ML-052).

`fit_ratings` = the weighted least-squares fit edge(A, B) ~ r_A - r_B (ratings
sum to 0). Every pair plays the SAME table configs and deals, so pair results
are correlated: the uncertainty comes from a bootstrap over CONFIGS (the same
resample for every pair keeps that correlation), and the non-transitivity test
compares the observed residual RMS with its null -- the residual RMS of a
perfectly transitive table (the fitted differences) plus the bootstrap's noise.
"""

from __future__ import annotations

import numpy as np


def fit_ratings(n: int, results: "dict[tuple[int, int], tuple[float, float]]"):
    """results: {(i, j): (edge, se)} with edge = player i's edge over player j.
    Returns (ratings (n,), residuals {(i, j): edge - (r_i - r_j)})."""
    rows, y, w = [], [], []
    for (i, j), (e, se) in results.items():
        row = np.zeros(n)
        row[i], row[j] = 1.0, -1.0
        rows.append(row)
        y.append(e)
        w.append(1.0 / max(se, 1e-6) ** 2)
    rows.append(np.ones(n))  # sum-to-zero constraint (heavily weighted)
    y.append(0.0)
    w.append(1e6)
    a = np.asarray(rows) * np.sqrt(np.asarray(w))[:, None]
    b = np.asarray(y) * np.sqrt(np.asarray(w))
    r, *_ = np.linalg.lstsq(a, b, rcond=None)
    resid = {k: e - (r[k[0]] - r[k[1]]) for k, (e, _se) in results.items()}
    return r, resid


def residual_rms(resid: dict) -> float:
    return float(np.sqrt(np.mean([v * v for v in resid.values()]))) if resid else 0.0


def bootstrap_league(
    n: int,
    per_config: "dict[tuple[int, int], list[np.ndarray]]",
    samples: int = 200,
    seed: int = 0,
    level: float = 0.95,
    anchor: "list[int] | None" = None,
) -> dict:
    """per_config: {(i, j): [config 0's pair edges, config 1's, ...]} -- every
    pair over the SAME configs, in the same order. Returns ratings, their
    confidence intervals (bootstrap over configs), the residual RMS and its
    transitive-null distribution, and a p-value for "the table is more
    non-transitive than noise explains". `anchor` (player indices): the scale
    is fixed by THEIR mean rating being 0 (in the point fit and in every
    bootstrap fit) instead of everyone's -- a fixed reference panel, so a
    rated newcomer cannot move the scale (evaluation/panel.py, ML-039)."""
    pairs = list(per_config)
    n_cfg = len(per_config[pairs[0]])
    if any(len(per_config[p]) != n_cfg for p in pairs):
        raise ValueError("every pair must have results for the same configs")

    def edges(sel: np.ndarray) -> "dict[tuple[int, int], float]":
        return {p: float(np.concatenate([per_config[p][k] for k in sel]).mean()) for p in pairs}

    full = np.arange(n_cfg)
    obs = edges(full)
    se = {}
    for p in pairs:
        x = np.concatenate(per_config[p])
        se[p] = float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else 1.0
    results = {p: (obs[p], se[p]) for p in pairs}

    def centered(r: np.ndarray) -> np.ndarray:
        return r - float(np.mean(r[anchor])) if anchor is not None else r

    r, resid = fit_ratings(n, results)
    r = centered(r)
    fitted = {p: float(r[p[0]] - r[p[1]]) for p in pairs}
    rms_obs = residual_rms(resid)

    rng = np.random.default_rng(seed)
    boot_r, null_rms = [], []
    for _ in range(int(samples)):
        sel = rng.integers(0, n_cfg, size=n_cfg)
        eb = edges(sel)
        rb, _ = fit_ratings(n, {p: (eb[p], se[p]) for p in pairs})
        boot_r.append(centered(rb))
        # A transitive truth (the fitted differences) plus this resample's noise.
        noisy = {p: (fitted[p] + (eb[p] - obs[p]), se[p]) for p in pairs}
        _, rn = fit_ratings(n, noisy)
        null_rms.append(residual_rms(rn))
    boot_r = np.asarray(boot_r)
    lo_q, hi_q = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    null = np.asarray(null_rms)
    return {
        "ratings": r,
        "ci_low": np.quantile(boot_r, lo_q, axis=0),
        "ci_high": np.quantile(boot_r, hi_q, axis=0),
        "residuals": resid,
        "residual_rms": rms_obs,
        "null_rms_median": float(np.median(null)),
        "null_rms_high": float(np.quantile(null, hi_q)),
        "p_nontransitive": float((null >= rms_obs).mean()),
        "edges": results,
    }
