"""A standing robustness track (2026-09-28, ML-039): rate checkpoints against a
FIXED panel of references.

Progress used to be judged against one or two references, which cannot tell
learning from cycling (a checkpoint can beat last week's model and lose to last
month's). A PANEL is a fixed set of 6-8 references -- eras, lineages, and a few
simple baseline policies -- whose round robin is played ONCE and cached; each
candidate then plays every member on the same table configs and deals
(`tables.play_duplicate`, common random numbers) and gets:

- one RATING on the panel's scale (the least-squares league fit with the
  panel's ratings summing to 0, so candidates rated months apart share a
  scale) with a 95% interval from a bootstrap over table configs
  (`league.bootstrap_league`);
- how NON-TRANSITIVE its results are: its own residual RMS against the panel
  (large = it beats some members its rating says it should lose to, the
  signature of a cycling policy) and the league's p-value.

`scripts/panel_eval.py` rates given checkpoints or every Nth numbered
checkpoint of a stem not yet rated, appending one JSON line per (candidate,
mode) to `runs/<panel>.panel.jsonl`.

A panel spec is JSON:

    {"name": "plo5-full-rev1",
     "members": [["vSix4_940", "checkpoints/prod_vSix4_940.pt"],
                 ["call", "baseline:call"], ...],
     "deals": 512, "configs_per_tier": 4, "seed": 0,
     "modes": ["argmax", "sampled"], "ev_samples": 64}

Every network member must read the same observation layout at this process's
observation revision (full-obs rev-1 lineages: PLO5BP_OBS_REV=1). Baseline
members: "baseline:call" (check/call, never fold or raise), "baseline:pot"
(raise the maximum whenever legal, else check/call) and "baseline:random"
(a uniformly random legal gate, raise size uniform in the legal window).
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.evaluation.league import bootstrap_league
from plo5bp.evaluation.loader import load_actor
from plo5bp.evaluation.tables import BB, TIERS, play_duplicate, sample_table, tier_spec_version

BASELINES = ("call", "pot", "random")


@dataclass(frozen=True)
class Panel:
    name: str
    members: tuple  # ((name, path or "baseline:<kind>"), ...)
    deals: int = 512
    configs_per_tier: int = 4
    seed: int = 0
    modes: tuple = ("argmax",)
    ev_samples: int = 64
    variant: str = "plo5_double_bomb"

    @classmethod
    def load(cls, path: "str | Path") -> "Panel":
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        members = tuple((str(n), str(p)) for n, p in raw.pop("members"))
        if len(members) < 2:
            raise ValueError(f"{path}: a panel needs at least 2 members")
        if len({n for n, _ in members}) != len(members):
            raise ValueError(f"{path}: member names must be unique")
        raw["modes"] = tuple(raw.get("modes", ("argmax",)))
        return cls(members=members, **raw)

    def key(self) -> str:
        """What the cached panel round robin depends on: the spec, the files'
        sizes and modification times, the table sampler's version."""
        files = []
        for _n, p in self.members:
            if p.startswith("baseline:"):
                files.append(p)
            else:
                st = Path(p).stat()
                files.append(f"{p}:{st.st_size}:{int(st.st_mtime)}")
        blob = json.dumps(
            {**asdict(self), "files": files, "tiers": tier_spec_version()}, sort_keys=True
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def configs(self) -> list:
        """The panel's table configs (tier, GameConfig): fixed by its seed."""
        crng = np.random.default_rng(self.seed)
        return [
            (tier, sample_table(tier, crng, self.variant))
            for tier in TIERS for _ in range(self.configs_per_tier)
        ]


class BaselineActor(torch.nn.Module):
    """A fixed policy with the networks' `act` interface, so it plays in
    `play_duplicate` like any member: "call", "pot" or "random" (see the
    module docstring). `deterministic` is ignored."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        if kind not in BASELINES:
            raise ValueError(f"unknown baseline {kind!r} (known: {BASELINES})")
        self.kind = kind
        self._dummy = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def act(self, obs, gate_mask, sizing, deterministic: bool = False):
        n = gate_mask.shape[0]
        gm = gate_mask.bool()
        call = torch.where(gm[:, GATE_CHECK_CALL], GATE_CHECK_CALL, GATE_FOLD)
        zero = torch.zeros(n, dtype=torch.int64, device=gate_mask.device)
        if self.kind == "call":
            return _Acts(call.to(torch.int64), zero)
        if self.kind == "pot":
            gate = torch.where(gm[:, GATE_RAISE], GATE_RAISE, call).to(torch.int64)
            chips = torch.where(gate == GATE_RAISE, sizing[:, 1].to(torch.int64), zero)
            return _Acts(gate, chips)
        # random: uniform over the legal gates, raise size uniform in the window
        w = gm.float()
        gate = torch.multinomial(w / w.sum(-1, keepdim=True).clamp_min(1.0), 1).squeeze(-1)
        lo, hi = sizing[:, 0].to(torch.float64), sizing[:, 1].to(torch.float64)
        u = torch.rand(n, dtype=torch.float64, device=gate_mask.device)
        chips = torch.floor(lo + u * (hi - lo + 1.0)).clamp(max=hi).to(torch.int64)
        return _Acts(gate.to(torch.int64), torch.where(gate == GATE_RAISE, chips, zero))


@dataclass
class _Acts:
    gate: torch.Tensor
    chips: torch.Tensor


def load_member(path: str, device) -> "tuple[torch.nn.Module, dict | None]":
    """A panel member or candidate: a baseline policy, or a checkpoint's actor
    (sized from the checkpoint; an observation-revision mismatch is refused)."""
    if path.startswith("baseline:"):
        return BaselineActor(path.split(":", 1)[1]).to(device), None
    return load_actor(path, device)


def play_pair(
    panel: Panel,
    a: torch.nn.Module,
    b: torch.nn.Module,
    configs: list,
    device,
    obs_mode: str,
    greedy: bool,
    play: Callable = play_duplicate,
) -> "list[list[float]]":
    """A's edge over B in bb per seat-hand, one list of deal-pair values per
    table config -- the SAME deals for every pair of the panel (per-config
    seeds), so the league's pairs share common random numbers."""
    torch.manual_seed(panel.seed)
    per = []
    for k, (_tier, cfg) in enumerate(configs):
        rng = np.random.default_rng((panel.seed, k))
        net, n_seats, _ = play(
            cfg, (a, b), panel.deals, device, rng, panel.ev_samples,
            greedy=(greedy, greedy), obs_mode=obs_mode,
        )
        per.append([float(x) for x in net / BB / n_seats])
    return per


def panel_round_robin(
    panel: Panel,
    models: list,
    device,
    obs_mode: str,
    cache: "Path | None" = None,
    play: Callable = play_duplicate,
    log: Callable = print,
) -> dict:
    """{mode: {"i,j": per-config lists}} of every member pair, from `cache`
    when it holds this panel's key, else played (and cached)."""
    key = panel.key()
    if cache is not None and cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if stored.get("key") == key and set(stored["modes"]) >= set(panel.modes):
            return stored["modes"]
    configs = panel.configs()
    table: dict = {}
    for mode in panel.modes:
        table[mode] = {}
        for i, j in itertools.combinations(range(len(models)), 2):
            table[mode][f"{i},{j}"] = play_pair(
                panel, models[i], models[j], configs, device, obs_mode, mode == "argmax", play
            )
            log(f"[panel] {mode} {panel.members[i][0]} vs {panel.members[j][0]}")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"key": key, "modes": table}), encoding="utf-8")
    return table


def rate(
    panel: Panel,
    table: dict,
    candidate_rows: "dict[str, dict[int, list]]",
    bootstrap: int = 200,
) -> "dict[str, dict]":
    """The candidate's rating per mode. `candidate_rows[mode][i]` = the
    candidate's per-config edges over member i."""
    n = len(panel.members)
    out = {}
    for mode in panel.modes:
        per_config = {}
        for k, v in table[mode].items():
            i, j = (int(x) for x in k.split(","))
            per_config[(i, j)] = [np.asarray(c, dtype=np.float64) for c in v]
        for i, v in candidate_rows[mode].items():
            per_config[(n, int(i))] = [np.asarray(c, dtype=np.float64) for c in v]
        # The panel's mean rating is 0 -- in the point fit and in every
        # bootstrap fit -- so candidates rated months apart share one scale.
        res = bootstrap_league(
            n + 1, per_config, samples=bootstrap, seed=panel.seed, anchor=list(range(n))
        )
        cand_resid = [v for (a, _b), v in res["residuals"].items() if a == n]
        out[mode] = {
            "rating": float(res["ratings"][n]),
            "ci": [float(res["ci_low"][n]), float(res["ci_high"][n])],
            "edges": {
                panel.members[i][0]: [float(res["edges"][(n, i)][0]), float(res["edges"][(n, i)][1])]
                for i in range(n)
            },
            "panel_ratings": {panel.members[i][0]: float(res["ratings"][i]) for i in range(n)},
            "candidate_residual_rms": float(np.sqrt(np.mean(np.square(cand_resid)))),
            "league_residual_rms": res["residual_rms"],
            "p_nontransitive": res["p_nontransitive"],
        }
    return out


def rate_candidate(
    panel: Panel,
    path: str,
    device,
    cache: "Path | None" = None,
    bootstrap: int = 200,
    play: Callable = play_duplicate,
    log: Callable = print,
    _members: "list | None" = None,
) -> dict:
    """Play `path` against every panel member (and the panel round robin once,
    cached) and rate it: {"candidate", "update", "panel", "panel_key",
    "tier_spec", "modes": {mode: rate(...)}}."""
    loaded = _members or [load_member(p, device) for _n, p in panel.members]
    models = [m for m, _ in loaded]
    metas = [m for _, m in loaded if m is not None]
    cand, cmeta = load_member(path, device)
    layouts = {m["obs_mode"] for m in metas} | ({cmeta["obs_mode"]} if cmeta else set())
    if len(layouts) > 1:
        raise ValueError(f"panel and candidate read different observation layouts: {layouts}")
    obs_mode = layouts.pop() if layouts else "full"
    table = panel_round_robin(panel, models, device, obs_mode, cache, play, log)
    configs = panel.configs()
    rows = {
        mode: {
            i: play_pair(panel, cand, models[i], configs, device, obs_mode, mode == "argmax", play)
            for i in range(len(models))
        }
        for mode in panel.modes
    }
    return {
        "candidate": Path(path).stem,
        "path": str(path),
        "update": None if cmeta is None else cmeta.get("update"),
        "panel": panel.name,
        "panel_key": panel.key(),
        "tier_spec": tier_spec_version(),
        "modes": rate(panel, table, rows, bootstrap),
    }
