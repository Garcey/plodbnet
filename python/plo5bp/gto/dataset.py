"""Supervised datasets for PolicyNet.

Two sources:
  1. **Teacher distill** — roll NLH env, soft labels from an ActorCritic
     (curriculum / debug only; not the GTO teacher).
  2. **Label shards** — ``LabelRecord`` JSONL from native rust_cfr export
     (or synthetic smoke canaries).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv
from plo5bp.gto.labels import LabelRecord, read_jsonl
from plo5bp.gto.roots import CLUBGG_NLH_ROOT
from plo5bp.network import ActorCritic, obs_adapter
from plo5bp.sizing import NLH_ANCHOR_SPEC, sizing_from_info


def _node_dist(model, device, obs, info):
    from plo5bp.ui.trainer import compute_node_distribution
    return compute_node_distribution(model, device, obs, info)


@dataclass
class SupervisedRow:
    obs: np.ndarray          # (OBS_DIM_NLH,) float32
    gate_mask: np.ndarray    # (3,) bool
    sizing: np.ndarray       # (4,) int64 min,max,pot,to_call
    gate_probs: np.ndarray   # (3,) float32 soft target
    anchor_probs: np.ndarray # (K,) float32 soft target (zeros if raise illegal)
    value_bb: float
    street: int


class PolicyDataset(Dataset):
    """In-memory supervised rows for DataLoader."""

    def __init__(self, rows: Sequence[SupervisedRow]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.rows[idx]
        return {
            "obs": torch.from_numpy(r.obs),
            "gate_mask": torch.from_numpy(r.gate_mask),
            "sizing": torch.from_numpy(r.sizing),
            "gate_probs": torch.from_numpy(r.gate_probs),
            "anchor_probs": torch.from_numpy(r.anchor_probs),
            "value_bb": torch.tensor(r.value_bb, dtype=torch.float32),
            "street": torch.tensor(r.street, dtype=torch.int64),
        }


def collect_teacher_distill(
    teacher: ActorCritic,
    *,
    n_decisions: int = 4096,
    seed: int = 0,
    seats: Sequence[int] = (2, 3, 4, 5, 6),
    stack_bb_range: tuple[float, float] = (20.0, 250.0),
    device: torch.device | str = "cpu",
    max_hands: int = 50_000,
) -> list[SupervisedRow]:
    """Roll NLH hands; at every decision store teacher soft labels.

    Multiway 2–6 is included so T1 play generalizes (decision #3).
    Stacks / seats randomize around the ClubGG stake structure.
    """
    teacher = teacher.to(device).eval()
    adapt = obs_adapter(teacher)
    rng = np.random.default_rng(int(seed))
    root = CLUBGG_NLH_ROOT
    rows: list[SupervisedRow] = []
    hands = 0
    k = NLH_ANCHOR_SPEC.count

    while len(rows) < int(n_decisions) and hands < int(max_hands):
        hands += 1
        n_seats = int(seats[int(rng.integers(0, len(seats)))])
        stacks_bb = [
            float(rng.uniform(stack_bb_range[0], stack_bb_range[1]))
            for _ in range(n_seats)
        ]
        cfg = root.game_config(
            num_seats=n_seats, starting_stacks_bb=stacks_bb
        )
        env = BombPotEnv(cfg)
        button = int(rng.integers(0, n_seats))
        hand_seed = int(rng.integers(0, 2**63 - 1))
        obs, info = env.reset(hand_seed, button)
        steps = 0
        while not info.terminal and steps < 200 and len(rows) < n_decisions:
            steps += 1
            if info.actor is None:
                break
            # Skip pure-moot auto-check nodes (no learning signal).
            if not bool(info.gate_mask.any()):
                break
            dist = _node_dist(
                teacher, torch.device(device), adapt(obs), info
            )
            gate_p = np.asarray(dist["gate_probs"], dtype=np.float32)
            if dist.get("head_version", 1) >= 2 and dist.get("anchor_probs"):
                ap = np.asarray(dist["anchor_probs"], dtype=np.float32)
                if ap.shape[0] != k:
                    # Pad / trim if teacher ladder differs (shouldn't for NLH).
                    out = np.zeros(k, dtype=np.float32)
                    m = min(k, ap.shape[0])
                    out[:m] = ap[:m]
                    s = float(out.sum())
                    ap = out / s if s > 0 else out
            else:
                ap = np.zeros(k, dtype=np.float32)
                if bool(info.gate_mask[2]):
                    ap[0] = 1.0

            rows.append(
                SupervisedRow(
                    obs=np.asarray(adapt(obs), dtype=np.float32).copy(),
                    gate_mask=np.asarray(info.gate_mask, dtype=bool).copy(),
                    sizing=sizing_from_info(info).astype(np.int64),
                    gate_probs=gate_p,
                    anchor_probs=ap,
                    value_bb=float(dist.get("value_bb", 0.0)),
                    street=int(info.raw_obs.get("street", 0)),
                )
            )
            # Advance with teacher sample (mixed) so trajectories cover
            # the on-policy support of the teacher.
            with torch.no_grad():
                o = torch.from_numpy(adapt(obs)).unsqueeze(0).to(device)
                m = torch.from_numpy(info.gate_mask).unsqueeze(0).to(device)
                b = torch.from_numpy(sizing_from_info(info)[None, :]).to(device)
                out = teacher.act(o, m, b, deterministic=False)
                gate = int(out.gate.item())
                chips = int(out.chips.item())
            obs, _, done, info = env.step_hybrid(gate, chips)
            if done:
                break
    return rows


def rows_from_label_records(
    labels: Sequence[LabelRecord],
    *,
    obs_fn=None,
    synthesize_obs: bool = True,
) -> list[SupervisedRow]:
    """Convert LabelRecords to SupervisedRows.

    Priority for obs:
      1. ``notes['obs']`` if present
      2. ``obs_fn(lab)`` if given
      3. ``obs_from_label`` synthesis (``OBS_DIM_NLH``) when ``synthesize_obs``
    """
    if synthesize_obs and obs_fn is None:
        from plo5bp.gto.obs_from_label import obs_from_label as _default_obs

        obs_fn = _default_obs

    k = NLH_ANCHOR_SPEC.count
    rows: list[SupervisedRow] = []
    for lab in labels:
        obs = None
        if lab.notes.get("obs") is not None:
            obs = np.asarray(lab.notes["obs"], dtype=np.float32)
        elif obs_fn is not None:
            obs = obs_fn(lab)
        if obs is None:
            continue
        fold_legal = int(lab.to_call_chips) > 0
        raise_legal = int(lab.max_raise_chips) > 0
        gate_mask = np.array([fold_legal, True, raise_legal], dtype=bool)
        g = np.asarray(lab.gate_probs, dtype=np.float32).copy()
        if len(g) != 3:
            g = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        g = g * gate_mask.astype(np.float32)
        if float(g.sum()) <= 0:
            g = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            g /= g.sum()
        ap = np.zeros(k, dtype=np.float32)
        for a in lab.action_probs:
            if a.gate == "raise" and a.anchor_k is not None:
                if 0 <= int(a.anchor_k) < k:
                    ap[int(a.anchor_k)] += float(a.prob)
        s = float(ap.sum())
        if s > 0:
            ap /= s
        elif raise_legal:
            ap[-1] = 1.0
        rows.append(
            SupervisedRow(
                obs=np.asarray(obs, dtype=np.float32),
                gate_mask=gate_mask,
                sizing=np.array(
                    [
                        lab.min_raise_chips,
                        lab.max_raise_chips,
                        lab.pot_chips,
                        lab.to_call_chips,
                    ],
                    dtype=np.int64,
                ),
                gate_probs=g,
                anchor_probs=ap,
                value_bb=float(lab.value_bb or 0.0),
                street=int(lab.street),
            )
        )
    return rows


def load_label_shard_rows(
    path: Path | str,
    *,
    synthesize_obs: bool = True,
) -> list[SupervisedRow]:
    """Load LabelRecord JSONL → supervised rows (obs synthesized by default)."""
    return rows_from_label_records(
        list(read_jsonl(path)), synthesize_obs=synthesize_obs
    )


def load_label_shards(
    paths: Sequence[Path | str],
    *,
    synthesize_obs: bool = True,
) -> list[SupervisedRow]:
    """Load multiple JSONL shards into one row list."""
    rows: list[SupervisedRow] = []
    for p in paths:
        rows.extend(load_label_shard_rows(p, synthesize_obs=synthesize_obs))
    return rows


def save_rows_npz(path: Path | str, rows: Sequence[SupervisedRow]) -> None:
    """Compact numpy bundle for fast reload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=np.stack([r.obs for r in rows]),
        gate_mask=np.stack([r.gate_mask for r in rows]),
        sizing=np.stack([r.sizing for r in rows]),
        gate_probs=np.stack([r.gate_probs for r in rows]),
        anchor_probs=np.stack([r.anchor_probs for r in rows]),
        value_bb=np.asarray([r.value_bb for r in rows], dtype=np.float32),
        street=np.asarray([r.street for r in rows], dtype=np.int64),
    )


def load_rows_npz(path: Path | str) -> list[SupervisedRow]:
    data = np.load(path, allow_pickle=False)
    n = data["obs"].shape[0]
    return [
        SupervisedRow(
            obs=data["obs"][i],
            gate_mask=data["gate_mask"][i].astype(bool),
            sizing=data["sizing"][i],
            gate_probs=data["gate_probs"][i],
            anchor_probs=data["anchor_probs"][i],
            value_bb=float(data["value_bb"][i]),
            street=int(data["street"][i]),
        )
        for i in range(n)
    ]
