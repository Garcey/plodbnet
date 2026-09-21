"""Supervised datasets for PolicyNet.

Two sources:
  1. **Teacher distill** — roll NLH env, soft labels from an ActorCritic
     (curriculum / debug only; not the GTO teacher).
  2. **Label shards** — ``LabelRecord`` JSONL from native rust_cfr export
     (or synthetic smoke canaries).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from plo5bp.config import GameConfig
from plo5bp.env import BombPotEnv
from plo5bp.gto.labels import (
    ILLEGAL_MASS_TOL,
    IllegalTeacherMassError,
    LabelRecord,
    illegal_teacher_mass,
    jam_anchor_index,
    read_jsonl,
)
from plo5bp.gto.obs_rev import UNSTAMPED_OBS_REV, current_obs_rev
from plo5bp.gto.roots import CLUBGG_NLH_ROOT
from plo5bp.network import ActorCritic, obs_adapter
from plo5bp.sizing import NLH_ANCHOR_SPEC, sizing_from_info


def _node_dist(model, device, obs, info):
    from plo5bp.ui.trainer import compute_node_distribution
    return compute_node_distribution(model, device, obs, info)


@dataclass(frozen=True)
class RowProvenance:
    """Where a supervised row came from (review 2026-09-20 D4/F6/F11).

    Carried per row so the CHECKPOINT meta (train root ids, label sources,
    teacher exploitability cap, seat/street coverage) is DERIVED from the
    records that were trained on — never asserted by the calling script.
    """

    root_id: str = ""            # LabelRecord.root_name ("" = not a solver root)
    source: str = ""             # LabelRecord.source / "rule_bootstrap" / ...
    num_seats: int = 0           # 0 = unknown
    expl_bb: float | None = None       # teacher root exploitability
    expl_verified: bool = False        # produced by a FINAL estimator
    teacher_cap_bb: float | None = None  # cap the exporter enforced
    # How the row's obs was produced: "canonical" (solver-root form — what
    # PolicyNetHost serves postflop), "live" (raw env obs), "synthetic_preflop",
    # "synthetic_fallback", "attached", "custom". The host canonicalizes at
    # serve only for nets trained on canonical rows (review 2026-09-20 D3).
    obs_form: str = ""


_NO_PROVENANCE = RowProvenance()


@dataclass
class SupervisedRow:
    obs: np.ndarray          # (OBS_DIM_NLH,) float32
    gate_mask: np.ndarray    # (3,) bool
    sizing: np.ndarray       # (4,) int64 min,max,pot,to_call
    gate_probs: np.ndarray   # (3,) float32 soft target
    anchor_probs: np.ndarray # (K,) float32 soft target (zeros if raise illegal)
    value_bb: float
    street: int
    # (review 2026-09-20 F11) False when the label had NO value target
    # (``value_bb is None`` — every CFR export today): the row is masked out
    # of the value loss instead of being trained toward 0.
    value_mask: bool = True
    prov: RowProvenance = _NO_PROVENANCE


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
            "value_mask": torch.tensor(bool(r.value_mask), dtype=torch.bool),
            "street": torch.tensor(r.street, dtype=torch.int64),
            # Row index: lets the trainer name the offending ROOT when an
            # assertion fires on a batch.
            "idx": torch.tensor(idx, dtype=torch.int64),
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
    canonical_obs: bool = True,
) -> list[SupervisedRow]:
    """Roll NLH hands; at every decision store teacher soft labels.

    Multiway 2–6 is included so T1 play generalizes (decision #3).
    Stacks / seats randomize around the ClubGG stake structure.

    ``canonical_obs`` (review 2026-09-20 D3): the STUDENT row stores the
    solver-root canonical obs postflop — the form ``PolicyNetHost`` serves —
    while the teacher is still queried on its own live obs.
    """
    from plo5bp.gto.obs_from_label import canonical_serve_obs

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

            canon = (
                canonical_serve_obs(info.raw_obs, bb=cfg.bb)
                if canonical_obs
                else None
            )
            rows.append(
                SupervisedRow(
                    obs=np.asarray(
                        adapt(obs) if canon is None else canon, dtype=np.float32
                    ).copy(),
                    gate_mask=np.asarray(info.gate_mask, dtype=bool).copy(),
                    sizing=sizing_from_info(info).astype(np.int64),
                    gate_probs=gate_p,
                    anchor_probs=ap,
                    value_bb=float(dist.get("value_bb", 0.0)),
                    street=int(info.raw_obs.get("street", 0)),
                    prov=RowProvenance(
                        source="teacher_distill",
                        num_seats=n_seats,
                        obs_form="live" if canon is None else "canonical",
                    ),
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


def provenance_from_label(lab: LabelRecord, *, obs_form: str = "") -> RowProvenance:
    """Teacher provenance of one LabelRecord, read from the RECORD itself."""
    notes = lab.notes or {}

    def _num(key: str) -> float | None:
        v = notes.get(key)
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return None

    return RowProvenance(
        root_id=str(lab.root_name or ""),
        source=str(lab.source or ""),
        num_seats=int(lab.num_seats),
        expl_bb=_num("exploitability_bb"),
        expl_verified=bool(notes.get("expl_verified") is True),
        teacher_cap_bb=_num("teacher_max_expl_bb"),
        obs_form=obs_form,
    )


def rows_from_label_records(
    labels: Sequence[LabelRecord],
    *,
    obs_fn=None,
    synthesize_obs: bool = True,
    include_lossy_obs: bool = False,
    stats: Any | None = None,
    kept: list[int] | None = None,
) -> list[SupervisedRow]:
    """Convert LabelRecords to SupervisedRows (THE row builder).

    ``kept`` (optional out-list) receives the index into ``labels`` of every
    returned row, so callers can align rows back to their records.

    Priority for obs:
      1. ``notes['obs']`` if present
      2. ``obs_fn(lab)`` if given
      3. ``obs_from_label`` synthesis (``OBS_DIM_NLH``) when ``synthesize_obs``

    (review 2026-09-20 D16) Postflop labels whose obs fell back to the LOSSY
    synthetic dict are counted, logged loudly and EXCLUDED unless
    ``include_lossy_obs=True``. ``stats`` (an
    :class:`plo5bp.gto.obs_from_label.ObsSynthesisStats`) receives the tally.

    (review 2026-09-20 D1/D2) Teacher mass on a gate / anchor the serve masks
    out raises :class:`IllegalTeacherMassError` (naming the root) instead of
    being renormalized away. Labels exported before the fix trip this —
    re-export them.
    """
    from plo5bp.gto.obs_from_label import (
        OBS_KIND_ENGINE,
        OBS_KIND_LOSSY,
        OBS_KIND_NO_HOLE,
        ObsSynthesisStats,
        obs_from_label_detailed,
    )

    if stats is None:
        stats = ObsSynthesisStats()
    k = NLH_ANCHOR_SPEC.count
    rows: list[SupervisedRow] = []
    for i_lab, lab in enumerate(labels):
        obs = None
        kind = "attached"
        if lab.notes.get("obs") is not None:
            obs = np.asarray(lab.notes["obs"], dtype=np.float32)
        elif obs_fn is not None:
            obs, kind = obs_fn(lab), "custom"
        elif synthesize_obs:
            obs, kind = obs_from_label_detailed(lab)
        if obs is None:
            stats.record(OBS_KIND_NO_HOLE, root=str(lab.root_name))
            continue
        if kind == OBS_KIND_LOSSY and not include_lossy_obs:
            stats.record(kind, root=str(lab.root_name), excluded=True)
            continue
        stats.record(kind, root=str(lab.root_name))

        where = f"root {lab.root_name!r} node {lab.solve_id!r}"
        fold_legal = int(lab.to_call_chips) > 0
        raise_legal = int(lab.max_raise_chips) > 0
        gate_mask = np.array([fold_legal, True, raise_legal], dtype=bool)
        g = np.asarray(lab.gate_probs, dtype=np.float32).copy()
        if len(g) != 3:
            g = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        masked_gate = float(np.clip(g, 0.0, None)[~gate_mask].sum())
        if masked_gate > ILLEGAL_MASS_TOL:
            raise IllegalTeacherMassError(
                f"{where}: {masked_gate:.6g} of gate mass on an illegal gate "
                f"(gate_probs={g.tolist()} mask={gate_mask.tolist()}, "
                f"to_call={lab.to_call_chips} max_raise={lab.max_raise_chips}) "
                f"— re-export the labels (review 2026-09-20 D1/D2)"
            )
        g = g * gate_mask.astype(np.float32)
        if float(g.sum()) <= 0:
            g = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            g /= g.sum()

        bad_anchor = illegal_teacher_mass(
            lab.action_probs,
            min_raise=lab.min_raise_chips,
            max_raise=lab.max_raise_chips,
            pot_chips=lab.pot_chips,
            to_call=lab.to_call_chips,
        )
        if bad_anchor > ILLEGAL_MASS_TOL:
            raise IllegalTeacherMassError(
                f"{where}: {bad_anchor:.6g} of teacher mass on serve-illegal "
                f"actions (sizing min={lab.min_raise_chips} "
                f"max={lab.max_raise_chips} pot={lab.pot_chips} "
                f"to_call={lab.to_call_chips}) — re-export the labels "
                f"(review 2026-09-20 D1)"
            )
        ap = np.zeros(k, dtype=np.float32)
        for a in lab.action_probs:
            if a.gate == "raise" and a.anchor_k is not None:
                if 0 <= int(a.anchor_k) < k:
                    ap[int(a.anchor_k)] += float(a.prob)
        s = float(ap.sum())
        if s > 0:
            ap /= s
        elif raise_legal:
            # No sizing info (raise weight is 0): park on the LEGAL jam anchor.
            ap[
                jam_anchor_index(
                    min_raise=lab.min_raise_chips,
                    max_raise=lab.max_raise_chips,
                    pot=lab.pot_chips,
                    to_call=lab.to_call_chips,
                )
            ] = 1.0
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
                value_bb=0.0 if lab.value_bb is None else float(lab.value_bb),
                street=int(lab.street),
                value_mask=lab.value_bb is not None,
                prov=provenance_from_label(
                    lab,
                    obs_form="canonical" if kind == OBS_KIND_ENGINE else kind,
                ),
            )
        )
        if kept is not None:
            kept.append(i_lab)
    stats.log(prefix="[gto-rows]")
    return rows


def load_label_shard_rows(
    path: Path | str,
    *,
    synthesize_obs: bool = True,
    include_lossy_obs: bool = False,
    stats: Any | None = None,
) -> list[SupervisedRow]:
    """Load LabelRecord JSONL → supervised rows (obs synthesized by default)."""
    return rows_from_label_records(
        list(read_jsonl(path)),
        synthesize_obs=synthesize_obs,
        include_lossy_obs=include_lossy_obs,
        stats=stats,
    )


def load_label_shards(
    paths: Sequence[Path | str],
    *,
    synthesize_obs: bool = True,
    include_lossy_obs: bool = False,
    stats: Any | None = None,
) -> list[SupervisedRow]:
    """Load multiple JSONL shards into one row list."""
    rows: list[SupervisedRow] = []
    for p in paths:
        rows.extend(
            load_label_shard_rows(
                p,
                synthesize_obs=synthesize_obs,
                include_lossy_obs=include_lossy_obs,
                stats=stats,
            )
        )
    return rows


class StaleObsCacheError(ValueError):
    """A cached row bundle embeds obs from another semantics revision."""


def save_rows_npz(path: Path | str, rows: Sequence[SupervisedRow]) -> None:
    """Compact numpy bundle for fast reload.

    Provenance is root-level, so it is stored as a small JSON table plus a
    per-row index (no pickle). The bundle EMBEDS encoded obs, so it is stamped
    with the observation semantics revision it was built under
    (:mod:`plo5bp.gto.obs_rev`).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table: list[RowProvenance] = []
    index: dict[RowProvenance, int] = {}
    prov_idx = np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        j = index.get(r.prov)
        if j is None:
            j = index[r.prov] = len(table)
            table.append(r.prov)
        prov_idx[i] = j
    np.savez_compressed(
        path,
        obs=np.stack([r.obs for r in rows]),
        gate_mask=np.stack([r.gate_mask for r in rows]),
        sizing=np.stack([r.sizing for r in rows]),
        gate_probs=np.stack([r.gate_probs for r in rows]),
        anchor_probs=np.stack([r.anchor_probs for r in rows]),
        value_bb=np.asarray([r.value_bb for r in rows], dtype=np.float32),
        street=np.asarray([r.street for r in rows], dtype=np.int64),
        value_mask=np.asarray([bool(r.value_mask) for r in rows], dtype=bool),
        obs_rev=np.asarray(current_obs_rev(), dtype=np.int64),
        prov_idx=prov_idx,
        prov_table=np.asarray(json.dumps([asdict(p) for p in table])),
    )


def load_rows_npz(
    path: Path | str, *, allow_stale_obs: bool = False
) -> list[SupervisedRow]:
    """Reload a row bundle. Raises :class:`StaleObsCacheError` when its obs
    were encoded under another semantics revision (no stamp ⇒ revision 1), so
    a stale cache is REBUILT rather than silently reused."""
    data = np.load(path, allow_pickle=False)
    have = int(data["obs_rev"]) if "obs_rev" in data else UNSTAMPED_OBS_REV
    if have != current_obs_rev() and not allow_stale_obs:
        raise StaleObsCacheError(
            f"{path}: cached obs are semantics revision {have}, this process "
            f"encodes revision {current_obs_rev()} (PLO5BP_OBS_REV) — rebuild "
            f"the bundle (or pass allow_stale_obs=True to study the old one)"
        )
    n = data["obs"].shape[0]
    # Bundles written before 2026-09-20 carry no value mask / provenance.
    value_mask = data["value_mask"] if "value_mask" in data else np.ones(n, bool)
    if "prov_table" in data and "prov_idx" in data:
        table = [RowProvenance(**d) for d in json.loads(str(data["prov_table"]))]
        prov_idx = data["prov_idx"]
    else:
        table, prov_idx = [_NO_PROVENANCE], np.zeros(n, dtype=np.int64)
    return [
        SupervisedRow(
            obs=data["obs"][i],
            gate_mask=data["gate_mask"][i].astype(bool),
            sizing=data["sizing"][i],
            gate_probs=data["gate_probs"][i],
            anchor_probs=data["anchor_probs"][i],
            value_bb=float(data["value_bb"][i]),
            street=int(data["street"][i]),
            value_mask=bool(value_mask[i]),
            prov=table[int(prov_idx[i])],
        )
        for i in range(n)
    ]
