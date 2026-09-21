#!/usr/bin/env python
"""After Step 7 batch: export + train + probe.

The GTO badge is DERIVED, never asserted here (review 2026-09-20 F6): the
checkpoint's label sources / teacher exploitability + cap / training root ids
come from the training records, the probe refuses a holdout that intersects
them, and ``stamp_probe_on_checkpoint`` re-derives ``is_gto_validated``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.cfr_export import export_teacher_dir
from plo5bp.gto.cfr_batch import _strategy_path
from plo5bp.gto.labels import read_jsonl
from plo5bp.gto.obs_from_label import labels_to_supervised_rows
from plo5bp.gto.policy_net import is_validated_gto_checkpoint, load_policy_checkpoint
from plo5bp.gto.probe import ProbeGates, probe_checkpoint, stamp_probe_on_checkpoint
from plo5bp.gto.teacher import TEACHER_MAX_EXPL_BB, expl_reject_reason
from plo5bp.gto.train import TrainConfig, train_policy_net


def _root_expl(path: Path) -> float | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    v = d.get("exploitability_bb")
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    strat_dir = Path("data/cfr/teacher_s7/strategies")
    labels_train = Path("data/gto_nlh/teacher_s7_train.jsonl")
    ckpt = Path("checkpoints/gto_teacher_s7.pt")
    load = Path("checkpoints/gto_teacher_ibr.pt")

    print("[step7] EXPORT", flush=True)
    res = export_teacher_dir(
        strat_dir,
        labels_train,
        source="rust_cfr_river",
        max_expl_bb=TEACHER_MAX_EXPL_BB,
    )
    print(
        f"[step7] export train={res.n_train} holdout={res.n_holdout} "
        f"skipped_expl={len(res.skipped_expl)}",
        flush=True,
    )
    print(
        f"[step7] split train_ids={res.train_root_ids} holdout_ids={res.holdout_root_ids}",
        flush=True,
    )
    if set(res.train_root_ids) & set(res.holdout_root_ids):
        print("[step7] ERROR overlapping split", file=sys.stderr)
        return 1
    if res.n_train == 0:
        print("[step7] ERROR no train labels", file=sys.stderr)
        return 1

    train_expl = {}
    for rid in res.train_root_ids:
        p = _strategy_path(Path("data/cfr/teacher_s7"), rid)
        train_expl[rid] = _root_expl(p) if p.exists() else None
    over = {
        rid: e
        for rid, e in train_expl.items()
        if expl_reject_reason(e, max_expl_bb=TEACHER_MAX_EXPL_BB) is not None
    }
    print(f"[step7] train-root expl {train_expl}", flush=True)
    if over:
        print(f"[step7] train roots over cap (will not stamp): {over}", flush=True)

    print("[step7] TRAIN", flush=True)
    labels = list(read_jsonl(labels_train))
    rows = labels_to_supervised_rows(labels)  # lossy-obs rows excluded + logged
    print(f"[step7] {len(rows)} supervised rows", flush=True)
    init = load if load.is_file() else None
    if init is not None:
        # A checkpoint from before 2026-09-20 records neither its training roots
        # nor its label provenance (and learned sizing from labels whose jam
        # mass was masked away — review D1). Warm-starting from it would make
        # this net unverifiable for good, so start clean instead.
        _m, init_meta = load_policy_checkpoint(init)
        if not init_meta.get("train_roots_known"):
            print(
                f"[step7] NOT warm-starting from {init}: no recorded training "
                f"roots / label provenance — training from scratch",
                flush=True,
            )
            init = None
    tr = train_policy_net(
        rows,
        ckpt,
        cfg=TrainConfig(
            hidden_dim=256,
            num_layers=2,
            epochs=6,
            batch_size=256,
            lr=3e-4,
            device="cpu",
            seed=0,
            log_every=40,
            value_coef=0.05,
        ),
        # No ``source`` / ``is_gto_validated`` here: train_policy_net derives
        # the source + label provenance from the rows.
        meta={
            "n_train": len(rows),
            "campaign": "teacher_s7",
            "planned_holdout_root_ids": list(res.holdout_root_ids),
        },
        init_ckpt=init,
    )
    print(json.dumps(tr.as_dict(), indent=2), flush=True)

    hold_path = res.holdout_path
    if not hold_path or not Path(hold_path).is_file() or res.n_holdout == 0:
        print("[step7] no holdout — not stamping", flush=True)
        return 0

    print("[step7] PROBE", flush=True)
    gates = ProbeGates(min_n=50)
    pr = probe_checkpoint(ckpt, hold_path, device="cpu", gates=gates)
    print(json.dumps(pr.as_dict(), indent=2), flush=True)
    if over:
        print(f"[step7] STAMP no over_cap={over}", flush=True)
        return 0
    meta = stamp_probe_on_checkpoint(ckpt, pr, device="cpu")
    print(
        f"[step7] STAMP probe_passed={pr.passed} "
        f"is_gto_validated={is_validated_gto_checkpoint(ckpt)} "
        f"note={meta.get('gto_badge_note')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
