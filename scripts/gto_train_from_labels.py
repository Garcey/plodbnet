#!/usr/bin/env python
"""Train PolicyNet from native rust_cfr LabelRecord JSONL (+ optional bootstrap mix).

  .venv/Scripts/python scripts/gto_train_from_labels.py \\
      --labels data/gto_nlh/pushfold_4h_labels.jsonl \\
      --epochs 10 --out checkpoints/gto_policy.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.bootstrap import collect_bootstrap  # noqa: E402
from plo5bp.gto.dataset import save_rows_npz  # noqa: E402
from plo5bp.gto.labels import read_jsonl  # noqa: E402
from plo5bp.gto.obs_from_label import labels_to_supervised_rows  # noqa: E402
from plo5bp.gto.train import TrainConfig, train_policy_net  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=Path, nargs="+", required=True)
    p.add_argument("--bootstrap-n", type=int, default=0,
                   help="Extra rule-bootstrap rows mixed in (preflop coverage)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("checkpoints/gto_policy.pt"))
    p.add_argument("--save-rows", type=Path, default=None)
    p.add_argument(
        "--load",
        type=Path,
        default=None,
        help="Warm-start PolicyNet weights (same hidden_dim / num_layers)",
    )
    args = p.parse_args()

    labels = []
    for path in args.labels:
        got = list(read_jsonl(path))
        print(f"[train] {path}: {len(got)} label records")
        labels.extend(got)

    rows = labels_to_supervised_rows(labels)
    print(f"[train] {len(rows)} supervised rows from solver labels")

    if args.bootstrap_n > 0:
        boot = collect_bootstrap(n_decisions=args.bootstrap_n, seed=args.seed + 1)
        print(f"[train] +{len(boot)} bootstrap rows")
        rows = rows + boot

    if not rows:
        print("no rows — need labels with hero_hole (not aggregate-only)")
        return 1

    if args.save_rows:
        save_rows_npz(args.save_rows, rows)
        print(f"[train] wrote {args.save_rows}")

    cfg = TrainConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        value_coef=0.05,
    )
    # Bootstrap mix is not badge-eligible as GTO until re-probed; source
    # must stay rust_cfr* without "bootstrap" for the GTO teacher check.
    if args.bootstrap_n > 0:
        source = "rust_cfr+bootstrap"
    else:
        source = "rust_cfr"
    result = train_policy_net(
        rows,
        args.out,
        cfg=cfg,
        meta={
            "source": source,
            "n_train": len(rows),
            "n_solver_labels": len(labels),
            "n_rows": len(rows),
            "is_gto_validated": False,  # require scripts/gto_probe.py --stamp
            "warm_start": None if args.load is None else str(args.load),
        },
        init_ckpt=args.load,
    )
    print(json.dumps(result.as_dict(), indent=2))
    print(
        "[train] badge will NOT claim GTO AI until: "
        "scripts/gto_probe.py --ckpt ... --holdout ... --stamp"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
