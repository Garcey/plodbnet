#!/usr/bin/env python
"""Native NLH CFR solve CLI.

Examples::

  .venv/Scripts/python scripts/cfr_solve.py --preflop --stack-bb 100 --iters 200
  .venv/Scripts/python scripts/cfr_solve.py --street 3 \\
      --board 48,44,40,36,32 --pot-bb 20 --stack-bb 50 --iters 300
  .venv/Scripts/python scripts/cfr_solve.py --kuhn --iters 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "python"))

from plo5bp.gto.cfr_api import (  # noqa: E402
    SIZE_PRESETS,
    RootSpec,
    SolveConfig,
    rust_cfr_available,
    solve,
    solve_kuhn,
)


def main() -> int:
    p = argparse.ArgumentParser(description="NLH CFR solve")
    p.add_argument("--preflop", action="store_true", help="HU preflop root")
    p.add_argument("--kuhn", action="store_true", help="Kuhn poker gate")
    p.add_argument(
        "--pipeline",
        action="store_true",
        help="Preflop MCCFR -> induce -> postflop DCFR",
    )
    p.add_argument("--street", type=int, default=None, help="0..3")
    p.add_argument("--board", type=str, default="", help="Comma card indices 0..51")
    p.add_argument("--pot-bb", type=float, default=10.0)
    p.add_argument("--stack-bb", type=float, default=100.0)
    p.add_argument("--num-seats", type=int, default=2)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--algorithm", type=str, default="dcfr")
    p.add_argument(
        "--size-preset",
        type=str,
        default="standard",
        choices=list(SIZE_PRESETS.keys()),
    )
    p.add_argument("--target-expl", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    avail = rust_cfr_available()
    print(f"[cfr] rust_cfr_available={avail}")

    if args.kuhn:
        if not avail:
            print("[cfr] need maturin develop for kuhn", file=sys.stderr)
            return 1
        rep = solve_kuhn(iterations=args.iters)
        print(json.dumps(rep, indent=2))
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        return 0

    if args.pipeline:
        if not avail:
            print("[cfr] need maturin develop for pipeline", file=sys.stderr)
            return 1
        from plo5bp import _engine  # type: ignore

        board = [int(x) for x in args.board.split(",") if x.strip() != ""] or [
            0,
            5,
            10,
            15,
            20,
        ]
        rep = dict(
            _engine.cfr_pipeline(
                stack_bb=float(args.stack_bb),
                preflop_iters=int(args.iters),
                postflop_iters=max(40, args.iters // 2),
                postflop_board=board,
                pot_bb=float(args.pot_bb),
                postflop_stack_bb=float(args.stack_bb) * 0.4,
                seed=int(args.seed),
            )
        )
        print(json.dumps(rep, indent=2))
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        return 0

    if args.preflop:
        root = RootSpec.preflop_hu(stack_bb=args.stack_bb)
        if args.algorithm == "dcfr":
            args.algorithm = "mccfr_es"
    else:
        street = 3 if args.street is None else int(args.street)
        board = [int(x) for x in args.board.split(",") if x.strip() != ""]
        sizes = list(SIZE_PRESETS[args.size_preset])
        root = RootSpec(
            street=street,
            pot_bb=args.pot_bb,
            effective_stack_bb=args.stack_bb,
            board=board,
            num_seats=args.num_seats,
            raise_sizes_pm=sizes,
        )

    cfg = SolveConfig(
        max_iterations=args.iters,
        thread_num=args.threads,
        seed=args.seed,
        algorithm=args.algorithm,
        target_exploitability_bb=args.target_expl,
    )
    try:
        report = solve(root, cfg)
    except ValueError as e:
        print(f"[cfr] invalid: {e}", file=sys.stderr)
        return 2

    print(json.dumps(report.as_dict(), indent=2))
    if args.out is not None:
        report.write_json(args.out)
        print(f"[cfr] wrote {args.out}")

    if report.status == "not_implemented":
        print("[cfr] binding missing — run: .venv/Scripts/maturin develop --release")
        return 1
    if report.status != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
