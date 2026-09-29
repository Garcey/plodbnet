"""Is the training pipeline still BIT-EXACT? (docs/training.md "Second efficiency pass")

Trains a tiny recipe with a git revision's code (default HEAD) and with the
working tree, then compares the SHA-256 of every tensor of every checkpoint
both runs wrote (numbered checkpoints, the final save and the optimizer
sidecar). Exit code 0 = identical, 1 = a difference (the moved tensors are
listed), 2 = a run failed.

    .venv/Scripts/python scripts/exactness_check.py                  # tiny + v6, CPU, vs HEAD
    .venv/Scripts/python scripts/exactness_check.py --recipe all
    .venv/Scripts/python scripts/exactness_check.py --ref 80048a0 --recipe v6
    .venv/Scripts/python scripts/exactness_check.py --same           # determinism: the tree twice
    python scripts/exactness_check.py --device cuda --recipe v6      # on the pod

Both sides use THIS tree's compiled engine (a Python-level check); pass
--ref-engine to give the reference its own build. Library + recipe list:
python/plo5bp/exactness.py. The same check runs in pytest
(tests/python/training/test_exactness.py, the "smoke" recipe).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from plo5bp import exactness as ex  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ref", default="HEAD",
                    help="git revision of the reference code (default HEAD)")
    ap.add_argument("--ref-dir", type=Path, default=None,
                    help="use this checkout (repo root with python/ and scripts/) "
                    "as the reference instead of --ref")
    ap.add_argument("--ref-engine", type=Path, default=None,
                    help="engine binary for an exported --ref tree (default: a "
                    "copy of this tree's)")
    ap.add_argument("--same", action="store_true",
                    help="determinism check: the working tree against itself")
    ap.add_argument("--recipe", action="append", default=None,
                    help=f"one of {', '.join(ex.RECIPES)} or 'all' (repeatable; "
                    "default: tiny and v6)")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--updates", type=int, default=None,
                    help="updates per run (default: the recipe's, 3)")
    ap.add_argument("--serial", action="store_true",
                    help="run the two sides one after the other (always on CUDA)")
    ap.add_argument("--keep", type=Path, default=None,
                    help="keep the runs (checkpoints + logs) under this directory")
    ap.add_argument("--list", action="store_true", help="list the recipes and exit")
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="after `--`: extra train.py flags for BOTH sides")
    args = ap.parse_args()

    if args.list:
        for name, r in ex.RECIPES.items():
            print(f"{name:8s} {r.description}")
        return 0
    names = args.recipe or ["tiny", "v6"]
    if "all" in names:
        names = list(ex.RECIPES)
    unknown = [n for n in names if n not in ex.RECIPES]
    if unknown:
        ap.error(f"unknown recipe(s) {unknown}; known: {list(ex.RECIPES)}")
    extra = tuple(a for a in args.extra if a != "--")

    root = Path(args.keep) if args.keep else ex.scratch_dir()
    root.mkdir(parents=True, exist_ok=True)
    if args.same:
        ref_tree, ref_label = REPO, "working tree (determinism)"
    elif args.ref_dir is not None:
        ref_tree, ref_label = args.ref_dir.resolve(), str(args.ref_dir)
    else:
        ref_tree = ex.export_git_tree(args.ref, root / "ref_tree", engine=args.ref_engine)
        ref_label = f"git {args.ref}"
        changed = ex.training_changes(REPO, args.ref)
        print(f"[exact] training sources changed vs {args.ref}: "
              f"{', '.join(changed) if changed else 'none'}")
    print(f"[exact] reference = {ref_label}; new = working tree; device {args.device}; "
          f"runs under {root}")

    rc = 0
    for name in names:
        t0 = time.time()
        try:
            res = ex.check(
                name, ref_tree, REPO, root / name, device=args.device,
                updates=args.updates,
                parallel=False if (args.serial or args.device == "cuda") else None,
                extra_args=extra,
            )
        except RuntimeError as e:
            print(f"[exact] {name}: RUN FAILED\n{e}")
            rc = max(rc, 2)
            continue
        dt = time.time() - t0
        if res.identical:
            n = sum(1 for _ in ex.checkpoint_digests(res.new_dir))
            print(f"[exact] {name}: IDENTICAL ({n} files, every tensor) in {dt:.0f}s")
        else:
            print(f"[exact] {name}: DIFFERENT ({dt:.0f}s)")
            for line in res.differences:
                print(f"          {line}")
            rc = max(rc, 1)
    if args.keep is None and rc == 0:
        shutil.rmtree(root, ignore_errors=True)
    elif rc != 0:
        print(f"[exact] runs kept for inspection: {root}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
