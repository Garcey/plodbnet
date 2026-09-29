"""Guardian helpers (2026-09-28, ML-024 / ML-035 / ML-057): the decisions the
pod's bash guardians make, in one tested place (scripts/guardian_lib.sh calls
this module).

    python -m plo5bp.train.guardian pick-warm --dir checkpoints --stem vSix6 \\
        --need hidden_dim=1024 --need obs_rev=1 --need-own critic_act=silu \\
        [--also checkpoints/r2b_1299.pt]
    python -m plo5bp.train.guardian heartbeat-age runs/vSix6.heartbeat --pid 1234

`pick-warm` resumes from the HIGHEST numbered `<stem>_<N>.pt` (not the newest
by modification time, which a cp / rsync / touch changes), or from the rolling
`<stem>.pt` when its stamped counter says it is newer than every numbered file
(the first update after a (re)launch writes no numbered file); it skips files
that are not compatible with the launch's requirements, then tries `--also`.
`heartbeat-age` says how long ago the trainer with that PID finished an update,
or -1 = UNKNOWN: no heartbeat file, one written by another process, one written
before this process started, or a start time that cannot be read. Unknown never
kills: a trainer from before 2026-09-28 writes no heartbeat, so a guardian that
reaches the pod ahead of the matching train.py falls back to its PID watch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def numbered(ckpt_dir: "str | Path", stem: str) -> "dict[int, Path]":
    """{N: path} of `<stem>_<N>.pt` (N all digits; `<stem>.optim.pt`,
    `*.pt.tmp` and other stems never match)."""
    pat = re.compile(rf"^{re.escape(stem)}_(\d+)\.pt$")
    out: dict[int, Path] = {}
    for p in Path(ckpt_dir).glob(f"{stem}_*.pt"):
        m = pat.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def _load(path: Path) -> "dict | None":
    try:
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=False)
        return ck if isinstance(ck, dict) else None
    except Exception:  # noqa: BLE001 -- a torn / foreign file is just not a candidate
        return None


def _facts(ck: dict) -> dict:
    """What a requirement can name: the actor's size / layout / revision from
    the checkpoint itself, then every config stamp (critic_* etc.)."""
    from plo5bp.evaluation.loader import checkpoint_meta

    facts = {str(k): v for k, v in (ck.get("config") or {}).items()}
    try:
        meta = checkpoint_meta(ck)
        facts.update({k: meta[k] for k in ("hidden_dim", "num_layers", "obs_mode", "obs_rev",
                                           "variant", "head_version")})
    except Exception:  # noqa: BLE001 -- not an actor checkpoint
        facts.setdefault("obs_rev", int(ck.get("obs_rev", 1)))
    facts.setdefault("critic_act", "relu")
    return facts


def compatible(path: "str | Path", need: "dict[str, str]", need_own: "dict[str, str]",
               stem: str) -> bool:
    """Does the checkpoint satisfy every `need` (and, for this stem's own files,
    every `need_own`)? Values compare as strings."""
    ck = _load(Path(path))
    if ck is None or "model" not in ck:
        return False
    facts = _facts(ck)
    reqs = dict(need)
    if Path(path).name.startswith(stem):
        reqs.update(need_own)
    return all(str(facts.get(k)) == str(v) for k, v in reqs.items())


def candidates(ckpt_dir: "str | Path", stem: str) -> "list[Path]":
    """Resume candidates, best first: the numbered files by N descending, with
    the rolling `<stem>.pt` first when its counter (a final save stamps the
    COUNT of updates, one more than the numbered file of the same weights)
    says it is newer than all of them."""
    nums = numbered(ckpt_dir, stem)
    order = [nums[n] for n in sorted(nums, reverse=True)]
    rolling = Path(ckpt_dir) / f"{stem}.pt"
    if rolling.exists():
        ck = _load(rolling)
        counter = None if ck is None else ck.get("update_counter")
        top = max(nums) if nums else -1
        if counter is not None and int(counter) - 1 > top:
            order.insert(0, rolling)
        else:
            order.append(rolling)
    return order


def pick_warm(ckpt_dir, stem, need, need_own, also=()) -> "Path | None":
    for p in list(candidates(ckpt_dir, stem)) + [Path(a) for a in also if a]:
        if p.exists() and compatible(p, need, need_own, stem):
            return p
    return None


def process_start_time(pid: int) -> "float | None":
    """When process `pid` started (epoch seconds), or None when this platform
    cannot say (Linux: /proc/<pid>/stat + boot time; psutil if installed)."""
    try:
        import psutil  # type: ignore

        return float(psutil.Process(int(pid)).create_time())
    except Exception:  # noqa: BLE001 -- psutil missing / no such process
        pass
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        start_ticks = int(fields[19])  # field 22 of stat: starttime, clock ticks after boot
        with open("/proc/stat", encoding="utf-8") as f:
            btime = next(int(ln.split()[1]) for ln in f if ln.startswith("btime"))
        return btime + start_ticks / os.sysconf("SC_CLK_TCK")
    except Exception:  # noqa: BLE001 -- not Linux, or the process is gone
        return None


def heartbeat_age(path: "str | Path", pid: "int | None" = None,
                  started: "float | None" = None) -> float:
    """Seconds since THIS trainer (`pid`) last wrote its heartbeat, else -1
    (unknown: missing / unreadable file, another PID's, written before the
    process started -- e.g. left by an earlier trainer that happened to get the
    same PID -- or no way to read the process's start time). `started`
    overrides the start-time lookup (tests)."""
    try:
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return -1.0
    if pid is not None:
        if int(rec.get("pid", -1)) != int(pid):
            return -1.0
        start = started if started is not None else process_start_time(int(pid))
        if start is None or float(rec.get("time", 0.0)) < start - 1.0:
            return -1.0
    return max(0.0, time.time() - float(rec.get("time", 0.0)))


def _kv(items: "list[str]") -> "dict[str, str]":
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep:
            raise SystemExit(f"expected key=value, got {it!r}")
        out[k.strip()] = v.strip()
    return out


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pw = sub.add_parser("pick-warm")
    pw.add_argument("--dir", default="checkpoints")
    pw.add_argument("--stem", required=True)
    pw.add_argument("--need", action="append", default=[])
    pw.add_argument("--need-own", action="append", default=[])
    pw.add_argument("--also", action="append", default=[])
    hb = sub.add_parser("heartbeat-age")
    hb.add_argument("path")
    hb.add_argument("--pid", type=int, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "pick-warm":
        p = pick_warm(args.dir, args.stem, _kv(args.need), _kv(args.need_own), args.also)
        if p is None:
            return 1
        print(p.as_posix())
        return 0
    print(int(heartbeat_age(args.path, args.pid)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
