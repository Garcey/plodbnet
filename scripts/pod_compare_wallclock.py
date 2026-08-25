#!/usr/bin/env python3
"""Parse vSix3/vSix4 logs for per-update wall-clock."""
import re
import statistics as st
from pathlib import Path


def parse(path: Path):
    d = {}
    if not path.exists():
        return d
    for line in path.read_text().splitlines():
        m = re.search(r"\[([0-9.]+)s\] update\s+(\d+)", line)
        if m:
            d[int(m.group(2))] = float(m.group(1))
    return d


def report(name: str, d: dict):
    ups = sorted(d)
    if len(ups) < 2:
        print(f"{name}: need >=2 updates (have {len(ups)})")
        return
    dts = [d[ups[i]] - d[ups[i - 1]] for i in range(1, len(ups))]
    print(
        f"{name}: updates {ups[0]}..{ups[-1]} n={len(ups)} "
        f"mean={st.mean(dts):.1f}s median={st.median(dts):.1f}s "
        f"min={min(dts):.1f}s max={max(dts):.1f}s"
    )
    last = dts[-10:]
    print(f"  last {len(last)}: mean={st.mean(last):.1f}s {[round(x, 1) for x in last]}")


def main():
    root = Path("/workspace/plodbnet/runs")
    report("vSix3", parse(root / "vSix3.log"))
    report("vSix4", parse(root / "vSix4.log"))


if __name__ == "__main__":
    main()
