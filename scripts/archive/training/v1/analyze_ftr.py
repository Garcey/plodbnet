import re
from collections import defaultdict

pat = re.compile(
    r"update\s+(\d+).*?bonus%\(F/T/R\)=\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+).*?seats=(\d+).*?block=\d+/3\((\w+)\)"
)

# Bash /tmp on git-bash usually maps to %TEMP% (e.g. C:\Users\themi\AppData\Local\Temp).
# But that path isn't visible to ls /tmp here — read via the windows path instead.
import os
candidates = [
    "/tmp/optimized_updates.txt",
    os.path.expandvars(r"%TEMP%\optimized_updates.txt"),
    r"C:\Users\themi\AppData\Local\Temp\optimized_updates.txt",
]
src = next((p for p in candidates if os.path.exists(p)), None)
if src is None:
    raise SystemExit(f"updates file not found in {candidates}")
print(f"reading {src}")
print()

rows = []
with open(src) as f:
    for line in f:
        m = pat.search(line)
        if not m:
            continue
        u, F, T, R, seats, block = m.groups()
        rows.append({
            "u": int(u),
            "F": float(F),
            "T": float(T),
            "R": float(R),
            "seats": int(seats),
            "block": block,
        })

print(f"parsed {len(rows)} updates, u{rows[0]['u']}..u{rows[-1]['u']}")
print()


def stats(samples):
    if not samples:
        return None
    F = [s["F"] for s in samples]
    T = [s["T"] for s in samples]
    R = [s["R"] for s in samples]
    return (
        sum(F) / len(F),
        sum(T) / len(T),
        sum(R) / len(R),
        len(samples),
    )


def fmt(label, samples):
    s = stats(samples)
    if s is None:
        print(f"  {label}: (none)")
        return
    F, T, R, n = s
    print(f"  {label}: F/T/R={F:5.1f}/{T:5.1f}/{R:5.1f}  n={n}")


print("=== overall by block ===")
for block in ["clubgg", "clubgg_deep", "deep"]:
    fmt(f"{block:11s}", [r for r in rows if r["block"] == block])
print()

print("=== by seats (all blocks) ===")
for s in sorted({r["seats"] for r in rows}):
    fmt(f"{s} seats", [r for r in rows if r["seats"] == s])
print()

print("=== block x seats (>=3) ===")
for block in ["clubgg", "clubgg_deep", "deep"]:
    print(f"  -- {block} --")
    for s in sorted({r["seats"] for r in rows if r["block"] == block and r["seats"] >= 3}):
        fmt(f"  {s}s", [r for r in rows if r["block"] == block and r["seats"] == s])
print()

print("=== windowed trend, 50-update buckets, 3-6 seats only ===")
filtered = [r for r in rows if r["seats"] >= 3]
window = 50
buckets = defaultdict(list)
for r in filtered:
    buckets[(r["u"] // window) * window].append(r)
for start in sorted(buckets):
    fmt(f"u{start:3d}-u{start+window-1}", buckets[start])
print()

print("=== 30-update rolling tail (3-6 seats) ===")
end = rows[-1]["u"]
for off in range(0, min(120, end + 1) + 1, 30):
    lo, hi = end - 29 - off, end - off
    win = [r for r in rows if lo <= r["u"] <= hi and r["seats"] >= 3]
    fmt(f"u{lo}..u{hi}", win)
