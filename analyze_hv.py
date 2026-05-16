import re
import os
from collections import defaultdict

pat = re.compile(
    r"update\s+(\d+)\s+pi=([+-]?[\d.]+)\s+v=([\d.]+)\s+H=([\d.]+)\s+kl=([+-]?[\d.]+).*?seats=(\d+).*?block=\d+/3\((\w+)\)"
)

candidates = [
    "/tmp/all_updates.txt",
    os.path.expandvars(r"%TEMP%\all_updates.txt"),
    r"C:\Users\themi\AppData\Local\Temp\all_updates.txt",
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
        u, pi, v, H, kl, seats, block = m.groups()
        rows.append({
            "u": int(u),
            "pi": float(pi),
            "v": float(v),
            "H": float(H),
            "kl": float(kl),
            "seats": int(seats),
            "block": block,
        })

print(f"parsed {len(rows)} updates, u{rows[0]['u']}..u{rows[-1]['u']}")
print()


def stats(samples, key):
    vals = [s[key] for s in samples]
    if not vals:
        return None
    return min(vals), sum(vals) / len(vals), max(vals), len(vals)


def fmt_hv(label, samples):
    if not samples:
        print(f"  {label}: (none)")
        return
    Hmin, Hmean, Hmax, n = stats(samples, "H")
    vmin, vmean, vmax, _ = stats(samples, "v")
    klmean = sum(s["kl"] for s in samples) / len(samples)
    print(f"  {label}: H={Hmin:.3f}/{Hmean:.3f}/{Hmax:.3f}  v={vmin:6.1f}/{vmean:6.1f}/{vmax:6.1f}  kl_mean={klmean:.4f}  n={n}")


print("=== overall ===")
fmt_hv("all   ", rows)
print()

print("=== windowed (50-update buckets, mixed-seat) ===")
W = 50
buckets = defaultdict(list)
for r in rows:
    buckets[(r["u"] // W) * W].append(r)
for start in sorted(buckets):
    fmt_hv(f"u{start:3d}-u{start+W-1}", buckets[start])
print()

print("=== windowed (50-update buckets, seats >= 4 only, removes 2/3-seat noise) ===")
filtered = [r for r in rows if r["seats"] >= 4]
buckets = defaultdict(list)
for r in filtered:
    buckets[(r["u"] // W) * W].append(r)
for start in sorted(buckets):
    fmt_hv(f"u{start:3d}-u{start+W-1}", buckets[start])
print()

print("=== by block (overall) ===")
for block in ["clubgg", "clubgg_deep", "deep"]:
    fmt_hv(f"{block:11s}", [r for r in rows if r["block"] == block])
print()

print("=== by seats (overall) ===")
for s in sorted({r["seats"] for r in rows}):
    fmt_hv(f"{s} seats", [r for r in rows if r["seats"] == s])
print()

print("=== 100-update buckets, by seat (3-6 only) ===")
W2 = 100
print(f"{'window':>14s}  {'seats':>5s}  H_mean  v_mean   n")
buckets = defaultdict(list)
for r in rows:
    if r["seats"] >= 3:
        buckets[((r["u"] // W2) * W2, r["seats"])].append(r)
for (start, s) in sorted(buckets):
    samples = buckets[(start, s)]
    Hmean = sum(x["H"] for x in samples) / len(samples)
    vmean = sum(x["v"] for x in samples) / len(samples)
    print(f"u{start:3d}-u{start+W2-1:3d}     {s}    {Hmean:.3f}   {vmean:6.1f}   {len(samples)}")
