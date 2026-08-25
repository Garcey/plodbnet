#!/bin/bash
set -e
echo "=== mounts (relevant) ==="
df -h / /workspace 2>/dev/null
echo
echo "=== /workspace top ==="
du -sh /workspace/* /workspace/.[!.]* 2>/dev/null | sort -hr
echo
echo "=== plodbnet top ==="
du -sh /workspace/plodbnet/* /workspace/plodbnet/.[!.]* 2>/dev/null | sort -hr | head -30
echo
echo "=== checkpoints by stem ==="
python3 <<'PY'
import os, re, collections
os.chdir("/workspace/plodbnet/checkpoints")
c, s = collections.Counter(), collections.Counter()
for f in os.listdir("."):
    if not f.endswith(".pt"):
        continue
    sz = os.path.getsize(f)
    stem = re.sub(r"_\d+\.pt$", "", f)
    if stem.endswith(".pt"):
        stem = stem[:-3]
    c[stem] += 1
    s[stem] += sz
for stem, n in c.most_common(40):
    print(f"{n:4d} files  {s[stem]/1e9:6.2f} GB  {stem}")
print(f"TOTAL {sum(s.values())/1e9:.2f} GB  {sum(c.values())} files")
PY
echo
echo "=== runs (largest) ==="
du -sh /workspace/plodbnet/runs/* 2>/dev/null | sort -hr | head -20
echo
echo "=== stray *.pt under plodbnet root ==="
ls -lhS /workspace/plodbnet/*.pt 2>/dev/null | head -20 || true
echo
echo "=== target / venv / cargo / rustup ==="
du -sh /workspace/plodbnet/target /workspace/plodbnet/.venv /workspace/.cargo /workspace/.rustup 2>/dev/null
echo
echo "=== root container overlay (not volume) ==="
du -sh /usr /opt /tmp /var /root 2>/dev/null | sort -hr
