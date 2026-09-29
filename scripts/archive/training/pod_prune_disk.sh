#!/bin/bash
# Safe disk prune on the training volume: incomplete profile dump +
# keep only the latest 8 numbered checkpoints for optimized7 / optimized9.
set -euo pipefail
cd /workspace/plodbnet

echo "=== before ==="
ls -lh runs/profile_update2.json.tmp 2>/dev/null || echo "profile tmp already gone"
echo "optimized7 count: $(ls checkpoints/optimized7_*.pt 2>/dev/null | wc -l)"
echo "optimized9 count: $(ls checkpoints/optimized9_*.pt 2>/dev/null | wc -l)"
df -h /workspace | tail -1

# 1) incomplete Chrome-trace dump (~19G) — not needed for training
if [ -f runs/profile_update2.json.tmp ]; then
  echo "deleting runs/profile_update2.json.tmp ..."
  rm -f runs/profile_update2.json.tmp
  echo "deleted profile_update2.json.tmp"
fi

# 2) prune a stem to the latest $keep by update number
prune_stem() {
  local stem="$1"
  local keep="${2:-8}"
  local dir=checkpoints
  local -a nums
  mapfile -t nums < <(
    ls -1 "$dir"/${stem}_*.pt 2>/dev/null \
      | sed -E "s|.*${stem}_([0-9]+)\.pt|\1|" \
      | sort -n
  )
  local n=${#nums[@]}
  if [ "$n" -eq 0 ]; then
    echo "$stem: no files"
    return
  fi
  if [ "$n" -le "$keep" ]; then
    echo "$stem: only $n files (<= $keep), nothing to prune"
    return
  fi
  local drop=$((n - keep))
  echo "$stem: $n files -> keep $keep latest, delete $drop older"
  local i
  for ((i = 0; i < drop; i++)); do
    rm -f "$dir/${stem}_${nums[i]}.pt"
  done
  echo "$stem kept:"
  ls -1 "$dir"/${stem}_*.pt | xargs -n1 basename | sort -t_ -k2 -n
}

prune_stem optimized7 8
prune_stem optimized9 8

echo
echo "=== after ==="
echo "optimized7 count: $(ls checkpoints/optimized7_*.pt 2>/dev/null | wc -l)"
echo "optimized9 count: $(ls checkpoints/optimized9_*.pt 2>/dev/null | wc -l)"
du -sh checkpoints runs 2>/dev/null
df -h /workspace | tail -1
echo "done"
