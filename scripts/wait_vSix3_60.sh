#!/bin/bash
set -e
for i in $(seq 1 90); do
  if test -f /workspace/plodbnet/checkpoints/vSix3_60.pt; then
    ls -la /workspace/plodbnet/checkpoints/vSix3_60.pt
    grep -E 'update +[0-9]+' /workspace/plodbnet/runs/vSix3.log | tail -5
    tail -3 /workspace/plodbnet/runs/vSix3_guardian.log
    echo READY
    exit 0
  fi
  latest=$(ls -t /workspace/plodbnet/checkpoints/vSix3_*.pt 2>/dev/null | head -1 | xargs -n1 basename)
  g=$(tail -1 /workspace/plodbnet/runs/vSix3_guardian.log)
  echo "waiting $(date -u +%H:%M:%S) latest=$latest $g"
  sleep 60
done
echo TIMEOUT
exit 1
