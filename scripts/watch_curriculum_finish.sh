#!/bin/bash
# Watch for orchestrator completion, then promote final p22_110bb checkpoint to UI.

ORCH_LOG=runs/orchestrator_timeline.log
WATCH_LOG=runs/curriculum_finish_watcher.log

echo "[$(date '+%H:%M:%S')] watcher start" > "$WATCH_LOG"

# Sanity timeout: 4 hours from now
DEADLINE=$(($(date +%s) + 14400))

while true; do
  if grep -q "ORCHESTRATOR COMPLETE" "$ORCH_LOG" 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] orchestrator completion detected" >> "$WATCH_LOG"
    break
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "[$(date '+%H:%M:%S')] timeout — abandoning watcher" >> "$WATCH_LOG"
    exit 1
  fi
  sleep 30
done

# Find final phase's latest snapshot
LAST=$(ls -t checkpoints/aggr_seq_p22_110bb_*.pt 2>/dev/null | head -1)
if [ -z "$LAST" ] || [ ! -f "$LAST" ]; then
  if [ -f checkpoints/aggr_seq_p22_110bb.pt ]; then
    LAST=checkpoints/aggr_seq_p22_110bb.pt
  fi
fi

if [ -z "$LAST" ] || [ ! -f "$LAST" ]; then
  echo "[$(date '+%H:%M:%S')] ERROR: no p22 checkpoint found" >> "$WATCH_LOG"
  exit 1
fi

echo "[$(date '+%H:%M:%S')] promoting $LAST" >> "$WATCH_LOG"
cp checkpoints/stub.pt checkpoints/stub.pt.bak_pre_curriculum_final
cp "$LAST" checkpoints/stub.pt

# Restart UI
PID_UI=$(netstat -ano 2>&1 | grep ":8765.*LISTENING" | awk '{print $NF}' | head -1)
if [ -n "$PID_UI" ]; then
  taskkill //F //PID "$PID_UI" > /dev/null 2>&1
  echo "[$(date '+%H:%M:%S')] killed old UI PID=$PID_UI" >> "$WATCH_LOG"
fi

.venv/Scripts/python -m uvicorn plo5bp.ui.server:app --port 8765 > runs/ui_curriculum_final.log 2>&1 &
sleep 5
NEW_PID=$(netstat -ano 2>&1 | grep ":8765.*LISTENING" | awk '{print $NF}' | head -1)
echo "[$(date '+%H:%M:%S')] UI restarted, new PID=$NEW_PID, served from $LAST" >> "$WATCH_LOG"
echo "[$(date '+%H:%M:%S')] DONE" >> "$WATCH_LOG"
