#!/bin/bash
# Watchdog for the phase-5 training run (launch_optimized5.sh). Checks every
# INTERVAL seconds; if scripts/train.py is gone (crash, OOM, pod maintenance),
# relaunches from the highest-numbered checkpoint. Decoupled from training:
# each is its own setsid session, so killing one doesn't kill the other.
#
# Uses phase-5-specific stop/lock/log files so it never collides with a
# phase-4 watchdog. STOP cleanly:  touch runs/watchdog5.stop
#
# Crash-loop guard: 3 relaunches that each die within ~30s -> writes the stop
# file and exits, so a genuine startup bug can't thrash forever.
#
# LIMITATION: a slow failure (e.g. an OOM many hours in) is NOT "rapid", so the
# watchdog WILL keep restarting on that cadence. Insurance against transient /
# maintenance events, not a substitute for watching health. Check
# runs/optimized5.log periodically.
set -uo pipefail
cd /workspace/plodbnet

STOP=runs/watchdog5.stop
WLOG=runs/watchdog5.log
INTERVAL=60

# Single-instance guard via flock (NOT pgrep — pgrep also matches the ssh
# command string that launches this script). flock auto-releases on death, so
# exactly one phase-5 watchdog can run.
SELF=$$
exec 9>runs/watchdog5.lock
if ! flock -n 9; then
  echo "[watchdog5 $(date -u +%Y-%m-%dT%H:%M:%S)] another watchdog holds the lock; exiting." >> "$WLOG"
  exit 0
fi

rm -f "$STOP"   # fresh start clears any stale stop request
echo "[watchdog5 $(date -u +%Y-%m-%dT%H:%M:%S)] started (pid $SELF)" >> "$WLOG"

fails=0
while true; do
  if [[ -f "$STOP" ]]; then
    echo "[watchdog5 $(date -u +%Y-%m-%dT%H:%M:%S)] stop file present; exiting (training left as-is)." >> "$WLOG"
    exit 0
  fi

  if pgrep -f 'scripts/train\.py' >/dev/null; then
    fails=0
    sleep "$INTERVAL"
    continue
  fi

  echo "[watchdog5 $(date -u +%Y-%m-%dT%H:%M:%S)] training DOWN; relaunching..." >> "$WLOG"
  bash launch_optimized5.sh >> "$WLOG" 2>&1
  sleep 30

  if pgrep -f 'scripts/train\.py' >/dev/null; then
    echo "[watchdog5 $(date -u +%Y-%m-%dT%H:%M:%S)] relaunch OK." >> "$WLOG"
    fails=0
  else
    fails=$((fails + 1))
    echo "[watchdog5 $(date -u +%Y-%m-%dT%H:%M:%S)] relaunch FAILED (consecutive=$fails)." >> "$WLOG"
    if [[ "$fails" -ge 3 ]]; then
      echo "[watchdog5 $(date -u +%Y-%m-%dT%H:%M:%S)] 3 rapid failures; writing stop file and giving up." >> "$WLOG"
      touch "$STOP"
      exit 1
    fi
    sleep 60
  fi
  sleep "$INTERVAL"
done
