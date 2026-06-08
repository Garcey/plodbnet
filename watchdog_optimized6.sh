#!/bin/bash
# Watchdog for the phase-6 training run (launch_optimized6.sh). Checks every
# INTERVAL seconds; if scripts/train.py is gone (crash, OOM, pod maintenance),
# relaunches from the highest-numbered checkpoint. Phase-6-specific
# stop/lock/log files so it never collides with other phases.
#
# STOP cleanly:  touch runs/watchdog6.stop
# Crash-loop guard: 3 relaunches each dying within ~30s -> writes stop file and
# exits. NOTE: a slow OOM many hours in is NOT "rapid", so it would keep
# restarting on that cadence -- check runs/optimized6.log periodically.
set -uo pipefail
cd /workspace/plodbnet

STOP=runs/watchdog6.stop
WLOG=runs/watchdog6.log
INTERVAL=60

SELF=$$
exec 9>runs/watchdog6.lock
if ! flock -n 9; then
  echo "[watchdog6 $(date -u +%Y-%m-%dT%H:%M:%S)] another watchdog holds the lock; exiting." >> "$WLOG"
  exit 0
fi

rm -f "$STOP"
echo "[watchdog6 $(date -u +%Y-%m-%dT%H:%M:%S)] started (pid $SELF)" >> "$WLOG"

fails=0
while true; do
  if [[ -f "$STOP" ]]; then
    echo "[watchdog6 $(date -u +%Y-%m-%dT%H:%M:%S)] stop file present; exiting." >> "$WLOG"
    exit 0
  fi
  if pgrep -f 'scripts/train\.py' >/dev/null; then
    fails=0; sleep "$INTERVAL"; continue
  fi
  echo "[watchdog6 $(date -u +%Y-%m-%dT%H:%M:%S)] training DOWN; relaunching..." >> "$WLOG"
  bash launch_optimized6.sh >> "$WLOG" 2>&1
  sleep 30
  if pgrep -f 'scripts/train\.py' >/dev/null; then
    echo "[watchdog6 $(date -u +%Y-%m-%dT%H:%M:%S)] relaunch OK." >> "$WLOG"; fails=0
  else
    fails=$((fails + 1))
    echo "[watchdog6 $(date -u +%Y-%m-%dT%H:%M:%S)] relaunch FAILED (consecutive=$fails)." >> "$WLOG"
    if [[ "$fails" -ge 3 ]]; then
      echo "[watchdog6 $(date -u +%Y-%m-%dT%H:%M:%S)] 3 rapid failures; giving up." >> "$WLOG"
      touch "$STOP"; exit 1
    fi
    sleep 60
  fi
  sleep "$INTERVAL"
done
