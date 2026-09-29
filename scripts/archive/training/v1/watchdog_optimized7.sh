#!/bin/bash
# Watchdog for the phase-7 training run (launch_optimized7.sh). Checks every
# INTERVAL seconds; if scripts/train.py is gone (crash, OOM, pod maintenance),
# relaunches from the highest-numbered checkpoint. Phase-7-specific
# stop/lock/log files so it never collides with other phases.
#
# STOP cleanly:  touch runs/watchdog7.stop
# Crash-loop guard: 3 relaunches each dying within ~30s -> writes stop file and
# exits. NOTE: a slow OOM many hours in is NOT "rapid", so it would keep
# restarting on that cadence -- check runs/optimized7.log periodically.
set -uo pipefail
cd /workspace/plodbnet

STOP=runs/watchdog7.stop
WLOG=runs/watchdog7.log
INTERVAL=60

SELF=$$
exec 9>runs/watchdog7.lock
if ! flock -n 9; then
  echo "[watchdog7 $(date -u +%Y-%m-%dT%H:%M:%S)] another watchdog holds the lock; exiting." >> "$WLOG"
  exit 0
fi

rm -f "$STOP"
echo "[watchdog7 $(date -u +%Y-%m-%dT%H:%M:%S)] started (pid $SELF)" >> "$WLOG"

fails=0
while true; do
  if [[ -f "$STOP" ]]; then
    echo "[watchdog7 $(date -u +%Y-%m-%dT%H:%M:%S)] stop file present; exiting." >> "$WLOG"
    exit 0
  fi
  if pgrep -f 'scripts/train\.py' >/dev/null; then
    fails=0; sleep "$INTERVAL"; continue
  fi
  echo "[watchdog7 $(date -u +%Y-%m-%dT%H:%M:%S)] training DOWN; relaunching..." >> "$WLOG"
  bash launch_optimized7.sh >> "$WLOG" 2>&1
  sleep 30
  if pgrep -f 'scripts/train\.py' >/dev/null; then
    echo "[watchdog7 $(date -u +%Y-%m-%dT%H:%M:%S)] relaunch OK." >> "$WLOG"; fails=0
  else
    fails=$((fails + 1))
    echo "[watchdog7 $(date -u +%Y-%m-%dT%H:%M:%S)] relaunch FAILED (consecutive=$fails)." >> "$WLOG"
    if [[ "$fails" -ge 3 ]]; then
      echo "[watchdog7 $(date -u +%Y-%m-%dT%H:%M:%S)] 3 rapid failures; giving up." >> "$WLOG"
      touch "$STOP"; exit 1
    fi
    sleep 60
  fi
  sleep "$INTERVAL"
done
