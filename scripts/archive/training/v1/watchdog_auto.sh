#!/bin/bash
# Stem-bumping watchdog (replaces watchdog_optimizedN.sh). Checks every
# INTERVAL seconds; if scripts/train.py is gone (crash, OOM, pod
# maintenance), relaunches via launch_auto.sh — which starts the NEXT
# stem (optimized<X+1>) warm-started from the newest loadable checkpoint
# of the highest existing stem. Anneal floors/baselines/block position
# ride in the checkpoint, so restarts resume the annealed schedule.
#
# STOP cleanly:  touch runs/watchdog_auto.stop
# Crash-loop guard: 3 relaunches each dying within ~30s -> writes stop
# file and exits. NOTE: a slow OOM many hours in is NOT "rapid", so it
# would keep restarting on that cadence (one new stem per death) --
# check runs/optimized<latest>.log periodically.
set -uo pipefail
cd /workspace/plodbnet

STOP=runs/watchdog_auto.stop
WLOG=runs/watchdog_auto.log
INTERVAL=60

SELF=$$
exec 9>runs/watchdog_auto.lock
if ! flock -n 9; then
  echo "[watchdog $(date -u +%Y-%m-%dT%H:%M:%S)] another watchdog holds the lock; exiting." >> "$WLOG"
  exit 0
fi

rm -f "$STOP"
echo "[watchdog $(date -u +%Y-%m-%dT%H:%M:%S)] started (pid $SELF)" >> "$WLOG"

fails=0
while true; do
  if [[ -f "$STOP" ]]; then
    echo "[watchdog $(date -u +%Y-%m-%dT%H:%M:%S)] stop file present; exiting." >> "$WLOG"
    exit 0
  fi
  if pgrep -f 'scripts/train\.py' >/dev/null; then
    fails=0; sleep "$INTERVAL"; continue
  fi
  echo "[watchdog $(date -u +%Y-%m-%dT%H:%M:%S)] training DOWN; relaunching next stem..." >> "$WLOG"
  bash launch_auto.sh >> "$WLOG" 2>&1
  sleep 30
  if pgrep -f 'scripts/train\.py' >/dev/null; then
    echo "[watchdog $(date -u +%Y-%m-%dT%H:%M:%S)] relaunch OK." >> "$WLOG"; fails=0
  else
    fails=$((fails + 1))
    echo "[watchdog $(date -u +%Y-%m-%dT%H:%M:%S)] relaunch FAILED (consecutive=$fails)." >> "$WLOG"
    if [[ "$fails" -ge 3 ]]; then
      echo "[watchdog $(date -u +%Y-%m-%dT%H:%M:%S)] 3 rapid failures; giving up." >> "$WLOG"
      touch "$STOP"; exit 1
    fi
    sleep 60
  fi
  sleep "$INTERVAL"
done
