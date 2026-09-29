#!/bin/bash
# Watchdog for the phase-4 training run. Launches training immediately (via
# launch_optimized4.sh), then checks every INTERVAL seconds; if scripts/train.py
# is gone (crash, OOM, pod maintenance), it relaunches from the highest-numbered
# checkpoint. Decoupled from training: each is its own setsid session, so
# killing one doesn't kill the other.
#
# STOP cleanly:  touch runs/watchdog.stop   (then optionally kill train.py)
#   The watchdog checks this file every loop and at startup-relaunch time, and
#   exits without relaunching. Starting a NEW watchdog clears a stale stop file.
#
# Crash-loop guard: 3 relaunches that each die within ~30s -> watchdog writes
# the stop file and exits, so a genuine startup bug can't thrash forever.
#
# LIMITATION: a slow failure (e.g. an OOM ~16h into a run) is NOT a "rapid"
# failure, so the watchdog WILL keep restarting it on that cadence. The OOM
# root cause was fixed (opponent-cache reverted) + expandable_segments is set,
# so this is insurance against transient/maintenance events, not a substitute
# for watching health. Check runs/optimized4.log periodically.
set -uo pipefail
cd /workspace/plodbnet

STOP=runs/watchdog.stop
WLOG=runs/watchdog.log
INTERVAL=60

# Single-instance guard via flock (NOT pgrep — pgrep also matches the ssh
# command string that launches this script, which made every instance think
# another was already running and exit). flock is held for the life of the
# process and auto-releases on death, so exactly one watchdog can run.
SELF=$$
exec 9>runs/watchdog.lock
if ! flock -n 9; then
  echo "[watchdog $(date -u +%Y-%m-%dT%H:%M:%S)] another watchdog holds the lock; exiting." >> "$WLOG"
  exit 0
fi

rm -f "$STOP"   # fresh start clears any stale stop request
echo "[watchdog $(date -u +%Y-%m-%dT%H:%M:%S)] started (pid $SELF)" >> "$WLOG"

fails=0
while true; do
  if [[ -f "$STOP" ]]; then
    echo "[watchdog $(date -u +%Y-%m-%dT%H:%M:%S)] stop file present; exiting (training left as-is)." >> "$WLOG"
    exit 0
  fi

  if pgrep -f 'scripts/train\.py' >/dev/null; then
    fails=0
    sleep "$INTERVAL"
    continue
  fi

  # Training is not running — relaunch.
  echo "[watchdog $(date -u +%Y-%m-%dT%H:%M:%S)] training DOWN; relaunching..." >> "$WLOG"
  bash launch_optimized4.sh >> "$WLOG" 2>&1
  sleep 30

  if pgrep -f 'scripts/train\.py' >/dev/null; then
    echo "[watchdog $(date -u +%Y-%m-%dT%H:%M:%S)] relaunch OK." >> "$WLOG"
    fails=0
  else
    fails=$((fails + 1))
    echo "[watchdog $(date -u +%Y-%m-%dT%H:%M:%S)] relaunch FAILED (consecutive=$fails)." >> "$WLOG"
    if [[ "$fails" -ge 3 ]]; then
      echo "[watchdog $(date -u +%Y-%m-%dT%H:%M:%S)] 3 rapid failures; writing stop file and giving up." >> "$WLOG"
      touch "$STOP"
      exit 1
    fi
    sleep 60
  fi
  sleep "$INTERVAL"
done
