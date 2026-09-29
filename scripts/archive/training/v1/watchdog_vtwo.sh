#!/bin/bash
# Stem-bumping watchdog for the v2 vTwo<N> family. Checks every
# INTERVAL seconds; if scripts/train.py is gone (crash, OOM, pod
# maintenance), relaunches via launch_vtwo.sh — which cold-starts
# vTwo1 when no vTwo checkpoints exist, else starts the NEXT stem
# warm-started from the newest loadable vTwo checkpoint. Anneal
# floors/baselines/block position ride in the checkpoint.
#
# STOP cleanly:  touch runs/watchdog_vtwo.stop
#
# NOTE: pgrep matches ANY scripts/train.py — the v1 optimized<N>
# watchdog uses the same probe. Run ONE watchdog family per pod
# (stop the other first: touch runs/watchdog_auto.stop).
#
# Crash-loop guard: 3 relaunches each dying within ~30s -> writes stop
# file and exits. A slow OOM many hours in is NOT "rapid", so it would
# keep restarting on that cadence (one new stem per death) — check
# runs/vTwo<latest>.log periodically.
set -uo pipefail
cd /workspace/plodbnet

STOP=runs/watchdog_vtwo.stop
WLOG=runs/watchdog_vtwo.log
INTERVAL=60

SELF=$$
exec 9>runs/watchdog_vtwo.lock
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
  bash launch_vtwo.sh >> "$WLOG" 2>&1
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
