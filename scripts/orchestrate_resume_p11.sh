#!/bin/bash
# Resume orchestrator: wait for already-running phase 10 (50bb) to hit its 10-min mark,
# then run phases 11-22 with 65bb and 70bb inserted between 60 and 75.

ORCH_LOG=runs/orchestrator_timeline.log

CFG="--batched --hidden-dim 2048 --num-layers 4 --device cuda \
--aggression-bonus-c 2.00 --entropy-coef 0.20 \
--stack-dist uniform --seats-dist uniform \
--num-updates 1000000 --checkpoint-every-sec 60"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$ORCH_LOG"; }

find_pid() {
  wmic process where "CommandLine like '%%$1%%' and CommandLine like '%%train.py%%' and not CommandLine like '%%wmic%%'" get ProcessId 2>&1 \
    | grep -oE "^\s*[0-9]+" | tr -d ' ' | head -1
}

latest() {
  ls -t checkpoints/"${1}"_*.pt 2>/dev/null | head -1
}

log "[orch-resume] waiting for phase 10 (50bb) to reach 15:28:01"
TARGET="15:28:01"
while [ "$(date '+%H:%M:%S')" \< "$TARGET" ]; do sleep 5; done

PID10=$(find_pid "aggr_seq_p10_50bb.pt")
log "[orch-resume] killing phase 10 PID=$PID10"
[ -n "$PID10" ] && taskkill //F //PID "$PID10" > /dev/null 2>&1
LAST=$(latest "aggr_seq_p10_50bb")
log "[aggr_seq_p10_50bb] END last=$LAST"

# New phase list with 65 and 70 inserted between 60 and 75
PHASES=(
  "aggr_seq_p11_55bb 2,3,4,5,6 55:55"
  "aggr_seq_p12_60bb 2,3,4,5,6 60:60"
  "aggr_seq_p13_65bb 2,3,4,5,6 65:65"
  "aggr_seq_p14_70bb 2,3,4,5,6 70:70"
  "aggr_seq_p15_75bb 2,3,4,5,6 75:75"
  "aggr_seq_p16_80bb 2,3,4,5,6 80:80"
  "aggr_seq_p17_85bb 2,3,4,5,6 85:85"
  "aggr_seq_p18_90bb 2,3,4,5,6 90:90"
  "aggr_seq_p19_95bb 2,3,4,5,6 95:95"
  "aggr_seq_p20_100bb 2,3,4,5,6 100:100"
  "aggr_seq_p21_105bb 2,3,4,5,6 105:105"
  "aggr_seq_p22_110bb 2,3,4,5,6 110:110"
)

for spec in "${PHASES[@]}"; do
  read -r name seats stack <<< "$spec"
  ckpt="checkpoints/${name}.pt"
  logf="runs/${name}.log"
  warm=""
  if [ -n "$LAST" ] && [ -f "$LAST" ]; then
    warm="--load-checkpoint $LAST"
  fi
  log "[$name] START seats=$seats stack=$stack warm=$LAST"
  .venv/Scripts/python -u scripts/train.py $CFG \
    --num-seats-range "$seats" \
    --stack-range "$stack" \
    $warm \
    --checkpoint "$ckpt" \
    > "$logf" 2>&1 &
  sleep 15
  PID=$(find_pid "$ckpt")
  log "[$name] PID=$PID"
  sleep 585
  if [ -n "$PID" ]; then
    taskkill //F //PID "$PID" > /dev/null 2>&1
  fi
  LAST=$(latest "$name")
  log "[$name] END last=$LAST"
done

log "ORCHESTRATOR COMPLETE."
