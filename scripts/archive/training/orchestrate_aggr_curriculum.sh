#!/bin/bash
# Sequential aggression-curriculum orchestrator.
# Phase 1: continue currently-running hu20 c=2.0 e=0.2 for 5 more min.
# Phases 2-20: 2-6 seat uniform at progressive fixed stack depths, 10 min each, warm-starting from prior.

ORCH_LOG=runs/orchestrator_timeline.log
echo "[orch] start $(date '+%Y-%m-%d %H:%M:%S')" > "$ORCH_LOG"

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

# Phase 1 — current run already going. Wait 5 min, then kill.
log "[phase01] hu20 c200 e20 already running. Sleeping 300s."
sleep 300
PID1=$(find_pid "run_aggr_hu20_c200_e20.pt")
log "[phase01] killing PID=$PID1"
[ -n "$PID1" ] && taskkill //F //PID "$PID1" > /dev/null 2>&1
LAST=$(latest "run_aggr_hu20_c200_e20")
log "[phase01] DONE last=$LAST"

# Phases 2..20
PHASES=(
  "aggr_seq_p02_20bb 2,3,4,5,6 20:20"
  "aggr_seq_p03_10bb 2,3,4,5,6 10:10"
  "aggr_seq_p04_15bb 2,3,4,5,6 15:15"
  "aggr_seq_p05_25bb 2,3,4,5,6 25:25"
  "aggr_seq_p06_30bb 2,3,4,5,6 30:30"
  "aggr_seq_p07_35bb 2,3,4,5,6 35:35"
  "aggr_seq_p08_40bb 2,3,4,5,6 40:40"
  "aggr_seq_p09_45bb 2,3,4,5,6 45:45"
  "aggr_seq_p10_50bb 2,3,4,5,6 50:50"
  "aggr_seq_p11_55bb 2,3,4,5,6 55:55"
  "aggr_seq_p12_60bb 2,3,4,5,6 60:60"
  "aggr_seq_p13_75bb 2,3,4,5,6 75:75"
  "aggr_seq_p14_80bb 2,3,4,5,6 80:80"
  "aggr_seq_p15_85bb 2,3,4,5,6 85:85"
  "aggr_seq_p16_90bb 2,3,4,5,6 90:90"
  "aggr_seq_p17_95bb 2,3,4,5,6 95:95"
  "aggr_seq_p18_100bb 2,3,4,5,6 100:100"
  "aggr_seq_p19_105bb 2,3,4,5,6 105:105"
  "aggr_seq_p20_110bb 2,3,4,5,6 110:110"
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
