#!/usr/bin/env bash
set -e
cd /c/Users/themi/plodbbot
PREV=checkpoints/multi_20bb_e01_412.pt
for STACK in 10 15 25 30 35 40 45 50 55 60 65; do
  CKPT="checkpoints/curr_${STACK}bb.pt"
  LOG="runs/curr_${STACK}bb.log"
  echo "=== Stage ${STACK}bb (warm-start from ${PREV}) ===" | tee -a runs/curriculum_master.log
  date | tee -a runs/curriculum_master.log
  .venv/Scripts/python -u scripts/train.py --batched \
    --num-seats-range "2,3,4,5,6" \
    --stack-range "${STACK}:${STACK}" \
    --stack-dist uniform \
    --seats-dist uniform \
    --entropy-coef 0.1 \
    --train-seconds 300 \
    --num-updates 1000000 \
    --device cuda \
    --load-checkpoint "${PREV}" \
    --checkpoint "${CKPT}" \
    --checkpoint-every-sec 300 \
    > "${LOG}" 2>&1
  echo "Stage ${STACK}bb done -> ${CKPT}" | tee -a runs/curriculum_master.log
  PREV="${CKPT}"
done
echo "=== Curriculum complete ===" | tee -a runs/curriculum_master.log
date | tee -a runs/curriculum_master.log
