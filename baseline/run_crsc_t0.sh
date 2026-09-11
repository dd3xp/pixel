#!/bin/bash
# CRSC gate T0 on GPU2, chained after probe_mdm2 so the card does not sit idle between cron ticks.
# Waits for probe_mdm2's own eval to finish (PROBE_MDM2_DONE, or its train failure), then measures the
# optimal extrapolation weight w*(t) of the three references (label / snapshot / composed) on v7h at
# 16 (ref 12), 20 (ref 16) and 24 (ref 16) px. Decision rule inside crsc_t0.py: GO if w* >= 1.3 on >= 30% of
# timesteps for any reference; STOP (-> T1) if all < 1.1.
# Launch: tmux new -d -s crsc_t0 "bash baseline/run_crsc_t0.sh 2>&1 | tee -a logs/crsc_t0.log"
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${GPU:-2}
until grep -q "PROBE_MDM2_DONE\|PROBE_MDM2_TRAIN_FAIL" logs/probe_mdm2.log; do sleep 120; done
echo "[$(date +%m%d-%H:%M)] probe_mdm2 finished, starting CRSC T0 on GPU $CUDA_VISIBLE_DEVICES"
for pair in "16 12" "20 16" "24 16"; do
  set -- $pair
  out=runs_out/crsc_t0_s$1_l$2.json
  [ -f $out ] && continue
  echo "[$(date +%m%d-%H:%M)] T0 size=$1 lower=$2"
  $P src/v6/crsc_t0.py --size $1 --lower $2 --n 3000 --out $out || { echo "CRSC_T0_FAIL size=$1"; touch logs/crsc_t0.FAILING; exit 1; }
done
echo CRSC_T0_ALL_DONE
