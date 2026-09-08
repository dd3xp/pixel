#!/bin/bash
# paper_outline gap #9: second metric family (Inception clean-FID + KID) on ALL saved matched-protocol samples at every R,
# so FD-DINOv2 vs FID rank agreement can be reported over every row.  Re-scoring only, no sampling.
# Launch: NEED_MB=6000 setsid nohup bash supervise.sh diag_metric2 2 bash baseline/diag_metric2.sh 2 >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-2}; export CUDA_VISIBLE_DEVICES=$GPU
for R in 16 20 24 12 32; do
  echo "[$(date +%m%d-%H:%M)] INCEP R=$R"
  $P src/v6/fid_kid_fair.py --size $R --floor --out runs_out/incep_scores.json --gen $(ls -d runs_out/*_matched_eval/s$R)
done
echo DIAG_METRIC2_DONE
