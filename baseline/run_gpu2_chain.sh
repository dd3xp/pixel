#!/bin/bash
# node03 GPU2 queue after the 09-12 disk-full outage.
# 1) 1b (GFT-form internaliser) confirmation: seed 0 gave 1-NFE FD 7.63 at w=1.5 (composed guidance: 7.53 at 2 NFE
#    + a stored snapshot; best CFG 12.49 at 2 NFE). INTERNALISER claim needs <= 7.8 over 3 seeds -> seeds 1, 2 at
#    w=1.5, plus w=1.25 on seed 0 to see which side of the optimum 1.5 sits.
# 2) then the SDXL generality pilot (run_sdxl_n3.sh; its wait on PROBE_GFT_DONE is already satisfied).
# Launch: NEED_MB=28000 setsid nohup bash supervise.sh gpu2_chain 2 bash baseline/run_gpu2_chain.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-2}
CK=workdir/probe_gft/model_latest.pt
for spec in "probe_gft_w1p5_seed1 0.666667 1" "probe_gft_w1p5_seed2 0.666667 2" "probe_gft_w1p25 0.8 0"; do
  set -- $spec
  [ -f runs_out/$1_matched_eval/.done ] && continue
  echo "[$(date +%m%d-%H:%M)] 1b $1 beta=$2 seed=$3"
  CKPT=$CK EXTRA="--cfg 1 --gft_beta $2 --seed $3" bash baseline/eval_matched_r.sh $1 16 $GPU \
    && touch runs_out/$1_matched_eval/.done || echo "GFT_EVAL_FAIL $1"
done
echo GFT_SEEDS_DONE
bash baseline/run_sdxl_n3.sh
