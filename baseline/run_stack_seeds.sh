#!/bin/bash
# Follow-up of run_stack2024.sh: 24 px 1b w1.25 + ICG1.5 = 41.50 (1b alone 43.35, ICG 46.21) -> seeds 1-2;
# 20 px stacking hurt at ICG 1.5/2 (27.39/29.80 vs 1b alone 24.06) -> try a weaker ICG 1.25.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-2}
CK=workdir/probe_gft_v7r/model_step010000_kept.pt
ev() {
  local tag=$1 R=$2; shift 2
  [ -f runs_out/${tag}_matched_eval/.done ] && return 0
  for try in 1 2 3; do
    CKPT=$CK EXTRA="$* --chunk 250" bash baseline/eval_matched_r.sh $tag $R $GPU && { touch runs_out/${tag}_matched_eval/.done; return 0; }
    sleep 300
  done
}
ev v7r_r24_gft_w1p25_icg1p5_seed1 24 --gft_beta 0.8 --cfg 1.5 --guide_mode icg --seed 1
ev v7r_r24_gft_w1p25_icg1p5_seed2 24 --gft_beta 0.8 --cfg 1.5 --guide_mode icg --seed 2
ev v7r_r20_gft_w1p25_icg1p25 20 --gft_beta 0.8 --cfg 1.25 --guide_mode icg
ev v7r_r20_gft_w1p5_icg1p25 20 --gft_beta 0.666667 --cfg 1.25 --guide_mode icg
echo STACK_SEEDS_DONE
