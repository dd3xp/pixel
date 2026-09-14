#!/bin/bash
# ICG (Gaussian random condition) came out at 7.14 @16px seed0 on v7r, below our label (8.81) and composed (8.22)
# references; confirm with seeds 1-2 before interpreting.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-2}
for s in 1 2; do
  t=v7r_icg_w1p5_seed$s
  [ -f runs_out/${t}_matched_eval/.done ] && continue
  CKPT=workdir/v7r/model_latest.pt EXTRA="--cfg 1.5 --guide_mode icg --seed $s --chunk 500" bash baseline/eval_matched_r.sh $t 16 $GPU \
    && touch runs_out/${t}_matched_eval/.done
done
echo ICG_SEEDS_DONE
