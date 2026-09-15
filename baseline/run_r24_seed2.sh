#!/bin/bash
# 24 px 1b (v7r) third seed (seeds 0/1: 43.16 / 43.54).
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-2}
t=v7r_r24_gft_w1p5_seed2
[ -f runs_out/${t}_matched_eval/.done ] || { CKPT=workdir/probe_gft_v7r/model_step010000_kept.pt EXTRA="--cfg 1 --gft_beta 0.666667 --seed 2 --chunk 250" \
  bash baseline/eval_matched_r.sh $t 24 $GPU && touch runs_out/${t}_matched_eval/.done; }
echo R24_SEED2_DONE
