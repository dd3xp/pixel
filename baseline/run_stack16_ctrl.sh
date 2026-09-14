#!/bin/bash
# Attribution of the 1b+ICG gain (09-15 21:40): base v7r + ICG1.5 = 7.14 (s0); 1b model with internal guidance OFF
# (beta=1) + ICG1.5 = 6.63; 1b at w1.25 + ICG1.5 = 6.17/6.23/6.34. Is the 7.14 -> 6.63 step just the extra 10k
# fine-tuning steps? Same test on the plain-GFT (empty-caption reference) model, trained with the identical recipe:
#   gftnull beta=1 + ICG1.5  (fine-tune without our reference)
#   gftnull beta=1/1.25 + ICG1.5 (its own internalised CFG + ICG)
# and the bare 1b / gftnull at beta=1 with no guidance at all (cfg 1) for reference.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-2}
until grep -q STACK16_DONE logs/stack16.log 2>/dev/null; do sleep 120; done
N=workdir/probe_gft_v7r_null/model_latest.pt
B=workdir/probe_gft_v7r/model_step010000_kept.pt
ev() {
  local tag=$1 ck=$2; shift 2
  [ -f runs_out/${tag}_matched_eval/.done ] && return 0
  for try in 1 2 3; do
    CKPT=$ck EXTRA="$* --chunk 500" bash baseline/eval_matched_r.sh $tag 16 $GPU && { touch runs_out/${tag}_matched_eval/.done; return 0; }
    sleep 300
  done
}
ev v7r_gftnull_w1p0_icg1p5 $N --gft_beta 1.0 --cfg 1.5 --guide_mode icg
ev v7r_gftnull_w1p25_icg1p5 $N --gft_beta 0.8 --cfg 1.5 --guide_mode icg
ev v7r_gft_w1p0_bare $B --gft_beta 1.0 --cfg 1
ev v7r_gftnull_w1p0_bare $N --gft_beta 1.0 --cfg 1
echo STACK16_CTRL_DONE
