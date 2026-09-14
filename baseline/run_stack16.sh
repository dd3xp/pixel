#!/bin/bash
# 1b (internalised cross-resolution guidance) + ICG (text-axis reference), 2 NFE, v7r @16 px.
# seed 0: 1b at w1.25 + ICG 1.5 = 6.17, vs 1b alone (w1.5, 1 NFE) 6.79 and ICG alone (2 NFE) 7.14 / 3-seed 7.46.
# Confirm with seeds 1-2, map the neighbourhood, and the control "1b model with its internal guidance OFF (beta=1)
# + ICG 1.5" -- separates the internalised resolution guidance from the fine-tune itself.
# Launch: NEED_MB=9000 setsid nohup bash supervise.sh stack16 2 bash baseline/run_stack16.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-2}
CK=workdir/probe_gft_v7r/model_step010000_kept.pt
ev() {
  local tag=$1; shift
  [ -f runs_out/${tag}_matched_eval/.done ] && return 0
  for try in 1 2 3; do
    CKPT=$CK EXTRA="$* --chunk 500" bash baseline/eval_matched_r.sh $tag 16 $GPU && { touch runs_out/${tag}_matched_eval/.done; return 0; }
    echo "STACK16_RETRY $tag $try"; sleep 300
  done
}
ev v7r_gft_w1p25_icg1p5_seed1 --gft_beta 0.8 --cfg 1.5 --guide_mode icg --seed 1
ev v7r_gft_w1p25_icg1p5_seed2 --gft_beta 0.8 --cfg 1.5 --guide_mode icg --seed 2
ev v7r_gft_w1p0_icg1p5 --gft_beta 1.0 --cfg 1.5 --guide_mode icg
ev v7r_gft_w1p25_icg1p25 --gft_beta 0.8 --cfg 1.25 --guide_mode icg
ev v7r_gft_w1p25_icg1p75 --gft_beta 0.8 --cfg 1.75 --guide_mode icg
ev v7r_gft_w1p15_icg1p5 --gft_beta 0.869565 --cfg 1.5 --guide_mode icg
ev v7r_gft_w1p35_icg1p5 --gft_beta 0.740741 --cfg 1.5 --guide_mode icg
echo STACK16_DONE
