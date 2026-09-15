#!/bin/bash
# 1b + ICG at 20 and 24 px (v7r, 2 NFE). 16 px: 1b w1.25 + ICG1.5 = 6.25 +- 0.09 vs ICG 7.46.
# 20 px: 1b alone 24.06, ICG 28.30 (w1.5) / 29.73 (w2), best CFG 34.21 (w2).
# 24 px: 1b alone 43.35, ICG 48.82 (w1.5) / 46.21 (w2), best CFG 65.87 (w3). Larger sizes preferred higher CFG,
# so ICG 1.5 and 2 are both tried.
# Launch: NEED_MB=9000 setsid nohup bash supervise.sh stack2024 2 bash baseline/run_stack2024.sh >/dev/null 2>&1 &
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
for R in 20 24; do
  ev v7r_r${R}_gft_w1p25_icg1p5 $R --gft_beta 0.8 --cfg 1.5 --guide_mode icg
  ev v7r_r${R}_gft_w1p25_icg2 $R --gft_beta 0.8 --cfg 2 --guide_mode icg
  ev v7r_r${R}_gft_w1p5_icg1p5 $R --gft_beta 0.666667 --cfg 1.5 --guide_mode icg
done
echo STACK2024_DONE
