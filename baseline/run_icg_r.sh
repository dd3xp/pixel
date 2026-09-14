#!/bin/bash
# ICG at 20 and 24 px on v7r (the user's range is 12-24). At 16 px ICG (7.46) beats our zero-training references;
# at 20 px ours: 1b 24.06 (3 seeds), composed 25.61, label 27.56, best CFG 34.21. Does ICG hold up at larger sizes?
# 09-15: first attempt on GPU7 OOMed (4/4 runs) next to other jobs -> GPU2, chunk 250, 3 tries each.
# Launch: NEED_MB=10000 setsid nohup bash supervise.sh icg_r 2 bash baseline/run_icg_r.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-2}
for spec in "v7r_r20_icg_w1p5 20 1.5" "v7r_r20_icg_w2 20 2" "v7r_r24_icg_w1p5 24 1.5" "v7r_r24_icg_w2 24 2"; do
  set -- $spec
  [ -f runs_out/$1_matched_eval/.done ] && continue
  CKPT=workdir/v7r/model_latest.pt EXTRA="--cfg $3 --guide_mode icg --chunk 500" bash baseline/eval_matched_r.sh $1 $2 $GPU \
    && touch runs_out/$1_matched_eval/.done
done
echo ICG_R_DONE
