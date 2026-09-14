#!/bin/bash
# ICG follow-ups (GPU2, after run_icg_seeds.sh): the weight curve is steep (w1.5 7.14 / 7.37, w2 10.04 on v7r),
# so add w1.25; and ICG on v7h (old captions) to compare with the v7h headline numbers (composed 7.53, 1b 7.23).
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-2}
until grep -q ICG_SEEDS_DONE logs/icg_seeds.log 2>/dev/null; do sleep 120; done
for spec in "v7r_icg_w1p25 workdir/v7r/model_latest.pt 1.25" "v7h_icg_w1p5 workdir/v7h/model_latest.pt 1.5" \
            "v7h_icg_w1p25 workdir/v7h/model_latest.pt 1.25"; do
  set -- $spec
  [ -f runs_out/$1_matched_eval/.done ] && continue
  CKPT=$2 EXTRA="--cfg $3 --guide_mode icg --chunk 500" bash baseline/eval_matched_r.sh $1 16 $GPU \
    && touch runs_out/$1_matched_eval/.done
done
echo ICG_MORE_DONE
