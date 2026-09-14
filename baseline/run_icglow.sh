#!/bin/bash
# Combined weak reference = ICG's Gaussian text condition + our lower resolution bucket, in ONE weak forward (2 NFE).
# Motivation (09-15): ICG alone 7.46 +- 0.37 beats our label (8.86) / composed (8.50) references on v7r @16 px, while a
# RANDOM bucket (ICG-label) is harmful (12.23). If the two degradations compose like label + snapshot did, the
# combination should beat both; also with the 10k snapshot as a third axis.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-2}
until grep -q ICG_MORE_DONE logs/icg_more.log 2>/dev/null; do sleep 120; done
CK=workdir/v7r/model_latest.pt
SNAP=workdir/v7r_snap10k/model_latest.pt
RUNS=("v7r_icglow12_w1p5 --cfg 1.5 --guide_mode icglow:12" "v7r_icglow12_w1p25 --cfg 1.25 --guide_mode icglow:12"
      "v7r_icglow12_w2 --cfg 2 --guide_mode icglow:12"
      "v7r_icglowstk12_w1p5 --cfg 1.5 --guide_mode icglow:12 --guide_ckpt $SNAP"
      "v7r_icgstk_w1p5 --cfg 1.5 --guide_mode icg --guide_ckpt $SNAP")
for r in "${RUNS[@]}"; do
  tag=${r%% *}; args=${r#* }
  [ -f runs_out/${tag}_matched_eval/.done ] && continue
  CKPT=$CK EXTRA="$args --chunk 500" bash baseline/eval_matched_r.sh $tag 16 $GPU && touch runs_out/${tag}_matched_eval/.done
done
echo ICGLOW_DONE
