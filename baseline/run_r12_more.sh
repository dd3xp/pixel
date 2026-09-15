#!/bin/bash
# 12 px label guidance on v7r8 keeps improving with w (bucket:8 w1.5 9.77, w2 8.15, w2.5 7.50; best CFG 8.40) -> w3, w3.5.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-2}
for w in 3 3.5; do
  t=v7r8_r12_bk8_w$(echo $w | tr . p)
  [ -f runs_out/${t}_matched_eval/.done ] && continue
  CKPT=workdir/v7r8/model_latest.pt EXTRA="--buckets 12,16,20,24,32,48,64,8 --cfg $w --guide_mode bucket:8 --chunk 500" \
    bash baseline/eval_matched_r.sh $t 12 $GPU && touch runs_out/${t}_matched_eval/.done
done
echo R12_MORE_DONE
