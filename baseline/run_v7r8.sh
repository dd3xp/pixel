#!/bin/bash
# v7r8: v7r + 8 px rung (train_v7r8.py), 20k-step fine-tune, then 12 px evaluation with 8 px as the lower bucket.
# 12 px so far (v7r, no lower bucket): best CFG 8.82 (w2, 3 seeds), autoguidance 8.36.
# Also a 16 px regression check (CFG 1.5, bucket:12 w2) so the fine-tune does not cost the main battlefield.
# Launch: NEED_MB=30000 setsid nohup bash supervise.sh v7r8 2 bash baseline/run_v7r8.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-2}
until grep -q GFT_CURVE_DONE logs/gft_curve.log 2>/dev/null && grep -q STACKB_DONE logs/stackb.log 2>/dev/null; do sleep 120; done
OUT=workdir/v7r8
if [ ! -f $OUT/.trained ]; then
  $P src/v6/train_v7r8.py --init_v6 workdir/v7r/model_latest.pt --steps 20000 --out $OUT \
     --exclude runs_out/holdout_exclude.txt --csv_suffix _recap --snap_every 5000 || { echo V7R8_TRAIN_FAIL; exit 1; }
  touch $OUT/.trained
fi
B=12,16,20,24,32,48,64,8
ev() {  # <tag> <R> <args>
  local tag=$1 R=$2; shift 2
  [ -f runs_out/${tag}_matched_eval/.done ] && return 0
  for try in 1 2 3; do
    CKPT=$OUT/model_latest.pt EXTRA="--buckets $B $* --chunk 500" bash baseline/eval_matched_r.sh $tag $R $GPU \
      && { touch runs_out/${tag}_matched_eval/.done; return 0; }
    sleep 300
  done
}
for w in 1.5 2 2.5; do ev v7r8_r12_cfg$(echo $w | tr . p) 12 --cfg $w; done
for w in 1.5 2 2.5; do ev v7r8_r12_bk8_w$(echo $w | tr . p) 12 --cfg $w --guide_mode bucket:8; done
ev v7r8_cfg1p5 16 --cfg 1.5
ev v7r8_bk12_w2 16 --cfg 2 --guide_mode bucket:12
echo V7R8_DONE
