#!/bin/bash
# 16 px push, round 2. 1b at 20k steps (7.65) was worse than at 10k (6.79) -> find the best training length, and try
# the snapshot-free reference (ref = the online network itself under the lower bucket; nothing stored even in training).
#   b    : same recipe as probe_gft_v7r, EMA kept at 5k / 7.5k / 10k
#   onl  : ref = online (label reference), EMA kept at 5k / 7.5k / 10k
# Eval: 16 px, w1.5 (beta 1/1.5), 1 NFE, seed 0; the best point then gets seeds 1-2 (by a later tick).
# Launch: NEED_MB=34000 setsid nohup bash supervise.sh gft_curve 2 bash baseline/run_gft_curve.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-2}
until grep -q IMPROVE16_DONE logs/improve16.log 2>/dev/null && grep -q STACK_SEEDS_DONE logs/stack_seeds.log 2>/dev/null; do sleep 120; done
for v in "b snapshot" "onl online"; do
  set -- $v
  OUT=workdir/probe_gft_v7r_$1
  if [ ! -f $OUT/.trained ]; then
    $P src/v6/train_gft_res.py --steps 10000 --out $OUT --ref $2 --init workdir/v7r/model_latest.pt \
       --snap workdir/v7r_snap10k/model_latest.pt --csv_suffix _recap --keep_steps 5000,7500 || { echo "GFT_CURVE_TRAIN_FAIL $1"; exit 1; }
    cp $OUT/model_latest.pt $OUT/model_step010000.pt; touch $OUT/.trained
  fi
  for s in 005000 007500 010000; do
    t=v7r_gft$1_s${s}_w1p5
    [ -f runs_out/${t}_matched_eval/.done ] && continue
    CKPT=$OUT/model_step$s.pt EXTRA="--cfg 1 --gft_beta 0.666667 --chunk 500" bash baseline/eval_matched_r.sh $t 16 $GPU \
      && touch runs_out/${t}_matched_eval/.done
  done
done
echo GFT_CURVE_DONE
