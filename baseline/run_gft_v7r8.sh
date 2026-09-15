#!/bin/bash
# 1b on v7r8 (8 px rung): 12 px finally gets our method. Chained after run_v7r8.sh (V7R8_DONE).
# 12 px bars (v7r): best CFG 8.82 (3 seeds), autoguidance 8.36. Also 16 px regression (1b alone / 1b + ICG, seed 0)
# against 1b-v7r 7.20~7.23 and 1b+ICG 6.25~6.31.
# Launch: NEED_MB=34000 setsid nohup bash supervise.sh gft_v7r8 2 bash baseline/run_gft_v7r8.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-2}
until grep -q V7R8_DONE logs/v7r8.log 2>/dev/null; do sleep 180; done
SNAP=workdir/v7r_snap10k/model_grown8.pt
[ -f $SNAP ] || $P src/v6/train_gft_res8.py --make_grown_snap workdir/v7r_snap10k/model_latest.pt $SNAP
OUT=workdir/probe_gft_v7r8
if [ ! -f $OUT/.trained ]; then
  $P src/v6/train_gft_res8.py --init workdir/v7r8/model_latest.pt --snap $SNAP --steps 10000 --out $OUT \
     --ref snapshot --csv_suffix _recap || { echo GFT_V7R8_TRAIN_FAIL; exit 1; }
  touch $OUT/.trained
fi
B=12,16,20,24,32,48,64,8
ev() {
  local tag=$1 R=$2; shift 2
  [ -f runs_out/${tag}_matched_eval/.done ] && return 0
  for try in 1 2 3; do
    CKPT=$OUT/model_latest.pt EXTRA="--buckets $B $* --chunk 500" bash baseline/eval_matched_r.sh $tag $R $GPU \
      && { touch runs_out/${tag}_matched_eval/.done; return 0; }
    sleep 300
  done
}
for s in 0 1 2; do
  t=v7r8_r12_gft_w1p5; [ $s != 0 ] && t=${t}_seed$s
  ev $t 12 --cfg 1 --gft_beta 0.666667 --seed $s
done
ev v7r8_r12_gft_w1p25_icg1p5 12 --gft_beta 0.8 --cfg 1.5 --guide_mode icg
ev v7r8_gft_w1p5 16 --cfg 1 --gft_beta 0.666667
ev v7r8_gft_w1p25_icg1p5 16 --gft_beta 0.8 --cfg 1.5 --guide_mode icg
echo GFT_V7R8_DONE
