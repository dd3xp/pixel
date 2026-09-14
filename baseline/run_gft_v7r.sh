#!/bin/bash
# 1b on the RECAPTIONED model: replicate "1-NFE internaliser beats composed guidance" (v7h: 7.23 +- 0.34 vs 7.53 +- 0.19).
# Same recipe as probe_gft (10k steps, beta ~ U(0.4,1), 25% plain, reference = 10k snapshot under the lower bucket),
# init = v7r, snapshot = v7r_snap10k, recaptioned CSVs. Evals use the recaptioned prompts (tags start with v7r).
# 16 px: w1.5 seeds 0-2; 20 and 24 px: w1.5 seed 0 (lower bucket 16 for both, as in training).
# Compare against v7r_evals' composed rows (v7r_stk12_w1p5*, v7r_r20_stk16_w1p5, v7r_r24_stk16_w1p5).
# Launch: NEED_MB=34000 setsid nohup bash supervise.sh gft_v7r 7 bash baseline/run_gft_v7r.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-7}
until grep -q V7R_DONE logs/v7r.log 2>/dev/null; do sleep 300; done
OUT=workdir/probe_gft_v7r
if [ ! -f $OUT/.trained ]; then
  echo "[$(date +%m%d-%H:%M)] STAGE train 1b on v7r"
  $P src/v6/train_gft_res.py --steps 10000 --out $OUT --ref snapshot --init workdir/v7r/model_latest.pt \
     --snap workdir/v7r_snap10k/model_latest.pt --csv_suffix _recap || { echo GFT_V7R_TRAIN_FAIL; exit 1; }
  touch $OUT/.trained
fi
for spec in "v7r_gft_w1p5 16 0" "v7r_gft_w1p5_seed1 16 1" "v7r_gft_w1p5_seed2 16 2" "v7r_r20_gft_w1p5 20 0" "v7r_r24_gft_w1p5 24 0"; do
  set -- $spec
  [ -f runs_out/$1_matched_eval/.done ] && continue
  echo "[$(date +%m%d-%H:%M)] eval $1 @$2px seed $3"
  CKPT=$OUT/model_latest.pt EXTRA="--cfg 1 --gft_beta 0.666667 --seed $3" bash baseline/eval_matched_r.sh $1 $2 $GPU \
    && touch runs_out/$1_matched_eval/.done || echo "GFT_V7R_EVAL_FAIL $1"
done
echo GFT_V7R_DONE
