#!/bin/bash
# Controls for 1b (does its gain come from the GFT training form or from OUR lower-resolution reference?).
# Identical recipe to probe_gft_v7r (init v7r, 10k steps, beta ~ U(0.4,1), 25% plain, recaptioned CSVs), only the
# reference changes, taken from the online network at the target bucket:
#   null = empty caption  -> plain GFT (Chen et al., ICML 2025), i.e. internalised CFG
#   icg  = Gaussian random condition -> internalised ICG (ICG alone: 7.46 +- 0.37 at 2 NFE)
# 1b-v7r (ours): 7.23 +- 0.45 at 1 NFE. Eval: 16 px, w1.5 (beta 1/1.5), seeds 0-2, 1 NFE.
# Launch: NEED_MB=34000 setsid nohup bash supervise.sh gft_ctrl 7 bash baseline/run_gft_ctrl.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-7}
for ref in null icg; do
  OUT=workdir/probe_gft_v7r_$ref
  if [ ! -f $OUT/.trained ]; then
    echo "[$(date +%m%d-%H:%M)] STAGE train GFT control ref=$ref"
    $P src/v6/train_gft_res.py --steps 10000 --out $OUT --ref $ref --init workdir/v7r/model_latest.pt \
       --csv_suffix _recap || { echo "GFT_CTRL_TRAIN_FAIL $ref"; exit 1; }
    touch $OUT/.trained
  fi
  for s in 0 1 2; do
    t=v7r_gft${ref}_w1p5; [ $s != 0 ] && t=${t}_seed$s
    [ -f runs_out/${t}_matched_eval/.done ] && continue
    CKPT=$OUT/model_latest.pt EXTRA="--cfg 1 --gft_beta 0.666667 --seed $s" bash baseline/eval_matched_r.sh $t 16 $GPU \
      && touch runs_out/${t}_matched_eval/.done
  done
done
echo GFT_CTRL_DONE
