#!/bin/bash
# 1b + CFG stacking (2 NFE, the same cost as the composed rule). 1b alone at 1 NFE is 7.23 +- 0.34 over 3 seeds on
# v7h, below composed guidance 7.53 +- 0.19 at 2 NFE. If adding CFG on top of the internalised model lowers FD further
# at 2 NFE, that is a same-cost win over the strongest rule. BetaWrap feeds the fixed beta to both CFG branches.
# Launch (waits until GPU7 has 20 GB free, i.e. after v7r's training):
#   NEED_MB=20000 setsid nohup bash supervise.sh gft_cfg 7 bash baseline/run_gft_cfg.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-7}
CK=workdir/probe_gft/model_latest.pt
for spec in "probe_gft_w1p5_cfg1p25 0.666667 1.25" "probe_gft_w1p5_cfg1p5 0.666667 1.5" "probe_gft_w1p25_cfg1p5 0.8 1.5"; do
  set -- $spec
  [ -f runs_out/$1_matched_eval/.done ] && continue
  echo "[$(date +%m%d-%H:%M)] 1b+CFG $1 beta=$2 cfg=$3"
  CKPT=$CK EXTRA="--cfg $3 --gft_beta $2 --chunk 500" bash baseline/eval_matched_r.sh $1 16 $GPU \
    && touch runs_out/$1_matched_eval/.done || echo "GFT_CFG_FAIL $1"
done
echo GFT_CFG_DONE
