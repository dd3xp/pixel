#!/bin/bash
# Seed-variance check for the ~1-point differences among the best guidance settings, plus a few
# stacked (snapshot + bucket:12) variants.  --seed 1 in EXTRA overrides eval_matched's --seed 0.
# Usage: bash baseline/diag_v7h6.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-2}; export CUDA_VISIBLE_DEVICES=$GPU
until grep -q DIAG_V7H5_DONE logs/diag_v7h5.log 2>/dev/null; do sleep 120; done
CK=workdir/v7h/model_latest.pt
WK=workdir/v7h/model_step010000.pt
run() {  # <tag> <extra sample_e args>
  local tag=$1; shift
  [ -f runs_out/${tag}_matched_q16/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag :: $*"
  CKPT=$CK EXTRA="$*" bash baseline/eval_matched.sh $tag $GPU && touch runs_out/${tag}_matched_q16/.done
}
run v7h_autog10k_w1p5_seed1 --cfg 1.5 --guide_ckpt $WK --seed 1
run v7h_gbk12_w2_seed1 --cfg 2 --guide_mode bucket:12 --seed 1
run v7h_gbk12s10k_w1p5_seed1 --cfg 1.5 --guide_mode bucket:12 --guide_ckpt $WK --seed 1
run v7h_seed1 --cfg 4 --seed 1
run v7h_gbk12s10k_w1p25 --cfg 1.25 --guide_mode bucket:12 --guide_ckpt $WK
run v7h_gbk12s5k_w1p5 --cfg 1.5 --guide_mode bucket:12 --guide_ckpt workdir/v7h/model_step005000.pt
echo "[$(date +%m%d-%H:%M)] fd_decomp"
$P src/v6/fd_decomp.py --size 16 --gen $(ls -d runs_out/v7h*seed1_matched_eval/s16 runs_out/v7h_gbk12s*_matched_eval/s16)
echo DIAG_V7H6_DONE
