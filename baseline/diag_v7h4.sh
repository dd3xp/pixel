#!/bin/bash
# Channel-decoupled autoguidance on v7h: RGB (palette) weight = --cfg, alpha (silhouette) weight = --cfg_alpha,
# weak model = same-run step10k EMA snapshot (best of diag_v7h).  Matched protocol.
# Usage: bash baseline/diag_v7h4.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-2}; export CUDA_VISIBLE_DEVICES=$GPU
until grep -q DIAG_V7H3_DONE logs/diag_v7h3.log 2>/dev/null; do sleep 120; done
CK=workdir/v7h/model_latest.pt
WK=workdir/v7h/model_step010000.pt
run() {  # <tag> <extra sample_e args>
  local tag=$1; shift
  [ -f runs_out/${tag}_matched_q16/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag :: $*"
  CKPT=$CK EXTRA="$*" bash baseline/eval_matched.sh $tag $GPU && touch runs_out/${tag}_matched_q16/.done
}
run v7h_autog10k_r2a1 --cfg 2 --cfg_alpha 1 --guide_ckpt $WK
run v7h_autog10k_r2a3 --cfg 2 --cfg_alpha 3 --guide_ckpt $WK
run v7h_autog10k_r3a2 --cfg 3 --cfg_alpha 2 --guide_ckpt $WK
run v7h_autog10k_r1p5a2p5 --cfg 1.5 --cfg_alpha 2.5 --guide_ckpt $WK
run v7h_autog10k_r2p5a1p5 --cfg 2.5 --cfg_alpha 1.5 --guide_ckpt $WK
echo "[$(date +%m%d-%H:%M)] fd_decomp"
$P src/v6/fd_decomp.py --size 16 --gen $(ls -d runs_out/v7h_autog10k_r*_matched_eval/s16)
echo DIAG_V7H4_DONE
