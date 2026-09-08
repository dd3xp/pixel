#!/bin/bash
# Text-alignment cost of reference guidance (diag_clipq16: CLIP 100cos @16 bare CFG4 30.06 > real 29.80 > composed 29.5 >
# bucket:12 29.37; R@1/100 18.3 / 16.4 / 14.4 / 13.1 %).  Zero-training fixes that put the CFG direction back:
#  (a) bucketu:<R'> reference = wrong bucket AND unconditional text (2 NFE/step, same cost)
#  (b) --cfg_text wt: additive plain-CFG term on top of the reference guidance (3 NFE/step)
# Success = CLIP back to >= bare (30.06) with FD not worse than the un-fixed row by more than seed sd (~0.3).
# Usage: bash baseline/diag_align.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-2}; export CUDA_VISIBLE_DEVICES=$GPU
CK=workdir/v7h/model_latest.pt
WK=workdir/v7h/model_step010000.pt
run() {  # <tag> <extra sample_e args>
  local tag=$1; shift
  [ -f runs_out/${tag}_matched_q16/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag :: $*"
  CKPT=$CK EXTRA="$*" bash baseline/eval_matched.sh $tag $GPU && touch runs_out/${tag}_matched_q16/.done
}
run v7h_bku12_w2            --cfg 2 --guide_mode bucketu:12
run v7h_bku12_w1p5          --cfg 1.5 --guide_mode bucketu:12
run v7h_bku12s10k_w1p5      --cfg 1.5 --guide_mode bucketu:12 --guide_ckpt $WK
run v7h_gbk12_w2_ct1p5      --cfg 2 --guide_mode bucket:12 --cfg_text 1.5
run v7h_gbk12s10k_w1p5_ct1p5 --cfg 1.5 --guide_mode bucket:12 --guide_ckpt $WK --cfg_text 1.5
run v7h_gbk12s10k_w1p5_ct2  --cfg 1.5 --guide_mode bucket:12 --guide_ckpt $WK --cfg_text 2
run v7h_bku12s10k_w1p25     --cfg 1.25 --guide_mode bucketu:12 --guide_ckpt $WK
echo "[$(date +%m%d-%H:%M)] CLIP + fd_decomp"
M=runs_out; E=_matched_eval
$P src/v6/clip_score.py --size 16 --out runs_out/clip_scores.json --dirs bku12_w2=$M/v7h_bku12_w2$E/s16 bku12_w1p5=$M/v7h_bku12_w1p5$E/s16 \
  bku12s10k_w1p5=$M/v7h_bku12s10k_w1p5$E/s16 bk12_w2_ct1p5=$M/v7h_gbk12_w2_ct1p5$E/s16 composed_ct1p5=$M/v7h_gbk12s10k_w1p5_ct1p5$E/s16 \
  composed_ct2=$M/v7h_gbk12s10k_w1p5_ct2$E/s16 bku12s10k_w1p25=$M/v7h_bku12s10k_w1p25$E/s16
$P src/v6/fd_decomp.py --size 16 --gen $(ls -d runs_out/v7h_bku12*_matched_eval/s16 runs_out/v7h_*_ct*_matched_eval/s16)
echo DIAG_ALIGN_DONE
