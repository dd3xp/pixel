#!/bin/bash
# Follow-up on the cross-resolution guide (bucket:12 w=2 -> 8.59, best raw so far): weight sweep,
# stacking with the step10k snapshot (doubly-weak reference), alpha decoupling, uncond mix.
# Usage: bash baseline/diag_v7h5.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-2}; export CUDA_VISIBLE_DEVICES=$GPU
until grep -q DIAG_V7H4_DONE logs/diag_v7h4.log 2>/dev/null; do sleep 120; done
CK=workdir/v7h/model_latest.pt
WK=workdir/v7h/model_step010000.pt
run() {  # <tag> <extra sample_e args>
  local tag=$1; shift
  [ -f runs_out/${tag}_matched_q16/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag :: $*"
  CKPT=$CK EXTRA="$*" bash baseline/eval_matched.sh $tag $GPU && touch runs_out/${tag}_matched_q16/.done
}
run v7h_gbk12_w1p5 --cfg 1.5 --guide_mode bucket:12
run v7h_gbk12_w2p5 --cfg 2.5 --guide_mode bucket:12
run v7h_gbk12_w3 --cfg 3 --guide_mode bucket:12
run v7h_gbk12s10k_w1p5 --cfg 1.5 --guide_mode bucket:12 --guide_ckpt $WK
run v7h_gbk12s10k_w2 --cfg 2 --guide_mode bucket:12 --guide_ckpt $WK
run v7h_gbk12s20k_w1p5 --cfg 1.5 --guide_mode bucket:12 --guide_ckpt workdir/v7h/model_step020000.pt
run v7h_gbk12_w2a1p5 --cfg 2 --cfg_alpha 1.5 --guide_mode bucket:12
run v7h_gbk12mix_w2 --cfg 2 --guide_mode bucketmix:12
run v7h_gbk20_w2 --cfg 2 --guide_mode bucket:20
echo "[$(date +%m%d-%H:%M)] fd_decomp"
$P src/v6/fd_decomp.py --size 16 --gen $(ls -d runs_out/v7h_gbk*_matched_eval/s16)
echo DIAG_V7H5_DONE
