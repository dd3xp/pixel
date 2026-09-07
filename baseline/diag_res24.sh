#!/bin/bash
# Resolution generalisation of cross-resolution autoguidance, 24 px (matched protocol at R=24, fd_fair --size 24).
# Reference = the model's own lower-resolution bucket embeddings (16 / 20 / 12) vs the same-run step10k snapshot,
# plus the stack.  Usage: bash baseline/diag_res24.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-2}; export CUDA_VISIBLE_DEVICES=$GPU
CK=workdir/v7h/model_latest.pt
WK=workdir/v7h/model_step010000.pt
R=24
run() {  # <tag> <extra sample_e args>
  local tag=$1; shift
  [ -f runs_out/${tag}_matched_eval/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag :: $*"
  CKPT=$CK EXTRA="$*" bash baseline/eval_matched_r.sh $tag $R $GPU && touch runs_out/${tag}_matched_eval/.done
}
run v7h_r24_cfg4 --cfg 4
run v7h_r24_bk16_w2 --cfg 2 --guide_mode bucket:16
run v7h_r24_autog10k_w1p5 --cfg 1.5 --guide_ckpt $WK
run v7h_r24_bk20_w2 --cfg 2 --guide_mode bucket:20
run v7h_r24_bk12_w2 --cfg 2 --guide_mode bucket:12
run v7h_r24_bk16s10k_w1p5 --cfg 1.5 --guide_mode bucket:16 --guide_ckpt $WK
run v7h_r24_bk32_w2 --cfg 2 --guide_mode bucket:32
echo "[$(date +%m%d-%H:%M)] fd_decomp"
$P src/v6/fd_decomp.py --size $R --gen $(ls -d runs_out/v7h_r24_*_matched_eval/s$R)
echo DIAG_RES24_DONE
