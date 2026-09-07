#!/bin/bash
# Paper materials: seed 1 for the key rows of the 12 / 20 / 24 px generalisation table (seed 0 in diag_res20/24),
# so every resolution has 2 seeds.  Usage: bash baseline/diag_seedR.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-2}; export CUDA_VISIBLE_DEVICES=$GPU
CK=workdir/v7h/model_latest.pt
WK=workdir/v7h/model_step010000.pt
run() {  # <R> <tag> <extra sample_e args>
  local R=$1 tag=$2; shift 2
  [ -f runs_out/${tag}_matched_eval/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag @$R :: $*"
  CKPT=$CK EXTRA="$*" bash baseline/eval_matched_r.sh $tag $R $GPU && touch runs_out/${tag}_matched_eval/.done
}
run 20 v7h_r20_cfg4_seed1          --cfg 4 --seed 1
run 20 v7h_r20_bk16_w2_seed1       --cfg 2 --guide_mode bucket:16 --seed 1
run 20 v7h_r20_autog10k_w1p5_seed1 --cfg 1.5 --guide_ckpt $WK --seed 1
run 20 v7h_r20_bk16s10k_w1p5_seed1 --cfg 1.5 --guide_mode bucket:16 --guide_ckpt $WK --seed 1
run 24 v7h_r24_cfg4_seed1          --cfg 4 --seed 1
run 24 v7h_r24_bk16_w2_seed1       --cfg 2 --guide_mode bucket:16 --seed 1
run 24 v7h_r24_autog10k_w1p5_seed1 --cfg 1.5 --guide_ckpt $WK --seed 1
run 24 v7h_r24_bk16s10k_w1p5_seed1 --cfg 1.5 --guide_mode bucket:16 --guide_ckpt $WK --seed 1
run 12 v7h_r12_cfg4_seed1          --cfg 4 --seed 1
run 12 v7h_r12_autog10k_w1p5_seed1 --cfg 1.5 --guide_ckpt $WK --seed 1
echo DIAG_SEEDR_DONE
