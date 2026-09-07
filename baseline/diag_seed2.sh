#!/bin/bash
# Paper materials: second extra seed (seed 2) for the key 16 px matched rows, so every headline number has 3 seeds
# (seed 0 / 1 exist).  Also the compute-saving variant (snapshot reference only in the low-noise half, --gi_snap 0 .5)
# at seed 0/1/2.  Usage: bash baseline/diag_seed2.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-2}; export CUDA_VISIBLE_DEVICES=$GPU
CK=workdir/v7h/model_latest.pt
WK=workdir/v7h/model_step010000.pt
STACK="--guide_mode bucket:12 --guide_ckpt $WK"
run() {  # <tag> <extra sample_e args>
  local tag=$1; shift
  [ -f runs_out/${tag}_matched_q16/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag :: $*"
  CKPT=$CK EXTRA="$*" bash baseline/eval_matched.sh $tag $GPU && touch runs_out/${tag}_matched_q16/.done
}
run v7h_gbk12s10k_w1p5_seed2 --cfg 1.5 $STACK --seed 2
run v7h_gbk12_w2_seed2       --cfg 2 --guide_mode bucket:12 --seed 2
run v7h_autog10k_w1p5_seed2  --cfg 1.5 --guide_ckpt $WK --seed 2
run v7h_seed2                --cfg 4 --seed 2
run v7h_stk_w1p5_snaplo_seed1 --cfg 1.5 $STACK --gi_snap 0.0 0.5 --seed 1
run v7h_stk_w1p5_snaplo_seed2 --cfg 1.5 $STACK --gi_snap 0.0 0.5 --seed 2
echo "[$(date +%m%d-%H:%M)] fd_decomp"
$P src/v6/fd_decomp.py --size 16 --gen $(ls -d runs_out/v7h_*seed2_matched_eval/s16 runs_out/v7h_stk_w1p5_snaplo_seed1_matched_eval/s16)
echo DIAG_SEED2_DONE
