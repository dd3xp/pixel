#!/bin/bash
# Paper gap-fillers on GPU2 (paper_outline.md items 3, 12, 5), all matched protocol, 16 px unless noted:
#  (3) second model v7_lowres: bucket:12 seed 1 (no early snapshot exists for this run, so no autoguidance row)
#  (12) sampler robustness of the composed reference and bucket:12: DDIM 50, DDPM 50, DDPM 200 (default DDPM 100)
#  (5) 20 px reverse controls: bucket:24 / bucket:32 as reference (higher bucket should hurt, as at 12/16/24 px)
# Usage: bash baseline/diag_misc.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-2}; export CUDA_VISIBLE_DEVICES=$GPU
CK=workdir/v7h/model_latest.pt
WK=workdir/v7h/model_step010000.pt
STACK="--guide_mode bucket:12 --guide_ckpt $WK"
run16() {  # <ckpt> <tag> <extra sample_e args>
  local ck=$1 tag=$2; shift 2
  [ -f runs_out/${tag}_matched_q16/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag :: $*"
  CKPT=$ck EXTRA="$*" bash baseline/eval_matched.sh $tag $GPU && touch runs_out/${tag}_matched_q16/.done
}
runR() {  # <R> <tag> <extra sample_e args>
  local R=$1 tag=$2; shift 2
  [ -f runs_out/${tag}_matched_eval/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag @$R :: $*"
  CKPT=$CK EXTRA="$*" bash baseline/eval_matched_r.sh $tag $R $GPU && touch runs_out/${tag}_matched_eval/.done
}
run16 workdir/v7_lowres/model_latest.pt v7lr_gbk12_w2_seed1 --cfg 2 --guide_mode bucket:12 --seed 1
run16 workdir/v7_lowres/model_latest.pt v7lr_seed1           --cfg 4 --seed 1
run16 $CK v7h_stk_w1p5_ddim50   --cfg 1.5 $STACK --sampler ddim --steps 50
run16 $CK v7h_stk_w1p5_ddpm50   --cfg 1.5 $STACK --steps 50
run16 $CK v7h_stk_w1p5_ddpm200  --cfg 1.5 $STACK --steps 200
run16 $CK v7h_ddim50            --cfg 4 --sampler ddim --steps 50
run16 $CK v7h_gbk12_w2_ddim50   --cfg 2 --guide_mode bucket:12 --sampler ddim --steps 50
runR 20 v7h_r20_bk24_w2 --cfg 2 --guide_mode bucket:24
runR 20 v7h_r20_bk32_w2 --cfg 2 --guide_mode bucket:32
echo "[$(date +%m%d-%H:%M)] fd_decomp"
$P src/v6/fd_decomp.py --size 16 --gen $(ls -d runs_out/v7lr_*_matched_eval/s16 runs_out/v7h_*dd*_matched_eval/s16)
echo DIAG_MISC_DONE
