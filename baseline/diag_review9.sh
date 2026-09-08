#!/bin/bash
# 32 px is the one resolution where composed-bucketu wins in BOTH coordinates, but that claim currently compares a 3-seed
# bucketu row (66.53 +- 1.09, diag_review8) against seed-0-only composed (69.27) and best-CFG (83.65).  Run seeds 1/2 of
# those two so the 32 px comparison is 3-seed on both sides.
# Launch: NEED_MB=8000 setsid nohup bash supervise.sh diag_review9 3 bash baseline/diag_review9.sh 3 >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-3}; export CUDA_VISIBLE_DEVICES=$GPU
H=workdir/v7h/model_latest.pt; WK=workdir/v7h/model_step010000.pt
runR() {  # <tag> <R> <extra sample_e args>
  local tag=$1 R=$2; shift 2
  [ -f runs_out/${tag}_matched_eval/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag @$R :: $*"
  CKPT=$H EXTRA="$*" bash baseline/eval_matched_r.sh $tag $R $GPU && touch runs_out/${tag}_matched_eval/.done
}
for s in 1 2; do
  runR v7h_r32_bk24s10k_w1p5_seed$s 32 --cfg 1.5 --guide_mode bucket:24 --guide_ckpt $WK --seed $s
  runR v7h_r32_cfg2_seed$s          32 --cfg 2 --seed $s
done
echo DIAG_REVIEW9_DONE
