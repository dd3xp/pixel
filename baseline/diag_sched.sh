#!/bin/bash
# Zero-training second-component candidates on the stack (weak = step10k snapshot AND bucket:12), 16 px matched:
#  (a) dual-reference scheduling: snapshot reference only inside a t/T interval (--gi_snap), label reference throughout
#  (b) frequency-decoupled guidance (FDG): --cfg for the 2x2 residual (high), --fdg for the 2x2-mean (low) part
# Baseline stack w1.5 = 7.67 (seed1 7.32), w2 = 10.83.  Usage: bash baseline/diag_sched.sh <gpu>
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
run v7h_stk_w1p5_fdg1     --cfg 1.5 $STACK --fdg 1.0
run v7h_stk_w2_fdg1p25    --cfg 2   $STACK --fdg 1.25
run v7h_stk_w1p5_snapmid  --cfg 1.5 $STACK --gi_snap 0.2 0.8
run v7h_stk_w1p5_snaplo   --cfg 1.5 $STACK --gi_snap 0.0 0.5
run v7h_stk_w1p5_snaphi   --cfg 1.5 $STACK --gi_snap 0.5 1.0
run v7h_stk_w2_fdg1       --cfg 2   $STACK --fdg 1.0
run v7h_stk_w1p5_fdg1p25  --cfg 1.5 $STACK --fdg 1.25
run v7h_stk_w2_snapmid    --cfg 2   $STACK --gi_snap 0.2 0.8
run v7h_stk_w2p5_fdg1     --cfg 2.5 $STACK --fdg 1.0
echo "[$(date +%m%d-%H:%M)] fd_decomp"
$P src/v6/fd_decomp.py --size 16 --gen $(ls -d runs_out/v7h_stk_*fdg*_matched_eval/s16 runs_out/v7h_stk_*snap*_matched_eval/s16)
echo DIAG_SCHED_DONE
