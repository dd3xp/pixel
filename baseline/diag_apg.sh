#!/bin/bash
# Second component: Adaptive Projected Guidance (Sadat et al. 2024) on top of the cross-resolution stack
# (weak = step10k snapshot AND bucket:12 label) at 16 px, matched protocol.  Baseline stack w1.5 = 7.67 (seed1 7.32).
# APG removes the component of the guidance difference parallel to x0 (colour over-saturation), so it should
# allow larger w without the mean/colour blow-up.  Runs after dres24.  Usage: bash baseline/diag_apg.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-2}; export CUDA_VISIBLE_DEVICES=$GPU
[ -n "$NOWAIT" ] || until grep -q DIAG_RES24_DONE logs/diag_res24.log 2>/dev/null; do sleep 120; done
CK=workdir/v7h/model_latest.pt
WK=workdir/v7h/model_step010000.pt
STACK="--guide_mode bucket:12 --guide_ckpt $WK"
run() {  # <tag> <extra sample_e args>
  local tag=$1; shift
  [ -f runs_out/${tag}_matched_q16/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag :: $*"
  CKPT=$CK EXTRA="$*" bash baseline/eval_matched.sh $tag $GPU && touch runs_out/${tag}_matched_q16/.done
}
run v7h_stk_w2_apgrgb      --cfg 2   $STACK --apg 0,0,-0.5,rgb
run v7h_stk_w1p5_apgrgb    --cfg 1.5 $STACK --apg 0,0,-0.5,rgb
run v7h_stk_w3_apgrgb      --cfg 3   $STACK --apg 0,0,-0.5,rgb
run v7h_stk_w2_apgfull     --cfg 2   $STACK --apg 0,0,-0.5
run v7h_stk_w2_apgproj     --cfg 2   $STACK --apg 0,0,0,rgb
run v7h_stk_w2_apgmom      --cfg 2   $STACK --apg 1,0,-0.5,rgb
run v7h_bk12_w2_apgrgb     --cfg 2   --guide_mode bucket:12 --apg 0,0,-0.5,rgb
run v7h_autog10k_w2_apgrgb --cfg 2   --guide_ckpt $WK --apg 0,0,-0.5,rgb
echo "[$(date +%m%d-%H:%M)] fd_decomp"
$P src/v6/fd_decomp.py --size 16 --gen $(ls -d runs_out/v7h_*apg*_matched_eval/s16)
echo DIAG_APG_DONE
