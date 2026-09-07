#!/bin/bash
# probe_cc follow-up (trained contrast-deficient reference: coarse w1.5 = 14.71 @16 vs bare 20.63, same-weights bk12 9.43):
#  (a) mechanism stats of the pure trained reference (--cfg 0 --guide_mode coarse) @16: TV vs the bucket:12 belief (21.6)
#  (b) 12 px, where no lower bucket exists: does the trained reference fill the gap?  probe_cc weights throughout.
# Usage: bash baseline/diag_cc12.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-3}; export CUDA_VISIBLE_DEVICES=$GPU
CK=workdir/probe_cc/model_latest.pt
WK=workdir/v7h/model_step010000.pt
run() {  # <R> <tag> <extra sample_e args>
  local R=$1 tag=$2; shift 2
  [ -f runs_out/${tag}_matched_eval/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag @$R :: $*"
  CKPT=$CK EXTRA="$*" bash baseline/eval_matched_r.sh $tag $R $GPU && touch runs_out/${tag}_matched_eval/.done
}
run 16 probe_cc_coarseonly --cfg 0 --guide_mode coarse
run 12 probe_cc_r12_cfg4 --cfg 4
run 12 probe_cc_r12_coarse_w1p5 --cfg 1.5 --guide_mode coarse
run 12 probe_cc_r12_autog10k_w1p5 --cfg 1.5 --guide_ckpt $WK
run 12 probe_cc_r12_coarse_s10k_w1p5 --cfg 1.5 --guide_mode coarse --guide_ckpt $WK
run 12 probe_cc_r12_coarse_w1p25 --cfg 1.25 --guide_mode coarse
echo "[$(date +%m%d-%H:%M)] stats"
CUDA_VISIBLE_DEVICES= $P src/v6/stats_simplicity.py --side 16 --dirs \
  real16=runs_out/ref_totensor_s16 v7h_cfg4=runs_out/v7h_matched_eval/s16 probe_cc_cfg4=runs_out/probe_cc_cfg4_matched_eval/s16 \
  bk12only=runs_out/v7h_bk12only_matched_eval/s16 cc_coarseonly=runs_out/probe_cc_coarseonly_matched_eval/s16 \
  cc_coarse_w1p5=runs_out/probe_cc_coarse_w1p5_matched_eval/s16 cc_bk12_w2=runs_out/probe_cc_bk12_w2_matched_eval/s16
echo DIAG_CC12_DONE
