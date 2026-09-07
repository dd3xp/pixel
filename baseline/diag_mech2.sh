#!/bin/bash
# paper_outline items 7 + 14 (GPU2):
#  (7) mechanism at 20/24 px: pure lower-bucket beliefs (--cfg 0) and their TV vs the strong model's samples at that R
#  (14) operating regime: 32 px (bare model already closer to the floor?) floor + bare / bk24 / bk16 / autog / composed
# Usage: bash baseline/diag_mech2.sh <gpu>
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
run 20 v7h_r20_bk16only --cfg 0 --guide_mode bucket:16
run 20 v7h_r20_bk12only --cfg 0 --guide_mode bucket:12
run 20 v7h_r20_bk24only --cfg 0 --guide_mode bucket:24
run 24 v7h_r24_bk16only --cfg 0 --guide_mode bucket:16
run 24 v7h_r24_bk32only --cfg 0 --guide_mode bucket:32
run 32 v7h_r32_cfg4 --cfg 4
run 32 v7h_r32_bk24_w2 --cfg 2 --guide_mode bucket:24
run 32 v7h_r32_bk16_w2 --cfg 2 --guide_mode bucket:16
run 32 v7h_r32_autog10k_w1p5 --cfg 1.5 --guide_ckpt $WK
run 32 v7h_r32_bk24s10k_w1p5 --cfg 1.5 --guide_mode bucket:24 --guide_ckpt $WK
run 32 v7h_r32_bk48_w2 --cfg 2 --guide_mode bucket:48
echo "[$(date +%m%d-%H:%M)] stats"
CUDA_VISIBLE_DEVICES= $P src/v6/stats_simplicity.py --side 20 --dirs real20=runs_out/heldout3000_totensor_s20 \
  v7h_r20_cfg4=runs_out/v7h_r20_cfg4_matched_eval/s20 bk16only=runs_out/v7h_r20_bk16only_matched_eval/s20 \
  bk12only=runs_out/v7h_r20_bk12only_matched_eval/s20 bk24only=runs_out/v7h_r20_bk24only_matched_eval/s20 \
  bk16_w2=runs_out/v7h_r20_bk16_w2_matched_eval/s20 composed=runs_out/v7h_r20_bk16s10k_w1p5_matched_eval/s20
CUDA_VISIBLE_DEVICES= $P src/v6/stats_simplicity.py --side 24 --dirs real24=runs_out/heldout3000_totensor_s24 \
  v7h_r24_cfg4=runs_out/v7h_r24_cfg4_matched_eval/s24 bk16only=runs_out/v7h_r24_bk16only_matched_eval/s24 \
  bk32only=runs_out/v7h_r24_bk32only_matched_eval/s24 bk16_w2=runs_out/v7h_r24_bk16_w2_matched_eval/s24 \
  composed=runs_out/v7h_r24_bk16s10k_w1p5_matched_eval/s24
echo DIAG_MECH2_DONE
