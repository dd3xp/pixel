#!/bin/bash
# Resolution generalisation, 20 px and 12 px (matched protocol at R, fd_fair --size R), plus a second-model check
# of bucket:12 guidance on the original (memorisation-tainted) v7_lowres at 16 px.  At 12 px no lower bucket
# exists, so only snapshot autoguidance applies (honest limitation of the zero-training form).
# Usage: bash baseline/diag_res20.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-3}; export CUDA_VISIBLE_DEVICES=$GPU
CK=workdir/v7h/model_latest.pt
WK=workdir/v7h/model_step010000.pt
run() {  # <R> <tag> <extra sample_e args>
  local R=$1 tag=$2; shift 2
  [ -f runs_out/${tag}_matched_eval/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag :: $*"
  CKPT=$CK EXTRA="$*" bash baseline/eval_matched_r.sh $tag $R $GPU && touch runs_out/${tag}_matched_eval/.done
}
run 20 v7h_r20_cfg4 --cfg 4
run 20 v7h_r20_bk16_w2 --cfg 2 --guide_mode bucket:16
run 20 v7h_r20_bk12_w2 --cfg 2 --guide_mode bucket:12
run 20 v7h_r20_autog10k_w1p5 --cfg 1.5 --guide_ckpt $WK
run 20 v7h_r20_bk16s10k_w1p5 --cfg 1.5 --guide_mode bucket:16 --guide_ckpt $WK
run 12 v7h_r12_cfg4 --cfg 4
run 12 v7h_r12_autog10k_w1p5 --cfg 1.5 --guide_ckpt $WK
run 12 v7h_r12_bk16_w2 --cfg 2 --guide_mode bucket:16
echo "[$(date +%m%d-%H:%M)] fd_decomp 20/12"
$P src/v6/fd_decomp.py --size 20 --gen $(ls -d runs_out/v7h_r20_*_matched_eval/s20)
$P src/v6/fd_decomp.py --size 12 --gen $(ls -d runs_out/v7h_r12_*_matched_eval/s12)
# second model (16 px, standard matched): original v7_lowres, bucket:12 w2 vs its bare cfg4 (16.66, tainted)
CK=workdir/v7_lowres/model_latest.pt
[ -f runs_out/v7_bk12_w2_matched_q16/.done ] || { CKPT=$CK EXTRA="--cfg 2 --guide_mode bucket:12" bash baseline/eval_matched.sh v7_bk12_w2 $GPU && touch runs_out/v7_bk12_w2_matched_q16/.done; }
echo DIAG_RES20_DONE
