#!/bin/bash
# paper_outline gap #6: guidance-weight sweep at 20 px (v7h, seed 0) for the label reference (bucket:16) and autoguidance
# (snapshot 10k).  Existing points: bk16 w2 = 32.89, autog w1.5 = 31.78, bare = 45.92.  16 px reference curve: bk12 w1.5/2/3.
# Launch: NEED_MB=8000 setsid nohup bash supervise.sh diag_wsweep20 2 bash baseline/diag_wsweep20.sh 2 >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-2}; export CUDA_VISIBLE_DEVICES=$GPU
H=workdir/v7h/model_latest.pt;  HW=workdir/v7h/model_step010000.pt
R=${R:-20}
runR() {  # <tag> <extra sample_e args>
  local tag=$1; shift
  [ -f runs_out/${tag}_matched_eval/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag @$R :: $*"
  CKPT=$H EXTRA="$*" bash baseline/eval_matched_r.sh $tag $R $GPU && touch runs_out/${tag}_matched_eval/.done
}
LB=16; [ "$R" = 24 ] && LB=16
for w in 1p25 1p5 2p5 3; do runR v7h_r${R}_bk${LB}_w${w}    --cfg ${w/p/.} --guide_mode bucket:$LB; done
for w in 1p25 2 2p5 3;   do runR v7h_r${R}_autog10k_w${w} --cfg ${w/p/.} --guide_ckpt $HW; done
echo DIAG_WSWEEP${R}_DONE
