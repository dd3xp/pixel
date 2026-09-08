#!/bin/bash
# Follow-up to diag_review1 R1: the 16 px CFG curve has its optimum far below the recipe default (w=1 16.39, w=1.5 12.24,
# w=4 21.98), so the honest "bare" baseline is best-tuned CFG, not w=4.  Re-tune the bare baseline everywhere it is used:
#  v7h @12/20/24/32 with CFG w=1.5 / 2 (seed 0); v7h @16 w=1.5 seeds 1/2 (3-seed best-CFG row); v7s @16/20/24 w=1.5 / 2.
# Launch: NEED_MB=8000 setsid nohup bash supervise.sh diag_review2 3 bash baseline/diag_review2.sh 3 >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-3}; export CUDA_VISIBLE_DEVICES=$GPU
H=workdir/v7h/model_latest.pt; S=workdir/v7s/model_latest.pt
runR() {  # <ckpt> <tag> <R> <extra sample_e args>
  local ck=$1 tag=$2 R=$3; shift 3
  [ -f runs_out/${tag}_matched_eval/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag @$R :: $*"
  CKPT=$ck EXTRA="$*" bash baseline/eval_matched_r.sh $tag $R $GPU && touch runs_out/${tag}_matched_eval/.done
}
for R in 20 24 12 32; do for w in 1p5 2; do runR $H v7h_r${R}_cfg${w} $R --cfg ${w/p/.}; done; done
for s in 1 2; do runR $H v7h_cfg1p5_seed${s} 16 --cfg 1.5 --seed $s; done
for w in 1p5 2; do runR $S v7s_cfg${w} 16 --cfg ${w/p/.}; done
for R in 20 24; do for w in 1p5 2; do runR $S v7s_r${R}_cfg${w} $R --cfg ${w/p/.}; done; done
echo DIAG_REVIEW2_DONE
