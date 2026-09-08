#!/bin/bash
# Best-CFG baseline, part 3: the CFG curves at 12 px (v7h) and for v7s (@16/20/24) were still descending at w=2, so extend to
# w=2.5 / 3 before declaring the best-CFG rows.  Seed 0.
# Launch: NEED_MB=8000 setsid nohup bash supervise.sh diag_review4 3 bash baseline/diag_review4.sh 3 >/dev/null 2>&1 &
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
for w in 2p5 3; do runR $H v7h_r12_cfg${w} 12 --cfg ${w/p/.}; done
for w in 2p5 3; do runR $S v7s_cfg${w} 16 --cfg ${w/p/.}; done
for R in 20 24; do for w in 2p5 3; do runR $S v7s_r${R}_cfg${w} $R --cfg ${w/p/.}; done; done
echo DIAG_REVIEW4_DONE
