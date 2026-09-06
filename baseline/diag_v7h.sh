#!/bin/bash
# Zero-training guidance sweep on the CLEAN baseline v7h, all under the matched protocol
# (baseline/eval_matched.sh: captions of 3000 training-excluded held-out sprites, n=1, seed 0).
# Waits for the v7h training/eval chain to finish (logs/v7h.done), then runs each setting.
# Usage: bash baseline/diag_v7h.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-2}; export CUDA_VISIBLE_DEVICES=$GPU
until [ -f logs/v7h.done ]; do sleep 120; done
CK=workdir/v7h/model_latest.pt
run() {  # <tag> <extra sample_e args>
  local tag=$1; shift
  [ -f runs_out/${tag}_matched_q16/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag :: $*"
  CKPT=$CK EXTRA="$*" bash baseline/eval_matched.sh $tag $GPU && touch runs_out/${tag}_matched_q16/.done
}
run v7h --cfg 4
run v7h_cfg7 --cfg 7
run v7h_cfg10 --cfg 10
run v7h_cads4 --cfg 4 --cads
run v7h_cads7 --cfg 7 --cads
run v7h_cads7s25 --cfg 7 --cads --cads_s 0.25
run v7h_gi7 --cfg 7 --gi 0.0 0.8
run v7h_autog10k_w2 --cfg 2 --guide_ckpt workdir/v7h/model_step010000.pt
run v7h_autog10k_w3 --cfg 3 --guide_ckpt workdir/v7h/model_step010000.pt
run v7h_autog20k_w2 --cfg 2 --guide_ckpt workdir/v7h/model_step020000.pt
run v7h_autog10k_cfg --cfg 7 --guide_ckpt workdir/v7h/model_step010000.pt --gi 0.0 0.8
echo "[$(date +%m%d-%H:%M)] fd_decomp"
$P src/v6/fd_decomp.py --size 16 --gen $(ls -d runs_out/v7h*_matched_eval/s16)
echo DIAG_V7H_DONE
