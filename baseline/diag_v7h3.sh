#!/bin/bash
# Zero-training "bad model" probes for guidance on v7h (pixel-art specific alternatives to the
# early-snapshot autoguidance reference): wrong resolution embedding (bucket:<px>), box-blurred
# input (blur:k), 1-px shifted input (shift).  Same matched protocol as diag_v7h.sh.
# Usage: bash baseline/diag_v7h3.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-2}; export CUDA_VISIBLE_DEVICES=$GPU
until grep -q DIAG_V7H2_DONE logs/diag_v7h2.log 2>/dev/null; do sleep 120; done
CK=workdir/v7h/model_latest.pt
run() {  # <tag> <extra sample_e args>
  local tag=$1; shift
  [ -f runs_out/${tag}_matched_q16/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag :: $*"
  CKPT=$CK EXTRA="$*" bash baseline/eval_matched.sh $tag $GPU && touch runs_out/${tag}_matched_q16/.done
}
run v7h_gbk64_w2 --cfg 2 --guide_mode bucket:64
run v7h_gbk24_w2 --cfg 2 --guide_mode bucket:24
run v7h_gbk12_w2 --cfg 2 --guide_mode bucket:12
run v7h_gblur2_w2 --cfg 2 --guide_mode blur:2
run v7h_gshift_w2 --cfg 2 --guide_mode shift
run v7h_gbk64_w1p5 --cfg 1.5 --guide_mode bucket:64
run v7h_gblur2_w1p5 --cfg 1.5 --guide_mode blur:2
echo "[$(date +%m%d-%H:%M)] fd_decomp"
$P src/v6/fd_decomp.py --size 16 --gen $(ls -d runs_out/v7h_g*_matched_eval/s16)
echo DIAG_V7H3_DONE
