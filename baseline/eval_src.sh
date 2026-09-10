#!/bin/bash
# probe_src evaluation: the source-bucket label is a SAMPLING KNOB, so it has to be swept, not assumed.
# The model was trained with one class embedding per (target bucket, source bucket) pair; label = t*7 + s.
# At 16 px the target index is 1, so the label for source bucket s is 7 + s_index.
# "native" (s=16) asks for something drawn at 16 px; higher s asks for something drawn larger and downscaled,
# which is what 93% of the evaluation reference actually is -- so the best source label is an empirical question.
# Bar: v7h best-CFG 12.49, current best (zero-training composed guidance) 7.53, floor 3.45.
# Launch: NEED_MB=20000 setsid nohup bash supervise.sh eval_src 3 bash baseline/eval_src.sh 3 >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-3}; export CUDA_VISIBLE_DEVICES=$GPU
CK=workdir/probe_src/model_latest.pt
run() {  # <tag> <extra>
  local tag=$1; shift
  [ -f runs_out/${tag}_matched_eval/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag :: $*"
  $P src/v6/sample_e.py --ckpt $CK --buckets 12,16,20,24,32,48,64 --sizes 16 \
     --prompts runs_out/heldout3000_prompts.txt --n 1 --seed 0 --out runs_out/${tag}_matched_eval "$@" \
    && $P src/v6/fd_fair.py --size 16 --gen runs_out/${tag}_matched_eval/s16 --out runs_out/fair_fd16.json \
    && touch runs_out/${tag}_matched_eval/.done
}
# sweep the source label at a fixed weight; src_bucket picks which (target,source) pair the class label encodes
for s in 12 16 20 24 32 48 64; do run src_s${s}_cfg2 --cfg 2 --src_bucket $s; done
echo EVAL_SRC_SWEEP_DONE
