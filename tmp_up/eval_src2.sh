#!/bin/bash
# The source sweep was at one weight only; sweep CFG on the best source (48) before deciding, so the kill is fair.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-3}; export CUDA_VISIBLE_DEVICES=$GPU
CK=workdir/probe_src/model_latest.pt
run() { local tag=$1; shift
  [ -f runs_out/${tag}_matched_eval/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag :: $*"
  $P src/v6/sample_e.py --ckpt $CK --buckets 12,16,20,24,32,48,64 --sizes 16 \
     --prompts runs_out/heldout3000_prompts.txt --n 1 --seed 0 --out runs_out/${tag}_matched_eval "$@" \
    && $P src/v6/fd_fair.py --size 16 --gen runs_out/${tag}_matched_eval/s16 --out runs_out/fair_fd16.json \
    && touch runs_out/${tag}_matched_eval/.done
}
for w in 1.25 1.5 2.5; do run src_s48_cfg$(echo $w | tr . p) --cfg $w --src_bucket 48; done
# and the composed cross-resolution guidance on top of the best source, to see if they stack
run src_s48_composed --cfg 1.5 --src_bucket 48 --guide_mode bucket:12 --guide_ckpt workdir/probe_src/model_step020000.pt
echo EVAL_SRC2_DONE
