#!/bin/bash
# After run_v8full.sh finishes the v8k pilot (train + the two CFG evals), score the pilot in the
# configuration the paper actually reports: ICG w2 + 4-colour palette projection.  The reference
# numbers are v8n at the same 20k steps: native FD 165.63 (CFG1.5) and 54.23 (ICG2+pal4).
# Usage: bash baseline/v8k_after.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
GPU=${1:-6}
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [ ! -f runs_out/v8k_cfg2_matched_eval/.done ]; do sleep 120; done
t=v8k_20k_icg2_pal4
if [ ! -f runs_out/${t}_matched_eval/.done ]; then
  CKPT=workdir/v8k/model_latest.pt EXTRA="--guide_mode icg --cfg 2 --pal 4 --chunk 500" \
    PROMPTS=runs_out/heldout3000_prompts_recap.txt bash baseline/eval_matched_r.sh $t 16 $GPU \
    && touch runs_out/${t}_matched_eval/.done
fi
CUDA_VISIBLE_DEVICES=$GPU $P src/v6/fd_fair.py --size 16 --native \
  --gen runs_out/${t}_matched_eval/s16 --out runs_out/fair_fd16_native.json
echo V8K_AFTER_DONE
