#!/bin/bash
# v8m finished training but run_v8full.sh died before the eval stage: the script was edited on disk
# while that bash was still reading it, so bash resumed at a shifted byte offset and mangled the next
# command.  Training is complete (model_step020000.pt), so only the eval stage has to be redone.
# Never edit a shell script that is currently executing.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
GPU=${1:-6}
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
touch workdir/v8m/.trained
for w in 1.5 2; do
  t=v8m_cfg$(echo $w | tr . p)
  if [ ! -f runs_out/${t}_matched_eval/.done ]; then
    CKPT=workdir/v8m/model_latest.pt EXTRA="--cfg $w --chunk 500" \
      PROMPTS=runs_out/heldout3000_prompts_recap.txt bash baseline/eval_matched_r.sh $t 16 $GPU \
      && touch runs_out/${t}_matched_eval/.done
  fi
  CUDA_VISIBLE_DEVICES=$GPU $P src/v6/fd_fair.py --size 16 --native \
    --gen runs_out/${t}_matched_eval/s16 --out runs_out/fair_fd16_native.json
done
echo V8M_EVAL_DONE
