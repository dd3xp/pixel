#!/bin/bash
# 12 px is the only resolution whose headline number is a single seed; 16, 20 and 24 all carry three.
# Same model, same best k as the sweep found there (k=8), seeds 1 and 2.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
GPU=${1:-6}
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for sd in 1 2; do
  t=v8n_r12_icg2_pal8_seed${sd}
  if [ ! -f runs_out/${t}_matched_eval/.done ]; then
    CKPT=workdir/v8n/model_latest.pt EXTRA="--guide_mode icg --cfg 2 --pal 8 --seed $sd --chunk 500" \
      PROMPTS=runs_out/heldout3000_prompts_recap.txt bash baseline/eval_matched_r.sh $t 12 $GPU \
      && touch runs_out/${t}_matched_eval/.done
  fi
  CUDA_VISIBLE_DEVICES=$GPU $P src/v6/fd_fair.py --size 12 --native \
    --gen runs_out/${t}_matched_eval/s12 --out runs_out/fair_fd12_native.json
done
echo SEEDS12_DONE
