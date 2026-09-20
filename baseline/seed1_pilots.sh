#!/bin/bash
# A second seed for the two 20k pilots that the cross-backbone claim rests on.  They land 0.36 apart
# under the projection, which is inside the seed spread measured at full length; this checks that the
# spread at 20k is comparable rather than assuming it.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
GPU=${1:-7}
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in v8m v8dit; do
  t=${m}_20k_icg2_pal4_seed1
  if [ ! -f runs_out/${t}_matched_eval/.done ]; then
    CKPT=workdir/${m}/model_latest.pt EXTRA="--guide_mode icg --cfg 2 --pal 4 --chunk 500 --seed 1" \
      PROMPTS=runs_out/heldout3000_prompts_recap.txt bash baseline/eval_matched_r.sh $t 16 $GPU \
      && touch runs_out/${t}_matched_eval/.done
  fi
  CUDA_VISIBLE_DEVICES=$GPU $P src/v6/fd_fair.py --size 16 --native \
    --gen runs_out/${t}_matched_eval/s16 --out runs_out/fair_fd16_native.json
done
echo SEED1_DONE
