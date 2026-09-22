#!/bin/bash
# The external table gives our number as a single seed (50.8) while the main table carries a spread.
# On 200 prompts the seed spread should be larger than on 3,000, so quoting one seed there and a mean
# elsewhere invites the question we would rather answer ourselves.
# Usage: bash baseline/ext_seeds.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
GPU=${1:-6}
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PROMPTS=runs_out/ext200/v8n/icg2_pal4/prompts.txt
for sd in 1 2; do
  d=runs_out/ext200/v8n/icg2_pal4_seed${sd}
  if [ ! -f $d/.done ]; then
    CUDA_VISIBLE_DEVICES=$GPU $P src/v6/sample_e.py --ckpt workdir/v8n/model_latest.pt \
      --buckets 12,16,20,24,32,48,64 --sizes 16 --prompts $PROMPTS --n 1 --seed $sd \
      --guide_mode icg --cfg 2 --pal 4 --chunk 200 --out $d && touch $d/.done || continue
  fi
  CUDA_VISIBLE_DEVICES=$GPU $P src/v6/fd_fair.py --size 16 --native \
    --gen $d/s16 --out runs_out/ext200/fd_native_v8n.json
done
echo EXT_SEEDS_DONE
