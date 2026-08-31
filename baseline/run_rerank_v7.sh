#!/bin/bash
# CLIP best-of-N reranking for v7 across the paper ladder 12/16/20/24
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com
B=12,16,20,24,32,48,64
CK=workdir/v7_lowres/model_latest.pt
for S in 12 16 20 24; do
  echo "[$(date +%m%d-%H:%M)] rerank size $S"
  $P src/v6/clip_rerank.py --ckpt $CK --buckets $B --prompts prompts/mc_items.txt      --size $S --n 16 --sweep 1 2 4 8 16 --save_all --out runs/rerank_v7_s$S || echo RERANK_FAIL_$S
done
echo RERANK_V7_DONE
