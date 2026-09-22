#!/bin/bash
# v8g: the margin policy. A sprite larger than its bucket is currently resized to fill the frame, which
# is where the model's 34% coverage comes from against 26% for sprites actually drawn at 16 px. With
# --margin 0.8 such a sprite is fitted to 0.8 of the bucket and centred, which moves the 16 px bucket's
# median coverage from 0.369 to 0.277 (real sprites: 0.262) -- checked before training, per the rule
# that a new target distribution is measured first.
# Same corpus and recipe as v8n otherwise, so the comparison is single-variable: v8n@20k scores 54.23
# native FD on the reported configuration. 20k first, because a pilot can reject a change and not
# accept one.
# Usage: bash baseline/run_v8g.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export CUDA_VISIBLE_DEVICES=${1:-6} HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=workdir/v8g DROP=data/corpus_v8_dupdrop.txt STEPS=20000 ARCH_ARG="--margin 0.8" \
  EXTRA_SRC=data/oga3_clean,data/oga3_captions_recap.csv,1 \
  bash baseline/run_v8full.sh
