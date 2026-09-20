#!/bin/bash
# v8p: the full-length model on the 09-20 corpus (v8n data + Kenney CC0 + the CC-BY-3.0 bundle).
# The 20k pilot earns this: on the configuration the paper reports, v8m reaches 42.87 native FD where
# v8n reaches 54.23 and v8k -- the same data minus the CC-BY-3.0 bundle -- reaches 53.19, so the gain
# is the new bundle and it is four times the seed spread.  80k steps, same recipe as v8n otherwise.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export CUDA_VISIBLE_DEVICES=${1:-6} HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=workdir/v8p DROP=data/corpus_v8_dupdrop.txt STEPS=80000 \
  EXTRA_SRC=data/oga3_clean,data/oga3_captions_recap.csv,1 \
  EXTRA_SRC2=data/kenney_cut,data/kenney_captions_final.csv,1 \
  EXTRA_SRC3=data/oga5_clean,data/oga5_captions_final.csv,1 \
  bash baseline/run_v8full.sh
