#!/bin/bash
# v8m: the 09-20 corpus pilot -- v8n data + the Kenney CC0 packs + the CC-BY-3.0 bundle (11,988 rows
# after captioning and content filtering).  20k steps, judged against v8n@20k (native FD 165.63 at
# CFG 1.5, 54.23 on ICG w2 + 4-colour projection) and v8k@20k (155.93 / 53.19).
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export CUDA_VISIBLE_DEVICES=${1:-6} HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=workdir/v8m DROP=data/corpus_v8_dupdrop.txt STEPS=20000 \
  EXTRA_SRC=data/oga3_clean,data/oga3_captions_recap.csv,1 \
  EXTRA_SRC2=data/kenney_cut,data/kenney_captions_final.csv,1 \
  EXTRA_SRC3=data/oga5_clean,data/oga5_captions_final.csv,1 \
  bash baseline/run_v8full.sh
