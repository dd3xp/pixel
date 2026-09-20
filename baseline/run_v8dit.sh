#!/bin/bash
# v8dit: the same 20k pilot as v8m -- same corpus, schedule, sampler and evaluation -- with the UNet
# replaced by a 69.7M transformer (train_v7.py --arch dit, against the UNet 72.5M).  The question is
# not whether the transformer is better; it is whether palette-projected sampling still wins on it,
# i.e. whether the paper method is a property of the recipe or of convolutions.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export CUDA_VISIBLE_DEVICES=${1:-7} HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=workdir/v8dit DROP=data/corpus_v8_dupdrop.txt STEPS=20000 ARCH_ARG=--arch=dit \
  EXTRA_SRC=data/oga3_clean,data/oga3_captions_recap.csv,1 \
  EXTRA_SRC2=data/kenney_cut,data/kenney_captions_final.csv,1 \
  EXTRA_SRC3=data/oga5_clean,data/oga5_captions_final.csv,1 \
  bash baseline/run_v8full.sh
