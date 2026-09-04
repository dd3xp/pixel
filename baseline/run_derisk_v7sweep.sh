#!/bin/bash
# De-risk experiment phase 1: v7 feed-forward quality vs resolution sweep.
# Samples v7 densely at 12/16/20/24/32 over 237 diverse objects, for an
# FD-DINOv2 quality-vs-resolution curve (the "does feed-forward degrade
# gracefully" half of the SDS-vs-feedforward thesis). Fully feed-forward, cheap.
set -eu
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel
cd "$ROOT"
export PYTHONNOUSERSITE=1
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
# build "a pixel art X" prompt file from the 237-object vocab
grep -v '^#' prompts/vocab_distill.txt | sed '/^$/d' | sed 's/^/a pixel art /' > runs_out/derisk_prompts.txt
echo "prompts: $(wc -l < runs_out/derisk_prompts.txt)"
$PY src/v6/sample_e.py --ckpt workdir/v7_lowres/model_latest.pt \
    --buckets 12,16,20,24,32,48,64 --sizes 12 16 20 24 32 \
    --prompts runs_out/derisk_prompts.txt --n 8 --seed 0 \
    --out runs_out/derisk_v7sweep
echo DERISK_V7SWEEP_DONE
