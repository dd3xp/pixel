#!/bin/bash
# Sample v_ord (ordinal) and v6f (absorbing) at 16px on the 413-vocab prompts,
# for the 3-way FD-DINOv2 comparison (v7 16px reused from derisk_v7sweep/s16).
set -eu
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel
cd "$ROOT"; export PYTHONNOUSERSITE=1
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
$PY src/v6/sample_discrete.py --ckpt workdir/v_ord/model_latest.pt --schedule ordinal \
    --prompts prompts/vocab_distill_v2.txt --n 8 --size 16 --out runs_out/cmp_ord
$PY src/v6/sample_discrete.py --ckpt workdir/v6f_discrete/model_latest.pt --schedule absorbing \
    --prompts prompts/vocab_distill_v2.txt --n 8 --size 16 --out runs_out/cmp_v6f
echo CMP3_SAMPLING_DONE
