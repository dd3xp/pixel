#!/bin/bash
set -eu
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel; cd "$ROOT"; export PYTHONNOUSERSITE=1
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
$PY src/v6/train_probe.py --init workdir/v7_lowres/model_latest.pt --probe tv --probe_w 0.1 \
    --steps 20000 --sample_every 4000 --out workdir/probe_tv --ema 0.999
echo PROBE_TV_DONE
