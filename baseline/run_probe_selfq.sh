#!/bin/bash
# Projection-in-the-loop self-conditioning probe (src/v6/train_selfq.py), fine-tuned from probe_tv.
# env: INIT STEPS SAMPLE_EVERY OUT K TV EXTRA
set -eu
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel; cd "$ROOT"; export PYTHONNOUSERSITE=1
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
$PY src/v6/train_selfq.py --init ${INIT:-workdir/probe_tv/model_latest.pt} --steps ${STEPS:-20000} \
    --sample_every ${SAMPLE_EVERY:-4000} --out workdir/${OUT:-probe_selfq} \
    --K ${K:-16} --tv ${TV:-0.1} ${EXTRA:-}
echo PROBE_SELFQ_DONE
