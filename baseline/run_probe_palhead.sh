#!/bin/bash
# Palette-factorised x0 head probe (src/v6/train_palhead.py), fine-tuned from v7.
# env: STEPS SAMPLE_EVERY OUT K PAL_MODE EXTRA (extra CLI args)
set -eu
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel; cd "$ROOT"; export PYTHONNOUSERSITE=1
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
$PY src/v6/train_palhead.py --init workdir/v7_lowres/model_latest.pt --steps ${STEPS:-20000} \
    --sample_every ${SAMPLE_EVERY:-4000} --out workdir/${OUT:-probe_palhead} \
    --K ${K:-16} --pal_mode ${PAL_MODE:-mlp} ${EXTRA:-}
echo PROBE_PALHEAD_DONE
