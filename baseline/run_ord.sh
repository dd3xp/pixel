#!/bin/bash
# Probe: train ordinal palette-manifold discrete diffusion (v_ord), same config
# as v6f_discrete (absorbing) for a fair schedule-only comparison. Evaluate at
# 16px vs v6f (naive discrete) vs v7 (continuous).
set -eu
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel
cd "$ROOT"
export PYTHONNOUSERSITE=1
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
$PY src/v6/train_ordinal.py --palette assets/palettes/dawnbringer32.hex \
    --steps 40000 --sample_every 4000 --out workdir/v_ord --ema 0.999
echo V_ORD_DONE
