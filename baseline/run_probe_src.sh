#!/bin/bash
# probe_src (cycle 9, architecture loop): the training corpus is an UNLABELLED MIXTURE over the downscale factor.
# Verified 2026-09-09: of the 3000 real sprites the FD@16 reference is built from, only 7.1% were drawn at <=16 px;
# 93% are 17-64 px artwork put through Image.BOX.  train_v7.py feeds the model the OUTPUT bucket only, and augments
# every sprite into every lower bucket, so at bucket 16 the model must average over "natively 16" and "downscaled
# from 24/32/48/64" without being told which.  That average is textbook mean regression and its signature is what we
# measure (73 colours vs 34 real, flat .030 vs .202).
# probe_src gives the model the missing variable: one class embedding per (target bucket, source bucket) pair, 49
# entries instead of 7.  Everything else -- data, exclusions, width, steps, lr, EMA, schedule -- is v7h exactly, so
# the comparison is paired.  At sampling time the source label becomes a knob: sweep it (eval_src.sh).
# Bar: v7h best-CFG 12.49, current best (zero-training composed guidance) 7.53, floor 3.45.
# Launch: NEED_MB=20000 setsid nohup bash supervise.sh probe_src 2 bash baseline/run_probe_src.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
STEPS=${STEPS:-80000}
OUT=${OUT:-workdir/probe_src}
EX=runs_out/holdout_exclude.txt
mkdir -p runs_out logs
echo "[$(date +%m%d-%H:%M)] STAGE train probe_src steps=$STEPS (paired with v7h: same data, exclusions, width, lr, EMA)"
$P src/v6/train_srcbucket.py --steps $STEPS --out $OUT --exclude $EX --snap_every 20000 || { echo PROBE_SRC_TRAIN_FAIL; exit 1; }
echo "[$(date +%m%d-%H:%M)] STAGE done training -> eval is baseline/eval_src.sh (source-label sweep)"
echo PROBE_SRC_DONE
