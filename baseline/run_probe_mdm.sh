#!/bin/bash
# probe_mdm (A1): masked discrete diffusion over per-channel 8-bit pixel tokens.
# The first genuinely NEW architecture of this cycle. The earlier discrete kill (v6f, 160.7) is confounded: it
# used a 32-colour corpus palette against a per-sprite median of 34 colours, so its vocabulary could not express
# the data. Per-channel 8-bit is lossless (corpus verified 8-bit RGBA), so the quantisation floor is 3.45.
# Bar: v7h best-CFG 12.49, current best (zero-training composed guidance) 7.53, floor 3.45.
# Decision (from arch_scout): bare matched FD <= 12 -> mechanism is real, add guidance and compare against 7.53;
#                             12-20 -> ambiguous, sweep the smoothing width and decoding steps; >= 20 -> family dead.
# Launch: NEED_MB=20000 setsid nohup bash supervise.sh probe_mdm 2 bash baseline/run_probe_mdm.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
STEPS=${STEPS:-40000}
OUT=${OUT:-workdir/probe_mdm}
echo "[$(date +%m%d-%H:%M)] STAGE train probe_mdm steps=$STEPS"
$P src/v6/train_mdm.py --steps $STEPS --out $OUT --exclude runs_out/holdout_exclude.txt \
   --bs_scale 0.15 --snap_every 20000 || { echo PROBE_MDM_TRAIN_FAIL; exit 1; }
echo PROBE_MDM_DONE
