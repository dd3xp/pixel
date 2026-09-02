#!/bin/bash
# Round-2 "draw anything" chain: pseudo-labels for 413 objects -> v13.
#
# Round 1 (237 objects) proved the mechanism and its costs: v12 keeps the
# gained vocabulary at a small in-domain colour cost (potions still washed
# out).  This round widens the vocabulary and keeps everything else at the
# settings round 1 already validated: mean_raw downscale, fix_pseudo palette
# and gated-saturation correction, repeat weight 15, fine-tune from v7c_bow
# (same base as v11/v12 so the three stay comparable).
#
# Self-contained and resumable: baseline_downscale skips prompts whose matted
# big image exists; every later stage is cheap to redo.  Run through the
# supervisor so a mid-chain kill restarts it:
#   setsid nohup bash supervise.sh distill2 <gpu> bash baseline/run_distill_v2.sh \
#       </dev/null >/dev/null 2>&1 &
set -eu
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel
cd "$ROOT"
export PYTHONNOUSERSITE=1        # node03: a ~/.local torch shadows the env one
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python

avail=$(df --output=avail -BG /mnt/data | tail -1 | tr -dc 0-9)
[ "$avail" -lt 40 ] && { echo "only ${avail}G free, refusing to start"; exit 1; }

# 1. SDXL renders + downscales every object it hasn't done yet (~413 x 4 sizes)
$PY src/v6/baseline_downscale.py --prompts prompts/vocab_distill_v2.txt \
    --out runs_out/distill_v2 --n 4

# 2. collect the mean_raw variants into a training source
$PY src/v6/prep_pseudo.py runs_out/distill_v2 prompts/vocab_distill_v2.txt data/pseudo2

# 3. palette + gated saturation correction (the v12 fix, unchanged)
$PY src/v6/fix_pseudo.py --src data/pseudo2 --dst data/pseudo2_fix

# 4. v13: fine-tune from the same base as v11/v12, with BOTH rounds of
#    corrected pseudo-labels, weight 15 each (the better v11 setting)
$PY src/v6/train_v7.py --steps 10000 --sample_every 2000 \
    --init workdir/v7c_bow/model_latest.pt \
    --extra data/pseudo_fix,data/pseudo_fix.csv,15 \
    --extra data/pseudo2_fix,data/pseudo2_fix.csv,15 \
    --out workdir/v13

# 5. re-measure coverage with the same 100-object protocol for a clean
#    v7c / v12 / v13 comparison
$PY src/v6/coverage.py --ckpt workdir/v13/model_latest.pt --size 16 \
    --out runs_out/coverage16_v13

echo DISTILL_V2_CHAIN_DONE
