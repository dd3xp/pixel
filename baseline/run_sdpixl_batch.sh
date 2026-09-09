#!/bin/bash
# SD-piXL (Binninger & Sorkine-Hornung, SIGGRAPH Asia 2024) as the DOMAIN-SPECIFIC external baseline.
# Unlike the generic "big model then downscale" route, this method is built for low-resolution quantised imagery,
# so it is the real competitor.  It optimises each image for ~10k score-distillation steps, so it can only be run
# on a small subset; that cost is itself a reportable difference (we sample 1000 sprites in ~2 minutes).
# Usage: bash run_sdpixl_batch.sh <n_prompts> <gpu> [steps]
set -u
ROOT=/mnt/data/kw/RoundSquisheen/pixel
cd $ROOT/SD-piXL
N=${1:-30}; GPU=${2:-3}; STEPS=${3:-10001}
export CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
OUT=$ROOT/pixel/runs_out/sdpixl_eval
mkdir -p $OUT
i=0
while read -r p; do
  [ $i -ge $N ] && break
  d=$OUT/$(printf "%05d" $i)
  if [ -d "$d" ] && [ -n "$(ls $d 2>/dev/null)" ]; then i=$((i+1)); continue; fi
  echo "[$(date +%H:%M:%S)] $i/$N :: $p"
  $PY main.py --config config.yaml --seed 0 -pt "$p" --size 16,16 \
      -respath "$d" >> $ROOT/pixel/logs/sdpixl.log 2>&1 || echo "  FAILED $i"
  i=$((i+1))
done < $ROOT/pixel/runs_out/heldout3000_prompts.txt
echo SDPIXL_BATCH_DONE
