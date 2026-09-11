#!/bin/bash
# SD-piXL (Binninger & Sorkine-Hornung, SIGGRAPH Asia 2024) on the RECAPTIONED held-out prompts, 16 px.
# The 8 sprites in runs_out/sdpixl_eval used the old BLIP prompts; every external baseline is being re-scored on
# heldout3000_prompts_recap.txt, so SD-piXL restarts from index 0 on those prompts.
# Runs on node09 (user 09-12: light jobs may use idle node09 GPUs, at most the high-index ones, never competing with
# the midi project, which uses up to 4 GPUs there). Same code and config as node03 (md5 checked).
# Resumable: an index whose directory already holds final_argmax.png is skipped.
# Usage: bash baseline/run_sdpixl_recap.sh <first_idx> <last_idx> <gpu> <workers>
set -u
ROOT=/mnt/data/kw/RoundSquisheen/pixel
cd $ROOT/SD-piXL
FIRST=${1:-0}; LAST=${2:-14}; GPU=${3:-7}; W=${4:-2}
export CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
OUT=$ROOT/pixel/runs_out/sdpixl_recap
PROMPTS=$ROOT/pixel/runs_out/heldout3000_prompts_recap.txt
mkdir -p $OUT $ROOT/pixel/logs

one() {
  local i=$1
  local d=$OUT/$(printf "%05d" $i)
  if find "$d" -name final_argmax.png 2>/dev/null | grep -q .; then
    echo "[$(date +%H:%M:%S)] $i already complete, skip"; return
  fi
  local p
  p=$(sed -n "$((i + 1))p" $PROMPTS)
  echo "[$(date +%H:%M:%S)] start $i on GPU $GPU :: $p"
  $PY main.py --config config.yaml --seed 0 -pt "$p" --size 16,16 -respath "$d" \
      >> $ROOT/pixel/logs/sdpixl_recap_$i.log 2>&1 || echo "[$(date +%H:%M:%S)] FAILED $i"
  echo "[$(date +%H:%M:%S)] done $i"
}

running=0
for i in $(seq $FIRST $LAST); do
  one "$i" &
  running=$((running + 1))
  if [ "$running" -ge "$W" ]; then wait -n 2>/dev/null || wait; running=$((running - 1)); fi
done
wait
echo SDPIXL_RECAP_DONE $FIRST-$LAST
