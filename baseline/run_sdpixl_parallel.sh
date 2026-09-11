#!/bin/bash
# SD-piXL is the domain-specific external baseline and the only real competitor, but it optimises each image
# for ~10k score-distillation steps (measured: ~4 h per 16 px sprite on a contended A100).  The first batch of 6
# ran strictly sequentially, which wastes a GPU that has room for several at once, so this version runs W of them
# in parallel.  Resumable: an index whose directory already holds final_argmax.png is skipped.
# Usage: bash run_sdpixl_parallel.sh <first_idx> <last_idx> <gpu> <workers>
set -u
ROOT=/mnt/data/kw/RoundSquisheen/pixel
cd $ROOT/SD-piXL
FIRST=${1:-6}; LAST=${2:-17}; GPU=${3:-3}; W=${4:-3}
export CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
OUT=$ROOT/pixel/runs_out/sdpixl_eval
PROMPTS=$ROOT/pixel/runs_out/heldout3000_prompts.txt
mkdir -p $OUT

one() {
  local i=$1
  local d=$OUT/$(printf "%05d" $i)
  if find "$d" -name final_argmax.png 2>/dev/null | grep -q .; then
    echo "[$(date +%H:%M:%S)] $i already complete, skip"; return
  fi
  local p
  p=$(sed -n "$((i + 1))p" $PROMPTS)
  echo "[$(date +%H:%M:%S)] start $i :: $p"
  $PY main.py --config config.yaml --seed 0 -pt "$p" --size 16,16 -respath "$d" \
      >> $ROOT/pixel/logs/sdpixl_$i.log 2>&1 || echo "[$(date +%H:%M:%S)] FAILED $i"
  echo "[$(date +%H:%M:%S)] done $i"
}

W=1  # GPU3 only fits one SD-piXL at a time (each needs ~21 GB and the card is shared); parallel attempts all OOMed
running=0
for i in $(seq $FIRST $LAST); do
  one "$i" &
  running=$((running + 1))
  if [ "$running" -ge "$W" ]; then wait -n 2>/dev/null || wait; running=$((running - 1)); fi
done
wait
echo SDPIXL_PARALLEL_DONE
